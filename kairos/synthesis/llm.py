"""Optional generative backends (Gemini / Ollama) behind the same interface as the extractive one.

Closed-handle citation: the model sees evidence labelled E1..En — never document IDs — and must
cite handles from that list. Handles are mapped back to chunk IDs here; anything that does not
resolve becomes 'UNRESOLVED:<handle>', which the firewall rejects as an invalid citation.

Reproducibility: every call goes through a record/replay cache keyed by a hash of the full request.
`synthesis.llm_mode: replay` runs the benchmark from cache only (no network, bit-identical output).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kairos.config import resolve_path
from kairos.retrieval.index import Hit, HybridIndex
from kairos.schemas.models import Claim, SubQuery
from kairos.synthesis.base import ClaimDraft, SynthesisResult, context_tokens, evidence_covers_focus, score_sentences

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                },
                "required": ["text", "evidence", "quote"],
                "additionalProperties": False,
            },
        },
        "gap": {"type": "string"},
    },
    "required": ["claims", "gap"],
    "additionalProperties": False,
}

TRANSFORM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source_claim_id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["source_claim_id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

SYNTH_SYSTEM = (
    "You write short factual claims that answer one sub-question, using ONLY the evidence provided. "
    "Each claim is one sentence, states one fact, and cites the evidence handles (E1, E2, ...) it relies on. "
    "'quote' must be copied verbatim from one cited evidence item. Do not use outside knowledge. "
    "If the evidence does not answer the sub-question, return no claims and describe what is missing in 'gap'; "
    "otherwise set 'gap' to an empty string. Write at most {n} claims."
)

TRANSFORM_SYSTEM = (
    "You rewrite existing answer claims for presentation only. Never add facts, numbers, names or conditions "
    "that are not in the source claim. Keep every number exactly. Return one output claim per source claim, "
    "carrying its source_claim_id. Instruction: {instruction}"
)


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.provider = cfg["backend"]
        self.model = cfg["gemini_model"] if self.provider == "gemini" else cfg["ollama_model"]
        self.resolved_model = self.model
        self.cache_dir: Path = resolve_path(cfg["llm_cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mode = cfg.get("llm_mode", "record")
        self._client = None

    def _key(self, system: str, user: str, schema: dict) -> str:
        blob = json.dumps({"p": self.provider, "m": self.model, "s": system, "u": user, "schema": schema},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def complete_json(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        key = self._key(system, user, schema)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return cached["data"], {**cached["usage"], "cache_hit": 1}
        if self.mode == "replay":
            raise LLMUnavailable(f"replay mode: no cached response for request {key}")
        data, usage = self._call(system, user, schema)
        path.write_text(json.dumps({"data": data, "usage": usage}, ensure_ascii=False), encoding="utf-8")
        return data, {**usage, "cache_hit": 0}

    def _call(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        if self.provider == "gemini":
            return self._call_gemini(system, user, schema)
        if self.provider == "ollama":
            import ollama

            try:
                res = ollama.chat(model=self.model, format=schema, options={"temperature": 0, "seed": 0},
                                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            except Exception as exc:  # ollama raises plain connection errors when the daemon is down
                raise LLMUnavailable(f"ollama error: {exc}") from exc
            usage = {"llm_prompt": int(res.get("prompt_eval_count") or 0),
                     "llm_completion": int(res.get("eval_count") or 0)}
            return json.loads(res["message"]["content"]), usage
        raise LLMUnavailable(f"unknown provider {self.provider}")

    def _call_gemini(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        import os
        import time

        from google import genai
        from google.genai import errors, types

        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise LLMUnavailable("GEMINI_API_KEY is not set (put it in kairos/.env)")
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=int(self.cfg.get("gemini_timeout_s", 30) * 1000)))
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_level=self.cfg.get("gemini_thinking_level", "low")),
        )
        delays = [1.5, 4.0]
        for attempt in range(len(delays) + 1):
            try:
                res = self._client.models.generate_content(model=self.model, contents=user, config=config)
                break
            except errors.APIError as exc:
                retryable = exc.code in (429, 500, 503)
                if not retryable or attempt == len(delays):
                    raise LLMUnavailable(f"gemini error {exc.code}: {str(exc)[:200]}") from exc
                time.sleep(delays[attempt])
            except Exception as exc:  # timeouts / connection errors surface as httpx exceptions
                raise LLMUnavailable(f"gemini request failed: {exc}") from exc
        if not res.text:
            raise LLMUnavailable("gemini returned no text (blocked or empty response)")
        meta = res.usage_metadata
        usage = {
            "llm_prompt": int(meta.prompt_token_count or 0) if meta else 0,
            "llm_completion": int((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)) if meta else 0,
        }
        self.resolved_model = res.model_version or self.model
        return json.loads(res.text), usage


class LLMSynthesizer:
    """Abstractive claims from closed evidence handles; falls back to extractive on LLM failure."""

    def __init__(self, index: HybridIndex, cfg: dict, fallback) -> None:
        self.index = index
        self.cfg = cfg
        self.client = LLMClient(cfg)
        self.fallback = fallback
        self.name = f"{cfg['backend']}:{self.client.model}"

    def build_prompt(self, sq: SubQuery, hits: list[Hit], avoid_facets=None, prefer_terms=None,
                     n: int | None = None) -> tuple[str, str, dict[str, str]]:
        n = n or self.cfg["sentences_per_subquery"]
        ranked = score_sentences(self.index, sq.query, hits, avoid_facets, prefer_terms)[:8]
        handles: dict[str, str] = {}
        lines = []
        for i, s in enumerate(ranked, 1):
            h = f"E{i}"
            handles[h] = s.chunk_id
            lines.append(f"[{h}] ({s.section_title}) {s.text}")
        user = f"Sub-question: {sq.text}\n\nEvidence:\n" + "\n".join(lines)
        return SYNTH_SYSTEM.format(n=n), user, handles

    def synthesize(self, sq: SubQuery, hits: list[Hit], n: int | None = None, avoid_facets=None,
                   prefer_terms=None, exclude_quotes=None) -> SynthesisResult:
        ctx = context_tokens(self.index, hits)
        if not hits or hits[0].rerank_score < self.cfg["tau_sufficiency"]:
            return SynthesisResult([], self.fallback.gap_text(sq), {"llm_prompt": 0, "llm_completion": 0}, ctx,
                                   self.name)
        covered, focus = evidence_covers_focus(self.index, sq.need_texts, hits)
        if not covered:
            return SynthesisResult([], self.fallback.gap_text(sq, focus), {"llm_prompt": 0, "llm_completion": 0},
                                   ctx, self.name)
        system, user, handles = self.build_prompt(sq, hits, avoid_facets, prefer_terms, n)
        try:
            data, usage = self.client.complete_json(system, user, CLAIMS_SCHEMA)
        except LLMUnavailable:
            result = self.fallback.synthesize(sq, hits, n, avoid_facets, prefer_terms, exclude_quotes)
            result.backend = f"{self.name}->extractive-fallback"
            return result
        drafts = []
        for item in data.get("claims", []):
            evidence = list(dict.fromkeys(handles.get(h, f"UNRESOLVED:{h}") for h in item.get("evidence", [])))
            drafts.append(ClaimDraft(item["text"].strip(), item.get("quote", "").strip(), evidence, 1.0, sq.id))
        gap = data.get("gap") or None
        if not drafts and not gap:
            gap = self.fallback.gap_text(sq)
        return SynthesisResult(drafts, gap if not drafts else None, usage, ctx, self.name)

    def transform(self, claims: list[Claim], instruction: str) -> tuple[list[dict], dict]:
        user = "\n".join(f"{c.claim_id}: {c.text}" for c in claims)
        data, usage = self.client.complete_json(TRANSFORM_SYSTEM.format(instruction=instruction), user,
                                                TRANSFORM_SCHEMA)
        return data.get("claims", []), usage

    def prompt_tokens_estimate(self, sq: SubQuery, hits: list[Hit]) -> int:
        from kairos.textutil import estimate_tokens

        system, user, _ = self.build_prompt(sq, hits)
        return estimate_tokens(system + user)
