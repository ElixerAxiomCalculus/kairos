"""SQuAD adapter that keeps contexts retrievable and QA annotations separate."""

from __future__ import annotations

import hashlib

from kairos.corpus.adapters.base import CorpusAdapter


class SquadAdapter(CorpusAdapter):
    namespace = "SQUAD"

    def validate(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        valid, rejected = [], []
        for idx, record in enumerate(records):
            if not str(record.get("context") or "").strip():
                rejected.append({"index": idx, "reason": "missing context"})
            elif not str(record.get("question") or "").strip():
                rejected.append({"index": idx, "reason": "missing question"})
            else:
                valid.append(record)
        return valid, rejected

    def transform(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        contexts: dict[str, dict] = {}
        labels = []
        for record in records:
            context = str(record["context"]).strip()
            context_id = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
            doc_id = f"SQUAD:{context_id}"
            contexts.setdefault(doc_id, {
                "doc_id": doc_id,
                "title": str(record.get("title") or f"SQuAD context {context_id}").strip(),
                "doc_type": "encyclopedic_context",
                "source_record_id": context_id,
                "effective_date": None,
                "body": context,
            })
            labels.append({
                "id": str(record.get("id") or ""),
                "document_id": doc_id,
                "question": str(record["question"]),
                "answers": record.get("answers", {}),
                "split": self.spec["split"],
            })
        return list(contexts.values()), labels
