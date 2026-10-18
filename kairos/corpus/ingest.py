"""Corpus ingestion: heading-aware chunking, corpus-derived facets and gazetteer.

Everything here is derived from the corpus itself — no external vocabularies:

* Facet dimensions come from *contrast siblings*: two sibling headings that differ by exactly
  one term ("Standard events" / "International events") define a contrast set
  {standard, international}. Chunks under those headings carry that facet value.
* The gazetteer collects mid-sentence proper nouns; ones that follow "in" or appear in
  document titles are classed as places (used for self-repair: "Pune — sorry, Mumbai").
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

import yaml

from kairos.textutil import STOPWORDS, content_set, content_terms, sentences, tokens

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")
MAX_CHUNK_WORDS = 110


@dataclass
class Section:
    doc_id: str
    number: str
    title: str
    level: int
    parent: "Section | None"
    body: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.doc_id}#{self.number}"

    def path(self) -> list["Section"]:
        node, out = self, []
        while node is not None and node.level > 1:
            out.append(node)
            node = node.parent
        return list(reversed(out))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_key: str
    section_number: str
    section_title: str
    section_path: str
    parent_key: str | None
    text: str
    sha1: str
    sha256: str
    facets: dict[str, str]
    effective_date: str | None
    doc_type: str | None
    ordinal: int
    source_dataset: str | None = None
    source_record_id: str | None = None
    source_url: str | None = None
    license: str | None = None
    dataset_revision: str | None = None
    content_checksum: str | None = None
    corpus_profile: str | None = None

    @property
    def index_text(self) -> str:
        return f"{self.doc_title}. {self.section_path}. {self.text}"

    def to_public(self, include_text: bool = True) -> dict:
        data = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section_key": self.section_key,
            "section": self.section_number,
            "section_title": self.section_title,
            "section_path": self.section_path,
            "facets": self.facets,
            "effective_date": self.effective_date,
            "doc_type": self.doc_type,
            "sha1": self.sha1,
            "content_hash": f"sha256:{self.sha256}",
            "content_hash_scheme": "sha256-v1",
            "source_dataset": self.source_dataset,
            "source_record_id": self.source_record_id,
            "source_url": self.source_url,
            "license": self.license,
            "dataset_revision": self.dataset_revision,
            "content_checksum": self.content_checksum,
            "corpus_profile": self.corpus_profile,
        }
        if include_text:
            data["text"] = self.text
        return data


@dataclass
class Corpus:
    chunks: list[Chunk]
    documents: dict[str, dict]
    contrast_groups: dict[str, set[str]]       # group id -> member terms
    qualifier_group: dict[str, str]            # term -> group id
    places: set[str]
    proper_nouns: set[str]
    sections: dict[str, Section]
    contrast_heads: set[str]

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def __post_init__(self) -> None:
        self._by_id = {c.chunk_id: c for c in self.chunks}

    def siblings_with_group(self, section_key: str, group: str) -> list[str]:
        """Section keys of siblings (same parent) that carry a facet from `group`."""
        section = self.sections.get(section_key)
        if section is None or section.parent is None:
            return []
        out = []
        for sib in section.parent.children:
            if group in _heading_facets(sib.title, self.qualifier_group, self.contrast_heads):
                out.append(sib.key)
        return out


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        return yaml.safe_load(fm) or {}, body
    return {}, raw


def _heading_facets(title: str, qualifier_group: dict[str, str], heads: set[str]) -> dict[str, str]:
    """Facet of a heading: its leading modifier, when the heading's head noun is a contrast head."""
    terms = content_terms(title)
    if len(terms) < 2 or terms[-1] not in heads:
        return {}
    group = qualifier_group.get(terms[0])
    return {group: terms[0]} if group else {}


def _parse_sections(doc_id: str, body: str) -> list[Section]:
    root = Section(doc_id, "0", "", 1, None)
    stack = [root]
    auto_counter: dict[int, int] = {}
    all_sections: list[Section] = []
    for line in body.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            level = len(m.group(1))
            if level == 1:
                continue  # document title
            title = m.group(2).strip()
            nm = NUMBERED_RE.match(title)
            if nm:
                number, title = nm.group(1), nm.group(2).strip()
            else:
                auto_counter[level] = auto_counter.get(level, 0) + 1
                number = f"s{level}.{auto_counter[level]}"
            while stack and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1] if stack else root
            section = Section(doc_id, number, title, level, parent)
            parent.children.append(section)
            stack.append(section)
            all_sections.append(section)
        elif line.strip():
            stack[-1].body.append(line.strip())
    return all_sections


def _contrast_groups(all_sections: list[Section]) -> tuple[dict[str, set[str]], set[str]]:
    parent_map: dict[int, list[Section]] = {}
    for s in all_sections:
        if s.parent is not None:
            parent_map.setdefault(id(s.parent), []).append(s)
    uf: dict[str, str] = {}
    heads: set[str] = set()

    def find(x: str) -> str:
        while uf.setdefault(x, x) != x:
            x = uf[x]
        return x

    def union(a: str, b: str) -> None:
        uf[find(a)] = find(b)

    for siblings in parent_map.values():
        term_lists = [content_terms(s.title) for s in siblings]
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                a, b = term_lists[i], term_lists[j]
                # Contrast siblings share the head noun and differ only in the leading modifier:
                # "Standard events" / "International events" -> {standard, international}.
                if len(a) == len(b) >= 2 and a[-1] == b[-1] and a[1:] == b[1:] and a[0] != b[0]:
                    union(a[0], b[0])
                    heads.add(a[-1])
    groups: dict[str, set[str]] = {}
    for term in list(uf):
        groups.setdefault(find(term), set()).add(term)
    named = {"|".join(sorted(members)): members for members in groups.values() if len(members) > 1}
    return named, heads


def _chunk_body(lines: list[str]) -> list[str]:
    text = " ".join(lines)
    if len(text.split()) <= MAX_CHUNK_WORDS:
        return [text] if text else []
    parts, current = [], []
    for sent in sentences(text):
        if current and len(" ".join(current + [sent]).split()) > MAX_CHUNK_WORDS:
            parts.append(" ".join(current))
            current = []
        current.append(sent)
    if current:
        parts.append(" ".join(current))
    return parts


def _gazetteer(docs: list[tuple[dict, str]]) -> tuple[set[str], set[str]]:
    proper: set[str] = set()
    places: set[str] = set()
    for meta, body in docs:
        for line in body.splitlines():
            if line.startswith("#"):
                continue
            for sent in sentences(line):
                toks = tokens(sent)
                for i, tok in enumerate(toks):
                    low = tok.lower()
                    if i == 0 or low in STOPWORDS or not tok[:1].isupper() or len(tok) < 3:
                        continue
                    proper.add(low)
                    j = i - 1
                    while j > 0 and toks[j][:1].isupper():
                        j -= 1
                    if toks[j].lower() == "in":
                        places.add(low)
    return proper, places


def load_corpus(path: Path | Sequence[Path], profile: str | None = None) -> Corpus:
    """Load one or more prepared Markdown roots without changing legacy citation IDs.

    Imported adapters provide already-namespaced ``doc_id`` values (for example
    ``SCIFACT:123``).  Duplicate IDs are rejected rather than silently overwriting provenance.
    """
    roots = [Path(p) for p in path] if isinstance(path, Sequence) and not isinstance(path, (str, Path)) else [Path(path)]
    files = sorted({f.resolve() for root in roots for f in root.glob("*.md")})
    if not files:
        raise FileNotFoundError(f"No markdown documents found in {roots}")
    parsed: list[tuple[dict, str]] = []
    for f in files:
        meta, body = _parse_front_matter(f.read_text(encoding="utf-8"))
        meta.setdefault("doc_id", f.stem.split("-")[0])
        meta.setdefault("title", f.stem)
        meta["source_file"] = f.name
        meta.setdefault("corpus_profile", profile)
        parsed.append((meta, body))

    doc_sections: dict[str, list[Section]] = {}
    for meta, body in parsed:
        if meta["doc_id"] in doc_sections:
            raise ValueError(f"Duplicate corpus doc_id: {meta['doc_id']}")
        doc_sections[meta["doc_id"]] = _parse_sections(meta["doc_id"], body)
    everything = [s for secs in doc_sections.values() for s in secs]
    groups, heads = _contrast_groups(everything)
    qualifier_group = {term: gid for gid, members in groups.items() for term in members}
    proper, places = _gazetteer(parsed)

    chunks: list[Chunk] = []
    documents: dict[str, dict] = {}
    sections_by_key: dict[str, Section] = {}
    for meta, _body in parsed:
        doc_id = meta["doc_id"]
        ordinal = 0
        for section in doc_sections[doc_id]:
            sections_by_key[section.key] = section
            path_nodes = section.path()
            facets: dict[str, str] = {}
            for node in path_nodes:
                facets.update(_heading_facets(node.title, qualifier_group, heads))
            for part in _chunk_body(section.body):
                ordinal += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}:C{ordinal:02d}",
                        doc_id=doc_id,
                        doc_title=meta["title"],
                        section_key=section.key,
                        section_number=section.number,
                        section_title=section.title,
                        section_path=" > ".join(n.title for n in path_nodes),
                        parent_key=section.parent.key if section.parent and section.parent.level > 1 else None,
                        text=part,
                        sha1=hashlib.sha1(part.encode("utf-8")).hexdigest(),
                        sha256=hashlib.sha256(part.encode("utf-8")).hexdigest(),
                        facets=facets,
                        effective_date=str(meta.get("effective_date")) if meta.get("effective_date") else None,
                        doc_type=meta.get("doc_type"),
                        ordinal=ordinal,
                        source_dataset=meta.get("source_dataset"),
                        source_record_id=str(meta.get("source_record_id")) if meta.get("source_record_id") is not None else None,
                        source_url=meta.get("source_url"),
                        license=meta.get("license"),
                        dataset_revision=meta.get("dataset_revision"),
                        content_checksum=meta.get("content_checksum"),
                        corpus_profile=meta.get("corpus_profile"),
                    )
                )
        documents[doc_id] = {
            "doc_id": doc_id,
            "title": meta["title"],
            "doc_type": meta.get("doc_type"),
            "effective_date": str(meta.get("effective_date")) if meta.get("effective_date") else None,
            "source_file": meta["source_file"],
            "chunks": ordinal,
            "source_dataset": meta.get("source_dataset"),
            "source_record_id": meta.get("source_record_id"),
            "source_url": meta.get("source_url"),
            "license": meta.get("license"),
            "dataset_revision": meta.get("dataset_revision"),
            "content_checksum": meta.get("content_checksum"),
            "corpus_profile": meta.get("corpus_profile"),
        }
    return Corpus(chunks, documents, groups, qualifier_group, places, proper, sections_by_key, heads)


def section_terms(chunk: Chunk) -> set[str]:
    return content_set(chunk.section_path)
