"""Adapter for BEIR corpus records (currently SciFact)."""

from __future__ import annotations

from kairos.corpus.adapters.base import CorpusAdapter


class BeirCorpusAdapter(CorpusAdapter):
    namespace = "SCIFACT"

    def validate(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        valid, rejected = [], []
        seen = set()
        for idx, record in enumerate(records):
            record_id = str(record.get("_id") or record.get("id") or "").strip()
            text = str(record.get("text") or "").strip()
            if not record_id or not text:
                rejected.append({"index": idx, "reason": "missing id or text"})
            elif record_id in seen:
                rejected.append({"index": idx, "reason": "duplicate id", "id": record_id})
            else:
                seen.add(record_id)
                valid.append(record)
        return valid, rejected

    def transform(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        docs = []
        for record in records:
            record_id = str(record.get("_id") or record["id"])
            docs.append({
                "doc_id": f"SCIFACT:{record_id}",
                "title": str(record.get("title") or f"SciFact article {record_id}").strip(),
                "doc_type": "scientific_abstract",
                "source_record_id": record_id,
                "effective_date": None,
                "body": str(record["text"]).strip(),
            })
        return docs, []
