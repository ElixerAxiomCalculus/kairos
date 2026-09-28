"""Okapi BM25 over normalised content terms, with optional per-query-term weights."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(docs):
            for t in set(d):
                self.postings.setdefault(t, []).append(i)

    def scores(self, query: dict[str, float]) -> np.ndarray:
        """query: term -> weight."""
        out = np.zeros(self.n, dtype=np.float64)
        for term, weight in query.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                f = self.tf[i][term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += weight * idf * f * (self.k1 + 1) / denom
        return out

    def idf_of(self, term: str) -> float:
        return self.idf.get(term, math.log(1 + (self.n + 0.5) / 0.5))
