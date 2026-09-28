"""Dense encoders. Default is LSA (TF-IDF + truncated SVD): no download, fully deterministic.
`sentence-transformers` is an opt-in upgrade (configs/neural.yaml)."""

from __future__ import annotations

import numpy as np


class LSAEncoder:
    name = "lsa"

    def __init__(self, texts: list[str], dims: int = 128) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english")
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = max(2, min(dims, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=0, algorithm="arpack")
        self.svd.fit(tfidf)
        self.model_id = f"lsa-tfidf-svd{n_comp}"

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.svd.transform(self.vectorizer.transform(texts))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class SentenceTransformerEncoder:
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model_id = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def build_encoder(cfg: dict, texts: list[str]):
    if cfg["dense_backend"] == "sentence-transformers":
        return SentenceTransformerEncoder(cfg["dense_model"])
    return LSAEncoder(texts, dims=cfg["lsa_dims"])
