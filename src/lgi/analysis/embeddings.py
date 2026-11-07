"""Offline semantic vectors with a local-model option and LSA fallback."""

from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_similarity(texts: list[str], config: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Never download a model: a local path is required for the optional backend."""
    count = len(texts)
    if count == 0:
        empty = np.empty((0, 0))
        return empty, empty, "none"
    safe_texts = [text if text.strip() else "empty" for text in texts]
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(safe_texts)
    tfidf_cosine = np.clip(cosine_similarity(tfidf), 0.0, 1.0)
    local_model = config.get("embedding_model")
    if local_model and Path(local_model).exists():
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(local_model, local_files_only=True)
            vectors = model.encode(safe_texts, show_progress_bar=False)
            return np.clip(cosine_similarity(vectors), 0.0, 1.0), tfidf_cosine, "sentence-transformers (local)"
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            pass
    dimension = min(8, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if dimension >= 1:
        vectors = TruncatedSVD(n_components=dimension, random_state=config["seed"]).fit_transform(tfidf)
        return np.clip(cosine_similarity(vectors), 0.0, 1.0), tfidf_cosine, "TF-IDF + TruncatedSVD (LSA)"
    return tfidf_cosine.copy(), tfidf_cosine, "TF-IDF (small sample fallback)"
