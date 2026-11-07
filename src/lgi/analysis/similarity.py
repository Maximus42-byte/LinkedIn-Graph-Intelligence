"""Person similarity and a deterministic nearest-colleagues graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _skill_set(profile: dict[str, Any]) -> set[str]:
    return {str(skill).strip().casefold() for skill in profile.get("skills", []) if str(skill).strip()}


def _document(profile: dict[str, Any]) -> str:
    """Use declared capabilities and public role text as the matching evidence."""
    basic = profile.get("basic", {})
    chunks = [str(skill) for skill in profile.get("skills", [])]
    chunks.extend(str(job.get("title", "")) for job in profile.get("experience", []))
    chunks.extend((str(basic.get("headline", "")), str(basic.get("summary", ""))))
    return " ".join(chunk.strip() for chunk in chunks if chunk.strip())


def analyze_similarity(
    profiles: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return matrices and a union of positive-similarity top-k selections."""
    ordered = sorted(profiles, key=lambda profile: str(profile["id"]))
    ids = [str(profile["id"]) for profile in ordered]
    n = len(ordered)
    requested_k = max(1, int(config.get("knn_k", 3)))
    k = min(requested_k, max(0, n - 1))
    documents = [_document(profile) for profile in ordered]
    cosine = np.zeros((n, n), dtype=float)
    if any(documents):
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            matrix = vectorizer.fit_transform(documents)
            cosine = cosine_similarity(matrix)
        except ValueError:
            # All documents can consist entirely of stop words in manual data.
            pass
    skill_sets = [_skill_set(profile) for profile in ordered]
    jaccard = np.zeros((n, n), dtype=float)
    for left in range(n):
        for right in range(left, n):
            union = skill_sets[left] | skill_sets[right]
            score = len(skill_sets[left] & skill_sets[right]) / len(union) if union else 0.0
            jaccard[left, right] = score
            jaccard[right, left] = score
    selected_by: dict[tuple[str, str], set[str]] = defaultdict(set)
    neighbours = []
    for left, person_id in enumerate(ids):
        ranked = [right for right in range(n) if right != left and cosine[left, right] > 0]
        ranked.sort(key=lambda right: (-cosine[left, right], ids[right]))
        selected = ranked[:k]
        neighbours.append({
            "id": person_id,
            "neighbour_ids": [ids[right] for right in selected],
            "similarities": [round(float(cosine[left, right]), 8) for right in selected],
        })
        for right in selected:
            pair = tuple(sorted((person_id, ids[right])))
            selected_by[pair].add(person_id)
    edges = []
    id_index = {person_id: index for index, person_id in enumerate(ids)}
    for (source, target), selectors in sorted(selected_by.items()):
        edges.append({
            "source": source,
            "target": target,
            "weight": round(float(cosine[id_index[source], id_index[target]]), 8),
            "selected_by": sorted(selectors),
        })
    return {
        "ids": ids,
        "cosine_method": "TF-IDF cosine over skills, job titles, headline, and summary",
        "jaccard_method": "Jaccard similarity over casefolded declared skill sets",
        "cosine_matrix": np.round(cosine, 8).tolist(),
        "jaccard_matrix": np.round(jaccard, 8).tolist(),
        "knn_k": k,
        "knn_graph": {"directed": False, "edges": edges},
        "neighbours": neighbours,
        "warnings": ["Profiles with no matching text have no nearest-colleague edge."] if any(not document for document in documents) else [],
    }
