"""Compare graph communities with profile features on the same people.

Each result is JSON compatible. Cluster numbers are canonicalised by the first
profile ID in each cluster, because community algorithms do not promise label order.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import warnings

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from lgi.analysis.util import stable_numbers
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


GRAPH_METHODS = ("louvain", "greedy_modularity", "label_propagation", "girvan_newman", "spectral")
FEATURE_METHODS = ("tfidf_kmeans", "ward", "embedding_kmeans", "hdbscan", "nmf_topics")


def _canonical_labels(ids: list[str], communities: list[set[str]]) -> dict[str, int]:
    """Give equivalent partitions byte-identical labels across runs."""
    groups = [set(group) for group in communities if group]
    covered = set().union(*groups) if groups else set()
    groups.extend({pid} for pid in ids if pid not in covered)
    groups.sort(key=lambda group: min(group))
    return {pid: index for index, group in enumerate(groups) for pid in sorted(group)}


def _canonical_array(ids: list[str], values: np.ndarray, preserve_noise: bool = False) -> dict[str, int]:
    """Map arbitrary estimator labels to a stable zero-based numbering."""
    groups: dict[int, set[str]] = {}
    for pid, value in zip(ids, values):
        groups.setdefault(int(value), set()).add(pid)
    noise = groups.pop(-1, set()) if preserve_noise else set()
    result = _canonical_labels(ids, list(groups.values())) if not noise else {
        pid: label for pid, label in _canonical_labels([pid for pid in ids if pid not in noise], list(groups.values())).items()
    }
    result.update({pid: -1 for pid in noise})
    return result


def _partition(labels: dict[str, int]) -> list[set[str]]:
    groups: dict[int, set[str]] = {}
    for pid, label in labels.items():
        groups.setdefault(label, set()).add(pid)
    return list(groups.values())


def _modularity(graph: nx.Graph, labels: dict[str, int]) -> float | None:
    if graph.number_of_edges() == 0:
        return None
    return float(nx.community.modularity(graph, _partition(labels), weight="weight"))


def _graph_communities(graph: nx.Graph, ids: list[str], seed: int, config: dict) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Run bounded community searches so a larger fixture remains usable."""
    if graph.number_of_edges() == 0:
        singles = _canonical_labels(ids, [{pid} for pid in ids])
        return {name: singles.copy() for name in GRAPH_METHODS[:4]}, ["Community methods found no graph edges"]

    results: dict[str, dict[str, int]] = {}
    results["louvain"] = _canonical_labels(ids, list(nx.community.louvain_communities(graph, weight="weight", seed=seed)))
    results["greedy_modularity"] = _canonical_labels(ids, list(nx.community.greedy_modularity_communities(graph, weight="weight")))
    results["label_propagation"] = _canonical_labels(ids, list(nx.community.asyn_lpa_communities(graph, weight="weight", seed=seed)))

    messages: list[str] = []
    max_nodes = int(config.get("girvan_newman_max_nodes", 50))
    if graph.number_of_nodes() > max_nodes:
        results["girvan_newman"] = results["louvain"].copy()
        messages.append("Girvan-Newman skipped above configured node limit; Louvain labels are shown as fallback")
    else:
        best: dict[str, int] | None = None
        best_score = float("-inf")
        max_cuts = min(graph.number_of_nodes() - 1, int(config.get("girvan_newman_max_cuts", 20)))
        generator = nx.community.girvan_newman(graph)
        for _ in range(max_cuts):
            try:
                groups = next(generator)
            except StopIteration:
                break
            labels = _canonical_labels(ids, [set(group) for group in groups])
            score = _modularity(graph, labels)
            if score is not None and score > best_score:
                best = labels
                best_score = score
        results["girvan_newman"] = best if best is not None else _canonical_labels(ids, [{pid} for pid in ids])
    return results, messages


def _feature_texts(profiles: list[dict]) -> tuple[list[str], list[str]]:
    combined = []
    skills = []
    for profile in profiles:
        skill_text = " ".join(profile.get("skills", []))
        title_text = " ".join(job.get("title", "") for job in profile.get("experience", []))
        skills.append(skill_text)
        combined.append(f"{skill_text} {title_text}".strip())
    return combined, skills


def _tfidf(texts: list[str]) -> tuple[np.ndarray | None, list[str]]:
    """No text produces no partition evidence; do not invent ID features."""
    if not any(text.strip() for text in texts):
        return None, []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return None, []
    return matrix.toarray(), list(vectorizer.get_feature_names_out())


def _score_features(matrix: np.ndarray | None, labels: list[int]) -> tuple[float | None, float | None]:
    if matrix is None or len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return None, None
    try:
        return float(silhouette_score(matrix, labels)), float(davies_bouldin_score(matrix, labels))
    except ValueError:
        return None, None


def _elbow_choice(candidates: list[dict]) -> int:
    """Use the maximum bend in the inertia curve, with a small-k tie break."""
    if len(candidates) < 3:
        return candidates[0]["k"]
    xs = np.array([row["k"] for row in candidates], dtype=float)
    ys = np.array([row["inertia"] for row in candidates], dtype=float)
    x_span, y_span = np.ptp(xs), np.ptp(ys)
    if y_span == 0:
        return candidates[0]["k"]
    xs = (xs - xs[0]) / x_span
    ys = (ys - ys[-1]) / y_span
    distances = np.abs((ys[-1] - ys[0]) * xs - (xs[-1] - xs[0]) * ys + xs[-1] * ys[0] - ys[-1] * xs[0])
    return candidates[int(np.argmax(distances))]["k"]


def _choose_k(matrix: np.ndarray | None, count: int, config: dict, seed: int) -> dict:
    low, high = (int(value) for value in config.get("k_range", [2, 5]))
    ks = range(max(2, low), min(high, count - 1) + 1)
    candidates = []
    if matrix is not None:
        for k in ks:
            model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(matrix)
            silhouette, _ = _score_features(matrix, model.labels_.tolist())
            candidates.append({"k": k, "silhouette": silhouette, "inertia": float(model.inertia_)})
    if not candidates:
        choice = max(1, min(2, count))
        return {"candidates": [], "silhouette_choice": None, "elbow_choice": None, "chosen_k": choice}
    elbow = _elbow_choice(candidates)
    scored = [row for row in candidates if row["silhouette"] is not None]
    silhouette_choice = max(scored, key=lambda row: (row["silhouette"], -abs(row["k"] - elbow), -row["k"]))["k"] if scored else None
    chosen = silhouette_choice if silhouette_choice is not None else elbow
    return {"candidates": candidates, "silhouette_choice": silhouette_choice, "elbow_choice": elbow, "chosen_k": chosen}


def _text_embeddings(texts: list[str], tfidf: np.ndarray, config: dict, seed: int) -> tuple[np.ndarray, str]:
    """Only a local model may replace LSA, preserving offline operation."""
    model_path = config.get("embedding_model")
    if model_path and Path(model_path).exists():
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(str(model_path), local_files_only=True)
            vectors = np.asarray(model.encode(texts, show_progress_bar=False), dtype=float)
            return vectors, "sentence-transformers (local)"
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            pass
    dimension = min(8, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if dimension >= 1:
        vectors = TruncatedSVD(n_components=dimension, random_state=seed).fit_transform(tfidf)
        return np.asarray(vectors, dtype=float), "TF-IDF + TruncatedSVD (LSA)"
    return tfidf.copy(), "TF-IDF (small sample fallback)"


def _spectral(graph: nx.Graph, ids: list[str], k: int, seed: int) -> dict[str, int]:
    if len(ids) < 2 or graph.number_of_edges() == 0 or k < 2:
        return _canonical_labels(ids, [{pid} for pid in ids])
    affinity = nx.to_numpy_array(graph, nodelist=ids, weight="weight")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            values = SpectralClustering(n_clusters=min(k, len(ids)), affinity="precomputed", random_state=seed, assign_labels="kmeans").fit_predict(affinity)
            return _canonical_array(ids, values)
        except (ValueError, np.linalg.LinAlgError):
            return _canonical_labels(ids, [set(group) for group in nx.connected_components(graph)])


def _topic_model(ids: list[str], texts: list[str], k: int, seed: int) -> tuple[dict[str, int], dict[str, list[float]], list[dict], np.ndarray | None]:
    matrix, terms = _tfidf(texts)
    if matrix is None:
        return {pid: -1 for pid in ids}, {pid: [] for pid in ids}, [], None
    topic_count = min(k, matrix.shape[0], matrix.shape[1])
    if topic_count < 1:
        return {pid: -1 for pid in ids}, {pid: [] for pid in ids}, [], matrix
    model = NMF(n_components=topic_count, init="nndsvda", random_state=seed, max_iter=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        weights = model.fit_transform(matrix)
    labels = {}
    mixtures = {}
    for index, pid in enumerate(ids):
        row = weights[index]
        total = float(row.sum())
        mixtures[pid] = [float(value / total) for value in row] if total > 0 else [0.0] * topic_count
        labels[pid] = int(np.argmax(row)) if total > 0 else -1
    topics = []
    for index, row in enumerate(model.components_):
        top_indices = np.argsort(-row, kind="stable")[:5]
        topics.append({"topic": index, "terms": [terms[int(term)] for term in top_indices]})
    return labels, mixtures, topics, matrix


def _quality(ids: list[str], graph: nx.Graph, labels: dict[str, dict[str, int]], matrices: dict[str, np.ndarray | None]) -> dict[str, dict]:
    result = {}
    for method, assignment in labels.items():
        values = [assignment[pid] for pid in ids]
        matrix = matrices.get(method)
        silhouette, db = _score_features(matrix, values)
        if method == "hdbscan" and len(set(values) - {-1}) < 2:
            silhouette, db = None, None
        result[method] = {
            "clusters": len(set(values) - {-1}),
            "noise_count": values.count(-1),
            "modularity": _modularity(graph, assignment) if method in GRAPH_METHODS else None,
            "silhouette": silhouette,
            "davies_bouldin": db,
        }
    return result


def _comparisons(ids: list[str], labels: dict[str, dict[str, int]]) -> list[dict]:
    rows = []
    for left, right in combinations(labels, 2):
        a = [labels[left][pid] for pid in ids]
        b = [labels[right][pid] for pid in ids]
        rows.append({"method_a": left, "method_b": right, "ari": float(adjusted_rand_score(a, b)), "nmi": float(normalized_mutual_info_score(a, b))})
    return rows


def _skill_families(skill_graph: nx.Graph, seed: int) -> list[list[str]]:
    if skill_graph.number_of_nodes() == 0:
        return []
    if skill_graph.number_of_edges() == 0:
        groups = [{node} for node in skill_graph]
    else:
        groups = nx.community.louvain_communities(skill_graph, weight="weight", seed=seed)
    return sorted([sorted(str(node).removeprefix("skill:") for node in group) for group in groups], key=lambda group: group[0])


def cluster_profiles(profiles: list[dict], bundle: dict, config: dict) -> dict:
    """Compare methods on aligned profile IDs and expose weak/degenerate cases."""
    profiles = sorted(profiles, key=lambda profile: profile["id"])
    ids = [profile["id"] for profile in profiles]
    if not ids:
        return {"labels": {}, "quality": {}, "comparisons": [], "k_selection": {"candidates": [], "chosen_k": 0, "silhouette_choice": None, "elbow_choice": None}, "topic_mixtures": {}, "topics": [], "skill_families": [], "embedding_method": "none", "warnings": ["No profiles supplied"]}
    seed = int(config["seed"])
    graph = bundle["graphs"]["combined_manual"].copy()
    graph.add_nodes_from(ids)
    text, skill_text = _feature_texts(profiles)
    tfidf, _ = _tfidf(text)
    k_selection = _choose_k(tfidf, len(ids), config, seed)
    k = k_selection["chosen_k"]
    labels, messages = _graph_communities(graph, ids, seed, config)
    labels["spectral"] = _spectral(graph, ids, k, seed)
    matrices: dict[str, np.ndarray | None] = {}
    embedding_method = "none"
    if tfidf is None:
        messages.append("Feature clustering skipped: no skills or job titles supplied")
        for method in ("tfidf_kmeans", "ward", "embedding_kmeans", "hdbscan"):
            labels[method] = {pid: -1 for pid in ids}
            matrices[method] = None
    else:
        tfidf_labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(tfidf)
        labels["tfidf_kmeans"] = _canonical_array(ids, tfidf_labels)
        matrices["tfidf_kmeans"] = tfidf
        ward_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(tfidf) if len(ids) >= 2 else np.array([0])
        labels["ward"] = _canonical_array(ids, ward_labels)
        matrices["ward"] = tfidf
        embeddings, embedding_method = _text_embeddings(text, tfidf, config, seed)
        embedded_labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embeddings)
        labels["embedding_kmeans"] = _canonical_array(ids, embedded_labels)
        matrices["embedding_kmeans"] = embeddings
        if len(ids) < 2:
            hdbscan_labels = np.array([-1])
        else:
            min_size = min(len(ids), max(2, int(config.get("hdbscan_min_cluster_size", 2))))
            hdbscan_labels = HDBSCAN(min_cluster_size=min_size, copy=False).fit_predict(embeddings)
        labels["hdbscan"] = _canonical_array(ids, hdbscan_labels, preserve_noise=True)
        matrices["hdbscan"] = embeddings
        if all(value == -1 for value in labels["hdbscan"].values()):
            messages.append("HDBSCAN labelled every profile as noise")
    topic_labels, mixtures, topics, skill_matrix = _topic_model(ids, skill_text, k, seed)
    labels["nmf_topics"] = topic_labels
    matrices["nmf_topics"] = skill_matrix
    if not topics:
        messages.append("NMF topics skipped: no usable skill text supplied")
    quality = _quality(ids, graph, labels, matrices)
    comparisons = _comparisons(ids, labels)
    return stable_numbers({
        "labels": labels,
        "quality": quality,
        "comparisons": comparisons,
        "k_selection": k_selection,
        "topic_mixtures": mixtures,
        "topics": topics,
        "skill_families": _skill_families(bundle["skill_projection"], seed),
        "embedding_method": embedding_method,
        "warnings": messages,
    })
