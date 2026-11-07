"""Topic and engagement analysis for observed public or synthetic activity."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _layer(graphs: Mapping[str, Any] | None, name: str) -> nx.Graph | None:
    if graphs is None:
        return None
    candidates: Mapping[str, Any] = graphs
    nested = graphs.get("layers")
    if isinstance(nested, Mapping):
        candidates = nested
    for key in (name, f"G_{name}"):
        graph = candidates.get(key)
        if isinstance(graph, nx.Graph):
            return graph
    return None


def _engagement_graph(profiles: list[dict[str, Any]]) -> nx.Graph:
    """Only named interactions can form edges; counts alone have no known endpoint."""
    ids = {str(profile["id"]) for profile in profiles}
    graph = nx.Graph()
    graph.add_nodes_from(sorted(ids))
    for profile in profiles:
        author = str(profile["id"])
        for post in profile.get("activity", []):
            for engager in post.get("engaged_ids", []):
                engager = str(engager)
                if engager not in ids or engager == author:
                    continue
                if graph.has_edge(author, engager):
                    graph[author][engager]["weight"] += 1
                else:
                    graph.add_edge(author, engager, weight=1)
    return graph


def _correlation(x: list[float], y: list[float]) -> dict[str, float | None]:
    """Undefined correlations remain null rather than becoming a misleading zero."""
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"spearman_rho": None, "p_value": None}
    result = spearmanr(x, y)
    return {
        "spearman_rho": round(float(result.statistic), 6),
        "p_value": round(float(result.pvalue), 6),
    }


def _edge_set(graph: nx.Graph, ids: set[str]) -> set[tuple[str, str]]:
    return {tuple(sorted((str(a), str(b)))) for a, b in graph.edges if str(a) in ids and str(b) in ids and a != b}


def _topics(
    profiles: list[dict[str, Any]], seed: int, requested_topics: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float | None]], str | None]:
    """Fit NMF on posts, then project declared skills into that topic basis."""
    post_records: list[tuple[str, str, str]] = []
    for profile in profiles:
        person_id = str(profile["id"])
        for index, post in enumerate(profile.get("activity", [])):
            body = str(post.get("text", "")).strip()
            if body:
                post_records.append((person_id, str(post.get("post_id") or f"{person_id}_{index}"), body))
    if not post_records:
        return [], [], {}, "No post text was available for topic modelling."
    documents = [record[2] for record in post_records]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return [], [], {}, "Post text contained no usable terms after tokenisation."
    n_topics = min(max(1, requested_topics), matrix.shape[0], matrix.shape[1])
    model = NMF(n_components=n_topics, init="nndsvda", random_state=seed, max_iter=400)
    post_weights = model.fit_transform(matrix)
    terms = vectorizer.get_feature_names_out()
    topics = []
    for topic_id, component in enumerate(model.components_):
        top = np.argsort(-component, kind="stable")[:5]
        topics.append({"topic_id": topic_id, "top_terms": [str(terms[index]) for index in top if component[index] > 0]})
    post_topics = []
    person_indexes: dict[str, list[int]] = defaultdict(list)
    for index, (person_id, post_id, _) in enumerate(post_records):
        weights = post_weights[index]
        topic_id = int(np.argmax(weights))
        post_topics.append({
            "id": person_id,
            "post_id": post_id,
            "dominant_topic": topic_id,
            "topic_weight": round(float(weights[topic_id]), 6),
        })
        person_indexes[person_id].append(index)
    consistency: dict[str, dict[str, float | None]] = {}
    for profile in profiles:
        person_id = str(profile["id"])
        indexes = person_indexes.get(person_id, [])
        skills = " ".join(str(skill) for skill in profile.get("skills", []) if str(skill).strip())
        if not indexes or not skills:
            consistency[person_id] = {"topic_cosine": None, "tfidf_cosine": None}
            continue
        skill_vector = vectorizer.transform([skills])
        post_vector = np.asarray(matrix[indexes].sum(axis=0)).reshape(1, -1)
        lexical = float(cosine_similarity(skill_vector, post_vector)[0, 0])
        skill_topics = model.transform(skill_vector)
        person_topics = post_weights[indexes].sum(axis=0).reshape(1, -1)
        topic_score = float(cosine_similarity(skill_topics, person_topics)[0, 0])
        consistency[person_id] = {
            "topic_cosine": round(topic_score, 6),
            "tfidf_cosine": round(lexical, 6),
        }
    return topics, post_topics, consistency, None


def analyze_activity(
    profiles: list[dict[str, Any]],
    graphs: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare observed engagement with workplace ties and declared skills."""
    settings = config or {}
    ordered = sorted(profiles, key=lambda profile: str(profile["id"]))
    ids = {str(profile["id"]) for profile in ordered}
    seed = int(settings.get("seed", 0))
    requested_topics = int(settings.get("activity_topics", 3))
    topics, post_topics, consistency, topic_warning = _topics(ordered, seed, requested_topics)
    org_graph = _layer(graphs, "org")
    activity_graph = _layer(graphs, "activity") or _engagement_graph(ordered)
    rows = []
    for profile in ordered:
        person_id = str(profile["id"])
        posts = profile.get("activity", [])
        rows.append({
            "id": person_id,
            "post_count": len(posts),
            "reaction_count": sum(max(0, int(post.get("reaction_count", 0))) for post in posts),
            "org_weighted_degree": round(float(org_graph.degree(person_id, weight="weight")), 6)
            if org_graph is not None and person_id in org_graph else None,
            "engagement_degree": int(activity_graph.degree(person_id)) if person_id in activity_graph else 0,
            "engagement_weighted_degree": round(float(activity_graph.degree(person_id, weight="weight")), 6)
            if person_id in activity_graph else 0.0,
            "topic_consistency": consistency.get(person_id, {}).get("topic_cosine"),
            "tfidf_consistency": consistency.get(person_id, {}).get("tfidf_cosine"),
        })
    org_edges = _edge_set(org_graph, ids) if org_graph is not None else set()
    activity_edges = _edge_set(activity_graph, ids)
    overlap = len(org_edges & activity_edges) / len(org_edges | activity_edges) if org_graph is not None and (org_edges | activity_edges) else None
    comparable = [row for row in rows if row["org_weighted_degree"] is not None]
    org_degree = [float(row["org_weighted_degree"]) for row in comparable]
    engagement_degree = [float(row["engagement_weighted_degree"]) for row in comparable]
    reactions = [float(row["reaction_count"]) for row in comparable]
    warnings = [topic_warning] if topic_warning else []
    if not activity_edges:
        warnings.append("No identified engaged_ids were available; the engagement graph comparison is skipped.")
    if org_graph is None:
        warnings.append("The organisational graph was unavailable; graph comparison is skipped.")
    return {
        "topic_method": "TF-IDF + NMF on post text",
        "topics": topics,
        "post_topics": post_topics,
        "people": rows,
        "engagement_graph": {
            "directed": False,
            "edges": [
                {"source": min(str(a), str(b)), "target": max(str(a), str(b)), "weight": float(data.get("weight", 1))}
                for a, b, data in sorted(activity_graph.edges(data=True), key=lambda edge: tuple(sorted((str(edge[0]), str(edge[1])))))
            ],
        },
        "graph_comparison": {
            "org_engagement_edge_jaccard": round(overlap, 6) if overlap is not None else None,
            "org_degree_vs_engagement_degree": _correlation(org_degree, engagement_degree) if activity_edges else {"spearman_rho": None, "p_value": None},
            "org_degree_vs_reactions": _correlation(org_degree, reactions),
        },
        "consistency_method": "Cosine of NMF topic mixtures for posts and projected skills; TF-IDF cosine as a lexical alternative.",
        "warnings": warnings,
    }
