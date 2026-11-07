"""Build independent evidence layers before combining profile relationships."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lgi.analysis.embeddings import semantic_similarity

LAYERS = ("org", "edu", "skill", "semantic", "activity")


def _month_number(value: str) -> int:
    year, month = map(int, value.split("-"))
    return year * 12 + month


def _overlap_months(a: dict, b: dict, reference: str) -> int:
    first = max(_month_number(a["start"]), _month_number(b["start"]))
    last = min(_month_number(a["end"] or reference), _month_number(b["end"] or reference))
    return max(0, last - first + 1)


def _org_weight(a: dict, b: dict, reference: str, ignored: set[str]) -> float:
    return float(sum(
        _overlap_months(x, y, reference)
        for x in a["experience"] for y in b["experience"]
        if x["company"] and x["company"].casefold() == y["company"].casefold()
        and x["company"].casefold() not in ignored
    ))


def _edu_weight(a: dict, b: dict) -> float:
    weight = 0.0
    for x in a["education"]:
        for y in b["education"]:
            same_school = x["school"].casefold() == y["school"].casefold()
            same_field = x["field"].casefold() == y["field"].casefold()
            if not (same_school or same_field):
                continue
            shared_years = max(0, min(x["end_year"], y["end_year"]) - max(x["start_year"], y["start_year"]) + 1)
            weight += shared_years + (1.0 if same_field else 0.0)
    return weight


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _ignored_employers(profiles: list[dict], maximum_share: float) -> list[dict]:
    """A nearly universal employer identifies the cohort, not a close pair."""
    if not 0 <= maximum_share <= 1:
        raise ValueError("org_ignore_employer_share must be in [0, 1]")
    holders: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    for profile in profiles:
        for job in profile["experience"]:
            company = job["company"].strip()
            if company:
                key = company.casefold()
                holders[key].add(profile["id"])
                display.setdefault(key, company)
    return [
        {"company": display[key], "profiles": len(people), "share": round(len(people) / len(profiles), 9)}
        for key, people in sorted(holders.items())
        if len(people) / len(profiles) > maximum_share
    ] if profiles else []


def _add_knn(graph: nx.Graph, ids: list[str], candidates: dict[tuple[str, str], dict], k: int, threshold: float) -> None:
    """Union each person's top-k positive ties; break equal scores by profile ID."""
    if k < 1:
        raise ValueError("similarity_knn_k must be positive")
    chosen: set[tuple[str, str]] = set()
    for pid in ids:
        neighbours = []
        for edge, data in candidates.items():
            weight = data["weight"]
            if pid in edge and weight > 0 and weight >= threshold:
                other = edge[1] if edge[0] == pid else edge[0]
                neighbours.append((weight, other, edge))
        neighbours.sort(key=lambda row: (-row[0], row[1]))
        chosen.update(edge for _, _, edge in neighbours[:k])
    for left, right in sorted(chosen):
        data = candidates[(left, right)]
        _add(graph, left, right, data["weight"], threshold, **data["alternatives"])


def _make_layer(ids: list[str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    return graph


def _add(graph: nx.Graph, a: str, b: str, weight: float, threshold: float, **attributes: float) -> None:
    if weight > 0 and weight >= threshold:
        graph.add_edge(a, b, weight=round(float(weight), 9), **{key: round(float(value), 9) for key, value in attributes.items()})


def _combine(graphs: dict[str, nx.Graph], weights: dict[str, float], scheme: str) -> nx.Graph:
    ids = sorted(next(iter(graphs.values())).nodes)
    result = _make_layer(ids)
    normalised: dict[str, dict[tuple[str, str], float]] = {}
    for layer in LAYERS:
        edges = {(min(a, b), max(a, b)): float(data["weight"]) for a, b, data in graphs[layer].edges(data=True)}
        values = list(edges.values())
        if not values:
            normalised[layer] = {}
            continue
        low, high = min(values), max(values)
        if scheme == "equal":
            normalised[layer] = {edge: (value - low) / (high - low) if high > low else 1.0 for edge, value in edges.items()}
        else:
            normalised[layer] = {edge: value / high for edge, value in edges.items()}
    all_edges = set().union(*(set(edges) for edges in normalised.values()))
    for edge in sorted(all_edges):
        score = sum(weights[layer] * normalised[layer].get(edge, 0.0) for layer in LAYERS)
        if score > 0:
            result.add_edge(*edge, weight=round(score, 12))
    return result


def combine_with_weights(graphs: dict[str, nx.Graph], weights: dict[str, float]) -> nx.Graph:
    """Ablation and sensitivity reuse the identical manual combination rule."""
    return _combine(graphs, weights, "manual")


def _bipartite(profiles: list[dict]) -> tuple[nx.Graph, nx.Graph, nx.Graph]:
    bipartite = nx.Graph()
    for profile in profiles:
        person = f"person:{profile['id']}"
        bipartite.add_node(person, kind="person")
        for skill in profile["skills"]:
            node = f"skill:{skill.casefold()}"
            bipartite.add_node(node, kind="skill", label=skill)
            bipartite.add_edge(person, node, weight=1.0)
    people = [node for node, data in bipartite.nodes(data=True) if data["kind"] == "person"]
    skills = [node for node, data in bipartite.nodes(data=True) if data["kind"] == "skill"]
    person_projection = nx.bipartite.weighted_projected_graph(bipartite, people)
    skill_projection = nx.bipartite.weighted_projected_graph(bipartite, skills)
    return bipartite, person_projection, skill_projection


def build_graphs(profiles: list[dict], config: dict) -> dict:
    """Store both similarity alternatives even when one drives the selected layer."""
    profiles = sorted(profiles, key=lambda p: p["id"])
    ids = [p["id"] for p in profiles]
    graphs = {name: _make_layer(ids) for name in LAYERS}
    threshold = config["thresholds"]
    reference = config.get("reference_month", "2026-01")
    ignored_employers = _ignored_employers(profiles, float(config.get("org_ignore_employer_share", 0.8)))
    ignored_keys = {row["company"].casefold() for row in ignored_employers}
    knn_k = int(config.get("similarity_knn_k", 3))
    skill_candidates: dict[tuple[str, str], dict] = {}
    semantic_candidates: dict[tuple[str, str], dict] = {}
    skill_sets = [{s.casefold() for s in p["skills"]} for p in profiles]
    skill_docs = [" ".join(sorted(values)) or "empty" for values in skill_sets]
    skill_tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(skill_docs)
    skill_cosine = cosine_similarity(skill_tfidf)
    semantic_texts = [" ".join([p["basic"]["headline"], p["basic"]["summary"], *[job["title"] for job in p["experience"]]]) for p in profiles]
    semantic, semantic_tfidf, embedding_method = semantic_similarity(semantic_texts, config)
    for i, a in enumerate(profiles):
        for j in range(i + 1, len(profiles)):
            b = profiles[j]
            left, right = a["id"], b["id"]
            _add(graphs["org"], left, right, _org_weight(a, b, reference, ignored_keys), threshold["org"])
            _add(graphs["edu"], left, right, _edu_weight(a, b), threshold["edu"])
            jaccard = _jaccard(skill_sets[i], skill_sets[j])
            tfidf = float(skill_cosine[i, j])
            skill_weight = jaccard if config["skill_metric"] == "jaccard" else tfidf
            skill_candidates[(left, right)] = {"weight": skill_weight, "alternatives": {"jaccard": jaccard, "tfidf_cosine": tfidf}}
            lsa = float(semantic[i, j])
            semantic_alt = float(semantic_tfidf[i, j])
            semantic_weight = lsa if config["semantic_metric"] == "lsa" else semantic_alt
            semantic_candidates[(left, right)] = {"weight": semantic_weight, "alternatives": {"embedding_cosine": lsa, "tfidf_cosine": semantic_alt}}
    _add_knn(graphs["skill"], ids, skill_candidates, knn_k, threshold["skill"])
    _add_knn(graphs["semantic"], ids, semantic_candidates, knn_k, threshold["semantic"])
    interactions = Counter()
    known = set(ids)
    for profile in profiles:
        for post in profile["activity"]:
            for engager in post["engaged_ids"]:
                if engager in known and engager != profile["id"]:
                    interactions[tuple(sorted((profile["id"], engager)))] += 1
    for (left, right), count in sorted(interactions.items()):
        _add(graphs["activity"], left, right, float(count), threshold["activity"])
    manual_weights = {name: float(config["layer_weights"][name]) for name in LAYERS}
    graphs["combined_manual"] = _combine(graphs, manual_weights, "manual")
    graphs["combined_equal"] = _combine(graphs, {name: 1 / len(LAYERS) for name in LAYERS}, "equal")
    bipartite, person_projection, skill_projection = _bipartite(profiles)
    multilayer = {pid: {name: round(float(graphs[name].degree(pid, weight="weight")), 12) for name in LAYERS} for pid in ids}
    warnings = [] if graphs["activity"].number_of_edges() else ["Activity layer skipped: no identified engagers supplied"]
    if ignored_employers:
        names = ", ".join(f"{row['company']} ({row['profiles']}/{len(profiles)})" for row in ignored_employers)
        warnings.append(f"Org layer ignored prevalent employers: {names}")
    return {"graphs": graphs, "ignored_employers": ignored_employers, "bipartite": bipartite, "person_projection": person_projection, "skill_projection": skill_projection, "multilayer": multilayer, "embedding_method": embedding_method, "warnings": warnings}


def _node_link(graph: nx.Graph) -> dict:
    return {
        "directed": graph.is_directed(),
        "nodes": [{"id": node, **dict(sorted(data.items()))} for node, data in sorted(graph.nodes(data=True))],
        "links": [{"source": min(a, b), "target": max(a, b), **dict(sorted(data.items()))} for a, b, data in sorted(graph.edges(data=True), key=lambda row: tuple(sorted(row[:2])))],
    }


def write_graph_json(bundle: dict, path: str | Path, metrics: dict | None = None, roles: dict | None = None, clusters: dict | None = None) -> None:
    """Export all layers with stable ordering and profile annotations."""
    graphs = bundle["graphs"]
    combined = _node_link(graphs["combined_manual"])
    for node in combined["nodes"]:
        pid = node["id"]
        node["metrics"] = (metrics or {}).get(pid, {})
        node["roles"] = (roles or {}).get(pid, [])
        node["clusters"] = {name: labels.get(pid) for name, labels in (clusters or {}).items()}
        node["multilayer"] = bundle["multilayer"][pid]
    output = {**combined, "layers": {name: _node_link(graph) for name, graph in sorted(graphs.items())}, "bipartite": _node_link(bundle["bipartite"]), "person_projection": _node_link(bundle["person_projection"]), "skill_projection": _node_link(bundle["skill_projection"]), "embedding_method": bundle["embedding_method"], "ignored_employers": bundle["ignored_employers"], "warnings": bundle["warnings"]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
