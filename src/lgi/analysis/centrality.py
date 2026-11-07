"""Compare complementary notions of centrality instead of trusting one score."""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.stats import kendalltau, spearmanr

METRICS = ("degree", "weighted_degree", "betweenness", "closeness", "harmonic", "eigenvector", "pagerank", "katz", "k_core", "constraint", "effective_size")


def _distance_graph(graph: nx.Graph) -> nx.Graph:
    result = graph.copy()
    for _, _, data in result.edges(data=True):
        data["distance"] = 1.0 / data["weight"]
    return result


def _finite(value: float, fallback: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else fallback


def _eigenvector(graph: nx.Graph) -> dict[str, float]:
    """Solve components independently because numpy's global solution is ambiguous."""
    output = {node: 0.0 for node in graph.nodes}
    for nodes in nx.connected_components(graph):
        component = graph.subgraph(nodes)
        if component.number_of_nodes() == 1:
            continue
        try:
            values = ({node: 2 ** -0.5 for node in component} if component.number_of_nodes() == 2 else nx.eigenvector_centrality_numpy(component, weight="weight"))
        except (nx.NetworkXException, TypeError, ValueError):
            values = nx.eigenvector_centrality(component, weight="weight", max_iter=1000)
        output.update({node: _finite(value) for node, value in values.items()})
    return output


def _katz(graph: nx.Graph) -> dict[str, float]:
    if not graph.nodes:
        return {}
    matrix = nx.to_numpy_array(graph, nodelist=sorted(graph), weight="weight")
    radius = float(max(abs(np.linalg.eigvals(matrix)))) if matrix.size else 0.0
    alpha = 0.9 / radius if radius > 0 else 0.1
    try:
        values = nx.katz_centrality(graph, alpha=alpha, beta=1.0, weight="weight", max_iter=1000, tol=1e-8)
    except nx.PowerIterationFailedConvergence:
        try:
            values = nx.katz_centrality_numpy(graph, alpha=alpha, beta=1.0, weight="weight")
        except (nx.NetworkXException, ValueError):
            values = {node: 0.0 for node in graph}
    return {node: _finite(value) for node, value in values.items()}


def compute_metrics(graph: nx.Graph) -> dict[str, dict[str, float]]:
    """Distances invert weights so a stronger relationship is a shorter path."""
    nodes = sorted(graph.nodes)
    if not nodes:
        return {}
    distance = _distance_graph(graph)
    core = nx.core_number(graph) if graph.number_of_edges() else {node: 0 for node in nodes}
    constraint = nx.constraint(graph, weight="weight")
    effective = nx.effective_size(graph, weight="weight")
    measures = {
        "degree": nx.degree_centrality(graph),
        "weighted_degree": dict(graph.degree(weight="weight")),
        "betweenness": nx.betweenness_centrality(distance, weight="distance"),
        "closeness": nx.closeness_centrality(distance, distance="distance"),
        "harmonic": nx.harmonic_centrality(distance, distance="distance"),
        "eigenvector": _eigenvector(graph),
        "pagerank": nx.pagerank(graph, weight="weight"),
        "katz": _katz(graph),
        "k_core": core,
        "constraint": constraint,
        "effective_size": effective,
    }
    return {node: {name: round(_finite(values.get(node, 0.0), 1.0 if name == "constraint" else 0.0), 12) for name, values in measures.items()} for node in nodes}


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else 0.0


def _roles(metrics: dict[str, dict[str, float]], config: dict) -> dict[str, list[str]]:
    thresholds = config["role_thresholds"]
    if not metrics:
        return {}
    output = {}
    for pid, values in metrics.items():
        roles = []
        if values["weighted_degree"] > 0 and values["weighted_degree"] >= _quantile([m["weighted_degree"] for m in metrics.values()], thresholds["hub_degree_quantile"]):
            roles.append("hub")
        if values["betweenness"] > 0 and values["betweenness"] >= _quantile([m["betweenness"] for m in metrics.values()], thresholds["bridge_betweenness_quantile"]) and values["constraint"] <= _quantile([m["constraint"] for m in metrics.values()], thresholds["bridge_constraint_quantile"]):
            roles.append("bridge")
        if values["pagerank"] > 0 and values["pagerank"] >= _quantile([m["pagerank"] for m in metrics.values()], thresholds["influencer_pagerank_quantile"]):
            roles.append("influencer")
        output[pid] = roles or ["peripheral"]
    return output


def _correlations(metrics: dict[str, dict[str, float]]) -> dict:
    nodes = sorted(metrics)
    output: dict[str, dict] = {"kendall": {}, "spearman": {}}
    for first in METRICS:
        output["kendall"][first] = {}
        output["spearman"][first] = {}
        x = [metrics[node][first] for node in nodes]
        for second in METRICS:
            y = [metrics[node][second] for node in nodes]
            if first == second:
                tau = rho = 1.0
            elif len(set(x)) < 2 or len(set(y)) < 2:
                tau = rho = float("nan")
            else:
                tau = kendalltau(x, y).statistic
                rho = spearmanr(x, y).statistic
            output["kendall"][first][second] = round(float(tau), 6) if np.isfinite(tau) else None
            output["spearman"][first][second] = round(float(rho), 6) if np.isfinite(rho) else None
    return output


def analyze_centrality(graph: nx.Graph, config: dict) -> dict:
    metrics = compute_metrics(graph)
    nodes = sorted(metrics)
    rankings = {}
    for metric in METRICS:
        reverse = metric != "constraint"
        rankings[metric] = sorted(nodes, key=lambda node: ((-1 if reverse else 1) * metrics[node][metric], node))
    borda = {node: 0 for node in nodes}
    mrr = {node: 0.0 for node in nodes}
    for ranking in rankings.values():
        for position, node in enumerate(ranking, 1):
            borda[node] += len(nodes) - position
            mrr[node] += 1 / position
    rankings["borda"] = sorted(nodes, key=lambda node: (-borda[node], node))
    rankings["mrr"] = sorted(nodes, key=lambda node: (-mrr[node], node))
    roles = _roles(metrics, config)
    aggregate = [{"id": node, "rank": position, "borda": borda[node], "mrr": round(mrr[node], 6), "roles": roles[node]} for position, node in enumerate(rankings["borda"], 1)]
    return {"metrics": metrics, "rankings": rankings, "roles": roles, "aggregate": aggregate, "correlations": _correlations(metrics)}
