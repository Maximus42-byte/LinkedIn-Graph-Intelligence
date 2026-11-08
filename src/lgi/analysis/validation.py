"""Test how much the synthetic graph conclusions depend on edges and weights.

All randomness comes from the configured seed. Ground truth is read only here and
only when its path is explicitly supplied by the fixture pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.stats import kendalltau, rankdata
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from lgi.analysis.graphs import LAYERS, combine_with_weights
from lgi.analysis.util import stable_numbers


def _ranking_metrics(graph: nx.Graph, seed: int) -> dict[str, dict[str, float]]:
    """Use three complementary measures so a single edge property cannot dominate."""
    ids = sorted(graph.nodes)
    if not ids:
        return {name: {} for name in ("weighted_degree", "betweenness", "pagerank")}
    lengths = graph.copy()
    for _, _, data in lengths.edges(data=True):
        data["distance"] = 1.0 / max(float(data.get("weight", 1.0)), 1e-12)
    degree = {pid: float(graph.degree(pid, weight="weight")) for pid in ids}
    sample_size = None if len(ids) <= 50 else min(32, len(ids))
    betweenness = nx.betweenness_centrality(
        lengths, k=sample_size, weight="distance", seed=seed
    )
    try:
        pagerank = nx.pagerank(graph, weight="weight", max_iter=1000, tol=1e-10)
    except nx.PowerIterationFailedConvergence:
        total = sum(degree.values())
        pagerank = {pid: degree[pid] / total if total else 1.0 / len(ids) for pid in ids}
    return {
        "weighted_degree": degree,
        "betweenness": {pid: float(betweenness[pid]) for pid in ids},
        "pagerank": {pid: float(pagerank[pid]) for pid in ids},
    }


def _ranking(graph: nx.Graph, seed: int) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Use average ranks for tied centrality values before Borda aggregation."""
    metrics = _ranking_metrics(graph, seed)
    ids = sorted(graph.nodes)
    rank_sum = {pid: 0.0 for pid in ids}
    for values in metrics.values():
        ranks = rankdata([-values[pid] for pid in ids], method="average")
        for pid, rank in zip(ids, ranks):
            rank_sum[pid] += float(rank)
    aggregate = rankdata([rank_sum[pid] for pid in ids], method="average") if ids else []
    metrics["aggregate_rank"] = {pid: float(rank) for pid, rank in zip(ids, aggregate)}
    return sorted(ids, key=lambda pid: (rank_sum[pid], pid)), metrics


def _topk_set(ranks: dict[str, float], k: int) -> set[str]:
    """Include every person tied at the requested cutoff."""
    if not ranks or k <= 0:
        return set()
    cutoff = sorted(ranks.values())[min(k, len(ranks)) - 1]
    return {pid for pid, rank in ranks.items() if rank <= cutoff}


def _tau(reference: dict[str, float], candidate: dict[str, float]) -> float | None:
    """Kendall tau-b accounts for ties without assigning arbitrary ID order."""
    ids = sorted(reference)
    if len(ids) < 2:
        return None
    statistic = kendalltau(
        [reference[pid] for pid in ids], [candidate[pid] for pid in ids]
    ).statistic
    return None if np.isnan(statistic) else float(statistic)


def _quantiles(values: list[float | int]) -> list[float] | None:
    if not values:
        return None
    return [float(x) for x in np.quantile(values, [0.025, 0.5, 0.975])]


def _communities(graph: nx.Graph, seed: int) -> dict[str, int]:
    """Louvain makes the cluster stability checks match the graph comparison."""
    ids = sorted(graph.nodes)
    if not graph.number_of_edges():
        return {pid: index for index, pid in enumerate(ids)}
    groups = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    ordered = sorted((sorted(group) for group in groups), key=lambda group: group[0])
    return {pid: index for index, group in enumerate(ordered) for pid in group}


def _ari(a: dict[str, int], b: dict[str, int], ids: list[str]) -> float:
    if not ids:
        return 1.0
    return float(adjusted_rand_score([a[pid] for pid in ids], [b[pid] for pid in ids]))


def _perturb(graph: nx.Graph, fraction: float, rng: np.random.Generator) -> nx.Graph:
    """Remove and add a similar number of ties while retaining the node set."""
    result = graph.copy()
    edges = sorted((min(a, b), max(a, b)) for a, b in graph.edges)
    ids = sorted(graph.nodes)
    if not edges:
        return result
    count = max(1, int(round(len(edges) * fraction))) if fraction > 0 else 0
    count = min(count, len(edges))
    if count:
        picked = rng.choice(len(edges), size=count, replace=False)
        for index in np.atleast_1d(picked):
            result.remove_edge(*edges[int(index)])
    nonedges = sorted((a, b) for i, a in enumerate(ids) for b in ids[i + 1 :] if not graph.has_edge(a, b))
    # A complete graph has no possible new pairs. Keep its removed edges absent
    # instead of silently restoring the original topology.
    added = min(count, len(nonedges))
    weights = [float(graph[a][b].get("weight", 1.0)) for a, b in edges]
    if added:
        selected = rng.choice(len(nonedges), size=added, replace=False)
        for index in np.atleast_1d(selected):
            a, b = nonedges[int(index)]
            result.add_edge(a, b, weight=float(rng.choice(weights)))
    return result


def _perturbation(graph: nx.Graph, config: dict[str, Any], seed: int, baseline: list[str], baseline_ranks: dict[str, float]) -> dict[str, Any]:
    iterations = max(0, int(config.get("perturbation_iterations", 1000)))
    fraction = float(config.get("perturbation_fraction", 0.10))
    rng = np.random.default_rng(seed + 101)
    ids = sorted(graph.nodes)
    ranks: dict[str, list[float]] = {pid: [] for pid in ids}
    top3_counts = {pid: 0 for pid in ids}
    taus: list[float] = []
    top3_exact = 0
    top1_exact = 0
    top3_overlap_total = 0
    top3_reference = _topk_set(baseline_ranks, 3)
    top1_reference = _topk_set(baseline_ranks, 1)
    edge_count = graph.number_of_edges()
    removed_count = min(edge_count, max(1, int(round(edge_count * fraction)))) if edge_count and fraction > 0 else 0
    possible_additions = len(ids) * (len(ids) - 1) // 2 - edge_count
    added_count = min(removed_count, possible_additions)
    for iteration in range(iterations):
        altered = _perturb(graph, fraction, rng)
        _, measures = _ranking(altered, seed + iteration + 1)
        candidate_ranks = measures["aggregate_rank"]
        candidate_top3 = _topk_set(candidate_ranks, 3)
        candidate_top1 = _topk_set(candidate_ranks, 1)
        top3_overlap_total += len(candidate_top3 & top3_reference)
        top1_exact += candidate_top1 == top1_reference
        for pid in ids:
            ranks[pid].append(candidate_ranks[pid])
            top3_counts[pid] += pid in candidate_top3
        tau = _tau(baseline_ranks, candidate_ranks)
        if tau is not None:
            taus.append(tau)
        top3_exact += candidate_top3 == top3_reference
    return {
        "iterations": iterations,
        "edge_fraction": fraction,
        "removed_edges_per_iteration": removed_count,
        "added_edges_per_iteration": added_count,
        "baseline_ranking": baseline,
        "baseline_top3_set": sorted(top3_reference),
        "baseline_top1_set": sorted(top1_reference),
        "top3_set_retention": top3_exact / iterations if iterations else None,
        "top3_mean_overlap": top3_overlap_total / iterations if iterations else None,
        "top3_chance_overlap": min(3, len(ids)) ** 2 / len(ids) if ids else None,
        "top1_set_retention": top1_exact / iterations if iterations else None,
        "kendall_tau_mean": float(np.mean(taus)) if taus else None,
        "kendall_tau_interval": _quantiles(taus),
        "per_person": {
            pid: {
                "baseline_rank": baseline_ranks[pid],
                "rank_interval": [float(value) for value in np.quantile(ranks[pid], [0.05, 0.5, 0.95])] if ranks[pid] else None,
                "rank_min": min(ranks[pid]) if ranks[pid] else None,
                "rank_max": max(ranks[pid]) if ranks[pid] else None,
                "top3_frequency": top3_counts[pid] / iterations if iterations else None,
            }
            for pid in ids
        },
    }


def _null_backbone(graph: nx.Graph, config: dict[str, Any]) -> tuple[nx.Graph, dict[str, Any]]:
    """A dense graph has no nonedges for a degree-preserving swap to use."""
    density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    limit = float(config.get("null_backbone_density_threshold", 0.80))
    if density < limit:
        return graph.copy(), {"used": False, "density": density, "edges": graph.number_of_edges()}
    keep_fraction = float(config.get("null_backbone_keep_fraction", 0.50))
    edges = sorted(
        graph.edges(data=True),
        key=lambda row: (-float(row[2].get("weight", 1.0)), min(row[0], row[1]), max(row[0], row[1])),
    )
    count = max(1, min(len(edges), int(round(len(edges) * keep_fraction)))) if edges else 0
    backbone = nx.Graph()
    backbone.add_nodes_from(graph.nodes(data=True))
    for a, b, data in edges[:count]:
        backbone.add_edge(a, b, **data)
    return backbone, {
        "used": True,
        "density": density,
        "edges": backbone.number_of_edges(),
        "keep_fraction": keep_fraction,
        "rule": "highest-weight edges of the combined graph",
    }


def _modularity_and_top(graph: nx.Graph, seed: int) -> dict[str, float]:
    labels = _communities(graph, seed)
    if graph.number_of_edges():
        groups: dict[int, set[str]] = {}
        for pid, label in labels.items():
            groups.setdefault(label, set()).add(pid)
        modularity = float(nx.community.modularity(graph, list(groups.values()), weight="weight"))
    else:
        modularity = 0.0
    _, metrics = _ranking(graph, seed)
    return {
        "modularity": modularity,
        "top_weighted_degree": max(metrics["weighted_degree"].values(), default=0.0),
        "top_betweenness": max(metrics["betweenness"].values(), default=0.0),
        "top_pagerank": max(metrics["pagerank"].values(), default=0.0),
    }


def _assign_weights(graph: nx.Graph, weights: list[float], rng: np.random.Generator) -> None:
    """Keep the empirical weight distribution in each topology null draw."""
    if not weights:
        return
    shuffled = rng.permutation(weights)
    for index, (a, b) in enumerate(sorted(graph.edges)):
        graph[a][b]["weight"] = float(shuffled[index])


def _null_summary(observed: dict[str, float], samples: list[dict[str, float]], failures: int) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    for key, real in observed.items():
        values = [sample[key] for sample in samples]
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            statistics[key] = {
                "observed": real,
                "null_mean": mean,
                "null_std": std,
                "z_score": (real - mean) / std if std > 1e-12 else None,
                "empirical_p_upper": (1 + sum(value >= real for value in values)) / (1 + len(values)),
            }
        else:
            statistics[key] = {"observed": real, "null_mean": None, "null_std": None, "z_score": None, "empirical_p_upper": None}
    return {"samples": len(samples), "failures": failures, "statistics": statistics}


def _null_models(graph: nx.Graph, config: dict[str, Any], seed: int) -> dict[str, Any]:
    backbone, backbone_info = _null_backbone(graph, config)
    ids = sorted(backbone.nodes)
    n = len(ids)
    m = backbone.number_of_edges()
    iterations = max(0, int(config.get("null_iterations", 100)))
    rng = np.random.default_rng(seed + 203)
    observed = _modularity_and_top(backbone, seed)
    weights = [float(data.get("weight", 1.0)) for _, _, data in backbone.edges(data=True)]
    swapped_samples: list[dict[str, float]] = []
    gnm_samples: list[dict[str, float]] = []
    swap_failures = 0
    swaps_per_node = max(1, int(config.get("null_swaps_per_node", 5)))
    swaps_per_draw = max(1, min(m, swaps_per_node * n))
    for iteration in range(iterations):
        if m >= 2:
            swapped = backbone.copy()
            try:
                nx.double_edge_swap(
                    swapped,
                    nswap=swaps_per_draw,
                    max_tries=max(100, 50 * swaps_per_draw),
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
            except (nx.NetworkXAlgorithmError, nx.NetworkXError):
                swap_failures += 1
            else:
                _assign_weights(swapped, weights, rng)
                swapped_samples.append(_modularity_and_top(swapped, seed + iteration + 1))
        else:
            swap_failures += 1
        random_graph = nx.gnm_random_graph(n, m, seed=int(rng.integers(0, 2**31 - 1)))
        random_graph = nx.relabel_nodes(random_graph, dict(enumerate(ids)))
        _assign_weights(random_graph, weights, rng)
        gnm_samples.append(_modularity_and_top(random_graph, seed + iteration + 1))
    minimum_n = int(config.get("null_min_informative_n", 30))
    caveat = (
        f"n={n} is too small for a statistically informative null comparison; "
        "empirical p-values and z-scores are descriptive only."
        if n < minimum_n
        else "Null comparisons remain conditional on the chosen graph and null models."
    )
    if not swapped_samples and iterations:
        caveat += " Degree-preserving edge swaps were infeasible for this topology."
    return {
        "iterations_requested": iterations,
        "swaps_per_draw": swaps_per_draw,
        "graph_nodes": n,
        "graph_edges": m,
        "backbone": backbone_info,
        "observed": observed,
        "degree_preserving": _null_summary(observed, swapped_samples, swap_failures),
        "gnm": _null_summary(observed, gnm_samples, 0),
        "statistically_informative": n >= minimum_n and bool(swapped_samples),
        "caveat": caveat,
    }


def _ablation(graphs: dict[str, nx.Graph], config: dict[str, Any], seed: int, baseline_ranks: dict[str, float], baseline_labels: dict[str, int]) -> dict[str, Any]:
    ids = sorted(graphs["combined_manual"].nodes)
    base_weights = {layer: float(config["layer_weights"][layer]) for layer in LAYERS}
    result: dict[str, Any] = {}
    for layer in LAYERS:
        weights = {name: (0.0 if name == layer else value) for name, value in base_weights.items()}
        reduced = combine_with_weights(graphs, weights)
        _, measures = _ranking(reduced, seed)
        candidate_ranks = measures["aggregate_rank"]
        labels = _communities(reduced, seed)
        result[layer] = {
            "remaining_edges": reduced.number_of_edges(),
            "rank_kendall_tau": _tau(baseline_ranks, candidate_ranks),
            "cluster_ari": _ari(baseline_labels, labels, ids),
            "top3_set_retained": _topk_set(candidate_ranks, 3) == _topk_set(baseline_ranks, 3),
        }
    return result


def _weight_sensitivity(graphs: dict[str, nx.Graph], config: dict[str, Any], seed: int, baseline_ranks: dict[str, float], baseline_labels: dict[str, int]) -> dict[str, Any]:
    iterations = max(0, int(config.get("weight_iterations", 100)))
    concentration = float(config.get("weight_dirichlet_concentration", 20.0))
    manual = np.array([float(config["layer_weights"][name]) for name in LAYERS], dtype=float)
    manual = manual / manual.sum()
    alpha = np.maximum(manual * concentration, 1e-6)
    rng = np.random.default_rng(seed + 307)
    ids = sorted(graphs["combined_manual"].nodes)
    taus: list[float] = []
    aris: list[float] = []
    top3_exact = 0
    top1_exact = 0
    top3_overlap_total = 0
    baseline_top3 = _topk_set(baseline_ranks, 3)
    baseline_top1 = _topk_set(baseline_ranks, 1)
    ranks: dict[str, list[float]] = {pid: [] for pid in ids}
    for iteration in range(iterations):
        sample = rng.dirichlet(alpha)
        weighted = combine_with_weights(graphs, dict(zip(LAYERS, sample)))
        _, measures = _ranking(weighted, seed + iteration + 1)
        candidate_ranks = measures["aggregate_rank"]
        labels = _communities(weighted, seed + iteration + 1)
        tau = _tau(baseline_ranks, candidate_ranks)
        if tau is not None:
            taus.append(tau)
        aris.append(_ari(baseline_labels, labels, ids))
        candidate_top3 = _topk_set(candidate_ranks, 3)
        top3_exact += candidate_top3 == baseline_top3
        top3_overlap_total += len(candidate_top3 & baseline_top3)
        top1_exact += _topk_set(candidate_ranks, 1) == baseline_top1
        for pid in ids:
            ranks[pid].append(candidate_ranks[pid])
    return {
        "iterations": iterations,
        "dirichlet_concentration": concentration,
        "baseline_weights": dict(zip(LAYERS, [float(value) for value in manual])),
        "kendall_tau_mean": float(np.mean(taus)) if taus else None,
        "kendall_tau_interval": _quantiles(taus),
        "cluster_ari_interval": _quantiles(aris),
        "top3_set_retention": top3_exact / iterations if iterations else None,
        "top3_mean_overlap": top3_overlap_total / iterations if iterations else None,
        "top3_chance_overlap": min(3, len(ids)) ** 2 / len(ids) if ids else None,
        "top1_set_retention": top1_exact / iterations if iterations else None,
        "per_person_rank_interval": {pid: _quantiles(ranks[pid]) for pid in ids},
    }


def _ground_truth(clustering: dict[str, Any], ids: list[str], path: str | Path | None) -> dict[str, Any] | None:
    """Read fixture-only labels here, including the designed mismatch and ceiling."""
    if path is None:
        return None
    truth = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = truth.get("profiles", {})
    missing = sorted(set(ids) - set(profiles))
    if missing:
        return {"kind": "synthetic validation", "status": "missing_ids", "missing_ids": missing, "results": {}}
    true_labels = [profiles[pid]["team"] for pid in ids]
    methods: dict[str, dict[str, Any]] = {}
    for method, labels in sorted(clustering.get("labels", {}).items()):
        if not isinstance(labels, dict) or any(pid not in labels for pid in ids):
            methods[method] = {"status": "missing_labels"}
            continue
        predicted = [labels[pid] for pid in ids]
        methods[method] = {
            "ari": float(adjusted_rand_score(true_labels, predicted)),
            "nmi": float(normalized_mutual_info_score(true_labels, predicted)),
        }
    teams = sorted(set(true_labels) - {"Bridge"})
    bridge_ids = [pid for pid in ids if profiles[pid]["team"] == "Bridge"]
    ceiling = None
    if len(teams) == 3 and bridge_ids:
        merges = []
        for team in teams:
            merged = [team if label == "Bridge" else label for label in true_labels]
            merges.append({
                "bridge_into": team,
                "ari": float(adjusted_rand_score(true_labels, merged)),
                "nmi": float(normalized_mutual_info_score(true_labels, merged)),
            })
        ceiling = {
            "ari": max(row["ari"] for row in merges),
            "nmi": max(row["nmi"] for row in merges),
            "candidate_merges": merges,
        }
    if ceiling is not None:
        for method, labels in sorted(clustering.get("labels", {}).items()):
            if method not in methods or "ari" not in methods[method]:
                continue
            team_clusters = [set(labels[pid] for pid in ids if profiles[pid]["team"] == team) for team in teams]
            methods[method]["recovered_three_teams"] = (
                all(len(clusters) == 1 for clusters in team_clusters)
                and len({next(iter(clusters)) for clusters in team_clusters}) == len(teams)
            )
    mismatch_ids = sorted(set(truth.get("post_skill_mismatch_ids", [])) & set(ids))
    return {
        "kind": "synthetic validation",
        "status": "ok",
        "n": len(ids),
        "truth_label_count": len(set(true_labels)),
        "three_cluster_ceiling": ceiling,
        "post_skill_mismatch_ids": mismatch_ids,
        "results": methods,
        "caveat": "Recovery of structure built into synthetic fixtures does not measure real-world accuracy.",
    }


def validate_analysis(
    bundle: dict[str, Any],
    centrality: dict[str, Any],
    clustering: dict[str, Any],
    config: dict[str, Any],
    ground_truth_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return JSON-safe stability checks for the selected combined graph.

    The centrality argument is accepted for the pipeline's common interface. This
    module recomputes its own three-metric Borda ranking on every random graph so
    baseline and perturbed ranks use exactly the same definition.
    """
    del centrality
    graphs = bundle["graphs"]
    graph = graphs["combined_manual"]
    seed = int(config["seed"])
    baseline, baseline_metrics = _ranking(graph, seed)
    baseline_ranks = baseline_metrics["aggregate_rank"]
    baseline_labels = _communities(graph, seed)
    ids = sorted(graph.nodes)
    return stable_numbers({
        "ranking_definition": "Borda of average ranks for weighted degree, weighted betweenness, and PageRank; top-k includes cutoff ties",
        "perturbation": _perturbation(graph, config, seed, baseline, baseline_ranks),
        "null_models": _null_models(graph, config, seed),
        "ablation": _ablation(graphs, config, seed, baseline_ranks, baseline_labels),
        "weight_sensitivity": _weight_sensitivity(graphs, config, seed, baseline_ranks, baseline_labels),
        "ground_truth": _ground_truth(clustering, ids, ground_truth_path),
    })
