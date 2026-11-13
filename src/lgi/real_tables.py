"""Write the tables for analyses actually run on real-source profiles."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from lgi.report import _table


SECTIONS = ("basic", "experience", "education", "skills", "interests", "activity")


def write_real_tables(
    profiles: list[dict], bundle: dict, centrality: dict, clustering: dict,
    validation: dict, output: Path, *, anonymised: bool, raw_profiles: list[dict],
) -> dict[str, str]:
    """Use the synthetic report's table writer, with real-source values only."""
    tables = output / "tables"
    rendered: dict[str, str] = {}

    def add(name: str, rows: list[dict], columns: list[tuple[str, str]]) -> None:
        rendered[name] = _table(rows, columns, tables / f"{name}.csv")

    text_removed = {
        "basic": any(any(p["basic"][field] for field in ("headline", "location", "summary")) for p in raw_profiles),
        "experience": any(any(job[field] for field in ("title", "location", "description")) for p in raw_profiles for job in p["experience"]),
        "education": any(any(row[field] for field in ("degree", "field")) for p in raw_profiles for row in p["education"]),
        "activity": any(item["text"] for p in raw_profiles for item in p["activity"]),
    }
    count = len(profiles)
    coverage = [
        {
            "section": section,
            "covered": sum(bool(p["coverage"][section]) for p in profiles),
            "profiles": count,
            "fraction": sum(bool(p["coverage"][section]) for p in profiles) / count if count else 0.0,
            "text_removed_by_anonymisation": bool(anonymised and text_removed.get(section, False)),
        }
        for section in SECTIONS
    ]
    add("coverage", coverage, [
        ("section", "Section"), ("covered", "Covered"), ("profiles", "Profiles"),
        ("fraction", "Fraction"), ("text_removed_by_anonymisation", "Text removed by anonymisation"),
    ])
    graph_rows = [
        {
            "layer": name, "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
            "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
            "components": nx.number_connected_components(graph),
            "mean_edge_weight": (
                sum(float(data["weight"]) for _, _, data in graph.edges(data=True)) / graph.number_of_edges()
                if graph.number_of_edges() else None
            ),
        }
        for name, graph in bundle["graphs"].items()
    ]
    add("graph_stats", graph_rows, [
        ("layer", "Layer"), ("nodes", "Nodes"), ("edges", "Edges"), ("density", "Density"),
        ("components", "Components"), ("mean_edge_weight", "Mean edge weight"),
    ])
    names = {p["id"]: p["basic"]["name"] for p in profiles}
    people = []
    for row in centrality["aggregate"]:
        pid = row["id"]
        metrics = centrality["metrics"][pid]
        people.append({
            "rank": row["rank"], "id": pid, "name": names[pid],
            "roles": ", ".join(row["roles"]), "borda": row["borda"], "mrr": row["mrr"],
            "weighted_degree": metrics["weighted_degree"], "betweenness": metrics["betweenness"],
            "pagerank": metrics["pagerank"],
        })
    key_columns = [
        ("rank", "Rank"), ("id", "ID"), ("name", "Person"), ("roles", "Roles"),
        ("borda", "Borda"), ("weighted_degree", "Weighted degree"),
        ("betweenness", "Betweenness"), ("pagerank", "PageRank"),
    ]
    add("key_people", people[:min(5, len(people))], key_columns)
    add("all_people", people, key_columns[:5] + [("mrr", "MRR")] + key_columns[5:])
    pairs = [
        ("degree", "weighted_degree"), ("betweenness", "pagerank"),
        ("pagerank", "eigenvector"), ("constraint", "effective_size"),
        ("closeness", "harmonic"),
    ]
    correlations = centrality["correlations"]
    agreement = [
        {
            "method_a": first, "method_b": second,
            "kendall_tau": correlations["kendall"][first][second],
            "spearman_rho": correlations["spearman"][first][second],
        }
        for first, second in pairs
    ]
    correlation_columns = [
        ("method_a", "Method A"), ("method_b", "Method B"),
        ("kendall_tau", "Kendall tau"), ("spearman_rho", "Spearman rho"),
    ]
    add("method_agreement", agreement, correlation_columns)
    full_correlations = [
        {
            "method_a": first, "method_b": second,
            "kendall_tau": correlations["kendall"][first][second],
            "spearman_rho": correlations["spearman"][first][second],
        }
        for first in correlations["kendall"] for second in correlations["kendall"][first]
    ]
    add("all_centrality_correlations", full_correlations, correlation_columns)
    add("k_selection", clustering["k_selection"]["candidates"], [
        ("k", "k"), ("silhouette", "Silhouette"), ("inertia", "Inertia"),
    ])
    add("clustering_quality", [
        {"method": method, **values} for method, values in clustering["quality"].items()
    ], [
        ("method", "Method"), ("clusters", "Clusters"), ("noise_count", "Noise"),
        ("modularity", "Modularity"), ("silhouette", "Silhouette"),
        ("davies_bouldin", "Davies–Bouldin"),
    ])
    add("clustering_pairwise", clustering["comparisons"], [
        ("method_a", "Method A"), ("method_b", "Method B"), ("ari", "ARI"), ("nmi", "NMI"),
    ])
    perturb = validation["perturbation"]
    sensitivity = validation["weight_sensitivity"]
    stability = [
        {
            "check": "Edge perturbation", "draws": perturb["iterations"],
            "top3_retention": perturb["top3_set_retention"],
            "top3_mean_overlap": perturb["top3_mean_overlap"],
            "top3_chance_overlap": perturb["top3_chance_overlap"],
            "top1_retention": perturb["top1_set_retention"],
            "kendall_tau_mean": perturb["kendall_tau_mean"],
            "kendall_tau_median": perturb["kendall_tau_interval"][1] if perturb["kendall_tau_interval"] else None,
            "cluster_ari_median": None,
        },
        {
            "check": "Weight sensitivity", "draws": sensitivity["iterations"],
            "top3_retention": sensitivity["top3_set_retention"],
            "top3_mean_overlap": sensitivity["top3_mean_overlap"],
            "top3_chance_overlap": sensitivity["top3_chance_overlap"],
            "top1_retention": sensitivity["top1_set_retention"],
            "kendall_tau_mean": sensitivity["kendall_tau_mean"],
            "kendall_tau_median": sensitivity["kendall_tau_interval"][1] if sensitivity["kendall_tau_interval"] else None,
            "cluster_ari_median": sensitivity["cluster_ari_interval"][1] if sensitivity["cluster_ari_interval"] else None,
        },
    ]
    add("stability_summary", stability, [
        ("check", "Check"), ("draws", "Draws"), ("top3_retention", "Exact top-three retention"),
        ("top3_mean_overlap", "Mean top-three overlap"), ("top3_chance_overlap", "Chance overlap"),
        ("top1_retention", "Top-one retention"), ("kendall_tau_mean", "Kendall tau mean"),
        ("kendall_tau_median", "Kendall tau median"), ("cluster_ari_median", "Cluster ARI median"),
    ])
    key_ranks = {row["id"]: row["rank"] for row in people}
    rank_intervals = []
    for pid, values in perturb["per_person"].items():
        interval = values["rank_interval"]
        rank_intervals.append({
            "id": pid, "key_people_rank": key_ranks[pid],
            "baseline_rank": values["baseline_rank"],
            "rank_p05": interval[0] if interval else None,
            "rank_p95": interval[2] if interval else None,
            "rank_min": values["rank_min"], "rank_max": values["rank_max"],
            "top3_frequency": values["top3_frequency"],
        })
    add("rank_intervals", rank_intervals, [
        ("id", "ID"), ("key_people_rank", "Key-people rank"), ("baseline_rank", "Stability rank"),
        ("rank_p05", "Rank 5th pct"), ("rank_p95", "Rank 95th pct"),
        ("rank_min", "Rank min"), ("rank_max", "Rank max"),
        ("top3_frequency", "Top-three frequency"),
    ])
    nulls = validation["null_models"]
    null_rows = [
        {"model": model, "metric": metric, **values, "samples": nulls[model]["samples"]}
        for model in ("degree_preserving", "gnm")
        for metric, values in nulls[model]["statistics"].items()
    ]
    add("null_models", null_rows, [
        ("model", "Null model"), ("metric", "Statistic"), ("samples", "Samples"),
        ("observed", "Observed"), ("null_mean", "Null mean"),
        ("z_score", "z"), ("empirical_p_upper", "Empirical upper p"),
    ])
    add("ablation", [
        {"layer": layer, **values} for layer, values in validation["ablation"].items()
    ], [
        ("layer", "Dropped layer"), ("remaining_edges", "Edges left"),
        ("rank_kendall_tau", "Rank Kendall tau"), ("cluster_ari", "Cluster ARI"),
        ("top3_set_retained", "Top-three retained"),
    ])
    return rendered
