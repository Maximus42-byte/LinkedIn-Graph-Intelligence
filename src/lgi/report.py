"""Generate an evidence-backed Markdown report and its inspectable CSV tables."""

from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx
from scipy.stats import kendalltau

from lgi.analysis.centrality import analyze_centrality


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "—"
    return str(value)


def _table(rows: list[dict], columns: list[tuple[str, str]], path: Path) -> str:
    """The report table and its CSV are written from the same rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key, _ in columns], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = [_fmt(row.get(key)).replace("|", "\\|").replace("\n", " ") for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _interval(values: list[float] | None) -> str:
    return "—" if values is None else f"{values[0]:.2f}–{values[2]:.2f}"


def _tau(first: list[str], second: list[str]) -> float | None:
    if len(first) < 2:
        return None
    position = {pid: index for index, pid in enumerate(second)}
    value = kendalltau(range(len(first)), [position[pid] for pid in first]).statistic
    return float(value) if value == value else None


def write_report(profiles: list[dict], bundle: dict, centrality: dict, clustering: dict, validation: dict, analysis: dict, config: dict, output_dir: str | Path) -> Path:
    """Every reported value derives from the supplied pipeline results."""
    output = Path(output_dir)
    tables = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    n = len(profiles)
    source = profiles[0]["source"] if profiles else "none"
    synthetic = source == "fixture"
    graph = bundle["graphs"]["combined_manual"]
    ids = sorted(p["id"] for p in profiles)
    names = {p["id"]: p["basic"]["name"] for p in profiles}
    coverage = [{"section": section, "covered": sum(p["coverage"][section] for p in profiles), "profiles": n, "fraction": sum(p["coverage"][section] for p in profiles) / n if n else 0.0} for section in ("basic", "experience", "education", "skills", "interests", "activity")]
    complete = sum(all(p["coverage"].values()) for p in profiles)
    lines = ["# Profile graph analysis report", "", "## Executive summary", ""]
    lines.append(f"Data source: **{'synthetic fixtures' if synthetic else source}**. The run processed **{n} profiles**; **{complete}** had complete coverage of the {len(coverage)} schema sections. The selected combined graph has **{graph.number_of_edges()} edges**.")
    perturb = validation["perturbation"]
    top3 = perturb["top3_set_retention"]
    lines.append(f"The top-three set survived **{top3:.1%}** of **{perturb['iterations']}** edge perturbations. This is a stability measure of the computed ranking, not a real-world accuracy estimate.")
    if n <= 10:
        lines.append(f"With n={n}, these results are illustrative and not statistically meaningful.")
    elif synthetic:
        lines.append("These synthetic results demonstrate pipeline behavior and are not estimates of real-world accuracy.")
    lines += ["", "## Data and coverage", ""]
    lines.append("All fixture pages carry a visible SYNTHETIC banner and a synthetic meta tag." if synthetic else "Coverage reflects only fields explicitly supplied by the source.")
    lines.append(_table(coverage, [("section", "Section"), ("covered", "Covered"), ("profiles", "Profiles"), ("fraction", "Fraction")], tables / "coverage.csv"))
    lines += ["", "## Graph statistics", ""]
    graph_rows = []
    for name, layer in bundle["graphs"].items():
        graph_rows.append({"layer": name, "nodes": layer.number_of_nodes(), "edges": layer.number_of_edges(), "density": nx.density(layer) if layer.number_of_nodes() > 1 else 0.0, "components": nx.number_connected_components(layer), "mean_edge_weight": sum(data["weight"] for _, _, data in layer.edges(data=True)) / layer.number_of_edges() if layer.number_of_edges() else None})
    lines.append(_table(graph_rows, [("layer", "Layer"), ("nodes", "Nodes"), ("edges", "Edges"), ("density", "Density"), ("components", "Components"), ("mean_edge_weight", "Mean edge weight")], tables / "graph_stats.csv"))
    lines.append(f"Semantic vectors used **{bundle['embedding_method']}**. Skill edges retain both Jaccard and TF-IDF cosine; semantic edges retain embedding and TF-IDF cosine alternatives. The chosen skill metric is **{config['skill_metric']}** and semantic metric is **{config['semantic_metric']}**.")
    lines += ["", "## Key people and roles", ""]
    key_rows = []
    for row in centrality["aggregate"]:
        pid = row["id"]
        metric = centrality["metrics"][pid]
        key_rows.append({"rank": row["rank"], "id": pid, "name": names[pid], "roles": ", ".join(row["roles"]), "borda": row["borda"], "mrr": row["mrr"], "weighted_degree": metric["weighted_degree"], "betweenness": metric["betweenness"], "pagerank": metric["pagerank"]})
    lines.append(_table(key_rows[:min(5, n)], [("rank", "Rank"), ("id", "ID"), ("name", "Person"), ("roles", "Roles"), ("borda", "Borda"), ("weighted_degree", "Weighted degree"), ("betweenness", "Betweenness"), ("pagerank", "PageRank")], tables / "key_people.csv"))
    _table(key_rows, [("rank", "Rank"), ("id", "ID"), ("name", "Person"), ("roles", "Roles"), ("borda", "Borda"), ("mrr", "MRR"), ("weighted_degree", "Weighted degree"), ("betweenness", "Betweenness"), ("pagerank", "PageRank")], tables / "all_people.csv")
    lines.append("Borda and mean reciprocal rank aggregate all centrality methods. Roles use configurable quantiles; weighted degree breaks hub ties in a dense graph.")
    lines += ["", "## Method comparison", ""]
    pairs = [("degree", "weighted_degree"), ("betweenness", "pagerank"), ("pagerank", "eigenvector"), ("constraint", "effective_size"), ("closeness", "harmonic")]
    agreement = [{"method_a": a, "method_b": b, "kendall_tau": centrality["correlations"]["kendall"][a][b], "spearman_rho": centrality["correlations"]["spearman"][a][b]} for a, b in pairs]
    lines.append(_table(agreement, [("method_a", "Method A"), ("method_b", "Method B"), ("kendall_tau", "Kendall tau"), ("spearman_rho", "Spearman rho")], tables / "method_agreement.csv"))
    full_correlation = [{"method_a": a, "method_b": b, "kendall_tau": centrality["correlations"]["kendall"][a][b], "spearman_rho": centrality["correlations"]["spearman"][a][b]} for a in centrality["correlations"]["kendall"] for b in centrality["correlations"]["kendall"][a]]
    _table(full_correlation, [("method_a", "Method A"), ("method_b", "Method B"), ("kendall_tau", "Kendall tau"), ("spearman_rho", "Spearman rho")], tables / "all_centrality_correlations.csv")
    equal_ranking = analyze_centrality(bundle["graphs"]["combined_equal"], config)["rankings"]["borda"]
    lines.append(f"Manual-weight and equal-normalised graph rankings have Kendall tau **{_fmt(_tau(centrality['rankings']['borda'], equal_ranking))}**. The two graph exports are retained in `graph.json`.")
    lines += ["", "## Clustering results", ""]
    k = clustering["k_selection"]
    lines.append(f"Feature k selection chose **{k['chosen_k']}** groups; silhouette choice **{_fmt(k['silhouette_choice'])}**, elbow choice **{_fmt(k['elbow_choice'])}**.")
    lines.append(_table(k["candidates"], [("k", "k"), ("silhouette", "Silhouette"), ("inertia", "Inertia")], tables / "k_selection.csv"))
    quality_rows = [{"method": method, **values} for method, values in clustering["quality"].items()]
    lines.append(_table(quality_rows, [("method", "Method"), ("clusters", "Clusters"), ("noise_count", "Noise"), ("modularity", "Modularity"), ("silhouette", "Silhouette"), ("davies_bouldin", "Davies–Bouldin")], tables / "clustering_quality.csv"))
    _table(clustering["comparisons"], [("method_a", "Method A"), ("method_b", "Method B"), ("ari", "ARI"), ("nmi", "NMI")], tables / "clustering_pairwise.csv")
    lines.append(f"All pairwise clustering ARI/NMI scores are in `tables/clustering_pairwise.csv`. The skill projection yielded **{len(clustering['skill_families'])}** skill families. Feature embeddings used **{clustering['embedding_method']}**.")
    if clustering["warnings"]:
        lines.append("Clustering warnings: " + "; ".join(clustering["warnings"]))
    lines += ["", "## Validation and stability", ""]
    stability_rows = [{"check": "Edge perturbation", "draws": perturb["iterations"], "top3_retention": perturb["top3_set_retention"], "kendall_tau": perturb["kendall_tau_mean"], "cluster_ari": None}, {"check": "Weight sensitivity", "draws": validation["weight_sensitivity"]["iterations"], "top3_retention": validation["weight_sensitivity"]["top3_set_retention"], "kendall_tau": validation["weight_sensitivity"]["kendall_tau_interval"][1] if validation["weight_sensitivity"]["kendall_tau_interval"] else None, "cluster_ari": validation["weight_sensitivity"]["cluster_ari_interval"][1] if validation["weight_sensitivity"]["cluster_ari_interval"] else None}]
    lines.append(_table(stability_rows, [("check", "Check"), ("draws", "Draws"), ("top3_retention", "Top-three retention"), ("kendall_tau", "Kendall tau (mean/median)"), ("cluster_ari", "Cluster ARI median")], tables / "stability_summary.csv"))
    lines.append(f"Edge perturbation removed **{perturb['removed_edges_per_iteration']}** edges and added **{perturb['added_edges_per_iteration']}** per draw. Its mean Kendall tau was **{perturb['kendall_tau_mean']:.3f}** with a central interval of **{_interval(perturb['kendall_tau_interval'])}**.")
    rank_rows = [{"id": pid, **values, "rank_low": values["rank_interval"][0] if values["rank_interval"] else None, "rank_high": values["rank_interval"][2] if values["rank_interval"] else None} for pid, values in perturb["per_person"].items()]
    lines.append(_table(rank_rows, [("id", "ID"), ("baseline_rank", "Baseline rank"), ("rank_low", "Rank low"), ("rank_high", "Rank high"), ("top3_frequency", "Top-three frequency")], tables / "rank_intervals.csv"))
    null = validation["null_models"]
    null_rows = []
    for model in ("degree_preserving", "gnm"):
        for metric, values in null[model]["statistics"].items():
            null_rows.append({"model": model, "metric": metric, **values, "samples": null[model]["samples"]})
    lines.append(_table(null_rows, [("model", "Null model"), ("metric", "Statistic"), ("samples", "Samples"), ("observed", "Observed"), ("null_mean", "Null mean"), ("z_score", "z"), ("empirical_p_upper", "Empirical upper p")], tables / "null_models.csv"))
    lines.append(null["caveat"])
    if null["backbone"]["used"]:
        lines.append(f"The combined graph was too dense for useful edge swaps, so the null comparison used a backbone containing the **{null['backbone']['edges']} strongest edges**; this changes the tested graph.")
    ablation_rows = [{"layer": layer, **values} for layer, values in validation["ablation"].items()]
    lines.append(_table(ablation_rows, [("layer", "Dropped layer"), ("remaining_edges", "Edges left"), ("rank_kendall_tau", "Rank Kendall tau"), ("cluster_ari", "Cluster ARI"), ("top3_set_retained", "Top-three retained")], tables / "ablation.csv"))
    lines.append(f"Dirichlet weight draws gave a rank Kendall tau interval of **{_interval(validation['weight_sensitivity']['kendall_tau_interval'])}** and a cluster ARI interval of **{_interval(validation['weight_sensitivity']['cluster_ari_interval'])}**.")
    truth = validation["ground_truth"]
    if truth:
        truth_rows = [{"method": method, **values} for method, values in truth["results"].items()]
        lines.append("**Synthetic validation:** comparison with labels used to construct the fixtures. It measures recovery of built-in structure, not real-world accuracy.")
        lines.append(_table(truth_rows, [("method", "Method"), ("ari", "ARI"), ("nmi", "NMI")], tables / "synthetic_ground_truth.csv"))
    lines += ["", "## Career, education, activity, and similarity findings", ""]
    career = analysis["career"]
    lines.append(f"Dated job history yielded **{len(career['movement_graph']['edges'])}** company-to-company routes at observation month **{_fmt(career['as_of_month'])}**. Exactly **{len(career['bus_factor_one_skills'])}** recorded skill{"s" if len(career["bus_factor_one_skills"]) != 1 else ""} ha{"ve" if len(career["bus_factor_one_skills"]) != 1 else "s"} one holder.")
    lines.append(_table(career["talent_sources"], [("company", "Talent source"), ("outbound_people", "Outbound people"), ("outbound_transitions", "Transitions")], tables / "talent_sources.csv"))
    lines.append(_table(career["bus_factor_one_skills"], [("skill", "Single-holder skill"), ("person_id", "Holder")], tables / "bus_factor.csv"))
    _table(career["skill_cooccurrence"], [("skill_a", "Skill A"), ("skill_b", "Skill B"), ("people_count", "People")], tables / "skill_cooccurrence.csv")
    _table(career["skill_gap_candidates"], [("id", "ID"), ("missing_common_skills", "Missing common recorded skills")], tables / "skill_gap_candidates.csv")
    activity = analysis["activity"]
    activity_rows = activity["people"]
    lines.append(_table(activity_rows, [("id", "ID"), ("post_count", "Posts"), ("reaction_count", "Reactions"), ("engagement_weighted_degree", "Engagement degree"), ("topic_consistency", "Topic consistency"), ("tfidf_consistency", "Lexical consistency")], tables / "activity_people.csv"))
    comparison = activity["graph_comparison"]
    lines.append(f"Organisational and engagement degree have Spearman rho **{_fmt(comparison['org_degree_vs_engagement_degree']['spearman_rho'])}**; their edge sets have Jaccard **{_fmt(comparison['org_engagement_edge_jaccard'])}**. Topic and lexical consistency are both shown because their judgments can differ.")
    topic_rows = [{"topic_id": row["topic_id"], "top_terms": row["top_terms"]} for row in activity["topics"]]
    lines.append(_table(topic_rows, [("topic_id", "Post topic"), ("top_terms", "Top terms")], tables / "activity_topics.csv"))
    similarity = analysis["similarity"]
    nearest = [{"id": row["id"], "neighbour_ids": row["neighbour_ids"], "similarities": row["similarities"]} for row in similarity["neighbours"]]
    lines.append(f"The positive-similarity nearest-colleague graph used k=**{similarity['knn_k']}** and contains **{len(similarity['knn_graph']['edges'])} edges**. Its cosine and Jaccard matrices are visualised in `similarity_heatmap.png`.")
    lines.append(_table(nearest, [("id", "ID"), ("neighbour_ids", "Nearest colleagues"), ("similarities", "Cosine values")], tables / "nearest_colleagues.csv"))
    lines += ["", "## Limitations", ""]
    lines.append(f"The graph contains **{n}** observed profiles. At this size, method rankings, modularity, p-values, and role labels can change sharply after a few edges change. The perturbation and weight-sensitivity tables quantify that dependence.")
    if synthetic:
        lines.append("All people, organisations, schools, jobs, skills, and posts are synthetic. Ground-truth agreement reflects designed fixture structure and does not validate real-person predictions.")
    lines.append("An edge means observed overlap or interaction under the configured rule; it does not prove a real relationship. Export and live sources can leave fields uncovered, and missing skills or activity must not be treated as absence in reality.")
    lines.append("Optional sentence-transformers vectors require an already available local model. The default run uses the offline fallback.")
    report_path = output / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
