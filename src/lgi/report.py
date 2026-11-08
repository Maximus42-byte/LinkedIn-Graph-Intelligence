"""Generate an evidence-backed Markdown report and its inspectable CSV tables."""

from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx
import numpy as np
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
        writer = csv.DictWriter(handle, fieldnames=[key for key, _ in columns], extrasaction="ignore", lineterminator="\n")
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


def _partition_signature(labels: dict[str, int]) -> tuple[tuple[str, ...], ...]:
    """Compare membership without depending on arbitrary cluster numbers."""
    groups: dict[int, list[str]] = {}
    for pid, label in labels.items():
        groups.setdefault(label, []).append(pid)
    return tuple(sorted(tuple(sorted(members)) for members in groups.values()))


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
    ignored = bundle["ignored_employers"]
    ignored_text = ", ".join(f"{row['company']} ({row['profiles']}/{n} profiles)" for row in ignored) or "none"
    lines.append(f"The org layer ignores employers present in more than **{config['org_ignore_employer_share']:.0%}** of profiles; ignored: **{ignored_text}**. Shared past employers still contribute when dates overlap.")
    lines.append(f"The skill and semantic layers keep each profile's **{config['similarity_knn_k']}** highest-scoring neighbours with positive weight at least **{config['thresholds']['skill']:.3f}** and **{config['thresholds']['semantic']:.3f}**, respectively. An edge survives if either endpoint selects the other; equal scores are broken by profile ID.")
    lines.append(f"The configured reference month **{config['reference_month']}** is the inclusive end month for open-ended employment in graph overlap and career tenure; it is an analysis cutoff, not the latest activity date.")
    lines += ["", "## Key people and roles", ""]
    key_rows = []
    for row in centrality["aggregate"]:
        pid = row["id"]
        metric = centrality["metrics"][pid]
        key_rows.append({"rank": row["rank"], "id": pid, "name": names[pid], "roles": ", ".join(row["roles"]), "borda": row["borda"], "mrr": row["mrr"], "weighted_degree": metric["weighted_degree"], "betweenness": metric["betweenness"], "pagerank": metric["pagerank"]})
    lines.append(_table(key_rows[:min(5, n)], [("rank", "Rank"), ("id", "ID"), ("name", "Person"), ("roles", "Roles"), ("borda", "Borda"), ("weighted_degree", "Weighted degree"), ("betweenness", "Betweenness"), ("pagerank", "PageRank")], tables / "key_people.csv"))
    _table(key_rows, [("rank", "Rank"), ("id", "ID"), ("name", "Person"), ("roles", "Roles"), ("borda", "Borda"), ("mrr", "MRR"), ("weighted_degree", "Weighted degree"), ("betweenness", "Betweenness"), ("pagerank", "PageRank")], tables / "all_people.csv")
    lines.append("Borda and mean reciprocal rank aggregate all centrality methods. Hub labels use weighted-degree quantiles; bridge and influencer labels use their configured metric quantiles.")
    if len(key_rows) >= 5:
        weighted_degrees = np.array([row["weighted_degree"] for row in key_rows], dtype=float)
        mean_degree = float(np.mean(weighted_degrees))
        degree_cv = float(np.std(weighted_degrees) / mean_degree) if mean_degree else None
        degree_gap = abs(key_rows[0]["weighted_degree"] - key_rows[4]["weighted_degree"])
        lines.append(f"Weighted degree has coefficient of variation **{_fmt(degree_cv)}**; the absolute weighted-degree gap between key-people ranks **{key_rows[0]['rank']}** and **{key_rows[4]['rank']}** is **{degree_gap:.3f}**. These near-ties are consistent with unstable ranks, but do not establish a cause.")
    lines += ["", "## Method comparison", ""]
    pairs = [("degree", "weighted_degree"), ("betweenness", "pagerank"), ("pagerank", "eigenvector"), ("constraint", "effective_size"), ("closeness", "harmonic")]
    agreement = [{"method_a": a, "method_b": b, "kendall_tau": centrality["correlations"]["kendall"][a][b], "spearman_rho": centrality["correlations"]["spearman"][a][b]} for a, b in pairs]
    lines.append(_table(agreement, [("method_a", "Method A"), ("method_b", "Method B"), ("kendall_tau", "Kendall tau"), ("spearman_rho", "Spearman rho")], tables / "method_agreement.csv"))
    constraint_tau = centrality["correlations"]["kendall"]["constraint"]["effective_size"]
    lines.append(f"Constraint and effective size have Kendall tau **{_fmt(constraint_tau)}**. A negative association is expected because greater constraint corresponds to fewer effective nonredundant contacts.")
    if constraint_tau is not None and constraint_tau < 0:
        lines.append("Their negative tau here is not method disagreement.")
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
    partition_groups: dict[tuple[tuple[str, ...], ...], list[str]] = {}
    for method, labels in clustering["labels"].items():
        partition_groups.setdefault(_partition_signature(labels), []).append(method)
    if partition_groups:
        agreeing = max(partition_groups.values(), key=lambda methods: (len(methods), sorted(methods)))
        lines.append(f"**{len(agreeing)} of {len(clustering['labels'])}** clustering methods produced an identical partition ({', '.join(sorted(agreeing))}). Agreement on these clean synthetic fixtures shows shared recovery of designed structure; it does not establish accuracy on real profiles.")
    hdbscan_labels = clustering["labels"].get("hdbscan")
    embedding_labels = clustering["labels"].get("embedding_kmeans")
    if hdbscan_labels is not None and embedding_labels is not None:
        if _partition_signature(hdbscan_labels) == _partition_signature(embedding_labels):
            lines.append("On this data, sklearn HDBSCAN returned the same partition as embedding KMeans. The HDBSCAN labels are its own estimator output, with no substituted fallback.")
        else:
            lines.append("On this data, sklearn HDBSCAN returned a different partition from embedding KMeans. The HDBSCAN labels are its own estimator output, with no substituted fallback.")
    if graph.number_of_edges() and graph.number_of_nodes() <= int(config["girvan_newman_max_nodes"]):
        lines.append("Girvan–Newman selected candidate cuts using NetworkX's default unweighted edge betweenness. Its cut modularity was scored with edge weights on the same combined graph used to score the other graph methods.")
    else:
        lines.append("Girvan–Newman did not run on this graph; its output is the documented fallback. All reported graph modularities use weights on the combined graph.")
    if clustering["warnings"]:
        lines.append("Clustering warnings: " + "; ".join(clustering["warnings"]))
    lines += ["", "## Validation and stability", ""]
    sensitivity = validation["weight_sensitivity"]
    stability_rows = [
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
    lines.append(_table(stability_rows, [
        ("check", "Check"), ("draws", "Draws"),
        ("top3_retention", "Exact top-three retention"),
        ("top3_mean_overlap", "Mean top-three overlap"),
        ("top3_chance_overlap", "Chance overlap"),
        ("top1_retention", "Top-one retention"),
        ("kendall_tau_mean", "Kendall tau mean"),
        ("kendall_tau_median", "Kendall tau median"),
        ("cluster_ari_median", "Cluster ARI median"),
    ], tables / "stability_summary.csv"))
    lines.append(f"Mean overlap counts shared members with the baseline top-three set; the nominal chance level is k×k/n = **{perturb['top3_chance_overlap']:.3f}** for k=**{min(3, n)}** among n=**{n}** profiles when each set has exactly k members. Cutoff ties are included in the observed sets. The overlap and top-one measures were added after seeing low exact top-three retention; exact retention remains the primary metric.")
    lines.append(f"Edge perturbation removed **{perturb['removed_edges_per_iteration']}** edges and added **{perturb['added_edges_per_iteration']}** per draw. Its mean Kendall tau was **{perturb['kendall_tau_mean']:.3f}** with a central interval of **{_interval(perturb['kendall_tau_interval'])}**. The stability ranking uses {validation['ranking_definition']}.")
    lines.append("The two rankings can differ because the key-people table aggregates all centrality methods while the stability ranking aggregates only weighted degree, weighted betweenness, and PageRank.")
    key_ranks = {row["id"]: row["rank"] for row in key_rows}
    rank_rows = []
    for pid, values in perturb["per_person"].items():
        interval = values["rank_interval"]
        rank_rows.append({
            "id": pid,
            "key_people_rank": key_ranks[pid],
            "baseline_rank": values["baseline_rank"],
            "rank_p05": interval[0] if interval else None,
            "rank_p95": interval[2] if interval else None,
            "rank_min": values["rank_min"],
            "rank_max": values["rank_max"],
            "top3_frequency": values["top3_frequency"],
        })
    lines.append(_table(rank_rows, [
        ("id", "ID"), ("key_people_rank", "Key-people rank"),
        ("baseline_rank", "Stability rank"),
        ("rank_p05", "Rank 5th pct"), ("rank_p95", "Rank 95th pct"),
        ("rank_min", "Rank min"), ("rank_max", "Rank max"),
        ("top3_frequency", "Top-three frequency"),
    ], tables / "rank_intervals.csv"))
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
        ceiling = truth.get("three_cluster_ceiling")
        truth_rows = []
        for method, values in truth["results"].items():
            truth_rows.append({
                "method": method,
                "ari": values.get("ari"),
                "ari_ceiling": ceiling["ari"] if ceiling else None,
                "nmi": values.get("nmi"),
                "nmi_ceiling": ceiling["nmi"] if ceiling else None,
            })
        lines.append("**Synthetic validation:** comparison with labels used to construct the fixtures. It measures recovery of built-in structure, not real-world accuracy.")
        lines.append(_table(truth_rows, [
            ("method", "Method"), ("ari", "ARI"),
            ("ari_ceiling", "Three-cluster ARI ceiling"),
            ("nmi", "NMI"), ("nmi_ceiling", "Three-cluster NMI ceiling"),
        ], tables / "synthetic_ground_truth.csv"))
        if ceiling:
            three_cluster_methods = sum(row["clusters"] == len(ceiling["candidate_merges"]) for row in clustering["quality"].values())
            lines.append(f"The fixture truth has **{truth['truth_label_count']}** labels, while **{three_cluster_methods}** methods returned **{len(ceiling['candidate_merges'])}** clusters. The three-cluster ceiling is the best ARI **{ceiling['ari']:.3f}** and NMI **{ceiling['nmi']:.3f}** across the **{len(ceiling['candidate_merges'])}** possible Bridge-to-team merges.")
            at_ceiling = []
            below_ceiling = []
            for method, values in truth["results"].items():
                if "ari" not in values:
                    continue
                if abs(values["ari"] - ceiling["ari"]) < 1e-9 and abs(values["nmi"] - ceiling["nmi"]) < 1e-9:
                    at_ceiling.append(method)
                elif values["ari"] < ceiling["ari"] - 1e-9 or values["nmi"] < ceiling["nmi"] - 1e-9:
                    below_ceiling.append(method)
            if at_ceiling:
                recovered = all(truth["results"][method].get("recovered_three_teams") for method in at_ceiling)
                recovery = "recovered all three designed teams" if recovered else "did not all recover the three designed teams"
                lines.append(f"Methods at that ceiling ({', '.join(at_ceiling)}) {recovery}. Methods below it: {', '.join(below_ceiling) if below_ceiling else 'none'}.")
            else:
                lines.append(f"No method reached the three-cluster ceiling. Methods below it: {', '.join(below_ceiling) if below_ceiling else 'none'}.")
    lines += ["", "## Career, education, activity, and similarity findings", ""]
    career = analysis["career"]
    bus_count = len(career["bus_factor_one_skills"])
    skill_noun = "skill" if bus_count == 1 else "skills"
    holder_verb = "has" if bus_count == 1 else "have"
    lines.append(f"Dated job history yielded **{len(career['movement_graph']['edges'])}** company-to-company routes at observation month **{_fmt(career['as_of_month'])}**. Exactly **{bus_count}** recorded {skill_noun} {holder_verb} one holder.")
    lines.append(_table(career["talent_sources"], [("company", "Talent source"), ("outbound_people", "Outbound people"), ("outbound_transitions", "Transitions")], tables / "talent_sources.csv"))
    lines.append(_table(career["bus_factor_one_skills"], [("skill", "Single-holder skill"), ("person_id", "Holder")], tables / "bus_factor.csv"))
    _table(career["skill_cooccurrence"], [("skill_a", "Skill A"), ("skill_b", "Skill B"), ("people_count", "People")], tables / "skill_cooccurrence.csv")
    _table(career["skill_gap_candidates"], [("id", "ID"), ("missing_common_skills", "Missing common recorded skills")], tables / "skill_gap_candidates.csv")
    activity = analysis["activity"]
    activity_rows = activity["people"]
    lines.append(_table(activity_rows, [("id", "ID"), ("post_count", "Posts"), ("reaction_count", "Reactions"), ("engagement_weighted_degree", "Engagement degree"), ("topic_consistency", "Topic consistency"), ("tfidf_consistency", "Lexical consistency")], tables / "activity_people.csv"))
    comparison = activity["graph_comparison"]
    lines.append(f"Organisational and engagement degree have Spearman rho **{_fmt(comparison['org_degree_vs_engagement_degree']['spearman_rho'])}**; their edge sets have Jaccard **{_fmt(comparison['org_engagement_edge_jaccard'])}**. Topic and lexical consistency are both shown because their judgments can differ.")
    mismatch_ids = truth.get("post_skill_mismatch_ids", []) if truth else []
    if mismatch_ids:
        lines.append(f"The fixture generator marked **{', '.join(mismatch_ids)}** as the designed post/skill mismatch. The flag rule below selects the lowest available score for each measure, including ties.")
        for score_key, label in (("topic_consistency", "Topic consistency"), ("tfidf_consistency", "Lexical consistency")):
            scored = sorted(
                (row for row in activity_rows if row[score_key] is not None),
                key=lambda row: (row[score_key], row["id"]),
            )
            if not scored:
                lines.append(f"{label} had no available scores, so it could not flag the designed mismatch.")
                continue
            lowest_score = scored[0][score_key]
            lowest_ids = {row["id"] for row in scored if row[score_key] == lowest_score}
            caught = sorted(lowest_ids & set(mismatch_ids))
            missed = sorted(set(mismatch_ids) - lowest_ids)
            low_profiles = ", ".join(
                f"{names[row['id']]} ({row['id']}, {row[score_key]:.3f})"
                for row in scored[:min(3, len(scored))]
            )
            lines.append(f"{label} flags {', '.join(caught) if caught else 'none'} and misses {', '.join(missed) if missed else 'none'} of the designed mismatch profiles. Lowest scores: {low_profiles}.")
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
