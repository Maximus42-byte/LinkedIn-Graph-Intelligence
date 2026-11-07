"""Write deterministic, readable figures from computed pipeline results."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from shutil import copyfile
from typing import Any

import matplotlib

matplotlib.use("Agg")  # The offline pipeline also runs on machines without a display.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_extraction.text import TfidfVectorizer


def _names(profiles: list[dict], ids: list[str]) -> dict[str, str]:
    """Keep IDs visible so duplicate or missing display names remain unambiguous."""
    by_id = {str(profile["id"]): profile for profile in profiles}
    result = {}
    for pid in ids:
        name = str(by_id.get(pid, {}).get("basic", {}).get("name", "")).strip()
        result[pid] = f"{pid}  {name}" if name else pid
    return result


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _empty(ax: plt.Axes, reason: str) -> None:
    """An unavailable plot is explicit instead of silently fabricating points."""
    ax.text(0.5, 0.5, reason, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _display_edges(graph: nx.Graph, limit_per_person: int = 3) -> nx.Graph:
    """A large graph needs a strongest-ties view to avoid a solid block of ink."""
    if graph.number_of_nodes() <= 40:
        return graph
    chosen = nx.Graph()
    chosen.add_nodes_from(graph.nodes(data=True))
    for pid in sorted(graph.nodes):
        neighbours = sorted(
            graph[pid].items(),
            key=lambda item: (-float(item[1].get("weight", 1.0)), str(item[0])),
        )[:limit_per_person]
        for neighbour, attributes in neighbours:
            chosen.add_edge(pid, neighbour, **attributes)
    return chosen


def _graph_plot(
    profiles: list[dict], bundle: dict, centrality: dict, clustering: dict,
    config: dict, path: Path,
) -> None:
    graph = _display_edges(bundle["graphs"]["combined_manual"])
    ids = sorted(graph.nodes)
    names = _names(profiles, ids)
    communities = clustering.get("labels", {}).get("louvain", {})
    pagerank = {pid: float(centrality.get("metrics", {}).get(pid, {}).get("pagerank", 0.0)) for pid in ids}
    peak = max(pagerank.values(), default=0.0)
    sizes = [250 + 1650 * pagerank[pid] / peak if peak else 350 for pid in ids]
    palette = plt.get_cmap("tab20")
    cluster_ids = sorted(set(communities.get(pid, -1) for pid in ids))
    colors = {cluster: "#a0a0a0" if cluster == -1 else palette(index % 20)
              for index, cluster in enumerate(cluster_ids)}
    node_colors = [colors[communities.get(pid, -1)] for pid in ids]
    fig, ax = plt.subplots(figsize=(11, 8))
    if ids:
        positions = nx.spring_layout(graph, seed=int(config["seed"]), weight="weight", iterations=70)
        edges = list(graph.edges(data=True))
        if edges:
            maximum = max(float(data.get("weight", 1.0)) for _, _, data in edges)
            widths = [0.4 + 2.2 * float(data.get("weight", 1.0)) / maximum for _, _, data in edges]
            nx.draw_networkx_edges(graph, positions, ax=ax, width=widths, alpha=0.22, edge_color="#455a64")
        nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=ids, node_size=sizes,
                               node_color=node_colors, edgecolors="#263238", linewidths=0.7)
        label_ids = ids if len(ids) <= 30 else [row["id"] for row in centrality.get("aggregate", [])[:20]]
        nx.draw_networkx_labels(graph, positions, labels={pid: pid for pid in label_ids if pid in graph},
                                ax=ax, font_size=8 if len(ids) <= 30 else 7)
        if len(cluster_ids) <= 12:
            handles = [Line2D([0], [0], marker="o", linestyle="", color="none",
                              markerfacecolor=colors[cluster], markeredgecolor="#263238",
                              label=f"Community {cluster}" if cluster != -1 else "Unassigned")
                       for cluster in cluster_ids]
            ax.legend(handles=handles, title="Louvain", loc="upper left", fontsize=8, framealpha=0.9)
    else:
        _empty(ax, "No profiles available")
    subtitle = "Strongest 3 ties per person shown" if bundle["graphs"]["combined_manual"].number_of_nodes() > 40 else "All positive ties shown"
    ax.set_title(f"Combined relationship graph — {subtitle}\nColor: Louvain community; size: PageRank")
    ax.axis("off")
    _save(fig, path)


def _similarity_plot(profiles: list[dict], analysis: dict, path: Path) -> None:
    similarity = analysis.get("similarity", {})
    ids = similarity.get("ids", [])
    names = _names(profiles, ids)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    for ax, key, title in zip(axes, ("cosine_matrix", "jaccard_matrix"),
                              ("TF-IDF cosine: profile text", "Jaccard: declared skills")):
        matrix = np.asarray(similarity.get(key, []), dtype=float)
        if len(ids) and matrix.shape == (len(ids), len(ids)):
            image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            stride = max(1, ceil(len(ids) / 25))
            ticks = list(range(0, len(ids), stride))
            labels = [names[ids[index]] if len(ids) <= 20 else ids[index] for index in ticks]
            ax.set_xticks(ticks, labels=labels, rotation=90, fontsize=7)
            ax.set_yticks(ticks, labels=labels, fontsize=7)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Similarity")
        else:
            _empty(ax, "Similarity matrix unavailable")
        ax.set_title(title)
        ax.set_xlabel("Profile")
        ax.set_ylabel("Profile")
    fig.suptitle("Person similarity matrices", fontsize=14)
    _save(fig, path)


def _ranking_plot(profiles: list[dict], centrality: dict, path: Path) -> None:
    rows = centrality.get("aggregate", [])[:20]
    names = _names(profiles, [row["id"] for row in rows])
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(rows) + 1.8)), layout="constrained")
    if rows:
        ax.barh([names[row["id"]] for row in rows], [row["borda"] for row in rows], color="#287da6")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        _empty(ax, "No centrality ranks available")
    ax.set_xlabel("Borda points across centrality methods")
    ax.set_title("Key people by aggregate centrality" + (" (top 20)" if len(centrality.get("aggregate", [])) > 20 else ""))
    _save(fig, path)


def _dendrogram_plot(profiles: list[dict], path: Path) -> None:
    ordered = sorted(profiles, key=lambda profile: str(profile["id"]))
    ids = [str(profile["id"]) for profile in ordered]
    names = _names(ordered, ids)
    documents = [" ".join([*profile.get("skills", []),
                           *[job.get("title", "") for job in profile.get("experience", [])]])
                 for profile in ordered]
    fig, ax = plt.subplots(figsize=(max(10, min(18, len(ids) * 0.65)), 6), layout="constrained")
    if len(ids) < 2 or not any(document.strip() for document in documents):
        _empty(ax, "At least two profiles with skills or job titles are needed")
    else:
        try:
            matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(documents).toarray()
            if len(ids) > 60:
                dendrogram(linkage(matrix, method="ward"), truncate_mode="lastp", p=30,
                           show_leaf_counts=True, ax=ax, color_threshold=None)
                ax.set_xlabel("Leaf cluster (count in parentheses)")
            else:
                dendrogram(linkage(matrix, method="ward"), labels=[names[pid] for pid in ids],
                           leaf_rotation=90, leaf_font_size=7, ax=ax, color_threshold=None)
                ax.set_xlabel("Profile")
            ax.set_ylabel("Ward distance on TF-IDF features")
        except ValueError:
            _empty(ax, "No usable skill or title terms for clustering")
    ax.set_title("Hierarchical clustering of skills and job titles")
    _save(fig, path)


def _stability_plot(profiles: list[dict], validation: dict, path: Path) -> None:
    perturbation = validation.get("perturbation", {})
    per_person = perturbation.get("per_person", {})
    rows = sorted(per_person.items(), key=lambda item: item[1]["baseline_rank"])[:20]
    names = _names(profiles, [pid for pid, _ in rows])
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(rows) + 1.8)), layout="constrained")
    if rows and all(data.get("rank_interval") is not None for _, data in rows):
        for index, (pid, data) in enumerate(rows):
            low, median, high = data["rank_interval"]
            ax.hlines(index, low, high, color="#287da6", linewidth=2)
            ax.plot(median, index, "o", color="#287da6", markersize=5)
            ax.plot(data["baseline_rank"], index, "x", color="#aa3a32", markersize=7)
        ax.set_yticks(range(len(rows)), [names[pid] for pid, _ in rows])
        ax.invert_yaxis()
        ax.set_xlim(0.5, len(per_person) + 0.5)
        ax.set_xticks(range(1, len(per_person) + 1, max(1, ceil(len(per_person) / 20))))
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(handles=[Line2D([0], [0], color="#287da6", marker="o", label="95% interval and median"),
                           Line2D([0], [0], color="#aa3a32", marker="x", linestyle="", label="Baseline rank")],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    else:
        _empty(ax, "Perturbation rank intervals unavailable")
    ax.set_xlabel("Rank (lower is more central)")
    iterations = perturbation.get("iterations", 0)
    ax.set_title(f"Rank stability under edge perturbation ({iterations} iterations)" +
                 (" — top 20" if len(per_person) > 20 else ""))
    _save(fig, path)


def _correlation_plot(centrality: dict, path: Path) -> None:
    correlations = centrality.get("correlations", {}).get("kendall", {})
    metrics = list(correlations)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    if metrics:
        matrix = np.array([[np.nan if correlations[first].get(second) is None else
                            correlations[first][second] for second in metrics] for first in metrics], dtype=float)
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(metrics)), labels=metrics, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(metrics)), labels=metrics, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Kendall tau")
        if len(metrics) <= 12:
            for row in range(len(metrics)):
                for column in range(len(metrics)):
                    value = matrix[row, column]
                    ax.text(column, row, "—" if np.isnan(value) else f"{value:.2f}",
                            ha="center", va="center", fontsize=6,
                            color="white" if np.isfinite(value) and abs(value) > 0.6 else "black")
    else:
        _empty(ax, "Centrality rank correlations unavailable")
    ax.set_title("Agreement between centrality methods")
    _save(fig, path)


def write_plots(
    profiles: list[dict], bundle: dict, centrality: dict, clustering: dict,
    validation: dict, analysis: dict, config: dict, output_dir: Path,
) -> dict[str, Path]:
    """Generate all report figures from one completed, offline analysis run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"{name}.png" for name in (
        "graph_combined", "similarity_heatmap", "centrality_ranking", "dendrogram",
        "stability", "rank_correlation_heatmap",
    )}
    _graph_plot(profiles, bundle, centrality, clustering, config, paths["graph_combined"])
    # The brief's generic filename points to the same computed combined view.
    paths["graph"] = output_dir / "graph.png"
    copyfile(paths["graph_combined"], paths["graph"])
    _similarity_plot(profiles, analysis, paths["similarity_heatmap"])
    _ranking_plot(profiles, centrality, paths["centrality_ranking"])
    _dendrogram_plot(profiles, paths["dendrogram"])
    _stability_plot(profiles, validation, paths["stability"])
    _correlation_plot(centrality, paths["rank_correlation_heatmap"])
    return paths
