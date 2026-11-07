"""Focused checks for the phase 6 validation draft."""

import json

import networkx as nx
import numpy as np

from lgi.analysis.graphs import LAYERS, combine_with_weights
from lgi.analysis.validation import _perturb, validate_analysis


def _sample_bundle():
    ids = [f"p{number:02d}" for number in range(1, 9)]
    graphs = {layer: nx.Graph() for layer in LAYERS}
    for graph in graphs.values():
        graph.add_nodes_from(ids)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids[i + 1 :], i + 1):
            if i // 4 == j // 4:
                graphs["org"].add_edge(a, b, weight=2.0)
                graphs["skill"].add_edge(a, b, weight=0.8)
            if (i + j) % 3 == 0:
                graphs["edu"].add_edge(a, b, weight=1.0)
            graphs["semantic"].add_edge(a, b, weight=0.1 + (i + j) / 50)
            if j == i + 1:
                graphs["activity"].add_edge(a, b, weight=1.0)
    config = {
        "seed": 73,
        "layer_weights": {"org": 0.25, "edu": 0.10, "skill": 0.25, "semantic": 0.25, "activity": 0.15},
        "perturbation_iterations": 8,
        "perturbation_fraction": 0.1,
        "null_iterations": 5,
        "weight_iterations": 5,
        "weight_dirichlet_concentration": 20,
        "null_backbone_density_threshold": 0.8,
        "null_backbone_keep_fraction": 0.5,
        "null_min_informative_n": 30,
    }
    graphs["combined_manual"] = combine_with_weights(graphs, config["layer_weights"])
    return {"graphs": graphs}, config, ids


def test_validation_is_deterministic_and_json_safe(tmp_path):
    bundle, config, ids = _sample_bundle()
    truth = {"profiles": {pid: {"team": "A" if index < 4 else "B", "role": "member"} for index, pid in enumerate(ids)}}
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(truth))
    clustering = {"labels": {"louvain": {pid: index // 4 for index, pid in enumerate(ids)}}}
    first = validate_analysis(bundle, {}, clustering, config, path)
    second = validate_analysis(bundle, {}, clustering, config, path)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["ground_truth"]["results"]["louvain"]["ari"] == 1.0
    assert len(first["perturbation"]["per_person"]) == len(ids)
    assert set(first["ablation"]) == set(LAYERS)
    assert first["null_models"]["backbone"]["used"] is True
    assert first["null_models"]["gnm"]["samples"] == 5
    assert first["null_models"]["statistically_informative"] is False


def test_complete_graph_perturbation_removes_edges():
    graph = nx.complete_graph(6)
    nx.set_edge_attributes(graph, 1.0, "weight")
    altered = _perturb(graph, 0.2, np.random.default_rng(73))
    assert altered.number_of_edges() < graph.number_of_edges()


def test_disconnected_graph_still_validates():
    bundle, config, ids = _sample_bundle()
    for graph in bundle["graphs"].values():
        graph.remove_edges_from(list(graph.edges()))
    result = validate_analysis(bundle, {}, {"labels": {}}, config)
    assert result["ground_truth"] is None
    assert result["perturbation"]["iterations"] == 8
    assert len(result["perturbation"]["baseline_ranking"]) == len(ids)


def test_tied_scores_use_average_ranks_and_shared_topk_cutoff():
    from lgi.analysis.validation import _perturbation, _ranking, _topk_set

    graph = nx.complete_graph(['a', 'b', 'c', 'd'])
    nx.set_edge_attributes(graph, 1.0, 'weight')
    order, metrics = _ranking(graph, seed=73)
    ranks = metrics['aggregate_rank']
    assert set(ranks.values()) == {2.5}
    assert _topk_set(ranks, 3) == set(graph)
    summary = _perturbation(graph, {'perturbation_iterations': 3, 'perturbation_fraction': 0.0}, 73, order, ranks)
    assert summary['top3_set_retention'] == 1.0
    assert summary['baseline_top3_set'] == sorted(graph)
    assert all(row['top3_frequency'] == 1.0 for row in summary['per_person'].values())
