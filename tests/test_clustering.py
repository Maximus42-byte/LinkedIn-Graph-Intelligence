"""Focused phase 5 checks; this draft lives outside the repository."""

import json
from pathlib import Path

import networkx as nx

from lgi.analysis.clustering import GRAPH_METHODS, FEATURE_METHODS, cluster_profiles
from lgi.analysis.graphs import build_graphs
from lgi.config import load_config
from lgi.sources.fixture import FixtureAdapter


def test_default_fixtures_compare_every_method():
    root = Path("/home/maximus/Projects/LinkedIn-Graph-Intelligence")
    config = load_config(root / "config.yaml")
    profiles = list(FixtureAdapter(root / "data/fixtures").iter_profiles())
    result = cluster_profiles(profiles, build_graphs(profiles, config), config)

    assert set(result["labels"]) == set(GRAPH_METHODS + FEATURE_METHODS)
    assert len(result["comparisons"]) == 45
    assert len(result["k_selection"]["candidates"]) >= 2
    assert result["skill_families"]
    assert all(set(assignment) == {profile["id"] for profile in profiles} for assignment in result["labels"].values())
    json.dumps(result, sort_keys=True, allow_nan=False)
    repeat = cluster_profiles(profiles, build_graphs(profiles, config), config)
    assert json.dumps(result, sort_keys=True) == json.dumps(repeat, sort_keys=True)


def test_disconnected_graph_and_all_noise_are_safe():
    profiles = [
        {"id": "p1", "skills": ["alpha"], "experience": [{"title": "Alpha Engineer"}]},
        {"id": "p2", "skills": ["alpha"], "experience": [{"title": "Alpha Analyst"}]},
        {"id": "p3", "skills": ["beta"], "experience": [{"title": "Beta Engineer"}]},
        {"id": "p4", "skills": ["beta"], "experience": [{"title": "Beta Analyst"}]},
    ]
    graph = nx.Graph()
    graph.add_nodes_from(profile["id"] for profile in profiles)
    graph.add_edge("p1", "p2", weight=1.0)
    graph.add_edge("p3", "p4", weight=1.0)
    config = {"seed": 73, "k_range": [2, 3], "hdbscan_min_cluster_size": 4}
    result = cluster_profiles(profiles, {"graphs": {"combined_manual": graph}, "skill_projection": nx.Graph()}, config)
    assert all(len(assignment) == 4 for assignment in result["labels"].values())
    assert result["quality"]["girvan_newman"]["modularity"] is not None
    assert result["labels"]["girvan_newman"] != result["labels"]["louvain"]

    no_text = [{"id": "p1", "skills": [], "experience": []}, {"id": "p2", "skills": [], "experience": []}]
    empty = nx.Graph()
    empty.add_nodes_from(["p1", "p2"])
    result = cluster_profiles(no_text, {"graphs": {"combined_manual": empty}, "skill_projection": nx.Graph()}, config)
    assert result["quality"]["hdbscan"]["noise_count"] == 2
    assert result["topics"] == []
    json.dumps(result, allow_nan=False)
