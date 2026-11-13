"""Figures must be generated from real offline pipeline results."""

from pathlib import Path

from lgi.analysis.centrality import analyze_centrality
from lgi.analysis.clustering import cluster_profiles
from lgi.analysis.graphs import build_graphs
from lgi.analysis.similarity import analyze_similarity
from lgi.analysis.validation import validate_analysis
from lgi.config import load_config
from lgi.plots import write_plots
from lgi.sources.fixture import FixtureAdapter


ROOT = Path(__file__).resolve().parents[1]


def test_required_plots_from_fixture_results(tmp_path):
    config = load_config(ROOT / "config.yaml")
    config["perturbation_iterations"] = 5
    config["null_iterations"] = 0
    config["weight_iterations"] = 0
    profiles = list(FixtureAdapter(ROOT / "data/fixtures").iter_profiles())
    bundle = build_graphs(profiles, config)
    centrality = analyze_centrality(bundle["graphs"]["combined_manual"], config)
    clustering = cluster_profiles(profiles, bundle, config)
    validation = validate_analysis(bundle, centrality, clustering, config)
    analysis = {"similarity": analyze_similarity(profiles, config)}

    paths = write_plots(profiles, bundle, centrality, clustering, validation, analysis, config, tmp_path)

    expected = {
        "graph", "similarity_heatmap", "centrality_ranking",
        "dendrogram", "stability", "rank_correlation_heatmap",
    }
    assert set(paths) == expected
    for path in paths.values():
        assert path.exists()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1000
