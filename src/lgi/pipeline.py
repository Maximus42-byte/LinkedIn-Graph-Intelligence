"""Run extraction, storage, and analysis from one command."""

from pathlib import Path
import json
from tempfile import TemporaryDirectory

from lgi.analysis.graphs import build_graphs, write_graph_json
from lgi.analysis.centrality import analyze_centrality
from lgi.analysis.clustering import cluster_profiles
from lgi.analysis.validation import validate_analysis
from lgi.analysis.career import analyze_career
from lgi.analysis.activity import analyze_activity
from lgi.analysis.similarity import analyze_similarity
from lgi.config import load_config
from lgi.fixtures_gen.generator import generate_fixtures
from lgi.sources.export import ExportAdapter
from lgi.sources.fixture import FixtureAdapter
from lgi.sources.manual import ManualAdapter
from lgi.store import write_profiles
from lgi.report import write_report
from lgi.plots import write_plots


def _path(config: dict, key: str) -> Path:
    return Path(config["root"]) / config["paths"][key]


def _process(profiles: list[dict], config: dict, ground_truth_path: Path | None = None) -> None:
    if not profiles:
        raise ValueError("Source supplied no profiles")
    write_profiles(profiles, _path(config, "profiles"), _path(config, "flat"))
    bundle = build_graphs(profiles, config)
    centrality = analyze_centrality(bundle["graphs"]["combined_manual"], config)
    clustering = cluster_profiles(profiles, bundle, config)
    validation = validate_analysis(bundle, centrality, clustering, config, ground_truth_path)
    analysis = {"career": analyze_career(profiles), "activity": analyze_activity(profiles, bundle["graphs"], config), "similarity": analyze_similarity(profiles, config)}
    analysis_path = _path(config, "output") / "analysis.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    validation_path = _path(config, "output") / "validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_graph_json(bundle, _path(config, "output") / "graph.json", metrics=centrality["metrics"], roles=centrality["roles"], clusters=clustering["labels"])
    write_report(profiles, bundle, centrality, clustering, validation, analysis, config, _path(config, "output"))
    write_plots(profiles, bundle, centrality, clustering, validation, analysis, config, _path(config, "output"))
    print(f"Processed {len(profiles)} profiles; graph has {bundle['graphs']['combined_manual'].number_of_edges()} combined edges")


def run_pipeline(args) -> None:
    """Keep generated scale runs isolated from the committed default fixtures."""
    config = load_config(args.config)
    if args.source == "fixture":
        if args.n == 10:
            profiles = list(FixtureAdapter(_path(config, "fixtures")).iter_profiles())
            _process(profiles, config, _path(config, "ground_truth"))
        else:
            with TemporaryDirectory() as temp:
                generate_fixtures(n=args.n, seed=config["seed"], root=Path(temp))
                profiles = list(FixtureAdapter(Path(temp) / "data/fixtures").iter_profiles())
                _process(profiles, config, Path(temp) / "data/ground_truth.json")
        return
    if args.source == "export":
        profiles = list(ExportAdapter(_path(config, "export")).iter_profiles())
    elif args.source == "manual":
        profiles = list(ManualAdapter(_path(config, "manual"), anonymise=args.anonymise).iter_profiles())
    else:
        raise NotImplementedError("Live adapter is added in phase 9")
    _process(profiles, config)
