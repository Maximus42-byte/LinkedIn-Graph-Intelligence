from pathlib import Path
from types import SimpleNamespace

import yaml

from lgi.pipeline import run_pipeline


def test_two_full_runs_are_byte_identical_and_reported(tmp_path: Path) -> None:
    original = Path.cwd()
    config = yaml.safe_load((original / 'config.yaml').read_text())
    config['paths']['fixtures'] = str(original / 'data/fixtures')
    config['paths']['ground_truth'] = str(original / 'data/ground_truth.json')
    config['perturbation_iterations'] = 8
    config['null_iterations'] = 5
    config['weight_iterations'] = 5
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    args = SimpleNamespace(source='fixture', n=10, config=str(path), anonymise=False)
    run_pipeline(args)
    first_graph = (tmp_path / 'output/graph.json').read_bytes()
    first_report = (tmp_path / 'output/report.md').read_bytes()
    run_pipeline(args)
    assert (tmp_path / 'output/graph.json').read_bytes() == first_graph
    assert (tmp_path / 'output/report.md').read_bytes() == first_report
    assert b'synthetic fixtures' in first_report
    assert (tmp_path / 'output/tables/coverage.csv').exists()
