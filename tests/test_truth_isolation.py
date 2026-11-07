from pathlib import Path


def test_only_validation_reads_ground_truth() -> None:
    paths = [Path('src/lgi/parse.py'), *Path('src/lgi/analysis').glob('*.py')]
    for path in paths:
        if path.name == 'validation.py':
            continue
        assert 'ground_truth.json' not in path.read_text()
        assert 'ground_truth_path' not in path.read_text()
