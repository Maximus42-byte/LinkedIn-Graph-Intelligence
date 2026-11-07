from pathlib import Path
import json

from lgi.fixtures_gen.generator import generate_fixtures


def test_generator_is_deterministic_and_scales(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_fixtures(n=10, seed=73, root=first)
    generate_fixtures(n=10, seed=73, root=second)
    first_files = sorted((first / "data/fixtures").glob("*.html"))
    second_files = sorted((second / "data/fixtures").glob("*.html"))
    assert len(first_files) == len(second_files) == 10
    assert [p.read_bytes() for p in first_files] == [p.read_bytes() for p in second_files]
    for file in first_files:
        html = file.read_text()
        assert '<meta name="synthetic" content="true">' in html
        assert "SYNTHETIC" in html
    truth = json.loads((first / "data/ground_truth.json").read_text())
    assert len(truth["profiles"]) == 10
    generate_fixtures(n=50, seed=73, root=first)
    assert len(list((first / "data/fixtures").glob("*.html"))) == 50
