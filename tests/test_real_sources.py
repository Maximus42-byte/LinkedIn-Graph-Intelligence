"""Offline checks for opt-in real-data sources. Every name here is fabricated."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lgi.analysis.graphs import build_graphs
from lgi.config import load_config
from lgi.real_pipeline import run_real_pipeline
from lgi.schema import validate_profile
from lgi.secrets import anonymise_profile, load_secrets, pseudonym
from lgi.sources.export import ExportAdapter
from lgi.sources.openalex import OpenAlexAdapter


def _git_repo(path: Path, ignore: str = "data/real/\n") -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / ".gitignore").write_text(ignore)
    return path


def _secrets(repo: Path, key: str = "fictional-test-key") -> Path:
    path = repo / "data/real/secrets.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"openalex_api_key": key, "anon_salt": "fictional-salt"}))
    path.chmod(0o600)
    return path


def test_secrets_guards_and_persistent_salt(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    path = _secrets(repo)
    key, salt = load_secrets(path, require_key=True)
    assert key == "fictional-test-key"
    assert salt == b"fictional-salt"
    path.write_text(json.dumps({"openalex_api_key": "", "anon_salt": ""}))
    with pytest.raises(ValueError, match="empty or still the placeholder"):
        load_secrets(path, require_key=True)
    path.write_text(json.dumps({"openalex_api_key": "PASTE_YOUR_OPENALEX_API_KEY_HERE", "anon_salt": ""}))
    with pytest.raises(ValueError, match="placeholder"):
        load_secrets(path, require_key=True)
    path.unlink()
    with pytest.raises(ValueError, match="requires a local secrets file"):
        load_secrets(path, require_key=True)
    _, first = load_secrets(path)
    _, second = load_secrets(path)
    assert first == second
    assert (path.parent / "anon_salt").exists()
    unsafe = _git_repo(tmp_path / "unsafe", ignore="")
    unsafe_path = _secrets(unsafe)
    with pytest.raises(ValueError, match="not git-ignored"):
        load_secrets(unsafe_path, require_key=True)


def test_export_header_discovery_and_undated_stub(tmp_path: Path) -> None:
    directory = tmp_path / "export"
    directory.mkdir()
    (directory / "Profile.csv").write_text(
        "Export note: this is fabricated\nFirst Name,Last Name,Headline,Location,Summary,Profile URL\n"
        "Fable,Aster,Imaginary Engineer,Nowhere,Fictional bio,https://example.invalid/fable\n"
    )
    (directory / "Connections.csv").write_text(
        "Note before header\nFirst Name,Last Name,Company,Position,Connected On\n"
        "Mira,Quill,Imaginary Cloud Works,Senior Engineer,2024-04-05\n"
        "Tavi,Flint,Imaginary Cloud Works,Junior Engineer,2024-05-06\n"
    )
    adapter = ExportAdapter(directory)
    profiles = list(adapter.iter_profiles())
    assert len(profiles) == 3
    assert profiles[1]["experience"][0]["start"] == ""
    assert profiles[1]["connected_date"] == "2024-04-05"
    assert not profiles[1]["coverage"]["experience"]
    assert adapter.discovered["Connections.csv"]["kind"] == "connections"
    config = load_config("config.yaml")
    config["org_ignore_employer_share"] = 1.0
    graphs = build_graphs(profiles, config)["graphs"]
    assert graphs["org"].has_edge("p02", "p03")


class FakeResponse:
    def __init__(self, data: dict, status: int = 200, key: str = ""):
        self.data = data
        self.status_code = status
        self.headers = {"Retry-After": "0"}
        self.url = f"https://api.openalex.org/authors?api_key={key}"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise ValueError(f"failed URL: {self.url}")

    def json(self) -> dict:
        return self.data


def _openalex_payloads(key: str) -> list[FakeResponse]:
    return [
        FakeResponse({"id": "https://openalex.org/I77", "display_name": "Imaginary Academy"}, key=key),
        FakeResponse({"results": [
            {
                "id": "https://openalex.org/A2", "display_name": "Mira Quill", "works_count": 4,
                "affiliations": [{"institution": {"display_name": "Imaginary Academy"}, "years": [2023, 2024]}],
                "topics": [{"display_name": "Fictional Graph Theory"}],
            },
            {
                "id": "https://openalex.org/A1", "display_name": "Tavi Flint", "works_count": 9,
                "affiliations": [{"institution": {"display_name": "Imaginary Academy"}, "years": [2024]}],
                "topics": [{"display_name": "Fictional Graph Theory"}],
            },
        ]}, key=key),
        FakeResponse({"results": [{
            "id": "https://openalex.org/W9", "publication_date": "2024-03-05",
            "title": "Imaginary graph study",
            "authorships": [
                {"author": {"id": "https://openalex.org/A1"}},
                {"author": {"id": "https://openalex.org/A2"}},
            ],
        }]}, key=key),
    ]


def test_openalex_retry_cache_mapping_and_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = "fictional-test-key"
    responses = [FakeResponse({}, status=429, key=key), FakeResponse({}, status=503, key=key), *_openalex_payloads(key)]
    session = Mock()
    session.headers = {}
    session.get.side_effect = responses
    monkeypatch.setattr("lgi.sources.openalex.time.sleep", lambda delay: None)
    adapter = OpenAlexAdapter("I77", key, tmp_path / "cache", session=session, max_requests=5, max_authors=2)
    profiles = list(adapter.iter_profiles())
    assert adapter.request_count == 5
    assert adapter.selected_ids == ["A1", "A2"]
    assert adapter.coauthor_counts == {("A1", "A2"): 1}
    assert profiles[0]["activity"][0]["type"] == "publication"
    assert profiles[0]["activity"][0]["reaction_count"] == 0
    assert not profiles[0]["coverage"]["education"]
    assert not profiles[0]["coverage"]["interests"]
    assert not profiles[0]["coverage"]["experience"]
    assert all(key not in file.read_text() for file in (tmp_path / "cache").glob("*.json"))
    second_session = Mock()
    second_session.headers = {}
    cached = OpenAlexAdapter("I77", key, tmp_path / "cache", session=second_session, max_authors=2)
    assert list(cached.iter_profiles()) == profiles
    assert cached.request_count == 0
    second_session.get.assert_not_called()
    constrained = OpenAlexAdapter("I77", key, tmp_path / "empty", session=second_session, max_requests=1)
    second_session.get.side_effect = _openalex_payloads(key)
    with pytest.raises(RuntimeError, match="budget reached"):
        list(constrained.iter_profiles())


def test_openalex_redacts_key_on_error(tmp_path: Path) -> None:
    key = "fictional-test-key"
    session = Mock()
    session.headers = {}
    session.get.side_effect = ValueError(f"https://api.openalex.org/authors?api_key={key}")
    adapter = OpenAlexAdapter("I77", key, tmp_path / "cache", session=session)
    with pytest.raises(RuntimeError) as error:
        list(adapter.iter_profiles())
    assert key not in str(error.value)
    assert not (tmp_path / "cache").exists()


def test_anonymisation_stable_and_no_names(tmp_path: Path) -> None:
    profile = {
        "id": "A1", "source": "openalex",
        "basic": {"name": "Mira Quill", "headline": "", "location": "", "summary": "", "profile_url": "https://openalex.org/A1"},
        "experience": [{"company": "Imaginary Academy", "title": "", "start": "2023-01", "end": "2024-12", "location": "", "description": ""}],
        "education": [], "skills": ["Fictional Graph Theory"], "interests": [],
        "activity": [{"post_id": "W9", "date": "2024-03-05", "type": "publication", "text": "Imaginary graph study", "reaction_count": 0, "engaged_ids": []}],
        "coverage": {"basic": False, "experience": False, "education": False, "skills": True, "interests": False, "activity": True},
        "warnings": ["Mira Quill incomplete at Imaginary Academy"],
    }
    validate_profile(profile)
    first = anonymise_profile(profile, b"fictional-salt")
    second = anonymise_profile(profile, b"fictional-salt")
    assert first == second
    assert first["experience"][0]["company"] == pseudonym(b"fictional-salt", "employer", "Imaginary Academy")
    content = json.dumps(first)
    for raw in ("Mira Quill", "Imaginary Academy", "Imaginary graph study", "https://openalex.org"):
        assert raw not in content


def test_real_export_outputs_stay_isolated(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "runrepo")
    _secrets(repo)
    directory = repo / "data/real/export"
    directory.mkdir()
    (directory / "Profile.csv").write_text(
        "First Name,Last Name,Headline,Location,Summary,Profile URL\n"
        "Fable,Aster,Fictional Engineer,Nowhere,Fictional bio,https://example.invalid/fable\n"
    )
    (directory / "Connections.csv").write_text(
        "First Name,Last Name,Company,Position,Connected On\n"
        "Mira,Quill,Imaginary Cloud Works,Senior Engineer,2024-04-05\n"
        "Tavi,Flint,Imaginary Cloud Works,Junior Engineer,2024-05-06\n"
    )
    config = load_config("config.yaml")
    config["root"] = str(repo)
    config["perturbation_iterations"] = 2
    config["null_iterations"] = 2
    config["weight_iterations"] = 2
    args = SimpleNamespace(source="export", secrets="data/real/secrets.json", no_anonymise=False)
    run_real_pipeline(args, config)
    report = (repo / "output_real/report.md").read_text()
    assert "Connection-only subgraph" in report
    assert "No real-data ground truth" in report
    assert "Mira Quill" not in report
    assert "Imaginary Cloud Works" not in report
    profiles = (repo / "data/real_profiles.json").read_text()
    assert "Mira Quill" not in profiles
    assert "Imaginary Cloud Works" not in profiles
    assert not (repo / "output").exists()
    assert not (repo / "data/profiles.json").exists()


def test_real_openalex_pipeline_has_coauthor_layer_and_no_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _git_repo(tmp_path / "openalexrepo")
    key = "fictional-test-key"
    _secrets(repo, key)
    session = Mock()
    session.headers = {}
    session.get.side_effect = _openalex_payloads(key)
    adapter = OpenAlexAdapter("I77", key, repo / "data/real/openalex_cache", session=session, max_authors=2)
    monkeypatch.setattr("lgi.real_pipeline.OpenAlexAdapter", lambda *args, **kwargs: adapter)
    config = load_config("config.yaml")
    config["root"] = str(repo)
    config["perturbation_iterations"] = 2
    config["null_iterations"] = 2
    config["weight_iterations"] = 2
    args = SimpleNamespace(
        source="openalex", secrets="data/real/secrets.json", no_anonymise=False,
        institution="I77", year_from=2020, max_authors=2, max_requests=20, refresh=False,
    )
    run_real_pipeline(args, config)
    graph = json.loads((repo / "output_real/graph.json").read_text())
    assert len(graph["layers"]["coauthor"]["links"]) == 1
    assert graph["layers"]["coauthor"]["links"][0]["weight"] == 1.0
    assert json.loads((repo / "output_real/validation.json").read_text())["ground_truth"] is None
    for path in [repo / "data/real_profiles.json", *list((repo / "output_real").rglob("*"))]:
        if path.is_file():
            text = path.read_text()
            assert key not in text
            assert "Mira Quill" not in text
            assert "Imaginary Academy" not in text
    assert not (repo / "output").exists()


def test_real_paths_are_gitignored() -> None:
    rules = Path(".gitignore").read_text().splitlines()
    for expected in ("data/real/", "data/real_profiles.json", "output_real/", "secrets.json", "*salt*"):
        assert expected in rules


def test_real_outputs_refuse_symlinks(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "symlinkrepo")
    _secrets(repo)
    (repo / "output").mkdir()
    (repo / "output_real").symlink_to(repo / "output", target_is_directory=True)
    config = load_config("config.yaml")
    config["root"] = str(repo)
    args = SimpleNamespace(source="export", secrets="data/real/secrets.json", no_anonymise=False)
    with pytest.raises(ValueError, match="must not be symlinks"):
        run_real_pipeline(args, config)
    assert list((repo / "output").iterdir()) == []
