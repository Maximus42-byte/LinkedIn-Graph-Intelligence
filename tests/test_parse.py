from pathlib import Path
import re

from lgi.fixtures_gen.generator import build_profiles
from lgi.parse import parse_html
from lgi.sources.fixture import FixtureAdapter


def test_every_fixture_recovers_all_six_sections_exactly() -> None:
    expected, _ = build_profiles()
    actual = list(FixtureAdapter('data/fixtures').iter_profiles())
    assert len(actual) == 10
    for source, parsed in zip(expected, actual):
        assert all(parsed['coverage'].values())
        for section in ('basic', 'experience', 'education', 'skills', 'interests', 'activity'):
            assert parsed[section] == source[section]
        assert parsed['experience'][0]['end'] is None
        assert all(re.fullmatch(r'\d{4}-\d{2}', job['start']) for job in parsed['experience'])
        assert len(parsed['skills']) == len({skill.casefold() for skill in parsed['skills']})


def test_missing_sections_are_not_guessed() -> None:
    html = '<html><head><script type="application/ld+json">{"@type":"Person","name":"Synthetic Person","url":"https://example.invalid/person"}</script></head><body><main></main></body></html>'
    parsed = parse_html(html, source='live', profile_id='x')
    assert parsed['basic']['name'] == 'Synthetic Person'
    assert parsed['experience'] == []
    assert not any(parsed['coverage'].values())
    assert any('activity' in warning for warning in parsed['warnings'])
