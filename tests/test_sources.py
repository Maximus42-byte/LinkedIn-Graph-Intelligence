import json
from pathlib import Path

from lgi.schema import empty_profile
from lgi.sources.export import ExportAdapter
from lgi.sources.manual import ManualAdapter


def test_export_owner_and_connection_stub() -> None:
    profiles = list(ExportAdapter(Path('tests/data/export_sample')).iter_profiles())
    assert len(profiles) == 2
    assert profiles[0]['basic']['name'] == 'Fable Aster'
    assert profiles[0]['experience'][0]['end'] is None
    assert profiles[0]['coverage']['skills']
    assert profiles[1]['basic']['headline'] == 'Synthetic Engineer at Imaginary Cloud Works'
    assert not profiles[1]['coverage']['experience']
    assert profiles[1]['coverage']['basic']


def test_manual_anonymisation_is_stable(tmp_path: Path) -> None:
    source = empty_profile('p01', 'manual')
    source['basic']['name'] = 'Fictional Example'
    source['basic']['profile_url'] = 'https://example.invalid/fictional'
    (tmp_path / 'profile.json').write_text(json.dumps(source))
    first = list(ManualAdapter(tmp_path, anonymise=True).iter_profiles())[0]
    second = list(ManualAdapter(tmp_path, anonymise=True).iter_profiles())[0]
    assert first == second
    assert first['basic']['name'] != source['basic']['name']
    assert first['basic']['profile_url'] != source['basic']['profile_url']
