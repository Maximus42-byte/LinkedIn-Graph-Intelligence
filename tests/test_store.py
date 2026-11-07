import json
from lgi.schema import empty_profile
from lgi.store import write_profiles


def test_store_writes_stable_flat_tables(tmp_path):
    profile = empty_profile('p01', 'manual')
    profile['skills'] = ['Graph Theory']
    write_profiles([profile], tmp_path / 'profiles.json', tmp_path / 'flat')
    assert json.loads((tmp_path / 'profiles.json').read_text())[0]['id'] == 'p01'
    assert 'Graph Theory' in (tmp_path / 'flat/skills.csv').read_text()
