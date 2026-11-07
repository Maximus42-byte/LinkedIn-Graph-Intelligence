import json

from lgi.analysis.graphs import build_graphs, write_graph_json
from lgi.config import load_config
from lgi.sources.fixture import FixtureAdapter


def test_layers_and_known_edges(tmp_path):
    profiles = list(FixtureAdapter('data/fixtures').iter_profiles())
    bundle = build_graphs(profiles, load_config())
    graphs = bundle['graphs']
    for graph in graphs.values():
        assert graph.number_of_nodes() == len(profiles)
        assert not list(graph.selfloop_edges()) if hasattr(graph, 'selfloop_edges') else all(a != b for a, b in graph.edges())
        assert all(data['weight'] > 0 for _, _, data in graph.edges(data=True))
    assert graphs['org'].has_edge('p01', 'p02')
    assert graphs['edu'].has_edge('p04', 'p05')
    assert graphs['activity'].number_of_edges() > 0
    assert {'jaccard', 'tfidf_cosine'} <= graphs['skill']['p01']['p02'].keys()
    assert {'embedding_cosine', 'tfidf_cosine'} <= graphs['semantic']['p01']['p02'].keys()
    assert bundle['bipartite'].number_of_edges() == sum(len(p['skills']) for p in profiles)
    path = tmp_path / 'graph.json'
    write_graph_json(bundle, path)
    saved = path.read_bytes()
    write_graph_json(build_graphs(profiles, load_config()), path)
    assert path.read_bytes() == saved
    assert len(json.loads(saved)['nodes']) == 10
