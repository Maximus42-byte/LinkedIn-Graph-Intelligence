import networkx as nx

from lgi.analysis.centrality import analyze_centrality, compute_metrics
from lgi.config import load_config


def test_path_centrality_matches_known_middle_node():
    graph = nx.path_graph(['a', 'b', 'c'])
    nx.set_edge_attributes(graph, 1.0, 'weight')
    metrics = compute_metrics(graph)
    assert metrics['b']['degree'] == 1.0
    assert metrics['b']['betweenness'] == 1.0
    assert metrics['a']['betweenness'] == 0.0
    assert metrics['b']['k_core'] == 1
    result = analyze_centrality(graph, load_config())
    assert result['rankings']['borda'][0] == 'b'


def test_disconnected_graph_does_not_crash():
    graph = nx.Graph()
    graph.add_nodes_from(['a', 'b', 'c'])
    graph.add_edge('a', 'b', weight=1.0)
    result = analyze_centrality(graph, load_config())
    assert set(result['metrics']) == {'a', 'b', 'c'}
    assert len(result['aggregate']) == 3
