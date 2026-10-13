"""Bounded graph neighborhoods retain truncation metadata."""

import pytest

from deepagents_graph_memory.backend import GraphMemoryBackend


def test_neighborhood_truncates_edges():
    backend = GraphMemoryBackend.create()
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "postgres")

    result = backend.store.get_neighbors("service", "langfuse", max_edges=1)

    assert len(result.edges) == 1
    assert result.truncated_edges
    backend.close()


@pytest.mark.parametrize("max_nodes", [1, 2])
def test_neighborhood_bounds_neighbor_endpoints(max_nodes):
    backend = GraphMemoryBackend.create()
    try:
        for index in range(3):
            backend.add_graph_edge("File", "root", "LINKS", "File", f"child{index}")
        neighborhood = backend.store.get_neighbors("File", "root", max_nodes=max_nodes)
        endpoints = {("File", "root")}
        for edge in neighborhood.edges:
            endpoints.update(((edge.source_label, edge.source_id), (edge.target_label, edge.target_id)))
        assert len(endpoints) == max_nodes
        assert neighborhood.truncated_nodes
    finally:
        backend.close()
