# - Checks that a graph page does not show more connections than requested.
# - Case: a small connection limit leaves out extra connections and tells the reader.

import pytest

from deepagents_graph_memory.backend import GraphMemoryBackend


def test_node_page_truncates_edges():
    backend = GraphMemoryBackend.create(max_edges=1)
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "postgres")

    result = backend.read("/nodes/service/langfuse.md")

    assert result.error is None
    assert "Results truncated" in result.file_data["content"]


@pytest.mark.parametrize("max_nodes", [1, 2])
def test_node_page_bounds_exposed_neighbor_endpoints(max_nodes):
    backend = GraphMemoryBackend.create(max_nodes=max_nodes)
    try:
        for index in range(3):
            backend.add_graph_edge("File", "root", "LINKS", "File", f"child{index}")
        neighborhood = backend.store.get_neighbors("File", "root", max_nodes=max_nodes)
        endpoints = {("File", "root")}
        for edge in neighborhood.edges:
            endpoints.update(((edge.source_label, edge.source_id), (edge.target_label, edge.target_id)))
        assert len(endpoints) == max_nodes
        assert neighborhood.truncated_nodes
        page = backend.read("/nodes/File/root.md").file_data["content"]
        assert page.count("/graph/nodes/File/child") == max_nodes - 1
        assert "Results truncated" in page
    finally:
        backend.close()
