# - Checks that a graph page does not show more connections than requested.
# - Case: a small connection limit leaves out extra connections and tells the reader.

from deepagents_graph_memory.backend import GraphMemoryBackend


def test_node_page_truncates_edges():
    backend = GraphMemoryBackend.create(max_edges=1)
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "postgres")

    result = backend.read("/nodes/service/langfuse.md")

    assert result.error is None
    assert "Results truncated" in result.file_data["content"]
