from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.stores import InMemoryGraphStore


def test_neighborhood_truncates_edges():
    store = InMemoryGraphStore()
    backend = GraphMemoryBackend(store, max_edges=1)
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "postgres")

    result = backend.read("/views/neighborhood/service/langfuse.md")

    assert result.error is None
    assert "Results truncated" in result.file_data["content"]
