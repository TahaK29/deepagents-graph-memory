from deepagents_graph_memory.backend import GraphMemoryBackend


def test_kuzu_memory_backend_round_trip():
    backend = GraphMemoryBackend.create()

    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")

    node = backend.read("/nodes/service/langfuse.md")
    recall = backend.recall_graph_memory("langfuse depend")

    assert node.error is None
    assert "langfuse" in node.file_data["content"]
    assert "DEPENDS_ON" in recall
