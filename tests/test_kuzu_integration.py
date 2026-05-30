import pytest

from deepagents_graph_memory.backend import GraphMemoryBackend


def test_kuzu_local_backend_round_trip(tmp_path):
    pytest.importorskip("kuzu")
    pytest.importorskip("langchain_community.graphs.kuzu_graph")

    backend = GraphMemoryBackend.local(str(tmp_path / "graph-memory"))

    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")

    node = backend.read("/nodes/service/langfuse.md")
    recall = backend.recall_graph_memory("langfuse depend")

    assert node.error is None
    assert "langfuse" in node.file_data["content"]
    assert "DEPENDS_ON" in recall
