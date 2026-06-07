from deepagents_graph_memory.backend import GraphMemoryBackend


def make_backend(max_edges=100):
    backend = GraphMemoryBackend.create(max_edges=max_edges)
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "postgres")
    backend.add_graph_edge("team", "sre-team", "OWNS", "service", "langfuse")
    backend.add_graph_edge("incident", "incident-123", "AFFECTED", "service", "langfuse")
    backend.add_graph_edge("incident", "incident-123", "RESOLVED_BY", "runbook", "restart-ingestion-workers")
    return backend


def test_read_index_and_schema():
    backend = make_backend()

    index = backend.read("/index.md")
    schema = backend.read("/graph/schema.md")

    assert index.error is None
    assert "Graph Memory" in index.file_data["content"]
    assert schema.error is None
    assert "DEPENDS_ON" in schema.file_data["content"]


def test_read_node_page():
    backend = make_backend()

    result = backend.read("/graph/nodes/service/langfuse.md")

    assert result.error is None
    content = result.file_data["content"]
    assert "# service: langfuse" in content
    assert "Depends on" in content
    assert "Owns (incoming)" in content
    assert "Affected (incoming)" in content


def test_read_neighborhood_page():
    backend = make_backend()

    result = backend.read("/views/neighborhood/service/langfuse.md")

    assert result.error is None
    assert "# Neighborhood: service: langfuse" in result.file_data["content"]


def test_read_search_page():
    backend = make_backend()

    result = backend.read("/search/redis.md")

    assert result.error is None
    assert "redis" in result.file_data["content"]


def test_read_missing_node_returns_error():
    backend = make_backend()

    result = backend.read("/nodes/service/missing.md")

    assert result.file_data is None
    assert "not found" in result.error


def test_read_limits_slice_lines():
    backend = make_backend()

    result = backend.read("/index.md", offset=0, limit=1)

    assert result.error is None
    assert result.file_data["content"] == "# Graph Memory"


def test_download_files_uses_virtual_graph_views():
    backend = make_backend()

    [response] = backend.download_files(["/graph/schema.md"])

    assert response.error is None
    assert response.content is not None
    assert b"Graph Schema" in response.content
