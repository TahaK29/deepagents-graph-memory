# - Checks what an agent sees when it opens a graph file.
# - Cases: overview, graph structure, individual items, nearby connections, and search results;
#        missing items, reading part of a page, and downloading a view.

import re

from deepagents.backends import CompositeBackend

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


def test_removed_neighborhood_page_is_rejected():
    backend = make_backend()

    result = backend.read("/graph/views/neighborhood/service/langfuse.md")

    assert result.file_data is None
    assert "Unsupported graph memory path" in result.error


def test_node_inspection_shows_stored_provenance_and_edge_properties():
    backend = GraphMemoryBackend.create(namespace=("project",))
    backend.add_graph_node("service", "api", {"tier": "prod", "search_text": "internal"}, source="incident-7", created_by="agent-1")
    backend.add_graph_edge("service", "api", "DEPENDS_ON", "service", "db", {"evidence": "timeout log"}, source="trace-1")
    backend.add_graph_edge("team", "ops", "OWNS", "service", "api", {"evidence": "on-call record"})

    content = backend.read("/graph/nodes/service/api.md").file_data["content"]

    assert "source" in content and "incident-7" in content
    assert "created_by" in content and "agent-1" in content
    assert "created_at" in content and "updated_at" in content
    db_line = next(line for line in content.splitlines() if line.startswith("- [db](/graph/nodes/service/db.md)"))
    ops_line = next(line for line in content.splitlines() if line.startswith("- [ops](/graph/nodes/team/ops.md)"))
    assert "timeout log" in db_line and "trace-1" in db_line and "on-call record" not in db_line
    assert "on-call record" in ops_line and "timeout log" not in ops_line
    assert "scope_key" not in content and "search_text" not in content


def test_read_search_page():
    backend = make_backend()

    result = backend.read("/search/redis.md")

    assert result.error is None
    assert "redis" in result.file_data["content"]


def test_relationship_search_links_to_inspectable_node():
    backend = make_backend()

    content = backend.read("/graph/search/depends.md").file_data["content"]

    assert "[langfuse DEPENDS_ON " in content
    assert "](/graph/nodes/service/langfuse.md)" in content


def test_inspection_links_route_through_composite_backend():
    backend = make_backend()
    composite = CompositeBackend(default=GraphMemoryBackend.create(), routes={"/graph/": backend})

    for page in ("/graph/nodes/service/langfuse.md", "/graph/search/depends.md"):
        result = composite.read(page)
        assert result.error is None
        links = re.findall(r"\]\((/graph/nodes/[^)]+)\)", result.file_data["content"])
        assert links
        assert all(composite.read(link).error is None for link in links)
    assert composite.read("/nodes/service/langfuse.md").file_data is None


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
