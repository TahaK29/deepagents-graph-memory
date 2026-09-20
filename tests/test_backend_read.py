# - Checks what an agent sees when it opens a graph file.
# - Cases: overview, graph structure, individual items, nearby connections, and search results;
#        missing items, reading part of a page, and downloading a view.

import re

import pytest
from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import ReadResult

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
    assert result.file_data["content"] == "# Graph Memory\n"


def test_download_files_uses_virtual_graph_views():
    backend = make_backend()

    [response] = backend.download_files(["/graph/schema.md"])

    assert response.error is None
    assert response.content is not None
    assert b"Graph Schema" in response.content


def test_read_preserves_newlines_and_rejects_offsets_past_end():
    backend = make_backend()
    try:
        assert backend.read("/index.md", limit=1).file_data["content"] == "# Graph Memory\n"
        assert backend.read("/index.md", offset=10000).error
        assert backend.read("/index.md", limit=0).file_data["content"] == ""
    finally:
        backend.close()


def test_read_normalizes_bounds_and_reports_pagination():
    backend = make_backend()
    try:
        full = backend.read("/index.md")
        first = backend.read("/index.md", offset=-10, limit=1)
        assert first.error is None
        assert first.file_data["content"] == "# Graph Memory\n"
        rest = backend.read("/index.md", offset=1)
        if hasattr(ReadResult, "next_offset"):
            assert (first.start_line, first.end_line, first.next_offset) == (1, 1, 1)
            assert first.total_lines == len(full.file_data["content"].splitlines())
            assert rest.start_line == 2 and rest.end_line == first.total_lines
            assert rest.next_offset is None
        assert first.file_data["content"] + rest.file_data["content"] == full.file_data["content"]
        for limit in (0, -1):
            empty = backend.read("/uninspected", offset=10000, limit=limit)
            assert empty.error is None and empty.file_data["content"] == ""
            if hasattr(ReadResult, "no_lines_requested"):
                assert empty.no_lines_requested
                assert (empty.start_line, empty.end_line, empty.next_offset, empty.total_lines) == (None, None, None, None)
    finally:
        backend.close()


def test_file_operations_return_errors_for_directories_and_closed_stores():
    backend = make_backend()
    assert backend.read("/graph/").error
    assert backend.download_files(["/graph/"])[0].error == "is_directory"
    backend.close()
    assert "closed" in backend.ls("/nodes").error
    assert "closed" in backend.read("/schema.md").error
    assert "closed" in backend.glob("**/*.md").error
    assert "closed" in backend.grep("service").error


@pytest.mark.asyncio
async def test_composite_async_backend_contract():
    backend = make_backend()
    composite = CompositeBackend(default=backend, routes={"/graph/": backend})
    path = "/graph/nodes/service/redis.md"
    try:
        assert path in {entry["path"] for entry in (await composite.als("/graph/nodes/service")).entries}
        assert (await composite.aread(path)).file_data["content"].startswith("# service: redis")
        assert path in {entry["path"] for entry in (await composite.aglob("*.md", path="/graph/nodes/service")).matches}
        assert (await composite.agrep("redis", path=path)).matches
        assert (await composite.awrite(path, "replacement")).error
        assert (await composite.aedit(path, "redis", "replacement")).error
        assert (await composite.aupload_files([(path, b"replacement")]))[0].error
        downloads = await composite.adownload_files([path, "/graph/nodes/service/missing.md"])
        assert downloads[0].content.startswith(b"# service: redis")
        assert downloads[1].error == "file_not_found"
    finally:
        backend.close()
