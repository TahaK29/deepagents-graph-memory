# - Checks that an agent can browse and find the graph's virtual files.
# - Cases: listing folders and items, matching file names, and searching saved details.

import pytest
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.protocol import GrepResult

from deepagents_graph_memory.backend import GraphMemoryBackend


def seed_backend():
    backend = GraphMemoryBackend.create()
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")
    return backend


def test_ls_root_and_prefixed_root():
    backend = seed_backend()

    root = backend.ls("/")
    prefixed = backend.ls("/graph/")

    assert root.error is None
    assert {entry["path"] for entry in root.entries} >= {"/index.md", "/schema.md", "/nodes/"}
    assert prefixed.error is None
    assert "/graph/index.md" in {entry["path"] for entry in prefixed.entries}


def test_ls_node_labels_and_ids():
    backend = seed_backend()

    labels = backend.ls("/nodes/")
    ids = backend.ls("/nodes/service/")

    assert "/nodes/service/" in {entry["path"] for entry in labels.entries}
    assert "/nodes/service/langfuse.md" in {entry["path"] for entry in ids.entries}
    assert "/nodes/service/redis.md" in {entry["path"] for entry in ids.entries}


def test_glob_matches_virtual_files():
    backend = seed_backend()

    result = backend.glob("**/*.md", path="/")

    paths = {entry["path"] for entry in result.matches}
    assert "/nodes/service/langfuse.md" in paths
    assert not any("/views/" in path for path in paths)
    assert not any("/views/" in entry["path"] for entry in backend.ls("/").entries)


def test_glob_accepts_explicit_none_path():
    backend = seed_backend()

    result = backend.glob("**/*.md", path=None)

    assert result.error is None
    assert "/schema.md" in {entry["path"] for entry in result.matches}


def test_grep_searches_graph_metadata():
    backend = seed_backend()

    result = backend.grep("redis", path="/")

    assert result.error is None
    assert any(match["path"] == "/nodes/service/redis.md" for match in result.matches)


def test_ls_accepts_directories_without_trailing_slashes():
    backend = seed_backend()
    try:
        for path in ("/nodes", "/graph/nodes/service"):
            assert backend.ls(path).entries == backend.ls(path + "/").entries
            assert backend.ls(path).entries
    finally:
        backend.close()


def test_glob_matches_basenames_at_any_depth_and_respects_root_anchors():
    backend = seed_backend()
    try:
        assert {entry["path"] for entry in backend.glob("/*.md").matches} == {"/index.md", "/schema.md"}
        assert {entry["path"] for entry in backend.glob("*.md").matches} == {
            "/index.md",
            "/schema.md",
            "/nodes/service/langfuse.md",
            "/nodes/service/redis.md",
        }
        assert [entry["path"] for entry in backend.glob("redis.md").matches] == ["/nodes/service/redis.md"]
        assert backend.glob("/redis.md").matches == []
        expected = {"/graph/nodes/service/langfuse.md", "/graph/nodes/service/redis.md"}
        assert {entry["path"] for entry in backend.glob("service/*.md", path="/graph/nodes").matches} == expected
        assert {entry["path"] for entry in backend.glob("**/*.md", path="/graph/nodes").matches} == expected
        assert backend.glob("SERVICE/*.md", path="/graph/nodes").matches == []
    finally:
        backend.close()


@pytest.mark.parametrize("pattern", ["../*.md", "..\\*.md", "{a,b}" * 11])
def test_invalid_glob_patterns_return_errors(pattern):
    backend = seed_backend()
    try:
        for result in (backend.glob(pattern), backend.grep("redis", glob=pattern)):
            assert result.error
            assert result.matches is None
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_grep_match_cap_and_composite_async_contract():
    backend = GraphMemoryBackend.create()
    backend.add_graph_node("service", "cache", {"description": "cobalt\ncobalt\ncobalt"})
    composite = CompositeBackend(default=StateBackend(), routes={"/graph/": backend})
    path = "/graph/nodes/service/cache.md"
    try:
        all_matches = backend.grep("cobalt", path=path).matches
        assert len(all_matches) == 3
        for cap in (0, 1, 3, 4):
            direct = backend.grep("cobalt", path=path, max_count=cap)
            if not hasattr(GrepResult, "truncated"):
                # Older upstream APIs have no cap argument or truncation metadata.
                if cap < len(all_matches):
                    assert "max_count" in direct.error and direct.matches is None
                else:
                    assert direct.matches == all_matches
                continue
            routed = await composite.agrep("cobalt", path=path, max_count=cap)
            for result in (direct, routed):
                assert result.error is None
                assert result.matches == all_matches[:cap]
                assert result.truncated is (cap < len(all_matches))
    finally:
        backend.close()


def test_grep_returns_literal_matching_lines_from_rendered_files():
    backend = seed_backend()
    backend.add_graph_node("service", "redis", {"description": "cache outage\ncache restored"})
    path = "/graph/nodes/service/redis.md"
    try:
        result = backend.grep("ache", path=path)
        assert result.error is None
        assert result.matches
        lines = backend.read(path).file_data["content"].splitlines()
        for match in result.matches:
            assert match["path"] == path
            assert "ache" in match["text"] == lines[match["line"] - 1]
        assert backend.grep("CACHE", path=path).matches == []
        assert backend.grep("cache.*", path=path).matches == []
        assert backend.grep("ache", path="/graph/nodes", glob="redis.md").matches == result.matches
        assert backend.grep("ache", path="/graph/nodes", glob="*.txt").matches == []
    finally:
        backend.close()


def test_listing_limits_report_errors_instead_of_silent_missing_files():
    backend = GraphMemoryBackend.create(max_nodes=1)
    backend.add_graph_node("service", "a")
    backend.add_graph_node("service", "b")
    try:
        assert [entry["path"] for entry in backend.glob("/index.md").matches] == ["/index.md"]
        assert {entry["path"] for entry in backend.glob("/*.md").matches} == {"/index.md", "/schema.md"}
        assert "max_nodes" in backend.ls("/nodes/service").error
        assert "max_nodes" in backend.glob("**/*.md").error
        assert "max_nodes" in backend.grep("service").error
        assert backend.grep("service", path="/nodes/service/b.md").matches
    finally:
        backend.close()
