from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.stores import InMemoryGraphStore


def seed_backend():
    store = InMemoryGraphStore()
    backend = GraphMemoryBackend(store)
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
    assert "/views/neighborhood/service/langfuse.md" in paths


def test_grep_searches_graph_metadata():
    backend = seed_backend()

    result = backend.grep("redis", path="/")

    assert result.error is None
    assert any(match["path"] == "/nodes/service/redis.md" for match in result.matches)
