from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.stores import InMemoryGraphStore


def test_scopes_isolate_reads_and_writes():
    store = InMemoryGraphStore()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))

    alice.add_graph_node("service", "langfuse", {"owner": "alice"})
    bob.add_graph_node("service", "redis", {"owner": "bob"})

    assert alice.read("/nodes/service/langfuse.md").error is None
    assert alice.read("/nodes/service/redis.md").file_data is None
    assert bob.read("/nodes/service/redis.md").error is None
    assert bob.read("/nodes/service/langfuse.md").file_data is None


def test_scope_metadata_is_written():
    store = InMemoryGraphStore()
    backend = GraphMemoryBackend(store, namespace=("alice",))

    backend.add_graph_node("service", "langfuse")

    node = store.get_node("service", "langfuse", scope_key="alice")
    assert node.properties["scope_key"] == "alice"
    assert "created_at" in node.properties
    assert "updated_at" in node.properties
