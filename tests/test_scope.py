from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.kuzu_store import KuzuGraphStore


# Lightweight stand-ins for LangChain GraphDocument/Node/Relationship. The store
# only duck-types these (nodes, relationships, type, id, properties, source,
# target), so the suite does not need the sunset langchain-community package.
@dataclass
class Node:
    id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    source: Node
    target: Node
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDocument:
    nodes: list[Node]
    relationships: list[Relationship]
    source: Any = None


def test_scopes_isolate_reads_and_writes():
    store = KuzuGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))

    alice.add_graph_node("service", "langfuse", {"owner": "alice"})
    bob.add_graph_node("service", "redis", {"owner": "bob"})

    assert alice.read("/nodes/service/langfuse.md").error is None
    assert alice.read("/nodes/service/redis.md").file_data is None
    assert bob.read("/nodes/service/redis.md").error is None
    assert bob.read("/nodes/service/langfuse.md").file_data is None


def test_scopes_isolate_shared_public_node_ids():
    store = KuzuGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))

    alice.add_graph_node("service", "shared", {"owner": "alice"})
    bob.add_graph_node("service", "shared", {"owner": "bob"})

    alice_node = alice.read("/nodes/service/shared.md")
    bob_node = bob.read("/nodes/service/shared.md")

    assert alice_node.error is None
    assert bob_node.error is None
    assert "alice" in alice_node.file_data["content"]
    assert "bob" not in alice_node.file_data["content"]
    assert "bob" in bob_node.file_data["content"]
    assert "alice" not in bob_node.file_data["content"]


def test_scoped_label_listing_hides_other_scopes():
    store = KuzuGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))

    bob.add_graph_node("secretlabel", "n1")

    labels = alice.ls("/nodes/")

    assert labels.error is None
    assert labels.entries == []


def test_scoped_node_listing_filters_before_limit():
    store = KuzuGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",), max_nodes=1)
    bob = GraphMemoryBackend(store, namespace=("bob",))

    bob.add_graph_node("service", "bob-1")
    bob.add_graph_node("service", "bob-2")
    bob.add_graph_node("service", "bob-3")
    alice.add_graph_node("service", "alice-1")

    ids = alice.ls("/nodes/service/")

    assert ids.error is None
    assert ids.entries == [{"path": "/nodes/service/alice-1.md", "is_dir": False, "size": 0, "modified_at": ""}]


def test_scoped_search_hides_other_scope_nodes():
    store = KuzuGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))

    bob.add_graph_node("service", "bob-secret", {"description": "scope-only-secret"})

    result = alice.read("/search/scope-only-secret.md")

    assert result.error is None
    assert "bob-secret" not in result.file_data["content"]
    assert "No matching graph facts found." in result.file_data["content"]


def test_scope_metadata_is_written():
    store = KuzuGraphStore.memory()
    backend = GraphMemoryBackend(store, namespace=("alice",))

    backend.add_graph_node("service", "langfuse")

    node = store.get_node("service", "langfuse", scope_key="alice")
    assert node.properties["scope_key"] == "alice"
    assert "created_at" in node.properties
    assert "updated_at" in node.properties


def test_graph_documents_preserve_scope_isolation():
    store = KuzuGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))

    alice_source = Node(id="alice-source", type="Entity", properties={"name": "alice source"})
    alice_target = Node(id="alice-target", type="Entity", properties={"name": "alice target"})
    bob_source = Node(id="bob-source", type="Entity", properties={"name": "bob source"})
    bob_target = Node(id="bob-target", type="Entity", properties={"name": "bob target"})

    alice.add_graph_documents(
        [
            GraphDocument(
                nodes=[alice_source, alice_target],
                relationships=[Relationship(source=alice_source, target=alice_target, type="LINKED_TO", properties={"reason": "alice"})],
                source=Document(page_content="alice graph document"),
            )
        ]
    )
    bob.add_graph_documents(
        [
            GraphDocument(
                nodes=[bob_source, bob_target],
                relationships=[Relationship(source=bob_source, target=bob_target, type="LINKED_TO", properties={"reason": "bob"})],
                source=Document(page_content="bob graph document"),
            )
        ]
    )

    assert alice.read("/nodes/Entity/alice-source.md").error is None
    assert alice.read("/nodes/Entity/bob-source.md").file_data is None
    assert bob.read("/nodes/Entity/bob-source.md").error is None
    assert bob.read("/nodes/Entity/alice-source.md").file_data is None

    alice_node = store.get_node("Entity", "alice-source", scope_key="alice")
    assert alice_node.properties["scope_key"] == "alice"
    assert store.get_node("Entity", "bob-source", scope_key="alice") is None
