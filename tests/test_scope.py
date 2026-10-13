# - Checks that separate scopes keep their graph context to themselves.
# - Cases: separate reads and writes even with shared item names, filtered listings and search;
#        saving scope details and keeping imported graph documents separate.

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.ladybug_store import LadybugGraphStore


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
    store = LadybugGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))

    alice.add_graph_node("service", "langfuse", {"owner": "alice"})
    bob.add_graph_node("service", "redis", {"owner": "bob"})

    assert store.get_node("service", "langfuse", scope_key="alice") is not None
    assert store.get_node("service", "redis", scope_key="alice") is None
    assert store.get_node("service", "redis", scope_key="bob") is not None
    assert store.get_node("service", "langfuse", scope_key="bob") is None


def test_scopes_isolate_shared_public_node_ids():
    store = LadybugGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))

    alice.add_graph_node("service", "shared", {"owner": "alice"})
    bob.add_graph_node("service", "shared", {"owner": "bob"})

    alice_node = store.get_node("service", "shared", scope_key="alice")
    bob_node = store.get_node("service", "shared", scope_key="bob")

    assert alice_node.properties["owner"] == "alice"
    assert bob_node.properties["owner"] == "bob"


def test_scoped_label_listing_hides_other_scopes():
    store = LadybugGraphStore.memory()
    bob = GraphMemoryBackend(store, namespace=("bob",))

    bob.add_graph_node("secretlabel", "n1")

    labels = store.list_labels(scope_key="alice")

    assert labels.items == []


def test_scoped_node_listing_filters_before_limit():
    store = LadybugGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))

    bob.add_graph_node("service", "bob-1")
    bob.add_graph_node("service", "bob-2")
    bob.add_graph_node("service", "bob-3")
    alice.add_graph_node("service", "alice-1")

    ids = store.list_node_ids("service", scope_key="alice", limit=1)

    assert ids.items == ["alice-1"]
    assert not ids.truncated


def test_scoped_search_hides_other_scope_nodes():
    store = LadybugGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))

    bob.add_graph_node("service", "bob-secret", {"description": "scope-only-secret"})

    result = alice.recall_graph_memory("scope-only-secret")

    assert "bob-secret" not in result
    assert "No matching graph memory found." in result


def test_scope_metadata_is_written():
    store = LadybugGraphStore.memory()
    backend = GraphMemoryBackend(store, namespace=("alice",))

    backend.add_graph_node("service", "langfuse")

    node = store.get_node("service", "langfuse", scope_key="alice")
    assert node.properties["scope_key"] == "alice"
    assert "created_at" in node.properties
    assert "updated_at" in node.properties


def test_graph_documents_preserve_scope_isolation():
    store = LadybugGraphStore.memory()
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

    assert store.get_node("Entity", "alice-source", scope_key="alice") is not None
    assert store.get_node("Entity", "bob-source", scope_key="alice") is None
    assert store.get_node("Entity", "bob-source", scope_key="bob") is not None
    assert store.get_node("Entity", "alice-source", scope_key="bob") is None

    alice_node = store.get_node("Entity", "alice-source", scope_key="alice")
    assert alice_node.properties["scope_key"] == "alice"
    assert store.get_node("Entity", "bob-source", scope_key="alice") is None
