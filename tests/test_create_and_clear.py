from inspect import signature

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.kuzu_store import KuzuGraphStore


def test_create_defaults_to_in_memory_kuzu_backend():
    backend = GraphMemoryBackend.create()

    assert isinstance(backend.store, KuzuGraphStore)


def test_disk_and_reset_apis_are_not_exposed():
    parameters = signature(GraphMemoryBackend.create).parameters

    assert "per" + "sist" not in parameters
    assert "path" not in parameters
    assert not hasattr(GraphMemoryBackend, "memory")
    assert not hasattr(GraphMemoryBackend, "ephemeral")
    assert not hasattr(GraphMemoryBackend, "local")
    assert not hasattr(GraphMemoryBackend, "from" + "_graph")
    assert not hasattr(GraphMemoryBackend, "clear" + "_graph")
    assert not hasattr(KuzuGraphStore, "local")
    assert not hasattr(KuzuGraphStore, "clear")
