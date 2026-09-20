# - Checks that a fresh backend starts with a temporary Ladybug graph.
# - Cases: the default store, and keeping reset options out of the public API.

from inspect import signature

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.ladybug_store import LadybugGraphStore


def test_create_defaults_to_in_memory_ladybug_backend():
    backend = GraphMemoryBackend.create()

    assert isinstance(backend.store, LadybugGraphStore)
    backend.add_graph_node("File", "temporary")
    another = GraphMemoryBackend.create()
    assert another.store.get_node("File", "temporary") is None
    backend.close()
    another.close()


def test_reset_apis_are_not_exposed():
    parameters = signature(GraphMemoryBackend.create).parameters

    assert "per" + "sist" not in parameters
    assert "path" in parameters
    assert not hasattr(GraphMemoryBackend, "memory")
    assert not hasattr(GraphMemoryBackend, "ephemeral")
    assert not hasattr(GraphMemoryBackend, "local")
    assert not hasattr(GraphMemoryBackend, "from" + "_graph")
    assert not hasattr(GraphMemoryBackend, "clear" + "_graph")
    assert not hasattr(LadybugGraphStore, "local")
    assert not hasattr(LadybugGraphStore, "clear")
