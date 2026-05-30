from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.errors import GraphMemoryValidationError
from deepagents_graph_memory.stores import InMemoryGraphStore


def test_rejects_unsafe_tool_write_labels():
    backend = GraphMemoryBackend(InMemoryGraphStore())

    try:
        backend.add_graph_node("bad-label", "id")
    except GraphMemoryValidationError as exc:
        assert "label" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_rejects_unsafe_relationship_name():
    backend = GraphMemoryBackend(InMemoryGraphStore())

    try:
        backend.add_graph_edge("service", "a", "DROP TABLE", "service", "b")
    except GraphMemoryValidationError as exc:
        assert "relationship" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_rejects_unserializable_properties():
    backend = GraphMemoryBackend(InMemoryGraphStore())

    try:
        backend.add_graph_node("service", "langfuse", {"bad": object()})
    except GraphMemoryValidationError as exc:
        assert "JSON serializable" in str(exc)
    else:
        raise AssertionError("expected validation error")
