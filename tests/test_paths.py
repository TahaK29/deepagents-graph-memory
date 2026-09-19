# - Checks that graph file addresses make sense and stay within the graph.
# - Cases: addresses with or without /graph/, connection and search pages, unsafe paths, and invalid names.

import pytest

from deepagents_graph_memory import make_graph_subject
from deepagents_graph_memory.errors import GraphMemoryPathError, GraphMemoryValidationError
from deepagents_graph_memory.paths import parse_graph_path, validate_identifier, validate_node_id


def test_parse_prefixed_and_stripped_paths():
    stripped = parse_graph_path("/nodes/service/langfuse.md")
    prefixed = parse_graph_path("/graph/nodes/service/langfuse.md")

    assert stripped.kind == "node"
    assert stripped.label == "service"
    assert stripped.node_id == "langfuse"
    assert not stripped.had_graph_prefix
    assert prefixed.kind == "node"
    assert prefixed.had_graph_prefix


def test_parse_search_path_and_reject_removed_neighborhood_path():
    search = parse_graph_path("/graph/search/langfuse.md")

    assert search.kind == "search"
    assert search.query == "langfuse"
    with pytest.raises(GraphMemoryPathError):
        parse_graph_path("/views/neighborhood/service/langfuse.md")


def test_rejects_path_traversal():
    with pytest.raises(GraphMemoryPathError):
        parse_graph_path("/graph/nodes/service/../secret.md")


def test_validates_labels_and_relationship_names():
    assert validate_identifier("DEPENDS_ON", field="relationship") == "DEPENDS_ON"
    with pytest.raises(GraphMemoryValidationError):
        validate_identifier("bad-name", field="relationship")


def test_validates_node_ids():
    assert validate_node_id("incident-123") == "incident-123"
    with pytest.raises(GraphMemoryValidationError):
        validate_node_id("../incident-123")


def test_graph_subject_is_canonical_and_unambiguous():
    subject = make_graph_subject(" tests/test_auth.py::test_login ", " result ", " linux ")
    assert subject == '["tests/test_auth.py::test_login","result","linux"]'
    assert subject == make_graph_subject("tests/test_auth.py::test_login", "result", "linux")
    assert subject != make_graph_subject("tests/test_auth.py::test_login", "result", "windows")
    assert subject != make_graph_subject("tests/test_auth.py::test_login", "result", "Linux")
    assert make_graph_subject("a,b", "c", "d") != make_graph_subject("a", "b,c", "d")


@pytest.mark.parametrize(
    "fields", [("", "result", "linux"), ("x", "  ", "linux"), ("x", "result", "\x00linux"), ("x\nkey", "result", "linux"), (1, "result", "linux")]
)
def test_graph_subject_rejects_unsafe_fields(fields):
    with pytest.raises(GraphMemoryValidationError):
        make_graph_subject(*fields)


def test_graph_subject_rejects_long_encoded_key():
    with pytest.raises(GraphMemoryValidationError, match="512"):
        make_graph_subject("x" * 500, "result", "linux")
