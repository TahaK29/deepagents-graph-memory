import pytest

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


def test_parse_neighborhood_and_search_paths():
    neighborhood = parse_graph_path("/views/neighborhood/service/langfuse.md")
    search = parse_graph_path("/graph/search/langfuse.md")

    assert neighborhood.kind == "neighborhood"
    assert neighborhood.label == "service"
    assert neighborhood.node_id == "langfuse"
    assert search.kind == "search"
    assert search.query == "langfuse"


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
