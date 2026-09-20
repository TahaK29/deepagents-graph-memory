# - Checks that the backend and real Ladybug database work together.
# - Case: save a dependency, open its item page, and find that connection through recall.

from deepagents_graph_memory.backend import GraphMemoryBackend


def test_ladybug_memory_backend_round_trip():
    backend = GraphMemoryBackend.create()

    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")

    node = backend.read("/nodes/service/langfuse.md")
    recall = backend.recall_graph_memory("langfuse depend")

    assert node.error is None
    assert "langfuse" in node.file_data["content"]
    assert "DEPENDS_ON" in recall


def test_native_metadata_preserves_labels_relationships_and_json_properties():
    backend = GraphMemoryBackend.create(namespace="project")
    try:
        backend.add_graph_node("Service", "api", properties={"config": {"retries": 3}, "aliases": ["gateway"]})
        backend.add_graph_edge("Service", "api", "DEPENDS_ON", "Database", "storage", properties={"evidence": {"run": 7}})
        node = backend.store.get_node("Service", "api", scope_key="project")
        assert node.label == "Service"
        assert node.properties["config"] == {"retries": 3}
        assert node.properties["aliases"] == ["gateway"]
        edge = backend.store.get_neighbors("Service", "api", scope_key="project").edges[0]
        assert (edge.source_label, edge.source_id, edge.relationship, edge.target_label, edge.target_id) == (
            "Service",
            "api",
            "DEPENDS_ON",
            "Database",
            "storage",
        )
        assert edge.properties["evidence"] == {"run": 7}
        assert not any(key.startswith("_") for key in node.properties | edge.properties)
    finally:
        backend.close()
