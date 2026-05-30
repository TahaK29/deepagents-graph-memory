from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.stores import InMemoryGraphStore
from deepagents_graph_memory.tools import graph_memory_tools


def test_graph_memory_tools_write_safe_facts():
    store = InMemoryGraphStore()
    backend = GraphMemoryBackend(store)
    tools = {tool.name: tool for tool in graph_memory_tools(backend)}

    node_result = tools["add_graph_node"].invoke({"label": "service", "node_id": "langfuse", "properties": {"tier": "prod"}})
    edge_result = tools["add_graph_edge"].invoke(
        {
            "source_label": "service",
            "source_id": "langfuse",
            "relationship": "DEPENDS_ON",
            "target_label": "service",
            "target_id": "redis",
            "properties": {},
        }
    )

    assert "Added graph node" in node_result
    assert "Added graph edge" in edge_result
    assert backend.read("/nodes/service/langfuse.md").error is None
