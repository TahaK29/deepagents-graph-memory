from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.tools import graph_memory_tools


def test_graph_memory_tools_write_safe_facts():
    backend = GraphMemoryBackend.create()
    tools = {tool.name: tool for tool in graph_memory_tools(backend, include_low_level_writes=True)}

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


def test_graph_memory_tools_default_to_structured_trace_writes():
    backend = GraphMemoryBackend.create()
    tools = {tool.name: tool for tool in graph_memory_tools(backend)}

    assert set(tools) == {"recall_graph_memory", "record_graph_trace"}
