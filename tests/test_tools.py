# - Checks the graph tools an agent receives and how it uses them to save context.
# - Cases: adding items and connections when enabled, and offering recall and work-history tools by default.

import asyncio

from langchain_core.messages import AIMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

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


def test_trace_tool_preserves_direct_invocation_and_hides_runtime_from_model():
    backend = GraphMemoryBackend.create()
    trace_tool = {item.name: item for item in graph_memory_tools(backend)}["record_graph_trace"]
    assert "runtime" not in trace_tool.tool_call_schema.model_json_schema()["properties"]
    payload = {"situation": "probe", "rationale": "output", "action": "checked", "outcome": "failed"}
    first = trace_tool.invoke({**payload, "operation_id": "manual-1"})
    second = trace_tool.invoke({**payload, "operation_id": "manual-1"})
    assert first == second
    assert trace_tool.invoke(payload) != trace_tool.invoke(payload)


def test_compiled_toolnode_reuses_tool_call_identity_sync_and_async():
    backend = GraphMemoryBackend.create()
    trace_tool = {item.name: item for item in graph_memory_tools(backend)}["record_graph_trace"]
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([trace_tool]))
    graph.add_edge(START, "tools")
    compiled = graph.compile()
    payload = {"situation": "probe", "rationale": "output", "action": "checked", "outcome": "failed"}
    call = AIMessage(content="", tool_calls=[{"name": "record_graph_trace", "args": payload, "id": "call-1"}])
    state = {"messages": [call]}
    config = {"configurable": {"thread_id": "thread-a"}}

    first = compiled.invoke(state, config=config)["messages"][-1].content
    second = asyncio.run(compiled.ainvoke(state, config=config))["messages"][-1].content
    assert first == second
    assert len(backend.store.list_node_ids("Trace").items) == 1
    compiled.invoke(state, config={"configurable": {"thread_id": "thread-b"}})
    assert len(backend.store.list_node_ids("Trace").items) == 2
    new_call = AIMessage(content="", tool_calls=[{"name": "record_graph_trace", "args": payload, "id": "call-2"}])
    compiled.invoke({"messages": [new_call]}, config=config)
    assert len(backend.store.list_node_ids("Trace").items) == 3
    explicit = {**payload, "operation_id": "external-retry"}
    for call_id in ("call-3", "call-4"):
        call = AIMessage(content="", tool_calls=[{"name": "record_graph_trace", "args": explicit, "id": call_id}])
        compiled.invoke({"messages": [call]}, config=config)
    assert len(backend.store.list_node_ids("Trace").items) == 4
