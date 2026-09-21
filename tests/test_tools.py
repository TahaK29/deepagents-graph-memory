# - Checks the graph tools an agent receives and how it uses them to save context.
# - Cases: adding items and connections when enabled, and offering recall and work-history tools by default.

import asyncio

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from deepagents_graph_memory import make_graph_subject
from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.errors import GraphMemoryValidationError
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


def test_graph_document_tool_accepts_json_and_rolls_back_invalid_batches():
    backend = GraphMemoryBackend.create(namespace="project")
    tool = next(tool for tool in graph_memory_tools(backend, include_low_level_writes=True) if tool.name == "add_graph_documents")
    source = {"type": "File", "id": "source", "properties": {"name": "source.py"}}
    target = {"type": "File", "id": "target", "properties": {"name": "target.py"}}
    document = {
        "nodes": [source, target],
        "relationships": [{"source": source, "target": target, "type": "DEPENDS_ON", "properties": {"reason": "import"}}],
    }
    try:
        assert tool.invoke({"documents": [document]}).startswith("Added")
        assert backend.store.get_node("File", "source", scope_key="project").properties["name"] == "source.py"
        assert backend.store.get_node("File", "source") is None
        edge = backend.store.get_neighbors("File", "source", scope_key="project").edges[0]
        assert (edge.source_id, edge.relationship, edge.target_id, edge.properties["reason"]) == ("source", "DEPENDS_ON", "target", "import")

        for invalid in ({"type": "File", "id": "bad/id"}, {"type": "File", "id": "bad", "properties": {"scope_key": "other"}}):
            batch = [{"nodes": [{"type": "File", "id": "partial"}], "relationships": []}, {"nodes": [invalid], "relationships": []}]
            assert tool.invoke({"documents": batch}).startswith("Error:")
            assert backend.store.list_node_ids("File", scope_key="project").items == ["source", "target"]
    finally:
        backend.close()


def test_graph_memory_tools_default_to_structured_trace_writes():
    backend = GraphMemoryBackend.create()
    tools = {tool.name: tool for tool in graph_memory_tools(backend)}

    assert set(tools) == {"recall_graph_memory", "record_graph_trace"}


def test_bound_subject_is_shared_and_cannot_be_overridden():
    backend = GraphMemoryBackend.create()
    subject = make_graph_subject("tests/test_auth.py::test_login", "result", "linux")
    workers = [{item.name: item for item in graph_memory_tools(backend, bound_subject=subject)}["record_graph_trace"] for _ in range(2)]
    base = {"rationale": "test output", "action": "checked", "outcome": "failed"}
    assert workers[0].invoke({**base, "situation": "login test failed"}).startswith("Recorded graph trace")
    assert workers[1].invoke({**base, "situation": "auth check failed", "subject": f" {subject} "}).startswith("Recorded graph trace")
    assert len(backend.store.list_node_ids("Trace").items) == 2
    assert len(backend.store.list_node_ids("Subject").items) == 1
    assert workers[0].invoke({**base, "situation": "wrong", "subject": "other"}).startswith("Error: ")
    assert len(backend.store.list_node_ids("Trace").items) == 2
    unbound = {item.name: item for item in graph_memory_tools(backend)}["record_graph_trace"]
    assert unbound.invoke({**base, "situation": "separate", "subject": "other"}).startswith("Recorded graph trace")
    assert len(backend.store.list_node_ids("Subject").items) == 2


def test_bound_subject_is_validated_when_tools_are_created():
    with pytest.raises(GraphMemoryValidationError, match="subject"):
        graph_memory_tools(GraphMemoryBackend.create(), bound_subject=" ")


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

    # Separate executions without a thread must not collapse into one trace.
    first = compiled.invoke(state)["messages"][-1].content
    second = asyncio.run(compiled.ainvoke(state))["messages"][-1].content
    assert first != second
    assert len(backend.store.list_node_ids("Trace").items) == 6


def test_worker_retries_and_nested_workers_keep_runtime_identity():
    backend = GraphMemoryBackend.create()
    trace_tool = next(item for item in graph_memory_tools(backend) if item.name == "record_graph_trace")
    payload = {"situation": "probe", "rationale": "output", "action": "checked", "outcome": "failed"}

    def record(namespace, *, thread="thread-a", call="call-1", **overrides):
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": thread, "checkpoint_ns": namespace}},
            stream_writer=lambda _: None,
            tool_call_id=call,
            store=None,
        )
        return trace_tool.func(**payload, **overrides, runtime=runtime)

    try:
        first = record("tools:worker|tools:write-1")
        assert record("tools:worker|tools:write-1") == first
        assert record("tools:worker|tools:write-2", call="call-2") != first
        assert record("tools:worker|tools:child|tools:write-1") != first
        assert record("tools:worker|tools:write-1", thread="thread-b") != first
        traces = [backend.store.get_node("Trace", item) for item in backend.store.list_node_ids("Trace").items]
        assert len(traces) == 4
        assert len({trace.properties["subagent_id"] for trace in traces}) == 3
        # Explicit names remain supported, but cannot merge workers' automatic retry keys.
        first_named = record("tools:left|tools:write", subagent_id="named-worker", agent_id="parent", run_id="run-1")
        second_named = record("tools:right|tools:write", subagent_id="named-worker", agent_id="parent", run_id="run-1")
        assert first_named != second_named
        node = backend.store.get_node("Trace", first_named.removeprefix("Recorded graph trace ").removesuffix("."))
        assert node.properties["subagent_id"] == "named-worker"
        assert node.properties["agent_id"] == "parent"
        assert node.properties["run_id"] == "run-1"
    finally:
        backend.close()
