"""Offline checks for using the Deep Agents filesystem alongside graph context."""

from __future__ import annotations

import asyncio

import deepagents.graph
import pytest
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import PrivateAttr

from deepagents_graph_memory import VFS_TOOL_NAMES, GraphMemoryBackend, graph_context_middleware, graph_memory_tools, register_vgs_harness_profile


class ScriptedModel(BaseChatModel):
    steps: list[tuple[str, dict]]
    _seen: list[list] = PrivateAttr(default_factory=list)
    _tool_names: set[str] = PrivateAttr(default_factory=set)

    @property
    def _llm_type(self) -> str:
        return "combined-context-test"

    def bind_tools(self, tools, **kwargs):
        self._tool_names.update(tool["function"]["name"] if isinstance(tool, dict) and "function" in tool else tool.name for tool in tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen.append(list(messages))
        count = sum(isinstance(message, ToolMessage) for message in messages)
        if count < len(self.steps):
            name, args = self.steps[count]
            message = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"step-{count}"}])
        else:
            message = AIMessage(content="finished")
        return ChatResult(generations=[ChatGeneration(message=message)])


class DelegatingModel(ScriptedModel):
    """Spawn workers dynamically; deliberately reuse tool-call IDs in every worker."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        who = next(message.text for message in messages if isinstance(message, HumanMessage))
        replies = [message for message in messages if isinstance(message, ToolMessage)]
        assert all(not message.text.startswith("Error:") for message in replies)
        if who == "parent" and not replies:
            calls = [
                {"name": "task", "args": {"description": f"worker-{i}", "subagent_type": "general-purpose"}, "id": f"delegate-{i}"} for i in range(7)
            ]
        elif who == "parent" and len(replies) == 7:
            anchors = [f"/graph/nodes/Trace/{reply.text.removeprefix('Recorded graph trace ').removesuffix('.')}.md" for reply in replies]
            calls = [
                {
                    "name": "recall_graph_memory",
                    "args": {"query": "worker findings", "anchors": anchors, "token_budget": 20000, "max_nodes": 200},
                    "id": "recall",
                }
            ]
        elif who != "parent" and len(replies) < 2:
            calls = [
                {
                    "name": "record_graph_trace",
                    "args": {"situation": who, "rationale": "test output", "action": "checked", "outcome": f"result-{len(replies)}"},
                    "id": f"record-{len(replies)}",
                }
            ]
        else:
            if who == "parent":
                assert all(f"worker-{i}" in replies[-1].text for i in range(7))
            calls = []
        content = replies[-1].text if not calls and who != "parent" else "finished"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content if not calls else "", tool_calls=calls))])


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("thread_id", [None, "shared-thread"])
def test_dynamic_workers_get_distinct_stable_identities_in_shared_graph(async_mode, thread_id):
    graph = GraphMemoryBackend.create(namespace="project")
    try:
        # No subagent specs or IDs: the default worker inherits the graph tools.
        agent = create_deep_agent(model=DelegatingModel(steps=[]), tools=graph_memory_tools(graph))
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        previous_workers = set()
        for run in range(2):
            state = {"messages": [HumanMessage(content="parent")]}
            if async_mode:
                asyncio.run(agent.ainvoke(state, config=config))
            else:
                agent.invoke(state, config=config)
            traces = [
                graph.store.get_node("Trace", item, scope_key="project") for item in graph.store.list_node_ids("Trace", scope_key="project").items
            ]
            assert len(traces) == 14 * (run + 1)
            workers = {trace.properties["subagent_id"] for trace in traces}
            assert len(workers - previous_workers) == 7
            for worker_id in workers:
                findings = [trace for trace in traces if trace.properties["subagent_id"] == worker_id]
                assert len(findings) == 2
                assert len({trace.properties["situation"] for trace in findings}) == 1
                assert {trace.properties["outcome"] for trace in findings} == {"result-0", "result-1"}
                assert graph.store.get_node("Subagent", worker_id, scope_key="project") is not None
            previous_workers = workers
    finally:
        graph.close()


def test_registered_graph_only_profile_works_with_real_agent(monkeypatch):
    graph = GraphMemoryBackend.create()
    model = ScriptedModel(steps=[])
    monkeypatch.setattr(deepagents.graph, "resolve_model", lambda _: model)
    register_vgs_harness_profile("test:graph-only-compatibility")
    try:
        agent = create_deep_agent(
            model="test:graph-only-compatibility",
            tools=graph_memory_tools(graph),
            backend=CompositeBackend(default=StateBackend(), routes={"/graph/": graph}),
            system_prompt="Preserve application instructions.",
            memory=["/graph/index.md"],
        )
        agent.invoke({"messages": [HumanMessage(content="Check graph context.")]})
        assert not model._tool_names & VFS_TOOL_NAMES
        assert {"recall_graph_memory", "record_graph_trace"} <= model._tool_names
        prompt = next(message.text for message in model._seen[0] if isinstance(message, SystemMessage))
        assert "Preserve application instructions." in prompt
        assert "Virtual Graph System" in prompt
        assert "Graph Memory" in prompt
        assert "## Filesystem Tools" not in prompt
    finally:
        graph.close()


def test_file_workflow_records_selective_graph_context_and_reads_source():
    graph = GraphMemoryBackend.create(namespace="project")
    model = ScriptedModel(
        steps=[
            ("write_file", {"file_path": "/notes/parser.txt", "content": "Parser failed in run 17."}),
            ("edit_file", {"file_path": "/notes/parser.txt", "old_string": "failed", "new_string": "passed"}),
            ("read_file", {"file_path": "/notes/parser.txt"}),
            (
                "record_graph_trace",
                {
                    "situation": "Parser check in run 17",
                    "rationale": "The captured report changed after recheck",
                    "action": "Inspected report",
                    "outcome": "Parser passed",
                    "subject": "parser-check",
                    "artifacts": ["/notes/parser.txt"],
                    "evidence_refs": [{"source_id": "run-17-report", "locator": "/notes/parser.txt", "revision": "2"}],
                },
            ),
            ("recall_graph_memory", {"query": "Parser check in run 17"}),
            ("read_file", {"file_path": "/notes/parser.txt"}),
        ],
    )
    agent = create_deep_agent(
        model=model,
        tools=graph_memory_tools(graph),
        middleware=[graph_context_middleware()],
        backend=CompositeBackend(default=StateBackend(), routes={"/graph/": graph}),
        system_prompt=SystemMessage(content="Project instructions: inspect the report."),
    )

    result = agent.invoke({"messages": [HumanMessage(content="Check the parser report and record the outcome.")]})
    replies = [message for message in result["messages"] if isinstance(message, ToolMessage)]

    assert {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "recall_graph_memory", "record_graph_trace"} <= model._tool_names
    assert any(
        "Project instructions" in message.text and "Virtual Graph System" in message.text
        for message in model._seen[0]
        if isinstance(message, SystemMessage)
    )
    assert "Parser passed in run 17." in replies[2].text
    assert "Recorded graph trace" in replies[3].text
    assert "Parser passed" in replies[4].text
    assert "Parser passed in run 17." in replies[5].text
    assert "run-17-report" in graph.recall_graph_memory("Parser check in run 17")
    graph.close()


def test_async_combined_agent_keeps_file_and_graph_tools():
    graph = GraphMemoryBackend.create()
    model = ScriptedModel(
        steps=[
            ("write_file", {"file_path": "/notes/async.txt", "content": "async check failed"}),
            ("edit_file", {"file_path": "/notes/async.txt", "old_string": "failed", "new_string": "passed"}),
            ("read_file", {"file_path": "/notes/async.txt"}),
            (
                "record_graph_trace",
                {
                    "situation": "Async check",
                    "rationale": "Saved current report",
                    "action": "Inspected async report",
                    "outcome": "Async check passed",
                    "artifacts": ["/notes/async.txt"],
                    "evidence_refs": [{"source_id": "async-report-1", "locator": "/notes/async.txt", "revision": "2"}],
                },
            ),
            ("recall_graph_memory", {"query": "Async check"}),
            ("read_file", {"file_path": "/notes/async.txt"}),
        ]
    )
    agent = create_deep_agent(model=model, tools=graph_memory_tools(graph), middleware=[graph_context_middleware()])
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="Save and inspect async evidence.")]}))
    replies = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert "async check passed" in replies[2].text
    assert "Recorded graph trace" in replies[3].text
    assert "Async check passed" in replies[4].text
    assert "async check passed" in replies[5].text
    assert "async-report-1" in graph.recall_graph_memory("Async check")
    assert "recall_graph_memory" in model._tool_names and "read_file" in model._tool_names
    graph.close()


def test_large_tool_result_offloads_to_file_without_copying_dump_into_graph():
    graph = GraphMemoryBackend.create()
    report = "Relevant: parser heap spike.\n" + "padding " * 12_000

    @tool
    def fetch_report() -> str:
        """Fetch the long parser report."""
        return report

    model = ScriptedModel(
        steps=[
            ("fetch_report", {}),
            ("read_file", {"file_path": "/large_tool_results/step-0", "limit": 1}),
            (
                "record_graph_trace",
                {
                    "situation": "Parser heap check",
                    "rationale": "The captured report identifies a heap spike",
                    "action": "Inspected offloaded report",
                    "outcome": "Heap spike found",
                    "artifacts": ["/large_tool_results/step-0"],
                    "evidence_refs": [{"source_id": "parser-report-17", "locator": "/large_tool_results/step-0"}],
                },
            ),
        ],
    )
    agent = create_deep_agent(model=model, tools=[fetch_report, *graph_memory_tools(graph)], middleware=[graph_context_middleware()])
    result = agent.invoke({"messages": [HumanMessage(content="Check the report and record the finding.")]})
    replies = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert "/large_tool_results/step-0" in replies[0].text
    assert len(replies[0].text) < len(report) // 2
    assert "Relevant: parser heap spike" in replies[1].text
    recall = graph.recall_graph_memory("Parser heap check")
    assert "Heap spike found" in recall and "parser-report-17" in recall
    assert "padding " not in recall
    graph.close()


def test_native_composite_routes_graph_reads_searches_and_rejects_writes(tmp_path):
    graph = GraphMemoryBackend.create()
    graph.add_graph_node("service", "parser", {"state": "checked"})
    files = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    backend = CompositeBackend(default=files, routes={"/graph/": graph})
    assert backend.write("/notes/source.txt", "source text").error is None
    assert backend.read("/notes/source.txt").file_data["content"] == "source text"
    assert "parser" in backend.read("/graph/nodes/service/parser.md").file_data["content"]
    assert any("/graph/nodes/service/parser.md" == item["path"] for item in backend.glob("**/*.md", path="/graph/").matches)
    assert any("parser" in item["text"] for item in backend.grep("parser", path="/graph/").matches)
    assert backend.write("/graph/nodes/service/parser.md", "wrong").error
    assert backend.edit("/graph/nodes/service/parser.md", "checked", "wrong").error
    assert backend.upload_files([("/graph/extra.md", b"wrong")])[0].error
    assert "checked" in backend.read("/graph/nodes/service/parser.md").file_data["content"]
    graph.close()


def test_configured_inline_subagent_shares_evidence_file_and_trace_with_parent():
    graph = GraphMemoryBackend.create()
    backend = CompositeBackend(default=StateBackend(), routes={"/graph/": graph})
    child = ScriptedModel(
        steps=[
            ("write_file", {"file_path": "/notes/child-report.txt", "content": "Child observed parser green."}),
            (
                "record_graph_trace",
                {
                    "situation": "Child parser check",
                    "rationale": "Report captured",
                    "action": "Checked parser",
                    "outcome": "Parser green",
                    "artifacts": ["/notes/child-report.txt"],
                    "evidence_refs": [{"source_id": "child-report", "locator": "/notes/child-report.txt"}],
                },
            ),
        ],
    )
    parent = ScriptedModel(
        steps=[
            ("task", {"subagent_type": "reporter", "description": "Save the parser report and record its finding."}),
            ("read_file", {"file_path": "/notes/child-report.txt"}),
            ("recall_graph_memory", {"query": "Child parser check"}),
        ],
    )
    agent = create_deep_agent(
        model=parent,
        tools=graph_memory_tools(graph),
        middleware=[graph_context_middleware()],
        backend=backend,
        subagents=[
            {
                "name": "reporter",
                "description": "Save and report parser evidence.",
                "system_prompt": "Inspect parser evidence.",
                "model": child,
                "tools": graph_memory_tools(graph),
                "middleware": [graph_context_middleware()],
            }
        ],
    )
    result = agent.invoke({"messages": [HumanMessage(content="Ask the reporter to check the parser.")]})
    replies = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert "Child observed parser green." in replies[1].text
    assert "Parser green" in replies[2].text
    assert any(
        "Virtual Graph System" in message.text and "Inspect parser evidence." in message.text
        for message in child._seen[0]
        if isinstance(message, SystemMessage)
    )
    assert "record_graph_trace" in child._tool_names and "read_file" in child._tool_names
    graph.close()


def test_routine_file_work_does_not_record_graph_and_changed_source_remains_historical(tmp_path):
    graph = GraphMemoryBackend.create()
    backend = CompositeBackend(default=FilesystemBackend(root_dir=tmp_path, virtual_mode=True), routes={"/graph/": graph})
    model = ScriptedModel(
        steps=[
            ("write_file", {"file_path": "/notes/source.txt", "content": "build passed"}),
            ("read_file", {"file_path": "/notes/source.txt"}),
        ],
    )
    agent = create_deep_agent(model=model, tools=graph_memory_tools(graph), middleware=[graph_context_middleware()], backend=backend)
    agent.invoke({"messages": [HumanMessage(content="Save and inspect the build report.")]})
    assert "No matching graph memory found" in graph.recall_graph_memory("build report")

    graph.record_graph_trace(
        situation="Build report",
        rationale="Captured report",
        action="Read report",
        outcome="Build passed",
        evidence_refs=[{"source_id": "build-1", "locator": "/notes/source.txt", "revision": "1"}],
    )
    historical = graph.recall_graph_memory("Build report")
    assert "Build passed" in historical
    assert backend.edit("/notes/source.txt", "build passed", "build failed").error is None
    assert backend.read("/notes/source.txt").file_data["content"] == "build failed"
    assert graph.recall_graph_memory("Build report") == historical
    (tmp_path / "notes" / "source.txt").unlink()
    assert backend.read("/notes/source.txt").error
    assert graph.recall_graph_memory("Build report") == historical
    graph.close()
