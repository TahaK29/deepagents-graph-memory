from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.tools import graph_memory_tools


def test_ephemeral_backend_records_reasoning_trace():
    backend = GraphMemoryBackend.create()

    trace_id = backend.record_graph_trace(
        situation="scope leak test failed because Alice could read Bob's graph node",
        rationale="all graph reads must use the active scope_key",
        action="edited backend.py and stores.py to pass scope_key through reads",
        outcome="pytest passed for scope isolation",
        artifacts=["src/deepagents_graph_memory/backend.py", "src/deepagents_graph_memory/stores.py"],
        evidence=["pytest tests/test_scope.py passed"],
        run_id="run-1",
        agent_id="agent-main",
        subagent_id="subagent-debugger",
        task_id="task-scope-fix",
    )

    trace = backend.read(f"/nodes/Trace/{trace_id}.md")
    assert trace.error is None
    content = trace.file_data["content"]
    assert "# Trace:" in content
    assert "Situation" in content
    assert "Rationale" in content
    assert "Action" in content
    assert "Outcome" in content

    node = backend.store.get_node("Trace", trace_id)
    assert node is not None
    assert node.properties["run_id"] == "run-1"
    assert node.properties["subagent_id"] == "subagent-debugger"


def test_record_graph_trace_omits_unset_optional_metadata():
    backend = GraphMemoryBackend.create()

    trace_id = backend.record_graph_trace(
        situation="trace captured a useful observation",
        rationale="the result should be reusable",
        action="recorded the trace",
        outcome="trace was stored",
    )

    trace = backend.read(f"/nodes/Trace/{trace_id}.md")

    assert trace.error is None
    content = trace.file_data["content"]
    assert "agent_id" not in content
    assert "run_id" not in content
    assert "subagent_id" not in content
    assert "task_id" not in content


def test_record_graph_trace_accepts_multiline_trace_text():
    backend = GraphMemoryBackend.create()

    trace_id = backend.record_graph_trace(
        situation="schema discovery",
        rationale="DDL has the compact column list",
        action="read DDL.csv",
        outcome="Knowledge state changed.\nBIKESHARE_TRIPS.start_station_id NUMBER\nBIKESHARE_STATION_INFO.station_id VARCHAR",
    )

    trace = backend.read(f"/nodes/Trace/{trace_id}.md")

    assert trace.error is None
    assert "BIKESHARE_TRIPS.start_station_id" in trace.file_data["content"]


def test_trace_recall_follows_level_three_path():
    backend = GraphMemoryBackend.create()
    backend.record_graph_trace(
        situation="sheep saw lion near the path",
        rationale="lion is dangerous and nearby",
        action="sheep ran away",
        outcome="sheep survived",
        evidence=["previous lion encounter ended badly"],
    )

    content = backend.recall_graph_memory("why did sheep run from lion", mode="deep", max_depth=3)

    assert "Situation" in content
    assert "Rationale" in content
    assert "Action" in content
    assert "Outcome" in content
    assert "LED_TO" in content
    assert "JUSTIFIED" in content
    assert "PRODUCED" in content


def test_trace_search_finds_artifacts():
    backend = GraphMemoryBackend.create()
    backend.record_graph_trace(
        situation="backend.py had a failing scope test",
        rationale="the backend read path did not preserve scope",
        action="patched backend.py",
        outcome="scope test passed",
        artifacts=["src/deepagents_graph_memory/backend.py"],
    )

    result = backend.read("/search/backend.py.md")

    assert result.error is None
    assert "backend.py" in result.file_data["content"]


def test_trace_recall_can_start_from_anchor():
    backend = GraphMemoryBackend.create()
    backend.record_graph_trace(
        situation="backend.py had a failing scope test",
        rationale="the backend read path did not preserve scope",
        action="patched backend.py",
        outcome="scope test passed",
        artifacts=["src/deepagents_graph_memory/backend.py"],
    )

    content = backend.recall_graph_memory("what happened here?", anchors=["backend.py"], mode="deep", max_depth=3)

    assert "patched backend.py" in content
    assert "scope test passed" in content


def test_record_graph_trace_tool_is_exposed():
    backend = GraphMemoryBackend.create()
    tools = {tool.name: tool for tool in graph_memory_tools(backend)}

    result = tools["record_graph_trace"].invoke(
        {
            "situation": "sheep saw lion",
            "rationale": "lion is dangerous",
            "action": "sheep ran away",
            "outcome": "sheep survived",
            "artifacts": [],
            "evidence": ["lion was nearby"],
        }
    )

    assert "Recorded graph trace" in result
    assert "sheep" in backend.recall_graph_memory("sheep lion")
