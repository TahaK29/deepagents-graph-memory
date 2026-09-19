# - Checks that the graph remembers what happened, why, what was done, and the result.
# - Cases: saving and reading that story, optional details, multiline text, and finding related artifacts;
#        following the story's connections, looking it up from an artifact, and recording it through a tool.

import hashlib

import pytest

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.errors import GraphMemoryValidationError
from deepagents_graph_memory.kuzu_store import KuzuGraphStore
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


def test_structured_source_is_shared_but_citation_summaries_belong_to_traces():
    backend = GraphMemoryBackend.create()
    base = {"source_id": "run-17", "locator": "logs/run-17.txt", "revision": "a1", "observed_at": "2026-09-19T10:00:00Z"}
    traces = [
        backend.record_graph_trace(
            situation="parser check",
            rationale="read test output",
            action="reported test",
            outcome="failed",
            subject="parser@linux",
            evidence_refs=[{**base, "summary": summary}],
            agent_id=agent,
        )
        for agent, summary in [("worker-a", "parser failed"), ("worker-b", "empty input failed")]
    ]
    source_id = "evidence-source-" + hashlib.sha256(b"run-17").hexdigest()
    source = backend.store.get_node("EvidenceSource", source_id)
    assert source is not None
    assert source.properties["locator"] == "logs/run-17.txt"
    assert source.properties["observed_at"] == "2026-09-19T10:00:00+00:00"
    for trace_id, summary in zip(traces, ("parser failed", "empty input failed"), strict=True):
        edges = backend.store.get_neighbors("Trace", trace_id).edges
        assert any(edge.relationship == "CITES" and edge.target_id == source_id and edge.properties["summary"] == summary for edge in edges)
    separate = backend.record_graph_trace(
        situation="parser check",
        rationale="read test output",
        action="reported test",
        outcome="failed",
        subject="parser@linux",
        evidence_refs=[{"source_id": "run-18", "locator": "logs/run-17.txt", "summary": "parser failed"}],
    )
    assert separate not in traces
    assert backend.store.get_node("EvidenceSource", "evidence-source-" + hashlib.sha256(b"run-18").hexdigest()) is not None
    context = backend.recall_graph_memory("run-17", anchors=[f"/graph/nodes/EvidenceSource/{source_id}.md"], max_nodes=3)
    assert "parser@linux" in context and "cited sources" in context
    assert "logs/run-17.txt" in context
    complete = backend.recall_graph_memory(
        "parser@linux", anchors=[f"/graph/nodes/EvidenceSource/{source_id}.md"], max_nodes=100, max_edges=200, token_budget=10000
    )
    assert "Distinct cited source IDs in returned findings: 2; independence not established." in complete


def test_structured_source_identity_conflict_rolls_back_and_is_scoped():
    store = KuzuGraphStore.memory()
    left = GraphMemoryBackend(store, namespace=("left",))
    right = GraphMemoryBackend(store, namespace=("right",))
    payload = dict(situation="check", rationale="output", action="ran test", outcome="failed")
    left.record_graph_trace(trace_id="first", evidence_refs=[{"source_id": "run-17", "locator": "logs/a", "revision": "a1"}], **payload)
    with pytest.raises(GraphMemoryValidationError, match="conflicting identity"):
        left.record_graph_trace(
            trace_id="conflict",
            evidence_refs=[{"source_id": "a-new", "locator": "logs/new"}, {"source_id": "run-17", "locator": "logs/a", "revision": "a2"}],
            **payload,
        )
    assert store.get_node("Trace", "conflict", scope_key="left") is None
    assert store.get_node("EvidenceSource", "evidence-source-" + hashlib.sha256(b"a-new").hexdigest(), scope_key="left") is None
    right.record_graph_trace(trace_id="other", evidence_refs=[{"source_id": "run-17", "locator": "logs/b"}], **payload)
    assert store.get_node("Trace", "other", scope_key="right") is not None


def test_structured_refs_validate_duplicates_and_retry_fingerprint():
    backend = GraphMemoryBackend.create()
    payload = dict(situation="check", rationale="output", action="ran test", outcome="failed", operation_id="check-17")
    ref = {"source_id": "run-17", "locator": "logs/a"}
    first = backend.record_graph_trace(evidence_refs=[ref, dict(ref)], **payload)
    assert backend.record_graph_trace(evidence_refs=[dict(ref)], **payload) == first
    ordered = [
        {"source_id": "run-18", "locator": "logs/b", "observed_at": "2026-09-19T10:00:00Z"},
        {"source_id": "run-19", "locator": "logs/c"},
    ]
    canonical = backend.record_graph_trace(evidence_refs=ordered, **{**payload, "operation_id": "check-18"})
    reversed_refs = [{"source_id": "run-19", "locator": "logs/c"}, {**ordered[0], "observed_at": "2026-09-19T06:00:00-04:00"}]
    assert backend.record_graph_trace(evidence_refs=reversed_refs, **{**payload, "operation_id": "check-18"}) == canonical
    with pytest.raises(GraphMemoryValidationError, match="different request"):
        backend.record_graph_trace(evidence_refs=[{"source_id": "run-18", "locator": "logs/a"}], **payload)
    for refs in (
        [{"source_id": "run-17"}],
        [{**ref, "unsupported": "x"}],
        [{**ref, "observed_at": "yesterday"}],
        [ref, {**ref, "summary": "different"}],
    ):
        with pytest.raises(GraphMemoryValidationError):
            backend.record_graph_trace(evidence_refs=refs, situation="check", rationale="output", action="ran test", outcome="failed")


def test_record_graph_trace_tool_accepts_structured_refs():
    backend = GraphMemoryBackend.create()
    record = {tool.name: tool for tool in graph_memory_tools(backend)}["record_graph_trace"]
    result = record.invoke(
        {
            "situation": "test failed",
            "rationale": "read log",
            "action": "ran test",
            "outcome": "failed",
            "evidence_refs": [{"source_id": "tool-run", "locator": "logs/tool-run"}],
        }
    )
    assert result.startswith("Recorded graph trace")
    source_id = "evidence-source-" + hashlib.sha256(b"tool-run").hexdigest()
    assert backend.store.get_node("EvidenceSource", source_id) is not None


@pytest.mark.parametrize("malformed", [None, 42])
def test_recall_ignores_malformed_low_level_evidence_refs_when_counting(malformed):
    backend = GraphMemoryBackend.create()
    subject_id = "subject-" + hashlib.sha256(b"question").hexdigest()
    backend.add_graph_node("Subject", subject_id, {"value": "question"})
    backend.add_graph_node("Trace", "malformed", {"subject": "question", "outcome": "unknown", "evidence_refs": malformed})
    backend.add_graph_edge("Trace", "malformed", "ABOUT", "Subject", subject_id)
    context = backend.recall_graph_memory("question", anchors=["/graph/nodes/Trace/malformed.md"], token_budget=10000)
    assert "Trace: malformed" in context
