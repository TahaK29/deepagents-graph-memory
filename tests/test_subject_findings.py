"""Regression checks for scoped, evidence-backed trace findings."""

import hashlib
from datetime import datetime

import pytest

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.errors import GraphMemoryValidationError
from deepagents_graph_memory.kuzu_store import KuzuGraphStore
from deepagents_graph_memory.tools import graph_memory_tools


def record(backend: GraphMemoryBackend, trace_id: str, outcome: str, **kwargs: object) -> str:
    return backend.record_graph_trace(
        trace_id=trace_id,
        situation="parser test on linux at a recorded revision",
        rationale="test output is evidence",
        action="ran parser test",
        outcome=outcome,
        subject="tests/test_parser.py::test_empty@linux",
        finding_type="state",
        evidence=[f"pytest output: {outcome}"],
        **kwargs,
    )


def test_observed_time_and_supersession_preserve_history_and_branches():
    backend = GraphMemoryBackend.create()
    record(backend, "old", "failed", observed_at="2026-09-19T10:00:00-04:00")
    record(backend, "new", "passed", observed_at="2026-09-19T15:00:00Z", supersedes=["old"])
    record(backend, "branch", "flaky", observed_at="2026-09-19T15:30:00Z", supersedes=["old"])

    old = backend.store.get_node("Trace", "old")
    assert old is not None
    assert old.properties["observed_at"] == "2026-09-19T14:00:00+00:00"
    assert old.properties["outcome"] == "failed"
    assert datetime.fromisoformat(old.properties["recorded_at"]).tzinfo is not None
    assert backend.store.get_node("Trace", "new").properties["observed_at"] == "2026-09-19T15:00:00+00:00"

    subject_id = hashlib.sha256(b"tests/test_parser.py::test_empty@linux").hexdigest()
    for anchor in [
        "/graph/nodes/Trace/old.md",
        "/graph/nodes/Outcome/old-outcome.md",
        f"/graph/nodes/Subject/subject-{subject_id}.md",
    ]:
        context = backend.recall_graph_memory("failed", anchors=[anchor], max_nodes=30, max_edges=40)
        assert '"outcome": "passed"' in context
        assert '"outcome": "flaky"' in context
        assert "SUPERSEDES" in context
        assert "pytest output: passed" in context
        assert "recorded_at" in context
        assert "source" in context
        assert context.index("Trace: new") < context.index("Trace: old")
        assert "superseded by [Trace: branch]" in context
        assert "[Trace: new](/graph/nodes/Trace/new.md); observed_at" in context


def test_invalid_supersession_is_atomic_and_delayed_report_cannot_win():
    backend = GraphMemoryBackend.create()
    record(backend, "old", "failed", observed_at="2026-09-19T10:00:00Z")
    record(backend, "undated", "unknown")
    backend.record_graph_trace(
        trace_id="explanation",
        situation="same test",
        rationale="maybe cache",
        action="reasoned",
        outcome="possible cause",
        subject="tests/test_parser.py::test_empty@linux",
        finding_type="interpretation",
        observed_at="2026-09-19T09:00:00Z",
    )
    for trace_id, kwargs in [
        ("equal", {"observed_at": "2026-09-19T10:00:00Z", "supersedes": ["old"]}),
        ("delayed", {"observed_at": "2026-09-19T09:00:00Z", "supersedes": ["old"]}),
        ("missing", {"observed_at": "2026-09-19T11:00:00Z", "supersedes": ["unknown"]}),
        ("mixed", {"observed_at": "2026-09-19T11:00:00Z", "supersedes": ["old", "unknown"]}),
        ("undated-target", {"observed_at": "2026-09-19T11:00:00Z", "supersedes": ["undated"]}),
        ("explanation-target", {"observed_at": "2026-09-19T11:00:00Z", "supersedes": ["explanation"]}),
    ]:
        with pytest.raises(GraphMemoryValidationError):
            record(backend, trace_id, "passed", **kwargs)
        assert backend.store.get_node("Trace", trace_id) is None
    with pytest.raises(GraphMemoryValidationError):
        backend.record_graph_trace(
            trace_id="interpretation",
            situation="same",
            rationale="maybe",
            action="checked",
            outcome="different",
            subject="tests/test_parser.py::test_empty@linux",
            observed_at="2026-09-19T11:00:00Z",
            supersedes=["old"],
            finding_type="interpretation",
            evidence=["guess"],
        )


def test_validation_and_scoping_and_old_api():
    store = KuzuGraphStore.memory()
    left = GraphMemoryBackend(store, namespace=("left",))
    right = GraphMemoryBackend(store, namespace=("right",))
    record(left, "old", "failed", observed_at="2026-09-19T10:00:00Z")
    record(right, "other", "passed", observed_at="2026-09-19T11:00:00Z")
    assert "other" not in left.recall_graph_memory("failed", anchors=["/graph/nodes/Trace/old.md"])
    with pytest.raises(GraphMemoryValidationError):
        record(right, "cross", "passed", observed_at="2026-09-19T12:00:00Z", supersedes=["old"])
    for trace_id, kwargs in [
        ("naive", {"observed_at": "2026-09-19T12:00:00"}),
        ("malformed", {"observed_at": "not a date"}),
        ("override", {"observed_at": "2026-09-19T12:00:00Z", "recorded_at": "2000-01-01T00:00:00Z"}),
        ("kind", {"finding_type": []}),
        ("subject", {"subject": " "}),
    ]:
        with pytest.raises(GraphMemoryValidationError):
            left.record_graph_trace(
                trace_id=trace_id,
                situation="one",
                rationale="two",
                action="three",
                outcome="four",
                **kwargs,
            )
        assert left.store.get_node("Trace", trace_id, scope_key="left") is None
    legacy = left.record_graph_trace(situation="one", rationale="two", action="three", outcome="four")
    assert "subject" not in left.store.get_node("Trace", legacy, scope_key="left").properties


def test_artifact_and_evidence_anchors_and_budget_warning():
    backend = GraphMemoryBackend.create()
    record(backend, "old", "failed", observed_at="2026-09-19T10:00:00Z", artifacts=["parser.py"])
    record(backend, "new", "passed", observed_at="2026-09-19T11:00:00Z", supersedes=["old"])
    for anchor in ["parser.py", "pytest output: failed"]:
        context = backend.recall_graph_memory("failed", anchors=[anchor], max_nodes=30, max_edges=40)
        assert '"outcome": "passed"' in context
    tiny = backend.recall_graph_memory("failed", anchors=["/graph/nodes/Trace/old.md"], max_nodes=1, max_edges=1, token_budget=1)
    assert tiny.startswith("Related findings incomplete")
    assert "no resolved/current answer" in tiny


def test_task_anchor_discovers_other_subject_findings_in_local_mode():
    backend = GraphMemoryBackend.create()
    record(backend, "first", "failed", observed_at="2026-09-19T10:00:00Z", task_id="task-first")
    record(backend, "second", "passed", observed_at="2026-09-19T11:00:00Z", task_id="task-second")
    context = backend.recall_graph_memory("task-first", anchors=["/graph/nodes/Task/task-first.md"], mode="local")
    assert '"outcome": "failed"' in context
    assert '"outcome": "passed"' in context


def test_tool_accepts_subject_fields():
    backend = GraphMemoryBackend.create()
    tool = {item.name: item for item in graph_memory_tools(backend)}["record_graph_trace"]
    result = tool.invoke(
        {
            "situation": "test failed",
            "rationale": "output",
            "action": "ran test",
            "outcome": "failed",
            "subject": "parser@linux",
            "observed_at": "2026-09-19T10:00:00Z",
            "finding_type": "state",
            "evidence": ["exit 1"],
        }
    )
    assert result.startswith("Recorded graph trace")


def test_explicit_resolution_reviews_competing_interpretations_without_deleting_them():
    store = KuzuGraphStore.memory()
    backend = GraphMemoryBackend(store, namespace=("project",))
    other = GraphMemoryBackend(store, namespace=("other",))
    for trace_id, outcome in [("cache", "cache caused failure"), ("parser", "parser caused failure")]:
        backend.record_graph_trace(
            trace_id=trace_id,
            situation="same incident",
            rationale="initial analysis",
            action="inspected logs",
            outcome=outcome,
            subject="incident-7::cause",
        )
    other.record_graph_trace(
        trace_id="foreign",
        situation="other incident",
        rationale="initial analysis",
        action="inspected logs",
        outcome="different",
        subject="incident-7::cause",
    )
    for trace_id, targets, evidence in [
        ("one", ["cache"], ["rerun"]),
        ("no-evidence", ["cache", "parser"], []),
        ("cross", ["cache", "foreign"], ["rerun"]),
    ]:
        with pytest.raises(GraphMemoryValidationError):
            backend.record_graph_trace(
                trace_id=trace_id,
                situation="review",
                rationale="checked both explanations",
                action="reran test",
                outcome="parser caused failure",
                subject="incident-7::cause",
                resolves=targets,
                evidence=evidence,
            )
        assert store.get_node("Trace", trace_id, scope_key="project") is None
    backend.record_graph_trace(
        trace_id="resolution",
        situation="same incident after rerun",
        rationale="cache cleared but parser still failed",
        action="reproduced with cache disabled",
        outcome="parser caused failure",
        subject="incident-7::cause",
        resolves=["cache", "parser"],
        evidence=["cache disabled; parser test exit 1"],
    )
    context = backend.recall_graph_memory("cache caused failure", anchors=["/graph/nodes/Trace/cache.md"])
    assert "RESOLVES" in context
    assert "reviewed in resolution [Trace: resolution]" in context
    assert context.index("Trace: resolution") < context.index("Trace: cache")
    assert store.get_node("Trace", "cache", scope_key="project").properties["outcome"] == "cache caused failure"
