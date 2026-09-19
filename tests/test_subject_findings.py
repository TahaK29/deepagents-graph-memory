"""Regression checks for scoped, evidence-backed trace findings."""

import hashlib
from datetime import datetime

import pytest

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.errors import GraphMemoryValidationError
from deepagents_graph_memory.kuzu_store import KuzuGraphStore
from deepagents_graph_memory.stores import GraphNode, valid_finding_link
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


def test_low_level_update_without_subject_is_not_a_valid_finding_link():
    newer = GraphNode("Trace", "newer", {"evidence_refs": [{"source_id": "run", "locator": "log"}]})
    older = GraphNode("Trace", "older")
    assert not valid_finding_link(newer, older, "RESOLVES", reviewed_count=2)


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


def test_structured_refs_support_updates_resolutions_and_dependency_review():
    backend = GraphMemoryBackend.create()
    premise = backend.record_graph_trace(
        trace_id="evidenced-premise",
        situation="test result",
        rationale="read log",
        action="checked",
        outcome="confirmed",
        evidence_refs=[{"source_id": "run-premise", "locator": "logs/premise"}],
    )
    fresh_decision = backend.record_graph_trace(
        trace_id="fresh-decision",
        situation="test result",
        rationale="relied on result",
        action="decided",
        outcome="continue",
        depends_on=[premise],
    )
    fresh_context = backend.recall_graph_memory("continue", anchors=[f"/graph/nodes/Trace/{fresh_decision}.md"])
    assert "Dependency status unknown" not in fresh_context
    common = dict(situation="parser check", rationale="read log", action="ran test", subject="parser@linux", finding_type="state")
    backend.record_graph_trace(trace_id="old", outcome="failed", observed_at="2026-09-19T10:00:00Z", **common)
    backend.record_graph_trace(
        trace_id="new",
        outcome="passed",
        observed_at="2026-09-19T11:00:00Z",
        supersedes=["old"],
        evidence_refs=[{"source_id": "run-new", "locator": "logs/new"}],
        **common,
    )
    decision = backend.record_graph_trace(
        trace_id="decision",
        situation="parser status",
        rationale="old result",
        action="decided",
        outcome="disable parser",
        depends_on=["old"],
    )
    context = backend.recall_graph_memory("disable parser", anchors=[f"/graph/nodes/Trace/{decision}.md"])
    assert "needs recheck" in context
    assert "run-new" in context and "logs/new" in context

    for trace_id, outcome in [("cause-a", "cache"), ("cause-b", "parser")]:
        backend.record_graph_trace(
            trace_id=trace_id,
            situation="investigation",
            rationale="initial guess",
            action="inspected",
            outcome=outcome,
            subject="parser::cause",
        )
    backend.record_graph_trace(
        trace_id="resolution-refs",
        situation="investigation",
        rationale="compared runs",
        action="reran",
        outcome="parser",
        subject="parser::cause",
        resolves=["cause-a", "cause-b"],
        evidence_refs=[{"source_id": "run-resolution", "locator": "logs/resolution"}],
    )
    assert "reviewed in resolution" in backend.recall_graph_memory("cache", anchors=["/graph/nodes/Trace/cause-a.md"])


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


def test_bounded_history_reaches_latest_observation_from_first_failure():
    backend = GraphMemoryBackend.create()
    for index in range(16):
        backend.record_graph_trace(
            trace_id=f"step-{index:02}",
            situation="parser check",
            rationale="test output",
            action="ran test",
            outcome="current-passed" if index == 15 else "previous-failed",
            subject="parser-empty@linux",
            finding_type="state",
            observed_at=f"2026-09-19T10:{index:02}:00Z",
            evidence=[f"run-{index}"],
            supersedes=[f"step-{index - 1:02}"] if index else None,
        )
    context = backend.recall_graph_memory("previous-failed", anchors=["/graph/nodes/Trace/step-00.md"], max_nodes=6, max_edges=20)
    assert "current-passed" in context
    assert "Related findings incomplete" in context


def test_subject_selection_keeps_branches_and_places_undated_findings_last():
    backend = GraphMemoryBackend.create()
    record(backend, "old", "failed", observed_at="2026-09-19T10:00:00Z")
    record(backend, "branch-a", "passed", observed_at="2026-09-19T11:00:00Z", supersedes=["old"])
    record(backend, "branch-b", "flaky", observed_at="2026-09-19T12:00:00Z", supersedes=["old"])
    record(backend, "undated", "unknown")
    subject_id = "subject-" + hashlib.sha256(b"tests/test_parser.py::test_empty@linux").hexdigest()
    selected = backend.store.list_subject_trace_ids(subject_id, limit=4)
    assert selected.items == ["branch-b", "branch-a", "undated", "old"]
    context = backend.recall_graph_memory("failed", anchors=["/graph/nodes/Trace/old.md"], max_nodes=3, max_edges=1)
    assert "flaky" in context and "passed" in context
    assert "Related findings incomplete" in context


def test_subject_selection_is_scoped_and_prioritizes_resolution():
    store = KuzuGraphStore.memory()
    left = GraphMemoryBackend(store, namespace=("left",))
    right = GraphMemoryBackend(store, namespace=("right",))
    for trace_id in ("cache", "parser"):
        left.record_graph_trace(
            trace_id=trace_id,
            situation="incident",
            rationale="logs",
            action="inspected",
            outcome=trace_id,
            subject="incident::cause",
        )
    left.record_graph_trace(
        trace_id="resolution",
        situation="review",
        rationale="rerun",
        action="checked",
        outcome="parser confirmed",
        subject="incident::cause",
        evidence=["rerun log"],
        resolves=["cache", "parser"],
    )
    right.record_graph_trace(
        trace_id="foreign",
        situation="other",
        rationale="logs",
        action="checked",
        outcome="other",
        subject="incident::cause",
    )
    subject_id = "subject-" + hashlib.sha256(b"incident::cause").hexdigest()
    assert store.list_subject_trace_ids(subject_id, scope_key="left", limit=1).items == ["resolution"]
    assert store.list_subject_trace_ids(subject_id, scope_key="right", limit=5).items == ["foreign"]


def test_subject_selection_without_about_relationship_is_empty():
    store = KuzuGraphStore.memory()
    store.add_node("Subject", "subject-orphan")
    store.add_node("Trace", "orphan")
    assert store.list_subject_trace_ids("subject-orphan").items == []


def test_multiple_anchored_subjects_keep_each_subjects_latest():
    backend = GraphMemoryBackend.create()
    for subject in ("subject-a", "subject-b"):
        backend.record_graph_trace(
            trace_id=f"{subject}-old",
            situation="test",
            rationale="log",
            action="ran",
            outcome="failed",
            subject=subject,
            finding_type="state",
            observed_at="2026-09-19T10:00:00Z",
            evidence=["old run"],
        )
        backend.record_graph_trace(
            trace_id=f"{subject}-new",
            situation="test",
            rationale="log",
            action="ran",
            outcome=f"{subject}-passed",
            subject=subject,
            finding_type="state",
            observed_at="2026-09-19T11:00:00Z",
            evidence=["new run"],
            supersedes=[f"{subject}-old"],
        )
    context = backend.recall_graph_memory(
        "test",
        anchors=["/graph/nodes/Trace/subject-a-old.md", "/graph/nodes/Trace/subject-b-old.md"],
        max_nodes=4,
    )
    assert "subject-a-passed" in context
    assert "subject-b-passed" in context
