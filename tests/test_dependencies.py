"""Trace dependencies keep decisions reviewable when supporting findings change."""

import pytest

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.errors import GraphMemoryValidationError
from deepagents_graph_memory.ladybug_store import LadybugGraphStore
from deepagents_graph_memory.tools import graph_memory_tools


def trace(backend: GraphMemoryBackend, trace_id: str, outcome: str, **kwargs: object) -> str:
    return backend.record_graph_trace(
        trace_id=trace_id,
        situation="checkout probe",
        rationale="recorded evidence",
        action="checked checkout",
        outcome=outcome,
        **kwargs,
    )


def test_changed_premise_flags_direct_and_transitive_decisions_without_reversing_them():
    backend = GraphMemoryBackend.create()
    trace(
        backend, "failed", "checkout failed", subject="checkout@staging", finding_type="state", observed_at="2026-09-19T10:00:00Z", evidence=["run 1"]
    )
    trace(backend, "decision", "disable checkout", depends_on=["failed"])
    trace(backend, "followup", "notify operator", depends_on=["decision"])
    before = backend.recall_graph_memory("notify operator", anchors=["/graph/nodes/Trace/followup.md"])
    assert "needs recheck" not in before

    trace(
        backend,
        "passed",
        "checkout passed",
        subject="checkout@staging",
        finding_type="state",
        observed_at="2026-09-19T11:00:00Z",
        evidence=["run 2"],
        supersedes=["failed"],
    )
    after = backend.recall_graph_memory("notify operator", anchors=["/graph/nodes/Trace/followup.md"])
    assert "needs recheck" in after
    assert "followup" in after and "decision" in after and "failed" in after and "passed" in after
    assert "SUPERSEDES" in after and "BASED_ON" in after
    assert backend.store.get_node("Trace", "decision").properties["outcome"] == "disable checkout"

    trace(backend, "rechecked", "enable checkout", depends_on=["passed"])
    current = backend.recall_graph_memory("enable checkout", anchors=["/graph/nodes/Trace/rechecked.md"])
    assert "rechecked" in current
    assert "Trace rechecked needs recheck" not in current


def test_dependencies_validate_and_retry_by_identity_atomically():
    store = LadybugGraphStore.memory()
    left = GraphMemoryBackend(store, namespace=("left",))
    right = GraphMemoryBackend(store, namespace=("right",))
    trace(left, "premise", "failed")
    trace(right, "foreign", "failed")
    for dependency in (["missing"], ["foreign"], ["bad/id"], [1]):
        with pytest.raises(GraphMemoryValidationError):
            trace(left, "invalid", "decision", depends_on=dependency)
        assert store.get_node("Trace", "invalid", scope_key="left") is None

    kwargs = dict(situation="checkout", rationale="probe", action="decided", outcome="disable", operation_id="decision-op")
    first = left.record_graph_trace(**kwargs, depends_on=["premise", "premise"])
    assert left.record_graph_trace(**kwargs, depends_on=["premise"]) == first
    with pytest.raises(GraphMemoryValidationError):
        left.record_graph_trace(**kwargs, depends_on=[])
    links = store.get_neighbors("Trace", first, scope_key="left", max_edges=30)
    assert len([edge for edge in links.edges if edge.relationship == "BASED_ON"]) == 1


def test_incomplete_or_cyclic_dependency_status_is_unknown_even_with_tiny_output_budget():
    backend = GraphMemoryBackend.create()
    for index in range(4):
        trace(backend, f"premise-{index}", f"result {index}", evidence=[f"run {index}"])
    trace(backend, "decision", "mitigate", depends_on=[f"premise-{index}" for index in range(4)])
    tiny = backend.recall_graph_memory("mitigate", anchors=["/graph/nodes/Trace/decision.md"], max_nodes=2, max_edges=1, token_budget=1)
    assert tiny.startswith("Dependency status unknown")

    backend.add_graph_edge("Trace", "premise-0", "BASED_ON", "Trace", "decision")
    cyclic = backend.recall_graph_memory("mitigate", anchors=["/graph/nodes/Trace/decision.md"])
    assert "Dependency status unknown" in cyclic


def test_trace_tool_accepts_dependencies():
    backend = GraphMemoryBackend.create()
    trace(backend, "premise", "failed")
    tool = {item.name: item for item in graph_memory_tools(backend)}["record_graph_trace"]
    result = tool.invoke({"situation": "failure", "rationale": "probe", "action": "decided", "outcome": "disable", "depends_on": ["premise"]})
    assert result.startswith("Recorded graph trace")
    trace_id = result.removeprefix("Recorded graph trace ").removesuffix(".")
    links = backend.store.get_neighbors("Trace", trace_id, max_edges=30)
    assert any(edge.relationship == "BASED_ON" and edge.target_id == "premise" for edge in links.edges)


def test_resolution_flags_only_its_reviewed_premises():
    backend = GraphMemoryBackend.create()
    for trace_id in ("cache", "parser", "network"):
        trace(backend, trace_id, trace_id, subject="incident::cause", evidence=[f"log {trace_id}"])
    trace(backend, "cache-decision", "disable cache", depends_on=["cache"])
    trace(backend, "network-decision", "restart network", depends_on=["network"])
    trace(backend, "resolution", "parser confirmed", subject="incident::cause", evidence=["rerun"], resolves=["cache", "parser"])
    reviewed = backend.recall_graph_memory("disable cache", anchors=["/graph/nodes/Trace/cache-decision.md"])
    unrelated = backend.recall_graph_memory("restart network", anchors=["/graph/nodes/Trace/network-decision.md"])
    assert "Trace cache-decision needs recheck" in reviewed
    assert "RESOLVES" in reviewed and "resolution" in reviewed
    assert "Trace network-decision needs recheck" not in unrelated


def test_component_anchor_reviews_returned_decision_and_legacy_evidence():
    backend = GraphMemoryBackend.create()
    trace(backend, "failed", "failed", evidence=["probe run 1"], subject="checkout@staging", finding_type="state", observed_at="2026-09-19T10:00:00Z")
    trace(backend, "decision", "disable", depends_on=["failed"], task_id="checkout-task")
    trace(
        backend,
        "passed",
        "passed",
        evidence=["probe run 2"],
        subject="checkout@staging",
        finding_type="state",
        observed_at="2026-09-19T11:00:00Z",
        supersedes=["failed"],
    )
    result = backend.recall_graph_memory("checkout-task", anchors=["/graph/nodes/Task/checkout-task.md"])
    assert "Trace decision needs recheck" in result
    assert "Dependency status unknown" not in result


def test_depth_limit_marks_transitive_dependency_unknown():
    backend = GraphMemoryBackend.create()
    trace(backend, "premise", "failed", evidence=["probe run"])
    trace(backend, "middle", "mitigate", depends_on=["premise"])
    trace(backend, "decision", "notify", depends_on=["middle"])
    shallow = backend.recall_graph_memory("notify", anchors=["/graph/nodes/Trace/decision.md"], max_depth=1, token_budget=1)
    assert shallow.startswith("Dependency status unknown")
    complete = backend.recall_graph_memory("notify", anchors=["/graph/nodes/Trace/decision.md"], max_depth=3)
    assert "Trace decision: dependency status unknown" not in complete


def test_component_owner_is_reviewed_when_trace_node_does_not_fit():
    backend = GraphMemoryBackend.create()
    trace(backend, "premise", "failed", evidence=["probe run"])
    trace(backend, "decision", "disable", depends_on=["premise"])
    result = backend.recall_graph_memory("disable", anchors=["/graph/nodes/Outcome/decision-outcome.md"], max_nodes=1, max_edges=1, token_budget=1)
    assert result.startswith("Dependency status unknown")
