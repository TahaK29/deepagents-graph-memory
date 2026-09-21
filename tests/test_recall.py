# - Checks that a question brings back useful, connected graph context.
# - Cases: matching items or relationships, aliases, starting from a known path, and following several connections;
#        stopping at relevance or size limits, keeping scopes separate, and offering recall as a tool.

import pytest

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.ladybug_store import LadybugGraphStore
from deepagents_graph_memory.tools import graph_memory_tools


def make_backend() -> GraphMemoryBackend:
    backend = GraphMemoryBackend.create()
    backend.add_graph_edge("incident", "incident-123", "AFFECTED", "service", "langfuse")
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "postgres")
    backend.add_graph_edge("service", "redis", "RUNS_ON", "cloud", "aws")
    return backend


def test_recall_direct_node_match():
    backend = make_backend()

    content = backend.recall_graph_memory("langfuse")

    assert "# Graph Memory Recall: langfuse" in content
    assert "[service: langfuse](/graph/nodes/service/langfuse.md)" in content


def test_recall_relationship_match():
    backend = make_backend()

    content = backend.recall_graph_memory("depends")

    assert "DEPENDS_ON" in content
    assert "redis" in content


def test_recall_adaptive_multi_hop_expansion():
    backend = make_backend()

    content = backend.recall_graph_memory("incident 123 depend", max_depth=3)

    assert "incident: incident-123" in content
    assert "AFFECTED" in content
    assert "DEPENDS_ON" in content
    assert "redis" in content


def test_recall_auto_stops_when_next_hop_is_not_relevant():
    backend = make_backend()

    content = backend.recall_graph_memory("langfuse owner", max_depth=3)

    assert "service: langfuse" in content
    assert "RUNS_ON" not in content
    assert "cloud: aws" not in content
    assert "Traversal note:" in content


def test_recall_uses_aliases_as_seed_text():
    backend = GraphMemoryBackend.create()
    backend.add_graph_node(
        "service",
        "auth-service",
        {
            "name": "auth-service",
            "aliases": ["auth", "authentication service", "login service"],
            "description": "Handles login, session refresh, and token validation.",
        },
    )
    backend.record_graph_trace(
        situation="login bug caused session refresh failures",
        rationale="auth-service was returning expired tokens",
        action="patched token refresh handling",
        outcome="login test passed",
        artifacts=["src/auth/session.ts"],
    )

    content = backend.recall_graph_memory("what did we try for the login bug?", mode="deep", max_depth=3)

    assert "service: auth-service" in content
    assert "patched token refresh handling" in content


def test_recall_uses_graph_path_anchor_as_seed():
    backend = make_backend()

    content = backend.recall_graph_memory("what is connected?", anchors=["/graph/nodes/service/langfuse.md"])

    assert "service: langfuse" in content
    assert "DEPENDS_ON" in content
    assert "/graph/views/neighborhood/" not in content


def test_recall_stops_at_edge_budget():
    backend = make_backend()

    content = backend.recall_graph_memory("langfuse depend", max_edges=1)

    assert "Results truncated for edges" in content


@pytest.mark.parametrize("max_nodes", [2, 3])
def test_recall_bounds_endpoints_across_multiple_seeds(max_nodes):
    backend = GraphMemoryBackend.create()
    try:
        backend.add_graph_edge("File", "a", "LINKS", "File", "c")
        backend.add_graph_node("File", "b")
        content = backend.recall_graph_memory("nonmatching", anchors=["/graph/nodes/File/a.md", "/graph/nodes/File/b.md"], max_nodes=max_nodes)
        assert "/graph/nodes/File/a.md" in content
        assert "/graph/nodes/File/b.md" in content
        assert ("/graph/nodes/File/c.md" in content) == (max_nodes == 3)
        assert ("LINKS" in content) == (max_nodes == 3)
        assert ("Results truncated for nodes" in content) == (max_nodes == 2)
    finally:
        backend.close()


def test_recall_stops_at_token_budget():
    backend = make_backend()

    content = backend.recall_graph_memory("langfuse depend", token_budget=20)

    assert "Results truncated for token budget" in content


def test_recall_respects_scope():
    store = LadybugGraphStore.memory()
    alice = GraphMemoryBackend(store, namespace=("alice",))
    bob = GraphMemoryBackend(store, namespace=("bob",))
    alice.add_graph_node("service", "langfuse")
    bob.add_graph_node("service", "redis")

    content = alice.recall_graph_memory("redis")

    assert "No matching graph memory found." in content
    assert "/graph/nodes/service/redis.md" not in content


def test_recall_tool_is_exposed():
    backend = make_backend()
    tools = {tool.name: tool for tool in graph_memory_tools(backend)}

    content = tools["recall_graph_memory"].invoke({"query": "langfuse depend"})

    assert "Graph Memory Recall" in content
    assert "DEPENDS_ON" in content


def test_structured_trace_recall_fits_without_repeating_generated_components():
    backend = GraphMemoryBackend.create()
    trace_id = backend.record_graph_trace(
        operation_id="retry-check-7",
        situation="parser check",
        rationale="captured output",
        action="ran parser test",
        outcome="timeout",
        subject="parser@linux",
        finding_type="state",
        observed_at="2026-09-19T10:00:00Z",
        evidence_refs=[{"source_id": "retry-run", "locator": "logs/retry-run.txt", "summary": "timeout"}],
        created_by_agent="worker-a",
    )
    content = backend.recall_graph_memory("parser", anchors=[f"/graph/nodes/Trace/{trace_id}.md"])
    assert "Results truncated for token budget" not in content
    assert content.count('"outcome": "timeout"') == 1
    assert '"situation": "parser check"' in content
    assert '"rationale": "captured output"' in content
    assert '"created_by_agent": "worker-a"' in content
    assert "CITES" in content and "retry-run" in content
    assert "## Source Paths" not in content
    assert f"[Outcome: {trace_id}-outcome]" not in content


def test_distinct_component_detail_remains_visible_in_structured_recall():
    backend = GraphMemoryBackend.create()
    backend.record_graph_trace(
        trace_id="one",
        situation="check",
        rationale="log",
        action="ran",
        outcome="failed",
        subject="parser@linux",
    )
    backend.add_graph_node("Outcome", "one-outcome", {"inspection_note": "manual review needed"})
    backend.add_graph_edge("Trace", "one", "HAS_OUTCOME", "Outcome", "one-outcome", {"review": "verified independently"})
    content = backend.recall_graph_memory("failed", anchors=["/graph/nodes/Trace/one.md"])
    assert "[Outcome: one-outcome]" in content
    assert "manual review needed" in content
    assert '"review": "verified independently"' in content


def test_two_structured_updates_fit_default_token_budget():
    backend = GraphMemoryBackend.create()
    common = dict(subject="parser@linux", finding_type="state", situation="parser check", rationale="captured output", action="ran test")
    backend.record_graph_trace(
        trace_id="old",
        outcome="failed",
        observed_at="2026-09-19T10:00:00Z",
        evidence=["run one"],
        **common,
    )
    backend.record_graph_trace(
        trace_id="new",
        outcome="passed",
        observed_at="2026-09-19T11:00:00Z",
        evidence=["run two"],
        supersedes=["old"],
        **common,
    )
    content = backend.recall_graph_memory("failed", anchors=["/graph/nodes/Trace/old.md"])
    assert "Results truncated for token budget" not in content
    assert content.count('"outcome": "failed"') == 1
    assert content.count('"outcome": "passed"') == 1
    assert "SUPERSEDES" in content


def test_legacy_trace_shows_full_reasoning_without_repeated_component_nodes():
    backend = GraphMemoryBackend.create()
    backend.record_graph_trace(trace_id="legacy", situation="test failed", rationale="log output", action="reran test", outcome="passed")
    content = backend.recall_graph_memory("test failed", anchors=["/graph/nodes/Trace/legacy.md"])
    assert "Results truncated for token budget" not in content
    assert '"situation": "test failed"' in content
    assert '"rationale": "log output"' in content
    assert '"action": "reran test"' in content
    assert '"outcome": "passed"' in content
    assert "Situation -LED_TO-> Rationale -JUSTIFIED-> Action -PRODUCED-> Outcome" in content
    assert "[Outcome: legacy-outcome]" not in content


def test_low_level_trace_fields_do_not_imply_reasoning_links():
    backend = GraphMemoryBackend.create()
    backend.add_graph_node(
        "Trace",
        "loose",
        {"situation": "check", "rationale": "log", "action": "ran", "outcome": "passed"},
    )
    content = backend.recall_graph_memory("check", anchors=["/graph/nodes/Trace/loose.md"])
    assert '"outcome": "passed"' in content
    assert "LED_TO" not in content and "JUSTIFIED" not in content and "PRODUCED" not in content


@pytest.mark.parametrize("trace_id", ["path/to/trace", "", "x" * 257])
def test_recall_ignores_invalid_trace_ids_in_generic_node_metadata(trace_id):
    backend = GraphMemoryBackend.create()
    try:
        backend.add_graph_node("Artifact", "foreign", {"trace_id": trace_id})
        backend.add_graph_edge("Artifact", "entry", "LINKS", "Artifact", "foreign")
        for node_id in ("foreign", "entry"):
            content = backend.recall_graph_memory("nonmatching", anchors=[f"/graph/nodes/Artifact/{node_id}.md"])
            assert "/graph/nodes/Artifact/foreign.md" in content
        assert backend.store.get_node("Artifact", "foreign").properties["trace_id"] == trace_id
    finally:
        backend.close()


@pytest.mark.parametrize("max_nodes", [100, 2])
def test_explicit_anchors_use_node_budget_and_report_omissions(max_nodes):
    backend = GraphMemoryBackend.create()
    try:
        anchors = []
        for index in range(21):
            backend.add_graph_node("Task", f"anchor{index:02d}")
            anchors.append(f"/graph/nodes/Task/anchor{index:02d}.md")
        content = backend.recall_graph_memory("nonmatching", anchors=anchors, max_nodes=max_nodes, token_budget=10000)
        returned = {path for path in anchors if f"]({path})" in content}
        assert returned == set(anchors[:max_nodes])
        if max_nodes < len(anchors):
            assert "Results truncated for nodes" in content
            assert "Anchor omitted" in content and anchors[-1] in content
        else:
            assert "truncat" not in content.lower()
    finally:
        backend.close()
