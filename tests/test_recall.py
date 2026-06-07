from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.kuzu_store import KuzuGraphStore
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
    assert "`/graph/nodes/service/langfuse.md`" in content


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


def test_recall_stops_at_edge_budget():
    backend = make_backend()

    content = backend.recall_graph_memory("langfuse depend", max_edges=1)

    assert "Results truncated for edges" in content


def test_recall_stops_at_token_budget():
    backend = make_backend()

    content = backend.recall_graph_memory("langfuse depend", token_budget=20)

    assert "Results truncated for token budget" in content


def test_recall_respects_scope():
    store = KuzuGraphStore.memory()
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
