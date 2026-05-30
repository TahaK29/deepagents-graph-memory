"""SRE graph memory example data."""

from deepagents_graph_memory import GraphMemoryBackend


def build_sre_graph_backend() -> GraphMemoryBackend:
    """Create a local graph backend seeded with SRE facts.

    Returns:
        Seeded graph memory backend.
    """
    backend = GraphMemoryBackend.local("./sre-graph-memory")
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "redis")
    backend.add_graph_edge("service", "langfuse", "DEPENDS_ON", "service", "postgres")
    backend.add_graph_edge("team", "sre-team", "OWNS", "service", "langfuse")
    backend.add_graph_edge("incident", "incident-123", "AFFECTED", "service", "langfuse")
    backend.add_graph_edge("incident", "incident-123", "RESOLVED_BY", "runbook", "restart-ingestion-workers")
    return backend
