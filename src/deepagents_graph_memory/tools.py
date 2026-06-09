"""Safe graph memory tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import tool

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.errors import GraphMemoryError
from deepagents_graph_memory.recall import RecallMode


def graph_memory_tools(graph_backend: GraphMemoryBackend, *, include_low_level_writes: bool = False) -> list[Any]:
    """Create safe graph memory recall and write tools.

    Args:
        graph_backend: Graph memory backend to mutate.
        include_low_level_writes: When true, expose generic node, edge, and
            graph-document write tools. Keep this false for unconstrained agent
            use; prefer domain-specific tools or `record_graph_trace`.

    Returns:
        LangChain tool objects for controlled graph writes.
    """

    @tool
    def recall_graph_memory(
        query: str,
        anchors: list[str] | None = None,
        mode: RecallMode = "auto",
        token_budget: int = 2000,
        max_depth: int = 3,
        max_nodes: int = 50,
        max_edges: int = 100,
    ) -> str:
        """Recall relevant graph memory: entities, relationships, prior reasoning traces, and connected context.

        When reporting state changes from retrieved traces, distinguish what actually changed from what the agent did in response.
        A switch in tool or action is not the same as the underlying condition changing.
        If the same outcome recurs across multiple steps, the state changed once when it first appeared and stayed unchanged afterward.
        """
        try:
            return graph_backend.recall_graph_memory(
                query,
                anchors=anchors,
                mode=mode,
                token_budget=token_budget,
                max_depth=max_depth,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )
        except GraphMemoryError as exc:
            return f"Error: {exc}"

    @tool
    def add_graph_node(label: str, node_id: str, properties: dict[str, Any] | None = None) -> str:
        """Add or update a graph node.

        Use this for entity memory such as any named thing in the workflow's domain.
        """
        try:
            graph_backend.add_graph_node(label, node_id, properties, source="graph_memory_tool")
        except GraphMemoryError as exc:
            return f"Error: {exc}"
        return f"Added graph node {label}/{node_id}."

    @tool
    def add_graph_edge(
        source_label: str,
        source_id: str,
        relationship: str,
        target_label: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Add or update a directed graph relationship.

        Use this for relationship memory such as any directed connection between two entities.
        """
        try:
            graph_backend.add_graph_edge(
                source_label,
                source_id,
                relationship,
                target_label,
                target_id,
                properties,
                source="graph_memory_tool",
            )
        except GraphMemoryError as exc:
            return f"Error: {exc}"
        return f"Added graph edge {source_label}/{source_id} -[{relationship}]-> {target_label}/{target_id}."

    @tool
    def add_graph_documents(documents: list[Any]) -> str:
        """Add LangChain graph documents to graph memory."""
        try:
            graph_backend.add_graph_documents(cast_documents(documents))
        except GraphMemoryError as exc:
            return f"Error: {exc}"
        return f"Added {len(documents)} graph document(s)."

    @tool
    def record_graph_trace(
        situation: str,
        rationale: str,
        action: str,
        outcome: str,
        artifacts: list[str] | None = None,
        evidence: list[str] | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        subagent_id: str | None = None,
        task_id: str | None = None,
    ) -> str:
        """Record a Situation/Rationale/Action/Outcome trace for long-running agent work."""
        try:
            trace_id = graph_backend.record_graph_trace(
                situation=situation,
                rationale=rationale,
                action=action,
                outcome=outcome,
                artifacts=artifacts,
                evidence=evidence,
                run_id=run_id,
                agent_id=agent_id,
                subagent_id=subagent_id,
                task_id=task_id,
                source="graph_trace_tool",
            )
        except GraphMemoryError as exc:
            return f"Error: {exc}"
        return f"Recorded graph trace {trace_id}."

    tools = [recall_graph_memory, record_graph_trace]
    if include_low_level_writes:
        tools.extend([add_graph_node, add_graph_edge, add_graph_documents])
    return tools


def cast_documents(documents: Sequence[Any]) -> Sequence[Any]:
    """Validate that a graph document payload is list-like.

    Args:
        documents: Graph documents.

    Returns:
        The original graph document sequence.

    Raises:
        GraphMemoryError: If the payload is not list-like.
    """
    if not isinstance(documents, list):
        msg = "documents must be a list."
        raise GraphMemoryError(msg)
    return documents
