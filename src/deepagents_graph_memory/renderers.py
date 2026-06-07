"""Markdown renderers for virtual graph files."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping

from deepagents_graph_memory.paths import neighborhood_path, node_path
from deepagents_graph_memory.stores import GraphEdge, GraphNode, NeighborhoodResult, SearchResult


def render_index() -> str:
    """Render the graph memory index page."""
    return "\n".join(
        [
            "# Graph Memory",
            "",
            "Graph memory exposes relationship-oriented facts as read-only markdown files.",
            "",
            "## Paths",
            "- `/graph/schema.md` - graph schema",
            "- `/graph/nodes/{label}/{id}.md` - node page",
            "- `/graph/views/neighborhood/{label}/{id}.md` - immediate relationships",
            "- `/graph/search/{query}.md` - graph search results",
            "",
            "Use graph memory tools to add or update graph facts. Generated graph views are read-only.",
        ]
    )


def render_schema(schema: str) -> str:
    """Render a schema page.

    Args:
        schema: Schema text from the graph store.

    Returns:
        Markdown schema page.
    """
    body = schema.strip() or "No graph schema has been created yet."
    return f"# Graph Schema\n\n```text\n{body}\n```"


def render_node(node: GraphNode, neighborhood: NeighborhoodResult | None = None) -> str:
    """Render a node page.

    Args:
        node: Node to render.
        neighborhood: Optional immediate neighborhood.

    Returns:
        Markdown node page.
    """
    lines = [f"# {node.label}: {node.id}", ""]
    lines.extend(_render_properties(node.properties))
    edges = neighborhood.edges if neighborhood else []
    if edges:
        lines.append("")
        lines.extend(_render_edge_sections(node, edges))
    if neighborhood and (neighborhood.truncated_edges or neighborhood.truncated_nodes):
        lines.append("")
        lines.append(_truncation_note(neighborhood.truncated_nodes, neighborhood.truncated_edges))
    return "\n".join(lines).rstrip() + "\n"


def render_neighborhood(result: NeighborhoodResult) -> str:
    """Render a node neighborhood page.

    Args:
        result: Neighborhood query result.

    Returns:
        Markdown neighborhood page.
    """
    lines = [f"# Neighborhood: {result.node.label}: {result.node.id}", ""]
    lines.extend(_render_properties(result.node.properties))
    if result.edges:
        lines.append("")
        lines.extend(_render_edge_sections(result.node, result.edges))
    else:
        lines.append("")
        lines.append("No relationships found.")
    if result.truncated_edges or result.truncated_nodes:
        lines.append("")
        lines.append(_truncation_note(result.truncated_nodes, result.truncated_edges))
    return "\n".join(lines).rstrip() + "\n"


def render_search(query: str, result: SearchResult) -> str:
    """Render graph search results.

    Args:
        query: Search query.
        result: Search result.

    Returns:
        Markdown search page.
    """
    lines = [f"# Graph Search: {query}", ""]
    if not result.items:
        lines.append("No matching graph facts found.")
    else:
        for item in result.items:
            suffix = f" - {item.text}" if item.text else ""
            lines.append(f"- [{item.title}]({item.path}){suffix}")
    if result.truncated:
        lines.append("")
        lines.append("Results truncated. Refine the query or increase the limit.")
    return "\n".join(lines).rstrip() + "\n"


def _render_properties(properties: Mapping[str, object]) -> list[str]:
    public = {
        key: value
        for key, value in properties.items()
        if key
        not in {
            "scope_key",
            "created_at",
            "updated_at",
            "created_by",
            "created_by_agent",
            "source_agent",
            "source",
            "search_text",
        }
    }
    if not public:
        return []
    lines = ["## Properties"]
    for key in sorted(public):
        lines.append(f"- **{key}**: {_format_value(public[key])}")
    return lines


def _render_edge_sections(node: GraphNode, edges: Iterable[GraphEdge]) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.source_label == node.label and edge.source_id == node.id:
            heading = _outgoing_heading(edge.relationship)
            grouped[heading].append(_node_link(edge.target_label, edge.target_id))
        elif edge.target_label == node.label and edge.target_id == node.id:
            heading = _incoming_heading(edge.relationship)
            grouped[heading].append(_node_link(edge.source_label, edge.source_id))
    lines: list[str] = []
    for heading in sorted(grouped):
        lines.append(f"## {heading}")
        for item in sorted(set(grouped[heading])):
            lines.append(f"- {item}")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _node_link(label: str, node_id: str) -> str:
    return f"[{node_id}]({node_path(label, node_id)})"


def _outgoing_heading(relationship: str) -> str:
    mapping = {
        "HAS_ACTION": "Action",
        "HAS_OUTCOME": "Outcome",
        "HAS_RATIONALE": "Rationale",
        "HAS_SITUATION": "Situation",
        "HAS_TRACE": "Traces",
        "INVOLVED": "Involved",
        "JUSTIFIED": "Justified",
        "LED_TO": "Led to",
        "PRODUCED": "Produced",
        "RECORDED": "Recorded",
        "SUPPORTS": "Supports",
    }
    return mapping.get(relationship, _humanize_relationship(relationship))


def _incoming_heading(relationship: str) -> str:
    mapping = {
        "HAS_ACTION": "Action for",
        "HAS_OUTCOME": "Outcome for",
        "HAS_RATIONALE": "Rationale for",
        "HAS_SITUATION": "Situation for",
        "HAS_TRACE": "Trace of",
        "INVOLVED": "Involved in",
        "JUSTIFIED": "Justified by",
        "LED_TO": "Led from",
        "PRODUCED": "Produced by",
        "RECORDED": "Recorded by",
        "SUPPORTS": "Supported by",
    }
    return mapping.get(relationship, f"{_humanize_relationship(relationship)} (incoming)")


def _humanize_relationship(relationship: str) -> str:
    return relationship.replace("_", " ").capitalize()


def _format_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return f"`{json.dumps(value, sort_keys=True)}`"


def _truncation_note(truncated_nodes: bool, truncated_edges: bool) -> str:
    targets = []
    if truncated_nodes:
        targets.append("nodes")
    if truncated_edges:
        targets.append("edges")
    label = " and ".join(targets) if targets else "results"
    return f"Results truncated for {label}. Refine the query or increase the limit."


def render_label_listing(label: str, ids: list[str], *, truncated: bool = False) -> str:
    """Render an internal label listing page for debugging and tests.

    Args:
        label: Node label.
        ids: Node ids.
        truncated: Whether the listing was truncated.

    Returns:
        Markdown listing.
    """
    lines = [f"# Nodes: {label}", ""]
    lines.extend(f"- [{node_id}]({node_path(label, node_id)})" for node_id in ids)
    if truncated:
        lines.extend(["", "Results truncated. Refine the listing or increase the limit."])
    return "\n".join(lines).rstrip() + "\n"


def render_neighborhood_link(label: str, node_id: str) -> str:
    """Render a markdown link to a neighborhood view."""
    return f"[Neighborhood]({neighborhood_path(label, node_id)})"
