"""Adaptive graph memory recall."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

from deepagents_graph_memory.errors import GraphMemoryPathError, GraphMemoryValidationError
from deepagents_graph_memory.paths import neighborhood_path, node_path, parse_graph_path
from deepagents_graph_memory.stores import GraphEdge, GraphNode, GraphStoreAdapter, JsonValue

RecallMode = Literal["auto", "local", "deep"]

_VALID_MODES = {"auto", "local", "deep"}
_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_MAX_QUERY_LENGTH = 1000
_MAX_DEPTH = 10


@dataclass(frozen=True, order=True)
class _NodeRef:
    label: str
    node_id: str


@dataclass
class _NodeRecord:
    node: GraphNode
    distance: int
    score: int


@dataclass
class _EdgeRecord:
    edge: GraphEdge
    distance: int
    score: int


@dataclass
class _RecallState:
    nodes: dict[_NodeRef, _NodeRecord]
    edges: dict[tuple[str, str, str, str, str], _EdgeRecord]
    search_truncated: bool = False
    truncated_nodes: bool = False
    truncated_edges: bool = False
    stopped_reason: str = ""


def recall_graph_memory(
    store: GraphStoreAdapter,
    query: str,
    *,
    scope_key: str | None = None,
    mode: RecallMode = "auto",
    token_budget: int = 2000,
    max_depth: int = 3,
    max_nodes: int = 50,
    max_edges: int = 100,
) -> str:
    """Recall a relevant slice of graph memory.

    Args:
        store: Internal graph integration adapter.
        query: Natural-language recall query.
        scope_key: Optional scope key used by the configured backend.
        mode: Recall expansion mode. `auto` expands while relevant, `local` reads one hop, and `deep` expands to `max_depth`.
        token_budget: Approximate output token budget.
        max_depth: Maximum traversal depth.
        max_nodes: Maximum nodes to include.
        max_edges: Maximum edges to include.

    Returns:
        Compact markdown with source graph paths.
    """
    query = _validate_recall_query(query)
    mode = _validate_mode(mode)
    token_budget, max_depth, max_nodes, max_edges = _validate_budgets(token_budget, max_depth, max_nodes, max_edges)

    terms = _query_terms(query)
    state = _RecallState(nodes={}, edges={})
    seeds = _find_seed_nodes(store, query, terms, scope_key=scope_key, limit=min(max_nodes, 20), state=state)
    if not seeds:
        return _render_recall(query, state, token_budget=token_budget)

    for index, seed in enumerate(seeds):
        node = store.get_node(seed.label, seed.node_id, scope_key=scope_key)
        if node is None:
            continue
        _add_node(state, node, distance=0, score=100 - index, max_nodes=max_nodes)

    target_depth = 1 if mode == "local" else max_depth
    frontier = list(seeds)
    seen_frontier = set(frontier)

    for depth in range(1, target_depth + 1):
        if not frontier:
            state.stopped_reason = "No new graph neighbors remained to inspect."
            break
        next_frontier: list[_NodeRef] = []
        layer_added_relevant_edge = False

        for ref in frontier:
            if len(state.edges) >= max_edges:
                state.truncated_edges = True
                break
            neighborhood = store.get_neighbors(ref.label, ref.node_id, scope_key=scope_key, depth=1, max_nodes=max_nodes, max_edges=max_edges)
            if neighborhood is None:
                continue
            if neighborhood.truncated_nodes:
                state.truncated_nodes = True
            if neighborhood.truncated_edges:
                state.truncated_edges = True
            _add_node(state, neighborhood.node, distance=depth - 1, score=_score_node(neighborhood.node, terms, query), max_nodes=max_nodes)

            for edge in neighborhood.edges:
                if len(state.edges) >= max_edges:
                    state.truncated_edges = True
                    break
                edge_score = _score_edge(edge, terms, query)
                include_edge = mode in {"deep", "local"} or depth == 1 or edge_score > 0
                if not include_edge:
                    continue
                _add_edge(state, edge, distance=depth, score=edge_score, max_edges=max_edges)
                if edge_score > 0:
                    layer_added_relevant_edge = True
                for neighbor in _edge_neighbors(edge, ref):
                    node = store.get_node(neighbor.label, neighbor.node_id, scope_key=scope_key)
                    if node is None:
                        node = GraphNode(label=neighbor.label, id=neighbor.node_id)
                    node_score = _score_node(node, terms, query)
                    _add_node(state, node, distance=depth, score=node_score, max_nodes=max_nodes)
                    should_expand = mode == "deep" or (mode == "auto" and (depth == 1 or edge_score > 0 or node_score > 0))
                    if should_expand and neighbor not in seen_frontier:
                        next_frontier.append(neighbor)
                        seen_frontier.add(neighbor)

        if state.truncated_edges or state.truncated_nodes:
            state.stopped_reason = "Stopped at the configured node or edge safety budget."
            break
        if mode == "auto" and depth > 1 and not layer_added_relevant_edge:
            state.stopped_reason = "Stopped because the next hop did not add relevant graph facts."
            break
        frontier = next_frontier

    if not state.stopped_reason:
        state.stopped_reason = f"Reached traversal depth {target_depth}."
    return _render_recall(query, state, token_budget=token_budget)


def _find_seed_nodes(
    store: GraphStoreAdapter,
    query: str,
    terms: set[str],
    *,
    scope_key: str | None,
    limit: int,
    state: _RecallState,
) -> list[_NodeRef]:
    seeds: list[_NodeRef] = []
    seen: set[_NodeRef] = set()
    for search_query in _search_queries(query, terms):
        result = store.search(search_query, scope_key=scope_key, limit=limit)
        state.search_truncated = state.search_truncated or result.truncated
        for item in result.items:
            try:
                parsed = parse_graph_path(item.path)
            except GraphMemoryPathError:
                continue
            if parsed.kind not in {"node", "neighborhood"} or parsed.label is None or parsed.node_id is None:
                continue
            ref = _NodeRef(parsed.label, parsed.node_id)
            if ref not in seen:
                seeds.append(ref)
                seen.add(ref)
            if len(seeds) >= limit:
                return seeds
    return seeds


def _search_queries(query: str, terms: set[str]) -> list[str]:
    queries = [query]
    queries.extend(sorted(terms, key=lambda item: (-len(item), item)))
    result: list[str] = []
    seen: set[str] = set()
    for item in queries:
        normalized = item.strip()
        if normalized and normalized.casefold() not in seen:
            result.append(normalized)
            seen.add(normalized.casefold())
    return result


def _add_node(state: _RecallState, node: GraphNode, *, distance: int, score: int, max_nodes: int) -> None:
    ref = _NodeRef(node.label, node.id)
    existing = state.nodes.get(ref)
    if existing is None:
        if len(state.nodes) >= max_nodes:
            state.truncated_nodes = True
            return
        state.nodes[ref] = _NodeRecord(node=node, distance=distance, score=score)
        return
    existing.distance = min(existing.distance, distance)
    existing.score = max(existing.score, score)
    if existing.node.properties == {} and node.properties:
        existing.node = node


def _add_edge(state: _RecallState, edge: GraphEdge, *, distance: int, score: int, max_edges: int) -> None:
    key = (edge.source_label, edge.source_id, edge.relationship, edge.target_label, edge.target_id)
    existing = state.edges.get(key)
    if existing is None:
        if len(state.edges) >= max_edges:
            state.truncated_edges = True
            return
        state.edges[key] = _EdgeRecord(edge=edge, distance=distance, score=score)
        return
    existing.distance = min(existing.distance, distance)
    existing.score = max(existing.score, score)


def _edge_neighbors(edge: GraphEdge, source_ref: _NodeRef) -> list[_NodeRef]:
    source = _NodeRef(edge.source_label, edge.source_id)
    target = _NodeRef(edge.target_label, edge.target_id)
    if source == source_ref:
        return [target]
    if target == source_ref:
        return [source]
    return [source, target]


def _score_node(node: GraphNode, terms: set[str], query: str) -> int:
    return _score_text(" ".join([node.label, node.id, _properties_text(node.properties)]), terms, query)


def _score_edge(edge: GraphEdge, terms: set[str], query: str) -> int:
    return _score_text(
        " ".join(
            [
                edge.source_label,
                edge.source_id,
                edge.relationship,
                edge.target_label,
                edge.target_id,
                _properties_text(edge.properties),
            ]
        ),
        terms,
        query,
    )


def _score_text(text: str, terms: set[str], query: str) -> int:
    haystack = text.casefold()
    score = 0
    if query.casefold() in haystack:
        score += 5
    for term in terms:
        if term in haystack:
            score += 1
    return score


def _properties_text(properties: dict[str, JsonValue]) -> str:
    public = {key: value for key, value in properties.items() if key not in {"scope_key", "created_at", "updated_at"}}
    return json.dumps(public, sort_keys=True)


def _query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for raw in _TERM_RE.findall(query.casefold()):
        if len(raw) < 2:
            continue
        terms.add(raw)
        terms.add(raw.replace("-", "_"))
        if raw.endswith("s") and len(raw) > 3:
            terms.add(raw[:-1])
        if raw in {"dependency", "dependencies", "depends", "depended"} or raw.startswith("depend"):
            terms.add("depend")
        if raw in {"owner", "owners", "owned", "owns"}:
            terms.add("own")
        if raw in {"affected", "affects", "impact", "impacted"}:
            terms.add("affect")
        if raw in {"resolved", "resolves", "resolver"}:
            terms.add("resolve")
    return terms


def _render_recall(query: str, state: _RecallState, *, token_budget: int) -> str:
    lines = [f"# Graph Memory Recall: {query}", ""]
    if not state.nodes and not state.edges:
        lines.append("No matching graph memory found.")
        return "\n".join(lines).rstrip() + "\n"

    if state.nodes:
        lines.append("## Nodes")
        for record in _sorted_nodes(state.nodes.values()):
            path = _prefixed(node_path(record.node.label, record.node.id))
            suffix = _property_suffix(record.node.properties)
            lines.append(f"- [{record.node.label}: {record.node.id}]({path}){suffix}")
        lines.append("")

    if state.edges:
        lines.append("## Relationships")
        for record in _sorted_edges(state.edges.values()):
            edge = record.edge
            source_path = _prefixed(node_path(edge.source_label, edge.source_id))
            target_path = _prefixed(node_path(edge.target_label, edge.target_id))
            lines.append(
                f"- [{edge.source_label}: {edge.source_id}]({source_path}) -[{edge.relationship}]-> "
                f"[{edge.target_label}: {edge.target_id}]({target_path})"
            )
        lines.append("")

    lines.append("## Source Paths")
    for path in _source_paths(state):
        lines.append(f"- `{path}`")
    lines.append("")
    if state.stopped_reason:
        lines.append(f"Traversal note: {state.stopped_reason}")
    notes = _truncation_notes(state)
    if notes:
        lines.extend(notes)
    if state.search_truncated:
        lines.append("Search results were truncated before traversal.")
    return _fit_token_budget(lines, token_budget)


def _sorted_nodes(records: Iterable[_NodeRecord]) -> list[_NodeRecord]:
    return sorted(records, key=lambda item: (-item.score, item.distance, item.node.label, item.node.id))


def _sorted_edges(records: Iterable[_EdgeRecord]) -> list[_EdgeRecord]:
    return sorted(
        records,
        key=lambda item: (
            -item.score,
            item.distance,
            item.edge.relationship,
            item.edge.source_label,
            item.edge.source_id,
            item.edge.target_label,
            item.edge.target_id,
        ),
    )


def _source_paths(state: _RecallState) -> list[str]:
    paths = {_prefixed(node_path(record.node.label, record.node.id)) for record in state.nodes.values()}
    paths.update(_prefixed(neighborhood_path(record.edge.source_label, record.edge.source_id)) for record in state.edges.values())
    return sorted(paths)


def _property_suffix(properties: dict[str, JsonValue]) -> str:
    public = {
        key: value
        for key, value in properties.items()
        if key not in {"scope_key", "created_at", "updated_at", "created_by", "created_by_agent", "source_agent", "source"}
    }
    if not public:
        return ""
    return f" - `{json.dumps(public, sort_keys=True)}`"


def _truncation_notes(state: _RecallState) -> list[str]:
    targets = []
    if state.truncated_nodes:
        targets.append("nodes")
    if state.truncated_edges:
        targets.append("edges")
    if not targets:
        return []
    return [f"Results truncated for {' and '.join(targets)}. Ask a narrower question or increase the recall budgets."]


def _fit_token_budget(lines: list[str], token_budget: int) -> str:
    char_budget = max(token_budget * 4, 80)
    output: list[str] = []
    total = 0
    for line in lines:
        projected = total + len(line) + 1
        if projected > char_budget:
            output.append("")
            output.append("Results truncated for token budget. Ask a narrower question or increase token_budget.")
            break
        output.append(line)
        total = projected
    return "\n".join(output).rstrip() + "\n"


def _prefixed(path: str) -> str:
    return f"/graph{path}"


def _validate_recall_query(query: str) -> str:
    if not isinstance(query, str):
        msg = "query must be a string."
        raise GraphMemoryValidationError(msg)
    normalized = query.strip()
    if not normalized or len(normalized) > _MAX_QUERY_LENGTH:
        msg = f"query must be between 1 and {_MAX_QUERY_LENGTH} characters."
        raise GraphMemoryValidationError(msg)
    if "\x00" in normalized or any(ord(char) < 32 for char in normalized):
        msg = "query must not contain NUL bytes or control characters."
        raise GraphMemoryValidationError(msg)
    return normalized


def _validate_mode(mode: str) -> RecallMode:
    if mode not in _VALID_MODES:
        msg = "mode must be one of: auto, local, deep."
        raise GraphMemoryValidationError(msg)
    return cast("RecallMode", mode)


def _validate_budgets(token_budget: int, max_depth: int, max_nodes: int, max_edges: int) -> tuple[int, int, int, int]:
    for name, value in {
        "token_budget": token_budget,
        "max_depth": max_depth,
        "max_nodes": max_nodes,
        "max_edges": max_edges,
    }.items():
        if not isinstance(value, int) or value < 1:
            msg = f"{name} must be a positive integer."
            raise GraphMemoryValidationError(msg)
    if max_depth > _MAX_DEPTH:
        msg = f"max_depth must be less than or equal to {_MAX_DEPTH}."
        raise GraphMemoryValidationError(msg)
    return token_budget, max_depth, max_nodes, max_edges
