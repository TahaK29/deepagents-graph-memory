# - Finds context for a question, then follows useful connections to fill in the story.
# - Stops when the next connection is unhelpful or the answer reaches its size limit.
# - Tests: test_recall.py covers finding matches, following connections, and stopping;
#        test_trace.py checks bringing back the reasons and outcomes behind past work.

"""Adaptive graph memory recall."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from deepagents_graph_memory.errors import GraphMemoryPathError, GraphMemoryValidationError
from deepagents_graph_memory.paths import node_path, parse_graph_path
from deepagents_graph_memory.stores import GraphEdge, GraphNode, GraphStoreAdapter, JsonValue, finding_observed_timestamp, valid_finding_link

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
    related_findings: bool = False
    related_incomplete: bool = False
    omitted_anchors: list[_NodeRef] | None = None
    dependency_notices: list[str] | None = None
    dependency_unknown: bool = False
    dependency_unknown_roots: list[str] | None = None


def recall_graph_memory(
    store: GraphStoreAdapter,
    query: str,
    *,
    scope_key: str | None = None,
    anchors: Sequence[str] | None = None,
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
        anchors: Optional concrete starting hints such as file paths, run ids, task ids, or subagent ids.
        mode: Recall expansion mode. `auto` expands while relevant, `local` reads one hop, and `deep` expands to `max_depth`.
        token_budget: Approximate output token budget.
        max_depth: Maximum traversal depth.
        max_nodes: Maximum nodes to include.
        max_edges: Maximum edges to include.

    Returns:
        Compact markdown with source graph paths.
    """
    query = _validate_recall_query(query)
    anchors = [_validate_anchor(anchor) for anchor in anchors or []]
    mode = _validate_mode(mode)
    token_budget, max_depth, max_nodes, max_edges = _validate_budgets(token_budget, max_depth, max_nodes, max_edges)

    terms = _query_terms(query)
    state = _RecallState(nodes={}, edges={})
    seeds = _find_seed_nodes(store, query, terms, anchors=anchors, scope_key=scope_key, limit=min(max_nodes, 20), state=state)
    if not seeds:
        return _render_recall(query, state, token_budget=token_budget)

    processed_subjects: set[str] = set()
    anchored_refs = {_seed_from_anchor(anchor) for anchor in anchors}
    _expand_subject_findings(
        store,
        seeds,
        state,
        processed_subjects=processed_subjects,
        scope_key=scope_key,
        max_nodes=max_nodes,
        max_edges=max_edges,
        anchored_refs=anchored_refs,
    )

    for index, seed in enumerate(seeds):
        node = store.get_node(seed.label, seed.node_id, scope_key=scope_key)
        if node is None:
            continue
        _add_node(state, node, distance=0, score=100 - index, max_nodes=max_nodes)
        if seed in anchored_refs and seed not in state.nodes:
            state.related_incomplete = True
            state.omitted_anchors = [*(state.omitted_anchors or []), seed]

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
    _expand_subject_findings(
        store,
        list(state.nodes),
        state,
        processed_subjects=processed_subjects,
        scope_key=scope_key,
        max_nodes=max_nodes,
        max_edges=max_edges,
        anchored_refs=anchored_refs,
    )
    _review_dependencies(store, state, scope_key=scope_key, max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges)
    return _render_recall(query, state, token_budget=token_budget)


def _review_dependencies(
    store: GraphStoreAdapter,
    state: _RecallState,
    *,
    scope_key: str | None,
    max_depth: int,
    max_nodes: int,
    max_edges: int,
) -> None:
    """Flag conclusions that rely on changed premises using focused, bounded graph links."""
    roots = list(
        dict.fromkeys(
            _NodeRef("Trace", trace_id)
            for ref, record in state.nodes.items()
            if (trace_id := ref.node_id if ref.label == "Trace" else record.node.properties.get("trace_id")) is not None
            and isinstance(trace_id, str)
        )
    )
    notices: list[str] = []
    unknown_roots: list[str] = []
    for root in roots:
        inspected: set[str] = set()
        read_nodes: set[str] = set()
        used_edges = 0
        unknown = False
        has_dependencies = False
        changes: set[tuple[str, str, str]] = set()

        def visit(
            trace_id: str,
            path: set[str],
            depth: int,
            inspected: set[str] = inspected,
            read_nodes: set[str] = read_nodes,
            changes: set[tuple[str, str, str]] = changes,
        ) -> None:
            nonlocal has_dependencies, used_edges, unknown
            if trace_id in path:
                unknown = True
                return
            if trace_id in inspected:
                return
            if len(read_nodes) >= max_nodes and trace_id not in read_nodes:
                unknown = True
                return
            trace = store.get_node("Trace", trace_id, scope_key=scope_key)
            if trace is None:
                unknown = True
                return
            inspected.add(trace_id)
            read_nodes.add(trace_id)
            if depth >= max_depth:
                if store.list_trace_edges(trace_id, "BASED_ON", scope_key=scope_key, limit=1).items:
                    has_dependencies = True
                    unknown = True
                return
            remaining = max_edges - used_edges
            if remaining <= 0:
                unknown = True
                return
            dependencies = store.list_trace_edges(trace_id, "BASED_ON", scope_key=scope_key, limit=remaining)
            has_dependencies |= bool(dependencies.items) or dependencies.truncated
            unknown |= dependencies.truncated
            used_edges += len(dependencies.items)
            next_path = path | {trace_id}
            for dependency in dependencies.items:
                premise_id = dependency.target_id
                if len(read_nodes) >= max_nodes and premise_id not in read_nodes:
                    unknown = True
                    continue
                premise = store.get_node("Trace", premise_id, scope_key=scope_key)
                if premise is None:
                    unknown = True
                    continue
                read_nodes.add(premise_id)
                _add_node(state, trace, distance=0, score=125, max_nodes=max_nodes)
                _add_node(state, premise, distance=1, score=125, max_nodes=max_nodes)
                _add_edge(state, dependency, distance=1, score=125, max_edges=max_edges)
                if _NodeRef("Trace", premise_id) not in state.nodes:
                    unknown = True
                if ("Trace", trace_id, "BASED_ON", "Trace", premise_id) not in state.edges:
                    unknown = True
                if not (premise.properties.get("evidence") or premise.properties.get("evidence_refs") or premise.properties.get("depends_on")) or (
                    premise.properties.get("finding_type") == "state" and finding_observed_timestamp(premise) is None
                ):
                    unknown = True
                for relationship in ("SUPERSEDES", "RESOLVES"):
                    remaining = max_edges - used_edges
                    if remaining <= 0:
                        unknown = True
                        break
                    updates = store.list_trace_edges(premise_id, relationship, incoming=True, scope_key=scope_key, limit=remaining)
                    unknown |= updates.truncated
                    used_edges += len(updates.items)
                    for update in updates.items:
                        if len(read_nodes) >= max_nodes and update.source_id not in read_nodes:
                            unknown = True
                            continue
                        successor = store.get_node("Trace", update.source_id, scope_key=scope_key)
                        if successor is None:
                            unknown = True
                            continue
                        read_nodes.add(successor.id)
                        reviewed_count = 0
                        if relationship == "RESOLVES":
                            remaining = max_edges - used_edges
                            if remaining <= 0:
                                unknown = True
                                continue
                            reviewed = store.list_trace_edges(successor.id, "RESOLVES", scope_key=scope_key, limit=remaining)
                            used_edges += len(reviewed.items)
                            for reviewed_edge in reviewed.items:
                                if len(read_nodes) >= max_nodes and reviewed_edge.target_id not in read_nodes:
                                    unknown = True
                                    break
                                target = store.get_node("Trace", reviewed_edge.target_id, scope_key=scope_key)
                                if target is None:
                                    unknown = True
                                    continue
                                read_nodes.add(target.id)
                                reviewed_count += target.properties.get("subject") == successor.properties.get("subject")
                            unknown |= reviewed.truncated and reviewed_count < 2
                        if valid_finding_link(successor, premise, relationship, reviewed_count=reviewed_count):
                            changes.add((premise_id, successor.id, relationship))
                            if store.list_trace_edges(successor.id, "BASED_ON", scope_key=scope_key, limit=1).items:
                                unknown = True
                            _add_node(state, successor, distance=1, score=125, max_nodes=max_nodes)
                            _add_edge(state, update, distance=1, score=125, max_edges=max_edges)
                            if _NodeRef("Trace", successor.id) not in state.nodes:
                                unknown = True
                            if ("Trace", successor.id, relationship, "Trace", premise_id) not in state.edges:
                                unknown = True
                visit(premise_id, next_path, depth + 1)

        visit(root.node_id, set(), 0)
        if changes:
            details = ", ".join(f"{old} updated by {new} via {relationship}" for old, new, relationship in sorted(changes))
            notices.append(f"Trace {root.node_id} needs recheck: supporting premise changed ({details}). The conclusion remains recorded.")
        if has_dependencies and unknown:
            state.dependency_unknown = True
            unknown_roots.append(root.node_id)
    state.dependency_notices = notices
    state.dependency_unknown_roots = unknown_roots


def _expand_subject_findings(
    store: GraphStoreAdapter,
    seeds: Sequence[_NodeRef],
    state: _RecallState,
    *,
    processed_subjects: set[str],
    scope_key: str | None,
    max_nodes: int,
    max_edges: int,
    anchored_refs: set[_NodeRef | None],
) -> None:
    """Bring findings for a retrieved subject together before normal traversal uses budgets."""
    subjects: set[str] = set()
    for seed in seeds:
        node = store.get_node(seed.label, seed.node_id, scope_key=scope_key)
        if node is None:
            continue
        if seed.label == "Subject":
            subjects.add(seed.node_id)
            continue
        trace = node if seed.label == "Trace" else None
        trace_id = node.properties.get("trace_id")
        if trace is None and isinstance(trace_id, str):
            trace = store.get_node("Trace", trace_id, scope_key=scope_key)
        if trace is None and seed.label in {"Artifact", "Evidence", "EvidenceSource"}:
            neighbors = store.get_neighbors(seed.label, seed.node_id, scope_key=scope_key, max_nodes=max_nodes, max_edges=max_edges)
            if neighbors is not None:
                state.related_incomplete |= neighbors.truncated_nodes or neighbors.truncated_edges
                for edge in neighbors.edges:
                    for ref in _edge_neighbors(edge, seed):
                        connected = store.get_node(ref.label, ref.node_id, scope_key=scope_key)
                        if connected is None:
                            continue
                        connected_trace_id = connected.id if connected.label == "Trace" else connected.properties.get("trace_id")
                        if isinstance(connected_trace_id, str):
                            connected_trace = store.get_node("Trace", connected_trace_id, scope_key=scope_key)
                            connected_subject = connected_trace.properties.get("subject") if connected_trace is not None else None
                            if isinstance(connected_subject, str):
                                subjects.add(f"subject-{hashlib.sha256(connected_subject.encode('utf-8')).hexdigest()}")
        subject = trace.properties.get("subject") if trace is not None else None
        if isinstance(subject, str):
            subjects.add(f"subject-{hashlib.sha256(subject.encode('utf-8')).hexdigest()}")

    for subject_id in sorted(subjects):
        if subject_id in processed_subjects:
            continue
        processed_subjects.add(subject_id)
        subject = store.get_node("Subject", subject_id, scope_key=scope_key)
        if subject is None:
            continue
        state.related_findings = True
        anchored_traces = [
            ref
            for ref in seeds
            if ref in anchored_refs
            and ref.label == "Trace"
            and (anchor := store.get_node("Trace", ref.node_id, scope_key=scope_key)) is not None
            and anchor.properties.get("subject") == subject.properties.get("value")
        ]
        available = max_nodes - len(state.nodes)
        reserve_anchor = bool(anchored_traces and available > 1)
        selected = store.list_subject_trace_ids(subject_id, scope_key=scope_key, limit=max(available - int(reserve_anchor), 0))
        state.related_incomplete |= selected.truncated
        trace_ids = selected.items
        if reserve_anchor:
            trace_ids.extend(ref.node_id for ref in anchored_traces if ref.node_id not in trace_ids)
        for trace_id in trace_ids:
            trace = store.get_node("Trace", trace_id, scope_key=scope_key)
            if trace is None:
                state.related_incomplete = True
                continue
            _add_node(state, trace, distance=1, score=110, max_nodes=max_nodes)
            if _NodeRef("Trace", trace_id) not in state.nodes:
                state.related_incomplete = True
        if len(state.nodes) < max_nodes:
            _add_node(state, subject, distance=0, score=120, max_nodes=max_nodes)
        for trace_id in trace_ids:
            trace_links = store.get_neighbors("Trace", trace_id, scope_key=scope_key, max_nodes=max_nodes, max_edges=max(max_edges, 100))
            if trace_links is None:
                continue
            if trace_links.truncated_edges:
                state.related_incomplete = True
            for link in trace_links.edges:
                if link.relationship in {"SUPERSEDES", "RESOLVES"} and link.source_id == trace_id:
                    if _NodeRef("Trace", link.target_id) in state.nodes:
                        _add_edge(state, link, distance=1, score=110, max_edges=max_edges)
                    else:
                        state.related_incomplete = True
                elif link.relationship == "ABOUT" and _NodeRef("Subject", subject_id) in state.nodes:
                    _add_edge(state, link, distance=1, score=110, max_edges=max_edges)


def _find_seed_nodes(
    store: GraphStoreAdapter,
    query: str,
    terms: set[str],
    *,
    anchors: Sequence[str],
    scope_key: str | None,
    limit: int,
    state: _RecallState,
) -> list[_NodeRef]:
    seeds: list[_NodeRef] = []
    seen: set[_NodeRef] = set()
    for anchor in anchors:
        ref = _seed_from_anchor(anchor)
        if ref is None or ref in seen:
            continue
        if store.get_node(ref.label, ref.node_id, scope_key=scope_key) is None:
            continue
        seeds.append(ref)
        seen.add(ref)
        if len(seeds) >= limit:
            return seeds
    for search_query in _search_queries(query, terms, anchors):
        result = store.search(search_query, scope_key=scope_key, limit=limit)
        state.search_truncated = state.search_truncated or result.truncated
        for item in result.items:
            try:
                parsed = parse_graph_path(item.path)
            except GraphMemoryPathError:
                continue
            if parsed.kind != "node" or parsed.label is None or parsed.node_id is None:
                continue
            ref = _NodeRef(parsed.label, parsed.node_id)
            if ref not in seen:
                seeds.append(ref)
                seen.add(ref)
            if len(seeds) >= limit:
                return seeds
    return seeds


def _seed_from_anchor(anchor: str) -> _NodeRef | None:
    try:
        parsed = parse_graph_path(anchor)
    except GraphMemoryPathError:
        return None
    if parsed.kind != "node" or parsed.label is None or parsed.node_id is None:
        return None
    return _NodeRef(parsed.label, parsed.node_id)


def _search_queries(query: str, terms: set[str], anchors: Sequence[str]) -> list[str]:
    queries = list(anchors)
    queries.append(query)
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
    public = {key: value for key, value in properties.items() if key not in {"scope_key", "created_at", "updated_at", "search_text"}}
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
    return terms


def _render_recall(query: str, state: _RecallState, *, token_budget: int) -> str:
    lines = [f"# Graph Memory Recall: {query}", ""]
    if state.dependency_notices:
        lines.extend(["## Dependency review", *state.dependency_notices, ""])
    if state.dependency_unknown_roots:
        lines.extend(
            ["## Dependency coverage", *(f"Trace {trace_id}: dependency status unknown." for trace_id in state.dependency_unknown_roots), ""]
        )
    if not state.nodes and not state.edges:
        lines.append("No matching graph memory found.")
        return "\n".join(lines).rstrip() + "\n"

    if state.related_findings:
        for ref in state.omitted_anchors or []:
            path = _prefixed(node_path(ref.label, ref.node_id))
            lines.append(f"Anchor omitted from bounded history: `{path}`. Read it directly for earlier context.")
        if state.omitted_anchors:
            lines.append("")
        lines.extend(_finding_history(state))

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
    incomplete = state.related_findings and (state.related_incomplete or state.truncated_nodes or state.truncated_edges)
    prefix = ""
    if state.dependency_unknown:
        prefix += "Dependency status unknown; inspect dependencies.\n"
    if state.dependency_notices:
        prefix += "Some conclusions need recheck.\n"
    available_tokens = max(1, token_budget - (len(prefix) + 3) // 4)
    if state.related_findings and not incomplete and not state.search_truncated:
        source_ids = {
            ref["source_id"]
            for record in state.nodes.values()
            if record.node.label == "Trace"
            if isinstance(record.node.properties.get("evidence_refs"), list)
            for ref in record.node.properties.get("evidence_refs", [])
            if isinstance(ref, dict) and isinstance(ref.get("source_id"), str)
        }
        count_line = f"Distinct cited source IDs in returned findings: {len(source_ids)}; independence not established."
        if source_ids and sum(len(line) + 1 for line in [*lines, count_line]) <= max(available_tokens * 4, 80):
            lines.append(count_line)
    return prefix + _fit_token_budget(
        lines,
        available_tokens,
        related_findings=state.related_findings,
        related_incomplete=incomplete,
    )


def _finding_history(state: _RecallState) -> list[str]:
    traces = {
        ref.node_id: record.node
        for ref, record in state.nodes.items()
        if ref.label == "Trace" and isinstance(record.node.properties.get("subject"), str)
    }
    successors: dict[str, list[str]] = {}
    resolution_targets: dict[str, set[str]] = {}
    for record in state.edges.values():
        edge = record.edge
        if edge.source_label != "Trace" or edge.target_label != "Trace":
            continue
        newer, older = traces.get(edge.source_id), traces.get(edge.target_id)
        if newer is None or older is None:
            continue
        if edge.relationship == "SUPERSEDES" and valid_finding_link(newer, older, "SUPERSEDES"):
            successors.setdefault(edge.target_id, []).append(edge.source_id)
        elif edge.relationship == "RESOLVES" and valid_finding_link(newer, older, "RESOLVES", reviewed_count=2):
            resolution_targets.setdefault(edge.source_id, set()).add(edge.target_id)
    reviewed: dict[str, list[str]] = {}
    for resolution_id, targets in resolution_targets.items():
        if len(targets) >= 2:
            for target_id in targets:
                reviewed.setdefault(target_id, []).append(resolution_id)
    ordered = sorted(
        traces.values(),
        key=lambda node: (
            node.id in successors or node.id in reviewed,
            finding_observed_timestamp(node) is None,
            -(finding_observed_timestamp(node) or 0),
            node.id,
        ),
    )
    lines = ["## Finding history"]
    for trace in ordered:
        path = _prefixed(node_path("Trace", trace.id))
        later = successors.get(trace.id, [])
        resolutions = reviewed.get(trace.id, [])
        statuses = []
        if later:
            statuses.append(
                "superseded by " + ", ".join(f"[Trace: {node_id}]({_prefixed(node_path('Trace', node_id))})" for node_id in sorted(later))
            )
        if resolutions:
            statuses.append(
                "reviewed in resolution "
                + ", ".join(f"[Trace: {node_id}]({_prefixed(node_path('Trace', node_id))})" for node_id in sorted(resolutions))
            )
        status = "; ".join(statuses) if statuses else "not superseded (requires comparison)"
        if not statuses and state.related_incomplete:
            status = "status unknown in partial history (requires comparison)"
        observed = trace.properties.get("observed_at", "unknown")
        outcome = trace.properties.get("outcome", "unknown")
        source = trace.properties.get("source", "unknown")
        evidence = trace.properties.get("evidence", "unknown")
        refs = trace.properties.get("evidence_refs")
        citations = f"; cited sources: {json.dumps(refs, sort_keys=True)}" if refs else ""
        lines.append(
            f"- [Trace: {trace.id}]({path}) — {status}; observed_at: {observed}; "
            f"outcome: {outcome}; source: {source}; unstructured evidence reports: {evidence}{citations}"
        )
    return [*lines, ""]


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
    return sorted(paths)


def _property_suffix(properties: dict[str, JsonValue]) -> str:
    public = {
        key: value
        for key, value in properties.items()
        if key not in {"scope_key", "created_at", "updated_at", "created_by", "created_by_agent", "source_agent", "search_text"}
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


def _fit_token_budget(lines: list[str], token_budget: int, *, related_findings: bool = False, related_incomplete: bool = False) -> str:
    char_budget = max(token_budget * 4, 80)
    if related_incomplete or (related_findings and sum(len(line) + 1 for line in lines) > char_budget):
        lines.insert(0, "Related findings incomplete; no resolved/current answer. Fetch more context.")
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


def _validate_anchor(anchor: str) -> str:
    if not isinstance(anchor, str):
        msg = "anchors must be strings."
        raise GraphMemoryValidationError(msg)
    normalized = anchor.strip()
    if not normalized or len(normalized) > _MAX_QUERY_LENGTH:
        msg = f"anchors must be between 1 and {_MAX_QUERY_LENGTH} characters."
        raise GraphMemoryValidationError(msg)
    if "\x00" in normalized or any(ord(char) < 32 for char in normalized):
        msg = "anchors must not contain NUL bytes or control characters."
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
