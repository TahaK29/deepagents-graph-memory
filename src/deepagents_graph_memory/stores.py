"""Internal graph integration adapters used by the graph memory backend."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from deepagents_graph_memory.errors import GraphMemoryValidationError
from deepagents_graph_memory.paths import neighborhood_path, node_path, validate_identifier, validate_node_id

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
Properties = dict[str, JsonValue]


@dataclass(frozen=True)
class GraphNode:
    """A graph node returned by an internal adapter."""

    label: str
    id: str
    properties: Properties = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    """A graph edge returned by an internal adapter."""

    source_label: str
    source_id: str
    relationship: str
    target_label: str
    target_id: str
    properties: Properties = field(default_factory=dict)


@dataclass(frozen=True)
class LimitedResult:
    """A bounded list result."""

    items: list[str]
    truncated: bool = False


@dataclass(frozen=True)
class SearchItem:
    """A graph search result."""

    path: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    """A bounded graph search result."""

    items: list[SearchItem]
    truncated: bool = False


@dataclass(frozen=True)
class NeighborhoodResult:
    """A bounded node neighborhood result."""

    node: GraphNode
    edges: list[GraphEdge]
    truncated_nodes: bool = False
    truncated_edges: bool = False


class GraphStoreAdapter(Protocol):
    """Internal adapter boundary for graph integrations.

    This is not a public graph database protocol. It keeps GraphMemoryBackend
    testable while each real database adapter continues to use its documented
    LangChain integration.
    """

    def get_schema(self, *, scope_key: str | None = None) -> str:
        """Return graph schema text."""

    def list_labels(self, *, scope_key: str | None = None, limit: int = 50) -> LimitedResult:
        """List known node labels."""

    def list_node_ids(self, label: str, *, scope_key: str | None = None, limit: int = 50) -> LimitedResult:
        """List ids for a node label."""

    def get_node(self, label: str, node_id: str, *, scope_key: str | None = None) -> GraphNode | None:
        """Return a single node."""

    def get_neighbors(
        self,
        label: str,
        node_id: str,
        *,
        scope_key: str | None = None,
        depth: int = 1,
        max_nodes: int = 50,
        max_edges: int = 100,
    ) -> NeighborhoodResult | None:
        """Return a bounded node neighborhood."""

    def search(self, query: str, *, scope_key: str | None = None, limit: int = 20) -> SearchResult:
        """Search graph metadata."""

    def add_node(self, label: str, node_id: str, *, properties: Properties | None = None, scope_key: str | None = None) -> None:
        """Add or update a node."""

    def add_edge(
        self,
        source_label: str,
        source_id: str,
        relationship: str,
        target_label: str,
        target_id: str,
        *,
        properties: Properties | None = None,
        scope_key: str | None = None,
    ) -> None:
        """Add or update an edge."""

    def add_graph_documents(self, documents: Sequence[Any], *, scope_key: str | None = None) -> None:
        """Add graph documents."""


def utc_now() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


def validate_properties(properties: Mapping[str, Any] | None) -> Properties:
    """Validate a graph properties mapping.

    Args:
        properties: Properties supplied by an agent or caller.

    Returns:
        A JSON-serializable properties dict.

    Raises:
        GraphMemoryValidationError: If the payload is not a safe JSON object.
    """
    if properties is None:
        return {}
    if not isinstance(properties, Mapping):
        msg = "properties must be a JSON object."
        raise GraphMemoryValidationError(msg)
    result: Properties = {}
    for key, value in properties.items():
        if not isinstance(key, str) or not key:
            msg = "property keys must be non-empty strings."
            raise GraphMemoryValidationError(msg)
        if key.startswith("_"):
            msg = "property keys must not start with underscore."
            raise GraphMemoryValidationError(msg)
        result[key] = _validate_json_value(value, path=key)
    return result


def merge_metadata(properties: Mapping[str, Any] | None, *, scope_key: str | None = None, metadata: Mapping[str, Any] | None = None) -> Properties:
    """Merge caller properties with graph-memory metadata.

    Args:
        properties: Caller properties.
        scope_key: Optional scope key.
        metadata: Additional metadata.

    Returns:
        Validated merged properties.
    """
    merged: dict[str, Any] = dict(validate_properties(properties))
    now = utc_now()
    merged.setdefault("created_at", now)
    merged["updated_at"] = now
    if scope_key is not None:
        merged["scope_key"] = scope_key
    if metadata:
        for key, value in metadata.items():
            if value is not None:
                merged[key] = value
    return validate_properties(merged)


def _validate_json_value(value: Any, *, path: str) -> JsonValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, path=path) for item in value]
    if isinstance(value, dict):
        return {str(key): _validate_json_value(item, path=f"{path}.{key}") for key, item in value.items()}
    try:
        json.dumps(value)
    except TypeError as exc:
        msg = f"property {path!r} must be JSON serializable."
        raise GraphMemoryValidationError(msg) from exc
    return value


class InMemoryGraphStore:
    """Small in-memory graph store for tests and examples."""

    def __init__(self) -> None:
        self._nodes: dict[tuple[str | None, str, str], GraphNode] = {}
        self._edges: dict[tuple[str | None, str, str, str, str, str], GraphEdge] = {}

    def get_schema(self, *, scope_key: str | None = None) -> str:
        """Return graph schema text."""
        labels = self.list_labels(scope_key=scope_key, limit=1000).items
        relationships = sorted({edge.relationship for edge in self._iter_edges(scope_key=scope_key)})
        if not labels and not relationships:
            return "No graph schema has been created yet."
        lines = ["Node labels:"]
        lines.extend(f"- {label}" for label in labels)
        lines.append("")
        lines.append("Relationship types:")
        lines.extend(f"- {relationship}" for relationship in relationships)
        return "\n".join(lines).rstrip()

    def list_labels(self, *, scope_key: str | None = None, limit: int = 50) -> LimitedResult:
        """List known node labels."""
        labels = sorted({label for node_scope, label, _node_id in self._nodes if self._scope_matches(node_scope, scope_key)})
        return LimitedResult(items=labels[:limit], truncated=len(labels) > limit)

    def list_node_ids(self, label: str, *, scope_key: str | None = None, limit: int = 50) -> LimitedResult:
        """List ids for a node label."""
        validate_identifier(label, field="label")
        ids = sorted(node_id for node_scope, node_label, node_id in self._nodes if node_label == label and self._scope_matches(node_scope, scope_key))
        return LimitedResult(items=ids[:limit], truncated=len(ids) > limit)

    def get_node(self, label: str, node_id: str, *, scope_key: str | None = None) -> GraphNode | None:
        """Return a single node."""
        validate_identifier(label, field="label")
        validate_node_id(node_id)
        return self._nodes.get((scope_key, label, node_id))

    def get_neighbors(
        self,
        label: str,
        node_id: str,
        *,
        scope_key: str | None = None,
        depth: int = 1,
        max_nodes: int = 50,
        max_edges: int = 100,
    ) -> NeighborhoodResult | None:
        """Return a bounded node neighborhood."""
        node = self.get_node(label, node_id, scope_key=scope_key)
        if node is None:
            return None
        frontier = {(label, node_id)}
        seen_nodes = {(label, node_id)}
        collected: list[GraphEdge] = []
        truncated_nodes = False
        truncated_edges = False
        for _level in range(max(depth, 1)):
            next_frontier: set[tuple[str, str]] = set()
            for edge in self._iter_edges(scope_key=scope_key):
                source = (edge.source_label, edge.source_id)
                target = (edge.target_label, edge.target_id)
                touches = source in frontier or target in frontier
                if not touches:
                    continue
                if len(collected) >= max_edges:
                    truncated_edges = True
                    continue
                collected.append(edge)
                for candidate in (source, target):
                    if candidate not in seen_nodes:
                        if len(seen_nodes) >= max_nodes:
                            truncated_nodes = True
                            continue
                        seen_nodes.add(candidate)
                        next_frontier.add(candidate)
            frontier = next_frontier
            if not frontier:
                break
        collected.sort(key=lambda edge: (edge.relationship, edge.source_label, edge.source_id, edge.target_label, edge.target_id))
        return NeighborhoodResult(node=node, edges=collected, truncated_nodes=truncated_nodes, truncated_edges=truncated_edges)

    def search(self, query: str, *, scope_key: str | None = None, limit: int = 20) -> SearchResult:
        """Search graph metadata."""
        needle = query.casefold()
        items: list[SearchItem] = []
        for node in sorted(self._iter_nodes(scope_key=scope_key), key=lambda item: (item.label, item.id)):
            haystack = " ".join([node.label, node.id, json.dumps(node.properties, sort_keys=True)]).casefold()
            if needle in haystack:
                items.append(
                    SearchItem(
                        path=node_path(node.label, node.id),
                        title=f"{node.label}: {node.id}",
                        text=_summarize_properties(node.properties),
                    )
                )
        for edge in sorted(self._iter_edges(scope_key=scope_key), key=lambda item: (item.relationship, item.source_id, item.target_id)):
            haystack = " ".join(
                [
                    edge.source_label,
                    edge.source_id,
                    edge.relationship,
                    edge.target_label,
                    edge.target_id,
                    json.dumps(edge.properties, sort_keys=True),
                ]
            ).casefold()
            if needle in haystack:
                path = neighborhood_path(edge.source_label, edge.source_id)
                title = f"{edge.source_id} {edge.relationship} {edge.target_id}"
                items.append(SearchItem(path=path, title=title, text=_summarize_properties(edge.properties)))
        return SearchResult(items=items[:limit], truncated=len(items) > limit)

    def add_node(self, label: str, node_id: str, *, properties: Properties | None = None, scope_key: str | None = None) -> None:
        """Add or update a node."""
        validate_identifier(label, field="label")
        validate_node_id(node_id)
        key = (scope_key, label, node_id)
        existing = self._nodes.get(key)
        merged = dict(existing.properties) if existing else {}
        merged.update(validate_properties(properties))
        self._nodes[key] = GraphNode(label=label, id=node_id, properties=merged)

    def add_edge(
        self,
        source_label: str,
        source_id: str,
        relationship: str,
        target_label: str,
        target_id: str,
        *,
        properties: Properties | None = None,
        scope_key: str | None = None,
    ) -> None:
        """Add or update an edge."""
        validate_identifier(source_label, field="source_label")
        validate_identifier(target_label, field="target_label")
        validate_identifier(relationship, field="relationship")
        validate_node_id(source_id)
        validate_node_id(target_id)
        self.add_node(source_label, source_id, properties={"scope_key": scope_key} if scope_key else None, scope_key=scope_key)
        self.add_node(target_label, target_id, properties={"scope_key": scope_key} if scope_key else None, scope_key=scope_key)
        key = (scope_key, source_label, source_id, relationship, target_label, target_id)
        existing = self._edges.get(key)
        merged = dict(existing.properties) if existing else {}
        merged.update(validate_properties(properties))
        self._edges[key] = GraphEdge(
            source_label=source_label,
            source_id=source_id,
            relationship=relationship,
            target_label=target_label,
            target_id=target_id,
            properties=merged,
        )

    def add_graph_documents(self, documents: Sequence[Any], *, scope_key: str | None = None) -> None:
        """Add graph documents."""
        for document in documents:
            nodes = _get_value(document, "nodes", default=[])
            relationships = _get_value(document, "relationships", default=[])
            for node in nodes:
                node_id = str(_get_value(node, "id"))
                label = str(_get_value(node, "type", default=_get_value(node, "label", default="Node")))
                properties = validate_properties(_get_value(node, "properties", default={}))
                self.add_node(label, node_id, properties=properties, scope_key=scope_key)
            for relationship in relationships:
                source = _get_value(relationship, "source")
                target = _get_value(relationship, "target")
                properties = validate_properties(_get_value(relationship, "properties", default={}))
                self.add_edge(
                    str(_get_value(source, "type", default=_get_value(source, "label", default="Node"))),
                    str(_get_value(source, "id")),
                    str(_get_value(relationship, "type")),
                    str(_get_value(target, "type", default=_get_value(target, "label", default="Node"))),
                    str(_get_value(target, "id")),
                    properties=properties,
                    scope_key=scope_key,
                )

    def _iter_nodes(self, *, scope_key: str | None = None) -> Iterable[GraphNode]:
        for node_scope, _label, _node_id in sorted(self._nodes):
            if self._scope_matches(node_scope, scope_key):
                yield self._nodes[(node_scope, _label, _node_id)]

    def _iter_edges(self, *, scope_key: str | None = None) -> Iterable[GraphEdge]:
        for key, edge in self._edges.items():
            if self._scope_matches(key[0], scope_key):
                yield edge

    @staticmethod
    def _scope_matches(item_scope: str | None, requested_scope: str | None) -> bool:
        if requested_scope is None:
            return item_scope is None
        return item_scope == requested_scope


def _summarize_properties(properties: Mapping[str, JsonValue]) -> str:
    public = {key: value for key, value in properties.items() if key not in {"scope_key"}}
    if not public:
        return ""
    return json.dumps(public, sort_keys=True)


def _get_value(obj: Any, key: str, *, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)
