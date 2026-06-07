"""Internal graph integration adapters used by the graph memory backend."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from deepagents_graph_memory.errors import GraphMemoryValidationError

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
Properties = dict[str, JsonValue]

SEARCH_TEXT_FIELDS = {
    "action",
    "aliases",
    "artifact",
    "artifacts",
    "description",
    "error",
    "evidence",
    "name",
    "outcome",
    "rationale",
    "summary",
    "situation",
    "text",
    "title",
    "value",
}
SEARCH_METADATA_FIELDS = {
    "scope_key",
    "created_at",
    "updated_at",
    "created_by",
    "created_by_agent",
    "source_agent",
    "source",
}
SEARCH_STOPWORDS = {
    "about",
    "after",
    "and",
    "did",
    "for",
    "from",
    "how",
    "the",
    "this",
    "try",
    "was",
    "what",
    "when",
    "where",
    "which",
    "why",
}
SEARCH_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")


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
    """Internal adapter boundary for the Kuzu graph store.

    This is not a public graph database protocol. It keeps GraphMemoryBackend
    testable while the real store uses Kuzu's documented LangChain integration.
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


def node_search_text(label: str, node_id: str, properties: Mapping[str, JsonValue]) -> str:
    """Build the normalized text indexed for graph-memory node recall."""
    parts = [label, node_id]
    for key in sorted(SEARCH_TEXT_FIELDS):
        if key in properties:
            parts.extend(_flatten_search_value(properties[key]))
    public = {key: value for key, value in properties.items() if key not in SEARCH_METADATA_FIELDS and key != "search_text"}
    if public:
        parts.append(json.dumps(public, sort_keys=True))
    return " ".join(part for part in parts if part).strip()


def lexical_search_score(query: str, text: str) -> int:
    """Score a query against searchable text without adding another package."""
    normalized_query = query.casefold().strip()
    haystack = text.casefold()
    if not normalized_query or not haystack:
        return 0
    score = 0
    if normalized_query in haystack:
        score += 100
    terms = _search_terms(normalized_query)
    tokens = set(_search_terms(haystack))
    for term in terms:
        if term in tokens:
            score += 20
        elif term in haystack:
            score += 8
    return score


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


def _flatten_search_value(value: JsonValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_flatten_search_value(item))
        return items
    if isinstance(value, dict):
        items = []
        for item in value.values():
            items.extend(_flatten_search_value(item))
        return items
    if value is None:
        return []
    return [str(value)]


def _search_terms(value: str) -> list[str]:
    terms = []
    seen = set()
    for raw in SEARCH_TERM_RE.findall(value.casefold()):
        if len(raw) < 2 or raw in SEARCH_STOPWORDS:
            continue
        variants = [raw, raw.replace("-", "_")]
        if raw.endswith("s") and len(raw) > 3:
            variants.append(raw[:-1])
        for term in variants:
            if term not in seen:
                terms.append(term)
                seen.add(term)
    return terms
