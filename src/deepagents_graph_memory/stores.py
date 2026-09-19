# - Defines the shared shapes for graph items, connections, and search results.
# - Also checks saved details and helps decide which text matches a question.
# - Tests: test_validation.py checks bad data; test_renderers.py uses the shared graph shapes.
#        test_scope.py and test_recall.py exercise the helpers through saving and finding context.

"""Internal graph integration adapters used by the graph memory backend."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
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


@dataclass(frozen=True)
class EdgeResult:
    """A bounded list of focused graph relationships."""

    items: list[GraphEdge]
    truncated: bool = False


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

    def list_subject_trace_ids(self, subject_id: str, *, scope_key: str | None = None, limit: int = 50) -> LimitedResult:
        """List subject traces with current findings first."""

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

    def list_trace_edges(
        self, trace_id: str, relationship: str, *, incoming: bool = False, scope_key: str | None = None, limit: int = 50
    ) -> EdgeResult:
        """Return one trace's relationships without unrelated neighbors using the budget."""

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

    def transaction(self) -> AbstractContextManager[None]:
        """Atomically group graph writes."""


def utc_now() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


def finding_observed_timestamp(node: GraphNode) -> float | None:
    """Return an aware observation's timestamp, or unknown for missing or invalid time."""
    value = node.properties.get("observed_at")
    try:
        observed = datetime.fromisoformat(value) if isinstance(value, str) else None
        return observed.timestamp() if observed is not None and observed.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def valid_finding_link(newer: GraphNode, older: GraphNode, relationship: str, *, reviewed_count: int = 0) -> bool:
    """Check whether a finding update or resolution has valid recorded semantics."""
    if newer.properties.get("subject") != older.properties.get("subject") or not newer.properties.get("evidence"):
        return False
    if relationship == "RESOLVES":
        return reviewed_count >= 2
    if relationship != "SUPERSEDES" or newer.properties.get("finding_type") != "state" or older.properties.get("finding_type") != "state":
        return False
    try:
        new_time = datetime.fromisoformat(newer.properties["observed_at"])
        old_time = datetime.fromisoformat(older.properties["observed_at"])
        return new_time.tzinfo is not None and old_time.tzinfo is not None and new_time > old_time
    except (KeyError, TypeError, ValueError):
        return False


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
    try:
        json.dumps(dict(properties), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        msg = "properties must be JSON serializable."
        raise GraphMemoryValidationError(msg) from exc
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
    if "scope_key" in merged and merged["scope_key"] != scope_key:
        msg = "scope_key cannot differ from the active namespace."
        raise GraphMemoryValidationError(msg)
    if metadata and "scope_key" in metadata and metadata["scope_key"] != scope_key:
        msg = "scope_key cannot differ from the active namespace."
        raise GraphMemoryValidationError(msg)
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
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"property {path!r} must be finite."
        raise GraphMemoryValidationError(msg)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, path=path) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            msg = f"property {path!r} must have string keys."
            raise GraphMemoryValidationError(msg)
        return {key: _validate_json_value(item, path=f"{path}.{key}") for key, item in value.items()}
    msg = f"property {path!r} must be JSON serializable."
    raise GraphMemoryValidationError(msg)


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
