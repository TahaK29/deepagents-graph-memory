"""Kuzu adapter for graph memory."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from deepagents_graph_memory.errors import GraphMemoryConfigurationError, GraphMemoryValidationError
from deepagents_graph_memory.paths import neighborhood_path, node_path, validate_identifier, validate_node_id
from deepagents_graph_memory.stores import (
    GraphEdge,
    GraphNode,
    LimitedResult,
    NeighborhoodResult,
    Properties,
    SearchItem,
    SearchResult,
    lexical_search_score,
    node_search_text,
    validate_properties,
)

try:
    import kuzu
except ImportError as exc:  # pragma: no cover - exercised when package is absent
    raise ImportError("Kuzu support requires the `kuzu` package.") from exc


class _KuzuGraph:
    """Minimal query and schema adapter over a `kuzu.Connection`.

    Replaces the langchain-community `KuzuGraph` wrapper. `KuzuGraphStore` only
    ever used that class for query execution and schema reflection, so this
    package no longer depends on the sunset `langchain-community` package for
    what is a thin wrapper over `kuzu.Connection`.
    """

    def __init__(self, database: Any) -> None:
        self.conn = kuzu.Connection(database)
        self.schema = ""
        self.refresh_schema()

    @property
    def get_schema(self) -> str:
        """Return the reflected graph schema text."""
        return self.schema

    def query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a Cypher query and return rows as dictionaries."""
        result = self.conn.execute(query, params or {})
        column_names = result.get_column_names()
        rows: list[dict[str, Any]] = []
        while result.has_next():
            rows.append(dict(zip(column_names, result.get_next(), strict=False)))
        return rows

    def refresh_schema(self) -> None:
        """Reflect node and relationship tables into a schema string.

        Uses public Cypher introspection (`SHOW_TABLES`, `TABLE_INFO`,
        `SHOW_CONNECTION`) rather than kuzu's private `_get_*` methods, so the
        adapter does not depend on unstable internals.
        """
        tables = self.query("CALL SHOW_TABLES() RETURN *;")
        node_labels = [row["name"] for row in tables if row["type"] == "NODE"]
        rel_labels = [row["name"] for row in tables if row["type"] == "REL"]

        node_properties = [{"properties": self._table_properties(label), "label": label} for label in node_labels]
        rel_properties = [{"properties": self._table_properties(label), "label": label} for label in rel_labels]
        relationships = [
            f"(:{connection['source table name']})-[:{label}]->(:{connection['destination table name']})"
            for label in rel_labels
            for connection in self.query(f"CALL SHOW_CONNECTION('{label}') RETURN *;")
        ]

        self.schema = f"Node properties: {node_properties}\nRelationships properties: {rel_properties}\nRelationships: {relationships}\n"

    def _table_properties(self, table: str) -> list[tuple[str, str]]:
        """Return ``(name, type)`` pairs for a node or relationship table."""
        return [(row["name"], row["type"]) for row in self.query(f"CALL TABLE_INFO('{table}') RETURN *;")]


class KuzuGraphStore:
    """Internal adapter for LangChain's Kuzu graph integration."""

    def __init__(self, graph: Any) -> None:
        """Initialize the adapter.

        Args:
            graph: LangChain Kuzu graph object.
        """
        self.graph = graph
        self._fts_ready_labels: set[str] = set()

    @classmethod
    def memory(cls) -> KuzuGraphStore:
        """Create an in-memory Kuzu graph store."""
        database = kuzu.Database(":memory:")
        return cls(_KuzuGraph(database))

    def get_schema(self, *, scope_key: str | None = None) -> str:
        """Return graph schema text."""
        del scope_key
        self._refresh_schema()
        if not self._labels() and not self._relationships():
            return "No graph schema has been created yet."
        schema = getattr(self.graph, "get_schema", "")
        return schema() if callable(schema) else str(schema)

    def list_labels(self, *, scope_key: str | None = None, limit: int = 50) -> LimitedResult:
        """List known node labels."""
        del scope_key
        labels = self._labels()
        return LimitedResult(items=labels[:limit], truncated=len(labels) > limit)

    def list_node_ids(self, label: str, *, scope_key: str | None = None, limit: int = 50) -> LimitedResult:
        """List ids for a node label."""
        validate_identifier(label, field="label")
        if label not in self._labels():
            return LimitedResult(items=[])
        rows = self._query(f"MATCH (n:{label}) RETURN n.id AS id, n.scope_key AS scope_key LIMIT {int(limit) + 1}", {})
        ids = [str(row["id"]) for row in rows if row.get("id") is not None and self._row_scope_matches(row, scope_key)]
        return LimitedResult(items=ids[:limit], truncated=len(ids) > limit)

    def get_node(self, label: str, node_id: str, *, scope_key: str | None = None) -> GraphNode | None:
        """Return a single node."""
        validate_identifier(label, field="label")
        validate_node_id(node_id)
        if label not in self._labels():
            return None
        rows = self._query(f"MATCH (n:{label} {{id: $id}}) RETURN n", {"id": node_id})
        if not rows:
            return None
        node = _coerce_node(rows[0].get("n"), default_label=label, default_id=node_id)
        if not _scope_matches(node.properties, scope_key):
            return None
        return node

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
        collected: dict[tuple[str, str, str, str, str], GraphEdge] = {}
        truncated_nodes = False
        truncated_edges = False

        for _level in range(max(depth, 1)):
            next_frontier: set[tuple[str, str]] = set()
            for frontier_label, frontier_id in frontier:
                for edge in self._get_immediate_edges(frontier_label, frontier_id, scope_key=scope_key, limit=max_edges + 1):
                    key = (edge.source_label, edge.source_id, edge.relationship, edge.target_label, edge.target_id)
                    if key not in collected:
                        if len(collected) >= max_edges:
                            truncated_edges = True
                            continue
                        collected[key] = edge
                    for candidate in ((edge.source_label, edge.source_id), (edge.target_label, edge.target_id)):
                        if candidate in seen_nodes:
                            continue
                        if len(seen_nodes) >= max_nodes:
                            truncated_nodes = True
                            continue
                        seen_nodes.add(candidate)
                        next_frontier.add(candidate)
            frontier = next_frontier
            if not frontier:
                break

        edges = sorted(collected.values(), key=lambda edge: (edge.relationship, edge.source_label, edge.source_id, edge.target_label, edge.target_id))
        return NeighborhoodResult(node=node, edges=edges, truncated_nodes=truncated_nodes, truncated_edges=truncated_edges)

    def search(self, query: str, *, scope_key: str | None = None, limit: int = 20) -> SearchResult:
        """Search graph metadata."""
        scored: list[tuple[float, SearchItem]] = []
        seen_paths: set[str] = set()
        for score, item in self._search_fts(query, scope_key=scope_key, limit=limit):
            if item.path in seen_paths:
                continue
            scored.append((score + 1000, item))
            seen_paths.add(item.path)
        for score, item in self._search_relationships(query, scope_key=scope_key, limit=limit):
            if item.path in seen_paths:
                continue
            scored.append((score, item))
            seen_paths.add(item.path)
        scored.sort(key=lambda item: (-item[0], item[1].path, item[1].title))
        items = [item for _score, item in scored]
        return SearchResult(items=items[:limit], truncated=len(items) > limit)

    def add_node(self, label: str, node_id: str, *, properties: Properties | None = None, scope_key: str | None = None) -> None:
        """Add or update a node."""
        validate_identifier(label, field="label")
        validate_node_id(node_id)
        self._ensure_node_table(label)
        existing = self.get_node(label, node_id, scope_key=scope_key)
        merged = dict(existing.properties) if existing else {}
        merged.update(validate_properties(properties))
        props = validate_properties(merged)
        search_text = node_search_text(label, node_id, props)
        self._query(
            f"""
            MERGE (n:{label} {{id: $id}})
            SET n.type = "entity",
                n.search_text = $search_text,
                n.properties = $properties,
                n.scope_key = $scope_key
            """,
            {"id": node_id, "search_text": search_text, "properties": json.dumps(props, sort_keys=True), "scope_key": scope_key},
        )

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
        props = validate_properties(properties)
        self.add_node(source_label, source_id, scope_key=scope_key)
        self.add_node(target_label, target_id, scope_key=scope_key)
        self._ensure_rel_table(relationship, source_label, target_label)
        self._query(
            f"""
            MATCH (source:{source_label} {{id: $source_id}}),
                  (target:{target_label} {{id: $target_id}})
            MERGE (source)-[rel:{relationship}]->(target)
            SET rel.properties = $properties,
                rel.scope_key = $scope_key
            """,
            {
                "source_id": source_id,
                "target_id": target_id,
                "properties": json.dumps(props, sort_keys=True),
                "scope_key": scope_key,
            },
        )

    def add_graph_documents(self, documents: Sequence[Any], *, scope_key: str | None = None) -> None:
        """Add graph documents through validated scoped writes."""
        for document in documents:
            nodes = getattr(document, "nodes", None)
            relationships = getattr(document, "relationships", None)
            if nodes is None or relationships is None:
                msg = "documents must contain LangChain GraphDocument-like objects."
                raise GraphMemoryValidationError(msg)
            for node in nodes:
                self.add_node(str(node.type), str(node.id), properties=_with_scope(node.properties, scope_key), scope_key=scope_key)
            for relationship in relationships:
                self.add_edge(
                    str(relationship.source.type),
                    str(relationship.source.id),
                    str(relationship.type),
                    str(relationship.target.type),
                    str(relationship.target.id),
                    properties=_with_scope(relationship.properties, scope_key),
                    scope_key=scope_key,
                )

    def _get_immediate_edges(self, label: str, node_id: str, *, scope_key: str | None, limit: int) -> list[GraphEdge]:
        rows = self._query(f"MATCH (n:{label} {{id: $id}})-[r]->(m) RETURN n, r, m LIMIT {int(limit)}", {"id": node_id})
        incoming_rows = self._query(f"MATCH (m)-[r]->(n:{label} {{id: $id}}) RETURN m, r, n LIMIT {int(limit)}", {"id": node_id})
        edges: list[GraphEdge] = []
        for row in rows:
            edge = _coerce_edge(row, source_key="n", target_key="m")
            if edge is not None and _scope_matches(edge.properties, scope_key):
                edges.append(edge)
        for row in incoming_rows:
            edge = _coerce_edge(row, source_key="m", target_key="n")
            if edge is not None and _scope_matches(edge.properties, scope_key):
                edges.append(edge)
        return edges

    def _search_fts(self, query: str, *, scope_key: str | None, limit: int) -> list[tuple[float, SearchItem]]:
        scored: list[tuple[float, SearchItem]] = []
        for label in self._labels():
            self._ensure_fts_index(label)
            try:
                rows = self._query(
                    f"""
                    CALL QUERY_FTS_INDEX('{label}', 'graph_memory_fts', $query, top := {int(limit) + 1})
                    RETURN node, score
                    ORDER BY score DESC
                    """,
                    {"query": query},
                )
            except GraphMemoryConfigurationError as exc:
                msg = f"Kuzu full-text search failed for label {label!r}. Ensure the Kuzu fts extension is available."
                raise GraphMemoryConfigurationError(msg) from exc
            for row in rows:
                node = _coerce_node(row.get("node"), default_label=label, default_id="")
                if not node.id or not _scope_matches(node.properties, scope_key):
                    continue
                score = float(row.get("score", 0.0) or 0.0)
                scored.append(
                    (
                        score,
                        SearchItem(
                            path=node_path(node.label, node.id),
                            title=f"{node.label}: {node.id}",
                            text=_summarize_properties(node.properties),
                        ),
                    )
                )
        return scored

    def _search_relationships(self, query: str, *, scope_key: str | None, limit: int) -> list[tuple[float, SearchItem]]:
        scored: list[tuple[float, SearchItem]] = []
        for relationship in self._relationships():
            score = lexical_search_score(query, relationship.replace("_", " "))
            if score <= 0:
                continue
            rows = self._query(
                f"""
                MATCH (source)-[r:{relationship}]->(target)
                RETURN source, r, target
                LIMIT {int(limit) + 1}
                """,
                {},
            )
            for row in rows:
                edge = _coerce_edge(row, source_key="source", target_key="target")
                if edge is None or not _scope_matches(edge.properties, scope_key):
                    continue
                item = SearchItem(
                    path=neighborhood_path(edge.source_label, edge.source_id),
                    title=f"{edge.source_id} {edge.relationship} {edge.target_id}",
                    text=_summarize_properties(edge.properties),
                )
                scored.append((float(score), item))
        return scored

    def _ensure_fts_index(self, label: str) -> bool:
        if label in self._fts_ready_labels:
            return True
        validate_identifier(label, field="label")
        try:
            self._query("LOAD fts;", {})
        except GraphMemoryConfigurationError as exc:
            message = str(exc).casefold()
            if "already" not in message and "loaded" not in message:
                msg = "Kuzu full-text search requires the `fts` extension, but loading it failed."
                raise GraphMemoryConfigurationError(msg) from exc
        try:
            self._query(f"CALL CREATE_FTS_INDEX('{label}', 'graph_memory_fts', ['id', 'type', 'search_text', 'properties']);", {})
        except GraphMemoryConfigurationError as exc:
            message = str(exc).casefold()
            if "already" not in message and "exist" not in message:
                msg = f"Kuzu full-text search index creation failed for label {label!r}."
                raise GraphMemoryConfigurationError(msg) from exc
        self._fts_ready_labels.add(label)
        return True

    def _labels(self) -> list[str]:
        schema_getter = getattr(self.graph, "get_schema_dict", None)
        if callable(schema_getter):
            schema = schema_getter()
            return sorted(str(node["label"]) for node in schema.get("nodes", []) if node.get("label") != "Chunk")
        rows = self._query("CALL SHOW_TABLES() RETURN *;", {})
        return sorted(str(row.get("name")) for row in rows if row.get("type") == "NODE")

    def _relationships(self) -> list[str]:
        rows = self._query("CALL SHOW_TABLES() RETURN *;", {})
        return sorted(str(row.get("name")) for row in rows if row.get("type") == "REL")

    def _refresh_schema(self) -> None:
        refresh_schema = getattr(self.graph, "refresh_schema", None)
        if callable(refresh_schema):
            refresh_schema()

    def _query(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return cast("list[dict[str, Any]]", self.graph.query(query, params))
        except Exception as exc:  # noqa: BLE001
            msg = f"Kuzu graph query failed: {exc}"
            raise GraphMemoryConfigurationError(msg) from exc

    def _ensure_node_table(self, label: str) -> None:
        self._query(
            f"""
            CREATE NODE TABLE IF NOT EXISTS {label} (
                id STRING,
                type STRING,
                search_text STRING,
                properties STRING,
                scope_key STRING,
                PRIMARY KEY(id)
            );
            """,
            {},
        )

    def _ensure_rel_table(self, relationship: str, source_label: str, target_label: str) -> None:
        self._query(
            f"""
            CREATE REL TABLE IF NOT EXISTS {relationship} (
                FROM {source_label} TO {target_label},
                properties STRING,
                scope_key STRING
            );
            """,
            {},
        )
        try:
            self._query(f"ALTER TABLE {relationship} ADD FROM {source_label} TO {target_label};", {})
        except GraphMemoryConfigurationError as exc:
            if "already exists" not in str(exc).casefold():
                raise

    @staticmethod
    def _row_scope_matches(row: dict[str, Any], scope_key: str | None) -> bool:
        if scope_key is None:
            return True
        value = row.get("scope_key")
        return value == scope_key


def _coerce_node(value: Any, *, default_label: str, default_id: str) -> GraphNode:
    data = value if isinstance(value, dict) else {}
    label = str(data.get("_label", data.get("label", default_label)))
    node_id = str(data.get("id", default_id))
    properties = _decode_properties(data)
    for key, item in data.items():
        if key not in {"_id", "_label", "id", "properties", "search_text", "type"} and item is not None:
            properties.setdefault(key, item)
    return GraphNode(label=label, id=node_id, properties=validate_properties(properties))


def _coerce_edge(row: dict[str, Any], *, source_key: str, target_key: str) -> GraphEdge | None:
    raw_edge = row.get("r")
    if not isinstance(raw_edge, dict):
        return None
    source = _coerce_node(row.get(source_key), default_label="Node", default_id="")
    target = _coerce_node(row.get(target_key), default_label="Node", default_id="")
    relationship = str(raw_edge.get("_label", raw_edge.get("label", raw_edge.get("type", "RELATED_TO"))))
    properties = _decode_properties(raw_edge)
    for key, item in raw_edge.items():
        if not key.startswith("_") and key not in {"label", "type", "properties"} and item is not None:
            properties.setdefault(key, item)
    return GraphEdge(
        source_label=source.label,
        source_id=source.id,
        relationship=relationship,
        target_label=target.label,
        target_id=target.id,
        properties=validate_properties(properties),
    )


def _decode_properties(data: dict[str, Any]) -> dict[str, Any]:
    raw = data.get("properties")
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {"properties": raw}
        return decoded if isinstance(decoded, dict) else {"properties": decoded}
    if isinstance(raw, dict):
        return raw
    return {}


def _summarize_properties(properties: dict[str, Any]) -> str:
    public = {key: value for key, value in properties.items() if key not in {"scope_key", "search_text"}}
    if not public:
        return ""
    return json.dumps(public, sort_keys=True)


def _with_scope(properties: dict[str, Any] | None, scope_key: str | None) -> Properties:
    scoped = dict(properties or {})
    if scope_key is not None:
        scoped["scope_key"] = scope_key
    return validate_properties(scoped)


def _scope_matches(properties: dict[str, Any], scope_key: str | None) -> bool:
    if scope_key is None:
        return True
    value = properties.get("scope_key")
    return value == scope_key
