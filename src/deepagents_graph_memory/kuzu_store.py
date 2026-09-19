# - Keeps graph items and their connections in Kuzu, and finds them again when asked.
# - Keeps each workspace's context separate when a scope is set.
# - Tests: test_kuzu_integration.py checks saving and reading a connection;
#        test_kuzu_search.py checks search, and test_scope.py checks separation.

"""Kuzu adapter for graph memory."""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import contextmanager
from functools import wraps
from threading import RLock
from typing import Any, cast

from deepagents_graph_memory.errors import GraphMemoryConfigurationError, GraphMemoryValidationError
from deepagents_graph_memory.paths import node_path, validate_identifier, validate_node_id
from deepagents_graph_memory.stores import (
    EdgeResult,
    GraphEdge,
    GraphNode,
    LimitedResult,
    NeighborhoodResult,
    Properties,
    SearchItem,
    SearchResult,
    finding_observed_timestamp,
    lexical_search_score,
    node_search_text,
    utc_now,
    valid_finding_link,
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


def _locked(method: Any) -> Any:
    @wraps(method)
    def wrapper(self: KuzuGraphStore, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _atomic(method: Any) -> Any:
    @wraps(method)
    def wrapper(self: KuzuGraphStore, *args: Any, **kwargs: Any) -> Any:
        with self.transaction():
            return method(self, *args, **kwargs)

    return wrapper


class KuzuGraphStore:
    """Internal adapter for LangChain's Kuzu graph integration."""

    def __init__(self, graph: Any) -> None:
        """Initialize the adapter.

        Args:
            graph: LangChain Kuzu graph object.
        """
        self.graph = graph
        self._fts_ready_labels: set[str] = set()
        # ponytail: one store lock serializes shared-connection work; split connections if throughput matters.
        self._lock = RLock()
        self._transaction_depth = 0
        self._rollback_only = False

    @contextmanager
    def transaction(self) -> Any:
        """Serialize and atomically group writes on the shared Kuzu connection."""
        with self._lock:
            outermost = self._transaction_depth == 0
            if outermost:
                self._query("BEGIN TRANSACTION;", {})
                self._rollback_only = False
            self._transaction_depth += 1
            try:
                yield
            except BaseException:
                self._rollback_only = True
                raise
            else:
                if outermost and self._rollback_only:
                    msg = "A nested graph write failed; transaction was rolled back."
                    raise GraphMemoryConfigurationError(msg)
            finally:
                self._transaction_depth -= 1
                if outermost:
                    if self._rollback_only:
                        try:
                            self._query("ROLLBACK;", {})
                        except GraphMemoryConfigurationError:
                            pass  # Preserve the original write failure if Kuzu already aborted.
                        self._fts_ready_labels.clear()
                    else:
                        try:
                            self._query("COMMIT;", {})
                        except GraphMemoryConfigurationError:
                            try:
                                self._query("ROLLBACK;", {})
                            except GraphMemoryConfigurationError:
                                pass
                            self._fts_ready_labels.clear()
                            self._rollback_only = False
                            raise
                    self._rollback_only = False

    @classmethod
    def memory(cls) -> KuzuGraphStore:
        """Create an in-memory Kuzu graph store."""
        database = kuzu.Database(":memory:")
        return cls(_KuzuGraph(database))

    @_locked
    def get_schema(self, *, scope_key: str | None = None) -> str:
        """Return graph schema text."""
        del scope_key
        self._refresh_schema()
        if not self._labels() and not self._relationships():
            return "No graph schema has been created yet."
        return self.graph.get_schema

    @_locked
    def list_labels(self, *, scope_key: str | None = None, limit: int = 50) -> LimitedResult:
        """List known node labels."""
        labels = [label for label in self._labels() if self._label_has_nodes(label, scope_key=scope_key)]
        return LimitedResult(items=labels[:limit], truncated=len(labels) > limit)

    @_locked
    def list_node_ids(self, label: str, *, scope_key: str | None = None, limit: int = 50) -> LimitedResult:
        """List ids for a node label."""
        validate_identifier(label, field="label")
        if label not in self._labels():
            return LimitedResult(items=[])
        scope_where, params = _scope_where("n", scope_key)
        rows = self._query(
            f"""
            MATCH (n:{label})
            WHERE {scope_where}
            RETURN n.id AS id
            ORDER BY id
            LIMIT {int(limit) + 1}
            """,
            params,
        )
        ids = [str(row["id"]) for row in rows if row.get("id") is not None]
        return LimitedResult(items=ids[:limit], truncated=len(ids) > limit)

    @_locked
    def list_subject_trace_ids(self, subject_id: str, *, scope_key: str | None = None, limit: int = 50) -> LimitedResult:
        """Rank one subject's findings before applying the output limit."""
        validate_node_id(subject_id)
        relationships = set(self._relationships())
        if "Subject" not in self._labels() or "Trace" not in self._labels() or "ABOUT" not in relationships:
            return LimitedResult(items=[])
        rows = self._query(
            "MATCH (t:Trace)-[:ABOUT]->(s:Subject {pk: $pk}) RETURN t",
            {"pk": _node_pk("Subject", subject_id, scope_key)},
        )
        traces = {node.id: node for row in rows if (node := _coerce_node(row.get("t"), default_label="Trace", default_id="")).id}
        predecessors: set[str] = set()
        # ponytail: sort one subject in Python; use indexed ranking if subject histories grow large enough to measure.
        for relationship in ("SUPERSEDES", "RESOLVES"):
            if relationship not in relationships:
                continue
            links = self._query(
                f"MATCH (a:Trace)-[:ABOUT]->(s:Subject {{pk: $pk}}) MATCH (a)-[:{relationship}]->(b:Trace) RETURN a, b",
                {"pk": _node_pk("Subject", subject_id, scope_key)},
            )
            targets_by_source: dict[str, set[str]] = {}
            for row in links:
                source_id = str(row["a"]["id"])
                target_id = str(row["b"]["id"])
                source = traces.get(source_id)
                target = traces.get(target_id)
                if source is not None and target is not None and source.properties.get("subject") == target.properties.get("subject"):
                    targets_by_source.setdefault(source_id, set()).add(target_id)
            for source_id, targets in targets_by_source.items():
                for target_id in targets:
                    if valid_finding_link(traces[source_id], traces[target_id], relationship, reviewed_count=len(targets)):
                        predecessors.add(target_id)

        def rank(trace: GraphNode) -> tuple[bool, bool, float, str]:
            timestamp = finding_observed_timestamp(trace)
            return (trace.id in predecessors, timestamp is None, -(timestamp or 0), trace.id)

        ordered = sorted(traces.values(), key=rank)
        return LimitedResult(items=[trace.id for trace in ordered[:limit]], truncated=len(ordered) > limit)

    @_locked
    def get_node(self, label: str, node_id: str, *, scope_key: str | None = None) -> GraphNode | None:
        """Return a single node."""
        validate_identifier(label, field="label")
        validate_node_id(node_id)
        if label not in self._labels():
            return None
        rows = self._query(f"MATCH (n:{label} {{pk: $pk}}) RETURN n", {"pk": _node_pk(label, node_id, scope_key)})
        if not rows:
            return None
        return _coerce_node(rows[0].get("n"), default_label=label, default_id=node_id)

    @_locked
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

    @_locked
    def list_trace_edges(
        self, trace_id: str, relationship: str, *, incoming: bool = False, scope_key: str | None = None, limit: int = 50
    ) -> EdgeResult:
        """Read a trace relationship without spending its limit on component links."""
        validate_node_id(trace_id)
        validate_identifier(relationship, field="relationship")
        if "Trace" not in self._labels() or relationship not in self._relationships():
            return EdgeResult(items=[])
        direction = (
            f"(other:Trace)-[r:{relationship}]->(trace:Trace {{pk: $pk}})"
            if incoming
            else f"(trace:Trace {{pk: $pk}})-[r:{relationship}]->(other:Trace)"
        )
        source, target = ("other", "trace") if incoming else ("trace", "other")
        rows = self._query(
            f"MATCH {direction} RETURN {source} AS source, r, {target} AS target ORDER BY source.id, target.id LIMIT {int(limit) + 1}",
            {"pk": _node_pk("Trace", trace_id, scope_key)},
        )
        edges = [edge for row in rows if (edge := _coerce_edge(row, source_key="source", target_key="target")) is not None]
        edges.sort(key=lambda edge: (edge.source_id, edge.target_id))
        return EdgeResult(items=edges[:limit], truncated=len(rows) > limit)

    @_locked
    def search(self, query: str, *, scope_key: str | None = None, limit: int = 20) -> SearchResult:
        """Search graph metadata."""
        if self._transaction_depth:
            msg = "search requires Kuzu auto transaction mode."
            raise GraphMemoryConfigurationError(msg)
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

    @_atomic
    def add_node(self, label: str, node_id: str, *, properties: Properties | None = None, scope_key: str | None = None) -> None:
        """Add or update a node."""
        validate_identifier(label, field="label")
        validate_node_id(node_id)
        self._ensure_node_table(label)
        existing = self.get_node(label, node_id, scope_key=scope_key)
        if existing is not None and properties is None:
            return
        merged = dict(existing.properties) if existing else {}
        merged.update(_scoped_properties(properties, scope_key))
        if existing and "created_at" in existing.properties:
            merged["created_at"] = existing.properties["created_at"]
            merged["updated_at"] = utc_now()
        props = validate_properties(merged)
        search_text = node_search_text(label, node_id, props)
        self._query(
            f"""
            MERGE (n:{label} {{pk: $pk}})
            SET n.id = $id,
                n.type = "entity",
                n.search_text = $search_text,
                n.properties = $properties,
                n.scope_key = $scope_key
            """,
            {
                "pk": _node_pk(label, node_id, scope_key),
                "id": node_id,
                "search_text": search_text,
                "properties": json.dumps(props, sort_keys=True),
                "scope_key": scope_key,
            },
        )

    @_atomic
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
        props = _scoped_properties(properties, scope_key)
        self.add_node(source_label, source_id, scope_key=scope_key)
        self.add_node(target_label, target_id, scope_key=scope_key)
        self._ensure_rel_table(relationship, source_label, target_label)
        rows = self._query(
            f"MATCH (source:{source_label} {{pk: $source_pk}})-[rel:{relationship}]->(target:{target_label} {{pk: $target_pk}}) RETURN rel",
            {"source_pk": _node_pk(source_label, source_id, scope_key), "target_pk": _node_pk(target_label, target_id, scope_key)},
        )
        if rows:
            existing_properties = _decode_properties(rows[0]["rel"])
            props = validate_properties({**existing_properties, **props})
            if "created_at" in existing_properties:
                props["created_at"] = existing_properties["created_at"]
                props["updated_at"] = utc_now()
        self._query(
            f"""
            MATCH (source:{source_label} {{pk: $source_pk}}),
                  (target:{target_label} {{pk: $target_pk}})
            MERGE (source)-[rel:{relationship}]->(target)
            SET rel.properties = $properties,
                rel.scope_key = $scope_key
            """,
            {
                "source_pk": _node_pk(source_label, source_id, scope_key),
                "target_pk": _node_pk(target_label, target_id, scope_key),
                "properties": json.dumps(props, sort_keys=True),
                "scope_key": scope_key,
            },
        )

    @_atomic
    def add_graph_documents(self, documents: Sequence[Any], *, scope_key: str | None = None) -> None:
        """Add graph documents through validated scoped writes."""
        for document in documents:
            nodes = getattr(document, "nodes", None)
            relationships = getattr(document, "relationships", None)
            if (
                not isinstance(nodes, Sequence)
                or isinstance(nodes, str | bytes)
                or not isinstance(relationships, Sequence)
                or isinstance(relationships, str | bytes)
            ):
                msg = "documents must contain LangChain GraphDocument-like objects."
                raise GraphMemoryValidationError(msg)
            for node in nodes:
                label, node_id, properties = _document_node(node, scope_key)
                self.add_node(label, node_id, properties=properties, scope_key=scope_key)
            for relationship in relationships:
                source_label, source_id, _ = _document_node(getattr(relationship, "source", None), scope_key)
                target_label, target_id, _ = _document_node(getattr(relationship, "target", None), scope_key)
                self.add_edge(
                    source_label,
                    source_id,
                    getattr(relationship, "type", None),
                    target_label,
                    target_id,
                    properties=_with_scope(getattr(relationship, "properties", None), scope_key),
                    scope_key=scope_key,
                )

    def _get_immediate_edges(self, label: str, node_id: str, *, scope_key: str | None, limit: int) -> list[GraphEdge]:
        params = {"pk": _node_pk(label, node_id, scope_key)}
        rows = self._query(
            f"""
            MATCH (n:{label} {{pk: $pk}})-[r]->(m)
            RETURN n, r, m
            LIMIT {int(limit)}
            """,
            params,
        )
        incoming_rows = self._query(
            f"""
            MATCH (m)-[r]->(n:{label} {{pk: $pk}})
            RETURN m, r, n
            LIMIT {int(limit)}
            """,
            params,
        )
        edges: list[GraphEdge] = []
        for row in rows:
            edge = _coerce_edge(row, source_key="n", target_key="m")
            if edge is not None:
                edges.append(edge)
        for row in incoming_rows:
            edge = _coerce_edge(row, source_key="m", target_key="n")
            if edge is not None:
                edges.append(edge)
        return edges

    def _search_fts(self, query: str, *, scope_key: str | None, limit: int) -> list[tuple[float, SearchItem]]:
        scored: list[tuple[float, SearchItem]] = []
        for label in self._labels():
            self._ensure_fts_index(label)
            scope_where, scope_params = _scope_where("node", scope_key)
            try:
                rows = self._query(
                    f"""
                    CALL QUERY_FTS_INDEX('{label}', 'graph_memory_fts', $query)
                    WHERE {scope_where}
                    RETURN node, score
                    ORDER BY score DESC
                    LIMIT {int(limit) + 1}
                    """,
                    {"query": query, **scope_params},
                )
            except GraphMemoryConfigurationError as exc:
                msg = f"Kuzu full-text search failed for label {label!r}. Ensure the Kuzu fts extension is available."
                raise GraphMemoryConfigurationError(msg) from exc
            for row in rows:
                node = _coerce_node(row.get("node"), default_label=label, default_id="")
                if not node.id:
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
            scope_where, params = _scope_where("r", scope_key)
            rows = self._query(
                f"""
                MATCH (source)-[r:{relationship}]->(target)
                WHERE {scope_where}
                RETURN source, r, target
                LIMIT {int(limit) + 1}
                """,
                params,
            )
            for row in rows:
                edge = _coerce_edge(row, source_key="source", target_key="target")
                if edge is None:
                    continue
                item = SearchItem(
                    path=node_path(edge.source_label, edge.source_id),
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
        rows = self._query("CALL SHOW_TABLES() RETURN *;", {})
        return sorted(str(row.get("name")) for row in rows if row.get("type") == "NODE")

    def _relationships(self) -> list[str]:
        rows = self._query("CALL SHOW_TABLES() RETURN *;", {})
        return sorted(str(row.get("name")) for row in rows if row.get("type") == "REL")

    def _refresh_schema(self) -> None:
        self.graph.refresh_schema()

    def _query(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if self._transaction_depth and self._rollback_only and query.strip().upper() != "ROLLBACK;":
            msg = "Graph transaction already failed; it must roll back."
            raise GraphMemoryConfigurationError(msg)
        try:
            return cast("list[dict[str, Any]]", self.graph.query(query, params))
        except Exception as exc:  # noqa: BLE001
            if self._transaction_depth:
                self._rollback_only = True
            msg = f"Kuzu graph query failed: {exc}"
            raise GraphMemoryConfigurationError(msg) from exc

    def _ensure_node_table(self, label: str) -> None:
        self._reject_table_collision(label, "NODE")
        self._query(
            f"""
            CREATE NODE TABLE IF NOT EXISTS {label} (
                pk STRING,
                id STRING,
                type STRING,
                search_text STRING,
                properties STRING,
                scope_key STRING,
                PRIMARY KEY(pk)
            );
            """,
            {},
        )

    def _ensure_rel_table(self, relationship: str, source_label: str, target_label: str) -> None:
        self._reject_table_collision(relationship, "REL")
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
        connections = self._query(f"CALL SHOW_CONNECTION('{relationship}') RETURN *;", {})
        if any(row["source table name"] == source_label and row["destination table name"] == target_label for row in connections):
            return
        self._query(f"ALTER TABLE {relationship} ADD FROM {source_label} TO {target_label};", {})

    def _reject_table_collision(self, name: str, expected_type: str) -> None:
        for row in self._query("CALL SHOW_TABLES() RETURN *;", {}):
            if str(row["name"]).casefold() == name.casefold() and (row["name"] != name or row["type"] != expected_type):
                msg = f"Graph table {name!r} conflicts with existing {row['type'].lower()} table {row['name']!r}."
                raise GraphMemoryValidationError(msg)

    def _label_has_nodes(self, label: str, *, scope_key: str | None) -> bool:
        scope_where, params = _scope_where("n", scope_key)
        rows = self._query(
            f"""
            MATCH (n:{label})
            WHERE {scope_where}
            RETURN n.id AS id
            LIMIT 1
            """,
            params,
        )
        return bool(rows)


def _coerce_node(value: Any, *, default_label: str, default_id: str) -> GraphNode:
    data = value if isinstance(value, dict) else {}
    label = str(data.get("_label", data.get("label", default_label)))
    node_id = str(data.get("id", default_id))
    properties = _decode_properties(data)
    for key, item in data.items():
        if key not in {"_id", "_label", "id", "pk", "properties", "search_text", "type"} and item is not None:
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
    public = {key: value for key, value in properties.items() if key not in {"pk", "scope_key", "search_text"}}
    if not public:
        return ""
    return json.dumps(public, sort_keys=True)


def _with_scope(properties: dict[str, Any] | None, scope_key: str | None) -> Properties:
    return _scoped_properties(properties, scope_key)


def _document_node(value: Any, scope_key: str | None) -> tuple[str, str, Properties]:
    label = validate_identifier(getattr(value, "type", None), field="label")
    node_id = validate_node_id(getattr(value, "id", None))
    return label, node_id, _with_scope(getattr(value, "properties", None), scope_key)


def _scoped_properties(properties: dict[str, Any] | None, scope_key: str | None) -> Properties:
    scoped = validate_properties(properties)
    if "scope_key" in scoped and scoped["scope_key"] != scope_key:
        msg = "scope_key cannot differ from the active namespace."
        raise GraphMemoryValidationError(msg)
    if scope_key is not None:
        scoped["scope_key"] = scope_key
    return scoped


def _node_pk(label: str, node_id: str, scope_key: str | None) -> str:
    return json.dumps([scope_key, label, node_id], separators=(",", ":"))


def _scope_where(alias: str, scope_key: str | None) -> tuple[str, dict[str, Any]]:
    if scope_key is None:
        return f"{alias}.scope_key IS NULL", {}
    return f"{alias}.scope_key = $scope_key", {"scope_key": scope_key}
