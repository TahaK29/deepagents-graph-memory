"""Optional Kuzu adapter for graph memory."""

from __future__ import annotations

import json
from typing import Any, cast

from deepagents_graph_memory.errors import GraphMemoryConfigurationError
from deepagents_graph_memory.paths import node_path, validate_identifier, validate_node_id
from deepagents_graph_memory.stores import (
    GraphEdge,
    GraphNode,
    LimitedResult,
    NeighborhoodResult,
    Properties,
    SearchItem,
    SearchResult,
    validate_properties,
)

try:
    import kuzu
except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
    raise ImportError("Kuzu support requires the `kuzu` package.") from exc

try:
    from langchain_community.graphs.kuzu_graph import KuzuGraph
except ImportError:  # pragma: no cover - exercised when optional extra is absent
    try:
        from langchain_kuzu.graphs.kuzu_graph import KuzuGraph
    except ImportError as fallback_exc:
        raise ImportError("Kuzu support requires `langchain-community` or a compatible `langchain-kuzu` package.") from fallback_exc


class KuzuGraphStore:
    """Internal adapter for LangChain's Kuzu graph integration."""

    def __init__(self, graph: Any) -> None:
        """Initialize the adapter.

        Args:
            graph: LangChain Kuzu graph object.
        """
        self.graph = graph

    @classmethod
    def local(cls, path: str) -> KuzuGraphStore:
        """Create a local Kuzu graph store.

        Args:
            path: Kuzu database path.

        Returns:
            Configured store adapter.
        """
        database = kuzu.Database(path)
        return cls(KuzuGraph(database, allow_dangerous_requests=True))

    def get_schema(self, *, scope_key: str | None = None) -> str:
        """Return graph schema text."""
        del scope_key
        refresh_schema = getattr(self.graph, "refresh_schema", None)
        if callable(refresh_schema):
            refresh_schema()
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
        rows = self._query(f"MATCH (n:{label}) RETURN n.id AS id, n.scope_key AS scope_key LIMIT {int(limit) + 1}", {})
        ids = [str(row["id"]) for row in rows if row.get("id") is not None and self._row_scope_matches(row, scope_key)]
        return LimitedResult(items=ids[:limit], truncated=len(ids) > limit)

    def get_node(self, label: str, node_id: str, *, scope_key: str | None = None) -> GraphNode | None:
        """Return a single node."""
        validate_identifier(label, field="label")
        validate_node_id(node_id)
        rows = self._query(f"MATCH (n:{label} {{id: $id}}) RETURN n", {"id": node_id})
        if not rows:
            return None
        node = _coerce_node(rows[0].get("n"), fallback_label=label, fallback_id=node_id)
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
        del depth, max_nodes
        node = self.get_node(label, node_id, scope_key=scope_key)
        if node is None:
            return None
        limit = int(max_edges) + 1
        rows = self._query(f"MATCH (n:{label} {{id: $id}})-[r]->(m) RETURN n, r, m LIMIT {limit}", {"id": node_id})
        edges: list[GraphEdge] = []
        for row in rows:
            edge = _coerce_edge(row, source_key="n", target_key="m")
            if edge is not None and _scope_matches(edge.properties, scope_key):
                edges.append(edge)
        incoming_rows = self._query(f"MATCH (m)-[r]->(n:{label} {{id: $id}}) RETURN m, r, n LIMIT {limit}", {"id": node_id})
        for row in incoming_rows:
            edge = _coerce_edge(row, source_key="m", target_key="n")
            if edge is not None and _scope_matches(edge.properties, scope_key):
                edges.append(edge)
        edges = edges[:max_edges]
        return NeighborhoodResult(node=node, edges=edges, truncated_edges=(len(rows) + len(incoming_rows)) > max_edges)

    def search(self, query: str, *, scope_key: str | None = None, limit: int = 20) -> SearchResult:
        """Search graph metadata."""
        needle = query.casefold()
        items: list[SearchItem] = []
        for label in self._labels():
            for node_id in self.list_node_ids(label, scope_key=scope_key, limit=limit).items:
                node = self.get_node(label, node_id, scope_key=scope_key)
                if node is None:
                    continue
                haystack = " ".join([node.label, node.id, json.dumps(node.properties, sort_keys=True)]).casefold()
                if needle in haystack:
                    items.append(SearchItem(path=node_path(node.label, node.id), title=f"{node.label}: {node.id}", text=""))
                if len(items) >= limit:
                    return SearchResult(items=items, truncated=True)
        return SearchResult(items=items)

    def add_node(self, label: str, node_id: str, *, properties: Properties | None = None, scope_key: str | None = None) -> None:
        """Add or update a node."""
        validate_identifier(label, field="label")
        validate_node_id(node_id)
        props = validate_properties(properties)
        self._ensure_node_table(label)
        self._query(
            f"""
            MERGE (n:{label} {{id: $id}})
            SET n.type = "entity",
                n.properties = $properties,
                n.scope_key = $scope_key
            """,
            {"id": node_id, "properties": json.dumps(props, sort_keys=True), "scope_key": scope_key},
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

    def add_graph_documents(self, documents: list[Any], *, scope_key: str | None = None) -> None:
        """Add graph documents using LangChain's Kuzu integration."""
        del scope_key
        self.graph.add_graph_documents(documents)

    def _labels(self) -> list[str]:
        schema_getter = getattr(self.graph, "get_schema_dict", None)
        if callable(schema_getter):
            schema = schema_getter()
            return sorted(str(node["label"]) for node in schema.get("nodes", []) if node.get("label") != "Chunk")
        rows = self._query("CALL SHOW_TABLES() RETURN *;", {})
        return sorted(str(row.get("name")) for row in rows if row.get("type") == "NODE")

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

    @staticmethod
    def _row_scope_matches(row: dict[str, Any], scope_key: str | None) -> bool:
        if scope_key is None:
            return True
        value = row.get("scope_key")
        return value == scope_key


def _coerce_node(value: Any, *, fallback_label: str, fallback_id: str) -> GraphNode:
    data = value if isinstance(value, dict) else {}
    label = str(data.get("_label", data.get("label", fallback_label)))
    node_id = str(data.get("id", fallback_id))
    properties = _decode_properties(data)
    for key, item in data.items():
        if key not in {"_id", "_label", "id", "properties"} and item is not None:
            properties.setdefault(key, item)
    return GraphNode(label=label, id=node_id, properties=validate_properties(properties))


def _coerce_edge(row: dict[str, Any], *, source_key: str, target_key: str) -> GraphEdge | None:
    raw_edge = row.get("r")
    if not isinstance(raw_edge, dict):
        return None
    source = _coerce_node(row.get(source_key), fallback_label="Node", fallback_id="")
    target = _coerce_node(row.get(target_key), fallback_label="Node", fallback_id="")
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


def _scope_matches(properties: dict[str, Any], scope_key: str | None) -> bool:
    if scope_key is None:
        return True
    value = properties.get("scope_key")
    return value == scope_key
