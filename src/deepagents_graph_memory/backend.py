"""Deep Agents backend implementation for graph memory."""

from __future__ import annotations

import fnmatch
import hashlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

from deepagents_graph_memory.errors import GraphMemoryPathError, GraphMemoryValidationError
from deepagents_graph_memory.kuzu_store import KuzuGraphStore
from deepagents_graph_memory.paths import (
    neighborhood_path,
    node_path,
    normalize_graph_path,
    parse_graph_path,
    validate_identifier,
    validate_namespace,
    validate_node_id,
)
from deepagents_graph_memory.recall import RecallMode
from deepagents_graph_memory.recall import recall_graph_memory as _recall_graph_memory
from deepagents_graph_memory.renderers import render_index, render_neighborhood, render_node, render_schema, render_search
from deepagents_graph_memory.stores import GraphStoreAdapter, merge_metadata

READ_ONLY_ERROR = "Graph memory views are read-only. Use graph memory tools to add or update graph facts."
TRACE_TEXT_ALLOWED_CONTROL_CHARS = frozenset({"\n", "\r", "\t"})
NamespaceFactory = Callable[[Any], tuple[str, ...]]
Namespace = str | Sequence[str] | NamespaceFactory | None


class GraphMemoryBackend(BackendProtocol):
    """Deep Agents backend that projects graph facts into graph context views."""

    def __init__(
        self,
        store: GraphStoreAdapter,
        *,
        namespace: Namespace = None,
        max_nodes: int = 50,
        max_edges: int = 100,
        neighborhood_depth: int = 1,
    ) -> None:
        """Initialize a graph memory backend.

        Args:
            store: Internal graph store adapter.
            namespace: Optional Deep Agents-style namespace factory or static namespace.
            max_nodes: Maximum nodes listed or traversed in bounded views.
            max_edges: Maximum edges rendered in neighborhood views.
            neighborhood_depth: Default neighborhood traversal depth.
        """
        self.store = store
        self.namespace = namespace
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.neighborhood_depth = neighborhood_depth

    @classmethod
    def create(
        cls,
        *,
        namespace: Namespace = None,
        max_nodes: int = 50,
        max_edges: int = 100,
        neighborhood_depth: int = 1,
    ) -> GraphMemoryBackend:
        """Create an in-memory Kuzu graph memory backend.

        Args:
            namespace: Optional Deep Agents-style namespace factory or static namespace.
            max_nodes: Maximum nodes listed or traversed in bounded views.
            max_edges: Maximum edges rendered in neighborhood views.
            neighborhood_depth: Default neighborhood traversal depth.

        Returns:
            Configured graph memory backend.
        """
        return cls(
            KuzuGraphStore.memory(),
            namespace=namespace,
            max_nodes=max_nodes,
            max_edges=max_edges,
            neighborhood_depth=neighborhood_depth,
        )

    def ls(self, path: str) -> LsResult:
        """List graph memory virtual files."""
        try:
            scope_key = self._scope_key()
            normalized, had_graph_prefix = normalize_graph_path(path)
            entries = self._ls_internal(normalized, scope_key=scope_key)
            if had_graph_prefix:
                entries = [_prefix_file_info(entry) for entry in entries]
            return LsResult(entries=entries)
        except (GraphMemoryPathError, GraphMemoryValidationError) as exc:
            return LsResult(error=str(exc), entries=None)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a graph memory virtual file."""
        if offset < 0 or limit < 0:
            return ReadResult(error="offset and limit must be non-negative.")
        try:
            scope_key = self._scope_key()
            parsed = parse_graph_path(file_path)
            if parsed.kind in {"root", "index"}:
                content = render_index()
            elif parsed.kind == "schema":
                content = render_schema(self.store.get_schema(scope_key=scope_key))
            elif parsed.kind == "node":
                assert parsed.label is not None and parsed.node_id is not None
                node = self.store.get_node(parsed.label, parsed.node_id, scope_key=scope_key)
                if node is None:
                    return ReadResult(error=f"Graph node '{parsed.label}/{parsed.node_id}' not found.")
                neighborhood = self.store.get_neighbors(
                    parsed.label,
                    parsed.node_id,
                    scope_key=scope_key,
                    depth=1,
                    max_nodes=self.max_nodes,
                    max_edges=self.max_edges,
                )
                content = render_node(node, neighborhood)
            elif parsed.kind == "neighborhood":
                assert parsed.label is not None and parsed.node_id is not None
                neighborhood = self.store.get_neighbors(
                    parsed.label,
                    parsed.node_id,
                    scope_key=scope_key,
                    depth=self.neighborhood_depth,
                    max_nodes=self.max_nodes,
                    max_edges=self.max_edges,
                )
                if neighborhood is None:
                    return ReadResult(error=f"Graph node '{parsed.label}/{parsed.node_id}' not found.")
                content = render_neighborhood(neighborhood)
            else:
                assert parsed.query is not None
                content = render_search(parsed.query, self.store.search(parsed.query, scope_key=scope_key, limit=self.max_nodes))
            return ReadResult(file_data=_file_data(_slice_lines(content, offset, limit)))
        except (GraphMemoryPathError, GraphMemoryValidationError) as exc:
            return ReadResult(error=str(exc))

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        """Search graph memory metadata for a literal pattern."""
        try:
            scope_key = self._scope_key()
            base_path, had_graph_prefix = normalize_graph_path(path or "/")
            matches: list[GrepMatch] = []
            static_pages = {
                "/index.md": render_index(),
                "/schema.md": render_schema(self.store.get_schema(scope_key=scope_key)),
            }
            for candidate_path, content in static_pages.items():
                if _path_allowed(candidate_path, base_path) and _glob_allowed(candidate_path, glob) and pattern in content:
                    matches.append({"path": _maybe_prefix(candidate_path, had_graph_prefix), "line": 1, "text": content.splitlines()[0]})
            for item in self.store.search(pattern, scope_key=scope_key, limit=self.max_nodes).items:
                if _path_allowed(item.path, base_path) and _glob_allowed(item.path, glob):
                    matches.append({"path": _maybe_prefix(item.path, had_graph_prefix), "line": 1, "text": item.title})
            return GrepResult(matches=matches)
        except (GraphMemoryPathError, GraphMemoryValidationError) as exc:
            return GrepResult(error=str(exc), matches=None)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find graph memory virtual files by glob pattern."""
        try:
            scope_key = self._scope_key()
            base_path, had_graph_prefix = normalize_graph_path(path or "/")
            internal_pattern, _pattern_had_graph_prefix = _normalize_pattern(pattern)
            matches = []
            for candidate in self._candidate_files(scope_key=scope_key):
                if _path_allowed(candidate["path"], base_path) and _glob_allowed(candidate["path"], internal_pattern):
                    matches.append(_prefix_file_info(candidate) if had_graph_prefix else candidate)
            matches.sort(key=lambda item: item["path"])
            return GlobResult(matches=matches)
        except (GraphMemoryPathError, GraphMemoryValidationError) as exc:
            return GlobResult(error=str(exc), matches=None)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Reject writes to generated graph memory views."""
        del content
        return WriteResult(error=READ_ONLY_ERROR, path=file_path)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Reject edits to generated graph memory views."""
        del old_string, new_string, replace_all
        return EditResult(error=READ_ONLY_ERROR, path=file_path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Reject uploads to generated graph memory views."""
        return [FileUploadResponse(path=path, error=READ_ONLY_ERROR) for path, _content in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download generated graph memory views for Deep Agents memory loading."""
        responses = []
        for path in paths:
            result = self.read(path, offset=0, limit=1_000_000)
            if result.error is not None or result.file_data is None:
                error = "file_not_found" if result.error and "not found" in result.error.casefold() else result.error
                responses.append(FileDownloadResponse(path=path, error=error))
                continue
            responses.append(FileDownloadResponse(path=path, content=result.file_data["content"].encode("utf-8")))
        return responses

    def add_graph_node(self, label: str, node_id: str, properties: dict[str, Any] | None = None, **metadata: Any) -> None:
        """Add or update a graph node through a controlled write path."""
        validate_identifier(label, field="label")
        validate_node_id(node_id)
        scope_key = self._scope_key()
        merged = merge_metadata(properties, scope_key=scope_key, metadata=metadata)
        self.store.add_node(label, node_id, properties=merged, scope_key=scope_key)

    def add_graph_edge(
        self,
        source_label: str,
        source_id: str,
        relationship: str,
        target_label: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> None:
        """Add or update a graph edge through a controlled write path."""
        validate_identifier(source_label, field="source_label")
        validate_identifier(target_label, field="target_label")
        validate_identifier(relationship, field="relationship")
        validate_node_id(source_id)
        validate_node_id(target_id)
        scope_key = self._scope_key()
        merged = merge_metadata(properties, scope_key=scope_key, metadata=metadata)
        self.store.add_edge(source_label, source_id, relationship, target_label, target_id, properties=merged, scope_key=scope_key)

    def add_graph_documents(self, documents: Sequence[Any]) -> None:
        """Add LangChain graph documents through the configured adapter."""
        self.store.add_graph_documents(documents, scope_key=self._scope_key())

    def record_graph_trace(
        self,
        *,
        situation: str,
        rationale: str,
        action: str,
        outcome: str,
        trace_id: str | None = None,
        artifacts: Sequence[str] | None = None,
        evidence: Sequence[str] | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        subagent_id: str | None = None,
        task_id: str | None = None,
        **metadata: Any,
    ) -> str:
        """Record a Situation/Rationale/Action/Outcome reasoning trace.

        Args:
            situation: What the agent observed.
            rationale: Why the agent chose the action.
            action: What the agent did.
            outcome: What happened after the action.
            trace_id: Optional caller-provided trace id.
            artifacts: Optional files or artifacts involved in the action.
            evidence: Optional evidence supporting the rationale or outcome.
            run_id: Optional run scope id.
            agent_id: Optional agent id.
            subagent_id: Optional subagent id.
            task_id: Optional task id.
            **metadata: Additional JSON-serializable metadata written to trace nodes and edges.

        Returns:
            The trace id used for the recorded graph.
        """
        trace_id = validate_node_id(trace_id or _new_trace_id())
        scope_key = self._scope_key()
        trace_context = _without_none(
            {
                "run_id": run_id,
                "agent_id": agent_id,
                "subagent_id": subagent_id,
                "task_id": task_id,
            }
        )
        trace_metadata = merge_metadata(
            {
                "kind": "reasoning_trace",
                "situation": _validate_trace_text(situation, field="situation"),
                "rationale": _validate_trace_text(rationale, field="rationale"),
                "action": _validate_trace_text(action, field="action"),
                "outcome": _validate_trace_text(outcome, field="outcome"),
                **trace_context,
            },
            scope_key=scope_key,
            metadata={**metadata, "source": metadata.get("source", "graph_trace")},
        )
        self.store.add_node("Trace", trace_id, properties=trace_metadata, scope_key=scope_key)

        node_specs = [
            ("Situation", f"{trace_id}-situation", situation),
            ("Rationale", f"{trace_id}-rationale", rationale),
            ("Action", f"{trace_id}-action", action),
            ("Outcome", f"{trace_id}-outcome", outcome),
        ]
        for label, node_id, text in node_specs:
            self.store.add_node(
                label,
                node_id,
                properties=merge_metadata(
                    {
                        "text": text,
                        "trace_id": trace_id,
                        **trace_context,
                    },
                    scope_key=scope_key,
                    metadata=metadata,
                ),
                scope_key=scope_key,
            )

        self._add_trace_edge("Trace", trace_id, "HAS_SITUATION", "Situation", f"{trace_id}-situation", scope_key=scope_key, metadata=metadata)
        self._add_trace_edge("Trace", trace_id, "HAS_RATIONALE", "Rationale", f"{trace_id}-rationale", scope_key=scope_key, metadata=metadata)
        self._add_trace_edge("Trace", trace_id, "HAS_ACTION", "Action", f"{trace_id}-action", scope_key=scope_key, metadata=metadata)
        self._add_trace_edge("Trace", trace_id, "HAS_OUTCOME", "Outcome", f"{trace_id}-outcome", scope_key=scope_key, metadata=metadata)
        self._add_trace_edge(
            "Situation",
            f"{trace_id}-situation",
            "LED_TO",
            "Rationale",
            f"{trace_id}-rationale",
            scope_key=scope_key,
            metadata=metadata,
        )
        self._add_trace_edge(
            "Rationale",
            f"{trace_id}-rationale",
            "JUSTIFIED",
            "Action",
            f"{trace_id}-action",
            scope_key=scope_key,
            metadata=metadata,
        )
        self._add_trace_edge("Action", f"{trace_id}-action", "PRODUCED", "Outcome", f"{trace_id}-outcome", scope_key=scope_key, metadata=metadata)

        self._link_scope_node("Run", run_id, "HAS_TRACE", trace_id, scope_key=scope_key, metadata=metadata)
        self._link_scope_node("Agent", agent_id, "RECORDED", trace_id, scope_key=scope_key, metadata=metadata)
        self._link_scope_node("Subagent", subagent_id, "RECORDED", trace_id, scope_key=scope_key, metadata=metadata)
        self._link_scope_node("Task", task_id, "HAS_TRACE", trace_id, scope_key=scope_key, metadata=metadata)

        for artifact in artifacts or []:
            artifact_text = _validate_trace_text(artifact, field="artifact")
            artifact_id = _value_id("artifact", artifact_text)
            self.store.add_node(
                "Artifact",
                artifact_id,
                properties=merge_metadata({"value": artifact_text, "trace_id": trace_id}, scope_key=scope_key, metadata=metadata),
                scope_key=scope_key,
            )
            self._add_trace_edge("Trace", trace_id, "INVOLVED", "Artifact", artifact_id, scope_key=scope_key, metadata=metadata)
            self._add_trace_edge("Action", f"{trace_id}-action", "INVOLVED", "Artifact", artifact_id, scope_key=scope_key, metadata=metadata)

        for item in evidence or []:
            evidence_text = _validate_trace_text(item, field="evidence")
            evidence_id = _value_id("evidence", evidence_text)
            self.store.add_node(
                "Evidence",
                evidence_id,
                properties=merge_metadata({"value": evidence_text, "trace_id": trace_id}, scope_key=scope_key, metadata=metadata),
                scope_key=scope_key,
            )
            self._add_trace_edge("Evidence", evidence_id, "SUPPORTS", "Rationale", f"{trace_id}-rationale", scope_key=scope_key, metadata=metadata)
            self._add_trace_edge("Evidence", evidence_id, "SUPPORTS", "Outcome", f"{trace_id}-outcome", scope_key=scope_key, metadata=metadata)

        return trace_id

    def recall_graph_memory(
        self,
        query: str,
        *,
        anchors: Sequence[str] | None = None,
        mode: RecallMode = "auto",
        token_budget: int = 2000,
        max_depth: int = 3,
        max_nodes: int = 50,
        max_edges: int = 100,
    ) -> str:
        """Recall relevant graph memory with adaptive traversal.

        Args:
            query: Natural-language recall query.
            anchors: Optional concrete starting hints such as file paths, run ids, task ids, or subagent ids.
            mode: Recall expansion mode. `auto` expands while relevant, `local` reads one hop, and `deep` expands to `max_depth`.
            token_budget: Approximate output token budget.
            max_depth: Maximum traversal depth.
            max_nodes: Maximum nodes to include.
            max_edges: Maximum edges to include.

        Returns:
            Compact markdown with source graph paths.
        """
        return _recall_graph_memory(
            self.store,
            query,
            scope_key=self._scope_key(),
            anchors=anchors,
            mode=mode,
            token_budget=token_budget,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def _ls_internal(self, normalized: str, *, scope_key: str | None) -> list[FileInfo]:
        if normalized == "/":
            return [
                {"path": "/index.md", "is_dir": False, "size": 0, "modified_at": ""},
                {"path": "/nodes/", "is_dir": True, "size": 0, "modified_at": ""},
                {"path": "/schema.md", "is_dir": False, "size": 0, "modified_at": ""},
                {"path": "/search/", "is_dir": True, "size": 0, "modified_at": ""},
                {"path": "/views/", "is_dir": True, "size": 0, "modified_at": ""},
            ]
        if normalized == "/nodes/":
            labels = self.store.list_labels(scope_key=scope_key, limit=self.max_nodes)
            return [{"path": f"/nodes/{label}/", "is_dir": True, "size": 0, "modified_at": ""} for label in labels.items]
        if normalized.startswith("/nodes/") and normalized.endswith("/"):
            parts = [part for part in normalized.split("/") if part]
            if len(parts) == 2:
                label = validate_identifier(parts[1], field="label")
                ids = self.store.list_node_ids(label, scope_key=scope_key, limit=self.max_nodes)
                return [{"path": node_path(label, node_id), "is_dir": False, "size": 0, "modified_at": ""} for node_id in ids.items]
        if normalized == "/views/":
            return [{"path": "/views/neighborhood/", "is_dir": True, "size": 0, "modified_at": ""}]
        if normalized == "/views/neighborhood/":
            labels = self.store.list_labels(scope_key=scope_key, limit=self.max_nodes)
            return [{"path": f"/views/neighborhood/{label}/", "is_dir": True, "size": 0, "modified_at": ""} for label in labels.items]
        if normalized.startswith("/views/neighborhood/") and normalized.endswith("/"):
            parts = [part for part in normalized.split("/") if part]
            if len(parts) == 3:
                label = validate_identifier(parts[2], field="label")
                ids = self.store.list_node_ids(label, scope_key=scope_key, limit=self.max_nodes)
                return [{"path": neighborhood_path(label, node_id), "is_dir": False, "size": 0, "modified_at": ""} for node_id in ids.items]
        return []

    def _candidate_files(self, *, scope_key: str | None) -> list[FileInfo]:
        files: list[FileInfo] = [
            {"path": "/index.md", "is_dir": False, "size": 0, "modified_at": ""},
            {"path": "/schema.md", "is_dir": False, "size": 0, "modified_at": ""},
        ]
        for label in self.store.list_labels(scope_key=scope_key, limit=self.max_nodes).items:
            for node_id in self.store.list_node_ids(label, scope_key=scope_key, limit=self.max_nodes).items:
                files.append({"path": node_path(label, node_id), "is_dir": False, "size": 0, "modified_at": ""})
                files.append({"path": neighborhood_path(label, node_id), "is_dir": False, "size": 0, "modified_at": ""})
        return files

    def _scope_key(self) -> str | None:
        namespace = self.namespace
        if namespace is None:
            return None
        if isinstance(namespace, str):
            return namespace
        if isinstance(namespace, Sequence) and not callable(namespace):
            return "|".join(validate_namespace(tuple(namespace)))
        runtime = _get_runtime_or_none()
        resolved = namespace(runtime)
        return "|".join(validate_namespace(resolved))

    def _add_trace_edge(
        self,
        source_label: str,
        source_id: str,
        relationship: str,
        target_label: str,
        target_id: str,
        *,
        scope_key: str | None,
        metadata: dict[str, Any],
    ) -> None:
        properties = merge_metadata({}, scope_key=scope_key, metadata=metadata)
        self.store.add_edge(source_label, source_id, relationship, target_label, target_id, properties=properties, scope_key=scope_key)

    def _link_scope_node(
        self,
        label: str,
        node_id: str | None,
        relationship: str,
        trace_id: str,
        *,
        scope_key: str | None,
        metadata: dict[str, Any],
    ) -> None:
        if node_id is None:
            return
        validate_node_id(node_id)
        self.store.add_node(label, node_id, properties=merge_metadata({}, scope_key=scope_key, metadata=metadata), scope_key=scope_key)
        self._add_trace_edge(label, node_id, relationship, "Trace", trace_id, scope_key=scope_key, metadata=metadata)


def _get_runtime_or_none() -> Any | None:
    try:
        from langgraph.runtime import get_runtime
    except ImportError:
        return None
    try:
        return get_runtime()
    except (RuntimeError, KeyError):
        return None


def _file_data(content: str) -> dict[str, str]:
    now = datetime.now(UTC).isoformat()
    return {
        "content": content,
        "encoding": "utf-8",
        "created_at": now,
        "modified_at": now,
    }


def _slice_lines(content: str, offset: int, limit: int) -> str:
    lines = content.splitlines()
    if limit == 0:
        return ""
    return "\n".join(lines[offset : offset + limit])


def _prefix_file_info(info: FileInfo) -> FileInfo:
    return {**info, "path": _maybe_prefix(info["path"], True)}


def _maybe_prefix(path: str, use_prefix: bool) -> str:
    if not use_prefix:
        return path
    if path == "/":
        return "/graph/"
    return f"/graph{path}"


def _path_allowed(candidate_path: str, base_path: str) -> bool:
    if base_path == "/":
        return True
    base = base_path if base_path.endswith("/") else f"{base_path}/"
    return candidate_path.startswith(base) or candidate_path == base_path


def _glob_allowed(candidate_path: str, pattern: str | None) -> bool:
    if not pattern:
        return True
    bare_candidate = candidate_path.lstrip("/")
    bare_pattern = pattern.lstrip("/")
    return fnmatch.fnmatch(candidate_path, pattern) or fnmatch.fnmatch(bare_candidate, bare_pattern)


def _normalize_pattern(pattern: str) -> tuple[str, bool]:
    if pattern.startswith("/graph/"):
        return f"/{pattern.removeprefix('/graph/')}", True
    if pattern == "/graph":
        return "/", True
    return pattern, False


def _new_trace_id() -> str:
    return f"trace-{uuid4().hex[:12]}"


def _value_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _validate_trace_text(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        msg = f"{field} must be a string."
        raise GraphMemoryValidationError(msg)
    normalized = value.strip()
    if not normalized:
        msg = f"{field} must not be empty."
        raise GraphMemoryValidationError(msg)
    if _has_unsafe_control_char(normalized):
        msg = f"{field} must not contain NUL bytes or unsafe control characters."
        raise GraphMemoryValidationError(msg)
    return normalized


def _has_unsafe_control_char(value: str) -> bool:
    return any(ord(char) < 32 and char not in TRACE_TEXT_ALLOWED_CONTROL_CHARS for char in value)


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
