# - Lets an agent browse graph context as if it were a set of files.
# - Also saves connected records of what happened, why, and how it turned out.
# - Tests: test_backend_read.py and test_backend_ls.py cover reading and finding context;
#        test_write_behavior.py checks that the file views stay read-only.
#        test_trace.py, test_scope.py, and test_limits.py cover work history, separation, and size limits;
#        test_create_and_clear.py checks how a fresh graph starts.

"""Deep Agents backend implementation for graph memory."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
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
    node_path,
    normalize_graph_path,
    parse_graph_path,
    validate_identifier,
    validate_namespace,
    validate_node_id,
    validate_subject,
)
from deepagents_graph_memory.recall import RecallMode
from deepagents_graph_memory.recall import recall_graph_memory as _recall_graph_memory
from deepagents_graph_memory.renderers import render_index, render_node, render_schema, render_search
from deepagents_graph_memory.stores import GraphStoreAdapter, merge_metadata, utc_now, validate_properties

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
    ) -> None:
        """Initialize a graph memory backend.

        Args:
            store: Internal graph store adapter.
            namespace: Optional Deep Agents-style namespace factory or static namespace.
            max_nodes: Maximum nodes listed or traversed in bounded views.
            max_edges: Maximum edges rendered in node pages.
        """
        self.store = store
        self.namespace = namespace
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    @classmethod
    def create(
        cls,
        *,
        path: str | Path | None = None,
        namespace: Namespace = None,
        max_nodes: int = 50,
        max_edges: int = 100,
    ) -> GraphMemoryBackend:
        """Create a Kuzu graph memory backend.

        Args:
            path: Persistent database file path; omitted for an in-memory graph.
            namespace: Optional Deep Agents-style namespace factory or static namespace.
            max_nodes: Maximum nodes listed or traversed in bounded views.
            max_edges: Maximum edges rendered in node pages.

        Returns:
            Configured graph memory backend.
        """
        return cls(
            KuzuGraphStore.memory() if path is None else KuzuGraphStore.disk(path),
            namespace=namespace,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def close(self) -> None:
        """Close resources owned by a Kuzu store."""
        if isinstance(self.store, KuzuGraphStore):
            self.store.close()

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
        operation_id: str | None = None,
        artifacts: Sequence[str] | None = None,
        evidence: Sequence[str] | None = None,
        evidence_refs: list[dict[str, str]] | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        subagent_id: str | None = None,
        task_id: str | None = None,
        subject: str | None = None,
        observed_at: str | None = None,
        supersedes: list[str] | None = None,
        resolves: list[str] | None = None,
        depends_on: list[str] | None = None,
        finding_type: Literal["state", "interpretation"] = "interpretation",
        **metadata: Any,
    ) -> str:
        """Record a Situation/Rationale/Action/Outcome reasoning trace.

        Args:
            situation: What the agent observed.
            rationale: Why the agent chose the action.
            action: What the agent did.
            outcome: What happened after the action.
            trace_id: Optional caller-provided trace id.
            operation_id: Optional stable identity reused only for retrying this exact request.
            artifacts: Optional files or artifacts involved in the action.
            evidence: Optional evidence supporting the rationale or outcome.
            evidence_refs: Optional references to captured evidence sources.
            run_id: Optional run scope id.
            agent_id: Optional agent id.
            subagent_id: Optional subagent id.
            task_id: Optional task id.
            subject: Stable question about one entity and environment within the namespace.
            observed_at: Time of observation, if known, as a timezone-aware ISO 8601 value.
            supersedes: Earlier state trace IDs this evidenced observation replaces.
            resolves: At least two same-subject traces reviewed by an evidenced resolution.
            depends_on: Existing trace IDs whose findings support this conclusion.
            finding_type: Whether the finding reports mutable state or an interpretation.
            **metadata: Additional JSON-serializable metadata written to trace nodes and edges.

        Returns:
            The trace id used for the recorded graph.
        """
        if operation_id is not None:
            operation_id = _validate_trace_text(operation_id, field="operation_id")
            if trace_id is not None:
                raise GraphMemoryValidationError("trace_id and operation_id cannot both be supplied.")
            trace_id = _value_id("trace", operation_id)
        trace_id = validate_node_id(_new_trace_id() if trace_id is None else trace_id)
        scope_key = self._scope_key()
        trace_context = _without_none(
            {
                "run_id": run_id,
                "agent_id": agent_id,
                "subagent_id": subagent_id,
                "task_id": task_id,
            }
        )
        situation = _validate_trace_text(situation, field="situation")
        rationale = _validate_trace_text(rationale, field="rationale")
        action = _validate_trace_text(action, field="action")
        outcome = _validate_trace_text(outcome, field="outcome")
        if artifacts is not None and (not isinstance(artifacts, Sequence) or isinstance(artifacts, str | bytes)):
            msg = "artifacts must be a sequence of strings."
            raise GraphMemoryValidationError(msg)
        if evidence is not None and (not isinstance(evidence, Sequence) or isinstance(evidence, str | bytes)):
            msg = "evidence must be a sequence of strings."
            raise GraphMemoryValidationError(msg)
        artifacts = [_validate_trace_text(value, field="artifact") for value in artifacts or []]
        evidence = [_validate_trace_text(value, field="evidence") for value in evidence or []]
        evidence_refs = _normalize_evidence_refs(evidence_refs)
        if subject is not None:
            subject = validate_subject(subject)
        if not isinstance(finding_type, str) or finding_type not in {"state", "interpretation"}:
            raise GraphMemoryValidationError("finding_type must be state or interpretation.")
        observed_at = _normalize_observed_at(observed_at) if observed_at is not None else None
        supersedes_supplied = supersedes is not None
        if supersedes_supplied and (not isinstance(supersedes, list) or any(not isinstance(item, str) for item in supersedes)):
            raise GraphMemoryValidationError("supersedes must be a list of trace IDs.")
        supersedes = list(dict.fromkeys(validate_node_id(item) for item in supersedes or []))
        if supersedes and (subject is None or finding_type != "state" or observed_at is None or not (evidence or evidence_refs)):
            raise GraphMemoryValidationError("supersedes requires a subject, state finding, observed_at, and evidence.")
        resolves_supplied = resolves is not None
        if resolves_supplied and (not isinstance(resolves, list) or any(not isinstance(item, str) for item in resolves)):
            raise GraphMemoryValidationError("resolves must be a list of trace IDs.")
        resolves = list(dict.fromkeys(validate_node_id(item) for item in resolves or []))
        if resolves and (subject is None or not (evidence or evidence_refs) or len(resolves) < 2 or supersedes):
            raise GraphMemoryValidationError("resolves requires a subject, evidence, at least two distinct traces, and no supersedes.")
        if depends_on is not None and (not isinstance(depends_on, list) or any(not isinstance(item, str) for item in depends_on)):
            raise GraphMemoryValidationError("depends_on must be a list of trace IDs.")
        depends_on = list(dict.fromkeys(validate_node_id(item) for item in depends_on or []))
        recorded_at = utc_now()
        node_specs = [
            ("Situation", validate_node_id(f"{trace_id}-situation"), situation),
            ("Rationale", validate_node_id(f"{trace_id}-rationale"), rationale),
            ("Action", validate_node_id(f"{trace_id}-action"), action),
            ("Outcome", validate_node_id(f"{trace_id}-outcome"), outcome),
        ]
        reserved = {
            "kind": "reasoning_trace",
            "situation": situation,
            "rationale": rationale,
            "action": action,
            "outcome": outcome,
            "trace_id": trace_id,
            **trace_context,
        }
        if any(key in metadata for key in ("operation_id", "request_fingerprint")):
            raise GraphMemoryValidationError("operation_id and request_fingerprint are reserved trace metadata.")
        if any(
            key in metadata
            for key in ("subject", "finding_type", "observed_at", "recorded_at", "supersedes", "resolves", "depends_on", "evidence", "evidence_refs")
        ):
            raise GraphMemoryValidationError("subject, finding, time, supersession, and evidence metadata must use their explicit arguments.")
        for key, value in reserved.items():
            if key in metadata and metadata[key] != value:
                msg = f"trace metadata cannot override {key}."
                raise GraphMemoryValidationError(msg)
        if "text" in metadata or any(key in metadata for key in ("run_id", "agent_id", "subagent_id", "task_id") if key not in trace_context):
            msg = "trace text and identity metadata must use their explicit arguments."
            raise GraphMemoryValidationError(msg)
        request_fingerprint = None
        if operation_id is not None:
            request = {
                "situation": situation,
                "rationale": rationale,
                "action": action,
                "outcome": outcome,
                "artifacts": sorted(set(artifacts)),
                "evidence": sorted(set(evidence)),
                "evidence_refs": evidence_refs,
                "run_id": run_id,
                "agent_id": agent_id,
                "subagent_id": subagent_id,
                "task_id": task_id,
                "subject": subject,
                "observed_at": observed_at,
                "supersedes": sorted(supersedes),
                "resolves": sorted(resolves),
                "depends_on": sorted(depends_on),
                "finding_type": finding_type,
                "metadata": validate_properties(metadata),
            }
            request_fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        edge_metadata = {**metadata, **trace_context, "trace_id": trace_id}
        shared_metadata = {key: metadata[key] for key in ("source", "created_by") if key in metadata}
        trace_metadata = merge_metadata(
            {
                "kind": "reasoning_trace",
                "situation": situation,
                "rationale": rationale,
                "action": action,
                "outcome": outcome,
                "recorded_at": recorded_at,
                "evidence": evidence,
                "evidence_refs": evidence_refs,
                **({"operation_id": operation_id, "request_fingerprint": request_fingerprint} if operation_id is not None else {}),
                **({"subject": subject, "finding_type": finding_type} if subject is not None else {}),
                **({"observed_at": observed_at} if observed_at is not None else {}),
                **({"depends_on": depends_on} if depends_on else {}),
                **trace_context,
            },
            scope_key=scope_key,
            metadata={**metadata, "source": metadata.get("source", "graph_trace")},
        )
        with self.store.transaction():
            if operation_id is not None:
                existing = self.store.get_node("Trace", trace_id, scope_key=scope_key)
                if existing is not None:
                    if (
                        existing.properties.get("operation_id") == operation_id
                        and existing.properties.get("request_fingerprint") == request_fingerprint
                    ):
                        return trace_id
                    raise GraphMemoryValidationError(f"operation_id {operation_id!r} was already used with a different request.")
            for reviewed_id in resolves:
                reviewed = self.store.get_node("Trace", reviewed_id, scope_key=scope_key)
                if reviewed is None or reviewed.properties.get("subject") != subject:
                    raise GraphMemoryValidationError(f"resolved trace {reviewed_id} must exist for the same subject.")
            for premise_id in depends_on:
                if self.store.get_node("Trace", premise_id, scope_key=scope_key) is None:
                    raise GraphMemoryValidationError(f"dependency trace {premise_id} must exist in the same namespace.")
            for old_id in supersedes:
                old = self.store.get_node("Trace", old_id, scope_key=scope_key)
                if old is None or old.properties.get("subject") != subject or old.properties.get("finding_type") != "state":
                    raise GraphMemoryValidationError(f"superseded trace {old_id} must be an existing state finding for the same subject.")
                old_time = old.properties.get("observed_at")
                if not isinstance(old_time, str) or _normalize_observed_at(old_time) >= observed_at:
                    raise GraphMemoryValidationError(f"superseded trace {old_id} must have an earlier observed_at.")
            for label, node_id in [("Trace", trace_id), *((label, node_id) for label, node_id, _text in node_specs)]:
                if self.store.get_node(label, node_id, scope_key=scope_key) is not None:
                    msg = f"trace node {label}/{node_id} already exists."
                    raise GraphMemoryValidationError(msg)
            self.store.add_node("Trace", trace_id, properties=trace_metadata, scope_key=scope_key)
            if subject is not None:
                subject_id = _value_id("subject", subject)
                existing_subject = self.store.get_node("Subject", subject_id, scope_key=scope_key)
                if existing_subject is not None and existing_subject.properties.get("value") != subject:
                    raise GraphMemoryValidationError(f"Subject/{subject_id} has a conflicting value.")
                if existing_subject is None:
                    self.store.add_node(
                        "Subject", subject_id, properties=merge_metadata({"value": subject}, scope_key=scope_key), scope_key=scope_key
                    )
                self._add_trace_edge("Trace", trace_id, "ABOUT", "Subject", subject_id, scope_key=scope_key, metadata=edge_metadata)
            for old_id in supersedes:
                self._add_trace_edge("Trace", trace_id, "SUPERSEDES", "Trace", old_id, scope_key=scope_key, metadata=edge_metadata)
            for reviewed_id in resolves:
                self._add_trace_edge("Trace", trace_id, "RESOLVES", "Trace", reviewed_id, scope_key=scope_key, metadata=edge_metadata)
            for premise_id in depends_on:
                self._add_trace_edge("Trace", trace_id, "BASED_ON", "Trace", premise_id, scope_key=scope_key, metadata=edge_metadata)

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
                        metadata=edge_metadata,
                    ),
                    scope_key=scope_key,
                )

            self._add_trace_edge(
                "Trace", trace_id, "HAS_SITUATION", "Situation", f"{trace_id}-situation", scope_key=scope_key, metadata=edge_metadata
            )
            self._add_trace_edge(
                "Trace", trace_id, "HAS_RATIONALE", "Rationale", f"{trace_id}-rationale", scope_key=scope_key, metadata=edge_metadata
            )
            self._add_trace_edge("Trace", trace_id, "HAS_ACTION", "Action", f"{trace_id}-action", scope_key=scope_key, metadata=edge_metadata)
            self._add_trace_edge("Trace", trace_id, "HAS_OUTCOME", "Outcome", f"{trace_id}-outcome", scope_key=scope_key, metadata=edge_metadata)
            self._add_trace_edge(
                "Situation",
                f"{trace_id}-situation",
                "LED_TO",
                "Rationale",
                f"{trace_id}-rationale",
                scope_key=scope_key,
                metadata=edge_metadata,
            )
            self._add_trace_edge(
                "Rationale",
                f"{trace_id}-rationale",
                "JUSTIFIED",
                "Action",
                f"{trace_id}-action",
                scope_key=scope_key,
                metadata=edge_metadata,
            )
            self._add_trace_edge(
                "Action", f"{trace_id}-action", "PRODUCED", "Outcome", f"{trace_id}-outcome", scope_key=scope_key, metadata=edge_metadata
            )

            self._link_scope_node("Run", run_id, "HAS_TRACE", trace_id, scope_key=scope_key, metadata=edge_metadata)
            self._link_scope_node("Agent", agent_id, "RECORDED", trace_id, scope_key=scope_key, metadata=edge_metadata)
            self._link_scope_node("Subagent", subagent_id, "RECORDED", trace_id, scope_key=scope_key, metadata=edge_metadata)
            self._link_scope_node("Task", task_id, "HAS_TRACE", trace_id, scope_key=scope_key, metadata=edge_metadata)

            for artifact_text in artifacts:
                artifact_id = _value_id("artifact", artifact_text)
                existing_artifact = self.store.get_node("Artifact", artifact_id, scope_key=scope_key)
                if existing_artifact is not None and existing_artifact.properties.get("value") != artifact_text:
                    msg = f"Artifact/{artifact_id} has a conflicting value."
                    raise GraphMemoryValidationError(msg)
                if existing_artifact is None:
                    self.store.add_node(
                        "Artifact",
                        artifact_id,
                        properties=merge_metadata({"value": artifact_text}, scope_key=scope_key, metadata=shared_metadata),
                        scope_key=scope_key,
                    )
                self._add_trace_edge("Trace", trace_id, "INVOLVED", "Artifact", artifact_id, scope_key=scope_key, metadata=edge_metadata)
                self._add_trace_edge("Action", f"{trace_id}-action", "INVOLVED", "Artifact", artifact_id, scope_key=scope_key, metadata=edge_metadata)

            for evidence_text in evidence:
                evidence_id = _value_id("evidence", evidence_text)
                existing_evidence = self.store.get_node("Evidence", evidence_id, scope_key=scope_key)
                if existing_evidence is not None and existing_evidence.properties.get("value") != evidence_text:
                    msg = f"Evidence/{evidence_id} has a conflicting value."
                    raise GraphMemoryValidationError(msg)
                if existing_evidence is None:
                    self.store.add_node(
                        "Evidence",
                        evidence_id,
                        properties=merge_metadata({"value": evidence_text}, scope_key=scope_key, metadata=shared_metadata),
                        scope_key=scope_key,
                    )
                self._add_trace_edge(
                    "Evidence", evidence_id, "SUPPORTS", "Rationale", f"{trace_id}-rationale", scope_key=scope_key, metadata=edge_metadata
                )
                self._add_trace_edge(
                    "Evidence", evidence_id, "SUPPORTS", "Outcome", f"{trace_id}-outcome", scope_key=scope_key, metadata=edge_metadata
                )
            for ref in evidence_refs:
                source_id = _value_id("evidence-source", ref["source_id"])
                identity = {key: ref[key] for key in ("source_id", "locator", "revision", "observed_at") if key in ref}
                source = self.store.get_node("EvidenceSource", source_id, scope_key=scope_key)
                if source is not None and any(
                    source.properties.get(key) != identity.get(key) for key in ("source_id", "locator", "revision", "observed_at")
                ):
                    raise GraphMemoryValidationError(f"EvidenceSource/{source_id} has conflicting identity metadata.")
                if source is None:
                    self.store.add_node("EvidenceSource", source_id, properties=merge_metadata(identity, scope_key=scope_key), scope_key=scope_key)
                citation_metadata = {key: value for key, value in edge_metadata.items() if key != "summary"}
                if "summary" in ref:
                    citation_metadata["summary"] = ref["summary"]
                self._add_trace_edge("Trace", trace_id, "CITES", "EvidenceSource", source_id, scope_key=scope_key, metadata=citation_metadata)

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
        return []

    def _candidate_files(self, *, scope_key: str | None) -> list[FileInfo]:
        files: list[FileInfo] = [
            {"path": "/index.md", "is_dir": False, "size": 0, "modified_at": ""},
            {"path": "/schema.md", "is_dir": False, "size": 0, "modified_at": ""},
        ]
        for label in self.store.list_labels(scope_key=scope_key, limit=self.max_nodes).items:
            for node_id in self.store.list_node_ids(label, scope_key=scope_key, limit=self.max_nodes).items:
                files.append({"path": node_path(label, node_id), "is_dir": False, "size": 0, "modified_at": ""})
        return files

    def _scope_key(self) -> str | None:
        namespace = self.namespace
        if namespace is None:
            return None
        if isinstance(namespace, str):
            return validate_namespace((namespace,))[0]
        if isinstance(namespace, Sequence) and not callable(namespace):
            return "|".join(validate_namespace(tuple(namespace)))
        if not callable(namespace):
            msg = "namespace must be a string, sequence of strings, or factory."
            raise GraphMemoryValidationError(msg)
        runtime = _get_runtime_or_none()
        try:
            resolved = namespace(runtime)
        except Exception as exc:
            msg = f"namespace factory failed: {exc}"
            raise GraphMemoryValidationError(msg) from exc
        if not isinstance(resolved, tuple):
            msg = "namespace factory must return a tuple of strings."
            raise GraphMemoryValidationError(msg)
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
        if self.store.get_node(label, node_id, scope_key=scope_key) is None:
            node_metadata = {key: metadata[key] for key in ("source", "created_by") if key in metadata}
            self.store.add_node(label, node_id, properties=merge_metadata({}, scope_key=scope_key, metadata=node_metadata), scope_key=scope_key)
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
    return f"trace-{uuid4().hex}"


def _value_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
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


def _normalize_evidence_refs(refs: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if refs is None:
        return []
    if not isinstance(refs, list) or len(refs) > 50:
        raise GraphMemoryValidationError("evidence_refs must be a list of at most 50 references.")
    limits = {"source_id": 512, "locator": 2048, "revision": 512, "observed_at": 64, "summary": 1000}
    normalized: dict[str, dict[str, str]] = {}
    for ref in refs:
        if not isinstance(ref, dict) or not {"source_id", "locator"} <= ref.keys() or ref.keys() - limits.keys():
            raise GraphMemoryValidationError("evidence_refs require source_id and locator and allow only revision, observed_at, and summary.")
        item: dict[str, str] = {}
        for key, limit in limits.items():
            if key not in ref:
                continue
            value = _validate_trace_text(ref[key], field=f"evidence_refs.{key}")
            if len(value) > limit:
                raise GraphMemoryValidationError(f"evidence_refs.{key} must be at most {limit} characters.")
            item[key] = _normalize_observed_at(value) if key == "observed_at" else value
        previous = normalized.get(item["source_id"])
        if previous is not None and previous != item:
            raise GraphMemoryValidationError(f"evidence_refs has conflicting references for source_id {item['source_id']!r}.")
        normalized[item["source_id"]] = item
    return [normalized[source_id] for source_id in sorted(normalized)]


def _normalize_observed_at(value: str) -> str:
    if not isinstance(value, str):
        raise GraphMemoryValidationError("observed_at must be a timezone-aware ISO 8601 string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GraphMemoryValidationError("observed_at must be a timezone-aware ISO 8601 string.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GraphMemoryValidationError("observed_at must be a timezone-aware ISO 8601 string.")
    try:
        return parsed.astimezone(UTC).isoformat()
    except OverflowError as exc:
        raise GraphMemoryValidationError("observed_at is outside the supported datetime range.") from exc


def _has_unsafe_control_char(value: str) -> bool:
    return any(ord(char) < 32 and char not in TRACE_TEXT_ALLOWED_CONTROL_CHARS for char in value)


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
