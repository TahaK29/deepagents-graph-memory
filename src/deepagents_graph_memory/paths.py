"""Virtual graph path parsing and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, unquote

from deepagents_graph_memory.errors import GraphMemoryPathError, GraphMemoryValidationError

PathKind = Literal["root", "schema", "index", "node", "neighborhood", "search"]

IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
NAMESPACE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9\-_.@+:~]+$")
MAX_ID_LENGTH = 256
MAX_QUERY_LENGTH = 512


@dataclass(frozen=True)
class ParsedGraphPath:
    """Parsed graph-memory virtual path."""

    kind: PathKind
    path: str
    label: str | None = None
    node_id: str | None = None
    query: str | None = None
    had_graph_prefix: bool = False


def validate_identifier(value: str, *, field: str) -> str:
    """Validate a graph label or relationship identifier.

    Args:
        value: Identifier to validate.
        field: User-facing field name for error messages.

    Returns:
        The validated value.

    Raises:
        GraphMemoryValidationError: If the identifier is unsafe.
    """
    if not IDENTIFIER_RE.fullmatch(value):
        msg = f"{field} must start with a letter and contain only letters, numbers, and underscores."
        raise GraphMemoryValidationError(msg)
    return value


def validate_node_id(value: str) -> str:
    """Validate a graph node id for use in virtual paths and graph writes.

    Args:
        value: Node id to validate.

    Returns:
        The validated node id.

    Raises:
        GraphMemoryValidationError: If the id is unsafe.
    """
    if not value or len(value) > MAX_ID_LENGTH:
        msg = f"node_id must be between 1 and {MAX_ID_LENGTH} characters."
        raise GraphMemoryValidationError(msg)
    if any(char in value for char in ("\x00", "/", "\\")) or value in {".", ".."}:
        msg = "node_id must not contain path separators, NUL bytes, or traversal segments."
        raise GraphMemoryValidationError(msg)
    if any(ord(char) < 32 for char in value):
        msg = "node_id must not contain control characters."
        raise GraphMemoryValidationError(msg)
    return value


def validate_search_query(value: str) -> str:
    """Validate a virtual search query.

    Args:
        value: Search query.

    Returns:
        The validated query.

    Raises:
        GraphMemoryPathError: If the query is unsafe.
    """
    if not value or len(value) > MAX_QUERY_LENGTH:
        msg = f"search query must be between 1 and {MAX_QUERY_LENGTH} characters."
        raise GraphMemoryPathError(msg)
    if any(char in value for char in ("\x00", "/", "\\")) or value in {".", ".."}:
        msg = "search query must not contain path separators, NUL bytes, or traversal segments."
        raise GraphMemoryPathError(msg)
    return value


def validate_namespace(namespace: tuple[str, ...]) -> tuple[str, ...]:
    """Validate a Deep Agents-style namespace tuple.

    Args:
        namespace: Namespace components.

    Returns:
        The validated namespace tuple.

    Raises:
        GraphMemoryValidationError: If a component is unsafe.
    """
    if not namespace:
        msg = "namespace must not be empty."
        raise GraphMemoryValidationError(msg)
    for index, component in enumerate(namespace):
        if not isinstance(component, str):
            msg = f"namespace component {index} must be a string."
            raise GraphMemoryValidationError(msg)
        if not component:
            msg = f"namespace component {index} must not be empty."
            raise GraphMemoryValidationError(msg)
        if not NAMESPACE_COMPONENT_RE.fullmatch(component):
            msg = (
                f"namespace component {index} contains disallowed characters. "
                "Only alphanumeric characters, hyphens, underscores, dots, @, +, colons, and tildes are allowed."
            )
            raise GraphMemoryValidationError(msg)
    return namespace


def encode_path_segment(value: str) -> str:
    """Encode a graph path segment.

    Args:
        value: Segment value.

    Returns:
        Percent-encoded segment safe for a virtual path.
    """
    return quote(value, safe="-_.~")


def normalize_graph_path(path: str) -> tuple[str, bool]:
    """Normalize a graph-memory path and strip an optional `/graph` prefix.

    Args:
        path: User-facing path.

    Returns:
        A tuple of normalized internal path and whether `/graph` was present.

    Raises:
        GraphMemoryPathError: If the path is unsafe.
    """
    if not path:
        msg = "path must not be empty."
        raise GraphMemoryPathError(msg)
    if "\x00" in path or "\\" in path:
        msg = "path must not contain NUL bytes or backslashes."
        raise GraphMemoryPathError(msg)
    normalized = path if path.startswith("/") else f"/{path}"
    parts = [part for part in normalized.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        msg = "path traversal is not allowed."
        raise GraphMemoryPathError(msg)

    had_graph_prefix = bool(parts and parts[0] == "graph")
    if had_graph_prefix:
        parts = parts[1:]

    internal = "/" + "/".join(parts)
    if internal != "/" and path.endswith("/"):
        internal += "/"
    return internal, had_graph_prefix


def parse_graph_path(path: str) -> ParsedGraphPath:
    """Parse a read target in the virtual graph filesystem.

    Args:
        path: Path to parse.

    Returns:
        Parsed path data.

    Raises:
        GraphMemoryPathError: If the path is not a supported graph-memory file.
    """
    normalized, had_graph_prefix = normalize_graph_path(path)
    if normalized == "/":
        return ParsedGraphPath(kind="root", path=normalized, had_graph_prefix=had_graph_prefix)
    if normalized == "/schema.md":
        return ParsedGraphPath(kind="schema", path=normalized, had_graph_prefix=had_graph_prefix)
    if normalized == "/index.md":
        return ParsedGraphPath(kind="index", path=normalized, had_graph_prefix=had_graph_prefix)

    parts = [part for part in normalized.split("/") if part]
    if len(parts) == 3 and parts[0] == "nodes" and parts[2].endswith(".md"):
        label = unquote(parts[1])
        node_id = unquote(parts[2][:-3])
        try:
            validate_identifier(label, field="label")
            validate_node_id(node_id)
        except GraphMemoryValidationError as exc:
            raise GraphMemoryPathError(str(exc)) from exc
        return ParsedGraphPath(kind="node", path=normalized, label=label, node_id=node_id, had_graph_prefix=had_graph_prefix)

    if len(parts) == 4 and parts[:2] == ["views", "neighborhood"] and parts[3].endswith(".md"):
        label = unquote(parts[2])
        node_id = unquote(parts[3][:-3])
        try:
            validate_identifier(label, field="label")
            validate_node_id(node_id)
        except GraphMemoryValidationError as exc:
            raise GraphMemoryPathError(str(exc)) from exc
        return ParsedGraphPath(kind="neighborhood", path=normalized, label=label, node_id=node_id, had_graph_prefix=had_graph_prefix)

    if len(parts) == 2 and parts[0] == "search" and parts[1].endswith(".md"):
        query = validate_search_query(unquote(parts[1][:-3]))
        return ParsedGraphPath(kind="search", path=normalized, query=query, had_graph_prefix=had_graph_prefix)

    msg = f"Unsupported graph memory path: {path}"
    raise GraphMemoryPathError(msg)


def node_path(label: str, node_id: str) -> str:
    """Return the virtual path for a node page.

    Args:
        label: Node label.
        node_id: Node id.

    Returns:
        Virtual node path.
    """
    return f"/nodes/{encode_path_segment(label)}/{encode_path_segment(node_id)}.md"


def neighborhood_path(label: str, node_id: str) -> str:
    """Return the virtual path for a node neighborhood page.

    Args:
        label: Node label.
        node_id: Node id.

    Returns:
        Virtual neighborhood path.
    """
    return f"/views/neighborhood/{encode_path_segment(label)}/{encode_path_segment(node_id)}.md"
