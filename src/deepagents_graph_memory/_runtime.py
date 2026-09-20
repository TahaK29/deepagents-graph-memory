"""Load the bundled runtime under Ladybug's canonical native module name."""

from __future__ import annotations

import ctypes
import sys
from functools import cache
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location
from pathlib import Path
from typing import Any

from deepagents_graph_memory.errors import GraphMemoryConfigurationError

_VENDOR = Path(__file__).resolve().parent / "_vendor"


class _BundledLadybug(MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname == "ladybug":
            return spec_from_file_location(fullname, _VENDOR / "ladybug" / "__init__.py")
        return None


def load_ladybug() -> Any:
    """Use one module identity, including when standalone Ladybug was imported first."""
    if not (_VENDOR / "ladybug").is_dir():
        # Source/editable development uses the explicitly installed test dependency.
        import ladybug

        return ladybug
    finder = _BundledLadybug()
    sys.meta_path.insert(0, finder)
    try:
        import ladybug as module
    finally:
        sys.meta_path.remove(finder)
    if module.Database.get_version() != "0.20.3":
        msg = "Graph memory bundles LadybugDB 0.20.3; another Ladybug version is already loaded. Use a separate process."
        raise ImportError(msg)
    return module


@cache
def _preload_windows_fts(path: Path) -> Any:
    # Ladybug uses LoadLibraryW, which ignores Python's added DLL directories.
    # ctypes uses LoadLibraryEx with secure search flags; retain its handle.
    return ctypes.CDLL(str(path))


def fts_load_query() -> str:
    """Load the wheel's FTS binary directly, without using a user extension cache."""
    name = "libfts.dll" if sys.platform == "win32" else "libfts.so"
    path = _VENDOR / name
    if not (_VENDOR / "ladybug").is_dir():
        return "LOAD fts;"
    if not path.is_file():
        msg = "Bundled LadybugDB search extension is missing; reinstall deepagents-graph-memory."
        raise GraphMemoryConfigurationError(msg)
    if sys.platform == "win32":
        try:
            _preload_windows_fts(path)
        except OSError as exc:
            msg = "Bundled LadybugDB search dependencies could not load; reinstall deepagents-graph-memory."
            raise GraphMemoryConfigurationError(msg) from exc
    escaped = path.as_posix().replace("\\", "\\\\").replace("'", "\\'")
    return f"LOAD EXTENSION '{escaped}';"
