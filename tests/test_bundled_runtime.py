"""Keep Windows extension loading on Python's secure DLL search path."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deepagents_graph_memory import _runtime
from deepagents_graph_memory.errors import GraphMemoryConfigurationError


def test_windows_fts_preloads_bundled_dll_once(tmp_path, monkeypatch):
    vendor = tmp_path / "package's runtime"
    (vendor / "ladybug").mkdir(parents=True)
    extension = vendor / "libfts.dll"
    extension.touch()
    loader = Mock()
    monkeypatch.setattr(_runtime, "_VENDOR", vendor)
    monkeypatch.setattr(_runtime, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr("ctypes.CDLL", loader)
    query = _runtime.fts_load_query()
    assert query == _runtime.fts_load_query()
    loader.assert_called_once_with(str(extension))
    assert "package\\'s runtime/libfts.dll" in query
    extension.unlink()
    with pytest.raises(GraphMemoryConfigurationError, match="missing"):
        _runtime.fts_load_query()
