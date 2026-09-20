"""Disk Ladybug lifecycle and reopening through the public backend factory."""

import gc
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from weakref import ref

import pytest

from deepagents_graph_memory import ladybug_store
from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.errors import GraphMemoryConfigurationError, GraphMemoryValidationError


def test_disk_graph_reopens_with_trace_relationships_scope_and_retry(tmp_path):
    path = tmp_path / "graph.ladybug"
    parent = GraphMemoryBackend.create(path=path, namespace="project")
    child = GraphMemoryBackend(parent.store, namespace="project")
    other = GraphMemoryBackend(parent.store, namespace="other")
    payload = dict(situation="a failure", rationale="logs showed a timeout", action="fixed retry", outcome="passed", operation_id="run-1")
    trace_id = child.record_graph_trace(**payload, artifacts=["src/retry.py"])
    other.record_graph_trace(**payload)
    before = parent.store.get_node("Trace", trace_id, scope_key="project")
    edges = parent.store.get_neighbors("Trace", trace_id, scope_key="project").edges
    assert any(edge.relationship == "HAS_ACTION" for edge in edges)
    parent.close()

    reopened = GraphMemoryBackend.create(path=path, namespace="project")
    assert reopened.store.get_node("Trace", trace_id, scope_key="project") == before
    assert reopened.store.get_neighbors("Trace", trace_id, scope_key="project").edges == edges
    assert "a failure" in reopened.read(f"/nodes/Trace/{trace_id}.md").file_data["content"]
    assert "fixed retry" in reopened.recall_graph_memory("retry", anchors=[f"/nodes/Trace/{trace_id}.md"])
    assert reopened.store.search("fixed retry", scope_key="project").items
    assert reopened.record_graph_trace(**payload, artifacts=["src/retry.py"]) == trace_id
    with pytest.raises(GraphMemoryValidationError, match="operation_id"):
        reopened.record_graph_trace(**{**payload, "outcome": "failed"}, artifacts=["src/retry.py"])
    assert reopened.store.get_node("Trace", trace_id, scope_key="other") is not None
    assert reopened.store.list_node_ids("Trace", scope_key="project").items == [trace_id]
    reopened.close()


def test_close_is_idempotent_and_shared_and_rejects_active_transaction(tmp_path):
    backend = GraphMemoryBackend.create(path=tmp_path / "graph.ladybug")
    sibling = GraphMemoryBackend(backend.store)
    with backend.store.transaction():
        with pytest.raises(GraphMemoryConfigurationError, match="transaction"):
            backend.close()
        sibling.add_graph_node("File", "committed")
    assert backend.store.get_node("File", "committed") is not None
    backend.close()
    backend.close()
    with pytest.raises(GraphMemoryConfigurationError, match="closed"):
        sibling.add_graph_node("File", "later")
    with pytest.raises(GraphMemoryConfigurationError, match="closed"):
        sibling.store.get_node("File", "committed")


@pytest.mark.parametrize("value", ["", "  ", ":memory:", "s3://bucket/db", "file:///tmp/db", "bad\x00path", 123])
def test_invalid_persistent_path_is_rejected(value):
    with pytest.raises(GraphMemoryConfigurationError, match="path"):
        GraphMemoryBackend.create(path=value)


def test_missing_mount_and_directory_database_path_fail_without_fallback(tmp_path):
    with pytest.raises(GraphMemoryConfigurationError, match="parent"):
        GraphMemoryBackend.create(path=tmp_path / "missing" / "graph.ladybug")
    assert not (tmp_path / "missing").exists()
    (tmp_path / "directory.ladybug").mkdir()
    with pytest.raises(GraphMemoryConfigurationError, match="directory"):
        GraphMemoryBackend.create(path=tmp_path / "directory.ladybug")


def test_disk_database_lock_releases_on_close(tmp_path):
    path = tmp_path / "graph.ladybug"
    backend = GraphMemoryBackend.create(path=path)
    script = "from deepagents_graph_memory.backend import GraphMemoryBackend; GraphMemoryBackend.create(path=__import__('sys').argv[1]).close()"
    command = [sys.executable, "-c", script, str(path)]
    blocked = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert blocked.returncode != 0
    assert "Could not open LadybugDB graph" in blocked.stderr and "lock" in blocked.stderr.lower()
    backend.close()
    assert subprocess.run(command, capture_output=True, check=False, timeout=30).returncode == 0


def test_same_process_reuses_store_instead_of_reopening_aliases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    path = tmp_path / "graph.ladybug"
    owner = GraphMemoryBackend.create(path=path)
    hardlink = tmp_path / "hardlink.ladybug"
    os.link(path, hardlink)
    for alias in (path, Path(os.path.relpath(path)), hardlink):
        with pytest.raises(GraphMemoryConfigurationError, match="reuse.*store"):
            GraphMemoryBackend.create(path=alias)
    owner.close()
    GraphMemoryBackend.create(path=path).close()


def test_same_process_rejects_symlink_alias(tmp_path):
    path = tmp_path / "graph.ladybug"
    symlink = tmp_path / "alias.ladybug"
    try:
        symlink.symlink_to(path)
    except OSError as exc:
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows account lacks symlink privilege")
        raise
    owner = GraphMemoryBackend.create(path=path)
    try:
        with pytest.raises(GraphMemoryConfigurationError, match="reuse.*store"):
            GraphMemoryBackend.create(path=symlink)
    finally:
        owner.close()


def test_concurrent_disk_factories_allow_one_owner(tmp_path):
    path = tmp_path / "graph.ladybug"
    start = Barrier(2)

    def open_graph():
        start.wait()
        try:
            return GraphMemoryBackend.create(path=path)
        except GraphMemoryConfigurationError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: open_graph(), range(2)))
    owners = [result for result in results if isinstance(result, GraphMemoryBackend)]
    assert len(owners) == 1
    assert len([result for result in results if isinstance(result, GraphMemoryConfigurationError)]) == 1
    owners[0].close()


def test_failed_initialization_releases_database_for_retry(tmp_path, monkeypatch):
    path = tmp_path / "graph.ladybug"
    original = ladybug_store._LadybugGraph.refresh_schema

    def fail_once(graph):
        monkeypatch.setattr(ladybug_store._LadybugGraph, "refresh_schema", original)
        raise RuntimeError("schema unavailable")

    monkeypatch.setattr(ladybug_store._LadybugGraph, "refresh_schema", fail_once)
    with pytest.raises(GraphMemoryConfigurationError, match="schema unavailable"):
        GraphMemoryBackend.create(path=path)
    GraphMemoryBackend.create(path=path).close()


def test_disk_registry_does_not_keep_abandoned_store_alive(tmp_path):
    path = tmp_path / "graph.ladybug"
    backend = GraphMemoryBackend.create(path=path)
    store_ref = ref(backend.store)
    del backend
    gc.collect()
    assert store_ref() is None
    GraphMemoryBackend.create(path=path).close()


def test_relative_path_stays_open_after_working_directory_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    path = tmp_path / "graph.ladybug"
    backend = GraphMemoryBackend.create(path=os.path.relpath(path))
    backend.add_graph_node("File", "saved")
    monkeypatch.chdir(tmp_path)
    backend.close()
    reopened = GraphMemoryBackend.create(path=path)
    assert reopened.store.get_node("File", "saved") is not None
    reopened.close()


def test_backend_does_not_close_an_external_adapter():
    class ExternalAdapter:
        def close(self):
            raise AssertionError("external resource is caller-owned")

    GraphMemoryBackend(ExternalAdapter()).close()


def test_committed_write_survives_process_exit_without_close(tmp_path):
    path = tmp_path / "graph.ladybug"
    script = (
        "from deepagents_graph_memory.backend import GraphMemoryBackend; "
        "import os, sys; b=GraphMemoryBackend.create(path=sys.argv[1]); "
        "b.add_graph_node('File', 'committed'); os._exit(0)"
    )
    assert subprocess.run([sys.executable, "-c", script, os.fspath(path)], check=False, timeout=30).returncode == 0
    backend = GraphMemoryBackend.create(path=Path(path))
    assert backend.store.get_node("File", "committed") is not None
    backend.close()


def test_uncommitted_write_is_rolled_back_after_process_exit(tmp_path):
    path = tmp_path / "graph.ladybug"
    backend = GraphMemoryBackend.create(path=path)
    backend.add_graph_node("File", "committed")
    backend.close()
    script = """from deepagents_graph_memory.backend import GraphMemoryBackend
import os
import sys

b = GraphMemoryBackend.create(path=sys.argv[1])
with b.store.transaction():
    b.add_graph_node("File", "partial")
    os._exit(0)
"""
    assert subprocess.run([sys.executable, "-c", script, os.fspath(path)], check=False, timeout=30).returncode == 0
    reopened = GraphMemoryBackend.create(path=path)
    assert reopened.store.get_node("File", "committed") is not None
    assert reopened.store.get_node("File", "partial") is None
    reopened.close()


def test_repeated_parameterized_writes_in_clean_process():
    """Keep native prepared-statement regressions from crashing the pytest process."""
    script = """from deepagents_graph_memory.backend import GraphMemoryBackend

backend = GraphMemoryBackend.create(namespace="project")
for revision in range(4):
    for name in ("api", "worker"):
        backend.add_graph_node("Service", name, properties={"revision": revision, "owner": name})
        backend.add_graph_edge("Service", name, "DEPENDS_ON", "Database", "storage", properties={"revision": revision, "owner": name})
        node = backend.store.get_node("Service", name, scope_key="project")
        assert node.properties["revision"] == revision and node.properties["owner"] == name
        edge = backend.store.get_neighbors("Service", name, scope_key="project").edges[0]
        assert edge.properties["revision"] == revision and edge.properties["owner"] == name
        assert edge.source_id == name and edge.target_id == "storage"
backend.close()
"""
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", script], capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
