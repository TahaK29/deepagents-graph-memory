"""Native Ladybug export/import preserves graph content and invalid files stay intact."""

import pytest

from deepagents_graph_memory import GraphMemoryBackend
from deepagents_graph_memory._runtime import fts_load_query
from deepagents_graph_memory.errors import GraphMemoryConfigurationError
from deepagents_graph_memory.ladybug_store import ladybug


def test_native_export_import_preserves_graph(tmp_path):
    export_path = tmp_path / "export"
    imported_path = tmp_path / "imported.lbdb"
    expected = {}
    source = GraphMemoryBackend.create(path=tmp_path / "source.lbdb")
    try:
        for scope in ("project", "other"):
            scoped = GraphMemoryBackend(source.store, namespace=scope)
            for node_id in ("report", "decision"):
                scoped.add_graph_node(
                    "Artifact",
                    node_id,
                    properties={
                        "description": "orchid evidence",
                        "source": scope,
                        "created_at": "2026-09-19T10:00:00+00:00",
                        "observed_at": "2026-09-19T09:00:00+00:00",
                        "payload": {"unicode": "café 雪", "values": [True, None, 3, 1.25]},
                    },
                )
            scoped.add_graph_edge(
                "Artifact",
                "report",
                "SUPPORTS",
                "Artifact",
                "decision",
                properties={"source_id": "run-1", "recorded_at": "2026-09-19T10:00:00+00:00"},
            )
            expected[scope] = (
                [source.store.get_node("Artifact", node_id, scope_key=scope) for node_id in ("decision", "report")],
                source.store.get_neighbors("Artifact", "report", scope_key=scope).edges,
            )
            assert len(source.store.search("orchid", scope_key=scope).items) == 2
        export = export_path.as_posix().replace("'", "\\'")
        source.store.graph.query(f"EXPORT DATABASE '{export}'")
    finally:
        source.close()

    with ladybug.Database(imported_path) as database, ladybug.Connection(database) as connection:
        connection.execute(fts_load_query()).close()
        connection.execute(f"IMPORT DATABASE '{export}'").close()

    imported = GraphMemoryBackend.create(path=imported_path)
    try:
        for scope, (nodes, edges) in expected.items():
            assert [(edge.source_id, edge.relationship, edge.target_id) for edge in edges] == [("report", "SUPPORTS", "decision")]
            assert imported.store.list_node_ids("Artifact", scope_key=scope).items == ["decision", "report"]
            assert [imported.store.get_node("Artifact", node_id, scope_key=scope) for node_id in ("decision", "report")] == nodes
            assert imported.store.get_neighbors("Artifact", "report", scope_key=scope).edges == edges
            assert len(imported.store.search("orchid", scope_key=scope).items) == 2
        assert not imported.store.search("orchid", scope_key="absent").items
    finally:
        imported.close()


def test_invalid_database_is_rejected_without_changing_bytes(tmp_path):
    path = tmp_path / "invalid.lbdb"
    original = b"not a database\x00" * 512
    path.write_bytes(original)
    with pytest.raises(GraphMemoryConfigurationError):
        GraphMemoryBackend.create(path=path)
    assert path.read_bytes() == original
