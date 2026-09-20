"""Logical migration from Kuzu 0.11.3, exercised in a separate CI job."""

import hashlib
import importlib.util
import subprocess
import sys
import textwrap

import ladybug
import pytest

from deepagents_graph_memory import GraphMemoryBackend
from deepagents_graph_memory.errors import GraphMemoryConfigurationError


@pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="Legacy migration CI installs kuzu==0.11.3 separately")
def test_legacy_file_rejected_unchanged_and_logical_export_preserves_graph(tmp_path):
    old_path = tmp_path / "old.kuzu"
    export_path = tmp_path / "export"
    new_path = tmp_path / "migrated.lbdb"
    script = textwrap.dedent(
        """
        import json
        import sys
        import kuzu

        assert kuzu.__version__ == "0.11.3"
        def pk(scope, node_id):
            return json.dumps([scope, "Artifact", node_id], separators=(",", ":"))

        with kuzu.Database(sys.argv[1]) as database, kuzu.Connection(database) as connection:
            connection.execute("CREATE NODE TABLE Artifact(pk STRING PRIMARY KEY, id STRING, type STRING, "
                               "search_text STRING, properties STRING, scope_key STRING)").close()
            connection.execute("CREATE REL TABLE SUPPORTS(FROM Artifact TO Artifact, properties STRING, scope_key STRING)").close()
            for scope in ("project", "other"):
                for node_id in ("report", "decision"):
                    properties = {"description": "orchid evidence", "scope_key": scope, "source": "worker",
                                  "created_at": "2026-09-19T10:00:00+00:00",
                                  "observed_at": "2026-09-19T09:00:00+00:00",
                                  "payload": {"unicode": "café 雪", "values": [True, None, 3, 1.25]}}
                    connection.execute("CREATE (:Artifact {pk:$pk,id:$id,type:'Artifact',search_text:'orchid evidence',"
                                       "properties:$properties,scope_key:$scope})",
                                       {"pk":pk(scope,node_id), "id":node_id, "scope":scope,
                                        "properties":json.dumps(properties)}).close()
                connection.execute("MATCH (a:Artifact {pk:$source}), (b:Artifact {pk:$target}) "
                                   "CREATE (a)-[:SUPPORTS {properties:$properties,scope_key:$scope}]->(b)",
                                   {"source":pk(scope,"report"), "target":pk(scope,"decision"), "scope":scope,
                                    "properties":json.dumps({"source_id":"run-1", "scope_key":scope,
                                                             "recorded_at":"2026-09-19T10:00:00+00:00"})}).close()
            connection.execute("LOAD fts").close()
            connection.execute("CALL CREATE_FTS_INDEX('Artifact','graph_memory_fts',['id','type','search_text','properties'])").close()
        with kuzu.Database(sys.argv[1], read_only=True) as database, kuzu.Connection(database) as connection:
            export = sys.argv[2].replace("\\\\", "/").replace("'", "\\\\'")
            connection.execute("EXPORT DATABASE '" + export + "'").close()
        """
    )
    result = subprocess.run([sys.executable, "-c", script, str(old_path), str(export_path)], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    original_hash = hashlib.sha256(old_path.read_bytes()).digest()
    with pytest.raises(GraphMemoryConfigurationError, match="Legacy Kuzu database files require migration"):
        GraphMemoryBackend.create(path=old_path)
    assert hashlib.sha256(old_path.read_bytes()).digest() == original_hash

    with ladybug.Database(new_path) as database, ladybug.Connection(database) as connection:
        connection.execute("LOAD fts").close()
        export = export_path.as_posix().replace("'", "\\'")
        connection.execute(f"IMPORT DATABASE '{export}'").close()

    backend = GraphMemoryBackend.create(path=new_path)
    try:
        for scope in ("project", "other"):
            assert backend.store.list_node_ids("Artifact", scope_key=scope).items == ["decision", "report"]
            report = backend.store.get_node("Artifact", "report", scope_key=scope)
            assert report.properties == {
                "description": "orchid evidence",
                "scope_key": scope,
                "source": "worker",
                "created_at": "2026-09-19T10:00:00+00:00",
                "observed_at": "2026-09-19T09:00:00+00:00",
                "payload": {"unicode": "café 雪", "values": [True, None, 3, 1.25]},
            }
            edges = backend.store.get_neighbors("Artifact", "report", scope_key=scope).edges
            assert len(edges) == 1
            assert (edges[0].source_id, edges[0].relationship, edges[0].target_id) == ("report", "SUPPORTS", "decision")
            assert edges[0].properties == {"source_id": "run-1", "scope_key": scope, "recorded_at": "2026-09-19T10:00:00+00:00"}
            assert len(backend.store.search("orchid", scope_key=scope).items) == 2
        assert not backend.store.search("orchid", scope_key="absent").items
    finally:
        backend.close()
    assert hashlib.sha256(old_path.read_bytes()).digest() == original_hash
