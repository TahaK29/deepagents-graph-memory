# - Checks which imports need Ladybug and what happens when it is missing.
# - Cases: Deep Agents alone can load without Ladybug; this graph package and its backend require it.

import subprocess
import sys
import textwrap


def test_importing_deepagents_does_not_import_ladybug():
    code = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ladybug" or name.startswith("ladybug."):
                raise AssertionError(f"unexpected Ladybug import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        import deepagents

        print(deepagents.__name__)
        """
    )

    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert "deepagents" in result.stdout


def test_importing_vgs_package_requires_ladybug():
    code = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ladybug":
                raise ImportError("blocked Ladybug import")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        try:
            import deepagents_graph_memory
        except ImportError as exc:
            print(str(exc))
        else:
            raise AssertionError("expected ImportError")
        """
    )

    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert "LadybugDB support requires" in result.stdout


def test_importing_backend_requires_ladybug():
    code = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ladybug":
                raise ImportError("blocked Ladybug import")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        try:
            from deepagents_graph_memory.backend import GraphMemoryBackend
        except ImportError as exc:
            print(str(exc))
        else:
            raise AssertionError("expected ImportError")
        """
    )

    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert "LadybugDB support requires" in result.stdout


def test_ladybug_fts_without_python_network(tmp_path):
    """Exercise provisioned FTS; Linux CI also runs this in a network namespace."""
    code = textwrap.dedent(
        """
        import socket
        import sys

        def block_network(event, args):
            if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}:
                raise AssertionError(f"unexpected runtime networking: {event}")

        sys.addaudithook(block_network)
        try:
            socket.create_connection(("127.0.0.1", 9))
        except AssertionError:
            pass
        else:
            raise AssertionError("network guard did not block connections")

        import deepagents_graph_memory
        from deepagents_graph_memory.backend import GraphMemoryBackend
        from deepagents_graph_memory.ladybug_store import _LadybugGraph

        original_query = _LadybugGraph.query
        queries = []

        def checked_query(self, query, params=None):
            assert "INSTALL" not in query.upper(), "runtime must not provision extensions"
            queries.append(query)
            return original_query(self, query, params)

        _LadybugGraph.query = checked_query
        backend = GraphMemoryBackend.create(path=sys.argv[1], namespace="project")
        backend.add_graph_node("File", "artifact", properties={"description": "cobalt"})
        assert [item.path for item in backend.store.search("cobalt", scope_key="project").items] == ["/nodes/File/artifact.md"]
        assert not backend.store.search("cobalt", scope_key="other").items
        backend.add_graph_node("File", "artifact", properties={"description": "saffron"})
        assert [item.path for item in backend.store.search("saffron", scope_key="project").items] == ["/nodes/File/artifact.md"]
        assert not backend.store.search("cobalt", scope_key="project").items
        backend.close()

        reopened = GraphMemoryBackend.create(path=sys.argv[1], namespace="project")
        assert [item.path for item in reopened.store.search("saffron", scope_key="project").items] == ["/nodes/File/artifact.md"]
        assert not reopened.store.search("cobalt", scope_key="project").items
        reopened.close()
        assert any("QUERY_FTS_INDEX" in query for query in queries)
        print("Ladybug FTS insert/update/reopen passed")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "offline.ladybug")],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Ladybug FTS insert/update/reopen passed" in result.stdout
