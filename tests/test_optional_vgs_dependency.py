import subprocess
import sys
import textwrap


def test_importing_deepagents_does_not_import_kuzu():
    code = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "kuzu" or name.startswith("langchain_community.graphs.kuzu_graph"):
                raise AssertionError(f"unexpected Kuzu import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        import deepagents

        print(deepagents.__name__)
        """
    )

    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert "deepagents" in result.stdout


def test_importing_vgs_package_requires_kuzu():
    code = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "kuzu":
                raise ImportError("blocked Kuzu import")
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

    assert "Kuzu support requires" in result.stdout


def test_importing_backend_requires_kuzu():
    code = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "kuzu":
                raise ImportError("blocked Kuzu import")
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

    assert "Kuzu support requires" in result.stdout
