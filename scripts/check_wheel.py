"""Exercise the installed wheel with no Ladybug installation or extension cache."""

import os
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as directory:
    env = dict(os.environ, HOME=directory, USERPROFILE=directory)
    env.pop("LD_LIBRARY_PATH", None)
    script = r"""
import importlib.abc
import pathlib
import sys

class NoExternalLadybug(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ladybug" or (fullname.startswith("ladybug.") and not all("_vendor" in p for p in path or [])):
            raise AssertionError("wheel must not depend on a separately installed Ladybug")

sys.meta_path.insert(0, NoExternalLadybug())

def block_network(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}:
        raise AssertionError("wheel must not download runtime dependencies")

sys.addaudithook(block_network)
from deepagents_graph_memory import GraphMemoryBackend
from deepagents_graph_memory.ladybug_store import _LadybugGraph, ladybug
assert "_vendor" in str(ladybug.__file__), ladybug.__file__
original = _LadybugGraph.query

def local_query(self, query, params=None):
    assert "INSTALL" not in query.upper(), query
    return original(self, query, params)

_LadybugGraph.query = local_query
path = pathlib.Path(sys.argv[1]) / "wheel.lbdb"
graph = GraphMemoryBackend.create(path=path, namespace="project")
trace = graph.record_graph_trace(situation="saffron parser failed", rationale="empty field", action="fixed parser", outcome="passed")
assert graph.store.search("saffron", scope_key="project").items
assert not graph.store.search("saffron", scope_key="another").items
graph.close()
reopened = GraphMemoryBackend.create(path=path, namespace="project")
assert reopened.store.search("saffron", scope_key="project").items
assert "passed" in reopened.read(f"/graph/nodes/Trace/{trace}.md").file_data["content"]
reopened.close()
assert not list(pathlib.Path(sys.argv[1]).rglob("*.lbug_extension")), "unexpected extension cache"
print("Bundled wheel: offline search, namespaces, and persistence passed")
"""
    command = [sys.executable, "-c", script, directory]
    if sys.platform == "darwin":
        # Deny both the symlinks and Cellar paths, without altering the host.
        policy = "(version 1)(allow default)(deny network*)" + "".join(
            f'(deny file-read* (subpath "{prefix}/{suffix}"))'
            for prefix in ("/opt/homebrew", "/usr/local")
            for suffix in ("opt/openssl@3", "Cellar/openssl@3", "opt/openssl@3.5", "Cellar/openssl@3.5")
        )
        command = ["sandbox-exec", "-p", policy, *command]
    subprocess.run(command, cwd=directory, env=env, check=True, timeout=90)

# Upgrades can leave the standalone package installed. Importing it after graph
# memory must reuse the same native binding, not register pybind types twice.
subprocess.run(
    [sys.executable, "-c", "import deepagents_graph_memory; import ladybug; assert ladybug.Database.get_version() == '0.20.3'"],
    check=True,
    timeout=30,
)
