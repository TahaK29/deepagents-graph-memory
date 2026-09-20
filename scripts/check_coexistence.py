"""Check upgrades that leave the previously required standalone Ladybug installed."""

import subprocess
import sys

for imports in (
    "import ladybug; ladybug.Database.get_version(); import deepagents_graph_memory",
    "import deepagents_graph_memory; import ladybug; ladybug.Database.get_version()",
):
    subprocess.run(
        [
            sys.executable,
            "-c",
            imports + "; graph = deepagents_graph_memory.GraphMemoryBackend.create(); "
            "graph.add_graph_node('File', 'parser', properties={'description': 'saffron'}); "
            "assert graph.store.search('saffron').items; graph.close()",
        ],
        check=True,
        timeout=45,
    )
print("Standalone Ladybug coexistence: both import orders passed")
