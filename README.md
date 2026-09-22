# deepagents-graph-memory

[![PyPI](https://img.shields.io/pypi/v/deepagents-graph-memory)](https://pypi.org/project/deepagents-graph-memory/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Give your agent a record of what it tried and what worked.**

For developers building LangChain Deep Agents that handle long-running tasks
and need to remember a lot of context. The graph connects attempts, evidence,
and results so agents can find earlier work before deciding what to do next.

**Works alongside the virtual filesystem (VFS):** files hold code, logs, and tool
dumps; the graph links findings to their evidence.
For example: **failing test → suspected cause → code change → test result**.

<p align="center">
  <img src="https://raw.githubusercontent.com/TahaK29/deepagents-graph-memory/main/assets/vgs-graph.png" alt="Virtual Graph System: connected reasoning traces" width="50%">
</p>

Inspired by Niels de Jong's Neo4j article,
[From recall to reasoning: How context graphs upgrade an agent’s brain](https://neo4j.com/blog/genai/from-recall-to-reasoning-how-context-graphs-upgrade-an-agents-brain/).
Built by **Pranav Bedi and Taha Khan**, using LadybugDB. Experimental.

## Install

```bash
pip install --upgrade deepagents-graph-memory
```

Requires Python 3.11–3.14 on Linux or Apple Silicon Mac; see [supported platforms](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#requirements).
The graph database, OpenSSL, and search extension come bundled. No database server
or separate search setup.

## Quick start: VGS and VFS together

Use both together by default: filesystem tools handle working files and logs,
while graph tools record findings and their relationships. This setup exposes
read-only graph views under `/graph/` and keeps VFS files writable.

Set your model provider's API key, then choose temporary or persistent graph storage.

### Temporary graph

The graph keeps context while the backend is open. Closing it or exiting the
process discards the graph.

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend

from deepagents_graph_memory import (
    GraphMemoryBackend,
    graph_context_middleware,
    graph_memory_tools,
)

graph = GraphMemoryBackend.create()

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=graph_memory_tools(graph),
    middleware=[graph_context_middleware()],
    backend=CompositeBackend(
        default=StateBackend(),  # Working files and tool output.
        routes={"/graph/": graph},  # Read-only graph views.
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Investigate the parser failure and record what you find.",
            }
        ]
    }
)

print(result["messages"][-1].content)
graph.close()
```

### Persistent graph

Supply a path to save the graph on disk, so later runs can reuse the same project's
context. Closing the backend keeps the saved graph.

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend

from deepagents_graph_memory import (
    GraphMemoryBackend,
    graph_context_middleware,
    graph_memory_tools,
)

graph = GraphMemoryBackend.create(path="project.lbdb")

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=graph_memory_tools(graph),
    middleware=[graph_context_middleware()],
    backend=CompositeBackend(
        default=StateBackend(),  # Working files and tool output.
        routes={"/graph/": graph},  # Read-only graph views.
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Investigate the parser failure and record what you find.",
            }
        ]
    }
)

print(result["messages"][-1].content)
graph.close()
```

### Choosing the database path

`path` names the database **file**, including its folder. Relative paths start
from the directory where you run Python, which may differ from the script's folder.

| Path | Where the graph is saved |
| --- | --- |
| `"project.lbdb"` | Your current working directory. |
| `"data/project.lbdb"` | A `data` subfolder of that directory. |
| `"../graph-data/project.lbdb"` | A `graph-data` folder one level above it. |
| `"/data/project.lbdb"` | An absolute location, independent of your working directory. |

The parent folder must exist and be writable. For example, replace the graph
creation line with this to keep the database in your home folder, outside the repo:

```python
from pathlib import Path

storage_dir = Path.home() / "graph-data"
storage_dir.mkdir(parents=True, exist_ok=True)

graph = GraphMemoryBackend.create(path=storage_dir / "project.lbdb")
```

Use `Path("/your/storage/folder")` instead of `Path.home() / "graph-data"` to
choose another absolute location. Opening the same file reuses its saved context.

### Azure and AWS storage

The simplest cloud setup is a Linux VM with a persistent disk:

| Cloud | Run the agent on | Store the graph on |
| --- | --- | --- |
| AWS | EC2 | [Amazon EBS](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-using-volumes.html) |
| Azure | Azure Linux VM | [Azure Managed Disk](https://learn.microsoft.com/en-us/azure/virtual-machines/managed-disks-overview) |

Mount the disk at `/data`, give the app write permission, and use:

```python
graph = GraphMemoryBackend.create(path="/data/project.lbdb")
```

These disks should fit the filesystem-based design, but cloud deployments have
not been tested here. Keep **one process writing to each database**.

For containers, [AWS EFS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/efs-volumes.html)
and [Azure Files](https://learn.microsoft.com/en-us/azure/container-apps/storage-mounts)
provide mounted storage; database locking and recovery still need validation.

**S3 and Azure Blob are backup options, not live database paths.** Close the graph
before uploading its database files, and download them to a filesystem before
reopening. Your application handles uploads and downloads.

See the [persistent storage guide](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#persistent-storage)
for reopening saved graphs, reading previous context, and closing the database.

Keep the database on durable storage. VFS files have a separate lifetime; persist
them with a file backend or a durable checkpointer.

The [full guide](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md)
covers deployment, shared agents, graph-only mode, and debugging. Agents choose
what to record; the graph doesn't automatically verify their claims.

[Report an issue](https://github.com/TahaK29/deepagents-graph-memory/issues) · [MIT license](https://github.com/TahaK29/deepagents-graph-memory/blob/main/LICENSE)
