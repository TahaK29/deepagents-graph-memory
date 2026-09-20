# deepagents-graph-memory

An experimental context-graph library for LangChain Deep Agents. Record linked
findings, actions, and outcomes in LadybugDB, then retrieve related work through
keyword search and graph traversal. Keep files and full tool output in your
existing filesystem backend.

Created by **Pranav Bedi and Taha Khan**.

[![PyPI](https://img.shields.io/pypi/v/deepagents-graph-memory)](https://pypi.org/project/deepagents-graph-memory/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://raw.githubusercontent.com/TahaK29/deepagents-graph-memory/main/assets/vgs-graph.png" alt="Virtual Graph System: connected reasoning traces" width="50%">
</p>

## Installation

Requires Python **3.11–3.14** and a compatible OpenSSL 3 runtime. `pip` installs
Deep Agents and LadybugDB, but not OpenSSL. On macOS with Homebrew, run
`brew install openssl@3` first; see [platform requirements](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#requirements)
for Windows setup and supported wheels.

```bash
pip install deepagents-graph-memory
```

### Full-text search setup

Install LadybugDB's search extension once, with network access, before running
graph search or recall:

```python
import ladybug

with ladybug.Database(":memory:", buffer_pool_size=64 * 1024 * 1024) as db:
    with ladybug.Connection(db) as conn:
        conn.execute("INSTALL fts;").close()
        conn.execute("LOAD fts;").close()
```

Provision it for the same runtime user, LadybugDB version, OS, and architecture.
For containers, retain the extension cache in the runtime image or volume.
See [installation details](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#installation).

## Quick Start

Configure your model provider and credentials, then add graph tools and a
read-only `/graph/` mount alongside the normal filesystem:

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents_graph_memory import (
    GraphMemoryBackend,
    graph_context_middleware,
    graph_memory_tools,
)

MODEL = "google_genai:gemini-3.5-flash"

# In-memory LadybugDB graph; no database path required
graph_backend = GraphMemoryBackend.create()
graph_tools = graph_memory_tools(graph_backend)

agent = create_deep_agent(
    model=MODEL,
    tools=graph_tools,
    middleware=[graph_context_middleware()],
    memory=["/graph/index.md", "/graph/schema.md"],
    backend=CompositeBackend(
        default=StateBackend(),  # Working files and offloaded tool results.
        routes={"/graph/": graph_backend},  # Read-only graph inspection.
    ),
    subagents=[{
        # Configure the default worker explicitly so it also gets graph guidance.
        "name": "general-purpose",
        "description": "Investigate a focused task and return findings with evidence.",
        "system_prompt": "Complete the assigned task and report findings with source paths.",
        "tools": graph_tools,
        "middleware": [graph_context_middleware()],
    }],
)
```

Close `graph_backend` after all agents finish using it. The two graph tools are
`record_graph_trace` (write) and `recall_graph_memory` (read). Subagents need their
own tool and middleware configuration, as shown above.

## Storage and limits

The graph lives in memory by default and disappears when its store closes or the
process exits. For persistence, use `GraphMemoryBackend.create(path="project.lbdb")`
on durable storage; each writable database has one owning process. Graph storage
doesn't persist your filesystem evidence. See [persistence and deployment](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#persistent-storage).

Namespaces scope project data; they aren't an authorization boundary. The graph
stores caller-supplied evidence and resolutions, but doesn't detect contradictions
or verify claims. Recall has traversal and output limits and can return incomplete
context. Ordinary preferences and notes belong in your existing memory backend.

## Documentation

- [Usage guide](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md): shared agents, evidence, conflicts, retries, tools, and deployment.
- [Inspecting the graph](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#inspecting-the-graph): debug through Python without a model or provider key.
- [Design and rationale](https://github.com/TahaK29/deepagents-graph-memory/blob/main/DESIGN.md): the context-graph model and implementation boundaries.
- [Evaluations](https://github.com/TahaK29/deepagents-graph-memory/blob/main/evals/README.md): offline workflow scenarios and an optional model comparison; offline passes don't establish better model decisions.

For development, install `pip install -e ".[test]"`, provision FTS as above, then
run `python -m pytest` and `python -m ruff check .`.
[Report issues](https://github.com/TahaK29/deepagents-graph-memory/issues).

## License

MIT
