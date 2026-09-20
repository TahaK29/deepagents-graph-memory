# deepagents-graph-memory

Give your Deep Agents a shared record of what they tried, what happened, and how
the findings connect. Built on LadybugDB. Experimental.

Created by **Pranav Bedi and Taha Khan**.

[![PyPI](https://img.shields.io/pypi/v/deepagents-graph-memory)](https://pypi.org/project/deepagents-graph-memory/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://raw.githubusercontent.com/TahaK29/deepagents-graph-memory/main/assets/vgs-graph.png" alt="Virtual Graph System: connected reasoning traces" width="50%">
</p>

## Why use this?

When an agent works on a long task, useful context gets spread across files,
logs, and earlier conversations. This library lets it save the connections:
**a failing test → the fix it tried → the result → the evidence**.

Later, that agent or another agent sharing the graph can look up the related
work. For example: “What have we already tried to fix this parser, and which
test checked the change?” Keep the full files and logs where they are; the graph
records findings and links to them. Agents choose what to record, so it doesn't
automatically capture every action or check whether a finding is true.

## Installation

Use Python **3.11–3.14**. LadybugDB also needs OpenSSL 3, a system library that
`pip` doesn't install. On a Mac with Homebrew, run `brew install openssl@3` first.
For other systems, see [requirements](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#requirements).

```bash
pip install deepagents-graph-memory
```

### Full-text search setup

**One-time step:** run this Python snippet with an internet connection. It
downloads and checks the add-on that lets your agent search saved findings.
The temporary database below is only for setup; it doesn't store your project's graph.

```python
import ladybug

with ladybug.Database(":memory:", buffer_pool_size=64 * 1024 * 1024) as db:
    with ladybug.Connection(db) as conn:
        conn.execute("INSTALL fts;").close()
        conn.execute("LOAD fts;").close()
```

Run it on the machine and under the account that will run your agent. Search
then works without downloading the add-on again. For Docker or cloud deployment,
follow the [setup guide](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#installation).

## Quick Start

This gives one agent tools to save and look up findings while keeping its normal
file tools. Set your model provider's API key first.

```python
from deepagents import create_deep_agent
from deepagents_graph_memory import (
    GraphMemoryBackend,
    graph_context_middleware,
    graph_memory_tools,
)

MODEL = "google_genai:gemini-3.5-flash"

# Temporary graph; add path="project.lbdb" to save it to disk.
graph_backend = GraphMemoryBackend.create()
graph_tools = graph_memory_tools(graph_backend)

agent = create_deep_agent(
    model=MODEL,
    tools=graph_tools,
    middleware=[graph_context_middleware()],
    subagents=[],  # Start with one agent; shared subagent setup is in the guide.
)
```

Call `graph_backend.close()` when you're done. See the
[full example](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#quick-start)
to share the graph with subagents and let agents browse it through file tools.

## Storage and limits

By default, the graph disappears when you close it or stop the program. Use
`GraphMemoryBackend.create(path="project.lbdb")` to keep it on disk. On a server,
keep that file on storage that survives restarts and let only one process open it
for writing. Save your evidence files separately. See [storage and deployment](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#persistent-storage).

Search returns a limited amount of related context, so it may leave things out.
Agents still need to check evidence and resolve disagreements. Namespaces separate
project data but don't enforce access permissions. Keep personal preferences and
ordinary notes in your existing memory backend.

## Documentation

- [Usage guide](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md): shared agents, evidence, conflicts, retries, tools, and deployment.
- [Inspecting the graph](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#inspecting-the-graph): debug through Python without a model or provider key.
- [Design and rationale](https://github.com/TahaK29/deepagents-graph-memory/blob/main/DESIGN.md): the context-graph model and implementation boundaries.
- [Evaluations](https://github.com/TahaK29/deepagents-graph-memory/blob/main/evals/README.md): offline workflow scenarios and an optional model comparison; offline passes don't establish better model decisions.

For development, install `pip install -e ".[test]"`, run the search setup above, then
run `python -m pytest` and `python -m ruff check .`.
[Report issues](https://github.com/TahaK29/deepagents-graph-memory/issues).

## License

MIT
