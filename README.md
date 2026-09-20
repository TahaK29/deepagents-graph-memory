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

An agent can find an old answer and still be missing why that answer made sense.
Neo4j's [From recall to reasoning](https://neo4j.com/blog/genai/from-recall-to-reasoning-how-context-graphs-upgrade-an-agents-brain/)
inspired this project: connect what happened, why an action was chosen, and what
followed, so the agent can retrieve the reasoning behind earlier work.

This library brings that pattern to Deep Agents. For example, an agent can record:
**a parser test failed → it suspected a missing empty field → it changed the parser → the test passed**,
with links to the file and test output. Later, an agent sharing that graph can
look up what was tried and why, with the evidence to check it.

Use it for long tasks or shared agent work where those connections matter.
Keep full files and logs in your filesystem. The agent chooses what to record
and does the reasoning; the graph stores and retrieves the links. It doesn't
automatically learn rules or verify that a recorded explanation is true.

## Installation

Use Python **3.11–3.14** on macOS 15+ (Apple Silicon or Intel), Windows x64,
or Linux x64/ARM64 with glibc 2.28+.

```bash
pip install --upgrade deepagents-graph-memory
```

The platform packages include LadybugDB, OpenSSL, and the search extension.
The graph needs no database server, separate OpenSSL install, or search setup command.
On Intel Macs, a separate Deep Agents dependency currently requires a source build.
See [platform and development details](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#requirements).

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

For source development, follow the [development setup](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#development),
then run `python -m pytest` and `python -m ruff check .`.
[Report issues](https://github.com/TahaK29/deepagents-graph-memory/issues).

## License

MIT
