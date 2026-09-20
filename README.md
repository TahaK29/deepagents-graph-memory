# deepagents-graph-memory

[![PyPI](https://img.shields.io/pypi/v/deepagents-graph-memory)](https://pypi.org/project/deepagents-graph-memory/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Give your agent a record of what it tried and what worked.**

For developers building LangChain Deep Agents that debug code, run experiments,
or investigate problems over many steps. Save attempts and results as connected
findings so your agent can look up earlier work before choosing its next step.

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

Requires Python 3.11–3.14 on a [supported platform](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#requirements).
The graph database, OpenSSL, and search extension come bundled. No database server
or separate search setup. Intel Macs have an upstream dependency build requirement;
see the platform details above.

## Quick start

Set your model provider's API key, then add the graph tools and guidance to your agent:

```python
from deepagents import create_deep_agent
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
)

# Use agent.invoke(...) in your application.
# Close the graph after the agent and all its workers finish:
# graph.close()
```

The graph is temporary by default. Use `GraphMemoryBackend.create(path="project.lbdb")`
to keep it on disk. Agents choose what to record and must still check evidence;
the graph doesn't automatically verify claims or resolve disagreements.

## Details

The [complete guide](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md)
covers [persistent storage and deployment](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#persistent-storage),
[shared agents](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#sharing-one-graph-between-agents),
[graph-only mode](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#optional-graph-only-mode),
and [inspecting the graph](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md#inspecting-the-graph),
plus API examples and development.

[Report an issue](https://github.com/TahaK29/deepagents-graph-memory/issues) · [MIT license](https://github.com/TahaK29/deepagents-graph-memory/blob/main/LICENSE)
