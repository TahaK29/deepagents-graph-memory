# deepagents-graph-memory

**For developers building coding agents with LangChain Deep Agents, especially
long debugging tasks shared between subagents.**

Let agents save what they tried, why they tried it, and what happened, with links
to the evidence. Another worker can look up earlier attempts instead of piecing
the story together from logs.

For example: **failing test → suspected cause → code change → test result**.
Keep the full files and logs in your filesystem; the graph connects the findings.

Inspired by Niels de Jong's Neo4j article,
[From recall to reasoning: How context graphs upgrade an agent’s brain](https://neo4j.com/blog/genai/from-recall-to-reasoning-how-context-graphs-upgrade-an-agents-brain/).
Built by **Pranav Bedi and Taha Khan**, using LadybugDB. Experimental.

[![PyPI](https://img.shields.io/pypi/v/deepagents-graph-memory)](https://pypi.org/project/deepagents-graph-memory/)

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

Default subagents share the graph and receive the graph instructions automatically.
Each spawned worker gets its own ID; you don't have to define or name each one.

The graph is temporary by default. Use `GraphMemoryBackend.create(path="project.lbdb")`
to keep it on disk. Agents choose what to record and must still check evidence;
the graph doesn't automatically verify claims or resolve disagreements.

## Details

The [complete guide](https://github.com/TahaK29/deepagents-graph-memory/blob/main/docs/guide.md)
covers persistence, deployment, custom workers, debugging, API examples, design,
evaluations, and development.

[Report an issue](https://github.com/TahaK29/deepagents-graph-memory/issues) · [MIT license](https://github.com/TahaK29/deepagents-graph-memory/blob/main/LICENSE)
