# deepagents-graph-memory

`deepagents-graph-memory` is an experimental graph-backed virtual filesystem backend for LangChain Deep Agents.

Deep Agents already expose memory through file tools such as `ls`, `read_file`, `write_file`, `edit_file`, `grep`, and `glob`. This package keeps that interface and adds an optional read-only graph memory folder, usually mounted at `/graph/`.

Use normal text memory for preferences, instructions, summaries, and notes. Use graph memory for entities and relationships: services, teams, incidents, dependencies, projects, tools, and runbooks.

The markdown files under `/graph/` are inspectable views over the graph, not the source of truth. The source of truth is the configured graph database. Agents can use `recall_graph_memory` to retrieve a relevant slice of long-term graph memory without knowing exact node paths first.

## Install

```bash
pip install deepagents-graph-memory
```

For the local Kuzu-backed default:

```bash
pip install "deepagents-graph-memory[kuzu]"
```

## Basic Usage

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents_graph_memory import GraphMemoryBackend, graph_memory_tools

graph_backend = GraphMemoryBackend.local("./graph-memory")

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[*graph_memory_tools(graph_backend)],
    memory=[
        "/memories/preferences.md",
        "/graph/index.md",
        "/graph/schema.md",
    ],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=lambda rt: (rt.server_info.user.identity,)),
            "/graph/": graph_backend,
        },
    ),
)
```

## Virtual Paths

The backend supports these generated markdown files:

```text
/graph/schema.md
/graph/index.md
/graph/nodes/{label}/{id}.md
/graph/views/neighborhood/{label}/{id}.md
/graph/search/{query}.md
```

When mounted under Deep Agents `CompositeBackend`, the `/graph/` prefix is stripped before the backend receives paths. This backend handles both prefixed and stripped paths.

## Writes

Generated graph views are read-only. `write` and `edit` return:

```text
Graph memory views are read-only. Use graph memory tools to add or update graph facts.
```

Graph updates go through controlled tools:

```python
graph_memory_tools(graph_backend)
```

The tools are:

- `recall_graph_memory`
- `add_graph_node`
- `add_graph_edge`
- `add_graph_documents`

`recall_graph_memory` searches for seed graph facts, expands through the graph while useful, and returns compact markdown with source `/graph/...` paths. It uses retrieval budgets so the graph can grow as long-term memory without dumping the whole graph into the model context.

Raw unrestricted Cypher is intentionally not exposed as an agent-facing read or write path.

## SRE Example

Graph facts:

```text
Langfuse DEPENDS_ON Redis
Langfuse DEPENDS_ON Postgres
SRE Team OWNS Langfuse
Incident 123 AFFECTED Langfuse
Incident 123 RESOLVED_BY Restart Ingestion Workers Runbook
```

Useful reads:

```text
/graph/nodes/service/langfuse.md
/graph/views/neighborhood/service/langfuse.md
```

Useful recall:

```text
recall_graph_memory("what services did incident 123 affect and what do they depend on?")
```

## Notes

This is an external prototype package. It does not modify Deep Agents core and does not replace `/memories/` or other normal text memory. Kuzu is the first local backend; other graph integrations can be added later through the private adapter boundary used by `GraphMemoryBackend`.
