# deepagents-graph-memory

`deepagents-graph-memory` is an experimental Virtual Graph System (VGS) for LangChain Deep Agents.

The goal is to give a long-running agent a structured graph scratchpad for workflow context: situations, rationales, actions, outcomes, files, tool calls, failures, evidence, experiments, dependencies, and decisions.

This is not ordinary user memory. Do not use it for facts like "Taha likes ice cream." Use it for connected work state like "this failing test led to this hypothesis, this edit, this result, and this final decision."

## Intended Mode

VGS should be off by default in a normal Deep Agents install.

When VGS is enabled:

- Kuzu is the supported graph store.
- Graph memory tools are exposed.
- Deep Agents default VFS tools are hidden: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, and `grep`.
- Generated `/graph/...` markdown views are read-only projections over the graph.
- Graph writes go through controlled tools, not raw graph queries.

This package can provide the backend, tools, and harness-profile helper. A native flag like `create_deep_agent(..., vgs=True)` would require an upstream change in the main `deepagents` library.

## Install

Base install:

```bash
pip install deepagents-graph-memory
```

Runtime VGS usage needs Kuzu:

```bash
pip install "deepagents-graph-memory[kuzu]"
```

Kuzu stays behind the `[kuzu]` extra and lazy import path so installing this package does not force every Deep Agents user to install a graph database.

## Basic Usage

```python
from deepagents import create_deep_agent
from deepagents_graph_memory import GraphMemoryBackend, graph_memory_tools, register_vgs_harness_profile

MODEL = "google_genai:gemini-3.5-flash"

register_vgs_harness_profile(MODEL)

graph_backend = GraphMemoryBackend.local("./graph-context")

agent = create_deep_agent(
    model=MODEL,
    tools=[*graph_memory_tools(graph_backend)],
    memory=[
        "/graph/index.md",
        "/graph/schema.md",
    ],
    backend=graph_backend,
)
```

`register_vgs_harness_profile(MODEL)` is the toggle this package can provide locally. It tells Deep Agents to hide the normal filesystem tools for that model key, so the agent works through graph tools instead of the VFS tool surface.

For a per-run scratchpad, pass a temporary Kuzu directory to `GraphMemoryBackend.local(...)` and delete it when the run ends. For durable project context, reuse the same Kuzu path.

## Graph Tools

```python
graph_memory_tools(graph_backend)
```

The tools are:

- `recall_graph_memory`
- `add_graph_node`
- `add_graph_edge`
- `add_graph_documents`
- `record_graph_trace`

`record_graph_trace` is the preferred high-level write tool for long-running agent work. It records the Level 3 context-graph shape:

```text
Situation -> Rationale -> Action -> Outcome
```

Example:

```text
Situation: sheep saw lion
Rationale: lion is dangerous
Action: sheep ran away
Outcome: sheep survived
```

The graph stores edges like:

```text
sheep saw lion --LED_TO--> lion is dangerous
lion is dangerous --JUSTIFIED--> sheep ran away
sheep ran away --PRODUCED--> sheep survived
```

`recall_graph_memory` searches for seed graph facts, expands through useful edges, and returns compact markdown with source `/graph/...` paths. Callers can pass anchors such as file paths, run ids, task ids, or subagent ids to give recall a concrete starting point.

## Virtual Graph Views

The backend supports these generated markdown paths:

```text
/graph/schema.md
/graph/index.md
/graph/nodes/{label}/{id}.md
/graph/views/neighborhood/{label}/{id}.md
/graph/search/{query}.md
```

These files are inspectable views over the graph, not storage. In VGS mode the normal VFS tools are hidden from the agent, but these paths can still be used by Deep Agents memory loading, direct backend calls, tests, and debugging.

## Writes

Generated graph views are read-only. `write` and `edit` return:

```text
Graph memory views are read-only. Use graph memory tools to add or update graph facts.
```

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

Useful recall:

```text
recall_graph_memory("what services did incident 123 affect and what do they depend on?")
```

## Development Note

`GraphMemoryBackend.ephemeral()` exists for tests and local development. Public VGS runtime usage should use `GraphMemoryBackend.local(...)` with the `[kuzu]` extra installed.
