# deepagents-graph-memory

`deepagents-graph-memory` is an experimental Virtual Graph System (VGS) for LangChain Deep Agents.

The goal is to give a long-running agent a structured graph scratchpad for workflow context: situations, rationales, actions, outcomes, files, tool calls, failures, evidence, experiments, dependencies, and decisions.

VGS is domain-agnostic: the Situation/Rationale/Action/Outcome trace shape and the recall path apply to any project's workflow, not a specific one. The schema beyond traces (node labels, relationship names) is whatever the agent's domain needs.

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

Install this VGS package when graph memory is enabled:

```bash
pip install deepagents-graph-memory
```

Installing `deepagents-graph-memory` installs the VGS runtime dependencies, including Kuzu and LangChain Community. Normal Deep Agents usage stays lightweight because upstream `deepagents` does not import this package unless VGS is explicitly used.

## Basic Usage

```python
from deepagents import create_deep_agent
from deepagents_graph_memory import GraphMemoryBackend, graph_memory_tools, register_vgs_harness_profile

MODEL = "google_genai:gemini-3.5-flash"

register_vgs_harness_profile(MODEL)

graph_backend = GraphMemoryBackend.create()

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

VGS is Kuzu-only. Importing `deepagents_graph_memory` requires the VGS runtime dependencies instead of falling back to another store.

`GraphMemoryBackend.create()` creates a Kuzu in-memory graph using `kuzu.Database(":memory:")`. The data lives in the Python process's RAM, so it works on a laptop, VM, or container while the process is alive, but it is lost on restart and is not shared across multiple workers.

Kuzu-backed recall uses Kuzu full-text search to find seed nodes, relationship-label search for relationship-oriented questions, and bounded graph traversal to recover connected context. Vector search and graph algorithms are intentionally not part of the default recall path; those are better suited to optional ranking, summarization, or domain-specific extensions.

There is no manual graph reset API. To start fresh, start a new Python process or create a new backend instance.

## Graph Tools

```python
graph_memory_tools(graph_backend)
```

The tools are:

- `recall_graph_memory`
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

Low-level graph write tools are available for application builders, but are not exposed by default because unconstrained agents can create drifting labels and relationship types over time:

```python
graph_memory_tools(graph_backend, include_low_level_writes=True)
```

This adds:

- `add_graph_node`
- `add_graph_edge`
- `add_graph_documents`

For production use, prefer domain-specific tools that call `graph_backend.add_graph_node(...)` and `graph_backend.add_graph_edge(...)` with your application's approved labels and relationships.

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

## Example Domain: SRE

VGS itself is domain-agnostic; this is one example of how a project might shape its
own labels and relationships on top of it.

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

Tests and local development should use `GraphMemoryBackend.create()`. It always creates a fresh Kuzu in-memory graph.
