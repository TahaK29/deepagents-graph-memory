# Design

## Purpose

`deepagents-graph-memory` adds a graph-backed context layer to LangChain Deep Agents.

The goal is not to replace Deep Agents memory, invent a new protocol, or store user
profile facts. The goal is to give long-running agents a structured scratchpad for
the relationships behind their work: what they tried, why they tried it, what
artifacts were touched, what failed, what succeeded, and what evidence supports the
current plan.

Deep Agents already have a virtual filesystem (VFS). The VFS is good for raw files,
notes, logs, summaries, large tool outputs, and context offloading. This project is
the graph alternative for relationship-heavy work: when VGS is enabled, the graph
tools should be on and the default VFS tools should be hidden from the agent.

## Core Claim

The project is useful when an agent needs to reason over connected state:

- Which experiments touched this column?
- Which code edits were made because of this failing test?
- Which attempted fixes failed before, and why?
- Which service dependencies are related to this incident?
- Which evidence supports this final decision?
- Which subagent produced this finding, and what source did it depend on?

Plain VFS can store all of this as text, but the agent has to rediscover the
structure every time. A graph stores the structure directly.

The intended architecture is:

```text
Default Deep Agents
  VFS tools on
  graph tools off

VGS mode
  VFS tools hidden
  Kuzu graph backend on
  graph tools on
  structured entities, relationships, provenance, decisions, outcomes
```

## What This Is

This project is a Deep Agents backend and tool pattern for graph context.

It provides:

- A read-only `/graph/` generated path surface.
- Controlled graph write tools.
- A recall tool that retrieves a relevant graph slice under budgets.
- A harness-profile helper that hides the default Deep Agents VFS tools.
- A Kuzu-backed runtime path.

The graph database is the source of truth for graph facts. The Markdown files under
`/graph/` are generated views, not storage.

## What This Is Not

This project should not become:

- A replacement for `/memories/`.
- A user preference memory system.
- A generic graph database UI.
- A raw Cypher tool for agents.
- A vector-only RAG system.
- A universal graph memory protocol.
- A broad memory framework that competes with LangGraph Store or Deep Agents memory.

Normal Deep Agents memory remains the right place for preferences, instructions,
summaries, notes, and other mostly linear text.

## Relationship To Deep Agents VFS

Deep Agents expose file tools such as `ls`, `read_file`, `write_file`, `edit_file`,
`grep`, and `glob`. In VGS mode, this package hides those tools through a
`HarnessProfile` so the agent uses graph tools instead of the normal VFS surface.

The graph is projected into paths such as:

```text
/graph/index.md
/graph/schema.md
/graph/nodes/{label}/{id}.md
/graph/views/neighborhood/{label}/{id}.md
/graph/search/{query}.md
```

These paths exist because Deep Agents backends speak file-like paths internally, and
they remain useful for memory loading, direct backend calls, tests, and debugging.
In the intended VGS mode, the agent-facing read path is `recall_graph_memory`, not
normal VFS file tools. Writes go through graph tools so validation, provenance, and
schema discipline can be enforced.

## Storage Lifetime

Durable storage is not the product promise. The main promise is better structured
context during long-running work.

Kuzu is the supported graph store. VGS uses Kuzu's in-memory database mode:

```python
kuzu.Database(":memory:")
```

The graph is a RAM scratchpad that disappears when the Python process exits. Do
not add on-disk Kuzu paths or manual graph reset APIs.

The graph should not default to user-profile semantics. Preferred
scopes are project/workflow oriented:

- `project_id`
- `workspace_id`
- `agent_id`
- `subagent_id`
- `run_id`
- `tenant_id`

## Why A Graph Can Beat Plain VFS

The graph helps when the useful context is relational:

- Action to outcome.
- Hypothesis to experiment.
- Experiment to metric.
- Failure to retry.
- File edit to test result.
- Incident to affected service.
- Claim to evidence.
- Subagent finding to source.

With plain VFS, the agent has to search text and reconstruct these links from logs
or notes. With the graph, these links are first-class edges.

This should improve:

- Multi-hop recall.
- Resume quality after long work.
- Avoiding repeated failed attempts.
- Explaining decisions from evidence.
- Rubric grading of process quality.
- Subagent coordination where findings need to be merged.

The graph is not automatically better. If writes are noisy, duplicated, stale, or
wrong, the graph can mislead the agent. The graph should be treated as a structured
index over raw evidence, not as the only truth.

## How This Stacks With Deep Agents Features

### Todo List

The native Deep Agents todo list remains the live planner.

```text
Todo list = what should I do next?
Graph = what happened, why, and what did it affect?
```

The graph can mirror important todo lifecycle events, but it should not replace the
todo system.

### VFS

Default Deep Agents VFS remains the right tool for file-like work when VGS is off.
When VGS is on, the default VFS tools should be hidden from the agent.

The graph can still store structured links to artifacts, paths, commits, reports,
tool calls, and evidence, but those links are graph nodes and edges rather than
files the agent edits directly.

### RubricMiddleware

Rubrics can benefit from the graph when grading process, not just final output.

Good graph-aware rubric checks include:

- Every material failed attempt has an outcome.
- Final decisions are linked to evidence.
- The agent did not repeat an experiment already marked failed.
- Code edits are linked to the issue or test failure they address.
- Important tool calls have provenance.

For simple rubrics like "tests pass" or "README has an install section", plain VFS is
enough.

### Subagents

Subagents are useful for context isolation. The graph can help merge their outputs.

Default preference:

- Use one graph per parent run or workspace.
- Scope subagent writes with `subagent_id` and `run_id`.
- Let the main agent recall across subagent outputs through graph traversal.

Separate physical graphs per subagent are simpler to isolate, but make cross-subagent
recall harder. Prefer scoped subgraphs unless isolation is more important than shared
reasoning.

## What Agents Should Write

Agents should not write every thought. They should write durable, queryable facts
that are likely to matter later in the task.

Useful node types:

- `Task`
- `Todo`
- `ToolCall`
- `Artifact`
- `File`
- `Symbol`
- `Hypothesis`
- `Decision`
- `Experiment`
- `Metric`
- `Failure`
- `Outcome`
- `Evidence`
- `Claim`
- `Service`
- `Incident`
- `Runbook`
- `Subagent`

Useful relationship types:

- `CREATED`
- `USED`
- `MODIFIED`
- `PRODUCED`
- `FAILED_BECAUSE`
- `SUCCEEDED_BECAUSE`
- `VALIDATED_BY`
- `INVALIDATED_BY`
- `DECIDED_BECAUSE`
- `RESPONDED_TO`
- `SUPERSEDED`
- `DEPENDS_ON`
- `AFFECTED`
- `RESOLVED_BY`
- `REPORTED_BY`
- `CITES`

The exact schema should stay flexible, but examples should guide agents toward
traceable workflow graphs rather than arbitrary node and edge spam.

## Read Strategy

The graph read path should be budgeted and targeted.

The agent should usually read graph context through:

- `recall_graph_memory(query)` for natural-language recall.
- `/graph/nodes/{label}/{id}.md` for one entity.
- `/graph/views/neighborhood/{label}/{id}.md` for connected context.
- `/graph/search/{query}.md` for literal search.

The recall flow should:

1. Find seed nodes from query terms.
2. Expand through relevant edges.
3. Stop at node, edge, depth, and token budgets.
4. Return compact Markdown with source `/graph/...` paths.
5. Return source paths or artifact ids so another system or developer can inspect
   raw evidence when needed.

## Kuzu Storage Dependency

This package is intentionally Kuzu-first. The graph backend should use a real Kuzu
database in memory.

The main Deep Agents package should not pull Kuzu. That matters if this code is
merged upstream: normal Deep Agents users should not download graph database
dependencies unless they enable VGS. This package is the explicit VGS package, so
installing it installs the VGS runtime dependencies.

VGS package install:

```bash
pip install deepagents-graph-memory
```

Do not maintain a separate Python in-memory graph store. Temporary graph memory
should be backed by Kuzu's in-memory database mode:

```python
kuzu.Database(":memory:")
```

If Kuzu or its LangChain integration is missing, fail with a clear configuration
error when graph memory is imported. Do not silently fall back to a weaker store.

## Safety And Boundaries

Generated graph views are read-only.

Do not expose raw unrestricted graph queries as agent-facing tools. Agent-facing
writes should be controlled and validated.

All writes should validate:

- Labels.
- Node ids.
- Relationship names.
- Namespaces/scopes.
- JSON properties.
- Provenance metadata.

This keeps the graph useful as a structured context layer instead of an unsafe
database shell.

## Evaluation Plan

The project should be judged against a plain VFS baseline.

Suggested baselines:

```text
A. Deep Agents with VFS only
B. Deep Agents with VGS only
C. Deep Agents with VGS plus rubric checks
```

Measure:

- Task success.
- Multi-hop recall accuracy.
- Resume quality.
- Repeated failed attempt rate.
- Decision provenance quality.
- Token usage.
- Tool call count.
- Graph correctness.
- False or stale relationship rate.

The graph is worth keeping if it improves long-running, relationship-heavy tasks
under the same model and budget. It is not worth using for simple notes or short
single-step tasks.

## Success Criteria

This project is working if:

- Agents can inspect graph context through `recall_graph_memory` and generated backend views.
- VGS mode hides default Deep Agents VFS tools.
- Agents can write structured facts without raw graph queries.
- The graph helps answer relationship questions faster than plain files.
- The graph improves resume and non-repetition behavior in long-running tasks.
- The graph remains separate from user memory and `/memories/`.
- Optional graph dependencies stay optional.
- The implementation stays an integration layer, not a new protocol.
