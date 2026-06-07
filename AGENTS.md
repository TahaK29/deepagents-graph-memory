# AGENTS.md

## Project Goal

`deepagents-graph-memory` is an experimental Virtual Graph System (VGS) for LangChain Deep Agents.

The goal is to add a graph-backed context scratchpad for long-running agent work: situations, rationales, actions, outcomes, artifacts, failures, evidence, decisions, dependencies, and provenance.

This project is intended to be mergeable on top of the main Deep Agents/LangChain repo without making normal Deep Agents users install graph database dependencies.

## Non-Negotiable Architecture

- Normal upstream Deep Agents install/import must stay lightweight. Do not put Kuzu or graph database packages in upstream Deep Agents base dependencies.
- `deepagents-graph-memory` is the explicit VGS package. Installing or importing this package requires the VGS runtime dependencies, including Kuzu.
- VGS runtime is Kuzu-only and RAM-only. Use Kuzu's `:memory:` mode for graph storage. Do not add on-disk Kuzu paths, path-based constructors, or graph clear/reset APIs.
- Do not recreate or restore a Python `InMemoryGraphStore`.
- Do not add silent fallback storage. If VGS dependencies are missing when `deepagents_graph_memory` or `deepagents_graph_memory.backend` is imported, fail clearly.
- Importing `deepagents_graph_memory` or `deepagents_graph_memory.backend` must import Kuzu.
- Generated `/graph/...` markdown paths are read-only views over graph data, not storage.
- Do not expose raw unrestricted Cypher as an agent-facing tool.
- Default agent-facing graph tools should stay constrained. Prefer `recall_graph_memory` and `record_graph_trace`; expose low-level writes only through explicit opt-in.

## Mental Model

- Deep Agents VFS is for ordinary file-like memory and context offloading.
- VGS is for relationship-heavy workflow context.
- The graph is the source of truth.
- Markdown views are generated projections for backend compatibility, memory loading, tests, and debugging.
- Search finds seed nodes; traversal recovers connected context.
- Recall must stay bounded by depth, node, edge, and token budgets.

## Coding Preferences

- Think before coding. State assumptions when the request is ambiguous.
- Prefer simple, surgical changes over broad refactors.
- Touch only files required by the task. Do not clean up unrelated code.
- Match existing style and naming.
- Do not add abstractions for one use case.
- Do not add fallback logic where code is expected to fail into a weaker path.
- Do not invent library behavior. Check official docs for Kuzu, LangChain, Deep Agents, or LangGraph when behavior matters.
- Preserve user changes in the worktree. Never revert unrelated dirty files.
- Use `rg` / `rg --files` for search.
- Use `apply_patch` for manual file edits.
- Keep comments rare and only for non-obvious logic.
- Prefer ASCII unless the edited file already clearly uses non-ASCII.

## Graph Write Principles

- Avoid unconstrained graph drift.
- Prefer durable, queryable facts over raw thoughts.
- Prefer traceable workflow shapes such as `Situation -> Rationale -> Action -> Outcome`.
- Validate labels, node ids, relationship names, namespaces/scopes, JSON properties, and provenance metadata.
- Low-level generic writes are acceptable as builder APIs, but production agent tools should become domain-specific or schema-validated where possible.
- Do not store ordinary user preferences or profile facts in this graph.

## Graph Read Principles

- Use Kuzu full-text search for node seed retrieval when using Kuzu-backed recall.
- Keep relationship-label matching small and explicit; do not pretend it is semantic vector retrieval.
- Use graph traversal to recover context after finding seeds.
- Do not make `backend.read()` the primary agent recall strategy. It exists for generated views, Deep Agents memory loading, direct backend calls, tests, and debugging.
- `recall_graph_memory` is the main agent-facing read path.

## Dependency Rules

- The VGS system is meant to be part of the Deep Agents/LangChain repo, but normal upstream Deep Agents users must not receive VGS dependencies automatically.
- This package is the explicit VGS portion. Its normal package dependencies should include Kuzu and every other runtime library this VGS code imports.

## Test Commands

Use focused tests first, then broader verification.

```bash
python3 -m pytest tests/test_optional_vgs_dependency.py -q
python3 -m pytest tests/test_kuzu_integration.py tests/test_kuzu_search.py -q
python3 -m pytest -q
python3 -m ruff check .
```

For small changes, run the most relevant focused tests plus `ruff`. For changes touching storage, recall, dependencies, or backend behavior, run the full suite.

## Documentation Rules

- Keep README user-facing and concise.
- Keep DESIGN.md for architecture and rationale.
- Keep AGENTS.md for instructions that change coding-agent behavior.
- If a change alters the VGS dependency boundary, Kuzu runtime model, graph read path, or graph write policy, update the relevant docs and tests.

## Current Success Criteria

- Installing Deepagents does not automatically install the VGS system
- Installing or importing `deepagents-graph-memory` requires the VGS runtime dependencies.
- VGS uses Kuzu `:memory:` only; graph state resets when the Python process exits.
- No disk-backed graph storage or manual graph clear/reset API exists.
- Recall finds seed nodes, traverses connected context, and returns bounded markdown.
- Generated graph views remain read-only.
- The graph improves long-running, relationship-heavy work without becoming a broad memory framework.
