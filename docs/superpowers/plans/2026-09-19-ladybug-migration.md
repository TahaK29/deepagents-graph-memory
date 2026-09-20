# LadybugDB Migration Implementation Plan

> **For agentic workers:** Use the assigned parallel task below. The parent owns integration, compatibility decisions, review, and the final commit.

**Goal:** Replace archived Kuzu with LadybugDB while preserving the public graph backend, tools, graph semantics, and persistence behavior.

**Architecture:** Keep the existing store boundary and Cypher implementation. Rename the engine module and classes, adapt only verified incompatibilities, and use one required runtime engine. Existing Kuzu files must not be silently corrupted or treated as empty databases.

**Tech Stack:** Python 3.11–3.14, Ladybug 0.20.4, existing Deep Agents dependencies, pytest, Ruff, GitHub Actions.

**Spec:** The approved conversation: migrate to LadybugDB with behavior parity, multiple Astra medium implementation threads, complete tests, and corrected code/documentation terminology.

## Global constraints

- Work on the existing main checkout. Do not create branches or worktrees.
- Keep `GraphMemoryBackend.create(path=None, ...)`, tools, namespaces, provenance, retry identity, history, recall budgets, and VGS/VFS composition stable.
- Preserve in-memory default and optional durable file path. Retain transaction rollback, shared-store synchronization, duplicate-open protection, and explicit close semantics.
- Use `ladybug>=0.20.4,<0.21` and `requires-python=">=3.11,<3.15"`; no Kuzu runtime fallback or new database abstraction.
- Engine names become `ladybug_store.py`, `LadybugGraphStore`, `_LadybugGraph`, and `ladybug`. The package remains `deepagents-graph-memory`.
- Verify full-text search with no network access at runtime. Do not weaken tests or silently fall back to another search method.
- Run actual Windows/Linux/macOS CI before claiming those platforms passed.
- Parent reviews and commits; implementation threads do not commit or push.

## Task 1 — Runtime adapter

**Owner:** Astra runtime thread. **Files:** `src/deepagents_graph_memory/**`, `pyproject.toml`.

- [x] Rename the current adapter and all current source terminology to LadybugDB; replace the required engine dependency.
- [x] Preserve all query, schema, transaction, lock, scope, search, and lifecycle behavior; change only engine incompatibilities reproduced by tests.
- [x] Check FTS availability and execute real full-text search after insert/update/reopen. Ladybug 0.20.4 does not bundle FTS: provision the native extension during setup, then load it locally at runtime.
- [x] Coordinate reproducible failures with the test thread and report exact behavior differences.

## Task 2 — Tests and platform verification

**Owner:** Astra tests thread. **Files:** `tests/**`, `.github/workflows/tests.yml`.

- [x] Rename engine tests and imports without removing assertions or reducing coverage.
- [x] Add a fresh-process check that imports the package and searches with Kuzu unavailable and runtime networking blocked.
- [x] Preserve crash recovery, transaction rollback, shared-agent collisions, namespace separation, file-view reads, temporal history, evidence, and async behavior tests.
- [x] Handle Windows symlink privilege differences without skipping ordinary path or hardlink protections.
- [x] Add CI jobs for Ubuntu, Windows, and macOS over Python 3.11–3.14; install the library with test extras and run the full suite and Ruff.

## Task 3 — Documentation and examples

**Owner:** Astra docs thread. **Files:** `README.md`, `DESIGN.md`, `AGENTS.md`, `evals/**`.

- [x] Replace current engine names, imports, diagrams, examples, and filenames; keep historical plans historically accurate.
- [x] Preserve the library name and external APIs. Describe LadybugDB's current OS/Python wheel requirements without claiming unrun tests passed.
- [x] Document old-file compatibility based on the parent's probe result, and keep persistence/cloud/VFS claims accurate.
- [x] Run example/evaluation checks where possible and report unresolved factual claims.

## Parent review and acceptance

- [x] Preserve the old source and generate a Kuzu baseline file and semantic snapshot before migration.
- [x] Compare old and new behavior, including stored nodes/edges/provenance and recall evaluations.
- [x] Test opening a disposable copy of the old database. If unsupported, provide explicit non-destructive conversion or clear version rejection; never mutate the only copy.
- [x] Review all diffs and remaining Kuzu references; retain references only where needed to explain historical compatibility.
- [x] Run full pytest, Ruff, format check, and package build/install checks locally.
- [ ] Run and inspect real platform CI.
- [ ] Commit on main with Taha's co-author trailer; push and verify the remote and CI results.

## Verified compatibility and local results

- Native result metadata uses uppercase keys. The adapter preserves node labels, relationship types, and JSON properties while excluding native metadata.
- Ladybug 0.20.4's implicit prepared-statement cache crashed on repeated parameterized writes. A fresh public `ladybug.PreparedStatement` per parameterized query avoids that cache; a subprocess regression covers repeated writes and reads.
- FTS requires explicit one-time provisioning for the target engine version, OS, architecture, and runtime user. The runtime only loads the installed extension. Linux CI additionally blocks native networking with a network namespace.
- Direct opening of a Kuzu 0.11.3 file fails without changing its hash. Native read-only export and import into a fresh Ladybug file preserved all 32 nodes, 59 edges, properties, timestamps, provenance, and retry identity in the baseline fixture; recall output was identical. A dedicated CI test exercises this conversion with Kuzu installed only in that job.
- macOS arm64 / Python 3.11: **186 tests passed**, Ruff and formatting checks passed, and the wheel and source distribution built successfully.
- Fresh wheel installation with Ladybug only, Deep Agents 0.6.12, and LangChain Core 1.6.3: **185 passed, 1 skipped** (the separate legacy conversion test). Ruff passed.
- All **12 offline workflow scenarios passed**. README shared-agent, persistence, and inspection examples executed successfully; Python documentation snippets parsed.
- Windows/Linux/macOS matrix results remain pending until the workflow runs on GitHub.
