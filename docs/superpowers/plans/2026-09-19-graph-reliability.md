# Graph reliability implementation plan

> For agentic workers: use superpowers:subagent-driven-development or superpowers:executing-plans for each task. The user requested six separate Sol medium threads. The parent owns design, review, verification, and integration.

**Goal:** Make bounded recall useful on long-running tasks, make retries safe, and preserve the evidence needed to revisit decisions.

**Architecture:** Extend the existing in-memory LadybugDB store, trace tool, recall path, and VGS prompt. Keep model judgment in the agent. Store explicit relationships and derive review status when reading instead of running background jobs or maintaining a second truth database.

**Tech stack:** Python 3.11+, LadybugDB, existing Deep Agents/LangChain/LangGraph dependencies, pytest, Ruff.

**Spec:** DESIGN.md and the approved discussion about the six gaps. The requirements and acceptance checks below define the implementation scope.

## Global constraints

- Start from main at a60a8c8. No branches or worktrees. Each implementation thread leaves changes uncommitted; parent reviews and commits to main. Do not push.
- Run from the actual package root, using .venv/bin/python.
- One source-code writer at a time: these tasks share backend.py, tools.py, recall.py, and documentation.
- No disk persistence, lifecycle/reset APIs, new agent-facing tools, model calls in storage, automatic arbitration, or new dependencies.
- Preserve ordinary trace calls, namespace isolation, transactions, original findings, timestamps, and evidence.
- Observation time informs ordering; report arrival is never a truth rule. Different current contenders must remain visible.
- An omitted finding, truncated dependency chain, or missing timestamp is not proof of agreement or correctness.
- No fuzzy merging of entity names, no text-based deduplication of separate executions, no voting by number of agents.
- Public changes must be documented with runnable examples and honest limits. Each task includes focused regressions and Ruff; run the full suite at its review checkpoint.

## Execution order and review checkpoints

1. Select relevant findings before consuming retrieval budgets.
2. Make repeated execution of one operation idempotent.
3. Give parent and workers a shared subject identity.
4. Expose conclusions whose supporting findings changed.
5. Distinguish actual evidence sources from repeated reports.
6. Evaluate realistic workflows and provide an optional live-model comparison.

Tasks 2 and 5 share payload fingerprinting; task 5 must extend the fingerprint. Tasks 4 and 5 extend the focused retrieval from task 1. Task 6 exercises all five rather than introducing another production subsystem.

## Task 1: Bounded recall that reaches the latest relevant findings

**Files:** stores.py, ladybug_store.py, recall.py, tests/test_subject_findings.py, README.md.

**Problem reproduced:** A 16-observation state history anchored at its first failure omits the newest passing observation with max_nodes=6. An incomplete warning appears, but the answer is not useful. Display sorting occurs after the candidate budget has already excluded the update.

**Implementation:**

- Add one internal store method for selecting same-subject trace IDs, returning the existing LimitedResult shape. Suggested signature:

      def list_subject_trace_ids(self, subject_id: str, *, scope_key: str | None = None,
                                 limit: int = 50) -> LimitedResult:

- Read only the requested subject's traces and their update/resolution relationships under the shared store lock. Rank before applying the output limit. Reuse validation rules for SUPERSEDES and RESOLVES so rendering and selection cannot disagree.
- Prefer terminal findings/resolutions over their explicitly superseded/reviewed predecessors. Within that set, use known observation times descending, missing times last, and ID as deterministic tie-breaker. Do not pick one branch as truth merely because it is newer.
- The first version may read and sort one subject's history in Python. Document that per-subject scan with a ponytail comment; do not scan the whole project or invent indexes before measurement.
- Replace the arbitrary-neighbor selection in subject expansion. Reserve budget for related Trace summaries and their links before boilerplate components. Include the user's original anchor when space permits, with a direct history reference when it does not.
- Pack the finding outcome, source/evidence, time, and review/update links together. Put explicit incompleteness before content when competitors, history, or evidence are omitted. Preserve the warning even at token_budget=1.
- Do not exceed node/edge output limits. If all live contenders cannot fit, show a partial result explicitly; never imply a unique current answer.

**Regression sketch:**

    backend = GraphMemoryBackend.create()
    for i in range(16):
        backend.record_graph_trace(
            trace_id=f"step-{i:02}", situation="parser check", rationale="test output",
            action="ran test", outcome="current-passed" if i == 15 else "previous-failed",
            subject="parser-empty@linux", finding_type="state",
            observed_at=f"2026-09-19T10:{i:02}:00Z", evidence=[f"run-{i}"],
            supersedes=[f"step-{i-1:02}"] if i else None,
        )
    text = backend.recall_graph_memory("previous-failed",
        anchors=["/graph/nodes/Trace/step-00.md"], max_nodes=6, max_edges=20)
    assert "current-passed" in text

**Additional cases:** two live branches, unknown observation time, old/component/subject/task anchors, namespace separation, large shared subjects, tiny token/edge/node limits, resolution chains, deterministic ordering.

- [x] Reproduce the omission before changing selection.
- [x] Implement store selection and recall integration without a second recall engine.
- [x] Run focused tests, full pytest, Ruff, and the README recall example.
- [x] Parent reviews candidate ordering, all budget notices, and query scope, then commits.

## Task 2: Retry safety by operation identity

**Files:** backend.py, tools.py, stores.py if shared normalization is needed, tests/test_shared_store.py, tests/test_tools.py, README.md.

**Implementation:**

- Add optional operation_id to record_graph_trace. A caller reuses it only for retrying the exact same operation. Its identity is scoped by namespace. A fresh operation, even with identical text, stays a separate observation.
- Derive a deterministic trace ID from the operation ID using the existing full-digest pattern. Reject supplying both operation_id and explicit trace_id to keep identity unambiguous. Existing explicit trace_id duplicate behavior stays unchanged.
- Save a canonical request fingerprint with the trace. Normalize text, observation time, optional values, and order-insensitive reference lists before hashing. Include all caller-supplied semantics and metadata; exclude server-generated times and derived graph IDs. Reject reserved fingerprint fields supplied by callers.
- Inside the existing transaction: same operation + same fingerprint returns the original trace ID without modifying timestamps/nodes/edges; same operation + different payload raises a validation error. A failed transaction must leave no operation reservation.
- Obtain a runtime tool-call ID automatically for ordinary agent execution; explicitly supplied operation_id is for replay across different tool-call IDs. Use namespace + runtime thread ID when present + tool-call ID as a stable, unambiguous runtime identity. Never use execution attempt IDs, which change on retries.
- Installed-runtime probe confirmed that ToolRuntime with a default of None is hidden from the model schema, is injected by a compiled ToolNode graph, and preserves direct dictionary invocation. The union annotation ToolRuntime | None is NOT recognized by this installed version. Use the working supported annotation with a brief compatibility comment, test it, and do not add middleware for this.
- No ID available means normal append behavior. Do not pretend content equality proves a retry.

**Regression sketch:**

    payload = dict(situation="timeout", rationale="probe", action="checked network", outcome="failed")
    first = backend.record_graph_trace(operation_id="probe-7", **payload)
    original = backend.store.get_node("Trace", first)
    assert backend.record_graph_trace(operation_id="probe-7", **payload) == first
    assert backend.store.get_node("Trace", first) == original
    assert backend.record_graph_trace(operation_id="probe-8", **payload) != first

**Additional cases:** conflicting payload, concurrent identical retries, retry after rollback, timezone-normalized equivalent input, changed evidence, separate namespaces, sync/async compiled ToolNode execution, direct dictionary invocation, no runtime field in model JSON schema.

- [x] Add transaction and real-runtime regression tests.
- [x] Implement fingerprinting and optional runtime injection.
- [x] Verify existing callers and public tool schemas still work.
- [x] Parent reviews and commits after full verification.

## Task 3: Shared subject identity without fuzzy merging

**Files:** paths.py, __init__.py, tools.py, vgs.py, tests/test_paths.py, tests/test_tools.py, README.md.

**Implementation:**

- Export make_graph_subject(entity: str, aspect: str, environment: str) -> str. Validate nonempty strings and produce a deterministic compact JSON tuple using stripped inputs. Preserve case and path spelling; do not guess equivalences. Enforce the existing 512-character subject limit.
- A subject names a question about one entity/environment. Keep changing revisions in evidence and trace context so observations before and after a patch still belong to that same question. Use separate subjects for environments which should not be compared as one state.
- Extend graph_memory_tools with optional bound_subject. Parent/application creates the key once and supplies it to each worker's tools. If a call omits subject, use the binding; if it supplies a different subject, fail clearly. Unbound parent tools retain current behavior.
- Prompt/documentation instruct the parent to give workers the same key rather than inventing display names independently. Free-form descriptions remain in existing trace text.

**Regression sketch:**

    subject = make_graph_subject("tests/test_auth.py::test_login", "result", "linux")
    assert subject == make_graph_subject("tests/test_auth.py::test_login", "result", "linux")
    assert subject != make_graph_subject("tests/test_auth.py::test_login", "result", "windows")
    worker_tools = graph_memory_tools(backend, bound_subject=subject)

**Additional cases:** tuple-separator collisions, empty fields, overlong subject, case preserved, workers using different descriptions but the same binding, attempts to override binding, unbound tools backward compatibility.

- [x] Test canonical encoding and binding behavior.
- [x] Add helper, binding, and a parent/worker example using existing APIs.
- [x] Parent reviews and commits after focused/full verification.

## Task 4: Decisions whose supporting findings changed

**Files:** backend.py, tools.py, recall.py, ladybug_store.py/stores.py only for a focused scoped lookup if necessary, vgs.py, tests/test_subject_findings.py, README.md.

**Implementation:**

- Add optional depends_on: list[str] to the existing trace tool/backend. It names prior Trace IDs relied on by the new conclusion. Validate existing same-namespace targets, deduplicate IDs, and write Trace -BASED_ON-> Trace atomically. Include this field in task 2 fingerprinting.
- New traces can only reference existing traces, preventing cycles through the normal API. Bound traversal with a visited set anyway because low-level builder tools remain available.
- During recall, follow the dependency chain within budgets. If a supporting trace was superseded or reviewed in an explicit resolution, label the dependent conclusion 'needs recheck' and show the changed premise plus its update/resolution path.
- Propagate that notice through discovered downstream dependencies, including a transitive chain. Do not mutate every dependent node, automatically reverse decisions, or claim that changed evidence proves the conclusion false.
- Unknown/truncated dependency status must stay unknown, never silently clean. This warning belongs before the conclusion it qualifies.
- Agent rechecking produces a new trace citing the evidence it actually used. Original decisions and their historical dependency status remain inspectable; use existing supersedes/resolves links where their semantics apply, without adding a generic truth-clearing API.

**Regression sketch:**

    decision = backend.record_graph_trace(
        situation="database unavailable", rationale="availability probe failed",
        action="chose a temporary mitigation", outcome="disable checkout",
        subject="checkout::mitigation@staging", depends_on=[failed_probe],
    )
    # Record a newer evidenced state superseding failed_probe.
    text = backend.recall_graph_memory("checkout", anchors=[f"/graph/nodes/Trace/{decision}.md"])
    assert "needs recheck" in text

**Additional cases:** unchanged premise, transitive dependency, unrelated change, cross-scope target, duplicate dependency IDs, malformed references, missing/partial evidence, low-level cycle, rechecked conclusion based on current premise, atomic failure.

- [x] Test a stale premise changing a previously recorded decision's recall status.
- [x] Add explicit dependencies and bounded read-time review notices.
- [x] Parent confirms this flags review rather than making decisions, then verifies and commits.

## Task 5: Evidence source identity

**Files:** backend.py, tools.py, recall.py, renderers.py if needed, vgs.py, tests/test_trace.py, tests/test_subject_findings.py, README.md.

**Implementation:**

- Keep evidence: list[str] working. Add optional evidence_refs: list[dict[str, str]] to the same tool. Closed allowed keys: source_id and locator required; revision, observed_at, summary optional. Validate all keys/types, nonempty supplied values, timestamps, and sensible bounded lengths before writing. No file/network fetching in storage.
- source_id names an actual observation or captured source snapshot, not its paraphrase or reporting agent. locator is a log/file/document reference; revision distinguishes source versions where available. Applications should assign these IDs at evidence collection.
- Store immutable EvidenceSource identity using the existing scoped full-digest node pattern. Conflicting locator/revision/observation-time for an existing source ID is an error, not an overwrite. Put each reporter's optional summary on its citation link so paraphrases don't create fake independent sources.
- Link Trace -CITES-> EvidenceSource and retain normalized references in trace metadata for bounded recall. Multiple agents citing the same source ID yield one source with multiple citations. Different real executions keep distinct source IDs even when output text matches.
- New references satisfy the existing evidence-required check for explicit updates/resolutions. Include normalized references in operation fingerprinting.
- Recall exposes source IDs and locators. Count distinct cited sources only when coverage is complete; do not call them independent experiments or use the count as a confidence vote. Legacy evidence strings remain clearly unstructured caller reports.

**Example:**

    evidence_refs = [{
        "source_id": "pytest-run-17",
        "locator": "logs/pytest-run-17.txt",
        "revision": "commit-a1",
        "observed_at": "2026-09-19T10:00:00Z",
        "summary": "test_empty failed",
    }]

**Additional cases:** two agents paraphrasing one source, two executions with identical text, changed revision under reused source ID, incomplete references, unsupported keys, same source ID in separate namespaces, citation truncation, backward-compatible evidence strings, retry identity including references, atomic rollback.

- [x] Test source deduplication and conflicting identity metadata before implementation.
- [x] Add source references through the existing primitives and show citations in recall.
- [x] Parent checks no inferred independence/truth claims, then verifies and commits.

## Task 6: Workflow evaluation rather than a feature-count claim

**Files:** evals/scenarios.json, evals/run_workflows.py, evals/README.md, tests/test_workflow_evals.py, README.md.

**Scenario set:** 12 small, deterministic cases: latest update in long history; two live contenders; delayed worker report; same-operation retry; two genuinely separate repeated executions; consistent parent/worker subject identity; stale premise affecting a decision; transitive stale premise; one source quoted by multiple workers; resolution followed by new contrary evidence; context-window loss while the in-process graph survives; missing or truncated evidence requiring abstention.

**Implementation:**

- Scenario records contain the same facts and provenance for graph and notes representations, a question/decision, expected source IDs, forbidden unsupported conclusions, and expected review/abstention state. No answer hidden only in the graph fixture.
- Default runner is offline and deterministic: populate graph through public APIs, exercise retrieval, and report explicit assertions, returned size, latency, and truncation. These are integration results, NOT proof of improved LLM task success.
- Provide an opt-in live-model mode requiring an explicit provider:model argument and bounded case/step counts. No model calls in default tests, no API keys checked into artifacts, no provider installed automatically, no live run without configured credentials/model choice.
- Live comparison uses the same model, facts, task instructions, action simulator, tool-step cap, and context budget for both arms. One arm uses graph recall; the other retrieves the same information as ordinary notes. Use ordinary create_agent with installed tool APIs to isolate the memory comparison; retain real Deep Agents/ToolNode integration tests for package wiring. Do not globally register a VGS profile that accidentally changes both arms.
- Include a simple deterministic simulated action/check tool where appropriate, so graders can inspect whether an agent repeated a failed action, checked the disputed evidence, or acted on an unresolved conclusion. No real shell/network actions by the evaluation agent.
- Grade objective behavior and source support from the transcript. Report correct decision, unsupported claim, missed competing finding, unnecessary repeated action, tool calls, latency, and actual provider usage metadata when available. Missing token usage is unknown, not zero. Approximate character counts must not be labeled billed tokens.
- Repeated live trials and a saved report are required before claiming graph superiority. Publish mixed/negative results rather than tuning the baseline to lose.

**Commands:**

    .venv/bin/python evals/run_workflows.py --offline
    .venv/bin/python evals/run_workflows.py --model PROVIDER:MODEL --max-cases 3 --max-steps 8
    .venv/bin/python -m pytest tests/test_workflow_evals.py

- [x] Write scenario fixtures and offline assertions through the public APIs.
- [x] Implement the small CLI with separate offline and opt-in live paths.
- [x] Test live-runner orchestration with a fake model; execute all offline cases.
- [x] Document what was measured and that live superiority remains unmeasured until a configured run.
- [x] Parent reviews evaluation fairness, runs final full pytest/Ruff/examples, and commits.

## Final integration checks

- [x] Every task has its own implementation thread and reviewed commit on main.
- [x] No source changed outside the approved six areas; inspect final diff and git status.
- [x] Existing tool names remain unchanged and schemas serialize correctly.
- [x] Sync/async calls, namespace isolation, transactions, rollback, and all prior tests pass.
- [x] README examples and offline workflow scenarios run without provider credentials.
- [x] Final report separates verified fixes from caller-dependent semantics and unrun live evaluations.

## Review follow-up: remove duplicated recall output

The first Task 6 run found truncation in all twelve scenarios at the original
2,000-token / 8,000-character defaults, including a single retried observation.
Trace fields appeared in finding history, full nodes, generated components, chain
relationships, and repeated source paths. Raising the evaluation cap would hide
this product issue.

- Keep the evaluation cap unchanged.
- Render each trace's meaningful fields, provenance, times, and evidence once.
- Suppress only generated component data and links proven redundant with that trace.
- Preserve explicit component anchors, changed component data, custom metadata and relationships.
- Keep direct debug views complete and retain every genuine incompleteness warning.
- [x] Review and test the compact rendering follow-up in the original Task 1 thread.
- [x] Repeat the offline scenarios at the original cap and report remaining limitations.

## Verified outcome

Six separate Sol medium implementation threads completed the six tasks. The parent
reviewed their changes and integrated them directly on main, including the recall
compaction follow-up uncovered by the workflow scenarios.

- Full suite: 153 tests passed; Ruff checks passed.
- All twelve offline scenarios passed at the original 2,000-token / 8,000-character defaults.
- Ten scenarios returned complete context; the long-history and deliberately tiny-budget cases retained their warnings.
- Six standalone README examples and a combined real ToolNode runtime check passed.
- The optional live runner was exercised with a fake tool-capable model. No live provider comparison ran, so no model-quality advantage is claimed.
- Storage remains in memory; no branches, worktrees, persistence APIs, or dependencies were added.
