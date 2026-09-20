# Workflow evaluation

For source installs, complete the [development setup](../docs/guide.md#development)
first. Published wheels include the search extension. Run the twelve deterministic
integration cases without a model or provider credentials:

```bash
.venv/bin/python evals/run_workflows.py
.venv/bin/python evals/run_workflows.py --offline
.venv/bin/python -m pytest tests/test_workflow_evals.py
```

Each case in `scenarios.json` supplies one event list to both the public graph trace API and ordinary notes. Notes retain the same facts, source references, links, observation times, and saved arrival times. An exact operation retry yields one entry in both. Offline checks assert returned facts, warnings, and actual graph node counts. They report returned characters, latency, and truncation. Passing them establishes mechanical behavior only; it does not measure agent task success.

An optional live comparison uses the installed LangChain `create_agent` API, one explicit provider and model, and at most the selected cases and tool steps:

```bash
.venv/bin/python evals/run_workflows.py --model PROVIDER:MODEL --max-cases 3 --max-steps 8 --report /tmp/graph-workflows.json
```

Configure provider integration and credentials yourself. The runner neither installs a provider nor calls one in offline mode. Both live arms use the same model, question, allowed decision choices, instructions, fixed 8,000-character memory response cap, tool step cap, evidence checker, and action simulator. One memory tool calls graph recall; the other searches the shared event notes by subject, query, and observation time. The simulator performs no shell or network action. Grading expectations and flags are withheld from both agents; the decision choices are shared, and factual source references remain available to both.

The CLI accepts 1–12 cases and 1–20 tool steps per arm. No-argument execution is offline.

The live report keeps both arms, including errors and completed tool traces. It grades the structured final `decision`, `status`, and `source_ids` against fixture expectations, exact source IDs visible in tool results, missed contenders, required disputed-source reference lookups, and simulated unsafe actions. `check_evidence` returns the fixture's captured reference metadata; it does not fetch a log or run a new check. The cap limits successful tool executions; denied attempts can still appear in `tool_calls` before agent recursion stops. Tool calls, returned characters, latency, and provider usage metadata are reported. Missing usage stays `null`; character counts are not billed tokens. These exact-match grades are deliberately narrow: a cautious free-form explanation can be valuable but fail the structured grade. Run repeated trials with a saved report and publish mixed or negative results before making any graph-superiority claim. No live trial was run as part of the default tests.
