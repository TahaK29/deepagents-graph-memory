"""Small offline workflow checks and optional model comparison."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from deepagents_graph_memory import GraphMemoryBackend

SCENARIOS = Path(__file__).with_name("scenarios.json")
CONTEXT_CHARS = 8_000


def load_scenarios() -> list[dict[str, Any]]:
    """Load the shared graph and notes event records."""
    return json.loads(SCENARIOS.read_text())


def prepare_case(case: dict[str, Any]) -> tuple[GraphMemoryBackend, list[dict[str, Any]]]:
    """Feed the same records into the public graph API and plain notes."""
    backend = GraphMemoryBackend.create()
    notes: list[dict[str, Any]] = []
    operations: dict[str, dict[str, Any]] = {}
    for record in case["records"]:
        trace_id = backend.record_graph_trace(**record)
        note = {**record, "trace_id": trace_id}
        saved = backend.store.get_node("Trace", trace_id)
        assert saved is not None
        for field in ("recorded_at", "observed_at", "evidence_refs"):
            if field in saved.properties:
                note[field] = saved.properties[field]
        operation_id = record.get("operation_id")
        if operation_id is None:
            notes.append(note)
        elif operation_id not in operations:
            operations[operation_id] = note
            notes.append(note)
        elif operations[operation_id] != note:
            raise ValueError(f"Conflicting retry in {case['id']}")
    return backend, notes


def retrieve_notes(notes: list[dict[str, Any]], query: str, anchor: str | None, *, char_budget: int = CONTEXT_CHARS, max_notes: int = 30) -> str:
    """Retrieve relevant complete notes with metadata under a character cap."""
    terms = {term.casefold() for term in query.split() if len(term) > 2}
    anchored = next((note for note in notes if note["trace_id"] == anchor), None)
    subject = anchored.get("subject") if anchored else None

    def rank(note: dict[str, Any]) -> tuple[int, int, str, str]:
        searchable = json.dumps(note, ensure_ascii=False).casefold()
        return (
            int(note.get("subject") == subject and subject is not None),
            sum(term in searchable for term in terms),
            note.get("observed_at", ""),
            note["trace_id"],
        )

    ordered = sorted(notes, key=rank, reverse=True)
    lines: list[str] = []
    used = 0
    for note in ordered[:max_notes]:
        line = json.dumps(note, ensure_ascii=False, sort_keys=True)
        if used + len(line) + 1 > char_budget:
            continue
        lines.append(line)
        used += len(line) + 1
    if len(lines) < len(notes):
        lines.insert(0, "Notes incomplete; inspect more context before concluding.")
    return "\n".join(lines)[:char_budget]


def graph_context(backend: GraphMemoryBackend, case: dict[str, Any], *, live: bool = False) -> str:
    """Use bounded public recall for one scenario."""
    return backend.recall_graph_memory(
        case["question"],
        anchors=[f"/graph/nodes/Trace/{case['anchor']}.md"] if case.get("anchor") else None,
        token_budget=2_000 if live else case.get("token_budget", 2_000),
        max_nodes=case.get("max_nodes", 30),
        max_edges=case.get("max_edges", 60),
    )[:CONTEXT_CHARS]


def run_offline(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run deterministic integration checks without an agent or provider."""
    results = []
    for case in cases:
        start = perf_counter()
        backend, notes = prepare_case(case)
        context = graph_context(backend, case)
        expected = case["expect"]
        checks = {marker: marker in context for marker in expected["contains"]}
        unique_traces = expected.get("unique_traces")
        if unique_traces is not None:
            traces = backend.ls("/graph/nodes/Trace/").entries or []
            checks["unique_traces"] = len(traces) == len(notes) == unique_traces
        unique_sources = expected.get("unique_sources")
        if unique_sources is not None:
            sources = {ref["source_id"] for note in notes for ref in note.get("evidence_refs", [])}
            graph_sources = backend.ls("/graph/nodes/EvidenceSource/").entries or []
            checks["unique_sources"] = len(graph_sources) == len(sources) == unique_sources
        results.append(
            {
                "id": case["id"],
                "passed": all(checks.values()),
                "checks": checks,
                "returned_chars": len(context),
                "latency_ms": round((perf_counter() - start) * 1_000, 2),
                "truncated": any(marker in context for marker in ("Related findings incomplete", "Dependency status unknown", "Results truncated")),
            }
        )
    return results


def grade_answer(case: dict[str, Any], answer: Any, *, retrieved: str, simulator_calls: list[dict[str, str]]) -> dict[str, bool]:
    """Grade observable model output without giving expectations to the model."""
    expected = case["expect"]
    valid = isinstance(answer, dict) and isinstance(answer.get("decision"), str) and isinstance(answer.get("status"), str)
    valid = valid and isinstance(answer.get("source_ids"), list) and all(isinstance(item, str) for item in answer["source_ids"])
    answer = answer if isinstance(answer, dict) else {}
    cited = answer.get("source_ids")
    cited = cited if isinstance(cited, list) and all(isinstance(item, str) for item in cited) else []
    visible_sources = set(re.findall(r'"source_id"\s*:\s*"([^"\\]+)"', retrieved))
    checked = {call["source_id"] for call in simulator_calls if call.get("tool") == "check_evidence" and "source_id" in call}
    actions = {call["action"] for call in simulator_calls if call.get("tool") == "simulate_action" and "action" in call}
    simulator = case.get("simulator", {})
    return {
        "invalid_answer": not valid,
        "correct_decision": bool(valid and answer.get("decision") == expected["decision"] and answer.get("status") == expected["status"]),
        "unsupported_claim": any(source_id not in visible_sources for source_id in cited)
        or answer.get("decision") in expected.get("forbidden_decisions", []),
        "missed_competing_finding": bool(expected.get("competing") and not set(expected["source_ids"]).issubset(cited)),
        "missed_expected_source": bool(
            answer.get("status") != "abstain" and expected["source_ids"] and not set(expected["source_ids"]).issubset(cited)
        ),
        "missed_required_check": not set(expected.get("required_checks", [])).issubset(checked),
        "unnecessary_repeated_action": bool(actions.intersection(simulator.get("failed_actions", []))),
        "acted_on_unresolved": bool(actions.intersection(simulator.get("unsafe_actions", []))),
    }


def run_live_case(case: dict[str, Any], model: Any, *, max_steps: int = 8) -> dict[str, dict[str, Any]]:
    """Compare graph and notes with the same model, tools, and caps."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    backend, notes = prepare_case(case)
    sources = {ref["source_id"]: ref for note in notes for ref in note.get("evidence_refs", [])}

    def run_arm(arm: str) -> dict[str, Any]:
        calls: list[str] = []
        simulator_calls: list[dict[str, str]] = []
        tool_outputs: list[str] = []
        tool_trace: list[dict[str, str]] = []

        @tool
        def retrieve_context(query: str) -> str:
            """Retrieve recorded workflow context for the question."""
            calls.append("retrieve_context")
            if len(calls) > max_steps:
                tool_trace.append({"tool": "retrieve_context", "query": query, "result": "Tool step cap reached."})
                return "Tool step cap reached."
            result = (
                graph_context(backend, {**case, "question": query}, live=True)
                if arm == "graph"
                else retrieve_notes(notes, query, case.get("anchor"), max_notes=case.get("max_nodes", 30))
            )
            tool_outputs.append(result)
            tool_trace.append({"tool": "retrieve_context", "query": query, "result": result})
            return result

        @tool
        def check_evidence(source_id: str) -> str:
            """Inspect captured reference metadata for one source ID."""
            calls.append("check_evidence")
            if len(calls) > max_steps:
                tool_trace.append({"tool": "check_evidence", "source_id": source_id, "result": "Tool step cap reached."})
                return "Tool step cap reached."
            result = json.dumps(sources[source_id]) if source_id in sources else "Unknown source ID."
            if source_id in sources:
                simulator_calls.append({"tool": "check_evidence", "source_id": source_id})
            tool_outputs.append(result)
            tool_trace.append({"tool": "check_evidence", "source_id": source_id, "result": result})
            return result

        @tool
        def simulate_action(action: Literal["rerun_parser_check", "deploy_parser", "keep_checkout_disabled", "notify_operator"]) -> str:
            """Try a simulated action; no real shell or network work occurs."""
            calls.append("simulate_action")
            if len(calls) > max_steps:
                tool_trace.append({"tool": "simulate_action", "action": action, "result": "Tool step cap reached."})
                return "Tool step cap reached."
            outcome = case.get("simulator", {}).get("outcomes", {}).get(action)
            if outcome is None:
                tool_trace.append({"tool": "simulate_action", "action": action, "result": "Action unavailable in this scenario."})
                return "Action unavailable in this scenario."
            simulator_calls.append({"tool": "simulate_action", "action": action})
            result = f"Simulated {action}: {outcome}. No external action occurred."
            tool_trace.append({"tool": "simulate_action", "action": action, "result": result})
            return result

        prompt = (
            "Answer the workflow question using the recorded context. Retrieve context first. "
            "Inspect disputed source references when needed. Avoid repeating a failed action or acting on an unresolved conclusion. "
            'Finish with one JSON object only: {"decision": string, "status": string, "source_ids": [strings]}. '
            "Status must be resolved, needs recheck, or abstain. "
            "Use source IDs that you saw in tool results; abstain when context is incomplete."
        )
        agent = create_agent(model, [retrieve_context, check_evidence, simulate_action], system_prompt=prompt)
        start = perf_counter()
        error = None
        messages = []
        try:
            question = f"{case['question']} Choose decision from: {', '.join(case['choices'])}."
            messages = agent.invoke({"messages": [{"role": "user", "content": question}]}, config={"recursion_limit": max_steps * 2 + 4})["messages"]
        except Exception as exc:  # A failed arm remains visible in the comparison report.
            error = f"{type(exc).__name__}: {exc}"
        elapsed = round((perf_counter() - start) * 1_000, 2)
        retrieved = "\n".join(tool_outputs)
        answer = None
        try:
            answer = json.loads(messages[-1].content) if messages else None
        except (json.JSONDecodeError, TypeError):
            pass
        model_messages = [message for message in messages if message.type == "ai"]
        usage_items = [message.usage_metadata for message in model_messages]
        usage = (
            {
                key: sum(item[key] for item in usage_items) if all(item is not None and key in item for item in usage_items) else None
                for key in ("input_tokens", "output_tokens", "total_tokens")
            }
            if usage_items and all(item is not None for item in usage_items)
            else None
        )
        return {
            "answer": answer,
            "grade": grade_answer(case, answer, retrieved=retrieved, simulator_calls=simulator_calls),
            "tool_calls": len(calls),
            "simulator_calls": simulator_calls,
            "tool_trace": tool_trace,
            "returned_chars": len(retrieved),
            "latency_ms": elapsed,
            "usage": usage,
            "error": error,
        }

    return {arm: run_arm(arm) for arm in ("graph", "notes")}


def main() -> int:
    """Run offline checks or a bounded opt-in live model comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Run deterministic checks with no model")
    parser.add_argument("--model", help="Explicit provider:model for a live comparison")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.offline and args.model:
        parser.error("choose --offline or --model PROVIDER:MODEL")
    if not 1 <= args.max_cases <= 12 or not 1 <= args.max_steps <= 20:
        parser.error("--max-cases must be 1..12 and --max-steps must be 1..20")
    cases = load_scenarios()
    if not args.model:
        results = run_offline(cases)
        kind = "offline integration checks; no model quality claim"
    else:
        if ":" not in args.model:
            parser.error("--model must be PROVIDER:MODEL")
        from langchain.chat_models import init_chat_model

        model = init_chat_model(args.model)
        results = {case["id"]: run_live_case(case, model, max_steps=args.max_steps) for case in cases[: args.max_cases]}
        kind = "single live trial; no superiority claim"
    report = {"kind": kind, "created_at": datetime.now().astimezone().isoformat(), "results": results}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n")
    return 0 if args.model or all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
