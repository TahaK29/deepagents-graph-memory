"""Offline workflow evaluation and bounded agent orchestration."""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from evals.run_workflows import grade_answer, load_scenarios, prepare_case, run_live_case, run_offline


class ToolCallingFake(BaseChatModel):
    mode: str = "normal"

    @property
    def _llm_type(self):
        return "workflow-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        tool_count = sum(message.type == "tool" for message in messages)
        if self.mode == "loop" or tool_count == 0 or (self.mode == "check_action" and tool_count < 3):
            name, args = (
                ("check_evidence", {"source_id": "pytest-history-15"})
                if tool_count == 1 and self.mode == "check_action"
                else ("simulate_action", {"action": "deploy_parser"})
                if tool_count == 2 and self.mode == "check_action"
                else ("retrieve_context", {"query": "parser check"})
            )
            message = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"step-{tool_count}"}])
            return ChatResult(generations=[ChatGeneration(message=message)])
        content = "oops" if self.mode == "malformed" else json.dumps({"decision": "abstain", "status": "abstain", "source_ids": []})
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def test_all_twelve_offline_scenarios_use_shared_records():
    cases = load_scenarios()
    assert len(cases) == 12
    results = run_offline(cases)
    assert all(result["passed"] for result in results), results
    for case in cases:
        backend, notes = prepare_case(case)
        assert len(notes) == len({record.get("operation_id") or record["trace_id"] for record in case["records"]})
        for record in case["records"]:
            assert record["outcome"] in json.dumps(notes)
            if "evidence_refs" in record:
                assert record["evidence_refs"][0]["source_id"] in json.dumps(notes)
        assert backend is not None
        assert all("recorded_at" in note for note in notes)


def test_both_live_arms_use_real_create_agent_with_fake_model():
    case = load_scenarios()[0]
    results = run_live_case(case, ToolCallingFake(), max_steps=3)
    assert set(results) == {"graph", "notes"}
    assert all(result["tool_calls"] == 1 for result in results.values())
    assert all(result["usage"] is None for result in results.values())
    assert all(result["returned_chars"] > 0 for result in results.values())
    assert all(result["grade"]["correct_decision"] for result in results.values())
    assert all(result["error"] is None for result in results.values())


def test_live_failures_keep_retrieved_context_and_invalid_answer_visible():
    case = load_scenarios()[0]
    malformed = run_live_case(case, ToolCallingFake(mode="malformed"), max_steps=3)
    assert all(result["grade"]["invalid_answer"] for result in malformed.values())
    capped = run_live_case(case, ToolCallingFake(mode="loop"), max_steps=1)
    assert all(result["returned_chars"] > 0 for result in capped.values())
    assert all(result["tool_calls"] > 1 for result in capped.values())
    assert all(result["error"] is not None for result in capped.values())
    assert all(result["tool_trace"][0]["tool"] == "retrieve_context" for result in capped.values())
    assert all(any(step["result"] == "Tool step cap reached." for step in result["tool_trace"]) for result in capped.values())


def test_live_check_and_simulated_action_are_graded_from_execution():
    case = load_scenarios()[0]
    results = run_live_case(case, ToolCallingFake(mode="check_action"), max_steps=3)
    for result in results.values():
        assert result["tool_calls"] == 3
        assert {call.get("source_id") for call in result["simulator_calls"]} >= {"pytest-history-15"}
        assert any(step["tool"] == "check_evidence" and "pytest-history-15" in step["result"] for step in result["tool_trace"])
        assert result["grade"]["acted_on_unresolved"]
        assert not result["grade"]["unsupported_claim"]


def test_grader_marks_unsupported_claim_and_repeated_action():
    case = next(case for case in load_scenarios() if case["id"] == "same-operation-retry")
    answer = {"decision": "deploy", "status": "resolved", "source_ids": ["invented-source"]}
    grade = grade_answer(
        case,
        answer,
        retrieved='{"source_id":"invented-source-extra"}',
        simulator_calls=[{"tool": "simulate_action", "action": "rerun_parser_check"}],
    )
    assert not grade["correct_decision"]
    assert grade["unsupported_claim"]
    assert grade["unnecessary_repeated_action"]
    assert grade_answer(case, ["not an object"], retrieved="", simulator_calls=[])["invalid_answer"]
