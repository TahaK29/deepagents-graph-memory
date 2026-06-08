from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

import deepagents_graph_memory.vgs as vgs
from deepagents_graph_memory import VFS_TOOL_NAMES, vgs_harness_profile


def test_vgs_harness_profile_excludes_default_vfs_tools():
    profile = vgs_harness_profile()

    assert profile.excluded_tools == VFS_TOOL_NAMES
    assert VFS_TOOL_NAMES == frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep"})


def test_vgs_harness_profile_appends_graph_prompt_after_existing_prompt():
    profile = vgs_harness_profile()
    middleware = profile.materialize_extra_middleware()[0]
    captured = {}
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="hello")],
        system_message=SystemMessage(content="Base prompt\n\n## Filesystem Tools"),
        tools=[],
    )

    def handler(next_request):
        captured["system_prompt"] = next_request.system_message.text
        return ModelResponse(result=[])

    middleware.wrap_model_call(request, handler)

    system_prompt = captured["system_prompt"]
    assert "## Virtual Graph System (VGS)" in system_prompt
    assert system_prompt.rfind("## Virtual Graph System (VGS)") > system_prompt.rfind("## Filesystem Tools")


def test_vgs_harness_profile_can_disable_graph_prompt():
    profile = vgs_harness_profile(system_prompt_suffix=None)

    assert profile.excluded_tools == VFS_TOOL_NAMES
    assert profile.materialize_extra_middleware() == []


def test_register_vgs_harness_profile_registers_model(monkeypatch):
    calls = []

    def fake_register_harness_profile(model, profile):
        calls.append((model, profile))

    monkeypatch.setattr(vgs, "register_harness_profile", fake_register_harness_profile)

    vgs.register_vgs_harness_profile("test-model")

    assert calls[0][0] == "test-model"
    assert calls[0][1].excluded_tools == VFS_TOOL_NAMES
