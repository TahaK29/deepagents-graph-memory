from deepagents.middleware.filesystem import EXECUTION_SYSTEM_PROMPT, FILESYSTEM_SYSTEM_PROMPT, FilesystemMiddleware
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

import deepagents_graph_memory.vgs as vgs
from deepagents_graph_memory import VFS_TOOL_NAMES, vgs_harness_profile


def test_vgs_harness_profile_excludes_default_vfs_tools():
    profile = vgs_harness_profile()

    assert profile.excluded_tools == VFS_TOOL_NAMES
    assert VFS_TOOL_NAMES == frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep"})


def test_vgs_excluded_tools_match_deepagents_filesystem_tools():
    filesystem_tool_names = {tool.name for tool in FilesystemMiddleware().tools}

    assert VFS_TOOL_NAMES < filesystem_tool_names
    assert "execute" in filesystem_tool_names
    assert "execute" not in VFS_TOOL_NAMES


def test_vgs_harness_profile_does_not_replace_base_prompt():
    profile = vgs_harness_profile()

    assert profile.base_system_prompt is None
    assert profile.system_prompt_suffix is None


def test_vgs_harness_profile_replaces_filesystem_prompt_only():
    profile = vgs_harness_profile()
    middleware = profile.materialize_extra_middleware()[0]
    captured = {}
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="hello")],
        system_message=SystemMessage(
            content_blocks=[
                {"type": "text", "text": "Base prompt"},
                {"type": "text", "text": f"\n\n{FILESYSTEM_SYSTEM_PROMPT}"},
                {"type": "text", "text": "\n\nCustom workflow tool guidance"},
            ],
        ),
        tools=[],
    )

    def handler(next_request):
        captured["system_prompt"] = next_request.system_message.text
        return ModelResponse(result=[])

    middleware.wrap_model_call(request, handler)

    system_prompt = captured["system_prompt"]
    assert "Base prompt" in system_prompt
    assert "Custom workflow tool guidance" in system_prompt
    assert "## Filesystem Tools" not in system_prompt
    assert "## Virtual Graph System (VGS)" in system_prompt


def test_vgs_harness_profile_preserves_execute_prompt_when_removing_filesystem_prompt():
    profile = vgs_harness_profile()
    middleware = profile.materialize_extra_middleware()[0]
    captured = {}
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="hello")],
        system_message=SystemMessage(
            content_blocks=[
                {
                    "type": "text",
                    "text": f"\n\n{FILESYSTEM_SYSTEM_PROMPT}\n\n{EXECUTION_SYSTEM_PROMPT}",
                },
            ],
        ),
        tools=[],
    )

    def handler(next_request):
        captured["system_prompt"] = next_request.system_message.text
        return ModelResponse(result=[])

    middleware.wrap_model_call(request, handler)

    system_prompt = captured["system_prompt"]
    assert "## Filesystem Tools" not in system_prompt
    assert "## Execute Tool `execute`" in system_prompt
    assert "## Virtual Graph System (VGS)" in system_prompt


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
