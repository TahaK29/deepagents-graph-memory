from deepagents.middleware._utils import append_to_system_message
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


def test_vgs_strips_prompt_from_real_filesystem_middleware():
    # Drive the real Deep Agents FilesystemMiddleware so this test exercises the
    # actual filesystem prompt text and content-block layout it injects, rather
    # than a copy of the prompt constants. If Deep Agents changes how it builds
    # or splits that prompt, VGS would silently stop stripping it; a
    # self-referential test cannot catch that, but this one fails loudly.
    filesystem_middleware = FilesystemMiddleware()
    vgs_middleware = vgs_harness_profile().materialize_extra_middleware()[0]
    captured = {}
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="hello")],
        system_message=SystemMessage(
            content_blocks=[
                {"type": "text", "text": "Base prompt"},
                {"type": "text", "text": "\n\nCustom workflow tool guidance"},
            ],
        ),
        tools=[],
    )

    def model_handler(next_request):
        captured["system_prompt"] = next_request.system_message.text
        return ModelResponse(result=[])

    def vgs_handler(next_request):
        return vgs_middleware.wrap_model_call(next_request, model_handler)

    # FilesystemMiddleware is the outer middleware (injects the prompt); VGS runs
    # inside it (strips the prompt), mirroring the real Deep Agents stack order.
    filesystem_middleware.wrap_model_call(request, vgs_handler)

    system_prompt = captured["system_prompt"]
    assert "Base prompt" in system_prompt
    assert "Custom workflow tool guidance" in system_prompt
    assert "## Filesystem Tools" not in system_prompt
    assert "## Following Conventions" not in system_prompt
    assert "## Virtual Graph System (VGS)" in system_prompt


def test_vgs_preserves_execute_prompt_with_real_prompt_assembly():
    # The execute prompt is only injected when a backend supports execution,
    # which needs a full sandbox backend to drive end-to-end. Instead, build the
    # combined filesystem+execute prompt exactly the way FilesystemMiddleware
    # does (joined with blank lines, then appended via Deep Agents' own
    # append_to_system_message helper), so the input still matches production
    # block structure rather than being hand-rolled.
    combined_prompt = "\n\n".join([FILESYSTEM_SYSTEM_PROMPT, EXECUTION_SYSTEM_PROMPT]).strip()
    system_message = append_to_system_message(
        SystemMessage(content_blocks=[{"type": "text", "text": "Base prompt"}]),
        combined_prompt,
    )
    vgs_middleware = vgs_harness_profile().materialize_extra_middleware()[0]
    captured = {}
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="hello")],
        system_message=system_message,
        tools=[],
    )

    def model_handler(next_request):
        captured["system_prompt"] = next_request.system_message.text
        return ModelResponse(result=[])

    vgs_middleware.wrap_model_call(request, model_handler)

    system_prompt = captured["system_prompt"]
    assert "Base prompt" in system_prompt
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
