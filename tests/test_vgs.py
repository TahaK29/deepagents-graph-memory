# - Checks the Deep Agents setup for working with graph context.
# - Cases: excluding the usual file tools, keeping application instructions;
#        turning graph guidance off and registering the setup for a model.

import asyncio

import pytest
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

import deepagents_graph_memory.vgs as vgs
from deepagents_graph_memory import VFS_TOOL_NAMES, graph_context_middleware, vgs_harness_profile

APP_FILESYSTEM_PROMPT = "Use filesystem tools for project files."
APP_EXECUTION_PROMPT = "Run project commands with execute."


def test_vgs_harness_profile_excludes_default_vfs_tools():
    profile = vgs_harness_profile()

    assert profile.excluded_tools == VFS_TOOL_NAMES
    assert VFS_TOOL_NAMES == frozenset({"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep"})


def test_vgs_excluded_tools_match_deepagents_filesystem_tools():
    filesystem_tool_names = {tool.name for tool in FilesystemMiddleware().tools}

    assert filesystem_tool_names - {"execute"} <= VFS_TOOL_NAMES
    assert "execute" in filesystem_tool_names
    assert "execute" not in VFS_TOOL_NAMES


def test_vgs_harness_profile_does_not_replace_base_prompt():
    profile = vgs_harness_profile()

    assert profile.base_system_prompt is None
    assert profile.system_prompt_suffix is None


def test_vgs_adds_guidance_after_real_filesystem_middleware():
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

    # Match the real middleware order without depending on upstream prompt text.
    filesystem_middleware.wrap_model_call(request, vgs_handler)

    system_prompt = captured["system_prompt"]
    assert "Base prompt" in system_prompt
    assert "Custom workflow tool guidance" in system_prompt
    assert "## Filesystem Tools" not in system_prompt
    assert "## Following Conventions" not in system_prompt
    assert "## Virtual Graph System (VGS)" in system_prompt


def test_vgs_preserves_custom_filesystem_and_execute_instructions():
    filesystem_middleware = FilesystemMiddleware(system_prompt=f"{APP_FILESYSTEM_PROMPT}\n\n{APP_EXECUTION_PROMPT}")
    vgs_middleware = vgs_harness_profile().materialize_extra_middleware()[0]
    captured = {}
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="hello")],
        system_message=SystemMessage(content="Base prompt"),
        tools=[],
    )

    def model_handler(next_request):
        captured["system_prompt"] = next_request.system_message.text
        return ModelResponse(result=[])

    filesystem_middleware.wrap_model_call(request, lambda request: vgs_middleware.wrap_model_call(request, model_handler))

    system_prompt = captured["system_prompt"]
    assert "Base prompt" in system_prompt
    assert APP_FILESYSTEM_PROMPT in system_prompt
    assert APP_EXECUTION_PROMPT in system_prompt
    assert "## Virtual Graph System (VGS)" in system_prompt


@pytest.mark.parametrize("prefix", ["/large_tool_results", "/artifacts/large_tool_results"])
@pytest.mark.parametrize("position", ["before_execute", "after_execute", "no_addition"])
def test_vgs_preserves_additions_to_legacy_filesystem_prompt(monkeypatch, prefix, position):
    # Current Deep Agents no longer supplies these constants; older CI versions
    # exercise their actual template through the same public middleware calls.
    template = getattr(
        vgs.filesystem,
        "_FILESYSTEM_SYSTEM_PROMPT_TEMPLATE",
        "## Following Conventions\n"
        "## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`\n"
        "## Large Tool Results\n"
        "Read `{large_tool_results_prefix}/`. Offloaded tool results are stored under `{large_tool_results_prefix}/<tool_call_id>`.",
    )
    monkeypatch.setattr(vgs.filesystem, "_FILESYSTEM_SYSTEM_PROMPT_TEMPLATE", template, raising=False)
    monkeypatch.setattr(vgs.filesystem, "FILESYSTEM_SYSTEM_PROMPT", template.format(large_tool_results_prefix="/large_tool_results"), raising=False)
    execution = getattr(vgs.filesystem, "EXECUTION_SYSTEM_PROMPT", "## Execute Tool `execute`\nRun commands only when authorized.")
    monkeypatch.setattr(vgs.filesystem, "EXECUTION_SYSTEM_PROMPT", execution, raising=False)
    application = "APPLICATION RULE: Require operator approval before deployment."
    parts = [template.format(large_tool_results_prefix=prefix), execution]
    if position != "no_addition":
        parts.insert(1 if position == "before_execute" else 2, application)
    filesystem_middleware = FilesystemMiddleware(system_prompt="\n\n".join(parts))
    graph_middleware = vgs_harness_profile().materialize_extra_middleware()[0]
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]), messages=[HumanMessage(content="hello")], system_message=SystemMessage(content="Base"), tools=[]
    )

    def model_handler(updated):
        prompt = updated.system_message.text
        if position != "no_addition":
            assert application in prompt
        assert execution in prompt
        assert "## Filesystem Tools" not in prompt
        assert "Base" in prompt and "Virtual Graph System" in prompt
        return ModelResponse(result=[])

    filesystem_middleware.wrap_model_call(request, lambda updated: graph_middleware.wrap_model_call(updated, model_handler))


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


def test_combined_middleware_preserves_filesystem_guidance_and_message_metadata():
    system_message = SystemMessage(
        content_blocks=[
            {"type": "text", "text": "Base prompt"},
            {"type": "text", "text": f"{APP_FILESYSTEM_PROMPT}\n\n{APP_EXECUTION_PROMPT}"},
            {"type": "text", "text": "Custom instructions"},
        ],
        id="system-1",
        name="project",
        additional_kwargs={"scope": "test"},
    )
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]), messages=[HumanMessage(content="hello")], system_message=system_message, tools=[]
    )
    captured = {}

    def model_handler(next_request):
        captured["message"] = next_request.system_message
        return ModelResponse(result=[])

    graph_context_middleware().wrap_model_call(request, model_handler)
    result = captured["message"]
    assert result.id == "system-1" and result.name == "project"
    assert result.additional_kwargs == {"scope": "test"}
    assert result.content_blocks[:3] == system_message.content_blocks
    assert APP_FILESYSTEM_PROMPT in result.text
    assert APP_EXECUTION_PROMPT in result.text
    assert "Custom instructions" in result.text
    assert "normal memory" in result.text.lower()
    assert "do not assume the default Deep Agents filesystem tools are available" not in result.text


def test_combined_middleware_preserves_content_blocks_and_metadata_async():
    system_message = SystemMessage(
        content=[
            {"type": "text", "text": "Base"},
            {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": APP_FILESYSTEM_PROMPT},
            {"type": "text", "text": "Custom"},
        ],
        id="system-async",
        additional_kwargs={"source": "application"},
    )
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]), messages=[HumanMessage(content="hello")], system_message=system_message, tools=[]
    )

    async def run():
        async def model_handler(next_request):
            return ModelResponse(result=[next_request.system_message])

        return await graph_context_middleware().awrap_model_call(request, model_handler)

    response = asyncio.run(run())
    result = response.result[0]
    assert result.content[:4] == system_message.content
    assert result.id == "system-async" and result.additional_kwargs == {"source": "application"}
    assert "Virtual Graph System" in result.text


def test_combined_middleware_does_not_register_a_model_profile(monkeypatch):
    def forbidden_registration(*args, **kwargs):
        raise AssertionError("combined setup must stay agent-local")

    monkeypatch.setattr(vgs, "register_harness_profile", forbidden_registration)
    assert graph_context_middleware() is not None


def test_task_guidance_preserves_request_and_is_not_duplicated_on_retry():
    middleware = graph_context_middleware()
    call = {"name": "task", "args": {"description": "Check parser", "subagent_type": "general-purpose"}, "id": "task-1", "type": "tool_call"}
    request = ToolCallRequest(tool_call=call, tool=None, state={}, runtime=None)
    received = []

    def handler(updated):
        received.append(updated)
        return ToolMessage(content="done", tool_call_id=updated.tool_call["id"])

    result = middleware.wrap_tool_call(request, handler)
    assert result.tool_call_id == "task-1"
    updated = received[-1]
    assert updated.tool_call["args"]["description"] == "Check parser\n\n" + vgs.GRAPH_CONTEXT_SYSTEM_PROMPT_SUFFIX
    assert call["args"] == {"description": "Check parser", "subagent_type": "general-purpose"}
    assert updated.tool_call["args"]["subagent_type"] == "general-purpose"

    async def async_handler(next_request):
        return handler(next_request)

    asyncio.run(middleware.awrap_tool_call(updated, async_handler))
    assert received[-1] is updated
    for name, args in (("record_graph_trace", {"description": "leave alone"}), ("task", {"description": 7}), ("task", {})):
        other = request.override(tool_call={**call, "name": name, "args": args})
        middleware.wrap_tool_call(other, handler)
        assert received[-1] is other
