"""VGS harness-profile helpers for Deep Agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from deepagents import HarnessProfile, register_harness_profile
from deepagents.middleware.filesystem import EXECUTION_SYSTEM_PROMPT, FILESYSTEM_SYSTEM_PROMPT
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ContentBlock, SystemMessage

VFS_TOOL_NAMES = frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep"})
"""Deep Agents default virtual-filesystem tool names."""

VGS_SYSTEM_PROMPT_SUFFIX = """## Virtual Graph System (VGS)

Graph context is enabled. The graph is the source of truth for structured workflow context:
situations, rationales, actions, outcomes, artifacts, failures, evidence, decisions,
dependencies, and provenance.

In VGS mode, do not assume the default Deep Agents filesystem tools are available.

## Reading Graph Context

- Use `recall_graph_memory` as the primary read path for graph context.
- Ask targeted questions. Include anchors such as graph source paths, file paths, run ids,
  task ids, artifact ids, or subagent ids when you have them.
- Start with the default `auto` mode and modest budgets. Use `local` for one known entity
  and `deep` only when the task clearly needs multi-hop context.
- Stop querying when recall returns enough connected context to act, returns no seeds,
  repeats the same facts, reports that the next hop was not relevant, or reaches a node,
  edge, depth, or token budget.
- Do not chase the graph to completeness. Use recalled facts to guide the next action,
  then verify current external state with the available primary tools when the answer
  depends on live files, tests, services, or command output.

## Writing Graph Context

- Use `record_graph_trace` for durable workflow events, especially meaningful observations,
  decisions, actions, failures, outcomes, artifacts, and evidence.
- Do not write every thought. Prefer facts that will help resume work, avoid repeated
  failed attempts, explain a decision, or connect evidence to an outcome.
- Record failures and dead ends with their outcomes so future work can avoid repeating them.
- Do not store ordinary user preferences, profile facts, or unrelated notes in the graph.
- Generated `/graph/...` markdown paths are read-only views over graph data, not storage locations to edit.
- If low-level graph write tools are exposed, use them only with clear labels, relationship
  names, scope/provenance metadata, and schema discipline. Do not create arbitrary node or
  edge types just because a fact could be represented.
"""
"""Default VGS system prompt."""


class _VGSSystemPromptMiddleware(AgentMiddleware[Any, Any, Any]):
    """Replace Deep Agents filesystem guidance with VGS guidance."""

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage:
        """Apply VGS prompt guidance before the model call."""
        return handler(
            request.override(system_message=_apply_vgs_system_text(request.system_message, self.system_prompt)),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        """Async variant of `wrap_model_call`."""
        return await handler(
            request.override(system_message=_apply_vgs_system_text(request.system_message, self.system_prompt)),
        )


def vgs_harness_profile(*, system_prompt_suffix: str | None = VGS_SYSTEM_PROMPT_SUFFIX) -> HarnessProfile:
    """Create a Deep Agents harness profile for VGS mode.

    Args:
        system_prompt_suffix: Optional VGS prompt text. It is appended through
            late middleware after removing Deep Agents filesystem guidance.

    Returns:
        Harness profile that excludes Deep Agents filesystem tools.
    """
    extra_middleware = (
        ()
        if system_prompt_suffix is None
        else (_VGSSystemPromptMiddleware(system_prompt_suffix),)
    )
    return HarnessProfile(excluded_tools=VFS_TOOL_NAMES, extra_middleware=extra_middleware)


def register_vgs_harness_profile(model: str, *, system_prompt_suffix: str | None = VGS_SYSTEM_PROMPT_SUFFIX) -> None:
    """Register VGS mode for a Deep Agents model key.

    Args:
        model: Model key passed to `create_deep_agent`.
        system_prompt_suffix: Optional VGS prompt text.
    """
    register_harness_profile(model, vgs_harness_profile(system_prompt_suffix=system_prompt_suffix))


def _apply_vgs_system_text(system_message: SystemMessage | None, text: str) -> SystemMessage:
    content_blocks = _remove_filesystem_guidance_blocks(
        list(system_message.content_blocks) if system_message else [],
    )
    if content_blocks:
        text = f"\n\n{text}"
    content_blocks.append({"type": "text", "text": text})
    return SystemMessage(content=cast("list[str | dict[str, str]]", content_blocks))


def _remove_filesystem_guidance_blocks(content_blocks: list[ContentBlock]) -> list[ContentBlock]:
    result: list[ContentBlock] = []
    for block in content_blocks:
        block_text = _text_block_text(block)
        if block_text is None:
            result.append(block)
            continue

        updated_text = _remove_filesystem_guidance_text(block_text)
        if not updated_text:
            continue
        if updated_text == block_text:
            result.append(block)
            continue
        result.append(cast("ContentBlock", {**block, "text": updated_text}))
    return result


def _text_block_text(block: ContentBlock) -> str | None:
    if not isinstance(block, dict) or block.get("type") != "text":
        return None
    text = block.get("text")
    return text if isinstance(text, str) else None


def _remove_filesystem_guidance_text(text: str) -> str:
    leading_whitespace = text[: len(text) - len(text.lstrip())]
    stripped = text.strip()
    if not _is_deepagents_filesystem_guidance(stripped):
        return text

    execute_index = stripped.find("## Execute Tool `execute`")
    if execute_index == -1:
        return ""
    return leading_whitespace + stripped[execute_index:]


def _is_deepagents_filesystem_guidance(text: str) -> bool:
    if text == FILESYSTEM_SYSTEM_PROMPT:
        return True
    if text == f"{FILESYSTEM_SYSTEM_PROMPT}\n\n{EXECUTION_SYSTEM_PROMPT}":
        return True

    if not text.startswith("## Following Conventions"):
        return False
    if "## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`" not in text:
        return False
    if "## Large Tool Results" not in text:
        return False
    if "Offloaded tool results are stored under " not in text:
        return False

    execute_index = text.find("## Execute Tool `execute`")
    if execute_index == -1:
        return True
    return text[execute_index:].strip() == EXECUTION_SYSTEM_PROMPT
