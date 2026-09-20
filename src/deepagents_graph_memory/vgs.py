# - Sets up Deep Agents to work through graph tools and explains how to use that context.
# - Adds graph guidance while preserving application-supplied instructions.
# - Tests: test_vgs.py and test_combined_context.py check prompt and tool behavior.

"""Graph context guidance for Deep Agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from deepagents import HarnessProfile, register_harness_profile
from deepagents.middleware import filesystem
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage

VFS_TOOL_NAMES = frozenset({"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep"})
"""Deep Agents default virtual-filesystem tool names."""

_GRAPH_CONTEXT_INTRO = """## Virtual Graph System (VGS)

Graph context is enabled. The graph is the source of truth for recorded structured workflow context:
situations, rationales, actions, outcomes, artifacts, failures, evidence, decisions,
dependencies, and provenance. Its contents are recorded claims and evidence, not verified truth.
"""

_COMMON_GRAPH_GUIDANCE = """## Reading Graph Context

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
- Compare findings linked to the same subject before a decision depends on them. Check
  revisions, inputs, environments, and observation times to distinguish state changes
  from competing explanations. If material disagreement remains, pause that dependent
  decision and verify the disputed point with a targeted independent check.
- Truncated related findings are not evidence that the visible claims agree. Fetch or
  narrow context before treating them as a resolved answer.
- A conclusion marked `needs recheck` relied on a changed premise. Inspect the
  premise and its update or resolution, then verify current external state before
  relying on that conclusion. `Dependency status unknown` also requires a narrower
  recall or direct source check; it does not mean the conclusion is current.

## Writing Graph Context

- Use `record_graph_trace` for durable workflow events, especially meaningful observations,
  decisions, actions, failures, outcomes, artifacts, and evidence.
- Spawned workers sharing these tools write to the same project graph. Omit `subagent_id`
  to use automatic worker attribution; pass useful findings and their trace IDs back to the parent.
- Do not write every thought. Prefer facts that will help resume work, avoid repeated
  failed attempts, explain a decision, or connect evidence to an outcome.
- Record failures and dead ends with their outcomes so future work can avoid repeating them.
- When citing captured output, pass `evidence_refs` with a source ID assigned at collection,
  a locator, and optional revision, observed_at, or your summary. Reuse the ID for the
  same captured source across workers; give a separate execution a new ID even if its
  output text matches. These are citations, not votes or proof of independent checks.
  Plain `evidence` strings remain useful but do not identify a shared source.
- For related findings, supply a stable, narrow `subject` within the project namespace.
  Reuse the exact subject key supplied by the parent or application for the same
  question. When a write tool has a bound subject, omit `subject` in the call. Do
  not invent a new key for another report of that question. Keep changing revisions
  in trace context or evidence, and use separate subjects for environments whose
  states should not be compared as one question.
  Supply `observed_at` only from known evidence or tool output; do not guess from the
  recording clock. Use `finding_type="state"` for mutable observed state and
  `"interpretation"` for explanations. Use `supersedes` only for an evidenced newer
  mutable state. Arrival order and elapsed time alone never supersede a finding;
  preserve parallel contenders until evidence resolves them.
- After checking competing claims, record an evidenced resolution with `resolves`
  pointing to at least two same-subject traces. Explain the review in the rationale
  and evidence. This records a judgment; it does not make the graph verify truth.
- When a decision relies on recorded findings, pass their Trace IDs in `depends_on`.
  Rechecking creates a new trace that cites the findings actually used; keep the
  original decision as history.
- Do not store ordinary user preferences, profile facts, or unrelated notes in the graph.
- Generated `/graph/...` markdown paths are read-only views over graph data, not storage locations to edit.
- If low-level graph write tools are exposed, use them only with clear labels, relationship
  names, scope/provenance metadata, and schema discipline. Do not create arbitrary node or
  edge types just because a fact could be represented.
"""

VGS_SYSTEM_PROMPT_SUFFIX = (
    _GRAPH_CONTEXT_INTRO + "\nIn VGS mode, do not assume the default Deep Agents filesystem tools are available.\n\n" + _COMMON_GRAPH_GUIDANCE
)
"""Default VGS system prompt."""

GRAPH_CONTEXT_SYSTEM_PROMPT_SUFFIX = (
    _GRAPH_CONTEXT_INTRO
    + """
The virtual filesystem holds raw artifacts, logs, large tool results, normal memory,
skills, and preferences. Keep selective linked findings and decisions in the graph;
do not copy every file, read, or edit into it. Graph persistence does not preserve
the source files cited by a trace.

Choose the first tool for the task: open a current file for direct file work; recall
history or dependencies for context questions. Read the actual source and current
revision when needed. Save captured evidence before citing its source ID, locator,
revision, or observation time. If a source is missing, inaccessible, or changed,
the recorded finding remains historical and unverified against the current source.

File writes and graph writes are separate. If trace recording fails, retry with the
same operation identity without repeating a successful external action.

"""
    + _COMMON_GRAPH_GUIDANCE
)


class _VGSSystemPromptMiddleware(AgentMiddleware[Any, Any, Any]):
    """Add graph guidance while preserving application-supplied instructions."""

    def __init__(self, system_prompt: str, *, strip_filesystem_guidance: bool = True) -> None:
        self.system_prompt = system_prompt
        self.strip_filesystem_guidance = strip_filesystem_guidance

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage:
        """Apply VGS prompt guidance before the model call."""
        return handler(
            request.override(system_message=_apply_vgs_system_text(request.system_message, self.system_prompt, self.strip_filesystem_guidance)),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        """Async variant of `wrap_model_call`."""
        return await handler(
            request.override(system_message=_apply_vgs_system_text(request.system_message, self.system_prompt, self.strip_filesystem_guidance)),
        )


def graph_context_middleware() -> AgentMiddleware[Any, Any, Any]:
    """Add agent-local guidance for using graph context with Deep Agents files."""
    return _VGSSystemPromptMiddleware(GRAPH_CONTEXT_SYSTEM_PROMPT_SUFFIX, strip_filesystem_guidance=False)


def vgs_harness_profile(*, system_prompt_suffix: str | None = VGS_SYSTEM_PROMPT_SUFFIX) -> HarnessProfile:
    """Create a Deep Agents harness profile for VGS mode.

    Args:
        system_prompt_suffix: Optional VGS prompt text appended through middleware.

    Returns:
        Harness profile that excludes Deep Agents filesystem tools.
    """
    extra_middleware = () if system_prompt_suffix is None else (_VGSSystemPromptMiddleware(system_prompt_suffix),)
    return HarnessProfile(excluded_tools=VFS_TOOL_NAMES, extra_middleware=extra_middleware)


def register_vgs_harness_profile(model: str, *, system_prompt_suffix: str | None = VGS_SYSTEM_PROMPT_SUFFIX) -> None:
    """Register VGS mode for a Deep Agents model key.

    Args:
        model: Model key passed to `create_deep_agent`.
        system_prompt_suffix: Optional VGS prompt text.
    """
    register_harness_profile(model, vgs_harness_profile(system_prompt_suffix=system_prompt_suffix))


def _apply_vgs_system_text(system_message: SystemMessage | None, text: str, strip_filesystem_guidance: bool) -> SystemMessage:
    if system_message is None:
        content_blocks = []
    elif isinstance(system_message.content, str):
        content_blocks = [{"type": "text", "text": system_message.content}]
    else:
        content_blocks = list(system_message.content)
    if strip_filesystem_guidance:
        content_blocks = _remove_legacy_filesystem_guidance(content_blocks)
    if content_blocks:
        text = f"\n\n{text}"
    content_blocks.append({"type": "text", "text": text})
    content = cast("list[str | dict[str, str]]", content_blocks)
    return system_message.model_copy(update={"content": content}) if system_message else SystemMessage(content=content)


def _remove_legacy_filesystem_guidance(content_blocks: list[Any]) -> list[Any]:
    # Deep Agents 0.7 puts this guidance in tool descriptions, not system prompts.
    if not getattr(filesystem, "FILESYSTEM_SYSTEM_PROMPT", None):
        return content_blocks
    execution_prompt = getattr(filesystem, "EXECUTION_SYSTEM_PROMPT", "")
    result = []
    for block in content_blocks:
        text = block.get("text") if isinstance(block, dict) and block.get("type") == "text" else None
        if not isinstance(text, str):
            result.append(block)
            continue
        stripped = text.strip()
        _, separator, execution = stripped.partition("## Execute Tool `execute`")
        execution = separator + execution
        if not (
            stripped.startswith("## Following Conventions")
            and "## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`" in stripped
            and "## Large Tool Results" in stripped
            and "Offloaded tool results are stored under " in stripped
            and (not execution or execution == execution_prompt)
        ):
            result.append(block)
        elif execution:
            result.append({**block, "text": text[: len(text) - len(text.lstrip())] + execution})
    return result
