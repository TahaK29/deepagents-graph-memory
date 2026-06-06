"""VGS harness-profile helpers for Deep Agents."""

from __future__ import annotations

from deepagents import HarnessProfile, register_harness_profile

VFS_TOOL_NAMES = frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep"})
"""Deep Agents default virtual-filesystem tool names."""

VGS_SYSTEM_PROMPT_SUFFIX = (
    "Graph context is enabled. Use graph memory tools for structured workflow context, provenance, decisions, actions, and outcomes. "
    "Do not assume Deep Agents filesystem tools are available in VGS mode."
)
"""Default prompt suffix for VGS mode."""


def vgs_harness_profile(*, system_prompt_suffix: str | None = VGS_SYSTEM_PROMPT_SUFFIX) -> HarnessProfile:
    """Create a Deep Agents harness profile that hides the default VFS tools.

    Args:
        system_prompt_suffix: Optional prompt suffix for graph-context mode.

    Returns:
        Harness profile that excludes Deep Agents filesystem tools.
    """
    return HarnessProfile(system_prompt_suffix=system_prompt_suffix, excluded_tools=VFS_TOOL_NAMES)


def register_vgs_harness_profile(model: str, *, system_prompt_suffix: str | None = VGS_SYSTEM_PROMPT_SUFFIX) -> None:
    """Register VGS mode for a Deep Agents model key.

    Args:
        model: Model key passed to `create_deep_agent`.
        system_prompt_suffix: Optional prompt suffix for graph-context mode.
    """
    register_harness_profile(model, vgs_harness_profile(system_prompt_suffix=system_prompt_suffix))
