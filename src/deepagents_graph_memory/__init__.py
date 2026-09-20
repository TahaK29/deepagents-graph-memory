# - Gives apps one place to import the main graph features.
# - Tests: test_vgs.py checks the public graph setup helpers;
#        test_optional_vgs_dependency.py checks what happens without LadybugDB.

"""Graph-backed memory backend for LangChain Deep Agents."""

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.paths import make_graph_subject
from deepagents_graph_memory.tools import graph_memory_tools
from deepagents_graph_memory.vgs import VFS_TOOL_NAMES, graph_context_middleware, register_vgs_harness_profile, vgs_harness_profile

__all__ = [
    "GraphMemoryBackend",
    "VFS_TOOL_NAMES",
    "graph_context_middleware",
    "graph_memory_tools",
    "make_graph_subject",
    "register_vgs_harness_profile",
    "vgs_harness_profile",
]
