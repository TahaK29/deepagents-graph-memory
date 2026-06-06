"""Minimal Deep Agents setup with Kuzu-backed VGS mode."""

from deepagents import create_deep_agent

from deepagents_graph_memory import GraphMemoryBackend, graph_memory_tools, register_vgs_harness_profile

MODEL = "google_genai:gemini-3.5-flash"

register_vgs_harness_profile(MODEL)
graph_backend = GraphMemoryBackend.local("./graph-memory")

agent = create_deep_agent(
    model=MODEL,
    tools=[*graph_memory_tools(graph_backend)],
    memory=[
        "/graph/index.md",
        "/graph/schema.md",
    ],
    backend=graph_backend,
)
