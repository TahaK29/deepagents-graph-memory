"""Minimal Deep Agents setup with graph memory mounted at `/graph/`."""

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

from deepagents_graph_memory import GraphMemoryBackend, graph_memory_tools

graph_backend = GraphMemoryBackend.local("./graph-memory")

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[*graph_memory_tools(graph_backend)],
    memory=[
        "/memories/preferences.md",
        "/graph/index.md",
        "/graph/schema.md",
    ],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=lambda rt: (rt.server_info.user.identity,)),
            "/graph/": graph_backend,
        },
    ),
)
