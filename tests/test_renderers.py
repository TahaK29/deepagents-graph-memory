from deepagents_graph_memory.renderers import render_neighborhood, render_node, render_schema, render_search
from deepagents_graph_memory.stores import GraphEdge, GraphNode, NeighborhoodResult, SearchItem, SearchResult


def test_render_schema():
    markdown = render_schema("Node labels:\n- service")

    assert markdown.startswith("# Graph Schema")
    assert "service" in markdown


def test_render_node_groups_relationships():
    node = GraphNode(label="service", id="langfuse", properties={"tier": "prod"})
    result = NeighborhoodResult(
        node=node,
        edges=[
            GraphEdge("service", "langfuse", "DEPENDS_ON", "service", "redis"),
            GraphEdge("team", "sre-team", "OWNS", "service", "langfuse"),
            GraphEdge("incident", "incident-123", "AFFECTED", "service", "langfuse"),
        ],
    )

    markdown = render_node(node, result)

    assert "# service: langfuse" in markdown
    assert "## Depends on" in markdown
    assert "[redis](/nodes/service/redis.md)" in markdown
    assert "## Owns (incoming)" in markdown
    assert "## Affected (incoming)" in markdown


def test_render_neighborhood_truncation():
    node = GraphNode(label="service", id="langfuse")
    result = NeighborhoodResult(node=node, edges=[], truncated_edges=True)

    markdown = render_neighborhood(result)

    assert "No relationships found." in markdown
    assert "Results truncated" in markdown


def test_render_search_results():
    result = SearchResult(items=[SearchItem(path="/nodes/service/langfuse.md", title="service: langfuse", text='{"tier": "prod"}')])

    markdown = render_search("langfuse", result)

    assert "# Graph Search: langfuse" in markdown
    assert "[service: langfuse](/nodes/service/langfuse.md)" in markdown
