# - Checks how the Ladybug store prepares and searches text using a pretend database.
# - Cases: using Ladybug text search when available, and including aliases and descriptions in searchable text.

from deepagents_graph_memory.ladybug_store import LadybugGraphStore


class FakeGraph:
    def __init__(self):
        self.queries = []

    def query(self, query, params=None):
        self.queries.append((query, params or {}))
        if "SHOW_TABLES" in query:
            return [{"name": "service", "type": "NODE"}]
        if "QUERY_FTS_INDEX" in query:
            return [
                {
                    "node": {
                        "_LABEL": "service",
                        "id": "auth-service",
                        "properties": '{"aliases": ["login service"], "scope_key": "tenant"}',
                    },
                    "score": 4.2,
                }
            ]
        if "RETURN n.id AS id" in query:
            return [{"id": "auth-service", "scope_key": "tenant"}]
        if "RETURN n" in query:
            return [
                {
                    "n": {
                        "_LABEL": "service",
                        "id": "auth-service",
                        "properties": '{"aliases": ["login service"], "scope_key": "tenant"}',
                    }
                }
            ]
        return []


def test_ladybug_search_uses_fts_when_available():
    graph = FakeGraph()
    store = LadybugGraphStore(graph)

    result = store.search("login service", scope_key="tenant")

    assert result.items[0].path == "/nodes/service/auth-service.md"
    queries = "\n".join(query for query, _params in graph.queries)
    assert any(query.startswith("LOAD ") and "fts" in query for query in queries.splitlines())
    assert "CREATE_FTS_INDEX" in queries
    assert "QUERY_FTS_INDEX" in queries


def test_ladybug_add_node_writes_search_text():
    graph = FakeGraph()
    store = LadybugGraphStore(graph)

    store.add_node("service", "auth-service", properties={"aliases": ["login service"], "description": "Handles login."})

    merge_params = [params for query, params in graph.queries if "MERGE (n:service" in query][0]
    assert "login service" in merge_params["search_text"]
    assert "Handles login." in merge_params["search_text"]


def test_relationship_search_limits_distinct_sources():
    store = LadybugGraphStore.memory()
    try:
        for index in range(4):
            store.add_edge("File", "many", "DEPENDS_ON", "File", f"dep{index}", properties={"reason": "build input"})
        store.add_edge("File", "other", "DEPENDS_ON", "File", "last", properties={"reason": "runtime input"})
        store.add_edge("File", "hidden", "DEPENDS_ON", "File", "secret", scope_key="other")

        result = store.search("depends", limit=2)
        assert {item.path for item in result.items} == {"/nodes/File/many.md", "/nodes/File/other.md"}
        assert not result.truncated
        assert all("DEPENDS_ON" in item.title and "input" in item.text for item in result.items)
        limited = store.search("depends", limit=1)
        assert len(limited.items) == 1
        assert limited.truncated
    finally:
        store.close()
