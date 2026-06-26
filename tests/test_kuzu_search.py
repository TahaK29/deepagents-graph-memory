import importlib
import sys
import types


def import_kuzu_store(monkeypatch):
    fake_kuzu = types.ModuleType("kuzu")
    fake_kuzu.Database = lambda path: object()

    fake_graph_module = types.ModuleType("langchain_community.graphs.kuzu_graph")
    fake_graph_module.KuzuGraph = object

    monkeypatch.setitem(sys.modules, "kuzu", fake_kuzu)
    monkeypatch.setitem(sys.modules, "langchain_community", types.ModuleType("langchain_community"))
    monkeypatch.setitem(sys.modules, "langchain_community.graphs", types.ModuleType("langchain_community.graphs"))
    monkeypatch.setitem(sys.modules, "langchain_community.graphs.kuzu_graph", fake_graph_module)
    sys.modules.pop("deepagents_graph_memory.kuzu_store", None)
    module = importlib.import_module("deepagents_graph_memory.kuzu_store")
    sys.modules.pop("deepagents_graph_memory.kuzu_store", None)
    return module


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
                        "_label": "service",
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
                        "_label": "service",
                        "id": "auth-service",
                        "properties": '{"aliases": ["login service"], "scope_key": "tenant"}',
                    }
                }
            ]
        return []


def test_kuzu_search_uses_fts_when_available(monkeypatch):
    kuzu_store = import_kuzu_store(monkeypatch)
    graph = FakeGraph()
    store = kuzu_store.KuzuGraphStore(graph)

    result = store.search("login service", scope_key="tenant")

    assert result.items[0].path == "/nodes/service/auth-service.md"
    queries = "\n".join(query for query, _params in graph.queries)
    assert "LOAD fts" in queries
    assert "CREATE_FTS_INDEX" in queries
    assert "QUERY_FTS_INDEX" in queries


def test_kuzu_add_node_writes_search_text(monkeypatch):
    kuzu_store = import_kuzu_store(monkeypatch)
    graph = FakeGraph()
    store = kuzu_store.KuzuGraphStore(graph)

    store.add_node("service", "auth-service", properties={"aliases": ["login service"], "description": "Handles login."})

    merge_params = [params for query, params in graph.queries if "MERGE (n:service" in query][0]
    assert "login service" in merge_params["search_text"]
    assert "Handles login." in merge_params["search_text"]
