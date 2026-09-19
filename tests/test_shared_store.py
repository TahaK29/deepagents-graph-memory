"""Regression checks for one in-process graph shared by multiple agent backends."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from deepagents_graph_memory.backend import GraphMemoryBackend
from deepagents_graph_memory.errors import GraphMemoryConfigurationError, GraphMemoryValidationError
from deepagents_graph_memory.kuzu_store import KuzuGraphStore


def trace(backend, trace_id, **kwargs):
    return backend.record_graph_trace(
        trace_id=trace_id,
        situation="observed failure",
        rationale="cause was identified",
        action="fixed source",
        outcome="test passed",
        **kwargs,
    )


def test_shared_writers_merge_node_edge_properties_and_first_use_schema():
    store = KuzuGraphStore.memory()
    parent = GraphMemoryBackend(store, namespace=("project",))
    child = GraphMemoryBackend(store, namespace=("project",))
    start = Barrier(2)

    def write(backend, key):
        start.wait()
        backend.add_graph_node("File", "shared", {key: True})
        backend.add_graph_edge("File", "shared", "DEPENDS_ON", "File", "target", {key: True})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, backend, key) for backend, key in ((parent, "parent"), (child, "child"))]
        for future in futures:
            future.result()

    node = store.get_node("File", "shared", scope_key="project")
    assert node.properties["parent"] is True and node.properties["child"] is True
    edge = next(edge for edge in store.get_neighbors("File", "shared", scope_key="project").edges if edge.relationship == "DEPENDS_ON")
    assert edge.properties["parent"] is True and edge.properties["child"] is True
    assert node.properties["created_at"] <= node.properties["updated_at"]


def test_concurrent_traces_share_values_with_distinct_provenance():
    store = KuzuGraphStore.memory()
    backends = [GraphMemoryBackend(store, namespace=("project",)) for _ in range(3)]
    start = Barrier(3)

    def write(index):
        start.wait()
        trace(
            backends[index],
            f"trace-{index}",
            artifacts=["src/shared.py"],
            evidence=["pytest passed"],
            agent_id="parent",
            subagent_id=f"child-{index}",
            run_id="run-1",
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(write, range(3)))

    for label in ("Artifact", "Evidence"):
        ids = store.list_node_ids(label, scope_key="project").items
        assert len(ids) == 1
        node = store.get_node(label, ids[0], scope_key="project")
        assert "trace_id" not in node.properties
        assert "subagent_id" not in node.properties
        edges = store.get_neighbors(label, ids[0], scope_key="project", depth=1).edges
        assert {edge.properties["subagent_id"] for edge in edges} == {"child-0", "child-1", "child-2"}
        assert {edge.properties["run_id"] for edge in edges} == {"run-1"}


def test_duplicate_and_component_collision_preserve_existing_graph():
    store = KuzuGraphStore.memory()
    one = GraphMemoryBackend(store, namespace=("one",))
    two = GraphMemoryBackend(store, namespace=("two",))
    trace(one, "stable")
    original = store.get_node("Trace", "stable", scope_key="one")
    with pytest.raises(GraphMemoryValidationError, match="already exists"):
        trace(one, "stable", artifacts=["new artifact"])
    assert store.get_node("Trace", "stable", scope_key="one") == original
    assert store.list_node_ids("Artifact", scope_key="one").items == []
    trace(two, "stable")
    one.add_graph_node("Outcome", "collision-outcome", {"prior": True})
    with pytest.raises(GraphMemoryValidationError, match="already exists"):
        trace(one, "collision")
    assert store.get_node("Trace", "collision", scope_key="one") is None
    assert store.get_node("Outcome", "collision-outcome", scope_key="one").properties["prior"] is True


def test_late_database_failure_rolls_back_trace_and_connection_recovers(monkeypatch):
    store = KuzuGraphStore.memory()
    backend = GraphMemoryBackend(store)
    trace(backend, "prior")
    original_query = store.graph.query

    def fail_once(query, params=None):
        if "MERGE (source)-[rel:PRODUCED]" in query:
            monkeypatch.setattr(store.graph, "query", original_query)
            raise RuntimeError("injected write failure")
        return original_query(query, params)

    monkeypatch.setattr(store.graph, "query", fail_once)
    with pytest.raises(GraphMemoryConfigurationError, match="injected write failure"):
        trace(backend, "failed")
    assert store.get_node("Trace", "failed") is None
    assert store.list_node_ids("Situation").items == ["prior-situation"]
    assert store.get_node("Trace", "prior") is not None
    trace(backend, "after")
    assert store.search("fixed source").items


def test_caught_database_error_cannot_autocommit_after_abort():
    store = KuzuGraphStore.memory()
    with pytest.raises(GraphMemoryConfigurationError, match="rolled back"):
        with store.transaction():
            store.add_node("File", "before")
            with pytest.raises(GraphMemoryConfigurationError):
                store._query("INVALID CYPHER", {})
            with pytest.raises(GraphMemoryConfigurationError, match="already failed"):
                store.add_node("File", "after")
    assert store.list_node_ids("File").items == []
    store.add_node("File", "recovered")
    assert store.list_node_ids("File").items == ["recovered"]


def test_document_batch_validation_failure_is_atomic():
    store = KuzuGraphStore.memory()
    good = SimpleNamespace(type="File", id="good", properties={"name": "good"})
    bad = SimpleNamespace(type="File", id="bad/id", properties={})
    docs = [SimpleNamespace(nodes=[good], relationships=[]), SimpleNamespace(nodes=[bad], relationships=[])]
    with pytest.raises(GraphMemoryValidationError):
        store.add_graph_documents(docs)
    assert store.list_node_ids("File").items == []
    store.add_graph_documents([docs[0]])
    assert store.list_node_ids("File").items == ["good"]


def test_relationship_label_accepts_multiple_endpoint_pairs():
    store = KuzuGraphStore.memory()
    store.add_edge("File", "a", "LINKS", "File", "b")
    store.add_edge("File", "a", "LINKS", "Task", "c")
    assert "(:File)-[:LINKS]->(:Task)" in store.get_schema()
    assert store.search("LINKS").items


def test_invalid_namespace_and_trace_inputs_make_no_changes():
    store = KuzuGraphStore.memory()
    backend = GraphMemoryBackend(store, namespace="a|b")
    with pytest.raises(GraphMemoryValidationError, match="namespace"):
        backend.add_graph_node("File", "x")
    assert store.list_labels().items == []
    backend = GraphMemoryBackend(store)
    for value in ("artifact", 0, {"a": "b"}):
        with pytest.raises(GraphMemoryValidationError, match="artifacts"):
            trace(backend, "invalid", artifacts=value)
    with pytest.raises(GraphMemoryValidationError):
        trace(backend, "a" * 250)
    with pytest.raises(GraphMemoryValidationError, match="cannot override"):
        trace(backend, "invalid", kind="other")
    assert store.list_labels().items == []


def test_search_filters_scope_before_limiting_fts_candidates():
    store = KuzuGraphStore.memory()
    for index in range(10):
        store.add_node("Entity", f"other-{index}", properties={"text": "needle needle needle"}, scope_key="other")
    store.add_node("Entity", "local", properties={"text": "needle " + "haystack " * 100})
    store.add_node("Entity", "wanted", properties={"text": "needle " + "haystack " * 100}, scope_key="wanted")
    assert [item.path for item in store.search("needle", limit=1).items] == ["/nodes/Entity/local.md"]
    assert [item.path for item in store.search("needle", scope_key="wanted", limit=1).items] == ["/nodes/Entity/wanted.md"]


def test_namespace_factory_reads_langgraph_runtime_context():
    from langgraph.graph import END, START, StateGraph

    store = KuzuGraphStore.memory()
    backend = GraphMemoryBackend(store, namespace=lambda runtime: (runtime.context["project"],))
    graph = StateGraph(dict)

    def write(state):
        backend.add_graph_node("File", "from-runtime")
        return state

    graph.add_node("write", write)
    graph.add_edge(START, "write")
    graph.add_edge("write", END)
    graph.compile().invoke({}, context={"project": "real"})
    assert store.get_node("File", "from-runtime", scope_key="real") is not None
    assert store.get_node("File", "from-runtime") is None


@pytest.mark.parametrize("factory", [lambda runtime: "a|b", lambda runtime: ("bad|name",), lambda runtime: (), lambda runtime: 3])
def test_bad_namespace_factory_result_is_validation_error(factory):
    backend = GraphMemoryBackend(KuzuGraphStore.memory(), namespace=factory)
    with pytest.raises(GraphMemoryValidationError, match="namespace"):
        backend.add_graph_node("File", "x")


def test_namespace_factory_exception_is_validation_error():
    def fail(runtime):
        raise KeyError("project missing")

    backend = GraphMemoryBackend(KuzuGraphStore.memory(), namespace=fail)
    with pytest.raises(GraphMemoryValidationError, match="namespace factory failed"):
        backend.add_graph_node("File", "x")


def test_late_trace_validation_rolls_back_all_nodes_and_edges():
    store = KuzuGraphStore.memory()
    backend = GraphMemoryBackend(store)
    trace(backend, "prior")
    with pytest.raises(GraphMemoryValidationError, match="node_id"):
        trace(backend, "bad", run_id="bad/id", artifacts=["src/x.py"])
    assert store.get_node("Trace", "bad") is None
    assert store.list_node_ids("Situation").items == ["prior-situation"]
    assert store.list_node_ids("Artifact").items == []
    trace(backend, "after")


def test_commit_failure_rolls_back_and_next_write_succeeds(monkeypatch):
    store = KuzuGraphStore.memory()
    original_query = store.graph.query

    def fail_commit(query, params=None):
        if query.strip().upper() == "COMMIT;":
            monkeypatch.setattr(store.graph, "query", original_query)
            raise RuntimeError("injected commit failure")
        return original_query(query, params)

    monkeypatch.setattr(store.graph, "query", fail_commit)
    with pytest.raises(GraphMemoryConfigurationError, match="injected commit failure"):
        store.add_node("File", "failed")
    assert store.list_node_ids("File").items == []
    store.add_node("File", "succeeded")
    assert store.list_node_ids("File").items == ["succeeded"]


def test_document_relationship_failure_rolls_back_prior_batch():
    store = KuzuGraphStore.memory()
    store.add_node("File", "prior", properties={"protected": True})
    good = SimpleNamespace(type="File", id="good", properties={})
    bad_relationship = SimpleNamespace(source=good, target=good, type="BAD-NAME", properties={})
    document = SimpleNamespace(nodes=[good], relationships=[bad_relationship])
    with pytest.raises(GraphMemoryValidationError, match="relationship"):
        store.add_graph_documents([document])
    assert store.list_node_ids("File").items == ["prior"]
    assert store.get_node("File", "prior").properties["protected"] is True


def test_async_tool_calls_share_one_store():
    import asyncio

    from deepagents_graph_memory.tools import graph_memory_tools

    store = KuzuGraphStore.memory()
    backends = [GraphMemoryBackend(store, namespace=("project",)) for _ in range(2)]
    tools = [{tool.name: tool for tool in graph_memory_tools(backend)} for backend in backends]

    async def write():
        return await asyncio.gather(
            tools[0]["record_graph_trace"].ainvoke(
                {"situation": "one", "rationale": "shared", "action": "write", "outcome": "done", "agent_id": "parent"}
            ),
            tools[1]["record_graph_trace"].ainvoke(
                {"situation": "two", "rationale": "shared", "action": "write", "outcome": "done", "subagent_id": "child"}
            ),
        )

    results = asyncio.run(write())
    assert all("Recorded graph trace" in result for result in results)
    assert len(store.list_node_ids("Trace", scope_key="project").items) == 2


def test_case_variant_table_names_are_rejected_without_aliasing():
    store = KuzuGraphStore.memory()
    store.add_node("Service", "one")
    with pytest.raises(GraphMemoryValidationError, match="conflicts"):
        store.add_node("service", "two")
    assert store.list_node_ids("Service").items == ["one"]
    store.add_node("Service", "two")
    assert store.list_node_ids("Service").items == ["one", "two"]
    with pytest.raises(GraphMemoryValidationError, match="conflicts"):
        store.add_edge("Service", "one", "service", "Service", "two")
    store.add_edge("Service", "one", "LINKS", "Service", "two")
    with pytest.raises(GraphMemoryValidationError, match="conflicts"):
        store.add_edge("Service", "one", "links", "Service", "two")


def test_concurrent_case_variant_schema_has_one_unambiguous_winner():
    store = KuzuGraphStore.memory()
    start = Barrier(2)

    def write(label):
        start.wait()
        store.add_node(label, label)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, label) for label in ("Thing", "thing")]
        outcomes = []
        for future in futures:
            try:
                future.result()
                outcomes.append("ok")
            except GraphMemoryValidationError:
                outcomes.append("rejected")
    assert sorted(outcomes) == ["ok", "rejected"]
    assert len(store.list_labels().items) == 1


@pytest.mark.parametrize("label,prefix,text", [("Artifact", "artifact", "src/shared.py"), ("Evidence", "evidence", "pytest passed")])
@pytest.mark.parametrize("properties", [{"value": "wrong"}, {}])
def test_existing_shared_value_collision_rejects_trace(label, prefix, text, properties):
    from deepagents_graph_memory.backend import _value_id

    store = KuzuGraphStore.memory()
    backend = GraphMemoryBackend(store)
    node_id = _value_id(prefix, text)
    store.add_node(label, node_id, properties=properties)
    kwargs = {"artifacts" if label == "Artifact" else "evidence": [text]}
    with pytest.raises(GraphMemoryValidationError, match="conflicting value"):
        trace(backend, "conflict", **kwargs)
    assert store.get_node("Trace", "conflict") is None
    assert store.get_node(label, node_id).properties == properties


def test_scope_metadata_override_and_invalid_json_are_rejected():
    store = KuzuGraphStore.memory()
    backend = GraphMemoryBackend(store, namespace=("project",))
    for properties in ({"scope_key": "other"}, {"nested": {1: "a", "1": "b"}}, {"bad": float("nan")}):
        with pytest.raises(GraphMemoryValidationError):
            backend.add_graph_node("File", "bad", properties)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(GraphMemoryValidationError, match="JSON serializable"):
        backend.add_graph_node("File", "bad", {"cycle": cycle})
    assert store.list_node_ids("File", scope_key="project").items == []
    with pytest.raises(GraphMemoryValidationError, match="scope_key"):
        backend.add_graph_edge("File", "a", "LINKS", "File", "b", {"scope_key": "other"})
    assert store.list_node_ids("File", scope_key="project").items == []


def test_node_and_edge_upserts_preserve_creation_time_and_omitted_properties():
    store = KuzuGraphStore.memory()
    backend = GraphMemoryBackend(store)
    backend.add_graph_node("File", "a", {"first": 1})
    first_node = store.get_node("File", "a")
    backend.add_graph_node("File", "a", {"second": 2})
    second_node = store.get_node("File", "a")
    assert second_node.properties["first"] == 1
    assert second_node.properties["created_at"] == first_node.properties["created_at"]
    backend.add_graph_edge("File", "a", "LINKS", "File", "b", {"first": 1})
    first_edge = next(edge for edge in store.get_neighbors("File", "a").edges if edge.relationship == "LINKS")
    backend.add_graph_edge("File", "a", "LINKS", "File", "b", {"second": 2})
    second_edge = next(edge for edge in store.get_neighbors("File", "a").edges if edge.relationship == "LINKS")
    assert second_edge.properties["first"] == 1
    assert second_edge.properties["created_at"] == first_edge.properties["created_at"]
    assert second_edge.properties["updated_at"] >= first_edge.properties["updated_at"]


def test_generated_trace_and_shared_value_ids_use_full_digests():
    store = KuzuGraphStore.memory()
    backend = GraphMemoryBackend(store)
    trace_id = backend.record_graph_trace(
        situation="observed", rationale="reasoned", action="acted", outcome="completed", artifacts=["src/a.py"], evidence=["test passed"]
    )
    assert trace_id.startswith("trace-") and len(trace_id.removeprefix("trace-")) == 32
    assert len(store.list_node_ids("Artifact").items[0].removeprefix("artifact-")) == 64
    assert len(store.list_node_ids("Evidence").items[0].removeprefix("evidence-")) == 64
