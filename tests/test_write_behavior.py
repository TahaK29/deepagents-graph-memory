from deepagents_graph_memory.backend import READ_ONLY_ERROR, GraphMemoryBackend
from deepagents_graph_memory.stores import InMemoryGraphStore


def test_write_and_edit_are_read_only():
    backend = GraphMemoryBackend(InMemoryGraphStore())

    write = backend.write("/schema.md", "content")
    edit = backend.edit("/schema.md", "old", "new")

    assert write.error == READ_ONLY_ERROR
    assert edit.error == READ_ONLY_ERROR
