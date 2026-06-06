import deepagents_graph_memory.vgs as vgs
from deepagents_graph_memory import VFS_TOOL_NAMES, vgs_harness_profile


def test_vgs_harness_profile_excludes_default_vfs_tools():
    profile = vgs_harness_profile()

    assert profile.excluded_tools == VFS_TOOL_NAMES
    assert VFS_TOOL_NAMES == frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep"})


def test_register_vgs_harness_profile_registers_model(monkeypatch):
    calls = []

    def fake_register_harness_profile(model, profile):
        calls.append((model, profile))

    monkeypatch.setattr(vgs, "register_harness_profile", fake_register_harness_profile)

    vgs.register_vgs_harness_profile("test-model")

    assert calls[0][0] == "test-model"
    assert calls[0][1].excluded_tools == VFS_TOOL_NAMES
