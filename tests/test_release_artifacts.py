"""Reject incomplete or mislabeled native releases before publishing."""

import subprocess
import sys
from pathlib import Path


def test_release_matrix(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "collect_wheels.py"
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.1"\n')
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    platforms = ("manylinux_2_28_x86_64", "manylinux_2_28_aarch64", "macosx_15_0_arm64", "macosx_15_0_x86_64", "win_amd64")
    for python in ("cp311", "cp312", "cp313", "cp314"):
        for platform in platforms:
            (artifacts / f"deepagents_graph_memory-0.1.1-{python}-{python}-{platform}.whl").touch()

    def run():
        return subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)

    selected = next(artifacts.glob("*cp311*win_amd64*"))
    selected.unlink()
    assert "Incomplete release matrix" in run().stderr
    selected.touch()
    wrong = selected.with_name(selected.name.replace("win_amd64", "manylinux_2_28_ppc64le"))
    selected.rename(wrong)
    assert "Unexpected platform" in run().stderr
    wrong.rename(selected)
    duplicate = artifacts / "duplicate"
    duplicate.mkdir()
    (duplicate / selected.name).touch()
    assert "Duplicate release target" in run().stderr
    (duplicate / selected.name).unlink()
    result = run()
    assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / "dist").glob("*.whl"))) == 20
    assert "must be empty" in run().stderr
