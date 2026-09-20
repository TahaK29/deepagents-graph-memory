"""Require the complete tested release matrix before uploading any wheel."""

import itertools
import re
import shutil
import tomllib
from pathlib import Path

version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
expected = set(itertools.product(("cp311", "cp312", "cp313", "cp314"), ("linux-x86_64", "linux-aarch64", "mac-arm64", "mac-x86_64", "windows")))
found = set()
wheels = list(Path("artifacts").rglob("*.whl"))
for wheel in wheels:
    name, wheel_version, python, abi, platform = wheel.stem.split("-")
    if name != "deepagents_graph_memory" or wheel_version != version or abi != python:
        raise ValueError(f"Unexpected release artifact: {wheel.name}")
    linux = [re.fullmatch(r"manylinux_2_(\d+)_(x86_64|aarch64)", tag) for tag in platform.split(".")]
    mac = re.fullmatch(r"macosx_15_0_(arm64|x86_64)", platform)
    if all(linux) and len({match[2] for match in linux}) == 1 and min(int(match[1]) for match in linux) <= 28:
        target = f"linux-{linux[0][2]}"
    elif mac:
        target = f"mac-{mac[1]}"
    elif platform == "win_amd64":
        target = "windows"
    else:
        raise ValueError(f"Unexpected platform: {platform}")
    if (python, target) in found:
        raise ValueError(f"Duplicate release target: {wheel.name}")
    found.add((python, target))
if found != expected:
    raise ValueError(f"Incomplete release matrix: missing {expected - found}, unexpected {found - expected}")
Path("dist").mkdir(exist_ok=True)
if list(Path("dist").iterdir()):
    raise ValueError("Release output directory must be empty")
for wheel in wheels:
    shutil.copy2(wheel, Path("dist") / wheel.name)
print(f"Collected {len(wheels)} tested wheels for {version}")
