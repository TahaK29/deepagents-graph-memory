"""Build self-contained platform wheels from the pinned upstream runtime."""

import os
import shutil
import sys
from importlib.metadata import distribution
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


class NativeDistribution(Distribution):
    """Mark the copied CPython extension as platform-specific."""

    def has_ext_modules(self) -> bool:
        """Prevent setuptools from producing a universal wheel."""
        return True


class NativeWheel(bdist_wheel):
    """Keep the upstream interpreter, ABI, and minimum platform tags."""

    def get_tag(self) -> tuple[str, str, str]:
        """Use the tag of the exact native payload being copied."""
        tag = next(line[5:] for line in distribution("ladybug").read_text("WHEEL").splitlines() if line.startswith("Tag: "))
        return tuple(tag.split("-"))


class BundleRuntime(build_py):
    """Copy the upstream runtime and search extension before wheel repair."""

    def run(self) -> None:
        """Build Python sources and include licensed native dependencies."""
        super().run()
        if self.editable_mode:
            return
        dll_directory = None
        if sys.platform == "win32" and os.environ.get("GRAPH_MEMORY_BUILD_OPENSSL"):
            dll_directory = os.add_dll_directory(os.environ["GRAPH_MEMORY_BUILD_OPENSSL"])
        import ladybug

        vendor = Path(self.build_lib) / "deepagents_graph_memory" / "_vendor"
        vendor.mkdir(parents=True, exist_ok=True)
        (vendor / "__init__.py").write_text('"""Unmodified, pinned LadybugDB runtime; populated during wheel builds."""\n')
        upstream = distribution("ladybug")
        assert upstream.version == "0.20.3"
        source = Path(ladybug.__file__).parent
        shutil.copytree(source, vendor / "ladybug", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # Upstream repairs may already put dependencies beside the Python package.
        for sibling in source.parent.glob("ladybug.*"):
            if sibling.is_dir() and sibling.name.endswith((".libs", ".dylibs")):
                shutil.copytree(sibling, vendor / sibling.name, dirs_exist_ok=True)
        for item in upstream.files:
            if "/licenses/" in str(item):
                destination = vendor / "licenses" / Path(str(item)).name
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(upstream.locate_file(item), destination)
        with ladybug.Database(":memory:", buffer_pool_size=64 * 1024 * 1024) as db, ladybug.Connection(db) as connection:
            connection.execute("INSTALL fts;").close()
            connection.execute("LOAD fts;").close()
            result = connection.execute("CALL SHOW_LOADED_EXTENSIONS() RETURN *;")
            try:
                while result.has_next():
                    name, _, path = result.get_next()
                    if name.casefold() == "fts":
                        # Standard wheel repair tools must see this dlopen dependency.
                        extension = "libfts.dll" if sys.platform == "win32" else "libfts.so"
                        shutil.copy2(path, vendor / extension)
                        break
                else:
                    raise RuntimeError("LadybugDB did not report the loaded FTS extension")
            finally:
                result.close()
        if dll_directory is not None:
            dll_directory.close()


setup(distclass=NativeDistribution, cmdclass={"build_py": BundleRuntime, "bdist_wheel": NativeWheel})
