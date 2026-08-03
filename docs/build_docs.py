#!/usr/bin/env python3
"""Build the XAIKit documentation site.

Renders two things into one output directory:

1. ``api/`` -- the API reference, generated from source by pdoc.
2. ``index.html`` / ``user_guide.html`` -- the narrative pages, written by hand
   as Markdown in this folder and converted here.

Usage::

    python docs/build_docs.py                 # -> site/
    python docs/build_docs.py -o /tmp/out      # somewhere else
    python docs/build_docs.py --mock           # stub heavy deps (see below)

``--mock`` replaces torch/shap/captum/... with empty stand-in modules so the
build runs without the full scientific stack installed. Signatures and
docstrings still render; only live introspection of those libraries is lost.
It makes CI installs small, at the cost of slightly less precise rendering of
any default value that comes from a mocked library.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# Documented in the API reference. Order controls the sidebar order.
MODULES = [
    "src",
    "src.api",
    "src.data_loaders",
    "src.ai_models",
    "src.xai_adapter",
    "src.experiment_planner",
    "src.cognitive_models",
    "src.virtual_experiment_executor",
    "src.statistical_analyst",
    "src.result_visualizer",
]

# Heavy third-party packages that only matter at runtime, never for signatures.
# numpy/pandas are deliberately absent: they appear in type annotations and
# default values, so they must stay real.
MOCK_PACKAGES = [
    "torch", "torchvision", "captum", "shap", "lime", "xgboost", "imodels",
    "mlxtend", "stable_baselines3", "sb3_contrib", "gymnasium", "gym",
    "skopt", "numba", "ucimlrepo", "datasets", "plotly", "seaborn", "skimage",
]

# Narrative pages: (markdown source, output file, nav title).
PAGES = [
    ("index.md", "index.html", "Home"),
    ("user_guide.md", "user_guide.html", "User Guide"),
]


class _MockMeta(type):
    """Metaclass so attribute access on a mocked class yields another class."""

    def __getattr__(cls, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        child = _mock_class(f"{cls.__name__}.{name}")
        setattr(cls, name, child)
        return child


class _MockObject(metaclass=_MockMeta):
    """Base for every mocked attribute.

    Mocked attributes must be real classes, not module-like objects. Libraries
    genuinely introspect them -- scipy, for one, calls
    ``issubclass(cls, torch.Tensor)`` whenever ``torch`` is present in
    ``sys.modules``, which raises TypeError against a non-class. Being a class
    also lets mocked types be subclassed, as in ``class Net(nn.Module)``.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __getattr__(self, name: str):
        return _mock_class(name)


def _mock_class(name: str) -> type:
    return _MockMeta(name.rsplit(".", 1)[-1], (_MockObject,), {})


class _Mock(types.ModuleType):
    """Stand-in module that yields another stand-in for any attribute."""

    __path__: list[str] = []

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        child = _mock_class(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child


class _MockFinder:
    """Serve a stand-in for any mocked package *and its submodules*.

    A plain ``sys.modules`` entry is not enough: ``import torch.nn as nn`` asks
    the import system for the ``torch.nn`` submodule specifically, which no
    attribute hook can satisfy. This finder answers for the whole subtree.
    """

    def __init__(self, prefixes: list[str]) -> None:
        self.prefixes = prefixes

    def _handles(self, fullname: str) -> bool:
        root = fullname.split(".")[0]
        return root in self.prefixes

    def find_module(self, fullname: str, path=None):  # legacy API, still called
        return self if self._handles(fullname) else None

    def load_module(self, fullname: str):
        if fullname not in sys.modules:
            sys.modules[fullname] = _Mock(fullname)
        return sys.modules[fullname]

    def find_spec(self, fullname: str, path=None, target=None):
        if not self._handles(fullname):
            return None
        from importlib.machinery import ModuleSpec

        spec = ModuleSpec(fullname, _MockLoader(), is_package=True)
        spec.submodule_search_locations = []
        return spec


class _MockLoader:
    def create_module(self, spec):
        return _Mock(spec.name)

    def exec_module(self, module) -> None:
        return None


def install_mocks() -> None:
    for name in MOCK_PACKAGES:
        if name not in sys.modules:
            sys.modules[name] = _Mock(name)
    sys.meta_path.insert(0, _MockFinder(MOCK_PACKAGES))


def resolve_lazy_exports() -> None:
    """Materialise ``src``'s PEP 562 lazy exports so pdoc can see them.

    ``src/__init__.py`` resolves its public names through ``__getattr__`` to keep
    ``import src`` cheap. pdoc looks names up in ``__dict__`` and reports every
    one as unresolvable, so pull them in once here. This touches only the
    in-memory module during the docs build; the source stays lazy.
    """
    src = importlib.import_module("src")
    resolved, failed = 0, []
    for name in getattr(src, "__all__", []):
        try:
            setattr(src, name, getattr(src, name))
            resolved += 1
        except Exception as exc:  # noqa: BLE001 - report, don't abort the build
            failed.append(f"{name}: {exc}")
    print(f"  resolved {resolved} lazy exports on 'src'")
    for line in failed:
        print(f"  WARNING unresolved {line}")


def render_api(out_dir: Path) -> None:
    import pdoc
    import pdoc.render

    pdoc.render.configure(
        docformat="google",
        search=True,
        show_source=True,
        footer_text="XAIKit — XAI Interpretation Simulator Toolkit",
    )
    pdoc.pdoc(*MODULES, output_directory=out_dir / "api")


def render_narrative(out_dir: Path) -> None:
    import markdown2

    template = (DOCS_DIR / "_page_template.html").read_text()
    nav = "\n".join(
        f'        <a href="{href}">{title}</a>' for _, href, title in PAGES
    ) + '\n        <a href="api/index.html">API Reference</a>'

    for source, target, title in PAGES:
        md = (DOCS_DIR / source).read_text()
        body = markdown2.markdown(
            md,
            extras=["fenced-code-blocks", "tables", "header-ids", "code-friendly"],
        )
        page = (
            template.replace("{{TITLE}}", title)
            .replace("{{NAV}}", nav)
            .replace("{{CONTENT}}", str(body))
        )
        (out_dir / target).write_text(page)
        print(f"  wrote {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default=str(REPO_ROOT / "site"))
    parser.add_argument("--mock", action="store_true", help="stub heavy deps")
    args = parser.parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    if args.mock:
        print(f"Mocking {len(MOCK_PACKAGES)} heavy packages")
        install_mocks()

    out_dir = Path(args.output).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print("Resolving lazy exports...")
    resolve_lazy_exports()

    print(f"Rendering API reference for {len(MODULES)} modules...")
    render_api(out_dir)

    print("Rendering narrative pages...")
    render_narrative(out_dir)

    # GitHub Pages otherwise runs Jekyll, which strips files beginning with "_".
    (out_dir / ".nojekyll").touch()

    pages = len(list(out_dir.rglob("*.html")))
    print(f"\nBuilt {pages} pages -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
