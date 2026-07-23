"""Neutral, dependency-free plotting helpers shared across domain modules.

This module is a leaf utility: it imports nothing from other ``src`` packages so
that both ``xai_adapter`` and ``ai_models`` (and any other module) can depend on
it without creating an import cycle.
"""

from __future__ import annotations

from typing import Any


def _patch_matplotlib_inline_rcparams(matplotlib_module: Any) -> None:
    """
    Compatibility shim for mixed matplotlib/matplotlib-inline environments.

    matplotlib-inline>=0.2 calls rcParams._get(), but older matplotlib runtimes
    only expose rcParams.get(). Adding the alias before pyplot creates figures
    prevents inline backend crashes in notebooks.
    """
    rc_params = getattr(matplotlib_module, "rcParams", None)
    if rc_params is not None and not hasattr(rc_params, "_get"):
        setattr(type(rc_params), "_get", lambda self, key: self.get(key))
