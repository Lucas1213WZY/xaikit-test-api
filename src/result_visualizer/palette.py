"""The one categorical palette every chart in this package draws from.

Eight hues, validated for colorblind separation (OKLab CVD delta-E >= 8 on
every adjacent pair, both a light and a dark surface) via the dataviz skill's
validator script. The same eight hex values back the interactive HTML report
in ``human_vs_model.py``, so a condition is the same color whether it appears
in a matplotlib grid or the browser report -- **the order is the safety
mechanism, not decoration**: never reorder or insert without re-running the
validator.
"""

from __future__ import annotations

from typing import Any, Optional

#: Light-surface steps, in validated order.
CATEGORICAL_LIGHT: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: The same eight hues, stepped for a dark surface.
CATEGORICAL_DARK: tuple[str, ...] = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)


def categorical_color(index: int, *, dark: bool = False) -> str:
    """The palette step for the ``index``-th series, cycling past 8.

    Args:
        index: Zero-based series position -- the same index must be used for
            the same condition across every plot for the shared scheme to
            hold.
        dark: Use the dark-surface steps instead of the light ones.
    """
    palette = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    return palette[index % len(palette)]




# -- semantic colors: identity, not position ------------------------------
#
# ``categorical_color`` assigns by *position*, so a level's color depends on
# how many levels there are and how they sort. That is the right default for
# an arbitrary factor, but wrong for a factor whose levels carry meaning:
# ``tested_w_xai=False`` sorts first and so took the blue that belongs to the
# explanation condition, while ``True`` -- the condition the study exists to
# test -- took the orange. Filtering to one level, or adding a third, would
# repaint them again.
#
# These factors get a color keyed on the *value*, so a condition is the same
# color in every panel, every report, and every run, whatever else is present.

#: Blue reads as "the explanation was there", red as "it was not". Both are
#: steps from CATEGORICAL_LIGHT/DARK -- blue index 0, red index 7 -- and the
#: pair was validated on its own (dataviz validator, light surface): protan
#: delta-E 21.6, tritan 34.5, normal 32.3, all well clear of the >= 8 target
#: and the >= 15 normal-vision floor.
_TESTED_WITH_XAI = {
    True: (CATEGORICAL_LIGHT[0], CATEGORICAL_DARK[0]),   # blue  -- with XAI
    False: (CATEGORICAL_LIGHT[7], CATEGORICAL_DARK[7]),  # red   -- without XAI
}

#: Columns whose levels mean something, and the value -> color map for each.
SEMANTIC_COLORS: dict[str, dict[Any, tuple[str, str]]] = {
    "tested_w_xai": _TESTED_WITH_XAI,
    "tested_with_xai": _TESTED_WITH_XAI,
    "Tested w/ XAI": _TESTED_WITH_XAI,
}


def _as_bool(value: Any) -> Optional[bool]:
    """Read the many spellings one boolean condition arrives in.

    A level reaches a plot as a real bool from the trial table, as the string
    ``"True"`` after a CSV round trip, or as ``"w/ XAI"`` from the published
    corpora -- the same condition either way, so it must get the same color.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "w/ xai", "with xai", "with_xai"}:
        return True
    if text in {"false", "0", "no", "n", "w/o xai", "without xai", "without_xai"}:
        return False
    return None


def semantic_color(column: Any, value: Any, *, dark: bool = False) -> Optional[str]:
    """The color a meaningful level always takes, or None if it has no meaning.

    Args:
        column: The factor the level belongs to, e.g. ``tested_w_xai``.
        value: The level itself, in any of its spellings.
        dark: Use the dark-surface step.

    Returns:
        A hex color, or None when this column is not semantic -- in which case
        the caller should fall back to ``categorical_color`` by position.
    """
    mapping = SEMANTIC_COLORS.get(str(column).strip())
    if mapping is None:
        return None
    key = _as_bool(value)
    if key is None or key not in mapping:
        return None
    return mapping[key][1 if dark else 0]


def level_color(
    column: Any, value: Any, index: int, *, dark: bool = False
) -> str:
    """A level's color: its meaning if it has one, otherwise its position.

    This is what plots should call. It keeps arbitrary factors behaving exactly
    as before while pinning the ones whose levels carry meaning.
    """
    explicit = semantic_color(column, value, dark=dark)
    return explicit if explicit is not None else categorical_color(index, dark=dark)


__all__ = [
    "CATEGORICAL_DARK",
    "CATEGORICAL_LIGHT",
    "SEMANTIC_COLORS",
    "categorical_color",
    "level_color",
    "semantic_color",
]
