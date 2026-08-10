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


__all__ = ["CATEGORICAL_DARK", "CATEGORICAL_LIGHT", "categorical_color"]
