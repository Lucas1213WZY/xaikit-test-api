"""Result visualization API."""

from .grid import (
    InteractionPlot,
    ResultGrid,
    plot_dv_by_two_ivs,
    plot_iv_dv_grid,
)
from .labels import (
    display_labels,
    display_order,
    level_display,
    pretty,
    prettify_condition_label,
)
from .study_comparisons import (
    STUDY_NAMES,
    available_studies,
    human_vs_model_report,
    load_coxam_fitted_trials,
    load_sim2real_fitted_trials,
    study_comparison,
)
from .human_vs_model import (
    ComparisonPanel,
    ComparisonStudy,
    comparison_panel,
    participant_summary,
    render_comparison_report,
)
from .panel_grid import render_comparison_figure, render_panel_grid_png

__all__ = [
    "STUDY_NAMES",
    "available_studies",
    "human_vs_model_report",
    "load_coxam_fitted_trials",
    "load_sim2real_fitted_trials",
    "study_comparison",
    "ComparisonPanel",
    "ComparisonStudy",
    "InteractionPlot",
    "ResultGrid",
    "comparison_panel",
    "display_labels",
    "display_order",
    "level_display",
    "participant_summary",
    "plot_dv_by_two_ivs",
    "plot_iv_dv_grid",
    "pretty",
    "prettify_condition_label",
    "render_comparison_report",
    "render_comparison_figure",
    "render_panel_grid_png",
]
