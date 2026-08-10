"""Tests for the human-vs-cognitive-model comparison panels and report."""

from __future__ import annotations

import pandas as pd
import pytest

from src.api import xaikitTest
from src.result_visualizer import (
    ComparisonPanel,
    ComparisonStudy,
    comparison_panel,
    participant_summary,
    render_comparison_report,
)


def _lopsided() -> pd.DataFrame:
    """One participant with ten hits, another with a single miss.

    Mean over trials is 10/11 = 0.909; mean over participant means is 0.5. Any
    test using this frame distinguishes the two estimators.
    """
    return pd.DataFrame(
        {
            "participantId": ["a"] * 10 + ["b"],
            "condition": ["x"] * 11,
            "hit": [1.0] * 10 + [0.0],
        }
    )


# -- the estimator --------------------------------------------------------


def test_summary_averages_participants_not_trials():
    summary = participant_summary(
        _lopsided(), participant_column="participantId", group_column="condition", value_column="hit"
    )
    assert summary.loc[0, "mean"] == pytest.approx(0.5)
    assert summary.loc[0, "n"] == 2  # participants, not the 11 trials


def test_summary_sem_and_ci_are_over_participant_means():
    frame = pd.DataFrame(
        {
            "participantId": ["a", "b", "c", "d"],
            "condition": ["x"] * 4,
            "hit": [0.0, 0.0, 1.0, 1.0],
        }
    )
    summary = participant_summary(
        frame, participant_column="participantId", group_column="condition", value_column="hit"
    )
    # sd of [0,0,1,1] with ddof=1 is 0.57735; sem = sd/sqrt(4)
    assert summary.loc[0, "mean"] == pytest.approx(0.5)
    assert summary.loc[0, "sem"] == pytest.approx(0.57735 / 2, rel=1e-3)
    # 95% CI uses the t multiplier for n=4 (df=3, t=3.182), not the normal 1.96
    assert summary.loc[0, "ci95"] == pytest.approx(summary.loc[0, "sem"] * 3.1824, rel=1e-3)


def test_a_single_participant_group_reports_zero_not_nan():
    """NaN would drop the bar; a lone observation should flatten the whisker."""
    frame = pd.DataFrame({"participantId": ["a"], "condition": ["x"], "hit": [1.0]})
    summary = participant_summary(
        frame, participant_column="participantId", group_column="condition", value_column="hit"
    )
    assert summary.loc[0, "sem"] == 0.0
    assert summary.loc[0, "ci95"] == 0.0
    assert summary.loc[0, "n"] == 1


def test_missing_values_are_dropped_not_counted_as_zero():
    frame = pd.DataFrame(
        {
            "participantId": ["a", "b"],
            "condition": ["x", "x"],
            "hit": [1.0, None],
        }
    )
    summary = participant_summary(
        frame, participant_column="participantId", group_column="condition", value_column="hit"
    )
    assert summary.loc[0, "mean"] == pytest.approx(1.0)
    assert summary.loc[0, "n"] == 1


def test_a_missing_column_is_named_in_the_error():
    with pytest.raises(KeyError, match="nope"):
        participant_summary(
            _lopsided(),
            participant_column="participantId",
            group_column="condition",
            value_column="nope",
        )


# -- panels ---------------------------------------------------------------


def _two_series() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participantId": ["a", "a", "b", "b"],
            "condition": ["x", "y", "x", "y"],
            "human": [1.0, 0.0, 1.0, 0.0],
            "model": [1.0, 1.0, 0.0, 0.0],
        }
    )


def test_panel_keeps_series_order_so_colours_are_stable():
    panel = comparison_panel(
        _two_series(),
        participant_column="participantId",
        group_column="condition",
        series={"Human": "human", "CoXAM": "model"},
        title="t",
        dv="d",
    )
    assert [entry["name"] for entry in panel.series] == ["Human", "CoXAM"]
    assert panel.categories == ["x", "y"]
    assert panel.series[0]["values"] == [pytest.approx(1.0), pytest.approx(0.0)]
    assert panel.series[1]["values"] == [pytest.approx(0.5), pytest.approx(0.5)]


def test_panel_honours_explicit_order_and_labels():
    panel = comparison_panel(
        _two_series(),
        participant_column="participantId",
        group_column="condition",
        series={"Human": "human"},
        title="t",
        dv="d",
        order=["y", "x"],
        group_labels={"x": "First", "y": "Second"},
    )
    assert panel.categories == ["Second", "First"]
    assert panel.series[0]["values"] == [pytest.approx(0.0), pytest.approx(1.0)]


def test_a_category_absent_from_a_series_becomes_none_not_zero():
    """A gap must render as a missing bar, never as a real zero."""
    frame = _two_series().copy()
    frame.loc[frame["condition"] == "y", "model"] = None
    panel = comparison_panel(
        frame,
        participant_column="participantId",
        group_column="condition",
        series={"Human": "human", "Model": "model"},
        title="t",
        dv="d",
    )
    assert panel.series[1]["values"][panel.categories.index("y")] is None


def test_panel_to_frame_round_trips_the_numbers():
    panel = comparison_panel(
        _two_series(),
        participant_column="participantId",
        group_column="condition",
        series={"Human": "human", "Model": "model"},
        title="t",
        dv="d",
    )
    frame = panel.to_frame()
    assert len(frame) == 4
    assert set(frame["series"]) == {"Human", "Model"}
    assert set(frame["interval"]) == {"95% CI"}
    assert frame.loc[frame["series"].eq("Human") & frame["category"].eq("x"), "mean"].iloc[0] == 1.0


# -- the report -----------------------------------------------------------


def _panel() -> ComparisonPanel:
    return comparison_panel(
        _two_series(),
        participant_column="participantId",
        group_column="condition",
        series={"Human": "human", "Model": "model"},
        title="Panel title",
        dv="Some measure",
    )


def test_report_is_self_contained(tmp_path):
    """A UI serves this file directly, so it must not fetch anything."""
    path = render_comparison_report(
        [ComparisonStudy(name="S", panels=[_panel()])], tmp_path / "r.html"
    )
    html = path.read_text()
    # The SVG XML namespace is a URI, not a fetch, so look for actual resource
    # loads and network calls rather than for the string "http".
    for forbidden in (
        'src="http',
        "src='http",
        'href="http',
        "href='http",
        'src="//',
        "@import",
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
    ):
        assert forbidden not in html, forbidden
    assert "Panel title" in html


def test_report_escapes_a_closing_tag_inside_the_data(tmp_path):
    """An unescaped </script> in a label would terminate the data block early."""
    panel = _panel()
    panel.title = "danger </script><b>x"
    path = render_comparison_report(
        [ComparisonStudy(name="S", panels=[panel])], tmp_path / "r.html"
    )
    html = path.read_text()
    assert "</script><b>x" not in html
    assert "<\\/script>" in html


def test_report_rejects_an_empty_study_list(tmp_path):
    with pytest.raises(ValueError, match="at least one study"):
        render_comparison_report([], tmp_path / "r.html")


def test_report_title_and_footer_are_substituted(tmp_path):
    path = render_comparison_report(
        [ComparisonStudy(name="S", panels=[_panel()])],
        tmp_path / "r.html",
        title="My title",
        footer="My footer",
    )
    html = path.read_text()
    assert "<title>My title</title>" in html
    assert "My footer" in html
    assert "__TITLE__" not in html and "__LEDE__" not in html and "__FOOTER__" not in html


# -- reachable from the API ----------------------------------------------


def test_compare_to_human_data_on_an_aligned_frame():
    study = xaikitTest("compare")
    panel = study.compare_to_human_data(
        _two_series(),
        group_column="condition",
        human_column="human",
        model_column="model",
        model_name="CoXAM",
    )
    assert [entry["name"] for entry in panel.series] == ["Human", "CoXAM"]
    assert panel.categories == ["x", "y"]


def test_compare_to_human_data_labels_the_series_from_the_selected_model():
    study = xaikitTest("compare")
    study.cognitive_model_id = "sim2real"
    panel = study.compare_to_human_data(
        _two_series(), group_column="condition", human_column="human", model_column="model"
    )
    assert panel.series[1]["name"] == "sim2real"


def test_compare_to_human_data_joins_the_stored_simulation():
    study = xaikitTest("compare")
    study.simulated_results = pd.DataFrame(
        {"participantId": ["a", "b"], "trialId": [1, 1], "model": [1.0, 0.0]}
    )
    human = pd.DataFrame(
        {"participantId": ["a", "b"], "trialId": [1, 1], "condition": ["x", "x"], "human": [1.0, 1.0]}
    )
    panel = study.compare_to_human_data(
        human,
        group_column="condition",
        human_column="human",
        model_column="model",
        on=["participantId", "trialId"],
        model_name="M",
    )
    assert panel.series[0]["values"] == [pytest.approx(1.0)]
    assert panel.series[1]["values"] == [pytest.approx(0.5)]


def test_joining_without_a_simulation_is_refused():
    study = xaikitTest("compare")
    with pytest.raises(RuntimeError, match="run_experiment"):
        study.compare_to_human_data(
            _two_series(),
            group_column="condition",
            human_column="human",
            model_column="model",
            on=["participantId"],
        )


def test_a_join_that_matches_nothing_raises_rather_than_plotting_empty():
    study = xaikitTest("compare")
    study.simulated_results = pd.DataFrame({"participantId": ["z"], "model": [1.0]})
    with pytest.raises(ValueError, match="matched no rows"):
        study.compare_to_human_data(
            _two_series(),
            group_column="condition",
            human_column="human",
            model_column="model",
            on=["participantId"],
        )


def test_render_human_vs_model_report_accepts_one_panel(tmp_path):
    study = xaikitTest("compare")
    study.output_dir = tmp_path
    panel = study.compare_to_human_data(
        _two_series(),
        group_column="condition",
        human_column="human",
        model_column="model",
        model_name="M",
    )
    path = study.render_human_vs_model_report(panel, "out.html", participants=2)
    assert path == tmp_path / "out.html"
    assert "Panel" in path.read_text() or "Human vs M" in path.read_text()


# -- the error-bar convention --------------------------------------------


def test_panels_default_to_a_95_percent_ci():
    panel = comparison_panel(
        _two_series(),
        participant_column="participantId",
        group_column="condition",
        series={"Human": "human"},
        title="t",
        dv="d",
    )
    assert panel.interval == "95% CI"


def test_asking_for_sem_gives_the_narrower_bar():
    """A CI is the SEM scaled by t, so it must never be the smaller of the two."""
    kwargs = dict(
        participant_column="participantId",
        group_column="condition",
        series={"Model": "model"},
        title="t",
        dv="d",
    )
    ci = comparison_panel(_two_series(), interval="ci95", **kwargs)
    sem = comparison_panel(_two_series(), interval="sem", **kwargs)
    assert ci.interval == "95% CI" and sem.interval == "SEM"
    assert ci.series[0]["error"][0] > sem.series[0]["error"][0]


def test_an_unknown_interval_is_rejected():
    with pytest.raises(ValueError, match="interval must be one of"):
        comparison_panel(
            _two_series(),
            participant_column="participantId",
            group_column="condition",
            series={"Human": "human"},
            title="t",
            dv="d",
            interval="stderr",
        )


def test_the_ci_multiplier_uses_t_not_the_normal_approximation():
    from src.result_visualizer.intervals import ci95_multiplier

    assert ci95_multiplier(11) == pytest.approx(2.2281, rel=1e-3)  # df=10
    assert ci95_multiplier(62) == pytest.approx(1.9996, rel=1e-3)  # df=61
    assert ci95_multiplier(1) == 0.0  # no spread from one observation


# -- the one-call study API ----------------------------------------------


def test_study_comparison_rejects_an_unknown_study():
    from src.result_visualizer import study_comparison

    with pytest.raises(ValueError, match="Unknown study"):
        study_comparison("not_a_study")


def test_available_studies_is_a_subset_of_the_known_names():
    from src.result_visualizer import STUDY_NAMES, available_studies

    assert set(available_studies()) <= set(STUDY_NAMES)


def test_coxam_loader_rejects_an_unknown_task():
    from src.result_visualizer import load_coxam_fitted_trials

    with pytest.raises(ValueError, match="forward.*counterfactual"):
        load_coxam_fitted_trials("sideways")


def test_coax_panels_are_built_from_a_replay_not_a_published_table():
    """The replay keeps the participant on the model rows, so both series get
    an interval; the published table sets Participant ID to NaN for the model."""
    from src.result_visualizer.study_comparisons import _coax

    replay = pd.DataFrame({
        "participantId": ["a", "a", "b", "b"],
        "dataId": ["adult"] * 4,
        "xai_type": ["none", "none", "none", "none"],
        "tested_w_xai": [True, False, True, False],
        "human_comparable": [True] * 4,
        "human_correct": [1.0, 0.0, 1.0, 1.0],
        "cognitive_correct_vs_ai": [1.0, 1.0, 0.0, 1.0],
    })
    study = _coax(replay=replay)
    assert study is not None and study.participants == 2
    overall = study.panels[0].to_frame()
    assert set(overall["series"]) == {"Human", "CoAX"}
    # Both series have n = participants, so both get a real interval.
    assert set(overall["n"]) == {2}
    assert (overall["error"] > 0).all()


def test_a_replay_with_no_comparable_rows_yields_no_study():
    from src.result_visualizer.study_comparisons import _coax

    replay = pd.DataFrame({
        "participantId": ["a"], "dataId": ["adult"], "xai_type": ["none"],
        "tested_w_xai": [True], "human_comparable": [False],
        "human_correct": [1.0], "cognitive_correct_vs_ai": [1.0],
    })
    assert _coax(replay=replay) is None
