import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.api import xaikitTest
from src.result_visualizer import plot_dv_by_two_ivs, plot_iv_dv_grid


def _responses():
    return pd.DataFrame({
        "participantId": [1, 1, 2, 2, 1, 2],
        "phase": ["testing", "testing", "testing", "testing", "training", "training"],
        "method": ["shap", "lime", "shap", "lime", "shap", "shap"],
        "tested_w_xai": [True, False, True, False, None, None],
        "accuracy": [1.0, 0.0, 0.5, 1.0, 99.0, 99.0],
        "confidence": [0.8, 0.4, 0.7, 0.9, 99.0, 99.0],
    })


def test_plot_iv_dv_grid_builds_every_pair_from_testing_results():
    result = plot_iv_dv_grid(
        _responses(),
        ivs=["method", "tested_w_xai"],
        dvs=["accuracy", "confidence"],
        iv_levels={
            "method": ["shap", "lime"],
            "tested_w_xai": [True, False],
        },
    )

    assert result.axes.shape == (2, 2)
    assert set(zip(result.summary["iv"], result.summary["dv"])) == {
        ("method", "accuracy"),
        ("tested_w_xai", "accuracy"),
        ("method", "confidence"),
        ("tested_w_xai", "confidence"),
    }
    shap_accuracy = result.summary.query(
        "iv == 'method' and dv == 'accuracy' and level == 'shap'"
    )
    assert shap_accuracy["mean"].iloc[0] == pytest.approx(0.75)
    assert result.summary["mean"].max() < 2


def test_study_plot_results_grid_uses_configured_ivs_and_dvs():
    study = xaikitTest("visualizer")
    study.add_iv("method", "within", ["shap", "lime"])
    study.add_iv(
        "tested_w_xai",
        "within",
        [True, False],
        randomization="trial",
    )
    study.add_dv("accuracy", ["continuous"])
    study.simulated_results = _responses()

    result = study.plot_results_grid(responses=study.simulated_results)

    assert result.axes.shape == (1, 2)


def _interaction_responses():
    rows = []
    for participant_id in [1, 2]:
        for tested_w_xai in [True, False]:
            for xai_type, accuracy in [
                ("none", 0.5),
                ("importance", 0.7),
                ("attribution", 0.9),
            ]:
                rows.append({
                    "participantId": participant_id,
                    "phase": "testing",
                    "tested_w_xai": tested_w_xai,
                    "xai_type": xai_type,
                    "forward_accuracy": accuracy - (0.1 if not tested_w_xai else 0),
                })
    return pd.DataFrame(rows)


def test_plot_dv_by_two_ivs_orders_presence_and_groups_explanation_types():
    result = plot_dv_by_two_ivs(
        _interaction_responses(),
        x_iv="tested_w_xai",
        hue_iv="xai_type",
        dv="forward_accuracy",
        x_levels=[True, False],
        hue_levels=["none", "importance", "attribution"],
        x_labels={True: "With explanation", False: "Without explanation"},
    )

    assert [label.get_text() for label in result.axis.get_xticklabels()] == [
        "With explanation",
        "Without explanation",
    ]
    assert len(result.axis.patches) == 6
    assert set(result.summary["hue_level"]) == {
        "none",
        "importance",
        "attribution",
    }


def test_study_plot_dv_by_two_ivs_uses_the_public_wrapper():
    study = xaikitTest("interaction_visualizer")
    study.add_iv("tested_w_xai", "within", [True, False])
    study.add_iv(
        "xai_type",
        "between",
        ["none", "importance", "attribution"],
    )
    study.add_dv("forward_accuracy", ["continuous"])
    study.simulated_results = _interaction_responses()

    result = study.plot_dv_by_two_ivs(
        x_iv="tested_w_xai",
        hue_iv="xai_type",
        dv="forward_accuracy",
        x_labels={True: "With explanation", False: "Without explanation"},
    )

    assert len(result.summary) == 6
