from types import SimpleNamespace

import pandas as pd

from src.xai_adapter.visualization import (
    _categorical_state_for_feature,
    plot_explanation_visual,
)


def _fake_data(value):
    dataset = SimpleNamespace(categorical_feature_options={0: [0]})
    return SimpleNamespace(
        df=pd.DataFrame({"Soil_Type_28": [value]}),
        dataset=dataset,
        raw_feature_names=["Soil_Type_28"],
        feature_names=["Soil_Type_28"],
    )


def test_one_hot_indicator_zero_renders_empty_circle():
    assert _categorical_state_for_feature(_fake_data(0), 0, "Soil_Type_28") == [True, False]


def test_one_hot_indicator_one_renders_filled_circle():
    assert _categorical_state_for_feature(_fake_data(1), 0, "Soil_Type_28") == [False, True]


def test_explanation_plot_can_hide_prediction_panel():
    data = SimpleNamespace(
        df=pd.DataFrame({"feature": [0.5]}),
        raw_feature_names=["feature"],
    )
    explanations = pd.DataFrame({
        "instanceId": [0],
        "expMethod": ["shap"],
        "pred": [1],
        "a0_i": [0.4],
    })

    figure, axes = plot_explanation_visual(
        explanations,
        data,
        visualization="importance",
        method="shap",
        instance_id=0,
        feature_names=["feature"],
        show_ai_prediction=False,
    )

    assert len(axes) == 4
    assert all(
        text.get_text() != "AI prediction"
        for axis in axes
        for text in axis.texts
    )
    figure.clf()
