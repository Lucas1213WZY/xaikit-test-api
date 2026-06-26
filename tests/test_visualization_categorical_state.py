from types import SimpleNamespace

import pandas as pd

from src.xai_adapter.visualization import _categorical_state_for_feature


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
