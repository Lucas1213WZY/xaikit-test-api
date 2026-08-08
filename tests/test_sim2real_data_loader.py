from pathlib import Path

import numpy as np
import pytest

from src.data_loaders import Sim2RealDataSource, UnifiedDataLoader
from src.xai_adapter import create_xai_method


ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets"
DATASET_DIR = ASSETS_ROOT / "ai_dataset" / "sim2real"
EXPLANATIONS_DIR = ASSETS_ROOT / "explanations" / "xai_desiderata"


@pytest.fixture()
def loader():
    return UnifiedDataLoader.from_sim2real(
        dataset_dir=DATASET_DIR,
        explanations_dir=EXPLANATIONS_DIR,
    )


def test_sim2real_loader_reads_dataset_and_all_explanation_tables(loader):
    assert isinstance(loader.data_source, Sim2RealDataSource)
    assert loader.get_summary() == {
        "source_type": "sim2real",
        "n_instances": 39,
        "n_features": 67,
        "n_apps": 1,
        "app_ids": ["adult_sim2real"],
        "has_predictions": True,
        "n_explanation_columns": 0,
        "filters_applied": 0,
    }
    assert loader.list_apps() == ["adult_sim2real"]
    assert loader.list_explanation_tables() == [
        "attribution",
        "counterfactuals_fake",
        "deltas",
        "importance",
        "none",
    ]
    assert loader.get_predictions([0, 1, 2]) == [0, 0, 0]
    assert loader.get_ai_predictions()["modelName"].unique().tolist() == ["synthetic_ai"]
    assert len(loader.get_features([0], normalize=False)[0]) == 67
    assert loader.get_features([0], normalize=False)[0][:4] == [25.0, 0.0, 0.0, 40.0]
    assert loader.get_features([0], normalize=True)[0][0] == pytest.approx((25.0 - 17.0) / (72.0 - 17.0))

    attribution = loader.get_explanation_table("attribution")
    assert attribution.groupby("instanceId")["qid"].nunique().eq(1).all()
    assert attribution.loc[attribution["instanceId"] == 0, "qid"].unique().tolist() == [12]
    assert attribution["modelName"].unique().tolist() == [
        "synthetic_ai"
    ]
    assert attribution["expMethod"].unique().tolist() == ["lime"]
    assert attribution["expProperty"].fillna("baseline").value_counts().to_dict() == {
        "baseline": 39,
        "faithful": 39,
        "robust": 39,
        "sparse": 39,
        "sparse_robust": 39,
    }


def test_from_assets_resolves_sim2real_and_xai_desiderata():
    loader = UnifiedDataLoader.from_assets(
        "sim2real",
        assets_root=ASSETS_ROOT,
        app_id="adult_sim2real",
        model_name="synthetic_ai",
    )

    assert loader.get_summary()["n_instances"] == 39
    assert len(loader.get_explanation_table("importance")) == 195


def test_xai_adapter_loads_one_precomputed_explanation_variant(loader):
    method = create_xai_method(
        "precomputed_csv",
        loader=loader,
        explanation_type="attribution",
        exp_property="faithful",
        model_name="synthetic_ai",
    )

    result = method.explain([0, 1])
    records = method.get_records([0, 1])

    assert result.values.shape == (2, 67)
    assert result.values[0, :4].tolist() == pytest.approx([0.09, 0.4, 0.04, 0.05])
    assert result.base_values.tolist() == [0.0, 0.0]
    assert result.metadata["predictions"] == [0, 0]
    assert len(result.metadata["feature_columns"]) == 67
    assert records[0].features[:4].tolist() == [25.0, 0.0, 0.0, 40.0]
    assert records[0].metadata["expMethod"] == "lime"
    assert records[0].metadata["expProperty"] == "faithful"
    assert records[0].metadata["qid"] == 12


def test_xai_adapter_defaults_to_lime_baseline_for_sim2real(loader):
    method = create_xai_method(
        "csv",
        loader=loader,
        explanation_type="attribution",
        model_name="synthetic_ai",
    )

    result = method.explain([0])
    record = method.get_records([0])[0]

    assert result.values.shape == (1, 67)
    assert record.metadata["expMethod"] == "lime"
    assert np.isnan(record.metadata["expProperty"])


def test_xai_adapter_can_normalize_values_by_i_max(loader):
    method = create_xai_method(
        "csv",
        loader=loader,
        explanation_type="importance",
        exp_property="faithful",
        normalize_explanations_by="i_max",
    )

    result = method.explain([0])

    assert result.values[0, 0] == pytest.approx(0.09 / 8.08)


def test_xai_adapter_loads_counterfactual_and_no_xai_vectors(loader):
    counterfactual = create_xai_method(
        "csv",
        loader=loader,
        explanation_type="counterfactuals_fake",
        exp_property="faithful",
    )
    no_xai = create_xai_method(
        "csv",
        loader=loader,
        explanation_type="none",
    )

    assert counterfactual.explain([0]).values.shape == (1, 67)
    assert counterfactual.explain([0]).values[0, :4].tolist() == [25.0, 0.0, 0.0, 40.0]
    np.testing.assert_array_equal(no_xai.explain([0]).values, np.zeros((1, 67)))


def test_xai_adapter_validates_sim2real_property_and_method(loader):
    with pytest.raises(ValueError, match="exp_property must be one of"):
        create_xai_method(
            "csv",
            loader=loader,
            explanation_type="attribution",
            exp_property="not_a_property",
        )

    with pytest.raises(ValueError, match="use exp_method='lime'"):
        create_xai_method(
            "csv",
            loader=loader,
            explanation_type="attribution",
            exp_method="shap",
        )


def test_exp_property_is_rejected_for_non_sim2real_loader():
    coax_loader = UnifiedDataLoader.from_assets("coax", assets_root=ASSETS_ROOT)

    with pytest.raises(ValueError, match="only for Sim2Real"):
        create_xai_method(
            "csv",
            loader=coax_loader,
            explanation_type="attribution",
            exp_property="faithful",
        )


def test_xai_adapter_rejects_non_vector_sim2real_table(loader):
    with pytest.raises(ValueError, match="no numbered aN_i"):
        create_xai_method(
            "csv",
            loader=loader,
            explanation_type="deltas",
        )
