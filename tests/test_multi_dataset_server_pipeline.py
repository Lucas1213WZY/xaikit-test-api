"""Server-side wiring for a native multi-dataset (between-subjects ``dataset``
IV) study.

``xaikitTest``/``trials.py``/``counterbalance.py`` already route per dataset
end to end (see ``tests/test_multi_dataset_trials.py`` and
``tests/test_multi_dataset_api.py``). These tests cover the three server-side
gaps that sat on top of that: resolving which dataset id(s) a request/design
names, the apparatus instance-override applying per dataset instead of to a
single ``study.data``, and the simulation stage routing coax/coxam through
``study.run_experiment(...)`` (where the per-dataset loop lives) instead of
calling the runner functions directly. Every test also doubles as a
backward-compatibility check: the single-dataset path must produce the exact
same calls/payloads as before.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from server.pipeline import (
    _resolve_dataset_ids,
    run_dataset_stage,
    run_simulation_stage,
    run_trials_stage,
)
from server.schemas import DatasetStageRequest, SimulationRequest, TrialsStageRequest
from src.experiment_planner.design_export import DesignExport


def _prepared(dataset_id, *, train_ids, test_ids):
    return SimpleNamespace(
        dataset_id=dataset_id,
        feature_names=["a"],
        model_feature_names=["a"],
        y_train=[0] * len(list(train_ids)),
        y_test=[0] * len(list(test_ids)),
        split=SimpleNamespace(
            train_instance_ids=np.asarray(list(train_ids)),
            test_instance_ids=np.asarray(list(test_ids)),
        ),
    )


# ---------------------------------------------------------------------------
# _resolve_dataset_ids
# ---------------------------------------------------------------------------


def test_resolve_dataset_ids_explicit_scalar_wins():
    request = DatasetStageRequest(dataset_id="wine_quality")
    design = SimpleNamespace(dataset_ids=["mushrooms"])
    assert _resolve_dataset_ids(request, design) == ["wine_quality"]


def test_resolve_dataset_ids_explicit_list_wins():
    request = DatasetStageRequest(dataset_id=["wine_quality", "mushrooms"])
    design = SimpleNamespace(dataset_ids=["forest_cover"])
    assert _resolve_dataset_ids(request, design) == ["wine_quality", "mushrooms"]


def test_resolve_dataset_ids_falls_back_to_the_design():
    request = DatasetStageRequest()
    design = SimpleNamespace(dataset_ids=["wine_quality", "mushrooms"])
    assert _resolve_dataset_ids(request, design) == ["wine_quality", "mushrooms"]


# ---------------------------------------------------------------------------
# run_dataset_stage: multi-dataset
# ---------------------------------------------------------------------------


def test_run_dataset_stage_multi_dataset_corpus_covered_skips_training():
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoXAM",
    )
    fake_by_id = {
        "wine_quality": _prepared("wine_quality", train_ids=[1], test_ids=[2]),
        "mushrooms": _prepared("mushrooms", train_ids=[3], test_ids=[4]),
    }
    study.prepare_dataset.return_value = fake_by_id

    result = run_dataset_stage(
        study, DatasetStageRequest(dataset_id=["wine_quality", "mushrooms"])
    )

    study.prepare_dataset.assert_called_once()
    assert study.prepare_dataset.call_args.args[0] == ["wine_quality", "mushrooms"]
    study.train_AI_model.assert_not_called()
    assert result["model"] is None
    assert set(result["datasets"]) == {"wine_quality", "mushrooms"}
    assert result["datasets"]["wine_quality"]["dataset_id"] == "wine_quality"
    assert "corpus" in result["model_skipped_reason"]


def test_run_dataset_stage_multi_dataset_raises_for_an_uncovered_dataset():
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoXAM",
    )

    with pytest.raises(ValueError, match="prima_diabetes"):
        run_dataset_stage(
            study, DatasetStageRequest(dataset_id=["wine_quality", "prima_diabetes"])
        )
    study.prepare_dataset.assert_not_called()


def test_run_dataset_stage_multi_dataset_rejects_sim2real():
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="Sim2Real",
    )

    with pytest.raises(ValueError, match="Sim2Real"):
        run_dataset_stage(
            study, DatasetStageRequest(dataset_id=["wine_quality", "mushrooms"])
        )
    study.prepare_dataset.assert_not_called()


def test_run_dataset_stage_single_dataset_is_unchanged():
    """Backward compatibility: the scalar path must call prepare_dataset with
    a bare string and build the old {"dataset": {...}} payload shape."""
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoXAM", dataset_id="wine_quality",
    )
    study.prepare_dataset.return_value = _prepared("wine_quality", train_ids=[1], test_ids=[2])

    result = run_dataset_stage(study, DatasetStageRequest())

    assert study.prepare_dataset.call_args.args[0] == "wine_quality"
    assert "datasets" not in result
    assert result["dataset"]["dataset_id"] == "wine_quality"


# ---------------------------------------------------------------------------
# run_trials_stage: apparatus override applies per dataset
# ---------------------------------------------------------------------------


def test_apparatus_override_applies_independently_per_dataset():
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN",
        participants_per_condition=1, trials_per_participant=8,
        apparatus_instance_ids=[1, 2, 10, 11],
        apparatus_instance_ids_by_dataset={"wine_quality": [1, 2], "mushrooms": [10, 11]},
        apparatus_training_instance_ids_by_dataset={"wine_quality": [], "mushrooms": [90]},
    )
    study.data_by_dataset = {
        "wine_quality": _prepared("wine_quality", train_ids=[1, 100, 101], test_ids=[2, 200, 201]),
        "mushrooms": _prepared("mushrooms", train_ids=[10, 300], test_ids=[11, 400]),
    }
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest(num_training=1, num_testing=2, seed=0))
    except Exception:
        pass  # only the split override is under test here

    wine = study.data_by_dataset["wine_quality"]
    mushrooms = study.data_by_dataset["mushrooms"]

    assert sorted(wine.split.test_instance_ids.tolist()) == [1, 2]
    assert sorted(mushrooms.split.test_instance_ids.tolist()) == [10, 11]

    # wine_quality declared no trainingInstanceIds, so one was sampled from its
    # own remaining pool ({100, 101, 200, 201} -- everything outside its
    # apparatus-declared test ids) -- never one of mushrooms' instance ids.
    assert wine.split.train_instance_ids.tolist()[0] in {100, 101, 200, 201}
    # mushrooms declared trainingInstanceIds=[90] directly.
    assert mushrooms.split.train_instance_ids.tolist() == [90]


def test_apparatus_override_raises_naming_the_dataset_that_ran_short():
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN",
        participants_per_condition=1, trials_per_participant=8,
        apparatus_instance_ids=[1, 2],
        apparatus_instance_ids_by_dataset={"wine_quality": [1, 2]},
        apparatus_training_instance_ids_by_dataset={},
    )
    # Only one instance left outside the declared testing ids -- not enough
    # for num_training=5.
    study.data_by_dataset = {
        "wine_quality": _prepared("wine_quality", train_ids=[1, 3], test_ids=[2]),
    }

    with pytest.raises(ValueError, match="wine_quality"):
        run_trials_stage(study, TrialsStageRequest(num_training=5, num_testing=2, seed=0))


def test_apparatus_override_single_dataset_study_is_unchanged():
    """Backward compatibility: a study with no data_by_dataset (a MagicMock's
    auto-vivified attribute must not be mistaken for one) still overrides
    study.data.split directly, exactly as before."""
    study = MagicMock()
    study.data_by_dataset = {}
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN",
        participants_per_condition=1, trials_per_participant=8,
        apparatus_instance_ids=[1, 2, 3, 4, 5],
        apparatus_training_instance_ids=[],
    )
    study.data.split.train_instance_ids = np.array([1, 100, 101, 102])
    study.data.split.test_instance_ids = np.array([2, 200, 201, 202])
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest(num_training=2, num_testing=5, seed=0))
    except Exception:
        pass  # only the split override is under test here

    assert sorted(study.data.split.test_instance_ids.tolist()) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# run_simulation_stage: coax/coxam dispatch through study.run_experiment
# ---------------------------------------------------------------------------


def test_coax_simulation_routes_through_run_experiment_with_cognitive_model():
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX",
    )
    study.trained_ai_model = None
    study.trials = [{"xai_type": "none", "tested_w_xai": True, "participantId": 1}]
    study.save_results.return_value = ("x.csv", "x.json")
    study.run_experiment.return_value = MagicMock(empty=False)

    run_simulation_stage(study, SimulationRequest(), output_subdir="x")

    kwargs = study.run_experiment.call_args.kwargs
    assert kwargs["source"] == "corpus"
    assert "cognitive_model" in kwargs
    assert "none" in kwargs["cognitive_model"]
