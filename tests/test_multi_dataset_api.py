"""`xaikitTest`'s native multi-dataset entry point.

`prepare_dataset(dataset_id=[...])` is the single knob: it builds
`self.data_by_dataset` and auto-registers a `dataset` between-subjects IV with
those levels, so `generate_trials()`/`run_experiment()` don't need a second,
separately-configurable signal to detect multi-dataset mode. Simulation stays
a loop over the *unmodified* per-agent runner (one call per dataset level,
concatenated) rather than making the runners themselves dataset-aware --
these tests spy on the runner the same way `test_run_experiment_dispatch.py`
does, instead of running real simulations.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.api import xaikitTest
from src.experiment_planner import DATASET_IV_NAME


# ---------------------------------------------------------------------------
# prepare_dataset(dataset_id=[...])
# ---------------------------------------------------------------------------


def test_prepare_dataset_with_a_list_builds_the_registry_and_between_iv(monkeypatch):
    fake_by_id = {
        "wine_quality": SimpleNamespace(dataset_id="wine_quality"),
        "mushrooms": SimpleNamespace(dataset_id="mushrooms"),
    }
    monkeypatch.setattr(
        "src.api.prepare_dataset", lambda dataset_id, **kwargs: fake_by_id[dataset_id]
    )
    study = xaikitTest()

    result = study.prepare_dataset(
        ["wine_quality", "mushrooms"], show_available=False, show_summary=False
    )

    assert result == fake_by_id
    assert study.data_by_dataset == fake_by_id
    assert study.data is None
    assert study.iv_config[DATASET_IV_NAME] == {
        "type": "between",
        "levels": ["wine_quality", "mushrooms"],
    }


def test_prepare_dataset_with_a_single_id_still_populates_self_data(monkeypatch):
    fake = SimpleNamespace(dataset_id="wine_quality")
    monkeypatch.setattr("src.api.prepare_dataset", lambda dataset_id, **kwargs: fake)
    study = xaikitTest()
    study.data_by_dataset = {"stale": object()}  # leftover from a prior list call

    result = study.prepare_dataset("wine_quality", show_available=False, show_summary=False)

    assert result is fake
    assert study.data is fake
    assert study.data_by_dataset == {}, "a single-id call must clear any stale registry"


def test_prepare_dataset_rejects_a_single_element_list():
    study = xaikitTest()
    with pytest.raises(ValueError, match="at least two"):
        study.prepare_dataset(["wine_quality"])


# ---------------------------------------------------------------------------
# generate_trials(): auto-detection
# ---------------------------------------------------------------------------


def _fake_prepared(dataset_id, *, train_ids, test_ids):
    return SimpleNamespace(
        dataset_id=dataset_id,
        train_instance_ids=list(train_ids),
        test_instance_ids=list(test_ids),
    )


def test_generate_trials_routes_through_data_by_dataset_when_set():
    study = xaikitTest()
    study.add_iv("xai_type", "within", ["decision_tree"], randomization="block")
    study.data_by_dataset = {
        "wine_quality": _fake_prepared("wine_quality", train_ids=range(5), test_ids=range(5, 10)),
        "mushrooms": _fake_prepared("mushrooms", train_ids=range(100, 105), test_ids=range(105, 110)),
    }
    study.add_iv(DATASET_IV_NAME, "between", ["wine_quality", "mushrooms"])

    study.generate_trials(participants_per_between_condition=2, num_testing=3, show=False)

    assert {trial["dataId"] for trial in study.trials} == {"wine_quality", "mushrooms"}
    levels_by_participant = {}
    for trial in study.trials:
        levels_by_participant.setdefault(trial["participantId"], set()).add(trial["dataId"])
    assert all(len(levels) == 1 for levels in levels_by_participant.values())


def test_generate_trials_rejects_balance_by_ai_prediction_for_multi_dataset():
    study = xaikitTest()
    study.add_iv("xai_type", "within", ["decision_tree"], randomization="block")
    study.data_by_dataset = {
        "wine_quality": _fake_prepared("wine_quality", train_ids=range(5), test_ids=range(5, 10)),
        "mushrooms": _fake_prepared("mushrooms", train_ids=range(100, 105), test_ids=range(105, 110)),
    }
    study.add_iv(DATASET_IV_NAME, "between", ["wine_quality", "mushrooms"])

    with pytest.raises(ValueError, match="not yet supported"):
        study.generate_trials(balance_by_ai_prediction=True, show=False)


# ---------------------------------------------------------------------------
# run_experiment(): per-dataset simulation loop
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_dataset_study():
    study = xaikitTest()
    study.data_by_dataset = {
        "wine_quality": SimpleNamespace(dataset_id="wine_quality"),
        "mushrooms": SimpleNamespace(dataset_id="mushrooms"),
    }
    study.trials = [
        {"participantId": 1, "dataId": "wine_quality", "instanceId": "1"},
        {"participantId": 2, "dataId": "mushrooms", "instanceId": "2"},
    ]
    study.set_cognitive_model(cognitive_model_id="coxam")
    return study


@pytest.fixture
def spy_coxam(monkeypatch):
    calls = []

    def fake_runner(study_arg, *, mode, participant_id, condition_filter, **kwargs):
        from src.experiment_planner import select_trial_rows

        selected = select_trial_rows(
            pd.DataFrame(study_arg.trials), mode,
            participant_id=participant_id, condition_filter=condition_filter,
        )
        calls.append({
            "dataset_id": study_arg.data.dataset_id,
            "trials": list(study_arg.trials),
        })
        return selected.assign(forward_accuracy=1.0) if not selected.empty else selected

    monkeypatch.setattr(
        "src.virtual_experiment_executor.experiment_simualtion.CoXAM."
        "coxam_study_runner.run_coxam_study",
        fake_runner,
    )
    return calls


def test_run_experiment_loops_each_dataset_and_concatenates(multi_dataset_study, spy_coxam):
    original_trials = list(multi_dataset_study.trials)

    result = multi_dataset_study.run_experiment(mode="whole_experiment")

    assert len(spy_coxam) == 2
    assert {call["dataset_id"] for call in spy_coxam} == {"wine_quality", "mushrooms"}
    # Each call only ever saw its own level's trial rows -- never both.
    for call in spy_coxam:
        assert {trial["dataId"] for trial in call["trials"]} == {call["dataset_id"]}

    assert set(result["dataset"]) == {"wine_quality", "mushrooms"}
    assert len(result) == 2

    # Study state is restored to the multi-dataset view after the loop.
    assert multi_dataset_study.data is None
    assert multi_dataset_study.trials == original_trials


def test_run_experiment_participant_by_participant_only_includes_owning_level(
    multi_dataset_study, spy_coxam
):
    result = multi_dataset_study.run_experiment(
        mode="participant_by_participant", participant_id=2
    )

    assert set(result["dataset"]) == {"mushrooms"}
    assert len(result) == 1


def test_run_experiment_raises_if_a_dataset_has_no_trial_rows(multi_dataset_study, spy_coxam):
    multi_dataset_study.trials = [
        {"participantId": 1, "dataId": "wine_quality", "instanceId": "1"},
    ]
    with pytest.raises(RuntimeError, match="mushrooms"):
        multi_dataset_study.run_experiment(mode="whole_experiment")


def test_run_experiment_multi_dataset_requires_an_agent_runner(multi_dataset_study, spy_coxam):
    multi_dataset_study.cognitive_model_id = "placeholder"
    with pytest.raises(ValueError, match="research-agent"):
        multi_dataset_study.run_experiment(mode="whole_experiment")
