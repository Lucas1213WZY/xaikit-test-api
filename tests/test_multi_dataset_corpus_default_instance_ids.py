"""When nothing (no apparatus instanceIds at all) constrains a multi-dataset
study's trials, each corpus-covered dataset must default to its *own*
published-corpus instance ids -- never a flat union across datasets.

Regression: a real CoAX export declared ``dataset`` as a between-subjects IV
with three levels (adult, wine_quality, forest_cover) and a single apparatus
configuration with empty ``params`` -- no ``instanceIds``, no
``trainingInstanceIds``, no ``appId``. With no restriction at all, trials
sampled freely from each dataset's own random 80/20 split and crashed at
simulation time the first time a trial referenced an instance CoAX's
published corpus never shipped ("Instance N not found for dataId=... in CoAX
features"). The first fix attempt built a single flat ``allowed_instance_ids``
list as the union of every corpus-covered dataset's ids -- which is unsound:
instance ids are dataset-local (each dataset numbers its own instances from
0), so unioning e.g. adult's 300 ids with wine_quality's 122 ids lets ids
122-299 leak into wine_quality's pool, since they're valid ids for *adult*
but happen to also be plain integers wine_quality's own random split can
produce. The real fix filters each dataset's own split in place against its
own corpus ids.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from server.pipeline import run_trials_stage
from server.schemas import TrialsStageRequest
from src.experiment_planner.design_export import DesignExport

EXPORT_DIR = Path(__file__).resolve().parents[1] / "tutorials" / "experiment_output"
NO_APPARATUS_EXPORT = EXPORT_DIR / "experiment-design_coax_multidataset_no_apparatus.json"


def _prepared(dataset_id, *, train_ids, test_ids):
    return MagicMock(
        dataset_id=dataset_id,
        split=MagicMock(
            train_instance_ids=np.asarray(list(train_ids)),
            test_instance_ids=np.asarray(list(test_ids)),
        ),
    )


def test_each_dataset_is_filtered_against_its_own_corpus_not_a_shared_union(monkeypatch):
    """Dataset A's corpus is 0-9 (10 ids); dataset B's is 0-49 (50 ids). A flat
    union would let A's split admit ids up to 49; the fix must not."""
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX",
        participants_per_condition=1, trials_per_participant=8,
        apparatus_instance_ids=[], apparatus_training_instance_ids=[],
    )
    study.data_by_dataset = {
        "small_corpus": _prepared("small_corpus", train_ids=range(0, 30), test_ids=range(30, 60)),
        "big_corpus": _prepared("big_corpus", train_ids=range(0, 30), test_ids=range(30, 60)),
    }
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    def fake_corpus_instance_ids(framework, dataset_id):
        if dataset_id == "small_corpus":
            return list(range(0, 10))
        if dataset_id == "big_corpus":
            return list(range(0, 50))
        return None

    monkeypatch.setattr("server.pipeline._corpus_instance_ids", fake_corpus_instance_ids)

    try:
        run_trials_stage(study, TrialsStageRequest(num_training=2, num_testing=2, seed=0))
    except Exception:
        pass  # only the per-dataset split filtering is under test here

    small = study.data_by_dataset["small_corpus"]
    big = study.data_by_dataset["big_corpus"]
    assert max(small.split.train_instance_ids.tolist(), default=-1) < 10
    assert max(small.split.test_instance_ids.tolist(), default=-1) < 10
    assert max(big.split.train_instance_ids.tolist(), default=-1) < 50
    # big_corpus legitimately has ids in [10, 49] that small_corpus must never see.
    assert any(10 <= i < 50 for i in big.split.train_instance_ids.tolist())


def test_a_dataset_that_already_got_a_real_override_is_left_alone(monkeypatch):
    """A dataset with its own declared apparatus instanceIds must not also be
    filtered by the corpus default -- its explicit override already stands."""
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX",
        participants_per_condition=1, trials_per_participant=8,
        apparatus_instance_ids=[100, 101],
        apparatus_instance_ids_by_dataset={"declared": [100, 101]},
        apparatus_training_instance_ids_by_dataset={"declared": [102]},
    )
    study.data_by_dataset = {
        "declared": _prepared("declared", train_ids=[1, 2, 100, 101, 102], test_ids=[3, 4, 100, 101]),
    }
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    def fake_corpus_instance_ids(framework, dataset_id):
        return [0]  # would wipe out {100, 101, 102} if wrongly applied

    monkeypatch.setattr("server.pipeline._corpus_instance_ids", fake_corpus_instance_ids)

    try:
        run_trials_stage(study, TrialsStageRequest(num_training=1, num_testing=2, seed=0))
    except Exception:
        pass

    declared = study.data_by_dataset["declared"]
    assert declared.split.test_instance_ids.tolist() == [100, 101]


@pytest.mark.skipif(not NO_APPARATUS_EXPORT.is_file(), reason="fixture export not present")
def test_the_real_export_runs_the_full_pipeline_without_leaking_ids(tmp_path):
    """End-to-end regression against the real export that motivated this."""
    from server.pipeline import build_study, run_dataset_stage, run_simulation_stage
    from server.schemas import DatasetStageRequest, SimulationRequest
    from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
        coax_available_instance_ids,
    )

    raw = json.loads(NO_APPARATUS_EXPORT.read_text())
    study = build_study(raw, project_name="coax-multi-no-apparatus", output_dir=tmp_path)
    assert study.design_export.dataset_ids == ["adult", "wine_quality", "forest_cover"]

    run_dataset_stage(study, DatasetStageRequest())
    trials_result = run_trials_stage(study, TrialsStageRequest())
    assert trials_result["counts"]["trials"] > 0

    trials = pd.DataFrame(study.trials)
    for dataset_id in ["adult", "wine_quality", "forest_cover"]:
        corpus_ids = set(coax_available_instance_ids(dataset_id))
        used_ids = set(trials.loc[trials["dataset"] == dataset_id, "instanceId"].astype(int))
        assert used_ids <= corpus_ids, f"{dataset_id} trials reference ids outside its own corpus"

    # CoAX's step count can exceed the trial count (e.g. multiple prediction
    # steps per trial for some conditions) -- the real invariant here is that
    # simulation completes at all, without the instance-not-found crash this
    # regression is about.
    sim_result = run_simulation_stage(study, SimulationRequest(mode="whole_experiment"), output_subdir="sim")
    assert sim_result["counts"]["steps"] > 0
