"""A multi-dataset apparatus config with no ``appId`` must broadcast, not
default to one dataset.

Regression: a real CoXAM export declared ``dataset`` as a between-subjects IV
with two levels (wine_quality, mushrooms) but its two apparatus configurations
(one per XAI form, LR/DT) named no ``appId`` at all -- only
``instanceIds``/``trainingInstanceIds``. ``_apparatus_instance_ids_by_dataset``
used to attribute an appId-less entry to the single legacy
``studyDesign.dataset`` field only, so mushrooms got no apparatus override at
all and its trials sampled unrestricted from the raw dataset, most of which
falls outside the CoXAM published corpus -- crashing with "N trial
instance(s) are outside the CoXAM asset corpus for appId='mushrooms'" at
simulation time. An appId-less entry must instead broadcast to every dataset
level the design declares.
"""

import json
from pathlib import Path

import pytest

from src.experiment_planner.design_export import parse_design_export

EXPORT_DIR = Path(__file__).resolve().parents[1] / "tutorials" / "experiment_output"
NO_APPID_EXPORT = EXPORT_DIR / "experiment-design_coxam_multidataset_no_appid.json"


def _raw(*, apparatus_params):
    return {
        "studyDesign": {
            "dataset": "Wine Quality",
            "independentVariables": [
                {"factor": "XAI Type", "levelsOrRange": "Decision Tree | Logistic Regression",
                 "allocation": "Between-subjects"},
                {"factor": "Dataset", "levelsOrRange": "Wine Quality | Mushroom",
                 "allocation": "Between-subjects"},
            ],
        },
        "apparatus": [{"id": "a1", "params": apparatus_params}],
        "userModel": "CoXAM",
    }


def test_an_appid_less_apparatus_entry_broadcasts_to_every_dataset_level():
    design = parse_design_export(_raw(
        apparatus_params={"instanceIds": "10-19", "trainingInstanceIds": "0-4"}
    ))

    assert design.dataset_ids == ["wine_quality", "mushrooms"]
    assert design.apparatus_instance_ids_by_dataset == {
        "wine_quality": list(range(10, 20)),
        "mushrooms": list(range(10, 20)),
    }
    assert design.apparatus_training_instance_ids_by_dataset == {
        "wine_quality": list(range(0, 5)),
        "mushrooms": list(range(0, 5)),
    }


def test_an_explicit_appid_still_targets_only_its_own_dataset():
    design = parse_design_export(_raw(
        apparatus_params={"appId": "mushrooms", "instanceIds": "10-19", "trainingInstanceIds": "0-4"}
    ))

    assert design.apparatus_instance_ids_by_dataset == {"mushrooms": list(range(10, 20))}
    assert "wine_quality" not in design.apparatus_instance_ids_by_dataset


def test_a_single_level_dataset_iv_is_followed_even_with_a_blank_dataset_field():
    """Regression: a design can name its one dataset only through a
    single-level 'dataset' IV, leaving studyDesign.dataset blank -- this used
    to fall through to the (empty) singular field and raise "no dataset
    given", since the IV was only trusted when it declared 2+ levels."""
    raw = {
        "studyDesign": {
            "dataset": "",
            "independentVariables": [
                {"factor": "XAI Type", "levelsOrRange": "Decision Tree | Logistic Regression",
                 "allocation": "Between-subjects"},
                {"factor": "Dataset", "levelsOrRange": "Mushroom", "allocation": "Between-subjects"},
            ],
        },
        "apparatus": [],
        "userModel": "CoXAM",
    }
    design = parse_design_export(raw)

    # dataset_ids (what run_dataset_stage actually resolves against) follows
    # the IV; the legacy singular dataset_id field stays blank since
    # studyDesign.dataset itself was never filled in.
    assert design.dataset_ids == ["mushrooms"]
    assert not design.is_multi_dataset


def test_single_dataset_design_is_unaffected():
    """Backward compatibility: with only one dataset level, broadcasting to
    'every dataset level' is the same single key as before."""
    raw = {
        "studyDesign": {
            "dataset": "Wine Quality",
            "independentVariables": [
                {"factor": "XAI Type", "levelsOrRange": "Decision Tree | Logistic Regression",
                 "allocation": "Between-subjects"},
            ],
        },
        "apparatus": [{"id": "a1", "params": {"instanceIds": "10-19", "trainingInstanceIds": "0-4"}}],
        "userModel": "CoXAM",
    }
    design = parse_design_export(raw)

    assert design.dataset_ids == ["wine_quality"]
    assert design.apparatus_instance_ids_by_dataset == {"wine_quality": list(range(10, 20))}


@pytest.mark.skipif(not NO_APPID_EXPORT.is_file(), reason="fixture export not present")
def test_the_real_export_runs_the_full_pipeline_without_the_corpus_error(tmp_path):
    """End-to-end regression against the real export that motivated this."""
    from server.pipeline import (
        build_study,
        run_dataset_stage,
        run_simulation_stage,
        run_trials_stage,
    )
    from server.schemas import DatasetStageRequest, SimulationRequest, TrialsStageRequest

    raw = json.loads(NO_APPID_EXPORT.read_text())
    study = build_study(raw, project_name="coxam-multi-no-appid", output_dir=tmp_path)
    assert study.design_export.dataset_ids == ["wine_quality", "mushrooms"]

    dataset_result = run_dataset_stage(study, DatasetStageRequest())
    assert set(dataset_result["datasets"]) == {"wine_quality", "mushrooms"}
    # Every dataset ends up with a real AI -- CoXAM's counterfactual environment
    # predicts with it, so there is nothing to skip. What differs per dataset is
    # *which* AI, and the entry records that (see _load_published_model).
    assert set(dataset_result["models"]) == {"wine_quality", "mushrooms"}
    for entry in dataset_result["models"].values():
        assert entry["model"] is not None
        assert not entry.get("model_skipped_reason")

    sources = {
        name: entry["model"]["model_source"]
        for name, entry in dataset_result["models"].items()
    }
    # wine_quality's prepared features line up with the published corpus, so the
    # study explains the same AI the human participants faced.
    assert sources["wine_quality"] == "published_weights"
    # mushrooms' one-hot encoding does not: the published weights expect 9
    # features and the prepared dataset has 10. The weights are positional, so
    # loading them anyway would silently produce meaningless predictions --
    # training a fresh model and saying why is the correct fallback.
    assert sources["mushrooms"] == "trained"
    assert "features" in dataset_result["models"]["mushrooms"]["model"]["model_source_note"]

    trials_result = run_trials_stage(study, TrialsStageRequest())
    assert trials_result["counts"]["trials"] > 0

    sim_result = run_simulation_stage(study, SimulationRequest(mode="whole_experiment"), output_subdir="sim")
    assert sim_result["counts"]["steps"] == trials_result["counts"]["trials"]
    assert set(study.simulated_results["dataset"].unique()) == {"wine_quality", "mushrooms"}
