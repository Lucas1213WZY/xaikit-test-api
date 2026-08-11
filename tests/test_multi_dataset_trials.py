"""Native multi-dataset (between-subjects `dataset` IV) trial generation.

`data_by_dataset` routes per dataset -- never merges instance-id spaces, since
one dataset's instance ids mean nothing in another's feature space. These
tests cover the three layers that make that true end to end: pool routing in
`build_trial_sequence`, config validation in `init_trial_build_config`, and
the full `generate_experimental_trials` pipeline including training-phase
instance sourcing in `_add_training_and_testing_phases`.
"""

from types import SimpleNamespace

import pytest

from src.experiment_planner.config import init_experiment_config, set_iv, set_factor
from src.experiment_planner.counterbalance import assign_participants, build_trial_sequence
from src.experiment_planner.trials import (
    DATASET_IV_NAME,
    _add_training_and_testing_phases,
    generate_experimental_trials,
    init_trial_build_config,
)


def _fake_dataset(dataset_id, *, train_ids, test_ids):
    """A minimal stand-in for `PreparedDataset` -- trials.py only reads
    `.dataset_id`/`.train_instance_ids`/`.test_instance_ids` off it."""
    return SimpleNamespace(
        dataset_id=dataset_id,
        train_instance_ids=list(train_ids),
        test_instance_ids=list(test_ids),
    )


# ---------------------------------------------------------------------------
# build_trial_sequence: per-level instance pool routing
# ---------------------------------------------------------------------------


def test_build_trial_sequence_rejects_both_pool_args_at_once():
    with pytest.raises(ValueError, match="exactly one"):
        build_trial_sequence(
            assignments=[{"participantId": 1, "within_order": ["a"]}],
            instance_pool=[{"dataId": "x", "instanceId": "1"}],
            instance_pool_by_level={"x": [{"dataId": "x", "instanceId": "1"}]},
            pool_selector_key="dataset",
        )


def test_build_trial_sequence_rejects_neither_pool_arg():
    with pytest.raises(ValueError, match="exactly one"):
        build_trial_sequence(assignments=[{"participantId": 1, "within_order": ["a"]}])


def test_build_trial_sequence_requires_pool_selector_key_with_pool_by_level():
    with pytest.raises(ValueError, match="pool_selector_key"):
        build_trial_sequence(
            assignments=[{"participantId": 1, "within_order": ["a"]}],
            instance_pool_by_level={"x": [{"dataId": "x", "instanceId": "1"}]},
        )


def test_build_trial_sequence_routes_each_participant_to_their_own_level_pool():
    orders = [["a"]]
    assignments = assign_participants(4, orders, {"dataset": ["x", "y"]})
    pool_by_level = {
        "x": [{"dataId": "x", "instanceId": str(i)} for i in range(10, 13)],
        "y": [{"dataId": "y", "instanceId": str(i)} for i in range(90, 93)],
    }

    trials = build_trial_sequence(
        assignments=assignments,
        instance_pool_by_level=pool_by_level,
        pool_selector_key="dataset",
        trials_per_participant=2,
        id_map={"dataId": "dataId", "instanceId": "instanceId"},
        seed=0,
    )

    for trial in trials:
        level = trial["dataset"]
        assert trial["dataId"] == level
        assert int(trial["instanceId"]) in {int(row["instanceId"]) for row in pool_by_level[level]}


# ---------------------------------------------------------------------------
# init_trial_build_config: validation
# ---------------------------------------------------------------------------


def test_init_trial_build_config_rejects_both_data_args():
    iv_config, cvs, _ = init_experiment_config()
    set_iv(iv_config, "xai_type", "within", ["decision_tree"])
    with pytest.raises(ValueError, match="exactly one"):
        init_trial_build_config(
            data=_fake_dataset("x", train_ids=[1], test_ids=[2]),
            iv_config=iv_config,
            cvs=cvs,
            data_by_dataset={"x": _fake_dataset("x", train_ids=[1], test_ids=[2])},
        )


def test_init_trial_build_config_rejects_neither_data_arg():
    iv_config, cvs, _ = init_experiment_config()
    with pytest.raises(ValueError, match="exactly one"):
        init_trial_build_config(data=None, iv_config=iv_config, cvs=cvs)


def test_init_trial_build_config_blocks_ai_prediction_balancing_with_multi_dataset():
    iv_config, cvs, _ = init_experiment_config()
    with pytest.raises(ValueError, match="not yet supported"):
        init_trial_build_config(
            data=None,
            iv_config=iv_config,
            cvs=cvs,
            data_by_dataset={"x": _fake_dataset("x", train_ids=[1], test_ids=[2])},
            ai_predictions_by_instance={1: "pos"},
        )


# ---------------------------------------------------------------------------
# generate_experimental_trials: end-to-end multi-dataset run
# ---------------------------------------------------------------------------


def _multi_dataset_iv_config():
    iv_config, cvs, _ = init_experiment_config()
    set_iv(iv_config, "xai_type", "within", ["decision_tree", "logistic_regression"], randomization="block")
    set_iv(iv_config, DATASET_IV_NAME, "between", ["wine_quality", "mushrooms"])
    return iv_config, cvs


def test_generate_experimental_trials_requires_dataset_iv_when_multi_dataset(tmp_path):
    iv_config, cvs, _ = init_experiment_config()
    set_iv(iv_config, "xai_type", "within", ["decision_tree"], randomization="block")
    config = init_trial_build_config(
        data=None,
        iv_config=iv_config,
        cvs=cvs,
        data_by_dataset={
            "wine_quality": _fake_dataset("wine_quality", train_ids=range(0, 5), test_ids=range(5, 10)),
            "mushrooms": _fake_dataset("mushrooms", train_ids=range(100, 105), test_ids=range(105, 110)),
        },
        num_testing=2,
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match=f"IV named {DATASET_IV_NAME!r}"):
        generate_experimental_trials(config, show=False)


def test_generate_experimental_trials_never_mixes_instance_id_spaces(tmp_path):
    iv_config, cvs = _multi_dataset_iv_config()
    config = init_trial_build_config(
        data=None,
        iv_config=iv_config,
        cvs=cvs,
        data_by_dataset={
            "wine_quality": _fake_dataset("wine_quality", train_ids=range(0, 8), test_ids=range(200, 210)),
            "mushrooms": _fake_dataset("mushrooms", train_ids=range(300, 308), test_ids=range(500, 510)),
        },
        participants_per_between_condition=3,
        num_training=2,
        num_testing=4,
        output_dir=tmp_path,
    )
    result = generate_experimental_trials(config, show=False)
    trials = result.trials

    assert trials, "expected trials to be generated"

    # Every trial's instanceId belongs to the pool of its own dataId.
    wine_ids = set(range(0, 8)) | set(range(200, 210))
    mushroom_ids = set(range(300, 308)) | set(range(500, 510))
    for trial in trials:
        instance_id = int(trial["instanceId"])
        if trial["dataId"] == "wine_quality":
            assert instance_id in wine_ids
        else:
            assert trial["dataId"] == "mushrooms"
            assert instance_id in mushroom_ids
        # The between-subjects IV value and the sourced dataId must agree.
        assert trial[DATASET_IV_NAME] == trial["dataId"]

    # No participant is assigned to more than one dataset level.
    levels_by_participant = {}
    for trial in trials:
        levels_by_participant.setdefault(trial["participantId"], set()).add(trial["dataId"])
    assert all(len(levels) == 1 for levels in levels_by_participant.values())

    # Both training and testing phase rows were generated.
    phases = {trial["phase"] for trial in trials}
    assert phases == {"training", "testing"}


def test_generate_experimental_trials_multi_dataset_matches_single_dataset_backward_compat(tmp_path):
    """The single-dataset call path (no data_by_dataset) must still work unchanged."""
    iv_config, cvs, _ = init_experiment_config()
    set_iv(iv_config, "xai_type", "within", ["decision_tree"], randomization="block")
    config = init_trial_build_config(
        data=_fake_dataset("wine_quality", train_ids=range(0, 5), test_ids=range(5, 15)),
        iv_config=iv_config,
        cvs=cvs,
        participants_per_between_condition=2,
        num_testing=3,
        output_dir=tmp_path,
    )
    result = generate_experimental_trials(config, show=False)
    assert all(trial["dataId"] == "wine_quality" for trial in result.trials)


# ---------------------------------------------------------------------------
# _add_training_and_testing_phases: per-level training pool
# ---------------------------------------------------------------------------


def test_add_training_and_testing_phases_bounds_check_is_per_level():
    """A dataset with a small training pool must fail on its own -- even when
    another level in the same run has plenty of training instances."""
    trials = [
        {"participantId": 1, "dataId": "small", "instanceId": "1"},
        {"participantId": 2, "dataId": "big", "instanceId": "2"},
    ]
    with pytest.raises(ValueError, match="only 1"):
        _add_training_and_testing_phases(
            trials,
            train_instance_ids_by_level={"small": [1], "big": list(range(100, 110))},
            num_training=2,
            seed=0,
        )


def test_add_training_and_testing_phases_uses_participants_own_level_dataid():
    trials = [
        {"participantId": 1, "dataId": "wine_quality", "instanceId": "1"},
        {"participantId": 2, "dataId": "mushrooms", "instanceId": "2"},
    ]
    phased = _add_training_and_testing_phases(
        trials,
        train_instance_ids_by_level={"wine_quality": [10, 11], "mushrooms": [90, 91]},
        num_training=1,
        seed=0,
    )
    training_rows = [t for t in phased if t["phase"] == "training"]
    for row in training_rows:
        if row["participantId"] == 1:
            assert row["dataId"] == "wine_quality"
            assert int(row["instanceId"]) in {10, 11}
        else:
            assert row["dataId"] == "mushrooms"
            assert int(row["instanceId"]) in {90, 91}
