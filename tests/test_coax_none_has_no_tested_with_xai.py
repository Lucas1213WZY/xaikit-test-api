"""CoAX's `none` arm shows no explanation, so it has no `tested_w_xai=True` half.

Generating both halves manufactures a comparison that cannot differ: the runner
gives `none` the same SensitiveFeatures strategy and requests no explanation
either way, so the two cells are one condition wearing two labels. It also
splits the arm's trials across the duplicate cells and charges the
multiple-comparison correction for the extra pairs.
"""

import warnings

import pandas as pd
import pytest

import src.api as xk
from src.experiment_planner.counterbalance import (
    assign_participants,
    build_trial_sequence,
    choose_counterbalancing,
)
from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
    COAX_IMPOSSIBLE_CELLS,
)

IV_CONFIG = {
    "xai_type": {"type": "between", "levels": ["none", "attribution", "importance"]},
    "tested_w_xai": {"type": "within", "randomization": "trial", "levels": [False, True]},
}


def _sequence(impossible_cells, *, trials_per_participant=10):
    orders, _ = choose_counterbalancing(["single_condition"], strategy="balanced_latin_square")
    assignments = assign_participants(6, orders, {"xai_type": ["none", "attribution", "importance"]})
    return build_trial_sequence(
        assignments=assignments,
        instance_pool=[{"dataId": "d", "instanceId": i} for i in range(200)],
        trials_per_participant=trials_per_participant,
        trial_randomized_ivs={"tested_w_xai": [False, True]},
        trial_randomization_strategy="balanced",
        id_map={"dataId": "dataId", "instanceId": "instanceId"},
        seed=1,
        impossible_cells=impossible_cells,
    )


def test_none_never_gets_a_tested_with_xai_trial():
    trials = pd.DataFrame(_sequence(COAX_IMPOSSIBLE_CELLS))
    none_rows = trials[trials["xai_type"] == "none"]
    assert len(none_rows) > 0
    assert not none_rows["tested_w_xai"].any()


def test_the_other_arms_keep_both_halves():
    trials = pd.DataFrame(_sequence(COAX_IMPOSSIBLE_CELLS))
    for arm in ("attribution", "importance"):
        rows = trials[trials["xai_type"] == arm]
        assert set(rows["tested_w_xai"]) == {True, False}, arm
        # Still evenly split -- dropping a cell elsewhere must not unbalance these.
        assert rows["tested_w_xai"].sum() * 2 == len(rows), arm


def test_the_none_arm_keeps_its_full_trial_budget():
    """The dropped cell is re-divided into the surviving one, not left as a hole
    -- a `none` participant runs as many trials as everyone else."""
    trials = pd.DataFrame(_sequence(COAX_IMPOSSIBLE_CELLS))
    per_participant = trials.groupby("participantId").size()
    assert set(per_participant) == {10}


def test_without_the_rule_the_duplicate_cell_is_still_generated():
    """The behaviour the rule exists to remove, pinned so the test above cannot
    pass for the wrong reason."""
    trials = pd.DataFrame(_sequence(None))
    none_rows = trials[trials["xai_type"] == "none"]
    assert none_rows["tested_w_xai"].any()


def test_an_arm_left_with_no_cells_at_all_is_an_error():
    with pytest.raises(ValueError, match="no possible trial-level cells"):
        _sequence([{"tested_w_xai": True}, {"tested_w_xai": False}])


# -- how a study picks the rule up ----------------------------------------


def _study(tmp_path):
    study = xk.xaikitTest(output_dir=tmp_path)
    study.set_design(iv_config=IV_CONFIG, dvs={"forward_accuracy": ["continuous"]}, show=False)
    return study


def test_framework_argument_selects_the_coax_rules(tmp_path):
    assert _study(tmp_path)._impossible_cells_for("coax") == COAX_IMPOSSIBLE_CELLS


def test_a_set_cognitive_model_is_enough_without_the_argument(tmp_path):
    study = _study(tmp_path)
    study.cognitive_model_id = "coax"
    assert study._impossible_cells_for(None) == COAX_IMPOSSIBLE_CELLS


def test_other_agents_are_left_alone(tmp_path):
    """The rule is CoAX's own, not a global claim about the word `none`."""
    study = _study(tmp_path)
    study.cognitive_model_id = "coxam"
    assert study._impossible_cells_for(None) is None


def test_an_unresolved_agent_warns_rather_than_silently_skipping(tmp_path):
    """cognitive_model_id is only set by set_cognitive_model, which studies
    commonly call after generating trials -- so this case is the common one."""
    study = _study(tmp_path)
    with pytest.warns(UserWarning, match="no agent is selected yet"):
        assert study._impossible_cells_for(None) is None


def test_a_design_with_no_none_level_does_not_warn(tmp_path):
    study = xk.xaikitTest(output_dir=tmp_path)
    study.set_design(
        iv_config={
            "xai_type": {"type": "between", "levels": ["attribution", "importance"]},
            "tested_w_xai": {"type": "within", "randomization": "trial", "levels": [False, True]},
        },
        dvs={"forward_accuracy": ["continuous"]},
        show=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert study._impossible_cells_for(None) is None
