"""The fitted-parameter pools behind ``mode="diverse_participant"``.

These guard the two things that break silently: a column rename in
``assets/build_human_data.py`` (which would empty a pool without raising), and
an assignment that is not reproducible or not one-participant-one-row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.virtual_experiment_executor.participant_pools import (
    COAX_POOL,
    COXAM_COUNTERFACTUAL_POOL,
    COXAM_FORWARD_POOL,
    POOLS,
    SIM2REAL_POOL,
    canon_coax_strategy,
    canon_coax_xai_type,
    canon_coxam_condition,
    canon_dataset,
    canon_tested,
    draw_participant_parameters,
    draws_to_frame,
    is_diverse_mode,
    load_pool,
    match_pool_rows,
    provenance_columns,
)


# -- canonicalization -----------------------------------------------------


def test_canonicalizers_bridge_the_two_spellings():
    assert canon_dataset("Wine Quality") == canon_dataset("wine_quality")
    assert canon_coxam_condition("DT") == "decision_tree"
    assert canon_coxam_condition("Linear Regression") == "linear_regression"
    assert canon_coxam_condition("DT+LR") == "hybrid"
    assert canon_coax_strategy("Sensitive-features categorization") == "SensitiveFeatures"
    assert canon_coax_xai_type(float("nan")) == "none"


def test_canon_tested_does_not_confuse_with_and_without():
    """``w/o XAI`` contains ``w/``; the negative has to be tested first."""
    assert canon_tested("w/ XAI") == "true"
    assert canon_tested("w/o XAI") == "false"
    assert canon_tested(True) == "true"
    assert canon_tested(False) == "false"


# -- loading --------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(POOLS))
def test_every_pool_loads_with_its_mapped_columns(name):
    spec = POOLS[name]
    pool = load_pool(name)
    assert not pool.empty, f"{name} pool is empty"
    assert spec.id_column in pool.columns
    missing = [column for column in spec.parameters if column not in pool.columns]
    assert not missing, f"{name} pool lost mapped column(s) {missing}"
    for pool_filter in spec.filters:
        assert pool_filter.column in pool.columns, (
            f"{name} pool lost filter column {pool_filter.column!r}"
        )


def test_pool_sizes_match_the_published_populations():
    """Sentinels: a rebuild that changes these silently changes every study."""
    assert load_pool("coax")[COAX_POOL.id_column].nunique() == 330
    assert (
        load_pool("coxam_counterfactual")[COXAM_COUNTERFACTUAL_POOL.id_column].nunique()
        == 270
    )
    assert load_pool("sim2real")[SIM2REAL_POOL.id_column].nunique() == 46


def test_coxam_forward_pool_covers_both_datasets_and_all_conditions():
    pool = load_pool("coxam_forward")
    assert set(pool["dataId"]) == {"wine_quality", "mushrooms"}
    assert {"decision_tree", "linear_regression", "hybrid"} <= set(pool["condition"])
    # Only the mushrooms fit swept chi_value, so wine rows carry no
    # opportunity_cost -- documented in build_coxam_forward_pool.
    wine = pool[pool["dataId"] == "wine_quality"]
    assert wine["opportunity_cost"].isna().all()
    assert pool[pool["dataId"] == "mushrooms"]["opportunity_cost"].notna().any()


# -- matching -------------------------------------------------------------


def test_match_pool_rows_filters_to_the_condition_cell():
    everything = load_pool("coxam_counterfactual")
    matched = match_pool_rows(
        COXAM_COUNTERFACTUAL_POOL,
        {"dataId": "wine_quality", "condition": "decision_tree", "complexity": "high"},
    )
    assert not matched.empty
    assert len(matched) < len(everything)
    assert set(matched["dataId"].map(canon_dataset)) == {"wine_quality"}
    assert set(matched["condition"].map(canon_coxam_condition)) == {"decision_tree"}


def test_match_ignores_axes_the_caller_cannot_resolve():
    """A design with no complexity factor matches across it, not to nothing."""
    matched = match_pool_rows(
        COXAM_COUNTERFACTUAL_POOL, {"dataId": "wine_quality", "condition": "hybrid"}
    )
    assert set(matched["complexity"]) == {"high", "low"}


def test_coax_condition_cells_are_populated_for_the_fitted_datasets():
    matched = match_pool_rows(
        COAX_POOL,
        {
            "dataId": "wine_quality",
            "xai_method": "lime",
            "xai_type": "attribution",
            "tested_w_xai": True,
            "strategy": "AttributionSum",
        },
    )
    assert len(matched) >= 10


# -- drawing --------------------------------------------------------------


def _draw(**overrides):
    kwargs = dict(
        condition={"exp_property": "faithful"},
        participants=list(range(1, 13)),
        seed=0,
    )
    kwargs.update(overrides)
    return draw_participant_parameters("sim2real", **kwargs)


def test_draw_is_deterministic_in_the_seed():
    first = _draw(seed=7)
    again = _draw(seed=7)
    other = _draw(seed=8)
    assert [d.fitted_participant_id for d in first.values()] == [
        d.fitted_participant_id for d in again.values()
    ]
    assert [d.fitted_participant_id for d in first.values()] != [
        d.fitted_participant_id for d in other.values()
    ]


def test_draw_deals_distinct_people_while_the_pool_lasts():
    """The whole point: 12 virtual participants must not be one person x 12."""
    draws = _draw()
    assigned = [d.fitted_participant_id for d in draws.values()]
    assert len(set(assigned)) == len(assigned)
    sampled = {tuple(sorted(d.parameters.items(), key=str)) for d in draws.values()}
    assert len(sampled) > 1


def test_draw_reshuffles_rather_than_running_out():
    draws = _draw(participants=list(range(1, 40)))
    assert len(draws) == 39
    assert all(d.parameter_source == "pool" for d in draws.values())


def test_draw_with_replacement_is_iid():
    draws = _draw(replace=True, seed=3)
    assigned = [d.fitted_participant_id for d in draws.values()]
    assert len(set(assigned)) < len(assigned)


def test_relaxation_widens_one_axis_at_a_time():
    """A cell the fit never covered borrows the nearest one, and says which."""
    from src.virtual_experiment_executor.participant_pools import match_with_relaxation

    exact, dropped = match_with_relaxation(
        COXAM_FORWARD_POOL,
        {"dataId": "wine_quality", "condition": "decision_tree"},
        relax=("complexity", "condition"),
    )
    assert not exact.empty and dropped == ()

    # The wine_quality forward fits were run per explanation family, so hybrid
    # has no cell of its own.
    widened, dropped = match_with_relaxation(
        COXAM_FORWARD_POOL,
        {"dataId": "wine_quality", "condition": "hybrid"},
        relax=("complexity", "condition"),
    )
    assert not widened.empty
    assert dropped == ("complexity", "condition")
    assert set(widened["dataId"]) == {"wine_quality"}


def test_relaxed_draw_is_still_a_real_person_not_a_fallback():
    with pytest.warns(UserWarning, match="relaxed"):
        draws = draw_participant_parameters(
            "coxam_forward",
            condition={"dataId": "wine_quality", "condition": "hybrid"},
            participants=[1, 2, 3],
            relax=("complexity", "condition"),
        )
    for draw in draws.values():
        assert draw.parameter_source.startswith("pool_relaxed")
        assert not draw.is_fallback
        assert draw.parameters


def test_unfitted_cell_falls_back_and_says_so():
    """CoAX never fitted mushrooms -- fall back loudly, do not raise."""
    with pytest.warns(UserWarning, match="mushrooms|No coax participants"):
        draws = draw_participant_parameters(
            "coax",
            condition={
                "dataId": "mushrooms",
                "xai_method": "lime",
                "xai_type": "attribution",
                "tested_w_xai": True,
                "strategy": "AttributionSum",
            },
            participants=[1, 2, 3],
        )
    assert all(d.parameter_source == "fitted_mean_fallback" for d in draws.values())
    assert all(d.parameters == {} for d in draws.values())


def test_drawn_values_are_clipped_into_the_trained_range():
    draws = draw_participant_parameters(
        "coxam_counterfactual",
        condition={"dataId": "wine_quality", "condition": "decision_tree"},
        participants=list(range(1, 25)),
    )
    for draw in draws.values():
        assert -2.0 <= draw.parameters["memory_recall_threshold"] <= 0.5
        assert 0.0 <= draw.parameters["counterfactual_overshoot_fraction"] <= 0.5
        assert 0.0 <= draw.parameters["time_penalty_weight"] <= 0.02


def test_coax_k_is_drawn_as_an_integer():
    draws = draw_participant_parameters(
        "coax",
        condition={
            "dataId": "wine_quality",
            "xai_type": "attribution",
            "strategy": "AttributionSum",
        },
        participants=[1, 2, 3],
    )
    for draw in draws.values():
        assert isinstance(draw.parameters["k"], int)
        # AttributionSum is fitted with scaling_factor, never sensitivity.
        assert "scaling_factor" in draw.parameters
        assert "sensitivity" not in draw.parameters


def test_sim2real_draw_varies_strategy_as_well_as_parameters():
    """The sparse fit selected the strategy per participant; keep that."""
    draws = draw_participant_parameters(
        "sim2real", condition={"exp_property": "sparse"}, participants=list(range(1, 13))
    )
    strategies = {d.parameters["strategy"] for d in draws.values()}
    assert len(strategies) > 1


# -- provenance -----------------------------------------------------------


def test_draws_to_frame_is_one_row_per_participant():
    draws = _draw()
    frame = draws_to_frame(draws)
    assert len(frame) == 12
    assert frame["participantId"].is_unique
    assert {"pool", "fitted_participant_id", "parameter_source"} <= set(frame.columns)
    assert any(column.startswith("sampled_") for column in frame.columns)


def test_provenance_columns_are_json_safe_scalars():
    draw = next(iter(_draw().values()))
    columns = provenance_columns(draw)
    assert columns["parameter_source"] == "pool"
    assert columns["parameter_pool"] == "sim2real"
    assert all(not isinstance(value, (dict, list, pd.Series)) for value in columns.values())
    assert provenance_columns(None) == {}


def test_is_diverse_mode_recognizes_the_mode():
    assert is_diverse_mode("diverse_participant")
    assert is_diverse_mode(" Diverse_Participant ")
    assert not is_diverse_mode("whole_experiment")
    assert not is_diverse_mode(None)


# -- trial selection ------------------------------------------------------


def test_diverse_mode_selects_the_whole_experiment():
    """Selection is identical to whole_experiment; only the runner differs."""
    from src.experiment_planner import select_trial_rows

    trials = pd.DataFrame(
        {"participantId": [1, 1, 2, 2], "instanceId": [10, 11, 10, 11]}
    )
    assert len(select_trial_rows(trials, "diverse_participant")) == len(trials)
    pd.testing.assert_frame_equal(
        select_trial_rows(trials, "diverse_participant"),
        select_trial_rows(trials, "whole_experiment"),
    )


def test_unknown_mode_names_the_new_one():
    from src.experiment_planner import select_trial_rows

    with pytest.raises(ValueError, match="diverse_participant"):
        select_trial_rows(pd.DataFrame({"participantId": [1]}), "nonsense_mode")
