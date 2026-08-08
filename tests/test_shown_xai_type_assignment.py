"""Trial generation must record which explanation family each trial shows.

A single-family condition (``xai_type='decision_tree'``) shows that family on
every trial. A multi-family condition (``xai_type='hybrid'``, which shows both)
alternates per trial -- balanced and seeded, so the assignment is reproducible
and auditable from the exported trial table rather than re-drawn at simulation
time.
"""

import pytest

from src.experiment_planner.trials import (
    MULTI_FAMILY_XAI_TYPES,
    _assign_shown_xai_type,
)


def _trials(xai_type, count, *, participant_id=1, phase="testing"):
    return [
        {
            "participantId": participant_id,
            "phase": phase,
            "xai_type": xai_type,
            "instanceId": str(index),
        }
        for index in range(count)
    ]


def test_single_family_condition_echoes_itself():
    for xai_type in ("decision_tree", "logistic_regression"):
        assigned = _assign_shown_xai_type(_trials(xai_type, 4), seed=1)
        assert [trial["shown_xai_type"] for trial in assigned] == [xai_type] * 4


def test_hybrid_alternates_between_both_families():
    assigned = _assign_shown_xai_type(_trials("hybrid", 4), seed=1)
    shown = [trial["shown_xai_type"] for trial in assigned]

    assert set(shown) == set(MULTI_FAMILY_XAI_TYPES["hybrid"])
    # Four trials over two families must split evenly.
    assert shown.count("decision_tree") == 2
    assert shown.count("logistic_regression") == 2


def test_hybrid_is_balanced_as_evenly_as_the_count_allows():
    """An odd trial count cannot split evenly, but must not be lopsided."""
    assigned = _assign_shown_xai_type(_trials("hybrid", 5), seed=3)
    shown = [trial["shown_xai_type"] for trial in assigned]

    counts = sorted([shown.count(family) for family in MULTI_FAMILY_XAI_TYPES["hybrid"]])
    assert counts == [2, 3]


def test_hybrid_assignment_is_reproducible_for_a_seed():
    first = _assign_shown_xai_type(_trials("hybrid", 6), seed=42)
    second = _assign_shown_xai_type(_trials("hybrid", 6), seed=42)

    assert [t["shown_xai_type"] for t in first] == [t["shown_xai_type"] for t in second]


def test_hybrid_balances_each_participant_and_phase_independently():
    trials = (
        _trials("hybrid", 4, participant_id=1, phase="training")
        + _trials("hybrid", 4, participant_id=1, phase="testing")
        + _trials("hybrid", 4, participant_id=2, phase="testing")
    )
    assigned = _assign_shown_xai_type(trials, seed=7)

    for participant_id in (1, 2):
        for phase in ("training", "testing"):
            group = [
                t["shown_xai_type"]
                for t in assigned
                if t["participantId"] == participant_id and t["phase"] == phase
            ]
            if not group:
                continue
            assert group.count("decision_tree") == group.count("logistic_regression")


def test_trials_without_an_xai_type_are_left_alone():
    trials = [{"participantId": 1, "phase": "testing", "instanceId": "0"}]
    assigned = _assign_shown_xai_type(trials, seed=1)

    assert "shown_xai_type" not in assigned[0]


@pytest.mark.parametrize("count", [1, 2, 3, 8, 9])
def test_every_hybrid_trial_gets_a_family(count):
    assigned = _assign_shown_xai_type(_trials("hybrid", count), seed=5)

    assert all(
        trial["shown_xai_type"] in MULTI_FAMILY_XAI_TYPES["hybrid"] for trial in assigned
    )
