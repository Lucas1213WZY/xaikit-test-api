"""A Sim2Real testing phase cannot be longer than the corpus it draws from.

Trials inside a block sample instances without replacement, so asking for more
testing trials than the corpus carries fails deep inside build_trial_sequence
with "need 30, got 29" -- a message that names no corpus and no fix. The server
clamps to what the corpus serves and says so instead.
"""

import warnings

import pytest

from src.experiment_planner.design_export import (
    SIM2REAL_DEFAULT_TEST_INSTANCE_IDS,
    SIM2REAL_DEFAULT_TRAIN_INSTANCE_IDS,
)
from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
    sim2real_available_instance_ids,
)


def test_the_default_split_is_the_corpus_split():
    """The default is the study's own split, not a subset of it: an earlier
    default stopped at id 30 and silently dropped 8 of the study's instances."""
    assert list(SIM2REAL_DEFAULT_TEST_INSTANCE_IDS) == sim2real_available_instance_ids(split="test")
    assert list(SIM2REAL_DEFAULT_TRAIN_INSTANCE_IDS) == sim2real_available_instance_ids(
        split="training"
    )


def test_the_default_test_split_matches_the_published_study():
    """29 testing instances, ids 10-38 -- the same count the human trials carry
    as distinct qids."""
    assert len(SIM2REAL_DEFAULT_TEST_INSTANCE_IDS) == 29
    assert min(SIM2REAL_DEFAULT_TEST_INSTANCE_IDS) == 10
    assert max(SIM2REAL_DEFAULT_TEST_INSTANCE_IDS) == 38


# -- the server's clamp ----------------------------------------------------


def _resolve(num_testing, apparatus_test_count):
    """The server's clamp, exercised on its own arithmetic."""
    warning = None
    if apparatus_test_count and num_testing > apparatus_test_count:
        warning = (
            f"num_testing={num_testing} exceeds the {apparatus_test_count} testing "
            f"instances this design's published corpus can serve, and trials draw "
            f"without replacement. Running with {apparatus_test_count}, the number "
            f"the original study used."
        )
        num_testing = apparatus_test_count
    return num_testing, warning


def test_an_over_long_testing_phase_is_clamped_and_explained():
    resolved, warning = _resolve(40, 29)
    assert resolved == 29
    assert "29" in warning and "original study" in warning


def test_a_testing_phase_that_fits_is_left_alone():
    resolved, warning = _resolve(20, 29)
    assert resolved == 20
    assert warning is None


def test_exactly_the_corpus_size_is_not_clamped():
    resolved, warning = _resolve(29, 29)
    assert resolved == 29
    assert warning is None


def test_no_declared_corpus_means_no_clamp():
    """A design with no apparatus instance ids has no pool to clamp against."""
    resolved, warning = _resolve(500, 0)
    assert resolved == 500
    assert warning is None
