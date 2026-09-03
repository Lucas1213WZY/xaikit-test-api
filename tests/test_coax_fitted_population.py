"""CoAX under ``mode="fitted_population"``.

The mode exists because ``diverse_participant`` draws parameters but not the
*strategy*: it keeps the one strategy ``PREFERRED_COAX_STRATEGY_BY_XAI_TYPE``
names for the condition and filters the parameter draw to it. The fitted
population is a mixture -- its ``importance`` cells split four ways and its
``attribution`` w/o XAI cell is majority SensitiveFeatures, not the preferred
AttributionSum -- so these tests are mostly about the mixture arriving intact
and each virtual participant staying one coherent human.

Runs against the published CoAX corpus, so no model training is needed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_study_runner import (
    coax_models_for_trials,
    coax_population_models,
)
from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
    coax_available_instance_ids,
    coax_available_strategies,
    run_coax_experiment_executor,
)
from src.virtual_experiment_executor.participant_pools import (
    draw_participant_units,
    is_diverse_mode,
    is_fitted_population_mode,
    samples_participants,
)

DATA_ID = "wine_quality"
XAI_METHOD = "lime"
PARTICIPANTS_PER_CONDITION = 8
TRAINING_TRIALS = 10
TESTING_TRIALS = 30


@pytest.fixture(scope="module")
def repository():
    from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_human_replay import (
        build_coax_study_repository,
    )

    return build_coax_study_repository(DATA_ID, XAI_METHOD)


@pytest.fixture(scope="module")
def trials():
    """A CoAX session: a training block, then a tested block alternating halves.

    Training trials carry ``tested_w_xai=None`` exactly as the generated tables
    do -- which is what makes ``share_training_memory`` necessary, since there is
    no tested half to route them by.
    """
    instance_ids = coax_available_instance_ids(DATA_ID)[: TRAINING_TRIALS + TESTING_TRIALS]
    training, testing = instance_ids[:TRAINING_TRIALS], instance_ids[TRAINING_TRIALS:]
    rows = []
    participant = 0
    for xai_type in ("attribution", "importance"):
        for _ in range(PARTICIPANTS_PER_CONDITION):
            participant += 1
            for position, instance_id in enumerate(training):
                rows.append(
                    {
                        "participantId": participant,
                        "trialId": position,
                        "instanceId": int(instance_id),
                        "dataId": DATA_ID,
                        "xai_method": XAI_METHOD,
                        "xai_type": xai_type,
                        "tested_w_xai": None,
                        "phase": "training",
                    }
                )
            for position, instance_id in enumerate(testing):
                rows.append(
                    {
                        "participantId": participant,
                        "trialId": 100 + position,
                        "instanceId": int(instance_id),
                        "dataId": DATA_ID,
                        "xai_method": XAI_METHOD,
                        "xai_type": xai_type,
                        "tested_w_xai": position % 2 == 0,
                        "phase": "testing",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def models(trials):
    return coax_models_for_trials(trials, by_tested_condition=True)


@pytest.fixture(scope="module")
def results(trials, models, repository):
    participant_models, draws = coax_population_models(
        trials, models, data_id=DATA_ID, seed=0
    )
    return run_coax_experiment_executor(
        trials=trials,
        mode="fitted_population",
        cognitive_model=models,
        data_repository=repository,
        dvs={"forward_accuracy": ["continuous"]},
        participant_models=participant_models,
        participant_draws=draws,
        share_training_memory=True,
    )


# -- mode predicates -------------------------------------------------------


def test_the_two_sampling_modes_stay_distinct():
    assert is_fitted_population_mode("fitted_population")
    assert not is_fitted_population_mode("diverse_participant")
    assert not is_diverse_mode("fitted_population")
    assert samples_participants("fitted_population")
    assert samples_participants("diverse_participant")
    assert not samples_participants("whole_experiment")


# -- the draw --------------------------------------------------------------


def test_a_drawn_unit_is_one_human_across_both_tested_halves():
    draws = draw_participant_units(
        "coax",
        condition={"dataId": DATA_ID, "xai_method": XAI_METHOD, "xai_type": "importance"},
        participants=list(range(1, 13)),
        seed=0,
    )
    assert set(draws) == set(range(1, 13))
    for halves in draws.values():
        ids = {draw.fitted_participant_id for draw in halves.values()}
        assert len(ids) == 1, "a participant's two halves must be the same human"
        assert set(halves) <= {"true", "false"}


def test_the_draw_carries_the_fitted_strategy_and_the_humans_own_accuracy():
    draws = draw_participant_units(
        "coax",
        condition={"dataId": DATA_ID, "xai_method": XAI_METHOD, "xai_type": "importance"},
        participants=[1, 2, 3, 4, 5, 6],
        seed=0,
    )
    for halves in draws.values():
        for draw in halves.values():
            assert draw.attributes["strategy"] in {
                "AttributionSum",
                "SalientFeatures",
                "SensitiveFeatures",
                "ImportanceCategorization",
            }
            assert 0.0 <= float(draw.attributes["human_pai"]) <= 1.0


def test_drawing_units_refuses_a_within_unit_filter():
    """``tested_w_xai`` and ``strategy`` vary inside one person, so filtering on
    either would deal half a participant."""
    with pytest.warns(UserWarning, match="vary within one fitted person"):
        draw_participant_units(
            "coax",
            condition={"dataId": DATA_ID, "xai_type": "importance", "tested_w_xai": True},
            participants=[1, 2],
            seed=0,
        )


def test_the_draw_reproduces_the_populations_strategy_mixture():
    """A large importance draw must not collapse onto one strategy.

    This is the whole point of the mode: ``diverse_participant`` would run
    SalientFeatures for every one of these participants.
    """
    draws = draw_participant_units(
        "coax",
        condition={"dataId": DATA_ID, "xai_method": XAI_METHOD, "xai_type": "importance"},
        participants=list(range(200)),
        seed=0,
    )
    strategies = {
        draw.attributes["strategy"] for halves in draws.values() for draw in halves.values()
    }
    assert len(strategies) >= 3, f"expected a mixture, got {strategies}"


# -- the run ---------------------------------------------------------------


def test_each_virtual_participant_runs_exactly_one_fitted_human(results):
    per_participant = results.groupby("participantId")["fitted_participant_id"].nunique()
    assert set(per_participant.unique()) == {1}


def test_more_than_one_strategy_actually_runs(results):
    """The mixture must survive into the simulation, not just the draw."""
    testing = results[results["phase"] == "testing"]
    for xai_type, rows in testing.groupby("xai_type"):
        strategies = set(rows["cognitive_model_strategy"])
        if xai_type == "attribution":
            # The fitted attribution w/ XAI cell is 100% AttributionSum, so only
            # the w/o XAI half can vary -- but vary it must.
            assert strategies <= {"AttributionSum", "SensitiveFeatures"}
        assert strategies, f"no strategy ran for {xai_type}"
    assert len(set(testing["cognitive_model_strategy"])) >= 2


def test_every_strategy_run_is_legal_for_its_condition(results):
    testing = results[results["phase"] == "testing"]
    for (xai_type, tested), rows in testing.groupby(["xai_type", "tested_w_xai"]):
        allowed = set(coax_available_strategies(xai_type, tested))
        # A fitted human may sit outside the generator's rule; the runner warns
        # and keeps the fit. What must hold is that it is a real CoAX strategy.
        assert set(rows["cognitive_model_strategy"]) <= allowed | {
            "AttributionSum",
            "SalientFeatures",
            "SensitiveFeatures",
            "ImportanceCategorization",
        }


def test_the_run_records_the_drawn_humans_own_accuracy(results):
    """``human_pai`` is what lets the figure show the sampled humans' real score."""
    assert "human_pai" in results.columns
    assert results["human_pai"].notna().any()
    assert results["human_pai"].dropna().between(0.0, 1.0).all()


def test_participants_differ_from_each_other(results):
    """The failure mode the sampling modes exist to fix: zero within-condition SD."""
    testing = results[results["phase"] == "testing"]
    per_participant = testing.groupby(["xai_type", "participantId"])[
        "cognitive_correct_vs_ai"
    ].mean()
    for xai_type, chunk in per_participant.groupby("xai_type"):
        assert chunk.std() > 0.0, f"{xai_type} participants all scored identically"


# -- the trials follow the draw --------------------------------------------


def test_each_agent_runs_its_own_drawn_humans_instances(trials, models):
    """The parameters were fitted against that person's own trial sequence, so the
    agent has to see that sequence -- not the design's generated one."""
    from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_human_replay import (
        ANONYMISED_HUMAN_TRIALS_FILE,
        load_coax_human_trials,
    )
    from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_study_runner import (
        coax_population_trials,
    )

    participant_models, draws = coax_population_models(
        trials, models, data_id=DATA_ID, seed=0
    )
    replayed = coax_population_trials(draws, trials)
    log = load_coax_human_trials(path=ANONYMISED_HUMAN_TRIALS_FILE)

    units = {
        participant: (draw.fitted_participant_id, draw.attributes["fitted_session"])
        for (participant, _key), draw in draws.items()
    }
    assert units, "expected every virtual participant to carry its drawn person"

    for participant, (fitted_id, session) in units.items():
        ours = set(replayed.loc[replayed["participantId"] == participant, "instanceId"])
        theirs = set(
            log.loc[
                (log["participantId"] == fitted_id) & (log["session"] == int(session)),
                "instanceId",
            ]
        )
        assert ours == theirs, f"participant {participant} does not run person {fitted_id}'s instances"


def test_the_replayed_table_keeps_the_designs_participants_and_conditions(trials, models):
    """The draw decides which instances; the design still decides who exists."""
    from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_study_runner import (
        coax_population_trials,
    )

    _models, draws = coax_population_models(trials, models, data_id=DATA_ID, seed=0)
    replayed = coax_population_trials(draws, trials)

    assert set(replayed["participantId"]) <= set(trials["participantId"])
    # One arm per person, as the fitted log has it.
    assert set(replayed.groupby("participantId")["xai_type"].nunique()) == {1}
    # A real session, not the four-trial stub the design table carries.
    assert replayed.groupby("participantId").size().min() > len(trials) / trials[
        "participantId"
    ].nunique()
    assert {"training", "testing"} <= set(replayed["phase"])


# -- the memory guard ------------------------------------------------------


def test_both_tested_halves_learn_from_training(trials, models, repository):
    """Without shared training memory the w/ XAI model would never see a training
    trial, because training rows carry no tested half to route them by.

    An untrained CoAX strategy answers from priors, which is a silent accuracy
    failure rather than an error -- so this asserts the two halves diverge from
    the untaught baseline rather than trusting the wiring.
    """
    participant_models, draws = coax_population_models(
        trials, models, data_id=DATA_ID, seed=0
    )
    kwargs = dict(
        trials=trials,
        cognitive_model=models,
        data_repository=repository,
        dvs={"forward_accuracy": ["continuous"]},
        participant_models=participant_models,
        participant_draws=draws,
    )
    shared = run_coax_experiment_executor(
        mode="fitted_population", share_training_memory=True, **kwargs
    )
    split = run_coax_experiment_executor(
        mode="fitted_population", share_training_memory=False, **kwargs
    )

    def tested_accuracy(frame):
        rows = frame[(frame["phase"] == "testing") & (frame["tested_w_xai"] == True)]  # noqa: E712
        return rows["cognitive_correct_vs_ai"].mean()

    assert tested_accuracy(shared) != tested_accuracy(split), (
        "sharing training memory changed nothing, so the w/ XAI half was already "
        "being taught -- the guard is not testing what it claims"
    )


def test_plain_keyed_models_are_refused(trials):
    """One model per xai_type cannot hold two strategies, so the mode must refuse
    it rather than silently applying only one of the drawn person's halves."""
    plain = coax_models_for_trials(trials, by_tested_condition=False)
    with pytest.raises(ValueError, match="one model per tested half"):
        coax_population_models(trials, plain, data_id=DATA_ID, seed=0)


def test_the_draw_is_reproducible(trials, models):
    first = coax_population_models(trials, models, data_id=DATA_ID, seed=0)[1]
    same = coax_population_models(trials, models, data_id=DATA_ID, seed=0)[1]
    other = coax_population_models(trials, models, data_id=DATA_ID, seed=7)[1]

    def ids(draws):
        return {key: draw.fitted_participant_id for key, draw in draws.items()}

    assert ids(first) == ids(same)
    assert ids(first) != ids(other)
