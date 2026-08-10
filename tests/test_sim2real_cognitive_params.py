"""A sim2real design must get sim2real's parameters, not CoAX's.

``default_cognitive_params`` used to return the CoAX-shaped set for anything
that was not coxam, so a UI-authored sim2real study carried ACT-R encoding and
drift-diffusion timing parameters (``cog_T_enc``, ``cog_ddm_a``, ...) into a
model that has no timing component and accepts none of those names.
"""

import pytest

from src.cognitive_models import default_cognitive_params
from src.cognitive_models.placeholder import SIM2REAL_COGNITIVE_PARAMS
from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
    FITTED_SIM2REAL_PARAMS,
    NEUTRAL_SIM2REAL_PARAMS,
    SIM2REAL_EXP_PROPERTIES,
    build_sim2real_model,
    fitted_sim2real_params,
)


def test_sim2real_gets_its_own_parameters():
    assert default_cognitive_params("sim2real") == SIM2REAL_COGNITIVE_PARAMS


@pytest.mark.parametrize("alias", ["sim2real", "Sim2Real", "SIM2REAL", "sim-2-real".replace("-", "")])
def test_the_name_is_normalized(alias):
    assert default_cognitive_params(alias) == SIM2REAL_COGNITIVE_PARAMS


def test_no_coax_timing_parameters_leak_in():
    """The specific bug: cog_* names reaching a model with no timing component."""
    params = default_cognitive_params("sim2real")
    assert not [name for name in params if name.startswith("cog_")]


def test_every_default_is_a_parameter_the_model_accepts():
    assert set(SIM2REAL_COGNITIVE_PARAMS) <= set(NEUTRAL_SIM2REAL_PARAMS)


def test_the_defaults_do_not_overwrite_the_per_condition_fit():
    """The reason only half the model's parameters are returned.

    ``fitted_sim2real_params`` lets explicit values win over the fitted table,
    so a default carrying comparison_scale=1.0 would replace every condition's
    fitted scale with the neutral one -- and at that scale a typical
    confidence_delta of ~1e-4 maps to p~0.5, i.e. the model answers
    "increases" on every trial by tie-break.
    """
    defaults = default_cognitive_params("sim2real")
    for exp_property in SIM2REAL_EXP_PROPERTIES:
        fitted = FITTED_SIM2REAL_PARAMS[exp_property]
        assert not set(defaults) & set(fitted), exp_property
        merged = fitted_sim2real_params(exp_property, **defaults)
        for name, value in fitted.items():
            assert merged[name] == value, (exp_property, name)


def test_the_defaults_build_a_model_for_every_condition():
    defaults = default_cognitive_params("sim2real")
    for exp_property in SIM2REAL_EXP_PROPERTIES:
        params = fitted_sim2real_params(exp_property, **defaults)
        assert build_sim2real_model(params, exp_property=exp_property) is not None


def test_the_three_agents_take_disjoint_parameters():
    """One shared default dict cannot be valid for any two of them."""
    sim2real = set(default_cognitive_params("sim2real"))
    coxam = set(default_cognitive_params("coxam"))
    coax = set(default_cognitive_params("coax"))
    assert not sim2real & coax
    assert not sim2real & coxam


def test_unknown_agents_still_get_the_coax_shaped_placeholder_set():
    """Unchanged behaviour for the placeholder and CoAX paths."""
    assert default_cognitive_params(None) == default_cognitive_params("coax")
    assert "cog_retrieval_threshold" in default_cognitive_params(None)


def test_defaults_come_from_the_fitted_population_not_the_constructor():
    """Same choice FITTED_COAX_PARAMS makes: fitted values, not arbitrary ones."""
    import inspect

    from src.cognitive_models.cognitive_models.Sim2Real.gcm_strategies import (
        Sim2RealFittedAttributionSum,
    )

    constructor = {
        name: parameter.default
        for name, parameter in inspect.signature(
            Sim2RealFittedAttributionSum.__init__
        ).parameters.items()
    }
    differs = {
        name: (value, constructor[name])
        for name, value in SIM2REAL_COGNITIVE_PARAMS.items()
        if name in constructor and value != constructor[name]
    }
    assert differs, "defaults are identical to the constructor's, so nothing was fitted"
    assert "memory_sensitivity" in differs


def test_memory_sensitivity_is_the_mean_over_memory_on_participants():
    """Pooling all 46 would drag it toward the placeholder 1.0 that
    memory-off candidates carry but never read."""
    assert SIM2REAL_COGNITIVE_PARAMS["memory_sensitivity"] == pytest.approx(3.2143, abs=1e-4)
    assert SIM2REAL_COGNITIVE_PARAMS["memory_sensitivity"] > 1.0
