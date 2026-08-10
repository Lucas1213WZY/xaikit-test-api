"""_merged_cognitive_config must not let an untouched slider's raw
modelDefault silently override a fitted/recommended default.

Regression: a design export flags an untouched cognitiveParameters entry with
source="model default" and sets value == modelDefault (not None, the
documented "use the model default" sentinel). Blindly trusting any non-None
value meant every unconfigured Sim2Real run silently used the strategy
class's neutral, uncalibrated defaults (confidence_intercept=0, aggregation=
"attribution", max_features_attended=12 -- NEUTRAL_SIM2REAL_PARAMS, which the
model's own docstring calls degenerate) instead of the fitted population
values, for every parameter a UI export like this one carries.
"""

from __future__ import annotations

from src.experiment_planner.design_export import _merged_cognitive_config


def test_an_untouched_model_default_does_not_override_the_fitted_default():
    raw = {
        "cognitiveParameters": [
            {
                "key": "max_features_attended",
                "value": 12,
                "modelDefault": 12,
                "recommendedDefault": 4,
                "source": "model default",
            },
            {
                "key": "aggregation",
                "value": "attribution",
                "modelDefault": "attribution",
                "recommendedDefault": "value_weighted",
                "source": "model default",
            },
        ]
    }
    assert _merged_cognitive_config(raw) == {}


def test_a_genuinely_customized_parameter_still_overrides():
    raw = {
        "cognitiveParameters": [
            {
                "key": "max_features_attended",
                "value": 6,
                "modelDefault": 12,
                "recommendedDefault": 4,
                "source": "user",
            },
        ]
    }
    assert _merged_cognitive_config(raw) == {"max_features_attended": 6}


def test_a_value_that_happens_to_match_model_default_but_has_no_source_flag_still_wins():
    """Older/simpler export shapes carry no source/modelDefault at all --
    those values are genuine (there is nothing to compare against), so they
    must not be dropped."""
    raw = {"cognitiveParameters": [{"key": "max_features_attended", "value": 12}]}
    assert _merged_cognitive_config(raw) == {"max_features_attended": 12}


def test_the_old_cognitive_config_dict_shape_is_unaffected():
    raw = {"cognitiveConfig": {"Max Features Attended": "6"}}
    assert _merged_cognitive_config(raw) == {"Max Features Attended": "6"}


def test_cognitive_parameters_still_win_over_cognitive_config_on_a_shared_key():
    raw = {
        "cognitiveConfig": {"max_features_attended": "6"},
        "cognitiveParameters": [
            {
                "key": "max_features_attended",
                "value": 8,
                "modelDefault": 12,
                "source": "user",
            }
        ],
    }
    assert _merged_cognitive_config(raw) == {"max_features_attended": 8}
