"""CoXAM's cognitive parameters must match its reference driver.

The standard is ``CoXAM/rl_agents/meta_policy_strategy_dashboard.py``: the three
parameters it exposes via ``fixed_eval_params`` are exactly the ones
``CombinedStrategyPolicyEnv._sample_episode_params`` samples, i.e. the only
cognitive parameters free at evaluation time. Ranges come from the run config of
the checkpoint actually loaded, not from ``CombinedPolicyConfig``'s dataclass
defaults (no run used those).

These tests read the dashboard source and the run config directly, so drifting
from the reference configuration fails loudly rather than silently.
"""

import json
import re
from pathlib import Path

import pytest

from src.cognitive_models import default_cognitive_params
from src.cognitive_models.placeholder import (
    COXAM_COGNITIVE_PARAMS,
    COXAM_COUNTERFACTUAL_PARAMS,
    COXAM_FORWARD_PARAMS,
)
from src.experiment_planner.support import load_support_matrix

COXAM_ROOT = Path(__file__).resolve().parents[1] / "src" / "cognitive_models" / "CoXAM"
DASHBOARD = COXAM_ROOT / "rl_agents" / "meta_policy_strategy_dashboard.py"
META_POLICY_ENV = COXAM_ROOT / "rl_agents" / "meta_policy_strategy.py"

#: The regime the loaded meta-policy checkpoint was trained in.
TRAINED_CONFIG = (
    COXAM_ROOT
    / "outputs"
    / "combined_strategy_meta_policy"
    / "meta_policy_strategy_masked_explanation_history_demo"
    / "mixed"
    / "config.json"
)

EXPECTED_PARAMS = {"decision_noise", "memory_recall_threshold", "opportunity_cost"}


def _dashboard_eval_params() -> set[str]:
    """Parameter names the dashboard passes through ``fixed_eval_params``."""
    source = DASHBOARD.read_text()
    block = source.split("fixed_eval_params = {", 1)[1].split("}", 1)[0]
    return set(re.findall(r'"(\w+)":', block))


def test_forward_defaults_match_the_dashboards_eval_params_exactly():
    assert set(COXAM_FORWARD_PARAMS) == _dashboard_eval_params() == EXPECTED_PARAMS


def test_default_values_match_the_dashboards_own_defaults():
    source = DASHBOARD.read_text()
    block = source.split("fixed_eval_params = {", 1)[1].split("}", 1)[0]
    for name, value in COXAM_FORWARD_PARAMS.items():
        # e.g. float(payload.get("decision_noise", 0.4))
        match = re.search(rf'"{name}",\s*([\d.-]+)\s*\)', block)
        assert match, f"{name} not found in the dashboard's fixed_eval_params"
        assert float(match.group(1)) == pytest.approx(value)


def test_support_matrix_declares_both_task_parameter_sets():
    """Forward and counterfactual are separate agents with separate parameters."""
    declared = set(load_support_matrix()["cognitive_models"]["coxam"]["cognitive_params"])
    assert EXPECTED_PARAMS <= declared
    assert set(COXAM_COUNTERFACTUAL_PARAMS) <= declared


def test_default_params_are_trimmed_to_the_requested_task():
    forward = default_cognitive_params("coxam", tasks=["forward_simulation"])
    counterfactual = default_cognitive_params("coxam", tasks=["counterfactual_simulation"])

    assert set(forward) == set(COXAM_FORWARD_PARAMS)
    assert set(counterfactual) == set(COXAM_COUNTERFACTUAL_PARAMS)
    # decision_noise is the forward meta-policy's; it must not leak into CF.
    assert "decision_noise" not in counterfactual
    assert "counterfactual_overshoot_fraction" not in forward


@pytest.mark.parametrize("name", sorted(EXPECTED_PARAMS))
def test_declared_ranges_match_the_trained_run_config(name):
    """Ranges must match the config the loaded checkpoint was trained under.

    Deliberately NOT ``CombinedPolicyConfig``'s dataclass defaults, which no run
    used (``memory_recall_threshold`` is [-1.0, 2.0] in every run config vs.
    [-5.0, 2.0] in the dataclass), and NOT the parameter sweeps, which explore
    wider than the trained regime.
    """
    trained = json.loads(TRAINED_CONFIG.read_text())
    expected = [trained[f"{name}_min"], trained[f"{name}_max"]]

    support = load_support_matrix()
    assert support["cognitive_models"]["coxam"]["cognitive_params"][name]["range"] == expected


def test_every_default_sits_inside_its_declared_range():
    param_spec = load_support_matrix()["cognitive_models"]["coxam"]["cognitive_params"]
    for name, value in default_cognitive_params("coxam").items():
        low, high = param_spec[name]["range"]
        assert low <= value <= high, f"{name}={value} outside declared [{low}, {high}]"


def test_coxam_and_coax_defaults_do_not_overlap():
    """A shared default dict cannot serve both agents -- that was the old bug."""
    assert not set(default_cognitive_params("coxam")) & set(default_cognitive_params("coax"))


def test_coxam_declares_both_tasks_now_that_both_have_runners():
    tasks = load_support_matrix()["cognitive_models"]["coxam"]["tasks"]
    assert set(tasks) == {"forward_simulation", "counterfactual_simulation"}


def test_the_forward_meta_policy_still_has_no_counterfactual_path():
    """Counterfactual is a separate agent -- it must not creep into the forward env.

    If this fails, the two agents have been conflated somewhere.
    """
    for path in (DASHBOARD, META_POLICY_ENV):
        source = path.read_text()
        assert "counterfactual" not in source
        assert "recall_change" not in source
