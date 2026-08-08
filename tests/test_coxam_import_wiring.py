"""The copied CoXAM RL-agent scripts must import cleanly from their new package
location, with no sys.path manipulation and no accidental resolution against
this repo's own top-level `src` package (both modules use `from src.X import`
in their original standalone layout, which the port rewrites to `from ..X
import`)."""

import sys


def test_meta_policy_strategy_imports_without_sys_path_hacks():
    module_name = "src.cognitive_models.cognitive_models.coxam.rl_agents.meta_policy_strategy"
    sys.modules.pop(module_name, None)

    import src.cognitive_models.cognitive_models.coxam.rl_agents.meta_policy_strategy as mps

    assert mps.META_STRATEGIES == ("lr_calculation", "lr_heuristic", "dt_traversal")
    assert mps.CONDITIONS == ("linear_regression", "decision_tree", "hybrid")


def test_individual_policy_training_f_imports_without_sys_path_hacks():
    module_name = (
        "src.cognitive_models.cognitive_models.coxam.rl_agents.individual_policy_training_f"
    )
    sys.modules.pop(module_name, None)

    import src.cognitive_models.cognitive_models.coxam.rl_agents.individual_policy_training_f as ipt

    assert ipt.STRATEGIES == ("lr_calculation", "lr_heuristic", "dt_traversal")


def test_coxam_package_does_not_shadow_repo_src_package():
    """`from src.X import ...` inside the ported scripts would previously have
    resolved (or failed to resolve) against this repo's own top-level `src`
    package rather than CoXAM's. Confirm the fix actually uses relative
    imports by checking the source text directly."""
    from pathlib import Path

    coxam_rl_agents = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "cognitive_models"
        / "cognitive_models"
        / "coxam"
        / "rl_agents"
    )
    for filename in ("individual_policy_training_f.py", "meta_policy_strategy.py"):
        text = (coxam_rl_agents / filename).read_text()
        assert "from src." not in text, f"{filename} still imports from the repo's own src package"
        assert "from rl_agents." not in text, f"{filename} still assumes a top-level rl_agents package"
