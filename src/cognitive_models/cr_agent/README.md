"""
CR Agent - Cognitive Reasoning Agent RL System

A unified API for strategy selection in forward and counterfactual
reasoning about AI predictions and explanations.

CRITICAL FINDINGS:

1. ⚠️  TWO SEPARATE META ROUTERS (NOT identical)
   - Forward Router: run_meta_on_batch() [function-based]
     * Observation: 8 + 4*S dims (trial_type one-hot encoding)
     * Action: Single strategy index
     * Reward: correctness - time*chi
   - Counterfactual Router: CounterfactualMetaRouter [Gym environment]
     * Observation: 5 + 3*S + varied_params dims (xai_type encoding)
     * Action: (strategy_idx, depth) tuple
     * Reward: success - time*chi
   → Both share meta PPO weights but have different implementations
   → Use appropriate router based on simulation type

2. ⚠️  HEADLESS POLICIES vs COUNTERFACTUAL STRATEGIES
   - Forward: Uses trained headless policies (HeadlessDTPolicy, HeadlessLRCalcPolicy, HeadlessLRHeurPolicy)
   - Counterfactual: Uses cognitive strategies from cognitive_models layer
     * Imported from: src.cognitive_models.counterfactual
     * Strategies: ZeroOutLRHeuristic, ZeroOutLRDisplayed, ChangeDTPath, RecallChanges, MemoryBasedCF
     * Wrapped by CounterfactualStrategyWrapper for interface compatibility
   - These are NOT interchangeable - each simulation type uses its own strategy set
   → Strategies are task-specific but wrapped in common interface

3. ✅ SHARED PATTERNS
   - Both routers use strategy interface (reset/step contract)
   - Both use same meta PPO model weights
   - Both track per-strategy statistics
   - Both support condition-based strategy gating

DIRECTORY STRUCTURE:

cr_agent/
├── agents/
│   ├── headless_policies.py      # Forward: DT, LR Calc, LR Heur policies
│   └── counterfactual_strategies.py # Wrappers for cognitive_models.counterfactual
├── environments/
│   ├── meta_router_env.py        # Forward: run_meta_on_batch() function
│   ├── counterfactual_meta_router.py # Counterfactual: CounterfactualMetaRouter gym env
│   └── __init__.py
├── weights/                       # Pre-trained model weights
│   ├── model_calculation/
│   ├── model_counterfactual/
│   ├── model_dt/
│   ├── model_heuristic/
│   ├── models_meta/
│   └── ...
├── interface.py                   # Public API (CRAgentRunner, MetaRunner)
├── registry.py                    # Agent/env registration and metadata
├── __init__.py                   # Package exports
└── README.md                      # This file

USAGE:

## Forward Simulation (using run_meta_on_batch function)
```python
from src.cognitive_models.cr_agent import CRAgentRunner

runner = CRAgentRunner(
    meta_model_path="./weights/models_meta/best/best_model.zip",
    dt_model_path="./weights/model_dt/simple_chi_model.zip",
    lr_calc_model_path="./weights/model_calculation/simple_chi_model.zip",
    lr_heur_model_path="./weights/model_heuristic/simple_chi_model.zip",
)

results = runner.run_forward_episode(
    X_raw=X_raw,
    y_raw=y_raw,
    X_norm=X_norm,
    condition="LR",
    chi_value=0.01,
)
```

## Counterfactual Simulation (using CounterfactualMetaRouter gym environment)
```python
from src.cognitive_models.cr_agent import (
    CounterfactualMetaRouter,
    load_counterfactual_strategies,
)

# Create router
routes = CounterfactualMetaRouter(
    meta_model=meta_ppo,
    strategies=load_counterfactual_strategies(),
    X_raw=X_raw,
    y_raw=y_raw,
    instances_per_episode=40,
    condition="DT+LR",
    chi_spec=(0.0, 0.05),
)

# Standard gym interface
obs, info = router.reset(seed=42)
for _ in range(40):
    action = meta_ppo.predict(obs)[0]
    obs, reward, terminated, truncated, info = router.step(action)
    if truncated:
        break
```

## Manual Forward Meta Episode (low-level)
```python
from src.cognitive_models.cr_agent import run_meta_on_batch

result = run_meta_on_batch(
    meta_model=meta_ppo,
    strategies={"dt": policy_dt, "lr_calc": policy_lr, "lr_heur": policy_heur},
    X_raw=X_raw,
    y_raw=y_raw,
    condition="DT",
    chi_value=0.01,
)
```

API REFERENCE:

### Core Classes

**CRAgentRunner**
- Full simulation runner with agent loading
- Methods: run_forward_episode(), run_counterfactual_episode()

**MetaRunner**
- Meta episode executor with pre-loaded strategies
- Methods: run_episode()

**HeadlessDTPolicy, HeadlessLRCalcPolicy, HeadlessLRHeurPolicy**
- Forward reasoning policies (trained PPO)
- Methods: reset(), step()

**CounterfactualStrategyWrapper** (from cognitive_models layer)
- Adapts cognitive_models.counterfactual strategies to cr_agent interface
- Wrapped strategies: ZeroOutLRHeuristic, ZeroOutLRDisplayed, ChangeDTPath, RecallChanges, MemoryBasedCF
- Methods: reset(), step()
- Factory function: load_counterfactual_strategies()

### Registry

**AgentRegistry**
- Methods: register(), get(), list_all(), list_forward(), list_counterfactual()

**EnvironmentRegistry**
- Methods: register(), get(), list_all()

COGNITIVE PARAMETERS:

Forward (training_cog_params):
- retrieval_threshold: [-2.0, 0.5]
- latency_factor: [0.0, 0.5]
- ddm_a, ddm_s: diffusion model parameters
- chi: [0.0, 0.02] (time cost)

Counterfactual:
- retrieval_threshold: [-2.0, 0.5]
- latency_factor: 1.0 (fixed)
- lapse: [0.1, 0.5]
- over_margin: [0.0, 0.5]
- chi: [0.0, 0.05] (time cost)

COMPARISON REPORT:

✅ IDENTICAL COMPONENTS
- Meta environment structure (run_meta_on_batch)
- Observation encoding (chi_norm, trial_norm, with_xai, conditions, trial types, stats)
- Reward computation (correctness - chi * time)
- Strategy ordering and action decoding
- Per-strategy statistics tracking (with_xai / without_xai)

⚠️  DIFFERENT COMPONENTS
- Strategy implementations (forward policies vs CF functions)
- Trial logic (forward vs counterfactual trial execution)
- Cognitive parameters (some differences in chi ranges)

✅ UNIFIED INTERFACE
- All strategies implement reset() and step() contract
- All return (output, time, info)
- MetaRouterEnvironment orchestrates transparently
