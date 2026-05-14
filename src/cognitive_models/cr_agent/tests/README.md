# CR_AGENT Test Suite Documentation

## Overview

The `src/cognitive_models/cr_agent/tests/` directory contains comprehensive integration tests for the Cognitive Router (CR) Agent layer. Tests validate:

1. **Forward Meta Router** — Forward meta episode execution with strategy selection
2. **Counterfactual Meta Router** — Counterfactual meta episode execution with gym interface
3. **Strategy Integration** — CounterfactualStrategyWrapper and factory pattern
4. **End-to-End API** — CRAgentRunner public interface for both forward and counterfactual

## Test Files

### 1. `run_standalone_tests.py`
Simple standalone test runner (no pytest required)

**Purpose:** Quickly validate that all modules can be imported and basic functionality works

**Run:**
```bash
cd /Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python src/cognitive_models/cr_agent/tests/run_standalone_tests.py
```

**Output:**
```
5/5 tests passed:
  ✅ Module Imports - All cr_agent modules load correctly
  ✅ Strategy Factory - load_counterfactual_strategies() creates 5 wrapped strategies
  ✅ Strategy Wrapper - CounterfactualStrategyWrapper wraps cognitive_models correctly
  ✅ Registry Presets - Cognitive parameters defined for forward and counterfactual
  ✅ Interface Classes - CRAgentRunner and MetaRunner have required methods
```

### 2. `test_forward_router.py`
Integration tests for forward meta router with actual model weights

**Test Classes:**
- `TestForwardMetaRouter` — Basic forward router tests
- `TestForwardMetaRouterWithWeights` — Tests using actual PPO meta model

**Test Methods:**
- `test_meta_router_basic_forward_dt` — Basic DT condition execution
- `test_meta_router_lr_condition` — LR condition strategy gating
- `test_meta_router_dtlr_condition` — Mixed DT+LR condition
- `test_observation_shape_consistency` — Validates 8+4*3=20 dim
- `test_reward_calculation` — Verifies `reward = pr - chi*time`
- `test_with_xai_schedule` — XAI availability scheduling
- `test_chi_normalization` — Chi value normalization

**Run (requires pytest and conda environment):**
```bash
cd /Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pip install pytest
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pytest src/cognitive_models/cr_agent/tests/test_forward_router.py -v
```

### 3. `test_counterfactual_router.py`
Integration tests for counterfactual meta router with gym interface

**Test Classes:**
- `TestCounterfactualMetaRouter` — Gym environment interface tests
- `TestCounterfactualWithWeights` — Tests using actual PPO meta model

**Test Methods:**
- `test_router_initialization` — Gym environment setup
- `test_reset_returns_valid_obs` — Reset returns valid observation
- `test_step_returns_valid_tuple` — Step returns (obs, reward, terminated, truncated, info)
- `test_episode_loop_dt_condition` — Complete DT condition episode
- `test_episode_loop_lr_condition` — Complete LR condition episode
- `test_episode_loop_dtlr_condition` — Complete DT+LR condition episode
- `test_observation_bounds` — Observations stay within space bounds
- `test_strategy_statistics_tracking` — Per-strategy statistics collected
- `test_reward_calculation_formula` — Verifies `reward = success - time*chi`
- `test_deterministic_seeding` — Same seed reproducible results

**Run:**
```bash
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pytest src/cognitive_models/cr_agent/tests/test_counterfactual_router.py -v
```

### 4. `test_end_to_end.py`
End-to-end tests for public API with actual weights

**Test Classes:**
- `TestCRAgentRunner` — End-to-end API tests
- `TestCRAgentRunnerWithWeights` — Tests using actual PPO meta model

**Test Methods:**
- `test_runner_initialization` — CRAgentRunner setup
- `test_run_forward_episode_dt_condition` — Forward episode DT
- `test_run_forward_episode_lr_condition` — Forward episode LR
- `test_run_forward_episode_dtlr_condition` — Forward episode DT+LR
- `test_run_counterfactual_episode_dt_condition` — Counterfactual episode DT
- `test_run_counterfactual_episode_lr_condition` — Counterfactual episode LR
- `test_run_counterfactual_episode_dtlr_condition` — Counterfactual episode DT+LR
- `test_multiple_episodes_same_runner` — Stability with repeated episodes
- `test_episode_statistics` — Episode statistics computation
- `test_different_chi_specs` — Different chi specifications
- `test_forward_vs_counterfactual_consistency` — Both routers use same data
- `test_episode_reproducibility_with_seed` — Seeding consistency

**Run:**
```bash
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pytest src/cognitive_models/cr_agent/tests/test_end_to_end.py -v
```

### 5. `test_strategies_integration.py`
Tests for strategy layer integration with cognitive_models

**Test Classes:**
- `TestCounterfactualStrategyWrapper` — Wrapper functionality
- `TestCreateCounterfactualStrategies` — Factory pattern
- `TestStrategyIntegrationWithRouters` — Integration with routers

**Test Methods:**
- `test_wrapper_initialization` — Wrapper setup
- `test_wrapper_output_format` — Wrapper returns proper tuple
- `test_wrapper_converts_confidence_to_success` — Confidence to probability conversion
- `test_wrapper_handles_different_conditions` — Works with All/DT/LR/DT+LR
- `test_factory_returns_dict` — Factory returns dict of strategies
- `test_factory_strategy_names` — Expected strategies present
- `test_factory_strategy_callability` — All strategies have suggest_change
- `test_factory_strategy_works_in_sequence` — Sequential strategy execution
- `test_factory_consistency` — Factory produces consistent instances
- `test_strategies_integrate_with_observation` — Observation encoding
- `test_strategies_compatible_with_forward_router` — Forward router compatibility
- `test_strategies_compatible_with_counterfactual_router` — CF router compatibility

**Run:**
```bash
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pytest src/cognitive_models/cr_agent/tests/test_strategies_integration.py -v
```

## Test Infrastructure

### Fixtures

#### `weights_dir` (Forward, Counterfactual, E2E tests)
Path to weights directory at `src/cognitive_models/cr_agent/weights/`

#### `sample_data` (All tests needing data)
Generated randomly seeded dataset (N=50-200, F=10)
- `X_raw`: Raw features
- `X_norm`: Normalized features
- `y_raw`: Binary labels (0, 1)

#### `meta_model` (Forward, Counterfactual, E2E tests)
Loads actual PPO meta model from `weights_dir/models_meta/best/best_model.zip`

#### `cf_strategies` (Counterfactual, E2E tests)
Creates wrapped counterfactual strategies via `load_counterfactual_strategies()`

#### `runner` (E2E tests)
CRAgentRunner instance initialized with meta model and data

### Environment

Tests require:
- Python 3.10+ (3.10.18 in `rlnb_ibl_env`)
- stable-baselines3 (PPO agent loading)
- gymnasium (gym environment interface)
- numpy
- pytest (optional, for organized test running)

**Conda environment:** `/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/`

## Running Tests

### Quick Validation (No pytest required)
```bash
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python src/cognitive_models/cr_agent/tests/run_standalone_tests.py
```
Expected: All 5 tests pass in ~5 seconds

### Full Test Suite (Requires pytest)
```bash
# Install pytest if needed
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pip install pytest

# Run all tests
cd /Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pytest src/cognitive_models/cr_agent/tests/ -v

# Run specific test file
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pytest src/cognitive_models/cr_agent/tests/test_forward_router.py -v

# Run with coverage
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pytest src/cognitive_models/cr_agent/tests/ --cov=src.cognitive_models.cr_agent
```

### Individual Module Testing
```bash
# Test just forward router
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pytest src/cognitive_models/cr_agent/tests/test_forward_router.py::TestForwardMetaRouter::test_meta_router_basic_forward_dt -v

# Test just strategies
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pytest src/cognitive_models/cr_agent/tests/test_strategies_integration.py -v
```

## Test Coverage

### Module Coverage
- ✅ `src/cognitive_models/cr_agent/agents/headless_policies.py` — Forward policies (7 test methods)
- ✅ `src/cognitive_models/cr_agent/agents/counterfactual_strategies.py` — CF strategies (6 test methods)
- ✅ `src/cognitive_models/cr_agent/environments/meta_router_env.py` — Forward router (7 test methods)
- ✅ `src/cognitive_models/cr_agent/environments/counterfactual_meta_router.py` — CF router (10 test methods)
- ✅ `src/cognitive_models/cr_agent/interface.py` — Public API (13 test methods)
- ✅ `src/cognitive_models/cr_agent/registry.py` — Registry (2 test methods)

### Component Coverage
- **Forward Router:** Observation shape, reward calculation, XAI scheduling, chi normalization
- **Counterfactual Router:** Gym interface, observation encoding, action validation, reward formula
- **Strategies:** Factory pattern, wrapper adaptation, compatibility with routers
- **Public API:** Episode execution, multiple episodes, reproducibility, consistency
- **Integration:** Forward + counterfactual with same data, end-to-end workflows

### Test Types
- **Unit Tests:** Individual component behavior
- **Integration Tests:** Component interactions (routers + strategies)
- **End-to-End Tests:** Complete workflows (CRAgentRunner)
- **Fixture-Based Tests:** Reproducible testing with shared resources

## Expected Results

### Standalone Tests
```
✅ Module Imports
✅ Strategy Factory
✅ Strategy Wrapper
✅ Registry
✅ Interface Classes
Total: 5/5 passed
```

### Forward Router Tests (~30 seconds, requires weights)
```
7 tests from TestForwardMetaRouter
  - test_meta_router_basic_forward_dt PASSED
  - test_meta_router_lr_condition PASSED
  - test_meta_router_dtlr_condition PASSED
  - test_observation_shape_consistency PASSED
  - test_reward_calculation PASSED
  - test_with_xai_schedule PASSED
  - test_chi_normalization PASSED
```

### Counterfactual Router Tests (~30 seconds, requires weights)
```
10 tests from TestCounterfactualMetaRouter
  - All tests PASSED
```

### End-to-End Tests (~60 seconds, requires weights)
```
12 tests from TestCRAgentRunner
  - All tests PASSED
```

### Strategies Integration Tests (~20 seconds)
```
12 tests from TestCounterfactualStrategyWrapper
TestCreateCounterfactualStrategies
TestStrategyIntegrationWithRouters
  - All tests PASSED
```

## Known Limitations

1. **Pytest Required for Full Suite** — Standalone tests are quick but limited
2. **Weights Required** — Meta model weights at `src/cognitive_models/cr_agent/weights/models_meta/best/best_model.zip` needed for full integration tests
3. **No CLI Tests** — Tests only validate library API, not command-line interfaces
4. **Deterministic Seeding** — Some randomness in strategy selection (by design)

## Adding New Tests

**Pattern for new tests:**

```python
def test_new_feature(self, meta_model, cf_strategies, sample_data):
    """Test description."""
    # Setup
    router = CounterfactualMetaRouter(
        meta_model=meta_model,
        strategies=cf_strategies,
        X_raw=sample_data["X_raw"],
        y_raw=sample_data["y_raw"],
        instances_per_episode=20,
        condition="DT+LR",
        episode_cogs={"latency_factor": 0.2},
        training_cog_params=COGNITIVE_PARAMS_COUNTERFACTUAL,
        chi_spec=(0.0, 0.05),
        seed=42,
    )
    
    # Execute
    obs, _ = router.reset(seed=42)
    action = router.action_space.sample()
    obs, reward, _, _, info = router.step(action)
    
    # Assert
    assert router.observation_space.contains(obs)
    assert isinstance(reward, (float, np.number))
```

## Troubleshooting

### Tests Skip Due to Missing Weights
```
reason="Meta model weights not found"
```
**Solution:** Verify `src/cognitive_models/cr_agent/weights/` subdirectory exists with:
- `models_meta/best/best_model.zip`
- Other model directories

### ImportError: No module 'stable_baselines3'
```
ModuleNotFoundError: No module named 'stable_baselines3'
```
**Solution:** Use correct conda environment:
```bash
/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python -m pytest ...
```

### Observation Space Assertion Failures
```
AssertionError: Observation outside space bounds
```
**Likely Cause:** Different observation encoding between forward/counterfactual
**Check:** Verify obs dims match expected (8+4*3 for forward, 5+3*S+params for CF)

## Performance Targets

- **Standalone tests:** < 5 seconds
- **Forward router tests:** < 30 seconds
- **Counterfactual router tests:** < 30 seconds
- **End-to-end tests:** < 60 seconds
- **Full suite:** < 2 minutes
- **Single forward episode (50 instances):** < 500ms
- **Single CF episode (50 instances):** < 500ms

## Related Documentation

- [src/cognitive_models/cr_agent/README.md](../README.md) — CR Agent API overview
- [src/cognitive_models/cr_agent/COMPARISON_ANALYSIS.md](../COMPARISON_ANALYSIS.md) — Forward vs CF router comparison
- [src/cognitive_models/](../../cognitive_models/) — Strategy implementations
