# Reasoning Strategies Guide

## Overview

The reasoning strategies module provides a plugin-based architecture for cognitive agent reasoning strategies. All 12 strategies are now fully integrated with the unified memory module, supporting both exemplar-based (CoAX) and ACT-R-based (CoXAM) memory backends.

## Architecture

```
src/cognitive_models/
├── interface.py              # Abstract contracts
├── registry.py              # Plugin management system
├── forward/                 # Inference-only strategies
│   ├── __init__.py
│   ├── coax_forward_rs.py   # 4 CoAX strategies
│   └── coxam_forward_rs.py  # 3 CoXAM strategies
├── counterfactual/          # Counterfactual explanation strategies
│   ├── __init__.py
│   └── coxam_counterfactual_rs.py  # 5 CoXAM strategies
└── __init__.py              # Public API + registry initialization
```

## Available Strategies

### CoAX Forward Strategies (4)
- **SensitiveFeatures**: Focus on discriminative features via t-test
- **SalientFeatures**: Focus on high-magnitude explanation components
- **ImportanceCategorization**: Use explanation vectors for categorization
- **AttributionSum**: Sum top-k attributions for binary decisions

### CoXAM Forward Strategies (3)
- **LRCalculation**: Memory-based likelihood ratio via MC sampling
- **LRHeuristic**: Simplified LR heuristic using nearest exemplar
- **DTTraversal**: Stochastic decision tree path following

### CoXAM Counterfactual Strategies (5)
- **ZeroOutLRHeuristic**: Suggest features to zero-out to flip decision
- **ZeroOutLRDisplayed**: Use displayed LR for zero-out suggestions
- **ChangeDTPath**: Suggest feature changes to follow different DT path
- **RecallChanges**: Retrieve stored counterfactual changes from memory
- **MemoryBasedCF**: Use memory to synthesize alternative decisions

## API Integration with Memory Module

### Memory Module API
The strategies now use the actual `src.cognitive_models.memory` API:

```python
from src.cognitive_models.memory import UnifiedMemory, MemoryConfig, Exemplar
from src.cognitive_models.memory import euclidean_distance, normalize_probabilities

# Create memory with CoAX backend
config = MemoryConfig.coax_defaults()
memory = UnifiedMemory(config)

# Store exemplar
exemplar = Exemplar(
    label=1,
    features=np.array([0.5, 0.3, 0.2]),
    label_probs={1: 1.0},
    explanation_vector=np.array([])
)
memory.store("ex_1", exemplar)

# Retrieve with activation scores
results = memory.retrieve(query_exemplar, k=5)
# Returns: List[Tuple[key, activation_score, exemplar]]

# Access backend
exemplar_backend = memory.get_exemplar_memory()
all_exemplars = exemplar_backend.get_exemplars()  # Dict[str, Exemplar]
```

### Key Integration Points in Strategies

#### 1. Exemplar Access Pattern
```python
# Get exemplars stored in memory
exemplar_backend = self.memory.get_exemplar_memory()
if exemplar_backend:
    exemplars_dict = exemplar_backend.get_exemplars()
else:
    exemplars_dict = {}

exemplars_list = list(exemplars_dict.values())
```

#### 2. Retrieve with Activation Scores
```python
# Create query exemplar
query = Exemplar(
    label=0,
    features=query_features,
    label_probs={},
    explanation_vector=np.array([])
)

# Retrieve returns (key, activation_score, exemplar) tuples
results = self.memory.retrieve(query, k=5)
for key, activation_score, exemplar in results:
    # activation_score incorporates memory activation
    similarity = np.exp(-self.sensitivity * dist) * (1.0 + activation_score)
```

#### 3. Utility Functions
```python
from src.cognitive_models.memory import (
    euclidean_distance,
    normalize_probabilities,
    temporal_decay,
    base_level_learning
)

# Distance computation
dist = euclidean_distance(features1, features2)

# Probability normalization
probs = normalize_probabilities(label_strengths)
# Converts {0: 10, 1: 20} → {0: 0.333, 1: 0.667}
```

## Usage Examples

### Example 1: Creating and Using SensitiveFeatures

```python
from src.cognitive_models import SensitiveFeatures, StrategyConfig, StrategyType
from src.cognitive_models.memory import Exemplar
import numpy as np

# Create strategy
config = StrategyConfig(
    strategy_name="sensitive_features",
    strategy_type=StrategyType.COAX_FORWARD,
    extra_params={'sensitivity': 10.0, 'k': 3}
)
strategy = SensitiveFeatures(config)

# Make inference
features = {'age': 0.5, 'income': 0.3, 'credit': 0.8}
explanation = [0.1, 0.2, 0.7]
probs, time_cost, info = strategy.infer(features, explanation)
# Returns: {0: 0.3, 1: 0.7}, 0.15, {...}

# Provide feedback (learn from true label)
true_label = 1
time = strategy.feedback(features, true_label, explanation)

# Signal trial boundary (stores last inference as exemplar)
strategy.new_instance()
```

### Example 2: Using Registry for Dynamic Strategy Selection

```python
from src.cognitive_models import StrategyRegistry, initialize_strategies, StrategyConfig, StrategyType

# Initialize all strategies
initialize_strategies()

# List all strategies
all_strategies = StrategyRegistry.list_strategies()
print(f"Available: {list(all_strategies.keys())}")

# List by type
coax_strategies = StrategyRegistry.list_by_type(StrategyType.COAX_FORWARD)
cf_strategies = StrategyRegistry.list_by_type(StrategyType.COXAM_COUNTERFACTUAL)

# Get strategy by name
config = StrategyConfig(
    strategy_name="lr_calculation",
    strategy_type=StrategyType.COXAM_FORWARD,
    extra_params={'sensitivity': 2.0}
)
strategy = StrategyRegistry.get("lr_calculation", config)
```

### Example 3: Counterfactual Explanation

```python
from src.cognitive_models import ZeroOutLRHeuristic, StrategyConfig, StrategyType

config = StrategyConfig(
    strategy_name="zeroout_lr_heuristic",
    strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
    extra_params={'k': 2}
)
strategy = ZeroOutLRHeuristic(config)

# Forward inference
features = {'age': 0.5, 'income': 0.8}
probs, time, _ = strategy.infer(features)

# Generate counterfactual
suggestion = strategy.suggest_change(
    features=features,
    explanation=[0.1, 0.9],
    current_prediction=1,
    target_label=0
)
print(f"Changed indices: {suggestion['changed_indices']}")
print(f"Suggested features: {suggestion['suggested_features']}")
print(f"Expected label: {suggestion['expected_label']}")
```

## Configuration Parameters

### StrategyConfig
- `strategy_name`: Unique strategy identifier
- `strategy_type`: StrategyType enum (COAX_FORWARD, COXAM_FORWARD, COXAM_COUNTERFACTUAL)
- `mode`: ReasoningMode (RETRIEVE, READ, HEURISTIC) - default: RETRIEVE
- `decay_param`: Memory decay rate (0.1-1.0) - default: 0.5
- `retrieval_threshold`: Min activation to retrieve - default: -2.5
- `sensitivity`: Feature sensitivity scaling (1.0-100.0) - default: 10.0
- `time_manager`: Optional time tracking object
- `extra_params`: Dict of strategy-specific parameters

### Extra Parameters by Strategy

#### SensitiveFeatures
```python
extra_params={
    'sensitivity': 10.0,  # Feature weighting
    'k': 3                # Number of features to focus
}
```

#### LRCalculation
```python
extra_params={
    'activation_noise': 0.1,      # Stochastic activation
    'noise_variance': 0.02,       # Noise std dev
    'ddm_v': 1.0,                 # Evidence model drift
    'ddm_a': 0.5                  # Evidence boundary
}
```

#### ZeroOutLRHeuristic
```python
extra_params={
    'k': 2,                       # Features to change
    'importance_scaling': 1.0     # Scale importance weights
}
```

## Integration with Unified Memory

### Memory Backend Selection
```python
from src.cognitive_models.memory import MemoryConfig, MemoryBackend, UnifiedMemory

# For CoAX strategies: Use ExemplarMemory backend
coax_config = MemoryConfig(
    backend=MemoryBackend.EXEMPLAR,
    decay_param=0.5,
    feature_similarity="euclidean"
)

# For CoXAM strategies: Use ACTRMemory backend
coxam_config = MemoryConfig(
    backend=MemoryBackend.ACTR,
    decay_param=0.5,
    wm_capacity=4,
    retrieval_threshold=-2.0
)

memory = UnifiedMemory(coax_config)
```

### Automatic Memory Initialization in Strategies
Each strategy automatically creates appropriate memory:
- **CoAX strategies**: `UnifiedMemory(MemoryConfig.coax_defaults())`
- **CoXAM forward**: `UnifiedMemory(MemoryConfig.coxam_defaults())`
- **CoXAM counterfactual**: `UnifiedMemory(MemoryConfig.coxam_defaults())`

Memory configuration is inherited from `StrategyConfig.decay_param` and `extra_params`.

## Implementation Details

### Memory Model Integration

#### CoAX Strategies (Exemplar-Based)
- Uses `ExemplarMemory` backend from memory module
- Stores decisions as exemplars with `label_probs`
- Retrieves via feature similarity with temporal decay
- Activation includes temporal decay factor

#### CoXAM Strategies (ACT-R-Based)
- Uses `ACTRMemory` backend from memory module
- Stores decisions as chunks or exemplars with slotsg
- Retrieves via BLL activation + spreading activation
- Latency computed from activation (DDM model)

### Retrieve Method Signature
```python
# All strategies use this signature
results = memory.retrieve(query, k=5)
# Returns: List[Tuple[str, float, Union[Exemplar, Chunk]]]
# Tuple: (key, activation_score, memory_item)
```

### Exemplar Storage Pattern
```python
exemplar = Exemplar(
    label=true_label,                    # Decision label
    features=np.array(feature_values),   # Feature vector
    label_probs={true_label: 1.0},      # Hard label
    explanation_vector=np.array([])      # Optional explanations
)
memory.store(f"ex_{memory.get_size()}", exemplar)
```

## State Management

### Persistence
```python
# Export strategy state
state = strategy.get_state()
# Returns: {'memory_size': int, ...}

# Export complete memory
memory_state = strategy.memory.export_state()

# Restore later
strategy.memory.import_state(memory_state)
```

### Trial Boundary
```python
# Signal end of trial - finalizes learning from last inference
strategy.new_instance()
```

## Next Steps

### Create Orchestrators
```python
# ForwardOrchestrator: Coordinate forward inference across strategies
# CounterfactualOrchestrator: Generate counterfactuals
# These would route infer()/suggest_change() to appropriate strategies
```

### Build API Gateway
```python
@app.post("/api/infer")
def infer(features, strategy_name="sensitive_features", **kwargs):
    strategy = StrategyRegistry.get(strategy_name, config)
    return strategy.infer(features, **kwargs)

@app.post("/api/counterfactual")
def counterfactual(features, strategy_name="zeroout_lr_heuristic"):
    strategy = StrategyRegistry.get(strategy_name, config)
    return strategy.suggest_change(features, **kwargs)
```

## Troubleshooting

### Issue: Memory import errors
**Solution**: Ensure `src/cognitive_models/memory/` files exist and are compiled:
```bash
python3 -m py_compile src/cognitive_models/memory/*.py
```

### Issue: Strategy not registering
**Solution**: Call `initialize_strategies()` before accessing registry:
```python
from src.cognitive_models import initialize_strategies, StrategyRegistry
initialize_strategies()
```

### Issue: Memory size growing unbounded
**Solution**: Call `strategy.new_instance()` to batch store memories:
```python
for trial in trials:
    probs, _, _ = strategy.infer(trial.features)
    strategy.feedback(trial.features, trial.label)
    strategy.new_instance()  # Consolidate memory
```

## References

- [Memory Consolidation Guide](MEMORY_CONSOLIDATION_GUIDE.md) - Unified memory API
- [src/cognitive_models/memory/interface.py](src/cognitive_models/memory/interface.py) - Memory contracts
- [src/cognitive_models/interface.py](src/cognitive_models/interface.py) - Strategy contracts
