# Unified Data Loader System

**Version**: 0.1.0

A comprehensive, extensible, plugin-based data loading layer that unifies access to CoAX synthetic data and CoXAM experiment data, with support for custom sources.

## Overview

The unified data loader system provides a **single, consistent API** for accessing data across different frameworks and sources, eliminating code duplication and enabling seamless extensibility.

### Key Features

- **🔌 Plugin Architecture**: Register custom data sources and normalizers
- **🎯 Unified API**: Same interface for CoAX, CoXAM, and custom data
- **🔗 Composable Filters**: Chain multiple filter conditions (`by_app()`, `by_participant()`, etc.)
- **📊 Built-in Normalizers**: Min-Max, Z-Score, and custom normalization strategies
- **🧠 XAI Integration**: Load explanation tables that `src.xai_adapter` can turn into explanation methods
- **♻️ Zero Duplication**: Single source of truth for all data loading logic

---

## Architecture

```
src/data_loaders/
├── __init__.py                    # Main exports
├── unified_loader.py              # Core API (UnifiedDataLoader)
│
├── base/                          # Abstract base classes
│   ├── data_source.py             # BaseDataSource
│   └── normalizer.py              # BaseNormalizer
│
├── sources/                       # Data source adapters
│   ├── coax_adapter.py            # CoAXDataSource
│   └── coxam_adapter.py           # CoXAMDataSource
│
├── normalizers/                   # Feature normalizers
│   ├── minmax.py                  # MinMaxNormalizer
│   └── zscore.py                  # ZScoreNormalizer
│
├── filters/                       # Composable filters
│   └── filter_builder.py          # FilterBuilder
│
└── examples.py                    # Comprehensive usage examples
```

---

## Quick Start

### 1. Install/Import

```python
from src.data_loaders import UnifiedDataLoader, FilterBuilder, XAIDatasetParser
from src.xai_adapter import create_xai_method
```

### 2. Load Data

```python
# CoAX synthetic data
loader = UnifiedDataLoader.from_coax(
    feature_file="assets/data/coax/values.csv",
    metadata_file="assets/data/coax/metadata.csv",
    prediction_file="assets/data/coax/none.csv"
)

# CoXAM experiment data
loader = UnifiedDataLoader.from_coxam(
    feature_file="assets/data/coxam/values.csv",
    metadata_file="assets/data/coxam/metadata.csv",
    prediction_file="assets/data/coxam/none.csv"
)
```

### 3. Access Data

```python
# Get instances with normalized features
features, predictions = loader.get_instances([0, 1, 2], normalize=True)

# Get features only
features = loader.get_features([0, 1, 2])

# Get predictions only
predictions = loader.get_predictions([0, 1, 2])

# Get explanations
explanations = loader.get_explanations([0, 1, 2])
```

### 4. Use Filters

```python
# Create composable filter
filter_builder = loader.filter()
filter_builder.by_app("wine_quality")
filter_builder.by_condition("LR")
filter_builder.by_xai_type("importance")

# Apply filter
loader.apply_filter(filter_builder)
```

### 5. Use XAI Methods

```python
# Get XAI method registry
from src.xai_adapter import get_adapter_registry
registry = get_adapter_registry()

# Create Decision Tree method
dt_exp = registry.create(
    'decision_tree',
    explanation_df=dt_df,
    metadata_df=metadata_df,
    app_id="wine_quality",
    model_name="mlp"
)

# Apply to instance
result = dt_exp.apply(instance_features)
print(result)  # {'probs': [...], 'class_index': 1, 'class_label': 'positive'}
```

---

## Core Components

### UnifiedDataLoader

Main API class providing factory methods and a consistent interface.

```python
# Factory methods
loader = UnifiedDataLoader.from_coax(...)
loader = UnifiedDataLoader.from_coxam(...)
loader = UnifiedDataLoader.from_custom(data_source)

# Core methods
features, predictions = loader.get_instances(ids, normalize=True)
loader.apply_filter(filter_builder)
from src.xai_adapter import get_adapter_registry
registry = get_adapter_registry()
summary = loader.get_summary()
```

### Data Sources

Adapters for different data sources implementing `BaseDataSource`.

**CoAXDataSource**
- Loads synthetic features, metadata, AI predictions
- Supports explanation columns
- Random sampling: `load_random(n_samples)`

**CoXAMDataSource**
- Loads experiment data + participant trials
- Methods: `get_participant_trials()`, `get_participant_ids()`
- Same feature/prediction API as CoAXDataSource

**Custom Sources**
- Inherit from `BaseDataSource`
- Implement: `load()`, `get_features()`, `get_predictions()`, `get_explanations()`

### XAI Methods

Registry-based system for model explanation methods.

CSV parsing for external XAI datasets lives in this layer:

```python
from src.data_loaders import XAIDatasetParser
from src.xai_adapter import create_xai_method

dataset = XAIDatasetParser.from_csv("assets/data/my_dataset_predictions.csv")
X = dataset.get_features([0, 1, 2])
y = dataset.get_predictions([0, 1, 2])

precomputed = create_xai_method("csv", dataset=dataset)
result = precomputed.explain([0, 1, 2])
```

**DecisionTreeSurrogateMethod**
- Loads tree structure from JSON
- Methods: `apply(instance)`, `print_tree(as_name=True)`

**LogisticRegressionSurrogateMethod**
- Complex unnormalization logic (continuous + categorical features)
- Methods: `apply(instance)`, `print_model(as_name=True)`
- Returns sigmoid probability

**Custom XAI Methods**
- Inherit from `XAIAdapter`
- Implement: `fit()` if setup is needed, then `explain()`

### Normalizers

Feature normalization strategies implementing `BaseNormalizer`.

**MinMaxNormalizer** (Default)
- Scales to [0, 1] range
- Clips values outside min-max bounds

**ZScoreNormalizer**
- Standardization: (x - mean) / std

**Custom Normalizers**
- Inherit from `BaseNormalizer`
- Implement: `normalize()`, `normalize_array()`

### Filters

Composable filter builder for data queries.

```python
filter_builder = FilterBuilder()
filter_builder.by_app("wine_quality")
filter_builder.by_participant(42)
filter_builder.by_condition("LR")
filter_builder.by_model("mlp")
filter_builder.by_xai_type("importance")
filter_builder.by_phase("forward")
filter_builder.by_custom(custom_function)

# Apply to data
filtered_df = filter_builder.apply(data_df)
```

---

## Advanced Usage

### Custom Data Source

```python
from src.data_loaders.base import BaseDataSource
from src.data_loaders import UnifiedDataLoader

class CustomDataSource(BaseDataSource):
    def __init__(self):
        super().__init__('custom')
    
    def load(self, **kwargs):
        # Your loading logic
        self.feature_values_df = ...
        self.metadata_df = ...
    
    def get_features(self, ids, normalize=True):
        # Your feature retrieval
        pass
    
    def get_predictions(self, ids):
        # Your prediction retrieval
        pass
    
    # ... implement other abstract methods

# Use it
source = CustomDataSource()
source.load(custom_param="value")
loader = UnifiedDataLoader.from_custom(source)
```

### Custom XAI Method

```python
import numpy as np

from src.xai_adapter import XAIAdapter, XAIAdapterResult
from src.data_loaders import UnifiedDataLoader

class CustomMethod(XAIAdapter):
    method_name = "custom"

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def fit(self, X=None, y=None, **kwargs):
        self.is_fitted = True
        return self

    def explain(self, instances):
        instances = np.asarray(instances)
        return XAIAdapterResult(
            values=np.zeros_like(instances, dtype=float),
            base_values=np.zeros(len(instances), dtype=float),
            method=self.method_name,
        )

# Register it
loader = UnifiedDataLoader.from_coax(...)
from src.xai_adapter import get_adapter_registry
registry = get_adapter_registry()
registry.register('custom', CustomMethod)

# Use it
custom_method = registry.create('custom', model=my_model).fit()
result = custom_method.explain(loader.get_features([0, 1]))
```

### Custom Normalizer

```python
from src.data_loaders.base import BaseNormalizer

class LogNormalizer(BaseNormalizer):
    def normalize(self, value, min_val, max_val):
        return np.log1p(value)
    
    def normalize_array(self, values, min_vals, max_vals):
        return [np.log1p(v) for v in values]

# Use it
loader = UnifiedDataLoader.from_coax(
    ...,
    normalizer=LogNormalizer()
)
```

---

## API Reference

### UnifiedDataLoader Methods

| Method | Purpose |
|--------|---------|
| `from_coax()` | Load CoAX synthetic data |
| `from_coxam()` | Load CoXAM experiment data |
| `from_custom()` | Load from custom source |
| `get_instances()` | Get features + predictions |
| `get_features()` | Get feature vectors |
| `get_predictions()` | Get AI predictions |
| `get_explanations()` | Get explanation data |
| `filter()` | Create filter builder |
| `apply_filter()` | Apply filters to data |
| `get_participant_trials()` | Get CoXAM participant trials |
| `get_participant_ids()` | Get CoXAM participant list |
| `get_summary()` | Get data summary |
| `list_apps()` | List available app IDs |

### FilterBuilder Methods

| Method | Purpose |
|--------|---------|
| `by_app()` | Filter by app/dataset ID |
| `by_participant()` | Filter by participant ID |
| `by_condition()` | Filter by experimental condition |
| `by_model()` | Filter by model name |
| `by_xai_type()` | Filter by XAI explanation type |
| `by_phase()` | Filter by trial phase |
| `by_custom()` | Apply custom filter function |
| `apply()` | Apply filters to DataFrame |

### XAIAdapterRegistry Methods

| Method | Purpose |
|--------|---------|
| `register()` | Register XAI method type |
| `get_class()` | Get explanation method class |
| `create()` | Create explanation method instance |
| `list_available()` | List registered types |
| `is_registered()` | Check if type exists |

---

## Benefits Over Original Code

### Before (Duplicated)
```
src/coax/data_loader.py          # AIDatasetLoader (CoAX only)
src/coxam/utils.py               # AIDatasetLoader (CoXAM variant)
src/coxam/utils.py               # DecisionTreeInterpreter
src/coxam/utils.py               # LogisticRegressionInterpreter
```

**Issues:**
- ❌ Code duplication in AIDatasetLoader
- ❌ Interpreters only available in CoXAM
- ❌ No filter composition system
- ❌ Hard to extend with new explanation methods or normalizers

### After (Unified)
```
src/data_loaders/                # Single unified data layer
├── unified_loader.py            # UnifiedDataLoader API
├── sources/                      # Pluggable data sources
├── normalizers/                  # Pluggable normalizers
└── filters/                      # Composable filters
```

**Benefits:**
- ✅ Single source of truth
- ✅ 90%+ less code duplication
- ✅ XAI methods are isolated in `src/xai_adapter`
- ✅ Consistent API across all frameworks
- ✅ Composable, chainable filters
- ✅ In-depth documentation and examples

---

## File Listing

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 60 | Module exports |
| `unified_loader.py` | 350+ | Core API |
| `base/__init__.py` | 10 | Base exports |
| `base/data_source.py` | 60 | Abstract DataSource |
| `base/normalizer.py` | 40 | Abstract Normalizer |
| `sources/coax_adapter.py` | 250+ | CoAX data source |
| `sources/coxam_adapter.py` | 250+ | CoXAM data source |
| `normalizers/minmax.py` | 50 | Min-Max normalizer |
| `normalizers/zscore.py` | 50 | Z-Score normalizer |
| `filters/filter_builder.py` | 180+ | Filter system |
| `examples.py` | 300+ | Usage examples |
| **TOTAL** | **~2,000+** | Complete unified system |

---

## Migration Guide

### From CoAX
```python
# Old way
from src.coax.data_loader import AIDatasetLoader
loader = AIDatasetLoader(feature_df, metadata_df, explanation_df)
scaled_features = loader.scale_feature_values(ids)

# New way
from src.data_loaders import UnifiedDataLoader
loader = UnifiedDataLoader.from_assets(source="coax", assets_root="assets")
features = loader.get_features(ids, normalize=True)
```

### From CoXAM
```python
# Old way
from src.coxam.utils import AIDatasetLoader, DecisionTreeInterpreter
ai_loader = AIDatasetLoader(...)
dt_exp = DecisionTreeInterpreter(dt_df, metadata_df, app_id, model_name)

# New way
from src.data_loaders import UnifiedDataLoader
loader = UnifiedDataLoader.from_coxam(...)
from src.xai_adapter import get_adapter_registry
registry = get_adapter_registry()
dt_exp = registry.create('decision_tree', explanation_df=dt_df, metadata_df=metadata_df, ...)
```

---

## Testing

Run examples:
```bash
python src/data_loaders/examples.py
```

Verify syntax of all modules:
```bash
python3 -c "
import ast
import glob
for f in glob.glob('src/data_loaders/**/*.py', recursive=True):
    ast.parse(open(f).read())
    print(f'✓ {f}')
"
```

---

## Future Extensions

The plugin architecture enables easy addition of:

1. **New Data Sources**: Database, HuggingFace, API-based
2. **New XAI Methods**: SHAP, LIME, Integrated Gradients, Attention
3. **New Normalizers**: Robust scaling, Quantile normalization, Custom
4. **New Filters**: Composition operators, Complex queries
5. **Caching Layer**: For expensive computations
6. **Streaming Support**: For large datasets

---

## License

Part of xaik-tool-cognitive-agent project.

---

## Support

For issues, questions, or extensions:
1. Check `examples.py` for usage patterns
2. Review base classes in `base/` for extension points
3. See existing implementations in `sources/`, `normalizers/`, and `src/xai_adapter/attribution/`
