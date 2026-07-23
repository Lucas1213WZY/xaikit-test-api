# XAI Adapter

`src.xai_adapter` wraps external XAI libraries behind a consistent adapter API.
It owns the sklearn-like XAI method interface for feature attribution, CoXAM
rules-vs-weights surrogates, concept/example-based explanations, glass-box
models, and thin wrappers around precomputed CSV explanations.

Every adapter's `explain(...)` returns an `XAIAdapterResult` with `values`
(signed per-feature attributions), `base_values` (per-instance intercept),
`attributions`, and `importances`.

## Registered adapters

Names below are what you pass to `create_xai_method(name, ...)`; aliases in
parentheses resolve to the same adapter ([registry.py](registry.py)).

**Attribution (local + global)**

- `lofo` (`leave_one_feature_out`): local leave-one-feature-out, numpy only
- `shap_kernel` (`shap`): SHAP `KernelExplainer`, model-agnostic black box
- `shap_tree` (`shap_treeexplainer`): SHAP `TreeExplainer` for tree ensembles
- `shap_linear` (`shap_linearexplainer`): SHAP `LinearExplainer` for linear/logistic models
- `shap_deep` (`shap_deepexplainer`): SHAP `DeepExplainer` for PyTorch/TensorFlow models
- `shap_gradient` (`shap_gradientexplainer`): SHAP `GradientExplainer` for PyTorch/TensorFlow models
- `lime` (`lime_tabular`): LIME tabular explainer
- `gradient_input` (`gradient_x_input`, `input_gradients`): gradient × input, torch models
- `deeplift` (`deep_lift`): Captum DeepLift
- `integrated_gradients` (`ig`): Captum Integrated Gradients
- `lrp` (`layer_relevance_propagation`): Layer-wise Relevance Propagation via Captum, torch models
- `sklearn_global` (`global_feature_importance`): sklearn-style `feature_importances_` or `coef_`
- `sim2real_property` (`property_optimized`, `xaisim2real`): property-optimized attribution matrices

**Surrogate (CoXAM rules/weights + rule extraction)**

- `decision_tree` (`dt`, `rules`): CoXAM decision-tree surrogate
- `logistic_regression` (`lr`, `weights`): CoXAM logistic-regression surrogate
- `rule_list` (`rulelist`): ordered rule-list surrogate (first-match semantics)
- `rule_set` (`ruleset`): unordered rule-set surrogate (confidence-sorted)
- `anchors` (`anchor_tabular`): Anchors rule-based local explanations via alibi

**Concept**

- `tcav`: TCAV concept attribution via Captum

**Glass-box / interpretable**

- `ebm` (`interpret_ebm`, `explainable_boosting`): InterpretML Explainable Boosting Machine

**Example-based**

- `counterfactual` (`cf`, `wachter`): Wachter et al. (2017) counterfactuals via alibi
- `dice` (`diverse_counterfactuals`): Diverse Counterfactual Explanations via dice-ml
- `prototypes` (`mmd_critic`, `criticisms`): MMD-Critic prototypes and criticisms

**Dataset-backed**

- `precomputed_csv` (`csv`, `csv_dataset`, `dataset_csv`): expose precomputed explanation vectors from a `data_loaders.XAIDatasetParser`

Construction helpers (functions, not registry names):

- `create_custom_xai_method` / `register_xai_method`: wrap a custom attribution function or object
- `make_surrogate` / `create_custom_surrogate_method`: wrap custom surrogate fit/explain callables
- `generate_surrogate_xai_methods`: train fresh rules/weights surrogates when a new CSV has instances and AI predictions but no precomputed CoXAM tables

`create_xai_method` is the single entry point for building any adapter. It takes
three optional wiring arguments on top of the adapter's own kwargs:

- `ai_model=`: a trained AI model (exposes `predict`, `model`,
  optionally `forward_logits_or_probs`). When given with `train_data=`, the model
  and training data are wired into each attribution adapter's expected kwargs
  automatically (LOFO, kernel SHAP, LIME, gradient×input, DeepLift, IG, LRP).
- `train_data=`: training data used with `ai_model=` (exposes `X`/`y`/
  `feature_names`/`categorical_feature_indices`).
- `loader=`: a CoXAM data loader. For a `rules`/`weights` (or `decision_tree`/
  `logistic_regression`) name, the surrogate's explanation and metadata tables
  are read from the loader and filtered by `model_name`/`depth`/`variant`.

All three default to `None`; omit them to pass adapter kwargs directly.

## Package layout

The public API is organized by adapter family:

```text
xai_adapter/
  base.py               # XAIAdapter base class, XAIAdapterResult, baseline helpers
  registry.py           # name -> adapter registry, create_xai_method / register_xai_method
  api.py                # engine/design/run helpers, custom + surrogate constructors
  attribution/          # local attribution and global importance methods
  surrogate/            # rules/weights, rule list/set, anchors, custom surrogates
  concept/              # TCAV concept attribution
  interpret/            # glass-box models (InterpretML EBM)
  example_based/        # counterfactuals (Wachter, DiCE) and prototypes (MMD-Critic)
  dataset.py            # precomputed CSV explanations
  metrics.py            # faithfulness / fidelity metrics
  visualization.py      # explanation plotting helpers
```

Library-backed attribution classes can be imported directly from `attribution`:

```python
from src.xai_adapter.attribution import (
    IntegratedGradients,
    DeepLift,
    GradientInput,
    KernelShap,
    Lime,
    LeaveOneFeatureOut,
)
```

`KernelShap` uses SHAP's `KernelExplainer`, and `Lime` uses LIME's tabular
explainer. In most cases, prefer the registry entry point `create_xai_method(name, ...)`
over importing classes directly.

Example:

```python
from src.xai_adapter import create_xai_method

method = create_xai_method(
    "shap",
    predict_fn=model.predict_proba,
)
method.fit(X_train)

result = method.explain(X_test[:5])
values = result.values
base_values = result.base_values
signed_attributions = result.attributions
absolute_importances = result.importances
```

Use `preprocessing_fn` and `postprocessing_fn` when the model consumes
preprocessed/OHE features but you want to explain raw feature rows.

For a trained AI model, pass `ai_model=` and `train_data=` to the
same `create_xai_method` — it wires the model and training data into the adapter:

```python
from src.xai_adapter import create_xai_method

method = create_xai_method(
    "lofo",
    ai_model=ai_model,
    train_data=train_data,
    preprocessing_fn=preprocessing_fn,
)

result = method.explain(instances)
```

For CoXAM rules/weights surrogates read from a data loader, pass `loader=`; the
method name selects the surrogate and the tables come from the loader:

```python
from src.xai_adapter import create_xai_method

rules = create_xai_method(
    "rules", loader=loader, app_id="wine_quality", model_name="mlp", depth=3,
)
weights = create_xai_method(
    "weights", loader=loader, app_id="wine_quality", model_name="mlp", variant="sparse",
)

predictions = rules.apply_batch(loader.get_features([0, 1, 2], normalize=False))
```

Custom XAI algorithms can be wrapped as adapters too:

```python
import numpy as np
from src.xai_adapter import create_custom_xai_method, register_xai_method

def my_attribution(x):
    return np.asarray(x) * 0.5

method = create_custom_xai_method(my_attribution, method_name="my_method")
result = method.explain(X_test)

register_xai_method("my_method", my_attribution, "mine")
result = create_xai_method("mine").explain(X_test)
```

Custom implementations may be plain functions, or objects/classes exposing
`fit` and `explain`. Legacy objects exposing `attribute` are still accepted,
but `explain` is the canonical call across attribution, surrogate, and
CSV-backed methods. Raw arrays, `(values, base_values)` tuples, and
`XAIAdapterResult` objects are normalized to the same result type used by
built-in adapters.

Custom surrogate methods use the same adapter shape, but provide separate
`fit_fn` and `explain_fn` callables:

```python
import numpy as np
from src.xai_adapter import create_custom_surrogate_method, make_surrogate

def fit_surrogate(X, y, **kwargs):
    state["mean_prediction"] = float(np.mean(y))

def explain_surrogate(instances):
    rows = np.asarray(instances, dtype=float)
    return rows * state["mean_prediction"]

state = {}

surrogate = make_surrogate(fit_surrogate, explain_surrogate, name="my_surrogate")
result = surrogate.fit(X_train, y_train).explain(X_test)

# Equivalent higher-level constructor.
surrogate = create_custom_surrogate_method(
    fit_surrogate,
    explain_surrogate,
    method_name="my_surrogate",
)
```

Rules-vs-weights methods use the same pattern:

```python
from src.xai_adapter import create_xai_method

rules = create_xai_method(
    "rules",
    app_id="wine_quality",
    model_name="mlp",
    depth=3,
)
rules.fit(explanation_df=decision_tree_df, metadata_df=metadata_df)

weights = create_xai_method(
    "weights",
    app_id="wine_quality",
    model_name="mlp",
    variant="sparse",
)
weights.fit(explanation_df=logistic_regression_df, metadata_df=metadata_df)
```

If a user provides a new dataset CSV instead of precomputed surrogate tables,
the same methods can generate fresh surrogates from feature rows and AI
predictions:

```python
from src.xai_adapter import create_xai_method

X = dataset_predictions_df[["v0", "v1", "v2"]].to_numpy()
y = dataset_predictions_df["pred"].to_numpy()

rules = create_xai_method("rules", app_id="my_dataset", model_name="external_model", depth=3)
rules.fit(X, y)

weights = create_xai_method(
    "weights",
    app_id="my_dataset",
    model_name="external_model",
    variant="sparse",
    top_k=3,
)
weights.fit(X, y)

rules_result = rules.explain(X)
weights_result = weights.explain(X)

# Optional: persist generated CoXAM-style tables.
decision_tree_df = rules.to_explanation_table()
logistic_regression_df = weights.to_explanation_table()
metadata_df = rules.to_metadata_table()
```

The dataset helper wraps that same flow when you want to start from a CSV-like
object:

```python
from src.data_loaders import XAIDatasetParser
from src.xai_adapter import generate_surrogate_xai_methods

dataset = XAIDatasetParser.from_dataframe(
    dataset_predictions_df,
    missing_explanation_strategy="zeros",
)

generated = generate_surrogate_xai_methods(
    dataset=dataset,
    app_id="my_dataset",
    model_name="external_model",
    methods=("decision_tree", "logistic_regression"),
    depths=(3,),
    variants=("dense", "sparse"),
    variant="sparse",
    top_k=3,
)

rules = generated.methods["rules"]
weights = generated.methods["weights"]

feature_rows = dataset.get_features(dataset.df["instanceId"].tolist())
rules_result = rules.explain(feature_rows)
weights_result = weights.explain(feature_rows)
```

CSV-backed predictions and explanations:

```text
instanceId,pred,v0,v1,v2,a0_i,a1_i,a2_i
0,1,0.2,0.7,0.1,0.4,0.5,0.1
```

```python
from src.data_loaders import XAIDatasetParser
from src.xai_adapter import create_xai_method

dataset = XAIDatasetParser.from_csv("assets/data/my_dataset_predictions.csv")
method = create_xai_method("csv", dataset=dataset)

records = dataset.get_records([0, 1, 2])
explanations = [record.explanation for record in records]
ai_predictions = [record.ai_prediction for record in records]
```

If the CSV uses the common `v0`, `v1`, ... and `a0_i`, `a1_i`, ... names, the
feature and explanation columns are inferred automatically. `intercept` is
optional; when it is absent, adapter result `base_values` default to `0.0`.

For a prediction-only CSV, set `missing_explanation_strategy="zeros"` on
`XAIDatasetParser` to produce control/no-XAI explanation vectors with the same
length as the feature vector.
