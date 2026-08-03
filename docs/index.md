# XAIKit

**XAI Interpretation Simulator Toolkit** — design, simulate, and analyse
human-subject studies of explainable-AI methods.

XAIKit covers the whole loop of an XAI user study: prepare a dataset, train the
AI model under study, generate explanations for it with the XAI method you want
to test, lay out a counterbalanced experimental design, run cognitive agents
through those trials as simulated participants, and analyse the results.

## Install

```bash
git clone https://github.com/Lucas1213WZY/xaikit-test-api.git
cd xaikit-test-api
pip install -r requirements.txt
```

Python 3.10 is recommended. Python 3.9 works for the CoAX/XAI stack, though pip
may select older wheels for some scientific packages.

## Two ways to use it

XAIKit exposes the same functionality through a guided orchestrator and through
the stage modules it delegates to. Use whichever fits.

**1. The `xaikitTest` orchestrator** — one object carries the study from design
through analysis, and each stage validates what the previous one produced:

```python
from src import xaikitTest

exp = xaikitTest("my_study")
exp.add_iv("xai_method", "within", ["shap", "lime", "none"])
exp.add_dv("forward_accuracy", [0, 1])
exp.prepare_dataset("wine_quality")
exp.train_AI_model(model_type="mlp", target_accuracy=0.8)
```

**2. The stage modules directly** — each is independently usable, with no
orchestrator involved:

```python
from src import prepare_dataset, create_xai_method, build_experiment_plan

data = prepare_dataset("wine_quality")
method = create_xai_method("shap", ai_model=engine, train_data=data.X_train)
```

## Where to go next

- **[User Guide](user_guide.html)** — the workflow stage by stage, with the
  decisions each stage asks you to make.
- **[API Reference](api/index.html)** — every public module, class, function,
  and argument, generated from the source.

## Project layout

| Module | Responsibility |
| --- | --- |
| [`src.api`](api/src/api.html) | The `xaikitTest` orchestrator facade |
| [`src.data_loaders`](api/src/data_loaders.html) | Dataset loading, feature selection, normalisation |
| [`src.ai_models`](api/src/ai_models.html) | Training and evaluating the AI model under study |
| [`src.xai_adapter`](api/src/xai_adapter.html) | XAI methods (attribution, surrogate, rule-based) |
| [`src.experiment_planner`](api/src/experiment_planner.html) | Design validation, trials, counterbalancing, protocol |
| [`src.cognitive_models`](api/src/cognitive_models.html) | Cognitive agents and machine-proxy baselines |
| [`src.virtual_experiment_executor`](api/src/virtual_experiment_executor.html) | Running agents through the trials |
| [`src.statistical_analyst`](api/src/statistical_analyst.html) | ANOVA and pairwise condition tests |
| [`src.result_visualizer`](api/src/result_visualizer.html) | Interaction plots and result grids |
