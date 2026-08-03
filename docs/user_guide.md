# User Guide

This guide walks the XAIKit workflow one stage at a time. Each section states
what the stage is for, the decisions it asks you to make, and the calls that
carry them out. Every API name links to its full signature in the
[API Reference](api/index.html).

The stage text here mirrors the built-in guides, so the same material is
available inside a notebook:

```python
exp.guide("design")     # or: exp.guide_design()
exp.guide("dataset")
exp.guide("trial_generation")
exp.guide("model_training")
exp.guide("explanation_generation")
exp.guide("cognitive_models")
exp.guide("cognitive_simulation")
```

## Contents

1. [Experimental design](#1-experimental-design)
2. [Dataset preparation](#2-dataset-preparation)
3. [AI model training](#3-ai-model-training)
4. [Trial generation](#4-trial-generation)
5. [Explanation generation](#5-explanation-generation)
6. [Choosing a cognitive model](#6-choosing-a-cognitive-model)
7. [Cognitive simulation](#7-cognitive-simulation)
8. [Analysis and visualisation](#8-analysis-and-visualisation)

---

## 1. Experimental design

**Goal:** decide which XAI methods you want to test, and how the study compares
them.

A design is expressed in three kinds of variable, plus the task participants
perform:

| Term | Meaning | Example |
| --- | --- | --- |
| **IV** | What you manipulate | `xai_method = ["shap", "lime", "none"]` |
| **CV** | Trial or participant metadata you control or record | age group, gender, `user_task` |
| **DV** | What you measure | `forward_accuracy` |
| **User task** | What participants and cognitive agents actually do | `forward_simulation` — predict the AI output from the instance and its explanation |

Each IV is also declared `"within"` or `"between"`, which is what drives
counterbalancing later:

| `iv_type` | Meaning | `randomization` |
| --- | --- | --- |
| `"within"` | Every participant sees every level | `"block"` (default) or `"trial"` |
| `"between"` | Each participant sees one level | not allowed — omit it |

### What's available

Everything below is generated from `src/experiment_planner/support_matrix.json`
at build time — the same file
[`validate`](api/src/api.html#xaikitTest.validate) checks your design against,
so this list cannot drift from what the planner will actually accept.

Four of these IVs are *semantic*: `dataset`, `user_task`, `ai_model`, and
`xai_faithfulness` describe the study rather than being manipulated within it,
so they are usually added with `add_cv` rather than `add_iv`.

<!--SUPPORT_MATRIX-->

```python
from src import xaikitTest

exp = xaikitTest("my_study", output_dir="output")
exp.add_iv("xai_method", "within", ["shap", "lime", "none"])
exp.add_cv("user_task", ["forward_simulation"])
exp.add_dv("forward_accuracy", [0, 1])
exp.validate(stage="design")
```

The usual order is: add IVs, add CVs, add `user_task`, add DVs, then
`validate(stage="design")`.

**Coming from the experiment-design UI?** Skip the manual calls — pass the
export straight to the constructor and its IVs, CVs, DVs and study protocol are
registered for you:

```python
exp = xaikitTest("my_study", design="tutorials/experiment-design.json")
```

See [`add_iv`](api/src/api.html#xaikitTest.add_iv),
[`add_cv`](api/src/api.html#xaikitTest.add_cv),
[`add_dv`](api/src/api.html#xaikitTest.add_dv),
[`set_design_export`](api/src/api.html#xaikitTest.set_design_export),
[`validate`](api/src/api.html#xaikitTest.validate).

<div class="todo"><strong>To expand:</strong> your lab's conventions for naming
IVs and DVs, and which designs you have found adequately powered.</div>

## 2. Dataset preparation

**Goal:** choose the dataset and the feature subset used for model training,
trials, and what participants see on screen.

XAIKit keeps two views of every dataset: **raw values** for display to
participants, and **model-ready values** for training. You never have to
reconcile them by hand.

```python
data = exp.prepare_dataset(
    "wine_quality",
    model_type="mlp",
    num_features=8,
    test_size=0.2,
    random_state=42,
)
```

Key arguments: `dataset_id`, either `feature_cols` or `num_features` (or the
built-in defaults), `test_size`, and `random_state`. Leave
`show_available=True` to print the datasets XAIKit can see.

See [`prepare_dataset`](api/src/api.html#xaikitTest.prepare_dataset) and the
[`src.data_loaders`](api/src/data_loaders.html) module.

## 3. AI model training

**Goal:** train the AI model that will later supply the predictions and the
explanations participants reason about.

Supported model types are `mlp`, `xgboost`, and `sim2real`.

```python
exp.train_AI_model(
    model_type="mlp",
    target_metric="accuracy",
    target_accuracy=0.8,
    max_epochs=300,
    batch_size=1000,
)
exp.metrics_table()
```

Training runs until the target metric is met or `max_epochs` is exhausted,
checking every `check_every_epochs`. To reuse a pre-trained model instead, call
[`load_AI_model`](api/src/api.html#xaikitTest.load_AI_model).

Inspect the result with
[`metrics_table`](api/src/api.html#xaikitTest.metrics_table),
[`evaluate`](api/src/api.html#xaikitTest.evaluate),
[`confusion_matrix_table`](api/src/api.html#xaikitTest.confusion_matrix_table),
[`plot_training_history`](api/src/api.html#xaikitTest.plot_training_history),
and [`plot_auc_curves`](api/src/api.html#xaikitTest.plot_auc_curves).

## 4. Trial generation

**Goal:** sample the training rows first and the held-out testing rows second,
then lay them out into counterbalanced participant sequences.

Train the AI model *before* this stage if you want prediction-balanced
sampling. With `balance_by_ai_prediction=True`, each phase draws equally from
the two predicted classes, so participants do not see a run of one answer.

```python
result = exp.generate_trials(
    participants_per_between_condition=24,
    num_training=0,
    num_testing=12,
    balance_by_ai_prediction=True,
    counterbalancing_strategy="auto",
    trial_randomization_strategy="balanced",
    seed=42,
)
```

Two ordering rules are worth knowing: the predicted-class order is randomised
*within* each phase, and the phase order is **never** counterbalanced —
training always precedes testing.

Preview what a given participant will actually see before committing:

```python
exp.preview_participant_trials(participant_id=1, visualization="importance")
```

See [`generate_trials`](api/src/api.html#xaikitTest.generate_trials),
[`preview_participant_trials`](api/src/api.html#xaikitTest.preview_participant_trials),
and [`src.experiment_planner`](api/src/experiment_planner.html).

## 5. Explanation generation

**Goal:** produce the XAI tables for the methods named in your design.

Methods default to the levels of your `xai_method` IV, and the model name
defaults to whatever you just trained, so the common case needs no arguments:

```python
path, table = exp.explanations(target=1, output_dir="generated_explanation")
```

Pass `method_kwargs` to tune individual methods — SHAP background size, LIME
sample count, and so on:

```python
exp.explanations(
    methods=["shap", "lime"],
    method_kwargs={
        "shap": {"background_size": 100},
        "lime": {"num_samples": 5000},
    },
)
```

See [`explanations`](api/src/api.html#xaikitTest.explanations),
[`plot_explanation`](api/src/api.html#xaikitTest.plot_explanation), and
[`src.xai_adapter`](api/src/xai_adapter.html) for the full method registry and
for registering your own via `register_xai_method`.

## 6. Choosing a cognitive model

**Goal:** pick the agent that will stand in for a human participant, and the
parameter ranges to sweep.

```python
exp.guide_cognitive_models()   # returns a table of available agents
```

Two families are available, and they are configured differently:

- **Machine-proxy baselines** — `knn`, `decision_tree`, `logistic_regression`,
  `mlp`. Configure these with `model_kwargs`.
- **Cognitive agents** — configure these with `cognitive_params`.

```python
exp.set_cognitive_model(cognitive_model_id="knn", model_kwargs={"n_neighbors": 5})
```

See [`set_cognitive_model`](api/src/api.html#xaikitTest.set_cognitive_model)
and [`src.cognitive_models`](api/src/cognitive_models.html).

<div class="todo"><strong>To expand:</strong> guidance on which agent
corresponds to which theoretical account, and the parameter ranges you have
found plausible for each.</div>

## 7. Cognitive simulation

**Goal:** run the chosen cognitive model over the generated trials to produce
simulated behaviour.

This stage is optional. If you only need a study *design* to hand to human
participants, stop after stage 5 and export the protocol.

Simulation requires trials, AI predictions, and a supported `user_task`/DV
combination.

```python
responses = exp.run_experiment(mode="participant_by_participant")
csv_path, json_path = exp.save_results(out_dir="simulated_results")
```

See [`run_experiment`](api/src/api.html#xaikitTest.run_experiment),
[`save_results`](api/src/api.html#xaikitTest.save_results), and
[`src.virtual_experiment_executor`](api/src/virtual_experiment_executor.html).

## 8. Analysis and visualisation

**Goal:** test whether the XAI method you manipulated actually moved the
dependent variable.

```python
exp.analyze_iv_dv(iv="xai_method", dv="forward_accuracy")

exp.plot_dv_by_two_ivs(
    x_iv="xai_method",
    hue_iv="user_task",
    dv="forward_accuracy",
    phase="testing",
    errorbar="sem",
)
exp.plot_results_grid()
```

See [`analyze_iv_dv`](api/src/api.html#xaikitTest.analyze_iv_dv),
[`plot_dv_by_two_ivs`](api/src/api.html#xaikitTest.plot_dv_by_two_ivs),
[`plot_results_grid`](api/src/api.html#xaikitTest.plot_results_grid),
[`src.statistical_analyst`](api/src/statistical_analyst.html), and
[`src.result_visualizer`](api/src/result_visualizer.html).

---

## The study protocol

Alongside the variables, a study carries a participant-facing protocol: title,
research questions, consent text, and procedure steps. It is validated and
exportable, so the same object that defines the simulation also defines what a
human participant would be shown.

```python
exp.set_study_protocol(
    study_title="Does SHAP improve forward simulation?",
    research_questions=["Do feature attributions improve prediction accuracy?"],
    consent_text="...",
    procedure_steps=[{"name": "training", "description": "..."}],
)
exp.preview_experiment_walkthrough(participant_id=1)
exp.approve_walkthrough(confirmed=True)
exp.save_study_protocol("study_protocol.json")
```

See [`set_study_protocol`](api/src/api.html#xaikitTest.set_study_protocol),
[`preview_experiment_walkthrough`](api/src/api.html#xaikitTest.preview_experiment_walkthrough),
[`approve_walkthrough`](api/src/api.html#xaikitTest.approve_walkthrough).

## Tutorials

Runnable end-to-end notebooks live in `tutorials/`:

- `feature_explanation_user_study.ipynb` — the full workflow
- `feature_explanation_user_study_from_design_export_coax.ipynb` — driving a study from a design-UI export
- `attribution_experiment_replication_guide.ipynb` — replicating a published attribution experiment
