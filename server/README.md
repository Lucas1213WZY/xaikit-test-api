# XAIKit study server

A FastAPI service that takes an experiment-design UI export and runs the same
pipeline as the tutorial notebooks: train the AI model, generate trials,
generate explanations, run virtual participants, analyze, plot.

Nothing here re-implements experiment logic. Every endpoint is a call into
`src/` — `xaikitTest` for the study stages, `run_coax_study` for CoAX
participants, `analyze_iv_dv` for statistics, `plot_dv_by_two_ivs` /
`plot_iv_dv_grid` for plot data. Change a rule in `src/`, and the server and the
notebooks change together.

## Running it

```bash
conda activate xaik-api-dev
pip install "fastapi>=0.115,<1" "uvicorn>=0.30,<1"    # already in requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000     # from the repo root
```

Interactive docs: <http://localhost:8000/docs>.

| Environment variable | Purpose |
| --- | --- |
| `XAIKIT_SERVER_RUNS_DIR` | Where per-study artifacts are written (default `./server_runs`) |
| `XAIKIT_ALLOWED_ORIGINS` | Comma-separated CORS origins, e.g. your UI's URL (default `*`) |
| `XAIKIT_API_TOKEN` | Shared bearer token; unset means no authentication (see below) |

Run **one** worker. A study is a stateful `xaikitTest` object held in the
process, and stages run on a single background thread because torch training
and matplotlib rendering are not safe to run concurrently in one process.

## Flow

```
POST /api/studies                     the design export JSON -> {study_id, design, validation}
POST /api/studies/{id}/dataset        prepare_dataset + train_AI_model   -> job
POST /api/studies/{id}/trials         generate_trials                    -> job
POST /api/studies/{id}/explanations   explanations                       -> job
POST /api/studies/{id}/simulate       virtual participants               -> job
GET  /api/jobs/{job_id}               state, elapsed, captured log, result
GET  /api/studies/{id}/results        step rows as JSON (phase/participant/limit/offset)
GET  /api/studies/{id}/results.csv    the same rows as a CSV attachment
POST /api/studies/{id}/analysis       analyze_iv_dv per IV x DV -- condition means + omnibus p-value
POST /api/studies/{id}/posthoc        pairwise_condition_tests -- pairwise condition means + corrected p-values
POST /api/studies/{id}/plots/interaction   one DV by two IVs
POST /api/studies/{id}/plots/grid          every DV by every IV
```

The four stage endpoints return `202` with a job immediately — training and
LIME/SHAP generation outlast any sensible request timeout — so poll
`GET /api/jobs/{job_id}` until `state` is `succeeded` or `failed`. The job's
`log` field carries the API layer's own progress output.

Anything the design export already answers can be omitted from a request body:
dataset id, participants per condition, XAI methods and DVs are read from the
export. `userModel` in the export selects the participant runner:

| `userModel` | runner |
| --- | --- |
| `"CoAX"` **without** an `xai_property` IV | `run_coax_study` |
| `"CoAX"` **with** an `xai_property` IV, or `"CoAX (XAI Property)"` | `run_sim2real_study` |
| `"CoXAM"` | `run_coxam_study` |
| `"Sim2Real"` | `run_sim2real_study` |
| anything else, e.g. `"KNN"` | `study.run_experiment` with that baseline model |

The first row is not a typo: a Sim2Real design's cognitive model is a
CoAX-derived attribution sum, so the UI genuinely exports `userModel: "CoAX"`
for it — the `xai_property` IV (faithful/sparse/robust/sparse_robust) is what
tells the two apart. `"CoAX (XAI Property)"` is the UI's own disambiguated
label for the same thing and resolves the same way, whether or not the IV
itself parsed. Routing lives in `DesignExport.resolved_framework`
(`src/experiment_planner/design_export.py`) — call that directly rather than
inspecting `userModel` yourself if you need to know which runner a design will
use.

## Case-by-case simulation

`POST /simulate` takes the API layer's own selection vocabulary in `mode`, so
one endpoint serves a step-through and the full run:

```jsonc
{"mode": "trial_by_trial"}                                   // one trial
{"mode": "participant_by_participant", "participant_id": 3}  // one participant
{"mode": "whole_condition", "condition_filter": {"xai_type": "attribution"}}
{"mode": "whole_experiment"}                                 // everything
```

Each run is saved as CSV and JSON under
`<runs_dir>/<study_id>/simulated_results/<mode>/`, and the most recent run is
what `/results`, `/results.csv`, `/analysis` and the plot endpoints read.

A CoAX training trial that shows an explanation produces **two** rows —
`infer_no_explanation` then `infer_with_explanation` — so step through on
`(phase, step)`, not on trial index.

## Plots

The plot endpoints return the participant-level aggregate the plot helpers drew
(`mean`, `std`, `sem`, `count` per cell), so the UI can render it with its own
chart library. Pass `"include_png": true` to also get a base64 PNG.

## Matching the apparatus: instance ids and existing corpora

A design export's `apparatus[]` array declares which instances the
participant-facing UI actually showed people, one entry per configuration
(e.g. one per condition):

```jsonc
"apparatus": [
  {"params": {"form": "LR", "instanceIds": "1-20"}},
  {"params": {"form": "DT", "instanceIds": "1-20", "trainingInstanceIds": "0-9"}}
]
```

`DesignExport.apparatus_instance_ids` / `apparatus_training_instance_ids` parse
every entry's `instanceIds` / `trainingInstanceIds` (`"10-19"` or `"1,3,5-9"`,
both ends inclusive) and union them across entries — an entry with no `params`
contributes nothing rather than erroring. `POST /trials` defaults
`allowed_instance_ids` to the union of both, so simulated trials reference
exactly the instances a human participant was shown, without the caller
repeating that JSON. Pass an explicit list (or `[]`) to override.

**`/dataset` skips AI training whenever a published corpus already covers the
design**, so the whole run — dataset through simulate — uses zero training
compute and reads only what already exists in `assets/`:

* **Sim2Real, always.** Its simulation reads a fixed corpus, never
  `trained_ai_model`.
* **CoXAM, when the dataset is one its corpus covers** (`wine_quality`,
  `mushrooms`, …, `COXAM_CORPUS_FEATURES`). `prepare_dataset` is also routed
  onto that corpus's own feature set automatically — see the next section —
  so `run_coxam_study(source="assets")` can read its predictions and DT/LR
  surrogates with nothing trained. A CoXAM dataset the corpus does *not* cover
  still trains, since only `source="fit"` can serve it.

When training is skipped, `/dataset`'s response has `"model": null` and
`"model_skipped_reason"` explaining why — check that field rather than
assuming; a dataset just outside the corpus trains normally. `POST /simulate`'s
`coxam_source` follows the same signal: unset, it resolves to `"assets"` when
no model was trained and `"fit"` when one was, so a plain `{"mode":
"whole_experiment"}` body works either way without the caller tracking what
`/dataset` did.

**CoXAM never needs `POST /explanations`.** `run_coxam_study` builds its own
DT/LR surrogates internally (from a trained model, or from the corpus) and
never reads `study.combined_explanations` — the table that endpoint produces.
Calling it for a CoXAM design when training was skipped will fail outright (no
model for LIME/SHAP to explain); for CoAX or a baseline it is still required,
since those runners do read that table.

**CoXAM's feature-set trap.** `prepare_dataset`'s ordinary default and the
published corpus disagree — for `wine_quality`, the loader's usual 5-feature
default omits `Chlorides`, which the corpus needs as its 6th, positional
feature (`a0..a5`). Passing `cognitive_model_id="coxam"` to `prepare_dataset`
(which `/dataset` does automatically from the design's resolved framework)
routes onto the corpus's exact feature set and disables target-ranking, which
would otherwise reorder them. CoAX has its own, separately-fixed 5-feature
corpus for the same datasets — this defaulting is CoXAM-specific and does not
touch CoAX.

## Cognitive parameters

**Every design needs `/dataset` called before `/trials`**, including CoXAM and
Sim2Real: `generate_trials()` requires a prepared dataset regardless of which
participant runner ends up simulating.

A design export uses one of two shapes for parameter overrides, and both are
read: the older `cognitiveConfig: {"Display Label": "string value"}` map, and
a newer `cognitiveParameters: [{key, label, min, max, value, source, ...}]`
list where `key` is already the agent's own parameter name and `value: null`
means "use the model default." Both can be present; a shared key in
`cognitiveParameters` wins, since it is the more specific, already-resolved
source.

### CoAX

Left unset, strategies are built from `FITTED_COAX_PARAMS` in
`src/virtual_experiment_executor/experiment_simualtion/CoAX/coax_study_runner.py`
— the per-strategy means of the parameters fitted to the original CoAX study
participants. To reproduce a notebook's hand-set values instead, pass them
through:

```jsonc
{
  "mode": "whole_experiment",
  "coax_params": {
    "none":        {"sensitivity": 1.0,  "k": 2, "retrieval_threshold": -2.5},
    "importance":  {"sensitivity": 10.0, "k": 2, "retrieval_threshold": -2.5},
    "attribution": {"scaling_factor": 1.0, "k": 2, "retrieval_threshold": -0.3}
  }
}
```

`coax_strategies` overrides which strategy serves an `xai_type` at all. CoAX's
parameters are nested per `(xai_type, tested_w_xai)` condition, so — unlike
CoXAM and Sim2Real below — a design export's flat `cognitiveConfig` is **not**
applied to CoAX; `coax_params`/`coax_strategies` on the `/simulate` request
body are the only way to set them.

### CoXAM (`coxam_eval_params`) and Sim2Real (`sim2real_params`)

Both take a flat `{name: value}` map, and both accept **either** the agent's
own parameter name or the design-export UI's display label — the same
resolver the design export itself uses
(`src.experiment_planner.design_export.normalize_cognitive_params`), so a
`cognitiveConfig` block from the UI and a request body built by hand behave
identically:

```jsonc
// equivalent
{"sim2real_params": {"Max Features Attended": "4", "Aggregation Strategy": "value_weighted"}}
{"sim2real_params": {"max_features_attended": 4, "aggregation": "value_weighted"}}
```

String values are coerced to numbers/bools; a label that resolves to a name
the agent does not read is passed through with a warning in the design's
validation report rather than being dropped, so a typo does not silently
disappear.

If a design was created via `POST /api/studies` with a `cognitiveConfig`
block, that already resolved and coerced the values onto the study
(`study.cognitive_params`) — **but `/simulate` does not read them
automatically**; the same values (or your own overrides) must be repeated in
`coxam_eval_params` / `sim2real_params` on the `/simulate` request body, or the
agent runs on its defaults with no error.

CoXAM's evaluation-time parameters:

| parameter | UI label | range | default |
| --- | --- | --- | --- |
| `decision_noise` | Diffusion Noise | 0.3 – 0.7 | 0.4 |
| `memory_recall_threshold` | Retrieval Threshold | −1.0 – 2.0 (forward) / −2.0 – 0.5 (counterfactual) | 0.5 / −0.75 |
| `opportunity_cost` | Opportunity Cost | 0.0 – 0.02 | 0.01 |
| `random_response_rate` | — | 0.1 – 0.5 | 0.3 |
| `counterfactual_overshoot_fraction` | Counterfactual Margin | 0.0 – 0.5 | 0.25 |
| `time_penalty_weight` | — | 0.0 – 0.02 | 0.01 |

`memory_recall_threshold`'s range depends on the task (forward vs.
counterfactual) — there is no one bound that is safe for both. None of these
are clamped server-side today; an out-of-range value is passed straight to the
environment.

Sim2Real's three UI-exposed parameters (population-fitted, n = 46):

| parameter | UI label | evidence-supported range | default |
| --- | --- | --- | --- |
| `max_features_attended` | Max Features Attended | 1 – 12 (hard bound) | 4 |
| `aggregation` | Aggregation Strategy | `attribution` \| `value_weighted` | `value_weighted` |
| `confidence_intercept` | Confidence Responsiveness | −3.0 – 1.0 (no hard bound) | −1.5 |

`confidence_intercept` is presented as "responsiveness" for what it does, not
its name: more negative moves the model onto the steep part of its own
confidence curve, so it reacts *more*, not less, to a changed feature.

## Authentication

Set `XAIKIT_API_TOKEN` and every request needs `Authorization: Bearer <token>`;
`/api/health` and CORS preflights stay open so health checks and browsers still
work. Leave it unset and the server logs a warning at startup and accepts
everything — only sane on localhost, since each endpoint starts a training job
on demand.

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # generate one
```

## Deployment

Serve the static UI wherever you like (GitHub Pages works for that half) and
point it at this API over HTTPS — a page served over HTTPS cannot call a plain
HTTP API. The API itself needs a real Python host: the torch/captum/shap layer
is several GB and jobs run for minutes, which rules out static hosting and most
serverless platforms. No GPU is needed for the MLP.

On a small VM (2 vCPU / 4 GB is enough for the wine-quality design), with the
domain's A record already pointing at it:

```bash
git clone <repo> && cd xaikit-test-api
cat > .env <<'EOF'
XAIKIT_DOMAIN=api.example.org
XAIKIT_ALLOWED_ORIGINS=https://you.github.io
XAIKIT_API_TOKEN=<the generated token>
EOF
docker compose up -d --build
```

Caddy obtains the TLS certificate on first start and renews it. Study artifacts
live on the `study_data` volume, so a rebuild keeps finished runs.

The same `Dockerfile` is what Fly.io, Render and Railway consume, if you would
rather not run a VM. Two constraints carry over: keep it to **one** instance
(studies are in-process state, not shared), and attach a persistent volume at
`/data`.

### What is not in the image

Image layers travel wherever the image goes — a registry, a teammate's laptop,
the host's disk — and anyone who can pull it can read every file inside,
whether or not the API serves it. So `.dockerignore` excludes all human
participant data:

| Path | Why |
| --- | --- |
| `assets/human_data/` | Raw CoAX and CoXAM participant responses |
| `assets/human_trials_and_cognitive_parameters/` | Human trial records |
| `src/cognitive_models/CoAX/results/` | Per-participant fitted parameters and trial-by-trial records, keyed by Prolific id |
| `src/cognitive_models/CoAX/data/`, `CoAX/UI/`, `CoAX/simulation_mockup/` | Reference-study material the service does not use |
| `src/cognitive_models/CoXAM/datasets/`, `CoXAM/outputs/` | User data and sweep outputs |

The pipeline was run end to end against exactly this copy set — design import,
training, trials, explanations, CoAX simulation, analysis, plots — with none of
those paths present. `FITTED_COAX_PARAMS` holds the aggregate means derived from
the fitted table, so the runner needs no per-participant row at runtime.

Check before pushing an image anywhere:

```bash
docker run --rm --entrypoint sh xaikit-api -c \
  'find / -path /proc -prune -o \( -path "*human_data*" -o -path "*CoAX/results*" \) -print'
```

Empty output means clean. If you later add a feature that genuinely needs human
data, mount it as a read-only volume at runtime rather than copying it in — a
volume can be withheld per deployment, a layer cannot.
