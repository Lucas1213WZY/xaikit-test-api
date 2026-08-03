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
POST /api/studies/{id}/analysis       analyze_iv_dv per IV x DV
POST /api/studies/{id}/plots/interaction   one DV by two IVs
POST /api/studies/{id}/plots/grid          every DV by every IV
```

The four stage endpoints return `202` with a job immediately — training and
LIME/SHAP generation outlast any sensible request timeout — so poll
`GET /api/jobs/{job_id}` until `state` is `succeeded` or `failed`. The job's
`log` field carries the API layer's own progress output.

Anything the design export already answers can be omitted from a request body:
dataset id, participants per condition, XAI methods and DVs are read from the
export. `userModel` in the export selects the participant runner — `"CoAX"`
routes to `run_coax_study`, anything else to `study.run_experiment` with a
baseline model.

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

## CoAX parameters

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

`coax_strategies` overrides which strategy serves an `xai_type` at all.

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
