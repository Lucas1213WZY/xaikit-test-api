"""HTTP surface for driving a XAIKit study from the experiment-design UI.

The flow mirrors the tutorial notebooks one stage at a time:

    POST /api/studies                    the design export -> a study
    POST /api/studies/{id}/dataset       prepare_dataset + train_AI_model
    POST /api/studies/{id}/trials        generate_trials
    POST /api/studies/{id}/explanations  explanations
    POST /api/studies/{id}/simulate      one trial, one participant, or all
    GET  /api/studies/{id}/results       step rows as JSON (paged)
    GET  /api/studies/{id}/results.csv   the same rows as a CSV download
    GET  /api/studies/{id}/analysis      analyze_iv_dv per IV x DV
    GET  /api/studies/{id}/posthoc       pairwise condition means + corrected p-values
    POST /api/studies/{id}/plots/...     aggregated plot data (PNG optional)

The four stage endpoints return a job immediately and do the work on a
background worker, because training and LIME/SHAP generation outlast any
sensible request timeout. Poll ``GET /api/jobs/{job_id}``.
"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Any, Optional

# Server processes have no display; select the non-interactive backend before
# anything imports pyplot.
import matplotlib

matplotlib.use("Agg")

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import pipeline
from .jobs import Job, StudyRegistry, StudySession
from .schemas import (
    AnalysisRequest,
    CreateStudyRequest,
    DatasetStageRequest,
    ExplanationStageRequest,
    GridPlotRequest,
    InteractionPlotRequest,
    PostHocRequest,
    SimulationRequest,
    TrialsStageRequest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("XAIKIT_SERVER_RUNS_DIR", REPO_ROOT / "server_runs"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("XAIKIT_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
#: Shared bearer token. Unset means no authentication, which is only sane on
#: localhost -- every endpoint here starts training jobs on demand.
API_TOKEN = os.environ.get("XAIKIT_API_TOKEN", "").strip()

#: Reachable without a token, so a load balancer can health-check the service.
PUBLIC_PATHS = {"/api/health"}

logger = logging.getLogger(__name__)

app = FastAPI(
    title="XAIKit study API",
    description=__doc__,
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    # The UI reads the download filename off this header.
    expose_headers=["Content-Disposition"],
)

registry = StudyRegistry(RUNS_ROOT)


@app.middleware("http")
async def require_token(request: Request, call_next):
    """Check the shared bearer token, when one is configured.

    Registered after CORSMiddleware, which makes it the outer layer, so
    preflight OPTIONS requests are let through -- browsers send them without
    an Authorization header and would otherwise never reach the CORS handler.
    """
    if (
        API_TOKEN
        and request.method != "OPTIONS"
        and request.url.path not in PUBLIC_PATHS
    ):
        header = request.headers.get("authorization", "")
        if not secrets.compare_digest(header, f"Bearer {API_TOKEN}"):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid bearer token."},
            )
    return await call_next(request)


@app.on_event("startup")
async def warn_when_unauthenticated() -> None:
    if not API_TOKEN:
        logger.warning(
            "XAIKIT_API_TOKEN is not set: every endpoint is open, and each one "
            "can start a training job. Set it before exposing this server."
        )


def _session(study_id: str) -> StudySession:
    try:
        return registry.get(study_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown study {study_id!r}. Studies live only in memory and do "
                "not survive a server restart or redeploy -- if this id worked "
                "before, POST /api/studies again to create a new one."
            ),
        )


def _job_response(job: Job) -> dict[str, Any]:
    return job.payload()


def _guard(call) -> Any:
    """Turn API-layer usage errors into 4xx instead of a 500."""
    try:
        return call()
    except (ValueError, KeyError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "studies": len(registry.list())}


# -- studies --------------------------------------------------------------


@app.post("/api/studies", status_code=201)
def create_study(request: CreateStudyRequest) -> dict[str, Any]:
    """Register a design export and return its normalized, validated view."""

    def build(output_dir: Path):
        return pipeline.build_study(
            request.design,
            project_name=request.project_name,
            output_dir=output_dir,
        )

    session = _guard(lambda: registry.create(build))
    return {
        **session.summary(),
        **_guard(lambda: pipeline.study_design_payload(session.study)),
    }


@app.get("/api/studies")
def list_studies() -> dict[str, Any]:
    return {"studies": [session.summary() for session in registry.list()]}


@app.get("/api/studies/{study_id}")
def get_study(study_id: str) -> dict[str, Any]:
    session = _session(study_id)
    return {
        **session.summary(),
        **pipeline.study_design_payload(session.study),
        "stages": session.stages,
        "jobs": [job.payload(log_tail=0) for job in registry.jobs_for(study_id)],
    }


@app.delete("/api/studies/{study_id}", status_code=204)
def delete_study(study_id: str) -> Response:
    _session(study_id)
    registry.delete(study_id)
    return Response(status_code=204)


# -- stages ---------------------------------------------------------------


@app.post("/api/studies/{study_id}/dataset", status_code=202)
def start_dataset_stage(
    study_id: str,
    request: DatasetStageRequest = Body(default=DatasetStageRequest()),
) -> dict[str, Any]:
    """Prepare the dataset and train the AI model that will be explained."""
    session = _session(study_id)
    job = registry.submit(
        session,
        "dataset",
        lambda item: pipeline.run_dataset_stage(item.study, request),
    )
    return _job_response(job)


@app.post("/api/studies/{study_id}/trials", status_code=202)
def start_trials_stage(
    study_id: str,
    request: TrialsStageRequest = Body(default=TrialsStageRequest()),
) -> dict[str, Any]:
    """Generate the balanced trial table for every participant."""
    session = _session(study_id)
    job = registry.submit(
        session,
        "trials",
        lambda item: pipeline.run_trials_stage(item.study, request),
    )
    return _job_response(job)


@app.post("/api/studies/{study_id}/explanations", status_code=202)
def start_explanations_stage(
    study_id: str,
    request: ExplanationStageRequest = Body(default=ExplanationStageRequest()),
) -> dict[str, Any]:
    """Generate one XAI table per method the design names."""
    session = _session(study_id)
    job = registry.submit(
        session,
        "explanations",
        lambda item: pipeline.run_explanations_stage(item.study, request),
    )
    return _job_response(job)


@app.post("/api/studies/{study_id}/simulate", status_code=202)
def start_simulation(
    study_id: str,
    request: SimulationRequest = Body(default=SimulationRequest()),
) -> dict[str, Any]:
    """Run virtual participants over one trial, one participant, or all of them."""
    session = _session(study_id)
    job = registry.submit(
        session,
        f"simulate:{request.mode}",
        lambda item: pipeline.run_simulation_stage(
            item.study,
            request,
            output_subdir=f"simulated_results/{request.mode}",
        ),
    )
    return _job_response(job)


# -- jobs -----------------------------------------------------------------


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, log_tail: int = Query(40, ge=0, le=5000)) -> dict[str, Any]:
    """Poll a stage: state, elapsed time, captured progress output, result."""
    try:
        job = registry.job(job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown job {job_id!r}. Jobs live only in memory and do not "
                "survive a server restart or redeploy -- the study that owned "
                "this job is gone too; POST /api/studies again to start over."
            ),
        )
    return job.payload(log_tail=log_tail or None)


@app.get("/api/studies/{study_id}/jobs")
def list_jobs(study_id: str) -> dict[str, Any]:
    _session(study_id)
    return {"jobs": [job.payload(log_tail=0) for job in registry.jobs_for(study_id)]}


# -- results --------------------------------------------------------------


@app.get("/api/studies/{study_id}/results")
def get_results(
    study_id: str,
    phase: Optional[str] = None,
    participant_id: Optional[int] = None,
    limit: Optional[int] = Query(None, ge=1, le=100000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Trial-by-trial step rows of the most recent run, paged for step-through."""
    session = _session(study_id)
    return _guard(
        lambda: pipeline.results_payload(
            session.study,
            phase=phase,
            participant_id=participant_id,
            limit=limit,
            offset=offset,
        )
    )


@app.get("/api/studies/{study_id}/results.csv")
def download_results_csv(
    study_id: str,
    phase: Optional[str] = None,
    participant_id: Optional[int] = None,
) -> Response:
    """The same rows as a CSV attachment the browser can save."""
    session = _session(study_id)
    csv_text = _guard(
        lambda: pipeline.results_csv(
            session.study, phase=phase, participant_id=participant_id
        )
    )
    filename = f"simulated_results_{study_id}.csv"
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# -- human-vs-model comparison ---------------------------------------------


@app.get("/api/human-comparison/{study}")
def get_human_comparison(study: str) -> dict[str, Any]:
    """A precomputed study's human-vs-model PNG and NLL/BIC table.

    Not tied to any running study session -- ``study`` is ``coax``, ``coxam``
    or ``sim2real``, matching the files ``assets/build_human_vs_model_plots.py``
    and ``assets/build_human_vs_model_fit_stats.py`` already wrote to disk.
    Nothing is fitted or rendered here, only read back.
    """
    payload = pipeline.human_comparison_payload(study)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No precomputed human comparison for {study!r}. Use one of "
                f"{pipeline.HUMAN_COMPARISON_STUDIES}, and make sure "
                "assets/build_human_vs_model_plots.py has been run."
            ),
        )
    return payload


# -- analysis and plots ---------------------------------------------------


@app.post("/api/studies/{study_id}/analysis")
def post_analysis(
    study_id: str,
    request: AnalysisRequest = Body(default=AnalysisRequest()),
) -> dict[str, Any]:
    """Descriptives and the inferential test for each IV x DV pair."""
    session = _session(study_id)
    return _guard(
        lambda: pipeline.analysis_for(session.study, ivs=request.ivs, dvs=request.dvs)
    )


@app.get("/api/studies/{study_id}/analysis")
def get_analysis(
    study_id: str,
    iv: Optional[list[str]] = Query(None),
    dv: Optional[list[str]] = Query(None),
) -> dict[str, Any]:
    """Same as the POST form, for links the UI can bookmark."""
    session = _session(study_id)
    return _guard(lambda: pipeline.analysis_for(session.study, ivs=iv, dvs=dv))


@app.post("/api/studies/{study_id}/posthoc")
def post_posthoc(
    study_id: str,
    request: PostHocRequest,
) -> dict[str, Any]:
    """Pairwise condition means and corrected p-values for one DV."""
    session = _session(study_id)
    return _guard(
        lambda: pipeline.posthoc_for(
            session.study,
            dv=request.dv,
            condition_cols=request.condition_cols,
            correction=request.correction,
            phase=request.phase,
        )
    )


@app.get("/api/studies/{study_id}/posthoc")
def get_posthoc(
    study_id: str,
    dv: str = Query(...),
    condition: Optional[list[str]] = Query(None),
    correction: Optional[str] = Query("holm"),
    phase: str = Query("testing"),
) -> dict[str, Any]:
    """Same as the POST form, for links the UI can bookmark."""
    session = _session(study_id)
    return _guard(
        lambda: pipeline.posthoc_for(
            session.study,
            dv=dv,
            condition_cols=condition,
            correction=correction,
            phase=phase,
        )
    )


@app.post("/api/studies/{study_id}/plots/interaction")
def post_interaction_plot(
    study_id: str,
    request: InteractionPlotRequest,
) -> dict[str, Any]:
    """Aggregated bars for one DV against two IVs."""
    session = _session(study_id)
    return _guard(
        lambda: pipeline.interaction_plot_payload(
            session.study,
            x_iv=request.x_iv,
            hue_iv=request.hue_iv,
            dv=request.dv,
            phase=request.phase,
            errorbar=request.errorbar,
            title=request.title,
            include_png=request.include_png,
        )
    )


@app.post("/api/studies/{study_id}/plots/grid")
def post_grid_plot(
    study_id: str,
    request: GridPlotRequest = Body(default=GridPlotRequest()),
) -> dict[str, Any]:
    """Aggregated bars for every DV against every IV in the design."""
    session = _session(study_id)
    return _guard(
        lambda: pipeline.grid_plot_payload(
            session.study,
            ivs=request.ivs,
            dvs=request.dvs,
            phase=request.phase,
            errorbar=request.errorbar,
            title=request.title,
            include_png=request.include_png,
        )
    )
