"""Study sessions and the background jobs that advance them.

A ``xaikitTest`` is stateful: the dataset, trained AI model, trials,
explanation table and simulated results all live on one object and each stage
reads what the previous stage stored. The server keeps that object per study
and runs stages as jobs, because training plus LIME/SHAP generation takes far
longer than an HTTP request may.

Jobs run on a single worker thread on purpose -- torch training and matplotlib
figure rendering are not safe to run concurrently in one process, and the study
object itself is mutated by every stage.

Every study and its job history is pickled/JSON-dumped into its own
``output_dir`` after each successful step, and reloaded from there at startup.
``output_dir`` lives under ``XAIKIT_SERVER_RUNS_DIR``, which on a paid Render
plan is a persistent disk mounted at ``/data`` -- so a redeploy or restart no
longer wipes every in-flight study, only the one job that happened to be
actively running at the moment of the restart (which genuinely cannot be
resumed, since the thread doing the work is gone; it is marked failed instead
of silently vanishing).
"""

from __future__ import annotations

import io
import json
import logging
import pickle
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

STUDY_PICKLE_NAME = "study.pkl"
STUDY_META_NAME = "meta.json"
JOBS_RECORD_NAME = "jobs.json"


@dataclass
class StudySession:
    """One study: the API-layer object plus where its artifacts are written."""

    study_id: str
    study: Any
    output_dir: Path
    created_at: float = field(default_factory=time.time)
    #: Stage name -> summary payload of the last successful run of that stage.
    stages: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def summary(self) -> dict[str, Any]:
        study = self.study
        design = getattr(study, "design_export", None)
        return {
            "study_id": self.study_id,
            "project_name": study.project_name,
            "study_title": getattr(design, "study_title", None),
            "model_framework": getattr(design, "model_framework", None),
            "created_at": self.created_at,
            "output_dir": str(self.output_dir),
            "completed_stages": sorted(self.stages),
            "state": {
                "dataset_ready": (
                    study.data is not None or bool(getattr(study, "data_by_dataset", None))
                ),
                "model_trained": study.trained_ai_model is not None,
                "trials_generated": bool(study.trials),
                "explanations_generated": study.combined_explanations is not None,
                "simulated": study.simulated_results is not None,
            },
        }


@dataclass
class Job:
    """One queued or finished stage run."""

    job_id: str
    study_id: str
    stage: str
    state: str = "queued"  # queued | running | succeeded | failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    _log: io.StringIO = field(default_factory=io.StringIO, repr=False)
    _future: Optional[Future] = field(default=None, repr=False)

    def log_text(self, tail: Optional[int] = None) -> str:
        """Captured stdout of the stage -- the API layer's progress output."""
        text = self._log.getvalue()
        if tail is None:
            return text
        return "\n".join(text.splitlines()[-tail:])

    def payload(self, *, log_tail: Optional[int] = 40) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "study_id": self.study_id,
            "stage": self.stage,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": (
                None
                if self.started_at is None
                else (self.finished_at or time.time()) - self.started_at
            ),
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
            "log": self.log_text(tail=log_tail),
        }

    def to_record(self) -> dict[str, Any]:
        """Everything needed to rehydrate this job after a restart."""
        return {
            "job_id": self.job_id,
            "study_id": self.study_id,
            "stage": self.stage,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
            "log": self.log_text(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Job":
        job = cls(job_id=record["job_id"], study_id=record["study_id"], stage=record["stage"])
        job.state = record.get("state", "failed")
        job.created_at = record.get("created_at", time.time())
        job.started_at = record.get("started_at")
        job.finished_at = record.get("finished_at")
        job.result = record.get("result")
        job.error = record.get("error")
        job.traceback = record.get("traceback")
        job._log = io.StringIO(record.get("log", ""))
        if job.state in ("queued", "running"):
            # The thread that was running this job is gone -- it cannot be
            # resumed, so say so instead of leaving it stuck "running" forever.
            job.state = "failed"
            job.error = "Interrupted by a server restart before this job finished."
            job.finished_at = job.finished_at or time.time()
        return job


class StudyRegistry:
    """In-process store of studies and their jobs, mirrored to disk."""

    def __init__(self, root: Path, *, max_workers: int = 1) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._studies: dict[str, StudySession] = {}
        self._jobs: dict[str, Job] = {}
        self._guard = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="xaikit")
        self._load_existing()

    # -- persistence -------------------------------------------------------

    def _pickle_path(self, study_id: str) -> Path:
        return self.root / study_id / STUDY_PICKLE_NAME

    def _meta_path(self, study_id: str) -> Path:
        return self.root / study_id / STUDY_META_NAME

    def _jobs_path(self, study_id: str) -> Path:
        return self.root / study_id / JOBS_RECORD_NAME

    def _persist_study(self, session: StudySession) -> None:
        try:
            session.output_dir.mkdir(parents=True, exist_ok=True)
            pickle_path = self._pickle_path(session.study_id)
            tmp_path = pickle_path.with_suffix(".tmp")
            with tmp_path.open("wb") as fh:
                pickle.dump(session.study, fh)
            tmp_path.replace(pickle_path)
            self._meta_path(session.study_id).write_text(
                json.dumps({"created_at": session.created_at, "stages": session.stages})
            )
        except Exception:  # noqa: BLE001 - persistence must never break a request
            logger.exception("Failed to persist study %s to disk", session.study_id)

    def _persist_job(self, job: Job) -> None:
        try:
            jobs_path = self._jobs_path(job.study_id)
            jobs_path.parent.mkdir(parents=True, exist_ok=True)
            records: list[dict[str, Any]] = []
            if jobs_path.is_file():
                try:
                    records = json.loads(jobs_path.read_text())
                except (json.JSONDecodeError, OSError):
                    records = []
            records = [r for r in records if r.get("job_id") != job.job_id]
            records.append(job.to_record())
            tmp_path = jobs_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(records))
            tmp_path.replace(jobs_path)
        except Exception:  # noqa: BLE001 - persistence must never break a request
            logger.exception("Failed to persist job %s to disk", job.job_id)

    def _load_existing(self) -> None:
        if not self.root.is_dir():
            return
        for entry in sorted(self.root.iterdir()):
            pickle_path = entry / STUDY_PICKLE_NAME
            if not entry.is_dir() or not pickle_path.is_file():
                continue
            study_id = entry.name
            try:
                with pickle_path.open("rb") as fh:
                    study = pickle.load(fh)
                meta: dict[str, Any] = {}
                meta_path = self._meta_path(study_id)
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text())
                session = StudySession(
                    study_id=study_id,
                    study=study,
                    output_dir=entry,
                    created_at=meta.get("created_at", time.time()),
                    stages=meta.get("stages", {}),
                )
                self._studies[study_id] = session

                jobs_path = self._jobs_path(study_id)
                if jobs_path.is_file():
                    for record in json.loads(jobs_path.read_text()):
                        job = Job.from_record(record)
                        self._jobs[job.job_id] = job
            except Exception:  # noqa: BLE001 - one bad study must not block the rest
                logger.exception("Failed to restore study %s from disk -- skipping it", study_id)

    # -- studies ---------------------------------------------------------
    def create(self, build: Callable[[Path], Any]) -> StudySession:
        """Register a study built by ``build(output_dir)``."""
        study_id = uuid.uuid4().hex[:12]
        output_dir = self.root / study_id
        output_dir.mkdir(parents=True, exist_ok=True)
        session = StudySession(study_id=study_id, study=build(output_dir), output_dir=output_dir)
        with self._guard:
            self._studies[study_id] = session
        self._persist_study(session)
        return session

    def get(self, study_id: str) -> StudySession:
        with self._guard:
            session = self._studies.get(study_id)
        if session is None:
            raise KeyError(study_id)
        return session

    def list(self) -> list[StudySession]:
        with self._guard:
            return sorted(self._studies.values(), key=lambda item: item.created_at)

    def delete(self, study_id: str) -> None:
        with self._guard:
            self._studies.pop(study_id, None)
            self._jobs = {
                job_id: job for job_id, job in self._jobs.items() if job.study_id != study_id
            }
        shutil.rmtree(self.root / study_id, ignore_errors=True)

    # -- jobs ------------------------------------------------------------
    def submit(
        self,
        session: StudySession,
        stage: str,
        run: Callable[[StudySession], Any],
    ) -> Job:
        """Queue a stage for a study and return its job immediately."""
        job = Job(job_id=uuid.uuid4().hex[:12], study_id=session.study_id, stage=stage)
        with self._guard:
            self._jobs[job.job_id] = job
        job._future = self._pool.submit(self._run, job, session, run)
        return job

    def job(self, job_id: str) -> Job:
        with self._guard:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def jobs_for(self, study_id: str) -> list[Job]:
        with self._guard:
            jobs = [job for job in self._jobs.values() if job.study_id == study_id]
        return sorted(jobs, key=lambda item: item.created_at)

    def _run(self, job: Job, session: StudySession, run: Callable[[StudySession], Any]) -> None:
        job.state = "running"
        job.started_at = time.time()
        try:
            # The API layer reports progress by printing; keep it as the job log
            # instead of losing it to the server's own stdout.
            with session.lock, redirect_stdout(job._log):
                result = run(session)
            job.result = result
            job.state = "succeeded"
            session.stages[job.stage] = result
        except BaseException as error:  # noqa: BLE001 - surfaced to the client
            job.state = "failed"
            job.error = f"{type(error).__name__}: {error}"
            job.traceback = traceback.format_exc()
        finally:
            job.finished_at = time.time()
            self._persist_study(session)
            self._persist_job(job)
