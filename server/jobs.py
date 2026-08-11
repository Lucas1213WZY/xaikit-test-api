"""Study sessions and the background jobs that advance them.

A ``xaikitTest`` is stateful: the dataset, trained AI model, trials,
explanation table and simulated results all live on one object and each stage
reads what the previous stage stored. The server keeps that object per study
and runs stages as jobs, because training plus LIME/SHAP generation takes far
longer than an HTTP request may.

Jobs run on a single worker thread on purpose -- torch training and matplotlib
figure rendering are not safe to run concurrently in one process, and the study
object itself is mutated by every stage.
"""

from __future__ import annotations

import io
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


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


class StudyRegistry:
    """In-process store of studies and their jobs."""

    def __init__(self, root: Path, *, max_workers: int = 1) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._studies: dict[str, StudySession] = {}
        self._jobs: dict[str, Job] = {}
        self._guard = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="xaikit")

    # -- studies ---------------------------------------------------------
    def create(self, build: Callable[[Path], Any]) -> StudySession:
        """Register a study built by ``build(output_dir)``."""
        study_id = uuid.uuid4().hex[:12]
        output_dir = self.root / study_id
        output_dir.mkdir(parents=True, exist_ok=True)
        session = StudySession(study_id=study_id, study=build(output_dir), output_dir=output_dir)
        with self._guard:
            self._studies[study_id] = session
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
