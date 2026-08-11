"""StudyRegistry persists studies/jobs to disk and reloads them on restart.

Regression for the "404: Unknown job ... Jobs live only in memory" report --
a Render redeploy (which happens on every push, per autoDeployTrigger: commit)
used to wipe every in-progress study. StudySession.study is a plain Python
object (proven pickle-safe against a real trained xaikitTest separately), so
each study is pickled/JSON-dumped into its own output_dir after every job and
reloaded from there when a fresh StudyRegistry is constructed against the same
root -- simulating what happens across a process restart.
"""

import json
import time

from server.jobs import STUDY_PICKLE_NAME, Job, StudyRegistry


class _FakeDesign:
    study_title = "A title"
    model_framework = "coax"


class _FakeStudy:
    """Picklable stand-in for xaikitTest -- only what StudySession.summary() reads."""

    def __init__(self, project_name: str = "demo") -> None:
        self.project_name = project_name
        self.design_export = _FakeDesign()
        self.data = None
        self.data_by_dataset = {}
        self.trained_ai_model = None
        self.trials = []
        self.combined_explanations = None
        self.simulated_results = None


def test_a_study_and_its_completed_job_survive_a_simulated_restart(tmp_path):
    registry = StudyRegistry(tmp_path)
    session = registry.create(lambda output_dir: _FakeStudy())
    job = registry.submit(session, "dataset", lambda item: {"ok": True})
    job._future.result(timeout=10)
    assert job.state == "succeeded"

    reloaded = StudyRegistry(tmp_path)
    restored_session = reloaded.get(session.study_id)
    assert restored_session.study.project_name == "demo"
    assert restored_session.stages == {"dataset": {"ok": True}}

    restored_job = reloaded.job(job.job_id)
    assert restored_job.state == "succeeded"
    assert restored_job.result == {"ok": True}


def test_a_job_still_running_at_restart_is_marked_failed_not_lost(tmp_path):
    registry = StudyRegistry(tmp_path)
    session = registry.create(lambda output_dir: _FakeStudy())

    stuck_job = Job(job_id="stuck123", study_id=session.study_id, stage="dataset")
    stuck_job.state = "running"
    stuck_job.started_at = time.time()
    registry._persist_job(stuck_job)

    reloaded = StudyRegistry(tmp_path)
    restored = reloaded.job("stuck123")
    assert restored.state == "failed"
    assert "restart" in restored.error.lower()


def test_delete_removes_the_study_from_disk_so_it_does_not_come_back(tmp_path):
    registry = StudyRegistry(tmp_path)
    session = registry.create(lambda output_dir: _FakeStudy())
    pickle_path = session.output_dir / STUDY_PICKLE_NAME
    assert pickle_path.is_file()

    registry.delete(session.study_id)
    assert not pickle_path.is_file()

    reloaded = StudyRegistry(tmp_path)
    try:
        reloaded.get(session.study_id)
        assert False, "deleted study should not be restored"
    except KeyError:
        pass


def test_a_corrupted_study_pickle_is_skipped_not_fatal(tmp_path):
    registry = StudyRegistry(tmp_path)
    session = registry.create(lambda output_dir: _FakeStudy())
    (session.output_dir / STUDY_PICKLE_NAME).write_bytes(b"not a pickle")

    # Must not raise -- a broken study is logged and skipped, the registry
    # still comes up so every other study stays reachable.
    reloaded = StudyRegistry(tmp_path)
    try:
        reloaded.get(session.study_id)
        assert False, "a corrupted pickle should not resolve to a session"
    except KeyError:
        pass


def test_jobs_json_stays_valid_json_after_several_jobs(tmp_path):
    registry = StudyRegistry(tmp_path)
    session = registry.create(lambda output_dir: _FakeStudy())
    for stage in ("dataset", "trials", "explanations"):
        job = registry.submit(session, stage, lambda item: {"stage": "done"})
        job._future.result(timeout=10)

    records = json.loads((session.output_dir / "jobs.json").read_text())
    assert {record["stage"] for record in records} == {"dataset", "trials", "explanations"}
    assert all(record["state"] == "succeeded" for record in records)
