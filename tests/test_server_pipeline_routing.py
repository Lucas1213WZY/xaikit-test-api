"""Which participant runner the server picks for a design's userModel.

Regression coverage for a real routing bug: ``participant_runner`` used to
re-derive the framework from the raw ``model_framework`` string instead of
``DesignExport.resolved_framework``, so it never applied the xai_property
discriminator. A Sim2Real design exported as ``userModel: "CoAX"`` (correct --
its cognitive model is a CoAX-derived attribution sum) or
``userModel: "CoAX (XAI Property)"`` both silently fell through to the generic
baseline runner and produced results that looked real but were not Sim2Real's.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from server.pipeline import (
    design_framework,
    participant_runner,
    resolve_baseline_model_id,
    run_explanations_stage,
)
from server.schemas import ExplanationStageRequest
from src.experiment_planner.design_export import DesignExport


@dataclass
class _FakeStudy:
    design_export: DesignExport


def _study(model_framework: str, *, xai_property: bool = False) -> _FakeStudy:
    """A study whose design carries only what routing reads.

    Builds the real ``DesignExport``, not a hand-rolled double, so these tests
    exercise ``resolved_framework`` exactly as the server does and cannot drift
    from it the way a parallel fake could.
    """
    ivs = [{"name": "xai_property", "type": "within", "levels": []}] if xai_property else []
    design = DesignExport(
        raw={},
        study_title="",
        research_questions=[],
        consent_text="",
        procedure_steps=[],
        ivs=ivs,
        model_framework=model_framework,
    )
    return _FakeStudy(design_export=design)


def test_plain_coax_routes_to_coax():
    assert participant_runner(_study("CoAX")) == "coax"


def test_coax_with_the_xai_property_iv_routes_to_sim2real():
    """The documented case: Sim2Real designs export userModel="CoAX"."""
    assert participant_runner(_study("CoAX", xai_property=True)) == "sim2real"


def test_coax_xai_property_label_routes_to_sim2real():
    """The UI's actual label. Slugified alone this is 'coax_xai_property',
    matching neither 'coax' nor 'sim2real' -- the IV discriminator must carry it."""
    assert participant_runner(_study("CoAX (XAI Property)", xai_property=True)) == "sim2real"


def test_coax_xai_property_label_resolves_even_if_the_iv_did_not_parse():
    """Belt and suspenders: the label carries its own discriminator
    (MODEL_FRAMEWORK_ALIASES), so it must not depend on xai_property having
    parsed into design.ivs -- IV parsing is a separate, fallible step."""
    assert participant_runner(_study("CoAX (XAI Property)", xai_property=False)) == "sim2real"


def test_plain_coax_without_the_iv_is_not_coerced_to_sim2real():
    """The alias is specific to the '(XAI Property)' label -- plain 'CoAX'
    must still depend on the IV, or every real CoAX design breaks."""
    assert participant_runner(_study("CoAX", xai_property=False)) == "coax"


def test_coxam_routes_to_coxam():
    assert participant_runner(_study("CoXAM")) == "coxam"


def test_plain_sim2real_label_routes_to_sim2real():
    assert participant_runner(_study("Sim2Real")) == "sim2real"


def test_an_unrecognised_model_routes_to_baseline():
    assert participant_runner(_study("KNN")) == "baseline"


def test_design_framework_matches_resolved_framework():
    """The server must not maintain a second, divergent notion of framework."""
    study = _study("CoAX (XAI Property)", xai_property=True)
    assert design_framework(study) == study.design_export.resolved_framework


def test_resolve_baseline_model_id_defers_to_a_real_runner():
    """Sim2Real/CoAX/CoXAM designs must never be coerced into the baseline path."""
    assert resolve_baseline_model_id(_study("CoAX", xai_property=True), None) is None
    assert resolve_baseline_model_id(_study("CoXAM"), None) is None
    assert resolve_baseline_model_id(_study("CoAX"), None) is None


def test_resolve_baseline_model_id_still_resolves_a_real_baseline():
    assert resolve_baseline_model_id(_study("KNN"), None) == "knn"


def test_an_explicit_request_override_always_wins():
    assert resolve_baseline_model_id(_study("CoAX (XAI Property)", xai_property=True), "knn") == "knn"


# -- Sim2Real skips AI training, and trial balancing follows that -------


def test_run_dataset_stage_skips_training_for_sim2real():
    """Sim2Real reads a fixed published corpus, never trained_ai_model -- so
    training one is real, unused compute this must not spend."""
    from unittest.mock import MagicMock

    from server.pipeline import run_dataset_stage
    from server.schemas import DatasetStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[{"name": "xai_property", "type": "within", "levels": []}],
        model_framework="CoAX", dataset_id="adult",
    )
    fake_data = MagicMock(
        dataset_id="adult", feature_names=["a"], model_feature_names=["a"],
        y_train=[0, 1], y_test=[0],
    )
    study.prepare_dataset.return_value = fake_data

    result = run_dataset_stage(study, DatasetStageRequest())

    study.train_AI_model.assert_not_called()
    study.evaluate.assert_not_called()
    assert result["model"] is None
    assert "sim2real" in result["model_skipped_reason"].lower()


def test_run_dataset_stage_still_trains_for_a_real_baseline():
    from unittest.mock import MagicMock

    from server.pipeline import run_dataset_stage
    from server.schemas import DatasetStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN", dataset_id="adult",
    )
    fake_data = MagicMock(
        dataset_id="adult", feature_names=["a"], model_feature_names=["a"],
        y_train=[0, 1], y_test=[0],
    )
    study.prepare_dataset.return_value = fake_data
    study.model_name = "knn"
    study.test_accuracy.return_value = 0.9
    study.training_summary_table.return_value.to_dict.return_value = []
    study.training_history_table.return_value.to_dict.return_value = []
    study.metrics_table.return_value.reset_index.return_value.to_dict.return_value = []
    study.confusion_matrix_table.return_value.reset_index.return_value.to_dict.return_value = []

    run_dataset_stage(study, DatasetStageRequest())
    study.train_AI_model.assert_called_once()


def test_trial_balancing_defaults_off_when_no_model_was_trained():
    """The Sim2Real case: /dataset skipped training, so /trials must not
    silently require a model that does not exist."""
    from unittest.mock import MagicMock

    from server.pipeline import run_trials_stage
    from server.schemas import TrialsStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX",
        participants_per_condition=2, trials_per_participant=8,
    )
    study.trained_ai_model = None
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest())
    except Exception:
        pass  # only the call into generate_trials is under test here

    assert study.generate_trials.call_args.kwargs["balance_by_ai_prediction"] is False


def test_trial_balancing_defaults_on_when_a_model_was_trained():
    from unittest.mock import MagicMock

    from server.pipeline import run_trials_stage
    from server.schemas import TrialsStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN",
        participants_per_condition=2, trials_per_participant=8,
    )
    study.trained_ai_model = object()
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest())
    except Exception:
        pass  # only the call into generate_trials is under test here

    assert study.generate_trials.call_args.kwargs["balance_by_ai_prediction"] is True


def test_an_explicit_balance_request_is_never_overridden():
    """True must still surface a clear error against an untrained model,
    rather than the server quietly picking False on the caller's behalf."""
    from unittest.mock import MagicMock

    from server.pipeline import run_trials_stage
    from server.schemas import TrialsStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX",
        participants_per_condition=2, trials_per_participant=8,
    )
    study.trained_ai_model = None
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest(balance_by_ai_prediction=True))
    except Exception:
        pass  # only the call into generate_trials is under test here

    assert study.generate_trials.call_args.kwargs["balance_by_ai_prediction"] is True


def test_an_unsupported_framework_raises_rather_than_silently_running_baseline():
    from server import pipeline

    original = pipeline.UNSUPPORTED_FRAMEWORKS
    pipeline.UNSUPPORTED_FRAMEWORKS = {"ebm"}
    try:
        with pytest.raises(ValueError, match="no virtual-participant runner"):
            resolve_baseline_model_id(_study("EBM"), None)
    finally:
        pipeline.UNSUPPORTED_FRAMEWORKS = original


# -- CoXAM skips training when its corpus covers the dataset ------------


def test_coxam_skips_training_when_the_corpus_covers_the_dataset():
    from unittest.mock import MagicMock

    from server.pipeline import run_dataset_stage
    from server.schemas import DatasetStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoXAM", dataset_id="wine_quality",
    )
    study.prepare_dataset.return_value = MagicMock(
        dataset_id="wine_quality", feature_names=["a"], model_feature_names=["a"],
        y_train=[0, 1], y_test=[0],
    )

    result = run_dataset_stage(study, DatasetStageRequest())

    study.train_AI_model.assert_not_called()
    assert result["model"] is None
    assert "corpus covers" in result["model_skipped_reason"]
    assert study.model_name == "mlp"  # request.model_type's default


def test_coxam_still_trains_for_a_dataset_outside_the_corpus():
    from unittest.mock import MagicMock

    from server.pipeline import run_dataset_stage
    from server.schemas import DatasetStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoXAM", dataset_id="prima_diabetes",
    )
    study.prepare_dataset.return_value = MagicMock(
        dataset_id="prima_diabetes", feature_names=["a"], model_feature_names=["a"],
        y_train=[0, 1], y_test=[0],
    )
    study.model_name = "mlp"
    study.test_accuracy.return_value = 0.9
    for attr in ("training_summary_table", "training_history_table"):
        getattr(study, attr).return_value.to_dict.return_value = []
    for attr in ("metrics_table", "confusion_matrix_table"):
        getattr(study, attr).return_value.reset_index.return_value.to_dict.return_value = []

    run_dataset_stage(study, DatasetStageRequest())
    study.train_AI_model.assert_called_once()


def test_coxam_source_defaults_to_assets_with_no_trained_model():
    from unittest.mock import MagicMock, patch

    from server.pipeline import run_simulation_stage
    from server.schemas import SimulationRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoXAM",
    )
    study.trained_ai_model = None
    study.save_results.return_value = ("x.csv", "x.json")

    with patch("server.pipeline.run_coxam_study") as mocked:
        mocked.return_value = MagicMock(empty=False)
        run_simulation_stage(study, SimulationRequest(), output_subdir="x")
        assert mocked.call_args.kwargs["source"] == "assets"


def test_coxam_source_defaults_to_fit_with_a_trained_model():
    from unittest.mock import MagicMock, patch

    from server.pipeline import run_simulation_stage
    from server.schemas import SimulationRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoXAM",
    )
    study.trained_ai_model = object()
    study.save_results.return_value = ("x.csv", "x.json")

    with patch("server.pipeline.run_coxam_study") as mocked:
        mocked.return_value = MagicMock(empty=False)
        run_simulation_stage(study, SimulationRequest(), output_subdir="x")
        assert mocked.call_args.kwargs["source"] == "fit"


# -- apparatus-declared instances override the dataset's own train/test split


def test_apparatus_test_instances_are_not_filtered_out_by_the_dataset_split():
    """Regression: filtering the dataset's own random split down to the
    apparatus's declared ids silently kept only whichever fraction happened to
    land in that split's test portion (10 of 20 for a real design) instead of
    using all 20 the apparatus actually declared."""
    from unittest.mock import MagicMock

    import numpy as np

    from server.pipeline import run_trials_stage
    from server.schemas import TrialsStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN",
        participants_per_condition=1, trials_per_participant=8,
        apparatus_instance_ids=[1, 2, 3, 4, 5],
        apparatus_training_instance_ids=[],
    )
    # A dataset split that overlaps the apparatus ids by only one instance --
    # the old behavior would have shrunk testing to just that one id.
    study.data.split.train_instance_ids = np.array([1, 100, 101, 102])
    study.data.split.test_instance_ids = np.array([2, 200, 201, 202])
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest(num_training=2, num_testing=5))
    except Exception:
        pass  # only the split override is under test here

    assert sorted(study.data.split.test_instance_ids.tolist()) == [1, 2, 3, 4, 5]
    train_ids = set(study.data.split.train_instance_ids.tolist())
    assert len(train_ids) == 2
    assert train_ids.isdisjoint({1, 2, 3, 4, 5})


def test_apparatus_training_ids_are_used_directly_when_declared():
    from unittest.mock import MagicMock

    import numpy as np

    from server.pipeline import run_trials_stage
    from server.schemas import TrialsStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN",
        participants_per_condition=1, trials_per_participant=8,
        apparatus_instance_ids=[10, 11, 12],
        apparatus_training_instance_ids=[0, 1],
    )
    study.data.split.train_instance_ids = np.array([0, 1, 2])
    study.data.split.test_instance_ids = np.array([10, 11])
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest(num_training=2, num_testing=3))
    except Exception:
        pass

    assert sorted(study.data.split.train_instance_ids.tolist()) == [0, 1]


def test_too_few_remaining_instances_for_training_raises_clearly():
    from unittest.mock import MagicMock

    import numpy as np

    from server.pipeline import run_trials_stage
    from server.schemas import TrialsStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN",
        participants_per_condition=1, trials_per_participant=8,
        apparatus_instance_ids=[1, 2, 3],
        apparatus_training_instance_ids=[],
    )
    study.data.split.train_instance_ids = np.array([1, 2, 3])
    study.data.split.test_instance_ids = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="no overlap allowed"):
        run_trials_stage(study, TrialsStageRequest(num_training=5, num_testing=3))


# -- num_training derives from the apparatus, not a flat constant --------


def test_num_training_derives_from_the_apparatus_testing_count():
    """Regression: num_training used to be a flat 4 regardless of the design,
    so num_testing derived as trials_per_participant - 4 -- 26 for a design
    whose apparatus only ever declared 20 testing instances, more than existed
    to draw from. It must come from trials_per_participant minus the
    apparatus's own declared testing count instead."""
    from unittest.mock import MagicMock

    import numpy as np

    from server.pipeline import run_trials_stage
    from server.schemas import TrialsStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN",
        participants_per_condition=1, trials_per_participant=30,
        apparatus_instance_ids=list(range(1, 21)),  # 20 testing instances
        apparatus_training_instance_ids=[],
    )
    study.data.split.train_instance_ids = np.arange(100, 200)
    study.data.split.test_instance_ids = np.arange(1, 21)
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest())
    except Exception:
        pass

    kwargs = study.generate_trials.call_args.kwargs
    assert kwargs["num_training"] == 10  # 30 - 20
    assert kwargs["num_testing"] == 20


def test_num_training_falls_back_to_four_with_no_apparatus():
    from unittest.mock import MagicMock

    from server.pipeline import run_trials_stage
    from server.schemas import TrialsStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN",
        participants_per_condition=1, trials_per_participant=12,
    )
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest())
    except Exception:
        pass

    kwargs = study.generate_trials.call_args.kwargs
    assert kwargs["num_training"] == 4
    assert kwargs["num_testing"] == 8


def test_an_explicit_num_training_is_never_overridden():
    from unittest.mock import MagicMock

    from server.pipeline import run_trials_stage
    from server.schemas import TrialsStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="KNN",
        participants_per_condition=1, trials_per_participant=30,
        apparatus_instance_ids=list(range(1, 21)),
    )
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest(num_training=2))
    except Exception:
        pass

    assert study.generate_trials.call_args.kwargs["num_training"] == 2


# -- explanations stage is a no-op for CoXAM and Sim2Real -------------------


def test_explanations_stage_is_a_noop_for_sim2real():
    """Sim2Real always skips AI training (run_dataset_stage) and reads a fixed
    published corpus (Sim2RealAttributionProjector.from_assets), never
    study.combined_explanations -- so calling study.explanations() here would
    only raise "Call train_AI_model(...) before generating explanations."
    for no reason. Regression: this no-op used to cover CoXAM only."""
    study = _study("CoAX", xai_property=True)
    result = run_explanations_stage(study, ExplanationStageRequest())
    assert result["rows"] == 0
    assert "skipped_reason" in result


def test_explanations_stage_is_a_noop_for_coxam():
    study = _study("CoXAM")
    result = run_explanations_stage(study, ExplanationStageRequest())
    assert result["rows"] == 0
    assert "skipped_reason" in result


# -- CoAX also skips training for datasets its own published corpus covers --


def test_run_dataset_stage_skips_training_for_a_corpus_covered_coax_dataset():
    """adult/forest_cover/wine_quality: run_coax_study(source='corpus') reads
    its own AI predictions and explanation vectors, so training a fresh model
    here would be spent on something nothing downstream reads -- and was the
    root cause of a real bug: the freshly trained model's predictions,
    restricted to an apparatus's declared instance range, happened to
    collapse to one class, crashing balance_by_ai_prediction sampling."""
    from unittest.mock import MagicMock

    from server.pipeline import run_dataset_stage
    from server.schemas import DatasetStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX", dataset_id="wine_quality",
    )
    fake_data = MagicMock(
        dataset_id="wine_quality", feature_names=["a"], model_feature_names=["a"],
        y_train=[0, 1], y_test=[0],
    )
    study.prepare_dataset.return_value = fake_data

    result = run_dataset_stage(study, DatasetStageRequest())

    study.train_AI_model.assert_not_called()
    assert result["model"] is None
    assert "coax" in result["model_skipped_reason"].lower()


def test_run_dataset_stage_still_trains_coax_for_an_uncovered_dataset():
    from unittest.mock import MagicMock

    from server.pipeline import run_dataset_stage
    from server.schemas import DatasetStageRequest

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX", dataset_id="mushrooms",
    )
    fake_data = MagicMock(
        dataset_id="mushrooms", feature_names=["a"], model_feature_names=["a"],
        y_train=[0, 1], y_test=[0],
    )
    study.prepare_dataset.return_value = fake_data
    study.model_name = "mlp"
    study.test_accuracy.return_value = 0.9
    study.training_summary_table.return_value.to_dict.return_value = []
    study.training_history_table.return_value.to_dict.return_value = []
    study.metrics_table.return_value.reset_index.return_value.to_dict.return_value = []
    study.confusion_matrix_table.return_value.reset_index.return_value.to_dict.return_value = []

    run_dataset_stage(study, DatasetStageRequest())
    study.train_AI_model.assert_called_once()


def test_explanations_stage_is_a_noop_for_coax_when_training_was_skipped():
    from unittest.mock import MagicMock

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX",
    )
    study.trained_ai_model = None
    result = run_explanations_stage(study, ExplanationStageRequest())
    assert result["rows"] == 0
    assert "skipped_reason" in result


def test_explanations_stage_still_runs_for_coax_when_a_model_was_trained():
    from unittest.mock import MagicMock

    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX",
    )
    study.trained_ai_model = object()
    pool = pd.DataFrame({"expMethod": ["lime", "lime"]})
    study.explanations.return_value = ("path.csv", pool)

    result = run_explanations_stage(study, ExplanationStageRequest())

    study.explanations.assert_called_once()
    assert result["rows"] == 2
