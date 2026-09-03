"""``/simulate`` with ``mode="fitted_population"``, and the paper-comparison payload.

Mocks the study the way ``test_server_diverse_participant`` does: what is under
test is the pipeline's routing and the comparison payload's shape, not the
simulation itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from server.pipeline import (
    coax_paper_comparison_payload,
    coax_paper_reference,
    run_simulation_stage,
)
from server.schemas import SimulationRequest
from src.experiment_planner.design_export import DesignExport


def _design(framework="CoAX", baselines=None):
    return DesignExport(
        raw={},
        study_title="",
        research_questions=[],
        consent_text="",
        procedure_steps=[],
        ivs=[],
        model_framework=framework,
        ml_proxy_baselines=list(baselines or []),
    )


def _study(design, results=None):
    study = MagicMock()
    study.design_export = design
    study.trained_ai_model = object()
    study.trials = pd.DataFrame(
        {
            "participantId": [1],
            "instanceId": [0],
            "xai_type": ["importance"],
            "tested_w_xai": [True],
            "phase": ["testing"],
        }
    )
    study.save_results.return_value = ("x.csv", "x.json")
    study.run_experiment.return_value = pd.DataFrame(
        {"phase": ["testing"], "step": [0], "participantId": [1], "forward_accuracy": [1]}
    )
    study.simulated_results = results
    study.participant_parameters = None
    study.participant_parameters_path = None
    return study


# -- routing ---------------------------------------------------------------


def test_fitted_population_forwards_the_sampling_options():
    study = _study(_design())
    request = SimulationRequest(mode="fitted_population", sampling_seed=3)
    run_simulation_stage(study, request, output_subdir="x")

    kwargs = study.run_experiment.call_args.kwargs
    assert kwargs["mode"] == "fitted_population"
    assert kwargs["sampling_seed"] == 3
    assert kwargs["sampling_replace"] is None


def test_fitted_population_builds_models_keyed_by_tested_half():
    """Per-half strategies need per-half models; every other mode must not get them."""
    study = _study(_design())
    run_simulation_stage(
        study, SimulationRequest(mode="fitted_population"), output_subdir="x"
    )
    models = study.run_experiment.call_args.kwargs["cognitive_model"]
    assert all(isinstance(key, tuple) for key in models)

    study = _study(_design())
    run_simulation_stage(
        study, SimulationRequest(mode="diverse_participant"), output_subdir="x"
    )
    models = study.run_experiment.call_args.kwargs["cognitive_model"]
    assert all(not isinstance(key, tuple) for key in models)


def test_baselines_are_downgraded_to_whole_experiment():
    """A proxy baseline has no fitted population to draw from."""
    study = _study(_design(baselines=["Decision Tree"]))
    result = run_simulation_stage(
        study, SimulationRequest(mode="fitted_population"), output_subdir="x"
    )
    assert result["baseline_mode"] == "whole_experiment"


# -- the published reference ----------------------------------------------


def test_the_paper_reference_carries_the_five_published_cells():
    reference = coax_paper_reference()
    if reference is None:
        pytest.skip("assets/coax_paper_reference.json has not been built")

    pooled = {
        (row["xai_type"], row["tested_w_xai"])
        for row in reference["pooled"]
        if row["agent"] == "coax"
    }
    assert pooled == {
        ("none", False),
        ("importance", False),
        ("importance", True),
        ("attribution", False),
        ("attribution", True),
    }
    # The `none` arm has no "tested with XAI" half -- the published rows for it
    # are N=0 placeholders and must not survive into the reference.
    assert ("none", True) not in pooled
    # mushrooms was never fitted for CoAX, so the paper reports nothing for it.
    assert "mushrooms" not in reference["datasets"]


# -- the comparison payload ------------------------------------------------


def _simulated(with_provenance=True):
    """Two participants per cell, one scored step per trial."""
    rows = []
    for xai_type, tested in (("importance", False), ("importance", True)):
        for participant in (1, 2):
            for trial in range(4):
                rows.append(
                    {
                        "participantId": participant + (0 if not tested else 10),
                        "phase": "testing",
                        "step": "infer_with_explanation"
                        if tested
                        else "infer_no_explanation",
                        "dataId": "wine_quality",
                        "xai_type": xai_type,
                        "tested_w_xai": tested,
                        "cognitive_correct_vs_ai": trial % 2 == 0,
                        **(
                            {"fitted_participant_id": 100 + participant, "human_pai": 0.7}
                            if with_provenance
                            else {}
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_the_comparison_payload_matches_the_existing_panel_shape():
    if coax_paper_reference() is None:
        pytest.skip("assets/coax_paper_reference.json has not been built")

    payload = coax_paper_comparison_payload(_study(_design(), results=_simulated()))

    assert payload["sampled_participants"] is True
    assert payload["panels"], "expected at least the pooled panel"
    panel = payload["panels"][0]
    for key in ("title", "dv", "categories", "series", "interval", "note"):
        assert key in panel
    names = {series["name"] for series in panel["series"]}
    assert {"Paper - human", "Paper - CoAX", "Simulated agents", "Drawn humans"} <= names
    for series in panel["series"]:
        assert len(series["values"]) == len(panel["categories"])
        assert len(series["error"]) == len(panel["categories"])


def test_a_run_without_sampling_carries_no_drawn_human_series():
    if coax_paper_reference() is None:
        pytest.skip("assets/coax_paper_reference.json has not been built")

    payload = coax_paper_comparison_payload(
        _study(_design(), results=_simulated(with_provenance=False))
    )
    assert payload["sampled_participants"] is False
    names = {s["name"] for panel in payload["panels"] for s in panel["series"]}
    assert "Drawn humans" not in names


def test_double_counted_steps_are_dropped():
    """A trial can emit two rows; only the one the participant answered counts."""
    if coax_paper_reference() is None:
        pytest.skip("assets/coax_paper_reference.json has not been built")

    scored = _simulated()
    # The pre-explanation guess of a w/ XAI trial: present in the results, but
    # not what the participant answered under.
    extra = scored[scored["tested_w_xai"]].copy()
    extra["step"] = "infer_no_explanation"
    extra["cognitive_correct_vs_ai"] = False

    from server.pipeline import _scored_coax_rows

    assert len(_scored_coax_rows(pd.concat([scored, extra]))) == len(scored)


def test_pooling_the_paper_over_all_datasets_matches_its_published_table():
    """The n-weighted pool is exact, so a full-coverage run is compared against
    the paper's own pooled numbers rather than an approximation of them."""
    from server.pipeline import _paper_series

    reference = coax_paper_reference()
    if reference is None:
        pytest.skip("assets/coax_paper_reference.json has not been built")

    published = _paper_series(reference, agent="coax")
    weighted = _paper_series(reference, agent="coax", data_ids=reference["datasets"])
    assert set(published) == set(weighted)
    for cell, values in published.items():
        assert values["mean"] == pytest.approx(weighted[cell]["mean"], abs=1e-9)


def test_a_single_dataset_run_is_not_compared_against_the_three_dataset_pool():
    """Otherwise a wine-only run is scored against an average including two
    datasets it never touched."""
    if coax_paper_reference() is None:
        pytest.skip("assets/coax_paper_reference.json has not been built")

    payload = coax_paper_comparison_payload(_study(_design(), results=_simulated()))
    pooled = next(p for p in payload["panels"] if p.get("role") == "summary")
    wine = next(p for p in payload["panels"] if p["title"] == "Wine quality")

    def paper_coax(panel):
        return next(s for s in panel["series"] if s["name"] == "Paper - CoAX")["values"]

    assert paper_coax(pooled) == paper_coax(wine)
    assert "wine_quality" in pooled["note"]


def test_diverse_participant_promises_no_drawn_human_bar():
    """It samples people but records no per-participant accuracy, so the payload
    must not claim a bar the panel does not carry."""
    if coax_paper_reference() is None:
        pytest.skip("assets/coax_paper_reference.json has not been built")

    rows = _simulated()
    rows = rows.drop(columns=["human_pai"])  # what diverse_participant produces
    payload = coax_paper_comparison_payload(_study(_design(), results=rows))

    assert payload["sampled_participants"] is True
    assert payload["has_drawn_human_accuracy"] is False
    names = {s["name"] for panel in payload["panels"] for s in panel["series"]}
    assert "Drawn humans" not in names
    assert "no drawn-human bar" in payload["panels"][0]["note"]


def test_a_non_coax_study_is_refused():
    study = _study(_design(framework="Sim2Real"), results=_simulated())
    with pytest.raises(ValueError, match="only applies to a CoAX study"):
        coax_paper_comparison_payload(study)


def test_a_study_with_no_results_is_refused():
    study = _study(_design(), results=None)
    with pytest.raises(ValueError, match="no simulated results"):
        coax_paper_comparison_payload(study)
