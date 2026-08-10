import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SIM2REAL_DIR = ROOT / "assets" / "ai_dataset" / "sim2real"
EXPLANATIONS_DIR = ROOT / "assets" / "explanations" / "xai_desiderata"
BUILD_CORPUS = SIM2REAL_DIR / "build_corpus.py"


@pytest.fixture(scope="module")
def corpus_builder():
    spec = importlib.util.spec_from_file_location(
        "sim2real_build_corpus_for_test", BUILD_CORPUS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cases(corpus_builder):
    loaded = []
    for split, filename in (("training", "raw_training"), ("test", "raw_testing")):
        for name, case in corpus_builder.load_cases(SIM2REAL_DIR / filename):
            loaded.append((split, name, case))
    return loaded


def test_load_cases_honours_declared_case_arrays(corpus_builder):
    training = corpus_builder.load_cases(SIM2REAL_DIR / "raw_training")
    testing = corpus_builder.load_cases(SIM2REAL_DIR / "raw_testing")

    assert [name for name, _ in training] == [f"case_{i}" for i in range(10)]
    assert [name for name, _ in testing] == [f"case_{i}" for i in range(29)]


def test_every_effective_delta_maps_values_to_counterfactual(corpus_builder, cases):
    for _, _, case in cases:
        resolved = corpus_builder.resolve_delta(case)
        original = corpus_builder.encode(case["observations"])
        changed = resolved["counterfactual_encoded"]
        actual_changed_dimensions = [
            i for i, (before, after) in enumerate(zip(original, changed))
            if before != after
        ]

        assert actual_changed_dimensions == resolved["changed_dimensions"]
        assert original[resolved["observed_dim_from"]] == pytest.approx(
            resolved["observed_value_from"]
            if resolved["feature_kind"] == "numeric"
            else 1
        )
        assert changed[resolved["dim_to"]] == pytest.approx(
            resolved["value_to"]
            if resolved["feature_kind"] == "numeric"
            else 1
        )


def test_raw_source_contradictions_are_preserved_as_provenance(corpus_builder, cases):
    resolved = [corpus_builder.resolve_delta(case) for _, _, case in cases]

    assert [i for i, item in enumerate(resolved) if not item["source_matches"]] == [
        2,
        8,
        9,
    ]
    assert [i for i, item in enumerate(resolved) if not item["has_change"]] == []
    assert [i for i, item in enumerate(resolved) if item["direction_reversed"]] == []

    # The corrected raw observation and declared delta now agree exactly.
    item = resolved[1]
    assert (item["value_from"], item["value_to"]) == ("Female", "Male")
    assert (item["dim_from"], item["dim_to"]) == (56, 57)
    assert (item["observed_value_from"], item["observed_dim_from"]) == (
        "Female",
        56,
    )
    assert (item["source_value_from"], item["source_value_to"]) == (
        "Female",
        "Male",
    )

    # Instance 9's suggested transition starts from what the participant can
    # actually see (Assoc-voc), not from the declaration's 9th: labelling a
    # change "9th -> 1st-4th" beside a case reading Assoc-voc would ask them to
    # reason from a value that is not on screen. The declaration survives
    # verbatim in source_value_from for audit.
    item = resolved[9]
    assert (item["value_from"], item["value_to"]) == ("Assoc-voc", "1st-4th")
    assert (item["dim_from"], item["dim_to"]) == (12, 7)
    assert (item["observed_value_from"], item["observed_dim_from"]) == (
        "Assoc-voc",
        12,
    )
    assert (item["source_value_from"], item["source_dim_from"]) == ("9th", 10)


def test_generated_delta_csv_uses_effective_mapping_without_changing_cf(
    corpus_builder, tmp_path
):
    dataset_dir = tmp_path / "dataset"
    explanations_dir = tmp_path / "explanations"
    corpus_builder.main(dataset_dir, explanations_dir)

    # Original values remain unchanged; generated derived files must match the
    # checked-in corrected counterfactual and delta mappings.
    assert (dataset_dir / "values.csv").read_bytes() == (
        SIM2REAL_DIR / "values.csv"
    ).read_bytes()
    assert (explanations_dir / "counterfactuals_fake.csv").read_bytes() == (
        EXPLANATIONS_DIR / "counterfactuals_fake.csv"
    ).read_bytes()
    assert (explanations_dir / "deltas.csv").read_bytes() == (
        EXPLANATIONS_DIR / "deltas.csv"
    ).read_bytes()

    with (explanations_dir / "deltas.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 39
    assert rows[1]["valueFrom"] == "Female"
    assert rows[1]["valueTo"] == "Male"
    assert rows[1]["dimFrom"] == "56"
    assert rows[1]["dimTo"] == "57"
    assert rows[1]["sourceValueFrom"] == "Female"
    assert rows[1]["sourceValueTo"] == "Male"
    assert rows[1]["observedValueFrom"] == "Female"
    assert rows[1]["observedDimFrom"] == "56"
    assert rows[1]["valueFromMatchesInstance"] == "1"
    assert rows[1]["sourceValueFromMatchesInstance"] == "1"
    assert rows[1]["counterfactualHasChange"] == "1"
    assert rows[1]["deltaDirectionReversed"] == "0"
    assert rows[1]["changedDimensions"] == "56;57"
    assert rows[1]["mappingStatus"] == "matched"
    assert rows[1]["originalPrediction"] == "0"
    assert rows[1]["answer"] == "1"

    # Instances 2, 8 and 9 are the three whose raw declaration names a source
    # the case does not hold. The label follows the observation so the
    # participant is never asked to reason from a value that is not on screen;
    # the declaration is kept in sourceValueFrom, and valueFromMatchesInstance
    # is 1 because valueFrom now does match the instance.
    assert rows[2]["valueFrom"] == "0.0"
    assert rows[2]["valueTo"] == "1205.0"
    assert rows[2]["observedValueFrom"] == "0.0"
    assert rows[2]["sourceValueFrom"] == "500"
    assert rows[2]["valueFromMatchesInstance"] == "1"
    assert rows[2]["sourceValueFromMatchesInstance"] == "0"

    assert rows[8]["valueFrom"] == "46.0"
    assert rows[8]["valueTo"] == "20.0"
    assert rows[8]["observedValueFrom"] == "46"
    assert rows[8]["sourceValueFrom"] == "50"

    assert rows[9]["valueFrom"] == "Assoc-voc"
    assert rows[9]["dimFrom"] == "12"
    assert rows[9]["observedValueFrom"] == "Assoc-voc"
    assert rows[9]["observedDimFrom"] == "12"
    assert rows[9]["sourceValueFrom"] == "9th"
    assert rows[9]["sourceDimFrom"] == "10"
    assert rows[9]["changedDimensions"] == "7;12"


def test_every_instance_level_output_preserves_raw_qid(corpus_builder, tmp_path):
    dataset_dir = tmp_path / "dataset"
    explanations_dir = tmp_path / "explanations"
    corpus_builder.main(dataset_dir, explanations_dir)

    expected = {}
    for filename in ("raw_training", "raw_testing"):
        for _, case in corpus_builder.load_cases(SIM2REAL_DIR / filename):
            expected[len(expected)] = int(case["qid"])

    instance_files = [
        dataset_dir / "values.csv",
        dataset_dir / "none.csv",
        explanations_dir / "values.csv",
        explanations_dir / "none.csv",
        explanations_dir / "attribution.csv",
        explanations_dir / "importance.csv",
        explanations_dir / "counterfactuals_fake.csv",
        explanations_dir / "deltas.csv",
    ]
    for path in instance_files:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert "qid" in rows[0], path
        assert all(int(row["qid"]) == expected[int(row["instanceId"])] for row in rows)

    # metadata.csv is one app-level schema row rather than an instance table.
    with (dataset_dir / "metadata.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert "qid" not in next(csv.reader(handle))
