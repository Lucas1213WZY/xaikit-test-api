"""Parsing the UI's levels field.

Pinned against the two real exports in ``tutorials/experiment_output``, which
are what the apparatus actually writes: a pipe-separated string whose levels are
already slugged, e.g. ``"faithful | sparse | robust | sparse_robust"``. A
combined level therefore arrives whole -- there is no separator inside it to
split on by accident.
"""

import json
from pathlib import Path

import pytest

from src.experiment_planner.design_export import (
    check_against_support_matrix,
    parse_design_export,
    slugify,
    split_levels,
)

EXPORT_DIR = Path(__file__).resolve().parents[1] / "tutorials" / "experiment_output"


@pytest.fixture(scope="module")
def exports():
    paths = sorted(EXPORT_DIR.glob("experiment-design_*.json"))
    assert paths, f"no UI exports found in {EXPORT_DIR}"
    return {path.stem.split("_", 1)[1]: json.loads(path.read_text()) for path in paths}


# -- the format the UI actually writes -------------------------------------


def test_the_ui_writes_pipe_separated_pre_slugged_levels(exports):
    """The premise the rest of this module rests on.

    If the UI ever starts writing commas or slashes, a level whose own name
    contains one (``sparse_robust`` spelled ``"Sparse, Robust"``) would split
    into two and the combined condition would vanish silently. This test is the
    tripwire for that.
    """
    for name, raw in exports.items():
        for row in raw.get("ivs", []):
            field = str(row.get("levelsOrRange", ""))
            assert "," not in field and "/" not in field, (name, field)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("faithful | sparse | robust | sparse_robust",
         ["faithful", "sparse", "robust", "sparse_robust"]),
        ("Decision Tree | Logistic Regression | Hybrid",
         ["decision_tree", "logistic_regression", "hybrid"]),
    ],
)
def test_the_real_export_strings_parse(text, expected):
    assert split_levels(text, name="xai_property") == expected


def test_a_combined_level_survives_as_one_level():
    """sparse_robust is one condition, not sparse plus robust."""
    levels = split_levels("faithful | sparse | robust | sparse_robust", name="xai_property")
    assert "sparse_robust" in levels
    assert len(levels) == 4


# -- separators and slugging -----------------------------------------------


@pytest.mark.parametrize("text", ["A | B", "A vs B", "A versus B", "A, B", "A / B"])
def test_every_declared_separator_splits(text):
    assert split_levels(text) == ["a", "b"]


def test_parentheticals_are_dropped():
    assert split_levels("Input Gradients (paper)", name="xai_method") == ["input_gradients"]


def test_blank_parts_are_ignored():
    assert split_levels("Sparse |  | Robust", name="xai_property") == ["sparse", "robust"]


def test_list_input_is_not_reparsed():
    assert split_levels(["Sparse", "Robust"], name="xai_property") == ["sparse", "robust"]


def test_boolean_ivs_are_coerced():
    assert split_levels("With XAI vs Without XAI", name="tested_w_xai") == [True, False]


def test_slugify_matches_the_declared_vocabulary():
    assert slugify("Sparse Robust") == "sparse_robust"
    assert slugify("Logistic Regression") == "logistic_regression"


# -- the whole export round-trips ------------------------------------------


def test_both_real_exports_parse(exports):
    for name, raw in exports.items():
        design = parse_design_export(raw)
        assert design.ivs, name
        assert design.resolved_framework, name


def test_the_sim2real_export_keeps_all_four_properties(exports):
    design = parse_design_export(exports["sim2real"])
    levels = {iv["name"]: iv["levels"] for iv in design.ivs}
    assert levels["xai_property"] == ["faithful", "sparse", "robust", "sparse_robust"]
    assert design.resolved_framework == "sim2real"


def test_the_coxam_export_keeps_all_three_xai_types(exports):
    design = parse_design_export(exports["coxam"])
    levels = {iv["name"]: iv["levels"] for iv in design.ivs}
    assert levels["xai_type"] == ["decision_tree", "logistic_regression", "hybrid"]
    assert design.resolved_framework == "coxam"


def test_no_export_carries_an_undeclared_level(exports):
    """Levels are checked against support_matrix.json, not just parsed."""
    for name, raw in exports.items():
        design = parse_design_export(raw)
        report = check_against_support_matrix(design)
        level_warnings = [
            issue
            for issue in report.warnings
            if "does not declare" in issue.message and issue.field.startswith("iv.")
        ]
        assert not level_warnings, (name, [issue.message for issue in level_warnings])
