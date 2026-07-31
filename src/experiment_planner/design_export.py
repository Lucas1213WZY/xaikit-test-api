"""Translate an experiment-design UI export into an xaikitTest study design.

The design UI exports a questionnaire-shaped JSON (``researchQuestions``,
``studyDesign``, ``apparatus``, ``procedure``, ``userModel``, ``_rawAnswers``)
with prose values: levels arrive as one pipe-delimited string and allocation as
"Between-subjects". This module normalizes that into the names and shapes the
study API expects, so a notebook can drive the whole design from the export
instead of retyping it as add_iv/add_cv/add_dv calls.

This is a different schema from the trial-generation config consumed by
``generate_trials_from_ui_config``; the two are not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Optional, Sequence

from .support import ValidationReport, print_validation_report


# DV measures whose UI name would otherwise lose meaning downstream. The executor
# only applies accuracy scoring to DVs whose name contains "accuracy", so a
# measure exported as "forward_sim" has to be renamed to stay scoreable.
DV_MEASURE_ALIASES: dict[str, str] = {
    "forward_sim": "forward_accuracy",
    "forward_simulation": "forward_accuracy",
    "counterfactual_sim": "counterfactual_accuracy",
    "counterfactual_simulation": "counterfactual_accuracy",
}

# The only DVs the virtual-participant executor can produce. A measure the UI
# collected that is not one of these (trust, decision_time, workload, ...) has no
# simulated counterpart. Those are kept under the name the UI gave them and
# reported as warnings rather than rewritten, so the design is never silently
# changed into something it did not say.
SUPPORTED_DVS: tuple[str, ...] = ("forward_accuracy", "counterfactual_accuracy")

# IV factor labels the UI phrases freely but the planner knows under a fixed name.
IV_FACTOR_ALIASES: dict[str, str] = {
    "tested_with_xai": "tested_w_xai",
    "tested_w_xai": "tested_w_xai",
    "xai_shown": "tested_w_xai",
    "xai_visible": "tested_w_xai",
    "xai_method": "xai_method",
    "explanation_method": "xai_method",
    "xai_type": "xai_type",
    "explanation_type": "xai_type",
}

# IVs the planner requires boolean levels for, with the phrasings the UI produces.
BOOLEAN_IVS: set[str] = {"tested_w_xai"}
BOOLEAN_LEVEL_ALIASES: dict[str, bool] = {
    "with_xai": True,
    "w_xai": True,
    "with": True,
    "yes": True,
    "true": True,
    "shown": True,
    "without_xai": False,
    "wo_xai": False,
    "no_xai": False,
    "without": False,
    "no": False,
    "false": False,
    "hidden": False,
}

# The UI lets people annotate a level, e.g. "Input Gradients (paper)". The
# annotation is commentary, not part of the level name.
PARENTHETICAL = re.compile(r"\s*\([^)]*\)")

ALLOCATION_ALIASES: dict[str, str] = {
    "between-subjects": "between",
    "between subjects": "between",
    "between": "between",
    "within-subjects": "within",
    "within subjects": "within",
    "within": "within",
}

# The UI records counterbalancing as prose. Only within-subject IVs use it, and it
# maps onto the randomization unit the trial builder expects.
RANDOMIZATION_PATTERNS: list[tuple[str, str]] = [
    (r"random", "trial"),
    (r"latin|counterbalanc|block", "block"),
]

# Level strings arrive as one field. The UI lets people separate levels with a
# pipe, a comma, a slash, or the word "vs".
LEVEL_SEPARATOR = re.compile(r"\s*(?:\||,|/|\bvs\.?\b|\bversus\b)\s*", re.IGNORECASE)

# Procedure step kinds the study protocol understands, matched against the step
# title the UI collected.
STEP_KIND_PATTERNS: list[tuple[str, str]] = [
    (r"consent", "consent"),
    (r"debrief", "debrief"),
    (r"survey|questionnaire|demograph", "survey"),
    (r"train|practice", "practice"),
    (r"test|trial|session", "trials"),
]


def slugify(value: Any) -> str:
    """Turn a UI label such as 'XAI Type' or 'Decision Tree' into a code name."""
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    return text.strip("_")


def split_levels(value: Any, *, name: str = "") -> list[Any]:
    """Split a levels field such as 'A | B | C' or 'With XAI vs Without XAI'.

    Parenthetical annotations are dropped, so 'Input Gradients (paper)' is the
    level ``input_gradients``. IVs the planner expects booleans for are coerced
    from the UI's wording to ``True``/``False``.
    """
    if isinstance(value, (list, tuple)):
        parts = [str(part) for part in value]
    else:
        parts = LEVEL_SEPARATOR.split(str(value))

    levels = [slugify(PARENTHETICAL.sub("", part)) for part in parts if str(part).strip()]
    levels = [level for level in levels if level]

    if name in BOOLEAN_IVS:
        return [BOOLEAN_LEVEL_ALIASES.get(level, level) for level in levels]
    return levels


def _randomization_for(counterbalancing: Any, name: str = "") -> str:
    # The planner only supports tested_w_xai as a trial-level within IV, whatever
    # counterbalancing prose the UI collected for it.
    if name in BOOLEAN_IVS:
        return "trial"
    text = str(counterbalancing).strip().lower()
    for pattern, unit in RANDOMIZATION_PATTERNS:
        if re.search(pattern, text):
            return unit
    return "block"


def _canonical_allocation(value: Any) -> str:
    key = str(value).strip().lower()
    if key not in ALLOCATION_ALIASES:
        raise ValueError(
            f"Unknown allocation {value!r}. Use Between-subjects or Within-subjects."
        )
    return ALLOCATION_ALIASES[key]


def _step_kind(title: str) -> str:
    lowered = str(title).lower()
    for pattern, kind in STEP_KIND_PATTERNS:
        if re.search(pattern, lowered):
            return kind
    return "instructions"


def _is_blank_row(row: dict[str, Any], keys: Sequence[str]) -> bool:
    """The UI emits empty placeholder rows for variables the user left unfilled."""
    return not any(str(row.get(key, "")).strip() for key in keys)


@dataclass
class DesignExport:
    """A normalized view of one experiment-design UI export."""

    raw: dict[str, Any]
    study_title: str
    research_questions: list[str]
    consent_text: str
    procedure_steps: list[dict[str, str]]
    ivs: list[dict[str, Any]] = field(default_factory=list)
    cvs: list[dict[str, Any]] = field(default_factory=list)
    dvs: list[dict[str, Any]] = field(default_factory=list)
    rvs: list[dict[str, Any]] = field(default_factory=list)
    dataset_id: str = ""
    model_framework: str = ""
    design_type: str = ""
    participants_per_condition: Optional[int] = None
    total_participants: Optional[int] = None
    trials_per_participant: Optional[int] = None
    total_trials: Optional[int] = None
    ml_proxy_baselines: list[Any] = field(default_factory=list)
    cognitive_config: dict[str, Any] = field(default_factory=dict)
    report: ValidationReport = field(default_factory=lambda: ValidationReport(stage="design_export"))

    @property
    def simulatable_dvs(self) -> list[str]:
        """DV names the virtual-participant executor can actually produce."""
        return [dv["name"] for dv in self.dvs if dv["name"] in SUPPORTED_DVS]

    def show_report(self) -> None:
        """Print the parse warnings in the same format as design validation."""
        print_validation_report(self.report)

    @property
    def between_ivs(self) -> list[dict[str, Any]]:
        return [iv for iv in self.ivs if iv["iv_type"] == "between"]

    @property
    def within_ivs(self) -> list[dict[str, Any]]:
        return [iv for iv in self.ivs if iv["iv_type"] == "within"]

    def summary_rows(self) -> list[dict[str, Any]]:
        """Flat table of every variable, for previewing what the UI actually said."""
        rows: list[dict[str, Any]] = []
        for iv in self.ivs:
            rows.append({
                "role": "IV",
                "name": iv["name"],
                "detail": iv["iv_type"],
                "levels": ", ".join(map(str, iv["levels"])),
                "source_label": iv["source_label"],
            })
        for cv in self.cvs:
            rows.append({
                "role": "CV",
                "name": cv["name"],
                "detail": cv["type"],
                "levels": ", ".join(map(str, cv["levels"])),
                "source_label": cv["source_label"],
            })
        for dv in self.dvs:
            rows.append({
                "role": "DV",
                "name": dv["name"],
                "detail": ", ".join(map(str, dv["levels"])),
                "levels": "",
                "source_label": dv["source_label"],
            })
        for rv in self.rvs:
            rows.append({
                "role": "RV",
                "name": rv["name"],
                "detail": ", ".join(map(str, rv["levels"])),
                "levels": "",
                "source_label": rv["source_label"],
            })
        return rows


def load_design_export(path: str | Path) -> DesignExport:
    """Read and normalize an experiment-design UI export."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Design export not found: {path}")
    return parse_design_export(json.loads(path.read_text()))


def parse_design_export(raw: dict[str, Any]) -> DesignExport:
    """Normalize an already-loaded experiment-design UI export."""
    study_design = raw.get("studyDesign", {})
    report = ValidationReport(stage="design_export")

    ivs: list[dict[str, Any]] = []
    for row in study_design.get("independentVariables", []):
        if _is_blank_row(row, ["factor", "levelsOrRange"]):
            continue
        name = slugify(row["factor"])
        name = IV_FACTOR_ALIASES.get(name, name)
        levels = split_levels(row.get("levelsOrRange", ""), name=name)
        if not levels:
            continue
        iv_type = _canonical_allocation(row.get("allocation", "between"))
        ivs.append({
            "name": name,
            "iv_type": iv_type,
            "levels": levels,
            # Only within-subject IVs carry a randomization unit. The UI records
            # counterbalancing prose instead, so block is the safe default.
            "randomization": _randomization_for(row.get("counterbalancing"), name) if iv_type == "within" else None,
            "counterbalancing": str(row.get("counterbalancing", "")).strip(),
            "source_label": row["factor"],
        })

    cvs: list[dict[str, Any]] = []
    for row in study_design.get("controlVariablesCV", []):
        if _is_blank_row(row, ["name"]):
            continue
        cvs.append({
            "name": slugify(row["name"]),
            "type": str(row.get("type", "")).strip(),
            # The UI records a CV's measurement type, not its levels; the single
            # level keeps it constant, which is what a control variable means here.
            "levels": [slugify(row.get("type") or "constant")],
            "source_label": row["name"],
        })

    dvs: list[dict[str, Any]] = []
    for row in study_design.get("dependentVariables", []):
        if _is_blank_row(row, ["measure", "name"]):
            continue
        measure = str(row.get("name") or row.get("measure")).strip()
        slug = slugify(measure)
        name = DV_MEASURE_ALIASES.get(slug, slug)
        if name not in SUPPORTED_DVS:
            report.add_warning(
                name,
                f"`{measure}` has no simulated counterpart, so the virtual "
                "participant cannot produce it.",
                f"Collect it from human participants, or analyse one of: "
                f"{', '.join(sorted(SUPPORTED_DVS))}.",
            )
        dvs.append({
            "name": name,
            "levels": ["continuous"],
            "source_label": measure,
            "simulatable": name in SUPPORTED_DVS,
        })

    rvs: list[dict[str, Any]] = []
    for row in study_design.get("randomVariablesRV", []):
        if _is_blank_row(row, ["name"]):
            continue
        rvs.append({
            "name": slugify(row["name"]),
            "levels": [slugify(row.get("type") or "categorical")],
            "source_label": row["name"],
        })

    procedure_steps = []
    consent_text = ""
    for step in raw.get("procedure", []):
        title = str(step.get("title", "")).strip()
        if not title:
            continue
        kind = _step_kind(title)
        procedure_steps.append({"title": title, "kind": kind})
        note = str(step.get("note", "")).strip()
        if kind == "consent" and note and not consent_text:
            consent_text = note

    research_questions = raw.get("researchQuestions", "")
    if isinstance(research_questions, str):
        research_questions = [q.strip() for q in research_questions.split("\n") if q.strip()]

    # `apparatus` is deliberately ignored: it configures the participant-facing UI,
    # which this API never renders.
    dataset_label = study_design.get("dataset", "")

    return DesignExport(
        raw=raw,
        study_title=(
            _extract_study_title(consent_text)
            or str(dataset_label).strip()
            or "study"
        ),
        research_questions=research_questions,
        consent_text=consent_text,
        procedure_steps=procedure_steps,
        ivs=ivs,
        cvs=cvs,
        dvs=dvs,
        rvs=rvs,
        dataset_id=slugify(dataset_label),
        model_framework=str(raw.get("userModel") or study_design.get("modelFramework", "")).strip(),
        design_type=str(study_design.get("design", "")).strip(),
        participants_per_condition=study_design.get("participantsPerCondition"),
        total_participants=study_design.get("totalParticipants"),
        trials_per_participant=study_design.get("trialsPerParticipant"),
        total_trials=study_design.get("totalTrials"),
        ml_proxy_baselines=list(raw.get("mlProxyBaselines") or []),
        cognitive_config=dict(raw.get("cognitiveConfig") or {}),
        report=report,
    )


def _extract_study_title(consent_text: str) -> str:
    match = re.search(r"study title\s*:\s*(.+)", consent_text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def apply_design_export(
    study: Any,
    design: DesignExport,
    *,
    dv_names: Optional[Sequence[str]] = None,
    include_rvs: bool = False,
    show: bool = False,
    report: bool = True,
) -> Any:
    """Register a parsed export's variables and protocol on an xaikitTest study.

    ``dv_names`` restricts which DVs are registered, for the common case where the
    export lists more measures than one run analyses. Measures with no simulated
    counterpart are registered under the name the UI gave them and reported as
    warnings; nothing is renamed on your behalf.
    """
    for iv in design.ivs:
        if iv["iv_type"] == "within":
            study.add_iv(iv["name"], "within", iv["levels"], randomization=iv["randomization"], show=show)
        else:
            study.add_iv(iv["name"], "between", iv["levels"], show=show)

    for cv in design.cvs:
        study.add_cv(cv["name"], cv["levels"], show=show)

    wanted = set(dv_names) if dv_names is not None else None
    for dv in design.dvs:
        if wanted is None or dv["name"] in wanted:
            study.add_dv(dv["name"], dv["levels"], show=show)

    if report:
        design.show_report()

    if include_rvs and hasattr(study, "add_rv"):
        for rv in design.rvs:
            study.add_rv(rv["name"], rv["levels"], show=show)

    study.set_study_protocol(
        study_title=design.study_title,
        research_questions=design.research_questions,
        consent_text=design.consent_text,
        procedure_steps=design.procedure_steps,
    )
    return study


__all__ = [
    "ALLOCATION_ALIASES",
    "BOOLEAN_IVS",
    "BOOLEAN_LEVEL_ALIASES",
    "DV_MEASURE_ALIASES",
    "IV_FACTOR_ALIASES",
    "RANDOMIZATION_PATTERNS",
    "DesignExport",
    "apply_design_export",
    "load_design_export",
    "parse_design_export",
    "slugify",
    "split_levels",
]
