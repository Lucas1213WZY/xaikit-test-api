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
    # sim2real varies explanation *properties*, which are a different factor
    # from xai_type -- its xai_type is "attribution" throughout. The UI exports
    # this as "XAI Property"; the fitter calls it exp_property and
    # experiment_planner.preview calls it explanation_property.
    "xai_property": "xai_property",
    "xai_properties": "xai_property",
    "explanation_property": "xai_property",
    "exp_property": "xai_property",
    # Which simulation the agent runs. A declared IV in its own right
    # (support_matrix ivs.user_task) and a semantic CV.
    "user_task": "user_task",
    "task": "user_task",
    "participant_task": "user_task",
}

# The UI names datasets in prose. Only mismatches need an entry -- "Wine
# Quality" already slugifies onto the declared "wine_quality".
DATASET_ALIASES: dict[str, str] = {
    "adult_income": "adult",
    "census_income": "adult",
    "wine": "wine_quality",
    "mushroom": "mushrooms",
    "breast_cancer_wisconsin": "breast_cancer",
}

# ML-proxy baseline labels the UI offers, mapped onto the ids
# ``cognitive_models`` declares. "MLP" already normalizes to mlp_baseline via
# normalize_baseline_model_id; these are the ones that do not.
BASELINE_ALIASES: dict[str, str] = {
    "linear_regression": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "decision_tree": "decision_tree",
    "knn": "knn",
    "k_nearest_neighbours": "knn",
    "k_nearest_neighbors": "knn",
    "mlp": "mlp_baseline",
    "mlp_baseline": "mlp_baseline",
    "neural_network": "mlp_baseline",
}

#: ``userModel`` strings that name a framework directly, keyed by their slug.
#: Plain ``"CoAX"`` is deliberately absent: on its own it is genuinely CoAX,
#: and only the ``xai_property`` IV (see ``resolved_framework``) tells the two
#: apart. ``"CoAX (XAI Property)"`` carries the discriminator in the label
#: itself, so it resolves directly without needing that IV to parse correctly.
MODEL_FRAMEWORK_ALIASES: dict[str, str] = {
    "coax_xai_property": "sim2real",
}

#: The task a DV implies, when the design does not name one outright.
TASK_BY_DV: dict[str, str] = {
    "forward_accuracy": "forward_simulation",
    "counterfactual_accuracy": "counterfactual_simulation",
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

    The UI writes this field pipe-separated with the levels already slugged
    (``"faithful | sparse | robust | sparse_robust"``), so a combined level
    arrives whole and there is nothing to disambiguate. ``check_against_support_matrix``
    reports any level that is not declared.
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
    #: Testing/training instance ids the apparatus's own configurations
    #: declared, unioned across every entry. Empty when the export names none
    #: (``apparatus[i].params`` can be ``{}``, as a real export's third
    #: configuration does).
    apparatus_instance_ids: list[int] = field(default_factory=list)
    apparatus_training_instance_ids: list[int] = field(default_factory=list)
    report: ValidationReport = field(default_factory=lambda: ValidationReport(stage="design_export"))

    @property
    def simulatable_dvs(self) -> list[str]:
        """DV names the virtual-participant executor can actually produce."""
        return [dv["name"] for dv in self.dvs if dv["name"] in SUPPORTED_DVS]

    @property
    def resolved_framework(self) -> str:
        """The agent this design actually runs, as a declared model id.

        ``userModel`` alone is not enough. A Sim2Real design exports
        ``userModel: "CoAX"`` -- correctly, since its cognitive model is a
        CoAX-derived attribution sum -- but it runs the Sim2Real fitted model
        and supports counterfactual simulation, which plain CoAX does not.

        The ``xai_property`` IV is the discriminator: only Sim2Real varies
        explanation properties (faithful / sparse / robust / sparse_robust), so
        a design that declares it is a Sim2Real study whatever ``userModel``
        says.
        """
        if any(iv["name"] == "xai_property" for iv in self.ivs):
            return "sim2real"
        slug = slugify(self.model_framework)
        return MODEL_FRAMEWORK_ALIASES.get(slug, slug)

    @property
    def user_tasks(self) -> list[str]:
        """Which simulations this design asks for, in declared order.

        A *list*, not a scalar: a design can name both ``forward_sim`` and
        ``counterfactual_sim`` as DVs, and for CoXAM those are two separately
        trained agents, so they are two runs rather than one.

        An explicit ``user_task`` IV or CV wins over the DV names, since it says
        outright what the DV only implies.
        """
        for variable in (*self.ivs, *self.cvs):
            if variable["name"] == "user_task" and variable["levels"]:
                return [str(level) for level in variable["levels"]]

        tasks: list[str] = []
        for dv in self.dvs:
            task = TASK_BY_DV.get(dv["name"])
            if task and task not in tasks:
                tasks.append(task)
        return tasks

    @property
    def baseline_model_ids(self) -> list[str]:
        """``ml_proxy_baselines`` mapped onto declared cognitive-model ids.

        Labels with no declared baseline (e.g. "Global SHAP", an XAI method
        rather than a proxy model) are dropped here and reported as warnings by
        the parser, so they are visible rather than silently carried along.
        """
        resolved: list[str] = []
        for label in self.ml_proxy_baselines:
            model_id = BASELINE_ALIASES.get(slugify(label))
            if model_id and model_id not in resolved:
                resolved.append(model_id)
        return resolved

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

    # `apparatus[i].params.instanceIds` / `trainingInstanceIds` name exactly the
    # instances the human-study UI showed participants -- unioned across every
    # configuration, since a design can carry several (one per condition). A
    # trial referencing an instance outside this set was never seen by a human,
    # so simulation and explanation should be constrained to it, not sampled
    # freely; see apply_design_export's trial-generation defaults.
    apparatus_instance_ids = _apparatus_instance_ids(raw.get("apparatus"), "instanceIds")
    apparatus_training_instance_ids = _apparatus_instance_ids(
        raw.get("apparatus"), "trainingInstanceIds"
    )
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
        dataset_id=DATASET_ALIASES.get(slugify(dataset_label), slugify(dataset_label)),
        model_framework=str(raw.get("userModel") or study_design.get("modelFramework", "")).strip(),
        design_type=str(study_design.get("design", "")).strip(),
        participants_per_condition=study_design.get("participantsPerCondition"),
        total_participants=study_design.get("totalParticipants"),
        trials_per_participant=study_design.get("trialsPerParticipant"),
        total_trials=study_design.get("totalTrials"),
        ml_proxy_baselines=list(raw.get("mlProxyBaselines") or []),
        cognitive_config=_merged_cognitive_config(raw),
        apparatus_instance_ids=apparatus_instance_ids,
        apparatus_training_instance_ids=apparatus_training_instance_ids,
        report=report,
    )


def _parse_instance_id_ranges(value: Any) -> list[int]:
    """``"10-19"`` or ``"1, 3, 5-9"`` -> sorted unique ints.

    Both ends of a range are inclusive, matching how the apparatus UI names an
    instance span. Blank or unparseable text yields an empty list rather than
    raising, so one malformed configuration does not break the whole export.
    """
    if value is None:
        return []
    ids: set[int] = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:  # skip a leading '-' so a bare negative isn't split
            start, _, end = part.partition("-")
            try:
                start_i, end_i = int(start.strip()), int(end.strip())
            except ValueError:
                continue
            ids.update(range(min(start_i, end_i), max(start_i, end_i) + 1))
        else:
            try:
                ids.add(int(part))
            except ValueError:
                continue
    return sorted(ids)


def _apparatus_instance_ids(apparatus: Any, key: str) -> list[int]:
    """Union of ``params[key]`` across every apparatus configuration entry.

    An entry with no ``params`` (a real export's third configuration is
    exactly ``{}``) contributes nothing, which is why this is a union rather
    than requiring every entry to agree.
    """
    if not isinstance(apparatus, list):
        return []
    ids: set[int] = set()
    for entry in apparatus:
        if isinstance(entry, dict):
            ids.update(_parse_instance_id_ranges(entry.get("params", {}).get(key)))
    return sorted(ids)


def _merged_cognitive_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Both shapes a design export has used for cognitive parameters.

    ``cognitiveConfig`` is the older ``{"Display Label": "string value"}``
    map. ``cognitiveParameters`` is a newer, richer form -- a list of
    ``{key, label, min, max, value, source, ...}`` records, where ``key`` is
    already the agent's own parameter name (not a display label) and
    ``value`` is ``None`` for "use the model default." Both can be present at
    once; ``cognitiveParameters`` entries win on a shared key, since it is the
    more specific, already-resolved source.
    """
    merged = dict(raw.get("cognitiveConfig") or {})
    for entry in raw.get("cognitiveParameters") or []:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        value = entry.get("value")
        if key and value is not None:
            merged[key] = value
    return merged


def _extract_study_title(consent_text: str) -> str:
    match = re.search(r"study title\s*:\s*(.+)", consent_text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def check_against_support_matrix(design: DesignExport) -> ValidationReport:
    """Report where an export disagrees with what the toolkit declares.

    Without this an export parses cleanly no matter what it says -- an IV the
    planner has never heard of, a dataset under a name it does not use, or a
    baseline that is actually an XAI method all arrive with zero warnings and
    fail much later, or silently do the wrong thing.

    Warnings rather than errors: the UI is the researcher's own record of the
    study, and this module's contract is to normalize it, never to overrule it.
    """
    from .support import load_support_matrix

    report = ValidationReport(stage="design_export_support")
    matrix = load_support_matrix()
    declared_ivs = matrix.get("ivs", {})
    declared_models = matrix.get("cognitive_models", {})

    for iv in design.ivs:
        name = iv["name"]
        spec = declared_ivs.get(name)
        if spec is None:
            report.add_warning(
                f"iv.{name}",
                f"{iv['source_label']!r} became {name!r}, which the toolkit does not declare "
                f"as an IV.",
                f"Declared IVs: {sorted(declared_ivs)}. Add an alias to IV_FACTOR_ALIASES "
                "or declare it in support_matrix.json.",
            )
            continue
        levels = spec.get("levels")
        unknown = [level for level in iv["levels"] if levels and level not in levels]
        if unknown:
            report.add_warning(
                f"iv.{name}",
                f"{name} carries level(s) {unknown} the toolkit does not declare.",
                f"Declared levels: {levels}.",
            )

    dataset_levels = declared_ivs.get("dataset", {}).get("levels") or []
    if design.dataset_id and dataset_levels and design.dataset_id not in dataset_levels:
        report.add_warning(
            "dataset",
            f"dataset {design.dataset_id!r} is not one the toolkit can load.",
            f"Available: {dataset_levels}. Add an entry to DATASET_ALIASES if the UI "
            "just names it differently.",
        )

    for label in design.ml_proxy_baselines:
        if slugify(label) not in BASELINE_ALIASES:
            report.add_warning(
                "ml_proxy_baselines",
                f"{label!r} is not a declared baseline cognitive model, so it will be "
                "skipped.",
                f"Declared: {sorted(k for k in declared_models if k not in ('placeholder',))}.",
            )

    framework = design.resolved_framework
    model_spec = declared_models.get(framework)
    if framework and framework != slugify(design.model_framework):
        report.add_reminder(
            "model_framework",
            f"userModel says {design.model_framework!r}, but the xai_property IV makes "
            f"this a {framework} design; it will run the {framework} model.",
        )
    if design.model_framework and model_spec is None:
        report.add_warning(
            "model_framework",
            f"userModel {design.model_framework!r} is not a declared cognitive model.",
            f"Declared: {sorted(declared_models)}.",
        )
    elif model_spec is not None:
        supported = model_spec.get("tasks") or []
        for task in design.user_tasks:
            if supported and task not in supported:
                report.add_warning(
                    "user_task",
                    f"the design asks for {task!r}, which {framework} does not support.",
                    f"{framework} supports: {supported}.",
                )
    return report


def apply_design_export(
    study: Any,
    design: DesignExport,
    *,
    dv_names: Optional[Sequence[str]] = None,
    include_rvs: bool = False,
    show: bool = False,
    report: bool = True,
    set_cognitive_model: bool = True,
) -> Any:
    """Register a parsed export's variables, agent and protocol on a study.

    ``dv_names`` restricts which DVs are registered, for the common case where the
    export lists more measures than one run analyses. Measures with no simulated
    counterpart are registered under the name the UI gave them and reported as
    warnings; nothing is renamed on your behalf.

    ``set_cognitive_model`` selects the agent named by ``userModel`` and gives it
    its cognitive parameters -- from ``cognitiveConfig`` where the UI supplied
    them, from the agent's own defaults where it did not. Pass ``False`` to
    register only the variables and choose the agent yourself.

    The design can ask for more than one task (a CoXAM export commonly names
    both ``forward_sim`` and ``counterfactual_sim`` as DVs). Those are separate
    runs -- see ``design.user_tasks`` -- so this registers the agent once and
    leaves the task choice to each ``run_experiment`` call.
    """
    for iv in design.ivs:
        if iv["iv_type"] == "within":
            study.add_iv(iv["name"], "within", iv["levels"], randomization=iv["randomization"], show=show)
        else:
            study.add_iv(iv["name"], "between", iv["levels"], show=show)

    for cv in design.cvs:
        study.add_cv(cv["name"], cv["levels"], show=show)

    declared = {iv["name"] for iv in design.ivs} | {cv["name"] for cv in design.cvs}
    if design.resolved_framework in {"coax", "coxam"} and "tested_w_xai" not in declared:
        # Both agents' whole comparison is with-vs-without the explanation, so
        # a design that never says otherwise is assumed to want it varied per
        # trial -- not silently defaulted to one constant value. Without this,
        # a trial's tested_w_xai is absent from the row entirely; the CoXAM
        # executor's own fallback for a missing value is False, which is
        # already what "training" phase forces regardless, so every *testing*
        # trial would show no explanation, every time, with no alternation.
        study.add_iv("tested_w_xai", "within", [True, False], randomization="trial", show=show)
        design.report.add_reminder(
            "tested_w_xai",
            f"userModel={design.model_framework!r} needs tested_w_xai to compare with "
            "and without the explanation, and the export did not declare it.",
            suggestion="Added as a trial-randomized within IV with levels [True, False].",
        )

    wanted = set(dv_names) if dv_names is not None else None
    for dv in design.dvs:
        if wanted is None or dv["name"] in wanted:
            study.add_dv(dv["name"], dv["levels"], show=show)

    if include_rvs and hasattr(study, "add_rv"):
        for rv in design.rvs:
            study.add_rv(rv["name"], rv["levels"], show=show)

    if set_cognitive_model:
        _apply_cognitive_model(study, design)

    study.set_study_protocol(
        study_title=design.study_title,
        research_questions=design.research_questions,
        consent_text=design.consent_text,
        procedure_steps=design.procedure_steps,
    )

    if report:
        design.report.extend(check_against_support_matrix(design))
        design.show_report()
    return study


#: UI ``cognitiveConfig`` labels that slugify to something the agent does not
#: read. The UI names are prose, so ``"Diffusion Noise"`` becomes
#: ``diffusion_noise`` while CoXAM's parameter is ``decision_noise`` -- storing
#: it under the slug would leave the real parameter at its default and the run
#: would succeed with the value silently ignored.
COGNITIVE_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "coxam": {
        "retrieval_threshold": "memory_recall_threshold",
        "memory_retrieval_threshold": "memory_recall_threshold",
        "diffusion_noise": "decision_noise",
        "decision_diffusion_noise": "decision_noise",
        "counterfactual_margin": "counterfactual_overshoot_fraction",
        "overshoot_fraction": "counterfactual_overshoot_fraction",
        "random_response": "random_response_rate",
        "time_penalty": "time_penalty_weight",
    },
    "sim2real": {
        "memory_retrieval_threshold": "retrieval_threshold",
        "exemplar_memory": "use_exemplar_memory",
        "lapse": "lapse_rate",
        # The UI labels these for what they do rather than for the maths:
        # `confidence_intercept` is presented as responsiveness because it sets
        # where the evidence lands on the confidence curve -- more negative puts
        # it on the steep part, so the answer moves more when the case changes.
        "confidence_responsiveness": "confidence_intercept",
        "aggregation_strategy": "aggregation",
        "max_features": "max_features_attended",
        "features_attended": "max_features_attended",
    },
    "coax": {
        "retrieval_threshold": "cog_retrieval_threshold",
        "encoding_time": "cog_T_enc",
        "operation_time": "cog_T_op",
        "latency_factor": "cog_latency_factor",
    },
}


#: Parameters each agent accepts beyond what ``default_cognitive_params``
#: returns. That function gives operating defaults, not the accepted set --
#: sim2real's deliberately covers only its condition-agnostic half, so the three
#: a study most often sets are missing from it. Used to warn on a likely typo,
#: never to reject.
ACCEPTED_COGNITIVE_PARAMS: dict[str, set[str]] = {
    "sim2real": {
        "aggregation", "normalize_by_i_max", "confidence_scale", "confidence_intercept",
        "max_features_attended", "guess_bias", "lapse_rate", "use_exemplar_memory",
        "memory_sensitivity", "memory_decay", "retrieval_threshold",
        "strategy", "k", "sensitivity", "always_attend_changed",
    },
    "coxam": {
        "decision_noise", "memory_recall_threshold", "opportunity_cost",
        "random_response_rate", "counterfactual_overshoot_fraction", "time_penalty_weight",
    },
}


def resolve_cognitive_param(framework: str, label: str) -> str:
    """The parameter name an agent reads, from a UI ``cognitiveConfig`` label.

    Args:
        framework: The agent the parameter is for, e.g. ``coxam``.
        label: The UI's display label, e.g. ``"Diffusion Noise"``.

    Returns:
        The agent's own parameter name, or the plain slug when no alias applies.
    """
    slug = slugify(label)
    return COGNITIVE_PARAM_ALIASES.get(str(framework).lower(), {}).get(slug, slug)


def normalize_cognitive_params(
    framework: str, params: Optional[Mapping[str, Any]]
) -> Optional[dict[str, Any]]:
    """Resolve UI labels to an agent's parameter names and coerce the values.

    The design-export path and the server's ``coxam_eval_params`` /
    ``sim2real_params`` both receive whatever the UI sends, so both go through
    this. Passing already-resolved names is a no-op, which keeps existing API
    callers working.

    Args:
        framework: The agent the parameters are for, e.g. ``sim2real``.
        params: What the caller supplied, keyed by UI label or parameter name.

    Returns:
        The parameters keyed by the agent's own names, or None if none given.
    """
    if not params:
        return None if params is None else {}
    return {
        resolve_cognitive_param(framework, key): _coerce_cognitive_value(value)
        for key, value in params.items()
    }


def _coerce_cognitive_value(value: Any) -> Any:
    """Turn the UI's string values into the numbers/bools the agents expect.

    ``cognitiveConfig`` arrives as JSON strings (``"-2.5"``), and a string
    silently poisons arithmetic downstream. Anything that is not numeric or
    boolean is passed through untouched.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        number = float(text)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in text and "e" not in text.lower() else number


def _apply_cognitive_model(study: Any, design: DesignExport) -> None:
    """Select the agent the export names, with its cognitive parameters.

    Without this the export's ``userModel`` is parsed and dropped: the study
    keeps the placeholder, and ``run_experiment()`` runs that placeholder
    rather than the agent the design asked for -- a wrong answer that looks
    like a successful run.

    Parameters come from ``cognitiveConfig`` where the UI supplied them, and
    from the agent's own task-aware defaults where it did not. Values the UI
    did give always win, so a partly-filled config is topped up rather than
    replaced.
    """
    from src.cognitive_models import default_cognitive_params

    framework = design.resolved_framework
    if not framework:
        return

    tasks = design.user_tasks or None
    try:
        params = dict(default_cognitive_params(framework, tasks=tasks))
    except TypeError:
        # Older signature without task awareness.
        params = dict(default_cognitive_params(framework))

    # `default_cognitive_params` is not the full accepted set -- sim2real
    # deliberately returns only its condition-agnostic half, so the parameters a
    # study most often sets (aggregation, max_features_attended,
    # confidence_intercept) are absent from it. Unrecognised keys are therefore
    # kept and warned about rather than dropped: dropping would discard valid
    # parameters, and passing a typo through silently is what the warning
    # catches.
    known = set(params) | ACCEPTED_COGNITIVE_PARAMS.get(framework, set())
    for key, value in (design.cognitive_config or {}).items():
        name = resolve_cognitive_param(framework, key)
        if known and name not in known:
            design.report.add_warning(
                "cognitiveConfig",
                f"{key!r} resolved to {name!r}, which is not a known {framework} "
                f"parameter. It is passed through, but the model may ignore it.",
                suggestion=f"Known parameters: {', '.join(sorted(known))}.",
            )
        params[name] = _coerce_cognitive_value(value)

    study.set_cognitive_model(cognitive_model_id=framework, cognitive_params=params)


__all__ = [
    "ALLOCATION_ALIASES",
    "BASELINE_ALIASES",
    "BOOLEAN_IVS",
    "BOOLEAN_LEVEL_ALIASES",
    "DATASET_ALIASES",
    "DV_MEASURE_ALIASES",
    "IV_FACTOR_ALIASES",
    "MODEL_FRAMEWORK_ALIASES",
    "RANDOMIZATION_PATTERNS",
    "TASK_BY_DV",
    "DesignExport",
    "apply_design_export",
    "check_against_support_matrix",
    "load_design_export",
    "parse_design_export",
    "slugify",
    "split_levels",
]
