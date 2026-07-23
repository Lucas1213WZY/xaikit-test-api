"""Semantic support checks for XAIKit experiment configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from difflib import get_close_matches
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Iterable, Optional


DEFAULT_SUPPORT_MATRIX = Path(__file__).with_name("support_matrix.json")
SUPPORTED_STAGES = frozenset({"design", "trial_generation", "explanation_generation", "execution"})
_STAGE_ORDER = {stage: index for index, stage in enumerate(("design", "trial_generation", "explanation_generation", "execution"))}

_DV_HINTS = {
    "accuracy": (
        "`accuracy` is ambiguous in the current XAIKit standard.",
        "Use `forward_accuracy` for forward simulation or `counterfactual_accuracy` for counterfactual simulation.",
    ),
    "task_time": (
        "`task_time` is not a currently supported DV.",
        "Use one of: counterfactual_accuracy, forward_accuracy.",
    ),
    "pred_time": (
        "`pred_time` is not a currently supported DV.",
        "Use one of: counterfactual_accuracy, forward_accuracy.",
    ),
    "response_time": (
        "`response_time` is not a currently supported DV.",
        "Use one of: counterfactual_accuracy, forward_accuracy.",
    ),
}


@dataclass
class ValidationIssue:
    """One support-check message."""

    field: str
    message: str
    suggestion: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {"field": self.field, "message": self.message}
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result


@dataclass
class ValidationReport:
    """Structured result for support validation."""

    stage: str
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    reminders: list[ValidationIssue] = field(default_factory=list)
    normalized: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, severity: str, field: str, message: str, suggestion: Optional[str] = None) -> None:
        getattr(self, severity).append(ValidationIssue(field, message, suggestion))

    def add_error(self, field: str, message: str, suggestion: Optional[str] = None) -> None:
        self.add("errors", field, message, suggestion)

    def add_warning(self, field: str, message: str, suggestion: Optional[str] = None) -> None:
        self.add("warnings", field, message, suggestion)

    def add_reminder(self, field: str, message: str, suggestion: Optional[str] = None) -> None:
        self.add("reminders", field, message, suggestion)

    def extend(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.reminders.extend(other.reminders)
        self.normalized.update(other.normalized)

    def raise_for_errors(self) -> None:
        if self.errors:
            first = self.errors[0]
            detail = f"{first.field}: {first.message}"
            if first.suggestion:
                detail += f" Suggested fix: {first.suggestion}"
            raise ValueError(detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "reminders": [issue.to_dict() for issue in self.reminders],
            "normalized": self.normalized,
        }

    def to_text(self, *, include_normalized: bool = True) -> str:
        """Return a notebook-friendly text rendering of this report."""
        return format_validation_report(self, include_normalized=include_normalized)

    def __str__(self) -> str:
        return self.to_text(include_normalized=True)

    def __repr__(self) -> str:
        return self.to_text(include_normalized=True)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("ValidationReport(...)" if cycle else self.to_text(include_normalized=True))


def load_support_matrix(path: str | Path | None = None) -> dict[str, Any]:
    """Load and compile the single support standard used by validation."""
    matrix_path = Path(path) if path is not None else DEFAULT_SUPPORT_MATRIX
    return deepcopy(_load_support_matrix_cached(str(matrix_path)))


@lru_cache(maxsize=8)
def _load_support_matrix_cached(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return _compile_support_matrix(json.load(handle))


def validate_design_support(
    iv_config: dict[str, dict[str, Any]],
    cvs: dict[str, list[Any]],
    dvs: dict[str, list[Any]],
    *,
    stage: str = "design",
    support: Optional[dict[str, Any]] = None,
    strict: bool = False,
    show: bool = False,
) -> ValidationReport:
    """Validate IV/DV names plus supported semantic CVs against the support matrix."""
    support = support or load_support_matrix()
    report = ValidationReport(stage=normalize_stage(stage))
    supported_ivs = support["global"]["ivs"]
    supported_dvs = set(support["global"]["dvs"])
    semantic_cvs = set(support.get("semantic_cvs", ()))
    normalized_ivs: dict[str, list[Any]] = {}
    normalized_semantic_cvs: dict[str, list[Any]] = {}
    normalized_dvs: list[str] = []

    if not iv_config:
        report.add_warning("ivs", "No IVs are configured yet.", "Add IVs with `add_iv(...)` or pass an `iv_config` before trial generation.")

    for raw_name, cfg in iv_config.items():
        name = normalize_iv_name(raw_name, support)
        spec = supported_ivs.get(name)
        if spec is None:
            report.add_error(raw_name, f"`{raw_name}` is not in the XAIKit IV support matrix.", _suggest_supported(raw_name, supported_ivs, prefix="Supported IVs"))
            continue

        levels = cfg.get("levels") if isinstance(cfg, dict) else None
        if not isinstance(levels, list) or not levels:
            report.add_error(name, "This IV has no levels to validate.", "Provide a non-empty `levels` list.")
            continue

        normalized_levels = [normalize_value(level, support) for level in levels]
        normalized_ivs[name] = normalized_levels
        for raw_level, normalized in zip(levels, normalized_levels):
            _validate_level(report, name, raw_level, normalized, spec)

        if name == "tested_w_xai" and (cfg.get("type") != "within" or cfg.get("randomization") != "trial"):
            report.add_warning(
                name,
                "`tested_w_xai` is supported as a trial-level within-subjects IV.",
                "Use `{'type': 'within', 'randomization': 'trial', 'levels': [True, False]}`.",
            )

    for raw_name, levels in cvs.items():
        name = normalize_iv_name(raw_name, support)
        if name not in semantic_cvs:
            continue
        spec = supported_ivs.get(name)
        if spec is None:
            continue
        if not isinstance(levels, list) or not levels:
            report.add_error(name, "This CV has no levels to validate.", "Provide a non-empty level list.")
            continue
        normalized_levels = [normalize_value(level, support) for level in levels]
        normalized_semantic_cvs[name] = normalized_levels
        for raw_level, normalized in zip(levels, normalized_levels):
            _validate_level(report, name, raw_level, normalized, spec)

    if not dvs:
        report.add_warning("dvs", "No DVs are configured yet.", f"Supported DVs are: {_format_values(supported_dvs)}.")

    for raw_name in dvs:
        name = normalize_value(raw_name, support)
        normalized_dvs.append(str(name))
        if name not in supported_dvs:
            message, suggestion = _DV_HINTS.get(
                _slug(raw_name),
                (f"`{raw_name}` is not a supported DV.", f"Use one of: {_format_values(supported_dvs)}."),
            )
            report.add_error(raw_name, message, suggestion)

    has_faithfulness = "xai_faithfulness" in normalized_ivs or "xai_faithfulness" in normalized_semantic_cvs
    if {"xai_robustness", "xai_sparsity"} & normalized_ivs.keys() and not has_faithfulness:
        report.add_reminder(
            "xai_faithfulness",
            "Robustness and sparsity settings usually assume faithfulness is controlled.",
            "If you follow the Sim2Real convention, set faithfulness to 0.8 or keep it fixed outside the IV list.",
        )

    report.normalized.update(
        {
            "ivs": normalized_ivs,
            "semantic_cvs": normalized_semantic_cvs,
            "dvs": normalized_dvs,
            "cvs": list(cvs),
        }
    )
    _finish_report(report, strict=strict, show=show)
    return report


def validate_ui_config_support(
    config: dict[str, Any],
    *,
    stage: str = "trial_generation",
    support: Optional[dict[str, Any]] = None,
    strict: bool = False,
    show: bool = True,
) -> ValidationReport:
    """Validate a UI-exported config before trial generation consumes it."""
    support = support or load_support_matrix()
    report = validate_design_support(
        config.get("ivs", {}),
        config.get("cvs", {}),
        config.get("dvs", {}),
        stage=stage,
        support=support,
        strict=False,
        show=False,
    )

    dataset_cfg = config.get("dataset")
    if not isinstance(dataset_cfg, dict):
        report.add_warning(
            "dataset",
            "No dataset block was found in the UI config.",
            "Include `dataset.dataset_id` and `dataset.explanation_csv` before trial generation; `dataset.model_type` is optional planning metadata.",
        )
    elif dataset_cfg.get("dataset_id"):
        _validate_dataset(report, support, dataset_cfg["dataset_id"], None)
    else:
        report.add_warning("dataset.dataset_id", "No dataset id was found in the UI config.", "Set `dataset.dataset_id`, for example `wine_quality`.")

    _finish_report(report, strict=strict, show=show)
    return report


def validate_xaikit_test(
    test: Any,
    *,
    stage: str = "design",
    support: Optional[dict[str, Any]] = None,
    strict: bool = False,
    show: bool = True,
) -> ValidationReport:
    """Validate the full XAIKitTest workflow object for a given stage."""
    support = support or load_support_matrix()
    stage = normalize_stage(stage)
    report = validate_design_support(
        getattr(test, "iv_config", {}),
        getattr(test, "CVs", {}),
        getattr(test, "DVs", {}),
        stage=stage,
        support=support,
        strict=False,
        show=False,
    )
    context = _extract_test_context(test, support, report)
    report.normalized["context"] = context
    _validate_stage_requirements(report, test, stage)
    _validate_model_specific(report, support, context, getattr(test, "cognitive_params", {}), stage)
    _finish_report(report, strict=strict, show=show)
    return report


def normalize_stage(stage: str) -> str:
    """Return the canonical validation stage."""
    key = _slug(stage)
    if key not in SUPPORTED_STAGES:
        raise ValueError(f"Unknown validation stage {stage!r}. Supported stages: {_format_values(SUPPORTED_STAGES)}.")
    return key


def normalize_iv_name(name: Any, support: Optional[dict[str, Any]] = None) -> str:
    """Normalize a UI/display IV name to the canonical support key."""
    return _aliases(support or load_support_matrix(), "iv").get(_slug(name), _slug(name))


def normalize_value(value: Any, aliases: Optional[dict[str, Any]] = None) -> Any:
    """Normalize user-facing labels while preserving booleans and numbers."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    flat_aliases = _aliases(aliases or load_support_matrix(), "value")
    text = str(value).strip()
    return flat_aliases.get(text.lower(), flat_aliases.get(_slug(text), _slug(text)))


def format_validation_report(report: ValidationReport, *, include_normalized: bool = False) -> str:
    """Render a friendly validation report."""
    has_messages = bool(report.errors or report.warnings or report.reminders)
    if not has_messages and not (include_normalized and report.normalized):
        return ""

    lines = [
        f"XAIKit validation: {report.stage}",
        "=" * (20 + len(report.stage)),
    ]
    if report.stage == "design" and has_messages:
        lines.append(_gentle_greeting())

    for title, issues in (("Fix before running", report.errors), ("Warnings", report.warnings), ("Reminders", report.reminders)):
        if issues:
            lines.extend(["", f"{title}:"])
            lines.extend(_format_issue_lines(issues))
    if include_normalized and report.normalized:
        lines.extend(_format_normalized_lines(report.normalized))
    return "\n".join(lines)


def print_validation_report(report: ValidationReport) -> None:
    """Print a rendered validation report."""
    text = format_validation_report(report)
    if text:
        print(text)


def _compile_support_matrix(raw: dict[str, Any]) -> dict[str, Any]:
    compiled = _resolve_refs(raw, raw)
    compiled["_aliases"] = {
        kind: _flatten_aliases(compiled.get("aliases", {}).get(kind, {}))
        for kind in ("iv", "param", "value")
    }
    compiled["global"] = {
        "ivs": compiled["ivs"],
        "dvs": compiled["groups"]["dvs"]["supported"],
        "aliases": compiled["_aliases"]["value"],
    }
    return compiled


def _resolve_refs(value: Any, root: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return deepcopy(_resolve_refs(_lookup_ref(root, value), root))
    if isinstance(value, dict):
        return {key: _resolve_refs(item, root) for key, item in value.items()}
    if isinstance(value, list):
        resolved: list[Any] = []
        for item in value:
            item = _resolve_refs(item, root)
            resolved.extend(item if isinstance(item, list) else [item])
        return _unique(resolved)
    return value


def _lookup_ref(root: dict[str, Any], ref: str) -> Any:
    node: Any = root
    for part in ref[1:].split("."):
        node = node[part]
    return node


def _flatten_aliases(canonical_aliases: dict[str, list[str]]) -> dict[str, str]:
    flat: dict[str, str] = {}
    for canonical, aliases in canonical_aliases.items():
        for label in [canonical, *aliases]:
            flat[str(label).lower()] = canonical
            flat[_slug(label)] = canonical
    return flat


def _validate_level(report: ValidationReport, name: str, raw_level: Any, normalized: Any, spec: dict[str, Any]) -> None:
    if spec.get("type") == "boolean":
        if not isinstance(raw_level, bool):
            report.add_error(name, f"`{raw_level}` is not a boolean level.", "Use `True` and `False`.")
        return

    allowed = spec.get("levels")
    if allowed and normalized in allowed:
        return

    numeric_range = spec.get("numeric_range") or spec.get("range")
    if numeric_range or spec.get("numeric"):
        value = _as_float(raw_level)
        if value is None:
            suggestion = f"Use a number or one of: {_format_values(allowed)}." if allowed else None
            report.add_error(name, f"`{raw_level}` must be numeric for `{name}`.", suggestion)
            return
        if spec.get("integer") and not value.is_integer():
            report.add_error(name, f"`{raw_level}` must be an integer.")
        if numeric_range:
            low, high = numeric_range
            if value < low or value > high:
                report.add_error(name, f"`{raw_level}` is outside the supported range [{low}, {high}].", f"Choose a value from {low} to {high}.")
        return

    if allowed:
        if name == "user_task":
            report.add_error(
                name,
                f"`{raw_level}` is not supported for cognitive simulation.",
                f"Planner-only OK. Supported: {_format_values(allowed)}.",
            )
            return
        report.add_error(name, f"`{raw_level}` is not supported for `{name}`.", f"Use one of: {_format_values(allowed)}.")


def _extract_test_context(test: Any, support: dict[str, Any], report: ValidationReport) -> dict[str, Any]:
    ivs = report.normalized.get("ivs", {})
    semantic_cvs = report.normalized.get("semantic_cvs", {})
    dataset_id = getattr(getattr(test, "data", None), "dataset_id", None) or _first_factor_value(ivs, semantic_cvs, "dataset")
    model_name = getattr(test, "model_name", None) or _first_factor_value(ivs, semantic_cvs, "ai_model")
    model_source = normalize_value(getattr(test, "model_source", None), support)
    cognitive_model_id = normalize_value(getattr(test, "cognitive_model_id", None) or "placeholder", support)

    if cognitive_model_id in {"placeholder", "custom"} and model_source in {"coax", "coxam"}:
        cognitive_model_id = model_source
        report.add_reminder(
            "cognitive_model_id",
            f"I inferred `{model_source}` from the AI model source for support checking.",
            f"Set `cognitive_model_id='{model_source}'` in `set_cognitive_model(...)` to make this explicit.",
        )

    return {
        "dataset_id": normalize_value(dataset_id, support) if dataset_id is not None else None,
        "model_name": normalize_value(model_name, support) if model_name is not None else None,
        "model_source": model_source,
        "cognitive_model_id": cognitive_model_id,
        "tasks": _factor_values(ivs, semantic_cvs, "user_task"),
        "xai_methods": ivs.get("xai_method", []),
        "xai_types": ivs.get("xai_type", []),
        "xai_faithfulness": _factor_values(ivs, semantic_cvs, "xai_faithfulness"),
        "dvs": report.normalized.get("dvs", []),
    }


def _validate_stage_requirements(report: ValidationReport, test: Any, stage: str) -> None:
    checks = (
        ("trial_generation", "data", lambda t, r: getattr(t, "data", None) is None, "No prepared dataset is attached to this XAIKitTest object.", "Call `prepare_dataset(...)` before this stage."),
        ("trial_generation", "xai_method", lambda t, r: not ({"xai_method", "xai_type"} & r.normalized.get("ivs", {}).keys()), "No XAI method/type IV was found.", "Add `xai_method` or `xai_type` if explanation condition is part of the experiment."),
        ("explanation_generation", "trained_engine", lambda t, r: getattr(t, "trained_engine", None) is None, "No trained AI model is attached yet.", "Call `train_AI_model(...)` before generating explanations."),
        ("execution", "trials", lambda t, r: not getattr(t, "trials", []), "No generated trials were found.", "Call `generate_trials(...)` before execution."),
        ("execution", "combined_explanations", lambda t, r: getattr(t, "combined_explanations", None) is None, "No combined explanation table was found.", "Call `explanations(...)`, or pass an `explanation_pool` to `run_experiment(...)`."),
    )
    for min_stage, field, predicate, message, suggestion in checks:
        if _STAGE_ORDER[stage] >= _STAGE_ORDER[min_stage] and predicate(test, report):
            report.add_warning(field, message, suggestion)

def _validate_model_specific(
    report: ValidationReport,
    support: dict[str, Any],
    context: dict[str, Any],
    cognitive_params: dict[str, Any],
    stage: str,
) -> None:
    model_id = context.get("cognitive_model_id") or "placeholder"
    models = support["cognitive_models"]
    model_spec = models.get(model_id)
    if model_spec is None:
        report.add_error("cognitive_model_id", f"`{model_id}` is not a supported cognitive model id.", f"Use one of: {_format_values(models)}.")
        return

    if not context.get("tasks"):
        report.add_reminder(
            "user_task",
            "Set `user_task` for cognitive simulation.",
            f"Planner-only OK. Supported: {_format_values(support['global']['ivs']['user_task']['levels'])}.",
        )

    _validate_dataset(report, support, context.get("dataset_id"), model_id)
    task_values = context.get("tasks") or ["forward_simulation"]
    globally_supported_tasks = set(support["global"]["ivs"]["user_task"]["levels"])
    if all(task in globally_supported_tasks for task in task_values):
        _check_allowed(report, "user_task", task_values, model_spec["tasks"], model_id, "tasks")
    _check_allowed(report, "dvs", [dv for dv in context.get("dvs", []) if dv in support["global"]["dvs"]], model_spec["dvs"], model_id, "DVs")
    _check_allowed(report, "xai_type", context.get("xai_types", []), model_spec["xai_types"], model_id, "XAI types", skip={"none"})
    _check_allowed(report, "xai_method", context.get("xai_methods", []), model_spec["xai_methods"], model_id, "XAI methods", skip={"none"})

    _validate_cognitive_params(report, support, model_id, model_spec, cognitive_params)
    if stage == "explanation_generation" and model_id == "coxam":
        report.add_reminder("xai_adapter", "CoXAM explanations are surrogate-style methods.", "Use decision-tree/logistic-regression adapters or precomputed CoXAM explanation tables.")


def _check_allowed(
    report: ValidationReport,
    field: str,
    values: Iterable[Any],
    allowed: Iterable[Any],
    owner: str,
    label: str,
    *,
    skip: set[Any] | None = None,
) -> None:
    allowed_set = set(allowed)
    for value in values:
        if value in (skip or set()):
            continue
        if value not in allowed_set:
            if field == "user_task":
                report.add_error(
                    field,
                    f"`{value}` is not supported by the selected cognitive model.",
                    f"Planner-only OK. This model supports: {_format_values(allowed_set)}.",
                )
                continue
            report.add_error(field, f"`{value}` is not supported by `{owner}`.", f"Supported {label} for `{owner}`: {_format_values(allowed_set)}.")


def _validate_dataset(report: ValidationReport, support: dict[str, Any], dataset_id: Any, model_id: Optional[str]) -> None:
    if dataset_id is None:
        return
    dataset_key = normalize_value(dataset_id, support)
    global_datasets = set(support["global"]["ivs"]["dataset"]["levels"])
    if dataset_key not in global_datasets:
        report.add_error("dataset", f"`{dataset_id}` is not a supported dataset id.", f"Use one of: {_format_values(global_datasets)}.")
        return

    model_spec = support["cognitive_models"].get(model_id or "")
    if model_spec is None:
        return
    supported = set(model_spec.get("datasets_supported", []))
    if dataset_key not in supported:
        report.add_error("dataset", f"`{dataset_key}` is not supported by `{model_id}`.", f"Supported datasets for `{model_id}`: {_format_values(supported)}.")
        return
    tested = set(model_spec.get("datasets_tested", []))
    if tested and dataset_key not in tested:
        report.add_warning("dataset", f"`{dataset_key}` is available for `{model_id}`, but it is not one of the paper-tested datasets in the support matrix.", f"Paper-tested datasets for `{model_id}`: {_format_values(tested)}.")


def _validate_cognitive_params(
    report: ValidationReport,
    support: dict[str, Any],
    model_id: str,
    model_spec: dict[str, Any],
    cognitive_params: dict[str, Any],
) -> None:
    param_spec = model_spec.get("cognitive_params", {})
    if not param_spec:
        return
    if not cognitive_params:
        report.add_warning("cognitive_params", f"No cognitive parameters were found for `{model_id}`.", f"Supported parameters: {_format_values(param_spec)}.")
        return

    unknown: list[str] = []
    seen: set[str] = set()
    for raw_name, value in cognitive_params.items():
        name = _aliases(support, "param").get(_slug(raw_name), _slug(raw_name))
        spec = param_spec.get(name)
        if spec is None:
            unknown.append(str(raw_name))
            continue
        seen.add(name)
        levels = spec.get("levels")
        if levels is not None:
            normalized = normalize_value(value, support)
            if normalized not in levels:
                report.add_error(raw_name, f"`{value}` is not supported for `{raw_name}`.", f"Use one of: {_format_values(levels)}.")
            continue
        low_high = spec.get("range") or spec.get("numeric_range")
        if low_high is not None:
            numeric_value = _as_float(value)
            if numeric_value is None:
                report.add_error(raw_name, f"`{raw_name}` must be numeric for `{model_id}`.")
                continue
            low, high = low_high
            if spec.get("integer") and not numeric_value.is_integer():
                report.add_error(raw_name, f"`{raw_name}` must be an integer for `{model_id}`.")
            if numeric_value < low or numeric_value > high:
                report.add_error(raw_name, f"`{value}` is outside the supported `{model_id}` range [{low}, {high}].", f"Choose a value from {low} to {high}.")

    if unknown:
        report.add_warning("cognitive_params", f"These parameters are not part of the standard `{model_id}` set: {_format_values(unknown)}.", f"Supported parameters: {_format_values(param_spec)}.")
    missing = [name for name in param_spec if name not in seen]
    if missing and model_id in {"coax", "coxam"}:
        report.add_reminder("cognitive_params", f"Some standard `{model_id}` cognitive parameters are not set: {_format_values(missing)}.", "Set them before fitting or simulating that cognitive model.")


def _finish_report(report: ValidationReport, *, strict: bool, show: bool) -> None:
    if show:
        print_validation_report(report)
    if strict:
        report.raise_for_errors()


def _aliases(source: dict[str, Any], kind: str) -> dict[str, str]:
    if "_aliases" in source:
        return source["_aliases"].get(kind, {})
    if "aliases" in source and kind in source["aliases"]:
        return _flatten_aliases(source["aliases"][kind])
    return source


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(values: Optional[list[Any]]) -> Any:
    return values[0] if values else None


def _factor_values(ivs: dict[str, list[Any]], semantic_cvs: dict[str, list[Any]], name: str) -> list[Any]:
    return ivs.get(name) or semantic_cvs.get(name) or []


def _first_factor_value(ivs: dict[str, list[Any]], semantic_cvs: dict[str, list[Any]], name: str) -> Any:
    return _first(_factor_values(ivs, semantic_cvs, name))


def _unique(values: Iterable[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _format_values(values: Iterable[Any]) -> str:
    return ", ".join(str(value) for value in sorted(values, key=str))


def _suggest_supported(raw: str, supported: Iterable[str], *, prefix: str) -> str:
    supported_list = sorted(supported)
    matches = get_close_matches(_slug(raw), supported_list, n=3)
    if matches:
        return f"Did you mean {_format_values(matches)}? {prefix}: {_format_values(supported_list)}."
    return f"{prefix}: {_format_values(supported_list)}."


def _format_issue_lines(issues: list[ValidationIssue]) -> list[str]:
    lines = []
    for issue in issues:
        lines.append(f"  - {issue.field}: {issue.message}")
        if issue.suggestion:
            lines.append(f"    Suggestion: {issue.suggestion}")
    return lines


def _format_normalized_lines(normalized: dict[str, Any]) -> list[str]:
    lines = ["", "Normalized:"]
    _append_mapping_lines(lines, "IVs", normalized.get("ivs"))
    _append_mapping_lines(lines, "Semantic CVs", normalized.get("semantic_cvs"))
    _append_sequence_lines(lines, "DVs", normalized.get("dvs"))
    _append_sequence_lines(lines, "CVs", normalized.get("cvs"))
    _append_mapping_lines(lines, "Context", normalized.get("context"), include_empty=True)
    return lines


def _append_mapping_lines(
    lines: list[str],
    heading: str,
    values: Optional[dict[str, Any]],
    *,
    include_empty: bool = False,
) -> None:
    if not values and not include_empty:
        return
    lines.append(f"  {heading}:")
    if not values:
        lines.append("    not set")
        return
    for key, value in values.items():
        lines.append(f"    {key}: {_display_value(value)}")


def _append_sequence_lines(lines: list[str], heading: str, values: Optional[list[Any]]) -> None:
    if values:
        lines.append(f"  {heading}: {_display_value(values)}")


def _display_value(value: Any) -> str:
    if value is None or value == [] or value == {}:
        return "not set"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _gentle_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning."
    if hour < 18:
        return "Good afternoon."
    return "Good evening."
