"""Convert a UI-exported experiment-design JSON into the XAIKit trial config.

The study-builder UI exports a human-facing document (`experiment-design.json`)
with display labels, free-text level strings, an apparatus list, and a procedure
outline. `generate_trials_from_ui_config` expects the canonical machine config
(`template.json`): slugged IV names, typed levels, and explicit dataset/sampling
/output blocks.

`convert_ui_design` bridges the two. Everything the trial generator cannot use
yet -- most notably the apparatus configurations -- is preserved on the
converted config under `apparatus`, `protocol`, and `planner_meta` so no
researcher intent is lost on the way through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Optional

from .protocol import normalize_study_protocol
from .support import load_support_matrix, normalize_iv_name, normalize_value
from .trials import TrialGenerationResult, generate_trials_from_ui_config


DEFAULT_MODEL_TYPE = "mlp"
DEFAULT_OUT_DIR = "experiment_output"
DEFAULT_SEED = 42

# UI shorthands for a supported DV -- the same measure under a different name.
# These are name aliases, not reinterpretations: `forward_sim` is the UI's label
# for forward-simulation accuracy.
#
# Anything measuring a *different* construct (`comprehension`, `decision_time`,
# `trust`) is deliberately absent and passes through unchanged, so support
# validation reports it rather than the converter guessing which XAIKit DV the
# researcher meant. Pass `dv_aliases` to `convert_ui_design` to declare such a
# mapping yourself.
DEFAULT_DV_ALIASES = {
    "forward_sim": "forward_accuracy",
    "forward_simulation": "forward_accuracy",
    "forward_simulation_accuracy": "forward_accuracy",
    "counterfactual_sim": "counterfactual_accuracy",
    "counterfactual_simulation": "counterfactual_accuracy",
    "counterfactual_simulation_accuracy": "counterfactual_accuracy",
}

_TRUE_LEVELS = {"with_xai", "xai", "yes", "true", "shown", "explanation", "on"}
_FALSE_LEVELS = {"without_xai", "no_xai", "none", "no", "false", "hidden", "control", "off"}

_COUNTERBALANCING_STRATEGIES = {
    "balanced_latin_square": "balanced_latin_square",
    "latin_square": "balanced_latin_square",
    "bradley": "balanced_latin_square",
    "complete": "complete",
    "complete_counterbalancing": "complete",
    "full_counterbalancing": "complete",
}

# Apparatus params the trial generator can still act on today.
_APPARATUS_READABLE_PARAMS = ("appId", "instanceIds", "xaiType", "expMethod")


@dataclass
class UIDesignConversion:
    """Result of converting a UI design document into a trial config."""

    config: dict[str, Any]
    apparatus: list[dict[str, Any]] = field(default_factory=list)
    protocol: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    def to_json(self, path: str | Path) -> str:
        """Write the converted config so it can be replayed with the JSON entry point."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2)
        return str(path)


def load_ui_design(path: str | Path) -> dict[str, Any]:
    """Load a UI-exported experiment-design JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def convert_ui_design(
    design: dict[str, Any],
    *,
    explanation_csv: Optional[str] = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    study_id: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    dv_aliases: Optional[dict[str, str]] = None,
    show: bool = True,
) -> UIDesignConversion:
    """Map a UI design document onto the canonical XAIKit trial-generation config.

    Args:
        design: Parsed `experiment-design.json` content from the study builder UI.
        explanation_csv: Explanation pool CSV. Defaults to
            `generated_explanation/de_{model_type}_{dataset_id}.csv`.
        model_type: AI model behind the explanation pool (planning metadata).
        study_id: Study identifier; defaults to `{dataset_id}_{user_model}_study`.
        seed: Random seed for participant assignment and instance sampling.
        out_dir: Directory for the generated trial artifacts.
        dv_aliases: Extra/overriding UI-DV to XAIKit-DV name mappings.
        show: Print the conversion report.

    Returns:
        UIDesignConversion whose `.config` is ready for `generate_trials_from_ui_config`.
    """
    support = load_support_matrix()
    notes: list[str] = []
    unsupported: list[str] = []

    study_design = design.get("studyDesign") or {}
    raw_answers = design.get("_rawAnswers") or {}

    counterbalancing: dict[str, str] = {}
    ivs = _convert_ivs(study_design, raw_answers, support, notes, counterbalancing)
    dvs = _convert_dvs(study_design, support, dv_aliases, notes, unsupported)
    cvs = _convert_factors(study_design.get("controlVariablesCV"), support)
    rvs = _convert_factors(study_design.get("randomVariablesRV"), support)

    dataset_id = _resolve_dataset_id(study_design, design, support, notes)
    if explanation_csv is None:
        explanation_csv = f"generated_explanation/de_{model_type}_{dataset_id}.csv"
        notes.append(
            f"No explanation CSV in the UI export; assuming `{explanation_csv}`. "
            "Pass `explanation_csv=...` if your pool lives elsewhere."
        )

    apparatus = _convert_apparatus(design.get("apparatus"), unsupported, notes)
    protocol = _convert_protocol(design, notes)
    sampling = _convert_sampling(study_design, ivs, counterbalancing, notes)

    user_model = design.get("userModel") or ""
    if study_id is None:
        study_id = "_".join(part for part in (dataset_id, _slug(user_model), "study") if part)

    config: dict[str, Any] = {
        "study_id": study_id,
        "seed": seed,
        "dataset": {
            "dataset_id": dataset_id,
            "model_type": model_type,
            "explanation_csv": explanation_csv,
            "id_map": {"dataId": "dataId", "instanceId": "instanceId"},
        },
        "ivs": ivs,
        "cvs": cvs,
        "dvs": dvs,
        "rvs": rvs,
        "sampling": sampling,
        "output": {
            "out_dir": str(out_dir),
            "trials_csv": "trials.csv",
            "trials_json": "trials.json",
            "summary_json": "design_summary.json",
        },
        # Retained but not consumed by trial generation yet.
        "apparatus": apparatus,
        "protocol": protocol,
        "planner_meta": {
            "research_questions": protocol["research_questions"],
            "user_model": user_model,
            "cognitive_config": design.get("cognitiveConfig") or {},
            "ml_proxy_baselines": design.get("mlProxyBaselines") or [],
            "design_label": study_design.get("design"),
            "ui_totals": {
                key: study_design.get(key)
                for key in (
                    "totalConditions",
                    "betweenSubjectsCells",
                    "participantsPerCondition",
                    "totalParticipants",
                    "trialsPerParticipant",
                    "totalTrials",
                )
                if study_design.get(key) is not None
            },
            "unsupported": unsupported,
            "conversion_notes": notes,
        },
    }

    _check_totals(study_design, config, notes)
    _check_trial_divisibility(config, notes)

    conversion = UIDesignConversion(
        config=config,
        apparatus=apparatus,
        protocol=protocol,
        notes=notes,
        unsupported=unsupported,
    )
    if show:
        print_ui_design_conversion(conversion)
    return conversion


def convert_ui_design_file(path: str | Path, **kwargs: Any) -> UIDesignConversion:
    """Load and convert a UI design JSON file in one call."""
    return convert_ui_design(load_ui_design(path), **kwargs)


def generate_trials_from_ui_design(
    source: str | Path | dict[str, Any],
    *,
    show: bool = True,
    strict: bool = False,
    validate_support: bool = True,
    **convert_kwargs: Any,
) -> tuple[UIDesignConversion, TrialGenerationResult]:
    """Convert a UI design document and generate trial artifacts from it.

    Args:
        source: Path to `experiment-design.json`, or the already-parsed dict.
        show: Print the conversion report, validation report, and artifact paths.
        strict: Raise on validation errors instead of only reporting them.
        validate_support: Run support validation before generating trials.
        **convert_kwargs: Forwarded to `convert_ui_design`.

    Returns:
        Tuple of (conversion, trial generation result).
    """
    design = source if isinstance(source, dict) else load_ui_design(source)
    conversion = convert_ui_design(design, show=show, **convert_kwargs)
    result = generate_trials_from_ui_config(
        conversion.config,
        validate_support=validate_support,
        strict=strict,
        show=show,
    )
    return conversion, result


def to_xaikit_test_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Split a converted config into the keyword groups `xaikitTest` consumes.

    `generate_trials_from_ui_config` reads the config as one document; the
    workflow object instead takes the same information through `set_design`,
    `prepare_dataset`, `set_study_protocol`, and `generate_trials`. This returns
    exactly those keyword groups so both entry points stay in sync.
    """
    dataset = config["dataset"]
    sampling = config["sampling"]
    return {
        "design": {
            "iv_config": config["ivs"],
            "cvs": config["cvs"],
            "dvs": config["dvs"],
        },
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "model_type": dataset["model_type"],
        },
        "trials": {
            "participants_per_between_condition": sampling["participants_per_between_condition"],
            "num_testing": sampling["trials_per_participant"],
            "counterbalancing_strategy": sampling["counterbalancing_strategy"],
            "trial_randomization_strategy": sampling["trial_randomization_strategy"],
            "instance_wise_explanation": sampling["instance_wise_explanation"],
            "shuffle_instances": sampling["shuffle_instances"],
            "seed": config["seed"],
            "output_dir": config["output"]["out_dir"],
        },
        "protocol": config.get("protocol") or {},
        "apparatus": config.get("apparatus") or [],
    }


def apply_ui_design(
    test: Any,
    source: str | Path | dict[str, Any],
    *,
    prepare_dataset: bool = True,
    set_protocol: bool = True,
    show: bool = True,
    **convert_kwargs: Any,
) -> UIDesignConversion:
    """Load a UI design onto a `xaikitTest` workflow object.

    Applies the converted IV/CV/DV design, optionally prepares the dataset and
    stores the study protocol, and attaches the retained apparatus list plus the
    converted config as `test.apparatus` / `test.ui_design_config`. Trial
    settings are not applied here -- pass
    `**conversion.config` through `to_xaikit_test_inputs(...)["trials"]` to
    `test.generate_trials(...)`.

    Args:
        test: A `xaikitTest` instance (duck-typed to avoid a circular import).
        source: Path to `experiment-design.json`, or the parsed dict.
        prepare_dataset: Call `test.prepare_dataset(...)` with the mapped dataset.
        set_protocol: Store the converted study protocol on the test object.
        show: Print the conversion report.
        **convert_kwargs: Forwarded to `convert_ui_design`.

    Returns:
        The `UIDesignConversion`; `to_xaikit_test_inputs(conversion.config)["trials"]`
        gives the matching `generate_trials(...)` keywords.
    """
    design = source if isinstance(source, dict) else load_ui_design(source)
    conversion = convert_ui_design(design, show=show, **convert_kwargs)
    inputs = to_xaikit_test_inputs(conversion.config)

    test.set_design(**inputs["design"], show=show)

    if prepare_dataset and inputs["dataset"]["dataset_id"]:
        test.prepare_dataset(
            inputs["dataset"]["dataset_id"],
            model_type=inputs["dataset"]["model_type"],
            show_available=False,
            show_summary=show,
        )

    if set_protocol and inputs["protocol"]:
        protocol = inputs["protocol"]
        # validate=False: the UI export carries no consent text yet.
        test.set_study_protocol(
            study_title=protocol.get("study_title") or test.project_name,
            research_questions=protocol.get("research_questions") or [],
            consent_text=protocol.get("consent_text") or "",
            procedure_steps=protocol.get("procedure_steps") or [],
            study_summary=protocol.get("study_summary") or "",
            validate=False,
        )
        if not protocol.get("consent_text"):
            conversion.notes.append(
                "Study protocol stored without consent text; call `set_study_protocol(...)` "
                "with the participant-facing consent before approving the walkthrough."
            )

    test.apparatus = conversion.apparatus
    test.ui_design_config = conversion.config
    return conversion


def print_ui_design_conversion(conversion: UIDesignConversion) -> None:
    """Print a compact summary of what the converter produced."""
    config = conversion.config
    dataset = config["dataset"]
    sampling = config["sampling"]

    print(f"UI design -> trial config: {config['study_id']}")
    print(f"  dataset     : {dataset['dataset_id']} (model_type={dataset['model_type']})")
    print(f"  explanations: {dataset['explanation_csv']}")
    print("  IVs:")
    for name, cfg in config["ivs"].items():
        randomization = cfg.get("randomization", "-")
        print(f"    {name:<16} type={cfg['type']:<8} randomization={randomization:<5} levels={cfg['levels']}")
    print(f"  CVs: {list(config['cvs'])}")
    print(f"  DVs: {list(config['dvs'])}")
    print(
        f"  sampling    : {sampling['participants_per_between_condition']} participants/between-cell, "
        f"{sampling['trials_per_participant']} trials/participant, "
        f"counterbalancing={sampling['counterbalancing_strategy']}"
    )
    if conversion.apparatus:
        print(f"  apparatus   : {len(conversion.apparatus)} configuration(s) retained (not yet consumed)")
        for entry in conversion.apparatus:
            instance_ids = entry.get("instance_ids") or []
            span = f"{len(instance_ids)} instance(s)" if instance_ids else "no instance ids"
            print(f"    - {entry['label']}: mode={entry['mode']}, {span}")
    if conversion.notes:
        print("  Notes:")
        for note in conversion.notes:
            print(f"    - {note}")
    if conversion.unsupported:
        print("  Not supported by the trial API yet:")
        for item in conversion.unsupported:
            print(f"    - {item}")


# ---------------------------------------------------------------- IV handling


def _convert_ivs(
    study_design: dict[str, Any],
    raw_answers: dict[str, Any],
    support: dict[str, Any],
    notes: list[str],
    counterbalancing: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Build the canonical IV config, preferring the raw machine-slugged answers.

    `counterbalancing` is filled in as an out-parameter with each IV's raw
    counterbalancing text, which `_convert_sampling` turns into a strategy.
    """
    raw_ivs = _load_raw_list(raw_answers.get("sd_ivs"))
    display_ivs = study_design.get("independentVariables") or []
    entries = _merge_iv_sources(raw_ivs, display_ivs)

    iv_config: dict[str, dict[str, Any]] = {}
    for entry in entries:
        raw_name, name = _resolve_iv_name(entry, support)
        if not name:
            continue
        if name != _slug(raw_name):
            notes.append(f"IV `{raw_name}` normalized to `{name}`.")

        is_boolean = (support["ivs"].get(name) or {}).get("type") == "boolean"
        levels = _parse_levels(entry.get("levels") or entry.get("levelsOrRange"), support, boolean=is_boolean)
        if not levels:
            notes.append(f"IV `{raw_name}` has no parsable levels and was skipped.")
            continue

        allocation = _slug(entry.get("alloc") or entry.get("allocation") or "within")
        iv_type = "between" if allocation.startswith("between") else "within"
        cfg: dict[str, Any] = {"type": iv_type, "levels": levels}
        if iv_type == "within":
            cfg["randomization"] = _resolve_randomization(entry, is_boolean)
        iv_config[name] = cfg
        counterbalancing[name] = str(entry.get("balancing") or entry.get("counterbalancing") or "")

    return iv_config


def _resolve_iv_name(entry: dict[str, Any], support: dict[str, Any]) -> tuple[str, str]:
    """Resolve a UI IV entry to a canonical name, trying its slug then its label.

    The UI's machine slug (`factor`) and its display label (`label`) can each be
    the one the support matrix knows -- e.g. `tested_xai` vs `Tested with XAI`.
    Prefer whichever resolves to a supported IV.
    """
    candidates = [entry.get("factor"), entry.get("label")]
    fallback = ""
    for candidate in candidates:
        if not str(candidate or "").strip():
            continue
        fallback = fallback or str(candidate)
        name = normalize_iv_name(candidate, support)
        if name in support["ivs"]:
            return str(candidate), name
    return fallback, normalize_iv_name(fallback, support) if fallback else ""


def _merge_iv_sources(raw_ivs: list[dict[str, Any]], display_ivs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlay the display IV list onto the raw one, matched by position then label."""
    if not raw_ivs:
        return list(display_ivs)

    by_label = {_slug(item.get("factor", "")): item for item in display_ivs}
    merged = []
    for index, raw in enumerate(raw_ivs):
        entry = dict(raw)
        display = display_ivs[index] if index < len(display_ivs) else by_label.get(_slug(raw.get("label", "")))
        if display:
            entry.setdefault("allocation", display.get("allocation"))
            entry.setdefault("counterbalancing", display.get("counterbalancing"))
            entry.setdefault("levelsOrRange", display.get("levelsOrRange"))
        merged.append(entry)
    return merged


def _resolve_randomization(entry: dict[str, Any], is_boolean: bool) -> str:
    """Pick block- vs trial-level randomization for a within-subjects IV."""
    hint = _slug(entry.get("balancing") or entry.get("counterbalancing") or "")
    if "randomized" in hint or "trial" in hint:
        return "trial"
    if hint:
        return "block"
    # Boolean IVs such as `tested_w_xai` vary trial by trial by convention.
    return "trial" if is_boolean else "block"


def _parse_levels(text: Any, support: dict[str, Any], *, boolean: bool = False) -> list[Any]:
    """Split a free-text level string into normalized, deduplicated levels."""
    if isinstance(text, list):
        raw_parts = [str(part) for part in text]
    elif text is None or not str(text).strip():
        return []
    else:
        raw_parts = _split_level_text(str(text))

    levels: list[Any] = []
    for part in raw_parts:
        cleaned = _strip_annotations(part)
        if not cleaned:
            continue
        if boolean:
            value = _to_boolean(cleaned)
        else:
            value = _as_number(cleaned)
            if value is None:
                value = normalize_value(cleaned, support)
        if value is None:
            continue
        if value not in levels:
            levels.append(value)
    return levels


def _split_level_text(text: str) -> list[str]:
    """Split on the separators the UI accepts, in order of specificity."""
    for pattern in (r"\|", r"\bvs\.?\b", r";", r",", r"/"):
        parts = [part.strip() for part in re.split(pattern, text, flags=re.IGNORECASE)]
        parts = [part for part in parts if part]
        if len(parts) > 1:
            return parts
    return [text.strip()] if text.strip() else []


def _strip_annotations(text: str) -> str:
    """Drop parenthetical notes like `Input Gradients (paper)` before normalizing."""
    return re.sub(r"\([^)]*\)", " ", text).strip()


def _to_boolean(text: str) -> Optional[bool]:
    """Map a boolean IV's free-text level onto True/False."""
    key = _slug(text)
    if key in _TRUE_LEVELS:
        return True
    if key in _FALSE_LEVELS:
        return False
    if key.startswith("without") or key.startswith("no_"):
        return False
    if key.startswith("with"):
        return True
    return None


# ------------------------------------------------------- DV / CV / RV handling


def _convert_dvs(
    study_design: dict[str, Any],
    support: dict[str, Any],
    dv_aliases: Optional[dict[str, str]],
    notes: list[str],
    unsupported: list[str],
) -> dict[str, list[Any]]:
    """Convert the UI dependent-variable list into `{name: [scale]}`."""
    aliases = {**DEFAULT_DV_ALIASES, **{_slug(k): v for k, v in (dv_aliases or {}).items()}}
    supported = set(support["global"]["dvs"])

    dvs: dict[str, list[Any]] = {}
    for item in study_design.get("dependentVariables") or []:
        raw_name = item.get("measure") or item.get("name") or ""
        if not str(raw_name).strip():
            continue
        name = str(normalize_value(raw_name, support))
        if _slug(raw_name) in aliases:
            name = aliases[_slug(raw_name)]
            notes.append(f"DV `{raw_name}` is a known alias for `{name}`.")
        if name not in supported:
            unsupported.append(
                f"DV `{raw_name}`: not in the XAIKit DV support matrix "
                f"(supported: {', '.join(sorted(supported))}). Kept in the config so "
                "validation reports it -- replace it with a supported DV, or measure it "
                "outside XAIKit."
            )
        dvs[name] = ["continuous"]
        if str(item.get("formula") or "").strip():
            notes.append(f"DV `{name}` has a custom formula in the UI export; compute it during analysis.")
    return dvs


def _convert_factors(items: Any, support: dict[str, Any]) -> dict[str, list[Any]]:
    """Convert a UI CV/RV list into `{name: [levels]}`."""
    factors: dict[str, list[Any]] = {}
    for item in items or []:
        if isinstance(item, str):
            name, levels_text = item, ""
        else:
            name = item.get("factor") or item.get("name") or ""
            levels_text = item.get("levels") or item.get("levelsOrRange") or ""
        if not str(name).strip():
            continue
        levels = _parse_levels(levels_text, support) or ["all"]
        factors[normalize_iv_name(name, support)] = levels
    return factors


# ------------------------------------------------------ dataset and sampling


def _resolve_dataset_id(
    study_design: dict[str, Any],
    design: dict[str, Any],
    support: dict[str, Any],
    notes: list[str],
) -> str:
    """Resolve the dataset id from the study design, falling back to apparatus params."""
    raw = study_design.get("dataset")
    if not str(raw or "").strip():
        for entry in design.get("apparatus") or []:
            app_id = (entry.get("params") or {}).get("appId")
            if str(app_id or "").strip():
                notes.append(f"Dataset taken from apparatus `appId={app_id}`.")
                raw = app_id
                break
    if not str(raw or "").strip():
        return ""

    dataset_id = normalize_value(raw, support)
    if str(dataset_id) != str(raw):
        notes.append(f"Dataset `{raw}` normalized to `{dataset_id}`.")
    return str(dataset_id)


def _convert_sampling(
    study_design: dict[str, Any],
    ivs: dict[str, dict[str, Any]],
    counterbalancing: dict[str, str],
    notes: list[str],
) -> dict[str, Any]:
    """Build the sampling block from UI participant/trial counts."""
    participants = _as_int(study_design.get("participantsPerCondition"), 24)
    trials = _as_int(study_design.get("trialsPerParticipant"), 12)

    # Block-counterbalanced within IVs are the ones a strategy applies to.
    strategy = "auto"
    for name, cfg in ivs.items():
        if cfg["type"] != "within" or cfg.get("randomization") != "block":
            continue
        hint = _COUNTERBALANCING_STRATEGIES.get(_slug(counterbalancing.get(name, "")))
        if hint:
            strategy = hint
            notes.append(f"Counterbalancing strategy `{hint}` taken from IV `{name}`.")
            break

    return {
        "participants_per_between_condition": participants,
        "trials_per_participant": trials,
        "counterbalancing_strategy": strategy,
        "trial_randomization_strategy": "balanced",
        "shuffle_instances": True,
        "instance_wise_explanation": False,
    }


def _check_totals(study_design: dict[str, Any], config: dict[str, Any], notes: list[str]) -> None:
    """Warn when the UI's headline participant count disagrees with the converted design."""
    expected_total = study_design.get("totalParticipants")
    if expected_total is None:
        return

    between_cells = 1
    for cfg in config["ivs"].values():
        if cfg["type"] == "between":
            between_cells *= len(cfg["levels"])
    derived = between_cells * config["sampling"]["participants_per_between_condition"]
    if int(expected_total) != derived:
        notes.append(
            f"UI reports {expected_total} total participants, but the converted design yields "
            f"{derived} ({between_cells} between-cells x "
            f"{config['sampling']['participants_per_between_condition']} per cell). "
            "Check the IV allocations."
        )


def within_cell_count(ivs: dict[str, dict[str, Any]]) -> int:
    """Number of block-within x trial-within cells a participant must cover.

    Balanced trial randomization requires `trials_per_participant` to be a
    multiple of this. See `build_trial_sequence` in `counterbalance.py`.
    """
    cells = 1
    for cfg in ivs.values():
        if cfg.get("type") == "within":
            cells *= max(1, len(cfg.get("levels") or []))
    return cells


def _check_trial_divisibility(config: dict[str, Any], notes: list[str]) -> None:
    """Warn when trials per participant cannot be balanced across within cells."""
    sampling = config["sampling"]
    if sampling["trial_randomization_strategy"] != "balanced":
        return

    cells = within_cell_count(config["ivs"])
    trials = sampling["trials_per_participant"]
    if cells <= 1 or trials % cells == 0:
        return

    suggestion = ((trials + cells - 1) // cells) * cells
    notes.append(
        f"{trials} trials per participant does not divide evenly across the {cells} "
        f"within-subjects cells, so balanced randomization will fail. Use "
        f"{suggestion} trials (the next multiple of {cells}), or set "
        "`trial_randomization_strategy='random'`."
    )


# ------------------------------------------------- apparatus (retained, unused)


def _convert_apparatus(
    apparatus: Any,
    unsupported: list[str],
    notes: list[str],
) -> list[dict[str, Any]]:
    """Normalize apparatus configurations so they survive the conversion intact.

    The trial generator does not consume apparatus yet. Entries are parsed into a
    stable shape (with instance-id ranges expanded) and carried on the config so a
    future interface layer -- or a manual export -- can pick them up.
    """
    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(_apparatus_entries(apparatus)):
        params = dict(raw.get("params") or {})
        instance_ids = _parse_instance_ids(params.get("instanceIds"))
        entries.append(
            {
                "id": raw.get("id") or f"apparatus_{index}",
                "label": raw.get("label") or f"Configuration {index + 1}",
                "group": raw.get("group") or "All participants",
                "mode": raw.get("mode") or "ours",
                "url": raw.get("url") or "",
                "app_id": params.get("appId"),
                "xai_type": params.get("xaiType"),
                "xai_method": params.get("expMethod"),
                "instance_ids": instance_ids,
                "params": params,
                "supported_by_trial_api": False,
            }
        )

    if entries:
        unsupported.append(
            f"apparatus ({len(entries)} configuration(s)): interface parameters such as "
            "showTutorial/showPrediction/userPrediction/focusOnImportant/widgets have no trial-API "
            "equivalent yet. Retained verbatim under `config['apparatus']`."
        )
        readable = sorted({
            key
            for entry in entries
            for key in entry["params"]
            if key in _APPARATUS_READABLE_PARAMS
        })
        if readable:
            notes.append(
                f"Apparatus params {', '.join(readable)} overlap with design fields; "
                "the converted config uses the Study Design values, not the apparatus values."
            )
    return entries


def _apparatus_entries(apparatus: Any) -> list[dict[str, Any]]:
    """Accept both apparatus shapes the UI emits: a list, or a single dict.

    Older exports describe the apparatus as one free-text block
    (`{"description": ..., "link": ...}`); newer ones list per-condition
    configurations with a `params` map.
    """
    if isinstance(apparatus, dict):
        if not any(str(value or "").strip() for value in apparatus.values()):
            return []
        entry = dict(apparatus)
        entry.setdefault("label", "Apparatus")
        entry.setdefault("url", entry.get("link", ""))
        entry["params"] = {
            key: value for key, value in apparatus.items() if key not in ("label", "url", "link")
        }
        return [entry]
    return [item for item in (apparatus or []) if isinstance(item, dict)]


def _parse_instance_ids(text: Any) -> list[int]:
    """Expand an instance-id spec such as `1-10` or `1,3,5-7` into a list of ids."""
    if text is None or not str(text).strip():
        return []

    ids: list[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", chunk)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            step = 1 if end >= start else -1
            ids.extend(range(start, end + step, step))
        elif chunk.isdigit():
            ids.append(int(chunk))
    return ids


# ------------------------------------------------------------------- protocol


def _convert_protocol(design: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    """Convert the UI procedure outline into a normalized study protocol."""
    steps = []
    consent_text = ""
    for step in design.get("procedure") or []:
        title = str(step.get("title") or "").strip()
        note = str(step.get("note") or "").strip()
        stage = _infer_step_stage(title)
        # A consent step's note is the participant-facing consent copy.
        if stage == "consent" and note and not consent_text:
            consent_text = note
        steps.append({"title": title, "stage": stage, "description": note})

    research_questions = design.get("researchQuestions") or ""
    protocol = normalize_study_protocol(
        {
            "study_title": str(design.get("studyTitle") or "").strip(),
            "research_questions": research_questions,
            "consent_text": consent_text,
            "procedure_steps": steps or None,
        }
    )
    if steps and not any(step["stage"] == "trials" for step in protocol["procedure_steps"]):
        notes.append(
            "No procedure step was recognized as the main trial block; "
            "set `stage='trials'` on one step before previewing the walkthrough."
        )
    return protocol


_STEP_STAGE_KEYWORDS = (
    ("consent", "consent"),
    ("practice", "practice"),
    ("training", "practice"),
    ("tutorial", "practice"),
    ("trial", "trials"),
    ("main task", "trials"),
    ("debrief", "debrief"),
    ("questionnaire", "survey"),
    ("survey", "survey"),
    ("demographic", "survey"),
)


def _infer_step_stage(title: str) -> str:
    """Infer a protocol step stage from its UI title."""
    lowered = title.lower()
    for keyword, stage in _STEP_STAGE_KEYWORDS:
        if keyword in lowered:
            return stage
    return "information"


# ---------------------------------------------------------------- small utils


def _load_raw_list(value: Any) -> list[dict[str, Any]]:
    """Parse a `_rawAnswers` field that holds a JSON-encoded list."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _as_number(text: str) -> Optional[int | float]:
    """Return a numeric level as int/float, or None when the text is not numeric."""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


__all__ = [
    "DEFAULT_DV_ALIASES",
    "UIDesignConversion",
    "apply_ui_design",
    "convert_ui_design",
    "convert_ui_design_file",
    "generate_trials_from_ui_design",
    "load_ui_design",
    "print_ui_design_conversion",
    "to_xaikit_test_inputs",
]
