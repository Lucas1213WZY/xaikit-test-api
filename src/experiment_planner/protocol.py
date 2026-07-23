"""Researcher-authored study protocol and notebook editor helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Optional


DEFAULT_PROCEDURE = [
    {
        "title": "Welcome and consent",
        "kind": "consent",
        "description": "Explain the study and collect informed consent.",
        "duration_min": 3,
    },
    {
        "title": "Start-of-study survey",
        "kind": "survey",
        "description": "Collect only the background information needed for the analysis.",
        "duration_min": 3,
    },
    {
        "title": "Instructions and practice",
        "kind": "practice",
        "description": "Explain the task and provide practice questions with feedback.",
        "duration_min": 5,
    },
    {
        "title": "Main questions",
        "kind": "trials",
        "description": "Present the randomized test questions and collect the dependent variables.",
        "duration_min": 15,
    },
    {
        "title": "End-of-study survey",
        "kind": "survey",
        "description": "Collect the planned post-task ratings and comments.",
        "duration_min": 3,
    },
    {
        "title": "Debrief",
        "kind": "debrief",
        "description": "Thank the participant and provide withdrawal and contact information.",
        "duration_min": 1,
    },
]


def default_study_protocol() -> dict[str, Any]:
    """Return a beginner-friendly blank protocol with a usable procedure outline."""
    return {
        "study_title": "",
        "research_questions": [],
        "study_summary": "",
        "consent_text": "",
        "start_survey_questions": [],
        "end_survey_questions": [],
        "procedure_steps": deepcopy(DEFAULT_PROCEDURE),
    }


def normalize_study_protocol(protocol: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Normalize strings/lists and fill omitted protocol fields."""
    normalized = default_study_protocol()
    if protocol:
        normalized.update(deepcopy(protocol))

    for key in ("research_questions", "start_survey_questions", "end_survey_questions"):
        value = normalized.get(key, [])
        if isinstance(value, str):
            value = [line.strip() for line in value.splitlines() if line.strip()]
        normalized[key] = [str(item).strip() for item in value if str(item).strip()]

    steps = []
    for position, raw_step in enumerate(normalized.get("procedure_steps") or [], start=1):
        step = dict(raw_step) if isinstance(raw_step, dict) else {"title": str(raw_step)}
        step.setdefault("title", f"Step {position}")
        step.setdefault("kind", "information")
        step.setdefault("description", "")
        step.setdefault("duration_min", None)
        step["title"] = str(step["title"]).strip() or f"Step {position}"
        step["kind"] = str(step["kind"]).strip().lower() or "information"
        step["description"] = str(step["description"]).strip()
        steps.append(step)
    normalized["procedure_steps"] = steps
    return normalized


def validate_study_protocol(protocol: dict[str, Any]) -> list[str]:
    """Return plain-language problems that must be resolved before approval."""
    protocol = normalize_study_protocol(protocol)
    problems = []
    if not protocol["study_title"].strip():
        problems.append("Add a study title.")
    if not protocol["research_questions"]:
        problems.append("Add at least one research question.")
    if not protocol["consent_text"].strip():
        problems.append("Add the consent information participants will see.")
    if not protocol["procedure_steps"]:
        problems.append("Add at least one procedure step.")
    if not any(step["kind"] == "trials" for step in protocol["procedure_steps"]):
        problems.append("Mark one procedure step with kind='trials'.")
    return problems


def edit_study_protocol(
    protocol: Optional[dict[str, Any]] = None,
    *,
    on_save: Optional[Callable[[dict[str, Any]], None]] = None,
) -> Any:
    """Display an ipywidgets form and save its values through ``on_save``."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        current = normalize_study_protocol(protocol)
        print(
            "ipywidgets is not installed, so the Python study setup above is being used. "
            "Install the packages in requirements.txt to enable the interactive form."
        )
        return current

    current = normalize_study_protocol(protocol)
    title = widgets.Text(value=current["study_title"], description="Title:", layout=widgets.Layout(width="100%"))
    questions = widgets.Textarea(
        value="\n".join(current["research_questions"]),
        description="RQs:", placeholder="RQ1: ...\nRQ2: ...",
        layout=widgets.Layout(width="100%", height="110px"),
    )
    summary = widgets.Textarea(
        value=current["study_summary"], description="Summary:",
        layout=widgets.Layout(width="100%", height="90px"),
    )
    consent = widgets.Textarea(
        value=current["consent_text"], description="Consent:",
        layout=widgets.Layout(width="100%", height="160px"),
    )
    start_survey = widgets.Textarea(
        value="\n".join(current["start_survey_questions"]),
        description="Start survey:", placeholder="One question per line",
        layout=widgets.Layout(width="100%", height="100px"),
    )
    end_survey = widgets.Textarea(
        value="\n".join(current["end_survey_questions"]),
        description="End survey:", placeholder="One question per line",
        layout=widgets.Layout(width="100%", height="100px"),
    )

    step_rows: list[dict[str, Any]] = []
    steps_box = widgets.VBox()

    def refresh_steps() -> None:
        for index, row in enumerate(step_rows):
            row["up"].disabled = index == 0
            row["down"].disabled = index == len(step_rows) - 1
        steps_box.children = tuple(row["box"] for row in step_rows)

    def add_step_row(step: Optional[dict[str, Any]] = None) -> None:
        value = step or {}
        step_title = widgets.Text(value=str(value.get("title", "New step")), description="Step:")
        kind = widgets.Dropdown(
            options=["consent", "survey", "practice", "trials", "debrief", "information"],
            value=str(value.get("kind", "information")), description="Type:",
        )
        duration = widgets.FloatText(value=float(value.get("duration_min") or 0), description="Minutes:")
        description = widgets.Textarea(
            value=str(value.get("description", "")), description="Details:",
            layout=widgets.Layout(width="100%", height="80px"),
        )
        resource = widgets.Text(
            value=str(value.get("resource", "")), description="File/link:",
            placeholder="Optional local path or URL", layout=widgets.Layout(width="100%"),
        )
        move_up = widgets.Button(description="Up", icon="arrow-up")
        move_down = widgets.Button(description="Down", icon="arrow-down")
        remove = widgets.Button(description="Remove step", icon="trash", button_style="danger")
        box = widgets.VBox([
            widgets.HBox([step_title, kind, duration]), description, resource,
            widgets.HBox([move_up, move_down, remove]),
        ])
        row = {"title": step_title, "kind": kind, "duration": duration,
               "description": description, "resource": resource, "box": box,
               "up": move_up, "down": move_down}
        step_rows.append(row)

        def move_row(delta: int) -> None:
            index = step_rows.index(row)
            new_index = min(max(0, index + delta), len(step_rows) - 1)
            if new_index != index:
                step_rows.insert(new_index, step_rows.pop(index))
                refresh_steps()

        move_up.on_click(lambda _button: move_row(-1))
        move_down.on_click(lambda _button: move_row(1))

        def remove_row(_button: Any) -> None:
            step_rows.remove(row)
            refresh_steps()

        remove.on_click(remove_row)
        refresh_steps()

    for procedure_step in current["procedure_steps"]:
        add_step_row(procedure_step)

    add_step = widgets.Button(description="Add procedure step", icon="plus")
    add_step.on_click(lambda _button: add_step_row())
    save = widgets.Button(description="Save study setup", icon="save", button_style="success")
    status = widgets.HTML()

    def save_values(_button: Any) -> None:
        updated = normalize_study_protocol({
            "study_title": title.value,
            "research_questions": questions.value,
            "study_summary": summary.value,
            "consent_text": consent.value,
            "start_survey_questions": start_survey.value,
            "end_survey_questions": end_survey.value,
            "procedure_steps": [
                {
                    "title": row["title"].value,
                    "kind": row["kind"].value,
                    "duration_min": row["duration"].value,
                    "description": row["description"].value,
                    "resource": row["resource"].value,
                }
                for row in step_rows
            ],
        })
        problems = validate_study_protocol(updated)
        if problems:
            status.value = "<b>Please complete:</b><br>" + "<br>".join(problems)
            return
        if on_save is not None:
            on_save(updated)
        status.value = "<b>Saved.</b> Preview the walkthrough before running the experiment."

    save.on_click(save_values)
    editor = widgets.VBox([
        widgets.HTML("<h3>Study setup</h3><p>Complete these fields in your own words.</p>"),
        title, questions, summary, consent,
        widgets.HTML("<h4>Survey questions</h4><p>Enter one question per line.</p>"),
        start_survey, end_survey,
        widgets.HTML("<h4>Procedure</h4><p>Put the steps in the order participants will experience them.</p>"),
        steps_box, add_step, widgets.HBox([save, status]),
    ])
    display(editor)
    return editor


__all__ = [
    "DEFAULT_PROCEDURE",
    "default_study_protocol",
    "edit_study_protocol",
    "normalize_study_protocol",
    "validate_study_protocol",
]
