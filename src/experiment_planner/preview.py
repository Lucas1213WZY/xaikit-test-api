"""Interactive/HTML preview of a participant's generated trials.

Standalone functions (no orchestrator needed): pass the prepared ``data``, the
generated ``trials``, and the combined explanation table explicitly. Training
trials show the AI prediction; testing trials keep it hidden. XAI-visible trials
show the matching explanation plot.
"""

from __future__ import annotations

import base64
import html
import io
import uuid
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from src.workflow_standard import EXPLANATION_METHOD_COL, INSTANCE_ID_COL, PREDICTION_COL
from .config import select_trial_rows
from .protocol import normalize_study_protocol, validate_study_protocol


def _coerce_bool(value: Any) -> bool:
    """Interpret notebook/CSV boolean-like values consistently."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    text = str(value).strip().lower()
    if text in {"", "0", "false", "n", "nan", "no", "none"}:
        return False
    if text in {"1", "true", "y", "yes"}:
        return True
    return bool(value)


def _show_ai_prediction(trial: dict[str, Any]) -> bool:
    """AI predictions are feedback in training and hidden during testing."""
    return str(trial.get("phase", "")).strip().lower() != "testing"


def _prediction_for_instance(explanation_df: pd.DataFrame, instance_id: int) -> Any:
    """Return the first stored AI prediction for an instance, if available."""
    required_cols = {INSTANCE_ID_COL, PREDICTION_COL}
    if not required_cols <= set(explanation_df.columns):
        return None

    matches = explanation_df[explanation_df[INSTANCE_ID_COL].astype(int) == int(instance_id)]
    if matches.empty:
        return None
    if EXPLANATION_METHOD_COL in matches:
        explanation_matches = matches[
            matches[EXPLANATION_METHOD_COL].astype(str).str.lower() != "none"
        ]
        if not explanation_matches.empty:
            matches = explanation_matches
    prediction = matches[PREDICTION_COL].dropna()
    if prediction.empty:
        return None
    value = prediction.iloc[0]
    try:
        numeric_value = float(value)
        return int(numeric_value) if numeric_value.is_integer() else numeric_value
    except (TypeError, ValueError):
        return value


def _has_explanation(
    explanation_df: pd.DataFrame,
    *,
    instance_id: int,
    method: str,
) -> bool:
    """Whether the pool contains the exact method/instance explanation."""
    required = {INSTANCE_ID_COL, EXPLANATION_METHOD_COL}
    if not required <= set(explanation_df.columns):
        return False
    return bool((
        (explanation_df[INSTANCE_ID_COL].astype(int) == int(instance_id))
        & (
            explanation_df[EXPLANATION_METHOD_COL].astype(str).str.lower()
            == str(method).lower()
        )
    ).any())


def preview_participant_trials(
    data: Any,
    trials: Any,
    combined_df: pd.DataFrame,
    *,
    participant_id: int = 1,
    visualization: str = "importance",
    top_n: int = 5,
    class_labels: Optional[Sequence[str]] = None,
    fallback: str = "auto",
) -> Any:
    """
    Interactively preview one participant's trials with Back/Next controls.

    Trials with `tested_w_xai=False` or a no-XAI method show the raw instance.
    Training trials show prediction feedback; testing trials hide it.
    """
    participant_trials = select_trial_rows(
        pd.DataFrame(trials),
        mode="participant_by_participant",
        participant_id=participant_id,
    ).reset_index(drop=True)
    if participant_trials.empty:
        raise ValueError(f"No trials found for participant_id={participant_id}.")

    fallback = fallback.lower().strip()
    if fallback not in {"auto", "widgets", "html"}:
        raise ValueError("fallback must be one of: 'auto', 'widgets', or 'html'.")
    if fallback == "html":
        return _display_participant_preview_html(
            participant_trials, combined_df, data,
            participant_id=participant_id, visualization=visualization,
            top_n=top_n, class_labels=class_labels,
        )

    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError:
        if fallback == "widgets":
            print("Interactive preview requires `ipywidgets` and IPython display support.")
            return participant_trials
        return _display_participant_preview_html(
            participant_trials, combined_df, data,
            participant_id=participant_id, visualization=visualization,
            top_n=top_n, class_labels=class_labels,
        )

    print("Use Back/Next to preview this participant's trials.")
    print("If widgets do not render in this notebook, call with `fallback='html'` after updating.")

    state = {"idx": 0}
    output = widgets.Output()
    back_button = widgets.Button(description="Back", icon="arrow-left")
    next_button = widgets.Button(description="Next", icon="arrow-right")
    label = widgets.HTML()

    def render() -> None:
        with output:
            clear_output(wait=True)
            trial = participant_trials.iloc[state["idx"]].to_dict()
            label.value = (
                f"<b>Participant {participant_id}</b> | "
                f"Trial {state['idx'] + 1} of {len(participant_trials)}"
            )
            _display_trial_preview(
                trial, combined_df, data,
                visualization=visualization, top_n=top_n, class_labels=class_labels,
            )
        back_button.disabled = state["idx"] == 0
        next_button.disabled = state["idx"] == len(participant_trials) - 1

    def go_back(_button: Any) -> None:
        state["idx"] = max(0, state["idx"] - 1)
        render()

    def go_next(_button: Any) -> None:
        state["idx"] = min(len(participant_trials) - 1, state["idx"] + 1)
        render()

    back_button.on_click(go_back)
    next_button.on_click(go_next)
    controls = widgets.HBox([back_button, next_button, label])
    render()
    display(widgets.VBox([controls, output]))
    return participant_trials


def build_walkthrough_pages(
    protocol: dict[str, Any],
    participant_trials: pd.DataFrame,
    *,
    max_trials: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Build ordered researcher-review, procedure, and trial preview pages."""
    protocol = normalize_study_protocol(protocol)
    trial_rows = participant_trials
    if max_trials is not None:
        if max_trials < 1:
            raise ValueError("max_trials must be at least 1 when provided.")
        trial_rows = trial_rows.head(max_trials)

    pages: list[dict[str, Any]] = [{
        "page_type": "researcher_review",
        "title": "Researcher review",
        "description": protocol.get("study_summary", ""),
        "research_questions": protocol.get("research_questions", []),
    }]
    for step_number, step in enumerate(protocol["procedure_steps"], start=1):
        if step["kind"] == "trials":
            for trial_number, (_, trial) in enumerate(trial_rows.iterrows(), start=1):
                pages.append({
                    "page_type": "trial",
                    "title": f"{step['title']}: question {trial_number}",
                    "step_number": step_number,
                    "trial": trial.to_dict(),
                })
            continue

        page = {
            "page_type": step["kind"],
            "title": step["title"],
            "description": step.get("description", ""),
            "duration_min": step.get("duration_min"),
            "resource": step.get("resource", ""),
            "step_number": step_number,
        }
        if step["kind"] == "consent":
            page["content"] = protocol.get("consent_text", "")
        elif step["kind"] == "survey":
            use_start = not any(p.get("page_type") == "survey" for p in pages)
            page["questions"] = protocol[
                "start_survey_questions" if use_start else "end_survey_questions"
            ]
        pages.append(page)
    return pages


def preview_experiment_walkthrough(
    protocol: dict[str, Any],
    data: Any,
    trials: Any,
    combined_df: pd.DataFrame,
    *,
    participant_id: int = 1,
    visualization: str = "importance",
    top_n: int = 5,
    class_labels: Optional[Sequence[str]] = None,
    max_trials: Optional[int] = None,
    on_approve: Optional[Callable[[], None]] = None,
    fallback: str = "auto",
) -> list[dict[str, Any]]:
    """Preview the complete session and require an explicit final approval click."""
    problems = validate_study_protocol(protocol)
    if problems:
        raise ValueError("Study setup is incomplete: " + " ".join(problems))

    participant_trials = select_trial_rows(
        pd.DataFrame(trials), mode="participant_by_participant", participant_id=participant_id,
    ).reset_index(drop=True)
    if participant_trials.empty:
        raise ValueError(f"No trials found for participant_id={participant_id}.")
    pages = build_walkthrough_pages(protocol, participant_trials, max_trials=max_trials)

    try:
        import ipywidgets as widgets
        from IPython.display import HTML, clear_output, display
    except ImportError:
        from IPython.display import HTML, display
        if fallback == "widgets":
            raise RuntimeError("Walkthrough widgets require ipywidgets and IPython.")
        print("Interactive widgets are unavailable; showing the procedure and trial preview separately.")
        for page in pages:
            if page["page_type"] != "trial":
                display(HTML(_walkthrough_page_html(page)))
        preview_participant_trials(
            data, participant_trials, combined_df, participant_id=participant_id,
            visualization=visualization, top_n=top_n, class_labels=class_labels,
            fallback="html",
        )
        print("Approval was not recorded. Call study.approve_walkthrough() after reviewing the HTML preview.")
        return pages

    state = {"index": 0}
    output = widgets.Output()
    back = widgets.Button(description="Back", icon="arrow-left")
    next_button = widgets.Button(description="Next", icon="arrow-right")
    progress = widgets.HTML()
    approval = widgets.Checkbox(
        value=False,
        description="I reviewed the setup and participant-facing walkthrough.",
        indent=False,
        layout=widgets.Layout(display="none", width="auto"),
    )
    approve = widgets.Button(
        description="Approve walkthrough", icon="check", button_style="success",
        disabled=True, layout=widgets.Layout(display="none"),
    )
    approval_status = widgets.HTML()

    def render() -> None:
        page = pages[state["index"]]
        with output:
            clear_output(wait=True)
            if page["page_type"] == "trial":
                display(HTML(f"<h3>{html.escape(page['title'])}</h3>"))
                _display_trial_preview(
                    page["trial"], combined_df, data,
                    visualization=visualization, top_n=top_n, class_labels=class_labels,
                )
            else:
                display(HTML(_walkthrough_page_html(page)))
        progress.value = f"<b>Page {state['index'] + 1} of {len(pages)}</b>"
        back.disabled = state["index"] == 0
        next_button.disabled = state["index"] == len(pages) - 1
        at_end = state["index"] == len(pages) - 1
        approval.layout.display = "flex" if at_end else "none"
        approve.layout.display = "inline-flex" if at_end else "none"

    def move(delta: int) -> None:
        state["index"] = min(max(0, state["index"] + delta), len(pages) - 1)
        render()

    back.on_click(lambda _button: move(-1))
    next_button.on_click(lambda _button: move(1))
    approval.observe(lambda change: setattr(approve, "disabled", not bool(change["new"])), names="value")

    def record_approval(_button: Any) -> None:
        if not approval.value:
            return
        if on_approve is not None:
            on_approve()
        approval_status.value = "<b>Approved.</b> The experiment execution cell is now unlocked."
        approve.disabled = True

    approve.on_click(record_approval)
    render()
    display(widgets.VBox([
        widgets.HTML("<h2>Experiment walkthrough</h2>"),
        widgets.HBox([back, next_button, progress]), output,
        approval, widgets.HBox([approve, approval_status]),
    ]))
    return pages


def _walkthrough_page_html(page: dict[str, Any]) -> str:
    """Render one non-trial walkthrough page."""
    title = html.escape(str(page.get("title", "Study step")))
    description = html.escape(str(page.get("description", ""))).replace("\n", "<br>")
    content = html.escape(str(page.get("content", ""))).replace("\n", "<br>")
    resource = str(page.get("resource", "")).strip()
    resource_html = f"<p><b>File/link:</b> {html.escape(resource)}</p>" if resource else ""
    questions = page.get("questions", [])
    question_html = ""
    if questions:
        question_html = "<h4>Questions shown</h4><ol>" + "".join(
            f"<li>{html.escape(str(question))}</li>" for question in questions
        ) + "</ol>"
    research_questions = page.get("research_questions", [])
    rq_html = ""
    if research_questions:
        rq_html = "<h4>Research questions</h4><ol>" + "".join(
            f"<li>{html.escape(str(question))}</li>" for question in research_questions
        ) + "</ol>"
    duration = page.get("duration_min")
    duration_html = f"<p><b>Planned duration:</b> {duration} minutes</p>" if duration else ""
    return (
        f"<div style='font-family: sans-serif; border: 1px solid #d1d5db; padding: 16px; border-radius: 8px'>"
        f"<h3>{title}</h3><p>{description}</p>{duration_html}{content}{rq_html}{question_html}{resource_html}</div>"
    )


def _display_participant_preview_html(
    participant_trials: pd.DataFrame,
    explanation_df: pd.DataFrame,
    data: Any,
    *,
    participant_id: int,
    visualization: str,
    top_n: int,
    class_labels: Optional[Sequence[str]],
) -> Any:
    """Display an HTML Back/Next preview when ipywidgets is unavailable."""
    try:
        from IPython.display import HTML, display
    except ImportError:
        print("Interactive preview requires IPython display support.")
        return participant_trials

    preview_id = f"xaikit-preview-{uuid.uuid4().hex}"
    slides = [
        _trial_preview_html(
            trial.to_dict(), explanation_df, data,
            slide_number=idx + 1, slide_count=len(participant_trials),
            visualization=visualization, top_n=top_n, class_labels=class_labels,
        )
        for idx, trial in participant_trials.iterrows()
    ]
    slides_html = "\n".join(
        f'<div class="xaikit-slide" data-index="{idx}" style="display: {"block" if idx == 0 else "none"};">'
        f"{slide}</div>"
        for idx, slide in enumerate(slides)
    )
    html_doc = f"""
    <div id="{preview_id}" class="xaikit-participant-preview">
      <style>
        #{preview_id} {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: #111827;
        }}
        #{preview_id} .xaikit-controls {{
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 8px 0 12px;
        }}
        #{preview_id} button {{
          border: 1px solid #9ca3af;
          background: #ffffff;
          color: #111827;
          padding: 5px 10px;
          border-radius: 4px;
          cursor: pointer;
        }}
        #{preview_id} button:disabled {{
          color: #9ca3af;
          cursor: default;
        }}
        #{preview_id} .xaikit-count {{
          font-weight: 600;
          margin-left: 4px;
        }}
        #{preview_id} table {{
          border-collapse: collapse;
          margin: 6px 0 12px;
          font-size: 13px;
        }}
        #{preview_id} th, #{preview_id} td {{
          border: 1px solid #d1d5db;
          padding: 5px 8px;
          text-align: left;
        }}
        #{preview_id} th {{
          background: #f3f4f6;
        }}
        #{preview_id} img {{
          max-width: 100%;
          height: auto;
        }}
        #{preview_id} .xaikit-note {{
          margin: 8px 0;
          color: #374151;
        }}
      </style>
      <div class="xaikit-controls">
        <button type="button" class="xaikit-back">Back</button>
        <button type="button" class="xaikit-next">Next</button>
        <span class="xaikit-count"></span>
      </div>
      {slides_html}
      <script>
        (function() {{
          const root = document.getElementById("{preview_id}");
          const slides = Array.from(root.querySelectorAll(".xaikit-slide"));
          const back = root.querySelector(".xaikit-back");
          const next = root.querySelector(".xaikit-next");
          const count = root.querySelector(".xaikit-count");
          let index = 0;
          function render() {{
            slides.forEach((slide, i) => {{
              slide.style.display = i === index ? "block" : "none";
            }});
            back.disabled = index === 0;
            next.disabled = index === slides.length - 1;
            count.textContent = "Participant {participant_id} | Trial " + (index + 1) + " of " + slides.length;
          }}
          back.addEventListener("click", function() {{
            index = Math.max(0, index - 1);
            render();
          }});
          next.addEventListener("click", function() {{
            index = Math.min(slides.length - 1, index + 1);
            render();
          }});
          render();
        }})();
      </script>
    </div>
    """
    print("ipywidgets unavailable. Showing HTML Back/Next preview.")
    display(HTML(html_doc))
    return participant_trials


def _trial_preview_html(
    trial: dict[str, Any],
    explanation_df: pd.DataFrame,
    data: Any,
    *,
    slide_number: int,
    slide_count: int,
    visualization: str,
    top_n: int,
    class_labels: Optional[Sequence[str]],
) -> str:
    """Render one trial preview as HTML for widget-free notebooks."""
    instance_id = int(trial["instanceId"])
    xai_method = str(trial.get("xai_method", trial.get("xai_type", "none"))).lower()
    tested_w_xai = _coerce_bool(
        trial.get("tested_w_xai", xai_method not in {"none", "no_xai", "control"})
    )
    is_training = str(trial.get("phase", "")).lower() == "training"
    show_ai_prediction = _show_ai_prediction(trial)
    has_xai = (
        (is_training or tested_w_xai)
        and xai_method not in {"none", "no_xai", "control"}
    )
    explanation_available = _has_explanation(
        explanation_df,
        instance_id=instance_id,
        method=xai_method,
    )

    summary_values = {
        "participantId": trial.get("participantId"),
        "trialId": trial.get("trialId"),
        "phase": trial.get("phase", "testing"),
        "instanceId": instance_id,
        "xai_method": xai_method,
        "tested_w_xai": tested_w_xai,
    }
    if show_ai_prediction:
        summary_values["ai_prediction"] = _prediction_for_instance(
            explanation_df, instance_id
        )
    for key in (
        "session", "phase_position", "task", "validation_condition",
        "explanation_property", "complexity", "shown_xai_type",
    ):
        if key in trial:
            summary_values[key] = trial.get(key)
    summary = pd.DataFrame([summary_values]).to_html(index=False, border=0)

    body = ""
    if has_xai and explanation_available:
        try:
            from src.xai_adapter import plot_explanation_visual
            import matplotlib.pyplot as plt

            fig, _axes = plot_explanation_visual(
                explanation_df, data,
                visualization=visualization, method=xai_method, instance_id=instance_id,
                feature_names=data.raw_feature_names, top_n=top_n,
                class_labels=list(class_labels) if class_labels is not None else None,
                show_ai_prediction=show_ai_prediction,
            )
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            image = base64.b64encode(buffer.getvalue()).decode("ascii")
            body = f'<img alt="Trial {slide_number} explanation" src="data:image/png;base64,{image}">'
        except Exception as exc:
            body = f'<p class="xaikit-note">Explanation rendering failed: {html.escape(str(exc))}</p>'

    if not body:
        row = data.df.iloc[instance_id].drop(labels=[data.label_column], errors="ignore")
        raw_values = row.to_frame().T.to_html(index=False, border=0)
        if has_xai and not explanation_available:
            note = (
                "This training exemplar is outside the generated test-explanation pool."
                if str(trial.get("phase", "")).lower() == "training"
                else "No generated explanation matched this method and instance."
            )
            note += (
                " Showing its raw attributes and AI prediction."
                if show_ai_prediction
                else " Showing its raw attributes."
            )
        else:
            note = "No XAI shown for this trial."
        body = f'<p class="xaikit-note">{note}</p>' + raw_values

    return (
        f'<div class="xaikit-note">Trial {slide_number} of {slide_count}</div>'
        f"{summary}"
        f"{body}"
    )


def _display_trial_preview(
    trial: dict[str, Any],
    explanation_df: pd.DataFrame,
    data: Any,
    *,
    visualization: str,
    top_n: int,
    class_labels: Optional[Sequence[str]],
) -> None:
    """Display one trial as either an explanation plot or no-XAI instance preview."""
    from IPython.display import display

    instance_id = int(trial["instanceId"])
    xai_method = str(trial.get("xai_method", trial.get("xai_type", "none"))).lower()
    tested_w_xai = _coerce_bool(
        trial.get("tested_w_xai", xai_method not in {"none", "no_xai", "control"})
    )
    is_training = str(trial.get("phase", "")).lower() == "training"
    show_ai_prediction = _show_ai_prediction(trial)
    has_xai = (
        (is_training or tested_w_xai)
        and xai_method not in {"none", "no_xai", "control"}
    )
    explanation_available = _has_explanation(
        explanation_df,
        instance_id=instance_id,
        method=xai_method,
    )

    summary = {
        "participantId": trial.get("participantId"),
        "trialId": trial.get("trialId"),
        "phase": trial.get("phase", "testing"),
        "instanceId": instance_id,
        "xai_method": xai_method,
        "tested_w_xai": tested_w_xai,
    }
    if show_ai_prediction:
        summary["ai_prediction"] = _prediction_for_instance(
            explanation_df, instance_id
        )
    for key in (
        "session", "phase_position", "task", "validation_condition",
        "explanation_property", "complexity", "shown_xai_type",
    ):
        if key in trial:
            summary[key] = trial.get(key)
    display(pd.DataFrame([summary]))

    if has_xai and explanation_available:
        from src.xai_adapter import plot_explanation_visual

        try:
            fig, _axes = plot_explanation_visual(
                explanation_df, data,
                visualization=visualization, method=xai_method, instance_id=instance_id,
                feature_names=data.raw_feature_names, top_n=top_n,
                class_labels=list(class_labels) if class_labels is not None else None,
                show_ai_prediction=show_ai_prediction,
            )
            display(fig)
            return
        except ValueError as exc:
            print(f"No generated explanation found for this trial: {exc}")

    if has_xai and not explanation_available:
        reason = (
            "This training exemplar is outside the generated test-explanation pool."
            if str(trial.get("phase", "")).lower() == "training"
            else "No generated explanation matched this method and instance."
        )
        suffix = (
            " Showing raw attributes and its AI prediction."
            if show_ai_prediction
            else " Showing raw attributes."
        )
        print(f"{reason}{suffix}")
    else:
        print("No XAI shown for this trial.")
    row = data.df.iloc[instance_id].drop(labels=[data.label_column], errors="ignore")
    display(row.to_frame().T)
