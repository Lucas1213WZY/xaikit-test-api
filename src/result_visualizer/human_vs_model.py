"""Human-vs-cognitive-model comparison panels and their interactive report.

``plot_iv_dv_grid`` answers "what did the simulation do?". This module answers
"did it do what people did?" -- the comparison you show as evidence that a
cognitive model reproduces participants rather than merely running.

The estimator is deliberately the **mean over participant means**, not over
trials, so a participant who answered more trials does not count for more; the
error bar is a 95% confidence interval on those means by default. Passing a frame whose
rows are already one-per-participant gives the same answer, so the same function
serves both trial-level and participant-level input.

Nothing here is study-specific: a panel is built from a frame, a participant
column, a grouping column, and one column per series. :func:`render_comparison_report`
writes a single self-contained HTML file -- no network access, no build step --
which is what a UI can serve directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .intervals import ci95_multiplier

__all__ = [
    "ComparisonPanel",
    "ComparisonStudy",
    "comparison_panel",
    "participant_summary",
    "render_comparison_report",
]


@dataclass
class ComparisonPanel:
    """One grouped-bar panel: a category axis with one bar per series."""

    title: str
    dv: str
    categories: list[str]
    series: list[dict[str, Any]]
    note: str = ""
    #: How the ``error`` half-widths were computed, for the reader.
    interval: str = "95% CI"

    def to_frame(self) -> pd.DataFrame:
        """The panel as a tidy table -- the numbers behind the bars."""
        rows = []
        for entry in self.series:
            for index, category in enumerate(self.categories):
                rows.append(
                    {
                        "series": entry["name"],
                        "category": category,
                        "mean": entry["values"][index],
                        "error": entry["error"][index],
                        "interval": self.interval,
                        "n": entry["n"][index],
                    }
                )
        return pd.DataFrame(rows)


@dataclass
class ComparisonStudy:
    """A named study and its panels."""

    name: str
    task: str = ""
    participants: Optional[int] = None
    panels: list[ComparisonPanel] = field(default_factory=list)


#: Error-bar conventions. ``ci95`` is the half-width of a 95% confidence
#: interval on the mean; ``sem`` is the standard error itself.
INTERVALS = ("ci95", "sem")

#: How each convention is described to a reader.
INTERVAL_LABELS = {"ci95": "95% CI", "sem": "SEM"}


def participant_summary(
    frame: pd.DataFrame,
    *,
    participant_column: str,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Mean over participant means per group, with its SEM and 95% CI half-width.

    A single-participant group gets ``sem = 0`` and ``ci95 = 0``: one
    observation has no spread, and reporting ``NaN`` would drop the bar rather
    than flatten its whisker.
    """
    for column in (participant_column, group_column, value_column):
        if column not in frame.columns:
            raise KeyError(f"Column {column!r} is not in the frame.")

    per_participant = (
        frame.dropna(subset=[value_column])
        .groupby([group_column, participant_column])[value_column]
        .mean()
        .reset_index()
    )
    rows = []
    for key, chunk in per_participant.groupby(group_column, sort=False):
        values = chunk[value_column].to_numpy(dtype=float)
        if values.size == 0:
            continue
        sem = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        rows.append(
            {
                "group": key,
                "mean": float(values.mean()),
                "sem": sem,
                "ci95": sem * ci95_multiplier(values.size),
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows, columns=["group", "mean", "sem", "ci95", "n"])


def comparison_panel(
    frame: pd.DataFrame,
    *,
    participant_column: str,
    group_column: str,
    series: Mapping[str, str],
    title: str,
    dv: str,
    note: str = "",
    order: Optional[Sequence[Any]] = None,
    group_labels: Optional[Mapping[Any, str]] = None,
    interval: str = "ci95",
) -> ComparisonPanel:
    """Build one panel comparing several measures over the same groups.

    Args:
        frame: One row per trial (or per participant).
        participant_column: Column identifying participants.
        group_column: Column whose values become the category axis.
        series: ``{display name: column}``, e.g.
            ``{"Human": "Response==AI", "CoXAM": "Model==AI"}``. Order is kept,
            so the first series takes the first categorical colour.
        title: Panel heading.
        dv: What the numbers measure; becomes the y-axis label.
        note: Caveat shown under the heading.
        order: Category order; defaults to sorted.
        group_labels: Display labels for category values.
        interval: ``"ci95"`` (default) for a 95% confidence interval on the
            mean, or ``"sem"`` for the standard error itself.

    Raises:
        KeyError: If a named column is missing.
        ValueError: If ``interval`` is not one of :data:`INTERVALS`.
    """
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {INTERVALS}, not {interval!r}.")
    summaries = {
        name: {row["group"]: row for _, row in participant_summary(
            frame,
            participant_column=participant_column,
            group_column=group_column,
            value_column=column,
        ).iterrows()}
        for name, column in series.items()
    }
    keys = list(order) if order is not None else sorted(
        {key for summary in summaries.values() for key in summary}
    )
    labels = dict(group_labels or {})
    return ComparisonPanel(
        title=title,
        dv=dv,
        note=note,
        interval=INTERVAL_LABELS[interval],
        categories=[str(labels.get(key, key)) for key in keys],
        series=[
            {
                "name": name,
                "values": [summaries[name].get(key, {}).get("mean") for key in keys],
                "error": [summaries[name].get(key, {}).get(interval) for key in keys],
                "n": [summaries[name].get(key, {}).get("n") for key in keys],
            }
            for name in series
        ],
    )


def _payload(studies: Sequence[ComparisonStudy], estimator: str) -> dict[str, Any]:
    return {
        "estimator": estimator,
        "studies": [
            {
                "name": study.name,
                "task": study.task,
                "participants": study.participants,
                "panels": [asdict(panel) for panel in study.panels],
            }
            for study in studies
        ],
    }


def render_comparison_report(
    studies: Sequence[ComparisonStudy],
    output_path: Path | str,
    *,
    title: str = "Human vs cognitive model",
    lede: Optional[str] = None,
    footer: str = "",
) -> Path:
    """Write the panels to one self-contained interactive HTML file.

    The data is inlined, so the page needs no network access and no build step.

    Args:
        studies: Studies to render, in order.
        output_path: Where to write the ``.html``.
        title: Page heading and ``<title>``.
        lede: Intro paragraph; a default explaining the estimator is used if None.
        footer: Provenance line shown at the bottom.

    Returns:
        The path written.
    """
    if not studies:
        raise ValueError("Pass at least one study to render.")

    estimator = "mean of participant means; error bars are 95% CI on those means"
    payload = _payload(studies, estimator)
    default_lede = (
        "Each bar is the mean over <em>participant</em> means, so nobody counts for more by "
        "answering more trials; whiskers are 95% confidence intervals on those means. Panels name "
        "their own dependent variable rather than sharing one scale. The dashed line is chance."
    )
    # Split the closing tag so the JSON can never terminate the <script> early.
    inlined = json.dumps(payload).replace("</", "<\\/")
    html = (
        TEMPLATE.replace("__TITLE__", title)
        .replace("__LEDE__", lede if lede is not None else default_lede)
        .replace("__FOOTER__", footer)
        .replace("__DATA__", inlined)
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --page: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --series-5: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
    --series-5: #d55181;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19;
  --page: #0d0d0d;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --baseline: #383835;
  --border: rgba(255,255,255,0.10);
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --series-4: #c98500;
  --series-5: #d55181;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.5;
  padding: 32px 20px 64px;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 6px; letter-spacing: -0.01em; }
.lede { color: var(--text-secondary); margin: 0 0 28px; max-width: 68ch; font-size: 0.94rem; }
h2 { font-size: 1.12rem; margin: 0 0 2px; }
.study-task { color: var(--text-secondary); font-size: 0.88rem; margin: 0 0 16px; }
.study {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 22px;
}
.panel { margin-top: 22px; }
.panel:first-of-type { margin-top: 8px; }
.panel-head { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: baseline; margin-bottom: 2px; }
.panel-title { font-weight: 600; font-size: 0.97rem; }
.panel-dv { color: var(--text-secondary); font-size: 0.83rem; }
.panel-note { color: var(--muted); font-size: 0.8rem; margin: 0 0 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; font-size: 0.82rem; color: var(--text-secondary); }
.swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; }
.chart-scroll { overflow-x: auto; }
svg { display: block; max-width: 100%; height: auto; }
.axis-label, .tick, .value-label { font-size: 11px; fill: var(--muted); }
.value-label { fill: var(--text-secondary); font-size: 10px; font-variant-numeric: tabular-nums; }
.cat-label { font-size: 11.5px; fill: var(--text-secondary); }
.bar { transition: opacity .12s; cursor: pointer; }
.bar:hover { opacity: .78; }
.controls { display: flex; gap: 8px; margin: 10px 0 0; }
button {
  font: inherit; font-size: 0.8rem; padding: 4px 11px; cursor: pointer;
  background: var(--surface-1); color: var(--text-secondary);
  border: 1px solid var(--border); border-radius: 999px;
}
button:hover { color: var(--text-primary); }
table { border-collapse: collapse; width: 100%; font-size: 0.82rem; margin-top: 10px; }
th, td { text-align: right; padding: 5px 9px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
th { color: var(--text-secondary); font-weight: 600; }
.hidden { display: none; }
#tip {
  position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
  background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 7px 10px; font-size: 0.79rem; line-height: 1.45;
  box-shadow: 0 4px 14px rgba(0,0,0,.14); z-index: 20; max-width: 240px;
}
#tip b { font-weight: 600; }
#tip .muted { color: var(--text-secondary); }
footer { color: var(--muted); font-size: 0.8rem; margin-top: 26px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="lede">__LEDE__</p>
  <div id="report"></div>
  <footer>__FOOTER__</footer>
</div>
<div id="tip" role="status" aria-live="polite"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const COLORS = ['var(--series-1)','var(--series-2)','var(--series-3)','var(--series-4)','var(--series-5)'];
const tip = document.getElementById('tip');
const fmt = v => v == null ? '—' : v.toFixed(3);
const pct = v => v == null ? '—' : (v * 100).toFixed(1) + '%';

function showTip(evt, html) {
  tip.innerHTML = html;
  tip.style.opacity = '1';
  const pad = 14, r = tip.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = x + 'px';
  tip.style.top = y + 'px';
}
const hideTip = () => { tip.style.opacity = '0'; };

function svgEl(name, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  return el;
}

function renderPanel(panel) {
  const cats = panel.categories, series = panel.series;
  // Left margin holds the rotated axis title *and* the tick labels; too small
  // and the two collide.
  const M = { t: 16, r: 14, b: 44, l: 62 };
  // Every panel is drawn at the same width so the sections line up, with the
  // group centred in its slot rather than bars stretching to fill it.
  const TARGET_W = 880;
  const minGroup = series.length * 34 + 30;
  const groupW = Math.max(minGroup, (TARGET_W - M.l - M.r) / cats.length);
  const W = M.l + M.r + cats.length * groupW;
  const H = 260, plotH = H - M.t - M.b;
  const y = v => M.t + plotH * (1 - v);

  const svg = svgEl('svg', {
    viewBox: `0 0 ${W} ${H}`, width: W, height: H,
    role: 'img', 'aria-label': `${panel.title}. ${panel.dv}.`
  });

  // gridlines + y ticks
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    svg.appendChild(svgEl('line', {
      x1: M.l, x2: W - M.r, y1: y(t), y2: y(t),
      stroke: t === 0 ? 'var(--baseline)' : 'var(--grid)', 'stroke-width': 1
    }));
    const lbl = svgEl('text', { x: M.l - 7, y: y(t) + 3.5, 'text-anchor': 'end', class: 'tick' });
    lbl.textContent = t.toFixed(2);
    svg.appendChild(lbl);
  }
  // chance reference
  svg.appendChild(svgEl('line', {
    x1: M.l, x2: W - M.r, y1: y(0.5), y2: y(0.5),
    stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': '4 4', opacity: 0.85
  }));

  const barW = Math.min(30, (groupW - 30) / series.length) - 2; // 2px surface gap
  const bandW = series.length * (barW + 2) - 2;
  cats.forEach((cat, ci) => {
    const gx = M.l + ci * groupW + (groupW - bandW) / 2;
    series.forEach((s, si) => {
      const v = s.values[ci];
      const x = gx + si * (barW + 2);
      if (v != null) {
        const h = Math.max(1, plotH * v);
        const bar = svgEl('rect', {
          x, y: y(v), width: barW, height: h, rx: 4, ry: 4,
          fill: COLORS[si % COLORS.length], class: 'bar'
        });
        const info = `<b>${s.name}</b> · ${cat}<br>` +
          `${panel.dv}: <b>${pct(v)}</b><br>` +
          `<span class="muted">${panel.interval || '95% CI'} ±${fmt(s.error[ci])} · n = ${s.n[ci] ?? '—'}</span>`;
        bar.addEventListener('mousemove', e => showTip(e, info));
        bar.addEventListener('mouseleave', hideTip);
        svg.appendChild(bar);

        // error bar
        const sem = s.error[ci];
        if (sem) {
          const lo = Math.max(0, v - sem), hi = Math.min(1, v + sem), cx = x + barW / 2;
          svg.appendChild(svgEl('line', {
            x1: cx, x2: cx, y1: y(lo), y2: y(hi),
            stroke: 'var(--text-secondary)', 'stroke-width': 1.5, opacity: .75
          }));
          for (const b of [lo, hi]) svg.appendChild(svgEl('line', {
            x1: cx - 3, x2: cx + 3, y1: y(b), y2: y(b),
            stroke: 'var(--text-secondary)', 'stroke-width': 1.5, opacity: .75
          }));
        }
        // direct label -- required: three light-mode slots are under 3:1.
        // Sits above the whisker, not the bar, or a wide interval collides with it.
        const top = sem ? Math.min(1, v + sem) : v;
        const t = svgEl('text', {
          x: x + barW / 2, y: y(top) - 5, 'text-anchor': 'middle', class: 'value-label'
        });
        t.textContent = v.toFixed(2);
        svg.appendChild(t);
      }
    });
    const cl = svgEl('text', {
      x: gx + bandW / 2, y: H - M.b + 17,
      'text-anchor': 'middle', class: 'cat-label'
    });
    cl.textContent = cat;
    svg.appendChild(cl);
  });

  // y axis title
  const yt = svgEl('text', { x: 0, y: 0, class: 'axis-label', transform: `translate(14 ${M.t + plotH / 2}) rotate(-90)`, 'text-anchor': 'middle' });
  yt.textContent = panel.dv;
  svg.appendChild(yt);
  return svg;
}

function renderTable(panel) {
  const t = document.createElement('table');
  t.className = 'hidden';
  const head = document.createElement('tr');
  head.innerHTML = '<th>Series</th>' + panel.categories.map(c => `<th>${c}</th>`).join('');
  t.appendChild(head);
  for (const s of panel.series) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${s.name}</td>` + s.values.map((v, i) =>
      `<td>${pct(v)}<br><span style="color:var(--muted);font-size:.75rem">±${fmt(s.error[i])} · n=${s.n[i] ?? '—'}</span></td>`
    ).join('');
    t.appendChild(tr);
  }
  return t;
}

const report = document.getElementById('report');
for (const study of DATA.studies) {
  const sec = document.createElement('section');
  sec.className = 'study';
  sec.innerHTML = `<h2>${study.name}</h2><p class="study-task">${study.task} · ${study.participants} participants</p>`;
  for (const panel of study.panels) {
    const box = document.createElement('div');
    box.className = 'panel';
    box.innerHTML =
      `<div class="panel-head"><span class="panel-title">${panel.title}</span>` +
      `<span class="panel-dv">${panel.dv}</span></div>` +
      `<p class="panel-note">${panel.note}</p>`;
    if (panel.series.length > 1) {
      const leg = document.createElement('div');
      leg.className = 'legend';
      leg.innerHTML = panel.series.map((s, i) =>
        `<span><i class="swatch" style="background:${COLORS[i % COLORS.length]}"></i>${s.name}</span>`
      ).join('');
      box.appendChild(leg);
    }
    const scroll = document.createElement('div');
    scroll.className = 'chart-scroll';
    scroll.appendChild(renderPanel(panel));
    box.appendChild(scroll);
    const table = renderTable(panel);
    const controls = document.createElement('div');
    controls.className = 'controls';
    const btn = document.createElement('button');
    btn.textContent = 'Show table';
    btn.setAttribute('aria-expanded', 'false');
    btn.onclick = () => {
      const shown = table.classList.toggle('hidden');
      btn.textContent = shown ? 'Show table' : 'Hide table';
      btn.setAttribute('aria-expanded', String(!shown));
    };
    controls.appendChild(btn);
    box.appendChild(controls);
    box.appendChild(table);
    sec.appendChild(box);
  }
  report.appendChild(sec);
}
</script>
</body>
</html>
"""
