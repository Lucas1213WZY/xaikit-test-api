"""Render each study's human-vs-model comparison as one lettered PNG.

Reads ``assets/human_vs_model_plot_data.json`` and writes one PNG per study
under ``assets/human_vs_model_plots/``, so the server can hand a finished
image straight to the UI with no per-request matplotlib work.

Run ``assets/build_human_vs_model_plot_data.py`` first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "assets" / "human_vs_model_plot_data.json"
OUTPUT_DIR = REPO_ROOT / "assets" / "human_vs_model_plots"


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")

    from src.result_visualizer import render_panel_grid_png

    if not DATA.is_file():
        raise FileNotFoundError(
            f"{DATA.relative_to(REPO_ROOT)} not found. "
            "Run assets/build_human_vs_model_plot_data.py first."
        )
    payload = json.loads(DATA.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for study in payload["studies"]:
        slug = study["name"].lower().replace(" ", "_")
        figure = render_panel_grid_png(
            study["panels"], title=f"Human vs {study['name']}"
        )
        out_path = OUTPUT_DIR / f"{slug}.png"
        figure.savefig(out_path, dpi=144, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(figure)
        print(f"  {out_path.relative_to(REPO_ROOT)}  {out_path.stat().st_size:,} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
