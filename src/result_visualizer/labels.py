"""Display-label prettification shared by plots and stats tables.

Both the matplotlib grid plots and the analysis/post-hoc JSON payloads need
the same "decision_tree" -> "Rules" / "xai_type" -> "XAI Type" vocabulary, so
it lives here once rather than being redefined per consumer.
"""

from __future__ import annotations

from typing import Any

#: Value -> display label overrides recognized automatically, checked before
#: generic prettification. CoXAM's conditions get its own vocabulary since
#: "Rules"/"Weights" aren't a mechanical transform of "decision_tree"/
#: "logistic_regression" -- title-casing alone cannot produce them.
KNOWN_LABELS: dict[str, str] = {
    "decision_tree": "Rules",
    "logistic_regression": "Weights",
    "hybrid": "Hybrid",
}

#: Words ``.title()`` mangles because they're acronyms, not ordinary words
#: (``"xai"`` -> ``"Xai"`` instead of ``"XAI"``). Matched whole-word, case
#: insensitively, after title-casing.
ACRONYMS = {"xai", "ai", "dv", "iv", "id"}


def pretty(text: Any) -> str:
    """A snake_case internal name, in the caption-case a UI should show.

    ``"counterfactual_accuracy"`` -> ``"Counterfactual Accuracy"``. Recognized
    values (see ``KNOWN_LABELS``) are swapped for their own display word
    instead, since prettifying ``"decision_tree"`` can never produce
    ``"Rules"``.
    """
    key = str(text).strip().lower()
    if key in KNOWN_LABELS:
        return KNOWN_LABELS[key]
    words = str(text).replace("_", " ").strip().title().split(" ")
    return " ".join(word.upper() if word.lower() in ACRONYMS else word for word in words)


def prettify_condition_label(raw: str) -> str:
    """Prettify a ``pairwise_condition_tests`` cell label.

    ``posthoc.py``'s ``_condition_label`` joins crossed condition columns as
    ``"xai_type=decision_tree | tested_w_xai=True"``; this reformats each
    ``column=value`` segment with ``pretty()`` -> ``"XAI Type=Rules | Tested
    W Xai=True"``.
    """
    segments = str(raw).split(" | ")
    prettified = []
    for segment in segments:
        if "=" in segment:
            column, _, value = segment.partition("=")
            prettified.append(f"{pretty(column)}={pretty(value)}")
        else:
            prettified.append(pretty(segment))
    return " | ".join(prettified)
