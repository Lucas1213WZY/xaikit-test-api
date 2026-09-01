"""Display-label prettification shared by plots and stats tables.

Both the matplotlib grid plots and the analysis/post-hoc JSON payloads need
the same "decision_tree" -> "Rules" / "xai_type" -> "XAI Type" vocabulary, so
it lives here once rather than being redefined per consumer.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

#: Value -> display label overrides recognized automatically, checked before
#: generic prettification. CoXAM's conditions get its own vocabulary since
#: "Rules"/"Weights" aren't a mechanical transform of "decision_tree"/
#: "logistic_regression" -- title-casing alone cannot produce them.
KNOWN_LABELS: dict[str, str] = {
    "decision_tree": "Rules",
    "logistic_regression": "Weights",
    "hybrid": "Hybrid",
    # CoXAM's runner writes the surrogate it actually showed per trial into
    # `explanation_type`, abbreviated. Title-casing those gives "Dt"/"Lr",
    # which names nothing a reader recognizes -- and leaves the same surrogate
    # labelled two different ways depending on which column a plot split on.
    "dt": "Rules",
    "lr": "Weights",
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


#: Display order for the factors whose levels have a conventional reading
#: order. A factor absent from here keeps whatever order the data presents --
#: this is a presentation convention, not a claim about the design, and it
#: never invents a level the results do not contain.
LEVEL_ORDER: dict[str, tuple[Any, ...]] = {
    # The no-explanation baseline first, then the two CoAX views in order of
    # what they reveal: magnitude only, then signed contribution.
    "xai_type": ("none", "importance", "attribution"),
    # The control reads first: accuracy without the explanation, then with it.
    "tested_w_xai": (False, True),
    "tested_with_xai": (False, True),
}

#: Level display names ``pretty()`` cannot produce. A boolean factor otherwise
#: shows ``True``/``False`` -- the value the trial table stores, not the
#: condition it stands for.
LEVEL_LABELS: dict[str, dict[Any, str]] = {
    "tested_w_xai": {False: "w/o XAI", True: "w/ XAI"},
    "tested_with_xai": {False: "w/o XAI", True: "w/ XAI"},
}

#: Spellings one boolean condition arrives in -- a real bool from the trial
#: table, ``"True"`` after a CSV round trip, ``"w/ XAI"`` from a published
#: corpus. Mirrors ``palette._as_bool``, which pins the same conditions' colors.
_TRUE_SPELLINGS = {"true", "1", "yes", "y", "w/ xai", "with xai", "with_xai"}
_FALSE_SPELLINGS = {"false", "0", "no", "n", "w/o xai", "without xai", "without_xai"}


def level_key(value: Any) -> Any:
    """A level's identity for matching, independent of how it is spelled.

    Only ever applied to a factor listed above, so collapsing ``0``/``1`` to a
    boolean cannot disturb an ordinary numeric level elsewhere.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().casefold()
    if text in _TRUE_SPELLINGS:
        return True
    if text in _FALSE_SPELLINGS:
        return False
    return text


def display_order(column: Any, observed: Sequence[Any]) -> Optional[list[Any]]:
    """``observed`` in this factor's conventional order, or None if it has none.

    Returns the observed values themselves rather than the registered ones, so
    the result stays comparable to the data by ordinary equality whatever
    spelling it arrived in. Levels with no registered position keep their
    observed order, after the ones that have.
    """
    preferred = LEVEL_ORDER.get(str(column).strip())
    if preferred is None:
        return None
    rank = {level_key(level): index for index, level in enumerate(preferred)}
    deduped: list[Any] = []
    for value in observed:
        if all(level_key(value) != level_key(seen) for seen in deduped):
            deduped.append(value)
    return sorted(deduped, key=lambda value: rank.get(level_key(value), len(rank)))


def level_display(column: Any, value: Any) -> str:
    """One level's display name: its convention if it has one, else ``pretty``."""
    mapping = LEVEL_LABELS.get(str(column).strip())
    if mapping is not None:
        key = level_key(value)
        for level, label in mapping.items():
            if level_key(level) == key:
                return label
    return pretty(value)


def display_labels(
    column: Any, observed: Sequence[Any]
) -> Optional[dict[Any, str]]:
    """Display names for ``observed``, keyed by the observed value, or None.

    None -- rather than an empty dict -- when the factor has no convention, so
    a caller can pass the result straight through to a plot helper and leave
    its own default labelling untouched.
    """
    if str(column).strip() not in LEVEL_LABELS:
        return None
    return {value: level_display(column, value) for value in observed}
