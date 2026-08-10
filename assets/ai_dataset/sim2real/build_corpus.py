#!/usr/bin/env python3
"""Convert the Sim2Real raw case dumps into the standardized asset corpus.

Reads   assets/ai_dataset/sim2real/{raw_training,raw_testing}
Writes  assets/ai_dataset/sim2real/{metadata,values,none}.csv
        assets/explanations/xai_desiderata/*.csv

The raw files are JS source: a series of `var case_N = {...}` object literals
followed by a `sets_*_cases = [...]` array. raw_training holds the ten cases the
study's training page is built from (`sets_practice_cases`, consumed by
`create_training_aliens` in ui.html) — not its comprehension checks, which live in
a file this repo does not have. Each case carries 12 human-readable
observations, five 67-dimensional explanation vectors, a single-feature `delta`
to consider, and the answer key.

The 67 dimensions are the one-hot expansion of the 12 observed features. The
layout is pandas `get_dummies` order: the four numeric columns first (in original
column order), then one block per categorical column (columns alphabetical,
levels alphabetical within a block). ONE_HOT below reproduces it; it was verified
against the source study's own explanation table, dimension by dimension.

The raw ``delta`` sometimes declares a source value that is not the value in
``observations``.  Both the counterfactual and the label the participant reads
follow the observation: ``valueFrom``/``dimFrom`` are the value actually in
``values.csv``, so the suggested change always starts from what is on screen.
If the declared target is already the observed value, the direction is reversed
so the declaration's source becomes the counterfactual target instead of
producing a no-op. The literal declaration survives in
``sourceValueFrom``/``sourceValueTo``/``sourceDimFrom``/``rawDelta``, and
``mappingStatus`` records where the two disagree.

Run from the repository root:
    python3 assets/ai_dataset/sim2real/build_corpus.py
"""

import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_OUT_DIR = HERE
EXPLANATIONS_OUT_DIR = os.path.abspath(
    os.path.join(HERE, "..", "..", "explanations", "xai_desiderata")
)

APP_ID = "adult_sim2real"
MODEL_NAME = "synthetic_ai"

# Order of the 12 values in each case's "observations" list.
OBSERVED = [
    "age", "capital-gain", "capital-loss", "education", "hours-per-week",
    "marital", "native-country", "occupation", "race", "relationship",
    "sex", "workclass",
]

NUMERIC = ["age", "capital-gain", "capital-loss", "hours-per-week"]

# Categorical blocks in encoded order, levels in encoded order.
CATEGORICAL = [
    ("education", ["10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th",
                   "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad",
                   "Masters", "Preschool", "Prof-school", "Some-college"]),
    ("marital", ["Divorced", "Married-AF-spouse", "Married-civ-spouse",
                 "Married-spouse-absent", "Never-married", "Separated", "Widowed"]),
    ("native-country", ["United-States", "other", "unknown"]),
    ("occupation", ["Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
                    "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
                    "Other-service", "Priv-house-serv", "Prof-specialty",
                    "Protective-serv", "Sales", "Tech-support", "Transport-moving",
                    "unknown"]),
    ("race", ["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"]),
    ("relationship", ["Husband", "Not-in-family", "Other-relative", "Own-child",
                      "Unmarried", "Wife"]),
    ("sex", ["Female", "Male"]),
    ("workclass", ["Federal-gov", "Local-gov", "Never-worked", "Private",
                   "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay",
                   "unknown"]),
]

# Flattened dimension names, index-aligned with every explanation vector.
ONE_HOT = list(NUMERIC) + [
    f"{col}_{level}" for col, levels in CATEGORICAL for level in levels
]
N_DIM = len(ONE_HOT)
assert N_DIM == 67, N_DIM

# Source key -> explanation property.  All vectors use LIME as their explanation
# method; a blank property denotes the unoptimized LIME baseline.
EXPLANATION_VARIANTS = [
    ("faithful_explanation", "faithful"),
    ("robust_explanation", "robust"),
    ("sparse_explanation", "sparse"),
    ("sparse_and_robust_explanation", "sparse_robust"),
    ("lime", ""),
]

CASE_RE = re.compile(r"^var (case_\d+) = (\{.*?\})\s*$", re.M | re.S)
CASE_SET_RE = re.compile(
    r"^sets_[A-Za-z0-9_]*cases\s*=\s*\[([^\]]*)\]\s*;?\s*$", re.M
)
CASE_NAME_RE = re.compile(r"case_\d+")
TAG_RE = re.compile(r"<[^>]+>")


def load_cases(path):
    """Parse cases in the order declared by the file's ``sets_*_cases`` array."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    found = {}
    for match in CASE_RE.finditer(text):
        name = match.group(1)
        if name in found:
            raise ValueError(f"Duplicate {name} in {path}")
        found[name] = json.loads(match.group(2))

    set_matches = CASE_SET_RE.findall(text)
    if len(set_matches) != 1:
        raise ValueError(
            f"Expected exactly one sets_*_cases array in {path}; "
            f"found {len(set_matches)}"
        )
    order = CASE_NAME_RE.findall(set_matches[0])
    if len(order) != len(set(order)):
        raise ValueError(f"Duplicate case reference in the sets_*_cases array: {path}")
    if set(order) != set(found):
        missing = sorted(set(found) - set(order))
        unknown = sorted(set(order) - set(found))
        raise ValueError(
            f"Case definitions and sets_*_cases disagree in {path}: "
            f"unreferenced={missing}, undefined={unknown}"
        )
    return [(name, found[name]) for name in order]


def parse_delta(raw):
    """'race: White <span..>→ Other</span>' -> ('race', 'White', 'Other')."""
    plain = TAG_RE.sub("", raw)
    feature, rest = plain.split(":", 1)
    before, after = rest.split("→")
    return feature.strip(), before.strip(), after.strip()


def encode(observations):
    """12 observed values -> the 67-dim one-hot row."""
    if len(observations) != len(OBSERVED):
        raise ValueError(
            f"Expected {len(OBSERVED)} observations, got {len(observations)}"
        )
    seen = dict(zip(OBSERVED, observations))
    row = [float(seen[name]) for name in NUMERIC]
    for col, levels in CATEGORICAL:
        value = seen[col]
        if value not in levels:
            raise ValueError(
                f"Unsupported {col!r} level {value!r}; expected one of {levels}"
            )
        row.extend(1 if level == value else 0 for level in levels)
    return row


def _coerce_delta_value(feature, value):
    return float(value) if feature in NUMERIC else value


def _values_equal(feature, left, right):
    if feature in NUMERIC:
        return float(left) == float(right)
    return str(left) == str(right)


def _dimension_for(feature, value):
    """Return the encoded dimension occupied by a raw feature value."""
    if feature in NUMERIC:
        return ONE_HOT.index(feature)
    dimension = f"{feature}_{value}"
    if dimension not in ONE_HOT:
        raise ValueError(f"Delta uses a level outside ONE_HOT: {dimension}")
    return ONE_HOT.index(dimension)


def _feature_dimensions(feature):
    """Return all encoded dimensions belonging to one raw feature."""
    if feature in NUMERIC:
        return {ONE_HOT.index(feature)}
    prefix = f"{feature}_"
    return {i for i, name in enumerate(ONE_HOT) if name.startswith(prefix)}


def resolve_delta(case):
    """Resolve one raw declaration into its effective encoded transition.

    ``source_value_from`` is provenance copied from the raw delta string.
    ``value_from``/``value_to`` describe the intended suggested transition.
    ``observed_value_from`` is the untouched value in ``observations`` and is
    the actual source row copied to construct the counterfactual. Keeping all
    three makes source contradictions visible without altering the UI instance.
    """
    feature, source_value_from, source_value_to = parse_delta(case["delta"])
    if feature not in OBSERVED:
        raise ValueError(f"Delta uses an unknown feature: {feature!r}")

    feature_index = OBSERVED.index(feature)
    observed_value_from = case["observations"][feature_index]
    source_matches = _values_equal(
        feature, observed_value_from, source_value_from
    )
    declared_value_to = _coerce_delta_value(feature, source_value_to)
    direction_reversed = (
        not source_matches
        and _values_equal(feature, observed_value_from, declared_value_to)
    )
    # The suggested transition always starts from the value the participant can
    # see. Three raw declarations name a source the case does not have
    # (instances 2, 8 and 9), and labelling the change "capital-gain: 500 ->
    # 1205" beside a case reading 0 would ask the participant to reason from a
    # value that is not there. The declaration is kept verbatim in
    # source_value_from for audit; only the label follows the observation.
    value_from = _coerce_delta_value(feature, observed_value_from)
    value_to = _coerce_delta_value(
        feature,
        source_value_from if direction_reversed else source_value_to,
    )
    value_from_matches_instance = _values_equal(
        feature, observed_value_from, value_from
    )

    counterfactual_observations = list(case["observations"])
    counterfactual_observations[feature_index] = value_to
    original_encoded = encode(case["observations"])
    counterfactual_encoded = encode(counterfactual_observations)
    changed_dimensions = [
        i for i, (original, changed) in enumerate(
            zip(original_encoded, counterfactual_encoded)
        )
        if original != changed
    ]

    has_change = not _values_equal(feature, observed_value_from, value_to)
    allowed_dimensions = _feature_dimensions(feature)
    if not set(changed_dimensions).issubset(allowed_dimensions):
        raise ValueError(
            f"Delta for {feature!r} changed unrelated dimensions: "
            f"{changed_dimensions}"
        )
    if has_change and not changed_dimensions:
        raise ValueError(f"Delta for {feature!r} was lost during encoding")
    if not has_change and changed_dimensions:
        raise ValueError(f"No-op delta for {feature!r} changed encoded values")

    dim_from = _dimension_for(feature, value_from)
    observed_dim_from = _dimension_for(feature, observed_value_from)
    dim_to = _dimension_for(feature, value_to)
    expected_dimensions = {observed_dim_from, dim_to} if has_change else set()
    if set(changed_dimensions) != expected_dimensions:
        raise ValueError(
            f"Encoded delta for {feature!r} changed {changed_dimensions}; "
            f"expected {sorted(expected_dimensions)}"
        )

    # A malformed declared source is retained for auditing. It need not be a
    # valid encoded category because it is never used to construct the row.
    if feature in NUMERIC:
        source_dim_from = ONE_HOT.index(feature)
    else:
        source_dim_from = index_of(f"{feature}_{source_value_from}")

    return {
        "feature": feature,
        "feature_kind": "numeric" if feature in NUMERIC else "categorical",
        "value_from": value_from,
        "value_to": value_to,
        "dim_from": dim_from,
        "dim_to": dim_to,
        "observed_value_from": observed_value_from,
        "observed_dim_from": observed_dim_from,
        "value_from_matches_instance": value_from_matches_instance,
        "source_value_from": source_value_from,
        "source_value_to": source_value_to,
        "source_dim_from": source_dim_from,
        "source_matches": source_matches,
        "direction_reversed": direction_reversed,
        "has_change": has_change,
        "changed_dimensions": changed_dimensions,
        "counterfactual_encoded": counterfactual_encoded,
    }


def main(dataset_out_dir=DATASET_OUT_DIR,
         explanations_out_dir=EXPLANATIONS_OUT_DIR):
    os.makedirs(dataset_out_dir, exist_ok=True)
    os.makedirs(explanations_out_dir, exist_ok=True)

    cases = []
    for split, filename in (("training", "raw_training"), ("test", "raw_testing")):
        for name, case in load_cases(os.path.join(HERE, filename)):
            cases.append((split, name, case))

    # ---- metadata.csv -----------------------------------------------------
    # One row. Numeric dimensions get v{i}_min/v{i}_max so the renderer draws a
    # meter; one-hot dimensions get a 2-option categorical so it draws no/yes.
    encoded = [encode(case["observations"]) for _, _, case in cases]
    # The source study's own question id, carried through so a response logged
    # by the real study can be joined back to the instance it came from.
    # metadata.csv is one app-level row and deliberately has none.
    qids = [case["qid"] for _, _, case in cases]
    header, row = ["appId"], [APP_ID]
    for i, name in enumerate(ONE_HOT):
        header.append(f"a{i}")
        row.append(name)
        if name in NUMERIC:
            column = [values[i] for values in encoded]
            header += [f"v{i}_min", f"v{i}_max"]
            row += [min(column), max(column)]
        else:
            header += [f"v{i}_options", f"v{i}_0", f"v{i}_1"]
            row += [2, "no", "yes"]
    header += ["y", "y0", "y1"]
    row += ["income", "Below $50,000", "Above $50,000"]
    write(os.path.join(dataset_out_dir, "metadata.csv"), header, [row])

    # ---- values.csv -------------------------------------------------------
    # `y` is left empty: the source records only the model's prediction, never a
    # ground-truth label, so there is nothing honest to put here.
    header = (
        ["appId", "instanceId", "qid"]
        + [f"v{i}" for i in range(N_DIM)]
        + ["y"]
    )
    rows = [
        [APP_ID, i, qids[i]] + values + [""]
        for i, values in enumerate(encoded)
    ]
    write(os.path.join(dataset_out_dir, "values.csv"), header, rows)
    write(os.path.join(explanations_out_dir, "values.csv"), header, rows)

    # ---- none.csv ---------------------------------------------------------
    preds = [1 if "above" in case["original_model_prediction"] else 0
             for _, _, case in cases]
    none_header = ["appId", "modelName", "instanceId", "qid", "pred", "i_max"]
    none_rows = [
        [APP_ID, MODEL_NAME, i, qids[i], pred, 0]
        for i, pred in enumerate(preds)
    ]
    write(os.path.join(dataset_out_dir, "none.csv"), none_header, none_rows)
    write(os.path.join(explanations_out_dir, "none.csv"), none_header, none_rows)

    # ---- attribution.csv / importance.csv ---------------------------------
    # The renderer draws each bar as a{i}_i / i_max, so i_max sets the full-scale
    # reference. The source study scales the same way and picks its denominator
    # over the *whole corpus* for one explanation type, keeping bars comparable
    # across cases, so i_max here is constant per expProperty rather than per row.
    #
    # One divergence: the source normalises positives by max and negatives by
    # |min| separately (get_scaled_number in ui.html). The renderer has a single
    # denominator, so max(|min|, max) is used for both. Bars on whichever sign
    # has the smaller extreme therefore read shorter here than in the original.
    #
    # `intercept` is left empty because the source has no bias term; the
    # renderer's auto-added "Others" row therefore draws nothing, which is right
    # — the 67 dimensions are the entire input, so no residual is unaccounted for.
    truncated = []
    vectors = {}
    for instance_id, (_, _, case) in enumerate(cases):
        for key, exp_property in EXPLANATION_VARIANTS:
            vector = case[key]
            if len(vector) != N_DIM:
                # raw_training case_9 carries a 27-element sparse_and_robust
                # vector: the source export is truncated mid-array. Pad with
                # blanks, never zeros — the renderer skips a null a{i}_i, so the
                # missing dimensions read as unknown instead of as "no effect".
                truncated.append((instance_id, exp_property, len(vector)))
                vector = list(vector) + [""] * (N_DIM - len(vector))
            vectors[(instance_id, exp_property)] = vector

    scales = {}
    for _, exp_property in EXPLANATION_VARIANTS:
        magnitudes = [abs(v) for (_, prop), vec in vectors.items() if prop == exp_property
                      for v in vec if v != ""]
        scales[exp_property] = max(magnitudes, default=0) or 1

    header = (
        [
            "appId", "modelName", "expMethod", "expProperty", "instanceId",
            "qid", "pred", "i_max",
        ]
        + [f"a{i}_i" for i in range(N_DIM)]
        + ["intercept"]
    )
    attribution, importance = [], []
    for instance_id in range(len(cases)):
        for _, exp_property in EXPLANATION_VARIANTS:
            vector = vectors[(instance_id, exp_property)]
            head = [
                APP_ID, MODEL_NAME, "lime", exp_property, instance_id,
                qids[instance_id], preds[instance_id],
            ]
            attribution.append(head + [scales[exp_property]] + list(vector) + [""])
            importance.append(
                head + [scales[exp_property]]
                + [("" if v == "" else abs(v)) for v in vector]
                + [""]
            )
    write(os.path.join(explanations_out_dir, "attribution.csv"), header, attribution)
    write(os.path.join(explanations_out_dir, "importance.csv"), header, importance)

    # ---- counterfactuals_fake.csv -----------------------------------------
    # The filename is forced: local/iframe.js hardcodes
    # xaiType=counterfactual -> "xai_methods/counterfactuals_fake.csv".
    # Despite the name the rows here are real, derived from each case's `delta`.
    #
    # For this xaiType the renderer reads a{i}_i as the *counterfactual value*,
    # not an attribution — numeric attributes take the value itself, categorical
    # ones take an index into the option list (0 = "no", 1 = "yes"). It applies
    # the same a{i}_i / i_max * 100 scaling first, so i_max is pinned to 100 to
    # make that a no-op. (This is why the shipped counterfactuals_fake.csv draws
    # nothing: it has no i_max column at all, so every value resolves to null.)
    header = (
        [
            "appId", "modelName", "expMethod", "expProperty", "instanceId",
            "qid", "pred", "i_max",
        ]
        + [f"a{i}_i" for i in range(N_DIM)]
        + ["intercept"]
    )
    rows = []
    resolved_deltas = []
    for instance_id, (_, _, case) in enumerate(cases):
        resolved = resolve_delta(case)
        resolved_deltas.append(resolved)
        for _, exp_property in EXPLANATION_VARIANTS:
            rows.append(
                [
                    APP_ID, MODEL_NAME, "lime", exp_property, instance_id,
                    qids[instance_id], preds[instance_id], 100,
                ]
                + resolved["counterfactual_encoded"] + [""]
            )
    write(
        os.path.join(explanations_out_dir, "counterfactuals_fake.csv"),
        header,
        rows,
    )

    # ---- deltas.csv -------------------------------------------------------
    # The counterfactual half of each case: the single feature change the
    # participant is asked to reason about, plus the answer key. No renderer
    # reads this file yet — see the notes in local/sim2real/README.md.
    #
    # valueFrom/dimFrom describe the intended suggested transition. The
    # observedValueFrom/observedDimFrom fields preserve the untouched source in
    # values.csv, which can contradict that declaration. sourceValueFrom,
    # sourceDimFrom, and rawDelta preserve the literal raw declaration even when
    # its direction is reversed to avoid a no-op (instance 1).
    # `presentationOrder` is 1-based within a split, and means different things
    # in each:
    #
    #   training — the static training page's Case 1..10 numbering, which groups
    #     the five "model predicted higher income" cases first. That page is what
    #     participants actually saw, so this is a real order to present in.
    #   test — source order only. The study shuffles the test cases per
    #     participant (`shuffle(set_of_aliens_chosen)`, ui.html), so there is no
    #     canonical sequence; a host survey should randomise for itself.
    order = {}
    for split in ("training", "test"):
        members = [i for i, (s, _, _) in enumerate(cases) if s == split]
        if split == "training":
            members.sort(key=lambda i: (
                0 if "above" in cases[i][2]["original_model_prediction"] else 1, i
            ))
        for position, instance_id in enumerate(members, start=1):
            order[instance_id] = position

    header = [
        "appId", "instanceId", "split", "presentationOrder", "sourceCase", "qid",
        "questionType", "feature", "featureKind", "valueFrom", "valueTo",
        "dimFrom", "dimTo", "observedValueFrom", "observedDimFrom",
        "sourceValueFrom", "sourceValueTo", "sourceDimFrom",
        "valueFromMatchesInstance", "sourceValueFromMatchesInstance",
        "counterfactualHasChange", "deltaDirectionReversed", "changedDimensions",
        "mappingStatus", "rawDelta", "originalPrediction", "answer",
    ]
    rows = []
    for instance_id, (split, name, case) in enumerate(cases):
        resolved = resolved_deltas[instance_id]
        if resolved["direction_reversed"]:
            mapping_status = "source_direction_reversed"
        elif resolved["source_matches"]:
            mapping_status = "matched"
        else:
            mapping_status = "declared_source_differs_from_observation"
        rows.append([
            APP_ID, instance_id, split, order[instance_id], name, case["qid"],
            case.get("question_type", ""), resolved["feature"],
            resolved["feature_kind"], resolved["value_from"],
            resolved["value_to"], resolved["dim_from"], resolved["dim_to"],
            resolved["observed_value_from"], resolved["observed_dim_from"],
            resolved["source_value_from"], resolved["source_value_to"],
            resolved["source_dim_from"],
            1 if resolved["value_from_matches_instance"] else 0,
            1 if resolved["source_matches"] else 0,
            1 if resolved["has_change"] else 0,
            1 if resolved["direction_reversed"] else 0,
            ";".join(str(i) for i in resolved["changed_dimensions"]),
            mapping_status, case["delta"],
            1 if "above" in case["original_model_prediction"] else 0,
            1 if case["ans"] == "Higher" else 0,
        ])
    write(os.path.join(explanations_out_dir, "deltas.csv"), header, rows)

    print(
        f"{len(cases)} instances, {len(attribution)} explanation rows -> "
        f"{dataset_out_dir} and {explanations_out_dir}"
    )
    for instance_id, resolved in enumerate(resolved_deltas):
        if not resolved["source_matches"]:
            print(
                f"  audit: declared source differs from observation — instanceId={instance_id} "
                f"feature={resolved['feature']} "
                f"declared={resolved['source_value_from']!r} "
                f"observed={resolved['observed_value_from']!r}"
            )
        if resolved["direction_reversed"]:
            print(
                f"  audit: reversed no-op declaration — instanceId={instance_id} "
                f"effective={resolved['value_from']!r} -> "
                f"{resolved['value_to']!r}"
            )
        if not resolved["has_change"]:
            print(
                f"  audit: source-data no-op — instanceId={instance_id} "
                f"feature={resolved['feature']} value={resolved['value_from']!r}"
            )
    for instance_id, exp_property, length in truncated:
        print(f"  truncated in source: instanceId={instance_id} "
              f"expProperty={exp_property} ({length}/{N_DIM} dimensions)")


def index_of(dimension):
    """Dimension index, or '' when the level is outside the encoding."""
    return ONE_HOT.index(dimension) if dimension in ONE_HOT else ""


def write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
