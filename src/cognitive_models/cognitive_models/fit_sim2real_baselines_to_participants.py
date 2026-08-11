"""Fit DT / logistic-regression / MLP proxy baselines to Sim2Real participants.

Companion to ``fit_sim2real_attribution_sum_to_participants.py``, scoring the
plain machine-proxy baselines in ``src/cognitive_models/baseline_models``
against the same participants instead of the grid-searched attribution-sum
cognitive model. This does not import or modify anything under
``cognitive_models`` beyond read-only use of ``Sim2RealAttributionProjector``.

Unlike the attribution-sum fitter, this reads human answers from the already
anonymised ``assets/human_data/Sim2Real/sim2real_human_trials.csv`` rather
than the raw external participant directory: that table already carries only
the 46 participants ``load_screened_participant_trials`` finds eligible (see
``assets/build_human_data.py``), so no external path is needed here and the
output participant ids already match every other Sim2Real table.

For each ``exp_property``, one baseline is trained once (not per participant)
on the ten training cases -- features plus that property's LIME explanation
predicting the AI's label, exactly the input shape
``DualInputClassifierBaseline`` expects. A single comparison logistic
(``comparison_scale``, ``comparison_intercept``) is then fit on those same
ten cases' ``confidence_delta -> Higher/Lower`` feedback from ``deltas.csv``,
mirroring ``gcm_strategies.py``'s own ``fit()`` methods. Every participant
assigned to that property is scored against the resulting fixed model, so
NLL/BIC use ``n_parameters=2`` -- the same two learned parameters every
Sim2Real cognitive model pays for at this stage, per
``LEARNED_PARAMETER_NAMES`` in the attribution-sum fitter -- not the model's
own (there is none; the baseline is not re-fit per participant).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cognitive_models.baseline_models import (  # noqa: E402
    DecisionTreeBaseline,
    LogisticRegressionBaseline,
    MLPBaseline,
)
from src.cognitive_models.cognitive_models.Sim2Real.gcm_strategies import (  # noqa: E402
    Sim2RealAttributionProjector,
)

HUMAN_DATA = REPO_ROOT / "assets" / "human_data" / "Sim2Real"
HUMAN_TEST_QUESTION_TYPE = "our_prop_good_under_model"
EXPECTED_TEST_QIDS = frozenset(range(29))
EXP_PROPERTY_BY_EXPLANATION_TYPE = {
    0: "faithful",
    1: "sparse",
    2: "robust",
    3: "sparse_robust",
}

#: The two parameters every scored model here pays for: the comparison
#: logistic fit on the ten training cases. See the module docstring.
N_LEARNED_PARAMETERS = 2

BASELINE_CLASSES = {
    "decision_tree": DecisionTreeBaseline,
    "logistic_regression": LogisticRegressionBaseline,
    "mlp": MLPBaseline,
}

_VALUE_COLUMNS = [f"v{i}" for i in range(67)]
_ATTRIBUTION_COLUMNS = [f"a{i}_i" for i in range(67)]

__all__ = [
    "BASELINE_CLASSES",
    "N_LEARNED_PARAMETERS",
    "load_test_trials",
    "score_baseline",
    "run_fitting",
]


def load_test_trials(
    path: Path = HUMAN_DATA / "sim2real_human_trials.csv",
) -> pd.DataFrame:
    """Screened human test-question answers from the consolidated table.

    ``sim2real_human_trials.csv`` already carries only participants who
    passed the fitter's completeness screening, so this only extracts the
    29-per-participant test rows and parses their answers -- it does not
    re-screen.
    """
    frame = pd.read_csv(path, low_memory=False)
    test = frame[frame["question_type"] == HUMAN_TEST_QUESTION_TYPE].copy()
    if test.empty:
        raise ValueError(f"No {HUMAN_TEST_QUESTION_TYPE!r} rows found in {path}")
    test["qid"] = pd.to_numeric(test["qid"], errors="raise").astype(int)
    explanation_type = pd.to_numeric(test["explanation_type"], errors="raise").astype(int)
    test["exp_property"] = explanation_type.map(EXP_PROPERTY_BY_EXPLANATION_TYPE)
    if test["exp_property"].isna().any():
        bad = sorted(explanation_type[test["exp_property"].isna()].unique().tolist())
        raise ValueError(f"Unmapped explanation_type value(s) in {path}: {bad}")
    test["human_answer"] = [json.loads(value)["Q0"] for value in test["responses"]]
    if not test["human_answer"].isin(["Higher", "Lower"]).all():
        raise ValueError(f"{path} has a test row whose Q0 is not Higher/Lower")
    test["human_label"] = test["human_answer"].map({"Lower": 0, "Higher": 1})

    counts = test.groupby("participant_id")["qid"].nunique()
    incomplete = counts[counts != len(EXPECTED_TEST_QIDS)]
    if len(incomplete):
        raise ValueError(
            f"Participants with an incomplete test set in {path}: "
            f"{incomplete.index.tolist()}"
        )
    return test[
        ["participant_id", "exp_property", "qid", "human_answer", "human_label"]
    ].reset_index(drop=True)


def _test_qid_order(projector: Sim2RealAttributionProjector) -> pd.DataFrame:
    """``instanceId`` for each test ``qid``, in ``qid`` order 0..28."""
    deltas = projector.deltas
    mapping = deltas[
        deltas["appId"].eq(projector.app_id) & deltas["split"].astype(str).eq("test")
    ][["qid", "instanceId"]].copy()
    mapping["qid"] = mapping["qid"].astype(int)
    mapping["instanceId"] = mapping["instanceId"].astype(int)
    if set(mapping["qid"]) != EXPECTED_TEST_QIDS:
        raise ValueError("Testing deltas must contain exactly qids 0..28")
    return mapping.sort_values("qid").reset_index(drop=True)


def _original_case_table(
    projector: Sim2RealAttributionProjector,
    exp_property: str,
    instance_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(features, explanation, AI predicted label) for the original case."""
    values = (
        projector.values[projector.values["appId"] == projector.app_id]
        .set_index("instanceId")
    )
    attributions = (
        projector.attributions[
            (projector.attributions["appId"] == projector.app_id)
            & (projector.attributions["expProperty"] == exp_property)
        ]
        .set_index("instanceId")
    )
    features = values.loc[list(instance_ids), _VALUE_COLUMNS].to_numpy(dtype=float)
    explanation = attributions.loc[list(instance_ids), _ATTRIBUTION_COLUMNS].to_numpy(
        dtype=float
    )
    # Sparse_robust training instance 9's explanation is truncated to 27 of 67
    # dimensions in the source (see build_corpus.py's own "truncated in
    # source" report); missing dimensions become 0, matching the
    # --missing-attributions=zero convention the attribution-sum fitter uses.
    explanation = np.nan_to_num(explanation, nan=0.0)
    ai_prediction = attributions.loc[list(instance_ids), "pred"].to_numpy(dtype=int)
    return features, explanation, ai_prediction


def _changed_case_features(
    projector: Sim2RealAttributionProjector,
    exp_property: str,
    instance_ids: Sequence[int],
) -> np.ndarray:
    """The counterfactual case's own encoding -- stored under the ``a*_i``
    columns of ``counterfactuals_fake.csv``, not an attribution weight."""
    counterfactuals = (
        projector.counterfactuals[
            (projector.counterfactuals["appId"] == projector.app_id)
            & (projector.counterfactuals["expProperty"] == exp_property)
        ]
        .set_index("instanceId")
    )
    changed = counterfactuals.loc[list(instance_ids), _ATTRIBUTION_COLUMNS].to_numpy(
        dtype=float
    )
    return np.nan_to_num(changed, nan=0.0)


def _fit_comparison_logistic(
    delta: np.ndarray, higher_label: np.ndarray
) -> tuple[float, float]:
    """Fit ``confidence_delta -> Higher`` the same way ``gcm_strategies.py``'s
    own ``fit()`` methods do: a 1D ``LogisticRegression`` on the training
    cases' feedback."""
    from sklearn.linear_model import LogisticRegression

    if len(set(higher_label.tolist())) < 2:
        # Degenerate training feedback (every case agrees) leaves nothing for
        # the logistic to contrast; a steep, centered fallback keeps the sign
        # of delta decisive without claiming a gradient was fit.
        return 1e6, 0.0
    estimator = LogisticRegression(C=1e6)
    estimator.fit(delta.reshape(-1, 1), higher_label)
    return float(estimator.coef_[0, 0]), float(estimator.intercept_[0])


def score_baseline(
    name: str,
    exp_property: str,
    projector: Sim2RealAttributionProjector,
) -> pd.DataFrame:
    """Train one baseline once on the training cases; predict every test qid.

    Returns one row per test ``qid`` -- the model is not re-fit per
    participant, so every participant sharing this ``exp_property`` is scored
    against the same predictions.
    """
    if name not in BASELINE_CLASSES:
        raise ValueError(f"Unknown baseline {name!r}. Known: {sorted(BASELINE_CLASSES)}.")

    training_ids = projector.instance_ids_for_split("training")
    x_train, e_train, y_train = _original_case_table(projector, exp_property, training_ids)
    model = BASELINE_CLASSES[name]()
    # Plain arrays, not DataFrames: DualInputClassifierBaseline remembers
    # fit-time column labels and requires an exact match at inference, which
    # a second freshly built DataFrame's default RangeIndex does not give it.
    model.fit(x_train, y_train, explanations=e_train)

    def _confidence_delta(instance_ids: Sequence[int]) -> np.ndarray:
        x_original, explanation, _ = _original_case_table(projector, exp_property, instance_ids)
        x_changed = _changed_case_features(projector, exp_property, instance_ids)
        p_original = model.predict_proba(x_original, explanations=explanation)[:, 1]
        p_changed = model.predict_proba(x_changed, explanations=explanation)[:, 1]
        return p_changed - p_original

    train_delta = _confidence_delta(training_ids)
    train_higher = projector.increase_labels(training_ids)
    scale, intercept = _fit_comparison_logistic(train_delta, train_higher)

    test_map = _test_qid_order(projector)
    test_ids = test_map["instanceId"].tolist()
    test_delta = _confidence_delta(test_ids)
    probability_higher = 1.0 / (1.0 + np.exp(-(intercept + scale * test_delta)))
    probability_higher = np.clip(probability_higher, 1e-9, 1 - 1e-9)

    return pd.DataFrame(
        {
            "baseline": name,
            "exp_property": exp_property,
            "qid": test_map["qid"].to_numpy(),
            "instanceId": test_map["instanceId"].to_numpy(),
            "comparison_scale": scale,
            "comparison_intercept": intercept,
            "model_probability_higher": probability_higher,
            "model_label": (probability_higher >= 0.5).astype(int),
        }
    )


def run_fitting(
    *,
    human_trials_path: Path = HUMAN_DATA / "sim2real_human_trials.csv",
    assets_root: Path | None = None,
    baselines: Sequence[str] = ("decision_tree", "logistic_regression", "mlp"),
) -> dict[str, pd.DataFrame]:
    """Fit every baseline for every ``exp_property``, score every participant.

    Returns ``{"predictions": ..., "fits": ...}`` -- the per-test-qid model
    output (participant-independent) and the per-participant NLL/BIC summary.
    """
    human_test = load_test_trials(human_trials_path)
    projector = Sim2RealAttributionProjector.from_assets(assets_root=assets_root)

    prediction_frames = [
        score_baseline(name, exp_property, projector)
        for name in baselines
        for exp_property in sorted(EXP_PROPERTY_BY_EXPLANATION_TYPE.values())
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)

    fit_rows: list[dict] = []
    for name in baselines:
        model_predictions = predictions[predictions["baseline"] == name]
        merged = human_test.merge(
            model_predictions[
                ["exp_property", "qid", "model_probability_higher", "model_label"]
            ],
            on=["exp_property", "qid"],
            how="inner",
            validate="many_to_one",
        )
        if len(merged) != len(human_test):
            raise ValueError(f"Baseline {name!r} did not cover every human test qid")
        for participant_id, rows in merged.groupby("participant_id", sort=True):
            probabilities = rows["model_probability_higher"].to_numpy(dtype=float)
            labels = rows["human_label"].to_numpy(dtype=int)
            total_nll = float(
                -np.sum(
                    labels * np.log(probabilities)
                    + (1 - labels) * np.log(1 - probabilities)
                )
            )
            n_trials = len(rows)
            fit_rows.append(
                {
                    "participant_id": participant_id,
                    "exp_property": rows["exp_property"].iloc[0],
                    "baseline": name,
                    "n_trials": n_trials,
                    "n_parameters": N_LEARNED_PARAMETERS,
                    "test_nll_model_participant": total_nll / n_trials,
                    "test_total_nll_model_participant": total_nll,
                    "test_bic_model_participant": (
                        N_LEARNED_PARAMETERS * np.log(n_trials) + 2.0 * total_nll
                    ),
                    "test_accuracy_model_participant": float(
                        np.mean(rows["model_label"].to_numpy() == labels)
                    ),
                }
            )
    fits = pd.DataFrame(fit_rows)
    return {"predictions": predictions, "fits": fits}


def main() -> int:
    outputs = run_fitting()
    HUMAN_DATA.mkdir(parents=True, exist_ok=True)
    predictions_path = HUMAN_DATA / "sim2real_baseline_fit_predictions.csv"
    fits_path = HUMAN_DATA / "sim2real_baseline_fits.csv"
    outputs["predictions"].to_csv(predictions_path, index=False)
    outputs["fits"].to_csv(fits_path, index=False)

    summary = (
        outputs["fits"]
        .groupby(["baseline", "exp_property"])
        .agg(
            participants=("participant_id", "nunique"),
            mean_nll=("test_nll_model_participant", "mean"),
            mean_bic=("test_bic_model_participant", "mean"),
            mean_accuracy=("test_accuracy_model_participant", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"\nwrote {predictions_path.relative_to(REPO_ROOT)}")
    print(f"wrote {fits_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
