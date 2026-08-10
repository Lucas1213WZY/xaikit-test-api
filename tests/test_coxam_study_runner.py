"""The CoXAM runner must drive the trained meta-policy from a study's own trials.

Kept deliberately small: two participants, two trials each, and a stub AI model
instead of a trained one wherever the real model's predictions are not what is
under test.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
    TrialDrivenCombinedStrategyPolicyEnv,
    _canonical_coxam_condition,
    _canonical_coxam_explanation_type,
    COXAM_CORPUS_FEATURES,
    _check_corpus_features_match,
    build_coxam_bundle,
    coxam_balanced_instance_ids,
    default_coxam_config,
    fit_coxam_surrogates,
    load_coxam_sub_policies,
)
from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_study_runner import (
    _model_input_transform,
    _predict_label,
    _trial_condition,
    _trial_explanation_family,
    _trial_shows_explanation,
)


class _StubAIModel:
    """Predicts by thresholding the first feature -- enough to fit surrogates against."""

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X[:, 0] > np.median(X[:, 0])).astype(int)


class _ProbabilityAIModel:
    """Returns per-class probabilities, the way this repo's real models do.

    ``predict()`` on the torch MLP returns the network's raw output and on the
    Keras one returns probabilities -- never integer labels. Every other stub
    here returns clean ints, so this one guards the conversion.
    """

    def predict(self, X):
        # Centred on 0.5 so it straddles the threshold across a transformed
        # [0, 1] feature as well as a raw standard-normal one.
        X = np.asarray(X, dtype=float)
        return 1.0 / (1.0 + np.exp(-10.0 * (X[:, 0] - 0.5)))


class _TwoColumnProbabilityAIModel(_ProbabilityAIModel):
    """Same, but shaped (n, 2) -- the other real output shape."""

    def predict(self, X):
        p_class_1 = super().predict(X)
        return np.column_stack([1.0 - p_class_1, p_class_1])


@pytest.fixture(scope="module")
def synthetic_bundle():
    rng = np.random.default_rng(0)
    n_instances, n_features = 40, 4
    X = rng.normal(size=(n_instances, n_features))
    instance_ids = list(range(n_instances))
    feature_names = [f"f{i}" for i in range(n_features)]

    surrogates = fit_coxam_surrogates(
        app_id="synthetic",
        model_name="stub",
        X=X,
        instance_ids=instance_ids,
        trained_ai_model=_StubAIModel(),
        feature_names=feature_names,
    )
    features = pd.DataFrame({"dataId": "synthetic", "instanceId": instance_ids})
    for index in range(n_features):
        features[f"x{index}"] = X[:, index]

    return build_coxam_bundle(
        app_id="synthetic",
        model_name="stub",
        features=features,
        predictions=surrogates["predictions"],
        lr_explanations=surrogates["lr_explanations"],
        dt_explanations=surrogates["dt_explanations"],
        metadata=surrogates["metadata"],
    )


# -- canonical mappings ---------------------------------------------------


@pytest.mark.parametrize(
    "xai_type, expected",
    [
        ("decision_tree", "decision_tree"),
        # CoXAM's own CONDITIONS tuple says linear_regression, the design says
        # logistic_regression -- this is the only value that actually renames.
        ("logistic_regression", "linear_regression"),
        ("hybrid", "hybrid"),
        ("Logistic Regression", "linear_regression"),
    ],
)
def test_condition_mapping(xai_type, expected):
    assert _canonical_coxam_condition(xai_type) == expected


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="Unknown CoXAM condition"):
        _canonical_coxam_condition("attribution")


@pytest.mark.parametrize(
    "family, shown, expected",
    [
        ("decision_tree", True, "dt"),
        ("logistic_regression", True, "lr"),
        ("decision_tree", False, "none"),
        ("logistic_regression", False, "none"),
    ],
)
def test_explanation_type_mapping(family, shown, expected):
    assert _canonical_coxam_explanation_type(family, shown=shown) == expected


# -- per-trial field resolution -------------------------------------------


def test_explanation_family_prefers_shown_xai_type_over_the_condition():
    """Under hybrid the condition alone cannot name the family -- the per-trial
    shown_xai_type the trial generator writes must win."""
    trial = {"xai_type": "hybrid", "shown_xai_type": "decision_tree"}
    assert _trial_explanation_family(trial) == "decision_tree"


def test_explanation_family_falls_back_for_older_trial_tables():
    assert _trial_explanation_family({"xai_type": "decision_tree"}) == "decision_tree"
    assert _trial_explanation_family({"xai_method": "logistic_regression"}) == "logistic_regression"


def test_condition_ignores_shown_xai_type():
    """A hybrid episode must stay hybrid so all three strategies remain legal,
    even though each of its trials shows just one family."""
    trial = {"xai_type": "hybrid", "shown_xai_type": "decision_tree"}
    assert _trial_condition(trial) == "hybrid"


@pytest.mark.parametrize(
    "trial, expected",
    [
        ({"phase": "testing", "tested_w_xai": True}, True),
        ({"phase": "testing", "tested_w_xai": False}, False),
        # Training trials always show the explanation, matching CoAX's own rule.
        ({"phase": "training", "tested_w_xai": False}, True),
        ({"phase": "testing", "tested_w_xai": "true"}, True),
    ],
)
def test_shows_explanation(trial, expected):
    assert _trial_shows_explanation(trial) is expected


# -- the forced-episode environment ---------------------------------------


def test_forced_episode_consumes_the_given_instances_in_order(synthetic_bundle):
    """The base env samples instances randomly; the subclass must not."""
    forced_ids = [7, 3, 11, 2]
    schedule = ["none", "dt", "lr", "none"]

    env = TrialDrivenCombinedStrategyPolicyEnv(
        bundles=[synthetic_bundle],
        config=default_coxam_config(
            condition_name="hybrid", instances_per_episode=len(forced_ids)
        ),
        sub_policies=load_coxam_sub_policies(),
        training=False,
    )
    env.queue_forced_episode(instance_ids=forced_ids, explanation_schedule=schedule)
    env.reset()

    assert env.sampled_instance_ids == forced_ids
    assert env.explanation_schedule == schedule


def test_forced_episode_rejects_mismatched_lengths(synthetic_bundle):
    env = TrialDrivenCombinedStrategyPolicyEnv(
        bundles=[synthetic_bundle],
        config=default_coxam_config(condition_name="hybrid", instances_per_episode=2),
        sub_policies=load_coxam_sub_policies(),
        training=False,
    )
    with pytest.raises(ValueError, match="same length"):
        env.queue_forced_episode(instance_ids=[1, 2], explanation_schedule=["none"])


def test_forced_episode_rejects_instances_outside_the_bundle(synthetic_bundle):
    env = TrialDrivenCombinedStrategyPolicyEnv(
        bundles=[synthetic_bundle],
        config=default_coxam_config(condition_name="hybrid", instances_per_episode=1),
        sub_policies=load_coxam_sub_policies(),
        training=False,
    )
    env.queue_forced_episode(instance_ids=[9999], explanation_schedule=["none"])
    with pytest.raises(ValueError, match="not in bundle"):
        env.reset()


@pytest.mark.parametrize(
    "condition, expected",
    [
        ("decision_tree", {"dt_traversal"}),
        ("linear_regression", {"lr_calculation", "lr_heuristic"}),
        # hybrid shows both families, so every strategy stays available.
        ("hybrid", {"lr_calculation", "lr_heuristic", "dt_traversal"}),
    ],
)
def test_condition_gates_which_strategies_are_available(synthetic_bundle, condition, expected):
    env = TrialDrivenCombinedStrategyPolicyEnv(
        bundles=[synthetic_bundle],
        config=default_coxam_config(condition_name=condition, instances_per_episode=1),
        sub_policies=load_coxam_sub_policies(),
        training=False,
    )
    env.episode_condition = condition
    available = {
        name for name in env.strategy_slots if env._strategy_available(name)
    }
    assert available == expected


# -- surrogate source selection -------------------------------------------


def test_asset_corpus_exposes_its_own_instance_ids():
    """The corpus serves a narrower range than a study's own split, so trial
    generation has to be constrained to it -- this is what tells it the range."""
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        coxam_available_instance_ids,
    )

    available = coxam_available_instance_ids("wine_quality")

    assert available
    assert available == sorted(set(available))
    assert all(isinstance(value, int) for value in available)


def test_unknown_source_is_rejected():
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_study_runner import (
        run_coxam_study,
    )

    class _Study:
        trials = [{"participantId": 1, "instanceId": 0}]
        data = object()

    with pytest.raises(ValueError, match="source must be 'fit' or 'assets'"):
        run_coxam_study(_Study(), source="nonsense")


def test_assets_source_builds_a_bundle_from_the_published_corpus():
    """source='assets' must reproduce the published tables without refitting."""
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        build_coxam_bundle,
        load_coxam_surrogates_from_assets,
    )

    surrogates = load_coxam_surrogates_from_assets(app_id="wine_quality")
    assert {"features", "predictions", "lr_explanations", "dt_explanations", "metadata"} <= set(
        surrogates
    )

    bundle = build_coxam_bundle(
        app_id="wine_quality",
        model_name="mlp",
        features=surrogates["features"],
        predictions=surrogates["predictions"],
        lr_explanations=surrogates["lr_explanations"],
        dt_explanations=surrogates["dt_explanations"],
        metadata=surrogates["metadata"],
    )

    assert bundle.app_id == "wine_quality"
    assert bundle.instance_ids
    # Both LR variants and both DT depths must be present for the episode's
    # complexity axis to have something to select.
    assert bundle.lr_dense is not None and bundle.lr_sparse is not None
    assert bundle.dt_depth2 is not None and bundle.dt_depth3 is not None


def test_assets_source_rejects_a_dataset_outside_the_corpus():
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        load_coxam_surrogates_from_assets,
    )

    with pytest.raises(ValueError, match="No CoXAM asset metadata"):
        load_coxam_surrogates_from_assets(app_id="not_a_real_dataset")


# -- server wiring --------------------------------------------------------


@pytest.mark.parametrize(
    "framework, expected",
    [("coax", "coax"), ("coxam", "coxam"), ("knn", "baseline"), ("", "baseline")],
)
def test_pipeline_routes_each_framework_to_its_runner(framework, expected):
    from types import SimpleNamespace

    from server import pipeline

    study = SimpleNamespace(design_export=SimpleNamespace(model_framework=framework))
    assert pipeline.participant_runner(study) == expected


def test_coxam_is_no_longer_rejected_as_unsupported():
    """It had no runner before, so the design was refused outright."""
    from server import pipeline

    assert "coxam" not in pipeline.UNSUPPORTED_FRAMEWORKS
    assert "coxam" in pipeline.COXAM_FRAMEWORKS


def test_simulation_request_carries_the_coxam_options():
    from server.schemas import SimulationRequest

    request = SimulationRequest()
    # None, not "fit": the pipeline resolves this from whether the dataset
    # stage actually trained a model (see run_simulation_stage), because a
    # static "fit" default would fail outright whenever training was skipped
    # for a corpus-covered dataset.
    assert request.coxam_source is None
    assert request.coxam_policy is None
    assert request.coxam_eval_params is None

    configured = SimulationRequest(
        coxam_source="assets",
        coxam_policy="dt_traversal",
        coxam_eval_params={"decision_noise": 0.4},
    )
    assert configured.coxam_source == "assets"
    assert configured.coxam_policy == "dt_traversal"
    assert configured.coxam_eval_params == {"decision_noise": 0.4}


# -- model output -> class label -------------------------------------------
#
# Every other stub in this file returns integer labels, but no real model in
# this repo does: predict() returns scores. Truncating a score with int()
# collapses every probability below 1.0 to class 0, which silently zeroed both
# the counterfactual DV and the labels the surrogates are fitted against.


class _StubTabularDataset:
    """The dataset's raw -> model-input transform, in miniature.

    Mirrors the real ``TabularDataset.prepare_instances_for_model``: min-max
    against *declared* boundaries rather than the observed range of X_raw, and
    deliberately no clipping, so a counterfactual perturbation outside the
    training range stays outside [0, 1].
    """

    def __init__(self, boundaries):
        self.boundaries = boundaries

    def prepare_instances_for_model(self, instances, one_hot_encode=True):
        instances = np.atleast_2d(np.asarray(instances, dtype=float))
        out = np.copy(instances)
        for index, (low, high) in self.boundaries.items():
            out[:, index] = (out[:, index] - low) / (high - low)
        return out


def _dataset_for(X_raw, boundaries):
    source = _StubTabularDataset(boundaries)
    return SimpleNamespace(
        split=SimpleNamespace(
            X_raw=X_raw,
            X_model=source.prepare_instances_for_model(X_raw),
            dataset=source,
        )
    )


@pytest.mark.parametrize("model", [_ProbabilityAIModel(), _TwoColumnProbabilityAIModel()])
def test_predict_label_thresholds_probabilities_instead_of_truncating(model):
    X_raw = np.linspace(-6.0, 6.0, 25).reshape(-1, 1)
    dataset = _dataset_for(X_raw, {0: (-6.0, 6.0)})

    labels = [_predict_label(model, dataset, row) for row in X_raw]

    assert set(labels) == {0, 1}, "int() truncation would make every label 0"
    # Monotone in the feature, so the labels must partition cleanly.
    assert labels[0] == 0 and labels[-1] == 1
    assert labels == sorted(labels)


def test_predict_label_uses_the_datasets_own_transform_not_a_reimplementation():
    """The scaling must come from the dataset, not from X_raw's observed range.

    These differ whenever the declared boundaries are wider than the data --
    which is the normal case. Reconstructing min-max from X_raw's own min/max
    produced a different vector and silently drove every instance to class 0.
    """
    X_raw = np.array([[2.0], [4.0], [6.0]])
    # Declared range is far wider than the observed [2, 6].
    dataset = _dataset_for(X_raw, {0: (0.0, 100.0)})

    seen = []

    class _Recorder:
        def predict(self, X):
            seen.append(np.asarray(X, dtype=float).reshape(-1)[0])
            return np.array([0.0])

    _predict_label(_Recorder(), dataset, X_raw[1])

    # Dataset transform: (4 - 0) / 100 = 0.04.
    # X_raw-derived min-max would give (4 - 2) / (6 - 2) = 0.5.
    assert seen == [pytest.approx(0.04)]


def test_predict_label_does_not_clip_out_of_range_counterfactuals():
    """A perturbation past the declared boundary must stay past it.

    Clipping to [0, 1] would erase exactly the changes a counterfactual makes.
    """
    dataset = _dataset_for(np.array([[0.0], [10.0]]), {0: (0.0, 10.0)})

    seen = []

    class _Recorder:
        def predict(self, X):
            seen.append(np.asarray(X, dtype=float).reshape(-1)[0])
            return np.array([0.0])

    _predict_label(_Recorder(), dataset, np.array([25.0]))

    assert seen == [pytest.approx(2.5)], "clipping would have flattened this to 1.0"


def test_surrogates_are_fitted_against_thresholded_labels():
    """A probability-returning model must not yield an all-zero label vector."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))

    surrogates = fit_coxam_surrogates(
        app_id="synthetic",
        model_name="stub",
        X=X,
        instance_ids=list(range(40)),
        trained_ai_model=_ProbabilityAIModel(),
        feature_names=["f0", "f1", "f2"],
    )

    labels = surrogates["predictions"]["pred"].dropna().astype(int)
    assert set(labels.unique()) == {0, 1}
    # Roughly balanced, since the stub thresholds a standard normal at 0.
    assert 0.2 < labels.mean() < 0.8


# -- the published corpus's feature order ----------------------------------


def test_corpus_feature_order_matches_the_published_metadata():
    """COXAM_CORPUS_FEATURES is pinned in code; it must still match the assets."""
    import re
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        COXAM_DATA_DIR,
    )

    metadata = pd.read_csv(COXAM_DATA_DIR / "metadata.csv")
    for app_id, expected in COXAM_CORPUS_FEATURES.items():
        row = metadata[metadata["dataId"] == app_id].iloc[0]
        columns = sorted(
            (c for c in metadata.columns if re.fullmatch(r"x\d+", c)),
            key=lambda c: int(c[1:]),
        )
        published = tuple(row[c] for c in columns if pd.notna(row[c]))
        assert published == expected, f"{app_id} corpus feature order drifted"


def test_corpus_covers_exactly_the_two_datasets_coxam_ran():
    """CoXAM's own studies and sweeps used wine_quality and mushrooms only."""
    assert set(COXAM_CORPUS_FEATURES) == {"wine_quality", "mushrooms"}


def test_asset_source_rejects_a_study_whose_features_differ():
    """The corpus's a{i} are positional, so a different feature set must raise.

    This repo's own wine_quality loader carries five features in a different
    order than the six-feature corpus, so silently combining them would attach
    every coefficient to the wrong feature.
    """
    with pytest.raises(ValueError, match="does not match this study's features"):
        _check_corpus_features_match(
            "wine_quality",
            ["Alcohol", "Vinegar Taint", "Sulphates", "SO2", "pH"],
        )


def test_asset_source_rejects_a_reordering_even_with_the_same_features():
    corpus = list(COXAM_CORPUS_FEATURES["mushrooms"])
    with pytest.raises(ValueError, match="different order"):
        _check_corpus_features_match("mushrooms", corpus[::-1])


def test_asset_source_accepts_the_corpus_order():
    for app_id, features in COXAM_CORPUS_FEATURES.items():
        _check_corpus_features_match(app_id, list(features))


def test_unknown_datasets_are_not_blocked():
    """A dataset outside the corpus is fit-only; nothing to cross-check."""
    _check_corpus_features_match("synthetic", ["f0", "f1"])


# -- class-balanced instance pools -----------------------------------------
#
# CoXAM's published corpus is balanced by construction (400 instances per
# dataset, ~50/50 by predicted class). A raw split usually is not -- this
# repo's wine_quality runs ~9% class 1 -- which leaves the counterfactual task
# asking for a flip of a confident majority-class prediction nine times in ten.


class _ImbalancedAIModel:
    """Predicts class 1 for only the first tenth of the rows."""

    def __init__(self, n_rows):
        self.n_rows = n_rows

    def predict(self, X):
        n = len(np.atleast_2d(np.asarray(X, dtype=float)))
        p = np.zeros(n)
        p[: max(1, self.n_rows // 10)] = 1.0
        return p


def test_balanced_pool_evens_out_a_skewed_split():
    n_rows = 100
    dataset = SimpleNamespace(
        split=SimpleNamespace(
            X_model=np.zeros((n_rows, 2)),
            raw_instance_ids=np.arange(n_rows),
        )
    )
    study = SimpleNamespace(data=dataset, trained_ai_model=_ImbalancedAIModel(n_rows))

    ids = coxam_balanced_instance_ids(study, per_class=10)

    # 10 of the scarce class and 10 of the plentiful one, not 90/10.
    assert len(ids) == 20
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_balanced_pool_takes_everything_when_a_class_is_too_small():
    n_rows = 30
    dataset = SimpleNamespace(
        split=SimpleNamespace(
            X_model=np.zeros((n_rows, 2)),
            raw_instance_ids=np.arange(n_rows),
        )
    )
    study = SimpleNamespace(data=dataset, trained_ai_model=_ImbalancedAIModel(n_rows))

    # Only 3 instances are class 1, so per_class=200 cannot be met.
    ids = coxam_balanced_instance_ids(study, per_class=200)
    assert len(ids) == n_rows


def test_balanced_pool_needs_a_trained_model():
    study = SimpleNamespace(data=SimpleNamespace(split=None), trained_ai_model=None)
    with pytest.raises(RuntimeError, match="No trained AI model"):
        coxam_balanced_instance_ids(study)


# -- one-hot resolution ----------------------------------------------------
#
# requires_one_hot_encoding says mlp/mlp_tf -> True, xgboost/sim2real -> False.
# But the flag only changes X_model's *width* when the dataset actually has
# categorical features: wine_quality is all continuous (5 -> 5) while mushrooms
# expands (5 -> 7). The transform resolves the flag by matching X_model's width,
# so it stays correct for both without being told the model type.


class _CategoricalStubDataset:
    """One continuous feature and one 3-way categorical."""

    def prepare_instances_for_model(self, instances, one_hot_encode=True):
        instances = np.atleast_2d(np.asarray(instances, dtype=float))
        continuous = instances[:, :1]
        if not one_hot_encode:
            return np.hstack([continuous, instances[:, 1:2]])
        categories = np.zeros((len(instances), 3))
        for row, value in enumerate(instances[:, 1].astype(int)):
            categories[row, min(max(value, 0), 2)] = 1.0
        return np.hstack([continuous, categories])


@pytest.mark.parametrize(
    "model_name, one_hot",
    [("mlp", True), ("mlp_tf", True), ("xgboost", False), ("sim2real", False)],
)
def test_transform_follows_requires_one_hot_encoding(model_name, one_hot):
    """The flag comes from the repo's own rule, not from a width guess.

    requires_one_hot_encoding returns True for mlp/mlp_tf, which is what the
    CoXAM notebook's run_ai_prediction hardcodes.
    """
    source = _CategoricalStubDataset()
    X_raw = np.array([[0.5, 0.0], [1.5, 1.0], [2.5, 2.0]])
    dataset = SimpleNamespace(
        split=SimpleNamespace(
            X_raw=X_raw,
            X_model=source.prepare_instances_for_model(X_raw, one_hot_encode=one_hot),
            dataset=source,
        )
    )

    rebuilt = np.vstack([_model_input_transform(dataset, model_name)(row) for row in X_raw])

    assert np.allclose(rebuilt, dataset.split.X_model)
    # 4 wide when expanded, 2 when not -- i.e. it really did pick a side.
    assert rebuilt.shape[1] == (4 if one_hot else 2)


def test_transform_matches_the_notebooks_hardcoded_choice_for_mlp():
    """CoXAM's run_ai_prediction passes one_hot_encode=True; mlp must agree."""
    from src.ai_models import requires_one_hot_encoding

    assert requires_one_hot_encoding("mlp") is True


@pytest.mark.parametrize("model_name", [None, "", "not_a_model"])
def test_transform_falls_back_to_x_model_width_for_an_unknown_model(model_name):
    """An unset model type must still reproduce X_model rather than raise."""
    source = _CategoricalStubDataset()
    X_raw = np.array([[0.5, 0.0], [1.5, 1.0]])
    dataset = SimpleNamespace(
        split=SimpleNamespace(
            X_raw=X_raw,
            X_model=source.prepare_instances_for_model(X_raw, one_hot_encode=False),
            dataset=source,
        )
    )

    rebuilt = np.vstack([_model_input_transform(dataset, model_name)(row) for row in X_raw])
    assert np.allclose(rebuilt, dataset.split.X_model)


# -- corpus feature aliases ----------------------------------------------


def test_corpus_aliases_only_rename_confirmed_equivalences():
    """The corpus abbreviates `Gill Spacing` and capitalises `chlorides`.

    Anything else must pass through untouched, so a genuinely different feature
    still fails the check rather than being quietly renamed onto a corpus slot.
    """
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        corpus_feature_names,
    )

    assert corpus_feature_names("wine_quality", ["chlorides"]) == ("Chlorides",)
    assert corpus_feature_names("mushrooms", ["Gill Spacing"]) == ("Gill",)
    assert corpus_feature_names("wine_quality", ["Alcohol", "pH"]) == ("Alcohol", "pH")
    assert corpus_feature_names("wine_quality", ["density"]) == ("density",)
    assert corpus_feature_names("not_a_dataset", ["chlorides"]) == ("chlorides",)


def test_the_loader_spelling_passes_the_corpus_check():
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        _check_corpus_features_match,
    )

    _check_corpus_features_match(
        "wine_quality", ["Alcohol", "Sulphates", "SO2", "Vinegar Taint", "pH", "chlorides"]
    )
    _check_corpus_features_match(
        "mushrooms", ["Bruises", "Height", "Width", "Shape", "Cap Diameter", "Gill Spacing"]
    )


def test_an_alias_does_not_excuse_the_wrong_order():
    """`a0..a5` are positional, so order still has to match after aliasing."""
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        _check_corpus_features_match,
    )

    with pytest.raises(ValueError, match="different order"):
        _check_corpus_features_match(
            "wine_quality", ["Sulphates", "Alcohol", "SO2", "Vinegar Taint", "pH", "chlorides"]
        )


def test_a_feature_outside_the_corpus_still_fails():
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        _check_corpus_features_match,
    )

    with pytest.raises(ValueError, match="different features"):
        _check_corpus_features_match(
            "wine_quality", ["Alcohol", "Sulphates", "SO2", "Vinegar Taint", "pH", "density"]
        )


def test_every_alias_target_is_a_real_corpus_feature():
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        COXAM_CORPUS_FEATURES,
        COXAM_CORPUS_FEATURE_ALIASES,
    )

    for app_id, aliases in COXAM_CORPUS_FEATURE_ALIASES.items():
        corpus = COXAM_CORPUS_FEATURES[app_id]
        for loader_name, corpus_name in aliases.items():
            assert corpus_name in corpus, f"{app_id}: {corpus_name} is not a corpus feature"
            assert loader_name not in corpus, f"{app_id}: {loader_name} needs no alias"
