# pip install lofo-importance

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone

from lofo import LOFOImportance, Dataset as LOFODataset


def lofo_global_feature_ranking_and_save_csv(
    tabular_dataset,
    model_pipeline,
    output_csv_path,
    scoring="roc_auc",
    cv_folds=5,
    random_state=42,
    max_k=10,
):
    """
    Compute LOFO global feature importance ranking and save it as a CSV with
    columns v1..vk (k <= max_k), where each column contains a feature name.

    Parameters
    ----------
    tabular_dataset : TabularDataset
        Must expose X or X_numpy, y or y_numpy, and feature_names.
    model_pipeline : sklearn estimator or Pipeline
    output_csv_path : str
        Path to save the ranking CSV.
    scoring : str
        sklearn-compatible scoring string.
    cv_folds : int
    random_state : int
    max_k : int
        Maximum number of ranked features to save.
    """

    # -------- Extract data from TabularDataset --------
    if hasattr(tabular_dataset, "X"):
        X = tabular_dataset.X
    elif hasattr(tabular_dataset, "X_numpy"):
        X = tabular_dataset.X_numpy
    else:
        raise AttributeError("TabularDataset must have X or X_numpy.")

    if hasattr(tabular_dataset, "y"):
        y = tabular_dataset.y
    elif hasattr(tabular_dataset, "y_numpy"):
        y = tabular_dataset.y_numpy
    else:
        raise AttributeError("TabularDataset must have y or y_numpy.")

    if not hasattr(tabular_dataset, "feature_names"):
        raise AttributeError("TabularDataset must have feature_names.")

    feature_names = list(tabular_dataset.feature_names)

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.shape[1] != len(feature_names):
        raise ValueError("X columns and feature_names length mismatch.")

    # -------- Build DataFrame for LOFO --------
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # -------- Choose CV strategy --------
    y_unique = np.unique(y[~np.isnan(y)]) if np.issubdtype(y.dtype, np.number) else np.unique(y)
    is_classification = (len(y_unique) <= 50) and np.all(np.equal(y_unique % 1, 0))

    if is_classification:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # -------- Run LOFO --------
    lofo_ds = LOFODataset(df=df, target="target", features=feature_names)

    lofo = LOFOImportance(
        dataset=lofo_ds,
        model=clone(model_pipeline),
        scoring=scoring,
        cv=cv,
    )

    ranking_df = (
        lofo.get_importance()
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    # -------- Save v1..vk CSV --------
    ranked_features = ranking_df["feature"].tolist()
    k = min(len(ranked_features), max_k)

    out_row = {f"v{i+1}": ranked_features[i] for i in range(k)}
    out_df = pd.DataFrame([out_row])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)

    return ranking_df


# ---------------- Example usage ----------------
if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    dataset = load_data()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear"))
    ])

    ranking = lofo_global_feature_ranking_and_save_csv(
        tabular_dataset=dataset,
        model_pipeline=model,
        output_csv_path="results/wine_quality_lofo_ranking.csv",
        scoring="roc_auc",
        cv_folds=5,
        max_k=10,
    )

    print(ranking)
