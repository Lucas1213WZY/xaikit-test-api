import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.xai_adapter.surrogate import (  # noqa: E402
    DecisionTreeSurrogateMethod,
    LogisticRegressionSurrogateMethod,
)
from src.xai_adapter.visualization import (  # noqa: E402
    plot_decision_tree_instance_view,
    plot_logistic_regression_instance_view,
)


def _toy_data():
    X = np.asarray(
        [
            [9.8, 0.75, 14.0],
            [11.2, 0.62, 30.0],
            [10.4, 0.81, 18.0],
            [12.0, 0.50, 42.0],
            [9.5, 0.92, 12.0],
            [11.7, 0.48, 33.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)
    return X, y


def test_logistic_regression_instance_view_renders_from_fitted_surrogate():
    import matplotlib.pyplot as plt

    X, y = _toy_data()
    surrogate = LogisticRegressionSurrogateMethod(
        app_id="toy",
        model_name="mlp",
        variant="dense",
        feature_names=["Alcohol", "Sulphates", "SO2"],
    ).fit(X, y)

    fig, axes = plot_logistic_regression_instance_view(
        surrogate,
        X[0],
        instance_id=0,
        class_labels=["Type 1", "Type 2"],
    )

    assert fig is not None
    assert len(axes) == 7
    plt.close(fig)


def test_decision_tree_instance_view_renders_from_fitted_surrogate():
    import matplotlib.pyplot as plt

    X, y = _toy_data()
    surrogate = DecisionTreeSurrogateMethod(
        app_id="toy",
        model_name="mlp",
        depth=2,
        feature_names=["Alcohol", "Sulphates", "SO2"],
        class_labels=["Type 1", "Type 2"],
    ).fit(X, y)

    fig, axes = plot_decision_tree_instance_view(
        surrogate,
        X[0],
        instance_id=0,
        class_labels=["Type 1", "Type 2"],
    )

    assert fig is not None
    assert len(axes) == 4
    plt.close(fig)
