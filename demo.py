"""
Demonstration script for the custom SLP classifier and regressor.

Runs binary classification, multiclass classification, digit recognition, and
regression demos; compares each custom estimator with sklearn's MLP
implementation; and saves task-specific evaluation plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor


OUTPUT_DIR = Path("demo_outputs")
RANDOM_STATE = 42


def slugify(name: str) -> str:
    """Create a filesystem-friendly dataset name."""
    return name.lower().replace(" ", "_")


def plot_loss_curves(
    title: str,
    custom_loss: list[float],
    sklearn_loss: list[float],
    output_path: Path,
) -> None:
    """Save a side-by-side training loss comparison plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(custom_loss, label="Custom SLP")
    plt.plot(sklearn_loss, label="sklearn MLP")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_score_comparison(
    title: str,
    metric: str,
    custom_score: float,
    sklearn_score: float,
    output_path: Path,
) -> None:
    """Save a compact custom-vs-sklearn score comparison."""
    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        ["Custom SLP", "sklearn MLP"],
        [custom_score, sklearn_score],
        color=["#2563eb", "#16a34a"],
    )
    plt.title(title)
    plt.ylabel(metric)
    lower = min(custom_score, sklearn_score)
    upper = max(custom_score, sklearn_score)
    padding = max(0.05, (upper - lower) * 0.35)
    if metric == "accuracy":
        plt.ylim(max(0, lower - padding), 1.0)
    else:
        plt.ylim(lower - padding, upper + padding)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    title: str,
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    class_labels: NDArray,
    output_path: Path,
) -> None:
    """Save a normalized confusion matrix for classification demos."""
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels, normalize="true")

    plt.figure(figsize=(7, 6))
    image = plt.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    tick_positions = np.arange(len(class_labels))
    plt.xticks(tick_positions, class_labels, rotation=45, ha="right")
    plt.yticks(tick_positions, class_labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value > 0.5 else "black"
            plt.text(col, row, f"{value:.2f}", ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confidence_histogram(
    title: str,
    probabilities: NDArray[np.floating],
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    output_path: Path,
) -> None:
    """Save confidence distributions for correct and incorrect predictions."""
    confidence = np.max(probabilities, axis=1)
    correct = y_true == y_pred

    plt.figure(figsize=(7, 4))
    plt.hist(
        confidence[correct],
        bins=15,
        range=(0, 1),
        alpha=0.75,
        label="Correct",
        color="#2563eb",
    )
    if np.any(~correct):
        plt.hist(
            confidence[~correct],
            bins=15,
            range=(0, 1),
            alpha=0.75,
            label="Incorrect",
            color="#dc2626",
        )
    plt.title(title)
    plt.xlabel("Predicted-class probability")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_regression_predictions(
    title: str,
    y_true: NDArray[np.floating],
    custom_pred: NDArray[np.floating],
    sklearn_pred: NDArray[np.floating],
    output_path: Path,
) -> None:
    """Save predicted-vs-actual scatter plots for both regressors."""
    min_value = min(np.min(y_true), np.min(custom_pred), np.min(sklearn_pred))
    max_value = max(np.max(y_true), np.max(custom_pred), np.max(sklearn_pred))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, pred, label, color in [
        (axes[0], custom_pred, "Custom SLP", "#2563eb"),
        (axes[1], sklearn_pred, "sklearn MLP", "#16a34a"),
    ]:
        ax.scatter(y_true, pred, alpha=0.75, color=color, edgecolor="none")
        ax.plot([min_value, max_value], [min_value, max_value], "k--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Actual target")
        ax.set_ylabel("Predicted target")
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residual_diagnostics(
    title: str,
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    output_path: Path,
) -> None:
    """Save residual scatter and residual histogram for regression demos."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.75, color="#2563eb", edgecolor="none")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals vs prediction")
    axes[0].set_xlabel("Predicted target")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=20, color="#2563eb", alpha=0.8)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Number of samples")

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def run_classification_demo(
    name: str,
    dataset_loader: Callable,
    hidden_layer_size: int,
    learning_rate: float,
    max_iter: int,
    batch_size: int,
    momentum: float,
) -> dict[str, str | float]:
    """Train and compare custom/sklearn classifiers on a dataset."""
    dataset = dataset_loader()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    custom_model = SimpleSLPClassifier(
        hidden_layer_size=hidden_layer_size,
        activation="relu",
        learning_rate=learning_rate,
        max_iter=max_iter,
        batch_size=batch_size,
        momentum=momentum,
        random_state=RANDOM_STATE,
    )
    custom_model.fit(X_train, y_train)
    custom_pred = custom_model.predict(X_test)
    custom_proba = custom_model.predict_proba(X_test)
    custom_score = accuracy_score(y_test, custom_pred)

    sklearn_model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation="relu",
        solver="sgd",
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        batch_size=batch_size,
        momentum=momentum,
        random_state=RANDOM_STATE,
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_score = accuracy_score(y_test, sklearn_pred)

    name_slug = slugify(name)
    loss_plot_path = OUTPUT_DIR / f"{name_slug}_loss.png"
    custom_confusion_path = OUTPUT_DIR / f"{name_slug}_custom_confusion_matrix.png"
    sklearn_confusion_path = OUTPUT_DIR / f"{name_slug}_sklearn_confusion_matrix.png"
    confidence_path = OUTPUT_DIR / f"{name_slug}_confidence_histogram.png"
    score_path = OUTPUT_DIR / f"{name_slug}_accuracy_comparison.png"

    plot_loss_curves(
        f"{name}: training loss",
        custom_model.loss_curve_,
        sklearn_model.loss_curve_,
        loss_plot_path,
    )
    plot_confusion_matrix(
        f"{name}: custom SLP confusion matrix",
        y_test,
        custom_pred,
        custom_model.classes_,
        custom_confusion_path,
    )
    plot_confusion_matrix(
        f"{name}: sklearn MLP confusion matrix",
        y_test,
        sklearn_pred,
        custom_model.classes_,
        sklearn_confusion_path,
    )
    plot_confidence_histogram(
        f"{name}: custom SLP confidence",
        custom_proba,
        y_test,
        custom_pred,
        confidence_path,
    )
    plot_score_comparison(
        f"{name}: accuracy comparison",
        "accuracy",
        custom_score,
        sklearn_score,
        score_path,
    )

    return {
        "dataset": name,
        "task": "classification",
        "metric": "accuracy",
        "custom_score": custom_score,
        "sklearn_score": sklearn_score,
        "plot": str(loss_plot_path),
        "plot_count": 5,
    }


def run_regression_demo(
    name: str,
    dataset_loader: Callable,
    hidden_layer_size: int,
    learning_rate: float,
    max_iter: int,
    batch_size: int,
    momentum: float,
) -> dict[str, str | float]:
    """Train and compare custom/sklearn regressors on a dataset."""
    dataset = dataset_loader()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
    )

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    custom_model = SimpleSLPRegressor(
        hidden_layer_size=hidden_layer_size,
        activation="relu",
        learning_rate=learning_rate,
        max_iter=max_iter,
        batch_size=batch_size,
        momentum=momentum,
        random_state=RANDOM_STATE,
    )
    custom_model.fit(X_train, y_train_scaled)
    custom_pred_scaled = custom_model.predict(X_test)
    custom_pred = y_scaler.inverse_transform(
        custom_pred_scaled.reshape(-1, 1)
    ).ravel()
    custom_score = r2_score(y_test, custom_pred)

    sklearn_model = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_size,),
        activation="relu",
        solver="sgd",
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        batch_size=batch_size,
        momentum=momentum,
        random_state=RANDOM_STATE,
    )
    sklearn_model.fit(X_train, y_train_scaled)
    sklearn_pred = y_scaler.inverse_transform(
        sklearn_model.predict(X_test).reshape(-1, 1)
    ).ravel()
    sklearn_score = r2_score(y_test, sklearn_pred)

    name_slug = slugify(name)
    loss_plot_path = OUTPUT_DIR / f"{name_slug}_loss.png"
    predictions_path = OUTPUT_DIR / f"{name_slug}_predicted_vs_actual.png"
    residuals_path = OUTPUT_DIR / f"{name_slug}_custom_residuals.png"
    score_path = OUTPUT_DIR / f"{name_slug}_r2_comparison.png"

    plot_loss_curves(
        f"{name}: training loss",
        custom_model.loss_curve_,
        sklearn_model.loss_curve_,
        loss_plot_path,
    )
    plot_regression_predictions(
        f"{name}: predicted vs actual",
        y_test,
        custom_pred,
        sklearn_pred,
        predictions_path,
    )
    plot_residual_diagnostics(
        f"{name}: custom SLP residual diagnostics",
        y_test,
        custom_pred,
        residuals_path,
    )
    plot_score_comparison(
        f"{name}: R2 comparison",
        "R2",
        custom_score,
        sklearn_score,
        score_path,
    )

    return {
        "dataset": name,
        "task": "regression",
        "metric": "R2",
        "custom_score": custom_score,
        "sklearn_score": sklearn_score,
        "plot": str(loss_plot_path),
        "plot_count": 4,
    }


def print_results(results: list[dict[str, str | float]]) -> None:
    """Print a compact score table."""
    print("\nDemonstration results")
    print("-" * 96)
    print(
        f"{'Dataset':<18} {'Task':<16} {'Metric':<10} "
        f"{'Custom SLP':>12} {'sklearn MLP':>12} {'Plots':>7}  Loss plot"
    )
    print("-" * 96)
    for result in results:
        print(
            f"{result['dataset']:<18} {result['task']:<16} {result['metric']:<10} "
            f"{result['custom_score']:>12.3f} {result['sklearn_score']:>12.3f}  "
            f"{result['plot_count']:>5}  "
            f"{result['plot']}"
        )
    print("-" * 96)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = [
        run_regression_demo(
            name="Diabetes",
            dataset_loader=load_diabetes,
            hidden_layer_size=50,
            learning_rate=0.001,
            max_iter=500,
            batch_size=32,
            momentum=0.9,
        ),
        run_classification_demo(
            name="Breast Cancer",
            dataset_loader=load_breast_cancer,
            hidden_layer_size=30,
            learning_rate=0.005,
            max_iter=300,
            batch_size=32,
            momentum=0.9,
        ),
        run_classification_demo(
            name="Wine",
            dataset_loader=load_wine,
            hidden_layer_size=30,
            learning_rate=0.005,
            max_iter=500,
            batch_size=16,
            momentum=0.9,
        ),
        run_classification_demo(
            name="Digits",
            dataset_loader=load_digits,
            hidden_layer_size=64,
            learning_rate=0.001,
            max_iter=300,
            batch_size=32,
            momentum=0.9,
        ),
    ]

    print_results(results)


if __name__ == "__main__":
    main()
