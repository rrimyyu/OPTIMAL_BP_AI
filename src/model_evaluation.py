# model_evaluation.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold
from confidenceinterval import roc_auc_score as ci_roc_auc_score

matplotlib.rcParams["font.family"] = "Arial"


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _predict_proba_binary(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return probability of the positive class for either a Keras model
    or an sklearn-like estimator. The output is a 1-D float array in [0,1].
    """
    # Keras models (Sequential/Functional)
    if isinstance(model, tf.keras.Model):
        # Keras predict returns (N,1) for binary; ravel() makes it (N,)
        proba = model.predict(X.to_numpy(), verbose=0).ravel()
        # If the model outputs logits, you may uncomment below:
        # proba = 1 / (1 + np.exp(-proba))
        return proba.astype(float)

    # sklearn-like models with predict_proba
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].ravel().astype(float)

    # Fallback: use decision_function or predict (not recommended, but safe-guard)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X).ravel().astype(float)
        # Min-max to [0,1] if scores are unbounded; adjust if you prefer sigmoid
        s_min, s_max = scores.min(), scores.max()
        proba = (scores - s_min) / (s_max - s_min + 1e-12)
        return proba
    else:
        # As a last resort, treat predicted labels as probabilities (0/1)
        return model.predict(X).ravel().astype(float)


def _youden_optimal_threshold(y_true: Iterable[int], y_proba: np.ndarray) -> float:
    """
    Compute Youden's J statistic to choose the optimal threshold.
    Returns the threshold value (float).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden = tpr - fpr  # equivalent to Sens + Spec - 1
    return thresholds[np.argmax(youden)]


def _binary_metrics_from_threshold(y_true: Iterable[int], y_proba: np.ndarray, threshold: float) -> Tuple[float, float, float, float]:
    """
    Compute Sensitivity, Specificity, PPV, NPV using a fixed threshold.
    Returns: (sensitivity, specificity, ppv, npv)
    """
    y_pred = (y_proba > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn + 1e-12)
    return sens, spec, ppv, npv


def _auc_with_ci(y_true: Iterable[int], y_proba: np.ndarray, confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """
    Compute AUC and its (lower, upper) CI using confidenceinterval.roc_auc_score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = auc(fpr, tpr)
    # ci_roc_auc_score returns (auc, (low, high))
    _, ci = ci_roc_auc_score(y_true, y_proba, confidence_level=confidence)
    return float(auc_val), (float(ci[0]), float(ci[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluators
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, train_X: pd.DataFrame, train_y: pd.Series,
                   valid_X: pd.DataFrame, valid_y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Evaluate a binary classifier on train and valid sets.
    Returns:
        df (pd.DataFrame): rows [train, valid] with AUC, Sensitivity, Specificity, PPV, NPV, CI.
        y_proba_train (np.ndarray): predicted probabilities on train set.
        y_proba_valid (np.ndarray): predicted probabilities on valid set.
    """
    # Predict probabilities
    y_proba_train = _predict_proba_binary(model, train_X)
    y_proba_valid = _predict_proba_binary(model, valid_X)

    # Choose threshold on each split via Youden
    thr_tr = _youden_optimal_threshold(train_y, y_proba_train)
    thr_va = _youden_optimal_threshold(valid_y, y_proba_valid)

    # Train metrics
    auc_tr, ci_tr = _auc_with_ci(train_y, y_proba_train)
    sens_tr, spec_tr, ppv_tr, npv_tr = _binary_metrics_from_threshold(train_y, y_proba_train, thr_tr)

    # Valid metrics
    auc_va, ci_va = _auc_with_ci(valid_y, y_proba_valid)
    sens_va, spec_va, ppv_va, npv_va = _binary_metrics_from_threshold(valid_y, y_proba_valid, thr_va)

    df = pd.DataFrame({
        "AUC":        [round(auc_tr, 3), round(auc_va, 3)],
        "Sensitivity":[round(sens_tr, 3), round(sens_va, 3)],
        "Specificity":[round(spec_tr, 3), round(spec_va, 3)],
        "PPV":        [round(ppv_tr, 3),  round(ppv_va, 3)],
        "NPV":        [round(npv_tr, 3),  round(npv_va, 3)],
        "CI":         [tuple(np.round(ci_tr, 3)), tuple(np.round(ci_va, 3))]
    }, index=["train", "valid"])

    return df, y_proba_train, y_proba_valid


# Clinical-only vs Clinical+SBP
def visualize_roc_curve(model_cln, X_cln: pd.DataFrame, y_cln: pd.Series,
                        model_bp,  X_bp:  pd.DataFrame, y_bp:  pd.Series,
                        p_value, out_path: str | Path) -> Tuple[pd.DataFrame, plt.Figure, plt.Figure]:
    """
    Draw paired ROC curves (clinical-only vs clinical+SBP) and export a figure.
    Returns:
        metrics_df (two rows), fig_cln, fig_bp (single-curve export figures).
    """
    # Probabilities
    proba_cln = _predict_proba_binary(model_cln, X_cln)
    proba_bp  = _predict_proba_binary(model_bp,  X_bp)

    # ROCs
    fpr_c, tpr_c, thr_c = roc_curve(y_cln, proba_cln)
    fpr_b, tpr_b, thr_b = roc_curve(y_bp,  proba_bp)

    auc_c = auc(fpr_c, tpr_c)
    auc_b = auc(fpr_b, tpr_b)

    _, ci_c = ci_roc_auc_score(y_cln, proba_cln, confidence_level=0.95)
    _, ci_b = ci_roc_auc_score(y_bp,  proba_bp,  confidence_level=0.95)

    # Thresholds (Youden) and confusion-matrix-based metrics
    thr_opt_c = thr_c[np.argmax(tpr_c - fpr_c)]
    thr_opt_b = thr_b[np.argmax(tpr_b - fpr_b)]

    sens_c, spec_c, ppv_c, npv_c = _binary_metrics_from_threshold(y_cln, proba_cln, thr_opt_c)
    sens_b, spec_b, ppv_b, npv_b = _binary_metrics_from_threshold(y_bp,  proba_bp,  thr_opt_b)

    metrics_df = pd.DataFrame({
        "AUC":        [round(auc_c, 3),          round(auc_b, 3)],
        "Sensitivity":[round(sens_c, 3),         round(sens_b, 3)],
        "Specificity":[round(spec_c, 3),         round(spec_b, 3)],
        "PPV":        [round(ppv_c, 3),          round(ppv_b, 3)],
        "NPV":        [round(npv_c, 3),          round(npv_b, 3)],
        "CI":         [tuple(np.round(ci_c, 3)), tuple(np.round(ci_b, 3))]
    }, index=["clinical_only", "clinical_plus_SBP"])

    # Combined ROC figure
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    ax.set_aspect("equal", "box")
    ax.plot(fpr_b, tpr_b, lw=1, label=f"Clinical + SBP (AUC = {auc_b:.2f})")
    ax.plot(fpr_c, tpr_c, lw=1, label=f"Clinical only (AUC = {auc_c:.2f})")
    ax.plot([0, 1], [0, 1], color="darkgrey", lw=1, linestyle="--")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("1 - Specificity", fontsize=14)
    ax.set_ylabel("Sensitivity", fontsize=14)
    ax.legend(loc="lower right", frameon=False, fontsize=12)

    # DeLong p-value annotation
    if isinstance(p_value, (np.ndarray, list)) and np.size(p_value) == 1:
        p_value = float(np.array(p_value).ravel()[0])
    if p_value < 0.001:
        ax.text(0.75, 0.5, r"$\it{P} < 0.001$", ha="center", va="center", fontsize=14)
    else:
        ax.text(0.75, 0.5, rf"$\it{{P}}$ = {p_value:.3f}", ha="center", va="center", fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", dpi=200)
    plt.close(fig)

    # Export single-curve figures (for paper-ready panels if needed)
    fig_cln = plt.figure(figsize=(5, 5))
    axc = fig_cln.gca()
    axc.set_aspect("equal", "box")
    axc.plot(fpr_c, tpr_c, lw=1, label=f"DNN (AUC = {auc_c:.2f})")
    axc.plot([0, 1], [0, 1], color="darkgrey", lw=1, linestyle="--")
    axc.legend(frameon=False)
    fig_cln.tight_layout()

    fig_bp = plt.figure(figsize=(5, 5))
    axb = fig_bp.gca()
    axb.set_aspect("equal", "box")
    axb.plot(fpr_b, tpr_b, lw=1, label=f"DNN (AUC = {auc_b:.2f})")
    axb.plot([0, 1], [0, 1], color="darkgrey", lw=1, linestyle="--")
    axb.legend(frameon=False)
    fig_bp.tight_layout()

    return metrics_df, fig_cln, fig_bp


def visualize_roc_comparison(list_of_models: List, X: pd.DataFrame, y: pd.Series,
                             base_fig: plt.Figure, out_path: str | Path) -> None:
    """
    Plot ROC curves of multiple classical ML models on a shared axis
    (overlayed on top of a base figure for consistent styling).
    The first element of `list_of_models` is assumed to be the reference (e.g., DNN)
    and is not re-plotted here.
    """
    # Compute ROCs
    roc_items = []
    for mdl in list_of_models:
        proba = _predict_proba_binary(mdl, X)
        fpr, tpr, _ = roc_curve(y, proba)
        roc_items.append((fpr, tpr, auc(fpr, tpr)))

    # Remove first item (the reference DNN already drawn in base figure)
    roc_items = roc_items[1:]

    labels = ["Decision Tree", "Extra Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    colors = ["limegreen", "limegreen", "dodgerblue", "dodgerblue", "dodgerblue", "dodgerblue"]
    linestyles = ["solid", "dotted", "solid", "dotted", "dashed", "dashdot"]

    # Draw on the provided base figure
    fig = base_fig
    ax = fig.gca()
    for (fpr, tpr, roc_auc), lab, col, ls in zip(roc_items, labels, colors, linestyles):
        ax.plot(fpr, tpr, color=col, linestyle=ls, lw=1, alpha=0.7, label=f"{lab} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], color="darkgrey", lw=1, linestyle="--")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("1 - Specificity", fontsize=14)
    ax.set_ylabel("Sensitivity", fontsize=14)
    ax.legend(loc="lower right", frameon=False, fontsize=12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", dpi=200)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Optional: training curves
# ──────────────────────────────────────────────────────────────────────────────

def visualize_loss_and_accuracy(history, history_bp, out_dir: str | Path = "results/reports") -> None:
    """
    Plot loss/accuracy curves for two histories (clinical-only and clinical+SBP).
    Files are saved in `out_dir` with fixed names for convenience.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clinical-only
    fig1 = plt.figure(figsize=(5, 5))
    ax1 = fig1.gca()
    ax1.plot(history.history["loss"], label="Training Loss")
    ax1.plot(history.history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epochs", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.legend(frameon=False, fontsize=12)
    fig1.tight_layout()
    fig1.savefig(out_dir / "7_LOSS_CLN.svg", format="svg", dpi=200)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(5, 5))
    ax2 = fig2.gca()
    ax2.plot(history.history["accuracy"], label="Training Accuracy")
    ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax2.set_xlabel("Epochs", fontsize=14)
    ax2.set_ylabel("Accuracy", fontsize=14)
    ax2.legend(frameon=False, fontsize=12)
    fig2.tight_layout()
    fig2.savefig(out_dir / "8_ACC_CLN.svg", format="svg", dpi=200)
    plt.close(fig2)

    # Clinical+SBP
    fig3 = plt.figure(figsize=(5, 5))
    ax3 = fig3.gca()
    ax3.plot(history_bp.history["loss"], label="Training Loss")
    ax3.plot(history_bp.history["val_loss"], label="Validation Loss")
    ax3.set_xlabel("Epochs", fontsize=14)
    ax3.set_ylabel("Loss", fontsize=14)
    ax3.legend(frameon=False, fontsize=12)
    fig3.tight_layout()
    fig3.savefig(out_dir / "9_LOSS_ADDED.svg", format="svg", dpi=200)
    plt.close(fig3)

    fig4 = plt.figure(figsize=(5, 5))
    ax4 = fig4.gca()
    ax4.plot(history_bp.history["accuracy"], label="Training Accuracy")
    ax4.plot(history_bp.history["val_accuracy"], label="Validation Accuracy")
    ax4.set_xlabel("Epochs", fontsize=14)
    ax4.set_ylabel("Accuracy", fontsize=14)
    ax4.legend(frameon=False, fontsize=12)
    fig4.tight_layout()
    fig4.savefig(out_dir / "10_ACC_ADDED.svg", format="svg", dpi=200)
    plt.close(fig4)
