# shap_analysis.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tensorflow as tf

import matplotlib
matplotlib.rcParams["font.family"] = "Arial"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Drop non-feature columns that should not be explained."""
    X = X.copy()
    if "multi" in X.columns:
        X = X.drop(columns=["multi"])
    return X


def _guess_feature_names(X: pd.DataFrame) -> list[str]:
    """
    Provide human-friendly feature names.
    If there are 20 columns (clinical-only), use the 20-name mapping.
    Otherwise, use the extended (clinical + SBP summary) mapping, falling back to column names if mismatch.
    """
    cols = list(X.columns)
    if len(cols) == 20:
        return [
            "Age", "Sex", "Hypertension", "Hyperlipidemia", "Smoking", "Previous stroke",
            "CAOD", "Active cancer", "Congestive heart failure", "PAOD", "NIHSS score",
            "Onset to registration", "IV tPA", "DM", "Atrial fibrillation", "Antiplatelet",
            "Anticoagulant", "Hemoglobin", "White blood cell", "Body mass index",
        ]
    # Extended (clinical + SBP summaries)
    extended = [
        "Age", "Sex", "Hypertension", "Hyperlipidemia", "Smoking", "Previous stroke",
        "CAOD", "Active cancer", "Congestive heart failure", "PAOD", "NIHSS score",
        "Onset to registration", "IV tPA", "SBP enroll", "DM", "Atrial fibrillation",
        "Antiplatelet", "Anticoagulant", "Hemoglobin", "White blood cell", "Body mass index",
        "Group", "SBP time rate", "SBP standard deviation",
        "SBP coefficient of variation", "SBP variation independent of the mean",
        "SBP max", "SBP min", "SBP mean",
    ]
    # If length mismatches, just use actual column names to avoid index errors.
    if len(extended) == len(cols):
        return extended
    return cols


def _make_background(X: np.ndarray, max_background: int = 100, method: str = "kmeans", seed: int = 42) -> np.ndarray:
    """
    Create a background dataset for SHAP explainers.
    - "kmeans": cluster centers from data (shap.kmeans)
    - "random": random uniform subsample
    """
    if X.shape[0] <= max_background:
        return X
    if method == "kmeans":
        try:
            return shap.kmeans(X, max_background).data
        except Exception:
            pass
    # Fallback to random subsample
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_background, replace=False)
    return X[idx]


def _choose_explainer(model, X_background: np.ndarray):
    """
    Choose the most stable SHAP explainer depending on model type.
    For tf.keras models: use GradientExplainer(model, background).
    For tree-based models: TreeExplainer.
    Otherwise: KernelExplainer.
    """
    import tensorflow as tf

    # 1) TensorFlow / Keras models → GradientExplainer(model, background)
    if isinstance(model, tf.keras.Model):
        try:
            return shap.GradientExplainer(model, X_background)
        except Exception:
            # Fallback: model-agnostic KernelExplainer
            def predict_fn(X):
                proba = model.predict(X, verbose=0)
                return proba.ravel()
            return shap.KernelExplainer(predict_fn, X_background)

    # 2) Tree-based models (RF/XGB/LGBM/CatBoost) → TreeExplainer
    is_tree_like = any(
        hasattr(model, attr) for attr in ("tree_", "trees_", "get_booster", "get_leaf_output")
    )
    if is_tree_like:
        return shap.TreeExplainer(model)

    # 3) Others → KernelExplainer
    def predict_fn(X):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.ndim > 1 else proba
        pred = model.predict(X)
        return pred.ravel() if hasattr(pred, "ravel") else np.asarray(pred)

    return shap.KernelExplainer(predict_fn, X_background)




# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def plot_of_SHAP(
    model,
    _X: pd.DataFrame,
    y_true: Optional[pd.Series],
    filename: str | Path,
    max_background: int = 100,
    background_method: str = "kmeans",
    max_display: int = 10,
) -> shap.Explainer:
    """
    Compute SHAP values and save a violin-style summary plot.

    Parameters
    ----------
    model : estimator or tf.keras.Model
        Trained binary classifier.
    _X : pd.DataFrame
        Feature frame used for explanation (same preprocessing as training).
    y_true : pd.Series or None
        Not used in the plot but kept for API compatibility.
    filename : str | Path
        SVG file to save the summary plot.
    max_background : int
        Max number of background samples to speed up SHAP (Deep/Kernel).
    background_method : {"kmeans","random"}
        Strategy to build background dataset when subsampling.
    max_display : int
        Max number of features to display in the summary plot.

    Returns
    -------
    explainer : shap.Explainer
        The fitted SHAP explainer (can be reused for dependence plots, etc.).
    """
    # 1) Clean & name features
    X_df = _clean_features(_X)
    feature_names = _guess_feature_names(X_df)

    # 2) Convert to ndarray
    X = X_df.to_numpy()

    # 3) Build background (only used by some explainers)
    X_bg = _make_background(X, max_background=max_background, method=background_method)

    # 4) Choose & fit explainer
    explainer = _choose_explainer(model, X_bg)

    # 5) Compute SHAP values
    # - For DeepExplainer: returns list/array; ensure shape is (n, n_features)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # e.g., for tf models it may return [values]
        shap_values = shap_values[0]
    shap_values = np.squeeze(np.array(shap_values))

    # 6) Plot summary (violin)
    # Note: shap.summary_plot handles DataFrame or array; we pass ndarray + feature_names.
    plt.figure(figsize=(8, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="violin",
        max_display=max_display,
        show=False,
    )
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, format="svg", dpi=200)
    plt.close()

    return explainer
