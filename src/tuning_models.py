# tuning_models.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Our helpers & DNN builder
from model_creation import create_deep_neural_network, default_callbacks


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set global seed across numpy and tensorflow (if available)."""
    import os, random
    import numpy as _np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Search spaces
# ──────────────────────────────────────────────────────────────────────────────

def param_distributions_tree(seed: int = 42) -> Dict[str, List[Any]]:
    return {
        "max_depth": [None, 3, 4, 5, 6, 8, 10, 12],
        "min_samples_split": [2, 4, 6, 8, 12, 16, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8, 12, 16, 18],
        "class_weight": [None, "balanced"],
        "criterion": ["gini"],
    }


def param_distributions_extra_tree(seed: int = 42) -> Dict[str, List[Any]]:
    return {
        "max_depth": [None, 3, 4, 5, 6, 8, 10, 12],
        "min_samples_split": [2, 4, 6, 8, 12, 16, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8, 12, 16, 18],
        "class_weight": [None, "balanced"],
        "criterion": ["gini"],
        "max_features": ["sqrt"],
        "splitter": ["random"],
    }


def param_distributions_rf(seed: int = 42) -> Dict[str, List[Any]]:
    return {
        "n_estimators": [10, 100, 200, 300, 500],
        "max_depth": [None, 4, 6, 8, 10, 12],
        "min_samples_split": [2, 4, 6, 8, 12],
        "min_samples_leaf": [1, 2, 4, 6, 8, 12],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"],
        "bootstrap": [True],
        "criterion": ["gini"],
    }


def param_distributions_xgb(seed: int = 42) -> Dict[str, List[Any]]:
    return {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "gamma": [0.0, 0.1, 0.3],
    }


def param_distributions_lgbm(seed: int = 42) -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 300, 600, 1000],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [-1, 4, 6, 8, 10],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "min_child_samples": [5, 10, 20, 30],
    }


def param_distributions_cat(seed: int = 42) -> Dict[str, List[Any]]:
    return {
        "iterations": [300, 600, 1000],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "depth": [4, 6, 8, 10, 12],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
        "border_count": [32, 64, 128],
        "subsample": [0.8, 1.0],
        "grow_policy": ["SymmetricTree"],
        "leaf_estimation_method": ["Newton"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single-model tuning wrappers (sklearn API)
# ──────────────────────────────────────────────────────────────────────────────

def _cv(seed: int = 42, n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _fit_random_search(
    estimator,
    param_distributions: Dict[str, List[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_iter: int,
    n_splits: int,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=_cv(seed, n_splits),
        verbose=1,
        n_jobs=n_jobs,
        random_state=seed,
        refit=True,
    )
    search.fit(X, y)
    return search


def tune_decision_tree(X, y, seed=42, n_iter=30, n_splits=5):
    est = DecisionTreeClassifier(random_state=seed)
    return _fit_random_search(est, param_distributions_tree(seed), X, y, seed, n_iter, n_splits)

def tune_extra_tree(X, y, seed=42, n_iter=30, n_splits=5):
    est = ExtraTreeClassifier(random_state=seed)
    return _fit_random_search(est, param_distributions_extra_tree(seed), X, y, seed, n_iter, n_splits)

def tune_random_forest(X, y, seed=42, n_iter=40, n_splits=5):
    est = RandomForestClassifier(random_state=seed, n_jobs=-1)
    return _fit_random_search(est, param_distributions_rf(seed), X, y, seed, n_iter, n_splits)

def tune_xgboost(X, y, seed=42, n_iter=50, n_splits=5):
    est = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="auto",
        random_state=seed,
        nthread=-1,
        use_label_encoder=False
    )
    return _fit_random_search(est, param_distributions_xgb(seed), X, y, seed, n_iter, n_splits)

def tune_lightgbm(X, y, seed=42, n_iter=50, n_splits=5):
    est = LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_jobs=-1,
    )
    return _fit_random_search(est, param_distributions_lgbm(seed), X, y, seed, n_iter, n_splits)

def tune_catboost(X, y, seed=42, n_iter=40, n_splits=5):
    est = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=seed,
        verbose=0,
        thread_count=-1,
    )
    return _fit_random_search(est, param_distributions_cat(seed), X, y, seed, n_iter, n_splits)


# ──────────────────────────────────────────────────────────────────────────────
# Public API: tune selected models and persist results
# ──────────────────────────────────────────────────────────────────────────────

def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: Optional[pd.DataFrame] = None,
    y_valid: Optional[pd.Series] = None,
    out_dir: Path = Path("results"),
    seed: int = 42,
    include_dnn: bool = True,
    dnn_epochs: int = 120,
    dnn_batch_size: int = 64,
    class_weight: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning for classical models (+ optionally DNN), save artifacts.
    Returns a summary dict with best scores/params/paths.
    """
    set_seed(seed)
    reports_dir = out_dir / "best_params"
    models_dir = out_dir / "models"
    ensure_dirs(reports_dir, models_dir)

    summary: Dict[str, Any] = {}

    # Decision Tree
    s = tune_decision_tree(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_decision_tree.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "decision_tree.json")
    summary["DecisionTree"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_decision_tree.joblib")}

    # Extra Tree
    s = tune_extra_tree(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_extra_tree.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "extra_tree.json")
    summary["ExtraTree"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_extra_tree.joblib")}

    # Random Forest
    s = tune_random_forest(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_random_forest.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "random_forest.json")
    summary["RandomForest"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_random_forest.joblib")}

    # XGBoost
    s = tune_xgboost(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_xgboost.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "xgboost.json")
    summary["XGBoost"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_xgboost.joblib")}

    # LightGBM
    s = tune_lightgbm(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_lightgbm.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "lightgbm.json")
    summary["LightGBM"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_lightgbm.joblib")}

    # CatBoost
    s = tune_catboost(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_catboost.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "catboost.json")
    summary["CatBoost"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_catboost.joblib")}

    # Global summary
    save_json(summary, reports_dir / "summary.json")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Example CLI (optional)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Minimal example:
        python -m tuning_models
    This block is optional; integrate from run_train.py as needed.
    """
    print("This module is intended to be imported and called from run_train.py.")
