from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from joblib import dump
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def set_seed(seed: int = 42) -> None:
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


def _cv(seed: int = 42, n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def param_grid_decision_tree() -> Dict[str, List[Any]]:
    return {
        "criterion": ["gini"],
        "splitter": ["best"],
        "max_depth": [4, 6, 8],
        "min_samples_split": [6, 8, 12],
        "min_samples_leaf": [12, 18],
        "class_weight": [None, "balanced"],
    }


def param_grid_extra_tree() -> Dict[str, List[Any]]:
    return {
        "criterion": ["gini"],
        "splitter": ["random"],
        "max_features": ["sqrt"],
        "max_depth": [6, 8, 10],
        "min_samples_split": [12, 20],
        "min_samples_leaf": [8, 12, 16],
        "class_weight": [None, "balanced"],
    }


def param_grid_random_forest() -> Dict[str, List[Any]]:
    return {
        "criterion": ["gini"],
        "bootstrap": [True],
        "n_estimators": [10, 100, 200],
        "max_depth": [4, 6, 8],
        "min_samples_split": [6, 8, 12],
        "min_samples_leaf": [8, 12, 16],
        "max_features": ["sqrt"],
        "class_weight": [None, "balanced"],
    }


def param_grid_xgboost() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.05, 0.1],
        "max_depth": [8, 10],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [0.0, 1.0],
        "reg_alpha": [0.0, 0.1],
        "gamma": [0.0, 0.1],
    }


def param_grid_lightgbm() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 300],
        "learning_rate": [0.01, 0.05],
        "max_depth": [4, 6, 8],
        "num_leaves": [31, 63],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [0.0, 1.0],
        "reg_alpha": [0.0, 0.1],
        "min_child_samples": [10, 20, 30],
    }


def param_grid_catboost() -> Dict[str, List[Any]]:
    return {
        "iterations": [600, 1000],
        "learning_rate": [0.03, 0.05],
        "depth": [10, 12],
        "leaf_estimation_method": ["Newton"],
        "grow_policy": ["SymmetricTree"],
        "subsample": [0.8, 1.0],
        "l2_leaf_reg": [3.0, 5.0],
        "border_count": [64, 128],
    }


def _fit_grid_search(
    estimator,
    param_grid: Dict[str, List[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_splits: int,
    n_jobs: int = -1,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=_cv(seed, n_splits),
        verbose=1,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)
    return search


def tune_decision_tree(X, y, seed: int = 42, n_splits: int = 5) -> GridSearchCV:
    est = DecisionTreeClassifier(random_state=seed)
    return _fit_grid_search(est, param_grid_decision_tree(), X, y, seed, n_splits)


def tune_extra_tree(X, y, seed: int = 42, n_splits: int = 5) -> GridSearchCV:
    est = ExtraTreeClassifier(random_state=seed)
    return _fit_grid_search(est, param_grid_extra_tree(), X, y, seed, n_splits)


def tune_random_forest(X, y, seed: int = 42, n_splits: int = 5) -> GridSearchCV:
    est = RandomForestClassifier(random_state=seed, n_jobs=-1)
    return _fit_grid_search(est, param_grid_random_forest(), X, y, seed, n_splits)


def tune_xgboost(X, y, seed: int = 42, n_splits: int = 5) -> GridSearchCV:
    est = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="auto",
        random_state=seed,
        nthread=-1,
        use_label_encoder=False,
    )
    return _fit_grid_search(est, param_grid_xgboost(), X, y, seed, n_splits)


def tune_lightgbm(X, y, seed: int = 42, n_splits: int = 5) -> GridSearchCV:
    est = LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_jobs=-1,
    )
    return _fit_grid_search(est, param_grid_lightgbm(), X, y, seed, n_splits)


def tune_catboost(X, y, seed: int = 42, n_splits: int = 5) -> GridSearchCV:
    est = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=seed,
        verbose=0,
        thread_count=-1,
    )
    return _fit_grid_search(est, param_grid_catboost(), X, y, seed, n_splits)


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    out_dir: Path = Path("results"),
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    reports_dir = out_dir / "best_params"
    models_dir = out_dir / "models"
    ensure_dirs(reports_dir, models_dir)

    summary: Dict[str, Any] = {}

    s = tune_decision_tree(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_decision_tree.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "decision_tree.json")
    summary["DecisionTree"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_decision_tree.joblib")}

    s = tune_extra_tree(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_extra_tree.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "extra_tree.json")
    summary["ExtraTree"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_extra_tree.joblib")}

    s = tune_random_forest(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_random_forest.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "random_forest.json")
    summary["RandomForest"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_random_forest.joblib")}

    s = tune_xgboost(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_xgboost.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "xgboost.json")
    summary["XGBoost"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_xgboost.joblib")}

    s = tune_lightgbm(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_lightgbm.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "lightgbm.json")
    summary["LightGBM"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_lightgbm.joblib")}

    s = tune_catboost(X_train, y_train, seed=seed)
    dump(s.best_estimator_, models_dir / "best_catboost.joblib")
    save_json({"best_params": s.best_params_, "best_auc_cv": float(s.best_score_)}, reports_dir / "catboost.json")
    summary["CatBoost"] = {"auc_cv": float(s.best_score_), "params": s.best_params_, "model": str(models_dir / "best_catboost.joblib")}

    save_json(summary, reports_dir / "summary.json")
    return summary


if __name__ == "__main__":
    print("This module is intended to be imported and called from run_train.py.")
