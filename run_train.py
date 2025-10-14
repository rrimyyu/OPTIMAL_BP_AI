# run_train.py
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Classical ML
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Gradient boosting families
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Stats & evaluation utilities
from compare_auc_delong import delong_roc_test
from model_creation import create_deep_neural_network, default_callbacks
from model_evaluation import (
    evaluate_model,
    visualize_roc_curve,
    visualize_roc_comparison,
    visualize_loss_and_accuracy,
)
from shap_analysis import plot_of_SHAP

# Our preprocessing pipeline (coverage-aware, phase-resampled)
from data_preprocessing import Config, build_dataset, drop_bp_features


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set global random seed for reproducibility (NumPy / TF)."""
    import os, random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def enable_gpu_memory_growth() -> None:
    """Enable TF GPU memory growth if a GPU is present (prevents TF from grabbing all VRAM)."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def load_train_indices_by_regno(df: pd.DataFrame, regno_list: list[float]) -> list[int]:
    """Map a list of registration numbers to row indices in the feature frame."""
    if "optimal_bp_reg_no" not in df.columns:
        return []
    return df[df["optimal_bp_reg_no"].isin(regno_list)].index.tolist()


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_indices: list[int] | None = None,
    test_size: float = 0.3,
    seed: int = 42,  # kept for signature compatibility; not used when random_state=None
):
    """
    Produce train/valid splits either by provided row indices or by stratified split.
    NOTE: random_state=None → a different split on each run.
    """
    if train_indices:
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_valid = X.drop(train_indices, errors="ignore")
        y_valid = y.drop(train_indices, errors="ignore")
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=None, stratify=y
        )
    return X_train, X_valid, y_train, y_valid


def save_json(obj, path: Path):
    """Save a dict-like object to JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    # -------------------------
    # CLI
    # -------------------------
    parser = argparse.ArgumentParser(
        description="Train clinical-only and clinical+SBP models, evaluate, and export reports."
    )
    parser.add_argument("--excel-path", type=str, required=True, help="Path to the OPTIMAL-BP Excel workbook.")
    parser.add_argument("--use-pp", action="store_true", help="Use Per-Protocol (PP=1) instead of ITT.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.30, help="Validation split ratio.")
    parser.add_argument("--train-regno-json", type=str, default="", help="JSON file of training registration numbers.")
    parser.add_argument("--epochs", type=int, default=300, help="Epochs for DNN.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DNN.")
    parser.add_argument("--class-weight", type=str, default="", choices=["", "auto"],
                        help="Use class_weight='auto' for imbalance.")
    args = parser.parse_args(argv)

    enable_gpu_memory_growth()

    # -------------------------
    # Build dataset via our preprocessing pipeline
    # -------------------------
    out_dir = Path(args.results_dir)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # keep_phase_columns=False so meta/array columns never leak into X
    cfg = Config(
        excel_path=Path(args.excel_path),
        use_per_protocol=args.use_pp,
        out_dir=out_dir,
        keep_phase_columns=False,
    )

    cs_df, is_df, bp_df, meta = build_dataset(cfg)

    # Snapshot of the combined feature table
    bp_snapshot_path = out_dir / "reports" / "bp_cln_df.xlsx"
    bp_df.to_excel(bp_snapshot_path, index=False)

    # -------------------------
    # Target & features
    # -------------------------
    y = bp_df["mRS_3months"].astype(int)
    feature_cols = [c for c in bp_df.columns if c not in ["multi", "mRS_3months"]]
    X = bp_df[feature_cols].copy()

    # Defensive: drop any leftover meta/series columns
    meta_like = [c for c in X.columns if c.startswith("__")] + ["chosen_phase"]
    if meta_like:
        X = X.drop(columns=[c for c in meta_like if c in X.columns], errors="ignore")

    # Optional: map provided registration numbers to indices
    train_indices = None
    if args.train_regno_json:
        regnos = json.loads(Path(args.train_regno_json).read_text(encoding="utf-8"))
        train_indices = load_train_indices_by_regno(bp_df, regnos)

    # Drop patient ID before training
    if "optimal_bp_reg_no" in X.columns:
        X = X.drop(columns=["optimal_bp_reg_no"])

    # -------------------------
    # Split (random on each run because random_state=None in split_data)
    # -------------------------
    X_train_full, X_valid_full, y_train, y_valid = split_data(
        X, y, train_indices=train_indices, test_size=args.test_size, seed=None
    )

    # Re-seed AFTER the split so model training is reproducible while split is random
    set_seed(args.seed)
    enable_gpu_memory_growth()

    # Clinical-only features (remove canonical BP/BPV features)
    X_train_cln = drop_bp_features(X_train_full.copy())
    X_valid_cln = drop_bp_features(X_valid_full.copy())

    # Clinical + SBP: keep ALL BP-related features
    X_train_cln_sbp = X_train_full.copy()
    X_valid_cln_sbp = X_valid_full.copy()

    # Class weights (optional)
    class_weight = None
    if args.class_weight == "auto":
        # inverse frequency weights for binary classification
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        w0 = (pos + neg) / (2.0 * max(neg, 1.0))
        w1 = (pos + neg) / (2.0 * max(pos, 1.0))
        class_weight = {0: w0, 1: w1}

    # -------------------------
    # Train neural networks
    # -------------------------
    model_cln = create_deep_neural_network(X_train_cln)
    hist_cln = model_cln.fit(
        X_train_cln, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_valid_cln, y_valid),
        callbacks=default_callbacks(str(out_dir / "models")),
        class_weight=class_weight,
        verbose=0
    )
    model_cln.save(out_dir / "models" / "cln_only_model.h5")

    df_cln, yhat_cln_train, yhat_cln_valid = evaluate_model(
        model_cln, X_train_cln, y_train, X_valid_cln, y_valid
    )

    model_cln_sbp = create_deep_neural_network(X_train_cln_sbp)
    hist_cln_sbp = model_cln_sbp.fit(
        X_train_cln_sbp, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_valid_cln_sbp, y_valid),
        callbacks=default_callbacks(str(out_dir / "models")),
        class_weight=class_weight,
        verbose=0
    )
    model_cln_sbp.save(out_dir / "models" / "cln_sbp_model.h5")

    df_cln_sbp, yhat_cln_sbp_train, yhat_cln_sbp_valid = evaluate_model(
        model_cln_sbp, X_train_cln_sbp, y_train, X_valid_cln_sbp, y_valid
    )

    # DeLong p-values (NN vs NN+SBP)
    p_train = delong_roc_test(y_train.to_numpy(), yhat_cln_train, yhat_cln_sbp_train)
    p_valid = delong_roc_test(y_valid.to_numpy(), yhat_cln_valid, yhat_cln_sbp_valid)

    # ROC curves for train/valid
    df_train_2, _plt1a, _plt1b = visualize_roc_curve(
        model_cln, X_train_cln, y_train,
        model_cln_sbp, X_train_cln_sbp, y_train,
        p_train, str(out_dir / "reports" / "1_ROC_CURVE_TRAIN.svg")
    )
    df_valid_2, plt2a, plt2b = visualize_roc_curve(
        model_cln, X_valid_cln, y_valid,
        model_cln_sbp, X_valid_cln_sbp, y_valid,
        p_valid, str(out_dir / "reports" / "2_ROC_CURVE_VALID.svg")
    )

    # SHAP (note: with TF2 you may prefer GradientExplainer if DeepExplainer has issues)
    plot_of_SHAP(model_cln, pd.concat([X_train_cln, X_valid_cln]), y, str(out_dir / "reports" / "5_SHAP_CLN.svg"))
    plot_of_SHAP(model_cln_sbp, pd.concat([X_train_cln_sbp, X_valid_cln_sbp]), y, str(out_dir / "reports" / "6_SHAP_ADDED.svg"))
    plot_of_SHAP(model_cln_sbp, X_valid_cln_sbp, y_valid, str(out_dir / "reports" / "6-1_SHAP_ADDED_valid.svg"))

    # Learning curves
    visualize_loss_and_accuracy(hist_cln, hist_cln_sbp, out_dir=out_dir / "reports")

    # -------------------------
    # Classical ML baselines (clinical-only)
    # -------------------------
    models_cln = {
        "TR":  DecisionTreeClassifier(max_depth=6, min_samples_leaf=18, min_samples_split=8),
        "ETR": ExtraTreeClassifier(max_depth=8, min_samples_leaf=12, min_samples_split=20),
        "RF":  RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=12, min_samples_split=8),
        "XGB": xgb.XGBClassifier(
            learning_rate=0.1, max_depth=10, n_estimators=300,
            subsample=0.9, colsample_bytree=0.8, eval_metric="logloss"
        ),
        "LGBM": LGBMClassifier(learning_rate=0.01, max_depth=6, n_estimators=500),
        "CAT": CatBoostClassifier(learning_rate=0.05, depth=8, iterations=500, verbose=0),
    }

    df_baselines_cln = []
    for name, mdl in models_cln.items():
        mdl.fit(X_train_cln, y_train)
        df_tmp, _, _ = evaluate_model(mdl, X_train_cln, y_train, X_valid_cln, y_valid)
        df_tmp.insert(0, "Model", f"{name}_CLN")
        df_baselines_cln.append(df_tmp)

    # -------------------------
    # Classical ML baselines (clinical + SBP summary)
    # -------------------------
    models_bp = {
        "TR":  DecisionTreeClassifier(max_depth=6, min_samples_leaf=18, min_samples_split=8),
        "ETR": ExtraTreeClassifier(max_depth=8, min_samples_leaf=12, min_samples_split=20),
        "RF":  RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=12, min_samples_split=8),
        "XGB": xgb.XGBClassifier(
            learning_rate=0.1, max_depth=10, n_estimators=300,
            subsample=0.9, colsample_bytree=0.8, eval_metric="logloss"
        ),
        "LGBM": LGBMClassifier(learning_rate=0.01, max_depth=6, n_estimators=500),
        "CAT": CatBoostClassifier(learning_rate=0.05, depth=8, iterations=500, verbose=0),
    }

    df_baselines_bp = []
    for name, mdl in models_bp.items():
        mdl.fit(X_train_cln_sbp, y_train)
        df_tmp, _, _ = evaluate_model(mdl, X_train_cln_sbp, y_train, X_valid_cln_sbp, y_valid)
        df_tmp.insert(0, "Model", f"{name}_CLN+SBP")
        df_baselines_bp.append(df_tmp)

    # -------------------------
    # ROC comparison plots
    # -------------------------
    visualize_roc_comparison(
        [model_cln] + [m for m in models_cln.values()],
        X_valid_cln, y_valid, plt2a,
        str(out_dir / "reports" / "3_ROC_CURVE_CLN.svg")
    )
    visualize_roc_comparison(
        [model_cln_sbp] + [m for m in models_bp.values()],
        X_valid_cln_sbp, y_valid, plt2b,
        str(out_dir / "reports" / "4_ROC_CURVE_ADDED.svg")
    )

    # -------------------------
    # Export combined table
    # -------------------------
    # Use reset_index to ensure first two rows exist as 0/1 positions
    df_nn_train = df_train_2.reset_index(drop=True)
    df_nn_valid = df_valid_2.reset_index(drop=True)

    out_table = pd.concat(
        [df_nn_train, df_nn_valid] + df_baselines_cln + df_baselines_bp,
        ignore_index=True
    ).round(3)

    out_excel = out_dir / "reports" / "output.xlsx"
    out_table.to_excel(out_excel, index=False)

    # -------------------------
    # Minimal run summary
    # -------------------------
    summary = {
        "seed": args.seed,
        "test_size": args.test_size,
        "use_per_protocol": args.use_pp,
        "excel_path": str(Path(args.excel_path).resolve()),
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "p_value_train": float(p_train),
        "p_value_valid": float(p_valid),
        "reports": str((out_dir / "reports").resolve()),
        "models": str((out_dir / "models").resolve()),
        "class_weight": args.class_weight or "none",
    }
    save_json(summary, out_dir / "reports" / "run_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
