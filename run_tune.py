# run_tune.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from data_preprocessing import Config, build_dataset, drop_bp_features
from tuning_models import tune_all_models


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


def split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    seed: int = 42,
):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def compute_class_weight(y_train: pd.Series) -> Dict[int, float]:
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    w0 = (pos + neg) / (2.0 * neg + 1e-12)
    w1 = (pos + neg) / (2.0 * pos + 1e-12)
    return {0: w0, 1: w1}


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for clinical-only and/or clinical+SBP models."
    )
    parser.add_argument("--excel-path", type=str, required=True, help="Path to the OPTIMAL-BP Excel workbook.")
    parser.add_argument("--use-pp", action="store_true", help="Use Per-Protocol (PP=1) instead of ITT.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.30, help="Validation split ratio.")
    parser.add_argument("--class-weight", type=str, default="", choices=["", "auto"], help="Use class_weight='auto'.")
    parser.add_argument("--include-dnn", action="store_true", help="Also tune a Keras DNN on top of classical models.")
    parser.add_argument("--dnn-epochs", type=int, default=80, help="Max epochs per DNN trial.")
    parser.add_argument("--dnn-batch-size", type=int, default=64, help="Batch size for DNN tuning.")
    parser.add_argument(
        "--which",
        type=str,
        default="both",
        choices=["clinical", "bp", "both"],
        help="Which feature set to tune: clinical-only, clinical+SBP (bp), or both.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.results_dir)
    (out_dir / "best_params").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # 1) Build dataset
    cfg = Config(
        excel_path=Path(args.excel_path),
        use_per_protocol=args.use_pp,
        out_dir=out_dir,
    )
    cs_df, is_df, bp_df, meta = build_dataset(cfg)

    # 2) Target & base features
    y = bp_df["mRS_3months"].astype(int)
    X_base = bp_df[[c for c in bp_df.columns if c not in ["multi", "mRS_3months", "optimal_bp_reg_no"]]].copy()

    # 3) Split once for fair model comparison
    X_train_full, X_valid_full, y_train, y_valid = split_stratified(
        X_base, y, test_size=args.test_size, seed=args.seed
    )

    # 4) Feature subsets
    # clinical-only (drop BP & BPV summary)
    X_train_cln = drop_bp_features(X_train_full.copy())
    X_valid_cln = drop_bp_features(X_valid_full.copy())

    # clinical + SBP summary (keep max/min, drop variability)
    drop_vary = ["systolic_mean", "systolic_SD", "systolic_CV", "systolic_VIM"]
    X_train_bp = X_train_full.drop(columns=[c for c in drop_vary if c in X_train_full.columns], errors="ignore")
    X_valid_bp = X_valid_full.drop(columns=[c for c in drop_vary if c in X_valid_full.columns], errors="ignore")

    # 5) Class weight
    class_weight = compute_class_weight(y_train) if args.class_weight == "auto" else None

    summaries = {}

    # 6) Tune clinical-only
    if args.which in ("clinical", "both"):
        print("\n[run_tune] Tuning: Clinical-only features")
        summaries["clinical"] = tune_all_models(
            X_train=X_train_cln,
            y_train=y_train,
            X_valid=X_valid_cln,
            y_valid=y_valid,
            out_dir=out_dir / "clinical",
            seed=args.seed,
            include_dnn=args.include_dnn,
            dnn_epochs=args.dnn_epochs,
            dnn_batch_size=args.dnn_batch_size,
            class_weight=class_weight,
        )

    # 7) Tune clinical + SBP summary
    if args.which in ("bp", "both"):
        print("\n[run_tune] Tuning: Clinical + SBP summary features")
        summaries["bp"] = tune_all_models(
            X_train=X_train_bp,
            y_train=y_train,
            X_valid=X_valid_bp,
            y_valid=y_valid,
            out_dir=out_dir / "bp",
            seed=args.seed,
            include_dnn=args.include_dnn,
            dnn_epochs=args.dnn_epochs,
            dnn_batch_size=args.dnn_batch_size,
            class_weight=class_weight,
        )

    # 8) Save a combined summary
    summary_path = out_dir / "best_params" / "tuning_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[run_tune] Done. Summary saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
