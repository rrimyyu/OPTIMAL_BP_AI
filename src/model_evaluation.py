import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    precision_score,
    brier_score_loss,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from confidenceinterval import roc_auc_score as roc_auc_ci  # assumed to return (auc, ci)
from compare_auc_delong import delong_roc_test

plt.rcParams["font.family"] = "Arial"


# =============================
# Basic evaluation (Train/Valid)
# =============================
def _predict_proba_any(model, X):
    """Supports both sklearn / keras: returns positive-class probability (1D)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    else:
        Xv = X.to_numpy() if hasattr(X, "to_numpy") else X
        proba = model.predict(Xv)

    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba.ravel()
    if proba.shape[1] == 1:
        return proba[:, 0].ravel()
    return proba[:, 1].ravel()


# =============================
# DCA helpers
# =============================
def compute_net_benefit(y_true, y_prob, pt_grid):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

    N = len(y_true)
    event_rate = y_true.mean()

    nb_model = np.zeros_like(pt_grid, dtype=float)
    nb_all = np.zeros_like(pt_grid, dtype=float)
    nb_none = np.zeros_like(pt_grid, dtype=float)

    for i, pt in enumerate(pt_grid):
        w = pt / (1 - pt)
        y_hat = (y_prob >= pt).astype(int)
        TP = np.sum((y_true == 1) & (y_hat == 1))
        FP = np.sum((y_true == 0) & (y_hat == 1))

        nb_model[i] = (TP / N) - (FP / N) * w
        nb_all[i] = event_rate - (1 - event_rate) * w
        nb_none[i] = 0.0

    return nb_model, nb_all, nb_none


def bootstrap_pvalue_brier(y_true, prob_a, prob_b, n_boot=2000, seed=42):
    """
    Paired bootstrap p-value for difference in Brier score.
    H0: BS_a - BS_b = 0
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    prob_a = np.asarray(prob_a).astype(float)
    prob_b = np.asarray(prob_b).astype(float)

    if not (len(y_true) == len(prob_a) == len(prob_b)):
        raise ValueError("y_true, prob_a, prob_b must have the same length (paired samples).")

    n = len(y_true)
    diffs = np.empty(n_boot, dtype=float)
    obs_diff = brier_score_loss(y_true, prob_a) - brier_score_loss(y_true, prob_b)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = brier_score_loss(y_true[idx], prob_a[idx]) - brier_score_loss(y_true[idx], prob_b[idx])

    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(np.clip(p, 0.0, 1.0)), float(obs_diff)


def visualize_calibration_curve(
    model_cln,
    X_cln,
    y_cln,
    model_bp,
    X_bp,
    y_bp,
    label_cln="Clinical only",
    label_bp="Clinical & SBP metrics",
    filename=None,
):
    y_proba_cln = _predict_proba_any(model_cln, X_cln)
    y_proba_bp = _predict_proba_any(model_bp, X_bp)

    y_cln = np.asarray(y_cln).astype(int)
    y_bp = np.asarray(y_bp).astype(int)

    brier_cln = brier_score_loss(y_cln, y_proba_cln)
    brier_bp = brier_score_loss(y_bp, y_proba_bp)

    print(f"[{label_cln}] Brier score = {brier_cln:.3f}")
    print(f"[{label_bp}]  Brier score = {brier_bp:.3f}")

    p_value_bs, obs_bs_diff = bootstrap_pvalue_brier(
        y_true=y_cln,
        prob_a=y_proba_bp,
        prob_b=y_proba_cln,
        n_boot=2000,
        seed=42,
    )
    print(f"ΔBS ({label_bp} - {label_cln}) = {obs_bs_diff:.4f}, P = {p_value_bs:.4f}")

    prob_true_cln, prob_pred_cln = calibration_curve(y_cln, y_proba_cln, n_bins=10)
    prob_true_bp, prob_pred_bp = calibration_curve(y_bp, y_proba_bp, n_bins=10)

    plt.figure(figsize=(5, 5))
    plt.plot(
        prob_pred_bp,
        prob_true_bp,
        marker="o",
        linestyle="-",
        color="r",
        label=f"{label_bp}(Brier = {brier_bp:.2f})",
    )
    plt.plot(
        prob_pred_cln,
        prob_true_cln,
        marker="o",
        linestyle="-",
        color="c",
        label=f"{label_cln}(Brier = {brier_cln:.2f})",
    )
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")

    if p_value_bs < 0.001:
        plt.text(
            0.75,
            0.25,
            r"$\it{P}$ (Brier) < 0.001",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )
    else:
        plt.text(
            0.75,
            0.25,
            f"$\\it{{P}}$ (Brier) = {p_value_bs:.3f}",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )

    plt.xlabel("Predicted probability", fontsize=16)
    plt.ylabel("Observed event rate", fontsize=16)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(loc="upper left", frameon=False, fontsize=10)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, format=filename.split(".")[-1])

    plt.show()

    return pd.DataFrame({"Model": [label_cln, label_bp], "Brier_score": [brier_cln, brier_bp]})


# =============================
# Bootstrap optimism (model_fn)
# =============================
def bootstrap_optimism_model_fn(
    model_fn,
    X,
    y,
    B=500,
    random_state=42,
    fit_kwargs=None,
    verbose_every=0,
):
    """
    Bootstrap optimism correction for AUC (internal validation).
    model_fn: returns a fresh model instance.
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int).ravel()

    rng = np.random.default_rng(random_state)
    n = len(y)

    base_model = model_fn()
    base_model.fit(X, y, **fit_kwargs)
    yhat_base = np.asarray(base_model.predict(X)).ravel()
    auc_apparent = float(roc_auc_score(y, yhat_base))

    optimism_list, apparent_list, test_list = [], [], []

    for b in range(B):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]

        if len(np.unique(yb)) < 2:
            continue

        m = model_fn()
        m.fit(Xb, yb, **fit_kwargs)

        yhat_b = np.asarray(m.predict(Xb)).ravel()
        auc_boot = float(roc_auc_score(yb, yhat_b))

        yhat_o = np.asarray(m.predict(X)).ravel()
        auc_orig = float(roc_auc_score(y, yhat_o))

        optimism = auc_boot - auc_orig
        optimism_list.append(optimism)
        apparent_list.append(auc_boot)
        test_list.append(auc_orig)

        if verbose_every and (b + 1) % verbose_every == 0:
            print(f"[bootstrap {b+1}/{B}] auc_boot={auc_boot:.4f}, auc_orig={auc_orig:.4f}, opt={optimism:.4f}")

    optimism_arr = np.array(optimism_list, dtype=float)
    mean_optimism = float(np.mean(optimism_arr)) if len(optimism_arr) else np.nan
    auc_corrected = float(auc_apparent - mean_optimism)

    return {
        "auc_apparent": auc_apparent,
        "auc_optimism_mean": mean_optimism,
        "auc_corrected": auc_corrected,
        "optimism_dist": optimism_arr,
        "apparent_dist": np.array(apparent_list, dtype=float),
        "test_dist": np.array(test_list, dtype=float),
        "n_effective_bootstrap": int(len(optimism_arr)),
    }


# =============================
# Validation ROC (DeLong)
# =============================
def visualize_roc_curve_valid_with_delong(
    model,
    X_valid,
    y_valid_mrs,
    model_bp,
    X_bp_valid,
    y_bp_valid_mrs,
    p_value_valid=None,
    filename="ROC_VALID_DELONG.svg",
):
    y_cln = np.asarray(y_valid_mrs).astype(int).ravel()
    y_bp = np.asarray(y_bp_valid_mrs).astype(int).ravel()

    if len(y_cln) != len(y_bp):
        raise ValueError("Validation sample size differs between models.")
    if not np.array_equal(y_cln, y_bp):
        raise ValueError("y_valid_mrs and y_bp_valid_mrs differ. Ensure identical samples and ordering.")

    y_proba_cln = _predict_proba_any(model, X_valid)
    y_proba_bp = _predict_proba_any(model_bp, X_bp_valid)

    if p_value_valid is None:
        p_value_valid = delong_roc_test(y_cln, y_proba_cln, y_proba_bp)

    p_scalar = float(p_value_valid[0, 0]) if hasattr(p_value_valid, "shape") else float(p_value_valid)

    fpr_cln, tpr_cln, _ = roc_curve(y_cln, y_proba_cln)
    fpr_bp, tpr_bp, _ = roc_curve(y_cln, y_proba_bp)

    auc_cln = float(roc_auc_score(y_cln, y_proba_cln))
    auc_bp = float(roc_auc_score(y_cln, y_proba_bp))

    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_aspect("equal", "datalim")

    plt.plot(fpr_bp, tpr_bp, color="r", lw=2, label=f"Clinical & SBP metrics (AUC = {auc_bp:.2f})")
    plt.plot(fpr_cln, tpr_cln, color="c", lw=2, label=f"Clinical only (AUC = {auc_cln:.2f})")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--", color="darkgrey")

    txt = r"$\it{P} < 0.001$" if p_scalar < 0.001 else f"$\\it{{P}}$ = {p_scalar:.3f}"
    plt.text(0.75, 0.5, txt, transform=ax.transAxes, ha="center", va="center", fontsize=16)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("1 - Specificity", fontsize=16)
    plt.ylabel("Sensitivity", fontsize=16)
    plt.legend(loc="lower right", frameon=False, fontsize=12)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, format=filename.split(".")[-1])
    plt.show()

    return p_value_valid, auc_cln, auc_bp


def visualize_roc_curve_valid_param_delong(
    model_full,
    X_full_valid,
    y_full_valid,
    model_red,
    X_red_valid,
    y_red_valid,
    n_full,
    n_reduced,
    label_prefix="Clinical only",
    p_value_valid=None,
    filename="ROC_VALID_PARAM_DELONG.svg",
):
    y_full = np.asarray(y_full_valid).astype(int).ravel()
    y_red = np.asarray(y_red_valid).astype(int).ravel()

    if len(y_full) != len(y_red):
        raise ValueError("Validation sample size differs between models.")
    if not np.array_equal(y_full, y_red):
        raise ValueError("y_full_valid and y_red_valid differ. Ensure identical samples and ordering.")

    y_proba_full = _predict_proba_any(model_full, X_full_valid)
    y_proba_red = _predict_proba_any(model_red, X_red_valid)

    if p_value_valid is None:
        p_value_valid = delong_roc_test(y_full, y_proba_full, y_proba_red)

    p_scalar = float(p_value_valid[0, 0]) if hasattr(p_value_valid, "shape") else float(p_value_valid)

    fpr_full, tpr_full, _ = roc_curve(y_full, y_proba_full)
    fpr_red, tpr_red, _ = roc_curve(y_full, y_proba_red)

    auc_full = float(roc_auc_score(y_full, y_proba_full))
    auc_red = float(roc_auc_score(y_full, y_proba_red))

    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_aspect("equal", "datalim")

    plt.plot(
        fpr_red,
        tpr_red,
        color="r",
        lw=2,
        label=f"{label_prefix} ({n_reduced} features, AUC = {auc_red:.2f})",
    )
    plt.plot(
        fpr_full,
        tpr_full,
        color="c",
        lw=2,
        label=f"{label_prefix} ({n_full} features, AUC = {auc_full:.2f})",
    )
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--", color="darkgrey")

    txt = r"$\it{P} < 0.001$" if p_scalar < 0.001 else f"$\\it{{P}}$ = {p_scalar:.3f}"
    plt.text(0.75, 0.5, txt, transform=ax.transAxes, ha="center", va="center", fontsize=16)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("1 - Specificity", fontsize=16)
    plt.ylabel("Sensitivity", fontsize=16)
    plt.legend(loc="lower right", frameon=False, fontsize=12)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, format=filename.split(".")[-1])
    plt.show()

    return p_value_valid, auc_full, auc_red


# =============================
# Validation bootstrap CI (AUC + ΔAUC)
# =============================
def bootstrap_auc_ci_valid(
    model,
    X_valid,
    y_valid,
    model_bp,
    X_bp_valid,
    y_bp_valid,
    n_bootstrap: int = 2000,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)

    y_valid = np.asarray(y_valid).astype(int).ravel()
    y_bp_valid = np.asarray(y_bp_valid).astype(int).ravel()

    if len(y_valid) != len(y_bp_valid):
        raise ValueError("Validation sample size differs between models.")
    if not np.array_equal(y_valid, y_bp_valid):
        raise ValueError("y_valid and y_bp_valid differ. Ensure identical samples and ordering.")

    n = len(y_valid)

    y_proba_cln = _predict_proba_any(model, X_valid)
    y_proba_bp = _predict_proba_any(model_bp, X_bp_valid)

    auc_cln = float(roc_auc_score(y_valid, y_proba_cln))
    auc_bp = float(roc_auc_score(y_valid, y_proba_bp))
    auc_diff = auc_bp - auc_cln

    aucs_cln_bs, aucs_bp_bs, aucs_diff_bs = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_bs = y_valid[idx]
        if len(np.unique(y_bs)) < 2:
            continue

        try:
            auc_cln_b = roc_auc_score(y_bs, y_proba_cln[idx])
            auc_bp_b = roc_auc_score(y_bs, y_proba_bp[idx])
        except ValueError:
            continue

        aucs_cln_bs.append(auc_cln_b)
        aucs_bp_bs.append(auc_bp_b)
        aucs_diff_bs.append(auc_bp_b - auc_cln_b)

    aucs_cln_bs = np.array(aucs_cln_bs)
    aucs_bp_bs = np.array(aucs_bp_bs)
    aucs_diff_bs = np.array(aucs_diff_bs)

    cln_low, cln_high = np.percentile(aucs_cln_bs, [2.5, 97.5])
    bp_low, bp_high = np.percentile(aucs_bp_bs, [2.5, 97.5])
    diff_low, diff_high = np.percentile(aucs_diff_bs, [2.5, 97.5])

    print("===== Bootstrap 95% CI (Validation AUC) =====")
    print(f"Model 1 AUC (cln)        : {auc_cln:.3f} [{cln_low:.3f}, {cln_high:.3f}]")
    print(f"Model 2 AUC (bp or other): {auc_bp:.3f} [{bp_low:.3f},  {bp_high:.3f}]")
    print("---------------------------------------------")
    print(f"AUC difference (model2 - model1): {auc_diff:.3f} [{diff_low:.3f}, {diff_high:.3f}]")
    print("=============================================")

    return {
        "auc_cln": auc_cln,
        "auc_cln_ci": (float(cln_low), float(cln_high)),
        "auc_bp": auc_bp,
        "auc_bp_ci": (float(bp_low), float(bp_high)),
        "auc_diff": auc_diff,
        "auc_diff_ci": (float(diff_low), float(diff_high)),
    }


# =============================
# Validation DCA
# =============================
def visualize_dca_valid(
    model,
    X_valid,
    y_valid_mrs,
    model_bp,
    X_bp_valid,
    y_bp_valid_mrs,
    pt_min=0.05,
    pt_max=0.60,
    n_pts=100,
    filename="DCA_VALID.svg",
):
    y_cln = np.asarray(y_valid_mrs).astype(int).ravel()
    y_bp = np.asarray(y_bp_valid_mrs).astype(int).ravel()

    if len(y_cln) != len(y_bp):
        raise ValueError("Validation sample size differs between models.")
    if not np.array_equal(y_cln, y_bp):
        raise ValueError("y_valid_mrs and y_bp_valid_mrs differ. Ensure identical samples and ordering.")

    p_cln = _predict_proba_any(model, X_valid)
    p_bp = _predict_proba_any(model_bp, X_bp_valid)

    pt_grid = np.linspace(pt_min, pt_max, n_pts)

    nb_cln, nb_all, nb_none = compute_net_benefit(y_cln, p_cln, pt_grid)
    nb_bp, _, _ = compute_net_benefit(y_cln, p_bp, pt_grid)

    dca_df = pd.DataFrame(
        {
            "threshold": pt_grid,
            "NB_cln": nb_cln,
            "NB_bp": nb_bp,
            "NB_all": nb_all,
            "NB_none": nb_none,
            "delta_NB": nb_bp - nb_cln,
        }
    )

    plt.figure(figsize=(6, 5))
    plt.plot(pt_grid, nb_bp, label="Clinical & SBP metrics", color="r", lw=2)
    plt.plot(pt_grid, nb_cln, label="Clinical only", color="c", lw=2)
    plt.plot(pt_grid, nb_all, label="Treat all", linestyle="--", color="grey", lw=1)
    plt.plot(pt_grid, nb_none, label="Treat none", linestyle="--", color="grey", lw=1)

    plt.axhline(0, color="grey", lw=0.5)
    plt.xlabel("Threshold probability", fontsize=16)
    plt.ylabel("Net benefit", fontsize=16)
    plt.xlim([pt_min, pt_max])
    plt.legend(loc="lower left", frameon=False, fontsize=12)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, format=filename.split(".")[-1])
    plt.show()

    return dca_df


# =============================
# DNN CV (5-fold only) + seed search
# =============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=30, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=10, min_lr=1e-6),
]


def k_fold_cross_validation_5fold(
    model_fn,
    feature,
    label,
    filename,
    group=None,
    color="c",
    n_splits=5,
    shuffle=True,
    random_state=42,
    epochs=300,
    batch_size=64,
    verbose=0,
):
    """
    5-fold CV for Keras DNN (NO repeats).
    - model_fn: creates a new model for each fold.
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    y = np.asarray(label).astype(int)

    if group is not None:
        g = np.asarray(group)
        strata = pd.Series(g).astype(str) + "_" + pd.Series(y).astype(str)
        print("Stratified by: treatment group + outcome")
    else:
        strata = pd.Series(y)
        print("Stratified by: outcome only")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    base_fpr = np.linspace(0, 1, 101)
    tprs, fold_aucs = [], []
    y_all, p_all = [], []

    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_aspect("equal", "datalim")
    linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5))]

    X = np.asarray(feature)

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, strata), start=1):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = model_fn()
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

        y_prob = model.predict(x_test, verbose=0).ravel()
        auc_fold = roc_auc_score(y_test, y_prob)
        fold_aucs.append(auc_fold)
        print(f"Fold {fold_idx} AUC: {auc_fold:.3f}")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(
            fpr,
            tpr,
            color=color,
            linestyle=linestyles[(fold_idx - 1) % len(linestyles)],
            lw=0.8,
            alpha=0.35,
            label=f"Fold {fold_idx} (AUC = {auc_fold:.2f})",
        )

        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        y_all.append(y_test)
        p_all.append(y_prob)

        tf.keras.backend.clear_session()

    fold_aucs = np.array(fold_aucs)
    auc_mean = float(fold_aucs.mean())
    auc_sd = float(fold_aucs.std(ddof=1))
    print(f"\n{n_splits}-fold CV AUC mean ± SD: {auc_mean:.3f} ± {auc_sd:.3f}")

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std_tprs = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
    tprs_lower = np.maximum(mean_tprs - std_tprs, 0)

    plt.plot(base_fpr, mean_tprs, color, label=f"Mean ROC (AUC = {auc_mean:.2f} ± {auc_sd:.2f})", lw=2)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=color, alpha=0.1)
    plt.plot([0, 1], [0, 1], color="darkgrey", lw=1, linestyle="--")

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel("Sensitivity", fontsize=16)
    plt.xlabel("1 - Specificity", fontsize=16)
    plt.legend(loc="lower right", frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, format="svg")
    plt.show()

    return {
        "fold_aucs": fold_aucs,
        "auc_mean": auc_mean,
        "auc_sd": auc_sd,
        "y_oof": y_all,
        "p_oof": p_all,
        "base_fpr": base_fpr,
        "mean_tprs": mean_tprs,
    }
