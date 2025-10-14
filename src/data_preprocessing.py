# data_preprocessing.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Input Excel
    excel_path: Path
    sheet_clinical: str = "305_ITT"
    sheet_cs: str = "conventional_systolic"
    sheet_is: str = "intensive_systolic"

    # ITT vs PP selection
    use_per_protocol: bool = False
    pp_column: str = "Per_protocol"
    pp_flag: int = 1

    # Outcome binarization
    outcome_col: str = "mRS_3months"
    good_outcome: Iterable[int] = (0, 1, 2)  # => 0 (good) if in {0,1,2}; else 1 (poor)

    # Time-grid settings
    hours: int = 24              # 1h .. 24h
    tr_interval_minutes: int = 60  # Δt used for TR on the chosen phase grid

    # Resampling policy (when we need to interpolate on the phase grids)
    max_gap_min: int = 90        # interpolate only inside gaps ≤ this span
    edge_fill: bool = False      # allow one-step edge fill?
    edge_fill_max: int = 30      # max allowed span for edge fill

    # Outputs
    out_dir: Path = Path("results")
    out_best_params_dir: Path = Path("results/best_params")
    out_models_dir: Path = Path("results/models")
    out_reports_dir: Path = Path("results/reports")

    # Save switches
    save_intermediate_excel: bool = True
    save_scalers: bool = True

    # Keep per-phase features (optional)
    keep_phase_columns: bool = True  # if False, only final “chosen” features are kept


# ──────────────────────────────────────────────────────────────────────────────
# Utilities: parsing & phase grids
# ──────────────────────────────────────────────────────────────────────────────

def _parse_time_to_min(col: str) -> int:
    """
    Convert SBP column name to minutes since enrollment.
    Handles: Systolic_enroll (=0), Systolic_15min, Systolic_1h, Systolic_1h_15min, Systolic_2h_45min ...
    """
    assert col.startswith("Systolic_")
    token = col.replace("Systolic_", "").lower().strip()

    # explicitly handle enroll / 0 time
    if token in {"enroll", "0", "0min", "0h"}:
        return 0

    # patterns like "15min"
    if "h" not in token and token.endswith("min"):
        return int(token.replace("min", ""))

    # patterns like "1h" or "1h_15min"
    minutes = 0
    if "h" in token:
        parts = token.split("_")
        # hours part
        h = int(parts[0].replace("h", ""))
        minutes += h * 60
        # optional "_15min" like suffix
        if len(parts) > 1 and parts[1].endswith("min"):
            minutes += int(parts[1].replace("min", ""))
        return minutes

    # fallback: try pure integer minutes
    return int(token)



def make_phase_grids(hours: int = 24) -> Dict[str, np.ndarray]:
    """
    Build four phase grids:
      p0  : 60, 120, ..., 1440
      p15 : 15, 75, ..., 1415
      p30 : 30, 90, ..., 1430
      p45 : 45, 105, ..., 1445 (but we cap at < 1440 for 24h horizon)
    Note: We cap p45 at <= 1439 to stay within 24h horizon.
    """
    end = hours * 60
    p0 = np.arange(60, end + 1, 60, dtype=float)
    p15 = np.arange(15, end, 60, dtype=float)
    p30 = np.arange(30, end, 60, dtype=float)
    p45 = np.arange(45, end, 60, dtype=float)
    return {"p0": p0, "p15": p15, "p30": p30, "p45": p45}


# ──────────────────────────────────────────────────────────────────────────────
# Load & clean tables
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_tables(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load clinical, conventional SBP, intensive SBP sheets."""
    cln = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet_clinical, engine="openpyxl")
    cs = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet_cs, engine="openpyxl")
    is_ = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet_is, engine="openpyxl")
    return cln, cs, is_


def clean_clinical_table(cln: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Clinical cleanup:
      - ITT vs PP selection
      - drop rows without outcome
      - select necessary columns
      - simple imputations
      - encode sex (M->1, F->0)
      - binarize outcome
    """
    if cfg.use_per_protocol and cfg.pp_column in cln:
        cln = cln[cln[cfg.pp_column] == cfg.pp_flag]

    cln = cln.dropna(subset=[cfg.outcome_col])

    keep_cols = [
        'multi', 'optimal_bp_reg_no',
        'pt_age', 'pt_sex', 'HiBP', 'Hyperlipidemia', 'Smoking',
        'Previous_stroke_existence', 'CAOD합친것', 'cancer_active', 'CHF_onoff', 'PAOD_existence',
        'NIHSS_IAT_just_before', 'Onset_to_registration_min',
        'IV_tPA', 'Systolic_enroll',
        'DM', 'A_fib합친것', 'Antiplatelet', 'Anticoagulant',
        'Hgb', 'WBC', 'BMI',
        cfg.outcome_col
    ]
    cln = cln[keep_cols]

    for col in ["WBC", "BMI"]:
        if col in cln:
            cln[col] = cln[col].fillna(cln[col].mean())

    cln["pt_sex"] = (
        cln["pt_sex"].astype(str).str.upper()
        .map({"M": 1, "MALE": 1, "F": 0, "FEMALE": 0})
        .fillna(0).astype(int)
    )

    cln[cfg.outcome_col] = (
        cln[cfg.outcome_col].apply(lambda x: 0 if int(x) in cfg.good_outcome else 1).astype(int)
    )
    return cln


def trim_bp_tables(cs: pd.DataFrame, is_: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only SBP columns (Systolic_*) + ID, drop last two rows (totals/remarks),
    and unify key to 'optimal_bp_reg_no'.
    """
    sbp_cols = ["연구대상자ID"] + [c for c in cs.columns if c.startswith("Systolic_") and c != "Systolic_enroll"]
    cs = cs[sbp_cols].iloc[:-2].copy()
    is_ = is_[sbp_cols].iloc[:-2].copy()
    cs = cs.rename(columns={"연구대상자ID": "optimal_bp_reg_no"})
    is_ = is_.rename(columns={"연구대상자ID": "optimal_bp_reg_no"})
    return cs, is_


# ──────────────────────────────────────────────────────────────────────────────
# Coverage & resampling
# ──────────────────────────────────────────────────────────────────────────────

def _collect_observed_times_and_values(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a row, collect all observed (time, value) pairs across any SBP columns.
    Filter out physically implausible values (<50 or >260).
    """
    xs, ys = [], []
    for c, v in row.items():
        if isinstance(c, str) and c.startswith("Systolic_") and pd.notna(v):
            val = float(v)
            if 50.0 <= val <= 260.0:
                xs.append(_parse_time_to_min(c))
                ys.append(val)
    if not xs:
        return np.array([]), np.array([])
    order = np.argsort(xs)
    return np.asarray(xs, dtype=float)[order], np.asarray(ys, dtype=float)[order]


def _coverage_for_phase(xs_obs: np.ndarray, grid: np.ndarray) -> float:
    """
    Phase coverage = (# of target grid points that were actually observed) / (len(grid)).
    We treat a target point as 'observed' if there exists an exact matching time in xs_obs.
    """
    if len(grid) == 0:
        return 0.0
    if xs_obs.size == 0:
        return 0.0
    # Since times are integers in minutes, exact matching is fine
    observed_points = np.intersect1d(grid.astype(int), xs_obs.astype(int), assume_unique=False)
    return float(len(observed_points)) / float(len(grid))


def _resample_with_gap_rules(
    xs: np.ndarray, ys: np.ndarray, target_minutes: np.ndarray,
    max_gap_min: int, edge_fill: bool, edge_fill_max: int
) -> np.ndarray:
    """
    Linear interpolation on a target grid with the following constraints:
      - Only interpolate inside brackets whose span ≤ max_gap_min
      - No extrapolation beyond [xs[0], xs[-1]] (optional short edge fill)
    """
    out = np.full_like(target_minutes, np.nan, dtype=float)
    if xs.size < 2:
        return out

    y_all = np.interp(target_minutes, xs, ys)

    in_domain = (target_minutes >= xs[0]) & (target_minutes <= xs[-1])
    idx = np.searchsorted(xs, target_minutes)
    left_ok = (idx > 0)
    right_ok = (idx < len(xs))
    interior = in_domain & left_ok & right_ok
    if np.any(interior):
        left = xs[idx[interior] - 1]
        right = xs[idx[interior]]
        gap = right - left
        ok = gap <= max_gap_min
        out[np.where(interior)[0][ok]] = y_all[interior][ok]

    if edge_fill:
        lead_mask = (target_minutes < xs[0]) & ((xs[0] - target_minutes) <= edge_fill_max)
        if np.any(lead_mask):
            out[lead_mask] = ys[0]
        trail_mask = (target_minutes > xs[-1]) & ((target_minutes - xs[-1]) <= edge_fill_max)
        if np.any(trail_mask):
            out[trail_mask] = ys[-1]

    return out


def _assign_exact_observations(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    보간 없음: 관측 시각(xs)과 grid가 '분 단위로 정확히 일치'하는 지점만 값을 채우고
    나머지는 NaN으로 둔다.
    """
    out = np.full_like(grid, np.nan, dtype=float)
    if xs.size == 0:
        return out

    # 분 단위 정수 매칭(코드 전체가 분 단위 정수 타임스탬프를 가정)
    m = {int(t): v for t, v in zip(xs, ys)}
    g_int = grid.astype(int)
    for j, t in enumerate(g_int):
        if t in m:
            out[j] = m[t]
    return out


def resample_to_all_phases(
    df: pd.DataFrame,
    grids: Dict[str, np.ndarray],
    max_gap_min: int,
    edge_fill: bool,
    edge_fill_max: int
) -> pd.DataFrame:
    """
    For each patient (row), compute:
      - observed (xs, ys)
      - coverage for each phase grid
      - resampled series for each phase grid (with gap rules)
    Stores:
      - '__phase_series_p0', '__phase_series_p15', '__phase_series_p30', '__phase_series_p45'
      - 'coverage_p0', 'coverage_p15', 'coverage_p30', 'coverage_p45'
    """
    df = df.copy()
    series_cols = {}
    cover_cols = {}
    for phase in ("p0", "p15", "p30", "p45"):
        series_col = f"__phase_series_{phase}"
        cov_col = f"coverage_{phase}"
        series_cols[phase] = series_col
        cover_cols[phase] = cov_col
        df[series_col] = None
        df[cov_col] = np.nan

    for i, row in df.iterrows():
        xs, ys = _collect_observed_times_and_values(row)
        for phase, grid in grids.items():
            cov = _coverage_for_phase(xs, grid)
            # res = _resample_with_gap_rules(xs, ys, grid, max_gap_min, edge_fill, edge_fill_max)
            res = _assign_exact_observations(xs, ys, grid)
            df.at[i, f"coverage_{phase}"] = cov
            df.at[i, f"__phase_series_{phase}"] = res

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def _fit_vim_safe(mean_series: pd.Series, sd_series: pd.Series) -> Tuple[float, float]:
    """
    Fit VIM on log–log space:
        log(SD) = b * log(mean) - log(k)  (we fit log(SD) = b * log(mean) + c; then k = exp(-c))
    Return (b, k). Fall back to (1.0, 1.0) if invalid/insufficient.
    """
    x = np.log(mean_series.replace([0, np.inf, -np.inf], np.nan)).dropna()
    y = np.log(sd_series.replace([0, np.inf, -np.inf], np.nan)).loc[x.index]
    if len(x) < 2 or len(y) < 2 or x.isna().any() or y.isna().any():
        return 1.0, 1.0
    try:
        b, c = curve_fit(lambda xx, b_, c_: b_ * xx + c_, x, y, maxfev=10000)[0]
        return float(b), float(np.exp(-c))
    except Exception:
        return 1.0, 1.0


def _features_from_series(
    arrs: List[np.ndarray], interval_min: int, prefix: str
) -> Dict[str, np.ndarray]:
    """
    Compute features from per-row arrays (already aligned on a grid, NaNs allowed).
    arrs: list of arrays (each shape (T,))
    Returns vectors for: max, min, mean, TR, SD, CV, VIM
    """
    mat = np.vstack(arrs)  # (n, T)
    f_max = np.nanmax(mat, axis=1)
    f_min = np.nanmin(mat, axis=1)
    f_mean = np.nanmean(mat, axis=1)

    diffs = np.abs(np.diff(mat, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        f_tr = np.nanmean(diffs / float(interval_min), axis=1)

    f_sd = np.nanstd(mat, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_cv = f_sd / f_mean

    # VIM fitted across valid rows, then applied row-wise
    df_tmp = pd.DataFrame({"mean": f_mean, "sd": f_sd}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_tmp) >= 2:
        beta, k = _fit_vim_safe(df_tmp["mean"], df_tmp["sd"])
    else:
        beta, k = 1.0, 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        f_vim = k * f_sd / (np.power(f_mean, beta))

    return {
        f"{prefix}_max": f_max,
        f"{prefix}_min": f_min,
        f"{prefix}_mean": f_mean,
        f"{prefix}_TR": f_tr,
        f"{prefix}_SD": f_sd,
        f"{prefix}_CV": f_cv,
        f"{prefix}_VIM": f_vim,
    }


def _compute_overall_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall max/min/mean USING ALL raw Systolic_* columns (across 15/30/45/hour…24h).
    This ignores phase choice; it strictly aggregates over every raw SBP column.
    """
    df = df.copy()
    sbp_cols = [c for c in df.columns if c.startswith("Systolic_")]
    arr = df[sbp_cols].to_numpy(dtype=float)
    # Filter physically implausible values
    arr[(arr < 50) | (arr > 260)] = np.nan
    df["systolic_overall_max"] = np.nanmax(arr, axis=1)
    df["systolic_overall_min"] = np.nanmin(arr, axis=1)
    df["systolic_overall_mean"] = np.nanmean(arr, axis=1)
    return df


def choose_phase_per_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best phase per patient using coverage:
      - chosen_phase: argmax over coverage_p0/p15/p30/p45
      - chosen_series: the resampled series for that phase
    """
    df = df.copy()
    cov_cols = ["coverage_p0", "coverage_p15", "coverage_p30", "coverage_p45"]
    phases = ["p0", "p15", "p30", "p45"]

    cov_mat = df[cov_cols].to_numpy(dtype=float)
    # If all coverages are NaN for a row, treat them as -1 for argmax so we still pick one deterministically
    cov_mat = np.where(np.isnan(cov_mat), -1.0, cov_mat)
    best_idx = np.argmax(cov_mat, axis=1)
    chosen = [phases[i] for i in best_idx]
    df["chosen_phase"] = chosen

    chosen_series = []
    for i, ph in enumerate(chosen):
        chosen_series.append(df.iloc[i][f"__phase_series_{ph}"])
    df["__chosen_series"] = chosen_series
    return df


def compute_phase_and_final_features(
    df: pd.DataFrame,
    grids: Dict[str, np.ndarray],
    interval_min: int,
    keep_phase_columns: bool = True
) -> pd.DataFrame:
    """
    1) Compute per-phase features (p0/p15/p30/p45) from their resampled arrays.
    2) Choose a phase per patient by coverage and compute “final” BPV features from that phase.
    3) Attach overall max/min/mean computed from ALL raw SBP columns.
    """
    df = df.copy()

    # Per-phase features
    per_phase_feats: Dict[str, Dict[str, np.ndarray]] = {}
    for phase, grid in grids.items():
        arrs = df[f"__phase_series_{phase}"].tolist()
        per_phase_feats[phase] = _features_from_series(arrs, interval_min, prefix=f"systolic_{phase}")

    # Attach per-phase features
    for phase, feats in per_phase_feats.items():
        for k, v in feats.items():
            df[k] = v

    # Choose phase per row
    df = choose_phase_per_row(df)

    # Final (coverage-chosen) BPV features (TR/SD/CV/VIM) from __chosen_series
    final_arrs = df["__chosen_series"].tolist()
    final_feats = _features_from_series(final_arrs, interval_min, prefix="systolic_final")
    # We DO NOT overwrite final max/min/mean with phase-based versions;
    # Instead we add “overall” max/min/mean from raw data (see below).
    df["systolic_final_TR"] = final_feats["systolic_final_TR"]
    df["systolic_final_SD"] = final_feats["systolic_final_SD"]
    df["systolic_final_CV"] = final_feats["systolic_final_CV"]
    df["systolic_final_VIM"] = final_feats["systolic_final_VIM"]

    # Overall (raw-based) max/min/mean across ALL Systolic_* columns
    df = _compute_overall_basic_stats(df)

    # Optionally drop per-phase feature columns to keep only the final features
    if not keep_phase_columns:
        drop_cols = []
        for phase in grids.keys():
            drop_cols.extend([c for c in df.columns if c.startswith(f"systolic_{phase}_")])
            drop_cols.append(f"__phase_series_{phase}")
            drop_cols.append(f"coverage_{phase}")
        df = df.drop(columns=drop_cols, errors="ignore")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Scaling
# ──────────────────────────────────────────────────────────────────────────────

def scale_by_groups(
    df: pd.DataFrame,
    groups: List[List[str]]
) -> Tuple[pd.DataFrame, Dict[Tuple[str, ...], StandardScaler]]:
    """
    Apply StandardScaler to each group of columns (only if they exist).
    """
    df = df.copy()
    scalers: Dict[Tuple[str, ...], StandardScaler] = {}
    for cols in groups:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        scaler = StandardScaler()
        df.loc[:, cols] = scaler.fit_transform(df[cols])
        scalers[tuple(cols)] = scaler
    return df, scalers


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    End-to-end preprocessing with coverage-aware phase selection:
      1) Load tables
      2) Clean clinical; trim SBP tables
      3) Resample each patient to p0/p15/p30/p45; compute coverage per phase
      4) Merge with clinical & assign group label (0: CS, 1: IS)
      5) Compute per-phase and final features (final BPV from chosen phase; overall max/min/mean from raw)
      6) Drop raw Systolic_* columns
      7) Scale clinical/BP/BPV groups
      8) Concatenate CS & IS
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    cln, cs, is_ = load_raw_tables(cfg)

    # 2) Clean & trim
    cln = clean_clinical_table(cln, cfg)
    cs, is_ = trim_bp_tables(cs, is_)

    # 3) Coverage & resampling
    grids = make_phase_grids(hours=cfg.hours)
    cs = resample_to_all_phases(cs, grids, cfg.max_gap_min, cfg.edge_fill, cfg.edge_fill_max)
    is_ = resample_to_all_phases(is_, grids, cfg.max_gap_min, cfg.edge_fill, cfg.edge_fill_max)

    # 4) Merge with clinical & cohort label
    cs = pd.merge(cs, cln, on="optimal_bp_reg_no", how="inner")
    is_ = pd.merge(is_, cln, on="optimal_bp_reg_no", how="inner")
    cs["Group"] = 0
    is_["Group"] = 1

    # 5) Features (per-phase + final), with overall raw-based max/min/mean
    cs = compute_phase_and_final_features(
        cs, grids, cfg.tr_interval_minutes, keep_phase_columns=cfg.keep_phase_columns
    )
    is_ = compute_phase_and_final_features(
        is_, grids, cfg.tr_interval_minutes, keep_phase_columns=cfg.keep_phase_columns
    )

    # 6) Drop raw systolic columns
    raw_cols_cs = [c for c in cs.columns if c.startswith("Systolic_") and c != "Systolic_enroll"]
    raw_cols_is = [c for c in is_.columns if c.startswith("Systolic_") and c != "Systolic_enroll"]
    cs = cs.drop(columns=raw_cols_cs, errors="ignore")
    is_ = is_.drop(columns=raw_cols_is, errors="ignore")

    # Optional snapshot
    if cfg.save_intermediate_excel:
        cs.to_excel(cfg.out_dir / "cs_data_before_scaling.xlsx", index=False)
        is_.to_excel(cfg.out_dir / "is_data_before_scaling.xlsx", index=False)

    # 7) Scaling groups
    scale_cols = ['pt_age', 'NIHSS_IAT_just_before', 'Onset_to_registration_min', 'Hgb', 'WBC', 'BMI', 'Systolic_enroll']

    # Canonical BP/BPV (what downstream code can use by default)
    # - Use overall (raw-based) max/min/mean
    # - Use “final” (coverage-chosen) BPV metrics
    bp_cols = ['Systolic_enroll', 'systolic_overall_max', 'systolic_overall_min', 'systolic_overall_mean']
    bpv_cols = ['systolic_final_TR', 'systolic_final_SD', 'systolic_final_CV', 'systolic_final_VIM']

    # Optionally also scale the per-phase features if you decided to keep them
    phase_cols: List[str] = []
    if cfg.keep_phase_columns:
        for phase in ("p0", "p15", "p30", "p45"):
            phase_cols.extend([
                f"systolic_{phase}_max", f"systolic_{phase}_min", f"systolic_{phase}_mean",
                f"systolic_{phase}_TR", f"systolic_{phase}_SD", f"systolic_{phase}_CV", f"systolic_{phase}_VIM",
            ])

    groups = [scale_cols, bp_cols, bpv_cols] + ([phase_cols] if phase_cols else [])

    cs, scalers_cs = scale_by_groups(cs, groups)
    is_, scalers_is = scale_by_groups(is_, groups)

    if cfg.save_scalers:
        joblib.dump(scalers_cs, cfg.out_dir / "scalers_cs.pkl")
        joblib.dump(scalers_is, cfg.out_dir / "scalers_is.pkl")

    # 8) Combine cohorts
    bp_all = pd.concat([cs, is_], ignore_index=True).fillna(0)

    meta = {
        "bp_cols": bp_cols,
        "bpv_cols": bpv_cols,
        "phase_cols": phase_cols,
        "scale_cols": scale_cols,
        "tr_interval_minutes": cfg.tr_interval_minutes,
        "grids": {k: v.tolist() for k, v in grids.items()},
        "keep_phase_columns": cfg.keep_phase_columns,
        "scalers_cs_path": str(cfg.out_dir / "scalers_cs.pkl"),
        "scalers_is_path": str(cfg.out_dir / "scalers_is.pkl"),
    }
    return cs, is_, bp_all, meta


# ──────────────────────────────────────────────────────────────────────────────
# Helper for clinical-only ablation
# ──────────────────────────────────────────────────────────────────────────────

def drop_bp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove canonical BP/BPV and group/enroll markers for clinical-only experiments.
    Canonical here means:
      - systolic_overall_max/min/mean (from all raw SBP columns)
      - systolic_final_* (coverage-chosen BPV metrics)
    """
    drop_cols = [
        "Systolic_enroll", "Group",
        "systolic_overall_max", "systolic_overall_min", "systolic_overall_mean",
        "systolic_final_TR", "systolic_final_SD", "systolic_final_CV", "systolic_final_VIM",
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
