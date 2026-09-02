#!/usr/bin/env python3
"""Build leakage-safe archetype and meta-handoff reliability labels.

This script implements ARCHETYPE_META_HANDOFF_V1 for the S52-style candidate
ledger.  It deliberately evaluates archetypes out-of-fold:

* train months fit scalers, a linear autoencoder proxy (PCA), and GMMs;
* validation months receive frozen assignments and train-derived priors only;
* outcome profiles/reliability priors are computed on train rows and joined to
  validation rows without using validation outcomes;
* the five requested tests are evaluated on validation rows.

The "AE" is a deterministic low-rank linear autoencoder via PCA.  It gives
latent coordinates and reconstruction error while keeping the implementation
light enough for smoke and full-ledger iteration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - exercised in integration tests when sklearn exists
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import RobustScaler
except Exception as exc:  # pragma: no cover
    PCA = None  # type: ignore[assignment]
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    GaussianMixture = None  # type: ignore[assignment]
    RobustScaler = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


DEFAULT_LEDGER = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_v1/s52_trailing_regime_scored_ledger.parquet"
)
DEFAULT_OUT_DIR = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_v1/archetype_meta_handoff_v1"
)
DEFAULT_SEEDS = (17, 29, 41)
OUTCOME_COLUMNS = (
    "exec_margin",
    "ret_net",
    "u_policy_net",
    "clean_exec",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "first_touch_bad_mae_1r",
    "timeout",
    "mae_norm",
    "mfe_norm",
    "underwater_bars_before_mfe_1r",
)
POTENTIAL_FEATURES = (
    "score",
    "side",
    "__regime_vol_12h__",
    "__regime_vol_48h__",
    "__regime_volume_12h__",
    "__regime_volume_48h__",
    "__regime_trend_12h__",
    "__regime_trend_48h__",
    "__meta_raw__volatility_zscore",
    "__meta_raw__asset_minus_mkt_oi_1d_peer_resid",
    "__meta_raw__return_autocorr_48",
    "G_VOL",
    "cluster_speed",
    "cluster_acceleration",
    "latent_speed",
    "latent_acceleration",
    "AE_reconstruction_error",
    "dae_reconstruction_error",
    "mahalanobis_distance",
    "expected_mahalanobis",
)
EXCLUDED_NUMERIC_FEATURES = {
    "target_soft",
    "target_hard",
    "first_pass_good",
    "first_pass_bad",
    "u_policy_net",
    "ret_net",
    "is_timeout",
    "mae_norm",
    "mfe_norm",
    "first_touch_net",
    "first_touch_mae_norm",
    "first_touch_mfe_norm",
    "first_touch_full_path_mae_norm",
    "mfe_1r_before_mae_1r",
    "mae_1r_before_mfe_1r",
    "max_adverse_before_mfe_1r",
    "underwater_bars_before_mfe_1r",
    "underwater_fraction_before_mfe_1r",
    "selected_top10",
    "selected_top20",
    "selected_top30",
    "first_touch_gross",
    "exec_margin",
    "ev_after_1pct",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "clean_exec",
    "dirty_positive",
    "long_path_clean_exec_label",
    "long_path_dirty_positive_label",
    "long_path_full_bad_mae_1r",
    "long_path_post_mfe_bad_drawdown",
    "long_path_post_mfe_drawdown_norm",
    "long_path_quality_soft",
    "long_path_slow_profit",
    "long_path_time_to_profit_bars",
}
EXCLUDED_NUMERIC_PREFIXES = (
    "regime_bad_mae_score",
    "regime_first_touch_bad_mae_score",
    "regime_timeout_score",
    "regime_dirty_positive_score",
    "regime_clean_exec_score",
    "regime_exec_margin_score",
    "regime_ev_score",
    "regime_lgbm_leaf_",
)
LEARNED_SCORE_COLUMNS = (
    "score_M0_no_archetype",
    "score_M1_hard_archetype_id",
    "score_M2_gmm_posterior_entropy",
    "score_M3_archetype_outcome_priors",
    "score_M4_base_reliability_priors",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _num(values: Any, *, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        if index is None:
            return pd.Series(dtype=np.float32)
        return pd.Series(default, index=index, dtype=np.float32)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _mean(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _q(values: Any, q: float) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.quantile(float(q))) if len(arr) else float("nan")


def _safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"__ts__", "__symbol__", "side_name", "month", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ledger is missing required columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["__ts__"], errors="coerce", utc=True).dt.tz_convert(None)
    out["symbol"] = out["__symbol__"].astype(str)
    out["side_name"] = out["side_name"].astype(str).str.lower()
    out["candidate_id"] = (
        out["timestamp"].astype(str) + "|" + out["symbol"].astype(str) + "|" + out["side_name"].astype(str)
    )
    out["base_score"] = _num(out.get("score"), index=out.index).astype(np.float32)
    out["u_econ_net"] = _num(out.get("exec_margin", out.get("u_policy_net")), index=out.index).astype(np.float32)
    out["clean_exec_positive"] = _num(out.get("clean_exec"), index=out.index).fillna(0.0).gt(0.5).astype(np.int8)
    out["dirty_positive"] = _num(out.get("dirty_positive"), index=out.index).fillna(0.0).gt(0.5).astype(np.int8)
    out["bad_MAE"] = _num(out.get("full_path_bad_mae_1r"), index=out.index).fillna(0.0).gt(0.5).astype(np.int8)
    out["timeout_label"] = _num(out.get("timeout", out.get("is_timeout")), index=out.index).fillna(0.0).gt(0.5).astype(np.int8)
    out["base_selected"] = _num(out.get("selected_top10"), index=out.index).fillna(0.0).gt(0.5).astype(np.int8)
    out["spread_bucket"] = "spread_missing"
    vol = _num(out.get("__meta_raw__volatility_zscore", out.get("G_VOL")), index=out.index)
    try:
        out["volatility_bucket"] = pd.qcut(
            vol.rank(method="first"),
            q=min(4, max(1, int(vol.notna().sum()))),
            labels=[f"vol_q{i}" for i in range(min(4, max(1, int(vol.notna().sum()))))],
            duplicates="drop",
        ).astype(str)
    except Exception:
        out["volatility_bucket"] = "vol_missing"
    out["side_spread_archetype_id"] = out["side_name"].astype(str) + "__" + out["spread_bucket"].astype(str)
    out["side_vol_archetype_id"] = out["side_name"].astype(str) + "__" + out["volatility_bucket"].astype(str)
    return out


def _folds_from_months(frame: pd.DataFrame, min_train_months: int) -> list[dict[str, Any]]:
    months = sorted(str(v) for v in frame["month"].dropna().astype(str).unique())
    folds: list[dict[str, Any]] = []
    for i in range(int(min_train_months), len(months)):
        folds.append(
            {
                "fold_id": f"fold_{i:02d}_train_to_{months[i]}",
                "train_months": months[:i],
                "valid_month": months[i],
            }
        )
    return folds


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = [col for col in POTENTIAL_FEATURES if col in frame.columns and _num(frame[col]).notna().any()]
    for col in frame.columns:
        if col in cols or col in EXCLUDED_NUMERIC_FEATURES:
            continue
        if any(str(col).startswith(prefix) for prefix in EXCLUDED_NUMERIC_PREFIXES):
            continue
        if str(col).startswith(("gmm_", "cluster_", "latent_", "dae_", "__regime_", "__meta_raw__")) or str(col) in {
            "AE_reconstruction_error",
            "ae_reconstruction_error",
            "mahalanobis_distance",
            "min_mahalanobis",
            "min_mahalanobis_delta_1",
            "expected_mahalanobis",
            "expected_mahalanobis_delta_1",
            "expected_mahalanobis_accel_1",
            "rolling_cluster_stability",
            "time_since_cluster_change",
            "side",
            "score",
            "G_VOL",
        }:
            values = _num(frame[col])
            if values.notna().any():
                cols.append(str(col))
    # Preserve order while removing duplicates.
    cols = list(dict.fromkeys(cols))
    if "score" not in cols:
        cols.insert(0, "score")
    return cols


def _matrix(frame: pd.DataFrame, cols: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median().fillna(0.0)
    x = x.fillna(medians).astype(np.float32)
    return x, medians


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=1)
    denom = math.log(max(p.shape[1], 2))
    return (ent / denom).astype(np.float32)


def _fit_assign_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    feature_cols: list[str],
    n_components: int,
    seed: int,
    n_latent: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if PCA is None or GaussianMixture is None or RobustScaler is None:
        raise RuntimeError(f"scikit-learn is required for archetype fitting: {_SKLEARN_IMPORT_ERROR}")
    x_train, medians = _matrix(train, feature_cols)
    x_valid, _ = _matrix(valid, feature_cols, medians)
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(x_train)
    valid_scaled = scaler.transform(x_valid)
    latent_dim = max(1, min(int(n_latent), train_scaled.shape[1], len(train_scaled) - 1))
    pca = PCA(n_components=latent_dim, random_state=int(seed))
    train_latent = np.asarray(pca.fit_transform(train_scaled), dtype=np.float64)
    valid_latent = np.asarray(pca.transform(valid_scaled), dtype=np.float64)
    train_recon = pca.inverse_transform(train_latent)
    valid_recon = pca.inverse_transform(valid_latent)
    train_err = np.mean((train_scaled - train_recon) ** 2, axis=1).astype(np.float32)
    valid_err = np.mean((valid_scaled - valid_recon) ** 2, axis=1).astype(np.float32)
    k = max(2, min(int(n_components), len(train_latent) // 50 if len(train_latent) >= 100 else 2))
    k = min(k, len(train_latent))
    last_exc: Exception | None = None
    for reg_covar in (1e-4, 1e-3, 1e-2):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=reg_covar,
                random_state=int(seed),
                max_iter=200,
            )
            gmm.fit(train_latent)
            break
        except ValueError as exc:
            last_exc = exc
    else:
        raise RuntimeError(f"GMM fit failed for all regularization levels: {last_exc}") from last_exc
    train_prob = gmm.predict_proba(train_latent).astype(np.float32)
    valid_prob = gmm.predict_proba(valid_latent).astype(np.float32)
    train_cluster = np.asarray(gmm.predict(train_latent), dtype=np.int16)
    valid_cluster = np.asarray(gmm.predict(valid_latent), dtype=np.int16)
    train_dist = _distance_to_centers(train_latent, gmm.means_)
    valid_dist = _distance_to_centers(valid_latent, gmm.means_)
    train_out = pd.DataFrame(index=train.index)
    valid_out = pd.DataFrame(index=valid.index)
    _assign_latent_columns(train_out, train_latent)
    _assign_latent_columns(valid_out, valid_latent)
    train_out["ae_reconstruction_error_oof"] = train_err
    valid_out["ae_reconstruction_error_oof"] = valid_err
    train_out["ae_gmm_archetype_id"] = [f"gmm_{int(v)}" for v in train_cluster]
    valid_out["ae_gmm_archetype_id"] = [f"gmm_{int(v)}" for v in valid_cluster]
    train_out["gmm_max_posterior"] = np.max(train_prob, axis=1)
    valid_out["gmm_max_posterior"] = np.max(valid_prob, axis=1)
    train_out["gmm_entropy_oof"] = _entropy(train_prob)
    valid_out["gmm_entropy_oof"] = _entropy(valid_prob)
    train_out["gmm_distance_to_centroid"] = train_dist[np.arange(len(train_dist)), train_cluster].astype(np.float32)
    valid_out["gmm_distance_to_centroid"] = valid_dist[np.arange(len(valid_dist)), valid_cluster].astype(np.float32)
    for i in range(k):
        train_out[f"gmm_posterior_{i}"] = train_prob[:, i]
        valid_out[f"gmm_posterior_{i}"] = valid_prob[:, i]
    fit_info = {
        "seed": int(seed),
        "n_components": int(k),
        "n_latent": int(latent_dim),
        "feature_cols": list(feature_cols),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "scaler_fit_scope": "outer_train_only",
        "ae_fit_scope": "outer_train_only",
        "gmm_fit_scope": "outer_train_only",
        "gmm_reg_covar": float(gmm.reg_covar),
        "validation_assignment_scope": "frozen_train_artifacts",
    }
    return train_out, valid_out, fit_info


def _assign_latent_columns(out: pd.DataFrame, latent: np.ndarray) -> None:
    for i in range(latent.shape[1]):
        out[f"ae_latent_{i + 1}"] = latent[:, i].astype(np.float32)


def _distance_to_centers(latent: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = latent[:, None, :] - centers[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)


def _train_score_buckets(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    score_train = _num(train["base_score"], index=train.index)
    score_valid = _num(valid["base_score"], index=valid.index)
    ranks_train = score_train.rank(pct=True, method="average").fillna(0.0)
    quantiles = score_train.dropna().quantile(np.linspace(0.1, 0.9, 9)).to_numpy(dtype=float)
    if len(quantiles) == 0:
        return pd.Series("mid", index=train.index), pd.Series("mid", index=valid.index)
    train_dec = np.searchsorted(quantiles, score_train.fillna(score_train.median()).to_numpy(dtype=float), side="right")
    valid_dec = np.searchsorted(quantiles, score_valid.fillna(score_train.median()).to_numpy(dtype=float), side="right")
    train_band = pd.Series(np.where(ranks_train >= 0.8, "high", np.where(ranks_train <= 0.4, "low", "mid")), index=train.index)
    valid_pct = pd.Series(score_valid.rank(pct=True, method="average"), index=valid.index)
    # Use train quantiles for labels and validation rank percentile for smooth scoring.
    valid_band = pd.Series(np.where(valid_dec >= 8, "high", np.where(valid_dec <= 3, "low", "mid")), index=valid.index)
    train_band = train_band.astype(str) + "_d" + pd.Series(train_dec, index=train.index).astype(str)
    valid_band = valid_band.astype(str) + "_d" + pd.Series(valid_dec, index=valid.index).astype(str)
    return train_band, valid_band


def _quality_label(row: pd.Series, global_rates: dict[str, float], min_support: int) -> str:
    if int(row.get("row_count", 0)) < int(min_support):
        return "low_support"
    clean = float(row.get("clean_exec_positive_rate", np.nan))
    dirty = float(row.get("dirty_positive_rate", np.nan))
    bad = float(row.get("bad_MAE_rate", np.nan))
    timeout = float(row.get("timeout_rate", np.nan))
    utility = float(row.get("mean_u_econ_net", np.nan))
    if bad >= 1.20 * global_rates.get("bad_MAE_rate", bad):
        return "bad_MAE_prone"
    if timeout >= 1.30 * max(global_rates.get("timeout_rate", timeout), 1e-6):
        return "timeout_prone"
    if dirty >= 1.20 * global_rates.get("dirty_positive_rate", dirty):
        return "dirty_positive"
    if clean >= 1.15 * global_rates.get("clean_exec_positive_rate", clean) and utility > 0.0:
        return "clean_executable"
    if utility < 0.0:
        return "negative_utility"
    return "neutral"


def _profile_table(
    train: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    fold_id: str,
    family: str,
    id_col: str,
    min_support: int,
) -> pd.DataFrame:
    work = train.reset_index(drop=True).copy()
    if id_col not in work.columns:
        work = pd.concat([work, assignments[[id_col]].reset_index(drop=True)], axis=1)
    global_rates = {
        "clean_exec_positive_rate": _rate(work["clean_exec_positive"]),
        "dirty_positive_rate": _rate(work["dirty_positive"]),
        "bad_MAE_rate": _rate(work["bad_MAE"]),
        "timeout_rate": _rate(work["timeout_label"]),
    }
    rows: list[dict[str, Any]] = []
    for archetype_id, group in work.groupby(id_col, dropna=False):
        side_dist = group["side_name"].astype(str).value_counts(normalize=True).to_dict()
        spread_dist = group["spread_bucket"].astype(str).value_counts(normalize=True).to_dict()
        monthly = group.groupby("month")["u_econ_net"].mean()
        rec = {
            "fold_id": fold_id,
            "archetype_family": family,
            "archetype_id": str(archetype_id),
            "row_count": int(len(group)),
            "support_pct": float(len(group) / max(len(work), 1)),
            "mean_u_econ_net": _mean(group["u_econ_net"]),
            "median_u_econ_net": _q(group["u_econ_net"], 0.50),
            "clean_exec_positive_rate": _rate(group["clean_exec_positive"]),
            "dirty_positive_rate": _rate(group["dirty_positive"]),
            "bad_MAE_rate": _rate(group["bad_MAE"]),
            "timeout_rate": _rate(group["timeout_label"]),
            "mean_MAE": _mean(group.get("mae_norm")),
            "mean_MFE": _mean(group.get("mfe_norm")),
            "mean_hold_time": _mean(group.get("underwater_bars_before_mfe_1r")),
            "worst_month_utility": float(monthly.min()) if len(monthly) else float("nan"),
            "monthly_stability": float(monthly.gt(0.0).mean()) if len(monthly) else float("nan"),
            "side_distribution": json.dumps(_json_safe(side_dist), sort_keys=True),
            "spread_distribution": json.dumps(_json_safe(spread_dist), sort_keys=True),
        }
        rec["quality_label"] = _quality_label(pd.Series(rec), global_rates, min_support)
        rec["failure_label"] = rec["quality_label"] if rec["quality_label"] in {"bad_MAE_prone", "timeout_prone", "dirty_positive", "negative_utility"} else "none"
        rec["human_readable_name"] = f"{family}:{rec['archetype_id']}:{rec['quality_label']}"
        rows.append(rec)
    return pd.DataFrame(rows)


def _reliability_table(
    train: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    fold_id: str,
    family: str,
    id_col: str,
    confidence_col: str,
    shrinkage_k: float,
) -> pd.DataFrame:
    work = train.reset_index(drop=True).copy()
    if id_col not in work.columns:
        work = pd.concat([work, assignments[[id_col]].reset_index(drop=True)], axis=1)
    global_clean = _rate(work["clean_exec_positive"])
    global_dirty = _rate(work["dirty_positive"])
    global_bad = _rate(work["bad_MAE"])
    global_timeout = _rate(work["timeout_label"])
    global_fp = _rate(_num(work["u_econ_net"], index=work.index).le(0.0))
    rows: list[dict[str, Any]] = []
    keys = [id_col, "side_name", "spread_bucket", confidence_col]
    for values, group in work.groupby(keys, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        support = int(len(group))
        weight = float(support / (support + float(shrinkage_k)))
        high_conf = group[confidence_col].astype(str).str.startswith("high")
        overconf = high_conf & group["clean_exec_positive"].eq(0)
        rec = {
            "fold_id": fold_id,
            "archetype_family": family,
            "archetype_id": str(values[0]),
            "side": str(values[1]),
            "spread_bucket": str(values[2]),
            "base_score_decile": str(values[3]),
            "support": support,
            "base_clean_rate": _rate(group["clean_exec_positive"]),
            "base_dirty_positive_rate": _rate(group["dirty_positive"]),
            "base_bad_MAE_rate": _rate(group["bad_MAE"]),
            "base_timeout_rate": _rate(group["timeout_label"]),
            "base_false_positive_rate": _rate(_num(group["u_econ_net"], index=group.index).le(0.0)),
            "base_overconfidence_rate": _rate(overconf),
            "mean_utility_residual": _mean(group["base_utility_residual"]),
            "mean_clean_residual": _mean(group["base_clean_residual"]),
            "mean_bad_mae_residual": _mean(group["base_bad_mae_residual"]),
            "mean_timeout_residual": _mean(group["base_timeout_residual"]),
            "shrinkage_weight": weight,
        }
        rec["shrunk_base_clean_rate"] = weight * rec["base_clean_rate"] + (1.0 - weight) * global_clean
        rec["shrunk_base_dirty_positive_rate"] = weight * rec["base_dirty_positive_rate"] + (1.0 - weight) * global_dirty
        rec["shrunk_base_bad_MAE_rate"] = weight * rec["base_bad_MAE_rate"] + (1.0 - weight) * global_bad
        rec["shrunk_base_timeout_rate"] = weight * rec["base_timeout_rate"] + (1.0 - weight) * global_timeout
        rec["shrunk_base_false_positive_rate"] = weight * rec["base_false_positive_rate"] + (1.0 - weight) * global_fp
        rows.append(rec)
    return pd.DataFrame(rows)


def _apply_train_priors(
    valid: pd.DataFrame,
    profiles: pd.DataFrame,
    reliability: pd.DataFrame,
    *,
    id_col: str,
    confidence_col: str,
) -> pd.DataFrame:
    out = valid.copy()
    profile_cols = [
        "fold_id",
        "archetype_family",
        "archetype_id",
        "quality_label",
        "failure_label",
        "human_readable_name",
        "row_count",
        "support_pct",
        "clean_exec_positive_rate",
        "dirty_positive_rate",
        "bad_MAE_rate",
        "timeout_rate",
        "mean_u_econ_net",
    ]
    prof = profiles[[col for col in profile_cols if col in profiles.columns]].copy()
    prof = prof.rename(
        columns={
            "row_count": "prior_support",
            "clean_exec_positive_rate": "prior_base_clean_rate",
            "dirty_positive_rate": "prior_base_dirty_positive_rate",
            "bad_MAE_rate": "prior_base_bad_MAE_rate",
            "timeout_rate": "prior_base_timeout_rate",
            "mean_u_econ_net": "prior_base_mean_u_econ_net",
            "support_pct": "prior_support_pct",
        }
    )
    out = out.merge(
        prof,
        left_on=["fold_id", "archetype_family", id_col],
        right_on=["fold_id", "archetype_family", "archetype_id"],
        how="left",
        suffixes=("", "_profile"),
    )
    rel_cols = [
        "fold_id",
        "archetype_family",
        "archetype_id",
        "side",
        "spread_bucket",
        "base_score_decile",
        "support",
        "shrunk_base_clean_rate",
        "shrunk_base_dirty_positive_rate",
        "shrunk_base_bad_MAE_rate",
        "shrunk_base_timeout_rate",
        "shrunk_base_false_positive_rate",
        "mean_utility_residual",
        "mean_clean_residual",
        "mean_bad_mae_residual",
        "mean_timeout_residual",
    ]
    rel = reliability[[col for col in rel_cols if col in reliability.columns]].copy()
    rel = rel.rename(columns={"support": "prior_reliability_support"})
    out = out.merge(
        rel,
        left_on=["fold_id", "archetype_family", id_col, "side_name", "spread_bucket", confidence_col],
        right_on=["fold_id", "archetype_family", "archetype_id", "side", "spread_bucket", "base_score_decile"],
        how="left",
        suffixes=("", "_rel"),
    )
    return out


def _encode_category(train: pd.DataFrame, valid: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    train_values = train.get(col, pd.Series("missing", index=train.index)).astype(str).fillna("missing")
    valid_values = valid.get(col, pd.Series("missing", index=valid.index)).astype(str).fillna("missing")
    mapping = {value: i for i, value in enumerate(sorted(train_values.unique()))}
    train_codes = train_values.map(mapping).fillna(-1).astype(np.float32)
    valid_codes = valid_values.map(mapping).fillna(-1).astype(np.float32)
    return train_codes, valid_codes


def _model_matrix(train: pd.DataFrame, valid: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data: dict[str, pd.Series] = {}
    valid_data: dict[str, pd.Series] = {}
    for col in feature_cols:
        if col not in train.columns and col not in valid.columns:
            continue
        if col.endswith("_id") or col in {
            "ae_gmm_archetype_id",
            "aegmm_cluster",
            "side_aegmm_cluster",
            "source_semantic_family",
            "source_volatility_state",
            "source_pressure_state",
            "source_trend_state",
            "source_score_intensity_tag",
            "base_score_decile",
            "quality_label",
            "failure_label",
            "side_name",
        }:
            train_data[col], valid_data[col] = _encode_category(train, valid, col)
        else:
            train_data[col] = _num(train.get(col), index=train.index).astype(np.float32)
            valid_data[col] = _num(valid.get(col), index=valid.index).astype(np.float32)
    train_x = pd.DataFrame(train_data, index=train.index)
    valid_x = pd.DataFrame(valid_data, index=valid.index)
    medians = train_x.replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(medians).astype(np.float32)
    valid_x = valid_x.replace([np.inf, -np.inf], np.nan).fillna(medians).astype(np.float32)
    return train_x, valid_x


def _meta_objective(frame: pd.DataFrame) -> pd.Series:
    """Risk-adjusted utility target for the diagnostic handoff ablation."""
    u = _num(frame["u_econ_net"], index=frame.index).fillna(0.0)
    clean = _num(frame["clean_exec_positive"], index=frame.index).fillna(0.0)
    bad = _num(frame["bad_MAE"], index=frame.index).fillna(0.0)
    timeout = _num(frame["timeout_label"], index=frame.index).fillna(0.0)
    dirty = _num(frame["dirty_positive"], index=frame.index).fillna(0.0)
    return (u + 0.0200 * clean - 0.0200 * bad - 0.0100 * timeout - 0.0050 * dirty).astype(np.float32)


def _fit_regressor_score(train_x: pd.DataFrame, y: pd.Series, valid_x: pd.DataFrame, seed: int) -> np.ndarray:
    if HistGradientBoostingRegressor is None:
        raise RuntimeError(f"scikit-learn is required for learned ablation scores: {_SKLEARN_IMPORT_ERROR}")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.045,
        max_iter=90,
        max_leaf_nodes=15,
        min_samples_leaf=180,
        l2_regularization=0.10,
        random_state=int(seed),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
    )
    model.fit(train_x, _num(y, index=train_x.index).fillna(0.0).to_numpy(dtype=np.float32))
    return np.asarray(model.predict(valid_x), dtype=np.float32)


def _add_learned_ablation_scores(train: pd.DataFrame, valid: pd.DataFrame, seed: int, feature_cols: list[str]) -> pd.DataFrame:
    out = valid.copy()
    posterior_cols = sorted([col for col in valid.columns if str(col).startswith("gmm_posterior_")])
    latent_cols = sorted([col for col in valid.columns if str(col).startswith("ae_latent_")])
    base_cols = ["base_score"]
    m1_cols = base_cols + ["ae_gmm_archetype_id", "side_name"]
    m2_cols = m1_cols + [
        "gmm_max_posterior",
        "gmm_entropy_oof",
        "gmm_distance_to_centroid",
        "ae_reconstruction_error_oof",
    ] + posterior_cols + latent_cols
    m3_cols = m2_cols + [
        "prior_support_pct",
        "prior_base_clean_rate",
        "prior_base_dirty_positive_rate",
        "prior_base_bad_MAE_rate",
        "prior_base_timeout_rate",
        "prior_base_mean_u_econ_net",
        "quality_label",
        "failure_label",
    ]
    m4_cols = m3_cols + list(feature_cols) + [
        "base_score_decile",
        "prior_reliability_support",
        "shrunk_base_clean_rate",
        "shrunk_base_dirty_positive_rate",
        "shrunk_base_bad_MAE_rate",
        "shrunk_base_timeout_rate",
        "shrunk_base_false_positive_rate",
        "mean_utility_residual",
        "mean_clean_residual",
        "mean_bad_mae_residual",
        "mean_timeout_residual",
        "aegmm_cluster",
        "side_aegmm_cluster",
        "aegmm_entropy_bin",
        "aegmm_distance_bin",
        "aegmm_expected_distance_bin",
        "reconstruction_bin",
        "dae_reconstruction_bin",
        "cluster_speed_bin",
        "cluster_acceleration_bin",
        "latent_speed_bin",
        "latent_acceleration_bin",
        "source_semantic_family",
        "source_volatility_state",
        "source_pressure_state",
        "source_trend_state",
        "source_score_intensity_tag",
    ]
    feature_sets = {
        "score_M0_no_archetype": base_cols,
        "score_M1_hard_archetype_id": m1_cols,
        "score_M2_gmm_posterior_entropy": m2_cols,
        "score_M3_archetype_outcome_priors": m3_cols,
        "score_M4_base_reliability_priors": m4_cols,
    }
    target = _meta_objective(train)
    for score_col, cols in feature_sets.items():
        train_x, valid_x = _model_matrix(train, out, cols)
        if train_x.empty:
            out[score_col] = _num(out["base_score"], index=out.index).rank(pct=True).to_numpy(dtype=np.float32)
            continue
        out[score_col] = _fit_regressor_score(train_x, target, valid_x, seed)
    return out


def _add_residuals(train: pd.DataFrame, valid: pd.DataFrame, confidence_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work_train = train.copy()
    work_valid = valid.copy()
    priors = (
        work_train.groupby(confidence_col, dropna=False)
        .agg(
            pred_u=("u_econ_net", "mean"),
            pred_clean=("clean_exec_positive", "mean"),
            pred_bad=("bad_MAE", "mean"),
            pred_timeout=("timeout_label", "mean"),
        )
        .reset_index()
    )
    global_vals = {
        "pred_u": _mean(work_train["u_econ_net"]),
        "pred_clean": _rate(work_train["clean_exec_positive"]),
        "pred_bad": _rate(work_train["bad_MAE"]),
        "pred_timeout": _rate(work_train["timeout_label"]),
    }
    for frame in (work_train, work_valid):
        merged = frame[[confidence_col]].merge(priors, on=confidence_col, how="left")
        for col, value in global_vals.items():
            frame[col] = _num(merged[col], index=frame.index).fillna(value).to_numpy(dtype=np.float32)
        frame["base_utility_residual"] = _num(frame["u_econ_net"], index=frame.index) - _num(frame["pred_u"], index=frame.index)
        frame["base_clean_residual"] = _num(frame["clean_exec_positive"], index=frame.index) - _num(frame["pred_clean"], index=frame.index)
        frame["base_bad_mae_residual"] = _num(frame["bad_MAE"], index=frame.index) - _num(frame["pred_bad"], index=frame.index)
        frame["base_timeout_residual"] = _num(frame["timeout_label"], index=frame.index) - _num(frame["pred_timeout"], index=frame.index)
    return work_train, work_valid


def _valid_row_labels(valid: pd.DataFrame) -> pd.DataFrame:
    out = valid.copy()
    high_conf = out["base_score_decile"].astype(str).str.startswith("high")
    low_or_mid = ~high_conf
    out["base_true_positive_clean"] = (out["base_selected"].eq(1) & out["clean_exec_positive"].eq(1)).astype(np.int8)
    out["base_dirty_positive"] = (out["base_selected"].eq(1) & out["dirty_positive"].eq(1)).astype(np.int8)
    out["base_false_positive"] = (out["base_selected"].eq(1) & _num(out["u_econ_net"], index=out.index).le(0.0)).astype(np.int8)
    out["base_timeout_failure"] = (out["base_selected"].eq(1) & out["timeout_label"].eq(1)).astype(np.int8)
    out["base_bad_mae_failure"] = (out["base_selected"].eq(1) & out["bad_MAE"].eq(1)).astype(np.int8)
    out["base_overconfident"] = (high_conf & out["clean_exec_positive"].eq(0)).astype(np.int8)
    out["base_underconfident"] = (low_or_mid & out["clean_exec_positive"].eq(1)).astype(np.int8)
    out["base_overconfident_dirty"] = (high_conf & out["dirty_positive"].eq(1)).astype(np.int8)
    out["base_overconfident_timeout"] = (high_conf & out["timeout_label"].eq(1)).astype(np.int8)
    out["base_overconfident_bad_MAE"] = (high_conf & out["bad_MAE"].eq(1)).astype(np.int8)
    out["base_high_score_dirty_positive_bad_MAE"] = (
        high_conf & _num(out["u_econ_net"], index=out.index).gt(0.0) & out["dirty_positive"].eq(1) & out["bad_MAE"].eq(1)
    ).astype(np.int8)
    return out


def _score_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if all(col in out.columns and _num(out[col], index=out.index).notna().any() for col in LEARNED_SCORE_COLUMNS):
        return out
    base = _num(out["base_score"], index=out.index).rank(pct=True).fillna(0.0)
    clean_prior = _num(out.get("prior_base_clean_rate"), index=out.index).fillna(_rate(out["clean_exec_positive"]))
    dirty_prior = _num(out.get("prior_base_dirty_positive_rate"), index=out.index).fillna(_rate(out["dirty_positive"]))
    bad_prior = _num(out.get("prior_base_bad_MAE_rate"), index=out.index).fillna(_rate(out["bad_MAE"]))
    timeout_prior = _num(out.get("prior_base_timeout_rate"), index=out.index).fillna(_rate(out["timeout_label"]))
    u_prior = _num(out.get("prior_base_mean_u_econ_net"), index=out.index).fillna(_mean(out["u_econ_net"]))
    rel_clean = _num(out.get("shrunk_base_clean_rate"), index=out.index).fillna(clean_prior)
    rel_bad = _num(out.get("shrunk_base_bad_MAE_rate"), index=out.index).fillna(bad_prior)
    rel_timeout = _num(out.get("shrunk_base_timeout_rate"), index=out.index).fillna(timeout_prior)
    rel_resid = _num(out.get("mean_utility_residual"), index=out.index).fillna(0.0)
    posterior = _num(out.get("gmm_max_posterior"), index=out.index).fillna(0.0)
    entropy = _num(out.get("gmm_entropy_oof"), index=out.index).fillna(1.0)
    if "score_M0_no_archetype" not in out.columns:
        out["score_M0_no_archetype"] = base
    if "score_M1_hard_archetype_id" not in out.columns:
        out["score_M1_hard_archetype_id"] = base + 0.20 * clean_prior - 0.12 * bad_prior - 0.08 * timeout_prior
    if "score_M2_gmm_posterior_entropy" not in out.columns:
        out["score_M2_gmm_posterior_entropy"] = out["score_M1_hard_archetype_id"] + 0.05 * posterior - 0.05 * entropy
    if "score_M3_archetype_outcome_priors" not in out.columns:
        out["score_M3_archetype_outcome_priors"] = base + 0.30 * clean_prior + 4.0 * u_prior - 0.18 * dirty_prior - 0.22 * bad_prior - 0.10 * timeout_prior
    if "score_M4_base_reliability_priors" not in out.columns:
        out["score_M4_base_reliability_priors"] = (
            out["score_M3_archetype_outcome_priors"] + 0.25 * rel_clean + 3.0 * rel_resid - 0.20 * rel_bad - 0.12 * rel_timeout
        )
    return out


def _eval_score(frame: pd.DataFrame, score_col: str, top_frac: float) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    for _, group in frame.groupby("fold_id", dropna=False):
        score = _num(group[score_col], index=group.index)
        valid = group[score.notna()].copy()
        if valid.empty:
            continue
        keep = max(1, int(math.ceil(len(valid) * float(top_frac))))
        rows.append(valid.assign(__score__=score.loc[valid.index]).sort_values("__score__", ascending=False).head(keep))
    selected = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return {"score_model": score_col, "selected_rows": 0}
    month_u = selected.groupby("month")["u_econ_net"].mean()
    side_share = selected["side_name"].astype(str).value_counts(normalize=True)
    return {
        "score_model": score_col,
        "selected_rows": int(len(selected)),
        "mean_u": _mean(selected["u_econ_net"]),
        "worst_month_u": float(month_u.min()) if len(month_u) else float("nan"),
        "clean_positive_rate": _rate(selected["clean_exec_positive"]),
        "bad_MAE_rate": _rate(selected["bad_MAE"]),
        "timeout_rate": _rate(selected["timeout_label"]),
        "dirty_positive_rate": _rate(selected["dirty_positive"]),
        "dominant_side_share": float(side_share.iloc[0]) if len(side_share) else float("nan"),
    }


def _test_separation(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rows.empty:
        return pd.DataFrame(), {"status": "fail", "reason": "no_oof_rows"}
    global_clean = _rate(rows["clean_exec_positive"])
    global_bad = _rate(rows["bad_MAE"])
    global_timeout = _rate(rows["timeout_label"])
    global_dirty = _rate(rows["dirty_positive"])
    out_rows: list[dict[str, Any]] = []
    for (family, arch), group in rows.groupby(["archetype_family", "ae_gmm_archetype_id"], dropna=False):
        rec = {
            "archetype_family": family,
            "archetype_id": arch,
            "rows": int(len(group)),
            "clean_rate": _rate(group["clean_exec_positive"]),
            "bad_MAE_rate": _rate(group["bad_MAE"]),
            "timeout_rate": _rate(group["timeout_label"]),
            "dirty_positive_rate": _rate(group["dirty_positive"]),
        }
        rec["clean_rate_lift"] = _safe_ratio(rec["clean_rate"], global_clean)
        rec["bad_MAE_lift"] = _safe_ratio(rec["bad_MAE_rate"], global_bad)
        rec["timeout_lift"] = _safe_ratio(rec["timeout_rate"], global_timeout)
        rec["dirty_positive_lift"] = _safe_ratio(rec["dirty_positive_rate"], global_dirty)
        rec["has_material_separation"] = bool(
            rec["clean_rate_lift"] >= 1.10
            or rec["clean_rate_lift"] <= 0.90
            or rec["bad_MAE_lift"] >= 1.10
            or rec["bad_MAE_lift"] <= 0.90
            or (math.isfinite(rec["timeout_lift"]) and (rec["timeout_lift"] >= 1.25 or rec["timeout_lift"] <= 0.75))
            or rec["dirty_positive_lift"] >= 1.10
            or rec["dirty_positive_lift"] <= 0.90
        )
        out_rows.append(rec)
    table = pd.DataFrame(out_rows)
    separated = int(table["has_material_separation"].sum()) if not table.empty else 0
    return table, {
        "status": "pass" if separated >= 2 else "fail",
        "separated_archetypes": separated,
        "global_clean_rate": global_clean,
        "global_bad_MAE_rate": global_bad,
        "global_timeout_rate": global_timeout,
        "global_dirty_positive_rate": global_dirty,
    }


def _test_stability(profiles: pd.DataFrame, seed_profiles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold_id, group in profiles.groupby("fold_id", dropna=False):
        label_counts = group["quality_label"].astype(str).value_counts(normalize=True).to_dict()
        main_support = np.sort(_num(group["support_pct"]).to_numpy(dtype=float))
        for seed, seed_group in seed_profiles[seed_profiles["fold_id"].eq(fold_id)].groupby("seed", dropna=False):
            seed_support = np.sort(_num(seed_group["support_pct"]).to_numpy(dtype=float))
            size = min(len(main_support), len(seed_support))
            support_l1 = float(np.abs(main_support[:size] - seed_support[:size]).sum()) if size else float("nan")
            seed_labels = set(seed_group["quality_label"].astype(str))
            main_labels = set(label_counts)
            label_jaccard = float(len(main_labels & seed_labels) / max(len(main_labels | seed_labels), 1))
            rows.append(
                {
                    "fold_id": fold_id,
                    "seed": int(seed),
                    "support_l1_vs_main": support_l1,
                    "quality_label_jaccard_vs_main": label_jaccard,
                    "main_quality_labels": ",".join(sorted(main_labels)),
                    "seed_quality_labels": ",".join(sorted(seed_labels)),
                }
            )
    table = pd.DataFrame(rows)
    mean_l1 = _mean(table.get("support_l1_vs_main"))
    mean_j = _mean(table.get("quality_label_jaccard_vs_main"))
    return table, {
        "status": "pass" if (math.isfinite(mean_l1) and mean_l1 <= 0.75 and mean_j >= 0.50) else "fail",
        "mean_support_l1": mean_l1,
        "mean_quality_label_jaccard": mean_j,
    }


def _test_base_reliability(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (arch, bucket), group in rows.groupby(["ae_gmm_archetype_id", "base_score_decile"], dropna=False):
        high = str(bucket).startswith("high")
        out.append(
            {
                "archetype_id": arch,
                "base_confidence_bucket": bucket,
                "rows": int(len(group)),
                "is_high_confidence": bool(high),
                "base_overconfident_bad_MAE_rate": _rate(group["base_overconfident_bad_MAE"]),
                "base_timeout_failure_rate": _rate(group["base_timeout_failure"]),
                "base_false_positive_rate": _rate(group["base_false_positive"]),
                "base_clean_hit_rate": _rate(group["base_true_positive_clean"]),
                "mean_base_residual": _mean(group["base_utility_residual"]),
                "mean_bad_mae_residual": _mean(group["base_bad_mae_residual"]),
                "mean_timeout_residual": _mean(group["base_timeout_residual"]),
            }
        )
    table = pd.DataFrame(out)
    high_rows = table[table["is_high_confidence"].astype(bool)] if not table.empty else pd.DataFrame()
    spread_bad = float(high_rows["base_overconfident_bad_MAE_rate"].max() - high_rows["base_overconfident_bad_MAE_rate"].min()) if len(high_rows) >= 2 else float("nan")
    spread_resid = float(table["mean_base_residual"].max() - table["mean_base_residual"].min()) if len(table) >= 2 else float("nan")
    return table, {
        "status": "pass" if ((math.isfinite(spread_bad) and spread_bad >= 0.05) or (math.isfinite(spread_resid) and spread_resid >= 0.005)) else "fail",
        "high_conf_overconf_bad_MAE_spread": spread_bad,
        "utility_residual_spread": spread_resid,
    }


def _test_ablation(rows: pd.DataFrame, top_frac: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored = _score_ablation(rows)
    models = [
        "score_M0_no_archetype",
        "score_M1_hard_archetype_id",
        "score_M2_gmm_posterior_entropy",
        "score_M3_archetype_outcome_priors",
        "score_M4_base_reliability_priors",
    ]
    table = pd.DataFrame([_eval_score(scored, model, top_frac) for model in models])
    base = table[table["score_model"].eq("score_M0_no_archetype")].iloc[0]
    candidates = table[table["score_model"].isin(["score_M3_archetype_outcome_priors", "score_M4_base_reliability_priors"])]
    pass_rows = candidates[
        (_num(candidates["mean_u"], index=candidates.index) > 0.0)
        & (_num(candidates["worst_month_u"], index=candidates.index) > 0.0)
        & (_num(candidates["dominant_side_share"], index=candidates.index) <= 0.85)
        & (
            (_num(candidates["bad_MAE_rate"], index=candidates.index) <= float(base["bad_MAE_rate"]) - 0.005)
            | (_num(candidates["timeout_rate"], index=candidates.index) <= float(base["timeout_rate"]) - 0.002)
            | (_num(candidates["clean_positive_rate"], index=candidates.index) >= float(base["clean_positive_rate"]) + 0.005)
            | (_num(candidates["mean_u"], index=candidates.index) >= float(base["mean_u"]) + 0.001)
        )
    ]
    return table, {
        "status": "pass" if not pass_rows.empty else "fail",
        "best_model": str(pass_rows.iloc[0]["score_model"]) if not pass_rows.empty else None,
        "baseline_mean_u": float(base.get("mean_u", np.nan)),
        "baseline_bad_MAE_rate": float(base.get("bad_MAE_rate", np.nan)),
    }


def _write_formula_spec(path: Path) -> None:
    text = """# ARCHETYPE_META_HANDOFF_V1 scoring formula spec
version: ARCHETYPE_META_HANDOFF_V1
score_components:
  - base_score
  - gmm_posterior
  - gmm_entropy
  - archetype_outcome_priors
  - base_reliability_priors
  - base_residual_targets
candidate_formulas:
  M0_no_archetype: base_score_rank
  M1_hard_archetype_id: base_score_rank + clean_prior - bad_mae_prior - timeout_prior
  M2_gmm_posterior_entropy: M1 + posterior_confidence - entropy_penalty
  M3_archetype_outcome_priors: base_score_rank + clean/u priors - dirty/bad_mae/timeout priors
  M4_base_reliability_priors: M3 + reliability clean/residual priors - reliability bad_mae/timeout priors
veto_candidates:
  - veto_bad_mae_archetype
  - veto_timeout_archetype
  - veto_low_support_high_entropy
promotion_metrics:
  - lower_bad_MAE
  - lower_timeout
  - higher_clean_positive_rate
  - positive_utility
  - stable_worst_month_behavior
  - controlled_side_spread_concentration
leakage_contract:
  assignment: scaler_pca_gmm_fit_train_only_validation_frozen
  priors: train_outcomes_only
  residuals: fold_train_score_bucket_priors_then_validation_residuals
"""
    path.write_text(text)


def build_handoff(
    *,
    ledger_path: Path,
    out_dir: Path,
    n_components: int = 6,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    min_train_months: int = 1,
    n_latent: int = 4,
    shrinkage_k: float = 200.0,
    min_support_frac: float = 0.005,
    top_frac: float = 0.10,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(ledger_path)
    folds = _folds_from_months(ledger, int(min_train_months))
    if not folds:
        raise ValueError("Need at least two months to build OOF archetype folds")
    feature_cols = _feature_columns(ledger)
    main_seed = int(seeds[0])

    row_frames: list[pd.DataFrame] = []
    profile_frames: list[pd.DataFrame] = []
    reliability_frames: list[pd.DataFrame] = []
    seed_profile_frames: list[pd.DataFrame] = []
    leakage_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        train = ledger[ledger["month"].astype(str).isin(fold["train_months"])].copy()
        valid = ledger[ledger["month"].astype(str).eq(str(fold["valid_month"]))].copy()
        if train.empty or valid.empty:
            continue
        train_assign, valid_assign, fit_info = _fit_assign_fold(
            train,
            valid,
            feature_cols=feature_cols,
            n_components=n_components,
            seed=main_seed,
            n_latent=n_latent,
        )
        train = train.reset_index(drop=True)
        valid = valid.reset_index(drop=True)
        train_assign = train_assign.reset_index(drop=True)
        valid_assign = valid_assign.reset_index(drop=True)
        train["fold_id"] = fold_id
        valid["fold_id"] = fold_id
        train["archetype_family"] = "A_AE_GMM_STATE"
        valid["archetype_family"] = "A_AE_GMM_STATE"
        train["base_score_decile"], valid["base_score_decile"] = _train_score_buckets(train, valid)
        train = pd.concat([train, train_assign], axis=1)
        valid = pd.concat([valid, valid_assign], axis=1)
        train, valid = _add_residuals(train, valid, "base_score_decile")
        min_support = max(20, int(math.ceil(len(train) * float(min_support_frac))))
        profiles = _profile_table(
            train,
            train[["ae_gmm_archetype_id"]],
            fold_id=fold_id,
            family="A_AE_GMM_STATE",
            id_col="ae_gmm_archetype_id",
            min_support=min_support,
        )
        reliability = _reliability_table(
            train,
            train[["ae_gmm_archetype_id"]],
            fold_id=fold_id,
            family="A_AE_GMM_STATE",
            id_col="ae_gmm_archetype_id",
            confidence_col="base_score_decile",
            shrinkage_k=shrinkage_k,
        )
        train = _apply_train_priors(
            train,
            profiles,
            reliability,
            id_col="ae_gmm_archetype_id",
            confidence_col="base_score_decile",
        )
        valid = _apply_train_priors(
            valid,
            profiles,
            reliability,
            id_col="ae_gmm_archetype_id",
            confidence_col="base_score_decile",
        )
        train = _valid_row_labels(train)
        valid = _valid_row_labels(valid)
        valid = _add_learned_ablation_scores(train, valid, main_seed, feature_cols)
        row_frames.append(valid)
        profile_frames.append(profiles)
        reliability_frames.append(reliability)
        leakage_rows.append({"fold_id": fold_id, **fit_info, "train_months": ",".join(fold["train_months"]), "valid_month": fold["valid_month"]})
        for seed in seeds:
            seed_train_assign, _, _ = _fit_assign_fold(
                train,
                valid,
                feature_cols=feature_cols,
                n_components=n_components,
                seed=int(seed),
                n_latent=n_latent,
            )
            seed_prof = _profile_table(
                train,
                seed_train_assign.reset_index(drop=True)[["ae_gmm_archetype_id"]],
                fold_id=fold_id,
                family="A_AE_GMM_STATE",
                id_col="ae_gmm_archetype_id",
                min_support=min_support,
            )
            seed_prof["seed"] = int(seed)
            seed_profile_frames.append(seed_prof)

    row_features = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    reliability = pd.concat(reliability_frames, ignore_index=True) if reliability_frames else pd.DataFrame()
    seed_profiles = pd.concat(seed_profile_frames, ignore_index=True) if seed_profile_frames else pd.DataFrame()
    leakage = pd.DataFrame(leakage_rows)

    separation_table, separation_test = _test_separation(row_features)
    stability_table, stability_test = _test_stability(profiles, seed_profiles)
    reliability_test_table, reliability_test = _test_base_reliability(row_features)
    ablation_table, ablation_test = _test_ablation(row_features, top_frac)
    leakage_test = {
        "status": "pass"
        if not leakage.empty
        and leakage["scaler_fit_scope"].eq("outer_train_only").all()
        and leakage["ae_fit_scope"].eq("outer_train_only").all()
        and leakage["gmm_fit_scope"].eq("outer_train_only").all()
        and leakage["validation_assignment_scope"].eq("frozen_train_artifacts").all()
        else "fail",
        "folds": int(len(leakage)),
    }

    tests = {
        "leakage_test": leakage_test,
        "stability_test": stability_test,
        "separation_test": separation_test,
        "base_reliability_test": reliability_test,
        "handoff_ablation_test": ablation_test,
    }
    overall_status = "pass" if all(v.get("status") == "pass" for v in tests.values()) else "needs_iteration"

    paths = {
        "row_features": out_dir / "archetype_row_features.parquet",
        "profile_table": out_dir / "archetype_profile_table.parquet",
        "reliability_table": out_dir / "base_reliability_by_archetype.parquet",
        "leakage_report": out_dir / "archetype_leakage_report.csv",
        "stability_report": out_dir / "archetype_stability_report.csv",
        "separation_report": out_dir / "archetype_separation_report.csv",
        "base_reliability_test": out_dir / "base_reliability_test.csv",
        "handoff_ablation": out_dir / "handoff_ablation_report.csv",
        "meta_scoring_spec": out_dir / "meta_scoring_formula_spec.yaml",
        "acceptance": out_dir / "archetype_acceptance_tests.json",
        "report": out_dir / "archetype_meta_handoff_v1_report.md",
        "manifest": out_dir / "manifest.json",
    }
    row_features.to_parquet(paths["row_features"], index=False)
    profiles.to_parquet(paths["profile_table"], index=False)
    reliability.to_parquet(paths["reliability_table"], index=False)
    leakage.to_csv(paths["leakage_report"], index=False)
    stability_table.to_csv(paths["stability_report"], index=False)
    separation_table.to_csv(paths["separation_report"], index=False)
    reliability_test_table.to_csv(paths["base_reliability_test"], index=False)
    ablation_table.to_csv(paths["handoff_ablation"], index=False)
    _write_formula_spec(paths["meta_scoring_spec"])
    paths["acceptance"].write_text(json.dumps(_json_safe(tests), indent=2, sort_keys=True))
    manifest = {
        "generated_by": "build_archetype_meta_handoff_v1",
        "status": overall_status,
        "ledger_path": str(ledger_path),
        "rows_in_ledger": int(len(ledger)),
        "oof_validation_rows": int(len(row_features)),
        "folds": folds,
        "feature_cols": feature_cols,
        "n_components": int(n_components),
        "seeds": [int(v) for v in seeds],
        "outputs": {key: str(value) for key, value in paths.items()},
        "acceptance_tests": tests,
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    _write_report(paths["report"], manifest, ablation_table)
    return manifest


def _write_report(path: Path, manifest: dict[str, Any], ablation: pd.DataFrame) -> None:
    tests = manifest["acceptance_tests"]
    lines = [
        "# Archetype Meta-Handoff V1",
        "",
        "## Scope",
        "",
        "Leakage-safe OOF archetype and base-reliability handoff for the S52 candidate ledger.",
        "Policy optimisation is intentionally out of scope here.",
        "",
        "## Status",
        "",
        f"- status: `{manifest['status']}`",
        f"- ledger rows: `{manifest['rows_in_ledger']}`",
        f"- OOF validation rows: `{manifest['oof_validation_rows']}`",
        f"- folds: `{len(manifest['folds'])}`",
        "",
        "## Five Tests",
        "",
    ]
    for name, payload in tests.items():
        lines.append(f"- {name}: `{payload.get('status')}`")
    lines += [
        "",
        "## Ablation",
        "",
        ablation.to_markdown(index=False) if not ablation.empty else "_No ablation rows._",
        "",
        "## Outputs",
        "",
    ]
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--min-train-months", type=int, default=1)
    parser.add_argument("--n-latent", type=int, default=4)
    parser.add_argument("--shrinkage-k", type=float, default=200.0)
    parser.add_argument("--min-support-frac", type=float, default=0.005)
    parser.add_argument("--top-frac", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = tuple(int(part.strip()) for part in str(args.seeds).split(",") if part.strip())
    manifest = build_handoff(
        ledger_path=args.ledger,
        out_dir=args.out_dir,
        n_components=int(args.n_components),
        seeds=seeds or DEFAULT_SEEDS,
        min_train_months=int(args.min_train_months),
        n_latent=int(args.n_latent),
        shrinkage_k=float(args.shrinkage_k),
        min_support_frac=float(args.min_support_frac),
        top_frac=float(args.top_frac),
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
