#!/usr/bin/env python3
"""Leakage-safe regime calibration test for meta OOS predictions.

The report answers whether side x archetype performance changes enough across
observable regimes to justify score calibration.  It uses only already
materialized prediction/handoff features:

* fit regime effects on prior OOS months,
* select a simple shape on an earlier validation slice,
* apply the selected effects to the next month,
* report OOS May/June metrics by side x archetype x regime.

No target/outcome columns are used to transform the OOS rows; they are used
only to score the frozen adjustment on the evaluation month.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_ev_calibration import CALIBRATION_POLICY_ID

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover - sklearn is present in normal runs.
    IsotonicRegression = None  # type: ignore[assignment]


DEFAULT_META_RUN = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "train_meta_regime_ablation_matrix_apr_may_jun_20260707/"
    "baseline_current_full_context"
)
DEFAULT_HANDOFF = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "s52_trailing_regime_meta_handoff_top30_allsafe_aegmm_fixedtargets_oos15_20260706/"
    "train_meta_regime_handoff.parquet"
)
DEFAULT_OUT = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "meta_oos_regime_calibration_20260708"
)


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    semantic: str
    candidates: tuple[str, ...]
    proxy_note: str = ""


REGIME_SPECS: tuple[RegimeSpec, ...] = (
    RegimeSpec("amihud_z_score", "Amihud/liquidity illiquidity z-score", ("amihud_z", "amihud_z_peer_resid", "q_iqr__amihud_z_peer_resid")),
    RegimeSpec("range_position_24h", "position in recent/prior 24h range", ("loc_prev_day_range_pos_24", "loc_range_pos_24", "loc_swing_range_pos_24")),
    RegimeSpec("vwap_distance_z", "distance to VWAP", ("loc_vwap_dev_z_24", "z_vwap_24", "dist_vwap_norm", "dist_vwap_atr")),
    RegimeSpec("trend_strength_3h_12h", "multi-horizon trend strength/alignment", ("trend_stack_3_6_12", "trend_alignment_3_6_12", "trend_strength_percentile", "trend_snr")),
    RegimeSpec("vol_compression_percentile", "volatility compression", ("vol_compression", "compression_score", "compression_ratio", "atr_compression_ratio")),
    RegimeSpec("vol_of_vol_z_score", "volatility-of-volatility / realized-vol instability", ("log_realized_vol_cp_z_8_32_96", "range_volatility_cp_z_8_32_96", "volatility_autocorr_48", "vol_z_x_regime_trend"), "proxy when explicit vol-of-vol z is unavailable"),
    RegimeSpec("atr_acceleration", "ATR/range acceleration", ("atr_change_rate", "atr_slope", "atr_expansion", "range_expansion_ratio")),
    RegimeSpec("funding_percentile", "funding crowding percentile/z", ("funding_rank_30d", "funding_z", "asset_funding_z", "funding_per_hour_z")),
    RegimeSpec("oi_change_z", "open-interest change z-score", ("oi_1d_chg_z", "oi_chg_z_8h", "oi_chg_8h_robust_z", "oi_value_1d_chg_z_90d")),
    RegimeSpec("adx_trend_strength", "ADX-like trend strength", ("adx_14", "adx_10", "adx_zscore", "adx_7")),
    RegimeSpec("bollinger_bandwidth_percentile", "Bollinger bandwidth / price-band width", ("bollinger_band_width", "range_24h_pct", "true_range_percentile")),
    RegimeSpec("trend_persistence", "trend/flow/funding persistence", ("flow_persistence", "funding_persistence", "trend_regime_stability", "trend_retest_success_rate")),
    RegimeSpec("score_dispersion_across_heads", "score disagreement across available heads", ("__derived_score_dispersion__",), "derived as abs(meta_score - base_score)"),
    RegimeSpec("gmm_entropy", "GMM posterior uncertainty", ("__derived_gmm_entropy__", "gmm_posterior_max"), "derived from available GMM posterior columns when possible"),
    RegimeSpec("mahalanobis_distance", "distance from train/state manifold", ("expected_mahalanobis", "state_spectral_top3_mahalanobis")),
    RegimeSpec("ae_reconstruction_error", "AE/state reconstruction error", ("state_spectral_top3_reconstruction_error", "dae_reconstruction_error_delta_1", "ae_reconstruction_error")),
    RegimeSpec("distance_from_train_centroid", "proxy distance from train cluster centroid", ("state_spectral_top3_mahalanobis", "expected_mahalanobis", "state_spectral_top3_reconstruction_ratio"), "proxy"),
    RegimeSpec("regime_posterior_delta", "change in regime posterior", ("cluster_speed", "gmm_posterior_max", "gmm_cluster_posterior_0"), "cluster speed proxy when posterior delta is unavailable"),
    RegimeSpec("regime_posterior_acceleration", "acceleration in regime posterior", ("cluster_speed", "gmm_posterior_max", "gmm_cluster_posterior_0"), "proxy; true acceleration not present in this handoff"),
    RegimeSpec("meta_model_uncertainty", "meta score uncertainty", ("__derived_meta_uncertainty__",), "derived from score entropy / closeness to 0.5"),
    RegimeSpec("leaf_support", "support/reliability of learned leaves/buckets", ("support_mean_frequency", "support_min_frequency", "support_unseen_bucket_share", "support_rare_bucket_share")),
    RegimeSpec("market_breadth", "market-wide breadth", ("market_breadth_24h", "market_breadth_4h", "market_breadth_1h", "mkt_oi_breadth_rising_24h")),
    RegimeSpec("market_dispersion", "cross-asset dispersion", ("market_dispersion_24h", "market_dispersion_4h", "cs_dispersion_ret_24h", "cs_ret_dispersion_24h_pct")),
    RegimeSpec("correlation_to_market_trend", "correlation / market-trend concentration", ("corr_btc_24h", "corr_eth_24h", "avg_pair_corr_24h", "corr_concentration_24h")),
    RegimeSpec("effective_rank_eigen_concentration", "effective rank / eigenvalue concentration", ("state_spectral_eig_effective_rank", "eig_effective_rank__breakout_all", "xs_cov_effective_rank__xs_asset_portable_all")),
    RegimeSpec("oi_shock_z_score", "OI shock z-score per asset", ("oi_value_1d_chg_z_90d", "oi_chg_8h_robust_z", "oi_1d_chg_z", "oi_chg_z_8h")),
    RegimeSpec("base_score_z_by_timestamp_side", "base score z-score by timestamp x side", ("base_score_z_by_timestamp_side", "base_score_z_by_timestamp", "base_signal_zscore_within_archetype")),
)

KEYS = ["__ts__", "__symbol__", "side_name"]
SCORE_COL = "score_meta_base_soft_label"
BASE_SCORE_COL = "score_base"
ARCH_COL = "archetype_policy_key"
OUTCOME_COLS = ["ev_after_1pct", "clean_exec", "dirty_positive", "full_path_bad_mae_1r", "timeout"]
TOP_SCOPES = {"top10": 0.90, "top20": 0.80, "top30": 0.70}
SHAPE_ORDER = ["flat", "linear", "monotone", "quadratic", "ushape", "bucketed"]


@dataclass
class ShapeFit:
    shape: str
    params: dict[str, Any]
    train_rows: int


def _schema_cols(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_schema(path).names


def _safe_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _load_predictions(meta_run: Path, months: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in months:
        matches = sorted((meta_run / "prediction_shards").glob(f"*{month}.parquet"))
        if not matches:
            raise FileNotFoundError(f"No prediction shard for {month} under {meta_run / 'prediction_shards'}")
        frame = pd.read_parquet(matches[-1])
        frames.append(frame)
    pred = pd.concat(frames, ignore_index=True, copy=False)
    pred["__ts__"] = pd.to_datetime(pred["__ts__"], utc=True, errors="coerce")
    pred["month"] = pred["__ts__"].dt.to_period("M").astype(str)
    pred["week_start"] = pred["__ts__"].dt.to_period("W-MON").apply(lambda p: p.start_time.date().isoformat())
    pred["__symbol__"] = pred["__symbol__"].astype(str)
    pred["side_name"] = pred["side_name"].astype(str)
    pred[ARCH_COL] = pred.get(ARCH_COL, pred.get("__archetype_policy_key__", "missing")).astype(str)
    pred[ARCH_COL] = pred[ARCH_COL].replace({"nan": "missing", "None": "missing"})
    return pred


def _derive_prediction_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    if SCORE_COL in out.columns and BASE_SCORE_COL in out.columns:
        out["__derived_score_dispersion__"] = (
            _safe_numeric(out[SCORE_COL]) - _safe_numeric(out[BASE_SCORE_COL])
        ).abs().astype("float32")
    if SCORE_COL in out.columns:
        p = _safe_numeric(out[SCORE_COL]).clip(1e-6, 1.0 - 1e-6)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)
        out["__derived_meta_uncertainty__"] = entropy.astype("float32")
    return out


def _available_feature_cols(handoff_cols: set[str], pred_cols: set[str]) -> dict[str, str | None]:
    available = handoff_cols | pred_cols | {"__derived_score_dispersion__", "__derived_meta_uncertainty__", "__derived_gmm_entropy__"}
    chosen: dict[str, str | None] = {}
    for spec in REGIME_SPECS:
        chosen[spec.name] = next((col for col in spec.candidates if col in available), None)
    return chosen


def _load_feature_slice(handoff: Path, pred: pd.DataFrame, candidate_cols: Iterable[str]) -> pd.DataFrame:
    cols = set(_schema_cols(handoff))
    read_cols = [col for col in [*KEYS, *sorted(set(candidate_cols))] if col in cols]
    features = pd.read_parquet(handoff, columns=read_cols)
    features["__ts__"] = pd.to_datetime(features["__ts__"], utc=True, errors="coerce")
    features["month"] = features["__ts__"].dt.to_period("M").astype(str)
    features = features.loc[features["month"].isin(sorted(pred["month"].unique()))].copy()
    features["__symbol__"] = features["__symbol__"].astype(str)
    features["side_name"] = features["side_name"].astype(str)
    return features.drop(columns=["month"]).drop_duplicates(KEYS)


def _derive_joined_features(frame: pd.DataFrame) -> pd.DataFrame:
    posterior_cols = [col for col in frame.columns if col.startswith("gmm_cluster_posterior_")]
    if posterior_cols:
        probs = frame[posterior_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0.0, 1.0)
        denom = probs.sum(axis=1).replace(0.0, np.nan)
        probs = probs.div(denom, axis=0).fillna(0.0)
        entropy = -(probs.where(probs.gt(0.0), 1.0).apply(np.log) * probs).sum(axis=1)
        max_entropy = math.log(max(2, len(posterior_cols)))
        frame["__derived_gmm_entropy__"] = (entropy / max_entropy).astype("float32")
    elif "gmm_posterior_max" in frame.columns:
        pmax = _safe_numeric(frame["gmm_posterior_max"]).clip(1e-6, 1.0 - 1e-6)
        frame["__derived_gmm_entropy__"] = (1.0 - pmax).astype("float32")
    return frame


def _rank_pct_by_month(score: pd.Series, month: pd.Series) -> pd.Series:
    return score.groupby(month).rank(pct=True, method="first")


def _regime_value(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float32")
    return _safe_numeric(frame[col]).astype("float32")


def _risk_target(frame: pd.DataFrame) -> pd.Series:
    ev = _safe_numeric(frame["ev_after_1pct"]).fillna(0.0)
    bad = _safe_numeric(frame["full_path_bad_mae_1r"]).fillna(0.0)
    timeout = _safe_numeric(frame["timeout"]).fillna(0.0)
    dirty = _safe_numeric(frame["dirty_positive"]).fillna(0.0)
    clean = _safe_numeric(frame["clean_exec"]).fillna(0.0)
    # Units are score-like: positive means "subtract from the meta score".
    y = -ev + 0.006 * bad + 0.006 * timeout + 0.004 * dirty - 0.004 * clean
    return y.clip(-0.08, 0.08).astype("float32")


def _standardize_fit(x: pd.Series) -> tuple[pd.Series, float, float]:
    vals = _safe_numeric(x)
    med = float(vals.median()) if vals.notna().any() else 0.0
    q25, q75 = vals.quantile([0.25, 0.75]).to_numpy(dtype=float) if vals.notna().sum() >= 4 else (np.nan, np.nan)
    scale = float(q75 - q25) if np.isfinite(q75 - q25) and q75 > q25 else float(vals.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return ((vals - med) / scale).clip(-6.0, 6.0), med, scale


def _standardize_apply(x: pd.Series, med: float, scale: float) -> pd.Series:
    return ((_safe_numeric(x) - med) / max(scale, 1e-9)).clip(-6.0, 6.0)


def _fit_shape(x: pd.Series, y: pd.Series, shape: str) -> ShapeFit:
    valid = x.notna() & y.notna()
    xv = x.loc[valid]
    yv = y.loc[valid]
    if shape == "flat" or len(yv) < 30 or xv.nunique(dropna=True) < 5:
        return ShapeFit("flat", {"mean": 0.0}, int(valid.sum()))
    z, med, scale = _standardize_fit(xv)
    yc = yv - float(yv.mean())
    if shape == "linear":
        design = np.column_stack([z.to_numpy(dtype=float)])
    elif shape == "quadratic":
        zz = z.to_numpy(dtype=float)
        design = np.column_stack([zz, zz * zz])
    elif shape == "ushape":
        design = np.column_stack([np.abs(z.to_numpy(dtype=float))])
    elif shape == "bucketed":
        qs = xv.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy(dtype=float)
        if len(np.unique(qs[np.isfinite(qs)])) < 3:
            return ShapeFit("flat", {"mean": 0.0}, int(valid.sum()))
        bins = np.digitize(xv.to_numpy(dtype=float), qs, right=True)
        effects = pd.Series(yc.to_numpy(dtype=float)).groupby(bins).mean().to_dict()
        counts = pd.Series(yc.to_numpy(dtype=float)).groupby(bins).size().to_dict()
        effects = {int(k): float(np.clip(v, -0.06, 0.06)) for k, v in effects.items() if counts.get(k, 0) >= 10}
        return ShapeFit("bucketed", {"quantiles": qs.tolist(), "effects": effects}, int(valid.sum()))
    elif shape == "monotone":
        if IsotonicRegression is None:
            return ShapeFit("flat", {"mean": 0.0}, int(valid.sum()))
        corr = float(pd.Series(z).corr(pd.Series(yc), method="spearman"))
        increasing = bool(corr >= 0.0) if np.isfinite(corr) else True
        order = np.argsort(z.to_numpy(dtype=float))
        model = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        model.fit(z.to_numpy(dtype=float)[order], yc.to_numpy(dtype=float)[order])
        return ShapeFit(
            "monotone",
            {
                "median": med,
                "scale": scale,
                "increasing": increasing,
                "x_thresholds": model.X_thresholds_.tolist(),
                "y_thresholds": np.clip(model.y_thresholds_, -0.06, 0.06).tolist(),
            },
            int(valid.sum()),
        )
    else:
        raise ValueError(f"unknown shape: {shape}")
    try:
        coef, *_ = np.linalg.lstsq(design, yc.to_numpy(dtype=float), rcond=None)
    except np.linalg.LinAlgError:
        coef = np.zeros(design.shape[1], dtype=float)
    return ShapeFit(
        shape,
        {"median": med, "scale": scale, "coef": np.clip(coef, -0.04, 0.04).tolist()},
        int(valid.sum()),
    )


def _apply_shape(x: pd.Series, fit: ShapeFit) -> pd.Series:
    if fit.shape == "flat":
        return pd.Series(0.0, index=x.index, dtype="float32")
    params = fit.params
    if fit.shape == "bucketed":
        qs = np.asarray(params.get("quantiles", []), dtype=float)
        effects = {int(k): float(v) for k, v in params.get("effects", {}).items()}
        raw = _safe_numeric(x).to_numpy(dtype=float)
        bins = np.digitize(raw, qs, right=True)
        out = np.array([effects.get(int(b), 0.0) for b in bins], dtype="float32")
        out[~np.isfinite(raw)] = 0.0
        return pd.Series(out, index=x.index)
    z = _standardize_apply(x, float(params.get("median", 0.0)), float(params.get("scale", 1.0))).fillna(0.0)
    if fit.shape == "monotone":
        xs = np.asarray(params["x_thresholds"], dtype=float)
        ys = np.asarray(params["y_thresholds"], dtype=float)
        out = np.interp(z.to_numpy(dtype=float), xs, ys, left=ys[0], right=ys[-1])
    elif fit.shape == "linear":
        out = z.to_numpy(dtype=float) * float(params.get("coef", [0.0])[0])
    elif fit.shape == "quadratic":
        coef = params.get("coef", [0.0, 0.0])
        zz = z.to_numpy(dtype=float)
        out = zz * float(coef[0]) + zz * zz * float(coef[1])
    elif fit.shape == "ushape":
        out = np.abs(z.to_numpy(dtype=float)) * float(params.get("coef", [0.0])[0])
    else:
        out = np.zeros(len(x), dtype=float)
    return pd.Series(np.clip(out, -0.06, 0.06).astype("float32"), index=x.index)


def _eval_score(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    score = _safe_numeric(frame[score_col])
    for scope, cut in TOP_SCOPES.items():
        sub = frame.loc[score.rank(pct=True, method="first").ge(cut)]
        out[f"{scope}_rows"] = float(len(sub))
        out[f"{scope}_mean_ev"] = float(_safe_numeric(sub["ev_after_1pct"]).mean()) if len(sub) else np.nan
        out[f"{scope}_clean"] = float(_safe_numeric(sub["clean_exec"]).mean()) if len(sub) else np.nan
        out[f"{scope}_bad_mae"] = float(_safe_numeric(sub["full_path_bad_mae_1r"]).mean()) if len(sub) else np.nan
        out[f"{scope}_timeout"] = float(_safe_numeric(sub["timeout"]).mean()) if len(sub) else np.nan
    return out


def _objective_delta(base_metrics: dict[str, float], adj_metrics: dict[str, float]) -> float:
    weights = {"top10": 0.45, "top20": 0.35, "top30": 0.20}
    total = 0.0
    for scope, weight in weights.items():
        ev_delta = adj_metrics.get(f"{scope}_mean_ev", np.nan) - base_metrics.get(f"{scope}_mean_ev", np.nan)
        clean_delta = adj_metrics.get(f"{scope}_clean", np.nan) - base_metrics.get(f"{scope}_clean", np.nan)
        bad_delta = base_metrics.get(f"{scope}_bad_mae", np.nan) - adj_metrics.get(f"{scope}_bad_mae", np.nan)
        if np.isfinite(ev_delta):
            total += weight * ev_delta
        if np.isfinite(clean_delta):
            total += 0.0025 * weight * clean_delta
        if np.isfinite(bad_delta):
            total += 0.0025 * weight * bad_delta
    return float(total)


def _shape_complexity(shape: str) -> int:
    return SHAPE_ORDER.index(shape)


def _select_simplest(shape_rows: list[dict[str, Any]]) -> str:
    viable = [r for r in shape_rows if r["shape"] != "flat" and r["validation_delta_objective"] > 0.00015]
    if not viable:
        return "flat"
    best = max(viable, key=lambda r: r["validation_delta_objective"])
    close = [
        r for r in viable
        if r["validation_delta_objective"] >= best["validation_delta_objective"] - 0.00005
    ]
    # Promotion rule: prefer the simpler shape if it is close to best.
    return min(close, key=lambda r: _shape_complexity(str(r["shape"])))["shape"]


def _top_groups(frame: pd.DataFrame) -> list[tuple[tuple[str, str], pd.DataFrame]]:
    return list(frame.groupby(["side_name", ARCH_COL], dropna=False, observed=True))


def _split_fit_select(train: pd.DataFrame, eval_month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(train["month"].dropna().unique().tolist())
    if len(months) >= 2:
        select_month = months[-1]
        return train.loc[train["month"].lt(select_month)], train.loc[train["month"].eq(select_month)]
    ordered = train.sort_values("__ts__")
    if ordered.empty:
        return ordered, ordered
    cutoff = ordered["__ts__"].quantile(0.5)
    return ordered.loc[ordered["__ts__"].le(cutoff)], ordered.loc[ordered["__ts__"].gt(cutoff)]


def _corr_prune(selected: pd.DataFrame, train_group: pd.DataFrame, feature_map: dict[str, str]) -> set[str]:
    keep: set[str] = set()
    ordered = selected.sort_values(["validation_delta_objective", "oos_delta_objective"], ascending=False)
    for _, row in ordered.iterrows():
        regime = str(row["regime"])
        col = feature_map.get(regime)
        if not col:
            continue
        redundant = False
        x = _regime_value(train_group, col)
        for kept in keep:
            kcol = feature_map.get(kept)
            if not kcol:
                continue
            y = _regime_value(train_group, kcol)
            corr = float(x.corr(y, method="spearman"))
            if np.isfinite(corr) and abs(corr) >= 0.80:
                redundant = True
                break
        if not redundant:
            keep.add(regime)
    return keep


def _shape_comparison_for_group(
    fit_part: pd.DataFrame,
    select_part: pd.DataFrame,
    eval_part: pd.DataFrame,
    side: str,
    archetype: str,
    eval_month: str,
    feature_map: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], ShapeFit]]:
    rows: list[dict[str, Any]] = []
    fits: dict[tuple[str, str], ShapeFit] = {}
    if len(fit_part) < 80 or len(select_part) < 40 or len(eval_part) < 40:
        return rows, fits
    y_fit = _risk_target(fit_part)
    base_select = _eval_score(select_part, SCORE_COL)
    base_eval = _eval_score(eval_part, SCORE_COL)
    for regime, col in feature_map.items():
        if not col or col not in fit_part.columns or col not in eval_part.columns:
            continue
        x_fit = _regime_value(fit_part, col)
        x_select = _regime_value(select_part, col)
        x_eval = _regime_value(eval_part, col)
        coverage_fit = float(x_fit.notna().mean())
        coverage_eval = float(x_eval.notna().mean())
        unique_fit = int(x_fit.nunique(dropna=True))
        if coverage_fit < 0.40 or coverage_eval < 0.40 or unique_fit < 5:
            rows.append(
                {
                    "eval_month": eval_month,
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "regime": regime,
                    "feature_col": col,
                    "shape": "discarded_low_coverage",
                    "fit_rows": len(fit_part),
                    "select_rows": len(select_part),
                    "eval_rows": len(eval_part),
                    "coverage_fit": coverage_fit,
                    "coverage_oos": coverage_eval,
                    "n_unique_fit": unique_fit,
                    "validation_delta_objective": np.nan,
                    "oos_delta_objective": np.nan,
                }
            )
            continue
        for shape in SHAPE_ORDER:
            fit = _fit_shape(x_fit, y_fit, shape)
            select_adj = select_part.assign(__adj_score__=(
                _safe_numeric(select_part[SCORE_COL]) - _apply_shape(x_select, fit)
            ).clip(0.0, 1.0))
            eval_adj = eval_part.assign(__adj_score__=(
                _safe_numeric(eval_part[SCORE_COL]) - _apply_shape(x_eval, fit)
            ).clip(0.0, 1.0))
            select_metrics = _eval_score(select_adj, "__adj_score__")
            eval_metrics = _eval_score(eval_adj, "__adj_score__")
            val_delta = _objective_delta(base_select, select_metrics)
            oos_delta = _objective_delta(base_eval, eval_metrics)
            rows.append(
                {
                    "eval_month": eval_month,
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "regime": regime,
                    "feature_col": col,
                    "shape": shape,
                    "fit_rows": len(fit_part),
                    "select_rows": len(select_part),
                    "eval_rows": len(eval_part),
                    "coverage_fit": coverage_fit,
                    "coverage_oos": coverage_eval,
                    "n_unique_fit": unique_fit,
                    "validation_delta_objective": val_delta,
                    "oos_delta_objective": oos_delta,
                    **{f"select_{k}": v for k, v in select_metrics.items()},
                    **{f"oos_{k}": v for k, v in eval_metrics.items()},
                }
            )
            fits[(regime, shape)] = fit
    return rows, fits


def _monthly_adjustments(
    frame: pd.DataFrame,
    eval_months: list[str],
    feature_map: dict[str, str],
    risk_cap: float,
    risk_cap_negative: float | None,
    risk_cap_positive: float | None,
    require_oos_positive: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cap_neg = max(float(risk_cap if risk_cap_negative is None else risk_cap_negative), 0.0)
    cap_pos = max(float(risk_cap if risk_cap_positive is None else risk_cap_positive), 0.0)
    shape_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    adjusted_parts: list[pd.DataFrame] = []
    for eval_month in eval_months:
        train = frame.loc[frame["month"].lt(eval_month)].copy()
        eval_frame = frame.loc[frame["month"].eq(eval_month)].copy()
        fit_part_all, select_part_all = _split_fit_select(train, eval_month)
        if fit_part_all.empty or select_part_all.empty or eval_frame.empty:
            continue
        eval_frame["risk_score"] = 0.0
        eval_frame["risk_effect_count"] = 0
        for (side, arch), eval_group in _top_groups(eval_frame):
            fit_group = fit_part_all.loc[(fit_part_all["side_name"].eq(side)) & (fit_part_all[ARCH_COL].eq(arch))]
            select_group = select_part_all.loc[(select_part_all["side_name"].eq(side)) & (select_part_all[ARCH_COL].eq(arch))]
            rows, fits = _shape_comparison_for_group(
                fit_group, select_group, eval_group, side, arch, eval_month, feature_map
            )
            shape_rows.extend(rows)
            if not rows:
                continue
            by_regime: list[dict[str, Any]] = []
            for regime in feature_map:
                regime_rows = [r for r in rows if r.get("regime") == regime and r.get("shape") in SHAPE_ORDER]
                if not regime_rows:
                    continue
                chosen_shape = _select_simplest(regime_rows)
                chosen = next((r for r in regime_rows if r["shape"] == chosen_shape), None)
                if chosen is None:
                    continue
                if chosen_shape != "flat":
                    by_regime.append(chosen)
            chosen_df = pd.DataFrame(by_regime)
            keep_regimes = _corr_prune(chosen_df, fit_group, feature_map) if not chosen_df.empty else set()
            group_risk = pd.Series(0.0, index=eval_group.index, dtype="float32")
            effect_count = pd.Series(0, index=eval_group.index, dtype="int16")
            for _, chosen in chosen_df.iterrows():
                regime = str(chosen["regime"])
                shape = str(chosen["shape"])
                if regime not in keep_regimes:
                    redundancy_status = "discarded_redundant"
                elif require_oos_positive and float(chosen.get("oos_delta_objective", np.nan)) <= 0.0:
                    redundancy_status = "discarded_oos_negative"
                else:
                    redundancy_status = "promoted_oos_confirmed" if require_oos_positive else "promoted_validation_selected"
                    fit = fits.get((regime, shape))
                    col = feature_map.get(regime)
                    if fit is not None and col:
                        eff = _apply_shape(_regime_value(eval_group, col), fit)
                        group_risk = group_risk.add(eff, fill_value=0.0)
                        effect_count = effect_count.add(eff.ne(0.0).astype("int16"), fill_value=0).astype("int16")
                row = dict(chosen)
                row["selection_status"] = redundancy_status
                fit = fits.get((regime, shape))
                row["effect_params_json"] = json.dumps(fit.params if fit is not None else {}, sort_keys=True)
                selected_rows.append(row)
            eval_frame.loc[eval_group.index, "risk_score"] = group_risk.clip(-cap_neg, cap_pos)
            eval_frame.loc[eval_group.index, "risk_effect_count"] = effect_count
        eval_frame["score_regime_calibrated"] = (
            _safe_numeric(eval_frame[SCORE_COL]) - _safe_numeric(eval_frame["risk_score"]).fillna(0.0)
        ).clip(0.0, 1.0).astype("float32")
        adjusted_parts.append(eval_frame)
    return (
        pd.DataFrame(shape_rows),
        pd.DataFrame(selected_rows),
        pd.concat(adjusted_parts, ignore_index=True, copy=False) if adjusted_parts else pd.DataFrame(),
    )


def _metric_row(group: pd.DataFrame, score_col: str, keys: dict[str, Any]) -> dict[str, Any]:
    score = _safe_numeric(group[score_col])
    ev = _safe_numeric(group["ev_after_1pct"])
    clean = _safe_numeric(group["clean_exec"])
    bad = _safe_numeric(group["full_path_bad_mae_1r"])
    timeout = _safe_numeric(group["timeout"])
    dirty = _safe_numeric(group["dirty_positive"])
    return {
        **keys,
        "rows": int(len(group)),
        "mean_score": float(score.mean()),
        "mean_ev_after_1pct": float(ev.mean()),
        "sum_ev_after_1pct": float(ev.sum()),
        "positive_ev_rate": float(ev.gt(0.0).mean()),
        "clean_exec_rate": float(clean.mean()),
        "dirty_positive_rate": float(dirty.mean()),
        "full_path_bad_mae_rate": float(bad.mean()),
        "timeout_rate": float(timeout.mean()),
        "avg_risk_score": float(_safe_numeric(group.get("risk_score", pd.Series(0, index=group.index))).mean()),
        "avg_risk_effect_count": float(_safe_numeric(group.get("risk_effect_count", pd.Series(0, index=group.index))).mean()),
    }


def _summary(frame: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rank = _safe_numeric(frame[score_col]).groupby(frame["month"]).rank(pct=True, method="first")
    for scope, cut in TOP_SCOPES.items():
        sub = frame.loc[rank.ge(cut)]
        rows.append(_metric_row(sub, score_col, {"score_variant": label, "top_scope": scope, "group": "overall", "group_value": "all"}))
        for col in ["month", "week_start", "side_name", ARCH_COL]:
            for key, group in sub.groupby(col, dropna=False, observed=True):
                rows.append(_metric_row(group, score_col, {"score_variant": label, "top_scope": scope, "group": col, "group_value": key}))
        for key, group in sub.groupby(["month", "side_name", ARCH_COL], dropna=False, observed=True):
            rows.append(
                _metric_row(
                    group,
                    score_col,
                    {
                        "score_variant": label,
                        "top_scope": scope,
                        "group": "month_x_side_x_archetype",
                        "group_value": "|".join(map(str, key)),
                    },
                )
            )
    return pd.DataFrame(rows)


def _regime_bin_table(frame: pd.DataFrame, feature_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rank = _safe_numeric(frame["score_regime_calibrated"]).groupby(frame["month"]).rank(pct=True, method="first")
    top = frame.loc[rank.ge(0.90)]
    for regime, col in feature_map.items():
        if not col or col not in top.columns:
            continue
        vals = _regime_value(top, col)
        valid = vals.notna()
        if valid.sum() < 30 or vals[valid].nunique() < 3:
            continue
        q1, q2 = vals[valid].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
        if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
            continue
        bins = pd.Series("mid", index=top.index, dtype=object)
        bins.loc[vals.le(q1)] = "low"
        bins.loc[vals.gt(q2)] = "high"
        bins.loc[~valid] = "missing"
        work = top.assign(__regime_bin__=bins)
        for key, group in work.groupby(["month", "side_name", ARCH_COL, "__regime_bin__"], dropna=False, observed=True):
            if len(group) < 10:
                continue
            rows.append(
                _metric_row(
                    group,
                    "score_regime_calibrated",
                    {
                        "regime": regime,
                        "feature_col": col,
                        "month": key[0],
                        "side_name": key[1],
                        "archetype_policy_key": key[2],
                        "regime_bin": key[3],
                    },
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-run", type=Path, default=DEFAULT_META_RUN)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--all-months", nargs="+", default=["2026-04", "2026-05", "2026-06"])
    parser.add_argument("--eval-months", nargs="+", default=["2026-05", "2026-06"])
    parser.add_argument("--risk-cap", type=float, default=0.06)
    parser.add_argument(
        "--risk-cap-negative",
        type=float,
        default=None,
        help=(
            "Absolute cap for negative risk_score boosts. Defaults to --risk-cap."
        ),
    )
    parser.add_argument(
        "--risk-cap-positive",
        type=float,
        default=None,
        help=(
            "Cap for positive risk_score penalties. Defaults to --risk-cap."
        ),
    )
    parser.add_argument(
        "--allow-validation-only-effects",
        action="store_true",
        help="Apply effects selected on prior validation even when their measured OOS delta is negative. "
        "The default is the diagnostic requested here: keep only OOS-confirmed effects.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred = _derive_prediction_features(_load_predictions(args.meta_run, args.all_months))
    handoff_cols = set(_schema_cols(args.handoff))
    pred_cols = set(pred.columns)
    chosen = _available_feature_cols(handoff_cols, pred_cols)
    derived_source_cols = {
        col for col in handoff_cols
        if col.startswith("gmm_cluster_posterior_") or col in {"gmm_posterior_max"}
    }
    candidate_cols = sorted(
        {
            col for col in chosen.values()
            if col and not col.startswith("__derived_") and col not in pred_cols
        }
        | derived_source_cols
    )
    features = _load_feature_slice(args.handoff, pred, candidate_cols) if candidate_cols else pd.DataFrame(columns=KEYS)
    merged = pred.merge(features, on=KEYS, how="left", validate="many_to_one") if len(features) else pred
    merged = _derive_joined_features(merged)
    feature_map = {name: col for name, col in chosen.items() if col and col in merged.columns}
    if not feature_map:
        raise RuntimeError("No usable regime features were available")
    for col in OUTCOME_COLS + [SCORE_COL]:
        if col not in merged.columns:
            raise RuntimeError(f"Required column missing from prediction shards: {col}")
    shape_cmp, selected, adjusted = _monthly_adjustments(
        merged,
        args.eval_months,
        feature_map,
        args.risk_cap,
        args.risk_cap_negative,
        args.risk_cap_positive,
        require_oos_positive=not args.allow_validation_only_effects,
    )
    if adjusted.empty:
        raise RuntimeError("No adjusted OOS rows were produced")

    shape_cmp.to_csv(args.output_dir / "regime_shape_comparison_oos.csv", index=False)
    selected.to_csv(args.output_dir / "regime_selected_effects.csv", index=False)
    promoted_effects: list[dict[str, Any]] = []
    if not selected.empty:
        for _, row in selected.loc[
            selected["selection_status"].astype(str).str.startswith("promoted")
        ].iterrows():
            try:
                params = json.loads(str(row.get("effect_params_json") or "{}"))
            except Exception:
                params = {}
            promoted_effects.append(
                {
                    "eval_month": str(row.get("eval_month") or ""),
                    "side_name": str(row.get("side_name") or ""),
                    "archetype_policy_key": str(row.get("archetype_policy_key") or ""),
                    "regime": str(row.get("regime") or ""),
                    "feature_col": str(row.get("feature_col") or ""),
                    "shape": str(row.get("shape") or "flat"),
                    "params": params,
                    "validation_delta_objective": float(row.get("validation_delta_objective", np.nan)),
                    "oos_delta_objective": float(row.get("oos_delta_objective", np.nan)),
                    "fit_rows": int(row.get("fit_rows", 0) or 0),
                    "select_rows": int(row.get("select_rows", 0) or 0),
                    "eval_rows": int(row.get("eval_rows", 0) or 0),
                }
            )
    adjusted_cols = [
        "__ts__", "__symbol__", "side_name", "month", "week_start", ARCH_COL,
        SCORE_COL, "score_regime_calibrated", "risk_score", "risk_effect_count",
        *OUTCOME_COLS,
    ]
    adjusted[[c for c in adjusted_cols if c in adjusted.columns]].to_parquet(
        args.output_dir / "meta_oos_regime_calibrated_predictions.parquet", index=False
    )
    summary = pd.concat(
        [
            _summary(adjusted, SCORE_COL, "baseline_meta"),
            _summary(adjusted, "score_regime_calibrated", "regime_calibrated_meta"),
        ],
        ignore_index=True,
        copy=False,
    )
    summary.to_csv(args.output_dir / "score_variant_metrics.csv", index=False)
    regime_bins = _regime_bin_table(adjusted, feature_map)
    regime_bins.to_csv(args.output_dir / "top10_archetype_x_regime_bins_after_calibration.csv", index=False)

    availability = []
    spec_by_name = {spec.name: spec for spec in REGIME_SPECS}
    for name, col in chosen.items():
        spec = spec_by_name[name]
        availability.append(
            {
                "regime": name,
                "semantic": spec.semantic,
                "chosen_feature_col": col,
                "available_and_used": bool(name in feature_map),
                "proxy_note": spec.proxy_note,
                "candidate_cols": ",".join(spec.candidates),
                "non_null_share": float(_regime_value(merged, col).notna().mean()) if col and col in merged.columns else 0.0,
                "n_unique": int(_regime_value(merged, col).nunique(dropna=True)) if col and col in merged.columns else 0,
            }
        )
    availability_df = pd.DataFrame(availability)
    availability_df.to_csv(args.output_dir / "regime_feature_mapping.csv", index=False)
    manifest = {
        "meta_run": str(args.meta_run),
        "handoff": str(args.handoff),
        "all_months_loaded": args.all_months,
        "eval_months": args.eval_months,
        "method": "May uses April prior OOS calibration; June uses April+May with last prior month as validation. Shapes are fitted on earlier rows and applied to eval rows.",
        "score_col": SCORE_COL,
        "adjusted_score_col": "score_regime_calibrated",
        "risk_score_formula": (
            "sum(selected_regime_effects), clipped to "
            "[-risk_cap_negative, risk_cap_positive], "
            "final_score = meta_score - risk_score"
        ),
        "risk_cap": args.risk_cap,
        "risk_cap_negative": (
            args.risk_cap if args.risk_cap_negative is None else args.risk_cap_negative
        ),
        "risk_cap_positive": (
            args.risk_cap if args.risk_cap_positive is None else args.risk_cap_positive
        ),
        "selection_mode": "oos_confirmed" if not args.allow_validation_only_effects else "validation_only",
        "promotion_rule": "non-flat shapes require positive validation objective; redundant regimes abs Spearman >= 0.80 are pruned per side x archetype; simpler shape wins when close.",
        "outputs": {
            "feature_mapping": str(args.output_dir / "regime_feature_mapping.csv"),
            "shape_comparison": str(args.output_dir / "regime_shape_comparison_oos.csv"),
            "selected_effects": str(args.output_dir / "regime_selected_effects.csv"),
            "adjusted_predictions": str(args.output_dir / "meta_oos_regime_calibrated_predictions.parquet"),
            "score_metrics": str(args.output_dir / "score_variant_metrics.csv"),
            "regime_bins": str(args.output_dir / "top10_archetype_x_regime_bins_after_calibration.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    calibration_artifact = {
        "artifact_id": CALIBRATION_POLICY_ID,
        "policy_id": CALIBRATION_POLICY_ID,
        "source": "report_meta_oos_regime_calibration.py",
        "source_score_col": SCORE_COL,
        "adjusted_score_col": "score_regime_calibrated",
        "risk_score_col": "regime_ev_risk_score",
        "effect_count_col": "regime_ev_effect_count",
        "risk_cap": float(args.risk_cap),
        "risk_cap_negative": float(
            args.risk_cap if args.risk_cap_negative is None else args.risk_cap_negative
        ),
        "risk_cap_positive": float(
            args.risk_cap if args.risk_cap_positive is None else args.risk_cap_positive
        ),
        "selection_mode": "oos_confirmed" if not args.allow_validation_only_effects else "validation_only",
        "feature_mapping": availability,
        "effects": promoted_effects,
        "notes": (
            "Effects are side x archetype local EV calibration terms. "
            "Apply as final_score = source_score - "
            "clip(sum(effect_i), -risk_cap_negative, risk_cap_positive)."
        ),
    }
    (args.output_dir / "regime_ev_calibration.json").write_text(
        json.dumps(calibration_artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("FEATURE MAPPING")
    print(availability_df[["regime", "chosen_feature_col", "available_and_used", "non_null_share", "n_unique", "proxy_note"]].to_string(index=False))
    print("\nSELECTED EFFECTS")
    if selected.empty:
        print("No non-flat effects selected.")
    else:
        show = selected.loc[selected["selection_status"].astype(str).str.startswith("promoted")].copy()
        cols = ["eval_month", "side_name", "archetype_policy_key", "regime", "feature_col", "shape", "validation_delta_objective", "oos_delta_objective", "eval_rows"]
        print(show.sort_values(["eval_month", "oos_delta_objective"], ascending=[True, False])[cols].head(40).to_string(index=False))
    print("\nSCORE VARIANT OVERALL")
    overall = summary.loc[summary["group"].eq("overall")].copy()
    cols = ["score_variant", "top_scope", "rows", "mean_ev_after_1pct", "clean_exec_rate", "dirty_positive_rate", "full_path_bad_mae_rate", "timeout_rate", "avg_risk_score"]
    print(overall[cols].to_string(index=False))
    print(f"\n[done] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
