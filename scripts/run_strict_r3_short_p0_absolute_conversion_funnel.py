#!/usr/bin/env python3
"""Strict-OOS absolute conversion funnel for the short P0/F90 base.

The short base is a useful *within-hour* ranker.  This runner deliberately
does not copy the long ten-head residual architecture: it selects the frozen
P0 winner from every target-free decision hour, then asks whether that hour
has an absolute, policy-aligned opportunity.

For every held month, all model fitting, OOF calibration and recent-outcome
features are restricted to labels resolved before that month.  Candidate
selection and the score-geometry features never inspect outcomes.  The
canonical long stack is neither imported nor modified.

Arms (specified before execution)
-------------------------------
M0  causal P0 anchor only
M1  direct Huber policy net
M2  P(policy net > 0), converted to bps with OOF isotonic calibration
M3  P(policy net > 100), converted to bps with OOF isotonic calibration
M4  ordinal policy margin, converted to bps with OOF isotonic calibration
M5  P0 anchor plus Huber residual
M6  M5 + P0 cross-sectional score geometry
M7  M6 + causal hour-level market state
M8  M7 + causal recent conversion state

The output keeps both causal threshold admissions and held-period percentile
diagnostics.  The latter are explicitly diagnostic and must never be treated
as a live threshold.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SIDE = "short"
POLICY_CLIP_BPS = 500.0
MIN_TRAIN_ROWS = 500
MIN_OOF_ROWS = 240
OOF_SPLITS = 3
HISTORY_DAYS = (7, 14, 28, 56)
ORDINAL_EDGES = (-300.0, -100.0, 50.0, 150.0, 300.0)
ORDINAL_VALUES = np.asarray((-400.0, -200.0, -25.0, 100.0, 225.0, 400.0))
ADMISSION_LEVELS = (0.0, 50.0, 100.0, 150.0)
CAUSAL_TRAIN_QUANTILES = (0.70, 0.80)


# A compact winner-specific context block.  These are all pre-existing
# P0/F90 fields; no label or post-decision value is included.
WINNER_CONTEXT = (
    "leverage_build",
    "distance_to_resistance_daily_vwap_atr",
    "loc_session_pos_24",
    "asset_minus_mkt_price_recovery_fraction_24h",
    "asset_minus_mkt_short_cover_intensity_1h",
    "exh_qual_surprise",
    "grind_score_surprise",
    "asset_minus_mkt_oi_recovery_fraction_24h",
    "oi_recovery_fraction_24h",
    "price_recovery_from_low_24h_atr",
    "efficiency_ratio_20",
    "volume_trend_48",
    "ob_trade_size_to_l1_depth_z_24h",
    "xasset_mkt_spread_bps_z_24h",
    "bars_since_price_low_24h_norm",
    "oi_recovery_fraction_72h",
    "price_minus_oi_recovery_24h",
    "price_minus_oi_recovery_72h",
    "up_down_semivol_ratio_tanh",
    "bars_since_price_low_72h_norm",
    "distance_to_support_atr",
    "oiw_z_delta_entry_dist_1d_atr",
    "oi_drawdown_from_peak_168h",
    "price_recovery_fraction_72h",
    "price_up_oi_down_4h_rz",
    "excess_6h_ts_resid",
    "price_recovery_from_low_72h_atr",
    "loc_swing_range_pos_24",
    "dist_oiw_z_delta_12h_atr",
    "price_recovery_fraction_24h",
    "loc_range_pos_24",
    "distance_to_support_daily_donchian_atr",
    "loc_range_pos_48",
    "asset_minus_mkt_oi_1d_cp_z_8_32_96",
    "dist_oiw_z_delta_96h_atr",
    "oi_drawdown_from_peak_72h",
    "donchian_zone_1d_atr",
    "loc_prev_day_range_pos_24",
)

# Existing global / cross-sectional state fields available in the F90 ledger.
MARKET_STATE = (
    "mkt_ret_24h",
    "mkt_ret_eq_4h",
    "pct_assets_price_down_oi_up_1h",
    "pct_assets_above_intraday_vwap",
    "mkt_pct_price_down_oi_up_1h",
    "pct_assets_recovering_from_intraday_low",
    "mkt_median_oi_drawdown_from_peak_24h",
    "mkt_recovery_from_24h_low_atr",
    "mkt_oi_breadth_rising_24h",
    "mkt_oi_chg_4h",
    "pct_assets_up_24h",
    "breadth_dispersion",
    "xasset_mkt_ob_stress_z_24h",
    "state_spectral_eig_top3_share",
    "state_spectral_eig_condition",
    "state_spectral_eig_gap_1_2",
    "eig_effective_rank__open_interest",
    "xs_dispersion__vol_z",
    "xs_dispersion__amihud_illiq",
    "xs_dispersion__funding_per_hour",
    "xs_dispersion__oi_value_1d_chg_z_90d",
    "xs_dispersion__oi_to_volume_7d_z_180d",
    "xs_dispersion__ob_depth_to_qv_z_x_rvol_z",
    "xs_dispersion__xasset_ob_liquidity_ts_resid",
    "q_tail_width__volatility_zscore",
    "q_tail_width__volume_z_12",
    "q_tail_width__oi_to_volume_7d_z_180d",
)

BASE_INPUT = (
    "prequential_base_score",
    "prequential_base_rank42",
    "prequential_base_anchor_bps",
    *WINNER_CONTEXT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _valid_policy(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & _finite(frame["p0_canonical_net_bps"]).notna()
        & frame["policy_label_available_at"].notna()
    )


def _load_ledger(roots: list[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.glob("ledger/month=*/prequential_base_ledger.parquet")))
    if not files:
        raise FileNotFoundError("no strict short P0 ledger partitions found")
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    required = {
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "policy_path_valid", "policy_label_available_at", "p0_canonical_net_bps",
        "prequential_base_score", "prequential_base_rank42", "prequential_base_anchor_bps",
        "base_feature_eligible", *WINNER_CONTEXT, *MARKET_STATE,
    }
    for path in files:
        available = set(pd.read_parquet(path, columns=None).columns)
        missing = sorted(required.difference(available))
        if missing:
            raise ValueError(f"ledger partition {path} misses required short-only columns: {missing}")
        frame = pd.read_parquet(path, columns=sorted(required))
        hashes[str(path)] = _sha256(path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.candidate_id.duplicated().any():
        duplicates = result.loc[result.candidate_id.duplicated(keep=False), "candidate_id"].head(5).tolist()
        raise ValueError(f"short ledger candidate identity overlap: {duplicates}")
    for column in ("__ts__", "__decision_ts__", "policy_label_available_at"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    observed = result.side_name.astype(str).str.lower().str.strip()
    if not observed.eq(SIDE).all():
        raise ValueError("absolute conversion funnel received a non-short ledger row")
    if not result["base_feature_eligible"].fillna(False).astype(bool).equals(
        result["prequential_base_score"].notna()
    ):
        # Scores should never be imputed.  A rare score failure is allowed,
        # but a score on a target-free-ineligible row is a contract breach.
        invalid_score = result["prequential_base_score"].notna() & ~result["base_feature_eligible"].fillna(False).astype(bool)
        if invalid_score.any():
            raise ValueError("base score exists on a feature-ineligible short row")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), hashes


def _safe_entropy(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0
    centered = values - np.median(values)
    scale = np.median(np.abs(centered)) * 1.4826
    scale = max(float(scale), 1e-6)
    z = np.clip(centered / scale, -12.0, 12.0)
    weights = np.exp(z - z.max())
    weights /= weights.sum()
    entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum()
    return float(entropy / np.log(len(weights)))


def _score_geometry(group: pd.DataFrame) -> dict[str, float]:
    ordered = group.sort_values(["prequential_base_score", "candidate_id"], ascending=[False, True], kind="stable")
    score = _finite(ordered["prequential_base_score"]).to_numpy(float)
    rank = _finite(ordered["prequential_base_rank42"]).to_numpy(float)
    score = score[np.isfinite(score)]
    rank = rank[np.isfinite(rank)]
    if len(score) == 0:
        raise ValueError("target-free base-eligible P0 universe contains no finite score")
    top = float(score[0])
    eps = max(abs(top), 1e-6)
    quant = lambda q: float(np.quantile(score, q))
    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med)))
    result = {
        "geom_candidate_count": float(len(score)),
        "geom_top1_score": top,
        "geom_top1_rank42": float(rank[0]) if len(rank) else np.nan,
        "geom_top1_minus_top2": float(top - score[min(1, len(score) - 1)]),
        "geom_top1_minus_top4": float(top - score[min(3, len(score) - 1)]),
        "geom_top1_minus_top8": float(top - score[min(7, len(score) - 1)]),
        "geom_top1_minus_median": float(top - med),
        "geom_score_std": float(np.std(score)),
        "geom_score_mad": mad,
        "geom_score_iqr": float(quant(.75) - quant(.25)),
        "geom_score_p90_p50": float(quant(.90) - quant(.50)),
        "geom_score_p99_p90": float(quant(.99) - quant(.90)),
        "geom_top_tail_slope": float(top - quant(.90)),
        "geom_fraction_within_1pct_top": float(np.mean((top - score) <= .01 * eps)),
        "geom_fraction_within_2pct_top": float(np.mean((top - score) <= .02 * eps)),
        "geom_fraction_within_5pct_top": float(np.mean((top - score) <= .05 * eps)),
        "geom_score_entropy": _safe_entropy(score),
        "geom_rank_entropy": _safe_entropy(rank),
        "geom_count_rank42_ge_p90": float(np.sum(rank >= .90)),
        "geom_count_rank42_ge_p95": float(np.sum(rank >= .95)),
        "geom_count_rank42_ge_p99": float(np.sum(rank >= .99)),
    }
    return result


def _top1_population(ledger: pd.DataFrame) -> pd.DataFrame:
    """Select P0 rank-1 target-free candidate and score geometry per hour."""
    eligible = ledger.loc[
        ledger["base_feature_eligible"].fillna(False).astype(bool)
        & _finite(ledger["prequential_base_score"]).notna()
    ].copy()
    if eligible.empty:
        raise ValueError("no target-free base-eligible short candidates")
    blocks: list[pd.DataFrame] = []
    for decision, group in eligible.groupby("__decision_ts__", sort=True):
        ordered = group.sort_values(["prequential_base_score", "candidate_id"], ascending=[False, True], kind="stable")
        winner = ordered.head(1).copy()
        geometry = _score_geometry(ordered)
        for field, value in geometry.items():
            winner[field] = value
        # Global / market fields should be contemporaneous market-universe
        # values.  The F90 source uses deterministic common fields; median is
        # deliberately used as a defensive reduction for fields that vary by
        # asset in a particular historical materialisation.
        for field in MARKET_STATE:
            winner[f"market__{field}"] = float(_finite(group[field]).median())
        winner["decision_hour_utc"] = float(_utc(decision).hour)
        winner["decision_weekend"] = float(_utc(decision).dayofweek >= 5)
        blocks.append(winner)
    result = pd.concat(blocks, ignore_index=True)
    if result["__decision_ts__"].duplicated().any():
        raise AssertionError("P0 top1 selection did not yield one target-free row per decision hour")
    return result.sort_values("__decision_ts__", kind="stable").reset_index(drop=True)


def _add_recent_conversion_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Add strictly-resolved rolling top1 policy state.

    The outcome queue is advanced with ``label_available < decision``.  This
    intentionally excludes a label resolving exactly at a decision boundary,
    which is stronger than the usual non-strict causal convention.
    """
    result = frame.copy()
    recent_fields: list[str] = []
    for days in HISTORY_DAYS:
        recent_fields.extend((
            f"recent_{days}d_ev_bps", f"recent_{days}d_hit_rate",
            f"recent_{days}d_gt100_rate", f"recent_{days}d_stop_rate",
            f"recent_{days}d_timeout_rate", f"recent_{days}d_trailing_rate",
            f"recent_{days}d_support",
        ))
    recent_fields.extend((
        "recent_56d_same_session_ev_bps", "recent_56d_same_session_support",
        "recent_56d_same_direction_ev_bps", "recent_56d_same_direction_support",
        "recent_56d_same_vol_ev_bps", "recent_56d_same_vol_support",
    ))
    for field in recent_fields:
        result[field] = np.nan

    valid = _valid_policy(result)
    outcomes = result.loc[valid, [
        "__decision_ts__", "policy_label_available_at", "p0_canonical_net_bps",
        "decision_hour_utc", "mkt_ret_24h", "xasset_mkt_ob_stress_z_24h",
    ]].copy()
    # Exact exit classes are optional in a P0 ledger.  The policy outcome is
    # enough for non-ambiguous risk rates; these conservative proxies remain
    # causal and avoid pretending an unavailable label exists.
    outcomes["is_stop"] = _finite(outcomes["p0_canonical_net_bps"]).le(-200.0)
    outcomes["is_timeout"] = _finite(outcomes["p0_canonical_net_bps"]).between(-50.0, 50.0, inclusive="both")
    outcomes["is_trailing"] = _finite(outcomes["p0_canonical_net_bps"]).gt(100.0)
    outcomes = outcomes.sort_values(["policy_label_available_at", "__decision_ts__"], kind="stable").reset_index(drop=True)
    available_ns = outcomes["policy_label_available_at"].astype("int64").to_numpy()
    cursor = 0
    active: deque[dict[str, object]] = deque()

    def _mean(rows: list[dict[str, object]], key: str) -> float:
        values = [float(row[key]) for row in rows if pd.notna(row[key])]
        return float(np.mean(values)) if values else np.nan

    for index, row in result.iterrows():
        decision = _utc(row["__decision_ts__"])
        end = int(np.searchsorted(available_ns, decision.value, side="left"))
        while cursor < end:
            active.append(outcomes.iloc[cursor].to_dict())
            cursor += 1
        minimum = decision - pd.Timedelta(days=max(HISTORY_DAYS))
        while active and _utc(active[0]["__decision_ts__"]) < minimum:
            active.popleft()
        active_rows = list(active)
        for days in HISTORY_DAYS:
            start = decision - pd.Timedelta(days=days)
            rows = [item for item in active_rows if _utc(item["__decision_ts__"]) >= start]
            result.at[index, f"recent_{days}d_support"] = float(len(rows))
            result.at[index, f"recent_{days}d_ev_bps"] = _mean(rows, "p0_canonical_net_bps")
            if rows:
                net = np.asarray([float(item["p0_canonical_net_bps"]) for item in rows])
                result.at[index, f"recent_{days}d_hit_rate"] = float(np.mean(net > 0.0))
                result.at[index, f"recent_{days}d_gt100_rate"] = float(np.mean(net > 100.0))
                result.at[index, f"recent_{days}d_stop_rate"] = float(np.mean([bool(item["is_stop"]) for item in rows]))
                result.at[index, f"recent_{days}d_timeout_rate"] = float(np.mean([bool(item["is_timeout"]) for item in rows]))
                result.at[index, f"recent_{days}d_trailing_rate"] = float(np.mean([bool(item["is_trailing"]) for item in rows]))
        last56 = [item for item in active_rows if _utc(item["__decision_ts__"]) >= decision - pd.Timedelta(days=56)]
        hour = float(row["decision_hour_utc"])
        direction = np.sign(float(row["mkt_ret_24h"])) if pd.notna(row["mkt_ret_24h"]) else 0.0
        vol = float(row["xasset_mkt_ob_stress_z_24h"]) if pd.notna(row["xasset_mkt_ob_stress_z_24h"]) else np.nan
        same_session = [item for item in last56 if float(item["decision_hour_utc"]) == hour]
        same_direction = [item for item in last56 if np.sign(float(item["mkt_ret_24h"])) == direction]
        if np.isfinite(vol):
            # A fixed distance rule avoids an outcome-dependent regime fit.
            same_vol = [item for item in last56 if pd.notna(item["xasset_mkt_ob_stress_z_24h"]) and abs(float(item["xasset_mkt_ob_stress_z_24h"]) - vol) <= .5]
        else:
            same_vol = []
        for prefix, rows in (("same_session", same_session), ("same_direction", same_direction), ("same_vol", same_vol)):
            result.at[index, f"recent_56d_{prefix}_support"] = float(len(rows))
            # Shrink sparse conditional states toward the full causal 56d
            # history; the support is retained as a separate model feature.
            local = _mean(rows, "p0_canonical_net_bps")
            global_ = _mean(last56, "p0_canonical_net_bps")
            if np.isnan(global_):
                shrunk = local
            elif np.isnan(local):
                shrunk = global_
            else:
                shrunk = (len(rows) * local + 10.0 * global_) / (len(rows) + 10.0)
            result.at[index, f"recent_56d_{prefix}_ev_bps"] = shrunk
    return result


def _feature_blocks(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    geometry = tuple(column for column in frame.columns if column.startswith("geom_"))
    market = tuple(f"market__{field}" for field in MARKET_STATE) + ("decision_hour_utc", "decision_weekend")
    recent = tuple(column for column in frame.columns if column.startswith("recent_"))
    missing = sorted(set(BASE_INPUT).difference(frame.columns))
    if missing:
        raise AssertionError(f"top1 P0 absolute-meta table lost base inputs: {missing}")
    return {
        "base": tuple(BASE_INPUT),
        "geometry": geometry,
        "market": market,
        "recent": recent,
    }


@dataclass(frozen=True)
class Arm:
    name: str
    target: Literal["anchor", "direct", "binary0", "binary100", "ordinal", "residual"]
    blocks: tuple[str, ...]
    description: str


ARMS: tuple[Arm, ...] = (
    Arm("M0", "anchor", (), "Causal same-model P0 policy anchor only."),
    Arm("M1", "direct", ("base",), "Direct Huber policy-net conversion."),
    Arm("M2", "binary0", ("base",), "P(policy net > 0), OOF bps calibrated."),
    Arm("M3", "binary100", ("base",), "P(policy net > 100), OOF bps calibrated."),
    Arm("M4", "ordinal", ("base",), "Six-class ordinal policy margin, OOF bps calibrated."),
    Arm("M5", "residual", ("base",), "P0 anchor plus absolute Huber residual."),
    Arm("M6", "residual", ("base", "geometry"), "M5 plus target-free P0 score geometry."),
    Arm("M7", "residual", ("base", "geometry", "market"), "M6 plus causal market state."),
    Arm("M8", "residual", ("base", "geometry", "market", "recent"), "M7 plus strictly-resolved recent conversion state."),
)


def _matrix(frame: pd.DataFrame, fields: Iterable[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    columns = list(fields)
    values = frame.loc[:, columns].apply(_finite)
    if medians is None:
        medians = values.median().fillna(0.0)
    values = values.fillna(medians).fillna(0.0).astype(np.float32)
    return values, medians


def _model(kind: str, *, seed: int):
    common = dict(
        n_estimators=160, learning_rate=.035, max_depth=3, num_leaves=15,
        min_child_samples=35, subsample=.85, colsample_bytree=.85,
        reg_lambda=4.0, reg_alpha=.10, random_state=seed, n_jobs=-1,
        verbosity=-1,
    )
    if kind in {"direct", "residual"}:
        return LGBMRegressor(objective="huber", alpha=.90, **common)
    if kind in {"binary0", "binary100"}:
        return LGBMClassifier(objective="binary", class_weight="balanced", **common)
    if kind == "ordinal":
        return LGBMClassifier(objective="multiclass", num_class=len(ORDINAL_VALUES), class_weight="balanced", **common)
    raise ValueError(kind)


def _target(frame: pd.DataFrame, kind: str) -> np.ndarray:
    net = _finite(frame["p0_canonical_net_bps"]).clip(-POLICY_CLIP_BPS, POLICY_CLIP_BPS).to_numpy(float)
    if kind == "direct":
        return net
    if kind == "residual":
        anchor = _finite(frame["prequential_base_anchor_bps"]).to_numpy(float)
        return np.clip(net - anchor, -POLICY_CLIP_BPS, POLICY_CLIP_BPS)
    if kind == "binary0":
        return (net > 0.0).astype(int)
    if kind == "binary100":
        return (net > 100.0).astype(int)
    if kind == "ordinal":
        return np.digitize(net, ORDINAL_EDGES, right=True).astype(int)
    raise ValueError(kind)


def _raw_prediction(model, x: pd.DataFrame, kind: str, anchor: np.ndarray) -> np.ndarray:
    if kind in {"direct", "residual"}:
        value = np.asarray(model.predict(x), dtype=float)
        if kind == "residual":
            value += anchor
        return value
    if kind in {"binary0", "binary100"}:
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if kind == "ordinal":
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        return probabilities @ ORDINAL_VALUES
    raise ValueError(kind)


def _chronological_oof_raw(train: pd.DataFrame, fields: tuple[str, ...], kind: str, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """OOF raw predictions used solely for the train-only score-to-bps map."""
    ordered = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    n = len(ordered)
    # Expanding folds: the earliest segment is warm-up, never calibration.
    boundaries = np.linspace(0, n, OOF_SPLITS + 2, dtype=int)
    pieces_raw: list[np.ndarray] = []
    pieces_idx: list[np.ndarray] = []
    for fold in range(OOF_SPLITS):
        fit_end = int(boundaries[fold + 1])
        val_end = int(boundaries[fold + 2])
        if fit_end < max(160, MIN_TRAIN_ROWS // 3) or val_end <= fit_end:
            continue
        fit = ordered.iloc[:fit_end]
        valid = ordered.iloc[fit_end:val_end]
        x_fit, medians = _matrix(fit, fields)
        x_valid, _ = _matrix(valid, fields, medians)
        estimator = _model(kind, seed=seed + fold)
        estimator.fit(x_fit, _target(fit, kind))
        raw = _raw_prediction(
            estimator, x_valid, kind,
            _finite(valid["prequential_base_anchor_bps"]).to_numpy(float),
        )
        pieces_raw.append(raw)
        pieces_idx.append(np.arange(fit_end, val_end, dtype=int))
    if not pieces_raw:
        raise ValueError("insufficient chronological support for OOF calibration")
    raw = np.concatenate(pieces_raw)
    indices = np.concatenate(pieces_idx)
    return indices, raw


def _fit_calibrator(train: pd.DataFrame, fields: tuple[str, ...], kind: str, *, seed: int) -> tuple[IsotonicRegression, float, float, int]:
    indices, raw = _chronological_oof_raw(train, fields, kind, seed=seed)
    observed = _finite(train.iloc[indices]["p0_canonical_net_bps"]).clip(-POLICY_CLIP_BPS, POLICY_CLIP_BPS).to_numpy(float)
    if len(raw) < MIN_OOF_ROWS or np.unique(raw).size < 4:
        raise ValueError("insufficient OOF score diversity for absolute-EV calibration")
    # Direction is learned from OOF data.  Isotonic must never silently force
    # an increasing calibration if a short meta arm is directionally inverted.
    correlation = pd.Series(raw).corr(pd.Series(observed), method="spearman")
    increasing = bool(np.nan_to_num(correlation, nan=0.0) >= 0.0)
    calibrator = IsotonicRegression(increasing=increasing, out_of_bounds="clip", y_min=-POLICY_CLIP_BPS, y_max=POLICY_CLIP_BPS)
    calibrator.fit(raw, observed)
    return calibrator, float(correlation), float(np.quantile(raw, .70)), len(raw)


def _fit_predict_arm(train: pd.DataFrame, held: pd.DataFrame, arm: Arm, blocks: dict[str, tuple[str, ...]], *, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    held = held.copy()
    if arm.target == "anchor":
        held["expected_net_bps"] = _finite(held["prequential_base_anchor_bps"])
        held["raw_meta_score"] = held["expected_net_bps"]
        train_scores = _finite(train["prequential_base_anchor_bps"])
        held["train_p70_expected_bps"] = float(train_scores.quantile(.70))
        held["train_p80_expected_bps"] = float(train_scores.quantile(.80))
        return held, {
            "feature_count": 1, "feature_fields": ["prequential_base_anchor_bps"],
            "calibration": "same-model base reserve map", "oof_calibration_rows": 0,
            "oof_spearman": np.nan,
            "train_p70": float(train_scores.quantile(.70)), "train_p80": float(train_scores.quantile(.80)),
        }
    fields = tuple(field for block in arm.blocks for field in blocks[block])
    if len(fields) != len(set(fields)):
        raise AssertionError(f"{arm.name} duplicated an input field")
    x_train, medians = _matrix(train, fields)
    x_held, _ = _matrix(held, fields, medians)
    calibrator, rho, p70_raw, oof_rows = _fit_calibrator(train, fields, arm.target, seed=seed)
    estimator = _model(arm.target, seed=seed + 100)
    estimator.fit(x_train, _target(train, arm.target))
    raw = _raw_prediction(
        estimator, x_held, arm.target,
        _finite(held["prequential_base_anchor_bps"]).to_numpy(float),
    )
    held["raw_meta_score"] = raw.astype(np.float32)
    held["expected_net_bps"] = calibrator.predict(raw).astype(np.float32)
    # Causal top-30/top-20 thresholds are score-domain cutoffs fitted from
    # chronological OOF predictions on training rows—not held percentiles.
    _, oof_raw = _chronological_oof_raw(train, fields, arm.target, seed=seed)
    held["train_p70_expected_bps"] = float(calibrator.predict(np.asarray([np.quantile(oof_raw, .70)]))[0])
    held["train_p80_expected_bps"] = float(calibrator.predict(np.asarray([np.quantile(oof_raw, .80)]))[0])
    return held, {
        "feature_count": len(fields), "feature_fields": list(fields),
        "calibration": "chronological_oof_isotonic_to_policy_net_bps",
        "oof_calibration_rows": int(oof_rows), "oof_spearman": float(rho),
        "train_p70_raw": p70_raw,
        "train_p70": float(held["train_p70_expected_bps"].iat[0]),
        "train_p80": float(held["train_p80_expected_bps"].iat[0]),
    }


def _selected_metrics(selected: pd.DataFrame) -> dict[str, float]:
    net = _finite(selected["p0_canonical_net_bps"])
    if selected.empty:
        return {"trades": 0.0, "share": 0.0, "net_bps_per_trade": np.nan, "total_net_bps": np.nan, "positive_rate": np.nan}
    return {
        "trades": float(len(selected)), "share": np.nan,
        "net_bps_per_trade": float(net.mean()), "total_net_bps": float(net.sum()),
        "positive_rate": float(np.mean(net > 0.0)),
    }


def _metrics(predictions: pd.DataFrame, *, arm: Arm, held_month: pd.Timestamp) -> list[dict[str, object]]:
    valid = predictions.loc[_valid_policy(predictions) & _finite(predictions["expected_net_bps"]).notna()].copy()
    valid["p0_canonical_net_bps"] = _finite(valid["p0_canonical_net_bps"])
    rows: list[dict[str, object]] = []
    if valid.empty:
        return rows
    y = valid["p0_canonical_net_bps"].to_numpy(float)
    score = valid["expected_net_bps"].to_numpy(float)
    base = {
        "arm": arm.name, "held_month": held_month.strftime("%Y-%m"), "valid_hours": int(len(valid)),
        "score_net_spearman": float(pd.Series(score).corr(pd.Series(y), method="spearman")),
    }
    for event, threshold in (("net_gt0", 0.0), ("net_gt100", 100.0)):
        truth = (y > threshold).astype(int)
        if truth.min() != truth.max():
            base[f"auc_{event}"] = float(roc_auc_score(truth, score))
            base[f"prauc_{event}"] = float(average_precision_score(truth, score))
        else:
            base[f"auc_{event}"] = np.nan
            base[f"prauc_{event}"] = np.nan
    definitions: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame], bool]] = []
    for threshold in ADMISSION_LEVELS:
        definitions.append((f"expected_ge_{int(threshold)}", lambda f, t=threshold: f.loc[_finite(f["expected_net_bps"]).ge(t)], True))
    definitions.extend((
        ("causal_train_top30", lambda f: f.loc[_finite(f["expected_net_bps"]).ge(float(f["train_p70_expected_bps"].iat[0]))], True),
        ("causal_train_top20", lambda f: f.loc[_finite(f["expected_net_bps"]).ge(float(f["train_p80_expected_bps"].iat[0]))], True),
        # These are useful diagnostic rank curves only.  They use the held
        # month and are separately labelled to prevent deployment misuse.
        ("diagnostic_held_top30", lambda f: f.nlargest(max(1, int(np.ceil(.30 * len(f)))), "expected_net_bps"), False),
        ("diagnostic_held_top20", lambda f: f.nlargest(max(1, int(np.ceil(.20 * len(f)))), "expected_net_bps"), False),
    ))
    for name, select, causal in definitions:
        chosen = select(valid)
        values = _selected_metrics(chosen)
        values["share"] = float(len(chosen) / len(valid))
        rows.append({**base, **values, "selection": name, "causal_selection": bool(causal)})
    return rows


def _aggregate_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (arm, selection, causal), block in monthly.groupby(["arm", "selection", "causal_selection"], sort=True):
        weights = block["trades"].to_numpy(float)
        total = float(block["total_net_bps"].sum(min_count=1))
        trades = float(weights.sum())
        ev = total / trades if trades else np.nan
        rows.append({
            "arm": arm, "selection": selection, "causal_selection": bool(causal),
            "months": int(block["held_month"].nunique()), "trades": trades,
            "net_bps_per_trade": ev, "total_net_bps": total,
            "worst_month_net_bps_per_trade": float(block["net_bps_per_trade"].min()),
            "positive_months": int((block["net_bps_per_trade"] > 0).sum()),
            "score_net_spearman_mean": float(block["score_net_spearman"].mean()),
            "auc_net_gt0_mean": float(block["auc_net_gt0"].mean()),
            "auc_net_gt100_mean": float(block["auc_net_gt100"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["causal_selection", "selection", "net_bps_per_trade"], ascending=[False, True, False], kind="stable")


def run(*, ledger_roots: list[Path], start: pd.Timestamp, end_exclusive: pd.Timestamp, out: Path, seed: int) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable short absolute-meta output exists: {out}")
    ledger, ledger_hashes = _load_ledger(ledger_roots)
    top1 = _top1_population(ledger)
    top1 = _add_recent_conversion_state(top1)
    blocks = _feature_blocks(top1)
    top1.to_parquet(out.parent / f".{out.name}.top1_staging.parquet", index=False, compression="zstd") if False else None
    out.mkdir(parents=True)
    top1.to_parquet(out / "short_p0_top1_hourly_population.parquet", index=False, compression="zstd")
    months = list(pd.date_range(start.normalize().replace(day=1), end_exclusive, freq="MS", inclusive="left"))
    monthly_metrics: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    predicted: list[pd.DataFrame] = []
    feature_contract: dict[str, object] = {
        "side": SIDE, "base": list(blocks["base"]), "geometry": list(blocks["geometry"]),
        "market": list(blocks["market"]), "recent": list(blocks["recent"]),
        "recent_feature_causality": "labels are added only when policy_label_available_at < decision timestamp",
    }
    for month in months:
        next_month = month + pd.offsets.MonthBegin(1)
        held = top1.loc[top1["__decision_ts__"].ge(month) & top1["__decision_ts__"].lt(next_month)].copy()
        # Train only on previously resolved P0 winners.  The input base score
        # is itself prequential from its own held month.
        train = top1.loc[
            top1["policy_label_available_at"].lt(month)
            & top1["__decision_ts__"].lt(month)
            & _valid_policy(top1)
        ].copy()
        if held.empty or len(train) < MIN_TRAIN_ROWS:
            audits.append({"held_month": month.strftime("%Y-%m"), "status": "skipped_insufficient_train", "held_hours": len(held), "train_hours": len(train)})
            continue
        for arm_idx, arm in enumerate(ARMS):
            try:
                arm_prediction, details = _fit_predict_arm(train, held, arm, blocks, seed=seed + 1000 * len(audits) + 13 * arm_idx)
            except ValueError as error:
                audits.append({"held_month": month.strftime("%Y-%m"), "arm": arm.name, "status": "skipped", "reason": str(error), "held_hours": len(held), "train_hours": len(train)})
                continue
            arm_prediction["arm"] = arm.name
            arm_prediction["held_month"] = month.strftime("%Y-%m")
            arm_prediction["score_family"] = "short_p0_f90_absolute_conversion"
            predicted.append(arm_prediction.loc[:, [
                "candidate_id", "__decision_ts__", "__symbol__", "side_name", "arm", "held_month",
                "prequential_base_score", "prequential_base_rank42", "prequential_base_anchor_bps",
                "expected_net_bps", "raw_meta_score", "train_p70_expected_bps", "train_p80_expected_bps",
                "policy_path_valid", "policy_label_available_at", "p0_canonical_net_bps",
            ]].copy())
            monthly_metrics.extend(_metrics(arm_prediction, arm=arm, held_month=month))
            audits.append({
                "held_month": month.strftime("%Y-%m"), "arm": arm.name, "status": "complete",
                "held_hours": len(held), "train_hours": len(train), **details,
            })
    if not predicted:
        raise RuntimeError("absolute conversion funnel produced no strict-OOS arm predictions")
    prediction_frame = pd.concat(predicted, ignore_index=True)
    prediction_frame.to_parquet(out / "short_absolute_conversion_oof_predictions.parquet", index=False, compression="zstd")
    metric_frame = pd.DataFrame(monthly_metrics)
    metric_frame.to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    aggregate = _aggregate_metrics(metric_frame)
    aggregate.to_parquet(out / "aggregate_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    (out / "feature_contract.json").write_text(json.dumps(feature_contract, indent=2) + "\n")
    decision = {
        "schema": "strict_r3_short_p0_absolute_conversion_funnel_v1",
        "side": SIDE,
        "status": "complete",
        "scope": "short only; long pipeline unmodified",
        "models": [{"name": arm.name, "target": arm.target, "description": arm.description, "blocks": list(arm.blocks)} for arm in ARMS],
        "training": "monthly expanding strict-prequential P0-top1 rows; labels resolve strictly before held month",
        "calibration": "chronological OOF model predictions -> isotonic expected policy net bps; M0 uses the existing same-model P0 anchor",
        "policy": "canonical short P0 parent policy net bps; labels are joined only after target-free P0 top1 selection",
        "deployment_note": "diagnostic_held_top20/top30 metrics are not live admission rules; only expected-bps and causal-train threshold selections are causal",
        "success_gate": "top 20–30% causal admission should be materially positive, robust by month, and approach the +100 to +150 bps/trade initial target before any top-K or consensus work",
        "ledger_sources": ledger_hashes,
    }
    (out / "run_manifest.json").write_text(json.dumps(decision, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, action="append", required=True)
    parser.add_argument("--start", default="2024-05-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2025-01-01T00:00:00Z")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start, end = _utc(args.start), _utc(args.end_exclusive)
    if end <= start:
        raise ValueError("end must follow start")
    print(run(ledger_roots=args.ledger_root, start=start, end_exclusive=end, out=args.out, seed=int(args.seed)))


if __name__ == "__main__":
    main()
