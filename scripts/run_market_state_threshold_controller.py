#!/usr/bin/env python3
"""Market-state encoder plus deterministic strategy-threshold controller.

This is a research ablation around the deployed simple-policy candidate ledger.
It keeps scores, rank references, costs and auction ordering fixed, then applies
penalty-only state-conditioned base-threshold increases before the existing
occupancy-aware dynamic threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)
from extreme_price_movements.performance_regimes.spectral_position import (  # noqa: E402
    MarketSpectralPositionConfig,
    fit_market_spectral_position_encoder,
    transform_market_spectral_position,
)


DEFAULT_TRAIN_BROAD = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_TRAIN_DEPLOYABLE = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_EVAL_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_arm_A0_anchor_only_20260625_jun15_22"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_TRAIN_FEATURE_STORE: Path | None = None
DEFAULT_EVAL_FEATURE_STORE = Path("data_perp/features/20260627_120000")
DEFAULT_POLICY_MANIFEST = Path(
    "data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625"
    "/A0_anchor_only/portfolio_policy_ablation_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_threshold_controller_20260625"
)
DEFAULT_DATA_ROOT = Path("data_perp")
DEFAULT_RANK_REFERENCE_RUN_ID = "reliability_blend_anchor_rank_reference_20260625_prejune"

HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
DECISION_KEY_COLS = ("timestamp", "symbol", "side", "strategy_id")
OUTCOME_TOKENS = (
    "return",
    "pnl",
    "exit",
    "target",
    "barrier_hit",
    "future",
    "label",
)
MARKET_STATE_FORBIDDEN_COLUMN_TOKENS = (
    "candidate_count",
    "strategy_count",
    "symbol_count",
    "accepted",
    "portfolio",
    "pnl",
    "net_return",
    "gross_return",
    "exit",
    "target",
    "label",
    "barrier",
    "threshold",
    "normalized_rank",
    "strategy_rank",
    "policy_rank",
    "rank_pct",
    "rank_score",
    "rank_path",
    "rank_ge",
    "__rank",
    "rank_mean",
    "rank_max",
    "score_path",
    "__score",
    "cross_head_score",
    "calibrated_score",
    "anchor_score",
    "prediction",
    "leaf",
    "centroid",
    "regime_",
    "state_model",
    "vote_entropy",
    "uncertainty",
    "fee",
    "slippage",
    "depth",
    "imbalance",
    "microprice",
    "orderbook",
    "order_book",
    "ob_top",
    "xasset_ob",
    "_ob_",
    "bid",
    "ask",
)
MARKET_STATE_ALLOWED_SPREAD_SUBSTRINGS = (
    "spread_proxy",
    "ema50_ema200_spread_atr",
)
BASE_NUMERIC_FEATURES = (
    "normalized_rank_score",
    "strategy_rank_pct",
    "rank_pct",
    "calibrated_score",
    "base_strategy_threshold",
    "deployment_rank_threshold",
    "expected_spread_bps",
    "expected_half_spread_bps",
    "spread_cost_bps",
    "fees_bps",
    "slippage_bps",
    "liquidity_capacity_weight",
    "barrier_pct",
    "policy_effective_barrier_pct",
)
AXIS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "state_shock": (
        "mkt_ret",
        "ret_eq",
        "symbol_minus_mkt",
        "price_gap",
        "gap_bps",
        "range_expansion",
        "tail_asymmetry",
    ),
    "state_realized_vol": (
        "vol",
        "rv",
        "atr",
        "barrier_pct",
        "range",
    ),
    "state_compression": (
        "compression",
        "bollinger",
        "low_vol",
        "range",
        "barrier_pct",
    ),
    "state_trend": (
        "trend",
        "efficiency",
        "autocorr",
        "slope",
        "direction",
    ),
    "state_deleveraging": (
        "oi_",
        "fund",
        "basis",
        "leverage",
        "unwind",
    ),
    "state_liquidity_stress_proxy": (
        "spread_proxy",
        "liquidity",
        "amihud",
        "range_to_volume",
        "volume",
    ),
    "state_transition": (
        "cov_shift",
        "transition",
        "volatility",
        "breadth",
        "dispersion",
    ),
}
OBSERVED_RELIABILITY_STATE_COLUMNS = {
    "state_input_coverage",
    "state_extreme_value_share",
    "state_novelty",
    "state_drift_score",
    "state_uncertainty",
    "state_low_input_coverage",
}
SPECTRAL_POSITION_STATE_COLUMNS = (
    "state_spectral_eig_lambda1_share",
    "state_spectral_eig_top3_share",
    "state_spectral_eig_effective_rank",
    "state_spectral_eig_entropy",
    "state_spectral_eig_gap_1_2",
    "state_spectral_eig_gap_ratio_1_2",
    "state_spectral_eig_condition",
    "state_spectral_pc1_score",
    "state_spectral_pc2_score",
    "state_spectral_pc3_score",
    "state_spectral_pc1_z",
    "state_spectral_pc2_z",
    "state_spectral_pc3_z",
    "state_spectral_abs_pc1_z",
    "state_spectral_abs_pc2_z",
    "state_spectral_abs_pc3_z",
    "state_spectral_sum_abs_top3_pc_z",
    "state_spectral_projection_norm_top3",
    "state_spectral_top3_reconstruction_error",
    "state_spectral_top3_reconstruction_ratio",
    "state_spectral_top3_mahalanobis",
)
FEATURE_STORE_COLUMN_CANDIDATES = (
    # Cross-market returns / breadth / dispersion.
    "mkt_ret_eq_1h",
    "mkt_ret_eq_4h",
    "mkt_ret_eq_24h",
    "market_breadth_1h",
    "market_breadth_4h",
    "market_breadth_24h",
    "market_dispersion_1h",
    "market_dispersion_4h",
    "market_dispersion_24h",
    "symbol_minus_mkt_ret_1h",
    "symbol_minus_mkt_ret_4h",
    "symbol_minus_mkt_ret_24h",
    # Trend / path / consolidation.
    "ema20_gt_ema50",
    "ema50_gt_ema200",
    "ema50_ema200_spread_atr",
    "ema50_slope",
    "trend_strength_percentile",
    "trend_acceleration",
    "efficiency_ratio_20",
    "path_efficiency_24",
    "choppiness_index_20",
    "direction_entropy_20",
    "spectral_entropy_ret_24",
    "return_autocorr_48",
    "variance_ratio_10_48",
    "up_down_semivol_ratio_tanh",
    "up_down_return_mass_ratio_tanh",
    "tail_asymmetry_q90_q10_atr_norm",
    # Volatility / compression.
    "realized_volatility_24h",
    "rv_24h",
    "volatility_ratio_short_long",
    "volatility_of_volatility_48",
    "volatility_autocorr_48",
    "atr_percentile",
    "atr_change_rate",
    "atr_compression_ratio",
    "compression_ratio",
    "compression_score",
    "bollinger_band_width",
    "range_expansion_ratio",
    "true_range_percentile",
    # OI / funding / leverage unwind.
    "oi_value_1d_chg_z_90d",
    "oi_value_1d_log_chg_z_90d",
    "oi_value_3d_chg_z_90d",
    "oi_value_3d_log_chg_z_90d",
    "oi_value_7d_chg_z_90d",
    "oi_value_7d_log_chg_z_90d",
    "mkt_oi_chg_z_24h",
    "mkt_oi_breadth_rising_24h",
    "mkt_oi_dispersion_24h",
    "funding_rate",
    "funding_z",
    "funding_abs_z",
    "funding_per_hour_z",
    "funding_rank_30d",
    "funding_persistence",
    "funding_mom_2h",
    "funding_mom_4h",
    "funding_mom_8h",
    "oi_1d_x_funding",
    "oi_3d_x_funding",
    "oi_7d_x_funding",
    "price_x_oi_1d",
    "price_x_oi_3d",
    "price_x_oi_7d",
    # Liquidity / execution state.  Only OHLCV-derived proxies are permitted
    # here; actual order-book spread/depth/imbalance columns are forbidden by
    # the market-state contract.
    "liquidity_ratio_peer_resid",
    "amihud_illiq",
    "amihud_z",
    "amihud_z_peer_resid",
    "spread_proxy_hl_range_bps_robust_z",
    "spread_proxy_abs_return_bps_robust_z",
    "spread_proxy_gap_bps_robust_z",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _infer_head(strategy_id: Any) -> str:
    sid = str(strategy_id)
    for head in HEADS:
        if sid.startswith(head):
            return head
    return "unknown"


def _default_strategy_threshold(strategy_id: Any) -> float:
    head = _infer_head(strategy_id)
    return 0.71 if head == "long_bars" else 0.70


def _load_policy_params(path: Path, variant: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("variant_params", {}).get(variant)
    if not isinstance(params, dict):
        raise KeyError(f"Missing variant_params[{variant!r}] in {path}")
    return portfolio_policy_params_from_live_config(params), payload


def _load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "head" not in df.columns:
        df["head"] = df["strategy_id"].map(_infer_head)
    for col in ("symbol", "side", "strategy_id", "head"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "deployment_rank_threshold" not in df.columns:
        df["deployment_rank_threshold"] = df.get("base_strategy_threshold", np.nan)
    return normalise_candidate_table(df)


def _apply_rank_contract(
    candidates: pd.DataFrame,
    contract: str,
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    rank_reference_run_id: str = DEFAULT_RANK_REFERENCE_RUN_ID,
) -> pd.DataFrame:
    """Apply a causal rank-reference contract to a candidate ledger.

    ``strict`` leaves the materialized rank columns unchanged.
    ``short_boll_timestamp_rank`` repairs only short_boll by using the
    head-by-timestamp cross-sectional rank of its live score.  That ranking is
    available at decision time and does not inspect returns.
    ``anchor_global_policy_rank_reference`` applies the same frozen global
    policy rank reference used by the current T1 static baseline.
    """

    out = candidates.copy()
    if contract == "strict":
        return normalise_candidate_table(out)
    if contract == "anchor_global_policy_rank_reference":
        from scripts.reliability_blend_rank_reference import apply_frozen_policy_rank_reference

        score_col = "anchor_score" if "anchor_score" in out.columns else "calibrated_score"
        ranked, _diag = apply_frozen_policy_rank_reference(
            out,
            data_root=data_root,
            run_id=rank_reference_run_id,
            score_col=score_col,
            allow_window_rank_debug=False,
        )
        ranked["rank_contract_source"] = "anchor_global_policy_rank_reference"
        return normalise_candidate_table(ranked)
    if contract != "short_boll_timestamp_rank":
        raise ValueError(f"Unknown rank contract: {contract}")
    mask = out["head"].astype(str).eq("short_boll")
    if not mask.any():
        return normalise_candidate_table(out)
    score_col = "anchor_score" if "anchor_score" in out.columns else "calibrated_score"
    repaired = (
        pd.to_numeric(out.loc[mask, score_col], errors="coerce")
        .groupby([out.loc[mask, "head"], out.loc[mask, "timestamp"]])
        .rank(method="average", pct=True)
    )
    for col in ("normalized_rank_score", "strategy_rank_pct", "policy_rank_pct", "rank_pct"):
        if col in out.columns:
            out.loc[mask, col] = repaired.to_numpy(dtype=np.float64)
    out.loc[mask, "rank_contract_source"] = "head_timestamp_rank_score"
    return normalise_candidate_table(out)


def _disable_heads(candidates: pd.DataFrame, disabled_heads: set[str]) -> pd.DataFrame:
    if not disabled_heads:
        return normalise_candidate_table(candidates)
    out = candidates.loc[~candidates["head"].astype(str).isin(disabled_heads)].copy()
    return normalise_candidate_table(out)


def _accepted_key_set(accepted: pd.DataFrame) -> set[tuple[Any, ...]]:
    if accepted.empty:
        return set()
    missing = [col for col in DECISION_KEY_COLS if col not in accepted.columns]
    if missing:
        raise KeyError(f"Accepted trades missing decision key columns: {missing}")
    keys = accepted.loc[:, DECISION_KEY_COLS].copy()
    keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True, errors="coerce")
    for col in ("symbol", "side", "strategy_id"):
        keys[col] = keys[col].astype(str)
    return set(map(tuple, keys.drop_duplicates().to_numpy()))


def _assert_unique_decision_keys(frame: pd.DataFrame, *, context: str) -> None:
    if frame.empty:
        return
    missing = [col for col in DECISION_KEY_COLS if col not in frame.columns]
    if missing:
        raise KeyError(f"{context} missing decision key columns: {missing}")
    keys = frame.loc[:, DECISION_KEY_COLS].copy()
    keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True, errors="coerce")
    for col in ("symbol", "side", "strategy_id"):
        keys[col] = keys[col].astype(str)
    dup = keys.duplicated(keep=False)
    if bool(dup.any()):
        preview = keys.loc[dup].head(5).to_dict(orient="records")
        raise ValueError(f"{context} contains duplicate decision keys; preview={preview}")


def _restrict_to_allowed_decision_keys(
    candidates: pd.DataFrame,
    allowed_keys: set[tuple[Any, ...]],
) -> pd.DataFrame:
    """Fail closed on candidates that were not accepted by the baseline arm.

    This is a post-selection overlay: first run the frozen baseline auction,
    then let state thresholds/sizing affect only those baseline decision keys.
    Freed capacity is not backfilled by different candidates.
    """

    if not allowed_keys:
        return normalise_candidate_table(candidates.iloc[0:0].copy())
    keys = candidates.loc[:, DECISION_KEY_COLS].copy()
    keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True, errors="coerce")
    for col in ("symbol", "side", "strategy_id"):
        keys[col] = keys[col].astype(str)
    keep = pd.Series(map(tuple, keys.to_numpy()), index=candidates.index).isin(allowed_keys)
    out = candidates.loc[keep].copy()
    return normalise_candidate_table(out)


def _allowed_decision_key_mask(
    frame: pd.DataFrame,
    allowed_keys: set[tuple[Any, ...]],
) -> pd.Series:
    if not allowed_keys:
        return pd.Series(False, index=frame.index)
    keys = frame.loc[:, DECISION_KEY_COLS].copy()
    keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True, errors="coerce")
    for col in ("symbol", "side", "strategy_id"):
        keys[col] = keys[col].astype(str)
    return pd.Series(map(tuple, keys.to_numpy()), index=frame.index).isin(allowed_keys)


def _parse_disabled_heads(value: str) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def _parse_enabled_heads(value: str) -> set[str] | None:
    heads = _parse_disabled_heads(value)
    return heads or None


def _active_heads(disabled_heads: set[str]) -> list[str]:
    return [head for head in HEADS if head not in set(disabled_heads)]


def _controller_enabled_heads_manifest(
    enabled_heads: set[str] | None,
    disabled_heads: set[str],
) -> dict[str, Any]:
    active = _active_heads(disabled_heads)
    if enabled_heads is None:
        return {
            "controller_enabled_heads": active,
            "controller_enabled_scope": "all_active_heads",
        }
    enabled = [head for head in HEADS if head in set(enabled_heads) and head in set(active)]
    ignored = sorted(set(enabled_heads) - set(active))
    return {
        "controller_enabled_heads": enabled,
        "controller_enabled_scope": "explicit",
        "controller_enabled_heads_ignored_inactive": ignored,
    }


def _parse_int_grid(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    vals: list[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(max(1, int(item)))
    return tuple(sorted(set(vals))) if vals else default


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _robust_z(train: pd.Series, values: pd.Series) -> pd.Series:
    train = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = float(train.median()) if train.notna().any() else 0.0
    iqr = float(train.quantile(0.75) - train.quantile(0.25)) if train.notna().sum() > 3 else 0.0
    scale = iqr / 1.349 if iqr > 1e-12 else float(train.std(ddof=0) or 1.0)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return ((values - med) / scale).clip(-6.0, 6.0)


def _market_state_column_violation(col: str) -> str | None:
    """Return the market-state contract violation for a source column, if any."""

    lower = str(col).lower()
    if lower == "timestamp":
        return None
    if "spread" in lower and not any(
        allowed in lower for allowed in MARKET_STATE_ALLOWED_SPREAD_SUBSTRINGS
    ):
        return "actual_or_unqualified_spread"
    for token in MARKET_STATE_FORBIDDEN_COLUMN_TOKENS:
        if token in lower:
            return token
    return None


def _filter_market_state_source_columns(cols: Iterable[str]) -> list[str]:
    return [str(col) for col in cols if _market_state_column_violation(str(col)) is None]


def _validate_market_state_source_frame(
    frame: pd.DataFrame,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate the market-wide source frame before state encoding.

    The market encoder may consume OHLCV/OI/funding-derived aggregates only.
    Candidate population, model, rank, outcome, true order-book and portfolio
    fields are rejected here so scoring, replay and bundle materialization share
    the same fail-closed contract.
    """

    if "timestamp" not in frame.columns:
        raise ValueError(f"{context} market-state source is missing timestamp")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{context} market-state source contains non-finite timestamps")
    duplicates = int(timestamps.duplicated().sum())
    if duplicates:
        raise ValueError(f"{context} market-state source has duplicate timestamps: {duplicates}")
    bad = {
        str(col): _market_state_column_violation(str(col))
        for col in frame.columns
        if _market_state_column_violation(str(col)) is not None
    }
    if bad:
        preview = ", ".join(f"{col}({reason})" for col, reason in list(bad.items())[:12])
        raise ValueError(f"{context} market-state source has forbidden columns: {preview}")
    numeric_cols = [
        c for c in frame.columns
        if c != "timestamp" and pd.api.types.is_numeric_dtype(frame[c])
    ]
    return {
        "context": context,
        "row_count": int(len(frame)),
        "feature_count": int(len(numeric_cols)),
        "forbidden_column_count": 0,
        "timestamp_unique": True,
        "market_wide_one_row_per_timestamp": True,
    }


def _common_feature_columns(train: pd.DataFrame, eval_df: pd.DataFrame, max_cols: int) -> list[str]:
    common = [c for c in train.columns if c in eval_df.columns]
    numeric_train = set(train.select_dtypes(include=[np.number, "bool"]).columns)
    numeric_eval = set(eval_df.select_dtypes(include=[np.number, "bool"]).columns)
    selected: list[str] = []
    for col in common:
        lower = col.lower()
        if col not in numeric_train or col not in numeric_eval:
            continue
        if any(token in lower for token in OUTCOME_TOKENS):
            continue
        if _market_state_column_violation(col) is not None:
            continue
        if col in BASE_NUMERIC_FEATURES:
            continue
        if any(
            token in lower for toks in AXIS_KEYWORDS.values() for token in toks
        ):
            selected.append(col)
    return selected[: max(1, int(max_cols))]


def _timestamp_aggregates(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    base = work.groupby("timestamp", sort=True).size().rename("candidate_count").to_frame()
    base["strategy_count"] = work.groupby("timestamp")["strategy_id"].nunique()
    base["symbol_count"] = work.groupby("timestamp")["symbol"].nunique()
    rank = _safe_numeric(work, "normalized_rank_score")
    threshold = _safe_numeric(work, "base_strategy_threshold")
    work["_rank_ge_threshold"] = (rank >= threshold).astype(float)
    work["_rank_ge_070"] = (rank >= 0.70).astype(float)
    for col in ("_rank_ge_threshold", "_rank_ge_070"):
        base[col + "_mean"] = work.groupby("timestamp")[col].mean()
    frames = [base]
    for col in feature_cols:
        vals = _safe_numeric(work, col)
        if vals.notna().sum() == 0:
            continue
        tmp = work[["timestamp"]].copy()
        tmp[col] = vals
        agg = tmp.groupby("timestamp")[col].agg(["mean", "std", "min", "max"])
        agg.columns = [f"{col}__{stat}" for stat in agg.columns]
        frames.append(agg)
    for head in HEADS:
        g = work.loc[work["head"].eq(head)].copy()
        if g.empty:
            continue
        h = g.groupby("timestamp").size().rename(f"{head}__rows").to_frame()
        h[f"{head}__frac_rank_ge_threshold"] = g.groupby("timestamp")["_rank_ge_threshold"].mean()
        h[f"{head}__rank_mean"] = _safe_numeric(g, "normalized_rank_score").groupby(g["timestamp"]).mean()
        h[f"{head}__rank_max"] = _safe_numeric(g, "normalized_rank_score").groupby(g["timestamp"]).max()
        h[f"{head}__score_mean"] = _safe_numeric(g, "calibrated_score").groupby(g["timestamp"]).mean()
        frames.append(h)
    out = pd.concat(frames, axis=1).sort_index()
    score_cols = [c for c in out.columns if c.endswith("__score_mean")]
    if score_cols:
        out["cross_head_score_mean_std"] = out[score_cols].std(axis=1)
        out["cross_head_score_mean_range"] = out[score_cols].max(axis=1) - out[score_cols].min(axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True)).fillna(0.0)
    out.index.name = "timestamp"
    return out.reset_index()


def _feature_store_available_columns(feature_dir: Path) -> set[str]:
    files = sorted(feature_dir.glob("symbol=*.parquet"))
    if not files:
        return set()
    sample = pd.read_parquet(files[0])
    return set(map(str, sample.columns))


def _select_feature_store_columns(train_dir: Path | None, eval_dir: Path | None, max_cols: int) -> list[str]:
    if train_dir is None or eval_dir is None or not train_dir.exists() or not eval_dir.exists():
        return []
    train_cols = _feature_store_available_columns(train_dir)
    eval_cols = _feature_store_available_columns(eval_dir)
    common = train_cols & eval_cols
    selected = [
        col
        for col in FEATURE_STORE_COLUMN_CANDIDATES
        if col in common and _market_state_column_violation(col) is None
    ]
    if len(selected) < max_cols:
        # Keep mechanism-specific operator columns when present, but avoid
        # outcome-like fields and raw identifiers.
        for col in sorted(common):
            lower = col.lower()
            if col in selected:
                continue
            if any(token in lower for token in OUTCOME_TOKENS):
                continue
            if _market_state_column_violation(col) is not None:
                continue
            if any(
                token in lower
                for token in (
                    "xs_",
                    "mkt_",
                    "market_",
                    "trend",
                    "efficiency",
                    "choppiness",
                    "entropy",
                    "rv",
                    "volatility",
                    "atr",
                    "compression",
                    "oi_",
                    "fund",
                    "liquidity",
                    "amihud",
                    "spread_proxy",
                    "breadth",
                    "dispersion",
                )
            ):
                selected.append(col)
            if len(selected) >= max_cols:
                break
    return selected[: max(0, int(max_cols))]


def _feature_store_symbol_from_path(path: Path) -> str:
    name = path.stem
    return name[len("symbol="):] if name.startswith("symbol=") else name


def _feature_store_symbol_path_map(feature_dir: Path | None) -> dict[str, Path]:
    if feature_dir is None or not feature_dir.exists():
        return {}
    return {
        _feature_store_symbol_from_path(path): path
        for path in sorted(feature_dir.glob("symbol=*.parquet"))
    }


def _select_feature_store_symbols(
    train_feature_dir: Path | None,
    eval_feature_dir: Path | None,
    *,
    symbol_cap: int,
) -> list[str]:
    train_symbols = set(_feature_store_symbol_path_map(train_feature_dir))
    eval_symbols = set(_feature_store_symbol_path_map(eval_feature_dir))
    common = sorted(train_symbols & eval_symbols)
    if symbol_cap > 0 and len(common) > symbol_cap:
        idx = np.linspace(0, len(common) - 1, int(symbol_cap)).round().astype(int)
        common = [common[i] for i in np.unique(idx)]
    return common


def _feature_store_universe_contract(
    feature_dir: Path | None,
    *,
    all_paths: list[Path],
    selected_paths: list[Path],
    symbol_cap: int,
    reason: str | None = None,
    eligible_symbols_override: list[str] | None = None,
    missing_eligible_symbols: list[str] | None = None,
) -> dict[str, Any]:
    available_symbols = [_feature_store_symbol_from_path(path) for path in all_paths]
    eligible_symbols = (
        [str(symbol) for symbol in eligible_symbols_override]
        if eligible_symbols_override is not None
        else [_feature_store_symbol_from_path(path) for path in selected_paths]
    )
    available_eligible_symbols = [_feature_store_symbol_from_path(path) for path in selected_paths]
    eligible_set = set(eligible_symbols)
    excluded_symbols = [symbol for symbol in available_symbols if symbol not in eligible_set]
    excluded_reason = "outside_frozen_eligible_universe" if eligible_symbols_override is not None else "symbol_cap_subsample"
    excluded_reasons = {symbol: excluded_reason for symbol in excluded_symbols}
    if not all_paths and reason:
        excluded_reasons = {}
    missing_eligible_symbols = [str(symbol) for symbol in (missing_eligible_symbols or [])]
    eligible_symbol_coverage = (
        float(len(available_eligible_symbols) / len(eligible_symbols))
        if eligible_symbols
        else 0.0
    )
    return {
        "universe_definition_version": "feature_store_timestamp_market_state_v1",
        "source": "feature_store_symbol_parquet_files",
        "feature_dir": str(feature_dir) if feature_dir is not None else None,
        "minimum_history": "upstream_feature_store_history_available_at_requested_timestamps",
        "minimum_volume": "upstream_feature_store_volume_filters_or_none",
        "oi_coverage_requirements": "optional_oi_features_are_used_when_present_and_finite",
        "funding_coverage_requirements": "optional_funding_features_are_used_when_present_and_finite",
        "symbol_cap": int(symbol_cap),
        "available_symbol_count": int(len(available_symbols)),
        "eligible_symbol_count": int(len(eligible_symbols)),
        "eligible_symbols": eligible_symbols,
        "available_eligible_symbol_count": int(len(available_eligible_symbols)),
        "available_eligible_symbols": available_eligible_symbols,
        "missing_eligible_symbol_count": int(len(missing_eligible_symbols)),
        "missing_eligible_symbols": missing_eligible_symbols,
        "eligible_symbol_coverage": eligible_symbol_coverage,
        "excluded_symbols": excluded_symbols,
        "excluded_reasons": excluded_reasons,
        "selection_reason": reason
        or (
            "frozen_eligible_symbol_reference"
            if eligible_symbols_override is not None
            else ("symbol_cap_subsample" if excluded_symbols else "all_available_symbols")
        ),
    }


def _standalone_market_state_universe_contract(
    *,
    train_fs_report: dict[str, Any],
    eval_fs_report: dict[str, Any],
    train_source_report: dict[str, Any],
    eval_source_report: dict[str, Any],
) -> dict[str, Any]:
    split_inputs = {
        "train": (dict(train_fs_report or {}), dict(train_source_report or {})),
        "eval": (dict(eval_fs_report or {}), dict(eval_source_report or {})),
    }
    split_contracts: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for split, (fs_report, source_report) in split_inputs.items():
        universe = dict(fs_report.get("universe_contract") or {})
        eligible_symbols = [str(symbol) for symbol in universe.get("eligible_symbols") or []]
        excluded_symbols = [str(symbol) for symbol in universe.get("excluded_symbols") or []]
        excluded_reasons = {
            str(symbol): str(reason)
            for symbol, reason in dict(universe.get("excluded_reasons") or {}).items()
        }
        row = {
            "split": split,
            "source": source_report.get("source"),
            "production_safe": bool(source_report.get("production_safe") is True),
            "candidate_fallback_enabled": bool(source_report.get("allow_candidate_fallback") is True),
            "strategy_independent": True,
            "candidate_independent": source_report.get("source") == "feature_store_market_aggregates"
            and not bool(source_report.get("allow_candidate_fallback") is True),
            "actual_order_book_features_allowed": False,
            "universe_definition_version": universe.get("universe_definition_version"),
            "universe_source": universe.get("source"),
            "feature_dir": universe.get("feature_dir"),
            "minimum_history": universe.get("minimum_history"),
            "minimum_volume": universe.get("minimum_volume"),
            "oi_coverage_requirements": universe.get("oi_coverage_requirements"),
            "funding_coverage_requirements": universe.get("funding_coverage_requirements"),
            "symbol_cap": universe.get("symbol_cap"),
            "available_symbol_count": universe.get("available_symbol_count"),
            "eligible_symbol_count": universe.get("eligible_symbol_count"),
            "eligible_symbols": eligible_symbols,
            "excluded_symbols": excluded_symbols,
            "excluded_symbols_and_reasons": excluded_reasons,
            "selection_reason": universe.get("selection_reason"),
            "feature_store_timestamp_coverage": fs_report.get("timestamp_coverage"),
            "feature_store_symbols_read": fs_report.get("symbols_read"),
        }
        split_contracts[split] = row
        if row["source"] != "feature_store_market_aggregates":
            failures.append(f"{split}_source_not_feature_store_market_aggregates")
        if row["production_safe"] is not True:
            failures.append(f"{split}_not_production_safe")
        if row["candidate_fallback_enabled"] is not False:
            failures.append(f"{split}_candidate_fallback_enabled")
        if row["candidate_independent"] is not True:
            failures.append(f"{split}_not_candidate_independent")
        if not eligible_symbols:
            failures.append(f"{split}_missing_eligible_symbols")
        if len(set(eligible_symbols)) != len(eligible_symbols):
            failures.append(f"{split}_duplicate_eligible_symbols")
        if row["eligible_symbol_count"] is None or int(row["eligible_symbol_count"]) != len(eligible_symbols):
            failures.append(f"{split}_eligible_symbol_count_mismatch")
        if row["available_symbol_count"] is None or int(row["available_symbol_count"]) < len(eligible_symbols):
            failures.append(f"{split}_available_symbol_count_lt_eligible")
        if any(symbol not in excluded_reasons for symbol in excluded_symbols):
            failures.append(f"{split}_excluded_symbols_missing_reasons")
    eligible_sets = {
        tuple(row.get("eligible_symbols") or [])
        for row in split_contracts.values()
        if isinstance(row.get("eligible_symbols"), list)
    }
    if len(eligible_sets) > 1:
        failures.append("eligible_symbol_list_not_constant_across_train_eval")
    common_eligible_symbols = list(next(iter(eligible_sets))) if len(eligible_sets) == 1 else []
    excluded_union: dict[str, str] = {}
    for row in split_contracts.values():
        excluded_union.update(
            {str(symbol): str(reason) for symbol, reason in dict(row.get("excluded_symbols_and_reasons") or {}).items()}
        )

    def _unique(key: str) -> list[Any]:
        values = [row.get(key) for row in split_contracts.values() if row.get(key) is not None]
        return sorted(set(values))

    return {
        "contract_version": "market_state_universe_contract_v1",
        "generated_by": "run_market_state_threshold_controller",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_source": "feature_store_market_aggregates",
        "universe_definition_versions": _unique("universe_definition_version"),
        "strategy_independent": True,
        "candidate_independent": bool(split_contracts)
        and all(row.get("candidate_independent") is True for row in split_contracts.values()),
        "actual_order_book_features_allowed": False,
        "candidate_population_fallback_enabled": any(
            bool(row.get("candidate_fallback_enabled")) for row in split_contracts.values()
        ),
        "feature_dirs": _unique("feature_dir"),
        "symbol_caps": [int(value) for value in _unique("symbol_cap")],
        "available_symbol_counts": [int(value) for value in _unique("available_symbol_count")],
        "eligible_symbol_counts": [int(value) for value in _unique("eligible_symbol_count")],
        "eligible_symbols": common_eligible_symbols,
        "eligible_symbol_count": len(common_eligible_symbols),
        "minimum_history": _unique("minimum_history"),
        "minimum_volume": _unique("minimum_volume"),
        "oi_coverage_requirements": _unique("oi_coverage_requirements"),
        "funding_coverage_requirements": _unique("funding_coverage_requirements"),
        "excluded_symbols_and_reasons": dict(sorted(excluded_union.items())),
        "fold_split_contracts": split_contracts,
        "validation": {
            "passed": not failures,
            "failures": failures,
            "fold_split_count": int(len(split_contracts)),
            "eligible_symbol_list_constant": len(eligible_sets) == 1,
        },
    }


def _feature_store_oi_weight_column(columns: list[str]) -> str | None:
    candidates = [
        "oi_value",
        "open_interest_value",
        "oi_value_usd",
        "oi_value_log",
        "open_interest",
    ]
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    for col in columns:
        lower = col.lower()
        if "oi_value" in lower and "chg" not in lower and "rank" not in lower and "pct" not in lower:
            return col
    return None


def _feature_store_weight_values(values: pd.Series, *, weight_col: str) -> pd.Series:
    weights = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if weight_col.lower().endswith("_log") or "log" in weight_col.lower():
        weights = np.exp(weights.clip(lower=-20.0, upper=20.0))
    weights = weights.where(weights > 0.0)
    return weights.astype("float64")


def _safe_share(series: pd.Series, predicate: Any) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan
    return float(predicate(values).mean())


def _symbol_value(group: pd.DataFrame, *, col: str, symbol_prefix: str) -> float:
    symbols = group["symbol"].astype(str).str.upper()
    mask = symbols.str.startswith(symbol_prefix)
    values = pd.to_numeric(group.loc[mask, col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[-1])


def _weighted_mean(group: pd.DataFrame) -> float:
    values = pd.to_numeric(group["_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    weights = pd.to_numeric(group["_weight"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = values.notna() & weights.notna() & (weights > 0.0)
    if not bool(mask.any()):
        return np.nan
    weight_sum = float(weights.loc[mask].sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.nan
    return float((values.loc[mask] * weights.loc[mask]).sum() / weight_sum)


def _feature_store_timestamp_aggregates(
    feature_dir: Path | None,
    timestamps: pd.Series,
    columns: list[str],
    *,
    symbol_cap: int,
    tail_reference_quantiles: dict[str, dict[str, float]] | None = None,
    eligible_symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    base = pd.DataFrame({"timestamp": ts})
    requested_columns = [str(col) for col in columns]
    columns = _filter_market_state_source_columns(requested_columns)
    rejected_columns = [col for col in requested_columns if col not in set(columns)]
    if feature_dir is None or not feature_dir.exists() or not columns:
        universe_contract = _feature_store_universe_contract(
            feature_dir,
            all_paths=[],
            selected_paths=[],
            symbol_cap=int(symbol_cap),
            reason="missing_dir_or_columns",
            eligible_symbols_override=eligible_symbols,
            missing_eligible_symbols=eligible_symbols,
        )
        return base, {
            "feature_dir": str(feature_dir) if feature_dir is not None else None,
            "enabled": False,
            "reason": "missing_dir_or_columns",
            "columns": [],
            "rejected_columns": rejected_columns,
            "universe_contract": universe_contract,
        }
    path_map = _feature_store_symbol_path_map(feature_dir)
    all_paths = [path_map[symbol] for symbol in sorted(path_map)]
    if eligible_symbols is not None:
        frozen_symbols = [str(symbol) for symbol in eligible_symbols]
        paths = [path_map[symbol] for symbol in frozen_symbols if symbol in path_map]
        missing_eligible_symbols = [symbol for symbol in frozen_symbols if symbol not in path_map]
    else:
        frozen_symbols = []
        paths = list(all_paths)
        missing_eligible_symbols = []
    if eligible_symbols is None and symbol_cap > 0 and len(paths) > symbol_cap:
        idx = np.linspace(0, len(paths) - 1, int(symbol_cap)).round().astype(int)
        paths = [paths[i] for i in np.unique(idx)]
    universe_contract = _feature_store_universe_contract(
        feature_dir,
        all_paths=all_paths,
        selected_paths=paths,
        symbol_cap=int(symbol_cap),
        eligible_symbols_override=frozen_symbols if eligible_symbols is not None else None,
        missing_eligible_symbols=missing_eligible_symbols,
    )
    ts_set = set(ts)
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=columns)
        except Exception:
            try:
                df = pd.read_parquet(path)
                keep = [col for col in columns if col in df.columns]
                df = df[keep]
            except Exception:
                continue
        if isinstance(df.index, pd.DatetimeIndex):
            stamp = pd.to_datetime(df.index, utc=True, errors="coerce")
        elif "timestamp" in df.columns:
            stamp = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        elif "ts" in df.columns:
            stamp = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        else:
            continue
        mask = pd.Series(stamp, index=df.index).isin(ts_set).to_numpy()
        if not mask.any():
            continue
        keep_cols = [col for col in columns if col in df.columns]
        if not keep_cols:
            continue
        part = df.loc[mask, keep_cols].copy()
        part.insert(0, "timestamp", pd.Series(stamp, index=df.index).loc[mask].to_numpy())
        part.insert(1, "symbol", _feature_store_symbol_from_path(path))
        for col in keep_cols:
            part[col] = pd.to_numeric(part[col], errors="coerce").astype("float32")
        frames.append(part)
    if not frames:
        return base, {
            "feature_dir": str(feature_dir),
            "enabled": True,
            "columns": columns,
            "rejected_columns": rejected_columns,
            "universe_contract": universe_contract,
            "symbols_read": len(paths),
            "eligible_symbol_denominator": int(
                len(frozen_symbols) if eligible_symbols is not None else len(paths)
            ),
            "rows_loaded": 0,
            "timestamp_coverage": 0.0,
        }
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    agg_frames: list[pd.DataFrame] = []
    present_cols = [col for col in columns if col in panel.columns]
    reference_quantiles: dict[str, dict[str, float]] = {}
    if tail_reference_quantiles:
        reference_quantiles.update(
            {
                str(col): {"q10": float(ref.get("q10", np.nan)), "q90": float(ref.get("q90", np.nan))}
                for col, ref in tail_reference_quantiles.items()
                if isinstance(ref, dict)
            }
        )
    for col in present_cols:
        if col in reference_quantiles:
            continue
        ref_values = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if ref_values.empty:
            continue
        reference_quantiles[col] = {
            "q10": float(ref_values.quantile(0.10)),
            "q90": float(ref_values.quantile(0.90)),
        }
    oi_weight_col = _feature_store_oi_weight_column(present_cols)
    for col in present_cols:
        vals = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() == 0:
            continue
        tmp = panel[["timestamp", "symbol"]].copy()
        tmp[col] = vals
        grouped = tmp.groupby("timestamp")[col]
        agg = grouped.agg(["mean", "std", "median", "count"])
        agg = agg.rename(columns={"count": "finite_count"})
        q = grouped.quantile([0.10, 0.25, 0.75, 0.90]).unstack()
        q.columns = ["p10", "p25", "p75", "p90"]
        out = pd.concat([agg, q], axis=1)
        out["iqr"] = out["p75"] - out["p25"]
        out["robust_dispersion"] = out["iqr"] / 1.349
        symbol_denominator = len(frozen_symbols) if eligible_symbols is not None else len(paths)
        out["finite_share"] = out["finite_count"] / max(1, symbol_denominator)
        finite_vals = tmp[col].where(tmp[col].notna())
        out["share_pos"] = finite_vals.gt(0.0).where(finite_vals.notna()).groupby(tmp["timestamp"]).mean()
        out["share_neg"] = finite_vals.lt(0.0).where(finite_vals.notna()).groupby(tmp["timestamp"]).mean()
        ref = reference_quantiles.get(col, {})
        q10_ref = float(ref.get("q10", np.nan))
        q90_ref = float(ref.get("q90", np.nan))
        if np.isfinite(q90_ref):
            out["share_gt_train_q90"] = (
                finite_vals.gt(q90_ref).where(finite_vals.notna()).groupby(tmp["timestamp"]).mean()
            )
        else:
            out["share_gt_train_q90"] = np.nan
        if np.isfinite(q10_ref):
            out["share_lt_train_q10"] = (
                finite_vals.lt(q10_ref).where(finite_vals.notna()).groupby(tmp["timestamp"]).mean()
            )
        else:
            out["share_lt_train_q10"] = np.nan
        symbols_upper = tmp["symbol"].astype(str).str.upper()
        for prefix, stat_name in (("BTC", "btc_value"), ("ETH", "eth_value")):
            mask = symbols_upper.str.startswith(prefix) & tmp[col].notna()
            if bool(mask.any()):
                out[stat_name] = tmp.loc[mask].groupby("timestamp")[col].last()
            else:
                out[stat_name] = np.nan
        if oi_weight_col and oi_weight_col in panel.columns and col != oi_weight_col:
            weighted = panel[["timestamp", col, oi_weight_col]].copy()
            weighted["_value"] = pd.to_numeric(weighted[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            weighted["_weight"] = _feature_store_weight_values(weighted[oi_weight_col], weight_col=oi_weight_col)
            valid = weighted["_value"].notna() & weighted["_weight"].notna() & (weighted["_weight"] > 0.0)
            weighted["_weighted_value"] = np.where(
                valid,
                weighted["_value"].to_numpy(dtype=float) * weighted["_weight"].to_numpy(dtype=float),
                np.nan,
            )
            value_sum = weighted["_weighted_value"].groupby(weighted["timestamp"]).sum(min_count=1)
            weight_sum = weighted["_weight"].where(valid).groupby(weighted["timestamp"]).sum(min_count=1)
            out["oi_weighted_mean"] = value_sum / weight_sum.where(weight_sum > 0.0)
        out.columns = [f"fs__{col}__{stat}" for stat in out.columns]
        agg_frames.append(out.astype("float32"))
    if not agg_frames:
        merged = base
    else:
        merged = base.merge(pd.concat(agg_frames, axis=1).reset_index(), on="timestamp", how="left")
    coverage = float(merged.drop(columns=["timestamp"], errors="ignore").notna().any(axis=1).mean()) if len(merged) else 0.0
    return merged, {
        "feature_dir": str(feature_dir),
        "enabled": True,
        "columns": present_cols,
        "rejected_columns": rejected_columns,
        "universe_contract": universe_contract,
        "symbols_read": len(paths),
        "eligible_symbol_denominator": int(
            len(frozen_symbols) if eligible_symbols is not None else len(paths)
        ),
        "rows_loaded": int(len(panel)),
        "timestamp_coverage": coverage,
        "feature_count": int(max(0, merged.shape[1] - 1)),
        "aggregation_contract": "median,p10,p90,iqr,finite_coverage,breadth,basis_assets,train_reference_tail_shares",
        "tail_reference_source": "provided_train_reference" if tail_reference_quantiles else "self_window_reference",
        "tail_reference_quantiles": reference_quantiles,
        "oi_weight_column": oi_weight_col,
    }


def _feature_store_timestamp_aggregate_pair(
    train_feature_dir: Path | None,
    eval_feature_dir: Path | None,
    train_timestamps: pd.Series,
    eval_timestamps: pd.Series,
    columns: list[str],
    *,
    symbol_cap: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    eligible_symbols = _select_feature_store_symbols(
        train_feature_dir,
        eval_feature_dir,
        symbol_cap=int(symbol_cap),
    )
    train_fs, train_report = _feature_store_timestamp_aggregates(
        train_feature_dir,
        train_timestamps,
        columns,
        symbol_cap=int(symbol_cap),
        eligible_symbols=eligible_symbols,
    )
    eval_fs, eval_report = _feature_store_timestamp_aggregates(
        eval_feature_dir,
        eval_timestamps,
        columns,
        symbol_cap=int(symbol_cap),
        tail_reference_quantiles=dict(train_report.get("tail_reference_quantiles") or {}),
        eligible_symbols=eligible_symbols,
    )
    train_report["tail_reference_role"] = "fit_on_training_timestamps"
    eval_report["tail_reference_role"] = "transformed_with_training_timestamp_reference"
    train_report["frozen_eligible_symbol_source"] = "common_train_eval_feature_store_symbols"
    eval_report["frozen_eligible_symbol_source"] = "common_train_eval_feature_store_symbols"
    return train_fs, train_report, eval_fs, eval_report


def _merge_timestamp_features(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right.empty or list(right.columns) == ["timestamp"]:
        return left
    out = left.merge(right, on="timestamp", how="left", validate="one_to_one")
    numeric = [c for c in out.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(out[c])]
    if numeric:
        out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
        out[numeric] = out[numeric].fillna(out[numeric].median(numeric_only=True)).fillna(0.0)
    return out


def _state_source_aggregate_frame(
    candidate_agg: pd.DataFrame,
    feature_store_agg: pd.DataFrame,
    *,
    allow_candidate_fallback: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the timestamp-level source frame used by the market-state encoder.

    The market-state encoder should describe market geometry, not the current
    candidate population.  Feature-store aggregates are therefore the default
    source.  Candidate aggregates are allowed only as an explicit fallback for
    smoke tests or legacy diagnostics.
    """

    base = candidate_agg[["timestamp"]].copy()
    fs_cols = [c for c in feature_store_agg.columns if c != "timestamp"]
    if fs_cols:
        out = _merge_timestamp_features(base, feature_store_agg)
        source = "feature_store_market_aggregates"
        production_safe = True
    elif allow_candidate_fallback:
        clean_cols = ["timestamp", *_filter_market_state_source_columns([c for c in candidate_agg.columns if c != "timestamp"])]
        out = candidate_agg.loc[:, clean_cols].copy()
        source = "debug_candidate_population_fallback_sanitized"
        production_safe = False
    else:
        out = base
        source = "timestamp_only_no_market_features"
        production_safe = True
    validation = _validate_market_state_source_frame(out, context=source)
    numeric = [c for c in out.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(out[c])]
    return out, {
        "source": source,
        "feature_count": int(len(numeric)),
        "allow_candidate_fallback": bool(allow_candidate_fallback),
        "production_safe": bool(production_safe),
        "validation": validation,
        "forbidden_candidate_aggregate_columns_removed": sorted(
            [
                str(col)
                for col in candidate_agg.columns
                if col != "timestamp" and _market_state_column_violation(str(col)) is not None
            ]
        ),
        "candidate_aggregate_feature_count": int(
            len([c for c in candidate_agg.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(candidate_agg[c])])
        ),
        "feature_store_aggregate_feature_count": int(len(fs_cols)),
        "contract": (
            "Market-state observed axes, forecast heads and latent probabilities "
            "use feature-store market aggregates by default. Candidate-ledger "
            "population aggregates are not used unless allow_candidate_fallback is true; "
            "that fallback is debug-only and sanitized before state encoding."
        ),
    }


def _axis_columns(agg_cols: list[str], keywords: tuple[str, ...]) -> list[str]:
    hits = []
    for col in agg_cols:
        lower = col.lower()
        if any(token in lower for token in keywords):
            hits.append(col)
    return hits


def _prefer_feature_store_axis_columns(
    axis: str,
    cols: list[str],
) -> list[str]:
    """Prefer market-wide feature-store aggregates for global state axes.

    Candidate-population aggregates are useful fallback inputs, but the market
    encoder should not change meaning when the deployable candidate universe or
    disabled heads change. Feature-store aggregates are timestamp-level market
    summaries and are therefore the primary source for observed market axes.
    """

    feature_store_cols = [col for col in cols if col.startswith("fs__")]
    if axis == "state_realized_vol" and feature_store_cols:
        preferred = [
            col
            for col in feature_store_cols
            if any(
                token in col.lower()
                for token in (
                    "realized_volatility",
                    "rv_",
                    "volatility",
                    "atr_percentile",
                    "atr_change",
                    "true_range",
                    "range_expansion",
                )
            )
            and "ema" not in col.lower()
        ]
        if preferred:
            return preferred
    if axis == "state_liquidity_stress_proxy" and feature_store_cols:
        preferred = [
            col
            for col in feature_store_cols
            if any(
                token in col.lower()
                for token in (
                    "liquidity",
                    "amihud",
                    "spread_proxy",
                    "range_to_volume",
                    "volume",
                )
            )
        ]
        if preferred:
            return preferred
    return feature_store_cols if feature_store_cols else cols


def _fit_robust_z_reference(train: pd.Series) -> dict[str, float]:
    train = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = float(train.median()) if train.notna().any() else 0.0
    iqr = float(train.quantile(0.75) - train.quantile(0.25)) if train.notna().sum() > 3 else 0.0
    scale = iqr / 1.349 if iqr > 1e-12 else float(train.std(ddof=0) or 1.0)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    q05 = float(train.quantile(0.05)) if train.notna().sum() > 3 else med
    q95 = float(train.quantile(0.95)) if train.notna().sum() > 3 else med
    return {"median": med, "scale": float(scale), "q05": q05, "q95": q95}


def _apply_robust_z_reference(values: pd.Series, reference: dict[str, float]) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = float(reference.get("median", 0.0))
    scale = float(reference.get("scale", 1.0))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return ((values - med) / scale).clip(-6.0, 6.0)


def _bounded_sigmoid(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, -12.0, 12.0)
    return (1.0 / (1.0 + np.exp(-arr))).astype(float)


def _fit_observed_reliability_reference(
    train_agg: pd.DataFrame,
    numeric_cols: list[str],
    column_refs: dict[str, dict[str, float]],
) -> dict[str, Any]:
    cols = [
        c
        for c in numeric_cols
        if c.startswith("fs__") and _market_state_column_violation(str(c)) is None
    ]
    if not cols:
        cols = [c for c in numeric_cols if _market_state_column_violation(str(c)) is None]
    cols = cols[:128]
    ref: dict[str, Any] = {
        "mode": "observed_reliability_train_reference_v1",
        "columns": cols,
        "rolling_window": 24,
        "novelty_reference": None,
        "drift_reference": None,
    }
    if not cols:
        ref["reason"] = "no_numeric_market_state_columns"
        return ref
    z_parts = [
        _apply_robust_z_reference(train_agg[col], column_refs.get(col, {}))
        for col in cols
    ]
    z_frame = pd.concat(z_parts, axis=1)
    novelty_raw = np.sqrt(np.nanmean(np.square(z_frame.to_numpy(dtype=float)), axis=1))
    novelty_raw = pd.Series(novelty_raw, index=train_agg.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    drift_raw = (
        z_frame.abs()
        .rolling(window=min(24, max(1, len(z_frame))), min_periods=1)
        .mean()
        .mean(axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    ref["novelty_reference"] = _fit_robust_z_reference(novelty_raw)
    ref["drift_reference"] = _fit_robust_z_reference(drift_raw)
    ref["train_novelty_median"] = float(novelty_raw.median())
    ref["train_novelty_q95"] = float(novelty_raw.quantile(0.95))
    ref["train_drift_median"] = float(drift_raw.median())
    ref["train_drift_q95"] = float(drift_raw.quantile(0.95))
    return ref


def _score_observed_reliability_channels(
    agg: pd.DataFrame,
    reliability_ref: dict[str, Any],
    column_refs: dict[str, dict[str, float]],
) -> pd.DataFrame:
    cols = list(reliability_ref.get("columns") or [])
    out = pd.DataFrame(index=agg.index)
    if not cols:
        out["state_input_coverage"] = 0.0
        out["state_extreme_value_share"] = 0.0
        out["state_novelty"] = 0.0
        out["state_drift_score"] = 0.0
        out["state_uncertainty"] = 1.0
        return out

    raw_parts: list[pd.Series] = []
    z_parts: list[pd.Series] = []
    extreme_parts: list[pd.Series] = []
    for col in cols:
        vals = (
            pd.to_numeric(agg[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if col in agg.columns
            else pd.Series(np.nan, index=agg.index, dtype=float)
        )
        raw_parts.append(vals)
        ref = column_refs.get(col, {})
        z_parts.append(_apply_robust_z_reference(vals, ref))
        q05 = float(ref.get("q05", np.nan))
        q95 = float(ref.get("q95", np.nan))
        if np.isfinite(q05) and np.isfinite(q95) and q95 > q05:
            extreme_parts.append(((vals < q05) | (vals > q95)).where(vals.notna()).astype(float))
        else:
            extreme_parts.append(pd.Series(np.nan, index=agg.index, dtype=float))

    raw_frame = pd.concat(raw_parts, axis=1)
    z_frame = pd.concat(z_parts, axis=1)
    extreme_frame = pd.concat(extreme_parts, axis=1)
    coverage = raw_frame.notna().mean(axis=1).astype(float)
    extreme_share = extreme_frame.mean(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    novelty_raw = np.sqrt(np.nanmean(np.square(z_frame.to_numpy(dtype=float)), axis=1))
    novelty_raw = pd.Series(novelty_raw, index=agg.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    drift_window = max(1, int(reliability_ref.get("rolling_window", 24) or 24))
    drift_raw = (
        z_frame.abs()
        .rolling(window=drift_window, min_periods=1)
        .mean()
        .mean(axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    novelty = _bounded_sigmoid(
        _apply_robust_z_reference(
            novelty_raw,
            dict(reliability_ref.get("novelty_reference") or {}),
        )
    )
    drift = _bounded_sigmoid(
        _apply_robust_z_reference(
            drift_raw,
            dict(reliability_ref.get("drift_reference") or {}),
        )
    )
    coverage_arr = coverage.fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    extreme_arr = extreme_share.to_numpy(dtype=float)
    uncertainty = np.clip(
        0.35 * novelty + 0.30 * drift + 0.20 * (1.0 - coverage_arr) + 0.15 * extreme_arr,
        0.0,
        1.0,
    )
    out["state_input_coverage"] = coverage_arr
    out["state_extreme_value_share"] = extreme_arr
    out["state_novelty"] = novelty
    out["state_drift_score"] = drift
    out["state_uncertainty"] = uncertainty
    return out


def fit_observed_axis_encoder(
    train_agg: pd.DataFrame,
    eval_agg: pd.DataFrame | None = None,
    *,
    minimum_input_coverage: float = 0.80,
) -> dict[str, Any]:
    """Fit train-only scaling/reference state for observed market axes.

    The ablation path historically called ``build_observed_axes(train, eval)``
    directly.  A deployable controller needs the same logic frozen as an
    artifact, so future/live rows can be transformed without refitting or
    re-reading the training candidate population.
    """

    train_validation = _validate_market_state_source_frame(train_agg, context="train_observed_axis_encoder")
    eval_validation = (
        _validate_market_state_source_frame(eval_agg, context="eval_observed_axis_encoder")
        if eval_agg is not None
        else None
    )
    axis_sources: dict[str, list[str]] = {}
    numeric_cols = [
        c
        for c in train_agg.columns
        if c != "timestamp"
        and (eval_agg is None or c in eval_agg.columns)
        and pd.api.types.is_numeric_dtype(train_agg[c])
    ]
    column_refs = {
        col: _fit_robust_z_reference(train_agg[col])
        for col in numeric_cols
    }
    axes: dict[str, list[str]] = {}
    for axis, keywords in AXIS_KEYWORDS.items():
        if axis == "state_transition":
            # Transition is a cross-feature displacement score, not a simple
            # keyword average. Build it explicitly below after the base axes.
            continue
        cols = _axis_columns(numeric_cols, keywords)
        if not cols and axis == "state_shock":
            cols = [c for c in numeric_cols if "uncert" in c.lower() or "gap" in c.lower()]
        if not cols and axis == "state_realized_vol":
            cols = [c for c in numeric_cols if "barrier" in c.lower()]
        if not cols:
            axes[axis] = []
            axis_sources[axis] = []
            continue
        cols = _prefer_feature_store_axis_columns(axis, cols)[:24]
        axes[axis] = cols
        axis_sources[axis] = cols
    ret_col = _pick_numeric_col(
        train_agg,
        [
            ("fs__mkt_ret_eq_1h",),
            ("mkt_ret_eq_1h",),
            ("fs__mkt_ret_eq_4h",),
            ("mkt_ret_eq_4h",),
            ("market", "ret", "1h"),
        ],
    )
    if ret_col and (eval_agg is None or ret_col in eval_agg.columns):
        axis_sources["state_shock_up"] = [ret_col, "positive_part_robust_z"]
        axis_sources["state_shock_down"] = [ret_col, "negative_part_robust_z"]
    else:
        ret_col = None
        axis_sources["state_shock_up"] = []
        axis_sources["state_shock_down"] = []
    axis_sources["state_consolidation"] = ["state_compression", "abs_state_trend"]
    axis_sources["state_vol_expansion"] = ["state_realized_vol", "state_compression"]
    transition_cols = [
        c
        for c in numeric_cols
        if c.startswith("fs__")
        and not any(token in c.lower() for token in ("candidate", "rank", "score"))
    ][:96]
    transition_ref: dict[str, Any] = {"columns": transition_cols, "disp_reference": None}
    if transition_cols:
        train_z = []
        for col in transition_cols:
            train_z.append(_apply_robust_z_reference(train_agg[col], column_refs[col]))
        train_z_frame = pd.concat(train_z, axis=1)
        train_disp = train_z_frame.diff().abs().mean(axis=1).fillna(0.0)
        transition_ref["disp_reference"] = _fit_robust_z_reference(train_disp)
        axis_sources["state_transition"] = transition_cols + ["mean_abs_feature_z_diff"]
    else:
        axis_sources["state_transition"] = []
    reliability_ref = _fit_observed_reliability_reference(train_agg, numeric_cols, column_refs)
    reliability_cols = list(reliability_ref.get("columns") or [])
    spectral_cols = [
        c
        for c in numeric_cols
        if c.startswith("fs__")
        and not any(token in c.lower() for token in ("candidate", "rank", "score"))
    ]
    spectral_encoder = fit_market_spectral_position_encoder(
        train_agg,
        timestamp_col="timestamp",
        feature_columns=spectral_cols,
        config=MarketSpectralPositionConfig(
            lookback=48,
            min_periods=24,
            top_k=3,
            max_features=64,
            shrinkage=0.10,
            prefix="state_spectral_",
        ),
    )
    spectral_state_cols = list(SPECTRAL_POSITION_STATE_COLUMNS)
    for col in spectral_state_cols:
        axis_sources[col] = list(spectral_encoder.get("feature_columns") or []) + [
            "rolling_covariance_shift_1",
            "eigenvector_sign_aligned",
        ]
    axis_sources["state_input_coverage"] = reliability_cols
    axis_sources["state_extreme_value_share"] = reliability_cols + ["train_q05_q95_bounds"]
    axis_sources["state_novelty"] = reliability_cols + ["train_diagonal_robust_distance"]
    axis_sources["state_drift_score"] = reliability_cols + ["causal_rolling_abs_robust_z"]
    axis_sources["state_uncertainty"] = [
        "state_novelty",
        "state_drift_score",
        "state_input_coverage",
        "state_extreme_value_share",
    ]
    min_cov = float(minimum_input_coverage)
    if not np.isfinite(min_cov):
        min_cov = 0.80
    min_cov = float(np.clip(min_cov, 0.0, 1.0))
    axis_sources["state_low_input_coverage"] = [
        "state_input_coverage",
        f"minimum_input_coverage={min_cov:.4f}",
    ]
    return {
        "mode": "observed_axis_robust_z_v1",
        "axes": axes,
        "column_refs": column_refs,
        "ret_col": ret_col,
        "transition": transition_ref,
        "reliability": reliability_ref,
        "spectral_position": spectral_encoder,
        "minimum_input_coverage": min_cov,
        "axis_sources": axis_sources,
        "source_validation": {
            "train": train_validation,
            "eval": eval_validation,
        },
        "fit_rows": int(len(train_agg)),
        "fit_timestamp_min": train_agg["timestamp"].min() if "timestamp" in train_agg else None,
        "fit_timestamp_max": train_agg["timestamp"].max() if "timestamp" in train_agg else None,
        "contract": "train-only robust scaling for continuous overlapping observed market-state axes",
    }


def transform_observed_axes(
    agg: pd.DataFrame,
    encoder: dict[str, Any],
) -> pd.DataFrame:
    state = agg[["timestamp"]].copy()
    column_refs: dict[str, dict[str, float]] = encoder.get("column_refs", {})
    for axis, cols in dict(encoder.get("axes", {})).items():
        cols = list(cols or [])
        if not cols:
            state[axis] = 0.0
            continue
        z_parts = []
        for col in cols:
            vals = agg[col] if col in agg.columns else pd.Series(np.nan, index=agg.index, dtype=float)
            z_parts.append(_apply_robust_z_reference(vals, column_refs.get(col, {})))
        state[axis] = pd.concat(z_parts, axis=1).mean(axis=1).astype(float)
    ret_col = encoder.get("ret_col")
    if ret_col:
        vals = agg[ret_col] if ret_col in agg.columns else pd.Series(np.nan, index=agg.index, dtype=float)
        ret_z = _apply_robust_z_reference(vals, column_refs.get(ret_col, {})).astype(float)
        state["state_shock_up"] = np.maximum(ret_z, 0.0)
        state["state_shock_down"] = np.maximum(-ret_z, 0.0)
    else:
        state["state_shock_up"] = 0.0
        state["state_shock_down"] = 0.0
    state["state_consolidation"] = (
        state.get("state_compression", 0.0) - state.get("state_trend", 0.0).abs()
    ).clip(-6.0, 6.0)
    state["state_vol_expansion"] = (
        state.get("state_realized_vol", 0.0) - state.get("state_compression", 0.0)
    ).clip(-6.0, 6.0)
    transition = dict(encoder.get("transition", {}))
    transition_cols = list(transition.get("columns") or [])
    if transition_cols:
        z_parts = []
        for col in transition_cols:
            vals = agg[col] if col in agg.columns else pd.Series(np.nan, index=agg.index, dtype=float)
            z_parts.append(_apply_robust_z_reference(vals, column_refs.get(col, {})))
        z_frame = pd.concat(z_parts, axis=1)
        disp = z_frame.diff().abs().mean(axis=1).fillna(0.0)
        state["state_transition"] = _apply_robust_z_reference(
            disp,
            transition.get("disp_reference") or {},
        ).astype(float)
    else:
        state["state_transition"] = 0.0
    spectral_encoder = encoder.get("spectral_position")
    if isinstance(spectral_encoder, dict):
        spectral = transform_market_spectral_position(agg, spectral_encoder)
        spectral_cols = [c for c in spectral.columns if c != "timestamp"]
        if spectral_cols:
            state = state.merge(spectral, on="timestamp", how="left", validate="one_to_one")
            for col in spectral_cols:
                state[col] = pd.to_numeric(state[col], errors="coerce").fillna(0.0).astype(float)
    # Explicit transition pressure from timestamp-to-timestamp state displacement.
    axis_cols = [c for c in state.columns if c.startswith("state_")]
    disp = state[axis_cols].diff().abs().mean(axis=1).fillna(0.0)
    state["state_transition_pressure"] = disp
    reliability = _score_observed_reliability_channels(
        agg,
        dict(encoder.get("reliability", {})),
        column_refs,
    )
    for col in reliability.columns:
        state[col] = reliability[col].to_numpy(dtype=float)
    min_input_coverage = float(encoder.get("minimum_input_coverage", 0.80) or 0.80)
    if not np.isfinite(min_input_coverage):
        min_input_coverage = 0.80
    min_input_coverage = float(np.clip(min_input_coverage, 0.0, 1.0))
    coverage_series = (
        pd.to_numeric(state["state_input_coverage"], errors="coerce")
        if "state_input_coverage" in state.columns
        else pd.Series(0.0, index=state.index, dtype=float)
    )
    low_input_coverage = coverage_series.fillna(0.0).lt(min_input_coverage)
    state["state_low_input_coverage"] = low_input_coverage.astype(float).to_numpy(dtype=float)
    if low_input_coverage.any():
        mechanism_cols = [
            c
            for c in state.columns
            if c != "timestamp"
            and c.startswith("state_")
            and c not in OBSERVED_RELIABILITY_STATE_COLUMNS
        ]
        if mechanism_cols:
            state.loc[low_input_coverage, mechanism_cols] = 0.0
        state.loc[low_input_coverage, "state_uncertainty"] = 1.0
    return state


def build_observed_axes(
    train_agg: pd.DataFrame,
    eval_agg: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    encoder = fit_observed_axis_encoder(train_agg, eval_agg)
    train_state = transform_observed_axes(train_agg, encoder)
    eval_state = transform_observed_axes(eval_agg, encoder)
    axis_sources = dict(encoder.get("axis_sources", {}))
    axis_sources["state_transition_pressure"] = ["mean_abs_state_axis_diff"]
    return train_state, eval_state, axis_sources


def _pick_numeric_col(
    frame: pd.DataFrame,
    token_groups: list[tuple[str, ...]],
    *,
    prefer_mean: bool = True,
) -> str | None:
    numeric_cols = [
        c
        for c in frame.columns
        if c != "timestamp" and pd.api.types.is_numeric_dtype(frame[c])
    ]
    for tokens in token_groups:
        hits = [
            c
            for c in numeric_cols
            if all(token in c.lower() for token in tokens)
        ]
        if not hits:
            continue
        if prefer_mean:
            mean_hits = [c for c in hits if "__mean" in c.lower()]
            if mean_hits:
                return mean_hits[0]
            median_hits = [c for c in hits if "__median" in c.lower()]
            if median_hits:
                return median_hits[0]
        return hits[0]
    return None


def _future_value_matrix(values: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    cols = [values.shift(-step).to_numpy(dtype=float) for step in range(1, int(horizon) + 1)]
    matrix = np.column_stack(cols) if cols else np.empty((len(values), 0), dtype=float)
    valid_count = np.isfinite(matrix).sum(axis=1)
    return matrix, valid_count


def _prior_sigma_from_returns(ret: pd.Series, horizon: int) -> pd.Series:
    ret = pd.to_numeric(ret, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    window = max(24, min(168, int(horizon) * 8))
    sigma = ret.rolling(window=window, min_periods=max(8, min(24, window // 4))).std(ddof=0).shift(1)
    fallback = float(ret.std(ddof=0) or 0.0)
    if not np.isfinite(fallback) or fallback <= 1e-10:
        fallback = 1e-4
    return sigma.fillna(fallback).clip(lower=1e-8)


def _empirical_cdf_transform(train_values: pd.Series, values: pd.Series) -> pd.Series:
    train = pd.to_numeric(train_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if train.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    sorted_vals = np.sort(train.to_numpy(dtype=float))
    ranks = np.searchsorted(sorted_vals, vals.to_numpy(dtype=float), side="right")
    out = ranks.astype(float) / max(1, len(sorted_vals))
    out[~np.isfinite(vals.to_numpy(dtype=float))] = np.nan
    return pd.Series(out, index=values.index, dtype=float).clip(0.0, 1.0)


def _empirical_cdf_reference(train_values: pd.Series) -> dict[str, Any]:
    """Persist the training-only empirical CDF used for target normalization."""

    train = pd.to_numeric(train_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if train.empty:
        return {
            "reference_version": "empirical_cdf_reference_v1",
            "n": 0,
            "sorted_values": [],
            "quantiles": {},
        }
    sorted_vals = np.sort(train.to_numpy(dtype=np.float64))
    quantiles = {
        f"q{int(q * 100):02d}": float(np.quantile(sorted_vals, q))
        for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    }
    return {
        "reference_version": "empirical_cdf_reference_v1",
        "n": int(len(sorted_vals)),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "mean": float(np.mean(sorted_vals)),
        "std": float(np.std(sorted_vals)),
        "quantiles": quantiles,
        "sorted_values": sorted_vals.astype(float).tolist(),
    }


def _forecast_validation_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    timestamp: pd.Series | None = None,
) -> dict[str, Any]:
    truth = pd.to_numeric(y_true, errors="coerce").replace([np.inf, -np.inf], np.nan)
    pred = pd.Series(np.asarray(y_pred, dtype=float), index=truth.index).replace([np.inf, -np.inf], np.nan)
    valid = truth.notna() & pred.notna()
    if int(valid.sum()) == 0:
        return {"validation_rows": 0}
    truth = truth.loc[valid].clip(0.0, 1.0)
    pred = pred.loc[valid].clip(0.0, 1.0)
    err = pred - truth
    hard = truth >= 0.90
    pred_hard = pred >= 0.90
    true_pos = int(hard.sum())
    pred_pos = int(pred_hard.sum())
    tp = int((hard & pred_hard).sum())
    fp = int((~hard & pred_hard).sum())
    fn = int((hard & ~pred_hard).sum())
    tn = int((~hard & ~pred_hard).sum())
    top_n = max(1, int(math.ceil(0.10 * len(pred))))
    top_idx = pred.sort_values(ascending=False).head(top_n).index
    hard_float = hard.astype(float)
    brier = float(np.mean(np.square(pred.to_numpy(dtype=float) - hard_float.to_numpy(dtype=float))))
    if true_pos > 0:
        ordered = hard.loc[pred.sort_values(ascending=False).index].to_numpy(dtype=float)
        cum_tp = np.cumsum(ordered)
        ranks = np.arange(1, len(ordered) + 1, dtype=float)
        precision_at_k = cum_tp / ranks
        average_precision = float((precision_at_k * ordered).sum() / max(true_pos, 1))
    else:
        average_precision = None
    false_alarm_rate = float(fp / max(fp + tn, 1)) if (fp + tn) else None
    miss_rate = float(fn / max(true_pos, 1)) if true_pos else None
    calibration_bins: list[dict[str, Any]] = []
    ece_num = 0.0
    for bin_idx in range(5):
        lo = bin_idx / 5.0
        hi = (bin_idx + 1) / 5.0
        if bin_idx == 4:
            mask = pred.ge(lo) & pred.le(hi)
        else:
            mask = pred.ge(lo) & pred.lt(hi)
        count = int(mask.sum())
        if count == 0:
            continue
        pred_mean = float(pred.loc[mask].mean())
        event_rate = float(hard.loc[mask].mean())
        ece_num += count * abs(pred_mean - event_rate)
        calibration_bins.append(
            {
                "bin": int(bin_idx),
                "lo": float(lo),
                "hi": float(hi),
                "count": count,
                "pred_mean": pred_mean,
                "event_rate": event_rate,
            }
        )
    pred_std = float(pred.std(ddof=0))
    pred_unique = int(pd.Series(np.round(pred.to_numpy(dtype=float), 8)).nunique(dropna=True))
    out: dict[str, Any] = {
        "validation_rows": int(len(truth)),
        "validation_mae": float(err.abs().mean()),
        "validation_rmse": float(np.sqrt(np.mean(np.square(err.to_numpy(dtype=float))))),
        "validation_target_mean": float(truth.mean()),
        "validation_pred_mean": float(pred.mean()),
        "validation_pred_std": pred_std,
        "validation_pred_unique": pred_unique,
        "validation_collapsed": bool(pred_std < 1e-6 or pred_unique <= 2),
        "validation_hard_tail_rate_p90": float(hard.mean()),
        "validation_pred_tail_rate_p90": float(pred_hard.mean()),
        "validation_tail_precision_p90": float(tp / pred_pos) if pred_pos else None,
        "validation_tail_recall_p90": float(tp / true_pos) if true_pos else None,
        "validation_tail_false_alarm_rate_p90": false_alarm_rate,
        "validation_tail_miss_rate_p90": miss_rate,
        "validation_tail_average_precision": average_precision,
        "validation_tail_ap_lift_p90": (
            float((average_precision - hard.mean()) / max(1.0 - hard.mean(), 1e-12))
            if average_precision is not None
            else None
        ),
        "validation_tail_brier_p90": brier,
        "validation_tail_ece_5bin": float(ece_num / max(len(pred), 1)),
        "validation_tail_calibration_bins": calibration_bins,
        "validation_hard_tail_support_p90": true_pos,
        "validation_pred_tail_support_p90": pred_pos,
        "validation_top_decile_target_mean": float(truth.loc[top_idx].mean()),
        "validation_top_decile_lift": float(truth.loc[top_idx].mean() - truth.mean()),
    }
    if truth.nunique(dropna=True) > 1 and pred.nunique(dropna=True) > 1:
        out["validation_pearson"] = float(truth.corr(pred, method="pearson"))
        out["validation_spearman"] = float(truth.corr(pred, method="spearman"))
    else:
        out["validation_pearson"] = None
        out["validation_spearman"] = None
    if timestamp is not None:
        ts = pd.to_datetime(timestamp.loc[truth.index], utc=True, errors="coerce").dropna()
        if not ts.empty:
            out["validation_start"] = ts.min()
            out["validation_end"] = ts.max()
    return out


def _latent_path_diagnostics(
    probs: np.ndarray,
    *,
    state_mean_duration: np.ndarray,
    state_transition_hazard: np.ndarray,
    duration_norm: float,
) -> dict[str, np.ndarray]:
    if probs.size == 0:
        n = int(probs.shape[0])
        return {
            "latent_time_since_state_change": np.zeros(n, dtype=float),
            "latent_expected_duration": np.zeros(n, dtype=float),
            "latent_transition_hazard": np.zeros(n, dtype=float),
            "latent_regime_maturity": np.zeros(n, dtype=float),
        }
    labels = np.asarray(np.argmax(probs, axis=1), dtype=int)
    time_since = np.zeros(len(labels), dtype=float)
    for i in range(1, len(labels)):
        time_since[i] = 0.0 if labels[i] != labels[i - 1] else time_since[i - 1] + 1.0
    expected_duration = probs @ np.asarray(state_mean_duration, dtype=float)
    transition_hazard = probs @ np.asarray(state_transition_hazard, dtype=float)
    maturity = np.log1p(time_since) / np.log1p(np.maximum(expected_duration, 1.0))
    return {
        "latent_time_since_state_change": time_since,
        "latent_time_since_state_change_log_norm": np.log1p(time_since) / max(math.log1p(duration_norm), 1e-9),
        "latent_expected_duration": expected_duration,
        "latent_transition_hazard": np.clip(transition_hazard, 0.0, 1.0),
        "latent_regime_maturity": np.clip(maturity, 0.0, 5.0),
    }


def _fit_latent_state_duration_stats(labels: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0:
        mean_duration = np.ones(int(n_states), dtype=float)
        hazard = np.zeros(int(n_states), dtype=float)
        return mean_duration, hazard, {"state_mean_duration": mean_duration.tolist(), "state_transition_hazard": hazard.tolist()}
    run_lengths: dict[int, list[int]] = {k: [] for k in range(int(n_states))}
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            state = int(labels[start])
            if 0 <= state < int(n_states):
                run_lengths[state].append(int(i - start))
            start = i
    global_mean = float(np.mean([v for vals in run_lengths.values() for v in vals] or [1.0]))
    mean_duration = np.array(
        [float(np.mean(run_lengths[k])) if run_lengths[k] else global_mean for k in range(int(n_states))],
        dtype=float,
    )
    hazard = np.zeros(int(n_states), dtype=float)
    for k in range(int(n_states)):
        opportunities = int((labels[:-1] == k).sum()) if len(labels) > 1 else 0
        transitions = int(((labels[:-1] == k) & (labels[1:] != k)).sum()) if len(labels) > 1 else 0
        hazard[k] = float(transitions / opportunities) if opportunities else 0.0
    return mean_duration, hazard, {
        "state_mean_duration": {str(k): float(mean_duration[k]) for k in range(int(n_states))},
        "state_transition_hazard": {str(k): float(hazard[k]) for k in range(int(n_states))},
        "global_mean_duration": global_mean,
    }


def _build_future_severity_targets(
    agg: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build primitive future market-pattern severity targets.

    Targets intentionally describe future market geometry, not strategy
    difficulty. They are later normalized through the training empirical CDF.
    """

    work = agg.sort_values("timestamp").reset_index(drop=True).copy()
    idx = work.index
    ret_col = _pick_numeric_col(
        work,
        [
            ("fs__mkt_ret_eq_1h",),
            ("mkt_ret_eq_1h",),
            ("fs__mkt_ret_eq_4h",),
            ("mkt_ret_eq_4h",),
            ("market", "ret", "1h"),
        ],
    )
    rv_col = _pick_numeric_col(
        work,
        [
            ("fs__realized_volatility_24h",),
            ("realized_volatility_24h",),
            ("fs__rv_24h",),
            ("rv_24h",),
            ("volatility",),
            ("barrier_pct",),
        ],
    )
    oi_col = _pick_numeric_col(
        work,
        [
            ("mkt_oi_chg_z_24h",),
            ("oi_value_1d_chg",),
            ("oi_value_1d_log_chg",),
            ("oi_", "chg"),
        ],
    )
    liquidity_col = _pick_numeric_col(
        work,
        [
            ("liquidity_stress_proxy",),
            ("spread_proxy",),
            ("spread", "bps"),
            ("amihud",),
            ("liquidity",),
            ("slippage",),
            ("cost",),
        ],
    )
    ret = (
        pd.to_numeric(work[ret_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if ret_col
        else pd.Series(0.0, index=idx, dtype=float)
    )
    matrix, valid_count = _future_value_matrix(ret, horizon)
    valid = valid_count > 0
    filled = np.where(np.isfinite(matrix), matrix, 0.0)
    cum = np.cumsum(filled, axis=1) if matrix.size else np.empty_like(matrix)
    abs_sum = np.abs(filled).sum(axis=1) if matrix.size else np.zeros(len(work), dtype=float)
    future_sum = filled.sum(axis=1) if matrix.size else np.zeros(len(work), dtype=float)
    future_rv = np.sqrt((filled * filled).sum(axis=1))
    prior_sigma = _prior_sigma_from_returns(ret, horizon).to_numpy(dtype=float)
    denom = prior_sigma * math.sqrt(max(1, int(horizon)))
    if rv_col:
        prior_rv_series = pd.to_numeric(work[rv_col], errors="coerce").replace([np.inf, -np.inf], np.nan).abs()
        rv_fallback = float(prior_rv_series.median()) if prior_rv_series.notna().any() else np.nan
        if not np.isfinite(rv_fallback) or rv_fallback <= 1e-10:
            rv_fallback = float(np.nanmedian(denom)) if np.isfinite(np.nanmedian(denom)) else 1e-4
        prior_rv = prior_rv_series.shift(1).fillna(rv_fallback).clip(lower=1e-8).to_numpy(dtype=float)
    else:
        prior_rv = np.maximum(denom, 1e-8)
    shock_up = np.nanmax(cum, axis=1) / np.maximum(denom, 1e-8) if matrix.size else np.zeros(len(work))
    shock_down = -np.nanmin(cum, axis=1) / np.maximum(denom, 1e-8) if matrix.size else np.zeros(len(work))
    if matrix.size:
        range_norm = (np.nanmax(cum, axis=1) - np.nanmin(cum, axis=1)) / np.maximum(denom, 1e-8)
    else:
        range_norm = np.zeros(len(work), dtype=float)
    trend_eff = future_sum / np.maximum(abs_sum, 1e-8)
    rv_ratio = np.log(np.maximum(future_rv, 1e-8) / np.maximum(prior_rv, 1e-8))
    consolidation = (1.0 - np.abs(trend_eff)) / (1.0 + np.exp(np.log(np.maximum(range_norm, 1e-8))))
    compression = -rv_ratio

    oi = (
        pd.to_numeric(work[oi_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if oi_col
        else pd.Series(0.0, index=idx, dtype=float)
    )
    oi_matrix, _ = _future_value_matrix(oi, horizon)
    oi_filled = np.where(np.isfinite(oi_matrix), oi_matrix, 0.0)
    future_oi = oi_filled.sum(axis=1) if oi_matrix.size else np.zeros(len(work), dtype=float)
    ret_z = _robust_z(pd.Series(future_sum), pd.Series(future_sum)).to_numpy(dtype=float)
    oi_z = _robust_z(pd.Series(future_oi), pd.Series(future_oi)).to_numpy(dtype=float)
    deleveraging = np.maximum(-ret_z, 0.0) * np.maximum(-oi_z, 0.0)

    if liquidity_col:
        liq = pd.to_numeric(work[liquidity_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        liq_matrix, _ = _future_value_matrix(liq, horizon)
        if liq_matrix.size:
            liq_filled = np.where(np.isfinite(liq_matrix), liq_matrix, -np.inf)
            liq_max = liq_filled.max(axis=1)
            liquidity_stress_proxy = np.where(np.isfinite(liq_max), liq_max, np.nan)
        else:
            liquidity_stress_proxy = np.zeros(len(work), dtype=float)
    else:
        liquidity_stress_proxy = np.zeros(len(work), dtype=float)

    targets = pd.DataFrame(
        {
            "timestamp": work["timestamp"],
            f"target_h{horizon}_shock_up": shock_up,
            f"target_h{horizon}_shock_down": shock_down,
            f"target_h{horizon}_rv_ratio": rv_ratio,
            f"target_h{horizon}_trend_efficiency": trend_eff,
            f"target_h{horizon}_consolidation": consolidation,
            f"target_h{horizon}_compression": compression,
            f"target_h{horizon}_deleveraging": deleveraging,
            f"target_h{horizon}_liquidity_stress_proxy": liquidity_stress_proxy,
        }
    )
    for col in targets.columns:
        if col == "timestamp":
            continue
        vals = pd.to_numeric(targets[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = vals.where(valid)
        targets[col] = vals
    return targets, {
        "horizon_steps": int(horizon),
        "source_columns": {
            "return": ret_col,
            "realized_volatility": rv_col,
            "open_interest": oi_col,
            "liquidity": liquidity_col,
        },
        "valid_rows": int(valid.sum()),
    }


def add_forecast_state_heads(
    train_state: pd.DataFrame,
    eval_state: pd.DataFrame,
    horizon_steps: int | list[int] | tuple[int, ...],
    *,
    train_agg: pd.DataFrame | None = None,
    eval_agg: pd.DataFrame | None = None,
    forecast_model_kind: str = "lightgbm",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, artifact, report = fit_forecast_state_heads(
        train_state,
        horizon_steps,
        train_agg=train_agg,
        forecast_model_kind=forecast_model_kind,
    )
    eval_out = transform_forecast_state_heads(eval_state, artifact, agg=eval_agg)
    return train, eval_out, report


def _append_forecast_reliability_channels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    forecast_cols = [
        c for c in out.columns
        if str(c).startswith("forecast_") and pd.api.types.is_numeric_dtype(out[c])
    ]
    if not forecast_cols:
        out["state_forecast_disagreement"] = 0.0
        if "state_uncertainty" not in out.columns:
            out["state_uncertainty"] = 0.0
        return out
    vals = out[forecast_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    disagreement = vals.std(axis=1).fillna(0.0).clip(0.0, 0.5) * 2.0
    out["state_forecast_disagreement"] = disagreement.to_numpy(dtype=float)
    if "state_uncertainty" in out.columns:
        existing = pd.to_numeric(out["state_uncertainty"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        out["state_uncertainty"] = np.maximum(existing.to_numpy(dtype=float), out["state_forecast_disagreement"].to_numpy(dtype=float))
    else:
        out["state_uncertainty"] = out["state_forecast_disagreement"].to_numpy(dtype=float)
    return out


def _forecast_feature_frame(
    state: pd.DataFrame,
    *,
    extra_agg: pd.DataFrame | None,
    extra_cols: list[str],
    features: list[str] | None = None,
) -> pd.DataFrame:
    frame = state.sort_values("timestamp").reset_index(drop=True).copy()
    if extra_agg is not None and extra_cols:
        present = [c for c in extra_cols if c in extra_agg.columns]
        if present:
            frame = frame.merge(
                extra_agg[["timestamp", *present]],
                on="timestamp",
                how="left",
                validate="one_to_one",
            )
    if features is not None:
        for col in features:
            if col not in frame.columns:
                frame[col] = np.nan
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _fallback_forecast_values(train: pd.DataFrame, fallback_axis: str | None) -> pd.Series:
    if isinstance(fallback_axis, str) and fallback_axis in train.columns:
        values = pd.to_numeric(train[fallback_axis], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.Series(_bounded_sigmoid(values), index=train.index)
    return pd.Series(0.5, index=train.index, dtype=float)


def _chronological_oof_forecast_predictions(
    *,
    make_model: Any,
    feature_frame: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    valid_norm: pd.Series,
    min_train_rows: int = 40,
    min_valid_rows: int = 10,
    max_folds: int = 4,
    seed_base: int = 1700,
) -> tuple[pd.Series, dict[str, Any]]:
    valid_positions = np.flatnonzero(valid_norm.to_numpy(dtype=bool))
    oof = pd.Series(np.nan, index=feature_frame.index, dtype=float)
    report: dict[str, Any] = {
        "train_prediction_mode": "chronological_expanding_oof_or_fallback",
        "oof_rows": 0,
        "oof_eligible_rows": int(len(valid_positions)),
        "oof_fold_count": 0,
        "oof_coverage": 0.0,
        "oof_min_train_rows": int(min_train_rows),
        "oof_min_valid_rows": int(min_valid_rows),
    }
    n_valid = int(len(valid_positions))
    if n_valid < int(min_train_rows) + int(min_valid_rows):
        report["oof_reason"] = "insufficient_valid_rows"
        return oof, report

    max_possible = max(1, (n_valid - int(min_train_rows)) // max(1, int(min_valid_rows)))
    fold_count = int(min(max_folds, max_possible))
    bounds = np.linspace(int(min_train_rows), n_valid, fold_count + 1).round().astype(int)
    fold_reports: list[dict[str, int]] = []
    for fold_idx in range(fold_count):
        start = int(bounds[fold_idx])
        end = int(bounds[fold_idx + 1])
        if end <= start:
            continue
        train_pos = valid_positions[:start]
        valid_pos = valid_positions[start:end]
        if len(train_pos) < int(min_train_rows) or len(valid_pos) < 1:
            continue
        train_idx = feature_frame.index[train_pos]
        valid_idx = feature_frame.index[valid_pos]
        fold_target = _empirical_cdf_transform(target.loc[train_idx], target).astype(float)
        y_fit = fold_target.loc[train_idx].replace([np.inf, -np.inf], np.nan)
        fit_mask = y_fit.notna()
        if int(fit_mask.sum()) < int(min_train_rows):
            continue
        fit_idx = train_idx[fit_mask.to_numpy(dtype=bool)]
        model = make_model(int(seed_base + fold_idx))
        model.fit(feature_frame.loc[fit_idx, features], y_fit.loc[fit_idx])
        oof.loc[valid_idx] = np.clip(model.predict(feature_frame.loc[valid_idx, features]), 0.0, 1.0)
        fold_reports.append(
            {
                "fold": int(fold_idx + 1),
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )

    oof_rows = int(oof.notna().sum())
    report.update(
        {
            "oof_rows": oof_rows,
            "oof_fold_count": int(len(fold_reports)),
            "oof_coverage": float(oof_rows / max(n_valid, 1)),
            "oof_folds": fold_reports,
        }
    )
    if not fold_reports:
        report["oof_reason"] = "no_valid_expanding_fold"
    return oof, report


def fit_forecast_state_heads(
    train_state: pd.DataFrame,
    horizon_steps: int | list[int] | tuple[int, ...],
    *,
    train_agg: pd.DataFrame | None = None,
    forecast_model_kind: str = "lightgbm",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fit train-only future market-pattern severity heads.

    The learned regressors are deliberately stored as an artifact so deployment
    scoring can add the same forecast columns without rebuilding targets or
    looking at evaluation-period future paths.
    """

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    forecast_kind = str(forecast_model_kind or "lightgbm")
    if forecast_kind not in {"lightgbm", "xgboost"}:
        raise ValueError(f"unknown forecast_model_kind={forecast_kind!r}")
    LGBMRegressor = None
    XGBRegressor = None
    if forecast_kind == "lightgbm":
        try:
            from lightgbm import LGBMRegressor as _LGBMRegressor

            LGBMRegressor = _LGBMRegressor
            forecast_backend = "lightgbm_lgbm_regressor"
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler

            forecast_backend = "sklearn_gradient_boosting_regressor_fallback"
    else:
        try:
            from xgboost import XGBRegressor as _XGBRegressor

            XGBRegressor = _XGBRegressor
            forecast_backend = "xgboost_xgb_regressor"
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("forecast_model_kind='xgboost' requires xgboost to be installed") from exc

    train = train_state.sort_values("timestamp").reset_index(drop=True).copy()
    axis_cols = [c for c in train.columns if c.startswith("state_")]
    shared_extra: list[str] = []
    if train_agg is not None:
        shared_extra = [
            c
            for c in train_agg.columns
            if c != "timestamp"
            and pd.api.types.is_numeric_dtype(train_agg[c])
            and not any(token in c.lower() for token in OUTCOME_TOKENS)
        ][:96]
    train_feature_frame = _forecast_feature_frame(
        train,
        extra_agg=train_agg,
        extra_cols=shared_extra,
    )
    features = [
        c
        for c in train_feature_frame.columns
        if c != "timestamp"
        and pd.api.types.is_numeric_dtype(train_feature_frame[c])
    ]
    if not features:
        features = axis_cols

    def make_forecast_model(seed: int):
        if forecast_kind == "xgboost" and XGBRegressor is not None:
            return make_pipeline(
                SimpleImputer(strategy="median"),
                XGBRegressor(
                    objective="reg:pseudohubererror",
                    random_state=int(seed),
                    n_estimators=90,
                    learning_rate=0.04,
                    max_depth=3,
                    min_child_weight=max(5.0, float(len(train)) * 0.01),
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=3.0,
                    tree_method="hist",
                    n_jobs=1,
                    verbosity=0,
                ),
            )
        if LGBMRegressor is not None:
            return make_pipeline(
                SimpleImputer(strategy="median"),
                LGBMRegressor(
                    objective="regression",
                    random_state=int(seed),
                    n_estimators=120,
                    learning_rate=0.035,
                    max_depth=3,
                    num_leaves=7,
                    min_child_samples=max(10, int(math.ceil(0.02 * len(train)))),
                    subsample=0.90,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    n_jobs=1,
                    deterministic=True,
                    force_col_wise=True,
                    verbosity=-1,
                ),
            )
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            GradientBoostingRegressor(
                random_state=int(seed),
                max_depth=2,
                n_estimators=90,
                learning_rate=0.04,
            ),
        )

    if isinstance(horizon_steps, (list, tuple)):
        horizons = tuple(sorted({max(1, int(x)) for x in horizon_steps}))
    else:
        horizons = (max(1, int(horizon_steps)),)
    report: dict[str, Any] = {
        "horizon_steps": list(horizons),
        "mode": "primitive_future_soft_severity_regressors",
        "forecast_model_kind": forecast_kind,
        "model_backend": forecast_backend,
        "features_used": int(len(features)),
        "targets": {},
        "reliability_channels": {
            "state_forecast_disagreement": "rowwise_cross_forecast_std_scaled_to_0_1",
            "state_uncertainty": "max_observed_uncertainty_and_forecast_disagreement",
        },
    }
    artifact: dict[str, Any] = {
        "mode": "primitive_future_soft_severity_regressors_v1",
        "forecast_model_kind": forecast_kind,
        "model_backend": forecast_backend,
        "horizon_steps": list(horizons),
        "features": features,
        "extra_agg_columns": shared_extra,
        "axis_columns": axis_cols,
        "targets": {},
    }
    for horizon in horizons:
        if train_agg is not None:
            target_frame, target_report = _build_future_severity_targets(train_agg, horizon)
            target_frame = train[["timestamp"]].merge(target_frame, on="timestamp", how="left", validate="one_to_one")
        else:
            target_report = {"horizon_steps": int(horizon), "mode": "axis_mean_fallback"}
            target_frame = train[["timestamp"]].copy()
            for axis in axis_cols:
                target_frame[f"target_h{horizon}_{axis.removeprefix('state_')}"] = (
                    train[axis]
                    .shift(-1)
                    .rolling(window=horizon, min_periods=1)
                    .mean()
                    .shift(-(horizon - 1))
                )
        report.setdefault("target_source_reports", {})[f"h{horizon}"] = target_report
        target_cols = [c for c in target_frame.columns if c != "timestamp"]
        for target_name in target_cols:
            target = pd.to_numeric(target_frame[target_name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = target.notna()
            forecast_name = target_name.replace("target_", "forecast_", 1)
            normalized_target = _empirical_cdf_transform(target.loc[valid], target).astype(float)
            valid_norm = normalized_target.notna()
            fallback_axis = "state_realized_vol" if "rv_ratio" in target_name else axis_cols[0] if axis_cols else None
            fallback_values_train = _fallback_forecast_values(train, fallback_axis)
            if valid.sum() < 40 or float(target.loc[valid].std(ddof=0)) < 1e-12:
                train[forecast_name] = fallback_values_train.to_numpy(dtype=float)
                report["targets"][forecast_name] = {
                    "mode": "current_axis_fallback",
                    "rows": int(valid.sum()),
                    "horizon_steps": int(horizon),
                    "raw_target": target_name,
                    "fallback_axis": fallback_axis,
                    "train_prediction_mode": "bounded_current_axis_fallback",
                }
                artifact["targets"][forecast_name] = {
                    "mode": "current_axis_fallback",
                    "horizon_steps": int(horizon),
                    "raw_target": target_name,
                    "fallback_axis": fallback_axis,
                    "target_cdf_reference": _empirical_cdf_reference(target.loc[valid]),
                }
                continue
            validation_report: dict[str, Any] = {
                "validation_mode": "chronological_holdout_not_used_for_final_fit",
                "validation_rows": 0,
            }
            valid_positions = np.flatnonzero(valid_norm.to_numpy(dtype=bool))
            if len(valid_positions) >= 80:
                holdout_rows = max(20, int(math.ceil(0.20 * len(valid_positions))))
                fit_positions = valid_positions[:-holdout_rows]
                holdout_positions = valid_positions[-holdout_rows:]
                if len(fit_positions) >= 40 and len(holdout_positions) >= 10:
                    fit_idx = train.index[fit_positions]
                    holdout_idx = train.index[holdout_positions]
                    normalized_holdout_target = _empirical_cdf_transform(
                        target.loc[fit_idx],
                        target,
                    ).astype(float)
                    fit_norm = normalized_holdout_target.loc[fit_idx].notna()
                    holdout_norm = normalized_holdout_target.loc[holdout_idx].notna()
                    if fit_norm.sum() >= 40 and holdout_norm.sum() >= 10:
                        fit_eval_idx = fit_idx[fit_norm.to_numpy(dtype=bool)]
                        holdout_eval_idx = holdout_idx[holdout_norm.to_numpy(dtype=bool)]
                        validation_model = make_forecast_model(900 + horizon)
                        validation_model.fit(
                            train_feature_frame.loc[fit_eval_idx, features],
                            normalized_holdout_target.loc[fit_eval_idx],
                        )
                        validation_pred = np.clip(
                            validation_model.predict(train_feature_frame.loc[holdout_eval_idx, features]),
                            0.0,
                            1.0,
                        )
                        validation_report = {
                            "validation_mode": "chronological_holdout_not_used_for_final_fit",
                            **_forecast_validation_metrics(
                                normalized_holdout_target.loc[holdout_eval_idx],
                                validation_pred,
                                timestamp=train.loc[holdout_eval_idx, "timestamp"],
                            ),
                        }
            model = make_forecast_model(101 + horizon)
            model.fit(train_feature_frame.loc[valid_norm, features], normalized_target.loc[valid_norm])
            oof_pred, oof_report = _chronological_oof_forecast_predictions(
                make_model=make_forecast_model,
                feature_frame=train_feature_frame,
                features=features,
                target=target,
                valid_norm=valid_norm,
                seed_base=1700 + int(horizon),
            )
            train[forecast_name] = oof_pred.fillna(fallback_values_train).clip(0.0, 1.0).to_numpy(dtype=float)
            hard = normalized_target.loc[valid_norm] >= 0.90
            report["targets"][forecast_name] = {
                "mode": "gbm_soft_empirical_cdf_target",
                "rows": int(valid_norm.sum()),
                "target_std": float(target.loc[valid].std(ddof=0)),
                "soft_target_mean": float(normalized_target.loc[valid_norm].mean()),
                "hard_tail_rate_p90": float(hard.mean()) if len(hard) else None,
                "horizon_steps": int(horizon),
                "raw_target": target_name,
                "fallback_axis": fallback_axis,
                **validation_report,
                **oof_report,
            }
            artifact["targets"][forecast_name] = {
                "mode": "gbm_soft_empirical_cdf_target",
                "horizon_steps": int(horizon),
                "raw_target": target_name,
                "target_cdf_reference": _empirical_cdf_reference(target.loc[valid]),
                "model": model,
            }
    train = _append_forecast_reliability_channels(train)
    return train, artifact, report


def transform_forecast_state_heads(
    state: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    agg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = state.sort_values("timestamp").reset_index(drop=True).copy()
    features = list(artifact.get("features", []))
    feature_frame = _forecast_feature_frame(
        out,
        extra_agg=agg,
        extra_cols=list(artifact.get("extra_agg_columns", [])),
        features=features,
    )
    for forecast_name, spec in dict(artifact.get("targets", {})).items():
        mode = str(spec.get("mode", ""))
        if mode == "gbm_soft_empirical_cdf_target":
            model = spec.get("model")
            if model is None:
                raise KeyError(f"Forecast artifact target {forecast_name!r} is missing fitted model")
            out[forecast_name] = np.clip(model.predict(feature_frame[features]), 0.0, 1.0)
        elif mode == "current_axis_fallback":
            fallback_axis = spec.get("fallback_axis")
            if isinstance(fallback_axis, str) and fallback_axis in out.columns:
                values = _fallback_forecast_values(out, fallback_axis)
            else:
                values = pd.Series(0.5, index=out.index, dtype=float)
            out[forecast_name] = pd.to_numeric(values, errors="coerce").fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=float)
        else:
            raise ValueError(f"Unknown forecast artifact mode for {forecast_name!r}: {mode!r}")
    return _append_forecast_reliability_channels(out)


def add_latent_state_probs(
    train_state: pd.DataFrame,
    eval_state: pd.DataFrame,
    n_states: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, artifact, report = fit_latent_state_probs(train_state, n_states)
    eval_out = transform_latent_state_probs(eval_state, artifact)
    return train, eval_out, report


def fit_latent_state_probs(
    train_state: pd.DataFrame,
    n_states: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fit train-only diagonal-GMM latent regime probability transformer."""

    from sklearn.impute import SimpleImputer
    from sklearn.mixture import GaussianMixture
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = train_state.copy()
    n_states = int(n_states)
    state_cols = [
        c
        for c in train.columns
        if c != "timestamp" and pd.api.types.is_numeric_dtype(train[c])
    ]
    if len(state_cols) < 2 or len(train) < max(30, n_states * 10):
        for k in range(n_states):
            train[f"latent_gmm_p{k}"] = 1.0 / n_states
        train["latent_entropy"] = math.log(n_states)
        train["latent_max_prob"] = 1.0 / n_states
        train["latent_transition_pressure"] = 0.0
        mean_duration = np.full(n_states, max(1.0, float(len(train))), dtype=float)
        hazard = np.zeros(n_states, dtype=float)
        diagnostics = _latent_path_diagnostics(
            np.full((len(train), n_states), 1.0 / n_states, dtype=float),
            state_mean_duration=mean_duration,
            state_transition_hazard=hazard,
            duration_norm=max(1.0, float(len(train))),
        )
        for col, values in diagnostics.items():
            train[col] = values
        artifact = {
            "mode": "uniform_fallback",
            "n_states": int(n_states),
            "state_cols": state_cols,
            "state_mean_duration": mean_duration,
            "state_transition_hazard": hazard,
            "duration_norm": max(1.0, float(len(train))),
        }
        return train, artifact, {
            "mode": "uniform_fallback",
            "n_states": int(n_states),
            "state_cols": state_cols,
            "latent_feature_contract": "probabilities_entropy_transition_hazard_duration_no_raw_cluster_ids",
        }
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X_train = pipe.fit_transform(train[state_cols])
    gmm = GaussianMixture(
        n_components=n_states,
        covariance_type="diag",
        random_state=211,
        reg_covar=1e-4,
        max_iter=200,
    )
    gmm.fit(X_train)
    p_train = gmm.predict_proba(X_train)
    eps = 1e-12
    for k in range(n_states):
        train[f"latent_gmm_p{k}"] = p_train[:, k]
    train_labels = np.asarray(np.argmax(p_train, axis=1), dtype=int)
    mean_duration, hazard, duration_report = _fit_latent_state_duration_stats(train_labels, n_states)
    train["latent_entropy"] = -np.sum(p_train * np.log(p_train + eps), axis=1)
    train["latent_max_prob"] = np.max(p_train, axis=1)
    train["latent_transition_pressure"] = (
        pd.DataFrame(p_train).diff().abs().mean(axis=1).fillna(0.0).to_numpy()
    )
    duration_norm = max(float(np.nanmax(mean_duration)), 1.0)
    diagnostics = _latent_path_diagnostics(
        p_train,
        state_mean_duration=mean_duration,
        state_transition_hazard=hazard,
        duration_norm=duration_norm,
    )
    for col, values in diagnostics.items():
        train[col] = values
    artifact = {
        "mode": "gaussian_mixture_diag",
        "n_states": int(n_states),
        "state_cols": state_cols,
        "preprocess": pipe,
        "gmm": gmm,
        "state_mean_duration": mean_duration,
        "state_transition_hazard": hazard,
        "duration_norm": duration_norm,
    }
    return train, artifact, {
        "mode": "gaussian_mixture_diag",
        "n_states": n_states,
        "state_cols": state_cols,
        "train_converged": bool(gmm.converged_),
        "latent_feature_contract": "probabilities_entropy_transition_hazard_duration_no_raw_cluster_ids",
        "hard_state_ids_not_semantic": True,
        **duration_report,
    }


def transform_latent_state_probs(state: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    out = state.copy()
    n_states = int(artifact.get("n_states", 1))
    mode = str(artifact.get("mode", ""))
    if mode == "uniform_fallback":
        probs = np.full((len(out), n_states), 1.0 / max(1, n_states), dtype=float)
    elif mode == "gaussian_mixture_diag":
        state_cols = list(artifact.get("state_cols", []))
        for col in state_cols:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = pd.to_numeric(out[col], errors="coerce")
        preprocess = artifact.get("preprocess")
        gmm = artifact.get("gmm")
        if preprocess is None or gmm is None:
            raise KeyError("Latent artifact is missing preprocess or GMM model")
        probs = gmm.predict_proba(preprocess.transform(out[state_cols]))
    else:
        raise ValueError(f"Unknown latent artifact mode: {mode!r}")
    eps = 1e-12
    for k in range(n_states):
        out[f"latent_gmm_p{k}"] = probs[:, k]
    out["latent_entropy"] = -np.sum(probs * np.log(probs + eps), axis=1)
    out["latent_max_prob"] = np.max(probs, axis=1)
    out["latent_transition_pressure"] = (
        pd.DataFrame(probs).diff().abs().mean(axis=1).fillna(0.0).to_numpy()
    )
    diagnostics = _latent_path_diagnostics(
        probs,
        state_mean_duration=np.asarray(artifact.get("state_mean_duration", np.ones(n_states)), dtype=float),
        state_transition_hazard=np.asarray(artifact.get("state_transition_hazard", np.zeros(n_states)), dtype=float),
        duration_norm=float(artifact.get("duration_norm", 1.0) or 1.0),
    )
    for col, values in diagnostics.items():
        out[col] = values
    return out


def _trade_outcome_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reason = out.get("simple_policy_exit_reason", pd.Series("", index=out.index)).astype(str).str.lower()
    out["_is_full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float)
    out["_is_timeout"] = reason.str.contains("timeout", regex=False).astype(float)
    out["_net_return"] = _safe_numeric(out, "net_return").fillna(0.0)
    out["_rank"] = _safe_numeric(out, "normalized_rank_score").fillna(0.0).clip(0.0, 1.0)
    threshold = _safe_numeric(out, "base_strategy_threshold")
    if "deployment_rank_threshold" in out.columns:
        deploy = _safe_numeric(out, "deployment_rank_threshold")
        threshold = threshold.where(threshold >= 0.50, deploy)
    default_floor = out["strategy_id"].map(_default_strategy_threshold).astype(float)
    threshold = threshold.where(threshold >= 0.50, default_floor)
    out["_threshold"] = threshold.fillna(default_floor).clip(0.0, 1.01)
    return out


class RankOutcomeCurves:
    def __init__(self, table: pd.DataFrame, global_values: dict[str, float]):
        self.table = table
        self.global_values = global_values

    def predict(self, strategy: pd.Series, rank: pd.Series, target: str) -> np.ndarray:
        out = np.zeros(len(rank), dtype=float)
        for strat in pd.Series(strategy).astype(str).unique():
            mask = pd.Series(strategy).astype(str).eq(strat).to_numpy()
            sub = self.table.loc[self.table["strategy_id"].eq(strat)]
            if sub.empty:
                out[mask] = float(self.global_values[target])
                continue
            centers = sub["rank_center"].to_numpy(dtype=float)
            vals = sub[target].to_numpy(dtype=float)
            out[mask] = np.interp(pd.Series(rank).to_numpy(dtype=float)[mask], centers, vals)
        return out

    def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
        pred = float(self.predict(pd.Series([strategy_id]), pd.Series([base_threshold]), target)[0])
        return float(np.clip(pred + pad, 0.05, 0.90))


def fit_rank_curves(df: pd.DataFrame, bins: int = 16) -> RankOutcomeCurves:
    work = _trade_outcome_flags(df)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    work["_rank_bin"] = pd.cut(work["_rank"], edges, include_lowest=True, labels=False)
    global_values = {
        "mu": float(work["_net_return"].mean()),
        "psl": float(work["_is_full_sl"].mean()),
        "pto": float(work["_is_timeout"].mean()),
    }
    rows: list[dict[str, Any]] = []
    for strategy, g in work.groupby("strategy_id"):
        strat_values = {
            "mu": float(g["_net_return"].mean()),
            "psl": float(g["_is_full_sl"].mean()),
            "pto": float(g["_is_timeout"].mean()),
        }
        for b in range(int(bins)):
            h = g.loc[g["_rank_bin"].eq(b)]
            center = float((edges[b] + edges[b + 1]) / 2.0)
            n = int(len(h))
            shrink = min(1.0, n / 80.0)
            rows.append(
                {
                    "strategy_id": strategy,
                    "rank_bin": b,
                    "rank_center": center,
                    "n": n,
                    "mu": shrink * float(h["_net_return"].mean() if n else strat_values["mu"])
                    + (1.0 - shrink) * strat_values["mu"],
                    "psl": shrink * float(h["_is_full_sl"].mean() if n else strat_values["psl"])
                    + (1.0 - shrink) * strat_values["psl"],
                    "pto": shrink * float(h["_is_timeout"].mean() if n else strat_values["pto"])
                    + (1.0 - shrink) * strat_values["pto"],
                }
            )
    table = pd.DataFrame(rows)
    # Light smoothing across neighboring bins for stability.
    for col in ("mu", "psl", "pto"):
        table[col] = table.groupby("strategy_id")[col].transform(
            lambda s: s.rolling(3, min_periods=1, center=True).mean()
        )
    return RankOutcomeCurves(table=table, global_values=global_values)


def _candidate_feature_columns(df: pd.DataFrame, state_cols: list[str], max_keyword_cols: int) -> list[str]:
    preferred = [
        "normalized_rank_score",
        "strategy_rank_pct",
        "rank_pct",
        "calibrated_score",
        "expected_spread_bps",
        "spread_cost_bps",
        "fees_bps",
        "slippage_bps",
        "liquidity_capacity_weight",
        "barrier_pct",
        "policy_effective_barrier_pct",
    ]
    keyword_cols = [
        c
        for c in df.columns
        if c not in preferred
        and pd.api.types.is_numeric_dtype(df[c])
        and not any(token in c.lower() for token in OUTCOME_TOKENS)
        and any(
            token in c.lower()
            for token in (
                "uncert",
                "drift",
                "leaf",
                "support",
                "rare",
                "centroid",
                "regime",
                "spread",
                "liquidity",
                "score",
            )
        )
    ]
    cols = [c for c in preferred if c in df.columns] + keyword_cols[: max(0, int(max_keyword_cols))] + state_cols
    return list(dict.fromkeys(cols))


def _is_state_interaction_col(col: str) -> bool:
    return col.startswith("state_") or col.startswith("forecast_")


def _cap_response_rows(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    # Preserve chronological coverage while keeping strategy representation.
    pieces: list[pd.DataFrame] = []
    per_strategy = max(1, int(max_rows / max(1, frame["strategy_id"].nunique())))
    for _, g in frame.sort_values("timestamp").groupby("strategy_id", sort=False):
        if len(g) <= per_strategy:
            pieces.append(g)
            continue
        # Deterministic stratified sampling by timestamp order.
        idx = np.linspace(0, len(g) - 1, per_strategy).round().astype(int)
        pieces.append(g.iloc[np.unique(idx)])
    out = pd.concat(pieces, axis=0).sort_values(["timestamp", "strategy_id"])
    if len(out) > max_rows:
        idx = np.linspace(0, len(out) - 1, max_rows).round().astype(int)
        out = out.iloc[np.unique(idx)]
    return out.reset_index(drop=True)


def build_response_frame(candidates: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in candidates.columns:
        raise ValueError("candidate frame is missing timestamp")
    if "timestamp" not in state.columns:
        raise ValueError("state frame is missing timestamp")
    state = state.copy()
    state["timestamp"] = pd.to_datetime(state["timestamp"], utc=True, errors="coerce")
    if state["timestamp"].isna().any():
        raise ValueError("state frame contains non-finite timestamps")
    duplicate_count = int(state["timestamp"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"state frame must have exactly one row per timestamp; duplicates={duplicate_count}")
    state_cols = [str(c) for c in state.columns if c != "timestamp"]
    overlap = sorted(set(state_cols) & set(map(str, candidates.columns)) - {"timestamp"})
    if overlap:
        preview = ", ".join(overlap[:12])
        raise ValueError(f"state columns overlap candidate columns before join: {preview}")
    out = candidates.merge(state, on="timestamp", how="left", validate="many_to_one")
    _validate_joined_state_invariance(out, state_cols)
    return _trade_outcome_flags(out)


def state_frame_contract_report(state: pd.DataFrame, *, context: str) -> dict[str, Any]:
    """Measured contract report for a timestamp-level market-state frame."""

    if "timestamp" not in state.columns:
        raise ValueError(f"{context} state frame is missing timestamp")
    timestamps = pd.to_datetime(state["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{context} state frame contains non-finite timestamps")
    duplicate_count = int(timestamps.duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{context} state frame must have exactly one row per timestamp; duplicates={duplicate_count}")
    feature_cols = [str(c) for c in state.columns if c != "timestamp"]
    numeric_cols = [
        c for c in feature_cols
        if c in state.columns and pd.api.types.is_numeric_dtype(state[c])
    ]
    if numeric_cols:
        values = state[numeric_cols].replace([np.inf, -np.inf], np.nan)
        nonfinite_count = int(values.isna().sum().sum())
        finite_share = float(values.notna().to_numpy(dtype=bool).mean()) if values.size else 1.0
    else:
        nonfinite_count = 0
        finite_share = 1.0
    return {
        "context": str(context),
        "row_count": int(len(state)),
        "timestamp_count": int(timestamps.nunique()),
        "one_row_per_timestamp": True,
        "duplicate_timestamp_count": 0,
        "state_feature_count": int(len(feature_cols)),
        "numeric_state_feature_count": int(len(numeric_cols)),
        "nonfinite_state_value_count": nonfinite_count,
        "finite_state_value_share": finite_share,
    }


def joined_state_invariance_report(
    frame: pd.DataFrame,
    state_cols: list[str],
    *,
    context: str,
) -> dict[str, Any]:
    """Measured contract report after market state is joined to candidates."""

    _validate_joined_state_invariance(frame, state_cols)
    present = [c for c in state_cols if c in frame.columns]
    if not present or frame.empty:
        max_nunique = 0
    else:
        nunique = frame.groupby("timestamp", sort=False)[present].nunique(dropna=False)
        max_nunique = int(nunique.max().max()) if not nunique.empty else 0
    return {
        "context": str(context),
        "row_count": int(len(frame)),
        "timestamp_count": int(pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").nunique())
        if "timestamp" in frame.columns
        else 0,
        "state_feature_count": int(len(present)),
        "state_join_timestamp_constant": True,
        "max_state_values_per_timestamp": max_nunique,
    }


def market_state_timestamp_panel(
    frames: Iterable[tuple[str, str, pd.DataFrame]],
) -> pd.DataFrame:
    """Build a long-form market-state timestamp panel.

    Each input frame must be timestamp-level. The output keeps the state
    feature columns wide, while tagging rows with ``split`` and
    ``state_level`` so training, evaluation and scoring panels can be audited
    together without recomputing fitted references.
    """

    parts: list[pd.DataFrame] = []
    for split, state_level, frame in frames:
        if frame is None or frame.empty:
            continue
        state_frame_contract_report(frame, context=f"{split}_{state_level}_timestamp_panel")
        part = frame.copy()
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True, errors="coerce")
        part.insert(0, "state_level", str(state_level))
        part.insert(0, "split", str(split))
        for col in part.columns:
            if col in {"split", "state_level", "timestamp"}:
                continue
            if pd.api.types.is_numeric_dtype(part[col]):
                part[col] = pd.to_numeric(part[col], errors="coerce").astype("float32")
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["split", "state_level", "timestamp"])
    out = pd.concat(parts, ignore_index=True, sort=False)
    return out.sort_values(["split", "state_level", "timestamp"]).reset_index(drop=True)


def market_state_feature_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-feature finite coverage for a market-state timestamp panel."""

    result_cols = [
        "split",
        "state_level",
        "feature",
        "row_count",
        "finite_count",
        "nonfinite_count",
        "finite_share",
        "mean",
        "std",
        "min",
        "max",
    ]
    if panel.empty:
        return pd.DataFrame(columns=result_cols)
    required = {"split", "state_level", "timestamp"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"market state timestamp panel missing columns: {sorted(missing)}")
    feature_cols = [c for c in panel.columns if c not in required]
    rows: list[dict[str, Any]] = []
    for (split, state_level), g in panel.groupby(["split", "state_level"], sort=True, dropna=False):
        for col in feature_cols:
            values = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            row_count = int(len(values))
            finite = values.notna()
            finite_count = int(finite.sum())
            rows.append(
                {
                    "split": str(split),
                    "state_level": str(state_level),
                    "feature": str(col),
                    "row_count": row_count,
                    "finite_count": finite_count,
                    "nonfinite_count": int(row_count - finite_count),
                    "finite_share": float(finite_count / max(row_count, 1)),
                    "mean": float(values.mean()) if finite_count else np.nan,
                    "std": float(values.std(ddof=0)) if finite_count else np.nan,
                    "min": float(values.min()) if finite_count else np.nan,
                    "max": float(values.max()) if finite_count else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=result_cols)


def _validate_joined_state_invariance(frame: pd.DataFrame, state_cols: list[str]) -> None:
    """Require market state to stay timestamp-level after candidate expansion."""

    if not state_cols or frame.empty:
        return
    if "timestamp" not in frame.columns:
        raise ValueError("joined frame is missing timestamp")
    present = [c for c in state_cols if c in frame.columns]
    if not present:
        return
    nunique = frame.groupby("timestamp", sort=False)[present].nunique(dropna=False)
    varying = nunique.columns[(nunique > 1).any(axis=0)].tolist()
    if varying:
        preview = ", ".join(map(str, varying[:12]))
        raise ValueError(f"joined state columns vary within timestamp: {preview}")


def _frontier_weights(
    frame: pd.DataFrame,
    *,
    frontier_gamma: float = 3.0,
    frontier_bandwidth: float = 0.06,
    balance_timestamps: bool = True,
    balance_strategies: bool = True,
) -> np.ndarray:
    """Policy-frontier response weights.

    The response model should learn state effects where a threshold decision can
    change. Timestamp and strategy balancing stop dense periods or high-volume
    heads from dominating the shared response layer.
    """

    w = np.ones(len(frame), dtype=float)
    if balance_timestamps:
        ts_counts = frame.groupby("timestamp")["timestamp"].transform("size").to_numpy(dtype=float)
        w *= 1.0 / np.maximum(ts_counts, 1.0)
    if balance_strategies:
        strategy_counts = frame.groupby("strategy_id")["strategy_id"].transform("size").to_numpy(dtype=float)
        w *= 1.0 / np.maximum(strategy_counts, 1.0)
    gamma = max(float(frontier_gamma), 0.0)
    bandwidth = max(float(frontier_bandwidth), 1e-6)
    rank = _safe_numeric(frame, "_rank").fillna(0.0).to_numpy(dtype=float)
    threshold = _safe_numeric(frame, "_threshold").fillna(0.0).to_numpy(dtype=float)
    frontier = 1.0 + gamma * np.exp(-np.abs(rank - threshold) / bandwidth)
    w *= frontier
    return w / max(float(np.mean(w)), 1e-12)


def _response_weight_report(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, Any]:
    weight = pd.Series(np.asarray(weights, dtype=float), index=frame.index).replace([np.inf, -np.inf], np.nan)
    finite = weight.dropna()
    if finite.empty:
        return {
            "finite_weight_rows": 0,
            "effective_sample_size": 0.0,
            "frontier_abs_rank_threshold_distance_q50": np.nan,
            "frontier_abs_rank_threshold_distance_q90": np.nan,
        }
    distance = (_safe_numeric(frame, "_rank").fillna(0.0) - _safe_numeric(frame, "_threshold").fillna(0.0)).abs()
    sum_w = float(finite.sum())
    sum_w2 = float(np.square(finite.to_numpy(dtype=float)).sum())
    strategy_mass = (
        frame.loc[finite.index]
        .assign(_response_weight=finite.to_numpy(dtype=float))
        .groupby("strategy_id")["_response_weight"]
        .sum()
        .sort_values(ascending=False)
    )
    return {
        "finite_weight_rows": int(len(finite)),
        "weight_mean": float(finite.mean()),
        "weight_q05": float(finite.quantile(0.05)),
        "weight_q50": float(finite.quantile(0.50)),
        "weight_q95": float(finite.quantile(0.95)),
        "effective_sample_size": float((sum_w * sum_w) / max(sum_w2, 1e-12)),
        "frontier_abs_rank_threshold_distance_q50": float(distance.loc[finite.index].quantile(0.50)),
        "frontier_abs_rank_threshold_distance_q90": float(distance.loc[finite.index].quantile(0.90)),
        "near_frontier_002_weight_share": float(
            finite.loc[distance.loc[finite.index] <= 0.02].sum() / max(sum_w, 1e-12)
        ),
        "near_frontier_005_weight_share": float(
            finite.loc[distance.loc[finite.index] <= 0.05].sum() / max(sum_w, 1e-12)
        ),
        "strategy_weight_mass_top5": {str(k): float(v) for k, v in strategy_mass.head(5).items()},
    }


def fit_response_models(
    train_frame: pd.DataFrame,
    state_cols: list[str],
    *,
    per_strategy_residual: bool,
    max_rows: int,
    max_keyword_cols: int,
    response_model_kind: str = "additive_ebm",
    response_frontier_weight_gamma: float = 3.0,
    response_frontier_weight_bandwidth: float = 0.06,
    response_balance_timestamps: bool = True,
    response_balance_strategies: bool = True,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import KBinsDiscretizer

    model_kind = str(response_model_kind or "additive_ebm")
    if model_kind not in {"additive_ebm", "hist_gradient_boosting", "xgboost"}:
        raise ValueError(f"unknown response_model_kind={model_kind!r}")

    curves = fit_rank_curves(train_frame)
    state_ood_reference = _fit_state_ood_reference(train_frame, state_cols)
    frame = _cap_response_rows(train_frame.copy(), int(max_rows))
    frame["_base_mu"] = curves.predict(frame["strategy_id"], frame["_rank"], "mu")
    frame["_base_psl"] = curves.predict(frame["strategy_id"], frame["_rank"], "psl")
    frame["_base_pto"] = curves.predict(frame["strategy_id"], frame["_rank"], "pto")
    frame["_resid_u"] = frame["_net_return"] - frame["_base_mu"]
    frame["_resid_sl"] = frame["_is_full_sl"] - frame["_base_psl"]
    frame["_resid_to"] = frame["_is_timeout"] - frame["_base_pto"]
    for axis in [c for c in state_cols if _is_state_interaction_col(c)]:
        frame[f"{axis}__x_rank"] = _safe_numeric(frame, axis).fillna(0.0) * frame["_rank"]
    dummies = pd.get_dummies(frame[["strategy_id", "side"]].astype(str), prefix=["strategy", "side"], dtype=float)
    frame = pd.concat([frame, dummies], axis=1)
    feature_cols = _candidate_feature_columns(frame, state_cols, max_keyword_cols=max_keyword_cols) + list(dummies.columns)
    feature_cols += [c for c in frame.columns if c.endswith("__x_rank")]
    feature_cols = [
        c
        for c in dict.fromkeys(feature_cols)
        if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])
    ]
    weights = _frontier_weights(
        frame,
        frontier_gamma=float(response_frontier_weight_gamma),
        frontier_bandwidth=float(response_frontier_weight_bandwidth),
        balance_timestamps=bool(response_balance_timestamps),
        balance_strategies=bool(response_balance_strategies),
    )
    weight_report = _response_weight_report(frame, weights)

    additive_bins = int(min(8, max(4, round(np.sqrt(max(len(frame), 1)) / 3))))

    def make_reg(seed: int, *, quantile: float | None = None):
        if model_kind == "hist_gradient_boosting":
            kwargs: dict[str, Any] = {
                "random_state": seed,
                "max_depth": 3,
                "max_leaf_nodes": 15,
                "max_iter": 80,
                "learning_rate": 0.06,
                "min_samples_leaf": max(20, int(len(frame) * 0.02)),
                "l2_regularization": 0.02,
                "early_stopping": True,
                "validation_fraction": 0.15,
                "n_iter_no_change": 8,
            }
            if quantile is not None:
                kwargs["loss"] = "quantile"
                kwargs["quantile"] = float(quantile)
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", HistGradientBoostingRegressor(**kwargs)),
                ]
            )
        if model_kind == "xgboost":
            try:
                from xgboost import XGBRegressor
            except Exception as exc:  # pragma: no cover - environment dependent.
                raise RuntimeError("response_model_kind='xgboost' requires xgboost to be installed") from exc
            kwargs = {
                "objective": "reg:pseudohubererror",
                "random_state": seed,
                "n_estimators": 90,
                "max_depth": 3,
                "learning_rate": 0.04,
                "min_child_weight": max(5.0, float(len(frame)) * 0.01),
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
                "tree_method": "hist",
                "n_jobs": 1,
                "verbosity": 0,
            }
            if quantile is not None:
                kwargs["objective"] = "reg:quantileerror"
                kwargs["quantile_alpha"] = float(quantile)
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", XGBRegressor(**kwargs)),
                ]
            )
        # Local EBM-style fallback: each numeric feature is discretized into
        # training-fitted quantile bins, then a linear model learns additive
        # shape effects over those bins. This keeps the response layer
        # deterministic and inspectable without introducing a heavyweight
        # optional dependency on interpret/pygam.
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "binner",
                    KBinsDiscretizer(
                        n_bins=additive_bins,
                        encode="onehot",
                        strategy="quantile",
                        quantile_method="linear",
                    ),
                ),
                ("model", Ridge(alpha=2.0)),
            ]
        )

    def fit_reg(model: Any, X_in: pd.DataFrame, y: pd.Series | np.ndarray, w: np.ndarray) -> None:
        model.fit(X_in, y, model__sample_weight=np.asarray(w, dtype=float))

    def target_weights(
        y: pd.Series | np.ndarray,
        *,
        lower_tail: bool = False,
        base_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        w = np.asarray(weights if base_weights is None else base_weights, dtype=float).copy()
        if not lower_tail or model_kind != "additive_ebm":
            return w
        arr = np.asarray(y, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size < 10:
            return w
        q = float(np.nanquantile(finite, 0.25))
        w *= np.where(arr <= q, 3.0, 1.0)
        return w / max(float(np.mean(w)), 1e-12)

    shared = {
        "eu_mean": make_reg(301),
        "eu_q10": make_reg(307, quantile=0.10),
    }
    risk = {
        "full_sl": make_reg(311),
        "timeout": make_reg(313),
    }
    targets = {
        "eu_mean": "_resid_u",
        "eu_q10": "_resid_u",
    }
    risk_targets = {
        "full_sl": "_resid_sl",
        "timeout": "_resid_to",
    }
    X = frame[feature_cols]
    for name, model in shared.items():
        fit_reg(
            model,
            X,
            frame[targets[name]],
            target_weights(frame[targets[name]], lower_tail=name == "eu_q10"),
        )
    for name, model in risk.items():
        fit_reg(model, X, frame[risk_targets[name]], target_weights(frame[risk_targets[name]]))
    risk_baseline = {
        "full_sl_residual_mean": float(np.average(frame["_resid_sl"], weights=weights)),
        "timeout_residual_mean": float(np.average(frame["_resid_to"], weights=weights)),
        "residual_scale": 1.0,
    }

    residual_models: dict[str, dict[str, Any]] = {}
    if per_strategy_residual:
        shared_pred = {
            name: np.asarray(model.predict(X), dtype=float)
            for name, model in shared.items()
        }
        for strategy, idx in frame.groupby("strategy_id").groups.items():
            rows = np.asarray(list(idx), dtype=int)
            if len(rows) < 250:
                continue
            residual_models[strategy] = {}
            for name in ("eu_q10",):
                resid_target = frame.iloc[rows][targets[name]].to_numpy(dtype=float) - shared_pred[name][rows]
                if float(np.nanstd(resid_target)) < 1e-8:
                    continue
                model = make_reg(400 + len(residual_models) * 17 + len(name))
                fit_reg(
                    model,
                    X.iloc[rows],
                    resid_target,
                    target_weights(resid_target, lower_tail=name == "eu_q10", base_weights=weights[rows]),
                )
                residual_models[strategy][name] = model

    return {
        "response_model_kind": model_kind,
        "curves": curves,
        "shared": shared,
        "risk": risk,
        "risk_baseline": risk_baseline,
        "residual": residual_models,
        "dummy_columns": list(dummies.columns),
        "state_ood_reference": state_ood_reference,
    }, feature_cols, {
        "train_rows": int(len(frame)),
        "raw_train_rows": int(len(train_frame)),
        "feature_count": int(len(feature_cols)),
        "response_model_kind": model_kind,
        "response_model_family": (
            "pooled_additive_binned_response_model"
            if model_kind == "additive_ebm"
            else (
                "rank_curve_plus_xgboost_response"
                if model_kind == "xgboost"
                else "rank_curve_plus_hist_gradient_boosting_response"
            )
        ),
        "additive_bins": int(additive_bins) if model_kind == "additive_ebm" else None,
        "additive_lower_tail_weight_multiplier": 3.0 if model_kind == "additive_ebm" else None,
        "per_strategy_residual": bool(per_strategy_residual),
        "residual_strategy_count": int(len(residual_models)),
        "risk_model": (
            "rank_curve_plus_additive_ebm_response"
            if model_kind == "additive_ebm"
            else (
                "rank_curve_plus_xgboost_response"
                if model_kind == "xgboost"
                else "rank_curve_plus_excess_risk_regressors"
            )
        ),
        "response_weighting": {
            "timestamp_balanced": bool(response_balance_timestamps),
            "strategy_balanced": bool(response_balance_strategies),
            "frontier_gamma": float(response_frontier_weight_gamma),
            "frontier_bandwidth": float(response_frontier_weight_bandwidth),
            **weight_report,
        },
        "risk_baseline": risk_baseline,
        "state_ood_reference": {
            "enabled": bool(state_ood_reference.get("enabled")),
            "column_count": int(len(state_ood_reference.get("columns", []))),
            "score_cutoff": state_ood_reference.get("score_cutoff"),
            "score_quantile": state_ood_reference.get("score_quantile"),
            "train_score_median": state_ood_reference.get("train_score_median"),
            "train_score_q95": state_ood_reference.get("train_score_q95"),
            "train_score_q99": state_ood_reference.get("train_score_q99"),
            "mean_state_coverage": state_ood_reference.get("mean_state_coverage"),
            "candidate_column_count": state_ood_reference.get("candidate_column_count"),
            "dropped_column_count": state_ood_reference.get("dropped_column_count"),
            "min_non_na_fraction": state_ood_reference.get("min_non_na_fraction"),
            "min_scale": state_ood_reference.get("min_scale"),
            "reason": state_ood_reference.get("reason"),
        },
    }


def _prepare_response_matrix(
    frame: pd.DataFrame,
    feature_cols: list[str],
    dummy_cols: list[str],
    state_cols: list[str],
) -> pd.DataFrame:
    out = frame.copy()
    for axis in [c for c in state_cols if _is_state_interaction_col(c)]:
        if axis in out.columns:
            out[f"{axis}__x_rank"] = _safe_numeric(out, axis).fillna(0.0) * _safe_numeric(out, "_rank").fillna(0.0)
    dummies = pd.get_dummies(out[["strategy_id", "side"]].astype(str), prefix=["strategy", "side"], dtype=float)
    for col in dummy_cols:
        if col not in dummies:
            dummies[col] = 0.0
    out = pd.concat([out, dummies[dummy_cols]], axis=1)
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
    return out[feature_cols]


def _row_finite_coverage(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(1.0, index=frame.index, dtype=float)
    numeric = frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.notna().mean(axis=1).astype(float)


def _fit_state_ood_reference(
    frame: pd.DataFrame,
    state_cols: list[str],
    *,
    quantile: float = 0.99,
    min_non_na_fraction: float = 0.50,
    min_scale: float = 1e-4,
) -> dict[str, Any]:
    cols = [
        c
        for c in state_cols
        if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not cols:
        return {"enabled": False, "columns": [], "reason": "no_state_columns"}
    data_all = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_fraction = data_all.notna().mean(axis=0)
    nunique = data_all.nunique(dropna=True)
    q75_all = data_all.quantile(0.75)
    q25_all = data_all.quantile(0.25)
    iqr_scale_all = ((q75_all - q25_all).abs() / 1.349).replace([np.inf, -np.inf], np.nan)
    std_scale_all = data_all.std(axis=0, ddof=0).replace([np.inf, -np.inf], np.nan)
    scale_all = iqr_scale_all.where(iqr_scale_all > 1e-8, std_scale_all).replace([np.inf, -np.inf], np.nan)
    keep_mask = (
        finite_fraction.ge(float(min_non_na_fraction))
        & nunique.gt(1)
        & scale_all.ge(float(min_scale))
    )
    dropped = {
        "missing_or_sparse": sorted(finite_fraction.index[finite_fraction.lt(float(min_non_na_fraction))].astype(str).tolist()),
        "constant": sorted(nunique.index[nunique.le(1)].astype(str).tolist()),
        "low_variance": sorted(scale_all.index[scale_all.lt(float(min_scale)).fillna(True)].astype(str).tolist()),
    }
    cols = [str(c) for c in scale_all.index[keep_mask].tolist()]
    if not cols:
        return {
            "enabled": False,
            "columns": [],
            "reason": "no_variable_state_columns",
            "candidate_column_count": int(len(data_all.columns)),
            "dropped_columns": dropped,
            "min_non_na_fraction": float(min_non_na_fraction),
            "min_scale": float(min_scale),
        }
    data = data_all[cols]
    coverage = data.notna().mean(axis=1)
    med = data.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    q75 = data.quantile(0.75)
    q25 = data.quantile(0.25)
    iqr_scale = ((q75 - q25).abs() / 1.349).replace([np.inf, -np.inf], np.nan)
    std_scale = data.std(axis=0, ddof=0).replace([np.inf, -np.inf], np.nan)
    scale = iqr_scale.where(iqr_scale > 1e-8, std_scale).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    scale = scale.clip(lower=1e-6)
    z = (data.fillna(med) - med) / scale
    score = np.sqrt(np.nanmean(np.square(z.to_numpy(dtype=float)), axis=1))
    score = pd.Series(score, index=frame.index).where(coverage > 0.0)
    finite = score.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return {
            "enabled": False,
            "columns": cols,
            "reason": "no_finite_training_scores",
            "median": med.to_dict(),
            "scale": scale.to_dict(),
            "candidate_column_count": int(len(data_all.columns)),
            "dropped_columns": dropped,
        }
    q = float(np.clip(float(quantile), 0.50, 0.999))
    cutoff = float(finite.quantile(q))
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        cutoff = float(finite.max())
    return {
        "enabled": True,
        "columns": cols,
        "median": med.to_dict(),
        "scale": scale.to_dict(),
        "score_cutoff": cutoff,
        "score_quantile": q,
        "train_score_median": float(finite.median()),
        "train_score_q95": float(finite.quantile(0.95)),
        "train_score_q99": float(finite.quantile(0.99)),
        "train_rows": int(len(frame)),
        "finite_score_rows": int(len(finite)),
        "mean_state_coverage": float(coverage.mean()),
        "candidate_column_count": int(len(data_all.columns)),
        "dropped_column_count": int(len(data_all.columns) - len(cols)),
        "dropped_columns": dropped,
        "min_non_na_fraction": float(min_non_na_fraction),
        "min_scale": float(min_scale),
    }


def _score_state_ood(frame: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    cols = list(reference.get("columns") or [])
    if not reference.get("enabled") or not cols:
        return pd.DataFrame(
            {
                "state_ood_score": np.nan,
                "state_ood_cutoff": np.nan,
                "state_ood_flag": False,
            },
            index=frame.index,
        )
    data = frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    coverage = data.notna().mean(axis=1)
    med = pd.Series(reference.get("median", {}), dtype=float).reindex(cols).fillna(0.0)
    scale = pd.Series(reference.get("scale", {}), dtype=float).reindex(cols).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1e-6)
    z = (data.fillna(med) - med) / scale
    score = np.sqrt(np.nanmean(np.square(z.to_numpy(dtype=float)), axis=1))
    cutoff = float(reference.get("score_cutoff", np.inf))
    flag = np.isfinite(score) & np.isfinite(cutoff) & (score > cutoff)
    flag |= coverage.to_numpy(dtype=float) <= 0.0
    return pd.DataFrame(
        {
            "state_ood_score": score,
            "state_ood_cutoff": cutoff,
            "state_ood_flag": flag,
        },
        index=frame.index,
    )


def _predict_risk_signal(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return a continuous risk signal from either classifier or regressor models."""

    try:
        proba = model.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] > 1:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba, dtype=float).reshape(-1)
    except AttributeError:
        return np.asarray(model.predict(X), dtype=float)


def predict_response(
    models: dict[str, Any],
    frame: pd.DataFrame,
    feature_cols: list[str],
    state_cols: list[str],
) -> pd.DataFrame:
    curves: RankOutcomeCurves = models["curves"]
    X = _prepare_response_matrix(frame, feature_cols, models["dummy_columns"], state_cols)
    out = frame[["timestamp", "strategy_id", "head", "_rank", "_threshold"]].copy()
    out["state_feature_coverage"] = _row_finite_coverage(frame, state_cols).to_numpy(dtype=float)
    out["response_feature_coverage"] = _row_finite_coverage(X, feature_cols).to_numpy(dtype=float)
    for reliability_col in ("state_input_coverage", "state_low_input_coverage"):
        if reliability_col in frame.columns:
            out[reliability_col] = pd.to_numeric(
                frame[reliability_col],
                errors="coerce",
            ).to_numpy(dtype=float)
    ood = _score_state_ood(frame, models.get("state_ood_reference", {}))
    for col in ood.columns:
        out[col] = ood[col].to_numpy()
    out["base_mu"] = curves.predict(frame["strategy_id"], frame["_rank"], "mu")
    out["base_psl"] = curves.predict(frame["strategy_id"], frame["_rank"], "psl")
    out["base_pto"] = curves.predict(frame["strategy_id"], frame["_rank"], "pto")
    for name, model in models["shared"].items():
        out[f"pred_{name}"] = np.asarray(model.predict(X), dtype=float)
    for name, model in models.get("risk", {}).items():
        out[f"pred_excess_{name}"] = _predict_risk_signal(model, X)
    for strategy, mods in models["residual"].items():
        mask = frame["strategy_id"].astype(str).eq(strategy)
        if not mask.any():
            continue
        for name, model in mods.items():
            out.loc[mask, f"pred_{name}"] += np.asarray(model.predict(X.loc[mask]), dtype=float)
    out["pred_mean_utility"] = out["base_mu"] + out["pred_eu_mean"]
    out["pred_lcb_utility"] = out["base_mu"] + out["pred_eu_q10"]
    risk_scale = float(models.get("risk_baseline", {}).get("residual_scale", 1.0))
    excess_full_sl = np.asarray(out.get("pred_excess_full_sl", np.zeros(len(out))), dtype=float) * risk_scale
    excess_timeout = np.asarray(out.get("pred_excess_timeout", np.zeros(len(out))), dtype=float) * risk_scale
    excess_full_sl = np.clip(excess_full_sl, -out["base_psl"].to_numpy(dtype=float), 1.0 - out["base_psl"].to_numpy(dtype=float))
    excess_timeout = np.clip(excess_timeout, -out["base_pto"].to_numpy(dtype=float), 1.0 - out["base_pto"].to_numpy(dtype=float))
    out["pred_full_sl"] = (out["base_psl"] + excess_full_sl).clip(0.0, 1.0)
    out["pred_timeout"] = (out["base_pto"] + excess_timeout).clip(0.0, 1.0)
    out["pred_excess_full_sl"] = out["pred_full_sl"] - out["base_psl"]
    out["pred_excess_timeout"] = out["pred_timeout"] - out["base_pto"]
    return out


def threshold_schedule(
    eval_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    curves: RankOutcomeCurves,
    *,
    delta_max: float,
    max_down_step: float,
    relax_alpha: float,
    controller_mode: str = "rank_grid",
    min_lcb_utility: float = 0.0,
    use_timeout_cap: bool = False,
    min_action_edge: float = 0.0,
    winner_sacrifice_multiplier: float = 1.0,
    min_removed_full_sl: float = 0.0,
    max_removed_timeout: float = 1.0,
    enabled_heads: set[str] | None = None,
    min_prediction_coverage: float = 0.80,
    min_usable_candidates: int = 1,
    min_frontier_candidates: int = 1,
    max_state_ood_score: float | None = None,
    accepted_decision_keys: set[tuple[Any, ...]] | None = None,
) -> pd.DataFrame:
    work_cols = ["timestamp", "strategy_id", "head", "_rank", "_threshold"]
    for col in DECISION_KEY_COLS:
        if col in eval_frame.columns and col not in work_cols:
            work_cols.append(col)
    work = eval_frame[work_cols].copy()
    pred_cols = ["pred_mean_utility", "pred_lcb_utility", "pred_full_sl", "pred_timeout"]
    optional_pred_cols = [
        "state_feature_coverage",
        "response_feature_coverage",
        "state_input_coverage",
        "state_low_input_coverage",
        "state_ood_score",
        "state_ood_cutoff",
        "state_ood_flag",
    ]
    pred = predictions[[c for c in pred_cols + optional_pred_cols if c in predictions.columns]].copy()
    work = pd.concat([work.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    utility_scale = float(work["pred_lcb_utility"].abs().quantile(0.75))
    if not np.isfinite(utility_scale) or utility_scale < 1e-4:
        utility_scale = 1e-4
    rows: list[dict[str, Any]] = []
    for (ts, strategy), g in work.groupby(["timestamp", "strategy_id"], sort=True):
        head = str(g["head"].iloc[0])
        action_enabled = enabled_heads is None or head in enabled_heads
        base = float(np.nanmedian(g["_threshold"]))
        sl_cap = curves.strategy_cap(str(strategy), "psl", base, pad=0.05)
        to_cap = curves.strategy_cap(str(strategy), "pto", base, pad=0.05)
        candidates = g.loc[g["_rank"] >= base].sort_values("_rank")
        frontier_upper_rank = float(min(1.01, base + float(delta_max)))
        min_frontier_candidate_count = max(0, int(min_frontier_candidates))
        target = base
        severity = 0.0
        reason = "base"
        lcb_q25 = np.nan
        sl_mean = np.nan
        to_mean = np.nan
        tail_candidate_count = 0
        suppressed_candidate_count = 0
        predicted_removed_loss_avoided = 0.0
        predicted_removed_winner_sacrificed = 0.0
        predicted_action_edge = 0.0
        predicted_removed_full_sl_mean = np.nan
        predicted_removed_timeout_mean = np.nan
        accepted_frontier_candidate_count = 0
        accepted_frontier_suppressed_count = 0
        force_base_threshold = False
        selected_rank = base
        finite_pred = g[pred_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        row_prediction_ok = finite_pred.notna().all(axis=1)
        if "state_feature_coverage" in g:
            row_prediction_ok &= pd.to_numeric(g["state_feature_coverage"], errors="coerce").fillna(0.0) >= float(min_prediction_coverage)
        if "response_feature_coverage" in g:
            row_prediction_ok &= pd.to_numeric(g["response_feature_coverage"], errors="coerce").fillna(0.0) >= float(min_prediction_coverage)
        ood_score = (
            pd.to_numeric(g["state_ood_score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if "state_ood_score" in g
            else pd.Series(np.nan, index=g.index)
        )
        if max_state_ood_score is not None:
            ood_cutoff = pd.Series(float(max_state_ood_score), index=g.index)
        elif "state_ood_cutoff" in g:
            ood_cutoff = pd.to_numeric(g["state_ood_cutoff"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            ood_cutoff = pd.Series(np.nan, index=g.index)
        if "state_ood_flag" in g:
            ood_flag = g["state_ood_flag"].fillna(False).astype(bool)
        else:
            ood_flag = pd.Series(False, index=g.index)
        ood_flag |= ood_score.notna() & ood_cutoff.notna() & (ood_score > ood_cutoff)
        if "state_low_input_coverage" in g:
            low_input_flag = pd.to_numeric(
                g["state_low_input_coverage"],
                errors="coerce",
            ).fillna(1.0).gt(0.0)
        else:
            low_input_flag = pd.Series(False, index=g.index)
        row_prediction_ok &= ~ood_flag
        row_prediction_ok &= ~low_input_flag
        prediction_coverage = float(row_prediction_ok.mean()) if len(row_prediction_ok) else 0.0
        mean_ood_score = float(ood_score.mean()) if ood_score.notna().any() else np.nan
        max_ood_score = float(ood_score.max()) if ood_score.notna().any() else np.nan
        mean_ood_cutoff = float(ood_cutoff.mean()) if ood_cutoff.notna().any() else np.nan
        ood_share = float(ood_flag.mean()) if len(ood_flag) else 0.0
        low_input_coverage_share = float(low_input_flag.mean()) if len(low_input_flag) else 0.0
        usable = g.loc[row_prediction_ok].copy()
        candidates = usable.loc[usable["_rank"] >= base].sort_values("_rank")
        if ood_share >= 1.0 and len(g) > 0:
            target = base
            reason = "state_ood_fallback"
            force_base_threshold = True
        elif low_input_coverage_share >= 1.0 and len(g) > 0:
            target = base
            reason = "low_input_coverage_fallback"
            force_base_threshold = True
        elif prediction_coverage < float(min_prediction_coverage):
            target = base
            reason = "insufficient_prediction_coverage"
            force_base_threshold = True
        elif len(candidates) < int(min_usable_candidates):
            # Already inactive under the existing rank contract. Do not add a
            # meaningless max penalty; the controller should only change
            # thresholds when it has live candidates whose conditional response
            # can be evaluated.
            target = base
            reason = "no_usable_candidate_above_base"
            force_base_threshold = True
        elif controller_mode in {
            "rank_grid",
            "action_aware_rank_grid",
            "frontier_rank_grid",
            "frontier_action_rank_grid",
            "accepted_frontier_action_rank_grid",
        }:
            max_allowed = frontier_upper_rank
            accepted_frontier_aware = controller_mode == "accepted_frontier_action_rank_grid"
            frontier_aware = controller_mode in {
                "frontier_rank_grid",
                "frontier_action_rank_grid",
                "accepted_frontier_action_rank_grid",
            }
            action_aware = controller_mode in {
                "action_aware_rank_grid",
                "frontier_action_rank_grid",
                "accepted_frontier_action_rank_grid",
            }
            actual_frontier_candidates = candidates.loc[candidates["_rank"] <= max_allowed]
            accepted_key_filter_active = bool(
                accepted_frontier_aware and accepted_decision_keys is not None
            )
            accepted_frontier_candidates = actual_frontier_candidates
            accepted_frontier_missing_key_cols = False
            if accepted_key_filter_active:
                try:
                    accepted_mask = _allowed_decision_key_mask(
                        actual_frontier_candidates,
                        accepted_decision_keys or set(),
                    )
                    accepted_frontier_candidates = actual_frontier_candidates.loc[accepted_mask].copy()
                except KeyError:
                    accepted_frontier_missing_key_cols = True
                    accepted_frontier_candidates = actual_frontier_candidates.iloc[0:0].copy()
            action_frontier_candidates = (
                accepted_frontier_candidates if accepted_key_filter_active else actual_frontier_candidates
            )
            accepted_frontier_candidate_count = int(len(accepted_frontier_candidates)) if accepted_key_filter_active else int(len(actual_frontier_candidates))
            accepted_frontier_suppressed_count = 0
            if len(action_frontier_candidates) < min_frontier_candidate_count:
                # A threshold raise can only suppress rows in the marginal
                # frontier [base, base + delta_max]. When that band is empty
                # or too thin, the model-side state estimate is not actionable
                # for this strategy timestamp, so fail closed to the base
                # threshold instead of applying a broad scaled penalty.
                selected_tail = candidates
                selected_removed = candidates.tail(0)
                selected_rank = base
                tail_candidate_count = int(len(selected_tail))
                suppressed_candidate_count = 0
                lcb_q25 = (
                    float(selected_tail["pred_lcb_utility"].quantile(0.25))
                    if len(selected_tail)
                    else np.nan
                )
                sl_mean = float(selected_tail["pred_full_sl"].mean()) if len(selected_tail) else np.nan
                to_mean = float(selected_tail["pred_timeout"].mean()) if len(selected_tail) else np.nan
                predicted_removed_loss_avoided = 0.0
                predicted_removed_winner_sacrificed = 0.0
                predicted_action_edge = 0.0
                predicted_removed_full_sl_mean = np.nan
                predicted_removed_timeout_mean = np.nan
                target = base
                severity = 0.0
                if accepted_key_filter_active and accepted_frontier_missing_key_cols:
                    reason = "missing_accepted_frontier_decision_keys"
                elif accepted_key_filter_active:
                    reason = "no_baseline_accepted_candidate_in_frontier"
                else:
                    reason = "insufficient_frontier_candidate_support"
                force_base_threshold = True
            else:
                frontier_candidates = action_frontier_candidates
                if frontier_candidates.empty:
                    frontier_candidates = candidates
                grid_ranks = candidates.loc[candidates["_rank"] <= max_allowed, "_rank"].to_numpy(dtype=float)
                if accepted_key_filter_active:
                    grid_ranks = action_frontier_candidates["_rank"].to_numpy(dtype=float)
                if accepted_frontier_aware and grid_ranks.size:
                    # The policy accepts rows with rank >= threshold. A grid
                    # point equal to a candidate rank does not suppress that
                    # candidate, so include the next representable float above
                    # each candidate rank. This lets the controller model a
                    # direct one-row accepted-frontier suppression instead of
                    # relying on downstream occupancy/path side effects.
                    grid_ranks = np.concatenate([grid_ranks, np.nextafter(grid_ranks, np.inf)])
                    grid_ranks = np.minimum(grid_ranks, max_allowed)
                threshold_grid = np.unique(
                    np.concatenate(
                        [
                            np.array([base], dtype=float),
                            grid_ranks,
                            np.array([max_allowed], dtype=float),
                        ]
                    )
                )
                threshold_grid = threshold_grid[np.isfinite(threshold_grid)]
                threshold_grid.sort()
                selected_tail: pd.DataFrame | None = None
                selected_removed: pd.DataFrame | None = None
                selected_rank = base
                for rank_threshold in threshold_grid:
                    tail = candidates.loc[candidates["_rank"] >= rank_threshold]
                    if tail.empty and not accepted_frontier_aware:
                        continue
                    removed = candidates.loc[candidates["_rank"] < rank_threshold]
                    if frontier_aware:
                        tail_metric = frontier_candidates.loc[frontier_candidates["_rank"] >= rank_threshold]
                        removed_metric = frontier_candidates.loc[frontier_candidates["_rank"] < rank_threshold]
                        if tail_metric.empty:
                            # The marginal band is fully suppressed; evaluate the
                            # surviving high-confidence tail rather than rejecting
                            # the threshold mechanically.
                            tail_metric = tail
                    else:
                        tail_metric = tail
                        removed_metric = removed
                    if tail_metric.empty and accepted_frontier_aware and not removed_metric.empty:
                        # Direct accepted-frontier suppression can deliberately
                        # remove every currently accepted row for a strategy
                        # timestamp. In that case there is no surviving tail to
                        # score; the action is judged entirely on the predicted
                        # utility/risk of the directly removed rows below.
                        tail_lcb_q25 = float(min_lcb_utility)
                        tail_sl_mean = 0.0
                        tail_to_mean = 0.0
                    else:
                        tail_lcb_q25 = float(tail_metric["pred_lcb_utility"].quantile(0.25))
                        tail_sl_mean = float(tail_metric["pred_full_sl"].mean())
                        tail_to_mean = float(tail_metric["pred_timeout"].mean())
                    timeout_ok = (tail_to_mean <= to_cap) if use_timeout_cap else True
                    if action_aware:
                        removed_loss_avoided = float((-np.clip(removed_metric["pred_lcb_utility"].to_numpy(dtype=float), None, 0.0)).sum())
                        removed_winner_sacrificed = float(np.clip(removed_metric["pred_mean_utility"].to_numpy(dtype=float), 0.0, None).sum())
                        action_edge = removed_loss_avoided - float(winner_sacrifice_multiplier) * removed_winner_sacrificed
                        if removed_metric.empty:
                            removed_full_sl_mean = np.nan
                            removed_timeout_mean = np.nan
                        else:
                            removed_full_sl_mean = float(removed_metric["pred_full_sl"].mean())
                            removed_timeout_mean = float(removed_metric["pred_timeout"].mean())
                        removed_risk_ok = (
                            rank_threshold <= base + 1e-9
                            or (
                                (not accepted_frontier_aware or len(removed_metric) > 0)
                                and
                                np.isfinite(removed_full_sl_mean)
                                and np.isfinite(removed_timeout_mean)
                                and removed_full_sl_mean >= float(min_removed_full_sl)
                                and removed_timeout_mean <= float(max_removed_timeout)
                            )
                        )
                        action_ok = (
                            rank_threshold <= base + 1e-9
                            or (action_edge > float(min_action_edge) and removed_risk_ok)
                        )
                    else:
                        removed_loss_avoided = 0.0
                        removed_winner_sacrificed = 0.0
                        action_edge = 0.0
                        removed_full_sl_mean = np.nan
                        removed_timeout_mean = np.nan
                        action_ok = True
                    if (
                        tail_lcb_q25 >= float(min_lcb_utility)
                        and tail_sl_mean <= sl_cap
                        and timeout_ok
                        and action_ok
                    ):
                        selected_tail = tail_metric
                        selected_removed = removed_metric if frontier_aware else removed
                        selected_rank = float(rank_threshold)
                        predicted_removed_loss_avoided = removed_loss_avoided
                        predicted_removed_winner_sacrificed = removed_winner_sacrificed
                        predicted_action_edge = action_edge
                        predicted_removed_full_sl_mean = removed_full_sl_mean
                        predicted_removed_timeout_mean = removed_timeout_mean
                        break
                if selected_tail is None:
                    if action_aware or accepted_frontier_aware:
                        selected_rank = base
                        selected_tail = candidates
                        selected_removed = candidates.tail(0)
                        reason = (
                            f"{controller_mode}_no_direct_positive_edge"
                            if accepted_frontier_aware
                            else f"{controller_mode}_no_positive_edge"
                        )
                        predicted_removed_full_sl_mean = np.nan
                        predicted_removed_timeout_mean = np.nan
                    else:
                        base_metric = frontier_candidates if frontier_aware else candidates
                        base_lcb_q25 = float(base_metric["pred_lcb_utility"].quantile(0.25))
                        base_utility_mean = float(base_metric["pred_mean_utility"].mean())
                        base_sl_mean = float(base_metric["pred_full_sl"].mean())
                        base_to_mean = float(base_metric["pred_timeout"].mean())
                        utility_penalty = np.clip(
                            (float(min_lcb_utility) - base_lcb_q25) / utility_scale,
                            0.0,
                            1.0,
                        )
                        sl_penalty = np.clip(
                            (base_sl_mean - sl_cap) / max(1.0 - sl_cap, 0.10),
                            0.0,
                            1.0,
                        )
                        to_penalty = np.clip(
                            (base_to_mean - to_cap) / max(1.0 - to_cap, 0.10),
                            0.0,
                            1.0,
                        )
                        timeout_component = 0.5 * to_penalty if use_timeout_cap else 0.0
                        severity = float(np.clip(max(utility_penalty, sl_penalty, timeout_component), 0.0, 1.0))
                        if base_utility_mean > 0.0 and base_sl_mean <= sl_cap:
                            severity *= 0.5
                        selected_rank = float(np.clip(base + float(delta_max) * severity, base, max_allowed))
                        selected_tail = (
                            frontier_candidates.loc[frontier_candidates["_rank"] >= selected_rank]
                            if frontier_aware
                            else candidates.loc[candidates["_rank"] >= selected_rank]
                        )
                        selected_removed = (
                            frontier_candidates.loc[frontier_candidates["_rank"] < selected_rank]
                            if frontier_aware
                            else candidates.loc[candidates["_rank"] < selected_rank]
                        )
                        if selected_tail.empty:
                            selected_tail = candidates.loc[candidates["_rank"] >= selected_rank]
                        predicted_removed_loss_avoided = float(
                            (-np.clip(selected_removed["pred_lcb_utility"].to_numpy(dtype=float), None, 0.0)).sum()
                        )
                        predicted_removed_winner_sacrificed = float(
                            np.clip(selected_removed["pred_mean_utility"].to_numpy(dtype=float), 0.0, None).sum()
                        )
                        predicted_action_edge = (
                            predicted_removed_loss_avoided
                            - float(winner_sacrifice_multiplier) * predicted_removed_winner_sacrificed
                        )
                        predicted_removed_full_sl_mean = (
                            float(selected_removed["pred_full_sl"].mean())
                            if len(selected_removed)
                            else np.nan
                        )
                        predicted_removed_timeout_mean = (
                            float(selected_removed["pred_timeout"].mean())
                            if len(selected_removed)
                            else np.nan
                        )
                        reason = f"{controller_mode}_scaled_no_feasible"
                else:
                    if selected_rank <= base + 1e-9:
                        reason = f"{controller_mode}_constraints_satisfied"
                    elif action_aware:
                        reason = f"{controller_mode}_positive_edge_penalty"
                    else:
                        reason = f"{controller_mode}_penalty"
                tail_candidate_count = int(len(selected_tail))
                if selected_removed is None:
                    selected_removed = (
                        frontier_candidates.loc[frontier_candidates["_rank"] < selected_rank]
                        if frontier_aware
                        else candidates.loc[candidates["_rank"] < selected_rank]
                    )
                suppressed_candidate_count = int((candidates["_rank"] < selected_rank).sum())
                if accepted_key_filter_active:
                    accepted_frontier_suppressed_count = int(
                        (action_frontier_candidates["_rank"] < selected_rank).sum()
                    )
                else:
                    accepted_frontier_suppressed_count = suppressed_candidate_count
                if (
                    accepted_frontier_aware
                    and selected_rank > base + 1e-9
                    and accepted_frontier_suppressed_count <= 0
                ):
                    selected_rank = base
                    selected_tail = candidates
                    selected_removed = candidates.tail(0)
                    suppressed_candidate_count = 0
                    accepted_frontier_suppressed_count = 0
                    target = base
                    severity = 0.0
                    reason = f"{controller_mode}_no_direct_suppression"
                    predicted_removed_loss_avoided = 0.0
                    predicted_removed_winner_sacrificed = 0.0
                    predicted_action_edge = 0.0
                    predicted_removed_full_sl_mean = np.nan
                    predicted_removed_timeout_mean = np.nan
                lcb_q25 = float(selected_tail["pred_lcb_utility"].quantile(0.25)) if len(selected_tail) else np.nan
                sl_mean = float(selected_tail["pred_full_sl"].mean()) if len(selected_tail) else np.nan
                to_mean = float(selected_tail["pred_timeout"].mean()) if len(selected_tail) else np.nan
                target = float(np.clip(selected_rank, base, max_allowed))
                utility_penalty = np.clip((float(min_lcb_utility) - (lcb_q25 if np.isfinite(lcb_q25) else -utility_scale)) / utility_scale, 0.0, 1.0)
                sl_penalty = np.clip(((sl_mean if np.isfinite(sl_mean) else 1.0) - sl_cap) / max(1.0 - sl_cap, 0.10), 0.0, 1.0)
                to_penalty = np.clip(((to_mean if np.isfinite(to_mean) else 1.0) - to_cap) / max(1.0 - to_cap, 0.10), 0.0, 1.0)
                timeout_component = 0.5 * to_penalty if use_timeout_cap else 0.0
                severity = float(
                    np.clip(
                        max((target - base) / max(float(delta_max), 1e-9), 0.0, utility_penalty, sl_penalty, timeout_component),
                        0.0,
                        1.0,
                    )
                )
        else:
            lcb_q25 = float(candidates["pred_lcb_utility"].quantile(0.25))
            sl_mean = float(candidates["pred_full_sl"].mean())
            to_mean = float(candidates["pred_timeout"].mean())
            utility_penalty = np.clip((-lcb_q25) / utility_scale, 0.0, 1.0)
            sl_penalty = np.clip((sl_mean - sl_cap) / max(1.0 - sl_cap, 0.10), 0.0, 1.0)
            to_penalty = np.clip((to_mean - to_cap) / max(1.0 - to_cap, 0.10), 0.0, 1.0)
            severity = float(np.clip((utility_penalty + sl_penalty + 0.5 * to_penalty) / 2.5, 0.0, 1.0))
            target = base + float(delta_max) * severity
            if severity <= 1e-6:
                reason = "risk_within_cap"
            elif utility_penalty >= max(sl_penalty, to_penalty):
                reason = "utility_lcb_penalty"
            elif sl_penalty >= to_penalty:
                reason = "full_sl_penalty"
            else:
                reason = "timeout_penalty"
            tail_candidate_count = int(len(candidates))
        if not action_enabled:
            target = base
            severity = 0.0
            reason = "head_not_enabled_for_threshold_action"
            suppressed_candidate_count = 0
            predicted_removed_loss_avoided = 0.0
            predicted_removed_winner_sacrificed = 0.0
            predicted_action_edge = 0.0
            predicted_removed_full_sl_mean = np.nan
            predicted_removed_timeout_mean = np.nan
            force_base_threshold = True
        rows.append(
            {
                "timestamp": ts,
                "strategy_id": strategy,
                "head": head,
                "base_threshold": base,
                "raw_state_threshold": float(target),
                "controller_mode": controller_mode,
                "threshold_action_enabled": bool(action_enabled),
                "force_base_threshold": bool(force_base_threshold),
                "sl_cap": sl_cap,
                "timeout_cap": to_cap,
                "risk_severity": severity,
                "controller_reason": reason,
                "prediction_coverage": prediction_coverage,
                "min_prediction_coverage": float(min_prediction_coverage),
                "state_ood_score_mean": mean_ood_score,
                "state_ood_score_max": max_ood_score,
                "state_ood_cutoff": mean_ood_cutoff,
                "state_ood_share": ood_share,
                "state_low_input_coverage_share": low_input_coverage_share,
                "mean_pred_utility": float(g["pred_mean_utility"].mean()),
                "mean_pred_lcb": float(g["pred_lcb_utility"].mean()),
                "mean_pred_full_sl": float(g["pred_full_sl"].mean()),
                "mean_pred_timeout": float(g["pred_timeout"].mean()),
                "base_candidate_count": int(len(candidates)),
                "frontier_candidate_count": int(
                    len(candidates.loc[candidates["_rank"] <= frontier_upper_rank])
                    if not candidates.empty
                    else 0
                ),
                "min_frontier_candidate_count": int(min_frontier_candidate_count),
                "frontier_upper_rank": float(frontier_upper_rank),
                "tail_candidate_count": tail_candidate_count,
                "suppressed_candidate_count": suppressed_candidate_count,
                "tail_lcb_q25": float(lcb_q25) if np.isfinite(lcb_q25) else np.nan,
                "tail_pred_full_sl": float(sl_mean) if np.isfinite(sl_mean) else np.nan,
                "tail_pred_timeout": float(to_mean) if np.isfinite(to_mean) else np.nan,
                "predicted_removed_loss_avoided": float(predicted_removed_loss_avoided),
                "predicted_removed_winner_sacrificed": float(predicted_removed_winner_sacrificed),
                "predicted_action_edge": float(predicted_action_edge),
                "predicted_removed_full_sl_mean": (
                    float(predicted_removed_full_sl_mean)
                    if np.isfinite(predicted_removed_full_sl_mean)
                    else np.nan
                ),
                "predicted_removed_timeout_mean": (
                    float(predicted_removed_timeout_mean)
                    if np.isfinite(predicted_removed_timeout_mean)
                    else np.nan
                ),
                "accepted_frontier_direct_required": bool(controller_mode == "accepted_frontier_action_rank_grid"),
                "accepted_frontier_key_filter_active": bool(
                    controller_mode == "accepted_frontier_action_rank_grid"
                    and accepted_decision_keys is not None
                ),
                "accepted_frontier_candidate_count": int(
                    accepted_frontier_candidate_count
                ),
                "accepted_frontier_suppressed_count": int(
                    accepted_frontier_suppressed_count
                ),
                "direct_suppression_threshold_floor": (
                    float(selected_rank)
                    if controller_mode == "accepted_frontier_action_rank_grid"
                    and selected_rank > base + 1e-9
                    and accepted_frontier_suppressed_count > 0
                    else np.nan
                ),
                "action_edge_per_suppressed": float(
                    predicted_action_edge / suppressed_candidate_count
                    if suppressed_candidate_count > 0
                    else 0.0
                ),
            }
        )
    sched = pd.DataFrame(rows).sort_values(["strategy_id", "timestamp"])
    smoothed: list[pd.DataFrame] = []
    for strategy, g in sched.groupby("strategy_id", sort=False):
        prev = None
        vals = []
        reasons = []
        suppressed_counts = []
        accepted_suppressed_counts = []
        predicted_loss_avoided_values = []
        predicted_winner_sacrificed_values = []
        predicted_action_edge_values = []
        predicted_removed_full_sl_values = []
        predicted_removed_timeout_values = []
        direct_floor_values = []
        risk_severity_values = []
        action_edge_per_suppressed_values = []
        for _, row in g.iterrows():
            target = float(row["raw_state_threshold"])
            base = float(row["base_threshold"])
            reason_value = row.get("controller_reason")
            suppressed_value = int(row.get("suppressed_candidate_count", 0) or 0)
            accepted_suppressed_value = int(row.get("accepted_frontier_suppressed_count", 0) or 0)
            predicted_loss_avoided = float(row.get("predicted_removed_loss_avoided", 0.0) or 0.0)
            predicted_winner_sacrificed = float(row.get("predicted_removed_winner_sacrificed", 0.0) or 0.0)
            predicted_edge = float(row.get("predicted_action_edge", 0.0) or 0.0)
            predicted_removed_full_sl = row.get("predicted_removed_full_sl_mean", np.nan)
            predicted_removed_timeout = row.get("predicted_removed_timeout_mean", np.nan)
            direct_floor = row.get("direct_suppression_threshold_floor", np.nan)
            risk_severity = float(row.get("risk_severity", 0.0) or 0.0)
            if bool(row.get("force_base_threshold", False)):
                cur = base
            elif prev is None:
                cur = target
            elif target > prev:
                cur = min(target, prev + max_down_step)
            else:
                cur = prev + relax_alpha * (target - prev)
            cur = float(np.clip(cur, base, min(1.01, base + delta_max)))
            if bool(row.get("accepted_frontier_direct_required", False)) and cur > base + 1e-9:
                direct_floor = float(row.get("direct_suppression_threshold_floor", np.nan))
                risk_is_rising = prev is None or target > prev
                if np.isfinite(direct_floor) and risk_is_rising:
                    # Accepted-frontier mode is only useful when the final
                    # schedule directly suppresses a baseline-accepted row.
                    # If the generic smoothing step lands below that row, jump
                    # to the minimum required floor instead of emitting a
                    # partial raise that has no executable effect. The value is
                    # still clipped to [base, base + delta_max] above.
                    cur = max(cur, min(float(target), direct_floor))
                if not np.isfinite(direct_floor) or cur + 1e-12 < direct_floor:
                    # A partial smoothed raise that does not cross a directly
                    # removable accepted-frontier row is a path/occupancy
                    # perturbation, not an accepted-trade suppression. Fail
                    # closed for this timestamp.
                    cur = base
                    reason_value = "accepted_frontier_action_rank_grid_smoothing_no_direct_suppression"
                    suppressed_value = 0
                    accepted_suppressed_value = 0
                    predicted_loss_avoided = 0.0
                    predicted_winner_sacrificed = 0.0
                    predicted_edge = 0.0
                    predicted_removed_full_sl = np.nan
                    predicted_removed_timeout = np.nan
                    direct_floor = np.nan
                    risk_severity = 0.0
            vals.append(cur)
            reasons.append(reason_value)
            suppressed_counts.append(suppressed_value)
            accepted_suppressed_counts.append(accepted_suppressed_value)
            predicted_loss_avoided_values.append(predicted_loss_avoided)
            predicted_winner_sacrificed_values.append(predicted_winner_sacrificed)
            predicted_action_edge_values.append(predicted_edge)
            predicted_removed_full_sl_values.append(predicted_removed_full_sl)
            predicted_removed_timeout_values.append(predicted_removed_timeout)
            direct_floor_values.append(direct_floor)
            risk_severity_values.append(risk_severity)
            action_edge_per_suppressed_values.append(
                float(predicted_edge / accepted_suppressed_value)
                if accepted_suppressed_value > 0
                else 0.0
            )
            prev = cur
        h = g.copy()
        h["state_threshold"] = vals
        h["controller_reason"] = reasons
        h["suppressed_candidate_count"] = suppressed_counts
        h["accepted_frontier_suppressed_count"] = accepted_suppressed_counts
        h["predicted_removed_loss_avoided"] = predicted_loss_avoided_values
        h["predicted_removed_winner_sacrificed"] = predicted_winner_sacrificed_values
        h["predicted_action_edge"] = predicted_action_edge_values
        h["predicted_removed_full_sl_mean"] = predicted_removed_full_sl_values
        h["predicted_removed_timeout_mean"] = predicted_removed_timeout_values
        h["direct_suppression_threshold_floor"] = direct_floor_values
        h["risk_severity"] = risk_severity_values
        h["action_edge_per_suppressed"] = action_edge_per_suppressed_values
        smoothed.append(h)
    return pd.concat(smoothed, ignore_index=True)


def apply_thresholds(candidates: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    sched = schedule[["timestamp", "strategy_id", "state_threshold"]].copy()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    out = candidates.merge(sched, on=["timestamp", "strategy_id"], how="left", validate="many_to_one")
    threshold = pd.to_numeric(out["state_threshold"], errors="coerce")
    base = _safe_numeric(out, "base_strategy_threshold")
    out["base_strategy_threshold"] = threshold.fillna(base).clip(lower=base, upper=1.01)
    out["deployment_rank_threshold"] = out["base_strategy_threshold"]
    return normalise_candidate_table(out.drop(columns=["state_threshold"]))


def threshold_action_audit(schedule: pd.DataFrame) -> pd.DataFrame:
    """Causal schedule-side audit of threshold-controller actions.

    This uses only the proposed threshold schedule and model-side diagnostics,
    not realized outcomes. It is suitable for live/shadow logs before trade
    labels have matured.
    """

    result_cols = [
        "scope",
        "scope_value",
        "schedule_rows",
        "timestamp_count",
        "strategy_count",
        "threshold_raised_count",
        "threshold_raised_share",
        "force_base_count",
        "force_base_share",
        "mean_base_threshold",
        "mean_state_threshold",
        "mean_threshold_delta",
        "p75_threshold_delta",
        "max_threshold_delta",
        "mean_raw_threshold_delta",
        "mean_risk_severity",
        "mean_prediction_coverage",
        "mean_min_prediction_coverage",
        "mean_state_ood_share",
        "max_state_ood_score",
        "mean_base_candidate_count",
        "mean_frontier_candidate_count",
        "mean_tail_candidate_count",
        "mean_suppressed_candidate_count",
        "mean_accepted_frontier_candidate_count",
        "mean_accepted_frontier_suppressed_count",
        "total_accepted_frontier_suppressed_count",
        "total_suppressed_candidate_count",
        "mean_predicted_removed_loss_avoided",
        "mean_predicted_removed_winner_sacrificed",
        "mean_predicted_action_edge",
        "mean_predicted_removed_full_sl",
        "mean_predicted_removed_timeout",
        "sum_predicted_action_edge",
        "first_timestamp",
        "last_timestamp",
        "top_controller_reason",
        "top_controller_reason_share",
    ]
    if schedule.empty:
        return pd.DataFrame(columns=result_cols)
    work = schedule.copy()
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    base = pd.to_numeric(work.get("base_threshold"), errors="coerce")
    state = pd.to_numeric(work.get("state_threshold"), errors="coerce")
    raw_state = pd.to_numeric(work.get("raw_state_threshold"), errors="coerce")
    work["_threshold_delta"] = (state - base).fillna(0.0)
    work["_raw_threshold_delta"] = (raw_state - base).fillna(work["_threshold_delta"])
    work["_threshold_raised"] = work["_threshold_delta"] > 1e-9
    if "force_base_threshold" in work.columns:
        work["_force_base"] = work["force_base_threshold"].fillna(False).astype(bool)
    else:
        work["_force_base"] = False

    def _num(g: pd.DataFrame, col: str) -> pd.Series:
        if col not in g.columns:
            return pd.Series(np.nan, index=g.index, dtype=float)
        return pd.to_numeric(g.get(col), errors="coerce")

    def _one(scope: str, scope_value: str, g: pd.DataFrame) -> dict[str, Any]:
        reason = g.get("controller_reason", pd.Series("", index=g.index)).astype(str)
        reason_counts = reason.value_counts(dropna=False)
        top_reason = str(reason_counts.index[0]) if not reason_counts.empty else ""
        top_share = float(reason_counts.iloc[0] / max(len(g), 1)) if not reason_counts.empty else 0.0
        ts = pd.to_datetime(g.get("timestamp"), utc=True, errors="coerce") if "timestamp" in g.columns else pd.Series(pd.NaT, index=g.index)
        return {
            "scope": scope,
            "scope_value": scope_value,
            "schedule_rows": int(len(g)),
            "timestamp_count": int(ts.nunique(dropna=True)),
            "strategy_count": int(g["strategy_id"].astype(str).nunique()) if "strategy_id" in g.columns else 0,
            "threshold_raised_count": int(g["_threshold_raised"].sum()),
            "threshold_raised_share": float(g["_threshold_raised"].mean()) if len(g) else 0.0,
            "force_base_count": int(g["_force_base"].sum()),
            "force_base_share": float(g["_force_base"].mean()) if len(g) else 0.0,
            "mean_base_threshold": float(_num(g, "base_threshold").mean()),
            "mean_state_threshold": float(_num(g, "state_threshold").mean()),
            "mean_threshold_delta": float(g["_threshold_delta"].mean()),
            "p75_threshold_delta": float(g["_threshold_delta"].quantile(0.75)),
            "max_threshold_delta": float(g["_threshold_delta"].max()),
            "mean_raw_threshold_delta": float(g["_raw_threshold_delta"].mean()),
            "mean_risk_severity": float(_num(g, "risk_severity").mean()),
            "mean_prediction_coverage": float(_num(g, "prediction_coverage").mean()),
            "mean_min_prediction_coverage": float(_num(g, "min_prediction_coverage").mean()),
            "mean_state_ood_share": float(_num(g, "state_ood_share").mean()),
            "max_state_ood_score": float(_num(g, "state_ood_score_max").max()),
            "mean_base_candidate_count": float(_num(g, "base_candidate_count").mean()),
            "mean_frontier_candidate_count": float(_num(g, "frontier_candidate_count").mean()),
            "mean_tail_candidate_count": float(_num(g, "tail_candidate_count").mean()),
            "mean_suppressed_candidate_count": float(_num(g, "suppressed_candidate_count").mean()),
            "mean_accepted_frontier_candidate_count": float(_num(g, "accepted_frontier_candidate_count").mean()),
            "mean_accepted_frontier_suppressed_count": float(_num(g, "accepted_frontier_suppressed_count").mean()),
            "total_accepted_frontier_suppressed_count": int(
                _num(g, "accepted_frontier_suppressed_count").fillna(0).sum()
            ),
            "total_suppressed_candidate_count": int(_num(g, "suppressed_candidate_count").fillna(0).sum()),
            "mean_predicted_removed_loss_avoided": float(_num(g, "predicted_removed_loss_avoided").mean()),
            "mean_predicted_removed_winner_sacrificed": float(_num(g, "predicted_removed_winner_sacrificed").mean()),
            "mean_predicted_action_edge": float(_num(g, "predicted_action_edge").mean()),
            "mean_predicted_removed_full_sl": float(_num(g, "predicted_removed_full_sl_mean").mean()),
            "mean_predicted_removed_timeout": float(_num(g, "predicted_removed_timeout_mean").mean()),
            "sum_predicted_action_edge": float(_num(g, "predicted_action_edge").fillna(0.0).sum()),
            "first_timestamp": ts.min().isoformat() if ts.notna().any() else None,
            "last_timestamp": ts.max().isoformat() if ts.notna().any() else None,
            "top_controller_reason": top_reason,
            "top_controller_reason_share": top_share,
        }

    rows: list[dict[str, Any]] = [_one("all", "all", work)]
    if "head" in work.columns:
        for head, g in work.groupby("head", sort=True, dropna=False):
            rows.append(_one("head", str(head), g))
    if "strategy_id" in work.columns:
        for strategy, g in work.groupby("strategy_id", sort=True, dropna=False):
            rows.append(_one("strategy_id", str(strategy), g))
    if "controller_reason" in work.columns:
        for reason_value, g in work.groupby("controller_reason", sort=True, dropna=False):
            rows.append(_one("controller_reason", str(reason_value), g))
    return pd.DataFrame(rows, columns=result_cols)


def _accepted_trades(candidates: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame()
    idx = pd.to_numeric(accepted["candidate_index"], errors="coerce").astype("Int64")
    accepted = accepted.loc[idx.notna()].copy()
    idx = idx.loc[idx.notna()].astype(int)
    cand = candidates.reset_index(drop=True).iloc[idx.to_numpy()].reset_index(drop=True)
    accepted = accepted.reset_index(drop=True)
    for col in ("head", "symbol", "side", "strategy_id", "net_return", "gross_return", "simple_policy_exit_reason"):
        if col in cand.columns:
            accepted[col] = cand[col].to_numpy()
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["position_size"] = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    accepted["net_return"] = pd.to_numeric(accepted["net_return"], errors="coerce").fillna(0.0)
    accepted["gross_return"] = pd.to_numeric(accepted["gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl"] = accepted["position_size"] * accepted["net_return"]
    accepted["gross_pnl"] = accepted["position_size"] * accepted["gross_return"]
    accepted["cost_pnl"] = accepted["gross_pnl"] - accepted["net_pnl"]
    _assert_unique_decision_keys(accepted, context="accepted trades")
    return accepted


def _worst_24h_net_pnl(accepted: pd.DataFrame) -> float:
    if accepted.empty:
        return 0.0
    work = accepted[["timestamp", "net_pnl"]].copy().sort_values("timestamp")
    values = []
    for ts in work["timestamp"].drop_duplicates().sort_values():
        start = ts - pd.Timedelta(hours=24)
        values.append(float(work.loc[(work["timestamp"] > start) & (work["timestamp"] <= ts), "net_pnl"].sum()))
    return float(min(values)) if values else 0.0


def _metrics_row(arm: str, metrics: dict[str, Any], accepted: pd.DataFrame, schedule: pd.DataFrame | None) -> dict[str, Any]:
    gross = float(metrics.get("gross_pnl", 0.0) or 0.0)
    net = float(metrics.get("net_pnl", 0.0) or 0.0)
    row = {
        "arm": arm,
        "trade_count": int(metrics.get("trade_count", 0) or 0),
        "net_pnl": net,
        "gross_pnl": gross,
        "cost_pnl": gross - net,
        "cost_to_abs_gross": float((gross - net) / max(abs(gross), 1e-9)),
        "compounded_return": metrics.get("compounded_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "worst_24h_net_pnl": _worst_24h_net_pnl(accepted),
        "full_sl_rate": metrics.get("full_sl_rate"),
        "timeout_rate": metrics.get("timeout_rate"),
        "avg_open_positions": metrics.get("avg_open_positions"),
    }
    if schedule is not None and not schedule.empty:
        delta = pd.to_numeric(schedule["state_threshold"], errors="coerce") - pd.to_numeric(schedule["base_threshold"], errors="coerce")
        row.update(
            {
                "mean_threshold_delta": float(delta.mean()),
                "p75_threshold_delta": float(delta.quantile(0.75)),
                "max_threshold_delta": float(delta.max()),
                "share_threshold_raised": float((delta > 1e-6).mean()),
            }
        )
    else:
        row.update(
            {
                "mean_threshold_delta": 0.0,
                "p75_threshold_delta": 0.0,
                "max_threshold_delta": 0.0,
                "share_threshold_raised": 0.0,
            }
        )
    return row


def _by_head(arm: str, accepted: pd.DataFrame) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame()
    reason = accepted["simple_policy_exit_reason"].astype(str).str.lower()
    accepted = accepted.copy()
    accepted["_full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float)
    accepted["_timeout"] = reason.str.contains("timeout", regex=False).astype(float)
    rows = []
    for head, g in accepted.groupby("head"):
        gross = float(g["gross_pnl"].sum())
        net = float(g["net_pnl"].sum())
        rows.append(
            {
                "arm": arm,
                "head": head,
                "trade_count": int(len(g)),
                "win_rate": float((g["net_return"] > 0).mean()),
                "net_pnl": net,
                "gross_pnl": gross,
                "cost_pnl": gross - net,
                "mean_net_return": float(g["net_return"].mean()),
                "q05_net_return": float(g["net_return"].quantile(0.05)),
                "full_sl_rate": float(g["_full_sl"].mean()),
                "timeout_rate": float(g["_timeout"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _accepted_overlap(accepted: pd.DataFrame, baseline_arm: str) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame()
    keys = ["timestamp", "symbol", "strategy_id"]
    base = accepted.loc[accepted["arm"].eq(baseline_arm)].copy()
    base_keys = base[keys].drop_duplicates()
    base_key_to_row = {
        tuple(row[keys].to_numpy()): row
        for _, row in base.drop_duplicates(keys).iterrows()
    }
    base_set = set(map(tuple, base_keys.to_numpy()))
    rows: list[dict[str, Any]] = []
    for arm, g in accepted.groupby("arm"):
        current = g.copy()
        current_keys = current[keys].drop_duplicates()
        current_key_to_row = {
            tuple(row[keys].to_numpy()): row
            for _, row in current.drop_duplicates(keys).iterrows()
        }
        current_set = set(map(tuple, current_keys.to_numpy()))
        entrant_keys = current_set - base_set
        removed_keys = base_set - current_set
        entrant_net = np.array(
            [
                float(current_key_to_row[k].get("net_pnl", 0.0) or 0.0)
                for k in entrant_keys
            ],
            dtype=float,
        )
        removed_net = np.array(
            [
                float(base_key_to_row[k].get("net_pnl", 0.0) or 0.0)
                for k in removed_keys
            ],
            dtype=float,
        )
        loss_avoided = float((-np.clip(removed_net, None, 0.0)).sum()) if len(removed_net) else 0.0
        winner_pnl_sacrificed = float(np.clip(removed_net, 0.0, None).sum()) if len(removed_net) else 0.0
        rows.append(
            {
                "arm": arm,
                "accepted": int(len(current_set)),
                "overlap_with_baseline": int(len(current_set & base_set)),
                "new_vs_baseline": int(len(entrant_keys)),
                "removed_vs_baseline": int(len(removed_keys)),
                "jaccard_vs_baseline": float(len(current_set & base_set) / max(1, len(current_set | base_set))),
                "position_size_sum": float(pd.to_numeric(g["position_size"], errors="coerce").fillna(0.0).sum()),
                "position_size_mean": float(pd.to_numeric(g["position_size"], errors="coerce").fillna(0.0).mean()),
                "entrant_net_pnl": float(entrant_net.sum()) if len(entrant_net) else 0.0,
                "removed_net_pnl": float(removed_net.sum()) if len(removed_net) else 0.0,
                "net_replacement_pnl": float(entrant_net.sum() - removed_net.sum()) if len(entrant_net) or len(removed_net) else 0.0,
                "removed_loss_avoided": loss_avoided,
                "removed_winner_pnl_sacrificed": winner_pnl_sacrificed,
                "defensive_success": loss_avoided - winner_pnl_sacrificed,
            }
        )
    return pd.DataFrame(rows)


def _post_selection_overlay_arm_name(arm: str) -> str:
    return f"{arm}__post_selection_overlay"


def _controller_state_diagnostics(accepted: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame()
    work = accepted.copy()
    sched_cols = [
        "arm",
        "timestamp",
        "strategy_id",
        "base_threshold",
        "state_threshold",
        "risk_severity",
        "threshold_action_enabled",
        "force_base_threshold",
        "controller_reason",
        "prediction_coverage",
        "min_prediction_coverage",
        "state_ood_score_mean",
        "state_ood_score_max",
        "state_ood_cutoff",
        "state_ood_share",
        "mean_pred_utility",
        "mean_pred_lcb",
        "mean_pred_full_sl",
        "mean_pred_timeout",
        "tail_candidate_count",
        "suppressed_candidate_count",
        "accepted_frontier_key_filter_active",
        "accepted_frontier_candidate_count",
        "accepted_frontier_suppressed_count",
        "tail_lcb_q25",
        "tail_pred_full_sl",
        "tail_pred_timeout",
        "predicted_removed_loss_avoided",
        "predicted_removed_winner_sacrificed",
        "predicted_action_edge",
        "action_edge_per_suppressed",
    ]
    if schedules.empty:
        for col in sched_cols:
            if col not in work.columns:
                work[col] = np.nan
    else:
        sched = schedules[[c for c in sched_cols if c in schedules.columns]].copy()
        sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
        if "base_threshold" in sched.columns:
            sched = sched.rename(columns={"base_threshold": "schedule_base_threshold"})
        work = work.merge(
            sched,
            on=["arm", "timestamp", "strategy_id"],
            how="left",
            validate="many_to_one",
        )
    base_for_delta = (
        pd.to_numeric(work.get("schedule_base_threshold"), errors="coerce")
        if "schedule_base_threshold" in work.columns
        else pd.to_numeric(work.get("base_threshold"), errors="coerce")
    )
    work["threshold_delta"] = (
        pd.to_numeric(work.get("state_threshold"), errors="coerce")
        - base_for_delta
    ).fillna(0.0)
    work["threshold_raised"] = work["threshold_delta"] > 1e-6
    reason = work["simple_policy_exit_reason"].astype(str).str.lower()
    work["_full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float)
    work["_timeout"] = reason.str.contains("timeout", regex=False).astype(float)
    rows: list[dict[str, Any]] = []
    for (arm, head, raised), g in work.groupby(["arm", "head", "threshold_raised"], dropna=False):
        rows.append(
            {
                "arm": arm,
                "head": head,
                "threshold_raised": bool(raised),
                "trade_count": int(len(g)),
                "net_pnl": float(g["net_pnl"].sum()),
                "gross_pnl": float(g["gross_pnl"].sum()),
                "cost_pnl": float((g["gross_pnl"] - g["net_pnl"]).sum()),
                "mean_net_return": float(g["net_return"].mean()),
                "win_rate": float((g["net_return"] > 0.0).mean()),
                "full_sl_rate": float(g["_full_sl"].mean()),
                "timeout_rate": float(g["_timeout"].mean()),
                "mean_threshold_delta": float(g["threshold_delta"].mean()),
                "mean_risk_severity": float(pd.to_numeric(g.get("risk_severity"), errors="coerce").mean()),
                "force_base_share": (
                    float(pd.Series(g["force_base_threshold"]).fillna(False).astype(bool).mean())
                    if "force_base_threshold" in g and pd.Series(g["force_base_threshold"]).notna().any()
                    else np.nan
                ),
                "mean_prediction_coverage": float(pd.to_numeric(g.get("prediction_coverage"), errors="coerce").mean()),
                "mean_min_prediction_coverage": float(pd.to_numeric(g.get("min_prediction_coverage"), errors="coerce").mean()),
                "mean_state_ood_score": float(pd.to_numeric(g.get("state_ood_score_mean"), errors="coerce").mean()),
                "max_state_ood_score": float(pd.to_numeric(g.get("state_ood_score_max"), errors="coerce").max()),
                "mean_state_ood_cutoff": float(pd.to_numeric(g.get("state_ood_cutoff"), errors="coerce").mean()),
                "mean_state_ood_share": float(pd.to_numeric(g.get("state_ood_share"), errors="coerce").mean()),
                "mean_pred_utility": float(pd.to_numeric(g.get("mean_pred_utility"), errors="coerce").mean()),
                "mean_pred_lcb": float(pd.to_numeric(g.get("mean_pred_lcb"), errors="coerce").mean()),
                "mean_pred_full_sl": float(pd.to_numeric(g.get("mean_pred_full_sl"), errors="coerce").mean()),
                "mean_pred_timeout": float(pd.to_numeric(g.get("mean_pred_timeout"), errors="coerce").mean()),
                "mean_tail_candidate_count": float(pd.to_numeric(g.get("tail_candidate_count"), errors="coerce").mean()),
                "mean_suppressed_candidate_count": float(pd.to_numeric(g.get("suppressed_candidate_count"), errors="coerce").mean()),
                "mean_tail_lcb_q25": float(pd.to_numeric(g.get("tail_lcb_q25"), errors="coerce").mean()),
                "mean_tail_pred_full_sl": float(pd.to_numeric(g.get("tail_pred_full_sl"), errors="coerce").mean()),
                "mean_tail_pred_timeout": float(pd.to_numeric(g.get("tail_pred_timeout"), errors="coerce").mean()),
                "mean_predicted_removed_loss_avoided": float(pd.to_numeric(g.get("predicted_removed_loss_avoided"), errors="coerce").mean()),
                "mean_predicted_removed_winner_sacrificed": float(pd.to_numeric(g.get("predicted_removed_winner_sacrificed"), errors="coerce").mean()),
                "mean_predicted_action_edge": float(pd.to_numeric(g.get("predicted_action_edge"), errors="coerce").mean()),
                "mean_action_edge_per_suppressed": float(pd.to_numeric(g.get("action_edge_per_suppressed"), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _normalised_decision_keys(df: pd.DataFrame) -> pd.DataFrame:
    keys = df.loc[:, [col for col in DECISION_KEY_COLS if col in df.columns]].copy()
    if "timestamp" in keys:
        keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True, errors="coerce")
    for col in ("symbol", "side", "strategy_id"):
        if col in keys:
            keys[col] = keys[col].astype(str)
    return keys


def _decision_key_set(df: pd.DataFrame) -> set[tuple[Any, ...]]:
    if df.empty:
        return set()
    keys = _normalised_decision_keys(df)
    return set(map(tuple, keys.drop_duplicates().to_numpy()))


def _threshold_action_utility(accepted: pd.DataFrame, baseline_arm: str) -> pd.DataFrame:
    """Measure whether threshold action removed losers or sacrificed winners.

    This reports the spec's "conditional utility of threshold adjustments" at
    the portfolio-decision level. Post-selection overlay arms should have zero
    entrants; pre-auction arms expose replacement quality separately.
    """

    if accepted.empty or "arm" not in accepted.columns:
        return pd.DataFrame()
    base = accepted.loc[accepted["arm"].eq(baseline_arm)].copy()
    if base.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []

    def one_scope(scope_name: str, scope_value: str, base_sub: pd.DataFrame, cur_sub: pd.DataFrame, arm: str) -> None:
        base_keys = _decision_key_set(base_sub)
        cur_keys = _decision_key_set(cur_sub)
        base_key_cols = list(_normalised_decision_keys(base_sub).columns) if not base_sub.empty else list(DECISION_KEY_COLS)
        cur_key_cols = list(_normalised_decision_keys(cur_sub).columns) if not cur_sub.empty else base_key_cols
        base_map = {
            tuple(row[base_key_cols].to_numpy()): row
            for _, row in base_sub.drop_duplicates(base_key_cols).iterrows()
        } if not base_sub.empty else {}
        cur_map = {
            tuple(row[cur_key_cols].to_numpy()): row
            for _, row in cur_sub.drop_duplicates(cur_key_cols).iterrows()
        } if not cur_sub.empty else {}
        entrant_keys = cur_keys - base_keys
        removed_keys = base_keys - cur_keys
        overlap_keys = cur_keys & base_keys
        entrant_net = np.array([float(cur_map[k].get("net_pnl", 0.0) or 0.0) for k in entrant_keys], dtype=float)
        removed_net = np.array([float(base_map[k].get("net_pnl", 0.0) or 0.0) for k in removed_keys], dtype=float)
        overlap_cur_net = np.array([float(cur_map[k].get("net_pnl", 0.0) or 0.0) for k in overlap_keys], dtype=float)
        overlap_base_net = np.array([float(base_map[k].get("net_pnl", 0.0) or 0.0) for k in overlap_keys], dtype=float)
        same_key_net_delta = float((overlap_cur_net - overlap_base_net).sum()) if len(overlap_keys) else 0.0
        loss_avoided = float((-np.clip(removed_net, None, 0.0)).sum()) if len(removed_net) else 0.0
        winner_sacrificed = float(np.clip(removed_net, 0.0, None).sum()) if len(removed_net) else 0.0
        replacement_delta = float(entrant_net.sum() - removed_net.sum()) if len(entrant_net) or len(removed_net) else 0.0
        rows.append(
            {
                "arm": arm,
                "scope": scope_name,
                "scope_value": scope_value,
                "baseline_accepted": int(len(base_keys)),
                "current_accepted": int(len(cur_keys)),
                "overlap": int(len(base_keys & cur_keys)),
                "entrants": int(len(entrant_keys)),
                "removed": int(len(removed_keys)),
                "entrant_net_pnl": float(entrant_net.sum()) if len(entrant_net) else 0.0,
                "removed_net_pnl": float(removed_net.sum()) if len(removed_net) else 0.0,
                "net_replacement_pnl": replacement_delta,
                "same_key_net_pnl_delta": same_key_net_delta,
                "net_action_pnl_delta": replacement_delta + same_key_net_delta,
                "removed_loss_avoided": loss_avoided,
                "removed_winner_pnl_sacrificed": winner_sacrificed,
                "defensive_success": loss_avoided - winner_sacrificed,
            }
        )

    for arm, cur in accepted.groupby("arm", sort=False):
        if arm == baseline_arm:
            continue
        one_scope("all", "all", base, cur.copy(), str(arm))
        for head in sorted(set(base.get("head", pd.Series(dtype=str)).astype(str)) | set(cur.get("head", pd.Series(dtype=str)).astype(str))):
            one_scope(
                "head",
                head,
                base.loc[base["head"].astype(str).eq(head)].copy(),
                cur.loc[cur["head"].astype(str).eq(head)].copy(),
                str(arm),
            )
        for strategy in sorted(
            set(base.get("strategy_id", pd.Series(dtype=str)).astype(str))
            | set(cur.get("strategy_id", pd.Series(dtype=str)).astype(str))
        ):
            one_scope(
                "strategy_id",
                strategy,
                base.loc[base["strategy_id"].astype(str).eq(strategy)].copy(),
                cur.loc[cur["strategy_id"].astype(str).eq(strategy)].copy(),
                str(arm),
            )
    return pd.DataFrame(rows)


def _accepted_trade_key_maps(frame: pd.DataFrame) -> dict[tuple[Any, ...], pd.Series]:
    if frame.empty:
        return {}
    keys = _normalised_decision_keys(frame)
    if keys.empty:
        return {}
    work = frame.copy()
    for col in keys.columns:
        work[col] = keys[col].to_numpy()
    return {
        tuple(row[keys.columns].to_numpy()): row
        for _, row in work.drop_duplicates(list(keys.columns)).iterrows()
    }


def _schedule_action_delta(
    base_sub: pd.DataFrame,
    cur_sub: pd.DataFrame,
) -> dict[str, Any]:
    base_map = _accepted_trade_key_maps(base_sub)
    cur_map = _accepted_trade_key_maps(cur_sub)
    base_keys = set(base_map)
    cur_keys = set(cur_map)
    entrant_keys = cur_keys - base_keys
    removed_keys = base_keys - cur_keys
    overlap_keys = cur_keys & base_keys
    entrant_net = np.array([float(cur_map[k].get("net_pnl", 0.0) or 0.0) for k in entrant_keys], dtype=float)
    removed_net = np.array([float(base_map[k].get("net_pnl", 0.0) or 0.0) for k in removed_keys], dtype=float)
    overlap_cur_net = np.array([float(cur_map[k].get("net_pnl", 0.0) or 0.0) for k in overlap_keys], dtype=float)
    overlap_base_net = np.array([float(base_map[k].get("net_pnl", 0.0) or 0.0) for k in overlap_keys], dtype=float)
    same_key_delta = float((overlap_cur_net - overlap_base_net).sum()) if len(overlap_keys) else 0.0
    replacement_delta = float(entrant_net.sum() - removed_net.sum()) if len(entrant_net) or len(removed_net) else 0.0
    loss_avoided = float((-np.clip(removed_net, None, 0.0)).sum()) if len(removed_net) else 0.0
    winner_sacrificed = float(np.clip(removed_net, 0.0, None).sum()) if len(removed_net) else 0.0
    return {
        "baseline_accepted": int(len(base_keys)),
        "current_accepted": int(len(cur_keys)),
        "overlap": int(len(overlap_keys)),
        "entrants": int(len(entrant_keys)),
        "removed": int(len(removed_keys)),
        "entrant_net_pnl": float(entrant_net.sum()) if len(entrant_net) else 0.0,
        "removed_net_pnl": float(removed_net.sum()) if len(removed_net) else 0.0,
        "net_replacement_pnl": replacement_delta,
        "same_key_net_pnl_delta": same_key_delta,
        "net_action_pnl_delta": replacement_delta + same_key_delta,
        "removed_loss_avoided": loss_avoided,
        "removed_winner_pnl_sacrificed": winner_sacrificed,
        "defensive_success": loss_avoided - winner_sacrificed,
    }


def _threshold_action_edge_validation(
    accepted: pd.DataFrame,
    schedules: pd.DataFrame,
    baseline_arm: str,
) -> pd.DataFrame:
    """Join predicted schedule action edge to realized local trade changes.

    The controller emits one row per arm/timestamp/strategy.  This diagnostic
    compares accepted baseline and current trades for that same local decision
    slice, so false positive threshold raises are visible before they are
    blurred into portfolio-level PnL.
    """

    if accepted.empty or schedules.empty or "arm" not in accepted.columns:
        return pd.DataFrame()
    accepted = accepted.copy()
    base = accepted.loc[accepted["arm"].eq(baseline_arm)].copy()
    if base.empty:
        return pd.DataFrame()
    sched = schedules.copy()
    required = {"arm", "timestamp", "strategy_id"}
    if not required.issubset(sched.columns):
        return pd.DataFrame()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    if "fold" in sched.columns:
        fold_keys = sorted(pd.to_numeric(sched["fold"], errors="coerce").dropna().astype(int).unique().tolist())
    else:
        fold_keys = [None]
    for frame in (accepted, base):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["strategy_id"] = frame["strategy_id"].astype(str)
    rows: list[dict[str, Any]] = []
    schedule_cols = [
        "base_threshold",
        "state_threshold",
        "raw_state_threshold",
        "risk_severity",
        "threshold_action_enabled",
        "force_base_threshold",
        "controller_reason",
        "prediction_coverage",
        "min_prediction_coverage",
        "state_ood_score_mean",
        "state_ood_score_max",
        "state_ood_cutoff",
        "state_ood_share",
        "mean_pred_utility",
        "mean_pred_lcb",
        "mean_pred_full_sl",
        "mean_pred_timeout",
        "tail_candidate_count",
        "suppressed_candidate_count",
        "tail_lcb_q25",
        "tail_pred_full_sl",
        "tail_pred_timeout",
        "predicted_removed_loss_avoided",
        "predicted_removed_winner_sacrificed",
        "predicted_action_edge",
        "action_edge_per_suppressed",
    ]
    for fold_value in fold_keys:
        if fold_value is None:
            sched_fold = sched
            base_fold = base
            accepted_fold = accepted
        else:
            sched_fold = sched.loc[pd.to_numeric(sched.get("fold"), errors="coerce").eq(fold_value)]
            base_fold = base.loc[pd.to_numeric(base.get("fold"), errors="coerce").eq(fold_value)]
            accepted_fold = accepted.loc[pd.to_numeric(accepted.get("fold"), errors="coerce").eq(fold_value)]
        if sched_fold.empty:
            continue
        for arm, arm_sched in sched_fold.groupby("arm", sort=False):
            if str(arm) == baseline_arm:
                continue
            cur_arm = accepted_fold.loc[accepted_fold["arm"].astype(str).eq(str(arm))].copy()
            for _, srow in arm_sched.iterrows():
                ts = pd.Timestamp(srow["timestamp"])
                strategy = str(srow["strategy_id"])
                base_sub = base_fold.loc[
                    base_fold["timestamp"].eq(ts)
                    & base_fold["strategy_id"].astype(str).eq(strategy)
                ]
                cur_sub = cur_arm.loc[
                    cur_arm["timestamp"].eq(ts)
                    & cur_arm["strategy_id"].astype(str).eq(strategy)
                ]
                rec: dict[str, Any] = {
                    "arm": str(arm),
                    "timestamp": ts,
                    "strategy_id": strategy,
                }
                if fold_value is not None:
                    rec["fold"] = int(fold_value)
                if "head" in srow.index:
                    rec["head"] = str(srow.get("head"))
                else:
                    head_val = None
                    if not cur_sub.empty and "head" in cur_sub:
                        head_val = str(cur_sub["head"].iloc[0])
                    elif not base_sub.empty and "head" in base_sub:
                        head_val = str(base_sub["head"].iloc[0])
                    rec["head"] = head_val
                for col in schedule_cols:
                    if col in srow.index:
                        rec[col] = srow.get(col)
                rec["threshold_delta"] = (
                    float(pd.to_numeric(pd.Series([rec.get("state_threshold")]), errors="coerce").iloc[0])
                    - float(pd.to_numeric(pd.Series([rec.get("base_threshold")]), errors="coerce").iloc[0])
                    if "state_threshold" in rec and "base_threshold" in rec
                    else np.nan
                )
                rec["threshold_raised"] = bool(np.isfinite(rec["threshold_delta"]) and rec["threshold_delta"] > 1e-9)
                rec.update(_schedule_action_delta(base_sub, cur_sub))
                rows.append(rec)
    return pd.DataFrame(rows)


def _threshold_action_edge_bucket_performance(
    action_edge: pd.DataFrame,
    *,
    buckets: int = 5,
) -> pd.DataFrame:
    if action_edge.empty or "predicted_action_edge" not in action_edge.columns:
        return pd.DataFrame()
    work = action_edge.copy()
    work["predicted_action_edge"] = pd.to_numeric(work["predicted_action_edge"], errors="coerce")
    work["threshold_delta"] = pd.to_numeric(work.get("threshold_delta"), errors="coerce").fillna(0.0)
    work["threshold_raised"] = work["threshold_delta"] > 1e-9
    rows: list[dict[str, Any]] = []
    group_cols = ["arm"]
    if "fold" in work.columns:
        group_cols.append("fold")
    for group_key, group in work.groupby(group_cols, sort=False, dropna=False):
        g = group.copy()
        finite = g["predicted_action_edge"].replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            g["predicted_action_edge_bucket"] = "missing"
        elif finite.nunique() <= 1:
            g["predicted_action_edge_bucket"] = "single"
        else:
            q = min(int(buckets), int(finite.nunique()))
            bucket = pd.qcut(g["predicted_action_edge"], q=q, labels=False, duplicates="drop")
            g["predicted_action_edge_bucket"] = bucket.astype("Int64").astype(str)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(group_cols, group_key))
        for (bucket_value, raised), sub in g.groupby(["predicted_action_edge_bucket", "threshold_raised"], sort=True, dropna=False):
            rec = {
                **group_meta,
                "predicted_action_edge_bucket": str(bucket_value),
                "threshold_raised": bool(raised),
                "schedule_rows": int(len(sub)),
                "baseline_accepted": int(pd.to_numeric(sub.get("baseline_accepted"), errors="coerce").fillna(0).sum()),
                "current_accepted": int(pd.to_numeric(sub.get("current_accepted"), errors="coerce").fillna(0).sum()),
                "entrants": int(pd.to_numeric(sub.get("entrants"), errors="coerce").fillna(0).sum()),
                "removed": int(pd.to_numeric(sub.get("removed"), errors="coerce").fillna(0).sum()),
                "mean_threshold_delta": float(pd.to_numeric(sub.get("threshold_delta"), errors="coerce").mean()),
                "mean_predicted_action_edge": float(sub["predicted_action_edge"].mean()),
                "sum_predicted_action_edge": float(sub["predicted_action_edge"].fillna(0.0).sum()),
                "net_replacement_pnl": float(pd.to_numeric(sub.get("net_replacement_pnl"), errors="coerce").fillna(0.0).sum()),
                "same_key_net_pnl_delta": float(pd.to_numeric(sub.get("same_key_net_pnl_delta"), errors="coerce").fillna(0.0).sum()),
                "net_action_pnl_delta": float(pd.to_numeric(sub.get("net_action_pnl_delta"), errors="coerce").fillna(0.0).sum()),
                "removed_loss_avoided": float(pd.to_numeric(sub.get("removed_loss_avoided"), errors="coerce").fillna(0.0).sum()),
                "removed_winner_pnl_sacrificed": float(pd.to_numeric(sub.get("removed_winner_pnl_sacrificed"), errors="coerce").fillna(0.0).sum()),
                "defensive_success": float(pd.to_numeric(sub.get("defensive_success"), errors="coerce").fillna(0.0).sum()),
            }
            rec["realized_minus_predicted_action_edge"] = rec["net_action_pnl_delta"] - rec["sum_predicted_action_edge"]
            rows.append(rec)
    return pd.DataFrame(rows)


def _threshold_candidate_suppression_utility(
    candidates: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    eligible_decision_keys: set[tuple[Any, ...]] | None = None,
) -> pd.DataFrame:
    """Measure realized utility of candidates suppressed by state thresholds.

    This is the direct diagnostic for the controller contract:
    E[Delta U | tau_state > tau_base]. It uses the broad executable candidate
    ledger, not only accepted trades, so it is not obscured by global-auction
    capacity replacement.
    """

    result_cols = [
        "arm",
        "scope",
        "scope_value",
        "suppressed_candidates",
        "raised_schedule_count",
        "mean_suppressed_per_raised_schedule",
        "mean_threshold_delta",
        "mean_risk_severity",
        "suppressed_net_return_sum",
        "mean_suppressed_net_return",
        "suppressed_loss_avoided",
        "suppressed_winner_pnl_sacrificed",
        "realized_defensive_success",
        "realized_defensive_success_per_candidate",
        "suppressed_win_rate",
        "suppressed_full_sl_rate",
        "suppressed_timeout_rate",
        "mean_predicted_action_edge",
        "sum_predicted_action_edge",
    ]

    def empty_result() -> pd.DataFrame:
        return pd.DataFrame(columns=result_cols)

    if candidates.empty or schedules.empty:
        return empty_result()
    sched_cols = [
        "arm",
        "timestamp",
        "strategy_id",
        "base_threshold",
        "state_threshold",
        "raw_state_threshold",
        "risk_severity",
        "threshold_action_enabled",
        "force_base_threshold",
        "controller_reason",
        "prediction_coverage",
        "min_prediction_coverage",
        "state_ood_score_mean",
        "state_ood_score_max",
        "state_ood_cutoff",
        "state_ood_share",
        "predicted_removed_loss_avoided",
        "predicted_removed_winner_sacrificed",
        "predicted_action_edge",
    ]
    sched = schedules[[c for c in sched_cols if c in schedules.columns]].copy()
    if "arm" not in sched or "timestamp" not in sched or "strategy_id" not in sched:
        return empty_result()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    sched["_threshold_delta"] = (
        pd.to_numeric(sched.get("state_threshold"), errors="coerce")
        - pd.to_numeric(sched.get("base_threshold"), errors="coerce")
    )
    raised_sched = sched.loc[sched["_threshold_delta"] > 1e-9].copy()
    if raised_sched.empty:
        return empty_result()

    cand = _trade_outcome_flags(candidates)
    if eligible_decision_keys is not None:
        mask = _allowed_decision_key_mask(cand, eligible_decision_keys)
        cand = cand.loc[mask].copy()
        if cand.empty:
            return empty_result()
    cand["timestamp"] = pd.to_datetime(cand["timestamp"], utc=True, errors="coerce")
    cols = [
        "timestamp",
        "strategy_id",
        "head",
        "symbol",
        "side",
        "_rank",
        "_net_return",
        "_is_full_sl",
        "_is_timeout",
    ]
    available = [c for c in cols if c in cand.columns]
    work = cand[available].merge(
        raised_sched,
        on=["timestamp", "strategy_id"],
        how="inner",
        validate="many_to_many",
    )
    if work.empty:
        return empty_result()
    rank = pd.to_numeric(work["_rank"], errors="coerce")
    base = pd.to_numeric(work["base_threshold"], errors="coerce")
    state = pd.to_numeric(work["state_threshold"], errors="coerce")
    work["_state_suppressed"] = (rank >= base) & (rank < state)
    work = work.loc[work["_state_suppressed"]].copy()
    if work.empty:
        return empty_result()
    work["_net_return"] = pd.to_numeric(work["_net_return"], errors="coerce").fillna(0.0)
    work["_loss_avoided"] = -np.minimum(work["_net_return"].to_numpy(dtype=float), 0.0)
    work["_winner_sacrificed"] = np.maximum(work["_net_return"].to_numpy(dtype=float), 0.0)
    work["_defensive_success"] = work["_loss_avoided"] - work["_winner_sacrificed"]
    rows: list[dict[str, Any]] = []

    def append_scope(scope: str, scope_value: str, g: pd.DataFrame) -> None:
        sched_keys = g[["arm", "timestamp", "strategy_id"]].drop_duplicates()
        threshold_delta = pd.to_numeric(g.get("_threshold_delta"), errors="coerce")
        net = pd.to_numeric(g["_net_return"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "arm": str(g["arm"].iloc[0]),
                "scope": scope,
                "scope_value": scope_value,
                "suppressed_candidates": int(len(g)),
                "raised_schedule_count": int(len(sched_keys)),
                "mean_suppressed_per_raised_schedule": float(len(g) / max(len(sched_keys), 1)),
                "mean_threshold_delta": float(threshold_delta.mean()),
                "mean_risk_severity": float(pd.to_numeric(g.get("risk_severity"), errors="coerce").mean()),
                "suppressed_net_return_sum": float(net.sum()),
                "mean_suppressed_net_return": float(net.mean()),
                "suppressed_loss_avoided": float(g["_loss_avoided"].sum()),
                "suppressed_winner_pnl_sacrificed": float(g["_winner_sacrificed"].sum()),
                "realized_defensive_success": float(g["_defensive_success"].sum()),
                "realized_defensive_success_per_candidate": float(g["_defensive_success"].mean()),
                "suppressed_win_rate": float((net > 0.0).mean()),
                "suppressed_full_sl_rate": float(pd.to_numeric(g.get("_is_full_sl"), errors="coerce").mean()),
                "suppressed_timeout_rate": float(pd.to_numeric(g.get("_is_timeout"), errors="coerce").mean()),
                "mean_predicted_action_edge": float(pd.to_numeric(g.get("predicted_action_edge"), errors="coerce").mean()),
                "sum_predicted_action_edge": float(
                    g[["arm", "timestamp", "strategy_id", "predicted_action_edge"]]
                    .drop_duplicates()
                    .loc[:, "predicted_action_edge"]
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .sum()
                ),
            }
        )

    for arm, g in work.groupby("arm", sort=False):
        append_scope("all", "all", g)
        if "head" in g:
            for head, h in g.groupby("head", sort=True):
                append_scope("head", str(head), h)
        for strategy, h in g.groupby("strategy_id", sort=True):
            append_scope("strategy_id", str(strategy), h)
    return pd.DataFrame(rows, columns=result_cols) if rows else empty_result()


def _state_bucket_performance(
    accepted: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    buckets: int = 4,
) -> pd.DataFrame:
    if accepted.empty or schedules.empty:
        return pd.DataFrame()
    sched_cols = [
        "arm",
        "timestamp",
        "strategy_id",
        "base_threshold",
        "state_threshold",
        "risk_severity",
        "threshold_action_enabled",
        "force_base_threshold",
        "prediction_coverage",
        "min_prediction_coverage",
        "state_ood_score_mean",
        "state_ood_score_max",
        "state_ood_cutoff",
        "state_ood_share",
        "mean_pred_utility",
        "mean_pred_lcb",
        "mean_pred_full_sl",
        "mean_pred_timeout",
        "tail_lcb_q25",
        "tail_pred_full_sl",
        "tail_pred_timeout",
        "predicted_removed_loss_avoided",
        "predicted_removed_winner_sacrificed",
        "predicted_action_edge",
        "action_edge_per_suppressed",
    ]
    sched = schedules[[c for c in sched_cols if c in schedules.columns]].copy()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    work = accepted.merge(
        sched,
        on=["arm", "timestamp", "strategy_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_schedule"),
    )
    if "risk_severity" not in work or pd.to_numeric(work["risk_severity"], errors="coerce").notna().sum() == 0:
        return pd.DataFrame()
    reason = work["simple_policy_exit_reason"].astype(str).str.lower()
    work["_full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float)
    work["_timeout"] = reason.str.contains("timeout", regex=False).astype(float)
    work["_threshold_delta"] = (
        pd.to_numeric(work.get("state_threshold"), errors="coerce")
        - pd.to_numeric(work.get("base_threshold_schedule", work.get("base_threshold")), errors="coerce")
    ).fillna(0.0)
    rows: list[dict[str, Any]] = []
    for (arm, head), g in work.dropna(subset=["risk_severity"]).groupby(["arm", "head"], sort=True):
        severity = pd.to_numeric(g["risk_severity"], errors="coerce")
        if severity.nunique(dropna=True) <= 1:
            bucket = pd.Series("single", index=g.index)
        else:
            q = min(int(buckets), int(severity.nunique(dropna=True)))
            bucket = pd.qcut(severity, q=q, labels=False, duplicates="drop").astype("Int64").astype(str)
        h = g.copy()
        h["state_risk_bucket"] = bucket
        for bucket_value, sub in h.groupby("state_risk_bucket", sort=True):
            gross = float(pd.to_numeric(sub["gross_pnl"], errors="coerce").fillna(0.0).sum())
            net = float(pd.to_numeric(sub["net_pnl"], errors="coerce").fillna(0.0).sum())
            rows.append(
                {
                    "arm": arm,
                    "head": head,
                    "state_risk_bucket": str(bucket_value),
                    "trade_count": int(len(sub)),
                    "net_pnl": net,
                    "gross_pnl": gross,
                    "cost_pnl": gross - net,
                    "win_rate": float((pd.to_numeric(sub["net_return"], errors="coerce") > 0.0).mean()),
                    "full_sl_rate": float(sub["_full_sl"].mean()),
                    "timeout_rate": float(sub["_timeout"].mean()),
                    "mean_threshold_delta": float(sub["_threshold_delta"].mean()),
                    "mean_risk_severity": float(pd.to_numeric(sub["risk_severity"], errors="coerce").mean()),
                    "mean_pred_utility": float(pd.to_numeric(sub.get("mean_pred_utility"), errors="coerce").mean()),
                    "mean_pred_lcb": float(pd.to_numeric(sub.get("mean_pred_lcb"), errors="coerce").mean()),
                    "mean_pred_full_sl": float(pd.to_numeric(sub.get("mean_pred_full_sl"), errors="coerce").mean()),
                    "mean_pred_timeout": float(pd.to_numeric(sub.get("mean_pred_timeout"), errors="coerce").mean()),
                    "mean_predicted_action_edge": float(pd.to_numeric(sub.get("predicted_action_edge"), errors="coerce").mean()),
                    "mean_action_edge_per_suppressed": float(pd.to_numeric(sub.get("action_edge_per_suppressed"), errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def _response_calibration(frame: pd.DataFrame, pred: pd.DataFrame, arm: str, buckets: int = 5) -> pd.DataFrame:
    if frame.empty or pred.empty:
        return pd.DataFrame()
    work = frame[["timestamp", "strategy_id", "head", "_net_return", "_is_full_sl", "_is_timeout"]].copy()
    for col in ("pred_lcb_utility", "pred_full_sl", "pred_timeout"):
        work[col] = pd.to_numeric(pred[col], errors="coerce").to_numpy(dtype=float)
    work["arm"] = arm
    rows: list[dict[str, Any]] = []
    for head, g in work.groupby("head", sort=True):
        if len(g) < max(10, buckets):
            continue
        try:
            bucket = pd.qcut(g["pred_lcb_utility"], q=min(buckets, g["pred_lcb_utility"].nunique()), labels=False, duplicates="drop")
        except ValueError:
            bucket = pd.Series(0, index=g.index)
        h = g.copy()
        h["pred_lcb_bucket"] = bucket.fillna(0).astype(int)
        for b, sub in h.groupby("pred_lcb_bucket", sort=True):
            rows.append(
                {
                    "arm": arm,
                    "head": head,
                    "pred_lcb_bucket": int(b),
                    "rows": int(len(sub)),
                    "mean_pred_lcb": float(sub["pred_lcb_utility"].mean()),
                    "mean_net_return": float(sub["_net_return"].mean()),
                    "mean_pred_full_sl": float(sub["pred_full_sl"].mean()),
                    "realized_full_sl": float(sub["_is_full_sl"].mean()),
                    "mean_pred_timeout": float(sub["pred_timeout"].mean()),
                    "realized_timeout": float(sub["_is_timeout"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _render_report(
    summary: pd.DataFrame,
    by_head: pd.DataFrame,
    overlap: pd.DataFrame,
    controller_diag: pd.DataFrame,
    action_utility: pd.DataFrame,
    action_edge_bucket: pd.DataFrame,
    suppression_utility: pd.DataFrame,
    baseline_accepted_suppression_utility: pd.DataFrame,
    state_bucket_perf: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# Market-State Threshold Controller",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Global Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Head Summary",
        "",
        by_head.to_markdown(index=False) if not by_head.empty else "_No accepted trades._",
        "",
        "## Accepted-Trade Overlap",
        "",
        overlap.to_markdown(index=False) if not overlap.empty else "_No accepted trades._",
        "",
        "## Threshold-Raised State Diagnostics",
        "",
        controller_diag.to_markdown(index=False) if not controller_diag.empty else "_No accepted trades._",
        "",
        "## Threshold Action Utility",
        "",
        action_utility.loc[action_utility["scope"].eq("all")].to_markdown(index=False)
        if not action_utility.empty and "scope" in action_utility
        else "_No action utility metrics._",
        "",
        "## Predicted Edge Bucket Validation",
        "",
        action_edge_bucket.to_markdown(index=False)
        if not action_edge_bucket.empty
        else "_No action-edge validation metrics._",
        "",
        "## Candidate Suppression Utility",
        "",
        suppression_utility.loc[suppression_utility["scope"].eq("all")].to_markdown(index=False)
        if not suppression_utility.empty and "scope" in suppression_utility
        else "_No suppressed candidate metrics._",
        "",
        "## Baseline-Accepted Candidate Suppression Utility",
        "",
        baseline_accepted_suppression_utility.loc[baseline_accepted_suppression_utility["scope"].eq("all")].to_markdown(index=False)
        if not baseline_accepted_suppression_utility.empty and "scope" in baseline_accepted_suppression_utility
        else "_No baseline-accepted suppressed candidate metrics._",
        "",
        "## State Bucket Performance",
        "",
        state_bucket_perf.to_markdown(index=False) if not state_bucket_perf.empty else "_No state bucket metrics._",
        "",
        "## Contract",
        "",
        f"- Active heads: `{', '.join(manifest.get('active_heads', [])) or 'none'}`.",
        f"- Disabled heads: `{', '.join(manifest.get('disabled_heads', [])) or 'none'}`.",
        (
            "- Controller-enabled heads: "
            f"`{', '.join(manifest.get('controller', {}).get('controller_enabled_heads', [])) or 'none'}` "
            f"({manifest.get('controller', {}).get('controller_enabled_scope', 'unknown')})."
        ),
        "- Scores, rank references and auction ordering are unchanged.",
        "- Threshold action is penalty-only: state thresholds may not fall below the existing base threshold.",
        "- The existing occupancy-aware dynamic threshold remains downstream of the state threshold.",
        "- Market state is represented as continuous overlapping axes and forecasted severity heads by default.",
        "- Latent/GMM probabilities are shadow-only unless explicitly requested for diagnostics.",
        "- Strategy response models learn residual utility, excess full-SL risk and excess timeout risk after rank-to-outcome curves.",
        "- Missing, non-finite, low-coverage or out-of-distribution state/response inputs force an immediate fallback to the base threshold.",
        "- Post-selection overlay arms first run the baseline auction, then restrict replay to the baseline accepted decision keys so freed capacity is not backfilled.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-broad-candidates", type=Path, default=DEFAULT_TRAIN_BROAD)
    parser.add_argument("--train-deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--eval-candidates", type=Path, default=DEFAULT_EVAL_CANDIDATES)
    parser.add_argument("--train-feature-store-dir", type=Path, default=DEFAULT_TRAIN_FEATURE_STORE)
    parser.add_argument("--eval-feature-store-dir", type=Path, default=DEFAULT_EVAL_FEATURE_STORE)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rank-reference-run-id", default=DEFAULT_RANK_REFERENCE_RUN_ID)
    parser.add_argument(
        "--rank-contract",
        choices=("strict", "short_boll_timestamp_rank", "anchor_global_policy_rank_reference"),
        default="anchor_global_policy_rank_reference",
        help="Causal rank-reference contract applied consistently to train and eval ledgers.",
    )
    parser.add_argument(
        "--disable-heads",
        default="",
        help="Comma-separated heads to remove from the candidate universe.",
    )
    parser.add_argument(
        "--controller-enabled-heads",
        default="",
        help=(
            "Comma-separated heads for which the state controller may raise thresholds. "
            "Empty means all active heads. Disabled heads still keep response diagnostics, "
            "but their thresholds remain at the base floor."
        ),
    )
    parser.add_argument("--max-feature-cols", type=int, default=128)
    parser.add_argument("--max-feature-store-cols", type=int, default=96)
    parser.add_argument("--feature-store-symbol-cap", type=int, default=220)
    parser.add_argument(
        "--allow-candidate-state-fallback",
        action="store_true",
        default=False,
        help=(
            "Allow candidate-ledger timestamp aggregates as a fallback source for "
            "market-state axes when feature-store aggregates are unavailable. "
            "Disabled by default to keep market state independent of the candidate population."
        ),
    )
    parser.add_argument("--forecast-horizon-steps", type=int, default=24)
    parser.add_argument(
        "--forecast-horizons-steps",
        default="6,24",
        help="Comma-separated future severity horizons in timestamp steps; overrides --forecast-horizon-steps when non-empty.",
    )
    parser.add_argument(
        "--forecast-model-kind",
        choices=("lightgbm", "xgboost"),
        default="lightgbm",
        help="Prospective market-state forecast backend. LightGBM is primary; XGBoost is the challenger.",
    )
    parser.add_argument("--latent-states", type=int, default=4)
    parser.add_argument(
        "--include-latent-shadow-arms",
        action="store_true",
        default=False,
        help=(
            "Include latent/GMM state-probability arms as shadow research benchmarks. "
            "They are disabled by default and are not part of the active production architecture."
        ),
    )
    parser.add_argument("--max-response-rows", type=int, default=6000)
    parser.add_argument("--max-response-keyword-cols", type=int, default=24)
    parser.add_argument(
        "--response-model-kind",
        choices=("additive_ebm", "hist_gradient_boosting", "xgboost"),
        default="additive_ebm",
        help=(
            "Strategy-response model family. additive_ebm uses deterministic "
            "training-fitted feature bins plus linear shape effects; "
            "hist_gradient_boosting and xgboost keep shallow tree challengers."
        ),
    )
    parser.add_argument(
        "--response-frontier-weight-gamma",
        type=float,
        default=3.0,
        help="Extra response-model sample weight applied near the strategy deployment threshold.",
    )
    parser.add_argument(
        "--response-frontier-weight-bandwidth",
        type=float,
        default=0.06,
        help="Rank-distance bandwidth for the response frontier sample-weight kernel.",
    )
    parser.add_argument(
        "--response-timestamp-balance",
        dest="response_balance_timestamps",
        action="store_true",
        default=True,
        help="Equalize response-model training mass across decision timestamps.",
    )
    parser.add_argument(
        "--no-response-timestamp-balance",
        dest="response_balance_timestamps",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--response-strategy-balance",
        dest="response_balance_strategies",
        action="store_true",
        default=True,
        help="Equalize response-model training mass across strategy IDs.",
    )
    parser.add_argument(
        "--no-response-strategy-balance",
        dest="response_balance_strategies",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--include-post-selection-overlay-arms",
        dest="include_post_selection_overlay_arms",
        action="store_true",
        default=False,
        help=(
            "Also replay post-selection overlay arms restricted to the S0 accepted "
            "decision keys. This isolates sizing/suppression without backfilling freed capacity."
        ),
    )
    parser.add_argument(
        "--include-guarded-arms",
        dest="include_post_selection_overlay_arms",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--threshold-delta-max", type=float, default=0.10)
    parser.add_argument("--max-threshold-up-step", type=float, default=0.03)
    parser.add_argument("--threshold-relax-alpha", type=float, default=0.25)
    parser.add_argument(
        "--controller-mode",
        choices=(
            "rank_grid",
            "action_aware_rank_grid",
            "frontier_rank_grid",
            "frontier_action_rank_grid",
            "accepted_frontier_action_rank_grid",
            "severity",
        ),
        default="rank_grid",
        help=(
            "rank_grid chooses the lowest admissible threshold from conditional rank tails; "
            "action_aware_rank_grid additionally requires predicted avoided downside to exceed "
            "predicted sacrificed upside; frontier_* variants evaluate the marginal raiseable "
            "rank band instead of letting high-confidence rows wash out frontier risk; "
            "accepted_frontier_action_rank_grid requires the final raise to directly suppress "
            "at least one accepted-frontier row with positive predicted edge; "
            "severity uses the older averaged-risk heuristic."
        ),
    )
    parser.add_argument("--controller-min-lcb-utility", type=float, default=0.0)
    parser.add_argument(
        "--controller-min-prediction-coverage",
        type=float,
        default=0.80,
        help="Minimum finite state/response prediction coverage required before a strategy timestamp may raise its threshold.",
    )
    parser.add_argument(
        "--controller-min-usable-candidates",
        type=int,
        default=1,
        help="Minimum usable candidates above the base threshold required before threshold action is allowed.",
    )
    parser.add_argument(
        "--controller-min-frontier-candidates",
        type=int,
        default=1,
        help=(
            "Minimum usable candidates in the actionable marginal band "
            "[base_threshold, base_threshold + threshold_delta_max] required before threshold action is allowed."
        ),
    )
    parser.add_argument(
        "--controller-max-state-ood-score",
        type=float,
        default=None,
        help=(
            "Optional absolute OOD distance cap for state inputs. "
            "When omitted, each response model uses its train-fold robust 99th percentile cutoff."
        ),
    )
    parser.add_argument(
        "--controller-min-action-edge",
        type=float,
        default=0.0,
        help="Minimum predicted removed downside benefit minus sacrificed upside required by action_aware_rank_grid.",
    )
    parser.add_argument(
        "--controller-min-removed-full-sl",
        type=float,
        default=0.0,
        help="Minimum predicted full-SL risk among removed marginal rows required by action-aware threshold actions.",
    )
    parser.add_argument(
        "--controller-max-removed-timeout",
        type=float,
        default=1.0,
        help="Maximum predicted timeout risk among removed marginal rows allowed by action-aware threshold actions.",
    )
    parser.add_argument(
        "--controller-winner-sacrifice-multiplier",
        type=float,
        default=1.0,
        help="Penalty multiplier applied to predicted positive utility removed by action_aware_rank_grid.",
    )
    parser.add_argument(
        "--enable-timeout-cap",
        dest="use_timeout_cap",
        action="store_true",
        default=False,
        help="Also require timeout risk to stay below its strategy cap. Disabled by default; timeout remains diagnostic.",
    )
    parser.add_argument(
        "--disable-timeout-cap",
        dest="use_timeout_cap",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    if args.train_feature_store_dir is None:
        raise SystemExit(
            "--train-feature-store-dir is required. The old default training feature store "
            "data_perp/features/20260605_070000 was deleted during cleanup, and this "
            "research controller should not silently substitute a different historical sample."
        )
    if not args.train_feature_store_dir.exists():
        raise SystemExit(f"--train-feature-store-dir does not exist: {args.train_feature_store_dir}")
    if not args.eval_feature_store_dir.exists():
        raise SystemExit(f"--eval-feature-store-dir does not exist: {args.eval_feature_store_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params, _ = _load_policy_params(args.policy_manifest, args.policy_variant)
    disabled_heads = _parse_disabled_heads(args.disable_heads)
    controller_enabled_heads = _parse_enabled_heads(args.controller_enabled_heads)
    active_heads = _active_heads(disabled_heads)
    controller_enabled_manifest = _controller_enabled_heads_manifest(controller_enabled_heads, disabled_heads)
    train_broad = _disable_heads(
        _apply_rank_contract(
            _load_candidates(args.train_broad_candidates),
            args.rank_contract,
            data_root=args.data_root,
            rank_reference_run_id=args.rank_reference_run_id,
        ),
        disabled_heads,
    )
    train_deployable = _disable_heads(
        _apply_rank_contract(
            _load_candidates(args.train_deployable_candidates),
            args.rank_contract,
            data_root=args.data_root,
            rank_reference_run_id=args.rank_reference_run_id,
        ),
        disabled_heads,
    )
    eval_candidates = _disable_heads(
        _apply_rank_contract(
            _load_candidates(args.eval_candidates),
            args.rank_contract,
            data_root=args.data_root,
            rank_reference_run_id=args.rank_reference_run_id,
        ),
        disabled_heads,
    )

    feature_cols = _common_feature_columns(train_broad, eval_candidates, args.max_feature_cols)
    train_candidate_agg = _timestamp_aggregates(train_broad, feature_cols)
    eval_candidate_agg = _timestamp_aggregates(eval_candidates, feature_cols)
    feature_store_cols = _select_feature_store_columns(
        args.train_feature_store_dir,
        args.eval_feature_store_dir,
        max_cols=int(args.max_feature_store_cols),
    )
    train_fs, train_fs_report, eval_fs, eval_fs_report = _feature_store_timestamp_aggregate_pair(
        args.train_feature_store_dir,
        args.eval_feature_store_dir,
        train_candidate_agg["timestamp"],
        eval_candidate_agg["timestamp"],
        feature_store_cols,
        symbol_cap=int(args.feature_store_symbol_cap),
    )
    train_state_source, train_state_source_report = _state_source_aggregate_frame(
        train_candidate_agg,
        train_fs,
        allow_candidate_fallback=bool(args.allow_candidate_state_fallback),
    )
    eval_state_source, eval_state_source_report = _state_source_aggregate_frame(
        eval_candidate_agg,
        eval_fs,
        allow_candidate_fallback=bool(args.allow_candidate_state_fallback),
    )
    observed_axis_encoder = fit_observed_axis_encoder(train_state_source, eval_state_source)
    train_observed = transform_observed_axes(train_state_source, observed_axis_encoder)
    eval_observed = transform_observed_axes(eval_state_source, observed_axis_encoder)
    axis_sources = dict(observed_axis_encoder.get("axis_sources") or {})
    axis_sources["state_transition_pressure"] = ["mean_abs_state_axis_diff"]
    forecast_horizons = _parse_int_grid(
        args.forecast_horizons_steps,
        (max(1, int(args.forecast_horizon_steps)),),
    )
    train_forecast, eval_forecast, forecast_report = add_forecast_state_heads(
        train_observed,
        eval_observed,
        horizon_steps=list(forecast_horizons),
        train_agg=train_state_source,
        eval_agg=eval_state_source,
        forecast_model_kind=str(args.forecast_model_kind),
    )
    observed_cols = [c for c in train_observed.columns if c != "timestamp"]
    forecast_cols = [c for c in train_forecast.columns if c != "timestamp"]
    if bool(args.include_latent_shadow_arms):
        train_latent, eval_latent, latent_report = add_latent_state_probs(
            train_forecast,
            eval_forecast,
            n_states=int(args.latent_states),
        )
        latent_cols = [c for c in train_latent.columns if c != "timestamp"]
    else:
        train_latent = eval_latent = pd.DataFrame()
        latent_cols = []
        latent_report = {
            "mode": "shadow_disabled_by_default",
            "reason": "latent_gmm_outputs_removed_from_active_controller_architecture",
        }

    ev_curve = fit_hierarchical_ev_curves(train_deployable)

    arms = {
        "S0_baseline_static_thresholds": {
            "state_train": None,
            "state_eval": None,
            "state_cols": [],
            "per_strategy_residual": False,
        },
        "S1_observed_axes_shared_response": {
            "state_train": train_observed,
            "state_eval": eval_observed,
            "state_cols": observed_cols,
            "per_strategy_residual": False,
        },
        "S2_observed_forecast_shared_response": {
            "state_train": train_forecast,
            "state_eval": eval_forecast,
            "state_cols": forecast_cols,
            "per_strategy_residual": False,
        },
    }
    if bool(args.include_latent_shadow_arms):
        arms.update(
            {
                "S3_observed_forecast_latent_shared_response": {
                    "state_train": train_latent,
                    "state_eval": eval_latent,
                    "state_cols": latent_cols,
                    "per_strategy_residual": False,
                },
                "S4_S3_plus_per_strategy_residual": {
                    "state_train": train_latent,
                    "state_eval": eval_latent,
                    "state_cols": latent_cols,
                    "per_strategy_residual": True,
                },
            }
        )

    summary_rows: list[dict[str, Any]] = []
    schedule_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    accepted_frames: list[pd.DataFrame] = []
    by_head_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    model_reports: dict[str, Any] = {}
    baseline_allowed_keys: set[tuple[Any, ...]] = set()

    for arm, spec in arms.items():
        eval_frame_for_overlay: pd.DataFrame | None = None
        pred_for_overlay: pd.DataFrame | None = None
        curves_for_overlay: RankOutcomeCurves | None = None
        if spec["state_train"] is None:
            candidate_arm = eval_candidates.copy()
            schedule = pd.DataFrame()
            model_reports[arm] = {"mode": "baseline_no_state_controller"}
        else:
            train_frame = build_response_frame(train_broad, spec["state_train"])
            eval_frame = build_response_frame(eval_candidates, spec["state_eval"])
            models, response_features, model_report = fit_response_models(
                train_frame,
                spec["state_cols"],
                per_strategy_residual=bool(spec["per_strategy_residual"]),
                max_rows=int(args.max_response_rows),
                max_keyword_cols=int(args.max_response_keyword_cols),
                response_model_kind=str(args.response_model_kind),
                response_frontier_weight_gamma=float(args.response_frontier_weight_gamma),
                response_frontier_weight_bandwidth=float(args.response_frontier_weight_bandwidth),
                response_balance_timestamps=bool(args.response_balance_timestamps),
                response_balance_strategies=bool(args.response_balance_strategies),
            )
            pred = predict_response(models, eval_frame, response_features, spec["state_cols"])
            eval_frame_for_overlay = eval_frame
            pred_for_overlay = pred
            curves_for_overlay = models["curves"]
            calibration_frames.append(_response_calibration(eval_frame, pred, arm))
            accepted_keys_for_schedule = (
                baseline_allowed_keys
                if str(args.controller_mode) == "accepted_frontier_action_rank_grid"
                and arm != "S0_baseline_static_thresholds"
                and baseline_allowed_keys
                else None
            )
            schedule = threshold_schedule(
                eval_frame,
                pred,
                models["curves"],
                delta_max=float(args.threshold_delta_max),
                max_down_step=float(args.max_threshold_up_step),
                relax_alpha=float(args.threshold_relax_alpha),
                controller_mode=str(args.controller_mode),
                min_lcb_utility=float(args.controller_min_lcb_utility),
                use_timeout_cap=bool(args.use_timeout_cap),
                min_action_edge=float(args.controller_min_action_edge),
                winner_sacrifice_multiplier=float(args.controller_winner_sacrifice_multiplier),
                min_removed_full_sl=float(args.controller_min_removed_full_sl),
                max_removed_timeout=float(args.controller_max_removed_timeout),
                enabled_heads=controller_enabled_heads,
                min_prediction_coverage=float(args.controller_min_prediction_coverage),
                min_usable_candidates=int(args.controller_min_usable_candidates),
                min_frontier_candidates=int(args.controller_min_frontier_candidates),
                max_state_ood_score=args.controller_max_state_ood_score,
                accepted_decision_keys=accepted_keys_for_schedule,
            )
            candidate_arm = apply_thresholds(eval_candidates, schedule)
            schedule["arm"] = arm
            schedule_frames.append(schedule)
            model_reports[arm] = model_report | {
                "response_feature_count": int(len(response_features)),
                "state_feature_count": int(len(spec["state_cols"])),
            }
        decisions, equity, metrics = replay_candidates(
            candidate_arm,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        decisions["arm"] = arm
        accepted = _accepted_trades(candidate_arm, decisions)
        accepted["arm"] = arm
        if arm == "S0_baseline_static_thresholds":
            baseline_allowed_keys = _accepted_key_set(accepted)
        summary_rows.append(_metrics_row(arm, metrics, accepted, schedule))
        by_head_frames.append(_by_head(arm, accepted))
        decision_frames.append(decisions)
        accepted_frames.append(accepted)

        if (
            args.include_post_selection_overlay_arms
            and arm != "S0_baseline_static_thresholds"
            and baseline_allowed_keys
        ):
            overlay_arm = _post_selection_overlay_arm_name(arm)
            overlay_candidates_base = _restrict_to_allowed_decision_keys(eval_candidates, baseline_allowed_keys)
            if eval_frame_for_overlay is not None and pred_for_overlay is not None and curves_for_overlay is not None:
                overlay_mask = _allowed_decision_key_mask(eval_frame_for_overlay, baseline_allowed_keys)
                overlay_frame = eval_frame_for_overlay.loc[overlay_mask].copy()
                overlay_pred = pred_for_overlay.loc[overlay_mask].copy()
                overlay_schedule = threshold_schedule(
                    overlay_frame,
                    overlay_pred,
                    curves_for_overlay,
                    delta_max=float(args.threshold_delta_max),
                    max_down_step=float(args.max_threshold_up_step),
                    relax_alpha=float(args.threshold_relax_alpha),
                    controller_mode=str(args.controller_mode),
                    min_lcb_utility=float(args.controller_min_lcb_utility),
                    use_timeout_cap=bool(args.use_timeout_cap),
                    min_action_edge=float(args.controller_min_action_edge),
                    winner_sacrifice_multiplier=float(args.controller_winner_sacrifice_multiplier),
                    min_removed_full_sl=float(args.controller_min_removed_full_sl),
                    max_removed_timeout=float(args.controller_max_removed_timeout),
                    enabled_heads=controller_enabled_heads,
                    min_prediction_coverage=float(args.controller_min_prediction_coverage),
                    min_usable_candidates=int(args.controller_min_usable_candidates),
                    min_frontier_candidates=int(args.controller_min_frontier_candidates),
                    max_state_ood_score=args.controller_max_state_ood_score,
                    accepted_decision_keys=(
                        baseline_allowed_keys
                        if str(args.controller_mode) == "accepted_frontier_action_rank_grid"
                        and baseline_allowed_keys
                        else None
                    ),
                )
            else:
                overlay_schedule = schedule.copy()
            overlay_candidates = apply_thresholds(overlay_candidates_base, overlay_schedule)
            overlay_decisions, overlay_equity, overlay_metrics = replay_candidates(
                overlay_candidates,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=args.market_mode,
            )
            overlay_decisions["arm"] = overlay_arm
            overlay_accepted = _accepted_trades(overlay_candidates, overlay_decisions)
            overlay_accepted["arm"] = overlay_arm
            if not overlay_schedule.empty:
                overlay_schedule["arm"] = overlay_arm
                schedule_frames.append(overlay_schedule)
            summary_rows.append(_metrics_row(overlay_arm, overlay_metrics, overlay_accepted, overlay_schedule))
            by_head_frames.append(_by_head(overlay_arm, overlay_accepted))
            decision_frames.append(overlay_decisions)
            accepted_frames.append(overlay_accepted)

    summary = pd.DataFrame(summary_rows)
    by_head = pd.concat([x for x in by_head_frames if not x.empty], ignore_index=True) if by_head_frames else pd.DataFrame()
    schedules = pd.concat(schedule_frames, ignore_index=True) if schedule_frames else pd.DataFrame()
    decisions_all = pd.concat(decision_frames, ignore_index=True)
    accepted_all = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    overlap = _accepted_overlap(accepted_all, "S0_baseline_static_thresholds")
    controller_diag = _controller_state_diagnostics(accepted_all, schedules)
    action_utility = _threshold_action_utility(accepted_all, "S0_baseline_static_thresholds")
    action_edge_validation = _threshold_action_edge_validation(accepted_all, schedules, "S0_baseline_static_thresholds")
    action_edge_bucket = _threshold_action_edge_bucket_performance(action_edge_validation)
    suppression_utility = _threshold_candidate_suppression_utility(eval_candidates, schedules)
    baseline_accepted_suppression_utility = _threshold_candidate_suppression_utility(
        eval_candidates,
        schedules,
        eligible_decision_keys=baseline_allowed_keys,
    )
    state_bucket_perf = _state_bucket_performance(accepted_all, schedules)
    calibration = (
        pd.concat([x for x in calibration_frames if not x.empty], ignore_index=True)
        if calibration_frames
        else pd.DataFrame()
    )

    summary.to_csv(args.output_dir / "market_state_threshold_summary.csv", index=False)
    by_head.to_csv(args.output_dir / "market_state_threshold_by_head.csv", index=False)
    overlap.to_csv(args.output_dir / "market_state_threshold_overlap.csv", index=False)
    controller_diag.to_csv(args.output_dir / "controller_state_diagnostics.csv", index=False)
    action_utility.to_csv(args.output_dir / "threshold_action_utility.csv", index=False)
    action_edge_validation.to_csv(args.output_dir / "threshold_action_edge_validation.csv", index=False)
    action_edge_bucket.to_csv(args.output_dir / "threshold_action_edge_bucket_performance.csv", index=False)
    suppression_utility.to_csv(args.output_dir / "threshold_candidate_suppression_utility.csv", index=False)
    baseline_accepted_suppression_utility.to_csv(args.output_dir / "threshold_baseline_accepted_suppression_utility.csv", index=False)
    state_bucket_perf.to_csv(args.output_dir / "state_bucket_performance.csv", index=False)
    calibration.to_csv(args.output_dir / "response_calibration.csv", index=False)
    schedules.to_csv(args.output_dir / "market_state_threshold_schedule.csv", index=False)
    decisions_all.to_parquet(args.output_dir / "decisions.parquet", index=False)
    accepted_all.to_parquet(args.output_dir / "accepted_trades.parquet", index=False)
    train_state_output = train_latent if bool(args.include_latent_shadow_arms) else train_forecast
    eval_state_output = eval_latent if bool(args.include_latent_shadow_arms) else eval_forecast
    train_state_output.to_csv(args.output_dir / "train_market_state_features.csv", index=False)
    eval_state_output.to_csv(args.output_dir / "eval_market_state_features.csv", index=False)
    pd.Series(feature_cols, name="candidate_feature").to_csv(args.output_dir / "candidate_features_used.csv", index=False)
    universe_contract = _standalone_market_state_universe_contract(
        train_fs_report=train_fs_report,
        eval_fs_report=eval_fs_report,
        train_source_report=train_state_source_report,
        eval_source_report=eval_state_source_report,
    )
    (args.output_dir / "market_state_universe_contract.json").write_text(
        json.dumps(_json_safe(universe_contract), indent=2) + "\n",
        encoding="utf-8",
    )
    import joblib

    training_reference_artifact = {
        "generated_by": "run_market_state_threshold_controller",
        "reference_version": "market_state_training_reference_bundle_v1",
        "rank_contract": str(args.rank_contract),
        "rank_reference_run_id": str(args.rank_reference_run_id),
        "data_root": str(args.data_root),
        "disabled_heads": sorted(disabled_heads),
        "active_heads": active_heads,
        "feature_store_columns": feature_store_cols,
        "feature_store_reports": {
            "train": train_fs_report,
            "eval": eval_fs_report,
        },
        "market_state_source_reports": {
            "train": train_state_source_report,
            "eval": eval_state_source_report,
        },
        "observed_axis_encoder": observed_axis_encoder,
        "axis_sources": axis_sources,
        "contract": (
            "Single-run controller state axes must be reproduced by applying "
            "observed_axis_encoder to the market-state source frame; validation/live "
            "rows must not refit robust references."
        ),
    }
    joblib.dump(training_reference_artifact, args.output_dir / "market_state_training_reference.joblib")

    manifest = {
        "generated_by": "run_market_state_threshold_controller",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_broad_candidates": str(args.train_broad_candidates),
        "train_deployable_candidates": str(args.train_deployable_candidates),
        "eval_candidates": str(args.eval_candidates),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "policy_params": asdict(params),
        "rank_contract": str(args.rank_contract),
        "rank_reference_run_id": str(args.rank_reference_run_id),
        "data_root": str(args.data_root),
        "disabled_heads": sorted(disabled_heads),
        "active_heads": active_heads,
        "candidate_feature_count": int(len(feature_cols)),
        "feature_store": {
            "selected_column_count": int(len(feature_store_cols)),
            "selected_columns": feature_store_cols,
            "train": train_fs_report,
            "eval": eval_fs_report,
            "universe_contract_artifact": str(args.output_dir / "market_state_universe_contract.json"),
        },
        "universe_contract": universe_contract,
        "market_state_source": {
            "train": train_state_source_report,
            "eval": eval_state_source_report,
        },
        "axis_sources": axis_sources,
        "observed_axis_encoder": {
            "mode": observed_axis_encoder.get("mode"),
            "minimum_input_coverage": observed_axis_encoder.get("minimum_input_coverage"),
            "fit_rows": observed_axis_encoder.get("fit_rows"),
            "fit_timestamp_min": observed_axis_encoder.get("fit_timestamp_min"),
            "fit_timestamp_max": observed_axis_encoder.get("fit_timestamp_max"),
            "axis_count": int(len(dict(observed_axis_encoder.get("axes") or {}))),
            "reference_column_count": int(len(dict(observed_axis_encoder.get("column_refs") or {}))),
            "source_validation": observed_axis_encoder.get("source_validation"),
            "training_reference_artifact": str(args.output_dir / "market_state_training_reference.joblib"),
        },
        "forecast_report": forecast_report,
        "latent_report": latent_report,
        "include_latent_shadow_arms": bool(args.include_latent_shadow_arms),
        "model_reports": model_reports,
        "controller": {
            "penalty_only": True,
            "threshold_delta_max": float(args.threshold_delta_max),
            "max_threshold_up_step": float(args.max_threshold_up_step),
            "threshold_relax_alpha": float(args.threshold_relax_alpha),
            "controller_mode": str(args.controller_mode),
            "controller_min_lcb_utility": float(args.controller_min_lcb_utility),
            "controller_min_prediction_coverage": float(args.controller_min_prediction_coverage),
            "controller_min_usable_candidates": int(args.controller_min_usable_candidates),
            "controller_min_frontier_candidates": int(args.controller_min_frontier_candidates),
            "controller_max_state_ood_score": (
                float(args.controller_max_state_ood_score)
                if args.controller_max_state_ood_score is not None
                else None
            ),
            "controller_min_action_edge": float(args.controller_min_action_edge),
            "controller_winner_sacrifice_multiplier": float(args.controller_winner_sacrifice_multiplier),
            "controller_min_removed_full_sl": float(args.controller_min_removed_full_sl),
            "controller_max_removed_timeout": float(args.controller_max_removed_timeout),
            "use_timeout_cap": bool(args.use_timeout_cap),
            "max_response_rows": int(args.max_response_rows),
            "max_response_keyword_cols": int(args.max_response_keyword_cols),
            "response_model_kind": str(args.response_model_kind),
            "forecast_model_kind": str(args.forecast_model_kind),
            "response_weighting": {
                "timestamp_balanced": bool(args.response_balance_timestamps),
                "strategy_balanced": bool(args.response_balance_strategies),
                "frontier_gamma": float(args.response_frontier_weight_gamma),
                "frontier_bandwidth": float(args.response_frontier_weight_bandwidth),
            },
            "include_post_selection_overlay_arms": bool(args.include_post_selection_overlay_arms),
            "post_selection_overlay_contract": "baseline accepted decision keys only; no freed-capacity backfill",
            **controller_enabled_manifest,
            "changes_scores_or_ranks": False,
            "changes_auction_ordering": False,
        },
        "outputs": {
            "summary": str(args.output_dir / "market_state_threshold_summary.csv"),
            "by_head": str(args.output_dir / "market_state_threshold_by_head.csv"),
            "schedule": str(args.output_dir / "market_state_threshold_schedule.csv"),
            "overlap": str(args.output_dir / "market_state_threshold_overlap.csv"),
            "controller_state_diagnostics": str(args.output_dir / "controller_state_diagnostics.csv"),
            "threshold_action_utility": str(args.output_dir / "threshold_action_utility.csv"),
            "threshold_action_edge_validation": str(args.output_dir / "threshold_action_edge_validation.csv"),
            "threshold_action_edge_bucket_performance": str(args.output_dir / "threshold_action_edge_bucket_performance.csv"),
            "threshold_candidate_suppression_utility": str(args.output_dir / "threshold_candidate_suppression_utility.csv"),
            "threshold_baseline_accepted_suppression_utility": str(args.output_dir / "threshold_baseline_accepted_suppression_utility.csv"),
            "state_bucket_performance": str(args.output_dir / "state_bucket_performance.csv"),
            "response_calibration": str(args.output_dir / "response_calibration.csv"),
            "accepted_trades": str(args.output_dir / "accepted_trades.parquet"),
            "decisions": str(args.output_dir / "decisions.parquet"),
            "market_state_universe_contract": str(args.output_dir / "market_state_universe_contract.json"),
            "market_state_training_reference": str(args.output_dir / "market_state_training_reference.joblib"),
            "report": str(args.output_dir / "market_state_threshold_controller_report.md"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    (args.output_dir / "market_state_threshold_controller_report.md").write_text(
        _render_report(
            summary,
            by_head,
            overlap,
            controller_diag,
            action_utility,
            action_edge_bucket,
            suppression_utility,
            baseline_accepted_suppression_utility,
            state_bucket_perf,
            manifest,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
