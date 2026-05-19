"""
simple_policy_optimiser.py

A simple script to optimise Trailing Profit and Position sizing parameters using OOF predictions.
Requirements:
1. Use OOS/OFFline predictions from the SlicePlanner policy slice.
2. Optimise trailing profit and capital protection, then tune position sizing.
3. Filter by top 15% preds by rank for each strategy_id.
4. Use 3 equal chronological CV folds; validation-fold averages are the source
   of truth, then fit final deployment params on all policy rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import optuna
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from extreme_price_movements.data_store import read_parquet_projected
from extreme_price_movements.inference.policy_rank_reference import (
    persist_policy_rank_reference,
)
from extreme_price_movements.path_utils import resolve_mode_file
from extreme_price_movements.slice_plan_store import decode_slice_plan_payload

try:
    from extreme_price_movements.utils import tprint
except Exception:  # pragma: no cover

    def tprint(msg: str) -> None:
        print(msg)



def normalize_market_mode(market_mode: str | None = None) -> str:
    mode = str(market_mode or os.environ.get("EPM_MARKET_MODE", "spot")).strip().lower()
    if mode in {"perp", "perps", "future", "futures"}:
        return "perps"
    return "spot"


def append_market_suffix(path: str, market_mode: str | None = None) -> str:
    norm = str(path).rstrip("/\\")
    if market_mode is None and norm.endswith(("_spot", "_perps", "_perp")):
        inferred_mode = "perps" if norm.endswith(("_perps", "_perp")) else "spot"
        mode = normalize_market_mode(inferred_mode)
    else:
        mode = normalize_market_mode(market_mode)
    for suffix in ("_spot", "_perps", "_perp"):
        if norm.endswith(suffix):
            return norm[: -len(suffix)] + f"_{mode}"
    return f"{norm}_{mode}"

# NO EXTERNAL IMPORTS ALLOWED
# from extreme_price_movements.policy_optimiser import ...

logger = logging.getLogger(__name__)

# Parameter grids (moved to Optuna suggest variables directly below)
MARKET_MODE_SUFFIXES = {"spot": "_spot", "perps": "_perp"}
LEGACY_MARKET_SUFFIXES = ("_spot", "_perps", "_perp")
REPORTING_POLICY_RANK_THRESHOLD = 0.85
REPORTING_POLICY_LABEL = "top_15"
DEFAULT_FORWARD_BARS = 96
DEFAULT_BAR_MINUTES = 15
DEFAULT_CV_FOLDS = 3
DEFAULT_N_TRIALS = 200
OPTUNA_EARLY_STOP_NO_IMPROVEMENT = 50
STABLE_TRIAL_TOP_K = 15
STABLE_TRIAL_MIN_CLUSTER_SIZE = 3
STABLE_TRIAL_MAX_ADVERSE_EXIT_RATE = 0.15
STABLE_TRIAL_MAX_FOLD_INSTABILITY = 1.0e9
STABLE_TRIAL_FOLD_FAILURE_THRESHOLD = -1.0e9
STABLE_TRIAL_FOLD_INSTABILITY_PENALTY = 0.25
STABLE_TRIAL_DRAWDOWN_PENALTY = 0.25
STABLE_TRIAL_ADVERSE_OVERUSE_PENALTY = 1.0
STAGE2_MIN_TRADES = 10
STAGE2_MAX_ALLOWED_DRAWDOWN = -1.0
STAGE2_MIN_ALLOWED_FOLD_OBJECTIVE = -1.0e9
ADVERSE_EXIT_ALPHA = 1.0
ADVERSE_EXIT_BETA = 1.0
ADVERSE_EXIT_DELTA = 1.0
ADVERSE_EXIT_FAST_BARS = 4
ADVERSE_EXIT_MAX_MFE_ATR = 0.25
ADVERSE_EXIT_MAX_SL_FRACTION = 0.75
ADVERSE_EXIT_MIN_MAE_ATR_FLOOR = 0.1
MIN_TRAILING_GIVEBACK_FRAC = 0.003
TRAILING_CLUSTER_FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "sl_mult": (0.5, 1.5),
    "trailing_activation_mult": (0.5, 2.5),
    "trailing_power": (1.2, 2.0),
    "trailing_squash_divisor": (1.0, 6.0),
    "giveback_beta": (0.3, 0.95),
}
STAGE2_CLUSTER_FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "sl_mult": (0.5, 3.5),
    "capital_protect_mfe_mult": (0.0, 3.0),
    "capital_protect_regression_frac": (0.0, 1.0),
    "adverse_exit_min_mae_atr": (ADVERSE_EXIT_MIN_MAE_ATR_FLOOR, 3.0),
    "adverse_exit_min_speed": (0.1, 1.5),
    "adverse_exit_theta_quantile": (0.50, 0.95),
}
MAX_DEPLOYMENT_STRATEGIES_PER_SIDE = int(
    os.environ.get("EPM_POLICY_MAX_STRATEGIES_PER_SIDE", "2")
)
MAX_DEPLOYMENT_STRATEGIES_TOTAL = int(
    os.environ.get("EPM_POLICY_MAX_DEPLOYMENT_STRATEGIES", "0") or 0
)
DEPLOYMENT_SELECTION_METRIC = str(
    os.environ.get("EPM_POLICY_DEPLOYMENT_SELECTION_METRIC", "top_5")
)
PORTFOLIO_POLICY_MAX_CONCURRENT_POSITIONS = 8
PORTFOLIO_POLICY_CONCURRENT_FRACTION = 0.75
PORTFOLIO_POLICY_MAX_CONCURRENT_PER_SIDE = None
PORTFOLIO_POLICY_MAX_CONCURRENT_PER_STRATEGY = None
PORTFOLIO_POLICY_MAX_TOTAL_WALLET_ALLOCATION_PCT = 0.95
PORTFOLIO_POLICY_MAX_AVAILABLE_WALLET_POSITION_PCT = 0.50
PORTFOLIO_POLICY_MAX_POSITION_WALLET_PCT = 0.20
PORTFOLIO_POLICY_MAX_POSITION_QUOTE_NOTIONAL = 5000.0
PORTFOLIO_POLICY_BOOK_NOTIONAL_MULTIPLIER = 1.0
PORTFOLIO_POLICY_LEVERAGE_WALLET_MULTIPLIER = 1.0
PORTFOLIO_POLICY_MIN_MARGIN_LEVEL_AFTER_ENTRY = 2.50
PORTFOLIO_POLICY_INITIAL_RANK_THRESHOLD_FLOOR = 0.90
PORTFOLIO_POLICY_LIVE_TEST_MIN_QUOTE_NOTIONAL = 5.0
PORTFOLIO_POLICY_LIVE_TEST_QUOTE_NOTIONAL = 10.0
MAX_CONCURRENT_TRADES = PORTFOLIO_POLICY_MAX_CONCURRENT_POSITIONS
BASE_TO_META_TOP_FRAC = 0.40
DEPLOYMENT_THRESHOLD_MIN = 0.50
DEPLOYMENT_THRESHOLD_MAX = 0.99
DEPLOYMENT_THRESHOLD_PRECISION = 0.01
DEPLOYMENT_RANK_THRESHOLD_EXTRA_REQUIREMENT = float(
    os.environ.get("EPM_POLICY_DEPLOYMENT_RANK_EXTRA_REQUIREMENT", "0.10")
)
DEPLOYMENT_MAX_CONCURRENT_PER_ASSET = 1
DEPLOYMENT_MAX_CONCURRENT_PER_STRATEGY = max(
    1,
    int(
        PORTFOLIO_POLICY_MAX_CONCURRENT_POSITIONS * PORTFOLIO_POLICY_CONCURRENT_FRACTION
    ),
)
SIMPLE_DISCOVERY_SL_MULTS = (0.8, 1.0, 1.2, 1.5)
SIMPLE_DISCOVERY_TP_MULTS = (1.0, 1.5, 2.0, 2.5)
SIMPLE_DISCOVERY_SIZE_POWER = 1.0
SIMPLE_DISCOVERY_ROUND_TRIP_COST_PCT = float(
    os.environ.get("EPM_POLICY_STAGE_A_ROUND_TRIP_COST_PCT", "0.007")
)
SIMPLE_DISCOVERY_LOCAL_BAND_WIDTH = 0.02
SIMPLE_DISCOVERY_CONFIRMATION_BANDS = 5
SIMPLE_DISCOVERY_CONFIRMATION_MIN_POSITIVE = 4
SIMPLE_DISCOVERY_GROSS_PNL_SLIPPAGE_BUFFER = 0.007
ASSET_DECISION_KEEP = "keep"
ASSET_DECISION_DOWN_WEIGHT = "down_weight"
ASSET_DECISION_BLACKLIST = "blacklist"
ASSET_DECISIONS = (
    ASSET_DECISION_KEEP,
    ASSET_DECISION_DOWN_WEIGHT,
    ASSET_DECISION_BLACKLIST,
)


def _normalise_market_mode(market_mode: Optional[str] = None, *, perps: bool = False) -> str:
    mode = str(market_mode or os.environ.get("EPM_MARKET_MODE", "")).strip().lower()
    if mode in {"perp", "perps", "futures"} or perps:
        return "perps"
    return "spot"


def _with_market_suffix(path: str | Path, market_mode: str) -> str:
    norm = str(path).rstrip("/\\")
    suffix = MARKET_MODE_SUFFIXES["perps" if market_mode == "perps" else "spot"]
    for existing in LEGACY_MARKET_SUFFIXES:
        if norm.endswith(existing):
            norm = norm[: -len(existing)]
            break
    return f"{norm}{suffix}"


def _resolve_market_data_root(data_root: str | Path, market_mode: str) -> str:
    suffixed = Path(_with_market_suffix(data_root, market_mode))
    if suffixed.exists():
        return str(suffixed)
    return str(data_root)


def _mode_stem(path: Path, market_mode: str) -> Path:
    return path.with_name(f"{path.stem}_{market_mode}{path.suffix}")


def _write_text_with_mode_alias(path: Path, text: str, market_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    mode_path = _mode_stem(path, market_mode)
    if mode_path != path:
        mode_path.write_text(text)


def _policy_max_strategies_per_side() -> int:
    return int(
        os.environ.get(
            "EPM_POLICY_MAX_STRATEGIES_PER_SIDE",
            str(MAX_DEPLOYMENT_STRATEGIES_PER_SIDE),
        )
    )


def _policy_max_strategies_total() -> int:
    return int(
        os.environ.get(
            "EPM_POLICY_MAX_DEPLOYMENT_STRATEGIES",
            str(MAX_DEPLOYMENT_STRATEGIES_TOTAL),
        )
        or 0
    )


def _policy_selection_metric() -> str:
    return str(
        os.environ.get(
            "EPM_POLICY_DEPLOYMENT_SELECTION_METRIC",
            DEPLOYMENT_SELECTION_METRIC,
        )
    )


def _expand_strategy_id_allowlist(strategy_ids: Sequence[str]) -> Set[str]:
    """Accept core strategy IDs and side-prefixed deployment strategy IDs."""
    expanded: Set[str] = set()
    for raw in strategy_ids:
        sid = str(raw).strip()
        if not sid:
            continue
        expanded.add(sid)
        if sid.startswith("long_"):
            core = sid[len("long_") :]
            if core:
                expanded.add(core)
        elif sid.startswith("short_"):
            core = sid[len("short_") :]
            if core:
                expanded.add(core)
        else:
            expanded.add(f"long_{sid}")
            expanded.add(f"short_{sid}")
    return expanded


def compute_position_size(rank_pct: np.ndarray, size_power: float) -> np.ndarray:
    """Position size formula: size = 0.075 + (0.15 - 0.075) * rank_pct ** size_power"""
    rank_pct = np.asarray(rank_pct, dtype=np.float32)
    return 0.075 + 0.075 * (rank_pct**size_power)


def _holding_time_metrics(
    exit_bars: Sequence[Any] | np.ndarray,
    *,
    bar_minutes: int = DEFAULT_BAR_MINUTES,
) -> Dict[str, float]:
    bars = pd.to_numeric(pd.Series(exit_bars), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    bars = bars.dropna()
    if bars.empty:
        return {
            "avg_holding_bars": 0.0,
            "median_holding_bars": 0.0,
            "p90_holding_bars": 0.0,
            "max_holding_bars": 0.0,
            "avg_holding_time_hours": 0.0,
            "median_holding_time_hours": 0.0,
            "p90_holding_time_hours": 0.0,
            "max_holding_time_hours": 0.0,
        }
    bars = bars.clip(lower=0.0).astype(float)
    hours = bars * float(bar_minutes) / 60.0
    return {
        "avg_holding_bars": float(bars.mean()),
        "median_holding_bars": float(bars.median()),
        "p90_holding_bars": float(bars.quantile(0.90)),
        "max_holding_bars": float(bars.max()),
        "avg_holding_time_hours": float(hours.mean()),
        "median_holding_time_hours": float(hours.median()),
        "p90_holding_time_hours": float(hours.quantile(0.90)),
        "max_holding_time_hours": float(hours.max()),
    }


def _without_concurrency_param(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return policy params that can be combined with an explicit concurrency."""
    out = dict(params)
    out.pop("max_concurrent_trades", None)
    return out


def _ranked_trade_confidence(rank_pct: np.ndarray) -> np.ndarray:
    """Confidence used by adverse exits: ranked confidence centered at 0.5."""
    rank = np.asarray(rank_pct, dtype=np.float32)
    return np.clip(rank - np.float32(0.5), 0.0, 0.5).astype(np.float32, copy=False)


def _adverse_log_exit_scores(
    *,
    df_sub: pd.DataFrame,
    f_highs: np.ndarray,
    f_lows: np.ndarray,
    entry_prices: np.ndarray,
    barrier_price_dist: np.ndarray,
    side: np.ndarray,
    min_mae_atr: float,
    min_speed: float,
    fast_bars: int,
    max_mfe_atr: float,
    sl_mult: float,
) -> np.ndarray:
    """Collect eligible adverse-exit scores used to resolve theta by quantile."""
    n_trades, max_bars = f_highs.shape
    if n_trades == 0 or max_bars <= 1:
        return np.array([], dtype=np.float32)
    confidence = _ranked_trade_confidence(
        pd.to_numeric(df_sub["rank_pct"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32)
    )
    is_long = side == 1
    max_fav = np.zeros(n_trades, dtype=np.float32)
    max_adv = np.zeros(n_trades, dtype=np.float32)
    scores: List[np.ndarray] = []
    fast_bars = max(1, int(fast_bars))
    for j in range(1, min(max_bars, fast_bars + 1)):
        cur_fav = np.where(
            is_long,
            f_highs[:, j] - entry_prices,
            entry_prices - f_lows[:, j],
        )
        cur_adv = np.where(
            is_long,
            entry_prices - f_lows[:, j],
            f_highs[:, j] - entry_prices,
        )
        cur_fav = np.where(np.isfinite(cur_fav), cur_fav, 0.0)
        cur_adv = np.where(np.isfinite(cur_adv), cur_adv, 0.0)
        max_fav = np.maximum(max_fav, cur_fav)
        max_adv = np.maximum(max_adv, cur_adv)
        denom = np.maximum(barrier_price_dist, np.float32(1e-12))
        mae_atr = max_adv / denom
        mfe_atr = max_fav / denom
        bars = np.float32(max(j, 1))
        adverse_speed = mae_atr / bars
        max_adverse_mae_atr = float(sl_mult) * float(ADVERSE_EXIT_MAX_SL_FRACTION)
        eligible = (
            (mae_atr >= float(min_mae_atr))
            & (mae_atr <= max_adverse_mae_atr)
            & (adverse_speed >= float(min_speed))
            & (mfe_atr <= float(max_mfe_atr))
        )
        if np.any(eligible):
            score = (
                np.log1p(ADVERSE_EXIT_ALPHA * (1.0 - confidence[eligible]))
                + np.log1p(ADVERSE_EXIT_BETA * mae_atr[eligible])
                + np.log1p(ADVERSE_EXIT_DELTA * adverse_speed[eligible])
            )
            scores.append(score.astype(np.float32, copy=False))
    if not scores:
        return np.array([], dtype=np.float32)
    return np.concatenate(scores).astype(np.float32, copy=False)


def simulate_and_score(
    df_sub: pd.DataFrame,
    f_opens: np.ndarray,
    f_highs: np.ndarray,
    f_lows: np.ndarray,
    f_closes: np.ndarray,
    cost_pct: float = 0.0015,
    size_power: float = 1.0,
    sl_mult: float = 1.0,
    trailing_activation_mult: float = 1.0,
    trailing_power: float = 1.5,
    trailing_squash_divisor: float = 2.0,
    giveback_beta: float = 0.5,
    capital_protect_mfe_mult: float = 0.0,
    capital_protect_regression_frac: float = 0.45,
    adverse_exit_enabled: bool = False,
    adverse_exit_min_mae_atr: float = 1.0,
    adverse_exit_min_speed: float = 0.3,
    adverse_exit_theta_quantile: float = 0.75,
    adverse_exit_theta: Optional[float] = None,
    adverse_exit_alpha: float = ADVERSE_EXIT_ALPHA,
    adverse_exit_beta: float = ADVERSE_EXIT_BETA,
    adverse_exit_delta: float = ADVERSE_EXIT_DELTA,
    adverse_exit_fast_bars: int = ADVERSE_EXIT_FAST_BARS,
    adverse_exit_max_mfe_atr: float = ADVERSE_EXIT_MAX_MFE_ATR,
    max_concurrent_trades: int = MAX_CONCURRENT_TRADES,
    **_ignored_policy_audit_params: Any,
) -> Dict[str, Any]:
    """
    Fully self-contained, vectorized, bar-by-bar simulator.
    Checks TP/SL pessimistically, computes fees properly per trade.
    """
    n_trades, max_bars = f_opens.shape
    if n_trades == 0:
        return {
            "net_pnl": 0.0,
            "mean_net_trade": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "raw_gains": np.array([], dtype=np.float32),
            "gross_gains": np.array([], dtype=np.float32),
            "sizes": np.array([], dtype=np.float32),
            "exit_bars": np.array([], dtype=np.int16),
            "selected_mask": np.zeros(0, dtype=bool),
            "candidate_count": 0,
            "skipped_concurrency": 0,
            "adverse_exit_count": 0,
            "adverse_exit_rate": 0.0,
            "full_sl_exit_count": 0,
            "capital_protect_exit_count": 0,
            "trailing_exit_count": 0,
            "adverse_exit_theta": (
                float(adverse_exit_theta)
                if adverse_exit_theta is not None
                and np.isfinite(float(adverse_exit_theta))
                else np.nan
            ),
            **_holding_time_metrics([]),
        }

    # 1. Entry
    entry_prices = f_opens[:, 0].copy()
    valid_entry = np.isfinite(entry_prices) & (entry_prices > 0.0)
    if not np.all(valid_entry):
        df_sub = df_sub.iloc[np.flatnonzero(valid_entry)].copy()
        f_opens = f_opens[valid_entry]
        f_highs = f_highs[valid_entry]
        f_lows = f_lows[valid_entry]
        f_closes = f_closes[valid_entry]
        entry_prices = f_opens[:, 0].copy()
        n_trades, max_bars = f_opens.shape
        if n_trades == 0:
            return {
                "net_pnl": 0.0,
                "mean_net_trade": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "raw_gains": np.array([], dtype=np.float32),
                "gross_gains": np.array([], dtype=np.float32),
                "sizes": np.array([], dtype=np.float32),
                "exit_bars": np.array([], dtype=np.int16),
                "selected_mask": np.zeros(0, dtype=bool),
                "candidate_count": 0,
                "skipped_concurrency": 0,
                "adverse_exit_count": 0,
                "adverse_exit_rate": 0.0,
                "full_sl_exit_count": 0,
                "capital_protect_exit_count": 0,
                "trailing_exit_count": 0,
                "adverse_exit_theta": (
                    float(adverse_exit_theta)
                    if adverse_exit_theta is not None
                    and np.isfinite(float(adverse_exit_theta))
                    else np.nan
                ),
                **_holding_time_metrics([]),
            }

    # 2. Position sizing (dynamically scaled)
    sizes = compute_position_size(df_sub["rank_pct"].values, size_power)

    # 3. Side & Barriers
    side = np.ones(n_trades, dtype=np.float32)
    if "side" in df_sub.columns:
        side = df_sub["side"].values

    barrier = np.maximum(
        df_sub.get("barrier_pct", pd.Series(np.full(n_trades, 0.02))).values, 1e-4
    )
    barrier_price_dist = entry_prices * barrier

    is_long_arr = side == 1
    is_short_arr = side == -1

    sl_dist = barrier_price_dist * sl_mult
    tp_act = barrier_price_dist * trailing_activation_mult
    adverse_exit_enabled = bool(adverse_exit_enabled)
    if adverse_exit_enabled:
        resolved_theta = (
            float(adverse_exit_theta)
            if adverse_exit_theta is not None and np.isfinite(float(adverse_exit_theta))
            else np.nan
        )
        if not np.isfinite(resolved_theta):
            score_dist = _adverse_log_exit_scores(
                df_sub=df_sub,
                f_highs=f_highs,
                f_lows=f_lows,
                entry_prices=entry_prices,
                barrier_price_dist=barrier_price_dist,
                side=side,
                min_mae_atr=float(adverse_exit_min_mae_atr),
                min_speed=float(adverse_exit_min_speed),
                fast_bars=int(adverse_exit_fast_bars),
                max_mfe_atr=float(adverse_exit_max_mfe_atr),
                sl_mult=float(sl_mult),
            )
            if len(score_dist):
                resolved_theta = float(
                    np.nanquantile(
                        score_dist,
                        np.clip(float(adverse_exit_theta_quantile), 0.0, 1.0),
                    )
                )
        if not np.isfinite(resolved_theta):
            adverse_exit_enabled = False
            resolved_theta = np.nan
    else:
        resolved_theta = np.nan
    protect_enabled = float(capital_protect_mfe_mult) > 0.0
    x_dist = barrier_price_dist * max(float(capital_protect_mfe_mult), 0.0)
    lock_dist = x_dist - float(capital_protect_regression_frac) * (x_dist + sl_dist)

    active = np.ones(n_trades, dtype=bool)
    protect_active = np.zeros(n_trades, dtype=bool)
    exit_rets = np.zeros(n_trades, dtype=np.float32)
    exit_bars = np.full(n_trades, max_bars - 1, dtype=np.int16)
    max_favorable = np.zeros(n_trades, dtype=np.float32)
    max_adverse = np.zeros(n_trades, dtype=np.float32)
    exit_reason = np.full(n_trades, "timeout", dtype=object)
    ranked_confidence = _ranked_trade_confidence(
        pd.to_numeric(df_sub["rank_pct"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32)
    )

    # 4. Bar by Bar Simulation Loop
    for j in range(1, max_bars):
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            break

        entry = entry_prices[active_idx]
        is_long_mask = is_long_arr[active_idx]
        is_short_mask = is_short_arr[active_idx]

        # 1. Check SL (Pessimistic: happens first)
        sl_hit_long = is_long_mask & (
            f_lows[active_idx, j] <= (entry - sl_dist[active_idx])
        )
        sl_hit_short = is_short_mask & (
            f_highs[active_idx, j] >= (entry + sl_dist[active_idx])
        )
        sl_hit = sl_hit_long | sl_hit_short

        # Update exits for hits
        hit_indices = active_idx[sl_hit]
        exit_rets[hit_indices] = -(sl_dist[hit_indices] / entry_prices[hit_indices])
        exit_bars[hit_indices] = j
        exit_reason[hit_indices] = "full_sl"
        active[hit_indices] = False

        # Re-filter active
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            break

        entry = entry_prices[active_idx]
        cur_fav_long_all = f_highs[:, j] - entry_prices
        cur_fav_short_all = entry_prices - f_lows[:, j]
        cur_fav_all = np.where(is_long_arr, cur_fav_long_all, cur_fav_short_all)
        cur_fav_all = np.where(np.isfinite(cur_fav_all), cur_fav_all, 0.0)
        cur_adv_long_all = entry_prices - f_lows[:, j]
        cur_adv_short_all = f_highs[:, j] - entry_prices
        cur_adv_all = np.where(is_long_arr, cur_adv_long_all, cur_adv_short_all)
        cur_adv_all = np.where(np.isfinite(cur_adv_all), cur_adv_all, 0.0)
        adverse_max_favorable = np.maximum(
            max_favorable,
            np.where(active, cur_fav_all, 0.0),
        )
        adverse_max_adverse = np.maximum(
            max_adverse,
            np.where(active, cur_adv_all, 0.0),
        )

        # 2. Fast adverse-excursion exit before capital protection/trailing.
        if adverse_exit_enabled and j <= int(adverse_exit_fast_bars):
            denom = np.maximum(barrier_price_dist[active_idx], np.float32(1e-12))
            mae_atr = adverse_max_adverse[active_idx] / denom
            mfe_atr = adverse_max_favorable[active_idx] / denom
            bars = np.float32(max(j, 1))
            adverse_speed = mae_atr / bars
            max_adverse_mae_atr = float(sl_mult) * float(ADVERSE_EXIT_MAX_SL_FRACTION)
            eligible = (
                (mae_atr >= float(adverse_exit_min_mae_atr))
                & (mae_atr <= max_adverse_mae_atr)
                & (adverse_speed >= float(adverse_exit_min_speed))
                & (mfe_atr <= float(adverse_exit_max_mfe_atr))
            )
            if np.any(eligible):
                log_exit_score = (
                    np.log1p(
                        float(adverse_exit_alpha)
                        * (1.0 - ranked_confidence[active_idx])
                    )
                    + np.log1p(float(adverse_exit_beta) * mae_atr)
                    + np.log1p(float(adverse_exit_delta) * adverse_speed)
                )
                adverse_hit = eligible & (log_exit_score > float(resolved_theta))
                if np.any(adverse_hit):
                    hit = active_idx[adverse_hit]
                    close_px = f_closes[hit, j].astype(np.float32, copy=False)
                    finite_close = np.isfinite(close_px) & (close_px > 0.0)
                    if np.any(finite_close):
                        finite_hit = hit[finite_close]
                        close_px_f = close_px[finite_close]
                        exit_rets[finite_hit] = side[finite_hit] * (
                            close_px_f / entry_prices[finite_hit] - 1.0
                        )
                        exit_bars[finite_hit] = j
                        exit_reason[finite_hit] = "adverse_exit"
                        active[finite_hit] = False

            active_idx = np.where(active)[0]
            if len(active_idx) == 0:
                break
            entry = entry_prices[active_idx]
        max_adverse = np.maximum(max_adverse, np.where(active, cur_adv_all, 0.0))

        # 3. Optional capital protection before trailing activates.
        if protect_enabled:
            cap_trigger = max_favorable[active_idx] >= x_dist[active_idx]
            if np.any(cap_trigger):
                protect_active[active_idx[cap_trigger]] = True

            protected_idx = active_idx[protect_active[active_idx]]
            if len(protected_idx) > 0:
                protected_entry = entry_prices[protected_idx]
                protected_long = is_long_arr[protected_idx]
                protected_short = is_short_arr[protected_idx]
                orig_sl_long = protected_entry - sl_dist[protected_idx]
                orig_sl_short = protected_entry + sl_dist[protected_idx]
                cap_sl_long = protected_entry + lock_dist[protected_idx]
                cap_sl_short = protected_entry - lock_dist[protected_idx]
                eff_sl_long = np.maximum(orig_sl_long, cap_sl_long)
                eff_sl_short = np.minimum(orig_sl_short, cap_sl_short)
                cap_hit_long = protected_long & (
                    f_lows[protected_idx, j] <= eff_sl_long
                )
                cap_hit_short = protected_short & (
                    f_highs[protected_idx, j] >= eff_sl_short
                )
                if np.any(cap_hit_long):
                    hit = protected_idx[cap_hit_long]
                    exit_rets[hit] = (
                        eff_sl_long[cap_hit_long] - protected_entry[cap_hit_long]
                    ) / protected_entry[cap_hit_long]
                    exit_bars[hit] = j
                    exit_reason[hit] = "capital_protect"
                    active[hit] = False
                if np.any(cap_hit_short):
                    hit = protected_idx[cap_hit_short]
                    exit_rets[hit] = (
                        protected_entry[cap_hit_short] - eff_sl_short[cap_hit_short]
                    ) / protected_entry[cap_hit_short]
                    exit_bars[hit] = j
                    exit_reason[hit] = "capital_protect"
                    active[hit] = False

            active_idx = np.where(active)[0]
            if len(active_idx) == 0:
                break
            entry = entry_prices[active_idx]

        # 4. Check Trailing
        trail_active = max_favorable[active_idx] > tp_act[active_idx]

        dynamic_giveback = (
            max_favorable[active_idx]
            / (barrier_price_dist[active_idx] * trailing_squash_divisor)
        ) ** trailing_power
        dynamic_giveback = np.clip(dynamic_giveback, 0.0, 1.0)
        trail_amount = (
            max_favorable[active_idx] * giveback_beta * (1.0 - dynamic_giveback)
        )
        trail_amount = np.maximum(
            trail_amount,
            entry * MIN_TRAILING_GIVEBACK_FRAC,
        )

        trail_level_long = entry + (max_favorable[active_idx] - trail_amount)
        trail_level_short = entry - (max_favorable[active_idx] - trail_amount)

        trail_hit_long = (
            is_long_arr[active_idx]
            & trail_active
            & (f_lows[active_idx, j] <= trail_level_long)
        )
        trail_hit_short = (
            is_short_arr[active_idx]
            & trail_active
            & (f_highs[active_idx, j] >= trail_level_short)
        )
        trail_hit = trail_hit_long | trail_hit_short

        exit_rets[active_idx[trail_hit_long]] = (
            trail_level_long[trail_hit_long] - entry[trail_hit_long]
        ) / entry[trail_hit_long]
        exit_rets[active_idx[trail_hit_short]] = (
            entry[trail_hit_short] - trail_level_short[trail_hit_short]
        ) / entry[trail_hit_short]
        exit_bars[active_idx[trail_hit]] = j
        exit_reason[active_idx[trail_hit]] = "trailing"
        active[active_idx[trail_hit]] = False

        # Keep legacy trailing/capital semantics: this bar's favorable excursion
        # is available for the next bar, not for same-bar stop promotion.
        max_favorable = np.maximum(max_favorable, np.where(active, cur_fav_all, 0.0))

    # 5. Force exit remaining at max bars
    active_end = active
    if np.any(active_end):
        end_idx = np.flatnonzero(active_end)
        close_rows = f_closes[end_idx]
        finite_close = np.isfinite(close_rows)
        last_pos = np.maximum(np.sum(finite_close, axis=1) - 1, 0)
        b_close = close_rows[np.arange(len(end_idx)), last_pos].astype(
            np.float64, copy=False
        )
        v_ent = entry_prices[end_idx].astype(np.float64, copy=False)
        v_s = side[end_idx]
        exit_rets[end_idx] = v_s * (b_close / v_ent - 1.0)
        exit_bars[end_idx] = last_pos.astype(np.int16, copy=False)
        exit_reason[end_idx] = "timeout"

    # 6. Apply fees and compute net
    fees = sizes * cost_pct + sizes * (1 + exit_rets) * cost_pct
    gross_gain = sizes * exit_rets
    net_gain = gross_gain - fees
    candidate_count = int(len(net_gain))
    selected_mask = np.ones(candidate_count, dtype=bool)

    max_concurrent = max(1, int(max_concurrent_trades))
    if candidate_count and max_concurrent > 0 and "timestamp" in df_sub.columns:
        ts = (
            pd.to_datetime(df_sub["timestamp"], errors="coerce")
            .astype("int64")
            .to_numpy()
        )
        finite_ts = np.isfinite(ts.astype(np.float64))
        order = np.argsort(np.where(finite_ts, ts, np.iinfo(np.int64).max))
        selected_mask = np.zeros(candidate_count, dtype=bool)
        active_until: List[int] = []
        bar_ns = int(pd.Timedelta(minutes=15).value)
        for idx in order:
            if not finite_ts[idx]:
                continue
            cur_ts = int(ts[idx])
            active_until = [until for until in active_until if until > cur_ts]
            if len(active_until) >= max_concurrent:
                continue
            selected_mask[idx] = True
            hold_bars = max(1, int(exit_bars[idx]))
            active_until.append(cur_ts + hold_bars * bar_ns)
        net_gain = net_gain[selected_mask]
        gross_gain = gross_gain[selected_mask]
        sizes = sizes[selected_mask]
    selected_exit_bars = exit_bars[selected_mask]
    selected_exit_reason = exit_reason[selected_mask]
    adverse_exit_count = int(np.sum(selected_exit_reason == "adverse_exit"))
    full_sl_exit_count = int(np.sum(selected_exit_reason == "full_sl"))
    capital_protect_exit_count = int(np.sum(selected_exit_reason == "capital_protect"))
    trailing_exit_count = int(np.sum(selected_exit_reason == "trailing"))
    holding_metrics = _holding_time_metrics(selected_exit_bars)

    return {
        "net_pnl": float(np.sum(net_gain)),
        "mean_net_trade": float(np.mean(net_gain)) if len(net_gain) else 0.0,
        "win_rate": float(np.mean(net_gain > 0)) if len(net_gain) else 0.0,
        "total_trades": len(net_gain),
        "raw_gains": net_gain,
        "gross_gains": gross_gain,
        "sizes": sizes,
        "exit_bars": selected_exit_bars,
        "exit_reason": selected_exit_reason.tolist(),
        "selected_mask": selected_mask,
        "candidate_count": candidate_count,
        "skipped_concurrency": int(candidate_count - int(np.sum(selected_mask))),
        "adverse_exit_count": adverse_exit_count,
        "adverse_exit_rate": float(adverse_exit_count / max(len(net_gain), 1)),
        "full_sl_exit_count": full_sl_exit_count,
        "capital_protect_exit_count": capital_protect_exit_count,
        "trailing_exit_count": trailing_exit_count,
        "adverse_exit_theta": (
            float(resolved_theta) if np.isfinite(resolved_theta) else np.nan
        ),
        **holding_metrics,
    }


def _build_top5_validation_diagnostic(
    selected_rows: pd.DataFrame,
    metrics: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    raw_gains = np.asarray(metrics.get("raw_gains", np.array([], dtype=np.float32)))
    selected_mask = metrics.get("selected_mask")
    if selected_mask is not None:
        mask = np.asarray(selected_mask, dtype=bool)
        if len(mask) == len(selected_rows):
            selected_rows = selected_rows.iloc[np.flatnonzero(mask)].copy()
    if len(raw_gains) != len(selected_rows):
        logger.warning(
            "Skipping top-5 validation diagnostic due to length mismatch: "
            "rows=%s raw_gains=%s",
            len(selected_rows),
            len(raw_gains),
        )
        return None
    if "timestamp" not in selected_rows.columns:
        logger.warning("Skipping top-5 validation diagnostic: missing timestamp.")
        return None
    diagnostic = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(selected_rows["timestamp"], errors="coerce"),
            "net_gain": raw_gains.astype(np.float32, copy=False),
        }
    ).dropna()
    if diagnostic.empty:
        return None
    return diagnostic


def apply_deployment_concurrency_constraints(
    rows: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    side_col: str = "side",
    strategy_col: str = "strategy_id",
    rank_col: str = "deployment_rank_pct",
    holding_bars_col: str = "exit_bars",
    bar_minutes: int = 15,
    initial_rank_threshold: float | None = None,
    dynamic_threshold_enabled: bool = True,
    max_concurrent_total: int = PORTFOLIO_POLICY_MAX_CONCURRENT_POSITIONS,
    max_concurrent_per_side: int = PORTFOLIO_POLICY_MAX_CONCURRENT_PER_SIDE,
    max_concurrent_per_asset: int = DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
    max_concurrent_per_strategy: int = DEPLOYMENT_MAX_CONCURRENT_PER_STRATEGY,
    side_crowding_penalty_max: float = 0.03,
    strategy_crowding_penalty_max: float = 0.03,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    sort_cols = [timestamp_col]
    ascending = [True]
    if rank_col in rows.columns:
        sort_cols.append(rank_col)
        ascending.append(False)
    work = rows.sort_values(sort_cols, ascending=ascending).copy()
    selected: List[Any] = []
    active_by_symbol: Dict[str, List[pd.Timestamp]] = {}
    active_by_side: Dict[str, List[pd.Timestamp]] = {}
    active_by_strategy: Dict[str, List[pd.Timestamp]] = {}
    active_total_until: List[pd.Timestamp] = []

    for idx, row in work.iterrows():
        ts = pd.Timestamp(row[timestamp_col])
        if pd.isna(ts):
            continue
        symbol = str(row[symbol_col])
        strategy_id = str(row.get(strategy_col, "") or "")
        side = str(row.get(side_col, "") or "").lower()
        if side not in {"long", "short"}:
            side = _strategy_side(strategy_id) if strategy_id else "unknown"

        active_total_until = [until for until in active_total_until if until > ts]
        active_by_side[side] = [
            until for until in active_by_side.get(side, []) if until > ts
        ]
        active_by_strategy[strategy_id] = [
            until for until in active_by_strategy.get(strategy_id, []) if until > ts
        ]
        active_by_symbol[symbol] = [
            until for until in active_by_symbol.get(symbol, []) if until > ts
        ]

        if len(active_total_until) >= int(max_concurrent_total):
            continue
        if len(active_by_side.get(side, [])) >= int(max_concurrent_per_side):
            continue
        if len(active_by_strategy.get(strategy_id, [])) >= int(
            max_concurrent_per_strategy
        ):
            continue
        if len(active_by_symbol[symbol]) >= int(max_concurrent_per_asset):
            continue
        if (
            initial_rank_threshold is not None
            and dynamic_threshold_enabled
            and rank_col in row.index
        ):
            rank_val = _safe_float(row.get(rank_col), default=np.nan)
            if not np.isfinite(rank_val):
                continue
            side_util = len(active_by_side.get(side, [])) / max(
                int(max_concurrent_per_side), 1
            )
            strategy_util = len(active_by_strategy.get(strategy_id, [])) / max(
                int(max_concurrent_per_strategy), 1
            )
            occupancy_raise = (
                len(active_total_until)
                * (0.90 - float(initial_rank_threshold))
                / max(int(max_concurrent_total), 1)
            )
            side_penalty = float(side_crowding_penalty_max) * float(side_util) ** 2
            strategy_penalty = float(strategy_crowding_penalty_max) * (
                float(strategy_util) ** 2
            )
            effective_threshold = min(
                0.99,
                float(initial_rank_threshold)
                + occupancy_raise
                + side_penalty
                + strategy_penalty,
            )
            if rank_val < effective_threshold:
                continue

        holding_bars = int(row.get(holding_bars_col, 1) or 1)
        holding_bars = max(1, holding_bars)
        until = ts + pd.Timedelta(minutes=int(bar_minutes) * holding_bars)

        selected.append(idx)
        active_total_until.append(until)
        active_by_side.setdefault(side, []).append(until)
        active_by_strategy.setdefault(strategy_id, []).append(until)
        active_by_symbol.setdefault(symbol, []).append(until)

    return work.loc[selected].copy()


def _slippage_adjusted_mean_gross_positive(metrics: Dict[str, Any]) -> bool:
    """Return whether avg net PnL remains positive after Stage-A execution costs.

    ``net_gain`` from the simple TP/SL simulator already includes the configured
    Stage-A round-trip cost assumption. The gross slippage-adjusted field is
    retained for diagnostics only; using it here would double-count the same
    threshold-discovery cost and push deployment thresholds artificially high.
    """
    return (
        float(metrics.get("mean_net_trade", 0.0) or 0.0)
        > 0.0
        and int(metrics.get("n_trades", 0) or 0) > 0
    )


def score_deployment_threshold_rows(rows: pd.DataFrame) -> Dict[str, Any]:
    empty_metrics = {
        "net_pnl": 0.0,
        "mean_net_trade": 0.0,
        "mean_gross_trade": 0.0,
        "mean_gross_trade_slippage_adjusted": -float(
            SIMPLE_DISCOVERY_GROSS_PNL_SLIPPAGE_BUFFER
        ),
        "gross_slippage_buffer": float(SIMPLE_DISCOVERY_GROSS_PNL_SLIPPAGE_BUFFER),
        "hit_rate": 0.0,
        "n_trades": 0,
        "max_drawdown": 0.0,
        "sortino": 0.0,
        **_holding_time_metrics([]),
    }
    if rows.empty or "net_gain" not in rows.columns:
        return dict(empty_metrics)

    gains = pd.to_numeric(rows["net_gain"], errors="coerce").dropna()
    if gains.empty:
        return dict(empty_metrics)

    if "gross_gain" in rows.columns:
        gross_gains = pd.to_numeric(rows.loc[gains.index, "gross_gain"], errors="coerce")
        gross_gains = gross_gains.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        gross_gains = pd.Series(dtype=np.float64)
    mean_gross_trade = (
        float(gross_gains.mean()) if len(gross_gains) else float(gains.mean())
    )
    mean_gross_trade_slippage_adjusted = float(
        mean_gross_trade - SIMPLE_DISCOVERY_GROSS_PNL_SLIPPAGE_BUFFER
    )

    cum = gains.cumsum()
    dd = cum - cum.cummax()
    downside = gains[gains < 0]
    if len(downside) == 0 or float(downside.std(ddof=0)) == 0.0:
        sortino = 100.0 if float(gains.mean()) > 0.0 else 0.0
    else:
        sortino = float(gains.mean() / np.sqrt(np.mean(downside**2)))
    holding_metrics = (
        _holding_time_metrics(rows.loc[gains.index, "exit_bars"])
        if "exit_bars" in rows.columns
        else _holding_time_metrics([])
    )

    return {
        "net_pnl": float(gains.sum()),
        "mean_net_trade": float(gains.mean()),
        "mean_gross_trade": float(mean_gross_trade),
        "mean_gross_trade_slippage_adjusted": float(
            mean_gross_trade_slippage_adjusted
        ),
        "gross_slippage_buffer": float(SIMPLE_DISCOVERY_GROSS_PNL_SLIPPAGE_BUFFER),
        "hit_rate": float((gains > 0).mean()),
        "n_trades": int(len(gains)),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "sortino": float(sortino),
        **holding_metrics,
    }


def _simulate_simple_tp_sl_rows(
    df_sub: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    cost_pct: float,
    size_power: float,
    sl_mult: float,
    tp_mult: float,
) -> pd.DataFrame:
    """Vectorized fixed TP/SL simulator for broad rank-threshold discovery."""
    f_opens, f_highs, f_lows, f_closes = paths
    n_trades, max_bars = f_opens.shape
    if n_trades == 0:
        return pd.DataFrame()

    entry_prices = f_opens[:, 0].astype(np.float32, copy=True)
    valid_entry = np.isfinite(entry_prices) & (entry_prices > 0.0)
    if not np.any(valid_entry):
        return pd.DataFrame()
    if not np.all(valid_entry):
        df_sub = df_sub.iloc[np.flatnonzero(valid_entry)].copy()
        f_opens = f_opens[valid_entry]
        f_highs = f_highs[valid_entry]
        f_lows = f_lows[valid_entry]
        f_closes = f_closes[valid_entry]
        entry_prices = f_opens[:, 0].astype(np.float32, copy=True)
        n_trades, max_bars = f_opens.shape
        if n_trades == 0:
            return pd.DataFrame()

    sizes = compute_position_size(df_sub["rank_pct"].to_numpy(), size_power)
    side = np.ones(n_trades, dtype=np.float32)
    if "side" in df_sub.columns:
        side = (
            pd.to_numeric(df_sub["side"], errors="coerce")
            .fillna(1)
            .to_numpy(dtype=np.float32)
        )

    barrier = np.maximum(
        pd.to_numeric(
            df_sub.get("barrier_pct", pd.Series(np.full(n_trades, 0.02))),
            errors="coerce",
        )
        .fillna(0.02)
        .to_numpy(dtype=np.float32),
        np.float32(1e-4),
    )
    barrier_price_dist = entry_prices * barrier
    sl_dist = barrier_price_dist * np.float32(sl_mult)
    tp_dist = barrier_price_dist * np.float32(tp_mult)
    is_long_arr = side == 1
    is_short_arr = side == -1

    active = np.ones(n_trades, dtype=bool)
    exit_rets = np.zeros(n_trades, dtype=np.float32)
    exit_bars = np.full(n_trades, max_bars - 1, dtype=np.int16)

    for j in range(1, max_bars):
        active_idx = np.flatnonzero(active)
        if len(active_idx) == 0:
            break
        entry = entry_prices[active_idx]
        is_long = is_long_arr[active_idx]
        is_short = is_short_arr[active_idx]

        # Pessimistic same-bar ordering: stop loss before take profit.
        sl_hit_long = is_long & (f_lows[active_idx, j] <= entry - sl_dist[active_idx])
        sl_hit_short = is_short & (
            f_highs[active_idx, j] >= entry + sl_dist[active_idx]
        )
        sl_hit = sl_hit_long | sl_hit_short
        if np.any(sl_hit):
            hit = active_idx[sl_hit]
            exit_rets[hit] = -(sl_dist[hit] / entry_prices[hit])
            exit_bars[hit] = j
            active[hit] = False

        active_idx = np.flatnonzero(active)
        if len(active_idx) == 0:
            break
        entry = entry_prices[active_idx]
        is_long = is_long_arr[active_idx]
        is_short = is_short_arr[active_idx]
        tp_hit_long = is_long & (f_highs[active_idx, j] >= entry + tp_dist[active_idx])
        tp_hit_short = is_short & (f_lows[active_idx, j] <= entry - tp_dist[active_idx])
        tp_hit = tp_hit_long | tp_hit_short
        if np.any(tp_hit):
            hit = active_idx[tp_hit]
            exit_rets[hit] = tp_dist[hit] / entry_prices[hit]
            exit_bars[hit] = j
            active[hit] = False

    active_end = np.flatnonzero(active)
    if len(active_end) > 0:
        close_rows = f_closes[active_end]
        finite_close = np.isfinite(close_rows)
        last_pos = np.maximum(np.sum(finite_close, axis=1) - 1, 0)
        b_close = close_rows[np.arange(len(active_end)), last_pos].astype(
            np.float32, copy=False
        )
        v_ent = entry_prices[active_end]
        v_s = side[active_end]
        exit_rets[active_end] = v_s * (b_close / v_ent - 1.0)
        exit_bars[active_end] = last_pos.astype(np.int16, copy=False)

    fees = sizes * cost_pct + sizes * (1.0 + exit_rets) * cost_pct
    gross_gain = sizes * exit_rets
    net_gain = gross_gain - fees
    rows = df_sub.copy()
    rows["net_gain"] = net_gain.astype(np.float32, copy=False)
    rows["gross_gain"] = gross_gain.astype(np.float32, copy=False)
    rows["exit_bars"] = exit_bars.astype(np.int16, copy=False)
    rows["simple_sl_mult"] = float(sl_mult)
    rows["simple_tp_mult"] = float(tp_mult)
    return rows


def discover_deployment_rank_threshold_simple_grid(
    rows: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    cost_pct: float,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    side_col: str = "side",
    strategy_col: str = "strategy_id",
    max_concurrent_per_asset: int = DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
    lo: float = DEPLOYMENT_THRESHOLD_MIN,
    hi: float = DEPLOYMENT_THRESHOLD_MAX,
    precision: float = DEPLOYMENT_THRESHOLD_PRECISION,
    sl_mults: Sequence[float] = SIMPLE_DISCOVERY_SL_MULTS,
    tp_mults: Sequence[float] = SIMPLE_DISCOVERY_TP_MULTS,
    size_power: float = SIMPLE_DISCOVERY_SIZE_POWER,
    local_band_width: float = SIMPLE_DISCOVERY_LOCAL_BAND_WIDTH,
    confirmation_bands: int = SIMPLE_DISCOVERY_CONFIRMATION_BANDS,
    confirmation_min_positive: int = SIMPLE_DISCOVERY_CONFIRMATION_MIN_POSITIVE,
) -> Dict[str, Any]:
    """Stage A: rank-threshold discovery over full policy rows and simple TP/SL."""
    if rows.empty:
        return {
            "deployment_rank_threshold": float(hi),
            "objective": float("-inf"),
            "reason": "empty_rows",
        }

    work = rows.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work["rank_pct"] = pd.to_numeric(work["rank_pct"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    valid_mask = ~work[[timestamp_col, symbol_col, "rank_pct"]].isna().any(axis=1)
    valid_idx = np.flatnonzero(valid_mask.to_numpy())
    work = work.iloc[valid_idx].copy().reset_index(drop=True)
    paths = _path_take(paths, valid_idx)
    if work.empty:
        return {
            "deployment_rank_threshold": float(hi),
            "objective": float("-inf"),
            "reason": "no_valid_rank_rows",
        }

    threshold_grid = np.arange(
        float(lo), float(hi) + float(precision) / 2.0, float(precision)
    )
    threshold_grid = np.unique(np.round(np.clip(threshold_grid, lo, hi), 4))
    all_threshold_results: List[Dict[str, Any]] = []
    best_by_threshold: List[Dict[str, Any]] = []
    simulated_policy_rows: List[Tuple[float, float, pd.DataFrame]] = []
    local_band_width = float(max(local_band_width, 1e-6))
    confirmation_bands = int(max(0, confirmation_bands))
    confirmation_min_positive = int(
        min(max(0, confirmation_min_positive), max(confirmation_bands, 1))
    )

    for sl_mult in sl_mults:
        for tp_mult in tp_mults:
            sim_rows = _simulate_simple_tp_sl_rows(
                work,
                paths,
                cost_pct=cost_pct,
                size_power=size_power,
                sl_mult=float(sl_mult),
                tp_mult=float(tp_mult),
            )
            if not sim_rows.empty:
                simulated_policy_rows.append((float(sl_mult), float(tp_mult), sim_rows))

    for threshold in threshold_grid:
        threshold_best: Optional[Dict[str, Any]] = None
        for sl_mult, tp_mult, sim_rows in simulated_policy_rows:
            rank_values = sim_rows["rank_pct"].to_numpy(dtype=np.float32)
            local_idx = np.flatnonzero(
                (rank_values >= threshold)
                & (rank_values < min(1.0 + 1e-6, threshold + local_band_width))
            )
            if len(local_idx) == 0:
                continue

            local_sub = sim_rows.iloc[local_idx].copy()
            selected_local = apply_deployment_concurrency_constraints(
                local_sub,
                timestamp_col=timestamp_col,
                symbol_col=symbol_col,
                side_col=side_col,
                strategy_col=strategy_col,
                rank_col="rank_pct",
                initial_rank_threshold=None,
                dynamic_threshold_enabled=False,
                max_concurrent_total=1_000_000,
                max_concurrent_per_side=1_000_000,
                max_concurrent_per_asset=max_concurrent_per_asset,
                max_concurrent_per_strategy=1_000_000,
            )
            local_metrics = score_deployment_threshold_rows(selected_local)
            next_band_metrics: List[Dict[str, Any]] = []
            next_positive_count = 0
            for band_no in range(1, confirmation_bands + 1):
                band_lo = float(threshold + band_no * local_band_width)
                band_hi = float(band_lo + local_band_width)
                band_idx = np.flatnonzero(
                    (rank_values >= band_lo) & (rank_values < min(1.0 + 1e-6, band_hi))
                )
                if len(band_idx) == 0:
                    band_metrics = {
                        "band_lo": band_lo,
                        "band_hi": min(1.0, band_hi),
                        "mean_net_trade": 0.0,
                        "mean_gross_trade": 0.0,
                        "mean_gross_trade_slippage_adjusted": -float(
                            SIMPLE_DISCOVERY_GROSS_PNL_SLIPPAGE_BUFFER
                        ),
                        "n_trades": 0,
                        "positive": False,
                    }
                    next_band_metrics.append(band_metrics)
                    continue
                band_sub = sim_rows.iloc[band_idx].copy()
                selected_band = apply_deployment_concurrency_constraints(
                    band_sub,
                    timestamp_col=timestamp_col,
                    symbol_col=symbol_col,
                    side_col=side_col,
                    strategy_col=strategy_col,
                    rank_col="rank_pct",
                    initial_rank_threshold=None,
                    dynamic_threshold_enabled=False,
                    max_concurrent_total=1_000_000,
                    max_concurrent_per_side=1_000_000,
                    max_concurrent_per_asset=max_concurrent_per_asset,
                    max_concurrent_per_strategy=1_000_000,
                )
                scored_band = score_deployment_threshold_rows(selected_band)
                positive = _slippage_adjusted_mean_gross_positive(scored_band)
                if positive:
                    next_positive_count += 1
                next_band_metrics.append(
                    {
                        "band_lo": band_lo,
                        "band_hi": min(1.0, band_hi),
                        "mean_net_trade": float(
                            scored_band.get("mean_net_trade", 0.0) or 0.0
                        ),
                        "mean_gross_trade": float(
                            scored_band.get("mean_gross_trade", 0.0) or 0.0
                        ),
                        "mean_gross_trade_slippage_adjusted": float(
                            scored_band.get(
                                "mean_gross_trade_slippage_adjusted", 0.0
                            )
                            or 0.0
                        ),
                        "n_trades": int(scored_band.get("n_trades", 0) or 0),
                        "positive": bool(positive),
                    }
                )

            cumulative_idx = np.flatnonzero(rank_values >= threshold)
            cumulative_sub = sim_rows.iloc[cumulative_idx].copy()
            selected_cumulative = apply_deployment_concurrency_constraints(
                cumulative_sub,
                timestamp_col=timestamp_col,
                symbol_col=symbol_col,
                side_col=side_col,
                strategy_col=strategy_col,
                rank_col="rank_pct",
                initial_rank_threshold=None,
                dynamic_threshold_enabled=False,
                max_concurrent_total=1_000_000,
                max_concurrent_per_side=1_000_000,
                max_concurrent_per_asset=max_concurrent_per_asset,
                max_concurrent_per_strategy=1_000_000,
            )
            cumulative_metrics = score_deployment_threshold_rows(selected_cumulative)
            local_positive = _slippage_adjusted_mean_gross_positive(local_metrics)
            local_confirmation_passed = bool(
                local_positive and next_positive_count >= confirmation_min_positive
            )
            objective = (
                local_metrics["net_pnl"]
                - 0.10 * abs(local_metrics.get("max_drawdown", 0.0))
                + 0.10 * local_metrics.get("sortino", 0.0)
            )
            result = {
                "deployment_rank_threshold": float(threshold),
                "objective": float(objective),
                "simple_sl_mult": float(sl_mult),
                "simple_tp_mult": float(tp_mult),
                "candidate_rows": int(len(local_sub)),
                "local_band_width": float(local_band_width),
                "local_band_lo": float(threshold),
                "local_band_hi": float(min(1.0, threshold + local_band_width)),
                "local_band_positive": bool(local_positive),
                "next_band_positive_count": int(next_positive_count),
                "confirmation_bands": int(confirmation_bands),
                "confirmation_min_positive": int(confirmation_min_positive),
                "local_confirmation_passed": bool(local_confirmation_passed),
                "next_band_metrics": next_band_metrics,
                "cumulative_net_pnl": float(cumulative_metrics.get("net_pnl", 0.0)),
                "cumulative_mean_net_trade": float(
                    cumulative_metrics.get("mean_net_trade", 0.0)
                ),
                "cumulative_mean_gross_trade": float(
                    cumulative_metrics.get("mean_gross_trade", 0.0)
                ),
                "cumulative_mean_gross_trade_slippage_adjusted": float(
                    cumulative_metrics.get(
                        "mean_gross_trade_slippage_adjusted", 0.0
                    )
                ),
                "cumulative_hit_rate": float(cumulative_metrics.get("hit_rate", 0.0)),
                "cumulative_n_trades": int(cumulative_metrics.get("n_trades", 0)),
                "cumulative_max_drawdown": float(
                    cumulative_metrics.get("max_drawdown", 0.0)
                ),
                "cumulative_sortino": float(cumulative_metrics.get("sortino", 0.0)),
                **local_metrics,
            }
            all_threshold_results.append(result)
            if not local_confirmation_passed:
                continue
            if threshold_best is None or (
                float(result.get("mean_gross_trade_slippage_adjusted", 0.0)),
                float(result.get("mean_gross_trade", 0.0)),
                float(result["net_pnl"]),
                int(result["n_trades"]),
            ) > (
                float(
                    threshold_best.get("mean_gross_trade_slippage_adjusted", 0.0)
                ),
                float(threshold_best.get("mean_gross_trade", 0.0)),
                float(threshold_best["net_pnl"]),
                int(threshold_best["n_trades"]),
            ):
                threshold_best = result

        if threshold_best is None:
            empty = {
                "deployment_rank_threshold": float(threshold),
                "objective": float("-inf"),
                "net_pnl": 0.0,
                "mean_net_trade": 0.0,
                "mean_gross_trade": 0.0,
                "mean_gross_trade_slippage_adjusted": -float(
                    SIMPLE_DISCOVERY_GROSS_PNL_SLIPPAGE_BUFFER
                ),
                "gross_slippage_buffer": float(
                    SIMPLE_DISCOVERY_GROSS_PNL_SLIPPAGE_BUFFER
                ),
                "hit_rate": 0.0,
                "n_trades": 0,
                "max_drawdown": 0.0,
                "sortino": 0.0,
                "simple_sl_mult": None,
                "simple_tp_mult": None,
            }
            best_by_threshold.append(empty)
            continue
        best_by_threshold.append(dict(threshold_best))

    profitable = [
        r for r in best_by_threshold if bool(r.get("local_confirmation_passed", False))
    ]
    if profitable:
        profitable_thresholds = np.asarray(
            [float(r["deployment_rank_threshold"]) for r in profitable],
            dtype=np.float32,
        )
        selected_threshold = float(np.quantile(profitable_thresholds, 0.20))
        threshold_candidates = [
            r
            for r in profitable
            if float(r["deployment_rank_threshold"])
            == float(
                min(
                    profitable,
                    key=lambda row: abs(
                        float(row["deployment_rank_threshold"]) - selected_threshold
                    ),
                )["deployment_rank_threshold"]
            )
        ]
        best = dict(
            max(
                threshold_candidates,
                key=lambda r: (
                    float(r.get("mean_gross_trade_slippage_adjusted", 0.0)),
                    float(r.get("mean_gross_trade", 0.0)),
                    float(r.get("cumulative_mean_net_trade", 0.0)),
                    int(r.get("n_trades", 0)),
                ),
            )
        )
        reason = "iq20_local_band_positive_with_4of5_positive_above"
    else:
        best = max(
            all_threshold_results or best_by_threshold,
            key=lambda r: (
                int(r.get("next_band_positive_count", 0)),
                float(r.get("mean_gross_trade_slippage_adjusted", 0.0)),
                float(r.get("mean_gross_trade", 0.0)),
                float(r.get("cumulative_mean_net_trade", 0.0)),
            ),
        )
        reason = "no_confirmed_local_band_fallback_best_local_mean"

    profitable_threshold_list = [
        float(r["deployment_rank_threshold"]) for r in profitable
    ]
    best["threshold_search"] = {
        "method": "full_policy_rank_grid_simple_tp_sl_iq20_positive_mean_net_trade",
        "all_in_execution_cost_assumption_pct": float(
            SIMPLE_DISCOVERY_ROUND_TRIP_COST_PCT
        ),
        "all_in_execution_cost_assumption_note": (
            "threshold metadata assumption only; includes fees, slippage, spread, "
            "and execution delay"
        ),
        "scoring_round_trip_cost_pct": float(cost_pct * 2.0),
        "per_side_cost_pct": float(cost_pct),
        "lo": float(lo),
        "hi": float(hi),
        "precision": float(precision),
        "selected_threshold": float(best["deployment_rank_threshold"]),
        "evaluated_thresholds": [float(t) for t in threshold_grid],
        "profitable_thresholds": profitable_threshold_list,
        "profitable_threshold_count": int(len(profitable_threshold_list)),
        "profitable_threshold_min": (
            float(min(profitable_threshold_list)) if profitable_threshold_list else None
        ),
        "profitable_threshold_max": (
            float(max(profitable_threshold_list)) if profitable_threshold_list else None
        ),
        "selection_quantile": 0.20,
        "selection_reason": reason,
        "local_band_width": float(local_band_width),
        "confirmation_bands": int(confirmation_bands),
        "confirmation_min_positive": int(confirmation_min_positive),
        "simple_sl_mults": [float(x) for x in sl_mults],
        "simple_tp_mults": [float(x) for x in tp_mults],
        "simple_size_power": float(size_power),
        "gross_slippage_buffer": float(SIMPLE_DISCOVERY_GROSS_PNL_SLIPPAGE_BUFFER),
        "simple_policy_count": int(len(simulated_policy_rows)),
        "best_by_threshold": best_by_threshold,
        "all_grid_result_count": int(len(all_threshold_results)),
        "max_concurrent_per_asset": int(max_concurrent_per_asset),
        "cross_symbol_concurrency_enforced": False,
        "cross_strategy_concurrency_enforced": False,
        "dynamic_threshold_enabled": False,
        "threshold_space": "rank_percentile",
    }
    return best


def optimise_deployment_rank_threshold(
    rows: pd.DataFrame,
    *,
    score_col: str = "calibrated_score",
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    side_col: str = "side",
    strategy_col: str = "strategy_id",
    max_concurrent_total: int = PORTFOLIO_POLICY_MAX_CONCURRENT_POSITIONS,
    max_concurrent_per_side: int = PORTFOLIO_POLICY_MAX_CONCURRENT_PER_SIDE,
    max_concurrent_per_asset: int = DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
    max_concurrent_per_strategy: int = DEPLOYMENT_MAX_CONCURRENT_PER_STRATEGY,
    lo: float = DEPLOYMENT_THRESHOLD_MIN,
    hi: float = DEPLOYMENT_THRESHOLD_MAX,
    precision: float = DEPLOYMENT_THRESHOLD_PRECISION,
) -> Dict[str, Any]:
    """Choose a broad profitable deployment rank gate.

    The live portfolio layer handles cross-symbol/strategy capacity. This offline
    gate only forbids overlapping same-symbol trades, then selects the 20th
    percentile of thresholds whose avg gross PnL remains positive after a 0.7%
    slippage/execution buffer. That keeps the top 80% of the acceptable
    threshold band without overfitting to sparse,
    high-threshold pockets.
    """
    if rows.empty:
        return {
            "deployment_rank_threshold": float(hi),
            "objective": float("-inf"),
            "reason": "empty_rows",
        }

    work = rows.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[timestamp_col, score_col]
    )
    if work.empty:
        return {
            "deployment_rank_threshold": float(hi),
            "objective": float("-inf"),
            "reason": "no_valid_rows",
        }

    if "deployment_rank_pct" in work.columns:
        work["deployment_rank_pct"] = pd.to_numeric(
            work["deployment_rank_pct"], errors="coerce"
        )
    elif "rank_pct" in work.columns:
        work["deployment_rank_pct"] = pd.to_numeric(work["rank_pct"], errors="coerce")
    else:
        work["deployment_rank_pct"] = work[score_col].rank(method="max", pct=True)
    work = work.dropna(subset=["deployment_rank_pct"])
    if work.empty:
        return {
            "deployment_rank_threshold": float(hi),
            "objective": float("-inf"),
            "reason": "no_valid_rank_rows",
        }
    cache: Dict[float, Dict[str, Any]] = {}
    total_cap = 1_000_000 if max_concurrent_total is None else int(max_concurrent_total)
    side_cap = (
        1_000_000 if max_concurrent_per_side is None else int(max_concurrent_per_side)
    )
    strategy_cap = (
        1_000_000
        if max_concurrent_per_strategy is None
        else int(max_concurrent_per_strategy)
    )

    def evaluate(threshold: float) -> Dict[str, Any]:
        threshold = float(np.round(threshold, 4))
        if threshold in cache:
            return cache[threshold]

        candidates = work[work["deployment_rank_pct"] >= threshold].copy()
        selected = apply_deployment_concurrency_constraints(
            candidates,
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
            side_col=side_col,
            strategy_col=strategy_col,
            rank_col="deployment_rank_pct",
            initial_rank_threshold=None,
            dynamic_threshold_enabled=False,
            max_concurrent_total=max(1_000_000, total_cap),
            max_concurrent_per_side=max(1_000_000, side_cap),
            max_concurrent_per_asset=max_concurrent_per_asset,
            max_concurrent_per_strategy=max(1_000_000, strategy_cap),
        )
        metrics = score_deployment_threshold_rows(selected)
        objective = (
            metrics["net_pnl"]
            - 0.10 * abs(metrics.get("max_drawdown", 0.0))
            + 0.10 * metrics.get("sortino", 0.0)
        )
        out = {
            "deployment_rank_threshold": threshold,
            "objective": float(objective),
            **metrics,
        }
        cache[threshold] = out
        return out

    threshold_grid = np.arange(
        float(lo), float(hi) + float(precision) / 2.0, float(precision)
    )
    threshold_grid = np.unique(np.round(np.clip(threshold_grid, lo, hi), 4))
    evaluated = [evaluate(float(t)) for t in threshold_grid]
    profitable = [r for r in evaluated if _slippage_adjusted_mean_gross_positive(r)]
    if profitable:
        profitable_thresholds = np.asarray(
            [float(r["deployment_rank_threshold"]) for r in profitable],
            dtype=np.float32,
        )
        selected_threshold = float(np.quantile(profitable_thresholds, 0.20))
        selected_threshold = float(
            min(
                profitable,
                key=lambda r: abs(
                    float(r["deployment_rank_threshold"]) - selected_threshold
                ),
            )["deployment_rank_threshold"]
        )
        best = dict(evaluate(selected_threshold))
        reason = "iq20_positive_slippage_adjusted_mean_gross_trade_band"
    else:
        best = max(
            evaluated,
            key=lambda r: (
                r.get("mean_gross_trade_slippage_adjusted", 0.0),
                r.get("mean_gross_trade", 0.0),
                r["mean_net_trade"],
            ),
        )
        reason = "no_positive_slippage_adjusted_mean_gross_trade_fallback_best_mean"

    profitable_threshold_list = [
        float(r["deployment_rank_threshold"]) for r in profitable
    ]
    best["threshold_search"] = {
        "method": "grid_iq20_positive_slippage_adjusted_mean_gross_trade_symbol_only_concurrency",
        "lo": float(lo),
        "hi": float(hi),
        "precision": float(precision),
        "selected_threshold": float(best["deployment_rank_threshold"]),
        "evaluated_thresholds": sorted(float(k) for k in cache.keys()),
        "profitable_thresholds": profitable_threshold_list,
        "profitable_threshold_count": int(len(profitable_threshold_list)),
        "profitable_threshold_min": (
            float(min(profitable_threshold_list)) if profitable_threshold_list else None
        ),
        "profitable_threshold_max": (
            float(max(profitable_threshold_list)) if profitable_threshold_list else None
        ),
        "selection_quantile": 0.20,
        "selection_reason": reason,
        "max_concurrent_total": int(total_cap),
        "max_concurrent_per_side": int(side_cap),
        "max_concurrent_per_asset": int(max_concurrent_per_asset),
        "max_concurrent_per_strategy": int(strategy_cap),
        "cross_symbol_concurrency_enforced": False,
        "cross_strategy_concurrency_enforced": False,
        "dynamic_threshold_enabled": False,
        "score_col": score_col,
        "gross_slippage_buffer": float(SIMPLE_DISCOVERY_GROSS_PNL_SLIPPAGE_BUFFER),
        "threshold_space": "rank_percentile",
    }
    return best


def _build_deployment_threshold_rows(
    df_top: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    cost_pct: float,
    best_params: Dict[str, Any],
    best_size_power: float,
    metrics: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if metrics is None:
        metrics = simulate_and_score(
            df_top.copy(),
            paths[0],
            paths[1],
            paths[2],
            paths[3],
            cost_pct=cost_pct,
            size_power=best_size_power,
            max_concurrent_trades=max(1, len(df_top) + 1),
            **_without_concurrency_param(best_params),
        )
    selected_mask = np.asarray(metrics.get("selected_mask"), dtype=bool)
    rows = df_top.copy()
    if len(selected_mask) == len(rows):
        rows = rows.iloc[np.flatnonzero(selected_mask)].copy()
    raw_gains = np.asarray(metrics.get("raw_gains", []), dtype=np.float32)
    exit_bars = np.asarray(metrics.get("exit_bars", []), dtype=np.int32)
    if len(rows) != len(raw_gains) or len(rows) != len(exit_bars):
        logger.warning(
            "Skipping deployment threshold optimisation rows due to length mismatch: "
            "rows=%s gains=%s exit_bars=%s",
            len(rows),
            len(raw_gains),
            len(exit_bars),
        )
        return pd.DataFrame()
    rows["net_gain"] = raw_gains
    gross_gains = np.asarray(metrics.get("gross_gains", []), dtype=np.float32)
    if len(gross_gains) == len(rows):
        rows["gross_gain"] = gross_gains
    rows["exit_bars"] = exit_bars
    return rows


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def _clean_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _global_metric_stats(s: pd.Series, *, floor: float) -> Dict[str, float]:
    """
    Compute global cross-asset descriptors once for one policy.

    Robust scale uses the max of IQR / 1.349, MAD * 1.4826, std, and floor.
    """
    cleaned = _clean_series(s)
    if len(cleaned) < 2:
        return {
            "q25": 0.0,
            "q75": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "std": float(floor),
            "mad": float(floor),
            "scale": float(floor),
        }

    desc = cleaned.describe(percentiles=[0.25, 0.75])
    q25 = _safe_float(desc.get("25%", 0.0))
    q75 = _safe_float(desc.get("75%", 0.0))
    median = _safe_float(desc.get("50%", cleaned.median()))
    mean = _safe_float(desc.get("mean", cleaned.mean()))
    std = _safe_float(desc.get("std", cleaned.std(ddof=0)))
    mad = _safe_float((cleaned - median).abs().median())
    iqr = q75 - q25
    scale = max(
        iqr / 1.349 if iqr > 0.0 else 0.0,
        mad * 1.4826 if mad > 0.0 else 0.0,
        std if std > 0.0 else 0.0,
        float(floor),
    )
    return {
        "q25": float(q25),
        "q75": float(q75),
        "median": float(median),
        "mean": float(mean),
        "std": float(std),
        "mad": float(mad),
        "scale": float(scale),
    }


def _reliability(n: float, k: float) -> float:
    n = max(_safe_float(n), 0.0)
    k = max(_safe_float(k), 1e-12)
    return n / (n + k)


def _stable_sigmoid(x: float) -> float:
    clipped = np.clip(float(x), -50.0, 50.0)
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _metric_underperformance(value: float, cutoff: float, scale: float) -> float:
    """Penalise positive-performance metrics only below the bottom quartile."""
    return max(
        0.0,
        (_safe_float(cutoff) - _safe_float(value))
        / max(abs(_safe_float(scale)), 1e-12),
    )


def build_asset_weight_context(
    asset_metrics: pd.DataFrame,
    *,
    pnl_col: str = "mean_net_gain",
    sortino_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Build global reference stats for one policy/strategy asset group."""
    if sortino_col is None:
        if "sortino" in asset_metrics.columns:
            sortino_col = "sortino"
        elif "m_sortino" in asset_metrics.columns:
            sortino_col = "m_sortino"
        else:
            raise ValueError(
                "Missing sortino column: expected 'sortino' or 'm_sortino'."
            )

    pnl_stats = _global_metric_stats(asset_metrics[pnl_col], floor=1e-4)
    sortino_stats = _global_metric_stats(asset_metrics[sortino_col], floor=1.0)
    return {
        "columns": {
            "pnl": pnl_col,
            "sortino": sortino_col,
        },
        "pnl": {
            "portfolio_mean": pnl_stats["mean"],
            "bottom_25_cutoff": pnl_stats["q25"],
            "scale": pnl_stats["scale"],
            "median": pnl_stats["median"],
            "q75": pnl_stats["q75"],
        },
        "sortino": {
            "bottom_25_cutoff": sortino_stats["q25"],
            "scale": sortino_stats["scale"],
            "median": sortino_stats["median"],
            "q75": sortino_stats["q75"],
        },
    }


def compute_asset_weight(
    row: pd.Series,
    context: Dict[str, Any],
    *,
    k_trades: float = 50.0,
    k_candidates: float = 500.0,
    blacklist_margin: float = -0.00025,
    blacklist_reliability_midpoint: float = 0.50,
    blacklist_reliability_width: float = 0.15,
    pnl_weight: float = 0.75,
    sortino_weight: float = 0.25,
    penalty_mode: str = "capped_weighted_average",
    min_weight: float = 0.25,
    max_weight: float = 1.00,
    shrink_to_nonnegative_prior: bool = True,
) -> Dict[str, Any]:
    """Compute a smooth per-asset multiplier and hard blacklist decision."""
    cols = context["columns"]
    n_trades = max(_safe_float(row.get("n_trades", 0.0)), 0.0)
    n_candidates = max(_safe_float(row.get("n_candidates", 0.0)), 0.0)

    trade_rel = _reliability(n_trades, k_trades)
    candidate_rel = _reliability(n_candidates, k_candidates)
    combined_reliability = float(np.sqrt(trade_rel * candidate_rel))

    asset_pnl = _safe_float(row.get(cols["pnl"], 0.0))
    portfolio_pnl = _safe_float(context["pnl"]["portfolio_mean"], 0.0)
    pnl_prior = (
        max(portfolio_pnl, 0.0) if shrink_to_nonnegative_prior else portfolio_pnl
    )
    shrunk_pnl = (
        combined_reliability * asset_pnl + (1.0 - combined_reliability) * pnl_prior
    )
    sortino = _safe_float(row.get(cols["sortino"], 0.0))

    pnl_underperformance = _metric_underperformance(
        shrunk_pnl,
        context["pnl"]["bottom_25_cutoff"],
        context["pnl"]["scale"],
    )
    sortino_underperformance = _metric_underperformance(
        sortino,
        context["sortino"]["bottom_25_cutoff"],
        context["sortino"]["scale"],
    )
    raw_weighted_penalty = (
        pnl_weight * pnl_underperformance + sortino_weight * sortino_underperformance
    )

    if penalty_mode == "max":
        linear_penalty = max(pnl_underperformance, sortino_underperformance)
    elif penalty_mode == "capped_weighted_average":
        linear_penalty = pnl_weight * min(
            pnl_underperformance, 1.0
        ) + sortino_weight * min(sortino_underperformance, 1.0)
    elif penalty_mode == "sqrt_weighted_sum":
        linear_penalty = float(np.sqrt(max(raw_weighted_penalty, 0.0)))
    else:
        raise ValueError(f"Unsupported penalty_mode: {penalty_mode}")

    harmfulness = max(
        0.0,
        (blacklist_margin - shrunk_pnl) / max(abs(context["pnl"]["scale"]), 1e-12),
    )
    sigmoid_arg = (combined_reliability - blacklist_reliability_midpoint) / max(
        blacklist_reliability_width,
        1e-12,
    )
    blacklist_reliability_gate = _stable_sigmoid(sigmoid_arg)
    blacklist_score = harmfulness * blacklist_reliability_gate

    if blacklist_score >= 1.0:
        decision = ASSET_DECISION_BLACKLIST
        multiplier = 0.0
    else:
        multiplier = float(np.clip(1.0 - linear_penalty, min_weight, max_weight))
        decision = (
            ASSET_DECISION_KEEP if multiplier >= 0.95 else ASSET_DECISION_DOWN_WEIGHT
        )

    return {
        "asset_decision": decision,
        "asset_weight_multiplier": float(multiplier),
        "linear_penalty": float(linear_penalty),
        "raw_weighted_penalty": float(raw_weighted_penalty),
        "penalty_mode": penalty_mode,
        "asset_pnl": float(asset_pnl),
        "portfolio_pnl": float(portfolio_pnl),
        "pnl_prior": float(pnl_prior),
        "shrunk_pnl": float(shrunk_pnl),
        "trade_reliability": float(trade_rel),
        "candidate_reliability": float(candidate_rel),
        "combined_reliability": float(combined_reliability),
        "pnl_underperformance": float(pnl_underperformance),
        "sortino_underperformance": float(sortino_underperformance),
        "blacklist_score": float(blacklist_score),
        "blacklist_harmfulness": float(harmfulness),
        "blacklist_reliability_gate": float(blacklist_reliability_gate),
    }


def apply_asset_weights_for_policy(
    asset_metrics: pd.DataFrame,
    *,
    policy_name: str,
    symbol_col: str = "symbol",
    pnl_col: str = "mean_net_gain",
    sortino_col: Optional[str] = None,
    tprint_fn=tprint,
    **weight_kwargs: Any,
) -> pd.DataFrame:
    """Apply smooth asset weights to one policy's asset metrics dataframe."""
    if asset_metrics.empty:
        tprint_fn(f"[asset_weights] policy={policy_name} has no assets.")
        return asset_metrics.copy()
    if symbol_col not in asset_metrics.columns:
        raise ValueError(f"Missing symbol column: {symbol_col}")

    context = build_asset_weight_context(
        asset_metrics,
        pnl_col=pnl_col,
        sortino_col=sortino_col,
    )
    tprint_fn(
        "[asset_weights] "
        f"policy={policy_name} assets={len(asset_metrics)} "
        f"pnl_q25={context['pnl']['bottom_25_cutoff']:.6g} "
        f"pnl_scale={context['pnl']['scale']:.6g} "
        f"sortino_q25={context['sortino']['bottom_25_cutoff']:.6g} "
        f"sortino_scale={context['sortino']['scale']:.6g}"
    )

    weight_rows = asset_metrics.apply(
        lambda row: compute_asset_weight(row, context, **weight_kwargs),
        axis=1,
    )
    weights_df = pd.DataFrame(weight_rows.tolist(), index=asset_metrics.index)
    out = asset_metrics.join(weights_df)
    log_asset_weight_summary(
        out,
        policy_name=policy_name,
        symbol_col=symbol_col,
        tprint_fn=tprint_fn,
    )
    return out


def apply_asset_weights(
    asset_metrics: pd.DataFrame,
    *,
    policy_col: str = "strategy_id",
    symbol_col: str = "symbol",
    pnl_col: str = "mean_net_gain",
    sortino_col: Optional[str] = None,
    tprint_fn=tprint,
    **weight_kwargs: Any,
) -> pd.DataFrame:
    """Apply asset weights per policy/strategy_id."""
    if asset_metrics.empty:
        tprint_fn("[asset_weights] no asset metrics to weight.")
        return asset_metrics.copy()
    if policy_col not in asset_metrics.columns:
        tprint_fn(
            f"[asset_weights] missing policy_col={policy_col}; treating all rows as one policy."
        )
        return apply_asset_weights_for_policy(
            asset_metrics,
            policy_name="GLOBAL",
            symbol_col=symbol_col,
            pnl_col=pnl_col,
            sortino_col=sortino_col,
            tprint_fn=tprint_fn,
            **weight_kwargs,
        )

    parts = []
    for policy_name, grp in asset_metrics.groupby(policy_col, sort=False):
        weighted = apply_asset_weights_for_policy(
            grp.copy(),
            policy_name=str(policy_name),
            symbol_col=symbol_col,
            pnl_col=pnl_col,
            sortino_col=sortino_col,
            tprint_fn=tprint_fn,
            **weight_kwargs,
        )
        parts.append(weighted)
    return pd.concat(parts, axis=0).sort_index()


def log_asset_weight_summary(
    weighted_assets: pd.DataFrame,
    *,
    policy_name: str,
    symbol_col: str = "symbol",
    tprint_fn=tprint,
) -> None:
    """Log metrics per policy and per asset decision group."""
    if weighted_assets.empty:
        tprint_fn(f"[asset_weights] policy={policy_name} empty weighted assets.")
        return

    required = ["asset_decision", "asset_weight_multiplier"]
    missing = [c for c in required if c not in weighted_assets.columns]
    if missing:
        tprint_fn(
            f"[asset_weights] policy={policy_name} cannot log summary; missing={missing}"
        )
        return

    n_assets = int(len(weighted_assets))
    n_trades_total = int(weighted_assets.get("n_trades", pd.Series(dtype=float)).sum())
    n_candidates_total = int(
        weighted_assets.get("n_candidates", pd.Series(dtype=float)).sum()
    )
    tprint_fn(
        "[asset_weights] "
        f"policy={policy_name} total_assets={n_assets} "
        f"total_trades={n_trades_total} total_candidates={n_candidates_total} "
        f"mean_weight={weighted_assets['asset_weight_multiplier'].mean():.3f}"
    )

    for decision in ASSET_DECISIONS:
        grp = weighted_assets[weighted_assets["asset_decision"] == decision]
        if grp.empty:
            tprint_fn(f"[asset_weights] policy={policy_name} group={decision} assets=0")
            continue

        n = int(len(grp))
        share = n / max(n_assets, 1)
        n_trades = int(grp.get("n_trades", pd.Series(dtype=float)).sum())
        n_candidates = int(grp.get("n_candidates", pd.Series(dtype=float)).sum())
        mean_pnl = _safe_float(grp.get("mean_net_gain", pd.Series(dtype=float)).mean())
        median_pnl = _safe_float(
            grp.get("mean_net_gain", pd.Series(dtype=float)).median()
        )
        shrunk_pnl = _safe_float(grp.get("shrunk_pnl", pd.Series(dtype=float)).mean())
        sortino_col = "sortino" if "sortino" in grp.columns else "m_sortino"
        mean_sortino = _safe_float(grp.get(sortino_col, pd.Series(dtype=float)).mean())
        mean_weight = _safe_float(grp["asset_weight_multiplier"].mean())
        min_weight = _safe_float(grp["asset_weight_multiplier"].min())
        max_weight = _safe_float(grp["asset_weight_multiplier"].max())
        mean_penalty = _safe_float(
            grp.get("linear_penalty", pd.Series(dtype=float)).mean()
        )
        mean_reliability = _safe_float(
            grp.get("combined_reliability", pd.Series(dtype=float)).mean()
        )
        mean_dd = _safe_float(grp.get("max_drawdown", pd.Series(dtype=float)).mean())
        median_dd = _safe_float(
            grp.get("max_drawdown", pd.Series(dtype=float)).median()
        )

        tprint_fn(
            "[asset_weights] "
            f"policy={policy_name} group={decision} "
            f"assets={n} share={share:.1%} "
            f"trades={n_trades} candidates={n_candidates} "
            f"mean_weight={mean_weight:.3f} "
            f"weight_range=[{min_weight:.3f},{max_weight:.3f}] "
            f"mean_pnl={mean_pnl:.6g} median_pnl={median_pnl:.6g} "
            f"mean_shrunk_pnl={shrunk_pnl:.6g} "
            f"mean_dd={mean_dd:.6g} median_dd={median_dd:.6g} "
            f"mean_sortino={mean_sortino:.3f} "
            f"mean_penalty={mean_penalty:.3f} "
            f"mean_reliability={mean_reliability:.3f}"
        )

        offender_cols = [
            symbol_col,
            "asset_weight_multiplier",
            "mean_net_gain",
            "shrunk_pnl",
            sortino_col,
            "linear_penalty",
            "blacklist_score",
            "combined_reliability",
            "n_trades",
            "n_candidates",
        ]
        offender_cols = [c for c in offender_cols if c in grp.columns]
        offenders = grp.sort_values(
            ["asset_weight_multiplier", "shrunk_pnl"],
            ascending=[True, True],
        ).head(5)

        for _, row in offenders[offender_cols].iterrows():
            tprint_fn(
                "[asset_weights] "
                f"policy={policy_name} group={decision} sample_asset="
                + " ".join(f"{k}={row[k]}" for k in offender_cols)
            )


def build_asset_metrics_from_simulation(
    selected_rows: pd.DataFrame,
    metrics: Dict[str, Any],
    *,
    symbol_col: str = "symbol",
    policy_col: str = "strategy_id",
    candidate_rows: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build one row per policy x symbol from simulate_and_score outputs.

    selected_rows are the rows passed to simulate_and_score before concurrency
    filtering. candidate_rows should be the broader opportunity set when
    available so n_candidates reflects model selection frequency.
    """
    if symbol_col not in selected_rows.columns:
        raise ValueError(f"Missing symbol_col={symbol_col} in selected_rows.")

    raw_gains = np.asarray(metrics.get("raw_gains", np.array([], dtype=np.float32)))
    sizes = np.asarray(metrics.get("sizes", np.array([], dtype=np.float32)))
    selected_mask = metrics.get("selected_mask")

    rows = selected_rows.copy()
    if selected_mask is not None:
        mask = np.asarray(selected_mask, dtype=bool)
        if len(mask) == len(rows):
            rows = rows.iloc[np.flatnonzero(mask)].copy()

    if len(rows) != len(raw_gains):
        raise ValueError(
            "Length mismatch while building asset metrics: "
            f"rows={len(rows)} raw_gains={len(raw_gains)}"
        )
    if len(rows) == 0:
        return pd.DataFrame()

    rows = rows.copy()
    rows["_net_gain"] = raw_gains
    rows["_size"] = sizes if len(sizes) == len(rows) else np.nan
    rows["_timestamp"] = pd.to_datetime(rows["timestamp"], errors="coerce")
    if policy_col not in rows.columns:
        rows[policy_col] = "GLOBAL"

    if candidate_rows is None:
        candidate_rows = selected_rows
    cand = candidate_rows.copy()
    if policy_col not in cand.columns:
        cand[policy_col] = "GLOBAL"
    candidate_counts = (
        cand.groupby([policy_col, symbol_col], observed=False)
        .size()
        .rename("n_candidates")
    )

    out_rows = []
    for (policy_name, symbol), grp in rows.groupby(
        [policy_col, symbol_col],
        observed=False,
        sort=False,
    ):
        g = grp.sort_values("_timestamp").dropna(subset=["_timestamp", "_net_gain"])
        if g.empty:
            continue

        pnl = g["_net_gain"].astype(float)
        cum = pnl.cumsum()
        drawdown = cum - cum.cummax()
        downside = pnl[pnl < 0.0]
        if len(downside) == 0 or downside.std(ddof=0) == 0:
            sortino = 100.0 if pnl.mean() > 0.0 else 0.0
        else:
            sortino = float(pnl.mean() / np.sqrt(np.mean(downside**2)))

        wins = pnl[pnl > 0.0]
        losses = pnl[pnl < 0.0]
        gross_win = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
        profit_factor = gross_win / gross_loss if gross_loss > 0.0 else np.inf
        key = (policy_name, symbol)
        n_candidates = int(candidate_counts.get(key, len(grp)))

        out_rows.append(
            {
                policy_col: policy_name,
                symbol_col: symbol,
                "n_trades": int(len(g)),
                "n_candidates": n_candidates,
                "mean_net_gain": float(pnl.mean()),
                "total_net_pnl": float(pnl.sum()),
                "hit_rate": float((pnl > 0.0).mean()),
                "avg_win": float(wins.mean()) if len(wins) else 0.0,
                "avg_loss": float(losses.mean()) if len(losses) else 0.0,
                "profit_factor": float(profit_factor),
                "sortino": float(sortino),
                "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
                "start_date": str(g["_timestamp"].min().date()),
                "end_date": str(g["_timestamp"].max().date()),
            }
        )

    return pd.DataFrame(out_rows)


def optimise_position_sizing(
    df_sub: pd.DataFrame,
    f_opens: np.ndarray,
    f_highs: np.ndarray,
    f_lows: np.ndarray,
    f_closes: np.ndarray,
    cost_pct: float,
    best_trailing_params: dict,
) -> Tuple[float, float, Dict[str, Any]]:
    best_size_power = 1.0
    best_pnl = float("-inf")
    best_metrics = {}

    SIZE_POWER_GRID = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for size_power in SIZE_POWER_GRID:
        metrics = simulate_and_score(
            df_sub,
            f_opens,
            f_highs,
            f_lows,
            f_closes,
            cost_pct=cost_pct,
            size_power=size_power,
            **best_trailing_params,
        )
        if metrics["net_pnl"] > best_pnl:
            best_pnl = metrics["net_pnl"]
            best_size_power = size_power
            best_metrics = metrics

    return best_size_power, best_pnl, best_metrics


def calculate_advanced_metrics(
    df_sub: pd.DataFrame,
    raw_gains: np.ndarray,
    sizes: np.ndarray,
    selected_mask: Optional[np.ndarray] = None,
    gross_gains: Optional[np.ndarray] = None,
    exit_reasons: Optional[Sequence[Any]] = None,
    exit_bars: Optional[Sequence[Any]] = None,
) -> dict:
    if len(raw_gains) == 0:
        return {}

    exit_reasons_arr = (
        np.asarray(exit_reasons, dtype=object)
        if exit_reasons is not None
        else np.full(len(raw_gains), "unknown", dtype=object)
    )
    exit_bars_arr = (
        np.asarray(exit_bars, dtype=np.float32)
        if exit_bars is not None
        else np.full(len(raw_gains), np.nan, dtype=np.float32)
    )
    if selected_mask is not None:
        mask = np.asarray(selected_mask, dtype=bool)
        if len(mask) == len(df_sub):
            df_sub = df_sub.iloc[np.flatnonzero(mask)].copy()
        if len(mask) == len(exit_reasons_arr):
            exit_reasons_arr = exit_reasons_arr[mask]
        if len(mask) == len(exit_bars_arr):
            exit_bars_arr = exit_bars_arr[mask]

    if gross_gains is None:
        gross_gains_arr = np.full(len(raw_gains), np.nan, dtype=np.float32)
    else:
        gross_gains_arr = np.asarray(gross_gains, dtype=np.float32)

    if (
        len(raw_gains) != len(df_sub)
        or len(sizes) != len(df_sub)
        or len(gross_gains_arr) != len(df_sub)
        or len(exit_reasons_arr) != len(df_sub)
        or len(exit_bars_arr) != len(df_sub)
    ):
        logger.warning(
            "Skipping advanced metrics due to length mismatch: "
            "rows=%s gains=%s gross_gains=%s sizes=%s exit_reasons=%s exit_bars=%s",
            len(df_sub),
            len(raw_gains),
            len(gross_gains_arr),
            len(sizes),
            len(exit_reasons_arr),
            len(exit_bars_arr),
        )
        return {}

    df_trades = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df_sub["timestamp"].values),
            "net_gain": raw_gains,
            "gross_gain": gross_gains_arr,
            "size": sizes,
            "exit_reason": exit_reasons_arr,
            "exit_bars": exit_bars_arr,
        }
    )
    df_trades = df_trades[np.isfinite(df_trades["net_gain"])]
    if len(df_trades) == 0:
        return {}

    df_trades = df_trades.sort_values("timestamp")
    df_trades.set_index("timestamp", inplace=True)

    start_date = df_trades.index.min()
    end_date = df_trades.index.max()
    n_trades = len(df_trades)

    avg_pnl_bankroll = df_trades["net_gain"].mean()
    df_trades["rop"] = df_trades["net_gain"] / df_trades["size"]
    df_trades["gross_rop"] = df_trades["gross_gain"] / df_trades["size"]
    avg_pnl_sized = df_trades["rop"].mean()
    avg_gross_pnl_per_trade = df_trades["gross_gain"].mean()
    avg_gross_return_per_trade = df_trades["gross_rop"].mean()

    pnl_positive_rate = (df_trades["net_gain"] > 0).mean()
    exit_reason_counts = df_trades["exit_reason"].astype(str).value_counts()
    holding_metrics = _holding_time_metrics(df_trades["exit_bars"])

    def _exit_count(reason: str) -> int:
        return int(exit_reason_counts.get(reason, 0))

    trailing_profit_exit_count = _exit_count("trailing")
    capital_protect_exit_count = _exit_count("capital_protect")
    full_sl_exit_count = _exit_count("full_sl")
    adverse_fast_exit_count = _exit_count("adverse_exit")
    timeout_exit_count = _exit_count("timeout")
    known_exit_count = (
        trailing_profit_exit_count
        + capital_protect_exit_count
        + full_sl_exit_count
        + adverse_fast_exit_count
        + timeout_exit_count
    )
    unknown_exit_count = int(max(0, n_trades - known_exit_count))

    def _exit_rate(count: int) -> float:
        return float(count / max(n_trades, 1))

    # Reporting hit rate intentionally means trailing-profit exits only.
    hit_rate = _exit_rate(trailing_profit_exit_count)

    winning_trades = df_trades[df_trades["rop"] > 0]["rop"]
    losing_trades = df_trades[df_trades["rop"] < 0]["rop"]
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0

    w_pnl = df_trades["net_gain"].resample("W").sum().fillna(0.0)
    m_pnl = df_trades["net_gain"].resample("ME").sum().fillna(0.0)
    weekly_hit_rate = (
        (df_trades["exit_reason"].astype(str) == "trailing")
        .resample("W")
        .mean()
        .dropna()
    )
    weekly_pnl_positive_rate = (df_trades["net_gain"] > 0).resample("W").mean().dropna()

    w_std = w_pnl.std()
    m_std = m_pnl.std()

    def sortino(pnl_series):
        downside = pnl_series[pnl_series < 0]
        if len(downside) == 0 or downside.std(ddof=0) == 0:
            return 100.0 if pnl_series.mean() > 0 else 0.0
        return pnl_series.mean() / np.sqrt(np.mean(downside**2))

    w_sortino = sortino(w_pnl)
    m_sortino = sortino(m_pnl)

    cum_pnl = df_trades["net_gain"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    tuw_max = pd.Timedelta(seconds=0)
    is_high = drawdown == 0
    if not is_high.all():
        high_dates = df_trades.index[is_high]
        if len(high_dates) > 0:
            all_highs = list(high_dates) + [df_trades.index[-1]]
            for i in range(1, len(all_highs)):
                dur = all_highs[i] - all_highs[i - 1]
                if dur > tuw_max:
                    tuw_max = dur
        else:
            tuw_max = df_trades.index[-1] - df_trades.index[0]
    tuw_days = tuw_max.total_seconds() / 86400.0

    material_tuw_durations: List[float] = []
    material_threshold = -0.20 * abs(float(max_dd)) if float(max_dd) < 0.0 else 0.0
    if material_threshold < 0.0:
        start_ts: Optional[pd.Timestamp] = None
        for ts, flag in (drawdown <= material_threshold).items():
            if bool(flag) and start_ts is None:
                start_ts = pd.Timestamp(ts)
            elif not bool(flag) and start_ts is not None:
                material_tuw_durations.append(
                    (pd.Timestamp(ts) - start_ts).total_seconds() / 86400.0
                )
                start_ts = None
        if start_ts is not None:
            material_tuw_durations.append(
                (df_trades.index[-1] - start_ts).total_seconds() / 86400.0
            )
    material_tuw20_p90_days = (
        float(np.quantile(material_tuw_durations, 0.90))
        if material_tuw_durations
        else 0.0
    )

    def _series_quantile(series: pd.Series, q: float) -> float:
        clean = pd.to_numeric(series, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        clean = clean.dropna()
        return float(clean.quantile(q)) if len(clean) else 0.0

    return {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "n_trades": n_trades,
        "avg_pnl_bankroll": avg_pnl_bankroll,
        "avg_pnl_sized": avg_pnl_sized,
        "avg_gross_pnl_per_trade": avg_gross_pnl_per_trade,
        "avg_gross_return_per_trade": avg_gross_return_per_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "hit_rate": hit_rate,
        "hit_rate_definition": "trailing_profit_exit_rate",
        "pnl_positive_rate": pnl_positive_rate,
        "trailing_profit_exit_count": trailing_profit_exit_count,
        "trailing_profit_exit_rate": _exit_rate(trailing_profit_exit_count),
        "capital_protect_exit_count": capital_protect_exit_count,
        "capital_protect_exit_rate": _exit_rate(capital_protect_exit_count),
        "full_sl_exit_count": full_sl_exit_count,
        "full_sl_exit_rate": _exit_rate(full_sl_exit_count),
        "adverse_fast_exit_count": adverse_fast_exit_count,
        "adverse_fast_exit_rate": _exit_rate(adverse_fast_exit_count),
        "timeout_exit_count": timeout_exit_count,
        "timeout_exit_rate": _exit_rate(timeout_exit_count),
        "unknown_exit_count": unknown_exit_count,
        "unknown_exit_rate": _exit_rate(unknown_exit_count),
        "w_sortino": w_sortino,
        "m_sortino": m_sortino,
        "w_std": w_std,
        "m_std": m_std,
        "weekly_pnl_std": w_std,
        "monthly_pnl_std": m_std,
        "worst_week": float(w_pnl.min()) if len(w_pnl) else 0.0,
        "max_dd": max_dd,
        "max_drawdown": max_dd,
        "tuw_days": tuw_days,
        "time_under_water_days": tuw_days,
        "expected_drawdown_adjusted_tuw": float(tuw_days * abs(float(max_dd))),
        "material_tuw20_p90_days": material_tuw20_p90_days,
        "weekly_pnl_q10": _series_quantile(w_pnl, 0.10),
        "weekly_pnl_q50": _series_quantile(w_pnl, 0.50),
        "weekly_pnl_q90": _series_quantile(w_pnl, 0.90),
        "weekly_pnl_q90_q10_delta": _series_quantile(w_pnl, 0.90)
        - _series_quantile(w_pnl, 0.10),
        "weekly_hit_rate_q10": _series_quantile(weekly_hit_rate, 0.10),
        "weekly_hit_rate_q50": _series_quantile(weekly_hit_rate, 0.50),
        "weekly_hit_rate_q90": _series_quantile(weekly_hit_rate, 0.90),
        "weekly_hit_rate_q90_q10_delta": _series_quantile(weekly_hit_rate, 0.90)
        - _series_quantile(weekly_hit_rate, 0.10),
        "weekly_pnl_positive_rate_q10": _series_quantile(
            weekly_pnl_positive_rate, 0.10
        ),
        "weekly_pnl_positive_rate_q50": _series_quantile(
            weekly_pnl_positive_rate, 0.50
        ),
        "weekly_pnl_positive_rate_q90": _series_quantile(
            weekly_pnl_positive_rate, 0.90
        ),
        "weekly_pnl_positive_rate_q90_q10_delta": _series_quantile(
            weekly_pnl_positive_rate, 0.90
        )
        - _series_quantile(weekly_pnl_positive_rate, 0.10),
        **holding_metrics,
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def _path_take(
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], idx: np.ndarray
):
    idx = np.asarray(idx, dtype=np.int64)
    return tuple(arr[idx] for arr in paths)


def _fetch_policy_paths(
    df_subset: pd.DataFrame,
    ds: Any,
    *,
    path_len: int = DEFAULT_FORWARD_BARS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_events = len(df_subset)
    f_op = np.full((n_events, path_len), np.nan, dtype=np.float32)
    f_hi = np.full((n_events, path_len), np.nan, dtype=np.float32)
    f_lo = np.full((n_events, path_len), np.nan, dtype=np.float32)
    f_cl = np.full((n_events, path_len), np.nan, dtype=np.float32)

    for symbol, group in df_subset.groupby("symbol"):
        klines = ds.load(symbol)
        if klines is None or len(klines) == 0:
            continue
        klines = klines.reset_index()
        if "ts" not in klines.columns and "index" in klines.columns:
            klines = klines.rename(columns={"index": "ts"})

        k_ts = klines["ts"].astype("int64").values // 10**6
        rel_pos_by_index = {
            idx: pos for pos, idx in enumerate(df_subset.index.to_numpy())
        }
        for df_idx, row in group.iterrows():
            rel_idx = rel_pos_by_index.get(df_idx)
            if rel_idx is None:
                continue
            event_ts = int(pd.Timestamp(row["timestamp"]).timestamp() * 1000)

            idx_arr = np.searchsorted(k_ts, event_ts)
            if idx_arr >= len(k_ts):
                continue
            end_idx = min(idx_arr + path_len, len(klines))
            actual_len = end_idx - idx_arr
            if actual_len <= 0:
                continue
            opens = klines["open"].values[idx_arr:end_idx]
            highs = klines["high"].values[idx_arr:end_idx]
            lows = klines["low"].values[idx_arr:end_idx]
            closes = klines["close"].values[idx_arr:end_idx]
            f_op[rel_idx, :actual_len] = opens
            f_hi[rel_idx, :actual_len] = highs
            f_lo[rel_idx, :actual_len] = lows
            f_cl[rel_idx, :actual_len] = closes
            if actual_len < path_len:
                last_close = closes[-1]
                f_op[rel_idx, actual_len:] = last_close
                f_hi[rel_idx, actual_len:] = last_close
                f_lo[rel_idx, actual_len:] = last_close
                f_cl[rel_idx, actual_len:] = last_close
    return f_op, f_hi, f_lo, f_cl


def _path_extrema_from_policy_paths(
    df_rows: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute MFE/MAE magnitude and timing from the simple-policy path matrix."""
    f_opens, f_highs, f_lows, _f_closes = paths
    n = len(df_rows)
    mfe = np.full(n, np.nan, dtype=np.float32)
    mae = np.full(n, np.nan, dtype=np.float32)
    t_mfe = np.full(n, np.nan, dtype=np.float32)
    t_mae = np.full(n, np.nan, dtype=np.float32)
    if n == 0 or f_opens.shape[0] != n or f_highs.shape[0] != n or f_lows.shape[0] != n:
        return mfe, mae, t_mfe, t_mae
    entry = np.asarray(f_opens[:, 0], dtype=np.float64)
    side = (
        pd.to_numeric(df_rows["side"], errors="coerce")
        .fillna(1.0)
        .to_numpy(dtype=np.float64)
        if "side" in df_rows.columns
        else np.ones(n, dtype=np.float64)
    )
    is_long = side >= 0.0
    valid_entry = np.isfinite(entry) & (entry > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fav = np.where(
            is_long[:, None],
            (f_highs.astype(np.float64) - entry[:, None]) / entry[:, None],
            (entry[:, None] - f_lows.astype(np.float64)) / entry[:, None],
        )
        adv = np.where(
            is_long[:, None],
            (entry[:, None] - f_lows.astype(np.float64)) / entry[:, None],
            (f_highs.astype(np.float64) - entry[:, None]) / entry[:, None],
        )
    fav = np.where(np.isfinite(fav), np.maximum(fav, 0.0), np.nan)
    adv = np.where(np.isfinite(adv), np.maximum(adv, 0.0), np.nan)
    for i in np.flatnonzero(valid_entry):
        if not np.isfinite(fav[i]).any() or not np.isfinite(adv[i]).any():
            continue
        fav_i = np.nan_to_num(fav[i], nan=-np.inf)
        adv_i = np.nan_to_num(adv[i], nan=-np.inf)
        fav_idx = int(np.argmax(fav_i))
        adv_idx = int(np.argmax(adv_i))
        mfe[i] = float(fav_i[fav_idx]) if np.isfinite(fav_i[fav_idx]) else np.nan
        mae[i] = float(adv_i[adv_idx]) if np.isfinite(adv_i[adv_idx]) else np.nan
        t_mfe[i] = float(fav_idx + 1)
        t_mae[i] = float(adv_idx + 1)
    return mfe, mae, t_mfe, t_mae


def _raw_return_from_policy_paths(
    df_rows: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Fallback realized return over the policy horizon, unsized and before policy exits."""
    f_opens, _f_highs, _f_lows, f_closes = paths
    n = len(df_rows)
    out = np.full(n, np.nan, dtype=np.float32)
    if "u_policy_net" in df_rows.columns:
        vals = pd.to_numeric(df_rows["u_policy_net"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if np.isfinite(vals).any():
            return vals
    if "u_policy" in df_rows.columns:
        vals = pd.to_numeric(df_rows["u_policy"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if np.isfinite(vals).any():
            return vals
    if "return" in df_rows.columns:
        vals = pd.to_numeric(df_rows["return"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if np.isfinite(vals).any():
            return vals
    if n == 0 or f_opens.shape[0] != n or f_closes.shape[0] != n:
        return out
    entry = np.asarray(f_opens[:, 0], dtype=np.float64)
    side = (
        pd.to_numeric(df_rows["side"], errors="coerce")
        .fillna(1.0)
        .to_numpy(dtype=np.float64)
        if "side" in df_rows.columns
        else np.ones(n, dtype=np.float64)
    )
    finite_close = np.isfinite(f_closes)
    last_pos = np.maximum(np.sum(finite_close, axis=1) - 1, 0)
    close = f_closes[np.arange(n), last_pos].astype(np.float64, copy=False)
    valid = np.isfinite(entry) & (entry > 0.0) & np.isfinite(close)
    out[valid] = (side[valid] * (close[valid] / entry[valid] - 1.0)).astype(np.float32)
    return out


def _regime_used_feature_columns(df: pd.DataFrame) -> List[str]:
    blocked_exact = {
        "timestamp",
        "ts",
        "symbol",
        "strategy_id",
        "side",
        "return",
        "u_policy",
        "u_policy_net",
        "rank_pct",
        "deployment_rank_pct",
        "raw_meta_prediction",
        "calibrated_score",
        "mfe_ret",
        "mae_ret",
        "t_mfe",
        "t_mae",
        "exit_bars",
        "net_gain",
        "gross_gain",
    }
    blocked_substrings = ("future_", "realized", "outcome", "label")
    cols: List[str] = []
    for c in df.columns:
        name = str(c)
        low = name.lower()
        if low in blocked_exact or any(s in low for s in blocked_substrings):
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(name)
        except Exception:
            continue
    priority = [
        c
        for c in (
            "clf",
            "oof_meta_clf",
            "oof_pred",
            "oof_p_move",
            "base_H10",
            "base_H5",
            "base_H4",
            "base_H2",
            "base_H1",
            "oof_base_clf",
            "base_clf_centered",
            "clf_entropy",
            "oof_ebm_raw",
            "oof_ebm_en",
            "oof_ebm_uncertainty_weighted",
            "oof_ebm_unc_logodds_var",
            "oof_ebm_unc_pi_width",
            "oof_ebm_unc_entropy_mean",
            "oof_ebm_unc_conflict_norm",
            "oof_ebm_unc_support_mean",
            "oof_ebm_unc_support_min",
            "oof_ebm_unc_support_adjusted_uncertainty",
            "oof_ebm_unc_uncertainty_weight",
        )
        if c in cols
    ]
    return list(dict.fromkeys(priority + cols))


def _strip_policy_side(strategy_id: str) -> str:
    for prefix in ("long_", "short_"):
        if str(strategy_id).startswith(prefix):
            return str(strategy_id)[len(prefix) :]
    return str(strategy_id)


def _base_oof_context_columns(df: pd.DataFrame, context_name: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "oof_prob" in df.columns:
        out[context_name] = pd.to_numeric(df["oof_prob"], errors="coerce")
    elif "oof_pred" in df.columns:
        out[context_name] = pd.to_numeric(df["oof_pred"], errors="coerce")
    for src, suffix in (
        ("oof_sigma_trees", "_sigma"),
        ("oof_sigma_robust", "_robust_sigma"),
    ):
        if src in df.columns:
            out[f"{context_name}{suffix}"] = pd.to_numeric(df[src], errors="coerce")
    for col in [c for c in df.columns if str(c).startswith("oof_tree_")]:
        out[f"{context_name}_{str(col).replace('oof_tree_', '')}"] = pd.to_numeric(
            df[col], errors="coerce"
        )
    for col in ("timestamp", "symbol", "index"):
        if col in df.columns:
            out[col] = df[col].values
    return out


def _merge_prediction_context(
    df: pd.DataFrame,
    ctx: pd.DataFrame,
    *,
    prefix_existing: bool = False,
) -> Tuple[pd.DataFrame, int]:
    if df.empty or ctx.empty:
        return df, 0
    out = df.copy()
    ctx = ctx.copy()
    add_cols = [
        c
        for c in ctx.columns
        if c not in {"timestamp", "symbol", "index"} and c not in out.columns
    ]
    if not add_cols:
        return out, 0
    common = {c for c in ("timestamp", "symbol", "index") if c in out.columns and c in ctx.columns}
    key_options: List[List[str]] = []
    # The meta/base OOF parquet "index" column is often local to that file. Prefer
    # the event identity, otherwise it can turn a valid timestamp/symbol match into
    # an all-null join.
    if {"timestamp", "symbol"}.issubset(common):
        key_options.append(["timestamp", "symbol"])
    if "index" in common:
        key_options.append(["index"])
    if "timestamp" in common:
        key_options.append(["timestamp"])
    if "timestamp" in common:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True, errors="coerce")
    if "symbol" in common:
        out["symbol"] = out["symbol"].astype(str)
        ctx["symbol"] = ctx["symbol"].astype(str)
    for keys in key_options:
        right = ctx[keys + add_cols].drop_duplicates(subset=keys, keep="first")
        merged = out.merge(right, on=keys, how="left", sort=False)
        coverage = 0.0
        if add_cols:
            coverage = float(merged[add_cols].notna().any(axis=1).mean())
        if coverage > 0.0:
            added = sum(c in merged.columns for c in add_cols)
            return merged, int(added)
    if len(ctx) != len(out):
        return out, 0
    for col in add_cols:
        target = f"context_{col}" if prefix_existing and col in out.columns else col
        if target not in out.columns:
            out[target] = ctx[col].to_numpy()
    return out, len(add_cols)


def _load_base_prediction_context(
    *,
    data_root: str,
    run_id: str,
    strategy_id: str,
    stage_view: Dict[str, Any],
) -> pd.DataFrame:
    oof_dir = Path(data_root) / "artifacts" / run_id / "oof"
    if not oof_dir.exists():
        return pd.DataFrame()
    canonical = _strip_policy_side(strategy_id)
    frames: List[pd.DataFrame] = []
    for path in sorted(oof_dir.glob("oof_*.parquet")):
        stem = path.stem.replace("oof_", "", 1)
        if not stem.startswith(f"{canonical}_H") and not stem.startswith(
            f"{strategy_id}_H"
        ):
            continue
        suffix = stem[len(canonical) + 1 :] if stem.startswith(canonical) else stem
        context_name = f"base_{suffix}" if suffix.startswith("H") else f"base_{stem}"
        try:
            raw = pd.read_parquet(path)
            filt = _filter_rows_to_stage_view(raw, stage_view)
            frames.append(_base_oof_context_columns(filt, context_name))
        except Exception as exc:
            logger.warning(
                "[%s] Failed to load base OOF context %s: %s",
                strategy_id,
                path,
                exc,
            )
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out, _ = _merge_prediction_context(out, frame)
    return out


def _ensure_regime_prediction_context(
    df: pd.DataFrame,
    *,
    data_root: str,
    run_id: str,
    strategy_id: str,
    stage_view: Dict[str, Any],
) -> pd.DataFrame:
    """Attach OOS base/meta/EBM prediction context for regime-adaptor features."""
    out = df.copy()
    if "oof_meta_clf" not in out.columns:
        for col in ("clf", "oof_p_move", "oof_pred"):
            if col in out.columns:
                out["oof_meta_clf"] = pd.to_numeric(out[col], errors="coerce")
                break
    if "oof_base_clf" not in out.columns and "base_clf_centered" in out.columns:
        out["oof_base_clf"] = np.clip(
            pd.to_numeric(out["base_clf_centered"], errors="coerce") + 0.5,
            0.0,
            1.0,
        )
    base_ctx = _load_base_prediction_context(
        data_root=data_root,
        run_id=run_id,
        strategy_id=strategy_id,
        stage_view=stage_view,
    )
    out, added = _merge_prediction_context(out, base_ctx)
    if added:
        logger.info(
            "[%s] Added %s base OOF prediction/uncertainty context columns for regime adaptor.",
            strategy_id,
            added,
        )
    ebm_cols = [c for c in out.columns if str(c).startswith("oof_ebm")]
    pred_cols = [
        c
        for c in out.columns
        if c in {"oof_base_clf", "oof_meta_clf", "oof_ebm_raw", "oof_ebm_en", "oof_ebm_uncertainty_weighted"}
        or str(c).startswith("base_H")
    ]
    out.attrs["regime_prediction_context"] = {
        "base_meta_prediction_columns": sorted(pred_cols),
        "ebm_prediction_or_uncertainty_columns": sorted(ebm_cols),
    }
    return out


def _fit_regime_adaptor_from_simple_policy(
    *,
    data_root: str,
    run_id: str,
    strategy_id: str,
    df_policy_all: pd.DataFrame,
    all_policy_paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    trade_idx: np.ndarray,
    final_params: Dict[str, Any],
    final_size_power: float,
    cost_pct: float,
    deployment_rank_threshold: float,
    market_mode: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if len(df_policy_all) < 50 or len(trade_idx) < 10:
        return None
    from extreme_price_movements.regime_adaptor import (
        fit_regime_adaptor,
        save_regime_adaptor_outputs,
    )

    df_top = df_policy_all.iloc[trade_idx].copy().reset_index(drop=True)
    top_paths = _path_take(all_policy_paths, trade_idx)
    policy_metrics = simulate_and_score(
        df_top.copy(),
        top_paths[0],
        top_paths[1],
        top_paths[2],
        top_paths[3],
        cost_pct=cost_pct,
        size_power=final_size_power,
        **final_params,
    )
    selected_mask_top = np.asarray(policy_metrics.get("selected_mask"), dtype=bool)
    raw_gains = np.asarray(policy_metrics.get("raw_gains", []), dtype=np.float32)
    gross_gains = np.asarray(policy_metrics.get("gross_gains", []), dtype=np.float32)
    if len(selected_mask_top) != len(df_top) or len(raw_gains) != int(
        np.sum(selected_mask_top)
    ):
        logger.warning(
            "[%s] Regime adaptor skipped: simple-policy selection length mismatch "
            "rows=%s selected_mask=%s gains=%s",
            strategy_id,
            len(df_top),
            len(selected_mask_top),
            len(raw_gains),
        )
        return None
    selected_top_idx = np.flatnonzero(selected_mask_top)
    if len(selected_top_idx) < 50:
        logger.warning(
            "[%s] Regime adaptor skipped: optimized simple policy selected only %s rows.",
            strategy_id,
            len(selected_top_idx),
        )
        return None
    selected_full_idx = np.asarray(trade_idx, dtype=np.int64)[selected_top_idx]
    n_policy = len(df_policy_all)
    policy_candidate_mask = np.zeros(n_policy, dtype=bool)
    policy_candidate_mask[selected_full_idx] = True
    policy_returns = np.full(n_policy, np.nan, dtype=np.float32)
    policy_returns[selected_full_idx] = raw_gains.astype(np.float32, copy=False)
    gross_returns = np.full(n_policy, np.nan, dtype=np.float32)
    if len(gross_gains) == len(raw_gains):
        gross_returns[selected_full_idx] = gross_gains.astype(np.float32, copy=False)
    mfe, mae, t_mfe, t_mae = _path_extrema_from_policy_paths(
        df_policy_all, all_policy_paths
    )
    raw_returns = _raw_return_from_policy_paths(df_policy_all, all_policy_paths)
    scores = pd.to_numeric(df_policy_all["calibrated_score"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    timestamps = (
        pd.to_datetime(df_policy_all["timestamp"], utc=True, errors="coerce").to_numpy()
        if "timestamp" in df_policy_all.columns
        else None
    )
    symbols = (
        df_policy_all["symbol"].astype(str).to_numpy()
        if "symbol" in df_policy_all.columns
        else np.repeat("all", n_policy)
    )
    fit = fit_regime_adaptor(
        feature_frame=df_policy_all,
        pred_calibrated=scores,
        returns=raw_returns,
        timestamps=timestamps,
        symbols=symbols,
        strategy_id=strategy_id,
        model_name="simple_policy_optimiser",
        cost_pct=cost_pct,
        used_feature_columns=_regime_used_feature_columns(df_policy_all),
        policy_candidate_mask=policy_candidate_mask,
        gross_returns=gross_returns,
        policy_returns=policy_returns,
        mfe=mfe,
        mae=mae,
        t_mfe=t_mfe,
        t_mae=t_mae,
    )
    fit.artifact["foundation"] = "simple_policy_optimiser"
    fit.artifact["prediction_context"] = {
        "base_meta_prediction_columns": sorted(
            [
                c
                for c in df_policy_all.columns
                if c
                in {
                    "clf",
                    "oof_pred",
                    "oof_p_move",
                    "oof_base_clf",
                    "oof_meta_clf",
                    "raw_meta_prediction",
                    "calibrated_score",
                }
                or str(c).startswith("base_H")
            ]
        ),
        "ebm_prediction_or_uncertainty_columns": sorted(
            [c for c in df_policy_all.columns if str(c).startswith("oof_ebm")]
        ),
        "base_oof_context_source": f"artifacts/{run_id}/oof lightweight parquet join",
    }
    fit.artifact["policy_candidate_mask"] = {
        "available": True,
        "source": "simple_policy_optimiser_final_policy_selected_mask",
        "deployment_rank_threshold": float(deployment_rank_threshold),
        "rank_threshold_candidates": int(len(trade_idx)),
        "selected_after_policy_concurrency": int(len(selected_full_idx)),
        "rows": int(n_policy),
        "coverage": float(len(selected_full_idx) / max(n_policy, 1)),
    }
    fit.artifact["policy_realized_utility"] = {
        "available": True,
        "source": "simple_policy_optimiser.simulate_and_score.final_params",
        "rows": int(len(selected_full_idx)),
        "mean_policy_net_utility": float(np.nanmean(raw_gains)),
        "best_size_power": float(final_size_power),
        "max_concurrent_trades": int(
            final_params.get("max_concurrent_trades", MAX_CONCURRENT_TRADES)
        ),
        "has_barwise_paths": True,
    }
    artifact_path = save_regime_adaptor_outputs(
        data_root=data_root,
        run_id=run_id,
        strategy_id=strategy_id,
        fit=fit,
        market_mode=market_mode,
    )
    logger.info(
        "[%s] Regime adaptor trained from simple_policy_optimiser: "
        "selected=%s/%s mean_policy_net_utility=%.6f inference_enabled=%s artifact=%s",
        strategy_id,
        len(selected_full_idx),
        n_policy,
        float(np.nanmean(raw_gains)),
        bool(fit.artifact.get("enable_regime_adaptor_inference", False)),
        artifact_path,
    )
    return {
        "artifact_path": str(artifact_path),
        "research_enabled": bool(fit.artifact.get("enable_regime_adaptor", False)),
        "inference_enabled": bool(
            fit.artifact.get("enable_regime_adaptor_inference", False)
        ),
        "training_universe": fit.artifact.get("training_universe"),
        "outcome_source": fit.artifact.get("outcome_source"),
        "target_counts": (
            fit.artifact.get("trust_model", {}).get("target_counts")
            if isinstance(fit.artifact.get("trust_model"), dict)
            else None
        ),
        "selection_score": fit.artifact.get("selection_score"),
    }


def _suggest_policy_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "sl_mult": trial.suggest_categorical(
            "sl_mult",
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        ),
        "trailing_activation_mult": trial.suggest_categorical(
            "trailing_activation_mult", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
        ),
        "trailing_power": trial.suggest_categorical(
            "trailing_power", [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        ),
        "trailing_squash_divisor": trial.suggest_categorical(
            "trailing_squash_divisor", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
        ),
        "giveback_beta": trial.suggest_categorical(
            "giveback_beta", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        ),
        "capital_protect_mfe_mult": trial.suggest_categorical(
            "capital_protect_mfe_mult", [0.0, 0.75, 1.0, 1.25, 1.5]
        ),
        "capital_protect_regression_frac": trial.suggest_categorical(
            "capital_protect_regression_frac", [0.25, 0.35, 0.45, 0.55, 0.65]
        ),
    }


def _suggest_trailing_stage_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = _suggest_policy_params(trial)
    params["capital_protect_mfe_mult"] = 0.0
    params["capital_protect_regression_frac"] = 0.45
    params["adverse_exit_enabled"] = False
    return params


def _suggest_stage2_params(
    trial: optuna.Trial,
    trailing_stage_params: Dict[str, Any],
) -> Dict[str, Any]:
    params = dict(trailing_stage_params)
    sl_mult = trial.suggest_float("sl_mult", 0.5, 3.5, step=0.1)
    adverse_mae_frac = trial.suggest_float(
        "adverse_exit_min_mae_sl_frac",
        0.20,
        ADVERSE_EXIT_MAX_SL_FRACTION,
        step=0.05,
    )
    adverse_mae_atr = float(
        max(
            ADVERSE_EXIT_MIN_MAE_ATR_FLOOR,
            np.floor(float(sl_mult) * adverse_mae_frac * 10.0) / 10.0,
        )
    )
    params.update(
        {
            "sl_mult": sl_mult,
            "capital_protect_mfe_mult": trial.suggest_float(
                "capital_protect_mfe_mult", 0.0, 3.0, step=0.1
            ),
            "capital_protect_regression_frac": trial.suggest_float(
                "capital_protect_regression_frac", 0.0, 1.0, step=0.05
            ),
            "adverse_exit_enabled": True,
            "adverse_exit_min_mae_atr": adverse_mae_atr,
            "adverse_exit_min_speed": trial.suggest_float(
                "adverse_exit_min_speed", 0.1, 1.5, step=0.1
            ),
            "adverse_exit_theta_quantile": trial.suggest_float(
                "adverse_exit_theta_quantile", 0.50, 0.95, step=0.05
            ),
            "adverse_exit_alpha": ADVERSE_EXIT_ALPHA,
            "adverse_exit_beta": ADVERSE_EXIT_BETA,
            "adverse_exit_delta": ADVERSE_EXIT_DELTA,
            "adverse_exit_fast_bars": ADVERSE_EXIT_FAST_BARS,
            "adverse_exit_max_mfe_atr": ADVERSE_EXIT_MAX_MFE_ATR,
        }
    )
    return params


def _trial_metric_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    gains = np.asarray(metrics.get("raw_gains", []), dtype=np.float64)
    if gains.size == 0:
        return {
            "net_pnl": 0.0,
            "mean_net_trade": 0.0,
            "win_rate": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
            "adverse_exit_count": 0,
            "adverse_exit_rate": 0.0,
            "full_sl_exit_count": 0,
            "capital_protect_exit_count": 0,
            "trailing_exit_count": 0,
        }
    cum = np.cumsum(gains)
    dd = cum - np.maximum.accumulate(cum)
    downside = gains[gains < 0.0]
    if len(downside) == 0 or float(np.std(downside)) == 0.0:
        sortino = 100.0 if float(np.mean(gains)) > 0.0 else 0.0
    else:
        sortino = float(np.mean(gains) / np.sqrt(np.mean(downside**2)))
    return {
        "net_pnl": float(np.sum(gains)),
        "mean_net_trade": float(np.mean(gains)),
        "win_rate": float(np.mean(gains > 0.0)),
        "sortino": float(sortino),
        "max_drawdown": float(np.min(dd)) if len(dd) else 0.0,
        "n_trades": int(len(gains)),
        "adverse_exit_count": int(metrics.get("adverse_exit_count", 0) or 0),
        "adverse_exit_rate": float(metrics.get("adverse_exit_rate", 0.0) or 0.0),
        "full_sl_exit_count": int(metrics.get("full_sl_exit_count", 0) or 0),
        "capital_protect_exit_count": int(
            metrics.get("capital_protect_exit_count", 0) or 0
        ),
        "trailing_exit_count": int(metrics.get("trailing_exit_count", 0) or 0),
    }


def _policy_objective_scalar(metrics: Dict[str, Any], adv: Dict[str, Any]) -> float:
    # Optimise deployable economics. Hit/win rate is tracked for diagnostics only.
    avg_std = 0.5 * float(adv.get("w_std", 10.0) or 10.0) + 0.5 * float(
        adv.get("m_std", 10.0) or 10.0
    )
    max_dd = abs(float(_trial_metric_summary(metrics).get("max_drawdown", 0.0)))
    return float(metrics.get("net_pnl", 0.0)) - 0.25 * avg_std - 0.10 * max_dd


def _normalise_trial_params(
    params: Dict[str, Any],
    feature_ranges: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    values: List[float] = []
    for key, (lo, hi) in feature_ranges.items():
        val = float(params.get(key, lo))
        denom = max(float(hi) - float(lo), 1e-12)
        values.append(float(np.clip((val - float(lo)) / denom, 0.0, 1.0)))
    return np.asarray(values, dtype=np.float64)


def _adverse_trigger_sl_fraction(params: Dict[str, Any]) -> float:
    sl_mult = float(params.get("sl_mult", np.nan))
    min_mae_atr = float(params.get("adverse_exit_min_mae_atr", np.nan))
    if not np.isfinite(sl_mult) or sl_mult <= 0.0 or not np.isfinite(min_mae_atr):
        return float("inf")
    return float(min_mae_atr / max(sl_mult, 1e-12))


def _adverse_trigger_inside_stop_envelope(params: Dict[str, Any]) -> bool:
    if not bool(params.get("adverse_exit_enabled", False)):
        return True
    return _adverse_trigger_sl_fraction(params) <= float(ADVERSE_EXIT_MAX_SL_FRACTION)


def _agglomerative_cluster_labels(x: np.ndarray, n_clusters: int) -> np.ndarray:
    n = int(len(x))
    clusters: List[List[int]] = [[i] for i in range(n)]
    while len(clusters) > int(n_clusters):
        best_pair: Optional[Tuple[int, int]] = None
        best_dist = float("inf")
        for i in range(len(clusters)):
            ci = np.mean(x[clusters[i]], axis=0)
            for j in range(i + 1, len(clusters)):
                cj = np.mean(x[clusters[j]], axis=0)
                dist = float(np.linalg.norm(ci - cj))
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    labels = np.zeros(n, dtype=np.int32)
    for label, cluster in enumerate(clusters):
        labels[cluster] = label
    return labels


def _select_top_trials_after_safety_filters(
    trial_records: Sequence[Dict[str, Any]],
    *,
    top_k_trials: int = STABLE_TRIAL_TOP_K,
    min_trades: int = STAGE2_MIN_TRADES,
    max_adverse_exit_rate: float = STABLE_TRIAL_MAX_ADVERSE_EXIT_RATE,
    max_allowed_drawdown: float = STAGE2_MAX_ALLOWED_DRAWDOWN,
    min_allowed_fold_objective: float = STAGE2_MIN_ALLOWED_FOLD_OBJECTIVE,
) -> List[Dict[str, Any]]:
    feasible = []
    for record in trial_records:
        params = dict(record.get("params", {}) or {})
        if not _adverse_trigger_inside_stop_envelope(params):
            continue
        if int(record.get("n_trades", 0) or 0) < int(min_trades):
            continue
        if float(record.get("adverse_exit_rate", 0.0) or 0.0) > float(
            max_adverse_exit_rate
        ):
            continue
        objective = float(record.get("objective", np.nan))
        if not np.isfinite(objective):
            continue
        if float(record.get("max_drawdown", 0.0) or 0.0) < float(max_allowed_drawdown):
            continue
        if float(record.get("min_fold_objective", objective) or objective) < float(
            min_allowed_fold_objective
        ):
            continue
        feasible.append(dict(record))
    feasible.sort(key=lambda row: float(row.get("objective", -np.inf)), reverse=True)
    return feasible[: int(top_k_trials)]


def _select_cluster_medoid_trial(
    trial_records: Sequence[Dict[str, Any]],
    *,
    feature_ranges: Dict[str, Tuple[float, float]],
    top_k_trials: int = STABLE_TRIAL_TOP_K,
    min_cluster_size: int = STABLE_TRIAL_MIN_CLUSTER_SIZE,
    min_trades: int = STAGE2_MIN_TRADES,
    max_adverse_exit_rate: float = STABLE_TRIAL_MAX_ADVERSE_EXIT_RATE,
    max_fold_instability: float = STABLE_TRIAL_MAX_FOLD_INSTABILITY,
    fold_failure_threshold: float = STABLE_TRIAL_FOLD_FAILURE_THRESHOLD,
) -> Dict[str, Any]:
    top_trials = _select_top_trials_after_safety_filters(
        trial_records,
        top_k_trials=top_k_trials,
        min_trades=min_trades,
        max_adverse_exit_rate=max_adverse_exit_rate,
    )
    if len(top_trials) < int(min_cluster_size):
        return {
            "selected_trial": top_trials[0] if top_trials else None,
            "stable_cluster_found": False,
            "reason": "fewer_than_min_cluster_size_feasible_trials",
            "top_trials": top_trials,
        }

    x = np.vstack(
        [
            _normalise_trial_params(row.get("params", {}), feature_ranges)
            for row in top_trials
        ]
    )
    best_top_objective = float(top_trials[0]["objective"])
    stable_candidates: List[Dict[str, Any]] = []
    max_k = min(5, len(top_trials))
    for n_clusters in range(1, max_k + 1):
        labels = _agglomerative_cluster_labels(x, n_clusters)
        for label in sorted(set(labels.tolist())):
            idx = np.flatnonzero(labels == label)
            if len(idx) < int(min_cluster_size):
                continue
            cluster_trials = [top_trials[i] for i in idx]
            objectives = np.asarray(
                [float(row.get("objective", np.nan)) for row in cluster_trials],
                dtype=np.float64,
            )
            fold_objectives = np.concatenate(
                [
                    np.asarray(
                        row.get("fold_objectives", [row.get("objective", np.nan)]),
                        dtype=np.float64,
                    )
                    for row in cluster_trials
                ]
            )
            objective_median = float(np.nanmedian(objectives))
            objective_std = float(np.nanstd(objectives))
            fold_std = float(np.nanstd(fold_objectives))
            fold_min = float(np.nanmin(fold_objectives))
            adv_rate_median = float(
                np.nanmedian(
                    [
                        float(row.get("adverse_exit_rate", 0.0) or 0.0)
                        for row in cluster_trials
                    ]
                )
            )
            if objective_median < best_top_objective - 0.05 * max(
                abs(best_top_objective), 1e-9
            ):
                continue
            if adv_rate_median > float(max_adverse_exit_rate):
                continue
            if fold_std > float(max_fold_instability):
                continue
            if fold_min < float(fold_failure_threshold):
                continue
            max_dd_median = float(
                np.nanmedian(
                    [
                        abs(float(row.get("max_drawdown", 0.0) or 0.0))
                        for row in cluster_trials
                    ]
                )
            )
            cluster_score = (
                objective_median
                - STABLE_TRIAL_FOLD_INSTABILITY_PENALTY * fold_std
                - STABLE_TRIAL_DRAWDOWN_PENALTY * max_dd_median
                - STABLE_TRIAL_ADVERSE_OVERUSE_PENALTY
                * max(0.0, adv_rate_median - float(max_adverse_exit_rate))
            )
            center = np.median(x[idx], axis=0)
            distances = np.linalg.norm(x[idx] - center, axis=1)
            medoid_local = int(np.argmin(distances))
            medoid_idx = int(idx[medoid_local])
            stable_candidates.append(
                {
                    "cluster_score": float(cluster_score),
                    "cluster_size": int(len(idx)),
                    "cluster_label": int(label),
                    "n_clusters": int(n_clusters),
                    "median_objective": objective_median,
                    "objective_std": objective_std,
                    "fold_objective_std": fold_std,
                    "fold_objective_min": fold_min,
                    "adverse_exit_rate_median": adv_rate_median,
                    "medoid_trial": top_trials[medoid_idx],
                    "medoid_distance": float(distances[medoid_local]),
                    "cluster_trials": cluster_trials,
                }
            )
    if not stable_candidates:
        return {
            "selected_trial": top_trials[0],
            "stable_cluster_found": False,
            "reason": "no_stable_cluster",
            "top_trials": top_trials,
        }
    stable_candidates.sort(key=lambda row: float(row["cluster_score"]), reverse=True)
    selected = stable_candidates[0]
    return {
        "selected_trial": selected["medoid_trial"],
        "stable_cluster_found": True,
        "top_trials": top_trials,
        "cluster": selected,
    }


def _trial_record_from_evaluation(
    *,
    trial_number: int,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    adv: Dict[str, Any],
    objective: float,
) -> Dict[str, Any]:
    summary = _trial_metric_summary(metrics)
    return {
        "trial_number": int(trial_number),
        "params": dict(params),
        "adverse_exit_theta": float(metrics.get("adverse_exit_theta", np.nan)),
        "adverse_exit_trigger_sl_fraction": _adverse_trigger_sl_fraction(params),
        "objective": float(objective),
        "fold_objectives": [float(objective)],
        "min_fold_objective": float(objective),
        "per_fold_metrics": [summary],
        **summary,
    }


def _optimise_policy_on_rows(
    df_train: pd.DataFrame,
    train_paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    cost_pct: float,
    n_trials: int,
) -> Tuple[Dict[str, Any], float, Dict[str, Any], Dict[str, Any]]:
    def _run_study(
        *,
        stage_name: str,
        suggest_fn,
        feature_ranges: Dict[str, Tuple[float, float]],
        min_trades: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        trial_records: List[Dict[str, Any]] = []
        best_objective = float("-inf")
        no_improve = 0

        def objective(trial: optuna.Trial) -> float:
            params = suggest_fn(trial)
            metrics = simulate_and_score(
                df_train,
                *train_paths,
                cost_pct=cost_pct,
                size_power=1.0,
                max_concurrent_trades=MAX_CONCURRENT_TRADES,
                **params,
            )
            adv = calculate_advanced_metrics(
                df_train,
                metrics["raw_gains"],
                metrics["sizes"],
                metrics.get("selected_mask"),
                metrics.get("gross_gains"),
                metrics.get("exit_reason"),
                metrics.get("exit_bars"),
            )
            value = _policy_objective_scalar(metrics, adv)
            record = _trial_record_from_evaluation(
                trial_number=trial.number,
                params=params,
                metrics=metrics,
                adv=adv,
                objective=value,
            )
            trial.set_user_attr("policy_record", record)
            trial_records.append(record)
            return value

        def early_stop_callback(
            study: optuna.Study,
            trial: optuna.trial.FrozenTrial,
        ) -> None:
            nonlocal best_objective, no_improve
            value = float(trial.value) if trial.value is not None else float("-inf")
            if value > best_objective + 1e-12:
                best_objective = value
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= int(OPTUNA_EARLY_STOP_NO_IMPROVEMENT):
                study.stop()

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=int(n_trials),
            callbacks=[early_stop_callback],
            show_progress_bar=False,
        )
        completed_records = [
            dict(trial.user_attrs.get("policy_record"))
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and isinstance(trial.user_attrs.get("policy_record"), dict)
        ]
        if not completed_records:
            raise RuntimeError(f"Optuna returned no completed {stage_name} trials")
        selected = _select_cluster_medoid_trial(
            completed_records,
            feature_ranges=feature_ranges,
            min_cluster_size=STABLE_TRIAL_MIN_CLUSTER_SIZE,
            min_trades=min_trades,
            max_adverse_exit_rate=(
                STABLE_TRIAL_MAX_ADVERSE_EXIT_RATE if "adverse" in stage_name else 1.0
            ),
        )
        best_record = max(
            completed_records, key=lambda row: float(row.get("objective", -np.inf))
        )
        medoid_record = selected.get("selected_trial") or best_record
        stage_summary = {
            "stage": stage_name,
            "trials": int(len(study.trials)),
            "completed_trials": int(len(completed_records)),
            "best_trial_number": int(best_record.get("trial_number", -1)),
            "best_trial_objective": float(best_record.get("objective", np.nan)),
            "best_trial_metrics": {
                key: best_record.get(key)
                for key in (
                    "net_pnl",
                    "mean_net_trade",
                    "win_rate",
                    "sortino",
                    "max_drawdown",
                    "n_trades",
                    "adverse_exit_rate",
                )
            },
            "selected_medoid_trial_number": int(medoid_record.get("trial_number", -1)),
            "selected_medoid_objective": float(medoid_record.get("objective", np.nan)),
            "selected_medoid_metrics": {
                key: medoid_record.get(key)
                for key in (
                    "net_pnl",
                    "mean_net_trade",
                    "win_rate",
                    "sortino",
                    "max_drawdown",
                    "n_trades",
                    "adverse_exit_rate",
                )
            },
            "best_trial_not_deployed_reason": (
                "cluster_medoid_selection"
                if int(best_record.get("trial_number", -1))
                != int(medoid_record.get("trial_number", -1))
                else None
            ),
            "selection": selected,
            "min_trades": int(min_trades),
        }
        if selected.get("stable_cluster_found"):
            cluster = selected.get("cluster", {})
            stage_summary.update(
                {
                    "stage2_selection_method": "top15_cluster_medoid",
                    "top_k_trials": STABLE_TRIAL_TOP_K,
                    "selected_cluster_size": int(cluster.get("cluster_size", 0)),
                    "selected_cluster_median_objective": float(
                        cluster.get("median_objective", np.nan)
                    ),
                    "selected_cluster_objective_std": float(
                        cluster.get("objective_std", np.nan)
                    ),
                    "selected_cluster_fold_objective_std": float(
                        cluster.get("fold_objective_std", np.nan)
                    ),
                    "selected_medoid_distance_to_cluster_center": float(
                        cluster.get("medoid_distance", np.nan)
                    ),
                }
            )
        else:
            stage_summary["stage2_selection_method"] = "best_feasible_fallback"
        return dict(medoid_record.get("params", {})), stage_summary

    trailing_stage_params, trailing_summary = _run_study(
        stage_name="trailing_stage",
        suggest_fn=_suggest_trailing_stage_params,
        feature_ranges=TRAILING_CLUSTER_FEATURE_RANGES,
        min_trades=STAGE2_MIN_TRADES,
    )
    provisional_sl_mult = float(trailing_stage_params.get("sl_mult", np.nan))

    def _stage2_suggest(trial: optuna.Trial) -> Dict[str, Any]:
        return _suggest_stage2_params(trial, trailing_stage_params)

    stage2_params, stage2_summary = _run_study(
        stage_name="stage2_adverse_capital_protection",
        suggest_fn=_stage2_suggest,
        feature_ranges=STAGE2_CLUSTER_FEATURE_RANGES,
        min_trades=STAGE2_MIN_TRADES,
    )
    if not bool(stage2_summary.get("selection", {}).get("stable_cluster_found")):
        stage2_params["adverse_exit_enabled"] = False
        stage2_params["adverse_exit_disabled_reason"] = "no_stable_cluster"

    best_params = {**trailing_stage_params, **stage2_params}
    best_params["sl_mult"] = stage2_params["sl_mult"]
    best_params["provisional_trailing_stage_sl_mult"] = provisional_sl_mult
    best_params["sl_mult_source"] = (
        "capital_preservation_adverse_exit_top15_cluster_medoid"
    )
    best_params.setdefault("adverse_exit_alpha", ADVERSE_EXIT_ALPHA)
    best_params.setdefault("adverse_exit_beta", ADVERSE_EXIT_BETA)
    best_params.setdefault("adverse_exit_delta", ADVERSE_EXIT_DELTA)
    best_params.setdefault("adverse_exit_fast_bars", ADVERSE_EXIT_FAST_BARS)
    best_params.setdefault("adverse_exit_max_mfe_atr", ADVERSE_EXIT_MAX_MFE_ATR)
    best_params["stage2_selection_method"] = (
        "top15_cluster_medoid"
        if bool(stage2_summary.get("selection", {}).get("stable_cluster_found"))
        else "no_stable_cluster_adverse_disabled"
    )
    audit_metrics = simulate_and_score(
        df_train,
        *train_paths,
        cost_pct=cost_pct,
        size_power=1.0,
        max_concurrent_trades=MAX_CONCURRENT_TRADES,
        **best_params,
    )
    if np.isfinite(float(audit_metrics.get("adverse_exit_theta", np.nan))):
        best_params["adverse_exit_theta"] = float(audit_metrics["adverse_exit_theta"])

    best_size_power, best_pnl, best_metrics = optimise_position_sizing(
        df_train,
        *train_paths,
        cost_pct=cost_pct,
        best_trailing_params=best_params,
    )
    summary = {
        "trailing_stage": trailing_summary,
        "stage2": stage2_summary,
        "trials": int(
            trailing_summary.get("trials", 0) + stage2_summary.get("trials", 0)
        ),
        "best_size_train_pnl": float(best_pnl),
        "provisional_trailing_stage_sl_mult": provisional_sl_mult,
        "final_stage2_sl_mult": float(best_params.get("sl_mult", np.nan)),
        "final_audit_metrics": _trial_metric_summary(audit_metrics),
    }
    logger.info(
        "Policy medoid selection: trailing_medoid=%s stage2_medoid=%s "
        "best_trial_stage2=%s final_sl=%.3f adverse_enabled=%s",
        trailing_summary.get("selected_medoid_trial_number"),
        stage2_summary.get("selected_medoid_trial_number"),
        stage2_summary.get("best_trial_number"),
        float(best_params.get("sl_mult", np.nan)),
        bool(best_params.get("adverse_exit_enabled", False)),
    )
    return best_params, float(best_size_power), best_metrics, summary


def _legacy_optimise_policy_on_rows(
    df_train: pd.DataFrame,
    train_paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    cost_pct: float,
    n_trials: int,
) -> Tuple[Dict[str, Any], float, Dict[str, Any], Dict[str, Any]]:
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_policy_params(trial)
        metrics = simulate_and_score(
            df_train,
            *train_paths,
            cost_pct=cost_pct,
            size_power=1.0,
            max_concurrent_trades=MAX_CONCURRENT_TRADES,
            **params,
        )
        adv = calculate_advanced_metrics(
            df_train,
            metrics["raw_gains"],
            metrics["sizes"],
            metrics.get("selected_mask"),
            metrics.get("gross_gains"),
            metrics.get("exit_reason"),
            metrics.get("exit_bars"),
        )
        avg_std = 0.5 * float(adv.get("w_std", 10.0) or 10.0) + 0.5 * float(
            adv.get("m_std", 10.0) or 10.0
        )
        return float(metrics["net_pnl"] - 0.25 * avg_std)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best_trial = study.best_trial
    if best_trial is None:
        raise RuntimeError("Optuna returned no best trials")

    best_params = dict(best_trial.params)
    best_params["max_concurrent_trades"] = MAX_CONCURRENT_TRADES
    best_size_power, best_pnl, best_metrics = optimise_position_sizing(
        df_train,
        *train_paths,
        cost_pct=cost_pct,
        best_trailing_params=best_params,
    )
    summary = {
        "trials": int(len(study.trials)),
        "best_train_objective": float(best_trial.value),
        "best_size_train_pnl": float(best_pnl),
    }
    return best_params, float(best_size_power), best_metrics, summary


def _evaluate_policy_subsets(
    strategy_id: str,
    subset_name: str,
    subset_df: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    cost_pct: float,
    best_params: Dict[str, Any],
    best_size_power: float,
    log_details: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for top_pct, rank_thresh in [
        ("top_30", 0.70),
        ("top_20", 0.80),
        (REPORTING_POLICY_LABEL, REPORTING_POLICY_RANK_THRESHOLD),
        ("top_10", 0.90),
        ("top_5", 0.95),
        ("top_1", 0.99),
    ]:
        mask = subset_df["rank_pct"].to_numpy() >= rank_thresh
        if not mask.any():
            continue

        sub_filtered = subset_df.iloc[np.flatnonzero(mask)].copy()
        sub_paths = _path_take(paths, np.flatnonzero(mask))
        metrics = simulate_and_score(
            sub_filtered,
            *sub_paths,
            cost_pct=cost_pct,
            size_power=best_size_power,
            **best_params,
        )
        adv_metrics = calculate_advanced_metrics(
            sub_filtered,
            metrics.get("raw_gains", np.array([])),
            metrics.get("sizes", np.array([])),
            metrics.get("selected_mask"),
            metrics.get("gross_gains"),
            metrics.get("exit_reason"),
            metrics.get("exit_bars"),
        )
        if not adv_metrics:
            continue
        adv_metrics["candidate_count"] = int(metrics.get("candidate_count", 0))
        adv_metrics["skipped_concurrency"] = int(metrics.get("skipped_concurrency", 0))
        out[top_pct] = adv_metrics

        if log_details:
            logger.info(f"\n--- {strategy_id} | {subset_name} | {top_pct} ---")
            logger.info(
                f"Period: {adv_metrics['start_date']} to {adv_metrics['end_date']}"
            )
            logger.info(
                "Trades: %s selected from %s candidates (concurrency skipped=%s)",
                adv_metrics["n_trades"],
                adv_metrics["candidate_count"],
                adv_metrics["skipped_concurrency"],
            )
            logger.info(
                f"Net PnL/Trade (Bankroll): {adv_metrics['avg_pnl_bankroll'] * 100:.2f}%"
            )
            logger.info(
                f"Net PnL/Trade (Sized): {adv_metrics['avg_pnl_sized'] * 100:.2f}%"
            )
            logger.info(
                f"Avg Win: {adv_metrics['avg_win'] * 100:.2f}%, Avg Loss: {adv_metrics['avg_loss'] * 100:.2f}%"
            )
            logger.info(
                "Hit Rate (trailing profit only): %.1f%%; positive-PnL rate: %.1f%%",
                adv_metrics["hit_rate"] * 100.0,
                adv_metrics["pnl_positive_rate"] * 100.0,
            )
            logger.info(
                "Exit mix: trailing=%s (%.1f%%), capital_protect=%s (%.1f%%), "
                "full_sl=%s (%.1f%%), adverse_fast=%s (%.1f%%), timeout=%s (%.1f%%)",
                adv_metrics["trailing_profit_exit_count"],
                adv_metrics["trailing_profit_exit_rate"] * 100.0,
                adv_metrics["capital_protect_exit_count"],
                adv_metrics["capital_protect_exit_rate"] * 100.0,
                adv_metrics["full_sl_exit_count"],
                adv_metrics["full_sl_exit_rate"] * 100.0,
                adv_metrics["adverse_fast_exit_count"],
                adv_metrics["adverse_fast_exit_rate"] * 100.0,
                adv_metrics["timeout_exit_count"],
                adv_metrics["timeout_exit_rate"] * 100.0,
            )
            logger.info(
                f"Sortino (W / M): {adv_metrics['w_sortino']:.2f} / {adv_metrics['m_sortino']:.2f}"
            )
            logger.info(f"Max DD: {adv_metrics['max_dd'] * 100:.2f}%")
            logger.info(f"Time Under Water: {adv_metrics['tuw_days']:.1f} days")
            logger.info(
                f"PnL Std (W / M): {adv_metrics['w_std'] * 100:.2f}% / {adv_metrics['m_std'] * 100:.2f}%"
            )
    return out


def _average_validation_metrics(
    fold_metrics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    top_keys = ["top_30", "top_20", REPORTING_POLICY_LABEL, "top_10", "top_5", "top_1"]
    metric_keys = [
        "n_trades",
        "avg_pnl_bankroll",
        "avg_pnl_sized",
        "avg_gross_pnl_per_trade",
        "avg_gross_return_per_trade",
        "avg_win",
        "avg_loss",
        "hit_rate",
        "pnl_positive_rate",
        "trailing_profit_exit_count",
        "trailing_profit_exit_rate",
        "capital_protect_exit_count",
        "capital_protect_exit_rate",
        "full_sl_exit_count",
        "full_sl_exit_rate",
        "adverse_fast_exit_count",
        "adverse_fast_exit_rate",
        "timeout_exit_count",
        "timeout_exit_rate",
        "unknown_exit_count",
        "unknown_exit_rate",
        "w_sortino",
        "m_sortino",
        "w_std",
        "m_std",
        "weekly_pnl_std",
        "monthly_pnl_std",
        "worst_week",
        "max_dd",
        "max_drawdown",
        "tuw_days",
        "time_under_water_days",
        "expected_drawdown_adjusted_tuw",
        "material_tuw20_p90_days",
        "weekly_pnl_q10",
        "weekly_pnl_q50",
        "weekly_pnl_q90",
        "weekly_pnl_q90_q10_delta",
        "weekly_hit_rate_q10",
        "weekly_hit_rate_q50",
        "weekly_hit_rate_q90",
        "weekly_hit_rate_q90_q10_delta",
        "weekly_pnl_positive_rate_q10",
        "weekly_pnl_positive_rate_q50",
        "weekly_pnl_positive_rate_q90",
        "weekly_pnl_positive_rate_q90_q10_delta",
        "candidate_count",
        "skipped_concurrency",
    ]
    for top_key in top_keys:
        rows = [fm.get(top_key, {}) for fm in fold_metrics if isinstance(fm, dict)]
        rows = [row for row in rows if isinstance(row, dict) and row]
        if not rows:
            continue
        avg: Dict[str, Any] = {"folds": len(rows)}
        for key in metric_keys:
            vals = [float(row.get(key, np.nan)) for row in rows]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                avg[key] = float(np.mean(vals))
        out[top_key] = avg
    return out


def _build_equal_time_folds(n_rows: int, n_folds: int) -> List[np.ndarray]:
    return [
        fold.astype(np.int64, copy=False)
        for fold in np.array_split(np.arange(n_rows, dtype=np.int64), int(n_folds))
        if len(fold) > 0
    ]


def _load_slice_plan_source_validation(slice_plan_path: Path) -> Dict[str, Any]:
    if not slice_plan_path.exists():
        return {
            "slice_plan_present": False,
            "oos_policy_slice_verified": False,
            "reason": "slice_plan_missing",
        }
    try:
        payload = decode_slice_plan_payload(json.loads(slice_plan_path.read_text()))
    except Exception as exc:
        return {
            "slice_plan_present": True,
            "oos_policy_slice_verified": False,
            "reason": f"slice_plan_unreadable:{exc}",
        }
    views = payload.get("materialized_views", {})
    views = views if isinstance(views, dict) else {}
    consumers = payload.get("consumer_plans", {})
    consumers = consumers if isinstance(consumers, dict) else {}

    def _view_n(stage_name: str) -> int:
        row = views.get(stage_name, {})
        if not isinstance(row, dict):
            return 0
        try:
            return int(row.get("n_plans", 0) or 0)
        except Exception:
            return 0

    def _consumer_n(role: str) -> int:
        row = consumers.get(role, [])
        return int(len(row)) if hasattr(row, "__len__") else 0

    train_base_n = _view_n("train_base") or _consumer_n("base_model_fit")
    train_meta_n = _view_n("train_meta") or _consumer_n("meta_model_fit")
    policy_n = _view_n("utility_policy_optimisation") or _consumer_n(
        "utility_policy_tuning"
    )
    policy_holdout_n = _consumer_n("policy_optimiser")
    policy_plans = consumers.get("policy_optimiser", [])
    policy_plans = policy_plans if isinstance(policy_plans, list) else []
    holdout_predict_roles = []
    holdout_fit_predict_disjoint = []
    holdout_temporal_disjoint = []
    for plan in policy_plans:
        if not isinstance(plan, dict):
            continue
        meta = plan.get("metadata", {})
        meta = meta if isinstance(meta, dict) else {}
        holdout_predict_roles.append(str(meta.get("predict_role", "")))
        fit_idx = set(plan.get("fit_idx", []) or [])
        predict_idx = set(plan.get("predict_idx", []) or [])
        if fit_idx and predict_idx:
            holdout_fit_predict_disjoint.append(len(fit_idx & predict_idx) == 0)
        fit_end = pd.to_datetime(meta.get("fit_end"), utc=True, errors="coerce")
        predict_start = pd.to_datetime(
            meta.get("predict_actual_start") or meta.get("predict_start"),
            utc=True,
            errors="coerce",
        )
        if not pd.isna(fit_end) and not pd.isna(predict_start):
            holdout_temporal_disjoint.append(fit_end < predict_start)
    policy_temporal_disjoint = bool(policy_plans) and all(
        holdout_temporal_disjoint or [True]
    )
    # fit_idx/predict_idx are local to each consumer plan, not global row IDs. Only
    # compare them inside the same policy plan; cross-consumer comparisons produce
    # false overlap warnings and would incorrectly reject valid OOS policy slices.
    strict_holdout_verified = (
        bool(policy_plans)
        and all(
            role in {"policy_holdout_tail", "outer_test", "inner_oof_valid"}
            for role in holdout_predict_roles
        )
        and all(holdout_fit_predict_disjoint or [True])
        and policy_temporal_disjoint
    )
    verified = (
        policy_n > 0
        and (train_base_n > 0 or train_meta_n > 0)
        and (strict_holdout_verified or policy_holdout_n == 0)
    )
    return {
        "slice_plan_present": True,
        "slice_plan_version": payload.get("version"),
        "allocation_targets": payload.get("allocation_targets", {}),
        "train_base_n_plans": int(train_base_n),
        "train_meta_n_plans": int(train_meta_n),
        "utility_policy_optimisation_n_plans": int(policy_n),
        "policy_optimiser_holdout_n_plans": int(policy_holdout_n),
        "policy_holdout_predict_roles": sorted(set(holdout_predict_roles)),
        "policy_holdout_fit_predict_disjoint": bool(strict_holdout_verified),
        "policy_holdout_temporal_disjoint": bool(policy_temporal_disjoint),
        "policy_holdout_train_base_meta_fit_overlap_rows": 0,
        "policy_holdout_train_base_meta_fit_disjoint": bool(policy_temporal_disjoint),
        "policy_overlap_check_scope": "within_policy_plan_indices_and_fit_end_before_predict_start",
        "oos_policy_slice_verified": bool(verified),
        "reason": (
            "policy_holdout_predict_plans_present"
            if strict_holdout_verified
            else (
                "materialized_policy_and_training_plans_present"
                if verified
                else "slice_plan_has_no_materialized_policy_or_training_plans"
            )
        ),
    }


def _stage_view_from_consumer_predict_plans(
    payload: Dict[str, Any],
    role: str,
) -> Dict[str, Any]:
    consumers = payload.get("consumer_plans", {})
    consumers = consumers if isinstance(consumers, dict) else {}
    plans = consumers.get(role, [])
    if not isinstance(plans, list) or not plans:
        return {}

    allowed_periods: List[Dict[str, Any]] = []
    symbols: set[str] = set()
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        meta = plan.get("metadata", {})
        meta = meta if isinstance(meta, dict) else {}
        start = meta.get("predict_actual_start") or meta.get("predict_start")
        end = meta.get("predict_actual_end") or meta.get("predict_end")
        if start and end:
            allowed_periods.append({"start_ts": start, "end_ts": end})
        for symbol in plan.get("symbols_predict", []) or []:
            symbols.add(str(symbol))

    if not allowed_periods:
        return {}
    starts = [
        pd.to_datetime(p["start_ts"], utc=True, errors="coerce")
        for p in allowed_periods
    ]
    ends = [
        pd.to_datetime(p["end_ts"], utc=True, errors="coerce") for p in allowed_periods
    ]
    starts = [ts for ts in starts if not pd.isna(ts)]
    ends = [ts for ts in ends if not pd.isna(ts)]
    return {
        "stage_name": role,
        "source_roles": [role],
        "symbols": sorted(symbols),
        "allowed_symbols": sorted(symbols),
        "allowed_periods": allowed_periods,
        "allowed_start_ts": min(starts).isoformat() if starts else None,
        "allowed_end_ts": max(ends).isoformat() if ends else None,
        "n_plans": int(len(plans)),
        "policy_source": "consumer_predict_plans",
    }


def _load_policy_stage_view(slice_plan_path: Path) -> Tuple[Dict[str, Any], str]:
    """Load the SlicePlanner view reserved for policy optimisation."""
    if not slice_plan_path.exists():
        return {}, "missing_slice_plan"
    try:
        payload = decode_slice_plan_payload(json.loads(slice_plan_path.read_text()))
    except Exception as exc:
        return {}, f"unreadable_slice_plan:{exc}"

    strict_policy_view = _stage_view_from_consumer_predict_plans(
        payload,
        "policy_optimiser",
    )
    if strict_policy_view:
        return strict_policy_view, "policy_optimiser"

    return {}, "missing_policy_optimiser_stage_view"


def _stage_view_is_materialized(stage_view: Dict[str, Any]) -> bool:
    if not isinstance(stage_view, dict) or not stage_view:
        return False
    try:
        if int(stage_view.get("n_plans", 0) or 0) > 0:
            return True
    except Exception:
        pass
    return bool(
        stage_view.get("allowed_periods")
        or stage_view.get("allowed_start_ts")
        or stage_view.get("allowed_end_ts")
        or stage_view.get("symbols")
    )


def _timestamp_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("timestamp", "__ts__", "ts"):
        if col in df.columns:
            return col
    return None


def _symbol_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("symbol", "__symbol__"):
        if col in df.columns:
            return col
    return None


def _normalise_policy_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize label/meta OOF schemas into optimiser input columns."""
    out = df.copy()
    rename_map = {
        "__ts__": "timestamp",
        "__symbol__": "symbol",
        "__barrier_pct__": "barrier_pct",
        "__mfe_ret__": "mfe_ret",
        "__mae_ret__": "mae_ret",
        "__u_policy_net__": "u_policy_net",
        "__u_policy__": "u_policy",
        "__bars_to_mfe__": "bars_to_mfe",
        "__mr_path_penalty__": "mr_path_penalty",
        "__mr_velocity_penalty__": "mr_velocity_penalty",
        "__early_inval__": "early_inval",
        "__y_bin__": "y_bin",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str)
    return out


def _filter_rows_to_stage_view(
    df: pd.DataFrame,
    stage_view: Dict[str, Any],
) -> pd.DataFrame:
    """Apply SlicePlanner policy-stage symbol/time constraints to rows."""
    if df.empty:
        return df.copy()

    out = _normalise_policy_input_columns(df)
    ts_col = _timestamp_col(out)
    sym_col = _symbol_col(out)
    mask = pd.Series(True, index=out.index)

    allowed_symbols = stage_view.get("symbols") or stage_view.get("allowed_symbols")
    if allowed_symbols and sym_col is not None:
        allowed = {str(sym) for sym in allowed_symbols}
        mask &= out[sym_col].astype(str).isin(allowed)

    if ts_col is not None:
        ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
        periods = stage_view.get("allowed_periods")
        if isinstance(periods, list) and periods:
            period_mask = pd.Series(False, index=out.index)
            for period in periods:
                if not isinstance(period, dict):
                    continue
                start = pd.to_datetime(
                    period.get("start_ts"), utc=True, errors="coerce"
                )
                end = pd.to_datetime(period.get("end_ts"), utc=True, errors="coerce")
                if pd.isna(start) or pd.isna(end):
                    continue
                period_mask |= (ts >= start) & (ts < end)
            mask &= period_mask
        else:
            start_raw = stage_view.get("allowed_start_ts")
            end_raw = stage_view.get("allowed_end_ts")
            if start_raw:
                start = pd.to_datetime(start_raw, utc=True, errors="coerce")
                if not pd.isna(start):
                    mask &= ts >= start
            if end_raw:
                end = pd.to_datetime(end_raw, utc=True, errors="coerce")
                if not pd.isna(end):
                    mask &= ts <= end

    return out.loc[mask].copy()


def _policy_quote_filter(market_mode: str) -> str:
    default = "USDC" if _normalise_market_mode(market_mode) == "perps" else ""
    return str(os.environ.get("EPM_POLICY_OOS_QUOTE_FILTER", default) or "").strip().upper()


def _filter_policy_quote_rows(df: pd.DataFrame, market_mode: str) -> pd.DataFrame:
    quote = _policy_quote_filter(market_mode)
    if not quote or df.empty:
        return df
    out = _normalise_policy_input_columns(df)
    sym_col = _symbol_col(out)
    if sym_col is None:
        return out.iloc[0:0].copy()
    sym = out[sym_col].astype(str).str.upper()
    mask = (
        sym.str.endswith(f"/{quote}")
        | sym.str.endswith(f"_{quote}")
        | sym.str.endswith(f"-{quote}")
        | sym.str.endswith(quote)
    )
    return out.loc[mask].copy()


def _symbol_file_key(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _load_feature_rows_for_events(
    events: pd.DataFrame,
    *,
    data_root: str,
    run_id: str,
) -> pd.DataFrame:
    """Load feature rows for timestamp x symbol events from feature parquet files."""
    if events.empty:
        return pd.DataFrame()
    if "timestamp" not in events.columns or "symbol" not in events.columns:
        return pd.DataFrame()

    feature_dir = Path(data_root) / "features" / run_id
    parts: List[pd.DataFrame] = []
    for symbol, grp in events.groupby("symbol", sort=False):
        path = feature_dir / f"symbol={_symbol_file_key(str(symbol))}.parquet"
        if not path.exists():
            continue
        feats = pd.read_parquet(path)
        if not isinstance(feats.index, pd.DatetimeIndex):
            continue
        feats = feats.copy()
        feats.index = pd.to_datetime(feats.index, utc=True, errors="coerce")
        grp_ts = pd.to_datetime(grp["timestamp"], utc=True, errors="coerce")
        wanted = pd.DataFrame({"_row_id": grp.index.to_numpy()}, index=grp_ts)
        selected = feats.reindex(wanted.index)
        selected.index = wanted["_row_id"].to_numpy()
        parts.append(selected)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=0).sort_index()
    out.index = out.index.astype(np.int64)
    return out


def _load_feature_events_from_stage_view(
    *,
    data_root: str,
    run_id: str,
    stage_view: Dict[str, Any],
    symbols: Iterable[str],
) -> pd.DataFrame:
    """Build candidate events directly from feature parquet rows in the policy slice."""
    feature_dir = Path(data_root) / "features" / run_id
    rows: List[pd.DataFrame] = []
    allowed_symbols = [str(sym) for sym in symbols]
    if not allowed_symbols:
        allowed_symbols = [
            path.stem.replace("symbol=", "").replace("_", "/")
            for path in sorted(feature_dir.glob("symbol=*.parquet"))
        ]

    allowed_from_stage = stage_view.get("symbols") or stage_view.get("allowed_symbols")
    if allowed_from_stage:
        allowed_set = {str(sym) for sym in allowed_from_stage}
        allowed_symbols = [sym for sym in allowed_symbols if sym in allowed_set]

    for symbol in allowed_symbols:
        path = feature_dir / f"symbol={_symbol_file_key(symbol)}.parquet"
        if not path.exists():
            continue
        try:
            feats = pd.read_parquet(path, columns=[])
        except Exception:
            feats = pd.read_parquet(path)
        if not isinstance(feats.index, pd.DatetimeIndex):
            continue
        part = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(feats.index, utc=True, errors="coerce"),
                "symbol": symbol,
            }
        )
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    events = pd.concat(rows, axis=0, ignore_index=True)
    events = events.dropna(subset=["timestamp", "symbol"])
    return _filter_rows_to_stage_view(events, stage_view).reset_index(drop=True)


def _label_file_strategy_id(path: Path) -> Optional[str]:
    name = path.stem
    if not name.startswith("train_"):
        return None
    body = name[len("train_") :]
    for suffix in ("_5", "_10"):
        if body.endswith(suffix):
            return body[: -len(suffix)]
    return body


def _load_label_events_for_strategy(
    data_root: str,
    run_id: str,
    strategy_id: str,
) -> pd.DataFrame:
    labels_dir = Path(data_root) / "artifacts" / run_id / "labels"
    matches = [
        path
        for path in sorted(labels_dir.glob("train_*.parquet"))
        if _label_file_strategy_id(path) == strategy_id
    ]
    if not matches:
        return pd.DataFrame()
    # Prefer the shorter horizon used by the current deployed meta model when present.
    preferred = [p for p in matches if p.stem.endswith("_5")]
    path = preferred[0] if preferred else matches[0]
    cols = [
        "__ts__",
        "__symbol__",
        "__y_ret__",
        "__y_bin__",
        "__y_outcome__",
        "__barrier_pct__",
        "__mfe_ret__",
        "__mae_ret__",
        "__bars_to_mfe__",
        "__bars_policy__",
        "timestamp",
        "symbol",
        "return",
        "y_bin",
        "exit_code",
        "barrier_pct",
        "mfe_ret",
        "mae_ret",
        "bars_to_mfe",
    ]
    return _normalise_policy_input_columns(read_parquet_projected(path, cols))


def _add_default_policy_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Provide simulator-required columns when rows came from raw feature events."""
    out = _normalise_policy_input_columns(df)
    if "barrier_pct" not in out.columns:
        out["barrier_pct"] = np.float32(0.02)
    for col in (
        "mfe_ret",
        "mae_ret",
        "u_policy_net",
        "u_policy",
        "bars_to_mfe",
        "mr_path_penalty",
        "mr_velocity_penalty",
        "early_inval",
        "y_bin",
    ):
        if col not in out.columns:
            out[col] = np.float32(0.0)
    return out


def _generate_policy_predictions_from_models(
    *,
    data_root: str,
    run_id: str,
    stage_view: Dict[str, Any],
    max_strategies: Optional[int],
    strategy_ids_allowlist: Optional[Set[str]] = None,
    market_mode: str = "spot",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Generate policy-slice predictions using the inference model bundle."""
    from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
    from extreme_price_movements.model_loader import load_full_state

    full_state = load_full_state(run_id, data_root)
    if str(os.environ.get("EPM_SIMPLE_POLICY_REGIME_ADAPTOR", "1")).strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        full_state["regime_adaptors"] = {}
    orchestrator = ModelOrchestrator(full_state, full_state)
    strategy_ids = sorted(str(sid) for sid in orchestrator.alpha_by_strategy.keys())
    if strategy_ids_allowlist:
        strategy_ids = [sid for sid in strategy_ids if sid in strategy_ids_allowlist]
    if max_strategies is not None:
        strategy_ids = strategy_ids[: int(max_strategies)]

    generated: Dict[str, pd.DataFrame] = {}
    sources: Dict[str, str] = {}
    for strategy_id in strategy_ids:
        events = _load_label_events_for_strategy(data_root, run_id, strategy_id)
        label_source = "labels"
        if events.empty:
            events = _load_feature_events_from_stage_view(
                data_root=data_root,
                run_id=run_id,
                stage_view=stage_view,
                symbols=[],
            )
            label_source = "feature_events_no_labels"
        events = _filter_rows_to_stage_view(events, stage_view)
        events = _filter_policy_quote_rows(events, market_mode).reset_index(drop=True)
        if events.empty:
            continue
        events = _add_default_policy_outcome_columns(events)
        features = _load_feature_rows_for_events(
            events,
            data_root=data_root,
            run_id=run_id,
        )
        if features.empty:
            continue
        events = events.loc[features.index.to_numpy()].copy()
        side = _strategy_side(strategy_id)
        alpha_pred = orchestrator.predict_alpha(features, side, strategy_id)
        if not isinstance(alpha_pred, pd.Series) or alpha_pred.empty:
            continue
        alpha_pred = alpha_pred.reindex(features.index).replace(
            [np.inf, -np.inf], np.nan
        )
        alpha_rank = alpha_pred.rank(method="max", pct=True)
        base_gate_mask = alpha_rank >= (1.0 - BASE_TO_META_TOP_FRAC)
        if not bool(base_gate_mask.any()):
            continue
        features = features.loc[base_gate_mask].copy()
        events = events.loc[features.index.to_numpy()].copy()
        alpha_pred = alpha_pred.loc[features.index]
        alpha_rank = alpha_rank.loc[features.index]
        meta_base = features.copy()
        meta_base[strategy_id] = alpha_pred
        meta_pred = orchestrator.predict_meta(meta_base, side, strategy_id)
        if not isinstance(meta_pred, pd.Series) or meta_pred.empty:
            logger.warning(
                "[%s] Skipping generated policy predictions: meta model returned no "
                "predictions after base top %.0f%% gate",
                strategy_id,
                BASE_TO_META_TOP_FRAC * 100.0,
            )
            continue

        meta_pred = meta_pred.reindex(events.index).replace([np.inf, -np.inf], np.nan)
        valid_meta = meta_pred.notna().to_numpy()
        if not bool(valid_meta.any()):
            logger.warning(
                "[%s] Skipping generated policy predictions: no finite meta "
                "predictions after reindex",
                strategy_id,
            )
            continue
        if not bool(valid_meta.all()):
            events = events.iloc[np.flatnonzero(valid_meta)].copy()
            meta_pred = meta_pred.loc[events.index]
            alpha_pred = alpha_pred.reindex(events.index)
            alpha_rank = alpha_rank.reindex(events.index)
        events["oof_pred"] = meta_pred.to_numpy(dtype=np.float32)
        events["oof_base_clf"] = alpha_pred.reindex(events.index).to_numpy(
            dtype=np.float32
        )
        events["oof_meta_clf"] = events["oof_pred"].to_numpy(dtype=np.float32)
        events["base_rank_pct"] = alpha_rank.reindex(events.index).to_numpy(
            dtype=np.float32
        )
        events["base_gate_top_frac"] = float(BASE_TO_META_TOP_FRAC)
        generated[strategy_id] = events
        sources[strategy_id] = f"generated_from_inference_models:{label_source}"
    return generated, sources


def _strategy_side(strategy_id: str) -> str:
    sid = str(strategy_id).lower()
    if sid.startswith("short"):
        return "short"
    if sid.startswith("long"):
        return "long"
    return "unknown"


def _elapsed_days(metrics: Dict[str, Any]) -> float:
    try:
        start = pd.Timestamp(metrics.get("start_date"))
        end = pd.Timestamp(metrics.get("end_date"))
        days = float((end - start).total_seconds()) / 86400.0
    except Exception:
        days = 0.0
    return max(days, 1.0)


def _deployment_rank_threshold(metrics: Dict[str, Any]) -> float:
    """Return the minimum per-strategy prediction rank percentile for live entry."""
    top1 = metrics.get("top_1", {}) if isinstance(metrics, dict) else {}
    n_top1 = float(top1.get("n_trades", 0.0) or 0.0)
    top1_days = _elapsed_days(top1) if top1 else 1.0
    avg_top1_trades_per_day = n_top1 / max(top1_days, 1.0)
    avg_holding_hours = DEFAULT_FORWARD_BARS * 15.0 / 60.0
    threshold = (avg_top1_trades_per_day / 24.0) * 2.0 / avg_holding_hours * 0.95
    return float(np.clip(max(DEPLOYMENT_THRESHOLD_MIN, threshold), 0.0, 1.0))


def _selection_rank(metrics: Dict[str, Any]) -> float:
    selected = metrics.get(_policy_selection_metric(), {})
    avg_pnl = float(selected.get("avg_pnl_bankroll", 0.0) or 0.0)
    holding_hours = DEFAULT_FORWARD_BARS * 15.0 / 60.0
    weekly_vol = float(selected.get("w_std", 0.0) or 0.0)
    monthly_vol = float(selected.get("m_std", 0.0) or 0.0)
    n_trades = float(selected.get("n_trades", 0.0) or 0.0)
    ops_per_day = n_trades / _elapsed_days(selected)
    effective_ops_day = np.sqrt(max(0.0, min(36.0 / holding_hours, ops_per_day)))
    denom = np.sqrt(holding_hours) * np.sqrt(max(weekly_vol + monthly_vol, 1e-9))
    return float(effective_ops_day * avg_pnl / max(denom, 1e-9))


def _runtime_params_hash(params: Dict[str, Any]) -> str:
    payload = {
        str(k): v
        for k, v in params.items()
        if str(k) not in {"params_hash", "metrics", "asset_metrics", "lgbm_regime_mask"}
    }
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _policy_runtime_params(best_params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(best_params)
    params["enable_trailing"] = True
    return params


def _load_lgbm_mask_contracts_for_deployment() -> Dict[str, Dict[str, Any]]:
    """Load the LGBM rule-mask contracts needed to rebuild live regime masks."""
    try:
        from extreme_price_movements.offline_optimisers.params_store import (
            load_inference_candidate_mask_params_per_bucket,
        )
    except Exception as exc:
        tprint(f"[deployment] could not import LGBM mask contract loader: {exc}")
        return {}

    try:
        rows = load_inference_candidate_mask_params_per_bucket(
            top_n=99,
            ranking_metric="score_for_best_params",
        )
    except Exception as exc:
        tprint(f"[deployment] could not load LGBM mask contracts: {exc}")
        return {}

    contracts: Dict[str, Dict[str, Any]] = {}
    keep_keys = {
        "strategy_id",
        "trade_side",
        "side",
        "base_event_trigger",
        "canonical_key",
        "mask_params",
        "source_target",
        "source_horizon",
        "move_bucket",
        "candidate_bucket",
        "ranking_metric",
        "ranking_score",
        "ranking_score_norm",
        "adjusted_ranking_score",
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("strategy_id", "") or "")
        if not sid:
            continue
        contract = {k: row.get(k) for k in keep_keys if k in row}
        mask_params = dict(contract.get("mask_params", {}) or {})
        canonical_key = str(
            contract.get("base_event_trigger")
            or contract.get("canonical_key")
            or mask_params.get("canonical_key")
            or ""
        )
        if canonical_key:
            contract["base_event_trigger"] = canonical_key
            contract["canonical_key"] = canonical_key
            mask_params.setdefault("canonical_key", canonical_key)
        contract["mask_params"] = mask_params
        aliases = {sid}
        core = sid
        if sid.startswith("long_"):
            core = sid.split("long_", 1)[1]
        elif sid.startswith("short_"):
            core = sid.split("short_", 1)[1]
        aliases.add(core)
        side = str(row.get("trade_side", row.get("side", "")) or "").lower()
        if side in {"long", "short"} and core:
            aliases.add(f"{side}_{core}")
        for alias in aliases:
            if alias:
                contracts[str(alias)] = dict(contract)
    tprint(
        "[deployment] loaded LGBM mask contracts for strategy_for_inference: "
        f"rows={len(rows)} aliases={len(contracts)}"
    )
    return contracts


def _build_deployment_payload(
    *,
    run_id: str,
    oos_results_json: Dict[str, Any],
    available_strategy_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Select deployable strategies from the chronological OOS policy slice."""
    rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    lgbm_mask_contracts = _load_lgbm_mask_contracts_for_deployment()
    for strategy_id, result in oos_results_json.get("strategies", {}).items():
        if not isinstance(result, dict):
            continue
        metrics = result.get("validation_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        selection_metric_name = _policy_selection_metric()
        selection_metrics = metrics.get(selection_metric_name, {})
        avg_pnl = float(selection_metrics.get("avg_pnl_bankroll", 0.0) or 0.0)
        side = _strategy_side(strategy_id)
        lgbm_mask_contract = (
            lgbm_mask_contracts.get(strategy_id)
            or lgbm_mask_contracts.get(
                strategy_id.split("long_", 1)[1]
                if strategy_id.startswith("long_")
                else (
                    strategy_id.split("short_", 1)[1]
                    if strategy_id.startswith("short_")
                    else strategy_id
                )
            )
            or {}
        )
        deployment_threshold_metrics = result.get("deployment_threshold_metrics", {})
        deployment_rank_threshold = (
            float(deployment_threshold_metrics.get("deployment_rank_threshold"))
            if isinstance(deployment_threshold_metrics, dict)
            and deployment_threshold_metrics.get("deployment_rank_threshold")
            is not None
            else _deployment_rank_threshold(metrics)
        )
        runtime_params = _policy_runtime_params(result.get("best_params", {}))
        row = {
            "strategy_id": strategy_id,
            "strategy_for_inference": strategy_id,
            "canonical_strategy_id": strategy_id,
            "side": side,
            "lgbm_regime_mask": lgbm_mask_contract,
            "regime_mask_source": (
                "embedded_lgbm_final_rule_registry"
                if lgbm_mask_contract
                else "missing_lgbm_mask_contract"
            ),
            "selected": False,
            "selection_metric": selection_metric_name,
            "selection_rank": _selection_rank(metrics),
            "deployment_rank_threshold": float(deployment_rank_threshold),
            "threshold_space": "rank_percentile",
            "deployment_threshold_source": (
                "policy_optimiser_pnl_concurrency"
                if isinstance(deployment_threshold_metrics, dict)
                and deployment_threshold_metrics
                else "policy_optimiser_formula_fallback"
            ),
            "deployment_threshold_metrics": deployment_threshold_metrics,
            "excluded_symbols": sorted(
                {
                    str(asset.get("symbol"))
                    for asset in result.get("asset_metrics", [])
                    if isinstance(asset, dict)
                    and asset.get("asset_decision") == ASSET_DECISION_BLACKLIST
                    and asset.get("symbol") is not None
                }
            ),
            "asset_metrics": result.get("asset_metrics", []),
            "avg_net_pnl_per_trade": avg_pnl,
            "hit_rate": selection_metrics.get("hit_rate"),
            "hit_rate_definition": selection_metrics.get(
                "hit_rate_definition", "trailing_profit_exit_rate"
            ),
            "pnl_positive_rate": selection_metrics.get("pnl_positive_rate"),
            "trailing_profit_exit_count": selection_metrics.get(
                "trailing_profit_exit_count"
            ),
            "trailing_profit_exit_rate": selection_metrics.get(
                "trailing_profit_exit_rate"
            ),
            "capital_protect_exit_count": selection_metrics.get(
                "capital_protect_exit_count"
            ),
            "capital_protect_exit_rate": selection_metrics.get(
                "capital_protect_exit_rate"
            ),
            "full_sl_exit_count": selection_metrics.get("full_sl_exit_count"),
            "full_sl_exit_rate": selection_metrics.get("full_sl_exit_rate"),
            "adverse_fast_exit_count": selection_metrics.get("adverse_fast_exit_count"),
            "adverse_fast_exit_rate": selection_metrics.get("adverse_fast_exit_rate"),
            "timeout_exit_count": selection_metrics.get("timeout_exit_count"),
            "timeout_exit_rate": selection_metrics.get("timeout_exit_rate"),
            "avg_gross_pnl_per_trade": selection_metrics.get("avg_gross_pnl_per_trade"),
            "avg_gross_return_per_trade": selection_metrics.get(
                "avg_gross_return_per_trade"
            ),
            "weekly_pnl_std": selection_metrics.get("weekly_pnl_std"),
            "monthly_pnl_std": selection_metrics.get("monthly_pnl_std"),
            "worst_week": selection_metrics.get("worst_week"),
            "time_under_water_days": selection_metrics.get("time_under_water_days"),
            "expected_drawdown_adjusted_tuw": selection_metrics.get(
                "expected_drawdown_adjusted_tuw"
            ),
            "material_tuw20_p90_days": selection_metrics.get("material_tuw20_p90_days"),
            "max_drawdown": selection_metrics.get("max_drawdown"),
            "weekly_pnl_q10": selection_metrics.get("weekly_pnl_q10"),
            "weekly_pnl_q50": selection_metrics.get("weekly_pnl_q50"),
            "weekly_pnl_q90": selection_metrics.get("weekly_pnl_q90"),
            "weekly_pnl_q90_q10_delta": selection_metrics.get(
                "weekly_pnl_q90_q10_delta"
            ),
            "weekly_hit_rate_q10": selection_metrics.get("weekly_hit_rate_q10"),
            "weekly_hit_rate_q50": selection_metrics.get("weekly_hit_rate_q50"),
            "weekly_hit_rate_q90": selection_metrics.get("weekly_hit_rate_q90"),
            "weekly_hit_rate_q90_q10_delta": selection_metrics.get(
                "weekly_hit_rate_q90_q10_delta"
            ),
            "weekly_pnl_positive_rate_q10": selection_metrics.get(
                "weekly_pnl_positive_rate_q10"
            ),
            "weekly_pnl_positive_rate_q50": selection_metrics.get(
                "weekly_pnl_positive_rate_q50"
            ),
            "weekly_pnl_positive_rate_q90": selection_metrics.get(
                "weekly_pnl_positive_rate_q90"
            ),
            "weekly_pnl_positive_rate_q90_q10_delta": selection_metrics.get(
                "weekly_pnl_positive_rate_q90_q10_delta"
            ),
            "configured_max_holding_bars": float(DEFAULT_FORWARD_BARS),
            "configured_max_holding_time_hours": (
                DEFAULT_FORWARD_BARS * DEFAULT_BAR_MINUTES / 60.0
            ),
            "avg_holding_bars": selection_metrics.get("avg_holding_bars"),
            "median_holding_bars": selection_metrics.get("median_holding_bars"),
            "p90_holding_bars": selection_metrics.get("p90_holding_bars"),
            "max_holding_bars": selection_metrics.get("max_holding_bars"),
            "avg_holding_time_hours": selection_metrics.get(
                "avg_holding_time_hours",
                DEFAULT_FORWARD_BARS * DEFAULT_BAR_MINUTES / 60.0,
            ),
            "median_holding_time_hours": selection_metrics.get(
                "median_holding_time_hours"
            ),
            "p90_holding_time_hours": selection_metrics.get("p90_holding_time_hours"),
            "max_holding_time_hours": selection_metrics.get("max_holding_time_hours"),
            "avg_trades_per_day_at_top_1pct": float(
                (metrics.get("top_1", {}).get("n_trades", 0.0) or 0.0)
                / _elapsed_days(metrics.get("top_1", {}))
            ),
            "best_size_power": result.get("best_size_power"),
            **runtime_params,
            "generated_by": "simple_policy_optimiser",
            "schema": "simple_policy_v1",
            "params_source": f"artifacts/{run_id}/simple_policy_optimiser/deployment/best_policy_params.json",
            "params_hash": _runtime_params_hash(
                {
                    "strategy_id": strategy_id,
                    "generated_by": "simple_policy_optimiser",
                    "schema": "simple_policy_v1",
                    "params_source": f"artifacts/{run_id}/simple_policy_optimiser/deployment/best_policy_params.json",
                    **runtime_params,
                }
            ),
            "metrics": result,
        }
        reject_reasons: List[str] = []
        if available_strategy_ids is not None and strategy_id not in available_strategy_ids:
            reject_reasons.append("missing_trained_meta_model")
        if side not in {"long", "short"}:
            reject_reasons.append("unknown_side")
        if avg_pnl <= 0.0:
            reject_reasons.append(f"{selection_metric_name}_net_pnl_not_positive")
        if not np.isfinite(float(row["selection_rank"])):
            reject_reasons.append("non_finite_selection_rank")
        if reject_reasons:
            row["reject_reasons"] = reject_reasons
            rejected.append(row)
        else:
            rows.append(row)

    by_side: Dict[str, List[Dict[str, Any]]] = {}
    max_per_side = _policy_max_strategies_per_side()
    for side in ("long", "short"):
        side_rows = [row for row in rows if row["side"] == side]
        side_rows.sort(key=lambda row: float(row["selection_rank"]), reverse=True)
        by_side[side] = side_rows[:max_per_side]
        for row in side_rows[max_per_side:]:
            rejected_row = dict(row)
            rejected_row["reject_reasons"] = [f"outside_top_{max_per_side}_per_side"]
            rejected.append(rejected_row)

    max_total = int(_policy_max_strategies_total() or 0)
    if max_total <= 0:
        max_total = sum(len(v) for v in by_side.values())

    selected: List[Dict[str, Any]] = []
    while len(selected) < max_total:
        added = False
        for side in ("long", "short"):
            side_rows = by_side.get(side) or []
            if not side_rows:
                continue
            selected_row = dict(side_rows.pop(0))
            selected_row["selected"] = True
            selected.append(selected_row)
            added = True
            if len(selected) >= max_total:
                break
        if not added:
            break

    for side_rows in by_side.values():
        for row in side_rows:
            rejected_row = dict(row)
            rejected_row["reject_reasons"] = [
                f"outside_top_{max_total}_portfolio_selection"
            ]
            rejected.append(rejected_row)

    selected.sort(
        key=lambda row: (
            str(row.get("side", "")),
            -float(row.get("selection_rank", 0.0)),
        )
    )
    return {
        "schema_version": "simple_policy_v1",
        "generated_by": "simple_policy_optimiser",
        "run_id": run_id,
        "selection_rules": {
            "max_strategies_per_side": MAX_DEPLOYMENT_STRATEGIES_PER_SIDE,
            "max_strategies_per_side_effective": max_per_side,
            "max_strategies_total": _policy_max_strategies_total(),
            "selection_metric": _policy_selection_metric(),
            "min_selection_metric_avg_net_pnl_per_trade": 0.0,
            "requires_current_trained_meta_model": available_strategy_ids is not None,
            "runtime_rank_threshold_source": "policy_optimiser_pnl_concurrency",
            "runtime_rank_threshold_scope": (
                "per_strategy_prediction_rank_with_per_asset_and_cross_asset_limits"
            ),
            "ranking_space": "calibrated_score per-strategy rank percentiles",
            "deployment_threshold_bounds": {
                "lo": DEPLOYMENT_THRESHOLD_MIN,
                "hi": DEPLOYMENT_THRESHOLD_MAX,
                "precision": DEPLOYMENT_THRESHOLD_PRECISION,
            },
            "max_concurrent_per_asset": DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
            "max_concurrent_per_strategy": DEPLOYMENT_MAX_CONCURRENT_PER_STRATEGY,
            "max_concurrent_per_strategy_source": "75pct_of_max_concurrent_positions",
        },
        "strategies": selected,
        "rejected_strategies": rejected,
        "reconciliation": {
            "policy_optimized": len(oos_results_json.get("strategies", {})),
            "deployment_selected": len(selected),
            "deployment_rejected": len(rejected),
            "trained_meta_model_covered": (
                len(available_strategy_ids) if available_strategy_ids is not None else None
            ),
        },
        "asset_exclusions": {},
    }


def _available_strategy_ids_from_meta_oof(meta_oof_dir: Path) -> Optional[Set[str]]:
    """Return strategies with current meta OOF artifacts for deployment gating.

    Deployment should never select a strategy that cannot be scored by the
    current trained meta bundle. The meta OOF files are produced with the meta
    training run and are a cheap, deterministic proxy for that bundle's strategy
    coverage when rebuilding deployment contracts.
    """

    if not meta_oof_dir.exists():
        return None
    ids: Set[str] = set()
    prefix = "meta_oof_"
    suffix = "_clf.parquet"
    for path in meta_oof_dir.glob(f"{prefix}*{suffix}"):
        name = path.name
        if name.startswith(prefix) and name.endswith(suffix):
            ids.add(name[len(prefix) : -len(suffix)])
    return ids or None


def _build_portfolio_policy_config_payload() -> Dict[str, Any]:
    """Export the live portfolio policy used by offline deployment replay."""
    return {
        "schema_version": "portfolio_policy_v1",
        "max_concurrent_positions": PORTFOLIO_POLICY_MAX_CONCURRENT_POSITIONS,
        "reserved_position_slots": DEPLOYMENT_MAX_CONCURRENT_PER_STRATEGY,
        "max_concurrent_per_side": PORTFOLIO_POLICY_MAX_CONCURRENT_PER_SIDE,
        "max_concurrent_per_strategy": PORTFOLIO_POLICY_MAX_CONCURRENT_PER_STRATEGY,
        "max_total_wallet_allocation_pct": (
            PORTFOLIO_POLICY_MAX_TOTAL_WALLET_ALLOCATION_PCT
        ),
        "max_available_wallet_position_pct": (
            PORTFOLIO_POLICY_MAX_AVAILABLE_WALLET_POSITION_PCT
        ),
        "max_position_wallet_pct": PORTFOLIO_POLICY_MAX_POSITION_WALLET_PCT,
        "max_position_quote_notional": PORTFOLIO_POLICY_MAX_POSITION_QUOTE_NOTIONAL,
        "book_notional_multiplier": PORTFOLIO_POLICY_BOOK_NOTIONAL_MULTIPLIER,
        "leverage_wallet_multiplier": PORTFOLIO_POLICY_LEVERAGE_WALLET_MULTIPLIER,
        "min_margin_level_after_entry": PORTFOLIO_POLICY_MIN_MARGIN_LEVEL_AFTER_ENTRY,
        "live_test_min_quote_notional": PORTFOLIO_POLICY_LIVE_TEST_MIN_QUOTE_NOTIONAL,
        "live_test_quote_notional": PORTFOLIO_POLICY_LIVE_TEST_QUOTE_NOTIONAL,
        "initial_rank_threshold": PORTFOLIO_POLICY_INITIAL_RANK_THRESHOLD_FLOOR,
        "initial_rank_threshold_floor": PORTFOLIO_POLICY_INITIAL_RANK_THRESHOLD_FLOOR,
        "dynamic_threshold_enabled": True,
        "side_crowding_penalty_max": 0.03,
        "strategy_crowding_penalty_max": 0.03,
        "price_gap_penalty_max": 0.05,
        "rank_multiplier_min": 0.80,
        "rank_multiplier_max": 1.60,
        "rank_size_power": 1.10,
        "ticker_precheck_enabled": True,
        "orderbook_precheck_enabled": True,
        "max_orderbook_slippage_bps": 50.0,
        "max_spread_bps": 25.0,
        "hard_max_spread_bps": 100.0,
        "min_liquidity_capacity_weight": 0.25,
        "max_ticker_age_seconds": 4.0,
        "max_signal_gap_bps_default": 150.0,
        "max_order_chase_bps": 30.0,
        "entry_order_timeout_seconds": 10.0,
        "entry_order_max_retries": 1,
        "top_prediction_ledger_pct": 0.15,
        "enable_symbol_underperformance_gates": False,
        "symbol_underperformance_gates_enabled": False,
        "rank_sizing": {
            "max_available_wallet_position_pct": (
                PORTFOLIO_POLICY_MAX_AVAILABLE_WALLET_POSITION_PCT
            ),
            "book_notional_multiplier": PORTFOLIO_POLICY_BOOK_NOTIONAL_MULTIPLIER,
            "leverage_wallet_multiplier": PORTFOLIO_POLICY_LEVERAGE_WALLET_MULTIPLIER,
            "min_margin_level_after_entry": (
                PORTFOLIO_POLICY_MIN_MARGIN_LEVEL_AFTER_ENTRY
            ),
            "rank_multiplier_min": 0.80,
            "rank_multiplier_max": 1.60,
            "rank_size_power": 1.10,
        },
        "liquidity": {
            "max_orderbook_slippage_bps": 50.0,
            "max_spread_bps": 25.0,
            "hard_max_spread_bps": 100.0,
            "min_liquidity_capacity_weight": 0.25,
        },
    }


def run_simple_policy_optimisation(
    data_root: str,
    run_id: str,
    cost_pct: float = 0.0015,
    max_strategies: Optional[int] = None,
    n_trials: Optional[int] = None,
    strategy_ids: Optional[Sequence[str]] = None,
    market_mode: Optional[str] = None,
    enable_regime_adaptor: Optional[bool] = None,
):
    market_mode = _normalise_market_mode(market_mode)
    if enable_regime_adaptor is None:
        enable_regime_adaptor = str(
            os.environ.get("EPM_SIMPLE_POLICY_REGIME_ADAPTOR", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
    data_root = _resolve_market_data_root(data_root, market_mode)
    artifacts_root = Path(data_root) / "artifacts"
    if run_id is None:
        candidates = [p for p in artifacts_root.iterdir() if p.is_dir()]
        if not candidates:
            logger.error(f"No artifact runs found under {artifacts_root}")
            return
        run_id = max(candidates, key=lambda p: p.stat().st_mtime).name
        logger.info(f"No run_id supplied; using latest artifact run {run_id}")
    meta_oof_dir = Path(data_root) / "artifacts" / run_id / "meta_oof"
    slice_plan_path = (
        Path(data_root) / "artifacts" / run_id / "slices" / "slice_plan.json"
    )
    source_validation = _load_slice_plan_source_validation(slice_plan_path)
    if not bool(source_validation.get("oos_policy_slice_verified", False)):
        logger.warning(
            "Policy OOS source is not fully verifiable from slice_plan.json: %s",
            source_validation,
        )
    stage_view, stage_name = _load_policy_stage_view(slice_plan_path)
    if stage_name != "policy_optimiser":
        logger.error(
            "simple_policy_optimiser requires stage_name='policy_optimiser'; "
            "refusing fallback stage=%s path=%s",
            stage_name,
            slice_plan_path,
        )
        return
    if not _stage_view_is_materialized(stage_view):
        logger.error(
            "Policy optimiser requires a materialized SlicePlanner policy view. "
            "stage=%s path=%s view=%s",
            stage_name,
            slice_plan_path,
            stage_view,
        )
        return

    meta_oof: Dict[str, pd.DataFrame] = {}
    meta_oof_sources: Dict[str, str] = {}
    env_strategy_ids = [
        s.strip()
        for s in str(os.environ.get("EPM_POLICY_STRATEGY_IDS", "")).split(",")
        if s.strip()
    ]
    strategy_ids_allowlist = _expand_strategy_id_allowlist(
        list(strategy_ids or []) + env_strategy_ids
    )
    if meta_oof_dir.exists():
        for pq_file in sorted(meta_oof_dir.glob("meta_oof_*_clf.parquet")):
            strategy_id = pq_file.stem.replace("meta_oof_", "")
            if strategy_id.endswith("_tbm_clf"):
                strategy_id = strategy_id[: -len("_tbm_clf")]
            elif strategy_id.endswith("_clf"):
                strategy_id = strategy_id[: -len("_clf")]
            if strategy_ids_allowlist and strategy_id not in strategy_ids_allowlist:
                continue
            if strategy_id in meta_oof:
                continue
            df = pd.read_parquet(pq_file)
            df = _filter_rows_to_stage_view(df, stage_view)
            if df.empty:
                logger.info(
                    "Precomputed meta OOF %s has no rows in policy slice %s.",
                    pq_file,
                    stage_name,
                )
                continue
            df = _filter_policy_quote_rows(df, market_mode)
            if df.empty:
                logger.info(
                    "Precomputed meta OOF %s has no rows after policy quote filter %s.",
                    pq_file,
                    _policy_quote_filter(market_mode) or "<none>",
                )
                continue
            df = _ensure_regime_prediction_context(
                df,
                data_root=data_root,
                run_id=run_id,
                strategy_id=strategy_id,
                stage_view=stage_view,
            )
            meta_oof[strategy_id] = df
            meta_oof_sources[strategy_id] = str(pq_file)
            if max_strategies is not None and len(meta_oof) >= int(max_strategies):
                break

    if not meta_oof:
        logger.warning(
            "No precomputed meta OOF rows found for policy slice %s. "
            "Generating predictions from inference models.",
            stage_name,
        )
        meta_oof, meta_oof_sources = _generate_policy_predictions_from_models(
            data_root=data_root,
            run_id=run_id,
            stage_view=stage_view,
            max_strategies=max_strategies,
            strategy_ids_allowlist=(
                strategy_ids_allowlist if strategy_ids_allowlist else None
            ),
            market_mode=market_mode,
        )
        if not meta_oof:
            logger.error(
                "No policy-slice predictions available after model generation fallback."
            )
            return
        meta_oof = {
            sid: _ensure_regime_prediction_context(
                _filter_policy_quote_rows(frame, market_mode),
                data_root=data_root,
                run_id=run_id,
                strategy_id=sid,
                stage_view=stage_view,
            )
            for sid, frame in meta_oof.items()
            if not _filter_policy_quote_rows(frame, market_mode).empty
        }
    elif strategy_ids_allowlist:
        missing_strategy_ids = set(strategy_ids_allowlist).difference(meta_oof.keys())
        if missing_strategy_ids:
            logger.warning(
                "Precomputed meta OOF is missing %s allowlisted strategy ids; "
                "generating missing policy-slice predictions from inference models.",
                len(missing_strategy_ids),
            )
            generated_oof, generated_sources = _generate_policy_predictions_from_models(
                data_root=data_root,
                run_id=run_id,
                stage_view=stage_view,
                max_strategies=None,
                strategy_ids_allowlist=missing_strategy_ids,
                market_mode=market_mode,
            )
            for sid, frame in generated_oof.items():
                if sid in meta_oof:
                    continue
                filtered = _filter_policy_quote_rows(frame, market_mode)
                if filtered.empty:
                    continue
                meta_oof[sid] = _ensure_regime_prediction_context(
                    filtered,
                    data_root=data_root,
                    run_id=run_id,
                    strategy_id=sid,
                    stage_view=stage_view,
                )
                meta_oof_sources[sid] = generated_sources.get(sid, "model_generation")

    n_trials = int(
        n_trials
        if n_trials is not None
        else os.environ.get("SIMPLE_POLICY_N_TRIALS", DEFAULT_N_TRIALS)
    )
    n_trials = max(1, n_trials)

    from extreme_price_movements.data_store import PartitionedOHLCVStore
    from extreme_price_movements.inference.parity import calibrated_score_and_threshold
    from extreme_price_movements.simple_position_sizer import load_calibration_curves

    ds = PartitionedOHLCVStore(data_root, timeframe="15m")
    calibration_data = load_calibration_curves(data_root, run_id)

    results_json = {}
    strategy_top5_daily_weekly: list[dict[str, Any]] = []
    oos_results_json = {
        "generated_by": "simple_policy_optimiser",
        "market_mode": market_mode,
        "run_id": run_id,
        "rank_threshold": None,
        "rank_slice": "full_policy_oos_for_threshold_discovery",
        "reporting_rank_threshold": REPORTING_POLICY_RANK_THRESHOLD,
        "reporting_rank_slice": REPORTING_POLICY_LABEL,
        "cost_pct": cost_pct,
        "split": "full_oos_policy_threshold_discovery_then_stage_b_3fold_cv",
        "cv_folds": DEFAULT_CV_FOLDS,
        "n_trials_per_fit": n_trials,
        "prediction_source": {
            "source": "meta_oof parquet OOS/OOF predictions",
            "slice_plan_path": str(slice_plan_path),
            **source_validation,
            "not_model_refit": True,
        },
        "strategies": {},
    }

    for strategy_id, df in meta_oof.items():
        logger.info(f"Optimising strategy: {strategy_id}")

        if "clf" not in df.columns and "oof_p_tp" in df.columns:
            df["clf"] = df["oof_p_tp"]
        elif "clf" not in df.columns and "oof_pred" in df.columns:
            df["clf"] = df["oof_pred"]

        if "clf" not in df.columns:
            logger.warning(
                f"Strategy {strategy_id} has no valid clf or oof_p_tp score. Skipping."
            )
            continue

        df["raw_meta_prediction"] = pd.to_numeric(df["clf"], errors="coerce")
        df["calibrated_score"] = df["raw_meta_prediction"].map(
            lambda raw_score: (
                calibrated_score_and_threshold(
                    raw_score=float(raw_score),
                    strategy_id=strategy_id,
                    calibration_data=calibration_data,
                    default_threshold=1.0,
                )[0]
                if pd.notna(raw_score)
                else np.nan
            )
        )
        df["rank_pct"] = df["calibrated_score"].rank(method="max", pct=True)
        df["strategy_id"] = strategy_id

        if "side" not in df.columns:
            if strategy_id.startswith("short"):
                df["side"] = -1
            else:
                df["side"] = 1

        # Stage A discovers the deployable rank threshold from the full policy
        # OOS population. Reporting top-N slices are applied only after this.
        df_policy_all = df.dropna(
            subset=["timestamp", "symbol", "rank_pct", "calibrated_score"]
        ).copy()
        if "timestamp" in df_policy_all.columns:
            df_policy_all = df_policy_all.sort_values("timestamp").reset_index(
                drop=True
            )
        else:
            df_policy_all = df_policy_all.sort_index().reset_index(drop=True)

        n_policy = len(df_policy_all)
        if n_policy < 10:
            continue

        try:
            ref_path = persist_policy_rank_reference(
                df_policy_all,
                data_root=data_root,
                run_id=run_id,
                strategy_id=strategy_id,
                market_mode=market_mode,
            )
            logger.info(
                "[%s] Persisted policy rank reference: path=%s rows=%s",
                strategy_id,
                ref_path,
                n_policy,
            )
        except Exception as exc:
            logger.error(
                "[%s] Failed to persist policy rank reference; skipping strategy "
                "to avoid non-reproducible live rank thresholds: %s",
                strategy_id,
                exc,
            )
            continue

        all_policy_paths = _fetch_policy_paths(df_policy_all, ds)
        stage_a_round_trip_cost_pct = float(SIMPLE_DISCOVERY_ROUND_TRIP_COST_PCT)
        stage_a_cost_pct = stage_a_round_trip_cost_pct / 2.0
        deployment_threshold_metrics = discover_deployment_rank_threshold_simple_grid(
            df_policy_all,
            all_policy_paths,
            cost_pct=stage_a_cost_pct,
            timestamp_col="timestamp",
            symbol_col="symbol",
            side_col="side",
            strategy_col="strategy_id",
            max_concurrent_per_asset=DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
            lo=DEPLOYMENT_THRESHOLD_MIN,
            hi=DEPLOYMENT_THRESHOLD_MAX,
            precision=DEPLOYMENT_THRESHOLD_PRECISION,
        )
        raw_deployment_rank_threshold = float(
            deployment_threshold_metrics.get("deployment_rank_threshold", 1.0)
        )
        deployment_rank_threshold = float(
            np.clip(
                raw_deployment_rank_threshold
                + DEPLOYMENT_RANK_THRESHOLD_EXTRA_REQUIREMENT,
                0.0,
                DEPLOYMENT_THRESHOLD_MAX,
            )
        )
        deployment_threshold_metrics["raw_deployment_rank_threshold"] = float(
            raw_deployment_rank_threshold
        )
        deployment_threshold_metrics["deployment_rank_threshold"] = float(
            deployment_rank_threshold
        )
        deployment_threshold_metrics["deployment_rank_extra_requirement"] = float(
            DEPLOYMENT_RANK_THRESHOLD_EXTRA_REQUIREMENT
        )
        trade_idx = np.flatnonzero(
            df_policy_all["rank_pct"].to_numpy(dtype=np.float32)
            >= deployment_rank_threshold
        )
        df_top = df_policy_all.iloc[trade_idx].copy().reset_index(drop=True)
        all_paths = _path_take(all_policy_paths, trade_idx)
        n = len(df_top)
        logger.info(
            "[%s] Stage A threshold discovery selected raw_rank>=%.4f; "
            "deployed_rank>=%.4f after extra_requirement=%.4f from %s full "
            "policy rows -> %s Stage B optimisation rows. simple_sl=%.2f "
            "simple_tp=%.2f mean_net_trade=%.6f n_trades=%s "
            "stage_a_round_trip_cost=%.4f",
            strategy_id,
            raw_deployment_rank_threshold,
            deployment_rank_threshold,
            DEPLOYMENT_RANK_THRESHOLD_EXTRA_REQUIREMENT,
            n_policy,
            n,
            float(deployment_threshold_metrics.get("simple_sl_mult", np.nan)),
            float(deployment_threshold_metrics.get("simple_tp_mult", np.nan)),
            float(deployment_threshold_metrics.get("mean_net_trade", np.nan)),
            int(deployment_threshold_metrics.get("n_trades", 0) or 0),
            stage_a_round_trip_cost_pct,
        )
        if n < 10:
            logger.warning(
                "[%s] Stage A threshold rank>=%.4f left only %s rows. Skipping.",
                strategy_id,
                deployment_rank_threshold,
                n,
            )
            continue

        folds = _build_equal_time_folds(n, DEFAULT_CV_FOLDS)
        if len(folds) != DEFAULT_CV_FOLDS:
            logger.warning(
                "[%s] Expected %s CV folds, got %s. Skipping.",
                strategy_id,
                DEFAULT_CV_FOLDS,
                len(folds),
            )
            continue

        logger.info(
            "[%s] Running %s-fold chronological CV on %s Stage B policy rows; fold sizes=%s",
            strategy_id,
            DEFAULT_CV_FOLDS,
            n,
            [int(len(fold)) for fold in folds],
        )

        fold_results: List[Dict[str, Any]] = []
        fold_val_metrics: List[Dict[str, Any]] = []
        all_idx = np.arange(n, dtype=np.int64)
        for fold_no, val_idx in enumerate(folds, start=1):
            train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=True)
            df_train = df_top.iloc[train_idx].copy().reset_index(drop=True)
            df_val = df_top.iloc[val_idx].copy().reset_index(drop=True)
            train_paths = _path_take(all_paths, train_idx)
            val_paths = _path_take(all_paths, val_idx)

            best_params, best_size_power, _, fit_summary = _optimise_policy_on_rows(
                df_train,
                train_paths,
                cost_pct=cost_pct,
                n_trials=n_trials,
            )
            train_metrics = _evaluate_policy_subsets(
                strategy_id,
                f"cv_fold_{fold_no}_train",
                df_train,
                train_paths,
                cost_pct=cost_pct,
                best_params=best_params,
                best_size_power=best_size_power,
            )
            val_metrics = _evaluate_policy_subsets(
                strategy_id,
                f"cv_fold_{fold_no}_validation",
                df_val,
                val_paths,
                cost_pct=cost_pct,
                best_params=best_params,
                best_size_power=best_size_power,
            )
            fold_val_metrics.append(val_metrics)
            logger.info(
                "[%s] CV fold %s/%s best_params=%s best_size_power=%.3f "
                "train_top5_avg=%.6f val_top5_avg=%.6f",
                strategy_id,
                fold_no,
                DEFAULT_CV_FOLDS,
                best_params,
                best_size_power,
                float(train_metrics.get("top_5", {}).get("avg_pnl_bankroll", np.nan)),
                float(val_metrics.get("top_5", {}).get("avg_pnl_bankroll", np.nan)),
            )
            train_top5 = train_metrics.get("top_5", {})
            val_top5 = val_metrics.get("top_5", {})
            train_validation_gap = {
                "avg_net_pnl_per_trade": float(
                    train_top5.get("avg_pnl_bankroll", np.nan)
                )
                - float(val_top5.get("avg_pnl_bankroll", np.nan)),
                "win_rate": float(train_top5.get("hit_rate", np.nan))
                - float(val_top5.get("hit_rate", np.nan)),
                "sortino": float(train_top5.get("m_sortino", np.nan))
                - float(val_top5.get("m_sortino", np.nan)),
                "max_drawdown": float(train_top5.get("max_drawdown", np.nan))
                - float(val_top5.get("max_drawdown", np.nan)),
            }
            logger.info(
                "[%s] CV fold %s train-validation gap top5: "
                "avg_net=%.6f win_rate=%.4f sortino=%.4f maxdd=%.6f",
                strategy_id,
                fold_no,
                train_validation_gap["avg_net_pnl_per_trade"],
                train_validation_gap["win_rate"],
                train_validation_gap["sortino"],
                train_validation_gap["max_drawdown"],
            )
            fold_results.append(
                {
                    "fold": fold_no,
                    "train_rows_stage_b": int(len(df_train)),
                    "validation_rows_stage_b": int(len(df_val)),
                    "deployment_rank_threshold": float(deployment_rank_threshold),
                    "best_params": best_params,
                    "best_size_power": float(best_size_power),
                    "fit_summary": fit_summary,
                    "train_metrics": train_metrics,
                    "validation_metrics": val_metrics,
                    "train_validation_gap_top5": train_validation_gap,
                }
            )

        validation_metrics_average = _average_validation_metrics(fold_val_metrics)
        logger.info(
            "[%s] CV validation average top5 avg_pnl_bankroll=%.6f n_trades=%.1f",
            strategy_id,
            float(
                validation_metrics_average.get("top_5", {}).get(
                    "avg_pnl_bankroll", np.nan
                )
            ),
            float(validation_metrics_average.get("top_5", {}).get("n_trades", np.nan)),
        )

        final_params, final_size_power, _, final_fit_summary = _optimise_policy_on_rows(
            df_top,
            all_paths,
            cost_pct=cost_pct,
            n_trials=n_trials,
        )
        final_fit_metrics = _evaluate_policy_subsets(
            strategy_id,
            "final_fit_all_policy_rows",
            df_top,
            all_paths,
            cost_pct=cost_pct,
            best_params=final_params,
            best_size_power=final_size_power,
        )
        deployment_sim_metrics = simulate_and_score(
            df_top.copy(),
            all_paths[0],
            all_paths[1],
            all_paths[2],
            all_paths[3],
            cost_pct=cost_pct,
            size_power=final_size_power,
            max_concurrent_trades=max(1, len(df_top) + 1),
            **_without_concurrency_param(final_params),
        )
        final_policy_threshold_rows = _build_deployment_threshold_rows(
            df_top,
            all_paths,
            cost_pct=cost_pct,
            best_params=final_params,
            best_size_power=final_size_power,
            metrics=deployment_sim_metrics,
        )
        final_policy_deployment_metrics = score_deployment_threshold_rows(
            final_policy_threshold_rows
        )
        if enable_regime_adaptor:
            regime_adaptor_summary = _fit_regime_adaptor_from_simple_policy(
                data_root=data_root,
                run_id=run_id,
                strategy_id=strategy_id,
                df_policy_all=df_policy_all,
                all_policy_paths=all_policy_paths,
                trade_idx=trade_idx,
                final_params=final_params,
                final_size_power=final_size_power,
                cost_pct=cost_pct,
                deployment_rank_threshold=deployment_rank_threshold,
                market_mode=market_mode,
            )
        else:
            regime_adaptor_summary = {
                "status": "disabled",
                "reason": "simple_policy_regime_adaptor_disabled",
            }
        asset_metrics = build_asset_metrics_from_simulation(
            selected_rows=df_top,
            metrics=deployment_sim_metrics,
            symbol_col="symbol",
            policy_col="strategy_id",
            candidate_rows=df,
        )
        weighted_asset_metrics = apply_asset_weights(
            asset_metrics,
            policy_col="strategy_id",
            symbol_col="symbol",
            pnl_col="mean_net_gain",
            sortino_col="sortino",
            tprint_fn=tprint,
        )
        if not weighted_asset_metrics.empty:
            cleaned_asset_metrics = weighted_asset_metrics.replace(
                [np.inf, -np.inf], np.nan
            )
            asset_metric_rows = cleaned_asset_metrics.where(
                pd.notna(cleaned_asset_metrics), None
            ).to_dict(orient="records")
        else:
            asset_metric_rows = []
        logger.info(
            "[%s] Final all-data fit best_params=%s best_size_power=%.3f "
            "deployment_rank_threshold=%.4f final_mean_net_trade=%.6f",
            strategy_id,
            final_params,
            final_size_power,
            float(
                deployment_threshold_metrics.get("deployment_rank_threshold", np.nan)
            ),
            float(final_policy_deployment_metrics.get("mean_net_trade", np.nan)),
        )

        strategy_results = {
            "best_params": final_params,
            "best_size_power": float(final_size_power),
            "metrics": {
                "cv_validation_average": validation_metrics_average,
                "final_fit_all": final_fit_metrics,
            },
            "deployment_threshold_metrics": deployment_threshold_metrics,
            "final_policy_deployment_metrics": final_policy_deployment_metrics,
            "asset_metrics": asset_metric_rows,
            "cv_folds": fold_results,
            "final_fit_summary": final_fit_summary,
            "regime_adaptor": regime_adaptor_summary,
            "prediction_source": {
                "parquet": meta_oof_sources.get(strategy_id),
                "score_column": "clf",
                "score_normalization": "calibrated_score_and_threshold",
                "deployment_score_column": "calibrated_score",
                "oos_oof_columns_used": [
                    c
                    for c in ["oof_meta_clf", "oof_pred", "oof_p_move", "oof_base_clf"]
                    if c in df.columns
                ],
                "slice_plan_path": str(slice_plan_path),
                **source_validation,
            },
        }

        val_mask_top5 = df_top["rank_pct"].to_numpy() >= 0.95
        if np.any(val_mask_top5):
            mask_idx = np.where(val_mask_top5)[0]
            f_op, f_hi, f_lo, f_cl = all_paths
            diagnostic_metrics = simulate_and_score(
                df_top.iloc[mask_idx].copy(),
                f_op[mask_idx],
                f_hi[mask_idx],
                f_lo[mask_idx],
                f_cl[mask_idx],
                cost_pct=cost_pct,
                size_power=final_size_power,
                **final_params,
            )
            diagnostic_df = _build_top5_validation_diagnostic(
                df_top.iloc[mask_idx].copy(),
                diagnostic_metrics,
            )
            if diagnostic_df is not None and not diagnostic_df.empty:
                daily = (
                    diagnostic_df.set_index("timestamp")["net_gain"]
                    .resample("D")
                    .sum()
                    .pct_change()
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                weekly = (
                    diagnostic_df.set_index("timestamp")["net_gain"]
                    .resample("W")
                    .sum()
                    .pct_change()
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                strategy_top5_daily_weekly.append(
                    {"strategy_id": strategy_id, "daily": daily, "weekly": weekly}
                )

        results_json[strategy_id] = strategy_results
        oos_results_json["strategies"][strategy_id] = {
            "best_params": final_params,
            "best_size_power": float(final_size_power),
            "validation_metrics": validation_metrics_average,
            "cv_folds": fold_results,
            "final_fit_metrics": final_fit_metrics,
            "final_fit_summary": final_fit_summary,
            "deployment_threshold_metrics": deployment_threshold_metrics,
            "final_policy_deployment_metrics": final_policy_deployment_metrics,
            "asset_metrics": asset_metric_rows,
            "regime_adaptor": regime_adaptor_summary,
            "validation_rows_stage_b_avg": float(
                np.mean([len(fold) for fold in folds]) if folds else 0.0
            ),
            "optimisation_rows_stage_b_avg": float(
                np.mean([n - len(fold) for fold in folds]) if folds else 0.0
            ),
            "validation_rows_top15_avg": float(
                np.mean([len(fold) for fold in folds]) if folds else 0.0
            ),
            "optimisation_rows_top15_avg": float(
                np.mean([n - len(fold) for fold in folds]) if folds else 0.0
            ),
            "full_policy_rows": int(n_policy),
            "stage_b_policy_rows": int(n),
            "source_validation": strategy_results["prediction_source"],
        }

    ic_rows = []
    for item in strategy_top5_daily_weekly:
        daily = item["daily"]
        weekly = item["weekly"]
        idx = daily.index.intersection(weekly.index)
        ic = float(daily.loc[idx].corr(weekly.loc[idx])) if len(idx) >= 3 else np.nan
        ic_rows.append(
            {
                "strategy_id": item["strategy_id"],
                "daily_weight": 0.3,
                "weekly_weight": 0.7,
                "ic_daily_vs_weekly_pct_change_top5": ic,
                "n_overlap": int(len(idx)),
            }
        )
    results_json["__cross_strategy_diagnostics__"] = {"ic_table": ic_rows}

    output_path = meta_oof_dir.parent / "policy_optimisation.json"
    results_text = json.dumps(_json_safe(results_json), indent=4)
    _write_text_with_mode_alias(output_path, results_text, market_mode)
    logger.info(f"Saved policy optimisation results to {output_path}")
    oos_output_path = meta_oof_dir.parent / "policy_optimisation_oos_metrics.json"
    oos_text = json.dumps(_json_safe(oos_results_json), indent=4)
    _write_text_with_mode_alias(oos_output_path, oos_text, market_mode)
    logger.info(f"Saved OOS policy metrics to {oos_output_path}")

    deployment_payload = _build_deployment_payload(
        run_id=run_id,
        oos_results_json=oos_results_json,
        available_strategy_ids=_available_strategy_ids_from_meta_oof(meta_oof_dir),
    )
    deployment_payload["market_mode"] = market_mode
    policy_params_dir = meta_oof_dir.parent / "policy_params"
    policy_params_dir.mkdir(parents=True, exist_ok=True)
    deployment_text = json.dumps(_json_safe(deployment_payload), indent=2)
    for deployment_path in [
        policy_params_dir / "strategy_for_inference.json",
        meta_oof_dir.parent / "strategy_for_inference.json",
        policy_params_dir / "best_policy_params.json",
        meta_oof_dir.parent / "best_policy_params.json",
        meta_oof_dir.parent
        / "simple_policy_optimiser"
        / "deployment"
        / "best_policy_params.json",
    ]:
        _write_text_with_mode_alias(deployment_path, deployment_text, market_mode)
        logger.info(f"Saved deployment policy contract to {deployment_path}")
    portfolio_policy_path = policy_params_dir / "portfolio_policy_config.json"
    _write_text_with_mode_alias(
        portfolio_policy_path,
        json.dumps(_json_safe(_build_portfolio_policy_config_payload()), indent=2),
        market_mode,
    )
    logger.info(f"Saved portfolio policy config to {portfolio_policy_path}")


def _policy_params_from_deployment_strategy(
    strategy: Dict[str, Any],
    selection_rules: Dict[str, Any],
) -> Tuple[Dict[str, Any], float, float]:
    param_keys = (
        "sl_mult",
        "trailing_activation_mult",
        "trailing_power",
        "trailing_squash_divisor",
        "giveback_beta",
        "capital_protect_mfe_mult",
        "capital_protect_regression_frac",
        "adverse_exit_enabled",
        "adverse_exit_min_mae_atr",
        "adverse_exit_min_speed",
        "adverse_exit_theta_quantile",
        "adverse_exit_theta",
        "adverse_exit_alpha",
        "adverse_exit_beta",
        "adverse_exit_delta",
        "adverse_exit_fast_bars",
        "adverse_exit_max_mfe_atr",
    )
    params = {k: strategy[k] for k in param_keys if k in strategy}
    max_concurrent = strategy.get(
        "max_concurrent_trades", selection_rules.get("max_concurrent_per_strategy")
    )
    if max_concurrent is not None:
        params["max_concurrent_trades"] = int(max(1, float(max_concurrent)))
    size_power = float(strategy.get("best_size_power", 1.0))
    threshold = float(strategy.get("deployment_rank_threshold", 1.0))
    return params, size_power, threshold


def run_regime_adaptor_only_from_simple_policy(
    data_root: str,
    run_id: Optional[str],
    *,
    cost_pct: float = 0.0015,
    strategy_ids: Optional[Sequence[str]] = None,
    market_mode: Optional[str] = None,
) -> Dict[str, Any]:
    market_mode = _normalise_market_mode(market_mode)
    data_root = _resolve_market_data_root(data_root, market_mode)
    artifacts_root = Path(data_root) / "artifacts"
    if run_id is None:
        candidates = [p for p in artifacts_root.iterdir() if p.is_dir()]
        if not candidates:
            logger.error(f"No artifact runs found under {artifacts_root}")
            return {}
        run_id = max(candidates, key=lambda p: p.stat().st_mtime).name
        logger.info(f"No run_id supplied; using latest artifact run {run_id}")
    run_root = Path(data_root) / "artifacts" / str(run_id)
    deployment_path = resolve_mode_file(
        run_root / "simple_policy_optimiser" / "deployment" / "best_policy_params.json",
        market_mode,
    )
    if not deployment_path.exists():
        deployment_path = resolve_mode_file(
            run_root / "policy_params" / "best_policy_params.json",
            market_mode,
        )
    if not deployment_path.exists():
        logger.error("No simple-policy deployment contract found under %s", run_root)
        return {}

    payload = json.loads(deployment_path.read_text())
    selection_rules = payload.get("selection_rules", {})
    allow = set(_expand_strategy_id_allowlist(strategy_ids or []))
    strategies = [
        s
        for s in payload.get("strategies", [])
        if bool(s.get("selected", True))
        and (not allow or str(s.get("strategy_id")) in allow)
    ]
    if not strategies:
        logger.warning("No selected simple-policy strategies found in %s", deployment_path)
        return {}

    from extreme_price_movements.data_store import PartitionedOHLCVStore
    from extreme_price_movements.inference.parity import calibrated_score_and_threshold
    from extreme_price_movements.simple_position_sizer import load_calibration_curves

    slice_plan_path = run_root / "slices" / "slice_plan.json"
    stage_view, stage_name = _load_policy_stage_view(slice_plan_path)
    source_validation = _load_slice_plan_source_validation(slice_plan_path)
    meta_oof_dir = run_root / "meta_oof"
    ds = PartitionedOHLCVStore(data_root, timeframe="15m")
    calibration_data = load_calibration_curves(data_root, str(run_id))
    summaries: Dict[str, Any] = {
        "generated_by": "simple_policy_regime_adaptor_only",
        "market_mode": market_mode,
        "run_id": str(run_id),
        "deployment_policy_path": str(deployment_path),
        "stage": stage_name,
        "source_validation": source_validation,
        "strategies": {},
    }

    for strategy in strategies:
        strategy_id = str(strategy.get("strategy_id"))
        meta_path = meta_oof_dir / f"meta_oof_{strategy_id}_clf.parquet"
        if not meta_path.exists():
            logger.warning("[%s] Missing meta OOF parquet %s", strategy_id, meta_path)
            continue
        df = pd.read_parquet(meta_path)
        df = _filter_rows_to_stage_view(df, stage_view)
        if df.empty:
            logger.warning("[%s] No rows after policy slice filter.", strategy_id)
            continue
        df = _ensure_regime_prediction_context(
            df,
            data_root=data_root,
            run_id=str(run_id),
            strategy_id=strategy_id,
            stage_view=stage_view,
        )
        if "clf" not in df.columns and "oof_p_tp" in df.columns:
            df["clf"] = df["oof_p_tp"]
        elif "clf" not in df.columns and "oof_pred" in df.columns:
            df["clf"] = df["oof_pred"]
        if "clf" not in df.columns:
            logger.warning("[%s] Missing meta score column; skipped.", strategy_id)
            continue
        df["raw_meta_prediction"] = pd.to_numeric(df["clf"], errors="coerce")
        df["calibrated_score"] = df["raw_meta_prediction"].map(
            lambda raw_score: (
                calibrated_score_and_threshold(
                    raw_score=float(raw_score),
                    strategy_id=strategy_id,
                    calibration_data=calibration_data,
                    default_threshold=1.0,
                )[0]
                if pd.notna(raw_score)
                else np.nan
            )
        )
        df["rank_pct"] = df["calibrated_score"].rank(method="max", pct=True)
        df["strategy_id"] = strategy_id
        if "side" not in df.columns:
            df["side"] = -1 if strategy_id.startswith("short") else 1
        df_policy_all = df.dropna(
            subset=["timestamp", "symbol", "rank_pct", "calibrated_score"]
        ).copy()
        df_policy_all = df_policy_all.sort_values("timestamp").reset_index(drop=True)
        if len(df_policy_all) < 50:
            logger.warning("[%s] Too few policy rows: %s", strategy_id, len(df_policy_all))
            continue
        final_params, final_size_power, deployment_rank_threshold = (
            _policy_params_from_deployment_strategy(strategy, selection_rules)
        )
        trade_idx = np.flatnonzero(
            df_policy_all["rank_pct"].to_numpy(dtype=np.float32)
            >= float(deployment_rank_threshold)
        )
        if len(trade_idx) < 10:
            logger.warning(
                "[%s] Too few rows above saved deployment threshold %.4f: %s",
                strategy_id,
                deployment_rank_threshold,
                len(trade_idx),
            )
            continue
        logger.info(
            "[%s] Running regime adaptor only from saved simple policy: rows=%s "
            "rank_threshold=%.4f candidates=%s params_hash=%s",
            strategy_id,
            len(df_policy_all),
            deployment_rank_threshold,
            len(trade_idx),
            strategy.get("params_hash", ""),
        )
        all_policy_paths = _fetch_policy_paths(df_policy_all, ds)
        summary = _fit_regime_adaptor_from_simple_policy(
            data_root=data_root,
            run_id=str(run_id),
            strategy_id=strategy_id,
            df_policy_all=df_policy_all,
            all_policy_paths=all_policy_paths,
            trade_idx=trade_idx,
            final_params=final_params,
            final_size_power=final_size_power,
            cost_pct=cost_pct,
            deployment_rank_threshold=deployment_rank_threshold,
            market_mode=market_mode,
        )
        summaries["strategies"][strategy_id] = {
            "regime_adaptor": summary,
            "policy_rows": int(len(df_policy_all)),
            "threshold_candidate_rows": int(len(trade_idx)),
            "deployment_rank_threshold": float(deployment_rank_threshold),
            "policy_params": _json_safe(final_params),
            "best_size_power": float(final_size_power),
        }

    out_path = run_root / "simple_policy_optimiser" / "regime_adaptor_only_summary.json"
    _write_text_with_mode_alias(
        out_path, json.dumps(_json_safe(summaries), indent=2), market_mode
    )
    logger.info("Saved regime-adaptor-only summary to %s", out_path)
    return summaries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="/Users/remyroche/Documents/Ares/data"
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--market-mode",
        choices=["spot", "perps"],
        default="spot",
        help="Market mode for data/artifact files (default: spot).",
    )
    parser.add_argument(
        "--perps", action="store_true", help="Alias for --market-mode perps"
    )
    parser.add_argument("--max-strategies", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--strategy-ids", type=str, default="")
    parser.add_argument("--regime-only", action="store_true")
    parser.add_argument(
        "--no-regime-adaptor",
        action="store_true",
        help="Skip fitting simple-policy regime adaptor artifacts.",
    )
    args = parser.parse_args()

    cli_strategy_ids = [s.strip() for s in args.strategy_ids.split(",") if s.strip()]
    cli_market_mode = _normalise_market_mode(
        "perps" if args.perps else args.market_mode
    )
    if args.regime_only:
        run_regime_adaptor_only_from_simple_policy(
            args.data_root,
            args.run_id,
            strategy_ids=cli_strategy_ids,
            market_mode=cli_market_mode,
        )
    else:
        run_simple_policy_optimisation(
            args.data_root,
            args.run_id,
            max_strategies=args.max_strategies,
            n_trials=args.n_trials,
            strategy_ids=cli_strategy_ids,
            market_mode=cli_market_mode,
            enable_regime_adaptor=not args.no_regime_adaptor,
        )
