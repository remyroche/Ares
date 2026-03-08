from __future__ import annotations

import hashlib
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Dict, Any, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.persistence.policy_params_store import (
    get_initial_params,
    load_params_store,
    save_params_store,
    store_best_params,
)
from extreme_price_movements.data_store import PartitionedOHLCVStore, load_features_selected
from extreme_price_movements.model_loader import load_full_state
from extreme_price_movements.offline_optimisers.params_store import (
    INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV,
    save_best_params_csv,
)
from extreme_price_movements.pnl import CostModel, trade_return_net_vec
from extreme_price_movements.ridge_position_sizer import (
    _aggregate_daily_values,
    _pnl_risk_objective,
    run_oof_grid_backtest,
    _stable_daily_pnl_metrics,
)
from extreme_price_movements.telemetry.tprint_hooks import emit_bucket_summary, emit_run_header
from extreme_price_movements.tpsl_optimiser import load_step_module
from extreme_price_movements.training_defaults import get_candidate_filter_defaults
from extreme_price_movements.utils import tprint


@dataclass(frozen=True)
class Policy:
    mode: Literal["train_baseline", "inference"] = "train_baseline"
    params_path: str | None = None
    ridge_weights_path: str | None = None  # Path to ridge sizer weights

    def baseline_params(self) -> dict:
        return {
            "tp_mult": 3.0,
            "sl_mult": 1.0,
            "atr_scale_lo": 0.6,
            "atr_scale_hi": 2.5,
            "risk_cut_mode": "TIMES",
            "theta0": 1.2,
            "theta_mae_min": 0.3,
            "lambda_rv": 0.5,
            "lambda_rng": 0.25,
            "sizing": {"k": 8.0, "c0": 0.70, "s_min": 0.03, "s_max": 0.15},
        }

    def resolve_params(self, bucket: str) -> dict:
        if self.mode == "train_baseline":
            return self.baseline_params()
        if not self.params_path:
            return self.baseline_params()
        p = Path(self.params_path)
        if not p.exists():
            return self.baseline_params()
        payload = json.loads(p.read_text())
        return payload.get("buckets", {}).get(str(bucket), self.baseline_params())
    
    def get_ridge_weights(self) -> Optional[Dict]:
        """Load ridge position sizer weights if available."""
        if not self.ridge_weights_path:
            return None
        p = Path(self.ridge_weights_path)
        if not p.exists():
            return None
        payload = json.loads(p.read_text())
        return payload.get("weights")


def load_ridge_weights_from_state(state_path: str) -> Optional[Dict]:
    """Load ridge sizer weights from training state file.
    
    Args:
        state_path: Path to trained_state.pkl
        
    Returns:
        Dict with weights, or None if not found
    """
    p = Path(state_path)
    if not p.exists():
        return None
    
    with open(p, "rb") as f:
        state = pickle.load(f)
    
    ridge_sizer = state.get("ridge_sizer", {})
    if ridge_sizer:
        return ridge_sizer.get("weights")
    return None


def load_ridge_offset_from_state(*, run_id: str | None, data_root: str) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    try:
        state = load_full_state(str(run_id), data_root)
        ridge_sizer = state.get("ridge_sizer")
        if ridge_sizer is None:
            return None
        bundle = getattr(ridge_sizer, "limit_offset_model_bundle_", None) or {}
        return {
            "base_name": bundle.get("base_name"),
            "smoother_name": bundle.get("smoother_name"),
            "features": list(getattr(ridge_sizer, "limit_offset_features_", None) or []),
            "diag": dict(getattr(ridge_sizer, "limit_offset_diag_", None) or {}),
        }
    except Exception as exc:
        tprint(f"optimise: WARNING could not load ridge offset optimiser: {exc}")
        return None


def run_optimise_from_ridge_oof(
    *,
    run_id: str,
    data_root: str,
    fee_roundtrip: float = 0.003,
    cooldown_hours: float = 0.0,
) -> Dict[str, Any]:
    """Run the cheap optimisation alternative on Ridge/limit-offset OOF outputs."""
    oof_dir = Path(data_root) / "artifacts" / str(run_id) / "ridge_sizer"
    oof_path = oof_dir / "ridge_sizer_oof_all.parquet"
    if not oof_path.exists():
        legacy_path = oof_dir / "ridge_sizer_oof.parquet"
        if legacy_path.exists():
            oof_path = legacy_path
    if not oof_path.exists():
        raise FileNotFoundError(
            f"Ridge OOF parquet not found at {oof_path}. Run the sizer step first."
        )

    oof_df = pd.read_parquet(oof_path)
    if oof_df.empty:
        raise ValueError(f"Ridge OOF parquet is empty: {oof_path}")

    tprint(
        "optimise: running Ridge OOF grid "
        f"(rows={len(oof_df)} fee_roundtrip={float(fee_roundtrip):.6f} cooldown_hours={float(cooldown_hours):.2f})"
    )
    if "bucket" in oof_df.columns and oof_df["bucket"].notna().any():
        bucket_frames: list[pd.DataFrame] = []
        bucket_best: dict[str, dict[str, Any]] = {}
        for bucket, bucket_df in oof_df.groupby(oof_df["bucket"].astype(str).str.upper(), sort=True):
            bucket_grid = run_oof_grid_backtest(
                oof_df=bucket_df,
                fee_roundtrip=float(fee_roundtrip),
                cooldown_hours=float(cooldown_hours),
            )
            if bucket_grid.empty:
                continue
            bucket_grid = bucket_grid.copy()
            bucket_grid["bucket"] = str(bucket)
            sort_cols = [c for c in ("net_pnl", "sortino", "win_rate") if c in bucket_grid.columns]
            if sort_cols:
                bucket_grid = bucket_grid.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
            bucket_frames.append(bucket_grid)
            bucket_best[str(bucket)] = bucket_grid.iloc[0].to_dict()
        if not bucket_frames:
            raise RuntimeError("Ridge OOF optimisation grid produced no rows for any bucket")
        grid_df = pd.concat(bucket_frames, ignore_index=True)
        best_row = dict(
            sorted(
                bucket_best.items(),
                key=lambda kv: (
                    float(kv[1].get("net_pnl", float("-inf"))),
                    float(kv[1].get("sortino", float("-inf"))),
                    float(kv[1].get("win_rate", float("-inf"))),
                ),
                reverse=True,
            )[0][1]
        )
    else:
        grid_df = run_oof_grid_backtest(
            oof_df=oof_df,
            fee_roundtrip=float(fee_roundtrip),
            cooldown_hours=float(cooldown_hours),
        )
        if grid_df.empty:
            raise RuntimeError("Ridge OOF optimisation grid produced no rows")
        sort_cols = [c for c in ("net_pnl", "sortino", "win_rate") if c in grid_df.columns]
        if sort_cols:
            ascending = [False] * len(sort_cols)
            grid_df = grid_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        best_row = grid_df.iloc[0].to_dict()
        bucket_best = {}

    out_dir = oof_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "ridge_oof_optimise_grid.csv"
    best_path = out_dir / "ridge_oof_optimise_best.json"
    grid_df.to_csv(grid_path, index=False)

    summary = {
        "mode": "ridge_oof",
        "run_id": str(run_id),
        "source_oof_path": str(oof_path),
        "grid_path": str(grid_path),
        "best": best_row,
        "best_by_bucket": bucket_best,
        "fee_roundtrip": float(fee_roundtrip),
        "cooldown_hours": float(cooldown_hours),
        "n_rows": int(len(oof_df)),
    }
    best_path.write_text(json.dumps(summary, indent=2))
    tprint(
        "optimise: Ridge OOF best "
        f"phase={best_row.get('phase')} q={best_row.get('quantile')} "
        f"offset={best_row.get('entry_offset_mode')} sizing={best_row.get('sizing_mode')} "
        f"net_pnl={float(best_row.get('net_pnl', 0.0)):.6f}"
    )
    tprint(f"optimise: Ridge OOF grid saved={grid_path}")
    tprint(f"optimise: Ridge OOF summary saved={best_path}")
    return summary


def _adapt_backtest_columns(trades: pd.DataFrame) -> pd.DataFrame:
    """Map backtest_results.csv columns to tpsl_optimiser expected schema."""
    df = trades.copy()
    # timestamp
    if "timestamp" not in df.columns and "entry_ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["entry_ts"], utc=True)
    # confidence
    if "confidence" not in df.columns and "score" in df.columns:
        df["confidence"] = df["score"].abs().clip(0, 1)
    # entry_price
    if "entry_price" not in df.columns and "entry_px" in df.columns:
        df["entry_price"] = df["entry_px"]
    if "signal_px" not in df.columns:
        df["signal_px"] = df["entry_price"]
    # exit_price — reconstruct from entry_px + gross_ret
    if "exit_price" not in df.columns:
        if "entry_px" in df.columns and "gross_ret" in df.columns and "side" in df.columns:
            is_long = (df["side"] == "long").astype(int)
            df["exit_price"] = np.where(
                is_long == 1,
                df["entry_px"] * (1.0 + df["gross_ret"]),
                df["entry_px"] * (1.0 - df["gross_ret"]),
            )
        else:
            df["exit_price"] = df.get("entry_price", df.get("entry_px", 1.0))
    # is_long
    if "is_long" not in df.columns and "side" in df.columns:
        df["is_long"] = (df["side"] == "long").astype(int)
    # bucket
    if "bucket" not in df.columns and "side" in df.columns and "dom" in df.columns:
        df["bucket"] = df["side"].str.upper() + "_" + df["dom"].str.upper()
    return df


def _to_utc_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return pd.Series(ts, index=series.index)


def _load_candidate_grid_context(
    *,
    trades: pd.DataFrame,
    run_id: str,
    data_root: str,
    store: PartitionedOHLCVStore | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the selector feature context used to score candidate-mask grids."""
    if store is None:
        store = PartitionedOHLCVStore(root_dir=data_root, timeframe="1h")

    ts_run = pd.to_datetime(run_id, format="%Y%m%d_%H%M%S", utc=True)
    cached = load_features_selected(
        ts_run,
        data_root,
        feature_keys=["range_12h_pct", "volatility_zscore"],
        symbols=None,
    ) or {}
    vol_z = cached.get("volatility_zscore")
    if not isinstance(vol_z, pd.DataFrame) or vol_z.empty:
        raise RuntimeError("Could not load volatility_zscore from cached features for candidate-mask optimisation")
    range_12h = cached.get("range_12h_pct")
    if not isinstance(range_12h, pd.DataFrame) or range_12h.empty:
        range_12h = pd.DataFrame(True, index=vol_z.index, columns=vol_z.columns, dtype=np.float32)

    entry_ts = _to_utc_timestamp(trades["entry_ts"])
    start_ts = entry_ts.min() - pd.Timedelta(hours=12)
    end_ts = entry_ts.max()
    symbols = list(vol_z.columns)
    
    # OPTIMIZATION: Parallel symbol loading using ThreadPoolExecutor
    def _load_symbol(sym: str) -> tuple[str, pd.Series]:
        """Load close data for a single symbol."""
        try:
            df_sym = store.load(sym, columns=["close"], start_ts=start_ts, end_ts=end_ts)
            if df_sym.empty or "close" not in df_sym.columns:
                return sym, pd.Series(dtype=np.float64)
            s = pd.to_numeric(df_sym["close"], errors="coerce").rename(sym)
            return sym, s
        except Exception:
            return sym, pd.Series(dtype=np.float64)
    
    close_parts: list[pd.Series] = []
    n_workers = min(32, max(4, len(symbols)))
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_load_symbol, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, series = future.result()
            if not series.empty:
                close_parts.append(series)
    
    if not close_parts:
        raise RuntimeError("Could not load OHLCV close data for candidate-mask optimisation")
    close = pd.concat(close_parts, axis=1).sort_index()
    close = close.reindex(columns=symbols)
    ret12h = close / close.shift(12) - 1.0
    idx = ret12h.index
    ret12h = ret12h[(idx >= start_ts) & (idx <= end_ts)]
    vol_z = vol_z[(vol_z.index >= start_ts) & (vol_z.index <= end_ts)].reindex(index=ret12h.index, columns=ret12h.columns)
    range_12h = range_12h[(range_12h.index >= start_ts) & (range_12h.index <= end_ts)].reindex(index=ret12h.index, columns=ret12h.columns)
    return ret12h.astype(np.float32), vol_z.astype(np.float32), range_12h.astype(np.float32)


def _select_candidate_trade_mask(
    trades: pd.DataFrame,
    ret12h: pd.DataFrame,
    vol_z: pd.DataFrame,
    *,
    pct: float,
    min_move_12h_pct: float,
    min_vol_zscore: float,
) -> pd.Series:
    """Return a boolean mask of trades that pass the candidate filter."""
    if ret12h.empty or vol_z.empty:
        return pd.Series(False, index=trades.index)

    arr = ret12h.to_numpy(dtype=np.float32, copy=False)
    valid = np.isfinite(arr)
    n_rows, n_cols = arr.shape
    if n_rows == 0 or n_cols == 0:
        return pd.Series(False, index=trades.index)
    k = max(1, int(n_cols * float(pct)))
    arr_top = np.where(valid, arr, -np.inf)
    arr_bot = np.where(valid, arr, np.inf)
    top_idx = np.argpartition(arr_top, kth=max(n_cols - k, 0), axis=1)[:, -k:]
    bot_idx = np.argpartition(arr_bot, kth=max(k - 1, 0), axis=1)[:, :k]
    rows = np.repeat(np.arange(n_rows, dtype=np.int32), k)
    top_mask_arr = np.zeros_like(valid, dtype=bool)
    bot_mask_arr = np.zeros_like(valid, dtype=bool)
    top_flat = top_idx.reshape(-1)
    bot_flat = bot_idx.reshape(-1)
    top_mask_arr[rows, top_flat] = valid[rows, top_flat]
    bot_mask_arr[rows, bot_flat] = valid[rows, bot_flat]
    move_mask = np.abs(arr) >= float(min_move_12h_pct)
    z_arr = vol_z.to_numpy(dtype=np.float32, copy=False)
    z_mask = np.isfinite(z_arr) & (z_arr >= float(min_vol_zscore))
    sign_long = arr > 0.0
    sign_short = arr < 0.0
    long_mask = pd.DataFrame(top_mask_arr & move_mask & z_mask & sign_long, index=ret12h.index, columns=ret12h.columns)
    short_mask = pd.DataFrame(bot_mask_arr & move_mask & z_mask & sign_short, index=ret12h.index, columns=ret12h.columns)

    trade_ts = _to_utc_timestamp(trades["entry_ts"])
    trade_sym = trades["symbol"].astype(str)
    long_idx = pd.MultiIndex.from_arrays([trade_ts, trade_sym])
    long_series = long_mask.stack(future_stack=True)
    short_series = short_mask.stack(future_stack=True)
    side = trades["side"].astype(str).str.lower()
    is_long = side.eq("long")
    is_short = side.eq("short")
    out = pd.Series(False, index=trades.index)
    if is_long.any():
        out.loc[is_long] = long_series.reindex(long_idx[is_long], fill_value=False).to_numpy(dtype=bool)
    if is_short.any():
        out.loc[is_short] = short_series.reindex(long_idx[is_short], fill_value=False).to_numpy(dtype=bool)
    return out


def _score_candidate_mask(
    trades: pd.DataFrame,
    selected_mask: pd.Series,
) -> dict[str, float]:
    selected = trades.loc[selected_mask.fillna(False)].copy()
    if selected.empty:
        return {
            "n_selected": 0.0,
            "PnL_per_day": -1e9,
            "IntradayRisk": 1e9,
            "ObjectiveScore": -1e9,
        }
    pnl = pd.to_numeric(selected["pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ts = _to_utc_timestamp(selected["entry_ts"]).to_numpy()
    total_pnl = float(np.sum(pnl))
    ts_valid = pd.to_datetime(ts)
    n_days = (ts_valid.max() - ts_valid.min()).total_seconds() / 86400.0 if len(ts_valid) else 1.0
    n_days = max(1.0 / 24.0, float(n_days))
    pnl_per_day = total_pnl / n_days
    daily_returns = _aggregate_daily_values(pnl, ts)
    _, max_dd, ulcer, tuw = _stable_daily_pnl_metrics(pnl, ts, start_equity=1.0)
    objective = _pnl_risk_objective(
        pnl_total=total_pnl,
        max_dd=max_dd,
        ulcer=ulcer,
        tuw=tuw,
        daily_returns=daily_returns,
    )
    intraday_risk = float((25.0 * max(float(max_dd), 0.0)) + (2.0 * max(float(ulcer), 0.0)) + (0.5 * max(float(tuw), 0.0)))
    return {
        "n_selected": float(len(selected)),
        "PnL_per_day": float(pnl_per_day),
        "IntradayRisk": float(intraday_risk),
        "ObjectiveScore": float(objective),
    }


def _optimise_candidate_mask_grid(
    *,
    trades: pd.DataFrame,
    run_id: str,
    data_root: str,
    output_path: str,
    store: PartitionedOHLCVStore | None = None,
) -> dict[str, float]:
    """Optimise inference candidate-mask parameters using the ridge-sizer objective."""
    defaults = get_candidate_filter_defaults()
    ret12h, vol_z, _ = _load_candidate_grid_context(
        trades=trades,
        run_id=run_id,
        data_root=data_root,
        store=store,
    )
    results: list[dict[str, float]] = []
    pct_grid = [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
    move_grid = [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
    z_grid = [1.4, 1.5, 1.6, 1.7, 1.8]
    for pct in pct_grid:
        for move in move_grid:
            for zthr in z_grid:
                mask = _select_candidate_trade_mask(
                    trades,
                    ret12h,
                    vol_z,
                    pct=pct,
                    min_move_12h_pct=move,
                    min_vol_zscore=zthr,
                )
                score = _score_candidate_mask(trades, mask)
                results.append(
                    {
                        "pct": float(pct),
                        "min_move_12h_pct": float(move),
                        "min_vol_zscore": float(zthr),
                        **score,
                    }
                )
    results_df = pd.DataFrame(results).sort_values(
        ["ObjectiveScore", "PnL_per_day", "n_selected"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best = results_df.iloc[0].to_dict() if not results_df.empty else {}
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        results_df.to_csv(output_path, index=False)
    best_params = {
        "train_extreme_pct_hourly": float(best.get("pct", defaults["train_extreme_pct_hourly"])),
        "train_min_move_12h_pct": float(best.get("min_move_12h_pct", defaults["train_min_move_12h_pct"])),
        "train_min_vol_zscore": float(best.get("min_vol_zscore", defaults["train_min_vol_zscore"])),
        "train_candidate_metric": "ret12h",
    }
    save_best_params_csv(
        INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV,
        best_params,
        metadata={
            "source": "optimise_candidate_mask_grid",
            "ObjectiveScore": float(best.get("ObjectiveScore", -1e9)),
            "PnL_per_day": float(best.get("PnL_per_day", -1e9)),
            "IntradayRisk": float(best.get("IntradayRisk", 1e9)),
        },
    )
    tprint(
        "optimise: candidate mask best params "
        f"pct={best_params['train_extreme_pct_hourly']:.2f} "
        f"move12h={best_params['train_min_move_12h_pct']:.2f} "
        f"volz={best_params['train_min_vol_zscore']:.1f} "
        f"objective={float(best.get('ObjectiveScore', -1e9)):.6f}"
    )
    return best_params


def run_optimise_step(
    trades: pd.DataFrame,
    atr_15m: pd.Series,
    output_path: str,
    policy: Policy | None = None,
    state_path: str | None = None,
    cost: CostModel | None = None,
    enforce_threaded_exit_stream: bool = True,
    store_base_dir: str | Path | None = None,
    run_id: str | None = None,
    data_root: str = "data",
    ohlcv_store: PartitionedOHLCVStore | None = None,
) -> dict:
    """Run the optimisation pipeline for TP/SL and position sizing.

    Args:
        trades: DataFrame with backtest trade results
        atr_15m: Series with 15-minute ATR values
        output_path: Path to save optimisation results
        policy: Policy configuration (mode, params_path, ridge_weights_path)
        state_path: Optional path to trained_state.pkl for loading ridge weights
        store_base_dir: Optional base directory whose `artifacts/` folder should
            hold the policy params store (defaults to module-local artifacts)

    Returns:
        Dict with optimisation results per bucket
    """
    policy = policy or Policy(mode="train_baseline")

    step_run_id = str(run_id or Path(output_path).stem)
    policy_version = step_run_id

    candidate_grid_path = str(Path(output_path).with_name("candidate_mask_grid.csv"))
    try:
        _optimise_candidate_mask_grid(
            trades=trades,
            run_id=step_run_id,
            data_root=str(data_root),
            output_path=candidate_grid_path,
            store=ohlcv_store,
        )
    except Exception as exc:
        tprint(f"optimise: WARNING candidate-mask optimisation failed: {exc}")


    # Try to load ridge weights from policy or state file
    ridge_weights = None
    if policy.ridge_weights_path:
        ridge_weights = policy.get_ridge_weights()
        if ridge_weights:
            tprint(f"Loaded ridge weights from policy path: {policy.ridge_weights_path}")
    if ridge_weights is None and state_path:
        ridge_weights = load_ridge_weights_from_state(state_path)
        if ridge_weights:
            tprint(f"Loaded ridge weights from state file: {state_path}")
    ridge_offset_model = load_ridge_offset_from_state(run_id=step_run_id, data_root=str(data_root))
    if ridge_offset_model:
        tprint(
            "Loaded ridge offset optimiser: "
            f"base={ridge_offset_model.get('base_name')} smoother={ridge_offset_model.get('smoother_name')}"
        )

    # Adapt column names from backtest output to tpsl_optimiser schema
    trades = _adapt_backtest_columns(trades)

    m00 = load_step_module("00_load_trades.py")
    m05 = load_step_module("05_entry_offset_opt.py")
    m10 = load_step_module("10_tp_sl_calibration.py")
    m20 = load_step_module("20_loss_limiter_opt.py")
    m30 = load_step_module("30_profit_exit_opt.py")
    m40 = load_step_module("40_position_sizing_opt.py")
    m50 = load_step_module("50_eval_holdout_report.py")
    mw = load_step_module("write_params_json.py")

    # Pre-load 15m data for all trades to avoid redundant disk I/O during TP/SL calibration.
    # FIX #12: slice to the relevant time window *before* storing, so we don't hold the full
    # multi-year history of each asset in memory.
    # FIX #13: log a warning per asset if the load raises an exception.
    # OPTIMIZATION: Parallel 15m data loading using ThreadPoolExecutor
    df_15m_dict = {}
    if "asset" in trades.columns and "timestamp" in trades.columns:
        from extreme_price_movements.hf_data_loader import _load_existing_data
        unique_assets = trades["asset"].unique()
        
        # Pre-compute time windows for all assets
        asset_windows = {}
        for asset in unique_assets:
            asset_trades = trades[trades["asset"] == asset]
            min_ts = pd.to_datetime(asset_trades["timestamp"].min(), utc=True)
            if "label_policy_max_hold_bars" in asset_trades.columns:
                max_hold_h = float(asset_trades["label_policy_max_hold_bars"].max() / 4.0)
            else:
                max_hold_h = 24.0
            max_ts = pd.to_datetime(asset_trades["timestamp"].max(), utc=True) + pd.Timedelta(hours=max_hold_h)
            asset_windows[asset] = (min_ts, max_ts)
        
        def _load_15m_asset(asset: str) -> tuple[str, pd.DataFrame | None, str | None]:
            """Load 15m data for a single asset."""
            try:
                min_ts, max_ts = asset_windows[asset]
                df_15m = _load_existing_data(asset)
                if df_15m.empty:
                    return asset, None, "empty data"
                # Slice to relevant window
                sliced = df_15m.loc[min_ts:max_ts]
                return asset, sliced, None
            except Exception as e:
                return asset, None, str(e)
        
        # Parallel loading of all assets
        n_workers = min(16, max(2, len(unique_assets)))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_load_15m_asset, asset): asset for asset in unique_assets}
            for future in as_completed(futures):
                asset, df_15m, error = future.result()
                if error:
                    tprint(f"optimise: WARNING could not load 15m data for {asset}: {error} — double-hit resolution unavailable for this asset")
                elif df_15m is not None and not df_15m.empty:
                    df_15m_dict[asset] = df_15m

    buckets = list(pd.Series(trades["bucket"].astype(str).unique()).sort_values())[:4]
    all_out = {}

    # List to collect all trials across all buckets and steps
    all_trials_log = []

    fee_pct = float((trades.attrs.get("fee_pct") if hasattr(trades, "attrs") else None) or 0.003)
    cost = cost or CostModel(fee_side=fee_pct / 2.0)
    cost_dict = {"fee_side": float(cost.fee_side), "slippage_side": float(cost.slippage_side), "round_trip": float(cost.round_trip)}
    cost_hash = hashlib.sha256(json.dumps(cost_dict, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    emit_run_header(tprint=tprint, run_id=step_run_id, policy_version=policy_version, cost_model=cost_dict, extra={"n_buckets": len(buckets)})

    store = load_params_store(store_base_dir)
    version_key = f"{policy_version}|{cost_hash}"

    for bucket in buckets:
        bucket_df = m00.load_trades_for_bucket(trades, bucket)
        bucket_df.attrs["threaded_exit_stream"] = bool(
            bucket_df.attrs.get("threaded_exit_stream", trades.attrs.get("threaded_exit_stream", False))
        )
        if bucket_df.empty:
            continue

        # Step 05: Learn fill-probability + pick EU-optimal entry offsets.
        step05_cfg = m05.EntryOffsetConfig()
        pol_features = m05.build_policy_features(bucket_df)
        fill_model, fill_meta = m05.fit_fill_model(bucket_df, pol_features, cfg=step05_cfg)
        pol_eval = m05.choose_entry_offsets(pol_features, fill_model, cfg=step05_cfg)
        bucket_df = m05.apply_effective_policy_params(
            bucket_df,
            pol_eval,
            base_params=policy.resolve_params(bucket),
            cfg=step05_cfg,
        )
        if "place_order" in bucket_df.columns:
            bucket_df = bucket_df[bucket_df["place_order"].astype(bool)].copy()
        if bucket_df.empty:
            tprint(f"optimise: bucket={bucket} no eligible trades after entry policy filter")
            continue
        _off_meta = {
            "mode": str(pol_eval.attrs.get("offset_engine_mode", "policy_only")),
            "lambda": float(pol_eval.attrs.get("offset_engine_lambda", 0.0)),
            "oos_score": float(pol_eval.attrs.get("offset_engine_oos_score", 0.0)),
            "oos_mean_eu": float(pol_eval.attrs.get("offset_engine_oos_mean_eu", 0.0)),
            "oos_place_rate": float(pol_eval.attrs.get("offset_engine_oos_place_rate", 0.0)),
        }
        entry_policy_payload = m05.build_entry_policy_payload(
            fill_model,
            step05_cfg,
            fill_meta,
            offset_engine_meta=_off_meta,
        )
        trials_05 = pd.DataFrame(
            {
                "delta_atr_star": bucket_df.get("delta_atr_star", pd.Series(dtype=float)).values,
                "p_fill_star": bucket_df.get("p_fill_star", pd.Series(dtype=float)).values,
                "eu_star": bucket_df.get("eu_star", pd.Series(dtype=float)).values,
                "place_order": bucket_df.get("place_order", pd.Series(dtype=bool)).astype(int).values,
            }
        )
        if not trials_05.empty:
            trials_05["bucket"] = bucket
            trials_05["step"] = "05_entry_policy"
            all_trials_log.append(trials_05)

        # Update store with Step 05 best params for next steps
        store = store_best_params(
            store=store,
            version_key=version_key,
            bucket_id=bucket,
            params={"entry_policy": entry_policy_payload},
            metrics={},
        )
        tprint(f"optimise: bucket={bucket} Step 05: saved entry_policy params to store")

        # Determine test split index (same as m50 uses)
        n = len(bucket_df)
        split_idx = max(1, int(n * 0.30)) if n > 0 else 0

        atr_scale = m10.compute_atr_scale(atr_15m.reindex(bucket_df.index).ffill().fillna(atr_15m.median()))

        # Reload params to get Step 05's best params
        params_init = get_initial_params(store, version_key, bucket, defaults=policy.resolve_params(bucket))
        tprint(f"optimise: bucket={bucket} loaded params for Step 10: keys={list(params_init.keys())}")

        # Step 10: TP/SL Calibration
        tp_sl, trials_10 = m10.calibrate_tp_sl(
            bucket_df, atr_scale, df_15m_dict=df_15m_dict, test_split_idx=split_idx, fee_pct=fee_pct, cost=cost,
            init_params=params_init.get("tp_sl", params_init)
        )
        trials_10["bucket"] = bucket
        trials_10["step"] = "10_tp_sl"
        all_trials_log.append(trials_10)

        # Update store with Step 10 best params for next steps
        store = store_best_params(
            store=store,
            version_key=version_key,
            bucket_id=bucket,
            params={"tp_sl": tp_sl},
            metrics={},
        )
        tprint(f"optimise: bucket={bucket} Step 10: saved tp_sl params to store (tp_mult={tp_sl.get('tp_mult', 0):.2f}, sl_mult={tp_sl.get('sl_mult', 0):.2f})")

        sl_pct = tp_sl["sl_mult"] * atr_scale.to_numpy()

        # Reload params to get Step 10's best params
        params_init = get_initial_params(store, version_key, bucket, defaults=policy.resolve_params(bucket))
        tprint(f"optimise: bucket={bucket} loaded params for Step 20: keys={list(params_init.keys())}")

        # Step 20: Loss Limiter Optimization
        risk_cut, trials_20 = m20.optimise_loss_limiter(bucket_df, sl_pct=sl_pct, test_split_idx=split_idx, fee_pct=fee_pct, cost=cost, init_params=params_init.get("loss_limiter", params_init))
        trials_20["bucket"] = bucket
        trials_20["step"] = "20_risk_cut"
        all_trials_log.append(trials_20)

        # Update store with Step 20 best params for next steps
        store = store_best_params(
            store=store,
            version_key=version_key,
            bucket_id=bucket,
            params={"loss_limiter": risk_cut},
            metrics={},
        )
        tprint(f"optimise: bucket={bucket} Step 20: saved loss_limiter params to store (theta0={risk_cut.get('theta0', 0):.2f}, theta_mae_min={risk_cut.get('theta_mae_min', 0):.2f})")

        raw_returns = np.where(bucket_df["is_long"].astype(int).to_numpy() == 1,
                               (bucket_df["exit_price"] - bucket_df["entry_price"]) / bucket_df["entry_price"],
                               (bucket_df["entry_price"] - bucket_df["exit_price"]) / bucket_df["entry_price"])

        tp_pct_entry = tp_sl["tp_mult"] * atr_scale.to_numpy()

        # Reload params to get Step 10 and 20's best params
        params_init = get_initial_params(store, version_key, bucket, defaults=policy.resolve_params(bucket))
        tprint(f"optimise: bucket={bucket} loaded params for Step 30: keys={list(params_init.keys())}")

        # Step 30: Profit Exit Optimization
        profit, trials_30 = m30.optimise_profit_exit(bucket_df, raw_returns, tp_pct_entry=tp_pct_entry, fee_pct=fee_pct, test_split_idx=split_idx, cost=cost, init_params=params_init.get("profit_exit", params_init))
        trials_30["bucket"] = bucket
        trials_30["step"] = "30_profit_exit"
        all_trials_log.append(trials_30)

        # Update store with Step 30 best params for next steps
        store = store_best_params(
            store=store,
            version_key=version_key,
            bucket_id=bucket,
            params={"profit_exit": profit},
            metrics={},
        )
        tprint(f"optimise: bucket={bucket} Step 30: saved profit_exit params to store (lambda_rv={profit.get('lambda_rv', 0):.2f}, lambda_rng={profit.get('lambda_rng', 0):.2f})")

        # Reload params to get Step 10, 20, and 30's best params
        params_init = get_initial_params(store, version_key, bucket, defaults=policy.resolve_params(bucket))
        tprint(f"optimise: bucket={bucket} loaded params for Step 40: keys={list(params_init.keys())}")

        # Step 40: Position Sizing Optimization
        threaded_exit_stream = bool(
            bucket_df.attrs.get("threaded_exit_stream", trades.attrs.get("threaded_exit_stream", False))
        )
        if enforce_threaded_exit_stream and not threaded_exit_stream:
            raise RuntimeError("Stage40 sizing is using stale exit stream; thread post-20/30 ledger first.")
        # Pass raw exit/entry/is_long as original code did, but metrics will use them
        sizing, trials_40 = m40.optimise_position_sizing(
            bucket_df,
            bucket_df["exit_price"].to_numpy(dtype=float),
            bucket_df["entry_price"].to_numpy(dtype=float),
            bucket_df["is_long"].to_numpy(dtype=int),
            bucket_df["confidence"].to_numpy(dtype=float),
            test_split_idx=split_idx,
            fee_pct=fee_pct,
            cost=cost,
            init_params=params_init.get("position_sizing", params_init.get("sizing", params_init))
        )
        trials_40["bucket"] = bucket
        trials_40["step"] = "40_sizing"
        all_trials_log.append(trials_40)

        tprint(f"optimise: bucket={bucket} Step 40: completed position sizing (k={sizing.get('k', 0):.2f}, c0={sizing.get('c0', 0):.2f})")

        # Apply ridge weights to confidence if available
        confidence = bucket_df["confidence"].to_numpy(dtype=float)
        if ridge_weights:
            # Ridge weights are for meta model combination
            # Here we use them to adjust confidence scaling
            # This is a simplified integration - full integration would combine
            # multiple model predictions using the weights
            tprint(f"  Ridge weights available for bucket {bucket}: using for confidence scaling")
            # Store ridge weights in sizing output for reference
            sizing["ridge_weights"] = ridge_weights
        if ridge_offset_model:
            sizing["ridge_offset_model"] = ridge_offset_model

        pos_size = m40.sigmoid_sizing(confidence, sizing["k"], sizing["c0"], sizing["s_min"], sizing["s_max"])
        net_returns = trade_return_net_vec(raw_ret_underlying=raw_returns, side=np.ones(len(raw_returns)), pos_w=pos_size, cost=cost)
        report = m50.evaluate_holdout(bucket_df, net_returns)
        holdout_ledger = m50.build_holdout_trade_ledger(bucket_df, net_returns, cost=cost)
        ledger_path = Path(output_path).with_name(f"{Path(output_path).stem}_{bucket}_holdout_ledger.csv")
        if not holdout_ledger.empty:
            holdout_ledger.to_csv(ledger_path, index=False)
        report["holdout_ledger_path"] = str(ledger_path)

        if not holdout_ledger.empty:
            emit_bucket_summary(
                tprint=tprint,
                run_id=step_run_id,
                bucket_id=bucket,
                kind="optimiser_eval",
                stats={
                    "ledger_rows": int(len(holdout_ledger)),
                    "ledger_checksum": hashlib.sha256(holdout_ledger.to_csv(index=False).encode("utf-8")).hexdigest()[:12],
                    "holdout_pnl_net": float(report.get("holdout_pnl_net", 0.0)),
                    "best_tp_mult": float(tp_sl.get("tp_mult", 0.0)),
                    "best_sl_mult": float(tp_sl.get("sl_mult", 0.0)),
                    "best_theta0": float(risk_cut.get("theta0", 0.0)),
                    "best_act_n": float(profit.get("act_n", 0.0)),
                    "best_size_k": float(sizing.get("k", 0.0)),
                    "threaded_exit_stream": threaded_exit_stream,
                },
            )

        combined = {
            "policy_mode": policy.mode,
            "baseline_seed": params_init,
            "entry_policy": entry_policy_payload,
            "tp_sl": tp_sl,
            "loss_limiter": risk_cut,
            "profit_exit": profit,
            "position_sizing": sizing,
            "evaluation": report,
        }
        
        # Add ridge weights to output if available
        if ridge_weights:
            combined["ridge_weights"] = ridge_weights
        if ridge_offset_model:
            combined["ridge_offset_model"] = ridge_offset_model
        
        all_out[bucket] = combined
        mw.merge_and_write_params(output_path, bucket, combined)

        store = store_best_params(
            store=store,
            version_key=version_key,
            bucket_id=bucket,
            params={
                "entry_policy": entry_policy_payload,
                "tp_sl": tp_sl,
                "loss_limiter": risk_cut,
                "profit_exit": profit,
                "position_sizing": sizing,
            },
            metrics={
                "holdout_pnl_net": float(report.get("holdout_pnl_net", 0.0)),
                "holdout_win_rate": float(report.get("holdout_win_rate", 0.0)),
                "holdout_trades": int(report.get("holdout_trades", 0)),
            },
        )
        tprint(f"optimise: bucket={bucket} trades={len(bucket_df)} saved={output_path}")

    # Concatenate and save consolidated report CSV
    if all_trials_log:
        consolidated_df = pd.concat(all_trials_log, ignore_index=True)
        # Reorder columns to put context first
        cols = ["bucket", "step"] + [c for c in consolidated_df.columns if c not in ["bucket", "step"]]
        consolidated_df = consolidated_df[cols]

        # Save CSV alongside JSON output (same directory, change extension)
        csv_path = Path(output_path).with_suffix(".csv")
        consolidated_df.to_csv(csv_path, index=False)
        tprint(f"optimise: detailed report saved={csv_path}")

    save_params_store(store, store_base_dir)
    return all_out
