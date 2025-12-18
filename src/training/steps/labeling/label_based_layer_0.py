from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from src.training.steps.labeling.multi_label_voting_utils import (
    compute_kalman_smoothed_price_and_volatility,
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, OptimizationConfig


DEFAULT_RANDOM_SEED = 42


def get_reproducible_random_state(base_seed: int = DEFAULT_RANDOM_SEED, offset: int = 0) -> int:
    try:
        base_seed_i = int(base_seed)
    except Exception:
        base_seed_i = DEFAULT_RANDOM_SEED
    try:
        offset_i = int(offset)
    except Exception:
        offset_i = 0
    return int((base_seed_i + offset_i) % (2**31 - 1))


def _rolling_sum_prefix(values: np.ndarray, window: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = int(v.shape[0])
    w = int(max(1, min(int(window), n)))
    c = np.cumsum(np.where(np.isfinite(v), v, 0.0), dtype=float)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def _ffill_nan(arr: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    n = int(x.shape[0])
    if n == 0:
        return x
    mask = np.isfinite(x)
    if not bool(mask.any()):
        if fallback is None:
            return x
        return np.asarray(fallback, dtype=float)
    idx = np.where(mask, np.arange(n, dtype=int), 0)
    idx = np.maximum.accumulate(idx)
    out = x[idx]
    if fallback is not None:
        fb = np.asarray(fallback, dtype=float)
        out = np.where(np.isfinite(out), out, fb)
    return out


def compute_rolling_vwap(
    close: pd.Series,
    volume: Optional[pd.Series],
    lookback: int,
) -> pd.Series:
    close_s = pd.to_numeric(close, errors="coerce")
    close_vals = close_s.to_numpy(dtype=float)
    n = int(close_vals.shape[0])
    lb = int(max(2, min(int(lookback), max(2, n))))

    if volume is None:
        sum_close = _rolling_sum_prefix(close_vals, lb)
        denom = np.minimum(np.arange(1, n + 1, dtype=float), float(lb))
        out = sum_close / (denom + 1e-12)
        out = _ffill_nan(out, fallback=close_vals)
        return pd.Series(out, index=close_s.index)

    vol_s = pd.to_numeric(volume, errors="coerce")
    vol_vals = vol_s.to_numpy(dtype=float)
    if not bool(np.isfinite(vol_vals).any()):
        sum_close = _rolling_sum_prefix(close_vals, lb)
        denom = np.minimum(np.arange(1, n + 1, dtype=float), float(lb))
        out = sum_close / (denom + 1e-12)
        out = _ffill_nan(out, fallback=close_vals)
        return pd.Series(out, index=close_s.index)

    pv_vals = close_vals * np.where(np.isfinite(vol_vals), vol_vals, 0.0)
    v_safe = np.where(np.isfinite(vol_vals) & (vol_vals > 0.0), vol_vals, 0.0)
    sum_pv = _rolling_sum_prefix(pv_vals, lb)
    sum_v = _rolling_sum_prefix(v_safe, lb)
    out = sum_pv / (sum_v + 1e-12)
    out = np.where(sum_v > 0.0, out, np.nan)
    out = _ffill_nan(out, fallback=close_vals)
    return pd.Series(out, index=close_s.index)


def run_layer0_kalman_vwap(
    *,
    market_data: pd.DataFrame,
    config: Dict[str, Any],
    outcomes_dir: Path,
    bundle_path: Optional[Path] = None,
    run_optimization: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    close_series = pd.to_numeric(market_data.get("close"), errors="coerce")
    volume_series = market_data.get("volume", None)
    if isinstance(volume_series, pd.Series):
        volume_series = pd.to_numeric(volume_series, errors="coerce")

    if bundle_path is None:
        bundle_path = outcomes_dir / "layer0_kalman_bundle.joblib"

    best_params: Dict[str, Any] = {}
    loaded_from: Optional[str] = None
    if (not run_optimization) and bundle_path.exists():
        try:
            payload = joblib.load(bundle_path)
            best_params = dict(payload.get("best_params", {}) or {})
            loaded_from = str(bundle_path)
        except Exception:
            best_params = {}

    if run_optimization or not best_params:
        def _objective(params: Dict[str, Any]) -> float:
            Q = float(params.get("kalman_Q", 1e-4))
            R = float(params.get("kalman_R", 0.01))
            vwap_lookback = int(params.get("vwap_lookback", 20))
            vwap_lambda = float(params.get("vwap_lambda", 0.25))

            vwap_series = compute_rolling_vwap(close_series, volume_series, vwap_lookback)
            vol_w = None
            if volume_series is not None:
                vol_vals = pd.to_numeric(volume_series, errors="coerce").to_numpy(dtype=float)
                n = int(vol_vals.shape[0])
                lb = int(max(2, min(int(vwap_lookback), max(2, n))))
                vol_mean = _rolling_sum_prefix(np.where(np.isfinite(vol_vals), vol_vals, 0.0), lb)
                denom = np.minimum(np.arange(1, n + 1, dtype=float), float(lb))
                vol_mean = vol_mean / (denom + 1e-12)
                vol_rel = vol_vals / (vol_mean + 1e-12)
                w_track = np.clip(vol_rel, 0.1, 10.0)
                w_vwap = np.clip(1.0 / (vol_rel + 1e-12), 0.1, 10.0)
                vol_w = (w_track, w_vwap)

            try:
                smoothed_close, _smoothed_vol = compute_kalman_smoothed_price_and_volatility(
                    prices=close_series,
                    volume=volume_series,
                    vwap=vwap_series,
                    process_noise=Q,
                    measurement_noise=R,
                    vol_window=20,
                )

                raw = close_series.to_numpy(dtype=float)
                smooth = pd.to_numeric(smoothed_close, errors="coerce").to_numpy(dtype=float)
                vwap_vals = pd.to_numeric(vwap_series, errors="coerce").to_numpy(dtype=float)

                mask = np.isfinite(raw) & np.isfinite(smooth) & np.isfinite(vwap_vals)
                if int(mask.sum()) < 100:
                    return -10.0

                raw_m = raw[mask]
                smooth_m = smooth[mask]
                vwap_m = vwap_vals[mask]
                denom = float(np.nanstd(np.diff(raw_m))) + 1e-9

                smooth_pen = float(np.mean(np.diff(smooth_m, n=2) ** 2) / (denom**2))

                if vol_w is None:
                    track_pen = float(np.mean((smooth_m - raw_m) ** 2) / (denom**2))
                    vwap_pen = float(np.mean((smooth_m - vwap_m) ** 2) / (denom**2))
                else:
                    w_track, w_vwap = vol_w
                    w_track_m = np.asarray(w_track, dtype=float)[mask]
                    w_vwap_m = np.asarray(w_vwap, dtype=float)[mask]
                    track_pen = float(np.mean(w_track_m * ((smooth_m - raw_m) ** 2)) / (denom**2))
                    vwap_pen = float(np.mean(w_vwap_m * ((smooth_m - vwap_m) ** 2)) / (denom**2))

                vwap_lambda = float(np.clip(vwap_lambda, 0.0, 0.5))
                loss = smooth_pen + track_pen + vwap_lambda * vwap_pen
                score = -float(loss)
                return float(score) if np.isfinite(score) else -10.0
            except Exception:
                return -10.0

        optimizer = BayesianTPEOptimizer(
            config=OptimizationConfig(
                n_trials=int(config.get("layer0_n_trials", config.get("stage0_n_trials", 50))),
                execution_mode=str(config.get("execution_mode", "light")),
                direction="maximize",
                seed=int(config.get("random_state", 42)),
            )
        )
        search_space = {
            "kalman_Q": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "kalman_R": {"type": "float", "low": 1e-4, "high": 2e-1, "log": True},
            "vwap_lookback": {"type": "int", "low": 10, "high": 200, "log": False},
            "vwap_lambda": {"type": "float", "low": 0.0, "high": 0.5, "log": False},
        }
        opt_res = optimizer.optimize(objective=_objective, search_space=search_space)
        best_params = dict(opt_res.get("best_params", {}) or {})
        loaded_from = None

    try:
        Q_best = float(best_params.get("kalman_Q", 1e-4))
        R_best = float(best_params.get("kalman_R", 0.01))
    except Exception:
        Q_best, R_best = 1e-4, 0.01

    try:
        vwap_lb = int(best_params.get("vwap_lookback", 20))
    except Exception:
        vwap_lb = 20

    try:
        vwap_lambda = float(best_params.get("vwap_lambda", 0.25))
    except Exception:
        vwap_lambda = 0.25
    vwap_lambda = float(np.clip(vwap_lambda, 0.0, 0.5))

    vwap_series = market_data.get("vwap", None)
    if not isinstance(vwap_series, pd.Series) or bool(pd.to_numeric(vwap_series, errors="coerce").isna().all()):
        vwap_series = compute_rolling_vwap(close_series, volume_series, vwap_lb)
        market_data["vwap"] = vwap_series

    try:
        kalman_price, kalman_vol = compute_kalman_smoothed_price_and_volatility(
            prices=market_data["close"],
            volume=market_data.get("volume", None),
            vwap=market_data.get("vwap", None),
            process_noise=float(Q_best),
            measurement_noise=float(R_best),
            vol_window=20,
        )
        market_data["kalman_price"] = kalman_price
        market_data["kalman_volatility"] = kalman_vol
    except Exception:
        pass

    payload = {
        "best_params": {
            "kalman_Q": float(Q_best),
            "kalman_R": float(R_best),
            "vwap_lookback": int(vwap_lb),
            "vwap_lambda": float(vwap_lambda),
        },
        "loaded_from": loaded_from,
    }
    try:
        joblib.dump(payload, bundle_path)
    except Exception:
        pass

    try:
        outcomes_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        ts = str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    except Exception:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    try:
        symbol = str(config.get("symbol", ""))
    except Exception:
        symbol = ""
    try:
        timeframe = str(config.get("timeframe", ""))
    except Exception:
        timeframe = ""

    try:
        idx = market_data.index
        start_ts = str(idx.min()) if len(idx) else ""
        end_ts = str(idx.max()) if len(idx) else ""
    except Exception:
        start_ts, end_ts = "", ""

    try:
        md_path = outcomes_dir / f"layer0_report_{symbol}_{timeframe}_{ts}.md"
        lines = [
            "# Layer0 Report\n",
            f"- timestamp: {ts}\n",
            f"- symbol: {symbol}\n",
            f"- timeframe: {timeframe}\n",
            f"- run_optimization: {bool(run_optimization)}\n",
            f"- bundle_path: {str(bundle_path)}\n",
            f"- loaded_from: {str(loaded_from) if loaded_from else ''}\n",
            f"- n_bars: {int(len(market_data))}\n",
            f"- date_range: {start_ts} -> {end_ts}\n",
            "\n## Best Params\n",
            f"- kalman_Q: {float(Q_best)}\n",
            f"- kalman_R: {float(R_best)}\n",
            f"- vwap_lookback: {int(vwap_lb)}\n",
            f"- vwap_lambda: {float(vwap_lambda)}\n",
        ]
        md_path.write_text("".join(lines))
    except Exception:
        pass

    try:
        summary_row = {
            "timestamp": ts,
            "symbol": symbol,
            "timeframe": timeframe,
            "run_optimization": bool(run_optimization),
            "loaded_from": str(loaded_from) if loaded_from else "",
            "bundle_path": str(bundle_path),
            "n_bars": int(len(market_data)),
            "start": start_ts,
            "end": end_ts,
            "kalman_Q": float(Q_best),
            "kalman_R": float(R_best),
            "vwap_lookback": int(vwap_lb),
            "vwap_lambda": float(vwap_lambda),
        }
        csv_path = outcomes_dir / f"layer0_summary_{symbol}_{timeframe}_{ts}.csv"
        pd.DataFrame([summary_row]).to_csv(csv_path, index=False)
    except Exception:
        pass

    return market_data, payload
