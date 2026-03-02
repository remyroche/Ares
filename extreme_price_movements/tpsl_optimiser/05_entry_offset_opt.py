from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from extreme_price_movements.limit_order_pricer import estimate_entry_limit_offset


EPS = 1e-12


@dataclass(frozen=True)
class EntryOffsetConfig:
    a: float = 1.0
    lambda_risk: float = 1.0
    c_atr: float = 0.3
    delta_atr_max: float = 3.0
    delta_atr_step: float = 0.25
    min_expected_utility: float = 0.0
    alpha0_default: float = 0.4
    beta_default: float = 0.4
    alpha_u_default: float = 0.25
    alpha_mae_default: float = 0.25
    q_sl: float = 1.3
    eta_stop: float = 0.4
    r_tp: float = 1.5
    trail_mult_k_delta: float = -0.08
    trail_mult_k_mfe: float = 0.06
    giveback_k_delta: float = 0.05
    giveback_k_dur: float = 0.04
    lock_amt_k_u: float = 0.08
    kill_c_k_mae: float = 0.08
    hold_h_k_dur: float = 0.20
    offset_engine_modes: Tuple[str, ...] = ("policy_only", "estimator_only", "blended")
    offset_blend_lambdas: Tuple[float, ...] = (0.25, 0.50, 0.75)
    offset_engine_oos_frac: float = 0.30

    @property
    def delta_atr_grid(self) -> np.ndarray:
        n = int(round(self.delta_atr_max / max(self.delta_atr_step, 1e-6)))
        return np.linspace(0.0, self.delta_atr_max, n + 1)


def _zscore(v: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    x = np.asarray(v, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, fallback, dtype=float)
    mu = float(np.nanmean(x[finite]))
    sd = float(np.nanstd(x[finite]))
    if sd < 1e-12:
        out = np.zeros_like(x, dtype=float)
        out[~finite] = fallback
        return out
    out = (x - mu) / sd
    out[~finite] = fallback
    return out


def _pick_first(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray | None:
    for c in cols:
        if c in df.columns:
            return np.asarray(df[c].values, dtype=float)
    return None


def _safe_atr_from_trades(df: pd.DataFrame) -> np.ndarray:
    cand = _pick_first(df, ("atr_pct_15m", "atr_pct", "atr", "sl_pct"))
    if cand is None:
        return np.full(len(df), 0.02, dtype=float)
    out = np.asarray(cand, dtype=float)
    out = np.where(np.isfinite(out), np.abs(out), np.nan)
    med = float(np.nanmedian(out[np.isfinite(out)])) if np.isfinite(out).any() else 0.02
    med = med if np.isfinite(med) and med > 1e-6 else 0.02
    out = np.where((out > 1e-6) & (out < 0.5), out, med)
    out = np.clip(out, 1e-4, 0.5)
    return out


def build_policy_features(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    n = len(df)
    score = np.asarray(df.get("confidence", df.get("score", pd.Series(np.zeros(n)))).values, dtype=float)

    u_hat = _pick_first(df, ("u_hat", "oof_u_hat", "u_policy_net", "u_policy"))
    if u_hat is None:
        u_hat = score.copy()
    mae_hat = _pick_first(df, ("mae_hat", "oof_log_mae_q70_hat", "mae_pct", "mae_ret"))
    if mae_hat is None:
        mae_hat = np.abs(score)
    mfe_hat = _pick_first(df, ("mfe_hat", "oof_log_mfe_hat", "mfe_pct", "mfe_ret"))
    if mfe_hat is None:
        mfe_hat = np.maximum(score, 0.0)
    dur_hat = _pick_first(df, ("dur_hat", "oof_log_dur_hat", "duration", "bars_to_mfe"))
    if dur_hat is None:
        dur_hat = np.zeros(n, dtype=float)

    u_hat_z = _zscore(u_hat)
    mae_hat_z = _zscore(mae_hat)
    mfe_hat_z = _zscore(mfe_hat)
    dur_hat_z = _zscore(dur_hat)

    signal_px = np.asarray(
        df.get("signal_px", df.get("entry_price", df.get("entry_px", pd.Series(np.ones(n))))).values,
        dtype=float,
    )
    signal_px = np.where(np.isfinite(signal_px) & (signal_px > 0), signal_px, 1.0)
    atr = _safe_atr_from_trades(df)

    out = pd.DataFrame(
        {
            "u_hat": np.asarray(u_hat, dtype=float),
            "u_hat_z": np.asarray(u_hat_z, dtype=float),
            "mae_hat": np.asarray(mae_hat, dtype=float),
            "mae_hat_z": np.asarray(mae_hat_z, dtype=float),
            "mfe_hat": np.asarray(mfe_hat, dtype=float),
            "mfe_hat_z": np.asarray(mfe_hat_z, dtype=float),
            "dur_hat": np.asarray(dur_hat, dtype=float),
            "dur_hat_z": np.asarray(dur_hat_z, dtype=float),
            "signal_px": signal_px,
            "atr_policy": atr,
        },
        index=df.index,
    )
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_fill_model(
    trades: pd.DataFrame,
    features: pd.DataFrame,
    cfg: EntryOffsetConfig | None = None,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    cfg = cfg or EntryOffsetConfig()
    n = len(trades)
    if n == 0:
        return {
            "alpha0": cfg.alpha0_default,
            "alpha_u": cfg.alpha_u_default,
            "alpha_mae": cfg.alpha_mae_default,
            "beta_delta": cfg.beta_default,
        }, {"source_quality": "empty"}

    y_fill = np.asarray(
        (trades.get("filled_via_limit", pd.Series(np.zeros(n))).astype(bool).values)
        | (trades.get("reason", pd.Series([""] * n)).astype(str).values != "limit_not_filled"),
        dtype=float,
    )

    if "signal_px" in trades.columns:
        signal_px = np.asarray(trades["signal_px"].values, dtype=float)
    else:
        signal_px = np.asarray(
            trades.get("entry_price", trades.get("entry_px", pd.Series(np.ones(n)))).values,
            dtype=float,
        )
    entry_px = np.asarray(
        trades.get("entry_price", trades.get("entry_px", pd.Series(np.ones(n)))).values,
        dtype=float,
    )
    signal_px = np.where(np.isfinite(signal_px) & (signal_px > 0), signal_px, entry_px)
    atr = np.asarray(features["atr_policy"].values, dtype=float)
    delta_obs = np.abs(signal_px - entry_px) / np.maximum(signal_px * atr, 1e-6)

    if np.nanstd(delta_obs) < 1e-6 or np.unique(np.round(delta_obs, 6)).size < 2:
        return {
            "alpha0": cfg.alpha0_default,
            "alpha_u": cfg.alpha_u_default,
            "alpha_mae": cfg.alpha_mae_default,
            "beta_delta": cfg.beta_default,
        }, {"source_quality": "fallback_no_delta_variation"}

    x_u = np.asarray(features["u_hat_z"].values, dtype=float)
    x_mae = np.asarray(features["mae_hat_z"].values, dtype=float)
    x_delta = np.asarray(delta_obs, dtype=float)
    X = np.column_stack([np.ones(n), x_u, -x_mae, -x_delta])
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y_fill)
    X = X[mask]
    y = y_fill[mask]
    if len(X) < 50:
        return {
            "alpha0": cfg.alpha0_default,
            "alpha_u": cfg.alpha_u_default,
            "alpha_mae": cfg.alpha_mae_default,
            "beta_delta": cfg.beta_default,
        }, {"source_quality": "fallback_insufficient_samples"}

    w = np.array([cfg.alpha0_default, cfg.alpha_u_default, cfg.alpha_mae_default, cfg.beta_default], dtype=float)
    lr = 0.05
    for _ in range(400):
        p = _sigmoid(X @ w)
        grad = (X.T @ (p - y)) / max(len(y), 1)
        w = w - lr * grad
        w[3] = np.clip(w[3], 0.02, 5.0)
        w[2] = np.clip(w[2], 0.0, 3.0)
        w[1] = np.clip(w[1], -3.0, 3.0)
        w[0] = np.clip(w[0], -4.0, 4.0)

    return {
        "alpha0": float(w[0]),
        "alpha_u": float(w[1]),
        "alpha_mae": float(w[2]),
        "beta_delta": float(w[3]),
    }, {"source_quality": "fitted"}


def choose_entry_offsets(
    features: pd.DataFrame,
    model_params: Dict[str, float],
    cfg: EntryOffsetConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or EntryOffsetConfig()
    df = features.copy()
    u = np.asarray(df["u_hat_z"].values, dtype=float)
    mae = np.asarray(df["mae_hat_z"].values, dtype=float)
    atr = np.asarray(df["atr_policy"].values, dtype=float)
    signal_px = np.asarray(df["signal_px"].values, dtype=float)
    delta_grid = np.asarray(cfg.delta_atr_grid, dtype=float)

    alpha0 = float(model_params.get("alpha0", cfg.alpha0_default))
    alpha_u = float(model_params.get("alpha_u", cfg.alpha_u_default))
    alpha_mae = float(model_params.get("alpha_mae", cfg.alpha_mae_default))
    beta = float(model_params.get("beta_delta", cfg.beta_default))
    n = len(df)

    best_idx = np.zeros(len(df), dtype=int)
    best_eu = np.full(len(df), -np.inf, dtype=float)
    best_pf = np.zeros(len(df), dtype=float)
    for i, d in enumerate(delta_grid):
        pfill = _sigmoid(alpha0 + alpha_u * u - alpha_mae * mae - beta * d)
        eu = pfill * (cfg.a * u + d - cfg.lambda_risk * mae - cfg.c_atr)
        mask = eu > best_eu
        best_eu[mask] = eu[mask]
        best_idx[mask] = i
        best_pf[mask] = pfill[mask]

    delta_policy = np.asarray(delta_grid[best_idx], dtype=float)
    offset_policy_bps = np.clip((delta_policy * atr) * 10000.0, 0.0, 1000.0)

    # Estimator mode: MAE/MFE-derived offset in bps.
    def _to_frac(v: np.ndarray, kind: str) -> np.ndarray:
        x = np.asarray(v, dtype=float)
        if kind == "mae":
            if np.nanmedian(np.abs(x[np.isfinite(x)])) < 0.20:
                y = np.expm1(np.clip(x, -20, 20))
            else:
                y = np.clip(np.abs(x), 0.0, None)
        elif kind == "mfe":
            if np.nanmedian(np.abs(x[np.isfinite(x)])) < 0.20:
                y = np.expm1(np.clip(x, -20, 20))
            else:
                y = np.clip(np.abs(x), 0.0, None)
        else:
            y = x
        y = np.where(np.isfinite(y), y, 0.0)
        return np.clip(y, 0.0, 1.0)

    mae_frac = _to_frac(np.asarray(df["mae_hat"].values, dtype=float), "mae")
    mfe_frac = _to_frac(np.asarray(df["mfe_hat"].values, dtype=float), "mfe")
    u_raw = np.asarray(df["u_hat"].values, dtype=float)
    conf = 1.0 / (1.0 + np.exp(-np.clip(np.abs(u), -10.0, 10.0)))
    offset_est_bps = np.zeros(n, dtype=float)
    for i in range(n):
        offset_est_bps[i] = float(
            estimate_entry_limit_offset(
                mae_hat=float(mae_frac[i]),
                mfe_hat=float(mfe_frac[i]),
                u_hat=float(u_raw[i] if np.isfinite(u_raw[i]) else 0.0),
                confidence=float(conf[i]),
            )
        )
    offset_est_bps = np.where(np.isfinite(offset_est_bps), offset_est_bps, 0.0)
    offset_est_bps = np.clip(offset_est_bps, 0.0, 1000.0)

    def _eval_candidate(offset_bps: np.ndarray) -> Dict[str, np.ndarray | float]:
        off = np.asarray(offset_bps, dtype=float)
        off = np.clip(np.where(np.isfinite(off), off, 0.0), 0.0, 1000.0)
        delta = np.clip((off / 10000.0) / np.maximum(atr, 1e-6), 0.0, cfg.delta_atr_max)
        pfill = _sigmoid(alpha0 + alpha_u * u - alpha_mae * mae - beta * delta)
        eu = pfill * (cfg.a * u + delta - cfg.lambda_risk * mae - cfg.c_atr)
        place = eu >= float(cfg.min_expected_utility)
        split = int(np.floor((1.0 - float(np.clip(cfg.offset_engine_oos_frac, 0.05, 0.95))) * n))
        split = int(np.clip(split, 1, max(n - 1, 1)))
        idx = slice(split, n) if n >= 2 else slice(0, n)
        mean_eu = float(np.nanmean(eu[idx])) if n > 0 else -np.inf
        place_rate = float(np.mean(place[idx])) if n > 0 else 0.0
        score = float(mean_eu + 0.02 * place_rate)
        return {
            "offset_bps": off,
            "delta": delta,
            "pfill": pfill,
            "eu": eu,
            "place": place,
            "score": score,
            "mean_eu": mean_eu,
            "place_rate": place_rate,
        }

    candidates = []
    for mode in cfg.offset_engine_modes:
        m = str(mode).lower()
        if m == "policy_only":
            ev = _eval_candidate(offset_policy_bps)
            candidates.append({"mode": m, "lambda": 0.0, "eval": ev})
        elif m == "estimator_only":
            ev = _eval_candidate(offset_est_bps)
            candidates.append({"mode": m, "lambda": 1.0, "eval": ev})
        elif m == "blended":
            for lam in cfg.offset_blend_lambdas:
                _lam = float(np.clip(lam, 0.0, 1.0))
                off = (1.0 - _lam) * offset_policy_bps + _lam * offset_est_bps
                ev = _eval_candidate(off)
                candidates.append({"mode": m, "lambda": _lam, "eval": ev})
    if not candidates:
        candidates = [{"mode": "policy_only", "lambda": 0.0, "eval": _eval_candidate(offset_policy_bps)}]
    best = max(candidates, key=lambda c: float(c["eval"]["score"]))

    offset_star_bps = np.asarray(best["eval"]["offset_bps"], dtype=float)
    delta_star = np.asarray(best["eval"]["delta"], dtype=float)
    place = np.asarray(best["eval"]["place"], dtype=bool)
    best_eu = np.asarray(best["eval"]["eu"], dtype=float)
    best_pf = np.asarray(best["eval"]["pfill"], dtype=float)
    delta_price = (offset_star_bps / 10000.0) * signal_px

    out = df.copy()
    out["delta_atr_star"] = delta_star
    out["delta_price_star"] = delta_price
    out["p_fill_star"] = best_pf
    out["eu_star"] = best_eu
    out["place_order"] = place.astype(bool)
    out["limit_offset_bps_dynamic"] = offset_star_bps
    out["limit_offset_bps_policy"] = offset_policy_bps
    out["limit_offset_bps_estimator"] = offset_est_bps
    out["offset_engine_mode"] = str(best["mode"])
    out["offset_engine_lambda"] = float(best["lambda"])
    out["entry_px_fill"] = np.maximum(signal_px - delta_price, EPS)
    out["delta_atr_grid"] = [delta_grid.tolist()] * len(out)
    out.attrs["offset_engine_mode"] = str(best["mode"])
    out.attrs["offset_engine_lambda"] = float(best["lambda"])
    out.attrs["offset_engine_oos_score"] = float(best["eval"]["score"])
    out.attrs["offset_engine_oos_mean_eu"] = float(best["eval"]["mean_eu"])
    out.attrs["offset_engine_oos_place_rate"] = float(best["eval"]["place_rate"])
    return out


def apply_effective_policy_params(
    trades: pd.DataFrame,
    policy_df: pd.DataFrame,
    base_params: Dict[str, float] | None = None,
    cfg: EntryOffsetConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or EntryOffsetConfig()
    base_params = base_params or {}
    out = trades.copy()
    p = policy_df.reindex(out.index)
    out = out.join(
        p[
            [
                "u_hat",
                "u_hat_z",
                "mae_hat",
                "mae_hat_z",
                "mfe_hat",
                "mfe_hat_z",
                "dur_hat",
                "dur_hat_z",
                "signal_px",
                "entry_px_fill",
                "delta_atr_star",
                "delta_price_star",
                "p_fill_star",
                "eu_star",
                "place_order",
                "atr_policy",
            ]
        ],
        how="left",
        rsuffix="_pol",
    )

    atr = np.asarray(out["atr_policy"].values, dtype=float)
    mae_atr = np.maximum(np.asarray(out["mae_hat_z"].values, dtype=float), 0.0)
    u = np.asarray(out["u_hat_z"].values, dtype=float)
    d = np.asarray(out["delta_atr_star"].values, dtype=float)
    mfe_z = np.asarray(out["mfe_hat_z"].values, dtype=float)
    dur_z = np.asarray(out["dur_hat_z"].values, dtype=float)

    stop_factor = np.clip(1.0 - cfg.eta_stop * d, 0.5, 1.0)
    sl_distance_atr = cfg.q_sl * np.maximum(mae_atr, 0.0) * stop_factor
    tp_core = np.maximum(cfg.a * u + d, 0.0)
    tp_distance_atr = cfg.r_tp * tp_core

    trail_base = float(base_params.get("trail_mult", out.get("trail_mult", pd.Series(np.full(len(out), 0.25))).median()))
    giveback_base = float(base_params.get("giveback_pct", 0.005))
    lock_amt_base = float(base_params.get("profit_lock_amount", 0.003))
    kill_c_base = float(base_params.get("kill_c", 0.005))
    hold_h_base = float(base_params.get("max_hold_hours", 24.0))

    trail_mult_eff = np.clip(
        trail_base * (1.0 + cfg.trail_mult_k_delta * d + cfg.trail_mult_k_mfe * mfe_z),
        0.05,
        1.2,
    )
    giveback_pct_eff = np.clip(giveback_base * (1.0 + cfg.giveback_k_delta * d + cfg.giveback_k_dur * dur_z), 0.001, 0.05)
    lock_amt_eff = np.clip(lock_amt_base * (1.0 + cfg.lock_amt_k_u * u), 0.0005, 0.05)
    kill_c_eff = np.clip(kill_c_base * (1.0 + cfg.kill_c_k_mae * mae_atr), 0.0001, 0.05)
    hold_h_eff = np.clip(hold_h_base * (1.0 + cfg.hold_h_k_dur * dur_z), 4.0, 72.0)

    out["stop_factor_eff"] = stop_factor
    out["sl_distance_atr_eff"] = sl_distance_atr
    out["tp_distance_atr_eff"] = tp_distance_atr
    out["trail_mult_eff"] = trail_mult_eff
    out["giveback_pct_eff"] = giveback_pct_eff
    out["profit_lock_amount_eff"] = lock_amt_eff
    out["kill_c_eff"] = kill_c_eff
    out["max_hold_hours_eff"] = hold_h_eff

    if "entry_price" in out.columns:
        out["entry_price"] = np.where(np.isfinite(out["entry_px_fill"]), out["entry_px_fill"], out["entry_price"])
    elif "entry_px" in out.columns:
        out["entry_px"] = np.where(np.isfinite(out["entry_px_fill"]), out["entry_px_fill"], out["entry_px"])

    return out


def build_entry_policy_payload(
    model_params: Dict[str, float],
    cfg: EntryOffsetConfig,
    fallback_meta: Dict[str, str] | None = None,
    offset_engine_meta: Dict[str, float | str] | None = None,
) -> Dict[str, object]:
    fallback_meta = fallback_meta or {}
    offset_engine_meta = offset_engine_meta or {}
    return {
        "model": {
            "alpha0": float(model_params.get("alpha0", cfg.alpha0_default)),
            "alpha_u": float(model_params.get("alpha_u", cfg.alpha_u_default)),
            "alpha_mae": float(model_params.get("alpha_mae", cfg.alpha_mae_default)),
            "beta_delta": float(model_params.get("beta_delta", cfg.beta_default)),
        },
        "objective": {
            "a": float(cfg.a),
            "lambda_risk": float(cfg.lambda_risk),
            "c_atr": float(cfg.c_atr),
            "delta_atr_grid": [float(x) for x in cfg.delta_atr_grid.tolist()],
            "delta_atr_max": float(cfg.delta_atr_max),
            "min_expected_utility": float(cfg.min_expected_utility),
        },
        "adaptation": {
            "q_sl": float(cfg.q_sl),
            "eta_stop": float(cfg.eta_stop),
            "r_tp": float(cfg.r_tp),
            "trail_mult_k_delta": float(cfg.trail_mult_k_delta),
            "trail_mult_k_mfe": float(cfg.trail_mult_k_mfe),
            "giveback_k_delta": float(cfg.giveback_k_delta),
            "giveback_k_dur": float(cfg.giveback_k_dur),
            "lock_amt_k_u": float(cfg.lock_amt_k_u),
            "kill_c_k_mae": float(cfg.kill_c_k_mae),
            "hold_h_k_dur": float(cfg.hold_h_k_dur),
        },
        "fallback": {
            "u_zscore_used": True,
            "mae_zscore_used": True,
            "source_quality": str(fallback_meta.get("source_quality", "unknown")),
        },
        "offset_engine": {
            "mode": str(offset_engine_meta.get("mode", "policy_only")),
            "lambda": float(offset_engine_meta.get("lambda", 0.0)),
            "oos_score": float(offset_engine_meta.get("oos_score", 0.0)),
            "oos_mean_eu": float(offset_engine_meta.get("oos_mean_eu", 0.0)),
            "oos_place_rate": float(offset_engine_meta.get("oos_place_rate", 0.0)),
        },
    }
