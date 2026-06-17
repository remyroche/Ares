from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_float_array(values: Any, n: int, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name}: expected {n} rows, got {arr.shape[0]}")
    return arr


def _finite_or_zero(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(arr), arr, np.float32(0.0)).astype(np.float32)


def _array_summary(values: Any) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    finite = np.isfinite(arr)
    n_finite = int(np.sum(finite))
    out: dict[str, float | int] = {
        "n": n,
        "n_finite": n_finite,
        "finite_fraction": float(n_finite / max(n, 1)),
    }
    if n_finite == 0:
        out.update(
            {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "p01": float("nan"),
                "p05": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            }
        )
        return out
    vals = arr[finite]
    p01, p05, p50, p95, p99 = np.percentile(vals, [1.0, 5.0, 50.0, 95.0, 99.0])
    out.update(
        {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "p01": float(p01),
            "p05": float(p05),
            "p50": float(p50),
            "p95": float(p95),
            "p99": float(p99),
            "max": float(np.max(vals)),
        }
    )
    return out


def summarize_target_audit(
    target_audit: dict[str, Any] | None,
) -> dict[str, dict[str, float | int]]:
    return {
        str(key): _array_summary(values)
        for key, values in dict(target_audit or {}).items()
    }


def _bucket_numeric_feature(values: Any, *, name: str) -> tuple[np.ndarray, int]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float32)
    finite = np.isfinite(arr)
    codes = np.zeros(arr.shape[0], dtype=np.int64)
    lowered = str(name).lower()
    if lowered.endswith("_pct") or lowered.endswith("_rank") or "percentile" in lowered:
        valid = np.clip(arr[finite], 0.0, 0.999999)
        codes[finite] = np.floor(valid * 5.0).astype(np.int64) + 1
        return codes, 6
    codes[finite] = np.digitize(arr[finite], [-1.0, 0.0, 1.0]).astype(np.int64) + 1
    return codes, 5


def _resolve_cluster_labels(
    df_local: pd.DataFrame,
    cfg: dict[str, Any],
    n: int,
) -> tuple[np.ndarray | None, str, list[str], str | None]:
    for candidate in list(cfg.get("rolling_alpha_target_cluster_columns", []) or []):
        if candidate in df_local.columns:
            return (
                df_local[candidate].astype(str).to_numpy(copy=False),
                "explicit_column",
                [str(candidate)],
                str(candidate),
            )

    candidates = list(
        cfg.get("rolling_alpha_target_cluster_feature_columns", []) or []
    )
    max_cols = max(
        0, int(cfg.get("rolling_alpha_target_cluster_feature_max_columns", 3))
    )
    selected = [str(c) for c in candidates if str(c) in df_local.columns][:max_cols]
    if not selected:
        return None, "none", [], None

    cluster_code = np.zeros(n, dtype=np.int64)
    multiplier = np.int64(1)
    for col in selected:
        codes, n_bins = _bucket_numeric_feature(df_local[col].values, name=col)
        cluster_code += codes.astype(np.int64, copy=False) * multiplier
        multiplier *= np.int64(max(int(n_bins), 1))
    return cluster_code, "feature_buckets", selected, None


def _leave_one_out_group_mean(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
) -> np.ndarray:
    n = len(frame)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    grouped = frame.groupby(group_cols, sort=False)[value_col]
    group_sum = grouped.transform("sum").to_numpy(dtype=np.float32, copy=False)
    group_count = grouped.transform("count").to_numpy(dtype=np.float32, copy=False)
    values = frame[value_col].to_numpy(dtype=np.float32, copy=False)
    out = np.zeros(n, dtype=np.float32)
    mask = group_count > 1.0
    out[mask] = (group_sum[mask] - values[mask]) / (group_count[mask] - 1.0)
    return _finite_or_zero(out)


def _rolling_beta_by_symbol(
    *,
    y: np.ndarray,
    factor: np.ndarray,
    symbol: np.ndarray,
    ts: np.ndarray,
    window: int,
    min_periods: int,
    default_beta: float,
    var_floor: float,
) -> np.ndarray:
    n = len(y)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    work = pd.DataFrame(
        {
            "_row": np.arange(n, dtype=np.int64),
            "symbol": symbol,
            "ts": ts,
            "y": np.asarray(y, dtype=np.float32),
            "factor": np.asarray(factor, dtype=np.float32),
        }
    ).sort_values(["symbol", "ts", "_row"], kind="mergesort")
    work = work.reset_index(drop=True)

    invalid = ~np.isfinite(work["y"].to_numpy()) | ~np.isfinite(
        work["factor"].to_numpy()
    )
    if bool(np.any(invalid)):
        work.loc[invalid, ["y", "factor"]] = np.nan

    grouped = work.groupby("symbol", sort=False)
    work["y_lag"] = grouped["y"].shift(1)
    work["factor_lag"] = grouped["factor"].shift(1)
    work["yf_lag"] = work["y_lag"] * work["factor_lag"]
    work["f2_lag"] = work["factor_lag"] * work["factor_lag"]

    roll = work.groupby("symbol", sort=False).rolling(
        window=int(window),
        min_periods=int(min_periods),
    )
    mean_y = roll["y_lag"].mean().reset_index(level=0, drop=True).to_numpy()
    mean_f = roll["factor_lag"].mean().reset_index(level=0, drop=True).to_numpy()
    mean_yf = roll["yf_lag"].mean().reset_index(level=0, drop=True).to_numpy()
    mean_f2 = roll["f2_lag"].mean().reset_index(level=0, drop=True).to_numpy()

    cov = mean_yf - mean_y * mean_f
    var = mean_f2 - mean_f * mean_f
    beta_sorted = np.full(n, float(default_beta), dtype=np.float32)
    valid = np.isfinite(cov) & np.isfinite(var) & (var > float(var_floor))
    beta_sorted[valid] = (cov[valid] / var[valid]).astype(np.float32)
    beta_sorted = np.where(
        np.isfinite(beta_sorted), beta_sorted, np.float32(default_beta)
    ).astype(np.float32)

    out = np.full(n, float(default_beta), dtype=np.float32)
    out[work["_row"].to_numpy(dtype=np.int64)] = beta_sorted
    return out


def _rolling_abs_scale_by_symbol(
    *,
    values: np.ndarray,
    symbol: np.ndarray,
    ts: np.ndarray,
    window: int,
    min_periods: int,
    floor: float,
) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    abs_values = np.abs(np.asarray(values, dtype=np.float32))
    fallback = float(floor)

    work = pd.DataFrame(
        {
            "_row": np.arange(n, dtype=np.int64),
            "symbol": symbol,
            "ts": ts,
            "abs_value": abs_values,
        }
    ).sort_values(["symbol", "ts", "_row"], kind="mergesort")
    work = work.reset_index(drop=True)
    work.loc[~np.isfinite(work["abs_value"].to_numpy()), "abs_value"] = np.nan
    grouped = work.groupby("symbol", sort=False)
    work["abs_lag"] = grouped["abs_value"].shift(1)
    scale_sorted = (
        work.groupby("symbol", sort=False)["abs_lag"]
        .rolling(window=int(window), min_periods=int(min_periods))
        .median()
        .reset_index(level=0, drop=True)
        .to_numpy(dtype=np.float32, copy=False)
    )
    scale_sorted = 1.4826 * scale_sorted
    scale_sorted = np.where(
        np.isfinite(scale_sorted), scale_sorted, np.float32(fallback)
    )
    scale_sorted = np.maximum(scale_sorted, np.float32(floor)).astype(np.float32)

    out = np.full(n, fallback, dtype=np.float32)
    out[work["_row"].to_numpy(dtype=np.int64)] = scale_sorted
    return out


def _kalman_filter_by_symbol(
    *,
    values: np.ndarray,
    symbol: np.ndarray,
    ts: np.ndarray,
    process_var: float,
    obs_var: float,
) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    work = pd.DataFrame(
        {
            "_row": np.arange(n, dtype=np.int64),
            "symbol": symbol,
            "ts": ts,
            "obs": np.asarray(values, dtype=np.float32),
        }
    ).sort_values(["symbol", "ts", "_row"], kind="mergesort")
    work = work.reset_index(drop=True)
    filtered = np.full(n, np.nan, dtype=np.float32)
    q = max(float(process_var), 1e-12)
    r = max(float(obs_var), 1e-12)
    for positions in work.groupby("symbol", sort=False).indices.values():
        x = np.nan
        p = 1.0
        for pos in positions:
            obs = float(work.at[int(pos), "obs"])
            if not np.isfinite(obs):
                filtered[int(pos)] = x if np.isfinite(x) else np.nan
                continue
            if not np.isfinite(x):
                x = obs
                p = r
            else:
                p_pred = p + q
                gain = p_pred / (p_pred + r)
                x = x + gain * (obs - x)
                p = (1.0 - gain) * p_pred
            filtered[int(pos)] = np.float32(x)

    out = np.asarray(values, dtype=np.float32).copy()
    valid = np.isfinite(filtered)
    out[work.loc[valid, "_row"].to_numpy(dtype=np.int64)] = filtered[valid]
    return _finite_or_zero(out)


def build_gross_residual_alpha_target(
    df_local: pd.DataFrame,
    *,
    side: str,
    y_ret: np.ndarray | None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(cfg or {})
    n = len(df_local)
    if "__ts__" not in df_local.columns or "__symbol__" not in df_local.columns:
        raise KeyError(
            "gross residual alpha target requires __ts__ and __symbol__ columns"
        )
    if y_ret is None:
        if "__y_ret__" not in df_local.columns:
            raise KeyError("gross residual alpha target requires __y_ret__")
        y_ret_arr = _as_float_array(df_local["__y_ret__"].values, n, name="__y_ret__")
    else:
        y_ret_arr = _as_float_array(y_ret, n, name="y_ret")

    side_sign = np.float32(-1.0 if str(side).lower() == "short" else 1.0)
    favorable_return = _finite_or_zero(side_sign * y_ret_arr)

    work = pd.DataFrame(
        {
            "__ts__": df_local["__ts__"].values,
            "__symbol__": df_local["__symbol__"].astype(str).values,
            "__favorable_return__": favorable_return,
        }
    )
    market_factor = _leave_one_out_group_mean(
        work,
        group_cols=["__ts__"],
        value_col="__favorable_return__",
    )
    market_beta = _rolling_beta_by_symbol(
        y=favorable_return,
        factor=market_factor,
        symbol=work["__symbol__"].values,
        ts=work["__ts__"].values,
        window=int(cfg.get("rolling_alpha_target_market_beta_window", 720)),
        min_periods=int(cfg.get("rolling_alpha_target_beta_min_periods", 168)),
        default_beta=float(cfg.get("rolling_alpha_target_default_market_beta", 1.0)),
        var_floor=float(cfg.get("rolling_alpha_target_beta_var_floor", 1e-10)),
    )
    market_component = (market_beta * market_factor).astype(np.float32)
    market_residual = (favorable_return - market_component).astype(np.float32)

    cluster_factor = np.zeros(n, dtype=np.float32)
    cluster_component = np.zeros(n, dtype=np.float32)
    cluster_beta = np.zeros(n, dtype=np.float32)
    (
        cluster_labels,
        cluster_source,
        cluster_feature_cols,
        cluster_col_used,
    ) = _resolve_cluster_labels(df_local, cfg, n)
    if cluster_labels is not None:
        work["__cluster__"] = cluster_labels
        work["__market_residual__"] = market_residual
        cluster_factor = _leave_one_out_group_mean(
            work,
            group_cols=["__ts__", "__cluster__"],
            value_col="__market_residual__",
        )
        cluster_beta = _rolling_beta_by_symbol(
            y=market_residual,
            factor=cluster_factor,
            symbol=work["__symbol__"].values,
            ts=work["__ts__"].values,
            window=int(cfg.get("rolling_alpha_target_cluster_beta_window", 720)),
            min_periods=int(cfg.get("rolling_alpha_target_beta_min_periods", 168)),
            default_beta=float(
                cfg.get("rolling_alpha_target_default_cluster_beta", 1.0)
            ),
            var_floor=float(cfg.get("rolling_alpha_target_beta_var_floor", 1e-10)),
        )
        cluster_component = (cluster_beta * cluster_factor).astype(np.float32)

    raw_alpha = (market_residual - cluster_component).astype(np.float32)
    kalman_alpha = raw_alpha.copy()
    target_raw = raw_alpha.copy()
    if bool(cfg.get("rolling_alpha_target_kalman_enabled", False)):
        kalman_alpha = _kalman_filter_by_symbol(
            values=raw_alpha,
            symbol=work["__symbol__"].values,
            ts=work["__ts__"].values,
            process_var=float(cfg.get("rolling_alpha_target_kalman_process_var", 1e-6)),
            obs_var=float(cfg.get("rolling_alpha_target_kalman_obs_var", 1e-4)),
        )
        blend = float(cfg.get("rolling_alpha_target_kalman_blend", 0.35))
        blend = float(np.clip(blend, 0.0, 1.0))
        target_raw = ((1.0 - blend) * raw_alpha + blend * kalman_alpha).astype(
            np.float32
        )

    scale = _rolling_abs_scale_by_symbol(
        values=raw_alpha,
        symbol=work["__symbol__"].values,
        ts=work["__ts__"].values,
        window=int(cfg.get("rolling_alpha_target_scale_window", 336)),
        min_periods=int(cfg.get("rolling_alpha_target_scale_min_periods", 48)),
        floor=float(cfg.get("rolling_alpha_target_scale_floor", 1e-4)),
    )

    transform = str(cfg.get("rolling_alpha_target_transform", "asinh_scaled")).lower()
    if transform in {"asinh_scaled", "arcsinh_scaled"}:
        target = np.arcsinh(target_raw / scale).astype(np.float32)
        transform_name = "asinh_scaled"
    elif transform in {"raw", "none", "identity"}:
        target = target_raw.astype(np.float32)
        transform_name = "raw"
    else:
        raise ValueError(f"Unknown rolling alpha target transform: {transform}")

    clip_abs = float(cfg.get("rolling_alpha_target_clip_abs", 20.0))
    if np.isfinite(clip_abs) and clip_abs > 0.0:
        target = np.clip(target, -clip_abs, clip_abs).astype(np.float32)

    horizon = int(cfg.get("rolling_alpha_target_horizon_hours", 5))
    target_name = f"{transform_name}_gross_residual_alpha_{horizon}h"
    if bool(cfg.get("rolling_alpha_target_kalman_enabled", False)):
        target_name += "_partial_kalman"
    target_audit = {
        f"target_gross_residual_alpha_{horizon}h": _finite_or_zero(target),
        f"raw_gross_residual_alpha_{horizon}h": _finite_or_zero(raw_alpha),
        f"market_factor_component_{horizon}h": _finite_or_zero(market_component),
        f"cluster_factor_component_{horizon}h": _finite_or_zero(cluster_component),
        f"market_beta_{horizon}h": _finite_or_zero(market_beta),
        f"cluster_beta_{horizon}h": _finite_or_zero(cluster_beta),
        f"gross_residual_alpha_scale_{horizon}h": _finite_or_zero(scale),
        f"kalman_gross_residual_alpha_{horizon}h": _finite_or_zero(kalman_alpha),
    }
    target_diagnostics = {
        "cluster_source": cluster_source,
        "cluster_columns": list(cluster_feature_cols),
        "explicit_cluster_column": cluster_col_used,
        "cluster_count": int(
            pd.Series(cluster_labels).nunique(dropna=True)
            if cluster_labels is not None
            else 0
        ),
        "target_transform": transform_name,
        "kalman_enabled": bool(cfg.get("rolling_alpha_target_kalman_enabled", False)),
    }

    return {
        "target": _finite_or_zero(target),
        "side_adjusted_return": favorable_return.astype(np.float32, copy=False),
        "raw_vol_norm_return": (target_raw / scale).astype(np.float32, copy=False),
        "residualized_return": raw_alpha.astype(np.float32, copy=False),
        "vol_source": "rolling_median_abs_gross_residual_alpha",
        "residualizer": None,
        "residualization_status": "gross_residual_alpha_market_cluster",
        "nuisance_columns": list(cluster_feature_cols),
        "target_name": target_name,
        "target_audit": target_audit,
        "target_audit_summary": summarize_target_audit(target_audit),
        "target_diagnostics": target_diagnostics,
    }
