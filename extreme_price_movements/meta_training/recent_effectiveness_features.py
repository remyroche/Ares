from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2 or y.size < 2:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float32)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float32)
    if np.nanstd(rx) <= 0.0 or np.nanstd(ry) <= 0.0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _ece_from_hist(pred: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    p = pred.astype(np.float32, copy=False)
    a = actual.astype(np.float32, copy=False)
    ok = np.isfinite(p) & np.isfinite(a)
    if int(ok.sum()) == 0:
        return np.nan
    p = p[ok]
    a = a[ok]
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    ece = 0.0
    n = float(p.size)
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n_bin = int(mask.sum())
        if n_bin > 0:
            ece += (n_bin / n) * abs(
                float(np.nanmean(p[mask])) - float(np.nanmean(a[mask]))
            )
    return float(ece)


def _window_stats(
    hist: pd.DataFrame,
    *,
    score_col: str,
    prob_col: str,
    y_col: str,
    ret_col: str,
    top_frac: float,
    min_samples: int,
    min_top_samples: int,
) -> dict[str, float]:
    out = {
        "n_samples": float(len(hist)),
        "n_valid": 0.0,
        "rolling_ic": np.nan,
        "model_ece": np.nan,
        "confidence_surprise": np.nan,
        "top_hit_rate": np.nan,
        "top_calibration_error": np.nan,
        "abs_top_calibration_error": np.nan,
        "top_ev": np.nan,
        "n_top": np.nan,
    }
    if len(hist) < min_samples:
        return out
    s = hist[score_col].to_numpy(dtype=np.float32)
    r = hist[ret_col].to_numpy(dtype=np.float32)
    p = hist[prob_col].to_numpy(dtype=np.float32)
    y = hist[y_col].to_numpy(dtype=np.float32)
    valid_all = np.isfinite(s) & np.isfinite(r) & np.isfinite(p) & np.isfinite(y)
    out["n_valid"] = float(int(valid_all.sum()))
    out["rolling_ic"] = _spearman_corr(s, r)
    out["model_ece"] = _ece_from_hist(p, y)
    pred_label = (p >= 0.5).astype(np.float32)
    pred_conf = np.maximum(p, 1.0 - p)
    hit = (pred_label == (y >= 0.5)).astype(np.float32)
    out["confidence_surprise"] = float(np.nanmean(pred_conf - hit))
    valid = np.isfinite(s)
    if int(valid.sum()) < min_top_samples:
        return out
    thr = np.nanquantile(s[valid], 1.0 - top_frac)
    top = valid & (s >= thr)
    top_ok = top & np.isfinite(p) & np.isfinite(y) & np.isfinite(r)
    out["n_top"] = float(int(top_ok.sum()))
    if int(top_ok.sum()) >= min_top_samples:
        out["top_hit_rate"] = float(np.nanmean(y[top_ok]))
        out["top_calibration_error"] = float(
            np.nanmean(p[top_ok]) - np.nanmean(y[top_ok])
        )
        out["abs_top_calibration_error"] = abs(out["top_calibration_error"])
        out["top_ev"] = float(np.nanmean(r[top_ok]))
    return out


def _compute_scope_timeseries(
    g: pd.DataFrame,
    *,
    ts_col: str,
    label_available_ts_col: str,
    windows: tuple[str, ...],
    metric_fn,
) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    g_sorted = g.sort_values(ts_col)
    ts_values = pd.to_datetime(g_sorted[ts_col], errors="coerce")
    label_values = pd.to_datetime(g_sorted[label_available_ts_col], errors="coerce")
    ts_ns = ts_values.astype("int64").to_numpy(copy=False)
    label_ns = label_values.astype("int64").to_numpy(copy=False)
    valid_label_ns = label_ns[label_ns != pd.NaT.value]
    monotone_label = bool(np.all(np.diff(valid_label_ns) >= 0))
    ts_unique = pd.Index(ts_values.dropna().unique()).sort_values()
    for t in ts_unique:
        rec: dict[str, float | pd.Timestamp] = {ts_col: t}
        t_ns = pd.Timestamp(t).value
        for win in windows:
            lo_ns = (pd.Timestamp(t) - pd.Timedelta(win)).value
            if monotone_label:
                lo = int(np.searchsorted(ts_ns, lo_ns, side="left"))
                hi = int(np.searchsorted(label_ns, t_ns, side="left"))
                hist = g_sorted.iloc[lo:hi] if hi > lo else g_sorted.iloc[:0]
            else:
                hist = g_sorted[
                    (label_values < t)
                    & (ts_values >= pd.Timestamp(t) - pd.Timedelta(win))
                ]
            vals = metric_fn(hist)
            for k, v in vals.items():
                rec[f"{k}_{win.lower()}"] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def _compute_scope_timeseries_vectorized(
    g: pd.DataFrame,
    *,
    ts_col: str,
    label_available_ts_col: str,
    windows: tuple[str, ...],
    score_col: str,
    prob_col: str,
    y_col: str,
    ret_col: str,
    top_frac: float,
    min_samples: int,
    min_top_samples: int,
    top_label: str,
) -> pd.DataFrame:
    """Precompute causal recent-effectiveness stats once per scope.

    Rows are indexed by label-availability time, then joined back to prediction
    timestamps with merge_asof. Time-window rolling operations use closed="left"
    so a label that becomes available exactly at t is not visible to the feature
    for a prediction at t, matching the strict ``label_available_ts < t`` rule.
    """
    hist = g[
        [
            label_available_ts_col,
            score_col,
            prob_col,
            y_col,
            ret_col,
        ]
    ].copy()
    hist[label_available_ts_col] = pd.to_datetime(
        hist[label_available_ts_col], errors="coerce"
    )
    hist = hist.dropna(subset=[label_available_ts_col]).sort_values(
        label_available_ts_col
    )
    if hist.empty:
        return pd.DataFrame(columns=[ts_col])

    for col in [score_col, prob_col, y_col, ret_col]:
        hist[col] = pd.to_numeric(hist[col], errors="coerce").astype(np.float32)

    valid_all = (
        np.isfinite(hist[score_col].to_numpy(dtype=np.float32, copy=False))
        & np.isfinite(hist[prob_col].to_numpy(dtype=np.float32, copy=False))
        & np.isfinite(hist[y_col].to_numpy(dtype=np.float32, copy=False))
        & np.isfinite(hist[ret_col].to_numpy(dtype=np.float32, copy=False))
    ).astype(np.float32)
    hist["_valid_all"] = valid_all
    hist["_cal_abs"] = np.abs(
        hist[prob_col].to_numpy(dtype=np.float32, copy=False)
        - hist[y_col].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32)
    pred_label = (
        hist[prob_col].to_numpy(dtype=np.float32, copy=False) >= 0.5
    ).astype(np.float32)
    pred_conf = np.maximum(
        hist[prob_col].to_numpy(dtype=np.float32, copy=False),
        1.0 - hist[prob_col].to_numpy(dtype=np.float32, copy=False),
    )
    hit = (
        pred_label
        == (hist[y_col].to_numpy(dtype=np.float32, copy=False) >= 0.5)
    ).astype(np.float32)
    hist["_confidence_surprise"] = (pred_conf - hit).astype(np.float32)
    idxed = hist.set_index(label_available_ts_col, drop=False)
    out = pd.DataFrame({ts_col: hist[label_available_ts_col].values}, index=hist.index)

    score = idxed[score_col]
    ret = idxed[ret_col]
    prob = idxed[prob_col]
    y = idxed[y_col]
    valid = idxed["_valid_all"]
    cal_abs = idxed["_cal_abs"]
    confidence_surprise = idxed["_confidence_surprise"]

    for win in windows:
        sfx = win.lower()
        roll_valid = valid.rolling(win, closed="left")
        n_valid = roll_valid.sum().astype(np.float32)
        n_samples = (
            pd.Series(np.ones(len(idxed), dtype=np.float32), index=idxed.index)
            .rolling(win, closed="left")
            .sum()
            .astype(np.float32)
        )

        score_mean = score.rolling(win, closed="left").mean()
        ret_mean = ret.rolling(win, closed="left").mean()
        score_ret_mean = (score * ret).rolling(win, closed="left").mean()
        score_std = score.rolling(win, closed="left").std(ddof=0)
        ret_std = ret.rolling(win, closed="left").std(ddof=0)
        corr = (score_ret_mean - score_mean * ret_mean) / (
            score_std * ret_std
        ).replace(0.0, np.nan)

        thr = score.rolling(win, closed="left").quantile(1.0 - top_frac)
        top = (score >= thr).astype(np.float32).where(np.isfinite(thr), 0.0)
        top_valid = (top * valid).astype(np.float32)
        n_top = top_valid.rolling(win, closed="left").sum().astype(np.float32)
        top_y_sum = (top_valid * y).rolling(win, closed="left").sum()
        top_p_sum = (top_valid * prob).rolling(win, closed="left").sum()
        top_ret_sum = (top_valid * ret).rolling(win, closed="left").sum()
        top_hit = top_y_sum / n_top.replace(0.0, np.nan)
        top_cal = (top_p_sum / n_top.replace(0.0, np.nan)) - top_hit
        top_ev = top_ret_sum / n_top.replace(0.0, np.nan)

        enough = n_valid >= float(min_samples)
        enough_top = n_top >= float(min_top_samples)
        out[f"n_samples_{sfx}"] = n_samples.to_numpy(dtype=np.float32)
        out[f"n_valid_{sfx}"] = n_valid.to_numpy(dtype=np.float32)
        out[f"rolling_ic_{sfx}"] = corr.where(enough).to_numpy(dtype=np.float32)
        out[f"model_ece_{sfx}"] = (
            cal_abs.rolling(win, closed="left").mean().where(enough).to_numpy(
                dtype=np.float32
            )
        )
        out[f"confidence_surprise_{sfx}"] = (
            confidence_surprise.rolling(win, closed="left")
            .mean()
            .where(enough)
            .to_numpy(dtype=np.float32)
        )
        out[f"{top_label}_hit_rate_{sfx}"] = top_hit.where(enough_top).to_numpy(
            dtype=np.float32
        )
        out[f"{top_label}_calibration_error_{sfx}"] = top_cal.where(
            enough_top
        ).to_numpy(dtype=np.float32)
        out[f"abs_{top_label}_calibration_error_{sfx}"] = np.abs(
            top_cal.where(enough_top).to_numpy(dtype=np.float32)
        )
        out[f"{top_label}_ev_{sfx}"] = top_ev.where(enough_top).to_numpy(
            dtype=np.float32
        )
        out[f"n_{top_label}_{sfx}"] = n_top.to_numpy(dtype=np.float32)

    return out.drop_duplicates(subset=[ts_col], keep="last").sort_values(ts_col)


def _add_scope_features(
    df: pd.DataFrame,
    *,
    scope_name: str,
    scope_cols: Iterable[str],
    windows: tuple[str, ...],
    ts_col: str,
    label_available_ts_col: str,
    score_col: str,
    prob_col: str,
    y_col: str,
    ret_col: str,
    top_frac: float,
    min_samples: int,
    min_top_samples: int,
) -> pd.DataFrame:
    top_label = f"top{int(round(top_frac * 100))}"
    metrics = [
        "rolling_ic",
        "model_ece",
        "confidence_surprise",
        f"{top_label}_hit_rate",
        f"{top_label}_calibration_error",
        f"abs_{top_label}_calibration_error",
        f"{top_label}_ev",
        "n_samples",
        f"n_{top_label}",
        "n_valid",
    ]
    init_cols: dict[str, object] = {}
    for win in windows:
        sfx = win.lower()
        for m in metrics:
            init_cols[f"recent_{scope_name}_{m}_{sfx}"] = np.full(
                len(df), np.nan, dtype=np.float32
            )
        init_cols[f"recent_{scope_name}_available_{sfx}"] = np.zeros(
            len(df), dtype=np.int8
        )
    if init_cols:
        df = pd.concat([df, pd.DataFrame(init_cols, index=df.index)], axis=1)

    grouped = (
        df.groupby(list(scope_cols), sort=False, dropna=False)
        if scope_cols
        else [(None, df)]
    )
    for _, g in grouped:

        def _scoped_window_stats(hist: pd.DataFrame) -> dict[str, float]:
            raw = _window_stats(
                hist,
                score_col=score_col,
                prob_col=prob_col,
                y_col=y_col,
                ret_col=ret_col,
                top_frac=top_frac,
                min_samples=min_samples,
                min_top_samples=min_top_samples,
            )
            return {
                "n_samples": raw["n_samples"],
                "n_valid": raw["n_valid"],
                "rolling_ic": raw["rolling_ic"],
                "model_ece": raw["model_ece"],
                "confidence_surprise": raw["confidence_surprise"],
                f"{top_label}_hit_rate": raw["top_hit_rate"],
                f"{top_label}_calibration_error": raw["top_calibration_error"],
                f"abs_{top_label}_calibration_error": raw["abs_top_calibration_error"],
                f"{top_label}_ev": raw["top_ev"],
                f"n_{top_label}": raw["n_top"],
            }

        try:
            stats_ts = _compute_scope_timeseries_vectorized(
                g,
                ts_col=ts_col,
                label_available_ts_col=label_available_ts_col,
                windows=windows,
                score_col=score_col,
                prob_col=prob_col,
                y_col=y_col,
                ret_col=ret_col,
                top_frac=top_frac,
                min_samples=min_samples,
                min_top_samples=min_top_samples,
                top_label=top_label,
            )
        except Exception:
            stats_ts = _compute_scope_timeseries(
                g,
                ts_col=ts_col,
                label_available_ts_col=label_available_ts_col,
                windows=windows,
                metric_fn=_scoped_window_stats,
            )
        if stats_ts.empty:
            continue
        left_ts = g[[ts_col]].copy()
        left_ts[ts_col] = pd.to_datetime(
            left_ts[ts_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
        right_ts = stats_ts.copy()
        right_ts[ts_col] = pd.to_datetime(
            right_ts[ts_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
        mapped = pd.merge_asof(
            left_ts.sort_values(ts_col),
            right_ts.sort_values(ts_col),
            on=ts_col,
            direction="backward",
            allow_exact_matches=True,
        )
        mapped.index = left_ts.sort_values(ts_col).index
        assign_cols: dict[str, np.ndarray] = {}
        for win in windows:
            sfx = win.lower()
            for m in metrics:
                key = f"{m}_{sfx}"
                out_key = f"recent_{scope_name}_{m}_{sfx}"
                raw = (
                    mapped[key]
                    if key in mapped.columns
                    else pd.Series(np.nan, index=mapped.index)
                )
                assign_cols[out_key] = pd.to_numeric(
                    raw, errors="coerce"
                ).to_numpy(dtype=np.float32)
            n_valid_raw = (
                mapped[f"n_valid_{sfx}"]
                if f"n_valid_{sfx}" in mapped.columns
                else pd.Series(np.nan, index=mapped.index)
            )
            model_ece_raw = (
                mapped[f"model_ece_{sfx}"]
                if f"model_ece_{sfx}" in mapped.columns
                else pd.Series(np.nan, index=mapped.index)
            )
            n_valid = pd.to_numeric(
                n_valid_raw, errors="coerce"
            ).to_numpy(dtype=np.float32)
            model_ece = pd.to_numeric(
                model_ece_raw, errors="coerce"
            ).to_numpy(dtype=np.float32)
            assign_cols[f"recent_{scope_name}_available_{sfx}"] = (
                (n_valid >= min_samples) & np.isfinite(model_ece)
            ).astype(np.int8)
        for col, values in assign_cols.items():
            df.loc[mapped.index, col] = values
    return df


def _causal_standardize_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(np.float32)
    hist_mean = numeric.expanding(min_periods=20).mean().shift(1)
    hist_std = numeric.expanding(min_periods=20).std(ddof=0).shift(1)
    z = (numeric - hist_mean) / hist_std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6.0, 6.0)


def standardize_recent_effectiveness_features(
    df: pd.DataFrame,
    *,
    prefix: str = "recent_",
) -> pd.DataFrame:
    """Apply causal scaling to recent-effectiveness metrics.

    The raw rolling metrics are already causal because they only use rows whose
    labels were available before the prediction timestamp. This second pass
    keeps availability flags binary and converts counts/rates/errors/EV metrics
    to bounded, expanding z-scores using only prior generated feature values.
    """
    out = df.copy()
    cols = [c for c in out.columns if str(c).startswith(prefix)]
    for col in cols:
        name = str(col)
        if name.endswith("_available") or "_available_" in name:
            out[col] = (
                pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int8)
            )
            continue
        raw = pd.to_numeric(out[col], errors="coerce")
        if "_n_" in name or name.endswith("_n"):
            raw = np.log1p(raw.clip(lower=0.0))
        elif "hit_rate" in name or "realized_rate" in name:
            raw = raw.clip(0.0, 1.0)
        elif (
            "ece" in name
            or "calibration_error" in name
            or "cal_error" in name
            or "confidence_surprise" in name
        ):
            raw = raw.clip(-1.0, 1.0)
        elif "rolling_ic" in name or "rank_ic" in name:
            raw = raw.clip(-1.0, 1.0)
        out[col] = _causal_standardize_series(raw).astype(np.float32)
    return out


def add_recent_effectiveness_features(
    df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    label_available_ts_col: str = "label_available_ts",
    score_col: str = "score",
    prob_col: str = "p_hat",
    y_col: str = "y_true",
    ret_col: str = "y_ret_net",
    group_cols: tuple[str, ...] = ("side", "horizon"),
    bucket_col: str = "bucket",
    regime_col: str = "regime",
    symbol_col: str = "symbol",
    windows: tuple[str, ...] = ("2D", "5D", "15D"),
    top_frac: float = 0.15,
    min_samples: int = 100,
    min_top_samples: int = 25,
    standardize: bool = True,
) -> pd.DataFrame:
    out = df.sort_values(ts_col).copy()
    required = [ts_col, label_available_ts_col, score_col, prob_col, y_col, ret_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out[label_available_ts_col] = pd.to_datetime(
        out[label_available_ts_col], errors="coerce"
    )
    scopes: dict[str, tuple[str, ...]] = {
        "global": (),
        "side_horizon": group_cols,
        "bucket": (bucket_col,),
        "bucket_side_horizon": (bucket_col, *group_cols),
        "regime": (regime_col,),
        "asset": (symbol_col,),
        "asset_side_horizon": (symbol_col, *group_cols),
    }
    for scope, cols in scopes.items():
        if all(c in out.columns for c in cols):
            out = _add_scope_features(
                out,
                scope_name=scope,
                scope_cols=cols,
                windows=windows,
                ts_col=ts_col,
                label_available_ts_col=label_available_ts_col,
                score_col=score_col,
                prob_col=prob_col,
                y_col=y_col,
                ret_col=ret_col,
                top_frac=top_frac,
                min_samples=min_samples,
                min_top_samples=min_top_samples,
            )
    _default_availability_col = f"recent_global_available_{windows[-1].lower()}"
    out["recent_effectiveness_available"] = out.get(
        _default_availability_col,
        pd.Series(np.zeros(len(out), dtype=np.int8), index=out.index),
    ).astype(np.int8)
    if standardize:
        out = standardize_recent_effectiveness_features(out)
    return out


def add_recent_meta_self_features(
    df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    label_available_ts_col: str = "label_available_ts",
    meta_score_col: str = "meta_score",
    meta_prob_col: str = "p_meta",
    y_success_col: str = "y_true",
    ret_col: str = "y_ret_net",
    meta_accept_col: str = "meta_accept",
    symbol_col: str = "symbol",
    windows: tuple[str, ...] = ("5D", "10D", "30D"),
    top_frac: float = 0.15,
    min_samples: int = 50,
    min_top_samples: int = 20,
    standardize: bool = True,
) -> pd.DataFrame:
    out = df.sort_values(ts_col).copy()
    required = [ts_col, label_available_ts_col, meta_prob_col, y_success_col, ret_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    base = out.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    base[ts_col] = pd.to_datetime(base[ts_col], errors="coerce")
    out[label_available_ts_col] = pd.to_datetime(
        out[label_available_ts_col], errors="coerce"
    )
    base[label_available_ts_col] = pd.to_datetime(
        base[label_available_ts_col], errors="coerce"
    )
    scopes = {
        "global": (),
        "side_horizon": ("side", "horizon"),
        "bucket": ("bucket",),
        "asset": (symbol_col,),
        "asset_side_horizon": (symbol_col, "side", "horizon"),
    }
    for scope, cols in scopes.items():
        if not all(c in base.columns for c in cols):
            continue
        grouped = (
            base.groupby(list(cols), sort=False, dropna=False)
            if cols
            else [(None, base)]
        )
        for win in windows:
            sfx = win.lower()
            top_label = f"top{int(round(top_frac * 100))}"
            for name in [
                "rank_ic",
                "ece",
                "brier",
                f"{top_label}_cal_error",
                f"{top_label}_hit_rate",
                f"{top_label}_ev",
                "accept_hit_rate",
                "accept_ev",
                "false_accept_rate",
                "false_reject_rate",
                "reject_opportunity_cost",
                "n",
            ]:
                out[f"recent_meta_{scope}_{name}_{sfx}"] = np.nan
            out[f"recent_meta_{scope}_available_{sfx}"] = np.int8(0)
        for _, g in grouped:
            idx = g.index.to_numpy()
            ts = g[ts_col]
            for i, ridx in enumerate(idx):
                t = ts.iloc[i]
                for win in windows:
                    hist = g[
                        (g[label_available_ts_col] < t)
                        & (g[ts_col] >= t - pd.Timedelta(win))
                    ]
                    sfx = win.lower()
                    n = len(hist)
                    out.at[ridx, f"recent_meta_{scope}_n_{sfx}"] = float(n)
                    if n < min_samples:
                        continue
                    p = hist[meta_prob_col].to_numpy(np.float32)
                    mscore = (
                        hist[meta_score_col].to_numpy(np.float32)
                        if meta_score_col in hist.columns
                        else p
                    )
                    y = hist[y_success_col].to_numpy(np.float32)
                    r = hist[ret_col].to_numpy(np.float32)
                    a = (
                        hist[meta_accept_col].to_numpy(np.float32)
                        if meta_accept_col in hist.columns
                        else (p >= 0.5).astype(np.float32)
                    )
                    out.at[ridx, f"recent_meta_{scope}_rank_ic_{sfx}"] = _spearman_corr(
                        mscore, y
                    )
                    out.at[ridx, f"recent_meta_{scope}_ece_{sfx}"] = _ece_from_hist(
                        p, y
                    )
                    brier_ok = np.isfinite(p) & np.isfinite(y)
                    out.at[ridx, f"recent_meta_{scope}_brier_{sfx}"] = (
                        float(np.nanmean((p[brier_ok] - y[brier_ok]) ** 2))
                        if int(brier_ok.sum()) > 0
                        else np.nan
                    )
                    finite_score = np.isfinite(mscore)
                    top = np.zeros_like(finite_score, dtype=bool)
                    if int(finite_score.sum()) >= min_top_samples:
                        thr = np.nanquantile(mscore[finite_score], 1.0 - top_frac)
                        top = finite_score & (mscore >= thr)
                    top_ok = top & np.isfinite(p) & np.isfinite(y) & np.isfinite(r)
                    if int(top_ok.sum()) >= min_top_samples:
                        out.at[
                            ridx, f"recent_meta_{scope}_{top_label}_cal_error_{sfx}"
                        ] = float(np.nanmean(p[top_ok]) - np.nanmean(y[top_ok]))
                        out.at[
                            ridx, f"recent_meta_{scope}_{top_label}_hit_rate_{sfx}"
                        ] = float(np.nanmean(y[top_ok]))
                        out.at[ridx, f"recent_meta_{scope}_{top_label}_ev_{sfx}"] = (
                            float(np.nanmean(r[top_ok]))
                        )
                    valid_accept = np.isfinite(a)
                    accept = valid_accept & (a > 0.0)
                    accept_ret = accept & np.isfinite(r)
                    if int(accept_ret.sum()) >= min_top_samples:
                        out.at[ridx, f"recent_meta_{scope}_accept_hit_rate_{sfx}"] = (
                            float(np.nanmean(y[accept_ret]))
                        )
                        out.at[ridx, f"recent_meta_{scope}_accept_ev_{sfx}"] = float(
                            np.nanmean(r[accept_ret])
                        )
                    reject = valid_accept & (a <= 0.0)
                    reject_ret = reject & np.isfinite(r)
                    out.at[ridx, f"recent_meta_{scope}_false_accept_rate_{sfx}"] = (
                        float(np.nanmean((r[accept_ret] <= 0.0).astype(np.float32)))
                        if int(accept_ret.sum()) > 0
                        else np.nan
                    )
                    out.at[ridx, f"recent_meta_{scope}_false_reject_rate_{sfx}"] = (
                        float(np.nanmean((r[reject_ret] > 0.0).astype(np.float32)))
                        if int(reject_ret.sum()) > 0
                        else np.nan
                    )
                    out.at[
                        ridx, f"recent_meta_{scope}_reject_opportunity_cost_{sfx}"
                    ] = (
                        float(np.nanmean(np.maximum(r[reject_ret], 0.0)))
                        if int(reject_ret.sum()) > 0
                        else np.nan
                    )
                    out.at[ridx, f"recent_meta_{scope}_available_{sfx}"] = np.int8(1)
    top_label = f"top{int(round(top_frac * 100))}"
    out["recent_meta_ece_30d"] = out.get("recent_meta_global_ece_30d")
    out["recent_meta_brier_30d"] = out.get("recent_meta_global_brier_30d")
    out["recent_meta_brier_10d"] = out.get("recent_meta_global_brier_10d")
    out["recent_meta_brier_5d"] = out.get("recent_meta_global_brier_5d")
    out[f"recent_meta_{top_label}_cal_error_30d"] = out.get(
        f"recent_meta_global_{top_label}_cal_error_30d"
    )
    out[f"recent_meta_{top_label}_cal_error_10d"] = out.get(
        f"recent_meta_global_{top_label}_cal_error_10d"
    )
    out[f"recent_meta_{top_label}_cal_error_5d"] = out.get(
        f"recent_meta_global_{top_label}_cal_error_5d"
    )
    out["recent_meta_accept_hit_rate_5d"] = out.get(
        "recent_meta_global_accept_hit_rate_5d"
    )
    if standardize:
        out = standardize_recent_effectiveness_features(out, prefix="recent_meta_")
    return out



def _infer_base_probability_columns(
    df: pd.DataFrame,
    *,
    meta_prob_col: str,
    explicit: Iterable[str] | None = None,
) -> list[str]:
    if explicit is not None:
        return [str(c) for c in explicit if str(c) in df.columns]
    prefixes = (
        "base_prob_",
        "base_probability_",
        "base_model_prob_",
        "p_base_",
        "pred_base_",
    )
    cols = [
        str(c)
        for c in df.columns
        if str(c) != meta_prob_col and any(str(c).startswith(p) for p in prefixes)
    ]
    # Horizon-style base prediction columns used in several runtime paths.
    cols.extend(
        str(c)
        for c in df.columns
        if str(c).startswith("base_H") or str(c).startswith("pred_H")
    )
    return list(dict.fromkeys(cols))


def add_recent_prediction_disagreement_features(
    df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    label_available_ts_col: str = "label_available_ts",
    meta_prob_col: str = "p_meta",
    y_success_col: str = "y_true",
    base_prob_cols: Iterable[str] | None = None,
    windows: tuple[str, ...] = ("3D", "7D", "15D"),
    min_samples: int = 20,
    standardize: bool = True,
) -> pd.DataFrame:
    """Add causal rolling meta/base calibration and disagreement diagnostics.

    Features are computed using only rows whose label was available before the
    current prediction timestamp.  The outputs include recent meta-model Brier
    score, base-vs-meta disagreement by subtraction and ratio, and internal base
    model disagreement aggregates over 3/7/15 day windows by default.
    """
    out = df.sort_values(ts_col).copy()
    if ts_col not in out.columns or label_available_ts_col not in out.columns:
        raise ValueError(f"Missing required columns: {[ts_col, label_available_ts_col]}")
    if meta_prob_col not in out.columns:
        raise ValueError(f"Missing required meta probability column: {meta_prob_col}")
    base_cols = _infer_base_probability_columns(
        out, meta_prob_col=meta_prob_col, explicit=base_prob_cols
    )
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out[label_available_ts_col] = pd.to_datetime(
        out[label_available_ts_col], errors="coerce"
    )
    meta = pd.to_numeric(out[meta_prob_col], errors="coerce").astype(np.float32)
    if base_cols:
        base_mat = out[base_cols].apply(pd.to_numeric, errors="coerce").astype(
            np.float32
        )
        base_mean = base_mat.mean(axis=1, skipna=True).astype(np.float32)
        base_std = base_mat.std(axis=1, skipna=True, ddof=0).astype(np.float32)
        base_range = (base_mat.max(axis=1, skipna=True) - base_mat.min(axis=1, skipna=True)).astype(np.float32)
    else:
        base_mean = pd.Series(np.nan, index=out.index, dtype=np.float32)
        base_std = pd.Series(np.nan, index=out.index, dtype=np.float32)
        base_range = pd.Series(np.nan, index=out.index, dtype=np.float32)
    out["base_meta_disagreement_sub"] = (base_mean - meta).astype(np.float32)
    out["base_meta_disagreement_abs_sub"] = (base_mean - meta).abs().astype(np.float32)
    out["base_meta_disagreement_ratio"] = (base_mean / meta.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)
    out["base_internal_disagreement_std"] = base_std
    out["base_internal_disagreement_range"] = base_range

    for win in windows:
        sfx = win.lower()
        for name in (
            "meta_brier",
            "base_meta_disagreement_sub_mean",
            "base_meta_disagreement_abs_sub_mean",
            "base_meta_disagreement_ratio_mean",
            "base_internal_disagreement_std_mean",
            "base_internal_disagreement_range_mean",
            "base_internal_disagreement_std_max",
            "base_internal_disagreement_range_max",
            "n",
        ):
            out[f"recent_{name}_{sfx}"] = np.nan
        out[f"recent_prediction_disagreement_available_{sfx}"] = np.int8(0)

    for ridx, row in out.iterrows():
        t = row[ts_col]
        if pd.isna(t):
            continue
        for win in windows:
            sfx = win.lower()
            hist = out[
                (out[label_available_ts_col] < t)
                & (out[ts_col] >= t - pd.Timedelta(win))
            ]
            n = int(len(hist))
            out.at[ridx, f"recent_n_{sfx}"] = float(n)
            if n < int(min_samples):
                continue
            p = pd.to_numeric(hist[meta_prob_col], errors="coerce").to_numpy(np.float32)
            if y_success_col in hist.columns:
                y = pd.to_numeric(hist[y_success_col], errors="coerce").to_numpy(np.float32)
                ok = np.isfinite(p) & np.isfinite(y)
                if int(ok.sum()) > 0:
                    out.at[ridx, f"recent_meta_brier_{sfx}"] = float(
                        np.nanmean((p[ok] - y[ok]) ** 2)
                    )
            for src, dst in (
                ("base_meta_disagreement_sub", "base_meta_disagreement_sub_mean"),
                ("base_meta_disagreement_abs_sub", "base_meta_disagreement_abs_sub_mean"),
                ("base_meta_disagreement_ratio", "base_meta_disagreement_ratio_mean"),
                ("base_internal_disagreement_std", "base_internal_disagreement_std_mean"),
                ("base_internal_disagreement_range", "base_internal_disagreement_range_mean"),
            ):
                vals = pd.to_numeric(hist[src], errors="coerce")
                out.at[ridx, f"recent_{dst}_{sfx}"] = float(vals.mean()) if vals.notna().any() else np.nan
            out.at[ridx, f"recent_base_internal_disagreement_std_max_{sfx}"] = float(
                pd.to_numeric(hist["base_internal_disagreement_std"], errors="coerce").max()
            )
            out.at[ridx, f"recent_base_internal_disagreement_range_max_{sfx}"] = float(
                pd.to_numeric(hist["base_internal_disagreement_range"], errors="coerce").max()
            )
            out.at[ridx, f"recent_prediction_disagreement_available_{sfx}"] = np.int8(1)
    if standardize:
        out = standardize_recent_effectiveness_features(out, prefix="recent_")
    return out
