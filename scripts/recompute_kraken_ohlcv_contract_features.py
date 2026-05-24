#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.kraken_actual_data import overlay_actual_volume_sidecar
from extreme_price_movements.perp_features import compute_features as compute_perp_features
from extreme_price_movements.features_residual import add_residual_features


PRICE_TARGET_COLUMNS = (
    "dist_vwap_norm",
    "vwap_zone_1d_atr",
    "vwap_zone_7d_atr",
    "dist_vwap_12_atr",
    "dist_vwap_24_atr",
    "dist_vwap_96_atr",
    "trapped_longs_12",
    "trapped_longs_24",
    "trapped_longs_96",
    "dist_stack",
)
OI_TARGET_COLUMNS = (
    "oi_rel_vol_2h",
    "oi_rel_vol_4h",
    "oi_rel_vol_8h",
    "oi_chg_2h",
    "oi_chg_4h",
    "oi_chg_8h",
    "oi_chg_z_2h",
    "oi_chg_z_4h",
    "oi_chg_z_8h",
    "oi_value_log_1d_robust_z",
    "oi_value_log_7d_robust_z",
    "oi_chg_2h_robust_z",
    "oi_chg_4h_robust_z",
    "oi_chg_8h_robust_z",
    "oi_vel_2h",
    "oi_vel_4h",
    "oi_vel_8h",
    "oi_chg_w",
    "unwind_score",
)
STATE_TARGET_COLUMNS = (
    "innovation_z_x_zr_1h",
    "innovation_z_x_zr_3h",
)
BASIS_TARGET_COLUMNS = (
    "basis",
    "basis_frac",
    "basis_pct",
    "basis_pct_z",
    "basis_frac_z_14d",
    "basis_frac_rank_30d",
    "basis_per_atr",
    "basis_stretch",
    "basis_vol",
    "basis_mom_2h",
    "basis_mom_4h",
    "basis_mom_8h",
    "basis_mom_w",
    "basis_fund_div_z",
    "basis_funding_div",
    "basis_funding_div_2h",
    "basis_funding_div_4h",
    "basis_funding_div_8h",
    "basis_up_agree",
    "leverage_build",
    "leverage_build_score",
    "unwind",
    "unwind_score",
    "squeeze_prob",
    "basis_adjusted_trend_5h",
    "basis_adjusted_trend_10h",
    "basis_adjusted_trend_self_z_5h",
    "basis_adjusted_trend_self_z_10h",
)
BASIS_RESIDUAL_TARGET_COLUMNS = (
    "basis_pct_mkt_resid",
    "basis_fund_div_mkt_resid",
    "squeeze_prob_mkt_resid",
)
LEGACY_TARGET_COLUMNS = (
    "tail_asymmetry_q90_q10_atr_norm",
)
MODEL_CONTRACT_TARGET_COLUMNS = PRICE_TARGET_COLUMNS + (
    "oi_rel_vol_2h",
    "oi_rel_vol_4h",
    "oi_rel_vol_8h",
    "oi_value_log_1d_robust_z",
    "oi_value_log_7d_robust_z",
    "oi_chg_2h_robust_z",
    "oi_chg_4h_robust_z",
    "oi_chg_8h_robust_z",
    "unwind_score",
) + STATE_TARGET_COLUMNS
TARGET_COLUMNS = tuple(dict.fromkeys(
    PRICE_TARGET_COLUMNS
    + OI_TARGET_COLUMNS
    + STATE_TARGET_COLUMNS
    + BASIS_TARGET_COLUMNS
    + BASIS_RESIDUAL_TARGET_COLUMNS
    + LEGACY_TARGET_COLUMNS
))


def _load_symbol_raw(ohlcv_root: Path, symbol_key: str) -> pd.DataFrame:
    files = sorted((ohlcv_root / f"symbol={symbol_key}").glob("year=*/compact-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw OHLCV partitions for {symbol_key}")
    raw = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["ts"]).sort_values("ts")
    return raw.drop_duplicates(subset=["ts"], keep="last").set_index("ts")


def _load_open_interest_sidecar(raw_root: Path, symbol_key: str, index: pd.DatetimeIndex) -> pd.Series:
    sidecar_root = raw_root / "open_interest_hourly"
    candidates = [
        sidecar_root / f"{symbol_key.replace(':', '_')}.parquet",
        sidecar_root / f"{symbol_key}.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "open_interest" not in frame.columns:
            continue
        if "ts" in frame.columns:
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        else:
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "ts"})
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["ts"]).sort_values("ts")
        frame = frame.drop_duplicates(subset=["ts"], keep="last").set_index("ts")
        return pd.to_numeric(frame["open_interest"], errors="coerce").reindex(index)
    return pd.Series(np.nan, index=index, dtype="float64")


def _series(raw: pd.DataFrame, column: str, index: pd.DatetimeIndex) -> pd.Series:
    if column not in raw.columns:
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce").reindex(index)


def _rolling_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    vol = volume.replace([np.inf, -np.inf], np.nan).where(lambda s: s > 0.0)
    px = price.replace([np.inf, -np.inf], np.nan).where(lambda s: s > 0.0)
    num = (px * vol).rolling(window, min_periods=1).sum()
    den = vol.rolling(window, min_periods=1).sum()
    return (num / den.replace(0.0, np.nan)).ffill()


def _atr_abs(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 24) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


def _rolling_quantile(x: pd.Series, window: int, q: float) -> pd.Series:
    return x.rolling(window, min_periods=5).quantile(q)


def _compute_price_targets(raw: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    index = feat.index
    close = _series(raw, "close", index).where(lambda s: s > 0.0).ffill()
    high = _series(raw, "high", index).where(lambda s: s > 0.0).ffill()
    low = _series(raw, "low", index).where(lambda s: s > 0.0).ffill()
    volume = _series(raw, "volume", index).where(lambda s: s > 0.0)
    log_close = np.log(close)
    atr = _atr_abs(high, low, close, 24).replace(0.0, np.nan)
    atr_pct = (atr / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    atr_ln = atr_pct.replace(0.0, np.nan)

    out = pd.DataFrame(index=index)
    vwap_24 = _rolling_vwap(close, volume, 24)
    out["dist_vwap_norm"] = ((close - vwap_24) / atr.replace(0.0, np.nan)).clip(-50, 50)

    ret1h = log_close.diff(1)
    q90 = _rolling_quantile(ret1h, 50, 0.90)
    q10 = _rolling_quantile(ret1h, 50, 0.10).abs()
    out["tail_asymmetry_q90_q10_atr_norm"] = np.tanh(
        np.log((q90 + 1e-8) / (q10 + 1e-8))
    )

    vwap_24_shift = vwap_24.shift(1)
    vwap_168 = _rolling_vwap(close, volume, 24 * 7)
    vwap_168_shift = vwap_168.shift(1)
    lower_day_vwap = vwap_24_shift.rolling(24, min_periods=1).min()
    upper_day_vwap = vwap_24_shift.rolling(24, min_periods=1).max()
    lower_week_vwap = vwap_168_shift.rolling(24 * 7, min_periods=1).min()
    upper_week_vwap = vwap_168_shift.rolling(24 * 7, min_periods=1).max()
    atr_1d_pct = atr_pct.rolling(24, min_periods=1).mean().shift(1)
    atr_7d_pct = atr_pct.rolling(24 * 7, min_periods=1).mean().shift(1)
    zone = ((close - lower_day_vwap) / (upper_day_vwap - lower_day_vwap + 1e-12)) / (
        atr_1d_pct + 1e-12
    )
    out["vwap_zone_1d_atr"] = zone.replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)
    zone_7d = (
        ((close - lower_week_vwap) / (upper_week_vwap - lower_week_vwap + 1e-12))
        / (atr_7d_pct + 1e-12)
    )
    out["vwap_zone_7d_atr"] = zone_7d.replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)

    for n in (12, 24, 96):
        vwap_n = np.log(_rolling_vwap(close, volume, n))
        dist = ((log_close - vwap_n) / (atr_ln + 1e-12)).clip(-50, 50)
        out[f"dist_vwap_{n}_atr"] = dist
        out[f"trapped_longs_{n}"] = (-dist).clip(lower=0.0)

    if {"dist_ema_fast", "trend_pct"}.issubset(feat.columns):
        out["dist_stack"] = (
            pd.to_numeric(feat["dist_ema_fast"], errors="coerce")
            + out["dist_vwap_norm"]
            + pd.to_numeric(feat["trend_pct"], errors="coerce")
        )

    return out.replace([np.inf, -np.inf], np.nan).astype("float32")


def _compute_oi_targets(raw: pd.DataFrame, raw_root: Path, symbol_key: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = _series(raw, "close", index)
    mark = _series(raw, "mark_price", index)
    if mark.notna().sum() == 0:
        mark = _series(raw, "mark_close", index)
    spot = _series(raw, "spot_close", index)
    raw_oi = _series(raw, "open_interest", index)
    sidecar_oi = _load_open_interest_sidecar(raw_root, symbol_key, index)
    open_interest = raw_oi.combine_first(sidecar_oi)
    df = pd.DataFrame(
        {
            "funding_rate": _series(raw, "funding_rate", index),
            "open_interest": open_interest,
            "perp_price": close,
            "spot_price": spot,
            "mark_price": mark,
            "volume": _series(raw, "volume", index),
            "quote_volume": _series(raw, "volume", index) * close,
            "close": close,
        },
        index=index,
    )
    perp = compute_perp_features(df)
    cols = [col for col in OI_TARGET_COLUMNS if col in perp.columns]
    return perp.reindex(columns=cols).astype("float32")


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(window, max(5, 24 * 30 if window >= 24 * 30 else window // 5))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((series - mean) / (std.replace(0.0, np.nan) + 1e-12)).replace(
        [np.inf, -np.inf], np.nan
    )


def _compute_basis_targets(raw: pd.DataFrame, raw_root: Path, symbol_key: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = _series(raw, "close", index).where(lambda s: s > 0.0)
    high = _series(raw, "high", index).where(lambda s: s > 0.0)
    low = _series(raw, "low", index).where(lambda s: s > 0.0)
    mark = _series(raw, "mark_price", index).where(lambda s: s > 0.0)
    if mark.notna().sum() == 0:
        mark = _series(raw, "mark_close", index).where(lambda s: s > 0.0)
    spot = _series(raw, "spot_close", index).where(lambda s: s > 0.0)
    raw_oi = _series(raw, "open_interest", index)
    sidecar_oi = _load_open_interest_sidecar(raw_root, symbol_key, index)
    open_interest = raw_oi.combine_first(sidecar_oi)
    perp_input = pd.DataFrame(
        {
            "funding_rate": _series(raw, "funding_rate", index),
            "open_interest": open_interest,
            "perp_price": close,
            "spot_price": spot,
            "mark_price": mark,
            "volume": _series(raw, "volume", index),
            "quote_volume": _series(raw, "volume", index) * close,
            "close": close,
        },
        index=index,
    )
    perp = compute_perp_features(perp_input)
    out = perp.reindex(columns=[col for col in BASIS_TARGET_COLUMNS if col in perp.columns])

    basis_pct = pd.to_numeric(perp.get("basis_pct"), errors="coerce")
    atr = _atr_abs(high, low, close, 24)
    atr_pct = (atr / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["basis_per_atr"] = (basis_pct / (atr_pct.abs() + 1e-12)).clip(-50, 50)
    if "basis_pct_z" in out and "funding_z" in perp:
        out["basis_fund_div_z"] = (
            pd.to_numeric(out["basis_pct_z"], errors="coerce")
            - pd.to_numeric(perp["funding_z"], errors="coerce")
        ).clip(-10, 10)

    log_close = np.log(close.replace(0.0, np.nan))
    for horizon in (5, 10):
        ret_h = log_close.diff(horizon)
        basis_adjusted = ret_h - basis_pct.diff(horizon)
        out[f"basis_adjusted_trend_{horizon}h"] = basis_adjusted.clip(-1.0, 1.0)
        out[f"basis_adjusted_trend_self_z_{horizon}h"] = _rolling_zscore(
            basis_adjusted, 14 * 24
        ).clip(-6.0, 6.0)

    return out.replace([np.inf, -np.inf], np.nan).astype("float32")


def _compute_state_targets(feat: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=feat.index)
    source_col = (
        "price_innovation_z"
        if "price_innovation_z" in feat.columns
        else "price_minus_state_z"
        if "price_minus_state_z" in feat.columns
        else None
    )
    if source_col is None:
        return out
    innovation_z = pd.to_numeric(feat[source_col], errors="coerce")
    zr_1h = None
    if "zr_1h" in feat.columns:
        zr_1h = pd.to_numeric(feat["zr_1h"], errors="coerce")
    elif {"ret1h", "ret3h", "zr_3h"}.issubset(feat.columns):
        ret1h = pd.to_numeric(feat["ret1h"], errors="coerce")
        ret3h = pd.to_numeric(feat["ret3h"], errors="coerce")
        zr3h = pd.to_numeric(feat["zr_3h"], errors="coerce")
        zr_1h = ret1h * zr3h / ret3h.replace(0.0, np.nan)
    if zr_1h is not None:
        out["innovation_z_x_zr_1h"] = innovation_z * zr_1h
    if "zr_3h" in feat.columns:
        out["innovation_z_x_zr_3h"] = innovation_z * pd.to_numeric(
            feat["zr_3h"], errors="coerce"
        )
    return out.replace([np.inf, -np.inf], np.nan).astype("float32")


def _feature_path_symbol(path: Path) -> str:
    return path.stem.removeprefix("symbol=")


def _load_wide_feature_frame(files: list[Path], column: str) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    index: pd.DatetimeIndex | None = None
    for path in files:
        symbol = _feature_path_symbol(path)
        try:
            frame = pd.read_parquet(path, columns=[column])
        except Exception:
            frame = pd.read_parquet(path)
            if column not in frame.columns:
                continue
            frame = frame[[column]]
        if not isinstance(frame.index, pd.DatetimeIndex):
            if "ts" not in frame.columns:
                continue
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
            frame = frame.set_index("ts")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True), name="ts")
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        series[symbol] = values.astype("float32")
        index = values.index if index is None else index.union(values.index)
    if index is None or not series:
        return pd.DataFrame()
    return pd.DataFrame({k: v.reindex(index) for k, v in series.items()}, index=index).sort_index()


def _recompute_basis_residual_targets(
    files: list[Path],
    selected: tuple[str, ...],
    *,
    dry_run: bool,
) -> pd.DataFrame:
    selected_set = set(selected).intersection(BASIS_RESIDUAL_TARGET_COLUMNS)
    if not selected_set:
        return pd.DataFrame()

    source_by_target = {
        "basis_pct_mkt_resid": "basis_pct_z",
        "basis_fund_div_mkt_resid": "basis_fund_div_z",
        "squeeze_prob_mkt_resid": "squeeze_prob",
    }
    feats: dict[str, pd.DataFrame] = {}
    for target in selected_set:
        source = source_by_target[target]
        if source not in feats:
            feats[source] = _load_wide_feature_frame(files, source)
    if not feats:
        return pd.DataFrame()

    add_residual_features(feats, mkt_gates=None, cfg={})
    residual_frames = {
        target: feats[target]
        for target in selected_set
        if target in feats and isinstance(feats[target], pd.DataFrame)
    }
    if not residual_frames:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for i, path in enumerate(files, start=1):
        symbol = _feature_path_symbol(path)
        status = "updated"
        error = ""
        before: dict[str, float] = {}
        after: dict[str, float] = {}
        try:
            feat = pd.read_parquet(path)
            if not isinstance(feat.index, pd.DatetimeIndex):
                feat["ts"] = pd.to_datetime(feat["ts"], utc=True, errors="coerce")
                feat = feat.set_index("ts")
            feat.index = pd.DatetimeIndex(pd.to_datetime(feat.index, utc=True), name="ts")
            for col, wide in residual_frames.items():
                before[col] = (
                    float(
                        pd.to_numeric(feat.get(col), errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                        .isna()
                        .mean()
                    )
                    if col in feat.columns
                    else 1.0
                )
                values = wide[symbol].reindex(feat.index) if symbol in wide.columns else pd.Series(np.nan, index=feat.index)
                after[col] = float(values.replace([np.inf, -np.inf], np.nan).isna().mean())
                if not dry_run:
                    feat[col] = values.astype("float32")
            if not dry_run:
                tmp = path.with_suffix(".parquet.tmp")
                feat.to_parquet(tmp, compression="zstd")
                tmp.replace(path)
        except Exception as exc:
            status = "failed"
            error = str(exc)
        row: dict[str, object] = {"symbol": symbol, "status": status, "error": error}
        for col in BASIS_RESIDUAL_TARGET_COLUMNS:
            if col not in selected_set:
                continue
            row[f"{col}_nan_before"] = before.get(col, np.nan)
            row[f"{col}_nan_after"] = after.get(col, np.nan)
        rows.append(row)
        if i <= 5 or i % 25 == 0 or i == len(files) or status == "failed":
            msg = f"residual {i:03d}/{len(files)} {symbol} {status}"
            if status == "failed":
                msg += f": {error}"
            print(msg, flush=True)
    return pd.DataFrame(rows)


def _selected_columns(args: argparse.Namespace) -> tuple[str, ...]:
    if args.columns:
        selected = tuple(dict.fromkeys(c.strip() for c in args.columns.split(",") if c.strip()))
    elif args.preset == "model-contract":
        selected = MODEL_CONTRACT_TARGET_COLUMNS
    else:
        selected = TARGET_COLUMNS
    unknown = sorted(set(selected).difference(TARGET_COLUMNS))
    if unknown:
        raise ValueError(f"unknown target columns: {unknown}")
    return selected


def recompute(args: argparse.Namespace) -> pd.DataFrame:
    feature_dir = Path(args.feature_dir)
    raw_root = Path(args.raw_root)
    ohlcv_root = raw_root / "ohlcv"
    files = sorted(feature_dir.glob("symbol=*.parquet"))
    selected = _selected_columns(args)
    selected_set = set(selected)
    rows: list[dict[str, object]] = []
    per_symbol_columns = selected_set.difference(BASIS_RESIDUAL_TARGET_COLUMNS)
    for i, path in enumerate(files, start=1):
        if not per_symbol_columns:
            break
        symbol_key = path.stem.removeprefix("symbol=")
        status = "updated"
        error = ""
        try:
            feat = pd.read_parquet(path)
            if not isinstance(feat.index, pd.DatetimeIndex):
                feat["ts"] = pd.to_datetime(feat["ts"], utc=True, errors="coerce")
                feat = feat.set_index("ts")
            feat.index = pd.DatetimeIndex(pd.to_datetime(feat.index, utc=True), name="ts")
            raw = _load_symbol_raw(ohlcv_root, symbol_key)
            raw = overlay_actual_volume_sidecar(raw, root_dir=raw_root, symbol=symbol_key)
            target_frames = []
            if selected_set.intersection(PRICE_TARGET_COLUMNS + LEGACY_TARGET_COLUMNS):
                target_frames.append(_compute_price_targets(raw, feat))
            if selected_set.intersection(OI_TARGET_COLUMNS):
                target_frames.append(_compute_oi_targets(raw, raw_root, symbol_key, feat.index))
            if selected_set.intersection(BASIS_TARGET_COLUMNS):
                target_frames.append(_compute_basis_targets(raw, raw_root, symbol_key, feat.index))
            if selected_set.intersection(STATE_TARGET_COLUMNS):
                target_frames.append(_compute_state_targets(feat))
            if target_frames:
                targets = pd.concat(target_frames, axis=1)
                targets = targets.loc[:, ~targets.columns.duplicated(keep="last")]
                targets = targets.reindex(
                    columns=[col for col in selected if col in targets.columns]
                )
            else:
                targets = pd.DataFrame(index=feat.index)
            before = {
                col: float(pd.to_numeric(feat.get(col), errors="coerce").replace([np.inf, -np.inf], np.nan).isna().mean())
                if col in feat.columns
                else 1.0
                for col in selected
            }
            after = {
                col: float(targets[col].replace([np.inf, -np.inf], np.nan).isna().mean())
                for col in selected
                if col in targets.columns
            }
            if not args.dry_run:
                for col in targets.columns:
                    feat[col] = targets[col].astype("float32")
                tmp = path.with_suffix(".parquet.tmp")
                feat.to_parquet(tmp, compression="zstd")
                tmp.replace(path)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            before = {col: np.nan for col in TARGET_COLUMNS}
            after = {col: np.nan for col in TARGET_COLUMNS}
        row: dict[str, object] = {"symbol": symbol_key, "status": status, "error": error}
        for col in TARGET_COLUMNS:
            if col not in selected_set:
                continue
            row[f"{col}_nan_before"] = before.get(col, np.nan)
            row[f"{col}_nan_after"] = after.get(col, np.nan)
        rows.append(row)
        if i <= 5 or i % 25 == 0 or i == len(files) or status == "failed":
            msg = f"{i:03d}/{len(files)} {symbol_key} {status}"
            if status == "failed":
                msg += f": {error}"
            print(msg, flush=True)
    report = pd.DataFrame(rows)
    residual_report = _recompute_basis_residual_targets(
        files,
        selected,
        dry_run=bool(args.dry_run),
    )
    if not residual_report.empty:
        if report.empty:
            report = residual_report
        else:
            report = pd.merge(
                report,
                residual_report,
                on=["symbol", "status", "error"],
                how="outer",
                suffixes=("", "_residual"),
            )
            for col in BASIS_RESIDUAL_TARGET_COLUMNS:
                for suffix in ("nan_before", "nan_after"):
                    base_col = f"{col}_{suffix}"
                    residual_col = f"{base_col}_residual"
                    if residual_col not in report.columns:
                        continue
                    if base_col in report.columns:
                        report[base_col] = report[residual_col].combine_first(report[base_col])
                    else:
                        report[base_col] = report[residual_col]
                    report.drop(columns=[residual_col], inplace=True)
    if not args.dry_run:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="data_perp/features/20260523_015947")
    parser.add_argument("--raw-root", default="data_perp/exchanges/krakenfutures")
    parser.add_argument(
        "--report",
        default="data_perp/artifacts/20260523_015947/features/model_contract_target_recompute_report.csv",
    )
    parser.add_argument(
        "--preset",
        choices=("model-contract", "all"),
        default="model-contract",
        help="model-contract recomputes only contract-sensitive sparse features.",
    )
    parser.add_argument(
        "--columns",
        default="",
        help="Comma-separated explicit target columns; overrides --preset.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = recompute(args)
    failed = report[report["status"].ne("updated")]
    print(
        f"completed updated={len(report) - len(failed)} failed={len(failed)} "
        f"report={args.report if not args.dry_run else '<dry-run>'}",
        flush=True,
    )
    if len(failed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
