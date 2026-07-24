#!/usr/bin/env python3
"""Proxy-test label quality before model training.

This script intentionally does not fit LightGBM or any other estimator. It
answers two cheaper questions first:

1. If we rank by the label itself, does the top bucket stay inside a plausible
   economic envelope?
2. Can simple, out-of-time univariate feature proxies recover that label well
   enough to select economically useful rows?
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_LABELS_DIR = Path(
    "data_perp/artifacts/"
    "20260702_180500_single_head_monthly_walkforward_"
    "july_feature_refresh_labels_labels_s10_policy_net_recent_cov95/labels"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/label_quality_proxy_diagnostics_s10_policy_net"
)
DEFAULT_FEATURE_DIR = Path("data_perp/features/20260629_050000")
DEFAULT_FEATURE_LIST_CSV = Path(
    "data_perp/artifacts/"
    "20260702_004500_single_head_monthly_walkforward_s10_policy_net_gateoff_"
    "train_april_score_may/quality_reports/base_model_feature_importance.csv"
)

ROUND_TRIP_COST = 0.0030
TOP_FRACS = (0.30, 0.10, 0.05, 0.03, 0.01)
PROXY_TOP_K_FEATURES = 8

FUTURE_OR_LABEL_COLUMNS = {
    "__y_lbl__",
    "__mfe__",
    "__mae__",
    "__tp__",
    "__sl__",
    "__is_timeout__",
    "__quality__",
    "__mae_ret__",
    "__mfe_ret__",
    "__bars_to_mfe__",
    "__bars_policy__",
    "__barrier_pct__",
    "__n_tp__",
    "__n_sl__",
    "__w_consensus__",
    "__y_bin__",
    "__y_ret__",
    "__y_outcome__",
    "__w__",
    "__ts__",
    "__symbol__",
    "__u_policy_net__",
    "__r_policy_net__",
    "__side__",
    "candidate_id",
    "side_name",
    "timeframe",
}


@dataclass(frozen=True)
class LabelArm:
    name: str
    description: str


LABEL_ARMS = (
    LabelArm("S0_current_y_bin", "current hard TP/SL/timeout label"),
    LabelArm("S2_cost_aware_return", "future return after explicit round-trip cost"),
    LabelArm("S3_path_quality", "MFE/MAE/timing path-quality soft label"),
    LabelArm("S6_asymmetric_downside", "path quality with hard downside caps"),
    LabelArm("S7_horizon_blended", "blend hard label, TP2/SL1 path, and fast MFE"),
    LabelArm("S8_timestamp_rank_path", "timestamp-local rank of path quality"),
    LabelArm("S9_fast_mfe_3bars", "fast 1R favorable excursion within three bars"),
    LabelArm("S10_policy_net_replay", "policy replay net utility at break-even center"),
    LabelArm("S10_policy_net_soft", "policy replay net utility with softer temperature"),
    LabelArm("S10_policy_net_margin25bps", "policy replay utility requiring 25 bps edge"),
    LabelArm("S12_policy_net_clean_mild", "policy replay utility times mild path-cleanliness envelope"),
    LabelArm("S13_policy_net_risk_adjusted", "policy replay utility minus MAE/barrier/time risk"),
    LabelArm("S14_policy_net_path_blend", "blend policy replay utility with asymmetric path quality"),
    LabelArm("S15_policy_net_clean_ts_rank", "timestamp-local rank of risk-adjusted policy utility"),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sigmoid(x: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)))


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    arr = _safe_numeric(values).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    arr = _safe_numeric(values).dropna()
    return float(arr.quantile(q)) if len(arr) else float("nan")


def _safe_std(values: Any) -> float:
    arr = _safe_numeric(values).dropna()
    return float(arr.std(ddof=0)) if len(arr) else float("nan")


def _spearman(x: Any, y: Any) -> float:
    xs = _safe_numeric(x)
    ys = _safe_numeric(y)
    mask = xs.notna() & ys.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    xr = xs[mask].rank(method="average")
    yr = ys[mask].rank(method="average")
    if xr.nunique(dropna=True) < 2 or yr.nunique(dropna=True) < 2:
        return float("nan")
    return float(xr.corr(yr))


def _rank_top_indices(score: Any, frac: float) -> np.ndarray:
    score_ser = _safe_numeric(score)
    valid = score_ser.notna().to_numpy()
    if not bool(valid.any()):
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = max(1, int(math.ceil(float(frac) * len(valid_idx))))
    order = np.argsort(-score_ser.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    return valid_idx[order[:k]].astype(np.int64, copy=False)


def _effective_n(values: Iterable[Any]) -> float:
    counts = pd.Series(list(values), dtype=object).value_counts(dropna=False)
    if counts.empty:
        return 0.0
    shares = counts.to_numpy(dtype=np.float64) / float(counts.sum())
    denom = float(np.sum(shares * shares))
    return 1.0 / denom if denom > 0.0 else 0.0


def _load_labels(path: Path) -> pd.DataFrame:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet label files found under {path}")
    frames = [pd.read_parquet(file) for file in files]
    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
    if "__ts__" not in out.columns:
        raise ValueError("Label frame must include __ts__")
    if "__symbol__" not in out.columns:
        raise ValueError("Label frame must include __symbol__")
    out["__ts__"] = pd.to_datetime(out["__ts__"], errors="coerce")
    out = out.sort_values(["__ts__", "__symbol__"], kind="mergesort").reset_index(drop=True)
    return out


def _symbol_to_feature_path(feature_dir: Path, symbol: str) -> Path:
    return feature_dir / f"symbol={str(symbol).replace('/', '_')}.parquet"


def _read_feature_list(path: Path | None, *, max_features: int | None = None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        raise ValueError(f"{path} must contain a 'feature' column")
    if "used_by_model" in frame.columns:
        used = frame["used_by_model"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
        frame = frame[used].copy()
    if "selected_feature_position" in frame.columns:
        frame = frame.sort_values("selected_feature_position")
    features = [str(v) for v in frame["feature"].dropna().drop_duplicates().tolist()]
    if max_features is not None and max_features > 0:
        features = features[: int(max_features)]
    return features


def _schema_names(path: Path) -> set[str]:
    try:
        from extreme_price_movements.data_store import _feature_schema_names

        return set(str(v) for v in _feature_schema_names(str(path)))
    except Exception:
        pass
    try:
        import pyarrow.parquet as pq

        return set(str(v) for v in pq.read_schema(path).names)
    except Exception:
        try:
            return set(str(v) for v in pd.read_parquet(path).columns)
        except Exception:
            return set()


def _load_feature_store_columns(
    frame: pd.DataFrame,
    *,
    feature_dir: Path,
    selected_features: list[str],
    min_feature_finite_frac: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not selected_features:
        return pd.DataFrame(index=frame.index), {
            "enabled": False,
            "reason": "empty_feature_list",
        }
    matrix = pd.DataFrame(index=frame.index, columns=selected_features, dtype=np.float32)
    loaded_symbols = 0
    missing_symbols = 0
    read_errors: list[str] = []
    available_feature_counts: list[int] = []
    ts_utc = pd.to_datetime(frame["__ts__"], utc=True)
    from extreme_price_movements.static_feature_store import (
        STATIC_FEATURE_ENDPOINT_VERSION,
        read_static_features,
    )

    try:
        feature_store_ts = pd.to_datetime(
            feature_dir.name, format="%Y%m%d_%H%M%S", utc=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Feature directory must be a timestamped shared static store, e.g. "
            "data_perp/features/20260711_070000"
        ) from exc
    if feature_dir.parent.name != "features":
        raise ValueError(
            f"Feature directory is outside the shared static-store layout: {feature_dir}"
        )
    data_root = feature_dir.parent.parent

    # A canonical B/M/E sample can consist of three broad, disjoint windows.
    # Read each window separately so the static store can push down its time
    # bounds. Ordinary contiguous training populations remain one block.
    unique_ts = pd.DatetimeIndex(ts_utc.dropna().unique()).sort_values()
    split_points = np.flatnonzero(
        np.diff(unique_ts.asi8) > pd.Timedelta(days=3).value
    )
    block_bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if 0 < len(split_points) < 8:
        starts = np.concatenate(([0], split_points + 1))
        ends = np.concatenate((split_points, [len(unique_ts) - 1]))
        block_bounds = [
            (pd.Timestamp(unique_ts[start]), pd.Timestamp(unique_ts[end]))
            for start, end in zip(starts, ends)
        ]
    if not block_bounds:
        block_bounds = [(pd.Timestamp(ts_utc.min()), pd.Timestamp(ts_utc.max()))]
    # Compact B/M/E selection samples can safely assemble the full symbol
    # cross-section per time block.  Multi-million-row model-fit populations
    # stay bounded to avoid a wide all-symbol materialization spike.
    # A feature-major store keeps one raw array per feature and symbol in the
    # lazy reader.  Loading 64 symbols across a 500+ column AE/GMM contract can
    # exceed 16 GB before the first symbol frame is projected.  Keep compact
    # selection samples wide, but bound full-population reads aggressively.
    if len(frame) <= 500_000:
        adaptive_batch = 256
    elif len(frame) <= 2_000_000:
        adaptive_batch = 16
    else:
        adaptive_batch = 8
    batch_size = max(
        1,
        int(os.environ.get("EPM_STATIC_FEATURE_SYMBOL_BATCH", str(adaptive_batch))),
    )
    for block_start, block_end in block_bounds:
        block_mask = ts_utc.ge(block_start) & ts_utc.le(block_end)
        block_positions = np.flatnonzero(block_mask.to_numpy())
        symbol_rows = [
            (str(symbol), block_positions[np.asarray(idx, dtype=np.int64)])
            for symbol, idx in frame.iloc[block_positions].groupby("__symbol__", sort=False).indices.items()
        ]
        for batch_start in range(0, len(symbol_rows), batch_size):
            batch = symbol_rows[batch_start : batch_start + batch_size]
            existing = [
                (symbol, rows)
                for symbol, rows in batch
                if _symbol_to_feature_path(feature_dir, symbol).exists()
            ]
            missing_symbols += len(batch) - len(existing)
            if not existing:
                continue
            batch_symbols = [symbol for symbol, _rows in existing]
            batch_positions = np.concatenate([rows for _symbol, rows in existing])
            batch_ts = ts_utc.iloc[batch_positions]
            try:
                static_features = read_static_features(
                    feature_store_ts=feature_store_ts,
                    data_root=data_root,
                    feature_keys=selected_features,
                    symbols=batch_symbols,
                    start_ts=batch_ts.min(),
                    end_ts=batch_ts.max(),
                )
            except Exception as exc:
                read_errors.append(
                    f"{block_start.isoformat()}..{block_end.isoformat()}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if not hasattr(static_features, "items"):
                continue
            available = [
                feature for feature in selected_features if feature in static_features
            ]
            available_by_symbol = {symbol: 0 for symbol, _rows in existing}
            symbol_ts = {
                symbol: pd.DatetimeIndex(ts_utc.iloc[rows])
                for symbol, rows in existing
            }
            if hasattr(static_features, "symbol_frame"):
                # The canonical store is feature-major on disk, but training
                # joins need timestamp x feature rows for one symbol. Building
                # that inverse view directly avoids materializing hundreds of
                # timestamp x universe DataFrames for every monthly partition.
                for symbol, rows in existing:
                    symbol_frame = static_features.symbol_frame(
                        symbol, keys=available
                    )
                    if not isinstance(symbol_frame, pd.DataFrame) or symbol_frame.empty:
                        continue
                    symbol_frame = symbol_frame.copy(deep=False)
                    symbol_frame.index = pd.DatetimeIndex(
                        pd.to_datetime(symbol_frame.index, utc=True)
                    )
                    symbol_available = [
                        feature for feature in available if feature in symbol_frame.columns
                    ]
                    if not symbol_available:
                        continue
                    aligned = symbol_frame.reindex(symbol_ts[symbol]).loc[
                        :, symbol_available
                    ]
                    matrix.loc[rows, symbol_available] = aligned.to_numpy(
                        dtype=np.float32, copy=False
                    )
                    available_by_symbol[symbol] = len(symbol_available)
            else:
                for feature in available:
                    panel = static_features[feature]
                    if not isinstance(panel, pd.DataFrame):
                        continue
                    panel_index = pd.DatetimeIndex(pd.to_datetime(panel.index, utc=True))
                    for symbol, rows in existing:
                        if symbol not in panel.columns:
                            continue
                        values = pd.to_numeric(panel[symbol], errors="coerce")
                        values.index = panel_index
                        matrix.loc[rows, feature] = values.reindex(
                            symbol_ts[symbol]
                        ).to_numpy(dtype=np.float32, copy=False)
                        available_by_symbol[symbol] += 1
            for symbol, _rows in existing:
                symbol_available = int(available_by_symbol[symbol])
                available_feature_counts.append(symbol_available)
                if symbol_available > 0:
                    loaded_symbols += 1
    if read_errors and loaded_symbols == 0:
        raise RuntimeError(
            "All canonical static-feature reads failed; first errors: "
            + " | ".join(read_errors[:3])
        )
    finite_floor = float(min_feature_finite_frac)
    if not 0.0 <= finite_floor <= 1.0:
        raise ValueError("min_feature_finite_frac must be between zero and one")
    finite_by_feature = matrix.notna().mean().to_dict()
    retained = [
        feature
        for feature in selected_features
        if float(finite_by_feature.get(feature, 0.0) or 0.0) >= finite_floor
    ]
    matrix = matrix.loc[:, retained].copy()
    return matrix, {
        "enabled": True,
        "feature_dir": str(feature_dir),
        "reader": "static_feature_store.read_static_features.symbol_frame_preferred",
        "static_feature_endpoint_version": STATIC_FEATURE_ENDPOINT_VERSION,
        "store_access": "read_only",
        "requested_features": int(len(selected_features)),
        "retained_features": int(len(retained)),
        "loaded_symbols": int(loaded_symbols),
        "missing_symbols": int(missing_symbols),
        "read_error_count": int(len(read_errors)),
        "read_errors": read_errors[:10],
        "mean_available_features_per_symbol": (
            float(np.mean(available_feature_counts)) if available_feature_counts else 0.0
        ),
        "mean_feature_finite_frac": (
            float(np.mean([finite_by_feature[f] for f in retained])) if retained else 0.0
        ),
        "time_blocks_loaded": int(len(block_bounds)),
        "min_feature_finite_frac": (
            float(np.min([finite_by_feature[f] for f in retained])) if retained else 0.0
        ),
        "retention_finite_frac_floor": finite_floor,
    }


def _path_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    def _col(name: str) -> pd.Series:
        if name in frame.columns:
            return _safe_numeric(frame[name]).reindex(frame.index)
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)

    out = pd.DataFrame(index=frame.index)
    barrier = _safe_numeric(frame.get("__barrier_pct__")).abs().clip(lower=1e-8)
    mfe = _safe_numeric(frame.get("__mfe_ret__")).clip(lower=0.0).fillna(0.0)
    mae_raw = _safe_numeric(frame.get("__mae_ret__"))
    finite_mae = mae_raw.dropna()
    if len(finite_mae) and float(finite_mae.median()) < 0.0:
        mae = (-mae_raw).clip(lower=0.0)
        mae_encoding = "signed_negative"
    else:
        mae = mae_raw.clip(lower=0.0)
        mae_encoding = "positive_abs"
    bars_to_mfe = _safe_numeric(frame.get("__bars_to_mfe__"))
    bars_to_mae = _safe_numeric(frame.get("__bars_to_mae__"))
    bars_policy = _safe_numeric(frame.get("__bars_policy__"))
    y_ret = _safe_numeric(frame.get("__y_ret__")).fillna(0.0)
    y_bin = _safe_numeric(frame.get("__y_bin__")).fillna(0.0).clip(0.0, 1.0)
    u_policy = _safe_numeric(frame.get("__u_policy_net__")).reindex(frame.index)
    utility_source = "__u_policy_net__"
    if not u_policy.notna().any():
        u_policy = y_ret.copy()
        utility_source = "__y_ret__"
    if "side" in frame.columns:
        side_raw = frame["side"]
    elif "__side__" in frame.columns:
        side_raw = frame["__side__"]
    else:
        side_raw = pd.Series(1.0, index=frame.index)
    side = _safe_numeric(side_raw).reindex(frame.index).fillna(1.0)
    out["barrier"] = barrier.fillna(0.0)
    out["side"] = np.where(side < 0.0, -1, 1).astype(np.int8)
    out["mfe"] = mfe
    out["mae"] = mae.fillna(0.0)
    out["mfe_norm"] = out["mfe"] / barrier
    out["mae_norm"] = out["mae"] / barrier
    out["bars_to_mfe"] = bars_to_mfe.fillna(bars_policy).fillna(24.0).clip(lower=0.0)
    out["bars_to_mae"] = bars_to_mae.fillna(np.nan)
    out["bars_policy"] = bars_policy.fillna(24.0).clip(lower=0.0)
    out["return"] = y_ret
    out["ret_net"] = y_ret - ROUND_TRIP_COST
    out["y_bin"] = y_bin
    out["is_timeout"] = _safe_numeric(frame.get("__is_timeout__")).fillna(0.0) > 0.5
    out["y_outcome"] = _safe_numeric(frame.get("__y_outcome__"))
    out["u_policy_net"] = u_policy
    first_touch_hit = _col("__first_touch_hit__")
    first_touch_clean = _col("__stage167_clean_first_touch_exec__").fillna(
        _col("__stage164_primary_clean_first_touch_exec__")
    )
    first_touch_stop = _col("__first_touch_stop__")
    first_touch_timeout = _col("__stage167_first_touch_timeout__").fillna(
        _col("__first_touch_timeout__")
    )
    first_touch_bar = _col("__first_touch_bar__")
    first_touch_mae_to_sl = _col("__stage167_first_touch_mae_to_sl__").fillna(
        _col("__first_touch_mae_to_sl__")
    )
    first_touch_mfe_to_tp = _col("__first_touch_mfe_to_tp__")
    first_touch_mae_norm = _col("__first_touch_mae_norm__")
    first_touch_mfe_norm = _col("__first_touch_mfe_norm__")
    first_touch_full_mae = _col("__first_touch_full_path_mae_norm__").fillna(
        _col("__first_touch_full_path_mae_to_sl__")
    )
    first_touch_full_mfe = _col("__first_touch_full_path_mfe_norm__").fillna(
        _col("__first_touch_full_path_mfe_to_tp__")
    )
    first_touch_net = _col("__stage167_first_touch_net__").fillna(
        _col("__first_touch_capture_net__")
    )
    first_touch_round_trip_cost = _col("__first_touch_round_trip_cost__")
    first_touch_available = (
        first_touch_hit.notna()
        | first_touch_clean.notna()
        | first_touch_stop.notna()
        | first_touch_timeout.notna()
        | first_touch_mae_to_sl.notna()
        | first_touch_net.notna()
    )
    out["first_touch_available"] = first_touch_available.astype(np.int8)
    out["first_touch_hit"] = first_touch_hit.fillna(first_touch_clean).fillna(0.0).clip(0.0, 1.0)
    out["first_touch_clean_exec"] = first_touch_clean.fillna(first_touch_hit).fillna(0.0).clip(0.0, 1.0)
    out["first_touch_stop"] = first_touch_stop.fillna(0.0).clip(0.0, 1.0)
    out["first_touch_timeout"] = first_touch_timeout.fillna(out["is_timeout"].astype(float)).clip(0.0, 1.0)
    out["first_touch_bar"] = first_touch_bar.fillna(out["bars_to_mfe"]).fillna(24.0).clip(lower=0.0)
    out["first_touch_mae_to_sl"] = first_touch_mae_to_sl.fillna(out["mae_norm"]).clip(lower=0.0)
    out["first_touch_mfe_to_tp"] = first_touch_mfe_to_tp.fillna(out["mfe_norm"]).clip(lower=0.0)
    out["first_touch_mae_norm"] = first_touch_mae_norm.fillna(out["first_touch_mae_to_sl"]).clip(lower=0.0)
    out["first_touch_mfe_norm"] = first_touch_mfe_norm.fillna(out["first_touch_mfe_to_tp"]).clip(lower=0.0)
    out["first_touch_full_path_mae_norm"] = first_touch_full_mae.fillna(out["mae_norm"]).clip(lower=0.0)
    out["first_touch_full_path_mfe_norm"] = first_touch_full_mfe.fillna(out["mfe_norm"]).clip(lower=0.0)
    out["first_touch_net"] = first_touch_net.fillna(out["u_policy_net"]).fillna(0.0)
    out["round_trip_cost"] = first_touch_round_trip_cost.fillna(float(ROUND_TRIP_COST)).clip(lower=0.0)
    for name in (
        "bars_to_mfe_05r",
        "bars_to_mfe_075r",
        "bars_to_mfe_1r",
        "bars_to_mfe_125r",
        "bars_to_mfe_15r",
        "bars_to_mae_05r",
        "bars_to_mae_075r",
        "bars_to_mae_1r",
        "bars_to_mae_15r",
        "mfe_1r_before_mae_05r",
        "mfe_1r_before_mae_075r",
        "mfe_1r_before_mae_1r",
        "mae_05r_before_mfe_1r",
        "mae_075r_before_mfe_1r",
        "mae_1r_before_mfe_1r",
        "max_adverse_before_mfe_1r",
        "underwater_bars_before_mfe_1r",
        "underwater_fraction_before_mfe_1r",
        "area_underwater_before_mfe_1r",
    ):
        out[name] = _col(f"__{name}__")
    # Label-side proxy only: this is not a pre-entry feature. It quantifies the
    # amount of time spent with adverse excursion before the favorable event.
    out["underwater_bars_before_mfe_proxy"] = out["underwater_bars_before_mfe_1r"].fillna(
        pd.Series(
            np.where(
                out["mae_norm"].to_numpy(dtype=np.float64) > 0.25,
                out["bars_to_mfe"].to_numpy(dtype=np.float64),
                0.0,
            ),
            index=out.index,
        )
    )
    out.attrs["mae_encoding"] = mae_encoding
    out.attrs["utility_source"] = utility_source
    return out


def _make_targets(frame: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    current = metrics["y_bin"].astype(float)
    tp2_sl1 = ((metrics["mfe_norm"] >= 2.0) & (metrics["mae_norm"] < 1.0)).astype(float)
    fast_mfe = ((metrics["mfe_norm"] >= 1.0) & (metrics["bars_to_mfe"] <= 3.0)).astype(float)
    cost_aware = pd.Series(_sigmoid(metrics["ret_net"] / 0.006), index=frame.index)

    path_raw = (
        1.05 * metrics["mfe_norm"]
        - 1.30 * metrics["mae_norm"]
        - 0.10 * np.log1p(metrics["bars_to_mfe"])
        + 0.35 * (metrics["return"] > 0.0).astype(float)
    )
    path_quality = pd.Series(_sigmoid((path_raw - 0.25) / 1.20), index=frame.index)

    downside_raw = (
        0.90 * metrics["mfe_norm"]
        - 1.85 * metrics["mae_norm"]
        + (metrics["ret_net"] / metrics["barrier"].clip(lower=1e-8))
        - 0.15 * np.log1p(metrics["bars_to_mfe"])
    )
    asymmetric = pd.Series(_sigmoid((downside_raw - 0.10) / 1.25), index=frame.index)
    bad_path = (metrics["mae_norm"] >= 1.0) | ((metrics["y_outcome"] == 0.0).fillna(False))
    asymmetric = asymmetric.where(~bad_path, np.minimum(asymmetric, 0.25))

    blended = (0.40 * current) + (0.30 * tp2_sl1) + (0.30 * fast_mfe)

    timestamp_rank = path_quality.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    timestamp_rank = timestamp_rank.fillna(path_quality.rank(method="average", pct=True)).clip(0.0, 1.0)
    rank_path = (0.50 * path_quality) + (0.50 * timestamp_rank)

    u = metrics["u_policy_net"]
    policy_break_even = pd.Series(_sigmoid(u.fillna(-0.02) / 0.004), index=frame.index)
    policy_soft = pd.Series(_sigmoid(u.fillna(-0.02) / 0.012), index=frame.index)
    policy_margin = pd.Series(_sigmoid((u.fillna(-0.02) - 0.0025) / 0.006), index=frame.index)
    clean_mild = (
        pd.Series(_sigmoid((1.25 - metrics["mae_norm"]) / 0.35), index=frame.index)
        * pd.Series(_sigmoid((0.025 - metrics["barrier"]) / 0.006), index=frame.index)
        * pd.Series(_sigmoid((14.0 - metrics["bars_to_mfe"]) / 5.0), index=frame.index)
    ).clip(0.0, 1.0)
    policy_clean = (policy_soft * clean_mild).clip(0.0, 1.0)
    risk_adjusted_u = (
        u.fillna(-0.02)
        - 0.0040 * (metrics["mae_norm"] - 0.75).clip(lower=0.0)
        - 0.0010 * np.log1p(metrics["bars_to_mfe"].clip(lower=0.0))
        - 0.35 * (metrics["barrier"] - 0.018).clip(lower=0.0)
    )
    policy_risk_adjusted = pd.Series(_sigmoid(risk_adjusted_u / 0.008), index=frame.index)
    policy_path_blend = (0.50 * policy_soft + 0.50 * asymmetric).clip(0.0, 1.0)
    risk_ts_rank = policy_risk_adjusted.groupby(frame["__ts__"], dropna=False).rank(
        method="average",
        pct=True,
    )
    risk_ts_rank = risk_ts_rank.fillna(policy_risk_adjusted.rank(method="average", pct=True)).clip(0.0, 1.0)
    policy_clean_ts_rank = (0.50 * policy_risk_adjusted + 0.50 * risk_ts_rank).clip(0.0, 1.0)

    raw_targets = {
        "S0_current_y_bin": current,
        "S2_cost_aware_return": cost_aware,
        "S3_path_quality": path_quality,
        "S6_asymmetric_downside": asymmetric,
        "S7_horizon_blended": blended,
        "S8_timestamp_rank_path": rank_path,
        "S9_fast_mfe_3bars": fast_mfe,
        "S10_policy_net_replay": policy_break_even,
        "S10_policy_net_soft": policy_soft,
        "S10_policy_net_margin25bps": policy_margin,
        "S12_policy_net_clean_mild": policy_clean,
        "S13_policy_net_risk_adjusted": policy_risk_adjusted,
        "S14_policy_net_path_blend": policy_path_blend,
        "S15_policy_net_clean_ts_rank": policy_clean_ts_rank,
    }
    hard_targets = {
        "S0_current_y_bin": current >= 0.5,
        "S2_cost_aware_return": metrics["ret_net"] > 0.0,
        "S3_path_quality": path_quality >= 0.55,
        "S6_asymmetric_downside": asymmetric >= 0.55,
        "S7_horizon_blended": blended >= 0.50,
        "S8_timestamp_rank_path": rank_path >= 0.70,
        "S9_fast_mfe_3bars": fast_mfe >= 0.5,
        "S10_policy_net_replay": u > 0.0,
        "S10_policy_net_soft": u > 0.0,
        "S10_policy_net_margin25bps": u > 0.0025,
        "S12_policy_net_clean_mild": (u > 0.0) & (clean_mild >= 0.35),
        "S13_policy_net_risk_adjusted": risk_adjusted_u > 0.0,
        "S14_policy_net_path_blend": (u > 0.0) & (asymmetric >= 0.45),
        "S15_policy_net_clean_ts_rank": policy_clean_ts_rank >= 0.70,
    }
    out: dict[str, pd.DataFrame] = {}
    for arm in LABEL_ARMS:
        soft = _safe_numeric(raw_targets[arm.name]).clip(0.0, 1.0)
        hard = pd.Series(hard_targets[arm.name], index=frame.index).fillna(False).astype(float)
        out[arm.name] = pd.DataFrame({"target_soft": soft, "target_hard": hard}, index=frame.index)
    return out


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in FUTURE_OR_LABEL_COLUMNS:
            continue
        if col.startswith("__") and not (
            col.startswith("__regime_") or col.startswith("__meta_raw__")
        ):
            continue
        ser = _safe_numeric(frame[col])
        if ser.notna().sum() < 100:
            continue
        if col == "side":
            if ser.nunique(dropna=True) >= 2:
                cols.append(col)
            continue
        if ser.nunique(dropna=True) < 3:
            continue
        cols.append(col)
    return cols


def _decile_diagnostics(score: pd.Series, utility: pd.Series) -> dict[str, Any]:
    score = _safe_numeric(score)
    utility = _safe_numeric(utility)
    mask = score.notna() & utility.notna()
    if int(mask.sum()) < 50 or score[mask].nunique(dropna=True) < 3:
        return {
            "decile_spearman_u": float("nan"),
            "decile_violations_u": float("nan"),
            "top_decile_mean_u": float("nan"),
            "bottom_decile_mean_u": float("nan"),
            "top_bottom_decile_spread_u": float("nan"),
        }
    ranks = score[mask].rank(method="first", pct=True)
    decile = np.ceil(ranks * 10.0).clip(1, 10).astype(int)
    grouped = utility[mask].groupby(decile).mean()
    ordered = grouped.reindex(range(1, 11))
    vals = ordered.to_numpy(dtype=np.float64)
    valid = np.isfinite(vals)
    if int(valid.sum()) < 3:
        corr = float("nan")
        violations = float("nan")
    else:
        corr = _spearman(pd.Series(np.arange(1, 11)[valid]), pd.Series(vals[valid]))
        violations = int(np.sum(np.diff(vals[valid]) < 0.0))
    top = float(ordered.loc[10]) if pd.notna(ordered.loc[10]) else float("nan")
    bottom = float(ordered.loc[1]) if pd.notna(ordered.loc[1]) else float("nan")
    return {
        "decile_spearman_u": corr,
        "decile_violations_u": violations,
        "top_decile_mean_u": top,
        "bottom_decile_mean_u": bottom,
        "top_bottom_decile_spread_u": top - bottom if math.isfinite(top) and math.isfinite(bottom) else float("nan"),
    }


def _selection_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
    selected_idx: np.ndarray | None = None,
) -> dict[str, Any]:
    idx = (
        _rank_top_indices(score, top_frac)
        if selected_idx is None
        else np.asarray(selected_idx, dtype=np.int64)
    )
    selected_metrics = metrics.iloc[idx] if len(idx) else metrics.iloc[:0]
    selected_frame = frame.iloc[idx] if len(idx) else frame.iloc[:0]
    selected_target = target.iloc[idx] if len(idx) else target.iloc[:0]
    utility = selected_metrics["u_policy_net"]
    mfe_mae_ratio = (
        selected_metrics["mfe_norm"] / selected_metrics["mae_norm"].clip(lower=0.25)
        if len(selected_metrics)
        else pd.Series(dtype=float)
    ).replace([np.inf, -np.inf], np.nan).clip(upper=10.0)
    symbols = selected_frame.get("__symbol__", pd.Series(dtype=object))
    timestamps = selected_frame.get("__ts__", pd.Series(dtype="datetime64[ns]"))
    selected_side = _safe_numeric(selected_metrics.get("side")).fillna(1.0)
    selected_long_rows = int((selected_side > 0.0).sum())
    selected_short_rows = int((selected_side < 0.0).sum())
    row = {
        "arm": arm,
        "selector": selector,
        "period": period,
        "top_frac": float(top_frac),
        "rows": int(len(frame)),
        "selected_rows": int(len(idx)),
        "selected_long_rows": selected_long_rows,
        "selected_short_rows": selected_short_rows,
        "selected_long_share": float(selected_long_rows / len(idx)) if len(idx) else 0.0,
        "selected_short_share": float(selected_short_rows / len(idx)) if len(idx) else 0.0,
        "target_top_soft_mean": _safe_mean(selected_target.get("target_soft")),
        "target_top_hard_rate": _safe_mean(selected_target.get("target_hard")),
        "mean_u": _safe_mean(utility),
        "median_u": _safe_quantile(utility, 0.50),
        "q10_u": _safe_quantile(utility, 0.10),
        "hit_u": _safe_mean(utility > 0.0),
        "mean_return_net": _safe_mean(selected_metrics["ret_net"]),
        "hit_return_net": _safe_mean(selected_metrics["ret_net"] > 0.0),
        "mean_barrier": _safe_mean(selected_metrics["barrier"]),
        "p90_barrier": _safe_quantile(selected_metrics["barrier"], 0.90),
        "wide_barrier_25bps_rate": _safe_mean(selected_metrics["barrier"] > 0.025),
        "wide_barrier_35bps_rate": _safe_mean(selected_metrics["barrier"] > 0.035),
        "mean_mae_norm": _safe_mean(selected_metrics["mae_norm"]),
        "p90_mae_norm": _safe_quantile(selected_metrics["mae_norm"], 0.90),
        "bad_mae_1r_rate": _safe_mean(selected_metrics["mae_norm"] >= 1.0),
        "mean_mfe_norm": _safe_mean(selected_metrics["mfe_norm"]),
        "mean_mfe_mae_ratio": _safe_mean(mfe_mae_ratio),
        "clean_row_rate": _safe_mean(
            (selected_metrics["u_policy_net"] > 0.0)
            & (selected_metrics["mae_norm"] <= 1.0)
            & (selected_metrics["barrier"] <= 0.025)
            & (selected_metrics["is_timeout"].astype(float) <= 0.0)
        ),
        "strict_clean_row_rate": _safe_mean(
            (selected_metrics["u_policy_net"] > 0.0)
            & (selected_metrics["mae_norm"] <= 0.85)
            & (selected_metrics["barrier"] <= 0.024)
            & (mfe_mae_ratio >= 1.35)
            & (selected_metrics["is_timeout"].astype(float) <= 0.0)
        ),
        "bounded_row_rate": _safe_mean(
            (selected_metrics["u_policy_net"] > 0.0)
            & (selected_metrics["mae_norm"] <= 1.0)
            & (selected_metrics["barrier"] <= 0.035)
            & (mfe_mae_ratio >= 1.25)
            & (selected_metrics["is_timeout"].astype(float) <= 0.0)
        ),
        "mean_bars_to_mfe": _safe_mean(selected_metrics["bars_to_mfe"]),
        "p90_bars_to_mfe": _safe_quantile(selected_metrics["bars_to_mfe"], 0.90),
        "timeout_rate": _safe_mean(selected_metrics["is_timeout"].astype(float)),
        "symbol_effective_n": _effective_n(symbols),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
        "timestamp_effective_n": _effective_n(timestamps.astype(str)),
        "top_timestamp_share": float(timestamps.astype(str).value_counts(normalize=True, dropna=False).iloc[0]) if len(timestamps) else 0.0,
    }
    return row


def _weekly_selection_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    top_frac: float,
) -> list[dict[str, Any]]:
    weeks = frame["__ts__"].dt.to_period("W-SUN").astype(str)
    rows: list[dict[str, Any]] = []
    for week, ids in pd.Series(np.arange(len(frame)), index=frame.index).groupby(weeks, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        if len(pos) < 20:
            continue
        local = _selection_metrics(
            frame=frame.iloc[pos].reset_index(drop=True),
            metrics=metrics.iloc[pos].reset_index(drop=True),
            target=target.iloc[pos].reset_index(drop=True),
            score=score.iloc[pos].reset_index(drop=True),
            arm=arm,
            selector=selector,
            period=str(week),
            top_frac=top_frac,
        )
        rows.append(local)
    return rows


def _feature_ic(frame: pd.DataFrame, features: list[str], target: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        ic = _spearman(frame[feature], target)
        if not math.isfinite(ic):
            continue
        rows.append({"feature": feature, "ic": ic, "abs_ic": abs(ic)})
    return pd.DataFrame(rows).sort_values("abs_ic", ascending=False) if rows else pd.DataFrame()


def _proxy_score(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], y_train: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    ic = _feature_ic(train, features, y_train)
    if ic.empty:
        return pd.Series(np.nan, index=valid.index), {"proxy_features": [], "proxy_top_abs_ic": float("nan")}
    chosen = ic.head(PROXY_TOP_K_FEATURES).copy()
    parts: list[pd.Series] = []
    for _, row in chosen.iterrows():
        feature = str(row["feature"])
        sign = 1.0 if float(row["ic"]) >= 0.0 else -1.0
        ranks = _safe_numeric(valid[feature]).rank(method="average", pct=True)
        if sign < 0.0:
            ranks = 1.0 - ranks
        parts.append(ranks.fillna(0.5))
    score = pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(np.nan, index=valid.index)
    diag = {
        "proxy_features": chosen["feature"].astype(str).tolist(),
        "proxy_top_abs_ic": float(chosen["abs_ic"].iloc[0]) if len(chosen) else float("nan"),
        "proxy_mean_top_abs_ic": float(chosen["abs_ic"].mean()) if len(chosen) else float("nan"),
    }
    return score.reindex(valid.index), diag


def _proxy_oos_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    arm: str,
) -> list[dict[str, Any]]:
    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    for month in months[1:]:
        train_mask = frame["__ts__"].dt.to_period("M").astype(str) < month
        valid_mask = frame["__ts__"].dt.to_period("M").astype(str) == month
        if int(train_mask.sum()) < 100 or int(valid_mask.sum()) < 50:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        valid_target = target.loc[valid_mask].copy().reset_index(drop=True)
        period_mean_u = _safe_mean(valid_metrics["u_policy_net"])
        period_hit_u = _safe_mean(valid_metrics["u_policy_net"] > 0.0)
        period_q10_u = _safe_quantile(valid_metrics["u_policy_net"], 0.10)
        score, diag = _proxy_score(train, valid, features, target.loc[train_mask, "target_soft"])
        score = score.reset_index(drop=True)
        valid_reset = valid.reset_index(drop=True)
        for frac in TOP_FRACS:
            row = _selection_metrics(
                frame=valid_reset,
                metrics=valid_metrics,
                target=valid_target,
                score=score,
                arm=arm,
                selector="feature_ic_proxy_oos",
                period=month,
                top_frac=frac,
            )
            row.update(
                {
                    "period_baseline_mean_u": period_mean_u,
                    "period_baseline_hit_u": period_hit_u,
                    "period_baseline_q10_u": period_q10_u,
                    "delta_mean_u_vs_period": (
                        float(row["mean_u"] - period_mean_u)
                        if math.isfinite(float(row["mean_u"])) and math.isfinite(period_mean_u)
                        else float("nan")
                    ),
                    "delta_hit_u_vs_period": (
                        float(row["hit_u"] - period_hit_u)
                        if math.isfinite(float(row["hit_u"])) and math.isfinite(period_hit_u)
                        else float("nan")
                    ),
                    "delta_q10_u_vs_period": (
                        float(row["q10_u"] - period_q10_u)
                        if math.isfinite(float(row["q10_u"])) and math.isfinite(period_q10_u)
                        else float("nan")
                    ),
                    "proxy_ic_soft": _spearman(score, valid_target["target_soft"]),
                    "proxy_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                    "proxy_features": ",".join(diag.get("proxy_features", [])),
                    "proxy_top_abs_ic": diag.get("proxy_top_abs_ic"),
                    "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic"),
                }
            )
            rows.append(row)
    return rows


def _summarise_arm(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    arm: str,
) -> dict[str, Any]:
    soft = target["target_soft"]
    hard = target["target_hard"]
    utility = metrics["u_policy_net"]
    ic = _feature_ic(frame, features, soft)
    top_ic = ic.head(PROXY_TOP_K_FEATURES)
    base = {
        "arm": arm,
        "rows": int(len(frame)),
        "finite_soft_frac": float(soft.notna().mean()) if len(soft) else float("nan"),
        "soft_mean": _safe_mean(soft),
        "soft_std": _safe_std(soft),
        "soft_p10": _safe_quantile(soft, 0.10),
        "soft_p50": _safe_quantile(soft, 0.50),
        "soft_p90": _safe_quantile(soft, 0.90),
        "soft_low_sat_rate": _safe_mean(soft <= 0.05),
        "soft_high_sat_rate": _safe_mean(soft >= 0.95),
        "hard_rate": _safe_mean(hard),
        "ic_soft_vs_u": _spearman(soft, utility),
        "ic_soft_vs_ret_net": _spearman(soft, metrics["ret_net"]),
        "ic_soft_vs_mae_norm": _spearman(soft, metrics["mae_norm"]),
        "ic_soft_vs_mfe_norm": _spearman(soft, metrics["mfe_norm"]),
        "feature_count": int(len(features)),
        "feature_top_abs_ic": float(top_ic["abs_ic"].iloc[0]) if len(top_ic) else float("nan"),
        "feature_mean_top_abs_ic": float(top_ic["abs_ic"].mean()) if len(top_ic) else float("nan"),
        "feature_n_abs_ic_ge_002": int((ic["abs_ic"] >= 0.02).sum()) if not ic.empty else 0,
        "feature_n_abs_ic_ge_005": int((ic["abs_ic"] >= 0.05).sum()) if not ic.empty else 0,
        "feature_top_names": ",".join(top_ic["feature"].astype(str).tolist()) if len(top_ic) else "",
    }
    base.update(_decile_diagnostics(soft, utility))
    return base


def _side_name_series(metrics: pd.DataFrame) -> pd.Series:
    side = (
        _safe_numeric(metrics["side"]).fillna(1.0)
        if "side" in metrics.columns
        else pd.Series(1.0, index=metrics.index)
    )
    return pd.Series(np.where(side < 0.0, "short", "long"), index=metrics.index)


def _side_summary_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    features: list[str],
) -> list[dict[str, Any]]:
    side_names = _side_name_series(metrics)
    rows: list[dict[str, Any]] = []
    for arm in LABEL_ARMS:
        target = targets[arm.name]
        for side_name in ("long", "short"):
            mask = side_names.eq(side_name)
            if int(mask.sum()) < 20:
                continue
            local_frame = frame.loc[mask].reset_index(drop=True)
            local_metrics = metrics.loc[mask].reset_index(drop=True)
            local_target = target.loc[mask].reset_index(drop=True)
            row = _summarise_arm(
                frame=local_frame,
                metrics=local_metrics,
                target=local_target,
                features=features,
                arm=arm.name,
            )
            row["side_name"] = side_name
            row["mean_u_all"] = _safe_mean(local_metrics["u_policy_net"])
            row["q10_u_all"] = _safe_quantile(local_metrics["u_policy_net"], 0.10)
            row["hit_u_all"] = _safe_mean(local_metrics["u_policy_net"] > 0.0)
            row["bad_mae_1r_rate_all"] = _safe_mean(local_metrics["mae_norm"] >= 1.0)
            for frac, prefix in ((0.30, "top30"), (0.10, "top10")):
                selected = _selection_metrics(
                    frame=local_frame,
                    metrics=local_metrics,
                    target=local_target,
                    score=local_target["target_soft"],
                    arm=arm.name,
                    selector=f"oracle_label_sort_{side_name}",
                    period=f"{side_name}_all",
                    top_frac=frac,
                )
                for key in (
                    "selected_rows",
                    "mean_u",
                    "q10_u",
                    "hit_u",
                    "bad_mae_1r_rate",
                    "mean_bars_to_mfe",
                    "top_symbol_share",
                ):
                    row[f"{prefix}_{key}"] = selected.get(key)
            rows.append(row)
    return rows


def _aggregate_proxy(proxy: pd.DataFrame) -> pd.DataFrame:
    if proxy.empty:
        return proxy
    rows: list[dict[str, Any]] = []
    for key, group in proxy.groupby(["arm", "top_frac"], dropna=False, observed=True):
        arm, frac = key
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        rows.append(
            {
                "arm": arm,
                "top_frac": float(frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "mean_u": float(mean_u.mean()) if len(mean_u) else float("nan"),
                "worst_month_mean_u": float(mean_u.min()) if len(mean_u) else float("nan"),
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "delta_mean_u_vs_period": _safe_mean(group.get("delta_mean_u_vs_period")),
                "delta_hit_u_vs_period": _safe_mean(group.get("delta_hit_u_vs_period")),
                "delta_q10_u_vs_period": _safe_mean(group.get("delta_q10_u_vs_period")),
                "proxy_ic_soft": _safe_mean(group["proxy_ic_soft"]),
                "proxy_ic_u": _safe_mean(group["proxy_ic_u"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["top_frac", "mean_u"], ascending=[True, False])


def _write_markdown(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    oracle: pd.DataFrame,
    proxy_agg: pd.DataFrame,
    weekly: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_quality_proxy_diagnostics.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    oracle_top30 = oracle[oracle["top_frac"].eq(0.30)].sort_values(
        ["mean_u", "q10_u"], ascending=[False, False]
    )
    oracle_top10 = oracle[oracle["top_frac"].eq(0.10)].sort_values(
        ["mean_u", "q10_u"], ascending=[False, False]
    )
    proxy_top30 = proxy_agg[proxy_agg["top_frac"].eq(0.30)].sort_values(
        ["mean_u", "worst_month_mean_u"], ascending=[False, False]
    )
    proxy_top10 = proxy_agg[proxy_agg["top_frac"].eq(0.10)].sort_values(
        ["mean_u", "worst_month_mean_u"], ascending=[False, False]
    )
    side_top30 = (
        side_summary.sort_values(["top30_mean_u", "top30_q10_u"], ascending=[False, False])
        if not side_summary.empty
        else pd.DataFrame()
    )

    weekly_top30 = weekly[weekly["top_frac"].eq(0.30)].copy()
    weekly_agg = (
        weekly_top30.groupby("arm", observed=True)
        .agg(
            weeks=("period", "nunique"),
            positive_weeks=("mean_u", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
            q25_week_mean_u=("mean_u", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.25))),
            worst_week_mean_u=("mean_u", lambda s: float(pd.to_numeric(s, errors="coerce").min())),
        )
        .reset_index()
        .sort_values(["q25_week_mean_u", "worst_week_mean_u"], ascending=[False, False])
        if not weekly_top30.empty
        else pd.DataFrame()
    )

    lines = [
        "# Label Quality Proxy Diagnostics",
        "",
        "Scope: proxy diagnostics only. No LightGBM, Optuna, or policy geometry fitting is performed.",
        f"Utility source: `{manifest.get('utility_source', '')}`.",
        "",
        "Interpretation: `oracle_label_sort` tests whether the label ranks economically good rows. `feature_ic_proxy_oos` tests whether simple feature associations learned on earlier months transfer to later months.",
        "",
        "## Label Shape And Feature Association",
        "",
        table(
            summary.sort_values(["feature_mean_top_abs_ic", "ic_soft_vs_u"], ascending=[False, False]),
            [
                "arm",
                "soft_mean",
                "soft_std",
                "soft_low_sat_rate",
                "soft_high_sat_rate",
                "hard_rate",
                "ic_soft_vs_u",
                "decile_spearman_u",
                "decile_violations_u",
                "feature_mean_top_abs_ic",
                "feature_n_abs_ic_ge_002",
                "feature_top_names",
            ],
        ),
        "",
        "## Oracle Label Sort Top 30%",
        "",
        table(
            oracle_top30,
            [
                "arm",
                "selected_rows",
                "selected_long_rows",
                "selected_short_rows",
                "mean_u",
                "q10_u",
                "hit_u",
                "mean_return_net",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
                "mean_bars_to_mfe",
                "top_symbol_share",
            ],
        ),
        "",
        "## Oracle Label Sort Top 10%",
        "",
        table(
            oracle_top10,
            [
                "arm",
                "selected_rows",
                "selected_long_rows",
                "selected_short_rows",
                "mean_u",
                "q10_u",
                "hit_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
                "mean_bars_to_mfe",
                "top_symbol_share",
            ],
        ),
        "",
        "## Side-Aware Long/Short Snapshot",
        "",
        table(
            side_top30,
            [
                "arm",
                "side_name",
                "rows",
                "mean_u_all",
                "q10_u_all",
                "hit_u_all",
                "bad_mae_1r_rate_all",
                "top30_selected_rows",
                "top30_mean_u",
                "top30_q10_u",
                "top30_hit_u",
                "top30_bad_mae_1r_rate",
                "top10_mean_u",
                "top10_q10_u",
                "top10_hit_u",
            ],
        ),
        "",
        "## Feature-IC Proxy OOS Top 30%",
        "",
        table(
            proxy_top30,
            [
                "arm",
                "months",
                "positive_months",
                "mean_u",
                "worst_month_mean_u",
                "hit_u",
                "q10_u",
                "delta_mean_u_vs_period",
                "delta_hit_u_vs_period",
                "proxy_ic_soft",
                "proxy_ic_u",
                "bad_mae_1r_rate",
                "top_symbol_share",
            ],
        ),
        "",
        "## Feature-IC Proxy OOS Top 10%",
        "",
        table(
            proxy_top10,
            [
                "arm",
                "months",
                "positive_months",
                "mean_u",
                "worst_month_mean_u",
                "hit_u",
                "q10_u",
                "delta_mean_u_vs_period",
                "delta_hit_u_vs_period",
                "proxy_ic_soft",
                "proxy_ic_u",
                "bad_mae_1r_rate",
                "top_symbol_share",
            ],
        ),
        "",
        "## Weekly Oracle Top 30% Stability",
        "",
        table(
            weekly_agg,
            ["arm", "weeks", "positive_weeks", "q25_week_mean_u", "worst_week_mean_u"],
        ),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Side summary: `{manifest['outputs']['side_summary']}`",
        f"- Oracle selection metrics: `{manifest['outputs']['oracle_selection']}`",
        f"- Weekly oracle metrics: `{manifest['outputs']['weekly_oracle_selection']}`",
        f"- Feature-IC metrics: `{manifest['outputs']['feature_ic']}`",
        f"- Feature-IC proxy OOS: `{manifest['outputs']['feature_ic_proxy_oos']}`",
        f"- Feature-IC proxy OOS aggregate: `{manifest['outputs']['feature_ic_proxy_oos_aggregate']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnostics(
    labels_path: Path,
    output_dir: Path,
    *,
    feature_dir: Path | None = None,
    feature_list_csv: Path | None = None,
    max_feature_store_features: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_store_features = _read_feature_list(
        feature_list_csv,
        max_features=max_feature_store_features,
    )
    feature_store_report: dict[str, Any] = {"enabled": False}
    if feature_dir is not None and selected_store_features:
        feature_matrix, feature_store_report = _load_feature_store_columns(
            frame,
            feature_dir=feature_dir,
            selected_features=selected_store_features,
        )
        for col in feature_matrix.columns:
            frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    metrics = _path_metrics(frame)
    targets = _make_targets(frame, metrics)
    features = _feature_columns(frame)

    summary_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    feature_ic_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []

    for arm in LABEL_ARMS:
        target = targets[arm.name]
        summary_rows.append(
            _summarise_arm(
                frame=frame,
                metrics=metrics,
                target=target,
                features=features,
                arm=arm.name,
            )
        )
        ic = _feature_ic(frame, features, target["target_soft"])
        if not ic.empty:
            ic.insert(0, "arm", arm.name)
            feature_ic_rows.extend(ic.head(25).to_dict("records"))
        for frac in TOP_FRACS:
            oracle_rows.append(
                _selection_metrics(
                    frame=frame,
                    metrics=metrics,
                    target=target,
                    score=target["target_soft"],
                    arm=arm.name,
                    selector="oracle_label_sort",
                    period="all",
                    top_frac=frac,
                )
            )
        weekly_rows.extend(
            _weekly_selection_rows(
                frame=frame,
                metrics=metrics,
                target=target,
                score=target["target_soft"],
                arm=arm.name,
                selector="oracle_label_sort",
                top_frac=0.30,
            )
        )
        proxy_rows.extend(
            _proxy_oos_rows(
                frame=frame,
                metrics=metrics,
                target=target,
                features=features,
                arm=arm.name,
            )
        )

    summary = pd.DataFrame(summary_rows)
    side_summary = pd.DataFrame(
        _side_summary_rows(
            frame=frame,
            metrics=metrics,
            targets=targets,
            features=features,
        )
    )
    oracle = pd.DataFrame(oracle_rows)
    weekly = pd.DataFrame(weekly_rows)
    feature_ic = pd.DataFrame(feature_ic_rows)
    proxy = pd.DataFrame(proxy_rows)
    proxy_agg = _aggregate_proxy(proxy)

    paths = {
        "summary": output_dir / "label_quality_summary.csv",
        "side_summary": output_dir / "label_side_quality_summary.csv",
        "oracle_selection": output_dir / "label_oracle_selection_metrics.csv",
        "weekly_oracle_selection": output_dir / "label_weekly_oracle_selection_metrics.csv",
        "feature_ic": output_dir / "label_feature_ic_top25.csv",
        "feature_ic_proxy_oos": output_dir / "label_feature_ic_proxy_oos_monthly.csv",
        "feature_ic_proxy_oos_aggregate": output_dir / "label_feature_ic_proxy_oos_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    side_summary.to_csv(paths["side_summary"], index=False)
    oracle.to_csv(paths["oracle_selection"], index=False)
    weekly.to_csv(paths["weekly_oracle_selection"], index=False)
    feature_ic.to_csv(paths["feature_ic"], index=False)
    proxy.to_csv(paths["feature_ic_proxy_oos"], index=False)
    proxy_agg.to_csv(paths["feature_ic_proxy_oos_aggregate"], index=False)

    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "side_counts": _side_name_series(metrics).value_counts(dropna=False).to_dict(),
        "features": features,
        "feature_count": int(len(features)),
        "feature_store": feature_store_report,
        "feature_list_csv": str(feature_list_csv) if feature_list_csv is not None else "",
        "max_feature_store_features": max_feature_store_features,
        "mae_encoding": metrics.attrs.get("mae_encoding"),
        "utility_source": metrics.attrs.get("utility_source"),
        "label_arms": [arm.__dict__ for arm in LABEL_ARMS],
        "top_fracs": list(TOP_FRACS),
        "round_trip_cost": ROUND_TRIP_COST,
        "proxy_top_k_features": PROXY_TOP_K_FEATURES,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(
        output_dir=output_dir,
        summary=summary,
        side_summary=side_summary,
        oracle=oracle,
        proxy_agg=proxy_agg,
        weekly=weekly,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--feature-list-csv", type=Path, default=None)
    parser.add_argument("--use-default-feature-store", action="store_true")
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_dir = args.feature_dir
    feature_list_csv = args.feature_list_csv
    if args.use_default_feature_store:
        feature_dir = feature_dir or DEFAULT_FEATURE_DIR
        feature_list_csv = feature_list_csv or DEFAULT_FEATURE_LIST_CSV
    manifest = run_diagnostics(
        args.labels_path,
        args.output_dir,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
