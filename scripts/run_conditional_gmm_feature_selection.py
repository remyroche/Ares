#!/usr/bin/env python3
"""Conditional side-aware feature selection for GMM predictability labels.

The pipeline deliberately selects from existing columns only. It never creates
new live features; future-looking targets are used only for offline scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TARGET_NAMES = (
    "side_adjusted_return",
    "utility",
    "risk_adjusted_utility",
    "bad_MAE",
    "timeout",
    "adverse_excursion",
    "favorable_excursion",
    "lower_tail_utility",
)
RISK_TARGETS = {"bad_MAE", "timeout", "adverse_excursion", "lower_tail_utility"}
MANDATORY_FAMILIES = (
    "momentum_trend",
    "reversion_extension",
    "volatility",
    "vol_of_vol",
    "volume",
    "open_interest",
    "funding",
    "liquidity_impact",
    "cross_sectional",
    "cross_asset",
    "market",
    "entropy",
    "setup",
    "side_aware",
)
EXCLUDED_EXACT = {
    "__ts__",
    "__symbol__",
    "candidate_id",
    "side_name",
    "timeframe",
    "asset",
    "symbol",
    "timestamp",
}
LEAKY_NAME_TOKENS = (
    "future",
    "forward",
    "lookahead",
    "target",
    "label",
    "outcome",
    "realized",
    "realised",
    "pnl",
    "utility",
    "policy",
    "mfe",
    "mae",
    "barrier",
)


@dataclass(frozen=True)
class ConditionalFeatureSelectionConfig:
    horizon_min_hours: float = 3.0
    horizon_max_hours: float = 7.0
    min_feature_finite_frac: float = 0.95
    min_unique_values: int = 5
    min_bucket_rows: int = 200
    min_side_rows: int = 100
    shrinkage_k: float = 500.0
    ic_threshold: float = 0.02
    corr_threshold: float = 0.95
    max_input_features: int = 300
    max_selected_pairs: int = 120
    max_selected_features: int = 80
    max_corr_rows: int = 20000
    min_horizon_relevance: float = 0.10
    max_abs_value: float = 1e6
    max_abs_p99: float = 1e5
    hard_max_lookback_hours: float | None = None
    require_both_sides: bool = True
    random_seed: int = 42


@dataclass
class ConditionalSelectionResult:
    feature_validity: pd.DataFrame
    bucket_ic_matrix: pd.DataFrame
    pair_scores: pd.DataFrame
    selected_pairs: pd.DataFrame
    selected_features: pd.DataFrame
    signature_columns: pd.DataFrame
    target_report: dict[str, Any]


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


def _safe_numeric(values: Any, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _load_labels(path: Path) -> pd.DataFrame:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet label files found under {path}")
    frames = [pd.read_parquet(file) for file in files]
    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
    if "__ts__" not in out.columns or "__symbol__" not in out.columns:
        raise ValueError("Label frame must include __ts__ and __symbol__")
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out = out.sort_values(["__ts__", "__symbol__"], kind="mergesort").reset_index(drop=True)
    return out


def _read_feature_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        raise ValueError(f"{path} must contain a 'feature' column")
    return [str(v) for v in frame["feature"].dropna().drop_duplicates().tolist()]


def _load_target_readiness(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None:
        return {"enabled": False, "reason": "not_provided"}
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    selected = summary.get("selected", {}) if isinstance(summary, dict) else {}
    gates = summary.get("selection_gates", {}) if isinstance(summary, dict) else {}
    selected_spec = summary.get("selected_spec", {}) if isinstance(summary, dict) else {}
    proxy_mean = float(selected.get("proxy_top10_mean_net", float("nan")))
    proxy_ic_net = float(selected.get("proxy_top10_ic_net", float("nan")))
    proxy_q10 = float(selected.get("proxy_top10_q10_net", float("nan")))
    proxy_hit = float(selected.get("proxy_top10_hit_net", float("nan")))
    passed_strict = bool(gates.get("require_proxy_positive_net", False))
    if passed_strict:
        passed_strict = (
            math.isfinite(proxy_mean)
            and proxy_mean >= float(gates.get("min_proxy_mean_net", 0.0))
            and math.isfinite(proxy_ic_net)
            and proxy_ic_net >= float(gates.get("min_proxy_ic_net", 0.0))
            and (not math.isfinite(proxy_hit) or proxy_hit >= float(gates.get("min_proxy_hit_net", 0.0)))
            and (not math.isfinite(proxy_q10) or proxy_q10 >= float(gates.get("min_proxy_q10_net", float("-inf"))))
        )
    weak_reasons: list[str] = []
    if math.isfinite(proxy_mean) and proxy_mean < 0.002:
        weak_reasons.append("proxy_top10_mean_net_below_20bps")
    if math.isfinite(proxy_ic_net) and proxy_ic_net < 0.05:
        weak_reasons.append("proxy_top10_ic_net_below_0p05")
    if math.isfinite(proxy_q10) and proxy_q10 < 0.0:
        weak_reasons.append("proxy_top10_q10_net_negative")
    promotion_status = "experimental"
    if passed_strict and not weak_reasons:
        promotion_status = "strict_gate_passed"
    elif not passed_strict:
        promotion_status = "not_ready"
    return {
        "enabled": True,
        "summary_path": str(summary_path),
        "promotion_status": promotion_status,
        "passed_strict_positive_net_gate": bool(passed_strict),
        "weak_reasons": weak_reasons,
        "selected_spec": selected_spec,
        "selected_metrics": {
            "objective": selected.get("objective"),
            "proxy_top10_mean_net": selected.get("proxy_top10_mean_net"),
            "proxy_top10_delta_mean": selected.get("proxy_top10_delta_mean"),
            "proxy_top10_hit_net": selected.get("proxy_top10_hit_net"),
            "proxy_top10_q10_net": selected.get("proxy_top10_q10_net"),
            "proxy_top10_ic_net": selected.get("proxy_top10_ic_net"),
            "proxy_top10_ic_soft": selected.get("proxy_top10_ic_soft"),
            "oracle_top10_mean_net": selected.get("oracle_top10_mean_net"),
            "hard_rate": selected.get("hard_rate"),
            "feasible_rate": selected.get("feasible_rate"),
        },
        "selection_gates": gates,
    }


def _schema_names(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(str(v) for v in pq.read_schema(path).names)
    except Exception:
        return set(str(v) for v in pd.read_parquet(path).columns)


def _symbol_to_feature_path(feature_dir: Path, symbol: str) -> Path:
    return feature_dir / f"symbol={str(symbol).replace('/', '_')}.parquet"


def _infer_feature_store_columns(feature_dir: Path, max_files: int = 5) -> list[str]:
    columns: set[str] = set()
    for path in sorted(feature_dir.glob("symbol=*.parquet"))[: max(1, int(max_files))]:
        columns.update(_schema_names(path))
    return sorted(columns)


def _load_feature_store_columns(
    frame: pd.DataFrame,
    *,
    feature_dir: Path,
    selected_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not selected_features:
        return pd.DataFrame(index=frame.index), {"enabled": False, "reason": "empty_feature_list"}
    matrix = pd.DataFrame(index=frame.index, columns=selected_features, dtype=np.float32)
    loaded_symbols = 0
    missing_symbols = 0
    ts_utc = pd.to_datetime(frame["__ts__"], utc=True)
    available_counts: list[int] = []
    for symbol, idx in frame.groupby("__symbol__", sort=False).indices.items():
        rows = np.asarray(idx, dtype=np.int64)
        path = _symbol_to_feature_path(feature_dir, str(symbol))
        if not path.exists():
            missing_symbols += 1
            continue
        names = _schema_names(path)
        available = [feature for feature in selected_features if feature in names]
        available_counts.append(len(available))
        if not available:
            continue
        try:
            features = pd.read_parquet(path, columns=available)
        except Exception:
            continue
        available = [feature for feature in available if feature in features.columns]
        if not available:
            continue
        for feature in available:
            features[feature] = pd.to_numeric(features[feature], errors="coerce")
        features.index = pd.to_datetime(features.index, utc=True)
        aligned = features.reindex(ts_utc.iloc[rows])
        matrix.loc[rows, available] = aligned.to_numpy(dtype=np.float32, copy=False)
        loaded_symbols += 1
    return matrix, {
        "enabled": True,
        "feature_dir": str(feature_dir),
        "requested_features": int(len(selected_features)),
        "loaded_symbols": int(loaded_symbols),
        "missing_symbols": int(missing_symbols),
        "mean_available_features_per_symbol": (
            float(np.mean(available_counts)) if available_counts else 0.0
        ),
    }


def _normalise_side(frame: pd.DataFrame) -> pd.Series:
    raw = frame.get("side", frame.get("__side__"))
    side = _safe_numeric(raw, index=frame.index).fillna(1.0)
    return pd.Series(np.where(side < 0.0, -1, 1), index=frame.index, dtype=np.int8)


def _existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            return _safe_numeric(frame[name], index=frame.index)
    return None


def build_side_aware_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    side = _normalise_side(frame)
    raw_ret = _existing_column(frame, ("side_adjusted_return", "__y_ret__", "y_ret"))
    if raw_ret is None:
        raise ValueError("Need side_adjusted_return or __y_ret__ to build targets")
    utility = _existing_column(
        frame,
        (
            "utility",
            "__u_econ_net__",
            "u_econ_net",
            "__u_policy_net__",
            "u_policy_net",
        ),
    )
    if utility is None:
        utility = raw_ret.copy()
    risk_adjusted = _existing_column(
        frame,
        (
            "risk_adjusted_utility",
            "__u_econ_adjusted_net__",
            "u_econ_adjusted_net",
            "__r_policy_net__",
            "r_policy_net",
        ),
    )
    barrier = _safe_numeric(frame.get("__barrier_pct__"), index=frame.index).abs().replace(0.0, np.nan)
    mae = _safe_numeric(frame.get("adverse_excursion", frame.get("__mae_ret__")), index=frame.index).abs()
    mfe = _safe_numeric(frame.get("favorable_excursion", frame.get("__mfe_ret__")), index=frame.index).abs()
    mae_norm = (mae / barrier).replace([np.inf, -np.inf], np.nan)
    if risk_adjusted is None:
        timeout_proxy = _safe_numeric(frame.get("__is_timeout__"), index=frame.index).fillna(0.0)
        risk_adjusted = utility - 0.0025 * mae_norm.fillna(0.0) - 0.0010 * timeout_proxy
    bad_mae = _existing_column(frame, ("bad_MAE", "bad_mae"))
    if bad_mae is None:
        bad_mae = (mae_norm >= 1.0).astype(float)
    timeout = _existing_column(frame, ("timeout", "__is_timeout__"))
    if timeout is None:
        outcome = _safe_numeric(frame.get("__y_outcome__"), index=frame.index)
        timeout = outcome.eq(1.0).astype(float)
    q20 = float(np.nanquantile(utility.to_numpy(dtype=np.float64), 0.20)) if utility.notna().any() else 0.0
    lower_tail = np.minimum(utility - q20, 0.0)
    targets = pd.DataFrame(
        {
            "side_adjusted_return": raw_ret,
            "utility": utility,
            "risk_adjusted_utility": risk_adjusted,
            "bad_MAE": bad_mae,
            "timeout": timeout,
            "adverse_excursion": mae,
            "favorable_excursion": mfe,
            "lower_tail_utility": lower_tail,
        },
        index=frame.index,
    )
    report = {
        "target_columns": {
            name: {
                "finite_frac": float(targets[name].notna().mean()) if len(targets) else 0.0,
                "mean": float(targets[name].mean()) if targets[name].notna().any() else float("nan"),
            }
            for name in TARGET_NAMES
        },
        "side_counts": {
            "long": int((side > 0).sum()),
            "short": int((side < 0).sum()),
        },
    }
    return targets.astype(np.float32), report


def classify_feature_family(name: str) -> str:
    lowered = str(name).lower()
    if lowered == "side" or lowered.startswith(("side_", "long_short_", "short_long_")):
        return "side_aware"
    if lowered.startswith(("xs_", "cs_")) or "__xs_" in lowered or "__cs_" in lowered:
        return "cross_sectional"
    if re.match(r"^(up|dn|down)_vol(?:_\d+)?$", lowered):
        return "volume"
    if any(token in lowered for token in ("xasset", "btc_", "eth_", "btceth", "peer_resid", "symbol_minus_mkt", "asset_minus_mkt")):
        return "cross_asset"
    if any(token in lowered for token in ("market_", "mkt_", "_mkt", "breadth", "bench", "benchmark")):
        return "market"
    if any(token in lowered for token in ("funding", "premium", "basis")):
        return "funding"
    if any(token in lowered for token in ("open_interest", "oi_", "_oi", "oi_change")):
        return "open_interest"
    if any(token in lowered for token in ("spread", "amihud", "impact", "liquid", "slippage", "depth")):
        return "liquidity_impact"
    if any(token in lowered for token in ("entropy", "spectral", "eig_effective_rank")):
        return "entropy"
    if any(token in lowered for token in ("vol_of_vol", "vov", "volatility_of_volatility")):
        return "vol_of_vol"
    if "volume" in lowered or "vol_z" in lowered or "rvol" in lowered:
        return "volume"
    if any(token in lowered for token in ("atr", "rv_", "realized_vol", "variance", "range", "volatility")):
        return "volatility"
    if any(
        token in lowered
        for token in ("ret", "mom", "trend", "slope", "ema", "breakout", "zr_", "adx", "di_plus", "di_minus")
    ):
        return "momentum_trend"
    if any(token in lowered for token in ("dist", "vwap", "zscore", "pullback", "loc_", "extension")):
        return "reversion_extension"
    if any(token in lowered for token in ("rank", "percentile", "pctile", "cross", "cs_", "xs_")):
        return "cross_sectional"
    if any(token in lowered for token in ("setup", "trigger", "signal", "quality")):
        return "setup"
    if any(token in lowered for token in ("sin_", "cos_", "hod", "dow", "hour", "dayofweek")):
        return "calendar"
    return "other"


def infer_lookback_hours(name: str) -> float | None:
    lowered = str(name).lower()
    values: list[float] = []
    for num, unit in re.findall(r"(?<!\d)(\d{1,4})(m|min|h|hr|hour|d|day)s?(?![a-z])", lowered):
        value = float(num)
        if unit in {"m", "min"}:
            value /= 60.0
        elif unit in {"d", "day"}:
            value *= 24.0
        values.append(value)
    no_unit_patterns = (
        r"(?:^|_)(?:adx|rsi|atr|rv|ret|vol|range|std)_(\d{1,3})(?:_|$)",
        r"(?:^|_)(?:ema|sma|ma)(\d{1,3})(?:_|$)",
        r"(?:^|_)(?:up_vol|dn_vol|down_vol)_(\d{1,3})(?:_|$)",
        r"(?:^|_)zscore_price_(\d{1,3})(?:_|$)",
    )
    for pattern in no_unit_patterns:
        for value in re.findall(pattern, lowered):
            hours = float(value)
            if 1.0 <= hours <= 240.0:
                values.append(hours)
    if "prior_day" in lowered:
        values.append(24.0)
    if not values:
        family = classify_feature_family(lowered)
        if family in {
            "momentum_trend",
            "reversion_extension",
            "volatility",
            "vol_of_vol",
            "volume",
            "open_interest",
            "funding",
            "liquidity_impact",
        }:
            for value in re.findall(r"_(\d{1,3})(?:_|$)", lowered):
                hours = float(value)
                if 1.0 <= hours <= 240.0:
                    values.append(hours)
    return max(values) if values else None


def horizon_relevance_score(
    lookback_hours: float | None,
    *,
    horizon_min_hours: float,
    horizon_max_hours: float,
) -> float:
    if lookback_hours is None:
        return 0.75
    mid = max((float(horizon_min_hours) + float(horizon_max_hours)) / 2.0, 1e-6)
    lookback = max(float(lookback_hours), 1e-6)
    score = math.exp(-abs(math.log(lookback / mid)) / 1.75)
    if lookback <= float(horizon_max_hours) * 12.0:
        score = max(score, 0.35)
    return float(max(0.0, min(1.0, score)))


def _is_excluded_feature_name(name: str) -> tuple[bool, str]:
    lowered = str(name).lower()
    if lowered in EXCLUDED_EXACT:
        return True, "metadata_column"
    if lowered.startswith("__") and not (
        lowered.startswith("__regime_") or lowered.startswith("__meta_raw__")
    ):
        return True, "internal_label_or_diagnostic"
    if lowered != "side" and any(token in lowered for token in LEAKY_NAME_TOKENS):
        return True, "leaky_or_target_like_name"
    return False, ""


def evaluate_feature_validity(
    frame: pd.DataFrame,
    candidate_features: list[str],
    *,
    config: ConditionalFeatureSelectionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in dict.fromkeys(str(f) for f in candidate_features):
        family = classify_feature_family(feature)
        lookback = infer_lookback_hours(feature)
        relevance = horizon_relevance_score(
            lookback,
            horizon_min_hours=config.horizon_min_hours,
            horizon_max_hours=config.horizon_max_hours,
        )
        excluded, reason = _is_excluded_feature_name(feature)
        row = {
            "feature": feature,
            "family": family,
            "lookback_hours": lookback,
            "horizon_relevance": relevance,
            "status": "pass",
            "reject_reason": "",
            "finite_frac": 0.0,
            "unique_values": 0,
            "std": float("nan"),
            "abs_p99": float("nan"),
            "abs_max": float("nan"),
        }
        if excluded:
            row.update({"status": "fail", "reject_reason": reason})
            rows.append(row)
            continue
        if feature not in frame.columns:
            row.update({"status": "fail", "reject_reason": "missing_from_existing_feature_frame"})
            rows.append(row)
            continue
        ser = _safe_numeric(frame[feature], index=frame.index)
        finite = np.isfinite(ser.to_numpy(dtype=np.float64))
        row["finite_frac"] = float(np.mean(finite)) if len(ser) else 0.0
        row["unique_values"] = int(ser[finite].nunique(dropna=True)) if finite.any() else 0
        row["std"] = float(ser[finite].std(ddof=0)) if finite.any() else float("nan")
        if finite.any():
            abs_vals = np.abs(ser[finite].to_numpy(dtype=np.float64))
            row["abs_p99"] = float(np.nanpercentile(abs_vals, 99.0))
            row["abs_max"] = float(np.nanmax(abs_vals))
        if not pd.api.types.is_numeric_dtype(ser):
            row.update({"status": "fail", "reject_reason": "non_numeric"})
        elif int(np.sum(~np.isfinite(ser.dropna().to_numpy(dtype=np.float64)))) > 0:
            row.update({"status": "fail", "reject_reason": "infinite_values"})
        elif row["finite_frac"] < float(config.min_feature_finite_frac):
            row.update({"status": "fail", "reject_reason": "missingness"})
        elif feature == "side" and row["unique_values"] < 2:
            row.update({"status": "fail", "reject_reason": "side_not_mixed"})
        elif feature != "side" and row["unique_values"] < int(config.min_unique_values):
            row.update({"status": "fail", "reject_reason": "near_constant"})
        elif not math.isfinite(float(row["std"])) or abs(float(row["std"])) <= 1e-12:
            row.update({"status": "fail", "reject_reason": "zero_variance"})
        elif (
            math.isfinite(float(row["abs_p99"]))
            and float(row["abs_p99"]) > float(config.max_abs_p99)
        ) or (
            math.isfinite(float(row["abs_max"]))
            and float(row["abs_max"]) > float(config.max_abs_value)
        ):
            row.update({"status": "fail", "reject_reason": "extreme_abs_value"})
        elif relevance < float(config.min_horizon_relevance):
            row.update({"status": "fail", "reject_reason": "horizon_irrelevant"})
        elif (
            config.hard_max_lookback_hours is not None
            and lookback is not None
            and float(lookback) > float(config.hard_max_lookback_hours)
        ):
            row.update({"status": "fail", "reject_reason": "lookback_above_hard_cap"})
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["status", "family", "feature"], ascending=[False, True, True]).reset_index(drop=True)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    xv = x[mask].to_numpy(dtype=np.float64)
    yv = y[mask].to_numpy(dtype=np.float64)
    if float(np.nanstd(xv)) <= 1e-12 or float(np.nanstd(yv)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    xr = x[mask].rank(method="average")
    yr = y[mask].rank(method="average")
    if xr.nunique(dropna=True) < 2 or yr.nunique(dropna=True) < 2:
        return float("nan")
    return float(xr.corr(yr))


def _shrink_linear(raw: float, prior: float, n: int, k: float) -> float:
    if not math.isfinite(raw):
        return float("nan")
    if not math.isfinite(prior):
        prior = 0.0
    w = float(n) / (float(n) + max(float(k), 0.0))
    return float(w * raw + (1.0 - w) * prior)


def _shrink_pearson(raw: float, prior: float, n: int, k: float) -> float:
    if not math.isfinite(raw):
        return float("nan")
    raw = float(np.clip(raw, -0.999, 0.999))
    prior = float(np.clip(prior if math.isfinite(prior) else 0.0, -0.999, 0.999))
    w = float(n) / (float(n) + max(float(k), 0.0))
    z = w * np.arctanh(raw) + (1.0 - w) * np.arctanh(prior)
    return float(np.tanh(z))


def _decile_stats(feature: pd.Series, target: pd.Series) -> dict[str, float]:
    mask = feature.notna() & target.notna()
    if int(mask.sum()) < 30 or feature[mask].nunique(dropna=True) < 3:
        return {
            "top_decile_target_mean": float("nan"),
            "bottom_decile_target_mean": float("nan"),
            "top_minus_bottom_spread": float("nan"),
            "monotonicity_score": float("nan"),
        }
    ranks = feature[mask].rank(method="first", pct=True)
    decile = np.ceil(ranks * 10.0).clip(1, 10).astype(int)
    means = target[mask].groupby(decile).mean().reindex(range(1, 11))
    top = float(means.loc[10]) if pd.notna(means.loc[10]) else float("nan")
    bottom = float(means.loc[1]) if pd.notna(means.loc[1]) else float("nan")
    valid = means.notna()
    mono = _spearman(pd.Series(np.arange(1, 11)[valid.to_numpy()]), means[valid].reset_index(drop=True))
    return {
        "top_decile_target_mean": top,
        "bottom_decile_target_mean": bottom,
        "top_minus_bottom_spread": (
            top - bottom if math.isfinite(top) and math.isfinite(bottom) else float("nan")
        ),
        "monotonicity_score": mono,
    }


def _bucket_series(frame: pd.DataFrame, mode: str, bucket_col: str | None) -> pd.Series:
    if bucket_col and bucket_col in frame.columns:
        return frame[bucket_col].astype(str).fillna("missing")
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    ts_bucket = ts.dt.tz_convert(None)
    if mode == "week":
        return ts_bucket.dt.to_period("W-SUN").astype(str)
    if mode == "day":
        return ts_bucket.dt.strftime("%Y-%m-%d").fillna("NaT")
    if mode == "all":
        return pd.Series("all", index=frame.index)
    return ts_bucket.dt.to_period("M").astype(str)


def compute_bucket_ic_matrix(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    valid_features: pd.DataFrame,
    *,
    config: ConditionalFeatureSelectionConfig,
    bucket_mode: str = "month",
    bucket_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_names = valid_features["feature"].astype(str).head(int(config.max_input_features)).tolist()
    feature_meta = valid_features.set_index("feature").to_dict(orient="index")
    buckets = _bucket_series(frame, bucket_mode, bucket_col)
    side = _normalise_side(frame)
    rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    for feature in feature_names:
        x = _safe_numeric(frame[feature], index=frame.index)
        for target_name in TARGET_NAMES:
            y = targets[target_name]
            global_s = _spearman(x, y)
            global_p = _pearson(x, y)
            long_ic = float("nan")
            short_ic = float("nan")
            if int((side > 0).sum()) >= int(config.min_side_rows):
                long_ic = _spearman(x[side > 0], y[side > 0])
            if int((side < 0).sum()) >= int(config.min_side_rows):
                short_ic = _spearman(x[side < 0], y[side < 0])
            side_coverage_ok = bool(math.isfinite(long_ic) and math.isfinite(short_ic))
            global_rows.append(
                {
                    "feature": feature,
                    "target": target_name,
                    "global_spearman_ic": global_s,
                    "global_pearson_ic": global_p,
                    "long_spearman_ic": long_ic,
                    "short_spearman_ic": short_ic,
                    "side_coverage_ok": side_coverage_ok,
                    "long_short_ic_difference": (
                        long_ic - short_ic
                        if math.isfinite(long_ic) and math.isfinite(short_ic)
                        else float("nan")
                    ),
                    **feature_meta.get(feature, {}),
                }
            )
            for bucket, idx in pd.Series(np.arange(len(frame)), index=frame.index).groupby(buckets, dropna=False):
                pos = idx.to_numpy(dtype=np.int64)
                if len(pos) < int(config.min_bucket_rows):
                    continue
                xb = x.iloc[pos].reset_index(drop=True)
                yb = y.iloc[pos].reset_index(drop=True)
                raw_s = _spearman(xb, yb)
                raw_p = _pearson(xb, yb)
                shrunk_s = _shrink_linear(raw_s, global_s, len(pos), config.shrinkage_k)
                shrunk_p = _shrink_pearson(raw_p, global_p, len(pos), config.shrinkage_k)
                stats = _decile_stats(xb, yb)
                se = 1.0 / math.sqrt(max(len(pos) - 3, 1))
                rows.append(
                    {
                        "bucket": str(bucket),
                        "feature": feature,
                        "target": target_name,
                        "rows": int(len(pos)),
                        "spearman_ic_raw": raw_s,
                        "pearson_ic_raw": raw_p,
                        "spearman_ic_shrunk": shrunk_s,
                        "pearson_ic_shrunk": shrunk_p,
                        "ic_se": se,
                        "ic_tstat": shrunk_s / se if math.isfinite(shrunk_s) and se > 0 else float("nan"),
                        **stats,
                        **feature_meta.get(feature, {}),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(global_rows)


def score_feature_target_pairs(
    bucket_ic: pd.DataFrame,
    global_ic: pd.DataFrame,
    *,
    config: ConditionalFeatureSelectionConfig,
) -> pd.DataFrame:
    if global_ic.empty:
        return pd.DataFrame()

    def finite_or_zero(value: Any) -> float:
        try:
            out = float(value)
        except Exception:
            return 0.0
        return out if math.isfinite(out) else 0.0

    bucket_groups = {
        key: group.copy()
        for key, group in bucket_ic.groupby(["feature", "target"], dropna=False)
    } if not bucket_ic.empty else {}
    rows: list[dict[str, Any]] = []
    for _, g in global_ic.iterrows():
        feature = str(g["feature"])
        target = str(g["target"])
        group = bucket_groups.get((feature, target), pd.DataFrame())
        vals = pd.to_numeric(group.get("spearman_ic_shrunk", pd.Series(dtype=float)), errors="coerce").dropna()
        threshold = float(config.ic_threshold)
        global_ic_abs = abs(finite_or_zero(g.get("global_spearman_ic")))
        bucket_std = float(vals.std(ddof=0)) if len(vals) else 0.0
        pos_share = float((vals > threshold).mean()) if len(vals) else 0.0
        neg_share = float((vals < -threshold).mean()) if len(vals) else 0.0
        sign_flip_rate = float(min(pos_share, neg_share) * 2.0)
        mean_ic = float(vals.mean()) if len(vals) else float("nan")
        mean_abs_mono = (
            float(pd.to_numeric(group.get("monotonicity_score"), errors="coerce").abs().mean())
            if not group.empty and "monotonicity_score" in group.columns
            else 0.0
        )
        mean_se = (
            float(pd.to_numeric(group.get("ic_se"), errors="coerce").mean())
            if not group.empty and "ic_se" in group.columns
            else 1.0
        )
        side_diff = abs(finite_or_zero(g.get("long_short_ic_difference")))
        side_coverage_ok = bool(g.get("side_coverage_ok", False))
        global_score = min(global_ic_abs / 0.05, 1.0)
        conditional_score = min(bucket_std / 0.05, 1.0)
        sign_flip_score = min(sign_flip_rate, 1.0)
        tail_score = global_score if target in RISK_TARGETS else 0.0
        side_score = min(side_diff / 0.05, 1.0)
        stability_score = min(abs(mean_ic if math.isfinite(mean_ic) else 0.0) / (bucket_std + 0.02), 1.0)
        mono_score = min(mean_abs_mono, 1.0)
        horizon_score = float(g.get("horizon_relevance", 0.75) or 0.75)
        noise_penalty = min(mean_se / 0.10, 1.0)
        pair_score = (
            0.24 * global_score
            + 0.23 * conditional_score
            + 0.14 * sign_flip_score
            + 0.15 * tail_score
            + 0.10 * mono_score
            + 0.10 * stability_score
            + 0.14 * side_score
            + 0.05 * horizon_score
            - 0.05 * noise_penalty
        )
        category_scores = {
            "global": global_score,
            "conditional": max(conditional_score, sign_flip_score),
            "risk_tail": tail_score,
            "side_asymmetric": side_score,
        }
        primary_category = max(category_scores, key=category_scores.get)
        rows.append(
            {
                "feature": feature,
                "target": target,
                "family": g.get("family", "other"),
                "lookback_hours": g.get("lookback_hours"),
                "horizon_relevance": horizon_score,
                "global_spearman_ic": g.get("global_spearman_ic"),
                "global_pearson_ic": g.get("global_pearson_ic"),
                "mean_bucket_spearman_ic_shrunk": mean_ic,
                "bucket_ic_std": bucket_std,
                "positive_bucket_share": pos_share,
                "negative_bucket_share": neg_share,
                "sign_flip_rate": sign_flip_rate,
                "mean_abs_monotonicity": mean_abs_mono,
                "mean_ic_se": mean_se,
                "long_spearman_ic": g.get("long_spearman_ic"),
                "short_spearman_ic": g.get("short_spearman_ic"),
                "side_coverage_ok": side_coverage_ok,
                "long_short_ic_difference": g.get("long_short_ic_difference"),
                "global_score": global_score,
                "conditional_score": conditional_score,
                "sign_flip_score": sign_flip_score,
                "tail_score": tail_score,
                "side_asymmetry_score": side_score,
                "stability_score": stability_score,
                "pair_score": pair_score,
                "primary_category": primary_category,
                "is_global_predictive": global_score >= 0.40,
                "is_conditional": max(conditional_score, sign_flip_score) >= 0.35,
                "is_risk_tail": target in RISK_TARGETS and tail_score >= 0.25,
                "is_side_asymmetric": side_score >= 0.35,
                "bucket_count": int(len(vals)),
            }
        )
    return pd.DataFrame(rows).sort_values("pair_score", ascending=False).reset_index(drop=True)


def _select_by_category(pair_scores: pd.DataFrame, config: ConditionalFeatureSelectionConfig) -> pd.DataFrame:
    if pair_scores.empty:
        return pair_scores
    if bool(config.require_both_sides) and "side_coverage_ok" in pair_scores.columns:
        pair_scores = pair_scores[pair_scores["side_coverage_ok"].astype(bool)].copy()
        if pair_scores.empty:
            return pair_scores
    max_pairs = int(config.max_selected_pairs)
    quotas = {
        "is_global_predictive": int(round(max_pairs * 0.30)),
        "is_conditional": int(round(max_pairs * 0.35)),
        "is_risk_tail": int(round(max_pairs * 0.20)),
        "is_side_asymmetric": int(round(max_pairs * 0.15)),
    }
    selected_idx: list[int] = []
    for flag, quota in quotas.items():
        if quota <= 0 or flag not in pair_scores.columns:
            continue
        rows = pair_scores[pair_scores[flag]].sort_values("pair_score", ascending=False)
        for idx in rows.index:
            if idx not in selected_idx:
                selected_idx.append(int(idx))
            if len([i for i in selected_idx if bool(pair_scores.loc[i, flag])]) >= quota:
                break
    for idx in pair_scores.sort_values("pair_score", ascending=False).index:
        if idx not in selected_idx:
            selected_idx.append(int(idx))
        if len(selected_idx) >= max_pairs:
            break
    selected = pair_scores.loc[selected_idx].copy()
    for family in MANDATORY_FAMILIES:
        if family not in set(pair_scores["family"].astype(str)):
            continue
        if family in set(selected["family"].astype(str)):
            continue
        add = pair_scores[pair_scores["family"].astype(str).eq(family)].head(1)
        if not add.empty:
            selected = pd.concat([selected, add], ignore_index=True)
    return selected.sort_values("pair_score", ascending=False).drop_duplicates(["feature", "target"]).reset_index(drop=True)


def _dedupe_selected_features(
    frame: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    *,
    config: ConditionalFeatureSelectionConfig,
) -> list[str]:
    feature_order = (
        selected_pairs.groupby(["feature", "family"], dropna=False)["pair_score"]
        .max()
        .reset_index()
        .sort_values("pair_score", ascending=False)
    )
    ordered = feature_order["feature"].astype(str).tolist()
    if len(ordered) <= 1:
        return ordered
    rng = np.random.default_rng(int(config.random_seed))
    rows = np.arange(len(frame))
    if len(rows) > int(config.max_corr_rows):
        rows = np.sort(rng.choice(rows, size=int(config.max_corr_rows), replace=False))
    kept: list[str] = []
    kept_by_family: dict[str, list[str]] = {}
    family_map = dict(zip(feature_order["feature"].astype(str), feature_order["family"].astype(str)))
    for feature in ordered:
        family = family_map.get(feature, "other")
        if feature not in frame.columns:
            continue
        reject = False
        x = _safe_numeric(frame[feature].iloc[rows]).fillna(0.0)
        for prev in kept_by_family.get(family, []):
            y = _safe_numeric(frame[prev].iloc[rows]).fillna(0.0)
            corr = _pearson(x.reset_index(drop=True), y.reset_index(drop=True))
            if math.isfinite(corr) and abs(corr) > float(config.corr_threshold):
                reject = True
                break
        if reject:
            continue
        kept.append(feature)
        kept_by_family.setdefault(family, []).append(feature)
        if len(kept) >= int(config.max_selected_features):
            break
    return kept


def _build_selected_features_table(selected_pairs: pd.DataFrame, kept_features: list[str]) -> pd.DataFrame:
    if selected_pairs.empty:
        return pd.DataFrame(columns=["feature"])
    kept = set(kept_features)
    rows = []
    for feature, group in selected_pairs[selected_pairs["feature"].isin(kept)].groupby("feature", dropna=False):
        best = group.sort_values("pair_score", ascending=False).iloc[0]
        rows.append(
            {
                "feature": str(feature),
                "family": best.get("family"),
                "max_pair_score": float(group["pair_score"].max()),
                "selected_pair_count": int(len(group)),
                "targets": ",".join(sorted(group["target"].astype(str).unique())),
                "primary_categories": ",".join(sorted(group["primary_category"].astype(str).unique())),
                "lookback_hours": best.get("lookback_hours"),
                "horizon_relevance": best.get("horizon_relevance"),
            }
        )
    return pd.DataFrame(rows).sort_values("max_pair_score", ascending=False).reset_index(drop=True)


def _build_training_feature_list(selected_features: pd.DataFrame) -> pd.DataFrame:
    """Export selected conditional-GMM inputs in the training feature-list schema."""
    columns = [
        "feature",
        "selected_feature_position",
        "selected_feature_count",
        "used_by_model",
        "source",
        "family",
        "max_pair_score",
        "selected_pair_count",
        "targets",
        "primary_categories",
        "lookback_hours",
        "horizon_relevance",
    ]
    if selected_features.empty or "feature" not in selected_features.columns:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    total = int(len(selected_features))
    for pos, row in enumerate(selected_features.to_dict(orient="records"), start=1):
        rows.append(
            {
                "feature": str(row.get("feature", "")),
                "selected_feature_position": int(pos),
                "selected_feature_count": total,
                "used_by_model": True,
                "source": "conditional_gmm_feature_selection",
                "family": row.get("family", ""),
                "max_pair_score": row.get("max_pair_score", float("nan")),
                "selected_pair_count": row.get("selected_pair_count", 0),
                "targets": row.get("targets", ""),
                "primary_categories": row.get("primary_categories", ""),
                "lookback_hours": row.get("lookback_hours", float("nan")),
                "horizon_relevance": row.get("horizon_relevance", float("nan")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_signature_columns(selected_pairs: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "spearman_ic_shrunk",
        "pearson_ic_shrunk",
        "ic_tstat",
        "top_minus_bottom_spread",
        "monotonicity_score",
        "long_short_ic_difference",
    )
    rows: list[dict[str, Any]] = []
    for _, row in selected_pairs.iterrows():
        feature = str(row["feature"])
        target = str(row["target"])
        safe_feature = re.sub(r"[^a-zA-Z0-9_]+", "_", feature).strip("_")
        safe_target = re.sub(r"[^a-zA-Z0-9_]+", "_", target).strip("_")
        for metric in metrics:
            rows.append(
                {
                    "signature_column": f"{safe_feature}__{safe_target}__{metric}",
                    "feature": feature,
                    "target": target,
                    "metric": metric,
                    "family": row.get("family"),
                    "pair_score": row.get("pair_score"),
                    "column_weight": max(0.25, min(1.0, float(row.get("pair_score", 0.0) or 0.0))),
                }
            )
    return pd.DataFrame(rows)


def run_conditional_selection_on_frame(
    frame: pd.DataFrame,
    *,
    candidate_features: list[str] | None = None,
    config: ConditionalFeatureSelectionConfig | None = None,
    bucket_mode: str = "month",
    bucket_col: str | None = None,
) -> ConditionalSelectionResult:
    cfg = config or ConditionalFeatureSelectionConfig()
    targets, target_report = build_side_aware_targets(frame)
    if candidate_features is None:
        candidate_features = [str(c) for c in frame.columns]
    validity = evaluate_feature_validity(frame, candidate_features, config=cfg)
    valid = validity[validity["status"].eq("pass")].copy()
    bucket_ic, global_ic = compute_bucket_ic_matrix(
        frame,
        targets,
        valid,
        config=cfg,
        bucket_mode=bucket_mode,
        bucket_col=bucket_col,
    )
    pair_scores = score_feature_target_pairs(bucket_ic, global_ic, config=cfg)
    selected_pairs = _select_by_category(pair_scores, cfg)
    kept_features = _dedupe_selected_features(frame, selected_pairs, config=cfg)
    selected_pairs = selected_pairs[selected_pairs["feature"].isin(kept_features)].copy().reset_index(drop=True)
    selected_features = _build_selected_features_table(selected_pairs, kept_features)
    signature_columns = _build_signature_columns(selected_pairs)
    return ConditionalSelectionResult(
        feature_validity=validity,
        bucket_ic_matrix=bucket_ic,
        pair_scores=pair_scores,
        selected_pairs=selected_pairs,
        selected_features=selected_features,
        signature_columns=signature_columns,
        target_report=target_report,
    )


def run_conditional_feature_selection(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path | None = None,
    feature_list_csv: Path | None = None,
    target_optimization_summary_json: Path | None = None,
    infer_feature_store_schema: bool = False,
    feature_store_schema_files: int = 5,
    bucket_mode: str = "month",
    bucket_col: str | None = None,
    config: ConditionalFeatureSelectionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ConditionalFeatureSelectionConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_readiness = _load_target_readiness(target_optimization_summary_json)
    frame = _load_labels(labels_path)
    explicit_features = _read_feature_list(feature_list_csv)
    feature_store_report: dict[str, Any] = {"enabled": False}
    if feature_dir is not None:
        if infer_feature_store_schema:
            inferred_features = _infer_feature_store_columns(feature_dir, max_files=feature_store_schema_files)
            explicit_features = list(dict.fromkeys(list(explicit_features) + list(inferred_features)))
        if explicit_features:
            features_to_load = [feature for feature in explicit_features if feature not in frame.columns]
            matrix, feature_store_report = _load_feature_store_columns(
                frame,
                feature_dir=feature_dir,
                selected_features=features_to_load,
            )
            feature_store_report["skipped_existing_frame_columns"] = int(
                len(explicit_features) - len(features_to_load)
            )
            if not matrix.empty:
                frame = pd.concat([frame, matrix], axis=1)
    candidate_features = explicit_features or [str(c) for c in frame.columns]
    result = run_conditional_selection_on_frame(
        frame,
        candidate_features=candidate_features,
        config=cfg,
        bucket_mode=bucket_mode,
        bucket_col=bucket_col,
    )
    paths = {
        "feature_validity": output_dir / "conditional_feature_validity.csv",
        "bucket_ic_matrix": output_dir / "conditional_bucket_feature_target_ic.csv",
        "pair_scores": output_dir / "conditional_feature_target_pair_scores.csv",
        "selected_pairs": output_dir / "conditional_selected_feature_target_pairs.csv",
        "selected_features": output_dir / "conditional_selected_features.csv",
        "training_feature_list": output_dir / "conditional_gmm_training_feature_list.csv",
        "signature_columns": output_dir / "conditional_gmm_signature_columns.csv",
        "manifest": output_dir / "manifest.json",
    }
    training_feature_list = _build_training_feature_list(result.selected_features)
    result.feature_validity.to_csv(paths["feature_validity"], index=False)
    result.bucket_ic_matrix.to_csv(paths["bucket_ic_matrix"], index=False)
    result.pair_scores.to_csv(paths["pair_scores"], index=False)
    result.selected_pairs.to_csv(paths["selected_pairs"], index=False)
    result.selected_features.to_csv(paths["selected_features"], index=False)
    training_feature_list.to_csv(paths["training_feature_list"], index=False)
    result.signature_columns.to_csv(paths["signature_columns"], index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "conditional_gmm_feature_selection_v1",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "bucket_mode": str(bucket_mode),
        "bucket_col": str(bucket_col or ""),
        "config": asdict(cfg),
        "feature_store": feature_store_report,
        "feature_list_csv": str(feature_list_csv) if feature_list_csv is not None else "",
        "target_optimization_summary_json": (
            str(target_optimization_summary_json)
            if target_optimization_summary_json is not None
            else ""
        ),
        "target_readiness": target_readiness,
        "existing_features_only": True,
        "creates_new_live_features": False,
        "target_report": result.target_report,
        "counts": {
            "candidate_features": int(len(candidate_features)),
            "valid_features": int(result.feature_validity["status"].eq("pass").sum()),
            "pair_scores": int(len(result.pair_scores)),
            "selected_pairs": int(len(result.selected_pairs)),
            "selected_features": int(len(result.selected_features)),
            "training_feature_list": int(len(training_feature_list)),
            "signature_columns": int(len(result.signature_columns)),
        },
        "selection_policy": {
            "scope": (
                "conditional, risk-tail, side-asymmetric, and global feature-target pair selection "
                "for 3-7h side-aware candidates; outputs are suitable for GMM signatures, "
                "live GMM-label predictor inputs, and downstream feature/target-pair ablations"
            ),
            "horizon_note": "Trailing lookbacks longer than 3-7h are allowed when causal; horizon relevance is a score, not a default hard cap.",
            "live_feature_rule": "Targets may use future data for offline scoring, but selected features must already exist ex-ante in the label frame or feature store.",
            "side_rule": "Selected pairs must have estimable long and short ICs when require_both_sides is enabled.",
            "retention_mix": {
                "global_predictive_pairs": 0.30,
                "conditional_pairs": 0.35,
                "risk_tail_pairs": 0.20,
                "side_asymmetric_pairs": 0.15,
            },
        },
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--feature-list-csv", type=Path, default=None)
    parser.add_argument("--target-optimization-summary-json", type=Path, default=None)
    parser.add_argument("--infer-feature-store-schema", action="store_true")
    parser.add_argument("--feature-store-schema-files", type=int, default=5)
    parser.add_argument("--bucket-mode", choices=["month", "week", "day", "all"], default="month")
    parser.add_argument("--bucket-col", default=None)
    parser.add_argument("--horizon-min-hours", type=float, default=3.0)
    parser.add_argument("--horizon-max-hours", type=float, default=7.0)
    parser.add_argument("--min-feature-finite-frac", type=float, default=0.95)
    parser.add_argument("--min-bucket-rows", type=int, default=200)
    parser.add_argument("--min-side-rows", type=int, default=100)
    parser.add_argument("--shrinkage-k", type=float, default=500.0)
    parser.add_argument("--ic-threshold", type=float, default=0.02)
    parser.add_argument("--max-input-features", type=int, default=300)
    parser.add_argument("--max-selected-pairs", type=int, default=120)
    parser.add_argument("--max-selected-features", type=int, default=80)
    parser.add_argument("--max-abs-value", type=float, default=1e6)
    parser.add_argument("--max-abs-p99", type=float, default=1e5)
    parser.add_argument("--hard-max-lookback-hours", type=float, default=None)
    parser.add_argument("--allow-single-side-pairs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ConditionalFeatureSelectionConfig(
        horizon_min_hours=float(args.horizon_min_hours),
        horizon_max_hours=float(args.horizon_max_hours),
        min_feature_finite_frac=float(args.min_feature_finite_frac),
        min_bucket_rows=int(args.min_bucket_rows),
        min_side_rows=int(args.min_side_rows),
        shrinkage_k=float(args.shrinkage_k),
        ic_threshold=float(args.ic_threshold),
        max_input_features=int(args.max_input_features),
        max_selected_pairs=int(args.max_selected_pairs),
        max_selected_features=int(args.max_selected_features),
        max_abs_value=float(args.max_abs_value),
        max_abs_p99=float(args.max_abs_p99),
        hard_max_lookback_hours=args.hard_max_lookback_hours,
        require_both_sides=not bool(args.allow_single_side_pairs),
    )
    manifest = run_conditional_feature_selection(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        target_optimization_summary_json=args.target_optimization_summary_json,
        infer_feature_store_schema=bool(args.infer_feature_store_schema),
        feature_store_schema_files=int(args.feature_store_schema_files),
        bucket_mode=str(args.bucket_mode),
        bucket_col=args.bucket_col,
        config=cfg,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
