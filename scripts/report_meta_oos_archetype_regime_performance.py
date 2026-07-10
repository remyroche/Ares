#!/usr/bin/env python3
"""Report meta OOS performance by archetype x market regime.

This report uses the monthly OOS meta prediction shards as the performance
source, then joins the pre-selection meta handoff feature universe to bin
existing trend/vol/volume/entropy/liquidity/VWAP/range-position features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_META_RUN = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "train_meta_regime_ablation_matrix_apr_may_jun_20260707/"
    "baseline_current_full_context"
)
DEFAULT_HANDOFF = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "s52_trailing_regime_meta_handoff_top30_allsafe_aegmm_fixedtargets_oos15_20260706/"
    "train_meta_regime_handoff.parquet"
)
DEFAULT_OUT = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "meta_oos_archetype_regime_performance_20260708"
)


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    semantic: str
    candidates: tuple[str, ...]
    binning: str = "tercile"


REGIME_SPECS: tuple[RegimeSpec, ...] = (
    RegimeSpec(
        "trend",
        "trend strength / trend-following context",
        (
            "trend_strength_percentile",
            "trend_t",
            "trend_snr",
            "trend_pct",
            "price_trend_7d_vol_norm",
            "__regime_trend_48h___y",
            "__regime_trend_48h___x",
        ),
    ),
    RegimeSpec(
        "vol",
        "realized volatility / vol regime",
        (
            "volatility_zscore",
            "vol_z24",
            "vol_z_4h",
            "asset_vol_level_pct",
            "__regime_vol_48h___y",
            "__regime_vol_48h___x",
        ),
    ),
    RegimeSpec(
        "volume",
        "relative volume / participation",
        (
            "volume_z_24",
            "volume_z_12",
            "volume_percentile",
            "volume_zscore_48h",
            "__regime_volume_48h___y",
            "__regime_volume_48h___x",
        ),
    ),
    RegimeSpec(
        "entropy",
        "path/return entropy",
        (
            "shannon_entropy_ret_16",
            "path_entropy_24_y",
            "path_entropy_24_x",
            "direction_entropy_20",
            "bar_direction_entropy",
            "state_spectral_eig_entropy",
        ),
    ),
    RegimeSpec(
        "ohlcv_liquidity_proxy",
        "OHLCV-derived spread/liquidity stress proxy",
        (
            "spread_proxy_hl_range_bps_robust_z",
            "spread_proxy_abs_return_bps_robust_z",
            "spread_proxy_body_bps_robust_z",
            "spread_proxy_wick_to_range_robust_z",
            "vol_price_spread",
        ),
    ),
    RegimeSpec(
        "position_to_vwap",
        "distance/position versus VWAP",
        (
            "dist_vwap_norm",
            "dist_vwap_atr",
            "distance_to_vwap",
            "loc_vwap_dev_z_24",
            "z_vwap_24",
        ),
        binning="signed_tercile",
    ),
    RegimeSpec(
        "position_to_last_day_hl",
        "position inside previous-day high-low range",
        (
            "loc_prev_day_range_pos_24",
            "loc_prev_day_range_pos_48",
            "loc_range_pos_24",
            "dist_prior_day_high",
            "dist_prior_day_low",
        ),
        binning="range_position",
    ),
)


PERF_COLS = [
    "exec_margin",
    "ev_after_1pct",
    "first_touch_gross",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "clean_exec",
    "dirty_positive",
    "score_base",
    "score_meta_base_soft_label",
]


def _schema_cols(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_schema(path).names


def _candidate_features(frame_cols: set[str], spec: RegimeSpec) -> list[str]:
    return [col for col in spec.candidates if col in frame_cols]


def _load_predictions(meta_run: Path, months: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in months:
        shard_matches = sorted((meta_run / "prediction_shards").glob(f"*{month}.parquet"))
        if not shard_matches:
            raise FileNotFoundError(f"No prediction shard for {month} under {meta_run / 'prediction_shards'}")
        frame = pd.read_parquet(shard_matches[-1])
        frames.append(frame)
    pred = pd.concat(frames, ignore_index=True, copy=False)
    pred["__ts__"] = pd.to_datetime(pred["__ts__"], utc=True, errors="coerce")
    pred["month"] = pred["__ts__"].dt.to_period("M").astype(str)
    pred["__symbol__"] = pred["__symbol__"].astype(str)
    pred["side_name"] = pred["side_name"].astype(str)
    return pred


def _load_feature_slice(handoff: Path, pred: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    cols = _schema_cols(handoff)
    needed = ["__ts__", "__symbol__", "side_name", *sorted(set(candidate_cols))]
    read_cols = [col for col in needed if col in cols]
    features = pd.read_parquet(handoff, columns=read_cols)
    features["__ts__"] = pd.to_datetime(features["__ts__"], utc=True, errors="coerce")
    features["month"] = features["__ts__"].dt.to_period("M").astype(str)
    features = features.loc[features["month"].isin(sorted(pred["month"].unique()))].copy()
    features["__symbol__"] = features["__symbol__"].astype(str)
    features["side_name"] = features["side_name"].astype(str)
    keys = ["__ts__", "__symbol__", "side_name"]
    features = features.drop_duplicates(keys)
    return features.drop(columns=["month"])


def _choose_feature_from_data(features: pd.DataFrame, spec: RegimeSpec) -> str | None:
    best_col: str | None = None
    best_score = -np.inf
    for col in spec.candidates:
        if col not in features.columns:
            continue
        x = pd.to_numeric(features[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        coverage = float(x.notna().mean())
        if coverage <= 0.0:
            continue
        valid = x.dropna()
        if valid.empty:
            continue
        unique = int(valid.nunique(dropna=True))
        spread = float(valid.quantile(0.95) - valid.quantile(0.05)) if len(valid) >= 20 else 0.0
        nondegenerate = unique >= 5 and np.isfinite(spread) and spread > 1e-9
        score = coverage + (0.50 if nondegenerate else 0.0) + min(unique, 20) / 1000.0
        if score > best_score:
            best_col = col
            best_score = score
    return best_col


def _bin_numeric(values: pd.Series, *, mode: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series("missing", index=values.index, dtype=object)
    valid = x.notna()
    if int(valid.sum()) < 20:
        out.loc[valid] = "valid"
        return out
    if mode == "range_position":
        out.loc[valid & x.lt(0.33)] = "low_in_prev_day_range"
        out.loc[valid & x.ge(0.33) & x.le(0.67)] = "mid_in_prev_day_range"
        out.loc[valid & x.gt(0.67)] = "high_in_prev_day_range"
        return out
    if mode == "signed_tercile":
        near = x.abs().le(0.25)
        out.loc[valid & x.lt(-0.25)] = "below_vwap"
        out.loc[valid & near] = "near_vwap"
        out.loc[valid & x.gt(0.25)] = "above_vwap"
        # If thresholds collapse most rows, fall back to terciles below.
        if out.loc[valid].nunique() >= 2:
            return out
        out.loc[valid] = "valid"
    q1, q2 = x.loc[valid].quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy(dtype=float)
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        out.loc[valid] = "valid"
        return out
    out.loc[valid & x.le(q1)] = "low"
    out.loc[valid & x.gt(q1) & x.le(q2)] = "mid"
    out.loc[valid & x.gt(q2)] = "high"
    return out


def _rank_scopes(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = frame.copy()
    score = pd.to_numeric(out[score_col], errors="coerce")
    out["_meta_rank_pct_month"] = score.groupby(out["month"]).rank(pct=True, method="first")
    return out


def _metric_row(group: pd.DataFrame, keys: dict[str, Any]) -> dict[str, Any]:
    ev = pd.to_numeric(group["ev_after_1pct"], errors="coerce")
    clean = pd.to_numeric(group["clean_exec"], errors="coerce")
    bad = pd.to_numeric(group["full_path_bad_mae_1r"], errors="coerce")
    first_bad = pd.to_numeric(group["first_touch_bad_mae_1r"], errors="coerce")
    timeout = pd.to_numeric(group["timeout"], errors="coerce")
    score = pd.to_numeric(group["score_meta_base_soft_label"], errors="coerce")
    return {
        **keys,
        "rows": int(len(group)),
        "mean_score": float(score.mean()),
        "mean_ev_after_1pct": float(ev.mean()),
        "sum_ev_after_1pct": float(ev.sum()),
        "positive_ev_rate": float(ev.gt(0.0).mean()),
        "mean_exec_margin": float(pd.to_numeric(group["exec_margin"], errors="coerce").mean()),
        "clean_exec_rate": float(clean.mean()),
        "dirty_positive_rate": float(pd.to_numeric(group["dirty_positive"], errors="coerce").mean()),
        "first_touch_bad_mae_rate": float(first_bad.mean()),
        "full_path_bad_mae_rate": float(bad.mean()),
        "timeout_rate": float(timeout.mean()),
    }


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "all": frame["_meta_rank_pct_month"].ge(0.0),
        "top30": frame["_meta_rank_pct_month"].ge(0.70),
        "top20": frame["_meta_rank_pct_month"].ge(0.80),
        "top10": frame["_meta_rank_pct_month"].ge(0.90),
    }
    rows: list[dict[str, Any]] = []
    group_cols = ["month", "side_name", "archetype_policy_key", "regime_family", "regime_bin", "feature_col"]
    for scope, mask in scopes.items():
        sub = frame.loc[mask].copy()
        if sub.empty:
            continue
        for key, group in sub.groupby(group_cols, dropna=False, observed=True):
            keys = {"top_scope": scope, **{col: val for col, val in zip(group_cols, key)}}
            rows.append(_metric_row(group, keys))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base_cols = ["top_scope", "month", "regime_family", "feature_col"]
    base = (
        out.groupby(base_cols, dropna=False, observed=True)
        .apply(lambda g: pd.Series({
            "baseline_mean_ev_after_1pct": np.average(g["mean_ev_after_1pct"], weights=g["rows"]),
            "baseline_clean_exec_rate": np.average(g["clean_exec_rate"], weights=g["rows"]),
            "baseline_full_path_bad_mae_rate": np.average(g["full_path_bad_mae_rate"], weights=g["rows"]),
            "baseline_timeout_rate": np.average(g["timeout_rate"], weights=g["rows"]),
        }))
        .reset_index()
    )
    out = out.merge(base, on=base_cols, how="left")
    out["delta_ev_vs_month_regime_baseline"] = out["mean_ev_after_1pct"] - out["baseline_mean_ev_after_1pct"]
    out["delta_clean_vs_month_regime_baseline"] = out["clean_exec_rate"] - out["baseline_clean_exec_rate"]
    out["delta_bad_mae_vs_month_regime_baseline"] = out["full_path_bad_mae_rate"] - out["baseline_full_path_bad_mae_rate"]
    out["delta_timeout_vs_month_regime_baseline"] = out["timeout_rate"] - out["baseline_timeout_rate"]
    return out.sort_values(["month", "top_scope", "regime_family", "archetype_policy_key", "regime_bin"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-run", type=Path, default=DEFAULT_META_RUN)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--months", nargs="+", default=["2026-05", "2026-06"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred = _load_predictions(args.meta_run, list(args.months))
    handoff_cols = set(_schema_cols(args.handoff))
    candidate_map = {spec.name: _candidate_features(handoff_cols, spec) for spec in REGIME_SPECS}
    candidate_cols = sorted({col for cols in candidate_map.values() for col in cols})
    if not candidate_cols:
        raise RuntimeError("No requested regime feature columns were available in the handoff artifact")
    features = _load_feature_slice(args.handoff, pred, candidate_cols)
    chosen = {
        spec.name: _choose_feature_from_data(features, spec)
        for spec in REGIME_SPECS
    }
    chosen = {k: v for k, v in chosen.items() if v is not None}
    if not chosen:
        raise RuntimeError("No requested regime feature columns had usable coverage in the handoff artifact")
    keys = ["__ts__", "__symbol__", "side_name"]
    merged = pred.merge(features, on=keys, how="left", validate="many_to_one")
    merged["archetype_policy_key"] = (
        merged.get("archetype_policy_key", merged.get("__archetype_policy_key__", "missing"))
        .astype(str)
        .replace({"nan": "missing", "None": "missing"})
    )
    merged = _rank_scopes(merged, "score_meta_base_soft_label")

    long_frames: list[pd.DataFrame] = []
    spec_by_name = {spec.name: spec for spec in REGIME_SPECS}
    for family, col in chosen.items():
        spec = spec_by_name[family]
        sub = merged.copy()
        sub["regime_family"] = family
        sub["feature_col"] = col
        sub["regime_bin"] = _bin_numeric(sub[col], mode=spec.binning)
        long_frames.append(sub)
    long = pd.concat(long_frames, ignore_index=True, copy=False)
    summary = _summarize(long)
    summary.to_csv(args.output_dir / "meta_oos_archetype_x_regime_performance.csv", index=False)

    # Compact tables for quick reading.
    compact = summary.loc[summary["rows"].ge(30)].copy()
    compact = compact.sort_values(["top_scope", "month", "mean_ev_after_1pct"], ascending=[True, True, False])
    compact.to_csv(args.output_dir / "meta_oos_archetype_x_regime_performance_min30.csv", index=False)
    top10 = summary.loc[(summary["top_scope"].eq("top10")) & summary["rows"].ge(10)].copy()
    top10.sort_values(["month", "mean_ev_after_1pct"], ascending=[True, False]).to_csv(
        args.output_dir / "meta_oos_archetype_x_regime_top10_min10.csv",
        index=False,
    )
    availability = []
    for spec in REGIME_SPECS:
        col = chosen.get(spec.name)
        availability.append(
            {
                "regime_family": spec.name,
                "semantic": spec.semantic,
                "chosen_feature_col": col,
                "candidate_feature_cols": ",".join(candidate_map.get(spec.name, ())),
                "available": col is not None,
                "non_null_share": float(pd.to_numeric(merged[col], errors="coerce").notna().mean()) if col else 0.0,
                "n_unique": int(pd.to_numeric(merged[col], errors="coerce").nunique(dropna=True)) if col else 0,
                "binning": spec.binning,
            }
        )
    availability_df = pd.DataFrame(availability)
    availability_df.to_csv(args.output_dir / "meta_oos_regime_feature_availability.csv", index=False)
    manifest = {
        "meta_run": str(args.meta_run),
        "prediction_shards": [str(p) for p in sorted((args.meta_run / "prediction_shards").glob("*.parquet"))],
        "handoff": str(args.handoff),
        "months": list(args.months),
        "score_col": "score_meta_base_soft_label",
        "top_scope_definition": {
            "top10": "score rank pct within OOS month >= 0.90",
            "top20": "score rank pct within OOS month >= 0.80",
            "top30": "score rank pct within OOS month >= 0.70",
        },
        "regime_features": availability,
        "outputs": {
            "full": str(args.output_dir / "meta_oos_archetype_x_regime_performance.csv"),
            "min30": str(args.output_dir / "meta_oos_archetype_x_regime_performance_min30.csv"),
            "top10_min10": str(args.output_dir / "meta_oos_archetype_x_regime_top10_min10.csv"),
            "availability": str(args.output_dir / "meta_oos_regime_feature_availability.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("FEATURES")
    print(availability_df.to_string(index=False))
    print("\nTOP10 BEST/WORST MIN10")
    show_cols = [
        "month",
        "side_name",
        "archetype_policy_key",
        "regime_family",
        "regime_bin",
        "rows",
        "mean_ev_after_1pct",
        "clean_exec_rate",
        "full_path_bad_mae_rate",
        "timeout_rate",
    ]
    print(top10.sort_values("mean_ev_after_1pct", ascending=False)[show_cols].head(15).to_string(index=False))
    print("\nWORST")
    print(top10.sort_values("mean_ev_after_1pct", ascending=True)[show_cols].head(15).to_string(index=False))
    print(f"\n[done] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
