#!/usr/bin/env python3
"""Report side-first archetype effects for threshold/size context diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARCHETYPE_PREFIXES = (
    "gmm_prob_",
    "gmm_cluster_posterior_",
    "gmm_dist_center_",
    "gmm_mahal_",
    "long_gmm_prob_",
    "long_gmm_cluster_posterior_",
    "long_gmm_dist_center_",
    "long_gmm_mahal_",
    "short_gmm_prob_",
    "short_gmm_cluster_posterior_",
    "short_gmm_dist_center_",
    "short_gmm_mahal_",
    "ctx_gmm_prob_",
    "ctx_gmm_cluster_posterior_",
    "ctx_gmm_dist_center_",
    "ctx_gmm_mahal_",
)
ARCHETYPE_EXACT = {
    "gmm_entropy",
    "cluster_entropy",
    "cluster_entropy_norm",
    "mahalanobis_distance",
    "min_mahalanobis",
    "expected_mahalanobis",
    "time_since_cluster_change",
    "rolling_cluster_stability",
    "cluster_flip_count_20",
    "AE_reconstruction_error",
    "ae_reconstruction_error",
    "dae_reconstruction_error",
    "dae_reconstruction_error_zscore",
    "cluster_speed",
    "cluster_acceleration",
    "latent_speed",
    "latent_acceleration",
    "long_gmm_posterior_max",
    "long_gmm_posterior_margin",
    "long_gmm_posterior_delta_1",
    "long_gmm_posterior_accel_1",
    "long_cluster_entropy_delta_1",
    "long_cluster_entropy_accel_1",
    "long_min_mahalanobis_delta_1",
    "long_expected_mahalanobis_delta_1",
    "long_expected_mahalanobis_accel_1",
    "long_gmm_entropy",
    "long_cluster_entropy",
    "long_cluster_entropy_norm",
    "long_mahalanobis_distance",
    "long_min_mahalanobis",
    "long_expected_mahalanobis",
    "long_time_since_cluster_change",
    "long_rolling_cluster_stability",
    "long_cluster_flip_count_20",
    "long_AE_reconstruction_error",
    "long_ae_reconstruction_error",
    "long_dae_reconstruction_error",
    "long_dae_reconstruction_error_zscore",
    "long_cluster_speed",
    "long_cluster_acceleration",
    "long_latent_speed",
    "long_latent_acceleration",
    "short_gmm_posterior_max",
    "short_gmm_posterior_margin",
    "short_gmm_posterior_delta_1",
    "short_gmm_posterior_accel_1",
    "short_cluster_entropy_delta_1",
    "short_cluster_entropy_accel_1",
    "short_min_mahalanobis_delta_1",
    "short_expected_mahalanobis_delta_1",
    "short_expected_mahalanobis_accel_1",
    "short_gmm_entropy",
    "short_cluster_entropy",
    "short_cluster_entropy_norm",
    "short_mahalanobis_distance",
    "short_min_mahalanobis",
    "short_expected_mahalanobis",
    "short_time_since_cluster_change",
    "short_rolling_cluster_stability",
    "short_cluster_flip_count_20",
    "short_AE_reconstruction_error",
    "short_ae_reconstruction_error",
    "short_dae_reconstruction_error",
    "short_dae_reconstruction_error_zscore",
    "short_cluster_speed",
    "short_cluster_acceleration",
    "short_latent_speed",
    "short_latent_acceleration",
    "ctx_gmm_entropy",
    "ctx_cluster_entropy",
    "ctx_cluster_entropy_norm",
    "ctx_mahalanobis_distance",
    "ctx_min_mahalanobis",
    "ctx_expected_mahalanobis",
    "ctx_time_since_cluster_change",
    "ctx_rolling_cluster_stability",
    "ctx_cluster_flip_count_20",
    "ctx_AE_reconstruction_error",
    "ctx_ae_reconstruction_error",
    "ctx_dae_reconstruction_error",
    "ctx_dae_reconstruction_error_zscore",
    "ctx_cluster_speed",
    "ctx_cluster_acceleration",
    "ctx_latent_speed",
    "ctx_latent_acceleration",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(c).lower(): str(c) for c in frame.columns}
    for name in names:
        if name in frame.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _side_series(frame: pd.DataFrame) -> pd.Series:
    col = _first_column(frame, ("side", "__side__", "trade_side"))
    if col is None:
        return pd.Series("unknown", index=frame.index)
    raw = frame[col]
    if raw.dtype == object:
        text = raw.astype(str).str.lower()
        return pd.Series(
            np.where(text.str.contains("short") | text.str.startswith("-"), "short", "long"),
            index=frame.index,
        )
    numeric = pd.to_numeric(raw, errors="coerce").fillna(1.0)
    return pd.Series(np.where(numeric < 0.0, "short", "long"), index=frame.index)


def _rank_band_series(frame: pd.DataFrame, bands: int) -> pd.Series:
    col = _first_column(
        frame,
        (
            "rank_pct",
            "selector_rank_pct",
            "selector_ts_rank_pct",
            "selector_ts_side_rank_pct",
            "meta_score_rank_pct",
            "meta_score_rank_pct_selected",
            "rank",
            "score_rank_pct",
            "model_rank_pct",
        ),
    )
    if col is None:
        score_col = _first_column(frame, ("score", "pred", "prediction", "meta_score", "base_score"))
        if score_col is None:
            return pd.Series("rank_unknown", index=frame.index)
        values = pd.to_numeric(frame[score_col], errors="coerce")
        rank = values.rank(pct=True)
    else:
        rank = pd.to_numeric(frame[col], errors="coerce")
        if rank.max(skipna=True) and float(rank.max(skipna=True)) > 1.5:
            rank = rank / 100.0
    labels = [f"{i / bands:.2f}-{(i + 1) / bands:.2f}" for i in range(bands)]
    try:
        return pd.cut(rank.clip(0.0, 1.0), bins=np.linspace(0.0, 1.0, bands + 1), labels=labels, include_lowest=True).astype(str)
    except Exception:
        return pd.Series("rank_unknown", index=frame.index)


def _month_series(frame: pd.DataFrame) -> pd.Series:
    col = _first_column(frame, ("timestamp", "__ts__", "ts", "entry_ts"))
    if col is None:
        return pd.Series("unknown", index=frame.index)
    ts = pd.to_datetime(frame[col], utc=True, errors="coerce")
    return pd.Series(ts.dt.to_period("M").astype(str), index=frame.index)


def _fold_series(frame: pd.DataFrame) -> pd.Series:
    col = _first_column(frame, ("fold", "fold_id", "decision_fold", "oof_fold_id"))
    if col is None:
        return pd.Series("unknown", index=frame.index)
    return frame[col].astype(str)


def _numeric_series(frame: pd.DataFrame, names: tuple[str, ...], default: float = 0.0) -> pd.Series:
    col = _first_column(frame, names)
    if col is None:
        return pd.Series(default, index=frame.index, dtype=np.float32)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _archetype_feature_columns(frame: pd.DataFrame, max_features: int) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        name = str(col)
        if name in {"gmm_cluster_id", "cluster_t", "ctx_gmm_cluster_id", "ctx_cluster_t"}:
            continue
        base_name = name[4:] if name.startswith("ctx_") else name
        if (
            name in ARCHETYPE_EXACT
            or base_name in ARCHETYPE_EXACT
            or any(name.startswith(prefix) or base_name.startswith(prefix) for prefix in ARCHETYPE_PREFIXES)
        ):
            series = pd.to_numeric(frame[col], errors="coerce")
            if int(series.notna().sum()) >= 50 and float(series.nunique(dropna=True)) > 2:
                cols.append(name)
    return cols[: int(max_features)]


def _stability(values: pd.Series, groups: pd.Series) -> float:
    rows: list[float] = []
    for _key, idx in groups.groupby(groups, dropna=False).groups.items():
        local = pd.to_numeric(values.loc[idx], errors="coerce")
        if int(local.notna().sum()) >= 5:
            rows.append(float(local.mean()))
    if not rows:
        return float("nan")
    signs = np.sign(np.asarray(rows, dtype=np.float32))
    dominant = 1.0 if float(np.nanmean(rows)) >= 0.0 else -1.0
    return float(np.mean(signs == dominant))


def build_effect_matrix(
    frame: pd.DataFrame,
    *,
    rank_bands: int = 5,
    max_features: int = 80,
    min_support: int = 25,
    quantile: float = 0.75,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.copy()
    work["_side"] = _side_series(work)
    work["_rank_band"] = _rank_band_series(work, int(rank_bands))
    work["_month"] = _month_series(work)
    work["_fold"] = _fold_series(work)
    work["_utility"] = _numeric_series(work, ("u_policy_net", "u_econ_net", "__u_econ_net__", "utility", "net", "mean_u"))
    work["_bad_mae"] = _numeric_series(work, ("bad_mae_1r", "bad_MAE", "bad_mae", "is_bad_mae"))
    work["_timeout"] = _numeric_series(work, ("timeout", "is_timeout", "__is_timeout__"))
    work["_full_sl"] = _numeric_series(work, ("full_stop_loss", "full_sl", "stop_loss", "__hit_sl__"))
    work["_clean_positive"] = ((work["_utility"] > 0.0) & (work["_bad_mae"] < 0.5) & (work["_timeout"] < 0.5) & (work["_full_sl"] < 0.5)).astype(np.float32)
    work["_dirty_positive"] = ((work["_utility"] > 0.0) & ((work["_bad_mae"] >= 0.5) | (work["_timeout"] >= 0.5) | (work["_full_sl"] >= 0.5))).astype(np.float32)
    base_group = ["_side", "_rank_band"]
    baseline = work.groupby(base_group, dropna=False)["_utility"].transform("mean")
    work["_residual_utility"] = work["_utility"] - baseline
    feature_cols = _archetype_feature_columns(work, int(max_features))
    rows: list[dict[str, Any]] = []
    for (side, rank_band), group in work.groupby(base_group, dropna=False):
        if len(group) < int(min_support):
            continue
        base_bad = float(group["_bad_mae"].mean())
        base_timeout = float(group["_timeout"].mean())
        base_sl = float(group["_full_sl"].mean())
        base_clean = float(group["_clean_positive"].mean())
        base_dirty = float(group["_dirty_positive"].mean())
        for feature in feature_cols:
            values = pd.to_numeric(group[feature], errors="coerce")
            if int(values.notna().sum()) < int(min_support):
                continue
            threshold = float(values.quantile(float(quantile)))
            high = values >= threshold
            support = int(high.sum())
            if support < int(min_support):
                continue
            selected = group.loc[high]
            residual = selected["_residual_utility"]
            excess_bad = float(selected["_bad_mae"].mean() - base_bad)
            excess_timeout = float(selected["_timeout"].mean() - base_timeout)
            excess_sl = float(selected["_full_sl"].mean() - base_sl)
            clean_lift = float(selected["_clean_positive"].mean() - base_clean)
            dirty_lift = float(selected["_dirty_positive"].mean() - base_dirty)
            mean_residual = float(residual.mean())
            month_stability = _stability(selected["_residual_utility"], selected["_month"])
            fold_stability = _stability(selected["_residual_utility"], selected["_fold"])
            if mean_residual > 0.0 and clean_lift > 0.0 and excess_bad <= 0.0 and excess_timeout <= 0.0:
                action = "candidate_size_lift_shadow"
            elif mean_residual < 0.0 or excess_bad > 0.03 or excess_timeout > 0.03 or dirty_lift > clean_lift:
                action = "candidate_penalty_shadow"
            else:
                action = "no_penalty_shadow"
            rows.append(
                {
                    "side": str(side),
                    "rank_band": str(rank_band),
                    "archetype_feature": str(feature),
                    "threshold_quantile": float(quantile),
                    "threshold": threshold,
                    "support": support,
                    "support_share_in_side_rank": float(support / max(len(group), 1)),
                    "residual_utility_mean": mean_residual,
                    "excess_bad_mae": excess_bad,
                    "excess_timeout": excess_timeout,
                    "excess_full_sl": excess_sl,
                    "clean_positive_lift": clean_lift,
                    "dirty_positive_lift": dirty_lift,
                    "month_stability": month_stability,
                    "fold_stability": fold_stability,
                    "action": action,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["side", "rank_band", "residual_utility_mean", "clean_positive_lift"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)
    manifest = {
        "rows": int(len(work)),
        "rank_bands": int(rank_bands),
        "archetype_feature_count": int(len(feature_cols)),
        "min_support": int(min_support),
        "quantile": float(quantile),
        "grouping": ["side", "rank_band", "archetype_feature"],
        "side_domain": ["long", "short"],
        "side_policy": "global_long_short_only; no per-strategy/head grouping",
    }
    return out, manifest


def run_report(
    *,
    input_path: Path,
    output_dir: Path,
    rank_bands: int,
    max_features: int,
    min_support: int,
    quantile: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _read_frame(input_path)
    effects, info = build_effect_matrix(
        frame,
        rank_bands=int(rank_bands),
        max_features=int(max_features),
        min_support=int(min_support),
        quantile=float(quantile),
    )
    paths = {
        "effect_matrix": output_dir / "side_archetype_effect_matrix.csv",
        "manifest": output_dir / "side_archetype_effect_matrix_manifest.json",
    }
    effects.to_csv(paths["effect_matrix"], index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        **info,
        "effect_rows": int(len(effects)),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Candidate/model ledger CSV or parquet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank-bands", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=80)
    parser.add_argument("--min-support", type=int, default=25)
    parser.add_argument("--quantile", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        input_path=args.input,
        output_dir=args.output_dir,
        rank_bands=int(args.rank_bands),
        max_features=int(args.max_features),
        min_support=int(args.min_support),
        quantile=float(args.quantile),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
