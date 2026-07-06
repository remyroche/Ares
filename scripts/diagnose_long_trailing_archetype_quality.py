#!/usr/bin/env python3
"""Diagnose long-side trailing-label archetype quality.

This is an in-sample archetype diagnostic, not an OOS trading claim. It checks
whether observable regime buckets or live-predictable AE/GMM state clusters
separate clean long trailing-profit outcomes from path-dirty outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import fit_ae_gmm_state, transform_ae_gmm_features  # noqa: E402
from scripts.run_first_touch_label_training_smoke import _first_touch_eval_metrics  # noqa: E402
from scripts.run_label_feature_store_model_smoke import _fold_ae_gmm_economic_targets  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
)


DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/20260704_s55_long_mixed_trailing_tp060_sl100_tr020_fast16_cost100bps_labels/labels"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/s57_long_archetype_quality_diagnosis_v1"
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _bucket_numeric(series: pd.Series, *, q: int = 5) -> pd.Series:
    values = _safe_numeric(series)
    finite = values[np.isfinite(values.to_numpy(dtype=np.float64, copy=False))]
    if len(finite) == 0:
        return pd.Series("missing", index=series.index, dtype=object)
    unique = finite.drop_duplicates()
    if len(unique) <= 12:
        return values.round(0).astype("Int64").astype(str).replace("<NA>", "missing")
    try:
        return pd.qcut(values, q=int(q), duplicates="drop").astype(str).replace("nan", "missing")
    except ValueError:
        return pd.Series("flat", index=series.index, dtype=object)


def _metric_row(frame: pd.DataFrame, metrics: pd.DataFrame, name: str, bucket: str) -> dict[str, Any]:
    net = _safe_numeric(metrics["first_touch_net"])
    gross = net + _safe_numeric(metrics.get("round_trip_cost", pd.Series(0.0, index=metrics.index))).fillna(0.0)
    clean = _safe_numeric(metrics["clean_first_touch_exec"]).fillna(0.0)
    stop = _safe_numeric(metrics["first_touch_stop"]).fillna(0.0)
    timeout = _safe_numeric(metrics["first_touch_timeout"]).fillna(0.0)
    bad = _safe_numeric(metrics["first_touch_mae_to_sl"]).ge(1.0)
    ts = pd.to_datetime(frame["__ts__"], errors="coerce")
    monthly = pd.DataFrame({"month": ts.dt.to_period("M").astype(str), "net": net})
    month_mean = monthly.groupby("month", observed=True)["net"].mean()
    return {
        "archetype_feature": str(name),
        "bucket": str(bucket),
        "rows": int(len(frame)),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)) if "__symbol__" in frame.columns else 0,
        "months": int(monthly["month"].nunique(dropna=True)),
        "mean_net": _safe_mean(net),
        "mean_gross": _safe_mean(gross),
        "q10_net": _safe_quantile(net, 0.10),
        "hit_net": _safe_mean(net > 0.0),
        "clean_exec_rate": _safe_mean(clean),
        "stop_rate": _safe_mean(stop),
        "timeout_rate": _safe_mean(timeout),
        "bad_mae_to_sl_rate": _safe_mean(bad),
        "p90_mae_to_sl": _safe_quantile(metrics["first_touch_mae_to_sl"], 0.90),
        "p90_first_touch_bar": _safe_quantile(metrics["first_touch_bar"], 0.90),
        "positive_months": int((month_mean > 0.0).sum()) if len(month_mean) else 0,
        "worst_month_net": float(month_mean.min()) if len(month_mean) else float("nan"),
        "top_symbol_share": float(frame["__symbol__"].astype(str).value_counts(normalize=True).iloc[0])
        if len(frame) and "__symbol__" in frame.columns
        else float("nan"),
    }


def _summarize_buckets(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_name: str,
    buckets: pd.Series,
    *,
    min_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    local = frame.reset_index(drop=True)
    metric_local = metrics.reset_index(drop=True)
    bucket_local = buckets.reset_index(drop=True).astype(str)
    for bucket, idx in bucket_local.groupby(bucket_local, sort=False).groups.items():
        pos = np.fromiter(idx, dtype=np.int64)
        if len(pos) < int(min_rows):
            continue
        rows.append(
            _metric_row(
                local.iloc[pos].reset_index(drop=True),
                metric_local.iloc[pos].reset_index(drop=True),
                feature_name,
                str(bucket),
            )
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["clean_exec_rate", "mean_net", "bad_mae_to_sl_rate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _observable_bucket_diagnostics(frame: pd.DataFrame, metrics: pd.DataFrame, *, min_rows: int) -> pd.DataFrame:
    candidates: list[str] = []
    for col in frame.columns:
        lower = str(col).lower()
        if (
            lower.startswith("__regime_")
            or lower.startswith("g_")
            or any(token in lower for token in ("volatility_zscore", "peer_resid", "return_autocorr", "funding", "oi_"))
        ):
            if pd.api.types.is_numeric_dtype(frame[col]):
                candidates.append(str(col))
    parts: list[pd.DataFrame] = []
    for col in list(dict.fromkeys(candidates))[:80]:
        buckets = _bucket_numeric(frame[col])
        diag = _summarize_buckets(frame, metrics, col, buckets, min_rows=min_rows)
        if not diag.empty:
            parts.append(diag)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _fit_state_archetypes(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    max_train_rows: int,
    ae_max_iter: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    work = frame.reset_index(drop=True)
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in work.columns]
        if new_cols:
            work = pd.concat(
                [work, feature_matrix.loc[:, new_cols].reset_index(drop=True).astype(np.float32, copy=False)],
                axis=1,
                copy=False,
            )
    feature_cols = _feature_columns(work)
    x = work[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True)
    x = x.fillna(med).fillna(0.0).astype(np.float32, copy=False)
    state = fit_ae_gmm_state(
        x,
        economic_targets=_fold_ae_gmm_economic_targets(metrics.reset_index(drop=True), train_frame=work),
        random_state=int(seed),
        max_train_rows=int(max_train_rows),
        ae_max_iter=int(ae_max_iter),
        require_both_sides=False,
        min_side_cluster_frac=0.02,
        min_side_cluster_rows=10,
    )
    manifest = {
        "feature_store": feature_report,
        "feature_count": int(len(feature_cols)),
        "state_enabled": bool(state.get("enabled", False)),
        "state_reason": state.get("reason"),
        "state_n_components": int(state.get("gmm_n_components", 0) or 0),
        "state_selected_config": state.get("selected_config", {}),
        "state_hpo_report_count": int(state.get("hpo_report_count", 0) or 0),
    }
    if not bool(state.get("enabled", False)):
        return pd.DataFrame(), pd.DataFrame(), manifest
    generated = transform_ae_gmm_features(x, state, index=work.index).reset_index(drop=True)
    if "gmm_cluster_id" in generated.columns:
        cluster_diag = _summarize_buckets(
            work,
            metrics,
            "gmm_cluster_id",
            generated["gmm_cluster_id"],
            min_rows=max(int(min_rows := 200), 1),
        )
    else:
        cluster_diag = pd.DataFrame()
    continuous_parts: list[pd.DataFrame] = []
    for col in generated.columns:
        lower = str(col).lower()
        if col == "gmm_cluster_id" or "prob" in lower or "posterior" in lower:
            continue
        if pd.api.types.is_numeric_dtype(generated[col]):
            diag = _summarize_buckets(
                work,
                metrics,
                f"state::{col}",
                _bucket_numeric(generated[col]),
                min_rows=200,
            )
            if not diag.empty:
                continuous_parts.append(diag)
    state_bucket_diag = pd.concat(continuous_parts, ignore_index=True) if continuous_parts else pd.DataFrame()
    return cluster_diag, state_bucket_diag, manifest


def _write_report(
    output_dir: Path,
    overall: pd.DataFrame,
    observable: pd.DataFrame,
    clusters: pd.DataFrame,
    state_buckets: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "long_trailing_archetype_quality.md"

    def table(df: pd.DataFrame, cols: list[str], limit: int = 20) -> str:
        if df.empty:
            return "No rows."
        view = df[[c for c in cols if c in df.columns]].head(limit).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "archetype_feature",
        "bucket",
        "rows",
        "symbols",
        "mean_net",
        "q10_net",
        "hit_net",
        "clean_exec_rate",
        "stop_rate",
        "timeout_rate",
        "bad_mae_to_sl_rate",
        "p90_mae_to_sl",
        "positive_months",
        "worst_month_net",
    ]
    lines = [
        "# Long Trailing Archetype Quality Diagnosis",
        "",
        "Scope: in-sample diagnosis of whether current long labels have useful observable/state archetypes. This does not claim OOS trading performance.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Period: `{manifest['timestamp_min']}` to `{manifest['timestamp_max']}`",
        f"AE/GMM state enabled: `{manifest['state'].get('state_enabled')}`; components: `{manifest['state'].get('state_n_components')}`",
        "",
        "## Overall",
        "",
        table(overall, cols, limit=5),
        "",
        "## Best Observable Buckets",
        "",
        table(observable.sort_values(["clean_exec_rate", "mean_net"], ascending=[False, False]), cols, limit=25),
        "",
        "## Best GMM Clusters",
        "",
        table(clusters, cols, limit=25),
        "",
        "## Best Continuous State Buckets",
        "",
        table(state_buckets.sort_values(["clean_exec_rate", "mean_net"], ascending=[False, False]), cols, limit=25),
        "",
        "## Outputs",
        "",
        f"- Observable buckets: `{manifest['outputs']['observable_buckets']}`",
        f"- GMM clusters: `{manifest['outputs']['gmm_clusters']}`",
        f"- State buckets: `{manifest['outputs']['state_buckets']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnosis(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    max_train_rows: int,
    ae_max_iter: int,
    min_rows: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path).reset_index(drop=True)
    metrics = _first_touch_eval_metrics(frame, _path_metrics(frame)).reset_index(drop=True)
    overall = pd.DataFrame([_metric_row(frame, metrics, "overall", "all")])
    observable = _observable_bucket_diagnostics(frame, metrics, min_rows=int(min_rows))
    clusters, state_buckets, state_manifest = _fit_state_archetypes(
        frame,
        metrics,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        max_train_rows=max_train_rows,
        ae_max_iter=ae_max_iter,
        seed=seed,
    )
    paths = {
        "overall": output_dir / "overall.csv",
        "observable_buckets": output_dir / "observable_archetype_buckets.csv",
        "gmm_clusters": output_dir / "gmm_cluster_archetypes.csv",
        "state_buckets": output_dir / "continuous_state_archetype_buckets.csv",
        "manifest": output_dir / "manifest.json",
    }
    overall.to_csv(paths["overall"], index=False)
    observable.to_csv(paths["observable_buckets"], index=False)
    clusters.to_csv(paths["gmm_clusters"], index=False)
    state_buckets.to_csv(paths["state_buckets"], index=False)
    manifest = {
        "scope": "long_trailing_archetype_quality_in_sample",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "state": state_manifest,
        "min_rows": int(min_rows),
        "max_train_rows": int(max_train_rows),
        "ae_max_iter": int(ae_max_iter),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    report = _write_report(output_dir, overall, observable, clusters, state_buckets, manifest)
    manifest["outputs"]["report"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--ae-max-iter", type=int, default=64)
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=57)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnosis(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        max_train_rows=int(args.max_train_rows),
        ae_max_iter=int(args.ae_max_iter),
        min_rows=int(args.min_rows),
        seed=int(args.seed),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
