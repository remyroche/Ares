#!/usr/bin/env python3
"""Materialize the Stage 15 source-conditioned clean-utility label artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_first_touch_execution_proxy_ablation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _first_touch_metrics,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_numeric,
    _safe_quantile,
)
from scripts.run_label_two_head_abstention_utility_proxy import (  # noqa: E402
    _global_bad_soft,
    _target_for_selection,
    _utility_targets,
)
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _source_context,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    _event_confirmation_features,
)


DEFAULT_OUTPUT_RUN_ID = "20260702_170000_stage15_quiet_mid_cleanutil_labels"
DEFAULT_SOURCE = "quiet_mid"
DEFAULT_UTILITY_TARGET = "clean_utility"
DEFAULT_BAD_GATE_COVERAGE = 0.05
DEFAULT_TOP_K = 20


def _monthly_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    month = frame["__ts__"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for period, group in frame.groupby(month, observed=True, dropna=False):
        rows.append(
            {
                "period": str(period),
                "rows": int(len(group)),
                "target_soft_mean": _safe_mean(group["__stage15_target_soft__"]),
                "target_hard_rate": _safe_mean(group["__stage15_target_hard__"]),
                "u_policy_net_mean": _safe_mean(group["__u_policy_net__"]),
                "u_policy_net_positive_rate": _safe_mean(_safe_numeric(group["__u_policy_net__"]) > 0.0),
                "first_touch_hit_rate": _safe_mean(group["__first_touch_hit__"]),
                "first_touch_stop_rate": _safe_mean(group["__first_touch_stop__"]),
                "first_touch_timeout_rate": _safe_mean(group["__first_touch_timeout__"]),
                "first_touch_bad_mae_to_sl_rate": _safe_mean(
                    _safe_numeric(group["__first_touch_mae_to_sl__"]) >= 1.0
                ),
                "p90_first_touch_mae_to_sl": _safe_quantile(group["__first_touch_mae_to_sl__"], 0.90),
                "p90_first_touch_bar": _safe_quantile(group["__first_touch_bar__"], 0.90),
                "source_quiet_score_mean": _safe_mean(group.get("__stage15_source_quiet_score__")),
            }
        )
    return rows


def _write_markdown(output_dir: Path, summary: dict[str, Any]) -> Path:
    path = output_dir / "stage15_source_label_artifact.md"
    monthly = pd.DataFrame(summary["monthly"])
    if monthly.empty:
        monthly_table = "No rows."
    else:
        view = monthly.copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        monthly_table = view.to_markdown(index=False)
    lines = [
        "# Stage 15 Source Label Artifact",
        "",
        f"Source labels: `{summary['source_labels_dir']}`",
        f"Output labels: `{summary['output_labels_dir']}`",
        f"Source: `{summary['source']}`",
        f"Utility target: `{summary['utility_target']}`",
        f"Rows: `{summary['rows']}`",
        f"Symbols: `{summary['symbols']}`",
        f"Timestamp range: `{summary['timestamp_min']}` to `{summary['timestamp_max']}`",
        "",
        "The artifact is filtered to the Stage 15 source mask. It stores the frozen target in `__stage15_target_soft__` and `__stage15_target_hard__`.",
        "",
        "## Monthly",
        "",
        monthly_table,
        "",
        "## Outputs",
        "",
        f"- Parquet: `{summary['dataset_file']}`",
        f"- Manifest: `{summary['manifest']}`",
        f"- Summary: `{summary['summary']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_materialization(
    *,
    source_labels_dir: Path,
    output_labels_dir: Path,
    output_run_id: str,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    event_feature_store_features: list[str],
    source: str,
    utility_target: str,
    run_gap_hours: float,
    bad_gate_coverage: float,
    top_k: int,
    overwrite: bool,
) -> dict[str, Any]:
    if output_labels_dir.exists() and any(output_labels_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_labels_dir} already exists; pass --overwrite to replace files")
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_labels(source_labels_dir)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    selected_features = list(dict.fromkeys(list(selected_features) + list(event_feature_store_features)))
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    work = frame.copy()
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in work.columns]
        work = pd.concat([work.reset_index(drop=True), feature_matrix.loc[:, new_cols].reset_index(drop=True)], axis=1)
    event_features, event_report = _event_confirmation_features(
        work,
        event_features=event_feature_store_features,
    )
    if not event_features.empty:
        new_event_cols = [col for col in event_features.columns if col not in work.columns]
        work = pd.concat([work.reset_index(drop=True), event_features.loc[:, new_event_cols].reset_index(drop=True)], axis=1)

    context = _source_context(work)
    source_masks = _build_sources(work, context, run_gap_hours=run_gap_hours)
    if source not in source_masks:
        raise ValueError(f"Unknown source: {source}")
    source_mask = source_masks[source].reindex(work.index, fill_value=False).astype(bool)

    metrics = _path_metrics(frame)
    ft = _first_touch_metrics(frame, metrics)
    utility_map = _utility_targets(frame, ft)
    if utility_target not in utility_map:
        raise ValueError(f"Unknown utility target: {utility_target}")
    utility_soft = utility_map[utility_target]
    bad_soft = _global_bad_soft(ft)
    target = _target_for_selection(ft, utility_soft, bad_soft)

    out = frame.loc[source_mask].copy().reset_index(drop=True)
    selected_context = context.loc[source_mask].reset_index(drop=True)
    selected_ft = ft.loc[source_mask].reset_index(drop=True)
    selected_target = target.loc[source_mask].reset_index(drop=True)
    selected_utility = utility_soft.loc[source_mask].reset_index(drop=True)
    selected_bad = bad_soft.loc[source_mask].reset_index(drop=True)

    for col in ("__mfe_ret__", "__mae_ret__", "__bars_to_mfe__", "__is_timeout__"):
        if col in out.columns:
            out[f"__stage15_original_{col.strip('_')}__"] = out[col].to_numpy(copy=False)

    barrier = _safe_numeric(out["__barrier_pct__"]).abs().clip(lower=1e-8)
    out["__mfe_ret__"] = (_safe_numeric(selected_ft["first_touch_mfe_to_tp"]) * barrier).astype(np.float32)
    out["__mae_ret__"] = (_safe_numeric(selected_ft["first_touch_mae_to_sl"]) * barrier).astype(np.float32)
    out["__bars_to_mfe__"] = _safe_numeric(selected_ft["first_touch_bar"]).astype(np.float32)
    out["__is_timeout__"] = _safe_numeric(selected_ft["first_touch_timeout"]).astype(np.float32)

    out["__stage15_target_soft__"] = _safe_numeric(selected_utility).clip(0.0, 1.0).astype(np.float32)
    out["__stage15_target_hard__"] = _safe_numeric(selected_target["target_hard"]).clip(0.0, 1.0).astype(np.float32)
    out["__stage15_bad_soft__"] = _safe_numeric(selected_bad).clip(0.0, 1.0).astype(np.float32)
    out["__stage15_source_mask__"] = np.ones(len(out), dtype=np.float32)
    out["__stage15_bad_gate_coverage__"] = np.float32(float(bad_gate_coverage))
    out["__stage15_top_k__"] = np.float32(float(top_k))
    out["__stage15_source_quiet_score__"] = _safe_numeric(
        selected_context.get("source_quiet_score")
    ).astype(np.float32)
    out["__stage15_source_loud_intensity__"] = _safe_numeric(
        selected_context.get("source_loud_intensity")
    ).astype(np.float32)
    out["__stage15_source_event_quality__"] = _safe_numeric(
        selected_context.get("source_event_quality")
    ).astype(np.float32)

    dataset_file = "train_stage15_quiet_mid_cleanutil.parquet"
    dataset_path = output_labels_dir / dataset_file
    out.to_parquet(dataset_path, index=False)

    source_manifest_path = source_labels_dir / "labels_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    summary = {
        "output_run_id": str(output_run_id),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_labels_dir": str(source_labels_dir),
        "source_manifest": str(source_manifest_path) if source_manifest_path.exists() else "",
        "output_labels_dir": str(output_labels_dir),
        "source": str(source),
        "utility_target": str(utility_target),
        "bad_gate_mode": "train_coverage",
        "bad_gate_coverage": float(bad_gate_coverage),
        "top_k": int(top_k),
        "selection_policy": "fit_profit_floors",
        "rows_source_total": int(len(frame)),
        "rows": int(len(out)),
        "source_row_frac": float(len(out) / len(frame)) if len(frame) else 0.0,
        "symbols": int(out["__symbol__"].nunique(dropna=True)),
        "timestamp_min": out["__ts__"].min(),
        "timestamp_max": out["__ts__"].max(),
        "target_soft_mean": _safe_mean(out["__stage15_target_soft__"]),
        "target_soft_std": float(_safe_numeric(out["__stage15_target_soft__"]).std(ddof=0)),
        "target_hard_rate": _safe_mean(out["__stage15_target_hard__"]),
        "u_policy_net_mean": _safe_mean(out["__u_policy_net__"]),
        "u_policy_net_positive_rate": _safe_mean(_safe_numeric(out["__u_policy_net__"]) > 0.0),
        "first_touch_bad_mae_to_sl_rate": _safe_mean(_safe_numeric(out["__first_touch_mae_to_sl__"]) >= 1.0),
        "p90_first_touch_mae_to_sl": _safe_quantile(out["__first_touch_mae_to_sl__"], 0.90),
        "feature_store": feature_store_report,
        "event_feature_report": event_report,
        "dataset_file": str(dataset_path),
        "monthly": _monthly_summary(out),
    }
    manifest = {
        "run_id": str(output_run_id),
        "source_labels_dir": str(source_labels_dir),
        "source_manifest": source_manifest,
        "stage15_materialization": {
            key: value
            for key, value in summary.items()
            if key not in {"monthly", "dataset_file"}
        },
        "datasets": {
            "train_stage15_quiet_mid_cleanutil": {
                "file": dataset_file,
                "rows": int(len(out)),
                "columns": list(out.columns),
            }
        },
    }
    manifest_path = output_labels_dir / "labels_manifest.json"
    summary_path = output_labels_dir / "stage15_source_label_summary.json"
    markdown_path = _write_markdown(
        output_labels_dir,
        {
            **summary,
            "manifest": str(manifest_path),
            "summary": str(summary_path),
        },
    )
    summary["manifest"] = str(manifest_path)
    summary["summary"] = str(summary_path)
    summary["markdown"] = str(markdown_path)
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-run-id", default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--output-labels-dir", type=Path, default=None)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--event-feature-store-features", default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--utility-target", default=DEFAULT_UTILITY_TARGET)
    parser.add_argument("--run-gap-hours", type=float, default=24.0)
    parser.add_argument("--bad-gate-coverage", type=float, default=DEFAULT_BAD_GATE_COVERAGE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_labels_dir = args.output_labels_dir
    if output_labels_dir is None:
        output_labels_dir = Path("data_perp/artifacts") / str(args.output_run_id) / "labels"
    result = run_materialization(
        source_labels_dir=args.source_labels_dir,
        output_labels_dir=output_labels_dir,
        output_run_id=str(args.output_run_id),
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        event_feature_store_features=_parse_csv(
            args.event_feature_store_features,
            DEFAULT_EVENT_FEATURE_STORE_FEATURES,
        ),
        source=str(args.source),
        utility_target=str(args.utility_target),
        run_gap_hours=float(args.run_gap_hours),
        bad_gate_coverage=float(args.bad_gate_coverage),
        top_k=int(args.top_k),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(_json_safe(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
