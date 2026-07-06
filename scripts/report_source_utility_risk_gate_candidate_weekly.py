#!/usr/bin/env python3
"""Weekly selected-row validation for shortlisted utility risk-gate candidates.

This is a diagnostic-only follow-up to
``run_source_utility_risk_gate_diagnostic.py``.  It replays a small set of
candidate label/feature/gate definitions, materializes the selected OOS rows,
and reports weekly path/utility stability.  It does not modify production
training artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_feature_store_model_smoke import _fit_predict, _month_model_frame  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _json_safe,
    _load_feature_store_columns,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import (  # noqa: E402
    DEFAULT_MONTHS,
    DEFAULT_QUALITY_LABELS,
    DEFAULT_SEEDS,
    _load_joined_frame,
    _parse_csv,
    _parse_int_csv,
    _source_feature_columns,
)
from scripts.run_source_utility_label_rework_diagnostic import (  # noqa: E402
    _build_target,
    _safe_numeric,
)
from scripts.run_source_utility_risk_gate_diagnostic import (  # noqa: E402
    RISK_GATES,
    _assert_gate_columns_causal,
    _gate_mask,
    _gate_specs_by_name,
    _label_specs_by_name,
    _rank_top_indices,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/utility_risk_gate_candidate_weekly"
)
DEFAULT_CANDIDATES = (
    "utility_linear_source_q80_v1|base_plus_source|all_rows|low_barrier_pressure_q50|0.01|gate_relative",
    "utility_linear_source_q80_v1|base_plus_source|all_rows|low_barrier_pressure_q50|0.03|gate_relative",
    "utility_linear_source_q80_v1|base_plus_source|risk_adjusted_capture_candidate|low_barrier_pressure_q50|0.05|budget_matched",
    "utility_linear_source_q80_v1|base_plus_source|risk_adjusted_capture_candidate|low_barrier_pressure_q50|0.10|gate_relative",
)

SELECTED_ROW_COLUMNS = [
    "candidate",
    "period",
    "__ts__",
    "__symbol__",
    "side",
    "side_name",
    "__side__",
    "timeframe",
    "candidate_id",
    "week_start",
    "label",
    "feature_set",
    "source_bucket",
    "risk_gate",
    "top_frac",
    "selection_mode",
    "score",
    "rank_in_gate",
    "bucket_rows",
    "gate_rows",
    "gate_coverage_vs_bucket",
    "u_policy_net",
    "barrier",
    "mae_norm",
    "is_timeout",
]
WEEKLY_COLUMNS = [
    "candidate",
    "period",
    "week_start",
    "label",
    "feature_set",
    "source_bucket",
    "risk_gate",
    "top_frac",
    "selection_mode",
    "rows",
    "mean_u",
    "q10_u",
    "hit_u",
    "bad_mae_1r_rate",
    "p90_mae_norm",
    "timeout_rate",
    "wide_barrier_25bps_rate",
    "top_symbol_share",
    "unique_symbols",
    "long_share",
    "short_share",
    "side_top_share",
]
AGGREGATE_COLUMNS = [
    "candidate",
    "label",
    "feature_set",
    "source_bucket",
    "risk_gate",
    "top_frac",
    "selection_mode",
    "weeks",
    "positive_weeks",
    "months",
    "total_rows",
    "mean_week_rows",
    "min_week_rows",
    "mean_u",
    "worst_week_u",
    "q25_week_u",
    "mean_bad_mae_1r_rate",
    "worst_week_bad_mae_1r_rate",
    "mean_timeout_rate",
    "mean_wide_barrier_25bps_rate",
    "max_top_symbol_share",
    "max_side_top_share",
    "median_unique_symbols",
]


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    feature_set: str
    source_bucket: str
    risk_gate: str
    top_frac: float
    selection_mode: str

    @property
    def name(self) -> str:
        frac = str(self.top_frac).replace(".", "p")
        return (
            f"{self.label}__{self.feature_set}__{self.source_bucket}__"
            f"{self.risk_gate}__top{frac}__{self.selection_mode}"
        )


def _candidate_from_string(value: str) -> CandidateSpec:
    parts = [part.strip() for part in str(value).split("|")]
    if len(parts) != 6:
        raise ValueError(
            "Candidate specs must use label|feature_set|source_bucket|risk_gate|top_frac|selection_mode"
        )
    label, feature_set, source_bucket, risk_gate, top_frac, selection_mode = parts
    if selection_mode not in {"gate_relative", "budget_matched"}:
        raise ValueError(f"Unsupported selection_mode for {value!r}: {selection_mode!r}")
    return CandidateSpec(
        label=label,
        feature_set=feature_set,
        source_bucket=source_bucket,
        risk_gate=risk_gate,
        top_frac=float(top_frac),
        selection_mode=selection_mode,
    )


def _parse_candidates(values: list[str] | None) -> list[CandidateSpec]:
    raw = values if values else list(DEFAULT_CANDIDATES)
    return [_candidate_from_string(value) for value in raw]


def _bool_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    lowered = values.astype(str).str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _source_bucket_mask(frame: pd.DataFrame, source_bucket: str) -> pd.Series:
    if source_bucket == "all_rows":
        return pd.Series(True, index=frame.index)
    if "primary_source_tag" in frame.columns:
        primary = frame["primary_source_tag"].fillna("").astype(str).eq(str(source_bucket))
        if bool(primary.any()):
            return primary
    tag_col = f"tag_{source_bucket}"
    if tag_col in frame.columns:
        return _bool_series(frame, tag_col)
    score_col = f"{source_bucket}_score"
    if score_col in frame.columns:
        score = _safe_numeric(frame[score_col])
        return score.ge(score.quantile(0.80))
    return pd.Series(False, index=frame.index)


def _week_start(ts: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(ts, utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.to_period("W-SUN")
        .apply(lambda value: value.start_time.date().isoformat() if pd.notna(value) else "")
    )


def _path_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {
            "rows": 0,
            "mean_u": float("nan"),
            "median_u": float("nan"),
            "q10_u": float("nan"),
            "hit_u": float("nan"),
            "bad_mae_1r_rate": float("nan"),
            "p90_mae_norm": float("nan"),
            "timeout_rate": float("nan"),
            "wide_barrier_25bps_rate": float("nan"),
            "wide_barrier_35bps_rate": float("nan"),
        }
    return {
        "rows": int(len(metrics)),
        "mean_u": _safe_mean(metrics["u_policy_net"]),
        "median_u": _safe_quantile(metrics["u_policy_net"], 0.50),
        "q10_u": _safe_quantile(metrics["u_policy_net"], 0.10),
        "hit_u": _safe_mean(metrics["u_policy_net"] > 0.0),
        "bad_mae_1r_rate": _safe_mean(metrics["mae_norm"] >= 1.0),
        "p90_mae_norm": _safe_quantile(metrics["mae_norm"], 0.90),
        "timeout_rate": _safe_mean(metrics["is_timeout"].astype(float)),
        "wide_barrier_25bps_rate": _safe_mean(metrics["barrier"] > 0.025),
        "wide_barrier_35bps_rate": _safe_mean(metrics["barrier"] > 0.035),
    }


def _weekly_summary(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    rows: list[dict[str, Any]] = []
    group_cols = [
        "candidate",
        "period",
        "week_start",
        "label",
        "feature_set",
        "source_bucket",
        "risk_gate",
        "top_frac",
        "selection_mode",
    ]
    for key, group in selected.groupby(group_cols, dropna=False, observed=True):
        context = dict(zip(group_cols, key, strict=False))
        metrics = pd.DataFrame(
            {
                "u_policy_net": group["u_policy_net"],
                "mae_norm": group["mae_norm"],
                "barrier": group["barrier"],
                "is_timeout": group["is_timeout"],
            }
        )
        top_symbol_share = (
            float(group["__symbol__"].value_counts(normalize=True).iloc[0])
            if "__symbol__" in group.columns and len(group)
            else float("nan")
        )
        side = _safe_numeric(group["side"]) if "side" in group.columns else pd.Series(dtype=float)
        side_name = side.map(lambda value: "short" if value < 0.0 else "long")
        side_top_share = (
            float(side_name.value_counts(normalize=True).iloc[0]) if len(side_name) else float("nan")
        )
        rows.append(
            {
                **context,
                **_path_summary(metrics),
                "top_symbol_share": top_symbol_share,
                "unique_symbols": int(group["__symbol__"].nunique()) if "__symbol__" in group.columns else 0,
                "long_share": _safe_mean(side > 0.0) if len(side) else float("nan"),
                "short_share": _safe_mean(side < 0.0) if len(side) else float("nan"),
                "side_top_share": side_top_share,
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate", "period", "week_start"], kind="mergesort")


def _aggregate_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return pd.DataFrame(columns=AGGREGATE_COLUMNS)
    rows: list[dict[str, Any]] = []
    group_cols = ["candidate", "label", "feature_set", "source_bucket", "risk_gate", "top_frac", "selection_mode"]
    for key, group in weekly.groupby(group_cols, dropna=False, observed=True):
        context = dict(zip(group_cols, key, strict=False))
        rows.append(
            {
                **context,
                "weeks": int(len(group)),
                "positive_weeks": int((_safe_numeric(group["mean_u"]) > 0.0).sum()),
                "months": int(group["period"].nunique()),
                "total_rows": int(group["rows"].sum()),
                "mean_week_rows": _safe_mean(group["rows"]),
                "min_week_rows": _safe_quantile(group["rows"], 0.0),
                "mean_u": _safe_mean(group["mean_u"]),
                "worst_week_u": _safe_quantile(group["mean_u"], 0.0),
                "q25_week_u": _safe_quantile(group["mean_u"], 0.25),
                "mean_bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "worst_week_bad_mae_1r_rate": _safe_quantile(group["bad_mae_1r_rate"], 1.0),
                "mean_timeout_rate": _safe_mean(group["timeout_rate"]),
                "mean_wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "max_top_symbol_share": _safe_quantile(group["top_symbol_share"], 1.0),
                "max_side_top_share": _safe_quantile(group.get("side_top_share", pd.Series(dtype=float)), 1.0),
                "median_unique_symbols": _safe_quantile(group["unique_symbols"], 0.50),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["mean_u", "q25_week_u"], ascending=[False, False], kind="mergesort")


def _candidate_model_groups(candidates: list[CandidateSpec]) -> dict[tuple[str, str], list[CandidateSpec]]:
    groups: dict[tuple[str, str], list[CandidateSpec]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.label, candidate.feature_set), []).append(candidate)
    return groups


def _source_columns_for_ledger(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in {"primary_source_tag"} or col.startswith("tag_") or col.endswith("_score"):
            cols.append(col)
    keep = [
        "trend_path_score",
        "shock_impulse_score",
        "execution_quality_score",
        "execution_risk_score",
        "barrier_pressure_score",
        "barrier_relief_score",
        "quiet_continuation_score",
        "loud_breakout_impulse_score",
        "dirty_shock_avoid_score",
        "risk_adjusted_capture_candidate_score",
        "clean_run_entry_score",
        "misleading_location_risk_score",
    ]
    for col in keep:
        if col in frame.columns and col not in cols:
            cols.append(col)
    return list(dict.fromkeys(cols))


def run_report(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
    candidates: list[CandidateSpec],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_names = sorted({candidate.label for candidate in candidates})
    gate_names = sorted({candidate.risk_gate for candidate in candidates})
    labels = {spec.name: spec for spec in _label_specs_by_name(label_names)}
    gates = {spec.name: spec for spec in _gate_specs_by_name(gate_names)}
    _assert_gate_columns_causal(list(gates.values()))

    frame, join_report = _load_joined_frame(quality_labels_path=quality_labels_path, labels_path=labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    metrics = _path_metrics(frame)
    base_features = list(feature_matrix.columns)
    source_features = _source_feature_columns(frame)
    feature_map = {"base": base_features, "base_plus_source": list(dict.fromkeys(base_features + source_features))}
    source_ledger_cols = _source_columns_for_ledger(frame)

    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    candidate_groups = _candidate_model_groups(candidates)
    selected_rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for month in months:
        valid_mask = month_period.eq(month)
        train_mask_base = month_period < month
        if train_lookback_months is not None and int(train_lookback_months) > 0:
            prior_months = sorted(month_period[train_mask_base].dropna().unique())
            keep = set(prior_months[-int(train_lookback_months) :])
            train_mask_base = train_mask_base & month_period.isin(keep)
        if int(valid_mask.sum()) < int(min_valid_rows):
            continue
        for (label_name, feature_set), group_candidates in candidate_groups.items():
            label_spec = labels[label_name]
            target, weights, _label_report = _build_target(
                frame=frame,
                metrics=metrics,
                train_mask=train_mask_base,
                valid_mask=valid_mask,
                spec=label_spec,
            )
            train_target_mask = train_mask_base & target["target_soft"].notna() & weights.gt(0.0)
            if int(train_target_mask.sum()) < int(min_train_rows):
                diagnostics.append(
                    {
                        "period": month,
                        "label": label_name,
                        "feature_set": feature_set,
                        "skipped": True,
                        "reason": "too_few_train_rows",
                        "train_rows": int(train_target_mask.sum()),
                        "valid_rows": int(valid_mask.sum()),
                    }
                )
                continue
            features = feature_map.get(feature_set)
            if not features:
                diagnostics.append(
                    {
                        "period": month,
                        "label": label_name,
                        "feature_set": feature_set,
                        "skipped": True,
                        "reason": "unknown_feature_set",
                        "train_rows": int(train_target_mask.sum()),
                        "valid_rows": int(valid_mask.sum()),
                    }
                )
                continue
            x_train, x_valid = _month_model_frame(
                frame,
                train_mask=train_target_mask,
                valid_mask=valid_mask,
                features=features,
            )
            pred_matrix = np.vstack(
                [
                    _fit_predict(
                        x_train=x_train,
                        y_train=target.loc[train_target_mask, "target_soft"],
                        w_train=weights.loc[train_target_mask],
                        x_valid=x_valid,
                        seed=seed,
                    )
                    for seed in seeds
                ]
            )
            score = pd.Series(np.nan, index=frame.index, dtype=np.float32)
            score.loc[valid_mask] = np.mean(pred_matrix, axis=0).astype(np.float32)
            for candidate in group_candidates:
                bucket_mask = valid_mask & _source_bucket_mask(frame, candidate.source_bucket)
                gate_mask, gate_report = _gate_mask(frame, train_mask_base, gates[candidate.risk_gate])
                scope_mask = bucket_mask & gate_mask
                bucket_rows = int(bucket_mask.sum())
                gate_rows = int(scope_mask.sum())
                if candidate.selection_mode == "budget_matched":
                    k = max(1, int(math.ceil(candidate.top_frac * bucket_rows)))
                else:
                    k = max(1, int(math.ceil(candidate.top_frac * gate_rows)))
                scope_idx = np.flatnonzero(scope_mask.to_numpy())
                scope_score = score.iloc[scope_idx].reset_index(drop=True)
                selected_local = _rank_top_indices(scope_score, k)
                selected_idx = scope_idx[selected_local] if len(selected_local) else np.array([], dtype=np.int64)
                selected_metrics = metrics.iloc[selected_idx].copy()
                diagnostics.append(
                    {
                        "period": month,
                        "candidate": candidate.name,
                        "label": candidate.label,
                        "feature_set": candidate.feature_set,
                        "source_bucket": candidate.source_bucket,
                        "risk_gate": candidate.risk_gate,
                        "top_frac": float(candidate.top_frac),
                        "selection_mode": candidate.selection_mode,
                        "skipped": False,
                        "train_rows": int(train_target_mask.sum()),
                        "valid_rows": int(valid_mask.sum()),
                        "bucket_rows": bucket_rows,
                        "gate_rows": gate_rows,
                        "gate_coverage_vs_bucket": float(gate_rows / bucket_rows) if bucket_rows else 0.0,
                        "selected_rows": int(len(selected_idx)),
                        "score_ic_u_gate": _spearman(score.loc[scope_mask], metrics.loc[scope_mask, "u_policy_net"]),
                        "missing_gate_columns": ",".join(gate_report.get("missing_gate_columns", [])),
                        "gate_thresholds_json": json.dumps(_json_safe(gate_report.get("thresholds", {})), sort_keys=True),
                    }
                )
                if not len(selected_idx):
                    continue
                selected_frame = frame.iloc[selected_idx].copy()
                contract_cols = [
                    col
                    for col in ("side", "side_name", "__side__", "timeframe", "candidate_id")
                    if col in selected_frame.columns
                ]
                selected_frame = selected_frame[
                    ["__ts__", "__symbol__"] + contract_cols + [col for col in source_ledger_cols if col in selected_frame.columns]
                ].copy()
                if "side" not in selected_frame.columns and "side" in selected_metrics.columns:
                    selected_frame["side"] = selected_metrics["side"].to_numpy(dtype=np.int8, copy=False)
                if "side_name" not in selected_frame.columns and "side" in selected_frame.columns:
                    selected_frame["side_name"] = np.where(
                        _safe_numeric(selected_frame["side"]) < 0.0,
                        "short",
                        "long",
                    )
                selected_frame.insert(0, "candidate", candidate.name)
                selected_frame.insert(1, "period", month)
                selected_frame["week_start"] = _week_start(selected_frame["__ts__"])
                selected_frame["label"] = candidate.label
                selected_frame["feature_set"] = candidate.feature_set
                selected_frame["source_bucket"] = candidate.source_bucket
                selected_frame["risk_gate"] = candidate.risk_gate
                selected_frame["top_frac"] = float(candidate.top_frac)
                selected_frame["selection_mode"] = candidate.selection_mode
                selected_frame["score"] = score.iloc[selected_idx].to_numpy(dtype=np.float32, copy=False)
                selected_frame["rank_in_gate"] = np.arange(1, len(selected_frame) + 1, dtype=np.int32)
                selected_frame["bucket_rows"] = bucket_rows
                selected_frame["gate_rows"] = gate_rows
                selected_frame["gate_coverage_vs_bucket"] = float(gate_rows / bucket_rows) if bucket_rows else 0.0
                selected_frame["u_policy_net"] = selected_metrics["u_policy_net"].to_numpy()
                selected_frame["barrier"] = selected_metrics["barrier"].to_numpy()
                selected_frame["mae_norm"] = selected_metrics["mae_norm"].to_numpy()
                selected_frame["is_timeout"] = selected_metrics["is_timeout"].to_numpy()
                selected_rows.append(selected_frame)

    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(columns=SELECTED_ROW_COLUMNS)
    weekly = _weekly_summary(selected)
    aggregate = _aggregate_weekly(weekly)
    diagnostics_frame = pd.DataFrame(diagnostics)

    paths = {
        "selected_rows_parquet": output_dir / "candidate_selected_rows.parquet",
        "selected_rows_csv": output_dir / "candidate_selected_rows.csv",
        "weekly": output_dir / "candidate_weekly_metrics.csv",
        "aggregate": output_dir / "candidate_weekly_aggregate.csv",
        "diagnostics": output_dir / "candidate_replay_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    selected.to_parquet(paths["selected_rows_parquet"], index=False)
    selected.to_csv(paths["selected_rows_csv"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics_frame.to_csv(paths["diagnostics"], index=False)
    manifest = {
        "scope": "source_utility_risk_gate_candidate_weekly",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "utility_source": metrics.attrs.get("utility_source"),
        "months": list(months),
        "seeds": [int(seed) for seed in seeds],
        "candidates": [candidate.__dict__ | {"name": candidate.name} for candidate in candidates],
        "join_report": join_report,
        "feature_store": feature_report,
        "base_feature_count": int(len(base_features)),
        "source_feature_count": int(len(source_features)),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, aggregate, weekly, diagnostics_frame, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _sort_if_present(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [col for col in cols if col in frame.columns]
    if frame.empty or not present:
        return frame
    return frame.sort_values(present, kind="mergesort")


def _write_markdown(
    output_dir: Path,
    aggregate: pd.DataFrame,
    weekly: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "candidate_weekly_report.md"
    aggregate_cols = [
        "candidate",
        "weeks",
        "positive_weeks",
        "total_rows",
        "mean_week_rows",
        "min_week_rows",
        "mean_u",
        "worst_week_u",
        "q25_week_u",
        "mean_bad_mae_1r_rate",
        "worst_week_bad_mae_1r_rate",
        "mean_timeout_rate",
        "mean_wide_barrier_25bps_rate",
        "max_top_symbol_share",
        "median_unique_symbols",
    ]
    weekly_cols = [
        "candidate",
        "period",
        "week_start",
        "rows",
        "mean_u",
        "q10_u",
        "hit_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "top_symbol_share",
        "unique_symbols",
    ]
    diag_cols = [
        "period",
        "candidate",
        "bucket_rows",
        "gate_rows",
        "gate_coverage_vs_bucket",
        "selected_rows",
        "score_ic_u_gate",
    ]
    lines = [
        "# Source Utility Risk Gate Candidate Weekly Report",
        "",
        "Scope: selected-row weekly validation for shortlisted utility risk-gate candidates.",
        "",
        f"Rows joined to label ledger: `{manifest['rows']}`",
        f"Utility source: `{manifest.get('utility_source', '')}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Candidates: `{len(manifest['candidates'])}`",
        "",
        "## Aggregate Weekly Stability",
        "",
        _table(aggregate, aggregate_cols, limit=40),
        "",
        "## Weekly Metrics",
        "",
        _table(_sort_if_present(weekly, ["candidate", "period", "week_start"]), weekly_cols, limit=120),
        "",
        "## Replay Diagnostics",
        "",
        _table(_sort_if_present(diagnostics, ["candidate", "period"]), diag_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Selected rows parquet: `{manifest['outputs']['selected_rows_parquet']}`",
        f"- Selected rows CSV: `{manifest['outputs']['selected_rows_csv']}`",
        f"- Weekly metrics: `{manifest['outputs']['weekly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="label|feature_set|source_bucket|risk_gate|top_frac|selection_mode. Repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        candidates=_parse_candidates(args.candidate),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
