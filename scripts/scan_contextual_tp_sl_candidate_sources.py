#!/usr/bin/env python3
"""Scan contextual TP/SL candidate sources for frozen validation readiness.

The frozen validation runner needs a source directory containing per-arm
candidate parquet files under `portfolio_replay/`.  This scanner finds those
directories and reports timestamp coverage, row counts, active heads, and
whether enough rows exist after an optional cutoff to justify replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency fallback
    pq = None


REQUIRED_ARMS = ("static", "rank_only", "joint_all", "independent_all")
OPTIONAL_ARMS = ("performance_only", "best_by_head")

DIAGNOSTIC_GROUPS: Mapping[str, Sequence[str]] = {
    "uncertainty": (
        "generated_score_uncertainty_p1mp",
        "generated_score_entropy",
        "oof_prob_uncertainty",
        "oof_contrib_entropy",
        "oof_rank_bin_se_oof",
        "oof_score_path_std",
        "oof_score_path_volatility",
        "oof_rank_path_std",
        "oof_score_reversal_count",
        "generated_score_abs_distance_from_half",
        "oof_score_margin_top10",
        "oof_score_margin_top20",
        "oof_score_margin_top30",
        "oof_rank_margin_top10",
        "oof_rank_margin_top20",
        "oof_rank_margin_top30",
    ),
    "drift": (
        "generated_score_abs_diff_1",
        "generated_score_abs_diff_4",
        "generated_score_abs_diff_24",
        "generated_score_abs_minus_prev24_mean",
        "generated_score_prev24_std",
        "generated_strategy_score_shift_abs_z",
        "oof_feature_drift_psi_core",
        "oof_feature_drift_ks_core",
        "oof_feature_drift_cov_shift",
    ),
    "ood": (
        "generated_strategy_score_ood_abs_z",
        "generated_strategy_barrier_ood_abs_z",
        "generated_strategy_friction_ood_abs_z",
        "oof_dae_reconstruction_error",
        "oof_dae_reconstruction_error_zscore",
        "oof_latent_mahalanobis_drift",
        "oof_support_gap",
        "oof_rare_leaf_fraction",
        "oof_leaf_count_mean",
        "oof_leaf_count_median",
        "oof_leaf_count_q25",
        "oof_leaf_count_p10",
        "oof_leaf_count_min",
        "oof_leaf_train_freq_mean",
        "oof_leaf_train_freq_p10",
        "oof_leaf_train_freq_min",
    ),
    "performance": (
        "generated_hr_surprise_24",
        "generated_hr_surprise_96",
        "generated_weighted_hr_surprise_24",
        "generated_weighted_hr_surprise_96",
        "generated_loss_rate_24",
        "generated_loss_rate_96",
        "generated_matured_count_24",
        "generated_matured_count_96",
    ),
    "recent_hr_surprise": (
        "generated_hr_surprise_24",
        "generated_hr_surprise_96",
        "generated_weighted_hr_surprise_24",
        "generated_weighted_hr_surprise_96",
    ),
}


def _json_safe(value: Any) -> Any:
    if not isinstance(value, (dict, list, tuple)):
        try:
            missing = pd.isna(value)
        except Exception:
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _head_name(strategy_id: pd.Series) -> pd.Series:
    text = strategy_id.astype(str)
    return text.str.extract(r"^(short_bollinger|long_bars|long_dist|short_asset)", expand=False)


def _source_dirs(roots: List[Path]) -> List[Path]:
    dirs = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("portfolio_replay/static_contextual_tp_sl_candidates.parquet"):
            dirs.add(path.parent.parent)
    return sorted(dirs)


def _arm_path(source_dir: Path, arm: str) -> Path:
    return source_dir / "portfolio_replay" / f"{arm}_contextual_tp_sl_candidates.parquet"


def _parquet_columns(path: Path) -> List[str]:
    if not path.exists():
        return []
    if pq is not None:
        try:
            return list(pq.read_schema(path).names)
        except Exception:
            pass
    try:
        return list(pd.read_parquet(path).columns)
    except Exception:
        return []


def _read_probe(path: Path) -> pd.DataFrame:
    cols = ["timestamp", "strategy_id"]
    try:
        return pd.read_parquet(path, columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)


def _source_record(
    source_dir: Path,
    cutoff: str,
    min_post_cutoff_rows: int,
    min_post_cutoff_timestamps: int,
    min_post_cutoff_active_heads: int,
    required_columns: List[str],
    required_diagnostic_groups: List[str],
    min_diagnostic_group_features: int,
) -> Dict[str, Any]:
    arm_status = {arm: _arm_path(source_dir, arm).exists() for arm in (*REQUIRED_ARMS, *OPTIONAL_ARMS)}
    arm_columns = {arm: set(_parquet_columns(_arm_path(source_dir, arm))) for arm in REQUIRED_ARMS}
    missing_columns_by_arm = {
        arm: sorted(set(required_columns) - arm_columns.get(arm, set()))
        for arm in REQUIRED_ARMS
        if required_columns
    }
    has_required_columns = all(not missing for missing in missing_columns_by_arm.values())
    missing_required_columns = ";".join(
        f"{arm}:{','.join(cols)}" for arm, cols in missing_columns_by_arm.items() if cols
    )
    if not missing_required_columns:
        missing_required_columns = "none"
    diagnostic_group_counts: Dict[str, Dict[str, int]] = {}
    missing_diagnostic_groups: List[str] = []
    for arm in REQUIRED_ARMS:
        cols = arm_columns.get(arm, set())
        diagnostic_group_counts[arm] = {}
        for group in required_diagnostic_groups:
            group_cols = DIAGNOSTIC_GROUPS.get(group)
            if group_cols is None:
                raise ValueError(f"Unknown diagnostic group {group!r}; valid groups: {sorted(DIAGNOSTIC_GROUPS)}")
            count = int(len(set(group_cols).intersection(cols)))
            diagnostic_group_counts[arm][group] = count
            if count < int(min_diagnostic_group_features):
                missing_diagnostic_groups.append(f"{arm}:{group}:{count}")
    has_required_diagnostic_groups = not missing_diagnostic_groups
    missing_required_diagnostic_groups = ";".join(missing_diagnostic_groups) if missing_diagnostic_groups else "none"
    diagnostic_group_coverage = ";".join(
        f"{arm}:{','.join(f'{group}={count}' for group, count in counts.items())}"
        for arm, counts in diagnostic_group_counts.items()
    ) or "none"
    static_path = _arm_path(source_dir, "static")
    frame = _read_probe(static_path) if static_path.exists() else pd.DataFrame()
    if frame.empty:
        missing_required_arms = [arm for arm in REQUIRED_ARMS if not arm_status[arm]]
        return {
            "source_dir": str(source_dir),
            "has_required_arms": all(arm_status[arm] for arm in REQUIRED_ARMS),
            "missing_required_arms": ",".join(missing_required_arms) if missing_required_arms else "none",
            "candidate_rows": 0,
            "post_cutoff_rows": 0,
            "post_cutoff_rows_needed": int(min_post_cutoff_rows),
            "post_cutoff_timestamps_needed": int(min_post_cutoff_timestamps),
            "post_cutoff_active_heads_needed": int(min_post_cutoff_active_heads),
            "has_required_columns": bool(has_required_columns),
            "missing_required_columns": missing_required_columns,
            "has_required_diagnostic_groups": bool(has_required_diagnostic_groups),
            "missing_required_diagnostic_groups": missing_required_diagnostic_groups,
            "diagnostic_group_coverage": diagnostic_group_coverage,
            "usable_post_cutoff": False,
            **{f"has_{arm}": bool(val) for arm, val in arm_status.items()},
        }
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    heads = _head_name(frame["strategy_id"]) if "strategy_id" in frame.columns else pd.Series(dtype=object)
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC") if cutoff else None
    post_mask = ts.ge(cutoff_ts) if cutoff_ts is not None else pd.Series(True, index=frame.index)
    post_heads = heads.loc[post_mask]
    unique_post_timestamps = int(ts.loc[post_mask].nunique())
    active_post_heads = int(post_heads.dropna().nunique())
    post_rows = int(post_mask.sum())
    missing_required_arms = [arm for arm in REQUIRED_ARMS if not arm_status[arm]]
    rows_needed = max(0, int(min_post_cutoff_rows) - post_rows)
    timestamps_needed = max(0, int(min_post_cutoff_timestamps) - unique_post_timestamps)
    heads_needed = max(0, int(min_post_cutoff_active_heads) - active_post_heads)
    return {
        "source_dir": str(source_dir),
        "has_required_arms": all(arm_status[arm] for arm in REQUIRED_ARMS),
        "missing_required_arms": ",".join(missing_required_arms) if missing_required_arms else "none",
        "has_required_columns": bool(has_required_columns),
        "missing_required_columns": missing_required_columns,
        "has_required_diagnostic_groups": bool(has_required_diagnostic_groups),
        "missing_required_diagnostic_groups": missing_required_diagnostic_groups,
        "diagnostic_group_coverage": diagnostic_group_coverage,
        **{f"has_{arm}": bool(val) for arm, val in arm_status.items()},
        "candidate_rows": int(len(frame)),
        "candidate_start": ts.min(),
        "candidate_end": ts.max(),
        "active_heads": int(heads.dropna().nunique()),
        "unique_timestamps": int(ts.nunique()),
        "cutoff": str(cutoff),
        "post_cutoff_rows": post_rows,
        "post_cutoff_start": ts.loc[post_mask].min() if post_rows else None,
        "post_cutoff_end": ts.loc[post_mask].max() if post_rows else None,
        "post_cutoff_timestamps": unique_post_timestamps,
        "post_cutoff_active_heads": active_post_heads,
        "post_cutoff_rows_needed": rows_needed,
        "post_cutoff_timestamps_needed": timestamps_needed,
        "post_cutoff_active_heads_needed": heads_needed,
        "usable_post_cutoff": bool(
            all(arm_status[arm] for arm in REQUIRED_ARMS)
            and has_required_columns
            and has_required_diagnostic_groups
            and post_rows >= int(min_post_cutoff_rows)
            and active_post_heads >= int(min_post_cutoff_active_heads)
            and unique_post_timestamps >= int(min_post_cutoff_timestamps)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-06-26T14:00:00Z")
    parser.add_argument("--min-post-cutoff-rows", type=int, default=1000)
    parser.add_argument("--min-post-cutoff-timestamps", type=int, default=20)
    parser.add_argument("--min-post-cutoff-active-heads", type=int, default=3)
    parser.add_argument(
        "--required-column",
        action="append",
        default=[],
        help="Column required in every required candidate arm. Repeatable.",
    )
    parser.add_argument(
        "--required-diagnostic-group",
        action="append",
        default=[],
        choices=sorted(DIAGNOSTIC_GROUPS),
        help=(
            "Diagnostic feature group that must have at least --min-diagnostic-group-features "
            "available columns in every required candidate arm. Repeatable."
        ),
    )
    parser.add_argument("--min-diagnostic-group-features", type=int, default=1)
    args = parser.parse_args()

    root_args = args.root or ["data_perp/reports", "data_perp/artifacts"]
    roots = [Path(p) for p in root_args]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _source_record(
            source_dir,
            str(args.cutoff),
            int(args.min_post_cutoff_rows),
            int(args.min_post_cutoff_timestamps),
            int(args.min_post_cutoff_active_heads),
            list(args.required_column or []),
            list(args.required_diagnostic_group or []),
            int(args.min_diagnostic_group_features),
        )
        for source_dir in _source_dirs(roots)
    ]
    frame = pd.DataFrame(records)
    if not frame.empty:
        if "has_required_diagnostic_groups" in frame.columns:
            frame["diagnostic_ready_sort"] = frame["has_required_diagnostic_groups"].astype(bool)
        else:
            frame["diagnostic_ready_sort"] = True
        frame = frame.sort_values(
            ["usable_post_cutoff", "diagnostic_ready_sort", "post_cutoff_rows", "candidate_end"],
            ascending=[False, False, False, False],
        )
        frame = frame.drop(columns=["diagnostic_ready_sort"], errors="ignore")
    frame.to_csv(args.out_dir / "contextual_tp_sl_candidate_source_scan.csv", index=False)
    payload: Dict[str, Any] = {
        "roots": [str(p) for p in roots],
        "cutoff": str(args.cutoff),
        "min_post_cutoff_rows": int(args.min_post_cutoff_rows),
        "min_post_cutoff_timestamps": int(args.min_post_cutoff_timestamps),
        "min_post_cutoff_active_heads": int(args.min_post_cutoff_active_heads),
        "required_columns": list(args.required_column or []),
        "required_diagnostic_groups": list(args.required_diagnostic_group or []),
        "min_diagnostic_group_features": int(args.min_diagnostic_group_features),
        "source_count": int(len(frame)),
        "usable_post_cutoff_count": int(frame["usable_post_cutoff"].sum()) if not frame.empty else 0,
        "sources": frame.to_dict(orient="records") if not frame.empty else [],
    }
    (args.out_dir / "contextual_tp_sl_candidate_source_scan.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Contextual TP/SL Candidate Source Scan",
        "",
        f"Cutoff: `{args.cutoff}`",
        f"Minimum post-cutoff rows: `{args.min_post_cutoff_rows}`",
        f"Minimum post-cutoff timestamps: `{args.min_post_cutoff_timestamps}`",
        f"Minimum post-cutoff active heads: `{args.min_post_cutoff_active_heads}`",
        f"Required columns: `{', '.join(args.required_column or []) or 'none'}`",
        f"Required diagnostic groups: `{', '.join(args.required_diagnostic_group or []) or 'none'}`",
        f"Minimum diagnostic columns per group/arm: `{args.min_diagnostic_group_features}`",
        "",
        frame.to_markdown(index=False) if not frame.empty else "_No sources found._",
    ]
    (args.out_dir / "contextual_tp_sl_candidate_source_scan.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "out_dir": str(args.out_dir),
                    "source_count": int(len(frame)),
                    "usable_post_cutoff_count": int(frame["usable_post_cutoff"].sum()) if not frame.empty else 0,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
