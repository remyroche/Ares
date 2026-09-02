#!/usr/bin/env python3
"""Ablate causal fixed net-EV admission targets by side x archetype.

The hierarchical EV map supplies a common net-return unit.  At each decision
timestamp, the existing threshold-basis policy estimates a shrunk 28-day
realized-minus-mapped EV correction for every side x archetype.  This ablation
admits rows when that corrected expected EV reaches a fixed target; unlike the
deployed comparator, it does not force a top-k activity quota.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.evaluate_side_archetype_expected_ev_policy import _load_rows


DEFAULT_TARGETS = (0.006, 0.007, 0.008, 0.009)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _policy_payload(
    *, reference_path: Path, target: float, reference_rows: int
) -> dict[str, Any]:
    target_bps = int(round(float(target) * 10_000.0))
    return {
        "schema_version": "threshold_basis_policy_v3",
        "enabled": True,
        "status": "ablation_only_not_promoted",
        "policy_id": f"side_archetype_fixed_ev_{target_bps}bps_recent28d_v1",
        "policy_name": f"s52_v9_tail95_hierev_sidearch_fixedev{target_bps}bps_28d_v1",
        "family": "side_archetype_expected_ev_recent_correction",
        "selection_mode": "fixed_corrected_ev_threshold",
        "fixed_target_net_ev": float(target),
        "window_days": 28,
        "recalibration_frequency": "1d_at_00_utc",
        "outcome_horizon_hours": 12,
        "min_reference_rows": 40,
        "side_support_target": 320.0,
        "local_support_target": 160.0,
        "recent_ev_correction_cap": 0.03,
        # Used only to map admitted rows into the regular [0.90, 1.00]
        # portfolio band; it is not an activity quota in this selection mode.
        "top_fraction": 0.10,
        "ev_rank_blend_weight": 1.0,
        "corrected_ev_tie_break_parent": False,
        "mapped_expected_ev_col": "expected_net_ev_after_1pct_side_archetype",
        "reference_mapped_expected_ev_col": "mapped_expected_ev",
        "reference_parent_rank_col": "rank_mlp_direct",
        "rank_blend_parent_col": "v9_tail95_predecessor_rank",
        "return_col": "ev_after_1pct",
        "reference_candidates_path": str(reference_path.resolve()),
        "reference_rows": int(reference_rows),
        "cost_contract": (
            "fixed_target_net_ev, mapped_expected_ev, and ev_after_1pct are net "
            "of the sole 1% round-trip cost; no additional fee is subtracted"
        ),
        "causal_contract": (
            "At each UTC day boundary, the 28-day correction uses only rows "
            "with outcome_resolved_at at or before that boundary. Local "
            "corrections shrink to side and global estimates when support is "
            "limited."
        ),
    }


def _period_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy(deep=False)
    ts = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["day"] = ts.dt.floor("D")
    out["month"] = ts.dt.strftime("%Y-%m")
    out["week_start"] = ts.dt.floor("D") - pd.to_timedelta(
        ts.dt.weekday, unit="D"
    )
    return out


def _group_metrics(
    rows: pd.DataFrame,
    groups: list[str],
    *,
    source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    group_arg: str | list[str] = groups[0] if len(groups) == 1 else groups
    report = (
        rows.groupby(group_arg, dropna=False, observed=True)
        .agg(
            selected_rows=("ev_after_1pct", "size"),
            days=("day", "nunique"),
            mean_net_ev=("ev_after_1pct", "mean"),
            sum_net_ev=("ev_after_1pct", "sum"),
            positive_ev_rate=("ev_after_1pct", lambda x: float((x > 0).mean())),
            clean_exec_rate=("clean_exec", "mean"),
            dirty_positive_rate=("dirty_positive", "mean"),
            bad_mae_rate=("full_path_bad_mae_1r", "mean"),
            timeout_rate=("timeout", "mean"),
            corrected_expected_ev_mean=("corrected_expected_ev", "mean"),
        )
        .reset_index()
        .assign(
            trades_per_day=lambda x: x["selected_rows"] / x["days"].clip(lower=1)
        )
    )
    report = report.rename(columns={"days": "active_days"})
    if source is not None and not source.empty:
        calendar = (
            source.groupby(group_arg, dropna=False, observed=True)["day"]
            .nunique()
            .rename("calendar_days")
            .reset_index()
        )
        report = report.merge(calendar, on=groups, how="left", validate="many_to_one")
        report["trades_per_calendar_day"] = report["selected_rows"] / report[
            "calendar_days"
        ].clip(lower=1)
    return report


def _summary(
    selected: pd.DataFrame, source: pd.DataFrame, arm: str
) -> dict[str, Any]:
    selected = _period_columns(selected.copy())
    source_ts = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    source_days = int(source_ts.dt.floor("D").nunique())
    source_bars = int(source_ts.nunique())
    week = selected.groupby("week_start", observed=True)["ev_after_1pct"].mean()
    month = selected.groupby("month", observed=True)["ev_after_1pct"].mean()
    ev = pd.to_numeric(selected["ev_after_1pct"], errors="coerce")
    selected_days = int(selected["day"].nunique())
    selected_bars = int(pd.to_datetime(selected["__ts__"], utc=True).nunique())
    return {
        "arm": arm,
        "selected_rows": int(len(selected)),
        "trades_per_day": float(len(selected) / max(source_days, 1)),
        "mean_net_ev": float(ev.mean()) if len(ev) else np.nan,
        "sum_net_ev": float(ev.sum()) if len(ev) else 0.0,
        "positive_ev_rate": float((ev > 0).mean()) if len(ev) else np.nan,
        "clean_exec_rate": float(selected["clean_exec"].mean()) if len(ev) else np.nan,
        "bad_mae_rate": float(selected["full_path_bad_mae_1r"].mean()) if len(ev) else np.nan,
        "timeout_rate": float(selected["timeout"].mean()) if len(ev) else np.nan,
        "worst_week": float(week.min()) if len(week) else np.nan,
        "q10_week": float(week.quantile(0.10)) if len(week) else np.nan,
        "worst_month": float(month.min()) if len(month) else np.nan,
        "positive_week_rate": float((week > 0).mean()) if len(week) else np.nan,
        "days_with_trades": selected_days,
        "no_trade_days": int(max(source_days - selected_days, 0)),
        "bars_with_trades": selected_bars,
        "no_trade_bars": int(max(source_bars - selected_bars, 0)),
    }


def _score_fixed_targets_daily(
    rows: pd.DataFrame,
    targets: list[float],
    *,
    window_days: int = 28,
    outcome_horizon_hours: int = 12,
    min_rows: int = 40,
    side_support_target: float = 320.0,
    local_support_target: float = 160.0,
    correction_cap: float = 0.03,
) -> dict[float, pd.DataFrame]:
    """Apply one causal daily correction pass shared by all target arms."""
    source = rows.sort_values("__ts__", kind="stable").reset_index(drop=True)
    ts = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    ts_ns = ts.astype("int64").to_numpy(dtype=np.int64, copy=False)
    outcome_ns = ts_ns + int(pd.Timedelta(hours=outcome_horizon_hours).value)
    mapped_all = pd.to_numeric(
        source["expected_net_ev_after_1pct_mlp_direct"], errors="coerce"
    ).to_numpy(dtype=np.float64, copy=False)
    realized_all = pd.to_numeric(
        source["ev_after_1pct"], errors="coerce"
    ).to_numpy(dtype=np.float64, copy=False)
    side_all = source["side_name"].astype(str).to_numpy(copy=False)
    arch_all = source["policy_archetype"].astype(str).to_numpy(copy=False)
    day_all = ts.dt.floor("D")
    result: dict[float, list[pd.DataFrame]] = {target: [] for target in targets}

    for day_value in sorted(day_all.dropna().unique()):
        day = pd.Timestamp(day_value)
        day_ns = int(day.value)
        start_ns = int((day - pd.Timedelta(days=window_days)).value)
        ref_start = int(np.searchsorted(ts_ns, start_ns, side="left"))
        ref_end = int(np.searchsorted(outcome_ns, day_ns, side="right"))
        current_idx = np.flatnonzero(day_all.eq(day).to_numpy())
        if ref_end <= ref_start or current_idx.size == 0:
            continue
        ref_idx = np.arange(ref_start, ref_end, dtype=np.int64)
        ref_residual = realized_all[ref_idx] - mapped_all[ref_idx]
        finite = np.isfinite(ref_residual) & np.isfinite(mapped_all[ref_idx])
        if int(finite.sum()) < int(min_rows):
            continue
        ref_idx = ref_idx[finite]
        ref_residual = ref_residual[finite]
        global_support = int(ref_residual.size)
        global_correction = float(
            np.clip(np.mean(ref_residual), -correction_cap, correction_cap)
        )
        ref_stats = pd.DataFrame(
            {
                "side_name": side_all[ref_idx],
                "policy_archetype": arch_all[ref_idx],
                "residual": ref_residual,
            }
        )
        side_stats = ref_stats.groupby(
            "side_name", sort=False, observed=True
        )["residual"].agg(["mean", "count"])
        side_corrections: dict[str, tuple[float, int]] = {}
        for side, stat in side_stats.iterrows():
            support = int(stat["count"])
            alpha = float(
                np.clip(support / max(side_support_target, 1.0), 0.0, 1.0)
            )
            correction = (1.0 - alpha) * global_correction + alpha * float(
                stat["mean"]
            )
            side_corrections[str(side)] = (
                float(np.clip(correction, -correction_cap, correction_cap)),
                support,
            )
        local_stats = ref_stats.groupby(
            ["side_name", "policy_archetype"], sort=False, observed=True
        )["residual"].agg(["mean", "count"])
        local_corrections: dict[tuple[str, str], tuple[float, int]] = {}
        for (side, archetype), stat in local_stats.iterrows():
            side_key, arch_key = str(side), str(archetype)
            parent = side_corrections.get(
                side_key, (global_correction, global_support)
            )[0]
            support = int(stat["count"])
            alpha = float(
                np.clip(support / max(local_support_target, 1.0), 0.0, 1.0)
            )
            correction = (1.0 - alpha) * parent + alpha * float(stat["mean"])
            local_corrections[(side_key, arch_key)] = (
                float(np.clip(correction, -correction_cap, correction_cap)),
                support,
            )

        corrections = np.empty(current_idx.size, dtype=np.float64)
        supports = np.empty(current_idx.size, dtype=np.int32)
        scopes = np.empty(current_idx.size, dtype=object)
        for pos, row_idx in enumerate(current_idx):
            side = str(side_all[row_idx])
            archetype = str(arch_all[row_idx])
            local = local_corrections.get((side, archetype))
            if local is not None:
                corrections[pos], supports[pos] = local
                scopes[pos] = (
                    "side_x_archetype"
                    if local[1] >= min_rows
                    else "side_x_archetype_shrunk_low_support"
                )
                continue
            parent = side_corrections.get(side)
            if parent is not None:
                corrections[pos], supports[pos] = parent
                scopes[pos] = "side_fallback"
            else:
                corrections[pos], supports[pos] = global_correction, global_support
                scopes[pos] = "global_fallback"
        corrected = mapped_all[current_idx] + corrections
        for target in targets:
            selected_mask = np.isfinite(corrected) & (corrected >= target)
            if not selected_mask.any():
                continue
            selected = source.iloc[current_idx[selected_mask]].copy()
            selected["selected"] = True
            selected["arm"] = (
                f"fixed_ev_{int(round(target * 10_000))}bps_recent28d"
            )
            selected["mapped_expected_ev"] = mapped_all[current_idx[selected_mask]]
            selected["recent_ev_correction"] = corrections[selected_mask]
            selected["corrected_expected_ev"] = corrected[selected_mask]
            selected["correction_scope"] = scopes[selected_mask]
            selected["local_support"] = supports[selected_mask]
            selected["fixed_target_net_ev"] = float(target)
            selected["calibration_asof"] = day
            result[target].append(selected)

    empty = source.iloc[:0].copy()
    return {
        target: (
            pd.concat(parts, ignore_index=True, copy=False)
            if parts
            else empty.copy()
        )
        for target, parts in result.items()
    }


def _read_comparator(path: Path | None, filename: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    source = path / filename
    return pd.read_csv(source) if source.exists() else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oos-predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparator-dir", type=Path)
    parser.add_argument("--start", default="2026-04-01T00:00:00Z")
    parser.add_argument(
        "--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS)
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = args.output_dir / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    rows = _load_rows(args.oos_predictions, start)
    reference_rows = int(len(pd.read_parquet(args.reference, columns=["timestamp"])))

    summaries: list[dict[str, Any]] = []
    report_parts: dict[str, list[pd.DataFrame]] = {
        "month": [],
        "week": [],
        "side": [],
        "archetype": [],
        "month_side_archetype": [],
    }
    selected_parts: list[pd.DataFrame] = []
    policies: list[str] = []
    targets = sorted(set(float(value) for value in args.targets))
    scored_targets = _score_fixed_targets_daily(rows, targets)
    for target in targets:
        target_bps = int(round(target * 10_000.0))
        arm = f"fixed_ev_{target_bps}bps_recent28d"
        policy_path = policy_dir / f"{arm}.json"
        _write_json(
            policy_path,
            _policy_payload(
                reference_path=args.reference,
                target=target,
                reference_rows=reference_rows,
            ),
        )
        policies.append(str(policy_path))
        selected = _period_columns(scored_targets[target])
        source_for_arm = _period_columns(rows.copy(deep=False))
        source_for_arm["arm"] = arm
        selected_parts.append(selected)
        summaries.append(_summary(selected, rows, arm))
        for name, groups in {
            "month": ["arm", "month"],
            "week": ["arm", "week_start"],
            "side": ["arm", "side_name"],
            "archetype": ["arm", "side_name", "policy_archetype"],
            "month_side_archetype": [
                "arm", "month", "side_name", "policy_archetype"
            ],
        }.items():
            report_parts[name].append(
                _group_metrics(selected, groups, source=source_for_arm)
            )
        print(f"completed {arm}: selected={len(selected):,}", flush=True)

    comparator_summary = _read_comparator(args.comparator_dir, "summary.csv")
    summary = pd.concat(
        [comparator_summary, pd.DataFrame(summaries)], ignore_index=True, copy=False
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    comparator_names = {
        "month": "month.csv",
        "week": "week.csv",
        "side": "side.csv",
        "archetype": "archetype.csv",
        "month_side_archetype": "month_side_archetype.csv",
    }
    for name, parts in report_parts.items():
        comparator = _read_comparator(args.comparator_dir, comparator_names[name])
        pd.concat([comparator, *parts], ignore_index=True, copy=False).to_csv(
            args.output_dir / f"{name}.csv", index=False
        )
    if selected_parts:
        pd.concat(selected_parts, ignore_index=True, copy=False).to_parquet(
            args.output_dir / "selected_row_diagnostics.parquet",
            index=False,
            compression="zstd",
        )
    manifest = {
        "schema": "side_archetype_fixed_ev_target_ablation_v1",
        "oos_predictions": str(args.oos_predictions),
        "reference": str(args.reference),
        "reference_rows": reference_rows,
        "start": start.isoformat(),
        "targets": targets,
        "recalibration": (
            "causal daily-at-00:00-UTC rolling 28d side x archetype residual correction"
        ),
        "selection": "corrected expected EV >= fixed target; no top-k quota",
        "cost_contract": (
            "ev_after_1pct and every fixed target are net of the sole 1% "
            "round-trip cost; no second cost subtraction"
        ),
        "evidence": (
            "model OOF predictions; causal policy reference with 12h outcome "
            "resolution; candidate admission before portfolio optimization"
        ),
        "policies": policies,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
