#!/usr/bin/env python3
"""State-overlay ablation for S52 selected ranker ledgers.

The overlay is learned on fit months only: for each live-predictable AE/GMM/state
feature, identify side-specific buckets whose selected top-k rows have dirty
path order, then apply that bucket blocklist to the holdout month. The default
mode filters selected rows without refilling; refill mode preserves the original
top-k budget per timestamp/side by taking the next-highest non-blocked rows.
This tests whether archetypes are useful as admission overlays.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s52_ranker_smoke import _bucket_state_feature, _state_feature_columns  # noqa: E402


DEFAULT_LEDGER = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "s52_ranker_smoke_best_archetype_overlay_v1/s52_ranker_smoke_scored_ledger.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_state_overlay_ablation_v1")


def _parse_csv(value: str | None, default: tuple[str, ...] = ()) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _safe_mean(values: Any) -> float:
    ser = pd.to_numeric(pd.Series(values), errors="coerce")
    ser = ser[np.isfinite(ser.to_numpy(dtype=np.float64))]
    return float(ser.mean()) if len(ser) else float("nan")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _metrics(frame: pd.DataFrame, *, round_trip_cost: float) -> dict[str, float]:
    if frame.empty:
        return {
            "selected_rows": 0.0,
            "mean_u": float("nan"),
            "ev_weighted_first_touch_precision": float("nan"),
            "first_pass_good_rate": float("nan"),
            "first_pass_bad_rate": float("nan"),
            "first_touch_bad_mae_1r_rate": float("nan"),
            "mfe_before_mae_1r_rate": float("nan"),
            "mae_before_mfe_1r_rate": float("nan"),
            "mean_max_adverse_before_mfe_1r": float("nan"),
            "mean_underwater_bars_before_mfe": float("nan"),
            "mean_underwater_fraction_before_mfe": float("nan"),
            "timeout_rate": float("nan"),
        }
    gross = pd.to_numeric(frame.get("first_touch_net"), errors="coerce").fillna(0.0) + float(round_trip_cost)
    good = pd.to_numeric(frame.get("first_pass_good"), errors="coerce").fillna(0.0).gt(0.5)
    denom = float(gross.abs().sum())
    evw = float(gross.where(good, 0.0).clip(lower=0.0).sum() / denom) if denom > 1e-12 else float("nan")
    return {
        "selected_rows": float(len(frame)),
        "mean_u": _safe_mean(frame.get("u_policy_net", [])),
        "ev_weighted_first_touch_precision": evw,
        "first_pass_good_rate": _safe_mean(good.astype(float)),
        "first_pass_bad_rate": _safe_mean(pd.to_numeric(frame.get("first_pass_bad"), errors="coerce").fillna(0.0)),
        "first_touch_bad_mae_1r_rate": _safe_mean(
            pd.to_numeric(frame.get("first_touch_mae_norm"), errors="coerce").ge(1.0).astype(float)
        ),
        "mfe_before_mae_1r_rate": _safe_mean(frame.get("mfe_1r_before_mae_1r", [])),
        "mae_before_mfe_1r_rate": _safe_mean(frame.get("mae_1r_before_mfe_1r", [])),
        "mean_max_adverse_before_mfe_1r": _safe_mean(frame.get("max_adverse_before_mfe_1r", [])),
        "mean_underwater_bars_before_mfe": _safe_mean(frame.get("underwater_bars_before_mfe_1r", [])),
        "mean_underwater_fraction_before_mfe": _safe_mean(frame.get("underwater_fraction_before_mfe_1r", [])),
        "timeout_rate": _safe_mean(pd.to_numeric(frame.get("is_timeout"), errors="coerce").fillna(0.0).gt(0.5)),
    }


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _block_buckets(
    selected_fit: pd.DataFrame,
    *,
    feature: str,
    min_fit_selected_rows: int,
    mae_before_threshold: float,
    adverse_threshold: float,
    underwater_bars_threshold: float,
    sides: set[str],
) -> pd.DataFrame:
    if selected_fit.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (side, bucket), group in selected_fit.groupby(["side_name", "__bucket__"], observed=True, dropna=False):
        if sides and str(side) not in sides:
            continue
        metrics = _metrics(group, round_trip_cost=0.0)
        dirty = (
            metrics["selected_rows"] >= float(min_fit_selected_rows)
            and (
                metrics["mae_before_mfe_1r_rate"] > float(mae_before_threshold)
                or metrics["mean_max_adverse_before_mfe_1r"] > float(adverse_threshold)
                or metrics["mean_underwater_bars_before_mfe"] > float(underwater_bars_threshold)
            )
        )
        if dirty:
            rows.append(
                {
                    "state_feature": str(feature),
                    "side": str(side),
                    "bucket": str(bucket),
                    **{f"fit_{k}": v for k, v in metrics.items()},
                }
            )
    return pd.DataFrame(rows)


def _apply_block_mask(local: pd.DataFrame, rules: pd.DataFrame) -> np.ndarray:
    if rules.empty:
        return np.zeros(len(local), dtype=bool)
    blocked = set(zip(rules["side"].astype(str), rules["bucket"].astype(str)))
    return np.asarray(
        [
            (str(side), str(bucket)) in blocked
            for side, bucket in zip(local["side_name"], local["__bucket__"])
        ],
        dtype=bool,
    )


def _select_refill_budget(
    local: pd.DataFrame,
    *,
    blocked_mask: np.ndarray,
    selected_col: str,
    score_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    """Select replacement rows after a blocklist while preserving top-k budgets."""
    if local.empty:
        return local.copy()
    missing = [col for col in group_cols if col not in local.columns]
    if missing:
        raise KeyError(f"Missing refill group column(s): {missing}")
    if score_col not in local.columns:
        raise KeyError(f"Missing score column {score_col!r}")
    work = local.copy()
    work["__blocked__"] = blocked_mask
    selected_parts: list[pd.DataFrame] = []
    for _, group in work.groupby(group_cols, observed=True, dropna=False, sort=False):
        baseline_n = int(pd.to_numeric(group[selected_col], errors="coerce").fillna(0.0).gt(0.5).sum())
        if baseline_n <= 0:
            continue
        eligible = group.loc[~group["__blocked__"]].copy()
        if eligible.empty:
            continue
        ordered = eligible.sort_values(score_col, ascending=False, kind="mergesort")
        selected_parts.append(ordered.head(baseline_n))
    if not selected_parts:
        return work.iloc[0:0].drop(columns=["__blocked__"], errors="ignore")
    return pd.concat(selected_parts, ignore_index=False).drop(columns=["__blocked__"], errors="ignore")


def _score_overlay(row: dict[str, Any]) -> float:
    rows = float(row.get("holdout_filtered_selected_rows", 0.0) or 0.0)
    baseline_rows = float(row.get("holdout_baseline_selected_rows", 0.0) or 0.0)
    retention = rows / max(baseline_rows, 1.0)
    return float(
        1.00 * (row.get("holdout_filtered_ev_weighted_first_touch_precision", 0.0) or 0.0)
        + 0.30 * max(row.get("holdout_filtered_mean_u", -0.02) or -0.02, -0.02)
        - 0.35 * max((row.get("holdout_filtered_mae_before_mfe_1r_rate", 1.0) or 1.0) - 0.35, 0.0)
        - 0.20 * max((row.get("holdout_filtered_mean_max_adverse_before_mfe_1r", 3.0) or 3.0) - 1.50, 0.0)
        - 0.02 * max((row.get("holdout_filtered_mean_underwater_bars_before_mfe", 20.0) or 20.0) - 10.0, 0.0)
        - 0.25 * max(0.65 - retention, 0.0)
    )


def build_report(
    *,
    ledger_path: Path,
    output_dir: Path,
    variant: str,
    selected_col: str,
    fit_months: list[str],
    holdout_month: str,
    round_trip_cost: float,
    min_fit_selected_rows: int,
    mae_before_threshold: float,
    adverse_threshold: float,
    underwater_bars_threshold: float,
    sides: list[str],
    overlay_mode: str,
    refill_group_cols: list[str],
    score_col: str,
) -> dict[str, Any]:
    ledger = pd.read_parquet(ledger_path)
    if variant:
        ledger = ledger[ledger["variant"].astype(str).eq(str(variant))].copy()
    if selected_col not in ledger.columns:
        raise KeyError(f"Missing selected column {selected_col!r} in {ledger_path}")
    if score_col not in ledger.columns:
        raise KeyError(f"Missing score column {score_col!r} in {ledger_path}")
    mode = str(overlay_mode or "filter").strip().lower()
    if mode not in {"filter", "refill"}:
        raise ValueError(f"Unsupported overlay mode {overlay_mode!r}; expected filter or refill")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = ledger[ledger[selected_col].astype(bool)].copy()
    selected["month"] = selected["month"].astype(str)
    side_set = {str(side).strip() for side in sides if str(side).strip()}

    baseline_fit = selected[selected["month"].isin(fit_months)].copy()
    baseline_holdout = selected[selected["month"].eq(str(holdout_month))].copy()
    baseline_all = selected.copy()
    state_cols = _state_feature_columns(selected.columns)
    rows: list[dict[str, Any]] = []
    rules_parts: list[pd.DataFrame] = []
    for feature in state_cols:
        bucketed = _bucket_state_feature(ledger[feature])
        local = ledger.assign(__bucket__=bucketed)
        local_selected = local[local[selected_col].astype(bool)].copy()
        fit_selected = local_selected[local_selected["month"].astype(str).isin(fit_months)].copy()
        rules = _block_buckets(
            fit_selected,
            feature=feature,
            min_fit_selected_rows=int(min_fit_selected_rows),
            mae_before_threshold=float(mae_before_threshold),
            adverse_threshold=float(adverse_threshold),
            underwater_bars_threshold=float(underwater_bars_threshold),
            sides=side_set,
        )
        if rules.empty:
            continue
        rules_parts.append(rules)
        selected_local = local_selected.copy()
        selected_blocked_mask = _apply_block_mask(selected_local, rules)
        if mode == "refill":
            full_blocked_mask = _apply_block_mask(local, rules)
            filtered = _select_refill_budget(
                local,
                blocked_mask=full_blocked_mask,
                selected_col=selected_col,
                score_col=score_col,
                group_cols=refill_group_cols,
            )
        else:
            filtered = selected_local.loc[~selected_blocked_mask].copy()
        filtered_fit = filtered[filtered["month"].astype(str).isin(fit_months)].copy()
        filtered_holdout = filtered[filtered["month"].astype(str).eq(str(holdout_month))].copy()
        row: dict[str, Any] = {
            "state_feature": str(feature),
            "blocked_bucket_count": int(len(rules)),
            "overlay_mode": mode,
        }
        row.update(_prefix("fit_baseline", _metrics(baseline_fit, round_trip_cost=float(round_trip_cost))))
        row.update(_prefix("fit_filtered", _metrics(filtered_fit, round_trip_cost=float(round_trip_cost))))
        row.update(_prefix("holdout_baseline", _metrics(baseline_holdout, round_trip_cost=float(round_trip_cost))))
        row.update(_prefix("holdout_filtered", _metrics(filtered_holdout, round_trip_cost=float(round_trip_cost))))
        row.update(_prefix("all_baseline", _metrics(baseline_all, round_trip_cost=float(round_trip_cost))))
        row.update(_prefix("all_filtered", _metrics(filtered, round_trip_cost=float(round_trip_cost))))
        row["holdout_retention"] = row["holdout_filtered_selected_rows"] / max(row["holdout_baseline_selected_rows"], 1.0)
        row["all_retention"] = row["all_filtered_selected_rows"] / max(row["all_baseline_selected_rows"], 1.0)
        row["overlay_score"] = _score_overlay(row)
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            [
                "overlay_score",
                "holdout_filtered_ev_weighted_first_touch_precision",
                "holdout_filtered_mae_before_mfe_1r_rate",
            ],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    rules_df = pd.concat(rules_parts, ignore_index=True) if rules_parts else pd.DataFrame()
    paths = {
        "summary": output_dir / "s52_state_overlay_ablation_summary.csv",
        "rules": output_dir / "s52_state_overlay_ablation_rules.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "s52_state_overlay_ablation.md",
    }
    summary.to_csv(paths["summary"], index=False)
    rules_df.to_csv(paths["rules"], index=False)
    manifest = {
        "scope": "s52_state_overlay_ablation",
        "ledger_path": str(ledger_path),
        "output_dir": str(output_dir),
        "variant": str(variant),
        "selected_col": str(selected_col),
        "fit_months": fit_months,
        "holdout_month": str(holdout_month),
        "round_trip_cost": float(round_trip_cost),
        "min_fit_selected_rows": int(min_fit_selected_rows),
        "mae_before_threshold": float(mae_before_threshold),
        "adverse_threshold": float(adverse_threshold),
        "underwater_bars_threshold": float(underwater_bars_threshold),
        "sides": sorted(side_set),
        "overlay_mode": mode,
        "refill_group_cols": refill_group_cols,
        "score_col": str(score_col),
        "state_features": state_cols,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    cols = [
        "state_feature",
        "overlay_mode",
        "blocked_bucket_count",
        "overlay_score",
        "holdout_retention",
        "holdout_filtered_ev_weighted_first_touch_precision",
        "holdout_filtered_mean_u",
        "holdout_filtered_mae_before_mfe_1r_rate",
        "holdout_filtered_mean_max_adverse_before_mfe_1r",
        "holdout_filtered_mean_underwater_bars_before_mfe",
        "holdout_baseline_ev_weighted_first_touch_precision",
        "holdout_baseline_mean_u",
        "holdout_baseline_mae_before_mfe_1r_rate",
        "holdout_baseline_mean_max_adverse_before_mfe_1r",
        "holdout_baseline_mean_underwater_bars_before_mfe",
    ]
    lines = [
        "# S52 State Overlay Ablation",
        "",
        f"Ledger: `{ledger_path}`",
        f"Variant: `{variant}`",
        f"Selected column: `{selected_col}`",
        f"Fit months: `{', '.join(fit_months)}`",
        f"Holdout month: `{holdout_month}`",
        "",
        "## Top Single-Feature Overlays",
        "",
        summary[[c for c in cols if c in summary.columns]].head(25).to_markdown(index=False)
        if not summary.empty
        else "No overlay rules found.",
        "",
        "## Outputs",
        "",
        f"- Summary: `{paths['summary']}`",
        f"- Rules: `{paths['rules']}`",
        f"- Manifest: `{paths['manifest']}`",
    ]
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default="ranker_side_specific_timestamp")
    parser.add_argument("--selected-col", default="selected_top10")
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--round-trip-cost", type=float, default=0.0100)
    parser.add_argument("--min-fit-selected-rows", type=int, default=40)
    parser.add_argument("--mae-before-threshold", type=float, default=0.40)
    parser.add_argument("--adverse-threshold", type=float, default=1.50)
    parser.add_argument("--underwater-bars-threshold", type=float, default=10.0)
    parser.add_argument("--sides", default="long")
    parser.add_argument("--overlay-mode", choices=("filter", "refill"), default="filter")
    parser.add_argument("--refill-group-cols", default="month,side_name")
    parser.add_argument("--score-col", default="score")
    args = parser.parse_args()
    manifest = build_report(
        ledger_path=args.ledger_path,
        output_dir=args.output_dir,
        variant=str(args.variant),
        selected_col=str(args.selected_col),
        fit_months=_parse_csv(args.fit_months, ()),
        holdout_month=str(args.holdout_month),
        round_trip_cost=float(args.round_trip_cost),
        min_fit_selected_rows=int(args.min_fit_selected_rows),
        mae_before_threshold=float(args.mae_before_threshold),
        adverse_threshold=float(args.adverse_threshold),
        underwater_bars_threshold=float(args.underwater_bars_threshold),
        sides=_parse_csv(args.sides, ()),
        overlay_mode=str(args.overlay_mode),
        refill_group_cols=_parse_csv(args.refill_group_cols, ("month", "side_name")),
        score_col=str(args.score_col),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
