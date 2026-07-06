#!/usr/bin/env python3
"""Fit-selected S52 AE/GMM state-archetype overlay report.

This report is deliberately selection-only. It reads a completed S52 ranker
ledger, forms live-predictable AE/GMM/state buckets, chooses promising
bucket-side overlays using fit months only, and then reports holdout metrics.

The goal is to test whether state/archetype descriptors can identify cleaner
slices of an already selected top-k candidate stream without leaking June
outcomes into the overlay choice.
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

from scripts.run_s52_ranker_smoke import _state_feature_columns  # noqa: E402
from scripts.run_s52_state_risk_penalty_overlay import (  # noqa: E402
    _apply_bucket_spec,
    _fit_bucket_spec,
)


DEFAULT_LEDGER = Path(
    "data_perp/reports/s52_soft_exec_ordered_ev_ranker_smoke_tp075_sl075_cost100bps_20260705_v1/"
    "s52_ranker_smoke_scored_ledger.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_state_archetype_overlay_v1")


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
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _parse_csv(value: str | None) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _num(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(float(default), index=frame.index, dtype=np.float64)


def _safe_mean(values: Any) -> float:
    ser = pd.to_numeric(pd.Series(values), errors="coerce")
    ser = ser[np.isfinite(ser.to_numpy(dtype=np.float64))]
    return float(ser.mean()) if len(ser) else float("nan")


def _metrics(frame: pd.DataFrame, *, round_trip_cost: float) -> dict[str, float]:
    if frame.empty:
        return {
            "selected_rows": 0.0,
            "ev_weighted_first_touch_precision": float("nan"),
            "first_pass_good_rate": float("nan"),
            "first_pass_bad_rate": float("nan"),
            "mean_first_touch_net": float("nan"),
            "mean_first_touch_gross": float("nan"),
            "mean_u": float("nan"),
            "first_touch_bad_mae_1r_rate": float("nan"),
            "mfe_before_mae_1r_rate": float("nan"),
            "mae_before_mfe_1r_rate": float("nan"),
            "mean_max_adverse_before_mfe_1r": float("nan"),
            "mean_underwater_bars_before_mfe_1r": float("nan"),
            "mean_underwater_fraction_before_mfe_1r": float("nan"),
            "timeout_rate": float("nan"),
            "mean_score": float("nan"),
        }
    net = _num(frame, "first_touch_net", 0.0).fillna(0.0)
    if "first_touch_gross" in frame.columns:
        gross = _num(frame, "first_touch_gross", 0.0).fillna(0.0)
    else:
        gross = net + float(round_trip_cost)
    good = _num(frame, "first_pass_good", 0.0).fillna(0.0).gt(0.5)
    denom = float(gross.abs().sum())
    evw = float(gross.where(good, 0.0).clip(lower=0.0).sum() / denom) if denom > 1e-12 else float("nan")
    return {
        "selected_rows": float(len(frame)),
        "ev_weighted_first_touch_precision": evw,
        "first_pass_good_rate": _safe_mean(good.astype(float)),
        "first_pass_bad_rate": _safe_mean(_num(frame, "first_pass_bad", 0.0).fillna(0.0).gt(0.5).astype(float)),
        "mean_first_touch_net": _safe_mean(net),
        "mean_first_touch_gross": _safe_mean(gross),
        "mean_u": _safe_mean(frame.get("u_policy_net", [])),
        "first_touch_bad_mae_1r_rate": _safe_mean(_num(frame, "first_touch_mae_norm", np.nan).ge(1.0).astype(float)),
        "mfe_before_mae_1r_rate": _safe_mean(frame.get("mfe_1r_before_mae_1r", [])),
        "mae_before_mfe_1r_rate": _safe_mean(frame.get("mae_1r_before_mfe_1r", [])),
        "mean_max_adverse_before_mfe_1r": _safe_mean(frame.get("max_adverse_before_mfe_1r", [])),
        "mean_underwater_bars_before_mfe_1r": _safe_mean(frame.get("underwater_bars_before_mfe_1r", [])),
        "mean_underwater_fraction_before_mfe_1r": _safe_mean(
            frame.get("underwater_fraction_before_mfe_1r", [])
        ),
        "timeout_rate": _safe_mean(_num(frame, "is_timeout", 0.0).fillna(0.0).gt(0.5).astype(float)),
        "mean_score": _safe_mean(frame.get("score", [])),
    }


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _fit_objective(rows: pd.DataFrame) -> pd.Series:
    evw = _num(rows, "fit_ev_weighted_first_touch_precision", 0.0).fillna(0.0)
    net = _num(rows, "fit_mean_first_touch_net", -0.02).fillna(-0.02)
    base_net = _num(rows, "fit_baseline_mean_first_touch_net", -0.02).fillna(-0.02)
    mae_before = _num(rows, "fit_mae_before_mfe_1r_rate", 1.0).fillna(1.0)
    bad_mae = _num(rows, "fit_first_touch_bad_mae_1r_rate", 1.0).fillna(1.0)
    adverse = _num(rows, "fit_mean_max_adverse_before_mfe_1r", 99.0).fillna(99.0)
    underwater = _num(rows, "fit_mean_underwater_bars_before_mfe_1r", 99.0).fillna(99.0)
    timeout = _num(rows, "fit_timeout_rate", 1.0).fillna(1.0)
    score = (
        evw
        + 10.0 * net.clip(lower=-0.02, upper=0.02)
        + 4.0 * (net - base_net).clip(lower=-0.02, upper=0.02)
        - 0.45 * (bad_mae - 0.25).clip(lower=0.0)
        - 0.35 * (mae_before - 0.35).clip(lower=0.0)
        - 0.20 * (adverse - 1.50).clip(lower=0.0)
        - 0.06 * (underwater - 10.0).clip(lower=0.0)
        - 0.15 * (timeout - 0.12).clip(lower=0.0)
    )
    return score.astype(np.float64)


def _eligible_rows(
    rows: pd.DataFrame,
    *,
    min_fit_selected_rows: int,
    min_fit_evw: float,
    min_fit_net: float,
    max_fit_first_touch_bad_mae: float,
    max_fit_mae_before: float,
    max_fit_adverse: float,
    max_fit_underwater: float,
    max_fit_timeout: float,
) -> pd.Series:
    mask = _num(rows, "fit_selected_rows", 0.0).ge(float(min_fit_selected_rows))
    mask &= _num(rows, "fit_ev_weighted_first_touch_precision", 0.0).ge(float(min_fit_evw))
    mask &= _num(rows, "fit_mean_first_touch_net", -np.inf).ge(float(min_fit_net))
    mask &= _num(rows, "fit_first_touch_bad_mae_1r_rate", 1.0).le(float(max_fit_first_touch_bad_mae))
    mask &= _num(rows, "fit_mae_before_mfe_1r_rate", 1.0).le(float(max_fit_mae_before))
    mask &= _num(rows, "fit_mean_max_adverse_before_mfe_1r", np.inf).le(float(max_fit_adverse))
    mask &= _num(rows, "fit_mean_underwater_bars_before_mfe_1r", np.inf).le(float(max_fit_underwater))
    mask &= _num(rows, "fit_timeout_rate", 1.0).le(float(max_fit_timeout))
    return mask.fillna(False)


def _gate_status(row: pd.Series) -> str:
    failures: list[str] = []
    selected_col = str(row.get("selected_col", ""))
    ev_floor = {"selected_top10": 0.70, "selected_top20": 0.65, "selected_top30": 0.60}.get(
        selected_col,
        0.60,
    )
    holdout_rows = float(row.get("holdout_selected_rows", 0.0) or 0.0)
    if not math.isfinite(holdout_rows) or holdout_rows <= 0.0:
        failures.append("no_holdout_rows")
    if float(row.get("holdout_ev_weighted_first_touch_precision", -np.inf)) < ev_floor:
        failures.append("holdout_evw")
    if float(row.get("holdout_mean_first_touch_net", -np.inf)) < 0.0:
        failures.append("holdout_net")
    if float(row.get("holdout_first_touch_bad_mae_1r_rate", np.inf)) > 0.25:
        failures.append("holdout_first_touch_bad_mae")
    if float(row.get("holdout_mae_before_mfe_1r_rate", np.inf)) > 0.35:
        failures.append("holdout_mae_before_mfe")
    if float(row.get("holdout_mean_max_adverse_before_mfe_1r", np.inf)) > 1.50:
        failures.append("holdout_max_adverse")
    if float(row.get("holdout_mean_underwater_bars_before_mfe_1r", np.inf)) > 10.0:
        failures.append("holdout_underwater")
    if float(row.get("holdout_timeout_rate", np.inf)) > 0.12:
        failures.append("holdout_timeout")
    return "pass" if not failures else "fail:" + ",".join(failures)


def build_archetype_candidates(
    ledger: pd.DataFrame,
    *,
    selected_cols: list[str],
    fit_months: list[str],
    holdout_month: str,
    round_trip_cost: float,
    q: int = 5,
    max_features: int = 0,
) -> pd.DataFrame:
    state_cols = [col for col in _state_feature_columns(ledger.columns) if col in ledger.columns]
    if int(max_features) > 0:
        state_cols = state_cols[: int(max_features)]
    if not state_cols:
        return pd.DataFrame()
    fit_mask = ledger["month"].astype(str).isin([str(m) for m in fit_months])
    holdout_mask = ledger["month"].astype(str).eq(str(holdout_month))
    rows: list[dict[str, Any]] = []
    for selected_col in selected_cols:
        if selected_col not in ledger.columns:
            raise KeyError(f"missing selected column {selected_col!r}")
        selected = ledger[selected_col].astype(bool)
        baselines: dict[tuple[str, str], dict[str, float]] = {}
        for period, period_mask in (("fit", fit_mask), ("holdout", holdout_mask)):
            for side, group in ledger[period_mask & selected].groupby("side_name", observed=True, dropna=False):
                baselines[(period, str(side))] = _metrics(group, round_trip_cost=float(round_trip_cost))
        for feature in state_cols:
            spec = _fit_bucket_spec(ledger.loc[fit_mask, feature], q=int(q))
            bucketed = _apply_bucket_spec(ledger[feature], spec).astype(str)
            work = ledger.assign(__bucket__=bucketed)
            for (side, bucket), fit_group in work[fit_mask & selected].groupby(
                ["side_name", "__bucket__"],
                observed=True,
                dropna=False,
            ):
                side_str = str(side)
                bucket_str = str(bucket)
                holdout_group = work[
                    holdout_mask
                    & selected
                    & work["side_name"].astype(str).eq(side_str)
                    & work["__bucket__"].astype(str).eq(bucket_str)
                ]
                fit_metrics = _metrics(fit_group, round_trip_cost=float(round_trip_cost))
                holdout_metrics = _metrics(holdout_group, round_trip_cost=float(round_trip_cost))
                fit_baseline = baselines.get(("fit", side_str), {})
                holdout_baseline = baselines.get(("holdout", side_str), {})
                row = {
                    "selected_col": str(selected_col),
                    "state_feature": str(feature),
                    "bucket": bucket_str,
                    "side": side_str,
                    "bucket_spec": json.dumps(_json_safe(spec), sort_keys=True),
                    **_prefix("fit", fit_metrics),
                    **_prefix("holdout", holdout_metrics),
                    **_prefix("fit_baseline", fit_baseline),
                    **_prefix("holdout_baseline", holdout_baseline),
                }
                row["fit_delta_evw"] = float(
                    row.get("fit_ev_weighted_first_touch_precision", np.nan)
                    - row.get("fit_baseline_ev_weighted_first_touch_precision", np.nan)
                )
                row["fit_delta_net"] = float(
                    row.get("fit_mean_first_touch_net", np.nan)
                    - row.get("fit_baseline_mean_first_touch_net", np.nan)
                )
                row["holdout_delta_evw"] = float(
                    row.get("holdout_ev_weighted_first_touch_precision", np.nan)
                    - row.get("holdout_baseline_ev_weighted_first_touch_precision", np.nan)
                )
                row["holdout_delta_net"] = float(
                    row.get("holdout_mean_first_touch_net", np.nan)
                    - row.get("holdout_baseline_mean_first_touch_net", np.nan)
                )
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fit_objective"] = _fit_objective(out)
    return out.sort_values(
        ["fit_objective", "fit_ev_weighted_first_touch_precision", "fit_mean_first_touch_net"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def select_archetype_overlay_rows(
    candidates: pd.DataFrame,
    *,
    selected_cols: list[str],
    sides: list[str],
    top_n_per_group: int,
    min_fit_selected_rows: int,
    min_fit_evw: float,
    min_fit_net: float,
    max_fit_first_touch_bad_mae: float,
    max_fit_mae_before: float,
    max_fit_adverse: float,
    max_fit_underwater: float,
    max_fit_timeout: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    rows = candidates.copy()
    rows["fit_objective"] = _fit_objective(rows)
    rows["eligible"] = _eligible_rows(
        rows,
        min_fit_selected_rows=int(min_fit_selected_rows),
        min_fit_evw=float(min_fit_evw),
        min_fit_net=float(min_fit_net),
        max_fit_first_touch_bad_mae=float(max_fit_first_touch_bad_mae),
        max_fit_mae_before=float(max_fit_mae_before),
        max_fit_adverse=float(max_fit_adverse),
        max_fit_underwater=float(max_fit_underwater),
        max_fit_timeout=float(max_fit_timeout),
    )
    selected_parts: list[pd.DataFrame] = []
    wanted_cols = selected_cols or sorted(rows["selected_col"].astype(str).unique())
    wanted_sides = sides or sorted(rows["side"].astype(str).unique())
    for selected_col in wanted_cols:
        for side in wanted_sides:
            group = rows[
                rows["selected_col"].astype(str).eq(str(selected_col))
                & rows["side"].astype(str).eq(str(side))
            ].copy()
            if group.empty:
                continue
            eligible = group[group["eligible"].astype(bool)].copy()
            pool = eligible if not eligible.empty else group
            pool = pool.sort_values(
                [
                    "fit_objective",
                    "fit_ev_weighted_first_touch_precision",
                    "fit_mean_first_touch_net",
                    "fit_first_touch_bad_mae_1r_rate",
                ],
                ascending=[False, False, False, True],
            )
            chosen = pool.head(max(1, int(top_n_per_group))).copy()
            chosen["selection_reason"] = "eligible_fit_best" if not eligible.empty else "fallback_no_eligible"
            selected_parts.append(chosen)
    if not selected_parts:
        return pd.DataFrame()
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["gate_status"] = selected.apply(_gate_status, axis=1)
    return selected.reset_index(drop=True)


def _write_report(output_dir: Path, selected: pd.DataFrame, candidates: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def table(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].head(n).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "selected_col",
        "side",
        "state_feature",
        "bucket",
        "selection_reason",
        "fit_objective",
        "fit_selected_rows",
        "holdout_selected_rows",
        "fit_ev_weighted_first_touch_precision",
        "holdout_ev_weighted_first_touch_precision",
        "fit_delta_evw",
        "holdout_delta_evw",
        "fit_mean_first_touch_net",
        "holdout_mean_first_touch_net",
        "fit_delta_net",
        "holdout_delta_net",
        "fit_first_touch_bad_mae_1r_rate",
        "holdout_first_touch_bad_mae_1r_rate",
        "fit_mae_before_mfe_1r_rate",
        "holdout_mae_before_mfe_1r_rate",
        "fit_mean_max_adverse_before_mfe_1r",
        "holdout_mean_max_adverse_before_mfe_1r",
        "fit_mean_underwater_bars_before_mfe_1r",
        "holdout_mean_underwater_bars_before_mfe_1r",
        "fit_timeout_rate",
        "holdout_timeout_rate",
        "gate_status",
    ]
    top_candidates = candidates.sort_values("fit_objective", ascending=False) if not candidates.empty else candidates
    lines = [
        "# S52 State Archetype Overlay",
        "",
        "Selection uses fit-month AE/GMM/state bucket metrics only. Holdout columns are reported after selection.",
        "",
        f"Candidate rows: `{len(candidates)}`",
        f"Selected rows: `{len(selected)}`",
        f"Variant: `{manifest['variant']}`",
        f"Fit months: `{', '.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        "",
        "## Selected Fit-Only State Buckets",
        "",
        table(selected, cols, n=60),
        "",
        "## Top Fit Candidates",
        "",
        table(top_candidates, cols, n=60),
        "",
        "## Outputs",
        "",
        f"- Selected: `{manifest['outputs']['selected']}`",
        f"- Candidates: `{manifest['outputs']['candidates']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    (output_dir / "s52_state_archetype_overlay.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(
    *,
    ledger_path: Path,
    output_dir: Path,
    variant: str,
    selected_cols: list[str],
    fit_months: list[str],
    holdout_month: str,
    sides: list[str],
    round_trip_cost: float,
    q: int,
    max_features: int,
    top_n_per_group: int,
    min_fit_selected_rows: int,
    min_fit_evw: float,
    min_fit_net: float,
    max_fit_first_touch_bad_mae: float,
    max_fit_mae_before: float,
    max_fit_adverse: float,
    max_fit_underwater: float,
    max_fit_timeout: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(ledger_path)
    if variant:
        ledger = ledger[ledger["variant"].astype(str).eq(str(variant))].copy()
    ledger = ledger.reset_index(drop=True)
    if ledger.empty:
        raise ValueError(f"No rows found for variant {variant!r}")
    candidates = build_archetype_candidates(
        ledger,
        selected_cols=selected_cols,
        fit_months=fit_months,
        holdout_month=holdout_month,
        round_trip_cost=float(round_trip_cost),
        q=int(q),
        max_features=int(max_features),
    )
    if not candidates.empty:
        candidates = candidates.copy()
        candidates["fit_objective"] = _fit_objective(candidates)
        candidates["eligible"] = _eligible_rows(
            candidates,
            min_fit_selected_rows=int(min_fit_selected_rows),
            min_fit_evw=float(min_fit_evw),
            min_fit_net=float(min_fit_net),
            max_fit_first_touch_bad_mae=float(max_fit_first_touch_bad_mae),
            max_fit_mae_before=float(max_fit_mae_before),
            max_fit_adverse=float(max_fit_adverse),
            max_fit_underwater=float(max_fit_underwater),
            max_fit_timeout=float(max_fit_timeout),
        )
        candidates["gate_status"] = candidates.apply(_gate_status, axis=1)
    selected = select_archetype_overlay_rows(
        candidates,
        selected_cols=selected_cols,
        sides=sides,
        top_n_per_group=int(top_n_per_group),
        min_fit_selected_rows=int(min_fit_selected_rows),
        min_fit_evw=float(min_fit_evw),
        min_fit_net=float(min_fit_net),
        max_fit_first_touch_bad_mae=float(max_fit_first_touch_bad_mae),
        max_fit_mae_before=float(max_fit_mae_before),
        max_fit_adverse=float(max_fit_adverse),
        max_fit_underwater=float(max_fit_underwater),
        max_fit_timeout=float(max_fit_timeout),
    )
    paths = {
        "candidates": output_dir / "s52_state_archetype_candidates.csv",
        "selected": output_dir / "s52_state_archetype_selected.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "s52_state_archetype_overlay.md",
    }
    candidates.to_csv(paths["candidates"], index=False)
    selected.to_csv(paths["selected"], index=False)
    manifest = {
        "scope": "s52_state_archetype_overlay",
        "ledger_path": str(ledger_path),
        "output_dir": str(output_dir),
        "variant": str(variant),
        "selected_cols": [str(v) for v in selected_cols],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "sides": [str(v) for v in sides],
        "round_trip_cost": float(round_trip_cost),
        "q": int(q),
        "max_features": int(max_features),
        "top_n_per_group": int(top_n_per_group),
        "min_fit_selected_rows": int(min_fit_selected_rows),
        "min_fit_evw": float(min_fit_evw),
        "min_fit_net": float(min_fit_net),
        "max_fit_first_touch_bad_mae": float(max_fit_first_touch_bad_mae),
        "max_fit_mae_before": float(max_fit_mae_before),
        "max_fit_adverse": float(max_fit_adverse),
        "max_fit_underwater": float(max_fit_underwater),
        "max_fit_timeout": float(max_fit_timeout),
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(selected)),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(output_dir, selected, candidates, manifest)
    return {
        "output_dir": str(output_dir),
        "selected": str(paths["selected"]),
        "candidates": str(paths["candidates"]),
        "report": str(paths["markdown"]),
        "top": _json_safe(selected.head(20).to_dict(orient="records")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default="ranker_timestamp_soft_ordered_ev")
    parser.add_argument("--selected-cols", default="selected_top10,selected_top20,selected_top30")
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--sides", default="long,short")
    parser.add_argument("--round-trip-cost", type=float, default=0.0100)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=0)
    parser.add_argument("--top-n-per-group", type=int, default=3)
    parser.add_argument("--min-fit-selected-rows", type=int, default=80)
    parser.add_argument("--min-fit-evw", type=float, default=0.65)
    parser.add_argument("--min-fit-net", type=float, default=-0.006)
    parser.add_argument("--max-fit-first-touch-bad-mae", type=float, default=0.25)
    parser.add_argument("--max-fit-mae-before", type=float, default=0.35)
    parser.add_argument("--max-fit-adverse", type=float, default=1.75)
    parser.add_argument("--max-fit-underwater", type=float, default=12.0)
    parser.add_argument("--max-fit-timeout", type=float, default=0.12)
    args = parser.parse_args()
    result = run_report(
        ledger_path=args.ledger_path,
        output_dir=args.output_dir,
        variant=str(args.variant),
        selected_cols=_parse_csv(args.selected_cols),
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        sides=_parse_csv(args.sides),
        round_trip_cost=float(args.round_trip_cost),
        q=int(args.q),
        max_features=int(args.max_features),
        top_n_per_group=int(args.top_n_per_group),
        min_fit_selected_rows=int(args.min_fit_selected_rows),
        min_fit_evw=float(args.min_fit_evw),
        min_fit_net=float(args.min_fit_net),
        max_fit_first_touch_bad_mae=float(args.max_fit_first_touch_bad_mae),
        max_fit_mae_before=float(args.max_fit_mae_before),
        max_fit_adverse=float(args.max_fit_adverse),
        max_fit_underwater=float(args.max_fit_underwater),
        max_fit_timeout=float(args.max_fit_timeout),
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
