#!/usr/bin/env python3
"""Ablate causal acceptance overlays on no-training label proxy ledgers.

The input ledgers are produced by `export_label_proxy_gated_candidate_ledger.py`.
This script does not train a model and does not use future outcome fields for
selection. Overlays use only proxy-time fields available in the ledger:
candidate score, risk score, timestamps, and weeks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORT_DIRS = (
    Path("data_perp/reports/label_proxy_gated_tail_label_grid_v1/S16_tail_utility_soft__W12_tail_timestamp_balanced"),
    Path("data_perp/reports/label_proxy_gated_policy_tail_blend_grid_v1/S24_policy_tail_s14_lean__W9_tail_utility"),
    Path("data_perp/reports/label_proxy_gated_candidate_ledger_s14_w12_v1"),
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_candidate_acceptance_layer_v1")


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


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    arr = _safe_numeric(values).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    arr = _safe_numeric(values).dropna()
    return float(arr.quantile(q)) if len(arr) else float("nan")


def _read_manifest(report_dir: Path) -> dict[str, Any]:
    path = report_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_report(report_dir: Path) -> pd.DataFrame:
    manifest = _read_manifest(report_dir)
    ledger_path = report_dir / "selected_ledger.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(ledger_path)
    ledger = pd.read_csv(ledger_path)
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], errors="coerce")
    ledger["label_arm"] = str(manifest.get("label_arm") or report_dir.name.split("__", 1)[0])
    ledger["weight_arm"] = str(manifest.get("weight_arm") or report_dir.name.split("__", 1)[-1])
    ledger["report_dir"] = str(report_dir)
    if "candidate_score" not in ledger.columns:
        ledger["candidate_score"] = _safe_numeric(ledger.get("score"))
    if "risk_score" not in ledger.columns:
        ledger["risk_score"] = np.nan
    ledger["risk_kind"] = ledger["risk_kind"].astype(str)
    ledger["risk_keep_frac"] = _safe_numeric(ledger["risk_keep_frac"])
    return ledger


def _rank_keep(frame: pd.DataFrame, score_col: str, keep_frac: float, *, ascending: bool = False) -> pd.Series:
    score = _safe_numeric(frame[score_col])
    valid = score.notna()
    keep = pd.Series(False, index=frame.index)
    if not bool(valid.any()):
        return keep
    ranks = score[valid].rank(method="first", pct=True, ascending=True)
    if ascending:
        keep.loc[ranks.index] = ranks <= float(keep_frac)
    else:
        keep.loc[ranks.index] = ranks >= (1.0 - float(keep_frac))
    return keep


def _overlay_mask(frame: pd.DataFrame, overlay: str) -> pd.Series:
    out = pd.Series(True, index=frame.index)
    if frame.empty or overlay == "base":
        return out
    work = frame.copy()
    work["score_margin"] = _safe_numeric(work["candidate_score"]) - _safe_numeric(work["risk_score"])
    if overlay.startswith("score_top_"):
        keep_frac = float(overlay.rsplit("_", 1)[-1])
        return _rank_keep(work, "candidate_score", keep_frac)
    if overlay.startswith("risk_low_"):
        keep_frac = float(overlay.rsplit("_", 1)[-1])
        return _rank_keep(work, "risk_score", keep_frac, ascending=True)
    if overlay.startswith("margin_min_"):
        threshold = float(overlay.rsplit("_", 1)[-1])
        return work["score_margin"] >= threshold
    if overlay.startswith("margin_top_"):
        keep_frac = float(overlay.rsplit("_", 1)[-1])
        return _rank_keep(work, "score_margin", keep_frac)
    if overlay.startswith("adj_b"):
        rest = overlay.removeprefix("adj_b")
        beta_text, keep_text = rest.split("_top_")
        beta = float(beta_text)
        keep_frac = float(keep_text)
        work["adjusted_score"] = _safe_numeric(work["candidate_score"]) - beta * _safe_numeric(work["risk_score"])
        return _rank_keep(work, "adjusted_score", keep_frac)
    if overlay.startswith("weekcap_"):
        cap = int(float(overlay.rsplit("_", 1)[-1]))
        score = _safe_numeric(work["candidate_score"]) - 0.5 * _safe_numeric(work["risk_score"])
        keep = pd.Series(False, index=work.index)
        for _week, group in work.assign(_accept_score=score).groupby("week", dropna=False, observed=True):
            chosen = group.sort_values("_accept_score", ascending=False, kind="mergesort").head(cap)
            keep.loc[chosen.index] = True
        return keep
    raise ValueError(f"Unknown overlay: {overlay}")


def _selection_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "mean_u": float("nan"),
            "hit_u": float("nan"),
            "q10_u": float("nan"),
            "bad_mae_1r_rate": float("nan"),
            "wide_barrier_25bps_rate": float("nan"),
            "wide_barrier_35bps_rate": float("nan"),
            "mean_candidate_score": float("nan"),
            "mean_risk_score": float("nan"),
            "top_symbol_share": 0.0,
        }
    symbol_counts = frame["__symbol__"].astype(str).value_counts(dropna=False)
    return {
        "rows": int(len(frame)),
        "mean_u": _safe_mean(frame["u_policy_net"]),
        "hit_u": _safe_mean(_safe_numeric(frame["u_policy_net"]) > 0.0),
        "q10_u": _safe_quantile(frame["u_policy_net"], 0.10),
        "bad_mae_1r_rate": _safe_mean(_safe_numeric(frame["mae_norm"]) >= 1.0),
        "wide_barrier_25bps_rate": _safe_mean(_safe_numeric(frame["barrier"]) >= 0.025),
        "wide_barrier_35bps_rate": _safe_mean(_safe_numeric(frame["barrier"]) >= 0.035),
        "mean_candidate_score": _safe_mean(frame["candidate_score"]),
        "mean_risk_score": _safe_mean(frame["risk_score"]),
        "top_symbol_share": float(symbol_counts.iloc[0] / len(frame)) if len(frame) else 0.0,
    }


def _summarize_group(
    *,
    frame: pd.DataFrame,
    base_rows: int,
    overlay: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    group_cols = ["label_arm", "weight_arm", "risk_kind", "risk_keep_frac", "top_frac"]
    identity = {col: frame[col].iloc[0] for col in group_cols}
    identity["overlay"] = overlay
    for month, month_frame in frame.groupby("month", dropna=False, observed=True):
        row = {**identity, "period": str(month), "base_rows": int(base_rows)}
        row.update(_selection_metrics(month_frame))
        monthly_rows.append(row)
        for week, week_frame in month_frame.groupby("week", dropna=False, observed=True):
            week_row = {**identity, "month": str(month), "week": str(week), "base_rows": int(base_rows)}
            week_row.update(_selection_metrics(week_frame))
            weekly_rows.append(week_row)
    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    month_mean = _safe_numeric(monthly["mean_u"]) if not monthly.empty else pd.Series(dtype=float)
    week_mean = _safe_numeric(weekly["mean_u"]) if not weekly.empty else pd.Series(dtype=float)
    accepted_rows = int(len(frame))
    aggregate = {
        **identity,
        "base_rows": int(base_rows),
        "accepted_rows": accepted_rows,
        "accepted_frac": float(accepted_rows / base_rows) if base_rows else 0.0,
        "months": int(monthly["period"].nunique()) if not monthly.empty else 0,
        "positive_months": int((month_mean > 0.0).sum()),
        "mean_u": _safe_mean(month_mean),
        "worst_month_mean_u": _safe_quantile(month_mean, 0.0),
        "selected_weeks": int(len(weekly)),
        "positive_selected_weeks": int((week_mean > 0.0).sum()),
        "q25_week_mean_u": _safe_quantile(week_mean, 0.25),
        "worst_week_mean_u": _safe_quantile(week_mean, 0.0),
        "hit_u": _safe_mean(monthly["hit_u"]) if not monthly.empty else float("nan"),
        "q10_u": _safe_mean(monthly["q10_u"]) if not monthly.empty else float("nan"),
        "bad_mae_1r_rate": _safe_mean(monthly["bad_mae_1r_rate"]) if not monthly.empty else float("nan"),
        "wide_barrier_25bps_rate": _safe_mean(monthly["wide_barrier_25bps_rate"]) if not monthly.empty else float("nan"),
        "wide_barrier_35bps_rate": _safe_mean(monthly["wide_barrier_35bps_rate"]) if not monthly.empty else float("nan"),
        "mean_candidate_score": _safe_mean(monthly["mean_candidate_score"]) if not monthly.empty else float("nan"),
        "mean_risk_score": _safe_mean(monthly["mean_risk_score"]) if not monthly.empty else float("nan"),
        "mean_rows_month": _safe_mean(monthly["rows"]) if not monthly.empty else float("nan"),
        "min_rows_month": int(_safe_numeric(monthly["rows"]).min()) if not monthly.empty else 0,
    }
    return monthly_rows, weekly_rows, aggregate


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_candidate_acceptance_layer.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "label_arm",
        "weight_arm",
        "risk_kind",
        "risk_keep_frac",
        "top_frac",
        "overlay",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "selected_weeks",
        "positive_selected_weeks",
        "q25_week_mean_u",
        "worst_week_mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "accepted_frac",
        "mean_rows_month",
    ]
    positive = aggregate[pd.to_numeric(aggregate["mean_u"], errors="coerce") > 0.0].copy()
    strict = positive[
        (pd.to_numeric(positive["worst_month_mean_u"], errors="coerce") >= -0.002)
        & (pd.to_numeric(positive["q25_week_mean_u"], errors="coerce") >= -0.006)
        & (
            pd.to_numeric(positive["positive_selected_weeks"], errors="coerce")
            >= 0.50 * pd.to_numeric(positive["selected_weeks"], errors="coerce")
        )
    ].sort_values(
        ["q25_week_mean_u", "worst_month_mean_u", "mean_u"],
        ascending=[False, False, False],
    )
    lines = [
        "# Label Candidate Acceptance Layer",
        "",
        "Scope: no model training. Overlays use only proxy-time scores from already causal candidate ledgers.",
        "",
        "## Strict Candidates",
        "",
        table(strict, cols, limit=30),
        "",
        "## Best By Weekly Lower Tail",
        "",
        table(
            positive.sort_values(
                ["q25_week_mean_u", "worst_month_mean_u", "mean_u"],
                ascending=[False, False, False],
            ),
            cols,
            limit=40,
        ),
        "",
        "## Best By Mean Utility",
        "",
        table(
            aggregate.sort_values(
                ["mean_u", "worst_month_mean_u", "q25_week_mean_u"],
                ascending=[False, False, False],
            ),
            cols,
            limit=40,
        ),
        "",
        "## All-Month-Positive",
        "",
        table(
            aggregate[pd.to_numeric(aggregate["positive_months"], errors="coerce") >= 3].sort_values(
                ["mean_u", "q25_week_mean_u"],
                ascending=[False, False],
            ),
            cols,
            limit=40,
        ),
        "",
        "## Outputs",
        "",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    report_dirs: tuple[Path, ...],
    output_dir: Path,
    overlays: tuple[str, ...],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledgers = [_load_report(path) for path in report_dirs]
    ledger = pd.concat(ledgers, ignore_index=True) if len(ledgers) > 1 else ledgers[0].copy()
    group_cols = ["label_arm", "weight_arm", "risk_kind", "risk_keep_frac", "top_frac"]
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for _key, group in ledger.groupby(group_cols, dropna=False, observed=True):
        group = group.copy()
        base_rows = int(len(group))
        for overlay in overlays:
            accepted_parts: list[pd.DataFrame] = []
            for _month, month_group in group.groupby("month", dropna=False, observed=True):
                mask = _overlay_mask(month_group, overlay)
                accepted_parts.append(month_group.loc[mask].copy())
            accepted = pd.concat(accepted_parts, ignore_index=False) if accepted_parts else group.iloc[:0].copy()
            if accepted.empty:
                continue
            monthly, weekly, aggregate = _summarize_group(
                frame=accepted,
                base_rows=base_rows,
                overlay=overlay,
            )
            monthly_rows.extend(monthly)
            weekly_rows.extend(weekly)
            aggregate_rows.append(aggregate)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    aggregate = pd.DataFrame(aggregate_rows)
    if not aggregate.empty:
        aggregate = aggregate.sort_values(
            ["mean_u", "worst_month_mean_u", "q25_week_mean_u"],
            ascending=[False, False, False],
        )
    paths = {
        "aggregate": output_dir / "acceptance_aggregate_summary.csv",
        "monthly": output_dir / "acceptance_monthly_summary.csv",
        "weekly": output_dir / "acceptance_weekly_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    aggregate.to_csv(paths["aggregate"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    manifest = {
        "report_dirs": [str(path) for path in report_dirs],
        "output_dir": str(output_dir),
        "overlays": list(overlays),
        "selection_fields": ["candidate_score", "risk_score", "__ts__", "week", "month"],
        "evaluation_fields": ["u_policy_net", "barrier", "mae_norm", "q10_u", "hit_u"],
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, action="append", default=list(DEFAULT_REPORT_DIRS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overlays",
        default=(
            "base,"
            "score_top_0.75,score_top_0.50,"
            "risk_low_0.75,risk_low_0.50,"
            "margin_min_0.20,margin_min_0.30,"
            "margin_top_0.75,margin_top_0.50,"
            "adj_b0.25_top_0.75,adj_b0.25_top_0.50,"
            "adj_b0.50_top_0.75,adj_b0.50_top_0.50,"
            "adj_b0.75_top_0.75,adj_b0.75_top_0.50,"
            "weekcap_1,weekcap_2,weekcap_3"
        ),
    )
    return parser.parse_args()


def _parse_overlays(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        report_dirs=tuple(args.report_dir),
        output_dir=args.output_dir,
        overlays=_parse_overlays(args.overlays),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
