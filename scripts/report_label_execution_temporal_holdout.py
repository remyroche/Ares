#!/usr/bin/env python3
"""Select label/execution ablation candidates on fit months and test holdout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("data_perp/reports/label_execution_alignment_ablation_s14_v2")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_execution_alignment_temporal_holdout_s14_v2")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    try:
        out = float(value)
    except Exception:
        return str(value)
    return out if math.isfinite(out) else None


def _safe_mean(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_min(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.min()) if len(arr) else float("nan")


def _score_fit(group: pd.DataFrame) -> dict[str, Any]:
    net = pd.to_numeric(group["net_pnl"], errors="coerce")
    n_trades = pd.to_numeric(group["n_trades"], errors="coerce")
    mean_trade = pd.to_numeric(group["mean_net_trade"], errors="coerce")
    hit = pd.to_numeric(group["hit_rate"], errors="coerce")
    full_sl = pd.to_numeric(group["full_sl_exit_rate"], errors="coerce")
    return {
        "fit_months": int(group["eval_month"].nunique()),
        "fit_net_pnl": float(net.sum()),
        "fit_positive_months": int((net > 0.0).sum()),
        "fit_worst_month_net_pnl": float(net.min()) if len(net) else float("nan"),
        "fit_n_trades": int(n_trades.sum()),
        "fit_mean_net_trade": _safe_mean(mean_trade),
        "fit_hit_rate": _safe_mean(hit),
        "fit_full_sl_exit_rate": _safe_mean(full_sl),
    }


def _score_holdout(row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {
            "holdout_present": False,
            "holdout_net_pnl": float("nan"),
            "holdout_n_trades": 0,
            "holdout_mean_net_trade": float("nan"),
            "holdout_hit_rate": float("nan"),
            "holdout_full_sl_exit_rate": float("nan"),
            "holdout_trailing_exit_rate": float("nan"),
        }
    return {
        "holdout_present": True,
        "holdout_net_pnl": float(row.get("net_pnl", float("nan"))),
        "holdout_n_trades": int(row.get("n_trades", 0) or 0),
        "holdout_mean_net_trade": float(row.get("mean_net_trade", float("nan"))),
        "holdout_hit_rate": float(row.get("hit_rate", float("nan"))),
        "holdout_full_sl_exit_rate": float(row.get("full_sl_exit_rate", float("nan"))),
        "holdout_trailing_exit_rate": float(row.get("trailing_exit_rate", float("nan"))),
    }


def _summarize(
    monthly: pd.DataFrame,
    *,
    fit_months: list[str],
    holdout_month: str,
    selector: str,
) -> pd.DataFrame:
    frame = monthly[monthly["selector"].astype(str).eq(selector)].copy()
    frame["eval_month"] = frame["eval_month"].astype(str)
    fit = frame[frame["eval_month"].isin(fit_months)].copy()
    holdout = frame[frame["eval_month"].eq(holdout_month)].copy()
    rows: list[dict[str, Any]] = []
    for (arm, top_frac), group in fit.groupby(["arm", "top_frac"], dropna=False, observed=True):
        row: dict[str, Any] = {
            "arm": str(arm),
            "selector": selector,
            "top_frac": float(top_frac),
        }
        row.update(_score_fit(group))
        hold = holdout[
            holdout["arm"].astype(str).eq(str(arm))
            & (pd.to_numeric(holdout["top_frac"], errors="coerce") == float(top_frac))
        ]
        row.update(_score_holdout(hold.iloc[0] if not hold.empty else None))
        row["decision"] = (
            "holdout_pass"
            if (
                row["fit_months"] == len(fit_months)
                and row["fit_positive_months"] == len(fit_months)
                and row["fit_net_pnl"] > 0.0
                and row["fit_worst_month_net_pnl"] >= -0.005
                and row["holdout_present"]
                and row["holdout_net_pnl"] > 0.0
                and row["holdout_n_trades"] >= 10
            )
            else "reject_or_rework"
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        [
            "decision",
            "holdout_net_pnl",
            "fit_worst_month_net_pnl",
            "fit_net_pnl",
            "top_frac",
        ],
        ascending=[True, False, False, False, True],
    )


def _write_markdown(summary: pd.DataFrame, manifest: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "label_execution_temporal_holdout.md"
    passes = summary[summary["decision"].eq("holdout_pass")].copy()
    rejected = summary[~summary["decision"].eq("holdout_pass")].copy()
    rejected = rejected.sort_values(
        ["holdout_net_pnl", "fit_net_pnl", "fit_worst_month_net_pnl"],
        ascending=[False, False, False],
    ).head(16)

    cols = [
        "decision",
        "arm",
        "top_frac",
        "fit_net_pnl",
        "fit_positive_months",
        "fit_worst_month_net_pnl",
        "fit_n_trades",
        "holdout_net_pnl",
        "holdout_n_trades",
        "holdout_mean_net_trade",
        "holdout_hit_rate",
        "holdout_full_sl_exit_rate",
    ]

    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    lines = [
        "# Label Execution Temporal Holdout",
        "",
        "Scope: select on fit months only, evaluate the later holdout month.",
        "",
        f"Input: `{manifest['input_dir']}`",
        f"Fit months: `{', '.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Selector: `{manifest['selector']}`",
        "",
        "## Holdout Pass",
        "",
        table(passes),
        "",
        "## Top Rejected/Rework",
        "",
        table(rejected),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--selector", default="ablation_lgbm")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = input_dir / "label_execution_monthly.csv"
    if not monthly_path.exists():
        raise FileNotFoundError(monthly_path)
    fit_months = [part.strip() for part in str(args.fit_months).split(",") if part.strip()]
    holdout_month = str(args.holdout_month).strip()
    monthly = pd.read_csv(monthly_path)
    summary = _summarize(
        monthly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        selector=str(args.selector),
    )
    summary_path = output_dir / "label_execution_temporal_holdout_summary.csv"
    manifest_path = output_dir / "manifest.json"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "input_dir": str(input_dir),
        "monthly_path": str(monthly_path),
        "fit_months": fit_months,
        "holdout_month": holdout_month,
        "selector": str(args.selector),
        "outputs": {
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
    }
    md_path = _write_markdown(summary, manifest, output_dir)
    manifest["outputs"]["markdown"] = str(md_path)
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
