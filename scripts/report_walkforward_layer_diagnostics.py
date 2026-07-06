#!/usr/bin/env python3
"""Build layer-by-layer diagnostics for monthly walk-forward policy-OOS runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORT_ID = (
    "20260701_193000_single_head_monthly_walkforward_forwardburnin_"
    "no_window_hpo_no_regime_fe"
)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _fmt_float(value: Any, digits: int = 4) -> str:
    val = _finite(value)
    if val is None:
        return ""
    return f"{val:.{digits}f}"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _diagnosis_for_row(row: pd.Series) -> str:
    oof_auc = _finite(row.get("oof_auc"))
    oof_ic = _finite(row.get("oof_ic"))
    label_mean = _finite(row.get("policy_oos_top15_mean_label_return"))
    selected_label = _finite(row.get("selected_label_mean_return"))
    gross_mean = _finite(row.get("sim_gross_mean_return"))
    net_mean = _finite(row.get("sim_net_mean_return"))
    source_top15 = _finite(row.get("source_candidate_top15_net_pnl"))
    best_overlay = _finite(row.get("best_overlay_net_pnl"))

    if oof_auc is not None and oof_auc < 0.5 and oof_ic is not None and oof_ic < 0.0:
        if label_mean is not None and label_mean > 0.0:
            if gross_mean is not None and gross_mean < 0.0:
                return "OOF ranking weak; OOS label still positive; exit path turns it gross-negative"
            return "OOF ranking weak, but OOS label survives before execution"
        return "OOF ranking weak and OOS label economics weak"
    if selected_label is not None and selected_label > 0.0 and gross_mean is not None and gross_mean < 0.0:
        return "Selection finds positive labels but execution geometry loses them"
    if source_top15 is not None and source_top15 > 0.0 and best_overlay is not None and best_overlay < 0.0:
        return "Broad candidates positive, selected overlay negative: policy/allocation mismatch"
    if net_mean is not None and net_mean < 0.0:
        return "Net execution negative"
    return "No single dominant loss layer"


def build_layer_diagnostics(report_dir: Path) -> pd.DataFrame:
    failure_dir = report_dir / "failure_attribution_tests"
    diag_dir = report_dir / "oos_month_week_diagnosis"

    model = _read_csv(failure_dir / "model_signal_oof_vs_oos.csv")
    execution = _read_csv(failure_dir / "execution_translation.csv")
    candidate = _read_csv(failure_dir / "candidate_vs_overlay.csv")
    vanilla = _read_csv(diag_dir / "clean_single_head_monthly.csv")
    optuna = _read_csv(diag_dir / "optuna_single_head_monthly.csv")

    months = sorted(
        {
            str(v)
            for frame in (model, execution, candidate, vanilla, optuna)
            if not frame.empty and "eval_month" in frame.columns
            for v in frame["eval_month"].dropna().astype(str).tolist()
        }
    )

    rows: list[dict[str, Any]] = []
    for month in months:
        row: dict[str, Any] = {"eval_month": month}
        if not model.empty:
            match = model[model["eval_month"].astype(str).eq(month)]
            if not match.empty:
                row.update(match.iloc[0].to_dict())
        if not execution.empty:
            match = execution[execution["eval_month"].astype(str).eq(month)]
            if not match.empty:
                row.update(match.iloc[0].to_dict())
        if not vanilla.empty:
            match = vanilla[vanilla["eval_month"].astype(str).eq(month)]
            if not match.empty:
                v = match.iloc[0]
                row.update(
                    {
                        "vanilla_trades": int(v.get("n_trades", 0) or 0),
                        "vanilla_net_pnl": _finite(v.get("net_pnl")),
                        "vanilla_mean_net": _finite(v.get("mean_net")),
                        "vanilla_hit_rate": _finite(v.get("hit_rate")),
                    }
                )
        if not optuna.empty:
            match = optuna[optuna["eval_month"].astype(str).eq(month)]
            if not match.empty:
                o = match.iloc[0]
                row.update(
                    {
                        "optuna_trades": int(o.get("n_trades", 0) or 0),
                        "optuna_net_pnl": _finite(o.get("net_pnl")),
                        "optuna_mean_net": _finite(
                            o.get("mean_net_trade", o.get("mean_net"))
                        ),
                        "optuna_hit_rate": _finite(o.get("hit_rate")),
                    }
                )
        if not candidate.empty:
            match = candidate[candidate["eval_month"].astype(str).eq(month)]
            if not match.empty:
                row.update(match.iloc[0].to_dict())
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["layer_diagnosis"] = out.apply(_diagnosis_for_row, axis=1)
    preferred_cols = [
        "eval_month",
        "oof_auc",
        "oof_ic",
        "oof_top15_hit_rate",
        "policy_oos_top15_label_hit_rate",
        "hit_rate_gap_pp",
        "oof_top15_mean_return",
        "policy_oos_top15_mean_label_return",
        "mean_return_gap",
        "selected_label_mean_return",
        "sim_gross_mean_return",
        "sim_net_mean_return",
        "label_to_gross_mean_return_drag",
        "gross_to_net_mean_return_drag",
        "full_sl_exit_rate",
        "trailing_exit_rate",
        "vanilla_trades",
        "vanilla_net_pnl",
        "optuna_trades",
        "optuna_net_pnl",
        "source_candidate_top15_net_pnl",
        "source_candidate_top15_n",
        "best_overlay_net_pnl",
        "positive_overlay_policies",
        "overlay_policy_count",
        "layer_diagnosis",
    ]
    cols = [col for col in preferred_cols if col in out.columns]
    cols.extend([col for col in out.columns if col not in set(cols)])
    return out[cols]


def _small_view(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "eval_month",
        "oof_auc",
        "oof_ic",
        "policy_oos_top15_mean_label_return",
        "selected_label_mean_return",
        "sim_gross_mean_return",
        "sim_net_mean_return",
        "full_sl_exit_rate",
        "vanilla_net_pnl",
        "optuna_net_pnl",
        "source_candidate_top15_net_pnl",
        "best_overlay_net_pnl",
        "layer_diagnosis",
    ]
    view = df[[col for col in cols if col in df.columns]].copy()
    for col in view.columns:
        if col == "eval_month" or col == "layer_diagnosis":
            continue
        view[col] = view[col].map(lambda value: _fmt_float(value, 4))
    return view


def write_report(report_dir: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = build_layer_diagnostics(report_dir)
    layer_path = output_dir / "layer_diagnostics.csv"
    layer.to_csv(layer_path, index=False)

    markdown_path = output_dir / "layer_diagnostics.md"
    view = _small_view(layer)
    lines = [
        "# Walk-Forward Layer Diagnostics",
        "",
        "This table joins model OOF quality, policy-OOS label quality, vanilla execution translation, Optuna policy output, and source-overlay evidence by month.",
        "",
    ]
    if view.empty:
        lines.append("No layer diagnostics could be built from the available artifacts.")
    else:
        lines.append(view.to_markdown(index=False))
        lines.extend(
            [
                "",
                "## Readout",
                "",
                "- `oof_*`: training-window OOF context for the monthly model.",
                "- `policy_oos_*`: label economics on untouched policy-OOS rows before execution.",
                "- `sim_*`: executable vanilla simulator returns after exit path and costs.",
                "- `source_candidate_*` and `best_overlay_*`: source-run broad candidates and selected overlays in the same exact OOS windows where available.",
            ]
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "report_dir": str(report_dir),
        "output_dir": str(output_dir),
        "layer_diagnostics": str(layer_path),
        "markdown": str(markdown_path),
        "rows": int(len(layer)),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "layer_diagnostics": str(layer_path),
        "markdown": str(markdown_path),
        "manifest": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-id",
        default=DEFAULT_REPORT_ID,
        help="Report id under data_perp/reports.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Explicit report directory; overrides --report-id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <report-dir>/layer_diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir or Path("data_perp") / "reports" / str(args.report_id)
    output_dir = args.output_dir or report_dir / "layer_diagnostics"
    paths = write_report(report_dir, output_dir)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
