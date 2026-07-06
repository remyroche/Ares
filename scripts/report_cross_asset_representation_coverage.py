#!/usr/bin/env python3
"""Report cross-asset representation OOF coverage and stability-readiness.

Cross-asset representation features can be materialized for every month after
the first source handoff month.  Stability priors need more history: with only
three source months, the final validation month can receive a prior value, but
the meta model has no earlier training rows with non-null priors to learn from.

This report makes that coverage contract explicit for a handoff/prediction
pair and can scan all local S52 handoff artifacts.
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

from scripts.audit_promoted_cross_asset_month_flip_attribution import _json_safe  # noqa: E402


DEFAULT_REPORT_ROOT = Path("data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1")
DEFAULT_OUT_DIR = DEFAULT_REPORT_ROOT / "cross_asset_representation_coverage_v1"
HANDOFF_NAME = "train_meta_regime_handoff.parquet"
PREDICTIONS_NAME = "cross_asset_representation_v1_predictions.parquet"
KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
        use_cols = [col for col in columns if col in available]
        return pd.read_parquet(path, columns=use_cols)
    except Exception:
        frame = pd.read_parquet(path)
        return frame[[col for col in columns if col in frame.columns]]


def _candidate_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith("selected_top"))


def _row_key(frame: pd.DataFrame) -> pd.Series:
    return frame[list(KEY_COLUMNS)].astype(str).agg("|".join, axis=1)


def _coverage_for_pair(handoff_path: Path, predictions_path: Path | None) -> tuple[dict[str, Any], pd.DataFrame]:
    handoff = _read_parquet(handoff_path)
    if "month" not in handoff.columns:
        raise ValueError(f"Handoff missing month column: {handoff_path}")
    missing_keys = [col for col in KEY_COLUMNS if col not in handoff.columns]
    if missing_keys:
        raise ValueError(f"Handoff missing key columns {missing_keys}: {handoff_path}")
    handoff["month"] = handoff["month"].astype(str)
    source_months = sorted(handoff["month"].dropna().unique().tolist())
    candidate_cols = _candidate_columns(handoff)
    selected_col = "selected_top10" if "selected_top10" in candidate_cols else (candidate_cols[0] if candidate_cols else None)
    handoff["_row_key"] = _row_key(handoff)
    if selected_col:
        selected_mask = pd.to_numeric(handoff[selected_col], errors="coerce").fillna(0.0).gt(0.5)
    else:
        selected_mask = pd.Series(True, index=handoff.index)

    pred = pd.DataFrame()
    if predictions_path is not None and predictions_path.exists():
        pred = _read_parquet(predictions_path, columns=[*KEY_COLUMNS, "month"])
        pred["month"] = pred["month"].astype(str)
        pred["_row_key"] = _row_key(pred)
    pred_keys = set(pred["_row_key"].astype(str).tolist()) if not pred.empty else set()

    rows: list[dict[str, Any]] = []
    for month in source_months:
        month_mask = handoff["month"].eq(month)
        selected_month = month_mask & selected_mask
        keys_all = set(handoff.loc[month_mask, "_row_key"].astype(str))
        keys_selected = set(handoff.loc[selected_month, "_row_key"].astype(str))
        represented_all = len(keys_all & pred_keys)
        represented_selected = len(keys_selected & pred_keys)
        rows.append(
            {
                "handoff_dir": str(handoff_path.parent),
                "month": month,
                "source_rows": int(month_mask.sum()),
                "selected_col": selected_col,
                "selected_rows": int(selected_month.sum()),
                "representation_rows": int(pred["month"].eq(month).sum()) if not pred.empty and "month" in pred.columns else 0,
                "represented_all_rows": int(represented_all),
                "represented_selected_rows": int(represented_selected),
                "coverage_all_rows": float(represented_all / max(int(month_mask.sum()), 1)),
                "coverage_selected_rows": float(represented_selected / max(int(selected_month.sum()), 1)),
            }
        )
    coverage = pd.DataFrame(rows)
    scored_months = sorted(pred["month"].dropna().astype(str).unique().tolist()) if not pred.empty else []
    expected_oof_months = source_months[1:]
    missing_oof_months = sorted(set(expected_oof_months).difference(scored_months))
    extra_scored_months = sorted(set(scored_months).difference(source_months))
    # Need at least four source months: M1 train-only, M2 OOF without prior,
    # M3 OOF with prior but no trained prior examples yet, M4 can validate with
    # train rows containing non-null stability priors.
    stability_learnable = len(source_months) >= 4 and len(scored_months) >= 3 and not missing_oof_months
    summary = {
        "handoff_dir": str(handoff_path.parent),
        "handoff_path": str(handoff_path),
        "predictions_path": str(predictions_path) if predictions_path else None,
        "source_months": source_months,
        "source_month_count": int(len(source_months)),
        "scored_months": scored_months,
        "scored_month_count": int(len(scored_months)),
        "expected_oof_months": expected_oof_months,
        "missing_oof_months": missing_oof_months,
        "extra_scored_months": extra_scored_months,
        "selected_col": selected_col,
        "source_rows": int(len(handoff)),
        "prediction_rows": int(len(pred)),
        "stability_context_learnable_in_month_forward_meta": bool(stability_learnable),
        "minimum_source_months_for_stability_learning": 4,
        "minimum_oof_representation_months_for_stability_learning": 3,
        "status": "ready_for_stability_meta_learning" if stability_learnable else "needs_more_source_months_or_oof_predictions",
    }
    return summary, coverage


def _find_pairs(report_root: Path) -> list[tuple[Path, Path | None]]:
    pairs: list[tuple[Path, Path | None]] = []
    for handoff in sorted(report_root.glob(f"**/{HANDOFF_NAME}")):
        base_dir = handoff.parent
        candidates = [
            base_dir / "cross_asset_archetype_representation_v1" / PREDICTIONS_NAME,
            base_dir / "cross_asset_representation_v1" / PREDICTIONS_NAME,
            base_dir / PREDICTIONS_NAME,
        ]
        pred = next((p for p in candidates if p.exists()), None)
        pairs.append((handoff, pred))
    return pairs


def _write_markdown(out_dir: Path, manifest: dict[str, Any], summary: pd.DataFrame, coverage: pd.DataFrame) -> Path:
    lines = [
        "# Cross-Asset Representation Coverage",
        "",
        "## Verdict",
        "",
        f"- scanned handoffs: `{manifest.get('scanned_handoffs')}`",
        f"- ready handoffs: `{manifest.get('ready_handoffs')}`",
        "",
        "## Summary",
        "",
    ]
    display_cols = [
        "status",
        "source_month_count",
        "scored_month_count",
        "source_months",
        "scored_months",
        "missing_oof_months",
        "handoff_dir",
    ]
    lines.append(summary[[col for col in display_cols if col in summary.columns]].to_markdown(index=False) if not summary.empty else "_No handoffs._")
    lines.extend(["", "## Monthly Coverage", ""])
    cov_cols = [
        "handoff_dir",
        "month",
        "source_rows",
        "selected_rows",
        "representation_rows",
        "coverage_selected_rows",
        "coverage_all_rows",
    ]
    lines.append(coverage[[col for col in cov_cols if col in coverage.columns]].to_markdown(index=False) if not coverage.empty else "_No coverage rows._")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Month-forward representation predictions are expected for every source month after the first.",
            "Stability priors become learnable by train_meta only when a validation fold has training rows that already contain non-null prior features. That requires at least four source months under this expanding month-forward setup.",
        ]
    )
    path = out_dir / "cross_asset_representation_coverage.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_report(
    *,
    report_root: Path,
    out_dir: Path,
    handoff_path: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    if handoff_path is not None:
        pairs = [(handoff_path, predictions_path)]
    else:
        pairs = _find_pairs(report_root)
    summaries: list[dict[str, Any]] = []
    coverage_parts: list[pd.DataFrame] = []
    for handoff, pred in pairs:
        summary, coverage = _coverage_for_pair(handoff, pred)
        summaries.append(summary)
        coverage_parts.append(coverage)
    summary_df = pd.DataFrame(summaries)
    coverage_df = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "cross_asset_representation_coverage_summary.csv"
    coverage_path = out_dir / "cross_asset_representation_coverage_months.csv"
    summary_df.to_csv(summary_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    ready = int(summary_df["stability_context_learnable_in_month_forward_meta"].sum()) if not summary_df.empty else 0
    manifest = {
        "generated_by": "report_cross_asset_representation_coverage",
        "report_root": str(report_root),
        "scanned_handoffs": int(len(summary_df)),
        "ready_handoffs": int(ready),
        "status": "ready_handoff_found" if ready else "needs_more_source_months_or_oof_predictions",
        "outputs": {
            "summary": str(summary_path),
            "coverage": str(coverage_path),
            "json": str(out_dir / "cross_asset_representation_coverage.json"),
            "markdown": str(out_dir / "cross_asset_representation_coverage.md"),
        },
    }
    markdown = _write_markdown(out_dir, manifest, summary_df, coverage_df)
    manifest["outputs"]["markdown"] = str(markdown)
    (out_dir / "cross_asset_representation_coverage.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--handoff-path", type=Path, default=None)
    parser.add_argument("--predictions-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_report(
        report_root=args.report_root,
        out_dir=args.out_dir,
        handoff_path=args.handoff_path,
        predictions_path=args.predictions_path,
    )
    print(json.dumps(_json_safe({"event": "cross_asset_representation_coverage_done", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
