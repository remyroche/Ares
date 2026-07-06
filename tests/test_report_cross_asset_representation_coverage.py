from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_cross_asset_representation_coverage import run_report  # noqa: E402


def _write_pair(root: Path, months: list[str], pred_months: list[str]) -> tuple[Path, Path]:
    rows = []
    preds = []
    for month in months:
        for idx in range(5):
            row = {
                "__ts__": f"{month}-{idx + 1:02d}T00:00:00Z",
                "__symbol__": f"SYM{idx}",
                "side_name": "short",
                "month": month,
                "selected_top10": True,
            }
            rows.append(row)
            if month in pred_months:
                preds.append(row | {"cross_lgbm_bad_mae_score": 0.2})
    handoff_dir = root / "handoff"
    pred_dir = handoff_dir / "cross_asset_archetype_representation_v1"
    pred_dir.mkdir(parents=True)
    handoff_path = handoff_dir / "train_meta_regime_handoff.parquet"
    pred_path = pred_dir / "cross_asset_representation_v1_predictions.parquet"
    pd.DataFrame(rows).to_parquet(handoff_path, index=False)
    pd.DataFrame(preds).to_parquet(pred_path, index=False)
    return handoff_path, pred_path


def test_cross_asset_coverage_requires_four_source_months(tmp_path: Path) -> None:
    handoff_path, pred_path = _write_pair(tmp_path, ["2026-03", "2026-04", "2026-05", "2026-06"], ["2026-04", "2026-05", "2026-06"])

    manifest = run_report(
        report_root=tmp_path,
        out_dir=tmp_path / "out",
        handoff_path=handoff_path,
        predictions_path=pred_path,
    )

    summary = pd.read_csv(manifest["outputs"]["summary"])
    assert bool(summary["stability_context_learnable_in_month_forward_meta"].iloc[0]) is True
    coverage = pd.read_csv(manifest["outputs"]["coverage"])
    assert coverage.loc[coverage["month"].eq("2026-03"), "coverage_selected_rows"].iloc[0] == 0.0
    assert coverage.loc[coverage["month"].eq("2026-06"), "coverage_selected_rows"].iloc[0] == 1.0


def test_cross_asset_coverage_blocks_three_source_months(tmp_path: Path) -> None:
    handoff_path, pred_path = _write_pair(tmp_path, ["2026-04", "2026-05", "2026-06"], ["2026-05", "2026-06"])

    manifest = run_report(
        report_root=tmp_path,
        out_dir=tmp_path / "out",
        handoff_path=handoff_path,
        predictions_path=pred_path,
    )

    summary = pd.read_csv(manifest["outputs"]["summary"])
    assert bool(summary["stability_context_learnable_in_month_forward_meta"].iloc[0]) is False
    assert summary["status"].iloc[0] == "needs_more_source_months_or_oof_predictions"
