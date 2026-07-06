from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_promoted_cross_asset_cell_effects import run_audit  # noqa: E402


def _write_smoke(path: Path, rows: list[dict], selector: str) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path / "s52_train_meta_regime_handoff_smoke_predictions.parquet", index=False)
    (path / "manifest.json").write_text(json.dumps({"best_selector": {"selector": selector}}))


def _rows(*, promoted: bool, damaged: bool = False) -> list[dict]:
    rows: list[dict] = []
    for month_idx, month in enumerate(("2026-05", "2026-06")):
        for idx in range(40):
            clean = idx < 22
            symbol = f"SYM{idx % 8}"
            score_base = float(idx) if clean else float(100 - idx)
            score_promoted = float(100 - idx) if clean else float(idx)
            if damaged and promoted:
                score_promoted = float(idx) if clean else float(100 - idx)
            row = {
                "__ts__": f"{month}-{(idx % 20) + 1:02d}T{idx % 24:02d}:00:00Z",
                "__symbol__": symbol,
                "side_name": "long",
                "month": month,
                "source_semantic_family": "mixed",
                "score_base": score_base,
                "score_meta_clean_exec": score_promoted if promoted else score_base,
                "exec_margin": 0.010 + 0.001 * month_idx if clean else -0.006,
                "clean_exec": 1.0 if clean else 0.0,
                "full_path_bad_mae_1r": 0.0 if clean else 1.0,
                "timeout": 0.0,
                "dirty_positive": 0.0 if clean else 1.0,
                "mfe_before_mae_1r": 1.0 if clean else 0.0,
                "mae_before_mfe_1r": 0.0 if clean else 1.0,
                "underwater_bars_before_mfe_1r": 2.0 if clean else 14.0,
            }
            rows.append(row)
    return rows


def test_cell_effect_audit_passes_supported_cell_lift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    promoted = tmp_path / "promoted"
    _write_smoke(baseline, _rows(promoted=False), "base_score")
    _write_smoke(promoted, _rows(promoted=True), "meta_clean_exec")

    manifest = run_audit(
        baseline_smoke_dir=baseline,
        promoted_smoke_dir=promoted,
        out_dir=tmp_path / "out",
        min_valid_rows=30,
        min_months=2,
        min_clean_rows=5,
        min_positive_rows=5,
    )

    assert manifest["summary"]["status"] == "pass"
    assert manifest["summary"]["beneficial_supported_cells"] >= 1
    effects = pd.read_csv(manifest["outputs"]["cell_effects"])
    keep10 = effects[effects["keep_frac"].eq(0.10)].iloc[0]
    assert keep10["support_pass"]
    assert keep10["delta_full_path_bad_mae"] < 0.0


def test_cell_effect_audit_flags_supported_degradation(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    promoted = tmp_path / "promoted"
    baseline_rows = _rows(promoted=True)
    for row in baseline_rows:
        row["score_base"] = row["score_meta_clean_exec"]
    _write_smoke(baseline, baseline_rows, "base_score")
    _write_smoke(promoted, _rows(promoted=True, damaged=True), "meta_clean_exec")

    manifest = run_audit(
        baseline_smoke_dir=baseline,
        promoted_smoke_dir=promoted,
        out_dir=tmp_path / "out",
        min_valid_rows=30,
        min_months=2,
        min_clean_rows=5,
        min_positive_rows=5,
    )

    assert manifest["summary"]["status"] == "diagnostic_or_blocked"
    assert manifest["summary"]["catastrophic_supported_degradation_cells"] >= 1
