from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_promoted_cross_asset_month_flip_attribution import run_audit  # noqa: E402


def _write_smoke(path: Path, rows: list[dict], selector: str) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path / "s52_train_meta_regime_handoff_smoke_predictions.parquet", index=False)
    (path / "manifest.json").write_text(json.dumps({"best_selector": {"selector": selector}}))


def _rows(*, promoted: bool) -> list[dict]:
    rows: list[dict] = []
    for month in ("2026-05", "2026-06"):
        for family in ("flip_cell", "stable_cell"):
            for idx in range(60):
                clean = idx < 25
                if family == "stable_cell":
                    base_score = float(100 - idx) if clean else float(idx)
                    promoted_score = base_score
                elif month == "2026-05":
                    base_score = float(100 - idx) if not clean else float(idx)
                    promoted_score = float(100 - idx) if clean else float(idx)
                else:
                    base_score = float(100 - idx) if clean else float(idx)
                    promoted_score = float(100 - idx) if not clean else float(idx)
                rows.append(
                    {
                        "__ts__": f"{month}-{(idx % 25) + 1:02d}T{idx % 24:02d}:00:00Z",
                        "__symbol__": f"{family.upper()}_SYM{idx % 8}",
                        "side_name": "short",
                        "month": month,
                        "source_semantic_family": family,
                        "score_base": base_score,
                        "score_meta_clean_exec": promoted_score if promoted else base_score,
                        "ev_after_1pct": 0.010 if clean else -0.012,
                        "exec_margin": 0.013 if clean else -0.009,
                        "clean_exec": 1.0 if clean else 0.0,
                        "dirty_positive": 0.0 if clean else 1.0,
                        "full_path_bad_mae_1r": 0.0 if clean else 1.0,
                        "timeout": 0.0 if clean else 0.5,
                        "mfe_before_mae_1r": 1.0 if clean else 0.0,
                        "mae_before_mfe_1r": 0.0 if clean else 1.0,
                        "underwater_bars_before_mfe_1r": 3.0 if clean else 18.0,
                    }
                )
    return rows


def test_month_flip_attribution_detects_positive_to_negative_cell(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    promoted = tmp_path / "promoted"
    _write_smoke(baseline, _rows(promoted=False), "base_score")
    _write_smoke(promoted, _rows(promoted=True), "meta_clean_exec")

    manifest = run_audit(
        baseline_smoke_dir=baseline,
        promoted_smoke_dir=promoted,
        out_dir=tmp_path / "out",
        min_valid_rows=20,
        min_clean_rows=5,
        min_positive_rows=5,
        max_week_share=1.0,
    )

    assert manifest["summary"]["status"] == "month_instability_detected"
    flips = pd.read_csv(manifest["outputs"]["month_flips"])
    bad = flips[
        flips["source_semantic_family"].eq("flip_cell")
        & flips["keep_frac"].eq(0.10)
        & flips["flip_type"].eq("positive_to_negative")
    ]
    assert not bad.empty
    row = bad.iloc[0]
    assert float(row["history_effect_value"]) > 0.0
    assert float(row["current_effect_value"]) < 0.0
    assert Path(manifest["outputs"]["markdown"]).exists()
