from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cross_asset_stability_prior_persistence import run_audit  # noqa: E402


def _row(month: str, family: str, keep: float, effect: float, ev: float) -> dict:
    return {
        "month": month,
        "keep_frac": keep,
        "side_name": "short",
        "source_semantic_family": family,
        "rows": 100,
        "clean_rows": 40,
        "positive_exec_rows": 45,
        "support_pass": True,
        "effect_value_score": effect,
        "delta_ev_after_1pct": ev,
        "delta_exec_margin": ev + 0.001,
        "delta_clean_exec_precision": 0.15 if effect > 0 else -0.15,
        "delta_full_path_bad_mae": -0.10 if effect > 0 else 0.20,
        "delta_timeout": 0.0,
        "delta_mfe_before_mae": 0.10 if effect > 0 else -0.10,
        "delta_mae_before_mfe": -0.10 if effect > 0 else 0.10,
        "delta_cell_oracle_overlap": 0.05 if effect > 0 else -0.05,
        "promoted_beneficial": effect > 0.35,
        "promoted_damaged": effect < -0.35,
    }


def test_stability_prior_persistence_reports_false_positive(tmp_path: Path) -> None:
    rows = []
    for keep in (0.10, 0.20, 0.30):
        rows.append(_row("2026-05", "stable_good", keep, 3.0, 0.008))
        rows.append(_row("2026-06", "stable_good", keep, 2.5, 0.007))
        rows.append(_row("2026-05", "flips_bad", keep, 3.0, 0.008))
        rows.append(_row("2026-06", "flips_bad", keep, -4.0, -0.010))
    month_cells = tmp_path / "month_cells.csv"
    pd.DataFrame(rows).to_csv(month_cells, index=False)

    manifest = run_audit(month_cells_path=month_cells, out_dir=tmp_path / "out")

    assert manifest["evaluated_prior_rows"] == 6
    summary = pd.read_csv(manifest["outputs"]["summary"])
    keep10 = summary[summary["keep_frac"].eq(0.10)].iloc[0]
    assert int(keep10["positive_to_negative_cells"]) == 1
    assert float(keep10["prior_positive_precision"]) == 0.5
    worst = pd.read_csv(manifest["outputs"]["worst"])
    assert "flips_bad" in set(worst["source_semantic_family"])
    assert Path(manifest["outputs"]["markdown"]).exists()
