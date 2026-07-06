from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_direct_context_weak_cell_drivers import run


def test_weak_cell_driver_report_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(41)
    rows = []
    preds = []
    for month in ("2026-03", "2026-04"):
        for i in range(240):
            risk_pressure = rng.normal()
            score = risk_pressure + rng.normal(scale=0.3)
            full_sl = int(risk_pressure > 0.7)
            timeout = int(risk_pressure < -1.2)
            ev = 0.01 - 0.006 * full_sl + 0.002 * score + rng.normal(scale=0.003)
            rows.append(
                {
                    "month": month,
                    "side_name": "long",
                    "source_archetype": "long_bars",
                    "risk_pressure": risk_pressure,
                    "xctx_latent_0": -risk_pressure,
                    "normalized_rank_score": score,
                    "exec_ev_after_1pct_cost": ev,
                    "full_sl": full_sl,
                    "timeout": timeout,
                    "clean_exec_proxy": int(ev > 0 and not full_sl and not timeout),
                }
            )
            preds.append(
                {
                    "score_s12_ev_clean_strong_risk": score,
                    "pred_ev": ev,
                    "pred_full_sl": max(0.0, min(1.0, 0.5 + 0.25 * risk_pressure)),
                    "pred_timeout": max(0.0, min(1.0, 0.2 - 0.1 * risk_pressure)),
                    "pred_clean": 0.5,
                }
            )
    handoff = pd.DataFrame(rows)
    predictions = pd.DataFrame(preds)
    handoff_path = tmp_path / "handoff.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    manifest_path = tmp_path / "features.json"
    worst_path = tmp_path / "worst.csv"
    accepted_path = tmp_path / "accepted.csv"
    output_dir = tmp_path / "out"
    handoff.to_parquet(handoff_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "feature_columns": ["normalized_rank_score", "risk_pressure", "xctx_latent_0"],
                "families": {
                    "f00_score_only": ["normalized_rank_score"],
                    "f10_xctx_latent": ["normalized_rank_score", "xctx_latent_0"],
                    "f99_risk_probe": ["normalized_rank_score", "risk_pressure"],
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "selector": "s12_ev_clean_strong_risk",
                "month": "2026-04",
                "side_name": "long",
                "source_archetype": "long_bars",
                "rows": 240,
                "selected_rows": 24,
                "delta_mean_ev_after_1pct": -0.001,
                "delta_precision_positive_ev": -0.01,
                "delta_full_sl_rate": 0.20,
                "delta_timeout_rate": -0.05,
            }
        ]
    ).to_csv(worst_path, index=False)
    pd.DataFrame(
        [
            {
                "month": "2026-04",
                "side_name": "long",
                "source_archetype": "long_bars",
                "variant": "f10_xctx_latent",
                "delta_mean_ev_after_1pct": 0.01,
                "family_rank_in_cell": 1,
            }
        ]
    ).to_csv(accepted_path, index=False)

    manifest = run(
        handoff_path=handoff_path,
        feature_manifest_path=manifest_path,
        predictions_path=predictions_path,
        worst_cells_path=worst_path,
        accepted_cells_path=accepted_path,
        output_dir=output_dir,
        selectors=("s12_ev_clean_strong_risk",),
        top_frac=0.10,
        max_cells=5,
        min_delta_full_sl=0.02,
    )

    assert manifest["weak_cell_count"] == 1
    drivers = pd.read_csv(manifest["outputs"]["feature_drivers"])
    assert {"feature", "family", "selected_z_delta", "corr_feature_full_sl"}.issubset(drivers.columns)
    assert "risk_pressure" in set(drivers["feature"])

    family = pd.read_csv(manifest["outputs"]["family_summary"])
    assert not family.empty
    assert {"family", "mean_abs_selected_z_delta"}.issubset(family.columns)

    accepted = pd.read_csv(manifest["outputs"]["accepted_context"])
    assert accepted["variant"].eq("f10_xctx_latent").any()

    report = Path(manifest["outputs"]["report"])
    assert report.exists()
    assert "Diagnostic only" in report.read_text()
