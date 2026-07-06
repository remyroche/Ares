from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_train_meta_composite_selector_hpo import run


def _predictions() -> pd.DataFrame:
    rng = np.random.default_rng(101)
    rows = []
    for month in pd.period_range("2026-01", "2026-04", freq="M"):
        for side in ("long", "short"):
            for source in ("long_dist", "short_bollinger"):
                for i in range(120):
                    x = rng.normal()
                    risk = max(rng.normal(), 0.0)
                    ev = 0.012 + 0.010 * x - 0.006 * risk
                    full_sl = int(risk > 1.0)
                    timeout = int(side == "long" and source == "long_dist" and x < -0.8)
                    s12 = x - 0.2 * risk
                    s16 = s12 - (0.4 * risk if side == "long" else 0.0)
                    s18 = s12 - (0.5 * risk if side == "long" else 0.0)
                    rows.append(
                        {
                            "month": str(month),
                            "side_name": side,
                            "source_archetype": source,
                            "exec_ev_after_1pct_cost": ev,
                            "full_sl": full_sl,
                            "timeout": timeout,
                            "clean_exec_proxy": int(ev > 0 and full_sl == 0 and timeout == 0),
                            "pred_ev": ev + rng.normal(scale=0.002),
                            "pred_full_sl": min(1.0, risk / 2.0),
                            "pred_timeout": 0.7 if timeout else 0.1,
                            "pred_clean": 0.8 if ev > 0 and full_sl == 0 else 0.2,
                            "score_s12_ev_clean_strong_risk": s12,
                            "score_s16_cell_prior_clean_risk_s12": s16,
                            "score_s18_long_cell_prior_ev_fullsl_s12": s18,
                        }
                    )
    return pd.DataFrame(rows)


def test_train_meta_composite_selector_hpo_outputs(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    _predictions().to_parquet(prediction_path, index=False)

    manifest = run(
        prediction_path=prediction_path,
        output_dir=tmp_path / "out",
        long_selectors=("s16_cell_prior_clean_risk_s12", "s18_long_cell_prior_ev_fullsl_s12"),
        min_group_rows=30,
        top_stage1_arms=2,
    )

    assert manifest["trial_count"] > 1
    assert manifest["best_arm"]
    assert manifest["recommended_arm"]
    assert "recommendation_contract" in manifest
    assert manifest["objective_contract"].startswith("precision-first")

    trials = pd.read_csv(manifest["outputs"]["trials"])
    assert {"objective", "long_selector", "full_sl_penalty", "timeout_penalty"}.issubset(trials.columns)
    assert "reference" in set(trials["stage"])
    assert "stage2_risk_penalty" in set(trials["stage"])

    aggregate = pd.read_csv(manifest["outputs"]["aggregate"])
    assert {"precision_positive_ev", "mean_ev_after_1pct", "full_sl_rate", "timeout_rate"}.issubset(
        aggregate.columns
    )

    cell_summary = pd.read_csv(manifest["outputs"]["cell_summary"])
    assert {"side_name", "source_archetype", "mean_ev_after_1pct"}.issubset(cell_summary.columns)
    assert Path(manifest["outputs"]["report"]).exists()
