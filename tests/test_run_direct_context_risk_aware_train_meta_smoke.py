from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_direct_context_risk_aware_train_meta_smoke import run


def _handoff() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    rows = []
    for month in pd.period_range("2026-01", "2026-04", freq="M"):
        for side in ("long", "short"):
            for source in ("mean_reversion", "breakout"):
                for i in range(180):
                    x = rng.normal()
                    risk = rng.normal()
                    ev = 0.02 + 0.012 * x - 0.008 * max(risk, 0) + rng.normal(scale=0.004)
                    full_sl = int(risk > 1.0)
                    timeout = int(x > 1.2)
                    clean = int(ev > 0 and full_sl == 0 and timeout == 0)
                    rows.append(
                        {
                            "__ts__": month.to_timestamp() + pd.Timedelta(minutes=i),
                            "__symbol__": f"SYM{i % 6}/USD:USD",
                            "month": str(month),
                            "side_name": side,
                            "source_archetype": source,
                            "normalized_rank_score": x,
                            "xctx_ev_score_oof": x + rng.normal(scale=0.1),
                            "xctx_cluster_entropy": abs(risk),
                            "oofctx_dae_reconstruction_error": max(risk, 0),
                            "exec_ev_after_1pct_cost": ev,
                            "full_sl": full_sl,
                            "timeout": timeout,
                            "clean_exec_proxy": clean,
                        }
                    )
    return pd.DataFrame(rows)


def test_risk_aware_train_meta_smoke_outputs(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.parquet"
    manifest_path = tmp_path / "features.json"
    output_dir = tmp_path / "out"
    _handoff().to_parquet(handoff_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "feature_columns": [
                    "normalized_rank_score",
                    "xctx_ev_score_oof",
                    "xctx_cluster_entropy",
                    "oofctx_dae_reconstruction_error",
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = run(
        handoff_path=handoff_path,
        feature_manifest_path=manifest_path,
        output_dir=output_dir,
        max_fit_rows=2_000,
        min_group_rows=50,
        seed=3,
    )

    assert manifest["feature_count"] == 4
    assert "s0_ev_only" in manifest["selectors"]
    assert "s4_ev_clean_minus_risk" in manifest["selectors"]
    assert "s12_ev_clean_strong_risk" in manifest["selectors"]
    assert "s14_cell_prior_fullsl_s12" in manifest["selectors"]
    assert "s17_cell_prior_ev_fullsl_s12" in manifest["selectors"]
    assert "s18_long_cell_prior_ev_fullsl_s12" in manifest["selectors"]
    assert "s19_long_s16_short_s12" in manifest["selectors"]

    preds = pd.read_parquet(manifest["outputs"]["predictions"])
    first_month = sorted(preds["month"].unique())[0]
    assert preds.loc[preds["month"].eq(first_month), "pred_ev"].isna().all()
    assert preds.loc[~preds["month"].eq(first_month), "pred_ev"].notna().any()
    assert {
        "pred_full_sl",
        "pred_timeout",
        "pred_clean",
        "score_s4_ev_clean_minus_risk",
        "score_s12_ev_clean_strong_risk",
        "score_s14_cell_prior_fullsl_s12",
        "score_s17_cell_prior_ev_fullsl_s12",
        "score_s18_long_cell_prior_ev_fullsl_s12",
        "score_s19_long_s16_short_s12",
        "prior_cell_excess_full_sl",
        "prior_cell_ev_shortfall",
        "prior_cell_ev_premium",
    }.issubset(preds.columns)
    assert preds.loc[~preds["month"].eq(first_month), "score_s14_cell_prior_fullsl_s12"].notna().any()
    assert preds.loc[~preds["month"].eq(first_month), "score_s17_cell_prior_ev_fullsl_s12"].notna().any()

    aggregate = pd.read_csv(manifest["outputs"]["aggregate"])
    assert {
        "precision_positive_ev",
        "full_sl_rate",
        "timeout_rate",
        "pred_timeout_mean",
        "selected_share",
        "guard_pass_rate",
    }.issubset(aggregate.columns)

    delta = pd.read_csv(manifest["outputs"]["aggregate_delta"])
    assert not delta.empty
    assert "delta_timeout_rate" in delta.columns

    cell_delta_summary = pd.read_csv(manifest["outputs"]["cell_delta_summary"])
    assert {"better_ev", "lower_full_sl", "lower_timeout"}.issubset(cell_delta_summary.columns)

    worst_cell_tradeoffs = pd.read_csv(manifest["outputs"]["worst_cell_tradeoffs"])
    assert {"selector", "side_name", "source_archetype", "delta_full_sl_rate"}.issubset(
        worst_cell_tradeoffs.columns
    )

    cell_score_events = pd.read_csv(manifest["outputs"]["cell_score_events"])
    assert not cell_score_events.empty
    assert cell_score_events["status"].eq("fit_prior_cell_adjustment").any()
    assert cell_score_events["status"].eq("fit_composite_score").any()
    assert {"ev_shortfall_scale", "ev_premium_scale"}.issubset(cell_score_events.columns)
