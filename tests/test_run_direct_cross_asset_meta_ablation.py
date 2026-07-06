from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_direct_cross_asset_meta_ablation import run


def _handoff() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for month in pd.period_range("2026-01", "2026-04", freq="M"):
        for side in ("long", "short"):
            for source in ("mean_reversion", "breakout"):
                for i in range(180):
                    ctx = rng.normal()
                    base = rng.normal()
                    ev = 0.02 + 0.012 * ctx + 0.003 * base + rng.normal(scale=0.006)
                    full_sl = int(ctx < -1.0)
                    timeout = int(ctx > 1.2)
                    rows.append(
                        {
                            "__ts__": month.to_timestamp() + pd.Timedelta(minutes=i),
                            "__symbol__": f"SYM{i % 7}/USD:USD",
                            "month": str(month),
                            "side_name": side,
                            "source_archetype": source,
                            "normalized_rank_score": base,
                            "calibrated_score": base + rng.normal(scale=0.1),
                            "ctx_btc_ret_24h_pct": ctx,
                            "oofctx_gmm_prob_0": 1.0 / (1.0 + np.exp(-ctx)),
                            "oofctx_dae_reconstruction_error": abs(ctx),
                            "xctx_latent_0": ctx,
                            "xctx_cluster_entropy": abs(ctx) / 3.0,
                            "xctx_ev_score_oof": ctx + rng.normal(scale=0.1) if str(month) != "2026-01" else np.nan,
                            "xctx_blend_score": ctx + base if str(month) != "2026-01" else np.nan,
                            "exec_ev_after_1pct_cost": ev,
                            "full_sl": full_sl,
                            "timeout": timeout,
                            "clean_exec_proxy": int(ev > 0 and not full_sl and not timeout),
                        }
                    )
    return pd.DataFrame(rows)


def test_direct_cross_asset_meta_ablation_outputs(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.parquet"
    output_dir = tmp_path / "out"
    _handoff().to_parquet(handoff_path, index=False)

    manifest = run(
        handoff_path=handoff_path,
        output_dir=output_dir,
        max_fit_rows=2_000,
        min_group_rows=50,
        seed=5,
    )

    assert manifest["target"] == "exec_ev_after_1pct_cost"
    assert "m0_score_only" in manifest["variants"]
    assert "m5_score_plus_all_context" in manifest["variants"]
    assert manifest["variant_contract"]["m2_score_plus_oof_aegmm"]["feature_count"] >= 3

    preds = pd.read_parquet(manifest["outputs"]["predictions"])
    first_month = sorted(preds["month"].unique())[0]
    assert preds.loc[preds["month"].eq(first_month), "score__m0_score_only"].isna().all()
    assert preds.loc[~preds["month"].eq(first_month), "score__m0_score_only"].notna().any()

    agg = pd.read_csv(manifest["outputs"]["aggregate"])
    assert {"precision_positive_ev", "ev_weighted_precision", "full_sl_rate", "timeout_rate"}.issubset(agg.columns)

    delta = pd.read_csv(manifest["outputs"]["aggregate_delta"])
    assert not delta.empty
    assert set(delta["variant"]).issubset(set(manifest["variants"]) - {"m0_score_only"})
