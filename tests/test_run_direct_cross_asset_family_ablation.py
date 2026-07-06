from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_direct_cross_asset_family_ablation import run


def _handoff() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    rows = []
    for month in pd.period_range("2026-01", "2026-04", freq="M"):
        for side in ("long", "short"):
            for source in ("mean_reversion", "breakout"):
                for i in range(160):
                    ctx = rng.normal()
                    base = rng.normal()
                    ev = 0.018 + 0.014 * ctx + 0.002 * base + rng.normal(scale=0.005)
                    rows.append(
                        {
                            "__ts__": month.to_timestamp() + pd.Timedelta(minutes=i),
                            "__symbol__": f"SYM{i % 5}/USD:USD",
                            "month": str(month),
                            "side_name": side,
                            "source_archetype": source,
                            "normalized_rank_score": base,
                            "calibrated_score": base + rng.normal(scale=0.1),
                            "ctx_pct_assets_up_1h": ctx,
                            "ctx_btc_ret_24h_pct": ctx * 0.5,
                            "ctx_xasset_btc_fund_z": -ctx,
                            "oofctx_gmm_prob_0": 1.0 / (1.0 + np.exp(-ctx)),
                            "oofctx_gmm_mahal_0": abs(ctx),
                            "oofctx_dae_b16_00": ctx,
                            "oofctx_dae_reconstruction_error": abs(ctx),
                            "oofctx_regime_centroid_similarity_train": ctx,
                            "xctx_latent_0": ctx,
                            "xctx_cluster_entropy": abs(ctx) / 3.0,
                            "xctx_ev_score_oof": ctx + rng.normal(scale=0.05) if str(month) != "2026-01" else np.nan,
                            "xctx_blend_score": ctx + base if str(month) != "2026-01" else np.nan,
                            "exec_ev_after_1pct_cost": ev,
                            "full_sl": int(ctx < -1.0),
                            "timeout": int(ctx > 1.3),
                            "clean_exec_proxy": int(ev > 0 and ctx >= -1.0 and ctx <= 1.3),
                        }
                    )
    return pd.DataFrame(rows)


def test_direct_cross_asset_family_ablation_outputs(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.parquet"
    output_dir = tmp_path / "out"
    _handoff().to_parquet(handoff_path, index=False)

    manifest = run(
        handoff_path=handoff_path,
        output_dir=output_dir,
        max_fit_rows=2_000,
        min_group_rows=50,
        seed=13,
    )

    assert manifest["baseline_variant"] == "m0_score_only"
    assert "f01_raw_breadth" in manifest["families"]
    assert "f11_xctx_scores" in manifest["families"]
    assert manifest["family_contract"]["f11_xctx_scores"]["added_feature_count"] >= 1

    aggregate = pd.read_csv(manifest["outputs"]["aggregate"])
    assert "m0_score_only" in set(aggregate["variant"])
    assert {"precision_positive_ev", "ev_weighted_precision", "full_sl_rate", "timeout_rate"}.issubset(
        aggregate.columns
    )

    accepted = pd.read_csv(manifest["outputs"]["accepted"])
    assert not accepted.empty
    assert (accepted["top_frac"] == 0.1).all()
    assert (accepted["delta_mean_ev_after_1pct"] > 0).all()
    assert (accepted["delta_precision_positive_ev"] >= 0).all()
