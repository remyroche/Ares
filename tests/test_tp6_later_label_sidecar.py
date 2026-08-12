from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_later_20260815_v1/assembled_exact_labels"


def test_later_sidecar_preserves_full_candidate_identity_and_explicit_invalids():
    frames = [
        pd.read_parquet(ARTIFACT / "parts/month=2026-07" / f"side={side}.parquet")
        for side in ("long", "short")
    ]
    x = pd.concat(frames, ignore_index=True)
    assert len(x) == 14_400
    assert x.candidate_id.is_unique
    assert set(x.side_name) == {"long", "short"}
    assert x.target_invalid.astype(bool).sum() > 0
    assert x.loc[x.target_invalid.astype(bool), "t4_tp6_sl4_net_bps"].isna().all()
    valid = ~x.target_invalid.astype(bool)
    assert valid.mean() >= 0.90
    assert np.allclose(
        x.loc[valid, "t4_tp6_sl4_gross_bps"].to_numpy(float) - 100.0,
        x.loc[valid, "t4_tp6_sl4_net_bps"].to_numpy(float),
        atol=2e-3,
        rtol=0.0,
    )


def test_later_sidecar_contract_is_causal_and_exact():
    manifest = (ARTIFACT / "run_manifest.json").read_text()
    assert '"status": "complete"' in manifest
    assert "TP +6 ATR / SL -4 ATR / H12" in manifest
    assert "exact next-minute open" in manifest
    assert "gross_bps - 100 exactly once" in manifest
    coverage = pd.read_parquet(ARTIFACT / "coverage.parquet")
    assert set(coverage.side) == {"long", "short"}
    assert (coverage.valid_fraction >= 0.90).all()
