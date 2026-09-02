from __future__ import annotations

import json
import subprocess
import sys
import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_c0_c1_agreement_tier import (
    TIER_OFFSET_BPS,
    UNPAIRED_ORDER_C0_THEN_C1,
    select_c0_c1_agreement_tiers,
)


def _scores(*, c0: list[tuple[float, float]], c1: list[tuple[float, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"S{idx}/USD:USD|long|2026-09-01T00:00:00Z" for idx in range(len(c0))]
    common = {
        "candidate_id": ids,
        "__symbol__": [f"S{idx}/USD:USD" for idx in range(len(c0))],
        "side_name": ["long"] * len(c0),
        "__decision_ts__": ["2026-09-01T00:00:00Z"] * len(c0),
    }
    def frame(values: list[tuple[float, float]]) -> pd.DataFrame:
        return pd.DataFrame({
            **common,
            "bcf_mc1_expected_bps": [v[0] for v in values],
            "current_mc1_expected_bps": [v[1] for v in values],
            "auction_priority_bps": [v[0] for v in values],
        })
    return frame(c0), frame(c1)


def test_agreement_tier_is_both_then_unpaired_by_raw_ev() -> None:
    # Rows 0, 1, 2 are respectively both, C1-only, and C0-only.  Raw C0-only
    # EV is highest, proving the tier—not accidental raw score order—wins.
    c0, c1 = _scores(
        c0=[(80, 80), (20, 80), (300, 300), (10, 10)],
        c1=[(90, 90), (110, 110), (20, 110), (10, 10)],
    )
    result = select_c0_c1_agreement_tiers(c0_scores=c0, c1_scores=c1)
    # Both always leads.  Among unpaired rows C0-only raw EV 300 outranks
    # C1-only raw EV 110; family must not create arbitrary precedence.
    assert result["admission_provenance"].tolist() == [
        "both_admitted", "c0_only", "c1_only",
    ]
    assert result["score_coordinate_source"].tolist() == ["C0", "C0", "C1"]
    assert result["agreement_tier"].tolist() == [2, 0, 1]
    assert result["portfolio_tier"].tolist() == [1, 0, 0]
    assert result.loc[0, "auction_priority_bps"] == 80.0
    assert result.loc[1, "auction_priority_bps"] == 300.0
    assert result.loc[2, "auction_priority_bps"] == 110.0
    assert result.loc[2, "c0_bcf_mc1_expected_bps"] == 20.0
    assert result.loc[2, "c1_bcf_mc1_expected_bps"] == 110.0
    assert result.loc[0, "portfolio_order_priority_bps"] == 80.0 + TIER_OFFSET_BPS
    assert result.loc[1, "portfolio_order_priority_bps"] == 300.0
    assert result.loc[2, "portfolio_order_priority_bps"] == 110.0


def test_canonical_order_is_both_then_c0_then_c1() -> None:
    c0, c1 = _scores(
        c0=[(80, 80), (20, 80), (300, 300)],
        c1=[(90, 90), (900, 900), (20, 110)],
    )
    result = select_c0_c1_agreement_tiers(
        c0_scores=c0,
        c1_scores=c1,
        unpaired_order=UNPAIRED_ORDER_C0_THEN_C1,
    )
    # C0-only must lead the much larger C1-only EV; C1 is explicit gap fill.
    assert result["admission_provenance"].tolist() == [
        "both_admitted", "c0_only", "c1_only",
    ]
    assert result["portfolio_tier"].tolist() == [2, 1, 0]
    assert result["portfolio_order_priority_bps"].tolist() == [
        10080.0 + TIER_OFFSET_BPS, 300.0 + TIER_OFFSET_BPS, 900.0,
    ]


def test_rejects_identity_mismatch_and_outcome_fields() -> None:
    c0, c1 = _scores(c0=[(80, 80)], c1=[(80, 80)])
    c1.loc[0, "candidate_id"] = "different"
    with pytest.raises(ValueError, match="identities"):
        select_c0_c1_agreement_tiers(c0_scores=c0, c1_scores=c1)
    c0, c1 = _scores(c0=[(80, 80)], c1=[(80, 80)])
    c0["policy_net_bps"] = 999.0
    with pytest.raises(ValueError, match="outcome-derived"):
        select_c0_c1_agreement_tiers(c0_scores=c0, c1_scores=c1)


def test_no_order_assembler_preserves_raw_ev_and_provenance(tmp_path) -> None:
    c0, c1 = _scores(
        c0=[(80, 80), (20, 80), (300, 300)],
        c1=[(90, 90), (110, 110), (20, 110)],
    )
    c0_path = tmp_path / "c0.parquet"
    c1_path = tmp_path / "c1.parquet"
    out = tmp_path / "out"
    c0.to_parquet(c0_path, index=False)
    c1.to_parquet(c1_path, index=False)
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "assemble_p8u_c0_c1_agreement_tier.py"
    subprocess.run(
        [sys.executable, str(runner), "--c0-scores", str(c0_path), "--c1-scores", str(c1_path), "--out", str(out)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert "no labels, policy, portfolio, exchange I/O, or order submission" in manifest["scope"]
    result = pd.read_parquet(out / "agreement_tier_target_free_scores.parquet")
    assert result["admission_provenance"].tolist() == ["both_admitted", "c0_only", "c1_only"]
    assert result.loc[2, "auction_priority_bps"] == 110.0
    assert result.loc[2, "portfolio_order_priority_bps"] == 110.0
