from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE = Path(__file__).resolve().parents[1] / "scripts/replay_c0_c1_agreement_tier_exact1m.py"
SPEC = importlib.util.spec_from_file_location("tiers", MODULE)
assert SPEC and SPEC.loader
tiers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tiers)


def _row(candidate_id: str, decision: str, priority: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "decision_timestamp": pd.Timestamp(decision, tz="UTC"),
        "timestamp": pd.Timestamp(decision, tz="UTC") + pd.Timedelta(minutes=5),
        "symbol": candidate_id.split("|")[0],
        "side": "long",
        "portfolio_priority_adjustment": priority,
    }


def _target_free(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"candidate_id": ids, "timestamp": [pd.Timestamp("2026-05-01T00:00Z")] * len(ids)})


def test_target_free_agreement_tiers_are_causal_and_disjoint() -> None:
    c0 = pd.DataFrame([_row("A|long|0", "2026-05-01T00:00", 80), _row("C|long|0", "2026-05-01T00:00", 120)])
    c1 = pd.DataFrame([_row("A|long|0", "2026-05-01T00:00", 90), _row("B|long|0", "2026-05-01T00:00", 110)])
    route, audit = tiers.select_c0_c1_agreement_tiers(
        c0_target_free=_target_free(["A|long|0", "C|long|0"]),
        c1_target_free=_target_free(["A|long|0", "B|long|0"]),
        c0_exact_candidates=c0,
        c1_exact_candidates=c1,
    )
    assert route.set_index("candidate_id")["agreement_tier"].to_dict() == {"A|long|0": 2, "B|long|0": 1, "C|long|0": 0}
    assert route.set_index("candidate_id").loc["A|long|0", "raw_bcf_mc1_priority_bps"] == 80
    assert audit["tier_both_evaluable_rows"] == 1
    assert audit["tier_c1_only_evaluable_rows"] == 1
    assert audit["tier_c0_only_evaluable_rows"] == 1


def test_rejects_unadmitted_exact_identity() -> None:
    c0 = pd.DataFrame([_row("A|long|0", "2026-05-01T00:00", 80)])
    try:
        tiers.select_c0_c1_agreement_tiers(
            c0_target_free=_target_free(["B|long|0"]),
            c1_target_free=_target_free([]),
            c0_exact_candidates=c0,
            c1_exact_candidates=pd.DataFrame(columns=c0.columns),
        )
    except AssertionError as exc:
        assert "subset" in str(exc)
    else:
        raise AssertionError("expected target-free identity guard")
