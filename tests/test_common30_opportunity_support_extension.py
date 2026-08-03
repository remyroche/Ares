from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_common30_opportunity_support_extension.py"
)
SPEC = importlib.util.spec_from_file_location("common30_support", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sources(hours: int = 800) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_rows = []
    label_rows = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for hour in range(hours):
        stamp = start + pd.Timedelta(hours=hour)
        for side in ("long", "short"):
            candidate_id = f"{hour}-{side}"
            decision = stamp + pd.Timedelta(hours=1)
            resolution = decision + pd.Timedelta(hours=12)
            score = np.sin(hour / 30) * 0.01 + (0.001 if side == "long" else 0)
            gross = score + np.cos(hour / 20) * 0.005
            cost = 0.01
            identity = {
                "candidate_id": candidate_id,
                "__ts__": stamp,
                "__symbol__": f"asset-{hour % 3}",
                "side_name": side,
                "execution_decision_utc": decision,
                "execution_label_end_utc": resolution,
            }
            oof_rows.append(
                {
                    **identity,
                    "candidate_month": stamp.strftime("%Y-%m"),
                    "historical_base_soft_oof": score,
                    "historical_direct_ev_oof": score,
                    "direct_oof_fold_start_utc": start,
                    "direct_oof_train_cutoff_utc": start - pd.Timedelta(hours=1),
                }
            )
            label_rows.append(
                {
                    **identity,
                    "execution_gross_ev_12h": gross,
                    "execution_cost_return": cost,
                    "execution_net_ev_12h": gross - cost,
                    "execution_exit_reason": "timeout",
                    "execution_exit_minute": 720,
                    "execution_mfe_return_12h": gross + 0.02,
                    "execution_mae_return_12h": -0.01,
                }
            )
    return pd.DataFrame(oof_rows), pd.DataFrame(label_rows)


def test_prepare_exact_candidates_preserves_identity_and_accounting() -> None:
    oof, labels = _sources()
    oof["execution_net_ev_12h"] = 999.0
    result, audit = MODULE.prepare_exact_candidates(
        oof,
        labels,
        minimum_reference_rows=100,
    )
    assert len(result) == len(oof)
    assert result.candidate_id.nunique() == len(oof)
    assert (
        result.execution_gross_ev_12h
        - result.execution_cost_return
        - result.execution_net_ev_12h
    ).abs().max() < 1e-12
    assert not result.execution_net_ev_12h.eq(999.0).any()
    assert len(audit) > 0
    assert result.mapped_eligible.any()


def test_prepare_rejects_label_timing_mismatch() -> None:
    oof, labels = _sources(20)
    labels.loc[0, "execution_label_end_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="timing mismatch"):
        MODULE.prepare_exact_candidates(
            oof,
            labels,
            minimum_reference_rows=10,
        )


def test_prepare_rejects_model_cutoff_after_signal() -> None:
    oof, labels = _sources(20)
    oof.loc[0, "direct_oof_train_cutoff_utc"] = (
        oof.loc[0, "__ts__"] + pd.Timedelta(minutes=1)
    )
    with pytest.raises(ValueError, match="cutoff must not follow signal"):
        MODULE.prepare_exact_candidates(
            oof,
            labels,
            minimum_reference_rows=10,
        )


def test_lineage_is_explicitly_separate() -> None:
    assert "common30" in MODULE.LINEAGE
    assert MODULE.LINEAGE not in {
        "historical_2025_raw_alpha",
        "current_2026_execution_ev",
    }
