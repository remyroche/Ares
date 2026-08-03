from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_causal_mapping_ic_ev_waterfall.py"
)
SPEC = importlib.util.spec_from_file_location("causal_map_waterfall", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in ("2026-05", "2026-06", "2026-07"):
        for index in range(4):
            signal = pd.Timestamp(f"{month}-07T0{index}:00:00Z")
            decision = signal + pd.Timedelta(hours=1)
            gross = index / 100.0
            rows.append(
                {
                    "candidate_id": f"A{index}/USD:USD|{signal.strftime('%Y-%m-%dT%H:%M:%SZ')}|1h|long",
                    "__symbol__": f"A{index}/USD:USD",
                    "side_name": "long" if index % 2 == 0 else "short",
                    "__ts__": signal,
                    "execution_decision_utc": decision,
                    "execution_label_end_utc": decision + pd.Timedelta(hours=12),
                    "execution_gross_ev_12h": gross,
                    "execution_cost_return": 0.01,
                    "execution_net_ev_12h": gross - 0.01,
                    "execution_mfe_return_12h": gross + 0.02,
                    "execution_mae_return_12h": -0.02,
                    "execution_exit_class": "timeout",
                    "catboost__residual__without_hpo__all_features": index / 100.0,
                    "causal_recent_isotonic_ev": index / 200.0,
                    "causal_recent_side_isotonic_ev": index / 300.0,
                    "causal_recent_side_isotonic_ev__is_oof": True,
                    "mapped_eligible": True,
                    "evaluation_origin": "historical_outer_oof",
                }
            )
    return pd.DataFrame(rows)


def test_build_uses_same_strict_oof_rows_and_both_exact_maps() -> None:
    frame = MODULE.build_frame(_source(), expected_rows=12)
    assert len(frame) == 12
    assert set(MODULE.score_columns(frame)) == {
        "score_raw_execution_ev",
        "score_causal_global_21d_ev",
        "score_causal_side_21d_ev",
    }


def test_build_rejects_non_oof_or_wrong_horizon() -> None:
    source = _source()
    source.loc[0, "evaluation_origin"] = "frozen_forward"
    with pytest.raises(ValueError, match="strict outer OOF"):
        MODULE.build_frame(source, expected_rows=12)

    source = _source()
    source.loc[0, "causal_recent_side_isotonic_ev__is_oof"] = False
    with pytest.raises(ValueError, match="strict mapped rows"):
        MODULE.build_frame(source, expected_rows=12)

    source = _source()
    source.loc[0, "execution_label_end_utc"] += pd.Timedelta(hours=12)
    with pytest.raises(ValueError, match="exact 12h"):
        MODULE.build_frame(source, expected_rows=12)
