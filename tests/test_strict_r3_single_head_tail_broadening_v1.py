from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "single_head_tail_broadening", ROOT / "scripts" / "run_strict_r3_single_head_tail_broadening_v1.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-01-01T00:00:00Z"] * 4 + ["2026-01-01T01:00:00Z"] * 4, utc=True)
    return pd.DataFrame({
        "candidate_id": list("abcdefgh"), "__decision_ts__": ts, "side_name": "long",
        "policy_net_bps": [200.0, 100.0, -50.0, -100.0, 150.0, 50.0, -25.0, -75.0],
        "score": [4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
    })


def test_timestamp_topk_is_not_global_tail() -> None:
    result = MODULE._topk(_frame(), "score", 1)
    assert result["rows"] == 2.0
    assert result["ev"] == 175.0


def test_dtp_weights_the_top_rank_more_than_second() -> None:
    result = MODULE._dtp(_frame(), "score", 2)
    assert result["value"] > 125.0


def test_guardrails_reject_top_one_regression() -> None:
    control = {"top1_ev": 100.0, "top2_ev": 100.0, "dtp5_value": 100.0, "dtp5_q10_week": 100.0, "dtp5_worst_month": 10.0}
    candidate = {"top1_ev": 96.0, "top2_ev": 100.0, "dtp5_value": 100.0, "dtp5_q10_week": 100.0, "dtp5_worst_month": 10.0}
    passed, failures = MODULE._guardrails(candidate, control)
    assert not passed and "top-1" in failures


def test_final_score_schema_is_outcome_free() -> None:
    forbidden = {"policy_net_bps", "label_available_ts", "policy_ordinal_base_grade", "policy_ordinal_base_valid"}
    schema = {"candidate_id", "__decision_ts__", "side_name", "head_score", "held_month"}
    assert not forbidden.intersection(schema)
