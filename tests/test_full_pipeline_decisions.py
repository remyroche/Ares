from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS_PATH = ROOT / "config" / "full_pipeline_decisions_20260724.json"


def _decisions() -> dict:
    return json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))


def test_all_blocking_decisions_are_locked() -> None:
    payload = _decisions()

    assert payload["status"] == "LOCKED_BEFORE_NEW_TRAINING"
    assert set(payload["decisions"]) == {f"DEC-{index:02d}" for index in range(1, 11)}


def test_fold_calendar_precedes_buffer_and_untouched_replay() -> None:
    decision = _decisions()["decisions"]["DEC-09"]
    folds = decision["outer_folds"]
    buffer_start, buffer_end = decision["no_use_buffer"]
    replay_start, replay_end = decision["untouched_replay_signal_interval"]

    assert all(start < end for start, end in folds)
    assert all(
        folds[index][1] <= folds[index + 1][0] for index in range(len(folds) - 1)
    )
    assert folds[-1][1] == buffer_start
    assert buffer_end == replay_start
    assert replay_start < replay_end
    assert decision["purge_hours"] + decision["post_label_embargo_hours"] == 24


def test_timing_failure_retains_enter_now_without_blocking_e1() -> None:
    decision = _decisions()["decisions"]["DEC-10"]

    assert decision["choice"] == "VALIDATE_AFTER_STABLE_E1_WINNER"
    assert decision["baseline"] == "enter_now"
    assert decision["failure_action"].startswith("retain enter_now")
    assert set(decision["required_action_value_components"]) == {
        "better_entry_benefit",
        "fill_and_missed_opportunity",
        "adverse_movement_risk",
        "cost_accounting",
    }
    price_contract = decision["price_suggestion_contract"]
    assert price_contract["long_formula"].startswith("decision_price -")
    assert price_contract["short_formula"].startswith("decision_price +")
    assert "does not round or place orders" in price_contract["model_output"]
    assert "fixed expiry and explicit fallback action" in price_contract["policy_gates"]


def test_common_ev_cost_is_applied_once() -> None:
    decision = _decisions()["decisions"]["DEC-08"]

    assert "applied exactly once" in decision["cost"]
    assert decision["selection_scope"] == "side-local top 10 percent"
