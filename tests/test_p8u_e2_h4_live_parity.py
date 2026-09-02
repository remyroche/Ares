from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference import p8u_e2_h4_continuation as continuation
from extreme_price_movements.inference.p8u_e2_h4_live_parity import (
    apply_e2_replacement,
    apply_h4_next_interval,
    validate_h4_giveback20_training_contract,
)
from extreme_price_movements.inference.p8u_e2_h4_rich_policy import (
    h4_effective_params,
    h4_interval_active,
)
from extreme_price_movements.inference.strict_r3_live_execution import (
    _stored_exit_vwap_adjusted_trigger,
)
from scripts.refit_strict_r3_p8u_e2_h4_live_parity_bundle import _action_aligned_h4_panel
from extreme_price_movements.inference.p8u_e2_h4_auction import apply_e2_before_auction


class _PairModel:
    feature_name_ = ()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.repeat(60.0, len(frame))


class _H4Model:
    feature_name_ = ()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([1.0, -1.0], dtype=float)


class _Bundle:
    manifest_sha256 = "bundle-hash"
    e2_features = ("margin__bcf_final_score", "incumbent_bcf_mc1_expected_bps")
    h4_features = ("state_x",)

    def e2_models(self):
        return _PairModel(), _PairModel()

    def h4_model(self):
        return _H4Model()


def test_e2_replaces_only_the_marginal_core_slot() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["core_a", "core_b", "reserve"],
        "__decision_ts__": ["2026-08-30T00:00:00Z"] * 3,
        "__symbol__": ["A", "B", "C"],
        "bcf_final_score": [.99, .90, .20],
        "bcf_mc1_expected_bps": [100., 90., 30.],
        "current_mc1_expected_bps": [100., 90., 25.],
        "dual_mc1_min_bps": [100., 90., 25.],
        "e2_feature_source_status": ["ok", "ok", "ok"],
    })
    selected, pairs = apply_e2_replacement(frame, bundle=_Bundle())
    assert len(pairs) == 1
    assert selected.loc[selected.e2_entry_selected, "candidate_id"].tolist() == ["core_a", "reserve"]
    assert selected.set_index("candidate_id").loc["core_b", "e2_action"] == "e2_q50_agreement_demoted_marginal"
    assert selected.groupby("__decision_ts__").e2_entry_selected.sum().tolist() == [2]


def test_h4_authority_is_next_interval_only_and_non_promotional() -> None:
    state = pd.DataFrame({"state_x": [0., 1.]})
    result = apply_h4_next_interval(state, bundle=_Bundle())
    assert result.h4_active.tolist() == [True, False]
    assert result.h4_activation_earlier.tolist() == [.5, .0]
    assert result.h4_giveback_tighten.tolist() == [.2, .0]
    assert result.h4_sl_tighten.tolist() == [.0, .0]
    assert result.h4_effective_from_next_interval.tolist() == [True, True]


def test_h4_giveback_runtime_action_requires_matching_counterfactual_target() -> None:
    valid = {
        "continuation_training_contract": {
            "target": "latched_activation50_giveback20_advantage_bps",
            "authority": {
                "activation_earlier": .5,
                "giveback_tighten": .2,
                "sl_tighten": .0,
                "effective_from": "next_completed_15m_interval",
            },
            "labels": {"path": "labels.parquet", "sha256": "a" * 64},
            "states": {"path": "states.parquet", "sha256": "b" * 64},
            "route": {"path": "route.parquet", "sha256": "c" * 64},
        }
    }
    validate_h4_giveback20_training_contract(valid)
    invalid = {**valid, "continuation_training_contract": {**valid["continuation_training_contract"], "target": "activation50_advantage_bps"}}
    with pytest.raises(ValueError, match="action-aligned target"):
        validate_h4_giveback20_training_contract(invalid)


def test_action_aligned_h4_panel_uses_target_free_route_for_mc1(tmp_path) -> None:
    state_ts = pd.Timestamp("2026-08-01T00:15:00Z")
    states = pd.DataFrame({
        "candidate_id": ["a"],
        "state_decision_ts": [state_ts],
        "entry_decision_ts": [pd.Timestamp("2026-08-01T00:00:00Z")],
        # Exact counterfactual simulation intentionally stores the neutral
        # value; it must not make the state ineligible for training.
        "MC1_expected_bps": [0.0],
    })
    labels = pd.DataFrame({
        "candidate_id": ["a"],
        "state_decision_ts": [state_ts],
        "activation50_advantage_bps": [12.0],
        "policy_label_available_ts": [pd.Timestamp("2026-08-01T12:00:00Z")],
    })
    route = pd.DataFrame({"candidate_id": ["a"], "bcf_mc1_expected_bps": [75.0]})
    paths = [tmp_path / name for name in ("states.parquet", "labels.parquet", "route.parquet")]
    states.to_parquet(paths[0], index=False)
    labels.to_parquet(paths[1], index=False)
    route.to_parquet(paths[2], index=False)
    panel = _action_aligned_h4_panel(states_path=paths[0], labels_path=paths[1], route_path=paths[2])
    assert panel.loc[0, "MC1_expected_bps"] == pytest.approx(75.0)
    assert panel.loc[0, "latched_activation50_giveback20_advantage_bps"] == pytest.approx(12.0)


@dataclass(frozen=True)
class _ParentParams:
    trailing_activation_mult: float = 1.5
    fixed_trailing_gap_mult: float = 0.25
    untouched: float = 7.0


def test_h4_interval_starts_after_its_completed_state_bar() -> None:
    position = {
        "p8u_h4_active": True,
        "p8u_h4_effective_from": "2026-08-30T00:15:00Z",
        "p8u_h4_effective_until": "2026-08-30T00:30:00Z",
    }
    assert not h4_interval_active(position, completed_bar_end="2026-08-30T00:15:00Z")
    assert h4_interval_active(position, completed_bar_end="2026-08-30T00:16:00Z")
    assert h4_interval_active(position, completed_bar_end="2026-08-30T00:30:00Z")
    assert not h4_interval_active(position, completed_bar_end="2026-08-30T00:31:00Z")
    child = h4_effective_params(_ParentParams(), active=True)
    assert child.trailing_activation_mult == .75
    assert child.fixed_trailing_gap_mult == .20
    assert child.untouched == 7.0


def test_h4_giveback_stop_is_converted_through_the_persisted_exit_vwap_contract() -> None:
    """H4 changes the *native* exit threshold, not merely a paper policy stop.

    A long's H4 giveback contract is 20% tighter.  The resulting policy stop
    must then be translated with the exact stored full-size exit-impact factor
    before a Kraken replacement order is sent.  This is the required bridge
    from the 15m H4 decision to executable stop/VWAP economics.
    """
    parent = _ParentParams()
    h4 = h4_effective_params(parent, active=True)
    entry, maximum_favourable, tp_distance = 100.0, 10.0, 5.0
    parent_policy_stop = entry + maximum_favourable - (
        tp_distance * parent.fixed_trailing_gap_mult
    )
    h4_policy_stop = entry + maximum_favourable - (
        tp_distance * h4.fixed_trailing_gap_mult
    )
    assert h4_policy_stop > parent_policy_stop

    position = {
        "side": "long",
        "protective_stop_vwap": {
            "enabled": True,
            "exit_vwap_impact_fraction": 0.01,
        },
    }
    trigger, vwap = _stored_exit_vwap_adjusted_trigger(
        position=position,
        policy_exit_price=h4_policy_stop,
    )
    assert trigger == pytest.approx(h4_policy_stop / 0.99)
    assert vwap["policy_exit_price"] == pytest.approx(h4_policy_stop)
    assert vwap["expected_exit_vwap_price"] == pytest.approx(h4_policy_stop)
    assert vwap["exit_vwap_impact_bps"] == pytest.approx(100.0)


class _OneH4Model:
    feature_name_ = ()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.repeat(1.0, len(frame))


class _OneH4Bundle(_Bundle):
    def h4_model(self):
        return _OneH4Model()


def test_h4_decision_is_persisted_as_hash_bound_next_interval(monkeypatch) -> None:
    state = pd.DataFrame({"state_x": [1.0]})
    monkeypatch.setattr(continuation, "completed_h4_state", lambda **_: state)
    position, scored = continuation.persist_h4_next_interval_decision(
        position={"candidate_id": "a"},
        bars_15m=pd.DataFrame(),
        state_decision_ts="2026-08-30T00:15:00Z",
        expectation_reference=pd.DataFrame(),
        bundle=_OneH4Bundle(),
    )
    assert scored.h4_active.tolist() == [True]
    assert position["p8u_h4_bundle_manifest_sha256"] == "bundle-hash"
    assert position["p8u_h4_prediction_bps"] == 1.0
    assert position["p8u_h4_effective_from"] == "2026-08-30T00:15:00+00:00"
    assert position["p8u_h4_effective_until"] == "2026-08-30T00:30:00+00:00"
    assert len(position["p8u_h4_last_state_hash"]) == 64


def test_e2_replacement_reaches_unchanged_auction_before_capacity() -> None:
    candidates = pd.DataFrame({
        "candidate_id": ["A|long|x", "B|long|x", "C|long|x"],
        "__decision_ts__": ["2026-08-30T00:00:00Z"] * 3,
        "__symbol__": ["A", "B", "C"],
        "bcf_mc1_expected_bps": [100.0, 90.0, 80.0],
        # B is the E2 replacement.  C may not enter simply because it has a
        # high BCF value: capacity is consumed only after E2's authority.
        "e2_entry_selected": [True, True, False],
    })
    result = apply_e2_before_auction(
        candidates, state={"open_positions": [], "pending_intents": []}, wallet_equity_quote=1_000.0
    ).set_index("candidate_id")
    assert result.loc["A|long|x", "execution_action"] == "propose"
    assert result.loc["B|long|x", "execution_action"] == "propose"
    assert result.loc["C|long|x", "rejection_reason"] == "not_selected_by_e2"
