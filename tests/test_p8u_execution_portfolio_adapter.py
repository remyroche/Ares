from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_execution_portfolio_adapter import (
    P8UExecutionContract,
    P8UPortfolioState,
    _auction,
    prepare_execution_intent,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[P8UExecutionContract, P8UPortfolioState, Path]:
    policy = tmp_path / "policy.json"
    policy.write_text('{"policy":"frozen"}\n', encoding="utf-8")
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("immutable\n", encoding="utf-8")
    bundle = tmp_path / "bundle.json"
    _write_json(bundle, {
        "schema": "strict_r3_p8u_preproduction_bundle_v1",
        "side": "long",
        "runtime": {"order_submission": False},
        "artifacts": {"fixture": {"path": "artifact.txt", "type": "file", "sha256": _sha256(artifact)}},
    })
    contract_path = tmp_path / "contract.json"
    adapter_module = tmp_path / "adapter_module.py"
    adapter_runner = tmp_path / "adapter_runner.py"
    adapter_module.write_text("sealed-module\n", encoding="utf-8")
    adapter_runner.write_text("sealed-runner\n", encoding="utf-8")
    _write_json(contract_path, {
        "schema": "strict_r3_p8u_execution_portfolio_adapter_v1",
        "side": "long",
        "order_submission": False,
        "bundle": {"path": "bundle.json", "sha256": _sha256(bundle)},
        "staged_score": {"contract_hash": "fixture-stage-contract"},
        "admission": {
            "bcf_min_expected_bps": 50.0,
            "current_min_expected_bps": 50.0,
            "priority": "bcf_mc1_expected_bps",
        },
        "portfolio": {
            "max_concurrent_positions": 8,
            "max_new_entries_per_decision": 2,
            "margin_budget_fraction": 0.8,
            "margin_slot_fraction": 0.1,
            "leverage": 7.0,
        },
        "execution": {"entry_delay_minutes": 5, "max_signal_age_minutes": 15},
        "policy": {"path": "policy.json", "sha256": _sha256(policy)},
        "adapter_runtime": {
            "module": {"path": "adapter_module.py", "sha256": _sha256(adapter_module)},
            "runner": {"path": "adapter_runner.py", "sha256": _sha256(adapter_runner)},
        },
    })
    contract = P8UExecutionContract.load(contract_path, workspace_root=tmp_path)
    state = P8UPortfolioState.empty(contract, wallet_equity_quote=10_000.0)
    commit = tmp_path / "commit"
    commit.mkdir()
    decision = pd.Timestamp("2026-08-29T10:00:00Z")
    source = decision - pd.Timedelta(hours=1)
    ids = [
        "A/USD:USD|long|2026-08-29T09:00:00Z",
        "B/USD:USD|long|2026-08-29T09:00:00Z",
        "C/USD:USD|long|2026-08-29T09:00:00Z",
        "D/USD:USD|long|2026-08-29T09:00:00Z",
    ]
    router = pd.DataFrame({
        "candidate_id": ids,
        "__decision_ts__": decision,
        "side_name": "long",
        "router50_eligible": [True, True, False, False],
        "router_fraction": 0.5,
    })
    router.to_parquet(commit / "router_scores.parquet", index=False)
    router.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].to_parquet(
        commit / "router_features.parquet", index=False
    )
    routed = pd.DataFrame({
        "candidate_id": ids[:2],
        "__decision_ts__": decision,
        "side_name": "long",
        "bcf_mc1_expected_bps": [120.0, 90.0],
        "current_mc1_expected_bps": [60.0, 70.0],
        "dual_mc1_admitted": [True, True],
    })
    routed.to_parquet(commit / "routed_scores.parquet", index=False)
    routed.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].to_parquet(
        commit / "routed_features.parquet", index=False
    )
    _write_json(commit / "receipt.json", {
        "schema": "strict_r3_p8u_staged_timestamp_executor_v1",
        "status": "pass_target_free_router_first_score",
        "source_timestamp": source.isoformat(),
        "decision_timestamp": decision.isoformat(),
        "candidate_rows": 4,
        "router50_rows": 2,
        "staged_contract_hash": "fixture-stage-contract",
        "outcome_columns_consumed": [],
        "policy_or_portfolio_called": False,
        "exchange_or_order_submission_called": False,
    })
    return contract, state, commit


def test_execution_adapter_enforces_router50_dual_gate_and_bcf_priority(tmp_path: Path) -> None:
    contract, state, commit = _fixture(tmp_path)

    auction, receipt, next_state = prepare_execution_intent(
        contract=contract,
        state=state,
        staged_commit=commit,
        now=pd.Timestamp("2026-08-29T10:10:00Z"),
    )

    proposed = auction.loc[auction.execution_action.eq("propose")]
    assert proposed.candidate_id.tolist() == [
        "A/USD:USD|long|2026-08-29T09:00:00Z",
        "B/USD:USD|long|2026-08-29T09:00:00Z",
    ]
    assert proposed.auction_priority_bps.tolist() == [120.0, 90.0]
    assert proposed.requested_notional_quote.tolist() == [7000.0, 7000.0]
    assert receipt["order_submission"] is False
    assert set(receipt["adapter_runtime_hashes"]) == {"module", "runner"}
    assert receipt["router_complete_universe_rows"] == 4
    assert receipt["router50_rows"] == 2
    assert receipt["dual_mc1_admitted_rows"] == 2
    assert len(next_state["pending_intents"]) == 1
    assert len(next_state["processed_score_commit_sha256"]) == 1


def test_execution_adapter_fails_closed_on_stale_score_or_bundle_drift(tmp_path: Path) -> None:
    contract, state, commit = _fixture(tmp_path)

    with pytest.raises(ValueError, match="stale"):
        prepare_execution_intent(
            contract=contract,
            state=state,
            staged_commit=commit,
            now=pd.Timestamp("2026-08-29T10:16:00Z"),
        )

    (tmp_path / "artifact.txt").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bundle artifact hash mismatch"):
        prepare_execution_intent(
            contract=contract,
            state=state,
            staged_commit=commit,
            now=pd.Timestamp("2026-08-29T10:10:00Z"),
        )


def test_execution_adapter_fails_closed_on_adapter_runtime_drift(tmp_path: Path) -> None:
    contract, state, commit = _fixture(tmp_path)
    (tmp_path / "adapter_module.py").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter module hash mismatch"):
        prepare_execution_intent(
            contract=contract,
            state=state,
            staged_commit=commit,
            now=pd.Timestamp("2026-08-29T10:10:00Z"),
        )


def test_agreement_tier_mode_orders_both_then_c0_then_c1_without_changing_raw_ev() -> None:
    contract = SimpleNamespace(payload={
        "admission": {
            "selection_mode": "c0_c1_agreement_tier",
            "bcf_min_expected_bps": 50.0,
            "current_min_expected_bps": 50.0,
            "priority": "portfolio_order_priority_bps",
            "tier_offset_bps": 10_000.0,
            "unpaired_order": "c0_then_c1",
        },
        "portfolio": {
            "max_concurrent_positions": 8,
            "max_new_entries_per_decision": 2,
            "margin_budget_fraction": 0.8,
            "margin_slot_fraction": 0.1,
            "leverage": 7.0,
        },
    })
    state = SimpleNamespace(open_positions=(), pending_intents=(), wallet_equity_quote=10_000.0)
    scores = pd.DataFrame({
        "candidate_id": [
            "BOTH/USD:USD|long|2026-09-01T00:00:00Z",
            "C1/USD:USD|long|2026-09-01T00:00:00Z",
            "C0/USD:USD|long|2026-09-01T00:00:00Z",
        ],
        "symbol": ["BOTH/USD:USD", "C1/USD:USD", "C0/USD:USD"],
        "bcf_mc1_expected_bps": [80.0, 110.0, 300.0],
        "current_mc1_expected_bps": [80.0, 110.0, 300.0],
        "auction_priority_bps": [80.0, 110.0, 300.0],
        "dual_mc1_admitted": [True, True, True],
        "c0_bcf_mc1_expected_bps": [80.0, 20.0, 300.0],
        "c0_current_mc1_expected_bps": [80.0, 80.0, 300.0],
        "c1_bcf_mc1_expected_bps": [90.0, 110.0, 20.0],
        "c1_current_mc1_expected_bps": [90.0, 110.0, 110.0],
        "c0_dual_mc1_admitted": [True, False, True],
        "c1_dual_mc1_admitted": [True, True, False],
        "agreement_tier": [2, 1, 0],
        "admission_provenance": ["both_admitted", "c1_only", "c0_only"],
        "score_coordinate_source": ["C0", "C1", "C0"],
        "portfolio_tier": [2, 0, 1],
        "c0_c1_unpaired_order": ["c0_then_c1"] * 3,
        "portfolio_order_priority_bps": [20_080.0, 110.0, 10_300.0],
    })
    result = _auction(scores, state, contract)
    proposed = result.loc[result.execution_action.eq("propose")]
    assert proposed.candidate_id.tolist() == [
        "BOTH/USD:USD|long|2026-09-01T00:00:00Z",
        "C0/USD:USD|long|2026-09-01T00:00:00Z",
    ]
    assert proposed.auction_priority_bps.tolist() == [80.0, 300.0]
    assert proposed.portfolio_order_priority_bps.tolist() == [20_080.0, 10_300.0]
