from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import run_root_cause_feature_information_audit as stage2


def _frame(rows_per_side_month: int = 30) -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(17)
    for side in ("long", "short"):
        for month in pd.period_range("2024-01", "2024-04", freq="M"):
            decision = pd.Timestamp(month.start_time, tz="UTC") + pd.Timedelta(days=4)
            for i in range(rows_per_side_month):
                x = float(rng.normal())
                records.append({
                    "candidate_id": f"{side}-{month}-{i}", "side": side,
                    "decision_ts": decision + pd.Timedelta(minutes=i),
                    "feature_cutoff_ts": decision + pd.Timedelta(minutes=i),
                    "label_available_ts": decision + pd.Timedelta(hours=12, minutes=i),
                    "gross_h12_bps": 20.0 * x + float(rng.normal()),
                    "net_h12_bps": 20.0 * x - 100.0,
                    "symbol": "X" if i % 2 else "Y", "policy_archetype": "a" if i % 3 else "b",
                    "price_signal": x, "mkt_rv_24h": float(rng.normal()),
                    "known_row_cost_bps": 100.0,
                })
    return pd.DataFrame(records)


def test_target_proximity_scanner_rejects_realised_cost_and_target_names() -> None:
    assert stage2.scan_target_proximity("known_row_cost_bps")["hard_reject_name"]
    assert stage2.scan_target_proximity("future_mfe_12h")["hard_reject_name"]
    assert not stage2.scan_target_proximity("mkt_rv_24h")["hard_reject_name"]


def test_chronological_folds_train_only_on_resolved_rows() -> None:
    frame = _frame(rows_per_side_month=100)
    folds = stage2.make_chronological_folds(frame, min_train_rows=20)
    assert folds
    for fold in folds:
        train = frame.iloc[fold.train_index]
        test = frame.iloc[fold.test_index]
        assert train.side.eq(fold.side).all()
        assert test.side.eq(fold.side).all()
        assert pd.to_datetime(train.label_available_ts, utc=True).lt(fold.start).all()
        assert pd.to_datetime(test.decision_ts, utc=True).ge(fold.start).all()


def test_univariate_tests_are_side_local_and_transport_signal() -> None:
    frame = _frame()
    inventory = stage2.build_feature_inventory(frame, ["price_signal", "mkt_rv_24h", "known_row_cost_bps"])
    folds = stage2.make_chronological_folds(frame, min_train_rows=20)
    detail, summary = stage2.run_univariate_tests(frame, inventory, folds, target_col="gross_h12_bps")
    assert set(detail.side) == {"long", "short"}
    signal = summary.loc[summary.feature_name.eq("price_signal")]
    assert signal.transported_ic_mean.gt(0.8).all()
    rejected = inventory.loc[inventory.feature_name.eq("known_row_cost_bps")].iloc[0]
    assert not rejected.causal_probe_eligible


def test_mechanism_group_classifier_is_deterministic() -> None:
    assert stage2.classify_mechanism("mkt_oi_chg_1h") == "open_interest"
    assert stage2.classify_mechanism("funding_rate") == "funding"
    assert stage2.classify_mechanism("market_pc1_variance_share_24h") == "cross_sectional_breadth"
    assert stage2.classify_mechanism("mkt_rv_24h") == "volatility"


def test_fold_local_gross_mapping_uses_oos_alpha_and_gross_residual_arithmetic() -> None:
    frame = _frame(rows_per_side_month=100)
    frame["score_base_alpha"] = 1.0 / (1.0 + np.exp(-frame.price_signal))
    frame["score_residual_alpha"] = np.clip(frame.score_base_alpha + 0.03, 0.0, 1.0)
    frame["residual_is_oof"] = True
    folds = stage2.make_chronological_folds(frame, min_train_rows=20)
    mapping, predictions = stage2.materialize_fold_local_gross_maps(frame, folds)
    assert mapping.status.eq("OK").all()
    assert set(predictions["head"]) == {"canonical_gross_base", "canonical_gross_residual_stack"}
    assert np.allclose(
        predictions.gross_mapping_residual_bps,
        predictions.gross_h12_bps - predictions.gross_mapped_prediction_bps,
    )
    assert mapping.train_label_available_before_test_start.all()


def test_end_to_end_runner_writes_explicit_not_run_residual_status(tmp_path) -> None:
    frame = _frame()
    ledger_columns = [
        "candidate_id", "symbol", "side", "decision_ts", "feature_cutoff_ts", "label_available_ts",
        "gross_h12_bps", "net_h12_bps", "policy_archetype",
    ]
    ledger = frame.loc[:, ledger_columns].copy()
    ledger["gross_h12_proxy_status"] = "EXECUTION_ADJUSTED_PRE_FEE"
    ledger_path = tmp_path / "ledger.parquet"; ledger.to_parquet(ledger_path, index=False)
    raw = frame.loc[:, ["candidate_id", "decision_ts", "symbol", "gross_h12_bps", "price_signal", "mkt_rv_24h", "known_row_cost_bps"]].copy()
    raw = raw.rename(columns={"decision_ts": "__ts__", "symbol": "__symbol__", "gross_h12_bps": "__reconstructed_soft_alpha_12h__"})
    raw_path = tmp_path / "raw.parquet"; raw.to_parquet(raw_path, index=False)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({"raw_feature_columns": ["price_signal", "mkt_rv_24h", "known_row_cost_bps"]}))
    output = tmp_path / "stage2"
    manifest = stage2.run(
        ledger_path=ledger_path, raw_panel_path=raw_path, raw_contract_path=contract_path,
        output=output, min_train_rows=20, max_folds=2,
    )
    assert manifest["status"] == "DIAGNOSTIC_ONLY_NO_MODEL_OR_POLICY_PROMOTION"
    assert (output / "feature_information_results.parquet").exists()
    assert (output / "feature_information_residual_probes.parquet").exists()
    probes = pd.read_parquet(output / "feature_information_residual_probes.parquet")
    assert probes.status.str.startswith("NOT_RUN_MISSING_FROZEN_OOF_ECONOMIC").any()
    assert (output / "feature_information_directional_alpha_diagnostics.parquet").exists()
    inventory = pd.read_parquet(output / "feature_information_inventory.parquet")
    assert not inventory.loc[inventory.feature_name.eq("known_row_cost_bps"), "causal_probe_eligible"].iloc[0]
