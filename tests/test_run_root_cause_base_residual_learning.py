"""Focused contract tests for the Stage 3--4 diagnostic-only runner."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_root_cause_base_residual_learning as runner


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(7)
    times = pd.date_range("2023-03-01", "2024-11-30", freq="12h", tz="UTC")
    rows = []
    for side in ("long", "short"):
        for index, timestamp in enumerate(times):
            x1, x2 = rng.normal(), rng.normal()
            alpha = float(1.0 / (1.0 + np.exp(-(0.7 * x1 - 0.4 * x2))))
            gross = float(180.0 * (alpha - .5) + rng.normal(scale=18.0))
            candidate_id = f"{side}-{index}"
            rows.append({"candidate_id": candidate_id, "__ts__": timestamp, "side_name": side, "__reconstructed_soft_alpha_12h__": alpha, "execution_gross_ev_12h": gross, "execution_net_ev_12h": gross - 100.0, "__label_available_at__": timestamp + pd.Timedelta(hours=12), "x1": x1, "x2": x2})
    raw = pd.DataFrame(rows)
    ledger = raw.loc[:, ["candidate_id", "execution_gross_ev_12h", "execution_net_ev_12h", "__label_available_at__"]].copy()
    ledger["postcost_h0_event"] = raw["x1"].gt(0)
    ledger["postcost_h0_favorable_minute"] = np.where(raw["x1"].gt(0), 30.0, np.nan)
    ledger["postcost_h0_adverse_minute"] = np.where(raw["x2"].gt(0), 45.0, np.nan)
    ledger["postcost_h0_resolved_minute"] = 60.0
    ledger["postcost_h25_event"] = raw["x2"].gt(0)
    ledger["postcost_h25_favorable_minute"] = 40.0
    ledger["postcost_h25_adverse_minute"] = 50.0
    ledger["postcost_h25_resolved_minute"] = 70.0
    ledger["postcost_h0_retained_net"] = raw["execution_net_ev_12h"].gt(0)
    ledger["postcost_h0_giveback_after_clear"] = raw["execution_net_ev_12h"].lt(0)
    ledger["exit_hour"] = 12.0
    stack = raw.loc[:, ["candidate_id"]].copy()
    stack["score_base_alpha"] = raw["__reconstructed_soft_alpha_12h__"]
    stack["score_base_expected_ev"] = raw["execution_gross_ev_12h"] / 10_000.0
    stack["score_residual_expected_ev"] = raw["execution_gross_ev_12h"] / 10_000.0
    stack["residual_is_oof"] = True
    substrate, panel, score = tmp_path / "ledger.parquet", tmp_path / "raw.parquet", tmp_path / "stack.parquet"
    ledger.to_parquet(substrate, index=False); raw.to_parquet(panel, index=False); stack.to_parquet(score, index=False)
    contract = tmp_path / "raw_feature_contract.json"
    contract.write_text(json.dumps({"raw_feature_columns": ["x1", "x2"]}))
    return substrate, panel, contract, score


def test_purged_train_has_real_label_purge_and_embargo() -> None:
    frame = pd.DataFrame({"__ts__": pd.to_datetime(["2024-06-29T23:00Z", "2024-06-30T12:00Z", "2024-06-30T23:00Z"], utc=True), "available": pd.to_datetime(["2024-06-30T11:00Z", "2024-07-01T00:00Z", "2024-07-01T11:00Z"], utc=True)})
    fold = runner.Fold("july", pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2024-08-01", tz="UTC"), "development_oof")
    mask = runner.purged_train_mask(frame, fold, time_col="__ts__", label_available_col="available")
    assert mask.tolist() == [True, False, False]


def test_allowlist_can_only_remove_raw_causal_fields(tmp_path: Path) -> None:
    allow = tmp_path / "allow.json"; allow.write_text(json.dumps({"causal_features": ["x2"]}))
    assert runner._load_causal_allowlist(allow, ["x1", "x2"]) == ["x2"]
    allow.write_text(json.dumps(["x1", "future_mfe"]))
    try:
        runner._load_causal_allowlist(allow, ["x1", "x2"])
    except ValueError as error:
        assert "expands raw contract" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a future field must be rejected")


def test_runner_keeps_base_and_stopped_gradient_residual_separate(tmp_path: Path) -> None:
    substrate, panel, contract, stack = _fixture(tmp_path)
    output = tmp_path / "out"
    manifest = runner.run(substrate=substrate, raw_panel=panel, feature_contract=contract, stack=stack, output=output, seeds=(11, 12), families=("prior", "ridge", "future_feature_oracle"), minimum_rows=50)
    assert manifest["invariants"]["residual_training_base_is_inner_oof"]
    predictions = pd.read_parquet(output / "base_residual_oof_predictions.parquet")
    trained = predictions.loc[predictions.model_family.eq("ridge")]
    assert {"base_alpha_prediction", "base_economic_prediction_bps", "residual_prediction_bps", "combined_economic_prediction_bps"}.issubset(trained.columns)
    lineage = json.loads((output / "fold_model_lineage.json").read_text())
    model_lineage = [item for item in lineage if item.get("fit_role") != "cached_inner_base_oof"]
    assert model_lineage and all(item["stopped_gradient"] for item in model_lineage)
    assert manifest["ladder_disposition"]["M7"] == "future_feature_oracle_hindsight_only"
    assert manifest["future_oracle_features"]
    metric = pd.read_parquet(output / "model_learning_efficiency.parquet")
    assert {"base_directional", "residual_economic"}.issubset(metric.component)
    base = metric.loc[metric.component.eq("base_directional")]
    residual = metric.loc[metric.component.eq("residual_economic")]
    assert base["base_directional__roc_auc"].notna().any()
    assert base["residual_economic__net_top10_bps"].isna().all()
    assert residual["residual_economic__net_top10_bps"].notna().any()
    assert residual["base_directional__roc_auc"].isna().all()
    assert (output / "metric_concordance.parquet").exists()
    concordance = pd.read_parquet(output / "metric_concordance.parquet")
    assert {"base_to_later_residual_association", "arm_outcome", "paired_day_bootstrap_vs_prior"}.issubset(set(concordance.record_type.dropna()))
    arm = concordance.loc[concordance.record_type.eq("arm_outcome")]
    assert {"later_worst_month_gross_bps", "later_worst_month_net_bps", "later_worst_side_gross_bps", "later_worst_side_net_bps"}.issubset(arm.columns)
    gaps = pd.read_parquet(output / "model_learning_gaps_and_seed_dispersion.parquet")
    assert "train_heldout_gap" in set(gaps.record_type.dropna())
    assert "seed_dispersion" in set(gaps.record_type.dropna())
    source = Path(runner.__file__).read_text()
    assert "simple_policy_optimiser(" not in source
    assert "execution_entry_timing" not in source
