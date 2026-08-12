from __future__ import annotations

import json
from hashlib import sha256
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
    stage_i_tiered_pruning_contract,
)
from extreme_price_movements.stage_i_model_hpo import (
    HPO_SCHEDULE_SCHEMA,
    StageIModelHPOError,
    StageIModelHPOResult,
    _top_tail_mean,
    run_stage_i_model_hpo,
)
from extreme_price_movements.stage_i_ranking import (
    RANKING_POLICY,
    stable_stage_i_rank_frame,
    stable_stage_i_topk_positions,
)
from extreme_price_movements.stage_i_winner_bundle import freeze_stage_i_winner_bundle
from extreme_price_movements.stage_i_r3_contract import r3_label_economics_contract


class _FakeBaseModel:
    classes_ = np.asarray([0, 1, 2], dtype=np.int8)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        signal = np.tanh(frame.iloc[:, 0].to_numpy(float))
        clear = 0.30 + 0.15 * signal
        adverse = 0.30 - 0.15 * signal
        weak = 1.0 - clear - adverse
        return np.column_stack([adverse, weak, clear])


class _FakeMetaModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.iloc[:, 0].to_numpy(float) * 2.0


class _ConstantBaseModel:
    classes_ = np.asarray([0, 1, 2], dtype=np.int8)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile(np.asarray([[0.2, 0.3, 0.5]]), (len(frame), 1))


def _fake_fit(_x, _y, _weight, *, classifier, params, **_kwargs):
    if classifier:
        assert params["objective"] == "multiclass"
        assert params["num_class"] == 3
        return _FakeBaseModel()
    assert params["objective"] == "huber"
    return _FakeMetaModel()


def _constant_fit(_x, _y, _weight, *, classifier, params, **_kwargs):
    assert classifier
    return _ConstantBaseModel()


def _regeneration_degenerate_fit(x, _y, _weight, *, classifier, params, **_kwargs):
    assert classifier
    return _ConstantBaseModel() if len(x) > 550 else _FakeBaseModel()


def test_hpo_is_pre2024_side_local_and_regenerates_true_multiclass_oof() -> None:
    timestamps = pd.date_range("2022-01-01", periods=900, freq="D", tz="UTC")
    frame = pd.DataFrame({"signal": np.sin(np.arange(900) / 13.0)})
    target = np.resize(np.asarray([0, 1, 2], dtype=np.int8), len(frame))
    net = np.where(target == 2, 175.0, np.where(target == 0, -190.0, 5.0))
    result = run_stage_i_model_hpo(
        frame, target,
        selected_feature_names=["signal"], exact_net_bps=net,
        candidate_ids=[f"long-{i:04d}" for i in range(len(frame))],
        decision_timestamps=timestamps,
        label_available_timestamps=timestamps + pd.Timedelta(hours=12),
        side="long", layer="base", hpo_trials=3, hpo_patience=10,
        n_validation_folds=3, min_train_rows=20, fit_model=_fake_fit,
    )
    assert result.requested_trials == result.actual_trials == result.completed_trials == 3
    assert result.patience == 10
    assert result.best_params["objective"] == "multiclass"
    assert result.best_params["num_class"] == 3
    assert result.oof_probabilities is not None
    finite = np.isfinite(result.oof_probabilities).all(axis=1)
    assert finite.any()
    np.testing.assert_allclose(result.oof_probabilities[finite].sum(axis=1), 1.0)
    np.testing.assert_allclose(
        result.oof_score[finite],
        result.oof_probabilities[finite, 2] - result.oof_probabilities[finite, 0],
    )
    assert result.hpo_cutoff_utc == "2024-01-01T00:00:00+00:00"
    assert result.ranking_policy == RANKING_POLICY
    assert result.feasibility_contract is not None
    feasibility = result.feasibility_contract
    assert feasibility["min_child_samples_upper"] == 16
    assert all(
        row["params"]["min_child_samples"] <= feasibility["min_child_samples_upper"]
        for row in result.trial_audit
        if row["status"] == "complete"
    )
    assert all(row["opportunity_score_unique_count"] >= 2 for row in result.oof_fold_audit)
    assert all(row["opportunity_score_std"] > 1e-8 for row in result.oof_fold_audit)
    assert all(
        pd.Timestamp(row["validation_max_label_available_utc"])
        < pd.Timestamp(result.hpo_cutoff_utc)
        for row in result.fold_audit
    )
    assert result.best_metrics["pooled_global_top10_exact_net_bps"] == max(
        row["economic_metrics"]["pooled_global_top10_exact_net_bps"]
        for row in result.trial_audit
    )


def test_identity_tied_global_topk_is_row_permutation_invariant() -> None:
    ids = np.asarray(["d", "b", "a", "c", "e"], dtype=object)
    score = np.ones(len(ids), dtype=float)
    decision = pd.date_range("2025-01-01", periods=len(ids), freq="h", tz="UTC")
    left = stable_stage_i_topk_positions(
        score,
        candidate_ids=ids,
        side_names="long",
        decision_timestamps=decision,
        count=3,
    )
    permutation = np.asarray([3, 1, 4, 0, 2])
    right = stable_stage_i_topk_positions(
        score[permutation],
        candidate_ids=ids[permutation],
        side_names="long",
        decision_timestamps=decision[permutation],
        count=3,
    )
    assert ids[left].tolist() == ["a", "b", "c"]
    assert ids[permutation][right].tolist() == ["a", "b", "c"]


def test_hpo_tail_matches_serialized_oof_identity_ranking_with_ties() -> None:
    ids = np.asarray(["d", "b", "a", "c", "e"], dtype=object)
    score = np.ones(len(ids), dtype=float)
    net = np.asarray([-20.0, 30.0, 120.0, -50.0, 10.0])
    decision = pd.date_range("2025-01-01", periods=len(ids), freq="h", tz="UTC")
    hpo_value = _top_tail_mean(
        score,
        net,
        np.ones(len(ids)),
        candidate_ids=ids,
        decision_timestamps=decision,
        side="long",
        fraction=0.40,
    )
    serialized = pd.DataFrame(
        {
            "candidate_id": ids,
            "side_name": "long",
            "decision_ts": decision,
            "r3_opportunity_score": score,
            "exact_net_bps": net,
        }
    ).sample(frac=1.0, random_state=7)
    ordered = stable_stage_i_rank_frame(serialized, score_column="r3_opportunity_score")
    expected = float(ordered.head(2).exact_net_bps.mean())
    assert hpo_value == expected == 75.0


def test_hpo_prunes_non_discriminating_trials_and_fails_closed() -> None:
    timestamps = pd.date_range("2022-01-01", periods=900, freq="D", tz="UTC")
    frame = pd.DataFrame({"signal": np.sin(np.arange(900) / 13.0)})
    target = np.resize(np.asarray([0, 1, 2], dtype=np.int8), len(frame))
    net = np.where(target == 2, 175.0, np.where(target == 0, -190.0, 5.0))
    with pytest.raises(StageIModelHPOError, match="no feasible discriminating"):
        run_stage_i_model_hpo(
            frame,
            target,
            selected_feature_names=["signal"],
            candidate_ids=[f"long-{i:04d}" for i in range(len(frame))],
            exact_net_bps=net,
            decision_timestamps=timestamps,
            label_available_timestamps=timestamps + pd.Timedelta(hours=12),
            side="long",
            layer="base",
            hpo_trials=2,
            hpo_patience=2,
            n_validation_folds=3,
            min_train_rows=20,
            fit_model=_constant_fit,
        )


def test_hpo_regeneration_fails_closed_when_winner_degenerates_later() -> None:
    timestamps = pd.date_range("2022-01-01", periods=900, freq="D", tz="UTC")
    frame = pd.DataFrame({"signal": np.sin(np.arange(900) / 13.0)})
    target = np.resize(np.asarray([0, 1, 2], dtype=np.int8), len(frame))
    net = np.where(target == 2, 175.0, np.where(target == 0, -190.0, 5.0))
    with pytest.raises(StageIModelHPOError, match="non-discriminating Stage-I opportunity score"):
        run_stage_i_model_hpo(
            frame,
            target,
            selected_feature_names=["signal"],
            candidate_ids=[f"long-{i:04d}" for i in range(len(frame))],
            exact_net_bps=net,
            decision_timestamps=timestamps,
            label_available_timestamps=timestamps + pd.Timedelta(hours=12),
            side="long",
            layer="base",
            hpo_trials=1,
            hpo_patience=1,
            n_validation_folds=3,
            min_train_rows=20,
            fit_model=_regeneration_degenerate_fit,
        )


def _write_selector_fixture(root: Path) -> None:
    n = 72
    timestamps = pd.date_range("2022-01-01", periods=n, freq="14D", tz="UTC")
    side = np.resize(np.asarray(["long", "short"]), n)
    r3 = np.resize(np.asarray([0, 1, 2], dtype=np.int8), n)
    ledger = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(n)],
        "__ts__": timestamps,
        "__symbol__": np.resize(np.asarray(["BTC", "ETH"]), n),
        "side_name": side,
        "label_available_ts": timestamps + pd.Timedelta(hours=13),
        "exact_net_bps": np.where(r3 == 2, 150.0, np.where(r3 == 0, -180.0, 0.0)),
        "exact_gross_bps": np.where(r3 == 2, 250.0, np.where(r3 == 0, -80.0, 100.0)),
        "label_valid": True,
        "r3_class": r3,
        "r3_metric_target": r3.astype(np.float32) - 1.0,
        "robust_clear_soft_b25_t50": np.where(
            r3 == 2, 0.9, np.where(r3 == 0, 0.1, 0.4)
        ).astype(np.float32),
        "t2_tp6_sl4_event": np.where(
            r3 == 2, 0, np.where(r3 == 0, 1, 2)
        ).astype(np.int8),
    })
    features = ledger[["candidate_id", "__ts__", "__symbol__"]].copy()
    features["base_signal"] = np.arange(n, dtype=np.float32)
    features["meta_context"] = np.arange(n, dtype=np.float32) / n
    root.mkdir(parents=True)
    ledger.to_parquet(root / "selector_ledger.parquet", index=False)
    features.to_parquet(root / "selector_features.parquet", index=False)
    coverage = pd.DataFrame({"feature_name": ["base_signal"], "status": ["accepted"]})
    coverage.to_parquet(root / "selector_exact_feature_coverage_audit.parquet", index=False)
    integrity = {
        "schema": "stage_i_selector_artifact_integrity_v1",
        "selector_ledger_sha256": sha256((root / "selector_ledger.parquet").read_bytes()).hexdigest(),
        "selector_features_sha256": sha256((root / "selector_features.parquet").read_bytes()).hexdigest(),
        "exact_coverage_audit_sha256": sha256((root / "selector_exact_feature_coverage_audit.parquet").read_bytes()).hexdigest(),
        "exact_coverage_month_side_audit_sha256": None,
        "r3_label_economics_contract_sha256": r3_label_economics_contract(ledger)["contract_sha256"],
        "r3_label_economics_contract": r3_label_economics_contract(ledger),
    }
    (root / "manifest.json").write_text(json.dumps({
        "schema": "stage_i_selector_sample_v1", "status": "complete",
        "causal_warmup_prefix_features": {},
        "artifact_integrity": integrity,
    }))
    (root / "selector_feature_contract.json").write_text(json.dumps({"features": 2}))


def _fake_hpo_result(
    *, side: str, layer: str, selected: list[str], rows: int
) -> StageIModelHPOResult:
    if layer == "base":
        probability = np.tile(np.asarray([[0.2, 0.3, 0.5]], np.float32), (rows, 1))
        score = probability[:, 2] - probability[:, 0]
        params = {"objective": "multiclass", "num_class": 3, "n_estimators": 25}
    else:
        probability = None
        score = np.linspace(-5, 5, rows, dtype=np.float32)
        params = {"objective": "huber", "n_estimators": 20}
    hpo_fold = ({
        "fold_id": 0, "validation_start_utc": "2022-01-01T00:00:00+00:00",
        "validation_end_utc": "2023-12-30T00:00:00+00:00",
        "validation_max_label_available_utc": "2023-12-30T13:00:00+00:00",
        "train_rows": 10, "validation_rows": rows, "strict_prior_resolved": True,
    },)
    oof_fold = ({
        "fold_id": 0, "validation_start_utc": "2022-01-01T00:00:00+00:00",
        "validation_end_utc": "2025-01-01T00:00:00+00:00",
        "validation_max_label_available_utc": "2025-01-01T13:00:00+00:00",
        "train_rows": 10, "validation_rows": rows, "strict_prior_resolved": True,
    },)
    schedule = {
        "schema": HPO_SCHEDULE_SCHEMA,
        "enabled": False,
    }
    schedule_sha = sha256(
        json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return StageIModelHPOResult(
        side=side, layer=layer, selected_feature_names=tuple(selected),
        best_params=params, oof_score=score, oof_probabilities=probability,
        requested_trials=7, actual_trials=7, completed_trials=7, patience=3,
        stop_reason="requested_trials_completed", best_trial_number=2,
        best_value=12.0, best_metrics={"pooled_global_top10_exact_net_bps": 12.0},
        hpo_cutoff_utc="2024-01-01T00:00:00+00:00", hpo_rows=20,
        trial_audit=(), fold_audit=hpo_fold, oof_fold_audit=oof_fold,
        hpo_schedule=schedule,
        hpo_schedule_sha256=schedule_sha,
        hpo_request_sha256="a" * 64,
    )


def test_meta_base_candidate_handoff_is_global_row_order_invariant_and_target_free() -> None:
    import scripts.run_stage_i_meta_feature_selection as meta_cli

    rows = 10
    signal = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    ledger = pd.DataFrame(
        {
            "candidate_id": [f"c{i:02d}" for i in range(rows)],
            "__ts__": signal,
            "__symbol__": ["BTC"] * rows,
            "decision_ts": signal + pd.Timedelta(hours=1),
            "side_name": ["long"] * rows,
            # This field is intentionally irrelevant to the handoff.
            "exact_net_bps": np.linspace(1000, -1000, rows),
        }
    )
    score = np.asarray([0.8, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    _positions, audit = meta_cli._base_candidate_handoff(
        ledger, score, fraction=0.30
    )
    selected = set(audit.loc[audit.selected_base_candidate, "candidate_id"])
    shuffled = ledger.sample(frac=1.0, random_state=17).reset_index(drop=True)
    shuffled_score = pd.Series(score, index=ledger.candidate_id).loc[
        shuffled.candidate_id
    ].to_numpy()
    shuffled["exact_net_bps"] *= -1
    _positions_2, audit_2 = meta_cli._base_candidate_handoff(
        shuffled, shuffled_score, fraction=0.30
    )
    selected_2 = set(
        audit_2.loc[audit_2.selected_base_candidate, "candidate_id"]
    )
    assert selected == selected_2 == {"c00", "c01", "c02"}
    assert audit.ranking_scope.str.contains("never_per_timestamp").all()


def test_base_and_meta_cli_reject_nonatomic_legacy_v5_roots_before_selection(
    tmp_path: Path, monkeypatch,
) -> None:
    import scripts.run_stage_i_base_feature_selection as base_cli
    import scripts.run_stage_i_meta_feature_selection as meta_cli

    selector = tmp_path / "selector"
    _write_selector_fixture(selector)

    def legacy(side: str, *, status: str = "complete") -> dict[str, object]:
        return {
            "side": side,
            "status": status,
            "stage_i_selector_schema": "stage_i_grouped_stability_mda_v5",
            "pruning_history": [],
        }

    def forbid_selection(*args, **kwargs):
        del args, kwargs
        pytest.fail("legacy all-sides preflight must run before selection")

    monkeypatch.setattr(base_cli, "run_stage_i_head_selection", forbid_selection)
    base_missing = tmp_path / "base_missing"
    (base_missing / "long").mkdir(parents=True)
    (base_missing / "long" / "manifest.json").write_text(
        json.dumps(legacy("long"))
    )
    monkeypatch.setattr(sys, "argv", [
        "base", "--selector-dir", str(selector),
        "--output-dir", str(base_missing), "--resume",
    ])
    with pytest.raises(ValueError, match="all-requested-sides exact completed"):
        base_cli.main()

    base_incomplete = tmp_path / "base_incomplete"
    for side, status in (("long", "complete"), ("short", "running")):
        (base_incomplete / side).mkdir(parents=True)
        (base_incomplete / side / "manifest.json").write_text(
            json.dumps(legacy(side, status=status))
        )
    monkeypatch.setattr(sys, "argv", [
        "base", "--selector-dir", str(selector),
        "--output-dir", str(base_incomplete), "--resume",
    ])
    with pytest.raises(ValueError, match="all-requested-sides exact completed"):
        base_cli.main()

    monkeypatch.setattr(meta_cli, "run_stage_i_head_selection", forbid_selection)
    meta_mixed = tmp_path / "meta_mixed"
    for side in ("long", "short"):
        (meta_mixed / side).mkdir(parents=True)
    (meta_mixed / "long" / "manifest.json").write_text(
        json.dumps(legacy("long"))
    )
    pruning = stage_i_tiered_pruning_contract()
    pruning_sha = sha256(
        json.dumps(pruning, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (meta_mixed / "short" / "manifest.json").write_text(json.dumps({
        "side": "short",
        "status": "complete",
        "stage_i_selector_schema": "stage_i_grouped_stability_mda_v6",
        "stage_i_pruning_contract": pruning,
        "stage_i_pruning_contract_sha256": pruning_sha,
        "pruning_history": [],
    }))
    monkeypatch.setattr(sys, "argv", [
        "meta", "--selector-dir", str(selector),
        "--base-selection-dir", str(tmp_path / "unused_base"),
        "--output-dir", str(meta_mixed), "--resume",
    ])
    with pytest.raises(ValueError, match="all-requested-sides exact completed"):
        meta_cli.main()


def test_cli_chain_consumes_hpo_args_emits_simplex_and_freezes_winner(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_stage_i_base_feature_selection as base_cli
    import scripts.run_stage_i_meta_feature_selection as meta_cli

    selector = tmp_path / "selector"
    base_root, meta_root = tmp_path / "base", tmp_path / "meta"
    _write_selector_fixture(selector)
    calls: list[tuple[str, str, int, int]] = []
    timestamp_calls: list[tuple[str, pd.Series, pd.Series, pd.Series]] = []

    def fake_select(
        frame, _target, *, contract, candidate_kwargs, correlation_policy, **_kwargs
    ):
        timestamp_calls.append((
            f"selection:{contract.layer}:{contract.side}",
            pd.to_datetime(candidate_kwargs["timestamps"], utc=True).reset_index(drop=True),
            pd.to_datetime(
                candidate_kwargs["label_available_timestamps"], utc=True
            ).reset_index(drop=True),
            pd.to_datetime(
                candidate_kwargs["mda_reference"]["timestamps"], utc=True
            ).reset_index(drop=True),
        ))
        selected = (
            ["base_signal"] if contract.layer == "base"
            else ["meta_context", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES]
        )
        pruning_contract = stage_i_tiered_pruning_contract()
        output = {
            "selected_feature_names": selected,
            "stage_i_selected_feature_contract": selected,
            "stage_i_required_same_side_base_oof_handoff_features": list(
                STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
            ),
            "stage_i_input_feature_count": len(frame.columns),
            "stage_i_input_features": list(frame.columns),
            "stage_i_correlation_policy": correlation_policy,
            "stage_i_correlation_pruning_stage": "spearman_groups_inside_mda",
            "stage_i_pruning_contract": pruning_contract,
            "stage_i_pruning_contract_sha256": sha256(
                json.dumps(
                    pruning_contract, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "metrics": {}, "pruning_history": [], "stage_i_prefix_confirmation": {},
        }
        if contract.layer == "meta":
            output["stage_i_base_oof_provenance"] = dict(
                candidate_kwargs["base_oof_provenance"]
            )
        return output

    def fake_hpo(
        frame, _target, *, side, layer, selected_feature_names, hpo_trials,
        hpo_patience, decision_timestamps, label_available_timestamps, **_kwargs
    ):
        calls.append((side, layer, hpo_trials, hpo_patience))
        decision = pd.to_datetime(decision_timestamps, utc=True).reset_index(drop=True)
        available = pd.to_datetime(
            label_available_timestamps, utc=True
        ).reset_index(drop=True)
        timestamp_calls.append((f"hpo:{layer}:{side}", decision, available, decision))
        return _fake_hpo_result(
            side=side, layer=layer, selected=list(selected_feature_names), rows=len(frame)
        )

    monkeypatch.setattr(base_cli, "run_stage_i_head_selection", fake_select)
    monkeypatch.setattr(base_cli, "run_stage_i_model_hpo", fake_hpo)
    monkeypatch.setattr(sys, "argv", [
        "base", "--selector-dir", str(selector), "--output-dir", str(base_root),
        "--hpo-trials", "7", "--hpo-patience", "3", "--side", "long",
    ])
    assert base_cli.main() == 0
    base_oof = pd.read_parquet(base_root / "long" / "selector_base_oof.parquet")
    assert {"r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score"}.issubset(base_oof)
    assert "decision_ts" in base_oof
    assert (
        pd.to_datetime(base_oof["decision_ts"], utc=True)
        == pd.to_datetime(base_oof["__ts__"], utc=True) + pd.Timedelta(hours=1)
    ).all()
    np.testing.assert_allclose(
        base_oof[["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].sum(axis=1), 1.0
    )
    # Resume a two-side invocation after only long completed: the v2 all-era
    # HPO schedule must validate long and execute short exactly once.
    monkeypatch.setattr(sys, "argv", [
        "base", "--selector-dir", str(selector), "--output-dir", str(base_root),
        "--hpo-trials", "7", "--hpo-patience", "3", "--resume",
    ])
    assert base_cli.main() == 0

    monkeypatch.setattr(meta_cli, "run_stage_i_head_selection", fake_select)
    monkeypatch.setattr(meta_cli, "run_stage_i_model_hpo", fake_hpo)
    monkeypatch.setattr(sys, "argv", [
        "meta", "--selector-dir", str(selector),
        "--base-selection-dir", str(base_root), "--output-dir", str(meta_root),
        "--hpo-trials", "7", "--hpo-patience", "3", "--side", "long",
    ])
    assert meta_cli.main() == 0
    meta_oof = pd.read_parquet(meta_root / "long" / "selector_meta_oof.parquet")
    map_audit = pd.read_parquet(
        meta_root / "long" / "prequential_value_map_audit.parquet"
    )
    assert "decision_ts" in meta_oof and "decision_ts" in map_audit
    assert (
        pd.to_datetime(map_audit["decision_ts"], utc=True)
        == pd.to_datetime(map_audit["__ts__"], utc=True) + pd.Timedelta(hours=1)
    ).all()
    monkeypatch.setattr(sys, "argv", [
        "meta", "--selector-dir", str(selector),
        "--base-selection-dir", str(base_root), "--output-dir", str(meta_root),
        "--hpo-trials", "7", "--hpo-patience", "3", "--resume",
    ])
    assert meta_cli.main() == 0
    for name, decision, available, mda_decision in timestamp_calls:
        assert (available == decision + pd.Timedelta(hours=12)).all(), name
        if name.startswith("selection:meta:"):
            assert len(mda_decision) >= len(decision)
            assert set(decision).issubset(set(mda_decision))
        else:
            pd.testing.assert_series_equal(decision, mda_decision, check_names=False)
    assert set(calls) == {
        ("long", "base", 7, 3), ("short", "base", 7, 3),
        ("long", "meta", 7, 3), ("short", "meta", 7, 3),
    }
    before_resume = list(calls)
    monkeypatch.setattr(sys, "argv", [
        "base", "--selector-dir", str(selector), "--output-dir", str(base_root),
        "--hpo-trials", "7", "--hpo-patience", "3", "--resume",
    ])
    assert base_cli.main() == 0
    monkeypatch.setattr(sys, "argv", [
        "meta", "--selector-dir", str(selector),
        "--base-selection-dir", str(base_root), "--output-dir", str(meta_root),
        "--hpo-trials", "7", "--hpo-patience", "3", "--resume",
    ])
    assert meta_cli.main() == 0
    assert calls == before_resume
    for root in (base_root, meta_root):
        for side in ("long", "short"):
            manifest = json.loads((root / side / "manifest.json").read_text())
            assert manifest["correlation_policy"] == "grouped-preserve"
            assert manifest["stage_i_pruning_contract"] == (
                stage_i_tiered_pruning_contract()
            )
            assert manifest["stage_i_pruning_contract_sha256"] == sha256(
                json.dumps(
                    manifest["stage_i_pruning_contract"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
    for side in ("long", "short"):
        base_manifest_path = base_root / side / "manifest.json"
        base_manifest_sha = sha256(base_manifest_path.read_bytes()).hexdigest()
        meta_manifest = json.loads((meta_root / side / "manifest.json").read_text())
        assert meta_manifest["base_correlation_lineage"] == {
            "schema": "stage_i_meta_base_correlation_lineage_v1",
            "side": side,
            "correlation_policy": "grouped-preserve",
            "base_selector_manifest_sha256": base_manifest_sha,
        }
        assert (
            meta_manifest["base_oof_provenance"]["correlation_lineage"]
            == meta_manifest["base_correlation_lineage"]
        )
        assert meta_manifest["dedicated_mda_reference_scope"] == (
            "full_valid_resolved_selector_side_pre_base_top30_action_gate"
        )
        assert meta_manifest["dedicated_mda_reference_rows"] > meta_manifest["rows"]
    # A completed cell cannot be relabelled as the matched literal-pruning arm
    # on resume. Both layer CLIs bind the policy in their immutable manifest.
    monkeypatch.setattr(sys, "argv", [
        "base", "--selector-dir", str(selector), "--output-dir", str(base_root),
        "--hpo-trials", "7", "--hpo-patience", "3", "--resume",
        "--correlation-policy", "pre-mda-spearman-representative",
    ])
    with pytest.raises(ValueError, match="resume contract drift"):
        base_cli.main()
    monkeypatch.setattr(sys, "argv", [
        "meta", "--selector-dir", str(selector),
        "--base-selection-dir", str(base_root), "--output-dir", str(meta_root),
        "--hpo-trials", "7", "--hpo-patience", "3", "--resume",
        "--correlation-policy", "pre-mda-spearman-representative",
    ])
    with pytest.raises(ValueError, match="must match the completed base arm"):
        meta_cli.main()

    # The matching policy is necessary but not sufficient: a completed meta
    # cell is bound to the exact base manifest bytes used for its candidate
    # handoff. A changed policy cannot resume, and a changed manifest with the
    # same policy is separately rejected as lineage/hash drift.
    long_base_manifest_path = base_root / "long" / "manifest.json"
    original_long_base_manifest = long_base_manifest_path.read_text()
    base_manifest = json.loads(original_long_base_manifest)
    base_manifest["correlation_policy"] = "pre-mda-spearman-representative"
    long_base_manifest_path.write_text(json.dumps(base_manifest))
    monkeypatch.setattr(sys, "argv", [
        "meta", "--selector-dir", str(selector),
        "--base-selection-dir", str(base_root), "--output-dir", str(meta_root),
        "--hpo-trials", "7", "--hpo-patience", "3", "--resume",
        "--side", "long",
    ])
    with pytest.raises(ValueError, match="must match the completed base arm"):
        meta_cli.main()
    long_base_manifest_path.write_text(original_long_base_manifest)

    base_manifest = json.loads(original_long_base_manifest)
    base_manifest["lineage_hash_drift_probe"] = True
    long_base_manifest_path.write_text(json.dumps(base_manifest))
    with pytest.raises(ValueError, match="resume contract drift"):
        meta_cli.main()
    long_base_manifest_path.write_text(original_long_base_manifest)

    feature_hash = "f" * 64
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({
        "schema": "stage_i_production_input_contract_v1", "status": "complete",
        "rows": 100, "min_signal_ts": "2022-01-01T00:00:00+00:00",
        "max_signal_ts": "2026-07-10T21:00:00+00:00",
        "feature_contract_sha256": feature_hash,
    }))
    (source / "frozen_feature_contract.json").write_text(json.dumps({
        "feature_contract_sha256": feature_hash,
        "feature_columns": ["base_signal", "meta_context"],
    }))
    bundle, status = freeze_stage_i_winner_bundle(
        base_selection_dir=base_root, meta_selection_dir=meta_root,
        input_contract_dir=source, output_path=tmp_path / "winner.json",
        code_revision="a" * 40,
    )
    assert status == "created_immutable_bundle"
    assert all(cell.lgbm_params for cell in bundle.cells)
    assert bundle.cell(layer="base", side="long").lgbm_params["num_class"] == 3
    assert bundle.cell(layer="meta", side="short").lgbm_params["objective"] == "huber"
