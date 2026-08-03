from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.shared_regime_residual_expert import SoftRegimeResidualConfig
from extreme_price_movements.stage_iii_shared_expert_runner import (
    StageIIIInputLineageContract,
    StageIIIRunnerConfig,
    StageIIISequentialRunnerError,
    StageIIIStack,
    build_expanding_environment_folds,
    declared_sequential_arms,
    run_stage_iii_sequential_funnel,
    stage_iii_advancement_gates,
    stage_iii_feature_contract_sha256,
)


SOFT = ("p_regime_calm", "p_regime_stress")
INVARIANT = ("r3_p_clear", "r3_p_adverse", "r3_p_weak", "market_confirmation")
RELATIVE = ("market_confirmation",)
INTERACTIONS = ("r3_p_clear", "r3_p_adverse")
VALIDITY = {
    "relationship_breaks": ("relationship_break_score",),
    "contribution_ood": ("contribution_ood_score",),
    "active_failure_probability": ("active_failure_probability",),
}


def _feature_sources() -> list[str]:
    return list(dict.fromkeys([
        *INVARIANT, *SOFT, *RELATIVE, *INTERACTIONS,
        *(feature for group in VALIDITY.values() for feature in group),
    ]))


def _digest(seed: str) -> str:
    return sha256(seed.encode()).hexdigest()


def _lineage(root: Path, **changes: object) -> StageIIIInputLineageContract:
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for key in ("r3", "base_map", "soft_regime", "label", "admission"):
        path = root / f"{key}.json"
        path.write_text(f'{{"artifact":"{key}"}}\n', encoding="utf-8")
        paths[f"{key}_artifact_path"] = str(path)
        hashes[f"{key}_artifact_sha256"] = sha256(path.read_bytes()).hexdigest()
    source_features = _feature_sources()
    feature_payload = {
        "schema": "stage_iii_feature_admission_v1",
        "config": {"min_coverage": 0.90},
        "admitted_ordered_features": source_features,
        "feature_audit": [
            {
                "feature_name": name,
                "classification": "INVARIANT_CORE" if name in INVARIANT else "REGIME_CONDITIONAL",
                "admitted": True,
                "coverage": 1.0,
                "null_fraction": 0.0,
                "finite_fraction": 1.0,
                "live_parity": True,
                "meta_allowed_key": True,
            }
            for name in source_features
        ],
    }
    feature_path = root / "feature_contract.json"
    feature_path.write_text(json.dumps(feature_payload, sort_keys=True), encoding="utf-8")
    paths["feature_contract_artifact_path"] = str(feature_path)
    hashes["feature_contract_artifact_sha256"] = sha256(feature_path.read_bytes()).hexdigest()
    values = {
        **paths, **hashes,
        "feature_contract_sha256": stage_iii_feature_contract_sha256(_feature_sources()),
    }
    values.update(changes)
    return StageIIIInputLineageContract(**values)


def _rewrite_feature_artifact(
    contract: StageIIIInputLineageContract, mutation,
) -> StageIIIInputLineageContract:
    path = Path(contract.feature_contract_artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return replace(
        contract, feature_contract_artifact_sha256=sha256(path.read_bytes()).hexdigest()
    )


def _frame(rows_per_environment: int = 12) -> pd.DataFrame:
    environments = ["era_0", "era_1", "era_2", "era_3", "era_4", "era_5"]
    rows = rows_per_environment * len(environments)
    signal = pd.date_range("2024-01-01", periods=rows, freq="12h", tz="UTC")
    index = np.arange(rows)
    phase = index / 5.0
    calm = np.clip(0.5 + 0.4 * np.sin(phase), 0.05, 0.95)
    clear = np.clip(0.5 + 0.3 * np.cos(phase / 2), 0.05, 0.90)
    adverse = np.clip(0.15 + 0.1 * np.sin(phase), 0.03, 0.40)
    weak = 1.0 - clear - adverse
    too_low = weak < 0.02
    clear[too_low] -= 0.02 - weak[too_low]
    weak = 1.0 - clear - adverse
    side = np.where(index % 2 == 0, "long", "short")
    base = 30.0 * (clear - adverse)
    outcome = base + 40.0 * np.sin(phase) + np.where(side == "long", 5.0, -5.0)
    decision = signal + pd.Timedelta(hours=1)
    return pd.DataFrame({
        "signal_close_ts": signal,
        "candidate_id": [f"candidate_{value:05d}" for value in index],
        "symbol": np.where(index % 3 == 0, "BTC/USD:USD", "ETH/USD:USD"),
        "decision_ts": decision,
        "label_available_ts": signal + pd.Timedelta(hours=13),
        "side_name": side,
        "environment": np.repeat(environments, rows_per_environment),
        "exact_net_bps": outcome,
        "exact_gross_bps": outcome + 100.0,
        "prequential_base_expected_net_bps": base,
        "total_cost_bps": 100.0,
        "r3_p_clear": clear,
        "r3_p_adverse": adverse,
        "r3_p_weak": weak,
        "p_regime_calm": calm,
        "p_regime_stress": 1.0 - calm,
        "market_confirmation": np.sin(phase / 3),
        "relationship_break_score": np.abs(np.sin(phase / 4)),
        "contribution_ood_score": np.abs(np.cos(phase / 3)),
        "active_failure_probability": np.clip(0.2 + 0.2 * np.sin(phase), 0, 1),
        "cost_to_atr": 1.0 + 0.05 * np.sin(phase),
        "cost_atr_is_causal": True,
        "broad_regime": np.where(calm >= 0.5, "calm", "stress"),
        "r3_is_strict_oof": True,
        "r3_source_side": side,
        "r3_fit_end_ts": decision - pd.Timedelta(days=1),
        "r3_score_semantics": "same_side_direct_strict_oof_probabilities_without_conversion",
        "base_map_is_prequential": True,
        "base_map_source_side": side,
        "base_map_max_label_available_ts": decision - pd.Timedelta(hours=1),
        "soft_regime_is_causal_prequential": True,
        "soft_regime_fit_end_ts": decision - pd.Timedelta(hours=1),
        "causal_21d_admitted": index % 3 != 0,
        "causal_21d_admission_is_prequential": True,
        "causal_21d_admission_source_side": side,
        "causal_21d_admission_max_label_available_ts": decision - pd.Timedelta(hours=1),
        "causal_21d_admission_window_days": 21,
    })


def _config() -> StageIIIRunnerConfig:
    return StageIIIRunnerConfig(
        hard_regime_column="broad_regime",
        min_train_environments=2,
        min_train_rows=12,
        min_rows_per_side=3,
        top_fractions=(0.05, 0.10),
        primary_top_fraction=0.10,
        calibration_min_rows=2,
        calibration_anchor="timestamp",
        baseline_config=SoftRegimeResidualConfig(
            min_global_rows=2, side_shrink_rows=4, regime_shrink_rows=4,
            regime_weight_cap=0.5, residual_scale_floor_bps=1,
        ),
        hpo_param_candidates=({"bias": -5.0}, {"bias": 0.0}, {"bias": 5.0}),
    )


class _PredictionModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


@dataclass
class _FakeFit:
    value: float
    max_label_available_utc: pd.Timestamp
    target_mode: str = "huber"

    def predict_candidate_residual_bps(self, frame: pd.DataFrame) -> np.ndarray:
        # One score for both sides, with candidate variation from an inference feature.
        return (self.value + 8.0 * frame["market_confirmation"].to_numpy(float)).astype(np.float32)


def _fake_fitter(frame: pd.DataFrame, **kwargs: object) -> _FakeFit:
    cutoff = pd.Timestamp(kwargs["fit_before_utc"])
    available = pd.to_datetime(frame["label_available_ts"], utc=True)
    assert (available < cutoff).all()
    assert frame.side_name.nunique() == 2
    params = kwargs.get("params", {})
    target = frame["candidate_residual_bps"].to_numpy(float)
    return _FakeFit(float(np.mean(target)) + float(params.get("bias", 0.0)), available.max())


class _FakePairSupport:
    def __init__(self, arm: str, rows: int):
        self.arm = arm
        self.rows = rows

    def to_dict(self) -> dict[str, object]:
        enabled = self.arm != "F0_pointwise"
        return {
            "separation_bps": 50.0 if self.arm.endswith("50bps") else (100.0 if enabled else None),
            "pair_selection": "synthetic_context_pairs" if enabled else "disabled_for_F0_pointwise",
            "constructed_pairs": self.rows // 2 if enabled else 0,
            "selected_pairs": self.rows // 4 if enabled else 0,
            "selected_pair_rows": self.rows // 2 if enabled else 0,
            "pair_ledger_sha256": sha256(f"all:{self.arm}:{self.rows}".encode()).hexdigest() if enabled else None,
            "selected_pair_ledger_sha256": sha256(f"selected:{self.arm}:{self.rows}".encode()).hexdigest() if enabled else None,
            "selected_pairs_by_side": [["long", self.rows // 8], ["short", self.rows // 8]] if enabled else [],
            "selected_unique_candidates": self.rows // 2 if enabled else 0,
            "max_pair_label_available_utc": None,
            "pair_builder_schema": "synthetic_test",
            "pair_builder_routing": "one_shared_model_no_local_experts",
            "pair_config": {},
        }


class _FakePairAudit:
    def __init__(self, arm: str, frame: pd.DataFrame):
        self.pair_support = _FakePairSupport(arm, len(frame))
        self.max_label_available_utc = pd.to_datetime(frame.label_available_ts, utc=True).max()


class _FakePairFit(_FakeFit):
    def __init__(self, value: float, available: pd.Timestamp, arm: str, frame: pd.DataFrame):
        super().__init__(value, available)
        self.arm = arm
        self.audit = _FakePairAudit(arm, frame)

    def predict_candidate_residual_bps(self, frame: pd.DataFrame) -> np.ndarray:
        base = super().predict_candidate_residual_bps(frame)
        adjustment = 0.0 if self.arm == "F0_pointwise" else (0.2 if self.arm.endswith("50bps") else 0.1)
        return (base + adjustment * frame.market_confirmation.to_numpy(float)).astype(np.float32)


def _fake_pairwise_fitter(frame: pd.DataFrame, **kwargs: object) -> _FakePairFit:
    cutoff = pd.Timestamp(kwargs["fit_before_utc"])
    available = pd.to_datetime(frame.label_available_ts, utc=True)
    assert (available < cutoff).all()
    target = frame.candidate_residual_bps.to_numpy(float)
    return _FakePairFit(float(target.mean()), available.max(), str(kwargs["arm"]), frame)


@dataclass
class _PerfectRobustFit:
    max_label_available_utc: pd.Timestamp

    def predict_candidate_residual_bps(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["candidate_residual_bps"].to_numpy(np.float32)


def _perfect_robust_fitter(frame: pd.DataFrame, **kwargs: object) -> _PerfectRobustFit:
    cutoff = pd.Timestamp(kwargs["fit_before_utc"])
    available = pd.to_datetime(frame["label_available_ts"], utc=True)
    assert (available < cutoff).all()
    return _PerfectRobustFit(available.max())


def _fake_target_preserving_fitter(
    frame: pd.DataFrame, **kwargs: object,
) -> _FakePairFit:
    assert isinstance(kwargs["base_model"], _PerfectRobustFit)
    cutoff = pd.Timestamp(kwargs["fit_before_utc"])
    available = pd.to_datetime(frame.label_available_ts, utc=True)
    assert (available < cutoff).all()
    fit = _FakePairFit(0.0, available.max(), str(kwargs["arm"]), frame)
    fit.predict_candidate_residual_bps = kwargs["base_model"].predict_candidate_residual_bps
    return fit


def test_declared_rounds_are_exact_and_non_factorial() -> None:
    assert declared_sequential_arms() == {
        "A_target_normalization": ("A0", "A1", "A2", "A3"),
        "T_residual_target": ("T0", "T1_200", "T1_400", "T2", "T3", "T4"),
        "B_training_robustness": ("B0", "B1", "B2", "B3"),
        "C_conditioning": ("C0", "C1", "C2", "C3"),
        "D_model_validity": ("D0", "D1", "D2", "D3", "D4"),
        "E_calibration": ("E0", "E1", "E2"),
        "F_pairwise_ranking": ("F0", "F1", "F2"),
    }
    assert sum(map(len, declared_sequential_arms().values())) == 29


def test_lineage_is_serializable_and_rejects_naming_only_or_hard_routing(tmp_path: Path) -> None:
    frame = _frame()
    contract = _lineage(tmp_path)
    restored = StageIIIInputLineageContract.from_dict(contract.to_dict())
    assert restored.contract_sha256 == contract.contract_sha256
    restored.validate(
        frame, config=_config(), soft_regime_columns=SOFT,
        invariant_features=INVARIANT, regime_relative_features=RELATIVE,
        restricted_interaction_features=INTERACTIONS, validity_feature_groups=VALIDITY,
    )

    with pytest.raises(StageIIISequentialRunnerError, match="row evidence"):
        restored.validate(
            frame.drop(columns="r3_is_strict_oof"), config=_config(),
            soft_regime_columns=SOFT, invariant_features=INVARIANT,
            regime_relative_features=RELATIVE, restricted_interaction_features=INTERACTIONS,
            validity_feature_groups=VALIDITY,
        )
    with pytest.raises(StageIIISequentialRunnerError, match="hard routing"):
        _lineage(tmp_path, routing="hard_routing").validate(
            frame, config=_config(), soft_regime_columns=SOFT, invariant_features=INVARIANT,
            regime_relative_features=RELATIVE, restricted_interaction_features=INTERACTIONS,
            validity_feature_groups=VALIDITY,
        )
    tampered = _lineage(tmp_path)
    Path(tampered.r3_artifact_path).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(StageIIISequentialRunnerError, match="hash mismatch"):
        tampered.validate(
            frame, config=_config(), soft_regime_columns=SOFT, invariant_features=INVARIANT,
            regime_relative_features=RELATIVE, restricted_interaction_features=INTERACTIONS,
            validity_feature_groups=VALIDITY,
        )


def test_gross_net_and_21d_admission_provenance_fail_closed(tmp_path: Path) -> None:
    wrong_gross = _frame()
    wrong_gross.loc[0, "exact_gross_bps"] += 1.0
    with pytest.raises(StageIIISequentialRunnerError, match="gross minus 100"):
        build_expanding_environment_folds(wrong_gross, config=_config())

    future_admission = _frame()
    future_admission["causal_21d_admission_max_label_available_ts"] = future_admission["decision_ts"]
    with pytest.raises(StageIIISequentialRunnerError, match="current/future"):
        _lineage(tmp_path).validate(
            future_admission, config=_config(), soft_regime_columns=SOFT,
            invariant_features=INVARIANT, regime_relative_features=RELATIVE,
            restricted_interaction_features=INTERACTIONS, validity_feature_groups=VALIDITY,
        )


@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("r3_is_strict_oof", "false"),
        ("base_map_is_prequential", np.nan),
        ("soft_regime_is_causal_prequential", 2),
        ("causal_21d_admitted", 1.0),
        ("causal_21d_admission_is_prequential", "true"),
        ("cost_atr_is_causal", 1.0),
    ],
)
def test_every_lineage_flag_requires_canonical_boolean(
    tmp_path: Path, column: str, bad_value: object,
) -> None:
    frame = _frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = bad_value
    with pytest.raises(StageIIISequentialRunnerError, match="canonical bool"):
        _lineage(tmp_path).validate(
            frame, config=_config(), soft_regime_columns=SOFT,
            invariant_features=INVARIANT, regime_relative_features=RELATIVE,
            restricted_interaction_features=INTERACTIONS,
            validity_feature_groups=VALIDITY,
        )


def test_candidate_identity_and_chronology_fail_closed() -> None:
    blank = _frame()
    blank.loc[0, "symbol"] = " "
    with pytest.raises(StageIIISequentialRunnerError, match="symbol"):
        build_expanding_environment_folds(blank, config=_config())

    duplicate = _frame()
    duplicate.loc[1, ["candidate_id", "symbol", "decision_ts", "side_name"]] = duplicate.loc[
        0, ["candidate_id", "symbol", "decision_ts", "side_name"]
    ].to_numpy()
    with pytest.raises(StageIIISequentialRunnerError, match="row identity must be unique"):
        build_expanding_environment_folds(duplicate, config=_config())

    unordered = _frame().iloc[::-1].reset_index(drop=True)
    with pytest.raises(StageIIISequentialRunnerError, match="chronological"):
        build_expanding_environment_folds(unordered, config=_config())

    duplicate_index = _frame()
    duplicate_index.index = np.zeros(len(duplicate_index), dtype=int)
    assert build_expanding_environment_folds(duplicate_index, config=_config())


def test_feature_admission_classes_parity_and_runtime_coverage_fail_closed(tmp_path: Path) -> None:
    conditional_invariant = _rewrite_feature_artifact(
        _lineage(tmp_path),
        lambda payload: payload["feature_audit"][0].update(classification="REGIME_CONDITIONAL"),
    )
    with pytest.raises(StageIIISequentialRunnerError, match="appropriate class"):
        conditional_invariant.validate(
            _frame(), config=_config(), soft_regime_columns=SOFT,
            invariant_features=INVARIANT, regime_relative_features=RELATIVE,
            restricted_interaction_features=INTERACTIONS, validity_feature_groups=VALIDITY,
        )

    unstable = _rewrite_feature_artifact(
        _lineage(tmp_path),
        lambda payload: payload["feature_audit"][-1].update(
            classification="UNSTABLE", admitted=False
        ),
    )
    with pytest.raises(StageIIISequentialRunnerError, match="forbidden admission class"):
        unstable.validate(
            _frame(), config=_config(), soft_regime_columns=SOFT,
            invariant_features=INVARIANT, regime_relative_features=RELATIVE,
            restricted_interaction_features=INTERACTIONS, validity_feature_groups=VALIDITY,
        )

    no_parity = _rewrite_feature_artifact(
        _lineage(tmp_path),
        lambda payload: payload["feature_audit"][-1].update(live_parity=False),
    )
    with pytest.raises(StageIIISequentialRunnerError, match="live-parity"):
        no_parity.validate(
            _frame(), config=_config(), soft_regime_columns=SOFT,
            invariant_features=INVARIANT, regime_relative_features=RELATIVE,
            restricted_interaction_features=INTERACTIONS, validity_feature_groups=VALIDITY,
        )

    sparse = _frame()
    sparse.loc[:9, "market_confirmation"] = np.nan
    with pytest.raises(StageIIISequentialRunnerError, match="coverage"):
        _lineage(tmp_path).validate(
            sparse, config=_config(), soft_regime_columns=SOFT,
            invariant_features=INVARIANT, regime_relative_features=RELATIVE,
            restricted_interaction_features=INTERACTIONS, validity_feature_groups=VALIDITY,
        )


def test_expanding_folds_are_environment_atomic_and_strictly_prior_resolved() -> None:
    frame = _frame()
    folds = build_expanding_environment_folds(frame, config=_config())
    assert [fold.validation_environment for fold in folds] == ["era_2", "era_3", "era_4", "era_5"]
    for fold in folds:
        assert fold.train_max_label_available_utc < fold.validation_start_utc
        assert frame.iloc[fold.validation_positions].environment.nunique() == 1
        assert set(frame.iloc[fold.train_positions].side_name) == {"long", "short"}

    repeated = frame.copy()
    repeated.loc[repeated.environment.eq("era_5"), "environment"] = "era_0"
    with pytest.raises(StageIIISequentialRunnerError, match="contiguous"):
        build_expanding_environment_folds(repeated, config=_config())


def test_full_funnel_freezes_round_winners_and_reports_global_tail_contributions(tmp_path: Path) -> None:
    result = run_stage_iii_sequential_funnel(
        _frame(), config=_config(), input_lineage=_lineage(tmp_path),
        soft_regime_columns=SOFT, invariant_features=INVARIANT,
        regime_relative_features=RELATIVE,
        restricted_interaction_features=INTERACTIONS,
        validity_feature_groups=VALIDITY, expert_fitter=_fake_fitter,
        pairwise_expert_fitter=_fake_pairwise_fitter,
        ordinal_target_fitter=_fake_fitter,
        quantile_target_fitter=_fake_fitter,
    )
    assert result.schema.endswith("sequential_funnel_v1")
    assert len(result.arms) == 32
    assert set(result.round_winners) == set(declared_sequential_arms())

    b_winner = next(arm for arm in result.arms if arm.arm == result.round_winners["B_training_robustness"])
    later = [arm for arm in result.arms if arm.round_name in {"C_conditioning", "D_model_validity", "E_calibration"}]
    assert {arm.selected_params_index for arm in later} == {b_winner.selected_params_index}

    calibration = [arm for arm in result.arms if arm.round_name == "E_calibration"]
    assert all(arm.shared_model_fit_count == 0 for arm in calibration)
    first_raw = calibration[0].oof_predictions.raw_shared_common_bps.to_numpy()
    for arm in calibration[1:]:
        np.testing.assert_array_equal(first_raw, arm.oof_predictions.raw_shared_common_bps)
    frozen_rows = result.arms[0].oof_predictions.__candidate_identity.to_numpy()
    for arm in result.arms[1:]:
        np.testing.assert_array_equal(frozen_rows, arm.oof_predictions.__candidate_identity.to_numpy())
        assert arm.model_feature_names
        assert len(arm.model_feature_contract_sha256) == 64
        assert arm.source_feature_contract_sha256 == result.arms[0].source_feature_contract_sha256

    f_results = [arm for arm in result.arms if arm.round_name == "F_pairwise_ranking"]
    assert len(f_results) == 6
    assert {arm.arm.split("@", 1)[0] for arm in f_results} == {"F0", "F1", "F2"}
    assert all("@E" in arm.arm for arm in f_results)
    assert len({arm.predecessor_arm for arm in f_results}) <= 2
    for arm in f_results:
        assert not arm.pair_support.empty
    for arm in f_results:
        if arm.arm.startswith("F0@"):
            assert arm.pair_support.selected_pairs.eq(0).all()
        else:
            assert arm.pair_support.selected_pair_ledger_sha256.notna().all()

    # Selection lift is paired against the frozen causal base over the exact
    # same global candidate population and k, not against the all-row mean.
    winner_oof = result.winner.oof_predictions
    k = max(1, int(np.ceil(_config().primary_top_fraction * len(winner_oof))))
    model_top = winner_oof.sort_values(
        ["score_bps", "__candidate_identity"], ascending=[False, True], kind="stable"
    ).head(k)
    base_top = winner_oof.sort_values(
        ["prequential_base_expected_net_bps", "__candidate_identity"],
        ascending=[False, True], kind="stable",
    ).head(k)
    expected_lift = model_top.exact_net_bps.mean() - base_top.exact_net_bps.mean()
    assert result.winner.selection_summary["pooled_top_lift_bps"] == pytest.approx(expected_lift)

    metrics = result.winner.metrics
    assert set(metrics.admission_scope) == {"without_21d", "with_21d"}
    assert {"month_diagnostic", "side_diagnostic", "environment_diagnostic"}.issubset(set(metrics.scope))
    for fraction in _config().top_fractions:
        local_metrics = metrics.loc[metrics.admission_scope.eq("without_21d")]
        pooled = local_metrics.loc[(local_metrics.scope == "pooled_global_tail") & (local_metrics.top_fraction == fraction)].iloc[0]
        contributions = local_metrics.loc[
            (local_metrics.scope == "pooled_global_selected_contribution") & (local_metrics.top_fraction == fraction)
        ]
        assert int(contributions.selected_rows.sum()) == int(pooled.selected_rows)
        assert contributions.side.isin(["long", "short"]).all()
    assert result.winner.fold_audit.shared_model_fit.all()
    assert not result.winner.fold_audit.hard_routing.any()
    assert not result.transport_matrix.empty
    successful = result.transport_matrix.loc[result.transport_matrix.status.eq("strict_prior_resolved_transport")]
    assert (successful.train_max_label_available_utc < successful.test_start_utc).all()
    assert successful.shared_model.all()
    assert not successful.hard_routing.any()
    assert set(successful.top_fraction) == {0.01, 0.05, 0.10, 0.20}
    assert successful.final_stack_identity.nunique() == 1
    assert successful.final_stack_identity.iloc[0] == result.advancement_gates[
        "expected_transport_stack_identity"
    ]
    assert successful.round_f_arm.eq(result.winner.arm).all()
    assert successful.calibration_mode.eq(result.winner.stack.calibration_mode).all()
    for column in (
        "top_gross_bps", "top_net_bps", "mae_bps", "huber_loss",
        "calibration_slope", "calibration_intercept", "long_selected_rows",
        "short_selected_rows", "long_gross_bps", "short_gross_bps",
        "long_net_bps", "short_net_bps", "train_candidate_identity_sha256",
        "test_candidate_identity_sha256", "selected_candidate_identity_sha256",
    ):
        assert column in successful
    assert "advances" in result.advancement_gates
    assert result.advancement_gates["terminal_decision_code"] in {
        "SHARED_RESIDUAL_EXPERT_TRANSPORTS",
        "SHARED_EXPERT_REQUIRES_REGIME_CONDITIONING",
        "SHARED_EXPERT_REMAINS_CROSS_ERA_UNSTABLE",
    }
    negative_transport = result.transport_matrix.copy()
    negative_transport.loc[
        negative_transport.status.eq("strict_prior_resolved_transport"), "paired_top_lift_bps"
    ] = -1.0
    blocked = stage_iii_advancement_gates(
        result.winner.oof_predictions, negative_transport, config=_config(),
        expected_transport_stack_identity=result.advancement_gates[
            "expected_transport_stack_identity"
        ],
    )
    assert blocked["gate_transport_worst_positive"] is False
    assert blocked["advances"] is False
    assert blocked["terminal_decision_code"] != "SHARED_RESIDUAL_EXPERT_TRANSPORTS"


def test_round_f_preserves_an_ordinal_or_quantile_target_winner(tmp_path: Path) -> None:
    result = run_stage_iii_sequential_funnel(
        _frame(), config=_config(),
        input_lineage=_lineage(tmp_path), soft_regime_columns=SOFT,
        invariant_features=INVARIANT, regime_relative_features=RELATIVE,
        restricted_interaction_features=INTERACTIONS,
        validity_feature_groups=VALIDITY, expert_fitter=_fake_fitter,
        pairwise_expert_fitter=_fake_pairwise_fitter,
        target_preserving_pairwise_fitter=_fake_target_preserving_fitter,
        ordinal_target_fitter=_perfect_robust_fitter,
        quantile_target_fitter=_perfect_robust_fitter,
    )
    assert result.winner.stack.residual_target_mode in {"ordinal", "quantile"}
    assert result.round_winners["F_pairwise_ranking"].startswith(("F0@", "F1@", "F2@"))
    round_f = [arm for arm in result.arms if arm.round_name == "F_pairwise_ranking"]
    assert len(round_f) == 6
    assert all(not arm.pair_support.empty for arm in round_f)
    transported = result.transport_matrix.loc[
        result.transport_matrix.status.eq("strict_prior_resolved_transport")
    ]
    assert not transported.empty
    assert transported.round_f_arm.eq(result.winner.arm).all()


def test_default_tail_contract_includes_top_20_percent() -> None:
    assert 0.20 in StageIIIRunnerConfig().top_fractions
