import json
from pathlib import Path

import pytest

from extreme_price_movements.inference.simple_policy_stop import (
    SIMPLE_POLICY_GENERATOR,
    SIMPLE_POLICY_SCHEMA,
    SimplePolicyStopParamsError,
    compute_simple_policy_initial_stop_decision,
    compute_simple_policy_stop_decision,
    extract_simple_policy_stop_params_by_strategy,
    load_simple_policy_stop_params_by_strategy,
    validate_simple_policy_stop_params,
)


def _params(**overrides):
    strategy_id = str(overrides.get("strategy_id", "long_mr"))
    base = Path("/tmp/ares_simple_policy_stop_tests")
    source = "artifacts/20260101_000000/simple_policy_optimiser/deployment/best_policy_params.json"
    artifact_path = base / source
    row = {
        "generated_by": SIMPLE_POLICY_GENERATOR,
        "schema": SIMPLE_POLICY_SCHEMA,
        "strategy_id": strategy_id,
        "barrier_pct": 0.02,
        "sl_mult": 1.0,
        "trailing_activation_mult": 1.0,
        "trailing_power": 1.5,
        "trailing_squash_divisor": 2.0,
        "giveback_beta": 0.5,
        "capital_protect_mfe_mult": 1.0,
        "capital_protect_regression_frac": 0.45,
        "adverse_exit_enabled": False,
        "adverse_exit_alpha": 1.0,
        "adverse_exit_beta": 1.0,
        "adverse_exit_delta": 1.0,
        "adverse_exit_theta_quantile": 0.75,
        "adverse_exit_theta": 0.65,
        "adverse_exit_fast_bars": 4,
        "adverse_exit_min_mae_atr": 1.0,
        "adverse_exit_min_speed": 0.3,
        "adverse_exit_max_mfe_atr": 0.25,
        "provisional_trailing_stage_sl_mult": None,
        "sl_mult_source": None,
        "stage2_selection_method": None,
        "adverse_exit_disabled_reason": None,
    }
    row.update({k: v for k, v in overrides.items() if k in row})
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "generated_by": SIMPLE_POLICY_GENERATOR,
                "schema_version": SIMPLE_POLICY_SCHEMA,
                "strategies": [row],
            },
            sort_keys=True,
        )
    )
    params = load_simple_policy_stop_params_by_strategy(
        str(base), run_id="20260101_000000"
    )[strategy_id]
    params.update({k: v for k, v in overrides.items() if k not in row})
    return params


def _write_artifact(tmp_path, run_id, strategies):
    path = (
        tmp_path
        / "artifacts"
        / run_id
        / "simple_policy_optimiser"
        / "deployment"
        / "best_policy_params.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "generated_by": SIMPLE_POLICY_GENERATOR,
                "schema_version": SIMPLE_POLICY_SCHEMA,
                "strategies": strategies,
            }
        )
    )
    return path


def test_compute_replacement_requires_artifact_metadata():
    params = _params()
    params.pop("generated_by")
    with pytest.raises(SimplePolicyStopParamsError, match="generated_by"):
        compute_simple_policy_stop_decision(
            state={"entry_price": 100.0, "stop_price": 98.0, "strategy_id": "long_mr"},
            latest_market_state={},
            policy_params=params,
            side="long",
        )


def test_compute_initial_requires_artifact_metadata():
    params = _params()
    params.pop("schema")
    with pytest.raises(SimplePolicyStopParamsError, match="schema"):
        compute_simple_policy_initial_stop_decision(
            entry_price=100.0,
            policy_params=params,
            side="long",
            strategy_id="long_mr",
        )


def test_initial_stop_accepts_live_barrier_from_state_not_artifact():
    params = _params()
    params.pop("barrier_pct")
    decision = compute_simple_policy_initial_stop_decision(
        entry_price=100.0,
        policy_params=params,
        side="long",
        strategy_id="long_mr",
        barrier_frac=0.015,
    )
    assert decision.stop_price == pytest.approx(98.5)
    assert decision.barrier_frac == pytest.approx(0.015)


def test_adverse_excursion_exit_uses_ranked_confidence_minus_half():
    params = _params(
        adverse_exit_enabled=True,
        adverse_exit_theta=0.5,
        adverse_exit_min_mae_atr=0.5,
        adverse_exit_min_speed=0.1,
        adverse_exit_max_mfe_atr=0.25,
    )
    decision = compute_simple_policy_stop_decision(
        state={
            "entry_price": 100.0,
            "stop_price": 98.0,
            "strategy_id": "long_mr",
            "rank_percentile": 0.90,
        },
        latest_market_state={
            "open": 100.0,
            "high": 100.1,
            "low": 98.8,
            "close": 99.0,
        },
        policy_params=params,
        side="long",
    )

    assert decision.should_exit is True
    assert decision.exit_reason == "adverse_excursion_exit"
    assert "ranked_confidence_minus_0_5=0.4" in decision.reason_detail


def test_loader_uses_stage2_medoid_sl_mult_and_adverse_params():
    params = _params(
        sl_mult=1.1,
        adverse_exit_enabled=True,
        adverse_exit_theta=0.6421,
        adverse_exit_min_mae_atr=1.0,
        adverse_exit_min_speed=0.3,
        provisional_trailing_stage_sl_mult=1.4,
        sl_mult_source="capital_preservation_adverse_exit_top15_cluster_medoid",
        stage2_selection_method="top15_cluster_medoid",
    )

    assert params["sl_mult"] == pytest.approx(1.1)
    assert params["adverse_exit_enabled"] is True
    assert params["adverse_exit_theta"] == pytest.approx(0.6421)
    assert params["provisional_trailing_stage_sl_mult"] == pytest.approx(1.4)
    assert (
        params["sl_mult_source"]
        == "capital_preservation_adverse_exit_top15_cluster_medoid"
    )


def test_initial_stop_requires_live_barrier_when_artifact_has_none():
    params = _params()
    params.pop("barrier_pct")
    with pytest.raises(SimplePolicyStopParamsError, match="barrier"):
        compute_simple_policy_initial_stop_decision(
            entry_price=100.0,
            policy_params=params,
            side="long",
            strategy_id="long_mr",
        )


@pytest.mark.parametrize(
    "params_source",
    ["", "simple_policy_optimiser", "unversioned_simple_policy_params"],
)
def test_generic_params_source_is_rejected(params_source):
    with pytest.raises(SimplePolicyStopParamsError, match="params_source"):
        validate_simple_policy_stop_params(_params(params_source=params_source))


def test_missing_params_hash_is_rejected():
    params = _params()
    params.pop("params_hash")
    with pytest.raises(SimplePolicyStopParamsError, match="params_hash"):
        validate_simple_policy_stop_params(params)


def test_wrong_strategy_id_is_rejected():
    with pytest.raises(SimplePolicyStopParamsError, match="strategy_id mismatch"):
        validate_simple_policy_stop_params(
            _params(strategy_id="long_other"), state={"strategy_id": "long_mr"}
        )


@pytest.mark.parametrize(
    "legacy_field",
    [
        "trail_mult",
        "giveback_pct",
        "profit_lock_amount",
        "fixed_stop_loss_pct",
        "stop_loss_pct",
        "stop_loss_frac",
        "mfe_early_exit_threshold",
        "trailing_override_alpha",
    ],
)
def test_legacy_stop_fields_are_rejected(legacy_field):
    with pytest.raises(SimplePolicyStopParamsError, match="unknown simple-policy"):
        validate_simple_policy_stop_params(_params(**{legacy_field: 1.0}))


def test_extractor_accepts_only_explicit_container_and_exact_strategy():
    params = _params()
    assert extract_simple_policy_stop_params_by_strategy({"long_mr": params}) == {}
    assert (
        extract_simple_policy_stop_params_by_strategy(
            {"simple_policy_stop_params_by_strategy": {"mr": params}}
        )
        == {}
    )
    extracted = extract_simple_policy_stop_params_by_strategy(
        {"simple_policy_stop_params_by_strategy": {"long_mr": params}}
    )
    assert set(extracted) == {"long_mr"}


def test_loader_selects_latest_valid_artifact_for_exact_strategy(tmp_path):
    stale = _params(params_source="ignored", params_hash="ignored", sl_mult=0.5)
    fresh = _params(params_source="ignored", params_hash="ignored", sl_mult=1.2)
    other = _params(
        params_source="ignored",
        params_hash="ignored",
        strategy_id="short_mr",
        sl_mult=9.0,
    )
    _write_artifact(tmp_path, "20260101_000000", [stale, other])
    latest_path = _write_artifact(tmp_path, "20260102_000000", [fresh])

    loaded = load_simple_policy_stop_params_by_strategy(
        str(tmp_path), run_id="20260102_000000"
    )

    assert loaded["long_mr"]["sl_mult"] == 1.2
    assert (
        loaded["long_mr"]["params_source"]
        == latest_path.relative_to(tmp_path).as_posix()
    )
    assert loaded["long_mr"]["strategy_id"] == "long_mr"
    assert "short_mr" not in loaded


def test_loader_ignores_wrong_generator_and_bad_strategy_rows(tmp_path):
    bad_path = (
        tmp_path
        / "artifacts"
        / "20260101_000000"
        / "simple_policy_optimiser"
        / "deployment"
        / "best_policy_params.json"
    )
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text(
        json.dumps(
            {
                "generated_by": "policy_optimiser",
                "schema_version": SIMPLE_POLICY_SCHEMA,
                "strategies": [_params()],
            }
        )
    )
    assert load_simple_policy_stop_params_by_strategy(str(tmp_path)) == {}


def test_legacy_order_manager_v2_is_quarantined():
    import importlib.util
    import sys
    from pathlib import Path

    module_path = Path("live_trading/order_manager_v2.py")
    spec = importlib.util.spec_from_file_location(
        "order_manager_v2_quarantine", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with pytest.raises(RuntimeError, match="quarantined"):
        module.OrderManagerV2(
            api_client=object(), portfolio_manager=object(), config={}
        )


def test_fake_concrete_params_source_without_loader_stamp_is_rejected():
    params = _params()
    params.pop("_loaded_from_simple_policy_artifact")
    with pytest.raises(SimplePolicyStopParamsError, match="not loaded"):
        validate_simple_policy_stop_params(params)


def test_initial_stop_rejects_non_positive_long_stop():
    params = _params(sl_mult=100.0, barrier_pct=0.02)
    with pytest.raises(SimplePolicyStopParamsError, match="positive"):
        compute_simple_policy_initial_stop_decision(
            entry_price=100.0,
            policy_params=params,
            side="long",
            strategy_id="long_mr",
        )


def test_manual_replacement_decision_with_fake_artifact_is_rejected():
    import sys
    import types

    from extreme_price_movements.inference.simple_policy_stop import (
        SimplePolicyStopDecision,
    )

    optuna = types.ModuleType("optuna")
    optuna_pruners = types.ModuleType("optuna.pruners")
    optuna_samplers = types.ModuleType("optuna.samplers")
    optuna_pruners.MedianPruner = object
    optuna_samplers.TPESampler = object
    sys.modules.setdefault("optuna", optuna)
    sys.modules.setdefault("optuna.pruners", optuna_pruners)
    sys.modules.setdefault("optuna.samplers", optuna_samplers)
    from extreme_price_movements.inference.trade_executor import (
        _validate_policy_stop_decision,
    )

    state = {
        "strategy_id": "long_mr",
        "bucket_key": "long_mr",
        "stop_policy_params_source": "artifacts/run/simple_policy_optimiser/deployment/best_policy_params.json",
        "stop_policy_params_hash": "real-hash",
    }
    artifact = _params(
        params_source="artifacts/run/simple_policy_optimiser/deployment/best_policy_params.json",
        params_hash="real-hash",
    )
    decision = SimplePolicyStopDecision(
        should_replace=True,
        stop_price=123.0,
        reason="manual",
        reason_detail="manual forged decision",
        strategy_id="long_mr",
        params_source="artifacts/fake/simple_policy_optimiser/deployment/best_policy_params.json",
        params_hash="fake-hash",
        barrier_frac=0.02,
        sl_mult=1.0,
    )

    valid, reason = _validate_policy_stop_decision(
        decision,
        position_state=state,
        artifact_params=artifact,
    )
    assert not valid
    assert "params_source" in reason


def test_inference_runtime_has_no_legacy_stop_field_references():
    forbidden = [
        "trail_mult",
        "giveback_pct",
        "profit_lock_amount",
        "fixed_stop_loss_pct",
        "stop_loss_pct",
        "stop_loss_frac",
        "mfe_early_exit_threshold",
        "trailing_override_alpha",
    ]
    roots = [Path("extreme_price_movements/inference")]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name == "simple_policy_stop.py":
                continue
            text = path.read_text(errors="ignore")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}:{token}")
    assert offenders == []
