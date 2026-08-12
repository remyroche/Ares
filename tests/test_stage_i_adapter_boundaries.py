from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.stage_i_adapter_production_oos import (
    StageIAdapterProductionInput,
)
from extreme_price_movements.stage_i_adapter_winner_bundle import (
    StageIAdapterWinnerBundle,
    StageIAdapterWinnerCell,
)
from extreme_price_movements.stage_i_production_oos import (
    run_stage_i_target_adapter_production_oos,
)
from extreme_price_movements.stage_i_strict_oof import (
    generate_stage_i_target_adapter_strict_oof,
)
from extreme_price_movements.stage_i_target_adapter import (
    FOLD_QUANTILE_RESIDUAL3,
    SOFT_SCALAR_S,
    bind_target_contract,
    canonical_sha256,
)
from extreme_price_movements.stage_i_winner_bundle import (
    build_stage_i_target_adapter_winner_bundle,
    freeze_stage_i_target_adapter_winner_bundle,
)


class _Regressor:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(0.5 + 0.4 * frame.iloc[:, 0].to_numpy(float), 0.0, 1.0)


class _Classifier:
    classes_ = np.array([0, 1, 2], dtype=np.int8)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        signal = np.tanh(frame.iloc[:, 0].to_numpy(float))
        return np.column_stack([
            0.3 - 0.1 * signal, np.full(len(frame), 0.4), 0.3 + 0.1 * signal,
        ])


def _fit(_x, _y, _weight, *, classifier, **_kwargs):
    return _Classifier() if classifier else _Regressor()


def _input(side: str, n: int = 96):
    signal_ts = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    decision = signal_ts + pd.Timedelta(hours=1)
    available = decision + pd.Timedelta(hours=12)
    signal = np.resize(np.asarray([-1.0, -0.4, 0.2, 0.8]), n)
    net = signal * 250.0
    contract = pd.DataFrame({
        "candidate_id": [f"{side}-{index}" for index in range(n)],
        "__ts__": signal_ts,
        "__symbol__": np.resize(np.asarray(["BTC", "ETH"]), n),
        "side_name": side,
        "base_target": np.clip((signal + 1.0) / 2.0, 0.0, 1.0),
        "meta_basis": net,
        "gross_bps": net + 100.0,
        "net_bps": net,
        "target_valid": True,
        "sample_weight": 1.0,
        "meta_sample_weight": 1.0,
    })
    base_contract = bind_target_contract(
        contract, family=SOFT_SCALAR_S, layer="base", target_name="S__sl4_tp6",
        geometry="sl4_tp6", target_columns=("base_target",),
    )
    meta_contract = bind_target_contract(
        contract, family=FOLD_QUANTILE_RESIDUAL3, layer="meta",
        target_name="fold_quantile_residual3", geometry="sl4_tp6",
        target_columns=("meta_basis",), weight_column="meta_sample_weight",
    )
    panel = {"schema": "test_panel_v1", "side": side, "rows": n}
    source = StageIAdapterProductionInput(
        side=side,
        frame=pd.DataFrame({"base_feature": signal, "context_feature": signal}),
        contract_frame=contract, candidate_ids=contract.candidate_id,
        decision_timestamps=decision, label_available_timestamps=available,
        base_target=contract.base_target, exact_gross_bps=contract.gross_bps,
        exact_net_bps=contract.net_bps, target_valid=contract.target_valid,
        sample_weight=contract.sample_weight, panel_manifest=panel,
        panel_manifest_sha256=canonical_sha256(panel),
        base_target_column="base_target", meta_basis_column="meta_basis",
        candidate_fraction=1.0, n_validation_folds=3, min_train_rows=12,
    )
    cell = StageIAdapterWinnerCell(
        side=side, base_features=("base_feature",),
        meta_features=("context_feature", "base_raw_score", "prequential_base_expected_net_bps"),
        base_params={"objective": "regression_l1"},
        meta_params={"objective": "multiclass", "num_class": 3},
        base_target_contract=base_contract, meta_target_contract=meta_contract,
        base_selector_manifest_sha256="1" * 64,
        meta_selector_manifest_sha256="2" * 64,
        required_same_side_base_handoff_features=(
            "base_raw_score", "prequential_base_expected_net_bps",
        ),
    )
    return source, cell


def test_existing_module_boundaries_dispatch_target_adapter_production(tmp_path) -> None:
    long_input, long_cell = _input("long")
    short_input, short_cell = _input("short")
    bundle = StageIAdapterWinnerBundle(
        cells=(long_cell, short_cell), code_revision="test-revision",
    )
    manifest = run_stage_i_target_adapter_production_oos(
        bundle=bundle, inputs=(long_input, short_input),
        output_dir=tmp_path / "oos", fit_model=_fit,
    )
    assert manifest["schema"] == "stage_i_production_target_adapter_oos_v2"
    assert manifest["status"] == "complete"
    assert manifest["rows"] == 192
    assert (tmp_path / "oos" / "strict_oof_predictions.parquet").is_file()
    assert (tmp_path / "oos" / "causal_21d_admission_metrics.parquet").is_file()


def test_existing_strict_oof_boundary_exposes_v2_dispatch() -> None:
    # The concrete behavior is covered by the strict-OOF integration test; the
    # established module must expose the v2 entrypoint without an import cycle.
    assert callable(generate_stage_i_target_adapter_strict_oof)


def test_existing_winner_boundary_freezes_v2_bundle(tmp_path) -> None:
    _, long_cell = _input("long")
    _, short_cell = _input("short")
    bundle = StageIAdapterWinnerBundle(
        cells=(long_cell, short_cell), code_revision="test-revision",
    )
    destination = tmp_path / "winner.json"
    assert freeze_stage_i_target_adapter_winner_bundle(bundle, destination) == "created_immutable_bundle"
    assert freeze_stage_i_target_adapter_winner_bundle(bundle, destination) == "reused_verified_immutable_bundle"
    assert callable(build_stage_i_target_adapter_winner_bundle)
