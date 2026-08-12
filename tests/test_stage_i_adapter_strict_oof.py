from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.stage_i_adapter_strict_oof import (
    StageIAdapterStrictOOFPlan,
    generate_stage_i_adapter_strict_oof,
)
from extreme_price_movements.stage_i_target_adapter import (
    FOLD_QUANTILE_RESIDUAL3,
    SOFT_SCALAR_S,
    bind_target_contract,
)


class _Regressor:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(0.5 + 0.4 * frame.iloc[:, 0].to_numpy(float), 0.0, 1.0)


class _Classifier:
    classes_ = np.array([0, 1, 2], dtype=np.int8)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        signal = np.tanh(frame.iloc[:, 0].to_numpy(float))
        return np.column_stack([0.3 - 0.1 * signal, np.full(len(frame), 0.4), 0.3 + 0.1 * signal])


def _fit(_x, _y, _weight, *, classifier, **_kwargs):
    return _Classifier() if classifier else _Regressor()


def test_adapter_oof_uses_candidate_only_meta_and_full_reference_scoring() -> None:
    n = 96
    signal_ts = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    decision = signal_ts + pd.Timedelta(hours=1)
    available = decision + pd.Timedelta(hours=12)
    signal = np.resize(np.asarray([-1.0, -0.4, 0.2, 0.8]), n)
    net = signal * 250.0
    gross = net + 100.0
    selected = np.arange(n) % 10 != 0
    contract_frame = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(n)],
        "__ts__": signal_ts,
        "__symbol__": np.resize(np.asarray(["BTC", "ETH"]), n),
        "side_name": "long",
        "base_target": np.clip((signal + 1.0) / 2.0, 0.0, 1.0),
        "meta_basis": net,
        "gross_bps": gross,
        "net_bps": net,
        "target_valid": True,
        "sample_weight": 1.0,
    })
    base_contract = bind_target_contract(
        contract_frame, family=SOFT_SCALAR_S, layer="base", target_name="S__sl4_tp6",
        geometry="sl4_tp6", target_columns=("base_target",),
    )
    meta_contract = bind_target_contract(
        contract_frame, family=FOLD_QUANTILE_RESIDUAL3, layer="meta",
        target_name="fold_quantile_residual3", geometry="sl4_tp6",
        target_columns=("meta_basis",),
    )
    feature_frame = pd.DataFrame({"base_feature": signal, "context_feature": signal})
    result = generate_stage_i_adapter_strict_oof(
        StageIAdapterStrictOOFPlan(
            side="long", frame=feature_frame, contract_frame=contract_frame,
            candidate_ids=contract_frame.candidate_id,
            decision_timestamps=decision, label_available_timestamps=available,
            base_target=contract_frame.base_target,
            exact_gross_bps=gross, exact_net_bps=net, target_valid=np.ones(n, bool),
            sample_weight=np.ones(n), base_target_contract=base_contract,
            meta_target_contract=meta_contract, base_feature_names=("base_feature",),
            meta_feature_names=("context_feature", "base_raw_score", "prequential_base_expected_net_bps"),
            base_params={}, meta_params={}, candidate_selected=selected,
            n_validation_folds=3, min_train_rows=12,
        ),
        fit_model=_fit,
    )
    prediction = result.predictions
    assert result.manifest["base_target_contract_sha256"] == base_contract.sha256
    assert result.manifest["meta_target_contract_sha256"] == meta_contract.sha256
    assert prediction.loc[~selected, "mapping_reference_only"].any()
    meta_rows = result.fold_provenance.loc[
        (result.fold_provenance.layer == "meta") & ~result.fold_provenance.skipped.fillna(False)
    ]
    assert not meta_rows.empty
    assert meta_rows.candidate_only_training.all()
    assert meta_rows.full_valid_reference_rows_scored.all()
    assert prediction.loc[prediction.strict_oof_available, "reconstructed_expected_net_bps"].notna().all()
    assert result.manifest["meta_fold_states"]
