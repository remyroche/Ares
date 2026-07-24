from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.features_gmm_ae import (
    ae_gmm_input_feature_order_hash,
    ae_gmm_learned_transform_hash,
    save_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)
from extreme_price_movements.static_feature_store import append_static_features
from scripts.verify_base_frozen_aegmm_static_store_parity import (
    _read_sidecar_sample,
    main,
    verify_base_frozen_aegmm_static_store_parity,
)


def _state() -> dict:
    latent_dim = 16
    state = {
        "enabled": True,
        "schema_version": "ae_gmm_v1",
        "feature_columns": ["f0", "f1"],
        "center": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "cycle_input_fill_values": {"f0": 0.0, "f1": 0.0},
        "ae_state": {"models": {}},
        "gmm_n_components": 2,
        "gmm_covariance_type": "diag",
        "gmm_numeric_contract": "fixed_order_rowwise_v1",
        "gmm_weights": [0.4, 0.6],
        "gmm_means": [[0.0] * latent_dim, [1.0] * latent_dim],
        "gmm_covariances": [[1.0] * latent_dim, [2.0] * latent_dim],
        "temporal_feature_contract": "row_independent_v1",
        "smooth_lambda": 0.0,
    }
    state["input_feature_order_hash"] = ae_gmm_input_feature_order_hash(
        state["feature_columns"]
    )
    state["cycle_state_hash"] = ae_gmm_learned_transform_hash(state)
    return state


def _fixture(
    tmp_path: Path,
    *,
    omit_f1: bool = False,
    partial_missing_f1: bool = False,
) -> tuple[Path, Path, Path, Path]:
    store_ts = pd.Timestamp("2026-07-01T12:00:00Z")
    index = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    symbols = ["AAA/USD:USD", "BBB/USD:USD"]
    f0 = pd.DataFrame(
        np.arange(24, dtype=np.float32).reshape(12, 2) / 10.0,
        index=index,
        columns=symbols,
    )
    f1 = pd.DataFrame(
        np.arange(24, dtype=np.float32).reshape(12, 2) / 20.0 + 1.0,
        index=index,
        columns=symbols,
    )
    if partial_missing_f1:
        f1.iloc[0, 0] = np.nan
    features = {"f0": f0}
    if not omit_f1:
        features["f1"] = f1
    append_static_features(
        features,
        feature_store_ts=store_ts,
        data_root=tmp_path,
        index=index,
        columns=symbols,
        source="test",
    )
    rows = pd.DataFrame(
        [
            (timestamp, symbol, side)
            for timestamp in index
            for symbol in symbols
            for side in (-1, 1)
        ],
        columns=["__ts__", "__symbol__", "side"],
    )
    # These decoys prove the verifier does not source frozen inputs from labels.
    rows["f0"] = -999.0
    rows["f1"] = -999.0
    labels_path = tmp_path / "labels.parquet"
    rows.sample(frac=1.0, random_state=17).to_parquet(labels_path, index=False)

    state_path = tmp_path / "state.pkl"
    state = _state()
    save_ae_gmm_state_artifact(state, state_path)
    key_pairs = zip(rows["__ts__"], rows["__symbol__"])
    raw = pd.DataFrame(
        {
            "f0": [f0.loc[timestamp, symbol] for timestamp, symbol in key_pairs],
            "f1": [
                f1.loc[timestamp, symbol]
                for timestamp, symbol in zip(rows["__ts__"], rows["__symbol__"])
            ],
        },
        dtype=np.float32,
    )
    generated = transform_ae_gmm_features(raw, state)[["gmm_prob_0", "gmm_entropy"]]
    sidecar = pd.concat(
        [rows[["__ts__", "__symbol__", "side"]].reset_index(drop=True), generated], axis=1
    )
    sidecar_path = tmp_path / "outputs.parquet"
    sidecar.to_parquet(sidecar_path, index=False)
    feature_store = tmp_path / "features" / store_ts.strftime("%Y%m%d_%H%M%S")
    return labels_path, feature_store, state_path, sidecar_path


def test_verifier_uses_static_store_and_deterministic_bme_keys(tmp_path: Path) -> None:
    labels, feature_store, state, sidecar = _fixture(tmp_path)

    report = verify_base_frozen_aegmm_static_store_parity(
        labels_path=labels,
        feature_store_path=feature_store,
        state_path=state,
        sidecar_path=sidecar,
        sample_rows=9,
    )

    assert report["pass"] is True
    assert report["sampling"]["sample_rows"] == 9
    assert report["static_input_loader"]["reader"] == "static_feature_store.read_static_features"
    assert report["raw_input_availability"]["f0"]["finite_rows"] == 9
    assert report["generated_output_differences"]["gmm_entropy"]["max_abs_diff"] == 0.0


def test_verifier_fails_closed_when_static_raw_input_is_missing(tmp_path: Path) -> None:
    labels, feature_store, state, sidecar = _fixture(tmp_path, omit_f1=True)

    report = verify_base_frozen_aegmm_static_store_parity(
        labels_path=labels,
        feature_store_path=feature_store,
        state_path=state,
        sidecar_path=sidecar,
        sample_rows=9,
    )

    assert report["pass"] is False
    assert "f1" in report["missing_or_nonfinite_raw_inputs"]
    assert "missing_or_all_nonfinite_static_raw_inputs" in report["errors"]


def test_verifier_preserves_frozen_transform_missing_value_contract(tmp_path: Path) -> None:
    labels, feature_store, state, sidecar = _fixture(tmp_path, partial_missing_f1=True)

    report = verify_base_frozen_aegmm_static_store_parity(
        labels_path=labels,
        feature_store_path=feature_store,
        state_path=state,
        sidecar_path=sidecar,
        sample_rows=128,
    )

    assert report["pass"] is True
    assert report["raw_input_availability"]["f1"]["missing_rows"] == 2
    assert report["generated_output_differences"]["gmm_entropy"]["within_tolerance"] is True


def test_cli_returns_nonzero_when_static_raw_input_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    labels, feature_store, state, sidecar = _fixture(tmp_path, omit_f1=True)
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_base_frozen_aegmm_static_store_parity.py",
            "--labels",
            str(labels),
            "--feature-store",
            str(feature_store),
            "--state",
            str(state),
            "--sidecar",
            str(sidecar),
            "--output",
            str(report_path),
        ],
    )

    assert main() == 1
    assert bool(pd.read_json(report_path, typ="series")["pass"]) is False


def test_verifier_fails_when_sidecar_output_exceeds_tolerance(tmp_path: Path) -> None:
    labels, feature_store, state, sidecar = _fixture(tmp_path)
    corrupted = pd.read_parquet(sidecar)
    corrupted.loc[0, "gmm_prob_0"] += np.float32(1e-3)
    corrupted.to_parquet(sidecar, index=False)

    report = verify_base_frozen_aegmm_static_store_parity(
        labels_path=labels,
        feature_store_path=feature_store,
        state_path=state,
        sidecar_path=sidecar,
        sample_rows=128,
    )

    assert report["pass"] is False
    assert report["generated_output_differences"]["gmm_prob_0"]["mismatch_rows"] == 1
    assert "output_mismatch:gmm_prob_0" in report["errors"]


def test_sidecar_reader_preserves_case_distinct_output_aliases(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-07-01T00:00:00Z")],
            "__symbol__": ["AAA/USD:USD"],
            "side": [1],
            "AE_reconstruction_error": [0.125],
            "ae_reconstruction_error": [0.25],
        }
    )
    path = tmp_path / "case_aliases.parquet"
    rows.to_parquet(path, index=False)

    loaded = _read_sidecar_sample(
        path,
        rows[["__ts__", "__symbol__", "side"]],
        ["AE_reconstruction_error", "ae_reconstruction_error"],
    )

    assert loaded["AE_reconstruction_error"].iloc[0] == 0.125
    assert loaded["ae_reconstruction_error"].iloc[0] == 0.25
