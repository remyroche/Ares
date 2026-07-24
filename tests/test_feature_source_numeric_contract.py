from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.feature_transform_contract import (
    FLOAT16_CLIPPED_THEN_FLOAT32_V1,
    FeatureSourceContract,
    apply_model_input_numeric_contract,
    build_model_input_numeric_contract,
    compare_model_matrices_exact,
    file_sha256,
    model_matrix_hash,
)
from extreme_price_movements.inference.feature_parity import (
    FeatureParityError,
    validate_feature_source_contract,
    validate_historical_model_matrix_exact,
)


def _source_contract(
    root: Path, *, oi_unit: str = "quote_notional"
) -> FeatureSourceContract:
    feature_file = root / "symbol=A.parquet"
    feature_file.write_bytes(b"immutable feature bytes")
    return FeatureSourceContract.create(
        run_id="run-1",
        source_root=root,
        market_mode="perps",
        feature_names=["f1", "f2"],
        model_feature_names=["f2", "f1"],
        universe_symbols=["A"],
        symbol_file_map={"A": feature_file.name},
        file_records={
            feature_file.name: {
                "sha256": file_sha256(feature_file),
                "rows": 100,
                "first_ts": "2026-01-01T00:00:00Z",
                "last_ts": "2026-02-01T00:00:00Z",
            }
        },
        source_start_ts="2026-01-01T00:00:00Z",
        source_end_ts="2026-02-01T00:00:00Z",
        required_warmup_hours=24,
        semantics={"open_interest_unit": oi_unit},
    )


def test_feature_source_contract_binds_files_symbols_warmup_and_oi_units(
    tmp_path,
) -> None:
    contract = _source_contract(tmp_path)
    report = validate_feature_source_contract(
        contract,
        run_id="run-1",
        model_feature_names=["f2", "f1"],
        symbols=["A"],
        end_ts="2026-01-31T00:00:00Z",
        source_root=tmp_path,
        symbol_file_map={"A": "symbol=A.parquet"},
    )
    assert report["ok"] is True

    (tmp_path / "symbol=A.parquet").write_bytes(b"mutated")
    with pytest.raises(FeatureParityError, match="Immutable feature source parity"):
        validate_feature_source_contract(
            contract,
            run_id="run-1",
            model_feature_names=["f2", "f1"],
            symbols=["A"],
            end_ts="2026-01-31T00:00:00Z",
            source_root=tmp_path,
        )


def test_feature_source_contract_rejects_native_open_interest(tmp_path) -> None:
    contract = _source_contract(tmp_path, oi_unit="native_contracts")
    with pytest.raises(FeatureParityError) as error:
        validate_feature_source_contract(
            contract,
            run_id="run-1",
            model_feature_names=["f2", "f1"],
            symbols=["A"],
            end_ts="2026-01-31T00:00:00Z",
            source_root=tmp_path,
        )
    assert (
        "feature_source_open_interest_unit_mismatch"
        in error.value.report["global_errors"]
    )


def test_feature_source_contract_rejects_universe_and_oi_semantics_drift(
    tmp_path,
) -> None:
    contract = _source_contract(tmp_path)
    with pytest.raises(FeatureParityError) as universe_error:
        validate_feature_source_contract(
            contract,
            run_id="run-1",
            model_feature_names=["f2", "f1"],
            symbols=["A", "B"],
            end_ts="2026-01-31T00:00:00Z",
            source_root=tmp_path,
        )
    assert universe_error.value.report["missing_symbols"] == ["B"]

    with pytest.raises(FeatureParityError) as map_error:
        validate_feature_source_contract(
            contract,
            run_id="run-1",
            model_feature_names=["f2", "f1"],
            symbols=["A"],
            end_ts="2026-01-31T00:00:00Z",
            source_root=tmp_path,
            symbol_file_map={"A": "symbol=A.parquet", "B": "extra.parquet"},
        )
    assert (
        "feature_source_symbol_file_map_mismatch"
        in map_error.value.report["global_errors"]
    )

    with pytest.raises(FeatureParityError) as semantics_error:
        validate_feature_source_contract(
            contract,
            run_id="run-1",
            model_feature_names=["f2", "f1"],
            symbols=["A"],
            end_ts="2026-01-31T00:00:00Z",
            source_root=tmp_path,
            expected_semantics={
                "open_interest_conversion": "native_contracts_x_mark_then_close"
            },
        )
    assert (
        "feature_source_semantics_mismatch:open_interest_conversion"
        in semantics_error.value.report["global_errors"]
    )


def test_feature_source_contract_rejects_per_symbol_stale_or_short_warmup(
    tmp_path,
) -> None:
    contract = _source_contract(tmp_path)
    contract.file_records["symbol=A.parquet"]["first_ts"] = "2026-01-31T12:00:00Z"
    contract.file_records["symbol=A.parquet"]["last_ts"] = "2026-01-31T22:00:00Z"
    contract.contract_hash = ""
    # Recreate seals after intentionally building a valid but incomplete source.
    incomplete = FeatureSourceContract.create(
        run_id=contract.run_id,
        source_root=contract.source_root,
        market_mode=contract.market_mode,
        feature_names=contract.feature_names,
        model_feature_names=contract.model_feature_names,
        universe_symbols=contract.universe_symbols,
        symbol_file_map=contract.symbol_file_map,
        file_records=contract.file_records,
        source_start_ts="2026-01-01T00:00:00Z",
        source_end_ts="2026-02-01T00:00:00Z",
        required_warmup_hours=24,
        semantics={"open_interest_unit": "quote_notional"},
    )
    with pytest.raises(FeatureParityError) as error:
        validate_feature_source_contract(
            incomplete,
            run_id="run-1",
            model_feature_names=["f2", "f1"],
            symbols=["A"],
            end_ts="2026-01-31T23:00:00Z",
            source_root=tmp_path,
        )
    assert (
        "feature_source_file_timestamp_incomplete"
        in error.value.report["global_errors"]
    )
    assert (
        "feature_source_file_warmup_incomplete" in error.value.report["global_errors"]
    )


def test_numeric_contract_reproduces_matrix_and_finds_first_divergence() -> None:
    raw = pd.DataFrame(
        {"f1": [0.1234567, 1.75e14], "f2": [-2.25, -1.75e14]},
        index=["r0", "r1"],
    )
    expected = pd.DataFrame(
        np.clip(
            raw.to_numpy(dtype=np.float32),
            -np.float32(np.finfo(np.float16).max),
            np.float32(np.finfo(np.float16).max),
        )
        .astype(np.float16)
        .astype(np.float32),
        index=raw.index,
        columns=raw.columns,
    )
    actual = apply_model_input_numeric_contract(
        raw, build_model_input_numeric_contract(raw.columns).asdict()
    )
    report = validate_historical_model_matrix_exact(expected, actual)
    assert report["ok"] is True
    assert report["expected_matrix_hash"] == report["actual_matrix_hash"]

    divergent = actual.copy()
    divergent.loc["r1", "f2"] += np.float32(1.0)
    report = compare_model_matrices_exact(expected, divergent)
    assert report["first_divergence"] == {
        "row_position": 1,
        "feature": "f2",
        "expected": -65504.0,
        "actual": -65503.0,
        "abs_delta": 1.0,
        "index": "r1",
    }


def test_numeric_contract_rejects_feature_reordering_and_mutation() -> None:
    frame = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    payload = build_model_input_numeric_contract(frame.columns).asdict()
    with pytest.raises(ValueError, match="feature order"):
        apply_model_input_numeric_contract(frame[["f2", "f1"]], payload)
    payload["clip_abs"] = 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_model_input_numeric_contract(frame, payload)


def test_numeric_contract_dataclass_binds_feature_order_and_reference_hash() -> None:
    frame = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    frozen_hash = model_matrix_hash(frame)
    contract = build_model_input_numeric_contract(
        frame.columns, reference_matrix_hash=frozen_hash
    )
    with pytest.raises(ValueError, match="feature order"):
        apply_model_input_numeric_contract(frame[["f2", "f1"]], contract)
    report = validate_historical_model_matrix_exact(
        frame, frame.copy(), reference_matrix_hash=frozen_hash
    )
    assert report["reference_matrix_hash"] == frozen_hash
    with pytest.raises(FeatureParityError) as error:
        validate_historical_model_matrix_exact(
            frame, frame.copy(), reference_matrix_hash="sha256:wrong"
        )
    assert error.value.report["error"] == "reference_matrix_hash_mismatch"


def test_saved_historical_july_matrix_exactly_matches_numeric_contract() -> None:
    root = Path(
        "data_perp/reports/"
        "s59_h5_fullthroughjul10_base_configfull_freshmda_fixedparams_wf30_20260713/"
        "_fold_cache/2026-06-26_2026-07-26"
    )
    matrix_path = root / "x_valid.parquet"
    row_path = root / "valid.parquet"
    if not matrix_path.is_file() or not row_path.is_file():
        pytest.skip("Historical July base fold is not present")
    matrix = pd.read_parquet(matrix_path)
    row_ids = pd.read_parquet(
        row_path, columns=["__ts__", "__symbol__", "side_name", "side"]
    )
    reproduced = apply_model_input_numeric_contract(
        matrix, FLOAT16_CLIPPED_THEN_FLOAT32_V1
    )
    frozen_hash = (
        "sha256:eaaf85559622978c4f90d39906a73b8b0949f584f68fd927bd4a413d7565a34c"
    )
    report = validate_historical_model_matrix_exact(
        matrix,
        reproduced,
        row_ids=row_ids,
        model_key="july_base_oos",
        reference_matrix_hash=frozen_hash,
    )
    assert report["rows"] == 104_560
    assert report["features"] == 84
    assert model_matrix_hash(matrix, row_ids=row_ids) == frozen_hash
