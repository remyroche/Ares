from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _sha256_file,
    _source_files_signature,
    _validate_frozen_ae_gmm_output_sidecar,
)


def test_precomputed_frozen_sidecar_is_bound_to_labels_and_state(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    label_file = labels / "part.parquet"
    pd.DataFrame({"__ts__": [pd.Timestamp("2026-01-01", tz="UTC")]}).to_parquet(
        label_file, index=False
    )
    state = tmp_path / "state.pkl"
    state.write_bytes(b"frozen-state")
    sidecar = tmp_path / "outputs.parquet"
    pd.DataFrame({"gmm_entropy": [0.2]}).to_parquet(sidecar, index=False)
    manifest = {
        "source_signature": _source_files_signature([label_file]),
        "source_rows": 1,
        "state_sha256": _sha256_file(state),
        "output_features": ["gmm_entropy"],
    }
    sidecar.with_suffix(".manifest.json").write_text(json.dumps(manifest))

    validated = _validate_frozen_ae_gmm_output_sidecar(
        labels_path=labels, state_path=state, sidecar_path=sidecar
    )
    assert validated["status"] == "validated_precomputed"

    state.write_bytes(b"different-state")
    with pytest.raises(RuntimeError, match="state_sha256"):
        _validate_frozen_ae_gmm_output_sidecar(
            labels_path=labels, state_path=state, sidecar_path=sidecar
        )
