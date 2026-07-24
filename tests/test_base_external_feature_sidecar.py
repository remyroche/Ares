from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _load_projected_labels,
)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
            "__symbol__": ["BTC", "ETH"],
            "side": [1, -1],
            "__target_soft__": [0.8, 0.2],
        }
    )


def test_projected_labels_join_external_representation_sidecar(tmp_path) -> None:
    labels_path = tmp_path / "labels.parquet"
    sidecar_path = tmp_path / "representations.parquet"
    _labels().to_parquet(labels_path, index=False)
    sidecar = _labels().loc[:, ["__ts__", "__symbol__", "side"]]
    sidecar["repr_entropy_norm"] = [0.25, 0.75]
    sidecar.to_parquet(sidecar_path, index=False)

    frame, contract = _load_projected_labels(
        labels_path,
        selected_features=["repr_entropy_norm"],
        ae_gmm_input_features=[],
        external_feature_sidecar_path=sidecar_path,
    )
    assert frame["repr_entropy_norm"].tolist() == [0.25, 0.75]
    assert contract["external_feature_sidecar"]["row_coverage"] == 1.0


def test_external_representation_sidecar_rejects_duplicate_keys(tmp_path) -> None:
    labels_path = tmp_path / "labels.parquet"
    sidecar_path = tmp_path / "representations.parquet"
    _labels().to_parquet(labels_path, index=False)
    sidecar = pd.concat([_labels().iloc[[0]], _labels().iloc[[0]]], ignore_index=True)
    sidecar["repr_entropy_norm"] = [0.25, 0.25]
    sidecar.to_parquet(sidecar_path, index=False)
    with pytest.raises(ValueError, match="unique"):
        _load_projected_labels(
            labels_path,
            selected_features=["repr_entropy_norm"],
            ae_gmm_input_features=[],
            external_feature_sidecar_path=sidecar_path,
        )


def test_small_label_sample_uses_key_filtered_sidecar_join(tmp_path) -> None:
    labels_path = tmp_path / "labels.parquet"
    sidecar_path = tmp_path / "representations.parquet"
    labels = _labels()
    labels.to_parquet(labels_path, index=False)
    sidecar = pd.concat(
        [
            pd.DataFrame(
                {
                    "__ts__": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
                    "__symbol__": ["BTC", "ETH"],
                    "side": [1, -1],
                    "repr_entropy_norm": [1.0, 0.0],
                }
            ),
            pd.DataFrame(
                {
                    "__ts__": pd.date_range("2026-01-01 01:00", periods=100, freq="h", tz="UTC"),
                    "__symbol__": ["BTC"] * 100,
                    "side": [1] * 100,
                    "repr_entropy_norm": list(range(2, 102)),
                }
            ),
        ],
        ignore_index=True,
    )
    sidecar.to_parquet(sidecar_path, index=False)

    frame, contract = _load_projected_labels(
        labels_path,
        selected_features=["repr_entropy_norm"],
        ae_gmm_input_features=[],
        external_feature_sidecar_path=sidecar_path,
    )

    assert frame["repr_entropy_norm"].tolist() == [1.0, 0.0]
    assert contract["external_feature_sidecar"]["join_mode"] == "key_filtered_duckdb"


def test_projected_labels_normalize_mixed_naive_and_aware_utc_shards(tmp_path) -> None:
    labels_path = tmp_path / "labels"
    labels_path.mkdir()
    first = _labels().iloc[[0]].copy()
    first["__ts__"] = first["__ts__"].dt.tz_localize(None)
    second = _labels().iloc[[1]].copy()
    first.to_parquet(
        labels_path / "train_global_long_5_2026_01.parquet", index=False
    )
    second.to_parquet(
        labels_path / "train_global_long_5_2026_02.parquet", index=False
    )

    sidecar_path = tmp_path / "representations.parquet"
    sidecar = _labels().loc[:, ["__ts__", "__symbol__", "side"]].copy()
    sidecar["repr_entropy_norm"] = [0.25, 0.75]
    sidecar.to_parquet(sidecar_path, index=False)

    frame, contract = _load_projected_labels(
        labels_path,
        selected_features=["repr_entropy_norm"],
        ae_gmm_input_features=[],
        external_feature_sidecar_path=sidecar_path,
    )

    assert frame["__ts__"].notna().all()
    assert str(frame["__ts__"].dtype) == "datetime64[ns, UTC]"
    assert frame["repr_entropy_norm"].tolist() == [0.25, 0.75]
    assert contract["external_feature_sidecar"]["row_coverage"] == 1.0
