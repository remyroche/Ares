from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_materialized_trailing_label_topk_lgbm_hpo as runner


def test_oos_model_input_parity_persists_side_specific_hashes_and_anchors(tmp_path) -> None:
    valid = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-04-01T02:00:00Z",
                    "2026-04-01T00:00:00Z",
                    "2026-04-01T01:00:00Z",
                ],
                utc=True,
            ),
            "__symbol__": ["BTC", "ETH", "BTC"],
            "side": [1, -1, 1],
        }
    )
    # This is the precise float16 -> float32 boundary seen by cached OOS scoring.
    x_valid = pd.DataFrame(
        np.asarray(
            [[0.1234567, 9.0, -2.0], [7.0, -0.3333333, 8.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        )
        .astype(np.float16)
        .astype(np.float32),
        columns=["long_a", "short_b", "shared_c"],
    )

    result = runner._persist_oos_model_input_parity(
        parity_root=tmp_path / "model_input_parity",
        fold="2026-04",
        valid=valid,
        x_valid=x_valid,
        valid_sides=np.asarray(["long", "short", "long"]),
        feature_contracts={"long": ["long_a", "shared_c"], "short": ["short_b"]},
        model_side_scope="per_side",
        anchor_rows=2,
    )

    hashes = pd.read_parquet(result["row_hashes_path"])
    anchors = pd.read_parquet(result["anchors_path"])
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert set(hashes["model_side"]) == {"long", "short"}
    assert hashes.groupby("model_side")["feature_contract_hash"].nunique().eq(1).all()
    assert (
        hashes.loc[hashes["model_side"] == "long", "feature_contract_hash"].iloc[0]
        == runner._feature_contract_hash(["long_a", "shared_c"])
    )
    assert (
        hashes.loc[hashes["model_side"] == "short", "feature_contract_hash"].iloc[0]
        == runner._feature_contract_hash(["short_b"])
    )
    assert hashes["model_input_row_hash"].str.len().eq(64).all()
    assert len(anchors) == 3
    np.testing.assert_array_equal(
        anchors.loc[anchors["model_side"] == "long", "long_a"].sort_values().to_numpy(),
        np.sort(x_valid.loc[[0, 2], "long_a"].to_numpy()),
    )
    assert manifest["model_side_scope"] == "per_side"
    assert (
        manifest["contracts_by_model_side"]["long"]["numeric_contract"]["name"]
        == "float16_clipped_then_float32_v1"
    )
    assert manifest["contracts_by_model_side"]["long"]["feature_names"] == [
        "long_a",
        "shared_c",
    ]
    assert manifest["contracts_by_model_side"]["short"]["feature_names"] == [
        "short_b"
    ]
