from __future__ import annotations

import pandas as pd

from scripts.materialize_meta_bme_feature_selection_sample import materialize_sample


def test_materialize_sample_preserves_exact_keys_and_temporal_bands(tmp_path) -> None:
    rows = []
    for pos in range(900):
        rows.append(
            {
                "__ts__": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=pos),
                "__symbol__": f"S{pos % 17}",
                "side_name": "long" if pos % 2 == 0 else "short",
                "score": float(pos) / 900.0,
                "selected_top30": True,
                "feature_x": float(pos),
            }
        )
    handoff = pd.DataFrame(rows)
    ledger = handoff.loc[
        :, ["__ts__", "__symbol__", "side_name", "score", "selected_top30"]
    ].copy()
    ledger["exec_margin"] = 0.01
    handoff_path = tmp_path / "handoff.parquet"
    ledger_path = tmp_path / "ledger.parquet"
    handoff.to_parquet(handoff_path, index=False)
    ledger.to_parquet(ledger_path, index=False)

    manifest = materialize_sample(
        handoff=handoff_path,
        ledger=ledger_path,
        out_dir=tmp_path / "sample",
        rows=300,
        seed=7,
    )

    sampled_handoff = pd.read_parquet(manifest["sampled_handoff"])
    sampled_ledger = pd.read_parquet(manifest["sampled_ledger"])
    keys = ["__ts__", "__symbol__", "side_name"]
    assert len(sampled_handoff) == 300
    assert len(sampled_ledger) == 300
    assert sampled_handoff[keys].equals(sampled_ledger[keys])
    assert [band["rows"] for band in manifest["bands"]] == [100, 100, 100]
