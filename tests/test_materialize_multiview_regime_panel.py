from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_multiview_regime_panel import (
    OUTPUT_FEATURES,
    materialize_multiview_regime_panel,
    parse_args,
    sha256,
)


def _write_signed_manifest(directory: Path) -> None:
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"schema": "synthetic_hourly_ledger"}) + "\n", encoding="utf-8")
    (directory / "manifest.sha256").write_text(
        f"{sha256(manifest)}  manifest.json\n", encoding="utf-8"
    )


def _root(root: Path, *, liquidity: bool = False) -> Path:
    root.mkdir()
    rows = 400
    values = np.arange(rows, dtype=float)
    source = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC").delete(200)
    values = np.delete(values, 200)
    panel = pd.DataFrame(
        {
            "source_utc": source,
            "calendar_segment_id": 1,
            "breadth": np.sin(values / 9.0),
            "funding": np.cos(values / 11.0),
            "correlation": np.sin(values / 17.0),
            "target__future_state": values,
            "state_context__current_state": 1.0,
        }
    )
    if liquidity:
        panel["volume_proxy"] = 1_000.0 + values
    panel.to_parquet(root / "hourly_state_calendar.parquet", index=False)
    _write_signed_manifest(root)
    return root


def test_materializes_signed_research_only_panel_with_no_gap_bridge(tmp_path: Path) -> None:
    source = _root(tmp_path / "ledger", liquidity=False)
    output = tmp_path / "out"
    report = materialize_multiview_regime_panel(input_path=source, output_dir=output)

    result = pd.read_parquet(output / OUTPUT_FEATURES)
    manifest = json.loads((output / "manifest.json").read_text())
    first_after_gap = result.index[result["source_utc"].eq(pd.Timestamp("2024-01-09 09:00:00+00:00"))][0]

    assert report["research_only"] is True
    assert manifest["input_contract"]["source_manifest"]["detached_checksum_verified"] is True
    assert manifest["multiview_contract"]["horizons"][-1] == "168h"
    assert manifest["families"]["dependence_covariance"] > 0
    assert manifest["families"]["liquidity_proxy"] == 0
    assert not any("target" in column or "state_context" in column for column in result.columns)
    assert pd.isna(result.loc[first_after_gap, "mv__breadth__delta_1h"])
    assert (output / "manifest.sha256").read_text().split()[0] == sha256(output / "manifest.json")


def test_uses_real_available_liquidity_proxy_and_rejects_explicit_targets(tmp_path: Path) -> None:
    source = _root(tmp_path / "ledger", liquidity=True)
    output = tmp_path / "out"
    report = materialize_multiview_regime_panel(input_path=source, output_dir=output)

    assert report["multiview_contract"]["liquidity_proxy_columns"] == ["volume_proxy"]
    assert report["families"]["liquidity_proxy"] > 0

    with pytest.raises(ValueError, match="forbidden"):
        materialize_multiview_regime_panel(
            input_path=source,
            output_dir=tmp_path / "bad",
            feature_columns=("target__future_state",),
        )


def test_cli_parses_direct_hourly_input_and_group_override(tmp_path: Path) -> None:
    parquet = _root(tmp_path / "ledger") / "hourly_state_calendar.parquet"
    args = parse_args(
        [
            "--input",
            str(parquet),
            "--output-dir",
            str(tmp_path / "out"),
            "--group-column",
            "calendar_segment_id",
            "--max-dependence-columns",
            "4",
        ]
    )

    assert args.input == parquet
    assert args.group_columns == ["calendar_segment_id"]
    assert args.max_dependence_columns == 4


def test_exact_signed_enrichment_adds_liquidity_without_changing_rows(
    tmp_path: Path,
) -> None:
    source = _root(tmp_path / "ledger")
    source_frame = pd.read_parquet(source / "hourly_state_calendar.parquet")
    enrichment_root = tmp_path / "enrichment"
    enrichment_root.mkdir()
    enrichment = source_frame[["source_utc", "calendar_segment_id"]].copy()
    enrichment["liquidity_market__median_spread_bps"] = np.arange(
        len(enrichment), dtype=float
    )
    enrichment_path = enrichment_root / "enrichment.parquet"
    enrichment.to_parquet(enrichment_path, index=False)
    _write_signed_manifest(enrichment_root)

    output = tmp_path / "out"
    report = materialize_multiview_regime_panel(
        input_path=source,
        output_dir=output,
        enrichment_path=enrichment_path,
    )
    result = pd.read_parquet(output / OUTPUT_FEATURES)

    assert len(result) == len(source_frame)
    assert report["input_contract"]["enrichment"]["join"].startswith("exact")
    assert report["families"]["liquidity_proxy"] > 0
    assert any(
        "median_spread_bps" in column and column.startswith("mv__liquidity__")
        for column in result.columns
    )
