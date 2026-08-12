from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.stage_ii_enriched_materializer import (
    StageIIEnrichedMaterializationError,
    materialize_stage_ii_enriched_ledger,
)
from extreme_price_movements.stage_ii_execution import validate_enriched_ledger_manifest


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    stage = tmp_path / "stage_i"
    stage.mkdir()
    _write_json(stage / "manifest.json", {"schema": "stage_i_production_winner_oos_v1", "status": "complete"})
    signal = pd.to_datetime(["2024-06-01", "2025-01-15", "2025-03-01"], utc=True)
    predictions = pd.DataFrame({
        "candidate_id": ["a", "b", "c"], "side_name": ["long", "short", "long"],
        "symbol": ["A", "B", "C"], "decision_ts": signal + pd.Timedelta(hours=1),
        "label_available_ts": signal + pd.Timedelta(hours=13), "base_strict_oof_available": [True] * 3,
    })
    predictions.to_parquet(stage / "full_history_raw_oof_predictions.parquet", index=False)
    context = tmp_path / "selected_panel"
    context.mkdir()
    predictions.assign(meta_context=1.0).loc[:, ["candidate_id", "side_name", "symbol", "decision_ts", "meta_context"]].to_parquet(context / "panel.parquet", index=False)
    spec = _write_json(tmp_path / "candidate.json", {
        "meta_feature_cols": ["meta_context"],
        "candidates": [{"causal_feature_cols": [], "config": {"path_descriptor_cols": ["path_arch_peak_mfe_atr"]}}],
    })
    sources = []
    windows = [
        ("historical_2024", "canonical_parquet", "2024-01-01", "2025-01-01", 0),
        ("native_january_2025", "native_path_descriptors", "2025-01-01", "2025-02-01", 1),
        ("path_archetype_2025_plus", "canonical_parquet", "2025-02-01", "2025-04-01", 2),
    ]
    for name, kind, start, end, position in windows:
        row = predictions.iloc[[position]].copy()
        row["__ts__"] = signal[position]
        row["__decision_ts__"] = row.decision_ts
        row["__label_end_ts__"] = row.label_available_ts
        row["__symbol__"] = row.symbol
        row["path_arch_peak_mfe_atr"] = 2.0
        source_path = tmp_path / f"{name}.parquet"
        row.to_parquet(source_path, index=False)
        sources.append({
            "source_id": name, "kind": kind, "start_utc": f"{start}T00:00:00Z", "end_utc": f"{end}T00:00:00Z",
            "paths": [source_path.name],
            "columns": {"symbol": "__symbol__", "signal_close_ts": "__ts__", "decision_ts": "__decision_ts__", "label_available_ts": "__label_end_ts__"},
            "descriptor_mapping": {"path_arch_peak_mfe_atr": "path_arch_peak_mfe_atr"},
        })
    source_map = _write_json(tmp_path / "source_map.json", {"schema": "stage_ii_enriched_path_source_map_v1", "sources": sources})
    return stage, context, spec, source_map


def test_materializes_exact_three_source_enriched_ledger(tmp_path: Path) -> None:
    stage, context, spec, source_map = _setup(tmp_path)
    output = materialize_stage_ii_enriched_ledger(
        stage_i_oos_dir=stage, selected_panel=context, candidate_spec=spec,
        source_map=source_map, output_dir=tmp_path / "out",
    )
    ledger = output / "stage_ii_enriched_ledger.parquet"
    manifest = json.loads((output / "manifest.json").read_text())
    frame = pd.read_parquet(ledger)
    assert len(frame) == 3
    assert set(frame.columns) >= {"meta_context", "path_arch_peak_mfe_atr"}
    assert len(manifest["source_map"]["sources"]) == 3
    validate_enriched_ledger_manifest(
        manifest, ledger_path=ledger, required_causal_columns=["meta_context"],
        required_path_columns=["path_arch_peak_mfe_atr"],
    )
    assert materialize_stage_ii_enriched_ledger(
        stage_i_oos_dir=stage, selected_panel=context, candidate_spec=spec,
        source_map=source_map, output_dir=output, resume=True,
    ) == output


def test_rejects_path_coverage_gap_instead_of_filling(tmp_path: Path) -> None:
    stage, context, spec, source_map = _setup(tmp_path)
    later = tmp_path / "path_archetype_2025_plus.parquet"
    pd.read_parquet(later).iloc[0:0].to_parquet(later, index=False)
    with pytest.raises(
        StageIIEnrichedMaterializationError,
        match="do not cover every direct Stage-I base OOF identity",
    ):
        materialize_stage_ii_enriched_ledger(
            stage_i_oos_dir=stage, selected_panel=context, candidate_spec=spec,
            source_map=source_map, output_dir=tmp_path / "out",
        )


def test_scopes_full_history_stage_i_population_to_declared_source_interval(tmp_path: Path) -> None:
    stage, context, spec, source_map = _setup(tmp_path)
    predictions = pd.read_parquet(stage / "full_history_raw_oof_predictions.parquet")
    earlier = predictions.iloc[[0]].copy()
    earlier["candidate_id"] = "pre_stage_ii"
    earlier["decision_ts"] = pd.Timestamp("2023-12-15T01:00:00Z")
    earlier["label_available_ts"] = pd.Timestamp("2023-12-15T13:00:00Z")
    pd.concat([earlier, predictions], ignore_index=True).to_parquet(
        stage / "full_history_raw_oof_predictions.parquet", index=False,
    )

    output = materialize_stage_ii_enriched_ledger(
        stage_i_oos_dir=stage, selected_panel=context, candidate_spec=spec,
        source_map=source_map, output_dir=tmp_path / "out",
    )
    ledger = pd.read_parquet(output / "stage_ii_enriched_ledger.parquet")
    manifest = json.loads((output / "manifest.json").read_text())
    assert "pre_stage_ii" not in set(ledger.candidate_id)
    assert len(ledger) == 3
    assert manifest["stage_i_oos"]["source_interval"] == {
        "start_utc": "2024-01-01 00:00:00+00:00",
        "end_utc": "2025-04-01 00:00:00+00:00",
    }


def test_reads_target_specific_stage_i_features_with_separate_contract(tmp_path: Path) -> None:
    stage, context, spec, source_map = _setup(tmp_path)
    panel = pd.read_parquet(context / "panel.parquet")
    (context / "panel.parquet").unlink()
    side_dir = context / "long"
    side_dir.mkdir()
    panel.loc[:, ["candidate_id", "symbol", "decision_ts", "meta_context"]].rename(
        columns={"symbol": "__symbol__"}
    ).assign(__ts__=lambda frame: frame.decision_ts - pd.Timedelta(hours=1)).loc[
        :, ["candidate_id", "__ts__", "__symbol__", "meta_context"]
    ].to_parquet(side_dir / "features.parquet", index=False)
    panel.loc[:, ["candidate_id", "side_name", "symbol", "decision_ts"]].rename(
        columns={"symbol": "__symbol__"}
    ).assign(__ts__=lambda frame: frame.decision_ts - pd.Timedelta(hours=1)).loc[
        :, ["candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts"]
    ].to_parquet(side_dir / "contract.parquet", index=False)

    output = materialize_stage_ii_enriched_ledger(
        stage_i_oos_dir=stage, selected_panel=context, candidate_spec=spec,
        source_map=source_map, output_dir=tmp_path / "out",
    )
    assert len(pd.read_parquet(output / "stage_ii_enriched_ledger.parquet")) == 3
