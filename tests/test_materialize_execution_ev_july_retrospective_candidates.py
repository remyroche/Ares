from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.base_candidate_population import deterministic_candidate_ids


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_july_retrospective_candidates",
    ROOT / "scripts" / "materialize_execution_ev_july_retrospective_candidates.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _contracts() -> dict[str, dict[str, list[str]]]:
    return {
        "long": {
            "base": ["shared", "long_only"],
            "residual": ["base_prediction", "shared"],
            "clean_favorable_event": ["shared"],
            "peak_mfe_conditional": ["long_only", "base_margin_to_cutoff"],
            "path_catboost": ["shared"],
            "final_head_preentry": ["existing_alpha_ev"],
        },
        "short": {
            "base": ["shared", "short_only"],
            "residual": ["base_prediction", "short_only"],
            "clean_favorable_event": ["shared"],
            "peak_mfe_conditional": ["short_only"],
            "path_catboost": ["shared"],
            "final_head_preentry": ["existing_alpha_ev"],
        },
    }


def _surface(
    tmp_path: Path,
    *,
    omit: str | None = None,
    nonfinite: bool = False,
    drop_last_timestamp: bool = False,
    extra_columns: list[str] | None = None,
) -> Path:
    features = tmp_path / "features"
    features.mkdir()
    timestamps = pd.to_datetime(
        ["2026-07-20T00:00:00Z", "2026-07-20T01:00:00Z"], utc=True
    )
    for number, symbol in enumerate(("AAA/USD:USD", "BBB/USD:USD"), start=1):
        values = {
            column: [float(number), float(number + 1)]
            for column in (extra_columns or [])
        }
        values.update(
            {
                "__symbol__": symbol,
                "shared": [float(number), float(number + 1)],
                "long_only": [2.0, 3.0],
                "short_only": [4.0, 5.0],
            }
        )
        frame = pd.DataFrame(values, index=timestamps)
        if omit:
            frame = frame.drop(columns=[omit])
        if nonfinite and symbol == "BBB/USD:USD":
            frame.loc[timestamps[1], "short_only"] = float("nan")
        if drop_last_timestamp:
            frame = frame.iloc[:-1]
        frame.to_parquet(features / f"symbol={symbol.replace('/', '_')}.parquet")
    return features


def test_materializes_complete_side_local_candidate_surface(tmp_path: Path) -> None:
    output = tmp_path / "output"
    result = materializer.materialize(
        features_dir=_surface(tmp_path),
        start=pd.Timestamp("2026-07-20T00:00:00Z"),
        end_exclusive=pd.Timestamp("2026-07-20T02:00:00Z"),
        output_dir=output,
        contracts=_contracts(),
    )
    candidates = pd.read_parquet(result["candidates"])
    manifest = json.loads(result["source_manifest"].read_text())
    coverage = pd.read_csv(result["hourly_coverage"])

    assert len(candidates) == 8
    assert set(candidates["side_name"]) == {"long", "short"}
    assert candidates["execution_decision_utc"].eq(
        candidates["__ts__"] + pd.Timedelta(hours=1)
    ).all()
    assert candidates["feature_available_at"].eq(candidates["__ts__"]).all()
    assert candidates["feature_available_at"].le(
        candidates["execution_decision_utc"]
    ).all()
    assert candidates["candidate_id"].equals(
        deterministic_candidate_ids(candidates, timeframe="1h")
    )
    assert candidates[["shared", "long_only", "short_only"]].notna().all().all()
    assert coverage["complete"].tolist() == [True, True]
    assert manifest["status"] == "materialized_retrospective_non_promotable"
    assert manifest["outcomes_used"] is False
    assert manifest["contracts"]["raw_static_columns_by_side"]["long"] == [
        "long_only",
        "shared",
    ]


def test_frozen_side_local_representation_is_not_requested_from_static_store() -> None:
    contracts = _contracts()
    frozen_representation = {
        "dae_b16_00",
        "dae_b16_02",
        "dae_b16_04",
        "dae_b16_08",
        "dae_b16_14",
        "expected_mahalanobis",
        "gmm_cluster_posterior_4",
        "gmm_dist_center_4",
        "gmm_dist_center_9",
        "gmm_ood_score",
        "gmm_representation_available",
    }
    contracts["long"]["path_catboost"].extend(sorted(frozen_representation))
    contracts["short"]["peak_mfe_conditional"].extend(sorted(frozen_representation))

    requirements = materializer.static_requirements(contracts)

    assert frozen_representation.isdisjoint(requirements["long"])
    assert frozen_representation.isdisjoint(requirements["short"])
    assert frozen_representation.issubset(contracts["long"]["path_catboost"])
    assert frozen_representation.issubset(contracts["short"]["peak_mfe_conditional"])


def test_real_frozen_ae_raw_contracts_materialize_and_are_hash_bound(
    tmp_path: Path,
) -> None:
    features_by_side, evidence = materializer.load_frozen_ae_contracts()
    contracts = _contracts()
    for side in ("long", "short"):
        contracts[side]["frozen_ae_raw"] = features_by_side[side]
    raw_union = sorted(set(features_by_side["long"]).union(features_by_side["short"]))

    result = materializer.materialize(
        features_dir=_surface(tmp_path, extra_columns=raw_union),
        start=pd.Timestamp("2026-07-20T00:00:00Z"),
        end_exclusive=pd.Timestamp("2026-07-20T02:00:00Z"),
        output_dir=tmp_path / "with_ae",
        contracts=contracts,
        frozen_ae_evidence=evidence,
    )

    candidates = pd.read_parquet(result["candidates"])
    manifest = json.loads(result["source_manifest"].read_text())
    assert len(features_by_side["long"]) == 256
    assert len(features_by_side["short"]) == 256
    assert len(raw_union) == 263
    assert set(raw_union).issubset(candidates.columns)
    assert manifest["frozen_side_local_ae_gmm"]["raw_feature_union_count"] == 263
    assert manifest["frozen_side_local_ae_gmm"]["outcomes_used"] is False
    assert manifest["frozen_side_local_ae_gmm"]["sides"]["long"]["state"]["sha256"] == (
        "13bbdf2f3d2d4acd23ccc859d1f98cd87323decd4c498aaca7f8a752447de4c9"
    )
    assert manifest["frozen_side_local_ae_gmm"]["sides"]["short"]["state"]["sha256"] == (
        "1bcf7048e542392b1ae02aac5cd09991e9d897e3137b6e954b75dff04de1d5cb"
    )


def test_missing_frozen_ae_raw_input_fails_closed(tmp_path: Path) -> None:
    features_by_side, evidence = materializer.load_frozen_ae_contracts()
    contracts = _contracts()
    for side in ("long", "short"):
        contracts[side]["frozen_ae_raw"] = features_by_side[side]
    raw_union = sorted(set(features_by_side["long"]).union(features_by_side["short"]))
    missing = raw_union[0]
    output = tmp_path / "missing_ae"

    with pytest.raises(materializer.CandidateSurfacePreflightError, match="incomplete"):
        materializer.materialize(
            features_dir=_surface(
                tmp_path,
                extra_columns=raw_union,
                omit=missing,
            ),
            start=pd.Timestamp("2026-07-20T00:00:00Z"),
            end_exclusive=pd.Timestamp("2026-07-20T02:00:00Z"),
            output_dir=output,
            contracts=contracts,
            frozen_ae_evidence=evidence,
        )

    manifest = json.loads((output / "source_manifest.json").read_text())
    incomplete = json.loads((output / "incomplete_source_rows.json").read_text())
    assert manifest["status"] == "blocked_incomplete_point_in_time_static_surface"
    assert manifest["candidates_written"] is False
    assert not (output / "candidate_features.parquet").exists()
    assert any(f"missing_column:{missing}" in row["reasons"] for row in incomplete["rows"])


def test_accepts_frozen_line_delimited_universe_without_losing_first_symbol(
    tmp_path: Path,
) -> None:
    features = _surface(tmp_path)
    universe = tmp_path / "universe.txt"
    universe.write_text(
        "# frozen inference-admissible universe\nAAA/USD:USD\nBBB/USD:USD\n",
        encoding="utf-8",
    )

    selected = materializer._load_universe(features, universe)

    assert [symbol for symbol, _ in selected] == ["AAA/USD:USD", "BBB/USD:USD"]


def test_fails_closed_and_persists_exact_missing_contract_report(tmp_path: Path) -> None:
    output = tmp_path / "blocked"
    with pytest.raises(materializer.CandidateSurfacePreflightError, match="incomplete"):
        materializer.materialize(
            features_dir=_surface(tmp_path, omit="short_only"),
            start=pd.Timestamp("2026-07-20T00:00:00Z"),
            end_exclusive=pd.Timestamp("2026-07-20T02:00:00Z"),
            output_dir=output,
            contracts=_contracts(),
        )
    manifest = json.loads((output / "source_manifest.json").read_text())
    incomplete = json.loads((output / "incomplete_source_rows.json").read_text())
    assert manifest["status"] == "blocked_incomplete_point_in_time_static_surface"
    assert manifest["candidates_written"] is False
    assert not (output / "candidate_features.parquet").exists()
    assert any("missing_column:short_only" in row["reasons"] for row in incomplete["rows"])


def test_rejects_nonfinite_required_source_values_without_fill(tmp_path: Path) -> None:
    output = tmp_path / "blocked_nonfinite"
    with pytest.raises(materializer.CandidateSurfacePreflightError):
        materializer.materialize(
            features_dir=_surface(tmp_path, nonfinite=True),
            start=pd.Timestamp("2026-07-20T00:00:00Z"),
            end_exclusive=pd.Timestamp("2026-07-20T02:00:00Z"),
            output_dir=output,
            contracts=_contracts(),
        )
    incomplete = json.loads((output / "incomplete_source_rows.json").read_text())
    assert any("nonfinite:short_only" in row["reasons"] for row in incomplete["rows"])


def test_rejects_missing_timestamp_and_reports_zero_source_rows(tmp_path: Path) -> None:
    output = tmp_path / "blocked_missing_timestamp"
    with pytest.raises(materializer.CandidateSurfacePreflightError):
        materializer.materialize(
            features_dir=_surface(tmp_path, drop_last_timestamp=True),
            start=pd.Timestamp("2026-07-20T00:00:00Z"),
            end_exclusive=pd.Timestamp("2026-07-20T02:00:00Z"),
            output_dir=output,
            contracts=_contracts(),
        )
    coverage = pd.read_csv(output / "hourly_raw_coverage.csv")
    incomplete = json.loads((output / "incomplete_source_rows.json").read_text())
    assert coverage["source_symbol_rows"].tolist() == [2, 0]
    assert any("missing_timestamp" in row["reasons"] for row in incomplete["rows"])
