from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.base_candidate_population import (
    BaseCandidatePopulationContract,
    candidate_identity_sha256,
    deterministic_candidate_ids,
    select_base_candidate_population,
)
from extreme_price_movements.side_aware import candidate_id_series

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_base_candidate_population",
    ROOT / "scripts" / "materialize_base_candidate_population.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)


def test_selects_top_40_per_timestamp_and_side_deterministically() -> None:
    rows = []
    for timestamp in ("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"):
        for side in ("long", "short"):
            for index, score in enumerate((0.9, 0.8, 0.8, 0.6, 0.5)):
                rows.append(
                    {
                        "__ts__": timestamp,
                        "__symbol__": f"S{index}",
                        "side_name": side,
                        "score": score,
                    }
                )
    selected = select_base_candidate_population(pd.DataFrame(rows))
    assert len(selected) == 8
    assert selected.groupby(["__ts__", "side_name"]).size().eq(2).all()
    assert selected["selected_top40"].all()
    assert set(selected["__symbol__"]) == {"S0", "S1"}
    assert selected["candidate_handoff_rank_scope"].eq("timestamp_side").all()


def test_candidate_hash_is_order_invariant_and_population_sensitive() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "__symbol__": ["A", "B"],
            "side_name": ["long", "short"],
        }
    )
    assert candidate_identity_sha256(frame) == candidate_identity_sha256(frame.iloc[::-1])
    changed = frame.copy()
    changed.loc[0, "__symbol__"] = "C"
    assert candidate_identity_sha256(frame) != candidate_identity_sha256(changed)


def test_contract_rejects_invalid_fraction() -> None:
    frame = pd.DataFrame(
        {"__ts__": ["2026-01-01"], "__symbol__": ["A"], "side_name": ["long"], "score": [1.0]}
    )
    try:
        select_base_candidate_population(frame, BaseCandidatePopulationContract(top_fraction=1.0))
    except ValueError as exc:
        assert "top_fraction" in str(exc)
    else:
        raise AssertionError("invalid fraction was accepted")


def test_deterministic_candidate_ids_require_exact_identity() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "__symbol__": ["A", "B"],
            "side_name": ["long", "short"],
        }
    )
    assert deterministic_candidate_ids(frame).tolist() == deterministic_candidate_ids(
        frame.iloc[::-1]
    ).sort_index().tolist()
    with pytest.raises(ValueError, match="not unique"):
        deterministic_candidate_ids(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


def test_base_population_candidate_ids_match_existing_path_label_contract() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2025-02-01T00:00:00Z"], utc=True),
            "__symbol__": ["AAVE/USD:USD"],
            "side_name": ["long"],
        }
    )
    expected = "AAVE/USD:USD|2025-02-01T00:00:00Z|1h|long"
    assert deterministic_candidate_ids(frame).iloc[0] == expected
    assert deterministic_candidate_ids(frame).iloc[0] == candidate_id_series(
        frame["__ts__"], frame["__symbol__"], "1h", frame["side_name"]
    ).iloc[0]


def test_materializer_records_manifest_boundaries_without_fabricating_resolution_proof(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base-oof.parquet"
    pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "score": [0.9],
            "oos_fold": ["7"],
        }
    ).to_parquet(source, index=False)
    manifest_dir = tmp_path / "models" / "7"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fold": "7",
                "valid_start": "2026-01-01T00:00:00Z",
                "valid_end": "2026-01-01T23:00:00Z",
                "leakage_contract": {"fit_scope": "prior_rows_only"},
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "population"
    manifest = materializer.run(
        source,
        output_dir,
        BaseCandidatePopulationContract(),
        model_manifest_root=tmp_path / "models",
    )
    output = pd.read_parquet(output_dir / "base_candidate_population.parquet")
    assert output["candidate_id"].iloc[0] == "BTC/USD:USD|2026-01-01T00:00:00Z|1h|long"
    assert output["validation_start"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert pd.isna(output["train_decision_cutoff"].iloc[0])
    assert output["train_decision_cutoff_evidence"].iloc[0] == "not_observed_in_model_manifest"
    assert manifest["fold_provenance"]["strict_execution_ev_handoff"]["status"] == (
        "blocked_regenerate_upstream_evidence"
    )
