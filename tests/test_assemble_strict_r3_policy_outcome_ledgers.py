from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "assemble_strict_r3_policy_outcome_ledgers.py"
SPEC = importlib.util.spec_from_file_location("assemble_strict_r3_policy_outcomes", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(candidate_id: str, ts: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id, "__decision_ts__": pd.Timestamp(ts),
        "policy_path_valid": True, "policy_gross_bps": 180.0,
        "policy_net_bps": 80.0, "policy_exit_bar_15m": 12,
        "policy_exit_reason": "TRAILING", "policy_entry_price": 100.0,
        "policy_exit_price": 101.8,
        "policy_label_available_ts": pd.Timestamp(ts) + pd.Timedelta(hours=4),
        "policy_outcome_source": "coarse_15m", "policy_cost_bps": 100.0,
    }


def _fragment(tmp_path: Path, name: str, rows: list[dict[str, object]], policy_hash: str = "same") -> Path:
    directory = tmp_path / name
    directory.mkdir()
    path = directory / "candidate_policy_outcomes.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    (directory / "run_manifest.json").write_text(json.dumps({
        "policy_json_sha256": policy_hash,
        "policy": {"sl_mult": 4.15, "trailing_activation_mult": 2.33, "fixed_trailing_gap_mult": 0.10},
    }))
    return path


def test_assembly_requires_exact_source_identity_and_one_policy(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2024-12-01T00:00:00Z", "2025-01-01T00:00:00Z"]),
        "side_name": ["long", "long"],
    }).to_parquet(source, index=False)
    first = _fragment(tmp_path, "first", [_row("a", "2024-12-01T00:00:00Z")])
    second = _fragment(tmp_path, "second", [_row("b", "2025-01-01T00:00:00Z")])
    output, audit = MODULE._assemble(
        source_panel=source, fragments=[first, second],
        start=pd.Timestamp("2024-12-01T00:00:00Z"), end=pd.Timestamp("2025-02-01T00:00:00Z"),
    )
    assert output["candidate_id"].tolist() == ["a", "b"]
    assert audit["valid_rows"] == 2
    assert audit["policy_json_sha256"] == "same"

    incompatible = _fragment(
        tmp_path, "incompatible", [_row("b", "2025-01-01T00:00:00Z")], "different",
    )
    with pytest.raises(ValueError, match="one identical optimized-policy contract"):
        MODULE._assemble(
            source_panel=source, fragments=[first, incompatible],
            start=pd.Timestamp("2024-12-01T00:00:00Z"), end=pd.Timestamp("2025-02-01T00:00:00Z"),
        )


def test_assembly_rejects_missing_source_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2024-12-01T00:00:00Z", "2025-01-01T00:00:00Z"]),
        "side_name": ["long", "long"],
    }).to_parquet(source, index=False)
    first = _fragment(tmp_path, "first", [_row("a", "2024-12-01T00:00:00Z")])
    with pytest.raises(ValueError, match="missing=1"):
        MODULE._assemble(
            source_panel=source, fragments=[first],
            start=pd.Timestamp("2024-12-01T00:00:00Z"), end=pd.Timestamp("2025-02-01T00:00:00Z"),
        )


def test_assembly_ignores_but_audits_outcome_superset(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pd.DataFrame({
        "candidate_id": ["a"],
        "__decision_ts__": pd.to_datetime(["2024-12-01T00:00:00Z"]),
        "side_name": ["long"],
    }).to_parquet(source, index=False)
    fragment = _fragment(tmp_path, "fragment", [
        _row("a", "2024-12-01T00:00:00Z"),
        _row("outside", "2025-01-01T00:00:00Z"),
    ])
    output, audit = MODULE._assemble(
        source_panel=source, fragments=[fragment],
        start=pd.Timestamp("2024-12-01T00:00:00Z"), end=pd.Timestamp("2025-02-01T00:00:00Z"),
    )
    assert output["candidate_id"].tolist() == ["a"]
    assert audit["ignored_outcome_superset_rows"] == 1
