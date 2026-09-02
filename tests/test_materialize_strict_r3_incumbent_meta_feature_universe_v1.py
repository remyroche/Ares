from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_strict_r3_incumbent_meta_feature_universe_v1.py"
SPEC = importlib.util.spec_from_file_location("incumbent_meta_materialiser", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_parse_months_requires_contiguous_ordered_interval() -> None:
    months = MODULE._parse_months("2025-11,2025-12,2026-01")
    assert [f"{item:%Y-%m}" for item in months] == ["2025-11", "2025-12", "2026-01"]
    with pytest.raises(ValueError, match="contiguous"):
        MODULE._parse_months("2025-11,2026-01")


def test_coverage_reports_only_numeric_feature_statistics() -> None:
    frame = pd.DataFrame({"a": [1.0, None, 2.0], "b": [1, 1, 1]})
    result = MODULE._coverage(frame, ["a", "b"]).set_index("feature")
    assert result.loc["a", "finite_rows"] == 2
    assert result.loc["a", "n_unique"] == 2
    assert result.loc["b", "finite_fraction"] == 1.0


def test_candidate_identity_rejects_non_50_50_upstream(tmp_path: Path) -> None:
    root = tmp_path / "source" / "month=2026-01"
    root.mkdir(parents=True)
    data = pd.DataFrame({
        "candidate_id": ["X|long|2026-01-01T00:00:00Z"],
        "__decision_ts__": ["2026-01-01T01:00:00Z"],
        "side_name": ["long"],
        "base_bps": [0.0],
        "efficiency_bps": [2.0],
        "timing_bps": [4.0],
        "enhanced_base_bps": [2.5],
        "base_rank_ts": [1.0],
        "enhanced_base_routed": [True],
        "e_minus_t": [-2.0],
        "e_minus_b0": [2.0],
        "t_minus_b0": [4.0],
        "base_component_std": [1.0],
    })
    data.to_parquet(root / "scores_features.parquet", index=False)
    with pytest.raises(AssertionError, match="50/50"):
        MODULE._candidate_identities(tmp_path / "source", pd.Timestamp("2026-01-01", tz="UTC"))


def test_predeclared_full_universe_batches_preserve_identity_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = pd.DataFrame({
        "candidate_id": ["B|long|t2", "A|long|t1"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T02:00:00Z", "2026-01-01T01:00:00Z"], utc=True),
        "side_name": ["long", "long"],
        "__ts__": pd.to_datetime(["2026-01-01T01:00:00Z", "2026-01-01T00:00:00Z"], utc=True),
        "__symbol__": ["B/USD:USD", "A/USD:USD"],
    })

    seen_context: list[tuple[str, ...]] = []

    def fake_materialize(out: Path, labels: pd.DataFrame, contract: dict[str, list[str]], *_args, **kwargs) -> Path:
        out.mkdir(parents=True, exist_ok=True)
        seen_context.append(tuple(kwargs["context_symbols"]))
        result = labels.loc[:, ["__ts__", "__symbol__"]].copy()
        for ordinal, field in enumerate(contract["long"]):
            result[field] = pd.to_numeric(result["__ts__"].astype("int64"), errors="raise") + ordinal
        path = out / "canonical120_features.parquet"
        result.to_parquet(path, index=False)
        return path

    monkeypatch.setattr(MODULE, "materialize_features", fake_materialize)
    output = MODULE._materialize_predeclared_full_universe(
        out_dir=tmp_path / "full", identities=identities,
        start=pd.Timestamp("2025-12-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"),
        fields=("f_a", "f_b", "f_c"), field_chunk_size=2, reference_symbols=(), context_symbols=("CTX/USD:USD",),
    )
    result = pd.read_parquet(output)
    assert list(result.columns) == ["__ts__", "__symbol__", "f_a", "f_b", "f_c"]
    assert result[["__ts__", "__symbol__"]].equals(
        identities[["__ts__", "__symbol__"]].sort_values(["__ts__", "__symbol__"], kind="stable").reset_index(drop=True)
    )
    assert not (tmp_path / "full" / "_predeclared_field_chunks").exists()
    audit = json.loads((tmp_path / "full" / "predeclared_field_batch_audit.json").read_text())
    assert audit["field_count"] == 3
    assert audit["identity_preserved_every_batch"] is True
    assert seen_context == [("CTX/USD:USD",), ("CTX/USD:USD",)]


def test_context_symbol_manifest_requires_a_substantial_frozen_market_universe(tmp_path: Path) -> None:
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(json.dumps({"symbols": ["A/USD:USD", "B/USD:USD"]}))
    with pytest.raises(AssertionError, match="complete causal context"):
        MODULE._context_symbols_from_manifest(manifest)
    symbols = [f"S{index}/USD:USD" for index in range(24)]
    manifest.write_text(json.dumps({"symbols": list(reversed(symbols))}))
    assert MODULE._context_symbols_from_manifest(manifest) == tuple(sorted(symbols))
