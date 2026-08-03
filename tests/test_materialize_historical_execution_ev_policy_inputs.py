from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.materialize_historical_execution_ev_policy_inputs import run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> argparse.Namespace:
    labels = tmp_path / "labels"
    labels.mkdir()
    ts = pd.to_datetime(["2025-02-01T00:00:00Z", "2025-02-01T01:00:00Z"])
    for side in ("long", "short"):
        frame = pd.DataFrame(
            {
                "__ts__": ts,
                "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
                "side_name": side,
                "candidate_id": [f"{side}-0", f"{side}-1"],
                "__decision_ts__": ts + pd.Timedelta(hours=1),
                "__barrier_pct__": [0.01, 0.02],
            }
        )
        frame.to_parquet(
            labels / f"train_global_{side}_5_2025_02.parquet", index=False
        )
    path_inputs: list[Path] = []
    for side in ("long", "short"):
        path = tmp_path / f"path-{side}.parquet"
        pd.DataFrame(
            {
                "__ts__": ts,
                "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
                "side_name": side,
                "candidate_id": [f"{side}-0", f"{side}-1"],
                "__barrier_pct__": [0.01, 0.02],
                "__path_auxiliary_atr_fraction__": [0.005, 0.006],
            }
        ).to_parquet(path, index=False)
        path_inputs.append(path)
    policy = tmp_path / "policy.json"
    policy.write_text('{"policy": "fixture"}\n')
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "geometry": {
                    "fallback_rate": 1.0,
                    "side_archetype_rows": 0,
                },
                "source": {"policy_sha256": _sha(policy)},
            }
        )
    )
    spread = tmp_path / "spread.csv"
    pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
            "p90_spread_bps": [2.0, 3.0],
        }
    ).to_csv(spread, index=False)
    return argparse.Namespace(
        labels_root=labels,
        path_input_files=path_inputs,
        reference_manifest=reference,
        policy_json=policy,
        spread_baseline=spread,
        start_month="2025-02",
        end_month="2025-02",
        minimum_join_coverage=1.0,
        symbol_allowlist=None,
        output_dir=tmp_path / "output",
    )


def test_builds_side_parent_inputs_with_exact_lineage(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    outputs = run(args)
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["status"] == (
        "ready_for_current_spread_counterfactual_materialization"
    )
    assert manifest["parity"]["barrier_mismatch_rows"] == 0
    assert manifest["parity"]["admitted_rows"] == 4
    context = pd.read_parquet(outputs["context"])
    assert set(context["policy_archetype"]) == {
        "historical_side_parent_fallback"
    }


def test_fails_closed_on_barrier_mismatch(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args.path_input_files[0]
    frame = pd.read_parquet(path)
    frame.loc[0, "__barrier_pct__"] = 0.03
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="canonical barrier differs"):
        run(args)


def test_fails_closed_on_missing_spread_symbol(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    pd.DataFrame(
        {"symbol": ["BTC/USD:USD"], "p90_spread_bps": [2.0]}
    ).to_csv(args.spread_baseline, index=False)
    with pytest.raises(ValueError, match="does not cover"):
        run(args)


def test_fails_closed_without_universal_parent_fallback(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    reference = json.loads(args.reference_manifest.read_text())
    reference["geometry"]["fallback_rate"] = 0.99
    reference["geometry"]["side_archetype_rows"] = 1
    args.reference_manifest.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="universal side-parent fallback"):
        run(args)


def test_optional_symbol_allowlist_is_explicit_and_frozen(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    allowlist = tmp_path / "symbols.txt"
    allowlist.write_text("BTC/USD:USD\n")
    args.symbol_allowlist = allowlist
    outputs = run(args)
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["universe"]["mode"] == "frozen_symbol_allowlist"
    assert manifest["universe"]["symbols"] == ["BTC/USD:USD"]
    assert manifest["parity"]["candidate_rows"] == 2
