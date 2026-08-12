from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_shared_population import (
    SharedPopulationError,
    SharedPopulationSpec,
    file_sha256,
    materialize_shared_population,
    validate_shared_population,
)


def _write(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _population(side: str, *, invalid_id: str | None = None) -> pd.DataFrame:
    timestamp = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "candidate_id": [f"{side}-{index}" for index in range(3)],
        "__ts__": timestamp, "__symbol__": ["BTC", "ETH", "SOL"], "side_name": side,
        "decision_ts": timestamp + pd.Timedelta(hours=1),
        "label_available_ts": timestamp + pd.Timedelta(hours=13),
        "target_valid": True,
    })
    if invalid_id is not None:
        frame.loc[frame.candidate_id.eq(invalid_id), "target_valid"] = False
    return frame


def _winner(root, *, invalid_by_side: dict[str, str]) -> None:
    frames = []
    for side in ("long", "short"):
        frames.append(_population(side, invalid_id=invalid_by_side.get(side)))
    handoff = pd.concat(frames, ignore_index=True)
    path = root / "winner_target_handoff.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    handoff.to_parquet(path, index=False)
    _write(root / "manifest.json", {
        "status": "complete", "artifact_sha256": {path.name: file_sha256(path)},
    })


def _r3(root) -> None:
    for side in ("long", "short"):
        frame = _population(side).drop(columns="target_valid")
        frame["exact_net_bps"] = np.asarray([-50.0, 0.0, 50.0])
        path = root / side / "selector_base_oof.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        _write(root / side / "manifest.json", {
            "status": "complete", "selector_base_oof_sha256": file_sha256(path),
        })


def test_common_universe_is_per_side_r3_scalar_ordinal_valid_intersection(tmp_path) -> None:
    r3, scalar, ordinal, output = tmp_path / "r3", tmp_path / "scalar", tmp_path / "ordinal", tmp_path / "shared"
    _r3(r3)
    _winner(scalar, invalid_by_side={"long": "long-2", "short": "short-1"})
    _winner(ordinal, invalid_by_side={"long": "long-1", "short": "short-2"})
    manifest = materialize_shared_population(SharedPopulationSpec(
        r3_base_selection_dir=r3, scalar_winner_dir=scalar, ordinal_winner_dir=ordinal,
        output_dir=output,
    ))
    frame, verified = validate_shared_population(output)
    assert manifest["contract_sha256"] == verified["contract_sha256"]
    assert frame.groupby("side_name").size().to_dict() == {"long": 1, "short": 1}
    assert set(frame.candidate_id) == {"long-0", "short-0"}
    assert set(verified["per_side"]) == {"long", "short"}


def test_shared_universe_rejects_post_publication_identity_drift(tmp_path) -> None:
    r3, scalar, ordinal, output = tmp_path / "r3", tmp_path / "scalar", tmp_path / "ordinal", tmp_path / "shared"
    _r3(r3); _winner(scalar, invalid_by_side={}); _winner(ordinal, invalid_by_side={})
    materialize_shared_population(SharedPopulationSpec(r3, scalar, ordinal, output))
    frame = pd.read_parquet(output / "shared_population.parquet")
    frame.loc[0, "candidate_id"] = "drift"
    frame.to_parquet(output / "shared_population.parquet", index=False)
    with pytest.raises(SharedPopulationError, match="checksum drift"):
        validate_shared_population(output)
