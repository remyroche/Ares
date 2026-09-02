from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_direct_state_forward import (
    forward_direct_state,
    validate_direct_source_state,
)


class _FakeExecutor:
    SOURCE_FIELDS = ("close", "volume")

    def __init__(self, *, root, symbols, market_basket) -> None:
        self.root = Path(root)
        self.symbols = tuple(symbols)
        self.calls: list[pd.Timestamp] = []

    def _read_ledger(self):
        if not self.calls:
            return {"last_timestamp": "2026-01-01T00:00:00+00:00"}
        return {"last_timestamp": self.calls[-1].isoformat(), "active_commit": "commits/fake"}

    def advance(self, *, timestamp, snapshot, required_features):
        assert required_features == ()
        assert set(snapshot) == set(self.SOURCE_FIELDS)
        assert all(value.dtype == np.float32 for value in snapshot.values())
        self.calls.append(pd.Timestamp(timestamp))
        return {"direct_feature": np.zeros(len(self.symbols), dtype=np.float32)}


def _source(*, start: str = "2026-01-01T00:00:00Z", periods: int = 3) -> dict[str, object]:
    symbols = tuple(f"S{index:03d}" for index in range(160))
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    values = np.ones((periods, len(symbols)), dtype=np.float32)
    return {
        "symbols": symbols,
        "panel": {
            "close": pd.DataFrame(values, index=index, columns=symbols),
            "volume": pd.DataFrame(values * 2, index=index, columns=symbols),
        },
    }


def test_validate_direct_source_requires_a_complete_hourly_frozen_universe() -> None:
    source = _source()
    symbols, _frames, index = validate_direct_source_state(source, source_fields=("close", "volume"))
    assert len(symbols) == 160
    assert len(index) == 3


def test_validate_direct_source_rejects_outcome_like_input() -> None:
    source = _source()
    source["policy_net_bps"] = object()
    with pytest.raises(ValueError, match="outcome-like"):
        validate_direct_source_state(source, source_fields=("close", "volume"))


def test_forward_requires_every_intermediate_hour_and_never_requests_model_features(tmp_path) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "marker").write_text("immutable")

    def clone(source: Path, destination: Path) -> str:
        destination.mkdir()
        (destination / "marker").write_text(source.joinpath("marker").read_text())
        return "test_clone"

    result = forward_direct_state(
        bootstrap_root=bootstrap,
        output_root=tmp_path / "out",
        source=_source(),
        through="2026-01-01T02:00:00Z",
        executor_factory=_FakeExecutor,
        tree_cloner=clone,
    )
    assert [stamp.isoformat() for stamp in result.committed_timestamps] == [
        "2026-01-01T01:00:00+00:00",
        "2026-01-01T02:00:00+00:00",
    ]
    assert result.direct_feature_count == 1


def test_forward_refuses_to_overwrite_an_existing_state_root(tmp_path) -> None:
    root = tmp_path / "already_exists"
    root.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        forward_direct_state(
            bootstrap_root=root,
            output_root=root,
            source=_source(),
            through="2026-01-01T02:00:00Z",
            executor_factory=_FakeExecutor,
        )
