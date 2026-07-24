from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_label_quality_proxy_diagnostics as diagnostics


class _SymbolFrameStore(dict):
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        keys = sorted({column for frame in frames.values() for column in frame.columns})
        super().__init__({key: object() for key in keys})
        self.frames = frames
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def symbol_frame(self, symbol: str, keys=None) -> pd.DataFrame:
        selected = tuple(str(key) for key in (keys or []))
        self.calls.append((str(symbol), selected))
        frame = self.frames.get(str(symbol), pd.DataFrame())
        return frame.reindex(columns=[key for key in selected if key in frame.columns])


def test_feature_loader_prefers_direct_symbol_frames(
    tmp_path: Path, monkeypatch
) -> None:
    feature_dir = tmp_path / "features" / "20260101_000000"
    feature_dir.mkdir(parents=True)
    for symbol in ("AAA/USD:USD", "BBB/USD:USD"):
        safe = symbol.replace("/", "_")
        (feature_dir / f"symbol={safe}.parquet").touch()

    index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    store = _SymbolFrameStore(
        {
            "AAA/USD:USD": pd.DataFrame(
                {"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]}, index=index
            ),
            "BBB/USD:USD": pd.DataFrame(
                {"f1": [7.0, 8.0, 9.0], "f2": [10.0, 11.0, 12.0]}, index=index
            ),
        }
    )

    import extreme_price_movements.static_feature_store as static_store

    monkeypatch.setattr(static_store, "read_static_features", lambda **_: store)
    frame = pd.DataFrame(
        {
            "__ts__": [index[0], index[2], index[1]],
            "__symbol__": ["AAA/USD:USD", "AAA/USD:USD", "BBB/USD:USD"],
        }
    )
    matrix, report = diagnostics._load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=["f1", "f2"],
        min_feature_finite_frac=1.0,
    )

    np.testing.assert_allclose(
        matrix.to_numpy(dtype=np.float32),
        np.asarray([[1.0, 4.0], [3.0, 6.0], [8.0, 11.0]], dtype=np.float32),
    )
    assert sorted(symbol for symbol, _keys in store.calls) == [
        "AAA/USD:USD",
        "BBB/USD:USD",
    ]
    assert report["reader"].endswith("symbol_frame_preferred")
