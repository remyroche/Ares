from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_oof_stack import RegimeOOFStackError
from scripts.materialize_oof_regime_stack_features import materialize


def _candidates() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=96, freq="12h", tz="UTC")
    return pd.DataFrame(
        {
            "candidate_id": [f"c{index}" for index in range(len(timestamps))],
            "__ts__": timestamps,
            "__symbol__": np.where(np.arange(len(timestamps)) % 2, "ETH/USD:USD", "BTC/USD:USD"),
            "side_name": np.where(np.arange(len(timestamps)) % 2, "short", "long"),
        }
    )


def _panel(candidates: pd.DataFrame, prefix: str) -> pd.DataFrame:
    index = np.arange(len(candidates), dtype=float)
    return candidates.assign(
        **{
            f"{prefix}_vol": np.sin(index / 5.0),
            f"{prefix}_breadth": np.cos(index / 7.0),
            f"{prefix}_liquidity": index / max(1.0, len(index)),
        }
    )


def test_materializer_emits_exact_regime_and_transition_oof_coverage(tmp_path: Path) -> None:
    candidates = _candidates()
    candidate_path = tmp_path / "candidates.parquet"
    regime_path = tmp_path / "regime.parquet"
    transition_path = tmp_path / "transition.parquet"
    candidates.to_parquet(candidate_path, index=False)
    _panel(candidates, "reg").to_parquet(regime_path, index=False)
    _panel(candidates, "transition").to_parquet(transition_path, index=False)
    output = materialize(
        candidates_path=candidate_path,
        regime_panel_path=regime_path,
        transition_panel_path=transition_path,
        output_dir=tmp_path / "output",
        evaluation_start="2026-01-20T00:00:00Z",
        frequency="week",
        n_components=2,
        max_features=3,
        pca_components=2,
    )
    ledger = pd.read_parquet(output / "oof_regime_stack_features.parquet")
    expected = candidates.loc[candidates["__ts__"].ge(pd.Timestamp("2026-01-20", tz="UTC"))]
    assert len(ledger) == len(expected)
    assert ledger["candidate_id"].nunique() == len(expected)
    assert {"regime_state_p__0", "transition_state_p__0", "regime_fold_id", "transition_fold_id"}.issubset(ledger.columns)
    assert (pd.to_datetime(ledger["regime_train_end_utc"], utc=True) < pd.to_datetime(ledger["__ts__"], utc=True)).all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["coverage"]["exact_identity_coverage"]
    assert manifest["transition"]["status"] == "MATERIALIZED"


def test_missing_transition_is_explicitly_fail_closed(tmp_path: Path) -> None:
    candidates = _candidates()
    candidate_path = tmp_path / "candidates.parquet"
    regime_path = tmp_path / "regime.parquet"
    candidates.to_parquet(candidate_path, index=False)
    _panel(candidates, "reg").to_parquet(regime_path, index=False)
    output = materialize(
        candidates_path=candidate_path,
        regime_panel_path=regime_path,
        output_dir=tmp_path / "output",
        evaluation_start="2026-01-20T00:00:00Z",
        frequency="week",
        n_components=2,
        max_features=3,
        pca_components=2,
    )
    ledger = pd.read_parquet(output / "oof_regime_stack_features.parquet")
    assert not any(column.startswith("transition_state_p__") for column in ledger.columns)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["transition"]["status"] == "UNAVAILABLE_FAIL_CLOSED"
    with pytest.raises(RegimeOOFStackError, match="required"):
        materialize(
            candidates_path=candidate_path,
            regime_panel_path=regime_path,
            output_dir=tmp_path / "requires_transition",
            evaluation_start="2026-01-20T00:00:00Z",
            frequency="week",
            n_components=2,
            require_transition=True,
        )
