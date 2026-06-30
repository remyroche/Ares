from pathlib import Path

import pandas as pd

from scripts.generate_live_finalfit_oos_predictions import (
    LIVE_FINALFIT_PREDICTION_COLUMNS,
    _load_sample_ledger,
    _make_feature_panels,
)
from scripts.materialize_live_ledger_blend_native_candidates import (
    EMPTY_CANDIDATE_COLUMNS,
    _attach_scores,
    _load_ledgers,
)


def test_empty_sample_ledger_keeps_symbol_norm_and_empty_feature_audit(tmp_path: Path) -> None:
    ledger = tmp_path / "sample.parquet"
    pd.DataFrame(
        columns=["timestamp", "symbol", "head", "strategy_id", "side"]
    ).to_parquet(ledger, index=False)

    samples = _load_sample_ledger(
        ledger,
        strategy_id="short_asset_demo",
        min_timestamp="2026-06-27T13:00:00Z",
        max_timestamp="2026-06-27T15:00:00Z",
    )
    panels, audit = _make_feature_panels(
        samples,
        feature_root=tmp_path / "features",
        feature_keys={"x", "y"},
    )

    assert samples.empty
    assert "symbol_norm" in samples.columns
    assert panels == {}
    assert audit["empty_sample_ledger"] is True
    assert audit["requested_feature_count"] == 2


def test_empty_prediction_schema_is_available_for_combined_exports() -> None:
    frame = pd.DataFrame(columns=list(LIVE_FINALFIT_PREDICTION_COLUMNS))

    assert frame.empty
    assert {"timestamp", "symbol", "strategy_id", "calibrated_score", "policy_rank_pct"}.issubset(
        frame.columns
    )


def test_empty_candidate_schema_includes_replay_required_fields() -> None:
    assert {"entry_price", "exit_price", "exit_timestamp", "holding_bars"}.issubset(
        set(EMPTY_CANDIDATE_COLUMNS)
    )


def test_empty_live_ledger_materializer_load_and_score_attach(tmp_path: Path) -> None:
    ledger = tmp_path / "combined_prediction_ledger.parquet"
    pd.DataFrame(columns=list(LIVE_FINALFIT_PREDICTION_COLUMNS)).to_parquet(ledger, index=False)

    loaded = _load_ledgers([ledger], start="2026-06-27T13:00:00Z", end="2026-06-27T15:00:00Z")
    scored, diag = _attach_scores(
        loaded,
        score_path=None,
        score_column="calibrated_score",
        allow_ledger_score=True,
    )

    assert loaded.empty
    assert scored.empty
    assert "reliability_blend_score" in scored.columns
    assert diag["scored_rows"] == 0
    assert diag["score_source"] == "empty_live_ledger"
