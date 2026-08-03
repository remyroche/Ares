from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_canonical_economic_conversion_transition_context import (
    BASE_CONTEXT_COLUMNS,
    COMPACT_REGIME_COLUMNS,
    CORE_MARKET_COLUMNS,
    DECISION_OBSERVABLE_COLUMNS,
    TRANSITION_COLUMNS,
    _validate_feature_surface,
    context_feature_columns,
    materialize_context,
)
from scripts.materialize_canonical_economic_conversion_transition_labels import (
    add_frozen_causal_score_deciles,
)


def _panel(hours: int = 3) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    origin = pd.Timestamp("2025-02-01T00:00:00Z")
    for hour in range(hours):
        for side in ("long", "short"):
            for rank in range(10):
                row: dict[str, object] = {
                    "candidate_id": f"{side}-{hour}-{rank}",
                    "side_name": side,
                    "__symbol__": f"S{rank:02d}",
                    "__ts__": origin + pd.Timedelta(hours=hour),
                }
                for number, column in enumerate(DECISION_OBSERVABLE_COLUMNS):
                    row[column] = float(10_000 * hour + 100 * number + rank)
                row["base_oof_score"] = float(100 - rank)
                records.append(row)
    return pd.DataFrame.from_records(records)


def _cohorts(panel: pd.DataFrame) -> pd.DataFrame:
    deciles = add_frozen_causal_score_deciles(panel)
    return (
        deciles.rename(columns={"__ts__": "cohort_anchor_utc"})[
            ["cohort_anchor_utc", "side_name", "frozen_base_score_decile"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def test_context_recomputes_label_deciles_and_aggregates_only_anchor_time_fields() -> None:
    panel = _panel()
    # An outcome-like column can be present in an upstream frame, but the
    # materializer never selects or aggregates it.
    panel["execution_net_ev_12h"] = np.linspace(-2.0, 2.0, len(panel))
    context = materialize_context(panel, _cohorts(panel))

    assert tuple(column for column in context if column.startswith("context__")) == context_feature_columns()
    assert len(CORE_MARKET_COLUMNS) == 5
    assert len(TRANSITION_COLUMNS) == 18
    assert set(BASE_CONTEXT_COLUMNS).issubset(DECISION_OBSERVABLE_COLUMNS)
    assert set(COMPACT_REGIME_COLUMNS).issubset(DECISION_OBSERVABLE_COLUMNS)
    assert not any("execution_net" in column or "mapped_" in column for column in context.columns)

    first = context.loc[
        context["cohort_anchor_utc"].eq(pd.Timestamp("2025-02-01T00:00:00Z"))
        & context["side_name"].eq("long")
        & context["frozen_base_score_decile"].eq(0)
    ].iloc[0]
    # Ten rows make every deterministic decile a one-candidate cohort.  Decile
    # zero is the score-100 candidate, independently of any outcome column.
    assert first["anchor_candidate_support"] == 1
    assert first["context__base_oof_score__mean"] == pytest.approx(100.0)
    assert first["context__range_24h_pct__mean"] == pytest.approx(1_400.0)
    assert first["context__side_sign"] == 1
    assert first["context__frozen_base_score_decile"] == 0


def test_tied_score_deciles_use_the_canonical_symbol_candidate_tie_break() -> None:
    panel = _panel(hours=1)
    panel["base_oof_score"] = 1.0
    context = materialize_context(panel, _cohorts(panel))
    top = context.loc[
        context["side_name"].eq("long") & context["frozen_base_score_decile"].eq(0)
    ].iloc[0]
    # The S00 / rank-0 row wins the tie-break and has this fixed source value.
    assert top["context__range_24h_pct__mean"] == pytest.approx(1_400.0)


def test_feature_allowlist_rejects_outcomes_and_maps_but_keeps_preentry_execution_composites() -> None:
    _validate_feature_surface(COMPACT_REGIME_COLUMNS)
    with pytest.raises(ValueError, match="non-observable"):
        _validate_feature_surface(("base_oof_score", "execution_net_ev_12h"))
    with pytest.raises(ValueError, match="non-observable"):
        _validate_feature_surface(("base_oof_score", "mapped_direct_net"))
