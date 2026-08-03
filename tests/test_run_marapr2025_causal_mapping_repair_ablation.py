from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_marapr2025_causal_mapping_repair_ablation import (
    align_huber_to_isotonic,
    build_day_maps,
    evaluate_arm,
    global_top,
    strict_rank_key,
)


TIME = "execution_decision_utc"
END = "execution_label_end_utc"
NET = "execution_net_ev_12h"


def _rows(
    rows: list[tuple[str, str, float, float, str]],
) -> pd.DataFrame:
    """Small exact-policy ledger with deterministic candidate identities."""

    frame = pd.DataFrame(
        rows,
        columns=["candidate_id", "side_name", "raw_score", NET, TIME],
    )
    frame["__symbol__"] = frame["candidate_id"].str.upper()
    frame["__ts__"] = pd.to_datetime(frame[TIME], utc=True) - pd.Timedelta(hours=1)
    frame[TIME] = pd.to_datetime(frame[TIME], utc=True)
    frame[END] = frame[TIME] + pd.Timedelta(hours=12)
    frame["execution_gross_ev_12h"] = frame[NET] + 0.01
    frame["execution_cost_return"] = 0.01
    return frame


def test_strict_rank_key_repairs_only_isotonic_plateau_ties() -> None:
    # I-R may use raw score only where the isotonic map is tied.  It may not
    # move an observation across distinct mapped-EV levels.
    mapped = np.array([0.10, 0.10, 0.40, 0.40, 0.70])
    raw = np.array([0.90, 0.20, 0.10, 0.80, 0.50])
    key = np.asarray(strict_rank_key(mapped, raw), dtype=float)

    # Higher mapped level always remains higher, even when raw says otherwise.
    assert key[2] > key[0]
    assert key[3] > key[1]
    assert key[4] > key[3]
    # Within each isotonic plateau, the raw source score is the sole tie-break.
    assert key[0] > key[1]
    assert key[3] > key[2]


def test_align_huber_uses_reference_only_robust_scale_and_clips() -> None:
    # Identical reference scales make the pre-clip alignment the identity.  The
    # only permitted changes are the declared isotonic p01/p99 bounds.
    reference_iso = np.arange(5.0)
    reference_huber = np.arange(5.0)
    evaluate_huber = np.array([-100.0, 1.0, 3.0, 100.0])
    aligned = align_huber_to_isotonic(
        reference_iso,
        reference_huber,
        evaluate_huber,
    )
    low, high = np.quantile(reference_iso, [0.01, 0.99])
    assert np.allclose(aligned, [low, 1.0, 3.0, high])

    # I-S is a fixed, non-tuned 25% Huber contribution.  With a known
    # isotonic prediction, its output must be exactly 75/25, not a refitted or
    # side-specific blend.
    isotonic_eval = np.array([0.20, 0.50, 0.80, 0.90])
    shrink = 0.75 * isotonic_eval + 0.25 * aligned
    assert np.allclose(
        shrink,
        [0.75 * 0.20 + 0.25 * low, 0.625, 1.35, 0.75 * 0.90 + 0.25 * high],
    )


def test_build_day_maps_excludes_unresolved_and_same_day_reference_rows() -> None:
    snapshot = pd.Timestamp("2025-05-10T00:00:00Z")
    history = _rows(
        [
            ("legal", "long", 0.1, 0.01, "2025-05-08T00:00:00Z"),
            # Its resolution time is on the snapshot day, so this row must not
            # enter a day-level map even though its decision is older.
            ("same_day", "long", 0.2, 0.02, "2025-05-09T18:00:00Z"),
            # Its label is still unresolved at the snapshot.
            ("unresolved", "short", 0.3, -0.01, "2025-05-09T20:00:00Z"),
        ]
    )
    history.loc[history.candidate_id.eq("legal"), END] = pd.Timestamp(
        "2025-05-09T12:00:00Z"
    )
    history.loc[history.candidate_id.eq("same_day"), END] = snapshot
    history.loc[history.candidate_id.eq("unresolved"), END] = snapshot + pd.Timedelta(
        hours=12
    )
    evaluate = _rows(
        [
            ("eval_a", "long", 0.15, 0.0, "2025-05-10T00:00:00Z"),
            ("eval_b", "short", 0.25, 0.0, "2025-05-10T00:00:00Z"),
        ]
    )

    _mapped, audit = build_day_maps(history, evaluate)
    assert int(audit.loc[0, "reference_rows"]) == 1
    assert bool(audit.loc[0, "all_reference_labels_before_snapshot"])
    assert bool(audit.loc[0, "zero_evaluation_reference_overlap"])
    assert pd.Timestamp(audit.loc[0, "reference_max_label_end_utc"]) < snapshot


def test_global_top_is_one_pooled_book_not_timestamp_or_side_books() -> None:
    frame = _rows(
        [
            ("a", "long", 0.99, 0.01, "2025-04-01T00:00:00Z"),
            ("b", "long", 0.98, 0.01, "2025-04-01T00:00:00Z"),
            ("c", "long", 0.97, 0.01, "2025-04-01T00:00:00Z"),
            ("d", "short", 0.20, 0.01, "2025-04-02T00:00:00Z"),
            ("e", "short", 0.10, 0.01, "2025-04-02T00:00:00Z"),
        ]
    )
    frame["mapped"] = frame.raw_score
    selected, metrics = global_top(frame, "mapped", "raw_score", 0.4)

    assert selected.candidate_id.tolist() == ["a", "b"]
    assert selected.side_name.tolist() == ["long", "long"]
    assert metrics["selection_scope"] == "pooled_global"
    assert int(metrics["selected_rows"]) == 2


def test_evaluate_arm_reconciles_side_contributions_and_reports_calendar_metrics() -> None:
    frame = _rows(
        [
            ("a", "long", 0.90, 0.02, "2025-04-01T00:00:00Z"),
            ("b", "short", 0.80, -0.01, "2025-04-01T00:00:00Z"),
            ("c", "long", 0.70, 0.03, "2025-04-02T00:00:00Z"),
            ("d", "short", 0.60, 0.01, "2025-04-03T00:00:00Z"),
        ]
    )
    frame["I-R"] = frame.raw_score
    result = evaluate_arm(
        frame,
        arm="I-R",
        score_col="I-R",
        raw_col="raw_score",
        fraction=0.5,
    )

    assert result["selection_scope"] == "pooled_global"
    assert result["selected_rows"] == 2
    assert set(result["side_contribution_bps"]) == {"long", "short"}
    assert np.isclose(
        result["net_bps"], sum(result["side_contribution_bps"].values())
    )
    assert result["effective_selected_days"] > 0.0
    assert 0.0 < result["top_three_day_share"] <= 1.0
