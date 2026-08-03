from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_short_winner_causal_recent_ev_mapping_v5.py"
)
SPEC = importlib.util.spec_from_file_location("short_mapping_v5", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _history(rows: int, *, short_rows: int | None = None) -> pd.DataFrame:
    snapshot = pd.Timestamp("2025-04-10T00:00:00Z")
    short_rows = rows // 2 if short_rows is None else short_rows
    side = np.where(np.arange(rows) < short_rows, "short", "long")
    score = np.linspace(-1.0, 1.0, rows)
    return pd.DataFrame(
        {
            "candidate_id": [f"h{i}" for i in range(rows)],
            "side_name": side,
            "__symbol__": ["A"] * rows,
            "__ts__": snapshot - pd.Timedelta(days=2),
            MODULE.base.TIME: snapshot - pd.Timedelta(days=2),
            MODULE.base.END: snapshot - pd.Timedelta(days=1),
            MODULE.base.Y: score * 0.01 + np.where(side == "short", -0.002, 0.002),
            "raw_score": score,
            "score_available_utc": snapshot - pd.Timedelta(days=2),
        }
    )


def _evaluation() -> pd.DataFrame:
    snapshot = pd.Timestamp("2025-04-10T00:00:00Z")
    return pd.DataFrame(
        {
            "candidate_id": ["eval-short", "eval-long"],
            "side_name": ["short", "long"],
            "__symbol__": ["A", "B"],
            "__ts__": [snapshot, snapshot],
            MODULE.base.TIME: [snapshot, snapshot],
            MODULE.base.END: [snapshot + pd.Timedelta(hours=12)] * 2,
            MODULE.base.Y: [0.01, -0.01],
            "raw_score": [0.4, 0.4],
            "score_available_utc": [snapshot, snapshot],
        }
    )


def test_weak_pooled_snapshot_is_unmapped_not_raw_or_zero() -> None:
    mapped, audit = MODULE.causal_map(
        _history(MODULE.POOL_MIN - 1), _evaluation(), add_side_residual=True
    )
    assert mapped.causal_pooled_side_21d.isna().all()
    assert not mapped.causal_pooled_side_21d_eligible.any()
    assert mapped.causal_pooled_side_21d_status.eq("unmapped_weak_pooled").all()
    assert not audit.pooled_support_pass.any()


def test_weak_side_uses_pooled_anchor_with_exact_zero_side_residual() -> None:
    history = _history(MODULE.POOL_MIN + 100, short_rows=MODULE.SIDE_MIN - 1)
    pooled, _ = MODULE.causal_map(history, _evaluation(), add_side_residual=False)
    mapped, _ = MODULE.causal_map(history, _evaluation(), add_side_residual=True)
    short = mapped.side_name.eq("short")
    assert mapped.loc[short, "causal_pooled_side_21d_status"].eq(
        "pooled_zero_side_residual"
    ).all()
    assert np.allclose(
        mapped.loc[short, "causal_pooled_side_21d"],
        pooled.loc[short, "causal_pooled_21d"],
    )
    assert mapped.loc[short, "causal_pooled_side_21d_side_weight"].eq(0.0).all()


def test_side_residual_matches_side_minus_pooled_isotonic_shrinkage() -> None:
    history = _history(2 * MODULE.SIDE_MIN + 200)
    evaluate = _evaluation()
    pooled, _ = MODULE.causal_map(history, evaluate, add_side_residual=False)
    mapped, audit = MODULE.causal_map(history, evaluate, add_side_residual=True)
    for side_name in ("short", "long"):
        reference = history.loc[history.side_name.eq(side_name)]
        side_model = IsotonicRegression(out_of_bounds="clip").fit(
            reference.raw_score, reference[MODULE.base.Y]
        )
        row = mapped.loc[mapped.side_name.eq(side_name)].iloc[0]
        pooled_row = pooled.loc[pooled.side_name.eq(side_name)].iloc[0]
        weight = len(reference) / (len(reference) + MODULE.SIDE_LAMBDA)
        expected = pooled_row.causal_pooled_21d + weight * (
            side_model.predict([row.raw_score])[0] - pooled_row.causal_pooled_21d
        )
        assert np.isclose(row.causal_pooled_side_21d, expected)
        assert np.isclose(row.causal_pooled_side_21d_side_weight, weight)
    assert audit.evaluation_reference_identity_overlap.eq(0).all()
    assert audit.strict_causal_window_pass.all()


def test_reference_window_uses_label_end_and_excludes_current_identity() -> None:
    history = _history(MODULE.POOL_MIN + 10)
    snapshot = pd.Timestamp("2025-04-10T00:00:00Z")
    history.loc[0, MODULE.base.TIME] = snapshot - pd.Timedelta(days=30)
    history.loc[0, MODULE.base.END] = snapshot - pd.Timedelta(days=20)
    duplicate = _evaluation().iloc[[0]].copy()
    duplicate[MODULE.base.END] = snapshot + pd.Timedelta(hours=12)
    combined = pd.concat([history, duplicate], ignore_index=True)
    _, audit = MODULE.causal_map(combined, _evaluation(), add_side_residual=False)
    row = audit.iloc[0]
    assert row.reference_rows == len(history)
    assert row.evaluation_reference_identity_overlap == 0
    assert row.reference_label_end_min_utc >= snapshot - MODULE.WINDOW
    assert row.reference_label_end_max_utc < snapshot
