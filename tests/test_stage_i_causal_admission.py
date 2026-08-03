from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import pytest

import extreme_price_movements.stage_i_causal_admission as admission_module
from extreme_price_movements.stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
    pooled_global_admission_comparison,
)


SPEC = Causal21dAdmissionSpec(min_reference_rows=4, bins=4, net_floor_bps=50.0)


def _reference_full_scan_admission(
    frame: pd.DataFrame,
    *,
    score_column: str,
    net_column: str = "net_bps",
    decision_column: str = "__ts__",
    label_available_column: str = "label_available_ts",
    identity_column: str = "candidate_id",
    spec: Causal21dAdmissionSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen pre-optimisation implementation used as an equivalence oracle."""
    out = admission_module._validate_input(
        frame,
        score_column=score_column,
        net_column=net_column,
        decision_column=decision_column,
        label_available_column=label_available_column,
        identity_column=identity_column,
    )
    original_index = out.index.copy()
    out["__admission_original_position__"] = np.arange(len(out), dtype=np.int64)
    out = out.sort_values([decision_column, identity_column], kind="stable").reset_index(drop=True)
    out["causal_21d_side_expected_net_bps"] = np.nan
    out["causal_21d_side_reference_rows"] = 0
    out["causal_21d_side_mapping_status"] = "unmapped_insufficient_side_support"
    out["causal_21d_side_admitted_ge_50bps"] = False
    score = pd.to_numeric(out[score_column], errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(out[net_column], errors="coerce").to_numpy(dtype=float)
    decision = out[decision_column]
    available = out[label_available_column]
    audit: list[dict[str, object]] = []
    for snapshot, current_idx in out.groupby(decision.dt.normalize(), sort=True).groups.items():
        snapshot = pd.Timestamp(snapshot)
        window_start = snapshot - pd.Timedelta(days=spec.window_days)
        for side in admission_module.SIDES:
            current_pos = np.asarray(list(current_idx), dtype=int)
            current_pos = current_pos[out.loc[current_pos, "side_name"].eq(side).to_numpy()]
            reference_mask = (
                out["side_name"].eq(side)
                & decision.lt(snapshot)
                & available.lt(snapshot)
                & available.ge(window_start)
                & np.isfinite(score)
                & np.isfinite(target)
            )
            reference_pos = np.flatnonzero(reference_mask.to_numpy())
            status = "mapped"
            if len(reference_pos) < spec.min_reference_rows:
                status = "unmapped_insufficient_side_support"
            elif not len(current_pos):
                status = "mapped_no_current_side_rows"
            else:
                mapped = admission_module._fit_predict_robust_isotonic(
                    score[reference_pos], target[reference_pos], score[current_pos], spec,
                )
                if not np.isfinite(mapped).all():
                    status = "unmapped_degenerate_score_support"
                else:
                    out.loc[current_pos, "causal_21d_side_expected_net_bps"] = mapped
                    out.loc[current_pos, "causal_21d_side_admitted_ge_50bps"] = mapped >= spec.net_floor_bps
            if len(current_pos):
                out.loc[current_pos, "causal_21d_side_reference_rows"] = len(reference_pos)
                out.loc[current_pos, "causal_21d_side_mapping_status"] = status
            audit.append({
                "snapshot_utc": snapshot,
                "side_name": side,
                "window_start_utc": window_start,
                "window_end_exclusive_utc": snapshot,
                "reference_rows": int(len(reference_pos)),
                "current_rows": int(len(current_pos)),
                "reference_max_label_available_ts": (
                    available.iloc[reference_pos].max() if len(reference_pos) else pd.NaT
                ),
                "strictly_prior_resolved": (
                    bool(available.iloc[reference_pos].lt(snapshot).all())
                    if len(reference_pos) else True
                ),
                "mapping_status": status,
            })
    out = out.sort_values("__admission_original_position__", kind="stable").drop(
        columns="__admission_original_position__",
    )
    out.index = original_index
    return out, pd.DataFrame(audit)


def _population() -> pd.DataFrame:
    snapshot = pd.Timestamp("2025-02-01 00:00Z")
    rows: list[dict[str, object]] = []
    for side, shift in (("long", 100.0), ("short", -150.0)):
        for number in range(4):
            decision = snapshot - pd.Timedelta(days=4 - number)
            rows.append({
                "candidate_id": f"{side}-history-{number}", "side_name": side,
                "__ts__": decision, "label_available_ts": decision + pd.Timedelta(hours=13),
                "raw_score": float(number), "net_bps": shift + 50.0 * number,
            })
        rows.append({
            "candidate_id": f"{side}-current", "side_name": side,
            "__ts__": snapshot, "label_available_ts": snapshot + pd.Timedelta(hours=13),
            "raw_score": 3.0, "net_bps": 0.0,
        })
    return pd.DataFrame(rows)


def test_side_local_prior_resolved_map_is_common_bps_and_preserves_population() -> None:
    mapped, audit = apply_causal_21d_side_admission(_population(), score_column="raw_score", spec=SPEC)
    assert len(mapped) == 10
    assert mapped.candidate_id.tolist() == _population().candidate_id.tolist()
    current = mapped[mapped.candidate_id.str.endswith("current")].set_index("side_name")
    assert current.loc["long", "causal_21d_side_expected_net_bps"] > 50.0
    assert current.loc["short", "causal_21d_side_expected_net_bps"] < 50.0
    assert bool(current.loc["long", "causal_21d_side_admitted_ge_50bps"])
    assert not bool(current.loc["short", "causal_21d_side_admitted_ge_50bps"])
    assert audit.strictly_prior_resolved.all()
    supported = audit.reference_rows.gt(0)
    assert audit.loc[supported, "reference_max_label_available_ts"].lt(
        audit.loc[supported, "snapshot_utc"]
    ).all()


def test_insufficient_side_support_fails_closed_without_global_fallback() -> None:
    frame = _population().query("not candidate_id.str.contains('short-history-3')", engine="python")
    mapped, _ = apply_causal_21d_side_admission(frame, score_column="raw_score", spec=SPEC)
    short_current = mapped.loc[mapped.candidate_id.eq("short-current")].iloc[0]
    assert pd.isna(short_current.causal_21d_side_expected_net_bps)
    assert short_current.causal_21d_side_mapping_status == "unmapped_insufficient_side_support"
    assert not bool(short_current.causal_21d_side_admitted_ge_50bps)


def test_future_label_mutation_cannot_change_an_earlier_mapping() -> None:
    base = _population()
    future = pd.DataFrame({
        "candidate_id": ["long-future", "short-future"], "side_name": ["long", "short"],
        "__ts__": [pd.Timestamp("2025-02-02 00:00Z")] * 2,
        "label_available_ts": [pd.Timestamp("2025-02-02 13:00Z")] * 2,
        "raw_score": [3.0, 3.0], "net_bps": [100_000.0, -100_000.0],
    })
    left, _ = apply_causal_21d_side_admission(base, score_column="raw_score", spec=SPEC)
    right, _ = apply_causal_21d_side_admission(pd.concat([base, future], ignore_index=True), score_column="raw_score", spec=SPEC)
    columns = ["causal_21d_side_expected_net_bps", "causal_21d_side_admitted_ge_50bps"]
    left_current = left[left.candidate_id.str.endswith("current")].sort_values("candidate_id")[columns]
    right_current = right[right.candidate_id.str.endswith("current")].sort_values("candidate_id")[columns]
    pd.testing.assert_frame_equal(left_current.reset_index(drop=True), right_current.reset_index(drop=True))


def test_reference_window_includes_lower_boundary_and_excludes_snapshot() -> None:
    snapshot = pd.Timestamp("2025-02-01 00:00Z")
    rows: list[dict[str, object]] = []
    included_available = [
        snapshot - pd.Timedelta(days=21),
        snapshot - pd.Timedelta(days=15),
        snapshot - pd.Timedelta(days=8),
        snapshot - pd.Timedelta(seconds=1),
    ]
    for number, label_available in enumerate(included_available):
        rows.append({
            "candidate_id": f"included-{number}",
            "side_name": "long",
            "__ts__": label_available - pd.Timedelta(hours=13),
            "label_available_ts": label_available,
            "raw_score": float(number),
            "net_bps": float(number * 100),
        })
    for name, label_available in (
        ("too-old", snapshot - pd.Timedelta(days=21, seconds=1)),
        ("not-strictly-prior", snapshot),
    ):
        rows.append({
            "candidate_id": name,
            "side_name": "long",
            "__ts__": label_available - pd.Timedelta(hours=13),
            "label_available_ts": label_available,
            "raw_score": 10.0,
            "net_bps": 10_000.0,
        })
    rows.append({
        "candidate_id": "current",
        "side_name": "long",
        "__ts__": snapshot + pd.Timedelta(hours=12),
        "label_available_ts": snapshot + pd.Timedelta(hours=25),
        "raw_score": 3.0,
        "net_bps": 0.0,
    })

    mapped, audit = apply_causal_21d_side_admission(
        pd.DataFrame(rows), score_column="raw_score", spec=SPEC,
    )

    current = mapped.loc[mapped.candidate_id.eq("current")].iloc[0]
    assert current.causal_21d_side_reference_rows == 4
    current_audit = audit.loc[
        audit.snapshot_utc.eq(snapshot) & audit.side_name.eq("long")
    ].iloc[0]
    assert current_audit.reference_rows == 4
    assert current_audit.reference_max_label_available_ts == snapshot - pd.Timedelta(seconds=1)


def test_comparison_ranks_globally_after_admission_not_by_side_or_timestamp() -> None:
    mapped, _ = apply_causal_21d_side_admission(_population(), score_column="raw_score", spec=SPEC)
    metrics = pooled_global_admission_comparison(mapped, raw_score_column="raw_score", top_fractions=(0.10,))
    with_admission = metrics.loc[metrics.comparison.eq("with_admission_mapped_pooled_global")].iloc[0]
    # The side with weak/negative mapped EV has no quota; the remaining global
    # ranker sees the pooled admitted population only.
    assert with_admission.selected_short_rows == 0
    assert with_admission.selected_long_rows == with_admission.selected_rows


@pytest.mark.parametrize("seed", range(8))
def test_window_index_is_exactly_equivalent_to_frozen_full_scan(seed: int) -> None:
    rng = np.random.default_rng(seed)
    row_count = 350
    origin = pd.Timestamp("2024-01-01 00:00Z")
    decision = (
        origin
        + pd.to_timedelta(rng.integers(0, 55, row_count), unit="D")
        + pd.to_timedelta(rng.integers(0, 24 * 60, row_count), unit="m")
    )
    # Vary exact resolution delays so availability order is not simply decision
    # order. This exercises both half-open window boundaries and stable score
    # ties in the canonical order restored before each fit.
    available = decision + pd.to_timedelta(rng.integers(1, 73, row_count), unit="h")
    score = rng.choice(np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]), row_count).astype(float)
    target = rng.normal(20.0, 180.0, row_count)
    target[rng.choice(row_count, 7, replace=False)] = np.nan
    frame = pd.DataFrame({
        "candidate_id": [f"candidate-{seed}-{row:04d}" for row in range(row_count)],
        "side_name": rng.choice(["long", "short"], row_count),
        "__ts__": decision,
        "label_available_ts": available,
        "raw_score": score,
        "net_bps": target,
    }).sample(frac=1.0, random_state=seed)
    if seed == 0:
        # Parquet readers can retain microsecond-resolution timestamp columns;
        # the indexed implementation must not mix those integers with ns
        # Timedelta values.
        frame["__ts__"] = pd.array(frame["__ts__"], dtype="datetime64[us, UTC]")
        frame["label_available_ts"] = pd.array(
            frame["label_available_ts"], dtype="datetime64[us, UTC]",
        )
    spec = Causal21dAdmissionSpec(
        min_reference_rows=12,
        bins=7,
        trim_fraction=0.10,
        net_floor_bps=50.0,
    )

    expected, expected_audit = _reference_full_scan_admission(
        frame, score_column="raw_score", spec=spec,
    )
    actual, actual_audit = apply_causal_21d_side_admission(
        frame, score_column="raw_score", spec=spec,
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(actual_audit, expected_audit, check_exact=True)


def test_large_input_window_index_performance_sanity() -> None:
    """Exercise production-shaped routing without an O(days * rows) scan."""
    row_count = 250_000
    rows_per_day = 1_000
    row = np.arange(row_count, dtype=np.int64)
    decision = (
        pd.Timestamp("2024-01-01 00:00Z")
        + pd.to_timedelta(row // rows_per_day, unit="D")
        + pd.to_timedelta(row % (24 * 60), unit="m")
    )
    frame = pd.DataFrame({
        "candidate_id": np.char.add("large-", row.astype(str)),
        "side_name": np.where(row % 2, "long", "short"),
        "__ts__": decision,
        "label_available_ts": decision + pd.Timedelta(hours=13),
        "raw_score": np.sin(row / 127.0) + (row % 31) / 100.0,
        "net_bps": 120.0 * np.sin(row / 211.0) + (row % 17),
    })
    spec = Causal21dAdmissionSpec(
        min_reference_rows=500,
        bins=20,
        trim_fraction=0.05,
        net_floor_bps=50.0,
    )

    started = perf_counter()
    mapped, audit = apply_causal_21d_side_admission(
        frame, score_column="raw_score", spec=spec,
    )
    elapsed = perf_counter() - started

    assert len(mapped) == row_count
    assert len(audit) == 2 * (row_count // rows_per_day)
    assert audit.strictly_prior_resolved.all()
    # Generous enough for shared CI, but catches accidental restoration of a
    # complete-ledger mask for every snapshot on this 250-day population.
    assert elapsed < 12.0, f"causal admission took {elapsed:.2f}s for {row_count:,} rows"
