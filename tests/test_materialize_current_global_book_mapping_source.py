from __future__ import annotations

import pandas as pd
import pytest

from scripts.materialize_current_global_book_mapping_source import (
    FORWARD_FLAG,
    GLOBAL_MAPPED_SCORE,
    MAPPED_SCORE,
    OOF_FLAG,
    RAW_SCORE,
    build_current_mapping_source,
)


def _sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    timestamps = pd.date_range("2026-05-01", periods=72, freq="h", tz="UTC")
    rows = []
    handoff = []
    for index, stamp in enumerate(timestamps):
        side = "long" if index % 2 == 0 else "short"
        gross = 0.02 + index / 100_000.0
        cost = 0.01
        identity = {
            "candidate_id": f"c{index:03d}",
            "__ts__": stamp,
            "__symbol__": "BTCUSD",
            "side_name": side,
        }
        rows.append(
            {
                **identity,
                RAW_SCORE: float(index),
                OOF_FLAG: True,
                FORWARD_FLAG: False,
                "promotion_eligible": True,
                "evaluation_origin": "historical_outer_oof",
                "execution_decision_utc": stamp + pd.Timedelta(hours=1),
                "execution_label_end_utc": stamp + pd.Timedelta(hours=13),
                "execution_gross_ev_12h": gross,
                "execution_net_ev_12h": gross - cost,
            }
        )
        handoff.append(
            {
                **identity,
                "execution_decision_utc": stamp + pd.Timedelta(hours=1),
                "execution_label_end_utc": stamp + pd.Timedelta(hours=13),
                "execution_gross_ev_12h": gross,
                "execution_cost_return": cost,
                "execution_net_ev_12h": gross - cost,
                "execution_mfe_return_12h": gross + 0.01,
                "execution_mae_return_12h": -0.01,
                "execution_exit_reason": "full_stop",
                "execution_exit_hour": 1.0,
            }
        )
    mapped = pd.DataFrame(rows)
    historical = pd.DataFrame(handoff)
    # The small fixture intentionally lowers the support threshold only at the
    # report level; build still requires the canonical 500, so repeat exact
    # identities across symbols to reach production-like support.
    copies = []
    handoff_copies = []
    for copy in range(20):
        left = mapped.copy()
        right = historical.copy()
        left["candidate_id"] += f"_{copy:02d}"
        right["candidate_id"] += f"_{copy:02d}"
        left["__symbol__"] = f"S{copy:02d}"
        right["__symbol__"] = f"S{copy:02d}"
        copies.append(left)
        handoff_copies.append(right)
    mapped = pd.concat(copies, ignore_index=True)
    historical = pd.concat(handoff_copies, ignore_index=True)
    audit = []
    for day, group in mapped.groupby(
        mapped["execution_decision_utc"].dt.floor("D"), sort=True
    ):
        resolved = mapped["execution_label_end_utc"]
        reference = resolved.lt(day) & resolved.ge(day - pd.Timedelta(days=21))
        if int(reference.sum()) >= 500:
            audit.append(
                {
                    "snapshot": day.isoformat(),
                    "reference_rows": int(reference.sum()),
                    "long_reference_rows": int(
                        (reference & mapped["side_name"].eq("long")).sum()
                    ),
                    "short_reference_rows": int(
                        (reference & mapped["side_name"].eq("short")).sum()
                    ),
                }
            )
    report = {
        "contract": {
            "window_days": 21,
            "min_reference_rows": 500,
            "per_timestamp_quota": False,
            "ranking_scope": "global pooled across timestamps and sides",
        },
        "daily_audit": audit,
    }
    return mapped, historical, historical.iloc[0:0].copy(), report


def test_build_binds_exact_economics_and_normalises_exit() -> None:
    mapped, historical, forward, report = _sources()
    result, audit, stats = build_current_mapping_source(
        mapped, historical, report
    )
    assert len(result) == len(mapped)
    assert result["execution_exit_class"].eq("full_stop").all()
    assert result["execution_mfe_return_12h"].notna().all()
    assert result["execution_mae_return_12h"].notna().all()
    assert (
        result["execution_gross_ev_12h"] - result["execution_cost_return"]
    ).equals(result["execution_net_ev_12h"])
    assert stats["warmup_unmapped_rows"] > 0
    assert stats["mapped_eligible_rows"] > 0
    assert result[GLOBAL_MAPPED_SCORE].notna().sum() == stats["mapped_eligible_rows"]
    assert result["mapped_global_direct_net"].equals(result[GLOBAL_MAPPED_SCORE])
    assert audit["reference_label_end_max_utc"].dropna().lt(
        audit.loc[
            audit["reference_label_end_max_utc"].notna(), "snapshot_utc"
        ]
    ).all()


def test_build_fails_when_handoff_economics_change() -> None:
    mapped, historical, forward, report = _sources()
    historical.loc[0, "execution_net_ev_12h"] += 0.001
    with pytest.raises(ValueError, match="accounting"):
        build_current_mapping_source(mapped, historical, report)


def test_build_rejects_promotable_forward_rows() -> None:
    mapped, historical, forward, report = _sources()
    mapped.loc[0, OOF_FLAG] = False
    mapped.loc[0, FORWARD_FLAG] = True
    with pytest.raises(ValueError, match="nonpromotable"):
        build_current_mapping_source(mapped, historical, report)
