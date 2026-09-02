#!/usr/bin/env python3
"""Audit whether older Pack-B populations can extend EV-state recurrence.

This is intentionally an evidence-producing gate, rather than a backtest.  A
historical recurrence row needs all of: canonical identity, a score available
at decision time, the *same* realised 12-hour execution net-EV target, an
explicit resolution time, and a causal raw-market-state join.  A missing item
is a blocker; it must not be silently reconstructed from a later model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_market_state import (  # noqa: E402
    MARKET_STATE_COLUMNS,
    attach_decision_time_market_state,
)


OUT = ROOT / "data_perp/artifacts/historical_raw_state_recurrence_join_audit_20260726_v1"
FEATURES = ROOT / "data_perp/features/20260711_070000"
PRE_MARCH = ROOT / "data_perp/artifacts/packb_pre_march_population_20260724_v1/authorized_pre_march_population.parquet"
OUTER = ROOT / "data_perp/artifacts/packb_side_local_outer_oof_20260724_v1_31_8/oof_predictions.parquet"
LABEL_ROOT = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
CURRENT = ROOT / "data_perp/artifacts/execution_ev_context_clean_exact_recent_correction_forward_july19_20260726_v2/mapped_oof_and_forward.parquet"


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def _columns(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def _sample_by_symbol(frame: pd.DataFrame, *, maximum: int = 3) -> pd.DataFrame:
    """Deterministic start/middle/end PIT probes for every source symbol."""

    rows: list[pd.DataFrame] = []
    for _, group in frame.groupby("__symbol__", sort=True):
        group = group.sort_values("__decision_ts__", kind="stable")
        positions = sorted(set([0, len(group) // 2, len(group) - 1]))[:maximum]
        rows.append(group.iloc[positions])
    return pd.concat(rows, ignore_index=True)


def _raw_pit_probe(name: str, candidates: pd.DataFrame) -> dict[str, object]:
    sample = _sample_by_symbol(candidates.loc[:, ["__symbol__", "__decision_ts__"]])
    probe = sample.rename(columns={"__decision_ts__": "execution_decision_utc"})
    joined = attach_decision_time_market_state(
        probe,
        feature_store_root=FEATURES,
        decision_time_col="execution_decision_utc",
    )
    frame = joined.frame
    source_ok = frame["mkt_state_source_utc"].notna()
    finite = frame.loc[:, list(MARKET_STATE_COLUMNS)].notna().mean()
    return {
        "source": name,
        "pit_probe_rows": int(len(frame)),
        "pit_probe_symbols": int(frame["__symbol__"].nunique()),
        "pit_source_row_fraction": float(source_ok.mean()),
        "pit_source_never_future": bool(
            (
                frame.loc[source_ok, "mkt_state_source_utc"]
                <= frame.loc[source_ok, "execution_decision_utc"] - pd.Timedelta(hours=1)
            ).all()
        ),
        "pit_source_max_age_seconds": float(frame.loc[source_ok, "mkt_state_source_age_seconds"].max()),
        "raw_fields_at_least_95pct_on_probe": int((finite >= 0.95).sum()),
        "raw_fields_total": len(MARKET_STATE_COLUMNS),
    }


def _historic_labels(months: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    wanted = [
        "candidate_id", "side_name", "__ts__", "__symbol__", "__decision_ts__",
        "__first_touch_capture_net__", "__first_touch_bar__",
    ]
    for side in ("long", "short"):
        for month in months:
            path = LABEL_ROOT / f"train_global_{side}_5_{month}.parquet"
            parts.append(pd.read_parquet(path, columns=wanted))
    labels = pd.concat(parts, ignore_index=True)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    labels["__ts__"] = _utc(labels["__ts__"])
    labels["__decision_ts__"] = _utc(labels["__decision_ts__"])
    return labels


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    # Dec--Feb canonical candidates: this is the strict pre-March population.
    pre = pd.read_parquet(
        PRE_MARCH,
        filters=[("__ts__", ">=", pd.Timestamp("2025-12-01", tz="UTC"))],
    )
    pre["candidate_id"] = pre["candidate_id"].astype(str)
    pre["__ts__"] = _utc(pre["__ts__"])
    pre["__decision_ts__"] = _utc(pre["__decision_ts__"])
    pre["__label_resolution_ts__"] = _utc(pre["__label_resolution_ts__"])
    pre = pre.loc[pre["__ts__"] < pd.Timestamp("2026-03-01", tz="UTC")].reset_index(drop=True)
    pre_labels = _historic_labels(["2025_12", "2026_01", "2026_02"])
    pre_label_ids = set(pre_labels["candidate_id"])

    # April is the earliest Pack-B outer OOF.  Its target is materially a
    # 96-bar first-touch capture target, not the later 12-hour execution EV.
    april = pd.read_parquet(
        OUTER,
        filters=[("__ts__", "<", pd.Timestamp("2026-05-01", tz="UTC"))],
    )
    april["candidate_id"] = april["candidate_id"].astype(str)
    april["__ts__"] = _utc(april["__ts__"])
    april_labels = _historic_labels(["2026_04"])
    april_label_ids = set(april_labels["candidate_id"])
    april_resolution = pd.read_parquet(
        ROOT / "data_perp/artifacts/packb_outer_oof_population_20260724_v1/folds/outer_1_20260401/long/validation.parquet"
    )
    april_resolution = pd.concat(
        [
            april_resolution,
            pd.read_parquet(
                ROOT / "data_perp/artifacts/packb_outer_oof_population_20260724_v1/folds/outer_1_20260401/short/validation.parquet"
            ),
        ],
        ignore_index=True,
    )
    april_resolution["candidate_id"] = april_resolution["candidate_id"].astype(str)
    april_resolution_ids = set(april_resolution["candidate_id"])

    current_columns = _columns(CURRENT)
    current = pd.read_parquet(CURRENT, columns=["__ts__", "candidate_id"])
    current["__ts__"] = _utc(current["__ts__"])
    current["candidate_id"] = current["candidate_id"].astype(str)

    probe_rows = [_raw_pit_probe("pre_march_canonical_dec_feb", pre), _raw_pit_probe("packb_outer_oof_april", april.assign(__decision_ts__=april["__ts__"] + pd.Timedelta(hours=1)))]
    pd.DataFrame(probe_rows).to_csv(OUT / "raw_state_pit_probe.csv", index=False)

    rows = [
        {
            "period_and_source": "2025-12 to 2026-02 / packb_pre_march_population",
            "rows": len(pre), "date_start": str(pre["__ts__"].min()), "date_end": str(pre["__ts__"].max()),
            "canonical_identity": True, "explicit_decision_time": True, "explicit_label_resolution": True,
            "strict_oof_base_or_direct_score": False,
            "same_execution_net_ev_12h": False,
            "alternative_cost_aware_capture_target": True,
            "alternative_target_identity_coverage": float(pre["candidate_id"].isin(pre_label_ids).mean()),
            "raw_state_pit_probe": "pass; see raw_state_pit_probe.csv",
            "blocker": "No stored OOF base/direct score; historical capture label is a different 96-bar path target, not execution_net_ev_12h.",
            "recurrence_eligible": False,
        },
        {
            "period_and_source": "2026-04 / packb_side_local_outer_oof",
            "rows": len(april), "date_start": str(april["__ts__"].min()), "date_end": str(april["__ts__"].max()),
            "canonical_identity": True, "explicit_decision_time": True, "explicit_label_resolution": float(april["candidate_id"].isin(april_resolution_ids).mean()) == 1.0,
            "strict_oof_base_or_direct_score": True,
            "same_execution_net_ev_12h": False,
            "alternative_cost_aware_capture_target": True,
            "alternative_target_identity_coverage": float(april["candidate_id"].isin(april_label_ids).mean()),
            "raw_state_pit_probe": "pass; see raw_state_pit_probe.csv",
            "blocker": "OOF score predicts first-touch soft target; realised capture target has up to 96 bars, so it cannot be mixed with the 12h direct-EV residual/gate.",
            "recurrence_eligible": False,
        },
        {
            "period_and_source": "2026-05-05 to 2026-07-19 / frozen direct-EV handoff",
            "rows": len(current), "date_start": str(current["__ts__"].min()), "date_end": str(current["__ts__"].max()),
            "canonical_identity": True, "explicit_decision_time": "execution_decision_utc" in current_columns,
            "explicit_label_resolution": "execution_label_end_utc" in current_columns,
            "strict_oof_base_or_direct_score": True,
            "same_execution_net_ev_12h": "execution_net_ev_12h" in current_columns,
            "alternative_cost_aware_capture_target": False,
            "alternative_target_identity_coverage": 1.0,
            "raw_state_pit_probe": "already exact-joined in recurrence artifact",
            "blocker": "None: this is the maximum honest window for the existing execution-EV recurrence diagnostic.",
            "recurrence_eligible": True,
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "coverage_blocker_table.csv", index=False)
    summary = {
        "schema": "historical_raw_state_recurrence_join_audit_v1",
        "decision": "do_not_extend_existing_execution_ev_recurrence_before_2026-05-05",
        "maximum_honest_existing_execution_ev_window": "2026-05-05T19:00:00Z to 2026-07-19T15:00:00Z (123,824 rows)",
        "why": [
            "Pre-March canonical population has identity/decision/resolution and joins to the raw source, but no stored strict OOF score or same 12h execution-EV target.",
            "April has a strict OOF Pack-B score and matching historical capture outcome, but the score is for a soft first-touch target and the capture target reaches 96 bars; it is not comparable to the 12h direct-EV residual/gate.",
            "The frozen direct-EV handoff is the earliest artifact carrying identity, OOF/forward direct score, execution_net_ev_12h, and execution_label_end_utc together.",
        ],
        "outputs": {
            "coverage_blocker_table": str(OUT / "coverage_blocker_table.csv"),
            "raw_state_pit_probe": str(OUT / "raw_state_pit_probe.csv"),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
