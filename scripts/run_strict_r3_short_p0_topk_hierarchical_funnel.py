#!/usr/bin/env python3
"""Short-only top-K continuation after the absolute-hour gate.

This is intentionally downstream of the M0–M8 absolute-conversion screen.
It compares three predeclared structures without creating a short consensus:

* A: top-1 P0 winner + M4 hour opportunity gate;
* B: P0 top-4 + candidate absolute residual filter, retaining P0 order;
* C: M4 hour gate -> P0 top-4 -> candidate residual filter, retaining P0
  order.

All rows are selected from the target-free P0 score universe.  Outcomes are
joined only after selection.  Per-month fitting uses only labels resolved
strictly before the held month.  This script is short-only and does not alter
the long pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_short_p0_absolute_conversion_funnel as absolute


SIDE = "short"
TOP_K = 4
HOUR_ARM = absolute.Arm(
    "M4_hour_gate", "ordinal", ("base",),
    "Frozen M4 ordinal absolute policy-margin hour gate.",
)
CANDIDATE_ARM = absolute.Arm(
    "C_residual", "residual", ("base", "geometry", "market"),
    "Candidate absolute policy residual; it filters but never reranks P0.",
)
FILTER_LEVELS = (0.0, 50.0)
# Frozen M4 OOF predictions retain an isotonic plateau in a serialized float
# representation while the historical p80 was kept at higher precision.  This
# is not an economic tolerance: it solely preserves members of the exact p80
# score plateau.  A 1e-4 bps band is far below any economically meaningful
# distinction and is recorded in the immutable manifest.
M4_PLATEAU_TIE_TOLERANCE_BPS = 1e-4


def _topk_population(ledger: pd.DataFrame, top1: pd.DataFrame) -> pd.DataFrame:
    eligible = ledger.loc[
        ledger["base_feature_eligible"].fillna(False).astype(bool)
        & absolute._finite(ledger["prequential_base_score"]).notna()
    ].copy()
    pieces: list[pd.DataFrame] = []
    for decision, group in eligible.groupby("__decision_ts__", sort=True):
        ordered = group.sort_values(
            ["prequential_base_score", "candidate_id"], ascending=[False, True], kind="stable",
        ).head(TOP_K).copy()
        ordered["p0_rank_within_hour"] = np.arange(1, len(ordered) + 1, dtype=np.int8)
        pieces.append(ordered)
    result = pd.concat(pieces, ignore_index=True)
    inherited = [
        "__decision_ts__", *[field for field in top1.columns if field.startswith(("geom_", "market__", "recent_"))],
        "decision_hour_utc", "decision_weekend",
    ]
    inherited = list(dict.fromkeys(inherited))
    result = result.merge(top1.loc[:, inherited], on="__decision_ts__", how="left", validate="many_to_one")
    if result.candidate_id.duplicated().any():
        raise AssertionError("short top-K target-free selection duplicated a candidate")
    counts = result.groupby("__decision_ts__").size()
    if counts.gt(TOP_K).any() or counts.lt(1).any():
        raise AssertionError("short top-K selection does not have one-to-four rows per hour")
    return result.sort_values(["__decision_ts__", "p0_rank_within_hour", "candidate_id"], kind="stable").reset_index(drop=True)


def _selection_metrics(frame: pd.DataFrame, *, name: str, held_month: pd.Timestamp) -> dict[str, Any]:
    valid = frame.loc[absolute._valid_policy(frame)].copy()
    net = absolute._finite(valid["p0_canonical_net_bps"])
    return {
        "selection": name,
        "held_month": held_month.strftime("%Y-%m"),
        "trades": int(len(valid)),
        "net_bps_per_trade": float(net.mean()) if len(valid) else np.nan,
        "total_net_bps": float(net.sum()) if len(valid) else np.nan,
        "positive_rate": float((net > 0.0).mean()) if len(valid) else np.nan,
        "hours": int(valid["__decision_ts__"].nunique()),
        "mean_p0_rank": float(pd.to_numeric(valid["p0_rank_within_hour"], errors="coerce").mean()) if len(valid) else np.nan,
    }


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for selection, block in monthly.groupby("selection", sort=True):
        trades = int(block.trades.sum())
        total = float(block.total_net_bps.sum(min_count=1))
        rows.append({
            "selection": selection,
            "months": int(block.held_month.nunique()),
            "trades": trades,
            "net_bps_per_trade": total / trades if trades else np.nan,
            "total_net_bps": total,
            "worst_month_net_bps_per_trade": float(block.net_bps_per_trade.min()),
            "positive_months": int((block.net_bps_per_trade > 0).sum()),
            "mean_p0_rank": float(block.mean_p0_rank.mean()),
        })
    return pd.DataFrame(rows).sort_values("net_bps_per_trade", ascending=False, kind="stable")


def _load_frozen_hour_m4(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the exact M4 OOF gate rather than silently refitting it.

    The top-K continuation is a downstream architecture comparison.  Its A
    control must use the exact M4 score/calibration values produced by the
    completed absolute-conversion funnel, including its fold seed and OOF
    isotonic map.  Re-fitting here would create a different hour gate.
    """
    required = {
        "candidate_id", "__decision_ts__", "side_name", "arm", "expected_net_bps", "train_p80_expected_bps",
    }
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for path in paths:
        available = set(pd.read_parquet(path, columns=None).columns)
        missing = sorted(required.difference(available))
        if missing:
            raise ValueError(f"absolute M4 OOF source lacks: {missing}")
        frame = pd.read_parquet(path, columns=sorted(required))
        frame = frame.loc[frame.arm.astype(str).eq("M4")].copy()
        hashes[str(path)] = absolute._sha256(path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result.empty or result.candidate_id.duplicated().any():
        raise ValueError("frozen M4 OOF sources are empty or overlap candidate identities")
    if not result.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError("frozen M4 OOF source is not short-local")
    return result.rename(columns={
        "expected_net_bps": "hour_m4_expected_net_bps",
        "train_p80_expected_bps": "hour_m4_train_p80_bps",
    }).drop(columns=["arm"]), hashes


def _m4_admitted(expected: pd.Series, threshold: pd.Series) -> pd.Series:
    """Causal M4 p80 gate with explicit serialized-isotonic plateau ties."""
    return absolute._finite(expected).ge(
        absolute._finite(threshold) - M4_PLATEAU_TIE_TOLERANCE_BPS,
    )


def _frozen_hour_top1_selection(held_hours: pd.DataFrame, hour_pred: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact frozen M4 gate without candidate-meta availability.

    A is a control for the completed absolute-conversion funnel.  It must not
    inherit the candidate residual model's feature-completeness requirement:
    doing so changes its selected identities and turns the comparison into an
    accidental feature-coverage ablation.
    """
    selected = held_hours.copy()
    selected["p0_rank_within_hour"] = np.int8(1)
    selected = selected.merge(
        hour_pred.loc[:, ["candidate_id", "__decision_ts__", "hour_m4_expected_net_bps", "hour_m4_train_p80_bps"]],
        on=["candidate_id", "__decision_ts__"],
        how="left",
        validate="one_to_one",
    )
    if selected.hour_m4_expected_net_bps.isna().any() or selected.hour_m4_train_p80_bps.isna().any():
        missing = selected.loc[selected.hour_m4_expected_net_bps.isna(), "candidate_id"].head(5).tolist()
        raise ValueError(f"frozen M4 OOF source lacks held P0 winners: {missing}")
    selected["hour_m4_admitted"] = _m4_admitted(
        selected.hour_m4_expected_net_bps, selected.hour_m4_train_p80_bps,
    )
    return selected.loc[selected.hour_m4_admitted].copy()


def run(*, ledger_roots: list[Path], hour_m4_paths: list[Path], start: pd.Timestamp, end_exclusive: pd.Timestamp, out: Path, seed: int) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable short top-K output exists: {out}")
    ledger, source_hashes = absolute._load_ledger(ledger_roots)
    top1 = absolute._add_recent_conversion_state(absolute._top1_population(ledger))
    top4 = _topk_population(ledger, top1)
    frozen_m4, m4_hashes = _load_frozen_hour_m4(hour_m4_paths)
    candidate_blocks = absolute._feature_blocks(top4)
    out.mkdir(parents=True)
    top4.to_parquet(out / "short_p0_top4_hourly_population.parquet", index=False, compression="zstd")
    predicted: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for fold, month in enumerate(pd.date_range(start.normalize().replace(day=1), end_exclusive, freq="MS", inclusive="left")):
        next_month = month + pd.offsets.MonthBegin(1)
        held_hours = top1.loc[top1.__decision_ts__.ge(month) & top1.__decision_ts__.lt(next_month)].copy()
        held_candidates = top4.loc[top4.__decision_ts__.ge(month) & top4.__decision_ts__.lt(next_month)].copy()
        train_hours = top1.loc[
            top1.policy_label_available_at.lt(month) & top1.__decision_ts__.lt(month) & absolute._valid_policy(top1)
        ].copy()
        train_candidates = top4.loc[
            top4.policy_label_available_at.lt(month) & top4.__decision_ts__.lt(month) & absolute._valid_policy(top4)
        ].copy()
        if len(train_hours) < absolute.MIN_TRAIN_ROWS or len(train_candidates) < absolute.MIN_TRAIN_ROWS:
            audit.append({"held_month": month.strftime("%Y-%m"), "status": "skipped_insufficient_train", "train_hours": len(train_hours), "train_candidates": len(train_candidates)})
            continue
        hour_pred = held_hours.loc[:, ["candidate_id", "__decision_ts__"]].merge(
            frozen_m4.loc[:, ["candidate_id", "__decision_ts__", "hour_m4_expected_net_bps", "hour_m4_train_p80_bps"]],
            on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one",
        )
        if hour_pred.hour_m4_expected_net_bps.isna().any() or hour_pred.hour_m4_train_p80_bps.isna().any():
            missing = hour_pred.loc[hour_pred.hour_m4_expected_net_bps.isna(), "candidate_id"].head(5).tolist()
            raise ValueError(f"frozen M4 OOF source lacks held P0 winners: {missing}")
        # A is exactly the completed M4 hourly gate.  Keep it outside the
        # candidate residual path so its selected IDs remain bit-for-bit
        # comparable with the source funnel.
        a = _frozen_hour_top1_selection(held_hours, hour_pred)
        metrics.append(_selection_metrics(a, name="A_top1_hour_M4", held_month=month))
        candidate_pred, candidate_details = absolute._fit_predict_arm(
            train_candidates, held_candidates, CANDIDATE_ARM, candidate_blocks, seed=seed + fold * 100 + 2,
        )
        hour_gate = hour_pred
        # ``candidate_pred`` owns the candidate identity.  The frozen M4
        # source contributes one causal *hour* gate, so merging its own
        # candidate_id would create pandas ``_x/_y`` identities and silently
        # break the stated P0-order preservation below.
        candidate_pred = candidate_pred.merge(
            hour_gate.loc[:, ["__decision_ts__", "hour_m4_expected_net_bps", "hour_m4_train_p80_bps"]],
            on="__decision_ts__",
            how="left",
            validate="many_to_one",
        )
        candidate_pred["hour_m4_admitted"] = _m4_admitted(
            candidate_pred.hour_m4_expected_net_bps, candidate_pred.hour_m4_train_p80_bps,
        )
        candidate_pred = candidate_pred.rename(columns={"expected_net_bps": "candidate_expected_net_bps"})
        # B/C: candidate model has authority only to filter.  Within the
        # survivors the original P0/F90 rank remains the auction priority.
        for threshold in FILTER_LEVELS:
            candidate_ok = absolute._finite(candidate_pred.candidate_expected_net_bps).ge(threshold)
            b = (
                candidate_pred.loc[candidate_ok]
                .sort_values(["__decision_ts__", "p0_rank_within_hour", "candidate_id"], kind="stable")
                .groupby("__decision_ts__", sort=False, as_index=False).head(1).copy()
            )
            c = (
                candidate_pred.loc[candidate_ok & candidate_pred.hour_m4_admitted]
                .sort_values(["__decision_ts__", "p0_rank_within_hour", "candidate_id"], kind="stable")
                .groupby("__decision_ts__", sort=False, as_index=False).head(1).copy()
            )
            suffix = int(threshold)
            metrics.append(_selection_metrics(b, name=f"B_top4_candidate_ev_ge_{suffix}", held_month=month))
            metrics.append(_selection_metrics(c, name=f"C_hierarchical_candidate_ev_ge_{suffix}", held_month=month))
        candidate_pred["held_month"] = month.strftime("%Y-%m")
        predicted.append(candidate_pred.loc[:, [
            "candidate_id", "__decision_ts__", "__symbol__", "side_name", "held_month", "p0_rank_within_hour",
            "prequential_base_score", "prequential_base_rank42", "prequential_base_anchor_bps",
            "candidate_expected_net_bps", "hour_m4_expected_net_bps", "hour_m4_train_p80_bps", "hour_m4_admitted",
            "policy_path_valid", "policy_label_available_at", "p0_canonical_net_bps",
        ]])
        audit.append({
            "held_month": month.strftime("%Y-%m"), "status": "complete", "train_hours": len(train_hours),
            "train_candidates": len(train_candidates), "held_hours": len(held_hours), "held_candidates": len(held_candidates),
            "hour_model": "exact completed M4 OOF source", "candidate_model": candidate_details,
        })
    if not predicted:
        raise RuntimeError("short top-K continuation produced no strict-OOS predictions")
    predictions = pd.concat(predicted, ignore_index=True)
    predictions.to_parquet(out / "short_p0_topk_hierarchical_oof_predictions.parquet", index=False, compression="zstd")
    monthly = pd.DataFrame(metrics)
    monthly.to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    _aggregate(monthly).to_parquet(out / "aggregate_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(audit).to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_p0_topk_hierarchical_funnel_v1", "status": "complete", "side": SIDE,
        "top_k": TOP_K,
        "hour_gate": "exact completed M4 ordinal OOF score/calibration, causal training p80 expected-bps threshold",
        "candidate_model": "absolute policy residual Huber model; candidate score is an admission filter only",
        "priority": "frozen P0/F90 rank within each hour; no candidate-meta reranking",
        "candidate_filter_thresholds_bps": list(FILTER_LEVELS),
        "hour_m4_plateau_tie_tolerance_bps": M4_PLATEAU_TIE_TOLERANCE_BPS,
        "strict_prequential": "all model labels resolve strictly before held month; target-free selection is performed before outcomes are joined",
        "source_hashes": source_hashes,
        "frozen_hour_m4_sources": m4_hashes,
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, action="append", required=True)
    parser.add_argument("--hour-m4-predictions", type=Path, action="append", required=True)
    parser.add_argument("--start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-08-01T00:00:00Z")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start, end = absolute._utc(args.start), absolute._utc(args.end_exclusive)
    if end <= start:
        raise ValueError("end must follow start")
    print(run(ledger_roots=args.ledger_root, hour_m4_paths=args.hour_m4_predictions, start=start, end_exclusive=end, out=args.out, seed=args.seed))


if __name__ == "__main__":
    main()
