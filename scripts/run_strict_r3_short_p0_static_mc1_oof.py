#!/usr/bin/env python3
"""Apply frozen short MC1/BCF mappers over strict OOS conversion scores.

The mapper parameters and training are frozen before ``--start``.  At each
UTC-day boundary, the only dynamic component is the existing causal recent
global residual shift, computed from policy outcomes resolved strictly before
that boundary.  This is deliberately at least as strict as the hourly live
contract and makes all OOS day-level evidence reproducible.

The current selected-consensus and BCF all-promoted-head families are scored
separately.  Their outputs are merely joined for later admission/portfolio
ablations; this script does not choose a threshold, blend, or portfolio rule.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_bcf_mc1_mapper import BCFMC1D2Bundle
from extreme_price_movements.strict_r3_mc1_mapper import MC1D2Bundle


SIDE = "short"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    for target in ([path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())):
        digest.update(str(target.relative_to(path) if path.is_dir() else target.name).encode())
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _load(path: Path, *, family: str) -> pd.DataFrame:
    required = {
        "candidate_id", "__decision_ts__", "side_name", "final_score",
        "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
    }
    # MC1 has a six-field frozen feature contract.  This OOS scorer neither
    # consumes nor should materialise the wide upstream feature panel.
    frame = pd.read_parquet(path, columns=list(required | {
        "base_rank42", "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank", "correctness_rank",
    }))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{family} score source lacks: {missing}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{family} score source has duplicate candidate identities")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="raise",
    )
    observed = frame["side_name"].astype(str).str.strip().str.lower()
    if frame.empty or not observed.eq(SIDE).all():
        raise ValueError(f"{family} source is not strictly short-local")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _daily_score(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    score: Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp], pd.DataFrame],
    family: str,
) -> pd.DataFrame:
    """Score held days under a frozen mapper and strict prior-day ledger.

    The mapper receives only valid scores with a policy label resolved
    *strictly before* the UTC-day decision boundary.  This matters because
    the legacy current-MC1 bundle accepts ``<= decision`` internally; the
    driver deliberately imposes the stronger strict-prequential contract.
    Restricting to finite final scores also prevents score-band construction
    from silently assigning a NaN score to a band.  The boundary is fixed at
    UTC midnight so held intraday labels cannot change earlier decisions in
    this reproducible OOS evaluation.
    """
    held = frame.loc[
        frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)
    ].copy()
    if held.empty:
        raise ValueError(f"{family} has no rows in requested held window")
    pieces: list[pd.DataFrame] = []
    # The frozen mapper uses only the prior-resolved *21-day* history.  Build
    # that exact window incrementally instead of copying the full OOS ledger
    # for every held day.  This leaves score values unchanged while reducing
    # memory from the wide 1.7m-row panel to roughly 22 compact daily blocks.
    valid_history = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(frame["policy_net_bps"], errors="coerce").notna()
        & pd.to_numeric(frame["final_score"], errors="coerce").notna(),
    ].sort_values("policy_label_available_ts", kind="stable").reset_index(drop=True)
    label_ns = valid_history["policy_label_available_ts"].astype("int64").to_numpy()
    cursor = 0
    active: deque[pd.DataFrame] = deque()
    for day, current in held.groupby(held["__decision_ts__"].dt.normalize(), sort=True):
        decision = _utc(day)
        end = int(np.searchsorted(label_ns, decision.value, side="left"))
        if end > cursor:
            candidate_window_start = decision - pd.Timedelta(days=21)
            added = valid_history.iloc[cursor:end]
            added = added.loc[added["__decision_ts__"].ge(candidate_window_start)]
            if not added.empty:
                active.append(added)
            cursor = end
        candidate_window_start = decision - pd.Timedelta(days=21)
        while active and active[0]["__decision_ts__"].max() < candidate_window_start:
            active.popleft()
        history = (
            pd.concat(active, ignore_index=True).loc[
                lambda value: value["__decision_ts__"].ge(candidate_window_start)
            ].copy()
            if active else valid_history.iloc[0:0].copy()
        )
        result = score(current.copy(), history, decision)
        if len(result) != len(current) or set(result["candidate_id"]) != set(current["candidate_id"]):
            raise AssertionError(f"{family} mapper changed current candidate identities")
        result["mc1_decision_day"] = decision
        result["mc1_history_label_cutoff"] = decision
        pieces.append(current.merge(result, on="candidate_id", how="inner", validate="one_to_one"))
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)


def _metric_rows(frame: pd.DataFrame, *, family: str, expected: str) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    work = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(frame["policy_net_bps"], errors="coerce").notna()
    ].copy()
    work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    for month, block in [("pooled", work), *work.groupby("month", sort=True)]:
        for threshold in (30.0, 50.0):
            selected = block.loc[pd.to_numeric(block[expected], errors="coerce").ge(threshold)]
            net = pd.to_numeric(selected["policy_net_bps"], errors="coerce")
            output.append({
                "family": family, "month": month, "threshold_bps": threshold,
                "trades": int(len(selected)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "total_net_bps": float(net.sum()) if len(net) else np.nan,
            })
    return pd.DataFrame(output)


def _dual_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate dual admission, never BCF-only admission under a dual name."""
    output: list[dict[str, object]] = []
    work = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(frame["policy_net_bps"], errors="coerce").notna()
    ].copy()
    work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    for month, block in [("pooled", work), *work.groupby("month", sort=True)]:
        for threshold in (30.0, 50.0):
            selected = block.loc[
                pd.to_numeric(block["mc1_d2_expected_net_bps"], errors="coerce").ge(threshold)
                & pd.to_numeric(block["bcf_mc1_expected_net_bps"], errors="coerce").ge(threshold)
            ]
            net = pd.to_numeric(selected["policy_net_bps"], errors="coerce")
            output.append({
                "family": "dual_mc1",
                "month": month,
                "threshold_bps": threshold,
                "trades": int(len(selected)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "total_net_bps": float(net.sum()) if len(net) else np.nan,
            })
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-scores", type=Path, required=True)
    parser.add_argument("--current-bundle", type=Path, required=True)
    parser.add_argument("--bcf-scores", type=Path, default=None)
    parser.add_argument("--bcf-bundle", type=Path, default=None)
    parser.add_argument("--start", default="2025-07-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-08-01T00:00:00Z")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable OOS mapper output exists: {args.out}")
    if (args.bcf_scores is None) != (args.bcf_bundle is None):
        raise ValueError("BCF score and bundle must be supplied together")
    start, end = _utc(args.start), _utc(args.end_exclusive)
    if end <= start:
        raise ValueError("end must follow start")
    current_bundle = MC1D2Bundle.load(args.current_bundle)
    if current_bundle.side != SIDE:
        raise ValueError("current MC1 bundle is not short-local")
    current = _load(args.current_scores, family="current")
    current_scored = _daily_score(
        current, start=start, end=end,
        score=lambda rows, history, decision: current_bundle.score(
            rows, resolved_history=history, decision_ts=decision,
        ),
        family="current",
    )
    args.out.mkdir(parents=True)
    current_scored.to_parquet(
        args.out / "short_current_mc1_oof_predictions.parquet",
        index=False, compression="zstd",
    )
    metrics = [_metric_rows(
        current_scored, family="current_mc1", expected="mc1_d2_expected_net_bps",
    )]
    manifest: dict[str, object] = {
        "schema": "strict_r3_short_p0_static_mc1_oof_v1",
        "status": "complete", "side": SIDE,
        "held_window": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "dynamic_calibration": (
            "frozen mapper plus causal recent-global shift from policy labels "
            "resolved strictly before each UTC-day decision boundary"
        ),
        "current": {
            "scores": {"path": str(args.current_scores), "sha256": _sha(args.current_scores)},
            "bundle": {"path": str(args.current_bundle), "sha256": _sha(args.current_bundle)},
        },
    }
    if args.bcf_scores is not None:
        bcf_bundle = BCFMC1D2Bundle.load(args.bcf_bundle)
        if bcf_bundle.side != SIDE:
            raise ValueError("BCF MC1 bundle is not short-local")
        bcf = _load(args.bcf_scores, family="BCF")
        bcf_scored = _daily_score(
            bcf, start=start, end=end,
            score=lambda rows, history, decision: bcf_bundle.score(
                rows, resolved_history=history, decision_ts=decision,
            ),
            family="BCF",
        )
        bcf_scored.to_parquet(
            args.out / "short_bcf_mc1_oof_predictions.parquet",
            index=False, compression="zstd",
        )
        overlap = current_scored.merge(
            bcf_scored.loc[:, ["candidate_id", "bcf_mc1_expected_net_bps", "bcf_mc1_available"]],
            on="candidate_id", how="outer", validate="one_to_one", indicator=True,
        )
        if not overlap["_merge"].eq("both").all():
            raise AssertionError("current and BCF held score populations differ")
        dual = overlap.drop(columns="_merge")
        dual["dual_admitted_ge_30bps"] = (
            dual["mc1_d2_expected_net_bps"].ge(30.0)
            & dual["bcf_mc1_expected_net_bps"].ge(30.0)
        )
        dual.to_parquet(
            args.out / "short_dual_mc1_oof_predictions.parquet",
            index=False, compression="zstd",
        )
        metrics.append(_metric_rows(
            bcf_scored, family="bcf_mc1", expected="bcf_mc1_expected_net_bps",
        ))
        metrics.append(_dual_metric_rows(dual))
        manifest["bcf"] = {
            "scores": {"path": str(args.bcf_scores), "sha256": _sha(args.bcf_scores)},
            "bundle": {"path": str(args.bcf_bundle), "sha256": _sha(args.bcf_bundle)},
            "dual_admission": "current MC1 >=30 and BCF MC1 >=30; comparison only",
        }
    pd.concat(metrics, ignore_index=True).to_parquet(
        args.out / "mapper_threshold_metrics.parquet", index=False, compression="zstd",
    )
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
