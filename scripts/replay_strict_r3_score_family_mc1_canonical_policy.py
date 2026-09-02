#!/usr/bin/env python3
"""Matched BCF/current-v5 MC1_d2 replay on one canonical policy substrate.

The score families remain distinct target-free candidate routes.  This runner
repairs only the outcome substrate: all policy fields are attached from one
source-aligned parent materialisation, invalid paths are excluded before MC1
fits and before portfolio capacity is allocated, and every fold is strictly
prequential on ``policy_label_available_ts``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_strict_r3_current_v5_policy_ledger import (
    POLICY_COLUMNS,
    _load_policy_contract,
    materialize_policy_contract,
)
from scripts.run_strict_r3_mc1_d2_controlled_ablation import (
    CORE,
    SEED,
    _causal_shifts,
    _day_balanced,
    _fit_hgb,
    _score_bands,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (
    CAUSAL_AUCTION_CURVE,
    _candidate_table,
    _metrics,
    _params,
)
from extreme_price_movements.portfolio_policy_replay import replay_candidates


SCORE_COLUMNS = (
    "candidate_id", "__decision_ts__", "__symbol__", "side_name", *CORE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _load_score_panel(path: Path, canonical_policy_path: Path, family: str) -> tuple[pd.DataFrame, dict[str, int]]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(SCORE_COLUMNS).difference(available))
    if missing:
        raise ValueError(f"{family} score panel misses MC1 input(s): {missing}")
    scores = pd.read_parquet(path, columns=list(SCORE_COLUMNS))
    scores["__decision_ts__"] = pd.to_datetime(scores["__decision_ts__"], utc=True, errors="raise")
    scores = scores.loc[
        scores.loc[:, list(CORE)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    ].copy().reset_index(drop=True)
    if scores.empty:
        raise ValueError(f"{family} has no target-free rows with the frozen MC1 inputs")
    if scores["candidate_id"].duplicated().any():
        raise ValueError(f"{family} score panel has duplicate candidate_id values")
    policy = _load_policy_contract(canonical_policy_path, candidate_ids=scores["candidate_id"])
    scores_before = scores.loc[:, list(SCORE_COLUMNS)].copy()
    joined, policy_audit = materialize_policy_contract(scores, policy)
    if not joined.loc[:, list(SCORE_COLUMNS)].equals(scores_before):
        raise AssertionError(f"{family} policy attachment changed target-free score fields")
    joined["score_band"] = _score_bands(joined)
    joined["day"] = joined["__decision_ts__"].dt.normalize()
    joined["family"] = family
    return joined, policy_audit


def _run_family(panel: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for fold_start in pd.date_range(start, end, freq="MS", tz="UTC"):
        fold_end = min(fold_start + pd.offsets.MonthBegin(1), end)
        if fold_start >= end:
            break
        fit = panel.loc[
            panel["policy_path_valid"].fillna(False).astype(bool)
            & panel["policy_net_bps"].notna()
            & panel["policy_label_available_ts"].lt(fold_start)
        ].copy()
        held = panel.loc[
            panel["__decision_ts__"].ge(fold_start)
            & panel["__decision_ts__"].lt(fold_end)
        ].copy()
        if len(fit) < 5_000 or held.empty:
            continue
        substrate = _day_balanced(fit)
        model, medians, curve, clip = _fit_hgb(substrate, CORE)
        matrix = held.loc[:, list(CORE)].apply(pd.to_numeric, errors="coerce").fillna(medians)
        held["static_expected_bps"] = model.predict(matrix)
        bucket = held["__decision_ts__"].dt.floor("1d")
        # ``_causal_shifts`` admits a label only when its availability bucket is
        # strictly earlier than the current bucket; passing the full scored
        # panel therefore remains prequential while allowing normal daily live
        # calibration recovery inside a held month.
        shifts = _causal_shifts(panel, curve, pd.DatetimeIndex(bucket.unique()), "1d")
        held["recent_shift_bps"] = bucket.map(shifts).fillna(0.0).to_numpy(float)
        held["mc1_expected_bps"] = held["static_expected_bps"] + held["recent_shift_bps"]
        held["fold_start"] = fold_start
        held["mc1_target_clip_low_bps"] = clip[0]
        held["mc1_target_clip_high_bps"] = clip[1]
        blocks.append(held)
    if not blocks:
        return panel.iloc[0:0].copy()
    return pd.concat(blocks, ignore_index=True)


def _raw_score_metrics(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    valid = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & frame["policy_net_bps"].notna()
    ].copy()
    rows: list[dict[str, object]] = []
    for period, piece in [("all", valid), *[(str(year), valid.loc[valid["__decision_ts__"].dt.year.eq(year)]) for year in (2025, 2026)]]:
        if piece.empty:
            continue
        ic = piece.groupby("__decision_ts__", sort=False).apply(
            lambda group: group["final_score"].corr(group["policy_net_bps"], method="spearman"),
            include_groups=False,
        ).dropna()
        rows.append({
            "family": family,
            "period": period,
            "rows": int(len(piece)),
            "score_spearman_ic": float(ic.mean()) if len(ic) else float("nan"),
        })
    return pd.DataFrame(rows)


def _policy_union(frames: list[pd.DataFrame]) -> pd.DataFrame:
    cols = ["candidate_id", "__decision_ts__", "final_score", *POLICY_COLUMNS]
    union = pd.concat([frame.loc[:, cols] for frame in frames], ignore_index=True)
    union = union.sort_values(["candidate_id", "__decision_ts__"], kind="stable").drop_duplicates("candidate_id", keep="first")
    if union["candidate_id"].duplicated().any():
        raise AssertionError("policy union must have unique candidate identities")
    return union


def _portfolio_replay(prediction: pd.DataFrame, policy: pd.DataFrame, *, family: str, out_dir: Path) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    for year in (2025, 2026):
        part = prediction.loc[prediction["__decision_ts__"].dt.year.eq(year)].copy()
        if part.empty:
            continue
        candidates = _candidate_table(part, policy, 50.0, invalid_outcome_mode="exclude")
        decisions, equity, _ = replay_candidates(
            candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
            market_mode="perps", initial_wallet=1000.0,
        )
        if decisions.empty:
            decisions = decisions.copy()
            decisions["policy_outcome_available"] = pd.Series(dtype=bool)
        else:
            provenance = candidates.loc[:, ["candidate_id", "policy_outcome_available"]].reset_index(drop=True)
            provenance.index.name = "candidate_index"
            decisions = decisions.merge(provenance, on="candidate_index", how="left", validate="many_to_one")
            if decisions["policy_outcome_available"].isna().any():
                raise AssertionError("portfolio decision lacks canonical outcome provenance")
        decisions.to_parquet(out_dir / f"{family}_{year}_decisions.parquet", index=False, compression="zstd")
        equity.to_parquet(out_dir / f"{family}_{year}_equity.parquet", index=False, compression="zstd")
        metric = _metrics(decisions, equity, family, str(year))
        metric["admission_threshold_bps"] = 50.0
        metrics.append(metric)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf-scores", required=True, type=Path)
    parser.add_argument("--current-scores", required=True, type=Path)
    parser.add_argument("--canonical-policy", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--start", default="2025-02-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--families", nargs="*", choices=("bcf", "current_v5"))
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    start, end = _utc(args.start), _utc(args.end)
    inputs = {"bcf": args.bcf_scores, "current_v5": args.current_scores}
    if args.families:
        inputs = {family: inputs[family] for family in args.families}
    audits: dict[str, dict[str, int]] = {}
    prediction_paths: dict[str, Path] = {}
    for family, path in inputs.items():
        panel, audit = _load_score_panel(path, args.canonical_policy, family)
        audits[family] = audit
        prediction = _run_family(panel, start=start, end=end)
        if prediction.empty:
            raise RuntimeError(f"{family} produced no prequential MC1 predictions")
        prediction_path = args.out_dir / f"predictions_{family}_mc1_d2.parquet"
        prediction.to_parquet(prediction_path, index=False, compression="zstd")
        prediction_paths[family] = prediction_path
        print(json.dumps({"event": "mc1_complete", "family": family, "rows": len(prediction)}), flush=True)
        del panel, prediction
        gc.collect()
    prediction_slices = [
        pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "final_score", *POLICY_COLUMNS])
        for path in prediction_paths.values()
    ]
    policy_union = _policy_union(prediction_slices)
    policy_union.to_parquet(args.out_dir / "canonical_policy_union.parquet", index=False, compression="zstd")
    del prediction_slices
    gc.collect()
    portfolio_metrics: list[dict[str, object]] = []
    raw_metrics: list[pd.DataFrame] = []
    for family, prediction_path in prediction_paths.items():
        prediction = pd.read_parquet(prediction_path)
        portfolio_metrics.extend(_portfolio_replay(prediction, policy_union, family=family, out_dir=args.out_dir))
        raw_metrics.append(_raw_score_metrics(prediction, family))
        del prediction
        gc.collect()
    pd.DataFrame(portfolio_metrics).to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False)
    pd.DataFrame(portfolio_metrics).to_csv(args.out_dir / "portfolio_metrics.csv", index=False)
    pd.concat(raw_metrics, ignore_index=True).to_parquet(args.out_dir / "raw_score_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_score_family_matched_mc1_canonical_policy_v1",
        "status": "complete",
        "purpose": "matched score-family comparison with one canonical policy substrate and invalid paths excluded before MC1 fitting and portfolio capacity",
        "inputs": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in inputs.items()},
        "canonical_policy": {"path": str(args.canonical_policy), "sha256": _sha256(args.canonical_policy)},
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "mc1_contract": {
            "features": list(CORE),
            "model": "HistGradientBoostingRegressor depth=2 iter=80 lr=.04 l2=20 min_leaf=100 seed=1729",
            "target": "canonical policy_net_bps, 2nd/98th percentile clipped within each prequential fit",
            "recent_shift": "21-day 10%-trimmed score-band residual; labels available strictly before daily bucket",
            "admission": "mc1_expected_bps >= +50",
        },
        "portfolio": "canonical controlled MC1 replay; long-only, 7x, 10% margin slots, 2 new entries, 8 concurrent, 80% wallet cap; invalid outcomes excluded before capacity allocation",
        "policy_attachment_audit": audits,
        "policy_union_rows": int(len(policy_union)),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
