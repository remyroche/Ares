#!/usr/bin/env python3
"""Build a strictly-prior BCF MC1 calibration ledger from recent replay trades.

The score half is assembled solely from target-free BCF score receipts made by
one frozen monthly bundle.  Parent-policy outcomes are joined only after that
score population is frozen.  This gives BCF MC1 its intended rolling 21-day
residual adjustment without borrowing current-v5 scores or using a neutral
cold-start prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "candidate_id", "__decision_ts__", "side_name",
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
)
FEATURES = REQUIRED[3:]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _manifest(path: Path) -> dict[str, object]:
    try:
        return dict(json.loads((path.parent / "run_manifest.json").read_text()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid BCF receipt manifest for {path}") from exc


def _read_scores(path: Path, *, expected_bundle_sha: str) -> pd.DataFrame:
    path = path.resolve()
    manifest = _manifest(path)
    if manifest.get("schema") != "strict_r3_bcf_live_score_v1":
        raise ValueError(f"unexpected BCF score schema: {path}")
    if str(manifest.get("bundle_sha256")) != expected_bundle_sha:
        raise ValueError(f"BCF bundle hash mismatch: {path}")
    if manifest.get("outcome_columns_consumed") != []:
        raise ValueError(f"BCF score receipt consumed outcomes: {path}")
    frame = pd.read_parquet(path, columns=list(REQUIRED))
    missing = sorted(set(REQUIRED).difference(frame.columns))
    if missing:
        raise ValueError(f"BCF score receipt lacks native mapper fields: {path}: {missing}")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"BCF score receipt duplicates candidate IDs: {path}")
    if not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError(f"BCF score receipt is not long-only: {path}")
    frame["__score_source_path__"] = str(path.relative_to(ROOT))
    frame["__score_source_sha256__"] = _sha(path)
    return frame


def _score_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    merged = left.merge(right, on="candidate_id", how="outer", indicator=True, suffixes=("__l", "__r"))
    if not merged["_merge"].eq("both").all():
        return False
    for field in FEATURES:
        a = pd.to_numeric(merged[f"{field}__l"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(merged[f"{field}__r"], errors="coerce").to_numpy(float)
        if not np.isclose(a, b, rtol=0.0, atol=1e-12, equal_nan=True).all():
            return False
    return True


def _priority(path: Path) -> tuple[int, str]:
    value = str(path)
    if "_live_" in value:
        return (0, value)
    if "stateful_recovery" in value:
        return (1, value)
    if "terminal" in value:
        return (2, value)
    return (3, value)


def _discover_receipts(
    *, root: Path, start: pd.Timestamp, end: pd.Timestamp, expected_bundle_sha: str,
) -> tuple[dict[pd.Timestamp, list[Path]], list[dict[str, object]]]:
    found: dict[pd.Timestamp, list[Path]] = defaultdict(list)
    rejected: list[dict[str, object]] = []
    for path in root.glob("**/bcf_score/predictions.parquet"):
        if "short_" in str(path):
            continue
        try:
            manifest = _manifest(path)
            if str(manifest.get("bundle_sha256")) != expected_bundle_sha:
                continue
            if manifest.get("schema") != "strict_r3_bcf_live_score_v1":
                continue
            held = pd.read_parquet(path, columns=["__decision_ts__"])
            stamps = pd.to_datetime(held["__decision_ts__"], utc=True).unique()
            if len(stamps) != 1:
                rejected.append({"path": str(path.relative_to(ROOT)), "reason": "not_single_held_hour"})
                continue
            stamp = pd.Timestamp(stamps[0])
            if start <= stamp < end:
                found[stamp].append(path)
        except Exception as exc:
            rejected.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    return found, rejected


def _select_hour(paths: list[Path], *, expected_bundle_sha: str) -> tuple[pd.DataFrame | None, dict[str, object]]:
    ordered = sorted(paths, key=_priority)
    selected = _read_scores(ordered[0], expected_bundle_sha=expected_bundle_sha)
    conflicts: list[str] = []
    for path in ordered[1:]:
        peer = _read_scores(path, expected_bundle_sha=expected_bundle_sha)
        if not _score_equal(selected, peer):
            conflicts.append(str(path.relative_to(ROOT)))
    if conflicts:
        return None, {
            "status": "conflict_fail_closed",
            "selected": str(ordered[0].relative_to(ROOT)),
            "conflicting": conflicts,
        }
    return selected, {
        "status": "selected",
        "selected": str(ordered[0].relative_to(ROOT)),
        "equivalent_duplicates": [str(path.relative_to(ROOT)) for path in ordered[1:]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-scores", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, default=ROOT / "data_perp/artifacts")
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--expected-bcf-bundle-sha", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--as-of", required=True, help="Strict label-availability cutoff, exclusive.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    start, as_of = _utc(args.start), _utc(args.as_of)
    if as_of <= start:
        raise ValueError("--as-of must be after --start")
    batch = _read_scores(args.batch_scores, expected_bundle_sha=args.expected_bcf_bundle_sha)
    batch = batch.loc[batch["__decision_ts__"].ge(start) & batch["__decision_ts__"].lt(as_of)].copy()
    batch_hours = set(batch["__decision_ts__"].unique())
    receipt_index, rejected_receipts = _discover_receipts(
        root=args.receipt_root.resolve(), start=start, end=as_of,
        expected_bundle_sha=args.expected_bcf_bundle_sha,
    )
    pieces = [batch]
    hour_audit: list[dict[str, object]] = []
    for stamp, paths in sorted(receipt_index.items()):
        if stamp in batch_hours:
            continue
        selected, audit = _select_hour(paths, expected_bundle_sha=args.expected_bcf_bundle_sha)
        audit["decision_ts"] = stamp.isoformat()
        hour_audit.append(audit)
        if selected is not None:
            pieces.append(selected)
    scores = pd.concat(pieces, ignore_index=True)
    if scores["candidate_id"].duplicated().any():
        raise AssertionError("BCF replay assembly duplicated candidate identities")
    labels = pd.read_parquet(args.policy_labels, columns=[
        "candidate_id", "__symbol__", "side_name", "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
    ])
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True)
    if labels["candidate_id"].duplicated().any():
        raise ValueError("policy-label source duplicates candidate IDs")
    if not labels["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("policy-label source is not long-only")
    replay = scores.merge(
        labels.drop(columns="side_name"), on="candidate_id", how="inner", validate="one_to_one",
    )
    replay = replay.loc[
        replay["policy_label_available_ts"].lt(as_of)
        & replay["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(replay["policy_net_bps"], errors="coerce").notna()
    ].copy()
    replay = replay.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if replay.empty:
        raise ValueError("same-bundle BCF replay has no strictly prior resolved rows")
    all_hours = pd.date_range(start.floor("h"), as_of.floor("h"), freq="h", inclusive="left")
    score_hours = set(pd.to_datetime(scores["__decision_ts__"], utc=True).unique())
    result_hours = set(pd.to_datetime(replay["__decision_ts__"], utc=True).unique())
    args.out_dir.mkdir(parents=True)
    ledger = args.out_dir / "bcf_mc1_recent_replay_ledger.parquet"
    replay.to_parquet(ledger, index=False, compression="zstd")
    coverage = pd.DataFrame({"__decision_ts__": all_hours})
    coverage["bcf_score_available"] = coverage["__decision_ts__"].isin(score_hours)
    coverage["resolved_replay_rows"] = coverage["__decision_ts__"].map(replay.groupby("__decision_ts__").size()).fillna(0).astype(int)
    coverage.to_parquet(args.out_dir / "hourly_coverage.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_bcf_same_bundle_recent_replay_ledger_v1",
        "target_free_score_assembly": True,
        "outcome_join_after_scoring": True,
        "strict_label_available_before": as_of.isoformat(),
        "window_start": start.isoformat(),
        "window_days": float((as_of - start) / pd.Timedelta(days=1)),
        "expected_bcf_bundle_sha256": args.expected_bcf_bundle_sha,
        "rows": int(len(replay)),
        "resolved_hours": int(len(result_hours)),
        "score_hours": int(len(score_hours)),
        "requested_hours": int(len(all_hours)),
        "missing_score_hours": [stamp.isoformat() for stamp in all_hours if stamp not in score_hours],
        "batch_scores": {"path": str(args.batch_scores), "sha256": _sha(args.batch_scores)},
        "policy_labels": {"path": str(args.policy_labels), "sha256": _sha(args.policy_labels)},
        "ledger_sha256": _sha(ledger),
        "receipt_hour_audit": hour_audit,
        "rejected_receipts": rejected_receipts,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **{k: manifest[k] for k in ("rows", "resolved_hours", "score_hours", "requested_hours", "missing_score_hours")}}))


if __name__ == "__main__":
    main()
