#!/usr/bin/env python3
"""Create hash-bound C0/C1 mapper inputs from strict successor OOF scores.

The materialiser first persists the complete target-free Router50/Base/Under
score population.  It only then left-joins exact rich-policy labels into a
separate replay view.  This prevents outcome validity from changing either a
candidate identity or the mapper's inference feature geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")
H12_RESOLUTION = pd.Timedelta(hours=12, minutes=5)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _identity_hash(frame: pd.DataFrame) -> str:
    work = frame.loc[:, list(IDENTITY)].copy().sort_values(list(IDENTITY), kind="stable")
    digest = hashlib.sha256()
    for row in work.itertuples(index=False, name=None):
        digest.update("|".join(map(str, row)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _read_labels(paths: list[Path]) -> pd.DataFrame:
    columns = [
        "candidate_id", "decision_timestamp", "entry_timestamp", "entry_price",
        "exit_timestamp", "exit_price", "gross_bps", "net_bps", "exit_reason",
        "exit_minute", "outcome_available", "outcome_invalid_reason", "outcome_source",
    ]
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=columns).copy()
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"label ledger duplicates candidate IDs: {path}")
        parts.append(frame)
    labels = pd.concat(parts, ignore_index=True)
    if labels["candidate_id"].duplicated().any():
        raise ValueError("label ledgers overlap candidate identities")
    labels["decision_timestamp"] = pd.to_datetime(labels["decision_timestamp"], utc=True, errors="raise")
    labels["policy_label_available_ts"] = labels["decision_timestamp"] + H12_RESOLUTION
    labels = labels.rename(columns={
        "outcome_available": "policy_path_valid",
        "entry_timestamp": "policy_entry_timestamp",
        "exit_timestamp": "policy_exit_timestamp",
        "gross_bps": "policy_gross_bps",
        "net_bps": "policy_net_bps",
        "exit_minute": "policy_exit_bar_15m",
        "entry_price": "policy_entry_price",
        "exit_price": "policy_exit_price",
        "exit_reason": "policy_exit_reason",
        "outcome_source": "policy_outcome_source",
        "outcome_invalid_reason": "policy_invalid_reason",
    })
    labels["policy_exit_bar_15m"] = pd.to_numeric(labels["policy_exit_bar_15m"], errors="coerce") / 15.0
    labels["policy_cost_bps"] = 100.0
    keep = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_entry_timestamp", "policy_exit_timestamp",
        "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_label_available_ts", "policy_outcome_source",
        "policy_invalid_reason", "policy_cost_bps",
    ]
    return labels.loc[:, keep]


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    temporary = output.with_name(f".{output.name}.build-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        upstream_path = args.upstream.resolve()
        scores = pd.read_parquet(upstream_path).copy()
        forward_paths = [path.resolve() for path in args.forward_upstream]
        forward_rows = 0
        for path in forward_paths:
            forward = pd.read_parquet(path).copy()
            forbidden = {
                column for column in forward.columns
                if column.startswith("policy_") or "outcome" in column.lower() or column.startswith("label_")
            }
            if forbidden:
                raise AssertionError(f"forward upstream score panel is not target-free: {sorted(forbidden)}")
            forward_rows += int(len(forward))
            scores = pd.concat((scores, forward), ignore_index=True, sort=False)
        required = {
            "candidate_id", "__decision_ts__", "side_name", "router50_eligible",
            "router_primary_rank", "base_score", "base_rank_ts", "under_rank_ts",
        }
        missing = sorted(required.difference(scores.columns))
        if missing:
            raise KeyError(f"strict upstream OOF ledger misses {missing}")
        scores["candidate_id"] = scores["candidate_id"].astype(str)
        scores["__decision_ts__"] = pd.to_datetime(scores["__decision_ts__"], utc=True, errors="raise")
        # Upstream OOF score ledgers are symbol-free by design; the
        # candidate-time identity carries the frozen symbol.  Derive it before
        # the full identity audit rather than making a historical producer's
        # incidental helper column a hidden input requirement.
        scores["__symbol__"] = scores["candidate_id"].str.split("|", n=1).str[0].astype(str)
        if scores.duplicated(list(IDENTITY)).any():
            raise AssertionError("historical and forward upstream score panels overlap candidate identities")
        # Router50 is fixed before any outcome label is read.  Base's OOF
        # availability is an upstream support fact, likewise independent of
        # the held policy outcome.
        target_free = scores.loc[
            scores["router50_eligible"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(scores["base_rank_ts"], errors="coerce"))
        ].copy()
        target_free["base_rank_ts"] = pd.to_numeric(target_free["base_rank_ts"], errors="raise")
        target_free["router_primary_rank"] = pd.to_numeric(target_free["router_primary_rank"], errors="raise")
        target_free["under_available"] = np.isfinite(pd.to_numeric(target_free["under_rank_ts"], errors="coerce")).astype(np.int8)
        target_free["under_rank_ts"] = pd.to_numeric(target_free["under_rank_ts"], errors="coerce").fillna(0.5)
        target_free["base_minus_router_rank"] = (
            target_free["base_rank_ts"] - target_free["router_primary_rank"]
        )
        target_free = target_free.loc[:, [
            *IDENTITY, "fold_month", "router_primary_rank", "base_score", "base_rank_ts",
            "under_rank_ts", "under_available", "base_minus_router_rank",
        ]].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if target_free.duplicated(list(IDENTITY)).any():
            raise AssertionError("target-free mapper source has duplicate identities")
        target_free.to_parquet(temporary / "target_free_upstream_scores.parquet", index=False, compression="zstd")

        # Label join happens after target-free evidence is written and hashed.
        labels = _read_labels([path.resolve() for path in args.labels])
        replay = target_free.merge(labels, on="candidate_id", how="left", validate="one_to_one")
        if len(replay) != len(target_free):
            raise AssertionError("policy-label join changed target-free membership")
        replay.to_parquet(temporary / "policy_attached_replay_panel.parquet", index=False, compression="zstd")
        _write_json(temporary / "run_manifest.json", {
            "schema": "p8u_successor_mc1_source_panel_v2_exact_timestamps",
            "status": "complete",
            "scope": "offline target-free successor mapper source; labels joined only after target-free score persistence",
            "upstream_oof": {"path": str(upstream_path), "sha256": _sha(upstream_path)},
            "forward_upstream": [
                {"path": str(path), "sha256": _sha(path)} for path in forward_paths
            ],
            "forward_upstream_rows": int(forward_rows),
            "labels": [{"path": str(path.resolve()), "sha256": _sha(path.resolve())} for path in args.labels],
            "target_free_rows": int(len(target_free)),
            "target_free_identity_sha256": _identity_hash(target_free),
            "policy_valid_rows": int(replay["policy_path_valid"].fillna(False).astype(bool).sum()),
            "router50_only": True,
            "base_score_only": True,
            "under_role": "telemetry-only mapper feature; zero upstream direct rank authority",
            "feature_contract": [
                "base_rank_ts", "router_primary_rank", "base_minus_router_rank", "under_rank_ts", "under_available",
            ],
            "policy": {
                "entry": "observed one-minute open at decision plus five minutes",
                "path": "720 completed observed one-minute bars",
                "cost": "100 bps embedded exactly once",
                "label_available": "decision timestamp plus 12h05m",
                "timing_fields": ["policy_entry_timestamp", "policy_exit_timestamp"],
            },
            "outputs": {
                "target_free_upstream_scores": {
                    "path": "target_free_upstream_scores.parquet",
                    "sha256": _sha(temporary / "target_free_upstream_scores.parquet"),
                },
                "policy_attached_replay_panel": {
                    "path": "policy_attached_replay_panel.parquet",
                    "sha256": _sha(temporary / "policy_attached_replay_panel.parquet"),
                },
            },
        })
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--forward-upstream", type=Path, action="append", default=[],
        help="Optional target-free forward score panel(s); identities may not overlap --upstream.",
    )
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
