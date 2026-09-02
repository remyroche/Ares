#!/usr/bin/env python3
"""Materialise one causal four-phase strict-R3 feature stream, resumably.

This research-only runner creates *new* artifacts.  It never uses labels,
future path completeness, pre-scored candidates, live state, or exchange I/O.
It consumes a complete point-in-time candidate universe in chronological
72-hour blocks, preserving a private causal source/feature state between
blocks.  Warm-up outputs are intentionally discarded after their state has
committed; held-period outputs retain every universe row for downstream
feature, score, admission, and portfolio audits.

The state is phase-specific.  A 15-minute shifted H1 boundary must never
reuse a :00 source or feature state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_STATE = ROOT / "scripts/update_strict_r3_feature_panel_state.py"
FEATURES = ROOT / "scripts/materialize_strict_r3_forward_features_incremental_v13.py"
SCHEMA = "strict_r3_phase_h1_feature_replay_v1"
UNIVERSE_ROWS = 170


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_block(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        filters=[
            ("__ts__", ">=", start.to_pydatetime()),
            ("__ts__", "<", end.to_pydatetime()),
        ],
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame = frame.sort_values(["__ts__", "__symbol__"], kind="stable").reset_index(drop=True)
    if frame.empty:
        return frame
    counts = frame.groupby("__ts__", sort=False)["__symbol__"].nunique()
    if not counts.eq(UNIVERSE_ROWS).all():
        bad = counts.loc[~counts.eq(UNIVERSE_ROWS)].head(5).to_dict()
        raise ValueError(f"candidate block is not the complete frozen universe: {bad}")
    if frame.duplicated(["__ts__", "__symbol__"]).any():
        raise ValueError("candidate block has duplicate timestamp/symbol identities")
    return frame


def _run(command: list[str]) -> None:
    print(json.dumps({"event": "subprocess_start", "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _load_checkpoint(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    return payload


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _parse_ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True, choices=(0, 15, 30, 45))
    parser.add_argument("--warmup-candidates", type=Path, required=True)
    parser.add_argument("--replay-candidates", type=Path, required=True)
    parser.add_argument("--start", required=True, help="Inclusive causal warm-up timestamp")
    parser.add_argument("--warmup-end", required=True, help="First retained replay timestamp")
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-hours", type=int, default=72)
    parser.add_argument(
        "--retain-warmup-from",
        help=(
            "Inclusive timestamp from which warm-up feature shards are retained "
            "as phase-native scorer-reference inputs (typically 42 days before "
            "the first replay decision)."
        ),
    )
    parser.add_argument(
        "--max-blocks", type=int,
        help="Optional bounded smoke/resume limit; never changes phase semantics.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.chunk_hours < 72:
        raise ValueError("chunk-hours must preserve at least the 72-hour causal tail")
    start = _parse_ts(args.start)
    warmup_end = _parse_ts(args.warmup_end)
    end = _parse_ts(args.end_exclusive)
    if not start.minute == args.phase or not warmup_end.minute == args.phase or not end.minute == args.phase:
        raise ValueError("all boundaries must match the declared shifted-H1 phase")
    if not start < warmup_end < end:
        raise ValueError("require start < warmup-end < end-exclusive")
    retain_warmup_from = (
        _parse_ts(args.retain_warmup_from)
        if args.retain_warmup_from else None
    )
    if retain_warmup_from is not None:
        if retain_warmup_from.minute != args.phase:
            raise ValueError("retain-warmup-from must match the declared phase")
        if not start <= retain_warmup_from < warmup_end:
            raise ValueError("retain-warmup-from must lie inside causal warm-up")

    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is None:
        if args.resume:
            raise FileNotFoundError("--resume requires a phase checkpoint")
        manifest = {
            "schema": SCHEMA,
            "phase_minutes": args.phase,
            "cadence": "one completed H1 observation per shifted boundary",
            "start": start.isoformat(),
            "warmup_end": warmup_end.isoformat(),
            "end_exclusive": end.isoformat(),
            "chunk_hours": args.chunk_hours,
            "retain_warmup_from": (
                retain_warmup_from.isoformat() if retain_warmup_from else None
            ),
            "warmup_candidates": str(args.warmup_candidates),
            "warmup_candidates_sha256": _sha(args.warmup_candidates),
            "replay_candidates": str(args.replay_candidates),
            "replay_candidates_sha256": _sha(args.replay_candidates),
            "outcome_columns_consumed": [],
            "live_state_consumed": [],
            "completed_blocks": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_checkpoint(checkpoint_path, manifest)
        cursor = start
        previous_state: Path | None = None
    else:
        if not args.resume:
            raise FileExistsError("phase output already has a checkpoint; use --resume")
        if int(checkpoint.get("phase_minutes")) != args.phase:
            raise ValueError("checkpoint phase mismatch")
        existing_retain = checkpoint.get("retain_warmup_from")
        desired_retain = retain_warmup_from.isoformat() if retain_warmup_from else None
        if existing_retain not in (None, desired_retain):
            raise ValueError("checkpoint warm-up retention contract mismatch")
        if existing_retain is None and desired_retain is not None:
            checkpoint["retain_warmup_from"] = desired_retain
            _write_checkpoint(checkpoint_path, checkpoint)
        # The initial manifest is committed before its first source/feature
        # block.  A crash in that block is therefore resumable from the
        # original causal start, rather than leaving an immutable but unusable
        # root with no ``next_timestamp``.  No output is trusted until the
        # block transaction publishes this cursor and its state path.
        next_timestamp = checkpoint.get("next_timestamp")
        if next_timestamp is None:
            if checkpoint.get("completed_blocks"):
                raise ValueError("completed phase checkpoint is missing next_timestamp")
            cursor = start
            previous_state = None
        else:
            cursor = pd.Timestamp(next_timestamp)
            cursor = (
                cursor.tz_localize("UTC")
                if cursor.tzinfo is None else cursor.tz_convert("UTC")
            )
            saved = checkpoint.get("current_source_state")
            previous_state = Path(str(saved)) if saved else None
            if previous_state is not None and not previous_state.is_file():
                raise FileNotFoundError("checkpoint source state is missing")

    scratch = root / "scratch"
    scratch.mkdir(exist_ok=True)
    source_states = root / "source_states"
    source_states.mkdir(exist_ok=True)
    replay_outputs = root / "replay_feature_shards"
    replay_outputs.mkdir(exist_ok=True)
    feature_cache = root / "feature_state"

    block_index = len(list(checkpoint.get("completed_blocks", []))) if checkpoint else 0
    blocks_this_invocation = 0
    while cursor < end:
        if args.max_blocks is not None and blocks_this_invocation >= args.max_blocks:
            break
        block_end = min(cursor + pd.Timedelta(hours=args.chunk_hours), end)
        # Candidate grids are deliberately separate at the hold-out boundary.
        # Never let a warm-up block request rows from the replay grid (or vice
        # versa): both inputs are target-free, but their artifact identity is
        # part of the causal scoring lineage.
        if cursor < warmup_end < block_end:
            block_end = warmup_end
        source_path = args.warmup_candidates if cursor < warmup_end else args.replay_candidates
        block = _read_block(source_path, cursor, block_end)
        if block.empty:
            raise ValueError(f"no candidate rows for requested block {cursor} -> {block_end}")
        if block["__ts__"].min() != cursor or block["__ts__"].max() != block_end - pd.Timedelta(hours=1):
            raise ValueError("candidate block has a causal timestamp gap")
        candidate_path = scratch / f"candidates_{block_index:04d}.parquet"
        # An interruption before checkpoint publication can leave only these
        # phase-private primitive scratch files.  They are never model inputs
        # and are safe to replace on explicit resume; the committed feature
        # cache and checkpoint remain authoritative.
        if candidate_path.exists():
            candidate_path.unlink()
        block.to_parquet(candidate_path, index=False)
        state_dir = source_states / f"block_{block_index:04d}"
        if state_dir.exists():
            if not args.resume:
                raise FileExistsError(f"immutable source state already exists: {state_dir}")
            shutil.rmtree(state_dir)
        source_command = [
            sys.executable, str(SOURCE_STATE),
            "--candidates", str(candidate_path),
            "--history-start", start.isoformat(),
            "--end-exclusive", block_end.isoformat(),
            "--bar-phase-minutes", str(args.phase),
            "--out-dir", str(state_dir),
        ]
        if previous_state is not None:
            source_command.extend([
                "--state-in", str(previous_state), "--preserve-sealed-overlap",
            ])
        _run(source_command)
        panel_state = state_dir / "feature_panel_state.joblib"
        if not panel_state.is_file():
            raise FileNotFoundError("source-state materialisation did not emit its panel state")

        is_warmup = cursor < warmup_end
        retain_warmup = bool(
            is_warmup
            and retain_warmup_from is not None
            and block_end > retain_warmup_from
        )
        feature_dir = (
            scratch / f"warmup_feature_{block_index:04d}"
            if is_warmup and not retain_warmup
            else (
                root / "reference_feature_shards" / f"block_{block_index:04d}"
                if is_warmup else replay_outputs / f"block_{block_index:04d}"
            )
        )
        if feature_dir.exists():
            if not args.resume:
                raise FileExistsError(f"immutable feature output already exists: {feature_dir}")
            shutil.rmtree(feature_dir)
        feature_command = [
            sys.executable, str(FEATURES),
            "--candidates", str(candidate_path),
            "--panel-state", str(panel_state),
            "--cache-dir", str(feature_cache),
            "--side", "long",
            "--out-dir", str(feature_dir),
        ]
        if previous_state is None:
            feature_command.append("--bootstrap-state")
        else:
            feature_command.extend([
                "--stateful-tail-hours", str(args.chunk_hours),
                # The append graph stays incremental, while only the frozen
                # long-memory closure is read from the full causal source
                # state.  This preserves training/inference feature semantics
                # for OI, liquidation and structural inputs without allowing
                # any future rows into the current block.
                "--hybrid-exact-long-memory",
            ])
        if not is_warmup or retain_warmup:
            feature_command.append("--emit-all-candidate-timestamps")
        _run(feature_command)
        feature_manifest = feature_dir / "feature_manifest.json"
        feature_matrix = feature_dir / "canonical120_features.parquet"
        if not feature_manifest.is_file() or not feature_matrix.is_file():
            raise FileNotFoundError("feature materialisation did not complete atomically")
        receipt = json.loads(feature_manifest.read_text())
        if list(receipt.get("outcome_columns_consumed") or []):
            raise AssertionError("target-free feature run consumed outcome columns")
        if not is_warmup or retain_warmup:
            rows = pd.read_parquet(feature_matrix, columns=["__ts__", "__symbol__"])
            if len(rows) != len(block) or rows.duplicated(["__ts__", "__symbol__"]).any():
                raise AssertionError("replay feature output lost complete-universe identities")

        if is_warmup and not retain_warmup:
            # State is committed before this cleanup.  Warm-up feature values
            # are not score inputs, whereas retaining every duplicate shard
            # would only consume space.  Keep the immutable checkpoint receipt.
            warmup_receipt = root / "warmup_receipts"
            warmup_receipt.mkdir(exist_ok=True)
            shutil.copy2(feature_manifest, warmup_receipt / f"block_{block_index:04d}.json")
            shutil.rmtree(feature_dir)
        candidate_path.unlink()
        retired_state = previous_state
        previous_state = panel_state
        checkpoint = _load_checkpoint(checkpoint_path) or {}
        completed = list(checkpoint.get("completed_blocks", []))
        completed.append({
            "index": block_index,
            "start": cursor.isoformat(),
            "end_exclusive": block_end.isoformat(),
            "kind": "warmup" if is_warmup else "replay",
            "retained_reference_shard": bool(retain_warmup),
            "candidate_rows": int(len(block)),
            "feature_rows": int(receipt["output_rows"]),
            "feature_runtime_seconds": receipt.get("runtime_seconds_before_state_commit"),
            "source_state": str(panel_state),
        })
        checkpoint["completed_blocks"] = completed
        checkpoint["next_timestamp"] = block_end.isoformat()
        checkpoint["current_source_state"] = str(panel_state)
        checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_checkpoint(checkpoint_path, checkpoint)
        # Only remove the prior *scratch* source state after the checkpoint
        # durably points to the successor.  A process failure can therefore
        # always resume from a self-contained current state; it never leaves a
        # checkpoint whose state path has already been removed.
        if retired_state is not None and retired_state.parent.exists():
            shutil.rmtree(retired_state.parent)
        print(json.dumps({"event": "block_complete", **completed[-1]}), flush=True)
        cursor = block_end
        block_index += 1
        blocks_this_invocation += 1


if __name__ == "__main__":
    main()
