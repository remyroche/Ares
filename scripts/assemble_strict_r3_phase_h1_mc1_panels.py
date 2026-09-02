#!/usr/bin/env python3
"""Build target-free, strictly prequential MC1 score panels by phase.

History supplies only earlier score coordinates.  Newly generated quarter-hour
scores replace the requested May--July interval.  Policy outcomes are excluded
here and joined only by the downstream prequential MC1 replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_bcf_mc1_mapper import derive_bcf_mc1_features
from scripts.run_strict_r3_mc1_d2_controlled_ablation import CORE


IDENTITY = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")
SCORE_COLUMNS = (*IDENTITY, *CORE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def read_scores(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"duplicate candidate IDs in {path}")
    return frame


def history(path: Path, start: pd.Timestamp) -> pd.DataFrame:
    frame = read_scores(path, list(SCORE_COLUMNS))
    frame = frame.loc[frame["__decision_ts__"].lt(start)].copy()
    # Match the frozen MC1 replay contract: a score row missing any of its
    # target-free coordinates is unavailable to the mapper, never imputed or
    # represented as an economic failure.
    frame = frame.loc[frame.loc[:, list(CORE)].notna().all(axis=1)].copy()
    return frame.loc[:, list(SCORE_COLUMNS)]


def current_replacement(root: Path, phase: int) -> pd.DataFrame:
    paths = sorted(root.glob(f"current/block=*/phase={phase:02d}/predictions.parquet"))
    if len(paths) != 4:
        raise ValueError(f"expected four current score blocks, found {len(paths)}")
    frame = pd.concat([read_scores(path, list(SCORE_COLUMNS)) for path in paths], ignore_index=True)
    # Current-v5 only exposes a final stack score for its timestamp-local
    # top-30 base route.  Unrouted rows are not negative examples.
    frame = frame.loc[frame.loc[:, list(CORE)].notna().all(axis=1)].copy()
    if frame["candidate_id"].duplicated().any():
        raise ValueError("current replacement blocks overlap")
    return frame.loc[:, list(SCORE_COLUMNS)]


def current_native_history(root: Path, phase: int, start: pd.Timestamp) -> pd.DataFrame:
    """Read earlier, same-phase current-v5 raw scores for MC1 fitting.

    A shifted-phase mapper must not borrow its historical score geometry from
    phase :00.  The raw-score producer emits one immutable directory per
    frozen upstream block; this helper accepts the historical subset and
    retains only decisions strictly before the held interval.
    """
    paths = sorted(root.glob(f"current/block=*/phase={phase:02d}/predictions.parquet"))
    if not paths:
        raise ValueError(f"phase={phase} has no phase-native current history")
    frame = pd.concat([read_scores(path, list(SCORE_COLUMNS)) for path in paths], ignore_index=True)
    frame = frame.loc[
        frame["__decision_ts__"].lt(start)
        & frame.loc[:, list(CORE)].notna().all(axis=1)
    ].copy()
    if frame["candidate_id"].duplicated().any():
        raise ValueError("phase-native current history overlaps across frozen blocks")
    return frame.loc[:, list(SCORE_COLUMNS)]


def current_identity(root: Path, phase: int) -> pd.DataFrame:
    paths = sorted(root.glob(f"current/block=*/phase={phase:02d}/predictions.parquet"))
    if len(paths) != 4:
        raise ValueError(f"expected four current identity blocks, found {len(paths)}")
    frame = pd.concat([read_scores(path, list(IDENTITY)) for path in paths], ignore_index=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("current identity blocks overlap")
    return frame


def current_identity_all(root: Path, phase: int) -> pd.DataFrame:
    """Return all available same-phase current identities, including history."""
    paths = sorted(root.glob(f"current/block=*/phase={phase:02d}/predictions.parquet"))
    if not paths:
        raise ValueError(f"phase={phase} has no phase-native current identities")
    frame = pd.concat([read_scores(path, list(IDENTITY)) for path in paths], ignore_index=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("phase-native current identity blocks overlap")
    return frame


def bcf_replacement(root: Path, current_root: Path, phase: int) -> pd.DataFrame:
    paths = sorted(root.glob(f"bcf/month=*/phase={phase:02d}/score/predictions.parquet"))
    if len(paths) != 3:
        raise ValueError(f"expected three BCF score months, found {len(paths)}")
    chunks: list[pd.DataFrame] = []
    for path in paths:
        raw = pd.read_parquet(path).copy()
        raw["candidate_id"] = raw["candidate_id"].astype(str)
        raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
        if raw["candidate_id"].duplicated().any():
            raise ValueError(f"duplicate BCF IDs in {path}")
        native = derive_bcf_mc1_features(raw)
        identity = current_identity(current_root, phase)
        chunks.append(
            raw.loc[:, ["candidate_id", "__decision_ts__", "side_name"]]
            .merge(identity.loc[:, ["candidate_id", "__symbol__"]], on="candidate_id", how="inner", validate="one_to_one")
            .merge(native, on="candidate_id", how="inner", validate="one_to_one")
            .loc[:, list(SCORE_COLUMNS)]
        )
    frame = pd.concat(chunks, ignore_index=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("BCF replacement blocks overlap")
    return frame


def bcf_native_history(root: Path, current_root: Path, phase: int, start: pd.Timestamp) -> pd.DataFrame:
    """Build target-free BCF MC1 coordinates from earlier same-phase scores."""
    paths = sorted(root.glob(f"bcf/month=*/phase={phase:02d}/score/predictions.parquet"))
    if not paths:
        raise ValueError(f"phase={phase} has no phase-native BCF history")
    identities = current_identity_all(current_root, phase).loc[:, ["candidate_id", "__symbol__"]]
    chunks: list[pd.DataFrame] = []
    for path in paths:
        raw = pd.read_parquet(path).copy()
        raw["candidate_id"] = raw["candidate_id"].astype(str)
        raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
        if raw["candidate_id"].duplicated().any():
            raise ValueError(f"duplicate BCF IDs in {path}")
        native = derive_bcf_mc1_features(raw)
        chunk = (
            raw.loc[:, ["candidate_id", "__decision_ts__", "side_name"]]
            .merge(identities, on="candidate_id", how="inner", validate="one_to_one")
            .merge(native, on="candidate_id", how="inner", validate="one_to_one")
            .loc[:, list(SCORE_COLUMNS)]
        )
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)
    frame = frame.loc[
        frame["__decision_ts__"].lt(start)
        & frame.loc[:, list(CORE)].notna().all(axis=1)
    ].copy()
    if frame["candidate_id"].duplicated().any():
        raise ValueError("phase-native BCF history overlaps across frozen months")
    return frame.loc[:, list(SCORE_COLUMNS)]


def combine(history_rows: pd.DataFrame, replacement: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if replacement["__decision_ts__"].lt(start).any() or replacement["__decision_ts__"].ge(end).any():
        raise ValueError("replacement scores fall outside the declared held interval")
    output = pd.concat([history_rows, replacement], ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise ValueError("history and replacement candidate IDs overlap")
    if output.loc[:, list(CORE)].isna().any().any():
        raise ValueError("target-free MC1 panel contains incomplete score coordinates")
    return output.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def audit(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, object]:
    held = frame.loc[frame["__decision_ts__"].between(start, end, inclusive="left")]
    forbidden = [
        name for name in frame.columns
        if name.startswith("policy_") or name in {"gross_bps", "net_bps", "outcome", "label"}
    ]
    return {
        "rows": int(len(frame)),
        "history_rows": int((frame["__decision_ts__"] < start).sum()),
        "replacement_rows": int(len(held)),
        "replacement_min_ts": held["__decision_ts__"].min().isoformat() if len(held) else None,
        "replacement_max_ts": held["__decision_ts__"].max().isoformat() if len(held) else None,
        "duplicate_candidate_ids": int(frame["candidate_id"].duplicated().sum()),
        "forbidden_outcome_columns": forbidden,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-history", type=Path)
    parser.add_argument("--bcf-history", type=Path)
    parser.add_argument(
        "--current-native-history-root", type=Path,
        help="Raw same-phase current-v5 score root; supersedes --current-history.",
    )
    parser.add_argument(
        "--bcf-native-history-root", type=Path,
        help="Raw same-phase BCF score root; supersedes --bcf-history.",
    )
    parser.add_argument("--current-raw-root", type=Path, required=True)
    parser.add_argument("--bcf-raw-root", type=Path, required=True)
    parser.add_argument("--phase", type=int, choices=(0, 15, 30, 45), required=True)
    parser.add_argument("--start", default="2026-05-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    start, end = utc(args.start), utc(args.end)
    if start >= end:
        raise ValueError("start must precede end")
    if args.current_native_history_root is None and args.current_history is None:
        raise ValueError("provide --current-history or --current-native-history-root")
    if args.bcf_native_history_root is None and args.bcf_history is None:
        raise ValueError("provide --bcf-history or --bcf-native-history-root")

    current_history = (
        current_native_history(args.current_native_history_root, args.phase, start)
        if args.current_native_history_root is not None
        else history(args.current_history, start)
    )
    bcf_history = (
        bcf_native_history(args.bcf_native_history_root, args.current_raw_root, args.phase, start)
        if args.bcf_native_history_root is not None
        else history(args.bcf_history, start)
    )

    current = combine(
        current_history,
        current_replacement(args.current_raw_root, args.phase),
        start,
        end,
    )
    bcf = combine(
        bcf_history,
        bcf_replacement(args.bcf_raw_root, args.current_raw_root, args.phase),
        start,
        end,
    )
    args.out_dir.mkdir(parents=True)
    current_path = args.out_dir / "current_scores_target_free.parquet"
    bcf_path = args.out_dir / "bcf_scores_target_free.parquet"
    current.to_parquet(current_path, index=False, compression="zstd")
    bcf.to_parquet(bcf_path, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_phase_h1_mc1_target_free_panel_v1",
        "phase_minutes": args.phase,
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "target_free_before_policy_join": True,
        "current": {"path": str(current_path), "sha256": sha256(current_path), **audit(current, start, end)},
        "bcf": {"path": str(bcf_path), "sha256": sha256(bcf_path), **audit(bcf, start, end)},
        "inputs": {
            "current_history": str(args.current_history) if args.current_history else None,
            "bcf_history": str(args.bcf_history) if args.bcf_history else None,
            "current_native_history_root": str(args.current_native_history_root) if args.current_native_history_root else None,
            "bcf_native_history_root": str(args.bcf_native_history_root) if args.bcf_native_history_root else None,
            "current_raw_root": str(args.current_raw_root),
            "bcf_raw_root": str(args.bcf_raw_root),
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "phase": args.phase, "current": manifest["current"], "bcf": manifest["bcf"]}, sort_keys=True))


if __name__ == "__main__":
    main()
