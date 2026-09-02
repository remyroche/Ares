#!/usr/bin/env python3
"""Create a target-free, phase-native historical score ledger.

This utility is deliberately limited to joining raw frozen current-v5 and BCF
scores from the *same shifted phase*.  It contains no outcome columns and is
the only permitted pre-May history for an off-hour MC1 mapper.
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


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _read_current(paths: list[Path]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=list(SCORE_COLUMNS)).copy()
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        pieces.append(frame)
    out = pd.concat(pieces, ignore_index=True)
    if out["candidate_id"].duplicated().any():
        raise ValueError("historical current raw blocks overlap candidate identities")
    if not out["side_name"].astype(str).eq("long").all():
        raise ValueError("phase-native ledger is long-only")
    return out.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _read_bcf(paths: list[Path], identity: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in paths:
        raw = pd.read_parquet(path).copy()
        raw["candidate_id"] = raw["candidate_id"].astype(str)
        raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
        if raw["candidate_id"].duplicated().any():
            raise ValueError(f"duplicate BCF candidate identities in {path}")
        native = derive_bcf_mc1_features(raw)
        piece = (
            raw.loc[:, ["candidate_id", "__decision_ts__", "side_name"]]
            .merge(identity.loc[:, ["candidate_id", "__symbol__"]], on="candidate_id", how="inner", validate="one_to_one")
            .merge(native, on="candidate_id", how="inner", validate="one_to_one")
            .loc[:, list(SCORE_COLUMNS)]
        )
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    if out["candidate_id"].duplicated().any():
        raise ValueError("historical BCF raw blocks overlap candidate identities")
    return out.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _audit(frame: pd.DataFrame, phase: int) -> dict[str, object]:
    forbidden = [
        name for name in frame.columns
        if name.startswith("policy_") or name in {"gross_bps", "net_bps", "outcome", "label"}
    ]
    minute = pd.to_datetime(frame["__decision_ts__"], utc=True).dt.minute
    if not minute.eq(phase).all():
        raise AssertionError(f"phase-native ledger contains non-{phase:02d} timestamps")
    if forbidden:
        raise AssertionError(f"target-free score ledger contains outcome columns: {forbidden}")
    return {
        "rows": int(len(frame)),
        "min_ts": frame["__decision_ts__"].min().isoformat(),
        "max_ts": frame["__decision_ts__"].max().isoformat(),
        "phase_minute_mismatches": int((~minute.eq(phase)).sum()),
        "forbidden_outcome_columns": forbidden,
        "duplicate_candidate_ids": int(frame["candidate_id"].duplicated().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--phase", type=int, choices=(15, 30, 45), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-current-blocks", type=int, default=3)
    parser.add_argument("--expected-bcf-months", type=int, default=2)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")

    current_paths = sorted(args.raw_root.glob(f"current/block=*/phase={args.phase:02d}/predictions.parquet"))
    bcf_paths = sorted(args.raw_root.glob(f"bcf/month=*/phase={args.phase:02d}/score/predictions.parquet"))
    if len(current_paths) != args.expected_current_blocks:
        raise ValueError(f"expected {args.expected_current_blocks} current blocks, found {len(current_paths)}")
    if len(bcf_paths) != args.expected_bcf_months:
        raise ValueError(f"expected {args.expected_bcf_months} BCF months, found {len(bcf_paths)}")

    current = _read_current(current_paths)
    bcf = _read_bcf(bcf_paths, current.loc[:, list(IDENTITY)])
    current_audit = _audit(current, args.phase)
    bcf_audit = _audit(bcf, args.phase)

    args.out_dir.mkdir(parents=True)
    current_path = args.out_dir / "current_scores_target_free.parquet"
    bcf_path = args.out_dir / "bcf_scores_target_free.parquet"
    current.to_parquet(current_path, index=False, compression="zstd")
    bcf.to_parquet(bcf_path, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_phase_native_historical_score_ledger_v1",
        "phase_minutes": args.phase,
        "target_free_before_policy_join": True,
        "current": {"path": str(current_path), "sha256": _sha(current_path), **current_audit},
        "bcf": {"path": str(bcf_path), "sha256": _sha(bcf_path), **bcf_audit},
        "raw_current_paths": [str(path) for path in current_paths],
        "raw_bcf_paths": [str(path) for path in bcf_paths],
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "phase": args.phase, "current": current_audit, "bcf": bcf_audit}, sort_keys=True))


if __name__ == "__main__":
    main()
