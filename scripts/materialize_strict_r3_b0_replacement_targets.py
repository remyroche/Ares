#!/usr/bin/env python3
"""Materialise predeclared B0-replacement labels from outcome-only sidecars.

The output is a supervised-label sidecar.  It has no role in target-free
feature construction or inference.  Missing/incomplete paths stay invalid.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
LADDERS = {
    "tbm_a": (1, 2, 3, 4, 6),
    "tbm_b": (2, 3, 4, 6, 8),
    "tbm_c": (1, 2, 4, 6, 10),
}


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(root: Path) -> list[str]:
    return sorted(path.name.split("=", 1)[1] for path in root.glob("month=*") if path.is_dir())


def _parts_root(root: Path) -> Path:
    return root / "parts" if (root / "parts").is_dir() else root


def _ordinal(net: pd.Series, floor50: bool) -> np.ndarray:
    values = pd.to_numeric(net, errors="coerce").to_numpy(float)
    edges = [50, 100, 150, 250, 400] if floor50 else [0, 50, 100, 200, 400]
    return np.searchsorted(edges, values, side="right").astype(np.int8)


def _tbm(frame: pd.DataFrame, ladder: tuple[int, ...]) -> np.ndarray:
    adverse_hit = frame["aux_reached_adverse_3atr"].fillna(0).astype(bool).to_numpy()
    adverse_time = pd.to_numeric(frame["aux_time_to_adverse_3atr_h"], errors="coerce").to_numpy(float)
    grade = np.zeros(len(frame), dtype=np.int8)
    for level in ladder:
        reached = frame[f"aux_reached_{level}atr"].fillna(0).astype(bool).to_numpy()
        favourable_time = pd.to_numeric(frame[f"aux_time_to_{level}atr_h"], errors="coerce").to_numpy(float)
        # Same 15-minute interval is deliberately ambiguous and does not earn
        # a favourable grade; historical data cannot prove intrabar ordering.
        clean = reached & (~adverse_hit | (favourable_time < adverse_time))
        grade[clean] = np.maximum(grade[clean], np.int8(ladder.index(level) + 1))
    return grade


def _path_quality(frame: pd.DataFrame) -> np.ndarray:
    reached2 = frame["aux_reached_2atr"].fillna(0).astype(bool).to_numpy()
    reached3 = frame["aux_reached_3atr"].fillna(0).astype(bool).to_numpy()
    reached4 = frame["aux_reached_4atr"].fillna(0).astype(bool).to_numpy()
    reached6 = frame["aux_reached_6atr"].fillna(0).astype(bool).to_numpy()
    adverse = frame["aux_reached_adverse_3atr"].fillna(0).astype(bool).to_numpy()
    t2 = pd.to_numeric(frame["aux_time_to_2atr_h"], errors="coerce").to_numpy(float)
    ta = pd.to_numeric(frame["aux_time_to_adverse_3atr_h"], errors="coerce").to_numpy(float)
    mae = pd.to_numeric(frame["aux_mae_atr_12h"], errors="coerce").to_numpy(float)
    grade = np.ones(len(frame), dtype=np.int8)
    severe = adverse & (~reached2 | (ta <= t2))
    grade[severe] = 0
    grade[reached2 & ~severe] = 2
    grade[reached3 & (mae <= 2.0) & ~severe] = 3
    grade[reached4 & (mae <= 1.5) & ~severe] = 4
    grade[reached6 & (mae <= 1.0) & ~severe] = 5
    return grade


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aux-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-month", default="2024-12")
    parser.add_argument("--end-month", default="2026-07")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    policy = pd.read_parquet(args.policy_path, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])
    policy.policy_label_available_ts = pd.to_datetime(policy.policy_label_available_ts, utc=True, errors="coerce")
    policy.policy_path_valid = policy.policy_path_valid.fillna(False).astype(bool)
    audit: list[dict[str, object]] = []
    parts_root = _parts_root(args.aux_root)
    for token in _months(parts_root):
        if token < args.start_month or token > args.end_month:
            continue
        path = parts_root / f"month={token}" / "auxiliary_path_labels.parquet"
        aux = pd.read_parquet(path).copy()
        aux["__decision_ts__"] = pd.to_datetime(aux["__decision_ts__"], utc=True, errors="raise")
        aux["aux_label_available_ts"] = pd.to_datetime(aux["aux_label_available_ts"], utc=True, errors="coerce")
        frame = aux.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        # The policy ledger starts later than the reusable auxiliary path
        # sidecar.  Preserve path labels; a missing policy outcome makes only
        # magnitude supervision invalid, never a fabricated adverse outcome.
        frame["policy_path_valid"] = frame["policy_path_valid"].fillna(False).astype(bool)
        path_valid = frame.aux_path_valid.fillna(False).astype(bool) & frame.aux_path_complete.fillna(False).astype(bool)
        policy_valid = frame.policy_path_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
        output = frame.loc[:, list(IDENTITY)].copy()
        output["label_available_ts"] = pd.concat([frame.aux_label_available_ts, frame.policy_label_available_ts], axis=1).max(axis=1)
        for name, ladder in LADDERS.items():
            output[f"{name}_valid"] = path_valid
            output[f"{name}_grade"] = _tbm(frame, ladder)
        output["policy_ordinal_base_valid"] = policy_valid
        output["policy_ordinal_base_grade"] = _ordinal(frame.policy_net_bps, False)
        output["policy_ordinal_floor50_valid"] = policy_valid
        output["policy_ordinal_floor50_grade"] = _ordinal(frame.policy_net_bps, True)
        output["path_quality_valid"] = path_valid
        output["path_quality_grade"] = _path_quality(frame)
        output["policy_net_bps"] = pd.to_numeric(frame.policy_net_bps, errors="coerce")
        destination = args.out / f"month={token}"
        destination.mkdir()
        output.to_parquet(destination / "b0_replacement_targets.parquet", index=False, compression="zstd")
        audit.append({"month": token, "rows": len(output), "path_valid": int(path_valid.sum()), "policy_valid": int(policy_valid.sum()), **{f"{name}_grade{grade}": int((output[f"{name}_grade"] == grade).sum()) for name in (*LADDERS, "policy_ordinal_base", "policy_ordinal_floor50", "path_quality") for grade in range(6)}})
    pd.DataFrame(audit).to_parquet(args.out / "coverage_and_grade_audit.parquet", index=False, compression="zstd")
    _exclusive_json(args.out / "run_manifest.json", {"schema": "strict_r3_b0_replacement_target_labels_v1", "scope": "labels only; never inference features", "tbm": {name: {"favourable_atr_ladder": ladder, "adverse_barrier": "3 ATR policy-aligned", "same_interval": "conservative non-success"} for name, ladder in LADDERS.items()}, "policy_bins": {"base": [0, 50, 100, 200, 400], "floor50": [50, 100, 150, 250, 400]}, "path_quality": "six-state clean-path ordinal", "months": [row["month"] for row in audit]})


if __name__ == "__main__":
    main()
