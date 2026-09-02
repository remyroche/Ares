#!/usr/bin/env python3
"""Score a frozen G3 O3-v2 contract into later target-free OOF receipts.

Feature selection is completed before this producer is invoked.  Each held
month is fitted only on its own six-month resolved window before a 28-day
reserve, then emitted without any outcome columns.  This makes a selected
G3 head directly consumable by the normal selected-slot MC1 portfolio runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_o3v2_greedy_features as g3  # noqa: E402
import run_strict_r3_o3v2_target_funnel as target  # noqa: E402


SCHEMA = "strict_r3_o3v2_g3_forward_score_v1"
PROHIBITED = set(target.PROHIBITED_SCORE_COLUMNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*.parquet")) if path.is_dir() else (path,):
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    months = tuple(pd.Timestamp(f"{token.strip()}-01", tz="UTC") for token in raw.split(",") if token.strip())
    if not months:
        raise ValueError("at least one held month is required")
    return months


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_contract(path: Path, target_name: str) -> tuple[dict[str, object], tuple[str, ...]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_o3v2_greedy_features_v2":
        raise AssertionError("forward G3 scoring requires the selected-slot v2 G3 contract")
    if payload.get("target") != target_name:
        raise AssertionError("G3 feature contract target differs from requested forward target")
    fields = tuple(str(value) for value in payload.get("contracts", {}).get("mixed", []))
    if not fields or not set(g3.CORE).issubset(fields):
        raise AssertionError("G3 mixed contract lacks the frozen core fields")
    return payload, fields


def _project(held: pd.DataFrame, *, slot: str, raw: np.ndarray, rank: np.ndarray) -> pd.DataFrame:
    out = held.loc[:, ["candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts"]].copy()
    out = out.rename(columns={"f1_base_rank_ts": "base_rank_ts"})
    out[f"head__{slot}__raw"] = np.asarray(raw, dtype=np.float32)
    out[f"head__{slot}__rank"] = np.asarray(rank, dtype=np.float32)
    leaked = PROHIBITED.intersection(out.columns)
    if leaked:
        raise AssertionError(f"target-free G3 forward projection leaked outcome fields: {sorted(leaked)}")
    if out["candidate_id"].duplicated().any():
        raise AssertionError("duplicate forward G3 candidate identities")
    return out


def run(
    *, history_panel: Path, policy_path: Path, g3_contract_path: Path,
    physical_slot_selection: Path, query_contract: Path, out: Path,
    target_name: str, months: Sequence[pd.Timestamp], n_jobs: int,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    contract, fields = _load_contract(g3_contract_path, target_name)
    query_mode = str(contract.get("query"))
    g3._load_query_contract(query_contract, query_mode)
    slot, cap, weight_mode = g3._load_physical_slot_contract(
        physical_slot_selection, target_name, query_mode=query_mode,
    )
    declared_slot = contract.get("physical_slot", {})
    if declared_slot != {"name": slot, "cap": cap, "weight_mode": weight_mode}:
        raise AssertionError("G3 forward slot differs from the sealed selection contract")
    g3_manifest_path = g3_contract_path.parent / "run_manifest.json"
    if not g3_manifest_path.exists():
        raise FileNotFoundError("G3 feature contract lacks its immutable manifest")
    g3_manifest = json.loads(g3_manifest_path.read_text())
    # Reuse the producer's byte-stream hash convention exactly.  The generic
    # directory hash below includes path names for its own manifest, whereas
    # the selected G3 receipt deliberately stores only the history file bytes.
    if g3_manifest.get("history_panel_sha256") != g3._sha256(history_panel):
        raise AssertionError("forward history panel differs from the G3-selected target-free ledger")
    history = g3._load_history(history_panel, fields)
    policy = g3._load_policy(policy_path)
    folds = g3._prepare_folds(history, policy, months)
    out.mkdir(parents=True)
    receipt_root = out / "target_free_scores" / target_name
    receipt_root.mkdir(parents=True)
    audit_rows: list[dict[str, object]] = []
    for index, fold in enumerate(folds):
        raw, rank = g3._fit_score(
            fold.train, fold.held, fields, target_name, g3.SEED + 80_000 + index,
            n_jobs=n_jobs, query_mode=query_mode, physical_slot=slot, cap=cap, weight_mode=weight_mode,
        )
        score = _project(fold.held, slot=slot, raw=raw, rank=rank)
        path = receipt_root / f"month={fold.month:%Y-%m}.parquet"
        score.to_parquet(path, index=False, compression="zstd")
        audit_rows.append({
            "month": f"{fold.month:%Y-%m}", "train_rows": int(len(fold.train)),
            "held_rows": int(len(score)), "train_end_reserve": str(fold.month - pd.Timedelta(days=g3.RESERVE_DAYS)),
            "head_slot": slot, "head_cap": cap, "head_weight_mode": weight_mode,
            "field_count": len(fields), "field_complete_fraction": float(score[f"head__{slot}__rank"].notna().mean()),
            "target_free": True,
        })
    pd.DataFrame(audit_rows).to_parquet(out / "forward_score_audit.parquet", index=False, compression="zstd")
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-prequential target-free G3 forward scoring; no MC1, admission, portfolio, live, or canonical mutation",
        "target": target_name, "months": [f"{month:%Y-%m}" for month in months],
        "history_panel": str(history_panel.resolve()), "history_panel_sha256": g3._sha256(history_panel),
        "policy_path": str(policy_path.resolve()), "policy_path_sha256": g3._sha256(policy_path),
        "g3_contract": str(g3_contract_path.resolve()), "g3_contract_sha256": _sha256(g3_contract_path),
        "physical_slot_selection": str(physical_slot_selection.resolve()),
        "physical_slot_selection_sha256": _sha256(physical_slot_selection),
        "query_contract": str(query_contract.resolve()), "query_contract_sha256": _sha256(query_contract),
        "head": {"slot": slot, "cap": cap, "weight_mode": weight_mode, "query": query_mode, "fields": list(fields)},
        "training": "six full calendar months preceding a 28-day reserve; only policy labels resolved before reserve",
        "routing": "exact deterministic timestamp-local top 30 percent",
        "causality": "target-free held receipts are sealed before any downstream policy join",
        "fit_runtime": {"lightgbm_n_jobs": n_jobs, "deterministic": True},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-panel", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--g3-contract", type=Path, required=True)
    parser.add_argument("--physical-slot-selection", type=Path, required=True)
    parser.add_argument("--query-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", choices=("T1_economic_residual_lambdarank", "T2_economic_residual_ordinal", "T4_hard_inversion_lambdarank", "T6_rank_error_ordinal"), required=True)
    parser.add_argument("--months", required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.n_jobs <= 8:
        parser.error("--n-jobs must be between 1 and 8")
    run(
        history_panel=args.history_panel, policy_path=args.policy_path, g3_contract_path=args.g3_contract,
        physical_slot_selection=args.physical_slot_selection, query_contract=args.query_contract, out=args.out,
        target_name=args.target, months=_months(args.months), n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
