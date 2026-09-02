#!/usr/bin/env python3
"""Freeze the five H1 family roles into the canonical O3 MC1 head slots.

Unlike the development head-subset adapter, this producer performs *no* head
selection.  It maps the five predeclared H1 role outputs into stable MC1 slots
so a subsequent MC1 fit can train solely on post-feature-selection scores.
That avoids using development-period policy outcomes twice: once to select a
head and again to fit the mapping.

Research-only.  Inputs and outputs are target-free score receipts; policy,
semantic labels, admission and live artifacts are not read or changed here.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_o3v2_target_funnel as target  # noqa: E402


SCHEMA = "strict_r3_o3v2_fixed_family_head_adapter_v1"
TARGET = "T6_rank_error_ordinal"
ROLE_TO_SLOT = (
    ("h1_base_geometry", "cap100_ordinary"),
    ("h1_query_geometry", "cap80_ordinary"),
    ("h1_recent_error", "cap120_equal_month"),
    ("h1_state_transition", "cap40_equal_month"),
    ("h1_g1_mixed", "cap60_equal_month"),
)
PROHIBITED = set(target.PROHIBITED_SCORE_COLUMNS)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for child in paths:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _source(root: Path, month: str) -> Path:
    path = root / "target_free_scores" / TARGET / f"month={month}" / "scores.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run(*, source_root: Path, contract_path: Path, out: Path, months: Sequence[str]) -> None:
    if out.exists():
        raise FileExistsError(out)
    contract = json.loads(contract_path.read_text())
    if contract.get("target") != TARGET:
        raise AssertionError(f"feature contract target {contract.get('target')!r} does not match {TARGET!r}")
    selection_end = max(pd.Timestamp(f"{token}-01", tz="UTC") for token in contract["development_months"]) + pd.offsets.MonthBegin(1)
    parsed_months = tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in months)
    if not parsed_months or min(parsed_months) < selection_end:
        raise AssertionError(
            "fixed-role adapter may emit only months after the feature-selection development block"
        )
    expected = {
        role: f"{TARGET.lower()}__{role}__rank"
        for role, _slot in ROLE_TO_SLOT
    }
    out.mkdir(parents=True)
    target_root = out / "target_free_scores" / TARGET
    target_root.mkdir(parents=True)
    audit: list[dict[str, object]] = []
    for month in months:
        source = _source(source_root, month)
        raw = pd.read_parquet(source)
        if leaked := sorted(PROHIBITED.intersection(raw.columns)):
            raise AssertionError(f"{source}: outcome field in H1 score receipt: {leaked}")
        required = {"candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", *expected.values()}
        if missing := sorted(required - set(raw.columns)):
            raise KeyError(f"{source}: missing fixed H1 role fields {missing}")
        result = raw.loc[:, ["candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", *expected.values()]].copy()
        result = result.rename(columns={"f1_base_rank_ts": "base_rank_ts"})
        role_values = []
        for role, slot in ROLE_TO_SLOT:
            value = pd.to_numeric(result.pop(expected[role]), errors="coerce").to_numpy(np.float32)
            result[f"head__{slot}__rank"] = value
            role_values.append(value)
        ranks = np.column_stack(role_values)
        result["conditional_consensus_rank"] = np.nanmedian(ranks, axis=1).astype(np.float32)
        result["ordinary_shadow_consensus_rank"] = result["conditional_consensus_rank"]
        result["head_agreement_std"] = np.nanstd(ranks, axis=1).astype(np.float32)
        result["o3v2_rank_75_25"] = (
            .75 * pd.to_numeric(result["base_rank_ts"], errors="coerce")
            + .25 * pd.to_numeric(result["conditional_consensus_rank"], errors="coerce")
        ).astype(np.float32)
        if result["candidate_id"].duplicated().any():
            raise AssertionError(f"{source}: duplicate candidate IDs")
        if leaked := sorted(PROHIBITED.intersection(result.columns)):
            raise AssertionError(f"{source}: adapter retained outcome fields: {leaked}")
        result.to_parquet(target_root / f"month={month}.parquet", index=False, compression="zstd")
        audit.append({
            "month": month, "rows": int(len(result)),
            "complete_fraction": float(result.notna().all(axis=1).mean()),
            "min_decision_ts": str(pd.to_datetime(result["__decision_ts__"], utc=True).min()),
            "max_decision_ts": str(pd.to_datetime(result["__decision_ts__"], utc=True).max()),
        })
    pd.DataFrame(audit).to_parquet(out / "adapter_audit.parquet", index=False, compression="zstd")
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "target-free fixed-H1-role adapter only; no selection, fit, outcome join, MC1, admission, portfolio, or live change",
        "target": TARGET,
        "months": list(months),
        "role_to_slot": {role: slot for role, slot in ROLE_TO_SLOT},
        "feature_contract": str(contract_path),
        "feature_contract_hash": _hash(contract_path),
        "selection_boundary": {
            "development_months": contract["development_months"],
            "first_emitted_month_must_be_after": f"{selection_end:%Y-%m}",
        },
        "source_root": str(source_root),
        "source_root_hash": _hash(source_root),
        "causality": {
            "roles": "five predeclared H1 family roles; no later head quality or outcome selection",
            "inputs": "strict target-free specialist score receipts only",
            "outputs": "strict target-free MC1-compatible head slots only",
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", required=True)
    args = parser.parse_args()
    run(
        source_root=args.source_root,
        contract_path=args.feature_contract,
        out=args.out,
        months=tuple(token for token in args.months.split(",") if token),
    )


if __name__ == "__main__":
    main()
