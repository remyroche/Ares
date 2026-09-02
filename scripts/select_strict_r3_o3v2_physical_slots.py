#!/usr/bin/env python3
"""Select exactly one frozen physical correction slot per target family.

The prior O3-v2 research averaged five physical cap/weight slots inside every
target family.  That makes the downstream mapper see several highly related
views of the same target.  This selector instead evaluates every physical
slot on a *later, declared development block* and freezes one winner per
target.  It never refits a score, writes no outcome column into a score panel,
and never changes live artifacts.

Selection order is deliberately explicit:

    preselection score-only query geometry
      -> target-free physical-slot score receipts
      -> later development outcome diagnostics
      -> one slot per target
      -> untouched forward scoring / support / MC1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCHEMA = "strict_r3_o3v2_physical_slot_selection_v1"
TAILS = ((.01, "top1"), (.02, "top2"), (.05, "top5"))
DEFAULT_ARMS = (
    "T1_economic_residual_lambdarank",
    "T2_economic_residual_ordinal",
    "T4_hard_inversion_lambdarank",
    "T6_rank_error_ordinal",
    "T8_exit5_lambdarank",
    "T9_exit5_ordinal",
)
PROHIBITED_SCORE_COLUMNS = {
    "policy_net_bps", "policy_gross_bps", "policy_path_valid",
    "semantic_path_valid", "semantic_archetype", "semantic_tbm_event",
    "semantic_axis_a_sequence", "semantic_axis_f_exit4", "semantic_axis_f_exit5",
    "semantic_label_available_ts",
}


def _months(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("at least one YYYY-MM month is required")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _utility(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    outcome = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
    valid = np.isfinite(score) & np.isfinite(outcome)
    if valid.sum() < 100:
        raise AssertionError("insufficient valid score/outcome pairs for a physical-slot diagnostic")
    values: dict[str, float] = {
        "rank_ic": float(spearmanr(score[valid], outcome[valid]).statistic),
    }
    for tail, name in TAILS:
        threshold = np.quantile(score[valid], 1.0 - tail, method="higher")
        selected = outcome[valid & (score >= threshold)]
        values[name] = float(np.mean(selected))
        values[f"{name}_trades"] = float(len(selected))
    values["utility"] = (
        .40 * values["top1"] + .35 * values["top2"] + .25 * values["top5"]
        + 25.0 * values["rank_ic"]
    )
    return values


def _load_query_contract(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_o3v2_query_selection_v1":
        raise AssertionError("query contract has an unknown schema")
    if not isinstance(payload.get("selected_query_mode"), str):
        raise AssertionError("query contract has no selected mode")
    months = payload.get("development_months")
    if not isinstance(months, list) or not months:
        raise AssertionError("query contract has no development months")
    return payload


def _require_later(months: Sequence[str], prior: Sequence[str], *, name: str) -> None:
    if min(months) <= max(prior):
        raise AssertionError(f"{name} must occur strictly after the query-selector development block")


def _score_path(root: Path, arm: str, month: str) -> Path:
    return root / "target_free_scores" / arm / f"month={month}.parquet"


def _load_score(path: Path, *, expected_query: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    leaked = PROHIBITED_SCORE_COLUMNS.intersection(frame.columns)
    if leaked:
        raise AssertionError(f"{path}: target-free score receipt contains outcome fields {sorted(leaked)}")
    required = {"candidate_id", "__decision_ts__", "side_name", "base_rank_ts"}
    if missing := required - set(frame.columns):
        raise AssertionError(f"{path}: missing identity/base fields {sorted(missing)}")
    slots = [column for column in frame.columns if column.startswith("head__") and column.endswith("__rank")]
    if len(slots) != 5 or len(set(slots)) != 5:
        raise AssertionError(f"{path}: expected exactly five physical head ranks, found {slots}")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{path}: duplicate candidate identity")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return frame


def _verify_source_contract(root: Path, query_contract: dict[str, object]) -> None:
    contract_path = root / "run_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    payload = json.loads(contract_path.read_text())
    actual = payload.get("query_mode")
    expected = query_contract["selected_query_mode"]
    if actual != expected:
        raise AssertionError(f"target score receipt query {actual!r} differs from sealed query {expected!r}")


def run(
    *, score_root: Path, policy_path: Path, query_contract_path: Path, out: Path,
    development_months: tuple[str, ...], forward_months: tuple[str, ...], arms: tuple[str, ...],
) -> None:
    if out.exists():
        raise FileExistsError(out)
    if not set(arms).issubset(DEFAULT_ARMS):
        raise ValueError(f"unknown target arms: {sorted(set(arms) - set(DEFAULT_ARMS))}")
    query_contract = _load_query_contract(query_contract_path)
    _require_later(development_months, tuple(str(value) for value in query_contract["development_months"]), name="physical-slot development")
    if forward_months and min(forward_months) <= max(development_months):
        raise AssertionError("forward months must be strictly later than physical-slot development")
    _verify_source_contract(score_root, query_contract)
    policy = pd.read_parquet(policy_path, columns=("candidate_id", "policy_path_valid", "policy_net_bps"))
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("policy source has duplicate candidate IDs")
    out.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    all_months = tuple(dict.fromkeys((*development_months, *forward_months)))
    for arm in arms:
        for month in all_months:
            score = _load_score(_score_path(score_root, arm, month), expected_query=str(query_contract["selected_query_mode"]))
            joined = score.merge(policy, on="candidate_id", how="left", validate="one_to_one")
            valid = joined["policy_path_valid"].fillna(False).astype(bool)
            joined = joined.loc[valid].copy()
            for column in [name for name in score.columns if name.startswith("head__") and name.endswith("__rank")]:
                # Preserve the deployed 75/25 base-plus-correction interface;
                # only the correction source changes from five slots to one.
                combined = (
                    .75 * pd.to_numeric(joined["base_rank_ts"], errors="coerce").to_numpy(float)
                    + .25 * pd.to_numeric(joined[column], errors="coerce").to_numpy(float)
                )
                metric = _utility(joined, combined)
                rows.append({"arm": arm, "month": month, "physical_slot": column.removeprefix("head__").removesuffix("__rank"), **metric})
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_parquet(out / "physical_slot_metrics.parquet", index=False, compression="zstd")
    selection_rows: list[dict[str, object]] = []
    winners: dict[str, str] = {}
    for arm, part in diagnostics.loc[diagnostics["month"].isin(development_months)].groupby("arm", sort=True):
        grouped = part.groupby("physical_slot", sort=True)
        candidate_rows = []
        for slot, slot_part in grouped:
            if slot_part["month"].nunique() != len(development_months):
                raise AssertionError(f"{arm} {slot}: incomplete physical-slot development coverage")
            utility = slot_part["utility"].to_numpy(float)
            candidate_rows.append({
                "arm": arm, "physical_slot": slot,
                "utility_mean": float(utility.mean()),
                "utility_std": float(utility.std(ddof=0)),
                "utility_worst_month": float(utility.min()),
                "selection_score": float(utility.mean() - .25 * utility.std(ddof=0) - max(0.0, -utility.min())),
                "top1_mean": float(slot_part["top1"].mean()),
                "top2_mean": float(slot_part["top2"].mean()),
                "top5_mean": float(slot_part["top5"].mean()),
                "rank_ic_mean": float(slot_part["rank_ic"].mean()),
            })
        table = pd.DataFrame(candidate_rows).sort_values(
            ["selection_score", "utility_worst_month", "top1_mean", "physical_slot"],
            ascending=[False, False, False, True], kind="stable",
        ).reset_index(drop=True)
        winner = table.iloc[0]
        winners[str(arm)] = str(winner["physical_slot"])
        selection_rows.extend(table.to_dict("records"))
    selection = pd.DataFrame(selection_rows)
    selection.to_parquet(out / "physical_slot_development_selection.parquet", index=False, compression="zstd")
    _exclusive_json(out / "selected_physical_slots.json", {
        "schema": SCHEMA,
        "query_contract_path": str(query_contract_path),
        "query_contract_sha256": _sha256(query_contract_path),
        "query_mode": query_contract["selected_query_mode"],
        "query_development_months": query_contract["development_months"],
        "development_months": list(development_months),
        "forward_months": list(forward_months),
        "selection": "one physical slot per target: utility mean minus 0.25 standard deviation, then worst month and top-1 mean",
        "score_interface": "0.75 base_rank_ts + 0.25 individual physical-slot rank",
        "selected_slots": winners,
        "held_out_rule": "forward months are diagnostics only and never read by this selector",
    })
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline O3-v2 physical-slot selection only; no live/MC1/admission/portfolio mutation",
        "score_root": str(score_root),
        "score_contract": str(score_root / "run_contract.json"),
        "policy_path": str(policy_path),
        "query_contract": str(query_contract_path),
        "development_months": list(development_months),
        "forward_months": list(forward_months),
        "arms": list(arms),
        "causality": {
            "scores": "existing target-free receipts are read before policy diagnostics are joined",
            "slot_selection": "strictly after sealed query selector and before forward months",
            "one_slot_per_target": True,
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--query-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-months", required=True)
    parser.add_argument("--forward-months", default="")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    args = parser.parse_args()
    run(
        score_root=args.score_root, policy_path=args.policy_path,
        query_contract_path=args.query_contract, out=args.out,
        development_months=_months(args.development_months),
        forward_months=_months(args.forward_months) if args.forward_months else (),
        arms=tuple(value for value in args.arms.split(",") if value),
    )


if __name__ == "__main__":
    main()
