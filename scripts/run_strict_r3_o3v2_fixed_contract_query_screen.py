#!/usr/bin/env python3
"""Strict target-free query-localisation screen for a frozen O3-v2 contract.

The feature contract is selected before this script is run.  For every
declared query geometry it retrains the *same* T1 economic-residual LambdaRank
head using only fully resolved rows before a 28-day reserve, then writes a
held-month score receipt before any held outcome is joined.  It is research
only: it does not modify MC1, admission, portfolio, canonical, or live
artifacts.

This is deliberately separate from the old T2/T6 query screen.  Those arms
use L2 objectives and, below the training cap, their query declaration cannot
affect a fit.  T1 uses native LambdaRank, so exact timestamp, base-band, and
four-hour query definitions genuinely reach the loss function.
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
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402
import run_strict_r3_o3v2_greedy_features as g3  # noqa: E402
import run_strict_r3_o3v2_target_funnel as target  # noqa: E402


SCHEMA = "strict_r3_o3v2_fixed_contract_query_screen_v1"
TARGET = "T1_economic_residual_lambdarank"
SEED = 1729
TRAIN_MONTHS = 6
RESERVE_DAYS = 28
MIN_ROWS = 5_000
QUERY_MODES = (
    "exact_timestamp_side",
    "exact_timestamp_baseband_side",
    "cycle_4h_side",
)
PROHIBITED = set(target.PROHIBITED_SCORE_COLUMNS)
# This small contract is deliberately declared in source rather than borrowed
# from a later outcome-selected G3 artifact.  It is used only to select the
# LambdaRank query geometry before *any* cross-family feature selection.  The
# fields are the immutable enhanced-base coordinates available at inference.
PRESELECTION_CORE_SCORE_FIELDS = (
    "f1_enhanced_base_bps", "f1_base_rank_ts", "f1_base_bps",
    "f1_efficiency_bps", "f1_timing_bps", "f1_e_minus_t",
    "f1_e_minus_b0", "f1_t_minus_b0", "f1_base_component_std",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*.parquet")) if path.is_dir() else (path,):
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, value: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(f"{token.strip()}-01", tz="UTC") for token in raw.split(",") if token.strip())


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _load_contract(path: Path | None, *, preselection_core_score_only: bool = False) -> tuple[str, ...]:
    if preselection_core_score_only:
        return PRESELECTION_CORE_SCORE_FIELDS
    if path is None:
        raise ValueError("--contract is required unless --preselection-core-score-only is set")
    raw = json.loads(path.read_text())
    if raw.get("target") != TARGET:
        raise AssertionError(f"fixed contract target {raw.get('target')!r} is not {TARGET!r}")
    contracts = raw.get("contracts")
    if not isinstance(contracts, dict):
        raise AssertionError("fixed contract lacks contracts")
    values = contracts.get("mixed")
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise AssertionError("fixed contract mixed fields are missing or malformed")
    if len(values) != len(set(values)):
        raise AssertionError("fixed contract has duplicate mixed fields")
    return tuple(values)


def _fit_t1(
    train: pd.DataFrame,
    held: pd.DataFrame,
    fields: Sequence[str],
    *,
    query_mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Fit the declared seven-grade T1 LambdaRank head on one strict fold."""
    base_rank = pd.to_numeric(train["f1_base_rank_ts"], errors="coerce").to_numpy(float)
    policy = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    anchor = target.IsotonicRegression(increasing=True, out_of_bounds="clip").fit(base_rank, policy)
    residual = np.clip(policy - anchor.predict(base_rank), -500.0, 500.0).astype(np.float32)
    grade = target._economic_residual_grade(residual)
    spec = parent.ConsensusHeadSpec(
        name=f"t1_fixed_query__{query_mode}", cap=100, weight_mode="ordinary", query=query_mode,
        fields=tuple(fields), target_edges_bps=(-100.0, -30.0, 30.0, 90.0),
        params={
            "objective": "lambdarank", "metric": "ndcg", "n_estimators": 120,
            "learning_rate": .035, "max_depth": 4, "num_leaves": 15,
            "min_child_samples": max(300, int(.015 * len(train))), "feature_fraction": .82,
            "bagging_fraction": .82, "bagging_freq": 1, "lambda_l1": .02,
            "lambda_l2": 2.0, "max_bin": 127, "label_gain": [0, 1, 2, 4, 7, 12, 20],
            "lambdarank_truncation_level": 10, "verbosity": -1,
        },
    )
    # ``exact_timestamp_baseband_side`` is defined on the canonical parent
    # alias.  The selected target-free panel keeps that provenance under the
    # explicit F1 name, so expose the alias only in this train-local frame.
    # Without it, a base-band request can silently fall back or fail before
    # groups are built; the assertion below makes the required wiring plain.
    source = train.assign(
        enhanced_base_bps=pd.to_numeric(train["f1_enhanced_base_bps"], errors="coerce"),
        base_rank_ts=pd.to_numeric(train["f1_base_rank_ts"], errors="coerce"),
    )
    if query_mode == "exact_timestamp_baseband_side" and "base_rank_ts" not in source:
        raise AssertionError("base-band query lacks canonical base_rank_ts in its train-local query frame")
    heads, _ = parent._fit_heads(source, residual, (spec,), objective="ordinal_lambdarank", grade=grade)
    head = heads[0]
    raw, rank = head.predict_rank(held)
    sampled_identity, _labels, groups = parent._sample_complete_consensus_queries(
        source.loc[:, ["candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts"]].rename(columns={"f1_base_rank_ts": "base_rank_ts"}),
        grade,
        spec,
        seed=seed + 1000,
    )
    return raw.astype(np.float32), rank.astype(np.float32), int(len(sampled_identity)), int(len(groups))


def _run_fold(
    *, history: pd.DataFrame, policy: pd.DataFrame, fields: Sequence[str], month: pd.Timestamp,
    query_mode: str, out: Path, seed: int,
) -> tuple[dict[str, object], dict[str, float]]:
    reserve_start = month - pd.Timedelta(days=RESERVE_DAYS)
    train_start = reserve_start - pd.DateOffset(months=TRAIN_MONTHS)
    history_start = pd.to_datetime(history["__decision_ts__"], utc=True, errors="raise").min()
    if history_start > train_start:
        raise AssertionError(
            f"{month:%Y-%m}: incomplete six-month training history; need {train_start.isoformat()}, "
            f"panel begins {history_start.isoformat()}"
        )
    train = history.loc[
        history["__decision_ts__"].ge(train_start) & history["__decision_ts__"].lt(reserve_start)
    ].merge(policy, on="candidate_id", how="left", validate="one_to_one")
    held = history.loc[
        history["__decision_ts__"].ge(month) & history["__decision_ts__"].lt(_month_end(month))
    ].copy()
    train_route = parent._exact_timestamp_top_fraction(train, "f1_enhanced_base_bps", parent.BASE_ROUTE)
    held_route = parent._exact_timestamp_top_fraction(held, "f1_enhanced_base_bps", parent.BASE_ROUTE)
    valid_train = (
        train_route.to_numpy(bool)
        & train["policy_path_valid"].fillna(False).astype(bool).to_numpy()
        & train["policy_label_available_ts"].lt(reserve_start).to_numpy()
        & np.isfinite(pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float))
    )
    train = train.loc[valid_train].copy()
    held = held.loc[held_route.to_numpy(bool)].copy()
    if len(train) < MIN_ROWS or len(held) < 1_000:
        raise AssertionError(f"{query_mode} {month:%Y-%m}: insufficient strict support train={len(train)} held={len(held)}")
    score_path = out / "target_free_scores" / query_mode / f"month={month:%Y-%m}.parquet"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if score_path.exists():
        score = pd.read_parquet(score_path)
        required = {"candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", "g3_raw", "g3_rank", "g3_mix_rank"}
        if missing := required - set(score.columns):
            raise AssertionError(f"{score_path}: incomplete immutable receipt: {sorted(missing)}")
        if not score["candidate_id"].astype(str).equals(held["candidate_id"].astype(str).reset_index(drop=True)):
            raise AssertionError(f"{score_path}: held identity differs from the declared target-free population")
        sampled_rows = sampled_queries = -1
    else:
        raw, rank, sampled_rows, sampled_queries = _fit_t1(
            train, held, fields, query_mode=query_mode, seed=seed,
        )
        score = held.loc[:, ["candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts"]].copy()
        score["g3_raw"] = raw
        score["g3_rank"] = rank
        score["g3_mix_rank"] = (
            .75 * pd.to_numeric(score["f1_base_rank_ts"], errors="coerce") + .25 * rank
        ).astype(np.float32)
        if leaked := PROHIBITED.intersection(score.columns):
            raise AssertionError(f"{query_mode} {month:%Y-%m}: score receipt leaked outcome fields {sorted(leaked)}")
        score.to_parquet(score_path, index=False, compression="zstd")
    held_policy = held.loc[:, ["candidate_id"]].merge(policy, on="candidate_id", how="left", validate="one_to_one")
    utility, metric = g3._metric(score, held_policy)
    audit = {
        "query_mode": query_mode, "month": f"{month:%Y-%m}", "train_start": str(train_start),
        "reserve_start": str(reserve_start), "train_rows": int(len(train)), "held_rows": int(len(held)),
        "sampled_rows": sampled_rows, "sampled_queries": sampled_queries,
        "held_target_free": True, "policy_labels_available_before_reserve": True,
        "query_reaches_lambdarank_loss": True,
    }
    return audit, {"utility": utility, **metric}


def run(
    *, history_panel: Path, contract_path: Path | None, policy_path: Path, out: Path,
    months: Sequence[pd.Timestamp], query_modes: Sequence[str],
    preselection_core_score_only: bool = False,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    unknown = sorted(set(query_modes) - set(QUERY_MODES))
    if unknown:
        raise ValueError(f"unsupported query modes: {unknown}")
    fields = _load_contract(contract_path, preselection_core_score_only=preselection_core_score_only)
    history = g3._load_history(history_panel, fields)
    policy = g3._load_policy(policy_path)
    out.mkdir(parents=True)
    metrics: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for query_index, query_mode in enumerate(query_modes):
        for month_index, month in enumerate(months):
            audit_row, metric = _run_fold(
                history=history, policy=policy, fields=fields, month=month, query_mode=query_mode,
                out=out, seed=SEED + 100_003 * query_index + 1_009 * month_index,
            )
            audit.append(audit_row)
            metrics.append({"query_mode": query_mode, "month": f"{month:%Y-%m}", **metric})
            print(json.dumps({"event": "scored", **audit_row, **metric}), flush=True)
    pd.DataFrame(metrics).to_parquet(out / "query_screen_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(audit).to_parquet(out / "query_screen_audit.parquet", index=False, compression="zstd")
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline fixed-contract query-localisation research; no MC1/admission/portfolio/canonical/live mutation",
        "target": TARGET,
        "contract_path": str(contract_path) if contract_path is not None else None,
        "contract_sha256": _sha256(contract_path) if contract_path is not None else None,
        "feature_contract": {
            "kind": "preselection_core_score_only" if preselection_core_score_only else "outcome_selected_g3_mixed",
            "fields": list(fields),
            "selection_lineage": (
                "declared score-only contract; no outcome-selected additions"
                if preselection_core_score_only
                else "frozen G3 mixed contract"
            ),
        },
        "history_panel": str(history_panel), "history_sha256": _sha256(history_panel),
        "policy_path": str(policy_path), "policy_sha256": _sha256(policy_path),
        "months": [f"{month:%Y-%m}" for month in months], "query_modes": list(query_modes),
        "routing": "exact deterministic timestamp-local top 30 percent by f1_enhanced_base_bps",
        "training": {"calendar_months": TRAIN_MONTHS, "reserve_days": RESERVE_DAYS, "labels": "policy labels resolved strictly before reserve"},
        "causality": {
            "held": "target-free score receipt is persisted before policy is joined for diagnostics",
            "query": "the selected query mode is passed to native LambdaRank groups; no L2-only proxy",
            "selection": (
                "query geometry is selected from the declared score-only contract before G3 feature selection"
                if preselection_core_score_only
                else "feature contract was frozen before this later query screen"
            ),
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-panel", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", required=True, help="comma-separated held YYYY-MM months")
    parser.add_argument("--query-modes", default=",".join(QUERY_MODES))
    parser.add_argument(
        "--preselection-core-score-only", action="store_true",
        help="use only the immutable enhanced-base score coordinates; required for chronology-safe query selection before G3",
    )
    args = parser.parse_args()
    if args.contract is None and not args.preselection_core_score_only:
        parser.error("--contract is required unless --preselection-core-score-only is set")
    run(
        history_panel=args.history_panel, contract_path=args.contract, policy_path=args.policy_path,
        out=args.out, months=_months(args.months),
        query_modes=tuple(token.strip() for token in args.query_modes.split(",") if token.strip()),
        preselection_core_score_only=args.preselection_core_score_only,
    )


if __name__ == "__main__":
    main()
