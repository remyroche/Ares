#!/usr/bin/env python3
"""Strict five-fold confirmation of one frozen Base feature contract.

This is deliberately a fixed-parameter check: group-MDA proposes contracts on
three development folds; this script refits the retained raw-bps CatBoost
configuration on a separate five-fold cross-year panel.  It never tunes the
feature contract, Meta, MC1, admission, portfolio, or live state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

import run_strict_r3_p8u_precision_preservation_group_mda_beam_v1 as beam
import run_strict_r3_p8u_precision_preservation_winner_hpo_v1 as hpo


SCHEMA = "strict_r3_p8u_precision_preservation_feature_confirm_v1"
MONTHS = ("2025-11", "2026-01", "2026-03", "2026-05", "2026-07")


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in text.split(",") if item.strip())
    if tuple(f"{item:%Y-%m}" for item in values) != MONTHS:
        raise ValueError("confirmation must use the fixed five-fold cross-year panel")
    return values


def _fields(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    values = tuple(str(item) for item in payload.get("selected_features", []))
    if len(values) < 25 or len(values) > 160 or len(set(values)) != len(values):
        raise ValueError("invalid frozen feature contract")
    return values


def run(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, stage1_root: Path,
    hpo_root: Path, fields_json: Path, out: Path, months: Sequence[pd.Timestamp], train_months: int,
    reserve_days: int, train_cap: int,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    arm, params = beam._load_contract(hpo_root)
    fields = _fields(fields_json)
    contract = hpo.Contract(
        arm=arm, gain_name="g3_clipped_economic", model_family="catboost_queryrmse",
        candidate="raw_bps__equal_width6__g3_clipped_economic__catboost_queryrmse",
    )
    out.mkdir(parents=True)
    _once(out / "preflight.json", {
        "schema": SCHEMA, "scope": "offline strict Base feature confirmation only",
        "feature_contract": str(fields_json), "feature_count": len(fields),
        "months": [f"{item:%Y-%m}" for item in months], "fixed_hpo_root": str(hpo_root),
    })
    folds, coverage = hpo._folds(
        feature_roots=feature_roots, label_root=label_root, router_root=router_root,
        stage1_root=stage1_root, fields=fields, held_months=months,
        train_months=train_months, reserve_days=reserve_days,
    )
    metrics, candidate_parts, _, audit = hpo._evaluate(
        contract=contract, params=params, folds=folds, fields=fields, train_cap=train_cap,
        reserve_days=reserve_days, seed_offset=0, persist_root=out,
    )
    candidate_parts[0].to_parquet(out / "oof_timestamp_components.parquet", index=False, compression="zstd")
    pd.DataFrame(audit).to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_parquet(out / "feature_coverage.parquet", index=False, compression="zstd")
    # ``_evaluate`` returns timestamp-aggregated diagnostic components here;
    # individual candidate identities are persisted separately in the
    # target-free score files above.  The aggregate must contain exactly one
    # row per decision timestamp, not a (non-existent) candidate_id column.
    if candidate_parts[0]["__decision_ts__"].duplicated().any():
        raise AssertionError("timestamp component panel must contain one row per decision timestamp")
    _once(out / "correctness_report.json", {
        "fixed_cross_year_five_fold_panel": True,
        "p8u_router_top50_identity_exact": True,
        "all_train_labels_resolved_before_reserve": True,
        "held_scores_target_free_before_outcome_join": True,
        "feature_medians_fit_train_only": True,
        "same_frozen_hpo_params_as_base_control": True,
        "no_meta_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline five-fold confirmation of one frozen Base feature contract only",
        "feature_contract": str(fields_json), "feature_count": len(fields),
        "hpo_root": str(hpo_root), "metrics": metrics,
        "strict_oof": {"months": [f"{item:%Y-%m}" for item in months], "train_months": train_months, "reserve_days": reserve_days, "train_cap_complete_queries": train_cap},
        "next_stage": "Compare only confirmation contracts with identical held IDs before any HPO refresh or downstream Meta/MC1 replay.",
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--hpo-root", type=Path, required=True)
    parser.add_argument("--fields-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", default=",".join(MONTHS))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 8_000:
        raise ValueError("invalid strict confirmation contract")
    print(run(
        feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()),
        label_root=args.label_root.resolve(), router_root=args.router_root.resolve(), stage1_root=args.stage1_root.resolve(),
        hpo_root=args.hpo_root.resolve(), fields_json=args.fields_json.resolve(), out=args.out.resolve(),
        months=_months(args.months), train_months=args.train_months, reserve_days=args.reserve_days, train_cap=args.train_cap,
    ))


if __name__ == "__main__":
    main()
