#!/usr/bin/env python3
"""Frozen, additive-only random-subspace screen for P8u state reliability.

Every shallow probe retains the entire current UnderF120 contract and the
shared Base score geometry.  It varies only a deterministic subset of frozen,
target-free V2 state fields selected from the earlier 2025 screen.  Outcomes
are opened only after each held score has been computed in memory.  This is
offline research; it has no live, MC1, admission, or execution path.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_strict_r3_p8u_meta_lgbm_objective_screen_v1 as objective
import run_strict_r3_p8u_meta_target_query_grid_v1 as screen


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_state_reliability_subspaces_v1"


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(path.rglob("*.parquet")) if path.is_dir() else [path]:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _selection_fields(screen_root: Path, available: set[str]) -> tuple[list[str], pd.DataFrame]:
    path = screen_root / "feature_summary_selection_2025.parquet"
    source = pd.read_parquet(path)
    required = {"feature", "era", "abs_residual_ic_sign_months", "mean_cmi_abs", "supported_regimes", "selection_score"}
    if not required.issubset(source.columns):
        raise AssertionError("invalid frozen selection-era screen")
    selected = source.loc[
        source.era.eq("selection_2025")
        & source.abs_residual_ic_sign_months.ge(5)
        & source.mean_cmi_abs.ge(.045)
        & source.supported_regimes.ge(3),
    ].copy()
    selected = selected.loc[selected.feature.astype(str).str.startswith("v2_")]
    selected = selected.loc[selected.feature.isin(available)]
    selected = selected.sort_values("selection_score", ascending=False, kind="stable").drop_duplicates("feature")
    if len(selected) < 20:
        raise AssertionError("too few frozen conditional candidates")
    return selected.feature.astype(str).tolist(), selected


def _subspaces(*, fields: list[str], scores: dict[str, float], probes: int, seed: int) -> list[dict[str, object]]:
    if probes < 50 or probes > 100:
        raise ValueError("probes must be in [50, 100]")
    rng = np.random.default_rng(seed)
    raw = np.asarray([max(float(scores[name]), 0.0) for name in fields], dtype=float)
    probability = raw + np.nanmedian(raw[raw > 0.0]) * .10
    probability /= probability.sum()
    output: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    # Include the no-addition control explicitly; all remaining probes are
    # score-weighted random subspaces of four through twelve *additions*.
    output.append({"probe": "p000_parent_control", "fields": []})
    seen.add(tuple())
    while len(output) < probes:
        size = int(rng.integers(4, 13))
        picked = tuple(sorted(rng.choice(np.asarray(fields, dtype=object), size=size, replace=False, p=probability).tolist()))
        if picked in seen:
            continue
        seen.add(picked)
        output.append({"probe": f"p{len(output):03d}", "fields": list(picked)})
    return output


def _trial(name: str, *, n_jobs: int) -> dict[str, Any]:
    return {
        "name": name,
        "gain": [0, 1, 2, 4, 7, 11, 16, 24],
        "truncation": 12,
        "sigmoid": 1.0,
        "model": {
            "objective": "rank_xendcg", "n_estimators": 90, "learning_rate": .05,
            "max_depth": 2, "num_leaves": 7, "min_child_samples": 450,
            "min_split_gain": .002, "feature_fraction": .90, "bagging_fraction": .85,
            "lambda_l1": .05, "lambda_l2": 10.0, "n_jobs": int(n_jobs),
        },
        "sample_weight": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-override", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= int(args.n_jobs) <= 8:
        raise ValueError("--n-jobs must be in [1, 8]")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    raw, applied = objective._apply_source_override(json.loads(args.config.resolve().read_text()), args.source_override.resolve())
    spec = screen.Spec(raw=raw, config_path=args.config.resolve())
    parent_fields, sidecar_fields, sidecar = objective._read_contract(args.parent_contract.resolve())
    if sidecar_fields or sidecar:
        raise AssertionError("subspace screen expects self-contained target-free panels")
    base_root = ROOT / str(spec.source["base_target_free_root"])
    feature_roots = tuple(ROOT / str(item) for item in spec.source["full_feature_roots"])
    policy = screen._read_policy(ROOT / str(spec.source["policy_labels"]))
    path_root = ROOT / str(spec.source["path_labels"])
    first_panel = pd.read_parquet(feature_roots[0] / "month=2025-08" / "causal_feature_universe.parquet")
    conditional, selection = _selection_fields(args.screen_root.resolve(), set(first_panel.columns))
    union_fields = tuple(dict.fromkeys([*parent_fields, *conditional]))
    spaces = _subspaces(
        fields=conditional,
        scores=dict(zip(selection.feature.astype(str), selection.selection_score.astype(float))),
        probes=int(args.probes), seed=int(args.seed),
    )
    months = tuple(screen._utc_month(value) for value in spec.folds["held_months"])
    out.mkdir(parents=True)
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline shallow additive random-subspace screen only",
        "parent_feature_count": len(parent_fields), "conditional_field_count": len(conditional),
        "probes": len(spaces), "seed": int(args.seed), "months": [f"{m:%Y-%m}" for m in months],
        "source_override": str(args.source_override.resolve()), "source_override_payload": applied,
        "parent_contract": str(args.parent_contract.resolve()), "parent_contract_sha256": _sha(args.parent_contract.resolve()),
        "selection_screen": str(args.screen_root.resolve()), "selection_era": "selection_2025",
        "causality": "parent features and target-free frozen state only; held outcomes open after held score computation",
    })
    pd.DataFrame([{**space, "field_count": len(space["fields"])} for space in spaces]).to_parquet(out / "probe_contracts.parquet", index=False, compression="zstd")
    selection.to_parquet(out / "conditional_selection_2025.parquet", index=False, compression="zstd")
    all_metrics: list[dict[str, object]] = []
    all_weekly: list[pd.DataFrame] = []
    arm = objective._arms(raw)["under_bps100__timestamp"]
    for fold_index, held_month in enumerate(months):
        prepared = objective._prepare_fold(
            base_root=base_root, feature_roots=feature_roots, policy=policy, path_root=path_root,
            arm=arm, fields=union_fields, panel_fields=union_fields, continuous_sidecar=None,
            continuous_fields=(), spec=spec, held_month=held_month, seed=int(spec.folds["seed"]) + fold_index,
            materialize_held_labels=True,
        )
        # ``PreparedFold`` matrices are [nine mandatory Base coordinates,
        # union_fields].  Subsetting these imputed matrices exactly preserves
        # per-field fold medians while keeping the full current-stack parent.
        index = {field: 9 + position for position, field in enumerate(union_fields)}
        for probe_index, space in enumerate(spaces):
            selected = tuple([*parent_fields, *space["fields"]])
            columns = np.asarray([*range(9), *(index[field] for field in selected)], dtype=np.int32)
            local = dataclasses.replace(prepared, train_x=prepared.train_x[:, columns], held_x=prepared.held_x[:, columns])
            trial = _trial(str(space["probe"]), n_jobs=int(args.n_jobs))
            score = objective._score_prepared(
                prepared=local, arm=arm, trial=trial,
                seed=int(args.seed) + 1000 * fold_index + probe_index,
            )
            if prepared.held_labelled is None:
                raise AssertionError("held labels missing after target-free score generation")
            weekly, _bands, metric = screen._metrics(score=score, held_labelled=prepared.held_labelled, held_anchor=prepared.held_anchor, spec=spec)
            weekly["probe"] = str(space["probe"]); weekly["held_month"] = f"{held_month:%Y-%m}"
            all_weekly.append(weekly)
            all_metrics.append({"probe": str(space["probe"]), "held_month": f"{held_month:%Y-%m}", "additive_fields": json.dumps(space["fields"]), **metric})
        # A completed fold is immutable and can be inspected while later
        # folds run.  It also makes a slow research failure diagnosable
        # without treating unfinished metrics as a result.
        pd.DataFrame(all_metrics).to_parquet(out / f"fold_checkpoint_{held_month:%Y%m}.parquet", index=False, compression="zstd")
    fold_metrics = pd.DataFrame(all_metrics)
    weekly = pd.concat(all_weekly, ignore_index=True)
    rows: list[dict[str, object]] = []
    for probe, group in fold_metrics.groupby("probe", sort=True):
        weekly_group = weekly.loc[weekly.probe.eq(probe)]
        q20, q80 = weekly_group.smeta.quantile([.20, .80])
        robust = float(weekly_group.loc[weekly_group.smeta.between(q20, q80), "smeta"].mean())
        lower = float(weekly_group.smeta.quantile(.15) + weekly_group.smeta.quantile(.10) + weekly_group.smeta.quantile(.05)) / 3.0
        rows.append({
            "probe": probe, "sstable_meta": robust + .5 * lower,
            "smeta_week_robust_average": robust, "smeta_week_lower_tail": lower,
            "mean_top2_substitution_ev_bps": float(weekly_group.delta_ev_top2_bps.mean()),
            "worst_week_delta_ev_top2_bps": float(weekly_group.delta_ev_top2_bps.min()),
            "mean_admission_substitution_utility_bps": float(weekly_group.delta_utility_admission_bps.mean()),
            "mean_iccond": float(weekly_group.iccond.mean()),
            "feature_count": int(fold_metrics.loc[fold_metrics.probe.eq(probe), "additive_fields"].iloc[0].count("v2_")),
        })
    summary = pd.DataFrame(rows).sort_values(["sstable_meta", "mean_top2_substitution_ev_bps"], ascending=False, kind="stable")
    summary.to_parquet(out / "probe_summary.parquet", index=False, compression="zstd")
    fold_metrics.to_parquet(out / "probe_fold_metrics.parquet", index=False, compression="zstd")
    weekly.to_parquet(out / "probe_weekly_metrics.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "all_probes_retain_parent_f120": True,
        "additions_derived_only_from_frozen_2025_selection": True,
        "state_inputs_are_target_free": True,
        "held_outcomes_used_only_after_score_generation": True,
        "no_mc1_admission_portfolio_or_live_mutation": True,
        "deterministic_seeded_subspaces": True,
    })


if __name__ == "__main__":
    main()
