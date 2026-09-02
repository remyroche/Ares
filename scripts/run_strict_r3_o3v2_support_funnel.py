#!/usr/bin/env python3
"""Training-only support/weight screen for the retained strict-R3 O3-v2 targets.

This is deliberately a separate immutable research stage.  It re-fits only
the target-screen-retained correction concepts while varying bounded
*training* weights.  Semantic/path fields are never present in persisted held
score panels and this module neither calls nor changes MC1, the canonical
stack, or live inference.
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
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402
import run_strict_r3_o3v2_target_funnel as target  # noqa: E402


SCHEMA = "strict_r3_o3v2_support_funnel_v4"
SEED = 1729
TRAIN_MONTHS = 6
RESERVE_DAYS = 28
DEFAULT_DEVELOPMENT_MONTHS = tuple(pd.date_range("2025-11-01", "2026-01-01", freq="MS", tz="UTC"))
TARGET_ARMS = (
    "T3_pair_residual_lambdarank",
    "T6_rank_error_ordinal",
)
SUPPORT_ARMS = (
    "S1_archetype_balance",
    "S2_label_certainty",
    "S4_hard_base_error",
    "S5_policy_robustness",
    "SB1_archetype_certainty",
    "SB2_archetype_certainty_error",
    "SB3_full_bounded",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*.parquet")) if path.is_dir() else (path,):
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    """Persist an immutable receipt without silently replacing evidence."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _normalise(weight: np.ndarray) -> np.ndarray:
    weight = np.asarray(weight, dtype=float)
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise AssertionError("support weight must be finite and positive")
    weight /= max(float(np.mean(weight)), 1e-12)
    return np.clip(weight, 0.25, 4.0).astype(np.float32)


def _components(train: pd.DataFrame) -> dict[str, np.ndarray]:
    archetype = train["semantic_archetype"].astype("string").fillna("invalid")
    counts = archetype.value_counts(dropna=False)
    archetype_weight = archetype.map(lambda value: np.sqrt(len(train) / max(float(counts.loc[value]), 1.0))).to_numpy(float)

    event = train["semantic_tbm_event"].astype("string")
    certainty = np.where(event.eq("ambiguous"), 0.50, np.where(event.eq("vertical"), 0.80, 1.00)).astype(float)

    policy = pd.to_numeric(train["semantic_policy_net_bps"], errors="coerce")
    realised_rank = policy.groupby(train["__decision_ts__"], sort=False).rank(pct=True, method="average").to_numpy(float)
    base_rank = pd.to_numeric(train["base_rank_ts"], errors="coerce").to_numpy(float)
    error = np.abs(realised_rank - base_rank)
    hard_error = 1.0 + np.where(error >= 0.25, 0.75, np.where(error >= 0.10, 0.25, 0.0))

    robustness = train["semantic_axis_k_policy_robustness"].astype("string")
    policy_robustness = np.where(robustness.eq("mixed_robustness"), 1.00, 0.75).astype(float)
    return {
        "archetype": _normalise(archetype_weight),
        "certainty": _normalise(certainty),
        "hard_base_error": _normalise(hard_error),
        "policy_robustness": _normalise(policy_robustness),
    }


def _weights(train: pd.DataFrame, arm: str) -> np.ndarray:
    comp = _components(train)
    if arm == "S1_archetype_balance":
        raw = comp["archetype"]
    elif arm == "S2_label_certainty":
        raw = comp["certainty"]
    elif arm == "S4_hard_base_error":
        raw = comp["hard_base_error"]
    elif arm == "S5_policy_robustness":
        raw = comp["policy_robustness"]
    elif arm == "SB1_archetype_certainty":
        raw = comp["archetype"] * comp["certainty"]
    elif arm == "SB2_archetype_certainty_error":
        raw = comp["archetype"] * comp["certainty"] * comp["hard_base_error"]
    elif arm == "SB3_full_bounded":
        raw = comp["archetype"] * comp["certainty"] * comp["hard_base_error"] * comp["policy_robustness"]
    else:
        raise ValueError(f"unsupported support arm: {arm}")
    return _normalise(raw)


def _sample_weights(sampled: pd.DataFrame, full_train: pd.DataFrame, full_weights: np.ndarray, spec: object) -> np.ndarray:
    by_id = pd.Series(np.asarray(full_weights, dtype=np.float32), index=full_train["candidate_id"].astype(str))
    weights = by_id.loc[sampled["candidate_id"].astype(str)].to_numpy(dtype=float)
    if spec.weight_mode == "equal_month":
        counts = sampled["__month__"].value_counts()
        weights *= sampled["__month__"].map(lambda token: 1.0 / float(counts.loc[token])).to_numpy(float)
    elif spec.weight_mode != "ordinary":
        raise ValueError(f"unknown frozen weight mode: {spec.weight_mode}")
    return _normalise(weights)


def _label_specs_for_target(specs: Sequence[object], grade: np.ndarray | None) -> tuple[object, ...]:
    """Give weighted LambdaRank the same declared gain support as its parent.

    The target funnel expands the physical five-head template to the number of
    classes in the selected target.  Support weighting changes only training
    weights, never label semantics, so it must reuse that exact conversion.
    In particular, T1's grades are 0..6 and cannot be fitted with the
    incumbent five-entry gain vector.
    """
    return target._label_specs(specs, grade)


def _fit_weighted_heads(
    train: pd.DataFrame,
    target_value: np.ndarray,
    specs: Sequence[object],
    *,
    objective: str,
    full_weights: np.ndarray,
    grade: np.ndarray | None,
    pairwise_mode: str = "none",
    n_jobs: int = 1,
) -> tuple[object, ...]:
    identity_columns = ["candidate_id", "__decision_ts__", "side_name"]
    if any(spec.query == "exact_timestamp_baseband_side" for spec in specs):
        identity_columns.append("base_rank_ts")
    identity = train.loc[:, identity_columns].copy().reset_index(drop=True)
    by_id = train.set_index("candidate_id", drop=False)
    target_by_id = pd.Series(np.asarray(target_value, dtype=np.float32), index=train["candidate_id"].astype(str))
    sampling_grade = (
        np.asarray(grade, dtype=np.int32)
        if grade is not None
        else parent._residual_grade(target_value, (-100.0, -30.0, 30.0, 90.0))
    )
    heads = []
    for index, spec in enumerate(specs):
        if pairwise_mode == "none":
            sampled_identity, _grades, groups = parent._sample_complete_consensus_queries(
                identity, sampling_grade, spec, seed=SEED + 1000 + index,
            )
            sampled = by_id.loc[
                sampled_identity["candidate_id"].to_numpy(),
                ["candidate_id", "__decision_ts__", "side_name", *spec.fields],
            ].copy()
            sampled["__query__"] = sampled_identity["__query__"].to_numpy()
            sampled["__month__"] = sampled_identity["__month__"].to_numpy()
            pair_target = None
        else:
            # The shared pair sampler is deliberately defined on the canonical
            # realised policy-outcome name.  This support runner retains the
            # semantic label name elsewhere so it cannot become an inference
            # input; expose the alias only inside the train-only sampler.
            # The declared T3 target is the train-only calibrated residual.
            # The common pair sampler reads ``policy_net_bps`` as its pair
            # quantity, so expose the residual under that private alias only
            # inside this resolved training fold.  Raw policy net is never
            # substituted here in schema-v3 support research.
            pair_train = train.rename(columns={"semantic_policy_net_bps": "policy_net_bps"}).copy()
            pair_train["policy_net_bps"] = np.asarray(target_value, dtype=np.float32)
            sampled, pair_target, groups = parent._sample_base_near_tie_pairs(
                pair_train,
                spec,
                pairwise_mode,
                seed=SEED + 1000 + index,
            )
        weights = _sample_weights(sampled, train, full_weights, spec)
        medians = parent._fit_medians(sampled, spec.fields)
        if pairwise_mode != "none":
            if objective != "ordinal_lambdarank" or pair_target is None:
                raise ValueError(f"pairwise support requires LambdaRank, got {objective}")
            params = dict(spec.params)
            params.update(random_state=SEED + 1000 + index, n_jobs=n_jobs, deterministic=True, force_col_wise=True)
            model = parent.LGBMRanker(**params).fit(
                parent._numeric_matrix(sampled, spec.fields, medians), np.asarray(pair_target, dtype=np.int32),
                group=groups, sample_weight=weights,
            )
        else:
            continuous = target_by_id.loc[sampled["candidate_id"].astype(str)].to_numpy(dtype=np.float32)
            if objective == "l2_regression":
                ignored = {"objective", "metric", "label_gain", "lambdarank_truncation_level"}
                params = {key: value for key, value in spec.params.items() if key not in ignored}
                params.update(objective="regression_l2", metric="l2", random_state=SEED + 1000 + index, n_jobs=n_jobs, deterministic=True, force_col_wise=True)
                model = parent.LGBMRegressor(**params).fit(
                    parent._numeric_matrix(sampled, spec.fields, medians), continuous, sample_weight=weights,
                )
            elif objective == "ordinal_lambdarank":
                if grade is None:
                    raise AssertionError("ordinal LambdaRank support requires a declared grade")
                grade_by_id = pd.Series(np.asarray(grade, dtype=np.int32), index=train["candidate_id"].astype(str))
                labels = grade_by_id.loc[sampled["candidate_id"].astype(str)].to_numpy(dtype=np.int32)
                params = dict(spec.params)
                params.update(random_state=SEED + 1000 + index, n_jobs=n_jobs, deterministic=True, force_col_wise=True)
                model = parent.LGBMRanker(**params).fit(
                    parent._numeric_matrix(sampled, spec.fields, medians), labels,
                    group=groups, sample_weight=weights,
                )
            else:
                raise ValueError(f"unsupported support objective: {objective}")
        raw = model.predict(parent._numeric_matrix(sampled, spec.fields, medians))
        heads.append(parent.FittedConsensusHead(
            spec, medians, model,
            parent.ScoreReference.fit(raw, source=f"{spec.name}_o3v2_{objective}_weighted_training_distribution"),
        ))
    return tuple(heads)


def run(*, feature_root: Path, semantic_root: Path, policy_path: Path, bundle_root: Path, out: Path, months: Sequence[pd.Timestamp], target_arms: Sequence[str], support_arms: Sequence[str], pairs: Sequence[tuple[str, str]] | None = None, query_mode: str = "exact_timestamp_side", physical_slot_selection: Path | None = None, resume: bool = False, n_jobs: int = 1) -> None:
    selected_pairs = tuple(pairs) if pairs is not None else tuple(
        (target_arm, support_arm) for target_arm in target_arms for support_arm in support_arms
    )
    if not selected_pairs:
        raise ValueError("at least one target/support pair is required")
    contract = {
        "schema": SCHEMA, "feature_root": str(feature_root), "semantic_root": str(semantic_root),
        "policy_path": str(policy_path), "bundle_root": str(bundle_root),
        "months": [f"{month:%Y-%m}" for month in months],
        "target_support_pairs": [{"target_arm": target_arm, "support_arm": support_arm} for target_arm, support_arm in selected_pairs],
        "query_mode": query_mode,
        "physical_slot_selection": str(physical_slot_selection) if physical_slot_selection else None,
        "physical_slot_selection_sha256": _sha256(physical_slot_selection) if physical_slot_selection else None,
        "lightgbm_n_jobs": n_jobs,
    }
    contract_path = out / "run_contract.json"
    if out.exists():
        if not resume:
            raise FileExistsError(out)
        if not contract_path.exists() or json.loads(contract_path.read_text()) != contract:
            raise AssertionError("support resume contract differs from immutable existing receipts")
    else:
        out.mkdir(parents=True)
        _write_json_exclusive(contract_path, contract)
    policy = pd.read_parquet(policy_path, columns=("candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"))
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy ledger contains duplicate identities")
    paths = parent.Paths(*(Path("unused") for _ in range(5)), bundle_root)
    base_fields = parent._base_fields(paths)
    physical_slots = target._load_physical_slot_contract(
        physical_slot_selection, target_arms, query_mode=query_mode,
    )
    if physical_slots is None:
        raise ValueError("post-selector support research requires --physical-slot-selection")
    all_specs = target._specs(base_fields, query_mode=query_mode)
    specs_by_target = {
        arm: tuple(spec for spec in all_specs if spec.name == physical_slots[arm])
        for arm in target_arms
    }
    if any(len(specs_by_target[arm]) != 1 for arm in target_arms):
        raise AssertionError("support physical-slot contract did not resolve one head per target")
    source_columns = tuple(dict.fromkeys((
        "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed", "enhanced_base_bps", "base_rank_ts",
        "base_bps", "efficiency_bps", "timing_bps", "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std", *base_fields,
    )))
    audit_root = out / "audit_parts"
    audit_root.mkdir(exist_ok=True)
    for target_arm, support_arm in selected_pairs:
        arm = f"{target_arm}__{support_arm}"
        root = out / "target_free_scores" / arm
        root.mkdir(parents=True, exist_ok=True)
        for month in months:
                score_path = root / f"month={month:%Y-%m}.parquet"
                audit_path = audit_root / f"{arm}__{month:%Y-%m}.json"
                if score_path.exists() and audit_path.exists():
                    continue
                if score_path.exists() != audit_path.exists():
                    raise AssertionError(f"partial immutable support receipt: {arm} {month:%Y-%m}")
                end = target._month_end(month)
                reserve_start = month - pd.Timedelta(days=RESERVE_DAYS)
                train_start = target._strict_train_start(month)
                train = target._training_window(feature_root, semantic_root, start=train_start, end=reserve_start, columns=source_columns)
                held = target._window_features(feature_root, month, end, source_columns)
                # The persisted historical flag predates the exact routing
                # repair and can admit one extra tied row at a timestamp.
                # Reconstruct the route from the target-free upstream score
                # for both fit and held panels before any geometry or label
                # operation, so this stage has the same top-30% contract as
                # live inference and the target funnel.
                train["enhanced_base_routed"] = parent._exact_timestamp_top_fraction(
                    train, "enhanced_base_bps", parent.BASE_ROUTE,
                ).to_numpy(bool)
                held["enhanced_base_routed"] = parent._exact_timestamp_top_fraction(
                    held, "enhanced_base_bps", parent.BASE_ROUTE,
                ).to_numpy(bool)
                train = target._base_geometry(train)
                held = target._base_geometry(held)
                valid = (
                    train["enhanced_base_routed"].fillna(False).astype(bool)
                    & train["semantic_path_valid"].fillna(False).astype(bool)
                    & train["semantic_label_available_ts"].lt(reserve_start)
                    & np.isfinite(pd.to_numeric(train["semantic_policy_net_bps"], errors="coerce"))
                )
                train = train.loc[valid].copy()
                held = held.loc[held["enhanced_base_routed"].fillna(False).astype(bool)].copy()
                if len(train) < 5_000 or len(held) < 1_000:
                    raise AssertionError(f"{arm} {month:%Y-%m}: insufficient strict support")
                target_value, grade, objective, mode = target._anchor_and_targets(train, target_arm)
                full_weights = _weights(train, support_arm)
                pairwise_mode = {
                    "pair_residual": "near_tie_diff50",
                    "hard_inversion": "base_inversion100",
                }.get(mode, "none")
                fit_specs = _label_specs_for_target(specs_by_target[target_arm], grade)
                heads = _fit_weighted_heads(
                    train, target_value, fit_specs, objective=objective,
                    full_weights=full_weights, grade=grade,
                    pairwise_mode=pairwise_mode, n_jobs=n_jobs,
                )
                scored = target._score_columns(parent._score_heads(held, heads))
                leaked = target.PROHIBITED_SCORE_COLUMNS.intersection(scored.columns)
                if leaked:
                    raise AssertionError(f"{arm} held score leaked training semantics: {sorted(leaked)}")
                scored.to_parquet(score_path, index=False, compression="zstd")
                audit_row = {
                    "target_arm": target_arm, "support_arm": support_arm, "month": f"{month:%Y-%m}",
                    "train_start": str(train_start), "reserve_start": str(reserve_start),
                    "train_rows": int(len(train)), "held_rows": int(len(held)), "objective": objective,
                    "weight_min": float(np.min(full_weights)), "weight_max": float(np.max(full_weights)),
                    "weight_mean": float(np.mean(full_weights)), "semantic_valid_fraction": float(train["semantic_path_valid"].mean()),
                    "physical_slot": physical_slots[target_arm],
                }
                _write_json_exclusive(audit_path, audit_row)
                print(json.dumps({"event": "scored", **audit_row}), flush=True)
    metrics: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for target_arm, support_arm in selected_pairs:
        arm = f"{target_arm}__{support_arm}"
        for month in months:
            score_path = out / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"
            audit_path = audit_root / f"{arm}__{month:%Y-%m}.json"
            if not score_path.exists() or not audit_path.exists():
                raise AssertionError(f"cannot finalise missing support receipt: {arm} {month:%Y-%m}")
            metrics.extend(target._metric_rows(pd.read_parquet(score_path), policy, arm=arm, month=month))
            audit.append(json.loads(audit_path.read_text()))
    metrics_path = out / "support_funnel_metrics.parquet"
    audit_path = out / "support_funnel_audit.parquet"
    if not metrics_path.exists():
        pd.DataFrame(metrics).to_parquet(metrics_path, index=False, compression="zstd")
    if not audit_path.exists():
        pd.DataFrame(audit).to_parquet(audit_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline O3-v2 support/weight research only; no live/canonical/MC1 mutation",
        "months": [f"{month:%Y-%m}" for month in months], "target_arms": list(target_arms), "support_arms": list(support_arms),
        "target_support_pairs": [{"target_arm": target_arm, "support_arm": support_arm} for target_arm, support_arm in selected_pairs],
        "feature_root": str(feature_root), "semantic_root": str(semantic_root), "policy_path": str(policy_path), "bundle_root": str(bundle_root), "lightgbm_n_jobs": n_jobs,
        "physical_slot_selection": str(physical_slot_selection) if physical_slot_selection else None,
        "physical_slot_selection_sha256": _sha256(physical_slot_selection) if physical_slot_selection else None,
        "selected_physical_slots": physical_slots,
        "source_hashes": {"feature_root": _sha256(feature_root), "semantic_root": _sha256(semantic_root), "policy_path": _sha256(policy_path)},
        "causality": {"fit": "six full preceding resolved calendar months before reserve", "reserve": "28 days excluded from fitting", "weights": "derived only from resolved pre-reserve path/policy labels", "held_scores": "target-free score receipts persisted before canonical policy outcome joins", "query": query_mode, "pair_residual": "T3 pairs train-only calibrated residual, never raw policy net", "inference": "support labels never enter held score panels or inference"},
    }
    manifest_path = out / "run_manifest.json"
    if not manifest_path.exists():
        _write_json_exclusive(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", help="comma-separated YYYY-MM; defaults to 2025-10..2025-12 development screen")
    parser.add_argument("--target-arms", help="comma-separated retained target concepts")
    parser.add_argument("--support-arms", help="comma-separated support/weight variants")
    parser.add_argument("--pairs", help="optional comma-separated target:support pairs; avoids a cross-product")
    parser.add_argument("--query-mode", default="exact_timestamp_side", choices=("exact_timestamp_side", "exact_timestamp_baseband_side", "cycle_4h_side"))
    parser.add_argument("--physical-slot-selection", type=Path, required=True, help="sealed one-physical-head-per-target contract")
    parser.add_argument("--resume", action="store_true", help="resume only an identical immutable receipt contract")
    parser.add_argument("--n-jobs", type=int, default=1, help="deterministic LightGBM worker count; completed receipts are never refit")
    args = parser.parse_args()
    months = tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in args.months.split(",")) if args.months else DEFAULT_DEVELOPMENT_MONTHS
    target_arms = tuple(args.target_arms.split(",")) if args.target_arms else TARGET_ARMS
    support_arms = tuple(args.support_arms.split(",")) if args.support_arms else SUPPORT_ARMS
    supported_targets = set(target.ALL_ARMS) - {"T0_current_o3_control"}
    if set(target_arms) - supported_targets:
        raise ValueError(f"unsupported retained targets: {sorted(set(target_arms) - supported_targets)}")
    if set(support_arms) - set(SUPPORT_ARMS):
        raise ValueError(f"unsupported support arms: {sorted(set(support_arms) - set(SUPPORT_ARMS))}")
    if args.n_jobs <= 0 or args.n_jobs > 8:
        parser.error("--n-jobs must be between 1 and 8")
    pairs = None
    if args.pairs:
        pairs = tuple(tuple(token.split(":", 1)) for token in args.pairs.split(","))
        if any(len(pair) != 2 or not pair[0] or not pair[1] for pair in pairs):
            raise ValueError("--pairs must contain target:support entries")
        invalid_target = sorted({pair[0] for pair in pairs} - supported_targets)
        invalid_support = sorted({pair[1] for pair in pairs} - set(SUPPORT_ARMS))
        if invalid_target or invalid_support:
            raise ValueError(f"unsupported pair values: targets={invalid_target}, supports={invalid_support}")
        target_arms = tuple(dict.fromkeys(pair[0] for pair in pairs))
        support_arms = tuple(dict.fromkeys(pair[1] for pair in pairs))
    run(feature_root=args.feature_root, semantic_root=args.semantic_root, policy_path=args.policy_path, bundle_root=args.bundle_root, out=args.out, months=months, target_arms=target_arms, support_arms=support_arms, pairs=pairs, query_mode=args.query_mode, physical_slot_selection=args.physical_slot_selection, resume=args.resume, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
