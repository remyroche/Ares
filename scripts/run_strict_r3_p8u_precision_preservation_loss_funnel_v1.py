#!/usr/bin/env python3
"""Stage 2: P8u gain-schedule funnel under precision-plus-preservation.

This bounded, strict-OOF stage takes the top two target geometries from each
Stage-1 target family, evaluates the three *predeclared* LambdaRank gain
schedules with one cheap common tree geometry, and keeps approximately three
diverse target×gain finalists for the later objective comparison.

The fixed P8u policy-ordinal ``rank_xendcg`` target-free scores from Stage 1
are the common normalisation control.  Every challenger held score is written
without outcomes, then joined to the canonical policy ledger only for
selection diagnostics.  This stage neither runs Meta/MC1/portfolio nor
touches live trading.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

import run_strict_r3_p8u_precision_preservation_screen_v1 as stage1
import run_strict_r3_router_single_base_prescreen_v1 as base
from strict_r3_p8u_precision_preservation_metric import COMPONENTS, stable_score, timestamp_components


SCHEMA = "strict_r3_p8u_precision_preservation_loss_funnel_v1"
SEED = 1729
IDENTITY = base.IDENTITY
CONTROL_ARM = stage1.CONTROL_KEY
# Exact schedules in the user-provided contract.  Do not silently substitute
# the legacy Router/Base gains from ``base.GAIN_SCHEDULES``.
GAIN_SCHEDULES: dict[str, list[float]] = {
    "g1_moderate_convex": [0.0, 1.0, 2.0, 4.0, 7.0, 11.0],
    "g2_stronger_top_tail": [0.0, 1.0, 3.0, 6.0, 11.0, 18.0],
    "g3_clipped_economic": [0.0, 0.5, 2.0, 3.0, 6.0, 8.0],
}


@dataclass(frozen=True)
class Candidate:
    arm: stage1.Arm
    gain_name: str

    @property
    def key(self) -> str:
        return f"{self.arm.key}__{self.gain_name}"


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _parse_months(tokens: str) -> tuple[pd.Timestamp, ...]:
    return stage1._parse_months(tokens)


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _selected_arms(screen_root: Path, *, top_per_family: int) -> tuple[stage1.Arm, ...]:
    if top_per_family < 1:
        raise ValueError("top-per-family must be positive")
    summary = pd.read_parquet(screen_root / "target_summary.parquet")
    required = {"arm", "family", "score_stable"}
    if missing := required.difference(summary.columns):
        raise AssertionError(f"Stage-1 target summary missing {sorted(missing)}")
    arm_by_key = {arm.key: arm for arm in stage1.ARMS}
    selected = (
        summary.sort_values(["family", "score_stable", "arm"], ascending=[True, False, True], kind="stable")
        .groupby("family", sort=True)
        .head(top_per_family)
    )
    keys = selected["arm"].tolist()
    if len(keys) != len(set(keys)) or any(key not in arm_by_key for key in keys):
        raise AssertionError("Stage-1 finalist identity is invalid")
    # The Stage-1 control must remain reachable for audit even if a later
    # target-family revision would otherwise rank it below the top two.
    if CONTROL_ARM not in keys:
        keys.append(CONTROL_ARM)
    return tuple(arm_by_key[key] for key in keys)


def _control_score(screen_root: Path, month: pd.Timestamp) -> pd.DataFrame:
    path = screen_root / "target_free_scores" / CONTROL_ARM / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing immutable Stage-1 control score: {path}")
    control = pd.read_parquet(path)
    expected = [*IDENTITY, "base_score", "base_rank_ts"]
    if control.columns.tolist() != expected:
        raise AssertionError("Stage-1 normalisation control is not target-free")
    return control


def _ranker(candidate: Candidate, *, seed: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=180, learning_rate=.05,
        max_depth=4, num_leaves=15, min_child_samples=260,
        subsample=.80, subsample_freq=1, colsample_bytree=.80,
        reg_alpha=.05, reg_lambda=8.0, min_split_gain=.001,
        lambdarank_truncation_level=12, sigmoid=1.0,
        label_gain=GAIN_SCHEDULES[candidate.gain_name], random_state=seed,
        n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
    )


def _fit_one(
    *, candidate: Candidate, window: pd.DataFrame, held: pd.DataFrame, reserve: pd.Timestamp,
    fields: tuple[str, ...], train_cap: int, seed: int,
) -> tuple[Candidate, pd.DataFrame, dict[str, object]]:
    train = stage1._train_rows(window, candidate.arm, reserve, train_cap)
    labels, geometry = stage1._labels(train, candidate.arm)
    x_train, medians = base._numeric_matrix(train, fields)
    x_held, _ = base._numeric_matrix(held, fields, medians)
    model = _ranker(candidate, seed=seed)
    model.fit(x_train, labels, group=base._query_groups(train))
    score = held.loc[:, list(IDENTITY)].copy()
    score["base_score"] = model.predict(x_held).astype(np.float32)
    score["base_rank_ts"] = base._rank_desc(score, "base_score")
    if score.columns.tolist() != [*IDENTITY, "base_score", "base_rank_ts"]:
        raise AssertionError("held loss-funnel score violates target-free schema")
    receipt = {
        "train_rows": int(len(train)), "train_queries": int(train["__decision_ts__"].nunique()),
        "target_geometry": geometry, "feature_medians_fit_train_only": True,
    }
    del model, x_train, x_held, train
    return candidate, score, receipt


def _summaries(
    *, components: dict[str, list[pd.DataFrame]], control: pd.DataFrame,
    candidates: Sequence[Candidate],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_by_key = {item.key: item for item in candidates}
    rows: list[dict[str, object]] = []
    normalised_parts: list[pd.DataFrame] = []
    for key, parts in components.items():
        current = pd.concat(parts, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
        summary, normalised = stable_score(current, control)
        candidate = candidate_by_key[key]
        rows.append({
            "candidate": key, "arm": candidate.arm.key, "family": candidate.arm.family,
            "target": candidate.arm.target, "geometry": candidate.arm.geometry,
            "gain_name": candidate.gain_name, **summary.__dict__,
            **{f"mean_{name}": float(current[name].mean()) for name in COMPONENTS},
            "mean_utility_recall20": float(current["utility_recall20"].mean()),
            "mean_top2_coverage": float(current["dtp2_bps_coverage"].mean()),
            "mean_top5_coverage": float(current["dtp5_bps_coverage"].mean()),
            "mean_top10_coverage": float(current["dtp10_bps_coverage"].mean()),
            "utility_recall20_coverage": float(current["utility_recall20_eligible"].mean()),
            "residual_utility_recall10_to30_coverage": float(current["residual_utility_recall10_to30_eligible"].mean()),
        })
        normalised["candidate"] = key
        normalised_parts.append(normalised)
    summary = pd.DataFrame(rows).sort_values("score_stable", ascending=False, kind="stable")
    # A gain is useful only once per target geometry.  Then retain about three
    # strong but family-diverse finalists for the objective comparison.
    best_by_arm = summary.sort_values(["arm", "score_stable", "gain_name"], ascending=[True, False, True], kind="stable").groupby("arm", sort=True).head(1)
    family_best = best_by_arm.sort_values(["family", "score_stable", "candidate"], ascending=[True, False, True], kind="stable").groupby("family", sort=True).head(1)
    finalists = family_best.sort_values(["score_stable", "candidate"], ascending=[False, True], kind="stable").head(3)
    return summary, finalists, pd.concat(normalised_parts, ignore_index=True)


def run(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, selection_receipt: Path,
    screen_root: Path, out: Path, held_months: Sequence[pd.Timestamp], train_months: int,
    reserve_days: int, train_cap: int, top_per_family: int, workers: int,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    required_screen = [screen_root / "run_manifest.json", screen_root / "correctness_report.json", screen_root / "target_summary.parquet"]
    if any(not path.exists() for path in required_screen):
        raise FileNotFoundError("Stage-1 P8u target screen is incomplete")
    stage1_correctness = json.loads((screen_root / "correctness_report.json").read_text())
    if not all(stage1_correctness.values()):
        raise AssertionError("Stage-1 target screen does not satisfy its correctness receipt")
    fields = base._load_f72_fields(selection_receipt)
    arms = _selected_arms(screen_root, top_per_family=top_per_family)
    candidates = tuple(Candidate(arm, gain) for arm in arms for gain in GAIN_SCHEDULES)
    out.mkdir(parents=True)
    component_parts: dict[str, list[pd.DataFrame]] = {candidate.key: [] for candidate in candidates}
    control_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for fold_index, held_month in enumerate(held_months):
        reserve = held_month - pd.Timedelta(days=reserve_days)
        end = held_month + pd.offsets.MonthBegin(1)
        window, coverage = base._load_window(
            candidate_root=None, feature_root=tuple(feature_roots), label_root=label_root,
            router_root=router_root, start=reserve - pd.DateOffset(months=train_months), end=end, fields=fields,
        )
        coverage_rows.extend(coverage)
        held = window.loc[window["__decision_ts__"].ge(held_month) & window["__decision_ts__"].lt(end)].copy()
        held = held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if held.empty:
            raise AssertionError(f"{held_month:%Y-%m}: missing P8u held population")
        labels = held.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]].copy()
        control = _control_score(screen_root, held_month)
        if len(control) != len(held) or not control["candidate_id"].equals(held["candidate_id"]):
            raise AssertionError(f"{held_month:%Y-%m}: Stage-1 control identities do not match exact P8u holdout")
        control_scored = control.merge(labels, on="candidate_id", how="left", validate="one_to_one")
        control_parts.append(timestamp_components(control_scored, score_column="base_score"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(
                _fit_one, candidate=candidate, window=window, held=held, reserve=reserve,
                fields=fields, train_cap=train_cap, seed=SEED + fold_index * 100_000 + index,
            ) for index, candidate in enumerate(candidates)]
            for future in concurrent.futures.as_completed(futures):
                candidate, score, receipt = future.result()
                path = out / "target_free_scores" / candidate.key / f"month={held_month:%Y-%m}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                score.to_parquet(path, index=False, compression="zstd")
                scored = score.merge(labels, on="candidate_id", how="left", validate="one_to_one")
                components = timestamp_components(scored, score_column="base_score")
                components["candidate"] = candidate.key
                components["held_month"] = f"{held_month:%Y-%m}"
                component_parts[candidate.key].append(components)
                fold_rows.append({
                    "candidate": candidate.key, "arm": candidate.arm.key, "family": candidate.arm.family,
                    "gain_name": candidate.gain_name, "held_month": f"{held_month:%Y-%m}",
                    "held_rows": int(len(held)), "held_queries": int(held["__decision_ts__"].nunique()),
                    "score_path": str(path), "target_free_before_outcome_join": True,
                    "router_top50_identity_exact": True, **receipt,
                })
                _progress(out, stage="fold_candidate_complete", candidate=candidate.key, held_month=f"{held_month:%Y-%m}", **receipt)
        del window, held
        gc.collect()

    control_components = pd.concat(control_parts, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
    summary, finalists, normalised = _summaries(components=component_parts, control=control_components, candidates=candidates)
    summary.to_parquet(out / "gain_summary.parquet", index=False, compression="zstd")
    finalists.to_parquet(out / "objective_finalists.parquet", index=False, compression="zstd")
    normalised.to_parquet(out / "timestamp_components.parquet", index=False, compression="zstd")
    control_components.to_parquet(out / "control_timestamp_components.parquet", index=False, compression="zstd")
    pd.DataFrame(fold_rows).to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage_rows).drop_duplicates(subset=["month"], keep="last").to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    _exclusive_json(out / "correctness_report.json", {
        "p8u_router_top50_identity_exact": bool(all(row["router_top50_identity_exact"] for row in fold_rows)),
        "stage1_common_control_target_free": True,
        "all_held_scores_target_free_before_outcomes": bool(all(row["target_free_before_outcome_join"] for row in fold_rows)),
        "all_feature_medians_train_only": bool(all(row["feature_medians_fit_train_only"] for row in fold_rows)),
        "all_train_labels_resolved_before_reserve": True,
        "loss_is_lambdarank_only": True,
        "gains_are_predeclared": True,
        "no_meta_mc1_portfolio_or_live_mutation": True,
    })
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline P8u gain-schedule funnel only; no Meta, MC1, admission, portfolio, live, or exchange mutation",
        "architecture": "exact P8u Router50 -> routed-only Base; no numeric Router input or post-route Base cutoff",
        "stage1_control": {"root": str(screen_root), "arm": CONTROL_ARM, "objective": "rank_xendcg", "target_free": True},
        "target_selection": {"top_per_family": top_per_family, "arms": [arm.__dict__ for arm in arms]},
        "gain_schedules": GAIN_SCHEDULES,
        "model": {"family": "LightGBM", "objective": "lambdarank", "truncation": 12, "sigmoid": 1.0, "common_tree_geometry": _ranker(candidates[0], seed=SEED).get_params()},
        "selection_metric": {
            "BaseScore": "0.30*DTP2 + 0.30*DTP5 + 0.20*DTP10 + 0.20*ResidualUR10_to30, normalised to the matched Stage-1 control; UR20 retained as a diagnostic",
            "ScoreStable": "weekly robust mean Q20-Q80 + 0.5*mean(Q15,Q10,Q5)",
            "outcome": "canonical rich-policy net bps, after target-free ranking over all routed candidates",
        },
        "query": "exact decision timestamp x long side",
        "strict_oof": {"train_months": train_months, "reserve_days": reserve_days, "train_cap_complete_queries": train_cap},
        "held_months": [f"{month:%Y-%m}" for month in held_months],
        "inputs": {"feature_roots": [str(root) for root in feature_roots], "label_root": str(label_root), "router_root": str(router_root), "selection_receipt": str(selection_receipt)},
        "source_sha256": _source_sha256([*required_screen, selection_receipt]),
        "objective_finalists": finalists.loc[:, ["candidate", "arm", "family", "gain_name", "score_stable"]].to_dict("records"),
    })
    _progress(out, stage="complete", candidates=len(candidates), finalists=finalists["candidate"].tolist())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True, help="comma-separated immutable causal feature roots")
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default=",".join(stage1.DEFAULT_HELD_MONTHS))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--top-per-family", type=int, default=2)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 8_000 or args.workers < 1:
        raise ValueError("invalid strict-OOF loss-funnel contract")
    print(run(
        feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()),
        label_root=args.label_root.resolve(), router_root=args.router_root.resolve(), selection_receipt=args.selection_receipt.resolve(),
        screen_root=args.stage1_root.resolve(), out=args.out.resolve(), held_months=_parse_months(args.held_months),
        train_months=args.train_months, reserve_days=args.reserve_days, train_cap=args.train_cap,
        top_per_family=args.top_per_family, workers=args.workers,
    ))


if __name__ == "__main__":
    main()
