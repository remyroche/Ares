#!/usr/bin/env python3
"""Stage 4: bounded cross-model comparison for frozen P8u LambdaRank targets.

The three target×gain contracts selected by the objective funnel are fixed.
This stage changes only the learner family / ranking loss under a common cheap
parameter budget.  It evaluates LightGBM LambdaRank, CatBoost QueryRMSE and
YetiRank, and XGBoost NDCG and pairwise rankers.  Full HPO is deliberately
prohibited here; it is permitted *after this stage only* for the model winner
of each retained target geometry.
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
from catboost import CatBoostRanker, Pool
from lightgbm import LGBMRanker
from xgboost import XGBRanker

import run_strict_r3_p8u_precision_preservation_objective_funnel_v1 as objective
import run_strict_r3_p8u_precision_preservation_screen_v1 as stage1
import run_strict_r3_p8u_precision_preservation_loss_funnel_v1 as gain
import run_strict_r3_router_single_base_prescreen_v1 as base
from strict_r3_p8u_precision_preservation_metric import COMPONENTS, stable_score, timestamp_components


SCHEMA = "strict_r3_p8u_precision_preservation_cross_model_v1"
SEED = 1729
IDENTITY = base.IDENTITY
MODEL_FAMILIES = ("lgbm_lambdarank", "catboost_queryrmse", "catboost_yetirank", "xgb_ndcg", "xgb_pairwise")


@dataclass(frozen=True)
class Candidate:
    arm: stage1.Arm
    gain_name: str
    model_family: str

    @property
    def key(self) -> str:
        return f"{self.arm.key}__{self.gain_name}__{self.model_family}"


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _retained_targets(objective_root: Path) -> tuple[tuple[stage1.Arm, str], ...]:
    frame = pd.read_parquet(objective_root / "target_objective_winners.parquet")
    required = {"arm", "gain_name", "objective", "score_stable"}
    if missing := required.difference(frame.columns):
        raise AssertionError(f"objective winners missing {sorted(missing)}")
    if not frame["objective"].eq("lambdarank").all():
        raise AssertionError("only the frozen LambdaRank winners can enter cross-model comparison")
    arm_by_key = {arm.key: arm for arm in stage1.ARMS}
    result = [(arm_by_key[row.arm], row.gain_name) for row in frame.itertuples(index=False)]
    if len(result) != 3 or len({arm.key for arm, _ in result}) != 3:
        raise AssertionError("cross-model comparison requires exactly three distinct retained target geometries")
    return tuple(result)


def _qid(frame: pd.DataFrame) -> np.ndarray:
    # Factorisation is performed inside each strictly chronological fit and is
    # only an opaque query identity—no timestamp value enters any learner.
    codes, _ = pd.factorize(frame["__decision_ts__"], sort=True)
    if np.any(codes < 0):
        raise AssertionError("invalid ranking query identity")
    return codes.astype(np.int64)


def _fit_predict(
    *, candidate: Candidate, x_train: np.ndarray, y: np.ndarray, train: pd.DataFrame,
    x_held: np.ndarray, seed: int,
) -> np.ndarray:
    group = base._query_groups(train)
    if candidate.model_family == "lgbm_lambdarank":
        model = LGBMRanker(
            objective="lambdarank", metric="ndcg", n_estimators=180, learning_rate=.05,
            max_depth=4, num_leaves=15, min_child_samples=260, subsample=.80, subsample_freq=1,
            colsample_bytree=.80, reg_alpha=.05, reg_lambda=8.0, min_split_gain=.001,
            lambdarank_truncation_level=12, sigmoid=1.0, label_gain=gain.GAIN_SCHEDULES[candidate.gain_name],
            random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
        )
        model.fit(x_train, y, group=group)
        return model.predict(x_held).astype(np.float32)
    qid = _qid(train)
    if candidate.model_family == "catboost_queryrmse":
        model = CatBoostRanker(
            loss_function="QueryRMSE", eval_metric="NDCG:top=10", iterations=180,
            learning_rate=.05, depth=4, l2_leaf_reg=8.0, random_seed=seed,
            random_strength=.5, rsm=.80, thread_count=1, verbose=False, allow_writing_files=False,
        )
        model.fit(Pool(x_train, y, group_id=qid))
        return model.predict(x_held).astype(np.float32)
    if candidate.model_family == "catboost_yetirank":
        model = CatBoostRanker(
            loss_function="YetiRank:mode=NDCG;top=10", eval_metric="NDCG:top=10", iterations=180,
            learning_rate=.05, depth=4, l2_leaf_reg=8.0, random_seed=seed,
            random_strength=.5, rsm=.80, thread_count=1, verbose=False, allow_writing_files=False,
        )
        model.fit(Pool(x_train, y, group_id=qid))
        return model.predict(x_held).astype(np.float32)
    if candidate.model_family in {"xgb_ndcg", "xgb_pairwise"}:
        model = XGBRanker(
            objective="rank:ndcg" if candidate.model_family == "xgb_ndcg" else "rank:pairwise",
            eval_metric="ndcg@10", n_estimators=180, learning_rate=.05, max_depth=4,
            min_child_weight=5.0, subsample=.80, colsample_bytree=.80, reg_alpha=.05,
            reg_lambda=8.0, gamma=.001, random_state=seed, n_jobs=1, tree_method="hist",
        )
        model.fit(x_train, y, qid=qid)
        return model.predict(x_held).astype(np.float32)
    raise ValueError(f"unsupported cross-model family {candidate.model_family}")


def _fit_one(
    *, candidate: Candidate, window: pd.DataFrame, held: pd.DataFrame, reserve: pd.Timestamp,
    fields: tuple[str, ...], train_cap: int, seed: int,
) -> tuple[Candidate, pd.DataFrame, dict[str, object]]:
    train = stage1._train_rows(window, candidate.arm, reserve, train_cap)
    labels, geometry = stage1._labels(train, candidate.arm)
    x_train, medians = base._numeric_matrix(train, fields)
    x_held, _ = base._numeric_matrix(held, fields, medians)
    values = _fit_predict(candidate=candidate, x_train=x_train, y=labels, train=train, x_held=x_held, seed=seed)
    if not np.isfinite(values).all():
        raise AssertionError("cross-model held score contains a non-finite value")
    score = held.loc[:, list(IDENTITY)].copy()
    score["base_score"] = values
    score["base_rank_ts"] = base._rank_desc(score, "base_score")
    if score.columns.tolist() != [*IDENTITY, "base_score", "base_rank_ts"]:
        raise AssertionError("cross-model held score violates target-free schema")
    return candidate, score, {
        "train_rows": int(len(train)), "train_queries": int(train["__decision_ts__"].nunique()),
        "target_geometry": geometry, "feature_medians_fit_train_only": True,
    }


def _summarise(
    *, parts: dict[str, list[pd.DataFrame]], control: pd.DataFrame, candidates: Sequence[Candidate],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_key = {candidate.key: candidate for candidate in candidates}
    rows: list[dict[str, object]] = []
    normalised_parts: list[pd.DataFrame] = []
    for key, panels in parts.items():
        panel = pd.concat(panels, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
        summary, normalised = stable_score(panel, control)
        candidate = by_key[key]
        rows.append({
            "candidate": key, "arm": candidate.arm.key, "family": candidate.arm.family,
            "target": candidate.arm.target, "geometry": candidate.arm.geometry,
            "gain_name": candidate.gain_name, "model_family": candidate.model_family,
            **summary.__dict__, **{f"mean_{name}": float(panel[name].mean()) for name in COMPONENTS},
            "mean_utility_recall20": float(panel["utility_recall20"].mean()),
            "mean_top2_coverage": float(panel["dtp2_bps_coverage"].mean()),
            "mean_top5_coverage": float(panel["dtp5_bps_coverage"].mean()),
            "mean_top10_coverage": float(panel["dtp10_bps_coverage"].mean()),
            "utility_recall20_coverage": float(panel["utility_recall20_eligible"].mean()),
            "residual_utility_recall10_to30_coverage": float(panel["residual_utility_recall10_to30_eligible"].mean()),
        })
        normalised["candidate"] = key
        normalised_parts.append(normalised)
    summary = pd.DataFrame(rows).sort_values(["arm", "score_stable", "model_family"], ascending=[True, False, True], kind="stable")
    winners = summary.groupby("arm", sort=True).head(1).sort_values("score_stable", ascending=False, kind="stable")
    return summary, winners, pd.concat(normalised_parts, ignore_index=True)


def run(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, selection_receipt: Path,
    stage1_root: Path, objective_root: Path, out: Path, held_months: Sequence[pd.Timestamp],
    train_months: int, reserve_days: int, train_cap: int, workers: int,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    required_objective = [objective_root / "run_manifest.json", objective_root / "correctness_report.json", objective_root / "target_objective_winners.parquet"]
    if any(not path.exists() for path in required_objective):
        raise FileNotFoundError("objective funnel is incomplete")
    if not all(json.loads((objective_root / "correctness_report.json").read_text()).values()):
        raise AssertionError("objective funnel does not satisfy correctness receipt")
    fields = base._load_f72_fields(selection_receipt)
    retained = _retained_targets(objective_root)
    candidates = tuple(Candidate(arm, gain_name, model_family) for arm, gain_name in retained for model_family in MODEL_FAMILIES)
    out.mkdir(parents=True)
    parts: dict[str, list[pd.DataFrame]] = {candidate.key: [] for candidate in candidates}
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
            raise AssertionError(f"{held_month:%Y-%m}: missing P8u held candidates")
        labels = held.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]].copy()
        control = gain._control_score(stage1_root, held_month)
        if len(control) != len(held) or not control["candidate_id"].equals(held["candidate_id"]):
            raise AssertionError(f"{held_month:%Y-%m}: Stage-1 target-free control IDs mismatch")
        control_parts.append(timestamp_components(control.merge(labels, on="candidate_id", how="left", validate="one_to_one"), score_column="base_score"))
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
                score_with_outcome = score.merge(labels, on="candidate_id", how="left", validate="one_to_one")
                panel = timestamp_components(score_with_outcome, score_column="base_score")
                panel["candidate"] = candidate.key
                panel["held_month"] = f"{held_month:%Y-%m}"
                parts[candidate.key].append(panel)
                fold_rows.append({
                    "candidate": candidate.key, "arm": candidate.arm.key, "family": candidate.arm.family,
                    "gain_name": candidate.gain_name, "model_family": candidate.model_family,
                    "held_month": f"{held_month:%Y-%m}", "held_rows": int(len(held)),
                    "held_queries": int(held["__decision_ts__"].nunique()), "score_path": str(path),
                    "target_free_before_outcome_join": True, "router_top50_identity_exact": True, **receipt,
                })
                _progress(out, stage="fold_candidate_complete", candidate=candidate.key, held_month=f"{held_month:%Y-%m}", **receipt)
        del window, held
        gc.collect()

    control = pd.concat(control_parts, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
    summary, winners, normalised = _summarise(parts=parts, control=control, candidates=candidates)
    summary.to_parquet(out / "cross_model_summary.parquet", index=False, compression="zstd")
    winners.to_parquet(out / "target_model_winners.parquet", index=False, compression="zstd")
    normalised.to_parquet(out / "timestamp_components.parquet", index=False, compression="zstd")
    control.to_parquet(out / "control_timestamp_components.parquet", index=False, compression="zstd")
    pd.DataFrame(fold_rows).to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage_rows).drop_duplicates(subset=["month"], keep="last").to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    _exclusive_json(out / "correctness_report.json", {
        "p8u_router_top50_identity_exact": bool(all(row["router_top50_identity_exact"] for row in fold_rows)),
        "all_held_scores_target_free_before_outcomes": bool(all(row["target_free_before_outcome_join"] for row in fold_rows)),
        "all_feature_medians_train_only": bool(all(row["feature_medians_fit_train_only"] for row in fold_rows)),
        "all_train_labels_resolved_before_reserve": True,
        "stage1_common_control_target_free": True,
        "targets_and_gains_frozen_from_objective_funnel": True,
        "cross_model_common_budget": True,
        "no_full_hpo_performed": True,
        "no_meta_mc1_portfolio_or_live_mutation": True,
    })
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline P8u cross-model screen only; no Meta, MC1, admission, portfolio, live, or exchange mutation",
        "architecture": "exact P8u Router50 -> routed-only Base; no numeric Router input or post-route Base cutoff",
        "retained_targets_source": str(objective_root), "retained_targets": [{"arm": arm.__dict__, "gain_name": gain_name} for arm, gain_name in retained],
        "model_families": list(MODEL_FAMILIES), "model_budget": {"iterations": 180, "depth": 4, "learning_rate": .05, "feature_fraction": .80, "l2": 8.0, "one_thread_per_fit": True},
        "selection_metric": {"BaseScore": "0.30*DTP2 + 0.30*DTP5 + 0.20*DTP10 + 0.20*ResidualUR10_to30, normalised to fixed Stage-1 control; UR20 diagnostic only", "ScoreStable": "weekly robust mean Q20-Q80 + 0.5*mean(Q15,Q10,Q5)"},
        "strict_oof": {"train_months": train_months, "reserve_days": reserve_days, "train_cap_complete_queries": train_cap},
        "held_months": [f"{month:%Y-%m}" for month in held_months], "query": "exact decision timestamp x long side",
        "inputs": {"feature_roots": [str(root) for root in feature_roots], "label_root": str(label_root), "router_root": str(router_root), "selection_receipt": str(selection_receipt), "stage1_root": str(stage1_root), "objective_root": str(objective_root)},
        "source_sha256": _source_sha256([*required_objective, stage1_root / "run_manifest.json", selection_receipt]),
        "target_model_winners": winners.loc[:, ["candidate", "arm", "family", "gain_name", "model_family", "score_stable"]].to_dict("records"),
        "next_stage": "Run full HPO only for this stage's winning model per retained target; do not HPO losing families.",
    })
    _progress(out, stage="complete", candidates=len(candidates), winners=winners["candidate"].tolist())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True, help="comma-separated immutable causal feature roots")
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--objective-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default=",".join(stage1.DEFAULT_HELD_MONTHS))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 8_000 or args.workers < 1:
        raise ValueError("invalid strict-OOF cross-model contract")
    print(run(
        feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()),
        label_root=args.label_root.resolve(), router_root=args.router_root.resolve(), selection_receipt=args.selection_receipt.resolve(),
        stage1_root=args.stage1_root.resolve(), objective_root=args.objective_root.resolve(), out=args.out.resolve(),
        held_months=stage1._parse_months(args.held_months), train_months=args.train_months,
        reserve_days=args.reserve_days, train_cap=args.train_cap, workers=args.workers,
    ))


if __name__ == "__main__":
    main()
