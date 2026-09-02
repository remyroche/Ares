#!/usr/bin/env python3
"""Random-subspace stability selection for the strict-R3 Router feature pool.

This is the second (and deliberately separate) stage of the full-universe
Router feature-selection funnel.  It consumes the prior three-fold 300--500
field serious pool, fits only strict-prequential cheap RankXENDCG models, and
measures feature inclusion against timestamp-local Router-50 utility.  Held
scores are materialised before canonical policy outcomes are joined for
diagnostics.  It has no live, exchange, base, consensus, MC1, or portfolio
side effects.

The output remains a *shortlist* for the later subset-compression ladder; it
is explicitly not a production feature contract.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_strict_r3_router_utility_contract_v1 as metric  # noqa: E402
import run_strict_r3_economic_recall_router as router  # noqa: E402
import screen_strict_r3_router_full_universe_v1 as screen  # noqa: E402


SCHEMA = "strict_r3_router_full_universe_stability_v1"
SEED = 1729
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
FEATURE_FRACTIONS = (.35, .45, .55)
QUERY_FRACTIONS = (.65, .70, .75, .80)
LABEL_GAIN = [0, 1, 2, 4, 7, 11]


def _write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _hash_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _parse_months(value: str) -> tuple[pd.Timestamp, ...]:
    months = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in value.split(",") if item.strip())
    if len(months) != 3 or tuple(sorted(months)) != months or len(set(months)) != 3:
        raise ValueError("held months must be exactly three unique chronological YYYY-MM values")
    span = (months[-1].year - months[0].year) * 12 + months[-1].month - months[0].month
    if len({month.year for month in months}) < 2 or span < 6:
        raise ValueError("held months must span two calendar years and at least six months")
    return months


def _roots(value: str) -> tuple[Path, ...]:
    result = tuple(Path(item.strip()).resolve() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("feature roots must be a non-empty unique list")
    return result


def _load_serious(path: Path, roots: tuple[Path, ...], hygiene: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_router_full_universe_prescreen_aggregate_v1":
        raise AssertionError("unexpected serious-feature contract schema")
    fields = tuple(str(value) for value in payload.get("feature_contract", ()))
    if not 300 <= len(fields) <= 500 or len(fields) != len(set(fields)):
        raise AssertionError("serious feature contract must contain 300--500 unique fields")
    if payload.get("feature_contract_sha256") != _hash_lines(fields):
        raise AssertionError("serious feature contract hash mismatch")
    hygiene_payload = json.loads(hygiene.read_text())
    if hygiene_payload.get("source_roots") != [str(root) for root in roots]:
        raise AssertionError("serious and hygiene source roots differ")
    hygienic = set(map(str, hygiene_payload.get("feature_contract", ())))
    if not set(fields).issubset(hygienic):
        raise AssertionError("serious contract contains non-hygienic fields")
    return fields


def _query_subsample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("query fraction must be in (0, 1]")
    if fraction == 1.0:
        return frame
    stamps = frame["__decision_ts__"].drop_duplicates().sort_values(kind="stable")
    hashed = stamps.astype(str).map(
        lambda token: int(hashlib.sha256(f"{seed}|{token}".encode()).hexdigest()[:16], 16) / 2**64
    )
    selected = set(stamps.loc[hashed.lt(fraction)])
    output = frame.loc[frame["__decision_ts__"].isin(selected)].copy()
    if output.empty or output["__decision_ts__"].nunique() < 12:
        raise AssertionError("whole-query subsample has inadequate timestamp support")
    return output


def _subspace(fields: Sequence[str], fraction: float, seed: int) -> tuple[str, ...]:
    count = max(32, int(np.ceil(len(fields) * fraction)))
    keyed = sorted(
        (int(hashlib.sha256(f"{seed}|{field}".encode()).hexdigest()[:16], 16), field)
        for field in fields
    )
    return tuple(field for _key, field in keyed[:count])


def _matrix(frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = values.median().fillna(0.0).to_numpy(np.float32) if medians is None else np.asarray(medians, dtype=np.float32)
    if len(fill) != len(fields):
        raise AssertionError("matrix feature/median mismatch")
    return values.fillna(pd.Series(fill, index=fields)).fillna(0.0).to_numpy(np.float32), fill


def _trial_model(seed: int, train_rows: int, n_jobs: int) -> LGBMRanker:
    return LGBMRanker(
        objective="rank_xendcg", metric="ndcg", label_gain=LABEL_GAIN,
        n_estimators=180, learning_rate=.05, max_depth=4, num_leaves=15,
        min_child_samples=max(300, int(.012 * train_rows)), min_split_gain=.002,
        subsample=.78, subsample_freq=1, colsample_bytree=.78,
        reg_alpha=.02, reg_lambda=1.5, max_bin=127,
        lambdarank_truncation_level=12, random_state=seed, n_jobs=n_jobs,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )


def _metric(held: pd.DataFrame, rank: np.ndarray, policy: pd.DataFrame) -> dict[str, float]:
    score = held.loc[:, list(IDENTITY)].copy()
    score["router_primary_rank"] = np.asarray(rank, dtype=np.float32)
    joined = score.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    joined["__net__"] = pd.to_numeric(joined["policy_net_bps"], errors="coerce")
    joined["__valid__"] = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(joined["__net__"])
    timestamp = metric._timestamp_primary(
        joined, "router_primary_rank", .50, metric.PRIMARY_POWER, metric.PRIMARY_CAP_BPS,
        metric.PRIMARY_GAMMA, metric.PRIMARY_TIMESTAMP_CAP,
    )
    return metric._primary_summary(timestamp)


def _prepare_fold(
    *, roots: tuple[Path, ...], fields: tuple[str, ...], policy: pd.DataFrame,
    held_month: pd.Timestamp, train_months: int, reserve_days: int,
    train_cap: int, held_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    reserve = held_month - pd.Timedelta(days=reserve_days)
    train_start = reserve - pd.DateOffset(months=train_months)
    held_end = held_month + pd.offsets.MonthBegin(1)
    train_feature = router._window_features(roots, train_start, reserve, (*IDENTITY, *fields))
    held = router._window_features(roots, held_month, held_end, (*IDENTITY, *fields))
    # Use the exact same target-free whole-query cap as the canonical forward
    # scorer.  A feature screen may be cheap, but its retained HPO contract
    # must not be based on a different timestamp sample from the deployment
    # path.
    train_feature = router._deterministic_query_cap(train_feature, cap=train_cap)
    train = router._prepare_train(train_feature, None, policy, reserve)
    train = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    held = screen._cap_queries(held, held_cap, SEED + 10_000 + held_month.month)
    held = held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(train) < 20_000 or len(held) < 5_000:
        raise AssertionError(f"{held_month:%Y-%m}: insufficient strict support {len(train)} / {len(held)}")
    if train["__decision_ts__"].nunique() < 500:
        raise AssertionError(f"{held_month:%Y-%m}: fewer than 500 eligible training timestamp queries")
    train_matrix, medians = _matrix(train, fields)
    held_matrix, _ = _matrix(held, fields, medians)
    return train, held, train_matrix, held_matrix, medians


def _fold_trials(
    *, root: Path, held_month: pd.Timestamp, train: pd.DataFrame, held: pd.DataFrame,
    train_matrix: np.ndarray, held_matrix: np.ndarray, fields: tuple[str, ...],
    policy: pd.DataFrame, primary_target: str, row_weight_scheme: str,
    trials: int, n_jobs: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lookup = pd.Series(np.arange(len(train), dtype=np.int64), index=train.index)
    positions = {field: index for index, field in enumerate(fields)}
    score_root = root / "target_free_trial_scores" / f"fold={held_month:%Y-%m}"
    score_root.mkdir(parents=True)
    metric_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    for trial in range(trials):
        seed = SEED + held_month.year * 10_000 + held_month.month * 100 + trial
        feature_fraction = FEATURE_FRACTIONS[trial % len(FEATURE_FRACTIONS)]
        query_fraction = QUERY_FRACTIONS[(trial // len(FEATURE_FRACTIONS)) % len(QUERY_FRACTIONS)]
        subset = _subspace(fields, feature_fraction, seed)
        sub_train = _query_subsample(train, query_fraction, seed)
        target = router._primary_target(sub_train, primary_target).astype(np.int32)
        sub_train = sub_train.copy()
        sub_train["__target__"] = target
        utility = router._primary_weight_utility(sub_train, primary_target, target)
        ordered, groups, weights, weight_summary = router._query_weights(
            sub_train, scheme=row_weight_scheme, primary_utility=utility,
        )
        original_positions = lookup.reindex(sub_train.index).to_numpy(np.int64)
        row_positions = original_positions[ordered["__row__"].to_numpy(np.int64)]
        field_positions = np.asarray([positions[field] for field in subset], dtype=np.int64)
        model = _trial_model(seed, len(row_positions), n_jobs)
        model.fit(
            train_matrix[row_positions][:, field_positions],
            sub_train.iloc[ordered["__row__"].to_numpy(np.int64)]["__target__"].to_numpy(np.int32),
            group=groups, sample_weight=weights,
        )
        train_raw = model.predict(train_matrix[row_positions][:, field_positions]).astype(np.float32)
        held_raw = model.predict(held_matrix[:, field_positions]).astype(np.float32)
        rank = screen._rank_reference(train_raw, held_raw)
        score = held.loc[:, list(IDENTITY)].copy()
        score["router_primary_rank"] = rank
        score["trial"] = trial
        score.to_parquet(score_root / f"trial={trial:02d}.parquet", index=False, compression="zstd")
        summary = _metric(held, rank, policy)
        metric_rows.append({
            "held_month": f"{held_month:%Y-%m}", "trial": trial, "seed": seed,
            "feature_fraction": feature_fraction, "query_fraction": query_fraction,
            "feature_count": len(subset), "train_rows": len(row_positions), "held_rows": len(held),
            **summary, **weight_summary,
        })
        membership_rows.extend({"held_month": f"{held_month:%Y-%m}", "trial": trial, "feature": field} for field in subset)
        del model, sub_train, train_raw, held_raw, score
        gc.collect()
    return metric_rows, membership_rows


def _feature_evidence(metrics: pd.DataFrame, membership: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_month = {month: frame.set_index("trial") for month, frame in metrics.groupby("held_month", sort=True)}
    included = membership.groupby(["held_month", "feature"], sort=False).trial.agg(set).to_dict()
    for field in fields:
        deltas: list[float] = []
        recalls: list[float] = []
        n_yes = n_no = 0
        for month, frame in by_month.items():
            yes_trials = included.get((month, field), set())
            yes_frame = frame.loc[frame.index.isin(yes_trials)]
            no_frame = frame.loc[~frame.index.isin(yes_trials)]
            if len(yes_frame) < 3 or len(no_frame) < 3:
                continue
            deltas.append(float(yes_frame["s_router"].mean() - no_frame["s_router"].mean()))
            recalls.append(float(yes_frame["r50_utility"].mean() - no_frame["r50_utility"].mean()))
            n_yes += len(yes_frame)
            n_no += len(no_frame)
        if len(deltas) != len(by_month):
            continue
        values = np.asarray(deltas, dtype=float)
        rows.append({
            "feature": field, "included_models": n_yes, "excluded_models": n_no,
            "mean_s_router_delta": float(values.mean()),
            "q25_s_router_delta": float(np.quantile(values, .25)),
            "worst_s_router_delta": float(values.min()),
            "positive_month_fraction": float((values >= 0.0).mean()),
            "mean_r50_utility_delta": float(np.mean(recalls)),
            "stability_score": float(.65 * values.mean() + .25 * np.quantile(values, .25) + .10 * values.min()),
        })
    evidence = pd.DataFrame(rows)
    if evidence.empty:
        raise AssertionError("random-subspace inclusion evidence is empty")
    return evidence.sort_values(["stability_score", "mean_r50_utility_delta", "feature"], ascending=[False, False, True], kind="stable")


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"immutable artifact already exists: {args.out}")
    roots = _roots(args.feature_roots)
    fields = _load_serious(args.serious_contract.resolve(), roots, args.hygiene_contract.resolve())
    held_months = _parse_months(args.held_months)
    policy = router._policy_window(
        args.policy.resolve(), held_months[0] - pd.DateOffset(months=args.train_months + 2),
        held_months[-1] + pd.offsets.MonthBegin(1),
    ).loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps", "policy_label_available_ts"]].copy()
    args.out.mkdir(parents=True)
    _write_once(args.out / "run_contract.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF Router random-subspace stability selection; no live, exchange, base, consensus, MC1, or portfolio mutation",
        "feature_roots": [str(root) for root in roots], "serious_contract": str(args.serious_contract.resolve()),
        "serious_feature_count": len(fields), "serious_feature_sha256": _hash_lines(fields),
        "hygiene_contract": str(args.hygiene_contract.resolve()), "policy": str(args.policy.resolve()),
        "held_months": [f"{month:%Y-%m}" for month in held_months],
        "strict_train": {"train_months": args.train_months, "reserve_days": args.reserve_days, "label_available_before_reserve": True},
        "cap": {"train_rows": args.train_cap, "held_rows": args.held_cap, "whole_timestamp_queries": True},
        "target": args.primary_target, "row_weight_scheme": args.row_weight_scheme,
        "random_subspaces": {"models_per_fold": args.random_models, "feature_fractions": FEATURE_FRACTIONS, "query_fractions": QUERY_FRACTIONS},
        "target_free_held_scores_persist_before_metric_join": True,
    })
    all_metrics: list[dict[str, object]] = []
    all_memberships: list[dict[str, object]] = []
    for held_month in held_months:
        train, held, train_matrix, held_matrix, _medians = _prepare_fold(
            roots=roots, fields=fields, policy=policy, held_month=held_month,
            train_months=args.train_months, reserve_days=args.reserve_days,
            train_cap=args.train_cap, held_cap=args.held_cap,
        )
        metrics, memberships = _fold_trials(
            root=args.out, held_month=held_month, train=train, held=held,
            train_matrix=train_matrix, held_matrix=held_matrix, fields=fields, policy=policy,
            primary_target=args.primary_target, row_weight_scheme=args.row_weight_scheme,
            trials=args.random_models, n_jobs=args.n_jobs,
        )
        all_metrics.extend(metrics)
        all_memberships.extend(memberships)
        with (args.out / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "fold_complete", "held_month": f"{held_month:%Y-%m}", "trials": args.random_models}) + "\n")
        del train, held, train_matrix, held_matrix
        gc.collect()
    metric_frame = pd.DataFrame(all_metrics)
    membership_frame = pd.DataFrame(all_memberships)
    evidence = _feature_evidence(metric_frame, membership_frame, fields)
    selected = evidence.head(args.survivor_count).feature.tolist()
    if len(selected) != args.survivor_count:
        raise AssertionError("stability shortlist unexpectedly underfilled")
    metric_frame.to_parquet(args.out / "random_subspace_metrics.parquet", index=False, compression="zstd")
    membership_frame.to_parquet(args.out / "random_subspace_membership.parquet", index=False, compression="zstd")
    evidence.to_parquet(args.out / "feature_inclusion_evidence.parquet", index=False, compression="zstd")
    _write_once(args.out / "stability_shortlist_contract.json", {
        "schema": SCHEMA,
        "scope": "research-only Router stability shortlist; requires subset ladder/HPO before any downstream comparison",
        "feature_contract": selected, "feature_contract_sha256": _hash_lines(selected), "feature_count": len(selected),
        "selection": "40 randomized whole-query feature/query subspaces per each of three strict-OOF cross-year folds; stable inclusion effect on S_router",
        "primary_target": args.primary_target, "row_weight_scheme": args.row_weight_scheme,
        "held_months": [f"{month:%Y-%m}" for month in held_months],
    })
    _write_once(args.out / "run_manifest.json", {
        "schema": SCHEMA, "status": "complete", "held_months": [f"{month:%Y-%m}" for month in held_months],
        "serious_features": len(fields), "shortlisted_features": len(selected),
        "scope": "offline Router stability selection complete; no refit or live mutation beyond research receipts",
    })
    print(json.dumps({"event": "complete", "shortlisted_features": len(selected), "trials": len(metric_frame)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True)
    parser.add_argument("--hygiene-contract", type=Path, required=True)
    parser.add_argument("--serious-contract", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=router.DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default="2025-10,2026-02,2026-06")
    parser.add_argument("--primary-target", default="U50_p050_c300", choices=router.ALL_PRIMARY_TARGETS)
    parser.add_argument("--row-weight-scheme", default="positive_125", choices=router._ROW_WEIGHT_SCHEMES)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=30_000)
    parser.add_argument("--held-cap", type=int, default=12_000)
    parser.add_argument("--random-models", type=int, default=40)
    parser.add_argument("--survivor-count", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.random_models < 40 or args.random_models > 60:
        raise ValueError("the predeclared randomized stability stage requires 40--60 models per fold")
    if not 300 <= args.survivor_count <= 500:
        raise ValueError("shortlist must retain 300--500 fields for the later compression ladder")
    if args.train_months < 3 or args.reserve_days < 28 or args.train_cap < 20_000 or args.held_cap < 5_000:
        raise ValueError("strict Router stability support below predeclared minimum")
    run(args)


if __name__ == "__main__":
    main()
