#!/usr/bin/env python3
"""Strict-OOF broad Router50 feature screen after target-free hygiene.

This is Phase J's bounded first pass.  It deliberately evaluates fields on
three chronologically separated folds, and persists held scores before rich
policy outcomes are joined for metrics.  The screen combines:

* full-model gain and split stability;
* global and top-tail absolute SHAP support on held feature matrices;
* train-only univariate Spearman rescue; and
* a deterministic .97 Spearman redundancy representative veto.

It produces a 300--500-field *serious* pool, not a final production feature
contract.  Random-subspace stability selection and the compression ladder are
separate later stages.  No score, label, or outcome is exposed as a feature.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_strict_r3_router_utility_contract_v1 as metric  # noqa: E402
import run_strict_r3_economic_recall_router as router  # noqa: E402


SCHEMA = "strict_r3_router_full_universe_prescreen_v1"
SEED = 1729
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")


def _write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _hash_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _months(value: str, *, minimum: int = 3) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{token.strip()}-01", tz="UTC") for token in value.split(",") if token.strip())
    if len(result) < minimum or tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError(f"held months must be at least {minimum} unique chronological YYYY-MM tokens")
    return result


def _roots(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(token.strip()).resolve() for token in value.split(",") if token.strip())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("feature roots must be a unique non-empty comma-separated list")
    return paths


def _load_hygiene(path: Path, roots: tuple[Path, ...]) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_router_full_universe_hygiene_v1":
        raise AssertionError("unexpected full-universe hygiene receipt")
    contract = tuple(str(field) for field in payload.get("feature_contract", []))
    if len(contract) < 100 or len(contract) != len(set(contract)):
        raise AssertionError("invalid hygiene feature contract")
    if payload.get("source_roots") != [str(root) for root in roots]:
        raise AssertionError("hygiene source roots differ from this feature screen")
    if payload.get("feature_contract_sha256") != _hash_lines(contract):
        raise AssertionError("hygiene feature hash mismatch")
    return contract


def _cap_queries(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Target-free deterministic whole-query cap."""
    if len(frame) <= cap:
        return frame
    sizes = frame.groupby("__decision_ts__", sort=False).size()
    stamps = pd.Series(frame["__decision_ts__"].drop_duplicates())
    hashes = stamps.astype(str).map(
        lambda value: int(hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()[:16], 16)
    )
    ordered = pd.DataFrame({"stamp": stamps, "hash": hashes}).sort_values(["hash", "stamp"], kind="stable")
    kept: list[pd.Timestamp] = []
    used = 0
    for stamp in ordered["stamp"]:
        count = int(sizes.loc[stamp])
        if used and used + count > cap:
            continue
        kept.append(stamp)
        used += count
        if used >= cap:
            break
    output = frame.loc[frame["__decision_ts__"].isin(kept)].copy()
    if len(output) < min(5_000, cap // 2):
        raise AssertionError("query cap left insufficient target-free support")
    return output


def _query_subsample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("query subsample fraction must be in (0, 1]")
    if fraction >= 1.0:
        return frame
    stamps = frame["__decision_ts__"].drop_duplicates()
    token = stamps.astype(str).map(
        lambda value: int(hashlib.sha256(f"subsample|{seed}|{value}".encode()).hexdigest()[:16], 16) / 2**64
    )
    selected = set(stamps.loc[token.lt(fraction)])
    output = frame.loc[frame["__decision_ts__"].isin(selected)].copy()
    if output.empty:
        raise AssertionError("target-free query subsample produced no rows")
    return output


def _feature_matrix(frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    work = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = work.median().fillna(0.0).to_numpy(np.float32) if medians is None else np.asarray(medians, dtype=np.float32)
    if len(fill) != len(fields):
        raise AssertionError("feature/median contract mismatch")
    return work.fillna(pd.Series(fill, index=fields)).fillna(0.0).to_numpy(np.float32), fill


def _rank_reference(raw: np.ndarray, held: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(raw, dtype=float)[np.isfinite(raw)], kind="stable")
    if len(values) < 100:
        raise AssertionError("insufficient finite strict-train rank reference")
    left = np.searchsorted(values, held, side="left")
    right = np.searchsorted(values, held, side="right")
    return np.clip(((left + right) * .5 + .5) / len(values), 0.0, 1.0).astype(np.float32)


def _ranker(*, fields: Sequence[str], train: pd.DataFrame, held: pd.DataFrame,
            primary: str, row_weight_scheme: str, seed: int,
            n_jobs: int) -> tuple[LGBMRanker, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    train = train.copy()
    target = router._primary_target(train, primary).astype(np.int32)
    train["__target__"] = target
    utility = router._primary_weight_utility(train, primary, target)
    ordered, groups, weights, weight_summary = router._query_weights(
        train, scheme=row_weight_scheme, primary_utility=utility,
    )
    train = train.iloc[ordered["__row__"].to_numpy(np.int64)].reset_index(drop=True)
    target = train["__target__"].to_numpy(np.int32)
    matrix, medians = _feature_matrix(train, fields)
    held_matrix, _ = _feature_matrix(held, fields, medians)
    classes = int(target.max()) + 1
    if classes != 6:
        raise AssertionError(f"{primary}: expected six frozen relevance grades, got {classes}")
    params: dict[str, object] = dict(
        objective="rank_xendcg", metric="ndcg", label_gain=[0, 1, 2, 4, 7, 11],
        n_estimators=180, learning_rate=.05, max_depth=4, num_leaves=15,
        min_child_samples=max(300, int(.012 * len(train))), min_split_gain=.002,
        subsample=.78, subsample_freq=1, colsample_bytree=.78,
        reg_alpha=.02, reg_lambda=1.5, max_bin=127,
        lambdarank_truncation_level=12, random_state=seed, n_jobs=n_jobs,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    model = LGBMRanker(**params).fit(matrix, target, group=groups, sample_weight=weights)
    train_raw = model.predict(matrix).astype(np.float32)
    held_raw = model.predict(held_matrix).astype(np.float32)
    held_rank = _rank_reference(train_raw, held_raw)
    return model, matrix, held_matrix, held_rank, weight_summary


def _sample_rows(frame: pd.DataFrame, max_rows: int, seed: int) -> np.ndarray:
    if len(frame) <= max_rows:
        return np.arange(len(frame), dtype=np.int64)
    token = pd.util.hash_pandas_object(frame["candidate_id"].astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    return np.argsort(token, kind="stable")[:max_rows]


def _univariate(train: pd.DataFrame, fields: Sequence[str], primary: str, max_rows: int, seed: int) -> pd.DataFrame:
    rows = _sample_rows(train, max_rows, seed)
    sample = train.iloc[rows]
    target = router._primary_weight_utility(
        sample, primary, router._primary_target(sample, primary),
    )
    target_rank = pd.Series(target).rank(method="average").to_numpy(float)
    target_rank -= target_rank.mean()
    target_norm = float(np.sqrt(np.dot(target_rank, target_rank)))
    records: list[dict[str, object]] = []
    for begin in range(0, len(fields), 48):
        block = list(fields[begin:begin + 48])
        values = sample.loc[:, block].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median()).fillna(0.0)
        ranks = values.rank(method="average").to_numpy(float)
        ranks -= ranks.mean(axis=0, keepdims=True)
        denom = np.sqrt(np.sum(ranks * ranks, axis=0)) * max(target_norm, 1e-12)
        corr = np.divide(ranks.T @ target_rank, denom, out=np.zeros(len(block)), where=denom > 0)
        records.extend({"feature": field, "univariate_spearman": float(value)} for field, value in zip(block, corr, strict=True))
    return pd.DataFrame(records)


def _metric(held: pd.DataFrame, rank: np.ndarray, policy: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    score = held.loc[:, list(IDENTITY)].copy()
    score["router_primary_rank"] = np.asarray(rank, dtype=np.float32)
    joined = score.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    joined["__net__"] = pd.to_numeric(joined["policy_net_bps"], errors="coerce")
    joined["__valid__"] = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(joined["__net__"])
    timestamp = metric._timestamp_primary(
        joined, "router_primary_rank", .50, metric.PRIMARY_POWER, metric.PRIMARY_CAP_BPS,
        metric.PRIMARY_GAMMA, metric.PRIMARY_TIMESTAMP_CAP,
    )
    return metric._primary_summary(timestamp), score


def _normalised_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True).fillna(0.0)


def _redundancy_veto(samples: pd.DataFrame, ranking: pd.DataFrame, keep: int) -> list[str]:
    order = ranking.sort_values(["screen_score", "feature"], ascending=[False, True], kind="stable")["feature"].tolist()
    # Calculating a large rank-correlation surface is both unhelpful and
    # expensive.  Restrict it to the candidate pool that could possibly enter
    # the serious contract, using only target-free bounded samples.
    pool = order[:min(len(order), max(keep * 2, 650))]
    values = samples.loc[:, pool].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median()).fillna(0.0).astype(np.float32)
    corr = values.corr(method="spearman").abs()
    selected: list[str] = []
    for field in order:
        if field not in pool:
            continue
        if not selected or float(corr.loc[field, selected].max()) < .97:
            selected.append(field)
        if len(selected) >= keep:
            break
    if len(selected) < min(100, keep):
        raise AssertionError("redundancy veto left inadequate serious feature support")
    return selected


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"immutable artifact already exists: {args.out}")
    roots = _roots(args.feature_roots)
    fields = _load_hygiene(args.hygiene_contract.resolve(), roots)
    held_months = _months(args.held_months, minimum=1 if args.single_fold_part else 3)
    # The incumbent Router ledger is identity keyed.  Reuse its narrow policy
    # reader rather than assuming a date-partitioned outcome schema.
    policy = router._policy_window(
        args.policy.resolve(), held_months[0] - pd.DateOffset(months=args.train_months + 2),
        held_months[-1] + pd.offsets.MonthBegin(1),
    )
    policy = policy.loc[:, [
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps", "policy_label_available_ts",
    ]].copy()
    args.out.mkdir(parents=True)
    _write_once(args.out / "run_contract.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF Router full-universe prescreen; no live, base, consensus, MC1, portfolio, or exchange mutation",
        "feature_roots": [str(root) for root in roots],
        "hygiene_contract": str(args.hygiene_contract.resolve()),
        "hygiene_feature_sha256": _hash_lines(fields),
        "feature_count": len(fields),
        "policy": str(args.policy.resolve()),
        "primary_target": args.primary_target,
        "row_weight_scheme": args.row_weight_scheme,
        "held_months": [f"{month:%Y-%m}" for month in held_months],
        "train_months": args.train_months, "reserve_days": args.reserve_days,
        "train_cap": args.train_cap, "held_cap": args.held_cap,
        "shap_rows": args.shap_rows, "tail_shap_rows": args.tail_shap_rows,
        "univariate_rows": args.univariate_rows, "serious_fields": args.serious_fields,
        "causality": "train labels must resolve before each fold reserve; held target-free scores persist before policy metrics join",
    })
    fold_importance: list[pd.DataFrame] = []
    fold_shap: list[pd.DataFrame] = []
    fold_univariate: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []
    redundancy_samples: list[pd.DataFrame] = []
    score_root = args.out / "target_free_scores"
    score_root.mkdir()
    for fold, held_month in enumerate(held_months):
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        train_start = reserve - pd.DateOffset(months=args.train_months)
        held_end = held_month + pd.offsets.MonthBegin(1)
        train_feature = router._window_features(roots, train_start, reserve, (*IDENTITY, *fields))
        held_feature = router._window_features(roots, held_month, held_end, (*IDENTITY, *fields))
        train = router._prepare_train(train_feature, None, policy, reserve)
        train = _cap_queries(train, args.train_cap, SEED + fold).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        held = _cap_queries(held_feature, args.held_cap, SEED + 100 + fold).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if len(train) < 20_000 or len(held) < 5_000:
            raise AssertionError(f"{held_month:%Y-%m}: insufficient strict support {len(train)} / {len(held)}")
        model, matrix, held_matrix, rank, weight_summary = _ranker(
            fields=fields, train=train, held=held, primary=args.primary_target,
            row_weight_scheme=args.row_weight_scheme, seed=SEED + fold, n_jobs=args.n_jobs,
        )
        # This is the decisive target-free receipt.  Policy outcomes are joined
        # only below, after this parquet has been atomically produced.
        score = held.loc[:, list(IDENTITY)].copy()
        score["router_primary_rank"] = rank
        score.to_parquet(score_root / f"month={held_month:%Y-%m}.parquet", index=False, compression="zstd")
        summary, _ = _metric(held, rank, policy)
        fold_metrics.append({"held_month": f"{held_month:%Y-%m}", "train_rows": len(train), "held_rows": len(held), **summary, **weight_summary})
        gain = model.booster_.feature_importance(importance_type="gain")
        split = model.booster_.feature_importance(importance_type="split")
        fold_importance.append(pd.DataFrame({"held_month": f"{held_month:%Y-%m}", "feature": fields, "gain": gain, "split": split}))
        sample = _sample_rows(held, args.shap_rows, SEED + 200 + fold)
        contribution = model.predict(held_matrix[sample], pred_contrib=True)
        fold_shap.append(pd.DataFrame({"held_month": f"{held_month:%Y-%m}", "feature": fields, "mean_abs_shap": np.abs(contribution[:, :-1]).mean(axis=0)}))
        tail_n = min(args.tail_shap_rows, len(held))
        tail = np.argsort(rank, kind="stable")[-tail_n:]
        contribution_tail = model.predict(held_matrix[tail], pred_contrib=True)
        fold_shap[-1]["tail_mean_abs_shap"] = np.abs(contribution_tail[:, :-1]).mean(axis=0)
        fold_univariate.append(_univariate(train, fields, args.primary_target, args.univariate_rows, SEED + 300 + fold).assign(held_month=f"{held_month:%Y-%m}"))
        sample_rows = _sample_rows(held, min(900, len(held)), SEED + 400 + fold)
        redundancy_samples.append(held.iloc[sample_rows].loc[:, list(fields)].reset_index(drop=True))
        print(json.dumps({"event": "fold_complete", "held_month": f"{held_month:%Y-%m}", "features": len(fields), "train_rows": len(matrix), "held_rows": len(held_matrix)}), flush=True)
        del model, train_feature, held_feature, train, held, matrix, held_matrix, contribution, contribution_tail
        gc.collect()
    print(json.dumps({"event": "aggregate_start", "folds": len(held_months)}), flush=True)
    importance = pd.concat(fold_importance, ignore_index=True)
    shap = pd.concat(fold_shap, ignore_index=True)
    univariate = pd.concat(fold_univariate, ignore_index=True)
    summary = importance.groupby("feature", sort=False).agg(
        gain_median=("gain", "median"), gain_mean=("gain", "mean"),
        split_median=("split", "median"), split_presence=("split", lambda x: float(pd.Series(x).gt(0).mean())),
    ).reset_index().merge(
        shap.groupby("feature", sort=False).agg(shap_median=("mean_abs_shap", "median"), tail_shap_median=("tail_mean_abs_shap", "median")).reset_index(),
        on="feature", how="inner", validate="one_to_one",
    ).merge(
        univariate.groupby("feature", sort=False).agg(univariate_spearman_median=("univariate_spearman", "median"), univariate_spearman_abs=("univariate_spearman", lambda x: float(np.median(np.abs(x))))).reset_index(),
        on="feature", how="inner", validate="one_to_one",
    )
    summary["screen_score"] = (
        .34 * _normalised_rank(summary["gain_median"])
        + .14 * _normalised_rank(summary["split_presence"])
        + .20 * _normalised_rank(summary["shap_median"])
        + .20 * _normalised_rank(summary["tail_shap_median"])
        + .12 * _normalised_rank(summary["univariate_spearman_abs"])
    )
    # Train-only univariate rescue: include the strongest fields whose full
    # model support is modest, so a correlated block cannot erase all signal
    # before replacement tests.  This does not bypass the redundancy veto.
    general = set(summary.nlargest(max(args.serious_fields * 2, 650), "screen_score")["feature"])
    rescue = set(summary.nlargest(max(args.serious_fields // 3, 100), "univariate_spearman_abs")["feature"])
    candidate = summary.loc[summary["feature"].isin(general | rescue)].copy()
    samples = pd.concat(redundancy_samples, ignore_index=True)
    selected = _redundancy_veto(samples, candidate, args.serious_fields)
    redundancy_output = samples
    del redundancy_samples
    gc.collect()
    summary["general_screen"] = summary["feature"].isin(general)
    summary["univariate_rescue"] = summary["feature"].isin(rescue)
    summary["serious_feature"] = summary["feature"].isin(selected)
    if len(selected) < args.serious_fields:
        # A .97 veto can legitimately reduce support slightly.  Preserve the
        # actual count rather than filling it with redundant fields.
        selected = [field for field in selected]
    importance.to_parquet(args.out / "fold_gain_split.parquet", index=False, compression="zstd")
    shap.to_parquet(args.out / "fold_shap.parquet", index=False, compression="zstd")
    univariate.to_parquet(args.out / "fold_univariate.parquet", index=False, compression="zstd")
    # A bounded target-free sample is enough for the later global redundancy
    # representative veto.  Persist it per isolated fold so worker memory is
    # released before the next fold starts.
    redundancy_output.to_parquet(
        args.out / "redundancy_sample.parquet", index=False, compression="zstd",
    )
    del redundancy_output
    summary.sort_values(["screen_score", "feature"], ascending=[False, True], kind="stable").to_parquet(args.out / "feature_screen.parquet", index=False, compression="zstd")
    pd.DataFrame(fold_metrics).to_parquet(args.out / "fold_metrics.parquet", index=False, compression="zstd")
    _write_once(args.out / "serious_feature_contract.json", {
        "schema": SCHEMA,
        "scope": "research-only Router serious feature pool; requires later random-subspace selection and compression ladder",
        "feature_contract": selected,
        "feature_contract_sha256": _hash_lines(selected),
        "feature_count": len(selected), "pre_veto_candidate_count": len(candidate),
        "selection": "three strict-OOF full-model folds; gain/split + global/tail SHAP + train-only univariate rescue + .97 Spearman representative veto",
        "primary_target": args.primary_target, "row_weight_scheme": args.row_weight_scheme,
        "held_months": [f"{month:%Y-%m}" for month in held_months],
    })
    _write_once(args.out / "run_manifest.json", {
        "schema": SCHEMA, "status": "complete",
        "scope": (
            "offline isolated strict-OOF Router feature prescreen part; requires aggregation"
            if args.single_fold_part else "offline Router feature prescreen only; no live/exchange mutation"
        ),
        "folds": len(held_months), "full_hygiene_fields": len(fields), "serious_features": len(selected),
        "target_free_score_receipts": [f"month={month:%Y-%m}.parquet" for month in held_months],
    })
    print(json.dumps({"event": "complete", "serious_features": len(selected)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True)
    parser.add_argument("--hygiene-contract", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=router.DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary-target", default="U50_p050_c300", choices=router.ALL_PRIMARY_TARGETS)
    parser.add_argument("--row-weight-scheme", default="positive_125", choices=router._ROW_WEIGHT_SCHEMES)
    parser.add_argument("--held-months", default="2025-10,2026-01,2026-04,2026-07")
    parser.add_argument("--single-fold-part", action="store_true", help="emit one immutable OOF fold part for later aggregation")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=45_000)
    parser.add_argument("--held-cap", type=int, default=20_000)
    parser.add_argument("--shap-rows", type=int, default=5_000)
    parser.add_argument("--tail-shap-rows", type=int, default=4_000)
    parser.add_argument("--univariate-rows", type=int, default=12_000)
    parser.add_argument("--serious-fields", type=int, default=420)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 20_000 or args.held_cap < 5_000:
        raise ValueError("strict OOF support settings below predeclared minimum")
    if not 300 <= args.serious_fields <= 500:
        raise ValueError("serious field pool must remain within the predeclared 300--500 range")
    run(args)


if __name__ == "__main__":
    main()
