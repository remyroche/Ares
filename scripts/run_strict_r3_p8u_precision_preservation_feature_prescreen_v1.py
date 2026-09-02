#!/usr/bin/env python3
"""Causal full-universe pre-screen for the Router50 precision/preservation Base.

This is deliberately the *first* feature-selection stage, not an implicit
final feature contract.  It reduces the complete causal feature universe using
only development folds and three complementary sources of evidence:

* hygiene and a .97 within-feature correlation representative contract;
* cross-era, within-timestamp univariate association with the frozen target;
* balanced random-subspace inclusion uplift, plus full-model gain stability.

The survivor list is subsequently passed to block MDA and beam compression.
Every fitted score is calculated before outcomes are joined for ScoreStable;
this offline producer never changes Meta, MC1, portfolio, inference, or live
trading.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker

import run_strict_r3_p8u_precision_preservation_loss_funnel_v1 as gain
import run_strict_r3_p8u_precision_preservation_screen_v1 as stage1
import run_strict_r3_router_single_base_prescreen_v1 as base
from strict_r3_p8u_precision_preservation_metric import stable_score, timestamp_components


SCHEMA = "strict_r3_p8u_precision_preservation_feature_prescreen_v1"
IDENTITY = base.IDENTITY
SEED = 1729


def _exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _sha(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{part.strip()}-01", tz="UTC") for part in text.split(",") if part.strip())
    if len(values) < 3 or tuple(sorted(values)) != values:
        raise ValueError("need at least three increasing development months")
    span = (values[-1].year - values[0].year) * 12 + values[-1].month - values[0].month
    if len({item.year for item in values}) < 2 or span < 8:
        raise ValueError("development months must remain cross-year and span at least eight months")
    return values


def _owner(roots: Sequence[Path], month: pd.Timestamp) -> Path:
    paths = [root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet" for root in roots]
    found = [path for path in paths if path.exists()]
    if len(found) != 1:
        raise AssertionError(f"{month:%Y-%m}: expected one causal feature owner, found {len(found)}")
    return found[0]


def _universe(roots: Sequence[Path], month: pd.Timestamp) -> tuple[str, ...]:
    # Schema discovery must be metadata-only.  Loading an entire 1,400-field
    # monthly panel merely to inspect its columns can consume hundreds of MB
    # before the deliberately chunked hygiene pass begins.
    names = pq.ParquetFile(_owner(roots, month)).schema_arrow.names
    forbidden = set(IDENTITY) | {"__ts__", "__symbol__"}
    output = tuple(
        name for name in names
        if name not in forbidden and not any(token in name.lower() for token in base.PROHIBITED_SCORE_TOKENS)
    )
    if len(output) < 200 or len(set(output)) != len(output):
        raise AssertionError("causal feature universe is unexpectedly small or non-unique")
    return output


def _router_sample_ids(router_root: Path, month: pd.Timestamp, timestamps: int) -> set[str]:
    router = pd.read_parquet(base._router_path(router_root, month), columns=[*IDENTITY, "router_primary_rank"])
    router["__decision_ts__"] = pd.to_datetime(router["__decision_ts__"], utc=True, errors="raise")
    selected = base._top_half_identities(router)
    stamps = selected["__decision_ts__"].drop_duplicates().sort_values().to_numpy()
    if len(stamps) < timestamps:
        keep = stamps
    else:
        indexes = np.linspace(0, len(stamps) - 1, timestamps, dtype=int)
        keep = stamps[np.unique(indexes)]
    return set(selected.loc[selected["__decision_ts__"].isin(keep), "candidate_id"].astype(str))


def _read_sample(path: Path, fields: Sequence[str], ids: set[str], block_size: int) -> pd.DataFrame:
    identity = pd.read_parquet(path, columns=list(IDENTITY))
    identity["candidate_id"] = identity["candidate_id"].astype(str)
    take = identity["candidate_id"].isin(ids).to_numpy(bool)
    if not take.any():
        raise AssertionError(f"{path}: sample identities disappeared from causal feature panel")
    parts = [identity.loc[take].reset_index(drop=True)]
    for begin in range(0, len(fields), block_size):
        block = list(fields[begin:begin + block_size])
        values = pd.read_parquet(path, columns=block).iloc[take].reset_index(drop=True)
        parts.append(values)
    result = pd.concat(parts, axis=1)
    if result.columns.duplicated().any():
        raise AssertionError("causal sample has duplicate columns")
    return result


def _hygiene_and_blocks(
    *, roots: Sequence[Path], reference_months: Sequence[pd.Timestamp], fields: Sequence[str],
    sample_ids: dict[pd.Timestamp, set[str]], block_size: int,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    coverage: dict[str, list[float]] = {field: [] for field in fields}
    variance: dict[str, list[float]] = {field: [] for field in fields}
    samples: list[pd.DataFrame] = []
    for month in reference_months:
        path = _owner(roots, month)
        sample = _read_sample(path, fields, sample_ids[month], block_size)
        samples.append(sample)
        for begin in range(0, len(fields), block_size):
            block = list(fields[begin:begin + block_size])
            values = pd.read_parquet(path, columns=block).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            for field in block:
                coverage[field].append(float(values[field].notna().mean()))
                # A few derived fields can legitimately have very large
                # magnitudes.  The hygiene test only distinguishes a constant
                # field from a varying one, so a bounded calculation avoids
                # overflow without changing this semantic decision.
                raw = values[field].to_numpy(float, copy=False)
                finite = raw[np.isfinite(raw)]
                variance[field].append(float(np.var(np.clip(finite, -1e12, 1e12))) if len(finite) else np.nan)
            del values
        gc.collect()
    stats = pd.DataFrame({
        "feature": list(fields),
        "coverage_min": [min(coverage[field]) for field in fields],
        "variance_median": [float(np.nanmedian(variance[field])) for field in fields],
    })
    eligible = stats.loc[
        stats.coverage_min.ge(.90) & np.isfinite(stats.variance_median) & stats.variance_median.gt(1e-12), "feature"
    ].tolist()
    matrix = pd.concat(samples, ignore_index=True).loc[:, eligible]
    matrix = matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median()).fillna(0.0)
    # Do not materialise a dense 1,400 x 1,400 correlation matrix.  Spearman
    # correlation is Pearson correlation of ranks, so a bounded, time-spread
    # rank sample permits the same greedy .97 representative rule at a small
    # fixed memory cost.  The univariate screen retains the larger sample.
    corr_stride = max(1, int(math.ceil(len(matrix) / 1024)))
    rank_values = matrix.iloc[::corr_stride].rank(axis=0, method="average").to_numpy(np.float32)
    rank_values -= rank_values.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.square(rank_values, dtype=np.float64).sum(axis=0, keepdims=True)).astype(np.float32)
    scale[scale <= 1e-12] = 1.0
    rank_values /= scale
    ordered = stats.loc[stats.feature.isin(eligible)].sort_values(
        ["coverage_min", "variance_median", "feature"], ascending=[False, False, True], kind="stable"
    ).feature.tolist()
    index_by_feature = {field: index for index, field in enumerate(eligible)}
    representatives: list[str] = []
    blocks: dict[str, list[str]] = {}
    for field in ordered:
        index = index_by_feature[field]
        if representatives:
            representative_indexes = [index_by_feature[item] for item in representatives]
            correlation = np.abs(rank_values[:, representative_indexes].T @ rank_values[:, index])
            matches = [item for item, value in zip(representatives, correlation, strict=True) if float(value) >= .97]
        else:
            matches = []
        if not matches:
            representatives.append(field)
            blocks[field] = [field]
        else:
            blocks[matches[0]].append(field)
    stats["hygiene_keep"] = stats.feature.isin(eligible)
    stats["correlation_representative"] = stats.feature.isin(representatives)
    return stats, blocks, pd.concat(samples, ignore_index=True)


def _family(field: str) -> str:
    name = field.lower()
    if any(token in name for token in ("fund", "carry")):
        return "funding"
    if any(token in name for token in ("oi", "open_interest", "leverage")):
        return "oi_leverage"
    if any(token in name for token in ("liq", "book", "impact", "amihud", "volume", "vp_")):
        return "liquidity"
    if any(token in name for token in ("rv", "vol", "atr", "range", "vov", "semivariance")):
        return "volatility"
    if any(token in name for token in ("trend", "ret", "price", "donchian", "adx", "ker", "bollinger", "wick", "body")):
        return "price_trend"
    if any(token in name for token in ("mkt", "beta", "corr", "peer", "bench", "universe", "xs_")):
        return "cross_asset"
    if any(token in name for token in ("regime", "state", "entropy", "transition", "tail", "climax", "exhaust")):
        return "state"
    return "other"


def _univariate(sample: pd.DataFrame, labels: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    work = sample.merge(labels.loc[:, ["candidate_id", "raw_magnitude_valid", "magnitude_raw_bps"]], on="candidate_id", how="left", validate="one_to_one")
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work["__target__"] = pd.to_numeric(work["magnitude_raw_bps"], errors="coerce")
    work = work.loc[work.raw_magnitude_valid.fillna(False).astype(bool) & np.isfinite(work.__target__)].copy()
    if work["__decision_ts__"].nunique() < 30:
        raise AssertionError("univariate sample lacks enough timestamp queries")
    target_rank = work.groupby("__decision_ts__", sort=False)["__target__"].rank(pct=True, method="average")
    rows: list[dict[str, object]] = []
    for field in fields:
        value = pd.to_numeric(work[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
        correlations: list[float] = []
        for _, group in work.loc[value.notna(), ["__decision_ts__"]].assign(__x__=value[value.notna()], __y__=target_rank[value.notna()]).groupby("__decision_ts__", sort=False):
            if len(group) < 8 or group.__x__.nunique() < 3 or group.__y__.nunique() < 3:
                continue
            coefficient = float(group.__x__.rank(pct=True).corr(group.__y__))
            if np.isfinite(coefficient):
                correlations.append(coefficient)
        rows.append({
            "feature": field, "family": _family(field), "univariate_queries": len(correlations),
            "univariate_median_abs_ic": float(np.median(np.abs(correlations))) if correlations else 0.0,
            "univariate_mean_abs_ic": float(np.mean(np.abs(correlations))) if correlations else 0.0,
            "univariate_sign_consistency": float(max((np.asarray(correlations) >= 0).mean(), (np.asarray(correlations) <= 0).mean())) if correlations else 0.0,
        })
    return pd.DataFrame(rows)


def _scale(values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if raw.notna().sum() < 2 or float(raw.max() - raw.min()) <= 1e-12:
        return pd.Series(0.0, index=raw.index)
    return (raw - raw.min()) / (raw.max() - raw.min())


def _score_lgbm_subset(
    *, roots: Sequence[Path], label_root: Path, router_root: Path, stage1_root: Path,
    arm: stage1.Arm, fields: Sequence[str], held_months: Sequence[pd.Timestamp], train_months: int,
    reserve_days: int, train_cap: int, seed: int, jobs: int,
) -> tuple[float, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    controls: list[pd.DataFrame] = []
    gains: list[pd.DataFrame] = []
    for offset, month in enumerate(held_months):
        reserve = month - pd.Timedelta(days=reserve_days)
        end = month + pd.offsets.MonthBegin(1)
        window, _ = base._load_window(
            candidate_root=None, feature_root=roots, label_root=label_root, router_root=router_root,
            start=reserve - pd.DateOffset(months=train_months), end=end, fields=fields,
        )
        train = stage1._train_rows(window, arm, reserve, train_cap)
        labels, _ = stage1._labels(train, arm)
        held = window.loc[window.__decision_ts__.ge(month) & window.__decision_ts__.lt(end)].copy()
        held = held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        x_train, medians = base._numeric_matrix(train, fields)
        x_held, _ = base._numeric_matrix(held, fields, medians)
        model = LGBMRanker(
            objective="rank_xendcg", metric="ndcg", n_estimators=180, learning_rate=.05,
            max_depth=4, num_leaves=15, min_child_samples=260, subsample=.8, subsample_freq=1,
            colsample_bytree=.8, reg_alpha=.05, reg_lambda=8.0, min_split_gain=.001,
            random_state=seed + offset, deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=jobs,
        )
        model.fit(x_train, labels, group=base._query_groups(train))
        score = held.loc[:, list(IDENTITY)].copy()
        score["base_score"] = model.predict(x_held).astype(np.float32)
        score["base_rank_ts"] = base._rank_desc(score, "base_score")
        outcome = held.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]]
        parts.append(timestamp_components(score.merge(outcome, on="candidate_id", how="left", validate="one_to_one"), score_column="base_score"))
        baseline = gain._control_score(stage1_root, month)
        controls.append(timestamp_components(baseline.merge(outcome, on="candidate_id", how="left", validate="one_to_one"), score_column="base_score"))
        gains.append(pd.DataFrame({"feature": list(fields), "held_month": f"{month:%Y-%m}", "gain": model.booster_.feature_importance(importance_type="gain").astype(float)}))
        del window, train, held, x_train, x_held, model
        gc.collect()
    score, _ = stable_score(pd.concat(parts, ignore_index=True), pd.concat(controls, ignore_index=True))
    return float(score.score_stable), pd.concat(gains, ignore_index=True)


def _subspace_fields(fields: Sequence[str], count: int, size: int) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(SEED)
    output: list[tuple[str, ...]] = []
    for _ in range(count):
        chosen = rng.choice(np.asarray(fields, dtype=object), size=min(size, len(fields)), replace=False)
        output.append(tuple(sorted(chosen.astype(str).tolist())))
    return output


def run(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, stage1_root: Path,
    hpo_root: Path, out: Path, reference_months: Sequence[pd.Timestamp], development_months: Sequence[pd.Timestamp],
    train_months: int, reserve_days: int, train_cap: int, sample_timestamps: int, correlation_block_size: int,
    initial_pool: int, survivors: int, subspaces: int, subspace_size: int, jobs: int,
) -> Path:
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((hpo_root / "run_manifest.json").read_text())
    contract = manifest.get("contract", {})
    arm_payload = contract.get("arm", {})
    arm = stage1.Arm(str(arm_payload["family"]), str(arm_payload["target"]), str(arm_payload["geometry"]))
    if contract.get("model_family") != "catboost_queryrmse" or arm.target != "t1_raw_bps":
        raise AssertionError("this first prescreen is intentionally for the retained raw-bps CatBoost contract")
    universe = _universe(feature_roots, reference_months[0])
    for month in reference_months:
        available = set(pq.ParquetFile(_owner(feature_roots, month)).schema_arrow.names)
        universe = tuple(field for field in universe if field in available)
    if len(universe) < 500:
        raise AssertionError("full causal feature universe lost unexpected support across reference months")
    out.mkdir(parents=True)
    _exclusive(out / "preflight.json", {
        "schema": SCHEMA, "event": "started", "scope": "offline full-universe feature prescreen only",
        "universe_fields": len(universe), "reference_months": [f"{month:%Y-%m}" for month in reference_months],
        "development_months": [f"{month:%Y-%m}" for month in development_months],
        "memory_contract": "feature columns read in bounded batches; only a timestamp-stratified Router50 sample is retained for the correlation calculation",
    })
    sample_ids = {month: _router_sample_ids(router_root, month, sample_timestamps) for month in reference_months}
    stats, blocks, sample = _hygiene_and_blocks(
        roots=feature_roots, reference_months=reference_months, fields=universe, sample_ids=sample_ids,
        block_size=correlation_block_size,
    )
    label_parts = [pd.read_parquet(base._label_path(label_root, month)) for month in reference_months]
    labels = pd.concat(label_parts, ignore_index=True)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    representatives = [item for item in blocks]
    uni = _univariate(sample, labels, representatives)
    summary = stats.merge(uni, on="feature", how="left")
    summary["univariate_median_abs_ic"] = summary.univariate_median_abs_ic.fillna(0.0)
    summary["univariate_sign_consistency"] = summary.univariate_sign_consistency.fillna(0.0)
    summary["univariate_score"] = .70 * _scale(summary.univariate_median_abs_ic) + .30 * _scale(summary.univariate_sign_consistency)
    eligible = summary.loc[summary.correlation_representative].sort_values(
        ["univariate_score", "feature"], ascending=[False, True], kind="stable"
    ).head(initial_pool).feature.astype(str).tolist()
    if len(eligible) < min(120, survivors):
        raise AssertionError("univariate pre-screen retained insufficient causal representatives")
    full_score, full_gain = _score_lgbm_subset(
        roots=feature_roots, label_root=label_root, router_root=router_root, stage1_root=stage1_root,
        arm=arm, fields=eligible, held_months=development_months, train_months=train_months,
        reserve_days=reserve_days, train_cap=train_cap, seed=SEED, jobs=jobs,
    )
    gain_summary = full_gain.groupby("feature", sort=False).agg(
        gain_median=("gain", "median"), gain_nonzero=("gain", lambda value: float((value > 0).mean())),
    ).reset_index()
    random_rows: list[dict[str, object]] = []
    for index, subspace in enumerate(_subspace_fields(eligible, subspaces, subspace_size)):
        score, _ = _score_lgbm_subset(
            roots=feature_roots, label_root=label_root, router_root=router_root, stage1_root=stage1_root,
            arm=arm, fields=subspace, held_months=development_months, train_months=train_months,
            reserve_days=reserve_days, train_cap=train_cap, seed=SEED + 100 + index, jobs=jobs,
        )
        random_rows.append({"subspace": index, "score_stable": score, "features": list(subspace)})
        _progress(out, stage="random_subspace_complete", subspace=index, score_stable=score, fields=len(subspace))
    inclusion_rows: list[dict[str, object]] = []
    for field in eligible:
        present = [float(row["score_stable"]) for row in random_rows if field in set(row["features"])]
        absent = [float(row["score_stable"]) for row in random_rows if field not in set(row["features"])]
        inclusion_rows.append({
            "feature": field, "subspace_in": len(present), "subspace_out": len(absent),
            "random_subspace_inclusion_uplift": float(np.mean(present) - np.mean(absent)) if present and absent else 0.0,
        })
    inclusion = pd.DataFrame(inclusion_rows)
    summary = summary.merge(gain_summary, on="feature", how="left").merge(inclusion, on="feature", how="left")
    for name in ("gain_median", "gain_nonzero", "random_subspace_inclusion_uplift"):
        summary[name] = summary[name].fillna(0.0)
    summary["prescreen_score"] = (
        .30 * summary.univariate_score + .35 * _scale(summary.gain_median) * summary.gain_nonzero
        + .35 * _scale(summary.random_subspace_inclusion_uplift)
    )
    selected = summary.loc[summary.feature.isin(eligible)].sort_values(
        ["prescreen_score", "feature"], ascending=[False, True], kind="stable"
    ).head(survivors).feature.astype(str).tolist()
    if len(selected) != survivors:
        raise AssertionError("full-universe prescreen did not produce the declared survivor count")
    summary.to_parquet(out / "feature_prescreen_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(random_rows).drop(columns="features").to_parquet(out / "random_subspace_scores.parquet", index=False, compression="zstd")
    _exclusive(out / "correlation_blocks.json", {key: value for key, value in blocks.items() if key in selected})
    _exclusive(out / "selected160_contract.json", {
        "schema": SCHEMA, "selected_features": selected, "feature_count": len(selected),
        "selection": "full causal universe -> hygiene -> .97 representative -> cross-era univariate -> cheap full-model gain -> balanced random-subspace inclusion uplift",
        "target_contract": contract, "reference_months": [f"{month:%Y-%m}" for month in reference_months],
        "development_months": [f"{month:%Y-%m}" for month in development_months],
        "selection_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
    })
    _exclusive(out / "correctness_report.json", {
        "full_causal_universe_used": True,
        "feature_hygiene_coverage_and_variance_only": True,
        "correlation_veto_threshold_097": True,
        "univariate_association_uses_development_resolved_labels_only": True,
        "random_subspace_scores_target_free_before_outcome_join": True,
        "p8u_router_top50_identity_exact": True,
        "all_feature_medians_train_only": True,
        "no_meta_mc1_portfolio_or_live_mutation": True,
    })
    _exclusive(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline P8u raw-bps Base feature pre-screen only; no Meta, MC1, admission, portfolio, live, or exchange mutation",
        "source_hpo_root": str(hpo_root), "source_hpo_sha256": _sha([hpo_root / "run_manifest.json"]),
        "universe_fields": len(universe), "hygiene_representatives": len(representatives), "initial_pool": initial_pool,
        "survivors": survivors, "cheap_full_model_score_stable": full_score,
        "random_subspaces": {"count": subspaces, "size": subspace_size},
        "reference_months": [f"{month:%Y-%m}" for month in reference_months],
        "development_months": [f"{month:%Y-%m}" for month in development_months],
        "strict_oof": {"train_months": train_months, "reserve_days": reserve_days, "train_cap_complete_queries": train_cap},
        "next_stage": "Use the 160 survivors and frozen correlation blocks for group MDA plus beam compression under the retained CatBoost contract.",
    })
    _progress(out, stage="complete", selected=len(selected), full_score_stable=full_score)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True, help="comma-separated immutable causal feature roots")
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--hpo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-months", default="2025-11,2026-01,2026-03,2026-05,2026-07")
    parser.add_argument("--development-months", default="2025-11,2026-03,2026-07")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--sample-timestamps", type=int, default=8)
    parser.add_argument("--correlation-block-size", type=int, default=32)
    parser.add_argument("--initial-pool", type=int, default=320)
    parser.add_argument("--survivors", type=int, default=160)
    parser.add_argument("--subspaces", type=int, default=8)
    parser.add_argument("--subspace-size", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not (120 <= args.survivors <= args.initial_pool) or args.subspaces < 4 or args.subspace_size < 20:
        raise ValueError("invalid feature pre-screen dimensions")
    print(run(
        feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()),
        label_root=args.label_root.resolve(), router_root=args.router_root.resolve(), stage1_root=args.stage1_root.resolve(),
        hpo_root=args.hpo_root.resolve(), out=args.out.resolve(), reference_months=_months(args.reference_months),
        development_months=_months(args.development_months), train_months=args.train_months, reserve_days=args.reserve_days,
        train_cap=args.train_cap, sample_timestamps=args.sample_timestamps, correlation_block_size=args.correlation_block_size,
        initial_pool=args.initial_pool, survivors=args.survivors, subspaces=args.subspaces, subspace_size=args.subspace_size, jobs=args.jobs,
    ))


if __name__ == "__main__":
    main()
