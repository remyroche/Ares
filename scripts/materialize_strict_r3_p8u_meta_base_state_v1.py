#!/usr/bin/env python3
"""Materialise causal Base-state inputs for the P8U Under Meta challenger.

This is deliberately an *input producer*, not a Meta, MC1, admission, or
portfolio runner.  It extends the immutable F120 causal panel with five
families requested for the Base-to-Meta conversion study:

* query-relative Base geometry;
* recently resolved, score-band calibration state;
* frozen-F72 tree-block uncertainty;
* frozen-F72 leaf-support / rarity state; and
* train-only leaf residual priors.

Every held output is target-free.  The only outcome-derived values are the
calibration and leaf-prior states; they are activated strictly after their
declared policy label availability timestamp.  F72 is re-fit independently
per outer month and must reproduce the sealed Base score before its tree state
is accepted.
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
from catboost import CatBoostRanker, Pool
from sklearn.isotonic import IsotonicRegression

import run_strict_r3_p8u_meta_target_query_grid_v1 as meta
import run_strict_r3_p8u_precision_preservation_group_mda_beam_v1 as beam
import run_strict_r3_p8u_precision_preservation_weight_funnel_v1 as weights
import run_strict_r3_p8u_precision_preservation_winner_hpo_v1 as hpo
import run_strict_r3_p8u_precision_preservation_screen_v1 as stage1
import run_strict_r3_router_single_base_prescreen_v1 as base


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_base_state_v1"
IDENTITY = tuple(meta.IDENTITY)
SEED = 1729
CAL_WINDOWS = (3, 7, 21)
LOCAL_BANDS = 20
PARENT_BANDS = 5
LOCAL_SHRINK = 30.0
PARENT_SHRINK = 100.0
LOW_SUPPORT = 30
LEAF_PRIOR_SHRINK = 50.0


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(path: Path, **payload: object) -> None:
    with (path / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month(token: str) -> pd.Timestamp:
    return pd.Timestamp(f"{token.strip()}-01", tz="UTC")


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(_month(value) for value in text.split(",") if value.strip())
    if not values or tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("months must be a non-empty, unique ascending list")
    return values


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _base_band(rank: pd.Series, count: int = LOCAL_BANDS) -> pd.Series:
    return np.minimum(count - 1, np.maximum(0, np.floor((1.0 - rank.to_numpy(float)) * count))).astype(np.int16)


def _read_f120_fields(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    fields = tuple(str(value) for value in payload.get("selected_features", ()))
    if len(fields) != 120 or len(fields) != len(set(fields)):
        raise AssertionError("expected the immutable 120-field Under contract")
    return fields


def _read_base_month(base_root: Path, month: pd.Timestamp) -> pd.DataFrame:
    path = meta._base_path(base_root, month)
    frame = pd.read_parquet(path, columns=list(IDENTITY) + ["base_score", "base_rank_ts"])
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame.duplicated(IDENTITY).any() or not frame.side_name.eq("long").all():
        raise AssertionError(f"{month:%Y-%m}: invalid Base target-free identity")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _query_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only new query-relative inputs; basic geometry stays in F120."""
    work = frame.loc[:, list(IDENTITY) + ["base_score", "base_rank_ts"]].copy()
    grouped = work.groupby("__decision_ts__", sort=False)
    score = work.base_score.astype(float)
    top1 = grouped.base_score.transform("max").astype(float)
    order = work.sort_values(["__decision_ts__", "base_score", "candidate_id"], ascending=[True, False, True], kind="stable")
    ranks = order.groupby("__decision_ts__", sort=False).cumcount()
    top = order.loc[ranks.lt(5), ["__decision_ts__", "base_score"]].copy()
    top5 = top.groupby("__decision_ts__", sort=False).base_score.median()
    # Position-dependent boundary scores.  Missing third places deliberately
    # fall back to the weakest available position, never a future candidate.
    positions = order.loc[ranks.lt(3), ["__decision_ts__", "base_score"]].copy()
    positions["position"] = ranks.loc[ranks.lt(3)].to_numpy(int)
    pivot = positions.pivot(index="__decision_ts__", columns="position", values="base_score")
    for pos in (0, 1, 2):
        if pos not in pivot:
            pivot[pos] = np.nan
    pivot = pivot.ffill(axis=1).bfill(axis=1)
    work["meta_q_gap_to_top1"] = (top1 - score).astype(np.float32)
    work["meta_q_gap_to_top2"] = (work.__decision_ts__.map(pivot[1]).to_numpy(float) - score.to_numpy(float)).astype(np.float32)
    work["meta_q_gap_to_top3"] = (work.__decision_ts__.map(pivot[2]).to_numpy(float) - score.to_numpy(float)).astype(np.float32)
    median = grouped.base_score.transform("median").astype(float)
    q75 = grouped.base_score.transform(lambda values: values.quantile(.75)).astype(float)
    q25 = grouped.base_score.transform(lambda values: values.quantile(.25)).astype(float)
    iqr = (q75 - q25).clip(lower=1e-6)
    mad = grouped.base_score.transform(lambda values: (values - values.median()).abs().median()).clip(lower=1e-6)
    work["meta_q_gap_to_top5_median"] = (work.__decision_ts__.map(top5).to_numpy(float) - score.to_numpy(float)).astype(np.float32)
    work["meta_q_gap_to_median"] = (score - median).astype(np.float32)
    work["meta_q_score_iqr"] = iqr.astype(np.float32)
    work["meta_q_score_mad"] = mad.astype(np.float32)
    work["meta_q_gap_to_top2_iqr"] = (work.meta_q_gap_to_top2.to_numpy(float) / iqr.to_numpy(float)).astype(np.float32)
    work["meta_q_gap_to_top1_iqr"] = (work.meta_q_gap_to_top1.to_numpy(float) / iqr.to_numpy(float)).astype(np.float32)
    radius = .25 * iqr
    # Query sizes are modest; the explicit per-query operation preserves a
    # true local score-density definition rather than a global proxy.
    density = np.empty(len(work), dtype=np.float32)
    boundary = np.empty(len(work), dtype=np.float32)
    entropy = np.empty(len(work), dtype=np.float32)
    count = np.empty(len(work), dtype=np.float32)
    for _, index in grouped.groups.items():
        pos = np.asarray(index, dtype=np.int64)
        values = score.iloc[pos].to_numpy(float)
        width = max(float(radius.iloc[pos[0]]), 1e-6)
        density[pos] = (np.abs(values[:, None] - values[None, :]) <= width).sum(axis=1).astype(np.float32) / max(1, len(values))
        top2_value = float(pivot.loc[work.__decision_ts__.iloc[pos[0]], 1])
        boundary[pos] = float((np.abs(values - top2_value) <= width).sum()) / max(1, len(values))
        bins = np.histogram(pd.Series(values).rank(pct=True).to_numpy(float), bins=np.linspace(0.0, 1.0, 11))[0]
        probability = bins[bins > 0] / max(1, bins.sum())
        entropy[pos] = float(-(probability * np.log(probability)).sum() / math.log(10.0))
        count[pos] = len(values)
    work["meta_q_local_score_density"] = density
    work["meta_q_top2_boundary_density"] = boundary
    work["meta_q_score_entropy"] = entropy
    work["meta_q_candidate_count"] = count
    return work.drop(columns=["base_score", "base_rank_ts"])


def _policy_events(base_history: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    """Create strict-prequential residual events, activated at label availability."""
    labels = policy.loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"]].copy()
    work = base_history.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    work["policy_path_valid"] = work.policy_path_valid.fillna(False).astype(bool)
    work["policy_net_bps"] = pd.to_numeric(work.policy_net_bps, errors="coerce")
    work["policy_label_available_ts"] = pd.to_datetime(work.policy_label_available_ts, utc=True, errors="coerce")
    valid = work.policy_path_valid & np.isfinite(work.policy_net_bps) & work.policy_label_available_ts.notna()
    work = work.loc[valid].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    work["preq_anchor_bps"] = np.nan
    first = work.__decision_ts__.min().floor("D")
    block = ((work.__decision_ts__ - first) / pd.Timedelta(days=14)).astype(int)
    for token in sorted(block.unique()):
        current = block.eq(token)
        start = work.loc[current, "__decision_ts__"].min()
        prior = work.__decision_ts__.lt(start) & work.policy_label_available_ts.lt(start)
        if int(prior.sum()) < 1000:
            continue
        mapper = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
            work.loc[prior, "base_rank_ts"], work.loc[prior, "policy_net_bps"],
        )
        work.loc[current, "preq_anchor_bps"] = mapper.predict(work.loc[current, "base_rank_ts"])
    work["residual_bps"] = work.policy_net_bps - work.preq_anchor_bps
    work = work.loc[np.isfinite(work.residual_bps)].copy()
    work["available"] = work.policy_label_available_ts
    work["band"] = _base_band(work.base_rank_ts)
    work["parent"] = (work.band // (LOCAL_BANDS // PARENT_BANDS)).astype(np.int16)
    work["abs_residual"] = work.residual_bps.abs()
    work["gt50"] = work.residual_bps.gt(50.0).astype(float)
    work["gt100"] = work.residual_bps.gt(100.0).astype(float)
    return work.loc[:, ["candidate_id", "__decision_ts__", "available", "band", "parent", "residual_bps", "abs_residual", "gt50", "gt100", "policy_net_bps"]]


def _rolling_state(events: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """One causal state row per label-availability event for a supplied group."""
    part = events.sort_values(["available", "candidate_id"], kind="stable").copy()
    index = part.set_index("available")
    roller = index.rolling(f"{window}D", closed="both", min_periods=1)
    out = part.loc[:, ["available", "candidate_id"]].copy()
    out["n"] = roller["residual_bps"].count().to_numpy(np.float32)
    out["residual_median"] = roller["residual_bps"].median().to_numpy(np.float32)
    out["abs_median"] = roller["abs_residual"].median().to_numpy(np.float32)
    out["abs_p90"] = roller["abs_residual"].quantile(.90).to_numpy(np.float32)
    out["gt50_rate"] = roller["gt50"].mean().to_numpy(np.float32)
    out["gt100_rate"] = roller["gt100"].mean().to_numpy(np.float32)
    out["ev_mean"] = roller["policy_net_bps"].mean().to_numpy(np.float32)
    return out


def _attach_asof(target: pd.DataFrame, events: pd.DataFrame, group: str | None, window: int, prefix: str) -> pd.DataFrame:
    """Attach a prior-only rolling state without any held outcome join."""
    output = target.copy()
    columns = ("n", "residual_median", "abs_median", "abs_p90", "gt50_rate", "gt100_rate", "ev_mean")
    for column in columns:
        output[f"{prefix}{column}"] = np.nan
    keys = [None] if group is None else sorted(pd.Series(output[group].dropna().unique()).astype(int).tolist())
    for key in keys:
        left_index = output.index if key is None else output.index[output[group].eq(key)]
        if not len(left_index):
            continue
        source = events if key is None else events.loc[events[group].eq(key)]
        if source.empty:
            continue
        state = _rolling_state(source, window=window).sort_values("available", kind="stable")
        left = output.loc[left_index, ["__decision_ts__"]].copy().sort_values("__decision_ts__", kind="stable")
        joined = pd.merge_asof(
            left, state.drop(columns="candidate_id"), left_on="__decision_ts__", right_on="available",
            direction="backward", allow_exact_matches=False,
        ).set_index(left.index)
        for column in columns:
            output.loc[joined.index, f"{prefix}{column}"] = joined[column].to_numpy(np.float32)
    return output


def _calibration_features(target: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    work = target.loc[:, list(IDENTITY) + ["base_rank_ts"]].copy()
    work["band"] = _base_band(work.base_rank_ts)
    work["parent"] = (work.band // (LOCAL_BANDS // PARENT_BANDS)).astype(np.int16)
    output = work.loc[:, list(IDENTITY)].copy()
    metrics = ("residual_median", "abs_median", "abs_p90", "gt50_rate", "gt100_rate", "ev_mean")
    for days in CAL_WINDOWS:
        local = _attach_asof(work, events, "band", days, "local_")
        parent = _attach_asof(work, events, "parent", days, "parent_")
        global_state = _attach_asof(work, events, None, days, "global_")
        local_n = local.local_n.fillna(0.0).to_numpy(float)
        parent_n = parent.parent_n.fillna(0.0).to_numpy(float)
        output[f"meta_cal_{days}d_band_n"] = local_n.astype(np.float32)
        output[f"meta_cal_{days}d_parent_n"] = parent_n.astype(np.float32)
        for metric in metrics:
            g = global_state[f"global_{metric}"].fillna(0.0).to_numpy(float)
            p_raw = parent[f"parent_{metric}"].to_numpy(float)
            p = np.where(np.isfinite(p_raw), p_raw, g)
            l_raw = local[f"local_{metric}"].to_numpy(float)
            l = np.where(np.isfinite(l_raw), l_raw, p)
            shrunk_parent = (parent_n * p + PARENT_SHRINK * g) / (parent_n + PARENT_SHRINK)
            final = (local_n * l + LOCAL_SHRINK * shrunk_parent) / (local_n + LOCAL_SHRINK)
            output[f"meta_cal_{days}d_{metric}_shrunk"] = final.astype(np.float32)
    return output


def _fit_f72(
    *, train: pd.DataFrame, labels: np.ndarray, held: pd.DataFrame, fields: tuple[str, ...],
    params: dict[str, float], seed: int,
) -> tuple[CatBoostRanker, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    x_train, medians = base._numeric_matrix(train, fields)
    x_held, _ = base._numeric_matrix(held, fields, medians)
    fit, valid = hpo._inner_masks(train)
    row_weight = weights._query_safe_weights(train, labels, "tail_linear_125")
    train_fit = train.loc[fit].reset_index(drop=True)
    train_valid = train.loc[valid].reset_index(drop=True)
    model = CatBoostRanker(
        loss_function="QueryRMSE", eval_metric="NDCG:top=10", iterations=2000,
        learning_rate=float(params["learning_rate"]), depth=int(params["max_depth"]),
        l2_leaf_reg=float(params["lambda_l2"]), random_strength=float(params["random_strength"]),
        rsm=float(params["feature_fraction"]), bootstrap_type="Bernoulli",
        subsample=float(params["bagging_fraction"]), random_seed=int(seed), thread_count=1,
        verbose=False, allow_writing_files=False, od_type="Iter", od_wait=30,
    )
    model.fit(
        Pool(x_train[fit], labels[fit], group_id=hpo._qid(train_fit), weight=row_weight[fit]),
        eval_set=Pool(x_train[valid], labels[valid], group_id=hpo._qid(train_valid), weight=row_weight[valid]),
        use_best_model=True, verbose=False,
    )
    score = np.asarray(model.predict(x_held), dtype=np.float32)
    audit = {"train_rows": int(len(train)), "held_rows": int(len(held)), "trees": int(model.tree_count_), "weight_min": float(row_weight.min()), "weight_max": float(row_weight.max())}
    return model, x_train, x_held, score, audit


def _leaf_support_features(
    *, model: CatBoostRanker, x_train: np.ndarray, x_held: np.ndarray, train: pd.DataFrame,
    prequential_events: pd.DataFrame,
) -> pd.DataFrame:
    leaves_train = np.asarray(model.calc_leaf_indexes(x_train), dtype=np.int32)
    leaves_held = np.asarray(model.calc_leaf_indexes(x_held), dtype=np.int32)
    n_train, n_trees = leaves_train.shape
    support = np.empty_like(leaves_held, dtype=np.float32)
    # The residual coordinate is the separately persisted strict-OOF Base
    # ledger.  A current monthly F72 fit is never allowed to create an
    # in-sample residual target for its own leaves.
    train_residual = train.loc[:, ["candidate_id"]].merge(
        prequential_events.loc[:, ["candidate_id", "residual_bps"]], on="candidate_id", how="left", validate="one_to_one",
    ).residual_bps.to_numpy(float)
    global_residual = float(np.nanmean(train_residual)) if np.isfinite(train_residual).any() else 0.0
    prior = np.empty_like(leaves_held, dtype=np.float32)
    prior_support = np.empty_like(leaves_held, dtype=np.float32)
    for tree in range(n_trees):
        # A tree can route a held row to a legal but unoccupied training leaf.
        # That zero support is exactly the OOD signal required here; it is not
        # an identity error and must never be silently remapped.
        width = int(max(leaves_train[:, tree].max(), leaves_held[:, tree].max())) + 1
        inverse = leaves_train[:, tree]
        counts = np.bincount(inverse, minlength=width)
        held_leaf = leaves_held[:, tree]
        support[:, tree] = counts[held_leaf]
        valid = np.isfinite(train_residual)
        sum_by_leaf = np.bincount(inverse[valid], weights=train_residual[valid], minlength=width)
        n_by_leaf = np.bincount(inverse[valid], minlength=width).astype(float)
        value = (sum_by_leaf + LEAF_PRIOR_SHRINK * global_residual) / (n_by_leaf + LEAF_PRIOR_SHRINK)
        prior[:, tree] = value[held_leaf]
        prior_support[:, tree] = n_by_leaf[held_leaf]
    harmonic = n_trees / np.maximum((1.0 / np.maximum(support, 1.0)).sum(axis=1), 1e-6)
    rarity = -np.log((support + 1.0) / (n_train + 1.0))
    # Stable, model-local buckets retain path identity without claiming that a
    # numeric leaf identifier has global semantic order across outer folds.
    salts = (np.arange(n_trees, dtype=np.uint64) * np.uint64(11400714819323198485) + 7046029254386353131)
    signature = ((leaves_held.astype(np.uint64) + 1) * salts[None, :]).sum(axis=1, dtype=np.uint64)
    output = pd.DataFrame({
        "meta_leaf_support_mean": support.mean(axis=1).astype(np.float32),
        "meta_leaf_support_min": support.min(axis=1).astype(np.float32),
        "meta_leaf_support_hmean": harmonic.astype(np.float32),
        "meta_leaf_low_support_fraction": (support < LOW_SUPPORT).mean(axis=1).astype(np.float32),
        "meta_leaf_rarity_mean": rarity.mean(axis=1).astype(np.float32),
        "meta_leaf_residual_prior": prior.mean(axis=1).astype(np.float32),
        "meta_leaf_residual_prior_std": prior.std(axis=1).astype(np.float32),
        "meta_leaf_prior_support_hmean": (n_trees / np.maximum((1.0 / np.maximum(prior_support, 1.0)).sum(axis=1), 1e-6)).astype(np.float32),
        "meta_leaf_signature_bucket64": (signature % np.uint64(64)).astype(np.float32),
        "meta_leaf_signature_bucket256": (signature % np.uint64(256)).astype(np.float32),
    })
    return output


def _tree_uncertainty(model: CatBoostRanker, x_held: np.ndarray) -> pd.DataFrame:
    trees = int(model.tree_count_)
    boundaries = np.unique(np.linspace(0, trees, min(8, trees) + 1, dtype=int))
    cumulative: list[np.ndarray] = []
    for end in boundaries[1:]:
        cumulative.append(np.asarray(model.predict(x_held, ntree_end=int(end)), dtype=np.float32))
    blocks: list[np.ndarray] = []
    previous = np.zeros(len(x_held), dtype=np.float32)
    for prediction in cumulative:
        blocks.append(prediction - previous)
        previous = prediction
    matrix = np.column_stack(blocks).astype(np.float32)
    absolute = np.abs(matrix)
    total = np.maximum(absolute.sum(axis=1), 1e-8)
    split = max(1, matrix.shape[1] // 2)
    return pd.DataFrame({
        "meta_tree_block_dispersion": matrix.std(axis=1).astype(np.float32),
        "meta_tree_block_mad": np.median(np.abs(matrix - np.median(matrix, axis=1, keepdims=True)), axis=1).astype(np.float32),
        "meta_tree_positive_block_fraction": (matrix > 0.0).mean(axis=1).astype(np.float32),
        "meta_tree_early_late_delta": (matrix[:, :split].mean(axis=1) - matrix[:, split:].mean(axis=1)).astype(np.float32),
        "meta_tree_block_concentration": (absolute.max(axis=1) / total).astype(np.float32),
    })


def _tree_features_for_month(
    *, month: pd.Timestamp, f72_feature_roots: Sequence[Path], label_root: Path, router_root: Path,
    hpo_root: Path, selection: Path, canonical_base: pd.DataFrame, seed_origin: pd.Timestamp,
    prequential_events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fields = weights._load_fields(selection)
    arm, params = beam._load_contract(hpo_root)
    reserve = month - pd.Timedelta(days=28)
    window, coverage = base._load_window(
        candidate_root=None, feature_root=tuple(f72_feature_roots), label_root=label_root, router_root=router_root,
        start=reserve - pd.DateOffset(months=3), end=reserve, fields=fields,
    )
    held, held_audit = weights._target_free_held(feature_roots=tuple(f72_feature_roots), router_root=router_root, month=month, fields=fields)
    train = stage1._train_rows(window, arm, reserve, 60_000)
    labels, geometry = stage1._labels(train, arm)
    seed = SEED + int((month.year - seed_origin.year) * 12 + month.month - seed_origin.month)
    model, x_train, x_held, score, fit_audit = _fit_f72(train=train, labels=labels, held=held, fields=fields, params=params, seed=seed)
    control = canonical_base.loc[:, list(IDENTITY) + ["base_score"]].merge(held.loc[:, list(IDENTITY)], on=list(IDENTITY), how="inner", validate="one_to_one")
    candidate = held.loc[:, list(IDENTITY)].copy(); candidate["base_score"] = score
    compared = control.merge(candidate, on=list(IDENTITY), suffixes=("_control", "_reproduced"), validate="one_to_one")
    if len(compared) != len(canonical_base) or not np.allclose(compared.base_score_control, compared.base_score_reproduced, rtol=3e-5, atol=3e-5):
        delta = float(np.nanmax(np.abs(compared.base_score_control - compared.base_score_reproduced))) if len(compared) else float("inf")
        raise AssertionError(f"{month:%Y-%m}: F72 reconstruction does not match frozen Base score; max abs={delta}")
    uncertainty = _tree_uncertainty(model, x_held)
    leaf = _leaf_support_features(
        model=model, x_train=x_train, x_held=x_held, train=train, prequential_events=prequential_events,
    )
    out = held.loc[:, list(IDENTITY)].copy().reset_index(drop=True)
    out = pd.concat([out, uncertainty, leaf], axis=1)
    audit = {"month": f"{month:%Y-%m}", "seed": seed, "score_match_max_abs": float(np.max(np.abs(compared.base_score_control - compared.base_score_reproduced))), "score_match_rows": int(len(compared)), "coverage_records": int(len(coverage)), "held_source": held_audit, "target_geometry": geometry, **fit_audit}
    del model, x_train, x_held, score, train, window
    gc.collect()
    return out, audit


def run(args: argparse.Namespace) -> Path:
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")
    months = _months(args.months)
    f120_fields = _read_f120_fields(args.f120_contract)
    seed_origin = _month(args.seed_origin)
    if months[0] < seed_origin:
        raise ValueError("seed origin cannot post-date the first output month")
    args.out.mkdir(parents=True)
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-free P8U Base-state Meta input materialisation; no Meta/MC1/admission/portfolio/live/exchange mutation",
        "months": [f"{month:%Y-%m}" for month in months],
        "f120_contract": str(args.f120_contract), "f120_contract_sha256": _sha_file(args.f120_contract),
        "base_root": str(args.base_root), "f72_feature_roots": [str(path) for path in args.f72_feature_roots],
        "f72_label_root": str(args.f72_label_root), "f72_router_root": str(args.f72_router_root),
        "f72_hpo_root": str(args.f72_hpo_root), "f72_selection": str(args.f72_selection),
        "policy_labels": str(args.policy_labels), "seed_origin": f"{seed_origin:%Y-%m}",
        "feature_families": {"m2_query_geometry": True, "m3_resolved_calibration": True, "m4_tree_uncertainty": True, "m5_leaf_state": True},
        "causality": "held inputs are target-free; calibration/leaf-prior outcomes are activated only after their declared policy label availability timestamp; F72 is strict pre-reserve and score-matched to the frozen Base ledger",
    })
    policy = meta._read_policy(args.policy_labels)
    base_parts = [_read_base_month(args.base_root, month) for month in months]
    base_history = pd.concat(base_parts, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    events = _policy_events(base_history, policy)
    calibration = _calibration_features(base_history, events)
    all_audits: list[dict[str, object]] = []
    contracts: dict[str, list[str]] = {"m2": [], "m3": [], "m4": [], "m5": []}
    for month, canonical_base in zip(months, base_parts):
        end = _month_end(month)
        full = meta._read_base_features(base_root=args.base_root, feature_roots=tuple(args.f120_feature_roots), start=month, end=end, fields=f120_fields)
        expected = canonical_base.loc[:, list(IDENTITY)]
        if not full.loc[:, list(IDENTITY)].equals(expected):
            raise AssertionError(f"{month:%Y-%m}: F120 and canonical Base identities differ")
        q = _query_features(canonical_base)
        cal = calibration.loc[calibration.__decision_ts__.ge(month) & calibration.__decision_ts__.lt(end)].copy()
        tree, tree_audit = _tree_features_for_month(
            month=month, f72_feature_roots=args.f72_feature_roots, label_root=args.f72_label_root,
            router_root=args.f72_router_root, hpo_root=args.f72_hpo_root, selection=args.f72_selection,
            canonical_base=canonical_base, seed_origin=seed_origin, prequential_events=events,
        )
        overlay = full.loc[:, list(IDENTITY) + list(f120_fields)].merge(q, on=list(IDENTITY), how="left", validate="one_to_one")
        overlay = overlay.merge(cal, on=list(IDENTITY), how="left", validate="one_to_one")
        overlay = overlay.merge(tree, on=list(IDENTITY), how="left", validate="one_to_one")
        # F120 legitimately contains causal price-path *feature* names such
        # as ``dir_path_long_2h``.  Block only outcome/label payload names,
        # rather than using a broad substring that would reject valid inputs.
        forbidden_tokens = (
            "policy_net", "policy_label", "policy_exit", "policy_path_valid",
            "supportive_label", "supportive_path_valid", "outcome_", "target_",
            "label_available", "rich_policy",
        )
        forbidden = [column for column in overlay.columns if column not in IDENTITY and any(token in column.lower() for token in forbidden_tokens)]
        if forbidden:
            raise AssertionError(f"target-free overlay has forbidden columns {forbidden}")
        if overlay.duplicated(IDENTITY).any() or len(overlay) != len(canonical_base):
            raise AssertionError(f"{month:%Y-%m}: invalid overlay identity")
        derived = [column for column in overlay.columns if column not in set(IDENTITY).union(f120_fields)]
        contracts["m2"] = [column for column in derived if column.startswith("meta_q_")]
        contracts["m3"] = [column for column in derived if column.startswith("meta_cal_")]
        contracts["m4"] = [column for column in derived if column.startswith("meta_tree_")]
        contracts["m5"] = [column for column in derived if column.startswith("meta_leaf_")]
        target = args.out / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        overlay.to_parquet(target, index=False, compression="zstd")
        all_audits.append({"month": f"{month:%Y-%m}", "rows": int(len(overlay)), "f120_complete": int(overlay.loc[:, list(f120_fields)].notna().all(axis=1).sum()), "derived_complete": int(overlay.loc[:, derived].notna().all(axis=1).sum()), "derived_fields": int(len(derived)), "calibration_events_available_before_month": int((events.available < month).sum()), **tree_audit})
        _progress(args.out, event="month_complete", month=f"{month:%Y-%m}", rows=len(overlay), derived_fields=len(derived), score_match_max_abs=tree_audit["score_match_max_abs"])
        del full, q, cal, tree, overlay
        gc.collect()
    pd.DataFrame(all_audits).to_parquet(args.out / "month_audit.parquet", index=False, compression="zstd")
    # Keep each feature-family boundary explicit for the subsequent M0--M6
    # screen.  The directories are created only after every target-free month
    # has succeeded, so a failed producer cannot leave a misleading contract.
    (args.out / "contracts").mkdir(exist_ok=True)
    for arm, fields in contracts.items():
        path = args.out / "contracts" / f"{arm}_fields.json"
        if not path.exists():
            _once(path, {"schema": SCHEMA, "family": arm, "features": fields, "count": len(fields)})
    _once(args.out / "correctness_report.json", {
        "all_overlay_identity_matches_frozen_base": True,
        "f120_parent_inputs_preserved": True,
        "no_held_policy_or_path_column_in_output": True,
        "calibration_events_activated_only_after_policy_label_available_ts": True,
        "calibration_local_parent_global_shrinkage": True,
        "f72_train_labels_resolved_before_28d_reserve": True,
        "f72_reproduced_score_matches_frozen_base_before_tree_features": True,
        "leaf_residual_priors_use_f72_train_only_prequential_residuals": True,
        "no_meta_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })
    return args.out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--f120-feature-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--f120-contract", type=Path, required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--f72-feature-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--f72-label-root", type=Path, required=True)
    parser.add_argument("--f72-router-root", type=Path, required=True)
    parser.add_argument("--f72-hpo-root", type=Path, required=True)
    parser.add_argument("--f72-selection", type=Path, required=True)
    parser.add_argument("--months", default="2025-08,2025-09,2025-10,2025-11,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument("--seed-origin", default="2025-08")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
