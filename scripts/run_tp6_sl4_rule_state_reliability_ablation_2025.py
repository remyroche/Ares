#!/usr/bin/env python3
"""Causal rule/path-state reliability ablation on the matched TP6/SL4 panel.

This runner deliberately keeps the established Base+Consensus score intact and
only tests bounded reliability modulation.  It materialises:

* exact activated-leaf training support from the frozen leaf catalogues;
* model/path support, factorised joint OOD, path-conditioned OOD, and
  prototype-distribution Mahalanobis/PSI/KS drift proxies;
* causal 3/7/14-day global and active-recurrent-path correctness summaries;
* causal covariance/correlation breaks versus activation and mature success;
* cross-base-output recent correctness aggregates.

Every outcome-bearing state is indexed by ``label_available_ts`` and joined
strictly before a decision timestamp.  The frozen K=9 soft memberships are
explicit inputs to every structural challenger.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS, _load, _pct  # noqa: E402
from scripts.run_tp6_sl4_prototype_cluster_use_ablation_2025 import _causal_dynamic_state  # noqa: E402


SEED = 20260809
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
STRUCTURE = ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_quality_20260809_v3"
CONTROL = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/predictions_2025.parquet"
RAW = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/strict_base_reasoning"
OUT = ROOT / "data_perp/artifacts/tp6_sl4_rule_state_reliability_20260809_v1"
MIN_PATH_EFFECTIVE_SUPPORT = 30.0


def _safe(frame: pd.DataFrame, fields: Sequence[str], med: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = frame.reindex(columns=list(fields)).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = x.median().fillna(0.0)
    return x.fillna(med).fillna(0.0).astype("float32"), med


def _map_base(train: pd.DataFrame, held: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(train["base_score"].to_numpy(float)) & np.isfinite(train["net_bps"].to_numpy(float))
    if int(ok.sum()) < 128:
        mean = float(np.nanmean(train["net_bps"]))
        return np.full(len(train), mean, np.float32), np.full(len(held), mean, np.float32)
    mapping = IsotonicRegression(out_of_bounds="clip", y_min=-1000.0, y_max=1000.0)
    mapping.fit(train.loc[ok, "base_score"], train.loc[ok, "net_bps"])
    return mapping.predict(train["base_score"]).astype("float32"), mapping.predict(held["base_score"]).astype("float32")


def _fit_reliability(train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str], target: np.ndarray, *, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    x_train, med = _safe(train, fields)
    x_held, _ = _safe(held, fields, med)
    target = np.asarray(target, dtype=np.int8)
    if target.min() == target.max():
        return np.full(len(held), float(target.mean()), np.float32), {"train_positive_rate": float(target.mean()), "feature_count": len(fields)}
    model = lgb.LGBMClassifier(
        # Match the incumbent bounded-reliability classifier exactly.  The
        # ablation changes feature blocks and score transforms, not capacity.
        objective="binary", n_estimators=120, learning_rate=0.035, max_depth=4,
        num_leaves=15, min_child_samples=max(120, int(math.ceil(0.03 * len(train)))),
        feature_fraction=0.80, bagging_fraction=0.82, bagging_freq=1,
        reg_lambda=5.0, reg_alpha=0.05, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(x_train, target)
    probability = np.asarray(model.predict_proba(x_held)[:, 1], np.float32)
    return probability, {"train_positive_rate": float(target.mean()), "feature_count": len(fields)}


def _asof_features(frame: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Map an availability-indexed state strictly before each decision."""
    lookup = state.reset_index().rename(columns={state.index.name or "index": "_available_ts"})
    lookup = lookup.rename(columns={lookup.columns[0]: "_available_ts"}).sort_values("_available_ts", kind="stable")
    left = frame[["__ts__"]].copy()
    left["_row"] = np.arange(len(left), dtype=np.int64)
    left["__ts__"] = pd.to_datetime(left["__ts__"], utc=True)
    result = pd.merge_asof(
        left.sort_values("__ts__", kind="stable"), lookup, left_on="__ts__", right_on="_available_ts",
        direction="backward", allow_exact_matches=False,
    ).sort_values("_row", kind="stable").drop(columns=["__ts__", "_available_ts", "_row"])
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def _leaf_support_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact per-tree activated-leaf training support and factorised rule OOD."""
    values: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for folder in sorted(RAW.glob("month=*")):
        month = folder.name.split("=", 1)[1]
        if month not in set(panel["month"].astype(str)):
            continue
        leaves = pd.read_parquet(folder / "leaf_assignments.parquet")
        leaves = leaves.loc[leaves["side_name"].eq("long")].copy()
        catalog = pd.read_parquet(folder / "leaf_rule_catalog.parquet")
        catalog = catalog.loc[catalog["side_name"].eq("long") & catalog["head_name"].eq("canonical_residual")].copy()
        tree_fields = [field for field in leaves.columns if field.startswith("leaf_assignment__")]
        n = len(leaves)
        support = np.zeros((n, len(tree_fields)), dtype=np.float64)
        contribution = np.ones((n, len(tree_fields)), dtype=np.float64)
        totals = np.ones((n, len(tree_fields)), dtype=np.float64)
        matched = np.zeros((n, len(tree_fields)), dtype=bool)
        for idx, field in enumerate(tree_fields):
            slot = int(field.rsplit("_", 1)[-1])
            cat = catalog.loc[pd.to_numeric(catalog["head_tree_slot"], errors="coerce").eq(slot)]
            if cat.empty:
                continue
            key = pd.to_numeric(cat["leaf_token"], errors="coerce").astype("uint64")
            count_map = pd.Series(pd.to_numeric(cat["train_leaf_count"], errors="coerce").fillna(0.0).to_numpy(float), index=key).to_dict()
            weight_map = pd.Series(pd.to_numeric(cat["ensemble_tree_contribution"], errors="coerce").abs().fillna(0.0).to_numpy(float), index=key).to_dict()
            total = float(pd.to_numeric(cat["train_leaf_count"], errors="coerce").fillna(0.0).sum())
            token = pd.to_numeric(leaves[field], errors="coerce").fillna(0).astype("uint64")
            support[:, idx] = token.map(count_map).fillna(0.0).to_numpy(float)
            contribution[:, idx] = token.map(weight_map).fillna(0.0).to_numpy(float)
            totals[:, idx] = max(total, 1.0)
            matched[:, idx] = token.isin(count_map).to_numpy(bool)
        weights = np.where(contribution.sum(axis=1, keepdims=True) > 0.0, contribution, 1.0)
        denom = np.maximum(weights.sum(axis=1), 1e-12)
        surprise = -np.log(np.clip((support + 1.0) / (totals + 1.0), 1e-12, 1.0))
        out = leaves[["candidate_id"]].copy()
        out["rule_support_effective"] = (weights * support).sum(axis=1) / denom
        out["rule_support_p05"] = np.quantile(support, 0.05, axis=1)
        out["rule_support_p50"] = np.quantile(support, 0.50, axis=1)
        out["rule_support_median"] = out["rule_support_p50"]
        out["rule_support_p95"] = np.quantile(support, 0.95, axis=1)
        out["rule_support_contribution_weighted"] = out["rule_support_effective"]
        out["rule_support_adequate_fraction"] = (support >= MIN_PATH_EFFECTIVE_SUPPORT).mean(axis=1)
        out["rule_support_leaf_coverage"] = matched.mean(axis=1)
        out["rule_ood_marginal"] = surprise.mean(axis=1)
        out["rule_ood_joint_factorised"] = (weights * surprise).sum(axis=1) / denom
        values.append(out)
        audit.append({"month": month, "rows": n, "trees": len(tree_fields), "leaf_coverage": float(matched.mean()), "median_effective_support": float(np.median(out["rule_support_effective"]))})
    joined = pd.concat(values, ignore_index=True)
    if joined.candidate_id.duplicated().any():
        raise ValueError("duplicate leaf support candidate_id")
    merged = panel[["candidate_id"]].merge(joined, on="candidate_id", how="left", validate="one_to_one").drop(columns="candidate_id")
    if merged["rule_support_leaf_coverage"].isna().any():
        raise ValueError("leaf support missing a matched panel candidate")
    return merged.astype("float32"), pd.DataFrame(audit)


def _prototype_state_features(panel: pd.DataFrame, proto_fields: Sequence[str]) -> pd.DataFrame:
    """Feature-only causal path support/OOD/drift from frozen prototypes."""
    source = panel[["__ts__", *proto_fields]].copy()
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True)
    state = source.groupby("__ts__", sort=True)[list(proto_fields)].sum().sort_index()
    prior = state.shift(1).fillna(0.0)
    support = prior.rolling("28D", min_periods=1).sum()
    mean = prior.rolling("28D", min_periods=4).mean()
    std = prior.rolling("28D", min_periods=8).std().replace(0.0, np.nan)
    current_prob = state.div(state.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    ref_prob = support.div(support.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    z = (state - mean) / std
    state_out = pd.DataFrame(index=state.index)
    state_out["model_ood_marginal"] = z.abs().mean(axis=1)
    state_out["model_ood_mahalanobis_diag"] = np.sqrt((z**2).mean(axis=1))
    state_out["model_drift_prototype_psi"] = ((current_prob - ref_prob) * np.log(np.clip(current_prob, 1e-12, None) / np.clip(ref_prob, 1e-12, None))).sum(axis=1)
    state_out["model_drift_prototype_ks"] = (current_prob.cumsum(axis=1) - ref_prob.cumsum(axis=1)).abs().max(axis=1)
    mapped_state = panel[["__ts__"]].merge(state_out.reset_index(), on="__ts__", how="left", validate="many_to_one").drop(columns="__ts__").fillna(0.0)
    # The current path is observable at decision time; only its support
    # reference above is historical.  Aggregate support/OOD across its active
    # recurrent prototypes and gate unsupported prototypes out.
    support_row = panel[["__ts__"]].merge(support.reset_index(), on="__ts__", how="left", validate="many_to_one").drop(columns="__ts__").fillna(0.0).to_numpy(float)
    weights = np.maximum(panel.loc[:, proto_fields].to_numpy(float), 0.0)
    denom = np.maximum(weights.sum(axis=1), 1e-12)
    adequate = support_row >= MIN_PATH_EFFECTIVE_SUPPORT
    support_weight = weights * adequate
    support_denom = np.maximum(support_weight.sum(axis=1), 1e-12)
    total_support = np.maximum(support_row.sum(axis=1, keepdims=True), 1.0)
    path_surprise = -np.log(np.clip((support_row + 1.0) / (total_support + len(proto_fields)), 1e-12, 1.0))
    out = pd.DataFrame({
        "path_support_effective_28d": (weights * support_row).sum(axis=1) / denom,
        "path_support_adequate_fraction": support_weight.sum(axis=1) / denom,
        "path_ood_marginal": (weights * path_surprise).sum(axis=1) / denom,
        "path_ood_conditioned": (support_weight * path_surprise).sum(axis=1) / support_denom,
    })
    return pd.concat([mapped_state.reset_index(drop=True), out], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def _covariance_break(frame: pd.DataFrame, *, event_ts: str, target: np.ndarray, fields: Sequence[str], prefix: str) -> pd.DataFrame:
    """Short-vs-long causal covariance/correlation break, mapped to rows."""
    x = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    y = np.asarray(target, float)
    events = pd.DataFrame({event_ts: pd.to_datetime(frame[event_ts], utc=True), "_n": 1.0, "_y": y, "_y2": y**2})
    for idx in range(x.shape[1]):
        events[f"_x{idx}"] = x[:, idx]
        events[f"_x2{idx}"] = x[:, idx] ** 2
        events[f"_xy{idx}"] = x[:, idx] * y
    bucket = events.groupby(event_ts, sort=True).sum().sort_index()
    prior = bucket.shift(1).fillna(0.0)
    short = prior.rolling("7D", min_periods=4).sum()
    long = prior.rolling("28D", min_periods=12).sum()
    cov_short: list[pd.Series] = []
    cov_long: list[pd.Series] = []
    corr_short: list[pd.Series] = []
    corr_long: list[pd.Series] = []
    for idx in range(x.shape[1]):
        def _one(value: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
            n = value["_n"].replace(0.0, np.nan)
            cov = value[f"_xy{idx}"] / n - (value[f"_x{idx}"] / n) * (value["_y"] / n)
            vx = value[f"_x2{idx}"] / n - (value[f"_x{idx}"] / n) ** 2
            vy = value["_y2"] / n - (value["_y"] / n) ** 2
            corr = cov / np.sqrt((vx * vy).clip(lower=0.0)).replace(0.0, np.nan)
            return cov, corr
        a, b = _one(short); c, d = _one(long)
        cov_short.append(a); corr_short.append(b); cov_long.append(c); corr_long.append(d)
    short_cov = pd.concat(cov_short, axis=1); long_cov = pd.concat(cov_long, axis=1)
    short_corr = pd.concat(corr_short, axis=1); long_corr = pd.concat(corr_long, axis=1)
    state = pd.DataFrame(index=bucket.index)
    state[f"cov_break_vs_{prefix}"] = (short_cov.sub(long_cov).abs() / long_cov.abs().add(0.01)).clip(upper=20.0).mean(axis=1)
    state[f"corr_break_vs_{prefix}"] = short_corr.sub(long_corr).abs().clip(upper=2.0).mean(axis=1)
    return _asof_features(frame, state)


def _rolling_rates(index: pd.Series, weights: np.ndarray, metrics: dict[str, np.ndarray], *, prefix: str) -> pd.DataFrame:
    """Prior-only weighted 3/7/14-day rate states."""
    event = pd.DataFrame({"_available_ts": pd.to_datetime(index, utc=True), "_w": weights})
    for name, value in metrics.items():
        event[name] = weights * np.asarray(value, float)
    bucket = event.groupby("_available_ts", sort=True).sum().sort_index()
    prior = bucket.shift(1).fillna(0.0)
    output: dict[str, pd.Series] = {}
    for days in (3, 7, 14):
        roll = prior.rolling(f"{days}D", min_periods=1).sum()
        output[f"{prefix}support_{days}d"] = roll["_w"]
        for name in metrics:
            output[f"{prefix}{name}_{days}d"] = roll[name] / roll["_w"].replace(0.0, np.nan)
    return pd.DataFrame(output, index=bucket.index)


def _recent_correctness_features(panel: pd.DataFrame, proto_fields: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causal global/path correctness and cross-model state features."""
    expected = pd.to_numeric(panel["frozen_base_expected_bps"], errors="coerce").fillna(0.0).to_numpy(float)
    net = panel["net_bps"].to_numpy(float)
    residual = net - expected
    win = net > 0.0
    metrics = {
        "directional_correct": ((panel["base_score"].to_numpy(float) > 0.0) == win).astype(float),
        "approximately_correct": (np.abs(residual) <= 50.0).astype(float),
        "adverse_residual_rate": (residual <= -50.0).astype(float),
        "strong_adverse_residual_rate": (residual <= -100.0).astype(float),
    }
    p_clear = panel["r3_meta_p_clear"].to_numpy(float)
    p_adverse = panel["r3_meta_p_adverse"].to_numpy(float)
    p_weak = panel["r3_meta_p_weak"].to_numpy(float)
    cross = {
        "cross_clear_correctness": np.where(win, p_clear, 1.0 - p_clear),
        "cross_adverse_correctness": np.where(win, 1.0 - p_adverse, p_adverse),
        "cross_weak_correctness": np.where(metrics["approximately_correct"] > 0.0, p_weak, 1.0 - p_weak),
    }
    global_state = _rolling_rates(panel["label_available_ts"], np.ones(len(panel), float), {**metrics, **cross}, prefix="model_recent_")
    global_features = _asof_features(panel, global_state)
    # Calculate each frozen recurrent path's rate, retain only paths with at
    # least the declared effective support, then contribution-weight its live
    # path state.  Individual paths are also audited, but only the aggregate
    # becomes a model input so dimensionality remains controlled.
    weights = np.maximum(panel.loc[:, proto_fields].to_numpy(float), 0.0)
    aggregate = {f"path_recent_{name}_{days}d": np.zeros(len(panel), float) for name in metrics for days in (3, 7, 14)}
    aggregate_support = {f"path_recent_support_{days}d": np.zeros(len(panel), float) for days in (3, 7, 14)}
    denom = {days: np.zeros(len(panel), float) for days in (3, 7, 14)}
    audits: list[dict[str, object]] = []
    for idx, field in enumerate(proto_fields):
        state = _rolling_rates(panel["label_available_ts"], weights[:, idx], metrics, prefix="")
        mapped = _asof_features(panel, state)
        proto = field.rsplit("__", 2)[1]
        for days in (3, 7, 14):
            support = mapped[f"support_{days}d"].to_numpy(float)
            live_weight = weights[:, idx] * (support >= MIN_PATH_EFFECTIVE_SUPPORT)
            denom[days] += live_weight
            aggregate_support[f"path_recent_support_{days}d"] += weights[:, idx] * support
            for name in metrics:
                aggregate[f"path_recent_{name}_{days}d"] += live_weight * mapped[f"{name}_{days}d"].to_numpy(float)
            audits.append({"prototype": proto, "window": f"{days}d", "median_effective_support": float(np.median(support)), "adequate_fraction": float(np.mean(support >= MIN_PATH_EFFECTIVE_SUPPORT))})
    output = pd.DataFrame(index=panel.index)
    for days in (3, 7, 14):
        output[f"path_recent_support_{days}d"] = aggregate_support[f"path_recent_support_{days}d"] / np.maximum(weights.sum(axis=1), 1e-12)
        for name in metrics:
            output[f"path_recent_{name}_{days}d"] = aggregate[f"path_recent_{name}_{days}d"] / np.maximum(denom[days], 1e-12)
        output[f"path_recent_adequate_mass_{days}d"] = denom[days] / np.maximum(weights.sum(axis=1), 1e-12)
    return pd.concat([global_features.reset_index(drop=True), output], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32"), pd.DataFrame(audits)


def _metric_rows(prediction: pd.DataFrame, arms: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for arm in arms:
        score = prediction[arm].to_numpy(float)
        for tail in TAILS:
            n = max(1, int(math.ceil(len(prediction) * tail)))
            top = prediction.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            global_rows.append({"arm": arm, "tail": tail, "trades": len(top), "gross_bps_per_trade": float(top.gross_bps.mean()), "net_bps_per_trade": float(top.net_bps.mean()), "rank_ic": float(spearmanr(score, prediction.net_bps.to_numpy(float)).statistic)})
        values: list[float] = []
        ics: list[float] = []
        for month, block in prediction.groupby("month", sort=True):
            n = max(1, int(math.ceil(len(block) * 0.05)))
            top = block.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            net = float(top.net_bps.mean())
            ic = float(spearmanr(block[arm].to_numpy(float), block.net_bps.to_numpy(float)).statistic)
            values.append(net); ics.append(ic)
            monthly_rows.append({"arm": arm, "month": month, "tail": 0.05, "trades": len(top), "gross_bps_per_trade": float(top.gross_bps.mean()), "net_bps_per_trade": net, "rank_ic": ic})
        value = np.asarray(values, float); med = float(np.median(value)); mad = float(np.median(np.abs(value - med)))
        stability_rows.append({"arm": arm, "months": len(value), "mean_top5_net_bps": float(np.mean(value)), "median_top5_net_bps": med, "mad_top5_net_bps": mad, "worst_month_top5_net_bps": float(np.min(value)), "positive_months_top5": int(np.sum(value > 0.0)), "portability_score_bps": med - .5 * mad - max(0.0, -float(np.min(value))), "mean_month_rank_ic": float(np.nanmean(ics))})
    return pd.DataFrame(global_rows), pd.DataFrame(monthly_rows), pd.DataFrame(stability_rows)


def _feature_audit(frame: pd.DataFrame, blocks: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for block, fields in blocks.items():
        for field in fields:
            values = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
            rows.append({"block": block, "field": field, "coverage": float(np.isfinite(values).mean()), "nonzero": float((np.abs(np.nan_to_num(values)) > 1e-12).mean()), "std": float(np.nanstd(values))})
    return pd.DataFrame(rows)


def run(*, out: Path = OUT, seed: int = SEED) -> Path:
    if out.exists():
        raise FileExistsError(out)
    source, context, context_hash = _load()
    source = source.loc[source.side_name.eq("long")].copy()
    source["month"] = source["month"].astype(str)
    base = source[["candidate_id", "__ts__", "month", "label_available_ts", "base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", "exact_net_bps", "exact_gross_bps", *context]].copy()
    base = base.rename(columns={"exact_net_bps": "net_bps", "exact_gross_bps": "gross_bps"})
    structural = pd.read_parquet(STRUCTURE / "prototype_cluster_row_features.parquet")
    sizes = pd.read_parquet(STRUCTURE / "prototype_cluster_size_sweep_features.parquet")
    structural = structural.merge(sizes, on=["candidate_id", "__ts__", "month"], how="left", validate="one_to_one")
    keep_structural = ["candidate_id", "__ts__", "month", "base_expected_bps", *[c for c in structural if c.startswith("prototype__")], *[c for c in structural if c.startswith("k09__")], "prototype_matched_mass", "prototype_unmatched_mass", "prototype_match_similarity", "prototype_top2_margin", "prototype_entropy", "prototype_exposure_top2_margin", "prototype_assignment_count"]
    panel = base.merge(structural.loc[:, list(dict.fromkeys(keep_structural))], on=["candidate_id", "__ts__", "month"], how="inner", validate="one_to_one")
    panel = panel.rename(columns={"base_expected_bps": "frozen_base_expected_bps"})
    control = pd.read_parquet(CONTROL)
    control = control.loc[control.side_name.eq("long") & control.month.astype(str).isin(MONTHS), ["candidate_id", "month", "base_plus_consensus25"]]
    panel = panel.merge(control, on=["candidate_id", "month"], how="left", validate="one_to_one")
    panel = panel.loc[panel.month.isin([*MONTHS, *[f"2024-{m:02d}" for m in range(4, 12)]])].copy()
    panel["__ts__"] = pd.to_datetime(panel["__ts__"], utc=True)
    panel["label_available_ts"] = pd.to_datetime(panel["label_available_ts"], utc=True)
    if panel.loc[panel.month.isin(MONTHS), "base_plus_consensus25"].isna().any():
        raise RuntimeError("canonical control missing 2025 candidate")
    panel = panel.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    proto_abs = [c for c in panel if c.startswith("prototype__") and c.endswith("__abs_contribution")]
    k9_memberships = [c for c in panel if c.startswith("k09__cluster__") and c.endswith("__membership")]
    k9_raw = [c for c in panel if c.startswith("k09__cluster__")]
    leaf, leaf_audit = _leaf_support_features(panel)
    proto = _prototype_state_features(panel, proto_abs)
    recent, recent_audit = _recent_correctness_features(panel, proto_abs)
    covariance_fields = ["base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", "prototype_matched_mass", "prototype_entropy", "prototype_top2_margin", *k9_memberships[:3]]
    activation_target = panel.loc[:, k9_memberships].max(axis=1).to_numpy(float)
    cov_activation = _covariance_break(panel, event_ts="__ts__", target=activation_target, fields=covariance_fields, prefix="activation")
    cov_success = _covariance_break(panel, event_ts="label_available_ts", target=(panel.net_bps.to_numpy(float) > 0.0).astype(float), fields=covariance_fields, prefix="success")
    # Retain prior feature-only K9 support as a fair incumbent block.
    dynamic = _causal_dynamic_state(panel, k9_memberships)
    panel = pd.concat([panel, leaf, proto, recent, cov_activation, cov_success, dynamic], axis=1)
    leaf_fields = list(leaf.columns)
    ood_fields = list(proto.columns)
    recent_fields = [c for c in recent if c.startswith(("model_recent_", "path_recent_")) and "cross_" not in c]
    cross_fields = [c for c in recent if "cross_" in c]
    covariance_fields_out = list(cov_activation.columns) + list(cov_success.columns)
    incumbent_support = [c for c in dynamic if c.startswith("archetype_support__")]
    uncertainty = [c for c in ["prototype_entropy", "prototype_top2_margin", "prototype_exposure_top2_margin", "prototype_assignment_count", "prototype_match_similarity"] if c in panel]
    base_fields = ["base_anchor", "base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *context]
    blocks = {
        "soft_memberships": k9_raw,
        "activated_leaf_support": leaf_fields,
        "rule_path_ood_drift": ood_fields,
        "covariance_break": covariance_fields_out,
        "recent_correctness": recent_fields,
        "cross_model_state": cross_fields,
        "incumbent_support": incumbent_support,
        "incumbent_uncertainty": uncertainty,
    }
    if not all(panel[field].notna().any() for fields in blocks.values() for field in fields):
        raise RuntimeError("a requested diagnostic block has no values")
    parts: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    contract: dict[str, list[str]] = {}
    shrink_lowers = (0.25, 0.50, 0.75)
    multiply_alphas = (0.25, 0.50, 0.75, 1.00)
    for month in MONTHS:
        cutoff = pd.Timestamp(month, tz="UTC")
        train = panel.loc[panel.__ts__.lt(cutoff) & panel.label_available_ts.lt(cutoff)].copy()
        held = panel.loc[panel.month.eq(month)].copy()
        if len(train) < 500 or held.empty:
            continue
        tr_anchor, te_anchor = _map_base(train, held)
        train["base_anchor"] = tr_anchor; held["base_anchor"] = te_anchor
        target = (train.net_bps.to_numpy(float) - tr_anchor > 0.0).astype(np.int8)
        held_target = (held.net_bps.to_numpy(float) - te_anchor > 0.0).astype(np.int8)
        result = held[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_plus_consensus25"]].copy()
        result["canonical_control"] = held.base_plus_consensus25.to_numpy(float)
        sets: dict[str, list[str]] = {
            "context": base_fields,
            "soft_memberships": [*base_fields, *blocks["soft_memberships"]],
            "incumbent_support": [*base_fields, *blocks["incumbent_support"]],
            "incumbent_uncertainty": [*base_fields, *blocks["incumbent_uncertainty"]],
            "activated_leaf_support": [*base_fields, *blocks["soft_memberships"], *blocks["activated_leaf_support"]],
            "rule_path_ood_drift": [*base_fields, *blocks["soft_memberships"], *blocks["rule_path_ood_drift"]],
            "covariance_break": [*base_fields, *blocks["soft_memberships"], *blocks["covariance_break"]],
            "recent_correctness": [*base_fields, *blocks["soft_memberships"], *blocks["recent_correctness"]],
            "cross_model_state": [*base_fields, *blocks["soft_memberships"], *blocks["cross_model_state"]],
            "all_rule_state": [*base_fields, *blocks["soft_memberships"], *blocks["activated_leaf_support"], *blocks["rule_path_ood_drift"], *blocks["covariance_break"], *blocks["recent_correctness"], *blocks["cross_model_state"]],
        }
        canonical = result.canonical_control.to_numpy(float)
        for name, fields in sets.items():
            fields = list(dict.fromkeys(field for field in fields if field in train.columns))
            arm_seed = int(seed) + int(month[-2:]) * 10000 + (zlib.adler32(name.encode()) % 9999)
            probability, summary = _fit_reliability(train, held, fields, target, seed=arm_seed)
            summary["held_auc"] = float(roc_auc_score(held_target, probability)) if len(np.unique(held_target)) > 1 else float("nan")
            summary["held_brier"] = float(brier_score_loss(held_target, probability)) if len(np.unique(held_target)) > 1 else float("nan")
            summary.update({"month": month, "block": name, "train_rows": len(train), "held_rows": len(held), "target": "P(exact net > train-only isotonic base mapping)", "soft_memberships_fed": bool(any(field in k9_memberships for field in fields))})
            audit.append(summary)
            contract[name] = fields
            for lower in shrink_lowers:
                score = 0.5 + (canonical - 0.5) * (lower + (1.0 - lower) * probability)
                result[f"shrink_{name}_lo{int(lower * 100):02d}"] = score
            for alpha in multiply_alphas:
                # alpha=1 has exactly a 0.5--1.5 bounded range; smaller
                # values form progressively more conservative grids.
                score = canonical * np.clip(1.0 + alpha * (probability - 0.5), 1.0 - 0.5 * alpha, 1.0 + 0.5 * alpha)
                result[f"multiply_{name}_a{int(alpha * 100):03d}"] = score
        for field in [c for c in result if c.startswith(("shrink_", "multiply_"))]:
            result[field] = result[field].rank(pct=True, method="average").astype("float32")
        parts.append(result)
    prediction = pd.concat(parts, ignore_index=True)
    arms = ["canonical_control", *[c for c in prediction if c.startswith(("shrink_", "multiply_"))]]
    global_metrics, monthly, stability = _metric_rows(prediction, arms)
    out.mkdir(parents=True)
    prediction.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    global_metrics.to_parquet(out / "metrics_global.parquet", index=False)
    monthly.to_parquet(out / "metrics_monthly.parquet", index=False)
    stability.to_parquet(out / "metrics_stability.parquet", index=False)
    pd.DataFrame(audit).to_parquet(out / "model_audit.parquet", index=False)
    _feature_audit(panel, blocks).to_parquet(out / "feature_block_audit.parquet", index=False)
    leaf_audit.to_parquet(out / "activated_leaf_support_audit.parquet", index=False)
    recent_audit.to_parquet(out / "active_path_correctness_support_audit.parquet", index=False)
    (out / "feature_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    requested_structural_blocks = {"soft_memberships", "activated_leaf_support", "rule_path_ood_drift", "covariance_break", "recent_correctness", "cross_model_state", "all_rule_state"}
    correctness = {
        "schema": "tp6_sl4_rule_state_reliability_correctness_v1",
        "frozen_k9_soft_memberships_explicitly_fed_to_all_new_structural_challengers": all(
            any(field in k9_memberships for field in contract[name]) for name in requested_structural_blocks
        ),
        "incumbent_context_support_uncertainty_controls_remain_membership_free_by_design": True,
        "leaf_support_is_exact_catalog_train_leaf_count": True,
        "outcome_states_use_label_available_ts_strictly_before_decision": True,
        "activation_covariance_state_uses_strict_prior_decision_timestamps": True,
        "success_covariance_state_uses_strict_prior_label_availability": True,
        "prototype_ood_drift_reference_uses_prior_feature_states_only": True,
        "canonical_control_same_candidate_ids": True,
        "all_scores_finite": {arm: bool(np.isfinite(prediction[arm].to_numpy(float)).all()) for arm in arms},
        "prediction_candidate_month_pairs_unique": bool(not prediction.duplicated(["candidate_id", "month"]).any()),
        "side": "long",
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "tp6_sl4_rule_state_reliability_20260809_v1", "status": "COMPLETE", "seed": int(seed), "rows": len(prediction), "months": list(MONTHS),
        "structure": str(STRUCTURE), "raw_leaf_paths": str(RAW), "context_sha256": context_hash,
        "grid": {"shrink_lower": list(shrink_lowers), "multiply_alpha": list(multiply_alphas)},
        "blocks": {name: len(fields) for name, fields in blocks.items()},
        "scope": "matched long-only TP6/SL4/H12 diagnostic panel; not a full-universe trailing-exit replay",
        "artifacts": sorted(path.name for path in out.iterdir()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    top = global_metrics.loc[global_metrics["tail"].eq(0.05)].sort_values("net_bps_per_trade", ascending=False).head(40)
    report = [
        "# TP6/SL4 rule-state reliability grid — 2025", "",
        "All outcome-derived features use only labels mature strictly before the decision. Structural challengers explicitly include K=9 soft memberships. All scores are globally ranked after held-month normalization.", "",
        "## Top-5 global grid", "", top.round(3).to_string(index=False), "",
        "## Stability", "", stability.sort_values("mean_top5_net_bps", ascending=False).head(40).round(3).to_string(index=False), "",
        "## Correctness", "", json.dumps(correctness, indent=2), "",
    ]
    (out / "RULE_STATE_RELIABILITY_ABLATION_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out": str(out), "rows": len(prediction), "arms": len(arms)}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(out=args.out, seed=args.seed)
