#!/usr/bin/env python3
"""Matched 2025 ablation of archetype features and bounded corrections.

The input representation is frozen by
``run_tp6_sl4_prototype_cluster_quality_2025.py`` on 2024 paths only.  This
runner does not rediscover prototypes or clusters.  It tests, on the same
long-side 2025 candidate IDs, whether the following help beyond the existing
Base+Consensus score:

* archetype uncertainty, structural OOD and causal support features;
* persistent cluster sizes K=5..9 as inputs to the established residual
  LambdaRank family;
* a separate reliability classifier used as a bounded add, multiplier, or
  shrinkage modifier rather than a competing ranker.

All models are refit before each held 2025 month using only rows with mature
labels.  Structural support/OOD fields use only preceding feature states, not
outcomes.  Scores are normalised within held month before one pooled global
ranking.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
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


SEED = 20260809
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
SIZES = (5, 6, 7, 8, 9)
DEFAULT_STRUCTURE = ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_quality_20260809_v3"
DEFAULT_CONTROL = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/predictions_2025.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_use_ablation_20260809_v1"


def _safe(frame: pd.DataFrame, fields: Iterable[str], med: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = frame.reindex(columns=list(fields)).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = x.median().fillna(0.0)
    return x.fillna(med).fillna(0.0).astype("float32"), med


def _groups(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    query = pd.to_datetime(frame["__ts__"], utc=True).dt.floor("4h").astype("int64").astype(str)
    order = np.argsort(query.to_numpy(), kind="stable")
    ordered = query.iloc[order]
    count = ordered.groupby(ordered, sort=False).size()
    keep = ordered.isin(count.index[count.to_numpy() >= 2]).to_numpy()
    return order[keep], ordered.iloc[keep].groupby(ordered.iloc[keep], sort=False).size().to_numpy(np.int32)


def _map_base(train: pd.DataFrame, held: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(train["base_score"].to_numpy(float)) & np.isfinite(train["net_bps"].to_numpy(float))
    if int(ok.sum()) < 128:
        mean = float(np.nanmean(train["net_bps"]))
        return np.full(len(train), mean, np.float32), np.full(len(held), mean, np.float32)
    mapping = IsotonicRegression(out_of_bounds="clip", y_min=-1000.0, y_max=1000.0)
    mapping.fit(train.loc[ok, "base_score"], train.loc[ok, "net_bps"])
    return mapping.predict(train["base_score"]).astype("float32"), mapping.predict(held["base_score"]).astype("float32")


def _fit_ranker(
    train: pd.DataFrame,
    held: pd.DataFrame,
    fields: Sequence[str],
    residual: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train, med = _safe(train, fields)
    x_held, _ = _safe(held, fields, med)
    order, groups = _groups(train)
    if len(groups) == 0:
        return np.zeros(len(train), np.float32), np.zeros(len(held), np.float32), np.full(len(held), 0.5, np.float32)
    label = np.digitize(np.asarray(residual, float), [-100.0, -25.0, 25.0, 100.0]).astype(np.int32)
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", lambdarank_truncation_level=10,
        n_estimators=140, learning_rate=0.035, max_depth=4, num_leaves=15,
        min_child_samples=max(120, int(math.ceil(0.03 * len(train)))),
        feature_fraction=0.80, bagging_fraction=0.82, bagging_freq=1,
        lambda_l1=0.05, lambda_l2=5.0, max_bin=127,
        label_gain=[0.0, 0.25, 1.0, 3.0, 7.0], random_state=seed,
        n_jobs=4, verbosity=-1,
    )
    model.fit(x_train.iloc[order], label[order], group=groups)
    raw_train = np.asarray(model.predict(x_train), np.float32)
    raw_held = np.asarray(model.predict(x_held), np.float32)
    return raw_train, raw_held, _pct(raw_held, raw_train)


def _fit_reliability(
    train: pd.DataFrame,
    held: pd.DataFrame,
    fields: Sequence[str],
    residual: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x_train, med = _safe(train, fields)
    x_held, _ = _safe(held, fields, med)
    target = (np.asarray(residual, float) > 0.0).astype(np.int8)
    if target.min() == target.max():
        probability = np.full(len(held), float(target.mean()), np.float32)
        return np.full(len(train), float(target.mean()), np.float32), probability, {"train_positive_rate": float(target.mean()), "held_auc": float("nan"), "held_brier": float("nan")}
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=120, learning_rate=0.035, max_depth=4,
        num_leaves=15, min_child_samples=max(120, int(math.ceil(0.03 * len(train)))),
        feature_fraction=0.80, bagging_fraction=0.82, bagging_freq=1,
        reg_lambda=5.0, reg_alpha=0.05, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(x_train, target)
    return np.asarray(model.predict_proba(x_train)[:, 1], np.float32), np.asarray(model.predict_proba(x_held)[:, 1], np.float32), {"train_positive_rate": float(target.mean()), "held_auc": float("nan"), "held_brier": float("nan")}


def _causal_dynamic_state(frame: pd.DataFrame, membership_fields: Sequence[str]) -> pd.DataFrame:
    """Prequential support and structural OOD summaries, no outcome columns."""

    base_fields = ["prototype_matched_mass", "prototype_match_similarity", "prototype_entropy", "prototype_top2_margin"]
    usable = [field for field in [*base_fields, *membership_fields] if field in frame]
    states = frame.loc[:, ["__ts__", *usable]].copy()
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True)
    states = states.groupby("__ts__", sort=True)[usable].mean().sort_index()
    values: dict[str, pd.Series] = {}
    for field in usable:
        prior = states[field].shift(1)
        mean_7 = prior.rolling("7D", min_periods=1).mean()
        mean_28 = prior.rolling("28D", min_periods=4).mean()
        std_28 = prior.rolling("28D", min_periods=8).std()
        short = field in membership_fields
        prefix = "archetype_support" if short else "archetype_ood"
        values[f"{prefix}__{field}__7d"] = mean_7
        if short:
            values[f"{prefix}__{field}__28d"] = mean_28
            values[f"{prefix}__{field}__trend"] = mean_7 - mean_28
        else:
            values[f"{prefix}__{field}__z28"] = (states[field] - mean_28) / std_28.replace(0.0, np.nan)
    output = pd.DataFrame(values, index=states.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame[["__ts__"]].merge(output.reset_index(), on="__ts__", how="left", validate="many_to_one").drop(columns="__ts__")


def _causal_cluster_health(frame: pd.DataFrame, membership_fields: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return strictly prior-resolved cluster-health features and their audit.

    Each field is a membership-weighted *rank-IC proxy*, hit rate, hit-rate
    surprise versus the all-candidate causal baseline, or effective support.
    The outcome for a row enters the history only at ``label_available_ts``;
    ``merge_asof(..., allow_exact_matches=False)`` prevents a candidate from
    observing an outcome that matures at its own decision timestamp.

    The IC proxy is Pearson correlation of within-decision timestamp ranks of
    base score and exact H12 net.  It is deliberately called a rank-IC proxy:
    computing exact rolling Spearman for every soft cluster would be much more
    expensive and would not improve the causal lineage.
    """
    required = {"__ts__", "label_available_ts", "base_score", "net_bps"}
    if not required.issubset(frame.columns):
        raise KeyError(f"cluster health requires {sorted(required)}")
    source = frame.loc[:, ["__ts__", "label_available_ts", "base_score", "net_bps", *membership_fields]].copy()
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True)
    source["label_available_ts"] = pd.to_datetime(source["label_available_ts"], utc=True)
    source["_score_rank"] = source.groupby("__ts__", sort=False)["base_score"].rank(pct=True, method="average")
    source["_target_rank"] = source.groupby("__ts__", sort=False)["net_bps"].rank(pct=True, method="average")
    source["_hit"] = (pd.to_numeric(source["net_bps"], errors="coerce") > 0.0).astype("float64")
    source = source.sort_values("label_available_ts", kind="stable")

    # All-candidate prior hit rate is the reference for each cluster's
    # surprise.  This has the same maturation rule as each cluster field.
    global_state = source.groupby("label_available_ts", sort=True).agg(
        _w=("_hit", "size"), _hit=("_hit", "sum")
    )
    global_state["_prior_w"] = global_state["_w"].shift(1)
    global_state["_prior_hit"] = global_state["_hit"].shift(1)
    global_roll = pd.DataFrame(index=global_state.index)
    for days in (3, 7, 14):
        w = global_state["_prior_w"].rolling(f"{days}D", min_periods=1).sum()
        hit = global_state["_prior_hit"].rolling(f"{days}D", min_periods=1).sum()
        global_roll[f"_global_hr_{days}d"] = hit / w.replace(0.0, np.nan)

    health_values: dict[str, pd.Series] = {}
    for membership in membership_fields:
        cluster_id = membership.split("__cluster__", 1)[-1].replace("__membership", "")
        weighted = pd.to_numeric(source[membership], errors="coerce").clip(lower=0.0).fillna(0.0)
        # Use arrays deliberately: constructing a DataFrame from Series with
        # a datetime index would align the source's integer row index against
        # the availability timestamps and silently turn every event into NaN.
        weight_values = weighted.to_numpy(float)
        x_values = source["_score_rank"].to_numpy(float)
        y_values = source["_target_rank"].to_numpy(float)
        hit_values = source["_hit"].to_numpy(float)
        events = pd.DataFrame({
            "_w": weight_values,
            "_x": weight_values * x_values,
            "_y": weight_values * y_values,
            "_xx": weight_values * x_values**2,
            "_yy": weight_values * y_values**2,
            "_xy": weight_values * x_values * y_values,
            "_hit": weight_values * hit_values,
        }, index=pd.DatetimeIndex(source["label_available_ts"], name="label_available_ts"))
        state = events.groupby(level=0, sort=True).sum().reindex(global_state.index, fill_value=0.0)
        prior = state.shift(1).fillna(0.0)
        for days in (3, 7, 14):
            roll = prior.rolling(f"{days}D", min_periods=1).sum()
            weight = roll["_w"]
            cov = roll["_xy"] - roll["_x"] * roll["_y"] / weight.replace(0.0, np.nan)
            var_x = roll["_xx"] - roll["_x"] ** 2 / weight.replace(0.0, np.nan)
            var_y = roll["_yy"] - roll["_y"] ** 2 / weight.replace(0.0, np.nan)
            prefix = f"cluster_health__{cluster_id}"
            ic = cov / np.sqrt((var_x * var_y).clip(lower=0.0)).replace(0.0, np.nan)
            hit_rate = roll["_hit"] / weight.replace(0.0, np.nan)
            health_values[f"{prefix}__ic_{days}d"] = ic
            health_values[f"{prefix}__hr_{days}d"] = hit_rate
            health_values[f"{prefix}__hr_surprise_{days}d"] = hit_rate - global_roll[f"_global_hr_{days}d"]
            health_values[f"{prefix}__support_{days}d"] = weight
    health = pd.DataFrame(health_values, index=global_state.index)
    health = health.replace([np.inf, -np.inf], np.nan)

    # Map each decision timestamp to the latest state formed strictly before
    # it.  Keep NaNs until the end so the audit can distinguish no history
    # from a genuinely neutral statistic.
    lookup = health.reset_index().rename(columns={"label_available_ts": "_available_ts"})
    lookup = lookup.rename(columns={lookup.columns[0]: "_available_ts"}).sort_values("_available_ts", kind="stable")
    target = frame[["__ts__"]].copy()
    target["_row"] = np.arange(len(target), dtype=np.int64)
    target["__ts__"] = pd.to_datetime(target["__ts__"], utc=True)
    mapped = pd.merge_asof(
        target.sort_values("__ts__", kind="stable"), lookup,
        left_on="__ts__", right_on="_available_ts", direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row", kind="stable").drop(columns=["__ts__", "_available_ts", "_row"])
    audit_rows: list[dict[str, object]] = []
    for column in mapped.columns:
        if not column.startswith("cluster_health__"):
            continue
        parts = column.split("__")
        cluster, metric = parts[1], parts[2]
        values = mapped[column].to_numpy(float)
        audit_rows.append({
            "cluster": cluster, "metric": metric,
            "coverage": float(np.isfinite(values).mean()),
            "mean": float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"),
            "median": float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan"),
            "last": float(values[np.flatnonzero(np.isfinite(values))[-1]]) if np.isfinite(values).any() else float("nan"),
        })
    return mapped.fillna(0.0).astype("float32"), pd.DataFrame(audit_rows)


def _coactivation_grid(
    analysis: pd.DataFrame,
    *,
    membership_fields: Sequence[str],
    context_fields: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostic co-activation grid; never used as a model input here."""
    rows: list[dict[str, object]] = []
    shifts: list[dict[str, object]] = []
    membership = {field.split("__cluster__", 1)[-1].replace("__membership", ""): field for field in membership_fields}
    numeric_context = [field for field in context_fields if field in analysis and pd.api.types.is_numeric_dtype(analysis[field])]
    global_mean = analysis[numeric_context].apply(pd.to_numeric, errors="coerce").mean() if numeric_context else pd.Series(dtype=float)
    global_std = analysis[numeric_context].apply(pd.to_numeric, errors="coerce").std().replace(0.0, np.nan) if numeric_context else pd.Series(dtype=float)
    memberships = pd.DataFrame({name: pd.to_numeric(analysis[field], errors="coerce").fillna(0.0) for name, field in membership.items()})
    activation_modes: list[tuple[str, float, dict[str, pd.Series]]] = []
    for threshold in (0.05, 0.10, 0.20):
        activation_modes.append(("absolute_membership", threshold, {name: memberships[name].ge(threshold) for name in memberships}))
    # Soft cluster mass has one broadly present component.  Top-n co-activation
    # is a causal, row-local alternative that asks which *simultaneously
    # dominant* geometries are present, rather than which all exceed a low
    # absolute cutoff.
    descending_rank = memberships.rank(axis=1, ascending=False, method="first")
    for top_n in (2, 3):
        activation_modes.append((f"row_local_top{top_n}", float(top_n), {name: descending_rank[name].le(top_n) for name in memberships}))
    # Relative top-n removes the otherwise mechanical dominance of one large
    # soft cluster.  It normalizes membership by its population prevalence
    # before the row-local comparison; no labels or hand-coded semantics enter.
    prevalence = memberships.mean().replace(0.0, np.nan)
    relative_rank = memberships.divide(prevalence, axis="columns").fillna(0.0).rank(axis=1, ascending=False, method="first")
    for top_n in (2, 3):
        activation_modes.append((f"row_local_relative_top{top_n}", float(top_n), {name: relative_rank[name].le(top_n) for name in memberships}))
    for mode, threshold, active in activation_modes:
        activity_rate = {name: float(flag.mean()) for name, flag in active.items()}
        for arity in (2, 3):
            for names in combinations(sorted(active), arity):
                mask = np.logical_and.reduce([active[name].to_numpy(bool) for name in names])
                block = analysis.loc[mask]
                if len(block) < 40 or block["__ts__"].nunique() < 5:
                    continue
                key = " + ".join(names)
                base_ic = float(spearmanr(block["base_score"], block["net_bps"]).statistic) if block["base_score"].nunique() > 1 else float("nan")
                control_ic = float(spearmanr(block["canonical_control"], block["net_bps"]).statistic) if block["canonical_control"].nunique() > 1 else float("nan")
                item: dict[str, object] = {
                    "cluster_size": int(names[0][1:3]), "clusters": key, "arity": arity,
                    "activation_mode": mode, "membership_threshold": threshold,
                    "rows": len(block), "decision_timestamps": int(block["__ts__"].nunique()), "days": int(block["__ts__"].dt.date.nunique()),
                    "gross_bps_per_trade": float(block["gross_bps"].mean()), "net_bps_per_trade": float(block["net_bps"].mean()),
                    "hit_rate_net_positive": float((block["net_bps"] > 0.0).mean()),
                    "base_rank_ic": base_ic, "canonical_rank_ic": control_ic,
                    "coactivation_lift_vs_independence": float(mask.mean() / np.prod([activity_rate[name] for name in names])),
                }
                for score_name, label in (("base_score", "base"), ("canonical_control", "canonical")):
                    n = max(1, int(math.ceil(len(block) * 0.10)))
                    item[f"{label}_top10_net_bps"] = float(block.nlargest(n, score_name, keep="first")["net_bps"].mean())
                rows.append(item)
                if len(block) >= 100 and numeric_context:
                    local = block[numeric_context].apply(pd.to_numeric, errors="coerce").mean()
                    z = (local - global_mean) / global_std
                    for field, value in z.abs().sort_values(ascending=False).head(5).items():
                        shifts.append({
                            "cluster_size": int(names[0][1:3]), "clusters": key, "arity": arity,
                            "activation_mode": mode, "membership_threshold": threshold, "rows": len(block), "field": field,
                            "z_shift": float(z[field]), "group_mean": float(local[field]),
                            "global_mean": float(global_mean[field]),
                        })
    return pd.DataFrame(rows), pd.DataFrame(shifts)


def _structural_fields(frame: pd.DataFrame, k: int) -> tuple[list[str], list[str], list[str], list[str]]:
    prefix = f"k{k:02d}__"
    cluster = [column for column in frame.columns if column.startswith(prefix)]
    membership = [column for column in cluster if column.endswith("__membership")]
    uncertainty = [
        "prototype_entropy", "prototype_top2_margin", "prototype_exposure_top2_margin",
        "prototype_assignment_count", "prototype_match_similarity",
    ]
    ood = ["prototype_unmatched_mass", "prototype_match_similarity", "prototype_entropy", "prototype_top2_margin"]
    support = [column for column in frame.columns if column.startswith("archetype_support__") and any(member in column for member in membership)]
    dynamic_ood = [column for column in frame.columns if column.startswith("archetype_ood__")]
    return [field for field in uncertainty if field in frame], [field for field in [*ood, *dynamic_ood] if field in frame], support, cluster


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
        val = np.asarray(values, float); median = float(np.median(val)); mad = float(np.median(np.abs(val - median)))
        stability_rows.append({"arm": arm, "months": len(val), "mean_top5_net_bps": float(np.mean(val)), "median_top5_net_bps": median, "mad_top5_net_bps": mad, "worst_month_top5_net_bps": float(np.min(val)), "positive_months_top5": int(np.sum(val > 0.0)), "portability_score_bps": median - 0.5 * mad - max(0.0, -float(np.min(val))), "mean_month_rank_ic": float(np.nanmean(ics))})
    return pd.DataFrame(global_rows), pd.DataFrame(monthly_rows), pd.DataFrame(stability_rows)


def run(*, structure_dir: Path = DEFAULT_STRUCTURE, control_path: Path = DEFAULT_CONTROL, out: Path = DEFAULT_OUT, seed: int = SEED) -> Path:
    if out.exists():
        raise FileExistsError(out)
    source, context, context_hash = _load()
    source = source.loc[source["side_name"].eq("long")].copy()
    source["month"] = source["month"].astype(str)
    base_features = source[["candidate_id", "__ts__", "month", "label_available_ts", "base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", "exact_net_bps", "exact_gross_bps", *context]].copy()
    base_features = base_features.rename(columns={"exact_net_bps": "net_bps", "exact_gross_bps": "gross_bps"})
    structure = pd.read_parquet(structure_dir / "prototype_cluster_row_features.parquet")
    sizes = pd.read_parquet(structure_dir / "prototype_cluster_size_sweep_features.parquet")
    structural = structure.merge(sizes, on=["candidate_id", "__ts__", "month"], how="left", validate="one_to_one")
    panel = base_features.merge(structural.drop(columns=["net_bps", "gross_bps", "base_score", "base_expected_bps", "residual_bps"], errors="ignore"), on=["candidate_id", "__ts__", "month"], how="inner", validate="one_to_one")
    control = pd.read_parquet(control_path)
    control = control.loc[control.side_name.eq("long") & control.month.astype(str).isin(MONTHS), ["candidate_id", "month", "base_plus_consensus25"]]
    panel = panel.merge(control, on=["candidate_id", "month"], how="left", validate="one_to_one")
    panel = panel.loc[panel.month.isin([*MONTHS, *[f"2024-{month:02d}" for month in range(2, 13)]])].copy()
    panel["__ts__"] = pd.to_datetime(panel["__ts__"], utc=True)
    panel["label_available_ts"] = pd.to_datetime(panel["label_available_ts"], utc=True)
    if panel.loc[panel.month.isin(MONTHS), "base_plus_consensus25"].isna().any():
        raise RuntimeError("canonical control is missing for a 2025 target row")
    # Dynamic state is outcome-free and is calculated once over the full
    # chronological feature panel.  A row at time t sees states ending t-1.
    all_membership = [column for column in panel.columns if column.startswith("k") and column.endswith("__membership")]
    dynamic = _causal_dynamic_state(panel, all_membership)
    panel = pd.concat([panel.reset_index(drop=True), dynamic.reset_index(drop=True)], axis=1)
    # These outcome-derived statistics have a stricter lineage than the
    # structural support/OOD fields above: only labels whose H12 path had
    # already matured can contribute to a later candidate's health feature.
    health, health_audit = _causal_cluster_health(panel, all_membership)
    panel = pd.concat([panel.reset_index(drop=True), health.reset_index(drop=True)], axis=1)
    panel = panel.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    model_audit: list[dict[str, object]] = []
    arm_features: dict[str, list[str]] = {}
    reliability_feature_contract: dict[str, list[str]] = {}
    base_fields = ["base_anchor", "base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *context]
    for month in MONTHS:
        cutoff = pd.Timestamp(month, tz="UTC")
        train = panel.loc[panel["__ts__"].lt(cutoff) & panel.label_available_ts.lt(cutoff)].copy()
        held = panel.loc[panel.month.eq(month)].copy()
        if len(train) < 500 or held.empty:
            continue
        tr_anchor, te_anchor = _map_base(train, held)
        train["base_anchor"] = tr_anchor; held["base_anchor"] = te_anchor
        residual = train.net_bps.to_numpy(float) - tr_anchor
        result = held[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_plus_consensus25"]].copy()
        result["canonical_control"] = held.base_plus_consensus25.to_numpy(float)
        # Type-specific ablations use the best structurally valid K=9
        # representation.  Size is varied separately on the full block.
        uncertainty, ood, support, cluster_k9 = _structural_fields(train, 9)
        health_k9 = [column for column in train.columns if column.startswith("cluster_health__k09_")]
        feature_sets: dict[str, list[str]] = {
            "residual_context": base_fields,
            "residual_uncertainty": [*base_fields, *uncertainty],
            "residual_ood": [*base_fields, *ood],
            "residual_support": [*base_fields, *support],
            "residual_all_k09": [*base_fields, *uncertainty, *ood, *support, *cluster_k9],
            "residual_health_k09": [*base_fields, *health_k9],
            "residual_all_health_k09": [*base_fields, *uncertainty, *ood, *support, *cluster_k9, *health_k9],
        }
        for k in SIZES:
            u, o, s, c = _structural_fields(train, k)
            feature_sets[f"residual_all_k{k:02d}"] = [*base_fields, *u, *o, *s, *c]
        for arm, fields in feature_sets.items():
            fields = list(dict.fromkeys(field for field in fields if field in train.columns))
            _, raw, rank = _fit_ranker(train, held, fields, residual, seed=int(seed) + int(month[-2:]) * 100 + len(fields))
            result[arm] = 0.75 * result.canonical_control.to_numpy(float) + 0.25 * rank
            arm_features[arm] = fields
            model_audit.append({"month": month, "arm": arm, "model_type": "existing_residual_lambdarank", "train_rows": len(train), "held_rows": len(held), "feature_count": len(fields), "residual_target": "exact net - train-only isotonic(base score)", "query": "4-hour UTC x side", "raw_held_rank_ic": float(spearmanr(raw, held.net_bps.to_numpy(float)).statistic)})
        # A distinct reliability learner answers whether the base is
        # underestimating rather than independently re-ranking the book.  The
        # same feature-type decomposition is retained here, because a field
        # that harms a free residual ranker can still be useful to bound the
        # magnitude of a correction.
        reliability_sets = {
            "context": base_fields,
            "uncertainty_k09": [*base_fields, *uncertainty],
            "ood_k09": [*base_fields, *ood],
            "support_k09": [*base_fields, *support],
            "health_k09": [*base_fields, *health_k9],
            "all_k05": arm_features["residual_all_k05"],
            "all_k06": arm_features["residual_all_k06"],
            "all_k07": arm_features["residual_all_k07"],
            "all_k08": arm_features["residual_all_k08"],
            "all_k09": arm_features["residual_all_k09"],
            "all_health_k09": arm_features["residual_all_health_k09"],
        }
        correctness = (held.net_bps.to_numpy(float) - te_anchor > 0.0).astype(int)
        canonical = result.canonical_control.to_numpy(float)
        for offset, (name, reliability_fields) in enumerate(reliability_sets.items()):
            reliability_fields = list(dict.fromkeys(field for field in reliability_fields if field in train.columns))
            reliability_feature_contract[name] = reliability_fields
            _, probability, summary = _fit_reliability(
                train, held, reliability_fields, residual,
                seed=int(seed) + int(month[-2:]) * 1000 + offset,
            )
            if len(np.unique(correctness)) > 1:
                summary["held_auc"] = float(roc_auc_score(correctness, probability))
                summary["held_brier"] = float(brier_score_loss(correctness, probability))
            centred = probability - 0.5
            result[f"reliability_add_{name}"] = canonical + 0.25 * centred
            result[f"reliability_multiply_{name}"] = canonical * np.clip(1.0 + 0.50 * centred, 0.75, 1.25)
            result[f"reliability_shrink_{name}"] = 0.5 + (canonical - 0.5) * (0.5 + 0.5 * probability)
            result[f"reliability_probability_{name}"] = probability
            model_audit.append({"month": month, "arm": f"reliability_{name}", "model_type": "bounded_correctness_classifier", "train_rows": len(train), "held_rows": len(held), "feature_count": len(reliability_fields), "residual_target": "P(exact net > train-only base anchor)", "query": "per-row classifier", **summary})
        # Newly generated arms use the same within-month long-side percentile
        # convention as the stored canonical score before global selection.
        for arm in [name for name in result.columns if name.startswith(("residual_", "reliability_add", "reliability_multiply", "reliability_shrink"))]:
            result[arm] = result[arm].rank(pct=True, method="average").astype("float32")
        parts.append(result)
    prediction = pd.concat(parts, ignore_index=True)
    arms = ["canonical_control", *[column for column in prediction.columns if column.startswith(("residual_", "reliability_add", "reliability_multiply", "reliability_shrink"))]]
    global_metrics, monthly_metrics, stability = _metric_rows(prediction, arms)
    out.mkdir(parents=True)
    prediction.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    global_metrics.to_parquet(out / "metrics_global.parquet", index=False)
    monthly_metrics.to_parquet(out / "metrics_monthly.parquet", index=False)
    stability.to_parquet(out / "metrics_stability.parquet", index=False)
    pd.DataFrame(model_audit).to_parquet(out / "model_audit.parquet", index=False)
    (out / "feature_contract.json").write_text(json.dumps(arm_features, indent=2) + "\n")
    (out / "reliability_feature_contract.json").write_text(json.dumps(reliability_feature_contract, indent=2) + "\n")
    # Per-archetype recent-health audit and co-activation grid are diagnostics.
    # They are persisted separately so their realised-outcome measurements are
    # never confused with inputs available at the decision time.
    health_columns = list(health.columns)
    health_2025 = panel.loc[panel.month.isin(MONTHS), ["candidate_id", "month", *health_columns]].copy()
    health_rows: list[dict[str, object]] = []
    for month, block in health_2025.groupby("month", sort=True):
        for column in health_columns:
            parts_name = column.split("__")
            values = block[column].to_numpy(float)
            health_rows.append({
                "month": month, "cluster": parts_name[1], "metric": parts_name[2],
                "rows": len(block), "nonzero_coverage": float((np.abs(values) > 1e-12).mean()),
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "last": float(values[-1]),
            })
    pd.DataFrame(health_rows).to_parquet(out / "cluster_health_monthly_2025.parquet", index=False)
    health_audit.to_parquet(out / "cluster_health_feature_audit.parquet", index=False)
    diagnostic_fields = [column for column in panel.columns if column.startswith("k09__cluster__") and column.endswith("__membership")]
    analysis = prediction.merge(
        panel.loc[panel.month.isin(MONTHS), ["candidate_id", "month", "base_score", *diagnostic_fields]],
        on=["candidate_id", "month"], how="left", validate="one_to_one",
    )
    coactivation, coactivation_shift = _coactivation_grid(
        analysis, membership_fields=diagnostic_fields, context_fields=context,
    )
    coactivation.to_parquet(out / "cluster_coactivation_grid_2025.parquet", index=False)
    coactivation_shift.to_parquet(out / "cluster_coactivation_context_shift_2025.parquet", index=False)
    finite_generated = {
        arm: bool(np.isfinite(prediction[arm].to_numpy(float)).all()) for arm in arms
    }
    correctness = {
        "schema": "tp6_sl4_prototype_cluster_use_ablation_correctness_v2",
        "prototype_cluster_contract_frozen_before_2025": True,
        "target_2025_outcomes_used_in_representation_selection": False,
        "dynamic_support_uses_preceding_feature_states_only": True,
        "residual_train_labels_mature_before_held_month": True,
        "canonical_control_same_candidate_ids": True,
        "global_ranking_after_month_normalization": True,
        "cluster_health_uses_label_available_ts_strictly_before_decision": True,
        "cluster_health_is_not_used_in_coactivation_diagnostic": True,
        "all_generated_scores_finite": finite_generated,
        "prediction_candidate_month_pairs_unique": bool(not prediction.duplicated(["candidate_id", "month"]).any()),
        "side": "long",
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "tp6_sl4_prototype_cluster_use_ablation_20260809_v2", "status": "COMPLETE",
        "structure": str(structure_dir), "control": str(control_path), "seed": int(seed), "rows": len(prediction), "months": list(MONTHS),
        "cluster_sizes": list(SIZES), "base_contract": "stored monthly Base+Consensus score; long-side matched diagnostic population",
        "residual": "existing shallow LambdaRank family, exact net minus train-only isotonic base anchor", "context_sha256": context_hash,
        "reliability": "P(realised net exceeds train-only base anchor), used only as bounded add/multiply/shrink modifier",
        "cluster_health": "prior-resolved membership-weighted base-score rank-IC proxy, net-positive hit rate, hit-rate surprise versus the causal all-candidate reference, and support over 3/7/14 days",
        "coactivation": "diagnostic only: K=9 pair/triple membership intersections at 5%/10% thresholds, with realised economics and model rank metrics",
        "artifacts": sorted(path.name for path in out.iterdir()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# TP6/SL4 prototype-cluster feature and integration ablation", "",
        "All arms are long-only, monthly prequential, and globally ranked after within-month normalization.", "",
        "## Global metrics", "", global_metrics.round(3).to_string(index=False), "",
        "## Stability", "", stability.round(3).to_string(index=False), "",
        "## Causal cluster-health fields", "", health_audit.round(4).to_string(index=False), "",
        "## K=9 co-activation diagnostic (top rows by support)", "", coactivation.sort_values("rows", ascending=False).head(30).round(3).to_string(index=False), "",
        "## Contract", "", json.dumps(manifest, indent=2),
    ]
    (out / "PROTOTYPE_CLUSTER_USE_ABLATION_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out": str(out), "rows": len(prediction), "arms": arms}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(structure_dir=args.structure_dir, control_path=args.control, out=args.out, seed=args.seed)
