#!/usr/bin/env python3
"""Reconstruct the rule lineage of selected current correctness-head leaves.

The placement ablation stores final selected membership names but not the
fold-local shallow-tree dump.  This diagnostic replay recreates only the
selected fold/side/horizon dictionaries from the immutable input, using the
same deterministic discovery and reference protocol.  It is read-only with
respect to the winning-score artifact and never refits the conversion meta
model or reads outer-test labels to choose a leaf.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb

from scripts.run_correctness_leaf_regime_oof import _represent, _rules, _screen, _target
from scripts.run_leaf_target_family_ablation import (
    AVAILABILITY,
    BASE_META_PROBABILITY_FIELDS,
    DISCOVERY_POOL_SIZE,
    INPUT,
    META_EXCLUDE,
    _diverse_discovery_pool,
)
from scripts.run_two_year_leaf_regime_top20_meta import _folds


PLACEMENT = ROOT / "data_perp/artifacts/transition_feature_placement_ablation_20260803_v1/current"
DEFAULT_OUT = ROOT / "data_perp/artifacts/current_correctness_head_leaf_semantics_20260804_v1"
FEATURE_RE = re.compile(
    r"^leafreg__correctness__(?P<horizon>row|period12h|period24h|period72h)"
    r"__f(?P<fold>\d+)__s(?P<side>long|short)__c(?P<cluster>\d+)__(?P<mode>.+)$"
)
HORIZONS = {"row": None, "period12h": 12, "period24h": 24, "period72h": 72}


def _selection(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path / "target_family_selection.parquet")
    table = table[(table.target_family.eq("correctness")) & table.accepted].copy()
    parsed = table.feature.str.extract(FEATURE_RE)
    if parsed.isna().any().any() or len(parsed) != len(table):
        raise ValueError("selected correctness features do not match the frozen leaf identity format")
    for name in ("fold", "cluster"):
        parsed[name] = pd.to_numeric(parsed[name], errors="raise").astype(int)
    parsed = parsed.rename(columns={"fold": "leaf_fold", "side": "leaf_side"}).reset_index(drop=True)
    return pd.concat([table.reset_index(drop=True), parsed], axis=1).sort_values(
        ["leaf_fold", "leaf_side", "horizon", "cluster", "mode"], kind="stable"
    )


def _data() -> tuple[pd.DataFrame, list[str], list[str]]:
    availability = pd.read_parquet(AVAILABILITY)
    store_raw = [
        name
        for name in availability.loc[availability.usable_90pct_nonconstant, "feature"].astype(str)
        if name not in META_EXCLUDE
    ]
    discovery_pool = _diverse_discovery_pool(store_raw)
    raw = list(
        dict.fromkeys(
            [
                *discovery_pool,
                *(field for field in BASE_META_PROBABILITY_FIELDS if field in store_raw),
            ]
        )
    )
    required = [
        "candidate_id", "__ts__", "label_available_ts", "side_name", "era", "gross_bps", "net_bps",
        "prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear", *raw,
    ]
    data = pd.read_parquet(INPUT, columns=list(dict.fromkeys(required)))
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
    data["label_available_ts"] = pd.to_datetime(data["label_available_ts"], utc=True)
    data = data[data.__ts__ < pd.Timestamp("2026-04-01", tz="UTC")].copy()
    data = data[np.isfinite(data.net_bps) & np.isfinite(data.prequential_base_expected_net_bps)]
    data = data.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    data["fold"] = _folds(data)
    return data, discovery_pool, raw


def _fit_dictionary(
    data: pd.DataFrame,
    discovery_pool: list[str],
    *,
    fold: int,
    side: str,
    horizon: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    test = data[data.fold.eq(fold)].copy()
    start = test.__ts__.min()
    history = data[data.label_available_ts < start].copy()
    discovery_cut = history.__ts__.quantile(.60)
    discovery = history[history.__ts__ <= discovery_cut].copy()
    meta_train = history[history.__ts__ > discovery_cut].copy()
    disc = discovery[discovery.side_name.eq(side)].copy()
    mt = meta_train[meta_train.side_name.eq(side)].copy()
    te = test[test.side_name.eq(side)].copy()
    if min(len(disc), len(mt), len(te)) < 300:
        raise ValueError(f"insufficient rows for fold={fold} side={side} horizon={horizon}")
    combined = pd.concat([mt, te], ignore_index=True)
    labelled = _target(pd.concat([disc, mt, te], ignore_index=True), disc, HORIZONS[horizon], "correctness")
    target_train = labelled.iloc[: len(disc)].copy()
    target_train = target_train[np.isfinite(target_train.target_value)].copy()
    chosen = _screen(target_train, discovery_pool, target_train.target_value.to_numpy(float))
    if len(chosen) < DISCOVERY_POOL_SIZE:
        raise ValueError(f"unexpected screened discovery field count: {len(chosen)}")
    median = target_train[chosen].median().fillna(0.0)
    iqr = (target_train[chosen].quantile(.75) - target_train[chosen].quantile(.25)).replace(0, 1).fillna(1.0)
    x = ((target_train[chosen].fillna(median) - median) / iqr).clip(-8, 8).to_numpy("float32")
    model = lgb.LGBMRegressor(
        objective="regression_l2", n_estimators=80, learning_rate=.04, num_leaves=16,
        max_depth=4, min_child_samples=max(80, int(.01 * len(target_train))),
        colsample_bytree=.8, reg_lambda=20.0, random_state=20260803 + fold, n_jobs=1, verbosity=-1,
    ).fit(x, target_train.target_value.to_numpy(float))
    reference = combined.iloc[: len(mt)].copy()
    reference.loc[:, chosen] = ((reference[chosen].fillna(median) - median) / iqr).clip(-8, 8)
    normalized = combined.copy()
    normalized.loc[:, chosen] = ((normalized[chosen].fillna(median) - median) / iqr).clip(-8, 8)
    target_name = f"correctness__{horizon}"
    rules, memberships = _rules(model, chosen, reference, 0.0)
    representation, rule_rows, similarity, output_fields, lineage = _represent(
        normalized, rules, memberships, side, target_name, fold, minimum_similarity=.70
    )
    representation = representation.merge(
        combined[["candidate_id", "net_bps", "prequential_base_expected_net_bps"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    return representation.iloc[len(mt):].reset_index(drop=True), rule_rows, lineage, output_fields


def _conditional_mean(values: pd.Series, active: pd.Series, state: bool) -> float:
    part = pd.to_numeric(values[active if state else ~active], errors="coerce").dropna()
    return float(part.mean()) if len(part) else np.nan


def run(output: Path = DEFAULT_OUT, placement: Path = PLACEMENT) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    selected = _selection(placement)
    data, discovery_pool, raw = _data()
    detail_rows: list[dict] = []
    rules_out: list[pd.DataFrame] = []
    contexts_out: list[pd.DataFrame] = []
    for (fold, side, horizon), group in selected.groupby(["leaf_fold", "leaf_side", "horizon"], sort=True):
        representation, rule_rows, lineage, output_fields = _fit_dictionary(
            data, discovery_pool, fold=int(fold), side=str(side), horizon=str(horizon)
        )
        needed = set(group.feature)
        unavailable = needed.difference(output_fields)
        if unavailable:
            raise AssertionError(f"frozen selected feature identities did not reproduce: {sorted(unavailable)}")
        for item in group.itertuples(index=False):
            cluster = int(item.cluster)
            base = f"leafreg__correctness__{item.horizon}__f{item.leaf_fold}__s{item.leaf_side}__c{cluster:02d}"
            cluster_rules = rule_rows[rule_rows.cluster.eq(cluster)].copy()
            leaf_lineage = lineage[lineage.feature.eq(item.feature)].copy()
            if cluster_rules.empty or leaf_lineage.empty:
                raise AssertionError(f"missing reconstructed lineage for {item.feature}")
            membership = pd.to_numeric(representation[item.feature], errors="coerce")
            active = membership.ge(.60)
            residual = representation.net_bps - representation.prequential_base_expected_net_bps
            valid = membership.notna() & residual.notna()
            rank_ic = spearmanr(membership[valid], residual[valid]).statistic if valid.sum() >= 20 else np.nan
            record = {
                "feature": item.feature,
                "fold": int(item.leaf_fold),
                "side_name": item.leaf_side,
                "horizon": item.horizon,
                "mode": item.mode,
                "cluster": cluster,
                "mda_logloss": float(item.mda_logloss),
                "phantom_q95": float(item.phantom_q95),
                "mda_excess_over_phantom": float(item.mda_excess_over_phantom),
                "activation_correlation": float(item.max_activation_correlation),
                "cluster_size": int(cluster_rules.cluster_size.iloc[0]),
                "cluster_signature": str(cluster_rules.cluster_signature.iloc[0]),
                "rule_count": len(cluster_rules),
                "outer_rows": len(representation),
                "outer_active_share_p60": float(active.mean()),
                "outer_active_share_p70": float(membership.ge(.70).mean()),
                "outer_active_share_p80": float(membership.ge(.80).mean()),
                "outer_active_share_p90": float(membership.ge(.90).mean()),
                "outer_membership_q95": float(membership.quantile(.95)),
                "outer_net_bps_active": _conditional_mean(representation.net_bps, active, True),
                "outer_net_bps_inactive": _conditional_mean(representation.net_bps, active, False),
                "outer_residual_bps_active": _conditional_mean(residual, active, True),
                "outer_residual_bps_inactive": _conditional_mean(residual, active, False),
                "outer_membership_residual_rank_ic": float(rank_ic) if np.isfinite(rank_ic) else np.nan,
            }
            detail_rows.append(record)
            rule_copy = cluster_rules.copy()
            rule_copy.insert(0, "selected_feature", item.feature)
            rules_out.append(rule_copy)
            context = representation.loc[:, ["candidate_id", "__ts__", "side_name", "net_bps", "prequential_base_expected_net_bps", item.feature]].copy()
            context["selected_feature"] = item.feature
            context["residual_bps"] = context.net_bps - context.prequential_base_expected_net_bps
            context["active_p60"] = context[item.feature].ge(.60)
            contexts_out.append(context)
    detail = pd.DataFrame(detail_rows).sort_values(["fold", "side_name", "horizon", "cluster"], kind="stable")
    detail.to_parquet(output / "selected_leaf_semantics.parquet", index=False)
    pd.concat(rules_out, ignore_index=True).to_parquet(output / "selected_leaf_rules.parquet", index=False)
    pd.concat(contexts_out, ignore_index=True).to_parquet(output / "selected_leaf_outer_context.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": "COMPLETED",
        "selection_source": str(placement / "target_family_selection.parquet"),
        "input": str(INPUT),
        "feature_contract": "current features only; no market-transition sidecar",
        "leaf_dictionary": "deterministic replay of fold-local discovery with a 0.70 rule-similarity threshold",
        "selected_feature_count": len(detail),
        "selected_feature_identity_exactly_reproduced": True,
        "context_metrics": "outer-fold descriptive only; they do not choose, reject, or rank representations",
        "outputs": ["selected_leaf_semantics.parquet", "selected_leaf_rules.parquet", "selected_leaf_outer_context.parquet"],
    }, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--placement", type=Path, default=PLACEMENT)
    args = parser.parse_args()
    print(run(args.out, args.placement))
