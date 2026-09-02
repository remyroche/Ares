#!/usr/bin/env python3
"""Frozen-family specialist heads for the strict O3-v2 research funnel.

This is deliberately an offline, research-only H1 stage.  It takes a
previously selected target-free F1--F6 panel, fits one small residual head per
family on a strictly prior, fully-resolved three-calendar-month window, and
writes held-month scores before any policy outcome is joined.  It does not
touch live bundles, MC1, admission, the portfolio engine, or MDA.

The use of a parent-only F5 contract is intentional: it avoids making a
specialist depend on a later O3 score, so a head keeps the same semantic input
definition throughout its usable history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402
import run_strict_r3_o3v2_support_funnel_v3 as support  # noqa: E402
import run_strict_r3_o3v2_target_funnel as target  # noqa: E402


SCHEMA = "strict_r3_o3v2_specialist_funnel_v4"
SEED = 1729
RESERVE_DAYS = 28
TRAIN_MONTHS = 6
TRAIN_CAP = 240_000
MIN_TRAIN_ROWS = 5_000
FAMILIES = ("f1", "f2", "f3", "f4", "f5", "f6")
TARGETS = ("T3_pair_residual_lambdarank", "T6_rank_error_ordinal")
ARCHITECTURES = ("H1_family", "H2_population", "H3_hybrid_f4_f5")
SUPPORT_BY_TARGET = {
    "T3_pair_residual_lambdarank": "SB1_error_archetype",
    "T6_rank_error_ordinal": "SB3_error_semantic",
}
PROHIBITED = set(target.PROHIBITED_SCORE_COLUMNS)
SHARED_CORE = (
    "f1_enhanced_base_bps", "f1_base_rank_ts", "f1_base_bps",
    "f1_efficiency_bps", "f1_timing_bps", "f1_e_minus_t",
    "f1_e_minus_b0", "f1_t_minus_b0", "f1_base_component_std",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for child in files:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _query(frame: pd.DataFrame) -> pd.Series:
    stamp = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return stamp.astype(str) + "|" + frame["side_name"].astype(str).str.lower()


def _read_history(panel_path: Path, fields: Sequence[str]) -> pd.DataFrame:
    required = [
        "candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts",
        # T3's train-only near-tie sampler uses the canonical upstream score
        # to identify adjacent candidates.  It is never written to a held
        # specialist output except where F1 itself selected it.
        "f1_enhanced_base_bps", *fields,
    ]
    columns = list(dict.fromkeys(required))
    # Inspect the Parquet schema first.  Selecting a safe subset of columns
    # alone is insufficient: a mistakenly outcome-enriched "feature" panel
    # must be rejected rather than silently tolerated.
    schema_fields = set(pq.ParquetFile(panel_path).schema_arrow.names)
    prohibited = PROHIBITED.intersection(schema_fields)
    if prohibited:
        raise AssertionError(f"target-free specialist panel contains outcome fields: {sorted(prohibited)}")
    missing = set(columns) - schema_fields
    if missing:
        raise KeyError(f"target-free specialist panel lacks selected fields: {sorted(missing)[:5]}")
    frame = pd.read_parquet(panel_path, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("target-free specialist history has duplicate candidate IDs")
    return frame


def _read_policy(path: Path) -> pd.DataFrame:
    result = pd.read_parquet(path, columns=(
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
    ))
    result["policy_label_available_ts"] = pd.to_datetime(
        result["policy_label_available_ts"], utc=True, errors="coerce",
    )
    if result["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy ledger has duplicate candidate IDs")
    return result


def _semantic_window(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    wanted = [
        "candidate_id", "__decision_ts__", "side_name", "semantic_path_valid",
        "semantic_label_available_ts", "semantic_archetype", "semantic_tbm_event",
        "semantic_axis_a_sequence", "semantic_axis_c_persistence",
        # SB3's declared training-only semantic weight uses both exit
        # partitions.  They are never persisted in held score receipts.
        "semantic_axis_f_exit4", "semantic_axis_f_exit5", "semantic_policy_net_bps",
    ]
    parts: list[pd.DataFrame] = []
    periods = pd.period_range(start.to_period("M"), (end - pd.Timedelta(nanoseconds=1)).to_period("M"), freq="M")
    for period in periods:
        path = root / "parts" / f"month={period.strftime('%Y-%m')}" / "semantics.parquet"
        if not path.exists():
            continue
        part = pd.read_parquet(path, columns=wanted)
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        part["semantic_label_available_ts"] = pd.to_datetime(
            part["semantic_label_available_ts"], utc=True, errors="coerce",
        )
        parts.append(part.loc[part["__decision_ts__"].ge(start) & part["__decision_ts__"].lt(end)].copy())
    if not parts:
        return pd.DataFrame(columns=wanted)
    output = pd.concat(parts, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("semantic window has duplicate candidate IDs")
    return output


def _train_target(train: pd.DataFrame, target_name: str) -> np.ndarray:
    policy = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    base_rank = pd.to_numeric(train["f1_base_rank_ts"], errors="coerce").to_numpy(float)
    if target_name == "T6_rank_error_ordinal":
        realised_rank = train.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(
            pct=True, method="average",
        ).to_numpy(float)
        return target._rank_error_grade(realised_rank - base_rank).astype(np.float32)
    raise ValueError(f"unsupported target {target_name}")


def _sample_queries(train: pd.DataFrame, *, equal_month: bool) -> pd.DataFrame:
    """Cap at complete exact-timestamp queries without splitting a query."""
    work = train.copy()
    work["__query__"] = _query(work).to_numpy()
    work["__month__"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    work = work.loc[work["__query__"].map(work["__query__"].value_counts()).ge(2)].copy()
    if work.empty:
        raise ValueError("specialist lacks multi-candidate query support")
    if len(work) <= TRAIN_CAP:
        return work.sort_values(["__query__", "__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    metadata = work.groupby("__query__", sort=False).agg(
        rows=("candidate_id", "size"), month=("__month__", "first"), first_ts=("__decision_ts__", "min"),
    ).reset_index()
    retained: list[str] = []
    if equal_month:
        allowance = max(2, TRAIN_CAP // max(metadata["month"].nunique(), 1))
        groups: Iterable[tuple[object, pd.DataFrame]] = metadata.groupby("month", sort=True)
    else:
        allowance = TRAIN_CAP
        groups = [("all", metadata)]
    for _month, local in groups:
        used = 0
        ordered = local.assign(__rnd__=rng.random(len(local))).sort_values(
            ["__rnd__", "first_ts", "__query__"], kind="stable",
        )
        for row in ordered.itertuples(index=False):
            if used + int(row.rows) <= allowance:
                retained.append(str(row.__getattribute__("__query__")))
                used += int(row.rows)
    output = work.loc[work["__query__"].isin(retained)].copy()
    return output.sort_values(["__query__", "__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


@dataclass
class Specialist:
    name: str
    fields: tuple[str, ...]
    medians: np.ndarray
    model: object
    reference: object
    support_arm: str

    def score(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = parent._numeric_matrix(frame, self.fields, self.medians)
        raw = self.model.predict(matrix).astype(np.float32)
        return raw, self.reference.cdf(raw).astype(np.float32)


def _fit_head(
    train: pd.DataFrame,
    fields: Sequence[str],
    values: np.ndarray,
    weights: np.ndarray,
    *, name: str, support_arm: str, equal_month: bool = False,
) -> tuple[Specialist, dict[str, object]]:
    work = train.loc[:, ["candidate_id", "__decision_ts__", "side_name", *fields]].copy()
    work["__target__"] = np.asarray(values, dtype=np.float32)
    work["__support_weight__"] = np.asarray(weights, dtype=np.float32)
    work = _sample_queries(work, equal_month=equal_month)
    target_values = work.pop("__target__").to_numpy(np.float32)
    sample_weight = work.pop("__support_weight__").to_numpy(np.float32)
    if len(work) < MIN_TRAIN_ROWS or np.nanstd(target_values) <= 1e-10:
        raise ValueError(f"{name}: insufficient target support rows={len(work)}")
    medians = parent._fit_medians(work, fields)
    model = LGBMRegressor(
        objective="regression_l2", metric="l2", n_estimators=120, learning_rate=.035,
        max_depth=4, num_leaves=15, min_child_samples=max(300, int(.015 * len(work))),
        feature_fraction=.82, bagging_fraction=.82, bagging_freq=1,
        lambda_l1=.02, lambda_l2=2.0, max_bin=127,
        random_state=SEED + sum(ord(char) for char in name), n_jobs=1,
        deterministic=True, force_col_wise=True, verbosity=-1,
    ).fit(parent._numeric_matrix(work, fields, medians), target_values, sample_weight=sample_weight)
    raw_train = model.predict(parent._numeric_matrix(work, fields, medians))
    specialist = Specialist(
        name=name, fields=tuple(fields), medians=medians, model=model,
        reference=parent.ScoreReference.fit(raw_train, source=f"{name}_strict_prequential_training_distribution"),
        support_arm=support_arm,
    )
    return specialist, {
        "head": name, "fields": list(fields), "train_rows": int(len(work)),
        "train_queries": int(work["__query__"].nunique()), "train_target_mean": float(np.mean(target_values)),
        "train_target_std": float(np.std(target_values)), "weight_mean": float(np.mean(sample_weight)),
        "weight_min": float(np.min(sample_weight)), "weight_max": float(np.max(sample_weight)),
    }


def _fit_pairwise_head(
    train: pd.DataFrame,
    fields: Sequence[str],
    weights: np.ndarray,
    *, name: str, support_arm: str, equal_month: bool = False,
) -> tuple[Specialist, dict[str, object]]:
    """Fit T3 with its declared near-tie LambdaRank supervision.

    T3 is not an ordinal regression target.  The pair construction is shared
    with the selected broad correction layer and sees realised policy labels
    only inside the strict pre-reserve training fold.  Held score receipts
    retain just the selected causal feature fields and model outputs.
    """
    needed = ["candidate_id", "__decision_ts__", "side_name", "policy_net_bps", "f1_enhanced_base_bps", *fields]
    source = train.loc[:, list(dict.fromkeys(needed))].copy()
    source["enhanced_base_bps"] = pd.to_numeric(source["f1_enhanced_base_bps"], errors="coerce")
    spec = parent.ConsensusHeadSpec(
        name=name,
        cap=100,
        weight_mode="equal_month" if equal_month else "ordinary",
        # Pair generation subsequently replaces these with two-row pair
        # queries; the complete-query sampler itself must use one of the
        # frozen parent identifiers.
        query="exact_timestamp_side",
        fields=tuple(fields),
        target_edges_bps=(-100.0, -30.0, 30.0, 90.0),
        params={
            "objective": "lambdarank", "metric": "ndcg", "n_estimators": 120,
            "learning_rate": .035, "max_depth": 4, "num_leaves": 15,
            "min_child_samples": max(300, int(.015 * len(source))),
            "feature_fraction": .82, "bagging_fraction": .82, "bagging_freq": 1,
            "lambda_l1": .02, "lambda_l2": 2.0, "max_bin": 127,
            "verbosity": -1,
        },
    )
    sampled, pair_target, groups = parent._sample_base_near_tie_pairs(
        source, spec, "near_tie_diff50", seed=SEED + sum(ord(char) for char in name),
    )
    weight_by_id = pd.Series(np.asarray(weights, dtype=np.float32), index=train["candidate_id"].astype(str))
    sample_weight = sampled["candidate_id"].astype(str).map(weight_by_id).fillna(1.0).to_numpy(np.float32)
    if equal_month:
        month = pd.to_datetime(sampled["__decision_ts__"], utc=True).dt.strftime("%Y-%m")
        frequency = month.value_counts()
        sample_weight *= month.map(lambda value: 1.0 / float(frequency.loc[value])).to_numpy(np.float32)
        sample_weight *= len(sample_weight) / max(float(sample_weight.sum()), 1e-12)
    if len(sampled) < 20 or np.unique(pair_target).size < 2:
        raise ValueError(f"{name}: insufficient near-tie pair support")
    medians = parent._fit_medians(sampled, fields)
    params = dict(spec.params)
    params.update(random_state=SEED + sum(ord(char) for char in name), n_jobs=1, deterministic=True, force_col_wise=True)
    model = LGBMRanker(**params).fit(
        parent._numeric_matrix(sampled, fields, medians), pair_target,
        group=groups, sample_weight=sample_weight,
    )
    raw_train = model.predict(parent._numeric_matrix(sampled, fields, medians))
    specialist = Specialist(
        name=name, fields=tuple(fields), medians=medians, model=model,
        reference=parent.ScoreReference.fit(raw_train, source=f"{name}_strict_prequential_pair_distribution"),
        support_arm=support_arm,
    )
    return specialist, {
        "head": name, "fields": list(fields), "train_rows": int(len(sampled)),
        "train_queries": int(len(groups)), "train_target_mean": float(np.mean(pair_target)),
        "train_target_std": float(np.std(pair_target)), "weight_mean": float(np.mean(sample_weight)),
        "weight_min": float(np.min(sample_weight)), "weight_max": float(np.max(sample_weight)),
        "pairwise_mode": "near_tie_diff50",
    }


def _metric_rows(
    scores: pd.DataFrame,
    policy: pd.DataFrame,
    *, target_name: str, month: pd.Timestamp,
) -> list[dict[str, object]]:
    joined = scores.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    valid = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(joined["policy_net_bps"], errors="coerce"))
    work = joined.loc[valid].copy()
    outcome = pd.to_numeric(work["policy_net_bps"], errors="coerce").to_numpy(float)
    rows: list[dict[str, object]] = []
    fields = [field for field in work if field.endswith("__rank")] + ["specialist_ensemble_rank"]
    for field in fields:
        score = pd.to_numeric(work[field], errors="coerce").to_numpy(float)
        finite = np.isfinite(score) & np.isfinite(outcome)
        ic = float(spearmanr(score[finite], outcome[finite]).statistic) if finite.sum() >= 12 else np.nan
        for tail in (.01, .02, .03, .05, .10):
            threshold = np.quantile(score[finite], 1.0 - tail, method="higher")
            selected = outcome[finite & (score >= threshold)]
            rows.append({
                "target": target_name, "month": f"{month:%Y-%m}", "score": field,
                "tail": tail, "trades": int(len(selected)),
                "net_ev_bps_per_trade": float(np.mean(selected)), "net_sum_bps": float(np.sum(selected)),
                "policy_rank_ic": ic,
            })
    return rows


def _head_correlation(scores: pd.DataFrame, *, target_name: str, month: pd.Timestamp) -> pd.DataFrame:
    ranks = [field for field in scores if field.endswith("__rank")]
    if len(ranks) < 2:
        return pd.DataFrame()
    corr = scores.loc[:, ranks].corr(method="spearman")
    values = []
    for left in ranks:
        for right in ranks:
            if left < right:
                values.append({"target": target_name, "month": f"{month:%Y-%m}", "left": left, "right": right, "spearman": float(corr.loc[left, right])})
    return pd.DataFrame(values)


def _coverage(frame: pd.DataFrame, fields: Sequence[str], month: pd.Timestamp) -> dict[str, object]:
    return {
        "month": f"{month:%Y-%m}", "rows": int(len(frame)),
        "routed_rows": int(len(frame)),
        "field_complete_fraction": float(frame.loc[:, list(fields)].notna().all(axis=1).mean()),
    }


def _head_definitions(
    selection: dict[str, list[str]], architecture: str, default_support: str,
    mixed_selection: Sequence[str] | None = None,
) -> tuple[tuple[str, tuple[str, ...], str, bool], ...]:
    # Frozen shared core for every heterogeneous head.  It contains the
    # upstream score and the compact B0/E/T geometry; each head's additional
    # fields then have one stable declared semantic role across all folds.
    shared_core = SHARED_CORE

    def with_core(extra: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*shared_core, *extra)))

    if architecture == "H1_family":
        # The requested H1 comparison is a five-head *role* family, not one
        # head per arbitrary feature prefix.  F5/F6 provenance/other-meta
        # fields are available through the final G1-selected mixed head; this
        # avoids silently turning the five-head O3 comparison into a six-head
        # capacity expansion.
        return (
            ("h1_base_geometry", with_core(selection["f1"]), default_support, False),
            ("h1_query_geometry", with_core(selection["f2"]), default_support, False),
            ("h1_recent_error", with_core(selection["f3"]), default_support, False),
            ("h1_state_transition", with_core(selection["f4"]), default_support, False),
            (
                "h1_g1_mixed",
                with_core(
                    mixed_selection if mixed_selection else tuple(
                        field for family in FAMILIES for field in selection[family]
                    )
                ),
                default_support,
                False,
            ),
        )
    if architecture == "H2_population":
        fields = with_core(tuple(field for family in FAMILIES for field in selection[family]))
        return (
            ("h2_ordinary", fields, "S0_uniform", False),
            ("h2_equal_month", fields, "S0_uniform", True),
            ("h2_equal_archetype", fields, "S1_archetype_balance", False),
            ("h2_hard_base_error", fields, "S4_hard_base_error", False),
            ("h2_policy_state", fields, "S5_sequential_policy", False),
        )
    if architecture == "H3_hybrid_f4_f5":
        return (
            ("h3_context_f5", with_core(selection["f5"]), default_support, False),
            ("h3_context_f6", with_core(selection["f6"]), default_support, False),
            ("h3_base_query", with_core((*selection["f1"], *selection["f2"])), default_support, False),
            ("h3_recent_error", with_core(selection["f3"]), default_support, False),
            ("h3_state_support", with_core(selection["f4"]), default_support, False),
        )
    raise ValueError(f"unsupported architecture {architecture}")


def _normalise_selection_contract(
    raw: object, *, target_name: str,
) -> tuple[dict[str, list[str]], str, tuple[str, ...]]:
    """Accept a direct F1--F6 screen or a sealed G3 feature contract.

    The specialist funnel used to accept only the broader pre-screen lists.
    That made it possible to run a careful strict-OOF G3 selection and then
    silently re-expand each specialist back to all of those candidates.  A
    G3 contract is target-specific and contains the shared core in every
    family list; strip that core here because ``_head_definitions`` supplies
    it exactly once to every specialist role.
    """
    if not isinstance(raw, dict):
        raise TypeError("selected-feature contract must be a JSON object")
    if set(raw) == set(FAMILIES):
        output = {family: list(raw[family]) for family in FAMILIES}
        source = "direct_f1_f6_screen"
        mixed = ()
    elif "contracts" in raw:
        if raw.get("target") != target_name:
            raise AssertionError(
                f"G3 feature contract target {raw.get('target')!r} does not match {target_name!r}"
            )
        contracts = raw["contracts"]
        if not isinstance(contracts, dict) or not set(FAMILIES).issubset(contracts):
            raise AssertionError("G3 contract must contain one frozen list for every F1--F6 family")
        shared_core = {
            "f1_enhanced_base_bps", "f1_base_rank_ts", "f1_base_bps",
            "f1_efficiency_bps", "f1_timing_bps", "f1_e_minus_t",
            "f1_e_minus_b0", "f1_t_minus_b0", "f1_base_component_std",
        }
        output = {}
        for family in FAMILIES:
            values = contracts[family]
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise AssertionError(f"G3 {family} contract must be a list of field names")
            additions = [value for value in values if value not in shared_core]
            if len(additions) > 4:
                raise AssertionError(f"G3 {family} contract exceeds its four-addition cap")
            output[family] = additions
        mixed_values = contracts.get("mixed", [])
        if not isinstance(mixed_values, list) or not all(isinstance(value, str) for value in mixed_values):
            raise AssertionError("G3 mixed contract must be a list of field names when present")
        mixed = tuple(value for value in mixed_values if value not in shared_core)
        family_winners = {value for values in output.values() for value in values}
        if set(mixed) - family_winners:
            raise AssertionError("G3 mixed contract contains a field not retained by a family winner")
        source = "g3_strict_oof_selected_subsets"
    else:
        raise AssertionError("selected-feature contract must be direct F1--F6 lists or a sealed G3 contract")
    if any(not all(isinstance(value, str) for value in values) for values in output.values()):
        raise AssertionError("selected-feature contract includes a non-string field name")
    return output, source, mixed


def run(
    *, history_panel: Path, selection_json: Path, policy_path: Path, semantic_root: Path,
    out: Path, months: Sequence[pd.Timestamp], target_name: str, architecture: str,
    train_months: int = TRAIN_MONTHS,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    if target_name not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    if train_months <= 0:
        raise ValueError("train_months must be positive")
    raw_selection = json.loads(selection_json.read_text())
    selection, selection_source, mixed_selection = _normalise_selection_contract(raw_selection, target_name=target_name)
    support_arm = SUPPORT_BY_TARGET[target_name]
    head_definitions = _head_definitions(selection, architecture, support_arm, mixed_selection=mixed_selection)
    # Read exactly the union of fields consumed by the declared role heads.
    # A G3 contract can retain candidates from another family for a later
    # architecture (for example F6), while H1 intentionally contains no F6
    # role.  Loading the entire contract would turn an unused optional field
    # into a false hard dependency and make a fixed H1 contract impossible to
    # replay on its deliberately parent-only history panel.
    fields = tuple(dict.fromkeys(
        field for _name, role_fields, _support, _equal_month in head_definitions
        for field in role_fields
    ))
    history = _read_history(history_panel, fields)
    policy = _read_policy(policy_path)
    out.mkdir(parents=True)
    score_root = out / "target_free_scores" / target_name
    score_root.mkdir(parents=True)
    metrics: list[dict[str, object]] = []
    head_audit: list[dict[str, object]] = []
    correlations: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    for month in months:
        end = _month_end(month)
        reserve_start = month - pd.Timedelta(days=RESERVE_DAYS)
        train_start = reserve_start - pd.DateOffset(months=int(train_months))
        train = history.loc[history["__decision_ts__"].ge(train_start) & history["__decision_ts__"].lt(reserve_start)].copy()
        held = history.loc[history["__decision_ts__"].ge(month) & history["__decision_ts__"].lt(end)].copy()
        if held.empty:
            raise AssertionError(f"{month:%Y-%m}: no held target-free history")
        # Existing architecture only sends its top-30% upstream candidates to
        # the consensus stage.  This specialist stage respects that routing.
        train_route = parent._exact_timestamp_top_fraction(train, "f1_enhanced_base_bps", parent.BASE_ROUTE)
        held_route = parent._exact_timestamp_top_fraction(held, "f1_enhanced_base_bps", parent.BASE_ROUTE)
        train = train.loc[train_route.to_numpy(bool)].copy()
        held = held.loc[held_route.to_numpy(bool)].copy()
        semantics = _semantic_window(semantic_root, train_start, reserve_start)
        train = train.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        train = train.merge(semantics, on=["candidate_id", "__decision_ts__", "side_name"], how="left", validate="one_to_one")
        valid = (
            train["policy_path_valid"].fillna(False).astype(bool)
            & train["semantic_path_valid"].fillna(False).astype(bool)
            & train["policy_label_available_ts"].lt(reserve_start)
            & train["semantic_label_available_ts"].lt(reserve_start)
            & np.isfinite(pd.to_numeric(train["policy_net_bps"], errors="coerce"))
        )
        train = train.loc[valid].copy()
        if len(train) < MIN_TRAIN_ROWS or len(held) < 1_000:
            raise AssertionError(f"{target_name} {month:%Y-%m}: insufficient strict support train={len(train)} held={len(held)}")
        values = None if target_name == "T3_pair_residual_lambdarank" else _train_target(train, target_name)
        pair_train: pd.DataFrame | None = None
        if target_name == "T3_pair_residual_lambdarank":
            # Reproduce the target funnel's train-only isotonic anchor before
            # the pair sampler sees a value.  Passing raw realised policy net
            # here would silently turn the declared residual objective into a
            # different target.
            target_input = train.assign(base_rank_ts=pd.to_numeric(train["f1_base_rank_ts"], errors="coerce"))
            calibrated_residual, _grade, _objective, mode = target._anchor_and_targets(target_input, target_name)
            if mode != "pair_residual":
                raise AssertionError(f"unexpected T3 specialist target mode: {mode}")
            pair_train = train.assign(policy_net_bps=np.asarray(calibrated_residual, dtype=np.float32))
        # The selected support contract is defined against the broad base
        # percentile; retain its historical name only inside this training
        # weight calculation.  This is never emitted as a held feature.
        weight_input = train.assign(base_rank_ts=pd.to_numeric(train["f1_base_rank_ts"], errors="coerce"))
        weights_by_arm = {
            arm: support._weights(weight_input, arm)
            for arm in sorted({definition[2] for definition in head_definitions})
        }
        held_columns = list(dict.fromkeys(("candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", *fields)))
        held_score = held.loc[:, held_columns].copy()
        for head_label, head_fields, head_support, equal_month in head_definitions:
            name = f"{target_name.lower()}__{head_label}"
            if target_name == "T3_pair_residual_lambdarank":
                assert pair_train is not None
                specialist, audit = _fit_pairwise_head(
                    pair_train, head_fields, weights_by_arm[head_support], name=name,
                    support_arm=head_support, equal_month=equal_month,
                )
            else:
                assert values is not None
                specialist, audit = _fit_head(
                    train, head_fields, values, weights_by_arm[head_support], name=name,
                    support_arm=head_support, equal_month=equal_month,
                )
            raw, rank = specialist.score(held_score)
            held_score[f"{name}__raw"] = raw
            held_score[f"{name}__rank"] = rank
            audit.update({
                "target": target_name, "month": f"{month:%Y-%m}", "support_arm": head_support,
                "equal_month": equal_month,
                "reserve_start": str(reserve_start), "train_start": str(train_start),
            })
            head_audit.append(audit)
        rank_fields = [field for field in held_score if field.endswith("__rank")]
        held_score["specialist_ensemble_rank"] = np.nanmedian(held_score.loc[:, rank_fields].to_numpy(float), axis=1).astype(np.float32)
        prohibited = PROHIBITED.intersection(held_score.columns)
        if prohibited:
            raise AssertionError(f"{target_name} {month:%Y-%m}: score receipt leaked outcome fields {sorted(prohibited)}")
        target_dir = score_root / f"month={month:%Y-%m}"
        target_dir.mkdir()
        held_score.to_parquet(target_dir / "scores.parquet", index=False, compression="zstd")
        coverage.append(_coverage(held_score, fields, month))
        metrics.extend(_metric_rows(held_score, policy, target_name=target_name, month=month))
        correlations.append(_head_correlation(held_score, target_name=target_name, month=month))
        print(json.dumps({"event": "scored", "target": target_name, "month": f"{month:%Y-%m}", "train_rows": len(train), "held_rows": len(held)}), flush=True)
    pd.DataFrame(metrics).to_parquet(out / "specialist_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(head_audit).to_parquet(out / "specialist_head_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_parquet(out / "specialist_coverage.parquet", index=False, compression="zstd")
    nonempty_correlations = [part for part in correlations if not part.empty]
    correlation_frame = pd.concat(nonempty_correlations, ignore_index=True) if nonempty_correlations else pd.DataFrame(
        columns=("target", "month", "left", "right", "spearman"),
    )
    correlation_frame.to_parquet(out / "specialist_head_correlation.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline strict O3-v2 H1 family-specialist research only; no live/MC1/admission/portfolio changes",
        "architecture": architecture,
        "head_definitions": {
            name: {"fields": list(values), "support_arm": arm, "equal_month": equal_month}
            for name, values, arm, equal_month in head_definitions
        },
        "target": target_name,
        "support_arm": support_arm, "months": [f"{month:%Y-%m}" for month in months],
        "selection": selection, "selection_source": selection_source,
        "mixed_selection": list(mixed_selection),
        "routing": "exact deterministic timestamp-local top 30 percent by f1_enhanced_base_bps",
        "training": {
            "window": f"{int(train_months)} calendar months ending before a 28-day reserve", "reserve_days": RESERVE_DAYS,
            "train_months": int(train_months),
            "max_rows": TRAIN_CAP,
            "objective": "near-tie LambdaRank" if target_name == "T3_pair_residual_lambdarank" else "L2 ordinal rank-error regression",
            "query": "adjacent base near-tie pairs" if target_name == "T3_pair_residual_lambdarank" else "exact timestamp x long; preserved for complete-query sampling",
        },
        "causality": {
            "held_scores": "persisted target-free before policy labels are joined for diagnostics",
            "fit": "policy and semantic labels available strictly before reserve start only",
            "reference": "each specialist rank CDF fitted only on its sampled training prediction distribution",
            "legacy_o3": "excluded; F5 uses current/BCF parent provenance only",
        },
        "source_hashes": {
            "history_panel": _sha256(history_panel), "selection": _sha256(selection_json),
            "policy": _sha256(policy_path), "semantic": _sha256(semantic_root),
        },
    }
    fd = os.open(out / "run_manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _parse_months(raw: str) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in raw.split(",") if token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-panel", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="H1_family")
    parser.add_argument("--months", required=True, help="comma-separated held YYYY-MM months")
    parser.add_argument("--train-months", type=int, default=TRAIN_MONTHS,
                        help="strict prior consensus training window; use 3 for the declared 9+3+6 protocol")
    args = parser.parse_args()
    run(history_panel=args.history_panel, selection_json=args.selection_json, policy_path=args.policy_path,
        semantic_root=args.semantic_root, out=args.out, months=_parse_months(args.months), target_name=args.target,
        architecture=args.architecture, train_months=args.train_months)


if __name__ == "__main__":
    main()
