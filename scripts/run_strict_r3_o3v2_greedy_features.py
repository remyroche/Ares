#!/usr/bin/env python3
"""Strict-OOF greedy feature contracts for the O3-v2 correction heads.

This is G3 only.  Conditional MDA is intentionally absent because the user
assigned MDA to a separate pipeline.  Feature candidates are pre-screened on
development data, then this script greedily accepts a field only when a
six-month/28-day strict-OOF head improves the declared correction utility.
The later portability period is never inspected here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402
import run_strict_r3_o3v2_target_funnel as target  # noqa: E402


SCHEMA = "strict_r3_o3v2_greedy_features_v2"
SEED = 1729
TRAIN_MONTHS = 6
RESERVE_DAYS = 28
MIN_ROWS = 5_000
# This is a deliberately small, per-family cap.  A family must first earn
# each of these additions independently under strict OOF evaluation.  The
# cross-family pass below may then consider only the resulting family winner
# blocks; it never reopens the raw pre-screen pool.
MAX_ADDITIONS = 4
TAILS = (.01, .02, .05)
CORE = (
    "f1_enhanced_base_bps", "f1_base_rank_ts", "f1_base_bps",
    "f1_efficiency_bps", "f1_timing_bps", "f1_e_minus_t",
    "f1_e_minus_b0", "f1_t_minus_b0", "f1_base_component_std",
)
FAMILIES = ("f1", "f2", "f3", "f4", "f5", "f6")
MODES = FAMILIES
PROHIBITED = set(target.PROHIBITED_SCORE_COLUMNS)
QUERY_MODES = (
    "exact_timestamp_side",
    "exact_timestamp_baseband_side",
    "cycle_4h_side",
)
# G3 is a feature-selection successor, not a new physical-head search.  Its
# sampling must therefore reproduce the one selected cap/weight slot for the
# target under evaluation.  The tuple is deliberately small and mirrors the
# immutable target-funnel physical-slot vocabulary.
PHYSICAL_SLOT_SETTINGS: dict[str, tuple[int, str]] = {
    "cap100_ordinary": (100, "ordinary"),
    "cap80_ordinary": (80, "ordinary"),
    "cap120_equal_month": (120, "equal_month"),
    "cap40_equal_month": (40, "equal_month"),
    "cap60_equal_month": (60, "equal_month"),
}


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(f"{value.strip()}-01", tz="UTC") for value in raw.split(",") if value.strip())


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _sha256(path: Path) -> str:
    """Stream a file hash without materialising a large historical panel."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_history(path: Path, fields: Sequence[str]) -> pd.DataFrame:
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    prohibited = PROHIBITED.intersection(schema)
    if prohibited:
        raise AssertionError(f"outcome field in target-free greedy panel: {sorted(prohibited)}")
    required = {"candidate_id", "__decision_ts__", "side_name", *fields}
    if missing := required - schema:
        raise KeyError(f"greedy history missing selected fields: {sorted(missing)[:10]}")
    frame = pd.read_parquet(path, columns=sorted(required))
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("duplicate target-free history identity")
    return frame


def _load_policy(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=("candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"))
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("duplicate canonical policy identity")
    return frame


def _load_query_contract(path: Path | None, query_mode: str) -> dict[str, object] | None:
    """Validate the sealed query selector instead of silently retuning it here."""
    if query_mode not in QUERY_MODES:
        raise ValueError(f"unsupported query mode: {query_mode}")
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_o3v2_query_selection_v1":
        raise AssertionError("query contract has an unknown schema")
    selected = payload.get("selected_query_mode")
    if selected != query_mode:
        raise AssertionError(f"requested query {query_mode!r} differs from frozen selector {selected!r}")
    development_months = payload.get("development_months")
    if not isinstance(development_months, list) or not development_months:
        raise AssertionError("query contract has no declared development period")
    return payload


def _load_physical_slot_contract(
    path: Path | None,
    target_name: str,
    *,
    query_mode: str,
) -> tuple[str, int, str]:
    """Return the sealed single physical slot for one G3 target.

    The old G3 implementation hard-coded cap-100/ordinary even after the
    physical-slot selector had frozen a different head.  That made a later
    feature contract describe a different model than the one it was intended
    to refine.  A post-selector G3 run must use the sealed target slot.
    """
    if path is None:
        raise ValueError("post-selector G3 requires --physical-slot-selection")
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_o3v2_physical_slot_selection_v1":
        raise AssertionError("unknown physical-slot selection schema")
    if payload.get("query_mode") != query_mode:
        raise AssertionError(
            f"physical-slot query {payload.get('query_mode')!r} differs from requested {query_mode!r}"
        )
    slots = payload.get("selected_slots")
    if not isinstance(slots, dict) or target_name not in slots:
        raise AssertionError(f"physical-slot contract does not cover G3 target {target_name}")
    slot = str(slots[target_name])
    if slot not in PHYSICAL_SLOT_SETTINGS:
        raise AssertionError(f"physical-slot contract names unknown slot {slot!r}")
    cap, weight_mode = PHYSICAL_SLOT_SETTINGS[slot]
    return slot, cap, weight_mode


def _reference_rank(train_raw: np.ndarray, held_raw: np.ndarray) -> np.ndarray:
    return parent.ScoreReference.fit(train_raw, source="g3_strict_training_distribution").cdf(held_raw).astype(np.float32)


@dataclass(frozen=True)
class PreparedFold:
    """One immutable strict-OOF fold after target-free routing is fixed.

    ``held`` remains completely target-free.  ``held_policy`` is kept in a
    separate object and is passed only to the diagnostic metric after a held
    score receipt has been sealed.  The route and label-availability filters
    do not depend on a candidate feature trial, so preparing them once is
    exactly equivalent to rebuilding them for every candidate and avoids a
    dominant repeated string-sort cost when a selection run is resumed.
    """

    month: pd.Timestamp
    train: pd.DataFrame
    held: pd.DataFrame
    held_policy: pd.DataFrame


def _prepare_folds(
    history: pd.DataFrame,
    policy: pd.DataFrame,
    months: Sequence[pd.Timestamp],
) -> tuple[PreparedFold, ...]:
    history_start = pd.to_datetime(history["__decision_ts__"], utc=True, errors="raise").min()
    folds: list[PreparedFold] = []
    for month in months:
        reserve = month - pd.Timedelta(days=RESERVE_DAYS)
        # The reserve is a true embargo: a declared six-calendar-month fit
        # must precede it in full.  Starting from ``month - six months`` and
        # then ending at the reserve quietly shortens the fit by 28 days.
        start = reserve - pd.DateOffset(months=TRAIN_MONTHS)
        # A strict six-calendar-month contract must be fully warmed up.  A
        # partially available predecessor window is causal, but it is not the
        # declared training protocol and must not be promoted as one.
        if history_start > start:
            raise AssertionError(
                f"{month:%Y-%m}: incomplete six-month G3 training history; "
                f"requires data from {start.isoformat()}, panel begins {history_start.isoformat()}"
            )
        train = history.loc[
            history["__decision_ts__"].ge(start) & history["__decision_ts__"].lt(reserve)
        ].merge(policy, on="candidate_id", how="left", validate="one_to_one")
        held = history.loc[
            history["__decision_ts__"].ge(month) & history["__decision_ts__"].lt(_month_end(month))
        ].copy()
        train_route = parent._exact_timestamp_top_fraction(train, "f1_enhanced_base_bps", parent.BASE_ROUTE)
        held_route = parent._exact_timestamp_top_fraction(held, "f1_enhanced_base_bps", parent.BASE_ROUTE)
        train = train.loc[
            train["policy_path_valid"].fillna(False).astype(bool)
            & train["policy_label_available_ts"].lt(reserve)
            & train_route.to_numpy(bool)
            & np.isfinite(pd.to_numeric(train["policy_net_bps"], errors="coerce"))
        ].copy()
        held = held.loc[held_route.to_numpy(bool)].copy()
        held_policy = held.loc[:, ["candidate_id"]].merge(
            policy, on="candidate_id", how="left", validate="one_to_one",
        )
        if not held_policy["candidate_id"].astype(str).equals(held["candidate_id"].astype(str).reset_index(drop=True)):
            raise AssertionError(f"{month:%Y-%m}: prepared diagnostic labels changed held identity/order")
        folds.append(PreparedFold(month=month, train=train, held=held, held_policy=held_policy))
    return tuple(folds)


def _fit_score(
    train: pd.DataFrame,
    held: pd.DataFrame,
    fields: Sequence[str],
    target_name: str,
    seed: int,
    *,
    n_jobs: int,
    query_mode: str,
    physical_slot: str,
    cap: int,
    weight_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    rank = pd.to_numeric(train["f1_base_rank_ts"], errors="coerce").to_numpy(float)
    policy = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    medians = parent._fit_medians(train, fields)
    if target_name == "T3_pair_residual_lambdarank":
        anchor = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(rank, policy)
        residual = np.clip(policy - anchor.predict(rank), -500.0, 500.0).astype(np.float32)
        spec = parent.ConsensusHeadSpec(
            name=physical_slot, cap=cap, weight_mode=weight_mode, query=query_mode,
            fields=tuple(fields), target_edges_bps=(-100., -30., 30., 90.),
            params={"objective": "lambdarank", "metric": "ndcg", "n_estimators": 120,
                    "learning_rate": .035, "max_depth": 4, "num_leaves": 15,
                    "min_child_samples": max(300, int(.015 * len(train))), "feature_fraction": .82,
                    "bagging_fraction": .82, "bagging_freq": 1, "lambda_l1": .02,
                    "lambda_l2": 2., "max_bin": 127, "verbosity": -1},
        )
        pair_train = train.assign(
            policy_net_bps=residual,
            enhanced_base_bps=pd.to_numeric(train["f1_enhanced_base_bps"], errors="coerce"),
        )
        sampled, labels, groups = parent._sample_base_near_tie_pairs(pair_train, spec, "near_tie_diff50", seed=seed)
        if len(sampled) < 20 or np.unique(labels).size < 2:
            raise ValueError("insufficient declared near-tie residual pairs")
        medians = parent._fit_medians(sampled, fields)
        model = LGBMRanker(**{**spec.params, "random_state": seed, "n_jobs": n_jobs, "deterministic": True, "force_col_wise": True}).fit(
            parent._numeric_matrix(sampled, fields, medians), labels, group=groups,
        )
        train_raw = model.predict(parent._numeric_matrix(sampled, fields, medians))
    elif target_name in {"T1_economic_residual_lambdarank", "T4_hard_inversion_lambdarank"}:
        # T1 ranks the seven economic-residual states in complete timestamp ×
        # side queries.  T4 uses the same train-only target through the
        # declared hard-base-inversion pair constructor.
        anchor = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(rank, policy)
        residual = np.clip(policy - anchor.predict(rank), -500.0, 500.0).astype(np.float32)
        grade = target._economic_residual_grade(residual)
        spec = parent.ConsensusHeadSpec(
            name=physical_slot, cap=cap, weight_mode=weight_mode, query=query_mode,
            fields=tuple(fields), target_edges_bps=(-100.0, -30.0, 30.0, 90.0),
            params={
                "objective": "lambdarank", "metric": "ndcg", "n_estimators": 120,
                "learning_rate": .035, "max_depth": 4, "num_leaves": 15,
                "min_child_samples": max(300, int(.015 * len(train))), "feature_fraction": .82,
                "bagging_fraction": .82, "bagging_freq": 1, "lambda_l1": .02,
                "lambda_l2": 2., "max_bin": 127, "label_gain": [0, 1, 2, 4, 7, 12, 20],
                "lambdarank_truncation_level": 10, "verbosity": -1,
            },
        )
        source = train.assign(enhanced_base_bps=pd.to_numeric(train["f1_enhanced_base_bps"], errors="coerce"))
        pairwise_mode = "base_inversion100" if target_name == "T4_hard_inversion_lambdarank" else "none"
        heads, _ = parent._fit_heads(
            source, residual, (spec,), objective="ordinal_lambdarank", grade=grade,
            pairwise_mode=pairwise_mode, n_jobs=n_jobs,
        )
        model = heads[0]
        train_raw, _ = model.predict_rank(source)
    elif target_name in {"T2_economic_residual_ordinal", "T6_rank_error_ordinal"}:
        realised = train.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(pct=True, method="average").to_numpy(float)
        if target_name == "T2_economic_residual_ordinal":
            anchor = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(rank, policy)
            labels = target._economic_residual_grade(np.clip(policy - anchor.predict(rank), -500.0, 500.0)).astype(np.float32)
        else:
            labels = target._rank_error_grade(realised - rank).astype(np.float32)
        # Use the parent fitter rather than an all-row regressor: it applies
        # the selected cap/weight sampling contract to ordinal heads too.
        spec = parent.ConsensusHeadSpec(
            name=physical_slot, cap=cap, weight_mode=weight_mode, query=query_mode,
            fields=tuple(fields), target_edges_bps=(-100.0, -30.0, 30.0, 90.0),
            params={
                "objective": "regression_l2", "metric": "l2", "n_estimators": 120,
                "learning_rate": .035, "max_depth": 4, "num_leaves": 15,
                "min_child_samples": max(300, int(.015 * len(train))), "feature_fraction": .82,
                "bagging_fraction": .82, "bagging_freq": 1, "lambda_l1": .02,
                "lambda_l2": 2., "max_bin": 127, "verbosity": -1,
            },
        )
        heads, _ = parent._fit_heads(
            train, labels, (spec,), objective="l2_regression", grade=labels.astype(np.int32), n_jobs=n_jobs,
        )
        model = heads[0]
        train_raw, _ = model.predict_rank(train)
    else:
        raise ValueError(f"G3 supports T1/T2/T3/T4/T6, got {target_name}")
    if isinstance(model, parent.FittedConsensusHead):
        held_raw, held_rank = model.predict_rank(held)
        return held_raw.astype(np.float32), held_rank.astype(np.float32)
    held_raw = model.predict(parent._numeric_matrix(held, fields, medians))
    return held_raw.astype(np.float32), _reference_rank(np.asarray(train_raw, dtype=float), np.asarray(held_raw, dtype=float))


def _metric(score: pd.DataFrame, policy: pd.DataFrame) -> tuple[float, dict[str, float]]:
    joined = score.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    valid = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(joined["policy_net_bps"], errors="coerce"))
    work = joined.loc[valid].copy()
    value = pd.to_numeric(work["g3_mix_rank"], errors="coerce").to_numpy(float)
    outcome = pd.to_numeric(work["policy_net_bps"], errors="coerce").to_numpy(float)
    finite = np.isfinite(value) & np.isfinite(outcome)
    if finite.sum() < 100:
        return -np.inf, {"rank_ic": np.nan, "top1": np.nan, "top2": np.nan, "top5": np.nan}
    result = {"rank_ic": float(spearmanr(value[finite], outcome[finite]).statistic)}
    for tail, label in ((.01, "top1"), (.02, "top2"), (.05, "top5")):
        cut = np.quantile(value[finite], 1.0 - tail, method="higher")
        result[label] = float(np.mean(outcome[finite & (value >= cut)]))
    # The metric mirrors a correction layer's downstream use: tail economics
    # dominate, but global rank IC resolves near-equal tail alternatives.
    utility = .40 * result["top1"] + .35 * result["top2"] + .25 * result["top5"] + 25.0 * result["rank_ic"]
    return float(utility), result


def _evaluate(
    folds: Sequence[PreparedFold],
    fields: Sequence[str],
    target_name: str,
    out: Path,
    tag: str,
    *,
    n_jobs: int,
    query_mode: str,
    physical_slot: str,
    cap: int,
    weight_mode: str,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for index, fold in enumerate(folds):
        month, train, held = fold.month, fold.train, fold.held
        if len(train) < MIN_ROWS or len(held) < 1_000:
            raise AssertionError(f"{tag} {month:%Y-%m}: strict support insufficient train={len(train)} held={len(held)}")
        path = out / "target_free_scores" / tag / f"month={month:%Y-%m}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # A resumed selection must use an already sealed score receipt
            # verbatim.  Re-fitting a completed candidate would silently turn
            # a safe interruption into a different experiment.
            score = pd.read_parquet(path)
            required = {"candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", "g3_raw", "g3_rank", "g3_mix_rank"}
            if missing := required - set(score.columns):
                raise AssertionError(f"{path}: incomplete resumed target-free score receipt: {sorted(missing)}")
            score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True, errors="raise")
            if score["candidate_id"].duplicated().any() or not score["candidate_id"].astype(str).equals(held["candidate_id"].astype(str).reset_index(drop=True)):
                raise AssertionError(f"{path}: resumed identity does not exactly match its declared held fold")
        else:
            raw, rank = _fit_score(
                train, held, fields, target_name, SEED + 1009 * index + len(fields),
                n_jobs=n_jobs, query_mode=query_mode, physical_slot=physical_slot,
                cap=cap, weight_mode=weight_mode,
            )
            score = held.loc[:, ["candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts"]].copy()
            score["g3_raw"] = raw
            score["g3_rank"] = rank
            score["g3_mix_rank"] = .75 * pd.to_numeric(score["f1_base_rank_ts"], errors="coerce") + .25 * rank
            score.to_parquet(path, index=False, compression="zstd")
        utility, metric = _metric(score, fold.held_policy)
        rows.append({"tag": tag, "month": f"{month:%Y-%m}", "utility": utility, "fields": len(fields), **metric})
    frame = pd.DataFrame(rows)
    # Penalise month concentration without using future months: this is the
    # actual greedy choice score, used only across the declared development folds.
    score = float(frame["utility"].mean() - .25 * frame["utility"].std(ddof=0) - max(0.0, -float(frame["utility"].min())))
    return score, frame


def _acceptance(
    *, candidate_score: float, current_score: float, candidate_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> tuple[bool, float, int, int]:
    """Apply the common strict-OOF advancement gate to one candidate block."""
    improvement = float(candidate_score - current_score)
    paired = candidate_metrics.merge(
        baseline_metrics.loc[:, ["month", "utility"]], on="month", how="inner",
        suffixes=("", "_baseline"), validate="one_to_one",
    )
    positive_folds = int((paired["utility"] > paired["utility_baseline"] + 1e-12).sum())
    required_positive_folds = max(1, min(4, int(np.ceil(2.0 * len(paired) / 3.0))))
    accepted = bool(improvement > 0.0 and positive_folds >= required_positive_folds)
    return accepted, improvement, positive_folds, required_positive_folds


def _combine_family_winners(
    *, folds: Sequence[PreparedFold], contracts: dict[str, list[str]],
    target_name: str, out: Path, traces: list[dict[str, object]], n_jobs: int,
    query_mode: str,
    physical_slot: str,
    cap: int,
    weight_mode: str,
) -> tuple[list[str], list[str]]:
    """Greedily combine only already-earned per-family winner blocks.

    Each candidate is the full selected addition set from a remaining family,
    rather than an individual raw field.  This preserves family semantics and
    makes every retained cross-family contribution independently measurable.
    """
    additions = {
        family: tuple(field for field in contracts[family] if field not in CORE)
        for family in FAMILIES
    }
    active = list(CORE)
    selected_families: list[str] = []
    current_score, current_metrics = _evaluate(
        folds, active, target_name, out, "cross_family__core", n_jobs=n_jobs, query_mode=query_mode,
        physical_slot=physical_slot, cap=cap, weight_mode=weight_mode,
    )
    traces.extend(current_metrics.assign(
        mode="cross_family", step=0, candidate="__core__", candidate_fields="",
        accepted=True, choice_score=current_score,
    ).to_dict("records"))

    remaining = [family for family in FAMILIES if additions[family]]
    step = 0
    while remaining:
        step += 1
        trial: list[tuple[str, tuple[str, ...], float, pd.DataFrame]] = []
        for family in remaining:
            fields = additions[family]
            tag = f"cross_family__step{step:02d}__{family}"
            candidate_score, candidate_metrics = _evaluate(
                folds, (*active, *fields), target_name, out, tag, n_jobs=n_jobs, query_mode=query_mode,
                physical_slot=physical_slot, cap=cap, weight_mode=weight_mode,
            )
            trial.append((family, fields, candidate_score, candidate_metrics))
        family, fields, candidate_score, candidate_metrics = max(
            trial, key=lambda row: (row[2], row[0]),
        )
        accepted, improvement, positive_folds, required_positive_folds = _acceptance(
            candidate_score=candidate_score, current_score=current_score,
            candidate_metrics=candidate_metrics, baseline_metrics=current_metrics,
        )
        traces.extend(candidate_metrics.assign(
            mode="cross_family", step=step, candidate=family,
            candidate_fields=",".join(fields), accepted=accepted,
            choice_score=candidate_score, incremental_score=improvement,
            positive_folds=positive_folds,
            required_positive_folds=required_positive_folds,
        ).to_dict("records"))
        if not accepted:
            break
        active.extend(fields)
        selected_families.append(family)
        current_score, current_metrics = candidate_score, candidate_metrics
        remaining.remove(family)
    return active, selected_families


def run(
    *, history_panel: Path, selected_json: Path, policy_path: Path, out: Path,
    months: tuple[pd.Timestamp, ...], target_name: str, modes: tuple[str, ...],
    max_additions: int, n_jobs: int, query_mode: str = "exact_timestamp_side",
    query_contract_path: Path | None = None,
    physical_slot_selection_path: Path | None = None,
) -> None:
    if out.exists() and any((out / name).exists() for name in ("g3_feature_contracts.json", "g3_strict_oof_trace.parquet")):
        raise FileExistsError(f"completed or partially published selection output cannot be resumed: {out}")
    if set(modes) != set(FAMILIES):
        raise ValueError(
            "the two-stage G3 protocol requires exactly F1--F6; partial family runs "
            "cannot produce a valid iterative cross-family contract"
        )
    if not 1 <= max_additions <= MAX_ADDITIONS:
        raise ValueError(f"max_additions must be between 1 and {MAX_ADDITIONS}")
    query_contract = _load_query_contract(query_contract_path, query_mode)
    physical_slot, cap, weight_mode = _load_physical_slot_contract(
        physical_slot_selection_path, target_name, query_mode=query_mode,
    )
    selection = json.loads(selected_json.read_text())
    if set(selection) != {"f1", "f2", "f3", "f4", "f5", "f6"}:
        raise AssertionError("G3 requires one pre-screened selection list per F1--F6 family")
    available = tuple(dict.fromkeys((*CORE, *(field for values in selection.values() for field in values))))
    history = _load_history(history_panel, available)
    policy = _load_policy(policy_path)
    folds = _prepare_folds(history, policy, months)
    out.mkdir(parents=True, exist_ok=True)
    contracts: dict[str, list[str]] = {}
    traces: list[dict[str, object]] = []
    for mode in modes:
        candidates = [field for field in selection[mode] if field not in CORE]
        active = list(CORE)
        base_score, base_metrics = _evaluate(
            folds, active, target_name, out, f"{mode}__core", n_jobs=n_jobs, query_mode=query_mode,
            physical_slot=physical_slot, cap=cap, weight_mode=weight_mode,
        )
        traces.extend(base_metrics.assign(mode=mode, step=0, candidate="__core__", accepted=True, choice_score=base_score).to_dict("records"))
        current_score = base_score
        current_metrics = base_metrics
        for step in range(1, max_additions + 1):
            remaining = [field for field in candidates if field not in active]
            if not remaining:
                break
            trial: list[tuple[str, float, pd.DataFrame]] = []
            for candidate in remaining:
                tag = f"{mode}__step{step:02d}__{candidate}"
                candidate_score, candidate_metrics = _evaluate(
                    folds, (*active, candidate), target_name, out, tag,
                    n_jobs=n_jobs, query_mode=query_mode, physical_slot=physical_slot,
                    cap=cap, weight_mode=weight_mode,
                )
                trial.append((candidate, candidate_score, candidate_metrics))
            candidate, candidate_score, candidate_metrics = max(trial, key=lambda row: (row[1], row[0]))
            accepted, improvement, positive_folds, required_positive_folds = _acceptance(
                candidate_score=candidate_score, current_score=current_score,
                candidate_metrics=candidate_metrics, baseline_metrics=current_metrics,
            )
            traces.extend(candidate_metrics.assign(mode=mode, step=step, candidate=candidate,
                candidate_fields=candidate, accepted=accepted, choice_score=candidate_score,
                incremental_score=improvement, positive_folds=positive_folds,
                required_positive_folds=required_positive_folds).to_dict("records"))
            if not accepted:
                break
            active.append(candidate)
            current_score = candidate_score
            current_metrics = candidate_metrics
        contracts[mode] = active
    # Stage two: append each *selected family block* only when it remains
    # incrementally useful in the presence of earlier accepted family blocks.
    # ``mixed`` remains the downstream-compatible name for this final frozen
    # cross-family contract; it is no longer a raw-field mixed screen.
    cross_fields, cross_families = _combine_family_winners(
        folds=folds, contracts=contracts, target_name=target_name, out=out, traces=traces,
        n_jobs=n_jobs, query_mode=query_mode, physical_slot=physical_slot, cap=cap,
        weight_mode=weight_mode,
    )
    contracts["mixed"] = cross_fields
    pd.DataFrame(traces).to_parquet(out / "g3_strict_oof_trace.parquet", index=False, compression="zstd")
    _exclusive_json(out / "g3_feature_contracts.json", {
        "schema": SCHEMA, "target": target_name, "development_months": [f"{month:%Y-%m}" for month in months],
        "training": "six calendar months preceding a 28-day reserve, only labels available before reserve",
        "fit_runtime": {
            "lightgbm_n_jobs": n_jobs,
            "resumed_score_receipts": "validated immutable reuse only",
        },
        "routing": "exact deterministic timestamp-local top 30 percent",
        "query": query_mode,
        "physical_slot": {"name": physical_slot, "cap": cap, "weight_mode": weight_mode},
        "selection": "strict OOF greedy improvement only; no MDA; later portability period never read",
        "per_family_selection": {
            "max_additions": max_additions,
            "absolute_max_additions": MAX_ADDITIONS,
            "rule": "each field must improve the strict-OOF utility and pass the broad-fold gate",
        },
        "cross_family_selection": {
            "rule": "greedy whole-family winner blocks; retain a family only if it improves the same strict-OOF utility and broad-fold gate",
            "selected_family_order": cross_families,
            "contract_name": "mixed",
        },
        "contracts": contracts,
    })
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF G3 feature selection only; no live, MC1, admission, or portfolio mutation",
        "history_panel": str(history_panel.resolve()),
        "history_panel_sha256": _sha256(history_panel),
        "selected_json": str(selected_json.resolve()),
        "selected_json_sha256": _sha256(selected_json),
        "policy_path": str(policy_path.resolve()),
        "policy_path_sha256": _sha256(policy_path),
        "target": target_name,
        "physical_slot_selection": {
            "path": str(physical_slot_selection_path.resolve()) if physical_slot_selection_path is not None else None,
            "sha256": _sha256(physical_slot_selection_path) if physical_slot_selection_path is not None else None,
            "name": physical_slot, "cap": cap, "weight_mode": weight_mode,
        },
        "query": {
            "mode": query_mode,
            "contract_path": str(query_contract_path.resolve()) if query_contract_path is not None else None,
            "contract_sha256": _sha256(query_contract_path) if query_contract_path is not None else None,
            "selection_development_months": query_contract.get("development_months") if query_contract else None,
            "selection_lineage": "sealed prior query selector" if query_contract else "explicit legacy/default query argument",
        },
        "development_months": [f"{month:%Y-%m}" for month in months],
        "history_start": str(pd.to_datetime(history["__decision_ts__"], utc=True, errors="raise").min()),
        "training": {
            "calendar_months": TRAIN_MONTHS,
            "reserve_days": RESERVE_DAYS,
            "full_window_required": True,
            "labels": "only policy labels resolved before each fold reserve boundary",
        },
        "routing": "exact deterministic timestamp-local top 30 percent",
        "selection": {
            "max_additions_per_family": max_additions,
            "cross_family": "iterative whole-family winner blocks with the identical strict-OOF gate",
            "later_portability_read": False,
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-panel", type=Path, required=True)
    parser.add_argument("--selected-json", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", required=True, help="development OOF months")
    parser.add_argument(
        "--target",
        choices=(
            "T1_economic_residual_lambdarank", "T2_economic_residual_ordinal",
            "T3_pair_residual_lambdarank", "T4_hard_inversion_lambdarank",
            "T6_rank_error_ordinal",
        ),
        required=True,
    )
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--max-additions", type=int, default=MAX_ADDITIONS)
    parser.add_argument("--n-jobs", type=int, default=1, help="deterministic LightGBM worker count; completed receipts are never refit")
    parser.add_argument("--query-mode", choices=QUERY_MODES, default="exact_timestamp_side")
    parser.add_argument("--query-contract", type=Path, help="sealed selector; required for chronology-safe post-selector G3")
    parser.add_argument("--physical-slot-selection", type=Path, required=True, help="sealed one-slot-per-target successor contract")
    args = parser.parse_args()
    if args.max_additions <= 0 or args.max_additions > MAX_ADDITIONS:
        parser.error(f"--max-additions must be between 1 and {MAX_ADDITIONS}")
    if args.n_jobs <= 0 or args.n_jobs > 8:
        parser.error("--n-jobs must be between 1 and 8")
    modes = tuple(value for value in args.modes.split(",") if value)
    if unsupported := set(modes) - set(MODES):
        parser.error(f"unsupported modes: {sorted(unsupported)}")
    run(history_panel=args.history_panel, selected_json=args.selected_json, policy_path=args.policy_path, out=args.out,
        months=_months(args.months), target_name=args.target, modes=modes, max_additions=args.max_additions,
        n_jobs=args.n_jobs, query_mode=args.query_mode, query_contract_path=args.query_contract,
        physical_slot_selection_path=args.physical_slot_selection)


if __name__ == "__main__":
    main()
