#!/usr/bin/env python3
"""Strict-OOS heterogeneous semantic-head ablation for the enhanced base.

This is Stage H of the enhanced-base funnel.  It deliberately keeps the
immutable P2/T1 first layer, fixed MC1 class, dual +30-bps admission, and
BCF-priority constrained portfolio unchanged.  The experiment asks one narrow
question: can a small collection of different conditional-error heads improve
on five correlated residual rankers?

For every monthly fold, all heads are fit only on preceding resolved policy
outcomes with the 28-day reserve excluded.  Monthly target-free scores are
persisted before MC1 joins policy outcomes.  This script is research only; it
has no exchange or live-configuration path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_strict_r3_enhanced_base_live_stack_challenger as parent
from scripts import run_strict_r3_enhanced_base_meta2_depth as stageg
from scripts.strict_r3_research_light_portfolio import replay_fixed_controlled_auction

# LightGBM/Scikit can emit this warning for every monthly prediction after a
# NumPy-backed fit.  It is not a contract failure, but a long interactive run
# can otherwise exhaust its detached stdout pipe before it reaches the
# portfolio receipt.  Keep progress as compact JSON events instead.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
)


SCHEMA = "strict_r3_enhanced_base_semantic_heads_v1"
SEED = 1729
TRAIN_MONTHS = 6
RESERVE_DAYS = 28
TRAIN_CAP = 180_000
FIRST_HELD_MONTH = pd.Timestamp("2025-10-01T00:00:00Z")
HEAD_BLEND = 0.25
BASE_BLEND = 1.0 - HEAD_BLEND
CONFLICT_RANGE = 0.20
RESIDUAL_CLIP = 500.0
ARMS = ("h1_semantic_median", "h2_semantic_independence")


@dataclass(frozen=True)
class HeadSpec:
    name: str
    purpose: str
    kind: str
    fields: tuple[str, ...]
    direction: float
    conflict_only: bool = False


@dataclass(frozen=True)
class FittedHead:
    spec: HeadSpec
    model: object
    medians: np.ndarray
    train_rows: int
    positive_rate: float | None


OUTPUT_FIELDS = (
    "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed",
    "enhanced_base_bps", "base_rank42", "base_anchor_bps",
    "conditional_consensus_rank", "ordinary_shadow_consensus_rank", "upstream",
    "correctness_rank", "head_agreement_std",
    "head__cap100_ordinary__rank", "head__cap80_ordinary__rank",
    "head__cap120_equal_month__rank", "head__cap40_equal_month__rank",
    "head__cap60_equal_month__rank",
)

GEOMETRY_FIELDS = (
    "stage1_current_score", "stage1_bcf_score", "enhanced_base_bps",
    "base_rank42", "conditional_consensus_rank", "ordinary_shadow_consensus_rank",
    "upstream", "correctness_rank", "head_agreement_std",
    "m2_base_rank", "m2_efficiency_rank", "m2_timing_rank", "m2_rank_min",
    "m2_rank_median", "m2_rank_max", "m2_rank_range", "m2_rank_mad",
    "m2_rank_std", "m2_base_high_path_low", "m2_path_high_base_low",
    "m2_efficiency_minus_timing_rank", "m2_fraction_above_p90",
    "m2_fraction_above_p95", "m2_fraction_above_p98",
    "head__cap100_ordinary__rank", "head__cap80_ordinary__rank",
    "head__cap120_equal_month__rank", "head__cap40_equal_month__rank",
    "head__cap60_equal_month__rank",
)
FAMILY_FIELDS = (
    "m2_family_anchor_bps", "m2_family_anchor_support",
    "m2_family_score_support", "m2_family_score_ood",
    "m2_family_recent_residual_mean_3d", "m2_family_recent_residual_mean_7d",
    "m2_family_recent_residual_mean_14d", "m2_family_recent_residual_std_7d",
    "m2_family_recent_residual_slope_3d_14d",
    "m2_family_recent_residual_support_log1p_7d",
)
PATH_CONTEXT_FIELDS = (
    "distance_to_resistance_atr", "bars_to_resistance_daily_donchian",
    "post_liquidation_rebound_score", "liquidation_climax_score",
    "mkt_return_accel_1h", "mkt_ret_15m", "mkt_ret_4h", "mkt_rv_4h",
    "mkt_oi_chg_15m", "mkt_oi_chg_accel_1h",
)


def _physical_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Exclude aliases which ``stageg._family_view`` creates after reading."""

    return stageg._persisted_ledger_fields(fields)


def _head_specs() -> tuple[HeadSpec, ...]:
    """Predeclared, semantically distinct conditional-error heads."""

    common = (*GEOMETRY_FIELDS, *FAMILY_FIELDS)
    transport = (*FAMILY_FIELDS, *parent.META_STATE_FIELDS)
    return (
        HeadSpec(
            "h1_residual_value",
            "ordinary policy conversion residual relative to enhanced base",
            "residual",
            tuple(dict.fromkeys(common)),
            +1.0,
        ),
        HeadSpec(
            "h2_adverse_overconfidence",
            "probability enhanced-base residual is at or below -100 bps",
            "severe",
            tuple(dict.fromkeys((*common, *parent.META_STATE_FIELDS))),
            -1.0,
        ),
        HeadSpec(
            "h3_underconfidence",
            "probability enhanced-base residual is at or above +100 bps",
            "under",
            tuple(dict.fromkeys(GEOMETRY_FIELDS)),
            +1.0,
        ),
        HeadSpec(
            "h4_path_usability_conflict",
            "residual only where base and supportive path heads disagree",
            "residual",
            tuple(dict.fromkeys((*GEOMETRY_FIELDS, *PATH_CONTEXT_FIELDS))),
            +1.0,
            conflict_only=True,
        ),
        HeadSpec(
            "h5_state_transport",
            "probability the base-to-policy mapping avoids a severe negative residual",
            "trust",
            tuple(dict.fromkeys(transport)),
            +1.0,
        ),
    )


def _residual(frame: pd.DataFrame) -> np.ndarray:
    return np.clip(
        pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
        - pd.to_numeric(frame["enhanced_base_bps"], errors="coerce").to_numpy(float),
        -RESIDUAL_CLIP,
        RESIDUAL_CLIP,
    )


def _matrix(frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray) -> np.ndarray:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return np.where(np.isfinite(values), values, medians)


def _medians(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").median().to_numpy(float)
    return np.nan_to_num(values, nan=0.0)


def _fit_head(train: pd.DataFrame, spec: HeadSpec, family: str) -> FittedHead:
    work = train.copy()
    residual = _residual(work)
    valid = np.isfinite(residual)
    if spec.conflict_only:
        valid &= pd.to_numeric(work["m2_rank_range"], errors="coerce").to_numpy(float) >= CONFLICT_RANGE
    work = work.loc[valid].copy()
    residual = residual[valid]
    if len(work) < 5_000:
        raise ValueError(f"{family}/{spec.name}: insufficient strict-OOS training support")
    work["__target__"] = residual
    # Keep sampling reproducible across Python processes: built-in ``hash``
    # is deliberately randomised per process.
    head_offset = {
        "h1_residual_value": 101,
        "h2_adverse_overconfidence": 211,
        "h3_underconfidence": 307,
        "h4_path_usability_conflict": 401,
        "h5_state_transport": 503,
    }[spec.name]
    work = stageg._sample(work, TRAIN_CAP, SEED + head_offset + (0 if family == "current" else 17))
    medians = _medians(work, spec.fields)
    matrix = _matrix(work, spec.fields, medians)
    common = dict(
        n_estimators=140, learning_rate=.035, max_depth=3, num_leaves=15,
        min_child_samples=max(180, int(.02 * len(work))), colsample_bytree=.80,
        subsample=.82, subsample_freq=1, reg_alpha=.10, reg_lambda=8.0,
        max_bin=127, random_state=SEED + (31 if family == "current" else 37),
        n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    if spec.kind == "residual":
        model: object = LGBMRegressor(objective="huber", alpha=.90, **common).fit(matrix, work["__target__"])
        positive_rate: float | None = None
    else:
        if spec.kind == "severe":
            target = (work["__target__"].to_numpy(float) <= -100.0).astype(np.int8)
        elif spec.kind == "under":
            target = (work["__target__"].to_numpy(float) >= 100.0).astype(np.int8)
        elif spec.kind == "trust":
            target = (work["__target__"].to_numpy(float) > -100.0).astype(np.int8)
        else:
            raise AssertionError(spec.kind)
        if np.unique(target).size < 2:
            raise ValueError(f"{family}/{spec.name}: one-class target")
        model = LGBMClassifier(objective="binary", **common).fit(matrix, target)
        positive_rate = float(target.mean())
    return FittedHead(spec, model, medians, len(work), positive_rate)


def _predict_head(bundle: FittedHead, frame: pd.DataFrame) -> np.ndarray:
    matrix = _matrix(frame, bundle.spec.fields, bundle.medians)
    if bundle.spec.kind == "residual":
        raw = np.asarray(bundle.model.predict(matrix), dtype=float)
        if bundle.spec.conflict_only:
            active = np.clip(
                (pd.to_numeric(frame["m2_rank_range"], errors="coerce").to_numpy(float) - CONFLICT_RANGE)
                / max(1e-6, 1.0 - CONFLICT_RANGE),
                0.0,
                1.0,
            )
            raw *= active
    else:
        raw = np.asarray(bundle.model.predict_proba(matrix)[:, 1], dtype=float)
    return bundle.spec.direction * raw


def _rank_from_reference(reference: np.ndarray, combined: np.ndarray) -> np.ndarray:
    valid = reference[np.isfinite(reference)]
    if len(valid) < 1_000:
        raise ValueError("semantic head reference has insufficient support")
    return parent.ScoreReference.fit(valid, source="same_model_semantic_head_reference").cdf(combined).astype(np.float32)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 100:
        return 0.0
    return float(pd.Series(left[valid]).corr(pd.Series(right[valid]), method="spearman") or 0.0)


def _independence_weights(reference_ranks: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Regularised nonnegative quality × independence weights from prior data."""

    n_heads = reference_ranks.shape[1]
    if n_heads == 1:
        return np.ones(1, dtype=float)
    corr = np.corrcoef(reference_ranks, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    redundancy = np.mean(np.abs(corr), axis=1)
    quality = np.array([
        max(.02, _spearman(reference_ranks[:, index], residual))
        for index in range(n_heads)
    ])
    raw = quality / np.maximum(1.0 + redundancy, 1e-6)
    raw /= raw.sum()
    weights = .75 * np.full(n_heads, 1.0 / n_heads) + .25 * raw
    weights = np.minimum(weights, .40)
    weights /= weights.sum()
    return weights


def _semantic_score(
    reference: pd.DataFrame,
    held: pd.DataFrame,
    bundles: Sequence[FittedHead],
    *,
    arm: str,
    month: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    combined = pd.concat([reference.assign(__reference__=True), held.assign(__reference__=False)], ignore_index=True)
    raw_values = np.column_stack([_predict_head(bundle, combined) for bundle in bundles])
    ref_mask = combined["__reference__"].to_numpy(bool)
    rank_values = np.column_stack([
        _rank_from_reference(raw_values[ref_mask, index], raw_values[:, index])
        for index in range(raw_values.shape[1])
    ])
    reference_residual = _residual(reference)
    eligible = (
        reference["policy_path_valid"].fillna(False).to_numpy(bool)
        & reference["policy_label_available_ts"].lt(month).to_numpy(bool)
        & np.isfinite(reference_residual)
    )
    if arm == "h1_semantic_median":
        weights = np.full(rank_values.shape[1], 1.0 / rank_values.shape[1])
        aggregate = np.median(rank_values, axis=1)
    elif arm == "h2_semantic_independence":
        weights = _independence_weights(rank_values[ref_mask][eligible], reference_residual[eligible])
        aggregate = rank_values @ weights
    else:
        raise AssertionError(arm)
    family_score = pd.to_numeric(combined["m2_stage1_score"], errors="coerce").to_numpy(float)
    final = (BASE_BLEND * family_score + HEAD_BLEND * aggregate).astype(np.float32)
    audit = {
        "month": f"{month:%Y-%m}",
        "arm": arm,
        "reference_rows": int(ref_mask.sum()),
        "quality_rows": int(eligible.sum()),
        "weights": json.dumps({bundle.spec.name: float(weight) for bundle, weight in zip(bundles, weights)}, sort_keys=True),
        "reference_pairwise_abs_corr": float(np.mean(np.abs(np.corrcoef(rank_values[ref_mask], rowvar=False)[np.triu_indices(len(bundles), 1)]))),
    }
    held_mask = ~ref_mask
    return final[held_mask], raw_values[held_mask], audit


def _score_arm(
    ledger_root: Path,
    policy: pd.DataFrame,
    *,
    arm: str,
    out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_root = out / "target_free_scores"
    score_root.mkdir(parents=True, exist_ok=False)
    policy_index = policy.set_index("candidate_id")
    specs = _head_specs()
    physical_model_fields = tuple(dict.fromkeys(_physical_fields(tuple(field for spec in specs for field in spec.fields))))
    family_state = tuple(
        f"m2_{family}_{suffix}"
        for family in ("current", "bcf")
        for suffix in (
            "anchor_bps", "anchor_support", "score_support", "score_ood",
            "recent_residual_mean_3d", "recent_residual_mean_7d", "recent_residual_mean_14d",
            "recent_residual_std_7d", "recent_residual_slope_3d_14d",
            "recent_residual_support_log1p_7d",
        )
    )
    required = tuple(dict.fromkeys((*OUTPUT_FIELDS, *physical_model_fields, *family_state, "stage1_current_score", "stage1_bcf_score")))
    fit_rows: list[dict[str, object]] = []
    combiner_rows: list[dict[str, object]] = []
    for month in parent.SCORE_MONTHS:
        end = month + pd.offsets.MonthBegin(1)
        if month < FIRST_HELD_MONTH:
            warm = stageg._load_ledger_months(ledger_root, month, end, columns=tuple(dict.fromkeys((*OUTPUT_FIELDS, "stage1_current_score", "stage1_bcf_score"))))
            for family, field in (("current", "stage1_current_score"), ("bcf", "stage1_bcf_score")):
                output = warm.loc[:, OUTPUT_FIELDS].copy()
                output["final_score"] = pd.to_numeric(warm[field], errors="coerce").to_numpy(np.float32)
                path = score_root / family / f"month={month:%Y-%m}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                output.to_parquet(path, index=False, compression="zstd")
                fit_rows.append({"month": f"{month:%Y-%m}", "family": family, "train_rows": 0, "authority": "immutable P2/T1 warm-up"})
            continue
        reserve_start = month - pd.Timedelta(days=RESERVE_DAYS)
        train_start = month - pd.DateOffset(months=TRAIN_MONTHS)
        ref_start = month - pd.Timedelta(days=parent.BCF_REFERENCE_DAYS)
        train = stageg._stream_train_sample(ledger_root, policy_index, start=train_start, end=reserve_start, columns=required)
        reference = stageg._label_join(stageg._load_ledger_months(ledger_root, ref_start, month, columns=required), policy_index)
        held = stageg._label_join(stageg._load_ledger_months(ledger_root, month, end, columns=required), policy_index)
        print(json.dumps({"event": "semantic_month_loaded", "arm": arm, "month": f"{month:%Y-%m}", "train_rows": len(train), "reference_rows": len(reference), "held_rows": len(held)}), flush=True)
        for family in ("current", "bcf"):
            family_train = stageg._family_view(train.copy(), family)
            bundles = [_fit_head(family_train, spec, family) for spec in specs]
            family_reference = stageg._family_view(reference.copy(), family)
            family_held = stageg._family_view(held.copy(), family)
            final, raw, audit = _semantic_score(family_reference, family_held, bundles, arm=arm, month=month)
            output = family_held.loc[:, OUTPUT_FIELDS].copy()
            output["final_score"] = final
            for index, spec in enumerate(specs):
                output[f"semantic__{spec.name}"] = raw[:, index].astype(np.float32)
            forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "policy_gross_bps"}
            if forbidden.intersection(output.columns):
                raise AssertionError("semantic target-free receipt contains outcomes")
            path = score_root / family / f"month={month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(path, index=False, compression="zstd")
            audit["family"] = family
            combiner_rows.append(audit)
            for bundle in bundles:
                fit_rows.append({
                    "month": f"{month:%Y-%m}", "family": family, "head": bundle.spec.name,
                    "purpose": bundle.spec.purpose, "kind": bundle.spec.kind,
                    "train_rows": bundle.train_rows, "feature_count": len(bundle.spec.fields),
                    "positive_rate": bundle.positive_rate,
                })
            print(json.dumps({"event": "semantic_family_scored", "arm": arm, "month": f"{month:%Y-%m}", "family": family}), flush=True)
        del train, reference, held
        gc.collect()
    return pd.DataFrame(fit_rows), pd.DataFrame(combiner_rows)


def _portfolio_input(frame: pd.DataFrame, priority: str) -> pd.DataFrame:
    """Build the already-admitted, label-complete research auction surface.

    This is a literal thin equivalent of the parent adapter.  The semantic
    experiment does not alter dual admission or auction authority: BCF mapped
    EV remains the priority after both MC1 maps clear their fixed threshold.
    """

    admitted = parent._dual_admission(frame, priority).copy()
    exit_bar = pd.to_numeric(admitted["policy_exit_bar_15m"], errors="coerce").astype(int)
    decision = pd.to_datetime(admitted["__decision_ts__"], utc=True)
    return pd.DataFrame({
        "timestamp": decision,
        "candidate_id": admitted["candidate_id"].astype(str),
        "symbol": admitted["__symbol__"].astype(str),
        "normalized_rank_score": admitted["auction_rank"].to_numpy(float),
        "calibrated_score": pd.to_numeric(admitted[priority], errors="coerce").to_numpy(float),
        "exit_timestamp": decision + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
        "net_return": pd.to_numeric(admitted["policy_net_bps"], errors="coerce").to_numpy(float) / 10_000.0,
        "gross_return": pd.to_numeric(admitted["policy_gross_bps"], errors="coerce").to_numpy(float) / 10_000.0,
        "exit_reason": admitted["policy_exit_reason"].astype(str),
        "exit_price": pd.to_numeric(admitted["policy_exit_price"], errors="coerce").to_numpy(float),
    })


def _research_metrics(decisions: pd.DataFrame, equity: pd.DataFrame, arm: str, period: str) -> dict[str, object]:
    """Terminal reporting metrics for the fixed, outcome-complete auction."""

    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    net = pd.to_numeric(accepted.get("position_net_return"), errors="coerce") * 10_000.0
    timestamp = pd.to_datetime(accepted.get("timestamp"), utc=True, errors="coerce")
    monthly = net.groupby(timestamp.dt.strftime("%Y-%m"), sort=True).mean()
    weekly = net.groupby(timestamp.dt.strftime("%G-W%V"), sort=True).mean()
    wallet = pd.to_numeric(equity.get("wallet"), errors="coerce").dropna()
    drawdown = float((wallet / wallet.cummax() - 1.0).min()) if len(wallet) else float("nan")
    return {
        "arm": arm, "period": period,
        "accepted_rows": int(len(accepted)), "realised_rows": int(len(accepted)),
        "outcome_coverage": 1.0 if len(accepted) else float("nan"),
        "net_ev_bps_per_realised_trade": float(net.mean()) if len(net) else float("nan"),
        "net_sum_bps_realised": float(net.sum()) if len(net) else 0.0,
        "net_ev_bps_per_selected_trade": float(net.mean()) if len(net) else float("nan"),
        "net_sum_bps_selected": float(net.sum()) if len(net) else 0.0,
        "worst_month_bps": float(monthly.min()) if len(monthly) else float("nan"),
        "worst_week_bps": float(weekly.min()) if len(weekly) else float("nan"),
        "positive_month_fraction": float(monthly.gt(0).mean()) if len(monthly) else float("nan"),
        "max_drawdown": drawdown,
        "final_wallet": float(wallet.iloc[-1]) if len(wallet) else 1_000.0,
        "candidate_admitted_rows": int(len(decisions)),
        "admission_threshold_bps": parent.MC1_THRESHOLD_BPS,
    }


def _portfolio_metrics(frame: pd.DataFrame, label: str, period: str, out: Path) -> dict[str, object]:
    candidates = _portfolio_input(frame, "bcf_mc1_expected_bps")
    decisions, equity = replay_fixed_controlled_auction(candidates, initial_wallet=1_000.0)
    decisions.to_parquet(out / f"{label}_{period}_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{label}_{period}_equity.parquet", index=False, compression="zstd")
    metric = _research_metrics(decisions, equity, label, period)
    metric["candidate_admitted_rows"] = int(len(candidates))
    return metric


def _stream_mc1_predictions(frame: pd.DataFrame, family: str, root: Path) -> pd.DataFrame:
    """Fit the unchanged MC1 map month by month and persist thin receipts.

    The parent helper correctly implements the map but accumulates every held
    output in one in-memory frame.  This experiment has 1.5m rows per family,
    so retaining that second panel can terminate the process before the
    portfolio step.  Each monthly fit below is deliberately line-for-line
    equivalent to the parent mapping logic; only receipt persistence is
    streamed.
    """

    receipt_root = root / "mc1_predictions" / family
    if receipt_root.exists():
        existing = sorted(receipt_root.glob("month=*.parquet"))
        expected = sum(month >= pd.Timestamp("2025-10-01T00:00:00Z") for month in parent.SCORE_MONTHS)
        if len(existing) == expected:
            return pd.read_parquet(root / f"mc1_{family}_fit_audit.parquet")
        raise RuntimeError(f"incomplete {family} streamed MC1 receipts; refusing overwrite")
    receipt_root.mkdir(parents=True)
    frame = frame.copy()
    frame["score_band"] = parent._score_bands(frame)
    audit: list[dict[str, object]] = []
    output_columns = (
        [
            "candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps",
            "policy_path_valid", "policy_gross_bps", "policy_net_bps",
            "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
            "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
            "side_name", "enhanced_base_routed",
        ]
        if family == "current"
        else ["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps"]
    )
    for month in parent.SCORE_MONTHS:
        if month < pd.Timestamp("2025-10-01T00:00:00Z"):
            continue
        end = parent._month_end(month)
        train_start = month - pd.DateOffset(months=parent.MC1_TRAIN_MONTHS)
        fit = frame.loc[
            frame["__decision_ts__"].ge(train_start) & frame["__decision_ts__"].lt(month)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & frame["policy_label_available_ts"].lt(month)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        ].copy()
        held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(end)].copy()
        if len(fit) < 5_000 or held.empty:
            audit.append({"family": family, "month": f"{month:%Y-%m}", "status": "insufficient", "train_rows": int(len(fit)), "held_rows": int(len(held))})
            del fit, held
            continue
        model, medians, curve, clip = parent._fit_mc1(fit)
        matrix = held.loc[:, list(parent.MC1_FEATURES)].apply(pd.to_numeric, errors="coerce").fillna(pd.Series(medians, index=parent.MC1_FEATURES))
        held["static_expected_bps"] = model.predict(matrix)
        shifts: dict[pd.Timestamp, float] = {}
        for day in pd.date_range(month.normalize(), (end - pd.Timedelta(days=1)).normalize(), freq="D", tz="UTC"):
            history = frame.loc[
                frame["__decision_ts__"].ge(day - pd.Timedelta(days=21)) & frame["__decision_ts__"].lt(day)
                & frame["policy_path_valid"].fillna(False).astype(bool)
                & frame["policy_label_available_ts"].lt(day)
                & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
            ]
            residual = pd.to_numeric(history["policy_net_bps"], errors="coerce").to_numpy(float) - curve[history["score_band"].to_numpy(int)]
            shifts[day] = parent._robust_mean(residual, trim=.10) if len(residual) else 0.0
        held["recent_shift_bps"] = held["__decision_ts__"].dt.normalize().map(shifts).fillna(0.0)
        held["mc1_expected_bps"] = held["static_expected_bps"] + held["recent_shift_bps"]
        held.loc[:, output_columns].to_parquet(receipt_root / f"month={month:%Y-%m}.parquet", index=False, compression="zstd")
        audit.append({"family": family, "month": f"{month:%Y-%m}", "status": "scored", "train_rows": int(len(fit)), "held_rows": int(len(held)), "clip_low": clip[0], "clip_high": clip[1]})
        print(json.dumps({"event": "semantic_mc1_month", "family": family, "month": f"{month:%Y-%m}", "held_rows": len(held)}), flush=True)
        del fit, held, matrix, model
        gc.collect()
    result = pd.DataFrame(audit)
    result.to_parquet(root / f"mc1_{family}_fit_audit.parquet", index=False, compression="zstd")
    return result


def _has_complete_mc1_receipts(root: Path, family: str) -> bool:
    paths = sorted((root / "mc1_predictions" / family).glob("month=*.parquet"))
    expected = sum(month >= pd.Timestamp("2025-10-01T00:00:00Z") for month in parent.SCORE_MONTHS)
    return len(paths) == expected and (root / f"mc1_{family}_fit_audit.parquet").exists()


def _read_mc1_terminal(root: Path, family: str) -> pd.DataFrame:
    paths = sorted((root / "mc1_predictions" / family).glob("month=*.parquet"))
    if not paths:
        raise FileNotFoundError(f"missing streamed MC1 receipts for {family}")
    result = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    return result.loc[parent._evaluation_mask(result)].copy()


def _evaluate(paths: parent.Paths, out: Path, *, arm: str, fit: pd.DataFrame, combiner: pd.DataFrame) -> None:
    if _has_complete_mc1_receipts(out, "current"):
        current_audit = pd.read_parquet(out / "mc1_current_fit_audit.parquet")
        print(json.dumps({"event": "semantic_evaluate_current_mc1_reused", "arm": arm}), flush=True)
    else:
        print(json.dumps({"event": "semantic_evaluate_current_panel", "arm": arm}), flush=True)
        policy = parent._load_policy(paths)
        current_panel = parent._read_score_panels(out, "current", policy)
        current_audit = _stream_mc1_predictions(current_panel, "current", out)
        del current_panel, policy
    gc.collect()
    if _has_complete_mc1_receipts(out, "bcf"):
        bcf_audit = pd.read_parquet(out / "mc1_bcf_fit_audit.parquet")
        print(json.dumps({"event": "semantic_evaluate_bcf_mc1_reused", "arm": arm}), flush=True)
    else:
        print(json.dumps({"event": "semantic_evaluate_bcf_panel", "arm": arm}), flush=True)
        policy = parent._load_policy(paths)
        bcf_panel = parent._read_score_panels(out, "bcf", policy)
        bcf_audit = _stream_mc1_predictions(bcf_panel, "bcf", out)
        del bcf_panel, policy
    gc.collect()
    # The per-month MC1 fits retain thin immutable receipts. Terminal reporting
    # reads only their predeclared evaluation months and column surface.
    current_columns = [
        "candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps",
        "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
        "side_name", "enhanced_base_routed",
    ]
    bcf_columns = ["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps"]
    policy = parent._load_policy(paths)
    baseline = parent._baseline(paths, policy)
    del policy
    gc.collect()
    # The predeclared semantic question is a matched comparison versus the
    # live-like control.  Filter each streamed MC1 family to that immutable
    # identity set *before* its wide two-family merge.  A broader 825k-row
    # coverage merge would provide no delta and can exceed the bounded
    # research worker memory budget.
    baseline_ids = pd.Index(baseline["candidate_id"].astype(str).unique())
    current = _read_mc1_terminal(out, "current")
    current = current.loc[current["candidate_id"].astype(str).isin(baseline_ids), current_columns].copy()
    bcf = _read_mc1_terminal(out, "bcf")
    bcf = bcf.loc[bcf["candidate_id"].astype(str).isin(baseline_ids), bcf_columns].copy()
    print(json.dumps({"event": "semantic_evaluate_matched_mc1", "arm": arm, "current_rows": len(current), "bcf_rows": len(bcf), "baseline_rows": len(baseline)}), flush=True)
    challenger = parent._combined_challenger(current, bcf)
    matched = challenger
    rows: list[dict[str, object]] = []
    for period, (start, end) in parent.EVALUATION_PERIODS.items():
        for label, part in (("live_baseline", baseline), ("semantic_matched_stack", matched)):
            rows.append(_portfolio_metrics(part.loc[part["__decision_ts__"].ge(start) & part["__decision_ts__"].lt(end)].copy(), label, period, out))
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(out / "live_like_portfolio_metrics.parquet", index=False, compression="zstd")
    left = metrics.loc[metrics["arm"].eq("live_baseline")].set_index("period")
    right = metrics.loc[metrics["arm"].eq("semantic_matched_stack")].set_index("period")
    delta = pd.DataFrame({"period": left.index.intersection(right.index)})
    for field in ("accepted_rows", "realised_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown"):
        delta[f"delta_{field}"] = right.loc[delta["period"], field].to_numpy(float) - left.loc[delta["period"], field].to_numpy(float)
    delta.to_parquet(out / "delta_vs_live_baseline.parquet", index=False, compression="zstd")
    fit.to_parquet(out / "semantic_head_fit_audit.parquet", index=False, compression="zstd")
    combiner.to_parquet(out / "semantic_combiner_audit.parquet", index=False, compression="zstd")
    pd.concat([current_audit, bcf_audit], ignore_index=True).to_parquet(out / "mc1_fit_audit.parquet", index=False, compression="zstd")
    sample = pd.read_parquet(next((out / "target_free_scores" / "current").glob("*.parquet")))
    if {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "policy_gross_bps"}.intersection(sample.columns):
        raise AssertionError("semantic target-free receipt contains outcome field")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline research only; no live configuration changes",
        "arm": arm,
        "heads": [{"name": spec.name, "purpose": spec.purpose, "kind": spec.kind, "conflict_only": spec.conflict_only} for spec in _head_specs()],
        "target": "policy net minus strict-OOF enhanced_base_bps; semantic binary heads use +/-100-bps residual boundaries",
        "combiner": "median semantic rank" if arm == "h1_semantic_median" else "quality x independence weighted semantic rank; 75% shrink to equal, 40% max head weight",
        "integration": "75% immutable P2/T1 family score + 25% heterogeneous semantic aggregate",
        "reserve": f"{RESERVE_DAYS} days excluded from every supervised head fit",
        "downstream": "fixed MC1 class/hyperparameters, dual current/BCF >= +30 bps, BCF-MC1 priority, canonical constrained portfolio",
        "causality": {"target_free_score_receipts": True, "no_held_window_percentile": True, "combiner_quality": "prior reference rows with labels resolved before held month"},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run(args: argparse.Namespace) -> None:
    root = args.out.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    ledger_root, _ = stageg._build_target_free_ledger(args.p2_score_root.resolve(), args.target_free_source.resolve(), args.policy_root.resolve(), args.stageg_root.resolve())
    for arm in args.arms:
        arm_out = root / arm
        if arm_out.exists():
            if not args.resume:
                raise FileExistsError(f"semantic arm already exists: {arm_out}")
            current_months = sorted((arm_out / "target_free_scores" / "current").glob("month=*.parquet"))
            bcf_months = sorted((arm_out / "target_free_scores" / "bcf").glob("month=*.parquet"))
            if len(current_months) != len(parent.SCORE_MONTHS) or len(bcf_months) != len(parent.SCORE_MONTHS):
                raise RuntimeError(f"{arm}: incomplete target-free receipt; refusing to overwrite it")
            # A detached interactive process can be interrupted after score
            # persistence but before audit writing.  Reuse those immutable
            # score receipts; only the deterministic MC1/portfolio consumer
            # is resumed.  Empty audit tables are explicitly labelled rather
            # than reconstructing training claims from memory.
            fit_path = arm_out / "semantic_head_fit_audit.parquet"
            combiner_path = arm_out / "semantic_combiner_audit.parquet"
            fit = pd.read_parquet(fit_path) if fit_path.exists() else pd.DataFrame({
                "resume_note": ["target-free semantic scores completed before detached-output interruption"]
            })
            combiner = pd.read_parquet(combiner_path) if combiner_path.exists() else pd.DataFrame({
                "resume_note": ["target-free semantic scores completed before detached-output interruption"]
            })
            print(json.dumps({"event": "semantic_evaluation_resume", "arm": arm}), flush=True)
        else:
            arm_out.mkdir(parents=True)
            policy = stageg._policy_labels(args.policy_root.resolve())
            fit, combiner = _score_arm(ledger_root, policy, arm=arm, out=arm_out)
        paths = parent.Paths(
            raw_ledger=args.raw_ledger.resolve(), direct_root=args.direct_root.resolve(), policy_root=args.policy_root.resolve(),
            current_mc1=args.current_mc1.resolve(), bcf_mc1=args.bcf_mc1.resolve(), bundle_root=args.bundle_root.resolve(),
        )
        _evaluate(paths, arm_out, arm=arm, fit=fit, combiner=combiner)
        gc.collect()
    (root / "run_manifest.json").write_text(json.dumps({
        "schema": SCHEMA, "scope": "offline research only", "arms": list(args.arms),
        "stageg_target_free_ledger": str(ledger_root),
        "p2_score_root": str(args.p2_score_root.resolve()),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stageg-root", type=Path, required=True)
    parser.add_argument("--p2-score-root", type=Path, required=True)
    parser.add_argument("--target-free-source", type=Path, required=True)
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--current-mc1", type=Path, required=True)
    parser.add_argument("--bcf-mc1", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
