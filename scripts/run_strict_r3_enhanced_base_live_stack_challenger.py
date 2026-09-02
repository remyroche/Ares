#!/usr/bin/env python3
"""Strict-OOS enhanced-base reconstruction of the live BCF/current stack.

Research-only challenger.  It replaces the strict-R3 base coordinate with the
already strict-OOS equal B0/efficiency/timing direct base score, then re-fits the
downstream layers using only prior resolved rich-policy labels:

    enhanced base -> top-30% route -> five selected residual LambdaRank heads
    -> 75/25 upstream -> correctness demotion -> current score CDF28
    -> BCF-style score CDF42 -> family-specific MC1 maps -> dual >=30 bps
    -> BCF-MC1 priority under the existing constrained portfolio mirror.

The script deliberately does not touch any live/canonical artifact.  Its live
baseline is the immutable current-v5/BCF dual mapper ledger, replayed through
the same policy and portfolio adapter over exactly the same held periods.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.research.consensus_ensemble import agreement_statistics  # noqa: E402


SCHEMA = "strict_r3_enhanced_base_live_stack_challenger_v1"
SEED = 1729
BASE_ROUTE = 0.30
RESERVE_DAYS = 28
BCF_REFERENCE_DAYS = 42
META_TRAIN_MONTHS = 6
MAP_TRAIN_MONTHS = 6
MC1_TRAIN_MONTHS = 6
MC1_THRESHOLD_BPS = 30.0
HEAD_TRAIN_CAP = 240_000
CORRECTNESS_TRAIN_CAP = 180_000
BASE_BLEND_WEIGHT = 0.75
CONSENSUS_BLEND_WEIGHT = 0.25
CORRECTNESS_HURDLE_BPS = 100.0
CORRECTNESS_TRAIN_FRACTION = 0.30
CORRECTNESS_FLOOR = 0.25
CORRECTNESS_SPAN = 0.75
TAIL_TRUST_MAX_DEMOTION = 0.10
TAIL_TRUST_SEVERE_BPS = -100.0
TAIL_TRUST_QUANTILE = 0.20
RESIDUAL_EDGES = (-100.0, -30.0, 30.0, 90.0)
SCORE_MONTHS = tuple(pd.date_range("2025-07-01", "2026-07-01", freq="MS", tz="UTC"))
EVALUATION_PERIODS = {
    "2025_q4": (pd.Timestamp("2025-10-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "2026_aprjul": (pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
CONSENSUS_CONTRACT = ROOT / "config" / "strict_r3_enhanced_base_consensus_top5_v1.json"
CONSENSUS_SCHEMA = "strict_r3_conditional_consensus_v1"
CONSENSUS_SUBSET_SCHEMA = "strict_r3_conditional_consensus_subset_v1"
EXPECTED_RESEARCH_HEADS = 5
CONSENSUS_TARGET_EDGES = (-150.0, -50.0, 50.0, 150.0)
# Frozen MC1_d2 input contract.  Keep a local literal so target-free score
# production does not import mapper/live modules merely to fetch a tuple.
MC1_FEATURES = (
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
)


@dataclass(frozen=True)
class ScoreReference:
    """Small immutable empirical CDF; kept local to avoid importing live code."""

    sorted_values: np.ndarray
    source: str

    @classmethod
    def fit(cls, values: Sequence[float], *, source: str) -> "ScoreReference":
        array = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)], kind="stable")
        if len(array) < 2:
            raise ValueError(f"{source} has insufficient finite score support")
        return cls(array, source)

    def cdf(self, values: Sequence[float]) -> np.ndarray:
        current = np.asarray(values, dtype=float)
        result = np.full(len(current), np.nan, dtype=float)
        valid = np.isfinite(current)
        left = np.searchsorted(self.sorted_values, current[valid], side="left")
        right = np.searchsorted(self.sorted_values, current[valid], side="right")
        result[valid] = (0.5 * (left + right) + 0.5) / len(self.sorted_values)
        return np.clip(result, 0.0, 1.0)


@dataclass(frozen=True)
class ConsensusHeadSpec:
    name: str
    cap: int
    weight_mode: str
    query: str
    fields: tuple[str, ...]
    target_edges_bps: tuple[float, ...]
    params: dict[str, object]


@dataclass(frozen=True)
class PolicyConversionLabelSpec:
    """Strictly post-resolution supervision for the consensus correction.

    These values are *never* written to the target-free score panels.  They
    are joined only while fitting a fold whose labels were available before the
    fold reserve begins.  The first specification intentionally reproduces
    the label used by the historical enhanced-stack code, rather than the
    stale parent-JSON declaration (which incorrectly said -150/-50/+50/+150).
    """

    name: str
    description: str
    source: str
    edges_bps: tuple[float, ...]
    objective: str = "ordinal_lambdarank"
    clip_abs_bps: float | None = None

    def values(self, frame: pd.DataFrame) -> np.ndarray:
        net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
        if self.source == "policy_residual":
            anchor = pd.to_numeric(frame["base_anchor_bps"], errors="coerce").to_numpy(float)
            value = net - anchor
        elif self.source == "policy_net":
            value = net
        elif self.source == "enhanced_residual":
            base = pd.to_numeric(frame["enhanced_base_bps"], errors="coerce").to_numpy(float)
            value = net - base
        else:
            raise ValueError(f"unsupported policy-conversion label source: {self.source}")
        if self.clip_abs_bps is not None:
            value = np.clip(value, -float(self.clip_abs_bps), float(self.clip_abs_bps))
        return value


# The direct-net labels are deliberately simple ordinal policy-conversion
# labels.  ``direct_economic`` is the economic four-boundary version used in
# prior conversion research: severe loss / loss / marginal clear / robust
# clear.  We do not use exit reason or a path descriptor here: they would
# change the policy target into a hindsight path classifier rather than the
# realised executable-policy outcome.
POLICY_CONVERSION_LABEL_SPECS: dict[str, PolicyConversionLabelSpec] = {
    "residual_actual_100_30_90": PolicyConversionLabelSpec(
        name="residual_actual_100_30_90",
        description="historical executed control: policy net minus prequential base anchor",
        source="policy_residual",
        edges_bps=(-100.0, -30.0, 30.0, 90.0),
    ),
    "direct_policy_100_30_90": PolicyConversionLabelSpec(
        name="direct_policy_100_30_90",
        description="direct canonical policy net using the historical residual grade boundaries",
        source="policy_net",
        edges_bps=(-100.0, -30.0, 30.0, 90.0),
    ),
    "direct_policy_economic_200_0_50_150": PolicyConversionLabelSpec(
        name="direct_policy_economic_200_0_50_150",
        description="direct canonical policy net: severe loss / nonpositive / marginal / robust clear",
        source="policy_net",
        edges_bps=(-200.0, 0.0, 50.0, 150.0),
    ),
    "enhanced_residual_l2_clip500": PolicyConversionLabelSpec(
        name="enhanced_residual_l2_clip500",
        description="policy net minus strict-OOF enhanced base; clipped at +/-500 bps; L2 regression",
        source="enhanced_residual",
        edges_bps=(-100.0, -30.0, 30.0, 90.0),
        objective="l2_regression",
        clip_abs_bps=500.0,
    ),
    "enhanced_residual_ordinal_clip500_100_30_90": PolicyConversionLabelSpec(
        name="enhanced_residual_ordinal_clip500_100_30_90",
        description="policy net minus strict-OOF enhanced base; clipped at +/-500 bps; ordinal LambdaRank",
        source="enhanced_residual",
        edges_bps=(-100.0, -30.0, 30.0, 90.0),
        objective="ordinal_lambdarank",
        clip_abs_bps=500.0,
    ),
    "enhanced_residual_huber_clip500": PolicyConversionLabelSpec(
        name="enhanced_residual_huber_clip500",
        description="policy net minus strict-OOF enhanced base; clipped at +/-500 bps; robust Huber regression",
        source="enhanced_residual",
        edges_bps=(-100.0, -30.0, 30.0, 90.0),
        objective="huber_regression",
        clip_abs_bps=500.0,
    ),
}


def _validate_policy_label_pairwise_compatibility(
    label_spec: PolicyConversionLabelSpec,
    pairwise_mode: str,
) -> None:
    """Prevent a declared label ablation from being silently ignored.

    Near-tie mode has its own fixed realised-policy ordering target.  It is a
    valid pairwise-control objective, but not a policy-conversion-label sweep.
    """

    if pairwise_mode != "none" and label_spec.name != "residual_actual_100_30_90":
        raise ValueError(
            "policy-conversion label ablations require --pairwise-mode none; "
            "near-tie mode always trains the fixed realised-policy ordering target"
        )


# A score-architecture switch is deliberately separate from the label switch.
# It lets the replacement meta research falsify the generic correctness layer
# without changing the base source, head fields, MC1 model class, admission,
# or portfolio policy.  The historical default preserves the previous runner.
SCORE_ARCHITECTURES = (
    "base_only",
    "base_consensus_no_correctness",
    "base_consensus_correctness",
)

# Stage F is deliberately an authority ablation, not a new alpha model.  A
# tail-trust arm may only lower the current-family upstream score of routed
# candidates.  MC1, its feature shape, dual admission, and the BCF auction
# remain unchanged.
TRUST_ARMS = (
    "none",
    "generic_correctness",
    "severe_overconfidence",
    "lower_quantile",
    "severe_overconfidence_support",
)


# Stage E deliberately changes only the selected five residual heads' input
# contract.  The strict 120-field upstream source, generic correctness model,
# MC1 model class, admission maps and auction remain frozen.  This avoids
# calling a broad second alpha model a residual-feature experiment.
META_FEATURE_CONTRACTS = (
    "current",
    "raw_heavy",
    "geometry_only",
    "geometry_score_ood",
    "geometry_recent_calibration",
    "geometry_score_ood_state",
    "geometry_score_ood_state_raw",
    "geometry_score_ood_recent_calibration_state_raw",
)
META_SCORE_FIELDS = (
    "enhanced_base_bps", "base_rank_ts", "base_bps", "efficiency_bps", "timing_bps",
    "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std",
)
META_SCORE_OOD_FIELDS = (
    "meta_score_ood_l1", "meta_score_ood_max", "meta_score_support_proxy",
)
# These fields are an operating-state summary, not an outcome leak.  For a
# decision at t they aggregate only policy outcomes whose declared
# ``policy_label_available_ts`` is strictly earlier than t.  The per-fold base
# anchor is frozen before the summary is calculated, so "residual" always
# means realised policy net minus that fold's *pre-existing* base map.
META_RECENT_CALIBRATION_FIELDS = (
    "meta_recent_base_residual_mean_3d",
    "meta_recent_base_residual_mean_7d",
    "meta_recent_base_residual_mean_14d",
    "meta_recent_base_residual_std_7d",
    "meta_recent_base_residual_slope_3d_14d",
    "meta_recent_base_rank_policy_corr_7d",
    "meta_recent_efficiency_residual_mean_7d",
    "meta_recent_timing_residual_mean_7d",
    "meta_recent_calibration_support_log1p_7d",
)
# These are stable causal market-state fields in the frozen base contract, not
# bundle-local K9 IDs or geometry memberships whose semantics vary by bundle.
META_STATE_FIELDS = (
    "state_spectral_eig_condition", "state_spectral_eig_gap_1_2",
    "state_spectral_eig_top3_share", "eig_effective_rank__open_interest",
    "prior_volatility", "mkt_rv_4h", "mkt_oi_dispersion_1h",
    "mkt_oi_dispersion_24h", "cross_asset_corr_1h",
    "cross_asset_downside_corr_4h", "negative_breadth_pct",
    "market_breadth_drawdown_from_6h_max",
)
# A compact, predefined market-context block is allowed only in F5.  It is
# chosen for distinct liquidity, leverage, breadth, volatility and structural
# price roles; it is not target-selected on held outcomes.
META_RAW_CONTEXT_FIELDS = (
    "mark_perp_dislocation", "mkt_return_accel_1h", "mkt_ret_15m", "mkt_ret_4h",
    "mkt_oi_chg_15m", "mkt_oi_chg_accel_1h", "mkt_oi_flush_z_30d",
    "mkt_ret_per_oi_change_1h", "pct_assets_up_15m", "pct_assets_up_4h",
    "pct_assets_price_up_oi_down_1h", "pct_assets_price_down_oi_down_1h",
    "pct_assets_extreme_oi_drop_1h", "liquidation_climax_score",
    "post_liquidation_rebound_score", "spike_score_surprise", "grind_score_surprise",
    "distance_to_resistance_atr", "bars_to_resistance_daily_donchian",
    "q_lower_tail__ob_spread_bps_z_24h", "q_lower_tail__ob_depth_usd_l20_z",
    "xs_dispersion__ob_spread_bps_z_24h", "xs_dispersion__rvol_z",
    "xs_dispersion__xasset_ob_liquidity_ts_resid",
)

# Pairwise correction is a deliberately narrow research objective.  The
# underlying ranker sees only pairs the enhanced base regards as comparable;
# it cannot obtain a result simply by rebuilding the broad base ordering.
# ``none`` preserves the frozen ordinary LambdaRank control.
PAIRWISE_MODES = (
    "none",
    "near_tie",
    "near_tie_diff50",
    "near_tie_diff100",
    "base_inversion100",
)


@dataclass(frozen=True)
class BpsIntegrationSpec:
    """Frozen authority for converting a calibrated meta residual into bps.

    The map from consensus raw score to residual bps is fitted only on the
    immediately preceding resolved reserve.  These choices therefore test
    *integration*, not another target or model search.
    """

    name: str
    description: str
    scale: float
    lower_bps: float | None = None
    upper_bps: float | None = None

    @property
    def is_rank_control(self) -> bool:
        return self.name == "rank_75_25"

    def correction(self, residual_bps: Sequence[float]) -> np.ndarray:
        value = self.scale * np.asarray(residual_bps, dtype=float)
        if self.lower_bps is not None or self.upper_bps is not None:
            value = np.clip(
                value,
                -np.inf if self.lower_bps is None else float(self.lower_bps),
                np.inf if self.upper_bps is None else float(self.upper_bps),
            )
        return value


# Stage D is intentionally a five-arm, predeclared funnel: one rank-space
# control and four interpretable common-bps correction authorities.
BPS_INTEGRATION_SPECS: dict[str, BpsIntegrationSpec] = {
    "rank_75_25": BpsIntegrationSpec(
        "rank_75_25", "frozen 75/25 base-rank / consensus-rank control", 0.0,
    ),
    "additive_025": BpsIntegrationSpec(
        "additive_025", "base anchor plus 0.25 times reserve-calibrated residual", 0.25,
    ),
    "additive_050": BpsIntegrationSpec(
        "additive_050", "base anchor plus 0.50 times reserve-calibrated residual", 0.50,
    ),
    "clip_50": BpsIntegrationSpec(
        "clip_50", "base anchor plus residual clipped to [-50,+50] bps", 1.0, -50.0, 50.0,
    ),
    "clip_half_100": BpsIntegrationSpec(
        "clip_half_100", "base anchor plus 0.50 residual clipped to [-100,+100] bps", 0.50, -100.0, 100.0,
    ),
}


@dataclass
class FittedConsensusHead:
    spec: ConsensusHeadSpec
    medians: np.ndarray
    model: object
    score_reference: ScoreReference

    def predict_rank(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = self.model.predict(_numeric_matrix(frame, self.spec.fields, self.medians))
        return raw.astype(np.float32), self.score_reference.cdf(raw).astype(np.float32)


def _numeric_matrix(frame: pd.DataFrame, fields: Sequence[str], medians: Sequence[float] | None = None) -> np.ndarray:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    filled = values.median().fillna(0.0).to_numpy(dtype=np.float32) if medians is None else np.asarray(medians, dtype=np.float32)
    if len(filled) != len(fields):
        raise ValueError("imputation medians do not match field contract")
    return values.fillna(pd.Series(filled, index=fields)).fillna(0.0).to_numpy(dtype=np.float32)


def _fit_medians(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    return frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median().fillna(0.0).to_numpy(dtype=np.float32)


def _canonical_base_hash(fields: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(fields), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_conditional_consensus_contract(base_fields: Sequence[str], *, side: str = "long") -> tuple[ConsensusHeadSpec, ...]:
    """Load the immutable parent contract and enforce the research top-five selector."""

    selector = json.loads(CONSENSUS_CONTRACT.read_text())
    if selector.get("schema") != CONSENSUS_SUBSET_SCHEMA or selector.get("side") != side:
        raise ValueError("not the enhanced-base long five-head selector")
    parent_name = str(selector.get("parent_contract", ""))
    parent = (CONSENSUS_CONTRACT.parent / parent_name).resolve()
    if parent.parent != CONSENSUS_CONTRACT.parent.resolve() or not parent.is_file():
        raise ValueError("five-head selector references an invalid parent contract")
    parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    if parent_hash != selector.get("parent_contract_sha256"):
        raise ValueError("five-head selector parent hash mismatch")
    payload = json.loads(parent.read_text())
    if payload.get("schema") != CONSENSUS_SCHEMA or payload.get("side") != side:
        raise ValueError("five-head selector parent is not the frozen long consensus contract")
    names = tuple(str(value) for value in selector.get("selected_heads", []))
    if len(names) != EXPECTED_RESEARCH_HEADS or len(set(names)) != len(names):
        raise ValueError("five-head selector is incomplete or has duplicate heads")
    available = {str(raw.get("name")): raw for raw in payload.get("heads", [])}
    if set(names) - set(available):
        raise ValueError("five-head selector contains an unknown parent head")
    if selector.get("base_contract_sha256") != _canonical_base_hash(base_fields):
        raise ValueError("five-head selector targets another base contract")
    if payload.get("base_contract_sha256") != _canonical_base_hash(base_fields):
        raise ValueError("conditional-consensus fields target another base contract")
    edges = tuple(float(value) for value in payload["target"]["edges_bps"])
    if edges != CONSENSUS_TARGET_EDGES:
        raise ValueError("frozen conditional-consensus target edges changed")
    params = dict(payload["ranker_params"])
    output: list[ConsensusHeadSpec] = []
    for name in names:
        raw = available[name]
        indices = tuple(int(value) for value in raw["field_indices"])
        if not indices or min(indices) < 0 or max(indices) >= len(base_fields):
            raise ValueError(f"{raw['name']} has invalid base-field indices")
        output.append(ConsensusHeadSpec(str(raw["name"]), int(raw["cap"]), str(raw["weight_mode"]), str(raw["query"]), tuple(str(base_fields[index]) for index in indices), edges, params))
    if len(output) != EXPECTED_RESEARCH_HEADS or len({spec.name for spec in output}) != EXPECTED_RESEARCH_HEADS:
        raise ValueError("conditional-consensus contract is not the exact selected five-head set")
    return tuple(output)


def _residual_grade(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    return np.select([np.asarray(values, dtype=float) <= float(edge) for edge in edges], [0, 1, 2, 3], default=4).astype(np.int32)


def _query(frame: pd.DataFrame, mode: str) -> pd.Series:
    timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True)
    if mode == "exact_timestamp_side":
        token = timestamp
    elif mode == "exact_timestamp_baseband_side":
        # Research-only causal localisation.  The band is calculated from the
        # decision-time enhanced-base percentile and deliberately uses fixed
        # boundaries rather than fold-specific quantiles.  It is never based
        # on a realised exit or a downstream outcome.
        if "base_rank_ts" not in frame.columns:
            raise KeyError("exact_timestamp_baseband_side requires base_rank_ts")
        rank = pd.to_numeric(frame["base_rank_ts"], errors="coerce").fillna(-np.inf)
        band = np.select(
            [rank.ge(.98), rank.ge(.95), rank.ge(.90), rank.ge(.80), rank.ge(.70)],
            ["98_100", "95_98", "90_95", "80_90", "70_80"],
            default="below_70",
        )
        token = timestamp.astype(str) + "|" + band.astype(str)
    elif mode == "cycle_4h_side":
        token = timestamp.dt.floor("4h")
    else:
        raise ValueError(f"unsupported frozen query: {mode}")
    return token.astype(str) + "|" + frame["side_name"].astype(str).str.lower()


def _sample_complete_consensus_queries(
    frame: pd.DataFrame,
    grade: np.ndarray,
    spec: ConsensusHeadSpec,
    *,
    seed: int,
    cap: int = HEAD_TRAIN_CAP,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Apply the frozen 240k cap without splitting LambdaRank queries."""

    work = frame.copy()
    work["__target_grade__"] = np.asarray(grade, dtype=np.int32)
    work["__query__"] = _query(work, spec.query).to_numpy()
    work["__month__"] = pd.to_datetime(work["__decision_ts__"], utc=True).dt.to_period("M").astype(str)
    work = work.loc[work["__query__"].map(work["__query__"].value_counts()).ge(2)].copy()
    if work.empty:
        raise ValueError(f"{spec.name} lacks multi-row query support")
    if len(work) > cap:
        meta = work.groupby("__query__", sort=False).agg(rows=("candidate_id", "size"), month=("__month__", "first"), first_ts=("__decision_ts__", "min")).reset_index()
        generator = np.random.default_rng(seed)
        retained: list[str] = []
        if spec.weight_mode == "equal_month":
            allowance = max(2, cap // max(meta["month"].nunique(), 1))
            for _, part in meta.groupby("month", sort=True):
                used = 0
                for row in part.assign(__random__=generator.random(len(part))).sort_values(["__random__", "first_ts", "__query__"], kind="stable").to_dict("records"):
                    if used + int(row["rows"]) <= allowance:
                        retained.append(str(row["__query__"])); used += int(row["rows"])
        else:
            used = 0
            for row in meta.assign(__random__=generator.random(len(meta))).sort_values(["__random__", "first_ts", "__query__"], kind="stable").to_dict("records"):
                if used + int(row["rows"]) <= cap:
                    retained.append(str(row["__query__"])); used += int(row["rows"])
        work = work.loc[work["__query__"].isin(retained)].copy()
    work = work.sort_values(["__query__", "__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    groups = work.groupby("__query__", sort=False).size().to_numpy(dtype=np.int32)
    return work, work.pop("__target_grade__").to_numpy(dtype=np.int32), groups


def _sample_base_near_tie_pairs(
    frame: pd.DataFrame,
    spec: ConsensusHeadSpec,
    mode: str,
    *,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return deterministic binary LambdaRank pairs near the base ordering.

    P1--P4 are intentionally *not* generic candidate rankers.  The source
    sample first preserves the head's frozen query/cap/weighting contract.
    Within each retained decision timestamp we compare only adjacent
    enhanced-base ranks; their percentile distance is at most 10%.  Outcomes
    appear only here, inside a pre-reserve training fold.
    """
    if mode not in set(PAIRWISE_MODES) - {"none"}:
        raise ValueError(f"unsupported pairwise mode: {mode}")
    identity_columns = ["candidate_id", "__decision_ts__", "side_name"]
    if spec.query == "exact_timestamp_baseband_side":
        identity_columns.append("base_rank_ts")
    identity = frame.loc[:, identity_columns].copy()
    # This grade is used only to preserve the historical complete-query
    # sampling strata.  It is never the pairwise training label.
    sampling_grade = _residual_grade(
        pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float),
        (-100.0, -30.0, 30.0, 90.0),
    )
    selected, _, _ = _sample_complete_consensus_queries(
        identity, sampling_grade, spec, seed=seed, cap=HEAD_TRAIN_CAP // 2,
    )
    source_columns = tuple(dict.fromkeys((
        "candidate_id", "__decision_ts__", "side_name", "policy_net_bps", "enhanced_base_bps", *spec.fields,
    )))
    source = frame.set_index("candidate_id", drop=False).loc[
        selected["candidate_id"].to_numpy(),
        list(source_columns),
    ].copy().reset_index(drop=True)
    source["__month__"] = pd.to_datetime(source["__decision_ts__"], utc=True).dt.to_period("M").astype(str)
    source = source.sort_values(
        ["__decision_ts__", "enhanced_base_bps", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    ).reset_index(drop=True)
    count = source.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(int)
    next_source = source.shift(-1)
    same_timestamp = source["__decision_ts__"].eq(next_source["__decision_ts__"]).to_numpy(bool)
    # Adjacent ranks are at most 10 percentage points apart only when the
    # timestamp offers eleven or more comparable routed candidates.
    near_tie = same_timestamp & (count >= 11)
    left_outcome = pd.to_numeric(source["policy_net_bps"], errors="coerce").to_numpy(float)
    right_outcome = pd.to_numeric(next_source["policy_net_bps"], errors="coerce").to_numpy(float)
    gap = left_outcome - right_outcome
    finite = np.isfinite(gap)
    if mode == "near_tie":
        keep = near_tie & finite & (np.abs(gap) > 1e-8)
    elif mode == "near_tie_diff50":
        keep = near_tie & finite & (np.abs(gap) > 50.0)
    elif mode == "near_tie_diff100":
        keep = near_tie & finite & (np.abs(gap) > 100.0)
    else:  # higher-base candidate is economically wrong by more than 100 bps
        keep = near_tie & finite & (gap < -100.0)
    if not keep.any():
        raise ValueError(f"{spec.name} lacks {mode} pair support")
    left = source.loc[keep].copy()
    right = next_source.loc[keep, left.columns].copy()
    if mode == "base_inversion100":
        positive, negative = right, left
    else:
        left_wins = gap[keep] > 0.0
        positive = pd.concat([left.loc[left_wins], right.loc[~left_wins]], ignore_index=True)
        negative = pd.concat([right.loc[left_wins], left.loc[~left_wins]], ignore_index=True)
    pair_id = pd.Series(np.arange(len(positive), dtype=np.int64), index=positive.index).astype(str)
    positive["__query__"] = "pair|" + pair_id.to_numpy(dtype=object)
    negative["__query__"] = "pair|" + pair_id.to_numpy(dtype=object)
    positive["__target_grade__"] = 1
    negative["__target_grade__"] = 0
    work = pd.concat([positive, negative], ignore_index=True)
    work = work.sort_values(["__query__", "__target_grade__", "candidate_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    groups = work.groupby("__query__", sort=False).size().to_numpy(np.int32)
    if not np.all(groups == 2):
        raise AssertionError("pairwise correction split a pair query")
    return work, work.pop("__target_grade__").to_numpy(np.int32), groups


def _fit_consensus_head_from_sample(
    sampled: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    spec: ConsensusHeadSpec,
    *,
    seed: int,
    n_jobs: int = 1,
) -> FittedConsensusHead:
    if len(sampled) < 20 or np.unique(target).size < 2:
        raise ValueError(f"{spec.name} lacks query/class support")
    medians = _fit_medians(sampled, spec.fields)
    weights = None
    if spec.weight_mode == "equal_month":
        frequency = sampled["__month__"].value_counts()
        # Pandas 3 may expose an immutable Arrow-backed view here.  Weight
        # normalisation is intentionally in-place below, so take an explicit
        # mutable NumPy copy.  This changes no weighting value or chronology.
        weights = sampled["__month__"].map(lambda month: 1.0 / float(frequency.loc[month])).to_numpy(float).copy()
        weights *= len(weights) / max(float(weights.sum()), 1e-12)
    elif spec.weight_mode != "ordinary":
        raise ValueError(f"unknown frozen consensus weighting: {spec.weight_mode}")
    params = dict(spec.params); params.update(random_state=int(seed), n_jobs=int(n_jobs), deterministic=True, force_col_wise=True)
    model = LGBMRanker(**params).fit(_numeric_matrix(sampled, spec.fields, medians), target, group=groups, sample_weight=weights)
    raw = model.predict(_numeric_matrix(sampled, spec.fields, medians))
    return FittedConsensusHead(spec, medians, model, ScoreReference.fit(raw, source=f"{spec.name}_prequential_training_distribution"))


def _fit_regression_head_from_sample(
    sampled: pd.DataFrame,
    target: np.ndarray,
    spec: ConsensusHeadSpec,
    *,
    objective: str,
    seed: int,
    n_jobs: int = 1,
) -> FittedConsensusHead:
    """Fit one bounded economic-residual head on the frozen query sample.

    R2 and R4 deliberately keep the five-head field/query/sample contract.
    Only their loss changes: they estimate a common-bps residual rather than
    an ordinal within-query grade.  The subsequent stage decides whether that
    residual deserves direct common-bps authority.
    """
    if objective not in {"l2_regression", "huber_regression"}:
        raise ValueError(f"unsupported regression objective: {objective}")
    target = np.asarray(target, dtype=np.float32)
    if len(sampled) < 20 or np.nanstd(target) <= 1e-8:
        raise ValueError(f"{spec.name} lacks continuous target support")
    medians = _fit_medians(sampled, spec.fields)
    weights = None
    if spec.weight_mode == "equal_month":
        frequency = sampled["__month__"].value_counts()
        weights = sampled["__month__"].map(lambda month: 1.0 / float(frequency.loc[month])).to_numpy(float).copy()
        weights *= len(weights) / max(float(weights.sum()), 1e-12)
    elif spec.weight_mode != "ordinary":
        raise ValueError(f"unknown frozen consensus weighting: {spec.weight_mode}")
    ignored = {"objective", "metric", "label_gain", "lambdarank_truncation_level"}
    params = {key: value for key, value in spec.params.items() if key not in ignored}
    params.update(
        objective="huber" if objective == "huber_regression" else "regression_l2",
        metric="huber" if objective == "huber_regression" else "l2",
        random_state=int(seed), n_jobs=int(n_jobs), deterministic=True, force_col_wise=True,
    )
    model = LGBMRegressor(**params).fit(
        _numeric_matrix(sampled, spec.fields, medians), target, sample_weight=weights,
    )
    raw = model.predict(_numeric_matrix(sampled, spec.fields, medians))
    return FittedConsensusHead(
        spec, medians, model,
        ScoreReference.fit(raw, source=f"{spec.name}_{objective}_prequential_training_distribution"),
    )


@dataclass(frozen=True)
class Paths:
    raw_ledger: Path
    direct_root: Path
    policy_root: Path
    current_mc1: Path
    bcf_mc1: Path
    bundle_root: Path


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_dir():
            children = sorted(path.rglob("*.parquet"))
        else:
            children = [path]
        for child in children:
            digest.update(str(child).encode())
            with child.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _rank_pct(frame: pd.DataFrame, field: str) -> pd.Series:
    return frame.groupby("__decision_ts__", sort=False)[field].rank(pct=True, method="average")


def _rank_desc(frame: pd.DataFrame, field: str) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__pos__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", field, "candidate_id"], ascending=[True, False, True], kind="stable")
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["__rank__"] = 1.0 - (rank - 0.5) / size
    return work.sort_values("__pos__", kind="stable")["__rank__"].to_numpy(float)


def _exact_timestamp_top_fraction(frame: pd.DataFrame, field: str, fraction: float) -> pd.Series:
    """Return an exact, deterministic timestamp-local top-fraction gate.

    Percentile cutoffs are convenient score features, but ``rank(pct=True) >=
    .70`` is not itself an exact top-30% route: depending on cross-sectional
    size and ties it can retain one extra candidate.  The conversion layer is
    explicitly capacity-constrained to the canonical base model's top 30%, so
    routing must use a stable ordinal position instead.  Candidate ID is the
    declared deterministic tie-breaker and no outcome-dependent input is read.
    """
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    required = {"__decision_ts__", "candidate_id", field}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"exact timestamp route lacks fields: {sorted(missing)}")
    work = frame.loc[:, ["__decision_ts__", "candidate_id", field]].copy()
    work["__pos__"] = np.arange(len(work), dtype=np.int64)
    # Non-finite scores cannot be a valid base-route candidate.  Sorting them
    # last avoids any accidental route admission while preserving determinism.
    work["__score__"] = pd.to_numeric(work[field], errors="coerce").fillna(-np.inf)
    work = work.sort_values(
        ["__decision_ts__", "__score__", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    work["__ordinal__"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    cutoff = np.ceil(float(fraction) * size.to_numpy(float)).astype(np.int64)
    work["__selected__"] = (
        work["__ordinal__"].to_numpy(np.int64) <= cutoff
    ) & np.isfinite(work["__score__"].to_numpy(float))
    return work.sort_values("__pos__", kind="stable")["__selected__"].astype(bool)


def _direct_components(root: Path) -> pd.DataFrame:
    """Read only target-free component predictions from the selected direct arm."""
    wanted = "S3_direct_efficiency_time_base_equal"
    pieces: list[pd.DataFrame] = []
    cols = ["candidate_id", "__decision_ts__", "arm", "base_bps", "efficiency_bps", "timing_bps"]
    for fold in sorted((root / "oof_prediction_parts").glob("fold=*")):
        parts: list[pd.DataFrame] = []
        for path in sorted(fold.glob("*.parquet")):
            probe = pd.read_parquet(path, columns=["arm"])
            if len(probe) and str(probe["arm"].iloc[0]) == wanted:
                parts.append(pd.read_parquet(path, columns=cols))
                break
        if not parts:
            raise FileNotFoundError(f"{fold}: direct selected arm is absent")
        pieces.extend(parts)
    result = pd.concat(pieces, ignore_index=True).drop(columns="arm")
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result["candidate_id"].duplicated().any():
        raise AssertionError("selected direct source has duplicate candidate IDs")
    result["enhanced_base_bps"] = _three_way_enhanced_base_bps(result)
    result["base_rank_ts"] = _rank_pct(result, "enhanced_base_bps")
    result["enhanced_base_routed"] = _exact_timestamp_top_fraction(
        result, "enhanced_base_bps", BASE_ROUTE,
    )
    result["e_minus_t"] = result["efficiency_bps"] - result["timing_bps"]
    result["e_minus_b0"] = result["efficiency_bps"] - result["base_bps"]
    result["t_minus_b0"] = result["timing_bps"] - result["base_bps"]
    matrix = result.loc[:, ["base_bps", "efficiency_bps", "timing_bps"]].to_numpy(float)
    result["base_component_std"] = np.nanstd(matrix, axis=1)
    return result


def _three_way_enhanced_base_bps(frame: pd.DataFrame) -> np.ndarray:
    """Return the sealed equal B0/efficiency/timing common-bps blend.

    The selected direct arm is explicitly `S3_direct_efficiency_time_base_equal`.
    A two-way efficiency/timing reconstruction is a different upstream model
    and must never be used to train or assess its conversion layer.
    """
    values = frame.loc[:, ["base_bps", "efficiency_bps", "timing_bps"]].apply(
        pd.to_numeric, errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("three-way enhanced base contains a non-finite direct component")
    return values.mean(axis=1)


def _base_fields(paths: Paths) -> tuple[str, ...]:
    """Load the sealed base-field contract without trusting block ordering.

    Research artifacts can contain an interrupted historical booster.  The
    feature contract is identical across valid bundles, but choosing the first
    path lexicographically made an unrelated truncated January artifact block
    every downstream read.  Accept only a readable bundle whose ordered field
    list matches the sealed contract hash; never infer fields from a score
    panel or silently accept a different upstream schema.
    """
    import hashlib
    import joblib
    expected_hash = "b2c2725813d30c02ee298f82292d848d0e1133eb01be3f1398003163523ec2a1"
    bundles = sorted(paths.bundle_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib"))
    for bundle in bundles:
        try:
            fields = tuple(joblib.load(bundle).base_fields)
        except (EOFError, OSError, ValueError):
            continue
        digest = hashlib.sha256("\n".join(fields).encode()).hexdigest()
        if len(fields) == 120 and len(fields) == len(set(fields)) and digest == expected_hash:
            return fields
        raise AssertionError(f"unexpected frozen base contract in {bundle}: {digest}")
    raise AssertionError("no readable upstream bundle with the sealed 120-field contract")


def _materialize_target_free(paths: Paths, out: Path, fields: tuple[str, ...]) -> tuple[Path, pd.DataFrame]:
    root = out / "target_free_monthly"
    root.mkdir(parents=True, exist_ok=False)
    direct = _direct_components(paths.direct_root)
    start, end = direct["__decision_ts__"].min(), direct["__decision_ts__"].max() + pd.Timedelta(hours=1)
    raw_cols = ["candidate_id", "__decision_ts__", "side_name", *fields]
    raw = pd.read_parquet(
        paths.raw_ledger,
        columns=raw_cols,
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    frame = direct.merge(raw, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
    if len(frame) != len(direct):
        raise AssertionError(f"target-free raw join changed identities: direct={len(direct)}, joined={len(frame)}")
    if not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("enhanced base source is not long-only")
    frame["month"] = frame["__decision_ts__"].dt.strftime("%Y-%m")
    coverage = []
    for month, part in frame.groupby("month", sort=True):
        coverage.append({
            "month": month, "rows": int(len(part)),
            "feature_complete_fraction": float(part.loc[:, list(fields)].notna().all(axis=1).mean()),
            "routed_rows": int(part["enhanced_base_routed"].sum()),
        })
        target = root / f"month={month}"
        target.mkdir()
        part.drop(columns="month").to_parquet(target / "scores_features.parquet", index=False, compression="zstd")
    return root, pd.DataFrame(coverage)


def _reuse_target_free(root: Path, fields: tuple[str, ...]) -> tuple[Path, pd.DataFrame]:
    """Validate a prior target-free materialisation for label-only research.

    Reusing this immutable source makes the target comparison much cheaper
    while preserving candidate identities and every decision-time feature.
    It is intentionally rejected if an outcome field has slipped into a
    source panel, or if the frozen base coverage is below the normal gate.
    """

    root = root.resolve()
    prohibited = {
        "policy_path_valid", "policy_net_bps", "policy_gross_bps",
        "policy_label_available_ts", "policy_exit_bar_15m", "policy_entry_price",
        "policy_exit_price", "policy_exit_reason", "policy_cost_bps",
    }
    coverage: list[dict[str, object]] = []
    for month in SCORE_MONTHS:
        path = root / f"month={month:%Y-%m}" / "scores_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"shared target-free source lacks {path}")
        probe = pd.read_parquet(path)
        leaked = sorted(prohibited.intersection(probe.columns))
        if leaked:
            raise AssertionError(f"shared target-free source contains policy outcomes: {leaked}")
        missing = sorted(set(fields) - set(probe.columns))
        if missing:
            raise AssertionError(f"shared target-free source lacks frozen base fields: {missing[:3]}")
        fraction = float(probe.loc[:, list(fields)].notna().all(axis=1).mean())
        coverage.append({
            "month": f"{month:%Y-%m}", "rows": int(len(probe)),
            "feature_complete_fraction": fraction,
            "routed_rows": int(probe["enhanced_base_routed"].fillna(False).sum()),
            "source": "reused_immutable_target_free",
        })
    result = pd.DataFrame(coverage)
    if result["feature_complete_fraction"].lt(.90).any():
        raise AssertionError("shared target-free feature coverage below 90%")
    return root, result


def _load_months(root: Path, start: pd.Timestamp, end: pd.Timestamp, columns: Sequence[str]) -> pd.DataFrame:
    months = pd.period_range(start.to_period("M"), (end - pd.Timedelta(nanoseconds=1)).to_period("M"), freq="M")
    pieces: list[pd.DataFrame] = []
    for month in months:
        path = root / f"month={month.strftime('%Y-%m')}" / "scores_features.parquet"
        if not path.exists():
            continue
        piece = pd.read_parquet(path, columns=list(columns))
        piece["__decision_ts__"] = pd.to_datetime(piece["__decision_ts__"], utc=True, errors="raise")
        pieces.append(piece.loc[piece["__decision_ts__"].ge(start) & piece["__decision_ts__"].lt(end)])
    if not pieces:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(pieces, ignore_index=True)


def _load_policy(paths: Paths) -> pd.DataFrame:
    fields = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    policy = pd.read_parquet(paths.policy_root, columns=fields)
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy labels have duplicate candidate IDs")
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    return policy


def _deterministic_query_sample(frame: pd.DataFrame, *, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    work = frame.copy()
    work["__month__"] = work["__decision_ts__"].dt.to_period("M").astype(str)
    work["__q__"] = work["__decision_ts__"].dt.floor("4h").astype(str) + "|long"
    meta = work.groupby(["__month__", "__q__"], sort=True).size().rename("rows").reset_index()
    # ``__q__`` is deliberately a private-looking column so it never leaks into
    # a feature contract.  Pandas does not, however, expose such names as stable
    # attributes on the namedtuples returned by ``itertuples``.  Use item/tuple
    # access throughout: large routed populations are the ones that exercise
    # this capped-sampling branch.
    meta["h"] = [
        int(hashlib.sha256(f"{seed}|{month}|{query}".encode()).hexdigest()[:16], 16)
        for month, query in meta[["__month__", "__q__"]].itertuples(index=False, name=None)
    ]
    limit = max(2, cap // max(1, meta["__month__"].nunique()))
    keep: list[str] = []
    for _, part in meta.groupby("__month__", sort=True):
        used = 0
        for query, rows, _ in part.sort_values(["h", "__q__"], kind="stable")[["__q__", "rows", "h"]].itertuples(index=False, name=None):
            if used + int(rows) <= limit:
                keep.append(str(query)); used += int(rows)
    return work.loc[work["__q__"].isin(keep)].drop(columns=["__month__", "__q__"]).copy()


def _policy_map(train: pd.DataFrame) -> IsotonicRegression:
    fit = _deterministic_query_sample(train, cap=HEAD_TRAIN_CAP, seed=SEED + 91)
    x = pd.to_numeric(fit["base_rank_ts"], errors="coerce")
    y = pd.to_numeric(fit["policy_net_bps"], errors="coerce")
    valid = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    if valid.sum() < 1_000:
        raise ValueError("enhanced base policy map lacks support")
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x.loc[valid], y.loc[valid])


@dataclass(frozen=True)
class ScoreOodTransform:
    fields: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray


def _fit_score_ood_transform(train: pd.DataFrame) -> ScoreOodTransform:
    """Fit a target-free score-space support/OOD reference on train rows.

    This is intentionally a support proxy in stable base/E/T score geometry,
    not leaf support or K9 membership.  Its fit uses no policy outcome and the
    held rows only receive the frozen transform.
    """

    values = train.loc[:, list(META_SCORE_FIELDS)].apply(pd.to_numeric, errors="coerce")
    median = values.median().fillna(0.0).to_numpy(dtype=np.float32)
    deviation = values.sub(pd.Series(median, index=META_SCORE_FIELDS)).abs().median()
    scale = (1.4826 * deviation).replace(0.0, np.nan).fillna(values.std().replace(0.0, np.nan)).fillna(1.0)
    return ScoreOodTransform(META_SCORE_FIELDS, median, scale.to_numpy(dtype=np.float32))


def _apply_score_ood_transform(frame: pd.DataFrame, transform: ScoreOodTransform) -> pd.DataFrame:
    out = frame.copy()
    values = out.loc[:, list(transform.fields)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    z = np.abs((values - transform.median[None, :]) / transform.scale[None, :])
    z = np.nan_to_num(z, nan=10.0, posinf=10.0, neginf=10.0)
    z = np.clip(z, 0.0, 10.0)
    l1 = z.mean(axis=1)
    out["meta_score_ood_l1"] = l1.astype(np.float32)
    out["meta_score_ood_max"] = z.max(axis=1).astype(np.float32)
    # Smooth proximity to the training score geometry.  The explicit proxy
    # name prevents treating it as a model-leaf support count.
    out["meta_score_support_proxy"] = np.exp(-l1).astype(np.float32)
    return out


def _causal_recent_calibration_features(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return causal global calibration-state features for each candidate.

    This deliberately summarises resolved *operating* outcomes by their
    availability timestamp, rather than by their decision timestamp.  A row
    decided at ``t`` receives only event buckets strictly before ``t``.  Thus
    modifying an unresolved or future outcome cannot change an earlier
    candidate's feature vector.  The state is global across the already routed
    long population: it describes whether the base/efficiency/timing mapping
    has recently transported, not whether the particular candidate is good.
    """

    required = {
        "candidate_id", "__decision_ts__", "enhanced_base_routed",
        "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
        "base_anchor_bps", "base_rank_ts", "efficiency_bps", "timing_bps",
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"recent calibration history lacks fields: {missing}")
    if history["candidate_id"].duplicated().any():
        raise AssertionError("recent calibration history has duplicate candidate IDs")

    work = history.copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work["policy_label_available_ts"] = pd.to_datetime(
        work["policy_label_available_ts"], utc=True, errors="coerce",
    )
    valid = (
        work["enhanced_base_routed"].fillna(False).astype(bool)
        & work["policy_path_valid"].fillna(False).astype(bool)
        & work["policy_label_available_ts"].notna()
        & np.isfinite(pd.to_numeric(work["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(work["base_anchor_bps"], errors="coerce"))
    )
    events = work.loc[valid, [
        "policy_label_available_ts", "policy_net_bps", "base_anchor_bps",
        "base_rank_ts", "efficiency_bps", "timing_bps",
    ]].copy()
    events["__available_hour__"] = events["policy_label_available_ts"].dt.floor("h")
    policy = pd.to_numeric(events["policy_net_bps"], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(events["base_anchor_bps"], errors="coerce").to_numpy(float)
    rank = pd.to_numeric(events["base_rank_ts"], errors="coerce").to_numpy(float)
    efficiency = pd.to_numeric(events["efficiency_bps"], errors="coerce").to_numpy(float)
    timing = pd.to_numeric(events["timing_bps"], errors="coerce").to_numpy(float)
    residual = policy - anchor
    events = pd.DataFrame({
        "__available_hour__": events["__available_hour__"].to_numpy(),
        "count": np.ones(len(events), dtype=np.float64),
        "residual_sum": residual,
        "residual_sq_sum": residual * residual,
        "efficiency_residual_sum": policy - efficiency,
        "timing_residual_sum": policy - timing,
        "rank_sum": rank,
        "rank_sq_sum": rank * rank,
        "policy_sum": policy,
        "policy_sq_sum": policy * policy,
        "rank_policy_sum": rank * policy,
    })
    # Drop non-finite score components only from statistics that require
    # them.  The policy/base-residual state remains usable whenever its own
    # declared inputs are finite.
    for field in (
        "efficiency_residual_sum", "timing_residual_sum", "rank_sum",
        "rank_sq_sum", "policy_sum", "policy_sq_sum", "rank_policy_sum",
    ):
        events[field] = pd.to_numeric(events[field], errors="coerce").fillna(0.0)
    if work.empty:
        raise ValueError("recent calibration history is empty")
    index = pd.date_range(
        work["__decision_ts__"].min().floor("h"),
        work["__decision_ts__"].max().floor("h"),
        freq="h", tz="UTC",
    )
    grouped = events.groupby("__available_hour__", sort=True).sum(numeric_only=True)
    hourly = grouped.reindex(index, fill_value=0.0).astype(float)

    def _past_sum(field: str, hours: int) -> pd.Series:
        # Shift one complete bucket so values becoming available exactly at a
        # decision hour do not influence that hour's prediction.
        return hourly[field].shift(1, fill_value=0.0).rolling(hours, min_periods=1).sum()

    count_3 = _past_sum("count", 3 * 24)
    count_7 = _past_sum("count", 7 * 24)
    count_14 = _past_sum("count", 14 * 24)
    residual_3 = _past_sum("residual_sum", 3 * 24) / count_3.replace(0.0, np.nan)
    residual_7 = _past_sum("residual_sum", 7 * 24) / count_7.replace(0.0, np.nan)
    residual_14 = _past_sum("residual_sum", 14 * 24) / count_14.replace(0.0, np.nan)
    residual_sq_7 = _past_sum("residual_sq_sum", 7 * 24) / count_7.replace(0.0, np.nan)
    residual_std_7 = np.sqrt(np.maximum(residual_sq_7 - residual_7 * residual_7, 0.0))
    rank_sum_7 = _past_sum("rank_sum", 7 * 24)
    policy_sum_7 = _past_sum("policy_sum", 7 * 24)
    rank_sq_7 = _past_sum("rank_sq_sum", 7 * 24)
    policy_sq_7 = _past_sum("policy_sq_sum", 7 * 24)
    rank_policy_7 = _past_sum("rank_policy_sum", 7 * 24)
    rank_mean_7 = rank_sum_7 / count_7.replace(0.0, np.nan)
    policy_mean_7 = policy_sum_7 / count_7.replace(0.0, np.nan)
    covariance = rank_policy_7 / count_7.replace(0.0, np.nan) - rank_mean_7 * policy_mean_7
    rank_var = rank_sq_7 / count_7.replace(0.0, np.nan) - rank_mean_7 * rank_mean_7
    policy_var = policy_sq_7 / count_7.replace(0.0, np.nan) - policy_mean_7 * policy_mean_7
    rank_policy_corr = covariance / np.sqrt(np.maximum(rank_var * policy_var, 0.0))
    features_by_hour = pd.DataFrame({
        "meta_recent_base_residual_mean_3d": residual_3,
        "meta_recent_base_residual_mean_7d": residual_7,
        "meta_recent_base_residual_mean_14d": residual_14,
        "meta_recent_base_residual_std_7d": residual_std_7,
        "meta_recent_base_residual_slope_3d_14d": residual_3 - residual_14,
        "meta_recent_base_rank_policy_corr_7d": rank_policy_corr,
        "meta_recent_efficiency_residual_mean_7d": (
            _past_sum("efficiency_residual_sum", 7 * 24) / count_7.replace(0.0, np.nan)
        ),
        "meta_recent_timing_residual_mean_7d": (
            _past_sum("timing_residual_sum", 7 * 24) / count_7.replace(0.0, np.nan)
        ),
        "meta_recent_calibration_support_log1p_7d": np.log1p(count_7),
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    decision_hours = work["__decision_ts__"].dt.floor("h")
    result = work.loc[:, ["candidate_id"]].copy()
    for field in META_RECENT_CALIBRATION_FIELDS:
        result[field] = features_by_hour.loc[decision_hours.to_numpy(), field].to_numpy(dtype=np.float32)
    audit = {
        "history_rows": int(len(work)),
        "resolved_rows": int(valid.sum()),
        "source": "global routed policy outcomes with policy_label_available_ts strictly before decision",
        "fields": list(META_RECENT_CALIBRATION_FIELDS),
        "max_window_days": 14,
    }
    return result, audit


def _meta_feature_fields(
    contract: str,
    base_fields: tuple[str, ...],
    selected_fields: tuple[str, ...],
) -> tuple[str, ...]:
    if contract not in META_FEATURE_CONTRACTS:
        raise ValueError(f"unknown meta feature contract: {contract}")
    if contract == "current":
        fields = (*selected_fields, *META_SCORE_FIELDS)
    elif contract == "raw_heavy":
        fields = (*base_fields, *META_SCORE_FIELDS)
    elif contract == "geometry_only":
        fields = META_SCORE_FIELDS
    elif contract == "geometry_score_ood":
        fields = (*META_SCORE_FIELDS, *META_SCORE_OOD_FIELDS)
    elif contract == "geometry_recent_calibration":
        fields = (*META_SCORE_FIELDS, *META_RECENT_CALIBRATION_FIELDS)
    elif contract == "geometry_score_ood_state":
        fields = (*META_SCORE_FIELDS, *META_SCORE_OOD_FIELDS, *META_STATE_FIELDS)
    elif contract == "geometry_score_ood_state_raw":
        fields = (*META_SCORE_FIELDS, *META_SCORE_OOD_FIELDS, *META_STATE_FIELDS, *META_RAW_CONTEXT_FIELDS)
    else:  # geometry_score_ood_recent_calibration_state_raw
        fields = (
            *META_SCORE_FIELDS, *META_SCORE_OOD_FIELDS,
            *META_RECENT_CALIBRATION_FIELDS, *META_STATE_FIELDS,
            *META_RAW_CONTEXT_FIELDS,
        )
    fields = tuple(dict.fromkeys(fields))
    missing = sorted(
        set(fields) - set(base_fields) - set(META_SCORE_FIELDS)
        - set(META_SCORE_OOD_FIELDS) - set(META_RECENT_CALIBRATION_FIELDS)
    )
    if missing:
        raise AssertionError(f"meta feature contract lacks frozen source fields: {missing[:5]}")
    return fields


def _head_specs(
    base_fields: tuple[str, ...],
    feature_contract: str = "current",
) -> tuple[ConsensusHeadSpec, ...]:
    result: list[ConsensusHeadSpec] = []
    for spec in load_conditional_consensus_contract(base_fields, side="long"):
        fields = _meta_feature_fields(feature_contract, base_fields, spec.fields)
        result.append(ConsensusHeadSpec(
            name=spec.name, cap=spec.cap, weight_mode=spec.weight_mode,
            query=spec.query, fields=fields, target_edges_bps=spec.target_edges_bps,
            params=dict(spec.params),
        ))
    return tuple(result)


def _fit_heads(
    train: pd.DataFrame,
    target_value: np.ndarray,
    specs: Sequence[ConsensusHeadSpec],
    *,
    objective: str,
    grade: np.ndarray | None = None,
    pairwise_mode: str = "none",
    n_jobs: int = 1,
) -> tuple[tuple[object, ...], list[dict[str, int]]]:
    if pairwise_mode not in PAIRWISE_MODES:
        raise ValueError(f"unknown pairwise correction mode: {pairwise_mode}")
    if pairwise_mode != "none" and objective != "ordinal_lambdarank":
        raise ValueError("pairwise correction requires the ordinal LambdaRank head objective")
    identity_columns = ["candidate_id", "__decision_ts__", "side_name"]
    if any(spec.query == "exact_timestamp_baseband_side" for spec in specs):
        identity_columns.append("base_rank_ts")
    identity = train.loc[:, identity_columns].copy().reset_index(drop=True)
    by_id = train.set_index("candidate_id", drop=False)
    target_by_id = pd.Series(np.asarray(target_value, dtype=np.float32), index=train["candidate_id"].astype(str))
    sampling_grade = _residual_grade(target_value, (-100.0, -30.0, 30.0, 90.0)) if grade is None else grade
    heads = []
    pairwise_audit: list[dict[str, int]] = []
    for index, spec in enumerate(specs):
        if pairwise_mode == "none":
            sampled_identity, target, groups = _sample_complete_consensus_queries(
                identity, sampling_grade, spec, seed=SEED + 1000 + index,
            )
            sampled = by_id.loc[sampled_identity["candidate_id"].to_numpy(), ["candidate_id", "__decision_ts__", "side_name", *spec.fields]].copy()
            sampled["__query__"] = sampled_identity["__query__"].to_numpy()
            sampled["__month__"] = sampled_identity["__month__"].to_numpy()
        else:
            sampled, target, groups = _sample_base_near_tie_pairs(
                train, spec, pairwise_mode, seed=SEED + 1000 + index,
            )
            pairwise_audit.append({
                "head_index": int(index),
                "train_rows": int(len(sampled)),
                "pair_queries": int(len(groups)),
                "winner_rows": int(np.sum(np.asarray(target) > 0)),
            })
        if objective == "ordinal_lambdarank":
            heads.append(_fit_consensus_head_from_sample(
                sampled, target, groups, spec, seed=SEED + 1000 + index, n_jobs=n_jobs,
            ))
        else:
            continuous = target_by_id.loc[sampled["candidate_id"].astype(str)].to_numpy(dtype=np.float32)
            heads.append(_fit_regression_head_from_sample(
                sampled, continuous, spec, objective=objective, seed=SEED + 1000 + index,
                n_jobs=n_jobs,
            ))
    return tuple(heads), pairwise_audit


@dataclass
class CorrectnessModel:
    fields: tuple[str, ...]
    medians: np.ndarray
    model: LGBMRanker
    reference: ScoreReference
    floor: float


def _fit_correctness(train: pd.DataFrame, fields: Sequence[str]) -> CorrectnessModel:
    work = train.copy()
    valid = np.isfinite(work.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
    work = work.loc[valid].copy()
    floor = float(np.quantile(work["upstream"].to_numpy(float), 1.0 - CORRECTNESS_TRAIN_FRACTION, method="higher"))
    work = work.loc[work["upstream"].ge(floor)].copy()
    work["__query__"] = _query(work, "cycle_4h_side")
    count = work.groupby("__query__", sort=False)["candidate_id"].transform("size")
    work = work.loc[count.ge(2)].copy()
    work = _deterministic_query_sample(work, cap=CORRECTNESS_TRAIN_CAP, seed=SEED + 2001)
    work = work.sort_values(["__query__", "__decision_ts__", "candidate_id"], kind="stable")
    target = (work["policy_net_bps"] - work["base_anchor_bps"] > CORRECTNESS_HURDLE_BPS).astype(np.int8).to_numpy()
    if len(work) < 1_000 or np.unique(target).size < 2:
        raise ValueError("enhanced correctness model has insufficient class support")
    group = work.groupby("__query__", sort=False).size().to_numpy(np.int32)
    medians = _fit_medians(work, fields)
    model = LGBMRanker(
        objective="lambdarank", n_estimators=120, learning_rate=0.035,
        max_depth=4, num_leaves=15, min_child_samples=max(120, int(.03 * len(work))),
        colsample_bytree=.80, subsample=.82, subsample_freq=1,
        reg_alpha=.05, reg_lambda=5.0, max_bin=127, label_gain=[0, 1],
        lambdarank_truncation_level=10, random_state=SEED + 2003,
        n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
    ).fit(_numeric_matrix(work, fields, medians), target, group=group)
    raw = model.predict(_numeric_matrix(work, fields, medians))
    return CorrectnessModel(tuple(fields), medians, model, ScoreReference.fit(raw, source="enhanced_base_correctness_training"), floor)


@dataclass
class TailTrustModel:
    """Bounded adverse-tail trust model used only to demote high upstreams."""

    arm: str
    fields: tuple[str, ...]
    medians: np.ndarray
    model: object
    reference: ScoreReference
    floor: float


def _tail_trust_fields(heads: Sequence[object]) -> tuple[str, ...]:
    """Return a compact, causal trust-only feature contract.

    The trust model receives disagreement, score-space support/OOD, causal
    recent calibration and stable market state.  It deliberately excludes the
    broad raw-alpha contract so a tail veto cannot become a second base model.
    """

    return tuple(dict.fromkeys((
        *META_SCORE_FIELDS,
        "base_anchor_bps", "conditional_consensus_rank",
        "ordinary_shadow_consensus_rank", "upstream", "head_agreement_std",
        *META_SCORE_OOD_FIELDS, *META_RECENT_CALIBRATION_FIELDS,
        *META_STATE_FIELDS,
        *[f"head__{head.spec.name}__rank" for head in heads],
    )))


def _fit_tail_trust(
    train: pd.DataFrame,
    fields: Sequence[str],
    arm: str,
) -> TailTrustModel:
    """Fit a pre-resolved adverse-tail trust model with bounded capacity.

    Residuals are anchored to the enhanced-base common-bps estimate, not the
    legacy B0 map.  This makes the question explicit: *when is the stronger
    base materially overconfident?*  The returned score is always oriented so
    larger values mean more trustworthy predictions.
    """

    if arm not in {"severe_overconfidence", "lower_quantile", "severe_overconfidence_support"}:
        raise ValueError(f"unsupported tail-trust arm: {arm}")
    work = train.copy()
    valid = np.isfinite(work.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
    work = work.loc[valid].copy()
    floor = float(np.quantile(work["upstream"].to_numpy(float), 1.0 - CORRECTNESS_TRAIN_FRACTION, method="higher"))
    work = work.loc[work["upstream"].ge(floor)].copy()
    work = _deterministic_query_sample(work, cap=CORRECTNESS_TRAIN_CAP, seed=SEED + 2111)
    residual = (
        pd.to_numeric(work["policy_net_bps"], errors="coerce").to_numpy(float)
        - pd.to_numeric(work["enhanced_base_bps"], errors="coerce").to_numpy(float)
    )
    if len(work) < 1_000 or not np.isfinite(residual).all():
        raise ValueError("tail-trust model has insufficient resolved support")
    medians = _fit_medians(work, fields)
    matrix = _numeric_matrix(work, fields, medians)
    if arm in {"severe_overconfidence", "severe_overconfidence_support"}:
        target = (residual <= TAIL_TRUST_SEVERE_BPS).astype(np.int8)
        if np.unique(target).size < 2:
            raise ValueError("tail-trust severe label has one class")
        model = LGBMClassifier(
            objective="binary", n_estimators=120, learning_rate=0.035,
            max_depth=3, num_leaves=15, min_child_samples=max(180, int(.03 * len(work))),
            colsample_bytree=.80, subsample=.82, subsample_freq=1,
            reg_alpha=.10, reg_lambda=8.0, max_bin=127,
            random_state=SEED + 2113, n_jobs=1, deterministic=True,
            force_col_wise=True, verbosity=-1,
        ).fit(matrix, target)
        raw = -model.predict_proba(matrix)[:, 1]
        source = "enhanced_base_severe_overconfidence_training"
    else:
        model = LGBMRegressor(
            objective="quantile", alpha=TAIL_TRUST_QUANTILE,
            n_estimators=120, learning_rate=0.035,
            max_depth=3, num_leaves=15, min_child_samples=max(180, int(.03 * len(work))),
            colsample_bytree=.80, subsample=.82, subsample_freq=1,
            reg_alpha=.10, reg_lambda=8.0, max_bin=127,
            random_state=SEED + 2115, n_jobs=1, deterministic=True,
            force_col_wise=True, verbosity=-1,
        ).fit(matrix, np.clip(residual, -500.0, 500.0))
        raw = model.predict(matrix)
        source = "enhanced_base_q20_residual_training"
    return TailTrustModel(arm, tuple(fields), medians, model, ScoreReference.fit(raw, source=source), floor)


def _tail_trust_rank_and_risk(model: TailTrustModel, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return trust rank and [0, 1] demotion risk without admission authority."""

    matrix = _numeric_matrix(frame, model.fields, model.medians)
    if model.arm in {"severe_overconfidence", "severe_overconfidence_support"}:
        probability = model.model.predict_proba(matrix)[:, 1]
        trust_rank = model.reference.cdf(-probability)
        risk = np.clip(probability, 0.0, 1.0)
        if model.arm == "severe_overconfidence_support":
            support = pd.to_numeric(frame["meta_score_support_proxy"], errors="coerce").to_numpy(float)
            support = np.clip(np.nan_to_num(support, nan=0.0), 0.0, 1.0)
            # Low score-space support may strengthen an already predicted
            # adverse-tail warning; it cannot exceed the same 10% cap.
            risk *= 0.5 + 0.5 * (1.0 - support)
    else:
        q20 = model.model.predict(matrix)
        trust_rank = model.reference.cdf(q20)
        risk = np.clip(1.0 - trust_rank, 0.0, 1.0)
    return trust_rank.astype(np.float32), risk.astype(np.float32)


def _score_heads(frame: pd.DataFrame, heads: Sequence[object]) -> pd.DataFrame:
    out = frame.copy()
    ranks: dict[str, np.ndarray] = {}
    raws: dict[str, np.ndarray] = {}
    for head in heads:
        raw, rank = head.predict_rank(out)
        out[f"head__{head.spec.name}__raw"] = raw
        out[f"head__{head.spec.name}__rank"] = rank
        raws[head.spec.name] = raw
        ranks[head.spec.name] = rank
    matrix = np.column_stack(list(ranks.values()))
    out["conditional_consensus_rank"] = np.nanmedian(matrix, axis=1)
    out["conditional_consensus_raw"] = np.nanmedian(np.column_stack(list(raws.values())), axis=1)
    ordinary = [rank for name, rank in ranks.items() if "ordinary" in name]
    ordinary_raw = [raw for name, raw in raws.items() if "ordinary" in name]
    if not ordinary:
        # A selected-slot successor may intentionally retain exactly one
        # equal-month head.  In that case there is no distinct ordinary
        # shadow ensemble; alias the sole physical head so downstream receipt
        # schemas remain stable without manufacturing a second model.
        if len(ranks) != 1:
            raise AssertionError("multi-head consensus contract lacks ordinary heads")
        ordinary = list(ranks.values())
        ordinary_raw = list(raws.values())
    out["ordinary_shadow_consensus_rank"] = np.nanmedian(np.column_stack(ordinary), axis=1)
    out["ordinary_shadow_consensus_raw"] = np.nanmedian(np.column_stack(ordinary_raw), axis=1)
    stats = agreement_statistics(ranks)
    out["head_agreement_std"] = stats["mad"]
    out["head_agreement_rank"] = _rank_desc(out.assign(__agreement__=-out["head_agreement_std"]), "__agreement__")
    return out


def _fit_reserve_residual_map(reserve: pd.DataFrame, raw_field: str) -> tuple[IsotonicRegression, dict[str, object]]:
    """Calibrate a raw consensus coordinate to policy residual in common bps.

    The head bundle is fitted before the reserve.  The reserve is then scored
    once by that frozen bundle and has fully resolved policy outcomes before
    the held month begins.  This keeps the bps bridge out of both head fitting
    and held-period calibration.
    """

    raw = pd.to_numeric(reserve[raw_field], errors="coerce").to_numpy(float)
    policy = pd.to_numeric(reserve["policy_net_bps"], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(reserve["base_anchor_bps"], errors="coerce").to_numpy(float)
    target = np.clip(policy - anchor, -500.0, 500.0)
    valid = np.isfinite(raw) & np.isfinite(target)
    if int(valid.sum()) < 1_000 or np.nanstd(raw[valid]) <= 1e-8:
        raise ValueError(f"reserve residual map lacks support for {raw_field}")
    model = IsotonicRegression(
        increasing=True, out_of_bounds="clip", y_min=-500.0, y_max=500.0,
    ).fit(raw[valid], target[valid])
    return model, {
        "field": raw_field,
        "rows": int(valid.sum()),
        "target_mean_bps": float(np.mean(target[valid])),
        "target_std_bps": float(np.std(target[valid])),
        "raw_min": float(np.min(raw[valid])),
        "raw_max": float(np.max(raw[valid])),
    }


def _score_fold(
    root: Path,
    policy: pd.DataFrame,
    base_fields: tuple[str, ...],
    label_spec: PolicyConversionLabelSpec,
    score_architecture: str,
    pairwise_mode: str,
    integration_spec: BpsIntegrationSpec,
    feature_contract: str,
    month: pd.Timestamp,
    out: Path,
    trust_arm: str = "generic_correctness",
) -> tuple[dict[str, object], Path, Path]:
    if score_architecture not in SCORE_ARCHITECTURES:
        raise ValueError(f"unknown score architecture: {score_architecture}")
    if pairwise_mode not in PAIRWISE_MODES:
        raise ValueError(f"unknown pairwise correction mode: {pairwise_mode}")
    if integration_spec.name not in BPS_INTEGRATION_SPECS:
        raise ValueError(f"unknown bps integration mode: {integration_spec.name}")
    if feature_contract not in META_FEATURE_CONTRACTS:
        raise ValueError(f"unknown meta feature contract: {feature_contract}")
    if trust_arm not in TRUST_ARMS:
        raise ValueError(f"unknown trust arm: {trust_arm}")
    if score_architecture == "base_only" and pairwise_mode != "none":
        raise ValueError("base-only waterfall control has no pairwise correction")
    if score_architecture == "base_only" and trust_arm != "none":
        raise ValueError("base-only control cannot fit a trust layer")
    if score_architecture == "base_consensus_no_correctness" and trust_arm != "none":
        raise ValueError("no-correctness architecture requires trust_arm=none")
    end = _month_end(month)
    reserve_start = month - pd.Timedelta(days=RESERVE_DAYS)
    train_start = month - pd.DateOffset(months=META_TRAIN_MONTHS)
    bcf_reference_start = month - pd.Timedelta(days=BCF_REFERENCE_DAYS)
    basic = [
        "candidate_id", "__decision_ts__", "side_name", "enhanced_base_bps", "base_rank_ts", "enhanced_base_routed",
        "base_bps", "efficiency_bps", "timing_bps", "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std", *base_fields,
    ]
    train = _load_months(root, train_start, reserve_start, basic).merge(policy, on="candidate_id", how="left", validate="one_to_one")
    train = train.loc[
        train["enhanced_base_routed"].fillna(False).astype(bool)
        & train["policy_path_valid"].fillna(False).astype(bool)
        & train["policy_label_available_ts"].lt(reserve_start)
        & np.isfinite(pd.to_numeric(train["policy_net_bps"], errors="coerce"))
    ].copy()
    if len(train) < 5_000:
        raise ValueError(f"{month:%Y-%m}: insufficient train support")
    mapping = _policy_map(train)
    train["base_anchor_bps"] = mapping.predict(train["base_rank_ts"])
    # The calibration/drift state needs its own compact source rather than the
    # 120-field training matrix.  It is deliberately materialised through the
    # held-month end so the same causal transformation can be applied to
    # train, reserve, reference and held rows.  `_causal_recent_calibration_features`
    # gates every contribution on label availability, never on the row's
    # decision date alone.
    recent_columns = [
        "candidate_id", "__decision_ts__", "enhanced_base_routed", "base_rank_ts",
        "base_bps", "efficiency_bps", "timing_bps",
    ]
    recent_start = min(train_start, bcf_reference_start) - pd.Timedelta(days=14)
    recent_history = _load_months(root, recent_start, end, recent_columns).merge(
        policy.loc[:, [
            "candidate_id", "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
        ]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    recent_history["base_anchor_bps"] = mapping.predict(recent_history["base_rank_ts"])
    recent_features, recent_calibration_audit = _causal_recent_calibration_features(recent_history)
    del recent_history
    train = train.merge(recent_features, on="candidate_id", how="left", validate="one_to_one")
    if train.loc[:, list(META_RECENT_CALIBRATION_FIELDS)].isna().any().any():
        raise AssertionError(f"{month:%Y-%m}: recent calibration feature join is incomplete")
    score_ood_transform = _fit_score_ood_transform(train)
    train = _apply_score_ood_transform(train, score_ood_transform)
    # Base-only is a real control: it retains the same prequential policy map
    # and downstream MC1/portfolio protocol, but does not fit or consume a
    # consensus target.  The other two architectures use the same selected
    # five heads and differ only in the generic correctness demotion.
    if score_architecture == "base_only":
        grade = np.empty(0, dtype=np.int32)
        heads: tuple[object, ...] = ()
        pairwise_head_audit: list[dict[str, int]] = []
    else:
        target_value = label_spec.values(train)
        if not np.isfinite(target_value).all():
            raise AssertionError(f"{month:%Y-%m}: policy-conversion target is incomplete after validity filtering")
        grade = _residual_grade(target_value, label_spec.edges_bps)
        heads, pairwise_head_audit = _fit_heads(
            train, target_value, _head_specs(base_fields, feature_contract),
            objective=label_spec.objective, grade=grade, pairwise_mode=pairwise_mode,
        )
    reserve = _load_months(root, reserve_start, month, basic).merge(
        policy, on="candidate_id", how="left", validate="one_to_one",
    )
    reserve = reserve.loc[
        reserve["enhanced_base_routed"].fillna(False).astype(bool)
        & reserve["policy_path_valid"].fillna(False).astype(bool)
        & reserve["policy_label_available_ts"].lt(month)
        & np.isfinite(pd.to_numeric(reserve["policy_net_bps"], errors="coerce"))
    ].copy()
    if len(reserve) < 1_000:
        raise ValueError(f"{month:%Y-%m}: reserve lacks resolved policy support")
    reference = _load_months(root, bcf_reference_start, month, basic)
    held = _load_months(root, month, end, basic)
    for role in (reserve, reference, held):
        role["base_anchor_bps"] = mapping.predict(role["base_rank_ts"])
        role_features = recent_features.set_index("candidate_id").loc[
            role["candidate_id"].to_numpy(), list(META_RECENT_CALIBRATION_FIELDS),
        ].to_numpy(dtype=np.float32)
        role.loc[:, list(META_RECENT_CALIBRATION_FIELDS)] = role_features
    reserve = _apply_score_ood_transform(reserve, score_ood_transform)
    reference = _apply_score_ood_transform(reference, score_ood_transform)
    held = _apply_score_ood_transform(held, score_ood_transform)
    del recent_features
    combined = pd.concat([reference.assign(__role__="reference"), held.assign(__role__="held")], ignore_index=True)
    combined["base_rank42"] = ScoreReference.fit(reference["enhanced_base_bps"], source="same_bundle_prior28_enhanced_base").cdf(combined["enhanced_base_bps"])
    if heads:
        combined = _score_heads(combined, heads)
        combined["rank_upstream"] = BASE_BLEND_WEIGHT * combined["base_rank42"] + CONSENSUS_BLEND_WEIGHT * combined["conditional_consensus_rank"]
        if integration_spec.is_rank_control:
            combined["conditional_consensus_residual_bps"] = np.nan
            combined["ordinary_shadow_consensus_residual_bps"] = np.nan
            combined["corrected_current_bps"] = np.nan
            combined["corrected_bcf_bps"] = np.nan
            combined["upstream"] = combined["rank_upstream"].to_numpy(float)
            integration_audit: dict[str, object] = {
                "mode": integration_spec.name,
                "reserve_rows": 0,
                "maps": [],
                "authority": "75/25 rank blend control",
            }
        else:
            # The bps map is fitted only on prior reserve rows scored by the
            # already frozen head bundle.  It never sees held outcomes.
            reserve_scored = _score_heads(reserve, heads)
            conditional_map, conditional_audit = _fit_reserve_residual_map(
                reserve_scored, "conditional_consensus_raw",
            )
            ordinary_map, ordinary_audit = _fit_reserve_residual_map(
                reserve_scored, "ordinary_shadow_consensus_raw",
            )
            combined["conditional_consensus_residual_bps"] = conditional_map.predict(
                combined["conditional_consensus_raw"].to_numpy(float),
            )
            combined["ordinary_shadow_consensus_residual_bps"] = ordinary_map.predict(
                combined["ordinary_shadow_consensus_raw"].to_numpy(float),
            )
            combined["corrected_current_bps"] = (
                combined["base_anchor_bps"].to_numpy(float)
                + integration_spec.correction(combined["conditional_consensus_residual_bps"])
            )
            combined["corrected_bcf_bps"] = (
                combined["base_anchor_bps"].to_numpy(float)
                + integration_spec.correction(combined["ordinary_shadow_consensus_residual_bps"])
            )
            current_reference = combined.loc[
                combined["__role__"].eq("reference") & combined["__decision_ts__"].ge(reserve_start),
                "corrected_current_bps",
            ]
            combined["upstream"] = ScoreReference.fit(
                current_reference, source="same_bundle_prior28_reserve_calibrated_current_bps",
            ).cdf(combined["corrected_current_bps"])
            integration_audit = {
                "mode": integration_spec.name,
                "reserve_rows": int(len(reserve_scored)),
                "maps": [conditional_audit, ordinary_audit],
                "authority": integration_spec.description,
            }
    else:
        # Neutral placeholders retain the unchanged six-column MC1 shape.
        # They contain no outcome or pseudo-consensus signal.
        combined["conditional_consensus_rank"] = combined["base_rank42"].to_numpy(float)
        combined["ordinary_shadow_consensus_rank"] = combined["base_rank42"].to_numpy(float)
        combined["head_agreement_std"] = 0.0
        combined["head_agreement_rank"] = 0.5
        combined["rank_upstream"] = combined["base_rank42"].to_numpy(float)
        combined["conditional_consensus_residual_bps"] = np.nan
        combined["ordinary_shadow_consensus_residual_bps"] = np.nan
        combined["corrected_current_bps"] = np.nan
        combined["corrected_bcf_bps"] = np.nan
        combined["upstream"] = combined["base_rank42"].to_numpy(float)
        integration_audit = {
            "mode": integration_spec.name,
            "reserve_rows": 0,
            "maps": [],
            "authority": "not applicable; base-only control",
        }
    # Trust is a separately named authority layer.  It consumes only
    # pre-resolved outcomes, applies after the five-head correction, and may
    # never promote a candidate above its upstream score.  The historical
    # generic-correctness arm is retained as T1; T2--T4 are deliberately
    # capped at a 10% demotion so they are vetoes rather than alpha models.
    trust_feature_count = 0
    trust_floor = float("nan")
    trust_authority = "neutral"
    if score_architecture == "base_consensus_correctness" and trust_arm != "none":
        correctness_train = _deterministic_query_sample(
            train,
            cap=CORRECTNESS_TRAIN_CAP,
            seed=SEED + 2001,
        )
        train_scored = _score_heads(correctness_train, heads)
        train_scored["base_rank42"] = train_scored["base_rank_ts"]
        train_scored["upstream"] = (
            BASE_BLEND_WEIGHT * train_scored["base_rank42"]
            + CONSENSUS_BLEND_WEIGHT * train_scored["conditional_consensus_rank"]
        )
        if trust_arm == "generic_correctness":
            correctness_fields = tuple(dict.fromkeys((*base_fields,
                "enhanced_base_bps", "base_rank_ts", "base_anchor_bps", "conditional_consensus_rank", "ordinary_shadow_consensus_rank", "upstream",
                "base_bps", "efficiency_bps", "timing_bps", "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std", "head_agreement_std",
                *[f"head__{head.spec.name}__rank" for head in heads],
            )))
            correctness = _fit_correctness(train_scored, correctness_fields)
            raw = correctness.model.predict(_numeric_matrix(combined, correctness.fields, correctness.medians))
            combined["correctness_rank"] = correctness.reference.cdf(raw)
            active = combined["rank_upstream"].ge(correctness.floor).to_numpy(bool)
            demoted = combined["upstream"].to_numpy(float) * (
                CORRECTNESS_FLOOR + CORRECTNESS_SPAN * combined["correctness_rank"].to_numpy(float)
            )
            combined["raw_current_score"] = np.where(active, demoted, combined["upstream"].to_numpy(float))
            trust_feature_count = len(correctness_fields)
            trust_floor = float(correctness.floor)
            trust_authority = "historical generic correctness multiplier"
        else:
            tail_fields = _tail_trust_fields(heads)
            missing_tail = sorted(set(tail_fields) - set(train_scored.columns))
            if missing_tail:
                raise AssertionError(f"tail-trust fields missing from training frame: {missing_tail[:5]}")
            tail_trust = _fit_tail_trust(train_scored, tail_fields, trust_arm)
            trust_rank, risk = _tail_trust_rank_and_risk(tail_trust, combined)
            combined["correctness_rank"] = trust_rank
            active = combined["rank_upstream"].ge(tail_trust.floor).to_numpy(bool)
            demoted = combined["upstream"].to_numpy(float) * (1.0 - TAIL_TRUST_MAX_DEMOTION * risk)
            combined["raw_current_score"] = np.where(active, demoted, combined["upstream"].to_numpy(float))
            trust_feature_count = len(tail_fields)
            trust_floor = float(tail_trust.floor)
            trust_authority = f"bounded {TAIL_TRUST_MAX_DEMOTION:.0%} demotion; {trust_arm}"
    else:
        # T0/no-correctness has zero authority.  Retain a neutral coordinate
        # so the frozen six-input MC1 shape is unchanged when refitted.
        combined["correctness_rank"] = 0.5
        combined["raw_current_score"] = combined["upstream"].to_numpy(float)
    # BCF preserves the independent ordinary-consensus geometry and uses the
    # longer same-model reference horizon.  It is a distinct score family for
    # the second MC1 map, not a numerical reuse of the current score.
    if score_architecture == "base_only":
        combined["raw_bcf_score"] = combined["base_rank42"].to_numpy(float)
    else:
        combined["raw_bcf_score"] = (
            BASE_BLEND_WEIGHT * combined["base_rank42"] + CONSENSUS_BLEND_WEIGHT * combined["ordinary_shadow_consensus_rank"]
            if integration_spec.is_rank_control else combined["corrected_bcf_bps"].to_numpy(float)
        )
    reference_mask = combined["__role__"].eq("reference")
    current_reference_mask = reference_mask & combined["__decision_ts__"].ge(reserve_start)
    bcf_reference_mask = combined["__decision_ts__"].ge(month - pd.Timedelta(days=BCF_REFERENCE_DAYS)) & reference_mask
    combined["current_final_score"] = ScoreReference.fit(combined.loc[current_reference_mask, "raw_current_score"], source="same_bundle_prior28_enhanced_current").cdf(combined["raw_current_score"])
    combined["bcf_final_score"] = ScoreReference.fit(combined.loc[bcf_reference_mask, "raw_bcf_score"], source="same_bundle_prior42_enhanced_bcf").cdf(combined["raw_bcf_score"])
    held_out = combined.loc[combined["__role__"].eq("held")].copy()
    # Persist every frozen-head rank in the target-free OOS receipt.  These
    # fields are needed for a genuine per-head audit (own economics,
    # redundancy and leave-one-out contribution); retaining only the median
    # consensus made that audit impossible without refitting the folds.
    common = [
        "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed",
        "enhanced_base_bps", "base_rank42", "base_anchor_bps",
        "conditional_consensus_rank", "ordinary_shadow_consensus_rank", "rank_upstream", "upstream",
        "conditional_consensus_residual_bps", "ordinary_shadow_consensus_residual_bps",
        "corrected_current_bps", "corrected_bcf_bps",
        "correctness_rank", "head_agreement_rank", "head_agreement_std",
        *[f"head__{head.spec.name}__rank" for head in heads],
    ]
    current = held_out.loc[:, common].copy()
    current["final_score"] = held_out["current_final_score"].to_numpy(float)
    bcf = held_out.loc[:, common].copy()
    bcf["final_score"] = held_out["bcf_final_score"].to_numpy(float)
    # BCF uses head-agreement as the family-native correctness coordinate;
    # current keeps its retrained correctness model output.
    bcf["correctness_rank"] = (
        held_out["head_agreement_rank"].to_numpy(float)
        if heads else np.full(len(held_out), 0.5, dtype=float)
    )
    current = current.drop(columns="head_agreement_rank")
    bcf = bcf.drop(columns="head_agreement_rank")
    for family, frame in (("current", current), ("bcf", bcf)):
        forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "policy_gross_bps"}
        if forbidden.intersection(frame.columns):
            raise AssertionError("outcome field entered target-free score panel")
        path = out / "target_free_scores" / family / f"month={month:%Y-%m}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")
    audit = {
        "month": f"{month:%Y-%m}", "train_start": train_start.isoformat(), "reserve_start": reserve_start.isoformat(),
        "held_end_exclusive": end.isoformat(), "train_rows": int(len(train)), "reference_rows": int(len(reference)), "held_rows": int(len(held)),
        "held_routed_rows": int(held["enhanced_base_routed"].sum()), "head_count": len(heads),
        "score_architecture": score_architecture,
        "meta_feature_contract": feature_contract,
        "head_feature_counts": [len(head.spec.fields) for head in heads],
        "score_ood_reference_rows": int(len(train)),
        "recent_calibration": recent_calibration_audit,
        "pairwise_mode": pairwise_mode,
        "bps_integration": integration_spec.name,
        "bps_integration_authority": integration_spec.description,
        "bps_integration_reserve_rows": integration_audit["reserve_rows"],
        "bps_integration_maps": integration_audit["maps"],
        "pairwise_training_target": (
            "resolved policy_net_bps ordering within timestamp-local base-near-tie pairs"
            if pairwise_mode != "none" else None
        ),
        "pairwise_train_rows_per_head": [item["train_rows"] for item in pairwise_head_audit],
        "pairwise_pair_queries_per_head": [item["pair_queries"] for item in pairwise_head_audit],
        "policy_conversion_label": label_spec.name, "policy_conversion_source": label_spec.source,
        "policy_conversion_edges_bps": list(label_spec.edges_bps),
        "policy_conversion_objective": label_spec.objective,
        "policy_conversion_clip_abs_bps": label_spec.clip_abs_bps,
        # Parquet cannot represent an empty struct.  A0 has no consensus
        # supervision by design, so retain an explicit marker instead of an
        # empty grade-count map.
        "policy_conversion_grade_counts": (
            {str(index): int(value) for index, value in enumerate(np.bincount(grade, minlength=5))}
            if len(grade) else {"not_applicable": 0}
        ),
        "trust_arm": trust_arm,
        "trust_feature_count": trust_feature_count,
        "trust_floor": trust_floor,
        "trust_authority": trust_authority,
    }
    return audit, out / "target_free_scores" / "current" / f"month={month:%Y-%m}.parquet", out / "target_free_scores" / "bcf" / f"month={month:%Y-%m}.parquet"


def _read_score_panels(root: Path, family: str, policy: pd.DataFrame) -> pd.DataFrame:
    pieces = [pd.read_parquet(path) for path in sorted((root / "target_free_scores" / family).glob("*.parquet"))]
    frame = pd.concat(pieces, ignore_index=True)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{family} target-free score panels duplicate candidate IDs")
    return frame.merge(policy, on="candidate_id", how="left", validate="one_to_one")


def _rebuild_score_fit_audit(
    target_free_root: Path,
    policy: pd.DataFrame,
    label_spec: PolicyConversionLabelSpec,
    score_architecture: str,
    pairwise_mode: str,
    feature_contract: str = "current",
    trust_arm: str = "generic_correctness",
) -> pd.DataFrame:
    """Rebuild a fold receipt without re-fitting/re-scoring any model.

    A completed target-free score panel is immutable.  If a terminal mapper or
    portfolio process is interrupted, rebuilding its receipt must not require
    loading the 120-field matrix or rewriting the OOS scores.  This compact
    audit reads only identities, timestamps, routing and already-resolved
    policy validity to prove the original causal fold boundaries.
    """
    if score_architecture not in SCORE_ARCHITECTURES:
        raise ValueError(f"unknown score architecture: {score_architecture}")
    if pairwise_mode not in PAIRWISE_MODES:
        raise ValueError(f"unknown pairwise correction mode: {pairwise_mode}")
    if feature_contract not in META_FEATURE_CONTRACTS:
        raise ValueError(f"unknown meta feature contract: {feature_contract}")
    if trust_arm not in TRUST_ARMS:
        raise ValueError(f"unknown trust arm: {trust_arm}")
    compact = policy.loc[:, [
        "candidate_id", "policy_path_valid", "policy_net_bps",
        "policy_label_available_ts",
    ]]
    fields = ["candidate_id", "__decision_ts__", "enhanced_base_routed"]
    rows: list[dict[str, object]] = []
    head_count = 0 if score_architecture == "base_only" else EXPECTED_RESEARCH_HEADS
    for month in SCORE_MONTHS:
        end = _month_end(month)
        reserve_start = month - pd.Timedelta(days=RESERVE_DAYS)
        train_start = month - pd.DateOffset(months=META_TRAIN_MONTHS)
        reference_start = month - pd.Timedelta(days=BCF_REFERENCE_DAYS)
        train = _load_months(target_free_root, train_start, reserve_start, fields)
        train = train.merge(compact, on="candidate_id", how="left", validate="one_to_one")
        valid_train = (
            train["enhanced_base_routed"].fillna(False).astype(bool)
            & train["policy_path_valid"].fillna(False).astype(bool)
            & train["policy_label_available_ts"].lt(reserve_start)
            & np.isfinite(pd.to_numeric(train["policy_net_bps"], errors="coerce"))
        )
        held = _load_months(target_free_root, month, end, fields)
        reference = _load_months(target_free_root, reference_start, month, fields)
        rows.append({
            "month": f"{month:%Y-%m}",
            "train_start": train_start.isoformat(),
            "reserve_start": reserve_start.isoformat(),
            "held_end_exclusive": end.isoformat(),
            "train_rows": int(valid_train.sum()),
            "reference_rows": int(len(reference)),
            "held_rows": int(len(held)),
            "held_routed_rows": int(held["enhanced_base_routed"].fillna(False).sum()),
            "head_count": head_count,
            "score_architecture": score_architecture,
            "meta_feature_contract": feature_contract,
            "pairwise_mode": pairwise_mode,
            "policy_conversion_label": label_spec.name,
            "policy_conversion_source": label_spec.source,
            "policy_conversion_edges_bps": list(label_spec.edges_bps),
            "policy_conversion_objective": label_spec.objective,
            "policy_conversion_clip_abs_bps": label_spec.clip_abs_bps,
            "policy_conversion_grade_counts": {"not_applicable": 0},
            "trust_arm": trust_arm,
            "trust_feature_count": np.nan,
            "trust_floor": float("nan"),
            "trust_authority": "rebuilt receipt only; score panel already persisted",
        })
        del train, held, reference
        gc.collect()
    return pd.DataFrame(rows)


def _score_bands(frame: pd.DataFrame) -> np.ndarray:
    out = frame.loc[:, ["candidate_id", "__decision_ts__", "final_score"]].copy()
    out["__pos__"] = np.arange(len(out), dtype=np.int64)
    out = out.sort_values(["__decision_ts__", "final_score", "candidate_id"], ascending=[True, False, True], kind="stable")
    rank = out.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    count = out.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    out["band"] = np.minimum(9, (10.0 * (rank + .5) / count).astype(np.int8))
    return out.sort_values("__pos__", kind="stable")["band"].to_numpy(np.int8)


def _robust_mean(values: Sequence[float], trim: float = .10) -> float:
    x = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    if not len(x):
        return float("nan")
    k = int(math.floor(len(x) * trim))
    x = x[k:len(x)-k] if k and len(x) > 2*k else x
    return float(x.mean())


def _fit_mc1(train: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingRegressor
    fit = train.copy()
    fit["score_band"] = _score_bands(fit)
    fit["day"] = fit["__decision_ts__"].dt.normalize()
    # same day-balanced cap style as the canonical mapper
    selected = []
    for _, group in fit.groupby("day", sort=True):
        group = group.sort_values(["__decision_ts__", "final_score", "candidate_id"], ascending=[True, False, True], kind="stable")
        selected.append(pd.concat([group.head(50), group.iloc[50:].sample(min(250, max(0, len(group)-50)), random_state=SEED)]))
    work = pd.concat(selected, ignore_index=True)
    y = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    low, high = y.quantile([.02, .98])
    work["target"] = y.clip(low, high)
    if len(work) > 50_000:
        work = work.sample(50_000, random_state=SEED)
    medians = work.loc[:, list(MC1_FEATURES)].apply(pd.to_numeric, errors="coerce").median().to_numpy(float)
    x = work.loc[:, list(MC1_FEATURES)].apply(pd.to_numeric, errors="coerce").fillna(pd.Series(medians, index=MC1_FEATURES))
    model = HistGradientBoostingRegressor(max_depth=2, max_iter=80, learning_rate=.04, l2_regularization=20.0, min_samples_leaf=100, random_state=SEED).fit(x, work["target"])
    # score-band structural curve, with monotonicity as in the production MC1 mapper
    global_mean = _robust_mean(work["target"])
    curve = np.full(10, global_mean, dtype=float)
    for band, group in work.groupby("score_band", sort=True):
        mean, sd, n = float(group["target"].mean()), max(float(group["target"].std(ddof=0)), 1.0), len(group)
        precision = n / (sd*sd + 1.0); prior = 80.0/(250.0**2)
        curve[int(band)] = (precision*mean + prior*global_mean)/(precision+prior)
    curve = -IsotonicRegression(increasing=True).fit_transform(np.arange(10), -curve)
    return model, medians, np.asarray(curve, dtype=float), (float(low), float(high))


def _mc1_predictions(frame: pd.DataFrame, family: str, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["score_band"] = _score_bands(frame)
    outputs: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for month in SCORE_MONTHS:
        end = _month_end(month)
        if month < pd.Timestamp("2025-10-01T00:00:00Z"):
            continue  # strict enhanced score warm-up before an MC1 map is fitted
        train_start = month - pd.DateOffset(months=MC1_TRAIN_MONTHS)
        fit = frame.loc[
            frame["__decision_ts__"].ge(train_start) & frame["__decision_ts__"].lt(month)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & frame["policy_label_available_ts"].lt(month)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        ].copy()
        held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(end)].copy()
        if len(fit) < 5_000 or held.empty:
            audit.append({"family": family, "month": f"{month:%Y-%m}", "status": "insufficient", "train_rows": int(len(fit)), "held_rows": int(len(held))})
            continue
        model, medians, curve, clip = _fit_mc1(fit)
        x = held.loc[:, list(MC1_FEATURES)].apply(pd.to_numeric, errors="coerce").fillna(pd.Series(medians, index=MC1_FEATURES))
        held["static_expected_bps"] = model.predict(x)
        shifts: dict[pd.Timestamp, float] = {}
        for day in pd.date_range(month.normalize(), (end - pd.Timedelta(days=1)).normalize(), freq="D", tz="UTC"):
            history = frame.loc[
                frame["__decision_ts__"].ge(day - pd.Timedelta(days=21)) & frame["__decision_ts__"].lt(day)
                & frame["policy_path_valid"].fillna(False).astype(bool)
                & frame["policy_label_available_ts"].lt(day)
                & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
            ]
            residual = pd.to_numeric(history["policy_net_bps"], errors="coerce").to_numpy(float) - curve[history["score_band"].to_numpy(int)]
            shifts[day] = _robust_mean(residual, trim=.10) if len(residual) else 0.0
        held["recent_shift_bps"] = held["__decision_ts__"].dt.normalize().map(shifts).fillna(0.0)
        held["mc1_expected_bps"] = held["static_expected_bps"] + held["recent_shift_bps"]
        held["mc1_family"] = family
        outputs.append(held)
        audit.append({"family": family, "month": f"{month:%Y-%m}", "status": "scored", "train_rows": int(len(fit)), "held_rows": int(len(held)), "clip_low": clip[0], "clip_high": clip[1]})
    prediction = pd.concat(outputs, ignore_index=True)
    path = out / f"enhanced_{family}_mc1_predictions.parquet"
    prediction.to_parquet(path, index=False, compression="zstd")
    return prediction, pd.DataFrame(audit)


def _dual_admission(frame: pd.DataFrame, priority: str) -> pd.DataFrame:
    """Apply the frozen dual-map gate without importing the portfolio engine."""

    required = {
        "enhanced_base_routed", "policy_path_valid", "policy_net_bps",
        "policy_exit_bar_15m", "current_mc1_expected_bps",
        "bcf_mc1_expected_bps", priority,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dual admission frame lacks fields: {missing}")
    valid = (
        frame["enhanced_base_routed"].fillna(False).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_exit_bar_15m"], errors="coerce"))
        & pd.to_numeric(frame["current_mc1_expected_bps"], errors="coerce").ge(MC1_THRESHOLD_BPS)
        & pd.to_numeric(frame["bcf_mc1_expected_bps"], errors="coerce").ge(MC1_THRESHOLD_BPS)
    )
    admitted = frame.loc[valid].copy()
    admitted["auction_rank"] = admitted.groupby("__decision_ts__", sort=False)[priority].rank(pct=True, method="average")
    return admitted


def _portfolio_input(frame: pd.DataFrame, priority: str) -> pd.DataFrame:
    # Import the broad portfolio package only at the terminal replay step.
    # It brings optional live/model-loader dependencies that are irrelevant
    # to score production and must not delay target-free OOS scoring.
    from extreme_price_movements.portfolio_policy_replay import normalise_candidate_table
    admitted = _dual_admission(frame, priority)
    # Reuse the same normalized candidate surface as the canonical controlled
    # portfolio adapter.  Only auction priority differs: BCF mapped EV is the
    # live dual-successor priority, not final-score percentile.
    exit_bar = pd.to_numeric(admitted["policy_exit_bar_15m"], errors="coerce").astype(int)
    decision = pd.to_datetime(admitted["__decision_ts__"], utc=True)
    candidate = pd.DataFrame({
        "timestamp": decision,
        "symbol": admitted["__symbol__"].astype(str), "side": "long",
        "strategy_id": "strict_r3_enhanced_live_stack_long",
        "policy_archetype": "strict_r3_enhanced_live_stack_long",
        "normalized_rank_score": admitted["auction_rank"].to_numpy(float),
        "strategy_rank_pct": admitted["auction_rank"].to_numpy(float),
        "base_strategy_threshold": 0.0,
        "calibrated_score": pd.to_numeric(admitted[priority], errors="coerce").to_numpy(float),
        "entry_price": pd.to_numeric(admitted["policy_entry_price"], errors="coerce"),
        "exit_timestamp": decision + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
        "exit_price": pd.to_numeric(admitted["policy_exit_price"], errors="coerce"),
        "net_return": pd.to_numeric(admitted["policy_net_bps"], errors="coerce") / 10_000.0,
        "gross_return": pd.to_numeric(admitted["policy_gross_bps"], errors="coerce") / 10_000.0,
        "holding_bars": exit_bar + 1,
        "simple_policy_exit_reason": admitted["policy_exit_reason"].astype(str),
        "fees_bps": 100.0, "slippage_bps": 0.0, "expected_friction_bps": 100.0,
        "price_gap_bps": 0.0, "liquidity_capacity_weight": 1.0,
        "source_month": decision.dt.strftime("%Y-%m"), "candidate_id": admitted["candidate_id"].astype(str),
        "mapped_expected_net_bps": pd.to_numeric(admitted[priority], errors="coerce"),
        "policy_outcome_available": np.ones(len(admitted), dtype=bool),
    })
    return normalise_candidate_table(candidate)


def _portfolio_metrics(
    frame: pd.DataFrame,
    label: str,
    period: str,
    out: Path,
    *,
    max_new_entries_per_bar: int | None = None,
) -> dict[str, object]:
    from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _metrics, _params
    from extreme_price_movements.portfolio_policy_replay import replay_candidates
    if max_new_entries_per_bar is not None and int(max_new_entries_per_bar) <= 0:
        raise ValueError("max_new_entries_per_bar must be positive when supplied")
    candidates = _portfolio_input(frame, "bcf_mc1_expected_bps")
    params = _params()
    if max_new_entries_per_bar is not None:
        # Reporting-only sensitivity: retain every canonical constraint and
        # vary only the timestamp-level entry capacity requested by the audit.
        params = replace(
            params,
            max_new_entries_per_bar=int(max_new_entries_per_bar),
            max_new_entries_per_strategy_per_bar=int(max_new_entries_per_bar),
        )
    decisions, equity, _ = replay_candidates(candidates, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perps", initial_wallet=1000.0)
    # The canonical normaliser omits this research-only coverage flag.  Every
    # candidate reaching this adapter was explicitly label-valid, so restore
    # the equivalent terminal metric field without changing auction inputs.
    if "policy_outcome_available" not in decisions.columns:
        decisions["policy_outcome_available"] = True
    decisions.to_parquet(out / f"{label}_{period}_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{label}_{period}_equity.parquet", index=False, compression="zstd")
    metric = _metrics(decisions, equity, label, period)
    metric["candidate_admitted_rows"] = int(len(candidates))
    metric["admission_threshold_bps"] = MC1_THRESHOLD_BPS
    metric["max_new_entries_per_bar"] = int(params.max_new_entries_per_bar)
    return metric


def _baseline(paths: Paths, policy: pd.DataFrame) -> pd.DataFrame:
    cols = ["candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score", "mc1_expected_bps"]
    current = pd.read_parquet(paths.current_mc1, columns=cols).rename(columns={"final_score": "current_final_score", "mc1_expected_bps": "current_mc1_expected_bps"})
    bcf = pd.read_parquet(paths.bcf_mc1, columns=cols).rename(columns={"final_score": "bcf_final_score", "mc1_expected_bps": "bcf_mc1_expected_bps"})
    current["__decision_ts__"] = pd.to_datetime(current["__decision_ts__"], utc=True)
    bcf["__decision_ts__"] = pd.to_datetime(bcf["__decision_ts__"], utc=True)
    # The final report has predeclared evaluation periods.  Keeping only
    # those rows avoids materialising the entire historical MC1 ledger merely
    # to replay a terminal report; it does not change any scored identity.
    evaluation = _evaluation_mask(current)
    current = current.loc[evaluation].copy()
    bcf = bcf.loc[_evaluation_mask(bcf)].copy()
    relevant_ids = pd.Index(current["candidate_id"].astype(str)).union(
        pd.Index(bcf["candidate_id"].astype(str))
    )
    policy = policy.loc[policy["candidate_id"].astype(str).isin(relevant_ids)].copy()
    current = current.drop(columns=["__symbol__", "side_name"])
    result = bcf.merge(current, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one").merge(policy, on="candidate_id", how="left", validate="one_to_one")
    result["enhanced_base_routed"] = True  # stored maps are emitted only after the live current route
    return result


def _evaluation_mask(frame: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    mask = pd.Series(False, index=frame.index)
    for start, end in EVALUATION_PERIODS.values():
        mask |= ts.ge(start) & ts.lt(end)
    return mask


def _combined_challenger(current: pd.DataFrame, bcf: pd.DataFrame) -> pd.DataFrame:
    left_keep = [
        "candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps",
        "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
        "policy_cost_bps", "side_name", "enhanced_base_routed",
    ]
    left = current.loc[:, left_keep].rename(columns={
        "final_score": "current_final_score", "mc1_expected_bps": "current_mc1_expected_bps",
    })
    right = bcf.loc[:, ["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps"]].rename(columns={
        "final_score": "bcf_final_score", "mc1_expected_bps": "bcf_mc1_expected_bps",
    })
    result = left.merge(right, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
    # Candidate IDs are the frozen target-free identity contract:
    # <Kraken symbol>|<side>|<signal timestamp>.  Recovering the symbol from
    # that immutable identifier is causal and avoids carrying it through the
    # compact model panel solely for terminal portfolio reporting.
    result["__symbol__"] = result["candidate_id"].astype(str).str.split("|", n=1, expand=True)[0]
    if result["__symbol__"].eq("").any() or result["__symbol__"].isna().any():
        raise AssertionError("candidate identity lacks portfolio symbol")
    return result


def _causality_audit(
    target_free_root: Path,
    fit_audit: pd.DataFrame,
    mc1_audit: pd.DataFrame,
    *,
    score_architecture: str = "base_consensus_correctness",
    pairwise_mode: str = "none",
    integration_spec: BpsIntegrationSpec | None = None,
    feature_contract: str = "current",
    trust_arm: str = "generic_correctness",
) -> dict[str, object]:
    panel = pd.read_parquet(next((target_free_root / "current").glob("*.parquet")))
    prohibited = {"policy_path_valid", "policy_net_bps", "policy_gross_bps", "policy_label_available_ts", "policy_exit_bar_15m"}
    if prohibited.intersection(panel.columns):
        raise AssertionError("target-free score receipt contains outcome columns")
    if fit_audit["train_rows"].le(0).any():
        raise AssertionError("downstream fold fitting incomplete")
    expected_heads = 0 if score_architecture == "base_only" else EXPECTED_RESEARCH_HEADS
    if fit_audit["head_count"].ne(expected_heads).any():
        raise AssertionError("score architecture has an unexpected head count")
    if mc1_audit.loc[mc1_audit["status"].eq("scored"), "train_rows"].le(0).any():
        raise AssertionError("MC1 fold lacks strict training support")
    trust_description = (
        "neutralised; no trust authority"
        if score_architecture == "base_only" or trust_arm == "none"
        else (
            "historical top-30% residual correctness demotion"
            if trust_arm == "generic_correctness"
            else f"strict-prequential enhanced-base adverse-tail trust; bounded {TAIL_TRUST_MAX_DEMOTION:.0%} demotion"
        )
    )
    return {
        "held_score_panels": "target-free persisted before policy labels joined", "reserve": f"{RESERVE_DAYS} calendar days excluded from downstream supervised fits",
        "base": "strict-OOS equal B0/efficiency/timing direct source",
        "score_architecture": score_architecture,
        "meta_feature_contract": feature_contract,
        "pairwise_mode": pairwise_mode,
        "bps_integration": (
            BPS_INTEGRATION_SPECS["rank_75_25"].name
            if integration_spec is None else integration_spec.name
        ),
        "bps_residual_map": (
            "not used; frozen rank-space blend control"
            if integration_spec is None or integration_spec.is_rank_control
            else "isotonic raw-consensus-to-policy-residual map fitted only on the prior resolved 28-day reserve"
        ),
        "pairwise_training_target": (
            "resolved policy_net_bps ordering within timestamp-local base-near-tie pairs"
            if pairwise_mode != "none" else None
        ),
        "heads": "five retrained selected LambdaRank residual heads" if expected_heads else "none; base-only ablation",
        # Keep the old receipt key for backward-compatible audit consumers;
        # ``trust`` is the canonical name for the new authority ablation.
        "correctness": trust_description,
        "trust": trust_description,
        "mc1": "two family-specific prequential HGB absolute-EV maps with prior-resolved 21-day residual shifts",
        "admission": "dual current and BCF MC1 >= +30 bps; priority BCF map", "portfolio": "existing global constrained mirror",
    }


def _finalize(
    paths: Paths,
    out: Path,
    current_pred: pd.DataFrame,
    bcf_pred: pd.DataFrame,
    fit_audit: pd.DataFrame,
    mc1_audit: pd.DataFrame,
    label_spec: PolicyConversionLabelSpec,
    target_free_source: Path | None = None,
    score_architecture: str = "base_consensus_correctness",
    pairwise_mode: str = "none",
    integration_spec: BpsIntegrationSpec | None = None,
    feature_contract: str = "current",
    trust_arm: str = "generic_correctness",
) -> Path:
    """Run only the terminal matched baseline/challenger comparison.

    This is restart-safe by design: target-free fold scores and strict MC1
    predictions are immutable once written, so a reporting/identity defect
    never justifies refitting upstream OOS models.
    """

    # These predictions have already been persisted as immutable MC1
    # receipts.  Restricting only terminal reporting to the predeclared held
    # periods materially lowers memory use without changing the score or any
    # candidate used by the replay.
    current_pred = current_pred.loc[_evaluation_mask(current_pred)].copy()
    bcf_pred = bcf_pred.loc[_evaluation_mask(bcf_pred)].copy()
    policy = _load_policy(paths)
    challenger = _combined_challenger(current_pred, bcf_pred)
    baseline = _baseline(paths, policy)
    del policy, current_pred, bcf_pred
    gc.collect()
    # The immutable live baseline was historically emitted on a narrower
    # dual-score universe.  It would be invalid to call a full enhanced-route
    # versus baseline comparison a delta: the population itself differs.
    # Keep that broader output as coverage evidence, but measure every delta
    # on exactly the baseline candidate identities.
    baseline_ids = pd.Index(baseline["candidate_id"].astype(str).unique())
    challenger_matched = challenger.loc[challenger["candidate_id"].astype(str).isin(baseline_ids)].copy()
    if challenger_matched.empty:
        raise AssertionError("enhanced challenger has no common candidate identities with live baseline")
    results: list[dict[str, object]] = []
    for period, (start, end) in EVALUATION_PERIODS.items():
        for label, frame in (
            ("live_baseline", baseline),
            ("enhanced_matched_stack", challenger_matched),
            ("enhanced_full_stack_coverage_only", challenger),
        ):
            part = frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)].copy()
            results.append(_portfolio_metrics(part, label, period, out))
    metrics = pd.DataFrame(results)
    metric_label = "arm"
    left = metrics.loc[metrics[metric_label].eq("live_baseline")].set_index("period")
    right = metrics.loc[metrics[metric_label].eq("enhanced_matched_stack")].set_index("period")
    shared = left.index.intersection(right.index)
    deltas = pd.DataFrame({"period": shared})
    for field in ("accepted_rows", "realised_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown"):
        if field in left and field in right:
            deltas[f"delta_{field}"] = right.loc[shared, field].to_numpy(float) - left.loc[shared, field].to_numpy(float)
    metrics.to_parquet(out / "live_like_portfolio_metrics.parquet", index=False, compression="zstd")
    deltas.to_parquet(out / "delta_vs_live_baseline.parquet", index=False, compression="zstd")
    causality = _causality_audit(
        out / "target_free_scores", fit_audit, mc1_audit,
        score_architecture=score_architecture,
        pairwise_mode=pairwise_mode,
        integration_spec=integration_spec,
        feature_contract=feature_contract,
        trust_arm=trust_arm,
    )
    manifest = {
        "schema": SCHEMA, "scope": "offline research only; does not alter the live stack",
        "enhanced_base": "strict-OOS equal B0/efficiency/timing direct blend", "base_route": "timestamp-local top 30%",
        "policy_conversion_label": {
            "name": label_spec.name, "description": label_spec.description,
            "source": label_spec.source, "edges_bps": list(label_spec.edges_bps),
            "objective": label_spec.objective, "clip_abs_bps": label_spec.clip_abs_bps,
            "control_note": "residual_actual_100_30_90 reproduces the actual historical runner code; it is not the stale parent JSON declaration",
        },
        "score_architecture": score_architecture,
        "meta_feature_contract": feature_contract,
        "trust_arm": trust_arm,
        "pairwise_mode": pairwise_mode,
        "bps_integration": {
            "name": BPS_INTEGRATION_SPECS["rank_75_25"].name if integration_spec is None else integration_spec.name,
            "description": BPS_INTEGRATION_SPECS["rank_75_25"].description if integration_spec is None else integration_spec.description,
            "reserve_contract": "prior resolved 28-day reserve only; held outcomes excluded",
        },
        "downstream": "five selected retrained residual LambdaRank heads, bounded integration, declared trust authority, paired current/BCF-like MC1 maps",
        "admission": "both retrained family maps >= +30 bps; BCF-like mapped EV priority", "periods": {k: [v[0].isoformat(), v[1].isoformat()] for k, v in EVALUATION_PERIODS.items()},
        "comparison_population": {
            "baseline_rows": int(len(baseline)), "enhanced_rows": int(len(challenger)), "matched_rows": int(len(challenger_matched)),
            "delta_definition": "enhanced_matched_stack minus live_baseline on exact common candidate_id universe; enhanced_full_stack_coverage_only is descriptive only",
        },
        "paths": {
            **{k: str(v) for k, v in vars(paths).items()},
            "target_free_source": str(target_free_source.resolve()) if target_free_source is not None else "materialized_in_this_run",
        }, "source_sha256": {
            **{k: _sha256([v]) for k, v in vars(paths).items()},
            **({"target_free_source": _sha256([target_free_source])} if target_free_source is not None else {}),
        },
        "causality": causality,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def _finalize_from_receipts(
    paths: Paths,
    out: Path,
    fit_audit: pd.DataFrame,
    mc1_audit: pd.DataFrame,
    label_spec: PolicyConversionLabelSpec,
    target_free_source: Path | None,
    score_architecture: str,
    pairwise_mode: str = "none",
    integration_spec: BpsIntegrationSpec | None = None,
    feature_contract: str = "current",
    trust_arm: str = "generic_correctness",
) -> Path:
    """Finish a run from compact persisted MC1 receipts, not live arrays."""
    current_path = out / "enhanced_current_mc1_predictions.parquet"
    bcf_path = out / "enhanced_bcf_mc1_predictions.parquet"
    if not current_path.exists() or not bcf_path.exists():
        raise FileNotFoundError("terminal finalization requires both persisted MC1 receipts")
    current_cols = [
        "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed",
        "final_score", "mc1_expected_bps", "policy_path_valid", "policy_gross_bps",
        "policy_net_bps", "policy_exit_bar_15m", "policy_entry_price",
        "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
        "policy_cost_bps",
    ]
    bcf_cols = ["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps"]
    current = pd.read_parquet(current_path, columns=current_cols)
    bcf = pd.read_parquet(bcf_path, columns=bcf_cols)
    return _finalize(
        paths, out, current, bcf, fit_audit, mc1_audit, label_spec,
        target_free_source, score_architecture, pairwise_mode, integration_spec,
        feature_contract, trust_arm,
    )


def run(
    paths: Paths,
    out: Path,
    *,
    label_spec: PolicyConversionLabelSpec,
    shared_target_free_root: Path | None = None,
    score_architecture: str = "base_consensus_correctness",
    pairwise_mode: str = "none",
    integration_spec: BpsIntegrationSpec | None = None,
    feature_contract: str = "current",
    trust_arm: str = "generic_correctness",
) -> Path:
    if out.exists():
        raise FileExistsError(out)
    integration_spec = BPS_INTEGRATION_SPECS["rank_75_25"] if integration_spec is None else integration_spec
    out.mkdir(parents=True)
    base_fields = _base_fields(paths)
    if shared_target_free_root is None:
        target_free_root, coverage = _materialize_target_free(paths, out, base_fields)
    else:
        target_free_root, coverage = _reuse_target_free(shared_target_free_root, base_fields)
    coverage.to_parquet(out / "target_free_feature_coverage.parquet", index=False, compression="zstd")
    if coverage["feature_complete_fraction"].lt(.90).any():
        raise AssertionError("enhanced feature coverage below 90%")
    policy = _load_policy(paths)
    fit_rows: list[dict[str, object]] = []
    for month in SCORE_MONTHS:
        print(json.dumps({"event": "score_month_begin", "month": f"{month:%Y-%m}"}), flush=True)
        audit, _, _ = _score_fold(
            target_free_root, policy, base_fields, label_spec,
            score_architecture, pairwise_mode, integration_spec, feature_contract, month, out, trust_arm,
        )
        fit_rows.append(audit)
        print(json.dumps({"event": "score_month_complete", **audit}), flush=True)
    fit_audit = pd.DataFrame(fit_rows)
    fit_audit.to_parquet(out / "downstream_fit_audit.parquet", index=False, compression="zstd")
    current_panel = _read_score_panels(out, "current", policy)
    current_pred, current_audit = _mc1_predictions(current_panel, "current", out)
    # MC1 prediction receipts are persisted immediately.  Do not retain a
    # million-row current-family panel while fitting the BCF family; that
    # historical peak was capable of interrupting a valid base-only control.
    del current_panel, current_pred
    gc.collect()
    bcf_panel = _read_score_panels(out, "bcf", policy)
    bcf_pred, bcf_audit = _mc1_predictions(bcf_panel, "bcf", out)
    del bcf_panel, bcf_pred
    gc.collect()
    mc1_audit = pd.concat([current_audit, bcf_audit], ignore_index=True)
    mc1_audit.to_parquet(out / "mc1_fit_audit.parquet", index=False, compression="zstd")
    return _finalize_from_receipts(
        paths, out, fit_audit, mc1_audit,
        label_spec, shared_target_free_root, score_architecture, pairwise_mode, integration_spec,
        feature_contract, trust_arm,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--current-mc1", type=Path, required=True)
    parser.add_argument("--bcf-mc1", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--policy-conversion-label",
        choices=tuple(POLICY_CONVERSION_LABEL_SPECS),
        default="residual_actual_100_30_90",
        help="resolved-policy supervision for the five consensus heads; never an inference feature",
    )
    parser.add_argument(
        "--shared-target-free-root", type=Path,
        help="immutable target-free monthly source; validates coverage/no policy outcome leakage before reusing it",
    )
    parser.add_argument(
        "--score-architecture", choices=SCORE_ARCHITECTURES,
        default="base_consensus_correctness",
        help="base/consensus/correctness waterfall arm; MC1/admission/portfolio remain unchanged",
    )
    parser.add_argument(
        "--pairwise-mode", choices=PAIRWISE_MODES, default="none",
        help="narrow base-near-tie pairwise correction objective for consensus heads",
    )
    parser.add_argument(
        "--bps-integration", choices=tuple(BPS_INTEGRATION_SPECS), default="rank_75_25",
        help="predeclared base-plus-residual integration; residual maps use only each fold's prior resolved reserve",
    )
    parser.add_argument(
        "--meta-feature-contract", choices=META_FEATURE_CONTRACTS, default="current",
        help="causal feature subset for the five residual heads; does not alter the base, MC1 or portfolio contracts",
    )
    parser.add_argument(
        "--trust-arm", choices=TRUST_ARMS, default="generic_correctness",
        help="post-consensus trust authority; never changes the base, MC1, admission or auction",
    )
    parser.add_argument("--resume-completed-root", type=Path, default=None,
                        help="Finalise already-written target-free and MC1 panels without refitting them.")
    parser.add_argument("--resume-score-panels-root", type=Path, default=None,
                        help="Resume immutable target-free score panels: rebuild compact causal audit, MC1 maps and terminal replay.")
    args = parser.parse_args()
    paths = Paths(
        raw_ledger=args.raw_ledger, direct_root=args.direct_root,
        policy_root=args.policy_root, current_mc1=args.current_mc1,
        bcf_mc1=args.bcf_mc1, bundle_root=args.bundle_root,
    )
    label_spec = POLICY_CONVERSION_LABEL_SPECS[str(args.policy_conversion_label)]
    integration_spec = BPS_INTEGRATION_SPECS[str(args.bps_integration)]
    _validate_policy_label_pairwise_compatibility(label_spec, args.pairwise_mode)
    if args.resume_completed_root is not None and args.resume_score_panels_root is not None:
        raise ValueError("choose only one resume mode")
    if args.resume_completed_root is not None:
        root = args.resume_completed_root.resolve()
        fit_audit = pd.read_parquet(root / "downstream_fit_audit.parquet")
        mc1_audit = pd.read_parquet(root / "mc1_fit_audit.parquet")
        print(json.dumps({"event": "complete", "out": str(_finalize_from_receipts(
            paths, root, fit_audit, mc1_audit, label_spec,
            args.shared_target_free_root, args.score_architecture, args.pairwise_mode, integration_spec,
            args.meta_feature_contract, args.trust_arm,
        ))}), flush=True)
    elif args.resume_score_panels_root is not None:
        if not integration_spec.is_rank_control:
            raise ValueError(
                "reserve-calibrated bps integration cannot rebuild from score panels alone; "
                "its immutable reserve-map receipts must be produced by a full scoring run"
            )
        root = args.resume_score_panels_root.resolve()
        target_root = (args.shared_target_free_root or (root / "target_free_monthly")).resolve()
        policy = _load_policy(paths)
        fit_audit = _rebuild_score_fit_audit(
            target_root, policy, label_spec, args.score_architecture, args.pairwise_mode,
            args.meta_feature_contract, args.trust_arm,
        )
        fit_audit.to_parquet(root / "downstream_fit_audit.parquet", index=False, compression="zstd")
        current_panel = _read_score_panels(root, "current", policy)
        _, current_audit = _mc1_predictions(current_panel, "current", root)
        del current_panel
        gc.collect()
        bcf_panel = _read_score_panels(root, "bcf", policy)
        _, bcf_audit = _mc1_predictions(bcf_panel, "bcf", root)
        del bcf_panel, policy
        gc.collect()
        mc1_audit = pd.concat([current_audit, bcf_audit], ignore_index=True)
        mc1_audit.to_parquet(root / "mc1_fit_audit.parquet", index=False, compression="zstd")
        print(json.dumps({"event": "complete", "out": str(_finalize_from_receipts(
            paths, root, fit_audit, mc1_audit, label_spec,
            args.shared_target_free_root, args.score_architecture, args.pairwise_mode, integration_spec,
            args.meta_feature_contract, args.trust_arm,
        ))}), flush=True)
    else:
        print(json.dumps({"event": "complete", "out": str(run(
            paths, args.out.resolve(), label_spec=label_spec,
            shared_target_free_root=args.shared_target_free_root,
            score_architecture=args.score_architecture,
            pairwise_mode=args.pairwise_mode,
            integration_spec=integration_spec,
            feature_contract=args.meta_feature_contract,
            trust_arm=args.trust_arm,
        ))}), flush=True)


if __name__ == "__main__":
    main()
