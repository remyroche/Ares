#!/usr/bin/env python3
"""Offline T6/T9 score-construction and dual-MC1 ablation runner.

This is deliberately separate from every live producer.  It implements the
two-stage research contract:

    enhanced base + frozen T6/T9 ranks
        -> candidate upstream-score contract
        -> separately fitted, strict-prequential Current and BCF MC1 maps
        -> dual admission -> unchanged constrained portfolio replay.

All score panels are target-free.  The rich-policy outcome is joined only for
the strict-prequential MC1 fit and the downstream replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402
import run_strict_r3_orthogonal_meta_mc1 as mc1  # noqa: E402


SCHEMA = "strict_r3_t6t9_dual_mc1_ablation_v1"
SEED = 1729
ROUTE_FRACTION = 0.30
DEFAULT_LEDGER = tuple(pd.date_range("2025-11-01", "2026-07-01", freq="MS", tz="UTC"))
DEFAULT_EVALUATION = tuple(pd.date_range("2026-05-01", "2026-07-01", freq="MS", tz="UTC"))
PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
    "policy_cost_bps", "semantic_path_valid", "semantic_sequence", "semantic_speed_bin",
    "semantic_persistence_bin", "semantic_pre_adverse_bin", "semantic_policy_conversion_bin",
    "semantic_exit_reason", "semantic_composite", "semantic_tbm_event",
})
POLICY_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
)
PARENT_COLUMNS = (
    "candidate_id", "__decision_ts__", "side_name", "enhanced_base_bps", "base_rank42",
    "base_anchor_bps", "correctness_rank",
)
COMPONENT_COLUMNS = (
    "candidate_id", "__decision_ts__", "side_name", "base_bps", "efficiency_bps", "timing_bps",
    "enhanced_base_bps", "base_rank_ts", "enhanced_base_routed",
)
SCORE_CONTRACTS: dict[str, tuple[float, float, float]] = {
    "S0_BASE": (1.00, 0.00, 0.00),
    "S1_CURRENT_T6T9": (0.75, 0.125, 0.125),
    "S2_T6_ONLY_25": (0.75, 0.25, 0.00),
    "S3_T9_ONLY_25": (0.75, 0.00, 0.25),
    "S4_T6_10": (0.90, 0.10, 0.00),
    "S5_T6_15": (0.85, 0.15, 0.00),
    "S6_T6_20": (0.80, 0.20, 0.00),
    "S7_T6_25": (0.75, 0.25, 0.00),
    "S8_T6_30": (0.70, 0.30, 0.00),
    "S9_T6_25_T9_0": (0.75, 0.25, 0.00),
    "S10_T6_225_T9_025": (0.75, 0.225, 0.025),
    "S11_T6_20_T9_05": (0.75, 0.20, 0.05),
    "S12_T6_1875_T9_0625": (0.75, 0.1875, 0.0625),
    "S13_T6_15_T9_10": (0.75, 0.15, 0.10),
    "S14_EQUAL_T6_T9": (0.75, 0.125, 0.125),
    # Stage-1 refinement after selecting the 80:20 T6:T9 allocation.  The
    # existing S11 arm is the 25% member of this total-authority ladder.
    "S15_M10_T6_08_T9_02": (0.90, 0.08, 0.02),
    "S16_M15_T6_12_T9_03": (0.85, 0.12, 0.03),
    "S17_M20_T6_16_T9_04": (0.80, 0.16, 0.04),
    "S19_M30_T6_24_T9_06": (0.70, 0.24, 0.06),
}


@dataclass(frozen=True)
class MC1Capacity:
    """Fixed mapper capacity used only by this offline challenger runner."""

    max_depth: int = 2
    max_leaf_nodes: int = 4
    max_iter: int = 80
    learning_rate: float = .04
    l2_regularization: float = 20.0
    min_samples_leaf: int = 100


DEFAULT_MC1_CAPACITY = MC1Capacity()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for child in paths:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _month_path(root: Path, token: str) -> Path:
    return root / "target_free_monthly" / f"month={token}" / "scores_features.parquet"


def _head_path(root: Path, arm: str, support: str, token: str) -> Path:
    return root / "target_free_scores" / f"{arm}__{support}" / f"month={token}.parquet"


def _read_target_free(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(columns) - available)
    if missing:
        raise AssertionError(f"{path}: missing required columns {missing}")
    probe = pd.read_parquet(path, columns=list(available & PROHIBITED))
    leaked = sorted(PROHIBITED.intersection(probe.columns))
    if leaked:
        raise AssertionError(f"{path}: outcome columns in target-free receipt {leaked}")
    return pd.read_parquet(path, columns=list(columns))


def _route(frame: pd.DataFrame) -> pd.Series:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "enhanced_base_bps"]].copy()
    work["__position__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "enhanced_base_bps", "candidate_id"], ascending=[True, False, True], kind="stable")
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["routed"] = rank < np.maximum(1, np.ceil(size * ROUTE_FRACTION))
    return work.sort_values("__position__", kind="stable")["routed"].astype(bool).reset_index(drop=True)


def _rank_from_bps(frame: pd.DataFrame, column: str) -> pd.Series:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", column]].copy()
    work["__position__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, True, True], kind="stable")
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["rank"] = (rank + .5) / size
    return work.sort_values("__position__", kind="stable")["rank"].astype(np.float32).reset_index(drop=True)


def _parse_months(raw: str | None, default: tuple[pd.Timestamp, ...]) -> tuple[pd.Timestamp, ...]:
    if not raw:
        return default
    return tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in raw.split(",") if value)


def _load_policy(path: Path) -> pd.DataFrame:
    policy = pd.read_parquet(path, columns=list(POLICY_COLUMNS))
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy ledger contains duplicate candidate identities")
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    return policy


def _base_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["b0_rank"] = _rank_from_bps(result, "base_bps")
    result["efficiency_rank"] = _rank_from_bps(result, "efficiency_bps")
    result["timing_rank"] = _rank_from_bps(result, "timing_bps")
    values = result.loc[:, ["base_bps", "efficiency_bps", "timing_bps"]].to_numpy(float)
    result["base_component_min"] = np.nanmin(values, axis=1)
    result["base_component_median"] = np.nanmedian(values, axis=1)
    result["base_component_max"] = np.nanmax(values, axis=1)
    result["base_component_range"] = result["base_component_max"] - result["base_component_min"]
    result["e_minus_b0"] = result["efficiency_bps"] - result["base_bps"]
    result["t_minus_b0"] = result["timing_bps"] - result["base_bps"]
    result["e_minus_t"] = result["efficiency_bps"] - result["timing_bps"]
    result["abs_e_minus_b0"] = result["e_minus_b0"].abs()
    result["abs_t_minus_b0"] = result["t_minus_b0"].abs()
    result["abs_e_minus_t"] = result["e_minus_t"].abs()
    return result


def _load_family(
    *, family: str, p2_root: Path, component_root: Path, t6_root: Path, t9_root: Path,
    ledger_months: tuple[pd.Timestamp, ...], weights: tuple[float, float, float],
) -> pd.DataFrame:
    b_weight, t6_weight, t9_weight = weights
    pieces: list[pd.DataFrame] = []
    for month in ledger_months:
        token = f"{month:%Y-%m}"
        parent_path = p2_root / "target_free_scores" / family / f"month={token}.parquet"
        component_path = _month_path(component_root, token)
        t6_path = _head_path(t6_root, "T6_rank_error_ordinal", "S0_uniform", token)
        t9_path = _head_path(t9_root, "T9_exit5_ordinal", "S5_tbm_coarse", token)
        for path in (parent_path, component_path, t6_path, t9_path):
            if not path.exists():
                raise FileNotFoundError(path)
        parent_frame = _read_target_free(parent_path, PARENT_COLUMNS)
        component = _read_target_free(component_path, COMPONENT_COLUMNS)
        for part in (parent_frame, component):
            part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        if parent_frame["candidate_id"].duplicated().any() or component["candidate_id"].duplicated().any():
            raise AssertionError(f"{family} {token}: duplicate candidate identity")
        # The parent receipt is the sealed upstream score contract.  Its bps
        # scale differs monotonically from the later component-only receipt,
        # so route from this source to reproduce the existing challenger
        # population exactly; component rows are then used only for explicit
        # G1 base-composition geometry.
        routed = parent._exact_timestamp_top_fraction(parent_frame, "enhanced_base_bps", parent.BASE_ROUTE)
        parent_frame = parent_frame.loc[routed.to_numpy()].copy()
        parent_frame = parent_frame.rename(columns={"enhanced_base_bps": "parent_enhanced_base_bps"})
        t6 = _read_target_free(t6_path, ("candidate_id", "base_rank_ts", "head__cap80_ordinary__rank"))
        t9 = _read_target_free(t9_path, ("candidate_id", "base_rank_ts", "head__cap120_equal_month__rank"))
        t6 = t6.rename(columns={"base_rank_ts": "t6_base_rank", "head__cap80_ordinary__rank": "t6_rank"})
        t9 = t9.rename(columns={"base_rank_ts": "t9_base_rank", "head__cap120_equal_month__rank": "t9_rank"})
        # The component receipt is the frozen enhanced-base score/routing
        # source.  The parent MC1 receipt carries only calibration/trust
        # coordinates.  Their bps scores are monotone-equivalent but stored
        # under different scale conventions, so score magnitude is not an
        # identity key.  Candidate, decision time, and side remain exact keys.
        frame = parent_frame.merge(component, on=["candidate_id", "__decision_ts__", "side_name"], how="inner", validate="one_to_one")
        frame = frame.merge(t6, on="candidate_id", how="inner", validate="one_to_one").merge(t9, on="candidate_id", how="inner", validate="one_to_one")
        # T6/T9 were frozen against the component-route receipt while the
        # established upstream route is the parent’s exact top-30 contract.
        # Their identity intersection is therefore the predeclared, causal
        # research universe.  It is held constant across every score arm and
        # audited explicitly below; unsupported rows are not imputed.
        if frame.empty:
            raise AssertionError(f"{family} {token}: no common parent/T6/T9 routed identities")
        frame["parent_route_rows"] = np.int32(len(parent_frame))
        frame["common_t6t9_route_rows"] = np.int32(len(frame))
        if not np.allclose(frame["base_rank_ts"], frame["t6_base_rank"], equal_nan=False):
            raise AssertionError(f"{family} {token}: T6 base ranks diverge from enhanced base receipt")
        if not np.allclose(frame["base_rank_ts"], frame["t9_base_rank"], equal_nan=False):
            raise AssertionError(f"{family} {token}: T9 base ranks diverge from enhanced base receipt")
        frame = _base_geometry(frame)
        frame["b_rank"] = pd.to_numeric(frame["base_rank_ts"], errors="raise").astype(np.float32)
        frame["final_score"] = (
            b_weight * frame["b_rank"] + t6_weight * frame["t6_rank"] + t9_weight * frame["t9_rank"]
        ).astype(np.float32)
        # Preserve the M0 coordinate contract: the specialist-local combined
        # ranks remain independent B+head views.  The upstream score itself is
        # carried by final_score and is rebuilt for every score contract.
        frame["t6_consensus_rank"] = frame["t6_rank"].astype(np.float32)
        frame["t9_consensus_rank"] = frame["t9_rank"].astype(np.float32)
        frame["t6_combined_rank"] = (.75 * frame["b_rank"] + .25 * frame["t6_rank"]).astype(np.float32)
        frame["t9_combined_rank"] = (.75 * frame["b_rank"] + .25 * frame["t9_rank"]).astype(np.float32)
        frame["score_base_weight"] = np.float32(b_weight)
        frame["score_t6_weight"] = np.float32(t6_weight)
        frame["score_t9_weight"] = np.float32(t9_weight)
        pieces.append(frame)
    output = pd.concat(pieces, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError(f"{family}: duplicate candidate identities across monthly receipts")
    return output


def _geometry_features(frame: pd.DataFrame, blocks: tuple[str, ...], *, t9_visible: bool) -> list[str]:
    result: list[str] = []
    # Baseline exactly matches the supplied M0 dual-MC1 schema.
    baseline = [
        "final_score", "base_rank42", "base_anchor_bps", "correctness_rank",
        "t6_consensus_rank", "t6_combined_rank", "t9_consensus_rank", "t9_combined_rank",
    ]
    result.extend(baseline if t9_visible else [field for field in baseline if not field.startswith("t9_")])
    if "G1" in blocks:
        result.extend([
            "base_bps", "efficiency_bps", "timing_bps", "enhanced_base_bps", "b0_rank", "efficiency_rank", "timing_rank", "b_rank",
            "e_minus_b0", "t_minus_b0", "e_minus_t", "abs_e_minus_b0", "abs_t_minus_b0", "abs_e_minus_t",
            "base_component_min", "base_component_median", "base_component_max", "base_component_range",
        ])
    if "G2" in blocks:
        result.extend(["t6_rank", "t6_minus_base", "abs_t6_minus_base", "t6_minus_final", "abs_t6_minus_final", "t6_upgrade_strength", "t6_downgrade_strength"])
    if "G3" in blocks:
        if not t9_visible:
            raise ValueError("G3 requires T9 to be visible to MC1")
        result.extend(["t9_rank", "t9_minus_base", "abs_t9_minus_base", "t9_minus_final", "abs_t9_minus_final", "t9_good_tail", "t9_bad_tail"])
    if "G4" in blocks:
        if not t9_visible:
            raise ValueError("G4 requires T9 to be visible to MC1")
        result.extend(["t6_up_t9_good", "t6_up_t9_bad", "t6_down_t9_good", "t6_down_t9_bad", "t6_minus_t9", "abs_t6_minus_t9"])
    if "G5" in blocks:
        result.extend(["base_tail_p80", "base_tail_p90", "base_tail_p95", "t6_up_x_base_tail", "t6_down_x_base_tail", "abs_t6_base_x_tail"])
    if "G6" in blocks:
        if not t9_visible:
            raise ValueError("G6 requires T9 to be visible to MC1")
        result.extend(["base_x_t9_good", "base_x_t9_bad", "base_tail_x_t9_good", "base_tail_x_t9_bad"])
    if "G7" in blocks:
        result.extend(["final_minus_base", "abs_final_minus_base", "final_minus_t6", "final_minus_t9"])
    if "G8" in blocks:
        result.extend(["query_rank", "gap_to_query_best", "gap_to_top5_median", "query_score_iqr", "top5_score_iqr", "query_count", "gap_to_best_iqr", "gap_to_top5_iqr"])
    if "G9" in blocks:
        result.extend(["margin_base_route", "margin_high_conviction", "margin_correctness", "minimum_upstream_margin", "mean_upstream_margin", "n_marginal_conditions", "barely_survived_any_gate", "barely_survived_multiple_gates"])
    if "G10" in blocks:
        result.extend([
            "g10_global_ev_3d", "g10_global_ev_7d", "g10_global_ev_21d", "g10_global_ev_42d",
            "g10_global_ev_3m21_delta", "g10_score_ev_21d", "g10_score_n_21d", "g10_score_shrink_21d",
            "g10_score_severe_rate_21d", "g10_score_calibration_z_21d",
            "g10_base_ev_21d", "g10_base_n_21d", "g10_base_shrink_21d",
            "g10_t6_ev_21d", "g10_t6_n_21d", "g10_t6_shrink_21d",
            "g10_t9_ev_21d", "g10_t9_n_21d", "g10_t9_shrink_21d",
        ])
    return list(dict.fromkeys(result))


def _add_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["t6_minus_base"] = out["t6_rank"] - out["b_rank"]
    out["abs_t6_minus_base"] = out["t6_minus_base"].abs()
    out["t6_minus_final"] = out["t6_rank"] - out["final_score"]
    out["abs_t6_minus_final"] = out["t6_minus_final"].abs()
    out["t6_upgrade_strength"] = out["t6_minus_base"].clip(lower=0)
    out["t6_downgrade_strength"] = (-out["t6_minus_base"]).clip(lower=0)
    out["t9_minus_base"] = out["t9_rank"] - out["b_rank"]
    out["abs_t9_minus_base"] = out["t9_minus_base"].abs()
    out["t9_minus_final"] = out["t9_rank"] - out["final_score"]
    out["abs_t9_minus_final"] = out["t9_minus_final"].abs()
    out["t9_good_tail"] = (out["t9_rank"] >= .80).astype(np.int8)
    out["t9_bad_tail"] = (out["t9_rank"] <= .20).astype(np.int8)
    out["t6_up_t9_good"] = out["t6_upgrade_strength"] * out["t9_rank"]
    out["t6_up_t9_bad"] = out["t6_upgrade_strength"] * (1.0 - out["t9_rank"])
    out["t6_down_t9_good"] = out["t6_downgrade_strength"] * out["t9_rank"]
    out["t6_down_t9_bad"] = out["t6_downgrade_strength"] * (1.0 - out["t9_rank"])
    out["t6_minus_t9"] = out["t6_rank"] - out["t9_rank"]
    out["abs_t6_minus_t9"] = out["t6_minus_t9"].abs()
    out["base_tail_p80"] = (out["b_rank"] >= .80).astype(np.int8)
    out["base_tail_p90"] = (out["b_rank"] >= .90).astype(np.int8)
    out["base_tail_p95"] = (out["b_rank"] >= .95).astype(np.int8)
    tail = np.maximum(0.0, out["b_rank"] - .70)
    out["t6_up_x_base_tail"] = out["t6_upgrade_strength"] * tail
    out["t6_down_x_base_tail"] = out["t6_downgrade_strength"] * tail
    out["abs_t6_base_x_tail"] = out["abs_t6_minus_base"] * tail
    out["base_x_t9_good"] = out["b_rank"] * out["t9_rank"]
    out["base_x_t9_bad"] = out["b_rank"] * (1.0 - out["t9_rank"])
    out["base_tail_x_t9_good"] = tail * out["t9_rank"]
    out["base_tail_x_t9_bad"] = tail * (1.0 - out["t9_rank"])
    out["final_minus_base"] = out["final_score"] - out["b_rank"]
    out["abs_final_minus_base"] = out["final_minus_base"].abs()
    out["final_minus_t6"] = out["final_score"] - out["t6_rank"]
    out["final_minus_t9"] = out["final_score"] - out["t9_rank"]
    rows: list[pd.DataFrame] = []
    for _, part in out.groupby("__decision_ts__", sort=False):
        ordered = part.sort_values(["final_score", "candidate_id"], ascending=[False, True], kind="stable").copy()
        n = len(ordered)
        top5 = ordered.head(min(5, n))["final_score"].to_numpy(float)
        q75, q25 = np.nanpercentile(ordered["final_score"], [75, 25])
        ordered["query_rank"] = (np.arange(n, dtype=float) + .5) / n
        ordered["gap_to_query_best"] = float(ordered["final_score"].iloc[0]) - ordered["final_score"]
        ordered["gap_to_top5_median"] = float(np.nanmedian(top5)) - ordered["final_score"]
        ordered["query_score_iqr"] = max(float(q75 - q25), 1e-6)
        ordered["top5_score_iqr"] = max(float(np.subtract(*np.nanpercentile(top5, [75, 25]))), 1e-6) if len(top5) > 1 else 1e-6
        ordered["query_count"] = n
        ordered["gap_to_best_iqr"] = ordered["gap_to_query_best"] / ordered["query_score_iqr"]
        ordered["gap_to_top5_iqr"] = ordered["gap_to_top5_median"] / ordered["query_score_iqr"]
        rows.append(ordered)
    out = pd.concat(rows, ignore_index=True)
    out["margin_base_route"] = out["b_rank"] - (1.0 - ROUTE_FRACTION)
    out["margin_high_conviction"] = out["final_score"] - .80
    out["margin_correctness"] = out["correctness_rank"] - .50
    margins = out.loc[:, ["margin_base_route", "margin_high_conviction", "margin_correctness"]].to_numpy(float)
    out["minimum_upstream_margin"] = np.nanmin(margins, axis=1)
    out["mean_upstream_margin"] = np.nanmean(margins, axis=1)
    out["n_marginal_conditions"] = (margins < .05).sum(axis=1).astype(np.int8)
    out["barely_survived_any_gate"] = (out["minimum_upstream_margin"] < .05).astype(np.int8)
    out["barely_survived_multiple_gates"] = (out["n_marginal_conditions"] >= 2).astype(np.int8)
    return out


def _recent_calibration_block(frame: pd.DataFrame) -> pd.DataFrame:
    """Append causal, shrinkage-calibrated recent-support features.

    This function runs only after policy labels have been joined for fitting or
    replay.  For each UTC decision day it uses rows whose *labels were already
    available before that day began*.  It is consequently conservative (and
    excludes same-day outcomes) but never provides a score receipt with a
    target-derived field.
    """
    out = frame.copy()
    decision = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    available = pd.to_datetime(out["policy_label_available_ts"], utc=True, errors="coerce")
    out["__g10_day__"] = decision.dt.normalize()
    out["__g10_available_day__"] = available.dt.normalize()
    target = pd.to_numeric(out["policy_net_bps"], errors="coerce").clip(-600.0, 600.0)
    valid = out["policy_path_valid"].fillna(False).astype(bool) & target.notna()
    out["__g10_target__"] = target
    # State bins are target-free functions of the frozen score geometry.
    out["__g10_score_bin__"] = np.minimum(9, np.floor(pd.to_numeric(out["final_score"], errors="coerce") * 10.0)).fillna(0).astype(np.int8)
    out["__g10_base_bin__"] = np.minimum(4, np.floor(pd.to_numeric(out["b_rank"], errors="coerce") * 5.0)).fillna(0).astype(np.int8)
    out["__g10_t6_bin__"] = np.minimum(4, np.floor(pd.to_numeric(out["t6_rank"], errors="coerce") * 5.0)).fillna(0).astype(np.int8)
    out["__g10_t9_bin__"] = np.minimum(4, np.floor(pd.to_numeric(out["t9_rank"], errors="coerce") * 5.0)).fillna(0).astype(np.int8)
    names = [
        "g10_global_ev_3d", "g10_global_ev_7d", "g10_global_ev_21d", "g10_global_ev_42d", "g10_global_ev_3m21_delta",
        "g10_score_ev_21d", "g10_score_n_21d", "g10_score_shrink_21d", "g10_score_severe_rate_21d", "g10_score_calibration_z_21d",
        "g10_base_ev_21d", "g10_base_n_21d", "g10_base_shrink_21d",
        "g10_t6_ev_21d", "g10_t6_n_21d", "g10_t6_shrink_21d",
        "g10_t9_ev_21d", "g10_t9_n_21d", "g10_t9_shrink_21d",
    ]
    for name in names:
        out[name] = np.float32(0.0)

    def robust_mean(values: pd.Series) -> float:
        array = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(float))
        if not len(array):
            return 0.0
        trim = int(np.floor(len(array) * .10))
        if trim and len(array) > 2 * trim:
            array = array[trim:-trim]
        return float(array.mean())

    def state_statistics(history: pd.DataFrame, state: str, current: pd.DataFrame, *, prefix: str, global_ev: float) -> None:
        # 80 pseudo-observations supplies conservative empirical-Bayes
        # shrinkage.  It is deliberately fixed before outcome comparisons.
        stats = history.groupby(state, observed=True)["__g10_target__"].agg(["mean", "count", "std"])
        key = current[state]
        count = key.map(stats["count"]).fillna(0.0).to_numpy(float)
        mean = key.map(stats["mean"]).fillna(global_ev).to_numpy(float)
        std = key.map(stats["std"]).fillna(250.0).clip(lower=25.0).to_numpy(float)
        shrink = count / (count + 80.0)
        posterior = shrink * mean + (1.0 - shrink) * global_ev
        out.loc[current.index, f"{prefix}_ev_21d"] = posterior.astype(np.float32)
        out.loc[current.index, f"{prefix}_n_21d"] = count.astype(np.float32)
        out.loc[current.index, f"{prefix}_shrink_21d"] = shrink.astype(np.float32)
        if prefix == "g10_score":
            severe = history.assign(__severe__=(history["__g10_target__"] <= -200.0).astype(float)).groupby(state, observed=True)["__severe__"].mean()
            rate = key.map(severe).fillna((history["__g10_target__"] <= -200.0).mean()).to_numpy(float)
            z = (posterior - global_ev) / (std / np.sqrt(np.maximum(count, 1.0)) + 1.0)
            out.loc[current.index, "g10_score_severe_rate_21d"] = rate.astype(np.float32)
            out.loc[current.index, "g10_score_calibration_z_21d"] = z.astype(np.float32)

    for day, current in out.groupby("__g10_day__", sort=True):
        # Only completely resolved, prior-day information is visible.  The
        # decision timestamp interval prevents old labels from being attached
        # to a different decision day by accident.
        history_all = out.loc[
            valid
            & out["__g10_day__"].lt(day)
            & out["__g10_available_day__"].lt(day)
            & out["__g10_day__"].ge(day - pd.Timedelta(days=42))
        ].copy()
        horizon_ev: dict[int, float] = {}
        for horizon in (3, 7, 21, 42):
            recent = history_all.loc[history_all["__g10_day__"].ge(day - pd.Timedelta(days=horizon))]
            value = robust_mean(recent["__g10_target__"])
            horizon_ev[horizon] = value
            out.loc[current.index, f"g10_global_ev_{horizon}d"] = np.float32(value)
        out.loc[current.index, "g10_global_ev_3m21_delta"] = np.float32(horizon_ev[3] - horizon_ev[21])
        recent21 = history_all.loc[history_all["__g10_day__"].ge(day - pd.Timedelta(days=21))]
        global21 = horizon_ev[21]
        state_statistics(recent21, "__g10_score_bin__", current, prefix="g10_score", global_ev=global21)
        state_statistics(recent21, "__g10_base_bin__", current, prefix="g10_base", global_ev=global21)
        state_statistics(recent21, "__g10_t6_bin__", current, prefix="g10_t6", global_ev=global21)
        state_statistics(recent21, "__g10_t9_bin__", current, prefix="g10_t9", global_ev=global21)
    return out.drop(columns=["__g10_day__", "__g10_available_day__", "__g10_target__", "__g10_score_bin__", "__g10_base_bin__", "__g10_t6_bin__", "__g10_t9_bin__"])


def _fit_mc1_with_capacity(
    train: pd.DataFrame,
    features: Sequence[str],
    capacity: MC1Capacity,
):
    """Exact MC1 fit contract with only predeclared tree capacity varied.

    Sampling, outcome clipping, score-band curve, shrinkage, seed, and the
    later recent-shift calculation remain identical to the frozen MC1
    implementation.  This lets C0/C1/C2/C3 isolate capacity rather than
    accidentally retuning the mapping construction itself.
    """
    fit = train.copy()
    fit["score_band"] = mc1._score_bands(fit)
    fit["day"] = fit["__decision_ts__"].dt.normalize()
    selected: list[pd.DataFrame] = []
    for _, part in fit.groupby("day", sort=True):
        part = part.sort_values(
            ["__decision_ts__", "final_score", "candidate_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        selected.append(pd.concat((part.head(50), part.iloc[50:].sample(min(250, max(0, len(part) - 50)), random_state=SEED))))
    work = pd.concat(selected, ignore_index=True)
    target = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    low, high = target.quantile([.02, .98])
    work["target"] = target.clip(low, high)
    if len(work) > 50_000:
        work = work.sample(50_000, random_state=SEED)
    medians = work.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
    matrix = work.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").fillna(medians)
    model = HistGradientBoostingRegressor(
        max_depth=capacity.max_depth,
        max_leaf_nodes=capacity.max_leaf_nodes,
        max_iter=capacity.max_iter,
        learning_rate=capacity.learning_rate,
        l2_regularization=capacity.l2_regularization,
        min_samples_leaf=capacity.min_samples_leaf,
        random_state=SEED,
    ).fit(matrix, work["target"])
    global_mean = mc1._robust_mean(work["target"])
    curve = np.full(10, global_mean, dtype=float)
    for band, part in work.groupby("score_band", sort=True):
        mean, std, count = float(part["target"].mean()), max(float(part["target"].std(ddof=0)), 1.0), len(part)
        precision, prior = count / (std * std + 1.0), 80.0 / (250.0 ** 2)
        curve[int(band)] = (precision * mean + prior * global_mean) / (precision + prior)
    curve = -IsotonicRegression(increasing=True).fit_transform(np.arange(10), -curve)
    return model, medians, curve, (float(low), float(high))


def _predictions(
    frame: pd.DataFrame,
    features: list[str],
    family: str,
    out: Path,
    evaluation_months: tuple[pd.Timestamp, ...],
    capacity: MC1Capacity,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the unchanged strict-prequential scorer under a local capacity fit."""
    old_months = mc1.SCORE_MONTHS
    original_fit = mc1._fit_mc1
    try:
        mc1.SCORE_MONTHS = evaluation_months
        mc1._fit_mc1 = lambda train, fields: _fit_mc1_with_capacity(train, fields, capacity)
        return mc1._predictions(frame, features, family, out)
    finally:
        mc1.SCORE_MONTHS = old_months
        mc1._fit_mc1 = original_fit


def _run_portfolio(frame: pd.DataFrame, label: str, out: Path, threshold: float) -> dict[str, object]:
    old = parent.MC1_THRESHOLD_BPS
    try:
        parent.MC1_THRESHOLD_BPS = float(threshold)
        return parent._portfolio_metrics(frame, label, "mayjul_2026", out)
    finally:
        parent.MC1_THRESHOLD_BPS = old


def _score_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Outcome-joined-only score diagnostics for Stage 1 selection."""
    valid = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    ].copy()
    valid["month"] = valid["__decision_ts__"].dt.strftime("%Y-%m")
    records: list[dict[str, object]] = []
    fractions = (.01, .02, .05, .10)
    periods: list[tuple[str, pd.DataFrame]] = [("all", valid)]
    periods.extend((month, part) for month, part in valid.groupby("month", sort=True))
    for period, part in periods:
        for scope in ("global", "timestamp"):
            for fraction in fractions:
                if scope == "global":
                    n = max(1, int(np.ceil(len(part) * fraction)))
                    selected = part.nlargest(n, "final_score", keep="all")
                else:
                    selected = pd.concat([
                        group.nlargest(max(1, int(np.ceil(len(group) * fraction))), "final_score", keep="all")
                        for _, group in part.groupby("__decision_ts__", sort=False)
                    ], ignore_index=True)
                records.append({
                    "period": period, "scope": scope, "fraction": fraction, "metric": "net_ev_bps_per_trade",
                    "value": float(pd.to_numeric(selected["policy_net_bps"], errors="coerce").mean()), "rows": int(len(selected)),
                })
        records.append({
            "period": period, "scope": "global", "fraction": np.nan, "metric": "rank_ic_spearman",
            "value": float(part["final_score"].corr(part["policy_net_bps"], method="spearman")), "rows": int(len(part)),
        })
        ts_ic = [group["final_score"].corr(group["policy_net_bps"], method="spearman") for _, group in part.groupby("__decision_ts__", sort=False) if len(group) > 2]
        records.append({
            "period": period, "scope": "timestamp", "fraction": np.nan, "metric": "rank_ic_spearman",
            "value": float(np.nanmean(ts_ic)) if ts_ic else np.nan, "rows": int(len(part)),
        })
    return pd.DataFrame(records)


def _admission_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    """Current, BCF, and dual calibration around the MC1 admission frontier."""
    valid = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    ].copy()
    schemes = {
        "current": pd.to_numeric(valid["current_mc1_expected_bps"], errors="coerce"),
        "bcf": pd.to_numeric(valid["bcf_mc1_expected_bps"], errors="coerce"),
        "dual_min": np.minimum(pd.to_numeric(valid["current_mc1_expected_bps"], errors="coerce"), pd.to_numeric(valid["bcf_mc1_expected_bps"], errors="coerce")),
    }
    edges = np.asarray([-np.inf, 0., 25., 50., 60., 75., 100., 150., 250., np.inf])
    labels = ("<0", "0-25", "25-50", "50-60", "60-75", "75-100", "100-150", "150-250", "250+")
    records: list[dict[str, object]] = []
    actual = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
    for name, values in schemes.items():
        bucket = pd.cut(values, edges, labels=labels, right=False)
        for label, part in valid.assign(_pred=values, _bucket=bucket, _actual=actual).groupby("_bucket", observed=False, sort=False):
            records.append({
                "mapper": name, "band": str(label), "rows": int(len(part)),
                "predicted_ev_bps": float(part["_pred"].mean()) if len(part) else np.nan,
                "realised_ev_bps": float(part["_actual"].mean()) if len(part) else np.nan,
                "calibration_bias_bps": float((part["_pred"] - part["_actual"]).mean()) if len(part) else np.nan,
                "severe_loss_rate": float((part["_actual"] <= -200.0).mean()) if len(part) else np.nan,
            })
    return pd.DataFrame(records)


def run(
    *, p2_root: Path, component_root: Path, t6_root: Path, t9_root: Path, policy_path: Path, out: Path,
    score_contract: str, geometry_blocks: tuple[str, ...], ledger_months: tuple[pd.Timestamp, ...],
    evaluation_months: tuple[pd.Timestamp, ...], thresholds: tuple[float, ...], t9_visible: bool,
    mc1_capacity: MC1Capacity = DEFAULT_MC1_CAPACITY,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    if score_contract not in SCORE_CONTRACTS:
        raise ValueError(f"unknown score contract: {score_contract}")
    if min(evaluation_months) - pd.DateOffset(months=6) < min(ledger_months):
        raise ValueError("evaluation requires six complete preceding ledger months")
    out.mkdir(parents=True)
    policy = _load_policy(policy_path)
    features: list[str] | None = None
    predictions: dict[str, pd.DataFrame] = {}
    audits: list[pd.DataFrame] = []
    source_panels: list[pd.DataFrame] = []
    for family in ("current", "bcf"):
        target_free = _load_family(
            family=family, p2_root=p2_root, component_root=component_root, t6_root=t6_root, t9_root=t9_root,
            ledger_months=ledger_months, weights=SCORE_CONTRACTS[score_contract],
        )
        target_free = _add_geometry(target_free)
        leaked = PROHIBITED.intersection(target_free.columns)
        if leaked:
            raise AssertionError(f"{family}: target-free panel contains outcomes {sorted(leaked)}")
        target_free.to_parquet(out / f"{family}_target_free_score_panel.parquet", index=False, compression="zstd")
        source_panels.append(target_free.loc[:, ["candidate_id", "__decision_ts__", "final_score"]].assign(family=family))
        candidate_features = _geometry_features(target_free, geometry_blocks, t9_visible=t9_visible)
        panel = target_free.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        if "G10" in geometry_blocks:
            panel = _recent_calibration_block(panel)
        missing = sorted(set(candidate_features) - set(panel.columns))
        if missing:
            raise AssertionError(f"{family}: missing geometry inputs {missing}")
        if features is None:
            features = candidate_features
        elif features != candidate_features:
            raise AssertionError("Current and BCF MC1 feature schemas differ")
        if family == "current":
            _score_diagnostics(panel).to_parquet(out / "score_diagnostics.parquet", index=False, compression="zstd")
        prediction, audit = _predictions(panel, candidate_features, family, out, evaluation_months, mc1_capacity)
        predictions[family] = prediction
        audits.append(audit)
    if features is None:
        raise AssertionError("no MC1 features")
    current, bcf = predictions["current"], predictions["bcf"]
    challenger = mc1._combine(current, bcf)
    start, end = min(evaluation_months), max(evaluation_months) + pd.offsets.MonthBegin(1)
    challenger = challenger.loc[challenger["__decision_ts__"].ge(start) & challenger["__decision_ts__"].lt(end)].copy()
    if challenger.empty:
        raise AssertionError("empty MC1 challenger output")
    pd.concat(source_panels, ignore_index=True).to_parquet(out / "target_free_score_identity_audit.parquet", index=False, compression="zstd")
    challenger.to_parquet(out / "dual_mc1_predictions.parquet", index=False, compression="zstd")
    pd.concat(audits, ignore_index=True).to_parquet(out / "mc1_fit_audit.parquet", index=False, compression="zstd")
    _admission_calibration(challenger).to_parquet(out / "mc1_admission_calibration.parquet", index=False, compression="zstd")
    metrics = []
    for threshold in thresholds:
        item = _run_portfolio(challenger, f"{score_contract}_{'+'.join(geometry_blocks) or 'M0'}_{int(threshold)}", out, threshold)
        item["threshold_bps"] = float(threshold)
        metrics.append(item)
    pd.DataFrame(metrics).to_parquet(out / "portfolio_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline challenger only; live bundles, processes, and policies untouched",
        "score_contract": score_contract,
        "score_weights": {"base": SCORE_CONTRACTS[score_contract][0], "t6": SCORE_CONTRACTS[score_contract][1], "t9": SCORE_CONTRACTS[score_contract][2]},
        "geometry_blocks": list(geometry_blocks), "mc1_features": features,
        "t9_visible_to_mc1": bool(t9_visible),
        "mc1_capacity": asdict(mc1_capacity),
        "ledger_months": [f"{m:%Y-%m}" for m in ledger_months],
        "evaluation_months": [f"{m:%Y-%m}" for m in evaluation_months],
        "thresholds_bps": list(thresholds),
        "causality": {
            "score_panels": "target-free enhanced-base/T6/T9 strict-OOF receipts only",
            "route": "timestamp-local enhanced-base top 30 percent reconstructed before MC1",
            "maps": "Current and BCF MC1 fitted separately on six complete prior months with resolved rich-policy labels only",
            "admission": "both independently fitted MC1 maps must clear threshold",
            "portfolio": "unchanged chronological constrained auction",
        },
        "source_hashes": {
            # Immutable upstream receipts already seal their full input trees.
            # Hash their top-level manifests rather than rereading every large
            # historical parquet in every score-arm trial.
            "p2_manifest": _sha256(p2_root / "run_manifest.json"),
            "component_manifest": _sha256(component_root / "targetfree_manifest.json"),
            "t6_manifest": _sha256(t6_root / "run_manifest.json"),
            "t9_manifest": _sha256(t9_root / "run_manifest.json"),
            "policy": _sha256(policy_path),
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-root", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--t6-root", type=Path, required=True)
    parser.add_argument("--t9-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-contract", choices=tuple(SCORE_CONTRACTS), required=True)
    parser.add_argument("--geometry-blocks", default="", help="comma-separated G1..G9; empty is M0")
    parser.add_argument("--ledger-months")
    parser.add_argument("--evaluation-months")
    parser.add_argument("--thresholds", default="50,60,70,80,100")
    parser.add_argument("--hide-t9-from-mc1", action="store_true", help="role-separation control: do not expose T9 to either MC1 mapper")
    parser.add_argument("--mc1-max-depth", type=int, default=DEFAULT_MC1_CAPACITY.max_depth)
    parser.add_argument("--mc1-max-leaf-nodes", type=int, default=DEFAULT_MC1_CAPACITY.max_leaf_nodes)
    parser.add_argument("--mc1-max-iter", type=int, default=DEFAULT_MC1_CAPACITY.max_iter)
    parser.add_argument("--mc1-learning-rate", type=float, default=DEFAULT_MC1_CAPACITY.learning_rate)
    parser.add_argument("--mc1-l2-regularization", type=float, default=DEFAULT_MC1_CAPACITY.l2_regularization)
    parser.add_argument("--mc1-min-samples-leaf", type=int, default=DEFAULT_MC1_CAPACITY.min_samples_leaf)
    args = parser.parse_args()
    blocks = tuple(value for value in args.geometry_blocks.split(",") if value)
    invalid = sorted(set(blocks) - {f"G{i}" for i in range(1, 11)})
    if invalid:
        parser.error(f"unsupported geometry blocks: {invalid}")
    if args.mc1_max_depth < 1 or args.mc1_max_leaf_nodes < 2 or args.mc1_max_iter < 1 or args.mc1_min_samples_leaf < 1:
        parser.error("MC1 capacity arguments must be positive and allow at least two leaves")
    run(
        p2_root=args.p2_root, component_root=args.component_root, t6_root=args.t6_root, t9_root=args.t9_root,
        policy_path=args.policy_path, out=args.out, score_contract=args.score_contract, geometry_blocks=blocks,
        ledger_months=_parse_months(args.ledger_months, DEFAULT_LEDGER),
        evaluation_months=_parse_months(args.evaluation_months, DEFAULT_EVALUATION),
        thresholds=tuple(float(value) for value in args.thresholds.split(",") if value),
        t9_visible=not args.hide_t9_from_mc1,
        mc1_capacity=MC1Capacity(
            max_depth=args.mc1_max_depth,
            max_leaf_nodes=args.mc1_max_leaf_nodes,
            max_iter=args.mc1_max_iter,
            learning_rate=args.mc1_learning_rate,
            l2_regularization=args.mc1_l2_regularization,
            min_samples_leaf=args.mc1_min_samples_leaf,
        ),
    )


if __name__ == "__main__":
    main()
