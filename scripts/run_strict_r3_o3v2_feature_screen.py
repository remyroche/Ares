#!/usr/bin/env python3
"""Causal F1--F6 information screen for the O3-v2 correction layer.

The program is deliberately a *selection* stage, not an inference scorer.  It
creates a target-free feature panel first, seals it, and only then joins
already-resolved policy outcomes to calculate training-only relevance metrics.
No feature is selected from an outcome-derived held score, and no MDA is run:
the user has assigned MDA to a separate pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_o3v2_target_funnel as target  # noqa: E402


SCHEMA = "strict_r3_o3v2_feature_screen_v3"
SEED = 1729
MIN_COVERAGE = 0.90
MAX_MI_ROWS = 12_000
TOP_PER_FAMILY = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for child in paths:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _month_path(root: Path, month: pd.Timestamp) -> Path:
    return root / f"month={month:%Y-%m}" / "scores_features.parquet"


def _parent_path(root: Path, family: str, month: pd.Timestamp) -> Path:
    return root / "target_free_scores" / family / f"month={month:%Y-%m}.parquet"


def _o3_path(roots: tuple[Path, ...], arm: str, month: pd.Timestamp) -> Path:
    candidates = [root / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet" for root in roots]
    found = [path for path in candidates if path.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"{arm} {month:%Y-%m}: expected exactly one O3 source, found {found}")
    return found[0]


def _safe_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average")


def _query_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact-timestamp geometry from target-free base ranks only."""
    out = pd.DataFrame(index=frame.index)
    rank = pd.to_numeric(frame["base_rank_ts"], errors="coerce")
    score = pd.to_numeric(frame["enhanced_base_bps"], errors="coerce")
    groups = frame.groupby("__decision_ts__", sort=False)["base_rank_ts"]
    out["f2_query_count"] = groups.transform("size").astype(float)
    out["f2_query_std"] = groups.transform("std").astype(float)
    out["f2_query_iqr"] = groups.transform(lambda x: float(np.nanquantile(x, .75) - np.nanquantile(x, .25))).astype(float)
    out["f2_gap_to_top"] = groups.transform("max").astype(float) - rank
    out["f2_gap_to_median"] = rank - groups.transform("median").astype(float)
    # Candidate-local quantities are calculated within the exact causal
    # timestamp.  Thresholds are fixed ex ante, never selected from outcomes.
    density = np.full(len(frame), np.nan, dtype=float)
    entropy = np.full(len(frame), np.nan, dtype=float)
    concentration = np.full(len(frame), np.nan, dtype=float)
    gap_above = np.full(len(frame), np.nan, dtype=float)
    gap_below = np.full(len(frame), np.nan, dtype=float)
    top1_top2 = np.full(len(frame), np.nan, dtype=float)
    top5_spread = np.full(len(frame), np.nan, dtype=float)
    route_cutoff_gap = np.full(len(frame), np.nan, dtype=float)
    within_10 = np.full(len(frame), np.nan, dtype=float)
    within_25 = np.full(len(frame), np.nan, dtype=float)
    within_50 = np.full(len(frame), np.nan, dtype=float)
    for _stamp, index in frame.groupby("__decision_ts__", sort=False).groups.items():
        idx = np.asarray(index, dtype=np.int64)
        original_rank = rank.loc[idx].to_numpy(float)
        original_score = score.loc[idx].to_numpy(float)
        values = original_rank[np.isfinite(original_rank)]
        if values.size == 0:
            continue
        sorted_values = np.sort(values)
        density[idx] = [float(np.mean(np.abs(sorted_values - value) <= .05)) if np.isfinite(value) else np.nan for value in original_rank]
        hist, _ = np.histogram(values, bins=np.linspace(0.0, 1.0, 11))
        p = hist[hist > 0] / max(hist.sum(), 1)
        entropy[idx] = -float(np.sum(p * np.log(p))) / np.log(10.0) if p.size else np.nan
        concentration[idx] = float(np.mean(values >= .90))
        order = np.argsort(-np.nan_to_num(original_score, nan=-np.inf), kind="stable")
        ordered_score = original_score[order]
        finite_score = ordered_score[np.isfinite(ordered_score)]
        if finite_score.size >= 2:
            top1_top2[idx] = finite_score[0] - finite_score[1]
        if finite_score.size >= 5:
            top5_spread[idx] = finite_score[0] - finite_score[4]
        if finite_score.size:
            cutoff_pos = min(finite_score.size - 1, max(0, int(np.ceil(.30 * finite_score.size)) - 1))
            route_cutoff_gap[idx] = original_score - finite_score[cutoff_pos]
            sorted_ascending = np.sort(finite_score)
            for local_pos, value in enumerate(original_score):
                if not np.isfinite(value):
                    continue
                insertion = int(np.searchsorted(sorted_ascending, value, side="left"))
                lower = sorted_ascending[insertion - 1] if insertion > 0 else np.nan
                upper = sorted_ascending[insertion] if insertion < len(sorted_ascending) else np.nan
                # Values are expected economics in bps: use score gaps, not
                # labels, to quantify locally contestable candidates.
                gap_below[idx[local_pos]] = value - lower if np.isfinite(lower) else np.nan
                gap_above[idx[local_pos]] = upper - value if np.isfinite(upper) else np.nan
                distance = np.abs(finite_score - value)
                within_10[idx[local_pos]] = float(np.sum(distance <= 10.0))
                within_25[idx[local_pos]] = float(np.sum(distance <= 25.0))
                within_50[idx[local_pos]] = float(np.sum(distance <= 50.0))
    out["f2_local_rank_density_005"] = density
    out["f2_rank_entropy"] = entropy
    out["f2_top_decile_concentration"] = concentration
    out["f2_top1_top2_bps_gap"] = top1_top2
    out["f2_top5_bps_spread"] = top5_spread
    out["f2_gap_to_rank_above_bps"] = gap_above
    out["f2_gap_to_rank_below_bps"] = gap_below
    out["f2_gap_to_route_cutoff_bps"] = route_cutoff_gap
    out["f2_candidates_within_10bps"] = within_10
    out["f2_candidates_within_25bps"] = within_25
    out["f2_candidates_within_50bps"] = within_50
    return out


def _recent_error_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolved-only multi-horizon base-error telemetry with shrinkage.

    Every output is a decision-time feature.  At a decision on day ``d`` the
    most recent usable outcome is one whose availability clock is strictly
    before ``d``; that deliberately conservative daily boundary eliminates
    same-day/outcome-clock ambiguity while retaining the requested 7/21/63d
    horizons.  Local base-decile estimates shrink through a five-decile
    region to the long-only global prior and always expose support.
    """
    required = {"policy_path_valid", "policy_net_bps", "policy_label_available_ts", "base_anchor_bps"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"recent-error features require {sorted(missing)}")
    work = frame.copy().reset_index(drop=True)
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work["policy_label_available_ts"] = pd.to_datetime(work["policy_label_available_ts"], utc=True, errors="coerce")
    work["__base_decile__"] = np.minimum(9, np.maximum(0, np.floor(pd.to_numeric(work["base_rank_ts"], errors="coerce").fillna(.5) * 10))).astype(int)
    valid = work["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(work["policy_net_bps"], errors="coerce"))
    work["__residual__"] = pd.to_numeric(work["policy_net_bps"], errors="coerce") - pd.to_numeric(work["base_anchor_bps"], errors="coerce")
    ranked = work.loc[valid, ["__decision_ts__", "policy_net_bps", "base_rank_ts"]].copy()
    ranked["__realised_rank__"] = ranked.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(pct=True, method="average")
    work["__rank_error__"] = np.nan
    work.loc[ranked.index, "__rank_error__"] = ranked["__realised_rank__"].to_numpy(float) - pd.to_numeric(ranked["base_rank_ts"], errors="coerce").to_numpy(float)
    labels = work.loc[
        valid & work["policy_label_available_ts"].notna() & np.isfinite(work["__residual__"]) & np.isfinite(work["__rank_error__"]),
        ["policy_label_available_ts", "__residual__", "__rank_error__", "__base_decile__"],
    ].copy()
    labels["__available_day__"] = labels["policy_label_available_ts"].dt.normalize()
    work["__decision_day__"] = work["__decision_ts__"].dt.normalize()

    def stats(sample: pd.DataFrame) -> dict[str, float]:
        if sample.empty:
            return {key: np.nan for key in ("median", "trimmed", "q10", "q25", "q75", "q90", "mad", "severe", "rank_error", "inversion")}
        residual = sample["__residual__"].to_numpy(float)
        rank_error = sample["__rank_error__"].to_numpy(float)
        ordered = np.sort(residual)
        trim = int(np.floor(.10 * len(ordered)))
        trimmed = ordered[trim:len(ordered) - trim] if len(ordered) > 2 * trim else ordered
        median = float(np.median(residual))
        return {
            "median": median,
            "trimmed": float(np.mean(trimmed)),
            "q10": float(np.quantile(residual, .10)), "q25": float(np.quantile(residual, .25)),
            "q75": float(np.quantile(residual, .75)), "q90": float(np.quantile(residual, .90)),
            "mad": float(np.median(np.abs(residual - median))),
            "severe": float(np.mean(residual <= -200.0)),
            "rank_error": float(np.mean(rank_error)),
            "inversion": float(np.mean(rank_error <= -.20)),
        }

    metric_names = ("median", "trimmed", "q10", "q25", "q75", "q90", "mad", "severe", "rank_error", "inversion")
    day_rows: list[dict[str, float]] = []
    days = sorted(pd.DatetimeIndex(work["__decision_day__"].drop_duplicates()).tz_convert("UTC"))
    for day in days:
        rows_for_day = work.index[work["__decision_day__"].eq(day)].to_numpy(dtype=np.int64)
        deciles = work.loc[rows_for_day, "__base_decile__"].to_numpy(int)
        output: dict[str, np.ndarray] = {"__row_id__": rows_for_day.astype(float)}
        for horizon in (7, 21, 63):
            past = labels.loc[
                labels["__available_day__"].lt(day) & labels["__available_day__"].ge(day - pd.Timedelta(days=horizon))
            ]
            global_stats = stats(past)
            global_n = float(len(past))
            local_stats = {decile: stats(past.loc[past["__base_decile__"].eq(decile)]) for decile in range(10)}
            local_n = {decile: float(np.sum(past["__base_decile__"].eq(decile))) for decile in range(10)}
            region_stats = {region: stats(past.loc[(past["__base_decile__"] // 2).eq(region)]) for region in range(5)}
            region_n = {region: float(np.sum((past["__base_decile__"] // 2).eq(region))) for region in range(5)}
            n = np.asarray([local_n[int(decile)] for decile in deciles], dtype=float)
            output[f"f3_support_{horizon}d"] = n
            output[f"f3_effective_support_{horizon}d"] = n
            for name in metric_names:
                values = np.empty(len(deciles), dtype=float)
                for pos, decile in enumerate(deciles):
                    local = local_stats[int(decile)][name]
                    region = region_stats[int(decile) // 2][name]
                    global_value = global_stats[name]
                    if not np.isfinite(global_value):
                        global_value = 0.0
                    if not np.isfinite(region):
                        region = global_value
                    if not np.isfinite(local):
                        local = region
                    # Two-stage hierarchical shrinkage.  The pseudo-counts
                    # are fixed ex ante and do not depend on held economics.
                    shrunk_region = (region_n[int(decile) // 2] * region + 40.0 * global_value) / (region_n[int(decile) // 2] + 40.0)
                    values[pos] = (local_n[int(decile)] * local + 20.0 * shrunk_region) / (local_n[int(decile)] + 20.0)
                output[f"f3_{name}_{horizon}d"] = values
        for name in ("trimmed", "severe", "rank_error", "inversion"):
            output[f"f3_{name}_drift_7v63"] = output[f"f3_{name}_7d"] - output[f"f3_{name}_63d"]
            output[f"f3_{name}_drift_21v63"] = output[f"f3_{name}_21d"] - output[f"f3_{name}_63d"]
        day_rows.append(pd.DataFrame(output))
    telemetry = pd.concat(day_rows, ignore_index=True).set_index("__row_id__").reindex(np.arange(len(work)))
    telemetry.index = frame.index
    return telemetry.astype(np.float32)


def _add_families(
    base: pd.DataFrame,
    current: pd.DataFrame,
    bcf: pd.DataFrame,
    o3: pd.DataFrame | None = None,
) -> pd.DataFrame:
    identity = ["candidate_id", "__decision_ts__", "side_name"]
    # Prefix every non-identity score-family field before joining.  The base
    # panel and parent receipts intentionally share routing columns, so pandas
    # suffixing is not a stable lineage contract here.
    current = current.rename(columns={field: f"current__{field}" for field in current.columns if field not in identity})
    bcf = bcf.rename(columns={field: f"bcf__{field}" for field in bcf.columns if field not in identity})
    frame = base.merge(current, on=identity, how="inner", validate="one_to_one")
    frame = frame.merge(bcf, on=identity, how="inner", validate="one_to_one")
    # The legacy O3 receipt is optional for the stable-specialist contract.
    # Leaving it out makes the F5 provenance family available from the first
    # month where both parent score families exist, rather than forcing a
    # circular dependency on a later O3 correction score.  Current/BCF
    # provenance remains target-free either way.
    if o3 is not None:
        o3 = o3.rename(columns={field: f"o3__{field}" for field in o3.columns if field != "candidate_id"})
        frame = frame.merge(o3, on="candidate_id", how="inner", validate="one_to_one")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("duplicate identity after target-free family merge")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    result = frame.loc[:, identity].copy()
    extra: dict[str, object] = {}
    # F1: the full causal three-score disagreement geometry.  ``base_bps``
    # is B0; E and T are the existing efficiency and timing score components.
    f1 = ("base_bps", "efficiency_bps", "timing_bps", "enhanced_base_bps", "base_rank_ts", "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std")
    for field in f1:
        extra[f"f1_{field}"] = pd.to_numeric(frame[field], errors="coerce").to_numpy()
    coords = frame.loc[:, ["base_bps", "efficiency_bps", "timing_bps"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    extra["f1_coord_min"] = np.nanmin(coords, axis=1)
    extra["f1_coord_max"] = np.nanmax(coords, axis=1)
    extra["f1_coord_median"] = np.nanmedian(coords, axis=1)
    extra["f1_coord_std"] = np.nanstd(coords, axis=1)
    extra["f1_coord_mad"] = np.nanmedian(np.abs(coords - np.nanmedian(coords, axis=1, keepdims=True)), axis=1)
    extra["f1_coord_range"] = extra["f1_coord_max"] - extra["f1_coord_min"]
    sorted_coords = np.sort(coords, axis=1)
    extra["f1_strongest_minus_second"] = sorted_coords[:, 2] - sorted_coords[:, 1]
    extra["f1_b0_minus_mean_et"] = coords[:, 0] - np.nanmean(coords[:, 1:], axis=1)
    extra["f1_min_et_minus_b0"] = np.nanmin(coords[:, 1:], axis=1) - coords[:, 0]
    # Causal component ranks and fixed high-conviction geometry.
    ranks = np.zeros_like(coords, dtype=float)
    for column in range(3):
        ranks[:, column] = frame.assign(__coord__=coords[:, column]).groupby("__decision_ts__", sort=False)["__coord__"].rank(pct=True, method="average").to_numpy(float)
    extra["f1_fraction_components_rank_ge90"] = np.nanmean(ranks >= .90, axis=1)
    extra["f1_fraction_components_rank_ge95"] = np.nanmean(ranks >= .95, axis=1)
    extra["f1_component_rank_order"] = (np.argsort(np.argsort(coords, axis=1), axis=1)[:, 0] * 9 + np.argsort(np.argsort(coords, axis=1), axis=1)[:, 1] * 3 + np.argsort(np.argsort(coords, axis=1), axis=1)[:, 2]).astype(float)
    b0_high, e_high, t_high = ranks[:, 0] >= .90, ranks[:, 1] >= .90, ranks[:, 2] >= .90
    geometry = np.select(
        [b0_high & e_high & t_high, b0_high & ~e_high & ~t_high, ~b0_high & e_high & t_high, e_high & ~t_high, t_high & ~e_high],
        [1., 2., 3., 4., 5.], default=0.,
    )
    extra["f1_disagreement_geometry_code"] = geometry
    # F4: compact causal state/transition candidates.  This must not become a
    # second generic base-feature dump: its job is to explain shifts in the
    # base -> realised-policy mapping.  We therefore retain existing frozen
    # fields only when their names establish a transition, acceleration,
    # breadth/dependence, liquidity-stress, or structural-state role.  Static
    # levels are deliberately excluded unless they are a named structural
    # state (``state_*``/``eig_*``), and the conditional screen below still
    # decides whether any such field earns a place in a head contract.
    protected = set(identity) | set(f1) | {"enhanced_base_routed"}
    f4_tokens = (
        "accel", "chg", "change", "delta", "recovery", "drawdown",
        "flush", "climax", "rebound", "rebuild", "stress", "surprise",
        "asymmetry", "bars_", "bars_to", "pct_", "breadth", "dispersion",
        "corr", "spectral", "eig_", "effective_rank", "liquid", "depth",
        "amihud", "spread", "funding", "oi_", "leverage", "trend",
        "momentum", "memory", "wick", "resistance", "exhaustion",
    )
    for field in base.columns:
        if field in protected:
            continue
        lower = field.lower()
        if not lower.startswith(("state_", "cross_asset_", "xasset_", "mkt_", "market_", "pct_", "xs_", "q_")) and not any(token in lower for token in f4_tokens):
            continue
        if not any(token in lower for token in f4_tokens) and not lower.startswith(("state_", "cross_asset_")):
            continue
        # ``frame`` is the routed identity intersection; indexing the
        # unmerged base panel here would misalign values after routing.
        values = pd.to_numeric(frame[field], errors="coerce")
        if values.notna().any():
            extra[f"f4_{field}"] = values.to_numpy()
    # Parent correction provenance.  Both score families are target-free and
    # are joined before policy labels enter the process.
    parent_fields = ("base_rank42", "base_anchor_bps", "conditional_consensus_rank", "ordinary_shadow_consensus_rank", "upstream", "correctness_rank", "head_agreement_std", "final_score")
    for field in parent_fields:
        current_name = f"current__{field}"
        bcf_name = f"bcf__{field}"
        if current_name in frame.columns and bcf_name in frame.columns:
            current_value = pd.to_numeric(frame[current_name], errors="coerce").to_numpy()
            bcf_value = pd.to_numeric(frame[bcf_name], errors="coerce").to_numpy()
            extra[f"f5_current_{field}"] = current_value
            extra[f"f5_bcf_{field}"] = bcf_value
            extra[f"f5_delta_current_minus_bcf_{field}"] = current_value - bcf_value
            # Explicitly retain the two score families' correction magnitude
            # relative to the same causal enhanced-base estimate.  These are
            # target-free provenance features, not held outcomes.
            if field == "base_anchor_bps":
                enhanced = pd.to_numeric(frame["enhanced_base_bps"], errors="coerce").to_numpy()
                extra["f5_current_anchor_minus_enhanced_base"] = current_value - enhanced
                extra["f5_bcf_anchor_minus_enhanced_base"] = bcf_value - enhanced
    for field in [name for name in frame.columns if name.startswith("current__head__")]:
        base_name = field.removeprefix("current__")
        bcf_name = f"bcf__{base_name}"
        if bcf_name in frame.columns:
            extra[f"f5_delta_{base_name}"] = (pd.to_numeric(frame[field], errors="coerce") - pd.to_numeric(frame[bcf_name], errors="coerce")).to_numpy()
    for field in ("conditional_consensus_rank", "o3v2_rank_75_25", "head_agreement_std"):
        o3_name = f"o3__{field}"
        if o3_name in frame.columns:
            extra[f"f5_o3_{field}"] = pd.to_numeric(frame[o3_name], errors="coerce").to_numpy()
    # F6 is the remaining numeric, target-free parent-meta universe.  The
    # repository contains frozen JSON contracts rather than a single
    # ``config.py`` registry, so discover this family from the two causal
    # parent score receipts and retain only actual numeric values.  This makes
    # the source/feature inventory auditable without silently assuming the
    # hand-curated F1--F5 list is exhaustive.
    used_parent = {
        name.removeprefix("f5_current_") for name in extra if name.startswith("f5_current_")
    } | {
        name.removeprefix("f5_bcf_") for name in extra if name.startswith("f5_bcf_")
    }
    for source_prefix in ("current__", "bcf__"):
        for field in frame.columns:
            if not field.startswith(source_prefix):
                continue
            raw_name = field.removeprefix(source_prefix)
            if raw_name in used_parent or raw_name in {"enhanced_base_routed"}:
                continue
            values = pd.to_numeric(frame[field], errors="coerce")
            if values.notna().any():
                extra[f"f6_{source_prefix.removesuffix('__')}_{raw_name}"] = values.to_numpy()
    # Clock fields are causal, compact, and are deliberately kept distinct
    # from static config IDs, which would not generalise across bundles.
    hour = frame["__decision_ts__"].dt.hour.astype(float)
    dow = frame["__decision_ts__"].dt.dayofweek.astype(float)
    extra["f6_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    extra["f6_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    extra["f6_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    extra["f6_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    query = _query_features(frame)
    result = pd.concat([result, pd.DataFrame(extra, index=result.index), query], axis=1)
    return result


def _load_parent(root: Path, family: str, month: pd.Timestamp) -> pd.DataFrame:
    path = _parent_path(root, family, month)
    frame = pd.read_parquet(path)
    prohibited = target.PROHIBITED_SCORE_COLUMNS.intersection(frame.columns)
    if prohibited:
        raise AssertionError(f"{path}: outcome field present in target-free parent receipt: {sorted(prohibited)}")
    required = {"candidate_id", "__decision_ts__", "side_name"}
    if missing := required - set(frame.columns):
        raise KeyError(f"{path}: missing {sorted(missing)}")
    return frame


def _load_o3(roots: tuple[Path, ...], arm: str, month: pd.Timestamp) -> pd.DataFrame:
    path = _o3_path(roots, arm, month)
    frame = pd.read_parquet(path)
    prohibited = target.PROHIBITED_SCORE_COLUMNS.intersection(frame.columns)
    if prohibited:
        raise AssertionError(f"{path}: outcome field present in target-free O3 receipt: {sorted(prohibited)}")
    return frame


def _build_target_free(
    base_root: Path,
    parent_root: Path,
    o3_roots: tuple[Path, ...],
    o3_arm: str | None,
    months: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    pieces = []
    for month in months:
        path = _month_path(base_root, month)
        if not path.exists():
            raise FileNotFoundError(path)
        base = pd.read_parquet(path)
        current = _load_parent(parent_root, "current", month)
        bcf = _load_parent(parent_root, "bcf", month)
        o3 = _load_o3(o3_roots, o3_arm, month) if o3_roots and o3_arm else None
        pieces.append(_add_families(base, current, bcf, o3))
    output = pd.concat(pieces, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("duplicate IDs across target-free screen panel")
    return output


def _add_recent_telemetry(
    panel: pd.DataFrame,
    policy: pd.DataFrame,
    parent_root: Path,
    months: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    # Parent current receipts provide the frozen prequential base anchor.  It
    # is used only to define past residual telemetry, never as a label input.
    parent_pieces = []
    for month in months:
        parent_pieces.append(_load_parent(parent_root, "current", month).loc[:, ["candidate_id", "base_anchor_bps"]])
    anchor = pd.concat(parent_pieces, ignore_index=True)
    joined = panel.loc[:, ["candidate_id", "__decision_ts__", "f1_base_rank_ts"]].rename(columns={"f1_base_rank_ts": "base_rank_ts"}).merge(anchor, on="candidate_id", how="left", validate="one_to_one")
    joined = joined.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    telemetry = _recent_error_features(joined)
    return pd.concat([panel.reset_index(drop=True), telemetry.reset_index(drop=True)], axis=1)


def _near_tie_pairs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute adjacent close-base-rank pairs once for every feature screen."""
    base = pd.to_numeric(frame["f1_base_rank_ts"], errors="coerce").to_numpy(float)
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    for _stamp, index in frame.groupby("__decision_ts__", sort=False).groups.items():
        idx = np.asarray(index, dtype=np.int64)
        idx = idx[np.isfinite(base[idx])]
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(base[idx])]
        left, right = idx[:-1], idx[1:]
        keep = np.abs(base[right] - base[left]) <= .10
        lefts.append(left[keep])
        rights.append(right[keep])
    if not lefts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(lefts), np.concatenate(rights)


def _near_tie_concordance(values: np.ndarray, target_value: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    if not len(left):
        return np.nan
    valid = np.isfinite(values[left]) & np.isfinite(values[right]) & np.isfinite(target_value[left]) & np.isfinite(target_value[right])
    delta_feature = values[right[valid]] - values[left[valid]]
    delta_target = target_value[right[valid]] - target_value[left[valid]]
    nonzero = (delta_feature != 0.0) & (delta_target != 0.0)
    return float(np.mean(delta_feature[nonzero] * delta_target[nonzero] > 0.0)) if np.any(nonzero) else np.nan


def _metric_table(panel: pd.DataFrame, outcome: np.ndarray, selection_months: tuple[str, ...]) -> pd.DataFrame:
    features = [field for field in panel.columns if field.startswith(("f1_", "f2_", "f3_", "f4_", "f5_", "f6_"))]
    rng = np.random.default_rng(SEED)
    decile = np.minimum(9, np.maximum(0, np.floor(pd.to_numeric(panel["f1_base_rank_ts"], errors="coerce").fillna(.5) * 10))).astype(int).to_numpy()
    month_tokens = pd.to_datetime(panel["__decision_ts__"], utc=True).dt.strftime("%Y-%m")
    near_tie_left, near_tie_right = _near_tie_pairs(panel)
    rows = []
    for feature in features:
        values = pd.to_numeric(panel[feature], errors="coerce").to_numpy(float)
        valid = np.isfinite(values) & np.isfinite(outcome)
        coverage = float(np.mean(np.isfinite(values)))
        if valid.sum() < 100 or coverage < MIN_COVERAGE or np.nanstd(values) <= 1e-12:
            rows.append({"feature": feature, "family": feature.split("_", 1)[0], "coverage": coverage, "eligible": False})
            continue
        index = np.flatnonzero(valid)
        # A deterministic decile-stratified proxy retains conditional support
        # without paying the cost of full-universe MI for every candidate field.
        if len(index) > MAX_MI_ROWS:
            per_decile = max(200, MAX_MI_ROWS // 10)
            sampled = []
            for bucket in range(10):
                local = index[decile[index] == bucket]
                if len(local) > per_decile:
                    local = rng.choice(local, size=per_decile, replace=False)
                sampled.append(local)
            index = np.concatenate(sampled)
        mi = float(mutual_info_regression(values[index].reshape(-1, 1), outcome[index], random_state=SEED, n_neighbors=5)[0])
        cmi_values = []
        for bucket in range(10):
            bucket_index = index[decile[index] == bucket]
            if len(bucket_index) >= 200:
                cmi_values.append(float(mutual_info_regression(values[bucket_index].reshape(-1, 1), outcome[bucket_index], random_state=SEED + bucket, n_neighbors=5)[0]))
        cmi = float(np.mean(cmi_values)) if cmi_values else np.nan
        rho = float(spearmanr(values[valid], outcome[valid]).statistic)
        month_rho = []
        for token in selection_months:
            local = valid & month_tokens.eq(token).to_numpy()
            if local.sum() >= 100:
                month_rho.append(float(spearmanr(values[local], outcome[local]).statistic))
        stability = float(np.mean(np.sign(month_rho) == np.sign(rho))) if month_rho and rho != 0.0 else 0.0
        rows.append({
            "feature": feature, "family": feature.split("_", 1)[0], "coverage": coverage, "eligible": True,
            "residual_mi": mi, "approx_cmi_base_decile": cmi, "residual_spearman": rho,
            "near_tie_concordance": _near_tie_concordance(values, outcome, near_tie_left, near_tie_right),
            "monthly_spearman_mean": float(np.mean(month_rho)) if month_rho else np.nan,
            "monthly_spearman_std": float(np.std(month_rho)) if month_rho else np.nan,
            "monthly_sign_stability": stability, "monthly_values": json.dumps(month_rho),
        })
    metrics = pd.DataFrame(rows)
    eligible = metrics["eligible"].fillna(False).astype(bool)
    for column in ("residual_mi", "approx_cmi_base_decile", "residual_spearman", "near_tie_concordance", "monthly_sign_stability"):
        value = metrics.loc[eligible, column]
        if column == "residual_spearman":
            value = value.abs()
        metrics.loc[eligible, f"rank_{column}"] = value.rank(pct=True, method="average")
    metrics["screen_score"] = metrics.loc[:, [column for column in metrics if column.startswith("rank_")]].mean(axis=1)
    return metrics.sort_values(["family", "screen_score"], ascending=[True, False], na_position="last").reset_index(drop=True)


def _base_residualised(panel: pd.DataFrame, field: str) -> pd.Series:
    """Remove a train-only coarse base-rank relationship before redundancy.

    The screen is run on its declared development panel only, so base-decile
    conditional means are a valid train-only monotonic/bin proxy for
    ``E[X | enhanced_base_rank]``.  This deliberately avoids rejecting two
    features merely because both correlate with the upstream score.
    """
    value = pd.to_numeric(panel[field], errors="coerce")
    rank = pd.to_numeric(panel["f1_base_rank_ts"], errors="coerce").fillna(.5)
    decile = np.minimum(9, np.maximum(0, np.floor(rank * 10))).astype(int)
    means = value.groupby(decile, sort=False).transform("mean")
    return value - means


def _select(metrics: pd.DataFrame, panel: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for family, local in metrics.groupby("family", sort=True):
        candidates = local.loc[local["eligible"].fillna(False).astype(bool)].sort_values("screen_score", ascending=False)
        keep: list[str] = []
        for feature in candidates["feature"]:
            if len(keep) >= TOP_PER_FAMILY:
                break
            values = _base_residualised(panel, feature)
            redundant = False
            for chosen in keep:
                corr = values.corr(_base_residualised(panel, chosen), method="spearman")
                if np.isfinite(corr) and abs(float(corr)) >= .85:
                    redundant = True
                    break
            if not redundant:
                keep.append(feature)
        result[family] = keep
    return result


def _strict_prequential_selection_target(
    history_with_policy: pd.DataFrame,
    *, selection_months: tuple[pd.Timestamp, ...], selection_target: str,
) -> pd.Series:
    """Return a selection target whose anchor is fitted strictly pre-fold.

    This replaces the earlier in-panel isotonic fit.  Each development month
    uses a six-month resolved training window and the same 28-day reserve as
    the target funnel, so G1 conditional information is genuinely OOF rather
    than an in-sample calibration artefact.
    """
    work = history_with_policy.copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work["policy_label_available_ts"] = pd.to_datetime(work["policy_label_available_ts"], utc=True, errors="coerce")
    valid = (
        work["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(work["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(work["f1_base_rank_ts"], errors="coerce"))
    )
    output: dict[str, float] = {}
    for month in selection_months:
        end = month + pd.offsets.MonthBegin(1)
        reserve = month - pd.Timedelta(days=28)
        start = month - pd.DateOffset(months=6)
        train = work.loc[
            valid & work["__decision_ts__"].ge(start) & work["__decision_ts__"].lt(reserve)
            & work["policy_label_available_ts"].lt(reserve)
        ].copy()
        held = work.loc[valid & work["__decision_ts__"].ge(month) & work["__decision_ts__"].lt(end)].copy()
        if len(train) < 2_000 or held.empty:
            continue
        base_train = pd.to_numeric(train["f1_base_rank_ts"], errors="coerce").to_numpy(float)
        policy_train = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
        anchor = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(base_train, policy_train)
        if selection_target in {"economic_residual", "economic_residual_ordinal"}:
            residual = pd.to_numeric(held["policy_net_bps"], errors="coerce").to_numpy(float) - anchor.predict(pd.to_numeric(held["f1_base_rank_ts"], errors="coerce").to_numpy(float))
            if selection_target == "economic_residual_ordinal":
                residual = np.digitize(np.clip(residual, -500.0, 500.0), (-250.0, -100.0, -30.0, 30.0, 100.0, 250.0)).astype(float)
            values = residual
        elif selection_target in {"rank_error", "rank_error_ordinal"}:
            realised = held.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(pct=True, method="average").to_numpy(float)
            values = realised - pd.to_numeric(held["f1_base_rank_ts"], errors="coerce").to_numpy(float)
            if selection_target == "rank_error_ordinal":
                values = np.digitize(values, (-.20, -.05, .05, .20)).astype(float)
        else:
            raise ValueError(f"unsupported selection target: {selection_target}")
        output.update(dict(zip(held["candidate_id"].astype(str), values, strict=True)))
    return pd.Series(output, dtype=float, name="__selection_target__")


def run(
    *, base_root: Path, parent_root: Path, o3_roots: tuple[Path, ...], o3_arm: str | None, policy_path: Path,
    out: Path, history_months: tuple[pd.Timestamp, ...], selection_months: tuple[pd.Timestamp, ...], selection_target: str,
    portability_months: tuple[pd.Timestamp, ...] = (), write_history_panel: bool = False,
    history_only: bool = False,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    if not set(selection_months).issubset(set(history_months)):
        raise ValueError("selection months must be a subset of history months")
    if not set(portability_months).issubset(set(history_months)):
        raise ValueError("portability months must be a subset of history months")
    out.mkdir(parents=True)
    panel = _build_target_free(base_root, parent_root, o3_roots, o3_arm, selection_months)
    # Seal the score/input receipt before outcomes can enter any calculation.
    panel.to_parquet(out / "target_free_feature_panel.parquet", index=False, compression="zstd")
    policy = pd.read_parquet(policy_path, columns=("candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"))
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    # Recent-error telemetry needs pre-selection history.  Build it from the
    # target-free parent/feature receipts, then retain only selection rows.
    # F3 is defined exclusively from base/current parent provenance, so it can
    # use older compatible history even when an optional O3 score starts only
    # later.  This preserves support without making O3 availability a hidden
    # candidate selector.
    all_panel = _build_target_free(base_root, parent_root, tuple(), None, history_months)
    all_panel = _add_recent_telemetry(all_panel, policy, parent_root, history_months)
    if write_history_panel:
        all_panel.to_parquet(out / "target_free_history_feature_panel_with_f3.parquet", index=False, compression="zstd")
    if history_only:
        # A later G3 contract may already be frozen.  In that case, running
        # a second MI/feature-selection pass merely spends memory and risks
        # blurring the distinction between history materialisation and
        # development selection.  Seal just the causal, target-free feature
        # ledger needed by downstream strict-OOS heads.
        fd = os.open(out / "run_manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as handle:
            json.dump({
                "schema": SCHEMA,
                "scope": "offline target-free O3-v2 history materialisation only; no feature selection, fit, MC1, portfolio, or live mutation",
                "history_months": [f"{month:%Y-%m}" for month in history_months],
                "history_panel_persisted": bool(write_history_panel),
                "history_only": True,
                "causality": {
                    "feature_panel": "sealed target-free before any downstream model fit",
                    "recent_error": "uses only labels with label_available_ts < decision timestamp",
                    "selection": "not run; any downstream feature contract must be separately frozen",
                },
                "source_hashes": {
                    "base_root": _sha256(base_root),
                    "parent_root": _sha256(parent_root),
                    "policy_path": _sha256(policy_path),
                },
            }, handle, indent=2, sort_keys=True)
        return
    panel = panel.merge(all_panel.loc[:, ["candidate_id", *[field for field in all_panel if field.startswith("f3_")]]], on="candidate_id", how="left", validate="one_to_one")
    # This enriched receipt contains no policy outcome field; its F3 values are
    # strictly prior-resolved telemetry and can be supplied to a future model.
    panel.to_parquet(out / "target_free_feature_panel_with_f3.parquet", index=False, compression="zstd")
    history_with_policy = all_panel.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    target_by_id = _strict_prequential_selection_target(history_with_policy, selection_months=selection_months, selection_target=selection_target)
    joined = panel.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    joined["__selection_target__"] = joined["candidate_id"].astype(str).map(target_by_id)
    valid = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(joined["__selection_target__"], errors="coerce"))
    work = joined.loc[valid].copy().reset_index(drop=True)
    outcome = pd.to_numeric(work["__selection_target__"], errors="coerce").to_numpy(np.float32)
    metrics = _metric_table(work, outcome, tuple(f"{month:%Y-%m}" for month in selection_months))
    selection = _select(metrics, work)
    metrics.to_parquet(out / "feature_screen_metrics.parquet", index=False, compression="zstd")
    (out / "selected_features.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    # A separate, strictly subsequent portability receipt is deliberately not
    # fed back into selection.  This keeps the development screen useful
    # while giving the later head-selection stage evidence about whether a
    # field's conditional relationship survives a distinct temporal block.
    if portability_months:
        portability_panel = _build_target_free(base_root, parent_root, o3_roots, o3_arm, portability_months)
        portability_panel = portability_panel.merge(
            all_panel.loc[:, ["candidate_id", *[field for field in all_panel if field.startswith("f3_")]]],
            on="candidate_id", how="left", validate="one_to_one",
        )
        portability_panel.to_parquet(out / "target_free_portability_feature_panel_with_f3.parquet", index=False, compression="zstd")
        portability_joined = portability_panel.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        portability_target = _strict_prequential_selection_target(
            history_with_policy, selection_months=portability_months, selection_target=selection_target,
        )
        portability_joined["__selection_target__"] = portability_joined["candidate_id"].astype(str).map(portability_target)
        portability_valid = portability_joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(
            pd.to_numeric(portability_joined["__selection_target__"], errors="coerce")
        )
        portability_work = portability_joined.loc[portability_valid].copy().reset_index(drop=True)
        portability_outcome = pd.to_numeric(portability_work["__selection_target__"], errors="coerce").to_numpy(np.float32)
        portability_metrics = _metric_table(
            portability_work, portability_outcome, tuple(f"{month:%Y-%m}" for month in portability_months),
        )
        selected_fields = {field for fields in selection.values() for field in fields}
        screen_columns = ["feature", "family", "screen_score", "residual_mi", "approx_cmi_base_decile", "residual_spearman"]
        portability_metrics = portability_metrics.merge(
            metrics.loc[:, screen_columns], on=["feature", "family"], how="left", suffixes=("_holdout", "_development"),
            validate="one_to_one",
        )
        portability_metrics["selected_on_development"] = portability_metrics["feature"].isin(selected_fields)
        portability_metrics = portability_metrics.loc[portability_metrics["selected_on_development"]].copy()
        portability_metrics.to_parquet(out / "feature_screen_portability_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline O3-v2 feature screening only; MDA intentionally deferred to the user's separate pipeline",
        "o3_arm": o3_arm,
        "history_months": [f"{month:%Y-%m}" for month in history_months],
        "selection_months": [f"{month:%Y-%m}" for month in selection_months],
        "portability_months": [f"{month:%Y-%m}" for month in portability_months],
        "selection_target": selection_target,
        "families": {"F1": "base/upstream score geometry", "F2": "same-timestamp query geometry", "F3": "fully-resolved availability-clock recent error", "F4": "frozen causal state/transition base fields", "F5": "target-free current/BCF correction provenance" + (" plus legacy O3 receipt" if o3_roots else ""), "F6": "causal clock/config metadata available in the ledger"},
        "selection": "top ten per family, strict-prequential target and base-residualised redundancy veto at |Spearman| >= 0.85",
        "history_panel_persisted": bool(write_history_panel),
        "causality": {"feature_panel": "sealed target-free before policy outcomes joined", "recent_error": "uses only labels with label_available_ts < decision timestamp", "selection": "development months only; no future held period used", "portability": "strict-prequential held target; persisted solely as diagnostic and never read by selection"},
        "source_hashes": {"base_root": _sha256(base_root), "parent_root": _sha256(parent_root), "o3_roots": {str(root): _sha256(root) for root in o3_roots}, "policy_path": _sha256(policy_path)},
    }
    fd = os.open(out / "run_manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def run_from_history(
    *, history_panel: Path, policy_path: Path, out: Path,
    selection_months: tuple[pd.Timestamp, ...], selection_target: str,
) -> None:
    """Screen an already-sealed target-free F1--F6 history panel.

    The full feature history is expensive to construct but has already been
    sealed before any label joins.  Rebuilding the identical source panel for
    every post-query feature screen wastes hours and adds no information.
    This entrypoint preserves the same strict-prequential selection target
    while reusing that immutable target-free ledger.
    """
    if out.exists():
        raise FileExistsError(out)
    schema = set(pq.ParquetFile(history_panel).schema_arrow.names)
    prohibited = target.PROHIBITED_SCORE_COLUMNS.intersection(schema)
    if prohibited:
        raise AssertionError(f"history-panel contains outcome columns: {sorted(prohibited)}")
    required = {"candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts"}
    if missing := required - schema:
        raise KeyError(f"history-panel lacks required target-free fields: {sorted(missing)}")
    out.mkdir(parents=True)
    history = pd.read_parquet(history_panel)
    history["__decision_ts__"] = pd.to_datetime(history["__decision_ts__"], utc=True, errors="raise")
    if history["candidate_id"].duplicated().any():
        raise AssertionError("history-panel has duplicate candidate identities")
    history_months = tuple(sorted(history["__decision_ts__"].dt.to_period("M").astype(str).unique()))
    requested = {f"{month:%Y-%m}" for month in selection_months}
    if not requested.issubset(set(history_months)):
        raise AssertionError("history-panel lacks one or more declared selection months")
    selection_mask = history["__decision_ts__"].dt.strftime("%Y-%m").isin(requested)
    panel = history.loc[selection_mask].copy()
    # The target-free receipt is published before the policy ledger is opened.
    panel.to_parquet(out / "target_free_feature_panel_with_f3.parquet", index=False, compression="zstd")
    policy = pd.read_parquet(
        policy_path,
        columns=("candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"),
    )
    policy["policy_label_available_ts"] = pd.to_datetime(
        policy["policy_label_available_ts"], utc=True, errors="coerce",
    )
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy ledger has duplicate identities")
    history_with_policy = history.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    target_by_id = _strict_prequential_selection_target(
        history_with_policy,
        selection_months=selection_months,
        selection_target=selection_target,
    )
    joined = panel.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    joined["__selection_target__"] = joined["candidate_id"].astype(str).map(target_by_id)
    valid = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(
        pd.to_numeric(joined["__selection_target__"], errors="coerce")
    )
    work = joined.loc[valid].copy().reset_index(drop=True)
    outcome = pd.to_numeric(work["__selection_target__"], errors="coerce").to_numpy(np.float32)
    metrics = _metric_table(work, outcome, tuple(f"{month:%Y-%m}" for month in selection_months))
    selection = _select(metrics, work)
    metrics.to_parquet(out / "feature_screen_metrics.parquet", index=False, compression="zstd")
    (out / "selected_features.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline O3-v2 post-query feature screening from a sealed target-free history panel; MDA intentionally deferred",
        "history_panel": str(history_panel.resolve()),
        "history_panel_sha256": _sha256(history_panel),
        "history_months": list(history_months),
        "selection_months": [f"{month:%Y-%m}" for month in selection_months],
        "selection_target": selection_target,
        "families": {"F1": "base/upstream score geometry", "F2": "same-timestamp query geometry", "F3": "fully-resolved availability-clock recent error", "F4": "frozen causal state/transition base fields", "F5": "target-free current/BCF correction provenance", "F6": "causal clock/config metadata"},
        "selection": "top ten per family, strict-prequential target and base-residualised redundancy veto at |Spearman| >= 0.85",
        "causality": {
            "feature_panel": "reused sealed target-free history receipt before any policy-outcome join",
            "selection_target": "six-month prequential anchor and 28-day reserve for every declared development month",
            "later_portability": "not read by selection",
        },
        "source_hashes": {"policy_path": _sha256(policy_path)},
    }
    fd = os.open(out / "run_manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _parse_months(raw: str) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in raw.split(",") if token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--history-panel", type=Path, help="reuse an immutable target-free F1--F6 history panel")
    parser.add_argument("--o3-root", help="Optional ROOT[|ROOT]; exactly one root must supply each requested O3 month")
    parser.add_argument("--o3-arm", help="Required only when --o3-root is supplied")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-months", default="", help="comma-separated YYYY-MM, including resolved telemetry history")
    parser.add_argument("--selection-months", required=True, help="comma-separated development YYYY-MM")
    parser.add_argument("--selection-target", choices=("economic_residual", "economic_residual_ordinal", "rank_error", "rank_error_ordinal"), default="economic_residual_ordinal")
    parser.add_argument("--portability-months", default="", help="comma-separated strictly later diagnostic months; never used for selection")
    parser.add_argument("--write-history-panel", action="store_true", help="persist the full causal feature history for downstream specialist research")
    parser.add_argument("--history-only", action="store_true",
                        help="materialise and seal only the target-free history; skip all outcome-based feature screening")
    args = parser.parse_args()
    if bool(args.o3_root) != bool(args.o3_arm):
        parser.error("--o3-root and --o3-arm must be supplied together")
    selection_months = _parse_months(args.selection_months)
    if args.history_panel is not None:
        if args.base_root is not None or args.parent_root is not None or args.o3_root or args.o3_arm:
            parser.error("--history-panel cannot be combined with source-root arguments")
        if args.portability_months or args.write_history_panel or args.history_only:
            parser.error("--history-panel supports the sealed post-query screen only")
        run_from_history(
            history_panel=args.history_panel, policy_path=args.policy_path, out=args.out,
            selection_months=selection_months, selection_target=args.selection_target,
        )
        return
    if args.base_root is None or args.parent_root is None or not args.history_months:
        parser.error("fresh screens require --base-root, --parent-root, and --history-months")
    run(base_root=args.base_root, parent_root=args.parent_root, o3_roots=tuple(Path(value) for value in (args.o3_root or "").split("|") if value), o3_arm=args.o3_arm, policy_path=args.policy_path,
        out=args.out, history_months=_parse_months(args.history_months), selection_months=selection_months, selection_target=args.selection_target,
        portability_months=_parse_months(args.portability_months),
        write_history_panel=args.write_history_panel, history_only=args.history_only)


if __name__ == "__main__":
    main()
