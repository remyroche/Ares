#!/usr/bin/env python3
"""Frozen portability audit for the short P0 -> O250/H6 -> C3/C59 -> K0 stack.

This is deliberately *not* a new feature-selection or economic-optimisation
run.  O45, C59, their target/model geometry and K0 are frozen.  The producer:

* audits the target-free feature contract for coverage, drift, redundancy and
  source reliability;
* records the already-completed chronological MDA stability rather than
  creating a fresh 2025--26 importance search;
* runs head-family dropouts with frozen geometry only as structural
  attribution diagnostics; and
* creates portability/stability-only O30 and C40 challenger contracts for a
  future untouched test.  They are not retrained, ranked or promoted here.

Every result that uses labels is explicitly marked diagnostic-only.  No output
from this file is consumed by the canonical short research stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round2 as r2  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_refinement as r3b  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair as r3d  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round4_k0_refinement as r4  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_feature_contract_portability_v1"
# v1 is retained as the failed publication receipt: its O-volatility table
# had no usable early-development folds.  v2 fixed that issue but grouped
# OI, funding and leverage too coarsely for the requested family attribution.
# v3 is the first receipt with both repairs.
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_feature_contract_portability_20260822_v3"
C59 = r3d.OUT / "C59_outer_oof_predictions.parquet"
ROUND2 = r2.DEFAULT_OUT
ROUND3B = r3b.OUT
FROZEN_DIAGNOSTICS = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_frozen_diagnostics_mc1_20260822_v2"
DEV_START = pd.Timestamp("2024-05-01T00:00:00Z")
DEV_END = pd.Timestamp("2025-01-01T00:00:00Z")
ADMISSION = 75.0
MU1 = ("isotonic", 0)
MU0 = ("anchor5", 500)
O_COMPACT_CAPS = (40, 35, 30)
C_COMPACT_CAPS = (50, 40)
REDUNDANCY_PRIMARY = .95
REDUNDANCY_SECONDARY = .90
MIN_BUCKET_ROWS = 25
SEED = 1729
VOLATILITY_PROXY = "q_tail_width__volatility_zscore"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _finite(values: pd.Series | np.ndarray | Iterable[float]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)


def _family(feature: str) -> str:
    """Stable semantic/provenance family assignment, never outcome-derived."""
    name = feature.lower()
    if "session" in name or name.startswith("loc_"):
        return "session_time"
    if "spectral" in name or name.startswith("eig_"):
        return "market_state_spectral"
    # Keep the distinct positioning inputs separate for dropout attribution.
    # A funding×OI interaction is assigned to funding because it cannot be
    # calculated without the funding source; `leverage` is also its own
    # state/derivative family rather than silently inflating OI attribution.
    if "funding" in name:
        return "funding"
    if "leverage" in name:
        return "leverage"
    if any(token in name for token in ("open_interest", "oi_", "_oi", "oiw")):
        return "open_interest_positioning"
    if any(token in name for token in ("ob_", "orderbook", "liquidity", "spread", "depth", "amihud")):
        return "liquidity_order_book"
    if "volume" in name:
        return "volume_activity"
    if any(token in name for token in ("volatility", "rvol", "semivol", "high_vol", "vol_z")):
        return "volatility_transition"
    if any(token in name for token in ("support", "resistance", "donchian", "vwap", "range_pos", "swing_range")):
        return "support_resistance_structure"
    if any(token in name for token in ("pct_assets", "breadth", "xs_", "xasset", "mkt_")):
        return "cross_asset_breadth"
    if any(token in name for token in ("ret", "price", "bars_since", "efficiency", "exh_", "grind", "breakout", "recovery", "up_down")):
        return "price_momentum_reversal"
    return "other"


def _source_group(feature: str) -> str:
    """Decision-time source lineage class; no outcome-derived assignment."""
    name = feature.lower()
    if "funding" in name:
        return "funding"
    if any(token in name for token in ("open_interest", "oi_", "_oi", "oiw", "leverage")):
        return "open_interest_positioning"
    if any(token in name for token in ("ob_", "orderbook", "liquidity", "spread", "depth", "amihud")):
        return "order_book"
    if any(token in name for token in ("pct_assets", "breadth", "xs_", "xasset", "mkt_", "spectral", "eig_")):
        return "cross_asset_market_panel"
    return "ohlcv_derived"


def _era(stamp: pd.Series) -> pd.Series:
    return stamp.dt.strftime("%Y")


def _quantile_edges(values: np.ndarray, count: int = 10) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 10:
        return np.array([-np.inf, np.inf], dtype=float)
    edges = np.unique(np.quantile(clean, np.linspace(0.0, 1.0, count + 1)))
    if len(edges) < 3:
        return np.array([-np.inf, np.inf], dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _psi(reference: np.ndarray, values: np.ndarray) -> float:
    edges = _quantile_edges(reference)
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(values, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10 or len(edges) < 3:
        return float("nan")
    p = np.diff(np.searchsorted(np.sort(ref), edges, side="left")) / max(len(ref), 1)
    q = np.diff(np.searchsorted(np.sort(cur), edges, side="left")) / max(len(cur), 1)
    # The finite edge construction can leave a small terminal rounding error.
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 10 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")
    value = pair["left"].corr(pair["right"], method="spearman")
    return float(value) if np.isfinite(value) else float("nan")


def _head_inventory(o_fields: Sequence[str], c_fields: Sequence[str]) -> pd.DataFrame:
    rows = [{"head": "O", "feature": field, "family": _family(field), "source_group": _source_group(field)} for field in o_fields]
    rows.extend({"head": "C", "feature": field, "family": _family(field), "source_group": _source_group(field)} for field in c_fields)
    inventory = pd.DataFrame(rows)
    if inventory.duplicated(["head", "feature"]).any():
        raise AssertionError("feature contract is not unique within head")
    return inventory.sort_values(["head", "family", "feature"], kind="stable").reset_index(drop=True)


def _load() -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], pd.DataFrame, dict[str, str]]:
    frame, o_fields, _, source_hashes = r3._load_frame()
    c_fields = r3d._c59()
    if len(o_fields) != 45 or len(c_fields) != 59:
        raise AssertionError("expected frozen O45/C59 contracts")
    # ``r3._load_frame`` intentionally only materialises the frozen model
    # fields.  The source-quality audit additionally needs one target-free
    # liquidity-pressure column for its *diagnostic* symbol tiers.  Load that
    # column directly from the immutable P0 population rather than widening
    # O45/C59 or deriving a tier from outcomes.
    liquidity_proxy = "ob_trade_size_to_l1_depth_z_24h"
    probe_parts: list[pd.DataFrame] = []
    for root in r1.DEFAULT_POPULATION_ROOTS:
        path = root / "short_p0_top1_hourly_population.parquet"
        probe_parts.append(pd.read_parquet(path, columns=["candidate_id", liquidity_proxy]))
    probe = pd.concat(probe_parts, ignore_index=True)
    repeated = probe["candidate_id"].duplicated(keep=False)
    if repeated.any():
        conflict = probe.loc[repeated].groupby("candidate_id", sort=False)[liquidity_proxy].nunique(dropna=False)
        if (conflict > 1).any():
            raise AssertionError("immutable P0 sources disagree on audit liquidity proxy")
        probe = probe.drop_duplicates("candidate_id", keep="first")
    if liquidity_proxy in frame.columns:
        # The current F115 panel happens to contain this field already.  The
        # explicit population read is retained as a provenance check, rather
        # than silently accepting a potentially transformed duplicate.
        check = frame.loc[:, ["candidate_id", liquidity_proxy]].merge(
            probe, on="candidate_id", how="left", validate="one_to_one", suffixes=("_frame", "_population"),
        )
        if not np.isclose(
            _finite(check[f"{liquidity_proxy}_frame"]), _finite(check[f"{liquidity_proxy}_population"]),
            rtol=0.0, atol=2e-6, equal_nan=True,
        ).all():
            raise AssertionError("feature panel and immutable P0 source disagree on audit liquidity proxy")
    else:
        frame = frame.merge(probe, on="candidate_id", how="left", validate="one_to_one")
    if len(frame) != len(probe) or frame[liquidity_proxy].isna().all():
        raise AssertionError("target-free liquidity proxy join is incomplete")
    required = set(o_fields) | set(c_fields) | {VOLATILITY_PROXY, liquidity_proxy}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"target-free source lacks required audit fields: {missing}")
    frame = frame.copy()
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["month"] = frame["__decision_ts__"].dt.strftime("%Y-%m")
    frame["era"] = _era(frame["__decision_ts__"])
    inventory = _head_inventory(o_fields, c_fields)
    return frame, tuple(o_fields), tuple(c_fields), inventory, source_hashes


def _family_proxy(frame: pd.DataFrame, all_features: Sequence[str], inventory: pd.DataFrame) -> pd.DataFrame:
    """Monthly feature-to-family proxy Spearman; proxy excludes the field itself."""
    unique_family = (
        inventory.drop_duplicates("feature").groupby("family", sort=True)["feature"].apply(list).to_dict()
    )
    rows: list[dict[str, Any]] = []
    for month, local in frame.groupby("month", sort=True):
        for family, fields in unique_family.items():
            use = [field for field in fields if field in all_features]
            if len(use) < 2:
                continue
            values = local.loc[:, use].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            ranks = values.rank(pct=True, method="average")
            for field in use:
                proxy = ranks.drop(columns=field).median(axis=1, skipna=True)
                rows.append({
                    "month": month, "feature": field, "family": family,
                    "proxy_spearman": _safe_corr(ranks[field], proxy),
                    "proxy_support": int(pd.DataFrame({"x": ranks[field], "p": proxy}).dropna().shape[0]),
                })
    return pd.DataFrame(rows)


def _month_feature_audit(frame: pd.DataFrame, features: Sequence[str], inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = frame.loc[frame["__decision_ts__"].ge(DEV_START) & frame["__decision_ts__"].lt(DEV_END)]
    if dev.empty:
        raise RuntimeError("target-free development period is unavailable")
    references: dict[str, dict[str, float | np.ndarray]] = {}
    for feature in features:
        values = _finite(dev[feature])
        clean = values[np.isfinite(values)]
        if not len(clean):
            references[feature] = {"median": float("nan"), "iqr": float("nan"), "p05": float("nan"), "p95": float("nan"), "values": clean}
            continue
        references[feature] = {
            "median": float(np.median(clean)), "iqr": float(np.subtract(*np.percentile(clean, [75, 25]))),
            "p05": float(np.percentile(clean, 5)), "p95": float(np.percentile(clean, 95)), "values": clean,
        }
    rows: list[dict[str, Any]] = []
    for month, local in frame.groupby("month", sort=True):
        for feature in features:
            values = _finite(local[feature])
            clean = values[np.isfinite(values)]
            ref = references[feature]
            coverage = float(len(clean) / len(values)) if len(values) else float("nan")
            if len(clean):
                median = float(np.median(clean))
                iqr = float(np.subtract(*np.percentile(clean, [75, 25])))
                p01, p05, p95, p99 = (float(np.percentile(clean, q)) for q in (1, 5, 95, 99))
                variance = float(np.var(clean))
            else:
                median = iqr = p01 = p05 = p95 = p99 = variance = float("nan")
            ref_iqr = float(ref["iqr"])
            scale = max(abs(ref_iqr), 1e-9)
            robust_z = np.abs((clean - float(ref["median"])) / scale) if len(clean) and np.isfinite(ref_iqr) else np.array([], dtype=float)
            ref_range = max(abs(float(ref["p95"]) - float(ref["p05"])), 1e-9)
            rows.append({
                "month": month, "era": str(month)[:4], "feature": feature,
                "finite_coverage": coverage, "missingness": 1.0 - coverage if np.isfinite(coverage) else float("nan"),
                "variance": variance, "median": median, "iqr": iqr,
                "p01": p01, "p05": p05, "p95": p95, "p99": p99,
                "median_shift_dev_iqr": (median - float(ref["median"])) / scale if np.isfinite(median) and np.isfinite(float(ref["median"])) else float("nan"),
                "iqr_ratio_dev": iqr / scale if np.isfinite(iqr) else float("nan"),
                "p05_p95_range_ratio_dev": (p95 - p05) / ref_range if np.isfinite(p95) and np.isfinite(p05) else float("nan"),
                "psi_vs_dev": _psi(np.asarray(ref["values"]), clean),
                "outlier_frequency_robust_z8": float(np.mean(robust_z > 8.0)) if len(robust_z) else float("nan"),
                "variance_near_zero": bool(np.isfinite(variance) and variance <= 1e-12),
            })
    monthly = pd.DataFrame(rows)
    proxy = _family_proxy(frame, features, inventory)
    monthly = monthly.merge(proxy, on=["month", "feature"], how="left", validate="many_to_one")
    return monthly, pd.DataFrame(references).T.reset_index(names="feature")


def _symbol_tier_coverage(frame: pd.DataFrame, features: Sequence[str], inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # The tiering proxy is audit metadata only; it is intentionally not added
    # to either frozen O45 or C59 contract.
    liquidity_proxy = "ob_trade_size_to_l1_depth_z_24h"
    columns = list(dict.fromkeys(["__decision_ts__", "__symbol__", "era", *features, VOLATILITY_PROXY, liquidity_proxy]))
    output = frame.loc[:, columns].copy()
    born = output.groupby("__symbol__", sort=True)["__decision_ts__"].min()
    output["symbol_age_days"] = (output["__decision_ts__"] - output["__symbol__"].map(born)).dt.total_seconds() / 86400.0
    output["symbol_age_tier"] = pd.cut(output["symbol_age_days"], [-1, 90, 365, np.inf], labels=["0_90d", "91_365d", "366d_plus"]).astype(str)
    # The frozen panel has no absolute volume/depth field.  This is explicitly
    # a *target-free liquidity-pressure proxy*, based on the historical median
    # trade-size-to-L1-depth z score: lower pressure denotes higher liquidity.
    proxy = liquidity_proxy
    dev = output.loc[output["__decision_ts__"].ge(DEV_START) & output["__decision_ts__"].lt(DEV_END)]
    by_symbol = dev.groupby("__symbol__", sort=True)[proxy].median()
    fallback = float(np.nanmedian(_finite(dev[proxy])))
    symbol_proxy = by_symbol.fillna(fallback)
    rank = symbol_proxy.rank(method="first", pct=True)
    tier = pd.cut(rank, [0.0, 1 / 3, 2 / 3, 1.0], labels=["high_liquidity_proxy", "mid_liquidity_proxy", "low_liquidity_proxy"], include_lowest=True)
    output["liquidity_pressure_tier"] = output["__symbol__"].map(tier.astype(str)).fillna("unknown")
    rows: list[dict[str, Any]] = []
    for tier_col, audit_name in (("symbol_age_tier", "symbol_age"), ("liquidity_pressure_tier", "liquidity_pressure_proxy")):
        for (era, tier_name), local in output.groupby(["era", tier_col], sort=True):
            for feature in features:
                values = _finite(local[feature])
                rows.append({"audit": audit_name, "era": era, "tier": str(tier_name), "feature": feature, "rows": int(len(local)), "finite_coverage": float(np.mean(np.isfinite(values)))})
    coverage = pd.DataFrame(rows).merge(inventory.drop_duplicates("feature"), on="feature", how="left", validate="many_to_many")
    reliability = []
    for (feature, era), local in coverage.groupby(["feature", "era"], sort=True):
        source = local.loc[local["audit"].eq("liquidity_pressure_proxy")]
        reliability.append({
            "feature": feature, "era": era,
            "min_tier_coverage": float(source["finite_coverage"].min()) if len(source) else float("nan"),
            "coverage_spread_across_tiers": float(source["finite_coverage"].max() - source["finite_coverage"].min()) if len(source) else float("nan"),
        })
    return coverage.loc[coverage["audit"].eq("symbol_age")].reset_index(drop=True), coverage.loc[coverage["audit"].eq("liquidity_pressure_proxy")].reset_index(drop=True), pd.DataFrame(reliability)


def _summarise_portability(monthly: pd.DataFrame, inventory: pd.DataFrame, tier_reliability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, local in monthly.groupby("feature", sort=True):
        dev = local.loc[local["era"].eq("2024") & local["month"].ge(DEV_START.strftime("%Y-%m"))]
        later = local.loc[local["era"].isin(("2025", "2026"))]
        stats: dict[str, Any] = {
            "feature": feature,
            "development_months": int(len(dev)), "later_months": int(len(later)),
            "dev_coverage": float(dev["finite_coverage"].mean()),
            "coverage_2025": float(local.loc[local["era"].eq("2025"), "finite_coverage"].mean()),
            "coverage_2026": float(local.loc[local["era"].eq("2026"), "finite_coverage"].mean()),
            "min_month_coverage": float(later["finite_coverage"].min()),
            "zero_variance_months_later": int(later["variance_near_zero"].sum()),
            "max_abs_median_shift_dev_iqr": float(np.nanmax(np.abs(later["median_shift_dev_iqr"]))) if later["median_shift_dev_iqr"].notna().any() else float("nan"),
            "max_psi_vs_dev": float(np.nanmax(later["psi_vs_dev"])) if later["psi_vs_dev"].notna().any() else float("nan"),
            "max_outlier_frequency": float(np.nanmax(later["outlier_frequency_robust_z8"])) if later["outlier_frequency_robust_z8"].notna().any() else float("nan"),
            "min_p05_p95_range_ratio": float(np.nanmin(later["p05_p95_range_ratio_dev"])) if later["p05_p95_range_ratio_dev"].notna().any() else float("nan"),
            "max_p05_p95_range_ratio": float(np.nanmax(later["p05_p95_range_ratio_dev"])) if later["p05_p95_range_ratio_dev"].notna().any() else float("nan"),
            "proxy_spearman_median": float(np.nanmedian(later["proxy_spearman"])) if later["proxy_spearman"].notna().any() else float("nan"),
            "proxy_spearman_min": float(np.nanmin(later["proxy_spearman"])) if later["proxy_spearman"].notna().any() else float("nan"),
        }
        reasons: list[str] = []
        # A one-month late-source gap is an important review item but should
        # not silently turn a 99%-available field into a permanent blacklist.
        # The hard rule is sustained era-level availability below 90%, exactly
        # the portability failure that led to the C60 repair.
        if min(stats["coverage_2025"], stats["coverage_2026"]) < .90:
            reasons.append("later_era_coverage_lt_90pct")
        if stats["zero_variance_months_later"] >= 2:
            reasons.append("near_zero_variance_in_2plus_later_months")
        stats["hard_blacklist"] = bool(reasons)
        stats["blacklist_reason"] = ";".join(reasons)
        review: list[str] = []
        if stats["min_month_coverage"] < .90:
            review.append("single_month_coverage_lt_90pct")
        if np.isfinite(stats["max_psi_vs_dev"]) and stats["max_psi_vs_dev"] > .50:
            review.append("psi_gt_0.50")
        if np.isfinite(stats["max_abs_median_shift_dev_iqr"]) and stats["max_abs_median_shift_dev_iqr"] > 4.0:
            review.append("median_shift_gt_4_dev_iqr")
        if np.isfinite(stats["max_outlier_frequency"]) and stats["max_outlier_frequency"] > .10:
            review.append("robust_outlier_frequency_gt_10pct")
        stats["review_flags"] = ";".join(review)
        rows.append(stats)
    summary = pd.DataFrame(rows).merge(inventory, on="feature", how="right", validate="one_to_many")
    summary = summary.merge(tier_reliability, on=["feature", "era"], how="left") if False else summary
    # Collapse tier reliability without letting it set a supervised keep/drop rule.
    rel = tier_reliability.groupby("feature", as_index=False).agg(
        source_min_tier_coverage=("min_tier_coverage", "min"),
        source_max_tier_coverage_spread=("coverage_spread_across_tiers", "max"),
    )
    summary = summary.merge(rel, on="feature", how="left", validate="many_to_one")
    return summary.sort_values(["head", "hard_blacklist", "max_psi_vs_dev", "feature"], ascending=[True, True, True, True], kind="stable").reset_index(drop=True)


def _source_reliability(
    monthly: pd.DataFrame,
    inventory: pd.DataFrame,
    tier_reliability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate historical input availability from unproven live reliability.

    The short P0/O/C stack is research-only: it has no sealed short live
    inference config or runtime-receipt chain.  It would be misleading to
    label a historical finite-coverage statistic as a live-source SLO.  This
    table preserves the useful historical source evidence and explicitly
    fails the live-reliability claim closed until such a contract exists.
    """
    reliability = (
        monthly.groupby("feature", as_index=False).agg(
            historical_mean_finite_coverage=("finite_coverage", "mean"),
            historical_min_month_finite_coverage=("finite_coverage", "min"),
            historical_max_missingness=("missingness", "max"),
            historical_zero_variance_months=("variance_near_zero", "sum"),
        )
        .merge(inventory, on="feature", how="right", validate="one_to_many")
        .merge(
            tier_reliability.groupby("feature", as_index=False).agg(
                historical_min_liquidity_tier_coverage=("min_tier_coverage", "min"),
                historical_max_liquidity_tier_coverage_spread=("coverage_spread_across_tiers", "max"),
            ),
            on="feature", how="left", validate="many_to_one",
        )
    )
    reliability["historical_source_status"] = np.where(
        reliability["historical_min_month_finite_coverage"].ge(.90), "historically_available", "historical_gap_observed",
    )
    reliability["short_live_source_contract_status"] = "not_deployed_no_sealed_short_runtime_contract"
    reliability["live_source_reliability_evidenced"] = False
    summary = reliability.groupby(["head", "source_group"], as_index=False).agg(
        features=("feature", "size"),
        historical_mean_coverage=("historical_mean_finite_coverage", "mean"),
        historical_min_coverage=("historical_min_month_finite_coverage", "min"),
        historical_gap_features=("historical_source_status", lambda values: int((values == "historical_gap_observed").sum())),
        live_source_reliability_evidenced=("live_source_reliability_evidenced", "all"),
    )
    summary["short_live_source_contract_status"] = "not_deployed_no_sealed_short_runtime_contract"
    return reliability.sort_values(["head", "source_group", "feature"], kind="stable").reset_index(drop=True), summary.sort_values(["head", "source_group"], kind="stable").reset_index(drop=True)


def _mda_stability(o_fields: Sequence[str], c_fields: Sequence[str], inventory: pd.DataFrame) -> pd.DataFrame:
    o = pd.read_parquet(ROUND2 / "round2_target_specific_stability_mda.parquet")
    o = o.loc[o["arm"].eq("O250_H6") & o["feature"].isin(o_fields)].copy()
    o["head"] = "O"
    c = pd.read_parquet(ROUND3B / "round3b_c3_stability_mda.parquet")
    c = c.loc[c["feature"].isin(c_fields)].copy()
    c["head"] = "C"
    selected = pd.concat([o, c], ignore_index=True)
    selected["mda_positive_share"] = selected["mda_positive_folds"] / selected["mda_folds"].clip(lower=1)
    return inventory.merge(selected.loc[:, ["head", "feature", "mda_mean", "mda_min", "mda_positive_folds", "mda_folds", "mda_positive_share", "rank"]], on=["head", "feature"], how="left", validate="one_to_one")


def _redundancy(frame: pd.DataFrame, fields: Sequence[str], head: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr = values.corr(method="spearman").abs()
    monthly_corr: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _, local in frame.groupby("month", sort=True):
        local_corr = local.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
        for left_index, left in enumerate(fields):
            for right in fields[left_index + 1:]:
                value = local_corr.loc[left, right]
                if np.isfinite(value):
                    monthly_corr[(left, right)].append(float(value))
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(fields):
        for right in fields[left_index + 1:]:
            value = float(corr.loc[left, right])
            month_values = monthly_corr[(left, right)]
            rows.append({
                "head": head, "left_feature": left, "right_feature": right,
                "abs_spearman": value,
                "monthly_abs_spearman_median": float(np.median(month_values)) if month_values else float("nan"),
                "monthly_abs_spearman_min": float(np.min(month_values)) if month_values else float("nan"),
                "months_ge_95": int(sum(item >= REDUNDANCY_PRIMARY for item in month_values)),
                "months": int(len(month_values)),
                "primary_cluster_edge": bool(value > REDUNDANCY_PRIMARY),
                "secondary_pair": bool(REDUNDANCY_SECONDARY < value <= REDUNDANCY_PRIMARY),
            })
    pairs = pd.DataFrame(rows).sort_values(["head", "abs_spearman", "left_feature", "right_feature"], ascending=[True, False, True, True], kind="stable")
    # Connected components are used only to define representatives.  The
    # primary threshold is deliberately strict and target-free.
    parents = {field: field for field in fields}
    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[b] = a
    for row in pairs.loc[pairs["primary_cluster_edge"]].itertuples(index=False):
        union(str(row.left_feature), str(row.right_feature))
    groups: dict[str, list[str]] = defaultdict(list)
    for field in fields:
        groups[find(field)].append(field)
    rows = []
    for index, members in enumerate(sorted((sorted(value) for value in groups.values()), key=lambda value: (value[0], len(value)))):
        for field in members:
            rows.append({"head": head, "cluster_id": f"{head}_r95_{index:02d}", "feature": field, "cluster_size": len(members), "redundancy_clustered": len(members) > 1})
    return pairs.reset_index(drop=True), pd.DataFrame(rows)


def _compact_contracts(summary: pd.DataFrame, stability: pd.DataFrame, clusters: pd.DataFrame, o_fields: Sequence[str], c_fields: Sequence[str]) -> dict[str, Any]:
    merged = stability.merge(summary.loc[:, ["head", "feature", "hard_blacklist", "dev_coverage", "coverage_2025", "coverage_2026", "max_psi_vs_dev", "max_abs_median_shift_dev_iqr", "source_min_tier_coverage"]], on=["head", "feature"], how="left", validate="one_to_one")
    merged = merged.merge(clusters.loc[:, ["head", "feature", "cluster_id", "cluster_size"]], on=["head", "feature"], how="left", validate="one_to_one")
    contracts: dict[str, Any] = {
        "selection_policy": {
            "prohibited": "2025-2026 policy EV, K0 admission economics, or post-hoc tuning",
            "allowed": "existing chronological MDA stability, target-free coverage/drift, source-tier coverage, strict redundancy and nominal computational cost",
            "primary_redundancy_abs_spearman_gt": REDUNDANCY_PRIMARY,
            "hard_blacklist": "sustained 2025 or 2026 coverage <90% or near-zero variance in >=2 later months",
        },
        "contracts": {
            "O45_canonical_frozen": list(o_fields),
            "C59_canonical_frozen": list(c_fields),
        },
        "future_untouched_evaluation": {
            "O30_C40": "predeclared compact stability/portability challenger",
            "O45_C59": "frozen current research control",
            "O60": "not emitted: it would expand beyond the frozen O45 contract and require a future supervised feature-generation/selection study, which this audit forbids",
        },
    }
    for head, original, caps in (("O", list(o_fields), O_COMPACT_CAPS), ("C", list(c_fields), C_COMPACT_CAPS)):
        local = merged.loc[merged["head"].eq(head)].copy()
        representatives: set[str] = set()
        for _, group in local.groupby("cluster_id", dropna=False, sort=True):
            ranked = group.sort_values(
                ["hard_blacklist", "mda_positive_share", "mda_mean", "coverage_2026", "coverage_2025", "dev_coverage", "max_psi_vs_dev", "feature"],
                ascending=[True, False, False, False, False, False, True, True], kind="stable",
            )
            representatives.add(str(ranked.iloc[0]["feature"]))
        local = local.loc[local["feature"].isin(representatives) & ~local["hard_blacklist"].fillna(True)].copy()
        local["nominal_compute_cost"] = local["feature"].str.len().astype(float)
        local = local.sort_values(
            ["mda_positive_share", "mda_mean", "coverage_2026", "coverage_2025", "dev_coverage", "source_min_tier_coverage", "max_psi_vs_dev", "nominal_compute_cost", "feature"],
            ascending=[False, False, False, False, False, False, True, True, True], kind="stable",
        )
        ordered = local["feature"].astype(str).tolist()
        contracts["contracts"][f"{head}_redundancy_portability_only"] = ordered
        for cap in caps:
            if cap > len(ordered):
                # Do not pad a supposedly portable contract with a field that
                # the audit itself blacklisted.  Preserve the requested name
                # as an explicit unavailable diagnostic instead.
                contracts.setdefault("unavailable_requested_contracts", {})[f"{head}{cap}_predeclared_stability_core"] = {
                    "available_fields": len(ordered),
                    "reason": "target-free hard portability/redundancy rules leave fewer fields than requested; no padded contract is emitted",
                }
                continue
            contracts["contracts"][f"{head}{cap}_predeclared_stability_core"] = ordered[:cap]
        contracts.setdefault("selection_audit", {})[head] = local.loc[:, ["feature", "family", "cluster_id", "cluster_size", "mda_positive_share", "mda_mean", "coverage_2025", "coverage_2026", "max_psi_vs_dev", "source_min_tier_coverage"]].to_dict(orient="records")
    return contracts


def _k0_from_prediction(prediction: pd.DataFrame, anchors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    ledger = prediction.merge(anchors, on="candidate_id", how="left", validate="one_to_one")
    if ledger[r4.P0_ANCHOR].isna().any():
        raise AssertionError("family-dropout prediction lacks P0 anchor")
    ledger["__decision_ts__"] = pd.to_datetime(ledger["__decision_ts__"], utc=True, errors="raise")
    ledger["__label_available_at__"] = pd.to_datetime(ledger["__label_available_at__"], utc=True, errors="coerce")
    k0, map_audit = r4._replay(ledger, mu1=MU1, mu0=MU0, admission=("absolute", ADMISSION))
    monthly, era, summary = r4._metrics(k0, "diagnostic")
    return k0, monthly, era, summary | {"map_complete_months": int(map_audit["status"].eq("complete").sum())}


def _assert_targetfree_parity(base: pd.DataFrame, candidate: pd.DataFrame, *, head: str) -> dict[str, Any]:
    column = "conversion_score" if head == "O" else "opportunity_raw_score"
    left = base.loc[:, ["candidate_id", column]].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    right = candidate.loc[:, ["candidate_id", column]].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    identities = left["candidate_id"].equals(right["candidate_id"])
    delta = float(np.nanmax(np.abs(_finite(left[column]) - _finite(right[column])))) if identities and len(left) else float("nan")
    if not identities or not np.isclose(delta, 0.0, rtol=0.0, atol=2e-6):
        raise AssertionError(f"{head} family dropout changed frozen other-head output")
    return {"dropped_head": head, "unchanged_upstream_or_downstream_field": column, "candidate_identity_exact": identities, "max_abs_delta": delta}


def _outer_head_prediction(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    head: str,
    fields: Sequence[str],
    target: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-fit exactly one frozen outer head for each held month.

    The original C59 producer refits an O model, a C model, and six inner
    OOF models per month.  For a *family dropout* we must only change one
    outer head: the other frozen source score is copied byte-for-byte, and
    the final K0 map is rebuilt prequentially below.  This is equivalent for
    the attribution question while avoiding a 20-arm nested re-training job.
    """
    if head not in {"O", "C"}:
        raise ValueError(head)
    score_column = "opportunity_raw_score" if head == "O" else "conversion_score"
    source = reference.copy().reset_index(drop=True)
    by_id = frame.set_index("candidate_id", verify_integrity=True)
    score = np.full(len(source), np.nan, dtype=np.float32)
    audit: list[dict[str, Any]] = []
    for held_month, positions in source.groupby("held_month", sort=True).groups.items():
        indexer = np.asarray(list(positions), dtype=int)
        held_ids = source.iloc[indexer]["candidate_id"].to_numpy()
        held = by_id.loc[held_ids].reset_index()
        start = pd.Timestamp(f"{held_month}-01", tz="UTC")
        train = frame.loc[
            frame["__decision_ts__"].lt(start)
            & frame["__label_available_at__"].lt(start)
            & r1._valid_label(frame)
        ].copy()
        month_index = (start.year - 2024) * 12 + (start.month - 5)
        record = {
            "head": head, "held_month": str(held_month), "status": "complete",
            "held_rows": int(len(held)), "outer_train_rows": int(len(train)),
            "max_train_label_available_at": train["__label_available_at__"].max().isoformat() if len(train) else None,
            "field_count": int(len(fields)),
        }
        if not train["__label_available_at__"].lt(start).all():
            raise AssertionError("head-only diagnostic consumed unresolved training labels")
        if head == "O":
            y = r1._event(train, r3.SPEC)
            if len(train) < r1.MIN_OUTER_TRAIN_ROWS or np.unique(y).size < 2:
                raise RuntimeError(f"insufficient frozen O support for {held_month}")
            x_train, medians = r1._matrix(train, fields)
            x_held, _ = r1._matrix(held, fields, medians)
            model = r2._binary_config(r2.FROZEN_CONFIG, r3b.O_SEED + 20_000 + month_index)
            model.fit(x_train, y, sample_weight=r2._weights(train, r3.SPEC, "uniform"))
            value = model.predict_proba(x_held)[:, 1]
            record["conditional_train_rows"] = int(r1._event(train, r3.SPEC).sum())
        else:
            c_train = train.loc[r1._event(train, r3.SPEC).astype(bool)].copy()
            y = r3._target(c_train, target)
            if len(c_train) < r3.MIN_C_ROWS or np.unique(y).size < 2:
                raise RuntimeError(f"insufficient frozen C support for {held_month}")
            x_train, medians = r1._matrix(c_train, fields)
            x_held, _ = r1._matrix(held, fields, medians)
            model = r3._model(target, r3b.C_SEED + 30_000 + month_index)
            model.fit(x_train, y, sample_weight=r3._c_weights(c_train, "uniform"))
            value = r3._predict(model, target, x_held)
            record["conditional_train_rows"] = int(len(c_train))
        score[indexer] = np.asarray(value, dtype=np.float32)
        audit.append(record)
    if not np.isfinite(score).all():
        raise AssertionError(f"head-only {head} replay left an unscored held row")
    source[score_column] = score
    return source, pd.DataFrame(audit)


def _family_dropouts(
    frame: pd.DataFrame, o_fields: Sequence[str], c_fields: Sequence[str], inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target = next(item for item in r3.TARGETS if item.name == "C3_normalized_regret")
    anchors = frame.loc[:, ["candidate_id", r4.P0_ANCHOR]].copy()
    reference = pd.read_parquet(C59)
    # First prove the direct outer-head replay reproduces the immutable C59
    # raw scores.  The result is an exact-seed/feature-contract receipt, not
    # a new economic comparison.
    direct_o, audit_o = _outer_head_prediction(frame, reference, head="O", fields=o_fields, target=target)
    direct_c, audit_c = _outer_head_prediction(frame, reference, head="C", fields=c_fields, target=target)
    for head, direct in (("O", direct_o), ("C", direct_c)):
        column = "opportunity_raw_score" if head == "O" else "conversion_score"
        delta = np.abs(_finite(reference[column]) - _finite(direct[column]))
        if not np.isclose(delta, 0.0, rtol=0.0, atol=2e-6, equal_nan=True).all():
            raise AssertionError(f"direct outer {head} head does not reproduce frozen C59 {column}")
    base = reference.copy()
    _, base_monthly, _, base_summary = _k0_from_prediction(base, anchors)
    base_summary["arm"] = "full_frozen_replay"
    rows = [base_summary]
    monthly_rows = [base_monthly.assign(arm="full_frozen_replay")]
    parity_rows = [{"dropped_head": "none", "unchanged_upstream_or_downstream_field": "both", "candidate_identity_exact": True, "max_abs_delta": 0.0}]
    head_audits = [audit_o.assign(arm="full_frozen_replay", replay_kind="direct_O_parity"), audit_c.assign(arm="full_frozen_replay", replay_kind="direct_C_parity")]
    for head, fields in (("O", tuple(o_fields)), ("C", tuple(c_fields))):
        for family in sorted(inventory.loc[inventory["head"].eq(head), "family"].unique()):
            dropped = tuple(field for field in fields if _family(field) == family)
            retained = tuple(field for field in fields if field not in dropped)
            if not dropped or len(retained) < 5:
                continue
            candidate, head_audit = _outer_head_prediction(frame, reference, head=head, fields=retained, target=target)
            parity = _assert_targetfree_parity(base, candidate, head=head)
            parity.update({"family": family, "removed_features": len(dropped)})
            parity_rows.append(parity)
            _, monthly, _, summary = _k0_from_prediction(candidate, anchors)
            name = f"drop_{head}_{family}"
            summary.update({"arm": name, "dropped_head": head, "dropped_family": family, "removed_features": len(dropped), "retained_features": len(retained)})
            for key in ("net_2025", "net_2026", "mean_net_bps_per_trade", "total_net_bps", "selected", "worst_month", "mean_cvar10"):
                if key in base_summary and key in summary:
                    summary[f"delta_{key}"] = float(summary[key] - base_summary[key])
            rows.append(summary)
            monthly_rows.append(monthly.assign(arm=name, dropped_head=head, dropped_family=family))
            head_audits.append(head_audit.assign(arm=name, replay_kind=f"drop_{head}", dropped_head=head, dropped_family=family))
    return pd.DataFrame(rows), pd.concat(monthly_rows, ignore_index=True), pd.DataFrame(parity_rows), pd.concat(head_audits, ignore_index=True), base_summary


def _conditional_c_mda(frame: pd.DataFrame, c_fields: Sequence[str]) -> pd.DataFrame:
    target = next(item for item in r3.TARGETS if item.name == "C3_normalized_regret")
    local = frame.loc[
        r1._valid_label(frame)
        & r1._event(frame, r3.SPEC).astype(bool)
        & frame["__decision_ts__"].ge(DEV_START)
        & frame["__decision_ts__"].lt(DEV_END)
    ].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    edges = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    rows: list[dict[str, Any]] = []
    for fold in range(r1.INNER_SPLITS):
        valid = local.iloc[int(edges[fold + 1]):int(edges[fold + 2])].copy()
        if len(valid) < MIN_BUCKET_ROWS:
            continue
        start = valid["__decision_ts__"].min()
        fit = local.loc[local["__label_available_at__"].lt(start)].copy()
        y = r3._target(fit, target)
        if len(fit) < r3.MIN_C_ROWS or np.unique(y).size < 2:
            continue
        x_fit, medians = r1._matrix(fit, c_fields)
        x_valid, _ = r1._matrix(valid, c_fields, medians)
        model = r3._model(target, r3b.C_SEED + fold)
        model.fit(x_fit, y, sample_weight=r3._c_weights(fit, "uniform"))
        base = r3._predict(model, target, x_valid)
        valid["mfe_bucket"] = pd.cut(r1._finite(valid["mfe_6h_bps"]), [250., 350., 500., 750., np.inf], labels=["250_350", "350_500", "500_750", "750_plus"], right=False)
        for bucket, part in valid.groupby("mfe_bucket", observed=False):
            positions = part.index.to_numpy(int)
            # valid was reset-indexed, so its labels are row positions here.
            positions = np.searchsorted(valid.index.to_numpy(), positions)
            if len(positions) < MIN_BUCKET_ROWS:
                continue
            base_objective = r3b._c_objective(valid.iloc[positions], base[positions])
            rng = np.random.default_rng(SEED + 30_000 + fold + len(str(bucket)))
            for field in c_fields:
                altered = x_valid.copy()
                altered.loc[altered.index[positions], field] = rng.permutation(altered.iloc[positions][field].to_numpy())
                score = r3._predict(model, target, altered)
                rows.append({"head": "C", "feature": field, "fold": fold, "validation_start": start, "mfe_bucket": str(bucket), "rows": int(len(positions)), "conditional_mda_delta": float(base_objective - r3b._c_objective(valid.iloc[positions], score[positions])), "base_objective": float(base_objective)})
    return pd.DataFrame(rows)


def _o_lift_and_mda_by_volatility(frame: pd.DataFrame, o_fields: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The April--December development slice does not leave enough resolved
    # history for an O250/H6 fit at the first chronological inner folds.  Use
    # four predeclared later 3-month diagnostic holds instead.  Volatility
    # quintile edges are fit on the preceding *training* population, never on
    # the held slice, so this remains a causal conditional diagnostic rather
    # than a cross-period normalization trick.
    local = frame.loc[r1._valid_label(frame)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    folds = (
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-07-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-04-01T00:00:00Z"),
    )
    lifts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for fold, start in enumerate(folds):
        stop = min(start + pd.DateOffset(months=3), pd.Timestamp("2026-08-01T00:00:00Z"))
        valid = local.loc[local["__decision_ts__"].ge(start) & local["__decision_ts__"].lt(stop)].copy().reset_index(drop=True)
        if len(valid) < MIN_BUCKET_ROWS:
            continue
        fit = local.loc[local["__label_available_at__"].lt(start)].copy()
        y_fit = r1._event(fit, r3.SPEC)
        if len(fit) < r1.MIN_OUTER_TRAIN_ROWS or np.unique(y_fit).size < 2:
            continue
        x_fit, medians = r1._matrix(fit, o_fields)
        x_valid, _ = r1._matrix(valid, o_fields, medians)
        model = r2._binary_config(r2.FROZEN_CONFIG, SEED + fold)
        model.fit(x_fit, y_fit, sample_weight=r2._weights(fit, r3.SPEC, "uniform"))
        base = model.predict_proba(x_valid)[:, 1]
        fit_vol = _finite(fit[VOLATILITY_PROXY])
        fit_vol = fit_vol[np.isfinite(fit_vol)]
        cut_points = np.unique(np.quantile(fit_vol, np.linspace(0.0, 1.0, 6))) if len(fit_vol) else np.array([])
        if len(cut_points) < 6:
            # Constant/non-varying volatility cannot establish an informative
            # causal quintile.  Skip this fold rather than derive held edges.
            continue
        cut_points[0], cut_points[-1] = -np.inf, np.inf
        valid["volatility_bucket"] = pd.cut(
            _finite(valid[VOLATILITY_PROXY]), cut_points,
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"], include_lowest=True,
        )
        y_valid = r1._event(valid, r3.SPEC)
        for bucket, part in valid.groupby("volatility_bucket", observed=False):
            positions = np.searchsorted(valid.index.to_numpy(), part.index.to_numpy(int))
            if len(positions) < MIN_BUCKET_ROWS or np.unique(y_valid[positions]).size < 2:
                continue
            rank = pd.Series(base[positions]).rank(method="first", pct=True).to_numpy(float)
            selected = rank >= .80
            prevalence = float(y_valid[positions].mean())
            precision = float(y_valid[positions][selected].mean()) if selected.any() else float("nan")
            lift = precision / prevalence if prevalence > 0 else float("nan")
            base_objective = r2._mda_objective(y_valid[positions], base[positions])
            lifts.append({"head": "O", "fold": fold, "validation_start": start, "validation_stop": stop, "volatility_proxy": VOLATILITY_PROXY, "volatility_bucket": str(bucket), "rows": int(len(positions)), "opportunity_prevalence": prevalence, "precision_top20": precision, "lift_top20": lift, "base_objective": base_objective, "quintile_edges_fit_prequentially": True})
            rng = np.random.default_rng(SEED + 40_000 + fold + len(str(bucket)))
            for field in o_fields:
                altered = x_valid.copy()
                altered.loc[altered.index[positions], field] = rng.permutation(altered.iloc[positions][field].to_numpy())
                score = model.predict_proba(altered)[:, 1]
                rows.append({"head": "O", "feature": field, "fold": fold, "validation_start": start, "validation_stop": stop, "volatility_bucket": str(bucket), "rows": int(len(positions)), "conditional_mda_delta": float(base_objective - r2._mda_objective(y_valid[positions], score[positions])), "base_objective": float(base_objective), "quintile_edges_fit_prequentially": True})
    return pd.DataFrame(lifts), pd.DataFrame(rows)


def _overlap(inventory: pd.DataFrame) -> pd.DataFrame:
    pivot = inventory.assign(present=True).pivot_table(index="feature", columns="head", values="present", aggfunc="any", fill_value=False)
    for head in ("O", "C"):
        if head not in pivot:
            pivot[head] = False
    output = pivot.reset_index()
    output["membership"] = np.where(output["O"] & output["C"], "shared", np.where(output["O"], "O_only", "C_only"))
    families = inventory.drop_duplicates("feature").loc[:, ["feature", "family"]]
    return output.merge(families, on="feature", how="left", validate="one_to_one").sort_values(["membership", "family", "feature"], kind="stable").reset_index(drop=True)


def _table(frame: pd.DataFrame, columns: Sequence[str], limit: int | None = None) -> str:
    local = frame.loc[:, [column for column in columns if column in frame]].copy()
    if limit is not None:
        local = local.head(limit)
    if local.empty:
        return "_No supported rows._"
    # Keep the receipt self-contained: ``DataFrame.to_markdown`` pulls the
    # optional ``tabulate`` package, which is not part of Ares' frozen runtime.
    # This minimal renderer is sufficient for immutable audit tables.
    headers = [str(column) for column in local.columns]
    rows = [
        [str(value).replace("|", "\\|").replace("\n", " ") for value in record]
        for record in local.itertuples(index=False, name=None)
    ]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(record) + " |" for record in rows),
    ])


def _report(out: Path, *, summary: pd.DataFrame, blacklist: pd.DataFrame, source_summary: pd.DataFrame, overlap: pd.DataFrame, dropout: pd.DataFrame, k0_bands: pd.DataFrame, compact: dict[str, Any], manifest: dict[str, Any]) -> None:
    family_summary = summary.groupby(["head", "family"], as_index=False).agg(features=("feature", "count"), hard_blacklisted=("hard_blacklist", "sum"), max_psi=("max_psi_vs_dev", "max"), min_coverage=("min_month_coverage", "min"))
    lines = [
        "# Frozen short O45/C59 feature-contract portability audit", "",
        "This is a robustness and attribution receipt.  It does not select a new O/C/K0 winner, alter the +75 bps threshold, or grant a compact contract any canonical/live authority.", "",
        "## Contract-family inventory", "", _table(family_summary, list(family_summary.columns)), "",
        "## Portability blacklist", "", _table(blacklist, ["head", "feature", "family", "blacklist_reason", "coverage_2025", "coverage_2026", "min_month_coverage", "zero_variance_months_later"], 50), "",
        "## Source reliability scope", "", "Historical finite availability is measured below. No short-side live inference contract is sealed, so no field is claimed live-source-reliable yet.", "", _table(source_summary, list(source_summary.columns)), "",
        "## O/C overlap", "", _table(overlap.groupby(["membership", "family"], as_index=False).size(), ["membership", "family", "size"]), "",
        "## Diagnostic family dropout", "", _table(dropout, ["arm", "dropped_head", "dropped_family", "removed_features", "net_2025", "net_2026", "mean_net_bps_per_trade", "delta_mean_net_bps_per_trade", "total_net_bps", "delta_total_net_bps", "worst_month", "mean_cvar10"], 80), "",
        "## Frozen K0 band reference", "", _table(k0_bands, list(k0_bands.columns)), "",
        "## Predeclared compact contracts", "", "```json", json.dumps(compact, indent=2), "```", "",
        "## Contract", "", "```json", json.dumps(manifest, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_FEATURE_CONTRACT_PORTABILITY_REPORT.md").write_text("\n".join(lines))


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    frame, o_fields, c_fields, inventory, source_hashes = _load()
    all_features = tuple(sorted(set(o_fields) | set(c_fields)))
    monthly, references = _month_feature_audit(frame, all_features, inventory)
    age_coverage, liquidity_coverage, tier_reliability = _symbol_tier_coverage(frame, all_features, inventory)
    summary = _summarise_portability(monthly, inventory, tier_reliability)
    source_reliability, source_reliability_summary = _source_reliability(monthly, inventory, tier_reliability)
    stability = _mda_stability(o_fields, c_fields, inventory)
    summary = summary.merge(stability.loc[:, ["head", "feature", "mda_mean", "mda_min", "mda_positive_folds", "mda_folds", "mda_positive_share", "rank"]], on=["head", "feature"], how="left", validate="one_to_one")
    pairs_o, clusters_o = _redundancy(frame, o_fields, "O")
    pairs_c, clusters_c = _redundancy(frame, c_fields, "C")
    pairs, clusters = pd.concat([pairs_o, pairs_c], ignore_index=True), pd.concat([clusters_o, clusters_c], ignore_index=True)
    compact = _compact_contracts(summary, stability, clusters, o_fields, c_fields)
    overlap = _overlap(inventory)
    c_mda = _conditional_c_mda(frame, c_fields)
    o_lift, o_mda = _o_lift_and_mda_by_volatility(frame, o_fields)
    dropout, dropout_monthly, dropout_parity, dropout_head_fit_audit, baseline = _family_dropouts(frame, o_fields, c_fields, inventory)
    k0_bands = pd.read_parquet(FROZEN_DIAGNOSTICS / "k0_ev_band_scorecard.parquet")
    blacklist = summary.loc[summary["hard_blacklist"].fillna(False)].copy()
    out.mkdir(parents=True)
    monthly.to_parquet(out / "feature_monthly_portability.parquet", index=False, compression="zstd")
    references.to_parquet(out / "feature_development_reference.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "feature_portability_summary.parquet", index=False, compression="zstd")
    blacklist.to_parquet(out / "feature_portability_blacklist.parquet", index=False, compression="zstd")
    age_coverage.to_parquet(out / "feature_coverage_by_symbol_age.parquet", index=False, compression="zstd")
    liquidity_coverage.to_parquet(out / "feature_coverage_by_liquidity_pressure_proxy.parquet", index=False, compression="zstd")
    tier_reliability.to_parquet(out / "feature_source_reliability_proxy.parquet", index=False, compression="zstd")
    source_reliability.to_parquet(out / "feature_source_lineage_and_reliability.parquet", index=False, compression="zstd")
    source_reliability_summary.to_parquet(out / "source_reliability_summary.parquet", index=False, compression="zstd")
    stability.to_parquet(out / "feature_existing_mda_stability.parquet", index=False, compression="zstd")
    pairs.to_parquet(out / "feature_redundancy_pairs.parquet", index=False, compression="zstd")
    clusters.to_parquet(out / "feature_redundancy_clusters.parquet", index=False, compression="zstd")
    overlap.to_parquet(out / "o_c_feature_overlap.parquet", index=False, compression="zstd")
    c_mda.to_parquet(out / "c_conditional_mda_by_mfe_bucket.parquet", index=False, compression="zstd")
    o_lift.to_parquet(out / "o_lift_by_volatility_bucket.parquet", index=False, compression="zstd")
    o_mda.to_parquet(out / "o_conditional_mda_by_volatility_bucket.parquet", index=False, compression="zstd")
    dropout.to_parquet(out / "family_dropout_metrics.parquet", index=False, compression="zstd")
    dropout_monthly.to_parquet(out / "family_dropout_monthly.parquet", index=False, compression="zstd")
    dropout_parity.to_parquet(out / "family_dropout_parity.parquet", index=False, compression="zstd")
    dropout_head_fit_audit.to_parquet(out / "family_dropout_head_fit_audit.parquet", index=False, compression="zstd")
    (out / "predeclared_compact_contracts.json").write_text(json.dumps(compact, indent=2) + "\n")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short",
        "scope": "frozen O45/C59 portability, redundancy, conditional diagnostic MDA and head-family attribution; no economic feature reselection",
        "frozen_contract": {
            "P0": "F90 target-free", "O": {"definition": "MFE_6h > 250 bps", "features": list(o_fields), "weights": "uniform", "calibration": "Platt", "geometry": "frozen"},
            "C": {"target": "C3 normalized regret conditional on O-positive rows", "features": list(c_fields), "weights": "uniform", "geometry": "frozen"},
            "K0": "isotonic mu1 + P0-anchor quintile mu0 k=500", "admission": "absolute expected policy net >=75 bps",
        },
        "portability": {"development_window": [DEV_START.isoformat(), DEV_END.isoformat()], "target_free": True, "hard_blacklist": "sustained 2025 or 2026 coverage <90% or variance near zero in >=2 later months", "redundancy": {"primary_abs_spearman_gt": REDUNDANCY_PRIMARY, "secondary_abs_spearman_gt": REDUNDANCY_SECONDARY}, "liquidity_tier": "historical target-free trade-size-to-L1-depth pressure proxy; no absolute depth field exists in frozen source", "live_source_reliability": "not evidenced: this is a short research stack without a sealed short live inference source/receipt contract; historical coverage must not be read as a live SLO"},
        "dropout": {"diagnostic_only": True, "model_geometry": "frozen outer O/C geometry and seeds; only the dropped head is refit", "K0": "strict-prequential isotonic/anchor5-k500/abs75", "selection_prohibited": True, "cross_head_parity": "O dropout preserves exact C59 C score; C dropout preserves exact C59 O score", "direct_outer_parity": "both fresh full-contract outer heads must reproduce immutable C59 raw scores before any dropout"},
        "conditional_diagnostics": {"C": "existing C3 MDA objective within ex-post MFE buckets; MFE never model input", "O": f"existing O MDA objective and Lift@20 within target-free {VOLATILITY_PROXY} quintiles"},
        "compact_contracts": "predeclared stability/portability-only challengers for a future untouched evaluation; not fitted or promoted here",
        "sources": {"c59_prediction_sha256": _sha256(C59), "round2_manifest_sha256": _sha256(ROUND2 / "run_manifest.json"), "round3b_manifest_sha256": _sha256(ROUND3B / "run_manifest.json"), "frozen_diagnostics_manifest_sha256": _sha256(FROZEN_DIAGNOSTICS / "run_manifest.json"), **source_hashes},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _report(out, summary=summary, blacklist=blacklist, source_summary=source_reliability_summary, overlap=overlap, dropout=dropout, k0_bands=k0_bands, compact=compact, manifest=manifest)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(run(args.out))


if __name__ == "__main__":
    main()
