#!/usr/bin/env python3
"""Transport-first raw leaf archetypes, clusters, and zero-exposure GAM OOF.

This runner addresses the observed failure of frozen family IDs.  It builds
recurrence-first archetypes from the monthly raw leaf catalogues, softly maps
new leaves to those archetypes, retains unmatched contribution mass, then
clusters the transported archetype exposures.  Only after that contract is
frozen does it fit cluster-specific varying-coefficient GAMs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.cofiring_economic_clusters import (  # noqa: E402
    discover_best_contract,
    materialize_memberships,
    pairwise_cofiring_similarity,
    refit_contract,
)
from extreme_price_movements.structural_archetypes import (  # noqa: E402
    build_recurrent_archetypes,
    match_catalog_to_archetypes,
    materialize_row_archetype_exposures,
)
from scripts.run_tp6_sl4_frozen_cluster_residual import _load  # noqa: E402


SIDE = "long"
DEV_END = pd.Timestamp("2025-01-01", tz="UTC")
SEED = 20260814
TOP_N = 3
ARCHETYPE_MIN_FOLDS = 3
ARCHETYPE_MIN_SIGN = 0.80
ARCHETYPE_MIN_FREQ = 0.005
ARCHETYPE_SEPARATED_GAP = 2
MATCH_TEMPERATURE = 0.08
MATCH_THRESHOLD = 0.55
CLUSTER_ACTIVE_THRESHOLD = 0.10
CLUSTER_REPORT_ACTIVE_THRESHOLD = 0.05
PLAUSIBLE_MATCH_THRESHOLD = 0.25
TRANSPORT_MIN_COVERAGE = 0.70
TRANSPORT_MAX_FAILED_FRACTION = 0.0
GAM_CONTEXT_CAP = 12
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
GAMMAS = (0.25, 0.50, 1.00)
DEFAULT_BASE = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_base_20260811_v1.parquet"
DEFAULT_FAMILY = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/meta_family_contribution_matrix.parquet"
DEFAULT_META = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_meta_pool_regime_20260811_v1.parquet"
DEFAULT_RAW = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/strict_base_reasoning"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v1"


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 32 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def _prep(frame: pd.DataFrame, fields: Sequence[str], med: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    z = frame.reindex(columns=list(fields)).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = z.median().fillna(0.0)
        med.attrs["scale"] = ((z - med).abs().median().replace(0.0, 1.0).fillna(1.0)).to_dict()
    scale = pd.Series(med.attrs.get("scale", {}), dtype=float).reindex(fields).fillna(1.0)
    return ((z.fillna(med).fillna(0.0) - med) / scale).clip(-20.0, 20.0).astype("float32"), med


def _fit_vc(
    train: pd.DataFrame,
    held: pd.DataFrame,
    fields: Sequence[str],
    exposure_train: np.ndarray,
    exposure_held: np.ndarray,
    membership_train: np.ndarray,
    residual_train: np.ndarray,
    *,
    intercept: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if fields:
        xtr, med = _prep(train, fields)
        xte, _ = _prep(held, fields, med)
        spline = SplineTransformer(n_knots=3, degree=2, knots="quantile", extrapolation="linear", include_bias=False)
        btr = spline.fit_transform(xtr)
        bte = spline.transform(xte)
    else:
        btr = np.zeros((len(train), 0), dtype=np.float32)
        bte = np.zeros((len(held), 0), dtype=np.float32)
    etr = np.clip(np.asarray(exposure_train, float), 0.0, 20.0)
    ete = np.clip(np.asarray(exposure_held, float), 0.0, 20.0)
    base_tr = np.column_stack([etr, etr[:, None] * btr])
    base_te = np.column_stack([ete, ete[:, None] * bte])
    if intercept:
        design_tr = np.column_stack([np.ones(len(etr)), base_tr])
        design_te = np.column_stack([np.ones(len(ete)), base_te])
    else:
        design_tr, design_te = base_tr, base_te
    w = np.clip(np.asarray(membership_train, float), 0.0, 1.0)
    ok = np.isfinite(residual_train) & np.isfinite(w) & (w > 1e-8)
    if int(ok.sum()) < 128 or float(w[ok].sum()) < 64.0:
        return np.zeros(len(held), float), np.zeros(len(train), float)
    # The principal specification must be exactly zero when exposure is zero.
    # Therefore it may not contain Ridge's implicit intercept.  The ablation
    # has an explicit all-ones column and likewise disables the estimator's
    # implicit intercept so the two specifications are identifiable.
    model = Ridge(alpha=20.0, fit_intercept=False)
    model.fit(design_tr[ok], np.asarray(residual_train, float)[ok], sample_weight=w[ok])
    return (
        np.clip(np.asarray(model.predict(design_te), float), -1000.0, 1000.0),
        np.clip(np.asarray(model.predict(design_tr), float), -1000.0, 1000.0),
    )


def _weighted_cmi(frame: pd.DataFrame, fields: Sequence[str], residual: np.ndarray, membership: np.ndarray, cap: int) -> tuple[list[str], pd.DataFrame]:
    from extreme_price_movements.conditional_cluster_residual import conditional_mi_scores

    audit = conditional_mi_scores(frame, fields, residual, membership)
    selected = audit.head(cap).feature.astype(str).tolist() if not audit.empty else []
    if not audit.empty:
        audit = audit.assign(selected=audit.feature.isin(selected))
    return selected, audit


def _metrics(frame: pd.DataFrame, score_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    glob, monthly, stability = [], [], []
    for col in score_cols:
        for tail in TAILS:
            n = max(1, int(math.ceil(len(frame) * tail)))
            top = frame.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            glob.append({"arm": col, "scope": "global_2025", "tail": tail, "trades": n, "gross_bps_per_trade": float(top.gross_bps.mean()), "net_bps_per_trade": float(top.net_bps.mean()), "rank_ic": _rank_ic(frame[col].to_numpy(float), frame.net_bps.to_numpy(float))})
        vals, ics = [], []
        for month, block in frame.groupby("month", sort=True):
            n = max(1, int(math.ceil(len(block) * 0.05)))
            top = block.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            vals.append(float(top.net_bps.mean())); ics.append(_rank_ic(block[col].to_numpy(float), block.net_bps.to_numpy(float)))
            monthly.append({"arm": col, "month": str(month), "tail": 0.05, "trades": n, "gross_bps_per_trade": float(top.gross_bps.mean()), "net_bps_per_trade": vals[-1], "rank_ic": ics[-1]})
        arr = np.asarray(vals, float); med = float(np.median(arr)); mad = float(np.median(np.abs(arr - med)))
        stability.append({"arm": col, "months": len(arr), "mean_top5_net_bps": float(np.mean(arr)), "median_top5_net_bps": med, "mad_top5_net_bps": mad, "worst_month_top5_net_bps": float(np.min(arr)), "positive_months_top5": int(np.sum(arr > 0)), "portability_score_bps": med - 0.75 * mad - max(0.0, -float(np.min(arr))), "mean_month_rank_ic": float(np.nanmean(ics))})
    return pd.DataFrame(glob), pd.DataFrame(monthly), pd.DataFrame(stability)


def _load_raw_month(raw_root: Path, month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = raw_root / f"month={month}"
    cat = pd.read_parquet(folder / "leaf_rule_catalog.parquet")
    leaves = pd.read_parquet(folder / "leaf_assignments.parquet")
    cat["fold_id"] = cat["fold_id"].astype(str)
    return cat, leaves


def _transport_candidate_metrics(
    abs_exposure: pd.DataFrame,
    labels: np.ndarray,
    month_values: Sequence[str],
    *,
    mapping_quality: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Score a candidate cluster partition using structure only.

    The metrics are intentionally independent of residual/economic outcomes.
    A cluster can be core (broadly supported) or episodic (reappearing in
    separated development periods); a contiguous one-off block is failed.
    """
    x = np.maximum(abs_exposure.to_numpy(float), 0.0)
    months = sorted(pd.Series(month_values).astype(str).unique())
    rows: list[dict[str, object]] = []
    statuses: list[str] = []
    for cluster_idx in range(int(np.max(labels)) + 1):
        members = labels == cluster_idx
        mass = x[:, members].sum(axis=1)
        active_by_month: list[float] = []
        mass_by_month: list[float] = []
        for month in months:
            block = mass[np.asarray(month_values, dtype=str) == month]
            active_by_month.append(float(np.mean(block >= CLUSTER_ACTIVE_THRESHOLD)) if len(block) else 0.0)
            mass_by_month.append(float(np.mean(block)) if len(block) else 0.0)
        active_periods = [i for i, value in enumerate(active_by_month) if value >= CLUSTER_REPORT_ACTIVE_THRESHOLD]
        segments = 0
        if active_periods:
            segments = 1 + sum(active_periods[i] - active_periods[i - 1] > 1 for i in range(1, len(active_periods)))
        coverage = float(np.mean(np.asarray(active_by_month) >= CLUSTER_REPORT_ACTIVE_THRESHOLD)) if months else 0.0
        mass_median = float(np.median(mass_by_month)) if mass_by_month else 0.0
        mass_cv = float(np.std(mass_by_month) / max(np.mean(mass_by_month), 1e-8)) if mass_by_month else 1.0
        active_mass_median = float(np.median([mass_by_month[i] for i in active_periods])) if active_periods else 0.0
        if coverage >= TRANSPORT_MIN_COVERAGE and mass_cv <= 1.0 and active_mass_median >= 0.02:
            status = "core"
        elif len(active_periods) >= 3 and segments >= 2 and active_mass_median >= 0.02:
            status = "episodic"
        else:
            status = "failed"
        statuses.append(status)
        rows.append({
            "cluster_index": cluster_idx,
            "coverage": coverage,
            "mass_coverage": mass_median,
            "active_mass_median": active_mass_median,
            "mass_cv": mass_cv,
            "active_periods": len(active_periods),
            "separated_segments": segments,
            "status": status,
        })
    audit = pd.DataFrame(rows)
    coverage = float(audit.coverage.mean()) if not audit.empty else 0.0
    mass_coverage = float(audit.mass_coverage.median()) if not audit.empty else 0.0
    mass_stability = float(np.median(1.0 / (1.0 + audit.mass_cv.to_numpy(float)))) if not audit.empty else 0.0
    core_fraction = float(np.mean(audit.status.eq("core"))) if not audit.empty else 0.0
    episodic_fraction = float(np.mean(audit.status.eq("episodic"))) if not audit.empty else 0.0
    failed_fraction = float(np.mean(audit.status.eq("failed"))) if not audit.empty else 1.0
    # Scale mass coverage because it is a fraction of total tree-path mass;
    # 20% median cluster mass is already a strong representation.
    mass_score = float(np.clip(mass_coverage / 0.20, 0.0, 1.0))
    mapping_score = float(np.clip(mapping_quality / 0.50, 0.0, 1.0))
    transport_score = float(
        0.35 * coverage
        + 0.30 * mass_score
        + 0.20 * mapping_score
        + 0.15 * mass_stability
    )
    # A partition is eligible only when every cluster is core or a genuinely
    # separated episodic recurrence.  This prevents economic scoring from
    # rescuing structurally non-transporting clusters.
    gate = bool(not audit.empty and failed_fraction <= TRANSPORT_MAX_FAILED_FRACTION and coverage >= 0.70 and mass_coverage >= 0.02 and mapping_quality >= 0.20)
    summary = {
        "transport_score": transport_score,
        "transport_gate": gate,
        "coverage": coverage,
        "mass_coverage": mass_coverage,
        "mapping_quality": float(mapping_quality),
        "mass_stability": mass_stability,
        "core_fraction": core_fraction,
        "episodic_fraction": episodic_fraction,
        "failed_fraction": failed_fraction,
    }
    return summary, audit


def _archetype_transport_detail(
    matches: pd.DataFrame,
    catalog: pd.DataFrame,
    archetypes: Sequence[object],
    *,
    plausible_threshold: float,
) -> pd.DataFrame:
    """Return per-fold/per-archetype path and contribution transport metrics."""
    key = catalog[["fold_id", "tree_index", "leaf_token", "ensemble_tree_contribution"]].copy()
    key["fold_id"] = key.fold_id.astype(str)
    key["leaf_token"] = key.leaf_token.astype(str)
    m = matches.copy()
    m["fold_id"] = m.fold_id.astype(str)
    m["leaf_token"] = m.leaf_token.astype(str)
    m = m.merge(key, on=["fold_id", "tree_index", "leaf_token"], how="left", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for fold, block in m.groupby("fold_id", sort=True):
        total_mass = float(np.abs(block.ensemble_tree_contribution.to_numpy(float)).sum())
        for arch in archetypes:
            assigned_mass = 0.0
            weighted_similarity = 0.0
            max_similarity = 0.0
            plausible = 0
            for j in range(1, 4):
                aid = block.get(f"top{j}_archetype")
                if aid is None:
                    continue
                mask = aid.astype(object).eq(arch.archetype_id).to_numpy()
                prob = block[f"top{j}_probability"].to_numpy(float)
                sim = block[f"top{j}_similarity"].to_numpy(float)
                mass = np.abs(block.ensemble_tree_contribution.to_numpy(float)) * prob
                assigned_mass += float(mass[mask].sum())
                weighted_similarity += float((mass[mask] * sim[mask]).sum())
                if mask.any():
                    max_similarity = max(max_similarity, float(np.max(sim[mask])))
                    plausible += int(np.sum(mask & (sim >= plausible_threshold)))
            rows.append({
                "fold_id": str(fold),
                "archetype_id": arch.archetype_id,
                "recurrence_count": int(arch.recurrence_count),
                "recurrence_folds": json.dumps(list(arch.recurrence_folds)),
                "plausible_live_path_fraction": float(plausible / max(len(block), 1)),
                "max_live_path_similarity": max_similarity,
                "assigned_contribution_mass_fraction": float(assigned_mass / max(total_mass, 1e-12)),
                "contribution_weighted_similarity": float(weighted_similarity / max(assigned_mass, 1e-12)) if assigned_mass > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def _cluster_transport_detail(
    frame: pd.DataFrame,
    contracts: Sequence[object],
    *,
    development_months: Sequence[str],
    archetypes: Sequence[object],
) -> pd.DataFrame:
    """Materialize per-month transport, exposure, and entropy diagnostics."""
    rows: list[dict[str, object]] = []
    recurrence = {a.archetype_id: int(a.recurrence_count) for a in archetypes}
    for month, block in frame.groupby(frame.month.astype(str), sort=True):
        for cluster in contracts:
            prefix = f"cluster__{cluster.cluster_id}__"
            membership = block[f"{prefix}membership"].to_numpy(float)
            mass = block[f"{prefix}abs_contribution"].to_numpy(float)
            active = membership >= CLUSTER_REPORT_ACTIVE_THRESHOLD
            member_ids = [str(x) for x in cluster.family_fields]
            rows.append({
                "month": str(month),
                "cluster_id": cluster.cluster_id,
                "archetype_count": len(member_ids),
                "archetype_recurrence_min": min((recurrence.get(x, 0) for x in member_ids), default=0),
                "archetype_recurrence_median": float(np.median([recurrence.get(x, 0) for x in member_ids])) if member_ids else 0.0,
                "rows": len(block),
                "active_rows": int(active.sum()),
                "activation_frequency": float(active.mean()) if len(active) else 0.0,
                "mean_cluster_contribution_mass": float(np.mean(mass)) if len(mass) else 0.0,
                "median_cluster_contribution_mass": float(np.median(mass)) if len(mass) else 0.0,
                "mass_cv": float(np.std(mass) / max(np.mean(mass), 1e-8)) if len(mass) else 0.0,
                "conditional_mass_active": float(np.mean(mass[active])) if active.any() else 0.0,
                "matched_mass_mean": float(block.archetype_matched_mass.mean()),
                "unmatched_mass_mean": float(block.archetype_unmatched_mass.mean()),
                "membership_entropy_mean": float(block.cluster_path_entropy.mean()),
                "is_development": str(month) in set(map(str, development_months)),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        dev = out.loc[out.is_development].copy()
        classifications: dict[str, str] = {}
        for cluster_id, block in dev.groupby("cluster_id", sort=True):
            active_periods = block.loc[block.activation_frequency >= CLUSTER_REPORT_ACTIVE_THRESHOLD].sort_values("month")
            month_positions = [list(sorted(map(str, development_months))).index(m) for m in active_periods.month]
            segments = 0
            if month_positions:
                segments = 1 + sum(month_positions[i] - month_positions[i - 1] > 1 for i in range(1, len(month_positions)))
            coverage = float(np.mean(block.activation_frequency.to_numpy(float) >= CLUSTER_REPORT_ACTIVE_THRESHOLD))
            mass_cv = float(np.std(block.mean_cluster_contribution_mass.to_numpy(float)) / max(np.mean(block.mean_cluster_contribution_mass.to_numpy(float)), 1e-8))
            active_mass = block.loc[block.activation_frequency >= CLUSTER_REPORT_ACTIVE_THRESHOLD, "mean_cluster_contribution_mass"]
            active_mass_median = float(active_mass.median()) if not active_mass.empty else 0.0
            if coverage >= TRANSPORT_MIN_COVERAGE and mass_cv <= 1.0 and active_mass_median >= 0.02:
                status = "core"
            elif len(month_positions) >= 3 and segments >= 2 and active_mass_median >= 0.02:
                status = "episodic"
            else:
                status = "failed"
            classifications[str(cluster_id)] = status
        out["transport_class"] = out.cluster_id.astype(str).map(classifications).fillna("unknown")
    return out


def run(*, base_path: Path = DEFAULT_BASE, family_path: Path = DEFAULT_FAMILY, meta_path: Path = DEFAULT_META, raw_root: Path = DEFAULT_RAW, out: Path = DEFAULT_OUT, development_end: pd.Timestamp = DEV_END) -> Path:
    if out.exists():
        raise FileExistsError(out)
    development_end = pd.Timestamp(development_end)
    development_end = development_end.tz_localize("UTC") if development_end.tz is None else development_end.tz_convert("UTC")
    frame, _, meta_fields = _load(base_path, family_path, meta_path, development_end=development_end)
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    frame["residual_bps"] = frame.net_bps.to_numpy(float) - frame.base_expected_bps.to_numpy(float)
    raw_catalogues, raw_leaves = [], []
    for month_path in sorted(raw_root.glob("month=*/leaf_rule_catalog.parquet")):
        month = month_path.parent.name.split("=", 1)[1]
        cat, leaves = _load_raw_month(raw_root, month)
        raw_catalogues.append(cat); raw_leaves.append((month, cat, leaves))
    all_catalog = pd.concat(raw_catalogues, ignore_index=True)
    dev_catalog = all_catalog.loc[all_catalog.fold_id.astype(str).str[:4].eq("2024")].copy()
    archetypes, archetype_audit = build_recurrent_archetypes(
        dev_catalog, min_folds=ARCHETYPE_MIN_FOLDS, min_sign_consistency=ARCHETYPE_MIN_SIGN,
        min_train_frequency=ARCHETYPE_MIN_FREQ, separated_gap=ARCHETYPE_SEPARATED_GAP,
    )
    matches_all, match_summary = match_catalog_to_archetypes(all_catalog, archetypes, temperature=MATCH_TEMPERATURE, unmatched_threshold=MATCH_THRESHOLD, top_n=TOP_N)
    catalog_mass = np.abs(all_catalog.ensemble_tree_contribution.to_numpy(float))
    mapping_quality = float(np.average(matches_all.best_similarity.to_numpy(float), weights=np.maximum(catalog_mass, 1e-12)))
    archetype_transport_detail = _archetype_transport_detail(
        matches_all,
        all_catalog,
        archetypes,
        plausible_threshold=PLAUSIBLE_MATCH_THRESHOLD,
    )
    match_by_month: dict[str, pd.DataFrame] = {}
    row_parts: list[pd.DataFrame] = []
    raw_transport_rows: list[dict[str, object]] = []
    for month, cat, leaves in raw_leaves:
        m = matches_all.loc[matches_all.fold_id.astype(str).eq(month)].copy()
        feats, mass = materialize_row_archetype_exposures(leaves, cat, m, archetypes)
        row = leaves[["candidate_id", "__ts__"]].copy()
        row = pd.concat([row.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
        row_parts.append(row)
        raw_transport_rows.append({
            "month": month, "model_leaf_rows": len(cat), "candidate_rows": len(row),
            "archetype_count": len(archetypes), "mean_best_similarity": float(m.best_similarity.mean()),
            "median_best_similarity": float(m.best_similarity.median()), "mean_unmatched_leaf_probability": float(m.unmatched_probability.mean()),
            "matched_mass_mean": float(feats.archetype_matched_mass.mean()), "matched_mass_median": float(feats.archetype_matched_mass.median()),
            "rows_matched_mass_ge_50": float((feats.archetype_matched_mass >= 0.5).mean()),
            "rows_matched_mass_ge_75": float((feats.archetype_matched_mass >= 0.75).mean()),
        })
        match_by_month[month] = m
    row_features = pd.concat(row_parts, ignore_index=True).drop_duplicates(["candidate_id", "__ts__"])
    frame = frame.merge(row_features, on=["candidate_id", "__ts__"], how="left", validate="one_to_one")
    archetype_fields = [f"archetype__{a.archetype_id}__abs_contribution" for a in archetypes]
    signed_fields = [f"archetype__{a.archetype_id}__signed_contribution" for a in archetypes]
    frame[archetype_fields + signed_fields] = frame[archetype_fields + signed_fields].fillna(0.0)
    frame["archetype_matched_mass"] = frame.archetype_matched_mass.fillna(0.0)
    frame["archetype_unmatched_mass"] = frame.archetype_unmatched_mass.fillna(1.0)
    dev = frame.loc[frame.__ts__.lt(development_end) & frame.label_available_ts.lt(development_end)].copy()
    held = frame.loc[frame.__ts__.ge(development_end)].copy()
    months = sorted(dev.month.astype(str).unique())
    split = max(2, len(months) - 2); discovery_months, validation_months = months[:split], months[split:]
    disc = dev.loc[dev.month.astype(str).isin(discovery_months) & dev.archetype_matched_mass.ge(0.5)].copy()
    validation = dev.loc[dev.month.astype(str).isin(validation_months) & dev.archetype_matched_mass.ge(0.5)].copy()
    if len(disc) < 500 or len(validation) < 200:
        raise ValueError(f"insufficient matched archetype development support: discovery={len(disc)} validation={len(validation)}")
    abs_disc = disc[archetype_fields].copy(); abs_disc.columns = [a.archetype_id for a in archetypes]
    abs_val = validation[archetype_fields].copy(); abs_val.columns = [a.archetype_id for a in archetypes]
    signed_disc = disc[signed_fields].copy(); signed_disc.columns = [a.archetype_id for a in archetypes]
    # Structural transport is evaluated on development exposures only, before
    # residual/economic selection.  Candidate K values are scored using the
    # same co-firing geometry as the economic utility, but no outcomes enter
    # these transport gates.
    signed_dev_all = dev[signed_fields].copy(); signed_dev_all.columns = [a.archetype_id for a in archetypes]
    abs_dev_all = dev[archetype_fields].copy(); abs_dev_all.columns = [a.archetype_id for a in archetypes]
    sim_transport, _, _ = pairwise_cofiring_similarity(
        abs_disc,
        signed_disc,
        disc.residual_bps.to_numpy(float),
        active_threshold=CLUSTER_ACTIVE_THRESHOLD,
    )
    transport_by_k: dict[int, dict[str, float]] = {}
    transport_audits_by_k: dict[int, pd.DataFrame] = {}
    for k in (2, 3, 4, 5, 6):
        if k >= len(archetypes):
            continue
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(1.0 - sim_transport)
        summary, detail = _transport_candidate_metrics(
            abs_dev_all,
            labels,
            dev.month.astype(str).to_numpy(),
            mapping_quality=mapping_quality,
        )
        transport_by_k[k] = summary
        detail["k"] = k
        transport_audits_by_k[k] = detail
    contracts0, candidate_audit, pair_audit, validation_diff = discover_best_contract(
        abs_disc, signed_disc, disc.residual_bps.to_numpy(float), abs_val, validation.residual_bps.to_numpy(float), k_values=(2, 3, 4, 5, 6), seed=SEED,
        active_threshold=CLUSTER_ACTIVE_THRESHOLD,
        transport_by_k=transport_by_k,
        transport_weight=0.35,
    )
    selected_candidate = candidate_audit.iloc[0]
    selected_candidate_valid = bool(selected_candidate.valid_contract)
    selected_candidate_transport_gate = bool(selected_candidate.transport_gate)
    selected_k = len(contracts0)
    dev_matched = dev.loc[dev.archetype_matched_mass.ge(0.5)].copy()
    abs_dev = dev_matched[archetype_fields].copy(); abs_dev.columns = [a.archetype_id for a in archetypes]
    signed_dev = dev_matched[signed_fields].copy(); signed_dev.columns = [a.archetype_id for a in archetypes]
    contracts, pair_refit, family_audit = refit_contract(
        abs_dev,
        signed_dev,
        dev_matched.residual_bps.to_numpy(float),
        k=selected_k,
        active_threshold=CLUSTER_ACTIVE_THRESHOLD,
    )
    all_abs = frame[archetype_fields].copy(); all_abs.columns = [a.archetype_id for a in archetypes]
    cluster_feats = materialize_memberships(all_abs, contracts)
    for col in cluster_feats.columns:
        frame[col] = cluster_feats[col].to_numpy()
    cluster_ids = [c.cluster_id for c in contracts]
    selected_labels = np.full(len(archetypes), -1, dtype=int)
    for cluster_idx, contract in enumerate(contracts):
        selected_labels[np.asarray(contract.family_indices, dtype=int)] = cluster_idx
    selected_transport_summary, selected_transport_clusters = _transport_candidate_metrics(
        abs_dev_all,
        selected_labels,
        dev.month.astype(str).to_numpy(),
        mapping_quality=mapping_quality,
    )
    selected_transport_clusters["cluster_id"] = selected_transport_clusters.cluster_index.map({i: c.cluster_id for i, c in enumerate(contracts)})
    selected_transport_clusters["selected_contract_transport_gate"] = bool(selected_transport_summary["transport_gate"])
    cluster_transport_detail = _cluster_transport_detail(
        frame,
        contracts,
        development_months=sorted(dev.month.astype(str).unique()),
        archetypes=archetypes,
    )

    out.mkdir(parents=True)
    (out / "archetype_contract.json").write_text(json.dumps({
        "schema": "tp6_sl4_recurrent_structural_archetype_contract_v1", "side": SIDE,
        "development_months": sorted(dev.month.astype(str).unique()), "archetype_count": len(archetypes),
        "min_recurrence_folds": ARCHETYPE_MIN_FOLDS, "min_sign_consistency": ARCHETYPE_MIN_SIGN,
        "archetypes": [a.to_dict() for a in archetypes],
        "soft_matching": {"temperature": MATCH_TEMPERATURE, "unmatched_threshold": MATCH_THRESHOLD, "top_n": TOP_N},
        "cluster_activation": {"active_exposure_threshold": CLUSTER_ACTIVE_THRESHOLD},
    }, indent=2) + "\n")
    (out / "cofiring_cluster_contract.json").write_text(json.dumps({
        "schema": "tp6_sl4_archetype_cofiring_cluster_contract_v1", "cluster_count": len(contracts),
        "clusters": [c.to_dict() for c in contracts], "discovery_months": discovery_months, "validation_months": validation_months,
        "cluster_activation_threshold": CLUSTER_ACTIVE_THRESHOLD,
        "membership_is_exposure_not_target": True, "archetype_fields": len(archetypes),
        "transport_gate_passed": bool(selected_transport_summary["transport_gate"]),
        "selected_candidate_valid": selected_candidate_valid,
    }, indent=2) + "\n")
    archetype_audit.to_parquet(out / "archetype_recurrence_audit.parquet", index=False)
    matches_all.to_parquet(out / "leaf_archetype_match_audit.parquet", index=False)
    match_summary.to_parquet(out / "leaf_archetype_match_summary.parquet", index=False)
    archetype_transport_detail.to_parquet(out / "archetype_transport_detail.parquet", index=False)
    pd.DataFrame(raw_transport_rows).to_parquet(out / "archetype_transport_by_month.parquet", index=False)
    row_features.to_parquet(out / "archetype_row_exposures.parquet", index=False)
    candidate_audit.to_parquet(out / "cluster_candidate_audit.parquet", index=False)
    selected_transport_clusters.to_parquet(out / "selected_cluster_transport_gate.parquet", index=False)
    cluster_transport_detail.to_parquet(out / "cluster_transport_by_month.parquet", index=False)
    validation_diff.to_parquet(out / "development_validation_differentiation.parquet", index=False)
    pair_audit.to_parquet(out / "pair_similarity_discovery.parquet", index=False)
    pair_refit.to_parquet(out / "pair_similarity_refit.parquet", index=False)
    family_audit.to_parquet(out / "archetype_economic_audit.parquet", index=False)

    prediction_parts, gam_fit_rows, cluster_own_rows, context_audits = [], [], [], []
    for month in sorted(held.month.astype(str).unique()):
        cutoff = pd.Timestamp(month, tz="UTC")
        train = frame.loc[frame.__ts__.lt(cutoff) & frame.label_available_ts.lt(cutoff) & frame.archetype_matched_mass.ge(0.5)].copy()
        test = frame.loc[frame.month.astype(str).eq(month)].copy()
        if len(train) < 500 or test.empty:
            continue
        p_list, m_list = [], []
        for cluster in contracts:
            prefix = f"cluster__{cluster.cluster_id}__"
            mtr = train[f"{prefix}membership"].to_numpy(float); mte = test[f"{prefix}membership"].to_numpy(float)
            etr = train[f"{prefix}abs_contribution"].to_numpy(float); ete = test[f"{prefix}abs_contribution"].to_numpy(float)
            selected, cmi = _weighted_cmi(train, meta_fields, train.residual_bps.to_numpy(float), mtr, GAM_CONTEXT_CAP)
            if not cmi.empty:
                context_audits.append(cmi.assign(month=month, cluster_id=cluster.cluster_id, train_rows=len(train)))
            pred_zero, _ = _fit_vc(train, test, selected, etr, ete, mtr, train.residual_bps.to_numpy(float), intercept=False)
            pred_intercept, _ = _fit_vc(train, test, selected, etr, ete, mtr, train.residual_bps.to_numpy(float), intercept=True)
            p_list.append((pred_zero, pred_intercept)); m_list.append(mte)
            active = mte > 0.05
            for mode, pred in (("zero", pred_zero), ("intercept", pred_intercept)):
                qdiff = np.nan
                if active.sum() >= 64:
                    q = pd.qcut(pd.Series(pred[active]), 5, labels=False, duplicates="drop")
                    vals = test.loc[active, "net_bps"].to_numpy(float)
                    if q.nunique() >= 2:
                        qdiff = float(np.mean(vals[q.to_numpy() == q.max()]) - np.mean(vals[q.to_numpy() == q.min()]))
                cluster_own_rows.append({"month": month, "cluster_id": cluster.cluster_id, "mode": mode, "active_rows": int(active.sum()), "mean_membership": float(mte.mean()), "active_rank_ic": _rank_ic(pred[active], test.loc[active, "residual_bps"]) if active.sum() >= 32 else np.nan, "active_net_mean_bps": float(test.loc[active, "net_bps"].mean()) if active.any() else np.nan, "active_delta_q5_minus_q1_net_bps": qdiff})
            gam_fit_rows.append({"month": month, "cluster_id": cluster.cluster_id, "train_rows": len(train), "held_rows": len(test), "active_train_rows": int((mtr > 0.05).sum()), "active_held_rows": int((mte > 0.05).sum()), "selected_count": len(selected), "selected_fields": json.dumps(selected), "target": "ordinary net residual; membership is weight/exposure", "zero_at_exposure": True})
        mtx = np.column_stack(m_list)
        zero = np.column_stack([x[0] for x in p_list]); intercept = np.column_stack([x[1] for x in p_list])
        zero_agg = np.divide((mtx * zero).sum(axis=1), np.maximum(mtx.sum(axis=1), 1e-8), out=np.zeros(len(test)), where=mtx.sum(axis=1) > 1e-8)
        int_agg = np.divide((mtx * intercept).sum(axis=1), np.maximum(mtx.sum(axis=1), 1e-8), out=np.zeros(len(test)), where=mtx.sum(axis=1) > 1e-8)
        out_frame = test[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_expected_bps", "base_score", "residual_bps", "archetype_matched_mass", "archetype_unmatched_mass"]].copy()
        out_frame["archetype_vcgam_zero_residual"] = zero_agg
        out_frame["archetype_vcgam_intercept_residual"] = int_agg
        for mode, agg in (("zero", zero_agg), ("intercept", int_agg)):
            for gamma in GAMMAS:
                out_frame[f"archetype_vcgam_{mode}_gamma{int(gamma * 100):03d}"] = out_frame.base_expected_bps.to_numpy(float) + gamma * agg
        prediction_parts.append(out_frame)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    score_cols = ["base_expected_bps", *[f"archetype_vcgam_{mode}_gamma{int(g * 100):03d}" for mode in ("zero", "intercept") for g in GAMMAS]]
    glob, monthly, stability = _metrics(predictions, score_cols)
    predictions.to_parquet(out / "archetype_cluster_vcgam_oof_predictions.parquet", index=False, compression="zstd")
    glob.to_parquet(out / "metrics_global.parquet", index=False); monthly.to_parquet(out / "metrics_monthly.parquet", index=False); stability.to_parquet(out / "metrics_stability.parquet", index=False)
    pd.DataFrame(gam_fit_rows).to_parquet(out / "cluster_gam_fit_audit.parquet", index=False); pd.DataFrame(cluster_own_rows).to_parquet(out / "cluster_own_metrics.parquet", index=False)
    if context_audits:
        pd.concat(context_audits, ignore_index=True).to_parquet(out / "cluster_gam_context_selection.parquet", index=False)
    selected_context = []
    if context_audits:
        ca = pd.concat(context_audits, ignore_index=True); selected_context = ca.loc[ca.selected, "feature"].astype(str).tolist()
    correctness = {
        "schema": "tp6_sl4_archetype_cluster_vcgam_correctness_v1", "rows_scored": int(len(predictions)), "candidate_ids_unique": bool(predictions.candidate_id.is_unique), "all_scores_finite": bool(np.isfinite(predictions[score_cols].to_numpy(float)).all()), "archetypes_recurrence_first": True, "archetype_unmatched_mass_explicit": True, "contract_frozen_before_2025": True, "cluster_membership_multiplied_into_target": False, "cluster_membership_used_as_weight_or_exposure": True, "gam_zero_at_exposure_principal": True, "held_2025_outcomes_used_for_archetype_or_contract_selection": False, "target_like_context_fields_selected": sorted(set(x for x in selected_context if any(t in x.lower() for t in ("net_bps", "gross_bps", "policy_", "future", "mfe", "mae")))), "global_ranking_after_score_generation": True,
        "selected_transport_gate": selected_candidate_transport_gate,
        "selected_cluster_contract_valid": selected_candidate_valid,
        "economic_promotion_allowed": bool(selected_candidate_valid and selected_transport_summary["transport_gate"]),
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {"schema": "tp6_sl4_archetype_cluster_vcgam_oof_v1", "status": "COMPLETE_DIAGNOSTIC_NO_VALID_CONTRACT" if not selected_candidate_valid else "COMPLETE", "side": SIDE, "base": str(base_path), "raw_root": str(raw_root), "development_end": str(development_end), "development_rows": len(dev), "held_rows": len(predictions), "archetype_count": len(archetypes), "cluster_count": len(contracts), "meta_pool_count": len(meta_fields), "principal_gam": "zero at exposure: exposure*(beta + spline(context))", "ablation_gam": "intercept + exposure*(beta + spline(context))", "target": "exact_net_bps - base_expected_bps; no membership multiplication", "cluster_activation_threshold": CLUSTER_ACTIVE_THRESHOLD, "selected_candidate_valid": selected_candidate_valid, "selected_transport_gate": selected_candidate_transport_gate, "economic_promotion_allowed": bool(selected_candidate_valid and selected_transport_summary["transport_gate"]), "artifacts": sorted(x.name for x in out.iterdir())}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "TP6_SL4_ARCHETYPE_CLUSTER_VCGAM_REPORT.md").write_text("# TP6/SL4 recurrence-first archetype, cluster, and zero-exposure GAM replay\n\n" + f"Status: {'DIAGNOSTIC_ONLY_NO_VALID_TRANSPORT_CONTRACT' if not selected_candidate_valid else 'COMPLETE'}\n\n" + glob.round(3).to_string(index=False) + "\n\n## Stability\n\n" + stability.round(3).to_string(index=False) + "\n\n## Transport gate\n\n" + json.dumps(selected_transport_summary, indent=2) + "\n")
    print(json.dumps({"output": str(out), "archetypes": len(archetypes), "clusters": len(contracts), "rows": len(predictions), "global_metric_rows": len(glob), "selected_candidate_valid": selected_candidate_valid, "transport_gate": selected_candidate_transport_gate}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE); parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY); parser.add_argument("--meta", type=Path, default=DEFAULT_META); parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW); parser.add_argument("--out", type=Path, default=DEFAULT_OUT); parser.add_argument("--development-end", type=str, default=str(DEV_END))
    args = parser.parse_args()
    run(base_path=args.base, family_path=args.family, meta_path=args.meta, raw_root=args.raw_root, out=args.out, development_end=pd.Timestamp(args.development_end))
