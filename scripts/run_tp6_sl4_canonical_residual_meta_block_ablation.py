#!/usr/bin/env python3
"""Strict-OOF residual meta layer on top of canonical TP6/SL4 Base+Consensus.

This is intentionally independent of the GAM branch.  The canonical score is
treated as the current system's belief; a train-only isotonic map converts it
to expected net bps, and the new target is the remaining exact-net residual.
The feature contract is compact and mechanism-oriented:

* uncertainty: raw eight-head dispersion, probability conviction, and
  base/consensus disagreement;
* support_ood: train-support distance and missing/outlier exposure for the
  frozen context;
* drift: causal recent-vs-history distribution and covariance-break proxies;
* market_state: a small, fixed volatility/breadth/dependence/liquidity/
  funding core from the existing 73-field context.

The A-H block arms are evaluated chronologically over 2025.  All held-month
targets and outcomes are excluded from the model fit for that month.  The
resulting residual rank is blended with the canonical rank at a frozen 75/25
weight, then evaluated by pooled global top-k ranking.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS, _load, _pct  # noqa: E402

DEFAULT_PANEL = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/predictions_2025.parquet"
DEFAULT_HEADS = ROOT / "data_perp/artifacts/tp6_sl4_canonical_head_health_2025_v1/canonical_head_health_2025.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_canonical_residual_meta_block_ablation_20260808_v1"
DEFAULT_SELECTION = ROOT / "data_perp/artifacts/tp6_sl4_canonical_residual_feature_selection_20260808_v1/selected_features.json"
SEED = 20260808
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)

BLOCKS: dict[str, list[str]] = {
    "uncertainty": [
        "consensus_head_rank_std", "consensus_head_rank_mad", "consensus_head_rank_iqr",
        "consensus_head_rank_min", "consensus_head_rank_max", "consensus_head_raw_std",
        "consensus_head_agreement_fraction", "base_consensus_disagreement",
        "base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak",
        "base_probability_entropy", "base_probability_top2_margin", "base_conviction",
    ],
    "support_ood": [
        "context_missing_fraction", "context_ood_mean_abs_z", "context_ood_p95_abs_z",
        "context_ood_outlier_fraction", "context_ood_tail_fraction",
        "support_recent_distance", "support_min_margin", "support_low_tail_fraction",
    ],
    "drift": [
        "recent_context_shift", "recent_context_covariance_break", "recent_score_shift",
        "recent_head_dispersion_shift", "score_history_ood", "recent_volatility_shift",
        "recent_breadth_shift", "recent_liquidity_shift",
    ],
    "market_state": [
        "median_rvol_z", "median_volume_z", "mkt_atr_expansion_1h", "mkt_atr_expansion_4h",
        "q_iqr__bars_in_high_vol_state_log_norm", "breadth_dispersion", "cs_dispersion_ret_24h",
        "cs_dispersion_ret_4h", "avg_pair_corr_24h", "corr_concentration_24h",
        "correlation_breakdown_dispersion", "median_spread_bps", "ob_depth_l10_to_qv_24h",
        "amihud_z_peer_resid", "liquidity_ratio_peer_resid", "xs_dispersion__amihud_z",
        "fund_abs_z", "fund_abs_z_14d", "fund_abs_z_mkt_resid",
        "mkt_abs_ret_per_oi_drop_1h", "oiw_intensity_entry_dist_7d_atr",
    ],
}

ARM_BLOCKS: dict[str, tuple[str, ...]] = {
    "A_control": (),
    "B_uncertainty": ("uncertainty",),
    "C_support_ood": ("support_ood",),
    "D_drift": ("drift",),
    "E_market_state": ("market_state",),
    "F_uncertainty_ood": ("uncertainty", "support_ood"),
    "G_uncertainty_ood_drift": ("uncertainty", "support_ood", "drift"),
    "H_full": ("uncertainty", "support_ood", "drift", "market_state"),
}


def _context_fields(panel: pd.DataFrame) -> list[str]:
    excluded = {
        "candidate_id", "__ts__", "label_available_ts", "side_name", "month",
        "exact_net_bps", "exact_gross_bps", "label_valid", "base_score",
        "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak",
        "base_expected_bps", "base_rank", "consensus_rank", "residual_rank",
        "base_plus_consensus25", "base_plus_residual25", "full_base_consensus_residual",
        "base_only", "consensus_only", "residual_only", "fold_month",
    }
    return [c for c in panel.columns if c not in excluded and pd.api.types.is_numeric_dtype(panel[c])]


def _safe_numeric(frame: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    return frame.reindex(columns=list(cols)).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _robust_stats(train: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    z = _safe_numeric(train, cols)
    med = z.median().fillna(0.0)
    mad = (z - med).abs().median().replace(0.0, 1.0).fillna(1.0)
    q01 = z.quantile(0.01).fillna(med - 6.0 * mad)
    q99 = z.quantile(0.99).fillna(med + 6.0 * mad)
    return med, mad, q01, q99


def _base_probability_features(frame: pd.DataFrame) -> pd.DataFrame:
    p = _safe_numeric(frame, ["r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak"]).to_numpy(float)
    p = np.clip(np.nan_to_num(p, nan=1.0 / 3.0), 1e-6, 1.0)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    order = np.sort(p, axis=1)
    return pd.DataFrame({
        "base_probability_entropy": -np.sum(p * np.log(p), axis=1),
        "base_probability_top2_margin": order[:, -1] - order[:, -2],
        "base_conviction": p[:, 2] - 0.5 * p[:, 1],
    }, index=frame.index)


def _feature_frame(train: pd.DataFrame, held: pd.DataFrame, context: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build train/held features using train-only normalization and drift stats."""
    med, mad, q01, q99 = _robust_stats(train, context)
    trc = _safe_numeric(train, context)
    tec = _safe_numeric(held, context)
    tr_z = ((trc - med) / mad).clip(-20.0, 20.0)
    te_z = ((tec - med) / mad).clip(-20.0, 20.0)
    out_tr = pd.DataFrame(index=train.index)
    out_te = pd.DataFrame(index=held.index)

    for name, frame, out in (("train", train, out_tr), ("held", held, out_te)):
        p = _base_probability_features(frame)
        for c in p.columns:
            out[c] = p[c].to_numpy(float)
        out["base_score"] = pd.to_numeric(frame["base_score"], errors="coerce").fillna(0.0).to_numpy(float)
        out["r3_meta_p_clear"] = pd.to_numeric(frame["r3_meta_p_clear"], errors="coerce").fillna(1.0 / 3.0).to_numpy(float)
        out["r3_meta_p_adverse"] = pd.to_numeric(frame["r3_meta_p_adverse"], errors="coerce").fillna(1.0 / 3.0).to_numpy(float)
        out["r3_meta_p_weak"] = pd.to_numeric(frame["r3_meta_p_weak"], errors="coerce").fillna(1.0 / 3.0).to_numpy(float)
        for c in [
            "consensus_head_rank_std", "consensus_head_rank_mad", "consensus_head_rank_iqr",
            "consensus_head_rank_min", "consensus_head_rank_max", "consensus_head_raw_std",
            "consensus_head_agreement_fraction", "base_consensus_disagreement",
        ]:
            out[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0.0).to_numpy(float)
        z = tr_z if name == "train" else te_z
        abs_z = z.abs().to_numpy(float)
        out["context_missing_fraction"] = _safe_numeric(frame, context).isna().mean(axis=1).to_numpy(float)
        out["context_ood_mean_abs_z"] = np.nanmean(abs_z, axis=1)
        out["context_ood_p95_abs_z"] = np.nanpercentile(abs_z, 95, axis=1)
        out["context_ood_outlier_fraction"] = np.nanmean(abs_z > 3.0, axis=1)
        low = (z.lt((q01 - med) / mad) | z.gt((q99 - med) / mad)).to_numpy(float)
        out["context_ood_tail_fraction"] = np.nanmean(low, axis=1)
        out["support_recent_distance"] = np.nanmean(np.minimum(np.abs(z), 6.0), axis=1)
        out["support_min_margin"] = np.nanmin(np.maximum(0.0, 3.0 - abs_z), axis=1)
        out["support_low_tail_fraction"] = np.nanmean(abs_z > 2.0, axis=1)

    # The following drift statistics are calculated from the most recent
    # available training month versus the older training history.  They are
    # constants within a held fold when appropriate, which is intentional: the
    # residual layer must be able to learn that the substrate itself has shifted.
    train_month = train.month.astype(str)
    recent_month = sorted(train_month.unique())[-1]
    recent = train_month.eq(recent_month)
    older = ~recent
    if older.sum() < 50:
        older = np.ones(len(train), dtype=bool)
    recent_z = tr_z.loc[recent]
    older_z = tr_z.loc[older]
    ctx_shift = float(np.nanmean(np.abs(recent_z.mean(axis=0).to_numpy(float) - older_z.mean(axis=0).to_numpy(float))))
    # Covariance break uses a compact, predeclared state core; no outcomes.
    core = [c for c in BLOCKS["market_state"] if c in context]
    if len(core) >= 3 and recent.sum() >= 30 and older.sum() >= 30:
        rc = tr_z.loc[recent, core].corr().fillna(0.0).to_numpy(float)
        oc = tr_z.loc[older, core].corr().fillna(0.0).to_numpy(float)
        cov_break = float(np.sqrt(np.mean((rc - oc) ** 2)))
    else:
        cov_break = 0.0
    score = pd.to_numeric(train["base_plus_consensus25"], errors="coerce").fillna(0.5).to_numpy(float)
    recent_score = float(np.nanmedian(score[recent])) if recent.any() else float(np.nanmedian(score))
    older_score = float(np.nanmedian(score[older])) if older.any() else float(np.nanmedian(score))
    score_shift = abs(recent_score - older_score)
    head_disp = pd.to_numeric(train["consensus_head_rank_std"], errors="coerce").fillna(0.0).to_numpy(float)
    head_shift = abs(float(np.nanmedian(head_disp[recent])) - float(np.nanmedian(head_disp[older])))
    recent_vol = _safe_numeric(train.loc[recent], ["median_rvol_z", "mkt_atr_expansion_4h"]).median().to_numpy(float)
    older_vol = _safe_numeric(train.loc[older], ["median_rvol_z", "mkt_atr_expansion_4h"]).median().to_numpy(float)
    vol_shift = float(np.nanmean(np.abs(recent_vol - older_vol)))
    recent_breadth = _safe_numeric(train.loc[recent], ["breadth_dispersion", "cs_dispersion_ret_24h"]).median().to_numpy(float)
    older_breadth = _safe_numeric(train.loc[older], ["breadth_dispersion", "cs_dispersion_ret_24h"]).median().to_numpy(float)
    breadth_shift = float(np.nanmean(np.abs(recent_breadth - older_breadth)))
    recent_liq = _safe_numeric(train.loc[recent], ["median_spread_bps", "ob_depth_l10_to_qv_24h", "amihud_z_peer_resid"]).median().to_numpy(float)
    older_liq = _safe_numeric(train.loc[older], ["median_spread_bps", "ob_depth_l10_to_qv_24h", "amihud_z_peer_resid"]).median().to_numpy(float)
    liq_shift = float(np.nanmean(np.abs(recent_liq - older_liq)))
    for out in (out_tr, out_te):
        out["recent_context_shift"] = ctx_shift
        out["recent_context_covariance_break"] = cov_break
        out["recent_score_shift"] = score_shift
        out["recent_head_dispersion_shift"] = head_shift
        out["score_history_ood"] = np.abs(pd.to_numeric(out["base_score"], errors="coerce") - float(np.nanmedian(pd.to_numeric(train["base_score"], errors="coerce"))))
        out["recent_volatility_shift"] = vol_shift
        out["recent_breadth_shift"] = breadth_shift
        out["recent_liquidity_shift"] = liq_shift

    # The compact market state block uses only existing causal context fields.
    for out, frame in ((out_tr, train), (out_te, held)):
        for c in BLOCKS["market_state"]:
            if c in frame:
                out[c] = pd.to_numeric(frame[c], errors="coerce").fillna(float(pd.to_numeric(train[c], errors="coerce").median()) if c in train else 0.0).to_numpy(float)
            else:
                out[c] = 0.0
    all_cols = sorted(set().union(*(BLOCKS[b] for b in BLOCKS)))
    for out in (out_tr, out_te):
        out[:] = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for c in all_cols:
            if c not in out:
                out[c] = 0.0
    audit = {
        "context_fields": context, "recent_train_month": recent_month,
        "recent_rows": int(recent.sum()), "older_rows": int(older.sum()),
        "context_shift": ctx_shift, "covariance_break": cov_break,
    }
    return out_tr[all_cols], out_te[all_cols], audit


def _groups(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q = pd.to_datetime(frame["__ts__"], utc=True).dt.floor("4h").astype("int64").astype(str)
    q = q + "__" + frame["side_name"].astype(str)
    order = np.argsort(q.to_numpy(), kind="stable")
    qs = q.iloc[order]
    counts = qs.groupby(qs, sort=False).size()
    valid = counts.index[counts.to_numpy() >= 2]
    keep = qs.isin(valid).to_numpy()
    order = order[keep]
    groups = qs.iloc[keep].groupby(qs.iloc[keep], sort=False).size().to_numpy(dtype=np.int32)
    return order, groups


def _fit_meta(train: pd.DataFrame, held: pd.DataFrame, x_train: pd.DataFrame, x_held: pd.DataFrame, target: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order, groups = _groups(train)
    if len(groups) == 0:
        return np.zeros(len(train), dtype=np.float32), np.zeros(len(held), dtype=np.float32), np.zeros(len(held), dtype=np.float32)
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", lambdarank_truncation_level=10,
        n_estimators=120, learning_rate=0.04, max_depth=4, num_leaves=12,
        min_child_samples=max(120, int(math.ceil(0.03 * len(train)))),
        feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1,
        lambda_l1=1.0, lambda_l2=10.0, max_bin=63,
        label_gain=[0.0, 0.25, 1.0, 3.0, 7.0], random_state=seed,
        n_jobs=4, verbosity=-1,
    )
    model.fit(x_train.iloc[order], target[order], group=groups)
    raw_train = np.asarray(model.predict(x_train), dtype=np.float32)
    raw_held = np.asarray(model.predict(x_held), dtype=np.float32)
    rank_held = _pct(raw_held, raw_train)
    importance = np.asarray(model.feature_importances_, dtype=float)
    return raw_train, raw_held, rank_held, importance


def _map_canonical(train: pd.DataFrame, held: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(train.base_plus_consensus25.to_numpy(float)) & np.isfinite(train.exact_net_bps.to_numpy(float))
    if ok.sum() < 100:
        value = float(np.nanmean(train.exact_net_bps))
        return np.full(len(train), value, dtype=np.float32), np.full(len(held), value, dtype=np.float32)
    model = IsotonicRegression(out_of_bounds="clip", y_min=-1000.0, y_max=1000.0)
    model.fit(train.loc[ok, "base_plus_consensus25"], train.loc[ok, "exact_net_bps"])
    return model.predict(train.base_plus_consensus25).astype(np.float32), model.predict(held.base_plus_consensus25).astype(np.float32)


def _metric(frame: pd.DataFrame, score: str, tail: float) -> dict[str, object]:
    n = max(1, int(math.ceil(len(frame) * tail)))
    top = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
    return {"tail": float(tail), "trades": int(len(top)), "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": float(frame[[score, "exact_net_bps"]].corr(method="spearman").iloc[0, 1])}


def run(*, panel_path: Path = DEFAULT_PANEL, head_path: Path = DEFAULT_HEADS, output_dir: Path = DEFAULT_OUTPUT, selection_path: Path | None = None) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    panel, context, context_hash = _load()
    # Restrict to the scored historical substrate and join the strict-OOF
    # consensus-head health outputs.  The head artifact includes 2024 history
    # needed to train January 2025; only 2025 rows are evaluated.
    heads = pd.read_parquet(head_path)
    panel = panel.merge(heads, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_head"))
    panel["base_plus_consensus25"] = pd.to_numeric(panel["base_plus_consensus25"], errors="coerce")
    if panel.loc[panel.month.astype(str).isin(MONTHS), "base_plus_consensus25"].isna().any():
        raise RuntimeError("missing canonical Base+Consensus score in 2025 held rows")
    selected_map: dict[str, list[str]] = {}
    if selection_path is not None:
        selected_doc = json.loads(selection_path.read_text())
        selected_map = {str(k): list(v) for k, v in selected_doc.get("selected_features", {}).items()}
    block_feature_map = {k: list(selected_map.get(k, v)) for k, v in BLOCKS.items()}
    parts: list[pd.DataFrame] = []
    fit_audit: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    dev_months = sorted(set(panel.month.astype(str)) & set(MONTHS))
    for month in dev_months:
        held = panel.loc[panel.month.astype(str).eq(month)].copy()
        train = panel.loc[
            (panel.__ts__ < pd.Timestamp(month, tz="UTC"))
            & (panel.label_available_ts < pd.Timestamp(month, tz="UTC"))
            & panel.month.astype(str).ne(month)
        ].copy()
        for side in ("long", "short"):
            tr = train.loc[train.side_name.eq(side)].copy()
            te = held.loc[held.side_name.eq(side)].copy()
            if len(tr) < 500 or te.empty:
                continue
            tr_expected, te_expected = _map_canonical(tr, te)
            tr["canonical_expected_net_bps"] = tr_expected
            te["canonical_expected_net_bps"] = te_expected
            residual = tr.exact_net_bps.to_numpy(float) - tr_expected
            tr["meta_residual_bps"] = residual
            # The held residual is retained for evaluation only.
            te["meta_residual_bps"] = te.exact_net_bps.to_numpy(float) - te_expected
            tr_x, te_x, feat_audit = _feature_frame(tr, te, context)
            fit_audit.append({"month": month, "side": side, "train_rows": int(len(tr)), "held_rows": int(len(te)), "recent_train_month": feat_audit["recent_train_month"], "recent_context_shift": feat_audit["context_shift"], "recent_context_covariance_break": feat_audit["covariance_break"]})
            base_score = te.base_plus_consensus25.to_numpy(float)
            for arm, block_names in ARM_BLOCKS.items():
                if arm == "A_control":
                    meta_rank = np.full(len(te), 0.5, dtype=np.float32)
                    raw = np.zeros(len(te), dtype=np.float32)
                    imp = np.zeros(0, dtype=float)
                    feature_names: list[str] = []
                else:
                    feature_names = [c for b in block_names for c in block_feature_map[b] if c not in {"canonical_expected_net_bps", "base_plus_consensus25"}]
                    feature_names = list(dict.fromkeys(feature_names))
                    # Preserve a small belief anchor in every residual model.
                    feature_names = ["canonical_expected_net_bps", "base_plus_consensus25", *feature_names]
                    xtr = tr_x.copy(); xte = te_x.copy()
                    xtr["canonical_expected_net_bps"] = tr.canonical_expected_net_bps.to_numpy(float)
                    xte["canonical_expected_net_bps"] = te.canonical_expected_net_bps.to_numpy(float)
                    xtr["base_plus_consensus25"] = tr.base_plus_consensus25.to_numpy(float)
                    xte["base_plus_consensus25"] = te.base_plus_consensus25.to_numpy(float)
                    xtr = xtr.reindex(columns=feature_names).fillna(0.0)
                    xte = xte.reindex(columns=feature_names).fillna(0.0)
                    grade = np.digitize(residual, [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
                    _, raw, meta_rank, imp = _fit_meta(tr, te, xtr, xte, grade, seed=SEED + int(month[-2:]) * 100 + (0 if side == "long" else 1) + len(feature_names))
                    for name, value in zip(feature_names, imp):
                        importance_rows.append({"month": month, "side": side, "arm": arm, "feature": name, "gain_importance": float(value)})
                out = te[["candidate_id", "__ts__", "month", "side_name", "exact_net_bps", "exact_gross_bps", "base_plus_consensus25", "canonical_expected_net_bps", "meta_residual_bps"]].copy()
                out["arm"] = arm
                out["meta_residual_rank"] = meta_rank
                out["meta_score"] = 0.75 * out.base_plus_consensus25.to_numpy(float) + 0.25 * meta_rank
                out["raw_meta_residual"] = raw
                out["failure_proxy"] = 1.0 - meta_rank
                out["actual_failure"] = (out.meta_residual_bps.to_numpy(float) <= -150.0).astype(np.int8)
                out["feature_count"] = len(feature_names)
                parts.append(out)
    pred = pd.concat(parts, ignore_index=True)
    # Evaluate 2025 only; rankings are already month/side-comparable canonical
    # ranks and residual ranks, then enter a single pooled global ranking.
    metrics: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    for arm, block in pred.groupby("arm", sort=True):
        for tail in TAILS:
            row = _metric(block, "meta_score", tail)
            row.update({"arm": arm, "scope": "global_2025"})
            metrics.append(row)
        monthly = []
        for month, m in block.groupby("month", sort=True):
            row = _metric(m, "meta_score", 0.05)
            monthly.append(row["net_bps_per_trade"])
        vals = np.asarray(monthly, dtype=float)
        med = float(np.nanmedian(vals))
        stability.append({"arm": arm, "months": int(len(vals)), "mean_top5_net_bps": float(np.nanmean(vals)), "median_top5_net_bps": med, "mad_top5_net_bps": float(np.nanmedian(np.abs(vals - med))), "worst_month_top5_net_bps": float(np.nanmin(vals)), "positive_months_top5": int(np.sum(vals > 0.0))})
        for month, m in block.groupby("month", sort=True):
            row = _metric(m, "meta_score", 0.05)
            row.update({"arm": arm, "month": month, "scope": "monthly_2025"})
            metrics.append(row)
    outdir = output_dir
    outdir.mkdir(parents=True)
    pred.to_parquet(outdir / "predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(outdir / "metrics.parquet", index=False)
    pd.DataFrame(stability).to_parquet(outdir / "stability.parquet", index=False)
    pd.DataFrame(fit_audit).to_parquet(outdir / "fit_audit.parquet", index=False)
    pd.DataFrame(importance_rows).to_parquet(outdir / "feature_importance.parquet", index=False)
    failure_rows: list[dict[str, object]] = []
    for arm, block in pred.groupby("arm", sort=True):
        for scope, grouped in [("global_2025", [("all", block)]), ("side", list(block.groupby("side_name", sort=True))), ("month", list(block.groupby("month", sort=True)))]:
            for key, g in grouped:
                if g.actual_failure.nunique() < 2:
                    auc = np.nan
                else:
                    auc = float(roc_auc_score(g.actual_failure, g.failure_proxy))
                failure_rows.append({"arm": arm, "scope": scope, "group": str(key), "rows": int(len(g)), "failure_rate": float(g.actual_failure.mean()), "failure_proxy_auc": auc, "failure_proxy_rank_ic": float(g[["failure_proxy", "meta_residual_bps"]].corr(method="spearman").iloc[0, 1])})
    pd.DataFrame(failure_rows).to_parquet(outdir / "failure_diagnostic.parquet", index=False)
    manifest = {
        "schema": "tp6_sl4_canonical_residual_meta_block_ablation_v1",
        "status": "COMPLETE", "rows": int(len(pred)), "development_months": list(MONTHS),
        "base_contract": "canonical TP6/SL4 Base+Consensus 75/25",
        "target": "exact_net_bps - train-only isotonic(CanonicalScore)",
        "residual_grades": [-150.0, -50.0, 50.0, 150.0],
        "query": "4-hour UTC x side",
        "model": {"objective": "lambdarank", "max_depth": 4, "num_leaves": 12, "min_child_fraction": 0.03, "learning_rate": 0.04, "trees": 120, "lambda_l1": 1.0, "lambda_l2": 10.0, "max_bin": 63},
        "blend": "0.75 canonical score + 0.25 residual rank",
        "blocks": block_feature_map, "arms": ARM_BLOCKS,
        "feature_selection": str(selection_path) if selection_path is not None else None,
        "canonical_context_sha256": context_hash,
        "no_gam": True,
        "artifacts": ["predictions.parquet", "metrics.parquet", "stability.parquet", "fit_audit.parquet", "feature_importance.parquet", "failure_diagnostic.parquet", "run_manifest.json"],
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Canonical residual meta block ablation",
        "",
        "The control is the exact no-GAM canonical Base+Consensus score. Every residual arm uses a train-only isotonic canonical-score-to-net map, ordinal residual grades, shallow LambdaRank, 4-hour × side queries, and a frozen 75/25 canonical/residual-rank blend.",
        "",
        "## Global and monthly metrics",
        "",
        pd.DataFrame(metrics).round(3).to_string(index=False),
        "",
        "## Stability",
        "",
        pd.DataFrame(stability).round(3).to_string(index=False),
    ]
    (outdir / "TP6_SL4_CANONICAL_RESIDUAL_META_BLOCK_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(outdir), "rows": int(len(pred)), "arms": sorted(pred.arm.unique().tolist())}, indent=2))
    return outdir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--heads", type=Path, default=DEFAULT_HEADS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", type=Path, default=None)
    args = parser.parse_args()
    run(panel_path=args.panel, head_path=args.heads, output_dir=args.output_dir, selection_path=args.selection)
