#!/usr/bin/env python3
"""Monthly-refit expanded-universe replay for the frozen top-three arms.

This replay uses the available 170-symbol source population (February--July
2026).  The source is explicitly a signal-close execution-margin contract,
not the smaller canonical TP6/SL4 panel.  It is nevertheless a strict
monthly walk-forward: side-local base models, side-local monthly causal
archetype clustering, and residual rankers are refit using only rows resolved
before the next month's start.  The top-three component masks are frozen from
the preceding 16-arm canonical funnel; no expanded-universe tuning is used to
change them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = ROOT / "data_perp/reports/s59_h5_signalclose_causal_stagec_packb_sliding365_meta_hpo150_wf30_20260721_v1/best_full_oos/s52_train_meta_regime_handoff_smoke_predictions.parquet"
DEFAULT_UNIVERSE = ROOT / "data_perp/reports/s59_h5_signalclose_causal_stagec_packb_sliding365_meta_hpo150_wf30_20260721_v1/best_full_oos/p90spread_fee15bps_eligible170/eligible_symbols.csv"
DEFAULT_TOP3 = ROOT / "data_perp/artifacts/tp6_sl4_component_combo_funnel_long_20260808_v1/top3_configs.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_20260808_v1"
SEED = 20260808
SIDES = ("long", "short")
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)

# A fixed, diverse, decision-time-only contract.  Outcome labels, policy
# archetype labels, and post-entry path summaries are deliberately excluded.
CAUSAL_FEATURES = [
    "acceleration_of_move", "adx_cp_z_8_32_96", "asset_atr_level", "asset_liquidation_phase_score",
    "asset_minus_mkt_oi_7d_z_180d", "atr_compression_ratio", "avg_pair_corr_24h", "bollinger_band_width",
    "dir_path_long_2h", "dist_from_low_48h", "dist_local_swing", "dist_prior_day_high", "dist_prior_day_low",
    "down_up_vol_ratio_24", "eth_btc_ret_24h", "ext_atrExp", "ffd_rv_12_06", "ffd_rv_2h_04", "flow_persistence",
    "fund_extreme_duration_24h", "funding_mean_15d_robust_z", "funding_rate_cross_asset_dispersion", "gmm_mahal_3",
    "hour_cos", "hour_sin", "impact_24", "impact_z", "jump_intensity", "ker_10", "ker_16", "ker_24",
    "loc_pullback_depth_48", "log_realized_vol_cp_logstd_8_32", "mark_perp_dislocation", "mark_vs_perp_bps",
    "median_alt_minus_btc", "median_volume_z", "mkt_atr_expansion_1h", "mkt_flush_exhaustion_score",
    "mkt_median_oi_recovery_fraction_24h", "mkt_oi_recovery_from_24h_low", "mkt_rv_24h", "mr_potential",
    "ob_depth_usd_l20_z", "ob_imbalance_mkt_resid", "oi_chg_2h_robust_z", "oi_expansion_compression_balance_96h",
    "oi_value_1d_log_chg_cp_logstd_8_32", "pct_assets_high_rvol", "pers_24", "price_entropy_cp_logstd_8_32",
    "price_rv_15d_robust_z", "prog_eff_12", "progress", "pullback_72", "q_iqr__ret48h_bench_resid",
    "q_iqr__volume_z_12", "q_lower_tail__oi_7d_x_funding", "q_tail_asym__price_x_oi_3d",
    "q_tail_asym__xasset_ob_liquidity_peer_resid", "q_tail_width__amihud_z_peer_resid",
    "q_tail_width__loc_swing_range_pos_48", "q_tail_width__oi_to_volume_7d_z_180d",
    "q_tail_width__price_x_oi_3d", "q_tail_width__ret48h_bench_resid", "q_tail_width__volatility_zscore",
    "q_upper_tail__ob_trade_size_to_l1_depth_z_24h", "q_upper_tail__oi_to_volume_7d_z_180d",
    "q_upper_tail__xasset_ob_liquidity_peer_resid", "range_16h_pct", "ret16h", "return_autocorr_cp_logstd_8_32",
    "rv_12h", "rv_6h", "session_progress", "symbol_minus_mkt_ret_4h", "thrust_decay_8", "trend_slope_120h",
    "upside_semivariance_8", "vol_asym", "vol_low", "vol_of_vol", "vol_regime_ratio", "vol_regime_switch_12h",
    "volatility_of_volatility_48", "volume_zscore_cp_z_8_32_96", "vov_fast_slow_ratio",
    "xs_dispersion__oi_7d_x_funding", "xs_dispersion__oi_value_7d_chg_z_180d", "xs_dispersion__price_x_oi_7d",
    "xs_std__oi_to_volume_7d_z_180d",
]

# The expanded prediction parquet does not carry the raw handoff feature
# matrix. It carries this already-materialised, causal support/context spine;
# use it only as an explicit fallback and record that contract.
SOURCE_FALLBACK_FEATURES = [
    "base_margin_to_cutoff", "base_margin_to_cutoff_z", "base_signal_zscore_within_archetype",
    "base_score_rank_pct_train_prior", "rel_rankband_rows_log1p", "rel_rankband_clean_rate",
    "rel_rankband_bad_mae_rate", "rel_rankband_timeout_rate", "rel_rankband_dirty_positive_rate",
    "rel_rankband_exec_margin_mean", "rel_rankband_edge", "rel_marginband_rows_log1p",
    "rel_marginband_clean_rate", "rel_marginband_bad_mae_rate", "rel_marginband_timeout_rate",
    "rel_marginband_dirty_positive_rate", "rel_marginband_exec_margin_mean", "rel_marginband_edge",
    "support_min_log_count", "support_mean_log_count", "support_min_frequency", "support_mean_frequency",
    "support_unseen_bucket_share", "support_rare_bucket_share",
    "base_arch_hit_recent_rate_hl3d", "base_arch_hit_expected_rate_hl3d", "base_arch_hit_surprise_hl3d",
    "base_arch_hit_surprise_z_hl3d", "base_arch_hit_support_log1p_hl3d", "base_arch_hit_effective_n_hl3d",
    "base_arch_hit_recent_rate_hl7d", "base_arch_hit_expected_rate_hl7d", "base_arch_hit_surprise_hl7d",
    "base_arch_hit_surprise_z_hl7d", "base_arch_hit_support_log1p_hl7d", "base_arch_hit_effective_n_hl7d",
    "base_arch_hit_recent_rate_hl14d", "base_arch_hit_expected_rate_hl14d", "base_arch_hit_surprise_hl14d",
    "base_arch_hit_surprise_z_hl14d", "base_arch_hit_support_log1p_hl14d", "base_arch_hit_effective_n_hl14d",
]

BASE_PARAMS = dict(objective="multiclass", num_class=3, n_estimators=140, learning_rate=.05,
                   num_leaves=31, min_child_samples=350, subsample=.8, colsample_bytree=.8,
                   reg_lambda=8.0, n_jobs=2, verbosity=-1)
META_PARAMS = dict(objective="lambdarank", metric="ndcg", lambdarank_truncation_level=10,
                   n_estimators=120, learning_rate=.04, max_depth=4, num_leaves=12,
                   min_child_samples=350, feature_fraction=.80, bagging_fraction=.80, bagging_freq=1,
                   lambda_l1=1.0, lambda_l2=10.0, max_bin=63, label_gain=[0., .25, 1., 3., 7.],
                   n_jobs=2, verbosity=-1)


def _matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy(np.float32)


def _rank_pct(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return np.full(len(x), .5, dtype=np.float32)
    return (rankdata(x, method="average") - 1.0) / (len(x) - 1.0)


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


def _fit_ranker(train: pd.DataFrame, x_train: pd.DataFrame, x_held: pd.DataFrame, target: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order, groups = _groups(train)
    if len(groups) == 0:
        return np.zeros(len(x_held), dtype=np.float32), np.zeros(len(x_held), dtype=np.float32)
    model = lgb.LGBMRanker(random_state=seed, **META_PARAMS)
    model.fit(x_train.iloc[order], target[order], group=groups)
    raw_tr = np.asarray(model.predict(x_train), dtype=np.float32)
    raw_te = np.asarray(model.predict(x_held), dtype=np.float32)
    return raw_te, _rank_pct(raw_te)


def _map_fit(score: np.ndarray, net: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(score, np.linspace(0., 1., 11)))
    if len(edges) < 3:
        return np.array([-np.inf, np.inf]), np.array([float(np.nanmean(net))])
    bins = np.clip(np.digitize(score, edges[1:-1], right=True), 0, 9)
    means = np.array([float(np.nanmean(net[bins == i])) if np.any(bins == i) else float(np.nanmean(net)) for i in range(10)])
    return edges, means


def _map_apply(score: np.ndarray, mapping: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    edges, means = mapping
    return means[np.clip(np.digitize(score, edges[1:-1], right=True), 0, len(means) - 1)]


def _load_top3(path: Path) -> list[dict[str, object]]:
    frame = pd.read_parquet(path)
    out = []
    for _, row in frame.sort_values("selection_rank").head(3).iterrows():
        label = str(row["arm"])
        components = label.split("+") if label != "control" else []
        out.append({"arm": label, "components": components})
    return out


def _cluster_features(train: pd.DataFrame, held: pd.DataFrame, feature_cols: list[str], seed: int, n_clusters: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    xtr = _matrix(train, feature_cols)
    xte = _matrix(held, feature_cols)
    med = np.nanmedian(xtr, axis=0)
    mad = np.nanmedian(np.abs(xtr - med), axis=0)
    mad[~np.isfinite(mad) | (mad < 1e-6)] = 1.0
    ztr = np.clip((xtr - med) / mad, -8., 8.)
    zte = np.clip((xte - med) / mad, -8., 8.)
    sample_n = min(len(ztr), 100_000)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(ztr), size=sample_n, replace=False) if sample_n < len(ztr) else np.arange(len(ztr))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3, batch_size=2048, max_iter=80, reassignment_ratio=.01)
    km.fit(ztr[sample])
    dtr = km.transform(ztr)
    dte = km.transform(zte)
    # Soft memberships from causal geometry, temperature set from training
    # median nearest/second-nearest separation rather than outcomes.
    order = np.sort(dtr, axis=1)
    gap = np.nanmedian(order[:, 1] - order[:, 0]) if n_clusters > 1 else 1.0
    temp = float(max(gap, .25))
    ptr = softmax(-dtr / temp, axis=1)
    pte = softmax(-dte / temp, axis=1)
    ctr = np.asarray(km.cluster_centers_, dtype=float)
    signed_tr = np.einsum("nd,kd->nk", ztr, ctr) / math.sqrt(max(1, ztr.shape[1]))
    signed_te = np.einsum("nd,kd->nk", zte, ctr) / math.sqrt(max(1, zte.shape[1]))
    out_tr = pd.DataFrame(index=train.index); out_te = pd.DataFrame(index=held.index)
    for k in range(n_clusters):
        out_tr[f"archetype_signed_{k:02d}"] = np.clip(ptr[:, k] * signed_tr[:, k], -3., 3.)
        out_te[f"archetype_signed_{k:02d}"] = np.clip(pte[:, k] * signed_te[:, k], -3., 3.)
        out_tr[f"archetype_prob_{k:02d}"] = ptr[:, k]; out_te[f"archetype_prob_{k:02d}"] = pte[:, k]
    for out, p, d in ((out_tr, ptr, dtr), (out_te, pte, dte)):
        so = np.sort(p, axis=1)
        out["archetype_entropy"] = -np.sum(np.where(p > 1e-12, p * np.log(np.maximum(p, 1e-12)), 0.), axis=1)
        out["archetype_top2_margin"] = so[:, -1] - so[:, -2]
        out["archetype_max_prob"] = p.max(axis=1)
        out["archetype_distance"] = d.min(axis=1)
        out["archetype_active_count"] = (p >= .15).sum(axis=1)
        out["archetype_signed_total"] = out[[f"archetype_signed_{k:02d}" for k in range(n_clusters)]].sum(axis=1)
        out["archetype_signed_abs_total"] = out[[f"archetype_signed_{k:02d}" for k in range(n_clusters)]].abs().sum(axis=1)
    audit = {"n_clusters": n_clusters, "feature_count": len(feature_cols), "train_rows": len(train), "held_rows": len(held), "temperature": temp, "centroid_sha256": hashlib.sha256(np.asarray(km.cluster_centers_, dtype=np.float32).tobytes()).hexdigest()}
    return out_tr, out_te, audit


def _context_features(train: pd.DataFrame, held: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    xtr = _matrix(train, feature_cols); xte = _matrix(held, feature_cols)
    med = np.nanmedian(xtr, axis=0); mad = np.nanmedian(np.abs(xtr - med), axis=0); mad[~np.isfinite(mad) | (mad < 1e-6)] = 1.0
    ztr = np.clip((xtr - med) / mad, -20., 20.); zte = np.clip((xte - med) / mad, -20., 20.)
    outs = []
    for z, frame in ((ztr, train), (zte, held)):
        out = pd.DataFrame(index=frame.index)
        az = np.abs(z)
        out["context_ood_mean_abs_z"] = np.nanmean(az, axis=1)
        out["context_ood_p95_abs_z"] = np.nanpercentile(az, 95, axis=1)
        out["context_ood_outlier_fraction"] = np.nanmean(az > 3., axis=1)
        out["context_ood_tail_fraction"] = np.nanmean(az > 2., axis=1)
        out["support_min_frequency"] = pd.to_numeric(frame.get("support_min_frequency", 0.), errors="coerce").fillna(0.).to_numpy(float) if "support_min_frequency" in frame else 0.
        out["support_unseen_bucket_share"] = pd.to_numeric(frame.get("support_unseen_bucket_share", 0.), errors="coerce").fillna(0.).to_numpy(float) if "support_unseen_bucket_share" in frame else 0.
        out["support_rare_bucket_share"] = pd.to_numeric(frame.get("support_rare_bucket_share", 0.), errors="coerce").fillna(0.).to_numpy(float) if "support_rare_bucket_share" in frame else 0.
        outs.append(out)
    return outs[0], outs[1]


def _base_features(frame: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=base_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.)


def _base_classes(frame: pd.DataFrame) -> np.ndarray:
    net = pd.to_numeric(frame["exec_margin"], errors="coerce").to_numpy(float) * 10000.
    return np.select([net < -50., net > 50.], [0, 2], default=1).astype(np.int8)


def _metric(frame: pd.DataFrame, score: str, tail: float) -> dict[str, object]:
    n = max(1, int(math.ceil(len(frame) * tail)))
    top = frame.sort_values([score, "row_id"], ascending=[False, True], kind="stable").head(n)
    return {"tail": float(tail), "trades": int(len(top)), "gross_bps_per_trade": float(top.first_touch_gross_bps.mean()), "net_bps_per_trade": float(top.net_bps.mean()), "rank_ic": float(frame[[score, "net_bps"]].corr(method="spearman").iloc[0, 1])}


def _feature_names(components: list[str]) -> list[str]:
    names: list[str] = []
    if "model_support_ood" in components:
        names.extend(["context_ood_mean_abs_z", "context_ood_p95_abs_z", "context_ood_outlier_fraction", "context_ood_tail_fraction", "support_min_frequency", "support_unseen_bucket_share", "support_rare_bucket_share"])
    if "uncertainty" in components:
        names.extend(["p_clear", "p_adverse", "p_weak", "base_entropy", "base_top2_margin", "base_conviction", "base_score"])
    if "archetype_signed_exposure" in components:
        names.extend([f"archetype_signed_{k:02d}" for k in range(6)] + ["archetype_signed_total", "archetype_signed_abs_total"])
    if "compact_structural" in components:
        names.extend(["archetype_entropy", "archetype_top2_margin", "archetype_max_prob", "archetype_distance", "archetype_active_count", "archetype_signed_total", "archetype_signed_abs_total"])
    return list(dict.fromkeys(names))


def run(*, output_dir: Path = DEFAULT_OUTPUT, source_path: Path = DEFAULT_SOURCE, universe_path: Path = DEFAULT_UNIVERSE, top3_path: Path = DEFAULT_TOP3) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source = pd.read_parquet(source_path)
    universe = pd.read_csv(universe_path)["symbol"].astype(str).tolist()
    symbol_col = "__symbol__" if "__symbol__" in source else "symbol"
    source = source.loc[source[symbol_col].astype(str).isin(universe)].copy()
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True)
    source["valid_end"] = pd.to_datetime(source["valid_end"], utc=True)
    source["month"] = source["__ts__"].dt.to_period("M").astype(str)
    source["row_id"] = source[symbol_col].astype(str) + "|" + source["__ts__"].astype(str) + "|" + source["side_name"].astype(str)
    source["net_bps"] = pd.to_numeric(source["exec_margin"], errors="coerce") * 10000.
    source["first_touch_gross_bps"] = pd.to_numeric(source["first_touch_gross"], errors="coerce") * 10000.
    source = source.loc[np.isfinite(source.net_bps) & np.isfinite(source.first_touch_gross_bps)].copy()
    raw_base_cols = [c for c in CAUSAL_FEATURES if c in source.columns and pd.to_numeric(source[c], errors="coerce").notna().any()]
    base_cols = raw_base_cols
    feature_contract_source = "raw_handoff_causal"
    if not base_cols:
        base_cols = [c for c in SOURCE_FALLBACK_FEATURES if c in source.columns and pd.to_numeric(source[c], errors="coerce").notna().any()]
        feature_contract_source = "materialised_causal_support_context_fallback"
    if not base_cols:
        raise RuntimeError("expanded source contains neither raw causal fields nor the expected support/context fallback")
    top3 = _load_top3(top3_path)
    months = sorted(source.month.unique())
    all_oos: list[pd.DataFrame] = []
    month_audit: list[dict[str, object]] = []
    cluster_audit: list[dict[str, object]] = []
    oof_history: dict[str, list[pd.DataFrame]] = {side: [] for side in SIDES}

    for month in months:
        start = pd.Timestamp(month + "-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        held_month = source.loc[(source["__ts__"] >= start) & (source["__ts__"] < end)].copy()
        month_parts: list[pd.DataFrame] = []
        for side_i, side in enumerate(SIDES):
            held = held_month.loc[held_month.side_name.eq(side)].copy()
            # The source declares rows whose path closes exactly at the next
            # month boundary as available before that month's first decision.
            # Include the boundary explicitly; rows resolving after it remain
            # excluded.
            train = source.loc[(source.side_name.eq(side)) & (source["valid_end"] <= start)].copy()
            if len(train) < 500 or held.empty:
                month_audit.append({"month": month, "side": side, "status": "SKIPPED_WARMUP_OR_EMPTY", "train_rows": len(train), "held_rows": len(held)})
                continue
            # Monthly, causal archetype discovery.  It is refit from prior
            # rows, then reused for every held row in this month.
            cluster_tr, cluster_te, ca = _cluster_features(train, held, base_cols, SEED + side_i + int(month[-2:]))
            ood_tr, ood_te = _context_features(train, held, base_cols)
            cluster_audit.append({"month": month, "side": side, **ca})
            # Side-local R3-style cost-aware three-state base model.
            y = _base_classes(train)
            base = lgb.LGBMClassifier(random_state=SEED + 100 + side_i + int(month[-2:]), **BASE_PARAMS)
            base.fit(_base_features(train, base_cols), y)
            p_train = base.predict_proba(_base_features(train, base_cols))
            p_held = base.predict_proba(_base_features(held, base_cols))
            def add_base(frame: pd.DataFrame, p: np.ndarray, cluster: pd.DataFrame, ood: pd.DataFrame) -> pd.DataFrame:
                out = frame[["row_id", "__ts__", "side_name", "month", "net_bps", "first_touch_gross_bps"]].copy()
                out = pd.concat([out.reset_index(drop=True), cluster.reset_index(drop=True), ood.reset_index(drop=True)], axis=1)
                out["p_adverse"] = p[:, 0]; out["p_weak"] = p[:, 1]; out["p_clear"] = p[:, 2]
                pp = np.clip(p, 1e-8, 1.); out["base_entropy"] = -np.sum(pp * np.log(pp), axis=1)
                so = np.sort(p, axis=1); out["base_top2_margin"] = so[:, -1] - so[:, -2]
                out["base_conviction"] = p[:, 2] - .5 * p[:, 1]
                out["base_score"] = p[:, 2] - p[:, 0]
                return out
            base_train = add_base(train, p_train, cluster_tr, ood_tr)
            held_score = add_base(held, p_held, cluster_te, ood_te)
            # Current month is a base OOS fold.  Residual training uses only
            # previously generated base OOF rows and prior-resolved labels.
            history = pd.concat(oof_history[side], ignore_index=True) if oof_history[side] else pd.DataFrame()
            if history.empty:
                for cfg in top3:
                    out = held_score.copy(); out["arm"] = cfg["arm"]; out["meta_raw"] = 0.; out["residual_rank"] = .5; out["base_rank"] = _rank_pct(out.base_score.to_numpy(float)); out["meta_score"] = out.base_rank
                    month_parts.append(out)
                residual_status = "NO_PRIOR_OOF_CONTROL_BASE_RANK"
            else:
                mapping = _map_fit(history.base_score.to_numpy(float), history.net_bps.to_numpy(float))
                held_score["expected_net_bps"] = _map_apply(held_score.base_score.to_numpy(float), mapping)
                history["expected_net_bps"] = _map_apply(history.base_score.to_numpy(float), mapping)
                history["residual_bps"] = history.net_bps - history.expected_net_bps
                grade = np.digitize(history.residual_bps.to_numpy(float), [-150., -50., 50., 150.]).astype(np.int32)
                residual_status = "FIT_PRIOR_OOF"
                for cfg in top3:
                    names = _feature_names(cfg["components"])
                    xtr = history.reindex(columns=names).fillna(0.)
                    xte = held_score.reindex(columns=names).fillna(0.)
                    raw, rank = _fit_ranker(history.assign(__ts__=pd.to_datetime(history.__ts__, utc=True)), xtr, xte, grade, SEED + 500 + side_i + int(month[-2:]) + len(names))
                    out = held_score.copy(); out["arm"] = cfg["arm"]; out["meta_raw"] = raw; out["residual_rank"] = rank; out["base_rank"] = _rank_pct(out.base_score.to_numpy(float)); out["meta_score"] = .75 * out.base_rank + .25 * rank
                    month_parts.append(out)
            month_audit.append({"month": month, "side": side, "status": residual_status, "train_rows": len(train), "held_rows": len(held), "base_features": len(base_cols), "train_clear_rate": float(np.mean(y == 2)), "held_symbols": int(held[symbol_col].nunique()), "components": [cfg["arm"] for cfg in top3]})
            # Save only this month's held-out base predictions as OOF
            # residual-training rows.  The in-sample base_train predictions
            # must never enter the residual learner.
            oof_history[side].append(held_score)
        if month_parts:
            all_oos.extend(month_parts)

    pred = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    metrics: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    per_side: list[dict[str, object]] = []
    if not pred.empty:
        for arm, block in pred.groupby("arm", sort=True):
            for tail in TAILS:
                row = _metric(block, "meta_score", tail); row.update({"arm": arm, "scope": "global_expanded_signal_close"}); metrics.append(row)
            for month, mb in block.groupby("month", sort=True):
                row = _metric(mb, "meta_score", .05); row.update({"arm": arm, "month": month, "scope": "monthly_expanded_signal_close"}); monthly.append(row)
            for side, sb in block.groupby("side_name", sort=True):
                for tail in (.01, .05, .10):
                    row = _metric(sb, "meta_score", tail); row.update({"arm": arm, "side": side, "scope": "side_expanded_signal_close"}); per_side.append(row)
    output_dir.mkdir(parents=True)
    pred.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(output_dir / "metrics.parquet", index=False)
    pd.DataFrame(monthly).to_parquet(output_dir / "monthly_metrics.parquet", index=False)
    pd.DataFrame(per_side).to_parquet(output_dir / "per_side_metrics.parquet", index=False)
    pd.DataFrame(month_audit).to_parquet(output_dir / "month_audit.parquet", index=False)
    pd.DataFrame(cluster_audit).to_parquet(output_dir / "cluster_audit.parquet", index=False)
    manifest = {
        "schema": "tp6_sl4_component_combo_expanded_monthly_v1", "status": "COMPLETE", "universe_count": len(universe), "source_rows_after_universe_filter": int(len(source)), "source_symbols_observed": int(source[symbol_col].nunique()), "source_period": [str(source["__ts__"].min()), str(source["__ts__"].max())],
        "source_contract": "signal-close execution-margin source: net_bps=exec_margin*10000; first_touch_gross_bps=first_touch_gross*10000; not the canonical TP6/SL4 panel", "cost_contract": "source exec_margin already includes the source fee/spread adjustment; no second cost subtraction",
        "monthly_refit": {"base": "side-local LGBM 3-state classifier refit using prior resolved rows", "residual": "side-local 4h x side LambdaRank refit using prior monthly base OOF rows", "archetype": "side-local MiniBatchKMeans K=6 refit on prior causal feature rows each month", "boundary_rule": "valid_end <= month_start is treated as resolved before the next month's first decision", "first_month": "warmup skipped because the source starts in February 2026 and has no earlier 170-symbol labelled history"},
        "base_target": "R3-style signal-close classes: adverse net < -50 bps, weak -50..+50 bps, clear > +50 bps", "residual_target": "net bps minus prior-OOF causal base-score-to-net map, ordinal grades [-150,-50,50,150]", "base_feature_count": len(base_cols), "base_features": base_cols, "base_feature_contract_source": feature_contract_source, "raw_causal_fields_available_in_source": len(raw_base_cols), "top3_frozen": top3, "artifacts": ["predictions.parquet", "metrics.parquet", "monthly_metrics.parquet", "per_side_metrics.parquet", "month_audit.parquet", "cluster_audit.parquet", "run_manifest.json"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    report = ["# Expanded 170-symbol monthly-refit replay", "", "This is a strict monthly walk-forward on the eligible 170-symbol signal-close source. It is not the exact canonical TP6/SL4 panel; the distinction is recorded in the manifest.", "", "## Global metrics", "", pd.DataFrame(metrics).round(3).to_string(index=False), "", "## Monthly top-5 metrics", "", pd.DataFrame(monthly).round(3).to_string(index=False), "", "## Per-side metrics", "", pd.DataFrame(per_side).round(3).to_string(index=False), "", "## Monthly/model audit", "", pd.DataFrame(month_audit).round(3).to_string(index=False), "", "## Manifest", "", json.dumps(manifest, indent=2, default=str)]
    (output_dir / "TP6_SL4_COMPONENT_COMBO_EXPANDED_MONTHLY_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(output_dir), "rows": int(len(pred)), "source_rows": int(len(source)), "source_symbols": int(source[symbol_col].nunique()), "metrics": metrics}, indent=2, default=str))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--top3", type=Path, default=DEFAULT_TOP3)
    args = parser.parse_args()
    run(output_dir=args.output_dir, source_path=args.source, universe_path=args.universe, top3_path=args.top3)
