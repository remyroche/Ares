#!/usr/bin/env python3
"""Full-universe causal TP6/SL4 base + eight residual-head replay.

This is the full-population counterpart to the earlier 20k-row downstream
experiment.  It refits a TP6/SL4 R3 base classifier and eight residual
LambdaRank heads before each 2025 month, then compares the resulting ranks
with the already-frozen 15-minute-policy downstream scores on the identical
1.46m-candidate population.

The residual heads are deliberately residual heads: their grade target is
exact TP6/SL4 net bps minus a train-only monotone map of the base score.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ARES = Path("/Users/remyroche/Documents/Ares")
CODEX = Path("/Users/remyroche/Documents/Codex")
HIST_DIR = CODEX / "artifacts/structural_conditional_portability_20260806_v1/candidate_feature_panel"
CURRENT_DIR = CODEX / "artifacts/policy_correction_2025_expanding_oof_20260807_v2/monthly_inputs"
LABEL_DIR = CODEX / "artifacts/full_2025_h12_tp6_sl4_15m_proxy_20260807_v1/parts"
SIDE_DIR = ARES / "data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1/parts"
ROBUST_DIR = ARES / "data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1/parts"
SETS_JSON = CODEX / "artifacts/structural_conditional_portability_2025_full_year_repaired_v1/structural_feature_sets.json"
INCUMBENT = CODEX / "artifacts/four_causal_stacks_two_exit_2025_20260807_v1/four_stack_predictions_2025.parquet"
OUT = ARES / "data_perp/artifacts/full_universe_tp6_sl4_retrain_20260807_v1"

MONTHS = tuple(f"2025-{m:02d}" for m in range(1, 13))
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
SEED = 20260807
BASE_SAMPLE_PER_MONTH = 40_000
CURRENT_SAMPLE_PER_MONTH = 20_000
MAX_RANK_TRAIN = 350_000

BASE_OUTPUTS = {"p_clear", "p_adverse", "p_weak", "base_score", "base_anchor"}


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def _base_fields() -> list[str]:
    obj = json.loads(SETS_JSON.read_text())
    fields = list(obj["sets"]["long"]["120"])
    if len(fields) != 120:
        raise ValueError(f"expected 120 base fields, got {len(fields)}")
    return fields


def _context_fields(base_fields: list[str]) -> list[str]:
    """Return a deterministic 73-field causal context contract.

    The old pre-2025 panel contains a wider context vocabulary, but the full
    2025 frozen population materialises only the 120-field contract.  We use
    a stable, diverse prefix of that contract and record the exact hash in the
    manifest instead of silently introducing unavailable fields.
    """
    preferred = [
        "grind_score_surprise", "spike_score_surprise", "rvol_z_peer_resid",
        "excess_6h_ts_resid", "ret4h_peer_resid", "volume_price_corr_ts_resid",
        "liquidation_climax_score", "post_liquidation_rebound_score",
        "post_flush_leverage_rebuild", "state_spectral_eig_condition",
        "cross_asset_downside_corr_4h", "cross_asset_corr_1h",
        "xs_dispersion__vol_z", "xs_dispersion__rvol_z",
        "xs_dispersion__vol_z_peer_resid", "xs_dispersion__xasset_ob_liquidity_ts_resid",
        "xs_dispersion__xasset_ob_liquidity_peer_resid", "q_lower_tail__xasset_ob_liquidity_peer_resid",
        "q_lower_tail__xasset_mkt_spread_bps", "q_lower_tail__ob_spread_bps_z_24h",
        "q_lower_tail__ob_depth_usd_l20_z", "xasset_mkt_ob_stress_z_24h",
        "mkt_close_location_1h", "breadth_recovery_from_6h_min", "mkt_return_accel_1h",
        "negative_breadth_pct", "mkt_oi_chg_accel_1h", "mkt_ret_15m",
        "down_barrier_pressure_daily_donchian", "mkt_price_up_oi_up_1h",
        "market_breadth_drawdown_from_6h_max", "mkt_rv_4h", "mkt_ret_per_oi_change_1h",
        "q_lower_tail__volume_z_24", "distance_to_resistance_atr",
        "pct_assets_price_up_oi_down_1h", "mkt_oi_chg_15m", "pct_assets_up_4h",
        "q_lower_tail__vol_z_peer_resid", "xasset_mkt_spread_bps_z_24h",
        "mkt_price_up_oi_up_4h", "q_iqr__ob_trade_size_to_l1_depth_z_24h",
        "mkt_oi_flush_z_30d", "q_upper_tail__xasset_ob_liquidity_ts_resid",
        "q_lower_tail__oi_7d_x_funding", "mkt_pct_price_up_oi_up_1h",
        "pct_assets_large_lower_wick", "mkt_pct_price_up_oi_down_1h", "mkt_ret_4h",
        "pct_assets_bullish_reversal_candle", "market_breadth_recovery_from_24h_min",
        "log_bars_since_below_3atr", "q_tail_width__xasset_ob_liquidity_ts_resid",
        "q_upper_tail__ob_spread_bps_z_24h", "xs_dispersion__oi_value_1d_chg_z_90d",
        "xs_dispersion__ob_depth_to_qv_z_x_rvol_z", "q_tail_asym__vol_z_4h",
        "q_upper_tail__bars_in_high_vol_state_log_norm", "breadth_chg_15m",
        "q_tail_width__volatility_zscore", "q_tail_asym__ob_spread_bps_z_24h",
        "state_spectral_eig_gap_1_2", "pct_assets_oi_down_1h", "mkt_pct_price_up_oi_up_4h",
        "mkt_median_oi_recovery_fraction_24h", "mkt_lower_wick_ratio_1h",
        "mkt_oi_dispersion_24h", "mkt_pct_oi_chg_1h_rz_lt_minus1", "memory_asymmetry_1ATR",
    ]
    out = [f for f in preferred if f in base_fields]
    out.extend(f for f in base_fields if f not in out)
    return out[:73]


def _load_label_sidecars() -> pd.DataFrame:
    cols = ["candidate_id", "t2_tp6_sl4_event", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "__label_available_at__"]
    frames = [_read(p, cols) for p in sorted(SIDE_DIR.glob("*.parquet"))]
    side = pd.concat(frames, ignore_index=True)
    side = side.drop_duplicates("candidate_id", keep="last")
    robust_cols = ["candidate_id", "label_valid", "target_invalid", "lower_touch_minute", "robust_clear_event_b25"]
    robust = pd.concat([_read(p, robust_cols) for p in sorted(ROBUST_DIR.glob("*.parquet"))], ignore_index=True)
    robust = robust.drop_duplicates("candidate_id", keep="last")
    out = side.merge(robust, on="candidate_id", how="left", validate="one_to_one")
    return out


def _load_historical(base_fields: list[str], labels: pd.DataFrame) -> pd.DataFrame:
    cols = ["candidate_id", "__ts__", "side_name", "label_available_ts", *base_fields]
    parts = []
    for p in sorted(HIST_DIR.glob("month=*.parquet")):
        frame = _read(p, cols)
        frame = frame.loc[frame["side_name"].eq("long")].copy()
        parts.append(frame)
    hist = pd.concat(parts, ignore_index=True)
    hist = hist.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    hist["label_available_ts"] = pd.to_datetime(hist["__label_available_at__"], utc=True)
    return _finish_labels(hist)


def _load_current(month: str, base_fields: list[str], labels: pd.DataFrame) -> pd.DataFrame:
    cols = ["__ts__", "__symbol__", "candidate_id", "side_name", "label_available_ts", "policy_net_bps", "policy_gross_bps", *base_fields]
    frame = _read(CURRENT_DIR / f"month={month}.parquet", cols)
    frame = frame.loc[frame["side_name"].eq("long")].copy()
    lab = _read(LABEL_DIR / f"month={month}/side=long.parquet", [
        "candidate_id", "label_valid", "target_invalid", "label_resolution", "lower_touch_bar",
        "robust_clear_event_b25", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "label_available_ts",
    ])
    frame = frame.drop(columns=["label_available_ts"], errors="ignore").merge(lab, on="candidate_id", how="left", validate="one_to_one")
    frame["month"] = month
    return _finish_labels(frame)


def _finish_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["label_available_ts"] = pd.to_datetime(frame["label_available_ts"], utc=True, errors="coerce")
    frame["tp6_net_bps"] = pd.to_numeric(frame["t4_tp6_sl4_net_bps"], errors="coerce")
    frame["tp6_gross_bps"] = pd.to_numeric(frame["t4_tp6_sl4_gross_bps"], errors="coerce")
    valid = frame.get("label_valid", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    clear = pd.to_numeric(frame.get("robust_clear_event_b25"), errors="coerce").eq(1)
    adverse = pd.to_numeric(frame.get("lower_touch_minute", frame.get("lower_touch_bar")), errors="coerce").ge(0)
    frame["r3_class"] = np.select([clear, adverse], [2, 0], default=1).astype("int8")
    frame["label_valid"] = valid & np.isfinite(frame["tp6_net_bps"])
    frame["month"] = frame.get("month", frame["__ts__"].dt.strftime("%Y-%m"))
    # Keep invalid/unresolved candidates in the scored population.  They are
    # excluded from supervised fitting, but retaining them is required for a
    # true full-universe replay and makes coverage explicit in the outputs.
    return frame.copy()


def _sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed).copy()


def _prep(frame: pd.DataFrame, fields: list[str], med: pd.Series | None = None):
    z = frame.reindex(columns=fields).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = z.median().fillna(0.0)
        med.attrs["scale"] = ((z - med).abs().median().replace(0.0, 1.0).fillna(1.0)).to_dict()
    scale = pd.Series(med.attrs.get("scale", {}), dtype=float).reindex(fields).fillna(1.0)
    return ((z.fillna(med).fillna(0.0) - med) / scale).clip(-20.0, 20.0).astype("float32"), med


def _base_fit(train: pd.DataFrame, held: pd.DataFrame, fields: list[str], seed: int):
    x, med = _prep(train, fields)
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=140, learning_rate=0.05,
        num_leaves=31, max_depth=-1, min_child_samples=350, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=8.0, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(x, train["r3_class"].to_numpy(), categorical_feature=[])
    px = model.predict_proba(x)
    hx, _ = _prep(held, fields, med)
    ph = model.predict_proba(hx)
    del model, x, hx
    gc.collect()
    p3 = px[:, 2]; p0 = px[:, 0]; q3 = ph[:, 2]; q0 = ph[:, 0]
    return p3 - 0.5 * p0, q3 - 0.5 * q0, px, ph


def _map_base(train_score, held_score, train_net):
    ok = np.isfinite(train_score) & np.isfinite(train_net)
    if ok.sum() < 100:
        mean = float(np.nanmean(train_net))
        return np.full(len(train_score), mean, np.float32), np.full(len(held_score), mean, np.float32)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=-2000.0, y_max=2000.0)
    iso.fit(train_score[ok], train_net[ok])
    return iso.predict(train_score).astype("float32"), iso.predict(held_score).astype("float32")


def _groups(frame: pd.DataFrame):
    q = pd.to_datetime(frame["__ts__"], utc=True).dt.floor("4h").astype("int64").to_numpy()
    order = np.argsort(q, kind="stable")
    qs = q[order]
    _, counts = np.unique(qs, return_counts=True)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    keep = counts >= 2
    order = order[np.repeat(keep, counts)]
    return order, counts[keep].astype(np.int32)


def _rank_fit(train: pd.DataFrame, held: pd.DataFrame, fields: list[str], grade: np.ndarray, equal_month: bool, seed: int):
    if len(train) > MAX_RANK_TRAIN:
        train = _sample(train, MAX_RANK_TRAIN, seed)
        grade = train["_grade"].to_numpy(np.int32)
    x, med = _prep(train, fields)
    order, groups = _groups(train)
    if len(groups) == 0:
        return np.zeros(len(train), np.float32), np.zeros(len(held), np.float32)
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", lambdarank_truncation_level=10,
        n_estimators=140, learning_rate=0.035, max_depth=5, num_leaves=31,
        min_child_samples=180, feature_fraction=0.82, bagging_fraction=0.82,
        bagging_freq=1, lambda_l1=0.02, lambda_l2=2.0, max_bin=127,
        label_gain=[0.0, 0.25, 1.0, 3.0, 7.0], random_state=seed, n_jobs=4, verbosity=-1,
    )
    weights = np.ones(len(train), dtype=np.float32)
    if equal_month:
        counts = train["month"].value_counts()
        weights = train["month"].map((1.0 / counts).to_dict()).to_numpy(float)
        weights = (weights * len(weights) / max(weights.sum(), 1e-12)).astype(np.float32)
    model.fit(x.iloc[order], grade[order], group=groups, sample_weight=weights[order])
    raw_tr = np.asarray(model.predict(x), dtype=np.float32)
    hx, _ = _prep(held, fields, med)
    raw_te = np.asarray(model.predict(hx), dtype=np.float32)
    del model, x, hx
    gc.collect()
    return raw_tr, raw_te


def _pct(values, ref):
    ref = np.sort(np.asarray(ref, dtype=float))
    return (np.searchsorted(ref, values, side="right") / max(len(ref), 1)).astype("float32")


def _norm_month(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        df[c] = df.groupby("month", sort=False)[c].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")


def _metrics(pred: pd.DataFrame):
    arms = ["new_tp6_base", "new_tp6_base_consensus25", "new_tp6_full_blend", "incumbent_r3_consensus", "incumbent_r3_full"]
    rows, monthly = [], []
    for arm in arms:
        for outcome in ("tp6", "policy"):
            netcol = f"{outcome}_net_bps"; grosscol = f"{outcome}_gross_bps"
            for tail in TAILS:
                n = max(1, int(math.ceil(len(pred) * tail)))
                top = pred.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
                rows.append({"arm": arm, "outcome": outcome, "tail": tail, "trades": n,
                    "gross_bps_per_trade": float(top[grosscol].mean()), "net_bps_per_trade": float(top[netcol].mean()),
                    "rank_ic": float(pred[[arm, netcol]].corr(method="spearman").iloc[0, 1])})
            for month, g in pred.groupby("month", sort=True):
                n = max(1, int(math.ceil(len(g) * 0.05)))
                top = g.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
                monthly.append({"arm": arm, "outcome": outcome, "month": month, "tail": .05,
                    "trades": n, "gross_bps_per_trade": float(top[grosscol].mean()),
                    "net_bps_per_trade": float(top[netcol].mean()),
                    "rank_ic": float(g[[arm, netcol]].corr(method="spearman").iloc[0, 1])})
    mon = pd.DataFrame(monthly)
    stability = []
    for (arm, outcome), g in mon.groupby(["arm", "outcome"], sort=True):
        vals = g["net_bps_per_trade"].to_numpy(float); med = float(np.nanmedian(vals))
        stability.append({"arm": arm, "outcome": outcome, "months": len(vals), "mean_top5_net_bps": float(np.nanmean(vals)),
            "median_top5_net_bps": med, "mad_top5_net_bps": float(np.nanmedian(np.abs(vals-med))),
            "worst_month_top5_net_bps": float(np.nanmin(vals)), "positive_months_top5": int((vals>0).sum()),
            "mean_month_rank_ic": float(g["rank_ic"].mean())})
    return pd.DataFrame(rows), mon, pd.DataFrame(stability)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    base_fields = _base_fields(); context = _context_fields(base_fields)
    labels = _load_label_sidecars()
    hist = _load_historical(base_fields, labels)
    incumbent = _read(INCUMBENT, ["candidate_id", "score__strict_r3_base_plus_consensus", "score__strict_r3_full"])
    prior_samples: list[pd.DataFrame] = []
    scored = []; head_audit = []
    for mi, month in enumerate(MONTHS, 1):
        held = _load_current(month, base_fields, labels)
        start = pd.Timestamp(month, tz="UTC")
        hist_pool = _sample(hist.loc[(hist["label_available_ts"] < start) & hist["label_valid"]], BASE_SAMPLE_PER_MONTH * min(15, mi + 3), SEED + mi)
        train_parts = [hist_pool]
        if prior_samples:
            train_parts.extend(prior_samples)
        train = pd.concat(train_parts, ignore_index=True)
        train = train.loc[(train["label_available_ts"] < start) & train["label_valid"]].copy()
        if len(train) > 500_000:
            train = _sample(train, 500_000, SEED + 1000 + mi)
        train_score, held_score, train_prob, held_prob = _base_fit(train, held, base_fields, SEED + mi)
        train["base_score"] = train_score; held["base_score"] = held_score
        train["p_clear"] = train_prob[:, 2]; train["p_adverse"] = train_prob[:, 0]; train["p_weak"] = train_prob[:, 1]
        held["p_clear"] = held_prob[:, 2]; held["p_adverse"] = held_prob[:, 0]; held["p_weak"] = held_prob[:, 1]
        tr_anchor, te_anchor = _map_base(train_score, held_score, train["tp6_net_bps"].to_numpy(float))
        train["base_anchor"] = tr_anchor; held["base_anchor"] = te_anchor
        resid = train["tp6_net_bps"].to_numpy(float) - tr_anchor
        consensus = []
        for cap in (25, 40, 60, 73):
            fields = ["base_anchor", "p_clear", "p_adverse", "p_weak", *context[:cap]]
            for equal in (False, True):
                local = train.copy(); local["_grade"] = np.digitize(resid, [-150, -50, 50, 150]).astype(np.int32)
                tr_raw, te_raw = _rank_fit(local, held, fields, local["_grade"].to_numpy(), equal, SEED + mi*100 + cap + int(equal))
                consensus.append(_pct(te_raw, tr_raw)); head_audit.append({"month":month,"head":f"cap{cap}_{'equal_month' if equal else 'ordinary'}","train_rows":len(train),"held_rows":len(held),"fields":len(fields),"query_groups":int(_groups(local)[1].size)})
        held["consensus"] = np.nanmedian(np.column_stack(consensus), axis=1).astype("float32")
        local = train.copy(); local["_grade"] = np.digitize(resid, [-100, -25, 25, 100]).astype(np.int32)
        rfields = ["base_anchor", "p_clear", "p_adverse", "p_weak", *context]
        rtr, rte = _rank_fit(local, held, rfields, local["_grade"].to_numpy(), True, SEED + mi*1000 + 99)
        held["residual"] = _pct(rte, rtr)
        held["base"] = held["base_score"].rank(pct=True, method="average").astype("float32")
        held["month"] = month
        out = held[["candidate_id", "__ts__", "month", "tp6_net_bps", "tp6_gross_bps", "policy_net_bps", "policy_gross_bps", "base", "consensus", "residual"]].copy()
        out = out.merge(incumbent, on="candidate_id", how="left", validate="one_to_one")
        out["new_tp6_base"] = out["base"]
        out["new_tp6_base_consensus25"] = .75*out["base"] + .25*out["consensus"]
        out["new_tp6_full_blend"] = .50*out["base"] + .25*out["consensus"] + .25*out["residual"]
        out["incumbent_r3_consensus"] = out["score__strict_r3_base_plus_consensus"]
        out["incumbent_r3_full"] = out["score__strict_r3_full"]
        scored.append(out)
        # Keep a bounded, labelled sample of each resolved held month for future
        # folds. The current month never contributes to its own fit.
        prior_samples.append(_sample(held.loc[held["label_valid"]], CURRENT_SAMPLE_PER_MONTH, SEED + 5000 + mi))
        print(json.dumps({"month":month,"train_rows":len(train),"held_rows":len(held),"consensus_heads":len(consensus)},), flush=True)
        del train, held, train_prob, held_prob
        gc.collect()
    pred = pd.concat(scored, ignore_index=True)
    newcols = ["new_tp6_base", "new_tp6_base_consensus25", "new_tp6_full_blend", "incumbent_r3_consensus", "incumbent_r3_full"]
    _norm_month(pred, newcols)
    g, m, s = _metrics(pred)
    pred.to_parquet(OUT/"predictions_2025.parquet", index=False, compression="zstd")
    g.to_parquet(OUT/"metrics_global.parquet", index=False); m.to_parquet(OUT/"metrics_monthly.parquet", index=False); s.to_parquet(OUT/"metrics_stability.parquet", index=False)
    pd.DataFrame(head_audit).to_parquet(OUT/"head_fit_audit.parquet", index=False)
    digest = hashlib.sha256("\n".join(context).encode()).hexdigest()
    manifest = {"schema":"full_universe_tp6_sl4_retrain_2025_v1","status":"COMPLETED","rows_scored":len(pred),"held_months":list(MONTHS),"side":"long","base_fields":base_fields,"base_field_count":len(base_fields),"context_fields":context,"context_field_count":len(context),"context_sha256":digest,"base_target":"TP6/SL4 robust-clear R3: class 2 robust_clear_event_b25, class 0 lower-first, class 1 valid weak/timeout","residual_target":"exact TP6/SL4 net bps minus train-only isotonic base anchor; consensus grades [-150,-50,50,150], residual grade [-100,-25,25,100]","consensus_heads":"4 caps (25,40,60,73) x ordinary/equal-month = 8 residual heads","query":"4-hour UTC x side LambdaRank","hpo":{"base_n_estimators":140,"base_learning_rate":0.05,"base_num_leaves":31,"base_min_child_samples":350,"base_subsample":0.8,"base_feature_fraction":0.8,"base_l2":8.0,"rank_n_estimators":140,"rank_learning_rate":0.035,"rank_max_depth":5,"rank_num_leaves":31,"rank_min_child_samples":180,"rank_feature_fraction":0.82,"rank_bagging_fraction":0.82,"rank_l1":0.02,"rank_l2":2.0,"rank_max_bin":127},"execution_outcomes":{"tp6":"12h TP6/SL4, 100 bps cost, 15m proxy","policy":"15m SL3/trailing activation 0.5 ATR/giveback 0.25 ATR, 100 bps cost"},"ranking":"monthly long percentile normalization followed by one pooled global top-k","incumbent_source":str(INCUMBENT),"no_held_month_outcomes_in_fit":True,"artifacts":["predictions_2025.parquet","metrics_global.parquet","metrics_monthly.parquet","metrics_stability.parquet","head_fit_audit.parquet","run_manifest.json"]}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2,default=str)+"\n")
    lines=["# Full-universe TP6/SL4 retraining — 2025", "", "All 1,463,365 long 2025 candidates are scored. The eight consensus heads are residual heads, not additional base probability heads.", "", "## Global metrics", "", g.round(3).to_string(index=False), "", "## Monthly top-5", "", m.round(3).to_string(index=False), "", "## Stability", "", s.round(3).to_string(index=False)]
    (OUT/"FULL_UNIVERSE_TP6_SL4_RETRAIN_2025_REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"output":str(OUT),"rows_scored":len(pred),"months":sorted(pred.month.unique().tolist())},indent=2))


if __name__ == "__main__":
    run()
