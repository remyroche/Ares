#!/usr/bin/env python3
"""Matched multi-base canonical replay on the canonical Jan--Dec 2025 panel.

This is the same downstream contract as the canonical single-R3 replay, with
four frozen complementary base heads added.  Each base signal is mapped
independently to TP6/SL4 net bps using only resolved rows before the held
month.  Eight residual consensus heads are then fit to the residual from the
median mapped base anchor.  Outputs are monthly side-normalised and ranked
once on the pooled global panel.
"""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/remyroche/Documents/Ares")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.complementary_base_heads import agreement_features  # noqa: E402
from extreme_price_movements.query_candidate_definitions import QueryDefinition  # noqa: E402
from extreme_price_movements.trailing_exit_grid import net_bps, simulate_h12_stop_trailing_grid  # noqa: E402
from scripts.run_complementary_base_heads_alternate_exit import _policy_labels  # noqa: E402
from scripts.run_multibase_canonical_reconciliation_2024 import (  # noqa: E402
    BASE_IDS,
    CONSENSUS_CAPS,
    _base_features,
    _cdf,
    _head_contract,
    _iso_map,
    _rank_fit,
    _rank_month,
    _selected_context,
    _target,
)

INPUT = ROOT / "data_perp/artifacts/r3_tp6_sl4_meta_target_ablation_20260803_v1/r3_meta_target_oof_predictions.parquet"
SELECTOR = ROOT / "data_perp/artifacts/stage_i_selector_sample_20260803_v5/selector_features.parquet"
HEAD_ROOT = ROOT / "data_perp/artifacts/complementary_base_heads_20260808_v2"
PATH_ROOT = ROOT / "data_perp/artifacts/h12_query_path_grid_20260805_v2"
BARS_ROOT = ROOT / "15m_ohlcv_perp"
SIDECAR_ROOT = ROOT / "data_perp/artifacts/current_2025_r3_proxy_labels_15m_20260807_v1/parts"
OUT = ROOT / "data_perp/artifacts/multibase_canonical_reconciliation_2025_20260808_v1"
MONTHS = tuple(f"2025-{m:02d}" for m in range(1, 13))
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


def _contracts() -> dict[str, dict]:
    return {h: _head_contract(h) for h in BASE_IDS if h != "r3"}


def _load() -> tuple[pd.DataFrame, list[str], dict[str, dict]]:
    context = _selected_context()
    contracts = _contracts()
    union = sorted(set().union(*[set(c["features"]["long"]) for c in contracts.values()]))
    required = [
        "candidate_id", "__ts__", "__symbol__", "side_name", "label_valid",
        "label_available_ts", "exact_net_bps", "exact_gross_bps",
        "t2_tp6_sl4_event", "r3_meta_p_adverse",
        "r3_meta_p_weak", "r3_meta_p_clear",
    ]
    panel = pd.read_parquet(INPUT, columns=list(dict.fromkeys(required + context + ["decision_ts"])))
    panel["__ts__"] = pd.to_datetime(panel["__ts__"], utc=True)
    panel["label_available_ts"] = pd.to_datetime(panel["label_available_ts"], utc=True)
    panel["__symbol__"] = panel["__symbol__"].astype(str)
    panel["side_name"] = panel["side_name"].astype(str).str.lower()
    # Materialise the frozen complementary-head inputs from the causal feature
    # store.  The join is exact symbol/timestamp; no as-of or forward fill.
    aux_fields = [f for f in union if f not in panel.columns]
    aux = pd.read_parquet(SELECTOR, columns=["__symbol__", "__ts__", *aux_fields])
    aux["__ts__"] = pd.to_datetime(aux["__ts__"], utc=True)
    for f in aux_fields:
        nunique = aux.groupby(["__symbol__", "__ts__"], sort=False)[f].nunique(dropna=True)
        if int((nunique > 1).sum()):
            raise ValueError(f"conflicting selector values for {f}")
    aux = aux.groupby(["__symbol__", "__ts__"], sort=False)[aux_fields].first().reset_index()
    panel = panel.merge(aux, on=["__symbol__", "__ts__"], how="left", validate="many_to_one")
    panel = panel[panel.side_name.eq("long")].copy()
    panel["month"] = panel["__ts__"].dt.strftime("%Y-%m")
    panel = panel[panel.month.isin([f"2024-{m:02d}" for m in range(2, 13)] + list(MONTHS))].copy()
    valid = panel.label_valid.fillna(False) & np.isfinite(panel.exact_net_bps) & np.isfinite(panel.exact_gross_bps)
    panel = panel.loc[valid].sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if panel.candidate_id.duplicated().any():
        raise ValueError("duplicate candidate IDs")
    for month in MONTHS:
        if len(panel[panel.month.eq(month)]) != 852:
            raise ValueError(f"expected 852 long rows in {month}")
    missing = [f for f in union if panel[f].notna().mean() < 0.01]
    if missing:
        raise ValueError(f"base feature join failed: {missing[:10]}")
    panel["base_score"] = pd.to_numeric(panel.r3_meta_p_clear, errors="coerce") - .5 * pd.to_numeric(panel.r3_meta_p_adverse, errors="coerce")
    return panel, context, contracts


def _metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arms = ["base_only", "consensus_only", "joint_75_25", *[f"base_{b}" for b in BASE_IDS]]
    exits = [
        ("tp6_sl4", "exact_net_bps", "exact_gross_bps"),
        ("trailing_15m", "alternate_policy_net_bps", "alternate_policy_gross_bps"),
    ]
    global_rows, month_rows = [], []
    for ex, net, gross in exits:
        scored = pred if ex == "tp6_sl4" else pred[pred.alternate_policy_valid].copy()
        for arm in arms:
            for tail in TAILS:
                n = max(1, int(math.ceil(len(scored) * tail)))
                top = scored.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
                global_rows.append({"exit": ex, "arm": arm, "scope": "pooled_global_2025", "tail": tail, "trades": n,
                    "gross_bps_per_trade": float(top[gross].mean()), "net_bps_per_trade": float(top[net].mean()),
                    "rank_ic": float(scored[[arm, net]].corr(method="spearman").iloc[0, 1]), "support_rows": len(scored)})
            for month, g in scored.groupby("month", sort=True):
                n = max(1, int(math.ceil(len(g) * .05)))
                top = g.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
                month_rows.append({"exit": ex, "arm": arm, "month": month, "tail": .05, "trades": n,
                    "gross_bps_per_trade": float(top[gross].mean()), "net_bps_per_trade": float(top[net].mean()),
                    "rank_ic": float(g[[arm, net]].corr(method="spearman").iloc[0, 1])})
    monthly = pd.DataFrame(month_rows)
    stability = []
    for (ex, arm), g in monthly.groupby(["exit", "arm"], sort=True):
        values = g.net_bps_per_trade.to_numpy(float)
        med = float(np.nanmedian(values))
        stability.append({"exit": ex, "arm": arm, "months": len(values),
            "mean_top5_net_bps": float(np.nanmean(values)), "median_top5_net_bps": med,
            "mad_top5_net_bps": float(np.nanmedian(np.abs(values - med))),
            "worst_month_top5_net_bps": float(np.nanmin(values)),
            "positive_months_top5": int((values > 0).sum())})
    return pd.DataFrame(global_rows), monthly, pd.DataFrame(stability)


def _policy_labels_2025(pred: pd.DataFrame) -> pd.DataFrame:
    """Materialise the frozen policy from the 2025 ATR sidecar and 15m bars.

    The historical path grid stops in 2024 and also uses a different symbol
    spelling.  The 2025 source partitions contain the same causal decision
    ATR, while the entry is reconstructed exactly as decision timestamp +1h
    at the first 15m open.
    """
    ids = set(pred.candidate_id.astype(str))
    parts = []
    for f in sorted(SIDECAR_ROOT.glob("month=2025-*/*.parquet")):
        z = pd.read_parquet(f, columns=["candidate_id", "__symbol__", "side_name", "atr_bps", "decision_ts"])
        z["candidate_id"] = z.candidate_id.astype(str)
        z = z[z.candidate_id.isin(ids)]
        if not z.empty:
            parts.append(z)
    meta = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["candidate_id", "__ts__", "__symbol__", "side_name", "atr_bps"])
    if meta.candidate_id.duplicated().any():
        raise ValueError("duplicate 2025 ATR sidecar candidate IDs")
    out = pred[["candidate_id", "__ts__", "side_name"]].copy()
    out["alternate_policy_net_bps"] = np.nan; out["alternate_policy_gross_bps"] = np.nan; out["alternate_policy_atr_bps"] = np.nan
    out["alternate_policy_valid"] = False
    out["alternate_policy_entry_ts"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    for symbol, group in meta.groupby("__symbol__", sort=True):
        bar_name = str(symbol).lower().replace("/", "").replace("_", "") + "_15m.parquet"
        bar_file = BARS_ROOT / bar_name
        if not bar_file.exists():
            raise FileNotFoundError(bar_file)
        bars = pd.read_parquet(bar_file)
        time_col = next((c for c in ("ts", "timestamp", "__index_level_0__") if c in bars.columns), None)
        if time_col is not None:
            bars = bars.set_index(time_col)
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars = bars.loc[:, ["open", "high", "low", "close"]].loc[~bars.index.duplicated(keep="last")].sort_index()
        z = group.merge(pred[["candidate_id", "__ts__", "side_name"]], on=["candidate_id"], how="left", validate="one_to_one")
        entry_ts = pd.to_datetime(z["__ts__"], utc=True) + pd.Timedelta(hours=1)
        starts = bars.index.get_indexer(entry_ts)
        entry = bars.open.to_numpy(float)
        atr_bps = pd.to_numeric(z.atr_bps, errors="coerce").to_numpy(float)
        valid = (starts >= 0) & (starts + 48 <= len(bars)) & np.isfinite(atr_bps) & (atr_bps > 0)
        if valid.any():
            sv = starts[valid].astype(np.int64)
            valid[valid] &= (bars.index.to_numpy()[sv + 47] - bars.index.to_numpy()[sv] == np.timedelta64(47 * 15, "m"))
        if not valid.any():
            continue
        zv = z.loc[valid]
        sv = starts[valid].astype(np.int64); e = entry[sv].astype(np.float32); a_bps = atr_bps[valid].astype(np.float32)
        a = e * a_bps / 10_000.0
        side = np.ones(len(zv), dtype=np.float32)
        gross_atr = simulate_h12_stop_trailing_grid(bars.high.to_numpy(float), bars.low.to_numpy(float), bars.close.to_numpy(float), sv, e, a, side, np.asarray([3.0], dtype=np.float32), np.asarray([0.5], dtype=np.float32), np.asarray([0.25], dtype=np.float32), horizon_bars=48).reshape(-1)
        net = net_bps(gross_atr.reshape(-1, 1, 1, 1), a_bps, cost_bps=100.0).reshape(-1)
        for cid, ets, n, atrv in zip(zv.candidate_id, entry_ts.loc[zv.index], net, a_bps):
            loc = out.index[out.candidate_id.eq(cid)]
            if len(loc) != 1:
                raise ValueError(f"policy candidate mismatch {cid}")
            loc = loc[0]; out.loc[loc, "alternate_policy_net_bps"] = float(n); out.loc[loc, "alternate_policy_gross_bps"] = float(n + 100.0); out.loc[loc, "alternate_policy_atr_bps"] = float(atrv); out.loc[loc, "alternate_policy_valid"] = bool(np.isfinite(n)); out.loc[loc, "alternate_policy_entry_ts"] = ets
    offsets = pd.to_datetime(out.loc[out.alternate_policy_valid, "alternate_policy_entry_ts"], utc=True) - pd.to_datetime(out.loc[out.alternate_policy_valid, "__ts__"], utc=True)
    if len(offsets) and not (offsets == pd.Timedelta(hours=1)).all():
        raise AssertionError("trailing entry is not exactly +1h")
    return out


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, context, contracts = _load()
    base_specs = {"r3": None, **contracts}
    pieces, fit_audit, windows = [], [], []
    for month in MONTHS:
        start = pd.Timestamp(month, tz="UTC")
        held = frame[frame.month.eq(month)].copy()
        train = frame[(frame.__ts__ < start) & (frame.label_available_ts < start)].copy()
        if len(held) != 852 or len(train) < 1000:
            raise ValueError(f"invalid split {month}: train={len(train)} held={len(held)}")
        tr_out = train[["candidate_id", "__ts__", "side_name", "month", "exact_net_bps"]].copy().reset_index(drop=True)
        te_out = held[["candidate_id", "__ts__", "side_name", "month", "exact_net_bps", "exact_gross_bps"]].copy().reset_index(drop=True)
        for c in ("r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak"):
            tr_out[c] = pd.to_numeric(train[c], errors="coerce").to_numpy(float)
            te_out[c] = pd.to_numeric(held[c], errors="coerce").to_numpy(float)
        raw_tr, raw_te, anc_tr, anc_te = {}, {}, {}, {}
        for base_id, contract in base_specs.items():
            if base_id == "r3":
                rt = train.base_score.to_numpy(float); rh = held.base_score.to_numpy(float)
            else:
                fields = list(contract["features"]["long"])
                rt, rh = _rank_fit(train, held, fields, _target(contract, train), QueryDefinition(**contract["query"]), contract["params"])
            at, ah = _iso_map(rt, train.exact_net_bps.to_numpy(float), rh)
            raw_tr[base_id], raw_te[base_id], anc_tr[base_id], anc_te[base_id] = rt, rh, at, ah
            fit_audit.append({"month": month, "layer": "base", "base_id": base_id, "train_rows": len(train), "held_rows": len(held), "label_available_before": str(start), "target": "r3_score" if base_id == "r3" else contract["target"], "query": "none" if base_id == "r3" else contract["query"]["name"], "feature_count": 0 if base_id == "r3" else len(contract["features"]["long"])})
        base_names = list(base_specs)
        for b in base_names:
            tr_out[f"{b}__raw"] = raw_tr[b]; te_out[f"{b}__raw"] = raw_te[b]
            tr_out[f"{b}__anchor"] = anc_tr[b]; te_out[f"{b}__anchor"] = anc_te[b]
            tr_out[f"{b}__cdf_rank"] = _cdf(raw_tr[b], raw_tr[b]); te_out[f"{b}__cdf_rank"] = _cdf(raw_tr[b], raw_te[b])
        ranks = [f"{b}__cdf_rank" for b in base_names]
        agree = agreement_features(pd.concat([tr_out, te_out], ignore_index=True), ranks)
        agree.columns = [f"base_heads_{c.removeprefix('base_heads_')}" for c in agree.columns]
        tr_out = pd.concat([tr_out, agree.iloc[:len(train)].reset_index(drop=True)], axis=1)
        te_out = pd.concat([te_out, agree.iloc[len(train):].reset_index(drop=True)], axis=1)
        tr_out["base_anchor_median"] = np.median(np.column_stack([anc_tr[b] for b in base_names]), axis=1).astype(np.float32)
        te_out["base_anchor_median"] = np.median(np.column_stack([anc_te[b] for b in base_names]), axis=1).astype(np.float32)
        tr_out["base_residual"] = tr_out.exact_net_bps.to_numpy(float) - tr_out.base_anchor_median.to_numpy(float)
        te_out["base_residual"] = te_out.exact_net_bps.to_numpy(float) - te_out.base_anchor_median.to_numpy(float)
        tr_out = pd.concat([tr_out, train[context].reset_index(drop=True)], axis=1)
        te_out = pd.concat([te_out, held[context].reset_index(drop=True)], axis=1)
        residual_fields = _base_features(tr_out, context, base_names)
        grades = np.digitize(tr_out.base_residual.to_numpy(float), [-150., -50., 50., 150.]).astype(np.int32)
        consensus_ranks = []
        for cap in CONSENSUS_CAPS:
            fields = [f for f in residual_fields if f not in context or f in context[:cap]]
            for equal_month in (False, True):
                rt, rh = _rank_fit(tr_out, te_out, fields, grades, QueryDefinition("residual_q4h_side", "cycle", cycle_hours=4), {
                    "objective": "lambdarank", "metric": "ndcg", "n_estimators": 140, "learning_rate": .035,
                    "max_depth": 5, "num_leaves": 31, "min_child_samples": 180, "feature_fraction": .82,
                    "bagging_fraction": .82, "bagging_freq": 1, "lambda_l1": .02, "lambda_l2": 2.,
                    "max_bin": 127, "lambdarank_truncation_level": 10, "label_gain": [0., .25, 1., 3., 7.]}, equal_month=equal_month)
                consensus_ranks.append(_cdf(rt, rh))
                fit_audit.append({"month": month, "layer": "consensus", "base_id": "multi_base", "cap": cap, "equal_month": equal_month, "train_rows": len(train), "held_rows": len(held), "label_available_before": str(start), "target": "exact_net_minus_median_base_anchor_grades_[-150,-50,50,150]", "query": "residual_q4h_side", "feature_count": len(fields)})
        consensus = np.nanmedian(np.column_stack(consensus_ranks), axis=1).astype(np.float32)
        te_out["base_rank"] = _rank_month(te_out.base_anchor_median.to_numpy(float)); te_out["consensus_rank"] = _rank_month(consensus)
        te_out["base_only"] = te_out.base_rank; te_out["consensus_only"] = te_out.consensus_rank; te_out["joint_75_25"] = .75 * te_out.base_rank + .25 * te_out.consensus_rank
        for b in base_names:
            te_out[f"base_{b}"] = _rank_month(te_out[f"{b}__anchor"].to_numpy(float))
        keep = ["candidate_id", "__ts__", "side_name", "month", "exact_net_bps", "exact_gross_bps", "base_anchor_median", "consensus_rank", "base_only", "consensus_only", "joint_75_25", *[f"base_{b}" for b in base_names], *[f"{b}__raw" for b in base_names], *[f"{b}__anchor" for b in base_names]]
        pieces.append(te_out[keep].copy())
        windows.append({"month": month, "base_train_rows": len(train), "held_rows": len(held), "base_models": 5, "consensus_heads": 8, "train_cutoff": str(start)})
    pred = pd.concat(pieces, ignore_index=True)
    pred.to_parquet(OUT / "predictions_2025_long_prepolicy.parquet", index=False, compression="zstd")
    policy_input = pred[["candidate_id", "__ts__", "side_name"]].copy()
    policy = _policy_labels_2025(policy_input)
    pred = pred.merge(policy.drop(columns=["__ts__", "side_name"]), on="candidate_id", how="left", validate="one_to_one")
    g, m, s = _metrics(pred)
    pred.to_parquet(OUT / "predictions_2025_long.parquet", index=False, compression="zstd")
    g.to_parquet(OUT / "metrics_global.parquet", index=False); m.to_parquet(OUT / "metrics_monthly.parquet", index=False); s.to_parquet(OUT / "metrics_stability.parquet", index=False)
    pd.DataFrame(fit_audit).to_parquet(OUT / "fit_audit.parquet", index=False); pd.DataFrame(windows).to_parquet(OUT / "monthly_training_windows.parquet", index=False)
    manifest = {"schema": "multibase_canonical_reconciliation_2025_v1", "status": "COMPLETED", "population": "long-only Jan-Dec 2025, 852 rows/month", "input": str(INPUT), "base_feature_source": str(SELECTOR), "base_models": {b: ("R3 p_clear - .5 p_adverse" if b == "r3" else {"target": contracts[b]["target"], "query": contracts[b]["query"], "feature_count": len(contracts[b]["features"]["long"])}) for b in BASE_IDS}, "base_mapping": "independent train-only isotonic TP6/SL4 net map per base; row-wise median anchor", "agreement_features": ["base_heads_frac_rank_ge_p99", "base_heads_frac_rank_ge_p95", "base_heads_frac_rank_ge_p90", "base_heads_weighted_mean_conviction", "base_heads_median_conviction", "base_heads_prediction_dispersion", "base_heads_prediction_std", "base_heads_prediction_iqr", "base_heads_agreement_entropy"], "consensus": "8 LambdaRank residual heads, context caps 25/40/60/73 x ordinary/equal-month", "query": "4-hour UTC x side", "normalization": "monthly side-local percentile rank then pooled global ranking", "final_score": "0.75 * median-base-rank + 0.25 * consensus-rank", "exits": {"tp6_sl4": "exact H12 TP6/SL4, 100 bps cost", "trailing_15m": "decision +1h 15m open, 48 bars, SL3 ATR, activation .5 ATR, giveback .25 ATR, 100 bps cost"}, "rows": len(pred), "trailing_policy_support_rows": int(pred.alternate_policy_valid.sum()), "trailing_policy_support_by_month": pred.groupby("month").alternate_policy_valid.sum().to_dict(), "no_held_month_outcomes_in_fit": True}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    lines = ["# Matched multi-base canonical reconciliation — long-only Jan–Dec 2025", "", "Five independently mapped base signals feed the canonical eight-head residual consensus. All fits use only resolved rows before each held month.", "", "## Pooled global metrics", "", g.round(3).to_string(index=False), "", "## Monthly top-5 metrics", "", m[m["tail"].eq(.05)].round(3).to_string(index=False), "", "## Stability", "", s.round(3).to_string(index=False)]
    (OUT / "MULTIBASE_CANONICAL_RECONCILIATION_2025_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT), "rows": len(pred), "months": sorted(pred.month.unique().tolist()), "global_rows": len(g)}, indent=2))


if __name__ == "__main__":
    run()
