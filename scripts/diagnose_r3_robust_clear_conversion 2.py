#!/usr/bin/env python3
"""Diagnose conversion conditional on a realised R3 robust clear.

This is deliberately an *outcome diagnostic*, not an inference feature
generator.  It answers a narrowly defined question behind the failed M6
transport: after the frozen R3 event has occurred, what distinguishes an
exact-net success (>50 bps) from a weak conversion (<=50 bps), by historical
era and trade side?

All path quantities in ``robust_clear_paths.parquet`` use future one-minute
bars and are marked diagnostic-only.  The causal-field tables contain only
decision-time fields, and are useful for identifying stable candidates for a
subsequent conversion model -- they are not a feature selection result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.materialize_full_universe_tp6_sl4_h12_sidecar import _minute_path  # noqa: E402

COST_BPS = 100.0
BUFFER_BPS = 25.0
HORIZON_MINUTES = 720
SUCCESS_NET_BPS = 50.0

# Retained, broadly-covered context contract used in the historical M6 replay.
# These are evaluated only for the 2023--24 full-universe era.  The 2022 store
# has a different (but still causal) feature schema and is audited separately.
CURRENT_CONTEXT = [
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h", "mkt_oi_chg_z_24h",
    "mkt_funding_dispersion", "cross_asset_corr_4h", "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score", "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market", "deleveraging_without_followthrough",
    "short_signal_recovery_conflict",
]
OLD_PREFIXES = ("ret_", "rv_", "downside_", "atr_", "range_", "drawdown_", "recovery_", "trend_", "path_", "volume_", "jump_", "market_", "btc_minus_", "transition_raw__")


@njit(cache=True)
def _path_diagnostics(high: np.ndarray, low: np.ndarray, close: np.ndarray, starts: np.ndarray,
                      entry: np.ndarray, atr: np.ndarray, sides: np.ndarray) -> tuple[np.ndarray, ...]:
    """Compute post-event diagnostics; all results are future/outcome-only."""
    n = len(starts)
    valid = np.zeros(n, np.bool_)
    time_clear = np.full(n, np.nan, np.float32)
    mfe_after = np.full(n, np.nan, np.float32)
    terminal_giveback = np.full(n, np.nan, np.float32)
    mae_before = np.full(n, np.nan, np.float32)
    efficiency = np.full(n, np.nan, np.float32)
    for r in range(n):
        start, e, a, side = starts[r], entry[r], atr[r], sides[r]
        if start < 0 or start + HORIZON_MINUTES > len(close) or not np.isfinite(e) or not np.isfinite(a) or e <= 0. or a <= 0.:
            continue
        threshold = (COST_BPS + BUFFER_BPS) / (a / e * 10000.)
        clear_at = -1
        best = -1.e30
        worst_before = 0.
        path_abs = 0.
        complete = True
        previous = e
        for off in range(HORIZON_MINUTES):
            pos = start + off
            hi, lo, cl = high[pos], low[pos], close[pos]
            if not np.isfinite(hi) or not np.isfinite(lo) or not np.isfinite(cl):
                complete = False
                break
            if side > 0.:
                fav, adverse = (hi - e) / a, (e - lo) / a
                signed_return = (cl - e) / e
                step_return = (cl - previous) / previous
            else:
                fav, adverse = (e - lo) / a, (hi - e) / a
                signed_return = (e - cl) / e
                step_return = (previous - cl) / previous
            # Same-minute adverse touch invalidates a clear on that minute,
            # exactly matching the frozen R3 tie policy.
            if clear_at < 0:
                if adverse >= 4.:
                    break
                if fav * (a / e * 10000.) > COST_BPS + BUFFER_BPS:
                    clear_at = off
                else:
                    if adverse > worst_before:
                        worst_before = adverse
            if fav > best:
                best = fav
            path_abs += abs(step_return)
            previous = cl
        if not complete or clear_at < 0:
            continue
        terminal_fav = signed_return * e / a
        valid[r] = True
        time_clear[r] = clear_at + 1
        # Peak favourable movement after entering, expressed in ATR. It is
        # intentionally not an executable policy return.
        mfe_after[r] = best
        terminal_giveback[r] = best - terminal_fav
        mae_before[r] = worst_before
        efficiency[r] = abs(signed_return) / max(path_abs, 1.e-12)
    return valid, time_clear, mfe_after, terminal_giveback, mae_before, efficiency


def _read_parts(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    parts = sorted((path / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts under {path}")
    return pd.concat([pd.read_parquet(p, columns=columns) for p in parts], ignore_index=True)


def _read_old() -> tuple[pd.DataFrame, list[str]]:
    root = ROOT / "data_perp/artifacts/historical_2022_tp6_sl4_h12_20260809_v2"
    x = _read_parts(root)
    causal = [c for c in x.columns if c.startswith(OLD_PREFIXES)]
    x = x.loc[x.label_valid.eq(True) & x.robust_clear_event_b25.eq(True)].copy()
    x["era"] = "2022_Jan_Aug"
    x["__symbol_key__"] = x["__symbol__"].astype(str).str.replace("/", "_", regex=False)
    return x, causal


def _read_current() -> tuple[pd.DataFrame, list[str]]:
    # Restrict the full candidate population to exactly the rows that entered
    # the pre-existing strict historical M6 replay.  This avoids accidentally
    # analysing a post-selection or different candidate universe.
    replay = pd.read_parquet(ROOT / "data_perp/artifacts/historical_2023_2024_r3_m6_rolling_20260809_v1/predictions.parquet")
    ids = set(replay.candidate_id.astype(str))
    label_root = ROOT / "data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1"
    chunks = []
    for part in sorted((label_root / "parts").glob("*.parquet")):
        z = pd.read_parquet(part)
        z = z[z.candidate_id.isin(ids)]
        if not z.empty:
            chunks.append(z)
    labels = pd.concat(chunks, ignore_index=True)
    labels = labels.loc[labels.label_valid.eq(True) & labels.robust_clear_event_b25.eq(True)].copy()
    outcome = replay[["candidate_id", "gross_bps", "net_bps"]].drop_duplicates("candidate_id")
    labels = labels.merge(outcome, on="candidate_id", how="inner", validate="one_to_one")
    # Load only fixed causal fields, one candidate panel partition at a time.
    panel = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
    needed = ["candidate_id", *CURRENT_CONTEXT]
    chunks = []
    wanted = set(labels.candidate_id.astype(str))
    for part in sorted((panel / "parts").glob("*.parquet")):
        z = pd.read_parquet(part, columns=needed)
        z = z[z.candidate_id.isin(wanted)]
        if not z.empty:
            chunks.append(z)
    context = pd.concat(chunks, ignore_index=True)
    labels = labels.merge(context, on="candidate_id", how="inner", validate="one_to_one")
    labels["era"] = "2023Sep_2024Feb"
    labels["__symbol_key__"] = labels["__symbol__"].astype(str)
    return labels, CURRENT_CONTEXT


def _add_paths(frame: pd.DataFrame) -> pd.DataFrame:
    result = []
    minute_root = ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv"
    for symbol, group in frame.groupby("__symbol_key__", sort=True):
        g = group.copy()
        start, end = g.__decision_ts__.min(), g.__decision_ts__.max() + pd.Timedelta(minutes=HORIZON_MINUTES)
        minute = _minute_path(minute_root, str(symbol), start, end)
        starts = minute.index.get_indexer(pd.to_datetime(g.__decision_ts__, utc=True)).astype(np.int64)
        # The two source frames are concatenated before this routine, so both
        # column names exist with NaNs in the opposite era.  Prefer the
        # explicitly materialised old entry and fall back row-by-row to the
        # full-universe sidecar entry; never mistake the mere presence of a
        # column for complete data.
        entry = pd.to_numeric(g.get("entry_price"), errors="coerce").to_numpy(float) if "entry_price" in g else np.full(len(g), np.nan)
        if "tp6_sl4_entry_price" in g:
            entry = np.where(np.isfinite(entry), entry, g.tp6_sl4_entry_price.to_numpy(float))
        atr = pd.to_numeric(g.get("atr_1h"), errors="coerce").to_numpy(float) if "atr_1h" in g else np.full(len(g), np.nan)
        if "atr_bps" in g:
            atr = np.where(np.isfinite(atr), atr, g.atr_bps.to_numpy(float) * entry / 10000.)
        side = np.where(g.side_name.eq("long"), 1., -1.)
        valid, t, mfe, giveback, mae, eff = _path_diagnostics(
            minute.high.to_numpy(float), minute.low.to_numpy(float), minute.close.to_numpy(float), starts, entry, atr, side
        )
        g["path_diagnostic_complete"] = valid
        g["time_to_robust_clear_minutes"] = t
        g["mfe_after_clear_atr"] = mfe
        g["terminal_giveback_atr"] = giveback
        g["mae_before_clear_atr"] = mae
        g["path_efficiency"] = eff
        result.append(g)
    return pd.concat(result, ignore_index=True)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = frame[frame.path_diagnostic_complete].copy()
    x["month"] = pd.to_datetime(x.__ts__, utc=True).dt.to_period("M").astype(str)
    for keys, g in x.groupby(["era", "side_name", "month"], sort=True):
        for conversion, h in [("all_robust_clear", g), ("successful_net_gt_50", g[g.net_bps > SUCCESS_NET_BPS]), ("weak_net_le_50", g[g.net_bps <= SUCCESS_NET_BPS])]:
            rows.append({"era": keys[0], "side_name": keys[1], "month": keys[2], "conversion": conversion,
                         "support": len(h), "success_prevalence": float((g.net_bps > SUCCESS_NET_BPS).mean()),
                         "mean_net_bps": float(h.net_bps.mean()) if len(h) else np.nan,
                         "median_net_bps": float(h.net_bps.median()) if len(h) else np.nan,
                         "mean_time_to_clear_minutes": float(h.time_to_robust_clear_minutes.mean()) if len(h) else np.nan,
                         "median_time_to_clear_minutes": float(h.time_to_robust_clear_minutes.median()) if len(h) else np.nan,
                         "mean_mfe_after_clear_atr": float(h.mfe_after_clear_atr.mean()) if len(h) else np.nan,
                         "mean_terminal_giveback_atr": float(h.terminal_giveback_atr.mean()) if len(h) else np.nan,
                         "mean_mae_before_clear_atr": float(h.mae_before_clear_atr.mean()) if len(h) else np.nan,
                         "mean_path_efficiency": float(h.path_efficiency.mean()) if len(h) else np.nan})
    return pd.DataFrame(rows)


def _field_separation(frame: pd.DataFrame, fields: list[str], origin: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = frame.copy()
    x["month"] = pd.to_datetime(x.__ts__, utc=True).dt.to_period("M").astype(str)
    records = []
    for (era, side, month), g in x.groupby(["era", "side_name", "month"], sort=True):
        y = (g.net_bps > SUCCESS_NET_BPS).to_numpy(int)
        if y.min() == y.max():
            continue
        for field in fields:
            a = pd.to_numeric(g[field], errors="coerce").to_numpy(float)
            mask = np.isfinite(a)
            if mask.sum() < 100 or np.unique(y[mask]).size < 2:
                continue
            yes, no = a[mask & (y == 1)], a[mask & (y == 0)]
            pooled = np.sqrt((yes.var() + no.var()) / 2.)
            # A constant causal field has no separating power.  It is omitted
            # rather than emitted as a misleading NaN stability record.
            if not np.isfinite(pooled) or pooled <= 1.e-12:
                continue
            effect = (yes.mean() - no.mean()) / pooled
            ranks = rankdata(a[mask])
            # AUC sign is oriented such that positive means higher field value
            # is associated with successful conversion.
            auc = (ranks[y[mask] == 1].sum() - len(yes) * (len(yes) + 1) / 2.) / (len(yes) * len(no))
            records.append({"origin": origin, "era": era, "side_name": side, "month": month, "field": field,
                            "support": int(mask.sum()), "coverage": float(mask.mean()), "success_rate": float(y[mask].mean()),
                            "standardized_mean_difference": float(effect), "single_field_auc": float(auc),
                            "mean_success": float(yes.mean()), "mean_weak": float(no.mean())})
    detail = pd.DataFrame(records)
    stable = []
    if not detail.empty:
        for (origin_, era, side, field), g in detail.groupby(["origin", "era", "side_name", "field"], sort=True):
            effect = g.standardized_mean_difference.dropna().to_numpy(float)
            if not len(effect):
                continue
            stable.append({"origin": origin_, "era": era, "side_name": side, "field": field, "months": len(g),
                           "median_effect": float(np.median(effect)), "effect_mad": float(np.median(np.abs(effect - np.median(effect)))),
                           "same_sign_month_fraction": float(max((effect > 0).mean(), (effect < 0).mean())),
                           "median_auc": float(g.single_field_auc.median()), "min_coverage": float(g.coverage.min()),
                           "eligible_for_cross_month_followup": bool(len(g) >= 2 and max((effect > 0).mean(), (effect < 0).mean()) >= .75)})
    return detail, pd.DataFrame(stable)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    old, old_fields = _read_old()
    current, current_fields = _read_current()
    raw = pd.concat([old, current], ignore_index=True, sort=False)
    paths = _add_paths(raw)
    if not np.allclose(paths.gross_bps - COST_BPS, paths.net_bps, atol=2e-3):
        raise ValueError("fixed 100-bps cost contract failed")
    summary = _summary(paths)
    old_detail, old_stable = _field_separation(paths[paths.era.eq("2022_Jan_Aug")], old_fields, "2022_causal_store")
    current_detail, current_stable = _field_separation(paths[paths.era.eq("2023Sep_2024Feb")], current_fields, "2023_24_fixed_m6_context")
    a.out.mkdir(parents=True)
    keep = ["candidate_id", "era", "__ts__", "__symbol__", "side_name", "gross_bps", "net_bps", "path_diagnostic_complete",
            "time_to_robust_clear_minutes", "mfe_after_clear_atr", "terminal_giveback_atr", "mae_before_clear_atr", "path_efficiency"]
    paths[keep].to_parquet(a.out / "robust_clear_paths.parquet", index=False)
    summary.to_parquet(a.out / "robust_clear_conversion_summary.parquet", index=False)
    pd.concat([old_detail, current_detail], ignore_index=True).to_parquet(a.out / "causal_field_separation.parquet", index=False)
    pd.concat([old_stable, current_stable], ignore_index=True).to_parquet(a.out / "causal_field_effect_stability.parquet", index=False)
    manifest = {"schema": "r3_robust_clear_conversion_diagnostic_v1", "status": "COMPLETED",
                "scope": "realised valid R3 b25 robust-clear rows from 2022 Jan--Aug and strict historical M6 replay rows 2023 Sep--2024 Feb",
                "contract": {"geometry": "TP +6 ATR / SL -4 ATR / H12", "entry": "decision +1h exact minute open", "cost_bps": COST_BPS,
                             "r3_clear": "pre-adverse MFE exceeds cost +25 bps; same-minute lower touch wins", "success": "exact net > +50 bps"},
                "diagnostic_only_path_fields": ["time_to_robust_clear_minutes", "mfe_after_clear_atr", "terminal_giveback_atr", "mae_before_clear_atr", "path_efficiency"],
                "causal_field_note": "separation/stability tables contain only decision-time causal fields; schemas differ by era and are not concatenated as one common feature contract"}
    (a.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(paths), "path_complete": int(paths.path_diagnostic_complete.sum()), "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
