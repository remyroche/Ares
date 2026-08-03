#!/usr/bin/env python3
"""Chronological permutation-MDA selection for the R5 residual meta layer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG
from scripts.run_tp6_sl4_r3_base_meta_r5 import META_PARAMS, _map_apply, _map_fit, _matrix

# Bounded, diverse candidates from the configured meta-only universe.  They
# cover market return/volatility, breadth, OI/funding, correlation, crash /
# recovery and regime-transition mechanisms.  The MDA, not this ordering,
# decides the retained subset.
CANDIDATES = [
    "mkt_ret_eq_24h", "mkt_ret_eq_4h", "regime_liquidity_score", "mkt_rv_1h", "mkt_rv_4h", "mkt_rv_24h", "mkt_rv_ratio_1h_24h",
    "mkt_oi_breadth_rising_24h", "mkt_oi_chg_z_24h", "mkt_oi_dispersion_24h", "mkt_oi_flush_z_30d", "mkt_funding_dispersion", "mkt_funding_mean_z_30d",
    "cross_asset_corr_1h", "cross_asset_corr_4h", "cross_asset_downside_corr_4h", "return_dispersion_1h", "return_dispersion_4h",
    "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score", "mkt_leverage_rebuild_score", "liquidation_onset_score", "liquidation_climax_score", "post_liquidation_rebound_score",
    "negative_breadth_pct", "extreme_negative_breadth_pct", "downside_breadth_intensity", "btc_resilience_alt_weakness", "peer_decoupling_acceleration", "short_covering_score_market",
    "deleveraging_without_followthrough", "short_breakout_exhaustion", "funding_deleveraging_divergence", "funding_confirmed_short_covering", "funding_confirmed_long_flush",
    "short_signal_recovery_conflict", "late_short_after_deleveraging", "false_clean_short", "market_state_transition_entropy_5d", "market_state_persistence_5d",
    "breakout_efficiency_4h", "breakout_participation_4h", "breakout_retention_4h", "breakout_confirmation_ratio", "breakout_disagreement_score", "breakout_bilateral_failure_score",
    "xs_dispersion__volatility_zscore", "xs_dispersion__funding_per_hour", "xs_dispersion__trend_pct_mkt_resid", "q_lower_tail__ret48h_bench_resid", "q_upper_tail__ret48h_bench_resid",
]
BASE_INPUTS = ["prob_adverse", "prob_weak", "prob_clear", "base_expected_bps"]


def _load_context(panel: Path, ids: set[str], columns: list[str]) -> pd.DataFrame:
    pieces = []
    for part in sorted((panel / "parts").glob("*.parquet")):
        x = pd.read_parquet(part, columns=["candidate_id", *columns])
        x = x.loc[x.candidate_id.isin(ids)]
        if not x.empty:
            pieces.append(x)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--side", choices=("long", "short"), required=True)
    p.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    p.add_argument("--max-selected", type=int, default=30)
    args = p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    allowed = set(CFG["MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS"])
    if not set(CANDIDATES) <= allowed: raise ValueError(f"non-meta candidates: {sorted(set(CANDIDATES)-allowed)}")
    oof = pd.concat([pd.read_parquet(path) for path in args.oof], ignore_index=True).sort_values(["fold", "candidate_id"], kind="mergesort")
    mapped = []
    for fold, chunk in oof.groupby("fold", observed=True):
        history = oof.loc[oof.fold.lt(fold)]
        if history.empty: continue
        part = chunk.copy(); part["base_expected_bps"] = _map_apply(part.base_raw.to_numpy(float), _map_fit(history.base_raw.to_numpy(float), history.net_bps.to_numpy(float)))
        mapped.append(part)
    data = pd.concat(mapped, ignore_index=True)
    context = _load_context(args.panel, set(data.candidate_id), CANDIDATES)
    data = data.merge(context, on="candidate_id", how="inner", validate="one_to_one")
    data["residual_target"] = data.net_bps - data.base_expected_bps
    train, valid = data.loc[data.fold.lt(3)], data.loc[data.fold.eq(3)]
    features = [*BASE_INPUTS, *CANDIDATES]
    model = lgb.LGBMRegressor(objective="huber", alpha=.9, random_state=20261001, **META_PARAMS)
    model.fit(_matrix(train, features), train.residual_target.to_numpy(float))
    # Fixed deterministic 30k OOF validation sample; permutation never reaches
    # inference and each importance is an out-of-fold residual-MSE increase.
    valid = valid.sort_values("candidate_id", kind="mergesort").head(30_000).copy()
    x = _matrix(valid, features); y = valid.residual_target.to_numpy(float)
    baseline = mean_squared_error(y, model.predict(x))
    rows = []
    for index, name in enumerate(features):
        if name in BASE_INPUTS:
            continue
        losses = []
        for seed in (20261002, 20261003):
            z = x.copy(); rng = np.random.default_rng(seed); z[:, index] = rng.permutation(z[:, index])
            losses.append(mean_squared_error(y, model.predict(z)) - baseline)
        rows.append({"feature": name, "mda_mse_increase_mean": float(np.mean(losses)), "mda_mse_increase_std": float(np.std(losses)), "gain_importance": int(model.feature_importances_[index])})
    result = pd.DataFrame(rows).sort_values(["mda_mse_increase_mean", "gain_importance"], ascending=False, kind="mergesort")
    selected = result.loc[result.mda_mse_increase_mean.gt(0), "feature"].head(args.max_selected).tolist()
    args.out.mkdir(parents=True)
    result.to_parquet(args.out / "meta_mda_importance.parquet", index=False)
    decision = {"schema": "tp6_r5_meta_chronological_mda_v1", "status": "COMPLETED", "side": args.side,
                "selection": {"mandatory_base_inputs": BASE_INPUTS, "selected_meta_context": selected, "candidate_count": len(CANDIDATES), "max_selected": args.max_selected},
                "validation": {"fold": 3, "rows": len(valid), "baseline_residual_mse": float(baseline), "permutations_per_feature": 2},
                "lineage": "fit folds 1-2 only; evaluate permutation MDA on fold 3 only"}
    (args.out / "meta_mda_selection.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision))


if __name__ == "__main__": main()
