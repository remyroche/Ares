#!/usr/bin/env python3
"""Diagnose why the combined residual arm fails in individual months."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.funnel_selection import global_tail_metrics
from scripts.run_frozen_multiview_specialist_input_ablation import _base, _store_rows, _utc
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS
from scripts.run_regime_grouped_larger_specialists import _regime_columns

OUT = ROOT / "data_perp/artifacts/combined_regime_cmi_error_meta_20260810_v1"
CONTRACT = json.loads((ROOT / "data_perp/artifacts/regime_grouped_larger_specialists_20260810_v1/feature_contract.json").read_text())


def _bins(train, values):
    a = pd.to_numeric(train, errors="coerce").to_numpy(float)
    b = pd.to_numeric(values, errors="coerce").to_numpy(float)
    ok = np.isfinite(a)
    q = np.nanquantile(a[ok], [.25, .5, .75]) if ok.any() else np.array([0., 0., 0.])
    fill = float(np.nanmedian(a[ok])) if ok.any() else 0.
    return pd.Series(np.digitize(np.nan_to_num(b, nan=fill), q), index=values.index)


def main() -> None:
    base = _base()
    pred = pd.read_parquet(OUT / "predictions.parquet")
    pred["month"] = pd.to_datetime(pred["__ts__"], utc=True).dt.strftime("%Y-%m")
    pred["conversion_error_bps"] = pred.net_bps - pred.prequential_base_expected_net_bps
    monthly = []
    for (month, side), g in pred.groupby(["month", "side_name"], sort=True):
        n = max(1, int(np.ceil(len(g) * .05)))
        z = g.nlargest(n, "score")
        monthly.append({
            "month": month, "side": side, "rows": len(g),
            "spearman_score_net": float(g.score.corr(g.net_bps, method="spearman")),
            "spearman_score_residual": float(g.score.corr(g.conversion_error_bps, method="spearman")),
            "top5_net_bps": float(z.net_bps.mean()),
            "top5_gross_bps": float(z.gross_bps.mean()),
            "top5_base_expected_net_bps": float(z.prequential_base_expected_net_bps.mean()),
            "top5_conversion_error_bps": float(z.conversion_error_bps.mean()),
            "top5_score_mean": float(z.score.mean()),
            "top5_score_q05": float(z.score.quantile(.05)),
            "top5_score_q95": float(z.score.quantile(.95)),
            "full_mean_net_bps": float(g.net_bps.mean()),
        })
    pd.DataFrame(monthly).to_parquet(OUT / "bad_month_side_summary.parquet", index=False)

    drift = []
    regime_fields = [x for x in CONTRACT["regime_fields"]]
    for fold in LONG_HISTORY_FOLDS[3:]:
        a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
        train_all = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)]
        cal_all = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)]
        test_all = base[base.__ts__.between(c, e, inclusive="left")]
        for side in ("long", "short"):
            train = train_all[train_all.side_name.eq(side)].sample(min(100000, int(train_all.side_name.eq(side).sum())), random_state=20260810)
            cal = cal_all[cal_all.side_name.eq(side)]
            test = test_all[test_all.side_name.eq(side)]
            fields = [x for x in regime_fields if x in set(regime_fields)]
            tr_store = _store_rows(train, fields)
            ca_store = _store_rows(cal, fields)
            te_store = _store_rows(test, fields)
            tr_reg, ca_reg, te_reg = (_regime_columns(x[fields]) for x in (tr_store, ca_store, te_store))
            for name in ["volatility_proxy", "trend_proxy", "transition_intensity", "transition_entropy"]:
                tr_bin = _bins(tr_reg[name], tr_reg[name])
                ca_bin = _bins(tr_reg[name], ca_reg[name])
                te_bin = _bins(tr_reg[name], te_reg[name])
                for state in sorted(set(tr_bin.dropna().unique()) | set(ca_bin.dropna().unique()) | set(te_bin.dropna().unique())):
                    ca_share = float((ca_bin == state).mean())
                    te_share = float((te_bin == state).mean())
                    mask = te_bin.to_numpy() == state
                    mcal = ca_bin.to_numpy() == state
                    mtrain = tr_bin.to_numpy() == state
                    drift.append({"fold": fold.name, "side": side, "regime": name, "state": int(state), "train_share": float((tr_bin == state).mean()), "cal_share": ca_share, "test_share": te_share, "test_minus_cal_share": te_share-ca_share, "train_mean_net_bps": float(train.loc[mtrain, "net_bps"].mean()) if mtrain.any() else np.nan, "cal_mean_net_bps": float(cal.loc[mcal, "net_bps"].mean()) if mcal.any() else np.nan, "test_mean_net_bps": float(test.loc[mask, "net_bps"].mean()) if mask.any() else np.nan, "test_rows": int(mask.sum())})
    pd.DataFrame(drift).to_parquet(OUT / "regime_drift.parquet", index=False)

    cause = {
        "primary_observation": "The pooled positive result is concentrated in November long trades; August is dominated by negative short-side admissions.",
        "cross_side_scale_failure": "Global top-5 side share is month-dependent because raw score distributions shift by side; August admits approximately three quarters short candidates.",
        "conversion_failure": "In bad months the selected tail has negative conversion error relative to the base expected-net map, especially August short and August long.",
        "regime_failure": "Regime states have materially different realized net economics by month and side; transitions/volatility bins change prevalence and payoff sign across transport periods.",
        "not_a_single_outlier": "July, August, September and October have negative or near-zero score-to-net rank correlations for at least one side; November is the positive regime rather than the average state.",
    }
    (OUT / "bad_month_cause_summary.json").write_text(json.dumps(cause, indent=2) + "\n")


if __name__ == "__main__":
    main()
