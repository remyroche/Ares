#!/usr/bin/env python3
"""Chronological M6 conversion replay over pre-existing strict R3 base OOF folds.

This is intentionally a *historical-era diagnostic*, not a new winner search.
Every base probability is produced by a side-local model fitted before its row;
M6 sees only that same-side output and a fixed, causal, meta-only context pack.
For each held-out OOF fold, M6 fits on earlier folds, reserves the last fifth of
that history for an out-of-fit raw-score -> net-bps calibration, and then scores
the later fold.  It makes the 2023--24 evidence comparable to the 2022 common
core replay without selecting features on the evaluation outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("long", "short")
META_CONTEXT = [
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h", "mkt_oi_chg_z_24h",
    "mkt_funding_dispersion", "cross_asset_corr_4h", "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score", "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market", "deleveraging_without_followthrough",
    # The two omitted R5 fields, market_state_transition_entropy_5d and
    # breakout_retention_4h, cover only 4.7% of the 2023--24 OOF rows.  They
    # are deliberately not silently zero-imputed into this era comparison.
    "short_signal_recovery_conflict",
]
BASE_FIELDS = ["prob_adverse", "prob_weak", "prob_clear", "base_raw"]
TOPS = (.005, .01, .02, .05, .10)


def _read_context(panel: Path, ids: set[str]) -> pd.DataFrame:
    cols = ["candidate_id", *META_CONTEXT]
    chunks = []
    for part in sorted((panel / "parts").glob("*.parquet")):
        x = pd.read_parquet(part, columns=cols)
        x = x[x.candidate_id.isin(ids)]
        if not x.empty:
            chunks.append(x)
    return pd.concat(chunks, ignore_index=True)


def _matrix(frame: pd.DataFrame, fields: list[str]) -> np.ndarray:
    return frame[fields].replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy(np.float32)


def _model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=120, learning_rate=.04, num_leaves=24,
        min_child_samples=400, colsample_bytree=.8, subsample=.8, reg_lambda=12.,
        random_state=20260820, n_jobs=1, verbosity=-1,
    )


def _metrics(frame: pd.DataFrame, rank: str, group: dict[str, object]) -> list[dict[str, object]]:
    out = []
    ordered = frame.sort_values([rank, "meta_raw", "candidate_id"], ascending=[False, False, True], kind="mergesort")
    for top in TOPS:
        take = ordered.head(int(np.ceil(len(ordered) * top)))
        out.append({**group, "top_fraction": top, "n": len(take), "gross_bps": float(take.gross_bps.mean()), "net_bps": float(take.net_bps.mean())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    ap.add_argument("--long-oof", type=Path, nargs="+", required=True)
    ap.add_argument("--short-oof", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if len(a.long_oof) != len(a.short_oof) or len(a.long_oof) < 3:
        raise ValueError("need matched strict OOF paths for at least three chronological folds")
    frames = []
    for side, paths in (("long", a.long_oof), ("short", a.short_oof)):
        for fold, path in enumerate(paths):
            x = pd.read_parquet(path)
            if set(x.side_name.unique()) != {side}:
                raise ValueError(f"{path} is not a pure {side} OOF file")
            x["historical_fold"] = fold
            frames.append(x)
    data = pd.concat(frames, ignore_index=True)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
    if not np.allclose(data.gross_bps - 100., data.net_bps, atol=2e-3):
        raise ValueError("TP6/SL4 fixed-100-bps cost contract mismatch")
    context = _read_context(a.panel, set(data.candidate_id))
    data = data.merge(context, on="candidate_id", how="inner", validate="one_to_one")
    if len(data) != sum(len(x) for x in frames):
        raise ValueError("context join lost base-OOF rows")
    coverage = 1. - data[BASE_FIELDS + META_CONTEXT].replace([np.inf, -np.inf], np.nan).isna().mean()
    if (coverage < .90).any():
        raise ValueError(f"causal M6 feature coverage below 90%: {coverage[coverage < .90].to_dict()}")
    outputs, metrics = [], []
    features = BASE_FIELDS + META_CONTEXT
    for fold in range(1, len(a.long_oof)):
        for side in SIDES:
            train = data[(data.side_name == side) & (data.historical_fold < fold)].sort_values("__ts__", kind="mergesort").copy()
            evaluation = data[(data.side_name == side) & (data.historical_fold == fold)].copy()
            cut = train.__ts__.quantile(.80)
            early, calibration = train[train.__ts__ < cut], train[train.__ts__ >= cut]
            if min(len(early), len(calibration), len(evaluation)) < 1000:
                raise ValueError("insufficient chronological M6 support")
            y_early = (early.net_bps > 50.).astype(int)
            y_train = (train.net_bps > 50.).astype(int)
            first = _model().fit(_matrix(early, features), y_early)
            calibration_raw = first.predict_proba(_matrix(calibration, features))[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(calibration_raw, calibration.net_bps.to_numpy(float))
            final = _model().fit(_matrix(train, features), y_train)
            scored = evaluation[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "historical_fold"]].copy()
            scored["meta_raw"] = final.predict_proba(_matrix(evaluation, features))[:, 1]
            scored["side_calibrated_score_bps"] = iso.predict(scored.meta_raw)
            # Causal, training-fitted coarse volatility regime.  It is a
            # diagnostic only, never an input feature or a tailored label.
            # Liquidity score is degenerate in this older panel, so use the
            # materially varying causal volatility-ratio state instead.  This
            # is still an evaluation diagnostic, not a learned input change.
            q1, q2 = train.mkt_rv_ratio_1h_24h.quantile([1 / 3, 2 / 3]).to_numpy(float)
            if not q1 < q2:
                scored["regime"] = "support_insufficient"
            else:
                scored["regime"] = pd.cut(evaluation.mkt_rv_ratio_1h_24h, [-np.inf, q1, q2, np.inf], labels=["low_vol", "mid_vol", "high_vol"]).astype(str)
            outputs.append(scored)
    pred = pd.concat(outputs, ignore_index=True)
    for fold, part in pred.groupby("historical_fold", sort=True):
        metrics += _metrics(part, "side_calibrated_score_bps", {"view": "global", "fold": int(fold), "month": "all"})
        for (side, month), x in part.assign(month=part.__ts__.dt.to_period("M").astype(str)).groupby(["side_name", "month"], sort=True):
            metrics += _metrics(x, "side_calibrated_score_bps", {"view": side, "fold": int(fold), "month": month})
        for regime, x in part.groupby("regime", sort=True):
            metrics += _metrics(x, "side_calibrated_score_bps", {"view": f"regime:{regime}", "fold": int(fold), "month": "all"})
    metrics += _metrics(pred, "side_calibrated_score_bps", {"view": "global_pooled", "fold": -1, "month": "all"})
    a.out.mkdir(parents=True)
    pred.to_parquet(a.out / "predictions.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(a.out / "metrics.parquet", index=False)
    manifest = {
        "schema": "historical_r3_m6_rolling_v1", "status": "COMPLETED",
        "contract": {"geometry": "TP=+6 ATR / SL=-4 ATR / H12", "cost_bps": 100,
                     "base": "strict same-side chronological R3 OOF", "meta_target": "M6 P(exact net > +50 bps)",
                     "meta_context": "fixed 16 causal R5 context fields; no outcome-selected historical feature contract"},
        "folds": len(a.long_oof), "feature_coverage": coverage.to_dict(),
        "evaluation": "each fold is scored by an M6 model fit only on earlier base-OOF folds; calibration is held out from its M6 fit",
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(pred), "months": sorted(pred.__ts__.dt.to_period("M").astype(str).unique()), "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
