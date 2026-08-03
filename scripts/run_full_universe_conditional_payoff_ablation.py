#!/usr/bin/env python3
"""Stage 5.2: bounded conditional-payoff replacement for frozen B2.

The base event simplex is frozen.  This experiment changes *only* the payoff
term in ``sum_event p(event) E[gross | event, side, context] - cost``.  The
context is deliberately limited to decision-time ATR/volatility, liquidity,
base entropy, and calendar regime fields.  A ridge estimate for every
side/event cell is strongly shrunk to its same-cell historical mean.

Selection is performed once on a chronological development hold-out; the
selected shrinkage is then re-fit on all labels resolved before 2024-08-01
and used unchanged for Aug--Nov.  No OOS outcome is read while fitting either
the B2 control or conditional model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROBS = ("p_upper", "p_lower", "p_timeout")
EVENTS = range(3)
SIDES = ("long", "short")
TOPS = (0.01, 0.05, 0.10, 0.20)
# Fixed and deliberately tiny.  TP/SL geometry and the 100 bps cost are
# metadata constants for this frozen TP3/SL2 experiment, rather than fake
# row features.
RAW_CONTEXT = (
    "atr_1h", "rv_24h", "mkt_rv_24h", "prior_volatility",
    "amihud_z", "ob_spread_bps_z_24h", "liquidity_ratio_peer_resid",
    "xasset_mkt_spread_bps", "assumed_round_trip_cost_bps",
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    p.add_argument("--base", type=Path, default=ROOT / "data_perp/artifacts/full_universe_base_hpo_20260802_v1")
    p.add_argument("--out", type=Path, default=ROOT / "data_perp/artifacts/full_universe_stage5_2_conditional_payoff_20260804_v1")
    p.add_argument("--dev-train-end", default="2024-06-15", help="exclusive resolved-label fit end for development")
    p.add_argument("--dev-end", default="2024-08-01", help="exclusive development evaluation end")
    p.add_argument("--oos-end", default="2024-12-01")
    return p.parse_args()


def _base(root: Path) -> pd.DataFrame:
    cols = ["candidate_id", "__ts__", "side_name", *PROBS]
    x = pd.concat([pd.read_parquet(root / side / "target_screen_predictions.parquet", columns=cols) for side in SIDES], ignore_index=True)
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    if x.candidate_id.duplicated().any():
        raise ValueError("base candidate IDs must be unique")
    if not np.allclose(x.loc[:, PROBS].to_numpy(float).sum(1), 1.0, atol=1e-5):
        raise ValueError("frozen base probabilities are not a closed simplex")
    return x


def _panel(path: Path) -> pd.DataFrame:
    cols = ["candidate_id", "__ts__", "__label_available_at__", "side_name", "t2_tp3_sl2_event", "t4_tp3_sl2_gross_bps", "t4_tp3_sl2_net_bps", *RAW_CONTEXT]
    frames = [pd.read_parquet(p, columns=cols) for p in sorted((path / "parts").glob("*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"no parts below {path}")
    x = pd.concat(frames, ignore_index=True).rename(columns={"t2_tp3_sl2_event": "event", "t4_tp3_sl2_gross_bps": "gross_bps", "t4_tp3_sl2_net_bps": "net_bps"})
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    x["__label_available_at__"] = pd.to_datetime(x["__label_available_at__"], utc=True)
    if not x.event.isin(EVENTS).all():
        raise ValueError("unexpected TP3/SL2 event")
    return x


def _features(x: pd.DataFrame) -> pd.DataFrame:
    out = x.loc[:, RAW_CONTEXT].copy()
    # Decision calendar is a causal time-regime proxy.  Use a smooth cycle,
    # not month dummies which would invite a one-off regime lookup.
    hour = x["__ts__"].dt.hour.to_numpy(float)
    dow = x["__ts__"].dt.dayofweek.to_numpy(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    p = x.loc[:, PROBS].to_numpy(float)
    safe = np.clip(p, 1e-12, 1.0)
    out["base_entropy"] = -(safe * np.log(safe)).sum(1) / np.log(3.0)
    return out.replace([np.inf, -np.inf], np.nan)


def _means(train: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    g = train.groupby("event", observed=True).gross_bps.mean().reindex(EVENTS)
    if g.isna().any():
        raise ValueError("fit window lacks an event")
    glob = g.to_numpy(float)
    by_side: dict[str, np.ndarray] = {}
    for side in SIDES:
        z = train[train.side_name.eq(side)].groupby("event", observed=True).gross_bps.mean().reindex(EVENTS).fillna(g)
        by_side[side] = z.to_numpy(float)
    return glob, by_side


def _models(train: pd.DataFrame, alpha: float) -> tuple[dict[tuple[str, int], object], dict[tuple[str, int], float], dict[str, np.ndarray]]:
    _, base_means = _means(train)
    fx = _features(train)
    fitted: dict[tuple[str, int], object] = {}
    means: dict[tuple[str, int], float] = {}
    for side in SIDES:
        for event in EVENTS:
            take = train.side_name.eq(side) & train.event.eq(event)
            if int(take.sum()) < 1000:
                raise ValueError(f"inadequate side/event support: {side}/{event}")
            fitted[side, event] = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)).fit(fx.loc[take], train.loc[take, "gross_bps"])
            means[side, event] = float(base_means[side][event])
    return fitted, means, base_means


def _scores(x: pd.DataFrame, models: dict[tuple[str, int], object], means: dict[tuple[str, int], float], side_means: dict[str, np.ndarray], shrink: float) -> pd.DataFrame:
    """Return B2 and shrunk conditional gross-to-net scores.

    ``shrink=0`` is exactly B2; values above zero blend in conditional ridge
    predictions.  This explicit interpolation is the strong-shrinkage guard.
    """
    fx = _features(x)
    conditional = np.empty((len(x), 3), dtype=float)
    for side in SIDES:
        take = x.side_name.eq(side).to_numpy()
        for event in EVENTS:
            raw = models[side, event].predict(fx.loc[take])
            conditional[take, event] = (1.0 - shrink) * means[side, event] + shrink * raw
    p = x.loc[:, PROBS].to_numpy(float)
    b2_pay = np.vstack([side_means[s] for s in x.side_name])
    out = x[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps"]].copy()
    out["b2_score_bps"] = np.einsum("ij,ij->i", p, b2_pay) - 100.0
    out["conditional_score_bps"] = np.einsum("ij,ij->i", p, conditional) - 100.0
    out["conditional_gross_upper_bps"] = conditional[:, 0]
    out["conditional_gross_lower_bps"] = conditional[:, 1]
    out["conditional_gross_timeout_bps"] = conditional[:, 2]
    return out


def _economics(x: pd.DataFrame, score: str) -> list[dict[str, float | int]]:
    ranked = x.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort")
    rows = []
    for q in TOPS:
        z = ranked.head(int(np.ceil(len(ranked) * q)))
        rows.append({"top_fraction": q, "n": len(z), "gross_bps": float(z.gross_bps.mean()), "net_bps": float(z.net_bps.mean()), "long_n": int(z.side_name.eq("long").sum()), "short_n": int(z.side_name.eq("short").sum())})
    return rows


def _selection_metric(x: pd.DataFrame) -> dict[str, float]:
    m = _economics(x, "conditional_score_bps")
    ic = float(spearmanr(x.conditional_score_bps, x.net_bps).statistic)
    # Predeclared: broad top-10 is primary, top-5 and rank association guard
    # against selecting a one-bucket spike.
    score = .60 * m[2]["net_bps"] + .25 * m[1]["net_bps"] + .15 * 100.0 * ic
    return {"development_selection_score": float(score), "top5_net_bps": float(m[1]["net_bps"]), "top10_net_bps": float(m[2]["net_bps"]), "net_spearman_ic": ic}


def main() -> None:
    a = args(); dev_train_end = pd.Timestamp(a.dev_train_end, tz="UTC"); dev_end = pd.Timestamp(a.dev_end, tz="UTC"); oos_end = pd.Timestamp(a.oos_end, tz="UTC")
    if not dev_train_end < dev_end < oos_end:
        raise ValueError("require dev-train-end < dev-end < oos-end")
    data = _panel(a.panel).merge(_base(a.base), on=["candidate_id", "__ts__", "side_name"], validate="one_to_one")
    # Conservative full-horizon proof is already held by label_available_at;
    # fit excludes every label unresolved by its declared boundary.
    dev_fit = data[data.__label_available_at__.lt(dev_train_end)].copy()
    dev = data[(data.__ts__.ge(dev_train_end)) & (data.__ts__.lt(dev_end))].copy()
    if dev.__label_available_at__.lt(dev_train_end).any():
        raise AssertionError("development candidate label overlaps its dev fit window")
    candidates = []
    for ridge_alpha in (100.0, 1000.0):
        models, means, side_means = _models(dev_fit, ridge_alpha)
        # At least half of every per-row estimate remains the observed
        # side/event mean.  This is intentionally a conditional-payoff
        # *calibration* ablation, not a second unrestricted value model.
        for shrink in (.10, .25, .50):
            scored = _scores(dev, models, means, side_means, shrink)
            candidates.append({"ridge_alpha": ridge_alpha, "shrink": shrink, **_selection_metric(scored), "b2_top10_net_bps": _economics(scored, "b2_score_bps")[2]["net_bps"]})
    grid = pd.DataFrame(candidates).sort_values(["development_selection_score", "ridge_alpha", "shrink"], ascending=[False, True, True], kind="mergesort")
    winner = grid.iloc[0].to_dict()
    # Preserve the exact development predictions produced by the selected
    # conditional-payoff arm.  They are needed to select a *downstream*
    # frozen residual/reliability composition without ever inspecting OOS
    # outcomes.  These models see only labels resolved before dev_train_end.
    dev_models, dev_means, dev_side_means = _models(dev_fit, float(winner["ridge_alpha"]))
    dev_scored = _scores(dev, dev_models, dev_means, dev_side_means, float(winner["shrink"]))
    # Frozen OOS refit: all labels resolved before Aug 1, no label at/after
    # the boundary can enter the conditional estimates.
    oos_start = dev_end
    final_fit = data[data.__label_available_at__.lt(oos_start)].copy()
    oos = data[(data.__ts__.ge(oos_start)) & (data.__ts__.lt(oos_end))].copy()
    if oos.__label_available_at__.lt(oos_start).any():
        raise AssertionError("OOS candidate label overlaps final fit window")
    models, means, side_means = _models(final_fit, float(winner["ridge_alpha"]))
    scored = _scores(oos, models, means, side_means, float(winner["shrink"]))
    if not np.isfinite(scored[["b2_score_bps", "conditional_score_bps"]].to_numpy(float)).all():
        raise AssertionError("conditional payoff scorer emitted a non-finite value")
    metrics = []
    for name, col in (("B2_side_mean_control", "b2_score_bps"), ("B3_conditional_payoff", "conditional_score_bps")):
        for row in _economics(scored, col): metrics.append({"variant": name, **row})
    a.out.mkdir(parents=True, exist_ok=True)
    dev_scored.to_parquet(a.out / "development_selected_predictions.parquet", index=False)
    scored.to_parquet(a.out / "oos_predictions.parquet", index=False)
    grid.to_parquet(a.out / "development_grid.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(a.out / "oos_global_economics.parquet", index=False)
    coverage = {name: {"fit_finite_fraction": float(np.isfinite(final_fit[name].to_numpy(float)).mean()), "oos_finite_fraction": float(np.isfinite(oos[name].to_numpy(float)).mean())} for name in RAW_CONTEXT}
    manifest = {"schema": "full_universe_stage5_2_conditional_payoff_v1", "status": "COMPLETED_DIAGNOSTIC_NO_PROMOTION", "base": str(a.base), "panel": str(a.panel), "contract": {"base": "frozen B2 TP3/SL2 event probabilities", "formula": "sum p(event)*E[gross|event,side,limited causal context]-100bps", "features": list(RAW_CONTEXT) + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "base_entropy"], "constants_not_row_features": {"geometry": "TP=3 ATR, SL=2 ATR, H12", "cost_bps": 100.0}, "model": "six side/event ridge regressions with explicit shrinkage to side/event conditional mean", "selection": "single chronological development grid; 0.60 top10 net + 0.25 top5 net + 0.15*100*net Spearman IC", "selection_grid": {"ridge_alpha": [100.0, 1000.0], "conditional_shrink": [.10, .25, .50], "maximum_conditional_weight": .50}, "selection_winner": winner}, "causality": {"development_fit_labels_resolved_before": str(dev_train_end), "development_eval": [str(dev_train_end), str(dev_end)], "oos_fit_labels_resolved_before": str(oos_start), "oos_eval": [str(oos_start), str(oos_end)], "assertion": "No OOS outcome label entered either final B2 means or conditional payoff models."}, "rows": {"development_fit": len(dev_fit), "development_eval": len(dev), "oos_fit": len(final_fit), "oos_eval": len(oos)}, "feature_coverage": coverage, "oos_metrics": metrics}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"winner": winner, "oos_top10": [x for x in metrics if x["top_fraction"] == .1]}, indent=2))


if __name__ == "__main__":
    main()
