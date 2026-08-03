#!/usr/bin/env python3
"""Stage 5.x: small, comparable-context pairwise gross-ranking ablation.

The Stage 5.2 conditional-payoff value remains the frozen opportunity score.
This experiment adds one deliberately constrained rank component, trained to
order realised *gross* payoff only among economically comparable candidates:
same side, decision week, train-fixed volatility bucket, predicted barrier
event, and train-fixed base-confidence bucket.  It is not an overlay and does
not reuse the residual/reliability heads.

One chronological development selection chooses a conservative blend strength.
The selected recipe is then refit solely on labels resolved before 2024-08-01
and replayed exactly once on Aug--Nov.  All scores are ranked globally; groups
exist only for pair construction, never for portfolio selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
SIDES = ("long", "short")
PROBS = ("p_upper", "p_lower", "p_timeout")
CONTEXT = (
    "atr_1h", "rv_24h", "mkt_rv_24h", "prior_volatility", "amihud_z",
    "ob_spread_bps_z_24h", "liquidity_ratio_peer_resid",
    "xasset_mkt_spread_bps", "assumed_round_trip_cost_bps",
)
TOPS = (.01, .05, .10, .20)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    p.add_argument("--base", type=Path, default=ROOT / "data_perp/artifacts/full_universe_base_hpo_20260802_v1")
    p.add_argument("--conditional", type=Path, default=ROOT / "data_perp/artifacts/full_universe_stage5_2_conditional_payoff_20260804_v1")
    p.add_argument("--out", type=Path, default=ROOT / "data_perp/artifacts/full_universe_stage5_pairwise_gross_20260804_v1")
    return p.parse_args()


def _load_base(root: Path) -> pd.DataFrame:
    cols = ["candidate_id", "__ts__", "side_name", *PROBS]
    x = pd.concat([pd.read_parquet(root / s / "target_screen_predictions.parquet", columns=cols) for s in SIDES], ignore_index=True)
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    if x.candidate_id.duplicated().any() or not np.allclose(x.loc[:, PROBS].to_numpy(float).sum(1), 1., atol=1e-5):
        raise ValueError("frozen base prediction contract is invalid")
    return x


def _load_panel(path: Path) -> pd.DataFrame:
    cols = ["candidate_id", "__ts__", "__label_available_at__", "side_name", "t4_tp3_sl2_gross_bps", *CONTEXT]
    frames = [pd.read_parquet(p, columns=cols) for p in sorted((path / "parts").glob("*.parquet"))]
    x = pd.concat(frames, ignore_index=True).rename(columns={"t4_tp3_sl2_gross_bps": "gross_bps"})
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    x["__label_available_at__"] = pd.to_datetime(x["__label_available_at__"], utc=True)
    return x


def _matrix(x: pd.DataFrame) -> np.ndarray:
    out = x.loc[:, CONTEXT].copy().replace([np.inf, -np.inf], np.nan)
    # LightGBM accepts NaNs.  All features below are direct decision-time
    # fields or algebraic transforms of the frozen simplex and timestamp.
    p = x.loc[:, PROBS].to_numpy(float)
    safe = np.clip(p, 1e-12, 1.)
    out["base_entropy"] = -(safe * np.log(safe)).sum(1) / np.log(3.)
    out["upper_lower_margin"] = p[:, 0] - p[:, 1]
    out["top_two_margin"] = np.partition(p, -2, axis=1)[:, -1] - np.partition(p, -2, axis=1)[:, -2]
    h = x.__ts__.dt.hour.to_numpy(float)
    d = x.__ts__.dt.dayofweek.to_numpy(float)
    out["hour_sin"] = np.sin(2 * np.pi * h / 24.)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24.)
    out["dow_sin"] = np.sin(2 * np.pi * d / 7.)
    out["dow_cos"] = np.cos(2 * np.pi * d / 7.)
    out["side_is_long"] = x.side_name.eq("long").to_numpy(int)
    return out.to_numpy(float)


def _group_contract(fit: pd.DataFrame) -> dict[str, np.ndarray]:
    # The base-confidence coordinate is label-free.  Boundaries are learned
    # from training candidates once and reused for calibration/evaluation.
    return {
        "vol": np.unique(np.quantile(fit.mkt_rv_24h.to_numpy(float), np.linspace(0., 1., 4))),
        "confidence": np.unique(np.quantile((fit.p_upper - fit.p_lower).to_numpy(float), np.linspace(0., 1., 5))),
    }


def _with_groups(x: pd.DataFrame, contract: dict[str, np.ndarray]) -> pd.DataFrame:
    out = x.copy()
    # ISO week satisfies the nearby-date restriction; no comparison crosses a
    # week boundary.  Fixed TP3/SL2 geometry is common to all rows.  Predicted
    # event supplies the remaining event-geometry comparability coordinate.
    week = out.__ts__.dt.to_period("W-MON").astype(str)
    vol = np.digitize(out.mkt_rv_24h.to_numpy(float), contract["vol"][1:-1], right=True)
    confidence = np.digitize((out.p_upper - out.p_lower).to_numpy(float), contract["confidence"][1:-1], right=True)
    event = np.argmax(out.loc[:, PROBS].to_numpy(float), axis=1)
    out["_group"] = out.side_name.astype(str) + "|" + week + "|v" + pd.Series(vol, index=out.index).astype(str) + "|e" + pd.Series(event, index=out.index).astype(str) + "|c" + pd.Series(confidence, index=out.index).astype(str)
    return out


def _rank_labels(x: pd.DataFrame) -> np.ndarray:
    pct = x.groupby("_group", observed=True).gross_bps.rank(method="average", pct=True).to_numpy(float)
    return np.minimum(np.floor(pct * 31.), 30).astype(int)


def _ranker() -> lgb.LGBMRanker:
    # Fixed, shallow, and heavily regularised: this is a local correction,
    # not a second unrestricted base model or an HPO sweep.
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", eval_at=(10,), n_estimators=160,
        learning_rate=.04, num_leaves=16, min_child_samples=500,
        colsample_bytree=.80, subsample=.80, reg_lambda=20.,
        lambdarank_truncation_level=40, random_state=20260804, n_jobs=1,
        verbosity=-1,
    )


def _fit_predict(train: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Fit ranker, then calibrate its output in bps on a later training slice.

    The ranker never sees the calibration-slice labels.  The calibrator is an
    affine Ridge map from rank score to gross bps.  It makes the small blend
    interpretable in payoff units; its fitted sign is recorded because a
    negative calibration slope transparently reverses the unstable raw order.
    """
    dates = np.sort(train.__ts__.dt.floor("D").unique())
    cut = dates[max(1, int(np.floor(len(dates) * .80)))]
    rank_fit = train[train.__ts__.lt(cut)].copy()
    calibration = train[train.__ts__.ge(cut)].copy()
    contract = _group_contract(rank_fit)
    rank_fit = _with_groups(rank_fit, contract).sort_values(["_group", "candidate_id"], kind="mergesort")
    sizes = rank_fit.groupby("_group", observed=True, sort=False).size()
    keep = sizes[sizes.ge(2)].index
    rank_fit = rank_fit[rank_fit._group.isin(keep)].copy()
    sizes = rank_fit.groupby("_group", observed=True, sort=False).size().to_numpy(int)
    if len(rank_fit) < 10_000 or len(calibration) < 10_000:
        raise ValueError("insufficient chronologically separated rank/calibration support")
    model = _ranker().fit(_matrix(rank_fit), _rank_labels(rank_fit), group=sizes)
    cal_raw = model.predict(_matrix(calibration))
    calibration_model = Ridge(alpha=100.).fit(cal_raw.reshape(-1, 1), calibration.gross_bps.to_numpy(float))
    slope = float(calibration_model.coef_[0])
    raw = model.predict(_matrix(evaluation))
    calibrated = calibration_model.predict(raw.reshape(-1, 1))
    diag = {
        "rank_fit_rows": int(len(rank_fit)), "calibration_rows": int(len(calibration)),
        "rank_groups": int(len(sizes)), "group_min": int(sizes.min()), "group_median": float(np.median(sizes)), "group_max": int(sizes.max()),
        "rank_fit_end_exclusive": str(cut), "volatility_edges": contract["vol"].tolist(), "confidence_edges": contract["confidence"].tolist(),
        "calibration": "Ridge(gross_bps ~ pairwise_score), fitted on later disjoint training slice; a negative slope is retained as an explicit, pre-evaluation orientation correction rather than hidden",
        "calibration_slope": slope, "calibration_intercept": float(calibration_model.intercept_),
        "features": list(CONTEXT) + ["base_entropy", "upper_lower_margin", "top_two_margin", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "side_is_long"],
    }
    return calibrated, diag


def _metrics(x: pd.DataFrame, score: str) -> list[dict]:
    ranked = x.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort")
    rows = []
    for q in TOPS:
        z = ranked.head(int(np.ceil(len(ranked) * q)))
        rows.append({"top_fraction": q, "n": int(len(z)), "gross_bps": float(z.gross_bps.mean()), "net_bps": float(z.gross_bps.mean() - 100.), "long_n": int(z.side_name.eq("long").sum()), "short_n": int(z.side_name.eq("short").sum())})
    return rows


def _objective(x: pd.DataFrame, score: str) -> dict:
    m = _metrics(x, score)
    ic = float(spearmanr(x[score], x.gross_bps).statistic)
    # Gross selection matches the pairwise target.  Fixed 100 bps cost means
    # the ordering is identical to net, reported separately for economics.
    return {"selection_score": .60 * m[2]["gross_bps"] + .25 * m[1]["gross_bps"] + .15 * 100. * ic, "gross_spearman_ic": ic, "metrics": m}


def _run_one(raw: pd.DataFrame, conditional: pd.DataFrame, fit_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    evaluation = conditional.merge(raw.drop(columns=["gross_bps"]), on=["candidate_id", "__ts__", "side_name"], validate="one_to_one")
    if evaluation.__label_available_at__.lt(fit_end).any():
        raise AssertionError("evaluation label became available in its ranker fit")
    train = raw[raw.__label_available_at__.lt(fit_end) & raw.__ts__.ge(pd.Timestamp("2024-04-15", tz="UTC"))].copy()
    pair_gross, diag = _fit_predict(train, evaluation)
    out = evaluation.copy()
    out["pairwise_calibrated_gross_bps"] = pair_gross
    # The component is deliberately a small raw-value correction toward its
    # calibrated gross estimate.  Conditional score is net, hence convert it
    # to gross before interpolation and back afterwards.
    for strength in (0., .10, .25, .40):
        out[f"score_blend_{strength:g}"] = (1. - strength) * out.conditional_score_bps + strength * (out.pairwise_calibrated_gross_bps - 100.)
    return out, diag


def main() -> None:
    a = _args()
    raw = _load_panel(a.panel).merge(_load_base(a.base), on=["candidate_id", "__ts__", "side_name"], validate="one_to_one")
    dev = pd.read_parquet(a.conditional / "development_selected_predictions.parquet")
    oos = pd.read_parquet(a.conditional / "oos_predictions.parquet")
    dev_end = pd.Timestamp("2024-06-15", tz="UTC")
    oos_end = pd.Timestamp("2024-08-01", tz="UTC")
    dev_scored, dev_diag = _run_one(raw, dev, dev_end)
    candidates = []
    for strength in (0., .10, .25, .40):
        name = f"score_blend_{strength:g}"
        candidates.append({"strength": strength, **_objective(dev_scored, name)})
    # One development-only selection; zero is preferred on an exact tie.
    winner = sorted(candidates, key=lambda r: (-r["selection_score"], r["strength"]))[0]
    selected = f"score_blend_{winner['strength']:g}"
    oos_scored, oos_diag = _run_one(raw, oos, oos_end)
    # These are a single frozen-model replay of the predeclared development
    # candidates.  They are retained for transparency, never used to alter
    # ``winner`` or the model/feature/group contract.
    oos_candidates = [
        {"strength": strength, **_objective(oos_scored, f"score_blend_{strength:g}")}
        for strength in (0., .10, .25, .40)
    ]
    a.out.mkdir(parents=True, exist_ok=True)
    dev_scored.to_parquet(a.out / "development_predictions.parquet", index=False)
    oos_scored.to_parquet(a.out / "oos_predictions.parquet", index=False)
    manifest = {
        "schema": "full_universe_stage5_pairwise_gross_v1", "status": "COMPLETED_UNTOUCHED_OOS_REPLAY",
        "contract": {"base_value": "frozen Stage5.2 side/event conditional-payoff expected net", "rank_component": "shallow fixed LightGBM LambdaRank trained on comparable-context gross ordering", "groups": "same side + same ISO week + rank-fit volatility tertile + base argmax event + rank-fit base-confidence quartile", "geometry": "fixed TP3/SL2 H12; therefore no varying geometry is a row feature", "target": "within-group ordinal realised gross bps", "calibration": "chronologically later, disjoint Ridge gross calibration", "selection": "one development-only gross objective; candidate strengths 0, 0.10, 0.25, 0.40", "final_ranking": "all candidates pooled globally, no side or timestamp quota", "overlay": "none: residual/reliability integrated overlay intentionally not joined"},
        "causality": {"development_fit_labels_resolved_before": str(dev_end), "development_eval": ["2024-06-15", "2024-08-01"], "oos_fit_labels_resolved_before": str(oos_end), "oos_eval": ["2024-08-01", "2024-12-01"], "assertion": "Each evaluation row has a label availability time at or after its model fit boundary."},
        "development_diagnostics": dev_diag, "oos_diagnostics": oos_diag, "development_candidates": candidates,
        "oos_frozen_replay_candidates_not_selectable": oos_candidates, "winner": winner,
        "oos_selected": {"strength": winner["strength"], **_objective(oos_scored, selected)},
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = ["# Stage 5.x — pairwise gross-ranking ablation", "", "The frozen Stage 5.2 conditional-payoff score is retained.  A single shallow, comparable-context gross ranker is blended separately; the residual/reliability overlay is intentionally not used.", "", "## Development-only selection", "", "| Component weight | Objective | Top-5 gross | Top-10 gross | Gross Spearman IC |", "|---:|---:|---:|---:|---:|"]
    for c in candidates:
        m = c["metrics"]
        lines.append(f"| {c['strength']:.2f} | {c['selection_score']:.2f} | {m[1]['gross_bps']:.2f} | {m[2]['gross_bps']:.2f} | {c['gross_spearman_ic']:.4f} |")
    m = manifest["oos_selected"]["metrics"]
    control = {r["top_fraction"]: r for r in oos_candidates[0]["metrics"]}
    lines += ["", f"Development-selected component weight: **{winner['strength']:.2f}**.", "", "## One untouched OOS replay", "", "| Tail | Gross bps | Net bps | N | Long | Short |", "|---|---:|---:|---:|---:|---:|"]
    for r in m:
        delta = r["net_bps"] - control[r["top_fraction"]]["net_bps"]
        lines.append(f"| Top {r['top_fraction']:.0%} | {r['gross_bps']:.2f} | {r['net_bps']:.2f} | {r['n']} | {r['long_n']} | {r['short_n']} |")
    lines += ["", "Against the same frozen conditional-payoff control, OOS net deltas at top 1/5/10/20 are: " + ", ".join(f"{r['net_bps'] - control[r['top_fraction']]['net_bps']:+.2f}" for r in m) + " bps.  This comparison is descriptive only; no OOS result changed the development-selected weight."]
    (a.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"winner": winner, "oos": manifest["oos_selected"], "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
