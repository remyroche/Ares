#!/usr/bin/env python3
"""Sequential C0--C3 conditioning ablation for the frozen shared residual B0.

The experiment holds the shared Huber LightGBM class fixed and predicts the
candidate-specific residual after removing both the causal base expected-net
map and the prior-resolved soft-regime residual baseline,
square-root *era* weights, exact TP6/SL4/H12 net outcome and the fixed
100-bps cost.  It is deliberately a four-step funnel, not a feature sweep:

* C0 uses the frozen invariant core and no regime field;
* C1 adds only the sealed causal soft-state surface;
* C2 adds a small, predeclared set of base-value x soft-state interactions;
* C3 adds the sealed prequential regime-relative residual/z values.

All models train on earlier complete eras and are scored once on the next
held-out era.  There are no regime-local models, routing rules, target changes
or timestamp-local rankings.  Results are selected lexicographically by
worst-era global top-1% net bps, then mean global top-1% net bps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_shared_residual_c0_c3_20260809_v1"
ERAS = ("2023-07_08", "2023-09_10", "2023-11_12", "2024-01_02", "2024-05_06", "2024-07_08", "2024-09_10", "2024-11")
BASE = ("p_adverse", "p_weak", "p_clear")
REGIME_PRIOR = "soft_regime_prior_residual_bps"
REGIME_CENTERED_TARGET = "target__soft_regime_centered_residual_bps"
CONTEXT = (
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h",
    "mkt_oi_chg_z_24h", "mkt_funding_dispersion", "cross_asset_corr_4h",
    "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score",
    "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market",
    "deleveraging_without_followthrough", "short_signal_recovery_conflict",
)
INVARIANT = ("side_is_long", *BASE, "prequential_base_expected_net_bps", *CONTEXT)
SOFT_STATE = (
    "regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition",
    "regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours",
)
# These are the only explicit interactions.  C2 must test the stated
# conversion hypothesis, not enumerate feature x regime combinations.
INTERACTION_SOURCE = ("p_adverse", "p_weak", "p_clear", "prequential_base_expected_net_bps")
INTERACTION_STATE = ("regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition")
RELATIVE_SOURCE = (*BASE, *CONTEXT)
RELATIVE = tuple(
    f"regime_{kind}__{field}"
    for kind in ("relative", "z")
    for field in RELATIVE_SOURCE
)
ARMS = {
    "C0_invariant_core": "invariant core only; no soft state or relative normalisation",
    "C1_soft_regime_state": "C0 plus soft regime probabilities, entropy and causal transition state",
    "C2_restricted_base_x_regime": "C1 plus predeclared base-value x soft-regime interactions only",
    "C3_prequential_regime_relative": "C2 plus prequential regime-relative residual and z features",
}
TOPS = (.01, .05, .10)
SEED = 20260809
LABEL_AVAILABILITY_DELAY = pd.Timedelta(hours=13)  # signal close +1h entry + H12
PARAMS: dict[str, Any] = dict(
    n_estimators=180, learning_rate=.035, num_leaves=24, min_child_samples=400,
    colsample_bytree=.80, subsample=.80, reg_lambda=12.,
)


def ensure_label_available_ts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "label_available_ts" in out:
        available = pd.to_datetime(out["label_available_ts"], utc=True, errors="raise")
    elif "__label_available_at__" in out:
        available = pd.to_datetime(out["__label_available_at__"], utc=True, errors="raise")
    else:
        available = pd.to_datetime(out["__ts__"], utc=True) + LABEL_AVAILABILITY_DELAY
    if available.isna().any() or (available < pd.to_datetime(out["__ts__"], utc=True) + LABEL_AVAILABILITY_DELAY).any():
        raise ValueError("H12 label availability must be signal-close +13h or later")
    out["label_available_ts"] = available
    return out


def assert_outer_train_resolved(train: pd.DataFrame, test: pd.DataFrame) -> None:
    cutoff = pd.to_datetime(test["__ts__"], utc=True).min()
    latest = pd.to_datetime(train["label_available_ts"], utc=True).max()
    if latest >= cutoff:
        raise ValueError(f"outer training contains unresolved labels: {latest} >= {cutoff}")


def fit_model(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="huber", alpha=.9, random_state=SEED, n_jobs=1, verbosity=-1, **PARAMS).fit(
        x, y, sample_weight=weight,
    )


def weights(train: pd.DataFrame) -> np.ndarray:
    """The fixed B0 winner recipe: mild, capped square-root era balance."""
    count = train.groupby("era").size()
    value = train.era.map(np.sqrt(len(train) / (len(count) * count))).to_numpy(float)
    return np.clip(value / value.mean(), .25, 4.)


def add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for left in INTERACTION_SOURCE:
        for right in INTERACTION_STATE:
            out[f"interaction__{left}__x__{right}"] = out[left].to_numpy(float) * out[right].to_numpy(float)
    return out


def columns_for(arm: str) -> tuple[str, ...]:
    if arm == "C0_invariant_core":
        return INVARIANT
    if arm == "C1_soft_regime_state":
        return (*INVARIANT, *SOFT_STATE)
    if arm == "C2_restricted_base_x_regime":
        interactions = tuple(f"interaction__{left}__x__{right}" for left in INTERACTION_SOURCE for right in INTERACTION_STATE)
        return (*INVARIANT, *SOFT_STATE, *interactions)
    if arm == "C3_prequential_regime_relative":
        interactions = tuple(f"interaction__{left}__x__{right}" for left in INTERACTION_SOURCE for right in INTERACTION_STATE)
        return (*INVARIANT, *SOFT_STATE, *interactions, *RELATIVE)
    raise ValueError(arm)


def matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    value = frame.loc[:, columns].to_numpy(dtype=np.float32)
    if not np.isfinite(value).all():
        bad = [c for c in columns if not np.isfinite(frame[c].to_numpy(float)).all()]
        raise ValueError(f"non-finite feature value: {bad}")
    return value


def metric_rows(frame: pd.DataFrame, score: np.ndarray, common: dict[str, Any], period: str, scope: str) -> list[dict[str, Any]]:
    z = frame.copy()
    z["score_bps"] = score
    rows: list[dict[str, Any]] = []
    for view, q in (("global", z), ("long", z[z.side_name.eq("long")]), ("short", z[z.side_name.eq("short")])):
        ic = spearmanr(q.score_bps, q.net_bps).statistic if len(q) > 1 else np.nan
        for top in TOPS:
            take = q.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort").head(max(1, int(np.ceil(len(q) * top))))
            rows.append({
                **common, "scope": scope, "period": period, "view": view, "top_fraction": top,
                "n": len(take), "net_bps": float(take.net_bps.mean()), "gross_bps": float(take.gross_bps.mean()),
                "all_rows_net_bps": float(q.net_bps.mean()), "score_net_spearman": float(ic),
                "selected_long_fraction": float(take.side_name.eq("long").mean()),
                "positive_net_fraction": float(take.net_bps.gt(0).mean()), "clear_50_fraction": float(take.net_bps.gt(50).mean()),
            })
    return rows


def robust(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean_top1_net_bps": float(a.mean()), "worst_top1_net_bps": float(a.min()), "std_top1_net_bps": float(a.std(ddof=0)), "positive_eras": int((a > 0).sum())}


def validate(data: pd.DataFrame) -> None:
    required = {"candidate_id", "__ts__", "label_available_ts", "side_name", "era", "net_bps", "gross_bps", "shared_regime_contract_complete", REGIME_PRIOR, REGIME_CENTERED_TARGET, *INVARIANT, *SOFT_STATE, *RELATIVE}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"shared regime v3 ledger misses columns: {missing}")
    if data.candidate_id.duplicated().any():
        raise ValueError("candidate IDs must be unique")
    if not np.allclose(data.gross_bps.to_numpy(float) - data.net_bps.to_numpy(float), 100., atol=.02):
        raise ValueError("fixed 100-bps cost contract failed")
    if not set(data.era.unique()).issubset(ERAS):
        raise ValueError("unexpected era")
    p = data[["regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition"]]
    if not np.allclose(p.sum(axis=1), 1., atol=1e-5):
        raise ValueError("soft regime simplex violates sum-to-one")
    # The materialised state/relative transforms carry their explicit weekly
    # prior reference cutoffs.  They cannot be fitted after the candidate.
    for cutoff in ("state_reference_cutoff_utc", "residual_reference_cutoff_utc"):
        ts = pd.to_datetime(data[cutoff], utc=True)
        if not ts.le(pd.to_datetime(data.__ts__, utc=True)).all():
            raise ValueError(f"{cutoff} is after a decision row")


def load() -> pd.DataFrame:
    x = pd.read_parquet(INPUT)
    x = x[x.shared_regime_contract_complete.astype(bool)].copy()
    x["side_is_long"] = x.side_name.eq("long").astype(float)
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    x = ensure_label_available_ts(x)
    # The cohort is fixed before arm comparisons.  Do not impute any C-only
    # feature; all four arms operate on the complete C3 cohort.
    needed = (*INVARIANT, *SOFT_STATE, *RELATIVE, REGIME_PRIOR, REGIME_CENTERED_TARGET, "state_reference_cutoff_utc", "residual_reference_cutoff_utc")
    x = x[np.isfinite(x.loc[:, [c for c in needed if c not in ("state_reference_cutoff_utc", "residual_reference_cutoff_utc")]].to_numpy(float)).all(axis=1)].copy()
    validate(x)
    return add_interactions(x.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only-arm", choices=tuple(ARMS))
    parser.add_argument("--only-era", choices=ERAS[1:])
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    stage = args.out.with_name(args.out.name + "_stage")
    if args.finalize:
        checkpoints = sorted((stage / "checkpoints").glob("*.parquet"))
        if not checkpoints:
            raise FileNotFoundError("no checkpoints")
        metrics = pd.concat([pd.read_parquet(p) for p in checkpoints], ignore_index=True).drop_duplicates(
            ["arm", "test_era", "scope", "period", "view", "top_fraction"], keep="last",
        )
        expected = {(arm, era) for arm in ARMS for era in ERAS[1:]}
        got = set(metrics.loc[metrics.scope.eq("era"), ["arm", "test_era"]].itertuples(index=False, name=None))
        if missing := sorted(expected - got):
            raise ValueError(f"missing outer cells: {missing}")
        outer = metrics[(metrics.scope.eq("era")) & (metrics.view.eq("global")) & (metrics.top_fraction.eq(.01))]
        summary = pd.DataFrame([{"arm": arm, **robust(q.net_bps.tolist())} for arm, q in outer.groupby("arm", sort=True)])
        # Explicit lexicographic selection—not a weighted score that could
        # exchange a catastrophic era for a higher mean.
        summary = summary.sort_values(["worst_top1_net_bps", "mean_top1_net_bps", "std_top1_net_bps"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
        pooled_rows: list[dict[str, Any]] = []
        for arm in ARMS:
            paths = sorted((stage / "predictions").glob(f"*_{arm}.parquet"))
            pred = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
            pooled_rows.extend(metric_rows(pred, pred.score_bps.to_numpy(float), {"arm": arm, "test_era": "ALL_OUTER", "train_through": "rolling", "model_name": "B0_reference", "weight_profile": "sqrt_era"}, "ALL_OUTER", "pooled_global"))
        args.out.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(args.out / "metrics.parquet", index=False)
        pd.DataFrame(pooled_rows).to_parquet(args.out / "pooled_global_metrics.parquet", index=False)
        summary.to_parquet(args.out / "summary.parquet", index=False)
        report = ["# C0–C3 shared regime conditioning funnel", "", "Fixed B0 shared Huber expert, fixed square-root era weights, strict chronological held-out eras and pooled-global rankings. No local or per-regime experts.", "", "## Lexicographic selection: worst era, then mean global top-1% net bps", "", "| rank | arm | worst net bps | mean net bps | std | positive eras |", "|---:|---|---:|---:|---:|---:|"]
        for i, row in enumerate(summary.itertuples(index=False), 1):
            report.append(f"| {i} | {row.arm} | {row.worst_top1_net_bps:.3f} | {row.mean_top1_net_bps:.3f} | {row.std_top1_net_bps:.3f} | {row.positive_eras} |")
        report += ["", "C0: invariant core only. C1: sealed soft state. C2: C1 plus 16 restricted base-value × state interactions. C3: C2 plus prequential regime-relative residual/z fields. All C arms use the same complete cohort; target columns and local/expert routing are excluded."]
        (args.out / "REPORT.md").write_text("\n".join(report) + "\n")
        manifest = {"schema": "tp6_shared_residual_c0_c3_v1", "status": "COMPLETED_DIAGNOSTIC_NO_PROMOTION", "input": str(INPUT), "contract": {"geometry": "TP6/SL4/H12", "cost_bps": 100., "target": REGIME_CENTERED_TARGET, "reconstruction": "prequential_base_expected_net_bps + soft_regime_prior_residual_bps + predicted_candidate_residual_bps", "expert": "one shared pooled Huber LGBM", "weights": "fixed square-root era weights", "ranking": "global top-k per held-out era and pooled outer ledger", "no_local_or_regime_experts": True, "no_feature_sweep": True}, "arms": ARMS, "features": {arm: list(columns_for(arm)) for arm in ARMS}, "selection": "lexicographic: worst global top-1% net bps, then mean, then lower dispersion", "winner": summary.iloc[0].to_dict()}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(json.dumps(summary.to_dict(orient="records"), indent=2))
        return
    if not args.only_era:
        raise ValueError("use --only-era for a resumable strict outer checkpoint, then --finalize")
    x = load()
    index = ERAS.index(args.only_era)
    train = x[x.era.isin(ERAS[:index])].copy()
    test = x[x.era.eq(args.only_era)].copy()
    if train.empty or test.empty:
        raise ValueError("empty strict rolling split")
    assert_outer_train_resolved(train, test)
    use_arms = (args.only_arm,) if args.only_arm else tuple(ARMS)
    rows: list[dict[str, Any]] = []
    for arm in use_arms:
        features = columns_for(arm)
        target = train[REGIME_CENTERED_TARGET].to_numpy(float)
        model = fit_model(matrix(train, features), target, weights(train))
        candidate_correction = model.predict(matrix(test, features))
        score = (test.prequential_base_expected_net_bps.to_numpy(float)
                 + test[REGIME_PRIOR].to_numpy(float) + candidate_correction)
        common = {"arm": arm, "test_era": args.only_era, "train_through": ERAS[index - 1], "train_rows": len(train), "test_rows": len(test), "model_name": "B0_reference", "weight_profile": "sqrt_era", "feature_count": len(features)}
        rows.extend(metric_rows(test, score, common, args.only_era, "era"))
        for month, q in test.assign(__month__=test.__ts__.dt.strftime("%Y-%m")).groupby("__month__", sort=True):
            local = q.index.to_numpy() - test.index.min()
            rows.extend(metric_rows(q, score[local], common, month, "month"))
        prediction = test[["candidate_id", "__ts__", "label_available_ts", "side_name", "net_bps", "gross_bps", "era", "prequential_base_expected_net_bps", REGIME_PRIOR]].copy()
        prediction["predicted_candidate_residual_bps"] = candidate_correction
        prediction["score_bps"] = score
        prediction["arm"] = arm
        prediction["feature_count"] = len(features)
        out = stage / "predictions"; out.mkdir(parents=True, exist_ok=True)
        prediction.to_parquet(out / f"{args.only_era}_{arm}.parquet", index=False)
    out = stage / "checkpoints"; out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.only_arm}" if args.only_arm else ""
    pd.DataFrame(rows).to_parquet(out / f"{args.only_era}{suffix}.parquet", index=False)
    print(json.dumps({"test_era": args.only_era, "arms": list(use_arms), "train_rows": len(train), "test_rows": len(test)}, indent=2))


if __name__ == "__main__":
    main()
