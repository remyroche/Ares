#!/usr/bin/env python3
"""Strict chronological cross-era transport audit for the TP6/SL4 M6 head.

The audit deliberately keeps the Jan--Aug 2022 inverse-PI population outside
the matrix: it has a different candidate/product and context schema.  It is
reported in the manifest as an external, non-combinable transport diagnostic,
not zero-imputed or silently pooled with the compatible 2023--24 population.

All rows in the compatible matrix are pre-existing same-side base OOF rows.
For a cell, M6 is fitted only on the selected training era (or its expanding
prefix) and evaluated only in a later era.  Context is the fixed 14-field
high-coverage causal pack plus the base output from that same side.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
OUT = ROOT / "data_perp/artifacts/tp6_m6_cross_era_transport_20260809_v3"

# The two sparse R5 transition fields are intentionally excluded.  The fields
# below had >=90% coverage in the compatible historical OOF population.
CONTEXT = [
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h", "mkt_oi_chg_z_24h",
    "mkt_funding_dispersion", "cross_asset_corr_4h", "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score", "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market", "deleveraging_without_followthrough",
    "short_signal_recovery_conflict",
]
BASE = ["p_adverse", "p_weak", "p_clear", "base_raw"]
FEATURES = BASE + CONTEXT
TOPS = (.01, .05, .10)
ERAS = (
    ("2023-07_08", "2023-07-01", "2023-09-01", "oof23_f0"),
    ("2023-09_10", "2023-09-01", "2023-11-01", "oof23_f1"),
    ("2023-11_12", "2023-11-01", "2024-01-01", "oof23_f2"),
    ("2024-01_02", "2024-01-01", "2024-03-01", "oof23_f3"),
    ("2024-05_06", "2024-05-01", "2024-07-01", "ledger24"),
    ("2024-07_08", "2024-07-01", "2024-09-01", "ledger24"),
    ("2024-09_10", "2024-09-01", "2024-11-01", "ledger24"),
    ("2024-11", "2024-11-01", "2024-12-01", "ledger24"),
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for part in iter(lambda: fh.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _matrix(x: pd.DataFrame, cols: list[str] = FEATURES) -> np.ndarray:
    return x[cols].replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy(np.float32)


def _model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=.04, num_leaves=24,
        min_child_samples=400, colsample_bytree=.8, subsample=.8, reg_lambda=12.,
        random_state=20260809, n_jobs=1, verbosity=-1,
    )


def _read_oof23() -> pd.DataFrame:
    pieces = []
    for side in ("long", "short"):
        for fold in range(4):
            path = ROOT / f"data_perp/artifacts/tp6_r3_r5_{side}_baseoof_fold{fold}_20260802_v1/base_oof_predictions.parquet"
            x = pd.read_parquet(path)
            if set(x.side_name.unique()) != {side}:
                raise ValueError(f"not pure same-side OOF: {path}")
            x = x.rename(columns={"prob_adverse": "p_adverse", "prob_weak": "p_weak", "prob_clear": "p_clear"})
            x["source"] = f"oof23_f{fold}"
            pieces.append(x[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", *BASE, "source"]])
    return pd.concat(pieces, ignore_index=True)


def _read_ledger24() -> pd.DataFrame:
    path = ROOT / "data_perp/artifacts/tp6_sl4_b10_bw4_base_oof_20260802_v1/base_oof_ledger.parquet"
    x = pd.read_parquet(path).rename(columns={
        "t4_tp6_sl4_net_bps": "net_bps", "base_expected_net_bps": "base_raw",
        "base_p_lower": "p_adverse", "base_p_timeout": "p_weak", "base_p_upper": "p_clear",
    })
    x["gross_bps"] = x.net_bps + 100.
    x["source"] = "ledger24"
    # Base fit must precede its candidate, preserving the documented OOF lineage.
    if not (pd.to_datetime(x.base_fit_resolved_before, utc=True) <= pd.to_datetime(x.__ts__, utc=True)).all():
        raise ValueError("2024 ledger contains non-chronological base score")
    return x[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", *BASE, "source"]]


def _load_era_raw(start: str, end: str, source: str) -> pd.DataFrame:
    """Read only one era's base-OOF cohort, keeping the join memory bounded."""
    if source.startswith("oof23_f"):
        fold = int(source.rsplit("f", 1)[1])
        pieces = []
        for side in ("long", "short"):
            path = ROOT / f"data_perp/artifacts/tp6_r3_r5_{side}_baseoof_fold{fold}_20260802_v1/base_oof_predictions.parquet"
            x = pd.read_parquet(path).rename(columns={"prob_adverse": "p_adverse", "prob_weak": "p_weak", "prob_clear": "p_clear"})
            x["source"] = source
            pieces.append(x[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", *BASE, "source"]])
        out = pd.concat(pieces, ignore_index=True)
    elif source == "ledger24":
        out = _read_ledger24()
    else:
        raise ValueError(f"unknown era source: {source}")
    out["__ts__"] = pd.to_datetime(out.__ts__, utc=True)
    out = out[(out.__ts__ >= pd.Timestamp(start, tz="UTC")) & (out.__ts__ < pd.Timestamp(end, tz="UTC"))].copy()
    if not np.allclose(out.gross_bps - out.net_bps, 100., atol=.01):
        raise ValueError(f"fixed 100-bps cost contract mismatch in {source} {start}")
    return out


def _read_context(ids: set[str]) -> pd.DataFrame:
    cols = ["candidate_id", *CONTEXT]
    got = []
    for part in sorted((PANEL / "parts").glob("*.parquet")):
        x = pd.read_parquet(part, columns=cols)
        x = x[x.candidate_id.isin(ids)]
        if not x.empty:
            got.append(x)
    out = pd.concat(got, ignore_index=True)
    if out.candidate_id.duplicated().any():
        raise ValueError("context candidate identity is not unique")
    return out


def _metric_rows(frame: pd.DataFrame, score: str, common: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for view, x in [("global", frame), ("long", frame[frame.side_name.eq("long")]), ("short", frame[frame.side_name.eq("short")])]:
        if x.empty:
            continue
        y = x.event.to_numpy(int); s = x[score].to_numpy(float)
        base = {**common, "view": view, "n": len(x), "event_prevalence": float(y.mean()),
                "roc_auc": float(roc_auc_score(y, s)) if y.min() != y.max() else np.nan,
                "pr_auc": float(average_precision_score(y, s)) if y.min() != y.max() else np.nan,
                "brier": float(brier_score_loss(y, np.clip(s, 1e-6, 1 - 1e-6))),
                "score_net_ic": float(spearmanr(s, x.net_bps).statistic),
                "mean_pred": float(s.mean()), "mean_net_bps": float(x.net_bps.mean())}
        rows.append({**base, "metric": "all", "top_fraction": np.nan, "net_bps": np.nan, "gross_bps": np.nan,
                     "selected_long_fraction": np.nan})
        order = x.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort")
        for top in TOPS:
            take = order.head(max(1, int(np.ceil(len(order) * top))))
            rows.append({**base, "metric": "top", "top_fraction": top, "net_bps": float(take.net_bps.mean()),
                         "gross_bps": float(take.gross_bps.mean()), "selected_long_fraction": float(take.side_name.eq("long").mean())})
    return rows


def _calibration(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    score = np.clip(score, 1e-5, 1 - 1e-5)
    if y.min() == y.max():
        return np.nan, np.nan
    lr = LogisticRegression(C=1e6, max_iter=200).fit(np.log(score / (1 - score)).reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def _psi(train: np.ndarray, test: np.ndarray) -> float:
    train = train[np.isfinite(train)]; test = test[np.isfinite(test)]
    if len(train) < 20 or len(test) < 20:
        return np.nan
    edges = np.unique(np.quantile(train, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(train, bins=edges)[0] / len(train)
    b = np.histogram(test, bins=edges)[0] / len(test)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))


def _shift_rows(train: pd.DataFrame, test: pd.DataFrame, common: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows = []
    for feature in FEATURES:
        a, b = train[feature].to_numpy(float), test[feature].to_numpy(float)
        rows.append({**common, "feature": feature, "train_missing": float((~np.isfinite(a)).mean()), "test_missing": float((~np.isfinite(b)).mean()),
                     "train_mean": float(np.nanmean(a)), "test_mean": float(np.nanmean(b)),
                     "train_std": float(np.nanstd(a)), "test_std": float(np.nanstd(b)), "psi": _psi(a, b),
                     "wasserstein": float(wasserstein_distance(a[np.isfinite(a)], b[np.isfinite(b)]))})
    # Adversarial AUC is a compact population-shift summary.  Fit balances eras
    # and uses no outcome or test labels.
    rng = np.random.default_rng(20260809)
    n = min(len(train), len(test), 75_000)
    a = train.iloc[rng.choice(len(train), n, replace=False)]
    b = test.iloc[rng.choice(len(test), n, replace=False)]
    x = np.vstack([_matrix(a), _matrix(b)]); y = np.r_[np.zeros(n), np.ones(n)]
    x = StandardScaler().fit_transform(x)
    adv = LogisticRegression(max_iter=200, C=.2, n_jobs=1).fit(x, y)
    adv_auc = float(roc_auc_score(y, adv.predict_proba(x)[:, 1]))
    corr_a = np.nan_to_num(np.corrcoef(_matrix(a), rowvar=False)); corr_b = np.nan_to_num(np.corrcoef(_matrix(b), rowvar=False))
    summary = {"adversarial_auc_in_sample": adv_auc, "correlation_frobenius_shift": float(np.linalg.norm(corr_a - corr_b, ord="fro"))}
    return rows, summary


def _concept_rows(test: pd.DataFrame, common: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for side, x in test.groupby("side_name", sort=True):
        y, s = x.event.to_numpy(int), x.meta_probability.to_numpy(float)
        intercept, slope = _calibration(y, s)
        # Binned residual is the practical conditional-shift diagnostic:
        # at a fixed training-model score, a non-zero actual-minus-predicted
        # conversion rate is an unmodelled change in P(event | available X).
        bin_id = pd.qcut(s, q=10, duplicates="drop", labels=False)
        for decile, z in x.assign(_bin=bin_id).groupby("_bin", sort=True):
            rows.append({**common, "side_name": side, "kind": "score_bin", "bin": int(decile), "n": len(z),
                         "mean_pred": float(z.meta_probability.mean()), "event_rate": float(z.event.mean()),
                         "event_residual": float(z.event.mean() - z.meta_probability.mean()), "mean_net_bps": float(z.net_bps.mean()),
                         "calibration_intercept": intercept, "calibration_slope": slope})
        rows.append({**common, "side_name": side, "kind": "aggregate", "bin": -1, "n": len(x),
                     "mean_pred": float(s.mean()), "event_rate": float(y.mean()), "event_residual": float(y.mean() - s.mean()),
                     "mean_net_bps": float(x.net_bps.mean()), "calibration_intercept": intercept, "calibration_slope": slope})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    stage = args.out.with_name(args.out.name + "_stage")
    if stage.exists():
        raise FileExistsError(f"stale transport stage exists: {stage}")
    stage.mkdir(parents=True)
    # Join each two-month era independently.  Reading the full 14-field panel
    # alongside all 1.3m OOF rows breaches the desktop-memory budget; this
    # bounded scan preserves exact candidate identity and the frozen contract.
    era_paths: dict[str, Path] = {}
    valid_counts = np.zeros(len(FEATURES), dtype=np.int64)
    total_rows = 0
    for name, start, end, source in ERAS:
        x = _load_era_raw(start, end, source)
        if x.empty:
            raise ValueError(f"empty era {name}")
        context = _read_context(set(x.candidate_id))
        x = x.merge(context, on="candidate_id", how="inner", validate="one_to_one")
        if len(x) != len(context):
            raise ValueError(f"context join lost OOF rows for {name}: {len(x)} != {len(context)}")
        valid_counts += np.isfinite(x[FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(float)).sum(axis=0)
        total_rows += len(x)
        x["event"] = x.net_bps.gt(50).astype(int)
        path = stage / f"{name}.parquet"
        x.to_parquet(path, index=False)
        era_paths[name] = path
        n = len(x)
        del x, context
        gc.collect()
        pa.default_memory_pool().release_unused()
        print(f"joined {name} rows={n:,}", flush=True)
    coverage = pd.Series(valid_counts / total_rows, index=FEATURES)
    if (coverage < .90).any():
        raise ValueError(f"feature coverage <90%: {coverage[coverage < .90].to_dict()}")
    labels = []
    for name, path in era_paths.items():
        x = pd.read_parquet(path)
        for side, z in x.groupby("side_name"):
            labels.append({"era": name, "side_name": side, "n": len(z), "p_net_gt_50": float(z.event.mean()),
                           "mean_net_bps": float(z.net_bps.mean()), "mean_net_given_positive": float(z.loc[z.event.eq(1), "net_bps"].mean()),
                           "mean_base_p_clear": float(z.p_clear.mean()), "note": "R3 realised-clear label unavailable in frozen base OOF ledger; base P(clear) reported, not substituted as a label"})
    result, shifts, concepts, preds = [], [], [], []
    era_names = list(era_paths)
    for mode in ("single_era", "expanding_prefix"):
        for i, train_name in enumerate(era_names[:-1]):
            train_names = era_names[:i + 1] if mode == "expanding_prefix" else [train_name]
            for test_name in era_names[i + 1:]:
                print(f"{mode}: train={train_name} test={test_name}", flush=True)
                train_all = pd.concat([pd.read_parquet(era_paths[n]) for n in train_names], ignore_index=True)
                test_all = pd.read_parquet(era_paths[test_name])
                common = {"mode": mode, "train_era": train_name, "train_eras": ",".join(train_names), "test_era": test_name,
                          "train_rows": len(train_all), "test_rows": len(test_all)}
                all_scored = []
                for side in ("long", "short"):
                    tr = train_all[train_all.side_name.eq(side)].sort_values("__ts__", kind="mergesort")
                    te = test_all[test_all.side_name.eq(side)].copy()
                    model = _model().fit(_matrix(tr), tr.event)
                    te["meta_probability"] = model.predict_proba(_matrix(te))[:, 1]
                    all_scored.append(te)
                scored = pd.concat(all_scored, ignore_index=True)
                # Ranking is globally pooled after probability is common, per
                # the policy contract.  No timestamp or per-side top-k is used.
                result += _metric_rows(scored, "meta_probability", common)
                crows, csummary = _shift_rows(train_all, test_all, common)
                shifts += crows
                concepts += _concept_rows(scored, {**common, **csummary})
                preds.append(scored[["candidate_id", "__ts__", "side_name", "net_bps", "event", "p_clear", "meta_probability"]].assign(**common))
                del train_all, test_all, scored
    args.out.mkdir(parents=True)
    pd.DataFrame(result).to_parquet(args.out / "transport_metrics.parquet", index=False)
    pd.DataFrame(shifts).to_parquet(args.out / "covariate_shift.parquet", index=False)
    pd.DataFrame(concepts).to_parquet(args.out / "concept_shift.parquet", index=False)
    pd.DataFrame(labels).to_parquet(args.out / "label_shift.parquet", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(args.out / "transport_predictions.parquet", index=False)
    metrics = pd.DataFrame(result)
    top = metrics[(metrics.view.eq("global")) & (metrics.metric.eq("top")) & (metrics.top_fraction.eq(.01))]
    lines = ["# TP6/SL4 M6 cross-era transport and shift report", "", "## Contract", "", "- Exact TP6/SL4/H12 labels; fixed 100 bps round-trip cost.", "- Same-side chronological base OOF only; M6 is side-local, then globally ranked.", "- Context: fixed 14 causal fields plus four same-side base outputs. Sparse 4.7%-coverage fields are excluded.", "- 2022 inverse-PI is deliberately external: product/candidate/context schema differ, so it is not pooled or zero-imputed.", "", "## Global top-1% net bps by matrix cell", "", "| Mode | Train through/era | Test era | Net bps | AUC | PR-AUC | IC |", "|---|---|---|---:|---:|---:|---:|"]
    for _, r in top.sort_values(["mode", "train_era", "test_era"]).iterrows():
        lines.append(f"| {r['mode']} | {r['train_era']} | {r['test_era']} | {r['net_bps']:+.2f} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['score_net_ic']:+.3f} |")
    lines += ["", "## Interpretation boundary", "", "This is a transport diagnosis, not M6 selection. A negative cell establishes failure of the stationary conversion relationship for that train/test path. A positive cell alone is insufficient evidence of a promotable head. `label_shift.parquet`, `covariate_shift.parquet`, and `concept_shift.parquet` provide the required three shift decompositions; realised R3-clear prevalence cannot be reconstructed exactly from the frozen OOF ledger and is explicitly not fabricated."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    manifest = {"schema": "tp6_m6_cross_era_transport_v1", "status": "COMPLETED_DIAGNOSTIC", "geometry": "TP6/SL4/H12", "cost_bps": 100,
                "base_lineage": "pre-existing strict same-side chronological OOF", "m6_target": "exact net > +50 bps", "context": CONTEXT,
                "coverage": coverage.to_dict(), "eras": ERAS, "historical_2022": {"status": "EXTERNAL_SEPARATE_POPULATION", "reason": "inverse-PI candidate/product and context schema are not compatible with the 14-field linear-PF contract", "prior_result": "2022 Jun-Aug rolling top1 pooled -46.53 bps"},
                "inputs_sha256": {"script": _sha(Path(__file__)), "ledger24": _sha(ROOT / "data_perp/artifacts/tp6_sl4_b10_bw4_base_oof_20260802_v1/base_oof_ledger.parquet")}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": int(total_rows), "cells": len(top), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
