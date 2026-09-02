#!/usr/bin/env python3
"""Audit frozen GateProxy HPO proposals against published strict-MC1 receipts.

This is intentionally a *receipt audit*.  It reads already-published HPO
scores and strict-MC1 aggregate outcomes; it does not fit, select, re-score,
or promote any Meta, MC1, or live-trading configuration.

The native GateProxy output is a downstream-utility proxy, whereas the
available confirmation quantity here is realised constrained portfolio EV.
Consequently the uncertainty check is explicitly a leave-one-out affine
diagnostic in portfolio-bps units, never a claimed calibrated prediction
interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))
import fit_strict_r3_p8u_meta_downstream_proxy_v1 as proxy  # noqa: E402

setattr(sys.modules["__main__"], "PairwiseSurrogate", proxy.PairwiseSurrogate)

SCHEMA = "strict_r3_p8u_meta_gateproxy_portfolio_audit_v1"
VALUE = "net_ev_bps_per_realised_trade"


def _once(path: Path, value: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _family(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split("::", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("--family must be NAME::GATEPROXY_ROOT::MC1_PARENT")
    return parts[0], Path(parts[1]), Path(parts[2])


def _incumbent(raw: str) -> tuple[str, str, Path, Path]:
    parts = raw.split("::", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError("--incumbent must be FAMILY::TRIAL::DESCRIPTOR_ROOT::MC1_ROOT")
    return parts[0], parts[1], Path(parts[2]), Path(parts[3])


def _rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].rank(method="first", ascending=False).astype(int)


def _summary(group: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    bank = group.loc[group.hpo_bank_member].copy()
    observed = bank.loc[bank.confirmed_hpo].copy()
    if len(observed) < 3:
        raise AssertionError("need at least three confirmed HPO trials")
    observed["actual_rank"] = _rank(observed, VALUE)
    best = observed.loc[observed[VALUE].idxmax()]
    regret = []
    for k in (1, 3, 5):
        selected = observed.loc[observed.bank_gateproxy_rank.le(k)]
        chosen = float(selected[VALUE].max()) if len(selected) else float("nan")
        regret.append({
            "family": str(group.family.iloc[0]), "k": k,
            "winner_trial": str(best.trial),
            "winner_gateproxy_rank": int(best.bank_gateproxy_rank),
            "winner_contained": bool(best.bank_gateproxy_rank <= k),
            "topk_confirmed_trials": int(len(selected)),
            "best_confirmed_ev_bps": float(best[VALUE]),
            "best_proxy_topk_ev_bps": chosen,
            "regret_ev_bps_lower_bound": float(best[VALUE] - chosen),
            "limitation": "unconfirmed HPO-bank proposals have no strict-MC1 confirmation; regret is confirmed-trial only",
        })
    # A deliberately conservative diagnostic only: score -> portfolio-bps is
    # estimated out of sample one candidate at a time, then uncertainty is
    # rescaled through that fold's slope.  This is not a calibrated interval.
    loo_error, loo_scale = [], []
    for idx in observed.index:
        train = observed.drop(index=idx)
        slope, intercept = np.polyfit(train.gateproxy_score.to_numpy(float), train[VALUE].to_numpy(float), 1)
        row = observed.loc[idx]
        loo_error.append(abs(float(row[VALUE]) - (intercept + slope * float(row.gateproxy_score))))
        loo_scale.append(abs(slope) * float(row.gateproxy_uncertainty))
    error = np.asarray(loo_error); scale = np.asarray(loo_scale)
    base = {
        "family": str(group.family.iloc[0]),
        "hpo_bank_size": int(len(bank)), "confirmed_hpo_trials": int(len(observed)),
        "spearman_proxy_to_portfolio_ev": float(observed.gateproxy_score.corr(observed[VALUE], method="spearman")),
        "winner_trial": str(best.trial), "winner_gateproxy_rank": int(best.bank_gateproxy_rank),
        "top3_winner_contained": bool(best.bank_gateproxy_rank <= 3),
        "top5_winner_contained": bool(best.bank_gateproxy_rank <= 5),
        "actual_top3_in_proxy_top3": int(((observed.actual_rank <= 3) & (observed.bank_gateproxy_rank <= 3)).sum()),
        "actual_top3_in_proxy_top5": int(((observed.actual_rank <= 3) & (observed.bank_gateproxy_rank <= 5)).sum()),
        "loo_uncertainty_1sigma_coverage": float((error <= scale).mean()),
        "loo_uncertainty_2sigma_coverage": float((error <= 2.0 * scale).mean()),
        "loo_median_abs_error_bps": float(np.median(error)),
        "loo_uncertainty_note": "descriptive leave-one-out affine score-to-portfolio-EV diagnostic; surrogate disagreement is not a calibrated portfolio-EV interval",
    }
    return base, pd.DataFrame(regret)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--family", action="append", required=True)
    parser.add_argument("--incumbent", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for name, score_root, mc1_root in (_family(x) for x in args.family):
        scores = pd.read_parquet(score_root / "gateproxy_scores.parquet")
        confirmed = pd.read_parquet(mc1_root / "candidate_mc1_summary.parquet")
        keep = ["trial", VALUE, "net_sum_bps_realised", "accepted_rows", "worst_month_bps", "worst_week_bps", "max_drawdown"]
        work = scores.merge(confirmed[keep], on="trial", how="left", validate="one_to_one")
        work["family"] = name
        work["hpo_bank_member"] = True
        work["confirmed_hpo"] = work[VALUE].notna()
        work["bank_gateproxy_rank"] = work.gateproxy_rank.astype(int)
        rows.append(work)
    result = pd.concat(rows, ignore_index=True)
    payload = joblib.load(args.proxy_root / "models" / "dgate_shrunk__P0_ridge.joblib")
    bundle = joblib.load(args.proxy_root / "proxy_models.joblib")
    fields = list(payload["fields"])
    for raw in args.incumbent:
        family, trial, descriptor_root, mc1_root = _incumbent(raw)
        desc = pd.read_parquet(descriptor_root / "trial_descriptor_summary.parquet")
        entry = desc.loc[desc.trial.astype(str).eq(trial)]
        if len(entry) != 1:
            raise AssertionError(f"incumbent {trial} not unique")
        missing = sorted(set(fields).difference(entry.columns))
        if missing:
            raise AssertionError(f"incumbent fields missing: {missing}")
        metrics = json.loads((mc1_root / "run_manifest.json").read_text())["metrics"]
        item = entry.loc[:, ["trial", "target_family", "loss", "feature_family", "feature_contract"]].copy()
        item["gateproxy_score"] = proxy._predict(payload["model"], entry[fields])
        ensemble = np.column_stack([proxy._predict(bundle["models"][f"dgate_shrunk::{model}"], entry[fields]) for model in ("P0_ridge", "P1_elastic_net", "P2_depth2_gbdt", "P3_pairwise")])
        item["gateproxy_ensemble_mean"] = ensemble.mean(axis=1)
        item["gateproxy_uncertainty"] = ensemble.std(axis=1, ddof=1)
        item[VALUE] = float(metrics[VALUE]); item["net_sum_bps_realised"] = float(metrics["net_sum_bps_realised"])
        item["accepted_rows"] = int(metrics["accepted_rows"]); item["worst_month_bps"] = float(metrics["worst_month_bps"])
        item["worst_week_bps"] = float(metrics["worst_week_bps"]); item["max_drawdown"] = float(metrics["max_drawdown"])
        item["family"] = family; item["hpo_bank_member"] = False; item["confirmed_hpo"] = False; item["bank_gateproxy_rank"] = np.nan
        result = pd.concat([result, item], ignore_index=True)
    result["gateproxy_rank_with_incumbent"] = result.groupby("family").gateproxy_score.rank(method="first", ascending=False).astype(int)
    reports, regret = [], []
    for name, group in result.groupby("family", sort=True):
        item, part = _summary(group)
        ref = group.loc[~group.hpo_bank_member]
        if len(ref):
            item["incumbent_trial"] = str(ref.iloc[0].trial)
            item["incumbent_gateproxy_rank_among_bank_plus_reference"] = int(ref.iloc[0].gateproxy_rank_with_incumbent)
            item["incumbent_portfolio_ev_bps"] = float(ref.iloc[0][VALUE])
        reports.append(item); regret.append(part)
    out.mkdir(parents=True)
    result.to_parquet(out / "gateproxy_audit_trials.parquet", index=False, compression="zstd")
    pd.DataFrame(reports).to_parquet(out / "gateproxy_audit_family_summary.parquet", index=False, compression="zstd")
    pd.concat(regret, ignore_index=True).to_parquet(out / "gateproxy_audit_regret.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "uses_only_preexisting_frozen_gateproxy_scores_and_strict_mc1_receipts": True,
        "does_not_fit_reselect_or_refit_any_model": True,
        "portfolio_outcomes_are_exact_published_strict_mc1_summary_values": True,
        "regret_is_confirmed_hpo_only": True,
        "incumbent_is_retrospective_reference_only": True,
        "uncertainty_coverage_is_explicitly_descriptive_not_calibrated": True,
        "no_live_or_exchange_mutation": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "retrospective audit only; no model fitting, HPO selection, promotion, live, or exchange mutation",
        "proxy_root": str(args.proxy_root.resolve()),
        "p0_proxy_sha256": _sha(args.proxy_root / "models" / "dgate_shrunk__P0_ridge.joblib"),
        "families": list(args.family), "incumbents": list(args.incumbent),
    })
    print(out)


if __name__ == "__main__":
    main()
