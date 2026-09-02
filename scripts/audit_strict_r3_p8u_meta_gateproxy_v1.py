#!/usr/bin/env python3
"""Retrospectively audit frozen GateProxy proposals against confirmed MC1s.

This script is deliberately audit-only.  It never fits a proxy, opens a new
candidate selection, scores Meta rows, refits MC1, or changes any research or
live contract.  It computes two separate diagnostics:

* native GateProxy calibration against the frozen-definition ``dgate_shrunk``;
* constrained-portfolio EV ranking/regret among actually confirmed trials.

The latter is necessarily partial when an HPO bank has unconfirmed trials.
That limitation is persisted rather than silently treating them as failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import build_strict_r3_p8u_meta_proxy_downstream_labels_v1 as labels  # noqa: E402
import fit_strict_r3_p8u_meta_downstream_proxy_v1 as proxy  # noqa: E402

# The frozen P3 surrogate was serialised while its class lived in a script
# ``__main__`` module.  Expose the same class name solely so this audit can
# read the already-fitted bundle; no surrogate is fitted or altered here.
setattr(sys.modules["__main__"], "PairwiseSurrogate", proxy.PairwiseSurrogate)


SCHEMA = "strict_r3_p8u_meta_gateproxy_audit_v1"
GATE_COMPONENTS = labels.GATE_COMPONENTS


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_family(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split("::", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("--family must be NAME::GATEPROXY_ROOT::CANDIDATE_MC1_PARENT")
    return parts[0], Path(parts[1]).resolve(), Path(parts[2]).resolve()


def _parse_incumbent(raw: str) -> tuple[str, str, Path, Path]:
    parts = raw.split("::", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError("--incumbent must be FAMILY::TRIAL::DESCRIPTOR_ROOT::MC1_ROOT")
    return parts[0], parts[1], Path(parts[2]).resolve(), Path(parts[3]).resolve()


def _load_proxy(proxy_root: Path) -> tuple[object, list[str]]:
    # The supplied GateProxy receipts bind the selected model hash.  This audit
    # reads that frozen P0 Ridge artifact only; it does not choose a model.
    model_path = proxy_root / "models" / "dgate_shrunk__P0_ridge.joblib"
    payload = joblib.load(model_path)
    if payload.get("schema") != proxy.SCHEMA or payload.get("target") != "dgate_shrunk":
        raise AssertionError("invalid frozen GateProxy payload")
    return payload["model"], list(payload["fields"])


def _frozen_norm(label_root: Path) -> tuple[dict[str, dict[str, float]], float]:
    manifest = json.loads((label_root / "run_manifest.json").read_text())
    normalisation = manifest.get("normalisation")
    if not isinstance(normalisation, dict):
        raise AssertionError("frozen GateProxy labels lack normalisation")
    table = pd.read_parquet(label_root / "downstream_trial_labels.parquet")
    tau = labels._robust_location_scale(table["dgate_raw"])[1]
    return normalisation, float(tau)


def _gate_value(metrics: dict[str, Any], weekly: pd.DataFrame, *, normalisation: dict[str, dict[str, float]], tau: float, seed: int) -> dict[str, float]:
    raw = 0.0
    for column, weight in GATE_COMPONENTS:
        spec = normalisation[column]
        raw += float(weight) * (float(metrics[column]) - float(spec["location"])) / float(spec["scale"])
    rng = np.random.default_rng(seed)
    # Semantically identical to the frozen builder's 1,000 weekly block
    # bootstraps, but vectorised so this audit does not become another costly
    # research run.  The RNG draw order is unchanged from 1,000 sequential
    # ``integers(..., size=n_weeks)`` calls.
    indices = rng.integers(0, len(weekly), size=(1000, len(weekly)))
    components = {
        "gate_admitted_ev_delta_bps": weekly.gate_admitted_ev_delta_bps.to_numpy(float),
        "gate_total_utility_delta_bps_per_timestamp": weekly.gate_total_utility_delta_bps.to_numpy(float),
        "gate_precision_gt50_delta": weekly.gate_precision_gt50_delta.to_numpy(float),
        "gate_precision_gt100_delta": weekly.gate_precision_gt100_delta.to_numpy(float),
        # The frozen label builder stores the weekly counterpart under the
        # concise ``gate_volume_delta`` name; the longer name is the aggregate
        # normalisation component name.
        "gate_volume_delta_per_timestamp": weekly.gate_volume_delta.to_numpy(float),
    }
    draws = np.zeros(len(indices), dtype=float)
    for column, weight in GATE_COMPONENTS:
        if column == "gate_weekly_q10_delta_bps":
            values = np.quantile(components["gate_admitted_ev_delta_bps"][indices], .10, axis=1)
        else:
            values = components[column][indices].mean(axis=1)
        params = normalisation[column]
        draws += float(weight) * (values - float(params["location"])) / float(params["scale"])
    se = float(np.std(draws, ddof=1))
    reliability = float(tau**2 / (tau**2 + se**2))
    return {"dgate_raw_frozen": raw, "dgate_bootstrap_se_frozen": se, "dgate_reliability_frozen": reliability, "dgate_shrunk_frozen": reliability * raw}


def _actual_row(*, control: pd.DataFrame, root: Path, trial: str, normalisation: dict[str, dict[str, float]], tau: float) -> dict[str, Any]:
    manifest, candidate, _ = labels._root_receipt(root)
    merged = labels._assert_matched(control, candidate, name=trial)
    metrics, weekly = labels._trial_metrics(trial, merged, dict(manifest["metrics"]))
    seed = 1729 + int.from_bytes(hashlib.sha256(trial.encode()).digest()[:4], "little")
    return {**metrics, **_gate_value(metrics, weekly, normalisation=normalisation, tau=tau, seed=seed)}


def _rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="first", ascending=False).astype(int)


def _safe_spearman(frame: pd.DataFrame, left: str, right: str) -> float:
    work = frame.loc[:, [left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    return float(work[left].corr(work[right], method="spearman")) if len(work) >= 3 else float("nan")


def _family_summary(family: str, rows: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    confirmed = rows.loc[rows.confirmed_hpo].copy()
    if confirmed.empty:
        raise AssertionError(f"{family}: no confirmed HPO trial")
    # ``_trial_metrics`` preserves the explicit portfolio prefix used by the
    # frozen MC1 receipts.
    confirmed["actual_ev_rank"] = _rank_desc(confirmed.portfolio_net_ev_bps_per_trade)
    confirmed["actual_gate_rank"] = _rank_desc(confirmed.dgate_shrunk_frozen)
    best = confirmed.loc[confirmed.portfolio_net_ev_bps_per_trade.idxmax()]
    all_scored = rows.loc[rows.hpo_bank_member].copy()
    regrets: list[dict[str, Any]] = []
    for k in (1, 3, 5):
        observed = confirmed.loc[confirmed.bank_gateproxy_rank.le(k)]
        top_value = float(observed.portfolio_net_ev_bps_per_trade.max()) if len(observed) else float("nan")
        regrets.append({
            "family": family,
            "k": k,
            "bank_size": int(len(all_scored)),
            "confirmed_bank_trials": int(len(confirmed)),
            "winner_trial": str(best.trial),
            "winner_gateproxy_rank": int(best.bank_gateproxy_rank),
            "winner_contained": bool(int(best.bank_gateproxy_rank) <= k),
            "topk_confirmed_trials": int(len(observed)),
            "best_confirmed_ev_bps": float(best.portfolio_net_ev_bps_per_trade),
            "best_observed_proxy_topk_ev_bps": top_value,
            "regret_ev_bps_lower_bound": float(best.portfolio_net_ev_bps_per_trade - top_value) if np.isfinite(top_value) else float("nan"),
            "limitation": "unconfirmed HPO-bank trials have no realised MC1 value; regret is relative to confirmed trials only",
        })
    error = confirmed.dgate_shrunk_frozen - confirmed.gateproxy_score
    scale = confirmed.gateproxy_uncertainty.replace(0.0, np.nan)
    summary = {
        "family": family,
        "hpo_bank_size": int(len(all_scored)),
        "confirmed_hpo_trials": int(len(confirmed)),
        "spearman_proxy_to_native_dgate": _safe_spearman(confirmed, "gateproxy_score", "dgate_shrunk_frozen"),
        "spearman_proxy_to_portfolio_ev": _safe_spearman(confirmed, "gateproxy_score", "portfolio_net_ev_bps_per_trade"),
        "winner_trial": str(best.trial),
        "winner_gateproxy_rank": int(best.bank_gateproxy_rank),
        "top3_winner_contained": bool(int(best.bank_gateproxy_rank) <= 3),
        "top5_winner_contained": bool(int(best.bank_gateproxy_rank) <= 5),
        "actual_top3_in_proxy_top3": int(((confirmed.actual_ev_rank <= 3) & (confirmed.bank_gateproxy_rank <= 3)).sum()),
        "actual_top3_in_proxy_top5": int(((confirmed.actual_ev_rank <= 3) & (confirmed.bank_gateproxy_rank <= 5)).sum()),
        "uncertainty_1sigma_coverage": float((error.abs() <= scale).mean()),
        "uncertainty_2sigma_coverage": float((error.abs() <= 2.0 * scale).mean()),
        "median_abs_standardized_error": float((error.abs() / scale).median()),
        "uncertainty_note": "four-surrogate disagreement is an uncalibrated acquisition uncertainty; coverage is diagnostic only",
    }
    return summary, pd.DataFrame(regrets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--frozen-label-root", type=Path, required=True)
    parser.add_argument("--family", action="append", required=True, help="NAME::GATEPROXY_ROOT::CANDIDATE_MC1_PARENT")
    parser.add_argument("--incumbent", action="append", default=[], help="FAMILY::TRIAL::DESCRIPTOR_ROOT::MC1_ROOT")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    control_manifest, control, _ = labels._root_receipt(args.control.resolve())
    normalisation, tau = _frozen_norm(args.frozen_label_root.resolve())
    model, fields = _load_proxy(args.proxy_root.resolve())
    parts: list[pd.DataFrame] = []
    for family, score_root, mc1_parent in (_parse_family(item) for item in args.family):
        score = pd.read_parquet(score_root / "gateproxy_scores.parquet")
        summary = pd.read_parquet(mc1_parent / "candidate_mc1_summary.parquet")
        rows = score.copy()
        rows["family"] = family
        rows["hpo_bank_member"] = True
        rows["confirmed_hpo"] = rows.trial.isin(set(summary.trial.astype(str)))
        rows["bank_gateproxy_rank"] = rows["gateproxy_rank"].astype(int)
        actual = []
        for record in summary.to_dict("records"):
            trial = str(record["trial"])
            root = mc1_parent / "candidate_mc1" / trial
            actual.append(_actual_row(control=control, root=root, trial=trial, normalisation=normalisation, tau=tau))
        actual_frame = pd.DataFrame(actual)
        rows = rows.merge(actual_frame, on="trial", how="left", validate="one_to_one")
        parts.append(rows)
    result = pd.concat(parts, ignore_index=True)
    for raw in args.incumbent:
        family, trial, descriptor_root, mc1_root = _parse_incumbent(raw)
        descriptor = pd.read_parquet(descriptor_root / "trial_descriptor_summary.parquet")
        entry = descriptor.loc[descriptor.trial.astype(str).eq(trial)].copy()
        if len(entry) != 1:
            raise AssertionError(f"{trial}: incumbent descriptor not unique")
        missing = sorted(set(fields).difference(entry.columns))
        if missing:
            raise AssertionError(f"{trial}: incumbent descriptor missing frozen GateProxy fields {missing}")
        row = entry.loc[:, ["trial", "target_family", "loss", "feature_family", "feature_contract"]].copy()
        row["descriptor_root"] = descriptor_root.name
        row["gateproxy_score"] = proxy._predict(model, entry[fields])
        bundle = joblib.load(args.proxy_root.resolve() / "proxy_models.joblib")
        ensemble = np.column_stack([proxy._predict(bundle["models"][f"dgate_shrunk::{name}"], entry[fields]) for name in ("P0_ridge", "P1_elastic_net", "P2_depth2_gbdt", "P3_pairwise")])
        row["gateproxy_ensemble_mean"] = ensemble.mean(axis=1)
        row["gateproxy_uncertainty"] = ensemble.std(axis=1, ddof=1)
        row["family"] = family; row["hpo_bank_member"] = False; row["confirmed_hpo"] = False; row["bank_gateproxy_rank"] = np.nan
        actual = _actual_row(control=control, root=mc1_root, trial=trial, normalisation=normalisation, tau=tau)
        for key, value in actual.items():
            row[key] = value
        result = pd.concat([result, row], ignore_index=True)
    result["gateproxy_rank_with_incumbent"] = result.groupby("family").gateproxy_score.rank(method="first", ascending=False).astype(int)
    summaries: list[dict[str, Any]] = []
    regret_parts: list[pd.DataFrame] = []
    for family, group in result.groupby("family", sort=True):
        summary, regret = _family_summary(str(family), group)
        incumbent = group.loc[~group.hpo_bank_member]
        if len(incumbent):
            summary["incumbent_trial"] = str(incumbent.iloc[0].trial)
            summary["incumbent_gateproxy_rank_among_bank_plus_reference"] = int(incumbent.iloc[0].gateproxy_rank_with_incumbent)
            summary["incumbent_portfolio_ev_bps"] = float(incumbent.iloc[0].portfolio_net_ev_bps_per_trade)
        summaries.append(summary); regret_parts.append(regret)
    out.mkdir(parents=True)
    result.to_parquet(out / "gateproxy_audit_trials.parquet", index=False, compression="zstd")
    pd.DataFrame(summaries).to_parquet(out / "gateproxy_audit_family_summary.parquet", index=False, compression="zstd")
    pd.concat(regret_parts, ignore_index=True).to_parquet(out / "gateproxy_audit_regret.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "uses_only_completed_frozen_gateproxy_and_mc1_receipts": True,
        "does_not_fit_or_reselect_gateproxy_or_meta_hpo": True,
        "native_dgate_uses_the_frozen_88_trial_normalisation": True,
        "candidate_and_control_mc1_identity_and_policy_are_exact": True,
        "portfolio_regret_is_explicitly_limited_to_confirmed_trials": True,
        "incumbent_is_retrospective_reference_only": True,
        "no_live_or_exchange_mutation": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "retrospective GateProxy audit only; no HPO/model/MC1 refit, selection, promotion, live, or exchange mutation",
        "control": str(args.control.resolve()),
        "proxy_root": str(args.proxy_root.resolve()),
        "proxy_model_sha256": _sha(args.proxy_root.resolve() / "models" / "dgate_shrunk__P0_ridge.joblib"),
        "frozen_label_root": str(args.frozen_label_root.resolve()),
        "frozen_dgate_scale": {"tau": tau, "normalisation": normalisation},
        "families": list(args.family), "incumbents": list(args.incumbent),
    })
    print(out)


if __name__ == "__main__":
    main()
