#!/usr/bin/env python3
"""Causal empirical-Bayes current-v5 multiplier over the frozen BCF MC1 map.

This is an offline challenger only.  BCF remains the sole primary EV map.
The current-v5 map can adjust it only through a bounded multiplier estimated
from *prior-resolved* parent-policy outcomes.  It never replaces BCF scores,
changes the base/residual models, or contacts an exchange.

For each UTC decision day, the posterior is fitted from the preceding 21
calendar days of outcomes whose label-availability timestamp predates that
day.  The posterior contains:

* an empirical-Bayes residual mean (BCF level error); and
* an empirical-Bayes slope on ``current_v5_EV - BCF_EV``.

The slope is deliberately included so that the multiplier is not merely a
recent average correction.  Every multiplier is clipped to a declared bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_canonical_sr_e2_mc1_input_ablation as base
from scripts import run_canonical_sr_e2_mc1_august_extension as august


HISTORY_CURRENT = august.HISTORY_CURRENT
HISTORY_BCF = august.HISTORY_BCF
DEFAULT_PREPARED = august.DEFAULT_PREPARED
DEFAULT_AUGUST_LABELS = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_august_parent_policy_labels_20260831_v1/frozen_policy_labels.parquet"
DEFAULT_FROZEN_MAP = august.DEFAULT_FROZEN_MAP
DEFAULT_OUT = ROOT / "data_perp/artifacts/frozen_bcf_current_eb_multiplier_20260831_v1"
EVAL_START = pd.Timestamp("2026-06-01T00:00:00Z")
EVAL_END = pd.Timestamp("2026-08-19T00:00:00Z")
LOOKBACK_DAYS = 21
MIN_DAYS = 10
MIN_ROWS = 500
ADMISSION_BPS = 50.0


@dataclass(frozen=True)
class Variant:
    name: str
    use_mean: bool
    use_slope: bool
    weight: float
    lower: float
    upper: float
    hard_current_gate: bool = False


VARIANTS = (
    Variant("B0_frozen_bcf_primary", False, False, 0.0, 1.0, 1.0),
    Variant("B1_frozen_dual_gate", False, False, 0.0, 1.0, 1.0, hard_current_gate=True),
    Variant("M1_EB_mean_w50_b075_125", True, False, 0.50, 0.75, 1.25),
    Variant("M2_EB_mean_slope_w50_b075_125", True, True, 0.50, 0.75, 1.25),
    Variant("M3_EB_mean_slope_w100_b050_150", True, True, 1.00, 0.50, 1.50),
    Variant("M4_EB_mean_slope_demotion_only", True, True, 1.00, 0.50, 1.00),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sources(prepared: Path, labels_path: Path, frozen_map: Path) -> tuple[base.FamilySource, base.FamilySource, pd.DataFrame]:
    current_history = base._load_family(HISTORY_CURRENT, "current_v5")
    bcf_history = base._load_family(HISTORY_BCF, "bcf")
    current_aug = august._load_aug_scores(prepared / "current_scores_core_complete.parquet", family="current_v5")
    bcf_aug = august._load_aug_scores(prepared / "bcf_scores_core_complete.parquet", family="bcf")
    if not current_aug["candidate_id"].equals(bcf_aug["candidate_id"]):
        raise AssertionError("August BCF/current candidate identity differs")
    labels_aug = august._load_aug_labels(labels_path, pd.Index(current_aug["candidate_id"]))
    mapping = pd.read_parquet(
        frozen_map,
        columns=["candidate_id", "bcf_mc1_expected_bps", "current_v5_mc1_expected_bps", "bcf_mc1_available", "current_v5_mc1_available"],
    ).copy()
    mapping["candidate_id"] = mapping["candidate_id"].astype(str)
    mapping = mapping.set_index("candidate_id").loc[current_aug["candidate_id"]].reset_index()
    if not (mapping["bcf_mc1_available"].fillna(False) & mapping["current_v5_mc1_available"].fillna(False)).all():
        raise AssertionError("August frozen-map source contains an unavailable score-complete row")
    current = august._append_family(
        current_history, current_aug, labels_aug,
        pd.to_numeric(mapping["current_v5_mc1_expected_bps"], errors="raise"),
    )
    bcf = august._append_family(
        bcf_history, bcf_aug, labels_aug,
        pd.to_numeric(mapping["bcf_mc1_expected_bps"], errors="raise"),
    )
    labels = base._candidate_labels(current, bcf)
    return current, bcf, labels


def _posterior(training: pd.DataFrame, day: pd.Timestamp) -> tuple[float, float, int, int, str]:
    """Return EB posterior mean/slope in 100-bps units, using only prior labels."""
    start = day - pd.Timedelta(days=LOOKBACK_DAYS)
    resolved_valid = training["policy_path_valid"].astype("boolean").fillna(False).astype(bool)
    fit = training.loc[
        training["__decision_ts__"].ge(start)
        & training["__decision_ts__"].lt(day)
        & training["policy_label_available_ts"].lt(day)
        & resolved_valid
    ].copy()
    fit["policy_net_bps"] = pd.to_numeric(fit["policy_net_bps"], errors="coerce")
    fit = fit.loc[np.isfinite(fit["policy_net_bps"])].copy()
    fit["fit_day"] = fit["__decision_ts__"].dt.normalize()
    n_days = int(fit["fit_day"].nunique())
    if n_days < MIN_DAYS or len(fit) < MIN_ROWS:
        return 0.0, 0.0, int(len(fit)), n_days, "insufficient_prior_resolved_support"

    # Day balancing prevents a high-turnover shock day from dominating the
    # local calibration.  x measures incremental current-v5 information after
    # taking the frozen BCF map as the primary estimate.
    per_day = fit.groupby("fit_day")["candidate_id"].transform("size").astype(float)
    weight = 1.0 / per_day
    x = ((fit["current_mc1_expected_bps"] - fit["bcf_mc1_expected_bps"]) / 100.0).clip(-3.0, 3.0).to_numpy(float)
    y = ((fit["policy_net_bps"] - fit["bcf_mc1_expected_bps"]) / 100.0).clip(-5.0, 5.0).to_numpy(float)
    w = weight.to_numpy(float)
    design = np.column_stack([np.ones(len(fit), dtype=float), x])
    # Empirical-Bayes Gaussian posterior: estimate the observation variance
    # from the local sample, then shrink the residual level and slope toward
    # zero with deliberately weak, fixed priors (50 bps and 75 bps units).
    center = float(np.average(y, weights=w))
    sigma2 = max(float(np.average((y - center) ** 2, weights=w)), 0.25**2)
    prior_precision = np.diag([1.0 / 0.50**2, 1.0 / 0.75**2])
    precision = (design.T * w) @ design / sigma2 + prior_precision
    rhs = (design.T @ (w * y)) / sigma2
    alpha, beta = np.linalg.solve(precision, rhs)
    return float(alpha), float(beta), int(len(fit)), n_days, "posterior"


def _multiplier_panel(target_free: pd.DataFrame, outcome: pd.DataFrame, variant: Variant) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_free = target_free.copy()
    outcome = outcome.copy()
    target_free["__decision_ts__"] = pd.to_datetime(target_free["__decision_ts__"], utc=True, errors="raise")
    outcome["__decision_ts__"] = pd.to_datetime(outcome["__decision_ts__"], utc=True, errors="raise")
    outcome["policy_label_available_ts"] = pd.to_datetime(outcome["policy_label_available_ts"], utc=True, errors="raise")
    eval_rows = target_free.loc[
        target_free["__decision_ts__"].ge(EVAL_START) & target_free["__decision_ts__"].lt(EVAL_END)
    ].copy()
    result: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for day, held in eval_rows.groupby(eval_rows["__decision_ts__"].dt.normalize(), sort=True):
        alpha, beta, rows, days, status = _posterior(outcome, day)
        x = ((held["current_mc1_expected_bps"] - held["bcf_mc1_expected_bps"]) / 100.0).clip(-3.0, 3.0)
        adjustment = 100.0 * ((alpha if variant.use_mean else 0.0) + (beta * x if variant.use_slope else 0.0))
        # A 100-bps denominator deliberately limits the influence on marginal
        # BCF estimates.  The declared outer bounds are the final authority.
        raw_multiplier = 1.0 + variant.weight * adjustment / np.maximum(np.abs(held["bcf_mc1_expected_bps"]), 100.0)
        multiplier = raw_multiplier.clip(variant.lower, variant.upper)
        if status != "posterior":
            multiplier = pd.Series(1.0, index=held.index)
        held = held.copy()
        held["eb_residual_mean_100bps"] = alpha
        held["eb_current_incremental_slope"] = beta
        held["eb_multiplier"] = multiplier.to_numpy(float)
        held["adjusted_bcf_expected_bps"] = held["bcf_mc1_expected_bps"] * held["eb_multiplier"]
        held["bcf_primary_admitted"] = held["bcf_mc1_expected_bps"].ge(ADMISSION_BPS)
        held["current_hard_admitted"] = held["current_mc1_expected_bps"].ge(ADMISSION_BPS)
        held["multiplier_admitted"] = held["adjusted_bcf_expected_bps"].ge(ADMISSION_BPS)
        held["admitted"] = held["multiplier_admitted"] & (held["current_hard_admitted"] if variant.hard_current_gate else True)
        held["auction_priority_bps"] = held["adjusted_bcf_expected_bps"]
        held["eb_status"] = status
        result.append(held)
        audits.append({
            "variant": variant.name, "decision_day": day, "status": status,
            "fit_rows": rows, "fit_days": days, "posterior_mean_100bps": alpha,
            "posterior_slope": beta, "held_rows": int(len(held)),
            "multiplier_mean": float(held["eb_multiplier"].mean()),
            "multiplier_min": float(held["eb_multiplier"].min()),
            "multiplier_max": float(held["eb_multiplier"].max()),
        })
    return pd.concat(result, ignore_index=True), pd.DataFrame(audits)


def _replay(panel: pd.DataFrame, outcome: pd.DataFrame, variant: Variant, out: Path) -> tuple[dict[str, object], pd.DataFrame]:
    # The frozen-control adapter has its own historical auction fields.  This
    # challenger must replace them with its bounded-multiplier decision fields
    # rather than allowing pandas to create ambiguous suffixes.
    outcome = outcome.drop(columns=["dual_admitted", "auction_priority_bps"], errors="ignore").merge(
        panel.loc[:, ["candidate_id", "admitted", "auction_priority_bps"]],
        on="candidate_id", how="inner", validate="one_to_one",
    )
    candidates = base._to_candidates(outcome, admission=outcome["admitted"], priority=outcome["auction_priority_bps"])
    decisions, equity, _ = base.replay_candidates(
        candidates, base._params(), mode="global_auction", ev_curve=base.CAUSAL_AUCTION_CURVE,
        market_mode="perps", initial_wallet=1000.0,
    )
    if not decisions.empty:
        provenance = candidates.loc[:, ["candidate_id"]].reset_index(drop=True)
        provenance.index.name = "candidate_index"
        decisions = decisions.merge(provenance, on="candidate_index", how="left", validate="many_to_one")
        decisions["policy_outcome_available"] = True
    decisions.to_parquet(out / f"{variant.name}_portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{variant.name}_portfolio_equity.parquet", index=False, compression="zstd")
    metric = base._metrics(decisions, equity, variant.name, "2026_jun_to_aug18")
    metric["admitted_rows"] = int(panel["admitted"].sum())
    metric["bcf_primary_rows"] = int(panel["bcf_primary_admitted"].sum())
    metric["promoted_rows"] = int((~panel["bcf_primary_admitted"] & panel["admitted"]).sum())
    metric["demoted_rows"] = int((panel["bcf_primary_admitted"] & ~panel["admitted"]).sum())
    metric["auction_priority"] = "bounded_eb_current_multiplier_x_frozen_bcf"
    accepted = decisions.loc[decisions.get("accepted", pd.Series(index=decisions.index, dtype=bool)).fillna(False).astype(bool)].copy()
    if accepted.empty:
        monthly = pd.DataFrame(columns=["variant", "month", "trades", "net_ev_bps_per_trade", "net_sum_bps"])
    else:
        accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="raise")
        accepted["month"] = accepted["timestamp"].dt.strftime("%Y-%m")
        accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
        monthly = accepted.groupby("month", sort=True).agg(
            trades=("net_bps", "size"), net_ev_bps_per_trade=("net_bps", "mean"), net_sum_bps=("net_bps", "sum"),
        ).reset_index()
        monthly.insert(0, "variant", variant.name)
    return metric, monthly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--august-labels", type=Path, default=DEFAULT_AUGUST_LABELS)
    parser.add_argument("--frozen-map", type=Path, default=DEFAULT_FROZEN_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prepared, labels_path, frozen_map, out = (args.prepared.resolve(), args.august_labels.resolve(), args.frozen_map.resolve(), args.out.resolve())
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True)

    current, bcf, labels = _load_sources(prepared, labels_path, frozen_map)
    target_free, outcome = base._frozen_control(current, bcf, labels)
    # Target-free control: no policy fields may be present before the final
    # outcome merge.  The multiplier itself uses only stored map values and a
    # prior-resolved posterior.
    leaked = sorted(base.POLICY_FORBIDDEN.intersection(target_free.columns))
    if leaked:
        raise AssertionError(f"frozen target-free panel leaked policy fields: {leaked}")
    target_free.to_parquet(out / "frozen_target_free_source.parquet", index=False, compression="zstd")

    metrics: list[dict[str, object]] = []
    monthly: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for variant in VARIANTS:
        panel, audit = _multiplier_panel(target_free, outcome, variant)
        panel.to_parquet(out / f"{variant.name}_target_free_admission.parquet", index=False, compression="zstd")
        metric, period = _replay(panel, outcome, variant, out)
        metrics.append(metric); monthly.append(period); audits.append(audit)
    summary = pd.DataFrame(metrics)
    baseline = summary.loc[summary["arm"].eq("B1_frozen_dual_gate")].iloc[0]
    for field in ["accepted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown"]:
        summary[f"delta_vs_dual_{field}"] = pd.to_numeric(summary[field], errors="coerce") - float(baseline[field])
    summary.to_csv(out / "portfolio_summary.csv", index=False)
    summary.to_parquet(out / "portfolio_summary.parquet", index=False, compression="zstd")
    pd.concat(monthly, ignore_index=True).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    pd.concat(audits, ignore_index=True).to_parquet(out / "posterior_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "frozen_bcf_current_empirical_bayes_multiplier_v1",
        "scope": "offline challenger; no live/canonical mutation or exchange calls",
        "period": {"start": EVAL_START.isoformat(), "end_exclusive": EVAL_END.isoformat(), "august_note": "paired archive ends 2026-08-18 21:00 UTC"},
        "base_map": "frozen BCF MC1 expected EV",
        "current_role": "bounded multiplier only; EB residual mean plus current-minus-BCF slope; no replacement of BCF map",
        "posterior": {"lookback_days": LOOKBACK_DAYS, "min_days": MIN_DAYS, "min_rows": MIN_ROWS, "day_balanced": True, "requires_prior_resolved_labels": True},
        "admission": "adjusted BCF mapped EV >= 50 bps; B1 remains exact frozen dual-gate control",
        "portfolio": "same controlled global long-only 7x/10%-slot, 2-new, 8-concurrent, 80%-wallet; invalid outcomes excluded before capacity",
        "variants": [variant.__dict__ for variant in VARIANTS],
        "sources": {
            "current": {"path": str(HISTORY_CURRENT), "sha256": _sha256(HISTORY_CURRENT)},
            "bcf": {"path": str(HISTORY_BCF), "sha256": _sha256(HISTORY_BCF)},
            "august_prepared": {"path": str(prepared), "manifest_sha256": _sha256(prepared / "run_manifest.json")},
            "august_labels": {"path": str(labels_path), "sha256": _sha256(labels_path)},
            "august_frozen_map": {"path": str(frozen_map), "sha256": _sha256(frozen_map)},
        },
        "status": "complete",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "out": str(out), "summary": summary.to_dict(orient="records")}, default=str))


if __name__ == "__main__":
    main()
