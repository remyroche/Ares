#!/usr/bin/env python3
"""Frozen short P0→O250/H6→C3/C59→K0 scorecard and isolated MC1 test.

The upstream stack is never retrained.  This produces the requested frozen
scorecards, component-neutralization counterfactuals, threshold sensitivity,
and an explicitly non-canonical MC1-style absolute-EV mapper using *only*
target-free fields derived from P0, O, C, and K0.  The MC1 test is strictly
prequential and has fixed shallow geometry; it is not an HPO or promotion run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round4_k0_refinement as r4  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_frozen_diagnostics_mc1_v1"
C59 = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair_20260822_v1/C59_outer_oof_predictions.parquet"
ROUND1 = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round1_20260821_v1"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_frozen_diagnostics_mc1_20260822_v1"
MU1 = ("isotonic", 0)
MU0 = ("anchor5", 500)
ADMISSION = 75.0
MC1_MIN_ROWS = 1_000
MC1_MIN_MONTHS = 3
MC1_PARAMS = {
    "max_depth": 2,
    "max_iter": 80,
    "learning_rate": .04,
    "l2_regularization": 20.0,
    "min_samples_leaf": 100,
    "random_state": 1729,
}
MC0_FIELDS = ("K0_expected_policy_net_bps",)
MC1_FIELDS = (
    "K0_expected_policy_net_bps",
    "opportunity_probability_round4",
    "conversion_score",
    "k0_mu1_round4_bps",
    "k0_mu0_round4_bps",
    "p0_component_bps",
    "o_component_bps",
    "prequential_base_anchor_bps",
    "prequential_base_rank42",
    "prequential_base_score",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in ([path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())):
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _finite(values: pd.Series | np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=fill, posinf=fill, neginf=fill)


def _valid(frame: pd.DataFrame) -> pd.Series:
    return r4._valid(frame)


def _month_count(frame: pd.DataFrame) -> int:
    return int(frame["held_month"].nunique())


def _table(frame: pd.DataFrame) -> str:
    cols = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def _metrics(frame: pd.DataFrame, score: str, threshold: float, arm: str, era: str) -> dict[str, float | int | str]:
    selected = frame.loc[_finite(frame[score]) >= threshold].copy()
    known = selected.loc[_valid(selected)].copy()
    net = _finite(known["policy_net_bps"])
    stop = pd.to_numeric(known["policy_stop_out"], errors="coerce").fillna(False).astype(bool).to_numpy()
    ordered = np.sort(net)
    cvar = float(ordered[:max(1, int(math.ceil(len(ordered) * .10)))].mean()) if len(ordered) else float("nan")
    return {
        "arm": arm, "era": era, "threshold_bps": threshold,
        "scored": int(len(frame)), "selected": int(len(selected)), "known": int(len(known)),
        "outcome_coverage": float(len(known) / len(selected)) if len(selected) else float("nan"),
        "net_bps_per_trade": float(net.mean()) if len(net) else float("nan"),
        "total_net_bps": float(net.sum()) if len(net) else 0.0,
        "hit_rate": float((net > 0).mean()) if len(net) else float("nan"),
        "cvar10_bps": cvar,
        "stop_rate": float(stop.mean()) if len(stop) else float("nan"),
        "fraction_lt_neg200": float((net < -200).mean()) if len(net) else float("nan"),
        "fraction_lt_neg400": float((net < -400).mean()) if len(net) else float("nan"),
        "fraction_lt_neg600": float((net < -600).mean()) if len(net) else float("nan"),
    }


def _scorecard(frame: pd.DataFrame, *, period: str, field: str, bins: Iterable[float], labels: Iterable[str]) -> pd.DataFrame:
    local = frame.loc[frame["held_month"].str.startswith(period) & _valid(frame) & _finite(frame["K0_expected_policy_net_bps"]).__ge__(ADMISSION)].copy()
    local["band"] = pd.cut(_finite(local[field]), list(bins), labels=list(labels), right=False, include_lowest=True)
    rows = []
    for band, part in local.groupby("band", observed=False):
        net = _finite(part["policy_net_bps"])
        ordered = np.sort(net)
        stop = pd.to_numeric(part["policy_stop_out"], errors="coerce").fillna(False).astype(bool).to_numpy()
        rows.append({
            "period": period, "field": field, "band": str(band), "trades": int(len(part)),
            "realized_net_bps": float(net.mean()) if len(net) else float("nan"),
            "hit_rate": float((net > 0).mean()) if len(net) else float("nan"),
            "cvar10_bps": float(ordered[:max(1, int(math.ceil(len(ordered) * .10)))].mean()) if len(ordered) else float("nan"),
            "stop_rate": float(stop.mean()) if len(stop) else float("nan"),
        })
    return pd.DataFrame(rows)


def _build_components() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    ledger, hashes = r4._load_ledger(C59)
    frame, _, _, frame_hashes = r3._load_frame()
    p0 = frame.loc[:, [
        "candidate_id", "prequential_base_anchor_bps", "prequential_base_rank42",
        "prequential_base_score", "policy_stop_out",
    ]]
    if p0["candidate_id"].duplicated().any():
        raise AssertionError("P0 lineage has duplicate candidate IDs")
    ledger = ledger.merge(p0, on="candidate_id", how="left", validate="one_to_one", suffixes=("", "_frame"))
    for field in ("prequential_base_anchor_bps", "prequential_base_rank42", "prequential_base_score", "policy_stop_out"):
        alternate = f"{field}_frame"
        if alternate in ledger:
            ledger[field] = ledger[field].where(ledger[field].notna(), ledger[alternate])
            ledger = ledger.drop(columns=[alternate])
    if ledger[["prequential_base_anchor_bps", "prequential_base_rank42", "prequential_base_score"]].isna().any().any():
        raise AssertionError("component ledger lacks a frozen P0-derived input")
    rows, audit = [], []
    for month, held in ledger.groupby("held_month", sort=True):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        history = ledger.loc[
            ledger["__decision_ts__"].lt(start)
            & ledger["__label_available_at__"].lt(start)
            & _valid(ledger)
        ].copy()
        record = {"held_month": month, "history_rows": int(len(history)), "history_months": _month_count(history), "history_opportunity_positive": int(r1._event(history, r3.SPEC).sum())}
        if len(history) < r4.MIN_HISTORY_ROWS or _month_count(history) < r4.MIN_HISTORY_MONTHS or int(r1._event(history, r3.SPEC).sum()) < r1.MIN_C_POSITIVES:
            record["status"] = "skipped_insufficient_prequential_support"
            audit.append(record)
            continue
        if not history["__label_available_at__"].lt(start).all():
            raise AssertionError("component history contains unresolved held-month outcomes")
        bundle = r4._fit_map(history, *MU1, *MU0, ("absolute", ADMISSION))
        out = r4._apply_map(bundle, held)
        p = out["opportunity_probability_round4"].to_numpy(float)
        mu1 = out["k0_mu1_round4_bps"].to_numpy(float)
        mu0 = out["k0_mu0_round4_bps"].to_numpy(float)
        y = np.clip(_finite(history["policy_net_bps"]), -r1.POLICY_CLIP_BPS, r1.POLICY_CLIP_BPS)
        event = r1._event(history, r3.SPEC).astype(bool)
        prior_p = float(event.mean())
        global_mu1 = float(y[event].mean())
        global_mu0 = r4._fit_mu0(history, "global", 500).predict(out["prequential_base_anchor_bps"].to_numpy(float))
        out["o_component_bps"] = (p * mu1).astype(np.float32)
        out["p0_component_bps"] = ((1.0 - p) * mu0).astype(np.float32)
        out["K0_no_conversion_bps"] = (p * global_mu1 + (1.0 - p) * mu0).astype(np.float32)
        out["K0_no_opportunity_bps"] = (prior_p * mu1 + (1.0 - prior_p) * mu0).astype(np.float32)
        out["K0_global_mu0_bps"] = (p * mu1 + (1.0 - p) * global_mu0).astype(np.float32)
        out["held_month"] = month
        rows.append(out)
        record.update({"status": "complete", "prior_opportunity_probability": prior_p, "global_mu1_bps": global_mu1, "threshold_bps": ADMISSION})
        audit.append(record)
    if not rows:
        raise RuntimeError("no frozen component map had sufficient prequential support")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audit), {"source_c59_sha256": _sha256(C59), **hashes, **frame_hashes}


def _counterfactuals(components: pd.DataFrame) -> pd.DataFrame:
    arms = {
        "full_K0": "K0_expected_policy_net_bps",
        "no_conversion_information": "K0_no_conversion_bps",
        "no_opportunity_information": "K0_no_opportunity_bps",
        "global_mu0": "K0_global_mu0_bps",
    }
    rows = []
    for arm, field in arms.items():
        for era in ("2025", "2026", "all_supported"):
            local = components if era == "all_supported" else components.loc[components["held_month"].str.startswith(era)]
            rows.append(_metrics(local, field, ADMISSION, arm, era))
    return pd.DataFrame(rows)


def _threshold_curve(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in (25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0):
        for era in ("2025", "2026", "all_supported"):
            local = components if era == "all_supported" else components.loc[components["held_month"].str.startswith(era)]
            rows.append(_metrics(local, "K0_expected_policy_net_bps", threshold, "frozen_threshold_sensitivity", era))
    return pd.DataFrame(rows)


def _matrix(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in ("2025", "2026"):
        local = components.loc[components["held_month"].str.startswith(period) & _valid(components) & (_finite(components["K0_expected_policy_net_bps"]) >= ADMISSION)].copy()
        if local.empty:
            continue
        local["p_o_band"] = pd.cut(local["opportunity_probability_round4"], [0., .2, .4, .6, .8, 1.000001], labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"], right=False, include_lowest=True)
        local["c_quintile"] = pd.qcut(local["conversion_score"].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        for (pband, cband), part in local.groupby(["p_o_band", "c_quintile"], observed=False):
            net = _finite(part["policy_net_bps"])
            rows.append({"period": period, "p_o_band": str(pband), "c_quintile": str(cband), "trades": int(len(part)), "realized_net_bps": float(net.mean()) if len(net) else float("nan"), "hit_rate": float((net > 0).mean()) if len(net) else float("nan")})
    return pd.DataFrame(rows)


def _mc_matrix(frame: pd.DataFrame, fields: tuple[str, ...], medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = values.median(axis=0, skipna=True).fillna(0.0)
    return values.fillna(medians).fillna(0.0).to_numpy(dtype=np.float32), medians


def _mc1(components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs, audit = [], []
    for month, held in components.groupby("held_month", sort=True):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        history = components.loc[
            components["__decision_ts__"].lt(start)
            & components["__label_available_at__"].lt(start)
            & _valid(components)
        ].copy()
        row = {"held_month": month, "history_rows": int(len(history)), "history_months": _month_count(history)}
        if len(history) < MC1_MIN_ROWS or _month_count(history) < MC1_MIN_MONTHS:
            row["status"] = "skipped_insufficient_prequential_support"
            audit.append(row)
            continue
        if not history["__label_available_at__"].lt(start).all():
            raise AssertionError("MC1 history has unresolved label")
        out = held.copy()
        y = np.clip(_finite(history["policy_net_bps"]), -r1.POLICY_CLIP_BPS, r1.POLICY_CLIP_BPS)
        for arm, fields in (("MC0_k0_only", MC0_FIELDS), ("MC1_p0_o_c_k0", MC1_FIELDS)):
            x_fit, medians = _mc_matrix(history, fields)
            x_held, _ = _mc_matrix(held, fields, medians)
            model = HistGradientBoostingRegressor(**MC1_PARAMS)
            model.fit(x_fit, y)
            out[f"{arm}_expected_policy_net_bps"] = model.predict(x_held).astype(np.float32)
        outputs.append(out)
        row["status"] = "complete"
        audit.append(row)
    if not outputs:
        raise RuntimeError("MC1 has no strict-prequential support")
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audit)


def _mc1_metrics(prediction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, field in (("K0_native_matched", "K0_expected_policy_net_bps"), ("MC0_k0_only", "MC0_k0_only_expected_policy_net_bps"), ("MC1_p0_o_c_k0", "MC1_p0_o_c_k0_expected_policy_net_bps")):
        for era in ("2025", "2026", "all_supported"):
            local = prediction if era == "all_supported" else prediction.loc[prediction["held_month"].str.startswith(era)]
            rows.append(_metrics(local, field, ADMISSION, arm, era))
    return pd.DataFrame(rows)


def _nearby_o() -> pd.DataFrame:
    rows = pd.read_parquet(ROUND1 / "round1_ranking.parquet")
    return rows.loc[rows["arm"].isin(("O200_H6", "O250_H6", "O250_H12", "O300_H12"))].copy()


def _execution_margin(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for friction in (0., 25., 50., 75., 100., 125., 150.):
        local = components.loc[(_finite(components["K0_expected_policy_net_bps"]) >= ADMISSION) & _valid(components)].copy()
        adjusted = _finite(local["policy_net_bps"]) - friction
        rows.append({"additional_execution_friction_bps": friction, "trades": int(len(local)), "net_bps_per_trade": float(adjusted.mean()), "positive_fraction": float((adjusted > 0).mean()), "cvar10_bps": float(np.sort(adjusted)[:max(1, int(math.ceil(len(adjusted) * .10)))].mean())})
    return pd.DataFrame(rows)


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    components, map_audit, hashes = _build_components()
    scorecards = pd.concat([
        _scorecard(components, period=period, field="K0_expected_policy_net_bps", bins=[75., 100., 150., 200., 300., np.inf], labels=["75-100", "100-150", "150-200", "200-300", ">300"])
        for period in ("2025", "2026")
    ], ignore_index=True)
    po_scorecards = pd.concat([
        _scorecard(components, period=period, field="opportunity_probability_round4", bins=[0., .2, .4, .6, .8, 1.000001], labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
        for period in ("2025", "2026")
    ], ignore_index=True)
    matrix = _matrix(components)
    counterfactual = _counterfactuals(components)
    thresholds = _threshold_curve(components)
    mc1_prediction, mc1_audit = _mc1(components)
    mc1_metrics = _mc1_metrics(mc1_prediction)
    nearby = _nearby_o()
    execution = _execution_margin(components)
    forward_gates = {
        "status": "predeclared monitoring proposal; not an automated live promotion rule",
        "minimum_resolved_trades": 100,
        "economic": {"net_ev_per_trade_bps_gt": 100.0, "preferred_gt": 125.0},
        "portability": {"worst_month_bps_per_trade_gt": -25.0},
        "tail": {"monitor": ["cvar10_bps", "fraction_lt_neg200", "fraction_lt_neg400", "fraction_lt_neg600"], "research_reference_cvar10_bps": -343.608653},
        "calibration": "K0 expected-EV bands must remain directionally ordered; no single asset/month may dominate total net bps.",
        "execution": "historical policy net and actual executable net must remain separate; report realized entry/exit friction before any promotion.",
    }
    selected = components.loc[_finite(components["K0_expected_policy_net_bps"]) >= ADMISSION]
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short",
        "scope": "frozen O/C/K0 scorecards and counterfactual diagnostics plus an isolated non-canonical MC1 mapper assessment",
        "frozen_stack": {"P0": "F90 target-free", "O": "O250_H6 stable45/uniform/Platt", "C": "C3 normalized regret C59/uniform", "K0": "isotonic mu1 + P0-anchor5 mu0 k500", "admission": "absolute expected policy net >=75 bps"},
        "mc1": {"status": "isolated challenger only; does not change frozen stack", "model": {"type": "HistGradientBoostingRegressor", **MC1_PARAMS}, "inputs": list(MC1_FIELDS), "target": "clipped exact policy net bps", "fit": "strict prior-resolved component-ledger rows", "forbidden": ["held outcomes", "correctness/global shift", "HPO", "O/C/K0 retraining", "live/canonical mutation"]},
        "forward_gates": forward_gates,
        "causality": {"components": "each held map uses only earlier outer-OOS rows with label_available_at < held month start", "mc1": "each held MC1 fit uses only prior resolved component rows", "invalid": "retained in score ledger; excluded from fitting and realized economics"},
        "sources": hashes,
    }
    out.mkdir(parents=True)
    components.to_parquet(out / "frozen_component_ledger.parquet", index=False, compression="zstd")
    map_audit.to_parquet(out / "frozen_component_map_audit.parquet", index=False, compression="zstd")
    scorecards.to_parquet(out / "k0_ev_band_scorecard.parquet", index=False, compression="zstd")
    po_scorecards.to_parquet(out / "opportunity_probability_scorecard.parquet", index=False, compression="zstd")
    matrix.to_parquet(out / "opportunity_conversion_2d_matrix.parquet", index=False, compression="zstd")
    counterfactual.to_parquet(out / "component_neutralization_counterfactual.parquet", index=False, compression="zstd")
    thresholds.to_parquet(out / "frozen_threshold_sensitivity.parquet", index=False, compression="zstd")
    mc1_prediction.to_parquet(out / "mc1_outer_oof_predictions.parquet", index=False, compression="zstd")
    mc1_audit.to_parquet(out / "mc1_fold_audit.parquet", index=False, compression="zstd")
    mc1_metrics.to_parquet(out / "mc1_metrics.parquet", index=False, compression="zstd")
    nearby.to_parquet(out / "nearby_o_definition_diagnostic.parquet", index=False, compression="zstd")
    execution.to_parquet(out / "execution_margin_sensitivity.parquet", index=False, compression="zstd")
    (out / "forward_validation_gates.json").write_text(json.dumps(forward_gates, indent=2) + "\n")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Frozen short P0 → O250/H6 → C3/C59 → K0 diagnostics", "",
        "The stack is frozen. MC1 below is a separately-labelled research mapper and does not alter the frozen stack.", "",
        "## K0 EV bands", "", _table(scorecards), "",
        "## Component-neutralization counterfactual", "", _table(counterfactual), "",
        "## Threshold sensitivity (diagnostic only)", "", _table(thresholds), "",
        "## Isolated MC1 mapper", "", _table(mc1_metrics), "",
        "## Nearby historical O definitions (not reselection)", "", _table(nearby), "",
        "## Execution-margin sensitivity", "", _table(execution), "",
        "## Forward gates", "", "```json", json.dumps(forward_gates, indent=2), "```", "",
    ]
    (out / "FROZEN_P0_OC_K0_SCORECARD_MC1_REPORT.md").write_text("\n".join(report))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(run(args.out))


if __name__ == "__main__":
    main()
