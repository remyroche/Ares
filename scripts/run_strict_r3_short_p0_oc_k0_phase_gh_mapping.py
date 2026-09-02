#!/usr/bin/env python3
"""Strict short K0 Phase G/H: residual mu1 and P0-anchor mu0 refinements.

This is deliberately a mapping-only funnel above the frozen O45/C59 outer-OOF
ledger.  It preserves the analytic form

    K0 = p(O) * mu1(C) + (1 - p(O)) * mu0(P0 anchor)

and the absolute +75 bps admission floor.  Each held month is mapped only
from prior outer-OOF rows whose policy labels were resolved before that month.
No model is fitted on held labels and no extra learned EV/risk layer is added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_phase_ef_conversion as ef  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_phase_gh_mapping_v1"
SOURCE = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_ef_conversion_202408_202607_20260822_v1/phase_ef_outer_oof_predictions.parquet"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_gh_mapping_202408_202607_20260822_v1"
CONTROL_ARM = "G0_H0_mu1_isotonic_mu0_anchor5_k500"
ADMISSION_BPS = 75.0
POLICY_CLIP_BPS = float(r1.POLICY_CLIP_BPS)
P0_ANCHOR = "prequential_base_anchor_bps"
MIN_HISTORY_ROWS = 500
MIN_HISTORY_MONTHS = 3
MIN_EVENT_ROWS = int(r1.MIN_C_POSITIVES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: pd.Series | np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=float), nan=fill, posinf=fill, neginf=fill)


def _valid(frame: pd.DataFrame) -> pd.Series:
    return r1._valid_label(frame) & pd.to_numeric(frame["policy_net_bps"], errors="coerce").notna()


def _event(frame: pd.DataFrame) -> np.ndarray:
    return r1._event(frame, r3.SPEC).astype(bool)


def _month_count(frame: pd.DataFrame) -> int:
    return int(frame["held_month"].nunique())


def _quantile_edges(values: np.ndarray, bins: int) -> np.ndarray | None:
    if len(values) < bins or not np.isfinite(values).any():
        return None
    edges = np.unique(np.quantile(values[np.isfinite(values)], np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


@dataclass
class Mu0Map:
    kind: str
    k: int
    global_mean: float
    edges: np.ndarray
    values: np.ndarray
    support: np.ndarray
    isotonic: IsotonicRegression | None = None

    def predict(self, anchor: np.ndarray) -> np.ndarray:
        x = _finite(anchor)
        if self.kind == "global":
            return np.full(len(x), float(self.values[0]), dtype=float)
        idx = np.searchsorted(self.edges, x, side="right") - 1
        idx = np.clip(idx, 0, len(self.support) - 1)
        if self.kind.startswith("anchor"):
            return self.values[idx]
        if self.isotonic is None:
            raise AssertionError("isotonic mu0 state missing estimator")
        raw = np.asarray(self.isotonic.predict(x), dtype=float)
        weight = self.support[idx] / (self.support[idx] + float(self.k))
        return weight * raw + (1.0 - weight) * self.global_mean


@dataclass
class Mu1ResidualMap:
    bins: int
    k: int
    base: IsotonicRegression
    residual_edges: np.ndarray | None
    residual_values: np.ndarray

    def predict(self, conversion: np.ndarray, probability: np.ndarray) -> np.ndarray:
        base = np.asarray(self.base.predict(_finite(conversion)), dtype=float)
        if self.bins == 0 or self.residual_edges is None:
            return base
        idx = np.searchsorted(self.residual_edges, np.asarray(probability, dtype=float), side="right") - 1
        idx = np.clip(idx, 0, len(self.residual_values) - 1)
        return np.clip(base + self.residual_values[idx], -POLICY_CLIP_BPS, POLICY_CLIP_BPS)


@dataclass
class K0Bundle:
    opportunity: IsotonicRegression
    mu1: Mu1ResidualMap
    mu0: Mu0Map


@dataclass(frozen=True)
class GSpec:
    name: str
    bins: int
    k: int


@dataclass(frozen=True)
class HSpec:
    name: str
    kind: str
    bins: int
    k: int


G_SPECS = (
    GSpec("G0_mu1_isotonic", 0, 0),
    *(GSpec(f"G1_pO_quintile_k{k}", 5, k) for k in (250, 500, 1000)),
    *(GSpec(f"G2_pO_tercile_k{k}", 3, k) for k in (250, 500, 1000)),
)
H0 = HSpec("H0_mu0_anchor5_k500", "anchor", 5, 500)
H_SPECS = (
    H0,
    HSpec("H1_mu0_anchor5_k250", "anchor", 5, 250),
    HSpec("H1_mu0_anchor5_k1000", "anchor", 5, 1000),
    HSpec("H1_mu0_anchor5_k2000", "anchor", 5, 2000),
    HSpec("H2_mu0_anchor10_k500", "anchor", 10, 500),
    HSpec("H2_mu0_anchor10_k1000", "anchor", 10, 1000),
    HSpec("H2_mu0_anchor10_k2000", "anchor", 10, 2000),
    HSpec("H3_mu0_isotonic_support_k500", "isotonic", 10, 500),
)


def _load(source: Path, source_arm: str = "E0_C3_multiclass_control") -> tuple[pd.DataFrame, dict[str, str]]:
    source = Path(source)
    raw = pd.read_parquet(source)
    arm_column = "feature_block_arm" if "feature_block_arm" in raw.columns else "arm"
    ledger = raw.loc[raw[arm_column].eq(source_arm)].copy()
    if ledger.empty:
        raise ValueError(f"source arm missing from score ledger: {arm_column}={source_arm}")
    if ledger.candidate_id.duplicated().any():
        raise AssertionError("E0 C3 outer OOF candidate identity is non-unique")
    frame, _fields, hashes = ef._load()
    anchors = frame.loc[:, ["candidate_id", P0_ANCHOR]].copy()
    if anchors.candidate_id.duplicated().any():
        raise AssertionError("P0 anchor source identity is non-unique")
    ledger = ledger.merge(anchors, on="candidate_id", how="left", validate="one_to_one")
    if ledger[P0_ANCHOR].isna().any():
        raise AssertionError("E0 C3 ledger lacks a causal P0 anchor")
    ledger["__decision_ts__"] = pd.to_datetime(ledger["__decision_ts__"], utc=True, errors="raise")
    ledger["__label_available_at__"] = pd.to_datetime(ledger["__label_available_at__"], utc=True, errors="raise")
    return ledger.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), {
        "phase_ef_outer_oof_sha256": _sha256(source),
        **hashes,
    }


def _fit_mu0(history: pd.DataFrame, spec: HSpec) -> Mu0Map:
    y = np.clip(_finite(history["policy_net_bps"]), -POLICY_CLIP_BPS, POLICY_CLIP_BPS)
    event = _event(history)
    negative = ~event
    if not negative.any():
        raise ValueError("K0 history has no O-negative rows")
    global_mean = float(np.mean(y))
    if spec.kind == "global":
        local = y[negative]
        value = float((local.sum() + spec.k * global_mean) / (len(local) + spec.k))
        return Mu0Map("global", spec.k, global_mean, np.array([-np.inf, np.inf]), np.array([value]), np.array([len(local)], dtype=float))
    x = _finite(history[P0_ANCHOR])
    edges = _quantile_edges(x[negative], spec.bins)
    if edges is None:
        return _fit_mu0(history, HSpec("fallback_global", "global", 1, spec.k))
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    support = np.asarray([np.sum(negative & (idx == bucket)) for bucket in range(len(edges) - 1)], dtype=float)
    if spec.kind == "anchor":
        values = np.asarray([
            float((y[negative & (idx == bucket)].sum() + spec.k * global_mean) / (support[bucket] + spec.k))
            for bucket in range(len(support))
        ], dtype=float)
        return Mu0Map("anchor", spec.k, global_mean, edges, values, support)
    fitted, _ = r1._fit_isotonic(x[negative], y[negative], -POLICY_CLIP_BPS, POLICY_CLIP_BPS)
    return Mu0Map("isotonic", spec.k, global_mean, edges, np.full(len(support), np.nan), support, fitted)


def _fit_mu1(history: pd.DataFrame, gspec: GSpec, probability: IsotonicRegression) -> Mu1ResidualMap:
    event = _event(history)
    source = history.loc[event].copy()
    if len(source) < MIN_EVENT_ROWS:
        raise ValueError("insufficient O-positive history for mu1")
    x = _finite(source["conversion_score"])
    y = np.clip(_finite(source["policy_net_bps"]), -POLICY_CLIP_BPS, POLICY_CLIP_BPS)
    base, _ = r1._fit_isotonic(x, y, -POLICY_CLIP_BPS, POLICY_CLIP_BPS)
    if gspec.bins == 0:
        return Mu1ResidualMap(0, 0, base, None, np.zeros(1, dtype=float))
    probability_source = np.asarray(probability.predict(_finite(source["opportunity_raw_score"])), dtype=float)
    edges = _quantile_edges(probability_source, gspec.bins)
    if edges is None:
        return Mu1ResidualMap(0, 0, base, None, np.zeros(1, dtype=float))
    idx = np.searchsorted(edges, probability_source, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    residual = y - np.asarray(base.predict(x), dtype=float)
    global_residual = float(np.mean(residual))
    values = np.asarray([
        float((residual[idx == bucket].sum() + gspec.k * global_residual) / (np.sum(idx == bucket) + gspec.k))
        for bucket in range(len(edges) - 1)
    ], dtype=float)
    return Mu1ResidualMap(gspec.bins, gspec.k, base, edges, values)


def _fit_bundle(history: pd.DataFrame, gspec: GSpec, hspec: HSpec) -> K0Bundle:
    event = _event(history)
    opportunity, _ = r1._fit_isotonic(_finite(history["opportunity_raw_score"]), event.astype(float), 0.0, 1.0)
    return K0Bundle(opportunity, _fit_mu1(history, gspec, opportunity), _fit_mu0(history, hspec))


def _apply(bundle: K0Bundle, held: pd.DataFrame) -> pd.DataFrame:
    out = held.copy()
    probability = np.asarray(bundle.opportunity.predict(_finite(out["opportunity_raw_score"])), dtype=float)
    mu1 = bundle.mu1.predict(_finite(out["conversion_score"]), probability)
    mu0 = bundle.mu0.predict(_finite(out[P0_ANCHOR]))
    expected = probability * mu1 + (1.0 - probability) * mu0
    out["opportunity_probability"] = probability.astype(np.float32)
    out["k0_mu1_bps"] = mu1.astype(np.float32)
    out["k0_mu0_bps"] = mu0.astype(np.float32)
    out["K0_expected_policy_net_bps"] = expected.astype(np.float32)
    out["K0_admitted"] = expected >= ADMISSION_BPS
    return out


def _replay(ledger: pd.DataFrame, gspec: GSpec, hspec: HSpec, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for month, held in ledger.groupby("held_month", sort=True):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        history = ledger.loc[
            ledger["__decision_ts__"].lt(start)
            & ledger["__label_available_at__"].lt(start)
            & _valid(ledger)
        ].copy()
        event_rows = int(_event(history).sum()) if len(history) else 0
        record: dict[str, Any] = {
            "arm": arm, "held_month": month, "history_rows": int(len(history)),
            "history_months": _month_count(history), "history_event_rows": event_rows,
        }
        if len(history) < MIN_HISTORY_ROWS or _month_count(history) < MIN_HISTORY_MONTHS or event_rows < MIN_EVENT_ROWS:
            record.update({"status": "skipped_insufficient_prequential_mapping_support"})
            audit.append(record)
            continue
        if not history["__label_available_at__"].lt(start).all():
            raise AssertionError("mapping history contains unresolved label")
        bundle = _fit_bundle(history, gspec, hspec)
        scored = _apply(bundle, held)
        scored["arm"] = arm
        rows.append(scored)
        record.update({"status": "complete", "admitted": int(scored["K0_admitted"].sum())})
        audit.append(record)
    if not rows:
        raise RuntimeError("no held month has strict prequential K0 support")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audit)


def _cvar(values: np.ndarray, fraction: float = .10) -> float:
    ordered = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    return float(ordered[: max(1, int(np.ceil(len(ordered) * fraction)))].mean()) if len(ordered) else float("nan")


def _metrics(prediction: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    records: list[dict[str, Any]] = []
    for (arm, month), part in prediction.groupby(["arm", "held_month"], sort=True):
        selected = part.loc[part["K0_admitted"]].copy()
        known = selected.loc[_valid(selected)].copy()
        values = _finite(known["policy_net_bps"])
        valid_all = part.loc[_valid(part)]
        expected_ic = float(pd.Series(_finite(valid_all["K0_expected_policy_net_bps"])).corr(pd.Series(_finite(valid_all["policy_net_bps"])), method="spearman")) if len(valid_all) >= 5 else float("nan")
        records.append({
            "arm": arm, "held_month": month, "scored": len(part), "admitted": len(selected), "known_admitted": len(known),
            "coverage": float(len(known) / len(selected)) if len(selected) else float("nan"),
            "net_bps_per_trade": float(values.mean()) if len(values) else float("nan"),
            "total_net_bps": float(values.sum()), "cvar10_bps": _cvar(values),
            "p_net_lt_neg200": float((values < -200.0).mean()) if len(values) else float("nan"),
            "p_net_lt_neg400": float((values < -400.0).mean()) if len(values) else float("nan"),
            "positive_fraction": float((values > 0.0).mean()) if len(values) else float("nan"),
            "k0_net_spearman": expected_ic,
        })
    monthly = pd.DataFrame(records)
    eras: list[dict[str, Any]] = []
    for (arm, era), group in monthly.assign(era=monthly.held_month.str[:4]).groupby(["arm", "era"], sort=True):
        weights = np.maximum(group.known_admitted.to_numpy(float), 1.0)
        row: dict[str, Any] = {"arm": arm, "era": era, "months": len(group), "admitted": int(group.admitted.sum()), "known_admitted": int(group.known_admitted.sum()), "total_net_bps": float(group.total_net_bps.sum()), "positive_months": int((group.net_bps_per_trade > 0.0).sum()), "worst_month_net_bps": float(group.net_bps_per_trade.min())}
        for column in ("coverage", "net_bps_per_trade", "cvar10_bps", "p_net_lt_neg200", "p_net_lt_neg400", "positive_fraction", "k0_net_spearman"):
            value = group[column].to_numpy(float)
            keep = np.isfinite(value)
            row[column] = float(np.average(value[keep], weights=weights[keep])) if keep.any() else float("nan")
        eras.append(row)
    era = pd.DataFrame(eras)
    return monthly, era, {}


def _summary(era: pd.DataFrame, monthly: pd.DataFrame, control: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, group in era.loc[era.era.isin(("2025", "2026"))].groupby("arm", sort=True):
        by = group.set_index("era")
        if not {"2025", "2026"}.issubset(by.index):
            continue
        weights = np.maximum(group.known_admitted.to_numpy(float), 1.0)
        months = monthly.loc[(monthly.arm == arm) & monthly.held_month.str[:4].isin(("2025", "2026"))]
        rows.append({
            "arm": arm, "net_2025": float(by.loc["2025", "net_bps_per_trade"]), "net_2026": float(by.loc["2026", "net_bps_per_trade"]),
            "mean_net_bps": float(np.average(group.net_bps_per_trade.to_numpy(float), weights=weights)),
            "total_net_bps": float(group.total_net_bps.sum()), "admitted": int(group.admitted.sum()), "known_admitted": int(group.known_admitted.sum()),
            "worst_month_net_bps": float(months.net_bps_per_trade.min()),
            "cvar10_bps": float(np.average(group.cvar10_bps.to_numpy(float), weights=weights)),
            "p_net_lt_neg200": float(np.average(group.p_net_lt_neg200.to_numpy(float), weights=weights)),
            "k0_net_spearman": float(np.average(group.k0_net_spearman.to_numpy(float), weights=weights)),
        })
    result = pd.DataFrame(rows)
    if control and control in set(result.arm):
        base = result.loc[result.arm.eq(control)].iloc[0]
        for column in ("net_2025", "net_2026", "mean_net_bps", "total_net_bps", "worst_month_net_bps", "cvar10_bps"):
            result[f"delta_{column}"] = result[column] - float(base[column])
        result["participation_vs_control"] = result.known_admitted / max(float(base.known_admitted), 1.0)
        result["passes_gate"] = (
            result.net_2025.ge(float(base.net_2025) - 10.0)
            & result.net_2026.ge(float(base.net_2026) - 10.0)
            & result.cvar10_bps.ge(float(base.cvar10_bps) - 25.0)
            & result.participation_vs_control.ge(.80)
        )
    return result.sort_values(["passes_gate", "mean_net_bps", "worst_month_net_bps", "cvar10_bps"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No strict-OOF output in this era._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        columns = [str(value) for value in frame.columns]
        rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        rows.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
        return "\n".join(rows)


def _choose(summary: pd.DataFrame, control: str) -> str:
    eligible = summary.loc[summary.passes_gate]
    return str(eligible.iloc[0].arm) if len(eligible) else control


def run(out: Path, source: Path = SOURCE, source_arm: str = "E0_C3_multiclass_control") -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    ledger, hashes = _load(source, source_arm)

    # Stage G: keep the specified H0 P0-anchor fallback fixed while testing the
    # one-dimensional p(O) residual correction to mu1.
    stage_g: list[tuple[str, GSpec, HSpec]] = []
    for gspec in G_SPECS:
        arm = CONTROL_ARM if gspec.bins == 0 else f"{gspec.name}__{H0.name}"
        stage_g.append((arm, gspec, H0))
    g_predictions: list[pd.DataFrame] = []
    g_audits: list[pd.DataFrame] = []
    for arm, gspec, hspec in stage_g:
        prediction, audit = _replay(ledger, gspec, hspec, arm)
        g_predictions.append(prediction); g_audits.append(audit)
    g_monthly, g_era, _ = _metrics(pd.concat(g_predictions, ignore_index=True))
    g_summary = _summary(g_era, g_monthly, CONTROL_ARM)
    winner_g = _choose(g_summary, CONTROL_ARM)
    chosen_g = next(g for arm, g, _ in stage_g if arm == winner_g)

    # Stage H: hold the selected mu1 formulation fixed and test only the
    # declared P0-anchor fallback variants.
    stage_h: list[tuple[str, GSpec, HSpec]] = []
    for hspec in H_SPECS:
        arm = f"{winner_g}__{hspec.name}"
        stage_h.append((arm, chosen_g, hspec))
    h_predictions: list[pd.DataFrame] = []
    h_audits: list[pd.DataFrame] = []
    for arm, gspec, hspec in stage_h:
        prediction, audit = _replay(ledger, gspec, hspec, arm)
        h_predictions.append(prediction); h_audits.append(audit)
    h_monthly, h_era, _ = _metrics(pd.concat(h_predictions, ignore_index=True))
    h_control = f"{winner_g}__{H0.name}"
    h_summary = _summary(h_era, h_monthly, h_control)
    winner_h = _choose(h_summary, h_control)

    out.mkdir(parents=True)
    pd.concat(g_predictions, ignore_index=True).to_parquet(out / "phase_g_outer_oof_predictions.parquet", index=False, compression="zstd")
    pd.concat(h_predictions, ignore_index=True).to_parquet(out / "phase_h_outer_oof_predictions.parquet", index=False, compression="zstd")
    pd.concat(g_audits, ignore_index=True).to_parquet(out / "phase_g_fold_audit.parquet", index=False, compression="zstd")
    pd.concat(h_audits, ignore_index=True).to_parquet(out / "phase_h_fold_audit.parquet", index=False, compression="zstd")
    g_monthly.to_parquet(out / "phase_g_monthly_metrics.parquet", index=False, compression="zstd")
    g_era.to_parquet(out / "phase_g_era_metrics.parquet", index=False, compression="zstd")
    g_summary.to_parquet(out / "phase_g_summary.parquet", index=False, compression="zstd")
    h_monthly.to_parquet(out / "phase_h_monthly_metrics.parquet", index=False, compression="zstd")
    h_era.to_parquet(out / "phase_h_era_metrics.parquet", index=False, compression="zstd")
    h_summary.to_parquet(out / "phase_h_summary.parquet", index=False, compression="zstd")
    final = pd.concat(h_predictions, ignore_index=True).loc[lambda x: x.arm.eq(winner_h)].copy()
    final.to_parquet(out / "phase_gh_winner_outer_oof_predictions.parquet", index=False, compression="zstd")

    coverage = pd.DataFrame([{
        "era": "2024", "status": "not independently scored", "reason": "Frozen O45 starts in October 2024; C59/K0 requires three purged inner C-OOF slices and first has strict support in April 2025. 2024 remains upstream causal warm-up, not a pooled metric.",
    }, {
        "era": "2025", "status": "scored", "reason": "strict prequential outer OOF monthly mapping"}, {
        "era": "2026", "status": "scored", "reason": "strict prequential outer OOF monthly mapping"},
    ])
    coverage.to_parquet(out / "era_coverage_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short", "scope": "Phase G/H K0-only research; no canonical/live change",
        "architecture": "frozen P0/F90 -> frozen O45 -> frozen C59 -> analytic K0 p(O)*mu1(C)+(1-p(O))*mu0(P0 anchor)",
        "admission": {"K0_expected_policy_net_bps_gte": ADMISSION_BPS},
        "selection": {"stage_g_control": CONTROL_ARM, "stage_g_winner": winner_g, "stage_h_control": h_control, "stage_h_winner": winner_h, "gate": "retain >=80% participation; neither 2025 nor 2026 EV worse by >10 bps; CVaR10 no worse by >25 bps"},
        "causality": {"upstream_scores": "frozen monthly outer OOF", "mapping_fit": "only prior outer OOF rows whose labels resolved before held month", "mu1_residual": "one-dimensional p(O) bin correction on true O-positive rows", "mu0": "P0-anchor only", "held_outcomes": "never used in mapping fit", "invalid_rows": "scored but excluded from map fitting and outcome metrics"},
        "eras": {"2024": "causal warm-up only; no independently scored full O/C/K0 month", "2025": "strict OOF", "2026": "strict OOF"},
        "sources": {**hashes, "score_ledger": str(source), "score_ledger_arm": source_arm},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Short P0 → O45 → C59 → K0: Phase G/H mapping refinements", "",
        "Research-only strict-prequential mapping test. It corrects μ0 to the specified P0-anchor fallback; no O/C model, feature, policy, admission threshold, or live artifact changes.", "",
        "## Era coverage", "", _table(coverage), "",
        "## Phase G — μ1(C) plus shrunk p(O) residual", "", _table(g_summary), "",
        "## Phase H — P0-anchor μ0 mapping", "", _table(h_summary), "",
        "## Decision", "", f"- Phase G winner: `{winner_g}`.", f"- Phase H winner: `{winner_h}`.", "- Advancement remains research-only; it requires final full-stack comparison against frozen A0 before any promotion.", "",
        "```json", json.dumps(manifest, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_PHASE_GH_MAPPING_REPORT.md").write_text("\n".join(report))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=SOURCE, help="strict-prequential outer-OOF score ledger")
    parser.add_argument("--source-arm", default="E0_C3_multiclass_control", help="arm value; feature-block ledgers use feature_block_arm")
    args = parser.parse_args()
    print(run(args.out, args.source, args.source_arm))


if __name__ == "__main__":
    main()
