#!/usr/bin/env python3
"""Round 4: strictly-prequential K0 mapping, shrinkage, and admission study.

This is intentionally a *mapping-only* experiment above the frozen short
P0 -> O250/H6 -> C3/C60/uniform score ledger.  It does not fit another model
or alter O/C.  Every held month is calibrated from earlier outer-OOS scores
whose exact policy labels were resolved before that month began.

The sequential funnel is:
  A. mu1(C): isotonic versus 10/20-bin empirical-Bayes monotonic maps;
  B. mu0: global versus P0-anchor quintile maps, each shrunk to the global
     policy-net prior;
  C. causal admission thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round2 as r2  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_round4_k0_refinement_v1"
ROUND3C = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3_c_hpo_20260822_v1"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round4_k0_refinement_20260822_v1"
DEFAULT_SOURCE_PREDICTION = ROUND3C / "C60_uniform_control_outer_oof_predictions.parquet"
SPEC = r3.SPEC
MIN_HISTORY_ROWS = 500
MIN_HISTORY_MONTHS = 3
P0_ANCHOR = "prequential_base_anchor_bps"
MU1_CONFIGS = (("isotonic", 0), *tuple((kind, k) for kind in ("bins10", "bins20") for k in (50, 100, 250, 500, 1000)))
MU0_CONFIGS = tuple((kind, k) for kind in ("global", "anchor5") for k in (50, 100, 250, 500, 1000))
ADMISSIONS: tuple[tuple[str, float], ...] = (
    ("quantile", .70), ("quantile", .75), ("quantile", .80), ("quantile", .85), ("quantile", .90),
    ("absolute", 0.0), ("absolute", 25.0), ("absolute", 50.0), ("absolute", 75.0), ("absolute", 100.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in ([path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())):
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _valid(frame: pd.DataFrame) -> pd.Series:
    return r1._valid_label(frame) & pd.to_numeric(frame["policy_net_bps"], errors="coerce").notna()


def _event(frame: pd.DataFrame) -> np.ndarray:
    return r1._event(frame, SPEC).astype(bool)


def _months(frame: pd.DataFrame) -> int:
    return int(frame["held_month"].nunique())


def _finite(values: pd.Series | np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=fill, posinf=fill, neginf=fill)


@dataclass
class Mu1Map:
    kind: str
    k: int
    model: IsotonicRegression

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(_finite(values)), dtype=float)


@dataclass
class Mu0Map:
    kind: str
    k: int
    edges: np.ndarray
    values: np.ndarray

    def predict(self, anchor: np.ndarray) -> np.ndarray:
        if self.kind == "global":
            return np.full(len(anchor), float(self.values[0]), dtype=float)
        idx = np.searchsorted(self.edges, _finite(anchor), side="right") - 1
        idx = np.clip(idx, 0, len(self.values) - 1)
        return self.values[idx]


@dataclass
class K0Map:
    probability: r2.ProbabilityCalibrator
    mu1: Mu1Map
    mu0: Mu0Map
    threshold: float


def _fit_mu1(history: pd.DataFrame, kind: str, k: int) -> Mu1Map:
    event = _event(history)
    source = history.loc[event].copy()
    if len(source) < r1.MIN_C_POSITIVES:
        raise ValueError("insufficient opportunity-positive history for mu1")
    x = _finite(source["conversion_score"])
    y = np.clip(_finite(source["policy_net_bps"]), -r1.POLICY_CLIP_BPS, r1.POLICY_CLIP_BPS)
    if kind == "isotonic":
        fitted, _ = r1._fit_isotonic(x, y, -r1.POLICY_CLIP_BPS, r1.POLICY_CLIP_BPS)
        return Mu1Map(kind, k, fitted)
    bins = int(kind.removeprefix("bins"))
    order = np.argsort(x, kind="stable")
    partitions = [part for part in np.array_split(order, bins) if len(part)]
    global_mu1 = float(np.mean(y))
    centers, means, weights = [], [], []
    for part in partitions:
        n = len(part)
        centers.append(float(np.mean(x[part])))
        means.append(float((np.sum(y[part]) + k * global_mu1) / (n + k)))
        weights.append(float(n))
    fitted, _ = r1._fit_isotonic(
        np.asarray(centers), np.asarray(means), -r1.POLICY_CLIP_BPS, r1.POLICY_CLIP_BPS,
    )
    # Weighted refit preserves the intended support-weighted monotonic PAVA.
    fitted.fit(np.asarray(centers), np.asarray(means), sample_weight=np.asarray(weights))
    return Mu1Map(kind, k, fitted)


def _fit_mu0(history: pd.DataFrame, kind: str, k: int) -> Mu0Map:
    y = np.clip(_finite(history["policy_net_bps"]), -r1.POLICY_CLIP_BPS, r1.POLICY_CLIP_BPS)
    event = _event(history)
    negative = ~event
    if not negative.any():
        raise ValueError("no O=0 history for mu0")
    global_mean = float(np.mean(y))
    if kind == "global":
        local = y[negative]
        value = float((local.sum() + k * global_mean) / (len(local) + k))
        return Mu0Map(kind, k, np.array([-np.inf, np.inf]), np.array([value]))
    anchors = _finite(history[P0_ANCHOR])
    support = anchors[negative]
    # Deduplicate quantiles to make the map stable under tied historical P0 scores.
    edges = np.unique(np.quantile(support, np.linspace(0.0, 1.0, 6)))
    if len(edges) < 2:
        return _fit_mu0(history, "global", k)
    edges[0], edges[-1] = -np.inf, np.inf
    index = np.searchsorted(edges, anchors, side="right") - 1
    index = np.clip(index, 0, len(edges) - 2)
    values = []
    for bucket in range(len(edges) - 1):
        local = y[negative & (index == bucket)]
        values.append(float((local.sum() + k * global_mean) / (len(local) + k)))
    return Mu0Map(kind, k, edges, np.asarray(values, dtype=float))


def _fit_map(history: pd.DataFrame, mu1_kind: str, mu1_k: int, mu0_kind: str, mu0_k: int, admission: tuple[str, float]) -> K0Map:
    event = _event(history)
    probability = r2._fit_probability("platt", _finite(history["opportunity_raw_score"]), event)
    mu1 = _fit_mu1(history, mu1_kind, mu1_k)
    mu0 = _fit_mu0(history, mu0_kind, mu0_k)
    p = probability.predict(_finite(history["opportunity_raw_score"]))
    expected = p * mu1.predict(_finite(history["conversion_score"])) + (1.0 - p) * mu0.predict(_finite(history[P0_ANCHOR]))
    family, value = admission
    threshold = float(np.quantile(expected, value)) if family == "quantile" else float(value)
    return K0Map(probability, mu1, mu0, threshold)


def _apply_map(bundle: K0Map, held: pd.DataFrame) -> pd.DataFrame:
    p = bundle.probability.predict(_finite(held["opportunity_raw_score"]))
    mu1 = bundle.mu1.predict(_finite(held["conversion_score"]))
    mu0 = bundle.mu0.predict(_finite(held[P0_ANCHOR]))
    expected = p * mu1 + (1.0 - p) * mu0
    output = held.copy()
    output["opportunity_probability_round4"] = p.astype(np.float32)
    output["k0_mu1_round4_bps"] = mu1.astype(np.float32)
    output["k0_mu0_round4_bps"] = mu0.astype(np.float32)
    output["K0_expected_policy_net_bps"] = expected.astype(np.float32)
    output["K0_train_p80_expected_policy_net_bps"] = np.full(len(output), bundle.threshold, dtype=np.float32)
    return output


def _replay(
    ledger: pd.DataFrame,
    *,
    mu1: tuple[str, int],
    mu0: tuple[str, int],
    admission: tuple[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, audit = [], []
    for month, held in ledger.groupby("held_month", sort=True):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        history = ledger.loc[
            ledger["__decision_ts__"].lt(start)
            & ledger["__label_available_at__"].lt(start)
            & _valid(ledger)
        ].copy()
        record = {
            "held_month": month, "history_rows": int(len(history)), "history_months": _months(history),
            "history_opportunity_positive_rows": int(_event(history).sum()),
            "history_max_label_available_at": history["__label_available_at__"].max().isoformat() if len(history) else None,
        }
        if (
            len(history) < MIN_HISTORY_ROWS
            or _months(history) < MIN_HISTORY_MONTHS
            or int(_event(history).sum()) < r1.MIN_C_POSITIVES
        ):
            record["status"] = "skipped_insufficient_prequential_support"
            audit.append(record)
            continue
        if not history["__label_available_at__"].lt(start).all():
            raise AssertionError("K0 map history contains unresolved held-month label")
        bundle = _fit_map(history, *mu1, *mu0, admission)
        rows.append(_apply_map(bundle, held))
        record.update({"status": "complete", "threshold_bps": bundle.threshold, "mu0_kind": mu0[0], "mu0_k": mu0[1], "mu1_kind": mu1[0], "mu1_k": mu1[1]})
        audit.append(record)
    if not rows:
        raise RuntimeError("no K0 mapping fold had strict prequential support")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audit)


def _metrics(prediction: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    monthly = pd.DataFrame([
        r1._k0_metrics(part, SPEC, pd.Timestamp(f"{month}-01", tz="UTC"))
        for month, part in prediction.groupby("held_month", sort=True)
    ])
    monthly["arm"] = arm
    era = r1._aggregate_k0(monthly)
    era["arm"] = arm
    use = era.loc[era["era"].isin(("2025", "2026"))].set_index("era")
    selected = float(use["outcome_known_candidates"].sum())
    target_months = monthly.loc[monthly["held_month"].str[:4].isin(("2025", "2026"))]
    return monthly, era, {
        "arm": arm,
        "net_2025": float(use.loc["2025", "net_bps_per_trade"]),
        "net_2026": float(use.loc["2026", "net_bps_per_trade"]),
        "mean_net_bps_per_trade": float(np.average(use["net_bps_per_trade"], weights=use["outcome_known_candidates"])),
        "total_net_bps": float(use["total_net_bps"].sum()),
        "selected": selected,
        "worst_month": float(target_months["net_bps_per_trade"].min()),
        "mean_cvar10": float(use["cvar10_bps"].mean()),
    }


def _rank(summary: pd.DataFrame, control_count: float) -> pd.DataFrame:
    out = summary.copy()
    out["participation_vs_control"] = out["selected"] / max(control_count, 1.0)
    out["passes_gate"] = out["net_2025"].ge(90.0) & out["net_2026"].ge(90.0) & out["participation_vs_control"].ge(.80)
    return out.sort_values(["passes_gate", "mean_net_bps_per_trade", "worst_month", "total_net_bps"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)


def _table(frame: pd.DataFrame) -> str:
    cols = [str(col) for col in frame.columns]
    return "\n".join(["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |", *("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))])


def _load_ledger(path: Path = DEFAULT_SOURCE_PREDICTION) -> tuple[pd.DataFrame, dict[str, str]]:
    path = Path(path)
    ledger = pd.read_parquet(path)
    frame, _, _, source_hashes = r3._load_frame()
    anchors = frame.loc[:, ["candidate_id", P0_ANCHOR]].copy()
    if anchors["candidate_id"].duplicated().any():
        raise AssertionError("P0 anchor source has duplicated candidate IDs")
    ledger = ledger.merge(anchors, on="candidate_id", how="left", validate="one_to_one")
    if ledger[P0_ANCHOR].isna().any():
        raise AssertionError("frozen C60 ledger lacks a P0 anchor")
    ledger["__decision_ts__"] = pd.to_datetime(ledger["__decision_ts__"], utc=True, errors="raise")
    ledger["__label_available_at__"] = pd.to_datetime(ledger["__label_available_at__"], utc=True, errors="coerce")
    return ledger.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), {
        "round3c_control_sha256": _sha256(path),
        "round3c_manifest_sha256": _sha256(ROUND3C / "run_manifest.json"),
        **source_hashes,
    }


def _run_stage(ledger: pd.DataFrame, configs: list[tuple[str, tuple[str, int], tuple[str, int], tuple[str, float]]]) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame], dict[str, pd.DataFrame]]:
    summaries, monthly_all, era_all, predictions = [], [], [], {}
    for arm, mu1, mu0, admission in configs:
        pred, _ = _replay(ledger, mu1=mu1, mu0=mu0, admission=admission)
        monthly, era, summary = _metrics(pred, arm)
        summaries.append(summary); monthly_all.append(monthly); era_all.append(era); predictions[arm] = pred
    return pd.DataFrame(summaries), monthly_all, era_all, predictions


def run(out: Path, source_prediction: Path = DEFAULT_SOURCE_PREDICTION) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    source_prediction = Path(source_prediction)
    ledger, hashes = _load_ledger(source_prediction)
    baseline = [("A_mu1_isotonic_k0", ("isotonic", 0), ("global", 500), ("quantile", .80))]
    stage_a = baseline + [(f"A_mu1_{kind}_k{k}", (kind, k), ("global", 500), ("quantile", .80)) for kind, k in MU1_CONFIGS if kind != "isotonic"]
    a_summary, a_monthly, a_era, a_preds = _run_stage(ledger, stage_a)
    a_rank = _rank(a_summary, float(a_summary.loc[a_summary["arm"].eq("A_mu1_isotonic_k0"), "selected"].iloc[0]))
    winner_a = str(a_rank.loc[a_rank["passes_gate"], "arm"].iloc[0])
    mu1_a = next(item[1] for item in stage_a if item[0] == winner_a)

    stage_b = [(f"B_mu0_{kind}_k{k}", mu1_a, (kind, k), ("quantile", .80)) for kind, k in MU0_CONFIGS]
    b_summary, b_monthly, b_era, b_preds = _run_stage(ledger, stage_b)
    b_rank = _rank(b_summary, float(a_summary.loc[a_summary["arm"].eq("A_mu1_isotonic_k0"), "selected"].iloc[0]))
    winner_b = str(b_rank.loc[b_rank["passes_gate"], "arm"].iloc[0])
    mu0_b = next(item[2] for item in stage_b if item[0] == winner_b)

    stage_c = [(f"C_admission_{kind}_{int(value * 100) if kind == 'quantile' else int(value)}", mu1_a, mu0_b, (kind, value)) for kind, value in ADMISSIONS]
    c_summary, c_monthly, c_era, c_preds = _run_stage(ledger, stage_c)
    c_rank = _rank(c_summary, float(a_summary.loc[a_summary["arm"].eq("A_mu1_isotonic_k0"), "selected"].iloc[0]))
    winner_c = str(c_rank.loc[c_rank["passes_gate"], "arm"].iloc[0])

    out.mkdir(parents=True)
    a_rank.to_parquet(out / "round4a_mu1_mapping_ranking.parquet", index=False, compression="zstd")
    b_rank.to_parquet(out / "round4b_mu0_mapping_ranking.parquet", index=False, compression="zstd")
    c_rank.to_parquet(out / "round4c_admission_ranking.parquet", index=False, compression="zstd")
    pd.concat([*a_monthly, *b_monthly, *c_monthly], ignore_index=True).to_parquet(out / "round4_monthly_metrics.parquet", index=False, compression="zstd")
    pd.concat([*a_era, *b_era, *c_era], ignore_index=True).to_parquet(out / "round4_era_metrics.parquet", index=False, compression="zstd")
    final = c_preds[winner_c]
    final.to_parquet(out / "round4_winner_outer_oof_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short",
        "scope": "K0-only strict-prequential mapping, shrinkage, and admission study; no O/C retraining and no extra learned layer",
        "architecture": "frozen P0 -> O250_H6 -> C3/C60/uniform -> K0",
        "upstream": {"source_prediction": str(source_prediction), "description": "frozen strict-OOS O/C score ledger"},
        "stages": {"A_mu1": [list(x) for x in MU1_CONFIGS], "B_mu0": [list(x) for x in MU0_CONFIGS], "C_admission": [list(x) for x in ADMISSIONS]},
        "selection": {"gate": "2025/2026 net EV/trade >=90 bps and participation >=80% of isotonic control", "A_winner": winner_a, "B_winner": winner_b, "C_winner": winner_c},
        "causality": {"source_scores": "frozen monthly outer OOS", "map_fit": "only prior OOS scores whose labels resolved before held month", "invalid_rows": "retained and scored but excluded from mapping fit and economics", "admission": "threshold derives from the same prior-resolved calibration history only", "forbidden": ["held outcomes", "held percentile operation", "O/C refit", "extra mapper/risk/consensus layer"]},
        "sources": hashes,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Short P0 -> O250/H6 -> C3/C60 -> K0 Round 4: mapping refinement", "",
        "Research-only. All upstream opportunity and conversion scores are frozen strict-OOS C60/control outputs.", "",
        "## Stage A — conditional mu1(C)", "", _table(a_rank), "",
        "## Stage B — conditional mu0(P0 anchor)", "", _table(b_rank), "",
        "## Stage C — causal admission", "", _table(c_rank), "",
        "```json", json.dumps(manifest, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_ROUND4_K0_REFINEMENT_REPORT.md").write_text("\n".join(report))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source-prediction", type=Path, default=DEFAULT_SOURCE_PREDICTION)
    args = parser.parse_args()
    print(run(args.out, args.source_prediction))


if __name__ == "__main__":
    main()
