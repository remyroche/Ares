#!/usr/bin/env python3
"""Phase D: strict-OOF false-positive training for frozen short O250/H6.

P0/F90, the O45 field contract, C59 conversion predictions, analytic K0 and
the +75-bps admission rule remain fixed.  The sole changed quantity is the
sample weight used to fit the binary opportunity head.  Hard negatives are
identified exclusively from the existing strict-OOF O45 ledger, never from an
in-sample score.  This permits a bounded 2024--2026 comparison without
creating a new downstream layer.
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
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round2 as r2  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair as c59  # noqa: E402
import run_strict_r3_short_p0_oc_k0_phase_a_timing as phase_a  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_phase_d_false_positive_v1"
TIMING_ROOT = phase_a.TIMING_ROOT
STATIC_C59 = phase_a.STATIC_C59
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_d_false_positive_202408_202607_20260822_v1"
SEED = 1729
ADMISSION_BPS = phase_a.ADMISSION_BPS
MIN_OOF_ROWS = phase_a.MIN_OOF_ROWS
MIN_OOF_MONTHS = phase_a.MIN_OOF_MONTHS
SPEC = r1.OpportunitySpec("O250_H6", 6, 250.0)


@dataclass(frozen=True)
class Arm:
    kind: str  # uniform | hard_negative | graded
    top_fraction: float | None = None
    multiplier: float | None = None

    @property
    def name(self) -> str:
        if self.kind == "uniform":
            return "D0_uniform"
        if self.kind == "graded":
            return "D2_graded_oof_hard_negative"
        return f"D1_hardneg_top{int(self.top_fraction * 100):02d}_x{self.multiplier:g}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid(frame: pd.DataFrame) -> pd.Series:
    # Explicit object conversion avoids pandas' future silent-downcast warning
    # without changing how a null timing label is handled.
    timing_valid = frame["event_timing_label_valid"].astype("boolean").fillna(False).astype(bool)
    timing_invalid = frame["event_timing_target_invalid"].astype("boolean").fillna(True).astype(bool)
    return r1._valid_label(frame) & timing_valid & ~timing_invalid


def _event(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["favourable_hit_6h"], errors="coerce").fillna(0).astype(int).to_numpy()


def _load() -> tuple[pd.DataFrame, tuple[str, ...], dict[str, str]]:
    frame, o45, _m4, source_hashes = r3._load_frame()
    if any(field not in frame for field in o45):
        raise AssertionError("frozen O45 field missing from target-free short frame")
    parts = sorted(TIMING_ROOT.glob("parts/month=*/side=short.parquet"))
    timing = pd.concat([pd.read_parquet(part, columns=[*r1.IDENTITY, "event_timing_label_valid", "event_timing_target_invalid", "favourable_hit_6h"]) for part in parts], ignore_index=True)
    for name in ("__ts__", "__decision_ts__"):
        timing[name] = r1._utc(timing[name])
    if timing.candidate_id.duplicated().any():
        raise AssertionError("timing label identity is non-unique")
    frame = frame.merge(timing, on=list(r1.IDENTITY), how="left", validate="one_to_one")
    static = pd.read_parquet(STATIC_C59, columns=["candidate_id", "__decision_ts__", "opportunity_probability", "conversion_score", "held_month"])
    static["__decision_ts__"] = r1._utc(static["__decision_ts__"])
    if static.candidate_id.duplicated().any():
        raise AssertionError("frozen C59 OOF identity is non-unique")
    static = static.rename(columns={"opportunity_probability": "static_oof_probability", "conversion_score": "frozen_c59_score", "held_month": "static_held_month"})
    frame = frame.merge(static, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    hashes = {**source_hashes, "timing_manifest": _sha256(TIMING_ROOT / "run_manifest.json"), "static_c59": _sha256(STATIC_C59)}
    return frame, tuple(o45), hashes


def _model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=.035, max_depth=3,
        num_leaves=15, min_child_samples=40, subsample=.85, subsample_freq=1,
        colsample_bytree=.85, reg_lambda=4.0, reg_alpha=.10, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _rank(values: pd.Series) -> np.ndarray:
    value = pd.to_numeric(values, errors="coerce").to_numpy(float)
    result = np.full(len(value), np.nan, dtype=float)
    ok = np.isfinite(value)
    if ok.any():
        order = np.argsort(value[ok], kind="stable")
        local = np.empty(int(ok.sum()), dtype=float)
        local[order] = (np.arange(int(ok.sum()), dtype=float) + 1.0) / float(ok.sum())
        result[ok] = local
    return result


def _weights(fit: pd.DataFrame, arm: Arm) -> tuple[np.ndarray, dict[str, Any]]:
    y = _event(fit)
    weight = np.ones(len(fit), dtype=float)
    if arm.kind == "uniform":
        return weight, {"hard_negative_rows": 0, "oof_covered_rows": int(fit.static_oof_probability.notna().sum())}
    rank = _rank(fit["static_oof_probability"])
    negative = y == 0
    if arm.kind == "hard_negative":
        assert arm.top_fraction is not None and arm.multiplier is not None
        selected = negative & np.isfinite(rank) & (rank > 1.0 - arm.top_fraction)
        weight[selected] = arm.multiplier
    elif arm.kind == "graded":
        selected = negative & np.isfinite(rank)
        weight[selected & (rank >= .70) & (rank < .80)] = 1.25
        weight[selected & (rank >= .80) & (rank < .90)] = 1.50
        weight[selected & (rank >= .90)] = 2.00
    else:
        raise ValueError(arm.kind)
    weight /= max(float(weight.mean()), 1e-12)
    return weight, {"hard_negative_rows": int((weight > 1.000001).sum()), "oof_covered_rows": int(np.isfinite(rank).sum()), "max_weight": float(weight.max())}


def _fit_k0(inner: pd.DataFrame) -> phase_a.K0Bundle:
    return phase_a._fit_k0(inner)


def _inner_oof(train: pd.DataFrame, fields: tuple[str, ...], arm: Arm, held_month: str, seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    local = train.loc[_valid(train) & train["frozen_c59_score"].notna()].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    bounds = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    pieces: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for fold in range(r1.INNER_SPLITS):
        valid = local.iloc[int(bounds[fold + 1]):int(bounds[fold + 2])].copy()
        if valid.empty:
            continue
        cutoff = valid["__decision_ts__"].min()
        fit = local.loc[local["__label_available_at__"].lt(cutoff)].copy()
        if len(fit) < r1.MIN_OUTER_TRAIN_ROWS or np.unique(_event(fit)).size < 2:
            continue
        x_fit, med = r1._matrix(fit, fields)
        x_valid, _ = r1._matrix(valid, fields, med)
        weights, weight_audit = _weights(fit, arm)
        model = _model(seed + fold)
        model.fit(x_fit, _event(fit), sample_weight=weights)
        part = valid.loc[:, [*r1.IDENTITY, "__label_available_at__", "policy_net_bps", "policy_regret_bps", "policy_gross_bps", "frozen_c59_score"]].copy()
        part["opp_oof_raw"] = model.predict_proba(x_valid)[:, 1].astype(np.float32)
        part["conversion_oof_raw"] = part.pop("frozen_c59_score").to_numpy(np.float32)
        part["event_target"] = _event(valid).astype(np.int8)
        part["held_month"] = held_month
        pieces.append(part)
        audit.append({"inner_fold": fold, "valid_rows": len(valid), "fit_rows": len(fit), **weight_audit})
    if not pieces:
        raise ValueError("no purged hard-negative inner OOF support")
    output = pd.concat(pieces, ignore_index=True)
    if len(output) < MIN_OOF_ROWS or output["__decision_ts__"].dt.strftime("%Y-%m").nunique() < MIN_OOF_MONTHS:
        raise ValueError("insufficient combined hard-negative/C59 OOF support")
    return output, audit


def _outer_month(frame: pd.DataFrame, fields: tuple[str, ...], arm: Arm, month: pd.Timestamp, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = month + pd.offsets.MonthBegin(1)
    held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(end)].copy()
    train = frame.loc[frame["__decision_ts__"].lt(month) & frame["__label_available_at__"].lt(month) & _valid(frame)].copy()
    if held.empty:
        raise ValueError("empty held month")
    if held["frozen_c59_score"].isna().any():
        raise ValueError("frozen C59 OOF score unavailable for held population")
    inner, inner_audit = _inner_oof(train, fields, arm, month.strftime("%Y-%m"), seed)
    bundle = _fit_k0(inner)
    x_train, med = r1._matrix(train, fields)
    x_held, _ = r1._matrix(held, fields, med)
    weights, outer_weight = _weights(train, arm)
    model = _model(seed + 10_000)
    model.fit(x_train, _event(train), sample_weight=weights)
    raw = model.predict_proba(x_held)[:, 1]
    output = held.loc[:, [*r1.IDENTITY, "__label_available_at__", "policy_net_bps", "policy_regret_bps", "policy_gross_bps", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid", "event_timing_label_valid", "event_timing_target_invalid", "favourable_hit_6h"]].copy().reset_index(drop=True)
    output["opportunity_raw_score"] = raw.astype(np.float32)
    output = pd.concat((output, phase_a._apply_k0(bundle, raw, held["frozen_c59_score"].to_numpy(float))), axis=1)
    output["held_month"] = month.strftime("%Y-%m")
    output["arm"] = arm.name
    return output, {"arm": arm.name, "held_month": month.strftime("%Y-%m"), "status": "complete", "held_rows": len(held), "outer_train_rows": len(train), "inner_oof_rows": bundle.oof_rows, "inner_oof_months": bundle.oof_months, "k0_mu0_bps": bundle.mu0, "outer_weight": outer_weight, "inner_weight": inner_audit}


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6))) if len(y) and np.unique(y).size > 1 else float("nan")


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(y) < 20 or np.unique(y).size < 2 or np.unique(p).size < 2:
        return float("nan"), float("nan")
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(logit, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _month_metrics(part: pd.DataFrame) -> dict[str, Any]:
    valid = part.loc[_valid(part)].copy()
    event = _event(valid)
    p = valid["opportunity_probability"].to_numpy(float)
    slope, intercept = _calibration(event, p)
    row: dict[str, Any] = {"arm": str(part["arm"].iloc[0]), "held_month": str(part["held_month"].iloc[0]), "valid_rows": len(valid), "event_prevalence": float(event.mean()) if len(event) else float("nan"), "auc": float(roc_auc_score(event, p)) if len(valid) and np.unique(event).size > 1 else float("nan"), "prauc": float(average_precision_score(event, p)) if len(valid) and np.unique(event).size > 1 else float("nan"), "brier": _brier(event, p), "calibration_slope": slope, "calibration_intercept": intercept}
    order = np.argsort(p, kind="stable"); rank = np.empty(len(valid), dtype=float); rank[order] = (np.arange(len(valid)) + 1.0) / max(len(valid), 1)
    for fraction in (.10, .20, .30):
        selected = rank > 1.0 - fraction
        precision = float(event[selected].mean()) if selected.any() else float("nan")
        row[f"precision_top{int(fraction * 100)}"] = precision
        row[f"lift_top{int(fraction * 100)}"] = precision / max(float(event.mean()), 1e-12) if np.isfinite(precision) else float("nan")
    row["false_positive_rate_top20"] = 1.0 - row["precision_top20"] if np.isfinite(row["precision_top20"]) else float("nan")
    admitted = part.loc[pd.to_numeric(part["K0_expected_policy_net_bps"], errors="coerce").ge(ADMISSION_BPS)]
    known = admitted.loc[_valid(admitted)]
    net = r1._finite(known["policy_net_bps"]).to_numpy(float)
    row.update({"admitted": len(admitted), "known_admitted": len(known), "net_bps_per_trade": float(net.mean()) if len(net) else float("nan"), "total_net_bps": float(net.sum()) if len(net) else 0.0, "cvar10_bps": r1._cvar(net), "p_net_lt_neg200": float((net < -200).mean()) if len(net) else float("nan"), "p_net_lt_neg400": float((net < -400).mean()) if len(net) else float("nan"), "positive_fraction": float((net > 0).mean()) if len(net) else float("nan")})
    return row


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float); weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() else float("nan")


def _era(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, era), group in monthly.assign(era=monthly.held_month.str[:4]).groupby(["arm", "era"], sort=True):
        weights = np.maximum(group.known_admitted.to_numpy(float), 1.0)
        row: dict[str, Any] = {"arm": arm, "era": era, "months": len(group), "admitted": int(group.admitted.sum()), "known_admitted": int(group.known_admitted.sum()), "total_net_bps": float(group.total_net_bps.sum()), "positive_months": int((group.net_bps_per_trade > 0).sum()), "worst_month_net_bps": float(group.net_bps_per_trade.min())}
        for name in ("auc", "prauc", "brier", "calibration_slope", "calibration_intercept", "precision_top10", "precision_top20", "precision_top30", "lift_top10", "lift_top20", "lift_top30", "false_positive_rate_top20", "net_bps_per_trade", "cvar10_bps", "p_net_lt_neg200", "p_net_lt_neg400", "positive_fraction"):
            row[name] = _weighted(group[name].to_numpy(float), weights)
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(monthly: pd.DataFrame, era: pd.DataFrame) -> pd.DataFrame:
    use = era.loc[era.era.isin(("2025", "2026"))]
    rows: list[dict[str, Any]] = []
    for arm, group in use.groupby("arm", sort=True):
        weights = np.maximum(group.known_admitted.to_numpy(float), 1.0)
        by = group.set_index("era")
        months = monthly.loc[monthly.arm.eq(arm) & monthly.held_month.str[:4].isin(("2025", "2026"))]
        rows.append({"arm": arm, "net_2025": float(by.loc["2025", "net_bps_per_trade"]) if "2025" in by.index else float("nan"), "net_2026": float(by.loc["2026", "net_bps_per_trade"]) if "2026" in by.index else float("nan"), "mean_net_bps": _weighted(group.net_bps_per_trade.to_numpy(float), weights), "total_net_bps": float(group.total_net_bps.sum()), "worst_era_net_bps": float(group.net_bps_per_trade.min()), "worst_month_net_bps": float(months.net_bps_per_trade.min()), "mean_auc": _weighted(group.auc.to_numpy(float), weights), "precision_top20": _weighted(group.precision_top20.to_numpy(float), weights), "false_positive_rate_top20": _weighted(group.false_positive_rate_top20.to_numpy(float), weights), "cvar10_bps": _weighted(group.cvar10_bps.to_numpy(float), weights), "admitted": int(group.admitted.sum())})
    out = pd.DataFrame(rows)
    out["advances_phase_d"] = out.mean_net_bps.ge(90.0) & out.net_2025.ge(90.0) & out.net_2026.ge(90.0) & out.mean_auc.gt(.5)
    return out.sort_values(["mean_net_bps", "worst_month_net_bps", "precision_top20", "total_net_bps"], ascending=False, kind="stable").reset_index(drop=True)


def _table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        cols = list(map(str, frame.columns))
        return "\n".join(["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |", *("| " + " | ".join(str(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))])


def run(out: Path, arms: tuple[Arm, ...], months: pd.DatetimeIndex) -> Path:
    if out.exists():
        raise FileExistsError(out)
    frame, fields, hashes = _load()
    predictions: list[pd.DataFrame] = []; audits: list[dict[str, Any]] = []; metrics: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(arms):
        for month_index, month in enumerate(months):
            try:
                pred, audit = _outer_month(frame, fields, arm, month, SEED + arm_index * 20_000 + month_index * 101)
                predictions.append(pred); audits.append(audit); metrics.append(_month_metrics(pred))
            except ValueError as exc:
                audits.append({"arm": arm.name, "held_month": month.strftime("%Y-%m"), "status": "skipped", "reason": str(exc)})
    if not predictions:
        raise RuntimeError("Phase D produced no strict OOF rows")
    monthly = pd.DataFrame(metrics); era = _era(monthly); summary = _summary(monthly, era)
    out.mkdir(parents=True)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "phase_d_outer_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "phase_d_fold_audit.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "phase_d_monthly_metrics.parquet", index=False, compression="zstd")
    era.to_parquet(out / "phase_d_era_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "phase_d_summary.parquet", index=False, compression="zstd")
    manifest = {"schema": SCHEMA, "status": "complete", "side": "short", "scope": "Phase D only; no canonical/live change", "period": {"candidate_start": "2024-05", "output_supported_start": "2024-10", "end_exclusive": "2026-08"}, "arms": [arm.name for arm in arms], "opportunity": "frozen short MFE6h >250bps", "features": {"O45": list(fields), "C59": "existing strict OOF C59 ledger"}, "admission": {"K0_expected_policy_net_bps_gte": ADMISSION_BPS}, "causality": {"hard_negative_mining": "only previous strict-OOF O45 probabilities, within the contemporaneous fit population", "outer_and_inner": "label_available_at < validation decision", "candidates": "all target-free P0 candidates scored", "invalidity": "incomplete paths excluded only from training/outcome metrics"}, "sources": hashes}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "SHORT_P0_OC_K0_PHASE_D_FALSE_POSITIVE_REPORT.md").write_text("\n".join(["# Short P0 → false-positive O training → frozen C59 → K0: Phase D", "", "Research-only. All results use strict OOF hard-negative mining and separate 2024, 2025 and 2026 evidence.", "", "## Era metrics", "", _table(era), "", "## Sequential selection", "", _table(summary), "", "```json", json.dumps(manifest, indent=2), "```", ""]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--months", nargs="*", default=None)
    parser.add_argument("--arm", action="append", default=[], help="uniform | graded | hard:10:1.5")
    args = parser.parse_args()
    if args.months:
        months = pd.DatetimeIndex([pd.Timestamp(value + "-01T00:00:00Z") for value in args.months])
    else:
        months = pd.date_range("2024-05-01T00:00:00Z", "2026-08-01T00:00:00Z", freq="MS", inclusive="left")
    if args.arm:
        parsed: list[Arm] = []
        for value in args.arm:
            if value == "uniform": parsed.append(Arm("uniform"))
            elif value == "graded": parsed.append(Arm("graded"))
            else:
                kind, top, multiplier = value.split(":", 2)
                if kind != "hard": raise ValueError(value)
                parsed.append(Arm("hard_negative", float(top) / 100.0, float(multiplier)))
        arms = tuple(parsed)
    else:
        arms = (Arm("uniform"), *(Arm("hard_negative", top, multiplier) for top in (.10, .20, .30) for multiplier in (1.5, 2.0, 2.5)), Arm("graded"))
    print(run(args.out, arms, months))


if __name__ == "__main__":
    main()
