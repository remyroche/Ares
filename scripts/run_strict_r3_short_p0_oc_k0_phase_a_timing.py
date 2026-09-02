#!/usr/bin/env python3
"""Phase A: strict-prequential short P0 -> time-to-event O -> frozen C59 -> K0.

This is a deliberately narrow, research-only implementation of the first
funnel in ``SHORT_P0_OC_K0_NEXT_ABLATION_REPORT``.  It keeps P0, the O45
feature contract and C59 normalized-regret conversion contract fixed.  The
only changed component is the 6-hour O head: a shared discrete-time hazard
for the exact event ``short MFE_6h > 250 bps``.

Every monthly outer model and all of its inner calibration predictions are
strictly prequential: their supervised rows have ``label_available_at`` before
the validation decision.  Target-free candidates are always scored; invalid
paths are excluded only from supervised fitting and outcome metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round2 as r2  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair as c59  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_phase_a_timing_v1"
TIMING_ROOT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_event_timing_labels_202405_202607_20260822_v8"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_a_timing_202408_202607_20260822_v1"
STATIC_C59 = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair_20260822_v1/C59_outer_oof_predictions.parquet"
SEED = 1729
MIN_C_ROWS = 500
MIN_OOF_ROWS = 1_000
MIN_OOF_MONTHS = 3
ADMISSION_BPS = 75.0
POLICY_CLIP_BPS = 500.0


@dataclass(frozen=True)
class TimingSpec:
    name: str
    endpoints_minutes: tuple[int, ...]
    weight_kind: str

    @property
    def bins(self) -> tuple[tuple[int, int], ...]:
        starts = (0, *self.endpoints_minutes[:-1])
        return tuple(zip(starts, self.endpoints_minutes, strict=True))


BASE_GEOMETRIES = {
    "A2a_1h": (60, 120, 180, 240, 360),
    "A2b_30m": (30, 60, 120, 240, 360),
    "A2c_2h": (120, 240, 360),
}
SPECS = tuple(
    TimingSpec(f"{name}__{weight}", endpoints, weight)
    for name, endpoints in BASE_GEOMETRIES.items()
    for weight in ("uniform", "early")
)


@dataclass
class K0Bundle:
    calibration: r2.ProbabilityCalibrator
    mu1: IsotonicRegression
    mu0: float
    oof_rows: int
    oof_months: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _valid(frame: pd.DataFrame) -> pd.Series:
    return (
        r1._valid_label(frame)
        & frame["event_timing_label_valid"].fillna(False).astype(bool)
        & ~frame["event_timing_target_invalid"].fillna(True).astype(bool)
    )


def _event(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["favourable_hit_6h"], errors="coerce").fillna(0).astype(int).to_numpy()


def _time(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["first_favourable_250bps_minute"], errors="coerce").to_numpy(float)


def _load() -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], dict[str, str]]:
    frame, o45, _m4, source_hashes = r3._load_frame()
    c_fields = c59._c59()
    if any(field not in frame for field in (*o45, *c_fields)):
        raise AssertionError("frozen O45/C59 field missing from target-free short frame")
    parts = sorted(TIMING_ROOT.glob("parts/month=*/side=short.parquet"))
    if not parts or not (TIMING_ROOT / "run_manifest.json").exists():
        raise FileNotFoundError(f"missing timing labels: {TIMING_ROOT}")
    columns = [
        *r1.IDENTITY, "event_timing_label_valid", "event_timing_target_invalid",
        "first_favourable_250bps_minute", "favourable_hit_1h", "favourable_hit_2h",
        "favourable_hit_4h", "favourable_hit_6h",
    ]
    labels = pd.concat([pd.read_parquet(part, columns=columns) for part in parts], ignore_index=True)
    for field in ("__ts__", "__decision_ts__"):
        labels[field] = r1._utc(labels[field])
    if labels.candidate_id.duplicated().any():
        raise AssertionError("exact timing labels have duplicate identities")
    frame = frame.merge(labels, on=list(r1.IDENTITY), how="left", validate="one_to_one")
    if frame.candidate_id.duplicated().any() or len(frame) == 0:
        raise AssertionError("target-free timing-label join changed identities")
    hashes = {**source_hashes, "timing_labels_manifest": _sha256(TIMING_ROOT / "run_manifest.json")}
    return frame, tuple(o45), tuple(c_fields), hashes


def _hazard_matrix(frame: pd.DataFrame, fields: tuple[str, ...], *, medians: pd.Series | None = None, bin_index: int) -> tuple[pd.DataFrame, pd.Series]:
    x, medians = r1._matrix(frame, fields, medians)
    x = x.copy()
    x["__time_bin__"] = np.int16(bin_index)
    return x, medians


def _risk_rows(frame: pd.DataFrame, spec: TimingSpec) -> tuple[pd.DataFrame, np.ndarray]:
    """Expand candidate rows only while they remain at risk of first event."""
    first = _time(frame)
    pieces: list[pd.DataFrame] = []
    target: list[np.ndarray] = []
    for index, (start, end) in enumerate(spec.bins):
        at_risk = ~np.isfinite(first) | (first > float(start))
        part = frame.loc[at_risk].copy()
        part["__time_bin__"] = np.int16(index)
        pieces.append(part)
        target.append((np.isfinite(first[at_risk]) & (first[at_risk] <= float(end))).astype(np.int8))
    return pd.concat(pieces, ignore_index=True), np.concatenate(target)


def _hazard_weights(expanded: pd.DataFrame, spec: TimingSpec) -> np.ndarray:
    weight = np.ones(len(expanded), dtype=float)
    if spec.weight_kind == "early":
        starts = np.asarray([spec.bins[int(i)][0] for i in expanded["__time_bin__"].to_numpy(int)], dtype=int)
        weight = np.where(starts < 60, 1.25, np.where(starts < 120, 1.15, np.where(starts < 240, 1.0, .85)))
    elif spec.weight_kind != "uniform":
        raise ValueError(spec.weight_kind)
    return weight / max(float(weight.mean()), 1e-12)


def _hazard_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=.035, max_depth=3,
        num_leaves=15, min_child_samples=40, subsample=.85, subsample_freq=1,
        colsample_bytree=.85, reg_lambda=4.0, reg_alpha=.10, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _fit_hazard(train: pd.DataFrame, fields: tuple[str, ...], spec: TimingSpec, seed: int) -> tuple[LGBMClassifier, pd.Series]:
    expanded, y = _risk_rows(train, spec)
    x, medians = r1._matrix(expanded, fields)
    x["__time_bin__"] = expanded["__time_bin__"].to_numpy(np.int16)
    model = _hazard_model(seed)
    model.fit(x, y, sample_weight=_hazard_weights(expanded, spec), categorical_feature=["__time_bin__"])
    return model, medians


def _predict_hazard(model: LGBMClassifier, held: pd.DataFrame, fields: tuple[str, ...], medians: pd.Series, spec: TimingSpec) -> dict[str, np.ndarray]:
    survival = np.ones(len(held), dtype=float)
    output: dict[str, np.ndarray] = {}
    for index, (start, end) in enumerate(spec.bins):
        x, _ = _hazard_matrix(held, fields, medians=medians, bin_index=index)
        h = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1.0 - 1e-6)
        before = survival.copy()
        # A coarser bin has no extra post-entry feature observation within the
        # bin.  For requested checkpoints inside it, use the constant-hazard
        # interpolation implied by that same fitted bin—not the later endpoint
        # probability.  Thus A2c's 1h output remains an honest 1h estimate.
        for horizon in (1, 2, 4, 6):
            minute = horizon * 60
            if start < minute <= end:
                fraction = (minute - start) / float(end - start)
                output[f"p_event_{horizon}h"] = (1.0 - before * np.power(1.0 - h, fraction)).astype(np.float32)
        survival *= 1.0 - h
    if sorted(output) != ["p_event_1h", "p_event_2h", "p_event_4h", "p_event_6h"]:
        raise AssertionError(f"timing checkpoints not covered by {spec.name}: {sorted(output)}")
    output["opportunity_raw_score"] = output["p_event_6h"].copy()
    return output


def _fit_k0(inner: pd.DataFrame) -> K0Bundle:
    event = inner["event_target"].astype(int).to_numpy()
    if int(event.sum()) < MIN_C_ROWS:
        raise ValueError("insufficient true O-positive timing OOF support")
    y = r1._finite(inner["policy_net_bps"]).clip(-POLICY_CLIP_BPS, POLICY_CLIP_BPS).to_numpy(float)
    calibration = r2._fit_probability("platt", inner["opp_oof_raw"].to_numpy(float), event)
    mu1, _ = r1._fit_isotonic(inner.loc[event.astype(bool), "conversion_oof_raw"].to_numpy(float), y[event.astype(bool)], -POLICY_CLIP_BPS, POLICY_CLIP_BPS)
    negative = ~event.astype(bool)
    global_mean = float(np.mean(y))
    mu0 = float((y[negative].sum() + 500.0 * global_mean) / (negative.sum() + 500.0))
    return K0Bundle(calibration, mu1, mu0, len(inner), int(inner["__decision_ts__"].dt.strftime("%Y-%m").nunique()))


def _apply_k0(bundle: K0Bundle, raw_o: np.ndarray, raw_c: np.ndarray) -> pd.DataFrame:
    p = bundle.calibration.predict(raw_o)
    expected = p * np.asarray(bundle.mu1.predict(raw_c), dtype=float) + (1.0 - p) * bundle.mu0
    return pd.DataFrame({
        "opportunity_probability": p.astype(np.float32),
        "conversion_score": np.asarray(raw_c, dtype=np.float32),
        "K0_expected_policy_net_bps": expected.astype(np.float32),
        "K0_admission_threshold_bps": np.full(len(expected), ADMISSION_BPS, dtype=np.float32),
    })


def _inner_oof(train: pd.DataFrame, o_fields: tuple[str, ...], c_fields: tuple[str, ...], spec: TimingSpec, held_month: str, seed: int) -> pd.DataFrame:
    local = train.loc[_valid(train)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    bounds = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    c_target = next(value for value in r3.TARGETS if value.name == "C3_normalized_regret")
    pieces: list[pd.DataFrame] = []
    for fold in range(r1.INNER_SPLITS):
        valid = local.iloc[int(bounds[fold + 1]):int(bounds[fold + 2])].copy()
        if valid.empty:
            continue
        cutoff = valid["__decision_ts__"].min()
        fit = local.loc[local["__label_available_at__"].lt(cutoff)].copy()
        c_fit = fit.loc[r1._event(fit, r3.SPEC).astype(bool)].copy()
        if len(fit) < r1.MIN_OUTER_TRAIN_ROWS or len(c_fit) < MIN_C_ROWS or r1._month_count(c_fit) < MIN_OOF_MONTHS:
            continue
        if np.unique(_event(fit)).size < 2:
            continue
        hazard, med_o = _fit_hazard(fit, o_fields, spec, seed + fold)
        raw_o = _predict_hazard(hazard, valid, o_fields, med_o, spec)["opportunity_raw_score"]
        y_c = r3._target(c_fit, c_target)
        if np.unique(y_c).size < 2:
            continue
        x_c, med_c = r1._matrix(c_fit, c_fields)
        x_v_c, _ = r1._matrix(valid, c_fields, med_c)
        c_model = r3._model(c_target, seed + 10_000 + fold)
        c_model.fit(x_c, y_c, sample_weight=r3._c_weights(c_fit, "uniform"))
        raw_c = r3._predict(c_model, c_target, x_v_c)
        part = valid.loc[:, [*r1.IDENTITY, "__label_available_at__", "policy_net_bps", "policy_regret_bps", "policy_gross_bps", "first_favourable_250bps_minute", "favourable_hit_1h", "favourable_hit_2h", "favourable_hit_4h", "favourable_hit_6h"]].copy()
        part["opp_oof_raw"] = raw_o.astype(np.float32)
        part["conversion_oof_raw"] = raw_c.astype(np.float32)
        part["event_target"] = _event(valid).astype(np.int8)
        part["held_month"] = held_month
        pieces.append(part)
    if not pieces:
        raise ValueError("no purged inner timing OOF support")
    out = pd.concat(pieces, ignore_index=True)
    if len(out) < MIN_OOF_ROWS or out["__decision_ts__"].dt.strftime("%Y-%m").nunique() < MIN_OOF_MONTHS:
        raise ValueError("insufficient combined timing/C OOF support")
    return out


def _outer_month(frame: pd.DataFrame, o_fields: tuple[str, ...], c_fields: tuple[str, ...], spec: TimingSpec, month: pd.Timestamp, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = month + pd.offsets.MonthBegin(1)
    held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(end)].copy()
    train = frame.loc[frame["__decision_ts__"].lt(month) & frame["__label_available_at__"].lt(month) & _valid(frame)].copy()
    if len(held) == 0:
        raise ValueError("empty held month")
    inner = _inner_oof(train, o_fields, c_fields, spec, month.strftime("%Y-%m"), seed)
    bundle = _fit_k0(inner)
    hazard, med_o = _fit_hazard(train, o_fields, spec, seed + 20_000)
    hazard_scores = _predict_hazard(hazard, held, o_fields, med_o, spec)
    c_target = next(value for value in r3.TARGETS if value.name == "C3_normalized_regret")
    c_train = train.loc[r1._event(train, r3.SPEC).astype(bool)].copy()
    y_c = r3._target(c_train, c_target)
    if len(c_train) < MIN_C_ROWS or np.unique(y_c).size < 2:
        raise ValueError("insufficient frozen C59 training support")
    x_c, med_c = r1._matrix(c_train, c_fields)
    x_h_c, _ = r1._matrix(held, c_fields, med_c)
    c_model = r3._model(c_target, seed + 30_000)
    c_model.fit(x_c, y_c, sample_weight=r3._c_weights(c_train, "uniform"))
    raw_c = r3._predict(c_model, c_target, x_h_c)
    output = held.loc[:, [*r1.IDENTITY, "__label_available_at__", "policy_net_bps", "policy_regret_bps", "policy_gross_bps", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid", "event_timing_label_valid", "event_timing_target_invalid", "first_favourable_250bps_minute", "favourable_hit_1h", "favourable_hit_2h", "favourable_hit_4h", "favourable_hit_6h"]].copy().reset_index(drop=True)
    for name, values in hazard_scores.items():
        output[name] = values
    output = pd.concat((output, _apply_k0(bundle, hazard_scores["opportunity_raw_score"], raw_c)), axis=1)
    output["held_month"] = month.strftime("%Y-%m")
    output["arm"] = spec.name
    audit = {"arm": spec.name, "held_month": month.strftime("%Y-%m"), "status": "complete", "held_rows": len(held), "outer_train_rows": len(train), "outer_c_rows": len(c_train), "inner_oof_rows": bundle.oof_rows, "inner_oof_months": bundle.oof_months, "k0_mu0_bps": bundle.mu0}
    return output, audit


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6))) if len(y) and np.unique(y).size > 1 else float("nan")


def _month_metrics(part: pd.DataFrame) -> dict[str, Any]:
    valid = part.loc[_valid(part)].copy()
    event = _event(valid)
    p6 = valid["opportunity_probability"].to_numpy(float)
    row: dict[str, Any] = {"arm": str(part["arm"].iloc[0]), "held_month": str(part["held_month"].iloc[0]), "valid_rows": len(valid), "event_prevalence": float(event.mean()) if len(event) else float("nan")}
    if len(valid) and np.unique(event).size > 1:
        row.update({"auc": float(roc_auc_score(event, p6)), "prauc": float(average_precision_score(event, p6)), "brier": _brier(event, p6)})
    else:
        row.update({"auc": float("nan"), "prauc": float("nan"), "brier": float("nan")})
    for horizon in (1, 2, 4, 6):
        y = pd.to_numeric(valid[f"favourable_hit_{horizon}h"], errors="coerce").fillna(0).astype(int).to_numpy()
        row[f"brier_{horizon}h"] = _brier(y, valid[f"p_event_{horizon}h"].to_numpy(float))
    row["integrated_brier"] = float(np.nanmean([row[f"brier_{h}h"] for h in (1, 2, 4, 6)]))
    if event.sum() >= 5:
        actual = valid.loc[event.astype(bool), "first_favourable_250bps_minute"].to_numpy(float)
        predicted_early = valid.loc[event.astype(bool), "p_event_2h"].to_numpy(float)
        target_early = (actual <= 120.0).astype(int)
        row["time_auc_early_vs_late"] = float(roc_auc_score(target_early, predicted_early)) if np.unique(target_early).size > 1 else float("nan")
        row["event_time_mean_error_minutes"] = float(np.mean(actual) - 360.0 * np.mean(1.0 - predicted_early))
    else:
        row["time_auc_early_vs_late"] = float("nan"); row["event_time_mean_error_minutes"] = float("nan")
    order = np.argsort(p6, kind="stable"); rank = np.empty(len(valid), dtype=float); rank[order] = (np.arange(len(valid)) + 1) / max(len(valid), 1)
    for fraction in (.10, .20, .30):
        selected = rank > 1.0 - fraction
        row[f"precision_top{int(fraction * 100)}"] = float(event[selected].mean()) if selected.any() else float("nan")
    admitted = part.loc[pd.to_numeric(part["K0_expected_policy_net_bps"], errors="coerce").ge(ADMISSION_BPS)]
    known = admitted.loc[_valid(admitted)]
    net = r1._finite(known["policy_net_bps"]).to_numpy(float)
    row.update({"admitted": len(admitted), "known_admitted": len(known), "net_bps_per_trade": float(net.mean()) if len(net) else float("nan"), "total_net_bps": float(net.sum()) if len(net) else 0.0, "cvar10_bps": r1._cvar(net), "positive_fraction": float((net > 0).mean()) if len(net) else float("nan")})
    return row


def _era_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, era), group in monthly.assign(era=monthly["held_month"].str[:4]).groupby(["arm", "era"], sort=True):
        known = group["known_admitted"].to_numpy(float)
        row: dict[str, Any] = {"arm": arm, "era": era, "months": len(group), "admitted": int(group["admitted"].sum()), "known_admitted": int(known.sum()), "total_net_bps": float(group["total_net_bps"].sum()), "positive_months": int((group["net_bps_per_trade"] > 0).sum()), "worst_month_net_bps": float(group["net_bps_per_trade"].min())}
        for column in ("auc", "prauc", "brier", "integrated_brier", "time_auc_early_vs_late", "precision_top10", "precision_top20", "precision_top30", "net_bps_per_trade", "cvar10_bps", "positive_fraction"):
            row[column] = _weighted_nanmean(group[column].to_numpy(float), np.maximum(known, 1.0)) if column in group else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    return float(np.average(values[mask], weights=weights[mask])) if mask.any() else float("nan")


def _report(out: Path, era: pd.DataFrame, summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def table(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            columns = [str(column) for column in frame.columns]
            return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |", *("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))])
    lines = ["# Short P0 → time-to-event O → C59 → K0: Phase A", "", "Research-only; P0 and C59 remain frozen.  All results use target-free scoring and strict prequential label availability.", "", "## Era metrics", "", table(era), "", "## Sequential selection", "", table(summary), "", "## Contract", "", "- Favourable event: short MFE within 6h strictly greater than 250 bps.", "- C59: frozen normalized-regret conversion head on true MFE6h >250 bps rows.", f"- K0 admission: expected exact policy net ≥ {ADMISSION_BPS:g} bps; no held-period top-k.", "- 2024 is retained as a prequential warm-up/diagnostic era; later K0 evidence starts only when all purged support gates are met.", "", "```json", json.dumps(manifest, indent=2), "```", ""]
    (out / "SHORT_P0_OC_K0_PHASE_A_TIMING_REPORT.md").write_text("\n".join(lines))


def build_static_control(out: Path) -> Path:
    """Materialise the frozen static O250/H6+C59 control at the Phase-A gate.

    The only difference from the existing immutable C59 handoff is evaluation
    at this funnel's predeclared absolute +75-bps K0 gate rather than its
    historic per-fold p80 screening threshold.  Scores, calibrators, models,
    features and labels are not re-fit or changed.
    """
    if out.exists():
        raise FileExistsError(out)
    source = pd.read_parquet(STATIC_C59)
    labels = pd.concat([pd.read_parquet(part, columns=[*r1.IDENTITY, "event_timing_label_valid", "event_timing_target_invalid", "favourable_hit_6h"]) for part in sorted(TIMING_ROOT.glob("parts/month=*/side=short.parquet"))], ignore_index=True)
    source = source.merge(labels, on=list(r1.IDENTITY), how="left", validate="one_to_one")
    if source["event_timing_label_valid"].isna().any():
        raise AssertionError("static C59 control misses exact timing-label identities")
    source["arm"] = "A0_static_binary__uniform"
    source["K0_admission_threshold_bps"] = np.float32(ADMISSION_BPS)
    source["held_month"] = pd.to_datetime(source["__decision_ts__"], utc=True).dt.strftime("%Y-%m")
    monthly_rows: list[dict[str, Any]] = []
    for month, part in source.groupby("held_month", sort=True):
        valid = part.loc[_valid(part)].copy()
        event = _event(valid)
        p = pd.to_numeric(valid["opportunity_probability"], errors="coerce").to_numpy(float)
        order = np.argsort(p, kind="stable"); rank = np.empty(len(valid), dtype=float); rank[order] = (np.arange(len(valid)) + 1) / max(len(valid), 1)
        admitted = part.loc[pd.to_numeric(part["K0_expected_policy_net_bps"], errors="coerce").ge(ADMISSION_BPS)]
        known = admitted.loc[_valid(admitted)]
        net = r1._finite(known["policy_net_bps"]).to_numpy(float)
        row = {"arm": "A0_static_binary__uniform", "held_month": month, "valid_rows": len(valid), "event_prevalence": float(event.mean()) if len(event) else float("nan"), "auc": float(roc_auc_score(event, p)) if len(valid) and np.unique(event).size > 1 else float("nan"), "prauc": float(average_precision_score(event, p)) if len(valid) and np.unique(event).size > 1 else float("nan"), "brier": _brier(event, p), "integrated_brier": float("nan"), "time_auc_early_vs_late": float("nan"), "precision_top10": float(event[rank > .9].mean()) if len(event) else float("nan"), "precision_top20": float(event[rank > .8].mean()) if len(event) else float("nan"), "precision_top30": float(event[rank > .7].mean()) if len(event) else float("nan"), "admitted": len(admitted), "known_admitted": len(known), "net_bps_per_trade": float(net.mean()) if len(net) else float("nan"), "total_net_bps": float(net.sum()) if len(net) else 0.0, "cvar10_bps": r1._cvar(net), "positive_fraction": float((net > 0).mean()) if len(net) else float("nan")}
        monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)
    era = _era_metrics(monthly)
    later = era.loc[era["era"].isin(("2025", "2026"))].copy()
    known = later["known_admitted"].to_numpy(float)
    summary = pd.DataFrame([{"arm": "A0_static_binary__uniform", "net_2025": float(later.set_index("era").loc["2025", "net_bps_per_trade"]), "net_2026": float(later.set_index("era").loc["2026", "net_bps_per_trade"]), "mean_net_bps": _weighted_nanmean(later["net_bps_per_trade"].to_numpy(float), np.maximum(known, 1.0)), "total_net_bps": float(later["total_net_bps"].sum()), "worst_era_net_bps": float(later["net_bps_per_trade"].min()), "worst_month_net_bps": float(monthly.loc[monthly.held_month.str[:4].isin(("2025", "2026")), "net_bps_per_trade"].min()), "mean_auc": _weighted_nanmean(later["auc"].to_numpy(float), np.maximum(known, 1.0)), "mean_integrated_brier": float("nan"), "admitted": int(later["admitted"].sum()), "advances_phase_a": True}])
    out.mkdir(parents=True)
    source.to_parquet(out / "phase_a_static_control_predictions.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "phase_a_static_control_monthly_metrics.parquet", index=False, compression="zstd")
    era.to_parquet(out / "phase_a_static_control_era_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "phase_a_static_control_summary.parquet", index=False, compression="zstd")
    manifest = {"schema": SCHEMA, "status": "complete_static_control", "side": "short", "architecture": "frozen static O250/H6 (O45) -> frozen C59 -> original analytic K0", "evaluation_change_only": {"admission": f"K0 expected policy net >= {ADMISSION_BPS:g} bps", "source_predictions": str(STATIC_C59)}, "timing_event": "strict short MFE6h >250bps joined only for matched target metrics", "causality": "uses existing strict-prequential C59 OOF predictions; no score/model re-fit", "sources": {"c59_prediction_sha256": _sha256(STATIC_C59), "timing_manifest_sha256": _sha256(TIMING_ROOT / "run_manifest.json")}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _report(out, era, summary, manifest)
    return out


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    frame, o45, c_fields, hashes = _load()
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    months = pd.date_range("2024-05-01T00:00:00Z", "2026-08-01T00:00:00Z", freq="MS", inclusive="left")
    for spec_index, spec in enumerate(SPECS):
        for month_index, month in enumerate(months):
            try:
                pred, audit = _outer_month(frame, o45, c_fields, spec, month, SEED + spec_index * 10_000 + month_index * 101)
                predictions.append(pred); audits.append(audit); metrics.append(_month_metrics(pred))
            except ValueError as exc:
                audits.append({"arm": spec.name, "held_month": month.strftime("%Y-%m"), "status": "skipped", "reason": str(exc)})
    if not predictions:
        raise RuntimeError("Phase A produced no strict prequential predictions")
    monthly = pd.DataFrame(metrics)
    era = _era_metrics(monthly)
    later = era.loc[era["era"].isin(("2025", "2026"))].copy()
    rows: list[dict[str, Any]] = []
    for arm, group in later.groupby("arm", sort=True):
        weights = group["known_admitted"].to_numpy(float)
        rows.append({"arm": arm, "net_2025": float(group.set_index("era").loc["2025", "net_bps_per_trade"]) if "2025" in set(group.era) else float("nan"), "net_2026": float(group.set_index("era").loc["2026", "net_bps_per_trade"]) if "2026" in set(group.era) else float("nan"), "mean_net_bps": _weighted_nanmean(group["net_bps_per_trade"].to_numpy(float), np.maximum(weights, 1.0)), "total_net_bps": float(group["total_net_bps"].sum()), "worst_era_net_bps": float(group["net_bps_per_trade"].min()), "worst_month_net_bps": float(monthly.loc[monthly.arm.eq(arm) & monthly.held_month.str[:4].isin(("2025", "2026")), "net_bps_per_trade"].min()), "mean_auc": _weighted_nanmean(group["auc"].to_numpy(float), np.maximum(weights, 1.0)), "mean_integrated_brier": _weighted_nanmean(group["integrated_brier"].to_numpy(float), np.maximum(weights, 1.0)), "admitted": int(group["admitted"].sum())})
    summary = pd.DataFrame(rows).sort_values(["mean_net_bps", "worst_month_net_bps", "total_net_bps"], ascending=False, kind="stable").reset_index(drop=True)
    summary["advances_phase_a"] = summary["mean_net_bps"].ge(90.0) & summary["worst_era_net_bps"].gt(0.0) & summary["mean_auc"].gt(.5)
    out.mkdir(parents=True)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "phase_a_outer_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "phase_a_fold_audit.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "phase_a_monthly_metrics.parquet", index=False, compression="zstd")
    era.to_parquet(out / "phase_a_era_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "phase_a_summary.parquet", index=False, compression="zstd")
    manifest = {"schema": SCHEMA, "status": "complete", "side": "short", "scope": "Phase A only; no canonical/live change", "period": {"start": "2024-05", "end_exclusive": "2026-08"}, "architecture": "frozen P0/F90 -> shared discrete-time time-to-event O45 -> frozen C59 -> analytic K0", "opportunity": {"event": "short MFE within six hours >250 bps", "geometries": {key: list(value) for key, value in BASE_GEOMETRIES.items()}, "weights": ["uniform", "early (<1h 1.25; 1-2h 1.15; 2-4h 1.0; 4-6h .85)"]}, "admission": {"expected_policy_net_bps_gte": ADMISSION_BPS}, "features": {"O45": list(o45), "C59": list(c_fields)}, "causality": {"outer_and_inner": "label_available_at < validation decision", "candidates": "all target-free P0 candidates are scored", "invalidity": "incomplete labels never become economic failures"}, "sources": hashes}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _report(out, era, summary, manifest)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--summary-from", type=Path, default=None, help="rebuild only metrics/report from immutable prediction artifact")
    parser.add_argument("--static-control", action="store_true", help="evaluate frozen static O250/H6+C59 at the Phase-A +75-bps gate")
    args = parser.parse_args()
    if args.static_control:
        if args.summary_from is not None:
            raise ValueError("--static-control and --summary-from are exclusive")
        print(build_static_control(args.out))
        return
    if args.summary_from is None:
        print(run(args.out))
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    source = args.summary_from
    prediction = pd.read_parquet(source / "phase_a_outer_oof_predictions.parquet")
    monthly = pd.DataFrame([_month_metrics(part) for _key, part in prediction.groupby(["arm", "held_month"], sort=True)])
    era = _era_metrics(monthly)
    later = era.loc[era["era"].isin(("2025", "2026"))].copy()
    rows: list[dict[str, Any]] = []
    for arm, group in later.groupby("arm", sort=True):
        weights = group["known_admitted"].to_numpy(float)
        later_months = monthly.loc[monthly.arm.eq(arm) & monthly.held_month.str[:4].isin(("2025", "2026"))]
        row = {"arm": arm, "net_2025": float(group.set_index("era").loc["2025", "net_bps_per_trade"]) if "2025" in set(group.era) else float("nan"), "net_2026": float(group.set_index("era").loc["2026", "net_bps_per_trade"]) if "2026" in set(group.era) else float("nan"), "mean_net_bps": _weighted_nanmean(group["net_bps_per_trade"].to_numpy(float), np.maximum(weights, 1.0)), "total_net_bps": float(group["total_net_bps"].sum()), "worst_era_net_bps": float(group["net_bps_per_trade"].min()), "worst_month_net_bps": float(later_months["net_bps_per_trade"].min()), "mean_auc": _weighted_nanmean(group["auc"].to_numpy(float), np.maximum(weights, 1.0)), "mean_integrated_brier": _weighted_nanmean(group["integrated_brier"].to_numpy(float), np.maximum(weights, 1.0)), "admitted": int(group["admitted"].sum())}
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["mean_net_bps", "worst_month_net_bps", "total_net_bps"], ascending=False, kind="stable").reset_index(drop=True)
    summary["advances_phase_a"] = summary["mean_net_bps"].ge(90.0) & summary["worst_era_net_bps"].gt(0.0) & summary["mean_auc"].gt(.5)
    args.out.mkdir(parents=True)
    for name in ("phase_a_outer_oof_predictions.parquet", "phase_a_fold_audit.parquet", "run_manifest.json"):
        shutil.copy2(source / name, args.out / name)
    monthly.to_parquet(args.out / "phase_a_monthly_metrics.parquet", index=False, compression="zstd")
    era.to_parquet(args.out / "phase_a_era_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "phase_a_summary.parquet", index=False, compression="zstd")
    manifest = json.loads((source / "run_manifest.json").read_text())
    manifest["status"] = "complete_summary_repair"
    manifest["summary_repair"] = {"source": str(source), "reason": "exclude zero-known-admission months from weighted per-trade aggregates; raw OOF scores unchanged"}
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _report(args.out, era, summary, manifest)
    print(args.out)


if __name__ == "__main__":
    main()
