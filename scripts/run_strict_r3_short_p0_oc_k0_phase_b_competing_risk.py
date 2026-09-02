#!/usr/bin/env python3
"""Phase B: strict-prequential competing-risk O for the short P0 -> O -> C -> K0 funnel.

This is deliberately a sequential, research-only stage.  It uses the chosen
Phase-A 1-hour discrete intervals and compares only the declared adverse
barriers (1.5/2.0/3.0 ATR) and two valid competing-risk constructions:

* a shared multinomial discrete hazard; and
* two cause-specific discrete hazards.

The first pass keeps the frozen C59 training population (B3a).  Its summary
selects at most two candidate O contracts.  ``--c-consistency`` can then run
the B3a/B3b C-population comparison *only* for those explicitly selected
contracts.  Every outer score, calibration input and conditional C row is
strictly prequential; labels are never scoring features.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round2 as r2  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair as c59  # noqa: E402
import run_strict_r3_short_p0_oc_k0_phase_a_timing as phase_a  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_phase_b_competing_risk_v1"
TIMING_ROOT = phase_a.TIMING_ROOT
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_b_competing_risk_202408_202607_20260822_v1"
SEED = 1729
MIN_C_ROWS = phase_a.MIN_C_ROWS
MIN_OOF_ROWS = phase_a.MIN_OOF_ROWS
MIN_OOF_MONTHS = phase_a.MIN_OOF_MONTHS
ADMISSION_BPS = phase_a.ADMISSION_BPS
POLICY_CLIP_BPS = phase_a.POLICY_CLIP_BPS
INTERVALS = ((0, 60), (60, 120), (120, 180), (180, 240), (240, 360))
ADVERSE = ("1p5", "2p0", "3p0")


@dataclass(frozen=True)
class Arm:
    formulation: str  # multinomial | cause_specific
    adverse: str      # 1p5 | 2p0 | 3p0
    c_contract: str   # B3a_all_mfe | B3b_favourable_before_adverse

    @property
    def name(self) -> str:
        return f"B_{self.formulation}__adverse_{self.adverse}atr__{self.c_contract}"


@dataclass
class K0Bundle:
    calibration: r2.ProbabilityCalibrator
    mu1: IsotonicRegression
    mu0: float
    oof_rows: int
    oof_months: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid(frame: pd.DataFrame) -> pd.Series:
    return (
        r1._valid_label(frame)
        & frame["event_timing_label_valid"].fillna(False).astype(bool)
        & ~frame["event_timing_target_invalid"].fillna(True).astype(bool)
    )


def _field(arm: Arm, stem: str) -> str:
    return f"{stem}_{arm.adverse}atr"


def _event(frame: pd.DataFrame, arm: Arm) -> np.ndarray:
    return pd.to_numeric(frame[_field(arm, "favourable_first")], errors="coerce").fillna(0).astype(int).to_numpy()


def _adverse(frame: pd.DataFrame, arm: Arm) -> np.ndarray:
    return pd.to_numeric(frame[_field(arm, "adverse_first")], errors="coerce").fillna(0).astype(int).to_numpy()


def _event_class(frame: pd.DataFrame, arm: Arm) -> np.ndarray:
    # 0 censor/no event; 1 favourable first; 2 adverse first.  Exact same-bar
    # ties were already assigned to adverse-first in the label materialiser.
    result = np.zeros(len(frame), dtype=np.int8)
    result[_event(frame, arm).astype(bool)] = 1
    result[_adverse(frame, arm).astype(bool)] = 2
    return result


def _favourable_time(frame: pd.DataFrame, arm: Arm) -> np.ndarray:
    value = pd.to_numeric(frame["first_favourable_250bps_minute"], errors="coerce").to_numpy(float)
    value[~_event(frame, arm).astype(bool)] = np.nan
    return value


def _adverse_time(frame: pd.DataFrame, arm: Arm) -> np.ndarray:
    return pd.to_numeric(frame[_field(arm, "first_adverse") + "_minute"], errors="coerce").to_numpy(float)


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
        "first_favourable_250bps_minute", *[
            f"{prefix}_{tag}atr{suffix}"
            for tag in ADVERSE
            for prefix, suffix in (("first_adverse", "_minute"), ("favourable_first", ""), ("adverse_first", ""), ("censored", ""))
        ],
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


def _risk_rows(frame: pd.DataFrame, arm: Arm) -> tuple[pd.DataFrame, np.ndarray]:
    """Candidate-period expansion through first favourable/adverse/censor event."""
    cls = _event_class(frame, arm)
    first_f = _favourable_time(frame, arm)
    first_a = _adverse_time(frame, arm)
    first = np.fmin(np.where(np.isfinite(first_f), first_f, np.inf), np.where(np.isfinite(first_a), first_a, np.inf))
    parts: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    for index, (start, end) in enumerate(INTERVALS):
        at_risk = first > float(start)
        part = frame.loc[at_risk].copy()
        part["__time_bin__"] = np.int16(index)
        parts.append(part)
        current = np.zeros(int(at_risk.sum()), dtype=np.int8)
        hit = first[at_risk] <= float(end)
        current[hit] = cls[at_risk][hit]
        targets.append(current)
    return pd.concat(parts, ignore_index=True), np.concatenate(targets)


def _hazard_params(objective: str, seed: int) -> dict[str, Any]:
    return dict(
        objective=objective, n_estimators=180, learning_rate=.035, max_depth=3,
        num_leaves=15, min_child_samples=40, subsample=.85, subsample_freq=1,
        colsample_bytree=.85, reg_lambda=4.0, reg_alpha=.10, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _fit_hazards(train: pd.DataFrame, fields: tuple[str, ...], arm: Arm, seed: int) -> tuple[object, pd.Series]:
    expanded, y = _risk_rows(train, arm)
    x, medians = r1._matrix(expanded, fields)
    x["__time_bin__"] = expanded["__time_bin__"].to_numpy(np.int16)
    if arm.formulation == "multinomial":
        model = LGBMClassifier(**_hazard_params("multiclass", seed), num_class=3)
        model.fit(x, y, categorical_feature=["__time_bin__"])
        return model, medians
    if arm.formulation == "cause_specific":
        favourable = LGBMClassifier(**_hazard_params("binary", seed))
        adverse = LGBMClassifier(**_hazard_params("binary", seed + 1))
        favourable.fit(x, (y == 1).astype(np.int8), categorical_feature=["__time_bin__"])
        adverse.fit(x, (y == 2).astype(np.int8), categorical_feature=["__time_bin__"])
        return (favourable, adverse), medians
    raise ValueError(arm.formulation)


def _hazards_for_bin(model: object, x: pd.DataFrame, arm: Arm) -> tuple[np.ndarray, np.ndarray]:
    if arm.formulation == "multinomial":
        probability = model.predict_proba(x)  # type: ignore[union-attr]
        return np.asarray(probability[:, 1], dtype=float), np.asarray(probability[:, 2], dtype=float)
    favourable, adverse = model  # type: ignore[misc]
    return np.asarray(favourable.predict_proba(x)[:, 1], dtype=float), np.asarray(adverse.predict_proba(x)[:, 1], dtype=float)


def _predict_cif(model: object, held: pd.DataFrame, fields: tuple[str, ...], medians: pd.Series, arm: Arm) -> dict[str, np.ndarray]:
    survival = np.ones(len(held), dtype=float)
    cif = np.zeros(len(held), dtype=float)
    output: dict[str, np.ndarray] = {}
    for index, (start, end) in enumerate(INTERVALS):
        x, _ = r1._matrix(held, fields, medians)
        x = x.copy(); x["__time_bin__"] = np.int16(index)
        h_f, h_a = _hazards_for_bin(model, x, arm)
        h_f = np.clip(h_f, 1e-7, 1.0 - 1e-7)
        h_a = np.clip(h_a, 1e-7, 1.0 - 1e-7)
        total = h_f + h_a
        # Cause-specific binary heads can marginally exceed a unit discrete
        # hazard.  Projection preserves their odds while retaining a valid
        # competing-risk survival process.
        over = total >= .999
        if over.any():
            h_f[over] *= .999 / total[over]
            h_a[over] *= .999 / total[over]
        before_survival = survival.copy()
        before_cif = cif.copy()
        cif += before_survival * h_f
        survival *= 1.0 - h_f - h_a
        for horizon in (1, 2, 4, 6):
            minute = horizon * 60
            if start < minute <= end:
                fraction = (minute - start) / float(end - start)
                # No post-entry data exist inside a bin.  Constant hazard
                # interpolation is therefore the only causal within-bin view.
                total_h = h_f + h_a
                partial_event = 1.0 - np.power(1.0 - total_h, fraction)
                share_f = h_f / np.maximum(total_h, 1e-12)
                output[f"p_event_{horizon}h"] = (before_cif + before_survival * partial_event * share_f).astype(np.float32)
    if sorted(output) != ["p_event_1h", "p_event_2h", "p_event_4h", "p_event_6h"]:
        raise AssertionError(f"interval checkpoints absent: {sorted(output)}")
    output["opportunity_raw_score"] = output["p_event_6h"].copy()
    return output


def _c_population(frame: pd.DataFrame, arm: Arm) -> pd.Series:
    if arm.c_contract == "B3a_all_mfe":
        return pd.Series(r1._event(frame, r3.SPEC).astype(bool), index=frame.index)
    if arm.c_contract == "B3b_favourable_before_adverse":
        return pd.Series(_event(frame, arm).astype(bool), index=frame.index)
    raise ValueError(arm.c_contract)


def _fit_k0(inner: pd.DataFrame) -> K0Bundle:
    event = inner["event_target"].astype(int).to_numpy()
    if int(event.sum()) < MIN_C_ROWS:
        raise ValueError("insufficient true competing-risk O-positive timing OOF support")
    y = r1._finite(inner["policy_net_bps"]).clip(-POLICY_CLIP_BPS, POLICY_CLIP_BPS).to_numpy(float)
    calibration = r2._fit_probability("platt", inner["opp_oof_raw"].to_numpy(float), event)
    mu1, _ = r1._fit_isotonic(inner.loc[event.astype(bool), "conversion_oof_raw"].to_numpy(float), y[event.astype(bool)], -POLICY_CLIP_BPS, POLICY_CLIP_BPS)
    negative = ~event.astype(bool)
    global_mean = float(np.mean(y))
    mu0 = float((y[negative].sum() + 500.0 * global_mean) / (negative.sum() + 500.0))
    return K0Bundle(calibration, mu1, mu0, len(inner), int(inner["__decision_ts__"].dt.strftime("%Y-%m").nunique()))


def _apply_k0(bundle: K0Bundle, raw_o: np.ndarray, raw_c: np.ndarray) -> pd.DataFrame:
    probability = bundle.calibration.predict(raw_o)
    expected = probability * np.asarray(bundle.mu1.predict(raw_c), dtype=float) + (1.0 - probability) * bundle.mu0
    return pd.DataFrame({
        "opportunity_probability": probability.astype(np.float32),
        "conversion_score": np.asarray(raw_c, dtype=np.float32),
        "K0_expected_policy_net_bps": expected.astype(np.float32),
        "K0_admission_threshold_bps": np.full(len(expected), ADMISSION_BPS, dtype=np.float32),
    })


def _fit_c_predict(fit: pd.DataFrame, valid: pd.DataFrame, c_fields: tuple[str, ...], arm: Arm, seed: int) -> tuple[np.ndarray, int]:
    c_fit = fit.loc[_c_population(fit, arm)].copy()
    target = next(value for value in r3.TARGETS if value.name == "C3_normalized_regret")
    if len(c_fit) < MIN_C_ROWS or r1._month_count(c_fit) < MIN_OOF_MONTHS:
        raise ValueError("insufficient declared C-population support")
    y = r3._target(c_fit, target)
    if np.unique(y).size < 2:
        raise ValueError("declared C target has only one class")
    x_fit, medians = r1._matrix(c_fit, c_fields)
    x_valid, _ = r1._matrix(valid, c_fields, medians)
    model = r3._model(target, seed)
    model.fit(x_fit, y, sample_weight=r3._c_weights(c_fit, "uniform"))
    return r3._predict(model, target, x_valid), len(c_fit)


def _inner_oof(train: pd.DataFrame, o_fields: tuple[str, ...], c_fields: tuple[str, ...], arm: Arm, held_month: str, seed: int) -> pd.DataFrame:
    local = train.loc[_valid(train)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    bounds = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    pieces: list[pd.DataFrame] = []
    for fold in range(r1.INNER_SPLITS):
        valid = local.iloc[int(bounds[fold + 1]):int(bounds[fold + 2])].copy()
        if valid.empty:
            continue
        cutoff = valid["__decision_ts__"].min()
        fit = local.loc[local["__label_available_at__"].lt(cutoff)].copy()
        if len(fit) < r1.MIN_OUTER_TRAIN_ROWS or np.unique(_event(fit, arm)).size < 2:
            continue
        try:
            raw_c, _ = _fit_c_predict(fit, valid, c_fields, arm, seed + 10_000 + fold)
        except ValueError:
            continue
        model, medians = _fit_hazards(fit, o_fields, arm, seed + fold)
        raw_o = _predict_cif(model, valid, o_fields, medians, arm)["opportunity_raw_score"]
        part = valid.loc[:, [*r1.IDENTITY, "__label_available_at__", "policy_net_bps", "policy_regret_bps", "policy_gross_bps"]].copy()
        part["opp_oof_raw"] = raw_o.astype(np.float32)
        part["conversion_oof_raw"] = raw_c.astype(np.float32)
        part["event_target"] = _event(valid, arm).astype(np.int8)
        part["held_month"] = held_month
        pieces.append(part)
    if not pieces:
        raise ValueError("no purged inner competing-risk OOF support")
    output = pd.concat(pieces, ignore_index=True)
    if len(output) < MIN_OOF_ROWS or output["__decision_ts__"].dt.strftime("%Y-%m").nunique() < MIN_OOF_MONTHS:
        raise ValueError("insufficient combined competing-risk/C OOF support")
    return output


def _outer_month(frame: pd.DataFrame, o_fields: tuple[str, ...], c_fields: tuple[str, ...], arm: Arm, month: pd.Timestamp, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = month + pd.offsets.MonthBegin(1)
    held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(end)].copy()
    train = frame.loc[frame["__decision_ts__"].lt(month) & frame["__label_available_at__"].lt(month) & _valid(frame)].copy()
    if held.empty:
        raise ValueError("empty held month")
    inner = _inner_oof(train, o_fields, c_fields, arm, month.strftime("%Y-%m"), seed)
    bundle = _fit_k0(inner)
    model, medians = _fit_hazards(train, o_fields, arm, seed + 20_000)
    raw_o = _predict_cif(model, held, o_fields, medians, arm)
    raw_c, c_rows = _fit_c_predict(train, held, c_fields, arm, seed + 30_000)
    columns = [*r1.IDENTITY, "__label_available_at__", "policy_net_bps", "policy_regret_bps", "policy_gross_bps", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid", "event_timing_label_valid", "event_timing_target_invalid", "first_favourable_250bps_minute", _field(arm, "first_adverse") + "_minute", _field(arm, "favourable_first"), _field(arm, "adverse_first"), _field(arm, "censored")]
    output = held.loc[:, columns].copy().reset_index(drop=True)
    for name, values in raw_o.items():
        output[name] = values
    output = pd.concat((output, _apply_k0(bundle, raw_o["opportunity_raw_score"], raw_c)), axis=1)
    output["held_month"] = month.strftime("%Y-%m")
    output["arm"] = arm.name
    audit = {"arm": arm.name, "held_month": month.strftime("%Y-%m"), "status": "complete", "held_rows": len(held), "outer_train_rows": len(train), "outer_c_rows": c_rows, "inner_oof_rows": bundle.oof_rows, "inner_oof_months": bundle.oof_months, "k0_mu0_bps": bundle.mu0}
    return output, audit


def _brier(y: np.ndarray, probability: np.ndarray) -> float:
    return float(brier_score_loss(y, np.clip(probability, 1e-6, 1.0 - 1e-6))) if len(y) and np.unique(y).size > 1 else float("nan")


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(y) < 20 or np.unique(y).size < 2 or np.unique(p).size < 2:
        return float("nan"), float("nan")
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(logit, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _time_target(frame: pd.DataFrame, arm_name: str, horizon: int) -> np.ndarray:
    adverse = arm_name.split("__adverse_")[1].split("atr", 1)[0]
    fav = pd.to_numeric(frame[f"favourable_first_{adverse}atr"], errors="coerce").fillna(0).astype(bool).to_numpy()
    minute = pd.to_numeric(frame["first_favourable_250bps_minute"], errors="coerce").to_numpy(float)
    return (fav & np.isfinite(minute) & (minute <= horizon * 60)).astype(np.int8)


def _month_metrics(part: pd.DataFrame) -> dict[str, Any]:
    valid = part.loc[_valid(part)].copy()
    arm = str(part["arm"].iloc[0])
    event = _event(valid, Arm("multinomial", arm.split("__adverse_")[1].split("atr", 1)[0], "B3a_all_mfe"))
    # Event only depends on the adverse barrier, never on hazard formulation/C.
    probability = valid["opportunity_probability"].to_numpy(float)
    slope, intercept = _calibration(event, probability)
    row: dict[str, Any] = {
        "arm": arm, "held_month": str(part["held_month"].iloc[0]), "valid_rows": len(valid),
        "favourable_first_prevalence": float(event.mean()) if len(event) else float("nan"),
        "adverse_first_prevalence": float(pd.to_numeric(valid[_field(Arm("multinomial", arm.split("__adverse_")[1].split("atr", 1)[0], "B3a_all_mfe"), "adverse_first")], errors="coerce").fillna(0).mean()) if len(valid) else float("nan"),
        "censored_prevalence": float(pd.to_numeric(valid[_field(Arm("multinomial", arm.split("__adverse_")[1].split("atr", 1)[0], "B3a_all_mfe"), "censored")], errors="coerce").fillna(0).mean()) if len(valid) else float("nan"),
        "auc": float(roc_auc_score(event, probability)) if len(valid) and np.unique(event).size > 1 else float("nan"),
        "prauc": float(average_precision_score(event, probability)) if len(valid) and np.unique(event).size > 1 else float("nan"),
        "brier": _brier(event, probability), "calibration_slope": slope, "calibration_intercept": intercept,
    }
    order = np.argsort(probability, kind="stable"); rank = np.empty(len(valid), dtype=float); rank[order] = (np.arange(len(valid)) + 1) / max(len(valid), 1)
    for fraction in (.10, .20, .30):
        selected = rank > 1.0 - fraction
        precision = float(event[selected].mean()) if selected.any() else float("nan")
        row[f"precision_top{int(fraction * 100)}"] = precision
        row[f"lift_top{int(fraction * 100)}"] = precision / max(float(event.mean()), 1e-12) if np.isfinite(precision) else float("nan")
    briers: list[float] = []
    for horizon in (1, 2, 4, 6):
        target = _time_target(valid, arm, horizon)
        value = _brier(target, valid[f"p_event_{horizon}h"].to_numpy(float))
        row[f"brier_{horizon}h"] = value; briers.append(value)
    row["integrated_brier"] = float(np.nanmean(briers))
    first = pd.to_numeric(valid["first_favourable_250bps_minute"], errors="coerce").to_numpy(float)
    first[~event.astype(bool)] = np.nan
    hit = np.isfinite(first)
    early = (first[hit] <= 120.0).astype(int)
    row["time_auc_early_vs_late"] = float(roc_auc_score(early, valid.loc[hit, "p_event_2h"].to_numpy(float))) if hit.sum() and np.unique(early).size > 1 else float("nan")
    admitted = part.loc[pd.to_numeric(part["K0_expected_policy_net_bps"], errors="coerce").ge(ADMISSION_BPS)]
    known = admitted.loc[_valid(admitted)]
    net = r1._finite(known["policy_net_bps"]).to_numpy(float)
    row.update({
        "admitted": len(admitted), "known_admitted": len(known),
        "net_bps_per_trade": float(net.mean()) if len(net) else float("nan"),
        "total_net_bps": float(net.sum()) if len(net) else 0.0,
        "cvar10_bps": r1._cvar(net), "p_net_lt_neg200": float((net < -200.0).mean()) if len(net) else float("nan"),
        "p_net_lt_neg400": float((net < -400.0).mean()) if len(net) else float("nan"),
        "positive_fraction": float((net > 0.0).mean()) if len(net) else float("nan"),
    })
    for kind, selector in (("favourable", event.astype(bool)), ("adverse", _adverse(valid, Arm("multinomial", arm.split("__adverse_")[1].split("atr", 1)[0], "B3a_all_mfe")).astype(bool)), ("censored", ~event.astype(bool) & ~_adverse(valid, Arm("multinomial", arm.split("__adverse_")[1].split("atr", 1)[0], "B3a_all_mfe")).astype(bool))):
        values = r1._finite(valid.loc[selector, "policy_net_bps"]).to_numpy(float)
        row[f"policy_net_{kind}_bps"] = float(values.mean()) if len(values) else float("nan")
    return row


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float); weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() else float("nan")


def _era_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, era), group in monthly.assign(era=monthly["held_month"].str[:4]).groupby(["arm", "era"], sort=True):
        weights = np.maximum(group["known_admitted"].to_numpy(float), 1.0)
        row: dict[str, Any] = {"arm": arm, "era": era, "months": len(group), "admitted": int(group["admitted"].sum()), "known_admitted": int(group["known_admitted"].sum()), "total_net_bps": float(group["total_net_bps"].sum()), "positive_months": int((group["net_bps_per_trade"] > 0).sum()), "worst_month_net_bps": float(group["net_bps_per_trade"].min())}
        for field in ("auc", "prauc", "brier", "calibration_slope", "calibration_intercept", "precision_top10", "precision_top20", "precision_top30", "lift_top10", "lift_top20", "lift_top30", "integrated_brier", "time_auc_early_vs_late", "net_bps_per_trade", "cvar10_bps", "p_net_lt_neg200", "p_net_lt_neg400", "positive_fraction", "favourable_first_prevalence", "adverse_first_prevalence", "censored_prevalence", "policy_net_favourable_bps", "policy_net_adverse_bps", "policy_net_censored_bps"):
            row[field] = _weighted_nanmean(group[field].to_numpy(float), weights)
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(monthly: pd.DataFrame, era: pd.DataFrame) -> pd.DataFrame:
    later = era.loc[era["era"].isin(("2025", "2026"))].copy()
    rows: list[dict[str, Any]] = []
    for arm, group in later.groupby("arm", sort=True):
        weights = np.maximum(group["known_admitted"].to_numpy(float), 1.0)
        months = monthly.loc[monthly.arm.eq(arm) & monthly.held_month.str[:4].isin(("2025", "2026"))]
        by_era = group.set_index("era")
        row = {
            "arm": arm,
            "net_2025": float(by_era.loc["2025", "net_bps_per_trade"]) if "2025" in by_era.index else float("nan"),
            "net_2026": float(by_era.loc["2026", "net_bps_per_trade"]) if "2026" in by_era.index else float("nan"),
            "mean_net_bps": _weighted_nanmean(group["net_bps_per_trade"].to_numpy(float), weights),
            "total_net_bps": float(group["total_net_bps"].sum()),
            "worst_era_net_bps": float(group["net_bps_per_trade"].min()),
            "worst_month_net_bps": float(months["net_bps_per_trade"].min()),
            "mean_auc": _weighted_nanmean(group["auc"].to_numpy(float), weights),
            "precision_top20": _weighted_nanmean(group["precision_top20"].to_numpy(float), weights),
            "cvar10_bps": _weighted_nanmean(group["cvar10_bps"].to_numpy(float), weights),
            "admitted": int(group["admitted"].sum()),
        }
        row["advances_phase_b"] = bool(row["mean_net_bps"] >= 90.0 and row["worst_era_net_bps"] > 0.0 and row["mean_auc"] > .5)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_net_bps", "worst_month_net_bps", "total_net_bps"], ascending=False, kind="stable").reset_index(drop=True)


def _table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        cols = [str(value) for value in frame.columns]
        return "\n".join(["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |", *("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))])


def _report(out: Path, era: pd.DataFrame, summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    text = [
        "# Short P0 → competing-risk O → C → K0: Phase B", "",
        "Research-only.  P0/F90, O45, C59 field contract and analytic K0 remain fixed.  This report keeps strict-prequential 2024 (valid October–December output), 2025, and 2026 (January–July) separate.",
        "", "## Era metrics", "", _table(era), "", "## Sequential selection", "", _table(summary), "",
        "## Contract", "", "- Favourable: first short-favourable move >250 bps within six hours.", "- Competing adverse move: first high excursion >=1.5/2.0/3.0 ATR; same-minute tie is adverse-first.", "- O: selected Phase-A 1-hour bins, multinomial or cause-specific discrete competing-risk hazards.", "- C: B3a all-MFE control or B3b matching favourable-before-adverse population, named in every arm.", "- K0: analytic p(O)×mu1(C)+(1-p(O))×mu0(P0 anchor), frozen +75 bps admission.", "", "```json", json.dumps(manifest, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_PHASE_B_COMPETING_RISK_REPORT.md").write_text("\n".join(text))


def run(out: Path, arms: tuple[Arm, ...], months: pd.DatetimeIndex) -> Path:
    if out.exists():
        raise FileExistsError(out)
    frame, o_fields, c_fields, hashes = _load()
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(arms):
        for month_index, month in enumerate(months):
            try:
                prediction, audit = _outer_month(frame, o_fields, c_fields, arm, month, SEED + arm_index * 20_000 + month_index * 101)
                predictions.append(prediction); audits.append(audit); metrics.append(_month_metrics(prediction))
            except ValueError as exc:
                audits.append({"arm": arm.name, "held_month": month.strftime("%Y-%m"), "status": "skipped", "reason": str(exc)})
    if not predictions:
        raise RuntimeError("Phase B produced no strict prequential predictions")
    monthly = pd.DataFrame(metrics)
    era = _era_metrics(monthly)
    summary = _summary(monthly, era)
    out.mkdir(parents=True)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "phase_b_outer_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "phase_b_fold_audit.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "phase_b_monthly_metrics.parquet", index=False, compression="zstd")
    era.to_parquet(out / "phase_b_era_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "phase_b_summary.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short", "scope": "Phase B only; no canonical/live change",
        "period": {"candidate_start": "2024-05", "output_supported_start": "2024-10", "end_exclusive": "2026-08"},
        "arms": [arm.name for arm in arms], "intervals_minutes": [list(value) for value in INTERVALS],
        "admission": {"expected_policy_net_bps_gte": ADMISSION_BPS},
        "features": {"O45": list(o_fields), "C59": list(c_fields)},
        "causality": {"outer_and_inner": "label_available_at < validation decision", "candidates": "all target-free P0 candidates scored", "invalidity": "incomplete paths excluded only from fitting/metrics", "same_bar_ties": "adverse-first"},
        "sources": hashes,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _report(out, era, summary, manifest)
    return out


def _parse_arm(value: str, c_contract: str) -> Arm:
    try:
        formulation, adverse = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("arm must be multinomial:1p5 or cause_specific:2p0") from exc
    if formulation not in ("multinomial", "cause_specific") or adverse not in ADVERSE:
        raise argparse.ArgumentTypeError("invalid competing-risk arm")
    return Arm(formulation, adverse, c_contract)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--months", nargs="*", default=None, help="optional YYYY-MM months; default full May-2024..Jul-2026 ledger")
    parser.add_argument("--c-consistency", action="store_true", help="run B3a/B3b comparison only for --arm inputs")
    parser.add_argument("--arm", action="append", default=[], help="formulation:adverse; repeatable")
    args = parser.parse_args()
    if args.months:
        months = pd.DatetimeIndex([pd.Timestamp(value + "-01T00:00:00Z") for value in args.months])
    else:
        months = pd.date_range("2024-05-01T00:00:00Z", "2026-08-01T00:00:00Z", freq="MS", inclusive="left")
    if args.arm:
        contracts = ("B3a_all_mfe", "B3b_favourable_before_adverse") if args.c_consistency else ("B3a_all_mfe",)
        arms = tuple(_parse_arm(value, contract) for contract in contracts for value in args.arm)
    elif args.c_consistency:
        raise ValueError("--c-consistency requires explicitly selected --arm values")
    else:
        # Predeclared first B pass.  No B3b C retrain is attempted until the
        # two downstream O contracts have been selected from these six arms.
        arms = tuple(Arm(formulation, adverse, "B3a_all_mfe") for formulation in ("multinomial", "cause_specific") for adverse in ADVERSE)
    print(run(args.out, arms, months))


if __name__ == "__main__":
    main()
