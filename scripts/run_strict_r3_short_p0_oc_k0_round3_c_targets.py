#!/usr/bin/env python3
"""Round 3A: strict-OOF conditional conversion-target funnel.

O is frozen to the Round-2 winner (O250_H6, stable-45, uniform, Platt).  This
runner changes exactly one component: the C target among true O-positive
training rows.  K0 retains its analytic p(O)·mu1(C)+(1-p(O))·mu0 form.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round2 as r2  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_round3_c_targets_v1"
ROUND2 = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round2_20260821_v1"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3_c_targets_20260821_v1"
SPEC = r1.OpportunitySpec("O250_H6", 6, 250.0)
SEED = 1729
MIN_C_ROWS = 500


@dataclass(frozen=True)
class CTarget:
    name: str
    kind: str  # classifier or regressor
    classes: int | None
    description: str


TARGETS = (
    CTarget("C0_t5_regret", "classifier", 5, "frozen current T5 low-regret ordinal"),
    CTarget("C1_policy_net_ordinal", "classifier", 6, "conditional policy-net ordinal: <-200, -200..0, 0..100, 100..250, 250..500, >500"),
    CTarget("C2_capture_efficiency", "regressor", None, "winsorized conditional policy gross / opportunity MFE"),
    CTarget("C3_normalized_regret", "classifier", 5, "ordinalized negative normalized regret, higher is more harvestable"),
    CTarget("C4_hybrid_quality", "regressor", None, "0.5 rank(policy net) + 0.5 rank(-normalized regret) within conditional training rows"),
)


def _o_fields() -> tuple[str, ...]:
    mda = pd.read_parquet(ROUND2 / "round2_target_specific_stability_mda.parquet")
    return r2._stable_prefixes(mda.loc[mda["arm"].eq(SPEC.name)].copy())[45]


def _mfe(frame: pd.DataFrame) -> np.ndarray:
    return r1._finite(frame[SPEC.label_field]).to_numpy(float)


def _norm_regret(frame: pd.DataFrame) -> np.ndarray:
    mfe = np.maximum(_mfe(frame), 1.0)
    gross = r1._finite(frame["policy_gross_bps"]).to_numpy(float)
    return np.clip((mfe - gross) / mfe, -1.0, 2.0)


def _target(frame: pd.DataFrame, target: CTarget) -> np.ndarray:
    if target.name == "C0_t5_regret":
        return r1._conversion_grade(frame)
    if target.name == "C1_policy_net_ordinal":
        return np.digitize(r1._finite(frame["policy_net_bps"]).to_numpy(float), [-200.0, 0.0, 100.0, 250.0, 500.0], right=False).astype(np.int8)
    if target.name == "C2_capture_efficiency":
        return np.clip(r1._finite(frame["policy_gross_bps"]).to_numpy(float) / np.maximum(_mfe(frame), 1.0), -1.0, 1.0)
    if target.name == "C3_normalized_regret":
        regret = _norm_regret(frame)
        # Low regret is good.  The outcome is a five-state monotone ordinal.
        return (4 - np.digitize(regret, [.25, .50, .75, 1.10], right=False)).astype(np.int8)
    if target.name == "C4_hybrid_quality":
        net_rank = pd.Series(r1._finite(frame["policy_net_bps"]).to_numpy(float)).rank(method="average", pct=True).to_numpy(float)
        quality_rank = pd.Series(-_norm_regret(frame)).rank(method="average", pct=True).to_numpy(float)
        return .5 * net_rank + .5 * quality_rank
    raise AssertionError(target.name)


def _c_weights(frame: pd.DataFrame, kind: str) -> np.ndarray:
    """Conditional-C authority, bounded to preserve calibration."""
    weight = np.ones(len(frame), dtype=float)
    if kind in {"equal_month", "equal_month_mfe"}:
        month = frame["__decision_ts__"].dt.strftime("%Y-%m")
        counts = month.value_counts()
        weight *= month.map(len(frame) / counts).to_numpy(float)
    if kind in {"equal_mfe", "equal_month_mfe"}:
        bucket = pd.Series(pd.cut(_mfe(frame), [SPEC.threshold_bps, 350.0, 500.0, 800.0, np.inf], include_lowest=True).astype(str), index=frame.index)
        counts = bucket.value_counts()
        weight *= bucket.map(len(frame) / counts).to_numpy(float)
    if kind not in {"uniform", "equal_month", "equal_mfe", "equal_month_mfe"}:
        raise ValueError(kind)
    weight = weight / max(float(np.mean(weight)), 1e-9)
    return np.clip(weight, .5, 2.0)


def _model(target: CTarget, seed: int, params: dict[str, Any] | None = None):
    common = dict(n_estimators=180, learning_rate=.035, max_depth=3, num_leaves=15, min_child_samples=40, subsample=.85, subsample_freq=1, colsample_bytree=.85, reg_lambda=4.0, reg_alpha=.10, random_state=seed, n_jobs=-1, verbosity=-1)
    if params:
        common.update(params)
    common["random_state"] = seed
    common.setdefault("n_jobs", -1)
    common.setdefault("verbosity", -1)
    if target.kind == "classifier":
        return LGBMClassifier(objective="multiclass", num_class=int(target.classes), class_weight="balanced", **common)
    return LGBMRegressor(objective="huber", alpha=.9, **common)


def _predict(model, target: CTarget, x: pd.DataFrame) -> np.ndarray:
    if target.kind == "classifier":
        return np.asarray(model.predict_proba(x) @ np.arange(int(target.classes), dtype=float), dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def _load_frame() -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], dict[str, str]]:
    m4 = r1._load_m4_fields(r1.DEFAULT_POPULATION_ROOTS[0])
    population, pop_hash = r1._load_population(r1.DEFAULT_POPULATION_ROOTS, m4)
    f115 = r1._load_f115_selection(r1.DEFAULT_FEATURE_SELECTION)
    features, feature_hash = r1._load_features(population, f115, r1.DEFAULT_FEATURE_PANELS)
    new = [field for field in f115 if field not in population.columns]
    labels, label_hash = r1._load_rich_labels(r1.DEFAULT_RICH_LABELS)
    frame = population.merge(features.loc[:, ["candidate_id", *new]], on="candidate_id", how="left", validate="one_to_one").merge(labels, on=r1.IDENTITY, how="left", validate="one_to_one")
    if len(frame) != len(population) or frame["candidate_id"].duplicated().any():
        raise AssertionError("target-free identity changed while joining supervised labels")
    hashes = {"population": json.dumps(pop_hash, sort_keys=True), "features": json.dumps(feature_hash, sort_keys=True), "labels": label_hash}
    return frame, _o_fields(), m4, hashes


def _inner_o(train: pd.DataFrame, o_fields: tuple[str, ...], held_month: str, seed: int) -> pd.DataFrame:
    local = train.loc[r1._valid_label(train)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    edges = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    parts = []
    for fold in range(r1.INNER_SPLITS):
        valid = local.iloc[int(edges[fold + 1]):int(edges[fold + 2])].copy()
        if valid.empty:
            continue
        fit = local.loc[local["__label_available_at__"].lt(valid["__decision_ts__"].min())].copy()
        if len(fit) < r1.MIN_OUTER_TRAIN_ROWS or np.unique(r1._event(fit, SPEC)).size < 2:
            continue
        x_fit, med = r1._matrix(fit, o_fields)
        x_valid, _ = r1._matrix(valid, o_fields, med)
        model = r2._binary_config(r2.FROZEN_CONFIG, seed + fold)
        model.fit(x_fit, r1._event(fit, SPEC), sample_weight=r2._weights(fit, SPEC, "uniform"))
        part = valid.loc[:, [*r1.IDENTITY, "__label_available_at__", SPEC.label_field, "policy_net_bps", "policy_regret_bps", "policy_gross_bps"]].copy()
        part["opp_oof_raw"] = model.predict_proba(x_valid)[:, 1].astype(np.float32)
        part["inner_fold"] = fold
        part["held_month"] = held_month
        parts.append(part)
    if not parts:
        raise ValueError("insufficient O inner OOF support")
    return pd.concat(parts, ignore_index=True)


def _inner_c(
    train: pd.DataFrame,
    c_fields: tuple[str, ...],
    target: CTarget,
    held_month: str,
    seed: int,
    weight_kind: str = "uniform",
    *,
    model_params: dict[str, Any] | None = None,
    seed_offsets: tuple[int, ...] = (0,),
) -> pd.DataFrame:
    local = train.loc[r1._valid_label(train)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    edges = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    parts = []
    for fold in range(r1.INNER_SPLITS):
        valid = local.iloc[int(edges[fold + 1]):int(edges[fold + 2])].copy()
        if valid.empty:
            continue
        fit = local.loc[local["__label_available_at__"].lt(valid["__decision_ts__"].min())].copy()
        c_fit = fit.loc[r1._event(fit, SPEC).astype(bool)].copy()
        y = _target(c_fit, target)
        if len(c_fit) < MIN_C_ROWS or np.unique(y).size < 2:
            continue
        x_fit, med = r1._matrix(c_fit, c_fields)
        x_valid, _ = r1._matrix(valid, c_fields, med)
        scores = []
        for offset in seed_offsets:
            model = _model(target, seed + fold + int(offset), model_params)
            model.fit(x_fit, y, sample_weight=_c_weights(c_fit, weight_kind))
            scores.append(_predict(model, target, x_valid))
        part = valid.loc[:, ["candidate_id"]].copy()
        part["conversion_oof_raw"] = np.mean(np.vstack(scores), axis=0).astype(np.float32)
        part["held_month"] = held_month
        parts.append(part)
    if not parts:
        raise ValueError("insufficient conditional C inner OOF support")
    return pd.concat(parts, ignore_index=True)


def _run_target(
    frame: pd.DataFrame,
    o_fields: tuple[str, ...],
    c_fields: tuple[str, ...],
    target: CTarget,
    seed: int,
    weight_kind: str = "uniform",
    *,
    o_seed: int | None = None,
    c_model_params: dict[str, Any] | None = None,
    c_seed_offsets: tuple[int, ...] = (0,),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one conditional-C arm while optionally freezing O's random stream.

    Feature/weight comparisons below the fixed O250/H6 winner must not alter
    O merely because a C arm receives a different seed.  ``o_seed`` preserves
    that upstream O stream while ``seed`` remains C-specific.
    """
    o_seed = seed if o_seed is None else int(o_seed)
    rows = []
    audit = []
    months = pd.date_range("2024-05-01T00:00:00Z", "2026-08-01T00:00:00Z", freq="MS", inclusive="left")
    for index, month in enumerate(months):
        stop = month + pd.offsets.MonthBegin(1)
        held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(stop)].copy()
        train = frame.loc[frame["__decision_ts__"].lt(month) & frame["__label_available_at__"].lt(month) & r1._valid_label(frame)].copy()
        try:
            inner_o = _inner_o(train, o_fields, month.strftime("%Y-%m"), o_seed + index * 101)
            inner_c = _inner_c(train, c_fields, target, month.strftime("%Y-%m"), seed + 10_000 + index * 101, weight_kind, model_params=c_model_params, seed_offsets=c_seed_offsets)
            inner = inner_o.merge(inner_c, on=["candidate_id", "held_month"], how="inner", validate="one_to_one")
            if len(inner) < r1.MIN_MAPPER_OOF_ROWS or r1._month_count(inner) < r1.MIN_MAPPER_MONTHS:
                raise ValueError("insufficient matching O/C K0 OOF support")
            bundle = r2._fit_k0(inner, SPEC, "platt")
            x_o, med_o = r1._matrix(train, o_fields)
            x_h_o, _ = r1._matrix(held, o_fields, med_o)
            o_model = r2._binary_config(r2.FROZEN_CONFIG, o_seed + 20_000 + index)
            o_model.fit(x_o, r1._event(train, SPEC), sample_weight=r2._weights(train, SPEC, "uniform"))
            raw_o = o_model.predict_proba(x_h_o)[:, 1]
            c_train = train.loc[r1._event(train, SPEC).astype(bool)].copy()
            y_c = _target(c_train, target)
            if len(c_train) < MIN_C_ROWS or np.unique(y_c).size < 2:
                raise ValueError("insufficient outer conditional C support")
            x_c, med_c = r1._matrix(c_train, c_fields)
            x_h_c, _ = r1._matrix(held, c_fields, med_c)
            c_scores = []
            for offset in c_seed_offsets:
                c_model = _model(target, seed + 30_000 + index + int(offset), c_model_params)
                c_model.fit(x_c, y_c, sample_weight=_c_weights(c_train, weight_kind))
                c_scores.append(_predict(c_model, target, x_h_c))
            raw_c = np.mean(np.vstack(c_scores), axis=0)
            part = held.loc[:, [*r1.IDENTITY, "__label_available_at__", SPEC.label_field, "policy_net_bps", "policy_regret_bps", "policy_gross_bps", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid"]].copy().reset_index(drop=True)
            part["opportunity_raw_score"] = raw_o.astype(np.float32)
            part = pd.concat((part, r2._apply_k0(bundle, raw_o, raw_c)), axis=1)
            part["held_month"] = month.strftime("%Y-%m")
            part["target"] = target.name
            rows.append(part)
            audit.append({"target": target.name, "held_month": month.strftime("%Y-%m"), "status": "complete", "outer_train_rows": len(train), "inner_oof_rows": len(inner), "c_outer_rows": len(c_train), "k0_threshold_bps": bundle.threshold})
        except ValueError as error:
            audit.append({"target": target.name, "held_month": month.strftime("%Y-%m"), "status": "skipped", "outer_train_rows": len(train), "reason": str(error)})
    if not rows:
        raise RuntimeError(f"{target.name} produced no strict OOF rows")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audit)


def _metrics(prediction: pd.DataFrame, target: CTarget) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.DataFrame([r1._k0_metrics(part, SPEC, pd.Timestamp(f"{month}-01", tz="UTC")) for month, part in prediction.groupby("held_month", sort=True)])
    monthly["target"] = target.name
    # _k0_metrics already provides an arm column for the opportunity contract;
    # substitute the C-target identity instead of creating duplicate labels.
    monthly["arm"] = target.name
    era = r1._aggregate_k0(monthly)
    era["target"] = target.name
    return monthly, era


def _summary(era: pd.DataFrame, monthly: pd.DataFrame, target: CTarget) -> dict[str, Any]:
    rows = era.loc[era["era"].isin(("2025", "2026"))].set_index("era")
    selected = float(rows["outcome_known_candidates"].sum())
    return {"target": target.name, "description": target.description, "net_2025": float(rows.loc["2025", "net_bps_per_trade"]), "net_2026": float(rows.loc["2026", "net_bps_per_trade"]), "mean_net_bps_per_trade": float(np.average(rows["net_bps_per_trade"], weights=rows["outcome_known_candidates"])), "total_net_bps": float(rows["total_net_bps"].sum()), "selected": selected, "worst_month": float(monthly.loc[monthly["held_month"].str[:4].isin(("2025", "2026")), "net_bps_per_trade"].min()), "mean_cvar10": float(rows["cvar10_bps"].mean())}


def _ranking(summary: pd.DataFrame, ref: float) -> pd.DataFrame:
    out = summary.copy()
    out["participation_vs_c0"] = out["selected"] / max(ref, 1.0)
    out["passes_gate"] = out["net_2025"].ge(90.0) & out["net_2026"].ge(90.0) & out["participation_vs_c0"].ge(.70)
    return out.sort_values(["passes_gate", "mean_net_bps_per_trade", "worst_month", "total_net_bps"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)


def _table(frame: pd.DataFrame) -> str:
    cols = [str(x) for x in frame.columns]
    return "\n".join(["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |", *("| " + " | ".join(str(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))])


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    frame, o_fields, c_fields, hashes = _load_frame()
    results = []
    monthly_all = []
    era_all = []
    audits = []
    output = {}
    for index, target in enumerate(TARGETS):
        pred, audit = _run_target(frame, o_fields, c_fields, target, SEED + index * 10_000)
        monthly, era = _metrics(pred, target)
        results.append(_summary(era, monthly, target))
        monthly_all.append(monthly); era_all.append(era); audits.append(audit); output[target.name] = pred
    summary = pd.DataFrame(results)
    reference = float(summary.loc[summary["target"].eq("C0_t5_regret"), "selected"].iloc[0])
    rank = _ranking(summary, reference)
    out.mkdir(parents=True)
    summary.to_parquet(out / "round3a_conversion_target_summary.parquet", index=False, compression="zstd")
    rank.to_parquet(out / "round3a_conversion_target_ranking.parquet", index=False, compression="zstd")
    pd.concat(monthly_all, ignore_index=True).to_parquet(out / "round3a_conversion_target_monthly.parquet", index=False, compression="zstd")
    pd.concat(era_all, ignore_index=True).to_parquet(out / "round3a_conversion_target_era.parquet", index=False, compression="zstd")
    pd.concat(audits, ignore_index=True).to_parquet(out / "round3a_fold_audit.parquet", index=False, compression="zstd")
    for name, pred in output.items():
        pred.to_parquet(out / f"{name}_outer_oof_predictions.parquet", index=False, compression="zstd")
    manifest = {"schema": SCHEMA, "status": "complete", "side": "short", "scope": "Round 3A conversion target-only strict OOF comparison; no live/canonical change", "architecture": "frozen O250_H6 stable45/uniform/Platt → C target variant → K0", "opportunity": {"definition": SPEC.description, "features": list(o_fields), "weights": "uniform", "calibration": "Platt"}, "conversion_features": {"frozen_C41": list(c_fields)}, "targets": [{"name": target.name, "description": target.description} for target in TARGETS], "causality": "all O/C fits use labels resolved strictly before each validation slice; target-free held candidates are scored before invalid labels are excluded", "selection_gate": {"both_eras_net_bps_per_trade_ge": 90.0, "participation_vs_C0_ge": .70}, "sources": hashes}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "SHORT_P0_OC_K0_ROUND3A_C_TARGET_REPORT.md").write_text("\n".join(["# Short P0 → O → C → K0 Round 3A: C targets", "", _table(rank), "", "## Contract", "", "```json", json.dumps(manifest, indent=2), "```", ""]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(run(args.out))


if __name__ == "__main__":
    main()
