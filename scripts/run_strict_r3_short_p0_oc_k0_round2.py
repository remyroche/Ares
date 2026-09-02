#!/usr/bin/env python3
"""Round 2: strict-prequential opportunity-head funnel above frozen P0→C→K0.

This is deliberately an O-only experiment.  It reuses *only* the matching,
strict-prequential conversion OOF predictions from Round 1 for each opportunity
definition, refits K0 from the combined OOF ledger, and never changes C or K0's
analytic form.  The sequence is feature stability/caps → training weights →
probability calibration → chronological Optuna HPO → final strict OOF replay.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_round2_v1"
ROUND1 = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round1_20260821_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round2_20260821_v1"
MDA_START = pd.Timestamp("2024-05-01T00:00:00Z")
MDA_END = pd.Timestamp("2025-01-01T00:00:00Z")
FINALISTS = (
    r1.OpportunitySpec("O250_H6", 6, 250.0),
    r1.OpportunitySpec("O300_H12", 12, 300.0),
)
FEATURE_CAPS = (30, 45, 60, 75, 90)
WEIGHTS = ("uniform", "boundary_confidence", "equal_month_boundary", "magnitude_positive")
CALIBRATIONS = ("raw", "platt", "isotonic", "beta")
HPO_TRIALS = 20
SEED = 1729


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: dict[str, Any]


@dataclass
class ProbabilityCalibrator:
    kind: str
    model: Any | None
    increasing: bool = True

    def predict(self, raw: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(raw, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.kind == "raw":
            return x
        if self.kind == "isotonic":
            return np.asarray(self.model.predict(x), dtype=float)
        if self.kind == "platt":
            return np.asarray(self.model.predict_proba(x.reshape(-1, 1))[:, 1], dtype=float)
        if self.kind == "beta":
            z = np.column_stack((np.log(x), -np.log1p(-x)))
            return np.asarray(self.model.predict_proba(z)[:, 1], dtype=float)
        raise AssertionError(self.kind)


@dataclass
class K0Bundle:
    o_calibrator: ProbabilityCalibrator
    mu1: IsotonicRegression
    mu0: float
    threshold: float


def _binary_config(config: ModelConfig, seed: int) -> LGBMClassifier:
    params = {
        "objective": "binary", "n_estimators": 180, "learning_rate": .035,
        "max_depth": 3, "num_leaves": 15, "min_child_samples": 40,
        "subsample": .85, "subsample_freq": 1, "colsample_bytree": .85,
        "reg_lambda": 4.0, "reg_alpha": .10, "max_bin": 127,
        "path_smooth": 0.0, "extra_trees": False, "class_weight": "balanced",
        "random_state": seed, "n_jobs": -1, "verbosity": -1,
    }
    params.update(config.params)
    return LGBMClassifier(**params)


FROZEN_CONFIG = ModelConfig("frozen", {})


def _valid(frame: pd.DataFrame) -> pd.Series:
    return r1._valid_label(frame)


def _event(frame: pd.DataFrame, spec: r1.OpportunitySpec) -> np.ndarray:
    return r1._event(frame, spec)


def _fit_probability(kind: str, raw: np.ndarray, event: np.ndarray) -> ProbabilityCalibrator:
    x = np.clip(np.asarray(raw, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(event, dtype=int)
    if len(x) < 50 or np.unique(y).size < 2:
        raise ValueError("insufficient OOF support for probability calibration")
    if kind == "raw":
        return ProbabilityCalibrator(kind, None)
    if kind == "isotonic":
        iso, increasing = r1._fit_isotonic(x, y.astype(float), 0.0, 1.0)
        return ProbabilityCalibrator(kind, iso, increasing)
    if kind == "platt":
        return ProbabilityCalibrator(kind, LogisticRegression(C=2.0, max_iter=500, random_state=SEED).fit(x.reshape(-1, 1), y))
    if kind == "beta":
        z = np.column_stack((np.log(x), -np.log1p(-x)))
        return ProbabilityCalibrator(kind, LogisticRegression(C=2.0, max_iter=500, random_state=SEED).fit(z, y))
    raise ValueError(kind)


def _fit_k0(oof: pd.DataFrame, spec: r1.OpportunitySpec, calibration: str) -> K0Bundle:
    event = _event(oof, spec).astype(bool)
    y = r1._finite(oof["policy_net_bps"]).clip(-r1.POLICY_CLIP_BPS, r1.POLICY_CLIP_BPS).to_numpy(float)
    if int(event.sum()) < r1.MIN_C_POSITIVES:
        raise ValueError("insufficient event-positive OOF support")
    cal = _fit_probability(calibration, oof["opp_oof_raw"].to_numpy(float), event)
    p = cal.predict(oof["opp_oof_raw"].to_numpy(float))
    mu1, _ = r1._fit_isotonic(oof.loc[event, "conversion_oof_raw"].to_numpy(float), y[event], -r1.POLICY_CLIP_BPS, r1.POLICY_CLIP_BPS)
    global_mean = float(np.mean(y))
    negative = ~event
    mu0 = float((y[negative].sum() + 500.0 * global_mean) / (negative.sum() + 500.0))
    k0 = p * np.asarray(mu1.predict(oof["conversion_oof_raw"].to_numpy(float)), dtype=float) + (1.0 - p) * mu0
    return K0Bundle(cal, mu1, mu0, float(np.quantile(k0, r1.P80)))


def _apply_k0(bundle: K0Bundle, raw_o: np.ndarray, raw_c: np.ndarray) -> pd.DataFrame:
    p = bundle.o_calibrator.predict(raw_o)
    expected = p * np.asarray(bundle.mu1.predict(raw_c), dtype=float) + (1.0 - p) * bundle.mu0
    return pd.DataFrame({
        "opportunity_probability": p.astype(np.float32),
        "conversion_score": np.asarray(raw_c, dtype=np.float32),
        "K0_expected_policy_net_bps": expected.astype(np.float32),
        "K0_train_p80_expected_policy_net_bps": np.full(len(p), bundle.threshold, dtype=np.float32),
    })


def _weights(frame: pd.DataFrame, spec: r1.OpportunitySpec, kind: str) -> np.ndarray:
    n = len(frame)
    weight = np.ones(n, dtype=float)
    mfe = r1._finite(frame[spec.label_field]).to_numpy(float)
    if kind in {"boundary_confidence", "equal_month_boundary"}:
        distance = np.abs(mfe - spec.threshold_bps)
        confidence = np.where(distance < 25.0, .25, np.where(distance < 75.0, .60, 1.0))
        weight *= confidence
    if kind == "equal_month_boundary":
        month = frame["__decision_ts__"].dt.strftime("%Y-%m")
        counts = month.value_counts()
        weight *= month.map(len(frame) / counts).to_numpy(float)
    if kind == "magnitude_positive":
        extra = mfe - spec.threshold_bps
        event = extra > 0.0
        weight[event & (extra > 100.0)] *= 1.2
        weight[event & (extra > 300.0)] *= 1.4 / 1.2
    if kind not in WEIGHTS:
        raise ValueError(kind)
    return weight / max(float(np.mean(weight)), 1e-9)


def _mda_objective(y: np.ndarray, score: np.ndarray) -> float:
    if len(y) < 20 or np.unique(y).size < 2:
        return float("nan")
    p = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    prevalence = float(np.mean(y))
    rank = pd.Series(p).rank(method="first", pct=True).to_numpy(float)
    top20 = y[rank >= .8]
    top30 = y[rank >= .7]
    lift20 = float(top20.mean() / max(prevalence, 1e-6)) if len(top20) else 0.0
    lift30 = float(top30.mean() / max(prevalence, 1e-6)) if len(top30) else 0.0
    skill = 1.0 - float(brier_score_loss(y, p)) / max(prevalence * (1.0 - prevalence), 1e-6)
    return float(average_precision_score(y, p) / max(prevalence, 1e-6) + .5 * lift20 + .5 * lift30 + skill)


def _strict_mda(frame: pd.DataFrame, fields: tuple[str, ...], spec: r1.OpportunitySpec, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = frame.loc[_valid(frame)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    boundaries = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    deltas: dict[str, list[float]] = {field: [] for field in fields}
    records: list[dict[str, Any]] = []
    for fold in range(r1.INNER_SPLITS):
        valid_start, valid_end = int(boundaries[fold + 1]), int(boundaries[fold + 2])
        valid = local.iloc[valid_start:valid_end].copy()
        if valid.empty:
            continue
        start = valid["__decision_ts__"].min()
        fit = local.loc[local["__label_available_at__"].lt(start)].copy()
        if len(fit) < r1.MIN_OUTER_TRAIN_ROWS or np.unique(_event(fit, spec)).size < 2:
            continue
        x_fit, med = r1._matrix(fit, fields)
        x_valid, _ = r1._matrix(valid, fields, med)
        model = _binary_config(FROZEN_CONFIG, seed + fold)
        model.fit(x_fit, _event(fit, spec))
        y = _event(valid, spec)
        base = _mda_objective(y, model.predict_proba(x_valid)[:, 1])
        rng = np.random.default_rng(seed + 100 + fold)
        for field in fields:
            permuted = x_valid.copy()
            permuted[field] = rng.permutation(permuted[field].to_numpy())
            value = _mda_objective(y, model.predict_proba(permuted)[:, 1])
            delta = base - value
            deltas[field].append(delta)
            records.append({"arm": spec.name, "feature": field, "fold": fold, "validation_start": start, "mda_delta": delta})
    result = pd.DataFrame({
        "arm": spec.name, "feature": list(fields),
        "mda_mean": [float(np.nanmean(deltas[f])) if deltas[f] else float("nan") for f in fields],
        "mda_min": [float(np.nanmin(deltas[f])) if deltas[f] else float("nan") for f in fields],
        "mda_positive_folds": [int(np.sum(np.asarray(deltas[f]) > 0.0)) for f in fields],
        "mda_folds": [len(deltas[f]) for f in fields],
    })
    result = result.sort_values(["mda_mean", "mda_min", "feature"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result, pd.DataFrame(records)


def _stable_prefixes(mda: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    required = np.ceil(.60 * mda["mda_folds"].max()).astype(int)
    stable = mda.loc[(mda["mda_positive_folds"] >= required) & mda["mda_mean"].gt(0.0)].copy()
    remaining = mda.loc[~mda["feature"].isin(stable["feature"])].copy()
    order = pd.concat((stable, remaining), ignore_index=True)["feature"].astype(str).tolist()
    return {cap: tuple(order[:cap]) for cap in FEATURE_CAPS}


def _conversion_lookups(spec: r1.OpportunitySpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = pd.read_parquet(ROUND1 / "round1_outer_oof_predictions.parquet")
    inner = pd.read_parquet(ROUND1 / "round1_inner_oof_ledger.parquet")
    for frame in (outer, inner):
        frame["__decision_ts__"] = r1._utc(frame["__decision_ts__"])
        frame["__label_available_at__"] = r1._utc(frame["__label_available_at__"])
    return (
        outer.loc[outer["arm"].eq(spec.name), ["candidate_id", "conversion_score"]].drop_duplicates("candidate_id"),
        inner.loc[inner["arm"].eq(spec.name), ["candidate_id", "held_month", "conversion_oof_raw"]].drop_duplicates(["candidate_id", "held_month"]),
    )


def _inner_oof_o(train: pd.DataFrame, spec: r1.OpportunitySpec, fields: tuple[str, ...], weight_kind: str, config: ModelConfig, seed: int, c_lookup: pd.DataFrame, held_month: str) -> pd.DataFrame:
    local = train.loc[_valid(train)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    boundaries = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    parts: list[pd.DataFrame] = []
    for fold in range(r1.INNER_SPLITS):
        start, end = int(boundaries[fold + 1]), int(boundaries[fold + 2])
        valid = local.iloc[start:end].copy()
        if valid.empty:
            continue
        decision_start = valid["__decision_ts__"].min()
        fit = local.loc[local["__label_available_at__"].lt(decision_start)].copy()
        c_fit = fit.loc[_event(fit, spec).astype(bool)].copy()
        # Round 2 changes O only, so its inner ledger must retain exactly the
        # rows for which the frozen matching C model was eligible in Round 1.
        if (
            len(fit) < r1.MIN_OUTER_TRAIN_ROWS
            or len(c_fit) < r1.MIN_C_POSITIVES
            or r1._month_count(c_fit) < r1.MIN_MAPPER_MONTHS
            or np.unique(_event(fit, spec)).size < 2
        ):
            continue
        x_fit, med = r1._matrix(fit, fields)
        x_valid, _ = r1._matrix(valid, fields, med)
        model = _binary_config(config, seed + fold)
        model.fit(x_fit, _event(fit, spec), sample_weight=_weights(fit, spec, weight_kind))
        part = valid.loc[:, [*r1.IDENTITY, "__label_available_at__", spec.label_field, "policy_net_bps", "policy_regret_bps"]].copy()
        part["opp_oof_raw"] = model.predict_proba(x_valid)[:, 1].astype(np.float32)
        part["inner_fold"] = fold
        part = part.merge(c_lookup.loc[c_lookup["held_month"].eq(held_month), ["candidate_id", "conversion_oof_raw"]], on="candidate_id", how="inner", validate="one_to_one")
        if len(part) != len(valid):
            raise AssertionError("frozen C inner OOF lookup does not match strict O inner rows")
        parts.append(part)
    if not parts:
        raise ValueError("insufficient strict O inner OOF support")
    out = pd.concat(parts, ignore_index=True)
    if len(out) < r1.MIN_MAPPER_OOF_ROWS or r1._month_count(out) < r1.MIN_MAPPER_MONTHS:
        raise ValueError("insufficient combined O/C OOF K0 support")
    return out


def _run_variant(frame: pd.DataFrame, spec: r1.OpportunitySpec, fields: tuple[str, ...], weight_kind: str, calibration: str, config: ModelConfig, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_c, inner_c = _conversion_lookups(spec)
    rows: list[pd.DataFrame] = []
    inner_rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    months = pd.date_range("2024-05-01T00:00:00Z", "2026-08-01T00:00:00Z", freq="MS", inclusive="left")
    for index, month in enumerate(months):
        end = month + pd.offsets.MonthBegin(1)
        held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(end)].copy()
        train = frame.loc[frame["__decision_ts__"].lt(month) & frame["__label_available_at__"].lt(month) & _valid(frame)].copy()
        try:
            # C was intentionally frozen from Round 1.  An early month that
            # had no valid C/K0 OOF ledger is not eligible for an O-only
            # comparison; skipping it preserves the joint-stack population.
            if inner_c.loc[inner_c["held_month"].eq(month.strftime("%Y-%m"))].empty:
                raise ValueError("matching frozen C inner OOF support unavailable")
            inner = _inner_oof_o(train, spec, fields, weight_kind, config, seed + index * 37, inner_c, month.strftime("%Y-%m"))
            bundle = _fit_k0(inner, spec, calibration)
            x_train, med = r1._matrix(train, fields)
            x_held, _ = r1._matrix(held, fields, med)
            model = _binary_config(config, seed + 1_000 + index * 37)
            model.fit(x_train, _event(train, spec), sample_weight=_weights(train, spec, weight_kind))
            raw = model.predict_proba(x_held)[:, 1]
            c = held.loc[:, [*r1.IDENTITY, "__label_available_at__", spec.label_field, "policy_net_bps", "policy_regret_bps", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid"]].copy().merge(outer_c, on="candidate_id", how="left", validate="one_to_one")
            if c["conversion_score"].isna().any():
                raise AssertionError("frozen C outer lookup is incomplete")
            c["opportunity_raw_score"] = raw.astype(np.float32)
            # The frozen C lookup already supplies conversion_score.  K0 emits
            # the same diagnostic field, so retain one canonical copy rather
            # than creating duplicate pandas labels for later calibration.
            mapped = _apply_k0(bundle, raw, c["conversion_score"].to_numpy(float)).drop(columns=["conversion_score"])
            c = pd.concat((c.reset_index(drop=True), mapped), axis=1)
            c["held_month"] = month.strftime("%Y-%m")
            rows.append(c)
            inner["held_month"] = month.strftime("%Y-%m")
            inner_rows.append(inner)
            audits.append({"held_month": month.strftime("%Y-%m"), "status": "complete", "train_rows": len(train), "inner_rows": len(inner), "k0_threshold_bps": bundle.threshold})
        except ValueError as error:
            audits.append({"held_month": month.strftime("%Y-%m"), "status": "skipped", "train_rows": len(train), "reason": str(error)})
    if not rows:
        raise RuntimeError("variant produced no strict OOF rows")
    return pd.concat(rows, ignore_index=True), pd.concat(inner_rows, ignore_index=True), pd.DataFrame(audits)


def _metrics(prediction: pd.DataFrame, spec: r1.OpportunitySpec, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_o = pd.DataFrame([r1._o_metrics(part, spec, pd.Timestamp(f"{month}-01", tz="UTC")) for month, part in prediction.groupby("held_month", sort=True)])
    monthly_o["arm"] = arm
    monthly_k0 = pd.DataFrame([r1._k0_metrics(part, spec, pd.Timestamp(f"{month}-01", tz="UTC")) for month, part in prediction.groupby("held_month", sort=True)])
    monthly_k0["arm"] = arm
    return monthly_o, monthly_k0, r1._aggregate_k0(monthly_k0)


def _summary(era: pd.DataFrame, monthly: pd.DataFrame, arm: str) -> dict[str, Any]:
    use = era.loc[era["era"].isin(("2025", "2026"))].copy()
    by = use.set_index("era")
    month = monthly.loc[monthly["held_month"].str[:4].isin(("2025", "2026"))]
    count = float(use["outcome_known_candidates"].sum())
    mean = float(np.average(use["net_bps_per_trade"], weights=use["outcome_known_candidates"])) if count else float("nan")
    return {
        "arm": arm,
        "net_2025": float(by.loc["2025", "net_bps_per_trade"]), "net_2026": float(by.loc["2026", "net_bps_per_trade"]),
        "mean_net_bps_per_trade": mean, "total_net_bps": float(use["total_net_bps"].sum()), "selected": count,
        "worst_month": float(month["net_bps_per_trade"].min()),
        "mean_cvar10": float(use["cvar10_bps"].mean()),
    }


def _select(summary: pd.DataFrame, *, ref_selected: float) -> pd.DataFrame:
    out = summary.copy()
    out["participation_vs_round1"] = out["selected"] / max(ref_selected, 1.0)
    out["passes_round2_gate"] = out["net_2025"].ge(90.0) & out["net_2026"].ge(90.0) & out["participation_vs_round1"].ge(.70)
    out = out.sort_values(["passes_round2_gate", "mean_net_bps_per_trade", "worst_month", "total_net_bps"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)
    return out


def _select_final(summary: pd.DataFrame, refs: dict[str, float]) -> pd.DataFrame:
    """Rank final controls and HPO challengers against their own Round-1 arm.

    An HPO model is a challenger, not an automatic replacement.  Retaining the
    preceding sequential winner is necessary to avoid discarding a valid,
    better-performing calibration merely because development-only HPO chose a
    different tree geometry.
    """
    out = summary.copy()
    out["participation_vs_round1"] = out.apply(lambda row: float(row["selected"]) / max(refs[str(row["spec"])], 1.0), axis=1)
    out["passes_round2_gate"] = out["net_2025"].ge(90.0) & out["net_2026"].ge(90.0) & out["participation_vs_round1"].ge(.70)
    return out.sort_values(["passes_round2_gate", "mean_net_bps_per_trade", "worst_month", "total_net_bps"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)


def _dev_hpo(frame: pd.DataFrame, spec: r1.OpportunitySpec, fields: tuple[str, ...], weight_kind: str, seed: int) -> tuple[ModelConfig, pd.DataFrame]:
    local = frame.loc[_valid(frame) & frame["__decision_ts__"].ge(MDA_START) & frame["__decision_ts__"].lt(MDA_END)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    boundaries = np.linspace(0, len(local), 5, dtype=int)
    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("max_depth", 3, 8)
        leaves = trial.suggest_int("num_leaves", 7, min(63, 2 ** depth))
        params = {
            "n_estimators": 2000, "learning_rate": trial.suggest_float("learning_rate", .01, .08, log=True),
            "max_depth": depth, "num_leaves": leaves, "min_child_samples": trial.suggest_int("min_child_samples", 50, 800),
            "subsample": trial.suggest_float("subsample", .6, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", .5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 20.0, log=True), "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0), "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 50.0), "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
        }
        values: list[float] = []
        best_iter: list[int] = []
        for fold in range(3):
            start, end = int(boundaries[fold + 1]), int(boundaries[fold + 2])
            valid = local.iloc[start:end].copy()
            if valid.empty:
                continue
            fit = local.loc[local["__label_available_at__"].lt(valid["__decision_ts__"].min())].copy()
            if len(fit) < r1.MIN_OUTER_TRAIN_ROWS or np.unique(_event(fit, spec)).size < 2:
                continue
            x_fit, med = r1._matrix(fit, fields)
            x_valid, _ = r1._matrix(valid, fields, med)
            model = _binary_config(ModelConfig("trial", params), seed + trial.number * 19 + fold)
            model.fit(x_fit, _event(fit, spec), sample_weight=_weights(fit, spec, weight_kind), eval_set=[(x_valid, _event(valid, spec))], callbacks=[early_stopping(30, verbose=False)])
            values.append(_mda_objective(_event(valid, spec), model.predict_proba(x_valid)[:, 1]))
            best_iter.append(int(model.best_iteration_ or model.n_estimators))
            trial.report(float(np.mean(values)), fold)
            if fold >= 1 and trial.should_prune():
                raise optuna.TrialPruned()
        if len(values) < 2:
            raise optuna.TrialPruned()
        trial.set_user_attr("best_n_estimators", int(np.median(best_iter)))
        return float(np.mean(values))
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.optimize(objective, n_trials=HPO_TRIALS, show_progress_bar=False)
    rows = []
    for trial in study.trials:
        rows.append({"arm": spec.name, "trial": trial.number, "state": trial.state.name, "value": trial.value, **trial.params, "best_n_estimators": trial.user_attrs.get("best_n_estimators")})
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("all HPO trials pruned")
    winner = study.best_trial
    params = dict(winner.params)
    params["n_estimators"] = int(winner.user_attrs["best_n_estimators"])
    return ModelConfig(f"optuna_{winner.number}", params), pd.DataFrame(rows)


def _report(out: Path, selections: pd.DataFrame, summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def table(frame: pd.DataFrame) -> str:
        columns = [str(column) for column in frame.columns]
        rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
        return "\n".join((
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ))
    lines = ["# Short P0 → O → C → K0 Round 2: opportunity learner funnel", "", "Research-only. C is the matching frozen strict-prequential T5 stream from Round 1; no other model layer is added.", "", "## Final selection", "", table(selections), "", "## All stage summaries", "", table(summary), "", "## Contract", "", "```json", json.dumps(manifest, indent=2), "```", ""]
    (out / "SHORT_P0_OC_K0_ROUND2_REPORT.md").write_text("\n".join(lines))


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {out}")
    m4 = r1._load_m4_fields(r1.DEFAULT_POPULATION_ROOTS[0])
    population, population_hashes = r1._load_population(r1.DEFAULT_POPULATION_ROOTS, m4)
    f115 = r1._load_f115_selection(r1.DEFAULT_FEATURE_SELECTION)
    features, feature_hashes = r1._load_features(population, f115, r1.DEFAULT_FEATURE_PANELS)
    labels, label_hash = r1._load_rich_labels(r1.DEFAULT_RICH_LABELS)
    # M4 fields are already present in the target-free population.  Merge only
    # genuinely additional F115 fields so the canonical population columns are
    # never suffixed or silently replaced.
    new_feature_fields = tuple(field for field in f115 if field not in population.columns)
    frame = population.merge(features.loc[:, ["candidate_id", *new_feature_fields]], on="candidate_id", how="left", validate="one_to_one").merge(labels, on=r1.IDENTITY, how="left", validate="one_to_one")
    if len(frame) != len(population) or frame["candidate_id"].duplicated().any():
        raise AssertionError("target-free identity changed while joining labels")
    coverage = pd.DataFrame({"feature": list(f115), "finite_fraction": [float(r1._finite(frame[f]).notna().mean()) for f in f115]})
    pool = tuple(coverage.loc[coverage["finite_fraction"].ge(.90), "feature"].astype(str))
    if len(pool) < max(FEATURE_CAPS):
        raise AssertionError("fewer than 90 feature-coverage-valid opportunity fields")
    dev = frame.loc[frame["__decision_ts__"].ge(MDA_START) & frame["__decision_ts__"].lt(MDA_END)].copy()
    all_mda: list[pd.DataFrame] = []
    all_mda_folds: list[pd.DataFrame] = []
    feature_sets: dict[str, dict[int, tuple[str, ...]]] = {}
    for index, spec in enumerate(FINALISTS):
        ranking, folds = _strict_mda(dev, pool, spec, SEED + index * 1_000)
        all_mda.append(ranking); all_mda_folds.append(folds)
        feature_sets[spec.name] = _stable_prefixes(ranking)
    stage_rows: list[dict[str, Any]] = []
    stage_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    round1_k0 = pd.read_parquet(ROUND1 / "round1_k0_era_metrics.parquet")
    for spec in FINALISTS:
        ref = float(round1_k0.loc[(round1_k0["arm"].eq(spec.name)) & round1_k0["era"].isin(("2025", "2026")), "outcome_known_candidates"].sum())
        for cap, fields in feature_sets[spec.name].items():
            arm = f"{spec.name}|A_cap{cap}"
            outer, inner, _ = _run_variant(frame, spec, fields, "uniform", "isotonic", FROZEN_CONFIG, seed=SEED + cap)
            _, monthly, era = _metrics(outer, spec, arm)
            stage_rows.append({"stage": "A_features", **_summary(era, monthly, arm), "spec": spec.name, "fields": cap, "weight": "uniform", "calibration": "isotonic", "model": "frozen"})
            stage_cache[arm] = (outer, inner)
        stage = _select(pd.DataFrame([row for row in stage_rows if row["spec"] == spec.name and row["stage"] == "A_features"]), ref_selected=ref)
        winner = str(stage.iloc[0]["arm"])
        fields = feature_sets[spec.name][int(stage.iloc[0]["fields"])]
        for weight in WEIGHTS:
            arm = f"{spec.name}|B_{weight}"
            if weight == "uniform":
                outer, inner = stage_cache[winner]
            else:
                outer, inner, _ = _run_variant(frame, spec, fields, weight, "isotonic", FROZEN_CONFIG, seed=SEED + 10_000 + len(stage_rows))
            _, monthly, era = _metrics(outer, spec, arm)
            stage_rows.append({"stage": "B_weights", **_summary(era, monthly, arm), "spec": spec.name, "fields": len(fields), "weight": weight, "calibration": "isotonic", "model": "frozen"})
            stage_cache[arm] = (outer, inner)
        stage_b = _select(pd.DataFrame([row for row in stage_rows if row["spec"] == spec.name and row["stage"] == "B_weights"]), ref_selected=ref)
        b_winner = str(stage_b.iloc[0]["arm"])
        raw_outer, raw_inner = stage_cache[b_winner]
        for calibration in CALIBRATIONS:
            arm = f"{spec.name}|C_{calibration}"
            # Recompute K0 maps from the O raw predictions; this is calibration-only and never refits O/C.
            pieces: list[pd.DataFrame] = []
            for month, held in raw_outer.groupby("held_month", sort=True):
                oof = raw_inner.loc[raw_inner["held_month"].eq(month)].copy()
                bundle = _fit_k0(oof, spec, calibration)
                part = held.copy()
                part = part.drop(columns=["opportunity_probability", "K0_expected_policy_net_bps", "K0_train_p80_expected_policy_net_bps"])
                part = pd.concat((part.reset_index(drop=True), _apply_k0(bundle, part["opportunity_raw_score"].to_numpy(float), part["conversion_score"].to_numpy(float))), axis=1)
                pieces.append(part)
            outer = pd.concat(pieces, ignore_index=True)
            _, monthly, era = _metrics(outer, spec, arm)
            stage_rows.append({"stage": "C_calibration", **_summary(era, monthly, arm), "spec": spec.name, "fields": int(stage_b.iloc[0]["fields"]), "weight": str(stage_b.iloc[0]["weight"]), "calibration": calibration, "model": "frozen"})
            stage_cache[arm] = (outer, raw_inner)
    summary = pd.DataFrame(stage_rows)
    final_rows: list[dict[str, Any]] = []
    final_controls: list[dict[str, Any]] = []
    hpo_rows: list[pd.DataFrame] = []
    final_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for idx, spec in enumerate(FINALISTS):
        ref = float(round1_k0.loc[(round1_k0["arm"].eq(spec.name)) & round1_k0["era"].isin(("2025", "2026")), "outcome_known_candidates"].sum())
        c_rows = summary.loc[(summary["spec"].eq(spec.name)) & (summary["stage"].eq("C_calibration"))]
        c_rank = _select(c_rows, ref_selected=ref)
        c_win = c_rank.iloc[0]
        final_controls.append({
            "stage": "C_frozen_control", "arm": str(c_win["arm"]), "net_2025": float(c_win["net_2025"]), "net_2026": float(c_win["net_2026"]),
            "mean_net_bps_per_trade": float(c_win["mean_net_bps_per_trade"]), "total_net_bps": float(c_win["total_net_bps"]),
            "selected": float(c_win["selected"]), "worst_month": float(c_win["worst_month"]), "mean_cvar10": float(c_win["mean_cvar10"]),
            "spec": spec.name, "fields": int(c_win["fields"]), "weight": str(c_win["weight"]), "calibration": str(c_win["calibration"]), "model": "frozen", "model_params": "{}",
        })
        fields = feature_sets[spec.name][int(c_win["fields"])]
        config, trials = _dev_hpo(frame, spec, fields, str(c_win["weight"]), SEED + 40_000 + idx * 1_000)
        hpo_rows.append(trials)
        arm = f"{spec.name}|D_{config.name}"
        outer, inner, audit = _run_variant(frame, spec, fields, str(c_win["weight"]), str(c_win["calibration"]), config, seed=SEED + 50_000 + idx * 1_000)
        o_month, monthly, era = _metrics(outer, spec, arm)
        final_rows.append({"stage": "D_hpo_final", **_summary(era, monthly, arm), "spec": spec.name, "fields": len(fields), "weight": str(c_win["weight"]), "calibration": str(c_win["calibration"]), "model": config.name, "model_params": json.dumps(config.params, sort_keys=True)})
        final_outputs[arm] = (outer, inner)
        audit.to_parquet(out.parent / f".{out.name}_{spec.name}_audit.tmp.parquet", index=False)
    final_summary = pd.concat((pd.DataFrame(final_controls), pd.DataFrame(final_rows)), ignore_index=True)
    refs = {spec.name: float(round1_k0.loc[(round1_k0["arm"].eq(spec.name)) & round1_k0["era"].isin(("2025", "2026")), "outcome_known_candidates"].sum()) for spec in FINALISTS}
    selection = _select_final(final_summary, refs)
    out.mkdir(parents=True)
    pd.concat(all_mda, ignore_index=True).to_parquet(out / "round2_target_specific_stability_mda.parquet", index=False, compression="zstd")
    pd.concat(all_mda_folds, ignore_index=True).to_parquet(out / "round2_target_specific_stability_mda_folds.parquet", index=False, compression="zstd")
    coverage.to_parquet(out / "round2_opportunity_feature_coverage.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "round2_stage_summaries.parquet", index=False, compression="zstd")
    final_summary.to_parquet(out / "round2_final_summaries.parquet", index=False, compression="zstd")
    selection.to_parquet(out / "round2_final_ranking.parquet", index=False, compression="zstd")
    pd.concat(hpo_rows, ignore_index=True).to_parquet(out / "round2_opportunity_hpo_trials.parquet", index=False, compression="zstd")
    for arm, (outer, inner) in final_outputs.items():
        slug = arm.replace("|", "_")
        outer.to_parquet(out / f"{slug}_outer_oof_predictions.parquet", index=False, compression="zstd")
        inner.to_parquet(out / f"{slug}_inner_oof_ledger.parquet", index=False, compression="zstd")
    for temp in out.parent.glob(f".{out.name}_*_audit.tmp.parquet"):
        temp.replace(out / temp.name.removeprefix(f".{out.name}_").replace(".tmp", ""))
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short", "scope": "Round 2 O-only strict-prequential funnel; no live/canonical change",
        "architecture": "P0 → O → frozen matching C → K0", "finalists": [spec.name for spec in FINALISTS],
        "mda": {"window": [MDA_START.isoformat(), MDA_END.isoformat()], "target_specific": True, "strict_fit": "label_available_at < validation_start", "stability": "positive MDA in >=60% of chronological folds then ranked prefix"},
        "stages": {"A": "feature caps 30/45/60/75/90", "B": list(WEIGHTS), "C": list(CALIBRATIONS), "D": f"Optuna {HPO_TRIALS} trials, three chronological development folds, median pruning after two folds"},
        "selection_gate": {"both_2025_2026_net_bps_per_trade_ge": 90.0, "participation_vs_round1_ge": .70},
        "sources": {"round1_manifest": r1._sha256(ROUND1 / "run_manifest.json"), "population_manifest_hashes": population_hashes, "rich_labels_manifest": label_hash, "feature_panels": feature_hashes},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _report(out, selection, pd.concat((summary, final_summary), ignore_index=True), manifest)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args.out))


if __name__ == "__main__":
    main()
