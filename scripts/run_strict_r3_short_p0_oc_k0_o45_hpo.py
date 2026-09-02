#!/usr/bin/env python3
"""Final bounded HPO for the selected short O45 opportunity head.

This runner intentionally tunes *only* the binary O45 head.  The C59
conversion score, analytic K0 mixture, P0 anchor fallback, target-free P0
population, policy labels, and +75 bps admission floor are all frozen.  The
development objective is O-head quality only; full K0 economics are measured
afterward on monthly outer OOF predictions.

It is a research-only short-side experiment.  It never alters long or live
contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score, brier_score_loss


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_phase_d_false_positive as phase_d  # noqa: E402
import run_strict_r3_short_p0_oc_k0_phase_a_timing as phase_a  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_o45_hpo_v1"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_o45_hpo_202408_202607_20260822_v1"
SEED = 1729
DEV_START = pd.Timestamp("2024-05-01T00:00:00Z")
DEV_END = pd.Timestamp("2025-01-01T00:00:00Z")
HPO_MAX_TRIALS = 60
HPO_STALE_TRIALS = 20
MAX_FIT_DAYS = 120
OUTER_MONTHS = pd.date_range("2024-05-01T00:00:00Z", "2026-08-01T00:00:00Z", freq="MS", inclusive="left")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _StopAfterStaleTrials:
    """Bound a study by consecutive non-improvements, not an arbitrary count."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.best_value = -np.inf
        self.stale_trials = 0
        self.stop_reason = "maximum_trials"

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = trial.value if trial.state == optuna.trial.TrialState.COMPLETE else None
        if value is not None and np.isfinite(value) and float(value) > self.best_value:
            self.best_value = float(value)
            self.stale_trials = 0
        else:
            self.stale_trials += 1
        if self.stale_trials >= self.limit:
            self.stop_reason = f"{self.limit}_consecutive_stale_trials"
            study.stop()


def _params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": 1600,
        "learning_rate": trial.suggest_float("learning_rate", .01, .06, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "num_leaves": trial.suggest_int("num_leaves", 7, 31),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 300),
        "subsample": trial.suggest_float("subsample", .70, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", .60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", .5, 80.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127]),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 30.0),
        "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
    }


def _model(seed: int, params: dict[str, Any]) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        subsample_freq=1,
        **params,
    )


def _whole_day_subsample(frame: pd.DataFrame, maximum_days: int = MAX_FIT_DAYS) -> pd.DataFrame:
    """Deterministically retain complete decision days when development is large."""
    if frame.empty:
        return frame.copy()
    day = frame["__decision_ts__"].dt.floor("D")
    days = pd.DatetimeIndex(day.drop_duplicates().sort_values())
    if len(days) <= maximum_days:
        return frame.copy()
    positions = np.unique(np.linspace(0, len(days) - 1, maximum_days, dtype=int))
    keep = set(days[positions])
    return frame.loc[day.isin(keep)].copy()


def _top20_precision(y: np.ndarray, probability: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    order = np.argsort(probability, kind="stable")
    size = max(1, int(np.ceil(.20 * len(order))))
    return float(np.mean(y[order[-size:]]))


def _development_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        phase_d._valid(frame)
        & frame["__decision_ts__"].ge(DEV_START)
        & frame["__decision_ts__"].lt(DEV_END)
    ].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _development_hpo(frame: pd.DataFrame, fields: tuple[str, ...]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    local = _development_frame(frame)
    boundaries = np.linspace(0, len(local), 5, dtype=int)

    def objective(trial: optuna.Trial) -> float:
        params = _params(trial)
        utilities: list[float] = []
        best_iterations: list[int] = []
        for fold in range(3):
            valid = local.iloc[int(boundaries[fold + 1]):int(boundaries[fold + 2])].copy()
            if valid.empty:
                continue
            start = valid["__decision_ts__"].min()
            fit = local.loc[local["__label_available_at__"].lt(start)].copy()
            fit = _whole_day_subsample(fit)
            y_fit = phase_d._event(fit)
            y_valid = phase_d._event(valid)
            if len(fit) < r1.MIN_OUTER_TRAIN_ROWS or np.unique(y_fit).size < 2 or np.unique(y_valid).size < 2:
                continue
            x_fit, medians = r1._matrix(fit, fields)
            x_valid, _ = r1._matrix(valid, fields, medians)
            model = _model(SEED + trial.number * 101 + fold, params)
            model.fit(
                x_fit, y_fit, eval_set=[(x_valid, y_valid)],
                callbacks=[early_stopping(30, verbose=False)],
            )
            probability = model.predict_proba(x_valid)[:, 1]
            prauc = float(average_precision_score(y_valid, probability))
            precision = _top20_precision(y_valid, probability)
            brier = float(brier_score_loss(y_valid, np.clip(probability, 1e-6, 1 - 1e-6)))
            # O-only objective: high-tail opportunity precision dominates;
            # Brier prevents a precision-only degenerate probability model.
            utilities.append(.45 * prauc + .40 * precision + .15 * (1.0 - brier))
            best_iterations.append(int(model.best_iteration_ or model.n_estimators))
            trial.report(float(np.mean(utilities)), fold)
            if fold >= 1 and trial.should_prune():
                raise optuna.TrialPruned()
        if len(utilities) < 2:
            raise optuna.TrialPruned()
        trial.set_user_attr("best_n_estimators", int(np.median(best_iterations)))
        return float(np.mean(utilities))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    stale = _StopAfterStaleTrials(HPO_STALE_TRIALS)
    study.optimize(objective, n_trials=HPO_MAX_TRIALS, callbacks=[stale], show_progress_bar=False)
    completed = [item for item in study.trials if item.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("all O45 HPO trials were pruned")
    winner = study.best_trial
    params = dict(winner.params)
    params["n_estimators"] = int(winner.user_attrs["best_n_estimators"])
    trials = pd.DataFrame([
        {
            "trial": item.number, "state": item.state.name, "value": item.value,
            **item.params, "best_n_estimators": item.user_attrs.get("best_n_estimators"),
        }
        for item in study.trials
    ])
    return params, trials, {
        "attempted_trials": int(len(study.trials)),
        "stale_trials_at_stop": int(stale.stale_trials),
        "stop_reason": stale.stop_reason,
        "max_trials": HPO_MAX_TRIALS,
        "stale_limit": HPO_STALE_TRIALS,
        "subsample": f"at most {MAX_FIT_DAYS} complete decision days per development fit",
    }


def _inner_oof(train: pd.DataFrame, fields: tuple[str, ...], params: dict[str, Any], held_month: str, seed: int) -> pd.DataFrame:
    local = train.loc[phase_d._valid(train) & train["frozen_c59_score"].notna()].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    bounds = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    parts: list[pd.DataFrame] = []
    for fold in range(r1.INNER_SPLITS):
        valid = local.iloc[int(bounds[fold + 1]):int(bounds[fold + 2])].copy()
        if valid.empty:
            continue
        cutoff = valid["__decision_ts__"].min()
        fit = local.loc[local["__label_available_at__"].lt(cutoff)].copy()
        y_fit = phase_d._event(fit)
        if len(fit) < r1.MIN_OUTER_TRAIN_ROWS or np.unique(y_fit).size < 2:
            continue
        x_fit, medians = r1._matrix(fit, fields)
        x_valid, _ = r1._matrix(valid, fields, medians)
        model = _model(seed + fold, params)
        model.fit(x_fit, y_fit)
        part = valid.loc[:, [*r1.IDENTITY, "__label_available_at__", "policy_net_bps", "policy_regret_bps", "policy_gross_bps", "frozen_c59_score"]].copy()
        part["opp_oof_raw"] = model.predict_proba(x_valid)[:, 1].astype(np.float32)
        part["conversion_oof_raw"] = part.pop("frozen_c59_score").to_numpy(np.float32)
        part["event_target"] = phase_d._event(valid).astype(np.int8)
        part["held_month"] = held_month
        parts.append(part)
    if not parts:
        raise ValueError("no purged inner OOF support")
    output = pd.concat(parts, ignore_index=True)
    if len(output) < phase_a.MIN_OOF_ROWS or output["__decision_ts__"].dt.strftime("%Y-%m").nunique() < phase_a.MIN_OOF_MONTHS:
        raise ValueError("insufficient purged inner OOF support")
    return output


def _outer_month(frame: pd.DataFrame, fields: tuple[str, ...], params: dict[str, Any], month: pd.Timestamp, arm: str, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = month + pd.offsets.MonthBegin(1)
    held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(end)].copy()
    train = frame.loc[frame["__decision_ts__"].lt(month) & frame["__label_available_at__"].lt(month) & phase_d._valid(frame)].copy()
    if held.empty:
        raise ValueError("empty held month")
    if held["frozen_c59_score"].isna().any():
        raise ValueError("frozen C59 OOF score unavailable for held population")
    inner = _inner_oof(train, fields, params, month.strftime("%Y-%m"), seed)
    bundle = phase_a._fit_k0(inner)
    x_train, medians = r1._matrix(train, fields)
    x_held, _ = r1._matrix(held, fields, medians)
    model = _model(seed + 10_000, params)
    model.fit(x_train, phase_d._event(train))
    raw = model.predict_proba(x_held)[:, 1]
    output = held.loc[:, [*r1.IDENTITY, "__label_available_at__", "policy_net_bps", "policy_regret_bps", "policy_gross_bps", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid", "event_timing_label_valid", "event_timing_target_invalid", "favourable_hit_6h"]].copy().reset_index(drop=True)
    output["opportunity_raw_score"] = raw.astype(np.float32)
    output = pd.concat((output, phase_a._apply_k0(bundle, raw, held["frozen_c59_score"].to_numpy(float))), axis=1)
    output["held_month"] = month.strftime("%Y-%m")
    output["arm"] = arm
    return output, {
        "arm": arm, "held_month": month.strftime("%Y-%m"), "status": "complete",
        "held_rows": int(len(held)), "outer_train_rows": int(len(train)),
        "inner_oof_rows": int(bundle.oof_rows), "inner_oof_months": int(bundle.oof_months),
    }


def _evaluate(frame: pd.DataFrame, fields: tuple[str, ...], params: dict[str, Any], arm: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for index, month in enumerate(OUTER_MONTHS):
        try:
            pred, audit = _outer_month(frame, fields, params, month, arm, SEED + index * 101)
            predictions.append(pred)
            audits.append(audit)
            metrics.append(phase_d._month_metrics(pred))
        except ValueError as exc:
            audits.append({"arm": arm, "held_month": month.strftime("%Y-%m"), "status": "skipped", "reason": str(exc)})
    if not predictions:
        raise RuntimeError("O45 HPO produced no outer predictions")
    monthly = pd.DataFrame(metrics)
    era = phase_d._era(monthly)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audits), monthly, era


def _rank(era: pd.DataFrame, control_era: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, group in era.loc[era.era.isin(("2025", "2026"))].groupby("arm", sort=True):
        by = group.set_index("era")
        control = control_era.loc[(control_era.arm.eq("O45_frozen_control")) & control_era.era.isin(("2025", "2026"))].set_index("era")
        known = group.known_admitted.to_numpy(float)
        values = group.net_bps_per_trade.to_numpy(float)
        rows.append({
            "arm": arm,
            "net_2025": float(by.loc["2025", "net_bps_per_trade"]),
            "net_2026": float(by.loc["2026", "net_bps_per_trade"]),
            "mean_net_bps_per_trade": float(np.average(values, weights=np.maximum(known, 1.0))),
            "total_net_bps": float(group.total_net_bps.sum()),
            "known_admitted": int(group.known_admitted.sum()),
            "worst_month": float(group.worst_month_net_bps.min()),
            "mean_cvar10": float(np.average(group.cvar10_bps.to_numpy(float), weights=np.maximum(known, 1.0))),
            "delta_2025": float(by.loc["2025", "net_bps_per_trade"] - control.loc["2025", "net_bps_per_trade"]),
            "delta_2026": float(by.loc["2026", "net_bps_per_trade"] - control.loc["2026", "net_bps_per_trade"]),
        })
    out = pd.DataFrame(rows)
    out["passes_gate"] = out.delta_2025.ge(-10.0) & out.delta_2026.ge(-10.0)
    return out.sort_values(["passes_gate", "mean_net_bps_per_trade", "worst_month", "total_net_bps"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)


def _table(frame: pd.DataFrame) -> str:
    """Render reports without requiring the optional ``tabulate`` dependency."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)


def _write_report(out: Path) -> Path:
    """Write or repair the human-readable report from immutable result tables."""
    manifest = json.loads((out / "run_manifest.json").read_text())
    ranking = pd.read_parquet(out / "o45_hpo_ranking.parquet")
    trials = pd.read_parquet(out / "o45_hpo_trials.parquet")
    report = out / "SHORT_P0_OC_K0_O45_HPO_REPORT.md"
    report.write_text("\n".join([
        "# Short P0 → O45 HPO → frozen C59 → K0", "", "Research-only final O-head HPO.", "",
        "## Selection", "", _table(ranking), "",
        "## Development trials", "", _table(trials), "",
        "```json", json.dumps(manifest, indent=2), "```", "",
    ]))
    return report


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    frame, fields, hashes = phase_d._load()
    winner, trials, hpo_audit = _development_hpo(frame, fields)
    control_params = {
        "n_estimators": 180, "learning_rate": .035, "max_depth": 3, "num_leaves": 15,
        "min_child_samples": 40, "subsample": .85, "colsample_bytree": .85,
        "reg_lambda": 4.0, "reg_alpha": .10, "min_split_gain": 0.0,
        "max_bin": 255, "path_smooth": 0.0, "extra_trees": False,
    }
    control, control_audit, control_monthly, control_era = _evaluate(frame, fields, control_params, "O45_frozen_control")
    candidate, candidate_audit, candidate_monthly, candidate_era = _evaluate(frame, fields, winner, "O45_hpo_winner")
    ranking = _rank(pd.concat((control_era, candidate_era), ignore_index=True), control_era)
    out.mkdir(parents=True)
    trials.to_parquet(out / "o45_hpo_trials.parquet", index=False, compression="zstd")
    ranking.to_parquet(out / "o45_hpo_ranking.parquet", index=False, compression="zstd")
    pd.concat((control, candidate), ignore_index=True).to_parquet(out / "o45_hpo_outer_oof_predictions.parquet", index=False, compression="zstd")
    pd.concat((control_audit, candidate_audit), ignore_index=True).to_parquet(out / "o45_hpo_fold_audit.parquet", index=False, compression="zstd")
    pd.concat((control_monthly, candidate_monthly), ignore_index=True).to_parquet(out / "o45_hpo_monthly_metrics.parquet", index=False, compression="zstd")
    pd.concat((control_era, candidate_era), ignore_index=True).to_parquet(out / "o45_hpo_era_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short", "scope": "final O45-only HPO; frozen C59 and analytic K0; research-only",
        "architecture": "frozen P0/F90 → O45 HPO candidate → frozen C59 → unchanged analytic K0 → fixed +75 bps admission",
        "hpo": {"window": [DEV_START.isoformat(), DEV_END.isoformat()], "objective": "O-only 0.45 PR-AUC + 0.40 precision@20 + 0.15 (1-Brier)", "fit": "label_available_at < chronological validation start", "pruner": "MedianPruner after two chronological folds", "winner_params": winner, **hpo_audit},
        "selection": {"gate": "neither 2025 nor 2026 K0 EV/trade may fall more than 10 bps vs frozen O45", "winner": str(ranking.loc[ranking.passes_gate, "arm"].iloc[0])},
        "causality": {"features": "frozen O45 target-free fields", "development_subsample": "complete decision days only", "outer": "label_available_at < held month start", "inner_k0": "strict OOF before held month", "forbidden": ["held outcomes", "joint C HPO", "additional mapper", "live/canonical mutation"]},
        "sources": hashes,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--repair-report", action="store_true", help="write the Markdown report for an otherwise completed immutable bundle")
    args = parser.parse_args()
    out = args.out.resolve()
    print(_write_report(out) if args.repair_report else run(out))


if __name__ == "__main__":
    main()
