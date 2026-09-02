#!/usr/bin/env python3
"""Round 3C: independently HPO the frozen C60/C3 conversion learner.

This is the final C-stage refinement after the strict paired C60-MDA/uniform
winner.  The opportunity stream, C target, C feature contract, sample weights,
and K0 analytic combiner remain fixed.  Only the C LightGBM geometry is tuned
on the predeclared 2024 chronological development window; a single winner and
its three-seed within-head average are then compared with the unchanged C60
control on 2025–2026 outer OOF predictions.
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
from lightgbm import early_stopping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_refinement as r3b  # noqa: E402
import run_strict_r3_short_p0_oc_k0_phase_c_feature_blocks as blocks  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair as c59  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_round3_c_hpo_v1"
ROUND3B = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3_c_refinement_20260822_v2"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3_c_hpo_20260822_v1"
FEATURE_BLOCK_SOURCE = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_c_feature_blocks_202408_202607_20260822_v1/feature_block_outer_oof_predictions.parquet"
TARGET = next(item for item in r3.TARGETS if item.name == "C3_normalized_regret")
MDA_START = pd.Timestamp("2024-05-01T00:00:00Z")
MDA_END = pd.Timestamp("2025-01-01T00:00:00Z")
# The research specification stops only after twenty *consecutive* completed
# or pruned trials fail to improve the best development utility.  A hard cap
# remains a safety guard; it is deliberately high enough to permit a full
# stale run after an improvement late in the initial exploration.
HPO_MAX_TRIALS = 60
HPO_STALE_TRIALS = 20
SEED = 1729
O_SEED = r3b.O_SEED
C_SEED = r3b.C_SEED
THREE_SEED_OFFSETS = (0, 10_000, 20_000)
# The conditional C3 development support cannot meaningfully split with the
# 1,000+ child counts suitable for a full-population classifier.  This remains
# inside the predeclared 100–1,500 range but caps the search at 600 based on
# the smallest chronological fit slice; larger values are represented by the
# deliberately low-quality constant-score behavior, not treated as a crash.
MAX_CHILD_SAMPLES = 600


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _c60_fields() -> tuple[str, ...]:
    manifest = json.loads((ROUND3B / "run_manifest.json").read_text())
    fields = tuple(manifest["conversion"]["feature_contracts"]["C60_mda"])
    if len(fields) != 60 or len(set(fields)) != 60:
        raise AssertionError("frozen C60 MDA contract is malformed")
    return fields


def _contract(contract: str) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], pd.DataFrame, dict[str, str]]:
    """Return one fixed C contract and its paired frozen-geometry control."""
    frame, o_fields, _m4, source_hashes = r3._load_frame()
    if contract == "C60":
        fields = _c60_fields()
        control = pd.read_parquet(ROUND3B / "C60_mda__uniform_outer_oof_predictions.parquet")
        return frame, o_fields, fields, control, source_hashes
    if contract == "C_SP":
        if not FEATURE_BLOCK_SOURCE.exists():
            raise FileNotFoundError(FEATURE_BLOCK_SOURCE)
        frame, sp_fields = blocks._add_prior_self_state(frame)
        fields = blocks._layer_fields(c59._c59(), sp_fields)
        raw = pd.read_parquet(FEATURE_BLOCK_SOURCE)
        control = raw.loc[raw["feature_block_arm"].eq("C_SP")].copy()
        if control.empty:
            raise AssertionError("C_SP fixed-geometry control missing")
        return frame, o_fields, fields, control, {
            **source_hashes,
            "feature_block_source_sha256": _sha256(FEATURE_BLOCK_SOURCE),
        }
    raise ValueError(contract)


class _StopAfterStaleTrials:
    """Stop an Optuna study after a declared run of non-improving trials."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.best_value = -np.inf
        self.stale_trials = 0
        self.stop_reason = "maximum_trials"

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Pruned trials carry no value but are still an attempted configuration
        # and therefore count against the stale budget.  A completed trial
        # resets it only on a strict improvement, avoiding accidental ties
        # extending a search indefinitely.
        value = trial.value if trial.state == optuna.trial.TrialState.COMPLETE else None
        if value is not None and np.isfinite(value) and float(value) > self.best_value:
            self.best_value = float(value)
            self.stale_trials = 0
        else:
            self.stale_trials += 1
        if self.stale_trials >= self.limit:
            self.stop_reason = f"{self.limit}_consecutive_stale_trials"
            study.stop()


def _dev_hpo(frame: pd.DataFrame, fields: tuple[str, ...]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    local = frame.loc[
        r1._valid_label(frame)
        & r1._event(frame, r3.SPEC).astype(bool)
        & frame["__decision_ts__"].ge(MDA_START)
        & frame["__decision_ts__"].lt(MDA_END)
    ].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    boundaries = np.linspace(0, len(local), 5, dtype=int)

    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("max_depth", 2, 6)
        # LightGBM applies ``max_depth`` independently; retaining the 7–31
        # leaf prior for depth two is valid and avoids an empty 7..4 domain.
        leaves = trial.suggest_int("num_leaves", 7, 31)
        params: dict[str, Any] = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", .01, .06, log=True),
            "max_depth": depth,
            "num_leaves": leaves,
            "min_child_samples": trial.suggest_int("min_child_samples", 100, MAX_CHILD_SAMPLES),
            "subsample": trial.suggest_float("subsample", .70, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", .60, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 30.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", .5, 100.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "max_bin": trial.suggest_categorical("max_bin", [63, 127]),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 50.0),
            "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
        }
        values: list[float] = []
        best_iterations: list[int] = []
        for fold in range(3):
            valid = local.iloc[int(boundaries[fold + 1]):int(boundaries[fold + 2])].copy()
            if valid.empty:
                continue
            start = valid["__decision_ts__"].min()
            fit = local.loc[local["__label_available_at__"].lt(start)].copy()
            y_fit = r3._target(fit, TARGET)
            if len(fit) < r3.MIN_C_ROWS or np.unique(y_fit).size < 2:
                continue
            x_fit, medians = r1._matrix(fit, fields)
            x_valid, _ = r1._matrix(valid, fields, medians)
            model = r3._model(TARGET, SEED + trial.number * 101 + fold, params)
            model.fit(
                x_fit,
                y_fit,
                sample_weight=r3._c_weights(fit, "uniform"),
                eval_set=[(x_valid, r3._target(valid, TARGET))],
                callbacks=[early_stopping(30, verbose=False)],
            )
            value = r3b._c_objective(valid, r3._predict(model, TARGET, x_valid))
            # A constant-score candidate has no rank information.  Preserve
            # it as an explicitly poor HPO point instead of allowing NaN to
            # abort the complete, predeclared study.
            values.append(float(value) if np.isfinite(value) else -2.0)
            best_iterations.append(int(model.best_iteration_ or model.n_estimators))
            trial.report(float(np.mean(values)), fold)
            if fold >= 1 and trial.should_prune():
                raise optuna.TrialPruned()
        if len(values) < 2:
            raise optuna.TrialPruned()
        trial.set_user_attr("best_n_estimators", int(np.median(best_iterations)))
        return float(np.mean(values))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    stale_stopper = _StopAfterStaleTrials(HPO_STALE_TRIALS)
    study.optimize(
        objective,
        n_trials=HPO_MAX_TRIALS,
        callbacks=[stale_stopper],
        show_progress_bar=False,
    )
    rows = [{
        "trial": trial.number, "state": trial.state.name, "value": trial.value,
        **trial.params, "best_n_estimators": trial.user_attrs.get("best_n_estimators"),
    } for trial in study.trials]
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("all C-HPO trials were pruned")
    winner = study.best_trial
    params = dict(winner.params)
    params["n_estimators"] = int(winner.user_attrs["best_n_estimators"])
    return params, pd.DataFrame(rows), {
        "attempted_trials": int(len(study.trials)),
        "stale_trials_at_stop": int(stale_stopper.stale_trials),
        "stop_reason": stale_stopper.stop_reason,
        "max_trials": HPO_MAX_TRIALS,
        "stale_limit": HPO_STALE_TRIALS,
    }


def _metrics(prediction: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    monthly = pd.DataFrame([
        r1._k0_metrics(part, r3.SPEC, pd.Timestamp(f"{month}-01", tz="UTC"))
        for month, part in prediction.groupby("held_month", sort=True)
    ])
    monthly["arm"] = arm
    era = r1._aggregate_k0(monthly)
    era["arm"] = arm
    use = era.loc[era["era"].isin(("2025", "2026"))].set_index("era")
    selected = float(use["outcome_known_candidates"].sum())
    months = monthly.loc[monthly["held_month"].str[:4].isin(("2025", "2026"))]
    return monthly, era, {
        "arm": arm,
        "net_2025": float(use.loc["2025", "net_bps_per_trade"]),
        "net_2026": float(use.loc["2026", "net_bps_per_trade"]),
        "mean_net_bps_per_trade": float(np.average(use["net_bps_per_trade"], weights=use["outcome_known_candidates"])),
        "total_net_bps": float(use["total_net_bps"].sum()),
        "selected": selected,
        "worst_month": float(months["net_bps_per_trade"].min()),
        "mean_cvar10": float(use["cvar10_bps"].mean()),
    }


def _rank(summary: pd.DataFrame, control: float) -> pd.DataFrame:
    out = summary.copy()
    out["participation_vs_c60"] = out["selected"] / max(control, 1.0)
    out["passes_gate"] = out["net_2025"].ge(90.0) & out["net_2026"].ge(90.0) & out["participation_vs_c60"].ge(.70)
    return out.sort_values(["passes_gate", "mean_net_bps_per_trade", "worst_month", "total_net_bps"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)


def _table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def run(out: Path, contract: str = "C60") -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    frame, o_fields, fields, control, source_hashes = _contract(contract)
    hpo_params, trials, hpo_audit = _dev_hpo(frame, fields)
    single, _ = r3._run_target(frame, o_fields, fields, TARGET, C_SEED, "uniform", o_seed=O_SEED, c_model_params=hpo_params)
    three_seed, _ = r3._run_target(frame, o_fields, fields, TARGET, C_SEED, "uniform", o_seed=O_SEED, c_model_params=hpo_params, c_seed_offsets=THREE_SEED_OFFSETS)
    r3b._assert_fixed_o(single, control)
    r3b._assert_fixed_o(three_seed, control)
    rows: list[dict[str, Any]] = []
    monthly_all: list[pd.DataFrame] = []
    era_all: list[pd.DataFrame] = []
    prefix = "C60" if contract == "C60" else "C_SP"
    predictions = {f"{prefix}_uniform_control": control, f"{prefix}_hpo_single": single, f"{prefix}_hpo_seed3": three_seed}
    for arm, prediction in predictions.items():
        monthly, era, summary = _metrics(prediction, arm)
        rows.append(summary); monthly_all.append(monthly); era_all.append(era)
    summary = pd.DataFrame(rows)
    reference = float(summary.loc[summary["arm"].eq(f"{prefix}_uniform_control"), "selected"].iloc[0])
    ranking = _rank(summary, reference)
    winner = str(ranking.loc[ranking["passes_gate"], "arm"].iloc[0])
    out.mkdir(parents=True)
    trials.to_parquet(out / "round3c_c60_hpo_trials.parquet", index=False, compression="zstd")
    ranking.to_parquet(out / "round3c_c60_hpo_ranking.parquet", index=False, compression="zstd")
    pd.concat(monthly_all, ignore_index=True).to_parquet(out / "round3c_c60_hpo_monthly.parquet", index=False, compression="zstd")
    pd.concat(era_all, ignore_index=True).to_parquet(out / "round3c_c60_hpo_era.parquet", index=False, compression="zstd")
    for arm, prediction in predictions.items():
        prediction.to_parquet(out / f"{arm}_outer_oof_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short",
        "scope": "C-only LightGBM HPO plus within-head seed-average check; research-only, no canonical/live mutation",
        "architecture": f"frozen P0 → O250_H6 → C3 normalized regret {contract}/uniform → K0",
        "feature_contract_name": contract, "feature_contract": list(fields),
        "hpo": {"window": [MDA_START.isoformat(), MDA_END.isoformat()], "population": "valid true O-positive rows", "fit": "label_available_at < chronological validation start", "attempted_trials": hpo_audit["attempted_trials"], "max_trials": HPO_MAX_TRIALS, "stale_stop_limit": HPO_STALE_TRIALS, "stale_trials_at_stop": hpo_audit["stale_trials_at_stop"], "stop_reason": hpo_audit["stop_reason"], "objective": "C3 rank utility plus conditional top-20 policy-net uplift", "pruner": "MedianPruner after >=2 chronological folds", "min_child_samples": [100, MAX_CHILD_SAMPLES], "winner_params": hpo_params},
        "seed_averaging": {"control": [0], "hpo_single": [0], "hpo_seed3_offsets": list(THREE_SEED_OFFSETS), "semantics": "within the one conditional C3 head; inner C OOF and outer C scores are averaged before K0"},
        "selection": {"gate": "2025/2026 net EV/trade >=90 bps and participation >=70% of C60 control", "winner": winner, "tie_break": "mean EV/trade, then worst month, then total bps"},
        "causality": {"outer": "label_available_at < held month start", "inner": "label_available_at < inner validation start", "targetfree": "all held candidates scored before invalid labels are excluded", "forbidden": ["held outcome features", "held percentile admission", "extra mapper/trust/consensus/risk layer"]},
        "sources": {"round3b_manifest_sha256": _sha256(ROUND3B / "run_manifest.json"), **source_hashes},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        f"# Short P0 → O250/H6 → C3/{contract} → K0: C HPO", "",
        f"Research-only. The only learned component changed is C3's LightGBM geometry; O250/H6, {contract} features, uniform C weighting, and K0 stay fixed.", "",
        "## Selection", "", _table(ranking), "",
        "## Optuna trials", "", _table(trials), "",
        "```json", json.dumps(manifest, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_ROUND3C_C_HPO_REPORT.md").write_text("\n".join(report))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--contract", choices=("C60", "C_SP"), default="C60")
    args = parser.parse_args()
    print(run(args.out, args.contract))


if __name__ == "__main__":
    main()
