#!/usr/bin/env python3
"""Strict chronological OOF residual-trust ablation for the regime stack.

This is deliberately a *ranking/context* experiment.  It consumes frozen OOF
base/residual scores, independent OOF regime and transition probabilities, and
the pre-existing OOF competing-risk sidecar.  Timing, MAE, target-price and
wait/action fields are rejected at the feature boundary.

Each evaluation quarter is predicted by a model and two EV maps fitted only on
earlier label-resolved candidates.  Final policy selection is one pooled global
top-k after that causal EV map, never per timestamp or per side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import (  # noqa: E402
    IDENTITY_COLUMNS,
    RegimeOOFStackError,
    assert_outcome_free,
    validate_candidate_identity,
)
from extreme_price_movements.regime_stack_evaluation import (  # noqa: E402
    EvaluationColumns,
    evaluate_matched_arms,
)


SCHEMA = "interaction_conditioned_residual_trust_oof_v1"
DEFAULT_SOFT = ROOT / "data_perp/artifacts/reconstructed_2023apr_2024_candidate_oof_regime_transition_20260730_v1/candidate_oof_regime_transition.parquet"
DEFAULT_SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet"
DEFAULT_RISK = ROOT / "data_perp/artifacts/clean_competing_risk_probability_oof_2023_2024_20260730_v1/clean_competing_probability_oof_sidecar.parquet"

ACTION_TOKENS = ("timing", "mae", "target_price", "targetprice", "wait", "action", "entry_price")
TARGET = "execution_net_ev_12h"
LABEL_DELAY = pd.Timedelta(hours=12)

# Frozen from the signed v2 discovery artifact.  Spread interactions were not
# stable under context-conditional monthly permutation, so cannot enter here.
STABLE_SCORE_STATE_INTERACTIONS = {
    "regime": ("regime_state_p__2",),
    "transition": ("transition_state_p__settled_destination",),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_action_fields(columns: Iterable[str]) -> None:
    leaked = [str(column) for column in columns if any(token in str(column).lower() for token in ACTION_TOKENS)]
    if leaked:
        raise RegimeOOFStackError(f"action-layer fields are forbidden in trust learner: {sorted(leaked)}")


def _require_available(frame: pd.DataFrame, columns: Sequence[str], *, prefix: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise RegimeOOFStackError(f"{prefix} is missing required columns: {missing}")


def _validate_oof_provenance(frame: pd.DataFrame) -> None:
    for prefix in ("regime", "transition"):
        cols = [f"{prefix}_train_end_utc", f"{prefix}_available_utc"]
        _require_available(frame, cols, prefix=prefix)
        train_end = pd.to_datetime(frame[cols[0]], utc=True, errors="coerce")
        available = pd.to_datetime(frame[cols[1]], utc=True, errors="coerce")
        if train_end.isna().any() or available.isna().any():
            raise RegimeOOFStackError(f"{prefix} OOF provenance contains invalid timestamps")
        if train_end.ge(frame["__ts__"]).any() or available.gt(frame["__ts__"]).any():
            raise RegimeOOFStackError(f"{prefix} provenance is not causally available at candidate time")
    probability_start = pd.to_datetime(frame["probability_evaluation_start_utc"], utc=True, errors="coerce")
    if probability_start.isna().any() or probability_start.gt(frame["__ts__"]).any():
        raise RegimeOOFStackError("competing-risk probability is not a valid OOF candidate prediction")


def build_panel(*, soft_path: Path, scores_path: Path, risk_path: Path) -> pd.DataFrame:
    """Return the exact, non-imputed candidate intersection of all six arms."""

    soft = validate_candidate_identity(pd.read_parquet(soft_path))
    scores = validate_candidate_identity(pd.read_parquet(scores_path))
    risk = validate_candidate_identity(pd.read_parquet(risk_path))
    _require_available(scores, [TARGET, "execution_gross_ev_12h", "execution_cost_return", "__reconstructed_soft_alpha_12h__", "score_base_expected_ev", "score_residual_expected_ev"], prefix="scores")
    _require_available(risk, ["probability_fold_id", "probability_evaluation_start_utc", "clean_opportunity_p__regime_plus_transition", "adverse_competing_risk_p__regime_plus_transition"], prefix="risk sidecar")
    panel = scores.merge(soft, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    panel = panel.merge(risk, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    panel = validate_candidate_identity(panel).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if panel.empty:
        raise RegimeOOFStackError("the three OOF artifacts have no common candidate population")
    _validate_oof_provenance(panel)
    return panel


def feature_lists() -> dict[str, list[str]]:
    """The only permitted predictors, explicitly excluding action-layer fields."""

    base = ["score_base_expected_ev", "score_residual_expected_ev"]
    regime = ["regime_state_p__0", "regime_state_p__1", "regime_state_p__2"]
    transition = [
        "transition_state_p__stable", "transition_state_p__approach", "transition_state_p__immediate_lead",
        "transition_state_p__transition", "transition_state_p__acceleration", "transition_state_p__early_destination",
        "transition_state_p__settled_destination",
    ]
    score = "score_residual_expected_ev"
    interactions = {
        "regime": [f"trust_int__{score}_x_{state}" for state in STABLE_SCORE_STATE_INTERACTIONS["regime"]],
        "transition": [f"trust_int__{score}_x_{state}" for state in STABLE_SCORE_STATE_INTERACTIONS["transition"]],
    }
    result = {
        "baseline": base,
        "regime_only": [*base, *regime, *interactions["regime"]],
        "transition_only": [*base, *transition, *interactions["transition"]],
        "regime_plus_transition": [*base, *regime, *transition, *interactions["regime"], *interactions["transition"]],
    }
    result["regime_plus_transition_plus_adverse_risk"] = [*result["regime_plus_transition"], "adverse_competing_risk_p__regime_plus_transition"]
    # Clean is deliberately a competing, not combined, probability ablation.
    result["regime_plus_transition_plus_clean_probability"] = [*result["regime_plus_transition"], "clean_opportunity_p__regime_plus_transition"]
    for fields in result.values():
        _reject_action_fields(fields)
        assert_outcome_free(pd.DataFrame(columns=fields), extra_forbidden=ACTION_TOKENS)
    return result


def add_interactions(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    score = pd.to_numeric(work["score_residual_expected_ev"], errors="coerce")
    for layer, states in STABLE_SCORE_STATE_INTERACTIONS.items():
        for state in states:
            name = f"trust_int__score_residual_expected_ev_x_{state}"
            work[name] = score * pd.to_numeric(work[state], errors="coerce")
    return work


def _matrix(train: pd.DataFrame, evaluation: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    _reject_action_fields(features)
    train_x = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    eval_x = evaluation.loc[:, features].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train_x.replace([np.inf, -np.inf], np.nan).fillna(medians).astype("float32"), eval_x.replace([np.inf, -np.inf], np.nan).fillna(medians).astype("float32")


def _fit_isotonic(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray):
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[valid], y_array[valid]
    if len(x_array) < 8 or np.unique(x_array).size < 2:
        value = float(np.nanmean(y_array)) if len(y_array) else 0.0
        return lambda data: np.full(len(data), value, dtype=float)
    model = IsotonicRegression(out_of_bounds="clip", increasing="auto").fit(x_array, y_array)
    return lambda data: np.asarray(model.predict(np.asarray(data, dtype=float)), dtype=float)


def _quarters(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    periods = pd.period_range(start=start.to_period("Q"), end=(end - pd.Timedelta(nanoseconds=1)).to_period("Q"), freq="Q")
    return [period.start_time.tz_localize("UTC") for period in periods]


def _fit_fold_arm(train: pd.DataFrame, evaluation: pd.DataFrame, features: list[str], *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit initial score map, residual-trust learner, and final causal EV map."""

    y_train = pd.to_numeric(train[TARGET], errors="coerce").fillna(0.0).to_numpy(float)
    initial = _fit_isotonic(train["score_residual_expected_ev"], y_train)
    initial_train = initial(train["score_residual_expected_ev"])
    initial_eval = initial(evaluation["score_residual_expected_ev"])
    x_train, x_eval = _matrix(train, evaluation, features)
    model = lgb.LGBMRegressor(
        n_estimators=180, learning_rate=0.035, num_leaves=15, min_child_samples=180,
        subsample=0.85, colsample_bytree=0.9, reg_lambda=3.0, random_state=seed, n_jobs=4, verbosity=-1,
    ).fit(x_train, y_train - initial_train)
    raw_train = initial_train + model.predict(x_train)
    raw_eval = initial_eval + model.predict(x_eval)
    final_map = _fit_isotonic(raw_train, y_train)
    return np.asarray(raw_eval, float), final_map(raw_eval)


def run(*, output_dir: Path, soft_path: Path = DEFAULT_SOFT, scores_path: Path = DEFAULT_SCORES, risk_path: Path = DEFAULT_RISK, start: str = "2023-04-01", end: str = "2025-01-01", min_train_rows: int = 5000, top_fraction: float = 0.10, seed: int = 314) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise RegimeOOFStackError(f"refusing to overwrite output: {output}")
    panel = add_interactions(build_panel(soft_path=Path(soft_path), scores_path=Path(scores_path), risk_path=Path(risk_path)))
    start_utc, end_utc = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    panel = panel.loc[panel["__ts__"].ge(start_utc) & panel["__ts__"].lt(end_utc)].copy()
    if panel.empty:
        raise RegimeOOFStackError("requested date window has no common candidates")
    arms = feature_lists()
    predictions: dict[str, list[pd.DataFrame]] = {arm: [] for arm in arms}
    provenance: list[dict[str, Any]] = []
    for fold_number, evaluation_start in enumerate(_quarters(start_utc, end_utc)):
        evaluation_end = min(evaluation_start + pd.DateOffset(months=3), end_utc)
        # +12h is the label-availability floor, not a feature; equality is not allowed.
        train = panel.loc[(panel["__ts__"] + LABEL_DELAY).lt(evaluation_start)].copy()
        evaluation = panel.loc[panel["__ts__"].ge(evaluation_start) & panel["__ts__"].lt(evaluation_end)].copy()
        if evaluation.empty or len(train) < int(min_train_rows):
            continue
        fold_id = f"trust_q{evaluation_start.year}_{evaluation_start.quarter}"
        for arm_number, (arm, features) in enumerate(arms.items()):
            raw, mapped = _fit_fold_arm(train, evaluation, features, seed=seed + fold_number * 17 + arm_number)
            predictions[arm].append(evaluation.loc[:, list(IDENTITY_COLUMNS)].assign(trust_fold_id=fold_id, trust_train_end_utc=evaluation_start, trust_label_available_before_utc=evaluation_start, raw_trust_score=raw, mapped_score=mapped))
        provenance.append({"trust_fold_id": fold_id, "evaluation_start_utc": evaluation_start, "evaluation_end_exclusive_utc": evaluation_end, "train_rows": int(len(train)), "evaluation_rows": int(len(evaluation)), "train_label_available_max_utc": train["__ts__"].max() + LABEL_DELAY, "features_contract": "base/residual scores; separate OOF regime/transition probabilities; two frozen stable score×state interactions; optional one OOF risk probability"})
    if not provenance:
        raise RegimeOOFStackError("no chronological evaluation fold has the requested training support")
    output.mkdir(parents=True)
    sidecar_dir = output / "prediction_sidecars"
    sidecar_dir.mkdir()
    joined_arms: dict[str, pd.DataFrame] = {}
    for arm, parts in predictions.items():
        if not parts:
            raise RegimeOOFStackError(f"arm {arm!r} emitted no OOF predictions")
        sidecar = pd.concat(parts, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable")
        validate_candidate_identity(sidecar)
        sidecar.to_parquet(sidecar_dir / f"{arm}.parquet", index=False)
        joined_arms[arm] = panel.merge(sidecar, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    columns = EvaluationColumns(mapped_score="mapped_score", alpha_target="__reconstructed_soft_alpha_12h__", net_ev=TARGET, gross_ev="execution_gross_ev_12h", cost="execution_cost_return")
    summary, period_metrics, category_metrics = evaluate_matched_arms(joined_arms, columns=columns, top_fraction=top_fraction, category_col="regime_state_id")
    provenance_frame = pd.DataFrame(provenance)
    provenance_frame.to_parquet(output / "fold_provenance.parquet", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    period_metrics.to_parquet(output / "period_metrics.parquet", index=False)
    category_metrics.to_parquet(output / "category_stability.parquet", index=False)
    (output / "feature_lists.json").write_text(json.dumps({"stable_interaction_discovery": "oof_regime_transition_interactions_2023q4_2024q1_20260730_v2", "only_stable_interactions": STABLE_SCORE_STATE_INTERACTIONS, "arms": arms}, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": SCHEMA,
        "status": "CHRONOLOGICAL_OOF_COMPLETE",
        "window_requested": [start_utc.isoformat(), end_utc.isoformat()],
        "window_common_population": [panel["__ts__"].min().isoformat(), panel["__ts__"].max().isoformat()],
        "common_candidate_rows": int(len(panel)),
        "predicted_candidate_rows": int(len(next(iter(joined_arms.values())))),
        "selection": {"basis": "pooled_global_post_causal_ev_mapping_top_k", "top_fraction": float(top_fraction), "per_timestamp_selection": False, "per_side_selection": False},
        "label_resolution": "+12h required before each fold evaluation start",
        "input_artifacts": {str(Path(path).resolve()): _sha(Path(path)) for path in (soft_path, scores_path, risk_path)},
        "exclusions": {"action_fields": list(ACTION_TOKENS), "untrusted_interactions": "GMM/risk summaries and non-stable spread interactions excluded"},
        "outputs": {str(path.relative_to(output)): _sha(path) for path in [output / "fold_provenance.parquet", output / "metrics_summary.csv", output / "period_metrics.parquet", output / "category_stability.parquet", output / "feature_lists.json", *sorted(sidecar_dir.glob("*.parquet"))]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (output / "manifest.sha256").write_text(_sha(output / "manifest.json") + "  manifest.json\n")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--soft", type=Path, default=DEFAULT_SOFT)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--start", default="2023-04-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--min-train-rows", type=int, default=5000)
    parser.add_argument("--top-fraction", type=float, default=.10)
    parser.add_argument("--seed", type=int, default=314)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(run(output_dir=args.output_dir, soft_path=args.soft, scores_path=args.scores, risk_path=args.risk, start=args.start, end=args.end, min_train_rows=args.min_train_rows, top_fraction=args.top_fraction, seed=args.seed))
