#!/usr/bin/env python3
"""Frozen historical-OOF two-stage conversion challenger for July 20-23.

Stage one is the immutable v2 clean-first/adverse challenger.  Stage two uses
only stage-one OOF predictions and causal raw features to learn absolute 12h
net economics.  All feature/model/blend/admission decisions are frozen before
the exact current-period outcome table is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY, sha256
from scripts.run_historical_to_july_meaningful_mfe_gate_challenger import (
    load_current_features,
    load_historical,
    select_features_nested,
)


SCHEMA = "two_stage_absolute_net_conversion_challenger_v1"
STAGE1 = Path(
    "data_perp/artifacts/"
    "historical_to_july_meaningful_mfe_gate_challenger_20260730_v2"
)
STAGE1_CLEAN = "catboost_hard_clean_first__probability"
STAGE1_ADVERSE = "catboost_adverse_1atr_gate__probability"
CURRENT_START = pd.Timestamp("2026-07-20T00:00:00Z")
STAGE2_WINDOWS = (
    ("2026-06-15T00:00:00Z", "2026-07-01T00:00:00Z"),
    ("2026-07-01T00:00:00Z", "2026-07-20T00:00:00Z"),
)
COMPONENTS = (
    "positive_probability",
    "timeout_probability",
    "positive_payoff",
    "adverse_loss",
    "timeout_loss",
    "other_loss",
    "direct_residual",
)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _file_binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path)}


def _assert_stage1(stage1_dir: Path) -> dict[str, Any]:
    manifest_path = stage1_dir / "manifest.json"
    report = json.loads(manifest_path.read_text())
    if report.get("schema") != "historical_to_july_meaningful_mfe_gate_challenger_v1":
        raise ValueError("unexpected stage-one manifest")
    state_path = stage1_dir / "frozen_before_current_evaluation.json"
    if sha256(state_path) != report["frozen_state_sha256"]:
        raise ValueError("stage-one frozen-state hash mismatch")
    for name in ("historical_oof_predictions", "current_predictions"):
        record = report["outputs"][name]
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"stage-one {name} hash mismatch")
    return {
        "manifest": _file_binding(manifest_path),
        "frozen_state": _file_binding(state_path),
        "historical_oof": report["outputs"]["historical_oof_predictions"],
        "current_predictions": report["outputs"]["current_predictions"],
    }


def stage2_folds(frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ts = pd.to_datetime(frame["__ts__"], utc=True)
    resolved = pd.to_datetime(frame["label_resolution_utc"], utc=True)
    result = []
    for index, (start_s, end_s) in enumerate(STAGE2_WINDOWS):
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        train = np.flatnonzero(((ts < start) & (resolved < start)).to_numpy())
        valid = np.flatnonzero(((ts >= start) & (ts < end)).to_numpy())
        if len(train) < 5_000 or len(valid) < 1_000:
            raise ValueError("insufficient leakage-safe stage-two fold support")
        if not bool((resolved.iloc[train] < start).all()):
            raise AssertionError("stage-two label chronology violated")
        result.append((f"fold_{index}", train, valid))
    return result


def derive_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    net = pd.to_numeric(result["execution_net_ev_12h"], errors="raise")
    result["positive_net"] = (net > 0).astype(np.int8)
    result["adverse_negative"] = (
        result["adverse_1atr_reached"].astype(bool) & net.le(0)
    ).astype(np.int8)
    result["timeout_negative"] = (
        result["execution_exit_reason"].astype(str).eq("timeout")
        & net.le(0)
        & ~result["adverse_negative"].astype(bool)
    ).astype(np.int8)
    result["other_negative"] = (
        net.le(0)
        & ~result["adverse_negative"].astype(bool)
        & ~result["timeout_negative"].astype(bool)
    ).astype(np.int8)
    result["positive_payoff"] = net.where(net.gt(0))
    result["adverse_loss"] = net.where(result["adverse_negative"].astype(bool))
    result["timeout_loss"] = net.where(result["timeout_negative"].astype(bool))
    result["other_loss"] = net.where(result["other_negative"].astype(bool))
    result["direct_residual"] = net - pd.to_numeric(
        result["existing_alpha_ev"], errors="raise"
    )
    return result


def _classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=250,
        reg_lambda=12.0,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=-1,
        n_jobs=4,
    )


def _regressor(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber",
        alpha=0.8,
        n_estimators=260,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=180,
        reg_lambda=15.0,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=-1,
        n_jobs=4,
    )


def _fit_component(
    component: str,
    matrix: pd.DataFrame,
    target_frame: pd.DataFrame,
    positions: np.ndarray,
    seed: int,
) -> Any:
    if component == "positive_probability":
        model = _classifier(seed)
        target = target_frame["positive_net"].to_numpy(int)
        mask = np.ones(len(target), dtype=bool)
    elif component == "timeout_probability":
        model = _classifier(seed)
        target = target_frame["timeout_negative"].to_numpy(int)
        mask = np.ones(len(target), dtype=bool)
    else:
        model = _regressor(seed)
        target = pd.to_numeric(target_frame[component], errors="coerce").to_numpy(float)
        mask = np.isfinite(target)
    fit = positions[mask[positions]]
    if len(fit) < 25:
        raise ValueError(f"insufficient support for {component}")
    if not component.endswith("probability") and len(fit) < 500:
        model = DummyRegressor(strategy="mean")
    model.fit(matrix.iloc[fit], target[fit])
    return model


def _predict_component(component: str, model: Any, matrix: pd.DataFrame) -> np.ndarray:
    if component.endswith("probability"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    return np.asarray(model.predict(matrix), dtype=float)


def compose_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    p_pos = np.clip(result["pred_positive_probability"], 0.0, 1.0)
    p_adv = np.clip(result[STAGE1_ADVERSE], 0.0, 1.0)
    p_timeout = np.minimum(
        np.clip(result["pred_timeout_probability"], 0.0, 1.0), 1.0 - p_pos
    )
    p_adv = np.minimum(p_adv, 1.0 - p_pos - p_timeout)
    p_other = np.maximum(1.0 - p_pos - p_timeout - p_adv, 0.0)
    result["hurdle_ev"] = (
        p_pos * result["pred_positive_payoff"]
        + p_adv * result["pred_adverse_loss"]
        + p_timeout * result["pred_timeout_loss"]
        + p_other * result["pred_other_loss"]
    )
    result["direct_ev"] = (
        pd.to_numeric(result["existing_alpha_ev"], errors="raise")
        + result["pred_direct_residual"]
    )
    return result


def _month_top10(frame: pd.DataFrame, score: str) -> list[float]:
    work = frame.loc[np.isfinite(frame[score])].copy()
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    result = []
    for _, local in work.groupby("month"):
        take = max(1, int(math.ceil(0.10 * len(local))))
        chosen = local.nlargest(take, score)
        result.append(float(chosen["execution_net_ev_12h"].mean() * 1e4))
    return result


def select_blend(oof: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for weight in np.linspace(0.0, 1.0, 5):
        name = f"blend_{weight:.2f}"
        oof[name] = weight * oof["direct_ev"] + (1.0 - weight) * oof["hurdle_ev"]
        months = _month_top10(oof, name)
        rows.append(
            {
                "weight_direct": float(weight),
                "score_column": name,
                "worst_month_top10_net_bps": float(min(months)),
                "mean_month_top10_net_bps": float(np.mean(months)),
                "months": len(months),
            }
        )
    trials = pd.DataFrame(rows).sort_values(
        ["worst_month_top10_net_bps", "mean_month_top10_net_bps"],
        ascending=False,
        kind="stable",
    )
    return trials.iloc[0].to_dict(), trials


def train_and_freeze(
    history: pd.DataFrame,
    raw_matrix: pd.DataFrame,
    stage1_oof: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    joined = derive_targets(
        history.merge(
            stage1_oof.loc[:, [*IDENTITY, STAGE1_CLEAN, STAGE1_ADVERSE]],
            on=list(IDENTITY),
            how="inner",
            validate="one_to_one",
        )
    )
    joined = joined.loc[joined[STAGE1_CLEAN].notna()].reset_index(drop=True)
    matrix = joined.loc[:, list(raw_matrix.columns)].copy()
    matrix[STAGE1_CLEAN] = joined[STAGE1_CLEAN].to_numpy(float)
    matrix[STAGE1_ADVERSE] = joined[STAGE1_ADVERSE].to_numpy(float)
    folds = stage2_folds(joined)
    oof = joined.loc[:, [*IDENTITY, "label_resolution_utc", "execution_net_ev_12h",
                         "existing_alpha_ev", "positive_net", "timeout_negative",
                         "adverse_negative"]].copy()
    for component in COMPONENTS:
        oof[f"pred_{component}"] = np.nan
    fold_records: list[dict[str, Any]] = []
    for fold_index, (name, train, valid) in enumerate(folds):
        side_values = joined["side_name"].astype(str).to_numpy()
        for side in ("long", "short"):
            train_side = train[side_values[train] == side]
            valid_side = valid[side_values[valid] == side]
            selected, _ = select_features_nested(
                matrix,
                joined["execution_net_ev_12h"].to_numpy(float),
                train_side,
                48,
            )
            for component in COMPONENTS:
                model = _fit_component(
                    component, matrix.loc[:, selected], joined, train_side,
                    seed + 100 * fold_index + (0 if side == "long" else 50),
                )
                oof.loc[valid_side, f"pred_{component}"] = _predict_component(
                    component, model, matrix.loc[valid_side, selected]
                )
            fold_records.append(
                {
                    "fold": name,
                    "side": side,
                    "train_rows": len(train_side),
                    "validation_rows": len(valid_side),
                    "features": selected,
                }
            )
    oof[STAGE1_CLEAN] = joined[STAGE1_CLEAN]
    oof[STAGE1_ADVERSE] = joined[STAGE1_ADVERSE]
    oof = compose_scores(oof.loc[oof["pred_direct_residual"].notna()].copy())
    winner, blend_trials = select_blend(oof)

    final_models: dict[str, dict[str, Any]] = {}
    final_records: dict[str, Any] = {}
    for side_index, side in enumerate(("long", "short")):
        positions = np.flatnonzero(joined["side_name"].astype(str).eq(side).to_numpy())
        selected, screen = select_features_nested(
            matrix,
            joined["execution_net_ev_12h"].to_numpy(float),
            positions,
            48,
        )
        final_models[side] = {}
        for component in COMPONENTS:
            final_models[side][component] = _fit_component(
                component, matrix.loc[:, selected], joined, positions,
                seed + 1000 + side_index * 100,
            )
        clean_threshold = float(
            np.quantile(joined.loc[positions, STAGE1_CLEAN], 0.90)
        )
        final_records[side] = {
            "rows": len(positions),
            "features": selected,
            "clean_admission_threshold": clean_threshold,
            "feature_screen_top": screen.head(20).to_dict("records"),
        }
    model_path = output_dir / "frozen_stage2_models.joblib"
    joblib.dump(final_models, model_path)
    blend_path = output_dir / "historical_blend_trials.csv"
    blend_trials.to_csv(blend_path, index=False)
    state = {
        "schema": SCHEMA,
        "selection_status": "fully_frozen_before_current_outcomes_loaded",
        "history_rows_with_stage1_oof": len(joined),
        "history_max_label_resolution": joined["label_resolution_utc"].max(),
        "folds": fold_records,
        "winner": winner,
        "sides": final_records,
        "model_artifact": _file_binding(model_path),
        "blend_trials": _file_binding(blend_path),
    }
    return state, oof


def score_current(
    state: Mapping[str, Any],
    model_path: Path,
    current_features: pd.DataFrame,
    current_stage1: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = joblib.load(model_path)
    current = current_features.merge(
        current_stage1.loc[
            :, [*IDENTITY, "existing_alpha_ev", STAGE1_CLEAN, STAGE1_ADVERSE]
        ],
        on=list(IDENTITY), how="inner", validate="one_to_one",
    )
    for component in COMPONENTS:
        current[f"pred_{component}"] = np.nan
    safeguards = []
    for side in ("long", "short"):
        mask = current["side_name"].astype(str).eq(side)
        positions = np.flatnonzero(mask.to_numpy())
        features = state["sides"][side]["features"]
        matrix = current.loc[positions, features]
        for component in COMPONENTS:
            current.loc[positions, f"pred_{component}"] = _predict_component(
                component, models[side][component], matrix
            )
        threshold = state["sides"][side]["clean_admission_threshold"]
        admitted = current.loc[positions, STAGE1_CLEAN].ge(threshold)
        rows = int(admitted.sum())
        coverage = float(admitted.mean())
        abstain = rows < 50 or coverage < 0.01
        safeguards.append(
            {
                "side": side,
                "threshold": threshold,
                "admitted_rows": rows,
                "coverage": coverage,
                "abstain": abstain,
                "reason": (
                    "stage1_support_below_50_or_1pct" if abstain else "eligible"
                ),
            }
        )
        current.loc[positions, "stage1_admitted"] = admitted.to_numpy()
        current.loc[positions, "side_abstained"] = abstain
    current = compose_scores(current)
    weight = float(state["winner"]["weight_direct"])
    current["selected_ev"] = (
        weight * current["direct_ev"] + (1.0 - weight) * current["hurdle_ev"]
    )
    current["eligible"] = (
        current["stage1_admitted"].astype(bool)
        & ~current["side_abstained"].astype(bool)
    )
    return current, pd.DataFrame(safeguards)


def economics(current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, eligible in (
        ("unrestricted_global_top10", np.ones(len(current), dtype=bool)),
        ("safeguarded_global_top10", current["eligible"].to_numpy(bool)),
    ):
        local = current.loc[eligible].copy()
        take = max(1, int(math.ceil(0.10 * len(local)))) if len(local) else 0
        selected = local.nlargest(take, "selected_ev") if take else local
        rows.append(
            {
                "arm": arm,
                "rows": len(selected),
                "coverage_population": len(selected) / len(current),
                "net_ev_bps": selected["execution_net_ev_12h"].mean() * 1e4,
                "positive_net_precision": (
                    selected["execution_net_ev_12h"].gt(0).mean()
                ),
                "long_rows": selected["side_name"].astype(str).eq("long").sum(),
                "short_rows": selected["side_name"].astype(str).eq("short").sum(),
            }
        )
    for floor_bps in (0, 25, 50):
        selected = current.loc[
            current["eligible"]
            & current["selected_ev"].ge(floor_bps / 1e4)
        ]
        rows.append(
            {
                "arm": f"safeguarded_absolute_floor_{floor_bps}bps",
                "rows": len(selected),
                "coverage_population": len(selected) / len(current),
                "net_ev_bps": selected["execution_net_ev_12h"].mean() * 1e4,
                "positive_net_precision": selected["execution_net_ev_12h"].gt(0).mean(),
                "long_rows": selected["side_name"].astype(str).eq("long").sum(),
                "short_rows": selected["side_name"].astype(str).eq("short").sum(),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    stage1_binding = _assert_stage1(args.stage1_dir)
    history, matrix, _, history_lineage = load_historical(
        args.historical_features,
        args.historical_feature_manifest,
        args.historical_grid,
        args.historical_grid_manifest,
    )
    stage1_oof = pd.read_parquet(
        Path(stage1_binding["historical_oof"]["path"])
    )
    state, oof = train_and_freeze(
        history, matrix, stage1_oof, args.output_dir, args.seed
    )
    state["stage1_lineage"] = stage1_binding
    state["history_lineage"] = history_lineage
    frozen_path = args.output_dir / "frozen_before_current_evaluation.json"
    _write_json(frozen_path, state)
    frozen_sha = sha256(frozen_path)

    # Current causal features are scored before the stage-one exact outcome
    # table is opened.  The stage-one table then supplies frozen probabilities
    # plus the single untouched current evaluation.
    current_features = load_current_features(args.current_packb, matrix.columns)
    current_stage1 = pd.read_parquet(
        Path(stage1_binding["current_predictions"]["path"])
    )
    scores, safeguards = score_current(
        state, Path(state["model_artifact"]["path"]),
        current_features, current_stage1,
    )
    outcome_columns = [
        *IDENTITY, "execution_net_ev_12h", "execution_exit_reason",
        "exact_net_positive", "audit_day_utc",
    ]
    scored = scores.merge(
        current_stage1.loc[:, outcome_columns],
        on=list(IDENTITY), how="inner", validate="one_to_one",
    )
    metric_table = economics(scored)
    probability_rows = []
    for side in ("pooled", "long", "short"):
        local = scored if side == "pooled" else scored.loc[
            scored["side_name"].astype(str).eq(side)
        ]
        y = local["execution_net_ev_12h"].gt(0).astype(int)
        p = np.clip(local["pred_positive_probability"], 0, 1)
        probability_rows.append(
            {
                "scope": side,
                "rows": len(local),
                "positive_prevalence": y.mean(),
                "positive_probability_mean": p.mean(),
                "positive_auc": roc_auc_score(y, p),
                "positive_brier": brier_score_loss(y, p),
            }
        )
    outputs = {}
    tables = {
        "historical_oof_stage2": oof,
        "current_predictions": scored,
        "current_economics": metric_table,
        "current_probability_metrics": pd.DataFrame(probability_rows),
        "current_side_safeguards": safeguards,
    }
    for name, table in tables.items():
        suffix = ".parquet" if "predictions" in name or "oof" in name else ".csv"
        path = args.output_dir / f"{name}{suffix}"
        if suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
        outputs[name] = {**_file_binding(path), "rows": len(table)}
    report = {
        "schema": SCHEMA,
        "status": "completed_research_only_no_promotion",
        "promotion_eligible": False,
        "current_outcomes_used_for_selection": False,
        "portfolio_replay": {
            "status": "not_compatible",
            "reason": (
                "exact candidate-local 12h outcomes overlap; a causal fill/order "
                "ledger is not present in this artifact, so summing candidates "
                "under portfolio limits would overstate executable economics"
            ),
        },
        "deployment_contract": {
            "status": "research_shadow_only",
            "note": (
                "The support abstention is an evaluation safeguard, not a "
                "deployed rule; the historical reliability grid selected no "
                "abstention policy."
            ),
            "shadow_monitors": [
                "short adverse probability",
                "249-feature out-of-distribution fraction",
                "clean-first probability",
                "leaf-mix Jensen-Shannon divergence",
            ],
            "candidate_regime_inputs": [
                "breadth/high-volatility state",
                "ETH-BTC relative state",
                "cross-sectional liquidity state",
            ],
        },
        "contract": {
            "stage1": "immutable v2 clean-first and adverse historical OOF probabilities",
            "stage2": (
                "side-local causal features; positive hurdle, timeout probability, "
                "conditional favorable/adverse/timeout/other payoffs, direct net "
                "residual; historical worst-month/mean-month blend"
            ),
            "selection": "strict historical temporal OOF; no current refit or selection",
            "ranking": "one global pooled order, never per timestamp",
            "conditional_peak_used": False,
        },
        "frozen_state": {
            "path": str(frozen_path),
            "sha256_before_current_outcomes_loaded": frozen_sha,
        },
        "outputs": outputs,
    }
    report_path = args.output_dir / "report.json"
    _write_json(report_path, report)
    manifest = {
        "schema": SCHEMA,
        "status": report["status"],
        "promotion_eligible": False,
        "current_outcomes_used_for_selection": False,
        "frozen_state_sha256": frozen_sha,
        "report": _file_binding(report_path),
        "outputs": outputs,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return report


def parser() -> argparse.ArgumentParser:
    historical = Path(
        "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2"
    )
    labels = Path(
        "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1"
    )
    current = Path(
        "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1-dir", type=Path, default=STAGE1)
    p.add_argument("--historical-features", type=Path, default=historical / "capture_feature_universe.parquet")
    p.add_argument("--historical-feature-manifest", type=Path, default=historical / "manifest.json")
    p.add_argument("--historical-grid", type=Path, default=labels / "meaningful_mfe_label_grid.parquet")
    p.add_argument("--historical-grid-manifest", type=Path, default=labels / "manifest.json")
    p.add_argument("--current-packb", type=Path, default=current / "packb/packb_forward_context.parquet")
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "data_perp/artifacts/two_stage_absolute_net_conversion_challenger_"
            "20260730_v2"
        ),
    )
    p.add_argument("--seed", type=int, default=20260730)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
