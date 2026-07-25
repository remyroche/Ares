#!/usr/bin/env python3
"""Run soft-binary auxiliary/routing/Execution-EV exact-policy ablations.

The five auxiliary challengers are side-local soft-label recalibration heads
on top of already-strict OOF auxiliary outputs.  Their outer predictions cover
June and July.  The execution-EV and timing-routing comparison is selected on
June only and evaluated on untouched July rows with exact deployed-policy
one-minute/1,440-minute net outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_policy_soft_binary_ablation import (  # noqa: E402
    HEADS,
    auxiliary_soft_targets,
    economic_metrics,
    execution_ev_soft_target,
    expanding_month_folds,
)

DEFAULT_JOINED = (
    ROOT
    / "data_perp/artifacts/execution_ev_joined_handoff_policy_labels_20260725_v2/joined.parquet"
)
DEFAULT_MAE = (
    ROOT
    / "data_perp/artifacts/execution_ev_joined_handoff_mae_competing_risk_20260725_v1/joined.parquet"
)
DEFAULT_TARGETS = (
    ROOT
    / "data_perp/artifacts/packb_path_auxiliary_targets_20260725_v1_31_8/targets.parquet"
)
DEFAULT_POLICY_LABELS = (
    ROOT
    / "data_perp/artifacts/execution_ev_policy_labels_20260725_v1/execution_ev_policy_labels.parquet"
)
DEFAULT_POLICY_MANIFEST = DEFAULT_POLICY_LABELS.with_name("manifest.json")
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/exact_policy_soft_binary_ablations_20260725_v4"
)
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")

HEAD_FEATURES: Mapping[str, tuple[str, ...]] = {
    "peak_mfe_12h_atr": ("pred_peak_MFE_12h_ATR",),
    "time_to_first_meaningful_mfe": (
        "pred_time_to_first_meaningful_MFE",
        "pred_time_to_meaningful_mfe_p_hit_by_2h",
        "pred_time_to_meaningful_mfe_p_hit_by_4h",
        "pred_time_to_meaningful_mfe_p_hit_by_8h",
        "pred_time_to_meaningful_mfe_p_hit_by_12h",
    ),
    "mae_before_meaningful_mfe_atr": (
        "pred_mae_before_meaningful_mfe_atr",
        "pred_mae_risk_favorable_before_0_5r",
        "pred_mae_risk_adverse_0_5r_before_mfe",
        "pred_mae_risk_neither_before_horizon",
        "pred_mae_risk_stop_1r_before_mfe",
    ),
    "bars_before_price_stops_decreasing": ("pred_bars_before_price_stops_decreasing",),
    "future_slope_atr_per_hour": ("pred_favorable_path_slope_atr_per_hour",),
}
HEAD_PROVENANCE: Mapping[str, tuple[str, str]] = {
    "peak_mfe_12h_atr": ("peak_mfe_oof_fold", "peak_mfe_train_decision_cutoff"),
    "time_to_first_meaningful_mfe": (
        "time_to_mfe_oof_fold",
        "time_to_mfe_train_decision_cutoff",
    ),
    "mae_before_meaningful_mfe_atr": (
        "mae_before_mfe_oof_fold",
        "mae_before_mfe_train_decision_cutoff",
    ),
    "bars_before_price_stops_decreasing": (
        "adverse_turn_oof_fold",
        "adverse_turn_train_decision_cutoff",
    ),
    "future_slope_atr_per_hour": (
        "path_slope_oof_fold",
        "path_slope_train_decision_cutoff",
    ),
}
PARAM_GRID: tuple[Mapping[str, Any], ...] = (
    {"num_leaves": 7, "min_child_samples": 100, "reg_lambda": 4.0},
    {"num_leaves": 15, "min_child_samples": 100, "reg_lambda": 4.0},
    {"num_leaves": 15, "min_child_samples": 250, "reg_lambda": 8.0},
    {"num_leaves": 31, "min_child_samples": 250, "reg_lambda": 8.0},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _model(params: Mapping[str, Any], *, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=240,
        learning_rate=0.035,
        max_depth=5,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.25,
        verbosity=-1,
        n_jobs=6,
        random_state=int(seed),
        **dict(params),
    )


def _matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result.to_numpy(np.float64)).all():
        raise ValueError("model feature matrix contains non-finite values")
    return result.astype(np.float32)


def _inner_split(
    frame: pd.DataFrame, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ordered_days = (
        frame.iloc[indices]["__ts__"].dt.floor("D").drop_duplicates().sort_values()
    )
    if len(ordered_days) < 10:
        raise ValueError("inner HPO requires at least ten training days")
    start = ordered_days.iloc[max(1, int(np.floor(len(ordered_days) * 0.80)))]
    valid = indices[frame.iloc[indices]["__ts__"].ge(start).to_numpy()]
    cutoff = pd.Timestamp(start) - pd.Timedelta(hours=25)
    train = indices[frame.iloc[indices]["__ts__"].lt(cutoff).to_numpy()]
    if len(train) < 500 or len(valid) < 200:
        raise ValueError("inner HPO split has insufficient purged support")
    return train, valid


def _fit_hpo(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[lgb.LGBMRegressor, dict[str, Any]]:
    matrix = _matrix(frame, features)
    inner_train, inner_valid = _inner_split(frame, train_indices)
    candidates: list[dict[str, Any]] = []
    for index, params in enumerate(PARAM_GRID):
        model = _model(params, seed=seed + index)
        model.fit(matrix.iloc[inner_train], target[inner_train])
        prediction = np.clip(model.predict(matrix.iloc[inner_valid]), 0.0, 1.0)
        loss = float(np.mean((prediction - target[inner_valid]) ** 2))
        candidates.append({"params": dict(params), "brier": loss})
    winner = min(candidates, key=lambda row: (row["brier"], str(row["params"])))
    fitted = _model(winner["params"], seed=seed + 100)
    fitted.fit(matrix.iloc[train_indices], target[train_indices])
    return fitted, {
        "selection_source": "purged trailing 20% of outer-train rows only",
        "winner": winner,
        "trials": candidates,
        "inner_train_rows": len(inner_train),
        "inner_validation_rows": len(inner_valid),
    }


def _validate_source_oof(frame: pd.DataFrame, head: str) -> None:
    fold_column, cutoff_column = HEAD_PROVENANCE[head]
    if frame[fold_column].isna().any():
        raise ValueError(f"{head}: source OOF fold is missing")
    cutoff = pd.to_datetime(frame[cutoff_column], utc=True, errors="coerce")
    if cutoff.isna().any() or not (cutoff < frame["__ts__"]).all():
        raise ValueError(f"{head}: source prediction is not strictly OOF")


def _soft_head_oof(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.loc[:, list(IDENTITY)].copy()
    reports: dict[str, Any] = {}
    folds = expanding_month_folds(frame["__ts__"])
    for head_index, head in enumerate(HEADS):
        _validate_source_oof(frame, head)
        prediction = np.full(len(frame), np.nan, dtype=np.float32)
        fold_id = np.full(len(frame), -1, dtype=np.int16)
        head_reports: list[dict[str, Any]] = []
        for fold in folds:
            train_all = np.asarray(fold["train_indices"], dtype=np.int64)
            valid_all = np.asarray(fold["validation_indices"], dtype=np.int64)
            for side_index, side in enumerate(("long", "short")):
                train = train_all[
                    frame.iloc[train_all]["side_name"].eq(side).to_numpy()
                ]
                valid = valid_all[
                    frame.iloc[valid_all]["side_name"].eq(side).to_numpy()
                ]
                model, hpo = _fit_hpo(
                    frame,
                    HEAD_FEATURES[head],
                    targets[head].to_numpy(np.float32),
                    train,
                    seed=seed
                    + head_index * 1000
                    + int(fold["fold"]) * 100
                    + side_index,
                )
                pred = np.clip(
                    model.predict(_matrix(frame.iloc[valid], HEAD_FEATURES[head])),
                    0.0,
                    1.0,
                )
                prediction[valid] = pred
                fold_id[valid] = int(fold["fold"])
                observed = targets[head].to_numpy(np.float64)[valid]
                head_reports.append(
                    {
                        "month": fold["month"],
                        "side": side,
                        "train_rows": len(train),
                        "validation_rows": len(valid),
                        "train_cutoff": fold["train_cutoff"],
                        "brier": float(np.mean((pred - observed) ** 2)),
                        "mae": float(np.mean(np.abs(pred - observed))),
                        "spearman": float(spearmanr(pred, observed).statistic),
                        "hpo": hpo,
                    }
                )
        output[f"soft_{head}"] = prediction
        output[f"soft_{head}__oof_fold"] = fold_id
        reports[head] = {
            "features": list(HEAD_FEATURES[head]),
            "folds": head_reports,
            "oof_rows": int(np.isfinite(prediction).sum()),
            "label_contract": "policy-anchored soft binary",
        }
    return output, reports


def _meta_features(frame: pd.DataFrame, arm: str) -> list[str]:
    alpha = ["existing_alpha_ev", "alpha_prediction_uncertainty", "alpha_leaf_support"]
    base = sorted(
        column for column in frame if column.startswith("base_archetype_label__")
    )
    non_timing = [
        "soft_peak_mfe_12h_atr",
        "soft_bars_before_price_stops_decreasing",
        "soft_future_slope_atr_per_hour",
    ]
    timing = [
        "soft_time_to_first_meaningful_mfe",
        "soft_mae_before_meaningful_mfe_atr",
    ]
    if arm == "alpha_context":
        return [*alpha, *base]
    if arm == "ev_non_timing_aux":
        return [*alpha, *base, *non_timing]
    if arm == "ev_all_aux":
        return [*alpha, *base, *non_timing, *timing]
    if arm.startswith("ev_soft_"):
        head = arm.removeprefix("ev_soft_")
        if head not in HEADS:
            raise ValueError(f"unknown soft auxiliary head arm: {head}")
        return [*alpha, *base, f"soft_{head}"]
    if arm.startswith("ev_original_"):
        head = arm.removeprefix("ev_original_")
        if head not in HEADS:
            raise ValueError(f"unknown original auxiliary head arm: {head}")
        return [*alpha, *base, *HEAD_FEATURES[head]]
    if arm == "timing_router_alpha_only":
        return ["existing_alpha_ev"]
    if arm == "timing_router_exclusive":
        return ["existing_alpha_ev", *timing]
    if arm == "timing_router_original":
        return [
            "existing_alpha_ev",
            *HEAD_FEATURES["time_to_first_meaningful_mfe"],
            *HEAD_FEATURES["mae_before_meaningful_mfe_atr"],
        ]
    raise ValueError(f"unknown meta arm: {arm}")


def _select_soft_ev_recipe(
    frame: pd.DataFrame,
    train: np.ndarray,
    features: Sequence[str],
    net: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, Mapping[str, Any], list[dict[str, Any]]]:
    inner_train, inner_valid = _inner_split(frame, train)
    matrix = _matrix(frame, features)
    trials: list[dict[str, Any]] = []
    for threshold in (0.0, 0.0025, 0.005, 0.0075):
        for temperature in (0.003, 0.005, 0.010):
            target = execution_ev_soft_target(
                net, threshold=threshold, temperature=temperature
            )
            for param_index, params in enumerate(PARAM_GRID):
                model = _model(params, seed=seed + len(trials) + param_index)
                model.fit(matrix.iloc[inner_train], target[inner_train])
                score = np.clip(model.predict(matrix.iloc[inner_valid]), 0.0, 1.0)
                metrics = economic_metrics(
                    frame.iloc[inner_valid].reset_index(drop=True),
                    score,
                )
                brier = float(np.mean((score - target[inner_valid]) ** 2))
                value = (
                    float(metrics["timestamp_side_top10_mean_net_return"])
                    + 0.5 * float(metrics["global_top10_mean_net_return"])
                    - 0.10 * brier
                )
                trials.append(
                    {
                        "threshold": threshold,
                        "temperature": temperature,
                        "params": dict(params),
                        "brier": brier,
                        "objective": value,
                        "economics": metrics,
                    }
                )
    winner = max(
        trials,
        key=lambda row: (
            row["objective"],
            -row["threshold"],
            -row["temperature"],
            str(row["params"]),
        ),
    )
    return (
        float(winner["threshold"]),
        float(winner["temperature"]),
        dict(winner["params"]),
        trials,
    )


def _july_meta_ablations(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    june_start = pd.Timestamp("2026-06-01", tz="UTC")
    july_start = pd.Timestamp("2026-07-01", tz="UTC")
    july_end = pd.Timestamp("2026-08-01", tz="UTC")
    train = np.flatnonzero(
        frame["__ts__"].ge(june_start).to_numpy()
        & frame["__ts__"].lt(july_start - pd.Timedelta(hours=25)).to_numpy()
    )
    valid = np.flatnonzero(
        frame["__ts__"].ge(july_start).to_numpy()
        & frame["__ts__"].lt(july_end).to_numpy()
    )
    net = frame["execution_net_ev_12h"].to_numpy(np.float64)
    scored = frame.iloc[valid].loc[:, list(IDENTITY) + ["execution_net_ev_12h"]].copy()
    reports: dict[str, Any] = {}
    arms = (
        "alpha_context",
        *tuple(f"ev_original_{head}" for head in HEADS),
        *tuple(f"ev_soft_{head}" for head in HEADS),
        "ev_non_timing_aux",
        "ev_all_aux",
        "timing_router_alpha_only",
        "timing_router_original",
        "timing_router_exclusive",
    )

    def arm_seed_offset(arm: str) -> int:
        if arm.startswith(("ev_original_", "ev_soft_")):
            head = arm.split("_", 2)[-1]
            return 1000 + 1000 * HEADS.index(head)
        if arm in ("timing_router_original", "timing_router_exclusive"):
            return 10_000
        stable = int(hashlib.sha256(arm.encode()).hexdigest()[:6], 16)
        return 20_000 + stable % 10_000

    for arm in arms:
        features = _meta_features(frame, arm)
        arm_seed = seed + arm_seed_offset(arm)
        threshold, temperature, params, trials = _select_soft_ev_recipe(
            frame,
            train,
            features,
            net,
            seed=arm_seed,
        )
        target = execution_ev_soft_target(
            net, threshold=threshold, temperature=temperature
        )
        model = _model(params, seed=arm_seed + 900)
        model.fit(_matrix(frame.iloc[train], features), target[train])
        score = np.clip(model.predict(_matrix(frame.iloc[valid], features)), 0.0, 1.0)
        scored[arm] = score
        reports[arm] = {
            "routing": (
                "timing and MAE excluded from execution-EV"
                if arm == "ev_non_timing_aux"
                else (
                    "timing and MAE only; no non-timing auxiliaries"
                    if arm == "timing_router_exclusive"
                    else "comparator"
                )
            ),
            "features": features,
            "label_threshold": threshold,
            "label_temperature": temperature,
            "params": params,
            "selection_trials": trials,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "economics": economic_metrics(scored, score),
            "admitted_score_ge_0_5": economic_metrics(
                scored, score, admitted=score >= 0.5
            ),
        }
    reports["paired_deltas_bps"] = {
        "all_aux_vs_alpha_context_global_top10": 10_000
        * (
            reports["ev_all_aux"]["economics"]["global_top10_mean_net_return"]
            - reports["alpha_context"]["economics"]["global_top10_mean_net_return"]
        ),
        "non_timing_aux_vs_all_aux_global_top10": 10_000
        * (
            reports["ev_non_timing_aux"]["economics"]["global_top10_mean_net_return"]
            - reports["ev_all_aux"]["economics"]["global_top10_mean_net_return"]
        ),
        "timing_exclusive_vs_alpha_only_global_top10": 10_000
        * (
            reports["timing_router_exclusive"]["economics"][
                "global_top10_mean_net_return"
            ]
            - reports["timing_router_alpha_only"]["economics"][
                "global_top10_mean_net_return"
            ]
        ),
        "individual_soft_head_vs_alpha_context_global_top10": {
            head: 10_000
            * (
                reports[f"ev_soft_{head}"]["economics"]["global_top10_mean_net_return"]
                - reports["alpha_context"]["economics"]["global_top10_mean_net_return"]
            )
            for head in HEADS
        },
        "soft_vs_original_head_global_top10": {
            head: 10_000
            * (
                reports[f"ev_soft_{head}"]["economics"]["global_top10_mean_net_return"]
                - reports[f"ev_original_{head}"]["economics"][
                    "global_top10_mean_net_return"
                ]
            )
            for head in HEADS
        },
        "soft_vs_original_timing_router_global_top10": 10_000
        * (
            reports["timing_router_exclusive"]["economics"][
                "global_top10_mean_net_return"
            ]
            - reports["timing_router_original"]["economics"][
                "global_top10_mean_net_return"
            ]
        ),
    }
    return scored, reports


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = pd.read_parquet(args.joined)
    mae = pd.read_parquet(
        args.mae_handoff,
        columns=[*IDENTITY, *HEAD_FEATURES["mae_before_meaningful_mfe_atr"]],
    )
    targets = pd.read_parquet(
        args.targets,
        columns=[
            *IDENTITY,
            "__path_auxiliary_atr_fraction__",
            "__peak_mfe_atr_12h__",
            "__time_to_first_meaningful_mfe_hours_12h__",
            "__mae_before_meaningful_mfe_atr_12h__",
            "__bars_to_confirmed_adverse_trough__",
            "__future_slope_atr_per_hour_12h__",
            "__meaningful_mfe_reached_12h__",
        ],
    )
    policy = pd.read_parquet(
        args.policy_labels, columns=[*IDENTITY, "execution_net_ev_12h"]
    )
    for frame in (joined, mae, targets, policy):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        if frame.duplicated(list(IDENTITY)).any():
            raise ValueError("source contains duplicate exact candidate identities")
    mae_extra = [
        column
        for column in HEAD_FEATURES["mae_before_meaningful_mfe_atr"]
        if column not in joined
    ]
    work = joined.merge(
        mae.loc[:, [*IDENTITY, *mae_extra]],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    work = work.merge(targets, on=list(IDENTITY), how="inner", validate="one_to_one")
    if "execution_net_ev_12h" in work:
        work = work.drop(columns="execution_net_ev_12h")
    work = work.merge(policy, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(work) != len(policy):
        raise ValueError(
            f"exact paired join lost rows: policy={len(policy)} joined={len(work)}"
        )
    work = work.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    return work, auxiliary_soft_targets(work)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    policy_manifest = json.loads(args.policy_manifest.read_text(encoding="utf-8"))
    exit_contract = policy_manifest.get("exit_policy_contract", {})
    if (
        policy_manifest.get("schema") != "execution_ev_deployed_policy_1m_labels_v1"
        or exit_contract.get("replay_timeframe") != "1m"
        or int(exit_contract.get("horizon_minutes", 0)) != 1440
    ):
        raise ValueError("policy labels are not the exact 1m/1,440m contract")
    frame, soft_targets = _load(args)
    head_oof, head_reports = _soft_head_oof(frame, soft_targets, seed=args.seed)
    head_oof.to_parquet(args.output / "soft_auxiliary_oof.parquet", index=False)
    merged = frame.merge(
        head_oof,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    complete = np.ones(len(merged), dtype=bool)
    for head in HEADS:
        complete &= np.isfinite(merged[f"soft_{head}"].to_numpy(np.float64))
    meta_frame = merged.loc[complete].reset_index(drop=True)
    scored, meta_reports = _july_meta_ablations(meta_frame, seed=args.seed + 50_000)
    scored.to_parquet(args.output / "july_exact_policy_oof_scores.parquet", index=False)
    summary = {
        "schema": "exact_policy_soft_binary_ablations_v1",
        "status": "research_ablation_complete_not_automatically_promoted",
        "exact_policy_contract": exit_contract,
        "paired_identity_rows": len(frame),
        "soft_auxiliary_oof_rows": len(meta_frame),
        "soft_auxiliary_heads": head_reports,
        "meta_ablations": meta_reports,
        "chronology": {
            "auxiliary_outer_oof": "June and July expanding train-before-validation",
            "execution_ev_and_timing_selection": "June only",
            "final_evaluation": "July only",
            "purge_hours": 25,
        },
        "sources": {
            "joined": {"path": args.joined, "sha256": _sha256(args.joined)},
            "mae_handoff": {
                "path": args.mae_handoff,
                "sha256": _sha256(args.mae_handoff),
            },
            "targets": {"path": args.targets, "sha256": _sha256(args.targets)},
            "policy_labels": {
                "path": args.policy_labels,
                "sha256": _sha256(args.policy_labels),
            },
            "policy_manifest": {
                "path": args.policy_manifest,
                "sha256": _sha256(args.policy_manifest),
            },
        },
        "limitations": [
            "soft auxiliary challengers are OOF recalibration layers over frozen OOF head outputs, not native raw-feature refits",
            "timing routing is an exact-policy enter/skip gate; counterfactual wait-price actions remain a separate experiment",
            "final paired meta evaluation is July only because June soft-head OOF is required for training",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--mae-handoff", type=Path, default=DEFAULT_MAE)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--policy-labels", type=Path, default=DEFAULT_POLICY_LABELS)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_safe(run(parse_args())), indent=2))
