#!/usr/bin/env python3
"""Strict-OOF trust and mapping-change ablation for the frozen EV score.

The head is deliberately not allowed to replace the frozen score with a raw
probability.  It learns three side-local quantities on resolved prior labels:

* residual economic utility: realised net EV minus frozen mapped EV;
* expected absolute mapping error, in the same net-return units; and
* trust that the base signal is both economically positive and approximately
  calibrated.

The quantities are used only as a residual interaction, an uncertainty
penalty, or an eligibility/abstention input before one pooled global top-k.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_execution_ev_trust_abstention_ablation import (  # noqa: E402
    DECISION,
    DEFAULT_COSTS,
    DEFAULT_FEATURES,
    DEFAULT_SCORES,
    FROZEN_RANK_SCORE,
    IDENTITY,
    RESOLVED,
    SIDE,
    TARGET,
    global_top_fraction_mask,
    prepare_inputs,
    weekly_purged_folds,
)

DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "execution_ev_trust_base_signal_mapping_20260726_v1"
)
SCHEMA = "execution_ev_trust_base_signal_mapping_v1"
ARMS = (
    "baseline",
    "residual_shrink_0.25",
    "residual_shrink_0.50",
    "residual_shrink_1.00",
    "trust_residual_interaction",
    "mapping_uncertainty_penalty",
    "trust_abstention",
    "combined_trust_mapping",
)


def trust_mapping_targets(
    net_ev: Sequence[float],
    frozen_ev: Sequence[float],
    *,
    utility_temperature: float = 0.005,
    mapping_temperature: float = 0.020,
) -> dict[str, np.ndarray]:
    """Return residual, absolute error and soft trust targets.

    Trust is high only when realised utility is positive and the frozen
    common-unit EV map is not badly wrong.  Both components are continuous.
    """

    realised = np.asarray(net_ev, dtype=np.float64)
    frozen = np.asarray(frozen_ev, dtype=np.float64)
    residual = realised - frozen
    positive_utility = 1.0 / (
        1.0
        + np.exp(
            -np.clip(realised / float(utility_temperature), -40.0, 40.0)
        )
    )
    mapping_reliability = np.exp(
        -np.abs(residual) / float(mapping_temperature)
    )
    trust = np.clip(positive_utility * mapping_reliability, 0.0, 1.0)
    return {
        "residual_utility": residual.astype(np.float32),
        "absolute_mapping_error": np.abs(residual).astype(np.float32),
        "trust_base_signal": trust.astype(np.float32),
    }


def _fit_regressor(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    evaluation_x: pd.DataFrame,
    *,
    seed: int,
    iterations: int,
    n_jobs: int,
) -> np.ndarray:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=int(iterations),
        learning_rate=0.04,
        depth=5,
        l2_leaf_reg=12.0,
        random_strength=0.5,
        bagging_temperature=1.0,
        bootstrap_type="Bayesian",
        random_seed=int(seed),
        thread_count=int(n_jobs),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_x, train_y)
    return np.asarray(model.predict(evaluation_x), dtype=np.float64)


def _fit_soft_classifier(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    evaluation_x: pd.DataFrame,
    *,
    seed: int,
    iterations: int,
    n_jobs: int,
) -> np.ndarray:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="CrossEntropy",
        iterations=int(iterations),
        learning_rate=0.04,
        depth=5,
        l2_leaf_reg=12.0,
        random_strength=0.5,
        bagging_temperature=1.0,
        bootstrap_type="Bayesian",
        random_seed=int(seed),
        thread_count=int(n_jobs),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_x, train_y)
    return np.asarray(model.predict_proba(evaluation_x)[:, 1], dtype=np.float64)


def fit_side_local_heads(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    targets: Mapping[str, np.ndarray],
    train_positions: np.ndarray,
    *,
    seed: int,
    iterations: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit three independent heads for each side."""

    residual = np.full(len(validation), np.nan, dtype=np.float64)
    error = np.full(len(validation), np.nan, dtype=np.float64)
    trust = np.full(len(validation), np.nan, dtype=np.float64)
    for side_index, side in enumerate(("long", "short")):
        train_mask = train[SIDE].astype(str).eq(side).to_numpy()
        validation_mask = validation[SIDE].astype(str).eq(side).to_numpy()
        if not train_mask.any() or not validation_mask.any():
            raise ValueError(f"empty side-local rows for {side}")
        train_x = train.loc[train_mask, list(features)]
        validation_x = validation.loc[validation_mask, list(features)]
        absolute_positions = train_positions[train_mask]
        residual[validation_mask] = _fit_regressor(
            train_x,
            targets["residual_utility"][absolute_positions],
            validation_x,
            seed=seed + side_index * 10,
            iterations=iterations,
            n_jobs=n_jobs,
        )
        error[validation_mask] = np.maximum(
            0.0,
            _fit_regressor(
                train_x,
                targets["absolute_mapping_error"][absolute_positions],
                validation_x,
                seed=seed + side_index * 10 + 1,
                iterations=iterations,
                n_jobs=n_jobs,
            ),
        )
        trust[validation_mask] = _fit_soft_classifier(
            train_x,
            targets["trust_base_signal"][absolute_positions],
            validation_x,
            seed=seed + side_index * 10 + 2,
            iterations=iterations,
            n_jobs=n_jobs,
        )
    if not (
        np.isfinite(residual).all()
        and np.isfinite(error).all()
        and np.isfinite(trust).all()
    ):
        raise ValueError("side-local trust heads left non-finite predictions")
    return residual, error, trust


def compose_arms(
    frozen_score: np.ndarray,
    residual: np.ndarray,
    expected_error: np.ndarray,
    trust: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compose only residual/context/abstention uses of the learned heads."""

    all_rows = np.ones(len(frozen_score), dtype=bool)
    residual_050 = frozen_score + 0.50 * residual
    return {
        "baseline": (frozen_score.copy(), all_rows),
        "residual_shrink_0.25": (
            frozen_score + 0.25 * residual,
            all_rows,
        ),
        "residual_shrink_0.50": (residual_050, all_rows),
        "residual_shrink_1.00": (frozen_score + residual, all_rows),
        "trust_residual_interaction": (
            frozen_score + 0.50 * residual * trust,
            all_rows,
        ),
        "mapping_uncertainty_penalty": (
            residual_050 - 0.25 * expected_error,
            all_rows,
        ),
        "trust_abstention": (residual_050, trust >= 0.40),
        "combined_trust_mapping": (
            frozen_score
            + 0.50 * residual * trust
            - 0.25 * expected_error,
            trust >= 0.40,
        ),
    }


def run_strict_oof(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    n_jobs: int,
    seed: int,
    top_fraction: float,
) -> pd.DataFrame:
    targets = trust_mapping_targets(
        frame[TARGET].to_numpy(np.float64),
        frame[FROZEN_RANK_SCORE].to_numpy(np.float64),
    )
    parts: list[pd.DataFrame] = []
    for fold in folds:
        train_positions = np.asarray(fold["train_positions"], dtype=np.int64)
        validation_positions = np.asarray(
            fold["validation_positions"], dtype=np.int64
        )
        train = frame.iloc[train_positions].reset_index(drop=True)
        validation = frame.iloc[validation_positions].reset_index(drop=True)
        week_start = pd.Timestamp(fold["week_start"])
        if not (
            train[RESOLVED].lt(week_start).all()
            and train[DECISION].lt(
                week_start - pd.Timedelta(hours=12)
            ).all()
        ):
            raise AssertionError("trust fold violates purge/resolution")
        residual, error, trust = fit_side_local_heads(
            train,
            validation,
            features,
            targets,
            train_positions,
            seed=seed + 100 * int(fold["fold"]),
            iterations=iterations,
            n_jobs=n_jobs,
        )
        frozen = validation[FROZEN_RANK_SCORE].to_numpy(np.float64)
        for arm, (score, eligible) in compose_arms(
            frozen, residual, error, trust
        ).items():
            selected = global_top_fraction_mask(
                score,
                eligible=eligible,
                population_rows=len(validation),
                fraction=top_fraction,
            )
            part = validation.loc[
                :,
                [
                    *IDENTITY,
                    DECISION,
                    RESOLVED,
                    TARGET,
                    "evaluation_origin",
                    FROZEN_RANK_SCORE,
                ],
            ].copy()
            part["fold"] = int(fold["fold"])
            part["week_start"] = week_start
            part["arm"] = arm
            part["predicted_residual_utility"] = residual
            part["predicted_absolute_mapping_error"] = error
            part["trust_base_signal"] = trust
            part["ranking_score"] = score
            part["eligible"] = eligible
            part["weekly_selected_diagnostic"] = selected
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def apply_pooled_global_topk(
    predictions: pd.DataFrame,
    *,
    top_fraction: float,
) -> pd.DataFrame:
    output = predictions.copy()
    output["pooled_global_selected"] = False
    for _, group in output.groupby("arm", sort=True):
        selected = global_top_fraction_mask(
            group["ranking_score"].to_numpy(np.float64),
            eligible=group["eligible"].to_numpy(bool),
            population_rows=len(group),
            fraction=top_fraction,
        )
        output.loc[group.index, "pooled_global_selected"] = selected
    return output


def economic_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    decision = pd.to_datetime(work[DECISION], utc=True)
    work["_month"] = decision.dt.strftime("%Y-%m")
    latest_week = pd.to_datetime(work["week_start"], utc=True).max()
    rows: list[dict[str, Any]] = []
    for arm, group in work.groupby("arm", sort=True):
        segments = [("all", "all", group)]
        segments.extend(
            ("month", str(month), local)
            for month, local in group.groupby("_month", sort=True)
        )
        segments.append(
            (
                "latest_week",
                latest_week.isoformat(),
                group.loc[
                    pd.to_datetime(group["week_start"], utc=True).eq(
                        latest_week
                    )
                ],
            )
        )
        for segment_type, segment, local in segments:
            selected = local.loc[local["pooled_global_selected"]]
            eligible = local.loc[local["eligible"]]
            positive = eligible[TARGET].gt(0.0)
            auc = (
                float(
                    roc_auc_score(
                        positive.astype(np.int8),
                        eligible["ranking_score"],
                    )
                )
                if positive.nunique() == 2
                else np.nan
            )
            rows.append(
                {
                    "arm": arm,
                    "segment_type": segment_type,
                    "segment": segment,
                    "evaluation_rows": int(len(local)),
                    "eligible_rows": int(len(eligible)),
                    "selected_rows": int(len(selected)),
                    "selection_coverage": float(
                        len(selected) / max(len(local), 1)
                    ),
                    "selected_mean_net_ev": (
                        float(selected[TARGET].mean())
                        if len(selected)
                        else np.nan
                    ),
                    "selected_mean_net_ev_bps": (
                        float(selected[TARGET].mean() * 10_000.0)
                        if len(selected)
                        else np.nan
                    ),
                    "selected_positive_rate": (
                        float(selected[TARGET].gt(0.0).mean())
                        if len(selected)
                        else np.nan
                    ),
                    "rank_positive_auc": auc,
                    "rank_net_ev_spearman": (
                        float(
                            spearmanr(
                                eligible["ranking_score"],
                                eligible[TARGET],
                            ).statistic
                        )
                        if len(eligible) > 1
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def head_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    base = predictions.loc[predictions["arm"].eq("baseline")].copy()
    base["_month"] = pd.to_datetime(base[DECISION], utc=True).dt.strftime(
        "%Y-%m"
    )
    rows: list[dict[str, Any]] = []
    for segment, group in [("all", base), *base.groupby("_month", sort=True)]:
        residual = (
            group[TARGET].to_numpy(np.float64)
            - group[FROZEN_RANK_SCORE].to_numpy(np.float64)
        )
        trust_positive = group[TARGET].gt(0.0)
        rows.append(
            {
                "segment": str(segment),
                "rows": int(len(group)),
                "residual_spearman": float(
                    spearmanr(
                        group["predicted_residual_utility"], residual
                    ).statistic
                ),
                "absolute_error_spearman": float(
                    spearmanr(
                        group["predicted_absolute_mapping_error"],
                        np.abs(residual),
                    ).statistic
                ),
                "trust_positive_auc": (
                    float(
                        roc_auc_score(
                            trust_positive.astype(np.int8),
                            group["trust_base_signal"],
                        )
                    )
                    if trust_positive.nunique() == 2
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--costs", type=Path, default=DEFAULT_COSTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--min-train-rows", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, features, preparation = prepare_inputs(
        pd.read_parquet(args.scores),
        pd.read_parquet(args.features),
        pd.read_parquet(args.costs),
    )
    folds = weekly_purged_folds(
        frame,
        purge_hours=12.0,
        min_train_rows=args.min_train_rows,
    )
    predictions = run_strict_oof(
        frame,
        features,
        folds,
        iterations=args.iterations,
        n_jobs=args.n_jobs,
        seed=args.seed,
        top_fraction=args.top_fraction,
    )
    predictions = apply_pooled_global_topk(
        predictions, top_fraction=args.top_fraction
    )
    economics = economic_metrics(predictions)
    heads = head_metrics(predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output / "strict_oof_predictions.parquet")
    economics.to_csv(args.output / "economic_metrics.csv", index=False)
    heads.to_csv(args.output / "head_metrics.csv", index=False)
    report = {
        "schema": SCHEMA,
        "status": "completed_research_oof_not_promotion_eligible",
        "contract": {
            "training": (
                "side-local expanding weekly OOF; train decisions purged 12h "
                "and labels resolved strictly before evaluation week"
            ),
            "trust_target": (
                "sigmoid(realised net EV / 50bps) multiplied by "
                "exp(-abs(realised-frozen EV)/200bps)"
            ),
            "mapping_target": (
                "realised cost-adjusted net EV minus frozen mapped EV"
            ),
            "score_use": (
                "trust is never a raw ranking score; only residual "
                "interaction, uncertainty penalty, or abstention input"
            ),
            "selection": "one pooled global top10 across all OOF rows",
        },
        "preparation": preparation,
        "features": list(features),
        "folds": [
            {
                key: value
                for key, value in fold.items()
                if key not in {"train_positions", "validation_positions"}
            }
            for fold in folds
        ],
        "arms": list(ARMS),
        "outputs": {
            "predictions": str(
                args.output / "strict_oof_predictions.parquet"
            ),
            "economics": str(args.output / "economic_metrics.csv"),
            "heads": str(args.output / "head_metrics.csv"),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(economics.to_string(index=False))
    print(heads.to_string(index=False))


if __name__ == "__main__":
    main()
