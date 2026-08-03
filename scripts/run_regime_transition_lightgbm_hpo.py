#!/usr/bin/env python3
"""Nested feature-count/HPO ablation for pooled transition onset horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_regime_transition_classifier_ablation import (  # noqa: E402
    _event_metrics,
    _groups,
    _safe,
)


DEFAULT_DATASET = Path(
    "data_perp/artifacts/regime_transition_research_20260726_v3/"
    "hourly_transition_dataset.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/regime_transition_lightgbm_hpo_20260726_v1"
)


CONFIGS = (
    {"name": "regularized", "num_leaves": 15, "min_child_samples": 100},
    {"name": "balanced", "num_leaves": 31, "min_child_samples": 50},
    {"name": "expressive", "num_leaves": 63, "min_child_samples": 30},
)
TOP_K = (32, 64, 96, 10_000)


def _model(seed: int, config: dict[str, Any], estimators: int = 320) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=estimators,
        learning_rate=0.035,
        num_leaves=int(config["num_leaves"]),
        min_child_samples=int(config["min_child_samples"]),
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=8.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
        importance_type="gain",
    )


def _nested_prediction(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    config: dict[str, Any],
    top_k: int,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    y = frame[target].astype(int).to_numpy()
    groups = _groups(frame)
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    prediction = np.full(len(frame), np.nan, dtype=np.float32)
    selections: list[pd.DataFrame] = []
    for fold, (train, evaluation) in enumerate(
        splitter.split(x, y, groups=groups)
    ):
        selected = features
        if top_k < len(features):
            selector = _model(seed + fold, config, estimators=180)
            selector.fit(x.iloc[train], y[train])
            importance = pd.Series(
                selector.feature_importances_, index=features
            ).sort_values(ascending=False)
            selected = importance.head(int(top_k)).index.tolist()
            selections.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "feature": selected,
                        "selection_rank": np.arange(1, len(selected) + 1),
                    }
                )
            )
        model = _model(seed + 100 + fold, config)
        model.fit(x.iloc[train][selected], y[train])
        prediction[evaluation] = model.predict_proba(
            x.iloc[evaluation][selected]
        )[:, 1]
    return prediction, (
        pd.concat(selections, ignore_index=True)
        if selections
        else pd.DataFrame(columns=["fold", "feature", "selection_rank"])
    )


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(
            log_loss(y, np.clip(prediction, 1e-7, 1 - 1e-7))
        ),
        "prevalence": float(y.mean()),
        "lift_over_prevalence": float(
            average_precision_score(y, prediction) / max(y.mean(), 1e-12)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    frame = pd.read_parquet(args.dataset)
    frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True)
    features = [
        name
        for name in frame.columns
        if name
        not in {
            "source_utc",
            "execution_decision_utc",
            "segment_id",
            "target__pooled_state",
        }
        and not name.startswith("target__")
        and pd.api.types.is_numeric_dtype(frame[name])
    ]
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int], np.ndarray] = {}
    selection_parts: list[pd.DataFrame] = []
    for config in CONFIGS:
        for top_k in TOP_K:
            prediction, selection = _nested_prediction(
                frame,
                features=features,
                target="target__onset_within_3h",
                config=config,
                top_k=top_k,
                folds=int(args.folds),
                seed=int(args.seed),
            )
            y = frame["target__onset_within_3h"].astype(int).to_numpy()
            rows.append(
                {
                    "configuration": config["name"],
                    "top_k": min(top_k, len(features)),
                    **_metrics(y, prediction),
                }
            )
            predictions[(str(config["name"]), top_k)] = prediction
            if len(selection):
                selection["configuration"] = config["name"]
                selection["top_k"] = top_k
                selection_parts.append(selection)
    metrics = pd.DataFrame(rows).sort_values(
        ["average_precision", "roc_auc"], ascending=False
    )
    winner = metrics.iloc[0]
    winning_config = next(
        item for item in CONFIGS if item["name"] == winner["configuration"]
    )
    winning_top_k = next(
        item for item in TOP_K if min(item, len(features)) == int(winner["top_k"])
    )
    horizon_rows: list[dict[str, Any]] = []
    horizon_oof = frame[
        ["source_utc", "segment_id", "target__event_id"]
    ].copy()
    for horizon in (1, 3, 6, 12):
        target = f"target__onset_within_{horizon}h"
        if horizon == 3:
            prediction = predictions[
                (str(winner["configuration"]), winning_top_k)
            ]
        else:
            prediction, _ = _nested_prediction(
                frame,
                features=features,
                target=target,
                config=winning_config,
                top_k=winning_top_k,
                folds=int(args.folds),
                seed=int(args.seed) + horizon * 10,
            )
        y = frame[target].astype(int).to_numpy()
        horizon_rows.append(
            {"horizon_hours": horizon, **_metrics(y, prediction)}
        )
        horizon_oof[target] = y
        horizon_oof[f"prediction__{horizon}h"] = prediction
    winner_prediction = horizon_oof["prediction__3h"].to_numpy()
    event_metrics = pd.DataFrame(_event_metrics(frame, winner_prediction))
    selections = (
        pd.concat(selection_parts, ignore_index=True)
        if selection_parts
        else pd.DataFrame()
    )
    winning_selection = selections.loc[
        selections["configuration"].eq(winner["configuration"])
        & selections["top_k"].eq(winning_top_k)
    ].copy()
    selection_frequency = (
        winning_selection.groupby("feature", observed=True)
        .agg(
            folds_selected=("fold", "nunique"),
            mean_selection_rank=("selection_rank", "mean"),
        )
        .sort_values(["folds_selected", "mean_selection_rank"], ascending=[False, True])
        .reset_index()
        if len(winning_selection)
        else pd.DataFrame(
            {
                "feature": features,
                "folds_selected": int(args.folds),
                "mean_selection_rank": np.nan,
            }
        )
    )
    metrics.to_csv(output / "hpo_metrics.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(
        output / "horizon_metrics.csv", index=False
    )
    event_metrics.to_csv(output / "winner_event_metrics.csv", index=False)
    selection_frequency.to_csv(
        output / "winning_feature_selection_frequency.csv", index=False
    )
    horizon_oof.to_parquet(output / "winner_horizon_grouped_oof.parquet", index=False)
    report = {
        "schema": "pooled_transition_lightgbm_nested_hpo_v1",
        "research_only": True,
        "validation": (
            "stratified event/control-block grouped CV; feature selection "
            "repeated inside every training fold"
        ),
        "feature_pool": len(features),
        "configurations": len(CONFIGS) * len(TOP_K),
        "winner": winner.to_dict(),
        "horizons": pd.DataFrame(horizon_rows).to_dict("records"),
        "stable_winner_features": int(
            selection_frequency["folds_selected"].ge(int(args.folds) - 1).sum()
        ),
    }
    (output / "report.json").write_text(
        json.dumps(_safe(report), indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    report = run(_parser().parse_args())
    print(json.dumps(_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
