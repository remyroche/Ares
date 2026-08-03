#!/usr/bin/env python3
"""Test whether old55 model-health adds to market transition recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_regime_transition_classifier_ablation import _groups  # noqa: E402


DEFAULT_DATASET = Path(
    "data_perp/artifacts/regime_transition_research_20260726_v3/"
    "hourly_transition_dataset.parquet"
)
DEFAULT_HEALTH = Path(
    "data_perp/artifacts/regime_transition_model_health_20260726_v1/"
    "hourly_model_health.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/regime_transition_model_health_ablation_20260726_v1"
)


def _model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=8.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def _prediction(
    frame: pd.DataFrame, features: list[str], *, folds: int, seed: int
) -> np.ndarray:
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    y = frame["target__onset_within_3h"].astype(int).to_numpy()
    groups = _groups(frame)
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    prediction = np.full(len(frame), np.nan, np.float32)
    for fold, (train, evaluation) in enumerate(
        splitter.split(x, y, groups=groups)
    ):
        model = _model(seed + fold)
        model.fit(x.iloc[train], y[train])
        prediction[evaluation] = model.predict_proba(x.iloc[evaluation])[:, 1]
    return prediction


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--health", type=Path, default=DEFAULT_HEALTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    market = pd.read_parquet(args.dataset)
    health = pd.read_parquet(args.health)
    for frame in (market, health):
        frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True)
    frame = market.merge(
        health.drop(columns="execution_decision_utc"),
        on="source_utc",
        how="inner",
        validate="one_to_one",
    )
    health_features = [
        name for name in frame if name.startswith("health__")
    ]
    market_features = [
        name
        for name in frame
        if name
        not in {
            "source_utc",
            "execution_decision_utc",
            "segment_id",
            "target__pooled_state",
        }
        and not name.startswith(("target__", "health__"))
        and pd.api.types.is_numeric_dtype(frame[name])
    ]
    sets = {
        "market_only_same_period": market_features,
        "model_health_only": health_features,
        "market_plus_model_health": market_features + health_features,
    }
    y = frame["target__onset_within_3h"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    oof = frame[
        ["source_utc", "target__event_id", "target__onset_within_3h"]
    ].copy()
    for index, (name, features) in enumerate(sets.items()):
        prediction = _prediction(
            frame, features, folds=int(args.folds), seed=1901 + index * 10
        )
        rows.append(
            {
                "feature_set": name,
                "feature_count": len(features),
                "rows": len(frame),
                "events": int(frame["target__event_id"].dropna().nunique()),
                "prevalence": float(y.mean()),
                "average_precision": float(
                    average_precision_score(y, prediction)
                ),
                "roc_auc": float(roc_auc_score(y, prediction)),
                "brier": float(brier_score_loss(y, prediction)),
            }
        )
        oof[f"prediction__{name}"] = prediction
    metrics = pd.DataFrame(rows).sort_values(
        ["average_precision", "roc_auc"], ascending=False
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    oof.to_parquet(output / "grouped_oof.parquet", index=False)
    report = {
        "schema": "old55_model_health_transition_ablation_v1",
        "research_only": True,
        "lineage_caveat": "old55 model-health, not current-model parity",
        "coverage_start": str(frame["source_utc"].min()),
        "coverage_end": str(frame["source_utc"].max()),
        "winner": metrics.iloc[0].to_dict(),
        "metrics": metrics.to_dict("records"),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
