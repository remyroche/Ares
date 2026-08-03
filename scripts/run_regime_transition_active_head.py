#!/usr/bin/env python3
"""Grouped pooled active-transition head on the symmetric research labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_regime_transition_classifier_ablation import _groups  # noqa: E402


DEFAULT_DATASET = Path(
    "data_perp/artifacts/regime_transition_research_20260726_v3/"
    "hourly_transition_dataset.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/regime_transition_active_head_20260726_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    frame = pd.read_parquet(args.dataset)
    frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True)
    features = [
        name
        for name in frame
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
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    y = frame["target__transition_active"].astype(int).to_numpy()
    groups = _groups(frame)
    prediction = np.full(len(frame), np.nan, np.float32)
    splitter = StratifiedGroupKFold(
        n_splits=int(args.folds), shuffle=True, random_state=2219
    )
    for fold, (train, evaluation) in enumerate(
        splitter.split(x, y, groups=groups)
    ):
        model = LGBMClassifier(
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
            random_state=2219 + fold,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(x.iloc[train], y[train])
        prediction[evaluation] = model.predict_proba(x.iloc[evaluation])[:, 1]
    hard = prediction >= 0.5
    report = {
        "schema": "pooled_active_transition_head_v1",
        "research_only": True,
        "rows": len(frame),
        "events": int(frame["target__event_id"].dropna().nunique()),
        "feature_count": len(features),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "f1_at_0_5": float(f1_score(y, hard)),
    }
    frame[
        ["source_utc", "target__event_id", "target__transition_active"]
    ].assign(prediction=prediction).to_parquet(
        output / "grouped_oof.parquet", index=False
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
