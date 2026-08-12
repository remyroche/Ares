#!/usr/bin/env python3
"""Long-only reliability-target boundary ablation with historical support."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frozen_conversion_reliability_learner_ablation import (
    CLASS_CENTERS, FEATURES, _join,
)

OUT = ROOT / "data_perp/artifacts/long_only_reliability_boundary_ablation_20260810_v1"
BOUNDARIES = (50.0, 75.0, 100.0)
ALPHAS = (0.25, 0.50, 1.00)
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)


def _fit_classifier(train: pd.DataFrame, test: pd.DataFrame, boundary: float) -> np.ndarray:
    if len(train) < 500:
        return np.zeros(len(test), dtype=float)
    labels = np.select(
        (train.residual_target <= -boundary, train.residual_target >= boundary),
        (0, 2), default=1,
    ).astype(np.int32)
    med = train[FEATURES].median(numeric_only=True)
    x_train = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).astype(np.float32)
    x_test = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).astype(np.float32)
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=160, learning_rate=.03,
        max_depth=4, num_leaves=15, min_child_samples=500, feature_fraction=.80,
        bagging_fraction=.80, bagging_freq=1, reg_lambda=20.0, random_state=20260810,
        n_jobs=1, verbosity=-1,
    )
    model.fit(x_train, labels)
    proba = model.predict_proba(x_test)
    full = np.zeros((len(test), 3), dtype=float)
    for col, cls in enumerate(model.classes_.astype(int)):
        full[:, cls] = proba[:, col]
    return np.clip(full @ CLASS_CENTERS, -200.0, 200.0)


def _metric(frame: pd.DataFrame, score: str, arm: str, period: str, selection: str) -> dict[str, object]:
    n = max(1, int(np.ceil(len(frame) * .05)))
    chosen = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
    return {
        "arm": arm, "period": period, "selection": selection, "tail": .05,
        "population_rows": int(len(frame)), "selected_rows": int(len(chosen)),
        "gross_bps": float(chosen.gross_bps.mean()), "net_bps": float(chosen.net_bps.mean()),
        "rank_ic": float(frame[score].rank().corr(frame.net_bps.rank())),
    }


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    frame = _join()
    frame = frame.loc[frame.side_name.eq("long")].copy()
    frame["month"] = frame["__ts__"].dt.to_period("M").astype(str)
    frame["residual_target"] = frame["net_bps"] - frame["score"]
    for boundary in BOUNDARIES:
        frame[f"corr_b{int(boundary)}"] = 0.0
        for month in sorted(frame.month.unique()):
            start = pd.Timestamp(month, tz="UTC")
            prior = frame.label_available_ts.lt(start) & frame.__ts__.lt(start)
            current = frame.index[frame.month.eq(month)].to_numpy()
            if len(current):
                frame.loc[current, f"corr_b{int(boundary)}"] = _fit_classifier(frame.loc[prior], frame.loc[current], boundary)
        for alpha in ALPHAS:
            frame[f"score_b{int(boundary)}_a{alpha:g}"] = frame.score + alpha * frame[f"corr_b{int(boundary)}"]
    # Preserve the full strict-OOS chronology (primary historical support plus
    # transport months) for cross-era diagnostics.  The existing transport
    # artifact below remains unchanged in scope.
    frame.loc[frame.side_name.eq("long")].to_parquet(
        out / "all_long_only_predictions.parquet", index=False, compression="zstd"
    )
    # Copy transport only after all candidate score columns have been materialized.
    transport = frame.loc[frame.fold.astype(str).str.startswith("transport")].copy()
    rows: list[dict[str, object]] = []
    arms = {"raw_control": "score"}
    for boundary in BOUNDARIES:
        for alpha in ALPHAS:
            arms[f"b{int(boundary)}_a{alpha:g}"] = f"score_b{int(boundary)}_a{alpha:g}"
    for arm, score in arms.items():
        rows.append({**_metric(transport, score, arm, "all_transport", "long_only_global")})
        dev = transport.loc[transport.month.lt("2024-11")]
        nov = transport.loc[transport.month.eq("2024-11")]
        rows.append({**_metric(dev, score, arm, "jul_oct", "selection_dev")})
        rows.append({**_metric(nov, score, arm, "november", "untouched_oos")})
        for month, group in transport.groupby("month", sort=True):
            rows.append({**_metric(group, score, arm, month, "monthly_diagnostic")})
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(out / "long_only_metrics.parquet", index=False, compression="zstd")
    transport_cols = ["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "score", *arms.values()]
    transport.loc[:, list(dict.fromkeys(transport_cols))].to_parquet(out / "long_only_predictions.parquet", index=False, compression="zstd")
    selected = metrics[(metrics.selection.eq("selection_dev"))].sort_values("net_bps", ascending=False).iloc[0]
    manifest = {
        "schema": "long_only_reliability_boundary_ablation_v1",
        "side": "long_only", "short_rows_used": 0,
        "input_sources": "current frozen ATR2/q4h + primary historical frozen ATR2/q4h",
        "boundaries_bps": BOUNDARIES, "alphas": ALPHAS,
        "target": "three-class under/accurate/over residual around frozen score",
        "selection": "highest July-October long-only top-5 net; November untouched",
        "selected_dev_arm": str(selected.arm), "selected_dev_net_bps": float(selected.net_bps),
        "rows": int(len(transport)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(run())
