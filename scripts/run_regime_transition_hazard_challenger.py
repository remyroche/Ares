#!/usr/bin/env python3
"""Run the grouped, censor-aware transition-onset hazard challenger.

The input defaults to the canonical transition v3 research materialization.
This is pooled grouped-OOF research evidence, not an approved live model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_transition_hazard import (  # noqa: E402
    HAZARD_HORIZONS,
    TRANSITION_HAZARD_SCHEMA,
    fit_grouped_transition_hazard,
)


DEFAULT_DATASET = ROOT / "data_perp/artifacts/regime_transition_research_20260726_v3/hourly_transition_dataset.parquet"
DEFAULT_EVENTS = ROOT / "data_perp/artifacts/regime_transition_research_20260726_v3/transition_events.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/regime_transition_hazard_challenger_20260727_v1"


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    frame = pd.read_parquet(args.dataset)
    events = pd.read_parquet(args.events) if args.events.exists() else None
    family = fit_grouped_transition_hazard(
        frame,
        events=events,
        folds=args.folds,
        seed=args.seed,
        severity_weight=args.severity_weight,
    )
    labels = family.pop("labels")
    prediction = family.pop("oof_prediction")
    fold_ids = family.pop("oof_fold_ids")
    oof_models = family.pop("oof_models")
    final_model = family.pop("final_model")
    base = labels.base_mask
    oof = frame.loc[base, ["source_utc", "execution_decision_utc", "segment_id", "target__event_id", "target__time_to_onset_hours"]].copy()
    oof["oof_fold"] = fold_ids[base]
    oof["event_kind"] = labels.event_kind[base]
    oof["event_severity"] = labels.severity[base]
    oof["sample_weight"] = labels.base_weight[base]
    oof["followup_hours"] = labels.followup_hours[base]
    for column, horizon in enumerate(HAZARD_HORIZONS):
        oof[f"p_onset_within_{horizon}h"] = prediction[base, column]
    oof.to_parquet(output / "grouped_oof_cumulative_probabilities.parquet", index=False)
    design = pd.DataFrame({
        "source_utc": frame["source_utc"],
        "segment_id": frame["segment_id"],
        "base_at_risk": labels.base_mask,
        "event_id": labels.event_ids,
        "event_time_hours": labels.event_time_hours,
        "followup_hours": labels.followup_hours,
        "validation_group": labels.group_ids,
        "event_kind": labels.event_kind,
        "severity": labels.severity,
        "sample_weight": labels.base_weight,
    })
    design.to_parquet(output / "censoring_and_label_design.parquet", index=False)
    joblib.dump({"features": family["features"], "model": final_model, "schema": TRANSITION_HAZARD_SCHEMA}, output / "final_hazard_model.joblib")
    joblib.dump(oof_models, output / "oof_hazard_models.joblib")
    report = {
        "schema": TRANSITION_HAZARD_SCHEMA,
        "research_only": True,
        "dataset": str(args.dataset),
        "events": str(args.events),
        "rows": int(len(frame)),
        "base_at_risk_rows": int(base.sum()),
        "feature_count": int(len(family["features"])),
        "intervals_hours": [[0, 1], [1, 3], [3, 6], [6, 12]],
        "severity_weight": float(args.severity_weight),
        **family,
        "caveats": [
            "The canonical transition labels use pooled future state geometry and are research-only; these metrics are not walk-forward promotion evidence.",
            "Severity weighting changes the estimand toward severe transitions. Set severity_weight=0 for ordinary incidence-probability calibration.",
            "Fixed false-alert thresholds are selected on the same global grouped OOF set; freeze them on a historical period before any prospective use.",
        ],
    }
    (output / "report.json").write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2219)
    parser.add_argument("--severity-weight", type=float, default=0.25)
    report = run(parser.parse_args())
    print(json.dumps(_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
