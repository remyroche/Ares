#!/usr/bin/env python3
"""Materialise target-only, strict-train input for short LambdaRank screening.

This is intentionally separate from causal features and model scores.  It
contains only label-resolved Jan--Mar rows and target grades, so the query
construction stage cannot consume April--June OOS outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_short_target_ablations_3m_oos import (
    OOS_START,
    TRAIN_START,
    SPECS,
    _existing_r3,
    _load_labels,
    _target_values,
    _valid_label,
)


def _grade_column(spec_name: str) -> str:
    return "grade_" + spec_name


def _to_grade(target: pd.Series | pd.DataFrame, *, spec_name: str) -> pd.Series:
    if isinstance(target, pd.Series):
        return target.astype("Int8")
    # LambdaRank needs ordered integer relevance.  Retain the time/margin/
    # ambiguity-weighted R3 soft semantics by discretising its economic
    # simplex score once on the training population; it is a label only, not a
    # quantile computed on future OOS rows.
    value = target["clear"] - 0.5 * target["adverse"]
    grade = np.floor(np.clip(value.to_numpy(float), 0.0, 0.999999) * 5.0).astype(np.int8)
    return pd.Series(grade, index=target.index, dtype="Int8")


def run(*, labels_root: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"output must be new: {out}")
    labels = _load_labels(labels_root)
    mask = (
        labels.__ts__.ge(TRAIN_START)
        & labels.__ts__.lt(OOS_START)
        & _valid_label(labels)
        & labels.__label_available_at__.lt(OOS_START)
    )
    source = labels.loc[mask].copy()
    if source.empty or source.candidate_id.duplicated().any():
        raise ValueError("strict LambdaRank query source is empty or duplicates candidates")
    out_frame = source.loc[:, [
        "candidate_id", "__ts__", "__decision_ts__", "side_name", "atr_bps",
        "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "__label_available_at__",
    ]].copy()
    out_frame = out_frame.rename(columns={
        "t4_tp6_sl4_gross_bps": "gross_bps",
        "t4_tp6_sl4_net_bps": "net_bps",
    })
    out_frame["entry_executable"] = True
    out_frame["fold"] = out_frame.__ts__.dt.strftime("%Y-%m")
    for spec in SPECS:
        target = _target_values(source, spec)
        out_frame[_grade_column(spec.name)] = _to_grade(target, spec_name=spec.name)
    if out_frame.filter(like="grade_").isna().any().any():
        raise AssertionError("label-valid LambdaRank population has an unassigned grade")
    out.mkdir(parents=True)
    out_frame.to_parquet(out / "short_lambdarank_query_population.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_short_lambdarank_query_population_v1",
        "side": "short",
        "source_labels": str(labels_root),
        "train_decision_window": f"[{TRAIN_START.isoformat()}, {OOS_START.isoformat()})",
        "label_availability": f"label_available_at < {OOS_START.isoformat()}",
        "future_oos_rows_used": 0,
        "rows": int(len(out_frame)),
        "grade_columns": [_grade_column(spec.name) for spec in SPECS],
        "targets": {spec.name: spec.description for spec in SPECS},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(labels_root=args.labels, out=args.out))


if __name__ == "__main__":
    main()
