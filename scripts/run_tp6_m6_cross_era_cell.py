#!/usr/bin/env python3
"""Run one strict cross-era M6 matrix cell from prepared compatible cohorts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_tp6_m6_cross_era_transport import (
    ERAS, _concept_rows, _matrix, _metric_rows, _model, _shift_rows,
)


def main() -> None:
    choices = [x[0] for x in ERAS]
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("single_era", "expanding_prefix"), required=True)
    ap.add_argument("--train-era", choices=choices, required=True)
    ap.add_argument("--test-era", choices=choices, required=True)
    args = ap.parse_args()
    names = choices
    train_i, test_i = names.index(args.train_era), names.index(args.test_era)
    if test_i <= train_i:
        raise ValueError("test era must be strictly later")
    if args.out.exists():
        raise FileExistsError(args.out)
    train_names = names[:train_i + 1] if args.mode == "expanding_prefix" else [args.train_era]
    train = pd.concat([pd.read_parquet(args.stage / f"{n}.parquet") for n in train_names], ignore_index=True)
    test = pd.read_parquet(args.stage / f"{args.test_era}.parquet")
    common = {"mode": args.mode, "train_era": args.train_era, "train_eras": ",".join(train_names), "test_era": args.test_era,
              "train_rows": len(train), "test_rows": len(test)}
    scored = []
    for side in ("long", "short"):
        tr = train[train.side_name.eq(side)].sort_values("__ts__", kind="mergesort")
        te = test[test.side_name.eq(side)].copy()
        model = _model().fit(_matrix(tr), tr.event)
        te["meta_probability"] = model.predict_proba(_matrix(te))[:, 1]
        scored.append(te)
    scored = pd.concat(scored, ignore_index=True)
    rows = _metric_rows(scored, "meta_probability", common)
    shifts, summary = _shift_rows(train, test, common)
    concepts = _concept_rows(scored, {**common, **summary})
    args.out.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(args.out / "metrics.parquet", index=False)
    pd.DataFrame(shifts).to_parquet(args.out / "covariate_shift.parquet", index=False)
    pd.DataFrame(concepts).to_parquet(args.out / "concept_shift.parquet", index=False)
    scored[["candidate_id", "__ts__", "side_name", "net_bps", "event", "p_clear", "meta_probability"]].to_parquet(args.out / "predictions.parquet", index=False)
    print(f"{args.mode} {args.train_era}->{args.test_era}: {len(scored):,} rows")


if __name__ == "__main__":
    main()
