#!/usr/bin/env python3
"""Materialise one bounded compatible cross-era M6 cohort.

This helper exists because the desktop runner enforces short process windows:
each exact base-OOF/context join is sealed separately before the full matrix.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pyarrow as pa

from run_tp6_m6_cross_era_transport import ERAS, FEATURES, _load_era_raw, _read_context


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", required=True, choices=[x[0] for x in ERAS])
    ap.add_argument("--stage", type=Path, required=True)
    args = ap.parse_args()
    name, start, end, source = next(x for x in ERAS if x[0] == args.era)
    args.stage.mkdir(parents=True, exist_ok=True)
    out = args.stage / f"{name}.parquet"
    if out.exists():
        raise FileExistsError(out)
    x = _load_era_raw(start, end, source)
    x = x.merge(_read_context(set(x.candidate_id)), on="candidate_id", how="inner", validate="one_to_one")
    if len(x) == 0 or (1 - x[FEATURES].replace([np.inf, -np.inf], np.nan).isna().mean() < .90).any():
        raise ValueError("empty era or <90% feature coverage")
    x["event"] = x.net_bps.gt(50).astype(int)
    x.to_parquet(out, index=False)
    print(f"{name}: {len(x):,} rows -> {out}")
    del x
    gc.collect(); pa.default_memory_pool().release_unused()


if __name__ == "__main__":
    main()
