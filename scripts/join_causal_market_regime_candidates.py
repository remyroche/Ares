#!/usr/bin/env python3
"""Join an already-frozen hourly regime timeline to candidate rows.

This keeps candidate materialisation cheap when the same causal OOF hourly
state sidecar is evaluated against several score populations.  It does not fit
or transform a regime model: candidates are joined backward only, and the
frozen timeline provenance is retained verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import (  # noqa: E402
    IDENTITY_COLUMNS,
    PROVENANCE_COLUMNS,
    asof_join_regime_timeline,
    validate_candidate_identity,
    validate_combined_regime_transition_outputs,
)


def join(*, timeline_path: Path, candidates_path: Path, output_path: Path, max_lag_hours: int = 2) -> Path:
    """Write one exact candidate-keyed, causal regime sidecar atomically."""
    timeline_path, candidates_path, output_path = map(Path, (timeline_path, candidates_path, output_path))
    if output_path.exists():
        raise FileExistsError(output_path)
    timeline = pd.read_parquet(timeline_path)
    candidates = validate_candidate_identity(pd.read_parquet(candidates_path)).loc[:, list(IDENTITY_COLUMNS)].copy()
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    source = pd.to_datetime(timeline["source_utc"], utc=True, errors="raise")
    # The frozen hourly timeline is an OOF evaluation population, not a claim
    # to cover the score ledger's earlier/later rows.  Restrict explicitly
    # before the as-of guard so uncovered rows cannot be silently treated as
    # missing regime values.
    candidates = candidates.loc[
        candidates["__ts__"].ge(source.min())
        & candidates["__ts__"].lt(source.max() + pd.Timedelta(hours=1))
    ].copy()
    if candidates.empty:
        raise ValueError("no candidate rows overlap the frozen regime timeline")
    joined = asof_join_regime_timeline(
        candidates,
        timeline,
        by=(),
        timeline_timestamp_col="source_utc",
        max_lag=pd.Timedelta(hours=int(max_lag_hours)),
        provenance_columns=PROVENANCE_COLUMNS,
    )
    validate_combined_regime_transition_outputs(joined)
    if len(joined) != len(candidates):
        raise RuntimeError("candidate regime join lost rows")
    if (pd.to_datetime(joined["source_utc"], utc=True) > joined["__ts__"]).any():
        raise RuntimeError("candidate regime join looked ahead")
    temporary = Path(tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)[1])
    try:
        joined.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-lag-hours", type=int, default=2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _args()
    print(join(timeline_path=args.timeline, candidates_path=args.candidates, output_path=args.output, max_lag_hours=args.max_lag_hours))
