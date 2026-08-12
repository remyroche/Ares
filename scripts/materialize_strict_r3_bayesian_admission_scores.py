#!/usr/bin/env python3
"""Project strict-prequential Bayesian trust outputs into causal admission scores.

The source Bayesian predictions may contain realised outcomes for later
evaluation, but this projection reads only candidate identity, decision time,
and posterior fields.  The separate admission runner joins outcomes after the
score is fixed and applies the prior-resolved 21-day EV map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-arm", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    fields = [
        "candidate_id", "__decision_ts__", "arm", "final_score", "posterior_expected_bps",
        "posterior_predictive_q10", "p_ev_positive", "p_adverse_tail",
    ]
    frame = pd.read_parquet(args.predictions, columns=fields)
    frame = frame.loc[frame["arm"].eq(args.source_arm)].copy()
    if frame.empty:
        raise ValueError(f"source arm {args.source_arm!r} is absent")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("source arm has duplicate candidate identities")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    definitions = {
        "final_score_control": pd.to_numeric(frame["final_score"], errors="coerce"),
        "bayes_posterior_mean_bps": pd.to_numeric(frame["posterior_expected_bps"], errors="coerce"),
        "bayes_posterior_q10_bps": pd.to_numeric(frame["posterior_predictive_q10"], errors="coerce"),
        "bayes_probability_utility": (
            pd.to_numeric(frame["p_ev_positive"], errors="coerce")
            - 0.5 * pd.to_numeric(frame["p_adverse_tail"], errors="coerce")
        ),
    }
    output: list[pd.DataFrame] = []
    for arm, score in definitions.items():
        if not np.isfinite(score).all():
            raise ValueError(f"{arm} has non-finite score values")
        part = frame.loc[:, ["candidate_id", "__decision_ts__"]].copy()
        part["corrected_score"] = score.to_numpy(float)
        part["arm"] = arm
        output.append(part)
    result = pd.concat(output, ignore_index=True)
    args.out_dir.mkdir(parents=True)
    result.to_parquet(args.out_dir / "predictions.parquet", index=False, compression="zstd")
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_bayesian_admission_score_projection_v1",
        "source_predictions": str(args.predictions), "source_arm": args.source_arm,
        "arms": list(definitions),
        "score_construction": "uses only precomputed strict-prequential Bayesian posterior fields; no outcome columns read",
        "admission": "external causal prior-resolved 21-day map required",
        "raw_k9_posterior_coordinates": False,
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": int(len(result)), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
