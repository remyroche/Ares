#!/usr/bin/env python3
"""Attach a strict Bayesian score correction to its full causal score ledger.

The Bayesian model is available only after its training cutoff.  A causal
21-day EV map nevertheless needs preceding resolved score/outcome history.
This materializer keeps every earlier row at its canonical ``final_score`` and
alters only the held Bayesian identities.  It is therefore the required input
to an admission-map test: it neither fits a model nor reuses held outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quality_percentile(predictions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the frozen within-timestamp B5 correction convention."""

    ordered = predictions.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    ).copy()
    position = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    count = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    active = position < np.maximum(1, np.ceil(count * 0.30).astype(int))
    base_percentile = np.full(len(ordered), 0.5, dtype=float)
    base_percentile[active] = 1.0 - (position[active] / count[active])
    quality = (
        pd.to_numeric(ordered["posterior_expected_rank_train"], errors="coerce").fillna(0.5)
        - 0.5 * pd.to_numeric(ordered["posterior_adverse_rank_train"], errors="coerce").fillna(0.5)
    )
    ranked = ordered.loc[active, ["candidate_id"]].copy()
    ranked["quality"] = quality.loc[active].to_numpy(float)
    ranked = ranked.sort_values(["quality", "candidate_id"], ascending=[True, True], kind="stable")
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["timestamp_count"] = ordered.loc[active].groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    # ``ranked`` is ordered across timestamps; reattach each timestamp count
    # explicitly to avoid accidentally treating the full held set as one query.
    ranked = ranked.merge(
        ordered.loc[active, ["candidate_id", "__decision_ts__"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    ranked["timestamp_rank"] = ranked.groupby("__decision_ts__", sort=False).cumcount() + 1
    ranked["timestamp_count"] = ranked.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    ranked["quality_percentile"] = ranked["timestamp_rank"] / ranked["timestamp_count"]
    joined = ordered[["candidate_id"]].merge(
        ranked[["candidate_id", "quality_percentile"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    by_id = pd.Series(joined["quality_percentile"].fillna(0.5).to_numpy(float), index=ordered["candidate_id"].astype(str))
    active_by_id = pd.Series(active, index=ordered["candidate_id"].astype(str))
    ids = predictions["candidate_id"].astype(str)
    base_by_id = pd.Series(base_percentile, index=ordered["candidate_id"].astype(str))
    return (
        active_by_id.reindex(ids).to_numpy(bool),
        by_id.reindex(ids).to_numpy(float),
        base_by_id.reindex(ids).to_numpy(float),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ledger", type=Path, required=True)
    parser.add_argument("--bayesian-predictions", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--correction-mode",
        choices=("percentile_additive", "intra_timestamp_rank_delta"),
        default="percentile_additive",
        help="Use rank-delta to reorder the active set without a broad score-level shift.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    if not 0.0 < args.alpha <= 0.20:
        raise ValueError("alpha must be in (0, 0.20]")

    required_base = [
        "candidate_id", "__decision_ts__", "side_name", "final_score",
        "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
        "stack_is_prequential", "conversion_bundle_sha256",
        "upstream_bundle_sha256", "geometry_bundle_sha256",
        "ev_score_family_id",
    ]
    baseline = pd.read_parquet(args.baseline_ledger, columns=required_base)
    prediction_columns = [
        "candidate_id", "__decision_ts__", "final_score", "bayes_available",
        "posterior_expected_rank_train", "posterior_adverse_rank_train",
    ]
    posterior = pd.read_parquet(args.bayesian_predictions, columns=prediction_columns)
    for frame, name in ((baseline, "baseline"), (posterior, "Bayesian predictions")):
        if frame["candidate_id"].duplicated().any():
            raise AssertionError(f"{name} contains duplicate candidate identities")
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if not baseline["stack_is_prequential"].fillna(False).astype(bool).all():
        raise AssertionError("baseline ledger contains non-prequential rows")
    overlap = posterior.merge(
        baseline[["candidate_id", "__decision_ts__", "final_score"]],
        on="candidate_id", how="left", suffixes=("_bayes", "_baseline"), validate="one_to_one",
    )
    if overlap["__decision_ts___baseline"].isna().any():
        raise AssertionError("Bayesian predictions contain identities absent from baseline ledger")
    if not overlap["__decision_ts___bayes"].equals(overlap["__decision_ts___baseline"]):
        raise AssertionError("Bayesian prediction timestamps disagree with baseline ledger")
    max_score_error = float(np.max(np.abs(
        pd.to_numeric(overlap["final_score_bayes"], errors="raise")
        - pd.to_numeric(overlap["final_score_baseline"], errors="raise")
    )))
    if max_score_error > 1e-8:
        raise AssertionError(f"Bayesian and baseline final_score differ (max {max_score_error})")

    posterior = posterior.loc[posterior["bayes_available"].fillna(False).astype(bool)].copy()
    active, quality, base_percentile = _quality_percentile(posterior)
    posterior["timestamp_top30"] = active
    posterior["bayesian_quality_percentile"] = quality.astype(np.float32)
    base_score = pd.to_numeric(posterior["final_score"], errors="raise").to_numpy(float)
    if args.correction_mode == "percentile_additive":
        delta = np.where(active, quality - 0.5, 0.0)
    else:
        delta = np.where(active, quality - base_percentile, 0.0)
    posterior["corrected_score"] = base_score + float(args.alpha) * delta
    correction = posterior[["candidate_id", "corrected_score", "timestamp_top30", "bayesian_quality_percentile"]]
    output = baseline.merge(correction, on="candidate_id", how="left", validate="one_to_one")
    output["bayes_available"] = output["corrected_score"].notna()
    output["corrected_score"] = output["corrected_score"].fillna(
        pd.to_numeric(output["final_score"], errors="raise")
    )
    output["timestamp_top30"] = output["timestamp_top30"].eq(True)
    output["bayesian_quality_percentile"] = output["bayesian_quality_percentile"].fillna(0.5).astype(np.float32)
    expected = pd.to_numeric(output["final_score"], errors="raise").to_numpy(float)
    changed = np.abs(pd.to_numeric(output["corrected_score"], errors="raise").to_numpy(float) - expected) > 0.0
    if changed.sum() and not output.loc[changed, "bayes_available"].all():
        raise AssertionError("a pre-Bayesian row was unexpectedly corrected")
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "full_ledger_score_correction.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_bayesian_full_ledger_correction_v1",
        "baseline_ledger": str(args.baseline_ledger),
        "baseline_ledger_sha256": _sha(args.baseline_ledger),
        "bayesian_predictions": str(args.bayesian_predictions),
        "bayesian_predictions_sha256": _sha(args.bayesian_predictions),
        "alpha": float(args.alpha),
        "correction_mode": str(args.correction_mode),
        "rows": int(len(output)),
        "bayesian_rows": int(output["bayes_available"].sum()),
        "changed_rows": int(changed.sum()),
        "max_baseline_score_reconciliation_error": max_score_error,
        "causality": (
            "all pre-Bayesian rows retain final_score exactly; held Bayesian correction uses only held score "
            "and train-derived posterior ranks; policy outcomes are preserved only for later admission evaluation"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
