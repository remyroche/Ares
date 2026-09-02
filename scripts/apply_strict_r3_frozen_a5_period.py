#!/usr/bin/env python3
"""Apply one frozen exact-reserve EV/R5/A5 bundle over a forward period.

The input predictions are already outcome-free and immutable.  Exact policy
outcomes are joined only to create a resolved evaluation ledger.  Cell-day
admission is recomputed one UTC day at a time and can consume only labels
available before that day.  The frozen R5 and bounded-A5 bundles never receive
held outcomes; they consume only score/reliability fields and the causal EV
estimate produced for that day.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_a5_trust import (  # noqa: E402
    apply_a5_bounded_10pct,
    load_a5_bundle,
)
from extreme_price_movements.strict_r3_cell_day_admission import (  # noqa: E402
    CELL_DAY_TRIM_15_CALIBRATION_MODE,
    apply_cell_day_trim15_admission_snapshot,
)
from extreme_price_movements.strict_r3_cell_day_trust import (  # noqa: E402
    load_cell_day_residual_trust_bundle,
)
from extreme_price_movements.strict_r3_ev_bridge import (  # noqa: E402
    load_strict_r3_ev_bridge,
)


POLICY_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_label_available_ts", "policy_cost_bps",
)


def _target_free_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove evaluation-only policy fields before any scoring/map call."""
    forbidden = {
        *POLICY_COLUMNS[1:],
        "policy_label_available_ts",
    }
    return frame.drop(
        columns=[column for column in forbidden if column in frame.columns],
        errors="ignore",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(frame: pd.DataFrame, name: str) -> None:
    if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
        raise ValueError(f"{name} requires unique non-null candidate IDs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--policy-outcomes", type=Path, required=True)
    parser.add_argument("--resolved-score-label-ledger", type=Path, required=True)
    parser.add_argument("--immediate-calibration-index", type=Path, required=True)
    parser.add_argument("--r5-bundle-dir", type=Path, required=True)
    parser.add_argument("--a5-bundle-dir", type=Path, required=True)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable frozen-A5 period output exists: {args.out_dir}")

    predictions = pd.read_parquet(args.predictions)
    outcomes = pd.read_parquet(args.policy_outcomes, columns=list(POLICY_COLUMNS))
    resolved = pd.read_parquet(args.resolved_score_label_ledger)
    for frame, name in (
        (predictions, "predictions"), (outcomes, "policy outcomes"),
        (resolved, "resolved score-label ledger"),
    ):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        _unique(frame, name)
    for frame in (predictions, resolved):
        frame["__decision_ts__"] = pd.to_datetime(
            frame["__decision_ts__"], utc=True, errors="raise",
        )
    outcomes["policy_label_available_ts"] = pd.to_datetime(
        outcomes["policy_label_available_ts"], utc=True, errors="coerce",
    )
    start = pd.Timestamp(args.evaluation_start)
    end = pd.Timestamp(args.evaluation_end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    held = predictions.merge(outcomes, on="candidate_id", how="left", validate="one_to_one")
    held = held.loc[
        held["__decision_ts__"].ge(start) & held["__decision_ts__"].lt(end)
    ].copy()
    if held.empty:
        raise ValueError("frozen-A5 forward evaluation window is empty")
    if held["policy_path_valid"].isna().any():
        raise ValueError("policy outcomes do not cover every forward score identity")
    valid = held["policy_path_valid"].fillna(False).astype(bool)
    finite = np.isfinite(pd.to_numeric(held["policy_net_bps"], errors="coerce"))
    if (valid & ~finite).any():
        raise ValueError("a valid forward policy path has non-finite policy net")
    with_gross = valid & np.isfinite(pd.to_numeric(held["policy_gross_bps"], errors="coerce"))
    if with_gross.any() and not np.allclose(
        held.loc[with_gross, "policy_net_bps"].to_numpy(float),
        held.loc[with_gross, "policy_gross_bps"].to_numpy(float) - 100.0,
        atol=1e-9, rtol=0.0,
    ):
        raise ValueError("forward policy outcomes do not apply 100-bps cost exactly once")

    index = pd.read_parquet(args.immediate_calibration_index)
    fitted = index.loc[index["status"].eq("fitted_immediate_exact_producer_calibration")]
    if len(fitted) != 1:
        raise ValueError("frozen period requires exactly one fitted immediate calibrator")
    bridge_path = Path(str(fitted.iloc[0]["ev_bridge_bundle"]))
    if not bridge_path.is_absolute():
        bridge_path = ROOT / bridge_path
    bridge = load_strict_r3_ev_bridge(bridge_path)

    # Append forward outcomes to the dynamic reference ledger.  The mapper
    # itself filters them by label_available_ts < current UTC day, so a row can
    # never influence its own day or any earlier decision.
    resolved_columns = list(resolved.columns)
    forward_resolved = held.reindex(columns=resolved_columns)
    history = pd.concat([resolved, forward_resolved], ignore_index=True)
    history = history.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    history = history.drop_duplicates("candidate_id", keep="last")
    mapped_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    for _day, current in held.groupby(held["__decision_ts__"].dt.normalize(), sort=True):
        current_scores = _target_free_snapshot(current)
        mapped, audit = apply_cell_day_trim15_admission_snapshot(
            resolved_score_ledger=history,
            current_scores=current_scores,
            bundle=bridge,
        )
        # Outcome join is evaluation-only and occurs after the map has returned
        # its candidate-identical target-free output.
        current_outcomes = current.loc[:, list(POLICY_COLUMNS)].copy()
        mapped = mapped.merge(
            current_outcomes, on="candidate_id", how="left", validate="one_to_one",
        )
        mapped_parts.append(mapped)
        audit_parts.append(audit)
    mapped = pd.concat(mapped_parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    audit = pd.concat(audit_parts, ignore_index=True)
    if not audit["strictly_prior_resolved"].fillna(False).astype(bool).all():
        raise AssertionError("forward Cell-day map consumed a non-prior outcome")

    r5 = load_cell_day_residual_trust_bundle(args.r5_bundle_dir)
    r5_input = mapped.loc[:, ["candidate_id", *r5.fields]].copy()
    r5_input["raw_expected_bps"] = pd.to_numeric(
        mapped["causal_21d_side_expected_net_bps"], errors="coerce",
    ).to_numpy(float)
    r5_score = r5.score(r5_input)
    output = mapped.merge(r5_score, on="candidate_id", how="inner", validate="one_to_one")
    posterior = pd.to_numeric(output["trust_posterior_expected_bps"], errors="coerce")
    output["trust_posterior_available"] = np.isfinite(posterior)
    output["trust_posterior_admitted_ge_50bps"] = (
        output["trust_posterior_available"] & posterior.ge(50.0)
    )
    output["r5_fit_status"] = "frozen_bundle"

    a4, calibration = load_a5_bundle(args.a5_bundle_dir)
    a4_input = output.loc[:, ["candidate_id", *a4.fields]].copy()
    a4_input["raw_expected_bps"] = pd.to_numeric(
        output["causal_21d_side_expected_net_bps"], errors="coerce",
    ).to_numpy(float)
    a4_score = a4.score(a4_input)
    output = output.merge(a4_score, on="candidate_id", how="inner", validate="one_to_one")
    a5_score = apply_a5_bounded_10pct(output, calibration=calibration)
    output = output.merge(a5_score, on="candidate_id", how="inner", validate="one_to_one")
    if len(output) != len(held) or set(output["candidate_id"]) != set(held["candidate_id"]):
        raise AssertionError("frozen EV/R5/A5 application changed forward identities")

    args.out_dir.mkdir(parents=True)
    output_path = args.out_dir / "frozen_a5_scored_label_ledger.parquet"
    score_ledger_path = args.out_dir / "scored_label_ledger.parquet"
    provenance_path = args.out_dir / "score_and_cell_day_admission_provenance.parquet"
    a5_path = args.out_dir / "a5_bounded10_forward_predictions.parquet"
    audit_path = args.out_dir / "cell_day_admission_audit.parquet"
    output.to_parquet(output_path, index=False, compression="zstd")
    held.sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).to_parquet(score_ledger_path, index=False, compression="zstd")
    provenance_columns = [
        column for column in output.columns
        if column in {
            "candidate_id", "__decision_ts__", "conversion_bundle_sha256",
            "geometry_bundle_sha256", "upstream_bundle_sha256",
            "ev_score_family_id", "stack_is_prequential",
        }
        or column.startswith("causal_") or column.startswith("cell_day_")
        or column.startswith("ev_mapping_") or column == "ev_bridge_bundle_identity"
    ]
    output.loc[:, provenance_columns].to_parquet(
        provenance_path, index=False, compression="zstd",
    )
    a5_columns = [
        "candidate_id", "trust_posterior_expected_bps",
        "a5_bounded10_expected_bps", "a5_timestamp_top15",
        "a5_bounded10_available", "a5_bounded10_admitted",
    ]
    output.loc[:, a5_columns].to_parquet(a5_path, index=False, compression="zstd")
    audit.to_parquet(audit_path, index=False)
    manifest = {
        "schema": "strict_r3_frozen_exact_reserve_a5_forward_period_v1",
        "scope": "long_only_forward_evaluation",
        "evaluation_start": start.isoformat(),
        "evaluation_end_exclusive": end.isoformat(),
        "rows": int(len(output)),
        "valid_policy_rows": int(valid.sum()),
        "days": int(output["__decision_ts__"].dt.normalize().nunique()),
        "cell_day_mapping": CELL_DAY_TRIM_15_CALIBRATION_MODE,
        "mapping": CELL_DAY_TRIM_15_CALIBRATION_MODE,
        "cell_day_strictly_prior_resolved": True,
        "raw_cell_day_admitted_rows": int(
            output["causal_21d_side_admitted_ge_50bps"].fillna(False).sum()
        ),
        "a0_r5_posterior_admitted_rows": int(
            output["trust_posterior_admitted_ge_50bps"].fillna(False).sum()
        ),
        "r5_available_rows": int(output["trust_posterior_available"].sum()),
        "a5_available_rows": int(output["a5_bounded10_available"].sum()),
        "a5_admitted_rows": int(output["a5_bounded10_admitted"].sum()),
        "a5_formula": "A0 + 0.10 * (calibrated_A4 - A0)",
        "a5_admission": "A0>=50 AND timestamp-local top15 by pre-trust final_score",
        "held_outcomes_used_for_scoring_or_a5": False,
        "held_outcomes_joined_after_predictions": True,
        "held_outcomes_physically_absent_from_ev_map_input": True,
        "cost_bps_once": 100.0,
        "sources": {
            "predictions": {"path": str(args.predictions), "sha256": _sha(args.predictions)},
            "policy_outcomes": {"path": str(args.policy_outcomes), "sha256": _sha(args.policy_outcomes)},
            "resolved_score_label_ledger": {
                "path": str(args.resolved_score_label_ledger),
                "sha256": _sha(args.resolved_score_label_ledger),
            },
            "immediate_calibration_index": {
                "path": str(args.immediate_calibration_index),
                "sha256": _sha(args.immediate_calibration_index),
            },
            "ev_bridge_bundle": {
                "path": str(bridge_path),
                "manifest_sha256": _sha(bridge_path / "run_manifest.json"),
            },
            "r5_bundle_manifest": {
                "path": str(args.r5_bundle_dir / "run_manifest.json"),
                "sha256": _sha(args.r5_bundle_dir / "run_manifest.json"),
            },
            "a5_bundle_manifest": {
                "path": str(args.a5_bundle_dir / "run_manifest.json"),
                "sha256": _sha(args.a5_bundle_dir / "run_manifest.json"),
            },
        },
        "output": str(output_path),
        "output_sha256": _sha(output_path),
        "portfolio_replay_inputs": {
            "scored_label_ledger": {
                "path": str(score_ledger_path), "sha256": _sha(score_ledger_path),
            },
            "admission_provenance": {
                "path": str(provenance_path), "sha256": _sha(provenance_path),
            },
            "a5_predictions": {"path": str(a5_path), "sha256": _sha(a5_path)},
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
