#!/usr/bin/env python3
"""Compare causal EV admission floors without changing score maps.

The supplied producer-specific map is frozen before this experiment.  The only
ablated decision is the common-bps admission floor, so 10/30/50-bps arms are
directly comparable and cannot gain information by refitting a map.  The
default is the exact-reserve bridge; a caller may explicitly select another
strictly causal map family (for example the same-model reserve-seeded map).
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

from extreme_price_movements.strict_r3_ev_bridge import (
    EXACT_PRODUCER_RESERVE_CALIBRATION_MODE,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(
    frame: pd.DataFrame,
    *,
    selected: pd.Series,
    arm: str,
    frequency: str,
    period: str,
) -> dict[str, object]:
    policy_valid = frame["policy_path_valid"].fillna(False).astype(bool)
    rows = frame.loc[selected & policy_valid].copy()
    net = pd.to_numeric(rows["policy_net_bps"], errors="coerce")
    gross = pd.to_numeric(rows["policy_gross_bps"], errors="coerce")
    expected = pd.to_numeric(rows["causal_21d_side_expected_net_bps"], errors="coerce")
    return {
        "arm": arm,
        "frequency": frequency,
        "period": period,
        "scored_rows": int(len(frame)),
        "selected_rows": int(selected.sum()),
        "valid_selected_rows": int(len(rows)),
        "admission_rate": float(selected.mean()) if len(frame) else np.nan,
        "expected_net_bps_per_trade": float(expected.mean()) if len(rows) else np.nan,
        "gross_bps_per_trade": float(gross.mean()) if len(rows) else np.nan,
        "net_bps_per_trade": float(net.mean()) if len(rows) else np.nan,
        "positive_net_rate": float(net.gt(0.0).mean()) if len(rows) else np.nan,
    }


def _summaries(frame: pd.DataFrame, *, selected: pd.Series, arm: str) -> list[dict[str, object]]:
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    output = [_metrics(frame, selected=selected, arm=arm, frequency="all", period="all")]
    for frequency in ("M", "W-MON"):
        periods = decision.dt.tz_localize(None).dt.to_period(frequency).astype(str)
        for period, positions in periods.groupby(periods, observed=True, sort=True).groups.items():
            index = np.asarray(list(positions), dtype=np.int64)
            output.append(_metrics(
                frame.iloc[index], selected=selected.iloc[index], arm=arm,
                frequency=frequency, period=str(period),
            ))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--frozen-base-availability", type=Path,
        help=(
            "Optional decision-time frozen-base availability sidecar. It is "
            "required when predictions predate the forward fail-closed gate."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold-bps", nargs="+", type=float, default=[10.0, 30.0, 50.0])
    parser.add_argument(
        "--required-vintage-mode",
        default=EXACT_PRODUCER_RESERVE_CALIBRATION_MODE,
        help="The one immutable causal EV-map mode accepted by this replay.",
    )
    parser.add_argument(
        "--arm-suffix", default="exact_reserve",
        help="Stable suffix used in selection-column names and portfolio-arm IDs.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable threshold ablation exists: {args.out_dir}")
    thresholds = sorted(set(float(value) for value in args.threshold_bps))
    if not thresholds or any(value < 0.0 for value in thresholds):
        raise ValueError("thresholds must be non-negative bps values")
    frame = pd.read_parquet(args.predictions)
    if args.frozen_base_availability is not None:
        availability = pd.read_parquet(args.frozen_base_availability)
        required_availability = {
            "candidate_id", "frozen_base_feature_count",
            "frozen_base_feature_fraction", "frozen_base_contract_complete",
        }
        missing = sorted(required_availability.difference(availability.columns))
        if missing:
            raise ValueError(f"frozen base availability lacks: {missing}")
        if availability["candidate_id"].duplicated().any():
            raise ValueError("frozen base availability has duplicate identities")
        if "frozen_base_contract_complete" in frame:
            frame = frame.drop(columns=[
                column for column in required_availability
                if column != "candidate_id" and column in frame
            ])
        frame = frame.merge(
            availability.loc[:, sorted(required_availability)],
            on="candidate_id", how="left", validate="one_to_one",
        )
    if "frozen_base_contract_complete" in frame:
        if frame["frozen_base_contract_complete"].isna().any():
            raise ValueError("frozen base availability does not cover every prediction identity")
        feature_complete = frame["frozen_base_contract_complete"].fillna(False).astype(bool)
    else:
        feature_complete = pd.Series(True, index=frame.index)
    required = {
        "candidate_id", "__decision_ts__",
        "causal_21d_side_expected_net_bps", "policy_path_valid", "policy_net_bps",
        "policy_gross_bps", "ev_mapping_vintage_mode",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"threshold ablation predictions lack: {missing}")
    # The lockstep ledger is deliberately decomposed into exact conversion and
    # upstream hashes.  Older calibrated ledgers additionally persist a
    # convenience producer ID.  Reconstruct that purely from lineage when it
    # is absent; it is provenance only and never an admission input.
    if "producer_bundle_id" not in frame:
        lineage = ["conversion_bundle_sha256", "upstream_bundle_sha256", "geometry_bundle_sha256"]
        lineage_missing = sorted(set(lineage).difference(frame.columns))
        if lineage_missing:
            raise ValueError(
                "predictions lack producer_bundle_id and cannot reconstruct it from "
                f"lineage: {lineage_missing}"
            )
        if frame.loc[:, lineage].isna().any().any():
            raise ValueError("cannot reconstruct producer provenance with null lineage")
        frame["producer_bundle_id"] = frame.loc[:, lineage].astype(str).agg("|".join, axis=1)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("threshold ablation requires unique candidate IDs")
    modes = set(frame["ev_mapping_vintage_mode"].dropna().astype(str))
    if modes != {str(args.required_vintage_mode)}:
        raise ValueError(
            "threshold ablation requires one declared causal map mode, "
            f"expected {args.required_vintage_mode!r}, observed {sorted(modes)}"
        )
    if not args.arm_suffix.replace("_", "").isalnum():
        raise ValueError("arm suffix must be alphanumeric with optional underscores")
    expected = pd.to_numeric(frame["causal_21d_side_expected_net_bps"], errors="coerce")
    selection = frame.loc[:, ["candidate_id", "__decision_ts__", "producer_bundle_id"]].copy()
    results: list[dict[str, object]] = []
    for threshold in thresholds:
        arm = f"E{int(round(threshold)):03d}_{args.arm_suffix}"
        selected = expected.ge(threshold).fillna(False) & feature_complete
        if (
            np.isclose(threshold, 50.0)
            and str(args.required_vintage_mode) == EXACT_PRODUCER_RESERVE_CALIBRATION_MODE
            and "causal_21d_side_admitted_ge_50bps" in frame
            and "frozen_base_contract_complete" in frame
            and not np.array_equal(
                selected.to_numpy(bool),
                frame["causal_21d_side_admitted_ge_50bps"].fillna(False).to_numpy(bool),
            )
        ):
            raise AssertionError(
                "E050 exact-reserve selection must equal the native EV map "
                "after the same frozen-base feature gate",
            )
        selection[f"{arm}__admitted"] = selected.to_numpy(bool)
        selection[f"{arm}__expected_net_bps"] = expected.to_numpy(float)
        results.extend(_summaries(frame, selected=selected, arm=arm))
    args.out_dir.mkdir(parents=True)
    selection.to_parquet(args.out_dir / "threshold_selection.parquet", index=False, compression="zstd")
    pd.DataFrame(results).to_parquet(args.out_dir / "threshold_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_causal_ev_threshold_ablation_v2",
        "predictions": str(args.predictions),
        "predictions_sha256": _sha(args.predictions),
        "threshold_bps": thresholds,
        "required_vintage_mode": str(args.required_vintage_mode),
        "arm_suffix": str(args.arm_suffix),
        "frozen_base_availability": (
            str(args.frozen_base_availability)
            if args.frozen_base_availability is not None else "embedded_or_not_supplied"
        ),
        "contract": (
            "same immutable causal maps and scores in every arm; only the "
            "causal common-bps EV admission floor changes"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
