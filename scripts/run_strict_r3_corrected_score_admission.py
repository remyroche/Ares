#!/usr/bin/env python3
"""Causal 21-day admission replay for a precomputed strict-R3 score variant.

This is the executable counterpart to the timestamp-local score diagnostics.
Scores must already be strict-prequential predictions.  The runner joins only
evaluation labels and side identity, then applies the canonical 21-day
hierarchical side-local expected-net map to ``corrected_score``.  It never
uses outcome availability to choose candidates or construct their scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_causal_admission import (  # noqa: E402
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)


TAILS = (0.005, 0.01, 0.02, 0.05)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--surfaces", type=Path, nargs="+", required=True)
    parser.add_argument("--arms", required=True, help="Comma-separated score arms to replay.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evaluation-start", required=True, help="Metrics only; prior score rows remain available for mapping.")
    parser.add_argument("--evaluation-end", required=True, help="Exclusive metrics end.")
    return parser.parse_args()


def _read_scores(paths: list[Path], arms: tuple[str, ...]) -> pd.DataFrame:
    required = ["candidate_id", "__decision_ts__", "arm", "corrected_score"]
    pieces = [pd.read_parquet(path, columns=required) for path in paths]
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.loc[frame["arm"].astype(str).isin(arms)].copy()
    if frame.empty:
        raise ValueError("score inputs contain none of the requested arms")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    duplicates = frame.duplicated(["candidate_id", "arm"])
    if duplicates.any():
        raise ValueError("score inputs overlap on candidate_id × arm")
    return frame


def _read_labels(paths: list[Path]) -> pd.DataFrame:
    required = [
        "candidate_id", "__decision_ts__", "side_name", "policy_path_valid",
        "policy_label_available_ts", "policy_net_bps", "policy_gross_bps", "policy_exit_reason",
        "policy_market_data_quality", "policy_outcome_source",
    ]
    pieces = [pd.read_parquet(path, columns=required) for path in paths]
    frame = pd.concat(pieces, ignore_index=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("source surfaces overlap on candidate_id")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce")
    return frame


def _metrics(frame: pd.DataFrame, arm: str, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    work = frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)].copy()
    groups = [("all", work)] + list(work.groupby(work["__decision_ts__"].dt.strftime("%Y-%m"), sort=True))
    rows: list[dict[str, object]] = []
    for period, block in groups:
        mapped = block.loc[pd.to_numeric(block["causal_21d_side_expected_net_bps"], errors="coerce").notna()]
        admitted = mapped.loc[mapped["causal_21d_side_admitted_ge_50bps"].astype(bool)]
        valid = admitted.loc[
            admitted["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(admitted["policy_net_bps"], errors="coerce"))
        ]
        net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
        rows.append({
            "arm": arm, "period": str(period), "kind": "admission",
            "score_rows": len(block), "mapped_rows": len(mapped), "admitted_rows": len(admitted),
            "admission_rate": float(len(admitted) / max(1, len(block))),
            "valid_outcomes": len(valid), "outcome_coverage": float(len(valid) / max(1, len(admitted))),
            "net_bps_per_admitted_trade": float(net.mean()) if len(net) else np.nan,
            "positive_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
        })
        # Retrospective tail metrics remain a diagnostic; ranking is in the
        # common-bps expected-net coordinate produced by the causal map.
        for tail in TAILS:
            if admitted.empty:
                continue
            selected = admitted.nlargest(
                max(1, int(math.ceil(tail * len(admitted)))), "causal_21d_side_expected_net_bps", keep="first",
            )
            selected_valid = selected.loc[
                selected["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce"))
            ]
            selected_net = pd.to_numeric(selected_valid["policy_net_bps"], errors="coerce")
            rows.append({
                "arm": arm, "period": str(period), "kind": f"admitted_top_{tail:g}",
                "score_rows": len(block), "mapped_rows": len(mapped), "admitted_rows": len(admitted),
                "admission_rate": float(len(admitted) / max(1, len(block))),
                "valid_outcomes": len(selected_valid), "outcome_coverage": float(len(selected_valid) / max(1, len(selected))),
                "net_bps_per_admitted_trade": float(selected_net.mean()) if len(selected_net) else np.nan,
                "positive_rate": float(selected_net.gt(0.0).mean()) if len(selected_net) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not arms:
        raise ValueError("at least one score arm is required")
    start = pd.Timestamp(args.evaluation_start, tz="UTC")
    end = pd.Timestamp(args.evaluation_end, tz="UTC")
    scores = _read_scores(list(args.score_predictions), arms)
    labels = _read_labels(list(args.surfaces))
    score_ids = set(scores["candidate_id"])
    labels = labels.loc[labels["candidate_id"].isin(score_ids)].copy()
    if len(labels) != scores["candidate_id"].nunique():
        missing = scores.loc[~scores["candidate_id"].isin(labels["candidate_id"]), "candidate_id"].nunique()
        raise ValueError(f"label surfaces do not cover {missing} scored candidates")
    parts: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    map_audits: list[pd.DataFrame] = []
    for arm in arms:
        score = scores.loc[scores["arm"].eq(arm), ["candidate_id", "corrected_score"]]
        frame = labels.merge(score, on="candidate_id", how="inner", validate="one_to_one")
        frame = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        mapped, audit = apply_causal_21d_side_admission(
            frame, score_column="corrected_score", net_column="policy_net_bps",
            decision_column="__decision_ts__", label_available_column="policy_label_available_ts",
            identity_column="candidate_id", spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
        )
        mapped["arm"] = arm
        parts.append(mapped)
        metrics.append(_metrics(mapped, arm, start=start, end=end))
        map_audits.append(audit.assign(arm=arm))
    args.out_dir.mkdir(parents=True)
    pd.concat(parts, ignore_index=True).to_parquet(args.out_dir / "admission_predictions.parquet", index=False, compression="zstd")
    pd.concat(metrics, ignore_index=True).to_parquet(args.out_dir / "metrics.parquet", index=False)
    pd.concat(map_audits, ignore_index=True).to_parquet(args.out_dir / "admission_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_corrected_score_causal_admission_v1",
        "score_predictions": [str(path) for path in args.score_predictions],
        "surfaces": [str(path) for path in args.surfaces],
        "source_sha256": {str(path): _sha(path) for path in [*args.score_predictions, *args.surfaces]},
        "arms": list(arms), "evaluation": [str(start), str(end)],
        "admission": "Causal21dAdmissionSpec(hierarchical_tail_side_shrinkage_v2), corrected-score input, 21 calendar-day fully-resolved labels, mapped EV >= 50 bps, fail closed",
        "causality": "candidate scores are precomputed OOF; each admission-map reference row has policy_label_available_ts strictly before its decision; outcomes are never used by corrected_score",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": sum(len(part) for part in parts), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
