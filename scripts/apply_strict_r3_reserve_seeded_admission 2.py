#!/usr/bin/env python3
"""Apply causal current-vintage EV admission with same-model reserve support.

Each strict-R3 producer scores a 42-day reserve using that *same* fitted base
and conversion bundle.  The reserve is excluded from producer fitting.  This
script joins policy outcomes only after scoring and uses the reserve's labels
only once they are resolved before each candidate decision.  It repairs the
otherwise avoidable exact-vintage 21-day map cold start without mixing score
domains or looking into held outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import apply_current_admission


LINEAGE = [
    "ev_score_family_id", "conversion_bundle_sha256",
    "upstream_bundle_sha256", "geometry_bundle_sha256",
]
OUTCOME = [
    "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_label_available_ts", "policy_outcome_source",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _with_outcomes(
    scores: pd.DataFrame, outcomes: pd.DataFrame, *, score_identity_unique: bool,
) -> pd.DataFrame:
    overlap = set(scores.columns).intersection(OUTCOME).difference({"candidate_id"})
    if overlap:
        scores = scores.drop(columns=sorted(overlap))
    # A reserve candidate can legitimately be rescored by two overlapping,
    # separately fitted producers.  Its policy outcome remains one immutable
    # row.  Held scores, by contrast, must be unique at the artifact boundary.
    joined = scores.merge(
        outcomes, on="candidate_id", how="left",
        validate="one_to_one" if score_identity_unique else "many_to_one",
    )
    if joined["policy_path_valid"].isna().any():
        raise AssertionError("same-model reserve contains identities absent from policy-outcome ledger")
    joined["__decision_ts__"] = pd.to_datetime(joined["__decision_ts__"], utc=True, errors="raise")
    joined["policy_label_available_ts"] = pd.to_datetime(
        joined["policy_label_available_ts"], utc=True, errors="raise",
    )
    joined["stack_is_prequential"] = True
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-ledger", type=Path, required=True)
    parser.add_argument("--same-model-reserve-scores", type=Path, required=True)
    parser.add_argument("--outcome-ledger", type=Path, required=True)
    parser.add_argument("--score-column", default="final_score")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")

    held = pd.read_parquet(args.held_ledger)
    reserve = pd.read_parquet(args.same_model_reserve_scores)
    outcomes = pd.read_parquet(args.outcome_ledger, columns=OUTCOME)
    if outcomes.candidate_id.duplicated().any():
        raise AssertionError("policy outcome ledger duplicates candidate identities")
    if args.score_column not in held or args.score_column not in reserve:
        raise KeyError(f"missing requested score column {args.score_column!r}")
    for frame, name in ((held, "held"), (reserve, "reserve")):
        missing = sorted(set(["candidate_id", "__decision_ts__", "calibration_activation_ts", *LINEAGE]).difference(frame.columns))
        if missing:
            raise KeyError(f"{name} score ledger lacks {missing}")
        if frame.candidate_id.duplicated().any() and name == "held":
            raise AssertionError("held ledger duplicates candidate identities")
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame["calibration_activation_ts"] = pd.to_datetime(frame["calibration_activation_ts"], utc=True, errors="raise")
        for field in LINEAGE:
            if frame[field].isna().any():
                raise AssertionError(f"{name} score ledger has null {field}")
    # A reserve row may occur under several separately fitted producers.  It
    # is intentionally unique only after restricting to one exact lineage.
    reserve = _with_outcomes(reserve, outcomes, score_identity_unique=False)
    held = _with_outcomes(held, outcomes, score_identity_unique=True)
    if args.score_column != "final_score":
        reserve["final_score"] = reserve[args.score_column]
        held["final_score"] = held[args.score_column]

    parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    producer_audit: list[dict[str, object]] = []
    for lineage, positions in held.groupby(LINEAGE, sort=False).groups.items():
        held_part = held.loc[positions].copy()
        activation = held_part["calibration_activation_ts"].drop_duplicates()
        if len(activation) != 1:
            raise AssertionError("one exact producer lineage has multiple activation cutoffs")
        cutoff = activation.iloc[0]
        mask = pd.Series(True, index=reserve.index)
        for field, value in zip(LINEAGE, lineage, strict=True):
            mask &= reserve[field].astype(str).eq(str(value))
        reserve_part = reserve.loc[mask].copy()
        if reserve_part.empty:
            raise AssertionError("held producer has no same-model reserve scores")
        if not reserve_part["calibration_activation_ts"].eq(cutoff).all():
            raise AssertionError("reserve producer activation differs from held producer")
        if not reserve_part["__decision_ts__"].lt(cutoff).all():
            raise AssertionError("same-model reserve is not strictly prior to held activation")
        if not held_part["__decision_ts__"].ge(cutoff).all():
            raise AssertionError("held rows precede their producer activation")
        combined = pd.concat([reserve_part, held_part], ignore_index=True, sort=False)
        # Duplicates across distinct producers are legitimate; duplicates
        # inside this one exact score domain are not.
        if combined.candidate_id.duplicated().any():
            raise AssertionError("same producer reserve overlaps held identities")
        mapped, audit = apply_current_admission(combined)
        mapped["ev_mapping_vintage_mode"] = "same_model_42d_reserve_seeded_v1"
        mapped["ev_mapping_reserve_rows"] = int(len(reserve_part))
        mapped["ev_mapping_reserve_activation_ts"] = cutoff
        result = mapped.loc[mapped.candidate_id.isin(set(held_part.candidate_id))].copy()
        if len(result) != len(held_part) or result.candidate_id.duplicated().any():
            raise AssertionError("reserve-seeded map changed held candidate identities")
        parts.append(result)
        audit_parts.append(audit.assign(
            ev_mapping_vintage_mode="same_model_42d_reserve_seeded_v1",
            ev_mapping_reserve_rows=len(reserve_part),
            ev_mapping_reserve_activation_ts=cutoff,
            **{field: str(value) for field, value in zip(LINEAGE, lineage, strict=True)},
        ))
        producer_audit.append({
            "calibration_activation_ts": cutoff,
            "held_rows": int(len(held_part)),
            "reserve_rows": int(len(reserve_part)),
            "reserve_labels_resolved_at_activation": int(
                reserve_part["policy_label_available_ts"].lt(cutoff).sum()
            ),
            **{field: str(value) for field, value in zip(LINEAGE, lineage, strict=True)},
        })
    output = pd.concat(parts, ignore_index=True)
    output = output.set_index("candidate_id").loc[held.candidate_id].reset_index()
    if len(output) != len(held) or output.candidate_id.duplicated().any():
        raise AssertionError("reserve-seeded output does not match held identity contract")
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "reserve_seeded_causal_admission_ledger.parquet", index=False, compression="zstd")
    pd.concat(audit_parts, ignore_index=True).to_parquet(args.out_dir / "causal_admission_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(producer_audit).to_parquet(args.out_dir / "reserve_seed_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_same_model_reserve_seeded_admission_v1",
        "held_ledger": str(args.held_ledger), "held_ledger_sha256": _sha(args.held_ledger),
        "same_model_reserve_scores": str(args.same_model_reserve_scores),
        "reserve_scores_sha256": _sha(args.same_model_reserve_scores),
        "outcome_ledger": str(args.outcome_ledger), "outcome_ledger_sha256": _sha(args.outcome_ledger),
        "score_column": str(args.score_column), "rows": int(len(output)),
        "producers": int(len(producer_audit)),
        "contract": (
            "Each held producer is mapped only with its own prior 42-day same-model reserve and chronologically "
            "accumulating held rows; policy labels enter only when label_available_ts < decision_ts; 50-bps "
            "hierarchical-tail-side-shrinkage admission; no cross-producer raw-score pooling"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "rows": len(output), "producers": len(producer_audit)}))


if __name__ == "__main__":
    main()
