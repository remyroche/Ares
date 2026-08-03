#!/usr/bin/env python3
"""Research-only causal scoring of a retrospective execution-EV population.

This is deliberately separate from :mod:`score_execution_ev_forward_population`.
It reuses the immutable final direct/capture heads, interaction, and frozen
calibrator seed for a historical decision block, but never treats the replay as
forward readiness or OOS performance evidence.  For each decision ``t`` the
21-day side-local isotonic map is refit solely from seed history (and, only when
explicitly supplied, validated resolved updates) whose label ended strictly
before ``t``.

The operational forward scorer retains its stricter post-cutoff requirement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_execution_ev_forward_calibrator_seed import (  # noqa: E402
    DECISION,
    IDENTITY,
    RESOLUTION,
    SIDES,
    TARGET,
)
from scripts.score_execution_ev_forward_population import (  # noqa: E402
    AVAILABILITY_COLUMNS,
    DEFAULT_HEAD_ROOT,
    DEFAULT_STATE,
    _score_raw_heads,
    _sha256,
    _write_json,
    apply_global_admission,
    causal_recent_isotonic_mapping,
    validate_resolved_updates,
)


SCHEMA = "execution_ev_retrospective_scored_population_v1"
DEFAULT_PREENTRY = Path(
    "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/"
    "preentry/preentry.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/"
    "final_head_scored"
)
LOOKBACK_DAYS = 21
MINIMUM_SIDE_ROWS = 100
FORBIDDEN_PREENTRY_COLUMNS = (TARGET, RESOLUTION)


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _identity_hash(frame: pd.DataFrame) -> str:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"candidate identity columns missing: {missing}")
    if frame.duplicated(list(IDENTITY)).any() or frame["candidate_id"].duplicated().any():
        raise ValueError("candidate identities must be globally unique")
    ordered = frame.loc[:, list(IDENTITY)].astype(str).sort_values(
        list(IDENTITY), kind="stable"
    )
    payload = "\n".join(
        "\x1f".join(row) for row in ordered.itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_history(history: pd.DataFrame) -> pd.DataFrame:
    """Validate immutable/resolved calibration rows before any causal slice."""

    required = {
        *IDENTITY,
        DECISION,
        RESOLUTION,
        TARGET,
        "frozen_margin_capture_interaction_raw",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"calibrator history columns missing: {missing}")
    work = history.copy()
    if work.duplicated(list(IDENTITY)).any() or work["candidate_id"].duplicated().any():
        raise ValueError("calibrator history contains duplicate identities")
    if not set(work["side_name"].astype(str)).issubset(SIDES):
        raise ValueError("calibrator history has an unsupported side")
    for column in (DECISION, RESOLUTION):
        if not isinstance(work[column].dtype, pd.DatetimeTZDtype):
            raise ValueError(f"calibrator history {column} must be timezone-aware")
        work[column] = work[column].dt.tz_convert("UTC")
    if (work[RESOLUTION] <= work[DECISION]).any():
        raise ValueError("calibrator history label end must follow the decision")
    values = work[[TARGET, "frozen_margin_capture_interaction_raw"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("calibrator history economics and scores must be finite")
    return work


def _validate_retro_preentry(
    frame: pd.DataFrame,
    *,
    first_decision_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    forbidden = sorted(set(FORBIDDEN_PREENTRY_COLUMNS).intersection(frame.columns))
    if forbidden:
        raise ValueError(
            "retrospective preentry must not contain resolved outcome columns: "
            f"{forbidden}"
        )
    _identity_hash(frame)
    if not set(frame["side_name"].astype(str)).issubset(SIDES):
        raise ValueError("retrospective preentry has an unsupported side")
    if set(frame["side_name"].astype(str)) != set(SIDES):
        raise ValueError("retrospective preentry must contain both sides")
    if not isinstance(frame[DECISION].dtype, pd.DatetimeTZDtype):
        raise ValueError("retrospective decision timestamps must be timezone-aware")
    work = frame.copy()
    work[DECISION] = work[DECISION].dt.tz_convert("UTC")
    if work[DECISION].empty:
        raise ValueError("retrospective preentry is empty")
    # This is the only retrospective exception: decisions must precede the
    # operational forward block, never replace or relax its cutoff.
    if (work[DECISION] >= first_decision_exclusive).any():
        raise ValueError(
            "retrospective decisions must be strictly before the operational "
            "first_decision_exclusive cutoff"
        )
    for column in AVAILABILITY_COLUMNS:
        if column not in work:
            raise ValueError(f"retrospective availability column missing: {column}")
        if not isinstance(work[column].dtype, pd.DatetimeTZDtype):
            raise ValueError(f"{column} must be stored timezone-aware")
        work[column] = work[column].dt.tz_convert("UTC")
        if (work[column] > work[DECISION]).any():
            raise ValueError(f"{column} occurs after the decision")
    return work


def causal_retrospective_mapping(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    *,
    lookback_days: int = LOOKBACK_DAYS,
    minimum_side_rows: int = MINIMUM_SIDE_ROWS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Map history-only scores and persist audit-ready support for every t/side.

    ``causal_recent_isotonic_mapping`` performs the actual fitting.  This
    wrapper deliberately reconstructs the exact training slice for each report
    so the persisted support proves the strict exclusion of labels resolving at
    or after the decision timestamp.
    """

    if int(lookback_days) != LOOKBACK_DAYS:
        raise ValueError(f"retrospective mapping lookback must remain {LOOKBACK_DAYS} days")
    if int(minimum_side_rows) < 1:
        raise ValueError("retrospective minimum_side_rows must be positive")
    checked_history = _validate_history(history)
    mapped, reports = causal_recent_isotonic_mapping(
        candidates,
        checked_history,
        lookback_days=LOOKBACK_DAYS,
        minimum_side_rows=int(minimum_side_rows),
    )
    resolution = pd.to_datetime(checked_history[RESOLUTION], utc=True, errors="raise")
    decision = pd.to_datetime(candidates[DECISION], utc=True, errors="raise")
    support: list[dict[str, Any]] = []
    for report in reports:
        timestamp = pd.Timestamp(report["decision_utc"])
        side = str(report["side"])
        lower = timestamp - pd.Timedelta(days=LOOKBACK_DAYS)
        rows = checked_history.loc[
            resolution.lt(timestamp)
            & resolution.ge(lower)
            & checked_history["side_name"].astype(str).eq(side)
        ]
        if rows.empty or len(rows) != int(report["history_rows"]):
            raise AssertionError("mapping support does not match fitted causal history")
        if not bool((rows[RESOLUTION] < timestamp).all()):
            raise AssertionError("retrospective mapping used an unresolved/future label")
        if not bool((rows[RESOLUTION] >= lower).all()):
            raise AssertionError("retrospective mapping exceeded its fixed lookback")
        support.append(
            {
                "execution_decision_utc": timestamp,
                "side_name": side,
                "candidate_rows_at_timestamp": int(
                    (decision.eq(timestamp) & candidates["side_name"].astype(str).eq(side)).sum()
                ),
                "lookback_days": LOOKBACK_DAYS,
                "history_rows": int(len(rows)),
                "history_resolution_min_utc": rows[RESOLUTION].min(),
                "history_resolution_max_utc": rows[RESOLUTION].max(),
                "history_resolved_strictly_before_decision": True,
            }
        )
    output = pd.DataFrame(support).sort_values(
        ["execution_decision_utc", "side_name"], kind="stable"
    ).reset_index(drop=True)
    return mapped, output


def _head_model_lineage(head_manifest: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for side in SIDES:
        for role, record_name in (
            ("direct", "direct_exact_net_residual"),
            ("capture", "capture_probability"),
        ):
            record = head_manifest["sides"][side]["models"][record_name]
            path = _resolve(record["path"])
            expected = str(record["sha256"])
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"{side} {role} head hash mismatch")
            records[f"{side}_{role}"] = {"path": path, "sha256": expected}
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Score the historical preentry population without changing forward policy."""

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    state = json.loads(args.calibrator_state.read_text(encoding="utf-8"))
    if state.get("schema") != "execution_ev_forward_calibrator_seed_v1":
        raise ValueError("unexpected calibrator-state schema")
    if int(state.get("lookback_days", -1)) != LOOKBACK_DAYS:
        raise ValueError(f"frozen state must bind a {LOOKBACK_DAYS}-day mapping lookback")
    if state.get("sequential_updates_only_after_resolution") is not True:
        raise ValueError("frozen state does not enforce resolved-only calibration updates")
    cutoff = pd.Timestamp(state["first_decision_exclusive_utc"])
    if cutoff.tzinfo is None:
        raise ValueError("frozen cutoff must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    history_path = _resolve(state["history"]["path"])
    if _sha256(history_path) != state["history"]["sha256"]:
        raise ValueError("calibrator seed-history hash mismatch")
    head_manifest_path = args.head_root / "manifest.json"
    feature_contract_path = args.head_root / "feature_contract.json"
    head_manifest = json.loads(head_manifest_path.read_text(encoding="utf-8"))
    if head_manifest.get("schema") != "execution_ev_forward_final_heads_v1":
        raise ValueError("unexpected final-head manifest schema")
    feature_contract = json.loads(feature_contract_path.read_text(encoding="utf-8"))
    if _sha256(feature_contract_path) != head_manifest["feature_contract"]["sha256"]:
        raise ValueError("final-head feature contract hash mismatch")
    head_models = _head_model_lineage(head_manifest)

    frame = _validate_retro_preentry(
        pd.read_parquet(args.preentry), first_decision_exclusive=cutoff
    )
    scored = _score_raw_heads(
        frame,
        head_root=args.head_root,
        head_manifest=head_manifest,
        feature_contract=feature_contract,
        state=state,
    )
    seed_history = _validate_history(pd.read_parquet(history_path))
    if args.resolved_updates is not None:
        updates = validate_resolved_updates(
            pd.read_parquet(args.resolved_updates), scored, seed_history
        )
        updates = _validate_history(updates)
        history = pd.concat([seed_history, updates], ignore_index=True)
        if history.duplicated(list(IDENTITY)).any() or history["candidate_id"].duplicated().any():
            raise ValueError("resolved updates duplicate seed history identity")
    else:
        history = seed_history
    scored["mapped_execution_ev"], support = causal_retrospective_mapping(
        scored,
        history,
        lookback_days=LOOKBACK_DAYS,
        minimum_side_rows=int(args.minimum_side_rows),
    )
    scored = apply_global_admission(scored)
    scored["direct_ev_available_at"] = scored[DECISION]
    scored["capture_probability_available_at"] = scored[DECISION]
    scored["mapping_available_at"] = scored[DECISION]
    scored["score_contract"] = (
        "research_only_frozen_final_heads_margin_interaction_"
        "causal_21d_retrospective"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scored_path = args.output_dir / "scored_population.parquet"
    support_path = args.output_dir / "calibration_support.parquet"
    scored.to_parquet(scored_path, index=False, compression="zstd")
    support.to_parquet(support_path, index=False, compression="zstd")

    manifest = {
        "schema": SCHEMA,
        "status": "research_only_retrospective_nonpromotable_not_forward_or_oos_evidence",
        "retrospective": True,
        "promotion_eligible": False,
        "contract": {
            "fixed_lookback_days": LOOKBACK_DAYS,
            "minimum_side_rows": int(args.minimum_side_rows),
            "history_rule": "execution_label_end_utc < decision and >= decision-21d",
            "resolved_updates": "none unless explicitly supplied and identity/score/timing validated",
            "mapping": "causal_recent_side_isotonic_ev_21d",
            "ranking": "one pooled global top10 across timestamps and sides after causal mapping",
            "admission_floors_bps": [0, 25, 50],
            "allow_zero_trades": True,
            "no_timestamp_side_asset_quota": True,
        },
        "rows": int(len(scored)),
        "candidate_identity_sha256": _identity_hash(scored),
        "globally_admitted_rows": int(scored["globally_admitted"].sum()),
        "decision_min_utc": scored[DECISION].min(),
        "decision_max_utc": scored[DECISION].max(),
        "calibration_support_rows": int(len(support)),
        "inputs": {
            "preentry": {"path": args.preentry, "sha256": _sha256(args.preentry)},
            "calibrator_state": {
                "path": args.calibrator_state,
                "sha256": _sha256(args.calibrator_state),
            },
            "seed_history": {
                "path": history_path,
                "sha256": _sha256(history_path),
            },
            "head_manifest": {
                "path": head_manifest_path,
                "sha256": _sha256(head_manifest_path),
            },
            "feature_contract": {
                "path": feature_contract_path,
                "sha256": _sha256(feature_contract_path),
            },
            "head_models": head_models,
            "resolved_updates": (
                {"path": args.resolved_updates, "sha256": _sha256(args.resolved_updates)}
                if args.resolved_updates is not None
                else None
            ),
        },
        "outputs": {
            "scored_population": {"path": scored_path, "sha256": _sha256(scored_path)},
            "calibration_support": {"path": support_path, "sha256": _sha256(support_path)},
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preentry", type=Path, default=DEFAULT_PREENTRY)
    parser.add_argument("--head-root", type=Path, default=DEFAULT_HEAD_ROOT)
    parser.add_argument("--calibrator-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--resolved-updates", type=Path)
    parser.add_argument("--minimum-side-rows", type=int, default=MINIMUM_SIDE_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(_parser())
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["rows"],
                "globally_admitted_rows": result["globally_admitted_rows"],
            },
            indent=2,
        )
    )
