#!/usr/bin/env python3
"""Run strict-R3 long-only bundles with lock-step immediate calibration.

Every 28-day refit rebuilds the upstream base/consensus and the conversion
model at one shared cutoff.  The preceding 28 calendar days are excluded from
*both* supervised fits, then target-free-scored by that exact producer pair.
After labels are joined, those reserve scores provide an immediate, exact
producer policy-net calibration from the first live hour.

This is intentionally separate from the historical staggered monthly-upstream
and four-week-conversion replay.  It tests the executable reserve contract;
it does not silently claim bit parity with the old cadence.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    CALIBRATION_RESERVE_DAYS,
    CORRECTNESS_FLOOR,
    CORRECTNESS_SPAN,
    FOUR_WEEK_DAYS,
    K9_TEMPERATURE_SCALE,
    META_TRAIN_MONTHS,
    REFERENCE_DAYS,
    persist_four_week_conversion_bundle,
    persist_monthly_upstream_bundle,
    load_four_week_conversion_bundle,
    load_monthly_upstream_bundle,
    score_four_week_conversion_bundle,
    score_four_week_conversion_bundle_lockstep,
    score_monthly_upstream_bundle,
    train_four_week_conversion_bundle,
    train_monthly_upstream_bundle,
    train_monthly_upstream_bundle_compact_features,
    FrozenGeometryK9View,
    ScoreReference,
    _aggregate_state_fields,
    _canonical_ldf_geometry_aliases,
    _current_ev_score_family_id,
    _numeric_matrix,
)
from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    assert_scoring_frame_is_target_free,
    load_geometry_bundle,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _fields(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = [str(value) for value in payload["base_fields_by_side"]["long"]]
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("schema-v4 requires the frozen 120-field long contract")
    return fields


def _prequential_ledger_columns(
    base_fields: list[str], *, include_base_fields: bool = True,
) -> list[str]:
    """Return the complete, minimal live-training ledger contract.

    The strict prequential ledger is intentionally wide because it is also a
    reusable research handoff.  The lock-step producer must not deserialize
    unrelated historical feature or diagnostic columns before its first
    checkpoint: doing so can exhaust memory before a bundle is fit, while
    adding neither a training input nor a scoring input.  This list is the
    union of the upstream, conversion, policy-supervision and identity
    contracts below.  It is not a feature-selection mechanism: every one of
    the frozen 120 base fields remains present.
    """
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "r3_class", "r3_label_available_ts",
        "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
        "prequential_base_score", "prequential_base_rank42",
        "prequential_base_anchor_bps", "prequential_consensus_rank",
        "prequential_residual_rank", "prequential_upstream",
        "stack_is_prequential",
        # policy_outcome_source is added only by the canonical-outcome join;
        # it is intentionally absent from the reusable prequential parquet.
        *(
            column for column in _CANONICAL_POLICY_COLUMNS
            if column != "policy_outcome_source"
        ),
    ]
    if include_base_fields:
        columns.extend(base_fields)
    # Parquet column projection rejects duplicates, whereas the policy
    # contract deliberately shares policy_path_valid with this local schema.
    return list(dict.fromkeys(columns))


def _materialise_frozen_features(
    source_panel: Path,
    ledger: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: list[str],
    label: str,
) -> pd.DataFrame:
    """Join the immutable source features only for one causal fit window.

    The ledger contains labels, prequential scores and identity lineage; the
    target-free panel contains the frozen 120-field feature contract.  Keeping
    them separate outside the active fit window prevents the multi-year wide
    panel from being retained across every lock-step block.  Candidate ID is
    deliberately the only join key: a natural-key fallback would hide a
    source-contract mismatch.
    """
    if ledger.empty:
        raise ValueError(f"{label} ledger is empty")
    features = _read_targetfree_source_columns(
        source_panel, start=start, end=end, fields=fields,
    ).loc[:, ["candidate_id", *fields]]
    if features["candidate_id"].duplicated().any():
        raise ValueError(f"{label} source feature window has duplicate candidate IDs")
    output = ledger.merge(features, on="candidate_id", how="left", validate="one_to_one")
    del features
    missing_identity_rows = int(output[fields].isna().all(axis=1).sum())
    missing_value_counts = output[fields].isna().sum()
    missing_value_fields = missing_value_counts.loc[missing_value_counts.gt(0)]
    if len(output) != len(ledger) or missing_identity_rows:
        raise ValueError(
            f"{label} source feature join is incomplete: {missing_identity_rows} "
            "candidate identities lack every frozen feature",
        )
    if len(missing_value_fields):
        # Preserve sparse source values until the receiving training routine
        # applies its persisted, training-fold-only medians.  This is the
        # same causal handling used by the upstream scorer; filling here
        # would risk using a wider or later fit population.
        print(
            json.dumps({
                "event": "source_feature_join_sparse_values",
                "label": label,
                "rows": int(len(output)),
                "missing_field_values": int(missing_value_counts.sum()),
                "missing_fields": int(len(missing_value_fields)),
                "sample": {
                    str(field): int(count)
                    for field, count in missing_value_fields.sort_values(ascending=False).head(12).items()
                },
                "handling": "receiving_bundle_training_fold_medians_only",
            }),
            flush=True,
        )
    return output


def _read_targetfree_source_columns(
    source_panel: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Read immutable frozen fields from a parquet panel or monthly store."""
    columns = ["candidate_id", "__decision_ts__", "__symbol__", "side_name", *fields]
    if source_panel.is_dir():
        manifest_path = source_panel / "run_manifest.json"
        if not manifest_path.exists():
            raise ValueError("monthly target-free source store lacks its manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "strict_r3_targetfree_month_store_v1":
            raise ValueError("source directory is not a strict target-free monthly store")
        declared = set(str(value) for value in manifest.get("fields", []))
        if not set(str(value) for value in fields).issubset(declared):
            raise ValueError("monthly target-free source store lacks requested frozen feature fields")
        periods = pd.period_range(
            start=start.tz_convert(None).to_period("M"),
            end=(end - pd.Timedelta(nanoseconds=1)).tz_convert(None).to_period("M"),
            freq="M",
        )
        paths = [source_panel / f"month={str(period)}" for period in periods]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise ValueError(f"monthly target-free source store is missing months: {missing}")
        frame = pd.concat(
            [pd.read_parquet(path, columns=columns) for path in paths],
            ignore_index=True,
        )
    else:
        frame = pd.read_parquet(
            source_panel,
            columns=columns,
            filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
        )
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    return frame.loc[
        frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)
    ].reset_index(drop=True)


def _materialise_selected_frozen_features(
    source_panel: Path,
    identities: pd.DataFrame,
    fields: list[str] | tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    """Project target-free fields for a fixed, causally selected ID set.

    Parquet performs the candidate-ID predicate before materialising the field
    columns.  This permits the strict 240k base cap and each complete-query
    ranker cap to be observed before the 120-field source matrix exists in
    memory.  The left merge retains the sampler's exact order.
    """
    # Working ledgers are indexed by candidate ID for progressive score
    # handoff.  Reset that index before the explicit ID join so pandas cannot
    # confuse it with the actual immutable identity column.
    identities = identities.reset_index(drop=True)
    if identities.empty or identities["candidate_id"].duplicated().any():
        raise ValueError(f"{label} identities are empty or duplicated")
    if "__decision_ts__" not in identities:
        raise ValueError(f"{label} identities need decision timestamps for bounded source projection")
    identities = identities.copy()
    identities["__selected_order__"] = np.arange(len(identities), dtype=np.int64)
    timestamp = pd.to_datetime(identities["__decision_ts__"], utc=True, errors="raise")
    months = timestamp.dt.to_period("M")
    pieces: list[pd.DataFrame] = []
    # A large `candidate_id IN (...)` predicate makes Arrow create an
    # oversized temporary hash table.  Reading one selected calendar month at
    # a time bounds that temporary state while retaining only the immutable
    # target-free source fields.  The candidate-ID merge remains the
    # authoritative source-alignment guard.
    for month in sorted(months.unique()):
        start = month.to_timestamp().tz_localize("UTC")
        end = (month + 1).to_timestamp().tz_localize("UTC")
        source_chunk = _read_targetfree_source_columns(
            source_panel, start=start, end=end, fields=fields,
        ).loc[:, ["candidate_id", *fields]]
        if source_chunk["candidate_id"].duplicated().any():
            raise ValueError(f"{label} source feature projection has duplicate candidate IDs")
        selected_chunk = identities.loc[months.eq(month)].copy()
        pieces.append(selected_chunk.merge(
            source_chunk, on="candidate_id", how="left", sort=False, validate="one_to_one",
        ))
        del source_chunk, selected_chunk
    output = pd.concat(pieces, ignore_index=True).sort_values(
        "__selected_order__", kind="stable",
    ).drop(columns="__selected_order__").reset_index(drop=True)
    print(json.dumps({"event": "selected_source_features_materialised", "label": label, "rows": int(len(output)), "fields": int(len(fields))}), flush=True)
    missing_identity_rows = int(output[list(fields)].isna().all(axis=1).sum())
    missing_value_counts = output[list(fields)].isna().sum()
    missing_value_fields = missing_value_counts.loc[missing_value_counts.gt(0)]
    if len(output) != len(identities) or missing_identity_rows:
        # A missing identity means the target-free source panel itself is not
        # aligned with the causally selected ledger.  That is fundamentally
        # different from a sparse primitive value: the frozen upstream
        # contract deliberately handles the latter through the bundle's
        # *training-fold* medians, which are then persisted and reused at
        # score time by ``_numeric_matrix``.  Rejecting sparse historical
        # values here would bypass that established causal path and makes
        # early replays impossible even though live scoring has the same
        # missing-value support.
        sample = {
            str(field): int(count)
            for field, count in missing_value_fields.sort_values(ascending=False).head(12).items()
        }
        raise ValueError(
            f"{label} source projection is incomplete: "
            f"{missing_identity_rows} rows lack every projected field; "
            f"{int(missing_value_counts.sum())} missing field values across "
            f"{len(missing_value_fields)} fields; sample={sample}",
        )
    if len(missing_value_fields):
        # The event is intentionally durable in stdout/checkpoint logs.  The
        # fitted bundle records medians separately; no fill occurs at this
        # source-projection boundary and no later rows can influence it.
        print(
            json.dumps({
                "event": "source_projection_sparse_values",
                "label": label,
                "rows": int(len(output)),
                "missing_field_values": int(missing_value_counts.sum()),
                "missing_fields": int(len(missing_value_fields)),
                "sample": {
                    str(field): int(count)
                    for field, count in missing_value_fields.sort_values(ascending=False).head(12).items()
                },
                "handling": "bundle_training_fold_medians_only",
            }),
            flush=True,
        )
    return output


def _read_target_free(
    source: Path | pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: list[str],
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.loc[
            source["__decision_ts__"].ge(start) & source["__decision_ts__"].lt(end)
        ].copy()
    else:
        frame = _read_targetfree_source_columns(
            source, start=start, end=end, fields=fields,
        )
    frame = frame.loc[frame["side_name"].astype(str).str.lower().eq("long")].copy()
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError(f"target-free source is empty or duplicated for {start} to {end}")
    assert_scoring_frame_is_target_free(frame)
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _read_prequential_ledger_window(
    ledger_path: Path,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
    fields: list[str],
) -> pd.DataFrame:
    """Load only the compact prequential supervision rows needed so far."""
    filters: list[tuple[str, str, pd.Timestamp]] = [("__decision_ts__", "<", end)]
    if start is not None:
        filters.insert(0, ("__decision_ts__", ">=", start))
    frame = pd.read_parquet(
        ledger_path,
        columns=_prequential_ledger_columns(fields, include_base_fields=False),
        filters=filters,
    )
    frame = frame.loc[frame["side_name"].astype(str).str.lower().eq("long")].copy()
    for column in (
        "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts",
        "h12_label_available_ts",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("prequential ledger window is empty or duplicated")
    return frame


_CANONICAL_POLICY_COLUMNS = (
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_label_available_ts", "policy_outcome_source",
    "policy_cost_bps",
)


def _require_source_aligned_ledger(ledger_path: Path, source_hash: str) -> dict[str, object]:
    """Reject a ledger made from another candidate/feature population.

    Candidate IDs are part of the executable contract.  A natural-key join is
    deliberately not an escape hatch: it could silently pair a score based on
    one cross-sectional population with labels or features generated from a
    different population.
    """
    manifest_path = ledger_path.parent / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"prequential ledger is missing its manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    observed = str(manifest.get("source_panel_sha256", ""))
    if observed != source_hash:
        raise ValueError(
            "prequential ledger was not generated from this target-free source "
            f"panel (ledger={observed or 'missing'}, source={source_hash}); "
            "regenerate strict OOF predictions on the scoring population"
        )
    reference_days = manifest.get("reference_window_days")
    if int(reference_days or -1) != REFERENCE_DAYS:
        raise ValueError(
            "canonical lock-step replay requires a prequential ledger with "
            f"reference_window_days={REFERENCE_DAYS}; observed "
            f"{reference_days if reference_days is not None else 'missing'}"
        )
    return manifest


def _attach_canonical_policy_supervision(
    ledger: pd.DataFrame,
    outcome_path: Path,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Replace inherited policy labels with the frozen-policy outcome contract.

    The prequential base target remains R3.  All map, consensus and conversion
    targets must instead use the exact same selected-policy net outcome that
    is joined after scoring for evaluation.  Rows preceding the available
    frozen-policy history are explicitly unavailable, rather than retaining a
    differently shaped legacy policy target.
    """
    required = {"candidate_id", "__decision_ts__", *_CANONICAL_POLICY_COLUMNS}
    columns = ["candidate_id", "__decision_ts__", *_CANONICAL_POLICY_COLUMNS]
    filters: list[tuple[str, str, pd.Timestamp]] = [("__decision_ts__", "<", end)]
    if start is not None:
        filters.insert(0, ("__decision_ts__", ">=", start))
    outcomes = pd.read_parquet(
        outcome_path,
        columns=columns,
        filters=filters,
    )
    missing = sorted(required.difference(outcomes.columns))
    if missing:
        raise ValueError(f"canonical policy outcome ledger lacks: {missing}")
    if outcomes["candidate_id"].duplicated().any():
        raise ValueError("canonical policy outcome ledger has duplicate candidate IDs")
    if ledger["candidate_id"].duplicated().any():
        raise ValueError("prequential ledger has duplicate candidate IDs")
    outcomes["__decision_ts__"] = pd.to_datetime(outcomes["__decision_ts__"], utc=True)
    # The caller owns the prequential ledger and does not retain an unmodified
    # copy.  Mutate it in place to avoid duplicating a multi-million-row,
    # 120-field frame immediately before every full replay.
    output = ledger
    output["__decision_ts__"] = pd.to_datetime(output["__decision_ts__"], utc=True)
    indexed = outcomes.set_index("candidate_id", drop=False)
    candidate_ids = pd.Index(output["candidate_id"])
    known = candidate_ids.isin(indexed.index)
    if known.any():
        expected_ts = indexed.loc[candidate_ids[known], "__decision_ts__"].to_numpy()
        actual_ts = output.loc[known, "__decision_ts__"].to_numpy()
        if not np.array_equal(expected_ts, actual_ts):
            raise ValueError("canonical policy outcome identity disagrees on decision timestamp")
    outcome_start = outcomes["__decision_ts__"].min()
    in_contract = output["__decision_ts__"].ge(outcome_start) & output["__decision_ts__"].lt(end)
    if (~known & in_contract).any():
        raise ValueError(
            "canonical policy outcome ledger does not cover every source-aligned "
            "prequential row within its declared history"
        )

    # Clear any historic policy outcome before attaching the one immutable
    # canonical target.  This avoids silently mixing a previous TP/trailing
    # geometry with the selected SimplePolicyOptimiser policy.
    output["policy_path_valid"] = pd.Series(False, index=output.index, dtype=bool)
    for column in _CANONICAL_POLICY_COLUMNS:
        if column == "policy_path_valid":
            continue
        if column == "policy_label_available_ts":
            output[column] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
        elif column in {"policy_exit_reason", "policy_outcome_source"}:
            output[column] = pd.Series(pd.NA, index=output.index, dtype="string")
        else:
            output[column] = np.nan
    if known.any():
        values = indexed.reindex(candidate_ids[known])
        positions = np.flatnonzero(known)
        for column in _CANONICAL_POLICY_COLUMNS:
            output.iloc[positions, output.columns.get_loc(column)] = values[column].to_numpy()

    label_ts = pd.to_datetime(output["policy_label_available_ts"], utc=True, errors="coerce")
    valid = output["policy_path_valid"].fillna(False).astype(bool)
    valid &= np.isfinite(pd.to_numeric(output["policy_net_bps"], errors="coerce"))
    valid &= label_ts.notna()
    if (label_ts.loc[valid] < output.loc[valid, "__decision_ts__"]).any():
        raise ValueError("canonical policy label becomes available before its decision")
    audit = {
        "outcome_path": str(outcome_path),
        "outcome_sha256": _sha(outcome_path),
        "policy_contract_start": outcome_start.isoformat(),
        "source_aligned_rows": int(len(output)),
        "canonical_policy_identity_rows": int(known.sum()),
        "canonical_policy_valid_rows": int(valid.sum()),
        "legacy_policy_rows_invalidated": int((~known).sum()),
        "in_contract_unmatched_rows": int((~known & in_contract).sum()),
        "policy_target": "frozen_selected_simple_policy_optimiser_net_bps",
    }
    return output, audit


def _initialise_working_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Seed earlier, already-prequential rows without changing their scores."""
    # The caller does not retain the unsuffixed input.  Reusing its storage
    # avoids a second full copy of the multi-million-row 120-field ledger at
    # the most memory-intensive point of a lock-step replay.
    output = ledger
    output["teacher_base_rank42"] = output["prequential_base_rank42"]
    aliases = {
        "base_score": "prequential_base_score",
        "base_rank42": "prequential_base_rank42",
        "base_anchor_bps": "prequential_base_anchor_bps",
        "conditional_consensus_rank": "prequential_consensus_rank",
        "upstream": "prequential_upstream",
        "ordinary_shadow_consensus_rank": "prequential_consensus_rank",
        "ordinary_shadow_upstream": "prequential_upstream",
    }
    for target, source in aliases.items():
        output[target] = output[source]
    return output.set_index("candidate_id", drop=False)


def _apply_upstream_scores(working: pd.DataFrame, score: pd.DataFrame) -> int:
    """Update only label-ledger identities; retain target-free-only candidates.

    The live candidate universe is deliberately larger than the supervised
    ledger because it contains rows without a resolved outcome.  Those rows
    must still be scored, but they cannot become future conversion-training
    rows merely to satisfy an internal join.
    """
    indexed = score.set_index("candidate_id", drop=False)
    known = indexed.index.intersection(working.index)
    # A scored target-free population and its prequential supervision ledger
    # must share the exact candidate identity.  Natural-key reconciliation is
    # intentionally not performed here: it can conceal a changed candidate
    # universe or changed cross-sectional feature values and would break the
    # training/scoring parity this runner is meant to guarantee.  Without this
    # guard, a zero-overlap handoff silently leaves later conversion fits on
    # stale upstream predictions.
    if len(known) == 0:
        raise ValueError(
            "lock-step upstream handoff has zero candidate-id overlap with "
            "the prequential ledger; regenerate a source-aligned strict-OOF "
            "ledger before continuing"
        )
    if len(known) != len(indexed):
        raise ValueError(
            "lock-step upstream handoff does not cover every scored target-free "
            "candidate; scoring and prequential training populations diverged"
        )
    mapping = {
        "base_score": "base_score",
        "base_rank42": "base_rank42",
        "prequential_base_rank42": "base_rank42",
        "base_anchor_bps": "base_anchor_bps",
        "conditional_consensus_rank": "conditional_consensus_rank",
        "upstream": "upstream",
        "ordinary_shadow_consensus_rank": "ordinary_shadow_consensus_rank",
        "ordinary_shadow_upstream": "ordinary_shadow_upstream",
    }
    for target, source in mapping.items():
        working.loc[known, target] = indexed.loc[known, source].to_numpy()
    working.loc[known, "stack_is_prequential"] = True
    return int(len(known))


def _attach_scores(raw: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id", "base_score", "base_rank42", "base_anchor_bps",
        "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream",
        "upstream_bundle_sha256",
    ]
    output = raw.merge(score.loc[:, columns], on="candidate_id", how="left", validate="one_to_one")
    if output[columns[1:]].isna().any().any():
        raise ValueError("lock-step upstream did not cover every conversion score row")
    return output


def _attach_outcomes_after_scoring(
    predictions: pd.DataFrame,
    outcome_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if "candidate_id" not in outcome_ledger or outcome_ledger["candidate_id"].duplicated().any():
        raise ValueError("outcome ledger requires unique candidate_id")
    columns = [
        "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
        "policy_exit_price", "policy_label_available_ts", "policy_outcome_source",
    ]
    available = [column for column in columns if column in outcome_ledger.columns]
    required = {"policy_path_valid", "policy_net_bps", "policy_label_available_ts"}
    if missing := sorted(required.difference(available)):
        raise ValueError(f"outcome ledger lacks required policy fields: {missing}")
    output = predictions.merge(
        outcome_ledger.loc[:, ["candidate_id", *available]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    if len(output) != len(predictions) or output["candidate_id"].duplicated().any():
        raise AssertionError("post-score outcome join changed prediction identities")
    return output, available


def _acquire_lock(directory: Path):
    path = directory / ".walkforward.run.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"output is already actively owned: {directory}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "out_dir": str(directory)}) + "\n")
    handle.flush()
    return handle


def _timestamp_chunks(frame: pd.DataFrame, *, hours: int):
    """Yield complete-hour chunks, never splitting a cross-section."""
    timestamps = pd.Index(pd.to_datetime(frame["__decision_ts__"], utc=True).unique()).sort_values()
    for start in range(0, len(timestamps), hours):
        values = timestamps[start:start + hours]
        yield frame.loc[frame["__decision_ts__"].isin(values)].copy()


def _history_with_memberships(
    history: pd.DataFrame,
    *,
    frame: pd.DataFrame,
    geometry_state: pd.DataFrame,
) -> pd.DataFrame:
    """Append target-free complete-hour K9 state for the next score chunk."""
    membership_columns = [f"k09__cluster_{index:02d}__membership" for index in range(9)]
    values = geometry_state.loc[:, membership_columns].copy()
    values.columns = [f"k{index}" for index in range(9)]
    values["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True).to_numpy()
    event = values.groupby("__decision_ts__", sort=True)[[f"k{index}" for index in range(9)]].sum().reset_index()
    output = pd.concat([history, event], ignore_index=True).sort_values("__decision_ts__", kind="stable")
    if output["__decision_ts__"].duplicated().any():
        raise AssertionError("lock-step geometry history repeated a scored timestamp")
    return output.reset_index(drop=True)


def _score_conversion_piece(
    bundle,
    *,
    frame: pd.DataFrame,
    role: str,
    geometry_history: pd.DataFrame,
    final_reference: ScoreReference | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one complete-hour piece without changing frozen model semantics."""
    local_parent = copy.copy(bundle.geometry.parent)
    local_parent.state_history = geometry_history
    local_geometry = FrozenGeometryK9View(
        parent=local_parent,
        temperature_scale=bundle.geometry.temperature_scale,
    )
    geometry_state = local_geometry.transform(frame)
    next_history = _history_with_memberships(
        geometry_history, frame=frame, geometry_state=geometry_state,
    )
    state = _canonical_ldf_geometry_aliases(
        pd.concat([geometry_state, bundle.leaf_trust.transform(frame)], axis=1),
    )
    aggregate = _aggregate_state_fields(state)
    combined = pd.concat(
        [frame.reset_index(drop=True), state.loc[:, list(aggregate)].reset_index(drop=True)], axis=1,
    )
    raw = bundle.correctness.model.predict(
        _numeric_matrix(combined, bundle.correctness.fields, bundle.correctness.medians),
    )
    combined["correctness_raw"] = raw
    combined["correctness_rank"] = bundle.correctness.score_reference.cdf(raw)
    combined["correctness_gate_active"] = combined["upstream"].ge(
        bundle.correctness.training_score_floor,
    )
    multiplier = CORRECTNESS_FLOOR + CORRECTNESS_SPAN * combined["correctness_rank"].to_numpy(float)
    combined["raw_correctness_demote"] = combined["upstream"].to_numpy(float) * np.where(
        combined["correctness_gate_active"].to_numpy(bool), multiplier, 1.0,
    )
    combined["final_score"] = (
        np.nan if final_reference is None
        else final_reference.cdf(combined["raw_correctness_demote"])
    )
    severe = bundle.severe_diagnostic
    combined["severe200_probability_shadow"] = (
        np.nan if severe.model is None else severe.model.predict_proba(
            _numeric_matrix(combined, severe.fields, severe.medians),
        )[:, 1]
    )
    combined["severe_affects_final_score"] = False
    combined["conversion_bundle_sha256"] = bundle.manifest.get("bundle_sha256", "unpersisted")
    combined["geometry_bundle_sha256"] = bundle.geometry.bundle_sha256
    combined["ev_score_family_id"] = _current_ev_score_family_id(bundle.geometry.bundle_sha256)
    upstream_fields = [
        "base_score", "base_rank42", "base_anchor_bps", "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream",
    ]
    columns = [
        "candidate_id", "__decision_ts__", *( ["__symbol__"] if "__symbol__" in combined else []),
        "side_name", *upstream_fields, "correctness_raw", "correctness_rank",
        "correctness_gate_active", "raw_correctness_demote", "final_score",
        "severe200_probability_shadow", "severe_affects_final_score",
        "conversion_bundle_sha256", "geometry_bundle_sha256", "ev_score_family_id",
        *( ["upstream_bundle_sha256"] if "upstream_bundle_sha256" in combined else []),
        *aggregate,
    ]
    output = combined.loc[:, columns].copy()
    output["__score_role__"] = role
    return output, next_history


def _score_conversion_memory_bound(
    bundle,
    *,
    reference: pd.DataFrame,
    held: pd.DataFrame,
    chunk_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper over the canonical inference scorer.

    Keeping the walk-forward producer as a caller rather than an independent
    implementation makes the persisted-bundle forward CLI and historical
    replay use one source of K9/CDF semantics.
    """
    return score_four_week_conversion_bundle_lockstep(
        bundle, reference=reference, held=held, chunk_hours=chunk_hours,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument(
        "--held-source-panel", type=Path,
        help=(
            "Optional target-free held-only feature panel.  It may extend the "
            "immutable historical source without changing the training prefix."
        ),
    )
    parser.add_argument(
        "--held-candidates", type=Path,
        help="Optional exact eligible candidate-ID population for the held-only panel.",
    )
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--geometry-bundle", type=Path, required=True)
    parser.add_argument("--outcome-ledger", type=Path)
    parser.add_argument(
        "--held-outcome-ledger", type=Path,
        help="Optional held-only outcomes joined strictly after target-free scoring.",
    )
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-reserve-days", type=int, default=CALIBRATION_RESERVE_DAYS)
    parser.add_argument(
        "--score-chunk-hours", type=int, default=72,
        help="Complete-hour conversion score chunk; preserves exact state while bounding memory.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse only hash-matched persisted lock-step bundle checkpoints.",
    )
    args = parser.parse_args()
    if args.calibration_reserve_days != CALIBRATION_RESERVE_DAYS:
        raise ValueError(f"lock-step immediate calibration requires {CALIBRATION_RESERVE_DAYS} reserve days")
    if args.score_chunk_hours < 1:
        raise ValueError("--score-chunk-hours must be positive")
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(f"immutable lock-step output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(args.out_dir)
    atexit.register(lock.close)

    fields = _fields(args.feature_contract)
    start, end = _utc(args.evaluation_start), _utc(args.evaluation_end)
    if end <= start:
        raise ValueError("evaluation end must be after start")
    geometry = load_geometry_bundle(args.geometry_bundle)
    print(json.dumps({"event": "runner_geometry_loaded"}), flush=True)
    audit = geometry.fit_audit
    if (
        audit.get("definition_start") != "2024-10-01T00:00:00+00:00"
        or audit.get("definition_end_exclusive") != "2025-01-01T00:00:00+00:00"
    ):
        raise ValueError("lock-step replay requires frozen Oct-Dec 2024 geometry/K9")
    # Keep the full frozen 120-field contract but do not materialise unrelated
    # research diagnostics from the multi-million-row ledger.  The exact
    # required-column checks inside the upstream and conversion trainers
    # remain the authoritative guard against a missing training input.
    # Retain only past compact supervision at process start.  Each scored
    # block's compact ledger rows are appended after its producer has been
    # fit, so subsequent bundles see exactly the same causal history without
    # holding the full 2026 target/feature population in memory.
    ledger = _read_prequential_ledger_window(
        args.prequential_ledger, start=None, end=start, fields=fields,
    )
    print(json.dumps({"event": "runner_initial_ledger_loaded", "rows": int(len(ledger))}), flush=True)
    source_panel_hash = (
        str(json.loads((args.source_panel / "run_manifest.json").read_text())["source_panel_sha256"])
        if args.source_panel.is_dir() else _sha(args.source_panel)
    )
    source_hashes = {
        "source_panel": source_panel_hash,
        "prequential_ledger": _sha(args.prequential_ledger),
        "feature_contract": _sha(args.feature_contract),
        "geometry_manifest": _sha(args.geometry_bundle / "run_manifest.json"),
        "calibration_reserve_days": str(args.calibration_reserve_days),
        "refit_cadence": f"lockstep_{FOUR_WEEK_DAYS}d",
    }
    if args.held_source_panel is not None:
        source_hashes["held_source_panel"] = _sha(args.held_source_panel)
        if args.held_candidates is None:
            raise ValueError("--held-source-panel requires --held-candidates")
        source_hashes["held_candidates"] = _sha(args.held_candidates)
        if len(pd.date_range(start, end, freq=f"{FOUR_WEEK_DAYS}D", inclusive="left")) != 1:
            raise ValueError("held-only source extension is limited to one fresh producer block")
    ledger_manifest = _require_source_aligned_ledger(
        args.prequential_ledger, source_hashes["source_panel"],
    )
    policy_training_audit: dict[str, object] | None = None
    if args.outcome_ledger is not None:
        # The initial compact ledger stops at the first live cutoff.  Loading
        # later labels here would be needless memory pressure and would make
        # the producer retain future outcome data before it is needed.
        ledger, policy_training_audit = _attach_canonical_policy_supervision(
            ledger, args.outcome_ledger, end=start,
        )
        print(json.dumps({"event": "runner_initial_policy_join_complete", "rows": int(len(ledger))}), flush=True)
        source_hashes["canonical_policy_outcomes"] = policy_training_audit["outcome_sha256"]
    working = _initialise_working_ledger(ledger)
    print(json.dumps({"event": "runner_working_ledger_initialised", "rows": int(len(working))}), flush=True)
    
    def load_checkpoint(directory: Path, loader, cutoff: pd.Timestamp, label: str):
        bundle = loader(directory)
        if _utc(bundle.cutoff) != cutoff:
            raise ValueError(f"{label} checkpoint has the wrong cutoff: {directory}")
        if dict(bundle.manifest.get("source_hashes", {})) != source_hashes:
            raise ValueError(f"{label} checkpoint has different source lineage: {directory}")
        if int(bundle.manifest.get("calibration_reserve_days", -1)) != args.calibration_reserve_days:
            raise ValueError(f"{label} checkpoint has the wrong reserve duration: {directory}")
        return bundle

    cutoffs = pd.date_range(start, end, freq=f"{FOUR_WEEK_DAYS}D", inclusive="left")
    block_rows: list[dict[str, object]] = []
    conversion_audits: list[pd.DataFrame] = []
    for index, cutoff in enumerate(cutoffs):
        cutoff = _utc(cutoff)
        scheduled_held_end = cutoff + pd.Timedelta(days=FOUR_WEEK_DAYS)
        # A final evaluation fragment is still scored by the same 28-day live
        # producer that would remain active through its scheduled refit.  Do
        # not let the end of the retrospective sample change the fitted bundle.
        held_end = min(scheduled_held_end, end)
        reserve_start = cutoff - pd.Timedelta(days=args.calibration_reserve_days)
        # By construction the working ledger contains exactly the rows before
        # this cutoff.  Passing it by reference avoids a second multi-year
        # compact ledger while the base/map/consensus sampler is active.
        prior = working
        print(json.dumps({"event": "runner_block_begin", "cutoff": cutoff.isoformat(), "prior_rows": int(len(prior))}), flush=True)
        # Do not retain an evaluation-wide 120-column source cache while both
        # fitted producers and the 28-day reference are resident.  The two
        # predicate-pushed parquet reads are byte-equivalent and cap peak RAM.
        raw_reference = _read_target_free(args.source_panel, start=reserve_start, end=cutoff, fields=fields)
        raw_held = _read_target_free(
            args.held_source_panel or args.source_panel,
            start=cutoff, end=held_end, fields=fields,
        )
        if args.held_source_panel is not None:
            held_candidates = pd.read_parquet(args.held_candidates, columns=["candidate_id"])
            if held_candidates["candidate_id"].duplicated().any():
                raise ValueError("held candidate population has duplicate candidate IDs")
            raw_held = raw_held.loc[
                raw_held["candidate_id"].isin(held_candidates["candidate_id"])
            ].copy().reset_index(drop=True)
            if raw_held.empty:
                raise ValueError("held candidate eligibility filter selected no feature rows")
        expected_reference_hours = pd.date_range(
            reserve_start, cutoff - pd.Timedelta(hours=1), freq="h", tz="UTC",
        )
        observed_reference_hours = pd.DatetimeIndex(
            raw_reference["__decision_ts__"].drop_duplicates(),
        ).sort_values()
        zero_candidate_reference_hours = expected_reference_hours.difference(
            observed_reference_hours,
        )
        reserve_rows = int(len(raw_reference))

        upstream_dir = args.out_dir / "bundles" / f"cutoff={cutoff:%Y%m%d}" / "upstream"
        if args.resume and (upstream_dir / "run_manifest.json").exists():
            upstream = load_checkpoint(
                upstream_dir, load_monthly_upstream_bundle, cutoff, "upstream",
            )
            upstream_status = "resumed"
        else:
            def load_selected_upstream_features(
                identities: pd.DataFrame,
                selected_fields: list[str] | tuple[str, ...],
                label: str,
            ) -> pd.DataFrame:
                return _materialise_selected_frozen_features(
                    args.source_panel, identities, selected_fields, label,
                )

            upstream = train_monthly_upstream_bundle_compact_features(
                cutoff=cutoff,
                training_ledger=prior,
                prior42_features=raw_reference,
                base_fields=fields,
                feature_loader=load_selected_upstream_features,
                permitted_empty_reference_hours=tuple(zero_candidate_reference_hours),
                source_hashes=source_hashes,
                calibration_reserve_days=args.calibration_reserve_days,
                held_end_exclusive=scheduled_held_end,
            )
            persist_monthly_upstream_bundle(upstream, upstream_dir)
            upstream_status = "fit"
        upstream_reference = score_monthly_upstream_bundle(
            upstream, raw_reference, allow_prior_reference=True, prior_reference_start=reserve_start,
        )
        upstream_held = score_monthly_upstream_bundle(upstream, raw_held)

        # The conversion trainer itself uses only this exact six-calendar-month
        # window.  Slicing here avoids retaining an older multi-year wide
        # upstream panel through leaf-trust and correctness fitting; it is
        # byte-equivalent to the trainer's own chronological predicate.
        conversion_train_start = (
            (cutoff.tz_convert(None).to_period("M") - META_TRAIN_MONTHS)
            .to_timestamp().tz_localize("UTC")
        )
        conversion_prior_meta = working.loc[
            working["__decision_ts__"].ge(conversion_train_start)
            & working["__decision_ts__"].lt(cutoff)
        ].copy().reset_index(drop=True)
        conversion_prior = _materialise_frozen_features(
            args.source_panel, conversion_prior_meta,
            start=conversion_train_start, end=cutoff, fields=fields,
            label="four-week conversion training",
        )
        del conversion_prior_meta

        conversion_dir = args.out_dir / "bundles" / f"cutoff={cutoff:%Y%m%d}" / "conversion"
        if args.resume and (conversion_dir / "run_manifest.json").exists():
            conversion = load_checkpoint(
                conversion_dir, load_four_week_conversion_bundle, cutoff, "conversion",
            )
            conversion_status = "resumed"
        else:
            conversion = train_four_week_conversion_bundle(
                cutoff=cutoff,
                upstream_ledger=conversion_prior,
                frozen_geometry=geometry,
                base_fields=fields,
                source_hashes=source_hashes,
                calibration_reserve_days=args.calibration_reserve_days,
            )
            persist_four_week_conversion_bundle(conversion, conversion_dir)
            conversion_status = "fit"
        reference_input = _attach_scores(raw_reference, upstream_reference)
        held_input = _attach_scores(raw_held, upstream_held)
        # Make the newly held block available only after its producer pair has
        # been fixed.  Its inherited prequential fields are then overwritten
        # by the new upstream score before the next cutoff, preserving the
        # original progressive handoff without retaining future rows.
        if args.held_source_panel is None:
            held_ledger = _read_prequential_ledger_window(
                args.prequential_ledger, start=cutoff, end=held_end, fields=fields,
            )
            if args.outcome_ledger is not None:
                held_ledger, _ = _attach_canonical_policy_supervision(
                    held_ledger, args.outcome_ledger, start=cutoff, end=held_end,
                )
            held_ledger = _initialise_working_ledger(held_ledger)
            if working.index.intersection(held_ledger.index).size:
                raise AssertionError("lock-step compact ledger attempted to append duplicate identities")
            working = pd.concat([working, held_ledger], axis=0, copy=False)
            del held_ledger
            upstream_supervised_rows = _apply_upstream_scores(working, upstream_held)
        else:
            # A fresh forward population is target-free and has no OOF ledger
            # yet.  It is scored by the newly frozen producer, but cannot be
            # appended to supervised history until its labels resolve and a
            # later prequential ledger is explicitly materialised.
            upstream_supervised_rows = 0
        del raw_reference, raw_held, upstream_reference, upstream_held
        scored, conversion_audit = _score_conversion_memory_bound(
            conversion,
            reference=reference_input,
            held=held_input,
            chunk_hours=args.score_chunk_hours,
        )
        scored["calibration_reserve_start"] = reserve_start
        scored["calibration_activation_ts"] = cutoff
        score_ts = pd.to_datetime(scored["__decision_ts__"], utc=True, errors="raise")
        scored["calibration_reference_oos_to_all_active_fits"] = (
            scored["__score_role__"].eq("reference")
            & score_ts.ge(reserve_start)
            & score_ts.lt(cutoff)
        )
        scored["calibration_reference_contract"] = (
            "full 28-day shared reserve excluded from lock-step upstream and conversion supervised fits"
        )
        held = scored.loc[scored["__score_role__"].eq("held")].drop(columns="__score_role__").copy()
        held["stack_is_prequential"] = True
        reference_scored = scored.loc[scored["__score_role__"].eq("reference")].copy()
        # Persist target-free score checkpoints immediately.  These make a
        # long live-like replay recoverable after an interruption without
        # altering model artifacts or reusing a different producer.
        score_dir = args.out_dir / "bundles" / f"cutoff={cutoff:%Y%m%d}" / "scores"
        score_dir.mkdir(exist_ok=True)
        held.to_parquet(score_dir / "held_target_free_scores.parquet", index=False, compression="zstd")
        reference_scored.to_parquet(
            score_dir / "reserve_target_free_scores.parquet", index=False, compression="zstd",
        )
        conversion_audit.to_parquet(score_dir / "conversion_score_audit.parquet", index=False)
        conversion_audits.append(conversion_audit.assign(block_index=index))
        block_rows.append({
            "block_index": index,
            "cutoff": cutoff,
            "held_end_exclusive": held_end,
            "scheduled_held_end_exclusive": scheduled_held_end,
            "reserve_start": reserve_start,
            "reserve_days": args.calibration_reserve_days,
            "reserve_zero_candidate_hours": [
                value.isoformat() for value in zero_candidate_reference_hours
            ],
            "held_rows": int(len(held)),
            "held_rows_available_to_future_supervised_conversion": upstream_supervised_rows,
            "reserve_rows": reserve_rows,
            "upstream_bundle_sha256": upstream.manifest["bundle_sha256"],
            "conversion_bundle_sha256": conversion.manifest["bundle_sha256"],
            "geometry_bundle_sha256": conversion.geometry.bundle_sha256,
            "upstream_status": upstream_status,
            "conversion_status": conversion_status,
            "same_refit_cutoff": True,
            "full_shared_reserve": True,
        })
        print(json.dumps({"event": "lockstep_block_complete", **block_rows[-1]}, default=str), flush=True)
        # The immutable score checkpoint is the cross-block handoff.  Do not
        # retain every 28-day reference and held population in memory while
        # fitting the next producer pair.
        del scored, reference_input, held_input, held, reference_scored

    held_paths = sorted(args.out_dir.glob("bundles/cutoff=*/scores/held_target_free_scores.parquet"))
    reference_paths = sorted(args.out_dir.glob("bundles/cutoff=*/scores/reserve_target_free_scores.parquet"))
    if len(held_paths) != len(cutoffs) or len(reference_paths) != len(cutoffs):
        raise AssertionError("lock-step replay is missing a target-free score checkpoint")
    final = pd.concat([pd.read_parquet(path) for path in held_paths], ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if final["candidate_id"].duplicated().any():
        raise AssertionError("lock-step replay duplicated held candidate identities")
    final.to_parquet(args.out_dir / "walkforward_predictions.parquet", index=False, compression="zstd")
    reference = pd.concat([pd.read_parquet(path) for path in reference_paths], ignore_index=True).sort_values(
        ["calibration_activation_ts", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    reference.to_parquet(
        args.out_dir / "immediate_calibration_reference_scores.parquet",
        index=False, compression="zstd",
    )
    outcome_columns: list[str] = []
    evaluation_outcome_ledger = args.held_outcome_ledger or args.outcome_ledger
    if evaluation_outcome_ledger is not None:
        outcomes = pd.read_parquet(evaluation_outcome_ledger)
        labelled, outcome_columns = _attach_outcomes_after_scoring(final, outcomes)
        labelled.to_parquet(args.out_dir / "walkforward_scored_label_ledger.parquet", index=False, compression="zstd")
    pd.DataFrame(block_rows).to_parquet(args.out_dir / "lockstep_block_audit.parquet", index=False)
    pd.concat(conversion_audits, ignore_index=True).to_parquet(
        args.out_dir / "conversion_reference_audit.parquet", index=False,
    )
    manifest = {
        "schema": "strict_r3_lockstep_immediate_calibration_v1",
        "side": "long",
        "evaluation_start": start.isoformat(),
        "evaluation_end_exclusive": end.isoformat(),
        "refit_cadence_days": FOUR_WEEK_DAYS,
        "conversion_score_chunk_hours": args.score_chunk_hours,
        "calibration_reserve": {
            "days": args.calibration_reserve_days,
            "shared_cutoff": True,
            "contract": "reserve excluded from every active supervised upstream and conversion fit",
        },
        "blocks": len(block_rows),
        "rows": len(final),
        "geometry": {
            "refit_cadence": "never",
            "parent_bundle_sha256": geometry.bundle_sha256,
            "definition_start": audit["definition_start"],
            "definition_end_exclusive": audit["definition_end_exclusive"],
        },
        "reference_window_days": REFERENCE_DAYS,
        "normalization": "same lock-step producer prior-28 CDF; no held-window percentile",
        "held_percentile_operations": 0,
        "outcomes_consumed_during_scoring": [],
        "outcome_join": {
            "performed_after_scoring": evaluation_outcome_ledger is not None,
            "evaluation_outcome_ledger": (
                str(evaluation_outcome_ledger) if evaluation_outcome_ledger else None
            ),
            "columns": outcome_columns,
        },
        "prequential_ledger": {
            "source_aligned": True,
            "manifest_schema": ledger_manifest.get("schema"),
            "source_panel_sha256": ledger_manifest.get("source_panel_sha256"),
            "reference_window_days": ledger_manifest.get("reference_window_days"),
        },
        "policy_training_supervision": policy_training_audit,
        "immediate_calibration_reference": {
            "target_free_scores": "immediate_calibration_reference_scores.parquet",
            "full_shared_reserve_rows": int(len(reference)),
            "identity": "candidate_id x conversion_bundle_sha256 x upstream_bundle_sha256 x activation",
        },
        "source_hashes": source_hashes,
        "meta_train_months": META_TRAIN_MONTHS,
        "k9_temperature_scale": K9_TEMPERATURE_SCALE,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}, default=str))


if __name__ == "__main__":
    main()
