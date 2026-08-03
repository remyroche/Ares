#!/usr/bin/env python3
"""Materialise frozen exact-H12 execution targets and supporting path labels.

The primary target is deliberately small and authoritative: the signed
current-frozen-spread execution replay's exact 12-hour net EV.  It is *not*
recomputed from a raw price path, because the replay can have a policy exit
which is not recoverable from OHLC alone.

Supporting labels use the same complete, immutable, decision-aligned 1-minute
path.  They are explicitly labelled as path outcomes rather than as realised
policy EV: (a) the existing five auxiliary target heads retain their frozen
unadjusted-path semantics; and (b) a zero-buffer, row-cost-aware competing
risk pack measures cost-clearing reachability before an ATR adverse move.
Every output is unavailable until ``decision_ts + 12h`` and is target-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_auxiliary_targets import (  # noqa: E402
    ALL_SUPPORTIVE_LABEL_COLUMNS,
    TARGET_COLUMNS,
    TARGET_SCHEMA,
    build_path_auxiliary_targets,
)
from scripts.materialize_execution_ev_cost_aware_competing_risk_labels import (  # noqa: E402
    HORIZON_MINUTES,
    build_row_cost_aware_competing_risk_labels,
)
from scripts.materialize_historical_exact_h12_alignment_sidecar import (  # noqa: E402
    COST_MODEL_ID,
    EXECUTION_POLICY_ID,
    TARGET_ID,
    validate_alignment,
)


SCHEMA = "root_cause_exact_h12_execution_target_pack_v1"
PATH_SEMANTICS_ID = "historical_exact_1m_unadjusted_decision_path_v1"
COMPETING_RISK_ID = "row_cost_aware_exact_1m_h12_competing_risk_v1"
PRIMARY_TARGET_ID = TARGET_ID
DEFAULT_ALIGNMENT = (
    ROOT
    / "data_perp/artifacts/historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
)
DEFAULT_PATHS = (
    ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_paths_20260730_v1/paths.parquet",
    ROOT / "data_perp/artifacts/failure_2024_exact1m_paths_20260730_v2/paths.parquet",
)
# v1 is retained as a superseded float32 accounting surface.  The default is
# the float64-contract revision; it is the only pack eligible for downstream
# exact-H12 target consumers.
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/root_cause_exact_h12_execution_target_pack_20260801_v2"

IDENTITY_COLUMNS = ("candidate_id", "symbol", "side", "decision_ts")
TIMING_COLUMNS = ("label_end_ts", "label_available_ts")
PATH_COLUMNS = (
    "candidate_id",
    "__ts__",
    "__symbol__",
    "side_name",
    "execution_future_path",
    "atr_1h",
    "decision_price",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _as_utc(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")


def _feature_set_id(alignment: pd.DataFrame) -> str:
    values = alignment["feature_set_id"].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError("alignment must bind exactly one raw feature-set ID")
    return str(values[0])


def validate_exact_h12_target_contract(alignment: pd.DataFrame) -> None:
    """Fail closed before any target is materialised from a path source."""
    required = {
        *IDENTITY_COLUMNS,
        "feature_cutoff_ts",
        *TIMING_COLUMNS,
        "entry_ts",
        "target_id",
        "execution_policy_id",
        "replay_execution_policy_id",
        "cost_model_id",
        "feature_set_id",
        "execution_geometry_id",
        "execution_geometry_key",
        "execution_geometry_source",
        "barrier_pct",
        "execution_entry_price",
        "exact_h12_gross_bps",
        "row_cost_bps",
        "exact_h12_net_bps",
    }
    missing = sorted(required.difference(alignment.columns))
    if missing:
        raise ValueError(f"alignment lacks required exact-H12 contract fields: {missing}")
    _as_utc(alignment, ("decision_ts", "feature_cutoff_ts", "entry_ts", *TIMING_COLUMNS))
    validate_alignment(alignment, feature_set_id=_feature_set_id(alignment))
    if not alignment["execution_geometry_key"].notna().all() or not alignment["execution_geometry_source"].notna().all():
        raise ValueError("execution geometry lineage is incomplete")
    for column in ("barrier_pct", "execution_entry_price"):
        values = pd.to_numeric(alignment[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError(f"{column} must be finite and strictly positive")
    if not alignment["target_id"].eq(PRIMARY_TARGET_ID).all():
        raise ValueError("primary target ID differs from the frozen exact-H12 contract")
    if not alignment["execution_policy_id"].eq(EXECUTION_POLICY_ID).all():
        raise ValueError("execution policy ID differs from the frozen exact-H12 contract")
    if not alignment["cost_model_id"].eq(COST_MODEL_ID).all():
        raise ValueError("cost model ID differs from the frozen exact-H12 contract")


def _load_alignment(path: Path) -> pd.DataFrame:
    alignment = pd.read_parquet(path)
    validate_exact_h12_target_contract(alignment)
    if alignment.candidate_id.duplicated().any():
        raise ValueError("alignment candidate identity is not unique")
    return alignment.set_index("candidate_id", verify_integrity=True)


def _decode_paths(values: Iterable[str], decision_ts: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parsed = [json.loads(value) for value in values]
    expected = {"timestamp", "open", "high", "low", "close"}
    if any(set(payload) != expected for payload in parsed):
        raise ValueError("exact path encoding must be the complete signed OHLC vector")
    arrays = tuple(
        np.asarray([payload[column] for payload in parsed], dtype=np.float64)
        for column in ("open", "high", "low", "close")
    )
    if any(array.ndim != 2 or array.shape[1] != HORIZON_MINUTES for array in arrays):
        raise ValueError("exact paths must contain exactly 720 one-minute OHLC observations")
    if any(not np.isfinite(array).all() or (array <= 0.0).any() for array in arrays):
        raise ValueError("exact path OHLC values must be finite and strictly positive")
    open_, high, low, close = arrays
    if (high < low).any() or (high < np.minimum(open_, close)).any() or (low > np.maximum(open_, close)).any():
        raise ValueError("exact path OHLC geometry is invalid")
    timestamps = np.asarray([payload["timestamp"] for payload in parsed], dtype=np.int64)
    expected_timestamps = (
        decision_ts.astype("int64").to_numpy(dtype=np.int64)[:, None]
        + np.arange(HORIZON_MINUTES, dtype=np.int64)[None, :] * pd.Timedelta(minutes=1).value
    )
    if not np.array_equal(timestamps, expected_timestamps):
        raise ValueError("path timestamps are not exact one-minute decision-aligned H12 paths")
    return arrays


def _side_sign(values: pd.Series) -> np.ndarray:
    side = values.astype(str).str.lower().to_numpy()
    if not np.isin(side, ("long", "short")).all():
        raise ValueError("only canonical long/short path identities are permitted")
    return np.where(side == "long", 1.0, -1.0)


def _validate_path_identity(paths: pd.DataFrame, context: pd.DataFrame) -> None:
    if paths.candidate_id.isna().any() or paths.candidate_id.duplicated().any():
        raise ValueError("path source candidate identity is missing or duplicated")
    ids = paths.candidate_id.astype(str)
    if not ids.isin(context.index).all():
        raise ValueError("path source contains a candidate outside the frozen alignment")
    matched = context.loc[ids].reset_index()
    comparisons = (
        # The path record retains the candidate's signal identity; its encoded
        # first one-minute OHLC timestamp is separately required to equal the
        # execution decision.  Collapsing those two clocks would silently
        # admit a one-bar look-ahead error.
        ("__ts__", "feature_cutoff_ts", "signal timestamp"),
        ("__symbol__", "symbol", "symbol"),
        ("side_name", "side", "side"),
    )
    for path_column, context_column, description in comparisons:
        if path_column not in paths:
            raise ValueError(f"path source lacks {path_column}")
        if path_column == "__ts__":
            lhs = pd.to_datetime(paths[path_column], utc=True, errors="coerce")
            rhs = pd.to_datetime(matched[context_column], utc=True, errors="coerce")
        else:
            lhs = paths[path_column].astype(str).str.lower() if path_column == "side_name" else paths[path_column].astype(str)
            rhs = matched[context_column].astype(str).str.lower() if context_column == "side" else matched[context_column].astype(str)
        if lhs.isna().any() or rhs.isna().any() or not lhs.reset_index(drop=True).eq(rhs).all():
            raise ValueError(f"path/alignment exact identity mismatch in {description}")


def _primary_output(context: pd.DataFrame, *, atr_fraction: np.ndarray) -> pd.DataFrame:
    net = context.exact_h12_net_bps.to_numpy(dtype=float)
    gross = context.exact_h12_gross_bps.to_numpy(dtype=float)
    cost = context.row_cost_bps.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "candidate_id": context.candidate_id.to_numpy(),
            "symbol": context.symbol.astype(str).to_numpy(),
            "side": context.side.astype(str).to_numpy(),
            "decision_ts": context.decision_ts.to_numpy(),
            "feature_cutoff_ts": context.feature_cutoff_ts.to_numpy(),
            "entry_ts": context.entry_ts.to_numpy(),
            "label_end_ts": context.label_end_ts.to_numpy(),
            "label_available_ts": context.label_available_ts.to_numpy(),
            "primary_target_id": np.full(len(context), PRIMARY_TARGET_ID, dtype=object),
            "execution_policy_id": context.execution_policy_id.astype(str).to_numpy(),
            "cost_model_id": context.cost_model_id.astype(str).to_numpy(),
            "feature_set_id": context.feature_set_id.astype(str).to_numpy(),
            "execution_geometry_id": context.execution_geometry_id.astype(str).to_numpy(),
            "execution_geometry_key": context.execution_geometry_key.astype(str).to_numpy(),
            "execution_geometry_source": context.execution_geometry_source.astype(str).to_numpy(),
            "execution_entry_price": context.execution_entry_price.to_numpy(dtype=np.float32),
            "path_auxiliary_atr_fraction": atr_fraction.astype(np.float32),
            # These are the authoritative accounting fields.  Keep their
            # ledger precision so the published surface itself preserves the
            # frozen ``gross - row_cost == net`` invariant exactly, rather
            # than merely approximately after a float32 round-trip.
            "execution_exact_h12_gross_bps": gross.astype(np.float64),
            "execution_exact_h12_cost_bps": cost.astype(np.float64),
            "execution_exact_h12_net_bps": net.astype(np.float64),
            "execution_exact_h12_net_positive": (net > 0.0).astype(np.int8),
            "execution_exact_h12_net_nonnegative": (net >= 0.0).astype(np.int8),
        }
    )


def build_exact_h12_target_rows(paths: pd.DataFrame, alignment: pd.DataFrame, *, include_full_auxiliary_support: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build primary and supporting labels for an already identity-matched batch.

    This public, in-memory seam is intentionally used by correctness tests.  A
    caller must pass the immutable raw path rows; all timing, OHLC and geometry
    violations raise rather than produce a partially comparable label.
    """
    missing = sorted(set(PATH_COLUMNS).difference(paths.columns))
    if missing:
        raise ValueError(f"path batch lacks required columns: {missing}")
    if alignment.index.name != "candidate_id":
        if "candidate_id" not in alignment:
            raise ValueError("alignment must carry candidate_id")
        alignment = alignment.set_index("candidate_id", verify_integrity=True)
    _validate_path_identity(paths, alignment)
    context = alignment.loc[paths.candidate_id.astype(str)].reset_index()
    _as_utc(context, ("decision_ts", "feature_cutoff_ts", "entry_ts", *TIMING_COLUMNS))
    decision_price = pd.to_numeric(paths.decision_price, errors="coerce").to_numpy(dtype=float)
    atr_absolute = pd.to_numeric(paths.atr_1h, errors="coerce").to_numpy(dtype=float)
    atr_fraction = np.divide(
        atr_absolute,
        decision_price,
        out=np.full(len(paths), np.nan, dtype=float),
        where=decision_price > 0.0,
    )
    if not np.isfinite(atr_fraction).all() or (atr_fraction <= 0.0).any():
        raise ValueError("path-source decision ATR fraction must be finite and strictly positive")
    open_, high, low, close = _decode_paths(paths.execution_future_path, context.decision_ts)
    sign = _side_sign(context.side)
    auxiliary = build_path_auxiliary_targets(
        entry_price=decision_price,
        future_high=high,
        future_low=low,
        atr_fraction=atr_fraction,
        side_sign=sign,
        bar_minutes=1,
        horizon_hours=12,
        include_supportive_columns=include_full_auxiliary_support,
    ).as_columns()
    competing = build_row_cost_aware_competing_risk_labels(
        open_, high, low, close,
        oof_entry_atr_fraction=atr_fraction,
        execution_cost_return=context.row_cost_bps.to_numpy(dtype=float) / 10_000.0,
        execution_entry_price=context.execution_entry_price.to_numpy(dtype=float),
        side_sign=sign,
        decision_utc=context.decision_ts,
        buffer_bps=0,
        use_upper_return_floor=True,
    ).rename(
        columns={
            "oof_entry_atr_fraction": "competing_risk_atr_fraction",
            "label_available_at": "competing_risk_label_available_ts",
            "label_resolution_utc": "competing_risk_label_end_ts",
        }
    )
    if not pd.Series(competing["competing_risk_label_end_ts"]).eq(context.label_end_ts).all():
        raise ValueError("competing-risk label end loses the exact H12 timing contract")
    primary = _primary_output(context, atr_fraction=atr_fraction)
    supporting: dict[str, Any] = {
        "candidate_id": context.candidate_id.to_numpy(),
        "symbol": context.symbol.astype(str).to_numpy(),
        "side": context.side.astype(str).to_numpy(),
        "decision_ts": context.decision_ts.to_numpy(),
        "label_end_ts": context.label_end_ts.to_numpy(),
        "label_available_ts": context.label_available_ts.to_numpy(),
        "support_path_semantics_id": np.full(len(context), PATH_SEMANTICS_ID, dtype=object),
        "competing_risk_target_id": np.full(len(context), COMPETING_RISK_ID, dtype=object),
        "execution_policy_id": context.execution_policy_id.astype(str).to_numpy(),
        "cost_model_id": context.cost_model_id.astype(str).to_numpy(),
        "execution_geometry_id": context.execution_geometry_id.astype(str).to_numpy(),
        "source_decision_price": decision_price.astype(np.float32),
        "path_auxiliary_atr_fraction": atr_fraction.astype(np.float32),
    }
    supporting.update(auxiliary)
    supporting.update({column: competing[column].to_numpy() for column in competing.columns})
    # Regression targets retain right censoring for unreached MFE.  These
    # companion columns are conditional targets, hence unavailable outside
    # their event stratum rather than spuriously encoded as zero.
    reached = np.asarray(auxiliary["__meaningful_mfe_reached_12h__"], dtype=bool)
    peak = np.asarray(auxiliary["__peak_mfe_atr_12h__"], dtype=np.float32)
    time = np.asarray(auxiliary["__time_to_first_meaningful_mfe_hours_12h__"], dtype=np.float32)
    supporting["conditional_peak_mfe_atr_given_meaningful_mfe"] = np.where(reached, peak, np.nan).astype(np.float32)
    supporting["conditional_time_to_meaningful_mfe_hours"] = np.where(reached, time, np.nan).astype(np.float32)
    # Retention is meaningful only after a path first clears the economic
    # barrier.  It deliberately remains a target, never a meta feature.
    clean = np.asarray(competing["clean_economic_favorable_first"], dtype=bool)
    supporting["conditional_exact_h12_net_bps_given_clean_economic_first"] = np.where(
        clean, context.exact_h12_net_bps.to_numpy(dtype=float), np.nan
    ).astype(np.float64)
    supporting["conditional_exact_h12_net_positive_given_clean_economic_first"] = np.where(
        clean, (context.exact_h12_net_bps.to_numpy(dtype=float) > 0.0).astype(float), np.nan
    ).astype(np.float64)
    supportive = pd.DataFrame(supporting)
    if supportive.candidate_id.duplicated().any() or len(supportive) != len(primary):
        raise ValueError("primary/supportive label identity is not one-to-one")
    return primary, supportive


def _label_metadata(primary: pd.DataFrame, supportive: pd.DataFrame) -> pd.DataFrame:
    common = {
        "candidate_id", "symbol", "side", "decision_ts", "feature_cutoff_ts", "entry_ts",
        "label_end_ts", "label_available_ts", "primary_target_id", "execution_policy_id",
        "cost_model_id", "feature_set_id", "execution_geometry_id", "execution_geometry_key",
        "execution_geometry_source", "support_path_semantics_id", "competing_risk_target_id",
        "source_decision_price", "path_auxiliary_atr_fraction", "competing_risk_atr_fraction",
        "competing_risk_label_available_ts", "competing_risk_label_end_ts",
    }
    primary_details = {
        "execution_exact_h12_gross_bps": ("primary", "continuous", "bps", "authoritative frozen replay gross EV before one row cost"),
        "execution_exact_h12_cost_bps": ("primary_component", "continuous", "bps", "authoritative frozen row cost; already accounted exactly once in net"),
        "execution_exact_h12_net_bps": ("primary", "continuous", "bps", "authoritative frozen exact-H12 policy net EV"),
        "execution_exact_h12_net_positive": ("primary", "hard_binary", "indicator", "1 iff authoritative exact-H12 net EV > 0"),
        "execution_exact_h12_net_nonnegative": ("primary", "hard_binary", "indicator", "1 iff authoritative exact-H12 net EV >= 0"),
    }
    competing_details = {
        "competing_risk_class": ("supportive", "hard_multiclass", "class", "0 timeout, 1 adverse first, 2 clean economic favourable first"),
        "competing_risk_event": ("supportive", "hard_multiclass", "event", "row-cost-aware exact-1m first-event outcome"),
        "timeout": ("supportive", "hard_binary", "indicator", "no competing barrier reached by H12"),
        "adverse_first": ("supportive", "hard_binary", "indicator", "one ATR adverse barrier first, including same-minute conflict"),
        "clean_economic_favorable_first": ("supportive", "hard_binary", "indicator", "row-cost-aware favourable barrier before adverse barrier"),
        "timeout_soft_clean_economic_favorable_viability": ("conditional", "soft_probability", "probability", "timeout-only endpoint interpolation; NaN for hit outcomes"),
        "timeout_soft_adverse_viability": ("conditional", "soft_probability", "probability", "timeout-only endpoint interpolation; NaN for hit outcomes"),
        "timeout_soft_timeout_viability": ("conditional", "soft_probability", "probability", "timeout-only endpoint interpolation; NaN for hit outcomes"),
        "conditional_peak_mfe_atr_given_meaningful_mfe": ("conditional", "continuous", "ATR", "defined only if meaningful MFE is reached"),
        "conditional_time_to_meaningful_mfe_hours": ("conditional", "continuous", "hours", "defined only if meaningful MFE is reached"),
        "conditional_exact_h12_net_bps_given_clean_economic_first": ("conditional", "continuous", "bps", "defined only after clean economic first event"),
        "conditional_exact_h12_net_positive_given_clean_economic_first": ("conditional", "hard_binary", "indicator", "defined only after clean economic first event"),
    }
    rows: list[dict[str, Any]] = []
    for surface, frame in (("primary", primary), ("supportive", supportive)):
        for column in frame.columns:
            if column in common:
                continue
            role, kind, unit, description = primary_details.get(
                column,
                competing_details.get(
                    column,
                    (
                        "supportive",
                        "continuous" if pd.api.types.is_numeric_dtype(frame[column]) else "categorical",
                        "native",
                        "frozen auxiliary path target/support diagnostic" if column in set(TARGET_COLUMNS.values()) | set(ALL_SUPPORTIVE_LABEL_COLUMNS) else "exact-H12 supporting path diagnostic",
                    ),
                ),
            )
            condition = "unconditional"
            if column.startswith("timeout_soft_"):
                condition = "timeout == 1"
            elif column.startswith("conditional_peak") or column.startswith("conditional_time"):
                condition = "__meaningful_mfe_reached_12h__ == 1"
            elif column.startswith("conditional_exact"):
                condition = "clean_economic_favorable_first == 1"
            rows.append(
                {
                    "surface": surface,
                    "label_name": column,
                    "role": role,
                    "label_kind": kind,
                    "unit": unit,
                    "condition": condition,
                    "availability": "decision_ts + 12h only",
                    "model_input_allowed": False,
                    "description": description,
                    "path_semantics": PATH_SEMANTICS_ID if surface == "supportive" else "frozen policy replay ledger",
                }
            )
    return pd.DataFrame(rows).sort_values(["surface", "label_name"], kind="stable").reset_index(drop=True)


def _support_report(primary_path: Path, supportive_path: Path) -> pd.DataFrame:
    # Read only the contract labels reported below.  The optional full
    # auxiliary support surface is intentionally wide, and loading its unused
    # diagnostic columns solely to group the core labels would be needlessly
    # memory intensive on the historical population.
    primary_labels = (
        "execution_exact_h12_gross_bps",
        "execution_exact_h12_cost_bps",
        "execution_exact_h12_net_bps",
        "execution_exact_h12_net_positive",
        "execution_exact_h12_net_nonnegative",
    )
    support_labels = (
        *TARGET_COLUMNS.values(),
        "__meaningful_mfe_reached_12h__",
        "clean_economic_favorable_first",
        "adverse_first",
        "timeout",
        "conditional_peak_mfe_atr_given_meaningful_mfe",
        "conditional_time_to_meaningful_mfe_hours",
        "conditional_exact_h12_net_bps_given_clean_economic_first",
        "conditional_exact_h12_net_positive_given_clean_economic_first",
    )
    primary = pd.read_parquet(primary_path, columns=["decision_ts", "side", *primary_labels])
    supportive = pd.read_parquet(supportive_path, columns=["decision_ts", "side", *support_labels])
    for frame in (primary, supportive):
        frame["month"] = pd.to_datetime(frame.decision_ts, utc=True).dt.strftime("%Y-%m")
    groups = ["month", "side"]
    rows: list[dict[str, Any]] = []
    for surface, frame, labels in (("primary", primary, primary_labels), ("supportive", supportive, support_labels)):
        for (month, side), group in frame.groupby(groups, sort=True, observed=True):
            for label in labels:
                if label not in group:
                    continue
                values = pd.to_numeric(group[label], errors="coerce")
                finite = values.dropna()
                rows.append(
                    {
                        "surface": surface,
                        "month": month,
                        "side": side,
                        "label_name": label,
                        "rows": int(len(group)),
                        "non_null_rows": int(finite.size),
                        "mean": float(finite.mean()) if not finite.empty else np.nan,
                        "std": float(finite.std(ddof=0)) if not finite.empty else np.nan,
                        "p05": float(finite.quantile(0.05)) if not finite.empty else np.nan,
                        "p50": float(finite.quantile(0.50)) if not finite.empty else np.nan,
                        "p95": float(finite.quantile(0.95)) if not finite.empty else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["surface", "label_name", "month", "side"], kind="stable").reset_index(drop=True)


def materialize(
    *,
    alignment_path: Path,
    path_files: tuple[Path, ...],
    output: Path,
    batch_rows: int = 256,
    include_full_auxiliary_support: bool = True,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if int(batch_rows) <= 0:
        raise ValueError("batch_rows must be positive")
    context = _load_alignment(alignment_path)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    primary_writer: pq.ParquetWriter | None = None
    supportive_writer: pq.ParquetWriter | None = None
    seen: set[str] = set()
    try:
        primary_path = stage / "primary_labels.parquet"
        supportive_path = stage / "supportive_labels.parquet"
        for path_file in path_files:
            parquet = pq.ParquetFile(path_file)
            missing = sorted(set(PATH_COLUMNS).difference(parquet.schema.names))
            if missing:
                raise ValueError(f"path source {path_file} lacks {missing}")
            for batch in parquet.iter_batches(batch_size=int(batch_rows), columns=list(PATH_COLUMNS)):
                paths = batch.to_pandas()
                paths = paths.loc[paths.candidate_id.astype(str).isin(context.index)].reset_index(drop=True)
                if paths.empty:
                    continue
                primary, supportive = build_exact_h12_target_rows(
                    paths,
                    context,
                    include_full_auxiliary_support=include_full_auxiliary_support,
                )
                ids = primary.candidate_id.astype(str)
                if ids.duplicated().any() or any(candidate_id in seen for candidate_id in ids):
                    raise ValueError("duplicate exact path candidate across source files")
                seen.update(ids)
                primary_table = pa.Table.from_pandas(primary, preserve_index=False)
                supportive_table = pa.Table.from_pandas(supportive, preserve_index=False)
                if primary_writer is None:
                    primary_writer = pq.ParquetWriter(primary_path, primary_table.schema, compression="zstd")
                    supportive_writer = pq.ParquetWriter(supportive_path, supportive_table.schema, compression="zstd")
                else:
                    primary_table = primary_table.cast(primary_writer.schema)
                    assert supportive_writer is not None
                    supportive_table = supportive_table.cast(supportive_writer.schema)
                primary_writer.write_table(primary_table)
                assert supportive_writer is not None
                supportive_writer.write_table(supportive_table)
        if primary_writer is None or supportive_writer is None:
            raise ValueError("no exact aligned paths were materialised")
        primary_writer.close()
        primary_writer = None
        supportive_writer.close()
        supportive_writer = None
        missing = set(context.index).difference(seen)
        if missing:
            raise ValueError(f"exact 1m path coverage incomplete: missing {len(missing)} frozen candidates")
        primary_rows = pq.ParquetFile(primary_path).metadata.num_rows
        supportive_rows = pq.ParquetFile(supportive_path).metadata.num_rows
        if primary_rows != len(context) or supportive_rows != len(context):
            raise ValueError("materialised row counts do not equal the frozen alignment population")
        # The dictionary needs names and dtypes only; reading the full output
        # here would add an avoidable second pass over the historical corpus.
        dictionary = _label_metadata(
            next(pq.ParquetFile(primary_path).iter_batches(batch_size=1)).to_pandas(),
            next(pq.ParquetFile(supportive_path).iter_batches(batch_size=1)).to_pandas(),
        )
        dictionary_path = stage / "label_dictionary.parquet"
        dictionary.to_parquet(dictionary_path, index=False, compression="zstd")
        report = _support_report(primary_path, supportive_path)
        report_path = stage / "support_report.parquet"
        report.to_parquet(report_path, index=False, compression="zstd")
        contract = {
            "schema": SCHEMA,
            "population": "frozen historical candidate-conditioned alignment only",
            "horizon_minutes": HORIZON_MINUTES,
            "timing": {
                "feature_cutoff": "feature_cutoff_ts <= decision_ts",
                "entry": "entry_ts == decision_ts",
                "label_end": "decision_ts + 12h",
                "availability": "label_end_ts only",
            },
            "primary": {
                "target_id": PRIMARY_TARGET_ID,
                "execution_policy_id": EXECUTION_POLICY_ID,
                "cost_model_id": COST_MODEL_ID,
                "target": "execution_exact_h12_net_bps",
                "cost_accounting": "exact_h12_gross_bps - row_cost_bps == exact_h12_net_bps; cost is charged exactly once",
            },
            "supporting_path_labels": {
                "path_semantics_id": PATH_SEMANTICS_ID,
                "source": "complete immutable raw one-minute OHLC from the decision bar through H12",
                "auxiliary_schema": TARGET_SCHEMA,
                "auxiliary_semantics": "unadjusted raw decision-price path; no policy exit; frozen five-head target kernel",
                "competing_risk_id": COMPETING_RISK_ID,
                "competing_risk_semantics": "row-cost-aware favourable barrier max(1.5 ATR, 1.5%, row cost), one-ATR adverse barrier, same-minute conflict is adverse",
                "soft_labels": "timeout-only endpoint simplex; never substitutes a barrier-hit class",
            },
            "prohibitions": [
                "all primary and supporting labels are forbidden decision-time model inputs",
                "path competing-risk outcomes do not replace realised frozen-policy net EV",
                "no incomplete path, timing, policy, cost, geometry or identity mismatch is accepted",
            ],
        }
        contract_path = stage / "execution_target_contract.json"
        _write_json(contract_path, contract)
        manifest = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_RESEARCH_ONLY_TARGETS_NOT_MODEL_INPUTS",
            "rows": int(primary_rows),
            "inputs": {str(path): _sha256(path) for path in (alignment_path, *path_files)},
            "output": {
                "primary_labels.parquet": _sha256(primary_path),
                "supportive_labels.parquet": _sha256(supportive_path),
                "label_dictionary.parquet": _sha256(dictionary_path),
                "support_report.parquet": _sha256(report_path),
                "execution_target_contract.json": _sha256(contract_path),
            },
            "include_full_auxiliary_support": bool(include_full_auxiliary_support),
            "assertions": [
                "frozen exact-H12 alignment contract validated before path decoding",
                "each raw OHLC path contains exactly 720 decision-aligned one-minute bars",
                "candidate identity and side/symbol/timestamp agree exactly with alignment",
                "all labels resolve and become available at decision_ts + 12h",
                "all frozen candidates have exactly one complete exact path",
                "primary net EV is sourced from signed replay ledger, not recomputed from OHLC",
            ],
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        if primary_writer is not None:
            primary_writer.close()
        if supportive_writer is not None:
            supportive_writer.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--paths", type=Path, nargs="+", default=list(DEFAULT_PATHS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--without-full-auxiliary-support", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                alignment_path=args.alignment,
                path_files=tuple(args.paths),
                output=args.output,
                batch_rows=args.batch_rows,
                include_full_auxiliary_support=not args.without_full_auxiliary_support,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
