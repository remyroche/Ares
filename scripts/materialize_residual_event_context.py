#!/usr/bin/env python3
"""Materialize frozen residual-event context for compact candidate ledgers.

The four exported fields are the canonical, inference-time replacements for
the older ``meta_resid_arch_*`` aliases.  They are created by the packaged
residual-event state from pre-entry static-store inputs, the frozen raw meta
score, and the routed policy archetype.  Resolved outcomes are intentionally
never loaded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.fast_funcs import numba_rolling_zscore_fused
from extreme_price_movements.inference.live_residual_event_state import (
    load_live_residual_event_state_payload,
    residual_event_state_input_feature_columns,
)
from extreme_price_movements.residual_event_archetypes import OUTCOME_COLUMNS
from extreme_price_movements.static_feature_store import read_static_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_ROOT = ROOT / "data_perp/reports/frozen_meta_contract_history_20260720_v1"
DEFAULT_ARTIFACT_RUN_ID = "s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2"
DEFAULT_FEATURE_ROOT = ROOT / "data_perp/features/20260711_070000"
CANONICAL_CONTEXT_COLUMNS = (
    "resid_event_aegmm_local_support_log1p",
    "resid_event_aegmm_gmm_entropy",
    "resid_event_aegmm_expected_market_peer_surprise",
    "resid_event_aegmm_expected_ev_timestamp_neutral_surprise",
)
KEY_COLUMNS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_store_timestamp(feature_root: Path) -> pd.Timestamp:
    return pd.to_datetime(feature_root.name, format="%Y%m%d_%H%M%S", utc=True)


def _read_static_candidate_inputs(
    candidates: pd.DataFrame,
    *,
    feature_root: Path,
    columns: list[str],
    symbol_batch_size: int,
) -> pd.DataFrame:
    """Read only state inputs and candidate keys through the canonical store."""

    keys = candidates.loc[:, ["__ts__", "__symbol__"]].drop_duplicates().copy()
    keys["__ts__"] = pd.to_datetime(keys["__ts__"], utc=True, errors="coerce")
    keys = keys.dropna(subset=["__ts__", "__symbol__"])
    if keys.empty:
        return pd.DataFrame(columns=["__ts__", "__symbol__", *columns])
    key_index = pd.MultiIndex.from_frame(keys, names=["__ts__", "__symbol__"])
    start, end = keys["__ts__"].min(), keys["__ts__"].max()
    needs_carry_repair = "carry_adj_ret_self_z_10h" in columns
    read_start = start - pd.Timedelta(days=15) if needs_carry_repair else start
    read_columns = [*columns]
    if needs_carry_repair:
        read_columns.extend(["ret10h", "fund_rate"])
    read_columns = list(dict.fromkeys(read_columns))
    symbols = sorted(keys["__symbol__"].astype(str).unique())
    rows: list[pd.DataFrame] = []
    data_root = feature_root.parents[1]
    for offset in range(0, len(symbols), max(1, int(symbol_batch_size))):
        batch = symbols[offset : offset + max(1, int(symbol_batch_size))]
        loaded = read_static_features(
            feature_store_ts=_feature_store_timestamp(feature_root),
            data_root=data_root,
            feature_keys=read_columns,
            symbols=batch,
            start_ts=read_start,
            end_ts=end,
            output_layout="panels",
        )
        if loaded is None:
            continue
        for symbol in batch:
            frame = loaded.symbol_frame(symbol, keys=read_columns)
            if frame.empty:
                continue
            frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
            frame = frame.loc[frame.index.notna() & frame.index.to_series().between(read_start, end)]
            if frame.empty:
                continue
            frame["__ts__"] = frame.index
            frame["__symbol__"] = symbol
            frame = frame.reset_index(drop=True)
            # Some older partitions predate the registered carry feature. Keep
            # its causal ingredients here so the exact 336-hour transform can
            # be rebuilt below; every other missing state input remains NaN,
            # matching the frozen transform's trained missing-value handling.
            rows.append(frame.reindex(columns=["__ts__", "__symbol__", *read_columns]))
    if not rows:
        return pd.DataFrame(columns=["__ts__", "__symbol__", *columns])
    result = pd.concat(rows, ignore_index=True, copy=False)
    result = result.drop_duplicates(["__ts__", "__symbol__"], keep="last")
    if needs_carry_repair:
        current = pd.to_numeric(
            result.get("carry_adj_ret_self_z_10h"), errors="coerce"
        )
        if current.notna().mean() < 1.0:
            ret = result.pivot(index="__ts__", columns="__symbol__", values="ret10h")
            funding = result.pivot(index="__ts__", columns="__symbol__", values="fund_rate")
            panel = numba_rolling_zscore_fused(
                ret.astype(np.float32, copy=False)
                - funding.reindex_like(ret).astype(np.float32, copy=False) * np.float32(10.0 / 8.0),
                14 * 24,
            ).clip(-6.0, 6.0)
            lookup_index = pd.MultiIndex.from_frame(
                result.loc[:, ["__ts__", "__symbol__"]], names=["__ts__", "__symbol__"]
            )
            repaired = panel.rename_axis(index="__ts__", columns="__symbol__").stack(
                dropna=False
            ).reindex(lookup_index).to_numpy(dtype=np.float32, copy=False)
            result["carry_adj_ret_self_z_10h"] = current.where(
                current.notna(), pd.Series(repaired, index=result.index)
            )
    result_index = pd.MultiIndex.from_frame(
        result.loc[:, ["__ts__", "__symbol__"]], names=key_index.names
    )
    result = result.loc[result_index.isin(key_index)].copy()
    return result.reindex(columns=["__ts__", "__symbol__", *columns])


def materialize_month(
    candidates: pd.DataFrame,
    *,
    payload: dict,
    feature_root: Path,
    symbol_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the exact frozen local residual-event transform side by side."""

    required = {"__ts__", "__symbol__", "side_name", "archetype_policy_key", "score_meta_base_soft_label"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Candidate ledger is missing residual-event routing columns: {missing}")
    forbidden = sorted(OUTCOME_COLUMNS.intersection(candidates.columns))
    if forbidden:
        raise ValueError(f"Candidate ledger unexpectedly contains outcomes: {forbidden}")
    state = payload["state"]
    state_inputs = sorted(residual_event_state_input_feature_columns(payload))
    static = _read_static_candidate_inputs(
        candidates,
        feature_root=feature_root,
        columns=state_inputs,
        symbol_batch_size=symbol_batch_size,
    )
    work = candidates.loc[:, list(KEY_COLUMNS) + ["score_meta_base_soft_label"]].copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="coerce")
    work["__symbol__"] = work["__symbol__"].astype(str)
    work["side_name"] = work["side_name"].astype(str).str.lower()
    work["archetype_policy_key"] = work["archetype_policy_key"].astype(str)
    work = work.merge(static, on=["__ts__", "__symbol__"], how="left", validate="m:1")
    work[state.config.score_col] = pd.to_numeric(
        work["score_meta_base_soft_label"], errors="coerce"
    ).astype(np.float32)
    transformed_parts: list[pd.DataFrame] = []
    for side, local in work.groupby("side_name", observed=True, sort=False):
        transformed = state.transform_oos(local)
        transformed_parts.append(
            transformed.loc[:, list(CANONICAL_CONTEXT_COLUMNS)].set_axis(local.index)
        )
    transformed = pd.concat(transformed_parts, axis=0).reindex(work.index)
    output = pd.concat([work.loc[:, list(KEY_COLUMNS)], transformed], axis=1)
    output.loc[:, list(CANONICAL_CONTEXT_COLUMNS)] = output.loc[:, list(CANONICAL_CONTEXT_COLUMNS)].astype(np.float32)
    finite = np.isfinite(output.loc[:, list(CANONICAL_CONTEXT_COLUMNS)].to_numpy(dtype=np.float32))
    coverage = {
        name: float(np.isfinite(pd.to_numeric(output[name], errors="coerce")).mean())
        for name in CANONICAL_CONTEXT_COLUMNS
    }
    diagnostics = {
        "candidate_rows": int(len(candidates)),
        "static_rows": int(len(static)),
        "static_join_rate": float(work[state_inputs].notna().any(axis=1).mean()) if state_inputs else 1.0,
        "output_finite_rate": float(finite.mean()) if finite.size else 0.0,
        "output_coverage": coverage,
        "required_static_feature_count": len(state_inputs),
    }
    if not finite.all():
        raise RuntimeError(f"Residual-event context emitted non-finite values: {diagnostics}")
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--artifact-run-id", default=DEFAULT_ARTIFACT_RUN_ID)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", nargs="*", default=[])
    parser.add_argument("--symbol-batch-size", type=int, default=32)
    args = parser.parse_args()

    payload = load_live_residual_event_state_payload("data_perp", str(args.artifact_run_id))
    if not payload:
        raise FileNotFoundError(f"No frozen residual-event state for run {args.artifact_run_id}")
    paths = sorted(args.candidate_root.glob("monthly/*/candidates.parquet"))
    selected_months = {str(month) for month in args.months}
    if selected_months:
        paths = [path for path in paths if path.parent.name in selected_months]
    if not paths:
        raise FileNotFoundError("No candidate months selected")
    args.output.mkdir(parents=True, exist_ok=True)
    processed_months = {path.parent.name for path in paths}
    month_rows: list[dict[str, object]] = []
    for path in paths:
        candidates = pd.read_parquet(path)
        output, diagnostics = materialize_month(
            candidates,
            payload=payload,
            feature_root=args.feature_root,
            symbol_batch_size=args.symbol_batch_size,
        )
        target = args.output / "monthly" / path.parent.name / "residual_event_context.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(target, index=False)
        month_rows.append({"month": path.parent.name, **diagnostics, "path": str(target)})
        print(f"materialized {path.parent.name}: rows={len(output)} finite={diagnostics['output_finite_rate']:.3f}", flush=True)
    # A resumed run need not re-read every historical static shard merely to
    # reconstruct its manifest. Keep existing sidecars in the contract, but
    # label their raw-store row diagnostic as unavailable rather than inventing
    # it after the fact.
    for path in sorted(args.candidate_root.glob("monthly/*/candidates.parquet")):
        month = path.parent.name
        sidecar = args.output / "monthly" / month / "residual_event_context.parquet"
        if month in processed_months or not sidecar.exists():
            continue
        existing = pd.read_parquet(sidecar, columns=list(CANONICAL_CONTEXT_COLUMNS))
        coverage = {
            name: float(np.isfinite(pd.to_numeric(existing[name], errors="coerce")).mean())
            for name in CANONICAL_CONTEXT_COLUMNS
        }
        month_rows.append({
            "month": month,
            "candidate_rows": int(len(pd.read_parquet(path, columns=["__ts__"]))),
            "static_rows": None,
            "static_join_rate": None,
            "output_finite_rate": float(np.isfinite(existing.to_numpy(dtype=np.float32)).mean()),
            "output_coverage": coverage,
            "required_static_feature_count": len(residual_event_state_input_feature_columns(payload)),
            "path": str(sidecar),
            "resumed_existing_sidecar": True,
        })
    month_rows.sort(key=lambda row: str(row["month"]))
    state_path = Path(str(payload["state_path"]))
    manifest = {
        "schema": "frozen_residual_event_context_v1",
        "evidence_scope": "observable_context_only_not_outcome_scoring",
        "candidate_root": str(args.candidate_root),
        "feature_root": str(args.feature_root),
        "static_source_api": "static_feature_store.read_static_features",
        "artifact_run_id": str(args.artifact_run_id),
        "frozen_state_path": str(state_path),
        "frozen_state_sha256": _sha256(state_path),
        "context_columns": list(CANONICAL_CONTEXT_COLUMNS),
        "outcomes_loaded": False,
        "months": month_rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
