#!/usr/bin/env python3
"""Build a causal residual-target meta source from a frozen base OOS ledger.

The resulting handoff retains the top-30 base candidate universe and adds:
1. base OOS reliability/IC/hit-surprise fields with a resolved-label embargo;
2. frozen side-local meta AE/GMM state outputs fit only before ``--fit-end``;
3. a scored ledger in the schema consumed by the V9 residual-expert trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_oos_reliability import (  # noqa: E402
    CausalBaseReliabilityBuilder,
    RELIABILITY_FEATURE_COLUMNS,
)
from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    ae_gmm_state_manifest,
    fit_ae_gmm_state,
    load_ae_gmm_state_artifact,
    save_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)


KEYS = ("__ts__", "__symbol__", "side_name")
OUTCOME_COLUMNS = (
    "__first_touch_capture_net__",
    "__first_touch_round_trip_cost__",
    "__first_touch_policy_soft__",
    "__first_touch_hit__",
    "__first_touch_mae_norm__",
    "__first_touch_full_path_mae_norm__",
    "__first_touch_timeout__",
    "__mfe_1r_before_mae_1r__",
    "__mae_1r_before_mfe_1r__",
)


def _quote(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _safe_numeric_columns(schema: pa.Schema) -> list[str]:
    forbidden = (
        "target", "outcome", "realized", "first_touch", "mfe", "mae", "timeout",
        "stop", "path", "profit", "pnl", "u_policy", "y_", "label", "action_",
        "expected_delta", "holdout_",
    )
    allowed_tokens = (
        "score", "rank", "margin", "gmm", "cluster", "mahal", "ae_", "dae_", "latent",
        "ood", "drift", "regime", "calendar", "structural", "reconstruction", "base_",
        "source", "confidence", "policy_tp", "policy_sl", "policy_trail",
    )
    values: list[str] = []
    for field in schema:
        name = str(field.name)
        lower = name.lower()
        if not pa.types.is_integer(field.type) and not pa.types.is_floating(field.type):
            continue
        if any(token in lower for token in forbidden):
            continue
        if any(token in lower for token in allowed_tokens):
            values.append(name)
    return values


def _time_spread(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    if len(frame) <= rows:
        return frame
    frame = frame.sort_values("__ts__", kind="stable").reset_index(drop=True)
    parts = np.array_split(np.arange(len(frame)), 3)
    counts = [rows // 3 + int(index < rows % 3) for index in range(3)]
    take = np.concatenate([
        part[np.linspace(0, len(part) - 1, count, dtype=np.int64)]
        for part, count in zip(parts, counts) if len(part) and count
    ])
    return frame.iloc[np.sort(take)].reset_index(drop=True)


def _outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, list(KEYS)].copy()
    capture = pd.to_numeric(frame["__first_touch_capture_net__"], errors="coerce").fillna(0.0)
    cost = pd.to_numeric(frame["__first_touch_round_trip_cost__"], errors="coerce").fillna(0.01)
    first_mae = pd.to_numeric(frame["__first_touch_mae_norm__"], errors="coerce")
    full_mae = pd.to_numeric(frame["__first_touch_full_path_mae_norm__"], errors="coerce")
    timeout = pd.to_numeric(frame["__first_touch_timeout__"], errors="coerce").fillna(0.0).gt(0.5)
    mfe_before = pd.to_numeric(frame["__mfe_1r_before_mae_1r__"], errors="coerce").fillna(0.0)
    mae_before = pd.to_numeric(frame["__mae_1r_before_mfe_1r__"], errors="coerce").fillna(0.0)
    # capture_net already includes the actual 1% round-trip cost in this
    # corrected-path ledger. Rebuild gross only to apply the same 1% once.
    gross = capture + cost
    out["ev_after_1pct"] = (gross - 0.01).astype(np.float32)
    out["exec_margin"] = out["ev_after_1pct"]
    out["first_touch_bad_mae_1r"] = first_mae.ge(1.0).astype(np.float32)
    out["full_path_bad_mae_1r"] = full_mae.ge(1.0).astype(np.float32)
    out["timeout"] = timeout.astype(np.float32)
    out["clean_exec"] = (
        out["exec_margin"].gt(0.0)
        & out["first_touch_bad_mae_1r"].lt(0.5)
        & out["timeout"].lt(0.5)
        & mfe_before.gt(0.5)
    ).astype(np.float32)
    out["dirty_positive"] = (
        out["exec_margin"].gt(0.0)
        & (out["first_touch_bad_mae_1r"].gt(0.5) | out["full_path_bad_mae_1r"].gt(0.5) | out["timeout"].gt(0.5) | mae_before.gt(0.5))
    ).astype(np.float32)
    return out


def _stream_reliability(input_path: Path, output_path: Path) -> int:
    """Derive causal features in day chunks, avoiding a 1.3m-row pandas copy."""
    aggregate = duckdb.sql(
        f"""SELECT
            avg(CAST(__first_touch_policy_soft__ AS DOUBLE)) AS soft,
            avg(CAST(__first_touch_hit__ AS DOUBLE)) AS hit,
            avg(CAST(__first_touch_capture_net__ AS DOUBLE)) AS ev
          FROM read_parquet('{str(input_path.resolve()).replace("'", "''")}')"""
    ).df().iloc[0]
    builder = CausalBaseReliabilityBuilder((float(aggregate.soft), float(aggregate.hit), float(aggregate.ev)))
    parquet = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    carry = pd.DataFrame()
    rows = 0
    for batch in parquet.iter_batches(batch_size=80_000):
        current = batch.to_pandas()
        current["__ts__"] = pd.to_datetime(current["__ts__"], utc=True, errors="coerce")
        if not carry.empty:
            current = pd.concat([carry, current], ignore_index=True, copy=False)
        current["__signal_day__"] = current["__ts__"].dt.normalize()
        last_day = current["__signal_day__"].iloc[-1]
        complete = current.loc[current["__signal_day__"].lt(last_day)].drop(columns="__signal_day__")
        carry = current.loc[current["__signal_day__"].eq(last_day)].drop(columns="__signal_day__").reset_index(drop=True)
        for _, day_frame in complete.groupby(complete["__ts__"].dt.normalize(), observed=True, sort=True):
            payload = builder.transform_day(day_frame.reset_index(drop=True))
            table = pa.Table.from_pandas(payload, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(payload)
        print(f"[reliability] rows={rows:,} carry={len(carry):,}", flush=True)
    if not carry.empty:
        for _, day_frame in carry.groupby(carry["__ts__"].dt.normalize(), observed=True, sort=True):
            payload = builder.transform_day(day_frame.reset_index(drop=True))
            table = pa.Table.from_pandas(payload, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(payload)
    if writer is not None:
        writer.close()
    return rows


def _fit_and_materialize_meta_aegmm(
    handoff: Path,
    *,
    out_dir: Path,
    fit_end: pd.Timestamp,
    ae_rows: int,
    gmm_rows: int,
    row_groups: tuple[int, ...] | None = None,
) -> tuple[Path, dict[str, object]]:
    schema = pq.read_schema(handoff)
    inputs = _safe_numeric_columns(schema)
    reliability = [column for column in RELIABILITY_FEATURE_COLUMNS if column in schema.names]
    inputs = list(dict.fromkeys([*inputs, *reliability]))
    # The score and current frozen base-state outputs are always part of the
    # meta representation, even when a schema dtype changed in an older shard.
    inputs = [column for column in inputs if column in schema.names]
    reference_columns = [*KEYS, *inputs]
    reference_rows = max(int(ae_rows), int(gmm_rows))
    per_band = max(1, int(np.ceil(reference_rows / 3.0)))
    query = f"""
        WITH source AS (
            SELECT {', '.join(_quote(column) for column in reference_columns)}
            FROM read_parquet('{str(handoff.resolve()).replace("'", "''")}')
            WHERE __ts__ < TIMESTAMPTZ '{fit_end:%Y-%m-%d %H:%M:%S}+00'
        ), bands AS (
            SELECT *, ntile(3) OVER (
                PARTITION BY lower(side_name) ORDER BY __ts__, __symbol__
            ) AS __band
            FROM source
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY lower(side_name), __band ORDER BY __ts__, __symbol__
            ) AS __row
            FROM bands
        )
        SELECT * EXCLUDE (__band, __row)
        FROM ranked
        WHERE __row <= {per_band}
        ORDER BY __ts__, __symbol__, side_name
    """
    reference = duckdb.sql(query).df()
    reference["__ts__"] = pd.to_datetime(reference["__ts__"], utc=True)
    states: dict[str, dict] = {}
    manifests: dict[str, object] = {"input_columns": inputs, "fit_end_exclusive": fit_end.isoformat()}
    for side in ("long", "short"):
        local = reference.loc[reference["side_name"].astype(str).str.lower().eq(side)].copy()
        if len(local) < 1_000:
            raise RuntimeError(f"meta AE/GMM {side}: insufficient reference rows={len(local)}")
        state_path = out_dir / f"meta_{side}_aegmm_state.pkl"
        if state_path.exists():
            state = load_ae_gmm_state_artifact(state_path)
            if list(state.get("feature_columns", [])) != inputs:
                raise RuntimeError(f"Existing {side} meta AE/GMM package uses a different input contract")
            states[side] = state
            manifests[side] = {**ae_gmm_state_manifest(state), "state_path": str(state_path), "reference_rows": int(len(local)), "reused_frozen_state": True}
            print(f"[meta-aegmm] reused frozen side={side}", flush=True)
            continue
        numeric = local.loc[:, inputs].apply(pd.to_numeric, errors="coerce")
        fill = numeric.median(axis=0).fillna(0.0).astype(np.float32)
        numeric = numeric.fillna(fill).clip(-1e6, 1e6).astype(np.float32)
        print(
            f"[meta-aegmm] fit side={side} reference_rows={len(numeric):,} inputs={len(inputs)}",
            flush=True,
        )
        # This is a frozen contextual state block, not a second architecture
        # search.  One compact b16 DAE fit avoids the 24-model generic DAE
        # grid, which otherwise dominates memory without improving a fixed
        # state transform comparison.
        dae_env = {
            "EPM_DAE_BOTTLENECKS": "16",
            "EPM_DAE_WIDTHS": "small",
            "EPM_DAE_NOISE_LEVELS": "0.10",
            "EPM_DAE_MASK_FRACTIONS": "0.0",
            "EPM_DAE_L2_ALPHAS": "0.0001",
        }
        prior_env = {key: os.environ.get(key) for key in dae_env}
        os.environ.update(dae_env)
        try:
            state = fit_ae_gmm_state(
                numeric,
                timestamps=local["__ts__"],
                random_state=20260722 + (0 if side == "long" else 1),
                max_train_rows=min(ae_rows, len(numeric)),
                gmm_max_train_rows=min(gmm_rows, len(numeric)),
                ae_max_iter=20,
                cluster_candidates=(4,),
                reg_covar_candidates=(0.003,),
                covariance_type_candidates=("diag",),
                smooth_lambda_candidates=(0.0,),
                outcome_free=True,
                temporal_feature_contract="row_independent_v1",
            )
        finally:
            for key, value in prior_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if not bool(state.get("enabled", False)):
            raise RuntimeError(f"meta AE/GMM {side} disabled: {state.get('reason')}")
        # Preserve the train fill vector as an explicit transform input rule.
        state["cycle_input_fill_values"] = {column: float(fill[column]) for column in inputs}
        state["meta_reliability_input_columns"] = list(RELIABILITY_FEATURE_COLUMNS)
        saved = save_ae_gmm_state_artifact(state, state_path)
        states[side] = state
        manifests[side] = {**ae_gmm_state_manifest(state), **saved, "reference_rows": int(len(local))}
        print(
            f"[meta-aegmm] fitted side={side} components={state.get('gmm_n_components')}",
            flush=True,
        )

    if row_groups is None:
        sidecar = out_dir / "meta_aegmm_state_features.parquet"
        if sidecar.exists():
            sidecar.unlink()
    else:
        parts_dir = out_dir / "meta_aegmm_state_feature_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        token = "_".join(f"{int(value):03d}" for value in row_groups)
        sidecar = parts_dir / f"part_{token}.parquet"
        if sidecar.exists():
            sidecar.unlink()
    writer: pq.ParquetWriter | None = None
    parquet = pq.ParquetFile(handoff)
    columns = [*KEYS, *inputs]
    # The two side blocks together are wide.  Ten thousand rows keeps peak
    # pandas/Arrow memory bounded on M-series machines while preserving a
    # sequential, deterministic transform.
    for batch in parquet.iter_batches(batch_size=10_000, columns=columns, row_groups=row_groups):
        part = batch.to_pandas()
        key = part.loc[:, list(KEYS)].copy()
        blocks = []
        for side, state in states.items():
            mask = part["side_name"].astype(str).str.lower().eq(side)
            prefix = f"meta_{side}_aegmm_"
            block = pd.DataFrame(0.0, index=part.index, columns=transform_ae_gmm_features(
                pd.DataFrame(np.zeros((0, len(inputs)), dtype=np.float32), columns=inputs), state, prefix=prefix
            ).columns, dtype=np.float32)
            if mask.any():
                values = part.loc[mask, inputs].apply(pd.to_numeric, errors="coerce")
                fills = state["cycle_input_fill_values"]
                values = values.fillna(pd.Series(fills)).clip(-1e6, 1e6).astype(np.float32)
                transformed = transform_ae_gmm_features(values, state, index=values.index, prefix=prefix)
                block.loc[values.index, transformed.columns] = transformed.to_numpy(dtype=np.float32)
            block[f"meta_{side}_aegmm_active"] = mask.to_numpy(dtype=np.float32)
            blocks.append(block)
        payload = pd.concat([key.reset_index(drop=True), *[block.reset_index(drop=True) for block in blocks]], axis=1, copy=False)
        table = pa.Table.from_pandas(payload, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(sidecar, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"[meta-aegmm] materialized rows={len(payload):,}", flush=True)
    if writer is not None:
        writer.close()
    (out_dir / "meta_aegmm_manifest.json").write_text(json.dumps(manifests, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return sidecar, manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ledger", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--meta-aegmm-fit-end", default="2026-03-01")
    parser.add_argument("--meta-aegmm-ae-rows", type=int, default=10_000)
    parser.add_argument("--meta-aegmm-gmm-rows", type=int, default=20_000)
    parser.add_argument(
        "--meta-aegmm-row-group",
        type=int,
        action="append",
        default=None,
        help="Materialize only named row group(s) from an existing reliability handoff; internal resume worker use.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit_end = pd.Timestamp(args.meta_aegmm_fit_end, tz="UTC")
    base = str(args.base_ledger.resolve()).replace("'", "''")
    handoff = str(args.handoff.resolve()).replace("'", "''")
    reliability_input = args.out_dir / "base_oos_reliability_input.parquet"
    handoff_columns = [
        *KEYS,
        "__label_path_end_ts__",
        "__archetype_policy_key__",
        "score",
        "base_rank_pct_by_timestamp_side",
    ]
    query = f"""
        COPY (
            SELECT {', '.join('h.' + _quote(column) for column in handoff_columns)},
                   {', '.join('b.' + _quote(column) for column in OUTCOME_COLUMNS)}
            FROM read_parquet('{handoff}') h
            INNER JOIN read_parquet('{base}') b
              ON h.__ts__ = b.__ts__
             AND h.__symbol__ = b.__symbol__
             AND lower(h.side_name) = lower(b.side_name)
            ORDER BY h.__ts__, h.__symbol__, h.side_name
        ) TO '{str(reliability_input.resolve()).replace("'", "''")}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    duckdb.sql(query)
    reliability_path = args.out_dir / "base_oos_reliability_features.parquet"
    reliability_rows = _stream_reliability(reliability_input, reliability_path)
    # The outcome table is narrow enough to construct separately from the
    # source parquet after the streaming reliability pass completes.
    outcome_input = pd.read_parquet(reliability_input, columns=[*KEYS, *OUTCOME_COLUMNS])
    outcomes = _outcomes(outcome_input)
    scored_path = args.out_dir / "meta_residual_scored_ledger.parquet"
    outcomes.to_parquet(scored_path, index=False, compression="zstd")
    augmented = args.out_dir / "train_meta_regime_handoff_with_base_reliability.parquet"
    reliability_sql = str(reliability_path.resolve()).replace("'", "''")
    duckdb.sql(f"""
        COPY (
            SELECT h.*, {', '.join('r.' + _quote(column) for column in RELIABILITY_FEATURE_COLUMNS)}
            FROM read_parquet('{handoff}') h
            INNER JOIN read_parquet('{reliability_sql}') r
              ON h.__ts__ = r.__ts__
             AND h.__symbol__ = r.__symbol__
             AND lower(h.side_name) = lower(r.side_name)
            ORDER BY h.__ts__, h.__symbol__, h.side_name
        ) TO '{str(augmented.resolve()).replace("'", "''")}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    sidecar, state_manifest = _fit_and_materialize_meta_aegmm(
        augmented,
        out_dir=args.out_dir,
        fit_end=fit_end,
        ae_rows=int(args.meta_aegmm_ae_rows),
        gmm_rows=int(args.meta_aegmm_gmm_rows),
    )
    final_handoff = args.out_dir / "train_meta_regime_handoff_with_base_reliability_meta_aegmm.parquet"
    duckdb.sql(f"""
        COPY (
            SELECT h.*, s.* EXCLUDE (__ts__, __symbol__, side_name)
            FROM read_parquet('{str(augmented.resolve()).replace("'", "''")}') h
            INNER JOIN read_parquet('{str(sidecar.resolve()).replace("'", "''")}') s
              ON h.__ts__ = s.__ts__
             AND h.__symbol__ = s.__symbol__
             AND lower(h.side_name) = lower(s.side_name)
            ORDER BY h.__ts__, h.__symbol__, h.side_name
        ) TO '{str(final_handoff.resolve()).replace("'", "''")}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    manifest = {
        "schema": "meta_base_reliability_source_v1",
        "base_ledger": str(args.base_ledger),
        "source_handoff": str(args.handoff),
        "final_handoff": str(final_handoff),
        "scored_ledger": str(scored_path),
        "reliability_features": list(RELIABILITY_FEATURE_COLUMNS),
        "reliability_contract": "full UTC calendar-day resolved-outcome embargo; OOS rows use only outcomes with label_path_end before the signal day",
        "target_contract": "V9 residual net EV filter: ev_after_1pct minus train-only side/archetype hierarchical expected EV",
        "meta_aegmm_fit_end_exclusive": fit_end.isoformat(),
        "meta_aegmm": state_manifest,
        "row_counts": {
            "reliability_input": int(len(outcomes)),
            "reliability_output": int(reliability_rows),
            "scored_ledger": int(len(outcomes)),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"handoff": str(final_handoff), "scored_ledger": str(scored_path), "rows": len(outcomes)}, sort_keys=True))


if __name__ == "__main__":
    main()
