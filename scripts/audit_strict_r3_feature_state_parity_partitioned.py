#!/usr/bin/env python3
"""Compare stateful strict-R3 features with bounded full-history partitions.

The canonical full graph is too large to coexist with the desktop runtime on
the live host.  This auditor evaluates disjoint frozen-field groups in fresh
processes, merges them by immutable candidate identity, applies the same final
causal repairs, and then compares all deployed fields with the state output.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from scripts.materialize_strict_r3_forward_features import (  # noqa: E402
    _IDENTITY_COLUMNS,
    _refresh_feature_coverage,
    _repair_cross_asset_state_fields,
)
from scripts.materialize_strict_r3_forward_features_incremental_v13 import (  # noqa: E402
    CROSS_SECTIONAL_STATE_FIELDS,
    EXACT_LONG_MEMORY_FIELDS,
    _compute_contract_features,
    _frozen_spectral_parent_keys,
    _latest_matrix,
)
from scripts.materialize_strict_r3_forward_features_incremental_v13 import (  # noqa: E402
    PRICE_MEMORY_STATE_FIELDS,
    RESIDUAL_SURPRISE_STATE_FIELDS,
)
from extreme_price_movements.inference.orderbook_feature_state import (  # noqa: E402
    ORDERBOOK_OUTPUTS,
)
from extreme_price_movements.inference.strict_r3_final14_state import (  # noqa: E402
    FINAL14_FIELD_ORDER,
)
from scripts.run_tp6_sl4_exact170_canonical_consensus import _load_contract  # noqa: E402
from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    FROZEN_GENERATION_DEPENDENCIES,
)
from scripts.update_strict_r3_feature_panel_state import STATE_SCHEMA  # noqa: E402


# This bounded auditor deliberately turns off every append-state operator in
# ``_reference_cfg`` so its partitions can fit on the live host.  Those fields
# cannot be called divergent when compared with the stateful producer: that
# would only measure the audit harness's intentionally different state
# contract.  Keep the classification explicit and fail promotion as
# *inconclusive*, rather than reporting a misleading feature mismatch.
_STATEFUL_REFERENCE_EXCLUSIONS = frozenset().union(
    EXACT_LONG_MEMORY_FIELDS,
    PRICE_MEMORY_STATE_FIELDS,
    RESIDUAL_SURPRISE_STATE_FIELDS,
    CROSS_SECTIONAL_STATE_FIELDS,
    FINAL14_FIELD_ORDER,
    ORDERBOOK_OUTPUTS,
    # These three are final, complete-universe repairs applied after the
    # request-driven generic graph.  The bounded reference intentionally
    # disables the streaming state and reopens a restricted source panel, so
    # it cannot recreate their ordered 15-minute/book provenance.  Comparing
    # it with the persisted live values would measure the audit harness, not
    # a deployed feature divergence.
    {
        "median_alt_minus_btc",
        "cross_asset_corr_1h",
        "q_lower_tail__xasset_mkt_spread_bps",
    },
)


def _reference_cfg(*, spectral_contract: Path, scratch: Path) -> dict:
    cfg = dict(CFG)
    cfg.pop("training_live_parity_contract", None)
    cfg.update(
        {
            "atr_n": 14,
            "use_perps": True,
            "feature_portability_mode": "off",
            "feature_portability_strict": False,
            "live_raw_feature_compute_preserve_portability_mode": True,
            "enable_orderbook_features": False,
            "enable_orderbook_wall_features": False,
            "live_materialize_orderbook_model_features": False,
            "live_lgbm_mask_feature_fast_path_enabled": False,
            "live_feature_cache_namespace": "strict_r3_partitioned_exact_reference",
            "live_feature_snapshot_cache_dir": str(scratch / "feature_cache"),
            "live_feature_snapshot_cache_enabled": False,
            "live_feature_rolling_cache_enabled": False,
            "live_feature_latest_row_incremental_enabled": False,
            "live_feature_return_latest_only": False,
            "live_feature_persist_after_scoring": False,
            "live_model_feature_tail_recompute_enabled": False,
            "live_raw_rolling_state_enabled": False,
            "live_causal_transform_state_enabled": False,
            "live_market_spectral_state_path": str(spectral_contract),
            "live_market_spectral_history_state_enabled": False,
            "live_derived_history_state_enabled": False,
            "live_fixed_ffd_state_enabled": False,
            "static_feature_store_write_enabled": False,
            "run_id": "strict_r3_partitioned_exact_reference",
        }
    )
    return cfg


def _worker(args: argparse.Namespace) -> None:
    state = joblib.load(args.panel_state)
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("unsupported source-panel state")
    candidates = pd.read_parquet(args.candidates)
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
    candidates["__decision_ts__"] = pd.to_datetime(
        candidates["__decision_ts__"], utc=True
    )
    fields = json.loads(args.worker_fields)
    spectral_requested = any(str(field).startswith("state_spectral_") for field in fields)
    # The frozen spectral contract names selected *summary columns*, while
    # their primitive source features are generated through a request-driven
    # graph.  Asking only for the selected bases is not a valid reference
    # computation: some of those primitives have cross-sectional ancestors
    # which are present only when the complete declared source pool is in the
    # workset.  Live materialisation always carries that pool.  The auditor
    # must mirror it, otherwise it can report a false missing-source failure
    # before it ever compares a deployed field.
    # The feature engine may request a spectral output indirectly while
    # repairing another composite in the same workset.  Therefore this must
    # be loaded for *every* partition, not only the partition containing a
    # visibly ``state_spectral_*`` output.  Otherwise a non-spectral chunk
    # can attempt the frozen geometry with an incomplete primitive source
    # workset and fail before comparison.
    payload = json.loads(args.spectral_contract.read_text())
    if payload.get("schema") != "strict_r3_market_spectral_source_state_v1":
        raise ValueError("unsupported market-spectral source state")
    spectral_parents = _frozen_spectral_parent_keys(args.spectral_contract)
    spectral_source_keys = [
        str(value) for value in payload.get("source_keys", ()) if str(value)
    ]
    # Partitioning is a memory strategy only. Every worker must retain the
    # same generation-dependency closure as the canonical 120-field call;
    # otherwise request-sensitive parent/composite ordering creates a false
    # "exact" reference for cross-sectional and order-book fields.
    compute_fields = list(dict.fromkeys([
        *fields,
        *FROZEN_GENERATION_DEPENDENCIES,
        *spectral_parents,
        *spectral_source_keys,
    ]))
    features, _, _ = _compute_contract_features(
        state["panel"],
        symbols=list(state["symbols"]),
        requested=compute_fields,
        cfg=_reference_cfg(
            spectral_contract=args.spectral_contract,
            scratch=args.worker_out.parent / (args.worker_out.stem + "_scratch"),
        ),
    )
    out = _latest_matrix(features, candidates=candidates, requested=fields)
    args.worker_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.worker_out, index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--panel-state", type=Path, required=True)
    parser.add_argument("--incremental-features", type=Path, required=True)
    parser.add_argument("--spectral-contract", type=Path, required=True)
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Run at most this many memory-bounded exact partitions concurrently.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--worker-fields")
    parser.add_argument("--worker-out", type=Path)
    parser.add_argument(
        "--resume-partitions",
        action="store_true",
        help="Reuse already completed immutable partition files after an interrupted audit.",
    )
    args = parser.parse_args()
    if args.worker_fields is not None:
        if args.worker_out is None:
            raise ValueError("worker mode requires --worker-out")
        _worker(args)
        return
    if args.out_dir.exists() and not args.resume_partitions:
        raise FileExistsError(f"immutable parity output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=args.resume_partitions)
    candidates = pd.read_parquet(args.candidates)
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
    candidates["__decision_ts__"] = pd.to_datetime(
        candidates["__decision_ts__"], utc=True
    )
    declared = json.loads(
        (ROOT / "config/strict_r3_canonical_v2_feature_contract.json").read_text()
    )
    fields = list(
        dict.fromkeys([*_load_contract()[args.side], *declared["severe_context_fields"]])
    )
    chunks = [fields[i : i + args.chunk_size] for i in range(0, len(fields), args.chunk_size)]
    def run_partition(index: int, chunk: list[str]) -> Path:
        output = args.out_dir / "partitions" / f"part_{index:03d}.parquet"
        if args.resume_partitions and output.is_file():
            return output
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--candidates", str(args.candidates),
            "--panel-state", str(args.panel_state),
            "--incremental-features", str(args.incremental_features),
            "--spectral-contract", str(args.spectral_contract),
            "--side", args.side,
            "--out-dir", str(args.out_dir),
            "--worker-fields", json.dumps(chunk),
            "--worker-out", str(output),
        ]
        log_path = output.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode:
            raise RuntimeError(
                f"partition {index} failed with exit={completed.returncode}; see {log_path}"
            )
        return output

    workers = max(1, int(args.max_workers))
    if workers == 1:
        outputs = [run_partition(index, chunk) for index, chunk in enumerate(chunks)]
    else:
        outputs_by_index: dict[int, Path] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_partition, index, chunk): index
                for index, chunk in enumerate(chunks)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                outputs_by_index[index] = future.result()
        outputs = [outputs_by_index[index] for index in range(len(chunks))]
    pieces = [pd.read_parquet(output) for output in outputs]
    reference = pieces[0]
    identity = list(_IDENTITY_COLUMNS)
    for piece in pieces[1:]:
        reference = reference.merge(piece, on=identity, how="inner", validate="one_to_one")
    if len(reference) != len(candidates):
        raise AssertionError("partition merge changed candidate identity")
    reference_path = args.out_dir / "partitioned_exact_features.parquet"
    reference.to_parquet(reference_path, index=False, compression="zstd")
    state = joblib.load(args.panel_state)
    _repair_cross_asset_state_fields(
        reference_path,
        candidates=candidates,
        start=pd.Timestamp(state["history_start"]),
        end=pd.Timestamp(state["end_exclusive"]),
    )
    _refresh_feature_coverage(
        reference_path, args.out_dir / "partitioned_exact_feature_coverage.parquet"
    )
    reference = pd.read_parquet(reference_path)
    incremental = pd.read_parquet(args.incremental_features)
    left = incremental.loc[:, ["candidate_id", *fields]].copy()
    right = reference.loc[:, ["candidate_id", *fields]].copy()
    joined = left.merge(right, on="candidate_id", validate="one_to_one", suffixes=("__state", "__exact"))
    rows = []
    for field in fields:
        a = pd.to_numeric(joined[f"{field}__state"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(joined[f"{field}__exact"], errors="coerce").to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        missing_mismatch = np.isfinite(a) ^ np.isfinite(b)
        delta = np.abs(a[finite] - b[finite])
        scale = np.maximum(1.0, np.abs(b[finite]))
        rows.append(
            {
                "feature": field,
                "comparison_scope": (
                    "stateful_unverified_by_partitioned_reference"
                    if field in _STATEFUL_REFERENCE_EXCLUSIONS
                    else "comparable_partitioned_reference"
                ),
                "rows": int(len(a)),
                "finite_pairs": int(finite.sum()),
                "missing_mismatch_rows": int(missing_mismatch.sum()),
                "max_abs_delta": float(delta.max()) if delta.size else 0.0,
                "max_relative_percent": float((100.0 * delta / scale).max()) if delta.size else 0.0,
                "rows_over_0p01_percent": int((100.0 * delta / scale > 0.01).sum()) if delta.size else 0,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_parquet(args.out_dir / "feature_parity_by_field.parquet", index=False)
    comparable = audit.loc[
        audit["comparison_scope"].eq("comparable_partitioned_reference")
    ]
    failing = comparable.loc[
        comparable["missing_mismatch_rows"].gt(0)
        | comparable["rows_over_0p01_percent"].gt(0)
    ]
    stateful_unverified = audit.loc[
        audit["comparison_scope"].eq("stateful_unverified_by_partitioned_reference"),
        "feature",
    ].tolist()
    manifest = {
        "schema": "strict_r3_partitioned_feature_state_parity_v2",
        "candidate_rows": int(len(joined)),
        "fields": int(len(fields)),
        "partitions": int(len(chunks)),
        "chunk_size": int(args.chunk_size),
        "max_workers": workers,
        "missing_mismatch_rows": int(audit["missing_mismatch_rows"].sum()),
        "rows_over_0p01_percent": int(audit["rows_over_0p01_percent"].sum()),
        "failing_fields": failing["feature"].tolist(),
        "comparable_fields": int(len(comparable)),
        "stateful_unverified_fields": stateful_unverified,
        "full_stateful_parity_proven": False,
        "status": (
            "fail"
            if not failing.empty
            else "inconclusive_stateful_reference_scope"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
