#!/usr/bin/env python3
"""Materialise strict-prequential H1--H5 health artifacts from strict roots.

The family manifests are deliberately explicit.  This script never chooses
context/covariance families from the evaluation root: those selections must be
frozen from predecessor resolved data by the later transport-selection stage.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_leaf_health import CausalLeafHealthConfig  # noqa: E402
from extreme_price_movements.causal_leaf_health_artifacts import (  # noqa: E402
    materialize_strict_oof_causal_leaf_health,
    spool_completed_strict_oof_family_inputs,
)
from extreme_price_movements.causal_leaf_health_streaming import (  # noqa: E402
    materialize_strict_oof_causal_leaf_health_streaming,
)
from extreme_price_movements.causal_leaf_health_scoped import (  # noqa: E402
    materialize_strict_oof_causal_leaf_health_scoped,
)
from extreme_price_movements.causal_leaf_health_event_incremental import (  # noqa: E402
    materialize_strict_oof_causal_leaf_health_event_incremental,
)
from extreme_price_movements.causal_leaf_health_prerequisites import (  # noqa: E402
    FAMILY_SELECTION_ROOT_STATUS,
    load_frozen_family_selection,
    load_strict_fold_context,
    validate_selection_application,
)
from extreme_price_movements.strict_event_store import load_strict_event_store  # noqa: E402


def _selection(path: Path | None, *, expected_kind: str | None = None) -> frozenset[tuple[str, str, str, str, str]]:
    if path is None:
        return frozenset()
    # New strict predecessor manifests include both the family kind and the
    # exact label-availability cutoff.  Retain the legacy compact JSON reader
    # for an already-frozen historical experiment, but any new materialiser
    # should use ``materialize_strict_predecessor_family_selections.py``.
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid frozen family selection manifest: {path}") from exc
    if isinstance(payload, dict) and payload.get("status") == "FROZEN_STRICT_PREDECESSOR_FAMILY_SELECTION":
        return load_frozen_family_selection(path, expected_kind=expected_kind).selected_families
    entries = payload.get("selected_families", payload.get("families", [])) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        raise SystemExit("family selection manifest must contain a list under selected_families or families")
    output: set[tuple[str, str, str, str, str]] = set()
    fields = ("feature_contract_sha256", "side_name", "head_name", "rule_signature", "contribution_direction")
    for entry in entries:
        if isinstance(entry, dict):
            try:
                key = tuple(str(entry[field]) for field in fields)
            except KeyError as exc:
                raise SystemExit(f"family selection row lacks {exc.args[0]}") from exc
        elif isinstance(entry, list) and len(entry) == 5:
            key = tuple(map(str, entry))
        else:
            raise SystemExit("family selection rows must be five-field objects or lists")
        output.add(key)
    return frozenset(output)


def _strict_selection(path: Path | None, *, expected_kind: str):
    """Return the strict selection object when this is a new manifest."""

    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid frozen family selection manifest: {path}") from exc
    if isinstance(payload, dict) and payload.get("status") == "FROZEN_STRICT_PREDECESSOR_FAMILY_SELECTION":
        return load_frozen_family_selection(path, expected_kind=expected_kind)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--strict-root", type=Path, action="append", help="completed strict root; repeat only for the legacy/reference path")
    source.add_argument("--event-store", type=Path, help="completed immutable strict event store; uses the scope-bounded vectorised H1--H5 path")
    parser.add_argument("--contribution-event-streams", type=Path, default=None, help="sealed paired contribution-event sidecar; with --event-store uses the bounded event-incremental H1--H5 path")
    context = parser.add_mutually_exclusive_group(required=True)
    context.add_argument("--causal-context", type=Path, help="parquet with regime_available_utc and causal numeric fields")
    context.add_argument("--context-sidecar-root", type=Path, help="verified July-2023--Nov-2024 strict causal context root")
    parser.add_argument("--context-column", action="append", default=None, help="predeclared context field; repeat up to ten times; defaults to sidecar contract fields")
    parser.add_argument("--context-family-manifest", type=Path, default=None)
    parser.add_argument("--covariance-family-manifest", type=Path, default=None)
    parser.add_argument("--relationship-family-manifest", type=Path, default=None)
    parser.add_argument("--family-selection-root", type=Path, default=None, help="strict predecessor-selection root containing all three H3/H4/H5 manifests")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bounded-streaming", action="store_true", help="use disk-spooled, chronological production H1--H5 materialisation")
    parser.add_argument("--input-spool-dir", type=Path, default=None, help="immutable strict input spool; created if absent when --bounded-streaming is used")
    parser.add_argument("--stream-batch-rows", type=int, default=25_000, help="maximum joined family rows decoded per streaming batch")
    parser.add_argument("--max-selected-state-rows", type=int, default=3_000_000, help="hard H4/H5 selected-state dataframe limit; exceedance fails closed")
    parser.add_argument("--vector-threads", type=int, default=2, help="DuckDB threads for the scope-bounded event-store path")
    parser.add_argument("--vector-memory-limit", default="2GB", help="per-scope DuckDB memory ceiling for the event-store path")
    parser.add_argument("--vector-temp-disk-limit", default="16GB", help="per-scope DuckDB spill ceiling for the event-store path; fails closed before consuming reserved artifact space")
    parser.add_argument("--verify-event-store-parts", action="store_true", help="re-hash every immutable event-store part before materialisation (slow; normal reuse validates sealed manifest and lineage)")
    args = parser.parse_args()
    if args.context_column is not None and len(args.context_column) > 10:
        parser.error("at most ten context fields are allowed for the H3/H4 contract")
    event_store = None
    if args.event_store is not None:
        if args.bounded_streaming or args.input_spool_dir is not None:
            parser.error("--event-store already supplies the bounded canonical source; do not combine it with --bounded-streaming or --input-spool-dir")
        event_store = load_strict_event_store(
            args.event_store, verify_parts=args.verify_event_store_parts, verify_source=True,
        )
        strict_roots = [Path(item) for item in event_store.manifest["source"]["strict_roots"]]
    else:
        strict_roots = list(args.strict_root or [])
        if args.contribution_event_streams is not None:
            parser.error("--contribution-event-streams requires --event-store")
    if args.family_selection_root is not None:
        if any((args.context_family_manifest, args.covariance_family_manifest, args.relationship_family_manifest)):
            parser.error("--family-selection-root cannot be combined with individual family manifests")
        root_manifest = args.family_selection_root / "manifest.json"
        try:
            root_payload = json.loads(root_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"invalid strict family-selection root: {args.family_selection_root}: {exc}")
        if root_payload.get("status") != FAMILY_SELECTION_ROOT_STATUS:
            parser.error("--family-selection-root is not a completed strict predecessor-selection artifact")
        args.context_family_manifest = args.family_selection_root / "h3_context_family_selection.json"
        args.covariance_family_manifest = args.family_selection_root / "h4_covariance_family_selection.json"
        args.relationship_family_manifest = args.family_selection_root / "h5_relationship_family_selection.json"
    strict_selections = [
        _strict_selection(args.context_family_manifest, expected_kind="context"),
        _strict_selection(args.covariance_family_manifest, expected_kind="covariance"),
        _strict_selection(args.relationship_family_manifest, expected_kind="relationship"),
    ]
    selection_activation = None
    if any(item is not None for item in strict_selections):
        selection_activation = validate_selection_application(
            [item for item in strict_selections if item is not None], strict_roots=strict_roots,
        )
    if args.context_sidecar_root is not None:
        context, context_columns, _ = load_strict_fold_context(
            args.context_sidecar_root, context_columns=args.context_column,
        )
    else:
        if not args.context_column:
            parser.error("--context-column is required when --causal-context is supplied")
        context = pd.read_parquet(args.causal_context)
        context_columns = tuple(args.context_column)
    config = replace(
        CausalLeafHealthConfig(),
        selected_context_families=_selection(args.context_family_manifest, expected_kind="context"),
        selected_covariance_families=_selection(args.covariance_family_manifest, expected_kind="covariance"),
        selected_relationship_families=_selection(args.relationship_family_manifest, expected_kind="relationship"),
        family_selection_effective_utc=selection_activation.isoformat() if selection_activation is not None else None,
    )
    if event_store is not None:
        if args.contribution_event_streams is not None:
            output = materialize_strict_oof_causal_leaf_health_event_incremental(
                event_store, args.contribution_event_streams, args.output_dir,
                causal_context=context, context_feature_columns=context_columns, config=config,
                batch_rows=args.stream_batch_rows, max_selected_state_rows=args.max_selected_state_rows,
                memory_limit=args.vector_memory_limit, temp_disk_limit=args.vector_temp_disk_limit,
                verify_event_store_parts=args.verify_event_store_parts,
            )
        else:
            output = materialize_strict_oof_causal_leaf_health_scoped(
                event_store, args.output_dir, causal_context=context,
                context_feature_columns=context_columns, config=config,
                threads=args.vector_threads, memory_limit=args.vector_memory_limit,
                verify_event_store_parts=args.verify_event_store_parts,
                max_selected_state_rows=args.max_selected_state_rows,
                temp_disk_limit=args.vector_temp_disk_limit,
            )
    elif args.bounded_streaming:
        if args.input_spool_dir is None:
            parser.error("--input-spool-dir is required with --bounded-streaming")
        if args.input_spool_dir.exists():
            spool_root = args.input_spool_dir
        else:
            spool_root = spool_completed_strict_oof_family_inputs(
                strict_roots, args.input_spool_dir,
            ).root
        output = materialize_strict_oof_causal_leaf_health_streaming(
            spool_root, args.output_dir, causal_context=context,
            context_feature_columns=context_columns, config=config,
            batch_rows=args.stream_batch_rows,
            max_selected_state_rows=args.max_selected_state_rows,
        )
    else:
        output = materialize_strict_oof_causal_leaf_health(
            strict_roots, args.output_dir, causal_context=context,
            context_feature_columns=context_columns, config=config,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
