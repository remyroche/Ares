"""Fail-closed readiness check for a native 12h first-touch base-label challenger."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import pyarrow.parquet as pq


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parquet_info(path: Path) -> dict[str, Any]:
    table=pq.ParquetFile(path)
    return {"path":str(path),"sha256":_sha(path),"rows":int(table.metadata.num_rows),"columns":table.schema_arrow.names}


def build_readiness_gate(
    *, base_oof: Path, native_label_example: Path, exact_12h_paths: Path, paths_manifest: Path,
) -> dict[str, Any]:
    """Assess exact-source readiness without constructing an outcome proxy.

    The comparison requires a label for every accepted base OOF identity and a
    frozen first-touch recipe. An exact execution EV or its policy exit fields
    are deliberately not accepted as a substitute native label.
    """
    base=_parquet_info(base_oof);native=_parquet_info(native_label_example);paths=_parquet_info(exact_12h_paths)
    manifest=json.loads(paths_manifest.read_text())
    required_native={"candidate_id","__decision_ts__","__barrier_pct__","__tp__","__sl__","__first_touch_target_soft__","__first_touch_capture_net__"}
    native_columns=set(native["columns"])
    native_geometry=sorted(required_native-native_columns)
    path_length=int((manifest.get("path") or {}).get("fixed_length",0))
    timing=manifest.get("timing") or {}
    cadence_minutes=int(timing.get("cadence_minutes",0))
    path_minutes=int(timing.get("path_minutes",0))
    full_identity=paths["rows"]==base["rows"]
    twelve_hour_path=path_length==720 and cadence_minutes==1 and path_minutes==720
    # Existing old label ledgers expose materialized outcomes but no immutable
    # recipe/state hash tying their 24h soft target to a particular path-order
    # function, same-bar convention, cost schedule and geometry contract.
    recipe_manifested=bool((manifest.get("native_first_touch_recipe") or {}).get("sha256"))
    ready=full_identity and twelve_hour_path and not native_geometry and recipe_manifested
    blockers=[]
    if not full_identity: blockers.append("MISSING_FULL_BASE_UNIVERSE_EXACT_12H_PATHS")
    if not recipe_manifested: blockers.append("MISSING_FROZEN_NATIVE_FIRST_TOUCH_RECIPE_AND_24H_REPLAY_PARITY")
    if native_geometry: blockers.append("MISSING_NATIVE_GEOMETRY_COLUMNS")
    if not twelve_hour_path: blockers.append("PATH_HORIZON_NOT_EXACT_12H")
    return {
        "schema":"native_12h_base_label_challenger_readiness_v1",
        "status":"READY_FOR_MATERIALISATION" if ready else "BLOCKED_EXACT_NATIVE_12H_LABEL_NOT_YET_CONSTRUCTIBLE",
        "research_only":True,
        "candidate_contract":{"accepted_base_oof_rows":base["rows"],"identical_rows_required":True,"same_features_and_monthly_folds_required":True},
        "source_evidence":{"base_oof":base,"archived_24h_native_label_example":native,"available_exact_12h_paths":paths,"available_exact_12h_path_manifest":{"path":str(paths_manifest),"sha256":_sha(paths_manifest),"fixed_length":path_length,"cadence_minutes":cadence_minutes,"horizon_minutes":path_minutes}},
        "construction_proof":{"native_label_must_not_use":"execution gross/cost/net EV or policy exit fields","required_target":"native first-touch soft label recomputed from ordered OHLC path, side/row barrier, TP/SL geometry, frozen same-bar convention and frozen native cost/target rule","required_resolution":"__decision_ts__ + 12h","available_path_is_exact_12h":twelve_hour_path,"available_path_has_full_accepted_identity_coverage":full_identity,"archived_native_geometry_columns_present":not native_geometry,"frozen_recipe_and_24h_parity_present":recipe_manifested},
        "blockers":blockers,
        "required_unblockers":[
            "Materialise canonical 720x1m OHLC paths for all 509868 accepted Feb-Apr base-OOF identities, not only the 205194 timestamp-side top40 rows.",
            "Freeze and hash the historical native first-touch recipe: row/side geometry, cost rule, ordered same-bar tie-break, soft-target transform, and 12h truncation; prove it reproduces the archived 24h native target on a fixed audit sample before changing horizon.",
            "Write native_first_touch_12h_soft plus gross/net/hit/stop/timeout and native_label_resolution_utc=decision+12h keyed by candidate_id; keep exact execution EV separate as an evaluation field.",
            "Then retrain the existing monthly side-local base folds with unchanged features/HPO protocol and compare on identical OOF identities.",
        ],
    }
