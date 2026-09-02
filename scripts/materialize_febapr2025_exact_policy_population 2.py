#!/usr/bin/env python3
"""Freeze the accepted Feb--Apr 2025 exact-policy base-training population.

The population contains only the newly reconstructed deployed-policy labels
whose one-minute paths are complete.  It intentionally carries *keys* into the
archived point-in-time feature ledgers rather than copying features or silently
using a different historical feature representation.
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

from scripts.reconstruct_janfeb2025_execution_ev_12h_oof import (  # noqa: E402
    normalize_symbol,
    source_paths,
)

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
LABEL_COLUMNS = (
    "execution_decision_utc",
    "execution_label_end_utc",
    "candidate_month",
    "execution_gross_ev_12h",
    "execution_net_ev_12h",
    "execution_cost_return",
    "execution_exit_reason",
    "execution_exit_minute",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
    "execution_soft_positive_12h",
)
DEFAULT_EXACT_LABELS = ROOT / (
    "data_perp/artifacts/febapr2025_execution_ev_current_spread_two_layer_oof_20260727_v2/"
    "exact_1m_execution_ev_12h_labels.parquet"
)
DEFAULT_ACTIVE_OOF = ROOT / (
    "data_perp/artifacts/regime_transition_active_head_20260726_v1/grouped_oof.parquet"
)
DEFAULT_PARITY = ROOT / (
    "data_perp/artifacts/deployed_policy_label_parity_20260727_v1/evidence_gate.json"
)
DEFAULT_READINESS = ROOT / (
    "data_perp/artifacts/historical_exact_policy_readiness_20260727_v2/evidence_gate.json"
)
DEFAULT_LABELS_ROOT = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v1"
CURRENT_31_8_MANIFESTS = {
    "long": ROOT / "data_perp/artifacts/packb_side_local_outer_oof_july20_20260726_v1_31_8/long/manifest.json",
    "short": ROOT / "data_perp/artifacts/packb_side_local_outer_oof_july20_20260726_v1_31_8/short/manifest.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_population(frame: pd.DataFrame) -> None:
    """Fail closed on identity, causal timestamp or exact-label violations."""

    required = {
        *IDENTITY,
        *LABEL_COLUMNS,
        "execution_label_available_at_utc",
        "feature_source_ledger",
        "has_exact_1m_path",
        "has_feature_store_join_key",
        "eligible_for_fresh_canonical_base_oof",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"population is missing required fields: {missing}")
    if frame.empty or frame.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("population identities must be nonempty and unique")
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    decision = pd.to_datetime(frame["execution_decision_utc"], utc=True, errors="coerce")
    end = pd.to_datetime(frame["execution_label_end_utc"], utc=True, errors="coerce")
    available = pd.to_datetime(frame["execution_label_available_at_utc"], utc=True, errors="coerce")
    if ts.isna().any() or decision.isna().any() or end.isna().any() or available.isna().any():
        raise ValueError("population timestamps must be valid UTC values")
    if not decision.equals(ts + pd.Timedelta(hours=1)):
        raise ValueError("decision timestamp must equal signal timestamp + one hour")
    if not end.equals(decision + pd.Timedelta(hours=12)):
        raise ValueError("exact label end must equal decision timestamp + twelve hours")
    if not available.equals(end):
        raise ValueError("label availability must equal exact label resolution timestamp")
    numeric = ["execution_gross_ev_12h", "execution_net_ev_12h", "execution_cost_return"]
    values = frame.loc[:, numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("exact-policy economics must be finite")
    if not np.allclose(values[:, 0] - values[:, 2], values[:, 1], rtol=0.0, atol=1e-7):
        raise ValueError("exact-policy gross-cost reconciliation failed")
    flags = ("has_exact_1m_path", "has_feature_store_join_key", "eligible_for_fresh_canonical_base_oof")
    if not frame.loc[:, list(flags)].all(axis=None):
        raise ValueError("frozen population contains an ineligible or incomplete row")
    if frame["feature_source_ledger"].isna().any():
        raise ValueError("feature-store source ledger key is incomplete")


def _load_feature_keys(labels_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = source_paths(labels_root, start_month="2025-02", end_month="2025-04")
    parts: list[pd.DataFrame] = []
    for path in paths:
        rows = pd.read_parquet(path, columns=[*IDENTITY, "__decision_ts__"])
        rows["__ts__"] = pd.to_datetime(rows["__ts__"], utc=True, errors="raise")
        rows["__symbol__"] = rows["__symbol__"].map(normalize_symbol)
        rows["side_name"] = rows["side_name"].astype(str).str.lower()
        rows["candidate_id"] = rows["candidate_id"].astype(str)
        rows["feature_source_ledger"] = str(path)
        rows["feature_store_signal_utc"] = rows["__ts__"]
        rows["feature_store_decision_utc"] = pd.to_datetime(rows.pop("__decision_ts__"), utc=True, errors="raise")
        parts.append(rows)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("archived feature-store identities are not unique")
    return result, paths


def _transition_context(active_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = pd.read_parquet(
        active_path,
        columns=["source_utc", "target__event_id", "target__transition_active"],
    )
    active["source_utc"] = pd.to_datetime(active["source_utc"], utc=True, errors="raise")
    if active["source_utc"].duplicated().any():
        raise ValueError("transition source hours must be unique")
    event_rows = active.loc[active["target__event_id"].notna()].copy()
    windows = (
        event_rows.groupby("target__event_id", sort=True)
        .agg(
            transition_window_start_utc=("source_utc", "min"),
            transition_window_end_utc=("source_utc", "max"),
            transition_active_hours=("target__transition_active", "sum"),
        )
        .reset_index()
        .rename(columns={"target__event_id": "transition_event_id"})
    )
    hourly = active.rename(
        columns={
            "source_utc": "__ts__",
            "target__event_id": "transition_event_id",
            "target__transition_active": "expost_transition_active",
        }
    )
    hourly["transition_window_member"] = hourly["transition_event_id"].notna()
    return hourly, windows


def _current_feature_contract() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side, path in CURRENT_31_8_MANIFESTS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = list(payload.get("features", ()))
        expected = 31 if side == "long" else 8
        if len(features) != expected:
            raise ValueError(f"current {side} feature contract is not {expected} fields")
        output[side] = {"manifest": str(path), "sha256": _sha256(path), "features": features}
    return output


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    readiness = json.loads(args.readiness_gate.read_text(encoding="utf-8"))
    parity = json.loads(args.parity_gate.read_text(encoding="utf-8"))
    accepted = next(
        (item for item in readiness.get("periods", ()) if item.get("period") == "2025-02..2025-04"),
        None,
    )
    if not parity.get("comparison", {}).get("parity_pass") or not accepted or not accepted.get("new_exact_policy_labels_accepted"):
        raise ValueError("Feb-Apr exact-policy readiness/parity prerequisite has not passed")
    labels = pd.read_parquet(args.exact_labels, columns=[*IDENTITY, *LABEL_COLUMNS])
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="raise")
    labels["__symbol__"] = labels["__symbol__"].map(normalize_symbol)
    labels["side_name"] = labels["side_name"].astype(str).str.lower()
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    keys, source_paths_used = _load_feature_keys(args.labels_root)
    population = labels.merge(keys, on=list(IDENTITY), how="left", validate="one_to_one")
    if population["feature_source_ledger"].isna().any():
        raise ValueError("accepted exact labels do not have a one-to-one feature-store key")
    hourly, windows = _transition_context(args.active_oof)
    population = population.merge(hourly, on="__ts__", how="left", validate="many_to_one")
    population = population.merge(windows, on="transition_event_id", how="left", validate="many_to_one")
    population["execution_label_available_at_utc"] = pd.to_datetime(
        population["execution_label_end_utc"], utc=True, errors="raise"
    )
    population["has_exact_1m_path"] = True
    population["has_feature_store_join_key"] = True
    population["deployed_policy_current_overlap_parity_pass"] = True
    population["frozen_canonical_path_subset"] = True
    population["historical_current_spread_counterfactual"] = True
    population["eligible_for_fresh_canonical_base_oof"] = True
    population["expost_transition_active"] = population["expost_transition_active"].fillna(0).astype(np.int8)
    population["transition_window_member"] = population["transition_window_member"].fillna(False).astype(bool)
    population = population.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    validate_population(population)
    population_windows = windows.loc[
        windows["transition_event_id"].isin(
            population.loc[population["transition_window_member"], "transition_event_id"].dropna().unique()
        )
    ].copy()
    # Keys are separate so a trainer can load the point-in-time feature ledger
    # by identity without consuming any target/path columns from this artifact.
    join_keys = population.loc[:, [
        *IDENTITY,
        "feature_source_ledger",
        "feature_store_signal_utc",
        "feature_store_decision_utc",
        "execution_label_available_at_utc",
        "eligible_for_fresh_canonical_base_oof",
    ]].copy()
    args.output_dir.mkdir(parents=True)
    population_path = args.output_dir / "population.parquet"
    identities_path = args.output_dir / "frozen_candidate_identities.parquet"
    join_keys_path = args.output_dir / "feature_store_join_keys.parquet"
    windows_path = args.output_dir / "transition_event_windows.parquet"
    manifest_path = args.output_dir / "population_gate.json"
    population.to_parquet(population_path, index=False, compression="zstd")
    population.loc[:, list(IDENTITY)].to_parquet(identities_path, index=False, compression="zstd")
    join_keys.to_parquet(join_keys_path, index=False, compression="zstd")
    population_windows.to_parquet(windows_path, index=False, compression="zstd")
    contract = _current_feature_contract()
    manifest = {
        "schema": "febapr2025_frozen_exact_policy_base_population_v1",
        "purpose": "authoritative accepted identity/label gate for fresh canonical 31/8 base OOF training",
        "upstream_gates": {
            "readiness_gate": str(args.readiness_gate),
            "readiness_gate_sha256": _sha256(args.readiness_gate),
            "deployed_policy_parity_gate": str(args.parity_gate),
            "deployed_policy_parity_gate_sha256": _sha256(args.parity_gate),
        },
        "identity": list(IDENTITY),
        "timing": {
            "signal": "__ts__",
            "decision": "execution_decision_utc = signal + 1h",
            "label_resolution": "execution_label_available_at_utc = execution_label_end_utc = decision + 12h",
            "training_rule": "a base OOF fit may use a row only after execution_label_available_at_utc",
        },
        "labels": {
            "target": "execution_net_ev_12h",
            "source": str(args.exact_labels),
            "source_sha256": _sha256(args.exact_labels),
            "cost_reconciliation": "execution_gross_ev_12h - execution_cost_return = execution_net_ev_12h",
            "execution": "deployed spread-aware simple_policy_optimiser target; candidate-local exact 1m replay",
        },
        "population": {
            "rows": int(len(population)),
            "by_side": {str(k): int(v) for k, v in population["side_name"].value_counts().sort_index().items()},
            "by_month": {str(k): int(v) for k, v in population["candidate_month"].value_counts().sort_index().items()},
            "active_transition_rows": int(population["expost_transition_active"].sum()),
            "active_transition_events": int(population.loc[population["expost_transition_active"].eq(1), "transition_event_id"].dropna().nunique()),
            "transition_window_rows": int(population["transition_window_member"].sum()),
        },
        "feature_store": {
            "join_keys": str(join_keys_path),
            "source_ledgers": {str(path): _sha256(path) for path in source_paths_used},
            "contract": "one-to-one raw point-in-time feature-ledger identity join; features are deliberately not copied into the target population",
            "fresh_canonical_31_8_contract": contract,
        },
        "exclusions": [
            "all missing exact-1m paths",
            "January 2025 because no canonical path-input join exists",
            "December 2025 from pooled economics because exact-1m coverage is insufficient",
            "invalidated exact_history_state_recurrence_20260727_v1 economics",
        ],
        "artifacts": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in (population_path, identities_path, join_keys_path, windows_path)
        },
    }
    _write_json(manifest_path, manifest)
    return {"population": population_path, "join_keys": join_keys_path, "windows": windows_path, "gate": manifest_path}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--exact-labels", type=Path, default=DEFAULT_EXACT_LABELS)
    result.add_argument("--active-oof", type=Path, default=DEFAULT_ACTIVE_OOF)
    result.add_argument("--parity-gate", type=Path, default=DEFAULT_PARITY)
    result.add_argument("--readiness-gate", type=Path, default=DEFAULT_READINESS)
    result.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    outputs = run(options)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
