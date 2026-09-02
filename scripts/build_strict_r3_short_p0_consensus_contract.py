#!/usr/bin/env python3
"""Freeze a side-specific short P0/F90 ten-head residual-consensus contract.

The contract is discovered exactly once on the designated Oct--Dec 2024
training-only population.  It uses conditional mutual information of the
policy-net residual grade, conditional on the strict-prequential base rank.
No held 2025 outcome is touched while selecting its input fields.  Geometry
inputs are restricted to the one frozen Geometry/K9 bundle passed on the CLI;
the builder never re-fits a latent representation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.specialist_head_selection import conditional_mi  # noqa: E402
from extreme_price_movements.strict_r3_canonical_v2 import load_geometry_bundle  # noqa: E402


SIDE = "short"
SCHEMA = "strict_r3_short_p0_cmi_consensus_v2"
EDGES = (-150.0, -50.0, 50.0, 150.0)
RANKER_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "n_estimators": 120,
    "learning_rate": 0.035, "max_depth": 5, "num_leaves": 31,
    "min_child_samples": 300, "feature_fraction": 0.82,
    "bagging_fraction": 0.82, "bagging_freq": 1, "lambda_l1": 0.02,
    "lambda_l2": 2.0, "max_bin": 127,
    "label_gain": [0, 0.25, 1, 3, 7], "lambdarank_truncation_level": 10,
    "verbosity": -1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fields(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = [str(value) for value in payload["feature_sets"]["90"]]
    if len(fields) != 90 or len(set(fields)) != 90:
        raise ValueError("short P0/F90 selection must contain exactly 90 fields")
    return fields


def _grade(values: pd.Series) -> np.ndarray:
    residual = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.select(
        [residual <= edge for edge in EDGES], [0, 1, 2, 3], default=4,
    ).astype(np.int8)


def _definition_frame(ledger_root: Path, fields: list[str]) -> pd.DataFrame:
    parts = []
    for month in ("2024-10", "2024-11", "2024-12"):
        path = ledger_root / "ledger" / f"month={month}" / "prequential_base_ledger.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing designated geometry/CMI month: {path}")
        parts.append(pd.read_parquet(path))
    frame = pd.concat(parts, ignore_index=True)
    valid = (
        frame["base_feature_eligible"].fillna(False).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(frame["p0_canonical_net_bps"], errors="coerce").notna()
        & pd.to_numeric(frame["prequential_base_anchor_bps"], errors="coerce").notna()
        & pd.to_numeric(frame["prequential_base_rank42"], errors="coerce").notna()
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("designated CMI population has no valid residual rows")
    frame["policy_residual_bps"] = (
        pd.to_numeric(frame["p0_canonical_net_bps"], errors="raise")
        - pd.to_numeric(frame["prequential_base_anchor_bps"], errors="raise")
    )
    frame["residual_grade"] = _grade(frame["policy_residual_bps"])
    if set(fields).difference(frame.columns):
        raise KeyError("ledger misses frozen F90 fields")
    return frame


def _stable_geometry_fields(state: pd.DataFrame) -> list[str]:
    # Raw membership slots are intentionally excluded from the initial
    # residual-head contract.  Although the bundle is frozen, these role
    # agnostic support/OOD/within-cluster aggregates are safer inputs for a
    # first side-specific consensus.  Raw memberships remain separately
    # auditable for a predeclared later ablation.
    fields = [str(column) for column in state.columns if not str(column).startswith("k09__")]
    if not fields:
        raise ValueError("geometry bundle emitted no stable aggregate fields")
    return fields


def _ranked_fields(frame: pd.DataFrame, fields: list[str], *, family: str) -> pd.DataFrame:
    rows = []
    for field in fields:
        rows.append({
            "field": field,
            "family": family,
            "conditional_mi": conditional_mi(
                frame[field], frame["residual_grade"], frame["prequential_base_rank42"], bins=10,
            ),
            "coverage": float(pd.to_numeric(frame[field], errors="coerce").notna().mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["conditional_mi", "field"], ascending=[False, True], kind="stable",
    ).reset_index(drop=True)


def _interleave(base: list[str], geometry: list[str], total: int, *, offset: int) -> list[str]:
    """Build a CMI-ranked, diverse fixed field subset for one head.

    This deliberately selects *fields*, rather than treating a head's compute
    cap as a feature count.  It mirrors the long ten-head contract: each head
    receives a stable CMI-ranked subset, while the 40..120 values govern the
    query-complete training cap in later training.  The 3:1 base/state cadence
    keeps the residual learner centred on P0 directional information while
    giving every head path/support/OOD context.
    """
    out: list[str] = []
    base_offset = offset % max(len(base), 1)
    geometry_offset = (offset // 2) % max(len(geometry), 1)
    base_index = base_offset
    geometry_index = geometry_offset
    while len(out) < total and (base_index < len(base) + base_offset or geometry_index < len(geometry) + geometry_offset):
        # Roughly three base fields to one structural state field.  The latter
        # should complement directional P0 features, not overwhelm them.
        for _ in range(3):
            if len(out) >= total or not base:
                break
            field = base[base_index % len(base)]
            base_index += 1
            if field not in out:
                out.append(field)
        if len(out) >= total or not geometry:
            continue
        field = geometry[geometry_index % len(geometry)]
        geometry_index += 1
        if field not in out:
            out.append(field)
    if len(out) != total:
        raise AssertionError("could not construct a unique diverse head subset")
    return out


def run(*, ledger_root: Path, selection: Path, geometry_dir: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    base_fields = _fields(selection)
    frame = _definition_frame(ledger_root, base_fields)
    geometry = load_geometry_bundle(geometry_dir)
    if str(geometry.fit_audit.get("side", geometry.fit_audit.get("side_name", "short"))).lower() != SIDE:
        raise ValueError("short consensus requires a short Geometry/K9 bundle")
    state = geometry.transform(frame).reset_index(drop=True)
    stable_geometry = _stable_geometry_fields(state)
    frame = pd.concat([frame.reset_index(drop=True), state.loc[:, stable_geometry]], axis=1)
    base_rank = _ranked_fields(frame, base_fields, family="base")
    geometry_rank = _ranked_fields(frame, stable_geometry, family="geometry")
    base_ranked = base_rank.field.tolist()
    geometry_ranked = geometry_rank.field.tolist()
    # Preserve the long production topology: six exact timestamp × side
    # rankers and four 4-hour × side rankers, with ordinary/equal-month
    # weighting and increasing query-complete training caps.  HPO may change
    # a head's target/query/parameters only through the later versioned
    # ensemble funnel; this definition freezes solely CMI field membership.
    templates = [
        ("cmi_cap40_ordinary", 40, "ordinary", "exact_timestamp_side", 0),
        ("cmi_cap40_equal_month", 40, "equal_month", "exact_timestamp_side", 4),
        ("cmi_cap60_ordinary", 60, "ordinary", "cycle_4h_side", 8),
        ("cmi_cap60_equal_month", 60, "equal_month", "exact_timestamp_side", 12),
        ("cmi_cap80_ordinary", 80, "ordinary", "exact_timestamp_side", 16),
        ("cmi_cap80_equal_month", 80, "equal_month", "cycle_4h_side", 20),
        ("cmi_cap100_ordinary", 100, "ordinary", "exact_timestamp_side", 24),
        ("cmi_cap100_equal_month", 100, "equal_month", "cycle_4h_side", 28),
        ("cmi_cap120_ordinary", 120, "ordinary", "cycle_4h_side", 32),
        ("cmi_cap120_equal_month", 120, "equal_month", "exact_timestamp_side", 36),
    ]
    heads = [
        {
            "name": name, "cap": cap, "weight_mode": weight, "query": query,
            "fields": _interleave(base_ranked, geometry_ranked, cap, offset=offset),
        }
        for name, cap, weight, query, offset in templates
    ]
    all_fields = set(base_fields).union(stable_geometry)
    if any(set(head["fields"]).difference(all_fields) for head in heads):
        raise AssertionError("head contract introduced an unregistered field")
    out.mkdir(parents=True)
    base_rank.to_parquet(out / "base_field_cmi.parquet", index=False, compression="zstd")
    geometry_rank.to_parquet(out / "geometry_field_cmi.parquet", index=False, compression="zstd")
    audit = pd.DataFrame(
        [
            {
                "head": head["name"], "cap": head["cap"], "query": head["query"],
                "weight_mode": head["weight_mode"], "base_fields": sum(field in base_fields for field in head["fields"]),
                "geometry_fields": sum(field in stable_geometry for field in head["fields"]),
                "field_sha256": hashlib.sha256(json.dumps(head["fields"], separators=(",", ":")).encode()).hexdigest(),
            }
            for head in heads
        ]
    )
    audit.to_parquet(out / "head_contract_audit.parquet", index=False, compression="zstd")
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "side": SIDE,
        "selection_population": {
            "window": "2024-10-01T00:00:00Z through 2025-01-01T00:00:00Z",
            "rows": int(len(frame)),
            "outcomes": "valid policy-net rows; used only to freeze CMI field membership",
            "condition": "strict-prequential base_rank42",
            "target": "policy_net_bps - prequential_base_anchor_bps ordinalized at -150/-50/+50/+150 bps",
        },
        "base_selection": {"path": str(selection), "sha256": _sha256(selection), "fields": base_fields},
        "geometry": {
            "path": str(geometry_dir), "bundle_sha256": geometry.bundle_sha256,
            "raw_memberships_included": False, "stable_fields": stable_geometry,
        },
        "target": {"name": "policy_net_residual_ordinal_150_50", "edges_bps": list(EDGES)},
        "ranker_params": RANKER_PARAMS,
        "hpo": {
            "status": "pending",
            "role": (
                "The follow-on HPO chooses a per-head target/query/parameter "
                "variant only by its incremental base-plus-ensemble economics; "
                "standalone head metrics are diagnostic only."
            ),
            "no_improvement_patience": 20,
        },
        "heads": heads,
        "ensemble_selection": "candidate heads are assessed only as OOF ensemble contributions; no standalone-head promotion",
    }
    (out / "short_consensus_contract.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_p0_cmi_consensus_contract_v2", "status": "complete",
        "source_hashes": {
            "ledger_manifest": _sha256(ledger_root / "run_manifest.json"),
            "selection": _sha256(selection), "geometry_manifest": _sha256(geometry_dir / "run_manifest.json"),
        },
        "contract_sha256": _sha256(out / "short_consensus_contract.json"),
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(
        ledger_root=args.ledger_root, selection=args.selection,
        geometry_dir=args.geometry_dir, out=args.out,
    ))


if __name__ == "__main__":
    main()
