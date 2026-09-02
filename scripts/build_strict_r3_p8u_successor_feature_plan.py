#!/usr/bin/env python3
"""Build the hash-bound Router/F72/Under feature union for a P8U successor.

The historical P8U 175-field plan cannot be reconstructed because its exact
Under-F120 selection artifact was deleted.  This utility creates a *new*
source-aligned plan from retained, provenance-bearing Router30, F72, and
Under contracts.  It does not score, train, or read targets/outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

# Exact ordered Router30 sequence documented in the retained P8U v6 handover.
# This is input metadata for a newly named successor, never a claim that the
# deleted historical Router model package has been restored.
ROUTER30 = (
    "liq_stop_safety_short_atr", "mark_perp_dislocation", "rv_rel_universe",
    "range_24h_pct", "ffd_rv_24h_04", "upside_semivariance_24",
    "ffd_rv_6h_06", "t_be_proxy", "dist_prior_day_high",
    "upside_semivariance_8", "dist_rolling_7d_high", "asset_atr_level",
    "vov_mad_60", "vov_iqr_20", "seasonality_strength",
    "realized_volatility_24h", "cvar_5pct", "dist_prior_day_low",
    "liquidity_ratio_peer_resid", "mark_trigger_risk_10h", "t_pl_proxy",
    "price_rv_7d_robust_z", "range_volatility", "rvol_hod_base",
    "range_per_volume", "price_rv_15d_robust_z", "ob_depth_l20_to_qv_24h",
    "beta_eth_24h", "beta_btc_24h", "rv_48h",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable feature plan already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ordered_union(*contracts: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(name for contract in contracts for name in contract))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f72-contract", type=Path, required=True)
    parser.add_argument("--under-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    f72_doc = json.loads(args.f72_contract.read_text(encoding="utf-8"))
    under_doc = json.loads(args.under_contract.read_text(encoding="utf-8"))
    f72 = tuple(map(str, f72_doc.get("feature_contract") or ()))
    under = tuple(map(str, under_doc.get("features") or ()))
    if len(ROUTER30) != 30 or len(set(ROUTER30)) != 30:
        raise AssertionError("Router30 specification is malformed")
    if len(f72) != 72 or len(set(f72)) != 72:
        raise ValueError("expected an ordered 72-field F72 contract")
    if len(under) != 120 or len(set(under)) != 120:
        raise ValueError("expected an ordered 120-field Under contract")
    union = _ordered_union(ROUTER30, f72, under)
    overlaps = {
        "router_base": len(set(ROUTER30).intersection(f72)),
        "router_under": len(set(ROUTER30).intersection(under)),
        "base_under": len(set(f72).intersection(under)),
    }
    payload: dict[str, Any] = {
        "schema": "strict_r3_p8u_successor_feature_plan_v1",
        "purpose": "separately named source-aligned September successor; not historical bit parity",
        "target_or_outcome_input": False,
        "router_features": list(ROUTER30),
        "f72_features": list(f72),
        "under_features": list(under),
        "full_union": union,
        "counts": {
            "router": len(ROUTER30), "f72": len(f72), "under": len(under),
            "full_union": len(union), **overlaps,
        },
        "source_contracts": {
            "f72": {"path": str(args.f72_contract.resolve()), "sha256": _sha256(args.f72_contract)},
            "under": {"path": str(args.under_contract.resolve()), "sha256": _sha256(args.under_contract)},
            "router": "retained P8U v6 documented Router30 order",
        },
    }
    _atomic_json(args.out, payload)
    print(json.dumps({"out": str(args.out), "full_union": len(union), "overlaps": overlaps}, sort_keys=True))


if __name__ == "__main__":
    main()
