#!/usr/bin/env python3
"""Create an exact-decision shadow state from canonical live holdings.

The live executor stores exchange identifiers and actual notionals, whereas
the policy simulator stores unit-wallet notionals plus adaptive-exit context.
This bridge keeps only simulator positions whose candidate IDs are present in
the live ledger, preserving their policy metadata and failing closed when a
live position cannot be matched.  It never creates a position from market
data and never calls the exchange.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


LIVE_SCHEMA = "strict_r3_kraken_live_state_v1"
SHADOW_SCHEMA = "strict_r3_shadow_portfolio_state_v3_adaptive_exit"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _positive_float(row: dict, field: str) -> float:
    value = float(row.get(field))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"live position requires positive finite {field}")
    return value


def _overlay_live_execution_state(*, shadow_row: dict, live_row: dict) -> dict:
    """Keep policy lineage, but anchor path-dependent state to actual fills.

    Unit-wallet ``gross_notional`` intentionally remains the simulator value.
    Entry price, ATR, timing and trailing state affect exit decisions and must
    therefore come from the canonical live ledger rather than the hypothetical
    shadow fill.
    """
    output = copy.deepcopy(shadow_row)
    output["entry_price"] = _positive_float(live_row, "entry_price")
    output["atr"] = _positive_float(live_row, "atr")
    output["entry_ts"] = _utc(live_row["entry_ts"]).isoformat()
    output["timeout_ts"] = _utc(live_row["timeout_ts"]).isoformat()
    output["next_bar_ts"] = _utc(
        live_row.get("next_bar_ts", live_row["entry_ts"])
    ).isoformat()
    output["maximum_favourable"] = float(live_row.get("maximum_favourable", 0.0))
    output["trailing_armed"] = bool(live_row.get("trailing_armed", False))
    output["effective_leverage"] = _positive_float(live_row, "effective_leverage")
    if live_row.get("trailing_activation_atr") is not None:
        output["trailing_activation_atr"] = float(live_row["trailing_activation_atr"])
    output["adaptive_entry_context"] = dict(
        live_row.get("adaptive_entry_context") or output.get("adaptive_entry_context") or {}
    )
    output["adaptive_score_history"] = list(
        live_row.get("adaptive_score_history") or output.get("adaptive_score_history") or []
    )
    output["adaptive_last_decision"] = (
        live_row.get("adaptive_last_decision")
        if live_row.get("adaptive_last_decision") is not None
        else output.get("adaptive_last_decision")
    )
    return output


def _persisted_live_policy_lineage(*, live_row: dict, wallet: float) -> dict | None:
    """Recover only explicitly persisted entry policy lineage from live state."""
    payload = live_row.get("shadow_policy_lineage")
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != "strict_r3_live_shadow_policy_lineage_v1":
        return None
    if str(payload.get("candidate_id")) != str(live_row.get("candidate_id")):
        raise ValueError("persisted live policy lineage candidate mismatch")
    if str(payload.get("symbol")) != str(live_row.get("symbol")):
        raise ValueError("persisted live policy lineage symbol mismatch")
    fraction = float(payload.get("gross_notional_fraction", float("nan")))
    if not math.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("persisted live policy lineage lacks positive notional fraction")
    return {
        "candidate_id": str(live_row["candidate_id"]),
        "symbol": str(live_row["symbol"]),
        "side": "long",
        "entry_price": _positive_float(live_row, "entry_price"),
        "atr": _positive_float(live_row, "atr"),
        "entry_ts": _utc(live_row["entry_ts"]).isoformat(),
        "timeout_ts": _utc(live_row["timeout_ts"]).isoformat(),
        "next_bar_ts": _utc(live_row.get("next_bar_ts", live_row["entry_ts"])).isoformat(),
        "gross_notional": float(wallet) * fraction,
        "maximum_favourable": float(live_row.get("maximum_favourable", 0.0)),
        "trailing_armed": bool(live_row.get("trailing_armed", False)),
        "effective_leverage": _positive_float(live_row, "effective_leverage"),
        "trailing_activation_atr": payload.get("trailing_activation_atr"),
        "adaptive_entry_context": dict(payload.get("adaptive_entry_context") or {}),
        "adaptive_score_history": list(payload.get("adaptive_score_history") or []),
        "adaptive_last_decision": payload.get("adaptive_last_decision"),
    }


def bridge_state(
    *,
    live: dict,
    shadow: dict,
    decision_ts: object,
    live_state_path: Path,
    shadow_reference_path: Path,
    policy_lineage_reference: dict | None = None,
    policy_lineage_reference_path: Path | None = None,
) -> dict:
    """Return a shadow-policy state constrained by actual live holdings."""
    if live.get("schema") != LIVE_SCHEMA:
        raise ValueError("unexpected canonical live-state schema")
    if shadow.get("schema") != SHADOW_SCHEMA:
        raise ValueError("unexpected shadow-state schema")
    decision = _utc(decision_ts)
    live_as_of = _utc(live["as_of_ts"])
    if live_as_of > decision:
        raise ValueError("live state is newer than the requested decision")

    live_positions = list(live.get("positions") or [])
    live_by_id = {str(row.get("candidate_id")): row for row in live_positions}
    if None in live_by_id or "None" in live_by_id:
        raise ValueError("every live position requires a candidate_id")
    if len(live_by_id) != len(live_positions):
        raise ValueError("live state has duplicate candidate positions")
    shadow_positions = list(shadow.get("open_positions") or [])
    shadow_by_id = {str(row.get("candidate_id")): row for row in shadow_positions}
    lineage_positions = list(
        (policy_lineage_reference or {}).get("open_positions") or []
    )
    lineage_by_id = {
        str(row.get("candidate_id")): row for row in lineage_positions
    }
    persisted_by_id = {
        key: lineage
        for key, live_row in live_by_id.items()
        if (lineage := _persisted_live_policy_lineage(
            live_row=live_row, wallet=float(shadow["wallet"]),
        )) is not None
    }
    missing = sorted(set(live_by_id).difference(shadow_by_id))
    unrecoverable = sorted(
        set(missing).difference(lineage_by_id).difference(persisted_by_id)
    )
    if unrecoverable:
        raise ValueError(
            "live positions lack policy-state lineage in the shadow reference: "
            f"{unrecoverable}"
        )
    kept = []
    execution_state_overlays = 0
    for key in sorted(live_by_id):
        row = shadow_by_id.get(key) or lineage_by_id.get(key) or persisted_by_id[key]
        live_row = live_by_id[key]
        if str(row["symbol"]) != str(live_row.get("symbol")):
            raise ValueError("live/shadow symbol mismatch")
        if str(row.get("side", "long")).lower() != "long":
            raise ValueError("strict-R3 bridge is long-only")
        kept.append(_overlay_live_execution_state(shadow_row=row, live_row=live_row))
        execution_state_overlays += 1

    return {
        "schema": SHADOW_SCHEMA,
        "as_of_ts": decision.isoformat(),
        "wallet": float(shadow["wallet"]),
        "open_positions": kept,
        "bridge_provenance": {
            "live_state": str(live_state_path),
            "live_state_sha256": _sha(live_state_path),
            "shadow_reference": str(shadow_reference_path),
            "shadow_reference_sha256": _sha(shadow_reference_path),
            "live_as_of_ts": live_as_of.isoformat(),
            "matched_positions": len(kept),
            "live_execution_state_overlays": execution_state_overlays,
            "dropped_hypothetical_positions": len(
                set(shadow_by_id).difference(live_by_id)
            ),
            "policy_lineage_recovered_positions": len(missing),
            "policy_lineage_persisted_live_positions": len(
                set(missing).intersection(persisted_by_id)
            ),
            "policy_lineage_reference": (
                str(policy_lineage_reference_path)
                if policy_lineage_reference_path is not None else None
            ),
            "policy_lineage_reference_sha256": (
                _sha(policy_lineage_reference_path)
                if policy_lineage_reference_path is not None else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-state", type=Path, required=True)
    parser.add_argument("--shadow-reference", type=Path, required=True)
    parser.add_argument(
        "--policy-lineage-reference", type=Path,
        help=(
            "Older immutable shadow state used only to recover policy metadata "
            "for an actual exchange position absent from the immediate predecessor."
        ),
    )
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable state bridge already exists: {args.out}")

    live = json.loads(args.live_state.read_text())
    shadow = json.loads(args.shadow_reference.read_text())
    policy_lineage_reference = (
        json.loads(args.policy_lineage_reference.read_text())
        if args.policy_lineage_reference is not None else None
    )
    payload = bridge_state(
        live=live,
        shadow=shadow,
        decision_ts=args.decision_ts,
        live_state_path=args.live_state,
        shadow_reference_path=args.shadow_reference,
        policy_lineage_reference=policy_lineage_reference,
        policy_lineage_reference_path=args.policy_lineage_reference,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
