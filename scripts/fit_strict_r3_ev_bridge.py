#!/usr/bin/env python3
"""Fit an immutable strict-OOF common-bps EV bridge for strict-R3.

The bridge is intentionally trained before the period it will admit.  It
standardises the same-producer prior-42 CDF into a common expected policy-net
bps prior and leaves short-horizon adaptation to the live residual correction.
It is not a cross-producer raw-score map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_ev_bridge import (  # noqa: E402
    EVBridgeSpec,
    fit_strict_r3_ev_bridge,
    persist_strict_r3_ev_bridge,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-oof-ledger", type=Path, required=True)
    parser.add_argument(
        "--fit-cutoff", required=True,
        help="UTC cutoff; only policy labels resolved strictly before it can fit the bridge.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prior-bins", type=int, default=20)
    parser.add_argument("--minimum-residual-rows", type=int, default=20)
    parser.add_argument("--net-floor-bps", type=float, default=50.0)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable EV bridge output already exists: {args.out_dir}")
    ledger = pd.read_parquet(args.strict_oof_ledger)
    spec = EVBridgeSpec(
        prior_bins=args.prior_bins,
        minimum_residual_rows=args.minimum_residual_rows,
        net_floor_bps=args.net_floor_bps,
    )
    bundle = fit_strict_r3_ev_bridge(ledger, fit_cutoff=args.fit_cutoff, spec=spec)
    manifest = dict(persist_strict_r3_ev_bridge(bundle, args.out_dir))
    manifest.update({
        "strict_oof_ledger": str(args.strict_oof_ledger),
        "strict_oof_ledger_sha256": _sha(args.strict_oof_ledger),
    })
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "event": "complete",
        "out_dir": str(args.out_dir),
        "fit_cutoff": str(args.fit_cutoff),
        "side_maps": sorted(bundle.side_maps),
        "bundle_sha256": manifest["bundle_sha256"],
    }))


if __name__ == "__main__":
    main()
