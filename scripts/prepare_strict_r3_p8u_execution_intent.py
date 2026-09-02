#!/usr/bin/env python3
"""Prepare a hash-verified P8U no-order execution intent from one score commit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_execution_portfolio_adapter import (
    P8UExecutionContract,
    P8UPortfolioState,
    prepare_execution_intent,
    write_execution_intent,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument("--state", type=Path, help="Existing reconciled no-order portfolio state.")
    state_group.add_argument(
        "--initialise-empty-state",
        type=Path,
        help="Create one immutable empty state before preparing the first no-order intent.",
    )
    parser.add_argument(
        "--wallet-equity-quote",
        type=float,
        default=None,
        help="Required only with --initialise-empty-state; never fetched from an exchange.",
    )
    parser.add_argument("--staged-commit", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--now",
        default=None,
        help="UTC evaluation time for stale-signal protection; omit only for offline historical audit.",
    )
    args = parser.parse_args()
    contract = P8UExecutionContract.load(args.contract)
    if args.initialise_empty_state is not None:
        if args.wallet_equity_quote is None:
            parser.error("--wallet-equity-quote is required with --initialise-empty-state")
        state_path = args.initialise_empty_state.resolve()
        if state_path.exists():
            raise FileExistsError(f"refusing to overwrite portfolio state: {state_path}")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = P8UPortfolioState.empty(contract, wallet_equity_quote=float(args.wallet_equity_quote))
        descriptor = os.open(state_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state.payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        assert args.state is not None
        state = P8UPortfolioState.load(args.state, contract)
    now = None if args.now is None else pd.Timestamp(args.now)
    auction, audit, next_state = prepare_execution_intent(
        contract=contract,
        state=state,
        staged_commit=args.staged_commit,
        now=now,
    )
    print(write_execution_intent(out_dir=args.out_dir, auction=auction, audit=audit, next_state=next_state))


if __name__ == "__main__":
    main()
