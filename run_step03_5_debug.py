#!/usr/bin/env python3
"""
CLI runner for Step03_5 debugging tools.

Examples:
  python run_step03_5_debug.py --symbol ETHUSDT --exchange BINANCE --timeframe 1m --data-dir data_cache
  python run_step03_5_debug.py --only data --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.tools.debug_step03_5 import run as run_debug


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Create a configuration dict compatible with Step 3.5."""
    # Provide both lowercase and uppercase variants for maximum compatibility
    cfg: Dict[str, Any] = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "data_dir": args.data_dir,
        "SYMBOL": args.symbol,
        "EXCHANGE": args.exchange,
        "TIMEFRAME": args.timeframe,
        "DATA_DIR": args.data_dir,
        "regime_clustering": {
            "enable_advanced_reporting": True,
            "enable_regime_analysis": True,
            "enable_transition_analysis": True,
        },
    }

    # Merge optional JSON config
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg_extra = json.load(f)
            cfg.update(cfg_extra)
        except Exception:
            pass

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step03_5 with debugging tools enabled")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol, e.g., ETHUSDT")
    parser.add_argument("--exchange", default="BINANCE", help="Exchange name")
    parser.add_argument("--timeframe", default="1m", help="Timeframe, e.g., 1m, 1h")
    parser.add_argument("--data-dir", default="data_cache", help="Base data directory")
    parser.add_argument("--config", default=None, help="Optional JSON config file to merge")
    parser.add_argument("--only", choices=["data", "hmm", "clustering", "analysis", "reports", "save"], default=None, help="Run only a specific sub-step")
    parser.add_argument("--output-dir", default=None, help="Directory to store debug artifacts")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose DEBUG logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    success = run_debug(cfg, only=args.only, verbose=args.verbose, output_dir=args.output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

