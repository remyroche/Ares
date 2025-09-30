#!/usr/bin/env python3
"""CLI entry point for NAS/TAS regime analysis."""

import argparse

from src.training.steps.market_analysis.regime_analysis import RegimeAnalysisService
from src.utils.tprint import tprint


def main() -> int:
    """Parse arguments and trigger the regime analysis workflow."""
    parser = argparse.ArgumentParser(description="Analyze regime distribution and metrics")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol to analyze")
    parser.add_argument(
        "--data-cache", default="data_cache", help="Path to data cache directory"
    )
    args = parser.parse_args()

    try:
        service = RegimeAnalysisService(data_cache_path=args.data_cache)
        service.analyze(symbol=args.symbol)
        tprint("🎉 Regime analysis completed successfully!", "SUCCESS")
    except Exception as exc:
        tprint(f"❌ Analysis failed: {exc}", "ERROR")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
