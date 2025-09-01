#!/usr/bin/env python3
"""
Compare aggregated trades formats between MEXC and Binance to ensure compatibility.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from src.utils.logger import system_logger
import asyncio
import sys
import pandas as pd

from exchange.factory import ExchangeFactory
from src.utils.error_handler import handle_errors

# Add the project root to the Python path
import project_root = Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("AggTradesFormatComparator")

@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="compare_agg_trades_formats",
)
async def compare_agg_trades_formats(symbol: str = "BTCUSDT", lookback_hours: int = 24) -> bool:
    """
    Compare aggregated trades formats between MEXC and Binance.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        lookback_hours: Number of hours to look back

    Returns:
        bool: True if formats match, False otherwise
    """
    logger.info(f"🔍 Comparing aggregated trades formats for {symbol}")

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=lookback_hours)

    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)

    logger.info(f"📅 Time range: {start_time} to {end_time}")

    # Download from both exchanges
    exchanges = {
        "binance": ExchangeFactory.get_exchange("binance"),
        "mexc": ExchangeFactory.get_exchange("mexc"),
    }

    results = {}

    for exchange_name, exchange in exchanges.items():
    pass
    pass
        logger.info(f"📥 Downloading from {exchange_name.upper()}...")
        trades = await exchange.get_historical_agg_trades(
            symbol, start_time_ms=start_time_ms, end_time_ms=end_time_ms, limit=100
        )
        if trades:
    pass
    pass
            df = pd.DataFrame(trades)
            results[exchange_name] = df
            logger.info(
                f"✅ Downloaded {len(trades)} trades from {exchange_name.upper()}",
            )
        else:
            print(warning(f"⚠️ No trades received from {exchange_name.upper()}"))
            results[exchange_name] = pd.DataFrame()

    # Compare formats (columns) between the two exchanges
    if "binance" in results and "mexc" in results:
    pass
    pass
        binance_cols = set(results["binance"].columns)
        mexc_cols = set(results["mexc"].columns)
        if binance_cols == mexc_cols:
    pass
    pass
            logger.info("✅ Column formats match between MEXC and Binance")
            return True
        else:
            missing_in_mexc = binance_cols - mexc_cols
            missing_in_binance = mexc_cols - binance_cols
            if missing_in_mexc:
    pass
    pass
                print(missing(f"❌ Columns missing in MEXC: {sorted(missing_in_mexc)}"))
            if missing_in_binance:
    pass
    pass
                print(missing(f"❌ Columns missing in Binance: {sorted(missing_in_binance)}"))
            return False

    logger.info("⚠️ Could not compare formats due to missing data")
    return False


def main():
    pass
    pass
    parser = argparse.ArgumentParser(description="Compare aggregated trades formats")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--hours", type=int, default=24, help="Lookback hours")
    args = parser.parse_args()

    asyncio.run(compare_agg_trades_formats(args.symbol, args.hours))


if __name__ == "__main__":
    pass
    pass
    main()
