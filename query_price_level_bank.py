#!/usr/bin/env python3
"""
Price Level Bank Query Interface

This script provides a command-line interface for querying the price level bank
and retrieving price levels with their historical tags.

Usage:
    python query_price_level_bank.py --symbol BTCUSDT --top 10
    python query_price_level_bank.py --symbol BTCUSDT --price-range 45000 55000
    python query_price_level_bank.py --symbol BTCUSDT --min-significance 0.7 --format json
    python query_price_level_bank.py --stats
    python query_price_level_bank.py --export --output levels.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import csv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
from tabulate import tabulate

from feature_generation.core.price_level_bank import (
    PriceLevelBank,
    PriceLevelBankConfig,
    get_global_price_level_bank
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PriceLevelBankQuery:
    """Query interface for the price level bank."""

    def __init__(self, bank_path: Optional[str] = None):
        """
        Initialize the query interface.

        Args:
            bank_path: Optional custom path to the bank storage
        """
        if bank_path:
            config = PriceLevelBankConfig(storage_path=bank_path)
            self.bank = PriceLevelBank(config)
        else:
            self.bank = get_global_price_level_bank()

        logger.info("Query interface initialized")

    def query_levels(self,
                    symbol: Optional[str] = None,
                    timeframe: Optional[str] = None,
                    min_price: Optional[float] = None,
                    max_price: Optional[float] = None,
                    min_significance: Optional[float] = None,
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query price levels with various filters.

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_significance: Minimum significance level
            limit: Maximum results to return

        Returns:
            List of level dictionaries
        """
        levels = self.bank.query_levels(
            symbol=symbol,
            timeframe=timeframe,
            min_price=min_price,
            max_price=max_price,
            min_significance=min_significance,
            limit=limit
        )

        # Convert to dictionaries for easy display/serialization
        level_dicts = []
        for level in levels:
            level_dict = level.to_dict()

            # Add derived metrics
            level_dict['total_activity'] = (
                level.historical_crossings +
                level.historical_bounces +
                level.historical_touch_density * 10  # Weighted
            )

            # Add formatted price change from current level
            if symbol:
                current_price = self._get_current_price(symbol)
                if current_price:
                    level_dict['price_change_pct'] = (
                        (level.price - current_price) / current_price * 100
                    )

            level_dicts.append(level_dict)

        return level_dicts

    def get_most_significant(self,
                           symbol: str,
                           timeframe: str,
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most significant price levels.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            top_k: Number of top levels to return

        Returns:
            List of level dictionaries
        """
        levels = self.bank.get_most_significant_levels(symbol, timeframe, top_k)
        return [level.to_dict() for level in levels]

    def get_by_price_range(self,
                          symbol: str,
                          min_price: float,
                          max_price: float,
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get levels within a price range.

        Args:
            symbol: Trading symbol
            min_price: Minimum price
            max_price: Maximum price
            limit: Maximum results

        Returns:
            List of level dictionaries
        """
        levels = self.bank.get_levels_by_price_range(symbol, min_price, max_price, limit)
        return [level.to_dict() for level in levels]

    def get_statistics(self) -> Dict[str, Any]:
        """Get bank statistics."""
        return self.bank.get_statistics()

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.
        This is a placeholder - replace with actual price fetching.
        """
        # Placeholder - in practice, you'd fetch from your exchange API
        # For now, return None to indicate we don't have live data
        return None

    def display_levels(self, levels: List[Dict[str, Any]], format_type: str = 'table'):
        """
        Display levels in the specified format.

        Args:
            levels: List of level dictionaries
            format_type: 'table', 'json', or 'csv'
        """
        if format_type == 'json':
            print(json.dumps(levels, indent=2, default=str))

        elif format_type == 'csv':
            if levels:
                writer = csv.DictWriter(sys.stdout, fieldnames=levels[0].keys())
                writer.writeheader()
                writer.writerows(levels)

        else:  # table format
            if not levels:
                print("No levels found.")
                return

            # Prepare data for table display
            table_data = []
            headers = ['Price', 'Level%', 'Crossings', 'Bounces', 'Volume', 'Density',
                      'Success Rate', 'Significance', 'Session', 'Activity']

            for level in levels:
                table_data.append([
                    f"{level['price']",.2f"}",
                    f"{level['level_pct']".1f"}%",
                    level['historical_crossings'],
                    level['historical_bounces'],
                    f"{level['historical_volume']",.0f"}",
                    f"{level['historical_touch_density']".3f"}",
                    f"{level['historical_success_rate']".2f"}",
                    f"{level['significance_level']".2f"}",
                    level.get('session_type', 'N/A'),
                    f"{level.get('total_activity', 0)".1f"}"
                ])

            print(tabulate(table_data, headers=headers, tablefmt='grid'))

    def display_statistics(self, stats: Dict[str, Any]):
        """Display bank statistics."""
        print("\n" + "="*50)
        print("PRICE LEVEL BANK STATISTICS")
        print("="*50)

        print(f"Total Levels: {stats['total_levels']","}")
        print(f"Symbols: {stats['symbols']}")
        print(f"Price Points: {stats['price_points']","}")
        print(f"Timeframes: {stats['timeframes']}")

        if 'metadata' in stats:
            meta = stats['metadata']
            print(f"Created: {meta.get('created_at', 'N/A')}")
            print(f"Last Updated: {meta.get('last_updated', 'N/A')}")

        print("="*50 + "\n")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Query Price Level Bank')

    # Query arguments
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)')
    parser.add_argument('--top', type=int, help='Get top N most significant levels')
    parser.add_argument('--price-range', type=float, nargs=2,
                       metavar=('MIN', 'MAX'), help='Price range filter')
    parser.add_argument('--min-significance', type=float,
                       help='Minimum significance level (0-1)')
    parser.add_argument('--limit', type=int, default=20,
                       help='Maximum number of results to return')

    # Output options
    parser.add_argument('--format', type=str, default='table',
                       choices=['table', 'json', 'csv'],
                       help='Output format')
    parser.add_argument('--output', type=str,
                       help='Output file (for JSON/CSV formats)')

    # Bank options
    parser.add_argument('--bank-path', type=str,
                       help='Custom path to the price level bank')

    # Actions
    parser.add_argument('--stats', action='store_true',
                       help='Show bank statistics only')
    parser.add_argument('--export', action='store_true',
                       help='Export all levels to file')
    parser.add_argument('--export-file', type=str, default='price_levels.csv',
                       help='Export filename')

    args = parser.parse_args()

    # Initialize query interface
    query = PriceLevelBankQuery(args.bank_path)

    try:
        if args.stats:
            # Show statistics
            stats = query.get_statistics()
            query.display_statistics(stats)

        elif args.export:
            # Export all levels
            logger.info("Exporting all levels...")
            all_levels = []

            # Get unique symbols and timeframes
            stats = query.get_statistics()
            symbols = list(stats.get('symbols', [args.symbol or 'BTCUSDT']))

            for symbol in symbols:
                levels = query.query_levels(symbol=symbol, limit=None)
                all_levels.extend(levels)
                logger.info(f"Exported {len(levels)} levels for {symbol}")

            # Save to file
            if args.export_file.endswith('.json'):
                with open(args.export_file, 'w') as f:
                    json.dump(all_levels, f, indent=2, default=str)
            else:
                with open(args.export_file, 'w', newline='') as f:
                    if all_levels:
                        writer = csv.DictWriter(f, fieldnames=all_levels[0].keys())
                        writer.writeheader()
                        writer.writerows(all_levels)

            logger.info(f"Exported {len(all_levels)} levels to {args.export_file}")

        elif args.top:
            # Get most significant levels
            levels = query.get_most_significant(args.symbol, args.timeframe, args.top)
            query.display_levels(levels, args.format)

        elif args.price_range:
            # Get levels in price range
            min_price, max_price = args.price_range
            levels = query.get_by_price_range(args.symbol, min_price, max_price, args.limit)
            query.display_levels(levels, args.format)

        elif args.symbol:
            # General query
            levels = query.query_levels(
                symbol=args.symbol,
                timeframe=args.timeframe,
                min_significance=args.min_significance,
                limit=args.limit
            )
            query.display_levels(levels, args.format)

        else:
            # Show help
            parser.print_help()
            print("\nExample usage:")
            print("  python query_price_level_bank.py --symbol BTCUSDT --top 10")
            print("  python query_price_level_bank.py --symbol BTCUSDT --price-range 45000 55000")
            print("  python query_price_level_bank.py --stats")
            print("  python query_price_level_bank.py --export --output levels.csv")

    except Exception as e:
        logger.error(f"Query failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()