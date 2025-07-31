# backtesting/ares_backtester.py

import pandas as pd
import numpy as np
import os
import sys
import traceback
import asyncio
from typing import Dict, Any
from src.analyst.analyst import Analyst
from src.config import CONFIG
from emails.ares_mailer import send_email  # Import the email function
from src.utils.async_utils import async_file_manager, async_process_manager


async def load_prepared_data():
    """Loads prepared data asynchronously. If the file is missing, calls the preparer script."""
    print("--- Step 1: Loading Prepared Data ---")
    # Access config values through the CONFIG dictionary
    prepared_data_filename = CONFIG["data_path"]
    preparer_script_name = "backtesting/ares_data_preparer.py"

    if not os.path.exists(prepared_data_filename):
        print(f"Prepared data file '{prepared_data_filename}' not found.")
        print(f"Calling data preparer script: '{preparer_script_name}'...")
        try:
            # Use async process manager to run the preparer script
            result = await async_process_manager.run_python_script(preparer_script_name)
            if result and result.get("success"):
                print("Data preparer script finished successfully.")
            else:
                print(f"Data preparer script failed: {result}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: The data preparer script failed with an error: {e}")
            sys.exit(1)

    try:
        # Use async file manager to read the CSV file
        csv_content = await async_file_manager.read_file(prepared_data_filename)
        if csv_content:
            # Parse CSV content
            from io import StringIO

            df = pd.read_csv(
                StringIO(csv_content), index_col="open_time", parse_dates=True
            )
            print(f"Loaded {len(df)} labeled candles.\n")
            return df
        else:
            print(
                f"ERROR: Failed to read prepared data file '{prepared_data_filename}'. Exiting."
            )
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load prepared data: {e}")
        sys.exit(1)


class PortfolioManager:
    """Handles equity tracking and performance reporting."""

    def __init__(self, equity):
        self.equity = equity
        self.trades = []

    def record(self, pnl, source):
        self.equity *= 1 + pnl
        self.trades.append({"pnl_pct": pnl, "source": source, "equity": self.equity})

    def report(self):
        df = pd.DataFrame(self.trades)
        if df.empty:
            return {}

        report = {}
        for src in df["source"].unique():
            subset = df[df["source"] == src]["pnl_pct"]
            # Sharpe Ratio assumes daily returns, adjust for H1 data
            sharpe = (
                (subset.mean() / subset.std()) * np.sqrt(365 * 24)
                if len(subset) > 1 and subset.std() > 0
                else 0
            )
            report[src] = {
                "num_trades": len(subset),
                "avg_profit_pct": subset.mean() * 100,
                "sharpe": sharpe,
            }
        return report


class Analyst:
    """A simplified analyst that makes decisions based on a single Confidence Score."""

    def __init__(self, params):
        self.params = params

    def get_signal(self, prev):
        """Generates a trade signal if the confidence score crosses the entry threshold."""
        score = prev["Confidence_Score"]
        threshold = self.params.get("trade_entry_threshold", 0.6)

        if score > threshold:
            # Determine Stop Loss based on ATR
            sl = prev["close"] - (
                prev["ATR"] * self.params.get("sl_atr_multiplier", 1.5)
            )
            return {"direction": 1, "source": "CONFIDENCE_LONG", "sl": sl}

        if score < -threshold:
            sl = prev["close"] + (
                prev["ATR"] * self.params.get("sl_atr_multiplier", 1.5)
            )
            return {"direction": -1, "source": "CONFIDENCE_SHORT", "sl": sl}

        return None


async def run_backtest(df, params=None):
    """
    Runs the full backtest using the new confidence score logic.
    Improved exit logic to prioritize stop-loss if both SL/TP are hit within the same candle.
    """
    # Use BEST_PARAMS from the main CONFIG
    if params is None:
        params = CONFIG.get("BEST_PARAMS", {})

    portfolio = PortfolioManager(CONFIG.get("INITIAL_EQUITY", 10000))
    analyst = Analyst(params)
    position, entry_price, trade_info = 0, 0, {}

    # Process data in chunks for better performance
    chunk_size = 1000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]

        for idx, row in chunk.iterrows():
            prev = (
                df.iloc[max(0, df.index.get_loc(idx) - 1)]
                if df.index.get_loc(idx) > 0
                else row
            )

            # Check for exit conditions if in position
            if position != 0:
                exit_signal = None

                # Check stop loss first (prioritized)
                if position == 1 and row["low"] <= trade_info["sl"]:
                    exit_signal = {"reason": "stop_loss", "price": trade_info["sl"]}
                elif position == -1 and row["high"] >= trade_info["sl"]:
                    exit_signal = {"reason": "stop_loss", "price": trade_info["sl"]}

                # Check take profit if no stop loss hit
                elif position == 1 and row["high"] >= trade_info["tp"]:
                    exit_signal = {"reason": "take_profit", "price": trade_info["tp"]}
                elif position == -1 and row["low"] <= trade_info["tp"]:
                    exit_signal = {"reason": "take_profit", "price": trade_info["tp"]}

                if exit_signal:
                    # Calculate PnL
                    if position == 1:
                        pnl = (exit_signal["price"] - entry_price) / entry_price
                    else:
                        pnl = (entry_price - exit_signal["price"]) / entry_price

                    portfolio.record(pnl, trade_info["source"])
                    position, entry_price, trade_info = 0, 0, {}

            # Check for new entry signal if not in position
            if position == 0:
                signal = analyst.get_signal(prev)
                if signal:
                    position = signal["direction"]
                    entry_price = row["close"]
                    trade_info = {
                        "source": signal["source"],
                        "sl": signal["sl"],
                        "tp": row["close"]
                        + (
                            position * row["ATR"] * params.get("tp_atr_multiplier", 2.0)
                        ),
                    }

        # Yield control to event loop periodically
        await asyncio.sleep(0)

    return portfolio.report()


async def save_backtest_results(results: Dict[str, Any], filename: str):
    """Save backtest results asynchronously"""
    try:
        results_json = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "results": results,
            "config": CONFIG,
        }
        await async_file_manager.write_json(filename, results_json)
        print(f"Backtest results saved to {filename}")
    except Exception as e:
        print(f"Failed to save backtest results: {e}")


async def main():
    """Main async function to run the backtest"""
    try:
        # Load prepared data
        df = await load_prepared_data()

        # Run backtest
        print("--- Step 2: Running Backtest ---")
        results = await run_backtest(df)

        # Display results
        print("--- Step 3: Backtest Results ---")
        for strategy, metrics in results.items():
            print(f"\n{strategy}:")
            print(f"  Number of trades: {metrics['num_trades']}")
            print(f"  Average profit: {metrics['avg_profit_pct']:.2f}%")
            print(f"  Sharpe ratio: {metrics['sharpe']:.2f}")

        # Save results
        results_file = (
            f"backtest_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        await save_backtest_results(results, results_file)

        # Send email notification if configured
        if CONFIG.get("SEND_EMAIL_NOTIFICATIONS", False):
            email_body = f"Backtest completed successfully.\nResults: {results}"
            await send_email("Backtest Complete", email_body)

        print("\nBacktest completed successfully!")

    except Exception as e:
        print(f"Backtest failed: {e}")
        traceback.print_exc()

        # Send error notification if configured
        if CONFIG.get("SEND_EMAIL_NOTIFICATIONS", False):
            error_body = f"Backtest failed with error: {e}\n\nTraceback:\n{traceback.format_exc()}"
            await send_email("Backtest Failed", error_body)


if __name__ == "__main__":
    asyncio.run(main())
