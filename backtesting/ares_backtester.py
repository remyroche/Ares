# backtesting/ares_backtester.py

import asyncio
import os
import sys
import traceback
from typing import Any

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

# Import the email function (optional)
try:
    from emails.ares_mailer import send_email

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

    def send_email(subject, body):
        print(f"Email would be sent: {subject}")
        print(f"Body: {body}")


from src.config import CONFIG
from src.utils.async_utils import async_file_manager, async_processes_manager


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
            result = await async_processes_manager.run_python_script(
                preparer_script_name,
            )
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
                StringIO(csv_content),
                index_col="open_time",
                parse_dates=True,
            )
            print(f"Loaded {len(df)} labeled candles.\n")
            return df
        print(
            f"ERROR: Failed to read prepared data file '{prepared_data_filename}'. Exiting.",
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


class BacktestAnalyst:
    """A simplified analyst that makes decisions based on a single Confidence Score."""

    def __init__(self, params):
        self.params = params
        # Position division parameters (with defaults)
        self.entry_confidence_threshold = params.get("entry_confidence_threshold", 0.7)
        self.additional_position_threshold = params.get("additional_position_threshold", 0.8)
        self.division_confidence_threshold = params.get("division_confidence_threshold", 0.85)
        self.max_division_ratio = params.get("max_division_ratio", 1.0)
        self.max_positions = params.get("max_positions", 3)

    def get_signal(self, prev: pd.Series) -> dict:
        """Get trading signal based on confidence score with position division support."""
        try:
            score = prev.get("Confidence_Score", 0.0)  # Default to 0.0 if missing
        except (KeyError, AttributeError):
            # Create confidence score from available data if Confidence_Score doesn't exist
            if "momentum" in prev:
                score = prev["momentum"] * 1000  # Scale momentum to reasonable confidence range
            else:
                score = 0.0  # Fallback if no momentum data
        
        # Determine signal based on confidence thresholds
        if score > self.entry_confidence_threshold:
            # Determine Stop Loss based on ATR
            sl = prev["close"] - (
                prev["ATR"] * self.params.get("sl_atr_multiplier", 1.5)
            )
            return {
                "direction": 1, 
                "source": "CONFIDENCE_LONG", 
                "sl": sl,
                "confidence": score,
                "entry_type": "primary" if score >= self.entry_confidence_threshold else "additional"
            }

        if score < -self.entry_confidence_threshold:
            sl = prev["close"] + (
                prev["ATR"] * self.params.get("sl_atr_multiplier", 1.5)
            )
            return {
                "direction": -1, 
                "source": "CONFIDENCE_SHORT", 
                "sl": sl,
                "confidence": abs(score),
                "entry_type": "primary" if abs(score) >= self.entry_confidence_threshold else "additional"
            }

        return None


async def run_backtest(df, params=None):
    """
    Runs the full backtest using the new confidence score logic with position division support.
    Improved exit logic to prioritize stop-loss if both SL/TP are hit within the same candle.
    """
    # Use BEST_PARAMS from the main CONFIG
    if params is None:
        params = CONFIG.get("BEST_PARAMS", {})

    portfolio = PortfolioManager(CONFIG.get("INITIAL_EQUITY", 10000))
    analyst = BacktestAnalyst(params)
    
    # Position tracking for multiple entries
    positions = []  # List of active positions
    max_positions = params.get("max_positions", 3)
    entry_confidence_threshold = params.get("entry_confidence_threshold", 0.7)
    additional_position_threshold = params.get("additional_position_threshold", 0.8)

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

            # Check for exit conditions on all positions
            positions_to_remove = []
            for pos_idx, position in enumerate(positions):
                exit_signal = None

                # Check stop loss first (prioritized)
                if (
                    position["direction"] == 1
                    and row["low"] <= position["sl"]
                    or position["direction"] == -1
                    and row["high"] >= position["sl"]
                ):
                    exit_signal = {"reason": "stop_loss", "price": position["sl"]}

                # Check take profit if no stop loss hit
                elif (
                    position["direction"] == 1
                    and row["high"] >= position["tp"]
                    or position["direction"] == -1
                    and row["low"] <= position["tp"]
                ):
                    exit_signal = {"reason": "take_profit", "price": position["tp"]}

                if exit_signal:
                    # Calculate PnL
                    if position["direction"] == 1:
                        pnl = (exit_signal["price"] - position["entry_price"]) / position["entry_price"]
                    else:
                        pnl = (position["entry_price"] - exit_signal["price"]) / position["entry_price"]

                    portfolio.record(pnl, position["source"])
                    positions_to_remove.append(pos_idx)

            # Remove closed positions (in reverse order to maintain indices)
            for pos_idx in reversed(positions_to_remove):
                positions.pop(pos_idx)

            # Check for new entry signals
            if len(positions) < max_positions:
                signal = analyst.get_signal(prev)
                if signal:
                    # Check if we should add this position based on confidence
                    should_add = False
                    
                    # First position or high confidence
                    if len(positions) == 0:
                        should_add = signal["confidence"] >= entry_confidence_threshold
                    # Additional position with higher confidence
                    elif len(positions) > 0:
                        should_add = signal["confidence"] >= additional_position_threshold
                    
                    if should_add:
                        new_position = {
                            "direction": signal["direction"],
                            "entry_price": row["close"],
                            "sl": signal["sl"],
                            "tp": row["close"] + (
                                signal["direction"] * row["ATR"] * params.get("tp_atr_multiplier", 2.0)
                            ),
                            "source": signal["source"],
                            "confidence": signal["confidence"],
                            "entry_type": signal["entry_type"]
                        }
                        positions.append(new_position)

        # Yield control to event loop periodically
        await asyncio.sleep(0)

    return portfolio


async def save_backtest_results(results: dict[str, Any], filename: str):
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
        portfolio = await run_backtest(df)

        # Display results
        print("--- Step 3: Backtest Results ---")
        results = portfolio.report()
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
        if CONFIG.get("SEND_EMAIL_NOTIFICATIONS", False) and EMAIL_AVAILABLE:
            email_body = f"Backtest completed successfully.\nResults: {results}"
            await send_email("Backtest Complete", email_body)

        print("\nBacktest completed successfully!")

    except Exception as e:
        print(f"Backtest failed: {e}")
        traceback.print_exc()

        # Send error notification if configured
        if CONFIG.get("SEND_EMAIL_NOTIFICATIONS", False) and EMAIL_AVAILABLE:
            error_body = f"Backtest failed with error: {e}\n\nTraceback:\n{traceback.format_exc()}"
            await send_email("Backtest Failed", error_body)


if __name__ == "__main__":
    asyncio.run(main())
