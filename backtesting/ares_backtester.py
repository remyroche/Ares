import pandas as pd
import numpy as np
import os
import subprocess
import sys
import traceback
from config import (
    INITIAL_EQUITY, 
    PREPARED_DATA_FILENAME, PREPARER_SCRIPT_NAME,
    BEST_PARAMS
)
from ares_mailer import send_email # Import the email function

def load_prepared_data():
    """Loads prepared data. If the file is missing, calls the preparer script."""
    print("--- Step 1: Loading Prepared Data ---")
    if not os.path.exists(PREPARED_DATA_FILENAME):
        print(f"Prepared data file '{PREPARED_DATA_FILENAME}' not found.")
        print(f"Calling data preparer script: '{PREPARER_SCRIPT_NAME}'...")
        try:
            subprocess.run([sys.executable, PREPARER_SCRIPT_NAME], check=True)
            print("Data preparer script finished.")
        except FileNotFoundError:
            print(f"ERROR: The script '{PREPARER_SCRIPT_NAME}' was not found.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: The data preparer script failed with an error: {e}")
            sys.exit(1)
    try:
        df = pd.read_csv(PREPARED_DATA_FILENAME, index_col='open_time', parse_dates=True)
        print(f"Loaded {len(df)} labeled candles.\n")
        return df
    except FileNotFoundError:
        print(f"ERROR: Prepared data file '{PREPARED_DATA_FILENAME}' still not found. Exiting.")
        sys.exit(1)

class PortfolioManager:
    """Handles equity tracking and performance reporting."""
    def __init__(self, equity):
        self.equity = equity
        self.trades = []

    def record(self, pnl, source):
        self.equity *= (1 + pnl)
        self.trades.append({'pnl_pct': pnl, 'source': source, 'equity': self.equity})

    def report(self):
        df = pd.DataFrame(self.trades)
        if df.empty: return {}
        
        report = {}
        for src in df['source'].unique():
            subset = df[df['source'] == src]['pnl_pct']
            sharpe = (subset.mean() / subset.std()) * np.sqrt(252 * 24 * 60) if len(subset) > 1 and subset.std() > 0 else 0
            report[src] = {
                'num_trades': len(subset), 
                'avg_profit_pct': subset.mean() * 100, 
                'sharpe': sharpe
            }
        return report

class Analyst:
    """A simplified analyst that makes decisions based on a single Confidence Score."""
    def __init__(self, params):
        self.params = params

    def get_signal(self, prev):
        """Generates a trade signal if the confidence score crosses the entry threshold."""
        score = prev['Confidence_Score']
        threshold = self.params.get('trade_entry_threshold', 0.6)

        if score > threshold:
            # Determine Stop Loss based on ATR
            sl = prev['close'] - (prev['ATR'] * self.params.get('sl_atr_multiplier', 1.5))
            return {'direction': 1, 'source': 'CONFIDENCE_LONG', 'sl': sl}
        
        if score < -threshold:
            sl = prev['close'] + (prev['ATR'] * self.params.get('sl_atr_multiplier', 1.5))
            return {'direction': -1, 'source': 'CONFIDENCE_SHORT', 'sl': sl}
            
        return None

def run_backtest(df, params=BEST_PARAMS):
    """
    Runs the full backtest using the new confidence score logic.
    Improved exit logic to prioritize stop-loss if both SL/TP are hit within the same candle.
    """
    portfolio = PortfolioManager(INITIAL_EQUITY)
    analyst = Analyst(params)
    position, entry_price, trade_info = 0, 0, {}
    
    dict_records = df.reset_index().to_dict('records')
    
    for i in range(1, len(dict_records)):
        row, prev = dict_records[i], dict_records[i-1]
        
        if position != 0: # Check for exit
            sl, tp = trade_info['sl'], trade_info['tp']
            exit_price = 0
            
            # Check for Stop Loss and Take Profit
            if position == 1: # Long position
                # Check if price went below Stop Loss
                sl_hit = row['low'] <= sl
                # Check if price went above Take Profit
                tp_hit = row['high'] >= tp

                if sl_hit:
                    # If Stop Loss is hit, exit at Stop Loss price
                    exit_price = sl
                elif tp_hit:
                    # If only Take Profit is hit, exit at Take Profit price
                    exit_price = tp
            elif position == -1: # Short position
                # Check if price went above Stop Loss
                sl_hit = row['high'] >= sl
                # Check if price went below Take Profit
                tp_hit = row['low'] <= tp

                if sl_hit:
                    # If Stop Loss is hit, exit at Stop Loss price
                    exit_price = sl
                elif tp_hit:
                    # If only Take Profit is hit, exit at Take Profit price
                    exit_price = tp

            if exit_price > 0: # If an exit condition was met
                pnl = (exit_price - entry_price) / entry_price if position == 1 else (entry_price - exit_price) / entry_price
                portfolio.record(pnl, trade_info['source'])
                position = 0 # Reset position after exit
                continue # Move to next candle

        if position == 0: # Check for entry
            signal = analyst.get_signal(prev)
            if signal:
                pos, src, sl = signal['direction'], signal['source'], signal['sl']
                entry = row['open'] # Enter on the next candle's open
                risk = abs(entry - sl)
                
                if risk > 0:
                    rr = params.get('take_profit_rr', 2.0)
                    tp = entry + (risk * rr) if pos == 1 else entry - (risk * rr)
                    position, entry_price = pos, entry
                    trade_info = {'sl': sl, 'tp': tp, 'source': src}
    return portfolio


def main():
    """Main execution function for the backtester."""
    email_subject = "Ares Backtester FAILED"
    email_body = "The script encountered an unexpected error."
    
    try:
        prepared_df = load_prepared_data()
        portfolio = run_backtest(prepared_df)

        report_lines = []
        separator = "="*80
        
        report_lines.append("Ares Full Backtest Performance Report")
        report_lines.append(separator)
        report_lines.append(f"\nFinal Equity: ${portfolio.equity:,.2f}")
        report_lines.append(f"Total Trades: {len(portfolio.trades)}\n")
        
        report = portfolio.report()
        sorted_report = sorted(report.items(), key=lambda item: item[1].get('num_trades', 0), reverse=True)

        report_lines.append(f"{'Trade Source':<25} | {'Num Trades':>12} | {'Sharpe Ratio':>15} | {'Avg Profit/Loss (%)':>20}")
        report_lines.append("-" * 80)
        for source, metrics in sorted_report:
            report_lines.append(f"{source:<25} | {metrics['num_trades']:>12} | {metrics['sharpe']:>15.2f} | {metrics['avg_profit_pct']:>20.2f}%")
        
        report_string = "\n".join(report_lines)
        print("\n" + report_string)

        email_subject = f"Ares Backtest Complete - Final Equity: ${portfolio.equity:,.2f}"
        email_body = report_string

    except Exception as e:
        print(f"\nAn error occurred during the backtest: {e}")
        email_body = f"An exception occurred during the backtest process:\n\n{traceback.format_exc()}"
        raise
        
    finally:
        send_email(email_subject, email_body)

if __name__ == "__main__":
    main()
