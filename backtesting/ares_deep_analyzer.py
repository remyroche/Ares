# backtesting/ares_deep_analyzer.py
import pandas as pd
import numpy as np
import random
import logging
from src.utils.logger import system_logger

# Import the main CONFIG dictionary
try:
    from src.config import CONFIG
except ImportError:
    # Provide a default config if the main one isn't available
    CONFIG = {
        "INITIAL_EQUITY": 10000.0,
        "BEST_PARAMS": {"trend_strength_threshold": 25},
        "fees": {"taker": 0.0004, "maker": 0.0002},
    }

from backtesting.ares_data_preparer import (
    load_raw_data,
    calculate_and_label_regimes,
    get_sr_levels,
)
from backtesting.ares_backtester import run_backtest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# --- Analysis Configuration ---
# Walk-Forward Analysis Settings
TRAINING_MONTHS = 12  # Number of months to use for the training set
TESTING_MONTHS = 3  # Number of months for the out-of-sample test set

# Monte Carlo Simulation Settings
MC_SIMULATIONS = 1000  # Number of simulations to run
CONFIDENCE_LEVEL = 0.95  # For reporting confidence intervals


def calculate_detailed_metrics(portfolio, num_days):
    # Access INITIAL_EQUITY from CONFIG
    initial_equity = CONFIG["INITIAL_EQUITY"]
    if not portfolio.trades:
        return {
            "Final Equity": portfolio.equity,
            "Total Trades": 0,
            "Sharpe Ratio": 0,
            "Sortino Ratio": 0,
            "Max Drawdown (%)": 0,
            "Calmar Ratio": 0,
            "Win Rate (%)": 0,
            "Profit Factor": 0,
        }
    trade_df = pd.DataFrame(portfolio.trades)
    trade_df["pnl"] = (
        trade_df["pnl_pct"] * initial_equity
    )  # Use initial_equity from CONFIG
    equity_series = pd.Series(
        [initial_equity] + [t["equity"] for t in portfolio.trades]
    )
    daily_returns = equity_series.pct_change().dropna()

    # Handle cases with no variance in returns
    if daily_returns.std() == 0:
        return {
            "Final Equity": portfolio.equity,
            "Total Trades": len(trade_df),
            "Sharpe Ratio": 0,
            "Sortino Ratio": 0,
            "Max Drawdown (%)": 0,
            "Calmar Ratio": 0,
            "Win Rate (%)": (len(trade_df[trade_df["pnl"] > 0]) / len(trade_df) * 100)
            if len(trade_df) > 0
            else 0,
            "Profit Factor": np.inf
            if trade_df[trade_df["pnl"] > 0]["pnl"].sum() > 0
            and trade_df[trade_df["pnl"] < 0]["pnl"].sum() == 0
            else 0,
        }

    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)

    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std() if not negative_returns.empty else 0
    sortino_ratio = (
        (daily_returns.mean() / downside_std) * np.sqrt(365) if downside_std != 0 else 0
    )

    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = -drawdown.min() * 100

    annual_return = daily_returns.mean() * 365
    calmar_ratio = annual_return / (max_drawdown / 100) if max_drawdown != 0 else 0

    wins = trade_df[trade_df["pnl"] > 0]
    losses = trade_df[trade_df["pnl"] < 0]
    win_rate = len(wins) / len(trade_df) * 100 if len(trade_df) > 0 else 0

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    return {
        "Final Equity": portfolio.equity,
        "Total Trades": len(trade_df),
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
    }


def run_walk_forward_analysis(full_df, params=None):  # Changed default to None
    """Performs a walk-forward analysis on the dataset."""
    report_lines = []
    separator = "=" * 80
    report_lines.append("Stage 1: Walk-Forward Analysis")
    report_lines.append(separator)

    total_months = (full_df.index.max() - full_df.index.min()).days / 30.44
    step_size = TESTING_MONTHS
    num_windows = (
        int((total_months - TRAINING_MONTHS) / step_size)
        if total_months > TRAINING_MONTHS
        else 0
    )

    if num_windows == 0:
        report_lines.append(
            f"Not enough data for a single walk-forward window. Need more than {TRAINING_MONTHS} months."
        )
        system_logger.warning(report_lines[-1])
        return "\n".join(report_lines)

    report_lines.append(f"Dataset covers ~{total_months:.1f} months.")
    report_lines.append(f"Running {num_windows} walk-forward windows...")

    all_metrics = []

    for i in range(num_windows):
        start_date = full_df.index.min() + pd.DateOffset(months=i * step_size)
        training_end_date = start_date + pd.DateOffset(months=TRAINING_MONTHS)
        testing_end_date = training_end_date + pd.DateOffset(months=TESTING_MONTHS)

        testing_set = full_df.loc[training_end_date:testing_end_date]

        if testing_set.empty:
            continue

        report_lines.append(f"\n--- Window {i + 1}/{num_windows} ---")
        report_lines.append(
            f"Testing:  {testing_set.index.min().date()} to {testing_set.index.max().date()}"
        )

        # Pass the params dict to the backtest
        portfolio = run_backtest(testing_set, params)

        num_days = (testing_set.index.max() - testing_set.index.min()).days
        metrics = calculate_detailed_metrics(portfolio, num_days)
        all_metrics.append(metrics)

        report_lines.append("Test Results:")
        for key, value in metrics.items():
            report_lines.append(f"  {key:<20}: {value:.2f}")

    summary_df = pd.DataFrame(all_metrics)
    report_lines.append("\n--- Walk-Forward Analysis Summary ---")
    report_lines.append(summary_df.describe().to_string())

    print("\n".join(report_lines))
    return "\n".join(report_lines)


def run_monte_carlo_simulation(full_df, params=None):  # Changed default to None
    """Runs a Monte Carlo simulation on the full backtest results."""
    report_lines = []
    separator = "=" * 80
    report_lines.append("\n" + separator)
    report_lines.append("Stage 2: Monte Carlo Simulation")
    report_lines.append(separator)

    report_lines.append("Running a full backtest to get the trade log...")
    # Pass the params dict to the backtest
    base_portfolio = run_backtest(full_df, params)

    if not base_portfolio.trades:
        no_trades_msg = "No trades were made in the base backtest. Cannot run Monte Carlo simulation."
        report_lines.append(no_trades_msg)
        print("\n".join(report_lines))
        return None, None, "\n".join(report_lines)

    trade_pnls = [t["pnl_pct"] for t in base_portfolio.trades]

    report_lines.append(
        f"Simulating {MC_SIMULATIONS} equity curves by shuffling {len(trade_pnls)} trades..."
    )

    final_equities = []
    all_simulated_curves = []

    # Access INITIAL_EQUITY from CONFIG
    initial_equity = CONFIG["INITIAL_EQUITY"]

    for _ in range(MC_SIMULATIONS):
        random.shuffle(trade_pnls)
        equity = initial_equity  # Use initial_equity from CONFIG
        equity_curve = [equity]
        for pnl in trade_pnls:
            equity *= 1 + pnl
            equity_curve.append(equity)
        final_equities.append(equity)
        all_simulated_curves.append(equity_curve)

    mean_final_equity = np.mean(final_equities)
    lower_bound = np.percentile(final_equities, (1 - CONFIDENCE_LEVEL) / 2 * 100)
    upper_bound = np.percentile(final_equities, (1 + CONFIDENCE_LEVEL) / 2 * 100)

    report_lines.append("\n--- Monte Carlo Simulation Results ---")
    report_lines.append(f"Original Final Equity: ${base_portfolio.equity:,.2f}")
    report_lines.append(f"Mean Simulated Equity: ${mean_final_equity:,.2f}")
    report_lines.append(
        f"{CONFIDENCE_LEVEL * 100}% Confidence Interval for Final Equity: ${lower_bound:,.2f} - ${upper_bound:,.2f}"
    )

    print("\n".join(report_lines))
    return all_simulated_curves, base_portfolio, "\n".join(report_lines)


def plot_results(mc_curves, base_portfolio):
    """Creates an interactive plot of the Monte Carlo simulation and drawdown."""
    if not mc_curves or base_portfolio is None:
        system_logger.warning("No data to plot for deep analysis.")
        return

    # Access INITIAL_EQUITY from CONFIG
    initial_equity = CONFIG["INITIAL_EQUITY"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Monte Carlo Equity Curves", "Base Strategy Drawdown"),
    )

    # Plot Monte Carlo simulations
    for curve in mc_curves:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(curve))),
                y=curve,
                mode="lines",
                line=dict(color="rgba(173, 216, 230, 0.3)"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Plot original equity curve
    original_curve = [initial_equity] + [t["equity"] for t in base_portfolio.trades]
    fig.add_trace(
        go.Scatter(
            x=list(range(len(original_curve))),
            y=original_curve,
            mode="lines",
            line=dict(color="blue", width=3),
            name="Original Strategy",
        ),
        row=1,
        col=1,
    )

    # Calculate and plot drawdown
    equity_series = pd.Series(original_curve)
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak * 100  # In percent

    fig.add_trace(
        go.Scatter(
            x=list(range(len(drawdown))),
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            line=dict(color="red"),
            name="Drawdown",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="Strategy Deep Analysis",
        yaxis_title_text="Equity ($)",
        yaxis2_title_text="Drawdown (%)",
        xaxis2_title_text="Number of Trades",
        height=800,
    )
    fig.show()


def main():
    """Main function to run the deep analysis."""
    print("--- Ares Strategy Deep Analyzer ---")

    # Load and prepare data once using BEST_PARAMS from config
    klines_df, agg_trades_df, futures_df = load_raw_data()  # Load futures_df here
    if klines_df is None or klines_df.empty:
        system_logger.error("Failed to load data. Halting deep analysis.")
        return

    daily_df = klines_df.resample("D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    sr_levels = get_sr_levels(daily_df)

    print("\nPreparing full dataset using BEST_PARAMS from config.py...")
    # Access BEST_PARAMS from CONFIG
    best_params = CONFIG.get("BEST_PARAMS", {})
    if "trend_strength_threshold" not in best_params:
        best_params["trend_strength_threshold"] = (
            25  # Default value if not in BEST_PARAMS
        )

    # Pass futures_df, best_params, and sr_levels to calculate_and_label_regimes
    prepared_df = calculate_and_label_regimes(
        klines_df,
        agg_trades_df,
        futures_df,
        best_params,
        sr_levels,
        best_params["trend_strength_threshold"],
    )

    # 1. Run Walk-Forward Analysis
    run_walk_forward_analysis(prepared_df, best_params)  # Pass best_params

    # 2. Run Monte Carlo Simulation
    mc_curves, base_portfolio, mc_report = run_monte_carlo_simulation(
        prepared_df, best_params
    )  # Pass best_params

    # 3. Plot the results
    plot_results(mc_curves, base_portfolio)


if __name__ == "__main__":
    main()
