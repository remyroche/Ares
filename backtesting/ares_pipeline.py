import os
import sys
import traceback
import signal  # Import signal module for graceful shutdown
import io  # Import io module for capturing output
from config import INITIAL_EQUITY, PIPELINE_PID_FILE  # Import PID file config
from ares_data_preparer import load_raw_data, get_sr_levels, calculate_and_label_regimes
from ares_optimizer import (
    run_grid_search_stage,
    run_coordinate_descent_stage,
    COARSE_PARAM_GRID,
)
from ares_deep_analyzer import (
    run_walk_forward_analysis,
    run_monte_carlo_simulation,
    plot_results,
)
from ares_mailer import send_email


def write_pid_file():
    """Writes the current process ID to a file."""
    try:
        with open(PIPELINE_PID_FILE, "w") as f:
            f.write(str(os.getpid()))
        print(f"PID {os.getpid()} written to {PIPELINE_PID_FILE}")
    except Exception as e:
        print(f"Error writing PID file: {e}")


def remove_pid_file():
    """Removes the PID file."""
    try:
        if os.path.exists(PIPELINE_PID_FILE):
            os.remove(PIPELINE_PID_FILE)
            print(f"PID file {PIPELINE_PID_FILE} removed.")
    except Exception as e:
        print(f"Error removing PID file: {e}")


def signal_handler(signum, frame):
    """Handles signals for graceful shutdown."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    remove_pid_file()
    sys.exit(0)


def main():
    """
    Main execution pipeline for the entire Ares project.
    This script orchestrates the download, optimization, and analysis process.
    """
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    # Write PID file at the very beginning
    write_pid_file()

    # Capture stdout and stderr to a string buffer
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_output

    email_subject = "Ares Pipeline FAILED"
    email_body = "The pipeline encountered an unexpected error."
    full_report = ["Ares Automated Pipeline Report"]
    separator = "=" * 80
    full_report.append(separator)

    try:
        # --- STAGE 1: Data Preparation ---
        print(separator)
        print("PIPELINE STAGE 1: DATA PREPARATION")
        print(separator)
        # CORRECTED: Load all three data sources
        klines_df, agg_trades_df, futures_df = load_raw_data()
        if klines_df is None:
            raise Exception("Failed to load raw data. Halting pipeline.")

        daily_df = klines_df.resample("D").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        sr_levels = get_sr_levels(daily_df)
        print("Data preparation complete.\n")
        send_email(
            "Ares Pipeline: Stage 1 Complete",
            "Data download and preparation has finished successfully.",
        )

        # --- STAGE 2: Coarse Optimization ---
        print(separator)
        print("PIPELINE STAGE 2: COARSE PARAMETER OPTIMIZATION")
        print(separator)
        # Pass futures_df and sr_levels to the optimization stage
        coarse_results = run_grid_search_stage(
            COARSE_PARAM_GRID,
            klines_df,
            agg_trades_df,
            futures_df,
            sr_levels,
            "Coarse Grid Search",
        )

        if (
            not coarse_results
            or coarse_results[0]["portfolio"].equity <= INITIAL_EQUITY
        ):
            raise Exception(
                "Coarse search did not yield any profitable results. Halting pipeline."
            )

        best_coarse_params = coarse_results[0]["params"]
        coarse_report = f"Best result from Coarse Search: ${coarse_results[0]['portfolio'].equity:,.2f}\nBest Coarse Params: {best_coarse_params}"
        print(coarse_report)
        full_report.append("STAGE 2: COARSE OPTIMIZATION RESULTS")
        full_report.append(coarse_report + "\n")
        send_email("Ares Pipeline: Stage 2 Complete", coarse_report)

        # --- STAGE 3: Fine-Tuning Optimization ---
        print(separator)
        print("PIPELINE STAGE 3: COORDINATE DESCENT FINE-TUNING")
        print(separator)
        # Pass futures_df and sr_levels to the optimization stage
        final_results = run_coordinate_descent_stage(
            best_coarse_params, klines_df, agg_trades_df, futures_df, sr_levels
        )

        best_params = final_results[0]["params"]
        final_portfolio = final_results[0]["portfolio"]

        fine_tuning_report = f"Best result from Fine-Tuning: ${final_portfolio.equity:,.2f}\nFinal Tuned Params: {best_params}"
        print(fine_tuning_report)
        full_report.append("STAGE 3: FINE-TUNING RESULTS")
        full_report.append(fine_tuning_report + "\n")
        send_email("Ares Pipeline: Stage 3 Complete", fine_tuning_report)

        # --- STAGE 4: Deep Analysis & Validation ---
        print(separator)
        print("PIPELINE STAGE 4: DEEP ANALYSIS & VALIDATION")
        print(separator)

        print("Re-preparing data with final optimized parameters for analysis...")
        # Now, trend_strength_threshold is part of best_params
        final_trend_threshold = best_params.get(
            "trend_strength_threshold"
        )  # Extract from params

        # Pass futures_df, best_params, and sr_levels to the final data preparation
        final_prepared_df = calculate_and_label_regimes(
            klines_df.copy(),
            agg_trades_df.copy(),
            futures_df.copy(),
            best_params,
            sr_levels,
            final_trend_threshold,
        )

        # The analyzer now needs the full params dict to run the backtest correctly
        wfa_report = run_walk_forward_analysis(final_prepared_df, best_params)
        mc_curves, mc_base_portfolio, mc_report = run_monte_carlo_simulation(
            final_prepared_df, best_params
        )

        plot_results(mc_curves, mc_base_portfolio)

        full_report.append("STAGE 4: DEEP ANALYSIS RESULTS")
        full_report.append(wfa_report)
        full_report.append(mc_report)

        # --- FINAL REPORT ---
        email_subject = (
            f"Ares Pipeline Complete - Final Equity: ${final_portfolio.equity:,.2f}"
        )
        email_body = "\n".join(full_report)

    except Exception as e:
        # Capture the current log output when an error occurs
        captured_logs = redirected_output.getvalue()
        print(
            f"\nAn error occurred during the pipeline: {e}", file=old_stderr
        )  # Print to original stderr
        email_body = f"An exception occurred during the pipeline process:\n\n{traceback.format_exc()}\n\n--- Pipeline Logs ---\n{captured_logs}"
        raise  # Re-raise the exception to be caught by the orchestrator

    finally:
        # Restore original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        print("\n" + separator)
        print("PIPELINE FINISHED. Sending final email report...")
        print(separator)
        send_email(email_subject, email_body)
        remove_pid_file()  # Ensure PID file is removed on normal exit or exception
        sys.exit(0)  # Explicitly exit to ensure the process terminates after email


if __name__ == "__main__":
    main()
