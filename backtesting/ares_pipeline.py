import io  # Import io module for capturing output
import os
import sys
import traceback

from ares_data_preparer import get_sr_levels, load_raw_data
from ares_deep_analyzer import (
    plot_results,
    run_monte_carlo_simulation,
    run_walk_forward_analysis,
)
from ares_mailer import send_email
from ares_optimizer import (
    COARSE_PARAM_GRID,
    run_coordinate_descent_stage,
    run_grid_search_stage,
)

from config import INITIAL_EQUITY, PIPELINE_PID_FILE  # Import PID file config

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.signal_handler import GracefulShutdown


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


def main():
    """
    Main execution pipeline for the entire Ares project.
    This script orchestrates the download, optimization, and analysis process.
    """
    # Use centralized signal handling
    with GracefulShutdown("BacktestingPipeline") as signal_handler:
        # Add cleanup callbacks
        signal_handler.add_shutdown_callback(remove_pid_file)

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
                },
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
                    "Coarse search did not yield any profitable results. Halting pipeline.",
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
                best_coarse_params,
                klines_df,
                agg_trades_df,
                futures_df,
                sr_levels,
            )

            best_params = final_results[0]["params"]
            final_portfolio = final_results[0]["portfolio"]

            fine_tuning_report = f"Best result from Fine-Tuning: ${final_portfolio.equity:,.2f}\nFinal Tuned Params: {best_params}"
            print(fine_tuning_report)
            full_report.append("STAGE 3: FINE-TUNING RESULTS")
            full_report.append(fine_tuning_report + "\n")
            send_email("Ares Pipeline: Stage 3 Complete", fine_tuning_report)

            # --- STAGE 4: Walk-Forward Analysis ---
            print(separator)
            print("PIPELINE STAGE 4: WALK-FORWARD VALIDATION")
            print(separator)
            walk_forward_results = run_walk_forward_analysis(
                klines_df,
                agg_trades_df,
                futures_df,
                best_params,
                sr_levels,
            )

            if walk_forward_results:
                walk_forward_report = f"Walk-Forward Analysis Complete\nAverage Performance: ${walk_forward_results['avg_equity']:,.2f}\nConsistency Score: {walk_forward_results['consistency_score']:.2f}"
                print(walk_forward_report)
                full_report.append("STAGE 4: WALK-FORWARD VALIDATION RESULTS")
                full_report.append(walk_forward_report + "\n")
                send_email("Ares Pipeline: Stage 4 Complete", walk_forward_report)
            else:
                print("Walk-Forward Analysis failed or produced no results.")
                full_report.append("STAGE 4: WALK-FORWARD VALIDATION FAILED\n")

            # --- STAGE 5: Monte Carlo Simulation ---
            print(separator)
            print("PIPELINE STAGE 5: MONTE CARLO SIMULATION")
            print(separator)
            monte_carlo_results = run_monte_carlo_simulation(
                klines_df,
                agg_trades_df,
                futures_df,
                best_params,
                sr_levels,
            )

            if monte_carlo_results:
                mc_report = f"Monte Carlo Simulation Complete\nConfidence Interval: {monte_carlo_results['confidence_interval']}\nRisk Assessment: {monte_carlo_results['risk_assessment']}"
                print(mc_report)
                full_report.append("STAGE 5: MONTE CARLO SIMULATION RESULTS")
                full_report.append(mc_report + "\n")
                send_email("Ares Pipeline: Stage 5 Complete", mc_report)
            else:
                print("Monte Carlo Simulation failed or produced no results.")
                full_report.append("STAGE 5: MONTE CARLO SIMULATION FAILED\n")

            # --- STAGE 6: Results Visualization ---
            print(separator)
            print("PIPELINE STAGE 6: RESULTS VISUALIZATION")
            print(separator)
            plot_results(
                klines_df,
                agg_trades_df,
                futures_df,
                best_params,
                sr_levels,
                final_results,
                walk_forward_results,
                monte_carlo_results,
            )
            print("Results visualization complete.\n")
            full_report.append("STAGE 6: RESULTS VISUALIZATION COMPLETE\n")

            # --- SUCCESS EMAIL ---
            email_subject = "Ares Pipeline SUCCESS"
            email_body = f"""
            Ares Pipeline completed successfully!
            
            Final Portfolio Value: ${final_portfolio.equity:,.2f}
            Best Parameters: {best_params}
            
            All stages completed without errors.
            """
            send_email(email_subject, email_body)

            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Print final summary
            print(separator)
            print("🎉 ARES PIPELINE COMPLETED SUCCESSFULLY!")
            print(separator)
            print("\n".join(full_report))
            print(separator)
            print("📊 Results saved to reports/ directory")
            print("📈 Charts and visualizations generated")
            print("📧 Success email sent")
            print(separator)

        except Exception as e:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Capture the error details
            error_details = redirected_output.getvalue()
            traceback_str = traceback.format_exc()

            print(separator)
            print("❌ ARES PIPELINE FAILED!")
            print(separator)
            print(f"Error: {e}")
            print(f"Traceback:\n{traceback_str}")
            print(f"Output captured:\n{error_details}")
            print(separator)

            # Send failure email
            email_body = f"""
            Ares Pipeline failed!
            
            Error: {e}
            
            Traceback:
            {traceback_str}
            
            Captured Output:
            {error_details}
            """
            send_email(email_subject, email_body)

            # Re-raise the exception
            raise


if __name__ == "__main__":
    main()
