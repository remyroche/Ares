# main_orchestrator.py
import argparse
import os
import signal
import subprocess
import sys
import time

# Import the main CONFIG dictionary
from src.config import CONFIG


def get_process_pid(pid_file):
    """Reads the PID from a specified PID file."""
    if os.path.exists(pid_file):
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            return pid
        except (ValueError, FileNotFoundError):
            return None
    return None


def is_process_running(pid):
    """Checks if a process with the given PID is currently running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # Sends no signal, just checks if process exists
        return True
    except OSError:
        return False


def terminate_process(pid, name="process"):
    """Attempts to gracefully terminate a process by PID, then force kills if necessary."""
    if pid is None:
        print(f"No PID provided to terminate {name}.")
        return False

    print(f"Attempting to terminate {name} (PID: {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)  # Send SIGTERM for graceful shutdown
        time.sleep(5)  # Give it some time to shut down
        if is_process_running(pid):
            print(f"{name} (PID: {pid}) did not terminate gracefully. Force killing...")
            os.kill(pid, signal.SIGKILL)  # Force kill
            time.sleep(2)  # Give it a moment after force kill
        if not is_process_running(pid):
            print(f"{name} (PID: {pid}) terminated.")
            return True
        print(f"Failed to terminate {name} (PID: {pid}).")
        return False
    except ProcessLookupError:
        print(f"{name} (PID: {pid}) was not found, likely already terminated.")
        return True  # Already gone, so consider it terminated
    except Exception as e:
        print(f"Error terminating {name} (PID: {pid}): {e}")
        return False


def start_process(script_name, name="process"):  # Removed log_prefix parameter
    """Starts a Python script as a subprocess, redirecting output to the console."""
    print(f"Starting {name} ({script_name})...")
    try:
        # Redirect stdout/stderr to sys.stdout and sys.stderr to print to the console
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=sys.stdout,  # Redirect to console
            stderr=sys.stderr,  # Redirect to console
            preexec_fn=os.setsid,
        )  # Detach from current process group
        print(f"{name} ({script_name}) started with PID {process.pid}.")
        return process
    except Exception as e:
        print(f"Error starting {name} ({script_name}): {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Ares Main Orchestrator for Backtesting or Live Trading.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=CONFIG["SYMBOL"],
        help=f"Trading symbol (e.g., ETHUSDT). Default: {CONFIG['SYMBOL']}",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange to use (e.g., binance). Default: binance",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="backtest",
        choices=[
            "backtest",
            "train",
            "live",
            "full_backtest",
        ],  # Added 'full_backtest' mode
        help="Operation mode: 'backtest', 'train' (ML/optimization), 'live' (trading bot), or 'full_backtest' (train then backtest). Default: backtest",
    )

    args = parser.parse_args()

    # Override CONFIG values based on command-line arguments for this run
    CONFIG["SYMBOL"] = args.symbol

    # Update filenames based on the new symbol
    CONFIG["KLINES_FILENAME"] = (
        f"data_cache/{CONFIG['SYMBOL']}_{CONFIG['INTERVAL']}_{CONFIG['LOOKBACK_YEARS']}y_klines.csv"
    )
    CONFIG["AGG_TRADES_FILENAME"] = (
        f"data_cache/{CONFIG['SYMBOL']}_{CONFIG['LOOKBACK_YEARS']}y_aggtrades.csv"
    )
    CONFIG["FUTURES_FILENAME"] = (
        f"data_cache/{CONFIG['SYMBOL']}_futures_{CONFIG['LOOKBACK_YEARS']}y_data.csv"
    )
    CONFIG["PREPARED_DATA_FILENAME"] = (
        f"data_cache/{CONFIG['SYMBOL']}_{CONFIG['INTERVAL']}_{CONFIG['LOOKBACK_YEARS']}y_prepared_data.csv"
    )

    # Update WebSocket stream names in live_trading config
    CONFIG["live_trading"]["websocket_streams"]["kline"] = (
        f"{CONFIG['SYMBOL'].lower()}@kline_{CONFIG['INTERVAL']}"
    )
    CONFIG["live_trading"]["websocket_streams"]["aggTrade"] = (
        f"{CONFIG['SYMBOL'].lower()}@aggTrade"
    )
    CONFIG["live_trading"]["websocket_streams"]["depth"] = (
        f"{CONFIG['SYMBOL'].lower()}@depth5@100ms"
    )

    print(
        f"--- Ares Main Orchestrator Starting in '{args.mode}' mode for {args.symbol} on {args.exchange} ---",
    )

    # Clean up any old PID or flag files
    pipeline_pid_file = CONFIG["PIPELINE_PID_FILE"]
    restart_flag_file = CONFIG["RESTART_FLAG_FILE"]

    if os.path.exists(pipeline_pid_file):
        os.remove(pipeline_pid_file)
    if os.path.exists(restart_flag_file):
        os.remove(restart_flag_file)

    # --- Mode-specific execution ---
    if args.mode == "backtest":
        print(f"Launching backtester for {args.symbol}...")
        try:
            from backtesting.ares_backtester import main as run_backtester_main

            run_backtester_main()
        except Exception as e:
            print(f"Error running backtester: {e}")
            sys.exit(1)
        sys.exit(0)

    elif args.mode == "train":
        print(f"Launching Enhanced ML/Fine-tuning pipeline for {args.symbol}...")
        try:
            import asyncio

            from src.database.sqlite_manager import SQLiteManager
            from src.training.enhanced_training_manager import EnhancedTrainingManager

            async def run_enhanced_training():
                # Initialize database manager
                db_manager = SQLiteManager({})
                await db_manager.initialize()

                # Initialize enhanced training manager
                training_manager = EnhancedTrainingManager(db_manager)

                # Run enhanced training pipeline
                session_id = await training_manager.run_full_training(
                    symbol=args.symbol,
                    exchange_name=args.exchange.upper(),
                    timeframe="1h",
                )

                if session_id:
                    print(
                        f"✅ Enhanced training completed successfully for {args.symbol}",
                    )
                else:
                    print(f"❌ Enhanced training failed for {args.symbol}")

                # Close database connection
                await db_manager.close()

            # Run the async training function
            asyncio.run(run_enhanced_training())
        except Exception as e:
            print(f"Error running enhanced training pipeline: {e}")
            sys.exit(1)
        sys.exit(0)

    elif args.mode == "full_backtest":
        print(
            f"--- Running Full Backtest Workflow: Train then Backtest for {args.symbol} ---",
        )

        # Step 1: Run the Enhanced Training Pipeline
        print(
            f"Step 1/2: Launching Enhanced ML/Fine-tuning pipeline for {args.symbol}...",
        )
        try:
            import asyncio

            from src.database.sqlite_manager import SQLiteManager
            from src.training.enhanced_training_manager import EnhancedTrainingManager

            async def run_enhanced_training():
                # Initialize database manager
                db_manager = SQLiteManager({})
                await db_manager.initialize()

                # Initialize enhanced training manager
                training_manager = EnhancedTrainingManager(db_manager)

                # Run enhanced training pipeline
                session_id = await training_manager.run_full_training(
                    symbol=args.symbol,
                    exchange_name=args.exchange.upper(),
                    timeframe="1h",
                )

                if session_id:
                    print(
                        f"✅ Enhanced training completed successfully for {args.symbol}",
                    )
                else:
                    print(f"❌ Enhanced training failed for {args.symbol}")

                # Close database connection
                await db_manager.close()

            # Run the async training function
            asyncio.run(run_enhanced_training())
            print(
                f"Step 1/2: Enhanced training pipeline completed successfully for {args.symbol}.",
            )
        except Exception as e:
            print(f"Error during enhanced training pipeline in full_backtest mode: {e}")
            sys.exit(1)

        # Step 2: Run the Backtester
        print(
            f"Step 2/2: Launching backtester for {args.symbol} with newly trained models...",
        )
        try:
            from backtesting.ares_backtester import main as run_backtester_main

            run_backtester_main()
            print(f"Step 2/2: Backtester completed successfully for {args.symbol}.")
        except Exception as e:
            print(f"Error running backtester in full_backtest mode: {e}")
            sys.exit(1)
        sys.exit(0)

    elif args.mode == "live":
        # ... (live mode logic remains the same) ...
        print(f"Launching live trading bot for {args.symbol}...")
        listener_process = start_process(
            "emails/email_command_listener.py",
            "Email Listener",
        )
        if listener_process is None:
            print("Failed to start Email Listener. Exiting.")
            sys.exit(1)

        pipeline_script_name = CONFIG["PIPELINE_SCRIPT_NAME"]  # src/ares_pipeline.py
        pipeline_process = start_process(pipeline_script_name, "Ares Pipeline")
        if pipeline_process is None:
            print("Failed to start Ares Pipeline. Exiting.")
            terminate_process(listener_process.pid, "Email Listener")
            sys.exit(1)

        try:
            while True:
                if not is_process_running(pipeline_process.pid):
                    print(
                        f"Ares Pipeline (PID: {pipeline_process.pid}) has stopped unexpectedly.",
                    )
                    print("Attempting to restart Ares Pipeline...")
                    pipeline_process = start_process(
                        pipeline_script_name,
                        "Ares Pipeline",
                    )
                    if pipeline_process is None:
                        print("Failed to restart Ares Pipeline. Exiting orchestrator.")
                        break

                if os.path.exists(restart_flag_file):
                    print(
                        f"'{restart_flag_file}' detected. Initiating pipeline restart.",
                    )
                    current_pipeline_pid = get_process_pid(pipeline_pid_file)
                    if current_pipeline_pid and is_process_running(
                        current_pipeline_pid,
                    ):
                        terminate_process(current_pipeline_pid, "Ares Pipeline")
                    else:
                        print(
                            "No active pipeline process found via PID file to stop. Proceeding with restart.",
                        )

                    try:
                        os.remove(restart_flag_file)
                        print(f"'{restart_flag_file}' removed.")
                    except Exception as e:
                        print(f"Error removing restart flag file: {e}")

                    pipeline_process = start_process(
                        pipeline_script_name,
                        "Ares Pipeline",
                    )
                    if pipeline_process is None:
                        print(
                            "Failed to restart Ares Pipeline after flag. Exiting orchestrator.",
                        )
                        break

                time.sleep(10)

        except KeyboardInterrupt:
            print("\nCtrl+C detected. Shutting down orchestrator.")
        except Exception as e:
            print(f"An unexpected error occurred in orchestrator: {e}")
        finally:
            print("--- Orchestrator Shutting Down ---")
            if listener_process and is_process_running(listener_process.pid):
                terminate_process(listener_process.pid, "Email Listener")
            if pipeline_process and is_process_running(pipeline_process.pid):
                terminate_process(pipeline_process.pid, "Ares Pipeline")

            if os.path.exists(pipeline_pid_file):
                os.remove(pipeline_pid_file)
            if os.path.exists(restart_flag_file):
                os.remove(restart_flag_file)
            print("All processes terminated and cleanup complete.")
    else:
        print(
            f"ERROR: Unknown mode '{args.mode}'. Please choose 'backtest', 'train', 'live', or 'full_backtest'.",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
