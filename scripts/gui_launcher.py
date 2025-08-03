#!/usr/bin/env python3
"""
GUI Launcher for Ares Trading Bot
Automatically launches the GUI when the bot is started in various modes.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GUILauncher:
    """Manages the automatic launch of the GUI with the trading bot."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.gui_dir = self.project_root / "GUI"
        self.api_server_process: subprocess.Popen | None = None
        self.frontend_process: subprocess.Popen | None = None
        self.gui_started = False
        self.shutdown_event = threading.Event()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down GUI...")
        self.shutdown_event.set()
        self.stop_gui()
        sys.exit(0)

    def check_gui_dependencies(self) -> bool:
        """Check if GUI dependencies are installed."""
        try:
            # Check if Node.js and npm are available
            subprocess.run(["node", "--version"], check=True, capture_output=True)
            subprocess.run(["npm", "--version"], check=True, capture_output=True)

            # Check if GUI directory exists
            if not self.gui_dir.exists():
                logger.error(f"GUI directory not found at {self.gui_dir}")
                return False

            # Check if package.json exists
            package_json = self.gui_dir / "package.json"
            if not package_json.exists():
                logger.error(f"package.json not found in {self.gui_dir}")
                return False

            # Check if Python dependencies are available
            # Note: FastAPI and Uvicorn are not required for basic GUI functionality
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Missing system dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return False

    def install_gui_dependencies(self) -> bool:
        """Install GUI dependencies if needed."""
        try:
            logger.info("Installing GUI dependencies...")

            # Change to GUI directory
            os.chdir(self.gui_dir)

            # Install npm dependencies
            result = subprocess.run(
                ["npm", "install"],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,  # 5 minutes timeout
            )

            if result.returncode != 0:
                logger.error(f"Failed to install npm dependencies: {result.stderr}")
                return False

            logger.info("GUI dependencies installed successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Timeout installing GUI dependencies")
            return False
        except Exception as e:
            logger.error(f"Error installing GUI dependencies: {e}")
            return False
        finally:
            # Return to project root
            os.chdir(self.project_root)

    def start_api_server(self) -> bool:
        """Start the FastAPI backend server."""
        try:
            logger.info("Starting API server...")

            # Change to GUI directory
            os.chdir(self.gui_dir)

            # Start the API server
            self.api_server_process = subprocess.Popen(
                [sys.executable, "api_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait a moment for the server to start
            time.sleep(3)

            # Check if the server is running
            if self.api_server_process.poll() is None:
                logger.info("API server started successfully")
                return True
            stdout, stderr = self.api_server_process.communicate()
            logger.error(f"API server failed to start: {stderr}")
            return False

        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            return False
        finally:
            # Return to project root
            os.chdir(self.project_root)

    def start_frontend(self) -> bool:
        """Start the React frontend development server."""
        try:
            logger.info("Starting frontend development server...")

            # Change to GUI directory
            os.chdir(self.gui_dir)

            # Start the frontend development server
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait a moment for the server to start
            time.sleep(5)

            # Check if the server is running
            if self.frontend_process.poll() is None:
                logger.info("Frontend server started successfully")
                return True
            stdout, stderr = self.frontend_process.communicate()
            logger.error(f"Frontend server failed to start: {stderr}")
            return False

        except Exception as e:
            logger.error(f"Error starting frontend server: {e}")
            return False
        finally:
            # Return to project root
            os.chdir(self.project_root)

    def start_gui(self) -> bool:
        """Start both API server and frontend."""
        try:
            logger.info("Starting GUI components...")

            # Start API server
            if not self.start_api_server():
                return False

            # Start frontend
            if not self.start_frontend():
                self.stop_gui()
                return False

            self.gui_started = True
            logger.info("GUI started successfully!")
            logger.info("Frontend available at: http://localhost:3000")
            logger.info("API documentation at: http://localhost:8000/docs")

            return True

        except Exception as e:
            logger.error(f"Error starting GUI: {e}")
            return False

    def stop_gui(self):
        """Stop both API server and frontend."""
        logger.info("Stopping GUI components...")

        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=10)
                logger.info("Frontend server stopped")
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                logger.warning("Force killed frontend server")
            except Exception as e:
                logger.error(f"Error stopping frontend: {e}")

        if self.api_server_process:
            try:
                self.api_server_process.terminate()
                self.api_server_process.wait(timeout=10)
                logger.info("API server stopped")
            except subprocess.TimeoutExpired:
                self.api_server_process.kill()
                logger.warning("Force killed API server")
            except Exception as e:
                logger.error(f"Error stopping API server: {e}")

        self.gui_started = False

    def monitor_gui(self):
        """Monitor GUI processes and restart if needed."""
        while not self.shutdown_event.is_set():
            try:
                # Check if processes are still running
                if (
                    self.api_server_process
                    and self.api_server_process.poll() is not None
                ):
                    logger.warning("API server stopped unexpectedly")
                    if not self.shutdown_event.is_set():
                        logger.info("Restarting API server...")
                        self.start_api_server()

                if self.frontend_process and self.frontend_process.poll() is not None:
                    logger.warning("Frontend server stopped unexpectedly")
                    if not self.shutdown_event.is_set():
                        logger.info("Restarting frontend server...")
                        self.start_frontend()

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in GUI monitoring: {e}")
                time.sleep(5)

    def run_with_bot(self, bot_command: list[str], auto_gui: bool = True):
        """Run the bot with optional GUI auto-start."""
        try:
            if auto_gui:
                logger.info("Auto-starting GUI with bot...")
                if not self.start_gui():
                    logger.warning("Failed to start GUI, continuing with bot only")

            # Start GUI monitoring in a separate thread
            if auto_gui:
                monitor_thread = threading.Thread(target=self.monitor_gui, daemon=True)
                monitor_thread.start()

            # Run the bot command
            logger.info(f"Starting bot with command: {' '.join(bot_command)}")
            bot_process = subprocess.Popen(
                bot_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Monitor the bot process
            while bot_process.poll() is None and not self.shutdown_event.is_set():
                time.sleep(1)

            if bot_process.returncode is not None:
                stdout, stderr = bot_process.communicate()
                if bot_process.returncode != 0:
                    logger.error(f"Bot process failed: {stderr}")
                else:
                    logger.info("Bot process completed successfully")

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error running bot: {e}")
        finally:
            self.stop_gui()


def detect_bot_mode() -> str | None:
    """Detect the bot launch mode from command line arguments."""
    if len(sys.argv) < 2:
        return None

    # Check for main_launcher.py modes
    if "main_launcher.py" in sys.argv[0]:
        if "trade" in sys.argv:
            if "paper" in sys.argv:
                return "paper_trading"
            if "live" in sys.argv:
                return "live_trading"
        elif "manager" in sys.argv:
            return "portfolio_manager"
        elif "backtest" in sys.argv:
            return "backtest"

    # Check for training_cli.py modes
    elif "training_cli.py" in sys.argv[0]:
        if "full-test-run" in sys.argv:
            return "full_test_run"
        if "train" in sys.argv:
            return "training"
        if "retrain" in sys.argv:
            return "retraining"

    return None


def main():
    """Main entry point for the GUI launcher."""
    parser = argparse.ArgumentParser(description="GUI Launcher for Ares Trading Bot")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable automatic GUI launch",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install GUI dependencies and exit",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check GUI dependencies and exit",
    )
    parser.add_argument(
        "--bot-command",
        nargs="+",
        help="Bot command to run (if not auto-detected)",
    )

    args = parser.parse_args()

    launcher = GUILauncher()

    # Check dependencies
    if args.check_deps:
        if launcher.check_gui_dependencies():
            print("✅ All GUI dependencies are available")
            sys.exit(0)
        else:
            print("❌ Missing GUI dependencies")
            sys.exit(1)

    # Install dependencies
    if args.install_deps:
        if launcher.install_gui_dependencies():
            print("✅ GUI dependencies installed successfully")
            sys.exit(0)
        else:
            print("❌ Failed to install GUI dependencies")
            sys.exit(1)

    # Auto-detect bot mode
    bot_mode = detect_bot_mode()
    if bot_mode:
        logger.info(f"Detected bot mode: {bot_mode}")

    # Determine bot command
    if args.bot_command:
        bot_command = args.bot_command
    else:
        # Use the original command line arguments
        bot_command = sys.argv[1:]

    # Run the bot with GUI
    auto_gui = not args.no_gui
    launcher.run_with_bot(bot_command, auto_gui=auto_gui)


if __name__ == "__main__":
    main()
