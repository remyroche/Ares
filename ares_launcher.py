#!/usr/bin/env python3
"""
Ares Comprehensive Launcher

This script provides a unified interface for launching the Ares trading bot with various modes:
1. Blank training run for testing
2. Backtesting for validation
3. Paper/Shadow trading for safe testing
4. Live trading for production
5. Portfolio management for multi-token trading

Usage:
    # Blank training run (fast testing)
    python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE
    
    # Backtesting
    python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE
    
    # Paper trading (shadow trading)
    python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE
    
    # Live trading for single token
    python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE
    
    # Live trading for all tokens + portfolio manager
    python ares_launcher.py portfolio
    
    # GUI only
    python ares_launcher.py gui
    
    # GUI + specific mode
    python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE
"""

import argparse
import asyncio
import sys
import os
import subprocess
import time
import signal
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Try to import requests for GUI health checks
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logging, system_logger
from src.config import CONFIG, settings
from src.database.sqlite_manager import SQLiteManager
from src.supervisor.global_portfolio_manager import GlobalPortfolioManager
from src.utils.state_manager import StateManager
from src.database.firestore_manager import FirestoreManager
from src.ares_pipeline import AresPipeline


class AresLauncher:
    """Comprehensive launcher for Ares trading bot."""
    
    def __init__(self):
        self.logger = system_logger.getChild("AresLauncher")
        self.processes = []  # Track subprocesses for cleanup
        self.gui_process = None
        self.portfolio_process = None
        
    def setup_logging(self):
        """Setup logging for the launcher."""
        setup_logging()
        self.logger.info("=" * 80)
        self.logger.info("🚀 ARES COMPREHENSIVE LAUNCHER")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def cleanup(self):
        """Cleanup processes on exit."""
        self.logger.info("🧹 Cleaning up processes...")
        
        # Terminate GUI process
        if self.gui_process and self.gui_process.poll() is None:
            self.logger.info("🔄 Terminating GUI process...")
            self.gui_process.terminate()
            try:
                self.gui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.gui_process.kill()
                
        # Terminate portfolio process
        if self.portfolio_process and self.portfolio_process.poll() is None:
            self.logger.info("🔄 Terminating portfolio process...")
            self.portfolio_process.terminate()
            try:
                self.portfolio_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.portfolio_process.kill()
                
        # Terminate other processes
        for process in self.processes:
            if process.poll() is None:
                self.logger.info(f"🔄 Terminating process {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
        self.logger.info("✅ Cleanup completed")
        
    def launch_gui(self, mode: Optional[str] = None, symbol: Optional[str] = None, exchange: Optional[str] = None):
        """Launch the GUI (API server + frontend)."""
        self.logger.info("🖥️  Launching GUI (API server + frontend)...")
        
        # Check if GUI directory exists
        gui_dir = Path("GUI")
        if not gui_dir.exists():
            self.logger.error("❌ GUI directory not found")
            return False
            
        # Check if start script exists
        start_script = gui_dir / "start.sh"
        if not start_script.exists():
            self.logger.error("❌ GUI start script not found")
            return False
            
        # Make start script executable
        start_script.chmod(0o755)
        
        # Start GUI using the start script
        gui_cmd = ["bash", str(start_script)]
        self.gui_process = subprocess.Popen(
            gui_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(gui_dir)
        )
        
        # Wait for GUI to start (longer wait for both API and frontend)
        self.logger.info("⏳ Waiting for GUI to start...")
        time.sleep(10)
        
        # Check if API server is running
        if REQUESTS_AVAILABLE:
            try:
                api_response = requests.get("http://localhost:8000", timeout=5)
                if api_response.status_code == 200:
                    self.logger.info("✅ API server started successfully on http://localhost:8000")
                else:
                    self.logger.warning("⚠️  API server may not be fully ready")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not verify API server: {e}")
                
            # Check if frontend is running
            try:
                frontend_response = requests.get("http://localhost:3000", timeout=5)
                if frontend_response.status_code == 200:
                    self.logger.info("✅ Frontend started successfully on http://localhost:3000")
                else:
                    self.logger.warning("⚠️  Frontend may not be fully ready")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not verify frontend: {e}")
        else:
            self.logger.info("ℹ️  Skipping GUI health checks (requests not available)")
            
        if self.gui_process.poll() is None:
            self.logger.info("✅ GUI started successfully")
            self.logger.info("📊 Dashboard: http://localhost:3000")
            self.logger.info("📚 API Docs: http://localhost:8000/docs")
            return True
        else:
            self.logger.error("❌ Failed to start GUI")
            return False
        
    def launch_portfolio_manager(self):
        """Launch the global portfolio manager."""
        self.logger.info("📊 Launching Global Portfolio Manager...")
        
        # Start portfolio manager
        portfolio_cmd = [sys.executable, "main_launcher.py", "manager"]
        self.portfolio_process = subprocess.Popen(
            portfolio_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for portfolio manager to start
        time.sleep(3)
        
        if self.portfolio_process.poll() is None:
            self.logger.info("✅ Portfolio manager started successfully")
        else:
            self.logger.error("❌ Failed to start portfolio manager")
            return False
            
        return True
        
    def run_blank_training(self, symbol: str, exchange: str, with_gui: bool = False):
        """Run blank training for testing."""
        self.logger.info(f"🧪 Running blank training for {symbol} on {exchange}...")
        
        # Launch GUI if requested
        if with_gui:
            if not self.launch_gui():
                self.logger.warning("⚠️  Failed to launch GUI, continuing without GUI")
        
        cmd = [
            sys.executable, 
            "scripts/blank_training_run.py",
            "--symbol", symbol,
            "--exchange", exchange
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            self.logger.info("✅ Blank training completed successfully")
            return True
        else:
            self.logger.error(f"❌ Blank training failed: {process.stderr}")
            return False
            
    def run_backtesting(self, symbol: str, exchange: str, with_gui: bool = False):
        """Run backtesting for validation."""
        self.logger.info(f"📊 Running backtesting for {symbol} on {exchange}...")
        
        # Launch GUI if requested
        if with_gui:
            if not self.launch_gui():
                self.logger.warning("⚠️  Failed to launch GUI, continuing without GUI")
        
        cmd = [
            sys.executable,
            "scripts/training_cli.py",
            "full-test-run",
            symbol,
            exchange
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            self.logger.info("✅ Backtesting completed successfully")
            return True
        else:
            self.logger.error(f"❌ Backtesting failed: {process.stderr}")
            return False
            
    def run_paper_trading(self, symbol: str, exchange: str, with_gui: bool = False):
        """Run paper trading (shadow trading)."""
        self.logger.info(f"📝 Running paper trading for {symbol} on {exchange}...")
        
        # Launch GUI if requested
        if with_gui:
            if not self.launch_gui():
                self.logger.warning("⚠️  Failed to launch GUI, continuing without GUI")
        
        # First run backtesting
        if not self.run_backtesting(symbol, exchange):
            return False
            
        # Then launch paper trading bot
        cmd = [
            sys.executable,
            "main_launcher.py",
            "trade",
            "--symbol", symbol,
            "--exchange", exchange,
            "paper"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(process)
        self.logger.info("✅ Paper trading bot started successfully")
        return True
        
    def run_live_trading(self, symbol: str, exchange: str, with_gui: bool = False):
        """Run live trading for production."""
        self.logger.info(f"💰 Running live trading for {symbol} on {exchange}...")
        
        # Launch GUI if requested
        if with_gui:
            if not self.launch_gui():
                self.logger.warning("⚠️  Failed to launch GUI, continuing without GUI")
        
        # First run backtesting
        if not self.run_backtesting(symbol, exchange):
            return False
            
        # Then launch live trading bot
        cmd = [
            sys.executable,
            "main_launcher.py",
            "trade",
            "--symbol", symbol,
            "--exchange", exchange,
            "live"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(process)
        self.logger.info("✅ Live trading bot started successfully")
        return True
        
    def run_portfolio_trading(self, with_gui: bool = False):
        """Run live trading for all tokens with portfolio manager."""
        self.logger.info("📈 Running portfolio trading for all tokens...")
        
        # Launch GUI if requested
        if with_gui:
            if not self.launch_gui():
                self.logger.warning("⚠️  Failed to launch GUI, continuing without GUI")
        
        # Launch portfolio manager
        if not self.launch_portfolio_manager():
            return False
            
        # Get all supported tokens from config
        supported_tokens = CONFIG.get("SUPPORTED_TOKENS", {})
        all_tokens = []
        
        for exchange, tokens in supported_tokens.items():
            for token in tokens:
                all_tokens.append((token, exchange))
                
        self.logger.info(f"📋 Found {len(all_tokens)} tokens to trade")
        
        # Launch trading bots for each token
        for symbol, exchange in all_tokens:
            self.logger.info(f"🚀 Launching live trading for {symbol} on {exchange}...")
            
            # Run backtesting first
            if not self.run_backtesting(symbol, exchange):
                self.logger.warning(f"⚠️  Skipping {symbol} due to backtesting failure")
                continue
                
            # Launch live trading bot
            cmd = [
                sys.executable,
                "main_launcher.py",
                "trade",
                "--symbol", symbol,
                "--exchange", exchange,
                "live"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(process)
            self.logger.info(f"✅ Live trading bot started for {symbol}")
            
        self.logger.info(f"🎉 Portfolio trading launched with {len(self.processes)} bots")
        return True
        
    def run_gui_only(self):
        """Run GUI only."""
        self.logger.info("🖥️  Launching GUI only...")
        return self.launch_gui()
        
    def run_gui_with_mode(self, mode: str, symbol: str, exchange: str):
        """Run GUI with specific trading mode."""
        self.logger.info(f"🖥️  Launching GUI with {mode} mode for {symbol} on {exchange}...")
        
        # Launch GUI
        if not self.launch_gui():
            return False
            
        # Run the specified mode
        if mode == "blank":
            return self.run_blank_training(symbol, exchange)
        elif mode == "backtest":
            return self.run_backtesting(symbol, exchange)
        elif mode == "paper":
            return self.run_paper_trading(symbol, exchange)
        elif mode == "live":
            return self.run_live_trading(symbol, exchange)
        else:
            self.logger.error(f"❌ Unknown mode: {mode}")
            return False
            
    def wait_for_user_input(self):
        """Wait for user input to stop the launcher."""
        try:
            self.logger.info("⏸️  Press Ctrl+C to stop all processes...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("🛑 Received stop signal...")
            self.cleanup()


def main():
    """Main entry point for the Ares launcher."""
    parser = argparse.ArgumentParser(
        description="Ares Comprehensive Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Blank training run (fast testing)
  python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE
  
  # Backtesting
  python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE
  
  # Paper trading (shadow trading)
  python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE
  
  # Live trading for single token
  python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE
  
  # Live trading for all tokens + portfolio manager
  python ares_launcher.py portfolio
  
  # GUI only
  python ares_launcher.py gui
  
  # GUI + specific mode
  python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE
  
  # All modes with GUI integration
  python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --gui
  python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE --gui
  python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE --gui
  python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE --gui
  python ares_launcher.py portfolio --gui
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")
    
    # Blank training parser
    blank_parser = subparsers.add_parser("blank", help="Run blank training for testing")
    blank_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., ETHUSDT)")
    blank_parser.add_argument("--exchange", required=True, help="Exchange name (e.g., BINANCE)")
    blank_parser.add_argument("--gui", action="store_true", help="Launch GUI with blank training")
    
    # Backtesting parser
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting for validation")
    backtest_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., ETHUSDT)")
    backtest_parser.add_argument("--exchange", required=True, help="Exchange name (e.g., BINANCE)")
    backtest_parser.add_argument("--gui", action="store_true", help="Launch GUI with backtesting")
    
    # Paper trading parser
    paper_parser = subparsers.add_parser("paper", help="Run paper trading (shadow trading)")
    paper_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., ETHUSDT)")
    paper_parser.add_argument("--exchange", required=True, help="Exchange name (e.g., BINANCE)")
    paper_parser.add_argument("--gui", action="store_true", help="Launch GUI with paper trading")
    
    # Live trading parser
    live_parser = subparsers.add_parser("live", help="Run live trading for production")
    live_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., ETHUSDT)")
    live_parser.add_argument("--exchange", required=True, help="Exchange name (e.g., BINANCE)")
    live_parser.add_argument("--gui", action="store_true", help="Launch GUI with live trading")
    
    # Portfolio trading parser
    portfolio_parser = subparsers.add_parser("portfolio", help="Run live trading for all tokens with portfolio manager")
    portfolio_parser.add_argument("--gui", action="store_true", help="Launch GUI with portfolio trading")
    
    # GUI parser
    gui_parser = subparsers.add_parser("gui", help="Launch GUI with optional mode")
    gui_parser.add_argument("--mode", choices=["blank", "backtest", "paper", "live"], help="Trading mode to run with GUI")
    gui_parser.add_argument("--symbol", help="Trading symbol (required if mode is specified)")
    gui_parser.add_argument("--exchange", help="Exchange name (required if mode is specified)")
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = AresLauncher()
    launcher.setup_logging()
    
    # Setup signal handlers for cleanup
    def signal_handler(signum, frame):
        launcher.logger.info("🛑 Received signal, cleaning up...")
        launcher.cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Execute the requested command
        if args.command == "blank":
            success = launcher.run_blank_training(args.symbol, args.exchange, with_gui=args.gui)
            if not success:
                sys.exit(1)
            if args.gui:
                launcher.wait_for_user_input()
                
        elif args.command == "backtest":
            success = launcher.run_backtesting(args.symbol, args.exchange, with_gui=args.gui)
            if not success:
                sys.exit(1)
            if args.gui:
                launcher.wait_for_user_input()
                
        elif args.command == "paper":
            success = launcher.run_paper_trading(args.symbol, args.exchange, with_gui=args.gui)
            if not success:
                sys.exit(1)
            launcher.wait_for_user_input()
            
        elif args.command == "live":
            success = launcher.run_live_trading(args.symbol, args.exchange, with_gui=args.gui)
            if not success:
                sys.exit(1)
            launcher.wait_for_user_input()
            
        elif args.command == "portfolio":
            success = launcher.run_portfolio_trading(with_gui=args.gui)
            if not success:
                sys.exit(1)
            launcher.wait_for_user_input()
            
        elif args.command == "gui":
            if args.mode:
                if not args.symbol or not args.exchange:
                    launcher.logger.error("❌ Symbol and exchange are required when mode is specified")
                    sys.exit(1)
                success = launcher.run_gui_with_mode(args.mode, args.symbol, args.exchange)
                if not success:
                    sys.exit(1)
                launcher.wait_for_user_input()
            else:
                success = launcher.run_gui_only()
                if not success:
                    sys.exit(1)
                launcher.wait_for_user_input()
                
    except Exception as e:
        launcher.logger.error(f"💥 Unexpected error: {e}")
        launcher.cleanup()
        sys.exit(1)
    finally:
        launcher.cleanup()


if __name__ == "__main__":
    main() 