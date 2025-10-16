#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
GUI Manager for Ares Launcher

This module handles GUI and process management, extracting this functionality
from the main launcher class to reduce complexity.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Try to import requests for GUI health checks
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ProcessManager:
    """Manages subprocess lifecycle and cleanup."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.processes: List[subprocess.Popen] = []
        self.gui_process: Optional[subprocess.Popen] = None
        self.portfolio_process: Optional[subprocess.Popen] = None
    
    def add_process(self, process: subprocess.Popen) -> None:
        """Add a process to the tracking list."""
        self.processes.append(process)
        self.logger.info(f"✅ Process {process.pid} added to tracking")
    
    def cleanup(self) -> None:
        """Cleanup all tracked processes."""
        self.logger.info("🧹 Cleaning up processes...")
        
        # Terminate GUI process
        if self.gui_process and self.gui_process.poll() is None:
            self.logger.info("🔄 Terminating GUI process...")
            self.gui_process.terminate()
            try:
                self.gui_process.wait(timeout = 5)
            except subprocess.TimeoutExpired:
                self.gui_process.kill()
        
        # Terminate portfolio process
        if self.portfolio_process and self.portfolio_process.poll() is None:
            self.logger.info("🔄 Terminating portfolio process...")
            self.portfolio_process.terminate()
            try:
                self.portfolio_process.wait(timeout = 5)
            except subprocess.TimeoutExpired:
                self.portfolio_process.kill()
        
        # Terminate any other tracked processes
        for process in self.processes:
            if process.poll() is None:
                self.logger.info(f"🔄 Terminating process {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout = 3)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        self.logger.info("✅ Cleanup completed")


class GUIManager:
    """Manages GUI server lifecycle and health checks."""
    
    def __init__(self, process_manager: ProcessManager, logger: logging.Logger):
        self.process_manager = process_manager
        self.logger = logger
    
    def launch_gui(
        self,
        mode: Optional[str] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> bool:
        """Launch the GUI server."""
        self.logger.info("🚀 Launching GUI server...")
        
        # Prefer unified start script which runs API and frontend
        script_path = Path("GUI/start.sh")
        env = os.environ.copy()
        
        # Allow user to override ports via env
        env.setdefault("API_PORT", env.get("API_PORT", "8000"))
        env.setdefault("FRONTEND_PORT", env.get("FRONTEND_PORT", "3000"))
        
        if script_path.exists():
            cmd = ["bash", str(script_path)]
        else:
            # Fallback: start API only (legacy behaviour)
            cmd = [sys.executable, "GUI/api_server.py"]
            # Pass optional mode args if provided and using api_server directly
            if mode and symbol and exchange:
                cmd.extend(["--mode", mode, "--symbol", symbol, "--exchange", exchange])
        
        self.process_manager.gui_process = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            env = env,
        )
        self.process_manager.add_process(self.process_manager.gui_process)
        self.logger.info(f"✅ GUI process started with PID {self.process_manager.gui_process.pid}")
        
        # Wait a moment for the server(s) to start
        time.sleep(3)
        
        # Health check: if requests is available, ping frontend then API
        if self.process_manager.gui_process.poll() is None:
            if REQUESTS_AVAILABLE:
                try:
                    fp = int(env.get("FRONTEND_PORT", "3000"))
                    ap = int(env.get("API_PORT", "8000"))
                    requests.get(f"http://localhost:{fp}", timeout = 2)
                    requests.get(f"http://localhost:{ap}/docs", timeout = 2)
                    self.logger.info("✅ GUI (frontend+API) appears healthy")
                except Exception as _hc_exc:
                    self.logger.warning(f"GUI health check skipped/failed: {_hc_exc}")
            self.logger.info("✅ GUI server is running")
            return True
        
        stdout, stderr = self.process_manager.gui_process.communicate()
        self.logger.error(f"❌ GUI start failed. STDERR: {stderr}\nSTDOUT: {stdout}")
        return False
    
    def launch_portfolio_manager(self) -> bool:
        """Launch the portfolio manager."""
        self.logger.info("🚀 Launching portfolio manager...")
        
        self.process_manager.portfolio_process = subprocess.Popen(
            [sys.executable, "src/supervisor/global_portfolio_manager.py"],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
        )
        self.process_manager.add_process(self.process_manager.portfolio_process)
        self.logger.info(f"✅ Portfolio manager started with PID {self.process_manager.portfolio_process.pid}")
        return True


class TradingProcessManager:
    """Manages trading process execution and monitoring."""
    
    def __init__(self, process_manager: ProcessManager, logger: logging.Logger):
        self.process_manager = process_manager
        self.logger = logger
    
    def run_trading_process(
        self,
        symbol: str,
        exchange: str,
        trading_mode: str,
    ) -> bool:
        """Run trading process with real-time output monitoring."""
        mode_display = "paper trading" if trading_mode == "PAPER" else "live trading"
        self.logger.info(f"📊 Running {mode_display} for {symbol} on {exchange}")
        tprint(f"📊 Running {mode_display} for {symbol} on {exchange}")
        tprint("=" * 80)
        
        try:
            # Set environment variable for trading mode
            os.environ["TRADING_MODE"] = trading_mode
            
            # Run the same pipeline but with different trading mode
            process = subprocess.Popen(
                [sys.executable, "src/ares_pipeline.py", symbol, exchange],
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,  # Redirect stderr to stdout
                text = True,
                bufsize = 1,  # Line buffered
                universal_newlines = True,
                env = dict(
                    os.environ,
                    TRADING_MODE = trading_mode,
                ),  # Pass environment variable
            )
            self.process_manager.add_process(process)
            
            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    tprint(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it
            
            # Get the final return code
            return_code = process.poll()
            
            if return_code == 0:
                self.logger.info(f"✅ {mode_display} completed successfully")
                tprint(f"✅ {mode_display} completed successfully")
                return True
            else:
                self.logger.error(f"❌ {mode_display} failed with return code: {return_code}")
                tprint(f"❌ {mode_display} failed with return code: {return_code}")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Failed to run {mode_display}: {e}")
            tprint(f"❌ Failed to run {mode_display}: {e}")
            return False
    
    def run_portfolio_trading(self, supported_tokens: List[str]) -> bool:
        """Run portfolio trading with multiple tokens."""
        self.logger.info("📈 Running portfolio trading")
        
        # Launch individual trading bots for each supported token
        for token in supported_tokens:
            self.logger.info(f"🚀 Launching trading bot for {token}")
            try:
                process = subprocess.Popen(
                    [sys.executable, "src/ares_pipeline.py", token, "BINANCE"],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE,
                    text = True,
                )
                self.process_manager.add_process(process)
                self.logger.info(f"✅ Trading bot for {token} started with PID {process.pid}")
            except Exception as e:
                self.logger.exception(f"❌ Failed to launch trading bot for {token}: {e}")
        
        return True


class UserInteractionManager:
    """Manages user interaction and input handling."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def wait_for_user_input(self) -> None:
        """Wait for user input to stop the launcher."""
        self.logger.info("⏸️ Press Enter to stop the launcher...")
        try:
            input()
        except KeyboardInterrupt:
            self.logger.info("🛑 Received keyboard interrupt")


class GUIManagerFactory:
    """Factory for creating GUI and process managers."""
    
    @staticmethod
    def create_managers(logger: logging.Logger) -> tuple:
        """Create all GUI and process managers."""
        process_manager = ProcessManager(logger)
        gui_manager = GUIManager(process_manager, logger)
        trading_process_manager = TradingProcessManager(process_manager, logger)
        user_interaction_manager = UserInteractionManager(logger)
        
        return (
            process_manager,
            gui_manager,
            trading_process_manager,
            user_interaction_manager
        )