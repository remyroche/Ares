#!/usr/bin/env python3
"""
Launcher Integration Script

This script provides integration between the GUI API server and the ares_launcher.py
to enable actual execution of launcher commands from the web interface.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logger = logging.getLogger(__name__)

class LauncherIntegration:
    """Handles integration between GUI and ares_launcher.py"""
    
    def __init__(self):
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.process_logs: Dict[str, List[str]] = {}
        self.project_root = Path(__file__).parent.parent
        
    def get_launcher_path(self) -> Path:
        """Get the path to ares_launcher.py"""
        return self.project_root / "ares_launcher.py"
    
    def validate_launcher_exists(self) -> bool:
        """Check if ares_launcher.py exists"""
        launcher_path = self.get_launcher_path()
        return launcher_path.exists()
    
    async def start_launcher_mode(
        self, 
        mode: str, 
        symbol: str, 
        exchange: str = "BINANCE",
        lookback_days: Optional[int] = None,
        additional_args: Optional[List[str]] = None
    ) -> Dict:
        """Start a launcher mode process"""
        
        if not self.validate_launcher_exists():
            return {
                "success": False,
                "error": "ares_launcher.py not found in project root"
            }
        
        # Build command
        cmd = [sys.executable, str(self.get_launcher_path()), mode]
        
        if symbol:
            cmd.extend(["--symbol", symbol])
        if exchange:
            cmd.extend(["--exchange", exchange])
        if lookback_days:
            cmd.extend(["--lookback-days", str(lookback_days)])
        if additional_args:
            cmd.extend(additional_args)
        
        # Create process key
        process_key = f"{mode}_{symbol}_{exchange}_{int(time.time())}"
        
        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(self.project_root)
            )
            
            # Store process info
            self.running_processes[process_key] = process
            self.process_logs[process_key] = []
            
            # Start log collection in background
            asyncio.create_task(self._collect_process_logs(process_key, process))
            
            return {
                "success": True,
                "process_key": process_key,
                "pid": process.pid,
                "command": " ".join(cmd),
                "message": f"Started {mode} mode for {symbol} on {exchange}"
            }
            
        except Exception as e:
            logger.exception(f"Failed to start launcher mode {mode}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def start_training(
        self,
        mode: str,
        symbol: str,
        exchange: str = "BINANCE",
        lookback_days: Optional[int] = None
    ) -> Dict:
        """Start training with specified parameters"""
        
        # Map training modes to launcher commands
        mode_mapping = {
            "light": "light",
            "blank": "blank", 
            "full": "full"
        }
        
        if mode not in mode_mapping:
            return {
                "success": False,
                "error": f"Invalid training mode: {mode}. Must be one of: {list(mode_mapping.keys())}"
            }
        
        launcher_mode = mode_mapping[mode]
        return await self.start_launcher_mode(
            launcher_mode, symbol, exchange, lookback_days
        )
    
    async def stop_process(self, process_key: str) -> Dict:
        """Stop a running process"""
        
        if process_key not in self.running_processes:
            return {
                "success": False,
                "error": f"Process {process_key} not found"
            }
        
        process = self.running_processes[process_key]
        
        try:
            if process.poll() is None:  # Process is still running
                process.terminate()
                
                # Wait for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                
                return {
                    "success": True,
                    "message": f"Process {process_key} stopped"
                }
            else:
                return {
                    "success": True,
                    "message": f"Process {process_key} was already stopped"
                }
                
        except Exception as e:
            logger.exception(f"Failed to stop process {process_key}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # Clean up
            if process_key in self.running_processes:
                del self.running_processes[process_key]
            if process_key in self.process_logs:
                del self.process_logs[process_key]
    
    async def stop_all_processes(self) -> Dict:
        """Stop all running processes"""
        
        stopped_processes = []
        errors = []
        
        for process_key in list(self.running_processes.keys()):
            result = await self.stop_process(process_key)
            if result["success"]:
                stopped_processes.append(process_key)
            else:
                errors.append(f"{process_key}: {result['error']}")
        
        return {
            "success": len(errors) == 0,
            "stopped_processes": stopped_processes,
            "errors": errors,
            "message": f"Stopped {len(stopped_processes)} processes"
        }
    
    async def get_process_status(self) -> Dict:
        """Get status of all running processes"""
        
        status = {
            "running_processes": [],
            "total_processes": len(self.running_processes),
            "last_check": datetime.now().isoformat()
        }
        
        for process_key, process in self.running_processes.items():
            process_info = {
                "process_key": process_key,
                "pid": process.pid,
                "status": "running" if process.poll() is None else "stopped",
                "return_code": process.returncode if process.poll() is not None else None
            }
            
            # Add recent logs
            if process_key in self.process_logs:
                recent_logs = self.process_logs[process_key][-10:]  # Last 10 lines
                process_info["recent_logs"] = recent_logs
            
            status["running_processes"].append(process_info)
        
        return status
    
    async def _collect_process_logs(self, process_key: str, process: subprocess.Popen):
        """Collect logs from a running process"""
        
        try:
            while process.poll() is None:
                line = process.stdout.readline()
                if line:
                    self.process_logs[process_key].append(line.strip())
                    # Keep only last 100 lines to prevent memory issues
                    if len(self.process_logs[process_key]) > 100:
                        self.process_logs[process_key] = self.process_logs[process_key][-100:]
                else:
                    await asyncio.sleep(0.1)
            
            # Collect any remaining output
            remaining_output = process.stdout.read()
            if remaining_output:
                for line in remaining_output.split('\n'):
                    if line.strip():
                        self.process_logs[process_key].append(line.strip())
                        
        except Exception as e:
            logger.exception(f"Error collecting logs for {process_key}: {e}")
        finally:
            # Clean up when process ends
            if process_key in self.running_processes:
                del self.running_processes[process_key]
    
    def get_available_modes(self) -> List[str]:
        """Get list of available launcher modes"""
        return [
            "paper", "live", "backtest", "blank", "light", "full", 
            "load", "precompute", "portfolio", "gui"
        ]
    
    def get_available_training_modes(self) -> List[str]:
        """Get list of available training modes"""
        return ["light", "blank", "full"]
    
    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchanges"""
        return ["BINANCE", "MEXC", "GATEIO"]


# Global instance
launcher_integration = LauncherIntegration()


# Convenience functions for API server
async def start_launcher_mode(mode: str, symbol: str, exchange: str = "BINANCE", **kwargs) -> Dict:
    """Start a launcher mode"""
    return await launcher_integration.start_launcher_mode(mode, symbol, exchange, **kwargs)


async def start_training(mode: str, symbol: str, exchange: str = "BINANCE", **kwargs) -> Dict:
    """Start training"""
    return await launcher_integration.start_training(mode, symbol, exchange, **kwargs)


async def stop_process(process_key: str) -> Dict:
    """Stop a specific process"""
    return await launcher_integration.stop_process(process_key)


async def stop_all_processes() -> Dict:
    """Stop all processes"""
    return await launcher_integration.stop_all_processes()


async def get_process_status() -> Dict:
    """Get process status"""
    return await launcher_integration.get_process_status()


def get_available_modes() -> List[str]:
    """Get available modes"""
    return launcher_integration.get_available_modes()


def get_available_training_modes() -> List[str]:
    """Get available training modes"""
    return launcher_integration.get_available_training_modes()


def get_available_exchanges() -> List[str]:
    """Get available exchanges"""
    return launcher_integration.get_available_exchanges()


if __name__ == "__main__":
    # Test the integration
    async def test():
        print("Testing Launcher Integration...")
        
        # Test validation
        if launcher_integration.validate_launcher_exists():
            print("✅ ares_launcher.py found")
        else:
            print("❌ ares_launcher.py not found")
            return
        
        # Test available modes
        print(f"Available modes: {get_available_modes()}")
        print(f"Available training modes: {get_available_training_modes()}")
        print(f"Available exchanges: {get_available_exchanges()}")
        
        # Test starting a process (dry run)
        print("\nTesting process start (dry run)...")
        result = await start_launcher_mode("blank", "ETHUSDT", "BINANCE")
        print(f"Result: {result}")
        
        if result["success"]:
            process_key = result["process_key"]
            print(f"Started process: {process_key}")
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Check status
            status = await get_process_status()
            print(f"Status: {status}")
            
            # Stop process
            stop_result = await stop_process(process_key)
            print(f"Stop result: {stop_result}")
    
    asyncio.run(test())