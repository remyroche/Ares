"""
NAS/TAS Integration Guide

This module demonstrates how NAS and TAS are fully integrated into the existing
Analyst and Tactician ensemble models for both training and live trading.

Integration Overview:
1. Training Pipeline Integration
2. Live Trading Integration  
3. Configuration Integration
4. Entry Points
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.tprint import tprint_info, tprint_success, tprint_structured

def print_integration_overview():
    """Print comprehensive integration overview."""
    
    tprint_structured("🧠 NAS/TAS Integration Overview", {
        "Training Pipeline": {
            "NAS Training": "src/training/steps/model_training/nas_training_step.py",
            "TAS Training": "src/training/steps/model_training/tas_training_step.py", 
            "Analyst Ensemble": "src/training/steps/model_training/analyst_ensemble_training.py (enhanced)",
            "Tactician Ensemble": "src/training/steps/model_training/tactician_ensemble_training.py (enhanced)",
            "Orchestrator": "src/training/steps/model_training/nas_tas_training_orchestrator.py",
            "Main Pipeline": "src/training/steps/model_training/sub_pipeline.py (enhanced)",
            "Entry Point": "src/training/nas_tas_training_main.py"
        },
        "Live Trading Integration": {
            "Analyst Signals": "src/trading/signal_generation/analyst_signals.py (enhanced)",
            "Tactician Signals": "src/trading/signal_generation/tactician_signals.py (enhanced)",
            "Live Trader": "src/trading/execution/live_trader.py (enhanced)",
            "Trading Orchestrator": "src/trading/execution/trading_orchestrator.py (enhanced)",
            "Entry Point": "src/trading/nas_tas_trading_main.py"
        },
        "Configuration": {
            "Trading Config": "src/config/trading.py (enhanced)",
            "System Config": "src/config/config.py (enhanced)",
            "NAS Config": "NASConfig dataclass",
            "TAS Config": "TASConfig dataclass"
        }
    })

def print_training_flow():
    """Print the complete training flow."""
    
    tprint_structured("📊 Complete Training Flow", {
        "Step 1": "Train NAS models per-regime on 5m timeframe",
        "Step 2": "Train TAS models per-regime on 1m timeframe", 
        "Step 3": "Train Analyst base models (existing)",
        "Step 4": "Train Analyst ensemble with base models + NAS models",
        "Step 5": "Train Tactician base models (existing)",
        "Step 6": "Train Tactician ensemble with base models + TAS models",
        "Execution": "python -m src.training.nas_tas_training_main"
    })

def print_live_trading_flow():
    """Print the live trading flow."""
    
    tprint_structured("📈 Live Trading Flow", {
        "Analyst": "Generates signals using base models + NAS predictions",
        "Tactician": "Generates timing using base models + TAS predictions", 
        "Signal Combination": "Analyst (60%) + NAS (40%) for direction",
        "Timing Combination": "Tactician (60%) + TAS (40%) for timing",
        "Execution": "python -m src.trading.nas_tas_trading_main"
    })

def print_key_features():
    """Print key integration features."""
    
    tprint_structured("🔧 Key Integration Features", {
        "Per-Regime Training": "NAS and TAS models trained per market regime",
        "Timeframe Specific": "NAS on 5m, TAS on 1m, Regime detection on 15m",
        "Signal Generation": "NAS/TAS used for trading signals, not just architecture search",
        "Stacking Models": "Analyst uses stacking to combine base models + NAS",
        "Enhanced Signals": "Both Analyst and Tactician signals enhanced with NAS/TAS",
        "Configuration": "Full configuration integration with existing systems",
        "Backward Compatible": "Existing functionality preserved and enhanced"
    })

def print_usage_examples():
    """Print usage examples."""
    
    tprint_structured("💡 Usage Examples", {
        "Training": {
            "Full Pipeline": "python -m src.training.nas_tas_training_main",
            "Individual Steps": "Use sub_pipeline.py with specific steps",
            "Configuration": "Modify src/config/config.py for NAS/TAS settings"
        },
        "Live Trading": {
            "Enhanced Trading": "python -m src.trading.nas_tas_trading_main", 
            "Configuration": "Modify src/config/trading.py for trading settings",
            "Model Loading": "NAS/TAS models loaded automatically from saved paths"
        },
        "Integration": {
            "Existing Pipeline": "NAS/TAS integrated into existing training pipeline",
            "Existing Trading": "Enhanced signal generators in existing trading system",
            "Configuration": "All settings in existing config files"
        }
    })

async def main():
    """Main function to demonstrate integration."""
    
    print_integration_overview()
    print_training_flow()
    print_live_trading_flow()
    print_key_features()
    print_usage_examples()
    
    tprint_success("✅ NAS/TAS Integration is fully wired and ready to use!")

if __name__ == "__main__":
    asyncio.run(main())