#!/usr/bin/env python3
"""
Configuration for the Automated Cryptocurrency Analysis Pipeline
"""

import os
from pathlib import Path

# Assets to analyze (major cryptocurrencies)
ASSETS = [
    "ETHUSDT", "ADAUSDT", "ALGOUSDT", "BTCUSDT", "BNBUSDT",
    "SOLUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT",
    "ATOMUSDT", "UNIUSDT", "LTCUSDT", "XRPUSDT", "BCHUSDT",
]

# Data configuration
DATA_CONFIG = {
    "years": 2,  # Number of years of historical data
    "interval": "15m",  # Kline interval
    "data_dir": "data",  # Data directory
    "output_dir": "results",  # Output directory for analysis results
}

# API configuration (optional - can work without API keys for basic analysis)
API_CONFIG = {
    "binance_api_key": os.getenv("BINANCE_API_KEY", ""),
    "binance_api_secret": os.getenv("BINANCE_API_SECRET", ""),
}

# Hardware configuration
HARDWARE_CONFIG = {
    "use_m1_optimizations": True,  # Use M1 chip optimizations if available
    "max_workers": 4,  # Maximum number of concurrent workers
}

# Analysis configuration
ANALYSIS_CONFIG = {
    "barrier_levels": [round(0.003 + i * 0.001, 4) for i in range(13)],  # 0.3% to 1.5% in 0.1% increments
    "generate_plots": True,
    "save_csv": True,
    "save_detailed_reports": True,
}

def create_directories():
    """Create necessary directories"""
    Path(DATA_CONFIG["data_dir"]).mkdir(exist_ok=True)
    Path(DATA_CONFIG["output_dir"]).mkdir(exist_ok=True)
    Path(DATA_CONFIG["output_dir"], "reports").mkdir(exist_ok=True)
    Path(DATA_CONFIG["output_dir"], "csv").mkdir(exist_ok=True)
    Path(DATA_CONFIG["output_dir"], "charts").mkdir(exist_ok=True)

def validate_config():
    """Validate configuration and return any errors"""
    errors = []
    
    # Check if assets list is not empty
    if not ASSETS:
        errors.append("No assets specified in ASSETS list")
    
    # Check data configuration
    if DATA_CONFIG["years"] <= 0:
        errors.append("Years must be positive")
    
    if DATA_CONFIG["interval"] not in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]:
        errors.append(f"Invalid interval: {DATA_CONFIG['interval']}")
    
    # Check hardware configuration
    if HARDWARE_CONFIG["max_workers"] <= 0:
        errors.append("max_workers must be positive")
    
    return errors

def get_config_summary():
    """Get a summary of the current configuration"""
    return {
        "Assets": len(ASSETS),
        "Years": DATA_CONFIG["years"],
        "Interval": DATA_CONFIG["interval"],
        "Data Dir": DATA_CONFIG["data_dir"],
        "Output Dir": DATA_CONFIG["output_dir"],
        "API Key Set": bool(API_CONFIG["binance_api_key"]),
        "M1 Optimizations": HARDWARE_CONFIG["use_m1_optimizations"],
        "Max Workers": HARDWARE_CONFIG["max_workers"],
        "Generate Plots": ANALYSIS_CONFIG["generate_plots"],
        "Save CSV": ANALYSIS_CONFIG["save_csv"],
    }
