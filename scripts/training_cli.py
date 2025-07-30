#!/usr/bin/env python3
"""
Training Command Line Interface for Ares Trading Bot

This script provides a command-line interface for training operations:
1. Full training for a specific token
2. Model retraining
3. Model import from database
4. Training status and history

Usage:
    python scripts/training_cli.py train <symbol> [exchange]
    python scripts/training_cli.py retrain <symbol> [exchange]
    python scripts/training_cli.py import <model_path> <symbol>
    python scripts/training_cli.py status <symbol>
    python scripts/training_cli.py list-tokens
    python scripts/training_cli.py list-models
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.sqlite_manager import SQLiteManager
from src.training.training_manager import TrainingManager
from src.utils.logger import setup_logging, system_logger
from src.config import CONFIG

class TrainingCLI:
    """Command-line interface for training operations."""
    
    def __init__(self):
        self.logger = system_logger.getChild('TrainingCLI')
        self.db_manager = SQLiteManager()
        self.training_manager = None
    
    async def initialize(self):
        """Initialize the training CLI."""
        await self.db_manager.initialize()
        self.training_manager = TrainingManager(self.db_manager)
    
    async def run_full_training(self, symbol: str, exchange_name: str = "BINANCE"):
        """Run full training pipeline for a symbol."""
        try:
            await self.initialize()
            
            print(f"🚀 Starting full training for {symbol} on {exchange_name}...")
            print("=" * 60)
            
            success = await self.training_manager.run_full_training(symbol, exchange_name)
            
            if success:
                print(f"✅ Full training completed successfully for {symbol}")
                print(f"📁 Models saved to: {self.training_manager.models_dir}")
                print(f"📊 Data saved to: {self.training_manager.data_dir}")
            else:
                print(f"❌ Full training failed for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Full training failed: {e}", exc_info=True)
            print(f"❌ Training error: {e}")
            return False
    
    async def retrain_models(self, symbol: str, exchange_name: str = "BINANCE"):
        """Retrain models for a symbol."""
        try:
            await self.initialize()
            
            print(f"🔄 Starting model retraining for {symbol} on {exchange_name}...")
            print("=" * 60)
            
            success = await self.training_manager.retrain_models(symbol, exchange_name)
            
            if success:
                print(f"✅ Model retraining completed successfully for {symbol}")
            else:
                print(f"❌ Model retraining failed for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}", exc_info=True)
            print(f"❌ Retraining error: {e}")
            return False
    
    async def import_model(self, model_path: str, symbol: str):
        """Import a model from file."""
        try:
            await self.initialize()
            
            print(f"📥 Importing model for {symbol} from {model_path}...")
            print("=" * 60)
            
            success = await self.training_manager.import_model(model_path, symbol)
            
            if success:
                print(f"✅ Model imported successfully for {symbol}")
                print(f"📁 Model saved to: models/{symbol}_imported_model.pkl")
            else:
                print(f"❌ Model import failed for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model import failed: {e}", exc_info=True)
            print(f"❌ Import error: {e}")
            return False
    
    async def get_training_status(self, symbol: str):
        """Get training status for a symbol."""
        try:
            await self.initialize()
            
            print(f"📊 Training status for {symbol}...")
            print("=" * 60)
            
            status_records = await self.training_manager.get_training_status(symbol)
            
            if not status_records:
                print(f"📋 No training records found for {symbol}")
                return
            
            print(f"📋 Found {len(status_records)} training records:")
            print()
            
            for i, record in enumerate(status_records, 1):
                print(f"{i}. {record.get('validation_type', 'Unknown')} - {record.get('status', 'Unknown')}")
                print(f"   Date: {record.get('created_at', 'Unknown')}")
                print(f"   Symbol: {record.get('symbol', 'Unknown')}")
                print()
            
        except Exception as e:
            self.logger.error(f"Failed to get training status: {e}", exc_info=True)
            print(f"❌ Status error: {e}")
    
    def list_supported_tokens(self):
        """List all supported tokens and exchanges."""
        print("🪙 Supported Tokens and Exchanges (100x Leverage):")
        print("=" * 60)
        
        supported_tokens = CONFIG.get('SUPPORTED_TOKENS', {})
        
        for exchange_name, tokens in supported_tokens.items():
            print(f"\n📈 {exchange_name}:")
            for token in tokens:
                print(f"   • {token}")
        
        print(f"\n💡 Usage: python scripts/training_cli.py train <SYMBOL> <EXCHANGE>")
        print(f"   Example: python scripts/training_cli.py train BTCUSDT BINANCE")
        print(f"   All tokens support 100x leverage for high-frequency trading")
    
    def list_model_types(self):
        """List available model types for training."""
        print("🤖 Available Model Types:")
        print("=" * 60)
        
        model_configs = CONFIG.get('MODEL_TRAINING', {}).get('model_types', {})
        
        for model_name, config in model_configs.items():
            enabled = "✅" if config.get('enabled', False) else "❌"
            print(f"{enabled} {model_name.upper()}")
            
            if model_name == 'lightgbm':
                print(f"   - Gradient boosting with LightGBM")
                print(f"   - Fast training and good performance")
            elif model_name == 'xgboost':
                print(f"   - Extreme gradient boosting")
                print(f"   - Excellent for structured data")
            elif model_name == 'neural_network':
                print(f"   - Multi-layer perceptron")
                print(f"   - Good for complex patterns")
            elif model_name == 'random_forest':
                print(f"   - Ensemble of decision trees")
                print(f"   - Robust and interpretable")
            
            print()
    
    def show_training_config(self):
        """Show current training configuration."""
        print("⚙️ Training Configuration:")
        print("=" * 60)
        
        config = CONFIG.get('MODEL_TRAINING', {})
        
        print(f"📊 Data retention: {config.get('data_retention_days', 'N/A')} days")
        print(f"📈 Min data points: {config.get('min_data_points', 'N/A')}")
        print(f"🔀 Train/test split: {config.get('train_test_split', 'N/A')}")
        print(f"✅ Validation split: {config.get('validation_split', 'N/A')}")
        print(f"🚶 Forward walk days: {config.get('forward_walk_days', 'N/A')}")
        print(f"🎲 Monte Carlo simulations: {config.get('monte_carlo_simulations', 'N/A')}")
        print(f"🔄 A/B test duration: {config.get('ab_test_duration_days', 'N/A')} days")
        
        print(f"\n🔧 Regularization:")
        reg_config = config.get('regularization', {})
        print(f"   - L1 alpha: {reg_config.get('l1_alpha', 'N/A')}")
        print(f"   - L2 alpha: {reg_config.get('l2_alpha', 'N/A')}")
        print(f"   - Dropout rate: {reg_config.get('dropout_rate', 'N/A')}")
        
        print(f"\n🎯 Hyperparameter tuning:")
        hp_config = config.get('hyperparameter_tuning', {})
        print(f"   - Enabled: {hp_config.get('enabled', 'N/A')}")
        print(f"   - Max trials: {hp_config.get('max_trials', 'N/A')}")
        print(f"   - Optimization metric: {hp_config.get('optimization_metric', 'N/A')}")

def print_usage():
    """Print usage information."""
    print(__doc__)
    print("\nExamples:")
    print("  # Full training for BTCUSDT")
    print("  python scripts/training_cli.py train BTCUSDT BINANCE")
    print("")
    print("  # Retrain models for ETHUSDT")
    print("  python scripts/training_cli.py retrain ETHUSDT BINANCE")
    print("")
    print("  # Import model from file")
    print("  python scripts/training_cli.py import models/btc_model.pkl BTCUSDT")
    print("")
    print("  # Check training status")
    print("  python scripts/training_cli.py status BTCUSDT")
    print("")
    print("  # List supported tokens")
    print("  python scripts/training_cli.py list-tokens")
    print("")
    print("  # List model types")
    print("  python scripts/training_cli.py list-models")
    print("")
    print("  # Show training configuration")
    print("  python scripts/training_cli.py config")

async def main():
    """Main function."""
    # Setup logging
    setup_logging()
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    cli = TrainingCLI()
    
    if command == "train":
        if len(sys.argv) < 3:
            print("❌ Symbol required for training")
            print_usage()
            sys.exit(1)
        symbol = sys.argv[2]
        exchange = sys.argv[3] if len(sys.argv) > 3 else "BINANCE"
        success = await cli.run_full_training(symbol, exchange)
        sys.exit(0 if success else 1)
        
    elif command == "retrain":
        if len(sys.argv) < 3:
            print("❌ Symbol required for retraining")
            print_usage()
            sys.exit(1)
        symbol = sys.argv[2]
        exchange = sys.argv[3] if len(sys.argv) > 3 else "BINANCE"
        success = await cli.retrain_models(symbol, exchange)
        sys.exit(0 if success else 1)
        
    elif command == "import":
        if len(sys.argv) < 4:
            print("❌ Model path and symbol required for import")
            print_usage()
            sys.exit(1)
        model_path = sys.argv[2]
        symbol = sys.argv[3]
        success = await cli.import_model(model_path, symbol)
        sys.exit(0 if success else 1)
        
    elif command == "status":
        if len(sys.argv) < 3:
            print("❌ Symbol required for status")
            print_usage()
            sys.exit(1)
        symbol = sys.argv[2]
        await cli.get_training_status(symbol)
        
    elif command == "list-tokens":
        cli.list_supported_tokens()
        
    elif command == "list-models":
        cli.list_model_types()
        
    elif command == "config":
        cli.show_training_config()
        
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 