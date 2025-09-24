"""
NAS/TAS Training Main Entry Point

This module provides the main entry point for training NAS and TAS models
integrated with the existing Analyst and Tactician ensemble models.

Usage:
    python -m src.training.nas_tas_training_main
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_success, tprint_error
from src.training.steps.model_training.nas_tas_training_orchestrator import create_nas_tas_training_orchestrator
from src.config.config import NASConfig, TASConfig, SystemConfig

logger = system_logger.getChild('NASTASTrainingMain')

async def main():
    """Main entry point for NAS/TAS training."""
    try:
        tprint_info("🚀 Starting NAS/TAS Training Pipeline...")
        
        # Initialize configuration
        nas_config = NASConfig()
        tas_config = TASConfig()
        system_config = SystemConfig()
        
        # Create training orchestrator
        orchestrator = create_nas_tas_training_orchestrator({
            'nas_config': nas_config,
            'tas_config': tas_config,
            'system_config': system_config
        })
        
        # Prepare training data (this would come from your data pipeline)
        training_data = {
            'X_5m': None,  # 5m timeframe features for NAS
            'y_5m': None,  # 5m timeframe targets for NAS
            'regime_labels': None,  # Regime labels for per-regime training
            'X_1m': None,  # 1m timeframe features for TAS
            'y_1m': None,  # 1m timeframe targets for TAS
            'analyst_signals': None  # Analyst signals for TAS training
        }
        
        # Execute complete training pipeline
        tprint_info("🧠 Executing NAS/TAS training pipeline...")
        results = await orchestrator.execute_complete_training_pipeline(training_data)
        
        tprint_success("✅ NAS/TAS training completed successfully!")
        tprint_info(f"📊 Training Results:")
        tprint_info(f"   - NAS Models: {len(results.get('nas_results', {}).get('trained_models', {}))}")
        tprint_info(f"   - TAS Models: {len(results.get('tas_results', {}).get('trained_models', {}))}")
        tprint_info(f"   - Analyst Ensemble: {results.get('analyst_ensemble_results', {}).get('status', 'unknown')}")
        tprint_info(f"   - Tactician Ensemble: {results.get('tactician_ensemble_results', {}).get('status', 'unknown')}")
        
        return results
        
    except Exception as e:
        tprint_error(f"❌ NAS/TAS training failed: {str(e)}")
        logger.error(f"NAS/TAS training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())