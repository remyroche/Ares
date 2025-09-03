#!/usr/bin/env python3
"""Comprehensive SR Training Pipeline.

This module provides a complete training pipeline that integrates:
1. Step7 enhanced matrix operations (comprehensive SR features)
2. Step2_5 SR optimization (SR levels and parameters)
3. Multi-output model training with full SR context

The pipeline ensures all ML models are trained on the complete feature set
including extensive SR features and optimized SR levels.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.training.multi_output_model_trainer import MultiOutputModelTrainer, MultiOutputModelConfig
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.centralized_decorators import (
    handle_errors,
    comprehensive_validation,
    performance_monitor,
    validate_data_structure,
    memory_efficient,
    secure_data_processing,
)

class ComprehensiveSRTrainingPipeline:
    """Comprehensive training pipeline with full SR feature integration."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("ComprehensiveSRTrainingPipeline")
        
        # Initialize multi-output model trainer
        self.model_trainer = MultiOutputModelTrainer(MultiOutputModelConfig())
        
        # Pipeline state
        self.step7_features_loaded = False
        self.step2_5_sr_levels_loaded = False
        self.training_data_prepared = False
        
        # Output paths
        self.step7_output_path = config.get("step7_output_path", "data/matrix_operations")
        self.step2_5_output_path = config.get("step2_5_output_path", "data/sr_optimization")
        self.training_output_path = config.get("training_output_path", "data/training")
        
        self.logger.info("🔧 Comprehensive SR Training Pipeline initialized")

    @handles_errors(fallback=False)
    @performance_monitor
    @memory_efficient
    async def execute_comprehensive_training(
        self,
        training_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """
        Execute comprehensive training with full SR feature integration.
        
        Args:
            training_data: Input training data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            dict: Training results and model artifacts
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"🚀 Starting comprehensive SR training for {symbol} on {exchange}")
            
            # Step 1: Load step07 features
            step7_success = await self._load_step7_features()
            if not step7_success:
                self.logger.warning("⚠️ Step7 features not loaded, continuing with available features")
            
            # Step 2: Load step2_5 SR levels
            step2_5_success = await self._load_step2_5_sr_levels()
            if not step2_5_success:
                self.logger.warning("⚠️ Step2_5 SR levels not loaded, continuing with available features")
            
            # Step 3: Prepare comprehensive training data
            comprehensive_data = await self._prepare_comprehensive_training_data(training_data)
            
            # Step 4: Train models with comprehensive features
            training_results = await self._train_models_with_comprehensive_features(
                comprehensive_data, symbol, exchange, timeframe
            )
            
            # Step 5: Validate and save results
            validation_results = await self._validate_and_save_results(training_results)
            
            execution_time = datetime.now() - start_time
            self.logger.info(f"✅ Comprehensive SR training completed in {execution_time}")
            
            return {
                "success": True,
                "execution_time": execution_time.total_seconds(),
                "step7_features_loaded": step7_success,
                "step2_5_sr_levels_loaded": step2_5_success,
                "training_results": training_results,
                "validation_results": validation_results,
                "comprehensive_data_info": {
                    "total_features": len(comprehensive_data.columns),
                    "sr_features": len([col for col in comprehensive_data.columns if 'sr_' in col.lower()]),
                    "data_shape": comprehensive_data.shape
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive SR training failed: {e}")
            return {"success": False, "error": str(e)}

    async def _load_step7_features(self) -> bool:
        """Load step07 enhanced matrix operations features."""
        try:
            self.logger.info("📊 Loading step07 enhanced matrix operations features...")
            
            success = await self.model_trainer.load_step7_features(self.step7_output_path)
            
            if success:
                self.step7_features_loaded = True
                self.logger.info(f"✅ Step7 features loaded: {len(self.model_trainer.step7_features)} features")
            else:
                self.logger.warning("⚠️ Step7 features not available")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error loading step07 features: {e}")
            return False

    async def _load_step2_5_sr_levels(self) -> bool:
        """Load step2_5 SR optimization levels."""
        try:
            self.logger.info("📊 Loading step2_5 SR optimization levels...")
            
            success = await self.model_trainer.load_step2_5_sr_levels(self.step2_5_output_path)
            
            if success:
                self.step2_5_sr_levels_loaded = True
                support_levels = self.model_trainer.step2_5_sr_levels.get("support_levels", [])
                resistance_levels = self.model_trainer.step2_5_sr_levels.get("resistance_levels", [])
                self.logger.info(f"✅ Step2_5 SR levels loaded: {len(support_levels)} support, {len(resistance_levels)} resistance")
            else:
                self.logger.warning("⚠️ Step2_5 SR levels not available")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error loading step2_5 SR levels: {e}")
            return False

    async def _prepare_comprehensive_training_data(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive training data with all SR features."""
        try:
            self.logger.info("🔧 Preparing comprehensive training data...")
            
            # Add comprehensive SR features
            comprehensive_data = await self.model_trainer._add_comprehensive_sr_features(training_data)
            
            # Validate feature completeness
            missing_features = self.model_trainer.validate_feature_completeness(comprehensive_data)
            if missing_features:
                self.logger.warning(f"⚠️ Missing features: {missing_features}")
            
            # Log comprehensive data statistics
            sr_features = [col for col in comprehensive_data.columns if 'sr_' in col.lower()]
            self.logger.info(f"📊 Comprehensive data prepared:")
            self.logger.info(f"   - Total features: {len(comprehensive_data.columns)}")
            self.logger.info(f"   - SR features: {len(sr_features)}")
            self.logger.info(f"   - Data shape: {comprehensive_data.shape}")
            
            self.training_data_prepared = True
            return comprehensive_data
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing comprehensive training data: {e}")
            return training_data

    async def _train_models_with_comprehensive_features(
        self,
        comprehensive_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """Train models with comprehensive SR features."""
        try:
            self.logger.info("🚀 Training models with comprehensive SR features...")
            
            # Prepare targets
            direction_column = "direction"
            profit_column = "potential_profit_pct"
            
            if direction_column not in comprehensive_data.columns:
                self.logger.warning(f"⚠️ Direction column '{direction_column}' not found, using default")
                comprehensive_data[direction_column] = 1  # Default direction
            
            if profit_column not in comprehensive_data.columns:
                self.logger.warning(f"⚠️ Profit column '{profit_column}' not found, using default")
                comprehensive_data[profit_column] = 0.0  # Default profit
            
            # Prepare multi-output data
            features, direction_target, profit_target = await self.model_trainer.prepare_multi_output_data(
                comprehensive_data,
                direction_column=direction_column,
                profit_column=profit_column,
                use_enhanced_feature_selection=True
            )
            
            # Train models
            training_results = {}
            
            # Train different model types
            model_types = ["LightGBM", "RandomForest", "XGBoost"]
            
            for model_type in model_types:
                try:
                    self.logger.info(f"🔄 Training {model_type} model...")
                    
                    # Update model config
                    self.model_trainer.config.model_type = model_type
                    
                    # Train model
                    model_result = await self.model_trainer.train_multi_output_model(
                        features, direction_target, profit_target, f"{model_type}_{symbol}"
                    )
                    
                    if model_result:
                        training_results[model_type] = model_result
                        self.logger.info(f"✅ {model_type} model trained successfully")
                    else:
                        self.logger.warning(f"⚠️ {model_type} model training failed")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error training {model_type} model: {e}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Error training models with comprehensive features: {e}")
            return {}

    async def _validate_and_save_results(self, training_results: dict[str, Any]) -> dict[str, Any]:
        """Validate and save training results."""
        try:
            self.logger.info("🔍 Validating and saving training results...")
            
            validation_results = {
                "models_trained": len(training_results),
                "model_types": list(training_results.keys()),
                "sr_feature_analysis": {},
                "feature_completeness": {},
                "model_performance": {}
            }
            
            # Analyze SR feature usage across models
            for model_type, result in training_results.items():
                if "sr_feature_analysis" in result:
                    validation_results["sr_feature_analysis"][model_type] = result["sr_feature_analysis"]
                
                if "feature_importance" in result:
                    validation_results["model_performance"][model_type] = {
                        "direction_accuracy": result.get("direction_metrics", {}).get("accuracy", 0.0),
                        "profit_r2": result.get("profit_metrics", {}).get("r2_score", 0.0),
                        "feature_count": len(result.get("feature_importance", {}))
                    }
            
            # Save comprehensive results
            output_file = Path(self.training_output_path) / "comprehensive_sr_training_results.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "validation_results": validation_results,
                    "training_results": training_results,
                    "pipeline_state": {
                        "step7_features_loaded": self.step7_features_loaded,
                        "step2_5_sr_levels_loaded": self.step2_5_sr_levels_loaded,
                        "training_data_prepared": self.training_data_prepared
                    }
                }, f, indent=2)
            
            self.logger.info(f"✅ Results saved to: {output_file}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Error validating and saving results: {e}")
            return {"error": str(e)}

    def get_comprehensive_feature_summary(self) -> dict[str, Any]:
        """Get comprehensive feature summary."""
        try:
            summary = {
                "step7_features": {
                    "loaded": self.step7_features_loaded,
                    "count": len(self.model_trainer.step7_features),
                    "features": self.model_trainer.step7_features
                },
                "step2_5_sr_levels": {
                    "loaded": self.step2_5_sr_levels_loaded,
                    "support_levels": len(self.model_trainer.step2_5_sr_levels.get("support_levels", [])),
                    "resistance_levels": len(self.model_trainer.step2_5_sr_levels.get("resistance_levels", []))
                },
                "sr_feature_columns": {
                    "count": len(self.model_trainer.sr_feature_columns),
                    "columns": self.model_trainer.sr_feature_columns
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error getting feature summary: {e}")
            return {"error": str(e)}

# Convenience function for easy usage
async def run_comprehensive_sr_training(
    training_data: pd.DataFrame,
    symbol: str,
    exchange: str = "BINANCE",
    timeframe: str = "1m",
    config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Run comprehensive SR training pipeline.
    
    Args:
        training_data: Input training data
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        config: Optional configuration
        
    Returns:
        dict: Training results
    """
    if config is None:
        config = {}
    
    pipeline = ComprehensiveSRTrainingPipeline(config)
    return await pipeline.execute_comprehensive_training(training_data, symbol, exchange, timeframe)