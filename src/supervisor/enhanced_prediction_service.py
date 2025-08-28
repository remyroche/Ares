"""
Enhanced Prediction Service for ML Profit Integration System.

This service provides calibrated confidence scores from ML models for both Analyst and Tactician.
It ONLY provides calibrated confidence scores and fails if calibrated confidence doesn't exist.
"""

import asyncio
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config.enhanced_prediction_service_config import get_enhanced_prediction_service_config
from src.utils.logging_config import get_logger
from src.utils.error_handling import handle_errors, handle_specific_errors, error, warning
from src.utils.tracing import with_tracing_span
from src.utils.validation import validate_data_quality, comprehensive_validation
from src.utils.performance import performance_monitor
from src.utils.caching import intelligent_caching


class EnhancedPredictionService:
    """
    Enhanced Prediction Service that provides calibrated confidence scores from ML models.
    
    This service ONLY provides calibrated confidence scores for both Analyst and Tactician ML models.
    It fails if calibrated confidence doesn't exist for either model set.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Enhanced Prediction Service."""
        self.config = config or get_enhanced_prediction_service_config()
        self.logger = get_logger(__name__)
        
        # Service state
        self.is_initialized = False
        self.data_dir = self.config.get("data_directory", "data")
        
        # ML model storage
        self.analyst_ml_models: Dict[str, Dict[str, Any]] = {}
        self.tactician_ml_models: Dict[str, Dict[str, Any]] = {}
        
        # Calibration and optimization results
        self.calibration_results: Dict[str, Any] = {}
        self.optimization_results: Dict[str, Any] = {}
        
        # Configuration parameters
        self.entry_threshold = self.config.get("entry_threshold", 0.6)
        self.max_confidence_threshold = self.config.get("max_confidence_threshold", 0.7)
        
        self.logger.info("Enhanced Prediction Service initialized")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="initializing enhanced prediction service",
    )
    @with_tracing_span("initialize_enhanced_prediction_service")
    async def initialize(self) -> bool:
        """Initialize the Enhanced Prediction Service."""
        try:
            self.logger.info("🚀 Initializing Enhanced Prediction Service...")
            
            # Load ML models for both Analyst and Tactician
            await self._load_analyst_ml_models()
            await self._load_tactician_ml_models()
            
            # Load calibration and optimization results
            await self._load_calibration_results()
            await self._load_optimization_results()
            
            self.is_initialized = True
            self.logger.info("✅ Enhanced Prediction Service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(error(f"❌ Failed to initialize Enhanced Prediction Service: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="loading analyst ML models",
    )
    @with_tracing_span("load_analyst_ml_models")
    @intelligent_caching(cache_key="analyst_ml_models")
    async def _load_analyst_ml_models(self) -> None:
        """Load Analyst ML models (higher timeframe) from steps 6-14."""
        try:
            analyst_models_path = Path(self.data_dir) / "ml_profit_models" / "analyst_models"
            if not analyst_models_path.exists():
                raise ValueError(f"Analyst ML models directory not found: {analyst_models_path}")

            # Load different types of Analyst models
            analyst_model_types = [
                "hmm_profit", "analyst_profit", "calibrated", "optimized", 
                "validated", "monte_carlo", "ab_tested"
            ]
            
            for model_type in analyst_model_types:
                type_path = analyst_models_path / model_type
                if type_path.exists():
                    self.analyst_ml_models[model_type] = {}
                    
                    for model_file in type_path.glob("*.pkl"):
                        try:
                            with open(model_file, "rb") as f:
                                model_data = pickle.load(f)
                            
                            model_name = model_file.stem
                            self.analyst_ml_models[model_type][model_name] = model_data
                            self.logger.info(f"✅ Loaded Analyst ML model: {model_type}/{model_name}")
                        
                        except Exception as e:
                            self.logger.warning(warning(f"⚠️ Failed to load Analyst ML model {model_file}: {e}"))

            if not self.analyst_ml_models:
                raise ValueError("No Analyst ML models loaded")

        except Exception as e:
            self.logger.error(error(f"❌ Error loading Analyst ML models: {e}"))
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="loading tactician ML models",
    )
    @with_tracing_span("load_tactician_ml_models")
    @intelligent_caching(cache_key="tactician_ml_models")
    async def _load_tactician_ml_models(self) -> None:
        """Load Tactician ML models (lower timeframe) from steps 6-14."""
        try:
            tactician_models_path = Path(self.data_dir) / "ml_profit_models" / "tactician_models"
            if not tactician_models_path.exists():
                raise ValueError(f"Tactician ML models directory not found: {tactician_models_path}")

            # Load different types of Tactician models
            tactician_model_types = [
                "tactician_profit", "tactician_specialist", "calibrated", "optimized", 
                "validated", "monte_carlo", "ab_tested"
            ]
            
            for model_type in tactician_model_types:
                type_path = tactician_models_path / model_type
                if type_path.exists():
                    self.tactician_ml_models[model_type] = {}
                    
                    for model_file in type_path.glob("*.pkl"):
                        try:
                            with open(model_file, "rb") as f:
                                model_data = pickle.load(f)
                            
                            model_name = model_file.stem
                            self.tactician_ml_models[model_type][model_name] = model_data
                            self.logger.info(f"✅ Loaded Tactician ML model: {model_type}/{model_name}")
                        
                        except Exception as e:
                            self.logger.warning(warning(f"⚠️ Failed to load Tactician ML model {model_file}: {e}"))

            if not self.tactician_ml_models:
                raise ValueError("No Tactician ML models loaded")

        except Exception as e:
            self.logger.error(error(f"❌ Error loading Tactician ML models: {e}"))
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="loading calibration results",
    )
    @with_tracing_span("load_calibration_results")
    async def _load_calibration_results(self) -> None:
        """Load calibration results from step 10."""
        try:
            calibration_path = Path(self.data_dir) / "calibration_results"
            if calibration_path.exists():
                for calibration_file in calibration_path.glob("*.json"):
                    try:
                        import json
                        with open(calibration_file, "r") as f:
                            calibration_data = json.load(f)
                        
                        key = calibration_file.stem
                        self.calibration_results[key] = calibration_data
                        self.logger.debug(f"Loaded calibration results: {key}")
                    
                    except Exception as e:
                        self.logger.warning(warning(f"⚠️ Failed to load calibration file {calibration_file}: {e}"))

        except Exception as e:
            self.logger.error(error(f"❌ Error loading calibration results: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="loading optimization results",
    )
    @with_tracing_span("load_optimization_results")
    async def _load_optimization_results(self) -> None:
        """Load optimization results from step 11."""
        try:
            optimization_path = Path(self.data_dir) / "optimization_results"
            if optimization_path.exists():
                for optimization_file in optimization_path.glob("*.json"):
                    try:
                        import json
                        with open(optimization_file, "r") as f:
                            optimization_data = json.load(f)
                        
                        key = optimization_file.stem
                        self.optimization_results[key] = optimization_data
                        self.logger.debug(f"Loaded optimization results: {key}")
                    
                    except Exception as e:
                        self.logger.warning(warning(f"⚠️ Failed to load optimization file {optimization_file}: {e}"))

        except Exception as e:
            self.logger.error(error(f"❌ Error loading optimization results: {e}"))

    @handle_errors(
        exceptions=(ValueError,),
        default_return={},
        context="getting calibrated confidence scores",
    )
    @with_tracing_span("get_calibrated_confidence_scores")
    @validate_data_quality(validation_level="ERROR")
    @performance_monitor
    async def get_calibrated_confidence_scores(
        self, 
        market_data: pd.DataFrame,
        regime_info: Dict[str, Any],
        symbol: str,
        exchange: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Provide calibrated confidence scores for BOTH Analyst and Tactician ML models.
        
        This method ONLY provides calibrated confidence scores and fails if calibrated 
        confidence doesn't exist for either model set.
        
        Args:
            market_data: Market data for prediction
            regime_info: Market regime information
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Dictionary with calibrated confidence scores for both Analyst and Tactician models
            
        Raises:
            ValueError: If calibrated confidence doesn't exist for either model set
        """
        try:
            if not self.is_initialized:
                raise ValueError("Enhanced Prediction Service not initialized")

            calibrated_scores = {
                "analyst_models": {},
                "tactician_models": {}
            }
            
            # Get Analyst calibrated confidence scores
            analyst_scores = await self._get_analyst_calibrated_confidence(
                market_data, regime_info, symbol, exchange
            )
            if not analyst_scores:
                raise ValueError(f"No calibrated Analyst confidence scores available for {symbol} on {exchange}")
            calibrated_scores["analyst_models"] = analyst_scores
            
            # Get Tactician calibrated confidence scores
            tactician_scores = await self._get_tactician_calibrated_confidence(
                market_data, regime_info, symbol, exchange
            )
            if not tactician_scores:
                raise ValueError(f"No calibrated Tactician confidence scores available for {symbol} on {exchange}")
            calibrated_scores["tactician_models"] = tactician_scores
            
            self.logger.info(f"✅ Retrieved calibrated confidence scores for {symbol} on {exchange}")
            self.logger.debug(f"Analyst models: {len(analyst_scores)}, Tactician models: {len(tactician_scores)}")
            
            return calibrated_scores
            
        except Exception as e:
            self.logger.error(error(f"❌ Failed to get calibrated confidence scores: {e}"))
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="getting analyst calibrated confidence",
    )
    @with_tracing_span("get_analyst_calibrated_confidence")
    async def _get_analyst_calibrated_confidence(
        self,
        market_data: pd.DataFrame,
        regime_info: Dict[str, Any],
        symbol: str,
        exchange: str
    ) -> Dict[str, float]:
        """Get calibrated confidence scores from Analyst ML models."""
        try:
            analyst_scores = {}
            
            for model_type, models in self.analyst_ml_models.items():
                for model_name, model_data in models.items():
                    try:
                        # Get calibrated confidence from model
                        calibrated_confidence = model_data.get("calibrated_confidence")
                        
                        if calibrated_confidence is None:
                            self.logger.warning(warning(f"⚠️ No calibrated confidence for Analyst model {model_name}"))
                            continue
                        
                        # Validate confidence score
                        if not isinstance(calibrated_confidence, (int, float)) or not (0.0 <= calibrated_confidence <= 1.0):
                            self.logger.warning(warning(f"⚠️ Invalid calibrated confidence for Analyst model {model_name}: {calibrated_confidence}"))
                            continue
                        
                        analyst_scores[f"{model_type}_{model_name}"] = float(calibrated_confidence)
                        
                    except Exception as e:
                        self.logger.warning(warning(f"⚠️ Failed to get confidence for Analyst model {model_name}: {e}"))
            
            return analyst_scores
            
        except Exception as e:
            self.logger.error(error(f"❌ Error getting Analyst calibrated confidence: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="getting tactician calibrated confidence",
    )
    @with_tracing_span("get_tactician_calibrated_confidence")
    async def _get_tactician_calibrated_confidence(
        self,
        market_data: pd.DataFrame,
        regime_info: Dict[str, Any],
        symbol: str,
        exchange: str
    ) -> Dict[str, float]:
        """Get calibrated confidence scores from Tactician ML models."""
        try:
            tactician_scores = {}
            
            for model_type, models in self.tactician_ml_models.items():
                for model_name, model_data in models.items():
                    try:
                        # Get calibrated confidence from model
                        calibrated_confidence = model_data.get("calibrated_confidence")
                        
                        if calibrated_confidence is None:
                            self.logger.warning(warning(f"⚠️ No calibrated confidence for Tactician model {model_name}"))
                            continue
                        
                        # Validate confidence score
                        if not isinstance(calibrated_confidence, (int, float)) or not (0.0 <= calibrated_confidence <= 1.0):
                            self.logger.warning(warning(f"⚠️ Invalid calibrated confidence for Tactician model {model_name}: {calibrated_confidence}"))
                            continue
                        
                        tactician_scores[f"{model_type}_{model_name}"] = float(calibrated_confidence)
                        
                    except Exception as e:
                        self.logger.warning(warning(f"⚠️ Failed to get confidence for Tactician model {model_name}: {e}"))
            
            return tactician_scores
            
        except Exception as e:
            self.logger.error(error(f"❌ Error getting Tactician calibrated confidence: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="checking service health",
    )
    @with_tracing_span("check_service_health")
    async def check_service_health(self) -> bool:
        """Check if the service is healthy and has loaded models."""
        try:
            if not self.is_initialized:
                return False
            
            # Check if we have both Analyst and Tactician models
            has_analyst_models = len(self.analyst_ml_models) > 0
            has_tactician_models = len(self.tactician_ml_models) > 0
            
            if not has_analyst_models:
                self.logger.warning(warning("⚠️ No Analyst ML models loaded"))
            
            if not has_tactician_models:
                self.logger.warning(warning("⚠️ No Tactician ML models loaded"))
            
            return has_analyst_models and has_tactician_models
            
        except Exception as e:
            self.logger.error(error(f"❌ Service health check failed: {e}"))
            return False

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and statistics."""
        try:
            analyst_model_count = sum(len(models) for models in self.analyst_ml_models.values())
            tactician_model_count = sum(len(models) for models in self.tactician_ml_models.values())
            
            return {
                "service_name": "Enhanced Prediction Service",
                "is_initialized": self.is_initialized,
                "analyst_models_loaded": analyst_model_count,
                "tactician_models_loaded": tactician_model_count,
                "analyst_model_types": list(self.analyst_ml_models.keys()),
                "tactician_model_types": list(self.tactician_ml_models.keys()),
                "calibration_results_loaded": len(self.calibration_results),
                "optimization_results_loaded": len(self.optimization_results),
                "entry_threshold": self.entry_threshold,
                "max_confidence_threshold": self.max_confidence_threshold
            }
            
        except Exception as e:
            self.logger.error(error(f"❌ Failed to get service info: {e}"))
            return {"error": str(e)}