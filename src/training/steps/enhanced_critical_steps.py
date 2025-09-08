from ..standardized_parquet_handler import standardized_parquet_handler
"""
Enhanced Critical Training Steps with Fail-Fast Behavior

This module provides enhanced versions of critical training steps that ensure:
1. No silent failures
2. Proper error propagation
3. Fail-fast behavior for critical processes
4. Comprehensive validation and monitoring
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.utils.logger import system_logger
from .enhanced_error_handling import (
import logging
import numpy as np
import time

    enhanced_error_handler,
    enhanced_async_error_handler,
    critical_process,
    critical_async_process,
    CriticalProcessError,
    ErrorSeverity,
    ErrorCategory
)

class EnhancedHMMClusteringStep:
    """Enhanced HMM clustering step with fail-fast behavior."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('EnhancedHMMClustering')
        self.error_handler = None  # Will be set by decorator
    
    @critical_async_process('hmm_clustering')
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HMM clustering with fail-fast behavior."""
        self.logger.info('🎯 Starting Enhanced HMM Clustering execution...')
        
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir')
            force_rerun = training_input.get('force_rerun', False)
            
            # Validate inputs
            if not symbol or not exchange or not timeframe:
                raise ValueError("Missing required parameters: symbol, exchange, timeframe")
            
            if not data_dir:
                raise ValueError("Data directory is required")
            
            # Check if data exists
            data_path = Path(data_dir) / f"{exchange}_{symbol}_processed.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"Required data file not found: {data_path}")
            
            # Load and validate data
            data = standardized_parquet_handler.read_parquet_standardized(data_path)
            if data.empty:
                raise ValueError("Loaded data is empty")
            
            # Validate required columns
            required_columns = ['close', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check data quality
            if data.isnull().sum().sum() > len(data) * 0.1:  # More than 10% NaN
                raise ValueError("Data quality too poor: too many NaN values")
            
            self.logger.info(f'✅ Data validation passed: {len(data)} rows, {len(data.columns)} columns')
            
            # Execute HMM clustering
            success = await self._perform_hmm_clustering(data, symbol, exchange, timeframe, data_dir)
            
            if not success:
                raise RuntimeError("HMM clustering execution failed")
            
            # Update pipeline state
            pipeline_state['hmm_clustering_completed'] = True
            pipeline_state['hmm_clustering_timestamp'] = datetime.now().isoformat()
            pipeline_state['hmm_clustering_data_shape'] = data.shape
            
            self.logger.info('✅ Enhanced HMM Clustering completed successfully')
            return pipeline_state
            
        except Exception as e:
            self.logger.exception(f'❌ HMM Clustering failed: {e}')
            raise  # Re-raise to trigger fail-fast
    
    async def _perform_hmm_clustering(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Perform the actual HMM clustering."""
        try:
            # Import and run HMM clustering
            from .market_analysis.hmm_clustering import run_enhanced_step
            
            enhanced_config = {
                'n_trials': 50,
                'timeout_minutes': 15,
                'cv_folds': 3,
                'random_state': 42,
                'ensemble_weights': {'hmm': 0.4, 'kmeans': 0.3, 'dbscan': 0.3},
                'initial_features': 20,
                'feature_increment': 10,
                'max_features': 100,
                'min_improvement': 0.001,
                'patience': 3
            }
            
            success = await run_enhanced_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=True,
                **enhanced_config
            )
            
            if not success:
                raise RuntimeError("HMM clustering algorithm failed")
            
            return True
            
        except Exception as e:
            self.logger.exception(f'HMM clustering algorithm failed: {e}')
            return False

class EnhancedFeatureGenerationStep:
    """Enhanced feature generation step with fail-fast behavior."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('EnhancedFeatureGeneration')
        self.error_handler = None  # Will be set by decorator
    
    @critical_async_process('feature_generation')
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature generation with fail-fast behavior."""
        self.logger.info('🎯 Starting Enhanced Feature Generation execution...')
        
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir')
            
            # Validate inputs
            if not symbol or not exchange or not timeframe:
                raise ValueError("Missing required parameters: symbol, exchange, timeframe")
            
            if not data_dir:
                raise ValueError("Data directory is required")
            
            # Check if HMM clustering was completed
            if not pipeline_state.get('hmm_clustering_completed', False):
                raise ValueError("HMM clustering must be completed before feature generation")
            
            # Load data
            data_path = Path(data_dir) / f"{exchange}_{symbol}_processed.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"Required data file not found: {data_path}")
            
            data = standardized_parquet_handler.read_parquet_standardized(data_path)
            if data.empty:
                raise ValueError("Loaded data is empty")
            
            # Execute feature generation
            success = await self._perform_feature_generation(data, symbol, exchange, timeframe, data_dir)
            
            if not success:
                raise RuntimeError("Feature generation execution failed")
            
            # Update pipeline state
            pipeline_state['feature_generation_completed'] = True
            pipeline_state['feature_generation_timestamp'] = datetime.now().isoformat()
            pipeline_state['feature_generation_data_shape'] = data.shape
            
            self.logger.info('✅ Enhanced Feature Generation completed successfully')
            return pipeline_state
            
        except Exception as e:
            self.logger.exception(f'❌ Feature Generation failed: {e}')
            raise  # Re-raise to trigger fail-fast
    
    async def _perform_feature_generation(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Perform the actual feature generation."""
        try:
            # Import and run feature generation
            from .market_analysis.step06_feature_engineering import FeatureInteractionEngine
            
            config = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'step06_feature_engineering': {
                    'enable_interactions': True,
                    'max_interactions': 100,
                    'interaction_threshold': 0.1
                }
            }
            
            engine = FeatureInteractionEngine(config)
            
            # Generate features
            features = await engine.generate_interaction_features(data)
            
            if features is None or features.empty:
                raise RuntimeError("Feature generation produced no features")
            
            # Validate feature quality
            if features.isnull().sum().sum() > len(features) * 0.2:  # More than 20% NaN
                raise ValueError("Generated features have too many NaN values")
            
            # Save features
            features_path = Path(data_dir) / f"{exchange}_{symbol}_features.parquet"
            standardized_parquet_handler.write_parquet_standardized(features, features_path)
            
            self.logger.info(f'✅ Features generated and saved: {features.shape}')
            return True
            
        except Exception as e:
            self.logger.exception(f'Feature generation algorithm failed: {e}')
            return False

class EnhancedMatrixOperationsStep:
    """Enhanced matrix operations step with fail-fast behavior."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('EnhancedMatrixOperations')
        self.error_handler = None  # Will be set by decorator
    
    @critical_async_process('matrix_operations')
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute matrix operations with fail-fast behavior."""
        self.logger.info('🎯 Starting Enhanced Matrix Operations execution...')
        
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir')
            
            # Validate inputs
            if not symbol or not exchange or not timeframe:
                raise ValueError("Missing required parameters: symbol, exchange, timeframe")
            
            if not data_dir:
                raise ValueError("Data directory is required")
            
            # Check if feature generation was completed
            if not pipeline_state.get('feature_generation_completed', False):
                raise ValueError("Feature generation must be completed before matrix operations")
            
            # Load features
            features_path = Path(data_dir) / f"{exchange}_{symbol}_features.parquet"
            if not features_path.exists():
                raise FileNotFoundError(f"Required features file not found: {features_path}")
            
            features = standardized_parquet_handler.read_parquet_standardized(features_path)
            if features.empty:
                raise ValueError("Loaded features are empty")
            
            # Execute matrix operations
            success = await self._perform_matrix_operations(features, symbol, exchange, timeframe, data_dir)
            
            if not success:
                raise RuntimeError("Matrix operations execution failed")
            
            # Update pipeline state
            pipeline_state['matrix_operations_completed'] = True
            pipeline_state['matrix_operations_timestamp'] = datetime.now().isoformat()
            pipeline_state['matrix_operations_data_shape'] = features.shape
            
            self.logger.info('✅ Enhanced Matrix Operations completed successfully')
            return pipeline_state
            
        except Exception as e:
            self.logger.exception(f'❌ Matrix Operations failed: {e}')
            raise  # Re-raise to trigger fail-fast
    
    async def _perform_matrix_operations(self, features: pd.DataFrame, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Perform the actual matrix operations."""
        try:
            # Import and run matrix operations
            from .market_analysis.step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep
            
            config = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir
            }
            
            step = EnhancedMatrixOperationsStep(config)
            
            # Perform matrix operations
            result = await step.execute_matrix_operations(features)
            
            if result is None or result.empty:
                raise RuntimeError("Matrix operations produced no results")
            
            # Validate result quality
            if result.isnull().sum().sum() > len(result) * 0.1:  # More than 10% NaN
                raise ValueError("Matrix operations result has too many NaN values")
            
            # Save results
            result_path = Path(data_dir) / f"{exchange}_{symbol}_matrix_operations.parquet"
            standardized_parquet_handler.write_parquet_standardized(result, result_path)
            
            self.logger.info(f'✅ Matrix operations completed and saved: {result.shape}')
            return True
            
        except Exception as e:
            self.logger.exception(f'Matrix operations algorithm failed: {e}')
            return False

class EnhancedMLModelTrainingStep:
    """Enhanced ML model training step with fail-fast behavior."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('EnhancedMLModelTraining')
        self.error_handler = None  # Will be set by decorator
    
    @critical_async_process('ml_model_training')
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ML model training with fail-fast behavior."""
        self.logger.info('🎯 Starting Enhanced ML Model Training execution...')
        
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir')
            
            # Validate inputs
            if not symbol or not exchange or not timeframe:
                raise ValueError("Missing required parameters: symbol, exchange, timeframe")
            
            if not data_dir:
                raise ValueError("Data directory is required")
            
            # Check if matrix operations were completed
            if not pipeline_state.get('matrix_operations_completed', False):
                raise ValueError("Matrix operations must be completed before ML model training")
            
            # Load processed data
            data_path = Path(data_dir) / f"{exchange}_{symbol}_matrix_operations.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"Required processed data file not found: {data_path}")
            
            data = standardized_parquet_handler.read_parquet_standardized(data_path)
            if data.empty:
                raise ValueError("Loaded processed data is empty")
            
            # Execute ML model training
            success = await self._perform_ml_model_training(data, symbol, exchange, timeframe, data_dir)
            
            if not success:
                raise RuntimeError("ML model training execution failed")
            
            # Update pipeline state
            pipeline_state['ml_model_training_completed'] = True
            pipeline_state['ml_model_training_timestamp'] = datetime.now().isoformat()
            pipeline_state['ml_model_training_data_shape'] = data.shape
            
            self.logger.info('✅ Enhanced ML Model Training completed successfully')
            return pipeline_state
            
        except Exception as e:
            self.logger.exception(f'❌ ML Model Training failed: {e}')
            raise  # Re-raise to trigger fail-fast
    
    async def _perform_ml_model_training(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Perform the actual ML model training."""
        try:
            # Import and run ML model training
            from .model_training.step09_hmm_based_training import HMMBasedTrainingStep
            
            config = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'training_config': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'cv_folds': 5,
                    'n_trials': 100
                }
            }
            
            step = HMMBasedTrainingStep(config)
            
            # Perform model training
            result = await step.execute_training(data)
            
            if not result:
                raise RuntimeError("ML model training algorithm failed")
            
            # Validate model was saved
            model_path = Path(data_dir) / f"{exchange}_{symbol}_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError("Trained model was not saved")
            
            self.logger.info(f'✅ ML model training completed and model saved')
            return True
            
        except Exception as e:
            self.logger.exception(f'ML model training algorithm failed: {e}')
            return False

class EnhancedSRLevelsDetectionStep:
    """Enhanced SR levels detection step with fail-fast behavior."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('EnhancedSRLevelsDetection')
        self.error_handler = None  # Will be set by decorator
    
    @critical_async_process('sr_levels_detection')
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SR levels detection with fail-fast behavior."""
        self.logger.info('🎯 Starting Enhanced SR Levels Detection execution...')
        
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir')
            
            # Validate inputs
            if not symbol or not exchange or not timeframe:
                raise ValueError("Missing required parameters: symbol, exchange, timeframe")
            
            if not data_dir:
                raise ValueError("Data directory is required")
            
            # Load data
            data_path = Path(data_dir) / f"{exchange}_{symbol}_processed.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"Required data file not found: {data_path}")
            
            data = standardized_parquet_handler.read_parquet_standardized(data_path)
            if data.empty:
                raise ValueError("Loaded data is empty")
            
            # Execute SR levels detection
            success = await self._perform_sr_levels_detection(data, symbol, exchange, timeframe, data_dir)
            
            if not success:
                raise RuntimeError("SR levels detection execution failed")
            
            # Update pipeline state
            pipeline_state['sr_levels_detection_completed'] = True
            pipeline_state['sr_levels_detection_timestamp'] = datetime.now().isoformat()
            
            self.logger.info('✅ Enhanced SR Levels Detection completed successfully')
            return pipeline_state
            
        except Exception as e:
            self.logger.exception(f'❌ SR Levels Detection failed: {e}')
            raise  # Re-raise to trigger fail-fast
    
    async def _perform_sr_levels_detection(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Perform the actual SR levels detection."""
        try:
            # Import and run SR levels detection
            from src.tactician.sr_levels.sr_breakout_predictor_enhanced import SRBreakoutPredictor
            
            predictor = SRBreakoutPredictor()
            
            # Detect SR levels
            sr_levels = await predictor.detect_sr_levels(data)
            
            if sr_levels is None or len(sr_levels) == 0:
                raise RuntimeError("SR levels detection produced no results")
            
            # Validate SR levels quality
            if len(sr_levels) < 5:  # Minimum number of SR levels
                raise ValueError("Too few SR levels detected")
            
            # Save SR levels
            sr_levels_path = Path(data_dir) / f"{exchange}_{symbol}_sr_levels.json"
            import json
            with open(sr_levels_path, 'w') as f:
                json.dump(sr_levels, f, indent=2)
            
            self.logger.info(f'✅ SR levels detected and saved: {len(sr_levels)} levels')
            return True
            
        except Exception as e:
            self.logger.exception(f'SR levels detection algorithm failed: {e}')
            return False

# Factory function to create enhanced steps
def create_enhanced_step(step_name: str, config: Dict[str, Any]):
    """Create an enhanced step instance."""
    step_classes = {
        'hmm_clustering': EnhancedHMMClusteringStep,
        'feature_generation': EnhancedFeatureGenerationStep,
        'matrix_operations': EnhancedMatrixOperationsStep,
        'ml_model_training': EnhancedMLModelTrainingStep,
        'sr_levels_detection': EnhancedSRLevelsDetectionStep
    }
    
    if step_name not in step_classes:
        raise ValueError(f"Unknown step name: {step_name}")
    
    return step_classes[step_name](config)

# Main execution function
async def run_enhanced_critical_step(step_name: str, 
                                   training_input: Dict[str, Any], 
                                   pipeline_state: Dict[str, Any],
                                   config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run an enhanced critical step with fail-fast behavior.
    
    Args:
        step_name: Name of the step to run
        training_input: Training input data
        pipeline_state: Current pipeline state
        config: Step configuration
        
    Returns:
        Updated pipeline state
        
    Raises:
        CriticalProcessError: If the step fails critically
    """
    try:
        step = create_enhanced_step(step_name, config)
        return await step.execute(training_input, pipeline_state)
    except Exception as e:
        logger = system_logger.getChild('EnhancedCriticalSteps')
        logger.exception(f'Failed to run enhanced critical step {step_name}: {e}')
        raise