#!/usr/bin/env python3
"""
Utility functions and decorators for HMM regime discovery.
Enhanced with common utilities integration.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List
import numpy as np
import pandas as pd

from ....core.decorators import handles_errors
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

# Import common utilities
from src.utils.common_operations import (
    validate_dataframe_columns,
    calculate_data_quality_metrics
)
from src.utils.common_utilities import safe_convert_dtypes
from src.utils.math_validation import safe_divide, safe_log
from src.utils.serialization_utils import JSONSerializer, PickleSerializer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from src.utils.ml_common.hmm_regime_detection import HMMRegimeDetector
from src.utils.ml_common.validation.cross_validation import TimeSeriesCrossValidator
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer


def create_fallback_logger() -> Any:
    """Create a fallback logger if system_logger is not available."""
    try:
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    except Exception as e:
        # If logging setup fails, create a minimal logger
        import sys
        class MinimalLogger:
            def info(self, msg): print(f"INFO: {msg}", file=sys.stdout)
            def warning(self, msg): print(f"WARNING: {msg}", file=sys.stderr)
            def error(self, msg): print(f"ERROR: {msg}", file=sys.stderr)
            def exception(self, msg): print(f"EXCEPTION: {msg}", file=sys.stderr)
        return MinimalLogger()

def ensure_directory(path: Path) -> Path:
    """Ensure directory exists and return the path."""
    try:
        if path is None:
            raise ValueError("Path cannot be None")
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logger = create_fallback_logger()
        logger.exception(f"Failed to create directory {path}: {e}")
        raise


class HMMCommonUtilities:
    """HMM utilities with common utilities integration."""
    
    def __init__(self):
        """Initialize HMM common utilities."""
        self.logger = create_fallback_logger()
        
        # Initialize common utilities
        self.matrix_ops = UnifiedMatrixOperations()
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()
        self.hmm_regime_detector = HMMRegimeDetector()
        self.cv_validator = TimeSeriesCrossValidator()
        self.hpo_optimizer = HyperparameterOptimizer()
        
        self.logger.info("🔧 HMM Common Utilities initialized")
    
    def prepare_features_with_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features with comprehensive validation using common utilities."""
        self.logger.info("🔧 Preparing features with validation...")
        
        try:
            # Validate DataFrame columns
            if not validate_dataframe_columns(data, data.columns.tolist()):
                self.logger.warning("DataFrame validation failed, proceeding with warnings")
            
            # Calculate data quality metrics
            quality_metrics = calculate_data_quality_metrics(data)
            self.logger.info(f"Data quality metrics: {quality_metrics}")
            
            # Convert dtypes for optimization
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            dtype_mapping = {col: 'float32' for col in numeric_columns}
            data_optimized = safe_convert_dtypes(data, dtype_mapping)
            
            # Use matrix operations for optimization if available
            if self.matrix_ops and hasattr(self.matrix_ops, 'optimize_for_clustering'):
                numeric_data = data_optimized[numeric_columns].values
                optimized_data = self.matrix_ops.optimize_for_clustering(numeric_data)
                data_optimized[numeric_columns] = optimized_data
            
            self.logger.info("✅ Features prepared with validation")
            return data_optimized
            
        except Exception as e:
            self.logger.error(f"❌ Feature preparation failed: {e}")
            raise
    
    def calculate_technical_indicators_safe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using safe math operations."""
        self.logger.info("📊 Calculating technical indicators...")
        
        try:
            result_data = data.copy()
            
            # RSI
            if 'close' in data.columns:
                result_data['rsi'] = TechnicalIndicators.calculate_rsi(data['close'])
            
            # MACD
            if 'close' in data.columns:
                result_data['macd'] = TechnicalIndicators.calculate_macd(data['close'])
            
            # Bollinger Bands
            if 'close' in data.columns:
                bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(data['close'])
                result_data['bb_upper'] = bb_upper
                result_data['bb_middle'] = bb_middle
                result_data['bb_lower'] = bb_lower
                result_data['bb_position'] = safe_divide(
                    data['close'] - bb_lower, 
                    bb_upper - bb_lower, 
                    0.5
                )
            
            # Volume indicators
            if 'volume' in data.columns and 'close' in data.columns:
                result_data['volume_ratio'] = safe_divide(
                    data['volume'], 
                    data['volume'].rolling(20).mean(), 
                    1.0
                )
            
            # Price indicators
            if 'close' in data.columns:
                result_data['returns'] = data['close'].pct_change()
                result_data['log_returns'] = safe_log(data['close'] / data['close'].shift(1))
                result_data['volatility'] = result_data['returns'].rolling(20).std()
            
            self.logger.info("✅ Technical indicators calculated")
            return result_data.dropna()
            
        except Exception as e:
            self.logger.error(f"❌ Technical indicators calculation failed: {e}")
            raise
    
    def run_cross_validation(self, model, data: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """Run cross-validation using ML common utilities."""
        self.logger.info(f"🔄 Running {cv_folds}-fold cross-validation...")
        
        try:
            if self.cv_validator and hasattr(self.cv_validator, 'cross_validate'):
                cv_results = self.cv_validator.cross_validate(
                    model, data, cv=cv_folds, scoring='neg_log_likelihood'
                )
            else:
                # Fallback cross-validation
                from sklearn.model_selection import KFold
                from sklearn.metrics import log_loss
                
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = []
                
                for train_idx, test_idx in kf.split(data):
                    X_train, X_test = data[train_idx], data[test_idx]
                    
                    # Fit model
                    model.fit(X_train)
                    
                    # Calculate score
                    score = model.score(X_test)
                    scores.append(score)
                
                cv_results = {
                    'test_score': np.array(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores)
                }
            
            self.logger.info(f"✅ Cross-validation completed: {cv_results.get('mean_score', 0):.3f} ± {cv_results.get('std_score', 0):.3f}")
            return cv_results
            
        except Exception as e:
            self.logger.error(f"❌ Cross-validation failed: {e}")
            raise
    
    def optimize_hyperparameters(self, model_class, data: np.ndarray, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Optimize hyperparameters using ML common utilities."""
        self.logger.info("🔧 Optimizing hyperparameters...")
        
        try:
            if self.hpo_optimizer and hasattr(self.hpo_optimizer, 'optimize'):
                best_params = self.hpo_optimizer.optimize(
                    model_class=model_class,
                    param_grid=param_grid,
                    X=data,
                    cv=5,
                    scoring='neg_log_likelihood'
                )
            else:
                # Fallback grid search
                from sklearn.model_selection import GridSearchCV
                
                grid_search = GridSearchCV(
                    model_class(), 
                    param_grid, 
                    cv=5, 
                    scoring='neg_log_likelihood',
                    n_jobs=-1
                )
                grid_search.fit(data)
                best_params = grid_search.best_params_
            
            self.logger.info(f"✅ Best parameters found: {best_params}")
            return best_params
            
        except Exception as e:
            self.logger.error(f"❌ Hyperparameter optimization failed: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> bool:
        """Save results using common serialization utilities."""
        self.logger.info(f"💾 Saving results to {filepath}")
        
        try:
            # Prepare results for serialization
            serializable_results = {}
            for key, value in results.items():
                if key in ['model', 'scaler']:
                    # Skip non-serializable objects
                    continue
                elif isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            
            # Save using appropriate serializer
            if filepath.endswith('.json'):
                success = self.json_serializer.save(serializable_results, filepath)
            else:
                success = self.pickle_serializer.save(serializable_results, filepath)
            
            if success:
                self.logger.info("✅ Results saved successfully")
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")
            return False
    
    def load_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load results using common serialization utilities."""
        self.logger.info(f"📂 Loading results from {filepath}")
        
        try:
            if filepath.endswith('.json'):
                results = self.json_serializer.load(filepath)
            else:
                results = self.pickle_serializer.load(filepath)
            
            if results:
                self.logger.info("✅ Results loaded successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load results: {e}")
            return None


class TechnicalIndicators:
    """Collection of technical indicator calculation methods."""
    
    @staticmethod
    @handles_errors(fallback=pd.Series())
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index using safe math operations."""
        try:
            if prices is None or prices.empty:
                raise ValueError("Prices series cannot be None or empty")
            if window < 1:
                raise ValueError("Window must be >= 1")
            if len(prices) < window:
                raise ValueError(f"Prices length ({len(prices)}) must be >= window ({window})")
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            # Use safe division to avoid division by zero
            rs = safe_divide(gain, loss, 1.0)  # Default to 1.0 if loss is 0
            rsi = 100 - safe_divide(100, 1 + rs, 50.0)  # Default to 50 if division fails
            return rsi.fillna(50)  # Fill NaN with neutral RSI value
        except Exception as e:
            logger = create_fallback_logger()
            logger.exception(f"RSI calculation failed: {e}")
            return pd.Series()

    @staticmethod
    @handles_errors(fallback=pd.Series())
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            if prices is None or prices.empty:
                raise ValueError("Prices series cannot be None or empty")
            if fast < 1 or slow < 1 or signal < 1:
                raise ValueError("All parameters must be >= 1")
            if fast >= slow:
                raise ValueError("Fast period must be < slow period")
            if len(prices) < slow:
                raise ValueError(f"Prices length ({len(prices)}) must be >= slow period ({slow})")
            
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except Exception as e:
            logger = create_fallback_logger()
            logger.exception(f"MACD calculation failed: {e}")
            return pd.Series()

    @staticmethod
    @handles_errors(fallback=pd.DataFrame())
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        try:
            if prices is None or prices.empty:
                raise ValueError("Prices series cannot be None or empty")
            if window < 1:
                raise ValueError("Window must be >= 1")
            if num_std <= 0:
                raise ValueError("num_std must be > 0")
            if len(prices) < window:
                raise ValueError(f"Prices length ({len(prices)}) must be >= window ({window})")
            
            sma = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            bb_upper = sma + std * num_std
            bb_lower = sma - std * num_std
            
            # Avoid division by zero
            bb_width = (bb_upper - bb_lower) / (sma + 1e-10)
            bb_position = (prices - bb_lower) / (bb_upper - bb_lower + 1e-10)
            
            bb_features = pd.DataFrame({
                'bb_upper': bb_upper, 
                'bb_middle': sma, 
                'bb_lower': bb_lower, 
                'bb_width': bb_width, 
                'bb_position': bb_position
            })
            return bb_features
        except Exception as e:
            logger = create_fallback_logger()
            logger.exception(f"Bollinger Bands calculation failed: {e}")
            return pd.DataFrame()


