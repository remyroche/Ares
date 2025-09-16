#!/usr/bin/env python3
"""
HMM Clustering Integration Example

This module demonstrates how to integrate the enhanced HMM clustering
with all available common utilities for comprehensive market analysis.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

# Import common utilities
from src.utils.common_operations import (
    get_m1_gpu_manager,
    get_m1_memory_optimizer, 
    get_m1_cpu_optimizer,
    safe_dataframe_operation,
    validate_dataframe_columns,
    calculate_data_quality_metrics
)
from src.utils.common_utilities import safe_convert_dtypes
from src.utils.math_validation import safe_divide, safe_log, validate_finite
from src.utils.serialization_utils import JSONSerializer, PickleSerializer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from src.utils.ml_common.hmm_regime_detection import HMMRegimeDetector
from src.utils.ml_common.validation.cross_validation import TimeSeriesCrossValidator
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer
from src.utils.kline_parquet import KlineParquetHandler
from src.utils.logger import system_logger

# Import the enhanced HMM clustering
from .enhanced_hmm_clustering import EnhancedHMMClustering, HMMClusteringConfig, HMMClusteringResults

logger = system_logger.getChild('HMMClusteringIntegration')

class HMMClusteringIntegration:
    """
    Comprehensive integration of HMM clustering with all common utilities.
    
    This class demonstrates how to use all available utilities together
    for a complete market analysis pipeline.
    """
    
    def __init__(self, config: Optional[HMMClusteringConfig] = None):
        """Initialize the integration."""
        self.logger = logger.getChild('HMMClusteringIntegration')
        
        # Initialize all common utilities
        self._initialize_utilities()
        
        # Initialize HMM clustering
        self.hmm_clustering = EnhancedHMMClustering(config or HMMClusteringConfig())
        
        # Initialize data handlers
        self.kline_handler = KlineParquetHandler()
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()
        
        self.logger.info("🚀 HMM Clustering Integration initialized with all utilities")
    
    def _initialize_utilities(self):
        """Initialize all common utilities."""
        self.logger.info("🔧 Initializing common utilities...")
        
        # Hardware utilities
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        # Matrix operations
        self.matrix_ops = UnifiedMatrixOperations()
        
        # ML utilities
        self.cv_validator = TimeSeriesCrossValidator()
        self.hpo_optimizer = HyperparameterOptimizer()
        self.hmm_regime_detector = HMMRegimeDetector()
        
        # Log utility status
        self.logger.info(f"✅ GPU Manager: {'Available' if self.gpu_manager else 'Not Available'}")
        self.logger.info(f"✅ Memory Optimizer: {'Available' if self.memory_optimizer else 'Not Available'}")
        self.logger.info(f"✅ CPU Optimizer: {'Available' if self.cpu_optimizer else 'Not Available'}")
        self.logger.info(f"✅ Matrix Operations: {'Available' if self.matrix_ops else 'Not Available'}")
    
    def load_market_data(self, filepath: str, symbol: str = 'BTCUSDT', 
                        timeframe: str = '1h') -> pd.DataFrame:
        """Load market data using kline parquet handler."""
        self.logger.info(f"📊 Loading market data: {symbol} {timeframe}")
        
        try:
            # Load data using kline handler
            data = self.kline_handler.load_klines(
                filepath=filepath,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Validate data quality
            quality_metrics = calculate_data_quality_metrics(data)
            self.logger.info(f"📈 Data quality metrics: {quality_metrics}")
            
            # Convert dtypes for optimization
            dtype_mapping = {
                'open': 'float32',
                'high': 'float32', 
                'low': 'float32',
                'close': 'float32',
                'volume': 'float32'
            }
            data = safe_convert_dtypes(data, dtype_mapping)
            
            self.logger.info(f"✅ Market data loaded: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load market data: {e}")
            raise
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for HMM clustering using common utilities."""
        self.logger.info("🔧 Preparing features...")
        
        try:
            # Create feature DataFrame
            features = pd.DataFrame()
            
            # Price-based features
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = safe_log(data['close'] / data['close'].shift(1))
            features['volatility'] = features['returns'].rolling(20).std()
            features['price_momentum'] = data['close'] / data['close'].shift(20) - 1
            
            # Volume-based features
            features['volume_ratio'] = safe_divide(data['volume'], data['volume'].rolling(20).mean())
            features['volume_momentum'] = data['volume'] / data['volume'].shift(20) - 1
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(data['close'])
            features['macd'] = self._calculate_macd(data['close'])
            features['bollinger_position'] = self._calculate_bollinger_position(data)
            
            # Cross-timeframe features
            features['high_low_ratio'] = safe_divide(data['high'], data['low'])
            features['close_open_ratio'] = safe_divide(data['close'], data['open'])
            
            # Remove NaN values
            features = features.dropna()
            
            # Validate features
            if not validate_dataframe_columns(features, features.columns.tolist()):
                raise ValueError("Feature validation failed")
            
            # Use matrix operations for optimization
            if self.matrix_ops:
                features_array = self.matrix_ops.optimize_for_clustering(features.values)
                features = pd.DataFrame(features_array, columns=features.columns)
            
            self.logger.info(f"✅ Features prepared: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Feature preparation failed: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI using safe math operations."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = safe_divide(avg_gain, avg_loss)
        rsi = 100 - safe_divide(100, 1 + rs)
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD using safe math operations."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        return macd_line - signal_line
    
    def _calculate_bollinger_position(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Bollinger Bands position."""
        close = data['close']
        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        # Position within bands (0 = lower band, 1 = upper band)
        position = safe_divide(close - lower_band, upper_band - lower_band)
        
        return position
    
    def run_comprehensive_analysis(self, data: pd.DataFrame, 
                                 optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """Run comprehensive HMM clustering analysis."""
        self.logger.info("🚀 Starting comprehensive HMM clustering analysis...")
        
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Optimize hyperparameters if requested
            best_params = {}
            if optimize_hyperparams:
                self.logger.info("🔧 Optimizing hyperparameters...")
                best_params = self.hmm_clustering.optimize_hyperparameters(features)
                
                # Update config with best parameters
                if best_params:
                    for param, value in best_params.items():
                        setattr(self.hmm_clustering.config, param, value)
            
            # Run HMM clustering
            results = self.hmm_clustering.fit(features)
            
            # Get performance summary
            performance_summary = self.hmm_clustering.get_performance_summary()
            
            # Create comprehensive results
            analysis_results = {
                'hmm_results': results,
                'performance_summary': performance_summary,
                'hyperparameter_optimization': best_params,
                'feature_quality': calculate_data_quality_metrics(features),
                'hardware_utilization': {
                    'gpu_used': self.gpu_manager.is_available() if self.gpu_manager else False,
                    'memory_optimized': self.memory_optimizer is not None,
                    'cpu_optimized': self.cpu_optimizer is not None
                },
                'analysis_timestamp': time.time()
            }
            
            self.logger.info("✅ Comprehensive analysis completed!")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive analysis failed: {e}")
            raise
    
    def save_analysis_results(self, results: Dict[str, Any], filepath: str) -> bool:
        """Save analysis results using serialization utilities."""
        self.logger.info(f"💾 Saving analysis results to {filepath}")
        
        try:
            # Prepare data for serialization
            serializable_results = self._prepare_for_serialization(results)
            
            # Save using appropriate serializer
            if filepath.endswith('.json'):
                success = self.json_serializer.save(serializable_results, filepath)
            else:
                success = self.pickle_serializer.save(serializable_results, filepath)
            
            if success:
                self.logger.info("✅ Analysis results saved successfully")
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save analysis results: {e}")
            return False
    
    def _prepare_for_serialization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for serialization by converting non-serializable objects."""
        serializable = {}
        
        for key, value in results.items():
            if key == 'hmm_results' and hasattr(value, '__dict__'):
                # Convert HMMClusteringResults to dict
                serializable[key] = {
                    'labels': value.labels.tolist() if hasattr(value.labels, 'tolist') else value.labels,
                    'probabilities': value.probabilities.tolist() if hasattr(value.probabilities, 'tolist') else value.probabilities,
                    'log_likelihood': value.log_likelihood,
                    'aic': value.aic,
                    'bic': value.bic,
                    'silhouette_score': value.silhouette_score,
                    'calinski_harabasz_score': value.calinski_harabasz_score,
                    'davies_bouldin_score': value.davies_bouldin_score,
                    'training_time': value.training_time,
                    'memory_usage': value.memory_usage,
                    'validation_metrics': value.validation_metrics
                }
            elif isinstance(value, (np.ndarray, np.generic)):
                # Convert numpy arrays to lists
                serializable[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
            elif isinstance(value, dict):
                # Recursively process dictionaries
                serializable[key] = self._prepare_for_serialization(value)
            else:
                serializable[key] = value
        
        return serializable
    
    def load_analysis_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load analysis results using serialization utilities."""
        self.logger.info(f"📂 Loading analysis results from {filepath}")
        
        try:
            if filepath.endswith('.json'):
                results = self.json_serializer.load(filepath)
            else:
                results = self.pickle_serializer.load(filepath)
            
            if results:
                self.logger.info("✅ Analysis results loaded successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load analysis results: {e}")
            return None
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis report."""
        self.logger.info("📊 Generating analysis report...")
        
        try:
            hmm_results = results.get('hmm_results', {})
            performance = results.get('performance_summary', {})
            
            report = f"""
# HMM Clustering Analysis Report

## Model Performance
- **Training Time**: {hmm_results.get('training_time', 0):.2f} seconds
- **Log Likelihood**: {hmm_results.get('log_likelihood', 0):.2f}
- **AIC**: {hmm_results.get('aic', 0):.2f}
- **BIC**: {hmm_results.get('bic', 0):.2f}

## Clustering Quality
- **Silhouette Score**: {hmm_results.get('silhouette_score', 0):.3f}
- **Calinski-Harabasz Score**: {hmm_results.get('calinski_harabasz_score', 0):.2f}
- **Davies-Bouldin Score**: {hmm_results.get('davies_bouldin_score', 0):.3f}

## Hardware Utilization
- **GPU Used**: {results.get('hardware_utilization', {}).get('gpu_used', False)}
- **Memory Optimized**: {results.get('hardware_utilization', {}).get('memory_optimized', False)}
- **CPU Optimized**: {results.get('hardware_utilization', {}).get('cpu_optimized', False)}

## Hyperparameter Optimization
{self._format_hyperparams(results.get('hyperparameter_optimization', {}))}

## Analysis Timestamp
{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results.get('analysis_timestamp', time.time())))}
"""
            
            self.logger.info("✅ Analysis report generated")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {e}")
            return f"Error generating report: {e}"
    
    def _format_hyperparams(self, hyperparams: Dict[str, Any]) -> str:
        """Format hyperparameters for report."""
        if not hyperparams:
            return "- No hyperparameter optimization performed"
        
        lines = []
        for param, value in hyperparams.items():
            lines.append(f"- **{param}**: {value}")
        
        return "\n".join(lines)


def run_complete_analysis_example():
    """Run a complete analysis example."""
    logger.info("🚀 Running complete HMM clustering analysis example...")
    
    try:
        # Create integration instance
        integration = HMMClusteringIntegration()
        
        # Generate sample market data
        logger.info("📊 Generating sample market data...")
        np.random.seed(42)
        n_samples = 2000
        
        # Create realistic market data
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        
        # Generate price data with trends and volatility clusters
        returns = np.random.normal(0, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Add volume data
        volume = np.random.lognormal(10, 0.5, n_samples)
        
        # Create OHLCV data
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': volume
        })
        
        # Run comprehensive analysis
        results = integration.run_comprehensive_analysis(
            market_data, 
            optimize_hyperparams=True
        )
        
        # Generate and print report
        report = integration.generate_report(results)
        print(report)
        
        # Save results
        integration.save_analysis_results(results, 'hmm_analysis_results.json')
        
        logger.info("✅ Complete analysis example finished successfully!")
        
    except Exception as e:
        logger.error(f"❌ Complete analysis example failed: {e}")
        raise


if __name__ == "__main__":
    run_complete_analysis_example()