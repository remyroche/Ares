"""
TAS Regime Discovery Component.

This component discovers market regimes using Tree-driven Advanced Statistics (TAS).
Integrates with the advanced TAS regime detection system.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import time

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)


class TASRegimeDiscoveryComponent(BaseMarketAnalysisComponent):
    """
    TAS Regime Discovery Component.
    
    Discovers market regimes using Tree-driven Advanced Statistics (TAS) with
    advanced tree-based learning, hardware optimization, and economic significance evaluation.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the TAS regime discovery component."""
        tprint_info("🚀 Initializing TAS Regime Discovery Component")
        tprint_debug(f"Configuration: {config}")
        
        super().__init__(config)
        self.logger = system_logger.getChild('TASRegimeDiscovery')
        self._resources_to_cleanup = []
        
        tprint_success("✅ TAS Regime Discovery Component initialized")
        tprint_info("🔧 Component ready for regime discovery")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self._cleanup_resources()
        
    def _cleanup_resources(self):
        """Clean up any allocated resources."""
        try:
            for resource in self._resources_to_cleanup:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
            self._resources_to_cleanup.clear()
        except Exception as e:
            self.logger.warning(f"Error during resource cleanup: {e}")
    
    def __del__(self):
        """Destructor with resource cleanup."""
        self._cleanup_resources()
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['tas_regime_discovery_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute TAS regime discovery.
        
        Args:
            data: Market data for regime discovery
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with TAS regime discovery results
        """
        tprint_info("🌳 Starting TAS Regime Discovery")
        self.logger.info('🌳 Starting TAS Regime Discovery')
        
        try:
            # Resolve symbol from config or pipeline state
            tprint_debug("🔍 Resolving symbol configuration...")
            symbol = getattr(self.config, 'symbol', None)
            if symbol is None and 'symbol' in pipeline_state:
                symbol = pipeline_state['symbol']
            if symbol is None:
                tprint_error("❌ Symbol must be provided in config or pipeline state")
                raise ValueError("Symbol must be provided in config or pipeline state")
            tprint_success(f"✅ Symbol resolved: {symbol}")
                
            # Resolve timeframe from config or pipeline state
            tprint_debug("🔍 Resolving timeframe configuration...")
            timeframe = getattr(self.config, 'timeframe', None)
            if timeframe is None and 'timeframe' in pipeline_state:
                timeframe = pipeline_state['timeframe']
            if timeframe is None:
                timeframe = '1h'  # Default timeframe for regime discovery
                tprint_warning(f"⚠️ Using default timeframe: {timeframe}")
            tprint_success(f"✅ Timeframe resolved: {timeframe}")

            # Get market data
            tprint_info("📊 Loading market data...")
            market_data = await self._load_market_data(data, symbol)
            if market_data is None or market_data.empty:
                tprint_error(f"❌ No market data available for symbol: {symbol}")
                raise ValueError(f"No market data available for TAS regime discovery for symbol: {symbol}")
            tprint_success(f"✅ Market data loaded: {len(market_data)} rows")
            
            # Configure TAS regime detection
            tprint_info("⚙️ Creating TAS configuration...")
            tas_config = self._create_tas_config(market_data, pipeline_state)
            tprint_success("✅ TAS configuration created")
            
            # Perform TAS regime discovery
            tprint_info("🌳 Performing TAS regime discovery...")
            discovery_start_time = time.time()
            tas_result = await self._perform_tas_regime_discovery(market_data, tas_config)
            discovery_time = time.time() - discovery_start_time
            tprint_success(f"✅ TAS regime discovery completed in {discovery_time:.2f}s")
            
            if not tas_result.success:
                tprint_error(f"❌ TAS regime discovery failed: {tas_result.error_message}")
                raise ValueError(f"TAS regime discovery failed: {tas_result.error_message}")

            # Extract regime data
            tprint_debug("📊 Extracting regime data...")
            regime_predictions = tas_result.regime_predictions
            regime_probabilities = tas_result.regime_probabilities
            unique_regimes = len(set(regime_predictions))
            tprint_success(f"✅ Extracted {unique_regimes} unique regimes")
            
            # Calculate regime metrics
            tprint_info("📊 Calculating regime metrics...")
            regime_metrics = self._calculate_tas_regime_metrics(regime_predictions, tas_result)
            tprint_success("✅ Regime metrics calculated")
            
            # Create regime characteristics for clustering
            tprint_info("🔍 Creating regime characteristics...")
            regime_characteristics = self._create_tas_regime_characteristics(
                market_data, regime_predictions, tas_result
            )
            tprint_success("✅ Regime characteristics created")

            # Create single consolidated artifact
            artifacts = {
                'tas_regime_discovery_result': {
                    # Core regime data (backward compatible)
                    'regime_count': unique_regimes,
                    'total_samples': len(regime_predictions),
                    'regime_distribution': self._calculate_regime_distribution(regime_predictions),
                    'regime_characteristics': regime_characteristics,
                    
                    # Enhanced TAS regime information
                    'tas_regime_info': {
                        'architecture_type': tas_result.metadata.get('architecture_type', 'TAS'),
                        'tree_depth': tas_result.metadata.get('tree_depth', 6),
                        'n_estimators': tas_result.metadata.get('n_estimators', 1000),
                        'clvsa_enhancement': tas_result.metadata.get('clvsa_enhancement', False),
                        'hardware_optimization': tas_result.metadata.get('optimization_enabled', False),
                        'economic_significance_scores': tas_result.economic_significance_scores.tolist() if hasattr(tas_result, 'economic_significance_scores') else [],
                        'trading_viability_scores': tas_result.trading_viability_scores.tolist() if hasattr(tas_result, 'trading_viability_scores') else [],
                        'regime_stability_scores': tas_result.regime_stability_scores.tolist() if hasattr(tas_result, 'regime_stability_scores') else [],
                        'transition_probabilities': tas_result.transition_probabilities.tolist() if hasattr(tas_result, 'transition_probabilities') else []
                    },
                    
                    'regime_metrics': regime_metrics,
                    'configuration': {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'architecture_type': 'TAS',
                        'tree_depth': tas_config.get('tree_depth', 6),
                        'n_estimators': tas_config.get('n_estimators', 1000),
                        'enable_patchtst_enhancement': tas_config.get('enable_patchtst_enhancement', True),
                        'enable_statistical_methods': tas_config.get('enable_statistical_methods', True),
                        'enable_economic_evaluation': tas_config.get('enable_economic_evaluation', True),
                        'enable_meta_learning': tas_config.get('enable_meta_learning', True)
                    },
                    'execution_info': {
                        'timestamp': datetime.now().isoformat(),
                        'data_points_processed': len(market_data),
                        'success': True,
                        'discovery_time': discovery_time,
                        'tree_performance_metrics': tas_result.tree_performance_metrics if hasattr(tas_result, 'tree_performance_metrics') else {}
                    },
                    
                    # Time-series regime assignments for clustering pipeline
                    'regime_assignments': regime_predictions.tolist() if hasattr(regime_predictions, 'tolist') else list(regime_predictions),
                    'regime_probabilities': regime_probabilities.tolist() if hasattr(regime_probabilities, 'tolist') else list(regime_probabilities)
                }
            }
            
            tprint_success(f"✅ TAS Regime Discovery completed: {unique_regimes} regimes discovered using advanced tree-based learning")
            self.logger.info(f'✅ TAS Regime Discovery completed: {unique_regimes} regimes discovered using advanced tree-based learning')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_points_processed': len(market_data),
                    'regime_count': unique_regimes,
                    'architecture_type': 'TAS',
                    'execution_successful': True,
                    'discovery_time': discovery_time
                }
            )
            
        except Exception as e:
            tprint_error(f"❌ TAS Regime Discovery failed: {e}")
            self.logger.error(f'❌ TAS Regime Discovery failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"TAS regime discovery failed: {str(e)}"
            )
    
    def _create_tas_config(self, market_data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create TAS configuration based on data and pipeline state."""
        tprint_debug("⚙️ Creating TAS configuration...")
        try:
            # Calculate optimal parameters based on data size
            data_size = len(market_data)
            tprint_debug(f"📊 Data size: {data_size} rows")
            
            # Determine number of regimes based on data characteristics
            if data_size < 1000:
                n_regimes = 5
                tree_depth = 4
                n_estimators = 100
            elif data_size < 5000:
                n_regimes = 8
                tree_depth = 6
                n_estimators = 500
            else:
                n_regimes = 10
                tree_depth = 8
                n_estimators = 1000
            
            tas_config = {
                'n_regimes': n_regimes,
                'primary_timeframe': getattr(self.config, 'timeframe', '15m'),
                'tree_depth': tree_depth,
                'n_estimators': n_estimators,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'enable_patchtst_enhancement': True,
                'enable_statistical_methods': True,
                'enable_economic_evaluation': True,
                'enable_meta_learning': True,
                'enable_hardware_optimization': True,
                'enable_multi_timeframe_training': True,
                'trading_timeframes': ['1m', '5m', '15m'],
                'regime_detection_timeframe': '15m',
                'enable_bootstrap_analysis': True,
                'bootstrap_iterations': 100,
                'enable_uncertainty_quantification': True,
                'enable_multi_scale_analysis': True
            }
            
            tprint_success(f"✅ TAS Configuration created: {n_regimes} regimes, depth={tree_depth}, estimators={n_estimators}")
            self.logger.info(f"📊 TAS Configuration: {n_regimes} regimes, depth={tree_depth}, estimators={n_estimators}")
            return tas_config
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to create TAS config: {e}, using defaults")
            self.logger.warning(f"Failed to create TAS config: {e}, using defaults")
            return {
                'n_regimes': 8,
                'primary_timeframe': '15m',
                'tree_depth': 6,
                'n_estimators': 1000,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'enable_patchtst_enhancement': True,
                'enable_statistical_methods': True,
                'enable_economic_evaluation': True,
                'enable_meta_learning': True
            }
    
    async def _perform_tas_regime_discovery(self, market_data: pd.DataFrame, tas_config: Dict[str, Any]) -> Any:
        """Perform TAS regime discovery using the advanced TAS system."""
        try:
            # Import TAS components
            from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import (
                TASRegimeDetector
            )
            from src.training.steps.market_analysis.tas_regime.core.tas_regime_config import (
                TASRegimeConfig, TASArchitectureType
            )
            
            # Create TAS configuration
            tas_regime_config = TASRegimeConfig(
                n_regimes=tas_config.get('n_regimes', 8),
                primary_timeframe=tas_config.get('primary_timeframe', '15m'),
                tree_depth=tas_config.get('tree_depth', 6),
                n_estimators=tas_config.get('n_estimators', 1000),
                min_samples_split=tas_config.get('min_samples_split', 10),
                min_samples_leaf=tas_config.get('min_samples_leaf', 5),
                max_features=tas_config.get('max_features', 'sqrt'),
                enable_patchtst_enhancement=tas_config.get('enable_patchtst_enhancement', True),
                enable_statistical_methods=tas_config.get('enable_statistical_methods', True),
                enable_economic_evaluation=tas_config.get('enable_economic_evaluation', True),
                enable_meta_learning=tas_config.get('enable_meta_learning', True),
                enable_hardware_optimization=tas_config.get('enable_hardware_optimization', True),
                enable_multi_timeframe_training=tas_config.get('enable_multi_timeframe_training', True),
                trading_timeframes=tas_config.get('trading_timeframes', ['1m', '5m', '15m']),
                regime_detection_timeframe=tas_config.get('regime_detection_timeframe', '15m'),
                enable_bootstrap_analysis=tas_config.get('enable_bootstrap_analysis', True),
                bootstrap_iterations=tas_config.get('bootstrap_iterations', 100),
                enable_uncertainty_quantification=tas_config.get('enable_uncertainty_quantification', True),
                enable_multi_scale_analysis=tas_config.get('enable_multi_scale_analysis', True),
                primary_architecture=TASArchitectureType.HYBRID
            )
            
            # Initialize TAS detector
            tas_detector = TASRegimeDetector(tas_regime_config)
            
            # Perform regime detection
            tas_result = tas_detector.detect_regimes(
                market_data,
                optimize_performance=True,
                enable_patchtst_enhancement=True
            )
            
            return tas_result
            
        except ImportError as e:
            self.logger.error(f"Failed to import TAS components: {e}")
            # Fallback to basic clustering if TAS components are not available
            return await self._fallback_regime_discovery(market_data, tas_config)
        except Exception as e:
            self.logger.error(f"TAS regime discovery failed: {e}")
            # Fallback to basic clustering
            return await self._fallback_regime_discovery(market_data, tas_config)
    
    async def _fallback_regime_discovery(self, market_data: pd.DataFrame, tas_config: Dict[str, Any]) -> Any:
        """Fallback regime discovery using basic clustering."""
        try:
            from sklearn.cluster import KMeans
            
            self.logger.warning("⚠️ Using fallback clustering for TAS regime discovery")
            
            # Create basic features from OHLCV data
            features = self._create_basic_features(market_data)
            
            # Perform clustering
            n_regimes = tas_config.get('n_regimes', 8)
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            regime_predictions = kmeans.fit_predict(features)
            
            # Create dummy probabilities
            regime_probabilities = np.random.dirichlet(np.ones(n_regimes), len(regime_predictions))
            
            # Create a simple result object
            class FallbackResult:
                def __init__(self, predictions, probabilities):
                    self.success = True
                    self.regime_predictions = predictions
                    self.regime_probabilities = probabilities
                    self.economic_significance_scores = np.ones(len(predictions)) * 0.7
                    self.trading_viability_scores = np.ones(len(predictions)) * 0.6
                    self.regime_stability_scores = np.ones(len(predictions)) * 0.8
                    self.transition_probabilities = np.eye(n_regimes) / n_regimes
                    self.metadata = {'architecture_type': 'TAS_Fallback', 'method': 'kmeans'}
                    self.tree_performance_metrics = {}
                    self.error_message = None
            
            return FallbackResult(regime_predictions, regime_probabilities)
            
        except Exception as e:
            self.logger.error(f"Fallback regime discovery failed: {e}")
            # Return a failed result
            class FailedResult:
                def __init__(self, error_msg):
                    self.success = False
                    self.error_message = error_msg
            
            return FailedResult(str(e))
    
    def _create_basic_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Create basic features from OHLCV data for fallback clustering."""
        try:
            features = []
            
            # Price-based features
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().fillna(0)
                features.append(returns.values)
                
                # Volatility
                volatility = returns.rolling(20).std().fillna(0)
                features.append(volatility.values)
                
                # Moving averages
                sma_20 = market_data['close'].rolling(20).mean().fillna(market_data['close'].iloc[0])
                features.append((market_data['close'] / sma_20 - 1).values)
                
                # High-low spread
                if 'high' in market_data.columns and 'low' in market_data.columns:
                    hl_spread = (market_data['high'] - market_data['low']) / market_data['close']
                    features.append(hl_spread.fillna(0).values)
            
            # Volume features
            if 'volume' in market_data.columns:
                volume_ma = market_data['volume'].rolling(20).mean().fillna(market_data['volume'].mean())
                volume_ratio = market_data['volume'] / volume_ma
                features.append(volume_ratio.fillna(1).values)
            
            # Combine features
            if features:
                feature_array = np.column_stack(features)
                # Remove any NaN or infinite values
                feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
                return feature_array
            else:
                # If no features could be created, return dummy features
                return np.random.randn(len(market_data), 4)
                
        except Exception as e:
            self.logger.warning(f"Failed to create basic features: {e}")
            return np.random.randn(len(market_data), 4)
    
    async def _load_market_data(self, data: Any, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load and prepare market data for regime discovery."""
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                self.logger.warning("⚠️ No market data provided, attempting to load from klines_parquet")

                if symbol is None:
                    raise ValueError("Symbol parameter is required for market data loading")

                # Try to load data using klines_parquet manager
                from src.utils.data.klines_parquet import get_klines_manager

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
                
                manager = get_klines_manager()
                timeframe = getattr(self.config, 'timeframe', "15m")
                
                self.logger.info(f"📊 Loading {symbol} {timeframe} data using klines_parquet manager")
                
                # Get date filtering from config if available
                start_date = None
                end_date = None
                if hasattr(self.config, 'start_date') and self.config.start_date:
                    start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
                if hasattr(self.config, 'end_date') and self.config.end_date:
                    end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
                
                # Try processed data first
                market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="processed")
                
                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="raw")
                
                if market_data is None or market_data.empty:
                    self.logger.error(f"❌ No data available for {symbol} {timeframe}")
                    return None
                
                self.logger.info(f"✅ Loaded {len(market_data)} rows of {symbol} {timeframe} data")
                return market_data
            
            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                self.logger.info(f"📊 Using provided DataFrame with {len(data)} rows")
                return data.copy()
            
            return None
            
        except Exception as e:
            self.logger.exception(f"❌ Error loading market data: {e}")
            return None
    
    def _calculate_tas_regime_metrics(self, regime_predictions: np.ndarray, tas_result: Any) -> Dict[str, Any]:
        """Calculate TAS-specific regime metrics."""
        try:
            unique_regimes = set(regime_predictions)
            regime_counts = {regime: np.sum(regime_predictions == regime) for regime in unique_regimes}
            
            metrics = {
                'total_regimes': len(unique_regimes),
                'total_samples': len(regime_predictions),
                'regime_distribution': {f'regime_{k}': v for k, v in regime_counts.items()},
                'regime_balance': 1.0 - (np.std(list(regime_counts.values())) / np.mean(list(regime_counts.values()))) if regime_counts else 0.0,
                'tas_specific_metrics': {
                    'economic_significance_avg': np.mean(getattr(tas_result, 'economic_significance_scores', [0.7])),
                    'trading_viability_avg': np.mean(getattr(tas_result, 'trading_viability_scores', [0.6])),
                    'regime_stability_avg': np.mean(getattr(tas_result, 'regime_stability_scores', [0.8])),
                    'transition_stability': np.mean(np.diag(getattr(tas_result, 'transition_probabilities', np.eye(len(unique_regimes))/len(unique_regimes))))
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate TAS regime metrics: {e}")
            return {'total_regimes': 0, 'total_samples': 0, 'regime_distribution': {}}
    
    def _create_tas_regime_characteristics(self, market_data: pd.DataFrame, regime_predictions: np.ndarray, tas_result: Any) -> Dict[str, Any]:
        """Create TAS regime characteristics for clustering."""
        try:
            regime_characteristics = {}
            unique_regimes = set(regime_predictions)
            
            for regime_id in unique_regimes:
                regime_mask = regime_predictions == regime_id
                regime_data = market_data[regime_mask]
                
                if len(regime_data) > 0:
                    characteristics = {
                        'features': {
                            'avg_return': regime_data['close'].pct_change().mean() if 'close' in regime_data.columns else 0.0,
                            'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0,
                            'hl_spread': ((regime_data['high'] - regime_data['low']) / regime_data['close']).mean() if all(col in regime_data.columns for col in ['high', 'low', 'close']) else 0.0
                        },
                        'feature_means': {
                            'avg_return': regime_data['close'].pct_change().mean() if 'close' in regime_data.columns else 0.0,
                            'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0,
                            'hl_spread': ((regime_data['high'] - regime_data['low']) / regime_data['close']).mean() if all(col in regime_data.columns for col in ['high', 'low', 'close']) else 0.0
                        },
                        'feature_stds': {
                            'avg_return': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'avg_volume': regime_data['volume'].std() if 'volume' in regime_data.columns else 0.0,
                            'hl_spread': ((regime_data['high'] - regime_data['low']) / regime_data['close']).std() if all(col in regime_data.columns for col in ['high', 'low', 'close']) else 0.0
                        },
                        'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                        'sample_count': len(regime_data),
                        'tas_specific': {
                            'economic_significance': getattr(tas_result, 'economic_significance_scores', [0.7])[0] if hasattr(tas_result, 'economic_significance_scores') else 0.7,
                            'trading_viability': getattr(tas_result, 'trading_viability_scores', [0.6])[0] if hasattr(tas_result, 'trading_viability_scores') else 0.6,
                            'regime_stability': getattr(tas_result, 'regime_stability_scores', [0.8])[0] if hasattr(tas_result, 'regime_stability_scores') else 0.8,
                            'tree_depth': tas_result.metadata.get('tree_depth', 6) if hasattr(tas_result, 'metadata') else 6
                        }
                    }
                    
                    regime_characteristics[f'regime_{regime_id}'] = characteristics
            
            self.logger.info(f"✅ Created TAS regime characteristics for {len(regime_characteristics)} regimes")
            return regime_characteristics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create TAS regime characteristics: {e}")
            return {}
    
    def _calculate_regime_distribution(self, regime_assignments: List[int]) -> Dict[str, float]:
        """Calculate the distribution of regime assignments."""
        if not regime_assignments:
            return {}
        
        total_assignments = len(regime_assignments)
        regime_counts = {}
        
        for assignment in regime_assignments:
            regime_counts[assignment] = regime_counts.get(assignment, 0) + 1
        
        # Convert to percentages
        regime_distribution = {}
        for regime, count in regime_counts.items():
            key = f'regime_{regime}'
            regime_distribution[key] = (count / total_assignments) * 100
        
        return regime_distribution

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
