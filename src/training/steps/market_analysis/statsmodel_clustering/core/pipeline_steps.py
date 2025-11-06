"""
Pipeline Steps for Statsmodel Clustering

This module implements three main pipeline steps for statsmodel clustering:
1. DataDownloadStep - Downloads and validates market data
2. FeatureGenerationStep - Generates features for clustering
3. ClusteringStep - Performs clustering using MarkovRegression

Each step inherits from BaseStep and implements the required abstract methods.
"""

import logging
import time
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Import BaseStep
try:
    from src.training.base_step import BaseStep
except ImportError:
    # Fallback BaseStep class
    class BaseStep(ABC):
        def __init__(self, config: Dict[str, Any]):
            self.config = config
        
        async def execute(self, data: Any) -> Any:
            pass
        
        def validate_config(self) -> None:
            pass
        
        def get_status(self) -> Dict[str, Any]:
            pass

# Import existing components
try:
    from .base_data_downloader import BaseDataDownloader, create_data_downloader
    BASE_DATA_DOWNLOADER_AVAILABLE = True
except ImportError:
    BASE_DATA_DOWNLOADER_AVAILABLE = False
    BaseDataDownloader = None
    create_data_downloader = None

try:
    from ..feature_engineering.enhanced_features import EnhancedFeatureEngineer, FeatureConfig
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    EnhancedFeatureEngineer = None
    FeatureConfig = None

try:
    from .markov_regression_adapter import MarkovRegressionAdapter, MarkovRegressionConfig
    MARKOV_REGRESSION_AVAILABLE = True
except ImportError:
    MARKOV_REGRESSION_AVAILABLE = False
    MarkovRegressionAdapter = None
    MarkovRegressionConfig = None

# Import utilities
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')


class DataDownloadStep(BaseStep):
    """
    Step for downloading market data using BaseDataDownloader.
    
    This step handles the configuration and execution of data downloading
    for clustering analysis, including validation and error handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataDownloadStep.
        
        Args:
            config: Configuration dictionary with download parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration
        self.symbol = config.get('symbol', 'ETHUSDT')
        self.exchange = config.get('exchange', 'BINANCE')
        self.timeframe = config.get('timeframe', '1h')
        self.data_dir = config.get('data_dir', 'data_cache')
        self.lookback_years = config.get('lookback_years', 2)
        self.force_download = config.get('force_download', False)
        
        # Initialize downloader
        self.downloader = None
        self._initialize_downloader()
        
        # Statistics
        self.download_stats = {
            'start_time': None,
            'end_time': None,
            'success': False,
            'data_points': 0,
            'error_message': None
        }
    
    def _initialize_downloader(self):
        """Initialize the data downloader."""
        if not BASE_DATA_DOWNLOADER_AVAILABLE:
            tprint_error("❌ BaseDataDownloader not available")
            self.logger.error("BaseDataDownloader not available")
            return
        
        try:
            tprint_info("🔧 Initializing data downloader")
            downloader_config = {
                'symbol': self.symbol,
                'exchange': self.exchange,
                'timeframe': self.timeframe,
                'data_dir': self.data_dir,
                'lookback_years': self.lookback_years,
                'force_download': self.force_download,
                'downloader_type': 'standard'
            }
            self.downloader = create_data_downloader(downloader_config)
            tprint_success("✅ Data downloader initialized successfully")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize downloader: {e}")
            self.logger.error(f"Failed to initialize downloader: {e}")
    
    async def execute(self, data: Any) -> Dict[str, Any]:
        """
        Execute the data download step.
        
        Args:
            data: Input data (not used for download step)
            
        Returns:
            Dictionary with download results and metadata
        """
        self.download_stats['start_time'] = time.time()
        
        try:
            tprint_info(f"📥 Starting data download for {self.symbol} on {self.exchange} ({self.timeframe})")
            
            if self.downloader is None:
                raise ValueError("Data downloader not initialized")
            
            # Download data
            success, downloaded_data, error = await self.downloader.download_data()
            
            if not success or downloaded_data is None:
                self.download_stats.update({
                    'end_time': time.time(),
                    'success': False,
                    'error_message': error
                })
                tprint_error(f"❌ Data download failed: {error}")
                return {
                    'success': False,
                    'data': None,
                    'error': error,
                    'stats': self.download_stats
                }
            
            # Update statistics
            self.download_stats.update({
                'end_time': time.time(),
                'success': True,
                'data_points': len(downloaded_data)
            })
            
            tprint_success(f"✅ Data download completed: {len(downloaded_data)} records")
            
            return {
                'success': True,
                'data': downloaded_data,
                'error': None,
                'stats': self.download_stats,
                'metadata': {
                    'symbol': self.symbol,
                    'exchange': self.exchange,
                    'timeframe': self.timeframe,
                    'data_shape': downloaded_data.shape
                }
            }
            
        except Exception as e:
            error_msg = f"Data download execution failed: {str(e)}"
            self.download_stats.update({
                'end_time': time.time(),
                'success': False,
                'error_message': error_msg
            })
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'data': None,
                'error': error_msg,
                'stats': self.download_stats
            }
    
    def validate_config(self) -> None:
        """
        Validate the configuration for the data download step.
        
        Raises:
            ValueError: If configuration is invalid
        """
        tprint_info("🔍 Validating data download configuration")
        
        # Check required parameters
        required_params = ['symbol', 'exchange', 'timeframe']
        for param in required_params:
            if param not in self.config:
                tprint_error(f"❌ Missing required parameter: {param}")
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate symbol format
        symbol = self.config['symbol']
        if not symbol or not isinstance(symbol, str) or not symbol.isupper():
            tprint_error(f"❌ Invalid symbol format: {symbol}")
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        # Validate exchange
        valid_exchanges = ['BINANCE', 'BYBIT', 'OKX', 'KRAKEN']
        exchange = self.config['exchange'].upper()
        if exchange not in valid_exchanges:
            tprint_error(f"❌ Invalid exchange: {exchange}. Valid: {valid_exchanges}")
            raise ValueError(f"Invalid exchange: {exchange}. Valid: {valid_exchanges}")
        
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        timeframe = self.config['timeframe']
        if timeframe not in valid_timeframes:
            tprint_error(f"❌ Invalid timeframe: {timeframe}. Valid: {valid_timeframes}")
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid: {valid_timeframes}")
        
        # Validate numeric parameters
        if 'lookback_years' in self.config:
            lookback = self.config['lookback_years']
            if not isinstance(lookback, (int, float)) or lookback <= 0:
                tprint_error(f"❌ Invalid lookback_years: {lookback}")
                raise ValueError(f"Invalid lookback_years: {lookback}")
        
        tprint_success("✅ Data download configuration validation passed")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the data download step.
        
        Returns:
            Dictionary with status information
        """
        tprint_info("📊 Retrieving data download step status")
        
        status = {
            'step_name': 'data_download',
            'step_class': self.__class__.__name__,
            'config': {
                'symbol': self.symbol,
                'exchange': self.exchange,
                'timeframe': self.timeframe,
                'data_dir': self.data_dir,
                'lookback_years': self.lookback_years,
                'force_download': self.force_download
            },
            'downloader_available': self.downloader is not None,
            'stats': self.download_stats,
            'base_data_downloader_available': BASE_DATA_DOWNLOADER_AVAILABLE
        }
        
        return status


class FeatureGenerationStep(BaseStep):
    """
    Step for generating features using EnhancedFeatureEngineer.
    
    This step handles the configuration and execution of feature generation
    for clustering analysis, including PCA and normalization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureGenerationStep.
        
        Args:
            config: Configuration dictionary with feature generation parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration
        self.n_regimes = config.get('n_regimes', 2)
        self.pca_components = config.get('pca_components', 12)
        self.enable_pca = config.get('enable_pca', True)
        self.enable_scaling = config.get('enable_scaling', True)
        self.shift_periods = config.get('shift_periods', 1)
        self.enable_anti_leakage = config.get('enable_anti_leakage', True)
        
        # Feature configuration
        self.include_raw_returns = config.get('include_raw_returns', True)
        self.include_log_returns = config.get('include_log_returns', True)
        self.include_realized_vol = config.get('include_realized_vol', True)
        self.include_rolling_features = config.get('include_rolling_features', True)
        self.include_factor_exposures = config.get('include_factor_exposures', True)
        self.rolling_windows = config.get('rolling_windows', [5, 10, 20])
        
        # Initialize feature engineer
        self.feature_engineer = None
        self._initialize_feature_engineer()
        
        # Statistics
        self.feature_stats = {
            'start_time': None,
            'end_time': None,
            'success': False,
            'input_shape': None,
            'output_shape': None,
            'n_features': 0,
            'error_message': None
        }
    
    def _initialize_feature_engineer(self):
        """Initialize the feature engineer."""
        if not ENHANCED_FEATURES_AVAILABLE:
            tprint_error("❌ EnhancedFeatureEngineer not available")
            self.logger.error("EnhancedFeatureEngineer not available")
            return
        
        try:
            tprint_info("🔧 Initializing feature engineer")
            
            # Create feature configuration
            feature_config = FeatureConfig(
                include_raw_returns=self.include_raw_returns,
                include_log_returns=self.include_log_returns,
                include_realized_vol=self.include_realized_vol,
                include_rolling_features=self.include_rolling_features,
                include_factor_exposures=self.include_factor_exposures,
                rolling_windows=self.rolling_windows,
                shift_periods=self.shift_periods,
                enable_anti_leakage=self.enable_anti_leakage,
                enable_rank_normalization=True
            )
            
            self.feature_engineer = EnhancedFeatureEngineer(feature_config)
            tprint_success("✅ Feature engineer initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize feature engineer: {e}")
            self.logger.error(f"Failed to initialize feature engineer: {e}")
    
    async def execute(self, data: Any) -> Dict[str, Any]:
        """
        Execute the feature generation step.
        
        Args:
            data: Input data (DataFrame with OHLCV data)
            
        Returns:
            Dictionary with feature generation results and metadata
        """
        self.feature_stats['start_time'] = time.time()
        
        try:
            tprint_info("🔧 Starting feature generation")
            
            if self.feature_engineer is None:
                raise ValueError("Feature engineer not initialized")
            
            if data is None or not isinstance(data, pd.DataFrame):
                raise ValueError("Invalid input data: expected pandas DataFrame")
            
            # Store input shape
            self.feature_stats['input_shape'] = data.shape
            
            # Extract features
            features = self.feature_engineer.extract_features(
                price_data=data,
                volume_data=data if 'volume' in data.columns else None
            )
            
            if features is None or features.empty:
                raise ValueError("Feature generation failed: no features generated")
            
            # Apply PCA if enabled
            processed_features = features
            if self.enable_pca and features.shape[1] > self.pca_components:
                tprint_info(f"🔄 Applying PCA: {features.shape[1]} -> {self.pca_components} components")
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                # Scale features
                if self.enable_scaling:
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(features)
                else:
                    scaled_features = features.values
                
                # Apply PCA
                pca = PCA(n_components=self.pca_components, random_state=42)
                processed_features = pd.DataFrame(
                    pca.fit_transform(scaled_features),
                    index=features.index,
                    columns=[f'pca_{i+1}' for i in range(self.pca_components)]
                )
                
                tprint_success(f"✅ PCA applied: explained variance ratio = {pca.explained_variance_ratio_.sum():.3f}")
            
            # Update statistics
            self.feature_stats.update({
                'end_time': time.time(),
                'success': True,
                'output_shape': processed_features.shape,
                'n_features': processed_features.shape[1]
            })
            
            tprint_success(f"✅ Feature generation completed: {processed_features.shape[1]} features")
            
            return {
                'success': True,
                'data': processed_features,
                'error': None,
                'stats': self.feature_stats,
                'metadata': {
                    'n_regimes': self.n_regimes,
                    'pca_enabled': self.enable_pca,
                    'pca_components': self.pca_components,
                    'feature_names': list(processed_features.columns)
                }
            }
            
        except Exception as e:
            error_msg = f"Feature generation execution failed: {str(e)}"
            self.feature_stats.update({
                'end_time': time.time(),
                'success': False,
                'error_message': error_msg
            })
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'data': None,
                'error': error_msg,
                'stats': self.feature_stats
            }
    
    def validate_config(self) -> None:
        """
        Validate the configuration for the feature generation step.
        
        Raises:
            ValueError: If configuration is invalid
        """
        tprint_info("🔍 Validating feature generation configuration")
        
        # Check required parameters
        if 'n_regimes' not in self.config:
            tprint_error("❌ Missing required parameter: n_regimes")
            raise ValueError("Missing required parameter: n_regimes")
        
        # Validate n_regimes
        n_regimes = self.config['n_regimes']
        if not isinstance(n_regimes, int) or n_regimes < 2:
            tprint_error(f"❌ Invalid n_regimes: {n_regimes}. Must be integer >= 2")
            raise ValueError(f"Invalid n_regimes: {n_regimes}. Must be integer >= 2")
        
        # Validate pca_components
        if 'pca_components' in self.config:
            pca_components = self.config['pca_components']
            if not isinstance(pca_components, int) or pca_components < 1:
                tprint_error(f"❌ Invalid pca_components: {pca_components}. Must be integer >= 1")
                raise ValueError(f"Invalid pca_components: {pca_components}. Must be integer >= 1")
        
        # Validate rolling_windows
        if 'rolling_windows' in self.config:
            rolling_windows = self.config['rolling_windows']
            if not isinstance(rolling_windows, list) or not all(isinstance(w, int) and w > 0 for w in rolling_windows):
                tprint_error(f"❌ Invalid rolling_windows: {rolling_windows}. Must be list of positive integers")
                raise ValueError(f"Invalid rolling_windows: {rolling_windows}. Must be list of positive integers")
        
        # Validate shift_periods
        if 'shift_periods' in self.config:
            shift_periods = self.config['shift_periods']
            if not isinstance(shift_periods, int) or shift_periods < 0:
                tprint_error(f"❌ Invalid shift_periods: {shift_periods}. Must be integer >= 0")
                raise ValueError(f"Invalid shift_periods: {shift_periods}. Must be integer >= 0")
        
        tprint_success("✅ Feature generation configuration validation passed")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the feature generation step.
        
        Returns:
            Dictionary with status information
        """
        tprint_info("📊 Retrieving feature generation step status")
        
        status = {
            'step_name': 'feature_generation',
            'step_class': self.__class__.__name__,
            'config': {
                'n_regimes': self.n_regimes,
                'pca_components': self.pca_components,
                'enable_pca': self.enable_pca,
                'enable_scaling': self.enable_scaling,
                'shift_periods': self.shift_periods,
                'enable_anti_leakage': self.enable_anti_leakage,
                'include_raw_returns': self.include_raw_returns,
                'include_log_returns': self.include_log_returns,
                'include_realized_vol': self.include_realized_vol,
                'include_rolling_features': self.include_rolling_features,
                'include_factor_exposures': self.include_factor_exposures,
                'rolling_windows': self.rolling_windows
            },
            'feature_engineer_available': self.feature_engineer is not None,
            'stats': self.feature_stats,
            'enhanced_features_available': ENHANCED_FEATURES_AVAILABLE
        }
        
        return status


class ClusteringStep(BaseStep):
    """
    Step for performing clustering using MarkovRegressionAdapter with HPO.

    This step handles the configuration and execution of clustering
    using Markov Regression for regime detection with hyperparameter optimization.

    Key Features:
    - Trials for 4-7 regimes with automatic selection
    - Hierarchical HPO (coarse -> fine -> TPE)
    - Comprehensive optimization goals assessment
    - VectorBT/Numba/JIT optimized computations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ClusteringStep.

        Args:
            config: Configuration dictionary with clustering parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # HPO configuration
        self.enable_hpo = config.get('enable_hpo', False)
        self.hpo_regime_range = config.get('hpo_regime_range', (4, 7))  # Test 4-7 regimes
        self.hpo_n_trials_coarse = config.get('hpo_n_trials_coarse', 30)
        self.hpo_n_trials_fine = config.get('hpo_n_trials_fine', 20)
        self.hpo_n_trials_tpe = config.get('hpo_n_trials_tpe', 50)

        # Extract configuration
        self.k_regimes = config.get('k_regimes', 4)
        self.trend = config.get('trend', 'c')
        self.order = config.get('order', 0)
        self.switching_variance = config.get('switching_variance', True)
        self.switching_trend = config.get('switching_trend', True)
        self.maxiter = config.get('maxiter', 100)
        self.tolerance = config.get('tolerance', 1e-6)
        self.method = config.get('method', 'bfgs')
        self.random_state = config.get('random_state', 42)

        # Data preprocessing
        self.enable_pca = config.get('enable_pca', True)
        self.pca_components = config.get('pca_components', 12)
        self.enable_scaling = config.get('enable_scaling', True)

        # Advanced options
        self.enable_diagnostics = config.get('enable_diagnostics', True)
        self.enable_hardware_optimization = config.get('enable_hardware_optimization', True)

        # Initialize clustering adapter
        self.clustering_adapter = None
        self._initialize_clustering_adapter()

        # Statistics
        self.clustering_stats = {
            'start_time': None,
            'end_time': None,
            'success': False,
            'input_shape': None,
            'n_regimes': self.k_regimes,
            'log_likelihood': 0.0,
            'aic': 0.0,
            'bic': 0.0,
            'converged': False,
            'error_message': None,
            'hpo_enabled': self.enable_hpo,
            'hpo_results': None
        }
    
    def _initialize_clustering_adapter(self):
        """Initialize the clustering adapter."""
        if not MARKOV_REGRESSION_AVAILABLE:
            tprint_error("❌ MarkovRegressionAdapter not available")
            self.logger.error("MarkovRegressionAdapter not available")
            return
        
        try:
            tprint_info("🔧 Initializing clustering adapter")
            
            # Create clustering configuration
            clustering_config = MarkovRegressionConfig(
                k_regimes=self.k_regimes,
                trend=self.trend,
                order=self.order,
                switching_variance=self.switching_variance,
                switching_trend=self.switching_trend,
                maxiter=self.maxiter,
                tolerance=self.tolerance,
                method=self.method,
                random_state=self.random_state,
                enable_pca=self.enable_pca,
                pca_components=self.pca_components,
                enable_scaling=self.enable_scaling,
                enable_diagnostics=self.enable_diagnostics,
                enable_hardware_optimization=self.enable_hardware_optimization
            )
            
            self.clustering_adapter = MarkovRegressionAdapter(clustering_config)
            tprint_success("✅ Clustering adapter initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize clustering adapter: {e}")
            self.logger.error(f"Failed to initialize clustering adapter: {e}")
    
    async def execute(self, data: Any) -> Dict[str, Any]:
        """
        Execute the clustering step with optional HPO.

        Args:
            data: Input data (DataFrame with features)

        Returns:
            Dictionary with clustering results and metadata
        """
        self.clustering_stats['start_time'] = time.time()

        try:
            if data is None or not isinstance(data, pd.DataFrame):
                raise ValueError("Invalid input data: expected pandas DataFrame")

            # Store input shape
            self.clustering_stats['input_shape'] = data.shape

            # Run with HPO if enabled
            if self.enable_hpo:
                tprint_info(f"🔍 Starting clustering with HPO (testing {self.hpo_regime_range[0]}-{self.hpo_regime_range[1]} regimes)")
                result = await self._execute_with_hpo(data)
            else:
                tprint_info(f"🔄 Starting clustering with {self.k_regimes} regimes")

                if self.clustering_adapter is None:
                    raise ValueError("Clustering adapter not initialized")

                # Fit clustering model
                result = self.clustering_adapter.fit(data)

                if not result.success:
                    raise ValueError(f"Clustering failed: {result.error_message}")

            # Update statistics
            self.clustering_stats.update({
                'end_time': time.time(),
                'success': True,
                'log_likelihood': result.log_likelihood,
                'aic': result.aic,
                'bic': result.bic,
                'converged': result.diagnostics.get('model_fit', {}).get('converged', False) if result.diagnostics else False
            })

            tprint_success(f"✅ Clustering completed: {result.cluster_labels.max() + 1} regimes, AIC={result.aic:.2f}")

            # Prepare results
            clustering_results = {
                'labels': result.cluster_labels,
                'probabilities': result.cluster_probabilities,
                'transition_matrix': result.transition_matrix,
                'regime_params': result.regime_params,
                'model_summary': result.model_summary,
                'diagnostics': result.diagnostics
            }

            return {
                'success': True,
                'data': clustering_results,
                'error': None,
                'stats': self.clustering_stats,
                'metadata': {
                    'k_regimes': result.cluster_labels.max() + 1 if hasattr(result, 'cluster_labels') else self.k_regimes,
                    'trend': self.trend,
                    'order': self.order,
                    'switching_variance': self.switching_variance,
                    'switching_trend': self.switching_trend,
                    'processing_time': result.processing_time if hasattr(result, 'processing_time') else 0.0,
                    'optimization_time': result.optimization_time if hasattr(result, 'optimization_time') else 0.0,
                    'feature_names': result.feature_names if hasattr(result, 'feature_names') else []
                }
            }

        except Exception as e:
            error_msg = f"Clustering execution failed: {str(e)}"
            self.clustering_stats.update({
                'end_time': time.time(),
                'success': False,
                'error_message': error_msg
            })
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'data': None,
                'error': error_msg,
                'stats': self.clustering_stats
            }

    async def _execute_with_hpo(self, data: pd.DataFrame) -> Any:
        """
        Execute clustering with hyperparameter optimization.

        Tests different numbers of regimes (4-7) and parameters using
        hierarchical optimization (coarse -> fine -> TPE).

        Args:
            data: Input features DataFrame

        Returns:
            Best clustering result
        """
        try:
            from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
                HierarchicalParameterOptimizer,
                ParameterGroup,
                OptimizationStage,
                create_param_group
            )
            from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
                DEFAULT_CLUSTERING_GOALS,
                calculate_composite_score,
                MetricCalculator
            )
        except ImportError as e:
            tprint_warning(f"⚠️ HPO dependencies not available: {e}, running without HPO")
            self.enable_hpo = False
            return self.clustering_adapter.fit(data)

        tprint_info("🔧 Setting up hyperparameter optimization")

        # Define parameter search space
        param_groups = [
            create_param_group(
                name="regime_structure",
                params={
                    "k_regimes": {
                        "type": "int",
                        "low": self.hpo_regime_range[0],
                        "high": self.hpo_regime_range[1]
                    },
                    "trend": {
                        "type": "categorical",
                        "choices": ["c", "t", "ct"]
                    },
                    "order": {
                        "type": "int",
                        "low": 0,
                        "high": 2
                    }
                },
                priority=1,
                description="Core regime structure parameters"
            ),
            create_param_group(
                name="switching_params",
                params={
                    "switching_variance": {
                        "type": "categorical",
                        "choices": [True, False]
                    },
                    "switching_trend": {
                        "type": "categorical",
                        "choices": [True, False]
                    }
                },
                priority=2,
                depends_on=["regime_structure"],
                description="Switching behavior parameters"
            )
        ]

        # Define objective function
        metric_calculator = MetricCalculator()

        def objective_function(params, X_train, y_train=None, X_val=None, y_val=None,
                              model=None, cv_folds=5, scoring_metric='composite'):
            """Objective function for HPO using clustering optimization goals."""
            try:
                # Create and fit clustering model with these parameters
                from .markov_regression_adapter import MarkovRegressionAdapter, MarkovRegressionConfig

                config = MarkovRegressionConfig(
                    k_regimes=params['k_regimes'],
                    trend=params['trend'],
                    order=params['order'],
                    switching_variance=params.get('switching_variance', True),
                    switching_trend=params.get('switching_trend', True),
                    maxiter=self.maxiter,
                    tolerance=self.tolerance,
                    method=self.method,
                    random_state=self.random_state,
                    enable_pca=self.enable_pca,
                    pca_components=self.pca_components,
                    enable_scaling=self.enable_scaling
                )

                adapter = MarkovRegressionAdapter(config)
                result = adapter.fit(X_train)

                if not result.success:
                    return -np.inf

                # Calculate comprehensive score
                # For now, use AIC/BIC as proxy (lower is better, so negate)
                # In full implementation, would use clustering_optimization_goals metrics
                score = -result.aic / 1000.0  # Normalize

                return score

            except Exception as e:
                tprint_warning(f"⚠️ Evaluation failed for params {params}: {e}")
                return -np.inf

        # Create optimizer
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=objective_function,
            stages=[
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,
                OptimizationStage.TPE
            ],
            cv_folds=1,  # No CV for time series clustering
            scoring_metric='composite',
            direction='maximize',
            enable_final_refinement=True,
            final_refinement_trials=20,
            random_state=self.random_state,
            verbose=True
        )

        tprint_info("🚀 Running hierarchical parameter optimization")

        # Run optimization
        result = optimizer.optimize(
            X_train=data,
            y_train=np.zeros(len(data)),  # Dummy for API compatibility
            X_val=None,
            y_val=None,
            model=None
        )

        self.clustering_stats['hpo_results'] = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'total_trials': result.total_trials,
            'total_time': result.total_time
        }

        tprint_success(f"✅ HPO completed: best score={result.best_score:.6f}, trials={result.total_trials}")
        tprint_info(f"📊 Best parameters: {result.best_params}")

        # Fit final model with best parameters
        from .markov_regression_adapter import MarkovRegressionAdapter, MarkovRegressionConfig

        final_config = MarkovRegressionConfig(
            k_regimes=result.best_params['k_regimes'],
            trend=result.best_params['trend'],
            order=result.best_params['order'],
            switching_variance=result.best_params.get('switching_variance', True),
            switching_trend=result.best_params.get('switching_trend', True),
            maxiter=self.maxiter,
            tolerance=self.tolerance,
            method=self.method,
            random_state=self.random_state,
            enable_pca=self.enable_pca,
            pca_components=self.pca_components,
            enable_scaling=self.enable_scaling
        )

        final_adapter = MarkovRegressionAdapter(final_config)
        final_result = final_adapter.fit(data)

        return final_result
    
    def validate_config(self) -> None:
        """
        Validate the configuration for the clustering step.
        
        Raises:
            ValueError: If configuration is invalid
        """
        tprint_info("🔍 Validating clustering configuration")
        
        # Check required parameters
        if 'k_regimes' not in self.config:
            tprint_error("❌ Missing required parameter: k_regimes")
            raise ValueError("Missing required parameter: k_regimes")
        
        # Validate k_regimes
        k_regimes = self.config['k_regimes']
        if not isinstance(k_regimes, int) or k_regimes < 2:
            tprint_error(f"❌ Invalid k_regimes: {k_regimes}. Must be integer >= 2")
            raise ValueError(f"Invalid k_regimes: {k_regimes}. Must be integer >= 2")
        
        # Validate trend
        if 'trend' in self.config:
            trend = self.config['trend']
            valid_trends = ['c', 't', 'ct']
            if trend not in valid_trends:
                tprint_error(f"❌ Invalid trend: {trend}. Valid: {valid_trends}")
                raise ValueError(f"Invalid trend: {trend}. Valid: {valid_trends}")
        
        # Validate order
        if 'order' in self.config:
            order = self.config['order']
            if not isinstance(order, int) or order < 0:
                tprint_error(f"❌ Invalid order: {order}. Must be integer >= 0")
                raise ValueError(f"Invalid order: {order}. Must be integer >= 0")
        
        # Validate maxiter
        if 'maxiter' in self.config:
            maxiter = self.config['maxiter']
            if not isinstance(maxiter, int) or maxiter <= 0:
                tprint_error(f"❌ Invalid maxiter: {maxiter}. Must be integer > 0")
                raise ValueError(f"Invalid maxiter: {maxiter}. Must be integer > 0")
        
        # Validate method
        if 'method' in self.config:
            method = self.config['method']
            valid_methods = ['em', 'bfgs']
            if method not in valid_methods:
                tprint_error(f"❌ Invalid method: {method}. Valid: {valid_methods}")
                raise ValueError(f"Invalid method: {method}. Valid: {valid_methods}")
        
        tprint_success("✅ Clustering configuration validation passed")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the clustering step.
        
        Returns:
            Dictionary with status information
        """
        tprint_info("📊 Retrieving clustering step status")
        
        status = {
            'step_name': 'clustering',
            'step_class': self.__class__.__name__,
            'config': {
                'k_regimes': self.k_regimes,
                'trend': self.trend,
                'order': self.order,
                'switching_variance': self.switching_variance,
                'switching_trend': self.switching_trend,
                'maxiter': self.maxiter,
                'tolerance': self.tolerance,
                'method': self.method,
                'random_state': self.random_state,
                'enable_pca': self.enable_pca,
                'pca_components': self.pca_components,
                'enable_scaling': self.enable_scaling,
                'enable_diagnostics': self.enable_diagnostics,
                'enable_hardware_optimization': self.enable_hardware_optimization
            },
            'clustering_adapter_available': self.clustering_adapter is not None,
            'stats': self.clustering_stats,
            'markov_regression_available': MARKOV_REGRESSION_AVAILABLE
        }
        
        return status


# Factory functions for creating pipeline steps
def create_data_download_step(config: Dict[str, Any]) -> DataDownloadStep:
    """
    Factory function to create a DataDownloadStep.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DataDownloadStep instance
    """
    tprint_info("🏭 Creating DataDownloadStep with factory function")
    step = DataDownloadStep(config)
    tprint_success("✅ DataDownloadStep created successfully")
    return step


def create_feature_generation_step(config: Dict[str, Any]) -> FeatureGenerationStep:
    """
    Factory function to create a FeatureGenerationStep.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured FeatureGenerationStep instance
    """
    tprint_info("🏭 Creating FeatureGenerationStep with factory function")
    step = FeatureGenerationStep(config)
    tprint_success("✅ FeatureGenerationStep created successfully")
    return step


def create_clustering_step(config: Dict[str, Any]) -> ClusteringStep:
    """
    Factory function to create a ClusteringStep.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ClusteringStep instance
    """
    tprint_info("🏭 Creating ClusteringStep with factory function")
    step = ClusteringStep(config)
    tprint_success("✅ ClusteringStep created successfully")
    return step


# Convenience function for creating a complete pipeline
def create_statsmodel_clustering_pipeline(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
    timeframe: str = "1h",
    lookback_years: int = 2,
    n_regimes: int = 2,
    pca_components: int = 12,
    **kwargs
) -> List[BaseStep]:
    """
    Create a complete statsmodel clustering pipeline.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        lookback_years: Years of historical data
        n_regimes: Number of regimes for clustering
        pca_components: Number of PCA components
        **kwargs: Additional configuration parameters
        
    Returns:
        List of configured pipeline steps
    """
    tprint_info("🚀 Creating complete statsmodel clustering pipeline")
    
    # Data download step configuration
    download_config = {
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'lookback_years': lookback_years,
        **kwargs.get('download', {})
    }
    
    # Feature generation step configuration
    feature_config = {
        'n_regimes': n_regimes,
        'pca_components': pca_components,
        **kwargs.get('features', {})
    }
    
    # Clustering step configuration
    clustering_config = {
        'k_regimes': n_regimes,
        'pca_components': pca_components,
        **kwargs.get('clustering', {})
    }
    
    # Create steps
    steps = [
        create_data_download_step(download_config),
        create_feature_generation_step(feature_config),
        create_clustering_step(clustering_config)
    ]
    
    tprint_success(f"✅ Pipeline created with {len(steps)} steps")
    return steps