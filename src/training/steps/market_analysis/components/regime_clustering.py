"""
Regime Clustering Component.

This component bridges HDBSCAN regime discovery and regime model training.
It processes regime labels from HDBSCAN and generates regime-specific features
for downstream model training.

Features:
- Processes HDBSCAN regime discovery results
- Generates regime-specific features
- Creates regime training datasets
- Integrates with ares_launcher and BaseStep architecture
- Compatible with HDBSCAN method and regime training
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import json
import pickle

# Import BaseStep and utilities
from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_timer
)
from src.utils.serialization_utils import save_pickle, load_pickle
from src.utils.data.klines_parquet import get_klines_manager

# Import shared utilities from market analysis
from ..shared_utils import (
    # Features
    prepare_market_features,
    FeatureConfig,
    FeaturePreparationResult,
    
    # Configuration
    validate_regime_count,
    normalize_weights,
    validate_algorithm_type,
    create_default_config,
    ConfigValidator,
    BaseConfig,
    
    # Logging
    get_logger,
    log_execution,
    log_performance,
    LoggingContext,
    
    # Metrics
    calculate_consensus_metrics,
    calculate_disagreement_metrics,
    calculate_economic_scores,
    calculate_trading_scores,
    calculate_stability_scores,
    MetricsCalculator,
    
    # Characteristics
    create_regime_characteristics,
    generate_cluster_characteristics,
    CharacteristicsGenerator,
)

# Import calibration registry utilities
try:
    from ..shared_utils.calibration_registry import (
        get_current_calibration,
        get_quality_thresholds as get_calibrated_thresholds,
        update_quality_calibration,
    )
    CALIBRATION_REGISTRY_AVAILABLE = True
except ImportError:
    CALIBRATION_REGISTRY_AVAILABLE = False
    tprint("⚠️ Calibration registry not available", "WARNING")

# Import matrix operations and hardware utilities
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor,
        safe_matrix_multiply,
        safe_correlation_matrix,
        gpu_matrix_multiply,
        correlation_matrix_gpu,
        optimize_dataframe,
        vectorized_rolling_features,
        matrix_correlation_analysis,
        batch_matrix_multiply,
        batch_feature_transformation,
        batch_correlation_analysis,
        get_hardware_performance_report,
        optimize_matrix_operation_with_hardware,
        cleanup_hardware_resources,
        get_processing_performance_stats
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    MATRIX_OPERATIONS_AVAILABLE = False
    tprint(f"Matrix operations not available: {e}", "WARNING")

try:
    from src.utils.hardware import (
        get_unified_hardware_manager,
        get_advanced_cpu_optimizer,
        get_enhanced_gpu_manager,
        get_advanced_memory_optimizer,
        get_adaptive_optimization_engine,
        optimize_for_workload,
        optimize_for_workload_adaptive,
        optimize_dataframe_advanced,
        record_performance_adaptive
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
    tprint("✅ Hardware optimization utilities imported successfully", "SUCCESS")
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    tprint(f"Hardware optimization not available: {e}", "WARNING")

# Import M1-specific hardware utilities
try:
    from src.utils.hardware.unified_hardware_manager import (
        get_unified_hardware_manager,
        WorkloadType,
        OptimizationLevel,
        HardwareConfig
    )
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATIONS_AVAILABLE = False
    tprint("⚠️ Mac M1 hardware optimizations not available", "WARNING")

logger = logging.getLogger(__name__)


class RegimeClusteringComponent(BaseStep):
    """
    Regime Clustering Component.
    
    Processes HDBSCAN regime discovery results and generates regime-specific features
    for downstream model training. This component bridges the gap between clustering
    and model training in the market analysis pipeline.
    """
    
    def __init__(self, step_name: str = "regime_clustering"):
        """Initialize the regime clustering component."""
        super().__init__(step_name)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Component state
        self.regime_data = None
        self.regime_features = None
        self.regime_labels = None
        self.regime_probabilities = None
        self.economic_profiles = None
        
        tprint("✅ RegimeClusteringComponent initialized", "SUCCESS")
    
    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute regime clustering step.
        
        Args:
            config: Configuration dictionary with parameters:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - data_dir: Data directory path
                - start_date: Start date (optional)
                - end_date: End date (optional)
                - execution_mode: 'full', 'light', or 'blank'
                
        Returns:
            Dictionary with execution results, artifacts, and metrics
        """
        start_time = datetime.now()
        
        try:
            tprint(f"🔍 Starting regime clustering for {config.get('symbol', 'UNKNOWN')}", "INFO")
            
            # Validate required parameters
            self._validate_config(config)
            
            # Load HDBSCAN regime discovery results
            regime_discovery_data = self._load_regime_discovery_data(config)
            if regime_discovery_data is None:
                raise ValueError("Failed to load HDBSCAN regime discovery data")
            
            # Load market data
            market_data = self._load_market_data(config)
            if market_data is None or len(market_data) == 0:
                raise ValueError("Failed to load market data")
            
            tprint(f"✅ Loaded data: {market_data.shape[0]} rows, {market_data.shape[1]} columns", "SUCCESS")
            
            # Process regime clustering
            regime_result = await self._process_regime_clustering(
                regime_discovery_data, market_data, config
            )
            
            if not regime_result['success']:
                raise ValueError(f"Regime clustering failed: {regime_result.get('error_message', 'Unknown error')}")
            
            # Create artifacts
            artifacts = self._create_artifacts(regime_result, config)
            
            # Save artifacts
            self._save_artifacts(artifacts, config)
            
            # Calculate metrics
            metrics = self._calculate_metrics(regime_result, start_time, config)
            
            # Create outcome report
            outcome_report = self._create_outcome_report(regime_result, metrics, config)
            
            tprint(f"✅ Regime clustering completed: {regime_result['n_regimes']} regimes processed", "SUCCESS")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'outcome_report': outcome_report,
                'regime_result': regime_result,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            error_msg = f"Regime clustering failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'artifacts': {},
                'metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        required_params = ['symbol', 'exchange', 'timeframe']
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
    
    def _load_regime_discovery_data(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load HDBSCAN regime discovery results."""
        try:
            tprint("📂 Loading HDBSCAN regime discovery data...", "INFO")
            
            # Look for HDBSCAN regime discovery artifacts
            data_dir = Path(config.get('data_dir', 'historical_data'))
            symbol = config['symbol']
            exchange = config['exchange']
            timeframe = config['timeframe']
            
            # Search for HDBSCAN regime discovery artifacts
            hdbscan_dir = data_dir / 'hdbscan_regime_discovery' / symbol
            if not hdbscan_dir.exists():
                tprint(f"⚠️ HDBSCAN regime discovery directory not found: {hdbscan_dir}", "WARNING")
                return None
            
            # Look for the most recent artifacts file
            artifacts_files = list(hdbscan_dir.glob(f"hdbscan_regime_artifacts_{symbol}_{timeframe}_*.pkl"))
            if not artifacts_files:
                tprint(f"⚠️ No HDBSCAN regime discovery artifacts found in {hdbscan_dir}", "WARNING")
                return None
            
            # Load the most recent artifacts file
            latest_artifacts_file = max(artifacts_files, key=lambda x: x.stat().st_mtime)
            regime_discovery_data = load_pickle(latest_artifacts_file)
            
            tprint(f"✅ Loaded HDBSCAN regime discovery data from {latest_artifacts_file.name}", "SUCCESS")
            return regime_discovery_data
            
        except Exception as e:
            tprint(f"❌ Failed to load HDBSCAN regime discovery data: {e}", "ERROR")
            return None
    
    def _load_market_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load market data using klines manager."""
        try:
            tprint("📂 Loading market data...", "INFO")
            
            # Get klines manager
            klines_manager = get_klines_manager(data_dir=config.get('data_dir', 'historical_data'))
            
            # Parse date filters if provided
            start_date = None
            end_date = None
            
            if 'start_date' in config and config['start_date']:
                start_date = pd.to_datetime(config['start_date'])
                tprint(f"📅 Using start_date filter: {start_date}", "INFO")
            
            if 'end_date' in config and config['end_date']:
                end_date = pd.to_datetime(config['end_date'])
                tprint(f"📅 Using end_date filter: {end_date}", "INFO")
            
            # Load data
            market_data = klines_manager.read_data(
                symbol=config['symbol'],
                interval=config['timeframe'],
                data_type="processed",
                start_date=start_date,
                end_date=end_date
            )
            
            if market_data is not None and len(market_data) > 0:
                tprint(f"✅ Market data loaded: {market_data.shape[0]} rows, {market_data.shape[1]} columns", "SUCCESS")
                tprint(f"📅 Date range: {market_data.index.min()} to {market_data.index.max()}", "INFO")
                return market_data
            else:
                tprint("❌ No market data loaded", "ERROR")
                return None
                
        except Exception as e:
            tprint(f"❌ Failed to load market data: {e}", "ERROR")
            return None
    
    async def _process_regime_clustering(
        self, 
        regime_discovery_data: Dict[str, Any], 
        market_data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process regime clustering from HDBSCAN results."""
        try:
            tprint("🔄 Processing regime clustering...", "INFO")
            
            # Extract regime labels and probabilities
            regime_labels = regime_discovery_data.get('regime_labels', np.array([]))
            regime_probabilities = regime_discovery_data.get('regime_probabilities', np.array([]))
            economic_profiles = regime_discovery_data.get('economic_profiles', [])
            
            if len(regime_labels) == 0:
                raise ValueError("No regime labels found in HDBSCAN regime discovery data")
            
            # Ensure data alignment
            if len(regime_labels) != len(market_data):
                tprint(f"⚠️ Data length mismatch: regime_labels={len(regime_labels)}, market_data={len(market_data)}", "WARNING")
                # Align data by taking the minimum length
                min_length = min(len(regime_labels), len(market_data))
                regime_labels = regime_labels[:min_length]
                regime_probabilities = regime_probabilities[:min_length] if len(regime_probabilities) > 0 else np.array([])
                market_data = market_data.iloc[:min_length]
                tprint(f"✅ Aligned data to {min_length} samples", "SUCCESS")
            
            # Generate regime-specific features
            regime_features = self._generate_regime_features(market_data, regime_labels, config)
            
            # Perform multi-step clustering optimization with sample reallocation
            tprint("🔄 Performing multi-step clustering optimization...", "INFO")
            optimization_result = await self._perform_multi_step_clustering_optimization(
                regime_features['features_df'], regime_labels, market_data, config
            )
            
            if not optimization_result['success']:
                tprint(f"⚠️ Clustering optimization failed: {optimization_result.get('error', 'Unknown error')}", "WARNING")
                # Use original labels as fallback
                optimized_labels = regime_labels
            else:
                optimized_labels = optimization_result['final_assignments']
                tprint(f"✅ Clustering optimization completed: {optimization_result['optimization_metrics']['total_changes']} changes made", "SUCCESS")
            
            # Create regime training datasets with optimized labels
            regime_datasets = self._create_regime_datasets(market_data, optimized_labels, regime_features, config)
            
            # Calculate regime statistics with optimized labels
            regime_stats = self._calculate_regime_statistics(optimized_labels, regime_probabilities, economic_profiles)
            
            # Generate regime characteristics using shared utilities
            regime_characteristics = self._generate_regime_characteristics(
                regime_features, optimized_labels, economic_profiles
            )
            
            # Store component state
            self.regime_data = market_data
            self.regime_features = regime_features
            self.regime_labels = optimized_labels
            self.regime_probabilities = regime_probabilities
            self.economic_profiles = economic_profiles
            
            tprint(f"✅ Regime clustering processed: {regime_stats['n_regimes']} regimes, {regime_stats['n_samples']} samples", "SUCCESS")
            
            return {
                'success': True,
                'regime_labels': optimized_labels,
                'regime_probabilities': regime_probabilities,
                'regime_features': regime_features,
                'regime_datasets': regime_datasets,
                'regime_stats': regime_stats,
                'economic_profiles': economic_profiles,
                'regime_characteristics': regime_characteristics,
                'clustering_optimization': optimization_result,
                'n_regimes': regime_stats['n_regimes'],
                'n_samples': regime_stats['n_samples']
            }
            
        except Exception as e:
            tprint(f"❌ Regime clustering processing failed: {e}", "ERROR")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _generate_regime_features(
        self, 
        market_data: pd.DataFrame, 
        regime_labels: np.ndarray, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate regime-specific features using shared utilities."""
        try:
            tprint("🔧 Generating regime-specific features...", "INFO")
            
            # Use shared feature preparation utilities
            feature_config = FeatureConfig(
                enable_technical_indicators=True,
                enable_volatility_features=True,
                enable_volume_features=True,
                enable_regime_features=True,
                window_sizes=[5, 10, 20, 50],
                technical_indicators=['rsi', 'bollinger_bands', 'atr', 'macd']
            )
            
            # Prepare market features using shared utilities
            feature_result = prepare_market_features(market_data, feature_config)
            
            if not feature_result.success:
                tprint(f"⚠️ Shared feature preparation failed: {feature_result.error_message}", "WARNING")
                # Fallback to basic feature generation
                return self._generate_basic_regime_features(market_data, regime_labels, config)
            
            # Add regime-specific features
            regime_features = self._add_regime_specific_features(
                feature_result.features_df, regime_labels, config
            )
            
            # Apply feature filters using shared utilities
            if MATRIX_OPERATIONS_AVAILABLE:
                try:
                    # Use shared feature filtering utilities
                    from ..shared_utils.feature_filters import (
                        winsorize_frame, filter_low_variance, prune_correlated_features
                    )
                    
                    # Apply winsorization
                    regime_features = winsorize_frame(regime_features, quantiles=(0.01, 0.99))
                    
                    # Filter low variance features
                    regime_features = filter_low_variance(regime_features, threshold=0.01)
                    
                    # Prune correlated features
                    regime_features = prune_correlated_features(regime_features, threshold=0.95)
                    
                except Exception as e:
                    tprint(f"⚠️ Feature filtering failed: {e}", "WARNING")
            
            tprint(f"✅ Generated {len(regime_features.columns)} regime-specific features", "SUCCESS")
            
            return {
                'features_df': regime_features,
                'feature_names': list(regime_features.columns),
                'n_features': len(regime_features.columns),
                'feature_config': feature_config,
                'feature_result': feature_result
            }
            
        except Exception as e:
            tprint(f"❌ Feature generation failed: {e}", "ERROR")
            # Fallback to basic feature generation
            return self._generate_basic_regime_features(market_data, regime_labels, config)
    
    def _generate_basic_regime_features(
        self, 
        market_data: pd.DataFrame, 
        regime_labels: np.ndarray, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback basic feature generation."""
        try:
            features = {}
            
            # Basic market features
            if 'close' in market_data.columns:
                features['returns'] = market_data['close'].pct_change().fillna(0)
                features['log_returns'] = np.log(market_data['close'] / market_data['close'].shift(1)).fillna(0)
                features['volatility'] = features['returns'].rolling(window=20).std().fillna(0)
                features['price_momentum'] = market_data['close'].pct_change(periods=5).fillna(0)
            
            if 'volume' in market_data.columns:
                features['volume'] = market_data['volume']
                features['volume_ma'] = market_data['volume'].rolling(window=20).mean().fillna(0)
                features['volume_ratio'] = market_data['volume'] / features['volume_ma']
                features['volume_ratio'] = features['volume_ratio'].fillna(1)
            
            # Regime-specific features
            features['regime_label'] = regime_labels
            features['regime_persistence'] = self._calculate_regime_persistence(regime_labels)
            features['regime_transitions'] = self._calculate_regime_transitions(regime_labels)
            
            # Technical indicators
            if 'high' in market_data.columns and 'low' in market_data.columns and 'close' in market_data.columns:
                features['rsi'] = self._calculate_rsi(market_data['close'])
                features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(market_data['close'])
                features['atr'] = self._calculate_atr(market_data['high'], market_data['low'], market_data['close'])
            
            # Create features DataFrame
            features_df = pd.DataFrame(features, index=market_data.index)
            
            return {
                'features_df': features_df,
                'feature_names': list(features.keys()),
                'n_features': len(features)
            }
            
        except Exception as e:
            tprint(f"❌ Basic feature generation failed: {e}", "ERROR")
            return {
                'features_df': pd.DataFrame(),
                'feature_names': [],
                'n_features': 0
            }
    
    def _add_regime_specific_features(
        self, 
        features_df: pd.DataFrame, 
        regime_labels: np.ndarray, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add regime-specific features to the base features."""
        try:
            # Add regime labels
            features_df['regime_label'] = regime_labels
            
            # Add regime persistence
            features_df['regime_persistence'] = self._calculate_regime_persistence(regime_labels)
            
            # Add regime transitions
            features_df['regime_transitions'] = self._calculate_regime_transitions(regime_labels)
            
            # Add regime characteristics using shared utilities
            if 'create_regime_characteristics' in globals():
                try:
                    regime_characteristics = create_regime_characteristics(
                        features_df, regime_labels
                    )
                    for char_name, char_values in regime_characteristics.items():
                        features_df[f'regime_{char_name}'] = char_values
                except Exception as e:
                    tprint(f"⚠️ Failed to add regime characteristics: {e}", "WARNING")
            
            return features_df
            
        except Exception as e:
            tprint(f"❌ Failed to add regime-specific features: {e}", "ERROR")
            return features_df
    
    def _create_regime_datasets(
        self, 
        market_data: pd.DataFrame, 
        regime_labels: np.ndarray, 
        regime_features: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create regime-specific training datasets."""
        try:
            tprint("📊 Creating regime-specific training datasets...", "INFO")
            
            datasets = {}
            unique_regimes = np.unique(regime_labels[regime_labels != -1])  # Exclude noise (-1)
            
            for regime_id in unique_regimes:
                # Get samples for this regime
                regime_mask = regime_labels == regime_id
                regime_data = market_data[regime_mask]
                regime_features_data = regime_features['features_df'][regime_mask]
                
                if len(regime_data) > 0:
                    datasets[f'regime_{regime_id}'] = {
                        'market_data': regime_data,
                        'features': regime_features_data,
                        'n_samples': len(regime_data),
                        'regime_id': regime_id,
                        'start_date': regime_data.index.min(),
                        'end_date': regime_data.index.max()
                    }
            
            tprint(f"✅ Created {len(datasets)} regime-specific datasets", "SUCCESS")
            
            return datasets
            
        except Exception as e:
            tprint(f"❌ Dataset creation failed: {e}", "ERROR")
            return {}
    
    def _calculate_regime_statistics(
        self, 
        regime_labels: np.ndarray, 
        regime_probabilities: np.ndarray, 
        economic_profiles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate regime statistics."""
        try:
            unique_regimes = np.unique(regime_labels[regime_labels != -1])
            noise_samples = np.sum(regime_labels == -1)
            
            stats = {
                'n_regimes': len(unique_regimes),
                'n_samples': len(regime_labels),
                'noise_samples': noise_samples,
                'noise_ratio': noise_samples / len(regime_labels) if len(regime_labels) > 0 else 0,
                'regime_sizes': {},
                'regime_durations': {},
                'economic_profiles_count': len(economic_profiles)
            }
            
            # Calculate regime sizes
            for regime_id in unique_regimes:
                regime_size = np.sum(regime_labels == regime_id)
                stats['regime_sizes'][f'regime_{regime_id}'] = regime_size
            
            # Calculate regime durations (simplified)
            for regime_id in unique_regimes:
                regime_mask = regime_labels == regime_id
                regime_indices = np.where(regime_mask)[0]
                if len(regime_indices) > 1:
                    duration = regime_indices[-1] - regime_indices[0] + 1
                    stats['regime_durations'][f'regime_{regime_id}'] = duration
                else:
                    stats['regime_durations'][f'regime_{regime_id}'] = 1
            
            return stats
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate regime statistics: {e}", "WARNING")
            return {
                'n_regimes': 0,
                'n_samples': 0,
                'noise_samples': 0,
                'noise_ratio': 0,
                'regime_sizes': {},
                'regime_durations': {},
                'economic_profiles_count': 0
            }
    
    def _calculate_regime_persistence(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime persistence for each sample."""
        persistence = np.zeros(len(regime_labels))
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == regime_labels[i-1]:
                persistence[i] = persistence[i-1] + 1
            else:
                persistence[i] = 0
        
        return persistence
    
    def _calculate_regime_transitions(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime transition indicators."""
        transitions = np.zeros(len(regime_labels))
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i-1]:
                transitions[i] = 1
        
        return transitions
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band.fillna(prices), lower_band.fillna(prices)
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr.fillna(0)
    
    def _create_artifacts(self, regime_result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create artifacts from regime clustering result."""
        try:
            artifacts = {
                # Core regime data
                'regime_labels': regime_result['regime_labels'],
                'regime_probabilities': regime_result['regime_probabilities'],
                'regime_features': regime_result['regime_features'],
                'regime_datasets': regime_result['regime_datasets'],
                'regime_stats': regime_result['regime_stats'],
                'economic_profiles': regime_result['economic_profiles'],
                'regime_characteristics': regime_result.get('regime_characteristics', {}),
                'clustering_optimization': regime_result.get('clustering_optimization', {}),
                
                # Configuration
                'config': config,
                
                # Timestamps
                'created_at': datetime.now().isoformat(),
                'symbol': config['symbol'],
                'exchange': config['exchange'],
                'timeframe': config['timeframe']
            }
            
            return artifacts
            
        except Exception as e:
            tprint(f"⚠️ Failed to create artifacts: {e}", "WARNING")
            return {}
    
    def _save_artifacts(self, artifacts: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Save artifacts to disk."""
        try:
            # Create output directory
            output_dir = Path(config.get('data_dir', 'historical_data')) / 'regime_clustering' / config['symbol']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save regime features as parquet
            if 'regime_features' in artifacts and 'features_df' in artifacts['regime_features']:
                features_file = output_dir / f"regime_features_{config['symbol']}_{config['timeframe']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                artifacts['regime_features']['features_df'].to_parquet(features_file)
                tprint(f"✅ Regime features saved to {features_file}", "SUCCESS")
            
            # Save full artifacts as pickle
            artifacts_file = output_dir / f"regime_clustering_artifacts_{config['symbol']}_{config['timeframe']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            save_pickle(artifacts, artifacts_file)
            tprint(f"✅ Full artifacts saved to {artifacts_file}", "SUCCESS")
            
            # Save regime datasets as JSON (metadata only)
            if 'regime_datasets' in artifacts:
                datasets_metadata = {}
                for regime_name, dataset in artifacts['regime_datasets'].items():
                    datasets_metadata[regime_name] = {
                        'n_samples': dataset['n_samples'],
                        'regime_id': dataset['regime_id'],
                        'start_date': dataset['start_date'].isoformat() if hasattr(dataset['start_date'], 'isoformat') else str(dataset['start_date']),
                        'end_date': dataset['end_date'].isoformat() if hasattr(dataset['end_date'], 'isoformat') else str(dataset['end_date'])
                    }
                
                datasets_file = output_dir / f"regime_datasets_metadata_{config['symbol']}_{config['timeframe']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(datasets_file, 'w') as f:
                    json.dump(datasets_metadata, f, indent=2)
                tprint(f"✅ Regime datasets metadata saved to {datasets_file}", "SUCCESS")
            
            # Save clustering optimization results
            if 'clustering_optimization' in artifacts and artifacts['clustering_optimization']:
                optimization_file = output_dir / f"clustering_optimization_{config['symbol']}_{config['timeframe']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                optimization_data = {
                    'optimization_metrics': artifacts['clustering_optimization'].get('optimization_metrics', {}),
                    'global_reallocation_stats': artifacts['clustering_optimization'].get('global_reallocation_stats', {}),
                    'local_reallocation_stats': artifacts['clustering_optimization'].get('local_reallocation_stats', {}),
                    'consolidation_stats': artifacts['clustering_optimization'].get('consolidation_stats', {}),
                    'initial_validation': artifacts['clustering_optimization'].get('initial_validation', {}),
                    'final_validation': artifacts['clustering_optimization'].get('final_validation', {}),
                    'n_regimes': artifacts['clustering_optimization'].get('n_regimes', 0),
                    'n_samples': artifacts['clustering_optimization'].get('n_samples', 0)
                }
                with open(optimization_file, 'w') as f:
                    json.dump(optimization_data, f, indent=2)
                tprint(f"✅ Clustering optimization results saved to {optimization_file}", "SUCCESS")
                    json.dump(datasets_metadata, f, indent=2)
                tprint(f"✅ Regime datasets metadata saved to {datasets_file}", "SUCCESS")
            
        except Exception as e:
            tprint(f"⚠️ Failed to save artifacts: {e}", "WARNING")
    
    def _calculate_metrics(self, regime_result: Dict[str, Any], start_time: datetime, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate step execution metrics."""
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metrics = {
                'processing_time_seconds': processing_time,
                'n_regimes': regime_result.get('n_regimes', 0),
                'n_samples': regime_result.get('n_samples', 0),
                'n_features': regime_result.get('regime_features', {}).get('n_features', 0),
                'n_datasets': len(regime_result.get('regime_datasets', {})),
                'success': regime_result.get('success', False),
                'execution_mode': config.get('execution_mode', 'light'),
                'symbol': config.get('symbol', 'UNKNOWN'),
                'exchange': config.get('exchange', 'UNKNOWN'),
                'timeframe': config.get('timeframe', 'UNKNOWN')
            }
            
            # Add regime statistics
            if 'regime_stats' in regime_result:
                stats = regime_result['regime_stats']
                metrics.update({
                    'noise_ratio': stats.get('noise_ratio', 0.0),
                    'noise_samples': stats.get('noise_samples', 0),
                    'economic_profiles_count': stats.get('economic_profiles_count', 0)
                })
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate metrics: {e}", "WARNING")
            return {'success': False, 'error': str(e)}
    
    def _create_outcome_report(self, regime_result: Dict[str, Any], metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create outcome report markdown."""
        try:
            report = f"""# Regime Clustering Outcome Report

## Execution Summary
- **Symbol**: {config['symbol']}
- **Exchange**: {config['exchange']}
- **Timeframe**: {config['timeframe']}
- **Execution Mode**: {config.get('execution_mode', 'light')}
- **Processing Time**: {metrics.get('processing_time_seconds', 0):.2f} seconds
- **Success**: {'✅ Yes' if regime_result.get('success', False) else '❌ No'}

## Regime Clustering Results
- **Number of Regimes**: {metrics.get('n_regimes', 0)}
- **Total Samples**: {metrics.get('n_samples', 0)}
- **Features Generated**: {metrics.get('n_features', 0)}
- **Regime Datasets**: {metrics.get('n_datasets', 0)}
- **Noise Ratio**: {metrics.get('noise_ratio', 0.0):.1%}

## Regime Statistics
"""
            
            if 'regime_stats' in regime_result:
                stats = regime_result['regime_stats']
                report += f"""
- **Noise Samples**: {stats.get('noise_samples', 0)}
- **Economic Profiles**: {stats.get('economic_profiles_count', 0)}
- **Regime Sizes**: {stats.get('regime_sizes', {})}
- **Regime Durations**: {stats.get('regime_durations', {})}
"""
            
            report += f"""
## Generated Files
- Regime features (parquet)
- Full artifacts (pickle)
- Regime datasets metadata (JSON)
- This report (markdown)

## Configuration
- **Symbol**: {config['symbol']}
- **Exchange**: {config['exchange']}
- **Timeframe**: {config['timeframe']}
- **Data Directory**: {config.get('data_dir', 'historical_data')}

---
*Generated by Regime Clustering Component at {datetime.now().isoformat()}*
"""
            
            return report
            
        except Exception as e:
            tprint(f"⚠️ Failed to create outcome report: {e}", "WARNING")
            return f"# Regime Clustering Outcome Report\n\nError creating report: {str(e)}"
    
    async def _perform_multi_step_clustering_optimization(
        self, 
        features_df: pd.DataFrame, 
        initial_assignments: np.ndarray, 
        market_data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform multi-step clustering optimization with sample reallocation."""
        try:
            tprint("🔄 Starting multi-step clustering optimization...", "INFO")
            
            # Convert DataFrame to numpy array for clustering
            features = features_df.values
            
            # Step 1: Feature optimization and dimensionality reduction
            tprint("Step 1: Optimizing features and reducing dimensionality...", "INFO")
            optimized_features, feature_metadata = self._optimize_features_for_clustering(features, config)
            
            # Step 2: Initial clustering validation
            tprint("Step 2: Validating initial clustering...", "INFO")
            initial_validation = self._validate_clustering_robustness(optimized_features, initial_assignments, market_data)
            
            # Step 3: Perform neighborhood analysis
            tprint("Step 3: Performing neighborhood analysis...", "INFO")
            neighborhood_results = self._perform_neighborhood_analysis(optimized_features, initial_assignments)
            
            # Step 4: Sample reallocation at global level
            tprint("Step 4: Performing global sample reallocation...", "INFO")
            global_reallocated, global_stats = self._perform_global_sample_reallocation(
                optimized_features, initial_assignments, neighborhood_results, config
            )
            
            # Step 5: Sample reallocation at local level
            tprint("Step 5: Performing local sample reallocation...", "INFO")
            local_reallocated, local_stats = self._perform_local_sample_reallocation(
                optimized_features, global_reallocated, neighborhood_results, config
            )
            
            # Step 6: Regime consolidation and optimization
            tprint("Step 6: Consolidating and optimizing regimes...", "INFO")
            final_assignments, consolidation_stats = self._consolidate_regimes(
                optimized_features, local_reallocated, neighborhood_results, config
            )
            
            # Step 7: Final validation
            tprint("Step 7: Performing final validation...", "INFO")
            final_validation = self._validate_clustering_robustness(optimized_features, final_assignments, market_data)
            
            # Calculate optimization metrics
            optimization_metrics = self._calculate_optimization_metrics(
                initial_assignments, final_assignments, 
                global_stats, local_stats, consolidation_stats,
                initial_validation, final_validation
            )
            
            tprint("✅ Multi-step clustering optimization completed", "SUCCESS")
            
            return {
                'success': True,
                'initial_assignments': initial_assignments,
                'final_assignments': final_assignments,
                'optimized_features': optimized_features,
                'feature_metadata': feature_metadata,
                'neighborhood_results': neighborhood_results,
                'global_reallocation_stats': global_stats,
                'local_reallocation_stats': local_stats,
                'consolidation_stats': consolidation_stats,
                'initial_validation': initial_validation,
                'final_validation': final_validation,
                'optimization_metrics': optimization_metrics,
                'n_regimes': len(np.unique(final_assignments)),
                'n_samples': len(final_assignments)
            }
            
        except Exception as e:
            tprint(f"❌ Multi-step clustering optimization failed: {e}", "ERROR")
            return {
                'success': False,
                'error': str(e),
                'final_assignments': initial_assignments
            }
    
    def _optimize_features_for_clustering(
        self, 
        features: np.ndarray, 
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize features for clustering using dimensionality reduction."""
        try:
            tprint("🔧 Optimizing features for clustering...", "INFO")
            
            # Apply PCA for dimensionality reduction
            if MATRIX_OPERATIONS_AVAILABLE:
                try:
                    from sklearn.decomposition import PCA
                    
                    # Use variance-based PCA
                    pca = PCA(n_components=0.95, svd_solver='full')  # Keep 95% of variance
                    optimized_features = pca.fit_transform(features)
                    
                    explained_variance = pca.explained_variance_ratio_.sum()
                    n_components = optimized_features.shape[1]
                    
                    tprint(f"✅ PCA reduction: {features.shape[1]} -> {n_components} features (variance: {explained_variance:.3f})", "SUCCESS")
                    
                    metadata = {
                        'method': 'pca',
                        'n_components': n_components,
                        'explained_variance': explained_variance,
                        'original_shape': features.shape,
                        'reduced_shape': optimized_features.shape
                    }
                    
                except Exception as e:
                    tprint(f"⚠️ PCA failed, using original features: {e}", "WARNING")
                    optimized_features = features
                    metadata = {
                        'method': 'none',
                        'n_components': features.shape[1],
                        'explained_variance': 1.0,
                        'original_shape': features.shape,
                        'reduced_shape': features.shape
                    }
            else:
                optimized_features = features
                metadata = {
                    'method': 'none',
                    'n_components': features.shape[1],
                    'explained_variance': 1.0,
                    'original_shape': features.shape,
                    'reduced_shape': features.shape
                }
            
            return optimized_features, metadata
            
        except Exception as e:
            tprint(f"❌ Feature optimization failed: {e}", "ERROR")
            return features, {'error': str(e)}
    
    def _validate_clustering_robustness(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        market_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Validate clustering robustness using multiple metrics."""
        try:
            tprint("🔍 Validating clustering robustness...", "INFO")
            
            validation_results = {}
            
            # Basic clustering metrics
            n_clusters = len(np.unique(assignments))
            n_samples = features.shape[0]
            
            validation_results['n_clusters'] = n_clusters
            validation_results['n_samples'] = n_samples
            
            # Calculate silhouette score if possible
            try:
                from sklearn.metrics import silhouette_score
                if n_clusters > 1 and n_samples > 1:
                    silhouette = silhouette_score(features, assignments)
                    validation_results['silhouette_score'] = silhouette
                    tprint(f"📊 Silhouette Score: {silhouette:.4f}", "INFO")
            except Exception as e:
                tprint(f"⚠️ Silhouette score calculation failed: {e}", "WARNING")
                validation_results['silhouette_score'] = None
            
            # Calculate Davies-Bouldin index if possible
            try:
                from sklearn.metrics import davies_bouldin_score
                if n_clusters > 1:
                    db_score = davies_bouldin_score(features, assignments)
                    validation_results['davies_bouldin_score'] = db_score
                    tprint(f"📊 Davies-Bouldin Index: {db_score:.4f}", "INFO")
            except Exception as e:
                tprint(f"⚠️ Davies-Bouldin score calculation failed: {e}", "WARNING")
                validation_results['davies_bouldin_score'] = None
            
            # Calculate cluster balance
            unique, counts = np.unique(assignments, return_counts=True)
            cluster_sizes = counts
            min_size = np.min(cluster_sizes)
            max_size = np.max(cluster_sizes)
            balance_ratio = min_size / max_size if max_size > 0 else 0
            
            validation_results['cluster_balance'] = {
                'min_size': int(min_size),
                'max_size': int(max_size),
                'balance_ratio': balance_ratio,
                'cluster_sizes': cluster_sizes.tolist()
            }
            
            tprint(f"📊 Cluster Balance: {min_size}-{max_size} samples (ratio: {balance_ratio:.3f})", "INFO")
            
            return validation_results
            
        except Exception as e:
            tprint(f"❌ Clustering validation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def _perform_neighborhood_analysis(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Perform neighborhood analysis for local structure insights."""
        try:
            tprint("🔍 Performing neighborhood analysis...", "INFO")
            
            neighborhood_results = {}
            
            # K-NN consistency analysis
            try:
                from sklearn.neighbors import NearestNeighbors
                
                n_neighbors = min(10, len(features) - 1)
                nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
                nn.fit(features)
                
                # Find neighbors for each point
                distances, indices = nn.kneighbors(features)
                
                # Calculate consistency scores
                consistency_scores = []
                for i, neighbor_indices in enumerate(indices):
                    neighbor_assignments = assignments[neighbor_indices[1:]]  # Exclude self
                    if len(neighbor_assignments) > 0:
                        # Calculate consistency as the ratio of neighbors with the same assignment
                        same_assignment = np.sum(neighbor_assignments == assignments[i])
                        consistency = same_assignment / len(neighbor_assignments)
                        consistency_scores.append(consistency)
                    else:
                        consistency_scores.append(0.0)
                
                neighborhood_results['knn_consistency'] = {
                    'consistency_scores': consistency_scores,
                    'mean_consistency': np.mean(consistency_scores),
                    'std_consistency': np.std(consistency_scores)
                }
                
                tprint(f"📊 K-NN Consistency: {np.mean(consistency_scores):.3f} ± {np.std(consistency_scores):.3f}", "INFO")
                
            except Exception as e:
                tprint(f"⚠️ K-NN analysis failed: {e}", "WARNING")
                neighborhood_results['knn_consistency'] = {'error': str(e)}
            
            # Local silhouette analysis
            try:
                local_scores = self._compute_local_silhouette_scores(features, assignments)
                neighborhood_results['local_silhouette'] = {
                    'local_scores': local_scores,
                    'mean_local_score': np.mean(local_scores),
                    'std_local_score': np.std(local_scores)
                }
                
                tprint(f"📊 Local Silhouette: {np.mean(local_scores):.3f} ± {np.std(local_scores):.3f}", "INFO")
                
            except Exception as e:
                tprint(f"⚠️ Local silhouette analysis failed: {e}", "WARNING")
                neighborhood_results['local_silhouette'] = {'error': str(e)}
            
            return neighborhood_results
            
        except Exception as e:
            tprint(f"❌ Neighborhood analysis failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def _compute_local_silhouette_scores(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray
    ) -> List[float]:
        """Compute local silhouette scores for each point."""
        try:
            local_scores = []
            
            for i in range(len(features)):
                # Get the cluster assignment for this point
                cluster_id = assignments[i]
                
                # Find all points in the same cluster
                same_cluster_mask = assignments == cluster_id
                same_cluster_indices = np.where(same_cluster_mask)[0]
                
                if len(same_cluster_indices) < 2:
                    local_scores.append(0.0)
                    continue
                
                # Calculate intra-cluster distance (a_i)
                intra_distances = []
                for j in same_cluster_indices:
                    if i != j:
                        dist = np.linalg.norm(features[i] - features[j])
                        intra_distances.append(dist)
                
                a_i = np.mean(intra_distances) if intra_distances else 0.0
                
                # Calculate inter-cluster distances (b_i)
                other_clusters = np.unique(assignments[assignments != cluster_id])
                inter_distances = []
                
                for other_cluster in other_clusters:
                    other_cluster_mask = assignments == other_cluster
                    other_cluster_indices = np.where(other_cluster_mask)[0]
                    
                    if len(other_cluster_indices) > 0:
                        cluster_distances = []
                        for j in other_cluster_indices:
                            dist = np.linalg.norm(features[i] - features[j])
                            cluster_distances.append(dist)
                        inter_distances.append(np.mean(cluster_distances))
                
                b_i = np.min(inter_distances) if inter_distances else 0.0
                
                # Calculate local silhouette score
                if max(a_i, b_i) > 0:
                    local_score = (b_i - a_i) / max(a_i, b_i)
                else:
                    local_score = 0.0
                
                local_scores.append(local_score)
            
            return local_scores
            
        except Exception as e:
            tprint(f"❌ Local silhouette calculation failed: {e}", "ERROR")
            return [0.0] * len(features)
    
    def _perform_global_sample_reallocation(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        neighborhood_results: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform global-level sample reallocation."""
        try:
            tprint("🔄 Performing global sample reallocation...", "INFO")
            
            # Get neighborhood analysis results
            knn_results = neighborhood_results.get('knn_consistency', {})
            local_silhouette = neighborhood_results.get('local_silhouette', {})
            
            if knn_results.get('error') or local_silhouette.get('error'):
                tprint("⚠️ Neighborhood analysis incomplete, skipping global reallocation", "WARNING")
                return assignments, {'reallocation_skipped': True}
            
            # Identify candidates for reallocation
            consistency_scores = knn_results.get('consistency_scores', [])
            local_scores = local_silhouette.get('local_scores', [])
            
            if not consistency_scores or not local_scores:
                tprint("⚠️ No consistency or local scores available, skipping reallocation", "WARNING")
                return assignments, {'reallocation_skipped': True}
            
            # Find points with poor global consistency
            consistency_threshold = 0.6
            local_threshold = -0.1
            
            misclustered_mask = np.array([
                consistency < consistency_threshold and local_score < local_threshold
                for consistency, local_score in zip(consistency_scores, local_scores)
            ])
            
            n_misclustered = np.sum(misclustered_mask)
            if n_misclustered == 0:
                tprint("ℹ️ No misclustered points found for global reallocation", "INFO")
                return assignments, {'reallocated_points': 0, 'reason': 'no_misclustered_points'}
            
            tprint(f"🔄 Found {n_misclustered} candidates for global reallocation", "INFO")
            
            # Perform reallocation using k-NN
            reallocated_assignments, reallocation_stats = self._reallocate_misclustered_points(
                features, assignments, misclustered_mask, knn_results, local_silhouette
            )
            
            tprint(f"✅ Global reallocation completed: {reallocation_stats.get('reallocated_points', 0)} points moved", "SUCCESS")
            
            return reallocated_assignments, reallocation_stats
            
        except Exception as e:
            tprint(f"❌ Global sample reallocation failed: {e}", "ERROR")
            return assignments, {'error': str(e)}
    
    def _perform_local_sample_reallocation(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        neighborhood_results: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform local-level sample reallocation."""
        try:
            tprint("🔄 Performing local sample reallocation...", "INFO")
            
            # For now, implement a simplified local reallocation
            # In a full implementation, this would use more sophisticated local optimization
            
            local_silhouette = neighborhood_results.get('local_silhouette', {})
            local_scores = local_silhouette.get('local_scores', [])
            
            if not local_scores:
                tprint("⚠️ No local scores available, skipping local reallocation", "WARNING")
                return assignments, {'reallocated_points': 0, 'reason': 'no_local_scores'}
            
            # Find points with very poor local silhouette scores
            local_threshold = -0.3
            poor_local_mask = np.array(local_scores) < local_threshold
            
            n_poor_local = np.sum(poor_local_mask)
            if n_poor_local == 0:
                tprint("ℹ️ No points with poor local scores found", "INFO")
                return assignments, {'reallocated_points': 0, 'reason': 'no_poor_local_points'}
            
            tprint(f"🔄 Found {n_poor_local} points with poor local scores", "INFO")
            
            # Simple local reallocation: move points to nearest cluster centroid
            reallocated_assignments = assignments.copy()
            reallocation_count = 0
            
            # Calculate cluster centroids
            unique_clusters = np.unique(assignments)
            centroids = {}
            for cluster_id in unique_clusters:
                cluster_mask = assignments == cluster_id
                cluster_points = features[cluster_mask]
                if len(cluster_points) > 0:
                    centroids[cluster_id] = np.mean(cluster_points, axis=0)
            
            # Reallocate poor local points
            for i in np.where(poor_local_mask)[0]:
                if i >= len(assignments):
                    continue
                    
                current_cluster = assignments[i]
                point = features[i]
                
                # Find nearest centroid
                best_cluster = current_cluster
                best_distance = float('inf')
                
                for cluster_id, centroid in centroids.items():
                    if cluster_id != current_cluster:
                        distance = np.linalg.norm(point - centroid)
                        if distance < best_distance:
                            best_distance = distance
                            best_cluster = cluster_id
                
                # Only reallocate if it's different and improves local score
                if best_cluster != current_cluster:
                    # Test the move
                    test_assignments = reallocated_assignments.copy()
                    test_assignments[i] = best_cluster
                    
                    # Calculate new local score
                    new_local_score = self._compute_local_silhouette_scores(features, test_assignments)[i]
                    current_local_score = local_scores[i]
                    
                    if new_local_score > current_local_score:
                        reallocated_assignments[i] = best_cluster
                        reallocation_count += 1
            
            tprint(f"✅ Local reallocation completed: {reallocation_count} points moved", "SUCCESS")
            
            return reallocated_assignments, {
                'reallocated_points': reallocation_count,
                'total_candidates': n_poor_local,
                'success_rate': reallocation_count / max(1, n_poor_local)
            }
            
        except Exception as e:
            tprint(f"❌ Local sample reallocation failed: {e}", "ERROR")
            return assignments, {'error': str(e)}
    
    def _reallocate_misclustered_points(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        misclustered_mask: np.ndarray, 
        knn_results: Dict[str, Any], 
        local_silhouette: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reallocate misclustered points using k-NN consensus."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            n_misclustered = np.sum(misclustered_mask)
            if n_misclustered == 0:
                return assignments, {'reallocated_points': 0, 'reason': 'no_misclustered_points'}
            
            # Fit k-NN for distance-based reallocation
            n_neighbors = min(10, len(features) - 1)
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            nn.fit(features)
            
            reallocated_assignments = assignments.copy()
            reallocation_count = 0
            
            # Throttling constraints
            max_moves = max(2, int(0.05 * len(assignments)))  # Cap at 5% of samples
            neighbor_consensus_threshold = 0.7  # Require 70% neighbor consensus
            
            for i in np.where(misclustered_mask)[0]:
                if reallocation_count >= max_moves:
                    break
                
                # Find neighbors
                distances, indices = nn.kneighbors(features[i:i+1])
                neighbor_indices = indices[0][1:]  # Exclude self
                neighbor_assignments = assignments[neighbor_indices]
                
                if len(neighbor_assignments) == 0:
                    continue
                
                # Find most common cluster among neighbors
                unique_clusters, counts = np.unique(neighbor_assignments, return_counts=True)
                consensus_ratio = np.max(counts) / len(neighbor_assignments)
                
                # Require neighbor consensus
                if consensus_ratio < neighbor_consensus_threshold:
                    continue
                
                best_neighbor_cluster = unique_clusters[np.argmax(counts)]
                current_cluster = assignments[i]
                
                # Only reallocate if different cluster
                if best_neighbor_cluster != current_cluster:
                    reallocated_assignments[i] = best_neighbor_cluster
                    reallocation_count += 1
            
            return reallocated_assignments, {
                'reallocated_points': reallocation_count,
                'total_misclustered': n_misclustered,
                'success_rate': reallocation_count / max(1, n_misclustered)
            }
            
        except Exception as e:
            tprint(f"❌ Misclustered points reallocation failed: {e}", "ERROR")
            return assignments, {'error': str(e)}
    
    def _consolidate_regimes(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        neighborhood_results: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Consolidate regimes by merging small clusters and splitting large ones."""
        try:
            tprint("🔄 Consolidating regimes...", "INFO")
            
            # Calculate cluster sizes
            unique_clusters, counts = np.unique(assignments, return_counts=True)
            cluster_sizes = dict(zip(unique_clusters, counts))
            
            # Define thresholds
            min_cluster_size = max(2, len(assignments) // 20)  # At least 5% of data
            max_cluster_size = len(assignments) // 3  # At most 33% of data
            
            consolidated_assignments = assignments.copy()
            consolidation_changes = 0
            
            # Merge small clusters
            small_clusters = [cluster for cluster, size in cluster_sizes.items() if size < min_cluster_size]
            if small_clusters:
                tprint(f"🔄 Merging {len(small_clusters)} small clusters", "INFO")
                
                # Find the largest cluster to merge small ones into
                largest_cluster = max(cluster_sizes.items(), key=lambda x: x[1])[0]
                
                for small_cluster in small_clusters:
                    mask = consolidated_assignments == small_cluster
                    consolidated_assignments[mask] = largest_cluster
                    consolidation_changes += np.sum(mask)
            
            # Split large clusters (simplified implementation)
            large_clusters = [cluster for cluster, size in cluster_sizes.items() if size > max_cluster_size]
            if large_clusters:
                tprint(f"🔄 Splitting {len(large_clusters)} large clusters", "INFO")
                
                for large_cluster in large_clusters:
                    # Simple splitting: use k-means with k=2
                    try:
                        from sklearn.cluster import KMeans
                        
                        cluster_mask = consolidated_assignments == large_cluster
                        cluster_points = features[cluster_mask]
                        
                        if len(cluster_points) > 2:
                            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                            sub_assignments = kmeans.fit_predict(cluster_points)
                            
                            # Create new cluster IDs
                            new_cluster_id = max(consolidated_assignments) + 1
                            
                            # Update assignments
                            cluster_indices = np.where(cluster_mask)[0]
                            for i, sub_assignment in enumerate(sub_assignments):
                                if sub_assignment == 1:  # Move second sub-cluster to new ID
                                    consolidated_assignments[cluster_indices[i]] = new_cluster_id
                                    consolidation_changes += 1
                    
                    except Exception as e:
                        tprint(f"⚠️ Failed to split cluster {large_cluster}: {e}", "WARNING")
            
            tprint(f"✅ Regime consolidation completed: {consolidation_changes} changes", "SUCCESS")
            
            return consolidated_assignments, {
                'consolidation_changes': consolidation_changes,
                'merged_clusters': len(small_clusters),
                'split_clusters': len(large_clusters)
            }
            
        except Exception as e:
            tprint(f"❌ Regime consolidation failed: {e}", "ERROR")
            return assignments, {'error': str(e)}
    
    def _calculate_optimization_metrics(
        self, 
        initial_assignments: np.ndarray, 
        final_assignments: np.ndarray, 
        global_stats: Dict[str, Any], 
        local_stats: Dict[str, Any], 
        consolidation_stats: Dict[str, Any],
        initial_validation: Dict[str, Any], 
        final_validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate optimization metrics."""
        try:
            # Calculate assignment changes
            total_changes = np.sum(initial_assignments != final_assignments)
            change_rate = total_changes / len(initial_assignments) if len(initial_assignments) > 0 else 0
            
            # Calculate cluster count changes
            initial_n_clusters = len(np.unique(initial_assignments))
            final_n_clusters = len(np.unique(final_assignments))
            
            # Calculate quality improvements
            initial_silhouette = initial_validation.get('silhouette_score', 0)
            final_silhouette = final_validation.get('silhouette_score', 0)
            silhouette_improvement = final_silhouette - initial_silhouette if initial_silhouette is not None and final_silhouette is not None else 0
            
            # Calculate balance improvements
            initial_balance = initial_validation.get('cluster_balance', {}).get('balance_ratio', 0)
            final_balance = final_validation.get('cluster_balance', {}).get('balance_ratio', 0)
            balance_improvement = final_balance - initial_balance
            
            return {
                'total_changes': int(total_changes),
                'change_rate': change_rate,
                'initial_n_clusters': initial_n_clusters,
                'final_n_clusters': final_n_clusters,
                'cluster_count_change': final_n_clusters - initial_n_clusters,
                'silhouette_improvement': silhouette_improvement,
                'balance_improvement': balance_improvement,
                'global_reallocations': global_stats.get('reallocated_points', 0),
                'local_reallocations': local_stats.get('reallocated_points', 0),
                'consolidation_changes': consolidation_stats.get('consolidation_changes', 0),
                'total_optimization_changes': (
                    global_stats.get('reallocated_points', 0) + 
                    local_stats.get('reallocated_points', 0) + 
                    consolidation_stats.get('consolidation_changes', 0)
                )
            }
            
        except Exception as e:
            tprint(f"❌ Optimization metrics calculation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def _generate_regime_characteristics(
        self, 
        regime_features: Dict[str, Any], 
        regime_labels: np.ndarray, 
        economic_profiles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate regime characteristics using shared utilities."""
        try:
            if 'create_regime_characteristics' not in globals():
                return {}
            
            tprint("🔧 Generating regime characteristics...", "INFO")
            
            # Use shared characteristics generation
            characteristics = create_regime_characteristics(
                regime_features['features_df'], regime_labels
            )
            
            # Add economic profile characteristics
            if economic_profiles:
                for profile in economic_profiles:
                    regime_id = profile.get('regime_id', 'unknown')
                    if f'regime_{regime_id}' in characteristics:
                        characteristics[f'regime_{regime_id}'].update({
                            'economic_profile': profile,
                            'key_stats': profile.get('key_stats', {}),
                            'confidence_intervals': profile.get('confidence_intervals', {}),
                            'avg_duration': profile.get('avg_duration', 0),
                            'works_best_for': profile.get('works_best_for', []),
                            'risk_caveats': profile.get('risk_caveats', [])
                        })
            
            tprint(f"✅ Generated characteristics for {len(characteristics)} regimes", "SUCCESS")
            
            return characteristics
            
        except Exception as e:
            tprint(f"⚠️ Failed to generate regime characteristics: {e}", "WARNING")
            return {}


# Register the step
def register_regime_clustering_step():
    """Register the regime clustering step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("regime_clustering", RegimeClusteringComponent)
    tprint("✅ Regime clustering step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_clustering_step()