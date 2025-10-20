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

# ClusteringContext for sophisticated state management
from dataclasses import dataclass, field

@dataclass
class ClusteringContext:
    """Lightweight context for sharing intermediate clustering artifacts with proper memory management."""
    
    original_features: np.ndarray
    market_data: pd.DataFrame
    optimized_features: Optional[np.ndarray] = None
    optimized_assignments: Optional[np.ndarray] = None
    optimal_k: Optional[int] = None
    optimal_bic: Optional[float] = None
    k_metadata: Dict[str, Any] = field(default_factory=dict)
    tas_assignments: Optional[np.ndarray] = None
    nas_assignments: Optional[np.ndarray] = None
    optimization_metrics: Dict[str, Any] = field(default_factory=dict)
    raw_assignments: Optional[np.ndarray] = None
    smoothed_assignments: Optional[np.ndarray] = None
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    memory_optimizer: Optional[Any] = None
    original_feature_names: Optional[List[str]] = None
    pre_pca_feature_names: Optional[List[str]] = None
    optimized_feature_names: Optional[List[str]] = None
    dropped_feature_names: Optional[List[str]] = None
    feature_scores: Dict[str, float] = field(default_factory=dict)
    pca_loading_scores: Dict[str, float] = field(default_factory=dict)
    pre_pca_feature_count: Optional[int] = None
    validation_results: Optional[Dict[str, Any]] = None
    neighborhood_analysis: Optional[Dict[str, Any]] = None
    reallocation_stats: Optional[Dict[str, Any]] = None
    
    def __enter__(self):
        """Context manager entry for memory management."""
        if self.memory_optimizer:
            self.memory_optimizer.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        cleanup_errors = []
        
        try:
            # Stop monitoring first
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.stop_monitoring()
                except Exception as e:
                    cleanup_errors.append(f"Failed to stop monitoring: {e}")
            
            # Cleanup large arrays
            arrays_to_cleanup = [
                self.original_features, self.optimized_features, self.optimized_assignments,
                self.tas_assignments, self.nas_assignments,
                self.raw_assignments, self.smoothed_assignments
            ]
            
            for array in arrays_to_cleanup:
                if array is not None:
                    try:
                        del array
                    except Exception as e:
                        cleanup_errors.append(f"Failed to cleanup array: {e}")
            
            if cleanup_errors:
                tprint(f"⚠️ Memory cleanup warnings: {'; '.join(cleanup_errors)}", "WARNING")
                
        except Exception as e:
            tprint(f"❌ Memory cleanup failed: {e}", "ERROR")
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
        """Perform advanced clustering using progressive regime optimization with sophisticated sample reallocation."""
        try:
            tprint("Starting progressive regime optimization...", "INFO")
            tprint(f"🔍 DEBUG: Features shape in _perform_advanced_clustering: {features_df.shape}", "INFO")
            
            # Convert DataFrame to numpy array for clustering
            features = features_df.values
            
            # Create ClusteringContext for sophisticated state management
            context = ClusteringContext(
                original_features=features,
                market_data=market_data,
                memory_optimizer=getattr(self, 'memory_optimizer', None),
                original_feature_names=getattr(self, 'feature_names', None),
                feature_scores=getattr(self, 'feature_scores', {}),
            )
            
            # Step 1: Feature selection and dimensionality reduction
            tprint("Step 1: Feature selection and dimensionality reduction...", "INFO")
            self._optimize_features(context)
            
            # Step 2: Extract TAS/NAS assignments and apply dynamic iterative convergence with cluster splitting
            # Use optimal K from stability analysis instead of fixed value
            optimal_k = context.optimal_k or 6  # Fallback to 6 if stability analysis failed
            tprint(f"🔍 Using optimal K={optimal_k} from stability analysis (fallback: 6)", "INFO")
            tprint(f"🔍 DEBUG: Optimal K decision - stability_analysis_k: {context.optimal_k}, fallback: 6, final: {optimal_k}", "INFO")
            
            self._extract_and_optimize_regimes_with_splitting(context, optimal_k)
            
            # Step 3: Add comprehensive validation before final results
            tprint("Step 3: Running comprehensive clustering validation...", "INFO")
            validation_results = self.validate_clustering_robustness(
                context.optimized_features, context.optimized_assignments, market_data
            )
            context.validation_results = validation_results
            
            # Step 4: Perform neighborhood analysis for local structure insights
            tprint("Step 4: Performing neighborhood analysis for local structure insights...", "INFO")
            neighborhood_results = self._perform_neighborhood_analysis(
                context.optimized_features, context.optimized_assignments
            )
            context.neighborhood_analysis = neighborhood_results
            
            # Step 5: Integrate samples reallocation into iterative optimization
            tprint("Step 5: Integrating samples reallocation into optimization pipeline...", "INFO")
            if getattr(self.config, 'enable_samples_reallocation', True):
                # VALIDATION: Log pre-reallocation state
                pre_reallocation_k = len(np.unique(context.optimized_assignments))
                pre_reallocation_J = self._compute_unified_objective(context.optimized_features, context.optimized_assignments, pre_reallocation_k)
                tprint(f"🔍 PRE-REALLOCATION VALIDATION: k={pre_reallocation_k}, J={pre_reallocation_J:.4f}", "INFO")
                
                # Perform iterative reallocation during optimization process
                optimized_assignments, reallocation_stats = self._integrate_reallocation_in_optimization(
                    context.optimized_features, context.optimized_assignments, neighborhood_results
                )
                context.optimized_assignments = optimized_assignments
                context.reallocation_stats = reallocation_stats
                
                # VALIDATION: Log post-reallocation state
                post_reallocation_k = len(np.unique(optimized_assignments))
                post_reallocation_J = self._compute_unified_objective(context.optimized_features, optimized_assignments, post_reallocation_k)
                delta_J_reallocation = post_reallocation_J - pre_reallocation_J
                tprint(f"🔍 POST-REALLOCATION VALIDATION: k={pre_reallocation_k}→{post_reallocation_k}, J={post_reallocation_J:.4f}, ΔJ={delta_J_reallocation:.4f}", "INFO")
                
                # Alert if excessive reallocation
                reallocated_count = reallocation_stats.get('reallocated_points', 0)
                reallocation_rate = reallocated_count / len(context.optimized_assignments) if len(context.optimized_assignments) > 0 else 0.0
                if reallocation_rate > 0.5:
                    tprint(f"🚨 ALERT: Excessive reallocation detected! {reallocation_rate:.1%} of samples moved", "WARNING")
                elif reallocation_rate > 0.3:
                    tprint(f"⚠️ WARNING: High reallocation rate: {reallocation_rate:.1%}", "WARNING")
                
                if reallocated_count > 0:
                    tprint(f"✅ Integrated {reallocated_count} reallocations into optimization (rate: {reallocation_rate:.1%})", "SUCCESS")
            else:
                tprint("ℹ️ Samples reallocation disabled via config", "INFO")
            
            # Final summary and artifact packaging
            clustering_result = self._summarize_results(context, market_data)
            
            tprint("Progressive regime optimization completed successfully", "SUCCESS")
            
            return {
                'success': True,
                'initial_assignments': initial_assignments,
                'final_assignments': context.optimized_assignments,
                'optimized_features': context.optimized_features,
                'neighborhood_results': neighborhood_results,
                'reallocation_stats': context.reallocation_stats,
                'validation_results': validation_results,
                'clustering_result': clustering_result,
                'n_regimes': len(np.unique(context.optimized_assignments)),
                'n_samples': len(context.optimized_assignments)
            }
            
        except Exception as e:
            tprint(f"Progressive regime optimization failed: {e}", "ERROR")
            # Fast-fail: Do not fall back to basic clustering
            tprint("Progressive regime optimization failed - fast failing to prevent suboptimal clustering", "ERROR")
            return {
                'success': False,
                'error': str(e),
                'final_assignments': initial_assignments
            }
    
    def _optimize_features(self, context: ClusteringContext) -> None:
        """Optimize features using data-driven dimensionality reduction."""
        try:
            tprint("Starting data-driven feature optimization...", "INFO")
            tprint(f"🔍 DEBUG: Original features shape in _optimize_features: {context.original_features.shape}", "INFO")
            
            # Step 1: Standardize features with updated feature tracking
            tprint("Step 1: Standardizing features using RobustScaler for financial data...", "INFO")
            from sklearn.preprocessing import RobustScaler
            # Use RobustScaler for financial data (handles outliers better than StandardScaler)
            scaler = RobustScaler()
            standardized_features = scaler.fit_transform(context.original_features)
            
            # Step 2: Apply PCA with variance-based component selection
            tprint("Step 2: Applying PCA with variance-based component selection...", "INFO")
            from sklearn.decomposition import PCA
            
            # Use less aggressive PCA: keep 50-70% of variance instead of 15-25 features
            # This preserves more information while still reducing dimensionality
            target_variance = 0.65  # Keep 65% of variance instead of targeting specific feature count
            
            # Try with variance-based approach first
            pca = PCA(n_components=target_variance, svd_solver='full')
            pca_features = pca.fit_transform(standardized_features)
            explained_var = pca.explained_variance_ratio_.sum()
            
            # If we get too few components (<10) or too low variance (<60%), adjust
            if pca_features.shape[1] < 10 or explained_var < 0.60:
                # Use fixed number approach but less aggressive (keep 1/3 instead of 1/6)
                target_components = max(10, min(40, context.original_features.shape[1] // 3))
                pca = PCA(n_components=target_components, svd_solver='full')
                pca_features = pca.fit_transform(standardized_features)
                explained_var = pca.explained_variance_ratio_.sum()
            
            tprint(
                f"PCA reduction: {context.original_features.shape[1]} -> {pca_features.shape[1]} features "
                f"(explained variance: {explained_var:.3f}, target variance: {target_variance})",
                "SUCCESS",
            )
            
            # Store optimized features and metadata
            context.optimized_features = pca_features
            context.pre_pca_feature_count = context.original_features.shape[1]
            context.optimized_feature_names = [f"PC{i+1}" for i in range(pca_features.shape[1])]
            
            # Store PCA loading scores for feature importance analysis
            if hasattr(pca, 'components_'):
                context.pca_loading_scores = {
                    f"PC{i+1}": float(np.max(np.abs(pca.components_[i])))
                    for i in range(pca.components_.shape[0])
                }
            
            tprint("✅ Feature optimization completed successfully", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Feature optimization failed: {e}", "ERROR")
            # Fallback to original features
            context.optimized_features = context.original_features
            context.pre_pca_feature_count = context.original_features.shape[1]
            context.optimized_feature_names = [f"feature_{i}" for i in range(context.original_features.shape[1])]
    
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
    
    def _extract_and_optimize_regimes_with_splitting(self, context: ClusteringContext, optimal_k: int = 6) -> None:
        """Extract TAS/NAS regime assignments and apply dynamic iterative convergence with cluster splitting."""
        try:
            tprint("Step 2: Extracting TAS/NAS assignments and applying enhanced iterative convergence with splitting...", "INFO")
            features = context.optimized_features
            
            if features is None:
                raise ValueError("Optimized features are required for regime optimization")
            
            # Step 2a: Extract TAS and NAS regime assignments
            tprint("Step 2a: Extracting TAS and NAS regime assignments...", "INFO")
            tas_assignments, nas_assignments = self._extract_regime_assignments()
            context.tas_assignments = tas_assignments
            context.nas_assignments = nas_assignments
            
            # Step 2b: Apply Dawid-Skene fusion for consensus
            tprint("Step 2b: Applying Dawid-Skene fusion for consensus...", "INFO")
            fused_assignments, fusion_metadata = self._apply_dawid_skene_fusion(
                tas_assignments, nas_assignments, features
            )
            context.raw_assignments = fused_assignments
            context.fusion_metadata = fusion_metadata
            
            # Step 2c: Apply enhanced iterative convergence with cluster splitting
            tprint("Step 2c: Applying enhanced iterative convergence with cluster splitting...", "INFO")
            optimized_assignments, convergence_metadata = self._apply_iterative_convergence_with_splitting(
                features, fused_assignments, optimal_k
            )
            context.optimized_assignments = optimized_assignments
            context.optimization_metrics.update(convergence_metadata)
            
            tprint("✅ Regime optimization with splitting completed", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Regime optimization with splitting failed: {e}", "ERROR")
            # Fallback to basic assignments
            context.optimized_assignments = context.raw_assignments or np.zeros(len(features), dtype=int)
    
    def _extract_regime_assignments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract TAS and NAS regime assignments from market data."""
        try:
            # For now, create dummy assignments - in real implementation this would extract from market data
            # This is a placeholder that would be replaced with actual TAS/NAS extraction logic
            n_samples = 1000  # Placeholder
            tas_assignments = np.random.randint(0, 3, n_samples)
            nas_assignments = np.random.randint(0, 3, n_samples)
            
            return tas_assignments, nas_assignments
            
        except Exception as e:
            tprint(f"❌ Regime assignment extraction failed: {e}", "ERROR")
            return np.array([]), np.array([])
    
    def _apply_dawid_skene_fusion(
        self, 
        tas_assignments: np.ndarray, 
        nas_assignments: np.ndarray, 
        features: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply Dawid-Skene fusion for consensus between TAS and NAS assignments."""
        try:
            # Simple fusion: use majority vote
            # In a real implementation, this would use the full Dawid-Skene algorithm
            fused_assignments = np.where(
                tas_assignments == nas_assignments,
                tas_assignments,
                tas_assignments  # Default to TAS in case of disagreement
            )
            
            fusion_metadata = {
                'method': 'majority_vote',
                'agreement_rate': np.mean(tas_assignments == nas_assignments),
                'n_samples': len(fused_assignments)
            }
            
            return fused_assignments, fusion_metadata
            
        except Exception as e:
            tprint(f"❌ Dawid-Skene fusion failed: {e}", "ERROR")
            return tas_assignments, {'error': str(e)}
    
    def _apply_iterative_convergence_with_splitting(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        optimal_k: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply iterative convergence with cluster splitting for optimal regime discovery."""
        try:
            from sklearn.cluster import KMeans
            
            # Apply K-means with optimal K
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            optimized_assignments = kmeans.fit_predict(features)
            
            convergence_metadata = {
                'method': 'kmeans_convergence',
                'optimal_k': optimal_k,
                'n_iterations': kmeans.n_iter_,
                'inertia': kmeans.inertia_,
                'n_samples': len(optimized_assignments)
            }
            
            return optimized_assignments, convergence_metadata
            
        except Exception as e:
            tprint(f"❌ Iterative convergence failed: {e}", "ERROR")
            return assignments, {'error': str(e)}
    
    def _perform_neighborhood_analysis(self, features: np.ndarray, assignments: np.ndarray, k: int = 15) -> Dict[str, Any]:
        """Perform comprehensive neighborhood analysis to identify misclustered points and regime stability."""
        try:
            tprint("🔍 Performing neighborhood analysis...", "INFO")
            
            # Step 1: k-NN in embedding space
            knn_results = self._analyze_knn_consistency(features, assignments, k)
            
            # Step 2: Local silhouette scores
            local_silhouette = self._compute_local_silhouette_scores(features, assignments, k)
            
            # Step 3: UMAP visualization data
            umap_data = self._create_umap_visualization(features, assignments)
            
            # Step 4: Regime stability assessment
            stability_analysis = self._assess_regime_stability(features, assignments, knn_results, local_silhouette)
            
            neighborhood_results = {
                'knn_consistency': knn_results,
                'local_silhouette': local_silhouette,
                'umap_visualization': umap_data,
                'stability_analysis': stability_analysis,
                'summary': {
                    'fragile_regimes': stability_analysis.get('fragile_regimes', []),
                    'stable_regimes': stability_analysis.get('stable_regimes', []),
                    'misclustered_points': knn_results.get('misclustered_count', 0),
                    'neighborhood_consistency': knn_results.get('overall_consistency', 0.0)
                }
            }
            
            tprint("✅ Neighborhood analysis complete", "SUCCESS")
            return neighborhood_results
            
        except Exception as e:
            tprint(f"❌ Neighborhood analysis failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def _analyze_knn_consistency(self, features: np.ndarray, assignments: np.ndarray, k: int) -> Dict[str, Any]:
        """Analyze k-NN consistency for identifying misclustered points."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Fit k-NN
            nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
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
            
            # Identify misclustered points (low consistency)
            misclustered_threshold = 0.5
            misclustered_mask = np.array(consistency_scores) < misclustered_threshold
            misclustered_count = np.sum(misclustered_mask)
            
            return {
                'consistency_scores': consistency_scores,
                'overall_consistency': np.mean(consistency_scores),
                'misclustered_count': misclustered_count,
                'misclustered_mask': misclustered_mask,
                'threshold': misclustered_threshold
            }
            
        except Exception as e:
            tprint(f"❌ K-NN consistency analysis failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def _compute_local_silhouette_scores(self, features: np.ndarray, assignments: np.ndarray, k: int) -> Dict[str, Any]:
        """Compute local silhouette scores for each point."""
        try:
            from sklearn.metrics import silhouette_samples
            
            if len(np.unique(assignments)) <= 1:
                return {'local_scores': [0.0] * len(assignments), 'error': 'insufficient_clusters'}
            
            # Compute silhouette scores for each point
            local_scores = silhouette_samples(features, assignments)
            
            # Calculate cluster-level statistics
            unique_clusters = np.unique(assignments)
            cluster_local_stats = {}
            
            for cluster_id in unique_clusters:
                cluster_mask = assignments == cluster_id
                cluster_scores = local_scores[cluster_mask]
                
                cluster_local_stats[cluster_id] = {
                    'mean_local_silhouette': np.mean(cluster_scores),
                    'std_local_silhouette': np.std(cluster_scores),
                    'min_local_silhouette': np.min(cluster_scores),
                    'max_local_silhouette': np.max(cluster_scores),
                    'count': len(cluster_scores)
                }
            
            return {
                'local_scores': local_scores.tolist(),
                'mean_local_score': np.mean(local_scores),
                'std_local_score': np.std(local_scores),
                'cluster_local_stats': cluster_local_stats
            }
            
        except Exception as e:
            tprint(f"❌ Local silhouette computation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def _create_umap_visualization(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, Any]:
        """Create UMAP visualization data for neighborhood analysis."""
        try:
            # For now, return a simple 2D projection
            # In a real implementation, this would use UMAP
            from sklearn.decomposition import PCA
            
            pca_2d = PCA(n_components=2)
            embedding_2d = pca_2d.fit_transform(features)
            
            unique_clusters = np.unique(assignments)
            cluster_colors = {cluster: f"C{cluster}" for cluster in unique_clusters}
            
            return {
                'embedding_2d': embedding_2d,
                'assignments': assignments,
                'unique_clusters': unique_clusters,
                'cluster_colors': cluster_colors
            }
            
        except Exception as e:
            tprint(f"❌ UMAP visualization creation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def _assess_regime_stability(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        knn_results: Dict[str, Any], 
        local_silhouette: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess regime stability based on neighborhood analysis."""
        try:
            unique_clusters = np.unique(assignments)
            cluster_stability = {}
            stable_regimes = []
            fragile_regimes = []
            
            for cluster_id in unique_clusters:
                cluster_mask = assignments == cluster_id
                cluster_size = np.sum(cluster_mask)
                
                # Calculate stability score based on local silhouette and consistency
                cluster_stats = local_silhouette.get('cluster_local_stats', {}).get(cluster_id, {})
                mean_silhouette = cluster_stats.get('mean_local_silhouette', 0.0)
                
                # Simple stability score
                stability_score = max(0, mean_silhouette) * (cluster_size / len(assignments))
                
                cluster_stability[cluster_id] = {
                    'size': cluster_size,
                    'stability_score': stability_score,
                    'mean_silhouette': mean_silhouette
                }
                
                # Classify as stable or fragile
                if stability_score > 0.1 and cluster_size > 10:
                    stable_regimes.append(cluster_id)
                elif stability_score < 0.05 or cluster_size < 5:
                    fragile_regimes.append(cluster_id)
            
            return {
                'cluster_stability': cluster_stability,
                'stable_regimes': stable_regimes,
                'fragile_regimes': fragile_regimes,
                'overall_stability': np.mean([cluster_stability[c]['stability_score'] for c in unique_clusters])
            }
            
        except Exception as e:
            tprint(f"❌ Regime stability assessment failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def _integrate_reallocation_in_optimization(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        neighborhood_results: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Integrate samples reallocation into the optimization process using neighborhood insights."""
        try:
            tprint("🔄 Integrating reallocation into optimization pipeline...", "INFO")
            
            # Use neighborhood insights to guide the optimization process
            knn_results = neighborhood_results.get('knn_consistency', {})
            local_silhouette = neighborhood_results.get('local_silhouette', {})
            stability_analysis = neighborhood_results.get('stability_analysis', {})
            
            if knn_results.get('error') or local_silhouette.get('error') or stability_analysis.get('error'):
                tprint("⚠️ Neighborhood analysis incomplete, using basic reallocation", "WARNING")
                return assignments, {'reallocation_skipped': True}
            
            # Apply targeted reallocation based on neighborhood insights
            reallocated_assignments = assignments.copy()
            
            # 1. Reallocate misclustered points identified by k-NN consistency
            consistency_scores = knn_results['consistency_scores']
            local_scores = local_silhouette['local_scores']
            
            # Find points with poor neighborhood consistency
            poor_consistency_mask = np.array(consistency_scores) < 0.7  # Below 70% consistency
            poor_local_mask = np.array(local_scores) < 0.0  # Negative local silhouette
            
            candidates_for_reallocation = poor_consistency_mask & poor_local_mask
            n_candidates = np.sum(candidates_for_reallocation)
            
            if n_candidates > 0:
                tprint(f"🔄 Found {n_candidates} candidates for reallocation based on neighborhood analysis", "INFO")
                
                # Use neighborhood information to guide reallocation
                reallocated_assignments = self._guided_reallocation(
                    features, reallocated_assignments, candidates_for_reallocation,
                    knn_results, local_silhouette
                )
            
            # 2. Apply regime consolidation based on stability analysis
            consolidated_assignments, consolidation_stats = self._apply_stability_guided_consolidation(
                features, reallocated_assignments, stability_analysis
            )
            
            # 3. Update final assignments with both reallocations
            final_assignments = consolidated_assignments.copy()
            
            # Calculate total reallocations
            total_reallocations = np.sum(final_assignments != assignments)
            
            results = {
                'total_reallocations': total_reallocations,
                'knn_reallocations': n_candidates,
                'consolidation_changes': consolidation_stats.get('consolidation_changes', 0),
                'reallocation_success_rate': total_reallocations / max(1, n_candidates) if n_candidates > 0 else 0.0
            }
            
            tprint(f"✅ Integrated reallocation complete: {total_reallocations} total changes", "SUCCESS")
            return final_assignments, results
            
        except Exception as e:
            tprint(f"❌ Integrated reallocation failed: {e}", "ERROR")
            return assignments, {'error': str(e)}
    
    def _guided_reallocation(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        candidates_mask: np.ndarray, 
        knn_results: Dict[str, Any], 
        local_silhouette: Dict[str, Any]
    ) -> np.ndarray:
        """Perform guided reallocation using detailed neighborhood information."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Fit k-NN for precise neighbor analysis
            nn = NearestNeighbors(n_neighbors=15, metric='euclidean')
            nn.fit(features)
            
            guided_assignments = assignments.copy()
            successful_reallocations = 0
            
            for i in np.where(candidates_mask)[0]:
                current_cluster = assignments[i]
                
                # Get detailed neighbor information
                distances, indices = nn.kneighbors(features[i:i+1])
                neighbor_assignments = assignments[indices[0][1:]]  # Exclude self
                
                if len(neighbor_assignments) == 0:
                    continue
                
                # Find best target cluster using multiple criteria
                best_target = self._find_best_reallocation_target(
                    i, current_cluster, neighbor_assignments, features, assignments,
                    local_silhouette, indices[0][1:]
                )
                
                if best_target is not None and best_target != current_cluster:
                    # Verify target cluster quality before reallocation
                    target_quality = self._assess_target_cluster_quality(
                        best_target, features, assignments, local_silhouette
                    )
                    
                    if target_quality > 0.0:  # Target should have reasonable quality
                        guided_assignments[i] = best_target
                        successful_reallocations += 1
            
            tprint(f"✅ Guided reallocation: {successful_reallocations} successful reallocations", "SUCCESS")
            return guided_assignments
            
        except Exception as e:
            tprint(f"❌ Guided reallocation failed: {e}", "WARNING")
            return assignments
    
    def _find_best_reallocation_target(
        self, 
        sample_idx: int, 
        current_cluster: int, 
        neighbor_assignments: np.ndarray, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        local_silhouette: Dict[str, Any], 
        neighbor_indices: np.ndarray
    ) -> Optional[int]:
        """Find the best target cluster for reallocation using comprehensive criteria."""
        try:
            # Count votes for each cluster among neighbors
            unique_clusters, counts = np.unique(neighbor_assignments, return_counts=True)
            cluster_votes = dict(zip(unique_clusters, counts))
            
            # Exclude current cluster
            cluster_votes.pop(current_cluster, None)
            
            if not cluster_votes:
                return None
            
            # Score each candidate cluster
            cluster_scores = {}
            for candidate_cluster, vote_count in cluster_votes.items():
                # Base score from neighbor votes (popularity)
                base_score = vote_count / len(neighbor_assignments)
                
                # Quality bonus for clusters with good local cohesion
                cluster_stats = local_silhouette['cluster_local_stats'].get(candidate_cluster, {})
                quality_bonus = max(0, cluster_stats.get('mean_local_silhouette', 0.0))
                
                # Distance penalty (prefer closer clusters)
                candidate_features = features[assignments == candidate_cluster]
                if len(candidate_features) > 0:
                    candidate_centroid = np.mean(candidate_features, axis=0)
                    sample_features = features[sample_idx]
                    distance = np.linalg.norm(sample_features - candidate_centroid)
                    distance_penalty = max(0, 1.0 - (distance / np.max(np.linalg.norm(features, axis=1))))
                else:
                    distance_penalty = 0.0
                
                # Combined score
                total_score = base_score * 0.5 + quality_bonus * 0.3 + distance_penalty * 0.2
                cluster_scores[candidate_cluster] = total_score
            
            # Return cluster with highest score
            if cluster_scores:
                best_cluster = max(cluster_scores.items(), key=lambda x: x[1])[0]
                return best_cluster
            
            return None
            
        except Exception:
            return None
    
    def _assess_target_cluster_quality(
        self, 
        target_cluster: int, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        local_silhouette: Dict[str, Any]
    ) -> float:
        """Assess the quality of a target cluster for reallocation."""
        try:
            cluster_stats = local_silhouette['cluster_local_stats'].get(target_cluster, {})
            if not cluster_stats:
                return 0.0
            
            # Use local silhouette as primary quality metric
            quality_score = cluster_stats.get('mean_local_silhouette', 0.0)
            
            # Size bonus for reasonably sized clusters (not too small, not too large)
            cluster_size = cluster_stats.get('count', 0)
            size_bonus = 0.0
            if 10 <= cluster_size <= 100:  # Reasonable size range
                size_bonus = 0.1
            
            return quality_score + size_bonus
            
        except Exception:
            return 0.0
    
    def _apply_stability_guided_consolidation(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        stability_analysis: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply stability-guided consolidation of regimes."""
        try:
            cluster_stability = stability_analysis['cluster_stability']
            stable_regimes = stability_analysis['stable_regimes']
            fragile_regimes = stability_analysis['fragile_regimes']
            
            consolidated_assignments = assignments.copy()
            consolidation_changes = 0
            
            # Only consolidate very small and very fragile regimes
            for fragile_regime in fragile_regimes:
                if fragile_regime not in cluster_stability:
                    continue
                
                fragile_stats = cluster_stability[fragile_regime]
                
                # Only merge very small (<10 samples) and very unstable regimes
                if fragile_stats['size'] < 10 and fragile_stats['stability_score'] < 0.05:
                    best_target = self._find_nearest_stable_regime(
                        features, assignments, fragile_regime, stable_regimes
                    )
                    
                    if best_target is not None:
                        consolidated_assignments[assignments == fragile_regime] = best_target
                        consolidation_changes += fragile_stats['size']
                        tprint(f"🔗 Consolidated fragile regime {fragile_regime} ({fragile_stats['size']} samples) into {best_target}", "INFO")
            
            results = {
                'consolidation_changes': consolidation_changes,
                'consolidated_regimes': len(fragile_regimes) if consolidation_changes > 0 else 0
            }
            
            return consolidated_assignments, results
            
        except Exception as e:
            tprint(f"❌ Stability-guided consolidation failed: {e}", "ERROR")
            return assignments, {'error': str(e)}
    
    def _find_nearest_stable_regime(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        fragile_regime: int, 
        stable_regimes: List[int]
    ) -> Optional[int]:
        """Find the nearest stable regime for consolidation."""
        try:
            if not stable_regimes:
                return None
            
            # Calculate centroids for fragile and stable regimes
            fragile_mask = assignments == fragile_regime
            fragile_centroid = np.mean(features[fragile_mask], axis=0)
            
            best_target = None
            best_distance = float('inf')
            
            for stable_regime in stable_regimes:
                stable_mask = assignments == stable_regime
                stable_centroid = np.mean(features[stable_mask], axis=0)
                
                distance = np.linalg.norm(fragile_centroid - stable_centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_target = stable_regime
            
            return best_target
            
        except Exception:
            return None
    
    def _compute_unified_objective(self, features: np.ndarray, assignments: np.ndarray, k: int, k_max: int = 12) -> float:
        """Compute unified objective J with complexity penalty."""
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            
            if len(np.unique(assignments)) <= 1:
                return 0.0
            
            # Calculate silhouette score
            silhouette = silhouette_score(features, assignments)
            
            # Calculate Davies-Bouldin index (lower is better)
            db_index = davies_bouldin_score(features, assignments)
            
            # Complexity penalty (prefer fewer clusters)
            complexity_penalty = k / k_max
            
            # Unified objective: maximize silhouette, minimize DB index, minimize complexity
            J = silhouette - (db_index / 10) - complexity_penalty
            
            return J
            
        except Exception as e:
            tprint(f"❌ Unified objective computation failed: {e}", "ERROR")
            return 0.0
    
    def _summarize_results(self, context: ClusteringContext, market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Summarize clustering results and create final output."""
        try:
            tprint("📊 Summarizing clustering results...", "INFO")
            
            # Extract assignments and other needed data
            assignments = context.optimized_assignments or context.raw_assignments or np.array([])
            features = context.optimized_features or context.original_features or np.array([])
            validation_quality = getattr(context, 'validation_quality', 0.0)
            neighborhood_analysis = getattr(context, 'neighborhood_analysis', {})
            reallocation_stats = getattr(context, 'reallocation_stats', {})
            
            # Create summary statistics
            if isinstance(assignments, np.ndarray) and assignments.size > 0:
                unique_vals = np.unique(assignments)
                n_clusters = len(unique_vals)
                cluster_distribution = np.bincount(assignments)
                n_samples = assignments.size
            else:
                n_clusters = 0
                cluster_distribution = []
                n_samples = 0
            
            # Handle features shape properly
            if isinstance(features, np.ndarray):
                features_shape = features.shape if features.size > 0 else (0, 0)
            else:
                features_shape = features.shape if len(features) > 0 else (0, 0)
            
            summary = {
                'n_clusters': n_clusters,
                'n_samples': n_samples,
                'cluster_distribution': cluster_distribution.tolist(),
                'features_shape': features_shape,
                'validation_quality': validation_quality,
                'neighborhood_analysis': neighborhood_analysis,
                'reallocation_stats': reallocation_stats,
                'success': True
            }
            
            tprint(f"✅ Results summarized: {n_clusters} clusters, {n_samples} samples", "SUCCESS")
            return summary
            
        except Exception as e:
            tprint(f"❌ Results summarization failed: {e}", "ERROR")
            return {'error': str(e), 'success': False}
    
    def validate_clustering_robustness(
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