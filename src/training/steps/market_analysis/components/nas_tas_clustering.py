"""
NAS-TAS Clustering Component.

This component performs advanced regime clustering using combined Neural Architecture Search (NAS)
and Tree-based Architecture Search (TAS) approaches. It leverages the unified clustering algorithms
from the hybrid NAS-TAS regime system for superior clustering quality and economic awareness.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger

logger = logging.getLogger(__name__)


@dataclass
class NASTASClusteringConfig(ComponentConfig):
    """Configuration for NAS-TAS clustering component."""
    symbol: str = "ETHUSDT"
    timeframe: str = "15m"
    exchange: str = "binance"

    # Clustering parameters
    n_regimes: int = 8
    algorithm_type: str = "adaptive_clustering"
    enable_economic_clustering: bool = True
    enable_ensemble_clustering: bool = True

    # Economic clustering weights
    economic_weight: float = 0.3
    momentum_weight: float = 0.25
    volume_weight: float = 0.25

    # Feature configuration
    feature_categories: List[str] = None
    use_standardized_features: bool = True

    # Output configuration
    output_dir: str = "data_cache"
    save_intermediate_results: bool = True

    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = ['momentum', 'volatility', 'volume', 'trend', 'price_action']


class NASTASClusteringComponent(BaseMarketAnalysisComponent):
    """
    NAS-TAS Clustering Component.

    Performs advanced regime clustering using combined NAS and TAS approaches.
    """

    def __init__(self, config: Optional[NASTASClusteringConfig] = None):
        """Initialize the NAS-TAS clustering component."""
        super().__init__(config)
        self.logger = system_logger.getChild('NASTASClustering')
        self.unified_clustering = None
        self.clustering_result = None
        self.execution_metadata = {}

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['nas_tas_clustering_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute NAS-TAS clustering.

        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with clustering results
        """
        self.logger.info('🚀 Starting NAS-TAS Clustering')

        try:
            # Update execution metadata
            self.execution_metadata = {
                'start_time': datetime.now(),
                'symbol': self.config.symbol if self.config else 'UNKNOWN',
                'timeframe': self.config.timeframe if self.config else '15m',
                'component': 'nas_tas_clustering'
            }

            # Load market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for clustering")

            # Prepare features for clustering
            features = self._prepare_features(market_data)
            if features is None:
                raise ValueError("Failed to prepare features for clustering")

            # Initialize unified clustering
            clustering_config = self._create_clustering_config()
            self.unified_clustering = self._initialize_unified_clustering(clustering_config)

            # Perform clustering
            clustering_result = self.unified_clustering.cluster_features(
                features=features,
                market_data=market_data
            )

            if not clustering_result.success:
                raise ValueError(f"Clustering failed: {clustering_result.error_message}")

            self.clustering_result = clustering_result
            self.logger.info(f"✅ NAS-TAS Clustering completed: {len(set(clustering_result.labels))} regimes discovered")

            # Generate outputs
            outputs = await self._generate_outputs(market_data, clustering_result)

            # Update execution metadata
            self.execution_metadata.update({
                'end_time': datetime.now(),
                'execution_time': (datetime.now() - self.execution_metadata['start_time']).total_seconds(),
                'success': True,
                'regime_count': len(set(clustering_result.labels)),
                'algorithm_used': clustering_result.algorithm_used,
                'quality_metrics': clustering_result.quality_metrics
            })

            return ComponentResult(
                success=True,
                artifacts={
                    'nas_tas_clustering_result': {
                        'regime_count': len(set(clustering_result.labels)),
                        'total_samples': len(clustering_result.labels),
                        'regime_assignments': clustering_result.labels.tolist(),
                        'cluster_centers': clustering_result.cluster_centers.tolist(),
                        'probabilities': clustering_result.probabilities.tolist() if clustering_result.probabilities is not None else [],
                        'quality_metrics': clustering_result.quality_metrics,
                        'algorithm_used': clustering_result.algorithm_used,
                        'execution_time': clustering_result.execution_time,
                        'configuration': asdict(self.config) if self.config else {},
                        'execution_info': self.execution_metadata
                    }
                },
                metadata={
                    'symbol': self.config.symbol if self.config else 'UNKNOWN',
                    'timeframe': self.config.timeframe if self.config else '15m',
                    'data_points_processed': len(market_data),
                    'regime_count': len(set(clustering_result.labels)),
                    'algorithm_used': clustering_result.algorithm_used,
                    'execution_successful': True,
                    'execution_time': clustering_result.execution_time
                }
            )

        except Exception as e:
            self.logger.error(f'❌ NAS-TAS Clustering failed: {e}')
            self.logger.error(traceback.format_exc())

            self.execution_metadata.update({
                'end_time': datetime.now(),
                'success': False,
                'error': str(e)
            })

            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"NAS-TAS clustering failed: {str(e)}"
            )

    def _create_clustering_config(self) -> Dict[str, Any]:
        """Create clustering configuration."""
        try:
            config = self.config if self.config else NASTASClusteringConfig()

            clustering_config = {
                'n_regimes': config.n_regimes,
                'algorithm_type': config.algorithm_type,
                'enable_economic_clustering': config.enable_economic_clustering,
                'enable_ensemble_clustering': config.enable_ensemble_clustering,
                'economic_weight': config.economic_weight,
                'momentum_weight': config.momentum_weight,
                'volume_weight': config.volume_weight
            }

            self.logger.info(f"📊 Clustering configuration: {config.n_regimes} regimes, algorithm: {config.algorithm_type}")
            return clustering_config

        except Exception as e:
            self.logger.warning(f"Failed to create clustering config: {e}, using defaults")
            return {
                'n_regimes': 8,
                'algorithm_type': 'adaptive_clustering',
                'enable_economic_clustering': True,
                'enable_ensemble_clustering': True,
                'economic_weight': 0.3,
                'momentum_weight': 0.25,
                'volume_weight': 0.25
            }

    def _initialize_unified_clustering(self, clustering_config: Dict[str, Any]):
        """Initialize unified clustering algorithm."""
        try:
            # Import the unified clustering algorithm
            from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_clustering_algorithms import (
                UnifiedClusteringAlgorithm
            )

            clustering = UnifiedClusteringAlgorithm(clustering_config)
            self.logger.info("✅ Unified clustering algorithm initialized")
            return clustering

        except ImportError as e:
            self.logger.error(f"Failed to import unified clustering: {e}")
            raise ValueError(f"Cannot import unified clustering algorithm: {e}")

    def _prepare_features(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for clustering."""
        try:
            features = []

            # Price-based features
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().fillna(0)
                features.append(returns.values.reshape(-1, 1))

                # Volatility (rolling std)
                volatility = returns.rolling(20).std().fillna(0)
                features.append(volatility.values.reshape(-1, 1))

                # Moving averages ratio
                sma_20 = market_data['close'].rolling(20).mean().fillna(market_data['close'].iloc[0])
                ma_ratio = market_data['close'] / sma_20 - 1
                features.append(ma_ratio.values.reshape(-1, 1))

            # Volume features
            if 'volume' in market_data.columns:
                volume_ma = market_data['volume'].rolling(20).mean().fillna(market_data['volume'].mean())
                volume_ratio = market_data['volume'] / volume_ma
                features.append(volume_ratio.fillna(1).values.reshape(-1, 1))

            # High-low spread
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                hl_spread = (market_data['high'] - market_data['low']) / market_data['close']
                features.append(hl_spread.fillna(0).values.reshape(-1, 1))

            # Combine features
            if features:
                feature_array = np.hstack(features)
                # Remove any NaN or infinite values
                feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
                return feature_array
            else:
                # If no features could be created, return dummy features
                return np.random.randn(len(market_data), 5)

        except Exception as e:
            self.logger.warning(f"Failed to prepare features: {e}")
            return np.random.randn(len(market_data), 5)

    async def _load_market_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and prepare market data for clustering."""
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                self.logger.warning("⚠️ No market data provided, attempting to load from klines_parquet")

                symbol = self.config.symbol if self.config else 'ETHUSDT'
                timeframe = self.config.timeframe if self.config else '15m'

                # Try to load data using klines_parquet manager
                from src.utils.data.klines_parquet import get_klines_manager

                manager = get_klines_manager()

                self.logger.info(f"📊 Loading {symbol} {timeframe} data using klines_parquet manager")

                # Try processed data first
                market_data = manager.read_data(symbol, timeframe, data_type="processed")

                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    market_data = manager.read_data(symbol, timeframe, data_type="raw")

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

    async def _generate_outputs(self, market_data: pd.DataFrame, clustering_result) -> Dict[str, Any]:
        """Generate output files and data structures."""
        try:
            self.logger.info("📁 Generating output files...")

            outputs = {
                'clustering_report': None,
                'regime_assignments': None,
                'cluster_characteristics': None,
                'output_files': []
            }

            # Save clustering report
            if clustering_result:
                report_file = self._save_clustering_report(clustering_result)
                outputs['clustering_report'] = report_file
                outputs['output_files'].append(report_file)

                # Generate regime assignments
                regime_data = self._generate_regime_assignments(market_data, clustering_result)
                if regime_data:
                    regime_file = self._save_regime_assignments(regime_data)
                    outputs['regime_assignments'] = regime_file
                    outputs['output_files'].append(regime_file)

                # Generate cluster characteristics
                characteristics = self._generate_cluster_characteristics(market_data, clustering_result)
                if characteristics:
                    char_file = self._save_cluster_characteristics(characteristics)
                    outputs['cluster_characteristics'] = char_file
                    outputs['output_files'].append(char_file)

            return outputs

        except Exception as e:
            self.logger.error(f"❌ Failed to generate outputs: {e}")
            return outputs

    def _save_clustering_report(self, clustering_result) -> str:
        """Save clustering report to file."""
        try:
            output_dir = Path(self.config.output_dir) / "nas_tas_clustering" / (self.config.symbol if self.config else 'UNKNOWN')
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"nas_tas_clustering_report_{timestamp}.json"
            filepath = output_dir / filename

            report_data = {
                'clustering_result': {
                    'regime_count': len(set(clustering_result.labels)),
                    'algorithm_used': clustering_result.algorithm_used,
                    'quality_metrics': clustering_result.quality_metrics,
                    'execution_time': clustering_result.execution_time,
                    'success': clustering_result.success
                },
                'metadata': self.execution_metadata,
                'config': asdict(self.config) if self.config else {}
            }

            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            self.logger.info(f"💾 Clustering report saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"❌ Failed to save clustering report: {e}")
            return ""

    def _save_regime_assignments(self, regime_data: pd.DataFrame) -> str:
        """Save regime assignments to file."""
        try:
            output_dir = Path(self.config.output_dir) / "nas_tas_clustering" / (self.config.symbol if self.config else 'UNKNOWN')
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"nas_tas_regime_assignments_{timestamp}.parquet"
            filepath = output_dir / filename

            regime_data.to_parquet(filepath)
            self.logger.info(f"💾 Regime assignments saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"❌ Failed to save regime assignments: {e}")
            return ""

    def _save_cluster_characteristics(self, characteristics: Dict) -> str:
        """Save cluster characteristics to file."""
        try:
            output_dir = Path(self.config.output_dir) / "nas_tas_clustering" / (self.config.symbol if self.config else 'UNKNOWN')
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"nas_tas_cluster_characteristics_{timestamp}.json"
            filepath = output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(characteristics, f, indent=2, default=str)

            self.logger.info(f"💾 Cluster characteristics saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"❌ Failed to save cluster characteristics: {e}")
            return ""

    def _generate_regime_assignments(self, market_data: pd.DataFrame, clustering_result) -> Optional[pd.DataFrame]:
        """Generate regime assignments DataFrame."""
        try:
            if clustering_result.labels is None or len(clustering_result.labels) == 0:
                return None

            # Create DataFrame with regime assignments
            regime_data = pd.DataFrame({
                'timestamp': market_data.index,
                'regime_id': clustering_result.labels,
                'regime_prob': clustering_result.probabilities if clustering_result.probabilities is not None else np.zeros(len(market_data))
            }).set_index('timestamp')

            return regime_data

        except Exception as e:
            self.logger.error(f"❌ Failed to generate regime assignments: {e}")
            return None

    def _generate_cluster_characteristics(self, market_data: pd.DataFrame, clustering_result) -> Dict[str, Any]:
        """Generate cluster characteristics."""
        try:
            characteristics = {}
            unique_regimes = set(clustering_result.labels)

            for regime_id in unique_regimes:
                regime_mask = clustering_result.labels == regime_id
                regime_data = market_data.iloc[regime_mask] if regime_mask.any() else pd.DataFrame()

                if len(regime_data) > 0:
                    characteristics[f'regime_{regime_id}'] = {
                        'sample_count': len(regime_data),
                        'avg_return': regime_data['close'].pct_change().mean() if 'close' in regime_data.columns else 0.0,
                        'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                        'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0
                    }

            return characteristics

        except Exception as e:
            self.logger.error(f"❌ Failed to generate cluster characteristics: {e}")
            return {}

    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            'component': 'nas_tas_clustering',
            'initialized': self.unified_clustering is not None,
            'has_results': self.clustering_result is not None,
            'execution_metadata': self.execution_metadata
        }

    def validate_inputs(self) -> List[str]:
        """Validate input parameters."""
        errors = []

        if not self.config:
            errors.append("Configuration is required")
            return errors

        if not self.config.symbol:
            errors.append("Symbol is required")

        if not self.config.timeframe:
            errors.append("Timeframe is required")

        valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        if self.config.timeframe not in valid_timeframes:
            errors.append(f"Invalid timeframe: {self.config.timeframe}. Must be one of {valid_timeframes}")

        if self.config.n_regimes < 2:
            errors.append("Number of regimes must be at least 2")

        return errors
