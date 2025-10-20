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
            
            # Create regime training datasets
            regime_datasets = self._create_regime_datasets(market_data, regime_labels, regime_features, config)
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(regime_labels, regime_probabilities, economic_profiles)
            
            # Store component state
            self.regime_data = market_data
            self.regime_features = regime_features
            self.regime_labels = regime_labels
            self.regime_probabilities = regime_probabilities
            self.economic_profiles = economic_profiles
            
            tprint(f"✅ Regime clustering processed: {regime_stats['n_regimes']} regimes, {regime_stats['n_samples']} samples", "SUCCESS")
            
            return {
                'success': True,
                'regime_labels': regime_labels,
                'regime_probabilities': regime_probabilities,
                'regime_features': regime_features,
                'regime_datasets': regime_datasets,
                'regime_stats': regime_stats,
                'economic_profiles': economic_profiles,
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
        """Generate regime-specific features."""
        try:
            tprint("🔧 Generating regime-specific features...", "INFO")
            
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
            
            tprint(f"✅ Generated {len(features)} regime-specific features", "SUCCESS")
            
            return {
                'features_df': features_df,
                'feature_names': list(features.keys()),
                'n_features': len(features)
            }
            
        except Exception as e:
            tprint(f"❌ Feature generation failed: {e}", "ERROR")
            return {
                'features_df': pd.DataFrame(),
                'feature_names': [],
                'n_features': 0
            }
    
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


# Register the step
def register_regime_clustering_step():
    """Register the regime clustering step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("regime_clustering", RegimeClusteringComponent)
    tprint("✅ Regime clustering step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_clustering_step()