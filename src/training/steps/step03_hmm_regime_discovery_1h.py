#!/usr/bin/env python3
"""Enhanced Step 3: HMM Regime Discovery using 1h timeframe only.

This module performs Hidden Markov Model (HMM) regime discovery exclusively on 1h data
and maps the results back to the trading timeframe.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.decorators import handles_errors, traced
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards

logger = system_logger.getChild("HMMRegimeDiscovery1H")


class HMMRegimeDiscovery1H:
    """HMM Regime Discovery using only 1h timeframe data."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("HMMRegimeDiscovery1H")
        self.standards = pipeline_standards
        
        # Force 1h for regime analysis
        self.regime_timeframe = '1h'
        self.min_hours_required = 720  # 30 days of 1h data
        
        # HMM parameters optimized for 1h
        self.hmm_params = {
            'n_components': 4,  # bull, bear, sideways, transition
            'covariance_type': 'full',
            'n_iter': 100,
            'random_state': 42
        }
        
        # Regime mapping configuration
        self.regime_names = {
            0: 'bear',
            1: 'sideways_bear',
            2: 'sideways_bull', 
            3: 'bull'
        }
        
    @handles_errors(fallback=False)
    @traced(span_name="hmm_regime_discovery_1h")
    async def execute(self, symbol: str, exchange: str, trading_timeframe: str, 
                     data_dir: str) -> Dict[str, Any]:
        """Execute HMM regime discovery using 1h data."""
        
        self.logger.info(f"Starting 1h regime discovery for {symbol}")
        self.logger.info(f"Trading timeframe: {trading_timeframe}, Regime analysis: {self.regime_timeframe}")
        
        try:
            # Step 1: Load or resample to 1h data
            data_1h = await self._load_or_create_1h_data(symbol, exchange, data_dir)
            
            if len(data_1h) < self.min_hours_required:
                raise ValueError(f"Insufficient 1h data: {len(data_1h)} hours (min: {self.min_hours_required})")
            
            # Step 2: Prepare features for HMM
            features = await self._prepare_hmm_features(data_1h)
            
            # Step 3: Fit HMM model
            hmm_model, regime_labels_1h = await self._fit_hmm_model(features)
            
            # Step 4: Calculate regime statistics
            regime_stats = await self._calculate_regime_statistics(data_1h, regime_labels_1h)
            
            # Step 5: Map regimes to trading timeframe if different
            if trading_timeframe != self.regime_timeframe:
                regime_labels_trading = await self._map_regimes_to_timeframe(
                    data_1h, regime_labels_1h, symbol, exchange, trading_timeframe, data_dir
                )
            else:
                regime_labels_trading = regime_labels_1h
            
            # Step 6: Calculate transition probabilities
            transition_matrix = self._calculate_transition_matrix(regime_labels_1h)
            
            # Step 7: Save results
            results = {
                'regime_model': hmm_model,
                'regime_labels_1h': regime_labels_1h,
                'regime_labels_trading': regime_labels_trading,
                'regime_timeframe': self.regime_timeframe,
                'trading_timeframe': trading_timeframe,
                'regime_statistics': regime_stats,
                'transition_matrix': transition_matrix,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._save_results(results, symbol, exchange, data_dir)
            
            self.logger.info("✅ 1h regime discovery completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Regime discovery failed: {e}")
            raise
    
    async def _load_or_create_1h_data(self, symbol: str, exchange: str, 
                                     data_dir: str) -> pd.DataFrame:
        """Load existing 1h data or resample from lower timeframe."""
        
        # Try to load existing 1h data
        path_1h = Path(data_dir) / f"{exchange}_{symbol}_1h_unified.parquet"
        
        if path_1h.exists():
            self.logger.info(f"Loading existing 1h data from {path_1h}")
            data_1h = pd.read_parquet(path_1h)
            data_1h = self.standards.standardize_timestamp(data_1h, 'timestamp')
            return data_1h
        
        # Otherwise, resample from lower timeframe
        self.logger.info("1h data not found, resampling from lower timeframe...")
        
        # Find available timeframes
        available_files = list(Path(data_dir).glob(f"{exchange}_{symbol}_*_unified.parquet"))
        
        if not available_files:
            raise FileNotFoundError(f"No unified data found for {symbol}")
        
        # Load the smallest available timeframe
        smallest_tf_file = min(available_files, key=lambda x: self._timeframe_to_minutes(x.stem.split('_')[2]))
        self.logger.info(f"Resampling from {smallest_tf_file}")
        
        data = pd.read_parquet(smallest_tf_file)
        data = self.standards.standardize_timestamp(data, 'timestamp')
        
        # Resample to 1h
        data_1h = self._resample_to_1h(data)
        
        # Save resampled data
        data_1h.to_parquet(path_1h)
        self.logger.info(f"Saved resampled 1h data to {path_1h}")
        
        return data_1h
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        multipliers = {'m': 1, 'h': 60, 'd': 1440}
        
        for suffix, multiplier in multipliers.items():
            if timeframe.endswith(suffix):
                return int(timeframe[:-1]) * multiplier
        
        return 60  # Default to 1h
    
    def _resample_to_1h(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample OHLCV data to 1h."""
        
        # Ensure datetime index
        if 'timestamp' in data.columns:
            data = data.set_index('timestamp')
        
        # Resample rules
        resample_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Apply resampling
        data_1h = data.resample('1h').agg(resample_rules).dropna()
        
        # Reset index to have timestamp as column
        data_1h = data_1h.reset_index()
        
        return data_1h
    
    async def _prepare_hmm_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM training."""
        
        features_dict = {}
        
        # Price features
        features_dict['returns'] = data['close'].pct_change()
        features_dict['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Volatility features (optimized for 1h)
        features_dict['volatility_10'] = features_dict['returns'].rolling(10).std()
        features_dict['volatility_24'] = features_dict['returns'].rolling(24).std()
        
        # Volume features
        features_dict['volume_ratio'] = data['volume'] / data['volume'].rolling(24).mean()
        
        # Price position features
        features_dict['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        features_dict['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Trend features for 1h
        features_dict['sma_ratio_24'] = data['close'] / data['close'].rolling(24).mean()
        features_dict['sma_ratio_168'] = data['close'] / data['close'].rolling(168).mean()  # 1 week
        
        # Create feature matrix
        feature_df = pd.DataFrame(features_dict).dropna()
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_df)
        
        return features_scaled
    
    async def _fit_hmm_model(self, features: np.ndarray) -> Tuple[Any, np.ndarray]:
        """Fit HMM model to features."""
        
        try:
            from hmmlearn import hmm
        except ImportError:
            self.logger.error("hmmlearn not installed. Please install with: pip install hmmlearn")
            raise
        
        # Initialize HMM
        model = hmm.GaussianHMM(**self.hmm_params)
        
        # Fit model
        self.logger.info("Fitting HMM model...")
        model.fit(features)
        
        # Predict regimes
        regime_labels = model.predict(features)
        
        # Sort regimes by average returns to ensure consistent naming
        regime_returns = {}
        returns = features[:, 0]  # First feature is returns
        
        for regime in range(self.hmm_params['n_components']):
            mask = regime_labels == regime
            regime_returns[regime] = returns[mask].mean()
        
        # Create mapping from old to new regime numbers (sorted by returns)
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
        # Remap regime labels
        regime_labels_mapped = np.array([regime_mapping[label] for label in regime_labels])
        
        return model, regime_labels_mapped
    
    async def _calculate_regime_statistics(self, data: pd.DataFrame, 
                                         regime_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for each regime."""
        
        # Add regime labels to data
        data_with_regimes = data.copy()
        data_with_regimes['regime'] = regime_labels[:len(data)]
        data_with_regimes['returns'] = data_with_regimes['close'].pct_change()
        
        stats = {}
        
        for regime_num, regime_name in self.regime_names.items():
            regime_data = data_with_regimes[data_with_regimes['regime'] == regime_num]
            
            if len(regime_data) == 0:
                continue
            
            stats[regime_name] = {
                'count': len(regime_data),
                'percentage': len(regime_data) / len(data_with_regimes) * 100,
                'avg_return': regime_data['returns'].mean(),
                'volatility': regime_data['returns'].std(),
                'sharpe': regime_data['returns'].mean() / regime_data['returns'].std() * np.sqrt(252 * 24),
                'avg_volume': regime_data['volume'].mean(),
                'duration_stats': self._calculate_duration_stats(data_with_regimes, regime_num)
            }
        
        return stats
    
    def _calculate_duration_stats(self, data: pd.DataFrame, regime: int) -> Dict[str, float]:
        """Calculate duration statistics for a regime."""
        
        # Find regime sequences
        regime_mask = data['regime'] == regime
        regime_changes = regime_mask != regime_mask.shift(1)
        regime_groups = regime_changes.cumsum()
        
        # Calculate durations
        durations = []
        for group in regime_groups[regime_mask].unique():
            group_data = data[(regime_groups == group) & regime_mask]
            duration_hours = len(group_data)
            durations.append(duration_hours)
        
        if durations:
            return {
                'avg_duration_hours': np.mean(durations),
                'min_duration_hours': np.min(durations),
                'max_duration_hours': np.max(durations),
                'std_duration_hours': np.std(durations)
            }
        else:
            return {}
    
    async def _map_regimes_to_timeframe(self, data_1h: pd.DataFrame, regime_labels_1h: np.ndarray,
                                       symbol: str, exchange: str, target_timeframe: str,
                                       data_dir: str) -> np.ndarray:
        """Map 1h regime labels to target trading timeframe."""
        
        self.logger.info(f"Mapping regimes from 1h to {target_timeframe}")
        
        # Create regime dataframe with timestamps
        regime_df = pd.DataFrame({
            'timestamp': data_1h['timestamp'],
            'regime': regime_labels_1h[:len(data_1h)]
        })
        
        # Load target timeframe data
        target_path = Path(data_dir) / f"{exchange}_{symbol}_{target_timeframe}_unified.parquet"
        
        if not target_path.exists():
            raise FileNotFoundError(f"Target timeframe data not found: {target_path}")
        
        target_data = pd.read_parquet(target_path)
        target_data = self.standards.standardize_timestamp(target_data, 'timestamp')
        
        # Merge using backward fill (use most recent 1h regime)
        merged = pd.merge_asof(
            target_data[['timestamp']].sort_values('timestamp'),
            regime_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        # Fill any NaN values with the most common regime
        mode_regime = regime_df['regime'].mode().iloc[0]
        merged['regime'] = merged['regime'].fillna(mode_regime)
        
        return merged['regime'].values
    
    def _calculate_transition_matrix(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime transition probability matrix."""
        
        n_regimes = len(self.regime_names)
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(regime_labels) - 1):
            from_regime = regime_labels[i]
            to_regime = regime_labels[i + 1]
            transition_counts[from_regime, to_regime] += 1
        
        # Convert to probabilities
        transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        
        return transition_matrix
    
    async def _save_results(self, results: Dict[str, Any], symbol: str, 
                           exchange: str, data_dir: str) -> None:
        """Save regime discovery results."""
        
        output_dir = Path(data_dir) / "regime_analysis"
        ensure_directory(output_dir)
        
        # Save regime labels
        np.save(
            output_dir / f"{exchange}_{symbol}_regime_labels_1h.npy",
            results['regime_labels_1h']
        )
        
        np.save(
            output_dir / f"{exchange}_{symbol}_regime_labels_{results['trading_timeframe']}.npy",
            results['regime_labels_trading']
        )
        
        # Save statistics and configuration
        save_dict = {
            'regime_statistics': results['regime_statistics'],
            'transition_matrix': results['transition_matrix'].tolist(),
            'regime_names': self.regime_names,
            'hmm_params': self.hmm_params,
            'timestamp': results['timestamp']
        }
        
        safe_json_dump(
            save_dict,
            output_dir / f"{exchange}_{symbol}_regime_analysis.json"
        )
        
        self.logger.info(f"Saved regime analysis results to {output_dir}")


# Integration with existing step
async def run_enhanced_regime_discovery(symbol: str, exchange: str, timeframe: str,
                                      data_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run enhanced 1h-only regime discovery."""
    
    # Initialize 1h regime discovery
    regime_discovery = HMMRegimeDiscovery1H(config)
    
    # Execute regime discovery
    results = await regime_discovery.execute(symbol, exchange, timeframe, data_dir)
    
    return results


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '5m',  # Trading timeframe
            'data_dir': 'data_cache'
        }
        
        results = await run_enhanced_regime_discovery(
            config['symbol'],
            config['exchange'],
            config['timeframe'],
            config['data_dir'],
            config
        )
        
        print(f"Regime discovery completed")
        print(f"Regime statistics: {results['regime_statistics']}")
    
    asyncio.run(main())