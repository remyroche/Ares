#!/usr/bin/env python3
"""
HMM Discovery Integration

This module handles the integration between HMM discovery outputs and regime clustering,
providing seamless data transformation and validation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HMMDiscoveryData:
    """Container for HMM discovery output data."""
    
    regime_features: pd.DataFrame
    regime_characteristics: Dict[str, Any]
    metadata: Dict[str, Any]
    original_data_points: int
    
    @property
    def regime_count(self) -> int:
        return len(self.regime_features)
    
    @property
    def total_samples(self) -> int:
        return self.regime_features['sample_count'].sum()

class HMMDiscoveryIntegration:
    """
    Integration class for HMM discovery outputs and regime clustering.
    
    This class handles:
    - Loading HMM discovery results
    - Validating and transforming data
    - Creating regime feature matrices
    - Providing data quality metrics
    """
    
    def __init__(self):
        """Initialize the HMM discovery integration."""
        self.logger = logger.getChild("HMMDiscoveryIntegration")
    
    def load_hmm_discovery_results(self, results_file: Union[str, Path]) -> HMMDiscoveryData:
        """
        Load HMM discovery results from JSON file.
        
        Args:
            results_file: Path to HMM discovery results JSON file
            
        Returns:
            HMMDiscoveryData with processed regime information
        """
        try:
            results_path = Path(results_file)
            if not results_path.exists():
                raise FileNotFoundError(f"HMM discovery results file not found: {results_path}")
            
            self.logger.info(f"Loading HMM discovery results from {results_path}")
            
            with open(results_path, 'r') as f:
                results_data = json.load(f)
            
            # Extract regime data
            regime_data = self._extract_regime_data(results_data)
            
            # Create regime features DataFrame
            regime_features = self._create_regime_features_df(regime_data)
            
            # Extract metadata
            metadata = self._extract_metadata(results_data)
            
            # Create HMMDiscoveryData object
            hmm_data = HMMDiscoveryData(
                regime_features=regime_features,
                regime_characteristics=results_data.get('artifacts', {}).get('hmm_regime_discovery_result', {}),
                metadata=metadata,
                original_data_points=metadata.get('data_points_processed', 0)
            )
            
            self.logger.info(f"Loaded {hmm_data.regime_count} regimes with {hmm_data.total_samples:,} total samples")
            
            return hmm_data
            
        except Exception as e:
            self.logger.error(f"Failed to load HMM discovery results: {e}")
            raise
    
    def _extract_regime_data(self, results_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract regime data from HMM discovery results."""
        
        # Navigate to regime characteristics
        artifacts = results_data.get('artifacts', {})
        hmm_result = artifacts.get('hmm_regime_discovery_result', {})
        regime_characteristics = hmm_result.get('regime_characteristics', {})
        
        if not regime_characteristics:
            raise ValueError("No regime characteristics found in HMM discovery results")
        
        regime_data = []
        
        for regime_name, regime_info in regime_characteristics.items():
            if not isinstance(regime_info, dict):
                continue
            
            # Extract feature means
            feature_means = regime_info.get('feature_means', {})
            if not feature_means:
                continue
            
            # Extract sample count
            sample_count = regime_info.get('sample_count', 0)
            if sample_count <= 0:
                continue
            
            # Parse regime name to extract M, V, Vol values if available
            momentum_level, volatility_level, volume_level = self._parse_regime_name(regime_name)
            
            regime_entry = {
                'regime_name': regime_name,
                'momentum_mean': feature_means.get('momentum_20', 0.0),
                'volatility_mean': feature_means.get('volatility_20', 0.0),
                'volume_mean': feature_means.get('volume_ratio_192m', 1.0),
                'trend_mean': feature_means.get('trend_score', 0.0),
                'sample_count': sample_count,
                'momentum_level': momentum_level,
                'volatility_level': volatility_level,
                'volume_level': volume_level
            }
            
            regime_data.append(regime_entry)
        
        if not regime_data:
            raise ValueError("No valid regime data extracted from HMM discovery results")
        
        self.logger.info(f"Extracted {len(regime_data)} valid regimes")
        return regime_data
    
    def _parse_regime_name(self, regime_name: str) -> tuple[int, int, int]:
        """Parse regime name to extract M, V, Vol levels."""
        
        # Expected format: "regime_M1_V2_Vol3" or similar
        momentum_level = 0
        volatility_level = 0
        volume_level = 0
        
        try:
            # Extract momentum level (M followed by number)
            momentum_match = re.search(r'M(\d+)', regime_name)
            if momentum_match:
                momentum_level = int(momentum_match.group(1))
            
            # Extract volatility level (V followed by number)
            volatility_match = re.search(r'V(\d+)', regime_name)
            if volatility_match:
                volatility_level = int(volatility_match.group(1))
            
            # Extract volume level (Vol followed by number)
            volume_match = re.search(r'Vol(\d+)', regime_name)
            if volume_match:
                volume_level = int(volume_match.group(1))
        
        except (ValueError, AttributeError):
            # If parsing fails, use default values
            pass
        
        return momentum_level, volatility_level, volume_level
    
    def _create_regime_features_df(self, regime_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a DataFrame from regime data."""
        
        df = pd.DataFrame(regime_data)
        
        # Set regime name as index
        df.set_index('regime_name', inplace=True)
        
        # Validate required columns
        required_columns = ['momentum_mean', 'volatility_mean', 'volume_mean', 'trend_mean', 'sample_count']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in regime features: {missing_columns}")
        
        # Validate data types and ranges
        self._validate_regime_features(df)
        
        return df
    
    def _validate_regime_features(self, df: pd.DataFrame) -> None:
        """Validate regime features DataFrame."""
        
        # Check for missing values
        if df.isnull().any().any():
            missing_info = df.isnull().sum()
            self.logger.warning(f"Found missing values in regime features: {missing_info.to_dict()}")
        
        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            self.logger.warning("Found infinite values in regime features")
        
        # Check sample counts
        if (df['sample_count'] <= 0).any():
            invalid_counts = df[df['sample_count'] <= 0]['sample_count'].to_dict()
            self.logger.warning(f"Found invalid sample counts: {invalid_counts}")
        
        # Check feature ranges (basic sanity checks)
        feature_ranges = {
            'momentum_mean': (-2.0, 2.0),  # Reasonable range for normalized momentum
            'volatility_mean': (-3.0, 3.0),  # Reasonable range for normalized volatility
            'volume_mean': (0.0, 10.0),  # Volume ratio should be positive
            'trend_mean': (-1.0, 1.0)  # Trend score should be in [-1, 1]
        }
        
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in df.columns:
                out_of_range = df[(df[feature] < min_val) | (df[feature] > max_val)]
                if not out_of_range.empty:
                    self.logger.warning(f"Found {len(out_of_range)} regimes with {feature} out of range [{min_val}, {max_val}]")
    
    def _extract_metadata(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from HMM discovery results."""
        
        metadata = results_data.get('metadata', {})
        
        # Extract key information
        return {
            'symbol': metadata.get('symbol', 'UNKNOWN'),
            'timeframe': metadata.get('timeframe', 'UNKNOWN'),
            'data_points_processed': metadata.get('data_points_processed', 0),
            'regime_count': metadata.get('regime_count', 0),
            'optimization_mode': metadata.get('optimization_mode', 'UNKNOWN'),
            'timestamp': results_data.get('timestamp', 'UNKNOWN'),
            'status': results_data.get('status', 'UNKNOWN')
        }
    
    def validate_for_clustering(self, hmm_data: HMMDiscoveryData) -> Dict[str, Any]:
        """
        Validate HMM discovery data for clustering suitability.
        
        Args:
            hmm_data: HMMDiscoveryData object to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'recommendations': [],
            'statistics': {}
        }
        
        df = hmm_data.regime_features
        
        # Basic statistics
        validation_results['statistics'] = {
            'regime_count': len(df),
            'total_samples': int(df['sample_count'].sum()),
            'avg_samples_per_regime': float(df['sample_count'].mean()),
            'min_samples_per_regime': int(df['sample_count'].min()),
            'max_samples_per_regime': int(df['sample_count'].max()),
            'sample_count_std': float(df['sample_count'].std())
        }
        
        # Check regime count
        if len(df) < 50:
            validation_results['warnings'].append(f"Low regime count: {len(df)} regimes may not provide enough granularity")
            validation_results['recommendations'].append("Consider reducing HMM discovery parameters for more regimes")
        elif len(df) > 5000:
            validation_results['warnings'].append(f"High regime count: {len(df)} regimes may be too granular")
            validation_results['recommendations'].append("Consider increasing HMM discovery parameters for fewer regimes")
        
        # Check sample distribution
        sample_counts = df['sample_count'].values
        small_regimes = (sample_counts < 100).sum()
        large_regimes = (sample_counts > 10000).sum()
        
        if small_regimes > len(df) * 0.5:
            validation_results['warnings'].append(f"Many small regimes: {small_regimes} regimes have < 100 samples")
            validation_results['recommendations'].append("Consider adjusting HMM parameters to reduce regime fragmentation")
        
        if large_regimes > 0:
            validation_results['warnings'].append(f"Some large regimes: {large_regimes} regimes have > 10,000 samples")
            validation_results['recommendations'].append("Consider if these large regimes should be split further")
        
        # Check feature distributions
        for feature in ['momentum_mean', 'volatility_mean', 'volume_mean', 'trend_mean']:
            if feature in df.columns:
                feature_std = df[feature].std()
                if feature_std < 0.01:
                    validation_results['warnings'].append(f"Low variance in {feature}: {feature_std:.4f}")
                    validation_results['recommendations'].append(f"Consider if {feature} provides sufficient discrimination")
        
        # Check for clustering suitability
        total_samples = df['sample_count'].sum()
        target_cluster_size = total_samples * 0.05  # 5% of total samples
        
        regimes_suitable_for_clustering = (sample_counts >= target_cluster_size * 0.1).sum()
        
        if regimes_suitable_for_clustering < 20:
            validation_results['warnings'].append(f"Few regimes suitable for clustering: {regimes_suitable_for_clustering} regimes")
            validation_results['recommendations'].append("Consider adjusting HMM discovery parameters or clustering targets")
        
        # Overall validation
        if validation_results['warnings']:
            validation_results['is_valid'] = False
        
        return validation_results
    
    def create_clustering_input(self, hmm_data: HMMDiscoveryData) -> pd.DataFrame:
        """
        Create input DataFrame for regime clustering.
        
        Args:
            hmm_data: HMMDiscoveryData object
            
        Returns:
            DataFrame ready for regime clustering
        """
        
        df = hmm_data.regime_features.copy()
        
        # Ensure required columns are present
        required_columns = ['momentum_mean', 'volatility_mean', 'volume_mean', 'trend_mean', 'sample_count']
        
        # Add any missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col == 'sample_count':
                    df[col] = 1  # Default to 1 sample if missing
                else:
                    df[col] = 0.0  # Default to 0.0 for feature columns
        
        # Sort by sample count (descending) for better clustering
        df = df.sort_values('sample_count', ascending=False)
        
        self.logger.info(f"Created clustering input with {len(df)} regimes")
        self.logger.info(f"Feature columns: {[col for col in df.columns if col.endswith('_mean')]}")
        self.logger.info(f"Sample count range: {df['sample_count'].min()} - {df['sample_count'].max()}")
        
        return df
    
    def load_latest_hmm_results(self, outcomes_dir: Union[str, Path], 
                              symbol: str, timeframe: str) -> HMMDiscoveryData:
        """
        Load the latest HMM discovery results for a given symbol and timeframe.
        
        Args:
            outcomes_dir: Directory containing HMM discovery outcome files
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '15m')
            
        Returns:
            HMMDiscoveryData with the latest results
        """
        
        outcomes_path = Path(outcomes_dir)
        if not outcomes_path.exists():
            raise FileNotFoundError(f"Outcomes directory not found: {outcomes_path}")
        
        # Find HMM discovery result files
        pattern = f"market_analysis_hmm_regime_discovery_outcome_*.json"
        hmm_files = list(outcomes_path.glob(pattern))
        
        if not hmm_files:
            raise FileNotFoundError(f"No HMM discovery result files found in {outcomes_path}")
        
        # Filter by symbol and timeframe, and sort by timestamp
        matching_files = []
        for file_path in hmm_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                file_symbol = metadata.get('symbol', '').upper()
                file_timeframe = metadata.get('timeframe', '')
                
                if file_symbol == symbol.upper() and file_timeframe == timeframe:
                    timestamp_str = data.get('timestamp', '')
                    matching_files.append((timestamp_str, file_path))
            
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Skipping invalid file {file_path}: {e}")
                continue
        
        if not matching_files:
            raise FileNotFoundError(f"No HMM discovery results found for {symbol} {timeframe}")
        
        # Sort by timestamp and get the latest
        matching_files.sort(key=lambda x: x[0], reverse=True)
        latest_file = matching_files[0][1]
        
        self.logger.info(f"Loading latest HMM discovery results from {latest_file}")
        
        return self.load_hmm_discovery_results(latest_file)


def create_hmm_integration() -> HMMDiscoveryIntegration:
    """Create and return a new HMMDiscoveryIntegration instance."""
    return HMMDiscoveryIntegration()