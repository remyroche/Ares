"""
Shared regime characteristics utilities for NAS-TAS regime detection.

This module provides common regime characteristics calculation functionality that
eliminates redundancy between NAS and TAS components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from src.utils.tprint import tprint, tprint_debug, tprint_success, tprint_warning, tprint_error


@dataclass
class CharacteristicsConfig:
    """Configuration for regime characteristics calculation."""
    # Feature calculation parameters
    include_price_features: bool = True
    include_volume_features: bool = True
    include_volatility_features: bool = True
    include_momentum_features: bool = True
    
    # Statistical parameters
    min_regime_size: int = 5
    handle_small_regimes: bool = True
    
    # Output configuration
    include_hybrid_specific: bool = True
    include_statistical_summaries: bool = True


class CharacteristicsGenerator:
    """Centralized regime characteristics generator for NAS-TAS components."""
    
    def __init__(self, config: Optional[CharacteristicsConfig] = None, verbose: bool = False):
        """Initialize characteristics generator.
        
        Args:
            config: Characteristics configuration
            verbose: Whether to enable verbose logging
        """
        self.config = config or CharacteristicsConfig()
        self.verbose = verbose
    
    def create_regime_characteristics(
        self,
        market_data: pd.DataFrame,
        regime_predictions: List[int],
        hybrid_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create regime characteristics for clustering and analysis.
        
        Args:
            market_data: Market data DataFrame
            regime_predictions: List of regime predictions
            hybrid_result: Optional hybrid analysis result for enhanced characteristics
            
        Returns:
            Dictionary containing regime characteristics
        """
        if self.verbose:
            tprint("🔬 [CHARACTERISTICS] Creating regime characteristics", color="blue")
        
        try:
            regime_characteristics = {}
            unique_regimes = set(regime_predictions)
            
            if self.verbose:
                tprint(f"🎯 [CHARACTERISTICS] Processing {len(unique_regimes)} unique regimes", color="cyan")
            
            for regime_id in unique_regimes:
                regime_mask = [i for i, r in enumerate(regime_predictions) if r == regime_id]
                regime_data = market_data.iloc[regime_mask] if regime_mask else pd.DataFrame()
                
                if len(regime_data) > 0:
                    if self.verbose:
                        tprint(f"📊 [CHARACTERISTICS] Processing regime {regime_id}: {len(regime_data)} samples", color="yellow")
                    
                    characteristics = self._calculate_regime_characteristics(regime_id, regime_data, hybrid_result)
                    regime_characteristics[f'regime_{regime_id}'] = characteristics
                else:
                    if self.verbose:
                        tprint(f"⚠️ [CHARACTERISTICS] Regime {regime_id} has no data samples", color="yellow")
            
            if self.verbose:
                tprint(f"✅ [CHARACTERISTICS] Created characteristics for {len(regime_characteristics)} regimes", color="green")
            
            return regime_characteristics
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ [CHARACTERISTICS] Regime characteristics creation failed: {e}")
            raise ValueError(f"Regime characteristics creation failed: {e}")
    
    def _calculate_regime_characteristics(
        self,
        regime_id: int,
        regime_data: pd.DataFrame,
        hybrid_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate characteristics for a single regime."""
        characteristics = {}
        
        # Basic regime information
        characteristics['regime_id'] = regime_id
        characteristics['sample_count'] = len(regime_data)
        
        # Price features
        if self.config.include_price_features and 'close' in regime_data.columns:
            characteristics.update(self._calculate_price_features(regime_data))
        
        # Volume features
        if self.config.include_volume_features and 'volume' in regime_data.columns:
            characteristics.update(self._calculate_volume_features(regime_data))
        
        # Volatility features
        if self.config.include_volatility_features and 'close' in regime_data.columns:
            characteristics.update(self._calculate_volatility_features(regime_data))
        
        # Momentum features
        if self.config.include_momentum_features and 'close' in regime_data.columns:
            characteristics.update(self._calculate_momentum_features(regime_data))
        
        # Statistical summaries
        if self.config.include_statistical_summaries:
            characteristics.update(self._calculate_statistical_summaries(regime_data))
        
        # Hybrid-specific characteristics
        if self.config.include_hybrid_specific and hybrid_result:
            characteristics.update(self._calculate_hybrid_characteristics(regime_id, hybrid_result))
        
        return characteristics
    
    def _calculate_price_features(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price-related features."""
        features = {}
        
        try:
            # Average return
            if 'close' in regime_data.columns and len(regime_data) > 1:
                returns = regime_data['close'].pct_change().dropna()
                features['avg_return'] = float(returns.mean()) if len(returns) > 0 else 0.0
                features['return_std'] = float(returns.std()) if len(returns) > 0 else 0.0
            
            # Price range
            if all(col in regime_data.columns for col in ['high', 'low', 'close']):
                features['hl_spread'] = float(((regime_data['high'] - regime_data['low']) / regime_data['close']).mean())
                features['hl_spread_std'] = float(((regime_data['high'] - regime_data['low']) / regime_data['close']).std())
            
            # Open-close spread
            if all(col in regime_data.columns for col in ['open', 'close']):
                features['oc_spread'] = float(((regime_data['close'] - regime_data['open']) / regime_data['open']).mean())
                features['oc_spread_std'] = float(((regime_data['close'] - regime_data['open']) / regime_data['open']).std())
            
            # Body size (candle body relative to total range)
            if all(col in regime_data.columns for col in ['open', 'close', 'high', 'low']):
                body_size = abs(regime_data['close'] - regime_data['open']) / (regime_data['high'] - regime_data['low'])
                features['body_size'] = float(body_size.mean())
                features['body_size_std'] = float(body_size.std())
        
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ [CHARACTERISTICS] Price feature calculation failed for regime: {e}")
        
        return features
    
    def _calculate_volume_features(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-related features."""
        features = {}
        
        try:
            if 'volume' in regime_data.columns:
                features['avg_volume'] = float(regime_data['volume'].mean())
                features['volume_std'] = float(regime_data['volume'].std())
                features['volume_min'] = float(regime_data['volume'].min())
                features['volume_max'] = float(regime_data['volume'].max())
                
                # Volume change
                if len(regime_data) > 1:
                    volume_change = regime_data['volume'].pct_change().dropna()
                    features['volume_change_mean'] = float(volume_change.mean()) if len(volume_change) > 0 else 0.0
                    features['volume_change_std'] = float(volume_change.std()) if len(volume_change) > 0 else 0.0
        
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ [CHARACTERISTICS] Volume feature calculation failed for regime: {e}")
        
        return features
    
    def _calculate_volatility_features(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility-related features."""
        features = {}
        
        try:
            if 'close' in regime_data.columns and len(regime_data) > 1:
                returns = regime_data['close'].pct_change().dropna()
                if len(returns) > 0:
                    features['volatility'] = float(returns.std())
                    features['volatility_mean'] = float(abs(returns).mean())
                    features['volatility_max'] = float(abs(returns).max())
        
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ [CHARACTERISTICS] Volatility feature calculation failed for regime: {e}")
        
        return features
    
    def _calculate_momentum_features(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum-related features."""
        features = {}
        
        try:
            if 'close' in regime_data.columns and len(regime_data) > 1:
                # Price momentum
                if len(regime_data) >= 5:
                    momentum_5 = (regime_data['close'].iloc[-1] - regime_data['close'].iloc[0]) / regime_data['close'].iloc[0]
                    features['momentum_5'] = float(momentum_5)
                
                if len(regime_data) >= 10:
                    momentum_10 = (regime_data['close'].iloc[-1] - regime_data['close'].iloc[-10]) / regime_data['close'].iloc[-10]
                    features['momentum_10'] = float(momentum_10)
                
                # RSI-like momentum
                returns = regime_data['close'].pct_change().dropna()
                if len(returns) > 0:
                    positive_returns = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
                    negative_returns = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
                    if positive_returns + negative_returns > 0:
                        features['rsi_momentum'] = float(positive_returns / (positive_returns + negative_returns))
        
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ [CHARACTERISTICS] Momentum feature calculation failed for regime: {e}")
        
        return features
    
    def _calculate_statistical_summaries(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical summaries."""
        features = {}
        
        try:
            # Feature means and standard deviations
            feature_means = {}
            feature_stds = {}
            
            numeric_columns = regime_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in regime_data.columns:
                    feature_means[f'{col}_mean'] = float(regime_data[col].mean())
                    feature_stds[f'{col}_std'] = float(regime_data[col].std())
            
            features['feature_means'] = feature_means
            features['feature_stds'] = feature_stds
            
            # Regime duration
            features['duration'] = len(regime_data)
            features['duration_percentage'] = len(regime_data) / len(regime_data) * 100  # Will be updated with total data length
        
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ [CHARACTERISTICS] Statistical summary calculation failed for regime: {e}")
        
        return features
    
    def _calculate_hybrid_characteristics(
        self,
        regime_id: int,
        hybrid_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate hybrid-specific characteristics."""
        features = {}
        
        try:
            # Consensus and disagreement metrics
            consensus_metrics = hybrid_result.get('consensus_metrics', {})
            features['consensus_strength'] = consensus_metrics.get('consensus_score', 0.0)
            
            # Economic significance
            economic_scores = hybrid_result.get('economic_significance_scores', [])
            if economic_scores:
                features['economic_significance'] = economic_scores[0] if len(economic_scores) > 0 else 0.7
            else:
                features['economic_significance'] = 0.7
            
            # Trading viability
            trading_scores = hybrid_result.get('trading_viability_scores', [])
            if trading_scores:
                features['trading_viability'] = trading_scores[0] if len(trading_scores) > 0 else 0.6
            else:
                features['trading_viability'] = 0.6
            
            # Regime stability
            stability_scores = hybrid_result.get('regime_stability_scores', [])
            if stability_scores:
                features['regime_stability'] = stability_scores[0] if len(stability_scores) > 0 else 0.8
            else:
                features['regime_stability'] = 0.8
            
            # Combination strategy
            features['combination_strategy'] = hybrid_result.get('combination_strategy', 'ensemble')
        
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ [CHARACTERISTICS] Hybrid characteristics calculation failed for regime {regime_id}: {e}")
        
        return features
    
    def generate_cluster_characteristics(
        self,
        market_data: pd.DataFrame,
        cluster_assignments: List[int],
        cluster_centers: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate cluster characteristics for clustering analysis.
        
        Args:
            market_data: Market data DataFrame
            cluster_assignments: Cluster assignments
            cluster_centers: Optional cluster centers
            
        Returns:
            Dictionary containing cluster characteristics
        """
        if self.verbose:
            tprint("🔬 [CHARACTERISTICS] Generating cluster characteristics", color="blue")
        
        try:
            # Use the same logic as regime characteristics
            cluster_characteristics = self.create_regime_characteristics(
                market_data, cluster_assignments
            )
            
            # Add cluster-specific information
            if cluster_centers is not None:
                # Handle both numpy arrays and lists
                if hasattr(cluster_centers, 'tolist'):
                    cluster_characteristics['cluster_centers'] = cluster_centers.tolist()
                else:
                    cluster_characteristics['cluster_centers'] = cluster_centers
                cluster_characteristics['n_clusters'] = len(cluster_centers)
            else:
                unique_clusters = set(cluster_assignments)
                cluster_characteristics['n_clusters'] = len(unique_clusters)
            
            if self.verbose:
                tprint(f"✅ [CHARACTERISTICS] Generated cluster characteristics for {cluster_characteristics['n_clusters']} clusters", color="green")
            
            return cluster_characteristics
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ [CHARACTERISTICS] Cluster characteristics generation failed: {e}")
            raise ValueError(f"Cluster characteristics generation failed: {e}")


# Convenience functions for backward compatibility
def create_regime_characteristics(
    market_data: pd.DataFrame,
    regime_predictions: List[int],
    hybrid_result: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Create regime characteristics for clustering and analysis."""
    generator = CharacteristicsGenerator(verbose=verbose)
    return generator.create_regime_characteristics(market_data, regime_predictions, hybrid_result)


def generate_cluster_characteristics(
    market_data: pd.DataFrame,
    cluster_assignments: List[int],
    cluster_centers: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Generate cluster characteristics for clustering analysis."""
    generator = CharacteristicsGenerator(verbose=verbose)
    return generator.generate_cluster_characteristics(market_data, cluster_assignments, cluster_centers)