"""
Economic Evaluator

Provides economic significance analysis for regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time

from ..config.hybrid_config import HybridRegimeConfig


class EconomicEvaluator:
    """
    Economic evaluator that provides economic significance analysis.
    
    This component:
    1. Evaluates economic significance of regimes
    2. Analyzes volatility, trend strength, and market efficiency
    3. Provides liquidity regime detection
    4. Assesses correlation structure analysis
    """
    
    def __init__(self, config: HybridRegimeConfig):
        """
        Initialize Economic Evaluator.
        
        Args:
            config: Hybrid regime configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info("✅ Economic Evaluator initialized")
        self.logger.info(f"🏛️ Economic modeling: {config.economic_modeling_enabled}")
        self.logger.info(f"📊 Economic threshold: {config.economic_significance_threshold}")
    
    def evaluate_economic_significance(self, 
                                      market_data: Union[pd.DataFrame, np.ndarray],
                                      regime_predictions: np.ndarray,
                                      regime_probabilities: Optional[np.ndarray] = None,
                                      timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate economic significance of regimes.
        
        Args:
            market_data: Market data
            regime_predictions: Regime predictions
            regime_probabilities: Regime probabilities
            timestamps: Optional timestamps
            
        Returns:
            Dictionary with economic evaluation results
        """
        start_time = time.time()
        self.logger.info("🏛️ Evaluating economic significance")
        
        try:
            # Prepare data
            prepared_data = self._prepare_data(market_data, timestamps)
            
            # Analyze volatility regimes
            volatility_analysis = self._analyze_volatility_regimes(
                prepared_data, regime_predictions
            )
            
            # Analyze trend strength
            trend_analysis = self._analyze_trend_strength(
                prepared_data, regime_predictions
            )
            
            # Evaluate market efficiency
            efficiency_analysis = self._evaluate_market_efficiency(
                prepared_data, regime_predictions
            )
            
            # Detect liquidity regimes
            liquidity_analysis = self._detect_liquidity_regimes(
                prepared_data, regime_predictions
            )
            
            # Analyze correlation structure
            correlation_analysis = self._analyze_correlation_structure(
                prepared_data, regime_predictions
            )
            
            # Calculate economic significance scores
            significance_scores = self._calculate_economic_significance_scores(
                volatility_analysis, trend_analysis, efficiency_analysis,
                liquidity_analysis, correlation_analysis
            )
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"✅ Economic evaluation completed in {execution_time:.2f}s")
            self.logger.info(f"📊 Average significance: {np.mean(significance_scores):.3f}")
            
            return {
                'success': True,
                'significance_scores': significance_scores,
                'volatility_analysis': volatility_analysis,
                'trend_analysis': trend_analysis,
                'efficiency_analysis': efficiency_analysis,
                'liquidity_analysis': liquidity_analysis,
                'correlation_analysis': correlation_analysis,
                'execution_time': execution_time,
                'metadata': {
                    'n_samples': len(regime_predictions),
                    'n_regimes': len(set(regime_predictions)),
                    'economic_threshold': self.config.economic_significance_threshold
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Economic evaluation failed: {e}")
            
            return {
                'success': False,
                'significance_scores': np.array([]),
                'volatility_analysis': {},
                'trend_analysis': {},
                'efficiency_analysis': {},
                'liquidity_analysis': {},
                'correlation_analysis': {},
                'execution_time': execution_time,
                'error_message': str(e)
            }
    
    def _prepare_data(self, 
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Prepare data for economic analysis."""
        if isinstance(market_data, pd.DataFrame):
            data = market_data
        else:
            if len(market_data.shape) == 1:
                market_data = market_data.reshape(-1, 1)
            data = pd.DataFrame(market_data, columns=[f"feature_{i}" for i in range(market_data.shape[1])])
        
        if timestamps is not None:
            data['timestamp'] = timestamps
        else:
            data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(data), freq='15T')
        
        return {
            'data': data,
            'n_samples': len(data),
            'n_features': len(data.columns)
        }
    
    def _analyze_volatility_regimes(self, 
                                    prepared_data: Dict[str, Any],
                                    regime_predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility regimes."""
        data = prepared_data['data']
        
        # Calculate volatility for each regime
        regime_volatilities = {}
        for regime_id in np.unique(regime_predictions):
            regime_mask = regime_predictions == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) > 1:
                # Calculate returns
                if 'close' in regime_data.columns:
                    returns = regime_data['close'].pct_change().dropna()
                    volatility = returns.std()
                else:
                    # Use first numeric column
                    numeric_cols = regime_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        returns = regime_data[numeric_cols[0]].pct_change().dropna()
                        volatility = returns.std()
                    else:
                        volatility = 0.0
                
                regime_volatilities[regime_id] = {
                    'volatility': float(volatility),
                    'n_samples': len(regime_data),
                    'regime_type': 'high_volatility' if volatility > 0.02 else 'low_volatility'
                }
            else:
                regime_volatilities[regime_id] = {
                    'volatility': 0.0,
                    'n_samples': len(regime_data),
                    'regime_type': 'unknown'
                }
        
        return regime_volatilities
    
    def _analyze_trend_strength(self, 
                                prepared_data: Dict[str, Any],
                                regime_predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze trend strength for each regime."""
        data = prepared_data['data']
        
        # Calculate trend strength for each regime
        regime_trends = {}
        for regime_id in np.unique(regime_predictions):
            regime_mask = regime_predictions == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) > 1:
                # Calculate trend strength
                if 'close' in regime_data.columns:
                    prices = regime_data['close']
                    # Linear regression slope as trend strength
                    x = np.arange(len(prices))
                    slope = np.polyfit(x, prices, 1)[0]
                    trend_strength = abs(slope) / (prices.mean() + 1e-8)
                else:
                    # Use first numeric column
                    numeric_cols = regime_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        prices = regime_data[numeric_cols[0]]
                        x = np.arange(len(prices))
                        slope = np.polyfit(x, prices, 1)[0]
                        trend_strength = abs(slope) / (prices.mean() + 1e-8)
                    else:
                        trend_strength = 0.0
                
                regime_trends[regime_id] = {
                    'trend_strength': float(trend_strength),
                    'n_samples': len(regime_data),
                    'trend_direction': 'up' if slope > 0 else 'down' if slope < 0 else 'sideways'
                }
            else:
                regime_trends[regime_id] = {
                    'trend_strength': 0.0,
                    'n_samples': len(regime_data),
                    'trend_direction': 'unknown'
                }
        
        return regime_trends
    
    def _evaluate_market_efficiency(self, 
                                    prepared_data: Dict[str, Any],
                                    regime_predictions: np.ndarray) -> Dict[str, Any]:
        """Evaluate market efficiency for each regime."""
        data = prepared_data['data']
        
        # Calculate market efficiency for each regime
        regime_efficiency = {}
        for regime_id in np.unique(regime_predictions):
            regime_mask = regime_predictions == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) > 1:
                # Calculate efficiency metrics
                if 'close' in regime_data.columns:
                    prices = regime_data['close']
                    returns = prices.pct_change().dropna()
                    
                    # Autocorrelation as inefficiency measure
                    if len(returns) > 1:
                        autocorr = returns.autocorr(lag=1)
                        efficiency = 1 - abs(autocorr) if not np.isnan(autocorr) else 1.0
                    else:
                        efficiency = 1.0
                else:
                    efficiency = 0.5  # Default efficiency
                
                regime_efficiency[regime_id] = {
                    'efficiency': float(efficiency),
                    'n_samples': len(regime_data),
                    'efficiency_level': 'high' if efficiency > 0.8 else 'medium' if efficiency > 0.5 else 'low'
                }
            else:
                regime_efficiency[regime_id] = {
                    'efficiency': 0.5,
                    'n_samples': len(regime_data),
                    'efficiency_level': 'unknown'
                }
        
        return regime_efficiency
    
    def _detect_liquidity_regimes(self, 
                                  prepared_data: Dict[str, Any],
                                  regime_predictions: np.ndarray) -> Dict[str, Any]:
        """Detect liquidity regimes."""
        data = prepared_data['data']
        
        # Calculate liquidity for each regime
        regime_liquidity = {}
        for regime_id in np.unique(regime_predictions):
            regime_mask = regime_predictions == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) > 1:
                # Calculate liquidity metrics
                if 'volume' in regime_data.columns:
                    volume = regime_data['volume']
                    avg_volume = volume.mean()
                    volume_volatility = volume.std() / (avg_volume + 1e-8)
                    liquidity = 1.0 / (volume_volatility + 1e-8)
                else:
                    liquidity = 0.5  # Default liquidity
                
                regime_liquidity[regime_id] = {
                    'liquidity': float(liquidity),
                    'n_samples': len(regime_data),
                    'liquidity_level': 'high' if liquidity > 2.0 else 'medium' if liquidity > 1.0 else 'low'
                }
            else:
                regime_liquidity[regime_id] = {
                    'liquidity': 0.5,
                    'n_samples': len(regime_data),
                    'liquidity_level': 'unknown'
                }
        
        return regime_liquidity
    
    def _analyze_correlation_structure(self, 
                                       prepared_data: Dict[str, Any],
                                       regime_predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze correlation structure for each regime."""
        data = prepared_data['data']
        
        # Calculate correlation structure for each regime
        regime_correlations = {}
        for regime_id in np.unique(regime_predictions):
            regime_mask = regime_predictions == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) > 1:
                # Calculate correlation matrix
                numeric_data = regime_data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr()
                    avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                    correlation_strength = abs(avg_correlation)
                else:
                    correlation_strength = 0.0
                
                regime_correlations[regime_id] = {
                    'correlation_strength': float(correlation_strength),
                    'n_samples': len(regime_data),
                    'correlation_level': 'high' if correlation_strength > 0.7 else 'medium' if correlation_strength > 0.3 else 'low'
                }
            else:
                regime_correlations[regime_id] = {
                    'correlation_strength': 0.0,
                    'n_samples': len(regime_data),
                    'correlation_level': 'unknown'
                }
        
        return regime_correlations
    
    def _calculate_economic_significance_scores(self, 
                                                volatility_analysis: Dict[str, Any],
                                                trend_analysis: Dict[str, Any],
                                                efficiency_analysis: Dict[str, Any],
                                                liquidity_analysis: Dict[str, Any],
                                                correlation_analysis: Dict[str, Any]) -> np.ndarray:
        """Calculate economic significance scores."""
        # Combine all analysis results into significance scores
        significance_scores = np.zeros(len(volatility_analysis))
        
        for i, regime_id in enumerate(volatility_analysis.keys()):
            # Weight different factors
            volatility_score = min(1.0, volatility_analysis[regime_id]['volatility'] * 50)
            trend_score = min(1.0, trend_analysis[regime_id]['trend_strength'] * 100)
            efficiency_score = efficiency_analysis[regime_id]['efficiency']
            liquidity_score = min(1.0, liquidity_analysis[regime_id]['liquidity'] / 2.0)
            correlation_score = correlation_analysis[regime_id]['correlation_strength']
            
            # Combined significance score
            significance = (0.3 * volatility_score + 
                           0.2 * trend_score + 
                           0.2 * efficiency_score + 
                           0.15 * liquidity_score + 
                           0.15 * correlation_score)
            
            significance_scores[i] = significance
        
        return significance_scores