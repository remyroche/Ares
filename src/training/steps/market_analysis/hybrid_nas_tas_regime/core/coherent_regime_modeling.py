"""
Coherent Regime Modeling for Hybrid NAS-TAS System

This module provides coherent regime modeling with:
- Economic regime analysis with significance scoring
- Financial regime analysis with trading viability assessment
- Micro-regime detection for subtle market changes
- Regime stability and transition analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

@dataclass
class CoherentRegimeResult:
    """Result from coherent regime modeling."""
    macro_regimes: Dict[str, Any]
    micro_regimes: Dict[str, Any]
    economic_analysis: Dict[str, Any]
    financial_analysis: Dict[str, Any]
    stability_analysis: Dict[str, Any]
    transition_analysis: Dict[str, Any]
    regime_hierarchy: Dict[str, Any]
    modeling_timestamp: str
    metadata: Dict[str, Any]

@dataclass
class MicroRegime:
    """Micro-regime definition."""
    micro_regime_id: int
    macro_regime_id: int
    description: str
    characteristics: Dict[str, float]
    significance_score: float
    stability_score: float
    transition_patterns: List[str]

class CoherentRegimeModeler:
    """
    Coherent regime modeling with economic and financial analysis.

    Provides comprehensive regime analysis including:
    - Macro and micro regime identification
    - Economic significance evaluation
    - Financial viability assessment
    - Stability and transition analysis
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize coherent regime modeler."""
        tprint_info("🚀 Initializing Coherent Regime Modeler")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration parameters
        tprint_debug("⚙️ Setting configuration parameters...")
        self.micro_regime_threshold = config.get('micro_regime_threshold', 0.3)
        self.stability_period = config.get('stability_period', 50)
        self.transition_smoothness = config.get('transition_smoothness', 0.8)
        tprint_success("✅ Configuration parameters set")

        tprint_success("✅ Coherent Regime Modeler initialized")
        self.logger.info("✅ Coherent Regime Modeler initialized")

    def model_regimes(self,
                     market_data: pd.DataFrame,
                     regime_labels: np.ndarray,
                     regime_probabilities: np.ndarray) -> CoherentRegimeResult:
        """
        Perform coherent regime modeling.

        Args:
            market_data: Market data
            regime_labels: Regime predictions
            regime_probabilities: Regime probabilities

        Returns:
            CoherentRegimeResult with comprehensive modeling results
        """
        try:
            tprint_info("🏗️ Starting coherent regime modeling...")
            tprint_debug(f"Market data shape: {market_data.shape}")
            tprint_debug(f"Regime labels shape: {regime_labels.shape}")
            tprint_debug(f"Regime probabilities shape: {regime_probabilities.shape}")
            self.logger.info("🏗️ Starting coherent regime modeling...")

            # Identify macro regimes
            tprint_debug("🔍 Identifying macro regimes...")
            macro_regimes = self._identify_macro_regimes(market_data, regime_labels)
            tprint_success(f"✅ Macro regimes identified: {len(macro_regimes)}")

            # Detect micro regimes
            tprint_debug("🔍 Detecting micro regimes...")
            micro_regimes = self._detect_micro_regimes(market_data, regime_labels, macro_regimes)
            tprint_success(f"✅ Micro regimes detected: {len(micro_regimes)}")

            # Perform economic analysis
            tprint_debug("💰 Performing economic analysis...")
            economic_analysis = self._economic_regime_analysis(market_data, regime_labels, macro_regimes)
            tprint_success("✅ Economic analysis completed")

            # Perform financial analysis
            financial_analysis = self._financial_regime_analysis(market_data, regime_labels, macro_regimes)

            # Analyze stability
            stability_analysis = self._regime_stability_analysis(regime_labels, regime_probabilities)

            # Analyze transitions
            transition_analysis = self._regime_transition_analysis(regime_labels, regime_probabilities)

            # Build regime hierarchy
            regime_hierarchy = self._build_regime_hierarchy(macro_regimes, micro_regimes)

            return CoherentRegimeResult(
                macro_regimes=macro_regimes,
                micro_regimes=micro_regimes,
                economic_analysis=economic_analysis,
                financial_analysis=financial_analysis,
                stability_analysis=stability_analysis,
                transition_analysis=transition_analysis,
                regime_hierarchy=regime_hierarchy,
                modeling_timestamp=datetime.now().isoformat(),
                metadata={
                    'n_macro_regimes': len(macro_regimes),
                    'n_micro_regimes': len(micro_regimes),
                    'total_regimes': len(macro_regimes) + len(micro_regimes),
                    'modeling_method': 'coherent_hybrid'
                }
            )

        except Exception as e:
            self.logger.error(f"Coherent regime modeling failed: {e}")
            raise

    def _identify_macro_regimes(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Identify macro regimes from clustering results."""
        try:
            self.logger.info("🔍 Identifying macro regimes...")

            macro_regimes = {}
            unique_labels = sorted(set(regime_labels))

            for regime_id in unique_labels:
                regime_mask = regime_labels == regime_id
                regime_data = market_data[regime_mask]

                if len(regime_data) < 10:  # Skip very small regimes
                    continue

                # Calculate regime characteristics
                characteristics = self._calculate_regime_characteristics(regime_data, regime_id)

                # Calculate economic significance
                economic_significance = self._calculate_economic_significance(regime_data, characteristics)

                # Calculate financial viability
                financial_viability = self._calculate_financial_viability(regime_data, characteristics)

                macro_regimes[f"regime_{regime_id}"] = {
                    'regime_id': regime_id,
                    'size': len(regime_data),
                    'percentage': len(regime_data) / len(market_data),
                    'characteristics': characteristics,
                    'economic_significance': economic_significance,
                    'financial_viability': financial_viability,
                    'description': self._generate_regime_description(characteristics, economic_significance)
                }

            self.logger.info(f"   Identified {len(macro_regimes)} macro regimes")
            return macro_regimes

        except Exception as e:
            self.logger.error(f"Macro regime identification failed: {e}")
            return {}

    def _detect_micro_regimes(self,
                             market_data: pd.DataFrame,
                             regime_labels: np.ndarray,
                             macro_regimes: Dict[str, Any]) -> Dict[str, Any]:
        """Detect micro-regimes within macro regimes."""
        try:
            self.logger.info("🔍 Detecting micro regimes...")

            micro_regimes = {}

            for macro_regime_name, macro_regime in macro_regimes.items():
                macro_id = macro_regime['regime_id']
                regime_mask = regime_labels == macro_id
                regime_data = market_data[regime_mask]

                # Look for sub-patterns within macro regime
                micro_patterns = self._find_micro_patterns(regime_data, macro_regime)

                for i, pattern in enumerate(micro_patterns):
                    micro_regime_id = f"{macro_id}.{i+1}"

                    micro_regimes[f"micro_regime_{micro_regime_id}"] = {
                        'micro_regime_id': micro_regime_id,
                        'macro_regime_id': macro_id,
                        'pattern_data': pattern,
                        'significance_score': pattern.get('significance', 0.5),
                        'stability_score': pattern.get('stability', 0.5),
                        'characteristics': pattern.get('characteristics', {}),
                        'description': pattern.get('description', f"Micro-pattern in regime {macro_id}")
                    }

            self.logger.info(f"   Detected {len(micro_regimes)} micro regimes")
            return micro_regimes

        except Exception as e:
            self.logger.error(f"Micro regime detection failed: {e}")
            return {}

    def _find_micro_patterns(self, regime_data: pd.DataFrame, macro_regime: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find micro-patterns within a macro regime."""
        try:
            patterns = []

            # Use rolling window analysis to find sub-patterns
            window_size = min(20, len(regime_data) // 3)

            if window_size < 5:
                return patterns

            # Calculate rolling characteristics
            price_series = regime_data['close'].values
            volume_series = regime_data.get('volume', np.ones(len(regime_data))).values

            for i in range(0, len(regime_data) - window_size, window_size // 2):
                window_data = regime_data.iloc[i:i+window_size]

                if len(window_data) < window_size * 0.8:
                    continue

                # Calculate window characteristics
                window_characteristics = self._calculate_regime_characteristics(window_data, 0)

                # Calculate significance of this window
                significance = self._calculate_window_significance(
                    window_characteristics, macro_regime['characteristics']
                )

                if significance > self.micro_regime_threshold:
                    patterns.append({
                        'start_index': i,
                        'end_index': i + window_size,
                        'characteristics': window_characteristics,
                        'significance': significance,
                        'stability': self._calculate_window_stability(window_data),
                        'description': self._generate_micro_regime_description(window_characteristics)
                    })

            return patterns

        except Exception as e:
            self.logger.warning(f"Micro pattern finding failed: {e}")
            return []

    def _calculate_regime_characteristics(self, regime_data: pd.DataFrame, regime_id: int) -> Dict[str, float]:
        """Calculate characteristics of a regime."""
        try:
            characteristics = {}

            # Price characteristics
            close_prices = regime_data['close'].values
            returns = np.diff(close_prices, prepend=close_prices[0])

            characteristics['mean_return'] = np.mean(returns)
            characteristics['volatility'] = np.std(returns)
            characteristics['skewness'] = pd.Series(returns).skew()
            characteristics['kurtosis'] = pd.Series(returns).kurtosis()

            # Trend characteristics
            if len(close_prices) > 10:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(close_prices)), close_prices)
                trend_strength = abs(r_value)
                characteristics['trend_strength'] = trend_strength

            return characteristics

        except Exception as e:
            self.logger.warning(f"Regime characteristics extraction failed: {e}")
            return {}

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

except ImportError:
    
    cp = None

    def _calculate_economic_significance(self, regime_data: pd.DataFrame, characteristics: Dict[str, float]) -> float:
        """Calculate economic significance of a regime."""
        try:
            significance = 0.0
            factors = []

            # Volatility significance (deviation from market average)
            volatility = characteristics.get('volatility', 0.01)
            if volatility > 0.03:  # High volatility
                factors.append(0.8)
            elif volatility < 0.01:  # Low volatility
                factors.append(0.6)
            else:
                factors.append(0.5)

            # Trend significance
            trend_strength = abs(characteristics.get('trend_r_squared', 0))
            if trend_strength > 0.3:
                factors.append(0.9)
            elif trend_strength > 0.1:
                factors.append(0.7)
            else:
                factors.append(0.4)

            # Volume profile significance
            volume_volatility = characteristics.get('volume_volatility', 1.0)
            if volume_volatility > 1.5:  # High volume volatility
                factors.append(0.7)
            elif volume_volatility < 0.5:  # Low volume volatility
                factors.append(0.8)
            else:
                factors.append(0.5)

            # Market efficiency significance
            autocorrelation = abs(characteristics.get('autocorrelation', 0))
            if autocorrelation > 0.3:  # Inefficient market (predictable)
                factors.append(0.8)
            elif autocorrelation < 0.1:  # Efficient market
                factors.append(0.6)
            else:
                factors.append(0.5)

            # Combine factors
            if factors:
                significance = np.mean(factors)

            return min(significance, 1.0)

        except Exception as e:
            self.logger.warning(f"Economic significance calculation failed: {e}")
            return 0.5

    def _calculate_financial_viability(self, regime_data: pd.DataFrame, characteristics: Dict[str, float]) -> float:
        """Calculate financial viability of a regime."""
        try:
            viability = 0.0
            factors = []

            # Calculate returns for viability assessment
            close_prices = regime_data['close'].values
            returns = np.diff(close_prices, prepend=close_prices[0])

            if len(returns) < 10:
                return 0.5

            # Sharpe ratio factor
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0

            if sharpe_ratio > 1.0:
                factors.append(0.9)
            elif sharpe_ratio > 0.5:
                factors.append(0.7)
            elif sharpe_ratio > 0:
                factors.append(0.5)
            else:
                factors.append(0.2)

            # Maximum drawdown factor
            cumulative = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = peak - cumulative
            max_drawdown = np.max(drawdown) / (peak[-1] if peak[-1] > 0 else 1)

            if max_drawdown < 0.05:  # Low drawdown
                factors.append(0.9)
            elif max_drawdown < 0.15:
                factors.append(0.7)
            elif max_drawdown < 0.25:
                factors.append(0.5)
            else:
                factors.append(0.2)

            # Win rate factor
            win_rate = np.sum(returns > 0) / len(returns)
            if win_rate > 0.6:
                factors.append(0.8)
            elif win_rate > 0.5:
                factors.append(0.6)
            else:
                factors.append(0.3)

            # Liquidity factor
            avg_spread = characteristics.get('avg_spread', 0.01)
            if avg_spread < 0.005:  # High liquidity
                factors.append(0.9)
            elif avg_spread < 0.02:
                factors.append(0.7)
            else:
                factors.append(0.4)

            # Combine factors
            if factors:
                viability = np.mean(factors)

            return min(viability, 1.0)

        except Exception as e:
            self.logger.warning(f"Financial viability calculation failed: {e}")
            return 0.5

    def _calculate_window_significance(self, window_chars: Dict[str, float], macro_chars: Dict[str, float]) -> float:
        """Calculate significance of a window compared to macro regime."""
        try:
            significance = 0.0
            comparisons = []

            # Compare key characteristics
            key_metrics = ['volatility', 'trend_r_squared', 'autocorrelation', 'avg_spread']

            for metric in key_metrics:
                if metric in window_chars and metric in macro_chars:
                    window_val = window_chars[metric]
                    macro_val = macro_chars[metric]

                    if macro_val != 0:
                        deviation = abs(window_val - macro_val) / abs(macro_val)
                        significance += min(deviation, 1.0) / len(key_metrics)

            return significance

        except Exception as e:
            self.logger.warning(f"Window significance calculation failed: {e}")
            return 0.5

    def _calculate_window_stability(self, window_data: pd.DataFrame) -> float:
        """Calculate stability of a window."""
        try:
            # Measure consistency within the window
            returns = window_data['close'].pct_change().dropna()

            if len(returns) < 5:
                return 0.5

            # Lower standard deviation indicates higher stability
            stability = 1.0 / (1.0 + np.std(returns))

            return min(stability, 1.0)

        except Exception as e:
            self.logger.warning(f"Window stability calculation failed: {e}")
            return 0.5

    def _generate_regime_description(self, characteristics: Dict[str, float], economic_significance: float) -> str:
        """Generate human-readable regime description."""
        try:
            descriptions = []

            # Volatility description
            volatility = characteristics.get('volatility', 0.01)
            if volatility > 0.03:
                descriptions.append("high volatility")
            elif volatility < 0.01:
                descriptions.append("low volatility")
            else:
                descriptions.append("moderate volatility")

            # Trend description
            trend_r2 = characteristics.get('trend_r_squared', 0)
            trend_slope = characteristics.get('trend_slope', 0)

            if trend_r2 > 0.3:
                if trend_slope > 0:
                    descriptions.append("strong upward trend")
                else:
                    descriptions.append("strong downward trend")
            elif trend_r2 > 0.1:
                if trend_slope > 0:
                    descriptions.append("moderate upward trend")
                else:
                    descriptions.append("moderate downward trend")
            else:
                descriptions.append("sideways/no trend")

            # Market efficiency description
            autocorrelation = abs(characteristics.get('autocorrelation', 0))
            if autocorrelation > 0.3:
                descriptions.append("inefficient/predictable")
            else:
                descriptions.append("efficient/random")

            # Combine descriptions
            if descriptions:
                return "Regime with " + ", ".join(descriptions)
            else:
                return "Mixed characteristics regime"

        except Exception as e:
            self.logger.warning(f"Regime description generation failed: {e}")
            return "Unknown regime characteristics"

    def _generate_micro_regime_description(self, characteristics: Dict[str, float]) -> str:
        """Generate micro-regime description."""
        try:
            # Focus on distinctive characteristics
            key_features = []

            volatility = characteristics.get('volatility', 0.01)
            if volatility > 0.05:
                key_features.append("extreme volatility")
            elif volatility < 0.005:
                key_features.append("calm period")

            trend_r2 = characteristics.get('trend_r_squared', 0)
            if trend_r2 > 0.5:
                key_features.append("strong directional movement")
            elif trend_r2 < 0.1:
                key_features.append("consolidation phase")

            autocorrelation = abs(characteristics.get('autocorrelation', 0))
            if autocorrelation > 0.4:
                key_features.append("predictable pattern")
            elif autocorrelation < 0.05:
                key_features.append("random movement")

            if key_features:
                return "Micro-regime: " + ", ".join(key_features)
            else:
                return "Micro-regime with mixed characteristics"

        except Exception as e:
            self.logger.warning(f"Micro-regime description generation failed: {e}")
            return "Undefined micro-regime"

    def _economic_regime_analysis(self,
                                 market_data: pd.DataFrame,
                                 regime_labels: np.ndarray,
                                 macro_regimes: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive economic analysis."""
        try:
            self.logger.info("📊 Performing economic regime analysis...")

            analysis = {
                'regime_significance_scores': {},
                'economic_regime_types': {},
                'cross_regime_correlations': {},
                'regime_economic_profiles': {}
            }

            # Calculate significance scores for each regime
            for regime_name, regime_info in macro_regimes.items():
                regime_id = regime_info['regime_id']
                regime_mask = regime_labels == regime_id

                if np.sum(regime_mask) > 0:
                    regime_data = market_data[regime_mask]
                    significance = self._calculate_economic_significance(
                        regime_data, regime_info['characteristics']
                    )

                    analysis['regime_significance_scores'][regime_name] = significance
                    analysis['regime_economic_profiles'][regime_name] = {
                        'volatility_profile': self._analyze_volatility_profile(regime_data),
                        'trend_profile': self._analyze_trend_profile(regime_data),
                        'efficiency_profile': self._analyze_efficiency_profile(regime_data),
                        'liquidity_profile': self._analyze_liquidity_profile(regime_data)
                    }

            # Analyze economic regime types
            analysis['economic_regime_types'] = self._classify_economic_regimes(macro_regimes)

            self.logger.info("   Economic analysis completed")
            return analysis

        except Exception as e:
            self.logger.error(f"Economic regime analysis failed: {e}")
            return {}

    def _financial_regime_analysis(self,
                                  market_data: pd.DataFrame,
                                  regime_labels: np.ndarray,
                                  macro_regimes: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive financial analysis."""
        try:
            self.logger.info("💰 Performing financial regime analysis...")

            analysis = {
                'trading_viability_scores': {},
                'risk_return_profiles': {},
                'regime_performance_metrics': {},
                'trading_regime_classification': {}
            }

            # Calculate financial metrics for each regime
            for regime_name, regime_info in macro_regimes.items():
                regime_id = regime_info['regime_id']
                regime_mask = regime_labels == regime_id

                if np.sum(regime_mask) > 0:
                    regime_data = market_data[regime_mask]

                    # Trading viability
                    viability = self._calculate_financial_viability(
                        regime_data, regime_info['characteristics']
                    )

                    # Risk-return profile
                    risk_return = self._calculate_risk_return_profile(regime_data)

                    analysis['trading_viability_scores'][regime_name] = viability
                    analysis['risk_return_profiles'][regime_name] = risk_return

                    analysis['regime_performance_metrics'][regime_name] = {
                        'sharpe_ratio': risk_return.get('sharpe_ratio', 0),
                        'max_drawdown': risk_return.get('max_drawdown', 0),
                        'win_rate': risk_return.get('win_rate', 0),
                        'profit_factor': risk_return.get('profit_factor', 1.0),
                        'avg_trade_duration': risk_return.get('avg_duration', 0)
                    }

            # Classify trading regimes
            analysis['trading_regime_classification'] = self._classify_trading_regimes(analysis)

            self.logger.info("   Financial analysis completed")
            return analysis

        except Exception as e:
            self.logger.error(f"Financial regime analysis failed: {e}")
            return {}

    def _regime_stability_analysis(self, regime_labels: np.ndarray, regime_probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze regime stability."""
        try:
            self.logger.info("🔍 Analyzing regime stability...")

            stability = {
                'overall_stability': 0.0,
                'regime_stability_scores': {},
                'stability_transitions': {},
                'persistent_regimes': [],
                'volatile_regimes': []
            }

            unique_regimes = sorted(set(regime_labels))

            for regime_id in unique_regimes:
                regime_mask = regime_labels == regime_id
                if np.sum(regime_mask) > 0:
                    regime_probs = regime_probabilities[regime_mask, regime_id]

                    # Calculate stability score
                    stability_score = np.mean(regime_probs)
                    stability['regime_stability_scores'][f"regime_{regime_id}"] = stability_score

                    if stability_score > 0.8:
                        stability['persistent_regimes'].append(regime_id)
                    elif stability_score < 0.5:
                        stability['volatile_regimes'].append(regime_id)

            # Overall stability
            if stability['regime_stability_scores']:
                stability['overall_stability'] = np.mean(list(stability['regime_stability_scores'].values()))

            self.logger.info(f"   Stability analysis completed (overall: {stability['overall_stability']:.3f})")
            return stability

        except Exception as e:
            self.logger.error(f"Regime stability analysis failed: {e}")
            return {}

    def _regime_transition_analysis(self, regime_labels: np.ndarray, regime_probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transitions."""
        try:
            self.logger.info("🔍 Analyzing regime transitions...")

            transition = {
                'transition_matrix': np.array([]),
                'transition_probabilities': {},
                'transition_smoothness': 0.0,
                'regime_persistence': {},
                'transition_patterns': []
            }

            # Calculate transition matrix
            n_regimes = len(set(regime_labels))
            transition_matrix = np.zeros((n_regimes, n_regimes))

            for i in range(len(regime_labels) - 1):
                current = regime_labels[i]
                next_regime = regime_labels[i + 1]

                if 0 <= current < n_regimes and 0 <= next_regime < n_regimes:
                    transition_matrix[current, next_regime] += 1

            # Normalize
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            transition['transition_matrix'] = transition_matrix / row_sums

            # Calculate transition smoothness
            diagonal_sum = np.trace(transition['transition_matrix'])
            transition['transition_smoothness'] = diagonal_sum / n_regimes

            # Regime persistence
            for i in range(n_regimes):
                persistence = transition['transition_matrix'][i, i]
                transition['regime_persistence'][f"regime_{i}"] = persistence

            self.logger.info(f"   Transition analysis completed (smoothness: {transition['transition_smoothness']:.3f})")
            return transition

        except Exception as e:
            self.logger.error(f"Regime transition analysis failed: {e}")
            return {}

    def _build_regime_hierarchy(self, macro_regimes: Dict[str, Any], micro_regimes: Dict[str, Any]) -> Dict[str, Any]:
        """Build hierarchy of regimes."""
        try:
            hierarchy = {
                'macro_regimes': macro_regimes,
                'micro_regimes': micro_regimes,
                'hierarchy_levels': {},
                'regime_relationships': {}
            }

            # Build hierarchy levels
            hierarchy['hierarchy_levels'] = {
                'level_1': list(macro_regimes.keys()),  # Macro regimes
                'level_2': list(micro_regimes.keys())   # Micro regimes
            }

            # Build relationships
            for micro_name, micro_regime in micro_regimes.items():
                macro_id = micro_regime['macro_regime_id']
                hierarchy['regime_relationships'][micro_name] = f"child_of_regime_{macro_id}"

            return hierarchy

        except Exception as e:
            self.logger.error(f"Regime hierarchy building failed: {e}")
            return {}

    def _analyze_volatility_profile(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volatility profile of a regime."""
        try:
            returns = regime_data['close'].pct_change().dropna()
            return {
                'mean_volatility': np.mean(np.abs(returns)),
                'volatility_std': np.std(np.abs(returns)),
                'max_volatility': np.max(np.abs(returns)),
                'volatility_skewness': pd.Series(np.abs(returns)).skew(),
                'volatility_clusters': self._detect_volatility_clusters(returns)
            }
        except:
            return {}

    def _analyze_trend_profile(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze trend profile of a regime."""
        try:
            prices = regime_data['close'].values
            returns = np.diff(prices, prepend=prices[0])

            # Linear trend analysis
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = linregress(x, prices)

            return {
                'trend_slope': slope,
                'trend_r_squared': r_value ** 2,
                'trend_p_value': p_value,
                'trend_significance': 1 - p_value if p_value < 0.1 else 0,
                'trend_direction': 'up' if slope > 0 else 'down'
            }
        except:
            return {}

    def _analyze_efficiency_profile(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze market efficiency profile."""
        try:
            returns = regime_data['close'].pct_change().dropna()
            return {
                'autocorrelation_lag1': returns.autocorr(lag=1),
                'autocorrelation_lag5': returns.autocorr(lag=5),
                'random_walk_test': self._random_walk_test(returns),
                'variance_ratio_test': self._variance_ratio_test(returns)
            }
        except:
            return {}

    def _analyze_liquidity_profile(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze liquidity profile."""
        try:
            spreads = (regime_data['high'] - regime_data['low']) / regime_data['close']
            volume = regime_data.get('volume', np.ones(len(regime_data)))

            return {
                'avg_spread': np.mean(spreads),
                'spread_volatility': np.std(spreads),
                'volume_price_impact': self._calculate_volume_impact(regime_data),
                'liquidity_ratio': np.mean(volume) / np.std(volume) if np.std(volume) > 0 else 0
            }
        except:
            return {}

    def _classify_economic_regimes(self, macro_regimes: Dict[str, Any]) -> Dict[str, str]:
        """Classify regimes into economic types."""
        try:
            classifications = {}

            for regime_name, regime_info in macro_regimes.items():
                characteristics = regime_info['characteristics']
                significance = regime_info['economic_significance']

                # Classification logic
                volatility = characteristics.get('volatility', 0.01)
                trend_r2 = characteristics.get('trend_r_squared', 0)
                autocorrelation = abs(characteristics.get('autocorrelation', 0))

                if volatility > 0.03 and trend_r2 > 0.3:
                    regime_type = "trending_high_volatility"
                elif volatility > 0.03 and trend_r2 < 0.1:
                    regime_type = "volatile_sideways"
                elif volatility < 0.01 and trend_r2 > 0.3:
                    regime_type = "stable_trending"
                elif volatility < 0.01 and trend_r2 < 0.1:
                    regime_type = "stable_sideways"
                elif autocorrelation > 0.3:
                    regime_type = "inefficient_predictable"
                else:
                    regime_type = "mixed_characteristics"

                classifications[regime_name] = regime_type

            return classifications

        except Exception as e:
            self.logger.warning(f"Economic regime classification failed: {e}")
            return {}

    def _classify_trading_regimes(self, financial_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Classify regimes by trading characteristics."""
        try:
            classifications = {}

            for regime_name, viability in financial_analysis['trading_viability_scores'].items():
                if viability > 0.8:
                    trading_type = "highly_tradable"
                elif viability > 0.6:
                    trading_type = "moderately_tradable"
                elif viability > 0.4:
                    trading_type = "challenging_trading"
                else:
                    trading_type = "difficult_trading"

                classifications[regime_name] = trading_type

            return classifications

        except Exception as e:
            self.logger.warning(f"Trading regime classification failed: {e}")
            return {}

    def _calculate_risk_return_profile(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk-return profile."""
        try:
            returns = regime_data['close'].pct_change().dropna()

            if len(returns) < 10:
                return {}

            mean_return = np.mean(returns)
            volatility = np.std(returns)

            # Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0

            # Maximum drawdown
            cumulative = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = peak - cumulative
            max_drawdown = np.max(drawdown) / (peak[-1] if peak[-1] > 0 else 1)

            # Win rate and profit factor
            wins = returns > 0
            win_rate = np.sum(wins) / len(returns)
            profit_factor = np.sum(returns[wins]) / abs(np.sum(returns[~wins])) if np.sum(~wins) > 0 else 1.0

            # Average trade duration (simplified)
            avg_duration = len(regime_data) / 10  # Rough estimate

            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_duration': avg_duration,
                'mean_return': mean_return,
                'volatility': volatility
            }

        except Exception as e:
            self.logger.warning(f"Risk-return profile calculation failed: {e}")
            return {}

    def _detect_volatility_clusters(self, returns: np.ndarray) -> int:
        """Detect number of volatility clusters in returns."""
        try:
            # Simple volatility clustering detection
            volatility = pd.Series(np.abs(returns)).rolling(window=10, min_periods=5).std()
            volatility = volatility.dropna()

            if len(volatility) < 20:
                return 1

            # Count significant changes in volatility
            vol_changes = np.abs(np.diff(volatility))
            threshold = np.percentile(vol_changes, 75)

            clusters = 1 + np.sum(vol_changes > threshold)
            return min(clusters, 5)  # Cap at 5 clusters

        except:
            return 1

    def _random_walk_test(self, returns: np.ndarray) -> float:
        """Test if returns follow random walk."""
        try:
            # Variance ratio test simplified
            n = len(returns)
            if n < 20:
                return 0.5

            # Compare variance of returns vs variance of cumulative returns
            return_var = np.var(returns)
            cumsum_var = np.var(np.cumsum(returns))

            if return_var == 0:
                return 1.0

            variance_ratio = cumsum_var / (n * return_var)
            return min(abs(1 - variance_ratio), 1.0)

        except:
            return 0.5

    def _variance_ratio_test(self, returns: np.ndarray) -> float:
        """Variance ratio test for market efficiency."""
        try:
            n = len(returns)
            if n < 40:
                return 0.5

            # Calculate variance ratios for different periods
            ratios = []
            for k in [2, 5, 10]:
                if n > 2 * k:
                    var_k = np.var([np.sum(returns[i:i+k]) for i in range(0, n - k, k)])
                    var_1 = np.var(returns)

                    if var_1 > 0:
                        ratio = var_k / (k * var_1)
                        ratios.append(abs(1 - ratio))

            return np.mean(ratios) if ratios else 0.5

        except:
            return 0.5

    def _calculate_volume_impact(self, regime_data: pd.DataFrame) -> float:
        """Calculate volume-price impact."""
        try:
            volume = regime_data.get('volume', np.ones(len(regime_data)))
            returns = regime_data['close'].pct_change().fillna(0)

            # Correlation between volume and absolute returns
            correlation = np.corrcoef(volume[1:], np.abs(returns[1:]))[0, 1]
            return abs(correlation)

        except:
            return 0.5

def create_coherent_regime_modeler(config: Dict[str, Any]) -> CoherentRegimeModeler:
    """Create coherent regime modeler."""
    return CoherentRegimeModeler(config)

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
