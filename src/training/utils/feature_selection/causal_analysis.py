from src.utils.tprint import tprint

"""
Causal Analysis Component

This module provides causal analysis capabilities for feature selection,
including causal inference, causal filtering, and domain knowledge integration.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import time
from collections import defaultdict

# Enhanced dependency management
try:
    from src.utils.logger import get_logger
    _LOGGER = get_logger("FeatureSelection.CausalAnalysis")
    tprint("✅ Custom logger available for FeatureSelection.CausalAnalysis")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("FeatureSelection.CausalAnalysis")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    SKLEARN_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SCIPY_AVAILABLE = False
    logger.warning("Scikit-learn/Scipy not available - limited causal analysis functionality")

# Import optimization utilities
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.matrix_operations import get_unified_matrix_operations
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Import common operations utilities
try:
    from src.utils.ml_common.utils import get_memory_usage
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError:
    COMMON_OPERATIONS_AVAILABLE = False

# Set matrix operations availability based on optimization imports
MATRIX_OPERATIONS_AVAILABLE = OPTIMIZATION_AVAILABLE

class CausalAnalyzer:
    """Causal analysis for feature selection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize causal analyzer."""
        self.config = config or {}
        self.logger = logger.getChild('CausalAnalyzer')

        # Causal analysis parameters
        self.correlation_threshold = self.config.get('correlation_threshold', 0.8)
        self.causal_significance_level = self.config.get('causal_significance_level', 0.05)
        self.domain_knowledge_weight = self.config.get('domain_knowledge_weight', 0.3)
        self.causal_inference_weight = self.config.get('causal_inference_weight', 0.7)

        # Domain knowledge patterns
        self.domain_patterns = self.config.get('domain_patterns', {})

        # Initialize optimization tools
        self._initialize_optimization_tools()

        _LOGGER.info("🔗 CausalAnalyzer initialized")
        _LOGGER.info(f"⚙️ Correlation threshold: {self.correlation_threshold}")
        _LOGGER.info(f"⚙️ Significance level: {self.causal_significance_level}")

    def _initialize_optimization_tools(self):
        """Initialize hardware optimization utilities."""
        try:
            if OPTIMIZATION_AVAILABLE and COMMON_OPERATIONS_AVAILABLE:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()

                if self.gpu_manager:
                    _LOGGER.info("✅ M1 GPU manager initialized for causal analysis")
                if self.memory_optimizer:
                    _LOGGER.info("✅ M1 memory optimizer initialized for causal analysis")
                if self.cpu_optimizer:
                    _LOGGER.info("✅ M1 CPU optimizer initialized for causal analysis")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        except Exception as e:
            _LOGGER.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

        try:
            if OPTIMIZATION_AVAILABLE and MATRIX_OPERATIONS_AVAILABLE:
                self.matrix_ops = get_unified_matrix_operations()
                _LOGGER.info("✅ Unified matrix operations initialized for causal analysis")
            else:
                self.matrix_ops = None
        except Exception as e:
            _LOGGER.warning(f"⚠️ Matrix operations initialization failed: {e}")
            self.matrix_ops = None

    def perform_causal_pre_filtering(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Perform causal pre-filtering of features."""
        start_time = time.time()
        _LOGGER.info(f"🔗 Starting causal pre-filtering...")
        _LOGGER.info(f"📊 Parameters - Data shape: {X.shape}")

        try:
            n_samples, n_features = X.shape

            # Apply domain knowledge filtering
            domain_filtered_features = self._domain_knowledge_filtering(X, y, feature_names)

            # Apply causal graph filtering
            causal_filtered_features = self._causal_graph_filtering(feature_names)

            # Apply statistical causal inference
            statistical_filtered_features = self._statistical_causal_inference(X, y, feature_names)

            # Combine filtering results
            filtered_features = self._combine_causal_filters(
                domain_filtered_features, causal_filtered_features, statistical_filtered_features
            )

            # Calculate causal relevance scores
            causal_scores = self._calculate_causal_relevance_scores(X, y, feature_names, filtered_features)

            execution_time = time.time() - start_time

            result = {
                'filtered_features': filtered_features,
                'causal_scores': causal_scores,
                'domain_filtered_features': domain_filtered_features,
                'causal_filtered_features': causal_filtered_features,
                'statistical_filtered_features': statistical_filtered_features,
                'method': 'causal_pre_filtering',
                'parameters': {
                    'correlation_threshold': self.correlation_threshold,
                    'causal_significance_level': self.causal_significance_level,
                    'domain_knowledge_weight': self.domain_knowledge_weight,
                    'causal_inference_weight': self.causal_inference_weight
                },
                'execution_time': execution_time,
                'success': True
            }

            _LOGGER.info(f"✅ Causal pre-filtering completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Filtered {len(filtered_features)} features from {len(feature_names)}")

            return result

        except Exception as e:
            _LOGGER.error(f"❌ Causal pre-filtering failed: {e}")
            return {
                'filtered_features': [],
                'causal_scores': {},
                'method': 'causal_pre_filtering',
                'error': str(e),
                'success': False
            }

    def perform_crypto_specific_causal_filtering(self, X: np.ndarray, y: np.ndarray,
                                               feature_names: List[str]) -> Dict[str, Any]:
        """Perform crypto-specific causal filtering."""
        start_time = time.time()
        _LOGGER.info(f"🔗 Starting crypto-specific causal filtering...")
        _LOGGER.info(f"📊 Parameters - Data shape: {X.shape}")

        try:
            n_samples, n_features = X.shape

            # Apply crypto-specific domain knowledge
            crypto_domain_features = self._crypto_specific_domain_filtering(X, y, feature_names)

            # Apply temporal causal analysis
            temporal_causal_features = self._temporal_causal_analysis(X, y, feature_names)

            # Apply market regime causal analysis
            regime_causal_features = self._market_regime_causal_analysis(X, y, feature_names)

            # Combine crypto-specific filters
            crypto_filtered_features = self._combine_crypto_causal_filters(
                crypto_domain_features, temporal_causal_features, regime_causal_features
            )

            # Calculate crypto-specific causal scores
            crypto_causal_scores = self._calculate_crypto_causal_scores(X, y, feature_names, crypto_filtered_features)

            execution_time = time.time() - start_time

            result = {
                'crypto_filtered_features': crypto_filtered_features,
                'crypto_causal_scores': crypto_causal_scores,
                'crypto_domain_features': crypto_domain_features,
                'temporal_causal_features': temporal_causal_features,
                'regime_causal_features': regime_causal_features,
                'method': 'crypto_specific_causal_filtering',
                'parameters': {
                    'correlation_threshold': self.correlation_threshold,
                    'causal_significance_level': self.causal_significance_level
                },
                'execution_time': execution_time,
                'success': True
            }

            _LOGGER.info(f"✅ Crypto-specific causal filtering completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Filtered {len(crypto_filtered_features)} features")

            return result

        except Exception as e:
            _LOGGER.error(f"❌ Crypto-specific causal filtering failed: {e}")
            return {
                'crypto_filtered_features': [],
                'crypto_causal_scores': {},
                'method': 'crypto_specific_causal_filtering',
                'error': str(e),
                'success': False
            }

    def _domain_knowledge_filtering(self, X: np.ndarray, y: np.ndarray,
                                  feature_names: List[str]) -> List[str]:
        """Apply domain knowledge filtering."""
        _LOGGER.debug("🔗 Applying domain knowledge filtering...")

        try:
            filtered_features = []

            for feature in feature_names:
                # Calculate domain relevance score
                feature_idx = feature_names.index(feature)
                feature_values = X[:, feature_idx]

                domain_score = self._calculate_domain_relevance_score(feature_values, feature, y)

                # Apply domain knowledge patterns
                pattern_score = self._apply_domain_patterns(feature, feature_values, y)

                # Combine scores
                combined_score = (domain_score + pattern_score) / 2.0

                if combined_score > 0.5:  # Threshold for domain relevance
                    filtered_features.append(feature)

            _LOGGER.debug(f"🔗 Domain knowledge filtering: {len(filtered_features)} features passed")
            return filtered_features

        except Exception as e:
            _LOGGER.warning(f"⚠️ Domain knowledge filtering failed: {e}")
            return feature_names  # Return all features if filtering fails

    def _causal_graph_filtering(self, feature_names: List[str]) -> List[str]:
        """Apply causal graph filtering."""
        _LOGGER.debug("🔗 Applying causal graph filtering...")

        try:
            filtered_features = []

            # Simple causal graph based on feature names and domain knowledge
            causal_edges = self._build_causal_graph(feature_names)

            for feature in feature_names:
                # Check if feature has causal path to target
                has_causal_path = self._has_causal_path_to_target(feature, 'target', causal_edges)

                if has_causal_path:
                    filtered_features.append(feature)

            _LOGGER.debug(f"🔗 Causal graph filtering: {len(filtered_features)} features passed")
            return filtered_features

        except Exception as e:
            _LOGGER.warning(f"⚠️ Causal graph filtering failed: {e}")
            return feature_names  # Return all features if filtering fails

    def _statistical_causal_inference(self, X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str]) -> List[str]:
        """Apply statistical causal inference."""
        _LOGGER.debug("🔗 Applying statistical causal inference...")

        try:
            filtered_features = []

            for feature in feature_names:
                feature_idx = feature_names.index(feature)
                feature_values = X[:, feature_idx]

                # Perform Granger causality test
                granger_p_value = self._granger_causality_test(feature_values, y)

                # Perform conditional independence test
                conditional_independence = self._conditional_independence_test(X, y, feature_idx)

                # Perform instrumental variable test
                iv_test = self._instrumental_variable_test(X, y, feature_idx)

                # Combine causal inference results
                causal_score = self._combine_causal_tests(granger_p_value, conditional_independence, iv_test)

                if causal_score > 0.5:  # Threshold for causal relevance
                    filtered_features.append(feature)

            _LOGGER.debug(f"🔗 Statistical causal inference: {len(filtered_features)} features passed")
            return filtered_features

        except Exception as e:
            _LOGGER.warning(f"⚠️ Statistical causal inference failed: {e}")
            return feature_names  # Return all features if filtering fails

    def _calculate_domain_relevance_score(self, feature_values: np.ndarray,
                                        feature_name: str, y: np.ndarray) -> float:
        """Calculate domain relevance score for a feature."""
        try:
            # Basic correlation with target
            corr = np.corrcoef(feature_values, y)[0, 1]
            correlation_score = abs(corr) if not np.isnan(corr) else 0.0

            # Feature name analysis
            name_score = self._analyze_feature_name(feature_name)

            # Feature distribution analysis
            distribution_score = self._analyze_feature_distribution(feature_values)

            # Combine scores
            domain_score = (correlation_score + name_score + distribution_score) / 3.0

            return domain_score

        except Exception as e:
            _LOGGER.debug(f"⚠️ Domain relevance score calculation failed: {e}")
            return 0.0

    def _analyze_feature_name(self, feature_name: str) -> float:
        """Analyze feature name for domain relevance."""
        try:
            # Simple keyword-based scoring
            relevant_keywords = [
                'price', 'volume', 'volatility', 'momentum', 'trend', 'support', 'resistance',
                'rsi', 'macd', 'bollinger', 'moving_average', 'ema', 'sma', 'vwap',
                'order_book', 'bid', 'ask', 'spread', 'depth', 'liquidity',
                'sentiment', 'fear', 'greed', 'social', 'news', 'twitter',
                'regime', 'market_state', 'volatility_regime', 'trend_regime'
            ]

            feature_lower = feature_name.lower()
            score = 0.0

            for keyword in relevant_keywords:
                if keyword in feature_lower:
                    score += 0.1

            return min(1.0, score)

        except Exception as e:
            _LOGGER.debug(f"⚠️ Feature name analysis failed: {e}")
            return 0.0

    def _analyze_feature_distribution(self, feature_values: np.ndarray) -> float:
        """Analyze feature distribution for relevance."""
        try:
            # Check for non-constant values
            if np.std(feature_values) == 0:
                return 0.0

            # Check for reasonable distribution
            if np.isnan(feature_values).any() or np.isinf(feature_values).any():
                return 0.0

            # Check for sufficient variation
            variation_score = min(1.0, np.std(feature_values) / (np.mean(np.abs(feature_values)) + 1e-10))

            return variation_score

        except Exception as e:
            _LOGGER.debug(f"⚠️ Feature distribution analysis failed: {e}")
            return 0.0

    def _apply_domain_patterns(self, feature_name: str, feature_values: np.ndarray, y: np.ndarray) -> float:
        """Apply domain-specific patterns."""
        try:
            pattern_score = 0.0

            # Apply configured domain patterns
            for pattern_name, pattern_config in self.domain_patterns.items():
                if self._matches_pattern(feature_name, pattern_config):
                    pattern_score += pattern_config.get('weight', 0.1)

            return min(1.0, pattern_score)

        except Exception as e:
            _LOGGER.debug(f"⚠️ Domain pattern application failed: {e}")
            return 0.0

    def _matches_pattern(self, feature_name: str, pattern_config: Dict[str, Any]) -> bool:
        """Check if feature matches a domain pattern."""
        try:
            pattern_type = pattern_config.get('type', 'name')

            if pattern_type == 'name':
                keywords = pattern_config.get('keywords', [])
                feature_lower = feature_name.lower()
                return any(keyword in feature_lower for keyword in keywords)

            elif pattern_type == 'regex':
                import re
                pattern = pattern_config.get('pattern', '')
                return bool(re.search(pattern, feature_name))

            return False

        except Exception as e:
            _LOGGER.debug(f"⚠️ Pattern matching failed: {e}")
            return False

    def _build_causal_graph(self, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Build causal graph based on domain knowledge."""
        try:
            edges = []

            # Define causal relationships based on domain knowledge
            causal_relationships = {
                'volume': ['price', 'volatility'],
                'volatility': ['price'],
                'momentum': ['price'],
                'trend': ['price'],
                'support': ['price'],
                'resistance': ['price'],
                'order_book': ['price', 'volume'],
                'sentiment': ['price', 'volume'],
                'regime': ['price', 'volatility', 'volume']
            }

            for feature in feature_names:
                feature_lower = feature_name.lower()

                # Check for causal relationships
                for cause, effects in causal_relationships.items():
                    if cause in feature_lower:
                        for effect in effects:
                            edges.append({
                                'from': feature,
                                'to': effect,
                                'type': 'causal',
                                'strength': 0.8
                            })

            return edges

        except Exception as e:
            _LOGGER.debug(f"⚠️ Causal graph building failed: {e}")
            return []

    def _has_causal_path_to_target(self, feature: str, target: str, edges: List[Dict[str, Any]]) -> bool:
        """Check if feature has causal path to target."""
        try:
            # Simple path finding
            visited = set()
            queue = [feature]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)

                if current == target:
                    return True

                # Find outgoing edges
                for edge in edges:
                    if edge['from'] == current and edge['to'] not in visited:
                        queue.append(edge['to'])

            return False

        except Exception as e:
            _LOGGER.debug(f"⚠️ Causal path checking failed: {e}")
            return True  # Assume causal path if check fails

    def _granger_causality_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """Perform Granger causality test."""
        try:
            if not SCIPY_AVAILABLE or len(x) < 20:
                return 0.5  # Neutral score if test cannot be performed

            # Simple implementation - in practice, use proper Granger causality test
            # This is a simplified version for demonstration

            # Calculate lagged correlation
            if len(x) > 1:
                x_lagged = x[:-1]
                y_current = y[1:]
                corr = np.corrcoef(x_lagged, y_current)[0, 1]
                return abs(corr) if not np.isnan(corr) else 0.0

            return 0.0

        except Exception as e:
            _LOGGER.debug(f"⚠️ Granger causality test failed: {e}")
            return 0.0

    def _conditional_independence_test(self, X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
        """Perform conditional independence test."""
        try:
            if not SCIPY_AVAILABLE or X.shape[1] < 2:
                return 0.5  # Neutral score if test cannot be performed

            feature_values = X[:, feature_idx]

            # Simple partial correlation test
            # Remove the feature and calculate partial correlation
            X_other = np.delete(X, feature_idx, axis=1)

            if X_other.shape[1] > 0:
                # Calculate partial correlation
                partial_corr = self._calculate_partial_correlation(feature_values, y, X_other)
                return abs(partial_corr) if not np.isnan(partial_corr) else 0.0

            return 0.0

        except Exception as e:
            _LOGGER.debug(f"⚠️ Conditional independence test failed: {e}")
            return 0.0

    def _calculate_partial_correlation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Calculate partial correlation between x and y controlling for z."""
        try:
            if not SKLEARN_AVAILABLE:
                return 0.0

            # Simple implementation using linear regression residuals
            # Fit x ~ z and y ~ z, then correlate residuals

            if z.shape[1] == 0:
                return np.corrcoef(x, y)[0, 1]

            # Fit x ~ z
            reg_x = LinearRegression().fit(z, x)
            x_residuals = x - reg_x.predict(z)

            # Fit y ~ z
            reg_y = LinearRegression().fit(z, y)
            y_residuals = y - reg_y.predict(z)

            # Correlate residuals
            partial_corr = np.corrcoef(x_residuals, y_residuals)[0, 1]
            return partial_corr if not np.isnan(partial_corr) else 0.0

        except Exception as e:
            _LOGGER.debug(f"⚠️ Partial correlation calculation failed: {e}")
            return 0.0

    def _instrumental_variable_test(self, X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
        """Perform instrumental variable test."""
        try:
            # Simplified instrumental variable test
            # In practice, this would require proper IV identification

            feature_values = X[:, feature_idx]

            # Simple test: check if feature has predictive power beyond correlation
            if len(feature_values) > 10:
                # Calculate R-squared when using feature to predict target
                if SKLEARN_AVAILABLE:
                    reg = LinearRegression().fit(feature_values.reshape(-1, 1), y)
                    r_squared = reg.score(feature_values.reshape(-1, 1), y)
                    return r_squared if not np.isnan(r_squared) else 0.0

            return 0.0

        except Exception as e:
            _LOGGER.debug(f"⚠️ Instrumental variable test failed: {e}")
            return 0.0

    def _combine_causal_tests(self, granger_p_value: float, conditional_independence: float, iv_test: float) -> float:
        """Combine results from different causal tests."""
        try:
            # Weighted combination of test results
            weights = [0.4, 0.3, 0.3]  # Granger, conditional independence, IV
            scores = [granger_p_value, conditional_independence, iv_test]

            combined_score = sum(w * s for w, s in zip(weights, scores))
            return combined_score

        except Exception as e:
            _LOGGER.debug(f"⚠️ Causal test combination failed: {e}")
            return 0.0

    def _combine_causal_filters(self, domain_features: List[str], causal_features: List[str],
                              statistical_features: List[str]) -> List[str]:
        """Combine results from different causal filters."""
        try:
            # Take intersection of all filters (most conservative approach)
            all_features = set(domain_features) & set(causal_features) & set(statistical_features)

            # If intersection is too small, use union
            if len(all_features) < len(domain_features) * 0.1:
                all_features = set(domain_features) | set(causal_features) | set(statistical_features)

            return list(all_features)

        except Exception as e:
            _LOGGER.warning(f"⚠️ Causal filter combination failed: {e}")
            return domain_features  # Fallback to domain features

    def _calculate_causal_relevance_scores(self, X: np.ndarray, y: np.ndarray,
                                         feature_names: List[str], filtered_features: List[str]) -> Dict[str, float]:
        """Calculate causal relevance scores for filtered features."""
        try:
            causal_scores = {}

            for feature in filtered_features:
                feature_idx = feature_names.index(feature)
                feature_values = X[:, feature_idx]

                # Calculate various causal relevance metrics
                predictive_power = self._calculate_predictive_power(feature_values, y)
                information_content = self._calculate_information_content(feature_values)
                temporal_stability = self._calculate_temporal_stability(feature_values, y)
                redundancy_penalty = self._calculate_redundancy_penalty(feature_values, X, feature_idx)

                # Combine into causal relevance score
                causal_score = (predictive_power + information_content + temporal_stability - redundancy_penalty) / 4.0
                causal_scores[feature] = max(0.0, causal_score)

            return causal_scores

        except Exception as e:
            _LOGGER.warning(f"⚠️ Causal relevance score calculation failed: {e}")
            return {}

    def _calculate_predictive_power(self, feature_values: np.ndarray, y: np.ndarray) -> float:
        """Calculate predictive power of a feature."""
        try:
            # Use correlation as a proxy for predictive power
            corr = np.corrcoef(feature_values, y)[0, 1]
            return abs(corr) if not np.isnan(corr) else 0.0

        except Exception as e:
            _LOGGER.debug(f"⚠️ Predictive power calculation failed: {e}")
            return 0.0

    def _calculate_information_content(self, feature_values: np.ndarray) -> float:
        """Calculate information content of a feature."""
        try:
            # Use entropy as a proxy for information content
            # Simplified version using variance
            variance = np.var(feature_values)
            return min(1.0, variance / (np.mean(np.abs(feature_values)) + 1e-10))

        except Exception as e:
            _LOGGER.debug(f"⚠️ Information content calculation failed: {e}")
            return 0.0

    def _calculate_temporal_stability(self, feature_values: np.ndarray, y: np.ndarray) -> float:
        """Calculate temporal stability of a feature."""
        try:
            # Calculate stability using rolling correlation
            if len(feature_values) < 20:
                return 0.5

            window_size = min(50, len(feature_values) // 4)
            correlations = []

            for i in range(window_size, len(feature_values)):
                window_feature = feature_values[i-window_size:i]
                window_target = y[i-window_size:i]

                corr = np.corrcoef(window_feature, window_target)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            if correlations:
                stability = 1.0 - np.std(correlations)  # Higher stability = lower variance
                return max(0.0, stability)

            return 0.0

        except Exception as e:
            _LOGGER.debug(f"⚠️ Temporal stability calculation failed: {e}")
            return 0.0

    def _calculate_redundancy_penalty(self, feature_values: np.ndarray, X: np.ndarray, feature_idx: int) -> float:
        """Calculate redundancy penalty for a feature."""
        try:
            # Calculate average correlation with other features
            correlations = []

            for i in range(X.shape[1]):
                if i != feature_idx:
                    corr = np.corrcoef(feature_values, X[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            if correlations:
                mean_correlation = np.mean(correlations)
                return mean_correlation  # Higher correlation = higher penalty

            return 0.0

        except Exception as e:
            _LOGGER.debug(f"⚠️ Redundancy penalty calculation failed: {e}")
            return 0.0

    # Crypto-specific methods
    def _crypto_specific_domain_filtering(self, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str]) -> List[str]:
        """Apply crypto-specific domain filtering."""
        _LOGGER.debug("🔗 Applying crypto-specific domain filtering...")

        try:
            crypto_keywords = [
                'btc', 'eth', 'crypto', 'bitcoin', 'ethereum', 'altcoin',
                'defi', 'nft', 'dao', 'yield', 'staking', 'liquidity',
                'whale', 'hodl', 'fomo', 'fud', 'pump', 'dump',
                'exchange', 'binance', 'coinbase', 'kraken', 'uniswap',
                'gas', 'fees', 'mining', 'hash', 'block', 'chain'
            ]

            filtered_features = []

            for feature in feature_names:
                feature_lower = feature.lower()

                # Check for crypto-specific keywords
                crypto_score = sum(0.1 for keyword in crypto_keywords if keyword in feature_lower)

                if crypto_score > 0.1:  # Has crypto relevance
                    filtered_features.append(feature)

            _LOGGER.debug(f"🔗 Crypto domain filtering: {len(filtered_features)} features passed")
            return filtered_features

        except Exception as e:
            _LOGGER.warning(f"⚠️ Crypto domain filtering failed: {e}")
            return feature_names

    def _temporal_causal_analysis(self, X: np.ndarray, y: np.ndarray,
                                feature_names: List[str]) -> List[str]:
        """Perform temporal causal analysis for crypto features."""
        _LOGGER.debug("🔗 Performing temporal causal analysis...")

        try:
            # This would implement temporal causal analysis specific to crypto markets
            # For now, return all features
            return feature_names

        except Exception as e:
            _LOGGER.warning(f"⚠️ Temporal causal analysis failed: {e}")
            return feature_names

    def _market_regime_causal_analysis(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str]) -> List[str]:
        """Perform market regime causal analysis."""
        _LOGGER.debug("🔗 Performing market regime causal analysis...")

        try:
            # This would implement regime-specific causal analysis
            # For now, return all features
            return feature_names

        except Exception as e:
            _LOGGER.warning(f"⚠️ Market regime causal analysis failed: {e}")
            return feature_names

    def _combine_crypto_causal_filters(self, domain_features: List[str],
                                     temporal_features: List[str],
                                     regime_features: List[str]) -> List[str]:
        """Combine crypto-specific causal filters."""
        try:
            # Take union of all crypto filters
            all_features = set(domain_features) | set(temporal_features) | set(regime_features)
            return list(all_features)

        except Exception as e:
            _LOGGER.warning(f"⚠️ Crypto causal filter combination failed: {e}")
            return domain_features

    def _calculate_crypto_causal_scores(self, X: np.ndarray, y: np.ndarray,
                                      feature_names: List[str],
                                      filtered_features: List[str]) -> Dict[str, float]:
        """Calculate crypto-specific causal scores."""
        try:
            # Use the same method as general causal scores
            return self._calculate_causal_relevance_scores(X, y, feature_names, filtered_features)

        except Exception as e:
            _LOGGER.warning(f"⚠️ Crypto causal score calculation failed: {e}")
            return {}
