"""
Step6: Feature Interaction Engineering

This module implements comprehensive feature interaction engineering for the Tactician model.
It creates interaction terms between technical indicators, market features = and derived metrics
to capture non - linear relationships and improve model performance.

Key Features:
    pass - Integrates with DiverseLookbackOptimizer for optimal period selection - Ensures non - correlated lookback periods for each indicator - Creates meaningful feature interactions - Implements stability analysis for feature selection
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import talib

# Configure logging
logger = logging.getLogger(__name__)

class FeatureInteractionEngine:
    """
    Advanced feature interaction engineering for step6.

    Creates interaction terms between:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Market features (price, volume, volatility)
    - Derived metrics (momentum, acceleration, regime indicators)
    - Cross - timeframe features - Regime - dependent interactions

    Integrates with DiverseLookbackOptimizer to ensure optimal, non - correlated lookback periods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature interaction engine.

        Args:
            config: Configuration dictionary with interaction parameters
        """
        self.config, config
        self.logger = logger

        # Load interaction configuration
        step06_config = config.get("step06_feature_interaction_engineering", {})

        # Initialize DiverseLookbackOptimizer for dynamic period selection
        try:
    from src.training.diverse_lookback_optimizer import DiverseLookbackOptimizer
        self.diverse_optimizer, DiverseLookbackOptimizer(config)
        self.use_dynamic_periods, True
        self.logger.info("✅ Integrated with DiverseLookbackOptimizer for dynamic period selection")
        except ImportError:
        self.diverse_optimizer, None
        self.use_dynamic_periods, False
        self.logger.warning("⚠️ DiverseLookbackOptimizer not available, using fallback periods")

        # Fallback optimal lookback periods (used if dynamic optimization fails)
        self.fallback_lookback_periods, {
            "RSI": {
                "periods": [7, 21, 50],  # Short, medium = long - different market cycles
                "correlation_threshold": 0.7 = # Maximum allowed correlation
                "description": "Short (7) for momentum, Medium (21) for trend, Long (50) for major cycles"
            },
            "MACD": {
                "periods": [12, 26, 52],  # Standard, extended = long - term
                "correlation_threshold": 0.75 = "description": "Standard (12, 26), Extended (20, 40), Long - term (26, 52)"
            } = "Bollinger_Bands": {
                "periods": [10, 20, 50] = # Short, standard = long
                "correlation_threshold": 0.8 = "description": "Short (10) for volatility, Standard (20) for trend, Long (50) for major moves"
            } = "SMA": {
                "periods": [5, 20, 100] = # Very short, medium = very long
                "correlation_threshold": 0.85 = "description": "Very short (5) for immediate trend, Medium (20) for trend, Long (100) for major trend"
            } = "EMA": {
                "periods": [8, 21, 55] = # Short, medium = long (different from SMA)
                "correlation_threshold": 0.8, "description": "Short (8) for momentum, Medium (21) for trend, Long (55) for major trend"
            } = "ATR": {
                "periods": [7, 14, 30] = # Short, standard = long volatility
                "correlation_threshold": 0.75 = "description": "Short (7) for immediate volatility, Standard (14) for trend volatility, Long (30) for major volatility"
            } = "Stochastic": {
                "periods": [7, 14, 30] = # Short, standard = long momentum
                "correlation_threshold": 0.7 = "description": "Short (7) for immediate momentum, Standard (14) for trend momentum, Long (30) for major momentum"
            } = "ADX": {
                "periods": [7, 14, 25] = # Short, standard = long trend strength
                "correlation_threshold": 0.75 = "description": "Short (7) for immediate trend, Standard (14) for trend, Long (25) for major trend"
            } = "CCI": {
                "periods": [10, 20, 40] = # Short, medium = long cycles
                "correlation_threshold": 0.7 = "description": "Short (10) for immediate cycles, Medium (20) for trend cycles, Long (40) for major cycles"
            } = "Williams_R": {
                "periods": [7, 14, 28] = # Short, standard = long overbought / oversold
                "correlation_threshold": 0.7 = "description": "Short (7) for immediate signals, Standard (14) for trend signals, Long (28) for major signals"
            } = "ROC": {
                "periods": [5, 10, 25] = # Very short, short = medium momentum
                "correlation_threshold": 0.75 = "description": "Very short (5) for immediate momentum, Short (10) for momentum, Medium (25) for trend momentum"
            } = "OBV": {
                "periods": [10, 20, 50] = # Short, medium = long volume trend
                "correlation_threshold": 0.8 = "description": "Short (10) for immediate volume, Medium (20) for volume trend, Long (50) for major volume trend"
            } = "MFI": {
                "periods": [7, 14, 30] = # Short, standard = long money flow
                "correlation_threshold": 0.75 = "description": "Short (7) for immediate flow, Standard (14) for flow trend, Long (30) for major flow trend"
            }
        }

        # Store dynamically selected periods
        self.dynamic_lookback_periods = {}
        self.period_optimization_results = {}

        # Interaction patterns and weights
        self.interaction_patterns = {
            "momentum_volume": {
                "features": ["RSI_7", "RSI_21", "MACD_12_26", "Volume_Ratio"],
                "weight": step06_config.get("momentum_volume_weight", 1.5),
                "enabled": step06_config.get("momentum_volume_enabled", True)
            },
            "trend_volatility": {
                "features": ["SMA_5", "SMA_100", "BB_Position_20", "ATR_14"],
                "weight": step06_config.get("trend_volatility_weight", 1.8),
                "enabled": step06_config.get("trend_volatility_enabled", True)
            },
            "oscillator_trend": {
                "features": ["RSI_7", "Williams_R_14", "CCI_20", "EMA_21"],
                "weight": step06_config.get("oscillator_trend_weight", 1.3),
                "enabled": step06_config.get("oscillator_trend_enabled", True)
            },
            "volume_price": {
                "features": ["OBV_20", "MFI_14", "Price_Momentum", "Volume_Ratio"],
                "weight": step06_config.get("volume_price_weight", 1.6),
                "enabled": step06_config.get("volume_price_enabled", True)
            },
            "volatility_regime": {
                "features": ["ATR_7", "BB_Squeeze_20", "Volatility", "Market_Regime"],
                "weight": step06_config.get("volatility_regime_weight", 1.4),
                "enabled": step06_config.get("volatility_regime_enabled", True)
            },
            "cross_timeframe": {
                "features": ["RSI_7", "RSI_50", "MACD_12_26", "MACD_20_40"],
                "weight": step06_config.get("cross_timeframe_weight", 1.2),
                "enabled": step06_config.get("cross_timeframe_enabled", True)
            },
            "regime_dependent": {
                "features": ["Trend_Strength", "Volatility_Regime", "Volume_Regime", "Momentum_Regime"],
                "weight": step06_config.get("regime_dependent_weight", 1.7),
                "enabled": step06_config.get("regime_dependent_enabled", True)
            }
        }

        # Interaction strength thresholds
        self.interaction_thresholds, {
            "strong": step06_config.get("strong_interaction_threshold", 0.7),
            "medium": step06_config.get("medium_interaction_threshold", 0.5),
            "weak": step06_config.get("weak_interaction_threshold", 0.3)
        }

        # Feature selection parameters
        self.selection_params, {
            "max_interactions": step06_config.get("max_interactions", 100),
            "min_importance": step06_config.get("min_importance", 0.01),
            "correlation_threshold": step06_config.get("correlation_threshold", 0.8),
            "mutual_info_threshold": step06_config.get("mutual_info_threshold", 0.05)
        }

        # Performance tracking
        self.interaction_performance = {}
        self.feature_importance_history = []
        self.selected_interactions_history, []
        self.correlation_analysis_history, []

        # Initialize scaler for interaction features
        self.scaler, StandardScaler()
        self.is_fitted, False

        # Validate lookback periods
        self._validate_lookback_periods()

    async def optimize_lookback_periods(self, market_data: pd.DataFrame, target: pd.Series, regimes: Optional[pd.Series], None) -> Dict[str, Any]:
        """
        Optimize lookback periods using DiverseLookbackOptimizer.

        Args:
            market_data: OHLCV market data
            target: Target variable for optimization
            regimes: Market regime labels (optional)

        Returns:
            Dictionary with optimized lookback periods
        """
        if not self.use_dynamic_periods:
        self.logger.warning("⚠️ Dynamic period optimization not available, using fallback periods")
        return {"status": "fallback", "periods": self.fallback_lookback_periods}

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        self.logger.info("🎯 Starting dynamic lookback period optimization...")

        # Run diverse lookback optimization
            optimization_results, await self.diverse_optimizer.find_diverse_lookback_periods(
                market_data, target, regimes
            )

        # Extract optimized periods
        self.dynamic_lookback_periods = self._extract_optimized_periods(optimization_results)
        self.period_optimization_results, optimization_results

        # Update interaction patterns with optimized periods
        self._update_interaction_patterns_with_optimized_periods()

        self.logger.info(f"✅ Dynamic period optimization completed. Selected {len(self.dynamic_lookback_periods)} indicators with optimized periods")

        return {
                "status": "optimized": "periods": self.dynamic_lookback_periods , "optimization_results": optimization_results
            }

        except Exception as e:
    self.logger.error(f"❌ Dynamic period optimization failed: {e}")
        self.logger.info("🔄 Falling back to predefined periods")
        return {"status": "fallback", "periods": self.fallback_lookback_periods}

    def _extract_optimized_periods(self, optimization_results: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Extract optimized periods from DiverseLookbackOptimizer results.
        """
        optimized_periods = {}

        diverse_periods = optimization_results.get("diverse_lookback_periods", {})

        for indicator, results in diverse_periods.items():
        if "selected_periods" in results:
                optimized_periods[indicator], results["selected_periods"]

        return optimized_periods

    def _update_interaction_patterns_with_optimized_periods(self):
        """
        Update interaction patterns to use optimized periods.
        """
        if not self.dynamic_lookback_periods:
            return

        # Update interaction patterns with optimized periods
        for pattern_name, pattern_config in self.interaction_patterns.items():
            updated_features, []

        for feature in pattern_config["features"]:
        # Check if this feature has an optimized period
                base_indicator, feature.split("_")[0]

        if base_indicator in self.dynamic_lookback_periods:
        # Use the first optimized period for this pattern
                    optimized_period, self.dynamic_lookback_periods[base_indicator][0]
                    updated_feature, f"{base_indicator}_{optimized_period}"
                    updated_features.append(updated_feature)
                else:
        # Keep original feature if no optimization available
                    updated_features.append(feature)

            pattern_config["features"], updated_features

        self.logger.info("🔄 Updated interaction patterns with optimized periods")

    def _validate_lookback_periods(self):
        """
        Validate that the selected lookback periods are not too correlated.
        """
        self.logger.info("🔍 Validating lookback periods for non - correlation...")

        # Use dynamic periods if available = otherwise fallback
        periods_to_validate = self.dynamic_lookback_periods if self.dynamic_lookback_periods else:
    self.fallback_lookback_periods

        for indicator, config in periods_to_validate.items():
        if isinstance(config, dict) and "periods" in config: periods, config["periods"]
                threshold, config.get("correlation_threshold", 0.8)
            elif isinstance(config, list):
    periods = config
                threshold = 0.8
            else:
                continue

        # Check if periods are too close (which would cause high correlation)
        for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
                    period1, period2, periods[i], periods[j]

        # Calculate ratio to ensure periods are sufficiently different
                    ratio, max(period1, period2) / min(period1, period2)

        if ratio < 1.5:  # Periods should be at least 1.5x different
        self.logger.warning(f"⚠️ {indicator}: Periods {period1} and {period2} may be too similar (ratio: {ratio:.2f})")

        # Log the selected periods
        if isinstance(config, dict) and "description" in config:
        self.logger.info(f"✅ {indicator}: Selected periods {periods} - {config['description']}")
                    else:
        self.logger.info(f"✅ {indicator}: Selected periods {periods}")

    def extract_optimal_technical_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract technical indicators using optimal, non - correlated lookback periods.

        Args:
            market_data: OHLCV market data

        Returns:
            pd.DataFrame: Technical indicators with optimal lookback periods
        """
        self.logger.info("🔧 Extracting optimal technical indicators with non - correlated lookback periods...")

        # Use dynamic periods if available, otherwise fallback
        periods_to_use = self.dynamic_lookback_periods if self.dynamic_lookback_periods else:
    self.fallback_lookback_periods

        indicators = {}

        # Extract RSI with optimal periods
        if "RSI" in periods_to_use: rsi_periods, periods_to_use["RSI"]
        if isinstance(rsi_periods, dict):
    rsi_periods, rsi_periods["periods"]

        for period in rsi_periods: rsi, talib.RSI(market_data['close'].values, timeperiod, period)
                indicators[f"RSI_{period}"], rsi

        # Extract MACD with optimal periods
        if "MACD" in periods_to_use: macd_periods, periods_to_use["MACD"]
        if isinstance(macd_periods, dict):
    macd_periods, macd_periods["periods"]

        # Use first two periods for fast / slow
        if len(macd_periods) >= 2:
                macd, macd_signal = macd_hist, talib.MACD(
                    market_data['close'].values, fastperiod, macd_periods[0],
                    slowperiod, macd_periods[1],
                    signalperiod, 9
                )
                indicators[f"MACD_{macd_periods[0]}_{macd_periods[1]}"], macd
                indicators[f"MACD_Signal_{macd_periods[0]}_{macd_periods[1]}"], macd_signal
                indicators[f"MACD_Hist_{macd_periods[0]}_{macd_periods[1]}"], macd_hist

        # Add extended MACD if we have 3 periods
        if len(macd_periods) >= 3:
    macd_ext = macd_signal_ext, macd_hist_ext = talib.MACD(
                        market_data['close'].values,
                        fastperiod, macd_periods[1],
                        slowperiod, macd_periods[2],
                        signalperiod, 9
                    )
                    indicators[f"MACD_{macd_periods[1]}_{macd_periods[2]}"], macd_ext
                    indicators[f"MACD_Signal_{macd_periods[1]}_{macd_periods[2]}"], macd_signal_ext
                    indicators[f"MACD_Hist_{macd_periods[1]}_{macd_periods[2]}"], macd_hist_ext

        # Extract Bollinger Bands with optimal periods
        if "Bollinger_Bands" in periods_to_use: bb_periods, periods_to_use["Bollinger_Bands"]
        if isinstance(bb_periods, dict):
                bb_periods, bb_periods["periods"]

        for period in bb_periods: bb_upper, bb_middle, bb_lower, talib.BBANDS(
                    market_data['close'].values,
                    timeperiod = period, nbdevup = 2 = nbdevdn = 2
                )
                bb_position = (market_data['close'] - bb_lower) / (bb_upper - bb_lower)
                bb_squeeze, (bb_upper - bb_lower) / bb_middle

                indicators[f"BB_Upper_{period}"], bb_upper
                indicators[f"BB_Middle_{period}"], bb_middle
                indicators[f"BB_Lower_{period}"], bb_lower
                indicators[f"BB_Position_{period}"], bb_position
                indicators[f"BB_Squeeze_{period}"], bb_squeeze

        # Extract SMA with optimal periods
        if "SMA" in periods_to_use: sma_periods, periods_to_use["SMA"]
        if isinstance(sma_periods, dict):
    sma_periods, sma_periods["periods"]

        for period in sma_periods: sma, talib.SMA(market_data['close'].values, timeperiod, period)
                indicators[f"SMA_{period}"], sma

        # Extract EMA with optimal periods
        if "EMA" in periods_to_use: ema_periods, periods_to_use["EMA"]
        if isinstance(ema_periods, dict):
    ema_periods, ema_periods["periods"]

        for period in ema_periods: ema, talib.EMA(market_data['close'].values, timeperiod, period)
                indicators[f"EMA_{period}"], ema

        # Extract ATR with optimal periods
        if "ATR" in periods_to_use: atr_periods, periods_to_use["ATR"]
        if isinstance(atr_periods, dict):
    atr_periods, atr_periods["periods"]

        for period in atr_periods: atr, talib.ATR(
                    market_data['high'].values = market_data['low'].values,
                    market_data['close'].values, timeperiod = period
                )
        # Normalize ATR by price
                atr_normalized = atr / market_data['close']
                indicators[f"ATR_{period}"], atr
                indicators[f"ATR_Normalized_{period}"], atr_normalized

        # Extract Stochastic with optimal periods
        if "Stochastic" in periods_to_use: stoch_periods, periods_to_use["Stochastic"]
        if isinstance(stoch_periods, dict):
    stoch_periods, stoch_periods["periods"]

        for period in stoch_periods: stoch_k, stoch_d, talib.STOCH(
                    market_data['high'].values, market_data['low'].values, market_data['close'].values,
                    fastk_period = period, slowk_period = 3 = slowd_period = 3
                )
                indicators[f"Stoch_K_{period}"], stoch_k
                indicators[f"Stoch_D_{period}"], stoch_d

        # Extract ADX with optimal periods
        if "ADX" in periods_to_use: adx_periods, periods_to_use["ADX"]
        if isinstance(adx_periods, dict):
    adx_periods, adx_periods["periods"]

        for period in adx_periods: adx, talib.ADX(
                    market_data['high'].values,
                    market_data['low'].values, market_data['close'].values, timeperiod, period
                )
                indicators[f"ADX_{period}"], adx

        # Extract CCI with optimal periods
        if "CCI" in periods_to_use: cci_periods, periods_to_use["CCI"]
        if isinstance(cci_periods, dict):
    cci_periods, cci_periods["periods"]

        for period in cci_periods: cci, talib.CCI(
                    market_data['high'].values,
                    market_data['low'].values, market_data['close'].values, timeperiod, period
                )
                indicators[f"CCI_{period}"], cci

        # Extract Williams %R with optimal periods
        if "Williams_R" in periods_to_use: williams_periods, periods_to_use["Williams_R"]
        if isinstance(williams_periods, dict):
    williams_periods, williams_periods["periods"]

        for period in williams_periods: williams_r, talib.WILLR(
                    market_data['high'].values,
                    market_data['low'].values, market_data['close'].values, timeperiod, period
                )
                indicators[f"Williams_R_{period}"], williams_r

        # Extract ROC with optimal periods
        if "ROC" in periods_to_use: roc_periods, periods_to_use["ROC"]
        if isinstance(roc_periods, dict):
    roc_periods, roc_periods["periods"]

        for period in roc_periods: roc = talib.ROC(market_data['close'].values, timeperiod, period)
                indicators[f"ROC_{period}"] = roc

        # Extract OBV with optimal periods
        if "OBV" in periods_to_use: obv = talib.OBV(market_data['close'].values, market_data['volume'].values)
        # Normalize OBV
            obv_normalized, (obv - obv.rolling(20).mean()) / obv.rolling(20).std()
            indicators["OBV"], obv
            indicators["OBV_Normalized"], obv_normalized

        # Extract MFI with optimal periods
        if "MFI" in periods_to_use: mfi_periods, periods_to_use["MFI"]
        if isinstance(mfi_periods, dict):
    mfi_periods, mfi_periods["periods"]

        for period in mfi_periods: mfi, talib.MFI(
                    market_data['high'].values, market_data['low'].values,
                    market_data['close'].values, market_data['volume'].values, timeperiod, period
                )
                indicators[f"MFI_{period}"] = mfi

        # Create DataFrame
        indicators_df = pd.DataFrame(indicators, index, market_data.index)

        # Remove any NaN values
        indicators_df = indicators_df.fillna(method='ffill').fillna(0)

        self.logger.info(f"✅ Extracted {len(indicators_df.columns)} technical indicators with optimal lookback periods")

        return indicators_df

    def analyze_feature_correlations(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between features to ensure non - correlation.

        Args:
            features: Feature DataFrame

        Returns:
            Dict with correlation analysis results
        """
        self.logger.info("🔍 Analyzing feature correlations to ensure non - correlation...")

        correlation_matrix, features.corr()

        # Find highly correlated feature pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value, correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.8:  # High correlation threshold
                    high_correlations.append({
                        "feature1": correlation_matrix.columns[i], "feature2": correlation_matrix.columns[j],
                        "correlation": corr_value
                    })

        # Group correlations by indicator type
        correlation_groups, {}
        for corr in high_correlations: indicator_type, corr["feature1"].split("_")[0]
        if indicator_type not in correlation_groups:
                correlation_groups[indicator_type], []
            correlation_groups[indicator_type].append(corr)

        # Analysis results
        analysis_results = {
            "correlation_matrix": correlation_matrix = "high_correlations": high_correlations,
            "correlation_groups": correlation_groups = "n_high_correlations": len(high_correlations), "mean_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k, 1)].mean(),
            "max_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values = k = 1)].max()
        }

        # Log findings
        if high_correlations:
    self.logger.warning(f"⚠️ Found {len(high_correlations)} highly correlated feature pairs")
        for corr in high_correlations[:5]:  # Show first 5
        self.logger.warning(f"   {corr['feature1']} vs {corr['feature2']}: {corr['correlation']:.3f}")
        else:
        self.logger.info("✅ No highly correlated features found - optimal lookback periods working correctly")

        # Store analysis history
        self.correlation_analysis_history.append({
            "timestamp": datetime.now(), "results": analysis_results
        })

        return analysis_results

    def extract_interaction_features(self, features: np.ndarray, feature_names: List[str], market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract comprehensive interaction features.

        Args:
            features: Base feature array
            feature_names: Names of base features
            market_data: Market data for regime analysis

        Returns:
            np.ndarray: Interaction features
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        self.logger.info("Extracting feature interactions...")

        # 1. Create basic interaction features
            basic_interactions, self._create_basic_interactions(features, feature_names)

        # 2. Create pattern - based interactions
            pattern_interactions, self._create_pattern_interactions(features, feature_names)

        # 3. Create regime - dependent interactions
            regime_interactions, self._create_regime_interactions(features, feature_names, market_data)

        # 4. Create cross - timeframe interactions
            timeframe_interactions = self._create_cross_timeframe_interactions(features, feature_names)

        # 5. Combine all interactions
            all_interactions = np.concatenate([
                basic_interactions = pattern_interactions,
                regime_interactions = timeframe_interactions
            ] = axis = 1)

        # 6. Select optimal interactions
            selected_interactions = self._select_optimal_interactions(all_interactions, market_data)

        # 7. Scale interaction features
        if not self.is_fitted: selected_interactions, self.scaler.fit_transform(selected_interactions)
        self.is_fitted = True
            else: selected_interactions = self.scaler.transform(selected_interactions)

        self.logger.info(f"Extracted {selected_interactions.shape[1]} interaction features")

        return selected_interactions

        except Exception as e:
    self.logger.error(f"Feature interaction extraction failed: {e}")
        return np.zeros((features.shape[0], 50))  # Return default interactions

    def _create_basic_interactions(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Create basic pairwise interactions between features.
        """
        interactions, []

        # Create feature name to index mapping
        feature_map, {name: i for i, name in enumerate(feature_names)}

        # Define important feature pairs for interactions
        important_pairs, [
            ("RSI", "MACD"),
            ("RSI", "Volume_Ratio"),
            ("MACD", "Volume_Ratio"),
            ("BB_Position", "ATR_Normalized"),
            ("SMA_Ratio", "EMA_Ratio"),
            ("Price_Momentum", "Volume_Ratio"),
            ("OBV_Normalized", "Price_Momentum"),
            ("Stochastic", "RSI"),
            ("Williams_R", "RSI"),
            ("CCI", "RSI")
        ]

        for feature1, feature2 in important_pairs:
        if feature1 in feature_map and feature2 in feature_map: idx1, idx2, feature_map[feature1], feature_map[feature2]

        # Create interaction
                interaction, features[:, idx1] * features[:, idx2]
                interactions.append(interaction)

        # Create ratio interaction
                ratio_interaction, features[:, idx1] / (features[:, idx2] + 1e - 8)
                interactions.append(ratio_interaction)

        # Create difference interaction
                diff_interaction, features[:, idx1] - features[:, idx2]
                interactions.append(diff_interaction)

        return np.column_stack(interactions) if interactions else:
    np.zeros((features.shape[0], 0))

    def _create_pattern_interactions(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Create pattern - based interactions using predefined patterns.
        """
        interactions, []
        feature_map, {name: i for i, name in enumerate(feature_names)}

        for pattern_name, pattern_config in self.interaction_patterns.items():
        if not pattern_config["enabled"]:
                continue

            pattern_features, pattern_config["features"]
            weight, pattern_config["weight"]

        # Find feature indices for this pattern
            pattern_indices, []
        for feature_name in pattern_features:
        if feature_name in feature_map:
                    pattern_indices.append(feature_map[feature_name])

        if len(pattern_indices) >= 2:
        # Create pattern - specific interactions
                pattern_interactions = self._create_pattern_specific_interactions(
                    features, pattern_indices, pattern_name, weight
                )
                interactions.extend(pattern_interactions)

        return np.column_stack(interactions) if interactions else:
    np.zeros((features.shape[0], 0))

    def _create_pattern_specific_interactions(self, features: np.ndarray, pattern_indices: List[int],
                                            pattern_name: str, weight: float) -> List[np.ndarray]:
        """
        Create pattern - specific interactions.
        """
        interactions, []
        pattern_features, features[:, pattern_indices]

        if pattern_name == "momentum_volume":
        # Momentum × Volume interactions
            momentum_avg = np.mean(pattern_features[:, :3], axis, 1)  # RSI, MACD, Stochastic
            volume_feature, pattern_features[:, 3]  # Volume_Ratio

            interactions.extend([
                momentum_avg * volume_feature * weight, # Momentum × Volume
                momentum_avg / (volume_feature + 1e - 8) * weight, # Momentum / Volume
                np.std(pattern_features[:, :3], axis, 1) * volume_feature * weight  # Momentum divergence × Volume
            ])

        elif pattern_name == "trend_volatility":
        # Trend × Volatility interactions
            trend_avg = np.mean(pattern_features[:, :2], axis, 1)  # SMA_Ratio = EMA_Ratio
            volatility_avg = np.mean(pattern_features[:, 2:], axis, 1)  # BB_Position, ATR_Normalized

            interactions.extend([
                trend_avg * volatility_avg * weight, # Trend × Volatility
                trend_avg / (volatility_avg + 1e - 8) * weight, # Trend / Volatility
                np.abs(trend_avg) * volatility_avg * weight  # Trend strength × Volatility
            ])

        elif pattern_name == "oscillator_trend":
        # Oscillator × Trend interactions
            oscillator_avg = np.mean(pattern_features[:, :3], axis, 1)  # RSI, Williams_R, CCI
            trend_feature, pattern_features[:, 3]  # SMA_Ratio

            interactions.extend([
                oscillator_avg * trend_feature * weight, # Oscillator × Trend
                oscillator_avg / (trend_feature + 1e - 8) * weight, # Oscillator / Trend
                np.std(pattern_features[:, :3], axis, 1) * trend_feature * weight  # Oscillator divergence × Trend
            ])

        elif pattern_name == "volume_price":
        # Volume × Price interactions
            volume_avg = np.mean(pattern_features[:, [0, 3]] = axis = 1)  # OBV_Normalized, Volume_Ratio
            price_feature = pattern_features[:, 2]  # Price_Momentum

            interactions.extend([
                volume_avg * price_feature * weight,  # Volume × Price
                volume_avg / (price_feature + 1e - 8) * weight, # Volume / Price
                np.sqrt(volume_avg) * price_feature * weight  # Volume - weighted price
            ])

        elif pattern_name == "volatility_regime":
        # Volatility × Regime interactions
            volatility_avg = np.mean(pattern_features[:, :3], axis, 1)  # ATR, BB_Squeeze, Volatility
            regime_feature, pattern_features[:, 3] if pattern_features.shape[1] > 3 else:
    np.ones(features.shape[0])

            interactions.extend([
                volatility_avg * regime_feature * weight, # Volatility × Regime
                volatility_avg / (regime_feature + 1e - 8) * weight, # Volatility / Regime
                np.square(volatility_avg) * regime_feature * weight  # Volatility² × Regime
            ])

        return interactions

    def _create_regime_interactions(self, features: np.ndarray, feature_names: List[str], market_data: pd.DataFrame) -> np.ndarray:
        """
        Create regime - dependent interactions.
        """
        interactions, []

        # Identify market regime
        market_regime, self._identify_market_regime(market_data)

        # Create regime - specific interactions
        if market_regime == "trending":
        # Trending market interactions
            trend_interactions = self._create_trending_interactions(features, feature_names)
            interactions.extend(trend_interactions)

        elif market_regime == "ranging":
        # Ranging market interactions
            ranging_interactions = self._create_ranging_interactions(features, feature_names)
            interactions.extend(ranging_interactions)

        elif market_regime == "volatile":
        # Volatile market interactions
            volatile_interactions = self._create_volatile_interactions(features, feature_names)
            interactions.extend(volatile_interactions)

        return np.column_stack(interactions) if interactions else:
    np.zeros((features.shape[0], 0))

    def _create_trending_interactions(self, features: np.ndarray, feature_names: List[str]) -> List[np.ndarray]:
        """
        Create interactions specific to trending markets.
        """
        interactions, []
        feature_map, {name: i for i, name in enumerate(feature_names)}

        # Trend - following interactions
        trend_features, ["SMA_Ratio", "EMA_Ratio", "MACD", "ADX"]
        momentum_features, ["RSI", "Stochastic", "CCI"]

        trend_indices, [feature_map.get(f) for f in trend_features if f in feature_map]
        momentum_indices, [feature_map.get(f) for f in momentum_features if f in feature_map]

        if trend_indices and momentum_indices: trend_avg, np.mean(features[:, trend_indices], axis, 1)
            momentum_avg, np.mean(features[:, momentum_indices], axis, 1)

            interactions.extend([
                trend_avg * momentum_avg * 1.5, # Trend × Momentum
                trend_avg / (momentum_avg + 1e - 8) * 1.3, # Trend / Momentum
                np.abs(trend_avg) * momentum_avg * 1.4  # Trend strength × Momentum
            ])

        return interactions

    def _create_ranging_interactions(self, features: np.ndarray, feature_names: List[str]) -> List[np.ndarray]:
        """
        Create interactions specific to ranging markets.
        """
        interactions, []
        feature_map, {name: i for i, name in enumerate(feature_names)}

        # Range - trading interactions
        oscillator_features, ["RSI", "Stochastic", "Williams_R", "CCI"]
        volume_features, ["Volume_Ratio", "OBV_Normalized", "MFI"]

        oscillator_indices, [feature_map.get(f) for f in oscillator_features if f in feature_map]
        volume_indices, [feature_map.get(f) for f in volume_features if f in feature_map]

        if oscillator_indices and volume_indices: oscillator_avg, np.mean(features[:, oscillator_indices], axis, 1)
            volume_avg, np.mean(features[:, volume_indices], axis, 1)

            interactions.extend([
                oscillator_avg * volume_avg * 1.6, # Oscillator × Volume
                oscillator_avg / (volume_avg + 1e - 8) * 1.4, # Oscillator / Volume
                np.std(features[:, oscillator_indices], axis, 1) * volume_avg * 1.5  # Oscillator divergence × Volume
            ])

        return interactions

    def _create_volatile_interactions(self, features: np.ndarray, feature_names: List[str]) -> List[np.ndarray]:
        """
        Create interactions specific to volatile markets.
        """
        interactions, []
        feature_map, {name: i for i, name in enumerate(feature_names)}

        # Volatility - focused interactions
        volatility_features, ["ATR_Normalized", "BB_Squeeze", "Volatility"]
        risk_features, ["RSI", "Stochastic", "Williams_R"]

        volatility_indices, [feature_map.get(f) for f in volatility_features if f in feature_map]
        risk_indices, [feature_map.get(f) for f in risk_features if f in feature_map]

        if volatility_indices and risk_indices: volatility_avg, np.mean(features[:, volatility_indices], axis, 1)
            risk_avg, np.mean(features[:, risk_indices], axis, 1)

            interactions.extend([
                volatility_avg * risk_avg * 1.8, # Volatility × Risk
                volatility_avg / (risk_avg + 1e - 8) * 1.6, # Volatility / Risk
                np.square(volatility_avg) * risk_avg * 1.7  # Volatility² × Risk
            ])

        return interactions

    def _create_cross_timeframe_interactions(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Create cross - timeframe interactions.
        """
        interactions, []
        feature_map, {name: i for i, name in enumerate(feature_names)}

        # Define timeframe pairs
        timeframe_pairs, [
            ("RSI_14", "RSI_30"),
            ("MACD_12_26", "MACD_20_40"),
            ("SMA_20", "SMA_50"),
            ("EMA_12", "EMA_26")
        ]

        for short_feature, long_feature in timeframe_pairs:
        if short_feature in feature_map and long_feature in feature_map: short_idx, long_idx, feature_map[short_feature], feature_map[long_feature]

        # Create cross - timeframe interactions
                interactions.extend([
                    features[:, short_idx] - features[:, long_idx],  # Divergence
                    features[:, short_idx] / (features[:, long_idx] + 1e - 8),  # Ratio
                    features[:, short_idx] * features[:, long_idx],  # Product
                    np.abs(features[:, short_idx] - features[:, long_idx])  # Absolute divergence
                ])

        return np.column_stack(interactions) if interactions else:
    np.zeros((features.shape[0], 0))

    def _identify_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Identify current market regime.
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Calculate regime indicators
            volatility, market_data['close'].pct_change().rolling(20).std().iloc[-1]
            trend_strength, abs(market_data['close'].rolling(20).mean().iloc[-1] -
                               market_data['close'].rolling(50).mean().iloc[-1]) / market_data['close'].iloc[-1]

        if volatility > 0.03:
        return "volatile"
            elif trend_strength > 0.02:
        return "trending"
            else:
        return "ranging"

        except Exception as e:
    self.logger.warning(f"Market regime identification failed: {e}")
        return "ranging"  # Default to ranging

    def _select_optimal_interactions(self, interactions: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """
        Select optimal interactions based on importance and correlation.
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Create dummy target for feature selection (in real implementation, use actual target)
            dummy_target = np.random.choice([0, 1], size, interactions.shape[0])

        # Calculate mutual information
            mi_scores = mutual_info_classif(interactions, dummy_target = random_state = 42)

        # Select interactions based on mutual information
            mi_threshold = self.selection_params["mutual_info_threshold"]
            important_indices, np.where(mi_scores > mi_threshold)[0]

        # Limit number of interactions
            max_interactions, self.selection_params["max_interactions"]
        if len(important_indices) > max_interactions:
        # Select top interactions by mutual information
                top_indices, np.argsort(mi_scores)[-max_interactions:]
                selected_interactions, interactions[:, top_indices]
            else: selected_interactions, interactions[:, important_indices]

        # Store selection history
        self.selected_interactions_history.append({
                "timestamp": datetime.now(),
                "n_interactions": selected_interactions.shape[1],
                "mi_scores": mi_scores[important_indices] if len(important_indices) > 0 else []
            })

        return selected_interactions

        except Exception as e:
    self.logger.error(f"Interaction selection failed: {e}")
        return interactions[:, :50]  # Return first 50 interactions as fallback

    def get_interaction_summary(self) -> Dict[str, Any]:
        """
        Get summary of interaction engineering results.
        """
        return {
            "interaction_patterns": self.interaction_patterns = "selection_params": self.selection_params,
            "performance_history": self.interaction_performance = "selected_interactions_count": len(self.selected_interactions_history) = "is_fitted": self.is_fitted = "scaler_params": {
                "mean": self.scaler.mean_.tolist() if self.is_fitted else:
    None, "scale": self.scaler.scale_.tolist() if self.is_fitted else:
    None
            }
        }

    def update_performance(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update interaction performance tracking.
        """
        self.interaction_performance[datetime.now()] = performance_metrics

    def get_feature_importance(self = interactions: np.ndarray = target: np.ndarray) -> np.ndarray:
        """
        Calculate importance of interaction features.
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Calculate mutual information for interaction importance
            mi_scores = mutual_info_classif(interactions, target, random_state, 42)

        # Store importance history
        self.feature_importance_history.append({
                "timestamp": datetime.now(), "importance_scores": mi_scores.tolist(),
                "mean_importance": np.mean(mi_scores),
                "max_importance": np.max(mi_scores)
            })

        return mi_scores

        except Exception as e:
    self.logger.error(f"Feature importance calculation failed: {e}")
        return np.ones(interactions.shape[1])  # Return uniform importance as fallback