# src/training/diverse_lookback_optimizer.py

"""
Diverse Lookback Period Optimizer

This module specializes in finding 2-3 lookback periods for each feature that deliver
meaningful yet significantly different information. It focuses on:
    pass  # TODO: Add implementation
1. Information diversity (different market insights)
2. Meaningful signal strength
3. Low correlation between selected periods
4. Complementary information content
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import shap
from sklearn.ensemble import RandomForestRegressor

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors


class DiverseLookbackOptimizer:
    """
    Optimizer that finds diverse yet meaningful lookback periods for each feature.

    For each feature, it finds 2-3 lookback periods that:
    - Provide meaningful signal strength
    - Deliver significantly different information
    - Have low correlation with each other
    - Capture complementary market dynamics
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the diverse lookback optimizer."""
        self.config = config
        self.logger = system_logger.getChild("DiverseLookbackOptimizer")

        # Diverse lookback optimization settings
        self.diverse_config = config.get("diverse_lookback_optimization", {
            "target_periods_per_feature": 3,
            "min_periods_per_feature": 2,
            "max_periods_per_feature": 3,
            "diversity_threshold": 0.3,  # Minimum correlation difference
            "meaningful_threshold": 0.1,  # Minimum SHAP importance
            "correlation_threshold": 0.7,  # Maximum correlation between periods
            "information_diversity_weight": 0.4,
            "signal_strength_weight": 0.4,
            "correlation_penalty_weight": 0.2,
            "lookback_ranges": {
                "RSI": {"min": 5, "max": 50, "step": 2},
                "MACD_fast": {"min": 5, "max": 25, "step": 1},
                "MACD_slow": {"min": 20, "max": 40, "step": 2},
                "Bollinger_Bands": {"min": 10, "max": 50, "step": 2},
                "SMA_short": {"min": 3, "max": 20, "step": 1},
                "SMA_long": {"min": 20, "max": 100, "step": 5},
                "EMA_short": {"min": 3, "max": 20, "step": 1},
                "EMA_long": {"min": 20, "max": 100, "step": 5},
                "ATR": {"min": 5, "max": 30, "step": 1},
                "Stochastic_k": {"min": 5, "max": 30, "step": 1},
                "Stochastic_d": {"min": 3, "max": 10, "step": 1},
                "ADX": {"min": 5, "max": 30, "step": 1},
                "CCI": {"min": 5, "max": 30, "step": 1}
            }
        })

        self.logger.info("🎯 Diverse Lookback Optimizer initialized")

    @handle_errors(exceptions=(Exception,), default_return={})
    async def find_diverse_lookback_periods(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        regimes: Optional[pd.Series] = None,
        symbol: str = "UNKNOWN",
        exchange: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> dict[str, Any]:
        """
        Find diverse lookback periods for each feature.

        Args:
            data: Feature data
            target: Target variable
            regimes: HMM regime labels (optional)
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            Dictionary with diverse lookback periods for each feature
        """
        self.logger.info(f"🎯 Finding diverse lookback periods for {symbol} on {exchange}")

        results = {
            "optimization_timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "diverse_lookback_periods": {},
            "diversity_analysis": {},
            "information_content_analysis": {},
            "regime_specific_periods": {}
        }

        # 1. Find diverse lookback periods for each feature
        self.logger.info("🔍 Finding diverse lookback periods...")
        diverse_periods = await self._find_diverse_periods_for_all_features(data, target)
        results["diverse_lookback_periods"] = diverse_periods

        # 2. Analyze diversity and information content
        self.logger.info("📊 Analyzing diversity and information content...")
        diversity_analysis = await self._analyze_diversity_and_information(data, target, diverse_periods)
        results["diversity_analysis"] = diversity_analysis

        # 3. Analyze information content for each period
        self.logger.info("🧠 Analyzing information content...")
        info_analysis = await self._analyze_information_content(data, target, diverse_periods)
        results["information_content_analysis"] = info_analysis

        # 4. Regime-specific diverse periods (if regimes available)
        if regimes is not None and len(regimes.unique()) > 1:
            self.logger.info("🔄 Finding regime-specific diverse periods...")
            regime_periods = await self._find_regime_specific_diverse_periods(
                data, target, regimes, diverse_periods
            )
            results["regime_specific_periods"] = regime_periods

        # 5. Save results
        await self._save_diverse_lookback_results(results, symbol, exchange, timeframe)

        self.logger.info("✅ Diverse lookback period optimization completed")
        return results

    async def _find_diverse_periods_for_all_features(
        self,
        data: pd.DataFrame,
        target: pd.Series
    ) -> dict[str, Any]:
        """Find diverse lookback periods for all features."""

        diverse_periods = {}

        for feature_name, lookback_config in self.diverse_config["lookback_ranges"].items():
            self.logger.info(f"🔍 Finding diverse periods for {feature_name}...")

            # Generate lookback periods to test
            periods = list(range(
                lookback_config["min"],
                lookback_config["max"] + 1,
                lookback_config["step"]
            ))

            # Find diverse periods for this feature
            feature_periods = await self._find_diverse_periods_for_feature(
                data, target, feature_name, periods
            )

            diverse_periods[feature_name] = feature_periods

        return diverse_periods

    async def _find_diverse_periods_for_feature(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        feature_name: str,
        periods: List[int]
    ) -> dict[str, Any]:
        """Find diverse lookback periods for a specific feature."""

        # 1. Calculate feature values for all periods
        period_features = {}
        period_scores = []

        for period in periods:
            feature_values = self._calculate_feature_with_period(data, feature_name, period)
            if feature_values is not None:
                # Calculate information score
                info_score = await self._calculate_information_score(feature_values, target)

                period_features[period] = feature_values
                period_scores.append({
                    "period": period,
                    "feature_values": feature_values,
                    "information_score": info_score
                })

        # 2. Filter meaningful periods (above threshold)
        meaningful_periods = [
            score for score in period_scores
            if score["information_score"] >= self.diverse_config["meaningful_threshold"]
        ]

        if len(meaningful_periods) < self.diverse_config["min_periods_per_feature"]:
            # If not enough meaningful periods, take top periods
            meaningful_periods = sorted(period_scores, key=lambda x: x["information_score"], reverse=True)
            meaningful_periods = meaningful_periods[:self.diverse_config["min_periods_per_feature"]]

        # 3. Find diverse subset using greedy algorithm
        diverse_subset = self._select_diverse_subset(meaningful_periods)

        # 4. Analyze the selected periods
        selected_periods = [item["period"] for item in diverse_subset]
        selected_features = {period: period_features[period] for period in selected_periods}

        # Calculate diversity metrics
        diversity_metrics = self._calculate_period_diversity_metrics(selected_features, target)

        return {
            "selected_periods": selected_periods,
            "period_scores": diverse_subset,
            "diversity_metrics": diversity_metrics,
            "all_period_scores": period_scores,
            "meaningful_periods": len(meaningful_periods),
            "total_periods_tested": len(periods)
        }

    def _select_diverse_subset(self, meaningful_periods: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Select diverse subset using greedy algorithm."""

        target_count = min(
            self.diverse_config["target_periods_per_feature"],
            len(meaningful_periods)
        )

        if target_count == 0:
            return []

        # Start with the period with highest information score
        selected = [meaningful_periods[0]]
        remaining = meaningful_periods[1:]

        # Greedy selection: add periods that maximize diversity
        while len(selected) < target_count and remaining:
            best_candidate = None
            best_diversity_score = -1

            for candidate in remaining:
                # Calculate diversity score for this candidate
                candidate_set = selected + [candidate]
                diversity_score = self._calculate_set_diversity_score(candidate_set)

                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected

    def _calculate_set_diversity_score(self, period_set: List[dict[str, Any]]) -> float:
        """Calculate diversity score for a set of periods."""

        if len(period_set) < 2:
            return 0.0

        # Extract feature values
        feature_values_list = [item["feature_values"] for item in period_set]

        # Calculate correlation matrix
        feature_df = pd.DataFrame(feature_values_list).T
        correlation_matrix = feature_df.corr().abs()

        # Calculate diversity as average inverse correlation
        n_periods = len(period_set)
        total_diversity = 0.0
        count = 0

        for i in range(n_periods):
            for j in range(i + 1, n_periods):
                correlation = correlation_matrix.iloc[i, j]
                diversity = 1.0 - correlation
                total_diversity += diversity
                count += 1

        avg_diversity = total_diversity / count if count > 0 else 0.0

        # Add information score component
        avg_info_score = np.mean([item["information_score"] for item in period_set])

        # Combined score
        diversity_weight = self.diverse_config["information_diversity_weight"]
        info_weight = self.diverse_config["signal_strength_weight"]

        combined_score = diversity_weight * avg_diversity + info_weight * avg_info_score

        return combined_score

    async def _calculate_information_score(self, feature_values: pd.Series, target: pd.Series) -> float:
        """Calculate information score using SHAP importance."""

        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            # Prepare data
            X = feature_values.values.reshape(-1, 1)
            y = target.values

            # Remove NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]

            if len(X_clean) < 100:  # Need sufficient data
                return 0.0

            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_clean, y_clean)

            # Calculate SHAP values
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_clean)

            # Calculate importance as mean absolute SHAP value
            importance = np.mean(np.abs(shap_values))

            return float(importance)

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating information score: {e}")
            return 0.0

    def _calculate_feature_with_period(
        self,
        data: pd.DataFrame,
        feature_name: str,
        period: int
    ) -> Optional[pd.Series]:
        """Calculate feature with specific lookback period."""

        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            if feature_name == "RSI":
                return self._calculate_rsi(data['close'], period)
            elif feature_name == "MACD_fast":
                return self._calculate_ema(data['close'], period)
            elif feature_name == "MACD_slow":
                return self._calculate_ema(data['close'], period)
            elif feature_name == "Bollinger_Bands":
                return self._calculate_bollinger_position(data, period)
            elif feature_name == "SMA_short":
                return self._calculate_sma(data['close'], period)
            elif feature_name == "SMA_long":
                return self._calculate_sma(data['close'], period)
            elif feature_name == "EMA_short":
                return self._calculate_ema(data['close'], period)
            elif feature_name == "EMA_long":
                return self._calculate_ema(data['close'], period)
            elif feature_name == "ATR":
                return self._calculate_atr(data, period)
            elif feature_name == "Stochastic_k":
                return self._calculate_stochastic_k(data, period)
            elif feature_name == "Stochastic_d":
                return self._calculate_stochastic_d(data, period)
            elif feature_name == "ADX":
                return self._calculate_adx(data, period)
            elif feature_name == "CCI":
                return self._calculate_cci(data, period)
            else:
                self.logger.warning(f"⚠️ Unknown feature: {feature_name}")
                return None

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating {feature_name} with period {period}: {e}")
            return None

    def _calculate_period_diversity_metrics(
        self,
        selected_features: dict[int, pd.Series],
        target: pd.Series
    ) -> dict[str, float]:
        """Calculate diversity metrics for selected periods."""

        if len(selected_features) < 2:
            return {"diversity_score": 0.0, "avg_correlation": 1.0}

        # Calculate correlation matrix
        feature_df = pd.DataFrame(selected_features)
        correlation_matrix = feature_df.corr().abs()

        # Calculate average correlation (excluding diagonal)
        n_periods = len(selected_features)
        total_correlation = 0.0
        count = 0

        for i in range(n_periods):
            for j in range(i + 1, n_periods):
                correlation = correlation_matrix.iloc[i, j]
                total_correlation += correlation
                count += 1

        avg_correlation = total_correlation / count if count > 0 else 1.0
        diversity_score = 1.0 - avg_correlation

        return {
            "diversity_score": diversity_score,
            "avg_correlation": avg_correlation,
            "n_periods": n_periods
        }

    async def _analyze_diversity_and_information(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        diverse_periods: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze diversity and information content across all features."""

        analysis = {
            "overall_diversity_score": 0.0,
            "feature_diversity_scores": {},
            "information_content_summary": {},
            "correlation_analysis": {},
            "complementarity_analysis": {}
        }

        total_diversity = 0.0
        feature_count = 0

        for feature_name, feature_data in diverse_periods.items():
            # Feature diversity score
            diversity_score = feature_data["diversity_metrics"]["diversity_score"]
            analysis["feature_diversity_scores"][feature_name] = diversity_score

            total_diversity += diversity_score
            feature_count += 1

            # Information content analysis
            avg_info_score = np.mean([
                item["information_score"] for item in feature_data["period_scores"]
            ])
            analysis["information_content_summary"][feature_name] = {
                "avg_information_score": avg_info_score,
                "n_periods": len(feature_data["selected_periods"]),
                "periods": feature_data["selected_periods"]
            }

            # Correlation analysis between periods
            if len(feature_data["selected_periods"]) > 1:
                correlation_analysis = self._analyze_period_correlations(
                    data, target, feature_name, feature_data["selected_periods"]
                )
                analysis["correlation_analysis"][feature_name] = correlation_analysis

        # Overall diversity score
        if feature_count > 0:
            analysis["overall_diversity_score"] = total_diversity / feature_count

        return analysis

    def _analyze_period_correlations(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        feature_name: str,
        periods: List[int]
    ) -> dict[str, Any]:
        """Analyze correlations between different periods of the same feature."""

        correlations = {}

        for i, period1 in enumerate(periods):
            for j, period2 in enumerate(periods[i+1:], i+1):
                feature1 = self._calculate_feature_with_period(data, feature_name, period1)
                feature2 = self._calculate_feature_with_period(data, feature_name, period2)

                if feature1 is not None and feature2 is not None:
                    # Calculate correlation
                    correlation = feature1.corr(feature2)
                    correlations[f"{period1}_vs_{period2}"] = {
                        "correlation": correlation,
                        "abs_correlation": abs(correlation),
                        "diversity": 1.0 - abs(correlation)
                    }

        return correlations

    async def _analyze_information_content(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        diverse_periods: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze information content for each selected period."""

        analysis = {}

        for feature_name, feature_data in diverse_periods.items():
            feature_analysis = {}

            for period_score in feature_data["period_scores"]:
                period = period_score["period"]
                feature_values = period_score["feature_values"]

                # Detailed information analysis
                info_analysis = await self._analyze_period_information(
                    feature_values, target, feature_name, period
                )

                feature_analysis[period] = info_analysis

            analysis[feature_name] = feature_analysis

        return analysis

    async def _analyze_period_information(
        self,
        feature_values: pd.Series,
        target: pd.Series,
        feature_name: str,
        period: int
    ) -> dict[str, Any]:
        """Analyze information content for a specific period."""

        analysis = {
            "period": period,
            "feature_name": feature_name,
            "information_score": 0.0,
            "signal_strength": 0.0,
            "predictive_power": 0.0,
            "market_insight": ""
        }

        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            # Calculate information score
            info_score = await self._calculate_information_score(feature_values, target)
            analysis["information_score"] = info_score

            # Calculate signal strength (volatility of feature)
            signal_strength = feature_values.std()
            analysis["signal_strength"] = signal_strength

            # Calculate predictive power (correlation with future returns)
            predictive_power = abs(feature_values.corr(target))
            analysis["predictive_power"] = predictive_power

            # Determine market insight based on period length
            if period <= 10:
                insight = "Short-term momentum"
            elif period <= 20:
                insight = "Medium-term trend"
            else:
                insight = "Long-term trend"

            analysis["market_insight"] = insight

        except Exception as e:
            self.logger.warning(f"⚠️ Error analyzing period information: {e}")

        return analysis

    async def _find_regime_specific_diverse_periods(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        regimes: pd.Series,
        global_periods: dict[str, Any]
    ) -> dict[str, Any]:
        """Find regime-specific diverse periods."""

        regime_periods = {}

        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_data = data[regime_mask]
            regime_target = target[regime_mask]

            if len(regime_data) >= 100:  # Minimum sample requirement
                self.logger.info(f"🔄 Finding diverse periods for regime {regime}...")

                regime_specific = await self._find_diverse_periods_for_all_features(
                    regime_data, regime_target
                )

                regime_periods[f"regime_{regime}"] = regime_specific

        return regime_periods

    async def _save_diverse_lookback_results(
        self,
        results: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ):
        """Save diverse lookback results to file."""

        output_dir = Path("data/diverse_lookback_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{exchange}_{symbol}_{timeframe}_diverse_lookback_periods.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"💾 Saved diverse lookback results to {filepath}")

    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI with specific period."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA with specific period."""
        return prices.rolling(window=period).mean()

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA with specific period."""
        return prices.ewm(span=period).mean()

    def _calculate_bollinger_position(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Bollinger Bands position with specific period."""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        position = (data['close'] - lower) / (upper - lower)
        return position

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR with specific period."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def _calculate_stochastic_k(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic %K with specific period."""
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        k = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        return k

    def _calculate_stochastic_d(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic %D with specific period."""
        k = self._calculate_stochastic_k(data, period)
        d = k.rolling(window=3).mean()
        return d

    def _calculate_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX with specific period."""
        # Simplified ADX calculation
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Simplified directional movement
        dm_plus = (data['high'] - data['high'].shift()).where(
            (data['high'] - data['high'].shift()) > (data['low'].shift() - data['low']), 0
        )
        dm_minus = (data['low'].shift() - data['low']).where(
            (data['low'].shift() - data['low']) > (data['high'] - data['high'].shift()), 0
        )

        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)

        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_cci(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate CCI with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mad)
        return cci

    def get_diverse_lookback_periods(
        self,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """Load diverse lookback periods."""

        filepath = Path(f"data/diverse_lookback_optimization/{exchange}_{symbol}_{timeframe}_diverse_lookback_periods.json")

        if not filepath.exists():
            self.logger.warning(f"⚠️ No diverse lookback results found for {symbol} on {exchange}")
            return {}

        try:
            with open(filepath, 'r') as f:
                results = json.load(f)

            return results.get("diverse_lookback_periods", {})

        except Exception as e:
            self.logger.error(f"❌ Error loading diverse lookback results: {e}")
            return {}