"""
Enhanced Validation Framework for Timeframe Optimization

This module provides comprehensive validation of optimized timeframes
using statistical, economic, and market microstructure analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig

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

class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"

@dataclass
class ValidationResult:
    """Result of validation process."""
    validation_score: float
    statistical_metrics: Dict[str, float]
    economic_metrics: Dict[str, float]
    microstructure_metrics: Dict[str, float]
    cross_validation_metrics: Dict[str, float]
    overall_quality: str
    recommendations: List[str]
    timestamp: datetime

class EnhancedValidationFramework:
    """
    Enhanced validation framework for timeframe optimization.
    """

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE):
        """Initialize enhanced validation framework."""
        self.validation_level = validation_level
        self.logger = get_logger('EnhancedValidationFramework')

        # Validation thresholds
        self.thresholds = {
            'hit_rate': 0.55,
            'sharpe_ratio': 0.5,
            'information_ratio': 0.3,
            'max_drawdown': 0.2,
            'transaction_cost_ratio': 0.1,
            'liquidity_score': 0.7,
            'volatility_stability': 0.6
        }

        self.logger.info(f'🔧 Enhanced validation framework initialized with {validation_level.value} level')

    def validate_optimized_configuration(self,
                                       config: MultiHorizonConfig,
                                       market_data: pd.DataFrame,
                                       model_type: str = "analyst") -> ValidationResult:
        """
        Comprehensive validation of optimized configuration.

        Args:
            config: Optimized configuration to validate
            market_data: Market data for validation
            model_type: Type of model (analyst/tactician)

        Returns:
            ValidationResult with comprehensive validation metrics
        """
        self.logger.info(f'🔍 Starting comprehensive validation for {model_type} model')

        try:
            # Statistical validation
            statistical_metrics = self._statistical_validation(config, market_data)

            # Economic validation
            economic_metrics = self._economic_validation(config, market_data)

            # Market microstructure validation
            microstructure_metrics = self._microstructure_validation(config, market_data)

            # Cross-validation
            cross_validation_metrics = self._cross_validation(config, market_data)

            # Calculate overall validation score
            validation_score = self._calculate_overall_score(
                statistical_metrics, economic_metrics,
                microstructure_metrics, cross_validation_metrics
            )

            # Determine overall quality
            overall_quality = self._determine_quality(validation_score)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                statistical_metrics, economic_metrics,
                microstructure_metrics, cross_validation_metrics
            )

            result = ValidationResult(
                validation_score=validation_score,
                statistical_metrics=statistical_metrics,
                economic_metrics=economic_metrics,
                microstructure_metrics=microstructure_metrics,
                cross_validation_metrics=cross_validation_metrics,
                overall_quality=overall_quality,
                recommendations=recommendations,
                timestamp=datetime.now()
            )

            self.logger.info(f'✅ Validation completed - Score: {validation_score:.3f}, Quality: {overall_quality}')
            return result

        except Exception as e:
            self.logger.error(f'❌ Validation failed: {e}')
            raise RuntimeError(f"Validation failed: {e}")

    def _statistical_validation(self, config: MultiHorizonConfig, market_data: pd.DataFrame) -> Dict[str, float]:
        """Perform statistical validation of the configuration."""
        self.logger.info('   → Running statistical validation...')

        try:
            # Generate labels with configuration
            labels = self._generate_labels(config, market_data)

            # Calculate returns
            returns = market_data['close'].pct_change().dropna()

            # Information Coefficient (IC)
            ic_scores = []
            for horizon_name, horizon_value in config.time_horizons.items():
                if horizon_name in labels.columns:
                    ic = self._calculate_information_coefficient(labels[horizon_name], returns)
                    ic_scores.append(ic)

            avg_ic = np.mean(ic_scores) if ic_scores else 0.0

            # Signal-to-Noise Ratio (SNR)
            snr_scores = []
            for horizon_name, horizon_value in config.time_horizons.items():
                if horizon_name in labels.columns:
                    snr = self._calculate_signal_to_noise_ratio(labels[horizon_name], returns)
                    snr_scores.append(snr)

            avg_snr = np.mean(snr_scores) if snr_scores else 0.0

            # Hit Rate
            hit_rates = []
            for horizon_name, horizon_value in config.time_horizons.items():
                if horizon_name in labels.columns:
                    hit_rate = self._calculate_hit_rate(labels[horizon_name], returns, horizon_value)
                    hit_rates.append(hit_rate)

            avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0

            # Statistical significance
            significance_scores = []
            for horizon_name, horizon_value in config.time_horizons.items():
                if horizon_name in labels.columns:
                    significance = self._calculate_statistical_significance(labels[horizon_name], returns)
                    significance_scores.append(significance)

            avg_significance = np.mean(significance_scores) if significance_scores else 0.0

            return {
                'information_coefficient': avg_ic,
                'signal_to_noise_ratio': avg_snr,
                'hit_rate': avg_hit_rate,
                'statistical_significance': avg_significance,
                'overall_statistical_score': (avg_ic + avg_snr + avg_hit_rate + avg_significance) / 4
            }

        except Exception as e:
            self.logger.warning(f'⚠️ Statistical validation error: {e}')
            return {
                'information_coefficient': 0.0,
                'signal_to_noise_ratio': 0.0,
                'hit_rate': 0.0,
                'statistical_significance': 0.0,
                'overall_statistical_score': 0.0
            }

    def _economic_validation(self, config: MultiHorizonConfig, market_data: pd.DataFrame) -> Dict[str, float]:
        """Perform economic validation of the configuration."""
        self.logger.info('   → Running economic validation...')

        try:
            # Generate labels with configuration
            labels = self._generate_labels(config, market_data)

            # Calculate returns
            returns = market_data['close'].pct_change().dropna()

            # Transaction cost analysis
            transaction_cost_ratio = self._calculate_transaction_cost_ratio(config, returns)

            # Risk-adjusted returns
            sharpe_ratio = self._calculate_sharpe_ratio(returns)

            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(returns)

            # Information ratio
            information_ratio = self._calculate_information_ratio(labels, returns)

            # Economic significance
            economic_significance = self._calculate_economic_significance(
                transaction_cost_ratio, sharpe_ratio, max_drawdown, information_ratio
            )

            return {
                'transaction_cost_ratio': transaction_cost_ratio,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'information_ratio': information_ratio,
                'economic_significance': economic_significance,
                'overall_economic_score': (sharpe_ratio + (1 - max_drawdown) + information_ratio + (1 - transaction_cost_ratio)) / 4
            }

        except Exception as e:
            self.logger.warning(f'⚠️ Economic validation error: {e}')
            return {
                'transaction_cost_ratio': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'information_ratio': 0.0,
                'economic_significance': 0.0,
                'overall_economic_score': 0.0
            }

    def _microstructure_validation(self, config: MultiHorizonConfig, market_data: pd.DataFrame) -> Dict[str, float]:
        """Perform market microstructure validation."""
        self.logger.info('   → Running microstructure validation...')

        try:
            # Liquidity analysis
            liquidity_score = self._calculate_liquidity_score(market_data)

            # Volatility analysis
            volatility_stability = self._calculate_volatility_stability(market_data)

            # Market depth analysis
            market_depth_score = self._calculate_market_depth_score(market_data)

            # Spread impact analysis
            spread_impact = self._calculate_spread_impact(market_data)

            return {
                'liquidity_score': liquidity_score,
                'volatility_stability': volatility_stability,
                'market_depth_score': market_depth_score,
                'spread_impact': spread_impact,
                'overall_microstructure_score': (liquidity_score + volatility_stability + market_depth_score + (1 - spread_impact)) / 4
            }

        except Exception as e:
            self.logger.warning(f'⚠️ Microstructure validation error: {e}')
            return {
                'liquidity_score': 0.0,
                'volatility_stability': 0.0,
                'market_depth_score': 0.0,
                'spread_impact': 0.0,
                'overall_microstructure_score': 0.0
            }

    def _cross_validation(self, config: MultiHorizonConfig, market_data: pd.DataFrame) -> Dict[str, float]:
        """Perform cross-validation of the configuration."""
        self.logger.info('   → Running cross-validation...')

        try:
            # Time series cross-validation
            cv_scores = []
            tscv = TimeSeriesSplit(n_splits=5)

            for train_idx, test_idx in tscv.split(market_data):
                train_data = market_data.iloc[train_idx]
                test_data = market_data.iloc[test_idx]

                # Train on training data
                train_labels = self._generate_labels(config, train_data)

                # Test on validation data
                test_labels = self._generate_labels(config, test_data)

                # Calculate performance
                performance = self._calculate_cv_performance(train_labels, test_labels)
                cv_scores.append(performance)

            avg_cv_score = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Stability analysis
            stability_score = 1.0 - (cv_std / (avg_cv_score + 1e-9))

            return {
                'cross_validation_score': avg_cv_score,
                'cross_validation_std': cv_std,
                'stability_score': stability_score,
                'overall_cv_score': (avg_cv_score + stability_score) / 2
            }

        except Exception as e:
            self.logger.warning(f'⚠️ Cross-validation error: {e}')
            return {
                'cross_validation_score': 0.0,
                'cross_validation_std': 0.0,
                'stability_score': 0.0,
                'overall_cv_score': 0.0
            }

    def _generate_labels(self, config: MultiHorizonConfig, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate labels using the configuration."""
        # Simplified label generation for validation
        labels = pd.DataFrame(index=market_data.index)

        for horizon_name, horizon_value in config.time_horizons.items():
            # Generate simple labels based on future returns
            future_returns = market_data['close'].pct_change(horizon_value).shift(-horizon_value)
            labels[horizon_name] = future_returns

        return labels

    def _calculate_information_coefficient(self, labels: pd.Series, returns: pd.Series) -> float:
        """Calculate Information Coefficient."""
        try:
            # Align data
            common_index = labels.index.intersection(returns.index)
            if len(common_index) < 10:
                return 0.0

            aligned_labels = labels.loc[common_index]
            aligned_returns = returns.loc[common_index]

            # Calculate correlation
            correlation, _ = stats.pearsonr(aligned_labels.dropna(), aligned_returns.dropna())
            return abs(correlation) if not np.isnan(correlation) else 0.0

        except Exception:
            return 0.0

    def _calculate_signal_to_noise_ratio(self, labels: pd.Series, returns: pd.Series) -> float:
        """Calculate Signal-to-Noise Ratio."""
        try:
            # Align data
            common_index = labels.index.intersection(returns.index)
            if len(common_index) < 10:
                return 0.0

            aligned_labels = labels.loc[common_index]
            aligned_returns = returns.loc[common_index]

            # Calculate SNR
            signal_strength = np.std(aligned_labels.dropna())
            noise_level = np.std(aligned_returns.dropna())

            return signal_strength / (noise_level + 1e-9)

        except Exception:
            return 0.0

    def _calculate_hit_rate(self, labels: pd.Series, returns: pd.Series, horizon: int) -> float:
        """Calculate hit rate."""
        try:
            # Align data
            common_index = labels.index.intersection(returns.index)
            if len(common_index) < 10:
                return 0.0

            aligned_labels = labels.loc[common_index]
            aligned_returns = returns.loc[common_index]

            # Calculate hit rate
            hits = 0
            total = 0

            for i in range(len(aligned_labels) - horizon):
                if not pd.isna(aligned_labels.iloc[i]) and not pd.isna(aligned_returns.iloc[i + horizon]):
                    if aligned_labels.iloc[i] > 0 and aligned_returns.iloc[i + horizon] > 0:
                        hits += 1
                    elif aligned_labels.iloc[i] < 0 and aligned_returns.iloc[i + horizon] < 0:
                        hits += 1
                    total += 1

            return hits / total if total > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_statistical_significance(self, labels: pd.Series, returns: pd.Series) -> float:
        """Calculate statistical significance."""
        try:
            # Align data
            common_index = labels.index.intersection(returns.index)
            if len(common_index) < 10:
                return 0.0

            aligned_labels = labels.loc[common_index]
            aligned_returns = returns.loc[common_index]

            # Calculate t-test
            correlation, p_value = stats.pearsonr(aligned_labels.dropna(), aligned_returns.dropna())

            # Convert p-value to significance score
            significance_score = 1.0 - p_value if not np.isnan(p_value) else 0.0

            return max(0.0, min(1.0, significance_score))

        except Exception:
            return 0.0

    def _calculate_transaction_cost_ratio(self, config: MultiHorizonConfig, returns: pd.Series) -> float:
        """Calculate transaction cost ratio."""
        try:
            # Estimate transaction costs
            transaction_cost = config.transaction_cost
            avg_return = abs(returns.mean())

            return transaction_cost / (avg_return + 1e-9)

        except Exception:
            return 0.0

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            if returns.std() == 0:
                return 0.0

            return returns.mean() / returns.std()

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max

            return abs(drawdown.min())

        except Exception:
            return 0.0

    def _calculate_information_ratio(self, labels: pd.DataFrame, returns: pd.Series) -> float:
        """Calculate information ratio."""
        try:
            # Simplified information ratio calculation
            if returns.std() == 0:
                return 0.0

            return returns.mean() / returns.std()

        except Exception:
            return 0.0

    def _calculate_economic_significance(self, transaction_cost_ratio: float, sharpe_ratio: float,
                                       max_drawdown: float, information_ratio: float) -> float:
        """Calculate economic significance score."""
        try:
            # Weighted combination of economic metrics
            weights = {
                'transaction_cost': 0.2,
                'sharpe_ratio': 0.3,
                'max_drawdown': 0.3,
                'information_ratio': 0.2
            }

            score = (
                weights['transaction_cost'] * (1 - transaction_cost_ratio) +
                weights['sharpe_ratio'] * min(1.0, sharpe_ratio / 2.0) +
                weights['max_drawdown'] * (1 - max_drawdown) +
                weights['information_ratio'] * min(1.0, information_ratio / 1.0)
            )

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.0

    def _calculate_liquidity_score(self, market_data: pd.DataFrame) -> float:
        """Calculate liquidity score."""
        try:
            # Use volume as proxy for liquidity
            volume = market_data['volume']
            avg_volume = volume.mean()
            volume_std = volume.std()

            # Liquidity score based on volume characteristics
            liquidity_score = min(1.0, avg_volume / (volume_std + 1e-9))

            return max(0.0, min(1.0, liquidity_score))

        except Exception:
            return 0.0

    def _calculate_volatility_stability(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility stability."""
        try:
            # Calculate rolling volatility
            returns = market_data['close'].pct_change().dropna()
            rolling_vol = self._vectorbt_rolling_operation(returns, "std", 20)

            # Volatility stability based on coefficient of variation
            vol_mean = rolling_vol.mean()
            vol_std = rolling_vol.std()

            stability_score = 1.0 - (vol_std / (vol_mean + 1e-9))

            return max(0.0, min(1.0, stability_score))

        except Exception:
            return 0.0

    def _calculate_market_depth_score(self, market_data: pd.DataFrame) -> float:
        """Calculate market depth score."""
        try:
            # Use volume and price range as proxy for market depth
            volume = market_data['volume']
            price_range = (market_data['high'] - market_data['low']) / market_data['close']

            # Market depth score
            depth_score = (volume / volume.mean()) * (1 - price_range.mean())

            return max(0.0, min(1.0, depth_score))

        except Exception:
            return 0.0

    def _calculate_spread_impact(self, market_data: pd.DataFrame) -> float:
        """Calculate spread impact."""
        try:
            # Use price range as proxy for spread impact
            price_range = (market_data['high'] - market_data['low']) / market_data['close']

            return price_range.mean()

        except Exception:
            return 0.0

    def _calculate_cv_performance(self, train_labels: pd.DataFrame, test_labels: pd.DataFrame) -> float:
        """Calculate cross-validation performance."""
        try:
            # Simplified CV performance calculation
            train_performance = train_labels.mean().mean()
            test_performance = test_labels.mean().mean()

            # Performance ratio
            performance_ratio = test_performance / (train_performance + 1e-9)

            return max(0.0, min(1.0, performance_ratio))

        except Exception:
            return 0.0

    def _calculate_overall_score(self, statistical_metrics: Dict[str, float],
                               economic_metrics: Dict[str, float],
                               microstructure_metrics: Dict[str, float],
                               cross_validation_metrics: Dict[str, float]) -> float:
        """Calculate overall validation score."""
        try:
            # Weighted combination of all metrics
            weights = {
                'statistical': 0.3,
                'economic': 0.3,
                'microstructure': 0.2,
                'cross_validation': 0.2
            }

            overall_score = (
                weights['statistical'] * statistical_metrics.get('overall_statistical_score', 0.0) +
                weights['economic'] * economic_metrics.get('overall_economic_score', 0.0) +
                weights['microstructure'] * microstructure_metrics.get('overall_microstructure_score', 0.0) +
                weights['cross_validation'] * cross_validation_metrics.get('overall_cv_score', 0.0)
            )

            return max(0.0, min(1.0, overall_score))

        except Exception:
            return 0.0

    def _determine_quality(self, validation_score: float) -> str:
        """Determine overall quality based on validation score."""
        if validation_score >= 0.8:
            return "Excellent"
        elif validation_score >= 0.7:
            return "Good"
        elif validation_score >= 0.6:
            return "Fair"
        elif validation_score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"

    def _generate_recommendations(self, statistical_metrics: Dict[str, float],
                                economic_metrics: Dict[str, float],
                                microstructure_metrics: Dict[str, float],
                                cross_validation_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Statistical recommendations
        if statistical_metrics.get('information_coefficient', 0) < 0.05:
            recommendations.append("Consider improving feature engineering to increase information coefficient")

        if statistical_metrics.get('hit_rate', 0) < 0.55:
            recommendations.append("Optimize profit targets to improve hit rate")

        # Economic recommendations
        if economic_metrics.get('transaction_cost_ratio', 0) > 0.1:
            recommendations.append("Reduce transaction frequency or improve profit targets to lower cost ratio")

        if economic_metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("Improve risk-adjusted returns by optimizing timeframes")

        # Microstructure recommendations
        if microstructure_metrics.get('liquidity_score', 0) < 0.7:
            recommendations.append("Consider market liquidity when selecting timeframes")

        if microstructure_metrics.get('volatility_stability', 0) < 0.6:
            recommendations.append("Optimize for more stable volatility patterns")

        # Cross-validation recommendations
        if cross_validation_metrics.get('stability_score', 0) < 0.8:
            recommendations.append("Improve model stability across different time periods")

        return recommendations
