"""
Economic Relevance Analysis for Market Dimensions.

This module analyzes the economic relevance of discovered implicit market dimensions
by examining their direct and indirect influence on price action. It determines which
dimensions beyond volume and volatility have meaningful impact on price movements.

Key Research Questions:
1. Which implicit dimensions influence price action (directly or indirectly)?
2. How do dimensions support momentum vs mean reversion strategies?
3. Which dimensions modulate volatility and affect price dynamics?
4. Do dimensions provide predictive power for future price movements?
5. What is the economic significance of each dimension for trading?

Examples of Price Action Influence:
- High volume near Bollinger Bands → affects breakout probability
- Volatility clustering → affects momentum persistence  
- Correlation patterns → affect mean reversion strength
- Microstructure changes → affect execution and slippage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

from src.utils.logger import system_logger


class PriceActionInfluence(Enum):
    """Types of price action influence."""
    MOMENTUM_SUPPORT = "momentum_support"
    MEAN_REVERSION_CATALYST = "mean_reversion_catalyst"
    VOLATILITY_MODULATION = "volatility_modulation"
    BREAKOUT_PREDICTION = "breakout_prediction"
    TREND_PERSISTENCE = "trend_persistence"
    REVERSAL_SIGNAL = "reversal_signal"
    EXECUTION_IMPACT = "execution_impact"


@dataclass
class DimensionEconomicRelevance:
    """Economic relevance metrics for a market dimension."""
    dimension_name: str
    price_action_influences: Dict[PriceActionInfluence, float]
    overall_relevance_score: float
    statistical_significance: Dict[str, float]
    trading_applications: List[str]
    economic_interpretation: str
    feature_contributions: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dimension_name': self.dimension_name,
            'price_action_influences': {k.value: v for k, v in self.price_action_influences.items()},
            'overall_relevance_score': self.overall_relevance_score,
            'statistical_significance': self.statistical_significance,
            'trading_applications': self.trading_applications,
            'economic_interpretation': self.economic_interpretation,
            'feature_contributions': self.feature_contributions
        }


class DimensionEconomicRelevanceAnalyzer:
    """
    Analyzer for economic relevance of market dimensions.
    
    This class determines which implicit dimensions have meaningful influence
    on price action beyond the known volume and volatility effects.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild('DimensionEconomicRelevanceAnalyzer')
    
    def analyze_dimension_economic_relevance(self,
                                           market_data: pd.DataFrame,
                                           dimension_features: pd.DataFrame,
                                           dimension_name: str) -> DimensionEconomicRelevance:
        """
        Analyze the economic relevance of a specific market dimension.
        
        Args:
            market_data: OHLCV market data
            dimension_features: Features representing this dimension
            dimension_name: Name of the dimension being analyzed
            
        Returns:
            Economic relevance analysis for the dimension
        """
        self.logger.info(f"💰 Analyzing economic relevance of {dimension_name} dimension")
        
        # Calculate price action influences
        price_influences = {}
        
        # 1. Momentum Support Analysis
        price_influences[PriceActionInfluence.MOMENTUM_SUPPORT] = self._analyze_momentum_support(
            market_data, dimension_features
        )
        
        # 2. Mean Reversion Catalyst Analysis
        price_influences[PriceActionInfluence.MEAN_REVERSION_CATALYST] = self._analyze_mean_reversion_catalyst(
            market_data, dimension_features
        )
        
        # 3. Volatility Modulation Analysis
        price_influences[PriceActionInfluence.VOLATILITY_MODULATION] = self._analyze_volatility_modulation(
            market_data, dimension_features
        )
        
        # 4. Breakout Prediction Analysis
        price_influences[PriceActionInfluence.BREAKOUT_PREDICTION] = self._analyze_breakout_prediction(
            market_data, dimension_features
        )
        
        # 5. Trend Persistence Analysis
        price_influences[PriceActionInfluence.TREND_PERSISTENCE] = self._analyze_trend_persistence(
            market_data, dimension_features
        )
        
        # Calculate overall relevance score
        overall_score = np.mean(list(price_influences.values()))
        
        # Statistical significance tests
        statistical_significance = self._calculate_statistical_significance(
            market_data, dimension_features, price_influences
        )
        
        # Determine trading applications
        trading_applications = self._determine_trading_applications(price_influences)
        
        # Economic interpretation
        economic_interpretation = self._generate_economic_interpretation(
            dimension_name, price_influences, overall_score
        )
        
        # Feature contributions within dimension
        feature_contributions = self._analyze_feature_contributions(
            market_data, dimension_features
        )
        
        return DimensionEconomicRelevance(
            dimension_name=dimension_name,
            price_action_influences=price_influences,
            overall_relevance_score=overall_score,
            statistical_significance=statistical_significance,
            trading_applications=trading_applications,
            economic_interpretation=economic_interpretation,
            feature_contributions=feature_contributions
        )
    
    def _analyze_momentum_support(self, 
                                market_data: pd.DataFrame,
                                dimension_features: pd.DataFrame) -> float:
        """Analyze how dimension supports momentum strategies."""
        
        if 'close' not in market_data.columns:
            return 0.0
        
        # Calculate momentum indicators
        returns = market_data['close'].pct_change().fillna(0)
        momentum_5 = returns.rolling(5).mean()
        momentum_20 = returns.rolling(20).mean()
        
        # Create composite dimension signal using weighted aggregation (not simple mean)
        dimension_signal = self._create_weighted_dimension_signal(market_data, dimension_features)
        dimension_signal_normalized = (dimension_signal - dimension_signal.mean()) / dimension_signal.std()
        
        # Analyze momentum support patterns
        momentum_support_scores = []
        
        for lookback in [5, 10, 20]:
            for lookahead in [5, 10]:
                # Look at dimension signal strength and subsequent momentum
                for i in range(lookback, len(dimension_signal_normalized) - lookahead):
                    current_signal = dimension_signal_normalized.iloc[i]
                    current_momentum = momentum_5.iloc[i]
                    future_momentum = momentum_5.iloc[i + lookahead]
                    
                    # Check if strong dimension signal supports momentum continuation
                    if abs(current_signal) > 1.0 and abs(current_momentum) > 0.001:  # Strong signal and existing momentum
                        # Momentum continuation when dimension signal is strong
                        if (current_momentum > 0 and future_momentum > 0) or (current_momentum < 0 and future_momentum < 0):
                            momentum_support_scores.append(abs(current_signal))
        
        # Calculate momentum support strength
        if momentum_support_scores:
            momentum_support = np.mean(momentum_support_scores)
            # Normalize to 0-1 scale
            momentum_support = min(momentum_support / 2.0, 1.0)
        else:
            momentum_support = 0.0
        
        return float(momentum_support)
    
    def _analyze_mean_reversion_catalyst(self, 
                                       market_data: pd.DataFrame,
                                       dimension_features: pd.DataFrame) -> float:
        """Analyze how dimension acts as catalyst for mean reversion."""
        
        if 'close' not in market_data.columns:
            return 0.0
        
        prices = market_data['close']
        ma_20 = prices.rolling(20).mean()
        price_deviation = (prices - ma_20) / ma_20
        
        # Create composite dimension signal
        dimension_signal = dimension_features.mean(axis=1)
        dimension_signal_normalized = (dimension_signal - dimension_signal.mean()) / dimension_signal.std()
        
        # Analyze mean reversion catalyst patterns
        reversion_catalyst_scores = []
        
        for i in range(20, len(dimension_signal_normalized) - 10):
            current_signal = dimension_signal_normalized.iloc[i]
            current_deviation = price_deviation.iloc[i]
            
            # Look for strong dimension signal when price is deviated
            if abs(current_signal) > 1.0 and abs(current_deviation) > 0.02:
                # Check if dimension signal catalyzes mean reversion
                future_prices = prices.iloc[i+1:i+11]
                current_price = prices.iloc[i]
                target_price = ma_20.iloc[i]
                
                # Calculate reversion strength
                if current_deviation > 0:  # Price above mean
                    min_future = future_prices.min()
                    reversion = (current_price - min_future) / current_price
                else:  # Price below mean
                    max_future = future_prices.max()
                    reversion = (max_future - current_price) / current_price
                
                # Stronger dimension signal should correlate with stronger reversion
                reversion_catalyst_scores.append(abs(current_signal) * reversion)
        
        # Calculate catalyst strength
        if reversion_catalyst_scores:
            catalyst_strength = np.mean(reversion_catalyst_scores)
            catalyst_strength = min(catalyst_strength / 0.1, 1.0)  # Normalize
        else:
            catalyst_strength = 0.0
        
        return float(catalyst_strength)
    
    def _analyze_volatility_modulation(self, 
                                     market_data: pd.DataFrame,
                                     dimension_features: pd.DataFrame) -> float:
        """Analyze how dimension modulates volatility."""
        
        if 'close' not in market_data.columns:
            return 0.0
        
        # Calculate realized volatility
        returns = market_data['close'].pct_change().fillna(0)
        volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualized
        
        # Create composite dimension signal
        dimension_signal = dimension_features.mean(axis=1)
        
        # Analyze volatility modulation
        # Look at correlation between dimension signal and future volatility
        volatility_modulation_correlations = []
        
        for lag in [1, 5, 10]:
            # Correlation between current dimension signal and future volatility
            future_volatility = volatility.shift(-lag)
            correlation = dimension_signal.corr(future_volatility)
            if not np.isnan(correlation):
                volatility_modulation_correlations.append(abs(correlation))
        
        # Calculate modulation strength
        if volatility_modulation_correlations:
            modulation_strength = np.mean(volatility_modulation_correlations)
        else:
            modulation_strength = 0.0
        
        return float(modulation_strength)
    
    def _analyze_breakout_prediction(self, 
                                   market_data: pd.DataFrame,
                                   dimension_features: pd.DataFrame) -> float:
        """Analyze dimension's ability to predict breakouts."""
        
        if not all(col in market_data.columns for col in ['high', 'low', 'close']):
            return 0.0
        
        # Calculate Bollinger Bands
        prices = market_data['close']
        ma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        upper_band = ma_20 + 2 * std_20
        lower_band = ma_20 - 2 * std_20
        
        # Create composite dimension signal
        dimension_signal = dimension_features.mean(axis=1)
        dimension_signal_normalized = (dimension_signal - dimension_signal.mean()) / dimension_signal.std()
        
        # Analyze breakout prediction
        breakout_predictions = []
        
        for i in range(20, len(prices) - 5):
            current_price = prices.iloc[i]
            current_signal = dimension_signal_normalized.iloc[i]
            
            # Check if near bands
            near_upper = abs(current_price - upper_band.iloc[i]) / current_price < 0.01
            near_lower = abs(current_price - lower_band.iloc[i]) / current_price < 0.01
            
            if (near_upper or near_lower) and abs(current_signal) > 1.0:
                # Look for breakout in next 5 periods
                future_prices = prices.iloc[i+1:i+6]
                
                if near_upper:
                    breakout = any(future_prices > upper_band.iloc[i])
                else:
                    breakout = any(future_prices < lower_band.iloc[i])
                
                # Strong signal should predict breakouts
                if breakout:
                    breakout_predictions.append(abs(current_signal))
        
        # Calculate breakout prediction power
        if breakout_predictions:
            breakout_power = np.mean(breakout_predictions) / 2.0  # Normalize
            breakout_power = min(breakout_power, 1.0)
        else:
            breakout_power = 0.0
        
        return float(breakout_power)
    
    def _analyze_trend_persistence(self, 
                                 market_data: pd.DataFrame,
                                 dimension_features: pd.DataFrame) -> float:
        """Analyze how dimension affects trend persistence."""
        
        if 'close' not in market_data.columns:
            return 0.0
        
        # Calculate trend indicators
        prices = market_data['close']
        ma_50 = prices.rolling(50).mean()
        trend_direction = np.where(prices > ma_50, 1, -1)
        
        # Create composite dimension signal
        dimension_signal = dimension_features.mean(axis=1)
        dimension_signal_normalized = (dimension_signal - dimension_signal.mean()) / dimension_signal.std()
        
        # Analyze trend persistence
        persistence_scores = []
        
        for i in range(50, len(prices) - 20):
            current_trend = trend_direction[i]
            current_signal = dimension_signal_normalized.iloc[i]
            
            # Look at trend persistence over next 20 periods
            future_trends = trend_direction[i+1:i+21]
            same_trend_periods = np.sum(future_trends == current_trend)
            persistence_rate = same_trend_periods / 20
            
            # Strong dimension signal should correlate with trend persistence
            if abs(current_signal) > 1.0:
                persistence_scores.append(persistence_rate * abs(current_signal))
        
        # Calculate trend persistence influence
        if persistence_scores:
            persistence_influence = np.mean(persistence_scores) / 2.0  # Normalize
            persistence_influence = min(persistence_influence, 1.0)
        else:
            persistence_influence = 0.0
        
        return float(persistence_influence)
    
    def _calculate_statistical_significance(self,
                                          market_data: pd.DataFrame,
                                          dimension_features: pd.DataFrame,
                                          price_influences: Dict[PriceActionInfluence, float]) -> Dict[str, float]:
        """Calculate statistical significance of dimension's price action influence."""
        
        significance_tests = {}
        
        if 'close' not in market_data.columns:
            return significance_tests
        
        returns = market_data['close'].pct_change().fillna(0)
        dimension_signal = dimension_features.mean(axis=1)
        
        # Test 1: Correlation significance
        correlation = dimension_signal.corr(returns)
        n_obs = len(dimension_signal.dropna())
        if n_obs > 10:
            # T-test for correlation significance
            t_stat = correlation * np.sqrt((n_obs - 2) / (1 - correlation**2)) if abs(correlation) < 1 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - 2))
            significance_tests['correlation_pvalue'] = float(p_value)
            significance_tests['correlation_coefficient'] = float(correlation)
        
        # Test 2: Granger causality (simplified)
        try:
            # Simple lag correlation test
            lag_correlations = []
            for lag in [1, 5, 10]:
                lag_corr = dimension_signal.corr(returns.shift(-lag))
                if not np.isnan(lag_corr):
                    lag_correlations.append(abs(lag_corr))
            
            if lag_correlations:
                significance_tests['max_lag_correlation'] = float(max(lag_correlations))
                significance_tests['predictive_power'] = float(np.mean(lag_correlations))
        except:
            pass
        
        # Test 3: Regime-based significance
        if len(price_influences) > 0:
            avg_influence = np.mean(list(price_influences.values()))
            significance_tests['average_influence_score'] = float(avg_influence)
            significance_tests['economically_significant'] = avg_influence > 0.1
        
        return significance_tests
    
    def _determine_trading_applications(self, 
                                      price_influences: Dict[PriceActionInfluence, float]) -> List[str]:
        """Determine trading applications based on price action influences."""
        
        applications = []
        threshold = 0.2  # Minimum influence score for trading application
        
        if price_influences.get(PriceActionInfluence.MOMENTUM_SUPPORT, 0) > threshold:
            applications.append("Momentum strategy signal enhancement")
            applications.append("Trend following strategy optimization")
        
        if price_influences.get(PriceActionInfluence.MEAN_REVERSION_CATALYST, 0) > threshold:
            applications.append("Mean reversion strategy timing")
            applications.append("Contrarian strategy signal generation")
        
        if price_influences.get(PriceActionInfluence.VOLATILITY_MODULATION, 0) > threshold:
            applications.append("Volatility forecasting enhancement")
            applications.append("Risk management optimization")
        
        if price_influences.get(PriceActionInfluence.BREAKOUT_PREDICTION, 0) > threshold:
            applications.append("Breakout strategy timing")
            applications.append("Support/resistance level confirmation")
        
        if price_influences.get(PriceActionInfluence.TREND_PERSISTENCE, 0) > threshold:
            applications.append("Trend strength assessment")
            applications.append("Position sizing optimization")
        
        if not applications:
            applications.append("Limited direct trading applications")
        
        return applications
    
    def _generate_economic_interpretation(self,
                                        dimension_name: str,
                                        price_influences: Dict[PriceActionInfluence, float],
                                        overall_score: float) -> str:
        """Generate economic interpretation of dimension relevance."""
        
        if overall_score > 0.3:
            interpretation = f"{dimension_name} dimension shows strong economic relevance for price action"
        elif overall_score > 0.15:
            interpretation = f"{dimension_name} dimension shows moderate economic relevance"
        else:
            interpretation = f"{dimension_name} dimension shows limited economic relevance"
        
        # Add specific insights
        max_influence = max(price_influences.items(), key=lambda x: x[1])
        interpretation += f". Strongest influence: {max_influence[0].value} (score: {max_influence[1]:.3f})"
        
        return interpretation
    
    def _analyze_feature_contributions(self,
                                     market_data: pd.DataFrame,
                                     dimension_features: pd.DataFrame) -> Dict[str, float]:
        """Analyze individual feature contributions within dimension."""
        
        if 'close' not in market_data.columns:
            return {}
        
        returns = market_data['close'].pct_change().fillna(0)
        feature_contributions = {}
        
        for feature in dimension_features.columns:
            feature_data = dimension_features[feature]
            
            # Calculate predictive correlation with future returns
            correlations = []
            for lag in [1, 5, 10]:
                future_returns = returns.shift(-lag)
                corr = feature_data.corr(future_returns)
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            if correlations:
                feature_contributions[feature] = float(np.mean(correlations))
            else:
                feature_contributions[feature] = 0.0
        
        return feature_contributions
    
    def _create_weighted_dimension_signal(self, 
                                        market_data: pd.DataFrame,
                                        dimension_features: pd.DataFrame) -> pd.Series:
        """
        Create weighted dimension signal using PCA loadings or feature importance.
        
        Avoids signal dilution from equal-weight averaging of potentially
        noisy features within a dimension.
        """
        
        if len(dimension_features.columns) == 1:
            return dimension_features.iloc[:, 0]
        
        try:
            # Method 1: PCA-based weighting (preferred)
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(dimension_features.fillna(0))
            
            # Apply PCA to get first component (captures most variance)
            pca = PCA(n_components=1)
            first_component = pca.fit_transform(features_scaled)
            
            # Use PCA loadings as weights
            loadings = pca.components_[0]
            
            # Create weighted signal
            weighted_signal = np.zeros(len(dimension_features))
            for i, (feature, loading) in enumerate(zip(dimension_features.columns, loadings)):
                weighted_signal += loading * dimension_features[feature].fillna(0).values
            
            return pd.Series(weighted_signal, index=dimension_features.index)
            
        except Exception as e:
            self.logger.warning(f"PCA weighting failed: {e}, trying Lasso weighting")
            
            try:
                # Method 2: Lasso-based weighting (fallback)
                from sklearn.linear_model import LassoCV
                
                if 'close' in market_data.columns:
                    # Use future returns as target for feature selection
                    returns = market_data['close'].pct_change().fillna(0)
                    target = returns.shift(-1).fillna(0)  # Next period return
                    
                    # Align target with features
                    min_len = min(len(dimension_features), len(target))
                    X = dimension_features.iloc[:min_len].fillna(0)
                    y = target.iloc[:min_len]
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply Lasso for feature selection
                    lasso = LassoCV(cv=3, random_state=42)
                    lasso.fit(X_scaled, y)
                    
                    # Use Lasso coefficients as weights
                    weights = np.abs(lasso.coef_)
                    
                    # Create weighted signal
                    weighted_signal = np.zeros(len(dimension_features))
                    for i, (feature, weight) in enumerate(zip(dimension_features.columns, weights)):
                        weighted_signal += weight * dimension_features[feature].fillna(0).values
                    
                    return pd.Series(weighted_signal, index=dimension_features.index)
                    
            except Exception as e2:
                self.logger.warning(f"Lasso weighting failed: {e2}, using equal weights")
                
                # Method 3: Equal weights (fallback)
                return dimension_features.mean(axis=1)


def analyze_all_dimensions_economic_relevance(market_data: pd.DataFrame,
                                            dimension_feature_groups: Dict[str, pd.DataFrame]) -> Dict[str, DimensionEconomicRelevance]:
    """
    Analyze economic relevance of all discovered dimensions.
    
    Args:
        market_data: OHLCV market data
        dimension_feature_groups: Dictionary mapping dimension names to their features
        
    Returns:
        Dictionary mapping dimension names to their economic relevance analysis
    """
    analyzer = DimensionEconomicRelevanceAnalyzer()
    results = {}
    
    for dimension_name, features in dimension_feature_groups.items():
        try:
            relevance = analyzer.analyze_dimension_economic_relevance(
                market_data, features, dimension_name
            )
            results[dimension_name] = relevance
        except Exception as e:
            analyzer.logger.error(f"Failed to analyze {dimension_name}: {e}")
            continue
    
    return results


def generate_economic_relevance_report(relevance_results: Dict[str, DimensionEconomicRelevance]) -> str:
    """Generate comprehensive economic relevance report."""
    
    report = []
    report.append("# Market Dimension Economic Relevance Analysis")
    report.append("=" * 60)
    report.append("")
    
    # Summary ranking
    dimensions_by_relevance = sorted(
        relevance_results.items(),
        key=lambda x: x[1].overall_relevance_score,
        reverse=True
    )
    
    report.append("## Dimension Relevance Ranking")
    report.append("")
    
    for i, (dim_name, relevance) in enumerate(dimensions_by_relevance, 1):
        status = "🟢" if relevance.overall_relevance_score > 0.3 else "🟡" if relevance.overall_relevance_score > 0.15 else "🔴"
        report.append(f"{i}. {status} **{dim_name.upper()}** - Score: {relevance.overall_relevance_score:.3f}")
        report.append(f"   - {relevance.economic_interpretation}")
        report.append("")
    
    # Detailed analysis
    report.append("## Detailed Price Action Influence Analysis")
    report.append("")
    
    for dim_name, relevance in dimensions_by_relevance:
        report.append(f"### {dim_name.upper()} Dimension")
        report.append("")
        
        # Price action influences
        report.append("**Price Action Influences:**")
        for influence, score in relevance.price_action_influences.items():
            status = "✅" if score > 0.2 else "⚠️" if score > 0.1 else "❌"
            report.append(f"- {status} {influence.value.replace('_', ' ').title()}: {score:.3f}")
        
        report.append("")
        
        # Trading applications
        report.append("**Trading Applications:**")
        for app in relevance.trading_applications:
            report.append(f"- {app}")
        
        report.append("")
        
        # Top contributing features
        if relevance.feature_contributions:
            top_features = sorted(relevance.feature_contributions.items(), key=lambda x: x[1], reverse=True)[:5]
            report.append("**Top Contributing Features:**")
            for feature, contribution in top_features:
                report.append(f"- {feature}: {contribution:.3f}")
        
        report.append("")
    
    # Beyond Volume and Volatility Analysis
    report.append("## Beyond Volume and Volatility: New Insights")
    report.append("")
    
    # Find dimensions other than volume and volatility with high relevance
    other_dimensions = [
        (name, relevance) for name, relevance in dimensions_by_relevance
        if 'volume' not in name.lower() and 'volatility' not in name.lower() and relevance.overall_relevance_score > 0.15
    ]
    
    if other_dimensions:
        report.append("**Dimensions Beyond Volume/Volatility with Economic Relevance:**")
        for name, relevance in other_dimensions:
            report.append(f"- **{name.upper()}**: {relevance.overall_relevance_score:.3f}")
            report.append(f"  - Key Influence: {max(relevance.price_action_influences.items(), key=lambda x: x[1])[0].value}")
            report.append(f"  - Trading Application: {relevance.trading_applications[0] if relevance.trading_applications else 'Limited'}")
            report.append("")
    else:
        report.append("❌ **No dimensions beyond volume/volatility show significant economic relevance**")
        report.append("- Consider expanding feature engineering to capture more market dynamics")
        report.append("- Focus on volume and volatility dimensions for regime-based strategies")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    high_relevance_dims = [name for name, rel in dimensions_by_relevance if rel.overall_relevance_score > 0.3]
    moderate_relevance_dims = [name for name, rel in dimensions_by_relevance if 0.15 < rel.overall_relevance_score <= 0.3]
    
    if high_relevance_dims:
        report.append(f"✅ **Focus on high-relevance dimensions**: {', '.join(high_relevance_dims)}")
        report.append("- Use these dimensions for regime identification and ML model training")
        report.append("")
    
    if moderate_relevance_dims:
        report.append(f"⚠️ **Consider moderate-relevance dimensions**: {', '.join(moderate_relevance_dims)}")
        report.append("- May provide additional signal when combined with high-relevance dimensions")
        report.append("")
    
    # Specific trading strategy recommendations
    momentum_dims = [name for name, rel in relevance_results.items() 
                    if rel.price_action_influences.get(PriceActionInfluence.MOMENTUM_SUPPORT, 0) > 0.2]
    reversion_dims = [name for name, rel in relevance_results.items() 
                     if rel.price_action_influences.get(PriceActionInfluence.MEAN_REVERSION_CATALYST, 0) > 0.2]
    
    if momentum_dims:
        report.append(f"🚀 **Momentum Strategy Enhancement**: Use {', '.join(momentum_dims)} dimensions")
    
    if reversion_dims:
        report.append(f"🔄 **Mean Reversion Strategy Enhancement**: Use {', '.join(reversion_dims)} dimensions")
    
    return "\n".join(report)