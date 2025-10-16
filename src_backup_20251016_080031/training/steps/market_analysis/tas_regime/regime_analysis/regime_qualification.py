"""
Regime Qualification and Validation System

Production-ready regime qualification system for trading applications.
Validates detected regimes for trading viability and economic significance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats
from scipy.stats import kstest, jarque_bera
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RegimeQualificationConfig:
    """Configuration for regime qualification."""
    
    # Qualification criteria
    min_regime_duration: int = 50
    min_volatility: float = 0.01
    max_volatility: float = 0.5
    min_trend_strength: float = 0.05
    min_economic_significance: float = 0.1
    
    # Statistical tests
    enable_normality_tests: bool = True
    enable_stationarity_tests: bool = True
    enable_autocorrelation_tests: bool = True
    significance_level: float = 0.05
    
    # Trading viability
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.2
    min_win_rate: float = 0.4
    min_profit_factor: float = 1.2
    
    # Regime stability
    stability_threshold: float = 0.7
    persistence_threshold: float = 0.6
    transition_probability_threshold: float = 0.1
    
    # Economic significance
    min_price_movement: float = 0.02
    min_volume_ratio: float = 0.8
    min_liquidity_score: float = 0.5

class RegimeQualifier:
    """
    Production-ready regime qualification system.
    
    Validates detected regimes for trading viability, economic significance,
    and statistical robustness.
    """
    
    def __init__(self, config: RegimeQualificationConfig):
        """Initialize regime qualifier.
        
        Args:
            config: Regime qualification configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Qualification state
        self.qualified_regimes = {}
        self.qualification_history = []
        self.regime_scores = {}
        
        self.logger.info("✅ Regime Qualifier initialized")
    
    def qualify_regimes(self, 
                       regime_results: Dict[str, Any],
                       market_data: pd.DataFrame,
                       timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Qualify detected regimes for trading viability.
        
        Args:
            regime_results: Results from regime detection
            market_data: Market data used for detection
            timestamps: Optional timestamps
            
        Returns:
            Qualification results
        """
        self.logger.info("🔍 Starting regime qualification")
        start_time = datetime.now()
        
        try:
            regimes = regime_results.get('regimes', {})
            regime_labels = regime_results.get('regime_labels', np.array([]))
            
            if not regimes:
                self.logger.warning("⚠️ No regimes to qualify")
                return self._create_empty_qualification_result()
            
            # Qualify each regime
            qualified_regimes = {}
            qualification_scores = {}
            
            for regime_name, regime_info in regimes.items():
                regime_id = regime_info.get('regime_id', 0)
                
                # Extract regime data
                regime_mask = regime_labels == regime_id
                regime_data = market_data[regime_mask]
                
                if len(regime_data) == 0:
                    continue
                
                # Perform qualification tests
                qualification_result = self._qualify_single_regime(
                    regime_name, regime_info, regime_data, market_data
                )
                
                if qualification_result['qualified']:
                    qualified_regimes[regime_name] = regime_info
                    qualification_scores[regime_name] = qualification_result['overall_score']
                
                # Store qualification history
                self.qualification_history.append({
                    'regime_name': regime_name,
                    'qualified': qualification_result['qualified'],
                    'score': qualification_result['overall_score'],
                    'timestamp': datetime.now().isoformat()
                })
            
            # Calculate qualification statistics
            qualification_stats = self._calculate_qualification_statistics(
                qualified_regimes, qualification_scores
            )
            
            # Create comprehensive results
            results = {
                'qualified_regimes': qualified_regimes,
                'qualification_scores': qualification_scores,
                'qualification_statistics': qualification_stats,
                'n_qualified': len(qualified_regimes),
                'n_total': len(regimes),
                'qualification_rate': len(qualified_regimes) / len(regimes) if regimes else 0.0,
                'timestamp': datetime.now().isoformat(),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
            # Update state
            self.qualified_regimes = qualified_regimes
            self.regime_scores = qualification_scores
            
            self.logger.info(f"✅ Regime qualification completed in {results['execution_time']:.2f}s")
            self.logger.info(f"📊 Qualified {len(qualified_regimes)}/{len(regimes)} regimes")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Regime qualification failed: {e}")
            raise
    
    def _qualify_single_regime(self, 
                              regime_name: str,
                              regime_info: Dict[str, Any],
                              regime_data: pd.DataFrame,
                              full_market_data: pd.DataFrame) -> Dict[str, Any]:
        """Qualify a single regime."""
        qualification_tests = {}
        
        # 1. Duration test
        duration_test = self._test_regime_duration(regime_info)
        qualification_tests['duration'] = duration_test
        
        # 2. Volatility test
        volatility_test = self._test_regime_volatility(regime_info, regime_data)
        qualification_tests['volatility'] = volatility_test
        
        # 3. Trend test
        trend_test = self._test_regime_trend(regime_info, regime_data)
        qualification_tests['trend'] = trend_test
        
        # 4. Economic significance test
        economic_test = self._test_economic_significance(regime_info, regime_data, full_market_data)
        qualification_tests['economic'] = economic_test
        
        # 5. Statistical tests
        if self.config.enable_normality_tests:
            normality_test = self._test_normality(regime_data)
            qualification_tests['normality'] = normality_test
        
        if self.config.enable_stationarity_tests:
            stationarity_test = self._test_stationarity(regime_data)
            qualification_tests['stationarity'] = stationarity_test
        
        if self.config.enable_autocorrelation_tests:
            autocorr_test = self._test_autocorrelation(regime_data)
            qualification_tests['autocorrelation'] = autocorr_test
        
        # 6. Trading viability test
        trading_test = self._test_trading_viability(regime_info, regime_data)
        qualification_tests['trading'] = trading_test
        
        # 7. Regime stability test
        stability_test = self._test_regime_stability(regime_info, regime_data)
        qualification_tests['stability'] = stability_test
        
        # Calculate overall qualification
        overall_score = self._calculate_overall_score(qualification_tests)
        qualified = overall_score >= 0.6  # Minimum qualification threshold
        
        return {
            'qualified': qualified,
            'overall_score': overall_score,
            'tests': qualification_tests,
            'regime_name': regime_name
        }
    
    def _test_regime_duration(self, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test if regime duration meets minimum requirements."""
        duration = regime_info.get('duration', 0)
        min_duration = self.config.min_regime_duration
        
        passed = duration >= min_duration
        score = min(duration / min_duration, 1.0)
        
        return {
            'passed': passed,
            'score': score,
            'duration': duration,
            'min_required': min_duration,
            'test_name': 'duration'
        }
    
    def _test_regime_volatility(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Test if regime volatility is within acceptable range."""
        volatility = regime_info.get('price_volatility', 0)
        min_vol = self.config.min_volatility
        max_vol = self.config.max_volatility
        
        passed = min_vol <= volatility <= max_vol
        
        # Score based on distance from optimal volatility (middle of range)
        optimal_vol = (min_vol + max_vol) / 2
        distance = abs(volatility - optimal_vol)
        max_distance = max(volatility - min_vol, max_vol - volatility)
        score = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        
        return {
            'passed': passed,
            'score': score,
            'volatility': volatility,
            'min_required': min_vol,
            'max_allowed': max_vol,
            'test_name': 'volatility'
        }
    
    def _test_regime_trend(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Test if regime has sufficient trend strength."""
        trend = abs(regime_info.get('price_trend', 0))
        min_trend = self.config.min_trend_strength
        
        passed = trend >= min_trend
        score = min(trend / min_trend, 1.0)
        
        return {
            'passed': passed,
            'score': score,
            'trend_strength': trend,
            'min_required': min_trend,
            'test_name': 'trend'
        }
    
    def _test_economic_significance(self, 
                                   regime_info: Dict[str, Any],
                                   regime_data: pd.DataFrame,
                                   full_market_data: pd.DataFrame) -> Dict[str, Any]:
        """Test economic significance of the regime."""
        # Price movement significance
        price_movement = abs(regime_info.get('price_trend', 0))
        min_price_movement = self.config.min_price_movement
        
        # Volume significance
        volume_ratio = regime_info.get('volume_ratio', 1.0)
        min_volume_ratio = self.config.min_volume_ratio
        
        # Additional economic significance tests
        economic_tests = self._perform_comprehensive_economic_tests(regime_info, regime_data, full_market_data)
        
        # Combined economic significance
        price_significant = price_movement >= min_price_movement
        volume_significant = volume_ratio >= min_volume_ratio
        
        # Additional economic significance checks
        liquidity_significant = economic_tests.get('liquidity_score', 0) >= self.config.min_liquidity_score
        market_impact_significant = economic_tests.get('market_impact_score', 0) >= 0.5
        
        passed = (price_significant and volume_significant and 
                 liquidity_significant and market_impact_significant)
        
        # Calculate comprehensive score
        base_score = (min(price_movement / min_price_movement, 1.0) + 
                     min(volume_ratio / min_volume_ratio, 1.0)) / 2.0
        
        economic_score = (base_score + 
                        economic_tests.get('liquidity_score', 0) + 
                        economic_tests.get('market_impact_score', 0)) / 3.0
        
        return {
            'passed': passed,
            'score': economic_score,
            'price_movement': price_movement,
            'volume_ratio': volume_ratio,
            'min_price_movement': min_price_movement,
            'min_volume_ratio': min_volume_ratio,
            'economic_tests': economic_tests,
            'test_name': 'economic_significance'
        }
    
    def _perform_comprehensive_economic_tests(self, 
                                             regime_info: Dict[str, Any],
                                             regime_data: pd.DataFrame,
                                             full_market_data: pd.DataFrame) -> Dict[str, float]:
        """Perform comprehensive economic significance tests."""
        tests = {}
        
        # 1. Liquidity test
        tests['liquidity_score'] = self._test_liquidity_significance(regime_data)
        
        # 2. Market impact test
        tests['market_impact_score'] = self._test_market_impact_significance(regime_data, full_market_data)
        
        # 3. Volatility significance test
        tests['volatility_significance'] = self._test_volatility_significance(regime_info)
        
        # 4. Trend strength test
        tests['trend_strength'] = self._test_trend_strength_significance(regime_info)
        
        # 5. Volume significance test
        tests['volume_significance'] = self._test_volume_significance(regime_data)
        
        # 6. Price action significance test
        tests['price_action_significance'] = self._test_price_action_significance(regime_data)
        
        return tests
    
    def _test_liquidity_significance(self, regime_data: pd.DataFrame) -> float:
        """Test liquidity significance of the regime."""
        if 'volume' not in regime_data.columns:
            return 0.5  # Default score if no volume data
        
        volume = regime_data['volume'].values
        
        # Calculate liquidity metrics
        avg_volume = np.mean(volume)
        volume_consistency = 1.0 - (np.std(volume) / avg_volume) if avg_volume > 0 else 0
        
        # Volume trend
        volume_trend = (volume[-1] - volume[0]) / volume[0] if volume[0] > 0 else 0
        
        # Liquidity score (0-1)
        liquidity_score = (min(volume_consistency, 1.0) + 
                          min(abs(volume_trend), 1.0)) / 2.0
        
        return liquidity_score
    
    def _test_market_impact_significance(self, regime_data: pd.DataFrame, full_market_data: pd.DataFrame) -> float:
        """Test market impact significance of the regime."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        regime_returns = regime_data['close'].pct_change().dropna()
        market_returns = full_market_data['close'].pct_change().dropna()
        
        if len(regime_returns) == 0 or len(market_returns) == 0:
            return 0.5
        
        # Calculate correlation with market
        correlation = np.corrcoef(regime_returns, market_returns[:len(regime_returns)])[0, 1]
        correlation = 0 if np.isnan(correlation) else correlation
        
        # Calculate relative volatility
        regime_vol = np.std(regime_returns)
        market_vol = np.std(market_returns)
        relative_vol = regime_vol / market_vol if market_vol > 0 else 1.0
        
        # Market impact score
        impact_score = (abs(correlation) + min(relative_vol, 2.0)) / 2.0
        
        return min(impact_score, 1.0)
    
    def _test_volatility_significance(self, regime_info: Dict[str, Any]) -> float:
        """Test volatility significance of the regime."""
        volatility = regime_info.get('price_volatility', 0)
        
        # Volatility should be significant but not extreme
        min_vol = 0.01  # 1%
        max_vol = 0.5   # 50%
        
        if volatility < min_vol:
            return 0.0  # Too low volatility
        elif volatility > max_vol:
            return 0.5  # Too high volatility
        else:
            # Optimal volatility range
            return 1.0
    
    def _test_trend_strength_significance(self, regime_info: Dict[str, Any]) -> float:
        """Test trend strength significance of the regime."""
        trend = abs(regime_info.get('price_trend', 0))
        
        # Trend should be significant
        min_trend = 0.02  # 2%
        max_trend = 0.5   # 50%
        
        if trend < min_trend:
            return 0.0  # No significant trend
        elif trend > max_trend:
            return 0.8  # Very strong trend (might be too extreme)
        else:
            # Good trend strength
            return min(trend / min_trend, 1.0)
    
    def _test_volume_significance(self, regime_data: pd.DataFrame) -> float:
        """Test volume significance of the regime."""
        if 'volume' not in regime_data.columns:
            return 0.5
        
        volume = regime_data['volume'].values
        
        # Volume should be consistent and significant
        avg_volume = np.mean(volume)
        volume_std = np.std(volume)
        volume_cv = volume_std / avg_volume if avg_volume > 0 else 1.0
        
        # Lower coefficient of variation is better
        volume_consistency = 1.0 - min(volume_cv, 1.0)
        
        return volume_consistency
    
    def _test_price_action_significance(self, regime_data: pd.DataFrame) -> float:
        """Test price action significance of the regime."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        close_prices = regime_data['close'].values
        
        # Calculate price action metrics
        price_range = np.max(close_prices) - np.min(close_prices)
        price_mean = np.mean(close_prices)
        price_range_pct = price_range / price_mean if price_mean > 0 else 0
        
        # Price action should show significant movement
        min_range = 0.01  # 1%
        max_range = 0.3   # 30%
        
        if price_range_pct < min_range:
            return 0.0  # No significant price action
        elif price_range_pct > max_range:
            return 0.7  # Very high price action (might be too volatile)
        else:
            return min(price_range_pct / min_range, 1.0)
    
    def _test_normality(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Test normality of regime returns."""
        if 'close' not in regime_data.columns:
            return {'passed': True, 'score': 1.0, 'test_name': 'normality'}
        
        close_prices = regime_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) < 3:
            return {'passed': True, 'score': 1.0, 'test_name': 'normality'}
        
        # Comprehensive normality tests
        normality_tests = self._perform_comprehensive_normality_tests(returns)
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = jarque_bera(returns)
            jb_passed = jb_pvalue > self.config.significance_level
            jb_score = jb_pvalue  # Higher p-value = more normal
        except:
            jb_passed = True
            jb_score = 1.0
            jb_stat = 0
        
        # Combined normality assessment
        passed = jb_passed and normality_tests.get('overall_normality', True)
        score = (jb_score + normality_tests.get('normality_score', 0.5)) / 2.0
        
        return {
            'passed': passed,
            'score': score,
            'jb_statistic': jb_stat,
            'jb_pvalue': jb_pvalue if 'jb_pvalue' in locals() else 1.0,
            'normality_tests': normality_tests,
            'test_name': 'normality'
        }
    
    def _perform_comprehensive_normality_tests(self, returns: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive normality tests."""
        tests = {}
        
        # 1. Jarque-Bera test
        try:
            jb_stat, jb_pvalue = jarque_bera(returns)
            tests['jarque_bera'] = {
                'statistic': jb_stat,
                'pvalue': jb_pvalue,
                'passed': jb_pvalue > self.config.significance_level
            }
        except:
            tests['jarque_bera'] = {'statistic': 0, 'pvalue': 1.0, 'passed': True}
        
        # 2. Kolmogorov-Smirnov test
        try:
            ks_stat, ks_pvalue = kstest(returns, 'norm', args=(np.mean(returns), np.std(returns)))
            tests['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'pvalue': ks_pvalue,
                'passed': ks_pvalue > self.config.significance_level
            }
        except:
            tests['kolmogorov_smirnov'] = {'statistic': 0, 'pvalue': 1.0, 'passed': True}
        
        # 3. Anderson-Darling test
        try:
            ad_stat, ad_critical, ad_significance = anderson(returns, dist='norm')
            tests['anderson_darling'] = {
                'statistic': ad_stat,
                'critical_values': ad_critical,
                'significance_levels': ad_significance,
                'passed': ad_stat < ad_critical[2]  # 5% significance level
            }
        except:
            tests['anderson_darling'] = {'statistic': 0, 'critical_values': [0, 0, 0], 'passed': True}
        
        # 4. Shapiro-Wilk test (for small samples)
        if len(returns) <= 5000:
            try:
                from scipy.stats import shapiro
                sw_stat, sw_pvalue = shapiro(returns)
                tests['shapiro_wilk'] = {
                    'statistic': sw_stat,
                    'pvalue': sw_pvalue,
                    'passed': sw_pvalue > self.config.significance_level
                }
            except:
                tests['shapiro_wilk'] = {'statistic': 0, 'pvalue': 1.0, 'passed': True}
        
        # 5. Visual normality tests
        tests['visual_tests'] = self._perform_visual_normality_tests(returns)
        
        # Overall normality assessment
        passed_tests = sum(1 for test in tests.values() if isinstance(test, dict) and test.get('passed', False))
        total_tests = len([test for test in tests.values() if isinstance(test, dict)])
        
        tests['overall_normality'] = passed_tests >= total_tests // 2
        tests['normality_score'] = passed_tests / total_tests if total_tests > 0 else 0.5
        
        return tests
    
    def _perform_visual_normality_tests(self, returns: np.ndarray) -> Dict[str, Any]:
        """Perform visual normality tests."""
        tests = {}
        
        # Skewness test
        skewness = stats.skew(returns)
        tests['skewness'] = {
            'value': skewness,
            'passed': abs(skewness) < 0.5,  # Close to 0 for normal distribution
            'score': max(0, 1.0 - abs(skewness))
        }
        
        # Kurtosis test
        kurtosis = stats.kurtosis(returns)
        tests['kurtosis'] = {
            'value': kurtosis,
            'passed': abs(kurtosis) < 0.5,  # Close to 0 for normal distribution
            'score': max(0, 1.0 - abs(kurtosis))
        }
        
        # Q-Q plot test (simplified)
        try:
            from scipy.stats import probplot

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
            _, r_value, _ = probplot(returns, dist="norm")
            tests['qq_plot'] = {
                'r_squared': r_value ** 2,
                'passed': r_value ** 2 > 0.95,
                'score': r_value ** 2
            }
        except:
            tests['qq_plot'] = {'r_squared': 0.5, 'passed': False, 'score': 0.5}
        
        return tests
    
    def _test_stationarity(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Test stationarity of regime data."""
        if 'close' not in regime_data.columns:
            return {'passed': True, 'score': 1.0, 'test_name': 'stationarity'}
        
        close_prices = regime_data['close'].values
        
        if len(close_prices) < 10:
            return {'passed': True, 'score': 1.0, 'test_name': 'stationarity'}
        
        # Simple stationarity test using rolling statistics
        window = min(10, len(close_prices) // 3)
        rolling_mean = pd.Series(close_prices).rolling(window).mean()
        rolling_std = pd.Series(close_prices).rolling(window).std()
        
        # Check if rolling statistics are relatively stable
        mean_stability = 1.0 - (rolling_mean.std() / rolling_mean.mean()) if rolling_mean.mean() != 0 else 1.0
        std_stability = 1.0 - (rolling_std.std() / rolling_std.mean()) if rolling_std.mean() != 0 else 1.0
        
        stability_score = (mean_stability + std_stability) / 2.0
        passed = stability_score >= 0.5  # Threshold for stationarity
        
        return {
            'passed': passed,
            'score': stability_score,
            'mean_stability': mean_stability,
            'std_stability': std_stability,
            'test_name': 'stationarity'
        }
    
    def _test_autocorrelation(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Test autocorrelation in regime data."""
        if 'close' not in regime_data.columns:
            return {'passed': True, 'score': 1.0, 'test_name': 'autocorrelation'}
        
        close_prices = regime_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) < 5:
            return {'passed': True, 'score': 1.0, 'test_name': 'autocorrelation'}
        
        # Calculate autocorrelation
        try:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            autocorr = 0 if np.isnan(autocorr) else autocorr
            
            # Low autocorrelation is good (closer to 0)
            autocorr_score = 1.0 - abs(autocorr)
            passed = autocorr_score >= 0.5
            
        except:
            autocorr = 0
            autocorr_score = 1.0
            passed = True
        
        return {
            'passed': passed,
            'score': autocorr_score,
            'autocorrelation': autocorr,
            'test_name': 'autocorrelation'
        }
    
    def _test_trading_viability(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Test trading viability of the regime."""
        # Calculate trading metrics
        if 'close' not in regime_data.columns:
            return {'passed': True, 'score': 1.0, 'test_name': 'trading'}
        
        close_prices = regime_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) == 0:
            return {'passed': True, 'score': 1.0, 'test_name': 'trading'}
        
        # Comprehensive trading viability tests
        trading_tests = self._perform_comprehensive_trading_tests(regime_info, regime_data, returns)
        
        # Basic trading metrics
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns) - 1
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.mean(returns > 0)
        
        # Profit factor (simplified)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        profit_factor = (np.sum(positive_returns) / abs(np.sum(negative_returns))) if len(negative_returns) > 0 else float('inf')
        
        # Test criteria
        sharpe_passed = sharpe_ratio >= self.config.min_sharpe_ratio
        drawdown_passed = abs(max_drawdown) <= self.config.max_drawdown_threshold
        winrate_passed = win_rate >= self.config.min_win_rate
        profit_passed = profit_factor >= self.config.min_profit_factor
        
        # Additional trading viability checks
        volatility_passed = trading_tests.get('volatility_viability', 0) >= 0.5
        liquidity_passed = trading_tests.get('liquidity_viability', 0) >= 0.5
        trend_passed = trading_tests.get('trend_viability', 0) >= 0.5
        
        passed = (sharpe_passed and drawdown_passed and winrate_passed and profit_passed and
                 volatility_passed and liquidity_passed and trend_passed)
        
        # Calculate comprehensive score
        base_score = (min(sharpe_ratio / self.config.min_sharpe_ratio, 1.0) +
                     (1.0 - abs(max_drawdown) / self.config.max_drawdown_threshold) +
                     min(win_rate / self.config.min_win_rate, 1.0) +
                     min(profit_factor / self.config.min_profit_factor, 1.0)) / 4.0
        
        trading_score = (base_score + 
                        trading_tests.get('volatility_viability', 0) +
                        trading_tests.get('liquidity_viability', 0) +
                        trading_tests.get('trend_viability', 0)) / 4.0
        
        return {
            'passed': passed,
            'score': trading_score,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trading_tests': trading_tests,
            'test_name': 'trading'
        }
    
    def _perform_comprehensive_trading_tests(self, 
                                           regime_info: Dict[str, Any],
                                           regime_data: pd.DataFrame,
                                           returns: np.ndarray) -> Dict[str, float]:
        """Perform comprehensive trading viability tests."""
        tests = {}
        
        # 1. Volatility viability
        tests['volatility_viability'] = self._test_volatility_trading_viability(regime_info, returns)
        
        # 2. Liquidity viability
        tests['liquidity_viability'] = self._test_liquidity_trading_viability(regime_data)
        
        # 3. Trend viability
        tests['trend_viability'] = self._test_trend_trading_viability(regime_info, returns)
        
        # 4. Risk-return viability
        tests['risk_return_viability'] = self._test_risk_return_viability(returns)
        
        # 5. Market efficiency viability
        tests['market_efficiency_viability'] = self._test_market_efficiency_viability(returns)
        
        # 6. Trading frequency viability
        tests['trading_frequency_viability'] = self._test_trading_frequency_viability(regime_data)
        
        return tests
    
    def _test_volatility_trading_viability(self, regime_info: Dict[str, Any], returns: np.ndarray) -> float:
        """Test volatility trading viability."""
        volatility = regime_info.get('price_volatility', 0)
        
        # Optimal volatility range for trading
        min_vol = 0.005  # 0.5%
        max_vol = 0.1    # 10%
        
        if volatility < min_vol:
            return 0.3  # Too low volatility (hard to trade)
        elif volatility > max_vol:
            return 0.4  # Too high volatility (too risky)
        else:
            # Optimal volatility range
            return 1.0
    
    def _test_liquidity_trading_viability(self, regime_data: pd.DataFrame) -> float:
        """Test liquidity trading viability."""
        if 'volume' not in regime_data.columns:
            return 0.5
        
        volume = regime_data['volume'].values
        
        # Volume consistency
        volume_cv = np.std(volume) / np.mean(volume) if np.mean(volume) > 0 else 1.0
        volume_consistency = 1.0 - min(volume_cv, 1.0)
        
        # Volume trend (increasing volume is good)
        volume_trend = (volume[-1] - volume[0]) / volume[0] if volume[0] > 0 else 0
        volume_trend_score = min(max(volume_trend, 0), 1.0)
        
        # Liquidity score
        liquidity_score = (volume_consistency + volume_trend_score) / 2.0
        
        return liquidity_score
    
    def _test_trend_trading_viability(self, regime_info: Dict[str, Any], returns: np.ndarray) -> float:
        """Test trend trading viability."""
        trend = abs(regime_info.get('price_trend', 0))
        
        # Trend should be significant but not too extreme
        min_trend = 0.01  # 1%
        max_trend = 0.3   # 30%
        
        if trend < min_trend:
            return 0.2  # No clear trend
        elif trend > max_trend:
            return 0.6  # Very strong trend (might be too extreme)
        else:
            # Good trend for trading
            return min(trend / min_trend, 1.0)
    
    def _test_risk_return_viability(self, returns: np.ndarray) -> float:
        """Test risk-return viability."""
        if len(returns) < 2:
            return 0.5
        
        # Calculate risk-return metrics
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Sharpe ratio
        sharpe = mean_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) if len(downside_returns) > 0 else volatility
        sortino = mean_return / downside_vol if downside_vol > 0 else 0
        
        # Risk-return score
        risk_return_score = (min(sharpe, 2.0) + min(sortino, 2.0)) / 4.0
        
        return min(risk_return_score, 1.0)
    
    def _test_market_efficiency_viability(self, returns: np.ndarray) -> float:
        """Test market efficiency viability."""
        if len(returns) < 10:
            return 0.5
        
        # Test for autocorrelation (efficiency indicator)
        autocorr_1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        autocorr_1 = 0 if np.isnan(autocorr_1) else autocorr_1
        
        # Test for momentum (inefficiency indicator)
        momentum = np.mean(returns[1:] * returns[:-1])
        
        # Efficiency score (lower autocorrelation and momentum = more efficient)
        efficiency_score = 1.0 - (abs(autocorr_1) + abs(momentum)) / 2.0
        
        return max(efficiency_score, 0.0)
    
    def _test_trading_frequency_viability(self, regime_data: pd.DataFrame) -> float:
        """Test trading frequency viability."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        close_prices = regime_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) == 0:
            return 0.5
        
        # Calculate trading frequency metrics
        n_periods = len(returns)
        n_significant_moves = np.sum(np.abs(returns) > 0.01)  # 1% moves
        
        # Trading frequency score
        frequency_score = min(n_significant_moves / n_periods, 1.0)
        
        return frequency_score
    
    def _test_regime_stability(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Test stability of the regime."""
        # Check if regime characteristics are consistent
        volatility = regime_info.get('price_volatility', 0)
        trend = regime_info.get('price_trend', 0)
        duration = regime_info.get('duration', 0)
        
        # Comprehensive stability tests
        stability_tests = self._perform_comprehensive_stability_tests(regime_info, regime_data)
        
        # Basic stability checks
        volatility_stable = 0.01 <= volatility <= 0.3  # Reasonable volatility range
        trend_stable = abs(trend) <= 0.5  # Not too extreme trend
        duration_stable = duration >= self.config.min_regime_duration
        
        # Additional stability checks
        persistence_stable = stability_tests.get('persistence_score', 0) >= self.config.persistence_threshold
        consistency_stable = stability_tests.get('consistency_score', 0) >= 0.5
        
        passed = (volatility_stable and trend_stable and duration_stable and
                 persistence_stable and consistency_stable)
        
        # Calculate comprehensive stability score
        base_score = (1.0 - abs(volatility - 0.1) / 0.3 if volatility <= 0.3 else 0.0 +
                     1.0 - abs(trend) / 0.5 if abs(trend) <= 0.5 else 0.0 +
                     min(duration / self.config.min_regime_duration, 1.0)) / 3.0
        
        stability_score = (base_score + 
                         stability_tests.get('persistence_score', 0) +
                         stability_tests.get('consistency_score', 0)) / 3.0
        
        return {
            'passed': passed,
            'score': stability_score,
            'volatility_stable': volatility_stable,
            'trend_stable': trend_stable,
            'duration_stable': duration_stable,
            'persistence_stable': persistence_stable,
            'consistency_stable': consistency_stable,
            'stability_tests': stability_tests,
            'test_name': 'stability'
        }
    
    def _perform_comprehensive_stability_tests(self, 
                                             regime_info: Dict[str, Any],
                                             regime_data: pd.DataFrame) -> Dict[str, float]:
        """Perform comprehensive regime stability tests."""
        tests = {}
        
        # 1. Persistence test
        tests['persistence_score'] = self._test_regime_persistence(regime_info, regime_data)
        
        # 2. Consistency test
        tests['consistency_score'] = self._test_regime_consistency(regime_info, regime_data)
        
        # 3. Structural stability test
        tests['structural_stability'] = self._test_structural_stability(regime_data)
        
        # 4. Temporal stability test
        tests['temporal_stability'] = self._test_temporal_stability(regime_data)
        
        # 5. Volatility stability test
        tests['volatility_stability'] = self._test_volatility_stability(regime_data)
        
        # 6. Trend stability test
        tests['trend_stability'] = self._test_trend_stability(regime_data)
        
        return tests
    
    def _test_regime_persistence(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> float:
        """Test regime persistence."""
        duration = regime_info.get('duration', 0)
        min_duration = self.config.min_regime_duration
        
        # Persistence based on duration
        duration_persistence = min(duration / min_duration, 1.0)
        
        # Persistence based on regime characteristics consistency
        if 'close' in regime_data.columns:
            close_prices = regime_data['close'].values
            
            # Calculate rolling volatility consistency
            window = min(20, len(close_prices) // 3)
            if window > 1:
                rolling_vol = pd.Series(close_prices).rolling(window).std()
                vol_consistency = 1.0 - (rolling_vol.std() / rolling_vol.mean()) if rolling_vol.mean() > 0 else 0
            else:
                vol_consistency = 0.5
            
            # Persistence score
            persistence_score = (duration_persistence + vol_consistency) / 2.0
        else:
            persistence_score = duration_persistence
        
        return persistence_score
    
    def _test_regime_consistency(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> float:
        """Test regime consistency."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        close_prices = regime_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) < 2:
            return 0.5
        
        # Consistency based on return characteristics
        return_consistency = 1.0 - (np.std(returns) / np.mean(np.abs(returns))) if np.mean(np.abs(returns)) > 0 else 0
        
        # Consistency based on price action
        price_consistency = 1.0 - (np.std(close_prices) / np.mean(close_prices)) if np.mean(close_prices) > 0 else 0
        
        # Overall consistency score
        consistency_score = (return_consistency + price_consistency) / 2.0
        
        return max(consistency_score, 0.0)
    
    def _test_structural_stability(self, regime_data: pd.DataFrame) -> float:
        """Test structural stability of the regime."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        close_prices = regime_data['close'].values
        
        # Test for structural breaks using rolling statistics
        window = min(10, len(close_prices) // 3)
        if window > 1:
            rolling_mean = pd.Series(close_prices).rolling(window).mean()
            rolling_std = pd.Series(close_prices).rolling(window).std()
            
            # Structural stability based on rolling statistics
            mean_stability = 1.0 - (rolling_mean.std() / rolling_mean.mean()) if rolling_mean.mean() > 0 else 0
            std_stability = 1.0 - (rolling_std.std() / rolling_std.mean()) if rolling_std.mean() > 0 else 0
            
            structural_score = (mean_stability + std_stability) / 2.0
        else:
            structural_score = 0.5
        
        return max(structural_score, 0.0)
    
    def _test_temporal_stability(self, regime_data: pd.DataFrame) -> float:
        """Test temporal stability of the regime."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        close_prices = regime_data['close'].values
        
        # Test for temporal patterns
        if len(close_prices) > 10:
            # Calculate autocorrelation
            returns = np.diff(close_prices) / close_prices[:-1]
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            autocorr = 0 if np.isnan(autocorr) else autocorr
            
            # Temporal stability (lower autocorrelation = more stable)
            temporal_score = 1.0 - abs(autocorr)
        else:
            temporal_score = 0.5
        
        return max(temporal_score, 0.0)
    
    def _test_volatility_stability(self, regime_data: pd.DataFrame) -> float:
        """Test volatility stability of the regime."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        close_prices = regime_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) < 2:
            return 0.5
        
        # Calculate rolling volatility
        window = min(5, len(returns) // 2)
        if window > 1:
            rolling_vol = pd.Series(returns).rolling(window).std()
            vol_stability = 1.0 - (rolling_vol.std() / rolling_vol.mean()) if rolling_vol.mean() > 0 else 0
        else:
            vol_stability = 0.5
        
        return max(vol_stability, 0.0)
    
    def _test_trend_stability(self, regime_data: pd.DataFrame) -> float:
        """Test trend stability of the regime."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        close_prices = regime_data['close'].values
        
        # Calculate trend stability
        if len(close_prices) > 5:
            # Simple trend calculation
            trend = (close_prices[-1] - close_prices[0]) / close_prices[0]
            
            # Trend consistency
            mid_point = len(close_prices) // 2
            first_half_trend = (close_prices[mid_point] - close_prices[0]) / close_prices[0]
            second_half_trend = (close_prices[-1] - close_prices[mid_point]) / close_prices[mid_point]
            
            # Trend consistency score
            trend_consistency = 1.0 - abs(first_half_trend - second_half_trend) / (abs(first_half_trend) + abs(second_half_trend) + 1e-8)
        else:
            trend_consistency = 0.5
        
        return max(trend_consistency, 0.0)
    
    def _calculate_overall_score(self, tests: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall qualification score."""
        if not tests:
            return 0.0
        
        # Weight different tests
        weights = {
            'duration': 0.10,
            'volatility': 0.10,
            'trend': 0.10,
            'economic': 0.20,
            'normality': 0.08,
            'stationarity': 0.08,
            'autocorrelation': 0.04,
            'trading': 0.20,
            'stability': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for test_name, test_result in tests.items():
            if test_name in weights:
                weight = weights[test_name]
                score = test_result.get('score', 0.0)
                weighted_score += weight * score
                total_weight += weight
        
        # Calculate comprehensive quality score
        base_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Additional quality factors
        quality_factors = self._calculate_quality_factors(tests)
        
        # Final score with quality adjustments
        final_score = base_score * quality_factors.get('quality_multiplier', 1.0)
        
        return min(final_score, 1.0)
    
    def _calculate_quality_factors(self, tests: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate additional quality factors for regime scoring."""
        factors = {}
        
        # 1. Consistency factor
        consistency_scores = [test.get('score', 0) for test in tests.values()]
        consistency_factor = 1.0 - np.std(consistency_scores) if consistency_scores else 1.0
        factors['consistency_factor'] = max(consistency_factor, 0.5)
        
        # 2. Completeness factor
        passed_tests = sum(1 for test in tests.values() if test.get('passed', False))
        total_tests = len(tests)
        completeness_factor = passed_tests / total_tests if total_tests > 0 else 0.5
        factors['completeness_factor'] = completeness_factor
        
        # 3. Robustness factor
        robustness_scores = []
        for test_name, test_result in tests.items():
            if 'tests' in test_result:  # Comprehensive tests
                sub_tests = test_result['tests']
                if isinstance(sub_tests, dict):
                    sub_scores = [sub_test.get('score', 0) for sub_test in sub_tests.values() if isinstance(sub_test, dict)]
                    if sub_scores:
                        robustness_scores.append(np.mean(sub_scores))
        
        robustness_factor = np.mean(robustness_scores) if robustness_scores else 0.5
        factors['robustness_factor'] = robustness_factor
        
        # 4. Economic significance factor
        economic_score = tests.get('economic', {}).get('score', 0.5)
        factors['economic_factor'] = economic_score
        
        # 5. Trading viability factor
        trading_score = tests.get('trading', {}).get('score', 0.5)
        factors['trading_factor'] = trading_score
        
        # Overall quality multiplier
        quality_multiplier = (
            factors['consistency_factor'] * 0.25 +
            factors['completeness_factor'] * 0.25 +
            factors['robustness_factor'] * 0.25 +
            factors['economic_factor'] * 0.15 +
            factors['trading_factor'] * 0.10
        )
        
        factors['quality_multiplier'] = quality_multiplier
        
        return factors
    
    def calculate_regime_quality_score(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive regime quality score."""
        # Perform all qualification tests
        qualification_result = self.qualify_regimes(
            {'regimes': {'test_regime': regime_info}}, 
            regime_data
        )
        
        if not qualification_result.get('qualified_regimes'):
            return {
                'quality_score': 0.0,
                'qualification_status': 'failed',
                'details': 'Regime failed basic qualification'
            }
        
        # Get qualification tests
        regime_tests = qualification_result.get('qualification_scores', {})
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(regime_info, regime_data)
        
        # Calculate final quality score
        base_score = regime_tests.get('test_regime', 0.5)
        quality_adjustments = self._calculate_quality_adjustments(quality_metrics)
        
        final_score = base_score * quality_adjustments
        
        return {
            'quality_score': final_score,
            'qualification_status': 'qualified' if final_score >= 0.6 else 'failed',
            'base_score': base_score,
            'quality_adjustments': quality_adjustments,
            'quality_metrics': quality_metrics,
            'details': self._generate_quality_report(quality_metrics, final_score)
        }
    
    def _calculate_quality_metrics(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate detailed quality metrics for regime scoring."""
        metrics = {}
        
        # 1. Statistical quality
        metrics['statistical_quality'] = self._calculate_statistical_quality(regime_data)
        
        # 2. Economic quality
        metrics['economic_quality'] = self._calculate_economic_quality(regime_info, regime_data)
        
        # 3. Trading quality
        metrics['trading_quality'] = self._calculate_trading_quality(regime_info, regime_data)
        
        # 4. Stability quality
        metrics['stability_quality'] = self._calculate_stability_quality(regime_info, regime_data)
        
        # 5. Persistence quality
        metrics['persistence_quality'] = self._calculate_persistence_quality(regime_info, regime_data)
        
        return metrics
    
    def _calculate_statistical_quality(self, regime_data: pd.DataFrame) -> float:
        """Calculate statistical quality of the regime."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        close_prices = regime_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) < 2:
            return 0.5
        
        # Statistical quality factors
        factors = []
        
        # 1. Normality
        try:
            jb_stat, jb_pvalue = jarque_bera(returns)
            normality_score = jb_pvalue  # Higher p-value = more normal
            factors.append(normality_score)
        except:
            factors.append(0.5)
        
        # 2. Stationarity
        try:
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(returns)
            stationarity_score = adf_pvalue  # Higher p-value = more stationary
            factors.append(stationarity_score)
        except:
            factors.append(0.5)
        
        # 3. Autocorrelation
        try:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            autocorr = 0 if np.isnan(autocorr) else autocorr
            autocorr_score = 1.0 - abs(autocorr)  # Lower autocorrelation = better
            factors.append(autocorr_score)
        except:
            factors.append(0.5)
        
        return np.mean(factors)
    
    def _calculate_economic_quality(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> float:
        """Calculate economic quality of the regime."""
        factors = []
        
        # 1. Price movement significance
        price_trend = abs(regime_info.get('price_trend', 0))
        price_significance = min(price_trend / 0.02, 1.0)  # 2% threshold
        factors.append(price_significance)
        
        # 2. Volatility appropriateness
        volatility = regime_info.get('price_volatility', 0)
        vol_appropriateness = 1.0 - abs(volatility - 0.05) / 0.05 if volatility <= 0.1 else 0.5
        factors.append(vol_appropriateness)
        
        # 3. Volume significance
        if 'volume' in regime_data.columns:
            volume = regime_data['volume'].values
            volume_consistency = 1.0 - (np.std(volume) / np.mean(volume)) if np.mean(volume) > 0 else 0
            factors.append(volume_consistency)
        else:
            factors.append(0.5)
        
        return np.mean(factors)
    
    def _calculate_trading_quality(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> float:
        """Calculate trading quality of the regime."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        close_prices = regime_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) == 0:
            return 0.5
        
        factors = []
        
        # 1. Sharpe ratio
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        sharpe_score = min(sharpe_ratio / 1.0, 1.0)  # 1.0 threshold
        factors.append(sharpe_score)
        
        # 2. Win rate
        win_rate = np.mean(returns > 0)
        factors.append(win_rate)
        
        # 3. Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns) - 1
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdown))
        drawdown_score = 1.0 - min(max_drawdown / 0.1, 1.0)  # 10% threshold
        factors.append(drawdown_score)
        
        return np.mean(factors)
    
    def _calculate_stability_quality(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> float:
        """Calculate stability quality of the regime."""
        factors = []
        
        # 1. Duration stability
        duration = regime_info.get('duration', 0)
        duration_stability = min(duration / 100, 1.0)  # 100 samples threshold
        factors.append(duration_stability)
        
        # 2. Volatility stability
        volatility = regime_info.get('price_volatility', 0)
        vol_stability = 1.0 - abs(volatility - 0.05) / 0.05 if volatility <= 0.1 else 0.5
        factors.append(vol_stability)
        
        # 3. Trend stability
        trend = abs(regime_info.get('price_trend', 0))
        trend_stability = 1.0 - abs(trend - 0.02) / 0.02 if trend <= 0.1 else 0.5
        factors.append(trend_stability)
        
        return np.mean(factors)
    
    def _calculate_persistence_quality(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> float:
        """Calculate persistence quality of the regime."""
        if 'close' not in regime_data.columns:
            return 0.5
        
        close_prices = regime_data['close'].values
        
        # Persistence based on regime characteristics consistency
        if len(close_prices) > 10:
            # Calculate rolling characteristics
            window = min(10, len(close_prices) // 3)
            rolling_vol = pd.Series(close_prices).rolling(window).std()
            
            # Persistence score based on consistency
            vol_consistency = 1.0 - (rolling_vol.std() / rolling_vol.mean()) if rolling_vol.mean() > 0 else 0
        else:
            vol_consistency = 0.5
        
        # Duration persistence
        duration = regime_info.get('duration', 0)
        duration_persistence = min(duration / 50, 1.0)  # 50 samples threshold
        
        return (vol_consistency + duration_persistence) / 2.0
    
    def _calculate_quality_adjustments(self, quality_metrics: Dict[str, float]) -> float:
        """Calculate quality adjustments for final scoring."""
        # Weight different quality aspects
        weights = {
            'statistical_quality': 0.25,
            'economic_quality': 0.25,
            'trading_quality': 0.25,
            'stability_quality': 0.15,
            'persistence_quality': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, score in quality_metrics.items():
            if metric_name in weights:
                weight = weights[metric_name]
                weighted_score += weight * score
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _generate_quality_report(self, quality_metrics: Dict[str, float], final_score: float) -> str:
        """Generate quality report for regime."""
        report = f"Regime Quality Score: {final_score:.3f}\n"
        report += f"Statistical Quality: {quality_metrics.get('statistical_quality', 0):.3f}\n"
        report += f"Economic Quality: {quality_metrics.get('economic_quality', 0):.3f}\n"
        report += f"Trading Quality: {quality_metrics.get('trading_quality', 0):.3f}\n"
        report += f"Stability Quality: {quality_metrics.get('stability_quality', 0):.3f}\n"
        report += f"Persistence Quality: {quality_metrics.get('persistence_quality', 0):.3f}\n"
        
        if final_score >= 0.8:
            report += "Status: Excellent regime for trading"
        elif final_score >= 0.6:
            report += "Status: Good regime for trading"
        elif final_score >= 0.4:
            report += "Status: Fair regime for trading"
        else:
            report += "Status: Poor regime for trading"
        
        return report
    
    def _calculate_qualification_statistics(self, 
                                           qualified_regimes: Dict[str, Any],
                                           qualification_scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate qualification statistics."""
        if not qualified_regimes:
            return {}
        
        scores = list(qualification_scores.values())
        
        return {
            'n_qualified': len(qualified_regimes),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'score_distribution': {
                'excellent': len([s for s in scores if s >= 0.9]),
                'good': len([s for s in scores if 0.7 <= s < 0.9]),
                'fair': len([s for s in scores if 0.5 <= s < 0.7]),
                'poor': len([s for s in scores if s < 0.5])
            }
        }
    
    def _create_empty_qualification_result(self) -> Dict[str, Any]:
        """Create empty qualification result."""
        return {
            'qualified_regimes': {},
            'qualification_scores': {},
            'qualification_statistics': {},
            'n_qualified': 0,
            'n_total': 0,
            'qualification_rate': 0.0,
            'timestamp': datetime.now().isoformat(),
            'execution_time': 0.0
        }
    
    def get_qualified_regimes(self) -> Dict[str, Any]:
        """Get currently qualified regimes."""
        return self.qualified_regimes
    
    def get_qualification_statistics(self) -> Dict[str, Any]:
        """Get qualification statistics."""
        return {
            'n_qualifications': len(self.qualification_history),
            'n_qualified_regimes': len(self.qualified_regimes),
            'mean_qualification_score': np.mean(list(self.regime_scores.values())) if self.regime_scores else 0.0,
            'qualification_history': self.qualification_history[-10:]  # Last 10 qualifications
        }
    
    def is_regime_qualified(self, regime_name: str) -> bool:
        """Check if a specific regime is qualified."""
        return regime_name in self.qualified_regimes
    
    def get_regime_qualification_score(self, regime_name: str) -> Optional[float]:
        """Get qualification score for a specific regime."""
        return self.regime_scores.get(regime_name)

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
