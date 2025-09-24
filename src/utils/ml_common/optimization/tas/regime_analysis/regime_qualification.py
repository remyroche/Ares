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
        
        # Combined economic significance
        price_significant = price_movement >= min_price_movement
        volume_significant = volume_ratio >= min_volume_ratio
        
        passed = price_significant and volume_significant
        score = (min(price_movement / min_price_movement, 1.0) + 
                min(volume_ratio / min_volume_ratio, 1.0)) / 2.0
        
        return {
            'passed': passed,
            'score': score,
            'price_movement': price_movement,
            'volume_ratio': volume_ratio,
            'min_price_movement': min_price_movement,
            'min_volume_ratio': min_volume_ratio,
            'test_name': 'economic_significance'
        }
    
    def _test_normality(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Test normality of regime returns."""
        if 'close' not in regime_data.columns:
            return {'passed': True, 'score': 1.0, 'test_name': 'normality'}
        
        close_prices = regime_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) < 3:
            return {'passed': True, 'score': 1.0, 'test_name': 'normality'}
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = jarque_bera(returns)
            passed = jb_pvalue > self.config.significance_level
            score = jb_pvalue  # Higher p-value = more normal
        except:
            passed = True
            score = 1.0
        
        return {
            'passed': passed,
            'score': score,
            'jb_statistic': jb_stat if 'jb_stat' in locals() else 0,
            'jb_pvalue': jb_pvalue if 'jb_pvalue' in locals() else 1.0,
            'test_name': 'normality'
        }
    
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
        
        # Sharpe ratio
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
        
        passed = sharpe_passed and drawdown_passed and winrate_passed and profit_passed
        
        # Calculate score
        sharpe_score = min(sharpe_ratio / self.config.min_sharpe_ratio, 1.0)
        drawdown_score = 1.0 - (abs(max_drawdown) / self.config.max_drawdown_threshold)
        winrate_score = min(win_rate / self.config.min_win_rate, 1.0)
        profit_score = min(profit_factor / self.config.min_profit_factor, 1.0)
        
        overall_score = (sharpe_score + drawdown_score + winrate_score + profit_score) / 4.0
        
        return {
            'passed': passed,
            'score': overall_score,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'test_name': 'trading'
        }
    
    def _test_regime_stability(self, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Test stability of the regime."""
        # Check if regime characteristics are consistent
        volatility = regime_info.get('price_volatility', 0)
        trend = regime_info.get('price_trend', 0)
        duration = regime_info.get('duration', 0)
        
        # Stability based on regime characteristics
        volatility_stable = 0.01 <= volatility <= 0.3  # Reasonable volatility range
        trend_stable = abs(trend) <= 0.5  # Not too extreme trend
        duration_stable = duration >= self.config.min_regime_duration
        
        passed = volatility_stable and trend_stable and duration_stable
        
        # Calculate stability score
        vol_score = 1.0 - abs(volatility - 0.1) / 0.3 if volatility <= 0.3 else 0.0
        trend_score = 1.0 - abs(trend) / 0.5 if abs(trend) <= 0.5 else 0.0
        duration_score = min(duration / self.config.min_regime_duration, 1.0)
        
        overall_score = (vol_score + trend_score + duration_score) / 3.0
        
        return {
            'passed': passed,
            'score': overall_score,
            'volatility_stable': volatility_stable,
            'trend_stable': trend_stable,
            'duration_stable': duration_stable,
            'test_name': 'stability'
        }
    
    def _calculate_overall_score(self, tests: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall qualification score."""
        if not tests:
            return 0.0
        
        # Weight different tests
        weights = {
            'duration': 0.15,
            'volatility': 0.15,
            'trend': 0.15,
            'economic': 0.20,
            'normality': 0.10,
            'stationarity': 0.10,
            'autocorrelation': 0.05,
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
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
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