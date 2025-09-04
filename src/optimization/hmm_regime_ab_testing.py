"""
HMM Regime-Based A/B Testing Framework
Integrates with existing per-HMM regime logic in the pipeline
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from scipy import stats

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.core.decorators.errors import handles_errors


@dataclass
class HMMRegimeABTestResult:
    """Result of an A/B test for a specific HMM regime"""
    regime: str
    test_name: str
    group: str
    trade_id: str
    timestamp: datetime
    pnl: float
    confidence: float
    barriers_used: Dict[str, float]
    timeframe: str
    metadata: Dict[str, Any]


class HMMRegimeABTestingFramework:
    """
    A/B Testing Framework integrated with existing HMM regime logic.
    Tests different barrier configurations across 20 HMM clusters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HMM Regime A/B Testing Framework.
        
        Args:
            config: Configuration dictionary with HMM regime settings
        """
        self.config = config
        self.logger = system_logger.getChild('HMMRegimeABTesting')
        
        # HMM Regime Configuration (integrated with existing pipeline)
        self.hmm_config = config.get('hmm_regimes', {})
        self.regime_names = [f"regime_{i:02d}" for i in range(20)]  # regime_00 to regime_19
        self.traffic_split = 0.5  # 50/50 split
        self.trading_fee = 0.0008  # 0.08% trading fee
        
        # A/B Test Storage
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, List[HMMRegimeABTestResult]] = {}
        self.test_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Integration with existing HMM pipeline
        self.regime_predictor = None  # Will be injected from existing HMM system
        self.barrier_optimizer = None  # Will be injected from existing barrier system
        
    @handles_errors(exceptions=(ValueError, AttributeError), default_return=False, context='HMM A/B testing initialization')
    async def initialize(self) -> bool:
        """
        Initialize the A/B testing framework with existing HMM components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing HMM Regime A/B Testing Framework...")
            
            # Initialize test storage
            for test_name in self._get_default_test_names():
                self.test_results[test_name] = []
                self.test_metrics[test_name] = {}
                
            # Create default A/B tests
            await self._create_default_ab_tests()
            
            self.logger.info("✅ HMM Regime A/B Testing Framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ HMM A/B Testing initialization failed: {e}")
            return False
    
    def _get_default_test_names(self) -> List[str]:
        """Get default A/B test names"""
        return [
            'conservative_vs_aggressive_barriers',
            'timeframe_adaptive_vs_fixed',
            'regime_specific_vs_universal',
            'confidence_threshold_optimization',
            'leverage_optimization'
        ]
    
    async def _create_default_ab_tests(self) -> None:
        """Create default A/B tests for HMM regimes"""
        
        # Test 1: Conservative vs Aggressive Barriers per Regime
        self.active_tests['conservative_vs_aggressive_barriers'] = {
            'description': 'Test conservative vs aggressive barrier configurations per HMM regime',
            'groups': {
                'group_a': {
                    'name': 'conservative',
                    'regime_configs': self._create_conservative_regime_configs()
                },
                'group_b': {
                    'name': 'aggressive',
                    'regime_configs': self._create_aggressive_regime_configs()
                }
            },
            'start_time': datetime.now(),
            'min_sample_size': 100,
            'max_duration_days': 30
        }
        
        # Test 2: Timeframe Adaptive vs Fixed per Regime
        self.active_tests['timeframe_adaptive_vs_fixed'] = {
            'description': 'Test timeframe-adaptive vs fixed barriers per HMM regime',
            'groups': {
                'group_a': {
                    'name': 'timeframe_adaptive',
                    'regime_configs': self._create_timeframe_adaptive_configs()
                },
                'group_b': {
                    'name': 'timeframe_fixed',
                    'regime_configs': self._create_timeframe_fixed_configs()
                }
            },
            'start_time': datetime.now(),
            'min_sample_size': 100,
            'max_duration_days': 30
        }
        
        # Test 3: Regime-Specific vs Universal Barriers
        self.active_tests['regime_specific_vs_universal'] = {
            'description': 'Test regime-specific vs universal barrier configurations',
            'groups': {
                'group_a': {
                    'name': 'regime_specific',
                    'regime_configs': self._create_regime_specific_configs()
                },
                'group_b': {
                    'name': 'universal',
                    'regime_configs': self._create_universal_configs()
                }
            },
            'start_time': datetime.now(),
            'min_sample_size': 100,
            'max_duration_days': 30
        }
    
    def _create_conservative_regime_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create conservative barrier configurations for each regime"""
        configs = {}
        
        for regime in self.regime_names:
            # Conservative: Lower risk, higher confidence threshold (accounting for 0.08% trading fee)
            configs[regime] = {
                'profit_take_multiplier': 0.0018,  # 0.18% (0.1% + 0.08% fee)
                'stop_loss_multiplier': 0.0013,    # 0.13% (0.05% + 0.08% fee)
                'confidence_threshold': 0.7,       # 70%
                'leverage_multiplier': 5.0,        # Lower leverage
                'timeframe_weights': {
                    '5m': 0.2,
                    '15m': 0.3,
                    '30m': 0.3,
                    '1h': 0.2
                }
            }
            
        return configs
    
    def _create_aggressive_regime_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create aggressive barrier configurations for each regime"""
        configs = {}
        
        for regime in self.regime_names:
            # Aggressive: Higher risk, lower confidence threshold (accounting for 0.08% trading fee)
            configs[regime] = {
                'profit_take_multiplier': 0.0038,  # 0.38% (0.3% + 0.08% fee)
                'stop_loss_multiplier': 0.0028,    # 0.28% (0.2% + 0.08% fee)
                'confidence_threshold': 0.5,       # 50%
                'leverage_multiplier': 10.0,       # Higher leverage
                'timeframe_weights': {
                    '5m': 0.4,
                    '15m': 0.4,
                    '30m': 0.15,
                    '1h': 0.05
                }
            }
            
        return configs
    
    def _create_timeframe_adaptive_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create timeframe-adaptive configurations for each regime"""
        configs = {}
        
        for regime in self.regime_names:
            # Adaptive: Barriers change based on timeframe
            configs[regime] = {
                'timeframe_configs': {
                    '5m': {
                        'profit_take_multiplier': 0.0016,  # 0.16% (0.08% + 0.08% fee)
                        'stop_loss_multiplier': 0.0012,    # 0.12% (0.04% + 0.08% fee)
                        'confidence_threshold': 0.55
                    },
                    '15m': {
                        'profit_take_multiplier': 0.0018,  # 0.18% (0.1% + 0.08% fee)
                        'stop_loss_multiplier': 0.0013,    # 0.13% (0.05% + 0.08% fee)
                        'confidence_threshold': 0.6
                    },
                    '30m': {
                        'profit_take_multiplier': 0.0028,  # 0.28% (0.2% + 0.08% fee)
                        'stop_loss_multiplier': 0.0018,    # 0.18% (0.1% + 0.08% fee)
                        'confidence_threshold': 0.65
                    },
                    '1h': {
                        'profit_take_multiplier': 0.0038,  # 0.38% (0.3% + 0.08% fee)
                        'stop_loss_multiplier': 0.0023,    # 0.23% (0.15% + 0.08% fee)
                        'confidence_threshold': 0.7
                    }
                },
                'leverage_multiplier': 8.0
            }
            
        return configs
    
    def _create_timeframe_fixed_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create timeframe-fixed configurations for each regime"""
        configs = {}
        
        for regime in self.regime_names:
            # Fixed: Same barriers across all timeframes (accounting for 0.08% trading fee)
            configs[regime] = {
                'profit_take_multiplier': 0.0028,  # 0.28% (0.2% + 0.08% fee)
                'stop_loss_multiplier': 0.0018,    # 0.18% (0.1% + 0.08% fee)
                'confidence_threshold': 0.6,       # 60%
                'leverage_multiplier': 8.0,
                'timeframe_weights': {
                    '5m': 0.3,
                    '15m': 0.4,
                    '30m': 0.2,
                    '1h': 0.1
                }
            }
            
        return configs
    
    def _create_regime_specific_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create regime-specific configurations (optimized per regime)"""
        configs = {}
        
        for i, regime in enumerate(self.regime_names):
            # Regime-specific: Optimized barriers per regime (accounting for 0.08% trading fee)
            regime_index = i
            
            # Vary parameters based on regime index, adding trading fee
            base_profit_take = 0.001 + (regime_index * 0.0001)  # 0.1% to 0.3%
            base_stop_loss = 0.0005 + (regime_index * 0.00005)  # 0.05% to 0.15%
            
            configs[regime] = {
                'profit_take_multiplier': base_profit_take + self.trading_fee,  # Add 0.08% fee
                'stop_loss_multiplier': base_stop_loss + self.trading_fee,      # Add 0.08% fee
                'confidence_threshold': 0.5 + (regime_index * 0.01),           # 50% to 69%
                'leverage_multiplier': 5.0 + (regime_index * 0.25),            # 5x to 9.75x
                'regime_specific_optimization': True
            }
            
        return configs
    
    def _create_universal_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create universal configurations (same for all regimes)"""
        universal_config = {
            'profit_take_multiplier': 0.0028,  # 0.28% (0.2% + 0.08% fee)
            'stop_loss_multiplier': 0.0018,    # 0.18% (0.1% + 0.08% fee)
            'confidence_threshold': 0.6,       # 60%
            'leverage_multiplier': 8.0,
            'regime_specific_optimization': False
        }
        
        # Same config for all regimes
        configs = {regime: universal_config.copy() for regime in self.regime_names}
        return configs
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='HMM A/B test trade execution')
    async def execute_ab_test_trade(
        self,
        trade_input: Dict[str, Any],
        test_name: str,
        regime: str,
        timeframe: str
    ) -> Optional[HMMRegimeABTestResult]:
        """
        Execute a trade with A/B test configuration for specific HMM regime.
        
        Args:
            trade_input: Trade input parameters
            test_name: Name of the A/B test
            regime: HMM regime name
            timeframe: Trading timeframe
            
        Returns:
            HMMRegimeABTestResult: Result of the A/B test trade
        """
        try:
            if test_name not in self.active_tests:
                self.logger.error(f"Test {test_name} not found in active tests")
                return None
            
            # Assign trade to group based on hash (ensures consistency)
            group = self._assign_trade_to_group(trade_input['trade_id'], test_name, regime)
            
            # Get test configuration for this regime and group
            test_config = self.active_tests[test_name]['groups'][group]['regime_configs'][regime]
            
            # Apply timeframe-specific configuration if available
            if 'timeframe_configs' in test_config and timeframe in test_config['timeframe_configs']:
                timeframe_config = test_config['timeframe_configs'][timeframe]
                final_config = {**test_config, **timeframe_config}
            else:
                final_config = test_config
            
            # Execute trade with A/B test configuration
            trade_result = await self._execute_trade_with_ab_config(
                trade_input, final_config, regime, timeframe
            )
            
            # Create A/B test result
            ab_result = HMMRegimeABTestResult(
                regime=regime,
                test_name=test_name,
                group=group,
                trade_id=trade_input['trade_id'],
                timestamp=datetime.now(),
                pnl=trade_result['pnl'],
                confidence=trade_result['confidence'],
                barriers_used=final_config,
                timeframe=timeframe,
                metadata=trade_result.get('metadata', {})
            )
            
            # Store result
            self.test_results[test_name].append(ab_result)
            
            # Update metrics
            await self._update_test_metrics(test_name, ab_result)
            
            return ab_result
            
        except Exception as e:
            self.logger.error(f"Error executing A/B test trade: {e}")
            return None
    
    def _assign_trade_to_group(self, trade_id: str, test_name: str, regime: str) -> str:
        """Assign trade to A or B group based on hash (ensures consistency)"""
        hash_value = hash(f"{trade_id}_{test_name}_{regime}") % 100
        return 'group_a' if hash_value < (self.traffic_split * 100) else 'group_b'
    
    async def _execute_trade_with_ab_config(
        self,
        trade_input: Dict[str, Any],
        config: Dict[str, Any],
        regime: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Execute trade with A/B test configuration"""
        
        # This would integrate with your existing trade execution system
        # For now, simulate the trade execution
        
        # Apply configuration to trade
        modified_input = trade_input.copy()
        modified_input['barrier_config'] = config
        modified_input['regime'] = regime
        modified_input['timeframe'] = timeframe
        
        # Simulate trade execution (replace with actual execution)
        trade_result = {
            'pnl': np.random.normal(0, 0.01),  # Simulated PnL
            'confidence': trade_input.get('confidence', 0.6),
            'metadata': {
                'execution_time': datetime.now(),
                'config_used': config
            }
        }
        
        return trade_result
    
    async def _update_test_metrics(self, test_name: str, result: HMMRegimeABTestResult) -> None:
        """Update test metrics with new result"""
        
        if test_name not in self.test_metrics:
            self.test_metrics[test_name] = {}
        
        # Initialize regime metrics if not exists
        if result.regime not in self.test_metrics[test_name]:
            self.test_metrics[test_name][result.regime] = {
                'group_a': {'trades': [], 'total_pnl': 0, 'trade_count': 0},
                'group_b': {'trades': [], 'total_pnl': 0, 'trade_count': 0}
            }
        
        # Update metrics
        regime_metrics = self.test_metrics[test_name][result.regime]
        group_metrics = regime_metrics[result.group]
        
        group_metrics['trades'].append(result.pnl)
        group_metrics['total_pnl'] += result.pnl
        group_metrics['trade_count'] += 1
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='HMM A/B test analysis')
    async def analyze_ab_test_results(
        self,
        test_name: str,
        min_sample_size: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze A/B test results for statistical significance.
        
        Args:
            test_name: Name of the A/B test to analyze
            min_sample_size: Minimum sample size for analysis
            
        Returns:
            Dict: Analysis results with statistical significance
        """
        try:
            if test_name not in self.test_metrics:
                self.logger.error(f"Test {test_name} not found in metrics")
                return None
            
            analysis_results = {
                'test_name': test_name,
                'analysis_timestamp': datetime.now(),
                'regime_results': {},
                'overall_results': {},
                'recommendations': []
            }
            
            # Analyze each regime
            for regime, regime_metrics in self.test_metrics[test_name].items():
                regime_analysis = self._analyze_regime_results(
                    regime, regime_metrics, min_sample_size
                )
                analysis_results['regime_results'][regime] = regime_analysis
            
            # Calculate overall results
            analysis_results['overall_results'] = self._calculate_overall_results(
                analysis_results['regime_results']
            )
            
            # Generate recommendations
            analysis_results['recommendations'] = self._generate_recommendations(
                analysis_results['regime_results']
            )
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing A/B test results: {e}")
            return None
    
    def _analyze_regime_results(
        self,
        regime: str,
        regime_metrics: Dict[str, Any],
        min_sample_size: int
    ) -> Dict[str, Any]:
        """Analyze results for a specific regime"""
        
        group_a_trades = regime_metrics['group_a']['trades']
        group_b_trades = regime_metrics['group_b']['trades']
        
        if len(group_a_trades) < min_sample_size or len(group_b_trades) < min_sample_size:
            return {
                'status': 'insufficient_data',
                'group_a_sample_size': len(group_a_trades),
                'group_b_sample_size': len(group_b_trades),
                'min_required': min_sample_size
            }
        
        # Calculate statistics
        group_a_stats = self._calculate_group_statistics(group_a_trades)
        group_b_stats = self._calculate_group_statistics(group_b_trades)
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(group_a_trades, group_b_trades)
        
        # Determine winner
        winner = 'group_a' if group_a_stats['mean_pnl'] > group_b_stats['mean_pnl'] else 'group_b'
        
        return {
            'status': 'sufficient_data',
            'group_a': group_a_stats,
            'group_b': group_b_stats,
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'confidence_level': 0.95
            },
            'winner': winner,
            'effect_size': self._calculate_effect_size(group_a_trades, group_b_trades)
        }
    
    def _calculate_group_statistics(self, trades: List[float]) -> Dict[str, Any]:
        """Calculate statistics for a group of trades"""
        
        if not trades:
            return {
                'sample_size': 0,
                'mean_pnl': 0,
                'std_pnl': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        trades_array = np.array(trades)
        
        return {
            'sample_size': len(trades),
            'mean_pnl': np.mean(trades_array),
            'std_pnl': np.std(trades_array),
            'total_pnl': np.sum(trades_array),
            'win_rate': np.sum(trades_array > 0) / len(trades),
            'sharpe_ratio': np.mean(trades_array) / np.std(trades_array) if np.std(trades_array) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(trades_array)
        }
    
    def _calculate_max_drawdown(self, trades: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(trades)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown) if len(drawdown) > 0 else 0
    
    def _calculate_effect_size(self, group_a: List[float], group_b: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if not group_a or not group_b:
            return 0
        
        mean_a, mean_b = np.mean(group_a), np.mean(group_b)
        std_a, std_b = np.std(group_a), np.std(group_b)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((len(group_a) - 1) * std_a**2 + (len(group_b) - 1) * std_b**2) / 
                            (len(group_a) + len(group_b) - 2))
        
        return (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
    
    def _calculate_overall_results(self, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall results across all regimes"""
        
        significant_regimes = []
        group_a_wins = 0
        group_b_wins = 0
        
        for regime, results in regime_results.items():
            if results.get('status') == 'sufficient_data':
                if results['statistical_test']['significant']:
                    significant_regimes.append(regime)
                    if results['winner'] == 'group_a':
                        group_a_wins += 1
                    else:
                        group_b_wins += 1
        
        return {
            'total_regimes': len(regime_results),
            'significant_regimes': len(significant_regimes),
            'group_a_wins': group_a_wins,
            'group_b_wins': group_b_wins,
            'overall_winner': 'group_a' if group_a_wins > group_b_wins else 'group_b',
            'significance_rate': len(significant_regimes) / len(regime_results) if regime_results else 0
        }
    
    def _generate_recommendations(self, regime_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        
        recommendations = []
        
        # Count significant results
        significant_count = sum(1 for r in regime_results.values() 
                              if r.get('status') == 'sufficient_data' and r['statistical_test']['significant'])
        
        if significant_count > len(regime_results) * 0.7:  # 70% of regimes significant
            recommendations.append("Strong evidence for configuration difference - consider implementing winning configuration")
        elif significant_count > len(regime_results) * 0.3:  # 30% of regimes significant
            recommendations.append("Moderate evidence for configuration difference - consider regime-specific implementation")
        else:
            recommendations.append("Weak evidence for configuration difference - continue testing or consider other factors")
        
        # Regime-specific recommendations
        for regime, results in regime_results.items():
            if results.get('status') == 'sufficient_data' and results['statistical_test']['significant']:
                effect_size = results.get('effect_size', 0)
                if abs(effect_size) > 0.8:  # Large effect
                    recommendations.append(f"Large effect size in {regime} - prioritize this regime for implementation")
                elif abs(effect_size) > 0.5:  # Medium effect
                    recommendations.append(f"Medium effect size in {regime} - consider regime-specific optimization")
        
        return recommendations
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all active A/B tests"""
        
        summary = {
            'active_tests': len(self.active_tests),
            'test_names': list(self.active_tests.keys()),
            'total_trades': sum(len(results) for results in self.test_results.values()),
            'regime_coverage': len(self.regime_names),
            'test_details': {}
        }
        
        for test_name, test_config in self.active_tests.items():
            summary['test_details'][test_name] = {
                'description': test_config['description'],
                'start_time': test_config['start_time'],
                'trade_count': len(self.test_results.get(test_name, [])),
                'regimes_tested': len(self.test_metrics.get(test_name, {}))
            }
        
        return summary