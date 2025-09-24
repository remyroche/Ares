"""
Regime Analyzer for Advanced Market Analysis.

This module provides advanced regime analysis capabilities that can be used
by both NAS and TAS regime detection systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from src.utils.logger import system_logger


@dataclass
class RegimeAnalysisConfig:
    """Configuration for regime analysis."""
    min_regime_samples: int = 50
    max_regime_imbalance: float = 0.9
    enable_statistical_tests: bool = True
    enable_regime_characterization: bool = True
    enable_transition_analysis: bool = True
    enable_stability_analysis: bool = True


@dataclass
class RegimeCharacteristics:
    """Characteristics of a market regime."""
    regime_id: int
    sample_count: int
    duration_mean: float
    duration_std: float
    return_mean: float
    return_std: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    transition_matrix: np.ndarray
    stability_score: float
    economic_significance: float


class RegimeAnalyzer:
    """
    Advanced regime analyzer for market analysis systems.

    This class provides comprehensive regime analysis capabilities including
    statistical testing, regime characterization, transition analysis, and
    stability assessment that can be used by both NAS and TAS systems.
    """

    def __init__(self, config: RegimeAnalysisConfig):
        """
        Initialize the regime analyzer.

        Args:
            config: Regime analysis configuration
        """
        self.logger = system_logger.getChild('RegimeAnalyzer')
        self.config = config

        self.logger.info("✅ Regime Analyzer initialized"
        self.logger.info(f"   Min samples: {config.min_regime_samples}")
        self.logger.info(f"   Max imbalance: {config.max_regime_imbalance}")

    def analyze_regimes(self,
                       regime_data: pd.DataFrame,
                       market_data: pd.DataFrame,
                       regime_column: str = 'regime') -> Dict[str, Any]:
        """
        Perform comprehensive regime analysis.

        Args:
            regime_data: DataFrame with regime assignments
            market_data: Market data with OHLCV
            regime_column: Name of regime column

        Returns:
            Dictionary with comprehensive regime analysis
        """
        try:
            self.logger.info("📊 Performing comprehensive regime analysis")

            # Basic regime statistics
            regime_stats = self._calculate_regime_statistics(regime_data, market_data, regime_column)

            # Regime characteristics
            regime_characteristics = {}
            if self.config.enable_regime_characterization:
                regime_characteristics = self._analyze_regime_characteristics(
                    regime_data, market_data, regime_column
                )

            # Transition analysis
            transition_analysis = {}
            if self.config.enable_transition_analysis:
                transition_analysis = self._analyze_regime_transitions(regime_data, regime_column)

            # Stability analysis
            stability_analysis = {}
            if self.config.enable_stability_analysis:
                stability_analysis = self._analyze_regime_stability(regime_data, market_data, regime_column)

            # Statistical tests
            statistical_tests = {}
            if self.config.enable_statistical_tests:
                statistical_tests = self._perform_statistical_tests(
                    regime_data, market_data, regime_column
                )

            analysis_result = {
                'regime_statistics': regime_stats,
                'regime_characteristics': regime_characteristics,
                'transition_analysis': transition_analysis,
                'stability_analysis': stability_analysis,
                'statistical_tests': statistical_tests,
                'analysis_metadata': {
                    'total_regimes': len(regime_data[regime_column].unique()),
                    'total_samples': len(regime_data),
                    'analysis_timestamp': pd.Timestamp.now(),
                    'min_regime_samples': self.config.min_regime_samples
                }
            }

            self.logger.info(f"✅ Regime analysis completed for {len(regime_characteristics)} regimes")
            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Regime analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_regime_statistics(self,
                                   regime_data: pd.DataFrame,
                                   market_data: pd.DataFrame,
                                   regime_column: str) -> Dict[str, Any]:
        """
        Calculate basic regime statistics.

        Args:
            regime_data: DataFrame with regime assignments
            market_data: Market data
            regime_column: Name of regime column

        Returns:
            Dictionary of regime statistics
        """
        try:
            stats = {}
            unique_regimes = sorted(regime_data[regime_column].unique())

            # Overall statistics
            stats['total_regimes'] = len(unique_regimes)
            stats['total_samples'] = len(regime_data)

            # Regime distribution
            regime_counts = regime_data[regime_column].value_counts()
            stats['regime_distribution'] = regime_counts.to_dict()
            stats['regime_percentages'] = (regime_counts / len(regime_data) * 100).to_dict()

            # Sample statistics
            stats['min_regime_samples'] = int(regime_counts.min())
            stats['max_regime_samples'] = int(regime_counts.max())
            stats['mean_regime_samples'] = float(regime_counts.mean())
            stats['std_regime_samples'] = float(regime_counts.std())

            # Regime balance
            stats['regime_balance_ratio'] = stats['min_regime_samples'] / stats['max_regime_samples']

            # Check minimum sample requirements
            sufficient_samples = all(count >= self.config.min_regime_samples for count in regime_counts)
            stats['sufficient_samples'] = sufficient_samples

            if not sufficient_samples:
                self.logger.warning(f"⚠️ Some regimes have insufficient samples (< {self.config.min_regime_samples})")

            return stats

        except Exception as e:
            self.logger.warning(f"⚠️ Regime statistics calculation failed: {e}")
            return {}

    def _analyze_regime_characteristics(self,
                                      regime_data: pd.DataFrame,
                                      market_data: pd.DataFrame,
                                      regime_column: str) -> Dict[str, RegimeCharacteristics]:
        """
        Analyze detailed characteristics of each regime.

        Args:
            regime_data: DataFrame with regime assignments
            market_data: Market data
            regime_column: Name of regime column

        Returns:
            Dictionary of regime characteristics
        """
        try:
            characteristics = {}
            unique_regimes = sorted(regime_data[regime_column].unique())

            for regime_id in unique_regimes:
                try:
                    regime_mask = regime_data[regime_column] == regime_id
                    regime_returns = market_data.loc[regime_mask, 'close'].pct_change().dropna()

                    if len(regime_returns) < self.config.min_regime_samples:
                        continue

                    # Calculate regime characteristics
                    char = RegimeCharacteristics(
                        regime_id=regime_id,
                        sample_count=len(regime_returns),
                        duration_mean=self._calculate_regime_duration(regime_data, regime_id, regime_column),
                        duration_std=0.0,  # Simplified
                        return_mean=regime_returns.mean(),
                        return_std=regime_returns.std(),
                        volatility=regime_returns.std(),
                        sharpe_ratio=self._calculate_sharpe_ratio(regime_returns),
                        max_drawdown=self._calculate_max_drawdown(regime_returns),
                        win_rate=(regime_returns > 0).mean(),
                        transition_matrix=self._calculate_regime_transition_matrix(regime_data, regime_column),
                        stability_score=self._calculate_regime_stability_score(regime_data, regime_id, regime_column),
                        economic_significance=self._calculate_economic_significance(regime_returns)
                    )

                    characteristics[f'regime_{regime_id}'] = char

                except Exception as e:
                    self.logger.warning(f"⚠️ Regime {regime_id} characteristics analysis failed: {e}")

            self.logger.info(f"✅ Analyzed characteristics for {len(characteristics)} regimes")
            return characteristics

        except Exception as e:
            self.logger.error(f"❌ Regime characteristics analysis failed: {e}")
            return {}

    def _calculate_regime_duration(self,
                                 regime_data: pd.DataFrame,
                                 regime_id: int,
                                 regime_column: str) -> float:
        """
        Calculate average duration of a regime.

        Args:
            regime_data: DataFrame with regime assignments
            regime_id: Regime ID
            regime_column: Name of regime column

        Returns:
            Average regime duration
        """
        try:
            # This is a simplified duration calculation
            # In practice, you would calculate actual time durations
            regime_mask = regime_data[regime_column] == regime_id
            duration = regime_mask.sum()  # Number of consecutive periods

            return float(duration)

        except Exception as e:
            self.logger.warning(f"⚠️ Regime duration calculation failed: {e}")
            return 1.0

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio for returns series.

        Args:
            returns: Returns series

        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) < 2:
                return 0.0

            mean_return = returns.mean()
            std_return = returns.std()

            if std_return == 0:
                return 0.0

            # Annualized Sharpe ratio (assuming daily returns)
            sharpe = mean_return / std_return * np.sqrt(252)
            return sharpe

        except Exception as e:
            self.logger.warning(f"⚠️ Sharpe ratio calculation failed: {e}")
            return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown for returns series.

        Args:
            returns: Returns series

        Returns:
            Maximum drawdown (positive value)
        """
        try:
            if len(returns) < 2:
                return 0.0

            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()

            # Calculate running maximum
            running_max = cum_returns.expanding().max()

            # Calculate drawdown
            drawdown = (cum_returns - running_max) / running_max

            # Return maximum drawdown (positive value)
            max_dd = abs(drawdown.min())
            return max_dd

        except Exception as e:
            self.logger.warning(f"⚠️ Max drawdown calculation failed: {e}")
            return 0.0

    def _calculate_regime_transition_matrix(self, regime_data: pd.DataFrame, regime_column: str) -> np.ndarray:
        """
        Calculate regime transition matrix.

        Args:
            regime_data: DataFrame with regime assignments
            regime_column: Name of regime column

        Returns:
            Transition matrix
        """
        try:
            unique_regimes = sorted(regime_data[regime_column].unique())
            n_regimes = len(unique_regimes)

            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))

            # Calculate transitions
            regime_sequence = regime_data[regime_column].values
            for i in range(len(regime_sequence) - 1):
                current_regime = unique_regimes.index(regime_sequence[i])
                next_regime = unique_regimes.index(regime_sequence[i + 1])
                transition_matrix[current_regime, next_regime] += 1

            # Normalize rows to probabilities
            row_sums = transition_matrix.sum(axis=1)
            for i in range(n_regimes):
                if row_sums[i] > 0:
                    transition_matrix[i, :] /= row_sums[i]

            return transition_matrix

        except Exception as e:
            self.logger.warning(f"⚠️ Transition matrix calculation failed: {e}")
            n_regimes = len(unique_regimes)
            return np.ones((n_regimes, n_regimes)) / n_regimes  # Uniform fallback

    def _calculate_regime_stability_score(self,
                                        regime_data: pd.DataFrame,
                                        regime_id: int,
                                        regime_column: str) -> float:
        """
        Calculate stability score for a regime.

        Args:
            regime_data: DataFrame with regime assignments
            regime_id: Regime ID
            regime_column: Name of regime column

        Returns:
            Stability score (0-1)
        """
        try:
            # Simplified stability score based on regime persistence
            regime_mask = regime_data[regime_column] == regime_id
            regime_sequence = regime_mask.astype(int).values

            # Calculate runs of consecutive regime occurrences
            if len(regime_sequence) < 2:
                return 0.5

            # Count transitions in and out of regime
            transitions = np.diff(regime_sequence)
            regime_changes = np.sum(transitions != 0)

            # Stability is higher with fewer transitions
            stability = 1.0 / (1.0 + regime_changes / len(regime_sequence))
            stability = max(0.0, min(1.0, stability))

            return stability

        except Exception as e:
            self.logger.warning(f"⚠️ Stability score calculation failed: {e}")
            return 0.5

    def _calculate_economic_significance(self, returns: pd.Series) -> float:
        """
        Calculate economic significance of a regime.

        Args:
            returns: Returns series for the regime

        Returns:
            Economic significance score (0-1)
        """
        try:
            if len(returns) < 10:
                return 0.0

            # Simple economic significance based on Sharpe ratio and consistency
            sharpe = self._calculate_sharpe_ratio(returns)
            win_rate = (returns > 0).mean()

            # Normalize Sharpe ratio (assuming 2.0 is excellent)
            normalized_sharpe = min(sharpe / 2.0, 1.0)

            # Economic significance combines Sharpe ratio and win rate
            significance = (normalized_sharpe * 0.7 + win_rate * 0.3)
            significance = max(0.0, min(1.0, significance))

            return significance

        except Exception as e:
            self.logger.warning(f"⚠️ Economic significance calculation failed: {e}")
            return 0.0

    def _analyze_regime_transitions(self,
                                  regime_data: pd.DataFrame,
                                  regime_column: str) -> Dict[str, Any]:
        """
        Analyze regime transitions and dynamics.

        Args:
            regime_data: DataFrame with regime assignments
            regime_column: Name of regime column

        Returns:
            Dictionary of transition analysis
        """
        try:
            analysis = {}

            # Transition matrix
            transition_matrix = self._calculate_regime_transition_matrix(regime_data, regime_column)
            analysis['transition_matrix'] = transition_matrix

            # Stationary distribution (simplified)
            try:
                # Solve for stationary distribution: πP = π, π∑ = 1
                n_regimes = transition_matrix.shape[0]
                P = transition_matrix.T  # Transpose for row stochastic
                P = P - np.eye(n_regimes)  # P - I
                P[-1, :] = 1.0  # Last equation for sum = 1

                b = np.zeros(n_regimes)
                b[-1] = 1.0

                stationary_dist = np.linalg.solve(P, b)
                stationary_dist = np.maximum(stationary_dist, 0)  # Ensure non-negative
                stationary_dist /= stationary_dist.sum()  # Normalize

                analysis['stationary_distribution'] = stationary_dist

            except:
                # Fallback uniform distribution
                n_regimes = transition_matrix.shape[0]
                analysis['stationary_distribution'] = np.ones(n_regimes) / n_regimes

            # Transition statistics
            unique_regimes = sorted(regime_data[regime_column].unique())
            analysis['regime_persistence'] = {}

            for i, regime in enumerate(unique_regimes):
                regime_mask = regime_data[regime_column] == regime
                persistence = self._calculate_regime_persistence(regime_data, regime, regime_column)
                analysis['regime_persistence'][f'regime_{regime}'] = persistence

            return analysis

        except Exception as e:
            self.logger.warning(f"⚠️ Transition analysis failed: {e}")
            return {}

    def _calculate_regime_persistence(self,
                                    regime_data: pd.DataFrame,
                                    regime_id: int,
                                    regime_column: str) -> Dict[str, float]:
        """
        Calculate regime persistence metrics.

        Args:
            regime_data: DataFrame with regime assignments
            regime_id: Regime ID
            regime_column: Name of regime column

        Returns:
            Dictionary of persistence metrics
        """
        try:
            regime_mask = regime_data[regime_column] == regime_id
            regime_sequence = regime_mask.values

            if len(regime_sequence) < 2:
                return {'avg_duration': 1.0, 'max_duration': 1.0, 'transitions': 0}

            # Calculate run lengths
            run_lengths = []
            current_run = 0

            for value in regime_sequence:
                if value:
                    current_run += 1
                else:
                    if current_run > 0:
                        run_lengths.append(current_run)
                        current_run = 0

            if current_run > 0:
                run_lengths.append(current_run)

            if not run_lengths:
                return {'avg_duration': 1.0, 'max_duration': 1.0, 'transitions': 0}

            return {
                'avg_duration': np.mean(run_lengths),
                'max_duration': np.max(run_lengths),
                'transitions': len(run_lengths) - 1
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Persistence calculation failed: {e}")
            return {'avg_duration': 1.0, 'max_duration': 1.0, 'transitions': 0}

    def _analyze_regime_stability(self,
                                regime_data: pd.DataFrame,
                                market_data: pd.DataFrame,
                                regime_column: str) -> Dict[str, Any]:
        """
        Analyze regime stability over time.

        Args:
            regime_data: DataFrame with regime assignments
            market_data: Market data
            regime_column: Name of regime column

        Returns:
            Dictionary of stability analysis
        """
        try:
            analysis = {}

            # Split data into time windows
            window_size = max(100, len(regime_data) // 10)  # 10 windows
            windows = []

            for i in range(0, len(regime_data), window_size):
                window_data = regime_data.iloc[i:i + window_size]
                if len(window_data) > 10:  # Minimum window size
                    windows.append(window_data)

            if len(windows) < 2:
                analysis['stability_score'] = 0.5
                analysis['regime_drift'] = 0.0
                return analysis

            # Calculate regime consistency across windows
            regime_consistency = []
            for i in range(len(windows)):
                for j in range(i + 1, len(windows)):
                    consistency = self._calculate_regime_consistency(windows[i], windows[j], regime_column)
                    regime_consistency.append(consistency)

            # Overall stability score
            if regime_consistency:
                avg_consistency = np.mean(regime_consistency)
                stability_score = min(avg_consistency * 2, 1.0)  # Scale to 0-1
            else:
                stability_score = 0.5

            analysis['stability_score'] = stability_score
            analysis['regime_consistency'] = regime_consistency
            analysis['n_windows'] = len(windows)
            analysis['window_size'] = window_size

            return analysis

        except Exception as e:
            self.logger.warning(f"⚠️ Stability analysis failed: {e}")
            return {'stability_score': 0.5, 'error': str(e)}

    def _calculate_regime_consistency(self,
                                    window1: pd.DataFrame,
                                    window2: pd.DataFrame,
                                    regime_column: str) -> float:
        """
        Calculate consistency between two regime windows.

        Args:
            window1: First regime window
            window2: Second regime window
            regime_column: Name of regime column

        Returns:
            Consistency score (0-1)
        """
        try:
            regimes1 = set(window1[regime_column].unique())
            regimes2 = set(window2[regime_column].unique())

            # Jaccard similarity of regime sets
            intersection = len(regimes1 & regimes2)
            union = len(regimes1 | regimes2)

            if union == 0:
                return 0.0

            set_similarity = intersection / union

            # Distribution similarity
            dist1 = window1[regime_column].value_counts(normalize=True)
            dist2 = window2[regime_column].value_counts(normalize=True)

            # Align distributions
            all_regimes = sorted(regimes1 | regimes2)
            dist1_aligned = [dist1.get(regime, 0) for regime in all_regimes]
            dist2_aligned = [dist2.get(regime, 0) for regime in all_regimes]

            # Jensen-Shannon divergence (lower = more similar)
            from scipy.spatial.distance import jensenshannon
            js_divergence = jensenshannon(dist1_aligned, dist2_aligned)

            distribution_similarity = 1.0 - js_divergence

            # Combined consistency
            consistency = (set_similarity * 0.4 + distribution_similarity * 0.6)
            consistency = max(0.0, min(1.0, consistency))

            return consistency

        except Exception as e:
            self.logger.warning(f"⚠️ Regime consistency calculation failed: {e}")
            return 0.0

    def _perform_statistical_tests(self,
                                 regime_data: pd.DataFrame,
                                 market_data: pd.DataFrame,
                                 regime_column: str) -> Dict[str, Any]:
        """
        Perform statistical tests on regime data.

        Args:
            regime_data: DataFrame with regime assignments
            market_data: Market data
            regime_column: Name of regime column

        Returns:
            Dictionary of statistical test results
        """
        try:
            tests = {}

            # Collect returns for each regime
            regime_returns = {}
            unique_regimes = regime_data[regime_column].unique()

            for regime in unique_regimes:
                regime_mask = regime_data[regime_column] == regime
                returns = market_data.loc[regime_mask, 'close'].pct_change().dropna()
                if len(returns) >= 10:  # Minimum samples for statistical tests
                    regime_returns[regime] = returns

            if len(regime_returns) >= 2:
                # ANOVA test for equal means
                try:
                    from scipy.stats import f_oneway
                    returns_values = list(regime_returns.values())
                    f_stat, p_value = f_oneway(*returns_values)
                    tests['anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    tests['anova'] = {'error': str(e)}

                # Kruskal-Wallis test (non-parametric)
                try:
                    from scipy.stats import kruskal
                    h_stat, p_value = kruskal(*returns_values)
                    tests['kruskal_wallis'] = {
                        'h_statistic': h_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    tests['kruskal_wallis'] = {'error': str(e)}

            # Regime stationarity tests (simplified)
            tests['regime_stationarity'] = {}
            for regime, returns in regime_returns.items():
                try:
                    # Simple stationarity test using rolling statistics
                    rolling_mean = returns.rolling(20).mean()
                    rolling_std = returns.rolling(20).std()

                    # Check if statistics are relatively stable
                    mean_stability = 1.0 / (1.0 + rolling_mean.std())
                    std_stability = 1.0 / (1.0 + rolling_std.std())

                    tests['regime_stationarity'][f'regime_{regime}'] = {
                        'mean_stability': mean_stability,
                        'std_stability': std_stability,
                        'stationary': mean_stability > 0.7 and std_stability > 0.7
                    }
                except Exception as e:
                    tests['regime_stationarity'][f'regime_{regime}'] = {'error': str(e)}

            return tests

        except Exception as e:
            self.logger.warning(f"⚠️ Statistical tests failed: {e}")
            return {'error': str(e)}