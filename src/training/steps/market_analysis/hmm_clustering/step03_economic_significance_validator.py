import numpy as np
from ..standardized_parquet_handler import standardized_parquet_handler
import pandas as pd

'Economic Significance Testing for Regime Validation - Pre-ML Model Approach.\n\nThis module validates that discovered regimes have economically meaningful differences\nBEFORE training ML models, using statistical tests on market data characteristics.\n'
from scipy import stats
import warnings
import os
import typing

warnings.filterwarnings('ignore')

class EconomicSignificanceValidator:
    """Validate economic significance of regimes before ML model training."""

    def __init__(self, config: Dict[str, Any]=None) -> None:
        self.config = config or {}
        self.significance_threshold = self.config.get('significance_threshold', 0.05)
        self.economic_threshold = self.config.get('economic_threshold', 0.001)

    def validate_regime_economics(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """
        Validate that regimes have economically meaningful differences.
        
        Args:
            data: Market data with OHLCV columns
            regimes: Regime labels for each data point
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {'overall_significant': False, 'return_significance': {}, 'risk_significance': {}, 'volume_significance': {}, 'volatility_significance': {}, 'regime_characteristics': {}, 'economic_metrics': {}, 'validation_summary': {}}
        returns = self._calculate_regime_returns(data, regimes)
        return_tests = self._test_return_distributions(regimes, returns)
        validation_results['return_significance'] = return_tests
        risk_tests = self._test_risk_adjusted_performance(regimes, returns)
        validation_results['risk_significance'] = risk_tests
        volume_tests = self._test_volume_patterns(data, regimes)
        validation_results['volume_significance'] = volume_tests
        volatility_tests = self._test_volatility_patterns(data, regimes)
        validation_results['volatility_significance'] = volatility_tests
        momentum_tests = self._test_momentum_patterns(data, regimes)
        validation_results['momentum_significance'] = momentum_tests
        regime_chars = self._calculate_regime_characteristics(data, regimes)
        validation_results['regime_characteristics'] = regime_chars
        economic_metrics = self._calculate_economic_metrics(data, regimes, returns)
        validation_results['economic_metrics'] = economic_metrics
        overall_significance = self._assess_overall_significance(validation_results)
        validation_results['overall_significant'] = overall_significance
        validation_summary = self._generate_validation_summary(validation_results)
        validation_results['validation_summary'] = validation_summary
        return validation_results

    def _calculate_regime_returns(self, data: pd.DataFrame, regimes: np.ndarray) -> pd.Series:
        """Calculate returns for each regime."""
        returns = data['close'].pct_change()
        regime_returns = pd.Series(index = data.index, dtype = float)
        for regime in np.unique(regimes):
            regime_mask = regimes == regime
            regime_data = data[regime_mask]
            if len(regime_data) > 1:
                regime_returns[regime_mask] = regime_data['close'].pct_change()
        return regime_returns.fillna(0)

    def _test_return_distributions(self, regimes: np.ndarray, returns: pd.Series) -> Dict[str, Any]:
        """Test if regime return distributions are significantly different."""
        results = {'statistically_significant': False, 'economically_significant': False, 'pairwise_tests': {}, 'overall_ks_test': {}, 'overall_mw_test': {}}
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        if n_regimes < 2:
            return results
        regime_returns = {}
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_returns[regime] = returns[regime_mask].dropna()
        significant_pairs = 0
        total_pairs = 0
        economic_pairs = 0
        for i, regime1 in enumerate(unique_regimes):
            for regime2 in unique_regimes[i + 1:]:
                if len(regime_returns[regime1]) > 10 and len(regime_returns[regime2]) > 10:
                    ks_stat, ks_pvalue = ks_2samp(regime_returns[regime1], regime_returns[regime2])
                    mw_stat, mw_pvalue = mannwhitneyu(regime_returns[regime1], regime_returns[regime2], alternative='two-sided')
                    mean_diff = np.mean(regime_returns[regime1]) - np.mean(regime_returns[regime2])
                    economic_significance = abs(mean_diff) > self.economic_threshold
                    statistical_significance = ks_pvalue < self.significance_threshold and mw_pvalue < self.significance_threshold
                    results['pairwise_tests'][f'regime_{regime1}_vs_{regime2}'] = {'ks_statistic': ks_stat, 'ks_pvalue': ks_pvalue, 'mw_statistic': mw_stat, 'mw_pvalue': mw_pvalue, 'mean_return_diff': mean_diff, 'economically_significant': economic_significance, 'statistically_significant': statistical_significance, 'regime1_mean': np.mean(regime_returns[regime1]), 'regime1_std': np.std(regime_returns[regime1]), 'regime2_mean': np.mean(regime_returns[regime2]), 'regime2_std': np.std(regime_returns[regime2])}
                    total_pairs += 1
                    if statistical_significance:
                        significant_pairs += 1
                    if economic_significance:
                        economic_pairs += 1
        results['statistically_significant'] = significant_pairs / total_pairs > 0.5 if total_pairs > 0 else False
        results['economically_significant'] = economic_pairs / total_pairs > 0.5 if total_pairs > 0 else False
        return results

    def _test_risk_adjusted_performance(self, regimes: np.ndarray, returns: pd.Series) -> Dict[str, Any]:
        """Test if regimes have different risk-adjusted performance."""
        results = {'sharpe_ratios': {}, 'sortino_ratios': {}, 'max_drawdowns': {}, 'volatilities': {}, 'risk_adjusted_significant': False}
        unique_regimes = np.unique(regimes)
        sharpe_ratios = []
        sortino_ratios = []
        max_drawdowns = []
        volatilities = []
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask].dropna()
            if len(regime_returns) > 20:
                mean_return = np.mean(regime_returns) * 252
                volatility = np.std(regime_returns) * np.sqrt(252)
                sharpe = mean_return / volatility if volatility > 0 else 0
                sharpe_ratios.append(sharpe)
                downside_returns = regime_returns[regime_returns < 0]
                downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
                sortino = mean_return / downside_volatility if downside_volatility > 0 else 0
                sortino_ratios.append(sortino)
                cumulative = np.cumprod(1 + regime_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown)
                max_drawdowns.append(max_drawdown)
                volatilities.append(volatility)
                results['sharpe_ratios'][f'regime_{regime}'] = sharpe
                results['sortino_ratios'][f'regime_{regime}'] = sortino
                results['max_drawdowns'][f'regime_{regime}'] = max_drawdown
                results['volatilities'][f'regime_{regime}'] = volatility
        if len(sharpe_ratios) > 1:
            try:
                f_stat, f_pvalue = stats.f_oneway(*[returns[regimes == r].dropna() for r in unique_regimes])
                results['risk_adjusted_significant'] = f_pvalue < self.significance_threshold
            except:
                results['risk_adjusted_significant'] = False
        return results

    def _test_volume_patterns(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Test if regimes have different volume patterns."""
        results = {'volume_means': {}, 'volume_volatilities': {}, 'volume_distributions': {}, 'volume_significant': False}
        unique_regimes = np.unique(regimes)
        volume_data = []
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_volume = data['volume'][regime_mask]
            if len(regime_volume) > 10:
                results['volume_means'][f'regime_{regime}'] = np.mean(regime_volume)
                results['volume_volatilities'][f'regime_{regime}'] = np.std(regime_volume)
                volume_data.append(regime_volume)
        if len(volume_data) > 1:
            try:
                f_stat, f_pvalue = stats.f_oneway(*volume_data)
                results['volume_significant'] = f_pvalue < self.significance_threshold
            except:
                results['volume_significant'] = False
        return results

    def _test_volatility_patterns(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Test if regimes have different volatility patterns."""
        results = {'volatility_means': {}, 'volatility_volatilities': {}, 'volatility_distributions': {}, 'volatility_significant': False}
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        unique_regimes = np.unique(regimes)
        volatility_data = []
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_volatility = volatility[regime_mask].dropna()
            if len(regime_volatility) > 10:
                results['volatility_means'][f'regime_{regime}'] = np.mean(regime_volatility)
                results['volatility_volatilities'][f'regime_{regime}'] = np.std(regime_volatility)
                volatility_data.append(regime_volatility)
        if len(volatility_data) > 1:
            try:
                f_stat, f_pvalue = stats.f_oneway(*volatility_data)
                results['volatility_significant'] = f_pvalue < self.significance_threshold
            except:
                results['volatility_significant'] = False
        return results

    def _test_momentum_patterns(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Test if regimes have different momentum patterns."""
        results = {'momentum_means': {}, 'momentum_volatilities': {}, 'momentum_distributions': {}, 'momentum_significant': False}
        momentum_5 = data['close'].pct_change(5)
        momentum_10 = data['close'].pct_change(10)
        momentum_20 = data['close'].pct_change(20)
        unique_regimes = np.unique(regimes)
        momentum_data_5 = []
        momentum_data_10 = []
        momentum_data_20 = []
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_momentum_5 = momentum_5[regime_mask].dropna()
            regime_momentum_10 = momentum_10[regime_mask].dropna()
            regime_momentum_20 = momentum_20[regime_mask].dropna()
            if len(regime_momentum_5) > 10:
                results['momentum_means'][f'regime_{regime}_5d'] = np.mean(regime_momentum_5)
                results['momentum_volatilities'][f'regime_{regime}_5d'] = np.std(regime_momentum_5)
                momentum_data_5.append(regime_momentum_5)
            if len(regime_momentum_10) > 10:
                results['momentum_means'][f'regime_{regime}_10d'] = np.mean(regime_momentum_10)
                results['momentum_volatilities'][f'regime_{regime}_10d'] = np.std(regime_momentum_10)
                momentum_data_10.append(regime_momentum_10)
            if len(regime_momentum_20) > 10:
                results['momentum_means'][f'regime_{regime}_20d'] = np.mean(regime_momentum_20)
                results['momentum_volatilities'][f'regime_{regime}_20d'] = np.std(regime_momentum_20)
                momentum_data_20.append(regime_momentum_20)
        momentum_significant = False
        if len(momentum_data_5) > 1:
            try:
                f_stat, f_pvalue = stats.f_oneway(*momentum_data_5)
                if f_pvalue < self.significance_threshold:
                    momentum_significant = True
            except:
                pass
        if len(momentum_data_10) > 1:
            try:
                f_stat, f_pvalue = stats.f_oneway(*momentum_data_10)
                if f_pvalue < self.significance_threshold:
                    momentum_significant = True
            except:
                pass
        if len(momentum_data_20) > 1:
            try:
                f_stat, f_pvalue = stats.f_oneway(*momentum_data_20)
                if f_pvalue < self.significance_threshold:
                    momentum_significant = True
            except:
                pass
        results['momentum_significant'] = momentum_significant
        results['momentum_autocorrelation'] = self._test_momentum_autocorrelation(data, regimes)
        results['momentum_persistence'] = self._test_momentum_persistence(data, regimes)
        results['momentum_reversal_patterns'] = self._test_momentum_reversal_patterns(data, regimes)
        return results

    def _test_momentum_autocorrelation(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Test momentum autocorrelation patterns across regimes."""
        results = {}
        momentum_5 = data['close'].pct_change(5)
        momentum_10 = data['close'].pct_change(10)
        unique_regimes = np.unique(regimes)
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_momentum_5 = momentum_5[regime_mask].dropna()
            regime_momentum_10 = momentum_10[regime_mask].dropna()
            if len(regime_momentum_5) > 20:
                autocorr_5 = regime_momentum_5.autocorr(lag = 1)
                autocorr_10 = regime_momentum_10.autocorr(lag = 1)
                results[f'regime_{regime}_autocorr_5d'] = autocorr_5
                results[f'regime_{regime}_autocorr_10d'] = autocorr_10
        return results

    def _test_momentum_persistence(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Test momentum persistence patterns across regimes."""
        results = {}
        momentum_5 = data['close'].pct_change(5)
        unique_regimes = np.unique(regimes)
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_momentum = momentum_5[regime_mask].dropna()
            if len(regime_momentum) > 20:
                momentum_sign = np.sign(regime_momentum)
                momentum_continuation = np.sum(momentum_sign[:-1] == momentum_sign[1:]) / len(momentum_sign[:-1])
                results[f'regime_{regime}_momentum_persistence'] = momentum_continuation
        return results

    def _test_momentum_reversal_patterns(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Test momentum reversal patterns across regimes."""
        results = {}
        momentum_5 = data['close'].pct_change(5)
        momentum_10 = data['close'].pct_change(10)
        unique_regimes = np.unique(regimes)
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_momentum_5 = momentum_5[regime_mask].dropna()
            regime_momentum_10 = momentum_10[regime_mask].dropna()
            if len(regime_momentum_5) > 20:
                momentum_sign_5 = np.sign(regime_momentum_5)
                momentum_reversals_5 = np.sum(momentum_sign_5[:-1] != momentum_sign_5[1:]) / len(momentum_sign_5[:-1])
                momentum_sign_10 = np.sign(regime_momentum_10)
                momentum_reversals_10 = np.sum(momentum_sign_10[:-1] != momentum_sign_10[1:]) / len(momentum_sign_10[:-1])
                results[f'regime_{regime}_reversal_rate_5d'] = momentum_reversals_5
                results[f'regime_{regime}_reversal_rate_10d'] = momentum_reversals_10
        return results

    def _calculate_regime_characteristics(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Calculate detailed regime characteristics."""
        results = {'regime_sizes': {}, 'regime_durations': {}, 'regime_transitions': {}, 'regime_balance': 0.0}
        unique_regimes = np.unique(regimes)
        regime_sizes = []
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_size = np.sum(regime_mask)
            regime_sizes.append(regime_size)
            results['regime_sizes'][f'regime_{regime}'] = regime_size
            regime_durations = self._calculate_regime_durations(regimes, regime)
            results['regime_durations'][f'regime_{regime}'] = {'mean_duration': np.mean(regime_durations), 'median_duration': np.median(regime_durations), 'max_duration': np.max(regime_durations), 'min_duration': np.min(regime_durations)}
        if len(regime_sizes) > 1:
            size_variance = np.var(regime_sizes)
            size_mean = np.mean(regime_sizes)
            results['regime_balance'] = 1 / (1 + size_variance / size_mean) if size_mean > 0 else 0
        results['regime_transitions'] = self._calculate_regime_transition_matrix(regimes)
        return results

    def _calculate_regime_durations(self, regimes: np.ndarray, target_regime: int) -> List[int]:
        """Calculate durations of a specific regime."""
        durations = []
        current_duration = 0
        for regime in regimes:
            if regime == target_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            durations.append(current_duration)
        return durations if durations else [0]

    def _calculate_regime_transition_matrix(self, regimes: np.ndarray) -> Dict[str, Any]:
        """Calculate regime transition matrix."""
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        if n_regimes < 2:
            return {'transition_matrix': np.array([]), 'transition_probabilities': {}}
        transition_matrix = np.zeros((n_regimes, n_regimes))
        regime_map = {regime: i for i, regime in enumerate(unique_regimes)}
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            current_idx = regime_map[current_regime]
            next_idx = regime_map[next_regime]
            transition_matrix[current_idx, next_idx] += 1
        row_sums = transition_matrix.sum(axis = 1, keepdims = True)
        transition_probabilities = np.divide(transition_matrix, row_sums, where = row_sums > 0)
        readable_transitions = {}
        for i, from_regime in enumerate(unique_regimes):
            readable_transitions[f'from_regime_{from_regime}'] = {}
            for j, to_regime in enumerate(unique_regimes):
                readable_transitions[f'from_regime_{from_regime}'][f'to_regime_{to_regime}'] = float(transition_probabilities[i, j])
        return {'transition_matrix': transition_matrix.tolist(), 'transition_probabilities': readable_transitions, 'regime_map': regime_map}

    def _calculate_economic_metrics(self, data: pd.DataFrame, regimes: np.ndarray, returns: pd.Series) -> Dict[str, Any]:
        """Calculate economic metrics for each regime."""
        results = {'regime_returns': {}, 'regime_volatilities': {}, 'regime_sharpe_ratios': {}, 'regime_max_drawdowns': {}, 'regime_win_rates': {}, 'regime_profit_factors': {}}
        unique_regimes = np.unique(regimes)
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask].dropna()
            if len(regime_returns) > 10:
                mean_return = np.mean(regime_returns)
                volatility = np.std(regime_returns)
                sharpe = mean_return / volatility if volatility > 0 else 0
                cumulative = np.cumprod(1 + regime_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown)
                win_rate = np.mean(regime_returns > 0)
                gross_profit = np.sum(regime_returns[regime_returns > 0])
                gross_loss = abs(np.sum(regime_returns[regime_returns < 0]))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                results['regime_returns'][f'regime_{regime}'] = mean_return
                results['regime_volatilities'][f'regime_{regime}'] = volatility
                results['regime_sharpe_ratios'][f'regime_{regime}'] = sharpe
                results['regime_max_drawdowns'][f'regime_{regime}'] = max_drawdown
                results['regime_win_rates'][f'regime_{regime}'] = win_rate
                results['regime_profit_factors'][f'regime_{regime}'] = profit_factor
        return results

    def _assess_overall_significance(self, validation_results: Dict[str, Any]) -> bool:
        """Assess overall economic significance."""
        return_tests = validation_results['return_significance']
        risk_tests = validation_results['risk_significance']
        volume_tests = validation_results['volume_significance']
        volatility_tests = validation_results['volatility_significance']
        momentum_tests = validation_results['momentum_significance']
        overall_significant = return_tests.get('statistically_significant', False) or return_tests.get('economically_significant', False) or risk_tests.get('risk_adjusted_significant', False) or volume_tests.get('volume_significant', False) or volatility_tests.get('volatility_significant', False) or momentum_tests.get('momentum_significant', False)
        return overall_significant

    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {'overall_significant': validation_results['overall_significant'], 'n_regimes': len(validation_results['regime_characteristics']['regime_sizes']), 'regime_balance_score': validation_results['regime_characteristics']['regime_balance'], 'significant_tests': [], 'recommendations': []}
        if validation_results['return_significance'].get('statistically_significant', False):
            summary['significant_tests'].append('return_distributions')
        if validation_results['return_significance'].get('economically_significant', False):
            summary['significant_tests'].append('return_economics')
        if validation_results['risk_significance'].get('risk_adjusted_significant', False):
            summary['significant_tests'].append('risk_adjusted_performance')
        if validation_results['volume_significance'].get('volume_significant', False):
            summary['significant_tests'].append('volume_patterns')
        if validation_results['volatility_significance'].get('volatility_significant', False):
            summary['significant_tests'].append('volatility_patterns')
        if validation_results['momentum_significance'].get('momentum_significant', False):
            summary['significant_tests'].append('momentum_patterns')
        if not summary['overall_significant']:
            summary['recommendations'].append('Regimes do not show significant economic differences. Consider adjusting regime detection parameters.')
        if summary['regime_balance_score'] < 0.5:
            summary['recommendations'].append('Regime sizes are highly imbalanced. Consider adjusting clustering parameters.')
        if len(summary['significant_tests']) == 0:
            summary['recommendations'].append('No significant differences found. Regimes may not be economically meaningful.')
        else:
            summary['recommendations'].append(f"Found {len(summary['significant_tests'])} significant differences. Regimes appear economically meaningful.")
        return summary
if __name__ == '__main__':
    np.random.seed(42)
    n_samples = 1000
    regime1_returns = np.random.normal(0.001, 0.02, 500)
    regime2_returns = np.random.normal(-0.0005, 0.03, 500)
    all_returns = np.concatenate([regime1_returns, regime2_returns])
    regimes = np.concatenate([np.ones(500), np.ones(500) * 2])
    prices = 100 * np.cumprod(1 + all_returns)
    data = pd.DataFrame({'open': prices, 'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.01)), 'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.01)), 'close': prices, 'volume': np.random.lognormal(10, 1, n_samples)})
    validator = EconomicSignificanceValidator()
    results = validator.validate_regime_economics(data, regimes)
    print('Economic Significance Validation Results:')
    print(f"Overall Significant: {results['overall_significant']}")
    print(f"Number of Regimes: {results['validation_summary']['n_regimes']}")
    print(f"Significant Tests: {results['validation_summary']['significant_tests']}")
    print(f"Recommendations: {results['validation_summary']['recommendations']}")
    print('\nDetailed Results:')
    print(f"Return Significance: {results['return_significance']['statistically_significant']}")
    print(f"Economic Significance: {results['return_significance']['economically_significant']}")
    print(f"Risk Adjusted Significance: {results['risk_significance']['risk_adjusted_significant']}")
    print(f"Volume Significance: {results['volume_significance']['volume_significant']}")
    print(f"Volatility Significance: {results['volatility_significance']['volatility_significant']}")