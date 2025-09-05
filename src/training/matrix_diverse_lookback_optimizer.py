'\nMatrix-Based Diverse Lookback Period Optimizer\n\nThis module uses matrix/vector operations to efficiently find 2-3 lookback periods\nfor each feature that deliver meaningful yet significantly different information.\n'
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import optuna
import shap
from optuna.samplers import TPESampler
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from .utils.logger import system_logger
from .core.decorators.errors import handles_errors
from .utils.feature_calculators import FeatureCalculatorRegistry
import pandas as pd
import numpy as np

class MatrixDiverseLookbackOptimizer:
    """
    Matrix-based optimizer that finds diverse yet meaningful lookback periods for each feature.

    Uses matrix/vector operations for efficient optimization:
    - Matrix-based correlation analysis
    - Vectorized feature calculation
    - Matrix optimization for period selection
    - Vector-based diversity scoring
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the matrix-based diverse lookback optimizer."""
        self.config = config
        self.logger = system_logger.getChild('MatrixDiverseLookbackOptimizer')
        self.matrix_config = config.get('matrix_diverse_lookback_optimization', {'target_periods_per_feature': 3, 'min_periods_per_feature': 2, 'max_periods_per_feature': 3, 'diversity_threshold': 0.3, 'meaningful_threshold': 0.1, 'correlation_threshold': 0.7, 'quality_thresholds': {'min_diversity_score': 0.2, 'min_information_score': 0.05, 'max_correlation': 0.8, 'min_periods_for_3': 2}, 'matrix_optimization': {'enabled': True, 'method': 'scipy', 'max_iterations': 1000, 'tolerance': 1e-06}, 'vector_operations': {'enabled': True, 'batch_size': 1000, 'parallel_processing': True}, 'lookback_ranges': {'RSI': {'min': 5, 'max': 50, 'step': 2}, 'MACD_fast': {'min': 5, 'max': 25, 'step': 1}, 'MACD_slow': {'min': 20, 'max': 40, 'step': 2}, 'Bollinger_Bands': {'min': 10, 'max': 50, 'step': 2}, 'SMA_short': {'min': 3, 'max': 20, 'step': 1}, 'SMA_long': {'min': 20, 'max': 100, 'step': 5}, 'EMA_short': {'min': 3, 'max': 20, 'step': 1}, 'EMA_long': {'min': 20, 'max': 100, 'step': 5}, 'ATR': {'min': 5, 'max': 30, 'step': 1}, 'Stochastic_k': {'min': 5, 'max': 30, 'step': 1}, 'Stochastic_d': {'min': 3, 'max': 10, 'step': 1}, 'ADX': {'min': 5, 'max': 30, 'step': 1}, 'CCI': {'min': 5, 'max': 30, 'step': 1}, 'Williams_R': {'min': 5, 'max': 30, 'step': 1}, 'MFI': {'min': 5, 'max': 30, 'step': 1}, 'ROC': {'min': 5, 'max': 30, 'step': 1}, 'MOM': {'min': 5, 'max': 30, 'step': 1}, 'TSI': {'min': 5, 'max': 30, 'step': 1}, 'UO': {'min': 5, 'max': 30, 'step': 1}, 'AO': {'min': 5, 'max': 30, 'step': 1}, 'CMF': {'min': 5, 'max': 30, 'step': 1}, 'VWAP': {'min': 5, 'max': 30, 'step': 1}, 'Pivot_Points': {'min': 5, 'max': 30, 'step': 1}, 'Ichimoku': {'min': 5, 'max': 30, 'step': 1}, 'Parabolic_SAR': {'min': 5, 'max': 30, 'step': 1}, 'Keltner_Channels': {'min': 5, 'max': 30, 'step': 1}, 'Donchian_Channels': {'min': 5, 'max': 30, 'step': 1}, 'Price_Channels': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Profile': {'min': 5, 'max': 30, 'step': 1}, 'OBV': {'min': 5, 'max': 30, 'step': 1}, 'AD': {'min': 5, 'max': 30, 'step': 1}, 'Chaikin_Money_Flow': {'min': 5, 'max': 30, 'step': 1}, 'Money_Flow_Index': {'min': 5, 'max': 30, 'step': 1}, 'Volume_RSI': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Stochastic': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Price_Trend': {'min': 5, 'max': 30, 'step': 1}, 'Accumulation_Distribution': {'min': 5, 'max': 30, 'step': 1}, 'On_Balance_Volume': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Weighted_Average_Price': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Price_Oscillator': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Price_Confirmation': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Price_Trend_Indicator': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Price_Oscillator_Histogram': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Price_Oscillator_Signal': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Price_Oscillator_Trigger': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Price_Oscillator_Zero_Line': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Price_Oscillator_Upper_Band': {'min': 5, 'max': 30, 'step': 1}, 'Volume_Price_Oscillator_Lower_Band': {'min': 5, 'max': 30, 'step': 1}, 'VWAP_Momentum': {'min': 3, 'max': 50, 'step': 1}, 'VWAP_Acceleration': {'min': 3, 'max': 50, 'step': 1}, 'VWAP_Volatility': {'min': 5, 'max': 50, 'step': 1}, 'VWAP_Momentum_Volatility': {'min': 5, 'max': 50, 'step': 1}, 'VWAP_Returns': {'min': 5, 'max': 50, 'step': 1}, 'VWAP_Log_Returns': {'min': 5, 'max': 50, 'step': 1}, 'Price_VWAP_Ratio': {'min': 5, 'max': 50, 'step': 1}, 'Price_VWAP_Deviation': {'min': 5, 'max': 50, 'step': 1}, 'Price_VWAP_Spread': {'min': 5, 'max': 50, 'step': 1}}})
        self.output_dir = Path('data/matrix_diverse_lookback_optimization')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info('🚀 Matrix-Based Diverse Lookback Optimizer initialized')
        self.logger.info(f'📁 Output directory: {self.output_dir.absolute()}')

    @handles_errors(fallback={})
    async def find_diverse_lookback_periods_matrix(self, data: pd.DataFrame, target: pd.Series, regimes: pd.Series | None=None, symbol: str='UNKNOWN', exchange: str='UNKNOWN', timeframe: str='1m') -> dict[str, Any]:
        """
        Find diverse lookback periods using matrix/vector optimization.

        Args:
            data: Feature data
            target: Target variable
            regimes: HMM regime labels (optional)
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            Dictionary with diverse lookback periods and file paths
        """
        self.logger.info(f'🎯 Finding diverse lookback periods for {symbol} on {exchange}')
        results = {'optimization_timestamp': datetime.now().isoformat(), 'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'diverse_lookback_periods': {}, 'matrix_optimization_results': {}, 'file_paths': {}, 'optimization_metadata': {}}
        self.logger.info('🔍 Performing matrix-based diverse period optimization...')
        diverse_periods = await self._matrix_optimize_diverse_periods(data, target)
        results['diverse_lookback_periods'] = diverse_periods
        self.logger.info('📊 Analyzing matrix optimization results...')
        matrix_results = await self._analyze_matrix_optimization(data, target, diverse_periods)
        results['matrix_optimization_results'] = matrix_results
        self.logger.info('💾 Saving optimization results...')
        file_paths = await self._save_matrix_optimization_results(results, symbol, exchange, timeframe)
        results['file_paths'] = file_paths
        self.logger.info('⚡ Generating optimized feature parameters...')
        optimized_params = self._generate_optimized_feature_parameters(diverse_periods)
        results['optimized_feature_parameters'] = optimized_params
        self.logger.info('💾 Saving optimized parameters for subsequent steps...')
        params_file_path = await self._save_optimized_parameters(optimized_params, symbol, exchange, timeframe)
        results['file_paths']['optimized_parameters'] = params_file_path
        if regimes is not None and len(regimes.unique()) > 1:
            self.logger.info('🔄 Performing regime-specific matrix optimization...')
            regime_results = await self._matrix_optimize_regime_specific_periods(data, target, regimes, diverse_periods)
            results['regime_specific_periods'] = regime_results
        self._log_file_paths(results['file_paths'])
        self.logger.info('✅ Matrix-based diverse lookback period optimization completed')
        return results

    async def _matrix_optimize_diverse_periods(self, data: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
        """Optimize diverse periods using matrix operations."""
        diverse_periods = {}
        for feature_name, lookback_config in self.matrix_config['lookback_ranges'].items():
            self.logger.info(f'🔍 Matrix optimizing {feature_name}...')
            periods = list(range(lookback_config['min'], lookback_config['max'] + 1, lookback_config['step']))
            feature_periods = await self._matrix_optimize_feature_periods(data, target, feature_name, periods)
            diverse_periods[feature_name] = feature_periods
        return diverse_periods

    async def _matrix_optimize_feature_periods(self, data: pd.DataFrame, target: pd.Series, feature_name: str, periods: list[int]) -> dict[str, Any]:
        """Matrix-based optimization for feature periods."""
        self.logger.info(f'   Calculating features for {len(periods)} periods...')
        feature_matrix = self._calculate_feature_matrix(data, feature_name, periods)
        self.logger.info('   Calculating information scores...')
        info_scores = await self._calculate_vectorized_info_scores(feature_matrix, target)
        self.logger.info('   Performing correlation analysis...')
        correlation_matrix = self._calculate_correlation_matrix(feature_matrix)
        self.logger.info('   Optimizing period selection...')
        selected_indices = self._matrix_optimize_period_selection(info_scores, correlation_matrix, periods)
        selected_periods = [periods[i] for i in selected_indices]
        selected_features = feature_matrix[:, selected_indices]
        diversity_metrics = self._calculate_matrix_diversity_metrics(selected_features, correlation_matrix[selected_indices][:, selected_indices])
        return {'selected_periods': selected_periods, 'period_scores': [{'period': periods[i], 'information_score': info_scores[i], 'feature_values': feature_matrix[:, i]} for i in selected_indices], 'diversity_metrics': diversity_metrics, 'correlation_matrix': correlation_matrix.tolist(), 'all_period_scores': [{'period': p, 'information_score': s} for p, s in zip(periods, info_scores, strict=False)], 'optimization_method': self.matrix_config['matrix_optimization']['method']}

    def _calculate_feature_matrix(self, data: pd.DataFrame, feature_name: str, periods: list[int]) -> np.ndarray:
        """Calculate feature matrix for all periods using vectorized operations."""
        n_samples = len(data)
        n_periods = len(periods)
        feature_matrix = np.full((n_samples, n_periods), np.nan)
        for i, period in enumerate(periods):
            feature_values = self._calculate_feature_with_period(data, feature_name, period)
            if feature_values is not None:
                feature_matrix[:, i] = feature_values.values
        valid_rows = ~np.all(np.isnan(feature_matrix), axis=1)
        return feature_matrix[valid_rows]

    async def _calculate_vectorized_info_scores(self, feature_matrix: np.ndarray, target: pd.Series) -> np.ndarray:
        """Calculate information scores using vectorized operations."""
        n_periods = feature_matrix.shape[1]
        info_scores = np.zeros(n_periods)
        for i in range(n_periods):
            feature_values = feature_matrix[:, i]
            valid_mask = ~np.isnan(feature_values)
            if np.sum(valid_mask) < 100:
                info_scores[i] = 0.0
                continue
            X = feature_values[valid_mask].reshape(-1, 1)
            y = target.iloc[valid_mask].values
            try:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(X)
                info_scores[i] = np.mean(np.abs(shap_values))
            except Exception as e:
                self.logger.warning(f'⚠️ Error calculating SHAP for period {i}: {e}')
                info_scores[i] = 0.0
        return info_scores

    def _calculate_correlation_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix using vectorized operations."""
        valid_mask = ~np.any(np.isnan(feature_matrix), axis=1)
        clean_matrix = feature_matrix[valid_mask]
        correlation_matrix = np.corrcoef(clean_matrix.T)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        return np.abs(correlation_matrix)

    def _matrix_optimize_period_selection(self, info_scores: np.ndarray, correlation_matrix: np.ndarray, periods: list[int]) -> list[int]:
        """Optimize period selection using matrix operations with quality-based fallback."""
        target_count = min(self.matrix_config['target_periods_per_feature'], len(periods))
        if target_count == 0:
            return []
        meaningful_mask = info_scores >= self.matrix_config['meaningful_threshold']
        if np.sum(meaningful_mask) < self.matrix_config['min_periods_per_feature']:
            top_indices = np.argsort(info_scores)[-self.matrix_config['min_periods_per_feature']:]
            meaningful_mask[top_indices] = True
        meaningful_indices = np.where(meaningful_mask)[0]
        meaningful_scores = info_scores[meaningful_mask]
        meaningful_correlations = correlation_matrix[meaningful_mask][:, meaningful_mask]
        if target_count == 3 and len(meaningful_indices) >= 3:
            selected_indices = self._try_3_period_optimization(meaningful_scores, meaningful_correlations, meaningful_indices)
            if self._check_quality_thresholds(selected_indices, meaningful_scores, meaningful_correlations):
                return selected_indices
            self.logger.info("   ⚠️ 3-period solution doesn't meet quality thresholds, trying 2 periods")
        target_count = 2
        return self._try_2_period_optimization(meaningful_scores, meaningful_correlations, meaningful_indices)

    def _try_3_period_optimization(self, meaningful_scores: np.ndarray, meaningful_correlations: np.ndarray, meaningful_indices: np.ndarray) -> list[int]:
        """Try to optimize for 3 periods."""
        target_count = 3
        if self.matrix_config['matrix_optimization']['method'] == 'scipy':
            selected_indices = self._scipy_matrix_optimization(meaningful_scores, meaningful_correlations, target_count)
        elif self.matrix_config['matrix_optimization']['method'] == 'optuna':
            selected_indices = self._optuna_matrix_optimization(meaningful_scores, meaningful_correlations, target_count)
        else:
            selected_indices = self._greedy_matrix_optimization(meaningful_scores, meaningful_correlations, target_count)
        return [meaningful_indices[i] for i in selected_indices]

    def _try_2_period_optimization(self, meaningful_scores: np.ndarray, meaningful_correlations: np.ndarray, meaningful_indices: np.ndarray) -> list[int]:
        """Optimize for 2 periods."""
        target_count = 2
        if self.matrix_config['matrix_optimization']['method'] == 'scipy':
            selected_indices = self._scipy_matrix_optimization(meaningful_scores, meaningful_correlations, target_count)
        elif self.matrix_config['matrix_optimization']['method'] == 'optuna':
            selected_indices = self._optuna_matrix_optimization(meaningful_scores, meaningful_correlations, target_count)
        else:
            selected_indices = self._greedy_matrix_optimization(meaningful_scores, meaningful_correlations, target_count)
        return [meaningful_indices[i] for i in selected_indices]

    def _check_quality_thresholds(self, selected_indices: list[int], meaningful_scores: np.ndarray, meaningful_correlations: np.ndarray) -> bool:
        """Check if selected periods meet quality thresholds."""
        if len(selected_indices) < 2:
            return False
        quality_thresholds = self.matrix_config['quality_thresholds']
        selected_scores = [meaningful_scores[i] for i in selected_indices]
        min_info_score = min(selected_scores)
        if min_info_score < quality_thresholds['min_information_score']:
            return False
        if len(selected_indices) >= 2:
            selected_corr = meaningful_correlations[selected_indices][:, selected_indices]
            np.fill_diagonal(selected_corr, 0)
            max_correlation = np.max(selected_corr)
            if max_correlation > quality_thresholds['max_correlation']:
                return False
        diversity_score = self._calculate_diversity_score(selected_indices, meaningful_correlations)
        return not diversity_score < quality_thresholds['min_diversity_score']

    def _calculate_diversity_score(self, selected_indices: list[int], correlation_matrix: np.ndarray) -> float:
        """Calculate diversity score for selected periods."""
        if len(selected_indices) < 2:
            return 0.0
        total_correlation = 0.0
        count = 0
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                correlation = correlation_matrix[selected_indices[i], selected_indices[j]]
                total_correlation += correlation
                count += 1
        avg_correlation = total_correlation / count if count > 0 else 1.0
        return 1.0 - avg_correlation

    def _scipy_matrix_optimization(self, info_scores: np.ndarray, correlation_matrix: np.ndarray, target_count: int) -> list[int]:
        """Matrix optimization using SciPy."""
        n_periods = len(info_scores)

        def objective(x: Any) -> float:
            if np.sum(x) != target_count:
                return 1000000.0
            selected_mask = x.astype(bool)
            info_component = -np.sum(info_scores[selected_mask])
            selected_correlations = correlation_matrix[selected_mask][:, selected_mask]
            np.fill_diagonal(selected_correlations, 0)
            diversity_penalty = np.sum(selected_correlations) * 0.5
            return info_component + diversity_penalty

        def constraint(x: Any) -> None:
            return np.sum(x) - target_count
        initial_guess = np.zeros(n_periods)
        top_indices = np.argsort(info_scores)[-target_count:]
        initial_guess[top_indices] = 1
        result = minimize(objective, initial_guess, constraints={'type': 'eq', 'fun': constraint}, bounds=[(0, 1)] * n_periods, method='SLSQP')
        selected_indices = np.where(result.x > 0.5)[0]
        return selected_indices.tolist()

    def _optuna_matrix_optimization(self, info_scores: np.ndarray, correlation_matrix: np.ndarray, target_count: int) -> list[int]:
        """Matrix optimization using Optuna."""

        def objective(trial: Any) -> None:
            selected_indices = trial.suggest_categorical('selected_periods', [list(combo) for combo in itertools.combinations(range(len(info_scores)), target_count)])
            info_component = -np.sum(info_scores[selected_indices])
            selected_correlations = correlation_matrix[selected_indices][:, selected_indices]
            np.fill_diagonal(selected_correlations, 0)
            diversity_penalty = np.sum(selected_correlations) * 0.5
            return info_component + diversity_penalty
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        return best_params['selected_periods']

    def _greedy_matrix_optimization(self, info_scores: np.ndarray, correlation_matrix: np.ndarray, target_count: int) -> list[int]:
        """Greedy matrix optimization."""
        selected_indices = [np.argmax(info_scores)]
        while len(selected_indices) < target_count:
            best_candidate = None
            best_score = -np.inf
            for i in range(len(info_scores)):
                if i in selected_indices:
                    continue
                candidate_set = selected_indices + [i]
                info_score = np.sum(info_scores[candidate_set])
                candidate_correlations = correlation_matrix[candidate_set][:, candidate_set]
                np.fill_diagonal(candidate_correlations, 0)
                diversity_score = -np.sum(candidate_correlations)
                combined_score = info_score + diversity_score * 0.5
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = i
            if best_candidate is not None:
                selected_indices.append(best_candidate)
            else:
                break
        return selected_indices

    def _calculate_matrix_diversity_metrics(self, selected_features: np.ndarray, correlation_matrix: np.ndarray) -> dict[str, float]:
        """Calculate diversity metrics using matrix operations."""
        if selected_features.shape[1] < 2:
            return {'diversity_score': 0.0, 'avg_correlation': 1.0}
        n_periods = selected_features.shape[1]
        total_correlation = 0.0
        count = 0
        for i in range(n_periods):
            for j in range(i + 1, n_periods):
                total_correlation += correlation_matrix[i, j]
                count += 1
        avg_correlation = total_correlation / count if count > 0 else 1.0
        diversity_score = 1.0 - avg_correlation
        return {'diversity_score': diversity_score, 'avg_correlation': avg_correlation, 'n_periods': n_periods, 'correlation_matrix': correlation_matrix.tolist()}

    async def _save_matrix_optimization_results(self, results: dict[str, Any], symbol: str, exchange: str, timeframe: str) -> dict[str, str]:
        """Save matrix optimization results with detailed file logging."""
        file_paths = {}
        main_filename = f'{exchange}_{symbol}_{timeframe}_matrix_diverse_lookback_periods.json'
        main_filepath = self.output_dir / main_filename
        with open(main_filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        file_paths['main_results'] = str(main_filepath.absolute())
        self.logger.info(f'💾 Saved main results to: {main_filepath.absolute()}')
        summary_filename = f'{exchange}_{symbol}_{timeframe}_diverse_periods_summary.json'
        summary_filepath = self.output_dir / summary_filename
        summary_data = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'optimization_timestamp': results['optimization_timestamp'], 'diverse_periods': {feature: data['selected_periods'] for feature, data in results['diverse_lookback_periods'].items()}, 'diversity_scores': {feature: data['diversity_metrics']['diversity_score'] for feature, data in results['diverse_lookback_periods'].items()}}
        with open(summary_filepath, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        file_paths['summary'] = str(summary_filepath.absolute())
        self.logger.info(f'💾 Saved summary to: {summary_filepath.absolute()}')
        matrix_filename = f'{exchange}_{symbol}_{timeframe}_matrix_optimization_details.json'
        matrix_filepath = self.output_dir / matrix_filename
        with open(matrix_filepath, 'w') as f:
            json.dump(results['matrix_optimization_results'], f, indent=2, default=str)
        file_paths['matrix_details'] = str(matrix_filepath.absolute())
        self.logger.info(f'💾 Saved matrix details to: {matrix_filepath.absolute()}')
        return file_paths

    def _generate_optimized_feature_parameters(self, diverse_periods: dict[str, Any]) -> dict[str, Any]:
        """Generate optimized feature parameters for subsequent steps."""
        optimized_params = {}
        for feature_name, feature_data in diverse_periods.items():
            selected_periods = feature_data['selected_periods']
            feature_params = []
            for period in selected_periods:
                if feature_name == 'RSI':
                    param = {'lookback_period': period, 'overbought_threshold': 75, 'oversold_threshold': 25}
                elif feature_name == 'MACD_fast':
                    param = {'fast_period': period, 'slow_period': period * 2, 'signal_period': 9}
                elif feature_name == 'MACD_slow':
                    param = {'fast_period': 12, 'slow_period': period, 'signal_period': 9}
                elif feature_name == 'Bollinger_Bands':
                    param = {'lookback_period': period, 'std_dev': 2.0, 'squeeze_threshold': 0.2}
                elif feature_name in ['SMA_short', 'SMA_long'] or feature_name in ['EMA_short', 'EMA_long']:
                    param = {'short_period': period, 'long_period': period * 2}
                elif feature_name == 'ATR':
                    param = {'lookback_period': period}
                elif feature_name == 'Stochastic_k':
                    param = {'k_period': period, 'd_period': 3, 'overbought': 80, 'oversold': 20}
                elif feature_name == 'Stochastic_d':
                    param = {'k_period': 14, 'd_period': period, 'overbought': 80, 'oversold': 20}
                elif feature_name == 'ADX':
                    param = {'lookback_period': period, 'threshold': 25}
                elif feature_name == 'CCI':
                    param = {'lookback_period': period, 'constant': 0.015}
                else:
                    param = {'lookback_period': period}
                feature_params.append(param)
            optimized_params[feature_name] = {'selected_periods': selected_periods, 'parameters': feature_params, 'diversity_score': feature_data['diversity_metrics']['diversity_score']}
        return optimized_params

    async def _save_optimized_parameters(self, optimized_params: dict[str, Any], symbol: str, exchange: str, timeframe: str) -> str:
        """Save optimized parameters for subsequent steps."""
        params_filename = f'{exchange}_{symbol}_{timeframe}_optimized_feature_parameters.json'
        params_filepath = self.output_dir / params_filename
        with open(params_filepath, 'w') as f:
            json.dump(optimized_params, f, indent=2, default=str)
        self.logger.info(f'💾 Saved optimized parameters to: {params_filepath.absolute()}')
        step_params_dir = Path('data/optimized_feature_parameters')
        step_params_dir.mkdir(parents=True, exist_ok=True)
        step_params_filepath = step_params_dir / params_filename
        with open(step_params_filepath, 'w') as f:
            json.dump(optimized_params, f, indent=2, default=str)
        self.logger.info(f'💾 Saved step parameters to: {step_params_filepath.absolute()}')
        return str(step_params_filepath.absolute())

    def _log_file_paths(self, file_paths: dict[str, str]) -> None:
        """Log all file paths for review."""
        self.logger.info('📁 OPTIMIZATION FILES SAVED:')
        self.logger.info('=' * 50)
        for file_type, file_path in file_paths.items():
            self.logger.info(f'{file_type.upper()}: {file_path}')
        self.logger.info('=' * 50)
        self.logger.info('📋 All files are ready for review and subsequent steps!')

    def get_optimized_feature_parameters(self, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Load optimized feature parameters for subsequent steps."""
        step_params_filepath = Path(f'data/optimized_feature_parameters/{exchange}_{symbol}_{timeframe}_optimized_feature_parameters.json')
        if not step_params_filepath.exists():
            main_params_filepath = Path(f'data/matrix_diverse_lookback_optimization/{exchange}_{symbol}_{timeframe}_optimized_feature_parameters.json')
            if not main_params_filepath.exists():
                self.logger.warning(f'⚠️ No optimized parameters found for {symbol} on {exchange}')
                return {}
            step_params_filepath = main_params_filepath
        try:
            with open(step_params_filepath) as f:
                optimized_params = json.load(f)
            self.logger.info(f'📂 Loaded optimized parameters from: {step_params_filepath.absolute()}')
            return optimized_params
        except Exception as e:
            self.logger.exception(f'❌ Error loading optimized parameters: {e}')
            return {}

    def _calculate_feature_with_period(self, data: pd.DataFrame, feature_name: str, period: int) -> pd.Series | None:
        """Calculate feature with specific lookback period using the feature calculator registry."""
        try:
            return FeatureCalculatorRegistry.calculate_feature(data, feature_name, period)
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating {feature_name} with period {period}: {e}')
            return None


    async def _analyze_matrix_optimization(self, data: pd.DataFrame, target: pd.Series, diverse_periods: dict[str, Any]) -> dict[str, Any]:
        """Analyze matrix optimization results."""
        analysis = {'optimization_method': self.matrix_config['matrix_optimization']['method'], 'matrix_operations_used': ['Vectorized feature calculation', 'Matrix correlation analysis', 'Vectorized information scoring', 'Matrix-based period selection'], 'performance_metrics': {}, 'diversity_analysis': {}}
        total_periods_tested = 0
        total_periods_selected = 0
        avg_diversity_score = 0.0
        for feature_data in diverse_periods.values():
            total_periods_tested += len(feature_data['all_period_scores'])
            total_periods_selected += len(feature_data['selected_periods'])
            avg_diversity_score += feature_data['diversity_metrics']['diversity_score']
        n_features = len(diverse_periods)
        if n_features > 0:
            avg_diversity_score /= n_features
        analysis['performance_metrics'] = {'total_periods_tested': total_periods_tested, 'total_periods_selected': total_periods_selected, 'reduction_ratio': total_periods_selected / total_periods_tested if total_periods_tested > 0 else 0.0, 'avg_diversity_score': avg_diversity_score, 'n_features_optimized': n_features}
        return analysis

    async def _matrix_optimize_regime_specific_periods(self, data: pd.DataFrame, target: pd.Series, regimes: pd.Series, global_periods: dict[str, Any]) -> dict[str, Any]:
        """Matrix optimization for regime-specific periods."""
        regime_results = {}
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_data = data[regime_mask]
            regime_target = target[regime_mask]
            if len(regime_data) >= 100:
                self.logger.info(f'🔄 Matrix optimizing regime {regime}...')
                regime_specific = await self._matrix_optimize_diverse_periods(regime_data, regime_target)
                regime_results[f'regime_{regime}'] = regime_specific
        return regime_results