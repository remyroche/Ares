from __future__ import annotations
'\nTimeframe Relevance Analyzer\n\nThis module analyzes the relevance of different timeframes for high leverage trading (10x-100x)\nand optimizes the ensemble configuration accordingly.\n'
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.utils.logger import system_logger
import asyncio

class TimeframeRelevanceAnalyzer:
    """
    Analyzes timeframe relevance for high leverage trading and optimizes ensemble configuration.

    Features:
    - Analyzes volatility patterns across timeframes
    - Evaluates signal quality for high leverage scenarios
    - Determines optimal timeframe weights for ensemble
    - Identifies timeframes that may be irrelevant for high leverage
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the timeframe relevance analyzer."""
        self.config = config
        self.logger = system_logger.getChild('TimeframeRelevanceAnalyzer')
        self.leverage_config = config.get('high_leverage_trading', {'min_leverage': 10, 'max_leverage': 100, 'target_leverage': 25, 'max_drawdown_threshold': 0.05, 'volatility_threshold': 0.02, 'signal_quality_threshold': 0.6, 'timeframe_analysis_window': 30})
        self.analysis_config = config.get('timeframe_analysis', {'min_data_points': 1000, 'volatility_lookback': 20, 'correlation_threshold': 0.7, 'signal_decay_factor': 0.95, 'ensemble_weight_min': 0.05, 'ensemble_weight_max': 0.5})
        self.logger.info('🚀 Timeframe Relevance Analyzer initialized')

    @handles_errors(fallback={})
    async def analyze_timeframe_relevance(self, data_dict: dict[str, pd.DataFrame], symbol: str='UNKNOWN', exchange: str='UNKNOWN', leverage_range: tuple[int, int]=(10, 100)) -> dict[str, Any]:
        """
        Analyze timeframe relevance for high leverage trading.

        Args:
            data_dict: Dictionary mapping timeframes to DataFrames
            symbol: Trading symbol
            exchange: Exchange name
            leverage_range: Range of leverage to analyze (min, max)

        Returns:
            Dictionary with timeframe relevance analysis and optimized ensemble config
        """
        self.logger.info(f'🎯 Starting timeframe relevance analysis for {symbol} on {exchange}')
        self.logger.info(f'💰 Analyzing for leverage range: {leverage_range[0]}x - {leverage_range[1]}x')
        results = {'analysis_timestamp': datetime.now().isoformat(), 'symbol': symbol, 'exchange': exchange, 'leverage_range': leverage_range, 'timeframe_analysis': {}, 'volatility_analysis': {}, 'signal_quality_analysis': {}, 'ensemble_optimization': {}, 'recommendations': {}}
        for timeframe, data in data_dict.items():
            self.logger.info(f'⏰ Analyzing timeframe: {timeframe}')
            timeframe_analysis = await self._analyze_single_timeframe(data, timeframe, leverage_range)
            results['timeframe_analysis'][timeframe] = timeframe_analysis
        self.logger.info('📊 Analyzing volatility patterns across timeframes...')
        volatility_analysis = await self._analyze_volatility_patterns(data_dict)
        results['volatility_analysis'] = volatility_analysis
        self.logger.info('🎯 Analyzing signal quality for high leverage...')
        signal_analysis = await self._analyze_signal_quality(data_dict, leverage_range)
        results['signal_quality_analysis'] = signal_analysis
        self.logger.info('⚖️ Optimizing ensemble configuration...')
        ensemble_config = await self._optimize_ensemble_configuration(results)
        results['ensemble_optimization'] = ensemble_config
        self.logger.info('💡 Generating recommendations...')
        recommendations = await self._generate_recommendations(results)
        results['recommendations'] = recommendations
        await self._save_analysis_results(results, symbol, exchange)
        self.logger.info('✅ Timeframe relevance analysis completed successfully')
        return results

    async def _analyze_single_timeframe(self, data: pd.DataFrame, timeframe: str, leverage_range: tuple[int, int]) -> dict[str, Any]:
        """Analyze a single timeframe for high leverage relevance."""
        analysis = {'timeframe': timeframe, 'data_points': len(data), 'date_range': {'start': data.index.min().isoformat() if len(data) > 0 else None, 'end': data.index.max().isoformat() if len(data) > 0 else None}, 'volatility_metrics': {}, 'leverage_metrics': {}, 'signal_metrics': {}, 'relevance_score': 0.0}
        if len(data) < self.analysis_config['min_data_points']:
            analysis['insufficient_data'] = True
            return analysis
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            analysis['volatility_metrics'] = {'daily_volatility': returns.std() * np.sqrt(252), 'annualized_volatility': returns.std() * np.sqrt(252), 'max_drawdown': self._calculate_max_drawdown(data['close']), 'volatility_of_volatility': returns.rolling(20).std().std(), 'volatility_regime_stability': self._calculate_volatility_stability(returns)}
        analysis['leverage_metrics'] = await self._calculate_leverage_metrics(data, leverage_range)
        analysis['signal_metrics'] = await self._calculate_signal_metrics(data)
        relevance_score = await self._calculate_relevance_score(analysis)
        analysis['relevance_score'] = relevance_score
        return analysis

    async def _analyze_volatility_patterns(self, data_dict: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Analyze volatility patterns across timeframes."""
        volatility_data = {}
        for timeframe, data in data_dict.items():
            if 'close' in data.columns and len(data) > 0:
                returns = data['close'].pct_change().dropna()
                volatility_data[timeframe] = {'daily_vol': returns.std() * np.sqrt(252), 'rolling_vol': returns.rolling(20).std().mean() * np.sqrt(252), 'vol_regime_changes': self._count_volatility_regime_changes(returns)}
        vol_correlations = {}
        timeframes = list(volatility_data.keys())
        for i, tf1 in enumerate(timeframes):
            for _j, tf2 in enumerate(timeframes[i + 1:], i + 1):
                if tf1 in volatility_data and tf2 in volatility_data:
                    corr = self._calculate_volatility_correlation(data_dict[tf1], data_dict[tf2])
                    vol_correlations[f'{tf1}_vs_{tf2}'] = corr
        return {'timeframe_volatilities': volatility_data, 'volatility_correlations': vol_correlations, 'volatility_regime_analysis': self._analyze_volatility_regimes(volatility_data)}

    async def _analyze_signal_quality(self, data_dict: dict[str, pd.DataFrame], leverage_range: tuple[int, int]) -> dict[str, Any]:
        """Analyze signal quality for high leverage trading."""
        signal_quality = {}
        for timeframe, data in data_dict.items():
            if len(data) < self.analysis_config['min_data_points']:
                continue
            quality_metrics = await self._calculate_signal_quality_metrics(data, timeframe, leverage_range)
            signal_quality[timeframe] = quality_metrics
        signal_decay_analysis = self._analyze_signal_decay(signal_quality)
        return {'timeframe_signal_quality': signal_quality, 'signal_decay_analysis': signal_decay_analysis, 'optimal_signal_horizon': self._calculate_optimal_signal_horizon(signal_quality)}

    async def _optimize_ensemble_configuration(self, analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Optimize ensemble configuration based on analysis results."""
        timeframe_analysis = analysis_results['timeframe_analysis']
        signal_analysis = analysis_results['signal_quality_analysis']
        weights = {}
        total_score = 0
        for timeframe, analysis in timeframe_analysis.items():
            if analysis.get('insufficient_data', False):
                continue
            relevance_score = analysis['relevance_score']
            signal_quality = signal_analysis.get('timeframe_signal_quality', {}).get(timeframe, {}).get('overall_quality', 0.5)
            volatility_factor = 1.0 - min(analysis['volatility_metrics'].get('daily_volatility', 0.5), 1.0)
            leverage_factor = analysis['leverage_metrics'].get('leverage_efficiency', 0.5)
            composite_score = relevance_score * 0.4 + signal_quality * 0.3 + volatility_factor * 0.2 + leverage_factor * 0.1
            weights[timeframe] = composite_score
            total_score += composite_score
        if total_score > 0:
            normalized_weights = {}
            for timeframe, weight in weights.items():
                normalized_weight = weight / total_score
                normalized_weight = max(self.analysis_config['ensemble_weight_min'], min(normalized_weight, self.analysis_config['ensemble_weight_max']))
                normalized_weights[timeframe] = normalized_weight
            total_normalized = sum(normalized_weights.values())
            if total_normalized > 0:
                for timeframe in normalized_weights:
                    normalized_weights[timeframe] /= total_normalized
        else:
            normalized_weights = {}
        return {'optimized_weights': normalized_weights, 'weight_justification': self._generate_weight_justification(timeframe_analysis, normalized_weights), 'ensemble_performance_estimate': self._estimate_ensemble_performance(timeframe_analysis, normalized_weights)}

    async def _generate_recommendations(self, analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Generate recommendations based on analysis."""
        timeframe_analysis = analysis_results['timeframe_analysis']
        ensemble_config = analysis_results['ensemble_optimization']
        recommendations = {'timeframe_recommendations': {}, 'ensemble_recommendations': {}, 'risk_management_recommendations': {}, 'general_recommendations': []}
        for timeframe, analysis in timeframe_analysis.items():
            if analysis.get('insufficient_data', False):
                recommendations['timeframe_recommendations'][timeframe] = {'action': 'exclude', 'reason': 'insufficient_data', 'data_points': analysis['data_points']}
                continue
            relevance_score = analysis['relevance_score']
            volatility = analysis['volatility_metrics'].get('daily_volatility', 0.5)
            if relevance_score < 0.3:
                recommendations['timeframe_recommendations'][timeframe] = {'action': 'exclude', 'reason': 'low_relevance', 'relevance_score': relevance_score}
            elif volatility > self.leverage_config['volatility_threshold']:
                recommendations['timeframe_recommendations'][timeframe] = {'action': 'reduce_weight', 'reason': 'high_volatility', 'volatility': volatility}
            else:
                recommendations['timeframe_recommendations'][timeframe] = {'action': 'include', 'reason': 'good_performance', 'relevance_score': relevance_score}
        weights = ensemble_config.get('optimized_weights', {})
        if weights:
            recommendations['ensemble_recommendations'] = {'primary_timeframes': [tf for tf, w in weights.items() if w > 0.2], 'secondary_timeframes': [tf for tf, w in weights.items() if 0.1 <= w <= 0.2], 'excluded_timeframes': [tf for tf in timeframe_analysis if tf not in weights], 'weight_distribution': weights}
        recommendations['risk_management_recommendations'] = {'max_leverage': self._calculate_safe_leverage(analysis_results), 'position_sizing': self._calculate_position_sizing(analysis_results), 'stop_loss_recommendations': self._calculate_stop_loss_recommendations(analysis_results)}
        recommendations['general_recommendations'] = ['Consider reducing ensemble complexity for high leverage trading', 'Focus on timeframes with stable volatility patterns', 'Implement dynamic position sizing based on volatility regime', 'Use regime-specific stop-loss levels']
        return recommendations

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return abs(drawdown.min())

    def _calculate_volatility_stability(self, returns: pd.Series) -> float:
        """Calculate volatility regime stability."""
        rolling_vol = returns.rolling(20).std()
        vol_changes = rolling_vol.pct_change().abs()
        return 1.0 - vol_changes.mean()

    def _count_volatility_regime_changes(self, returns: pd.Series) -> int:
        """Count volatility regime changes."""
        rolling_vol = returns.rolling(20).std()
        vol_median = rolling_vol.median()
        regime_changes = (rolling_vol > vol_median).astype(int).diff().abs().sum()
        return int(regime_changes)

    def _calculate_volatility_correlation(self, data1: pd.DataFrame, data2: pd.DataFrame) -> float:
        """Calculate correlation between volatility of two timeframes."""
        if 'close' not in data1.columns or 'close' not in data2.columns:
            return 0.0
        returns1 = data1['close'].pct_change().dropna()
        returns2 = data2['close'].pct_change().dropna()
        common_index = returns1.index.intersection(returns2.index)
        if len(common_index) < 100:
            return 0.0
        vol1 = returns1.loc[common_index].rolling(20).std()
        vol2 = returns2.loc[common_index].rolling(20).std()
        return vol1.corr(vol2)

    async def _calculate_leverage_metrics(self, data: pd.DataFrame, leverage_range: tuple[int, int]) -> dict[str, Any]:
        """Calculate leverage-specific metrics."""
        metrics = {'leverage_efficiency': 0.0, 'max_safe_leverage': 0, 'leverage_volatility_ratio': 0.0, 'leverage_risk_score': 0.0}
        if 'close' not in data.columns:
            return metrics
        returns = data['close'].pct_change().dropna()
        if len(returns) < 100:
            return metrics
        volatility = returns.std() * np.sqrt(252)
        max_safe_leverage = min(leverage_range[1], int(1.0 / (volatility * 2)))
        metrics['max_safe_leverage'] = max_safe_leverage
        target_leverage = self.leverage_config['target_leverage']
        if max_safe_leverage >= target_leverage:
            metrics['leverage_efficiency'] = target_leverage / max_safe_leverage
        else:
            metrics['leverage_efficiency'] = max_safe_leverage / target_leverage
        metrics['leverage_volatility_ratio'] = target_leverage * volatility
        max_drawdown = self._calculate_max_drawdown(data['close'])
        metrics['leverage_risk_score'] = max_drawdown * target_leverage
        return metrics

    async def _calculate_signal_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate signal quality metrics."""
        metrics = {'signal_stability': 0.0, 'signal_persistence': 0.0, 'signal_accuracy': 0.0, 'overall_quality': 0.0}
        if 'close' in data.columns and len(data) > 100:
            returns = data['close'].pct_change().dropna()
            positive_moves = (returns > 0).rolling(20).mean()
            signal_stability = 1.0 - positive_moves.std()
            metrics['signal_stability'] = max(0.0, signal_stability)
            signal_persistence = self._calculate_signal_persistence(returns)
            metrics['signal_persistence'] = signal_persistence
            metrics['overall_quality'] = metrics['signal_stability'] * 0.4 + metrics['signal_persistence'] * 0.3 + 0.3
        return metrics

    def _calculate_signal_persistence(self, returns: pd.Series) -> float:
        """Calculate signal persistence."""
        direction = np.sign(returns)
        persistence = 0
        max_persistence = 0
        for i in range(1, len(direction)):
            if direction.iloc[i] == direction.iloc[i - 1]:
                persistence += 1
                max_persistence = max(max_persistence, persistence)
            else:
                persistence = 0
        return min(max_persistence / 10.0, 1.0)

    async def _calculate_relevance_score(self, analysis: dict[str, Any]) -> float:
        """Calculate overall relevance score for a timeframe."""
        if analysis.get('insufficient_data', False):
            return 0.0
        volatility = analysis['volatility_metrics'].get('daily_volatility', 0.5)
        leverage_efficiency = analysis['leverage_metrics'].get('leverage_efficiency', 0.0)
        signal_quality = analysis['signal_metrics'].get('overall_quality', 0.0)
        vol_score = max(0.0, 1.0 - volatility / self.leverage_config['volatility_threshold'])
        leverage_score = leverage_efficiency
        signal_score = signal_quality
        relevance_score = vol_score * 0.4 + leverage_score * 0.3 + signal_score * 0.3
        return min(1.0, max(0.0, relevance_score))

    async def _calculate_signal_quality_metrics(self, data: pd.DataFrame, timeframe: str, leverage_range: tuple[int, int]) -> dict[str, Any]:
        """Calculate detailed signal quality metrics."""
        metrics = {'signal_decay_rate': 0.0, 'signal_noise_ratio': 0.0, 'signal_consistency': 0.0, 'overall_quality': 0.0}
        if 'close' not in data.columns:
            return metrics
        returns = data['close'].pct_change().dropna()
        metrics['signal_decay_rate'] = self._calculate_signal_decay_rate(returns)
        metrics['signal_noise_ratio'] = self._calculate_signal_noise_ratio(returns)
        metrics['signal_consistency'] = self._calculate_signal_consistency(returns)
        metrics['overall_quality'] = (1.0 - metrics['signal_decay_rate']) * 0.4 + metrics['signal_noise_ratio'] * 0.3 + metrics['signal_consistency'] * 0.3
        return metrics

    def _calculate_signal_decay_rate(self, returns: pd.Series) -> float:
        """Calculate signal decay rate."""
        autocorr_1 = returns.autocorr(lag=1)
        returns.autocorr(lag=5)
        autocorr_10 = returns.autocorr(lag=10)
        decay_rate = (autocorr_1 - autocorr_10) / 10.0 if autocorr_1 > 0 else 0.0
        return min(1.0, max(0.0, decay_rate))

    def _calculate_signal_noise_ratio(self, returns: pd.Series) -> float:
        """Calculate signal-to-noise ratio."""
        trend = returns.rolling(20).mean()
        noise = returns - trend
        signal_power = trend.var()
        noise_power = noise.var()
        if noise_power > 0:
            snr = signal_power / noise_power
            return min(1.0, snr / 10.0)
        return 0.0

    def _calculate_signal_consistency(self, returns: pd.Series) -> float:
        """Calculate signal consistency."""
        direction = np.sign(returns)
        return direction.rolling(10).apply(lambda x: abs(x.sum()) / len(x), raw=True).mean()

    def _analyze_signal_decay(self, signal_quality: dict[str, Any]) -> dict[str, Any]:
        """Analyze signal decay across timeframes."""
        decay_rates = {}
        for timeframe, metrics in signal_quality.items():
            decay_rates[timeframe] = metrics.get('signal_decay_rate', 0.0)
        return {'timeframe_decay_rates': decay_rates, 'optimal_holding_period': self._calculate_optimal_holding_period(decay_rates)}

    def _calculate_optimal_holding_period(self, decay_rates: dict[str, float]) -> dict[str, int]:
        """Calculate optimal holding period for each timeframe."""
        holding_periods = {}
        for timeframe, decay_rate in decay_rates.items():
            if decay_rate > 0:
                optimal_period = int(1.0 / decay_rate)
                holding_periods[timeframe] = max(1, min(optimal_period, 100))
            else:
                holding_periods[timeframe] = 50
        return holding_periods

    def _calculate_optimal_signal_horizon(self, signal_quality: dict[str, Any]) -> dict[str, int]:
        """Calculate optimal signal horizon for each timeframe."""
        horizons = {}
        for timeframe, metrics in signal_quality.items():
            decay_rate = metrics.get('signal_decay_rate', 0.5)
            horizon = int(5.0 / (1.0 + decay_rate))
            horizons[timeframe] = max(1, min(horizon, 20))
        return horizons

    def _generate_weight_justification(self, timeframe_analysis: dict[str, Any], weights: dict[str, float]) -> dict[str, str]:
        """Generate justification for each weight."""
        justifications = {}
        for timeframe, weight in weights.items():
            if timeframe in timeframe_analysis:
                analysis = timeframe_analysis[timeframe]
                relevance_score = analysis.get('relevance_score', 0.0)
                volatility = analysis['volatility_metrics'].get('daily_volatility', 0.5)
                if weight > 0.3:
                    justifications[timeframe] = f'High relevance ({relevance_score:.2f}) and low volatility ({volatility:.3f})'
                elif weight > 0.15:
                    justifications[timeframe] = f'Moderate relevance ({relevance_score:.2f}) with acceptable volatility ({volatility:.3f})'
                else:
                    justifications[timeframe] = f'Lower weight due to relevance score ({relevance_score:.2f}) and volatility ({volatility:.3f})'
        return justifications

    def _estimate_ensemble_performance(self, timeframe_analysis: dict[str, Any], weights: dict[str, float]) -> dict[str, float]:
        """Estimate ensemble performance based on weighted combination."""
        total_volatility = 0.0
        total_relevance = 0.0
        total_weight = 0.0
        for timeframe, weight in weights.items():
            if timeframe in timeframe_analysis:
                analysis = timeframe_analysis[timeframe]
                volatility = analysis['volatility_metrics'].get('daily_volatility', 0.5)
                relevance = analysis.get('relevance_score', 0.0)
                total_volatility += volatility * weight
                total_relevance += relevance * weight
                total_weight += weight
        if total_weight > 0:
            avg_volatility = total_volatility / total_weight
            avg_relevance = total_relevance / total_weight
        else:
            avg_volatility = 0.5
            avg_relevance = 0.0
        return {'estimated_volatility': avg_volatility, 'estimated_relevance': avg_relevance, 'risk_adjusted_score': avg_relevance / (1.0 + avg_volatility)}

    def _calculate_safe_leverage(self, analysis_results: dict[str, Any]) -> int:
        """Calculate safe maximum leverage based on analysis."""
        timeframe_analysis = analysis_results['timeframe_analysis']
        max_safe_leverages = []
        for analysis in timeframe_analysis.values():
            if not analysis.get('insufficient_data', False):
                leverage = analysis['leverage_metrics'].get('max_safe_leverage', 0)
                max_safe_leverages.append(leverage)
        if max_safe_leverages:
            return min(max_safe_leverages)
        return 10

    def _calculate_position_sizing(self, analysis_results: dict[str, Any]) -> dict[str, float]:
        """Calculate position sizing recommendations."""
        volatility_analysis = analysis_results['volatility_analysis']
        avg_volatility = 0.0
        count = 0
        for metrics in volatility_analysis.get('timeframe_volatilities', {}).values():
            avg_volatility += metrics.get('daily_vol', 0.5)
            count += 1
        if count > 0:
            avg_volatility /= count
        base_position_size = 1.0 / (1.0 + avg_volatility * 10)
        return {'base_position_size': base_position_size, 'volatility_adjusted_size': base_position_size * (1.0 - avg_volatility), 'max_position_size': min(1.0, base_position_size * 1.5)}

    def _calculate_stop_loss_recommendations(self, analysis_results: dict[str, Any]) -> dict[str, float]:
        """Calculate stop-loss recommendations."""
        timeframe_analysis = analysis_results['timeframe_analysis']
        max_drawdowns = []
        for analysis in timeframe_analysis.values():
            if not analysis.get('insufficient_data', False):
                drawdown = analysis['volatility_metrics'].get('max_drawdown', 0.1)
                max_drawdowns.append(drawdown)
        if max_drawdowns:
            avg_drawdown = np.mean(max_drawdowns)
            max_drawdown = np.max(max_drawdowns)
        else:
            avg_drawdown = 0.1
            max_drawdown = 0.2
        return {'conservative_stop_loss': avg_drawdown * 0.5, 'moderate_stop_loss': avg_drawdown, 'aggressive_stop_loss': max_drawdown * 0.8, 'recommended_stop_loss': avg_drawdown * 0.7}

    async def _save_analysis_results(self, results: dict[str, Any], symbol: str, exchange: str) -> None:
        """Save analysis results to file."""
        output_dir = Path('data/timeframe_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'{exchange}_{symbol}_timeframe_analysis.json'
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f'💾 Saved timeframe analysis results to {filepath}')

    def get_optimized_ensemble_config(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Load optimized ensemble configuration."""
        filepath = Path(f'data/timeframe_analysis/{exchange}_{symbol}_timeframe_analysis.json')
        if not filepath.exists():
            self.logger.warning(f'⚠️ No timeframe analysis found for {symbol} on {exchange}')
            return {}
        try:
            with open(filepath) as f:
                results = json.load(f)
            return results.get('ensemble_optimization', {})
        except Exception as e:
            self.logger.exception(f'❌ Error loading timeframe analysis: {e}')
            return {}