"""Regime tactics component for tactician specialist training."""
import asyncio
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger
from copy import copy

class RegimeTactics:
    """Handles regime-specific tactics determination and configuration."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the regime tactics handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('tactics', {})
        self.logger = system_logger.getChild('regime_tactics')
        self.tactic_definitions = {'breakout': {'description': 'Identifies and trades price breakouts from key levels', 'indicators': ['volatility', 'volume', 'momentum'], 'regime_affinity': {'trending': 0.9, 'volatile': 0.7, 'range_bound': 0.3, 'calm': 0.5}}, 'reversal': {'description': 'Identifies and trades trend reversals', 'indicators': ['rsi', 'divergence', 'support_resistance'], 'regime_affinity': {'trending': 0.4, 'volatile': 0.8, 'range_bound': 0.9, 'calm': 0.6}}, 'continuation': {'description': 'Trades in the direction of the prevailing trend', 'indicators': ['trend_strength', 'momentum', 'ma_alignment'], 'regime_affinity': {'trending': 0.95, 'volatile': 0.5, 'range_bound': 0.2, 'calm': 0.7}}, 'range_bound': {'description': 'Trades oscillations within a defined range', 'indicators': ['bollinger_bands', 'support_resistance', 'mean_reversion'], 'regime_affinity': {'trending': 0.2, 'volatile': 0.6, 'range_bound': 0.95, 'calm': 0.8}}}
        self.default_tactic_config = {'enabled': True, 'min_confidence': 0.75, 'max_position_size': 1.0, 'stop_loss_multiplier': 1.0, 'take_profit_multiplier': 2.0}

    @handles_errors(exceptions=(Exception,), default_return={}, context='regime tactics determination')
    async def determine_tactics(self, regime_id: str, regime_info: Dict[str, Any], regime_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Determine appropriate tactics for a regime.
        
        Args:
            regime_id: Regime identifier
            regime_info: Regime metadata
            regime_data: Market data for the regime
            
        Returns:
            Dictionary of tactic configurations
        """
        self.logger.info(f'Determining tactics for regime {regime_id}')
        regime_characteristics = await self._analyze_regime_characteristics(regime_data, regime_info)
        regime_type = self._classify_regime_type(regime_characteristics)
        self.logger.info(f'Regime {regime_id} classified as: {regime_type}')
        tactics = {}
        for tactic_name, tactic_def in self.tactic_definitions.items():
            affinity = tactic_def['regime_affinity'].get(regime_type, 0.5)
            tactic_config = self.config.get(tactic_name, self.default_tactic_config.copy())
            adjusted_config = await self._adjust_tactic_for_regime(tactic_name, tactic_config, regime_characteristics, affinity)
            adjusted_config['enabled'] = affinity >= 0.5 and tactic_config.get('enabled', True)
            adjusted_config['affinity_score'] = affinity
            tactics[tactic_name] = adjusted_config
        return tactics

    async def _analyze_regime_characteristics(self, data: pd.DataFrame, regime_info: Dict[str, Any]) -> Dict[str, float]:
        """Analyze characteristics of a regime.
        
        Args:
            data: Market data for the regime
            regime_info: Regime metadata
            
        Returns:
            Dictionary of regime characteristics
        """
        characteristics = {'volatility': 0.0, 'trend_strength': 0.0, 'mean_reversion': 0.0, 'volume_profile': 0.0, 'price_range': 0.0, 'momentum': 0.0}
        if 'close' not in data.columns or len(data) < 20:
            return characteristics
        returns = data['close'].pct_change().dropna()
        characteristics['volatility'] = returns.std()
        if len(data) > 1:
            x = np.arange(len(data))
            y = data['close'].values
            slope = np.polyfit(x, y, 1)[0]
            characteristics['trend_strength'] = abs(slope) / data['close'].mean()
        if len(returns) > 50:
            lags = range(2, min(20, len(returns) // 2))
            tau = []
            for lag in lags:
                subseries = [returns.iloc[i:i + lag] for i in range(0, len(returns) - lag, lag)]
                rs_values = []
                for series in subseries:
                    if len(series) > 1:
                        mean = series.mean()
                        std = series.std()
                        if std > 0:
                            cumsum = (series - mean).cumsum()
                            R = cumsum.max() - cumsum.min()
                            S = std
                            rs_values.append(R / S)
                if rs_values:
                    tau.append(np.mean(rs_values))
            if tau:
                log_lags = np.log(list(lags))
                log_tau = np.log(tau)
                hurst = np.polyfit(log_lags, log_tau, 1)[0]
                characteristics['mean_reversion'] = max(0, 0.5 - hurst) * 2
        if 'volume' in data.columns:
            avg_volume = data['volume'].mean()
            volume_volatility = data['volume'].std() / avg_volume if avg_volume > 0 else 0
            characteristics['volume_profile'] = volume_volatility
        if 'high' in data.columns and 'low' in data.columns:
            avg_range = (data['high'] - data['low']).mean()
            avg_price = data['close'].mean()
            characteristics['price_range'] = avg_range / avg_price if avg_price > 0 else 0
        if len(data) > 20:
            roc = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
            characteristics['momentum'] = abs(roc)
        if 'volatility_regime' in regime_info:
            vol_map = {'low': 0.2, 'normal': 0.5, 'high': 0.8}
            characteristics['volatility'] = vol_map.get(regime_info['volatility_regime'], 0.5)
        return characteristics

    def _classify_regime_type(self, characteristics: Dict[str, float]) -> str:
        """Classify regime type based on characteristics.
        
        Args:
            characteristics: Regime characteristics
            
        Returns:
            Regime type string
        """
        volatility = characteristics.get('volatility', 0)
        trend_strength = characteristics.get('trend_strength', 0)
        mean_reversion = characteristics.get('mean_reversion', 0)
        if trend_strength > 0.01 and mean_reversion < 0.3:
            return 'trending'
        elif mean_reversion > 0.6 and trend_strength < 0.005:
            return 'range_bound'
        elif volatility > 0.02:
            return 'volatile'
        else:
            return 'calm'

    async def _adjust_tactic_for_regime(self, tactic_name: str, base_config: Dict[str, Any], regime_characteristics: Dict[str, float], affinity: float) -> Dict[str, Any]:
        """Adjust tactic configuration for regime characteristics.
        
        Args:
            tactic_name: Name of the tactic
            base_config: Base tactic configuration
            regime_characteristics: Regime characteristics
            affinity: Regime affinity score
            
        Returns:
            Adjusted tactic configuration
        """
        config = base_config.copy()
        if affinity < 0.7:
            config['min_confidence'] = min(0.95, base_config.get('min_confidence', 0.75) + 0.1)
        volatility = regime_characteristics.get('volatility', 0.01)
        if volatility > 0.02:
            config['stop_loss_multiplier'] = base_config.get('stop_loss_multiplier', 1.0) * 1.5
            config['max_position_size'] = base_config.get('max_position_size', 1.0) * 0.7
        elif volatility < 0.005:
            config['stop_loss_multiplier'] = base_config.get('stop_loss_multiplier', 1.0) * 0.7
            config['max_position_size'] = min(1.0, base_config.get('max_position_size', 1.0) * 1.3)
        if tactic_name == 'breakout':
            momentum = regime_characteristics.get('momentum', 0)
            if momentum > 0.05:
                config['take_profit_multiplier'] *= 1.2
        elif tactic_name == 'reversal':
            mean_rev = regime_characteristics.get('mean_reversion', 0)
            if mean_rev > 0.7:
                config['min_confidence'] *= 0.95
        elif tactic_name == 'continuation':
            trend = regime_characteristics.get('trend_strength', 0)
            if trend > 0.02:
                config['take_profit_multiplier'] *= 1.3
                config['max_position_size'] *= 1.1
        elif tactic_name == 'range_bound':
            price_range = regime_characteristics.get('price_range', 0)
            if price_range > 0.02:
                config['take_profit_multiplier'] *= 0.8
        return config

    @handles_errors(exceptions=(Exception,), default_return={}, context='optimal tactics selection')
    async def select_optimal_tactics(self, tactics: Dict[str, Dict[str, Any]], performance_history: Optional[Dict[str, Any]]=None) -> List[str]:
        """Select optimal tactics based on configuration and history.
        
        Args:
            tactics: Tactic configurations
            performance_history: Historical performance data
            
        Returns:
            List of selected tactic names
        """
        selected_tactics = []
        sorted_tactics = sorted([(name, config) for name, config in tactics.items()], key=lambda x: x[1].get('affinity_score', 0), reverse=True)
        for tactic_name, config in sorted_tactics:
            if not config.get('enabled', False):
                continue
            if performance_history and tactic_name in performance_history:
                hist_perf = performance_history[tactic_name]
                if hist_perf.get('win_rate', 0.5) < 0.4:
                    self.logger.info(f'Skipping {tactic_name} due to poor historical performance')
                    continue
            selected_tactics.append(tactic_name)
            if len(selected_tactics) >= 3:
                break
        return selected_tactics