import pandas as pd
import numpy as np

'Context-Aware S/R Calculator Module.\n\nThis module adjusts S/R parameters and calculations based on market context,\nincluding time of day, volatility regime, news events, and correlations.\n'
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os
from .core.decorators import handles_errors, traced
from .utils.logger import system_logger
from .tactician.sr_modules.sr_probability_calculator import SRProbabilityCalculator
from .tactician.sr_strength_optimizer import SRLevelIdentifier

@dataclass
class MarketContext:
    """Current market context information."""
    timestamp: datetime
    is_market_hours: bool
    session_type: str
    volatility_regime: str
    current_volatility: float
    volume_profile: str
    trend_strength: float
    correlation_state: Dict[str, float]
    news_impact: str
    economic_calendar: List[Dict[str, Any]]

@dataclass
class ContextAdjustedParameters:
    """Context-adjusted S/R parameters."""
    strength_multiplier: float = 1.0
    proximity_threshold_multiplier: float = 1.0
    volume_weight_multiplier: float = 1.0
    volatility_weight_multiplier: float = 1.0
    momentum_weight_multiplier: float = 1.0
    min_touches_adjustment: int = 0
    bounce_requirement_multiplier: float = 1.0
    age_decay_acceleration: float = 1.0
    adjustment_reason: str = ''
    confidence_in_adjustment: float = 1.0

class MarketContextAnalyzer:
    """Analyzes current market context."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('MarketContextAnalyzer')
        self.market_hours = {'asian': (time(0, 0), time(9, 0)), 'european': (time(7, 0), time(16, 0)), 'us': (time(13, 30), time(21, 0))}
        self.volatility_thresholds = {'low': 0.005, 'normal': 0.015, 'high': 0.025}

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='analyze market context')
    async def analyze_context(self, market_data: pd.DataFrame, correlation_data: Optional[Dict[str, pd.DataFrame]]=None, news_data: Optional[List[Dict[str, Any]]]=None) -> MarketContext:
        """Analyze current market context."""
        current_time = datetime.now()
        session_type = self._determine_session(current_time)
        is_market_hours = session_type != 'closed'
        volatility_regime, current_volatility = self._analyze_volatility(market_data)
        volume_profile = self._analyze_volume_profile(market_data)
        trend_strength = self._analyze_trend_strength(market_data)
        correlation_state = {}
        if correlation_data:
            correlation_state = self._analyze_correlations(market_data, correlation_data)
        news_impact, economic_calendar = self._analyze_news_impact(news_data)
        return MarketContext(timestamp=current_time, is_market_hours=is_market_hours, session_type=session_type, volatility_regime=volatility_regime, current_volatility=current_volatility, volume_profile=volume_profile, trend_strength=trend_strength, correlation_state=correlation_state, news_impact=news_impact, economic_calendar=economic_calendar)

    def _determine_session(self, current_time: datetime) -> str:
        """Determine current trading session."""
        current_hour_utc = current_time.hour
        if 7 <= current_hour_utc <= 9:
            return 'asian_european_overlap'
        elif 13 <= current_hour_utc <= 16:
            return 'european_us_overlap'
        for session, (start, end) in self.market_hours.items():
            if start <= current_time.time() <= end:
                return session
        return 'closed'

    def _analyze_volatility(self, market_data: pd.DataFrame) -> Tuple[str, float]:
        """Analyze current volatility regime."""
        returns = market_data['close'].pct_change()
        recent_vol = returns.iloc[-20:].std()
        long_vol = returns.iloc[-100:].std()
        atr = self._calculate_atr(market_data, 14)
        atr_vol = atr.iloc[-1] / market_data['close'].iloc[-1]
        current_volatility = recent_vol * 0.5 + long_vol * 0.3 + atr_vol * 0.2
        if current_volatility < self.volatility_thresholds['low']:
            regime = 'low'
        elif current_volatility < self.volatility_thresholds['normal']:
            regime = 'normal'
        elif current_volatility < self.volatility_thresholds['high']:
            regime = 'high'
        else:
            regime = 'extreme'
        return (regime, current_volatility)

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    def _analyze_volume_profile(self, market_data: pd.DataFrame) -> str:
        """Analyze current volume profile."""
        recent_volume = market_data['volume'].iloc[-20:].mean()
        avg_volume = market_data['volume'].iloc[-100:].mean()
        ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        if ratio < 0.5:
            return 'thin'
        elif ratio < 1.5:
            return 'normal'
        else:
            return 'heavy'

    def _analyze_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Analyze current trend strength."""
        close_prices = market_data['close']
        sma_short = close_prices.rolling(10).mean()
        sma_long = close_prices.rolling(50).mean()
        if len(close_prices) >= 50:
            trend_direction = 1 if sma_short.iloc[-1] > sma_long.iloc[-1] else -1
            separation = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            trend_strength = min(separation * 100, 1.0) * trend_direction
        else:
            trend_strength = 0.0
        return trend_strength

    def _analyze_correlations(self, market_data: pd.DataFrame, correlation_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze correlations with other assets."""
        correlations = {}
        returns = market_data['close'].pct_change().iloc[-100:]
        for asset, data in correlation_data.items():
            asset_returns = data['close'].pct_change().iloc[-100:]
            if len(returns) == len(asset_returns):
                corr = returns.corr(asset_returns)
                correlations[asset] = corr
        return correlations

    def _analyze_news_impact(self, news_data: Optional[List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Analyze news and economic calendar impact."""
        if not news_data:
            return ('none', [])
        current_time = datetime.now()
        economic_calendar = []
        impact_scores = []
        for event in news_data:
            event_time = event.get('datetime', current_time)
            time_diff = (event_time - current_time).total_seconds() / 3600
            if -1 <= time_diff <= 4:
                economic_calendar.append(event)
                importance = event.get('importance', 'medium')
                importance_score = {'low': 0.3, 'medium': 0.6, 'high': 1.0}.get(importance, 0.5)
                proximity_score = 1 - min(abs(time_diff) / 4, 1)
                impact_scores.append(importance_score * proximity_score)
        if not impact_scores:
            impact = 'none'
        else:
            avg_impact = np.mean(impact_scores)
            if avg_impact < 0.3:
                impact = 'low'
            elif avg_impact < 0.6:
                impact = 'medium'
            else:
                impact = 'high'
        return (impact, economic_calendar)

class ContextAwareSRCalculator:
    """Adjusts S/R calculations based on market context."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('ContextAwareSRCalculator')
        self.context_analyzer = MarketContextAnalyzer(config)
        self.base_parameters = self._load_base_parameters()
        self.adjustment_rules = self._initialize_adjustment_rules()

    def _load_base_parameters(self) -> Dict[str, Any]:
        """Load base S/R parameters."""
        try:
            from .utils.sr_parameter_loader import SRParameterLoader

            return SRParameterLoader.load_optimized_parameters(self.config)
        except Exception as e:
            self.logger.error(f'Error loading base parameters: {e}')
            return {}

    def _initialize_adjustment_rules(self) -> Dict[str, Any]:
        """Initialize context-based adjustment rules."""
        return {'volatility_adjustments': {'low': {'proximity_threshold_multiplier': 0.8, 'min_touches_adjustment': 1, 'bounce_requirement_multiplier': 0.8}, 'high': {'proximity_threshold_multiplier': 1.5, 'volatility_weight_multiplier': 1.5, 'min_touches_adjustment': -1}, 'extreme': {'proximity_threshold_multiplier': 2.0, 'volatility_weight_multiplier': 2.0, 'age_decay_acceleration': 1.5}}, 'session_adjustments': {'asian': {'volume_weight_multiplier': 0.8, 'momentum_weight_multiplier': 0.9}, 'us': {'volume_weight_multiplier': 1.2, 'momentum_weight_multiplier': 1.1}, 'overlap': {'strength_multiplier': 1.2, 'volume_weight_multiplier': 1.3}}, 'news_adjustments': {'high': {'volatility_weight_multiplier': 1.5, 'age_decay_acceleration': 1.3, 'proximity_threshold_multiplier': 1.3}, 'medium': {'volatility_weight_multiplier': 1.2, 'volume_weight_multiplier': 1.1}}, 'trend_adjustments': {'strong_trend': {'momentum_weight_multiplier': 1.3, 'bounce_requirement_multiplier': 1.2}, 'no_trend': {'proximity_threshold_multiplier': 0.9, 'min_touches_adjustment': 1}}}

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='calculate context-aware SR')
    @traced(span_name='ContextSR.calculate')
    async def calculate_context_aware_sr(self, market_data: pd.DataFrame, correlation_data: Optional[Dict[str, pd.DataFrame]]=None, news_data: Optional[List[Dict[str, Any]]]=None) -> Dict[str, Any]:
        """
        Calculate S/R levels with context-aware adjustments.
        
        Args:
            market_data: Historical market data
            correlation_data: Data for correlated assets
            news_data: News and economic calendar events
            
        Returns:
            Dictionary containing adjusted S/R levels and context
        """
        try:
            context = await self.context_analyzer.analyze_context(market_data, correlation_data, news_data)
            adjustments = self._get_context_adjustments(context)
            adjusted_params = self._apply_adjustments(self.base_parameters, adjustments)
            sr_levels, probabilities = self._calculate_sr_with_adjusted_params(market_data, adjusted_params)
            return {'sr_levels': sr_levels, 'probabilities': probabilities, 'context': asdict(context), 'adjustments': asdict(adjustments), 'adjusted_parameters': adjusted_params}
        except Exception as e:
            self.logger.error(f'Error in context-aware S/R calculation: {e}')
            return {}

    def _calculate_sr_with_adjusted_params(self, market_data: pd.DataFrame, adjusted_params: Dict[str, Any]) -> Tuple[List[Any], Dict[str, float]]:
        """Calculate S/R levels and probabilities with adjusted parameters."""
        original_params = self.config.get('sr_probability_calculation', {})
        self.config['sr_probability_calculation'] = adjusted_params
        try:
            calculator = SRProbabilityCalculator(self.config)
            identifier = SRLevelIdentifier(self.config)
            sr_levels = identifier.identify_strong_sr_levels(market_data)
            current_price = market_data['close'].iloc[-1]
            sr_context = {'support': [l for l in sr_levels if l.type == 'support'], 'resistance': [l for l in sr_levels if l.type == 'resistance']}
            probabilities = calculator.calculate_probabilities(market_data, current_price, sr_context)
            return (sr_levels, probabilities)
        finally:
            self.config['sr_probability_calculation'] = original_params

    def _get_context_adjustments(self, context: MarketContext) -> ContextAdjustedParameters:
        """Get parameter adjustments based on context."""
        adjustments = ContextAdjustedParameters()
        reasons = []
        vol_adjustments = self.adjustment_rules['volatility_adjustments'].get(context.volatility_regime, {})
        self._apply_adjustment_dict(adjustments, vol_adjustments)
        if vol_adjustments:
            reasons.append(f'volatility_{context.volatility_regime}')
        session_adjustments = self.adjustment_rules['session_adjustments'].get(context.session_type, {})
        if 'overlap' in context.session_type:
            session_adjustments = self.adjustment_rules['session_adjustments']['overlap']
        self._apply_adjustment_dict(adjustments, session_adjustments)
        if session_adjustments:
            reasons.append(f'session_{context.session_type}')
        news_adjustments = self.adjustment_rules['news_adjustments'].get(context.news_impact, {})
        self._apply_adjustment_dict(adjustments, news_adjustments)
        if news_adjustments:
            reasons.append(f'news_{context.news_impact}')
        if abs(context.trend_strength) > 0.7:
            trend_adjustments = self.adjustment_rules['trend_adjustments']['strong_trend']
            self._apply_adjustment_dict(adjustments, trend_adjustments)
            reasons.append('strong_trend')
        elif abs(context.trend_strength) < 0.3:
            trend_adjustments = self.adjustment_rules['trend_adjustments']['no_trend']
            self._apply_adjustment_dict(adjustments, trend_adjustments)
            reasons.append('no_trend')
        if context.volume_profile == 'thin':
            adjustments.volume_weight_multiplier *= 0.7
            adjustments.min_touches_adjustment += 1
            reasons.append('thin_volume')
        elif context.volume_profile == 'heavy':
            adjustments.volume_weight_multiplier *= 1.2
            reasons.append('heavy_volume')
        if context.correlation_state:
            high_correlations = [asset for asset, corr in context.correlation_state.items() if abs(corr) > 0.8]
            if high_correlations:
                adjustments.momentum_weight_multiplier *= 1.1
                reasons.append(f'high_correlation_{len(high_correlations)}')
        adjustments.adjustment_reason = ', '.join(reasons)
        adjustments.confidence_in_adjustment = self._calculate_adjustment_confidence(context, len(reasons))
        return adjustments

    def _apply_adjustment_dict(self, adjustments: ContextAdjustedParameters, adjustment_dict: Dict[str, float]) -> None:
        """Apply adjustment dictionary to parameters."""
        for param, value in adjustment_dict.items():
            if hasattr(adjustments, param):
                current = getattr(adjustments, param)
                if param.endswith('_multiplier'):
                    setattr(adjustments, param, current * value)
                elif param.endswith('_adjustment'):
                    setattr(adjustments, param, current + value)
                else:
                    setattr(adjustments, param, value)

    def _calculate_adjustment_confidence(self, context: MarketContext, num_adjustments: int) -> float:
        """Calculate confidence in the adjustments."""
        confidence = 0.8
        confidence -= num_adjustments * 0.05
        if context.volatility_regime == 'extreme':
            confidence -= 0.1
        if context.news_impact == 'high':
            confidence -= 0.1
        if context.is_market_hours:
            confidence += 0.1
        return max(0.3, min(confidence, 1.0))

    def _apply_adjustments(self, base_params: Dict[str, float], adjustments: ContextAdjustedParameters) -> Dict[str, float]:
        """Apply adjustments to base parameters."""
        adjusted = base_params.copy()
        weight_adjustments = {'volume_weight': adjustments.volume_weight_multiplier, 'volatility_weight': adjustments.volatility_weight_multiplier, 'momentum_weight': adjustments.momentum_weight_multiplier}
        for param, multiplier in weight_adjustments.items():
            if param in adjusted:
                adjusted[param] *= multiplier
        if 'proximity_threshold' in adjusted:
            adjusted['proximity_threshold'] *= adjustments.proximity_threshold_multiplier
        if 'min_touches' in adjusted:
            adjusted['min_touches'] = max(1, adjusted['min_touches'] + adjustments.min_touches_adjustment)
        if 'age_decay_factor' in adjusted:
            adjusted['age_decay_factor'] = adjusted['age_decay_factor'] ** adjustments.age_decay_acceleration
        if 'min_bounce_ratio' in adjusted:
            adjusted['min_bounce_ratio'] *= adjustments.bounce_requirement_multiplier
        weight_keys = ['price_action_weight', 'momentum_weight', 'trend_strength_weight', 'volume_weight', 'volatility_weight']
        total_weight = sum((adjusted.get(k, 0) for k in weight_keys))
        if total_weight > 0:
            for key in weight_keys:
                if key in adjusted:
                    adjusted[key] /= total_weight
        return adjusted

    def save_context_history(self, context: MarketContext, adjustments: ContextAdjustedParameters, results: Dict[str, Any]) -> None:
        """Save context and adjustments for analysis."""
        try:
            history_file = os.path.join(self.config.get('model_save_path', 'models'), 'context_sr_history.json')
            history = []
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            entry = {'timestamp': context.timestamp.isoformat(), 'context': asdict(context), 'adjustments': asdict(adjustments), 'num_sr_levels': len(results.get('sr_levels', [])), 'probabilities': results.get('probabilities', {})}
            history.append(entry)
            history = history[-1000:]
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            self.logger.error(f'Error saving context history: {e}')