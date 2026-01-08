"""
GMM-Based Trading Logic Module

This module implements sophisticated trading logic based on GMM regime analysis,
including trend runway estimation, dynamic trailing stops, and high-conviction entry detection.

Key Features:
- GMM-State trend runway analysis
- Dynamic trailing stop adjustment based on overextended clusters
- GMM-Shock high-conviction entry point detection
- Regime-aware position sizing and risk management
- Multi-timeframe analysis integration
- Performance monitoring and analytics

Usage:
    trading_engine = GMMTradingEngine(config)
    signals = trading_engine.generate_signals(gmm_features, market_data)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from datetime import datetime, timedelta

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success


class RegimeState(Enum):
    """Enumeration of possible regime states."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    TRANSITIONING = "transitioning"
    OVEREXTENDED = "overextended"
    UNCERTAIN = "uncertain"


class SignalType(Enum):
    """Enumeration of trading signal types."""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    ADJUST_STOP = "adjust_stop"
    REDUCE_POSITION = "reduce_position"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """Container for trading signals."""
    timestamp: pd.Timestamp
    signal_type: SignalType
    confidence: float
    regime_state: RegimeState
    runway_remaining: float
    recommended_stop: Optional[float]
    recommended_target: Optional[float]
    position_size_multiplier: float
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class RunwayAnalysis:
    """Container for trend runway analysis results."""
    current_regime: RegimeState
    runway_estimate: float  # Estimated remaining trend duration (bars)
    runway_confidence: float
    momentum_strength: float
    regime_maturity: float  # 0 = beginning, 1 = end
    overextension_risk: float
    reversal_probability: float


@dataclass
class ShockEvent:
    """Container for GMM-Shock events."""
    timestamp: pd.Timestamp
    shock_type: str
    magnitude: float
    confidence: float
    expected_direction: str  # 'long' or 'short'
    duration_estimate: int  # Expected duration in bars
    follow_through_probability: float


class GMMTradingEngine:
    """
    Advanced trading engine based on GMM regime analysis.
    
    This engine implements sophisticated trading logic using GMM-based regime detection,
    trend runway analysis, and shock event detection for optimal entry/exit timing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize GMM Trading Engine."""
        self.config = config or {}
        
        # Runway analysis configuration
        self.runway_config = self.config.get('runway_analysis', {
            'momentum_window': 20,
            'regime_maturity_threshold': 0.7,
            'overextension_threshold': 0.8,
            'minimum_runway': 5,
            'maximum_runway': 100
        })
        
        # Shock detection configuration
        self.shock_config = self.config.get('shock_detection', {
            'confidence_threshold': 0.6,
            'magnitude_threshold': 0.3,
            'follow_through_threshold': 0.7,
            'min_shock_duration': 3,
            'max_shock_duration': 20
        })
        
        # Position sizing configuration
        self.position_config = self.config.get('position_sizing', {
            'base_size': 1.0,
            'max_size': 3.0,
            'runway_scaling': True,
            'confidence_scaling': True,
            'volatility_adjustment': True
        })
        
        # Stop loss configuration
        self.stop_config = self.config.get('stop_management', {
            'base_stop_atr_multiple': 2.0,
            'overextended_tighten_factor': 0.5,
            'shock_widening_factor': 1.5,
            'trailing_stop_activation': 0.5,  # ATR
            'max_stop_move': 0.1  # Maximum stop adjustment per bar
        })
        
        # Internal state
        self.current_regime = RegimeState.UNCERTAIN
        self.current_runway = RunwayAnalysis(RegimeState.UNCERTAIN, 0, 0, 0, 0, 0, 0)
        self.active_shocks: List[ShockEvent] = []
        self.position_state = {
            'in_position': False,
            'position_type': None,
            'entry_price': None,
            'current_stop': None,
            'position_size': 0.0
        }
        
        # Performance tracking
        self.performance_metrics = {
            'signals_generated': 0,
            'successful_entries': 0,
            'false_signals': 0,
            'average_runway_accuracy': 0.0,
            'shock_detection_accuracy': 0.0
        }
    
    def analyze_regime_runway(self, 
                            gmm_features: pd.DataFrame,
                            market_data: pd.DataFrame) -> RunwayAnalysis:
        """
        Analyze trend runway based on GMM regime features.
        
        Args:
            gmm_features: DataFrame with GMM features
            market_data: DataFrame with OHLCV data
            
        Returns:
            RunwayAnalysis with trend runway estimates
        """
        tprint_info("🛫 Analyzing GMM-State trend runway...")
        
        try:
            # Extract key GMM features
            latest_features = gmm_features.iloc[-1]
            
            # 1. Determine current regime state
            regime_state = self._determine_regime_state(latest_features)
            
            # 2. Calculate momentum strength
            momentum_strength = self._calculate_momentum_strength(gmm_features)
            
            # 3. Estimate regime maturity
            regime_maturity = self._estimate_regime_maturity(gmm_features, regime_state)
            
            # 4. Calculate overextension risk
            overextension_risk = self._calculate_overextension_risk(latest_features, regime_maturity)
            
            # 5. Estimate reversal probability
            reversal_probability = self._estimate_reversal_probability(
                regime_maturity, overextension_risk, momentum_strength
            )
            
            # 6. Estimate runway remaining
            runway_estimate = self._estimate_runway_remaining(
                regime_maturity, momentum_strength, reversal_probability
            )
            
            # 7. Calculate runway confidence
            runway_confidence = self._calculate_runway_confidence(
                momentum_strength, regime_maturity, len(gmm_features)
            )
            
            # Create runway analysis
            runway_analysis = RunwayAnalysis(
                current_regime=regime_state,
                runway_estimate=runway_estimate,
                runway_confidence=runway_confidence,
                momentum_strength=momentum_strength,
                regime_maturity=regime_maturity,
                overextension_risk=overextension_risk,
                reversal_probability=reversal_probability
            )
            
            self.current_runway = runway_analysis
            self.current_regime = regime_state
            
            tprint_success(f"✅ Runway analysis: {regime_state.value}, {runway_estimate:.1f} bars remaining")
            
            return runway_analysis
            
        except Exception as e:
            tprint_error(f"❌ Runway analysis failed: {e}")
            return RunwayAnalysis(RegimeState.UNCERTAIN, 0, 0, 0, 0, 0, 0)
    
    def detect_shock_events(self, 
                           gmm_features: pd.DataFrame,
                           market_data: pd.DataFrame) -> List[ShockEvent]:
        """
        Detect GMM-Shock events for high-conviction entries.
        
        Args:
            gmm_features: DataFrame with GMM features
            market_data: DataFrame with OHLCV data
            
        Returns:
            List of detected shock events
        """
        tprint_info("⚡ Detecting GMM-Shock events...")
        
        shock_events = []
        
        try:
            # Get latest features
            latest_features = gmm_features.iloc[-1]
            previous_features = gmm_features.iloc[-2] if len(gmm_features) > 1 else latest_features
            
            # 1. Detect probability jumps
            prob_shocks = self._detect_probability_shocks(latest_features, previous_features)
            
            # 2. Detect Z-familiarity jumps
            z_fam_shocks = self._detect_z_familiarity_shocks(latest_features, previous_features)
            
            # 3. Detect entropy drops
            entropy_shocks = self._detect_entropy_shocks(latest_features, previous_features)
            
            # 4. Detect composite shocks
            composite_shocks = self._detect_composite_shocks(latest_features)
            
            # Combine all shock types
            all_shocks = prob_shocks + z_fam_shocks + entropy_shocks + composite_shocks
            
            # Filter and rank shocks
            for shock_data in all_shocks:
                if shock_data['confidence'] >= self.shock_config['confidence_threshold']:
                    shock_event = ShockEvent(
                        timestamp=gmm_features.index[-1],
                        shock_type=shock_data['type'],
                        magnitude=shock_data['magnitude'],
                        confidence=shock_data['confidence'],
                        expected_direction=shock_data['direction'],
                        duration_estimate=shock_data['duration'],
                        follow_through_probability=shock_data['follow_through']
                    )
                    shock_events.append(shock_event)
            
            # Sort by confidence and magnitude
            shock_events.sort(key=lambda x: x.confidence * x.magnitude, reverse=True)
            
            # Keep only top shocks
            max_shocks = 3  # Maximum concurrent shocks to track
            self.active_shocks = shock_events[:max_shocks]
            
            if self.active_shocks:
                tprint_success(f"✅ Detected {len(self.active_shocks)} shock events")
                for shock in self.active_shocks:
                    tprint_info(f"   ⚡ {shock.shock_type}: {shock.expected_direction} (conf: {shock.confidence:.2f})")
            else:
                tprint_info("ℹ️ No significant shock events detected")
            
            return self.active_shocks
            
        except Exception as e:
            tprint_error(f"❌ Shock detection failed: {e}")
            return []
    
    def generate_trading_signals(self, 
                               gmm_features: pd.DataFrame,
                               market_data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate comprehensive trading signals based on GMM analysis.
        
        Args:
            gmm_features: DataFrame with GMM features
            market_data: DataFrame with OHLCV data
            
        Returns:
            List of trading signals
        """
        tprint_info("🎯 Generating trading signals...")
        
        signals = []
        
        try:
            # 1. Analyze runway
            runway_analysis = self.analyze_regime_runway(gmm_features, market_data)
            
            # 2. Detect shocks
            shock_events = self.detect_shock_events(gmm_features, market_data)
            
            # 3. Generate entry signals
            entry_signals = self._generate_entry_signals(runway_analysis, shock_events, market_data)
            signals.extend(entry_signals)
            
            # 4. Generate exit/adjustment signals
            adjustment_signals = self._generate_adjustment_signals(runway_analysis, shock_events, market_data)
            signals.extend(adjustment_signals)
            
            # 5. Update performance metrics
            self.performance_metrics['signals_generated'] += len(signals)
            
            # 6. Log summary
            if signals:
                tprint_success(f"✅ Generated {len(signals)} trading signals")
                for signal in signals:
                    tprint_info(f"   📊 {signal.signal_type.value}: {signal.confidence:.2f} conf")
            else:
                tprint_info("ℹ️ No trading signals generated")
            
            return signals
            
        except Exception as e:
            tprint_error(f"❌ Signal generation failed: {e}")
            return []
    
    def _determine_regime_state(self, latest_features: pd.Series) -> RegimeState:
        """Determine current regime state from GMM features."""
        # Check for overextended clusters
        overextended_clusters = [
            k for k in range(8)  # Assuming 8 clusters
            if latest_features.get(f'cluster_{k}_is_overextended', 0) > 0.5
        ]
        
        if overextended_clusters:
            return RegimeState.OVEREXTENDED
        
        # Check GMM shock composite signal
        shock_composite = latest_features.get('gmm_shock_composite', 0)
        if shock_composite > 0.5:
            return RegimeState.TRANSITIONING
        
        # Determine trend direction from GMM state signal
        gmm_signal = latest_features.get('macro_gmm_signal', 0)
        momentum = latest_features.get('macro_regime_velocity', 0)
        
        if abs(gmm_signal) < 0.01 and abs(momentum) < 0.01:
            return RegimeState.SIDEWAYS
        elif gmm_signal > 0 and momentum > 0:
            return RegimeState.BULLISH
        elif gmm_signal < 0 and momentum < 0:
            return RegimeState.BEARISH
        else:
            return RegimeState.UNCERTAIN
    
    def _calculate_momentum_strength(self, gmm_features: pd.DataFrame) -> float:
        """Calculate momentum strength from GMM features."""
        # Use recent velocity and acceleration features
        recent_window = min(10, len(gmm_features))
        recent_features = gmm_features.tail(recent_window)
        
        # Calculate average absolute velocity
        velocity_cols = [col for col in recent_features.columns if 'velocity' in col]
        if velocity_cols:
            avg_velocity = recent_features[velocity_cols].abs().mean().mean()
        else:
            avg_velocity = 0.0
        
        # Normalize to 0-1 range
        momentum_strength = min(avg_velocity * 10, 1.0)  # Scale and cap
        
        return momentum_strength
    
    def _estimate_regime_maturity(self, 
                                 gmm_features: pd.DataFrame, 
                                 regime_state: RegimeState) -> float:
        """Estimate how mature the current regime is (0 = beginning, 1 = end)."""
        # Use entropy and probability concentration as maturity indicators
        latest_features = gmm_features.iloc[-1]
        
        # High entropy = less mature (more uncertainty)
        entropy = latest_features.get('macro_entropy', 0)
        max_entropy = np.log(8)  # Assuming 8 clusters
        
        # Low entropy concentration = more mature
        entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.5
        
        # Invert so that low entropy = high maturity
        maturity = 1.0 - entropy_normalized
        
        # Adjust based on regime consistency
        if len(gmm_features) > 5:
            recent_regime_changes = gmm_features['macro_regime_velocity'].tail(5).abs().sum()
            consistency_factor = max(0, 1.0 - recent_regime_changes * 5)  # Penalize frequent changes
            maturity *= consistency_factor
        
        return np.clip(maturity, 0, 1)
    
    def _calculate_overextension_risk(self, 
                                    latest_features: pd.Series, 
                                    regime_maturity: float) -> float:
        """Calculate overextension risk based on cluster analysis."""
        # Check overextended cluster scores
        overextended_scores = []
        for k in range(8):  # Assuming 8 clusters
            score = latest_features.get(f'cluster_{k}_overextended_score', 0)
            overextended_scores.append(score)
        
        max_overextension = max(overextended_scores) if overextended_scores else 0
        
        # Combine with regime maturity (mature regimes have higher overextension risk)
        combined_risk = (max_overextension * 0.7 + regime_maturity * 0.3)
        
        return np.clip(combined_risk, 0, 1)
    
    def _estimate_reversal_probability(self, 
                                     regime_maturity: float,
                                     overextension_risk: float,
                                     momentum_strength: float) -> float:
        """Estimate probability of regime reversal."""
        # High maturity + high overextension = high reversal probability
        # Low momentum = higher reversal probability
        
        maturity_factor = regime_maturity * 0.4
        overextension_factor = overextension_risk * 0.4
        momentum_factor = (1.0 - momentum_strength) * 0.2
        
        reversal_probability = maturity_factor + overextension_factor + momentum_factor
        
        return np.clip(reversal_probability, 0, 1)
    
    def _estimate_runway_remaining(self, 
                                 regime_maturity: float,
                                 momentum_strength: float,
                                 reversal_probability: float) -> float:
        """Estimate remaining trend runway in bars."""
        # Base runway depends on maturity and momentum
        base_runway = (1.0 - regime_maturity) * 50  # Max 50 bars for new regimes
        momentum_boost = momentum_strength * 20  # Add up to 20 bars for strong momentum
        
        # Reduce based on reversal probability
        reversal_reduction = reversal_probability * 30  # Reduce up to 30 bars
        
        estimated_runway = base_runway + momentum_boost - reversal_reduction
        
        # Apply bounds
        min_runway = self.runway_config['minimum_runway']
        max_runway = self.runway_config['maximum_runway']
        
        return np.clip(estimated_runway, min_runway, max_runway)
    
    def _calculate_runway_confidence(self, 
                                    momentum_strength: float,
                                    regime_maturity: float,
                                    data_length: int) -> float:
        """Calculate confidence in runway estimate."""
        # Confidence based on momentum strength and data availability
        momentum_confidence = momentum_strength * 0.5
        data_confidence = min(data_length / 100, 1.0) * 0.3  # More data = more confidence
        
        # Mature regimes are more predictable
        maturity_confidence = regime_maturity * 0.2
        
        total_confidence = momentum_confidence + data_confidence + maturity_confidence
        
        return np.clip(total_confidence, 0, 1)
    
    def _detect_probability_shocks(self, 
                                 latest_features: pd.Series,
                                 previous_features: pd.Series) -> List[Dict[str, Any]]:
        """Detect probability-based shock events."""
        shocks = []
        
        for k in range(8):  # Assuming 8 clusters
            prob_jump_col = f'gmm_shock_prob_jump_{k}'
            if prob_jump_col in latest_features:
                jump_magnitude = latest_features[prob_jump_col]
                if jump_magnitude > 0:
                    # Determine direction based on cluster return characteristics
                    cluster_return = latest_features.get(f'cluster_{k}_overextended_score', 0)
                    direction = 'long' if cluster_return > 0 else 'short'
                    
                    shocks.append({
                        'type': 'probability_jump',
                        'magnitude': float(jump_magnitude),
                        'confidence': float(jump_magnitude),
                        'direction': direction,
                        'duration': int(5 + jump_magnitude * 10),  # 5-15 bars
                        'follow_through': float(jump_magnitude * 0.8)
                    })
        
        return shocks
    
    def _detect_z_familiarity_shocks(self, 
                                   latest_features: pd.Series,
                                   previous_features: pd.Series) -> List[Dict[str, Any]]:
        """Detect Z-familiarity based shock events."""
        shocks = []
        
        z_fam_jump = latest_features.get('gmm_shock_z_fam_jump', 0)
        if z_fam_jump > 0:
            # Z-familiarity jumps indicate regime transitions
            # Direction based on GMM signal
            gmm_signal = latest_features.get('macro_gmm_signal', 0)
            direction = 'long' if gmm_signal > 0 else 'short'
            
            shocks.append({
                'type': 'z_familiarity_jump',
                'magnitude': float(z_fam_jump),
                'confidence': float(z_fam_jump),
                'direction': direction,
                'duration': int(8 + z_fam_jump * 12),  # 8-20 bars
                'follow_through': float(z_fam_jump * 0.9)
            })
        
        return shocks
    
    def _detect_entropy_shocks(self, 
                             latest_features: pd.Series,
                             previous_features: pd.Series) -> List[Dict[str, Any]]:
        """Detect entropy-based shock events."""
        shocks = []
        
        entropy_drop = latest_features.get('gmm_shock_entropy_drop', 0)
        if entropy_drop > 0:
            # Entropy drops indicate regime clarification
            # Direction based on momentum
            momentum = latest_features.get('macro_regime_velocity', 0)
            direction = 'long' if momentum > 0 else 'short'
            
            shocks.append({
                'type': 'entropy_drop',
                'magnitude': float(entropy_drop),
                'confidence': float(entropy_drop),
                'direction': direction,
                'duration': int(3 + entropy_drop * 7),  # 3-10 bars
                'follow_through': float(entropy_drop * 0.7)
            })
        
        return shocks
    
    def _detect_composite_shocks(self, latest_features: pd.Series) -> List[Dict[str, Any]]:
        """Detect composite shock events."""
        shocks = []
        
        composite_shock = latest_features.get('gmm_shock_composite', 0)
        shock_confidence = latest_features.get('gmm_shock_confidence', 0)
        
        if composite_shock > 0 and shock_confidence > 0.5:
            # Composite shocks are the most reliable
            gmm_signal = latest_features.get('macro_gmm_signal', 0)
            direction = 'long' if gmm_signal > 0 else 'short'
            
            shocks.append({
                'type': 'composite_shock',
                'magnitude': float(composite_shock),
                'confidence': float(shock_confidence),
                'direction': direction,
                'duration': int(10 + shock_confidence * 15),  # 10-25 bars
                'follow_through': float(shock_confidence)
            })
        
        return shocks
    
    def _generate_entry_signals(self, 
                              runway_analysis: RunwayAnalysis,
                              shock_events: List[ShockEvent],
                              market_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate entry signals based on runway and shock analysis."""
        signals = []
        current_price = market_data['close'].iloc[-1]
        
        # 1. High-conviction shock entries
        for shock in shock_events:
            if shock.confidence >= self.shock_config['confidence_threshold']:
                signal_type = SignalType.ENTRY_LONG if shock.expected_direction == 'long' else SignalType.ENTRY_SHORT
                
                # Calculate position size based on shock confidence
                position_size = self._calculate_position_size(shock.confidence, runway_analysis)
                
                # Calculate recommended stop and target
                atr = self._calculate_atr(market_data)
                recommended_stop = self._calculate_stop_loss(current_price, shock.expected_direction, atr, shock_type=True)
                recommended_target = self._calculate_target(current_price, shock.expected_direction, atr, shock_type=True)
                
                signal = TradingSignal(
                    timestamp=shock.timestamp,
                    signal_type=signal_type,
                    confidence=shock.confidence,
                    regime_state=runway_analysis.current_regime,
                    runway_remaining=runway_analysis.runway_estimate,
                    recommended_stop=recommended_stop,
                    recommended_target=recommended_target,
                    position_size_multiplier=position_size,
                    reasoning=f"High-conviction {shock.shock_type} detected",
                    metadata={'shock_event': asdict(shock)}
                )
                signals.append(signal)
        
        # 2. Trend-following entries based on runway
        if runway_analysis.runway_estimate > 10 and runway_analysis.runway_confidence > 0.6:
            if runway_analysis.current_regime == RegimeState.BULLISH:
                signal_type = SignalType.ENTRY_LONG
                direction = 'long'
            elif runway_analysis.current_regime == RegimeState.BEARISH:
                signal_type = SignalType.ENTRY_SHORT
                direction = 'short'
            else:
                return signals  # No trend-following entries in sideways/uncertain markets
            
            # Lower confidence for trend-following vs shock entries
            confidence = runway_analysis.runway_confidence * 0.7
            position_size = self._calculate_position_size(confidence, runway_analysis)
            
            atr = self._calculate_atr(market_data)
            recommended_stop = self._calculate_stop_loss(current_price, direction, atr)
            recommended_target = self._calculate_target(current_price, direction, atr)
            
            signal = TradingSignal(
                timestamp=market_data.index[-1],
                signal_type=signal_type,
                confidence=confidence,
                regime_state=runway_analysis.current_regime,
                runway_remaining=runway_analysis.runway_estimate,
                recommended_stop=recommended_stop,
                recommended_target=recommended_target,
                position_size_multiplier=position_size,
                reasoning=f"Trend-following entry based on {runway_analysis.runway_estimate:.1f} bars runway",
                metadata={'runway_analysis': asdict(runway_analysis)}
            )
            signals.append(signal)
        
        return signals
    
    def _generate_adjustment_signals(self, 
                                   runway_analysis: RunwayAnalysis,
                                   shock_events: List[ShockEvent],
                                   market_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate position adjustment signals."""
        signals = []
        
        if not self.position_state['in_position']:
            return signals  # No adjustments if not in position
        
        current_price = market_data['close'].iloc[-1]
        
        # 1. Dynamic trailing stop adjustments
        if runway_analysis.overextension_risk > 0.7:
            # Tighten stops in overextended markets
            new_stop = self._calculate_tightened_stop(current_price, market_data)
            
            if new_stop != self.position_state['current_stop']:
                signal = TradingSignal(
                    timestamp=market_data.index[-1],
                    signal_type=SignalType.ADJUST_STOP,
                    confidence=runway_analysis.overextension_risk,
                    regime_state=runway_analysis.current_regime,
                    runway_remaining=runway_analysis.runway_estimate,
                    recommended_stop=new_stop,
                    recommended_target=None,
                    position_size_multiplier=1.0,
                    reasoning="Tightening stop due to overextension risk",
                    metadata={'overextension_risk': runway_analysis.overextension_risk}
                )
                signals.append(signal)
        
        # 2. Position reduction on high reversal probability
        if runway_analysis.reversal_probability > 0.8:
            reduction_factor = 0.5  # Reduce position by 50%
            
            signal = TradingSignal(
                timestamp=market_data.index[-1],
                signal_type=SignalType.REDUCE_POSITION,
                confidence=runway_analysis.reversal_probability,
                regime_state=runway_analysis.current_regime,
                runway_remaining=runway_analysis.runway_estimate,
                recommended_stop=None,
                recommended_target=None,
                position_size_multiplier=reduction_factor,
                reasoning="Reducing position due to high reversal probability",
                metadata={'reversal_probability': runway_analysis.reversal_probability}
            )
            signals.append(signal)
        
        # 3. Exit signals based on regime changes
        if (self.position_state['position_type'] == 'long' and 
            runway_analysis.current_regime in [RegimeState.BEARISH, RegimeState.OVEREXTENDED]):
            
            signal = TradingSignal(
                timestamp=market_data.index[-1],
                signal_type=SignalType.EXIT_LONG,
                confidence=0.8,
                regime_state=runway_analysis.current_regime,
                runway_remaining=runway_analysis.runway_estimate,
                recommended_stop=None,
                recommended_target=current_price,
                position_size_multiplier=0.0,
                reasoning="Exiting long position due to bearish regime",
                metadata={'regime_change': True}
            )
            signals.append(signal)
        
        elif (self.position_state['position_type'] == 'short' and 
              runway_analysis.current_regime in [RegimeState.BULLISH, RegimeState.OVEREXTENDED]):
            
            signal = TradingSignal(
                timestamp=market_data.index[-1],
                signal_type=SignalType.EXIT_SHORT,
                confidence=0.8,
                regime_state=runway_analysis.current_regime,
                runway_remaining=runway_analysis.runway_estimate,
                recommended_stop=None,
                recommended_target=current_price,
                position_size_multiplier=0.0,
                reasoning="Exiting short position due to bullish regime",
                metadata={'regime_change': True}
            )
            signals.append(signal)
        
        return signals
    
    def _calculate_position_size(self, 
                               confidence: float, 
                               runway_analysis: RunwayAnalysis) -> float:
        """Calculate optimal position size based on confidence and runway."""
        base_size = self.position_config['base_size']
        
        # Scale by confidence
        if self.position_config['confidence_scaling']:
            confidence_multiplier = confidence
        else:
            confidence_multiplier = 1.0
        
        # Scale by runway
        if self.position_config['runway_scaling']:
            runway_multiplier = min(runway_analysis.runway_estimate / 20, 1.5)  # Cap at 1.5x
        else:
            runway_multiplier = 1.0
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * runway_multiplier
        max_size = self.position_config['max_size']
        
        return np.clip(position_size, 0.1, max_size)
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = market_data['high']
        low = market_data['low']
        close = market_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else close.std() * 0.02  # Fallback
    
    def _calculate_stop_loss(self, 
                           current_price: float,
                           direction: str,
                           atr: float,
                           shock_type: bool = False) -> float:
        """Calculate recommended stop loss."""
        if shock_type:
            # Wider stops for shock entries
            atr_multiplier = self.stop_config['shock_widening_factor'] * self.stop_config['base_stop_atr_multiple']
        else:
            atr_multiplier = self.stop_config['base_stop_atr_multiple']
        
        stop_distance = atr * atr_multiplier
        
        if direction == 'long':
            return current_price - stop_distance
        else:  # short
            return current_price + stop_distance
    
    def _calculate_target(self, 
                         current_price: float,
                         direction: str,
                         atr: float,
                         shock_type: bool = False) -> float:
        """Calculate recommended target."""
        if shock_type:
            # Larger targets for shock entries
            atr_multiplier = 3.0  # 3x ATR for shock entries
        else:
            atr_multiplier = 2.0  # 2x ATR for regular entries
        
        target_distance = atr * atr_multiplier
        
        if direction == 'long':
            return current_price + target_distance
        else:  # short
            return current_price - target_distance
    
    def _calculate_tightened_stop(self, current_price: float, market_data: pd.DataFrame) -> float:
        """Calculate tightened stop for overextended conditions."""
        atr = self._calculate_atr(market_data)
        
        # Use tighter multiplier for overextended conditions
        tightened_multiplier = self.stop_config['overextended_tighten_factor'] * self.stop_config['base_stop_atr_multiple']
        stop_distance = atr * tightened_multiplier
        
        if self.position_state['position_type'] == 'long':
            return current_price - stop_distance
        else:  # short
            return current_price + stop_distance
    
    def update_position_state(self, signal: TradingSignal, current_price: float):
        """Update internal position state based on signal."""
        if signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
            self.position_state['in_position'] = True
            self.position_state['position_type'] = 'long' if signal.signal_type == SignalType.ENTRY_LONG else 'short'
            self.position_state['entry_price'] = current_price
            self.position_state['current_stop'] = signal.recommended_stop
            self.position_state['position_size'] = signal.position_size_multiplier
            
        elif signal.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]:
            self.position_state['in_position'] = False
            self.position_state['position_type'] = None
            self.position_state['entry_price'] = None
            self.position_state['current_stop'] = None
            self.position_state['position_size'] = 0.0
            
        elif signal.signal_type == SignalType.ADJUST_STOP:
            self.position_state['current_stop'] = signal.recommended_stop
            
        elif signal.signal_type == SignalType.REDUCE_POSITION:
            self.position_state['position_size'] *= signal.position_size_multiplier
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the trading engine."""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'current_regime': self.current_regime.value,
            'current_runway': asdict(self.current_runway),
            'active_shocks': len(self.active_shocks),
            'position_state': self.position_state.copy(),
            'config': self.config
        }


# Convenience function for quick signal generation
def generate_gmm_trading_signals(gmm_features: pd.DataFrame,
                                market_data: pd.DataFrame,
                                config: Optional[Dict[str, Any]] = None) -> List[TradingSignal]:
    """
    Convenience function to generate trading signals from GMM features.
    
    Args:
        gmm_features: DataFrame with GMM features
        market_data: DataFrame with OHLCV data
        config: Trading engine configuration
        
    Returns:
        List of trading signals
    """
    engine = GMMTradingEngine(config)
    return engine.generate_trading_signals(gmm_features, market_data)


# Export main classes and functions
__all__ = [
    'GMMTradingEngine',
    'TradingSignal',
    'RunwayAnalysis',
    'ShockEvent',
    'RegimeState',
    'SignalType',
    'generate_gmm_trading_signals'
]
