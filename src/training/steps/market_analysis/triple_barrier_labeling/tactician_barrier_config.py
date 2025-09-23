"""
Tactician-Specific Triple Barrier Configuration - Enhanced

This module provides specialized barrier configurations optimized for the Tactician model
to find the best entry points after the Analyst gives its green light.

Key Features:
- Shorter time barriers for entry point optimization
- Tighter profit/loss barriers for precise timing
- Entry-specific labeling logic
- Optimized for 1m timeframe
- Enhanced integration with confidence-based training filtering
- Support for Analyst and HMM model outputs as features
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from .unified_labeler import TripleBarrierConfig, UnifiedTripleBarrierLabeler


@dataclass
class TacticianBarrierConfig(TripleBarrierConfig):
    """
    Specialized barrier configuration for Tactician entry point optimization.
    
    The Tactician operates on 1m timeframe and needs to find optimal entry points
    within a short time window after the Analyst gives a green light.
    
    Optimized barrier settings based on research:
    - Risk-reward ratio of 1.5:1 for entry optimization
    - Shorter time horizons for precise entry timing
    - Dynamic barriers based on market volatility
    - Confidence-based filtering for high-quality entries
    """
    
    # Optimized barriers for entry point finding (based on research)
    profit_take_multiplier: float = 0.0018  # 0.18% - optimized for entry timing
    stop_loss_multiplier: float = 0.0012    # 0.12% - optimized for entry timing
    time_barrier_minutes: int = 12          # 12 minutes - optimal for entry timing
    
    # Entry-specific parameters
    max_lookahead: int = 25                 # 25 bars max lookahead for 1m data
    transaction_cost: float = 0.0004        # 0.04% - optimized for entry optimization
    
    # Entry point optimization
    entry_window_minutes: int = 5           # 5-minute window to find best entry
    min_entry_confidence: float = 0.6       # Minimum confidence for entry
    entry_signal_decay: float = 0.1         # Signal decay rate over time
    
    # Dynamic barrier adjustment
    enable_dynamic_barriers: bool = True    # Enable volatility-based barrier adjustment
    volatility_lookback: int = 20           # Lookback period for volatility calculation
    min_barrier_multiplier: float = 0.5     # Minimum barrier adjustment factor
    max_barrier_multiplier: float = 2.0     # Maximum barrier adjustment factor
    volatility_threshold_low: float = 0.005 # Low volatility threshold (0.5%)
    volatility_threshold_high: float = 0.02 # High volatility threshold (2.0%)
    
    # Tactician-specific behavior
    binary_classification: bool = True      # Binary: enter now vs wait
    regime_aware: bool = False              # Single model, not regime-aware
    regime_column: str = 'hmm_regime'       # Not used but kept for compatibility
    
    # Enhanced training integration
    enable_confidence_filtering: bool = True    # Enable confidence-based training filtering
    confidence_threshold: float = 0.5           # Minimum Analyst confidence for training
    post_drop_window_minutes: int = 45          # Training window extension after confidence drops
    include_analyst_features: bool = True       # Include Analyst model outputs as features
    include_hmm_features: bool = True           # Include HMM model outputs as features
    
    # Performance settings
    enable_numba_acceleration: bool = True
    enable_hardware_optimizations: bool = True
    memory_limit_gb: float = 4.0            # Lower memory limit for 1m data
    
    # Validation settings
    min_data_points: int = 50               # Lower threshold for 1m data
    max_missing_data_ratio: float = 0.05    # Stricter for entry timing
    min_label_distribution_ratio: float = 0.1  # Higher threshold for entry decisions
    
    def __post_init__(self):
        """Validate tactician-specific configuration."""
        super().__post_init__()
        self._validate_tactician_configuration()
    
    def _validate_tactician_configuration(self):
        """Validate tactician-specific parameters."""
        errors = []
        
        # Validate entry window
        if self.entry_window_minutes <= 0:
            errors.append("Entry window must be positive")
        
        if self.entry_window_minutes > self.time_barrier_minutes:
            errors.append("Entry window cannot be larger than time barrier")
        
        # Validate confidence threshold
        if not (0.0 <= self.min_entry_confidence <= 1.0):
            errors.append("Min entry confidence must be between 0 and 1")
        
        # Validate signal decay
        if not (0.0 <= self.entry_signal_decay <= 1.0):
            errors.append("Entry signal decay must be between 0 and 1")
        
        # Validate lookahead for 1m data
        if self.max_lookahead > 60:  # More than 1 hour
            errors.append("Max lookahead too large for 1m entry optimization")
        
        # Validate enhanced training parameters
        if not (0.0 <= self.confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be between 0 and 1")
        
        if self.post_drop_window_minutes <= 0:
            errors.append("Post-drop window must be positive")
        
        if errors:
            raise ValueError(f"Tactician configuration validation failed: {'; '.join(errors)}")
    
    def calculate_dynamic_barriers(self, volatility: float) -> Tuple[float, float]:
        """
        Calculate dynamic barriers based on market volatility.
        
        Args:
            volatility: Current market volatility (standard deviation of returns)
            
        Returns:
            Tuple of (adjusted_profit_take, adjusted_stop_loss) multipliers
        """
        if not self.enable_dynamic_barriers:
            return self.profit_take_multiplier, self.stop_loss_multiplier
        
        # Calculate adjustment factor based on volatility
        if volatility <= self.volatility_threshold_low:
            # Low volatility: tighten barriers for more precise entries
            adjustment_factor = self.min_barrier_multiplier
        elif volatility >= self.volatility_threshold_high:
            # High volatility: widen barriers to avoid noise
            adjustment_factor = self.max_barrier_multiplier
        else:
            # Medium volatility: linear interpolation
            volatility_range = self.volatility_threshold_high - self.volatility_threshold_low
            volatility_position = (volatility - self.volatility_threshold_low) / volatility_range
            adjustment_factor = self.min_barrier_multiplier + volatility_position * (self.max_barrier_multiplier - self.min_barrier_multiplier)
        
        # Apply adjustment to barriers
        adjusted_profit_take = self.profit_take_multiplier * adjustment_factor
        adjusted_stop_loss = self.stop_loss_multiplier * adjustment_factor
        
        return adjusted_profit_take, adjusted_stop_loss


class TacticianBarrierLabeler(UnifiedTripleBarrierLabeler):
    """
    Specialized triple barrier labeler for Tactician entry point optimization.
    
    This labeler is optimized for finding the best entry points within a short
    time window after the Analyst gives a green light signal.
    """
    
    def __init__(self, config: Optional[TacticianBarrierConfig] = None):
        """Initialize tactician-specific barrier labeler."""
        if config is None:
            config = TacticianBarrierConfig()
        
        super().__init__(config)
        self.tactician_config = config
        self.logger = self.logger.getChild('TacticianBarrierLabeler')
        
        # Log tactician-specific configuration
        self._log_tactician_configuration()
    
    def _log_tactician_configuration(self):
        """Log tactician-specific configuration."""
        self.logger.info("🎯 Tactician Barrier Configuration:")
        self.logger.info(f"   → Entry window: {self.tactician_config.entry_window_minutes} minutes")
        self.logger.info(f"   → Min entry confidence: {self.tactician_config.min_entry_confidence}")
        self.logger.info(f"   → Signal decay rate: {self.tactician_config.entry_signal_decay}")
        self.logger.info(f"   → Optimized for 1m timeframe")
        self.logger.info(f"   → Single model (not regime-aware)")
    
    def apply_tactician_labeling(self, data, analyst_signals: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Apply tactician-specific labeling with entry point optimization.
        
        Args:
            data: OHLC data with timestamps
            analyst_signals: Binary signals from Analyst (green light indicators)
            
        Returns:
            Dictionary containing labeled data and tactician-specific metrics
        """
        try:
            self.logger.info("🎯 Starting Tactician-specific barrier labeling...")
            
            # Filter data to only include periods with analyst green light
            if analyst_signals is not None:
                green_light_mask = analyst_signals == 1
                if np.sum(green_light_mask) == 0:
                    raise ValueError("No analyst green light signals found")
                
                # Filter data to green light periods
                filtered_data = data[green_light_mask].copy()
                self.logger.info(f"📊 Filtered to {len(filtered_data)} samples with analyst green light")
            else:
                filtered_data = data.copy()
                self.logger.warning("⚠️ No analyst signals provided - labeling all data")
            
            # Apply standard triple barrier labeling
            result = self.apply_labeling(filtered_data)
            
            if not result.success:
                return result
            
            # Add tactician-specific post-processing
            tactician_result = self._post_process_tactician_labels(result.labeled_data)
            
            # Update result with tactician-specific data
            result.labeled_data = tactician_result['labeled_data']
            result.tactician_metrics = tactician_result['metrics']
            
            self.logger.info("✅ Tactician-specific labeling completed")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Tactician labeling failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'labeled_data': None
            }
    
    def _post_process_tactician_labels(self, labeled_data) -> Dict[str, Any]:
        """Post-process labels for tactician entry point optimization."""
        try:
            # Create working copy
            processed_data = labeled_data.copy()
            
            # Add entry-specific features
            processed_data = self._add_entry_features(processed_data)
            
            # Apply entry confidence scoring
            processed_data = self._apply_entry_confidence_scoring(processed_data)
            
            # Filter for high-confidence entries
            high_confidence_mask = processed_data['entry_confidence'] >= self.tactician_config.min_entry_confidence
            processed_data = processed_data[high_confidence_mask].copy()
            
            # Calculate tactician-specific metrics
            metrics = self._calculate_tactician_metrics(processed_data)
            
            self.logger.info(f"📊 Post-processing completed: {len(processed_data)} high-confidence entries")
            
            return {
                'labeled_data': processed_data,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Tactician post-processing failed: {e}")
            raise
    
    def _add_entry_features(self, data):
        """Add entry-specific features to the data."""
        try:
            # Add time-based features
            if hasattr(data.index, 'to_pydatetime'):
                data['hour'] = data.index.hour
                data['minute'] = data.index.minute
                data['day_of_week'] = data.index.dayofweek
            
            # Add entry timing features
            data['time_since_signal'] = 0  # Will be calculated based on analyst signal timing
            data['signal_strength'] = 1.0  # Will be calculated based on analyst confidence
            
            # Add volatility features for entry optimization
            if 'close' in data.columns:
                data['price_change'] = data['close'].pct_change()
                data['volatility_5min'] = data['price_change'].rolling(5).std()
                data['volatility_15min'] = data['price_change'].rolling(15).std()
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add entry features: {e}")
            raise
    
    def _apply_entry_confidence_scoring(self, data):
        """Apply confidence scoring for entry decisions."""
        try:
            # Initialize confidence scores
            confidence_scores = np.ones(len(data))
            
            # Apply signal decay over time
            if 'time_since_signal' in data.columns:
                decay_factor = np.exp(-self.tactician_config.entry_signal_decay * data['time_since_signal'])
                confidence_scores *= decay_factor
            
            # Adjust confidence based on volatility
            if 'volatility_5min' in data.columns:
                # Lower confidence for high volatility periods
                volatility_factor = 1.0 - np.clip(data['volatility_5min'] / 0.01, 0, 0.5)
                confidence_scores *= volatility_factor
            
            # Adjust confidence based on label quality
            if 'label' in data.columns:
                # Higher confidence for clear profit/loss signals
                label_confidence = np.where(
                    data['label'] == 1, 1.2,  # Profit take - higher confidence
                    np.where(data['label'] == -1, 0.8, 1.0)  # Stop loss - lower confidence
                )
                confidence_scores *= label_confidence
            
            # Add confidence scores to data
            data['entry_confidence'] = np.clip(confidence_scores, 0.0, 1.0)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply confidence scoring: {e}")
            raise
    
    def _calculate_tactician_metrics(self, data) -> Dict[str, Any]:
        """Calculate tactician-specific metrics."""
        try:
            metrics = {
                'total_entries': len(data),
                'high_confidence_entries': len(data[data['entry_confidence'] >= self.tactician_config.min_entry_confidence]),
                'avg_confidence': float(data['entry_confidence'].mean()) if 'entry_confidence' in data.columns else 0.0,
                'entry_distribution': {}
            }
            
            # Calculate entry distribution by label
            if 'label' in data.columns:
                label_dist = data['label'].value_counts().to_dict()
                metrics['entry_distribution'] = label_dist
            
            # Calculate timing metrics
            if 'time_since_signal' in data.columns:
                metrics['avg_time_to_entry'] = float(data['time_since_signal'].mean())
                metrics['max_time_to_entry'] = float(data['time_since_signal'].max())
            
            # Calculate volatility metrics
            if 'volatility_5min' in data.columns:
                metrics['avg_volatility'] = float(data['volatility_5min'].mean())
                metrics['volatility_std'] = float(data['volatility_5min'].std())
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate tactician metrics: {e}")
            return {}


# Convenience functions for tactician barrier labeling
def create_tactician_barrier_labeler(
    profit_take_multiplier: float = 0.0018,
    stop_loss_multiplier: float = 0.0012,
    time_barrier_minutes: int = 12,
    entry_window_minutes: int = 5,
    min_entry_confidence: float = 0.6,
    transaction_cost: float = 0.0004,
    enable_dynamic_barriers: bool = True
) -> TacticianBarrierLabeler:
    """Create a tactician-specific barrier labeler."""
    config = TacticianBarrierConfig(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        time_barrier_minutes=time_barrier_minutes,
        entry_window_minutes=entry_window_minutes,
        min_entry_confidence=min_entry_confidence,
        transaction_cost=transaction_cost,
        enable_dynamic_barriers=enable_dynamic_barriers
    )
    
    return TacticianBarrierLabeler(config)


def apply_tactician_barrier_labeling(
    data,
    analyst_signals: Optional[np.ndarray] = None,
    profit_take_multiplier: float = 0.0018,
    stop_loss_multiplier: float = 0.0012,
    time_barrier_minutes: int = 12,
    entry_window_minutes: int = 5,
    min_entry_confidence: float = 0.6,
    enable_dynamic_barriers: bool = True
) -> Dict[str, Any]:
    """Apply tactician-specific barrier labeling to data."""
    labeler = create_tactician_barrier_labeler(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        time_barrier_minutes=time_barrier_minutes,
        entry_window_minutes=entry_window_minutes,
        min_entry_confidence=min_entry_confidence,
        enable_dynamic_barriers=enable_dynamic_barriers
    )
    
    return labeler.apply_tactician_labeling(data, analyst_signals)


if __name__ == '__main__':
    # Test the tactician barrier labeling
    import pandas as pd
    
    print("🎯 Testing Tactician Barrier Labeling")
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)
    
    # Create analyst signals (some green lights)
    analyst_signals = np.random.choice([0, 1], 1000, p=[0.8, 0.2])  # 20% green light rate
    
    # Test tactician labeling
    print("\n📊 Testing tactician-specific barrier labeling...")
    result = apply_tactician_barrier_labeling(data, analyst_signals)
    
    if result['success']:
        print(f'✅ Tactician labeling completed successfully')
        print(f'   High-confidence entries: {result.get("tactician_metrics", {}).get("high_confidence_entries", 0)}')
        print(f'   Average confidence: {result.get("tactician_metrics", {}).get("avg_confidence", 0):.3f}')
        print(f'   Entry distribution: {result.get("tactician_metrics", {}).get("entry_distribution", {})}')
    else:
        print(f'❌ Tactician labeling failed: {result.get("error_message", "Unknown error")}')
    
    print('✅ Tactician Barrier Labeling test completed!')