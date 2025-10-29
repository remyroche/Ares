"""
Optimized Parameters Integration

This module ensures that optimized parameters from final_parameters_optimization
are properly used throughout the trading system.
"""

from typing import Dict, Any, Optional
from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from .unified_model_loader import get_unified_model_loader

logger = system_logger.getChild('OptimizedParametersIntegration')


class OptimizedParametersIntegration:
    """
    Helper class to integrate optimized parameters throughout trading components.
    """
    
    def __init__(self) -> None:
        """Initialize optimized parameters integration."""
        self.logger = logger.getChild('OptimizedParametersIntegration')
        self.unified_loader = get_unified_model_loader()
        self._cached_parameters: Optional[Dict[str, Any]] = None
        tprint_info("🔄 Optimized parameters integration initialized")
    
    async def get_optimized_parameters(
        self,
        symbol: str = "ETHUSDT",
        exchange: str = "binance",
        timeframe: str = "15m",
        direction: str = "long"
    ) -> Dict[str, Any]:
        """
        Get optimized parameters, caching them for reuse.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction
            
        Returns:
            Dictionary of optimized parameters
        """
        if self._cached_parameters is None:
            tprint_info(f"🔄 Loading optimized parameters for {symbol} ({timeframe}, {direction})")
            self._cached_parameters = await self.unified_loader.load_optimized_parameters(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction
            )
            if self._cached_parameters:
                tprint_success(f"✅ Loaded {len(self._cached_parameters)} optimized parameters")
            else:
                tprint_warning("⚠️ No optimized parameters found, using defaults")
        
        return self._cached_parameters.copy()
    
    def apply_to_position_sizer(self, position_sizer: Any, params: Dict[str, Any]) -> None:
        """Apply optimized parameters to PositionSizer."""
        try:
            tprint_info("🔄 Applying optimized parameters to PositionSizer")
            # Validate and apply parameters
            if hasattr(position_sizer, 'confidence_threshold'):
                value = params.get('confidence_threshold', 0.75)
                if not (0.0 <= value <= 1.0):
                    tprint_error(f"❌ confidence_threshold must be between 0.0 and 1.0, got {value}")
                    raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {value}")
                position_sizer.confidence_threshold = value
                tprint_info(f"📊 Set confidence_threshold to {value}")
                
            if hasattr(position_sizer, 'position_sizing_factor'):
                value = params.get('position_sizing_factor', 0.02)
                if value < 0 or value > 1.0:
                    raise ValueError(f"position_sizing_factor must be between 0.0 and 1.0, got {value}")
                position_sizer.position_sizing_factor = value
                
            if hasattr(position_sizer, 'max_position_size'):
                # Adjust max position size based on optimized factor
                base_max = getattr(position_sizer, 'max_position_size', 0.5)
                factor = params.get('position_sizing_factor', 0.02)
                if factor < 0 or factor > 1.0:
                    raise ValueError(f"position_sizing_factor must be between 0.0 and 1.0, got {factor}")
                position_sizer.max_position_size = min(base_max, factor * 10)  # Scale appropriately
            
            # Set confidence multiplier from final_parameters_optimization
            # Check for various parameter names that might contain the multiplier
            confidence_multiplier = (
                params.get('confidence_multiplier') or
                params.get('position_size_confidence_multiplier') or
                params.get('long_position_size_multiplier') or  # For long trades
                params.get('short_position_size_multiplier') or  # For short trades
                1.0  # Default if not found
            )
            
            if hasattr(position_sizer, 'set_confidence_multiplier'):
                position_sizer.set_confidence_multiplier(confidence_multiplier)
            elif hasattr(position_sizer, 'confidence_multiplier'):
                position_sizer.confidence_multiplier = confidence_multiplier
                
            tprint_success(f"✅ Applied optimized parameters to PositionSizer (confidence_multiplier: {confidence_multiplier})")
            self.logger.info(f"✅ Applied optimized parameters to PositionSizer (confidence_multiplier: {confidence_multiplier})")
        except Exception as e:
            tprint_error(f"❌ Failed to apply parameters to PositionSizer: {e}")
            self.logger.warning(f"⚠️ Failed to apply parameters to PositionSizer: {e}")
            raise

    def apply_to_risk_calculator(self, risk_calculator: Any, params: Dict[str, Any]) -> None:
        """Apply optimized parameters to RiskCalculator."""
        try:
            tprint_info("🔄 Applying optimized parameters to RiskCalculator")
            if hasattr(risk_calculator, 'stop_loss_pct'):
                value = params.get('stop_loss_pct', 0.03)
                if value < 0 or value > 1.0:
                    raise ValueError(f"stop_loss_pct must be between 0.0 and 1.0, got {value}")
                risk_calculator.stop_loss_pct = value
                
            if hasattr(risk_calculator, 'take_profit_pct'):
                value = params.get('take_profit_pct', 0.06)
                if value < 0 or value > 1.0:
                    raise ValueError(f"take_profit_pct must be between 0.0 and 1.0, got {value}")
                risk_calculator.take_profit_pct = value
                
            if hasattr(risk_calculator, 'max_position_risk'):
                # Adjust max position risk based on stop_loss_pct
                stop_loss_pct = params.get('stop_loss_pct', 0.03)
                if stop_loss_pct < 0 or stop_loss_pct > 1.0:
                    raise ValueError(f"stop_loss_pct must be between 0.0 and 1.0, got {stop_loss_pct}")
                risk_calculator.max_position_risk = stop_loss_pct  # Use stop_loss as max risk
                
            tprint_success("✅ Applied optimized parameters to RiskCalculator")
            self.logger.info("✅ Applied optimized parameters to RiskCalculator")
        except Exception as e:
            tprint_error(f"❌ Failed to apply parameters to RiskCalculator: {e}")
            self.logger.warning(f"⚠️ Failed to apply parameters to RiskCalculator: {e}")
            raise

    def apply_to_leverage_manager(self, leverage_manager: Any, params: Dict[str, Any]) -> None:
        """Apply optimized parameters to LeverageManager."""
        try:
            tprint_info("🔄 Applying optimized parameters to LeverageManager")
            if hasattr(leverage_manager, 'leverage_multiplier'):
                value = params.get('leverage_multiplier', 1.5)
                if value < 1.0 or value > 100.0:
                    raise ValueError(f"leverage_multiplier must be between 1.0 and 100.0, got {value}")
                leverage_manager.leverage_multiplier = value
                
            tprint_success("✅ Applied optimized parameters to LeverageManager")
            self.logger.info("✅ Applied optimized parameters to LeverageManager")
        except Exception as e:
            tprint_error(f"❌ Failed to apply parameters to LeverageManager: {e}")
            self.logger.warning(f"⚠️ Failed to apply parameters to LeverageManager: {e}")
            raise

    def apply_to_signal_components(self, signal_components: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Apply optimized parameters to signal generation components."""
        try:
            tprint_info("🔄 Applying optimized parameters to signal components")
            # Analyst signals
            if 'analyst' in signal_components:
                analyst = signal_components['analyst']
                if hasattr(analyst, 'confidence_threshold'):
                    value = params.get('confidence_threshold', 0.75)
                    if not (0.0 <= value <= 1.0):
                        raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {value}")
                    analyst.confidence_threshold = value

            # Tactician signals
            if 'tactician' in signal_components:
                tactician = signal_components['tactician']
                if hasattr(tactician, 'confidence_threshold'):
                    value = params.get('confidence_threshold', 0.75)
                    if not (0.0 <= value <= 1.0):
                        raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {value}")
                    tactician.confidence_threshold = value

            # Signal combiner
            if 'signal_combiner' in signal_components:
                combiner = signal_components['signal_combiner']
                if hasattr(combiner, 'weights'):
                    if hasattr(combiner.weights, 'confidence_threshold'):
                        value = params.get('confidence_threshold', 0.75)
                        if not (0.0 <= value <= 1.0):
                            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {value}")
                        combiner.weights.confidence_threshold = value

            tprint_success("✅ Applied optimized parameters to signal components")
            self.logger.info("✅ Applied optimized parameters to signal components")
        except Exception as e:
            tprint_error(f"❌ Failed to apply parameters to signal components: {e}")
            self.logger.warning(f"⚠️ Failed to apply parameters to signal components: {e}")
            raise

    def apply_to_config(self, config: Any, params: Dict[str, Any]) -> None:
        """Apply optimized parameters to TradingConfig."""
        try:
            tprint_info("🔄 Applying optimized parameters to TradingConfig")
            if hasattr(config, 'regime_confidence_threshold'):
                value = params.get('regime_confidence_threshold', 0.7)
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"regime_confidence_threshold must be between 0.0 and 1.0, got {value}")
                config.regime_confidence_threshold = value
                
            if hasattr(config, 'signal_confidence_threshold'):
                value = params.get('signal_confidence_threshold', 0.6)
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"signal_confidence_threshold must be between 0.0 and 1.0, got {value}")
                config.signal_confidence_threshold = value
                
            if hasattr(config, 'ensemble_weight_analyst'):
                value = params.get('ensemble_weight_analyst', 0.6)
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"ensemble_weight_analyst must be between 0.0 and 1.0, got {value}")
                config.ensemble_weight_analyst = value
                
            if hasattr(config, 'ensemble_weight_tactician'):
                value = params.get('ensemble_weight_tactician', 0.4)
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"ensemble_weight_tactician must be between 0.0 and 1.0, got {value}")
                config.ensemble_weight_tactician = value

            # Validate weights sum to 1.0
            analyst_weight = params.get('ensemble_weight_analyst', 0.6)
            tactician_weight = params.get('ensemble_weight_tactician', 0.4)
            if abs(analyst_weight + tactician_weight - 1.0) > 0.01:
                tprint_warning(
                    f"⚠️ Ensemble weights don't sum to 1.0: {analyst_weight} + {tactician_weight} = "
                    f"{analyst_weight + tactician_weight}"
                )
                self.logger.warning(
                    f"⚠️ Ensemble weights don't sum to 1.0: {analyst_weight} + {tactician_weight} = "
                    f"{analyst_weight + tactician_weight}"
                )

            # Store all optimized parameters in custom_params for access
            if not hasattr(config, 'custom_params'):
                config.custom_params = {}
            config.custom_params.update(params)

            tprint_success("✅ Applied optimized parameters to TradingConfig")
            self.logger.info("✅ Applied optimized parameters to TradingConfig")
        except Exception as e:
            tprint_error(f"❌ Failed to apply parameters to config: {e}")
            self.logger.warning(f"⚠️ Failed to apply parameters to config: {e}")
            raise


# Global instance
_optimized_params_integration: Optional[OptimizedParametersIntegration] = None


def get_optimized_params_integration() -> OptimizedParametersIntegration:
    """
    Get or create global optimized parameters integration instance.
    
    Returns:
        OptimizedParametersIntegration instance
    """
    global _optimized_params_integration
    
    if _optimized_params_integration is None:
        tprint_info("🔄 Creating optimized parameters integration instance")
        _optimized_params_integration = OptimizedParametersIntegration()
        tprint_success("✅ Optimized parameters integration instance created")
    
    return _optimized_params_integration
