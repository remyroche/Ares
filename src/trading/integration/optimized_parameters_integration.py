"""
Optimized Parameters Integration

This module ensures that optimized parameters from final_parameters_optimization
are properly used throughout the trading system.
"""

from typing import Dict, Any, Optional
from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning
from .unified_model_loader import get_unified_model_loader

logger = system_logger.getChild('OptimizedParametersIntegration')


class OptimizedParametersIntegration:
    """
    Helper class to integrate optimized parameters throughout trading components.
    """
    
    def __init__(self):
        self.logger = logger.getChild('OptimizedParametersIntegration')
        self.unified_loader = get_unified_model_loader()
        self._cached_parameters: Optional[Dict[str, Any]] = None
    
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
            self._cached_parameters = await self.unified_loader.load_optimized_parameters(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction
            )
        
        return self._cached_parameters.copy()
    
    def apply_to_position_sizer(self, position_sizer: Any, params: Dict[str, Any]) -> None:
        """Apply optimized parameters to PositionSizer."""
        try:
            if hasattr(position_sizer, 'confidence_threshold'):
                position_sizer.confidence_threshold = params.get('confidence_threshold', 0.75)
            if hasattr(position_sizer, 'position_sizing_factor'):
                position_sizer.position_sizing_factor = params.get('position_sizing_factor', 0.02)
            if hasattr(position_sizer, 'max_position_size'):
                # Adjust max position size based on optimized factor
                base_max = getattr(position_sizer, 'max_position_size', 0.5)
                factor = params.get('position_sizing_factor', 0.02)
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
                
            self.logger.info(f"✅ Applied optimized parameters to PositionSizer (confidence_multiplier: {confidence_multiplier})")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply parameters to PositionSizer: {e}")
    
    def apply_to_risk_calculator(self, risk_calculator: Any, params: Dict[str, Any]) -> None:
        """Apply optimized parameters to RiskCalculator."""
        try:
            if hasattr(risk_calculator, 'stop_loss_pct'):
                risk_calculator.stop_loss_pct = params.get('stop_loss_pct', 0.03)
            if hasattr(risk_calculator, 'take_profit_pct'):
                risk_calculator.take_profit_pct = params.get('take_profit_pct', 0.06)
            if hasattr(risk_calculator, 'max_position_risk'):
                # Adjust max position risk based on stop_loss_pct
                stop_loss_pct = params.get('stop_loss_pct', 0.03)
                risk_calculator.max_position_risk = stop_loss_pct  # Use stop_loss as max risk
            self.logger.info("✅ Applied optimized parameters to RiskCalculator")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply parameters to RiskCalculator: {e}")
    
    def apply_to_leverage_manager(self, leverage_manager: Any, params: Dict[str, Any]) -> None:
        """Apply optimized parameters to LeverageManager."""
        try:
            if hasattr(leverage_manager, 'leverage_multiplier'):
                leverage_manager.leverage_multiplier = params.get('leverage_multiplier', 1.5)
            self.logger.info("✅ Applied optimized parameters to LeverageManager")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply parameters to LeverageManager: {e}")
    
    def apply_to_signal_components(self, signal_components: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Apply optimized parameters to signal generation components."""
        try:
            # Analyst signals
            if 'analyst' in signal_components:
                analyst = signal_components['analyst']
                if hasattr(analyst, 'confidence_threshold'):
                    analyst.confidence_threshold = params.get('confidence_threshold', 0.75)
            
            # Tactician signals
            if 'tactician' in signal_components:
                tactician = signal_components['tactician']
                if hasattr(tactician, 'confidence_threshold'):
                    tactician.confidence_threshold = params.get('confidence_threshold', 0.75)
            
            # Signal combiner
            if 'signal_combiner' in signal_components:
                combiner = signal_components['signal_combiner']
                if hasattr(combiner, 'weights'):
                    if hasattr(combiner.weights, 'confidence_threshold'):
                        combiner.weights.confidence_threshold = params.get('confidence_threshold', 0.75)
            
            self.logger.info("✅ Applied optimized parameters to signal components")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply parameters to signal components: {e}")
    
    def apply_to_config(self, config: Any, params: Dict[str, Any]) -> None:
        """Apply optimized parameters to TradingConfig."""
        try:
            if hasattr(config, 'regime_confidence_threshold'):
                config.regime_confidence_threshold = params.get('regime_confidence_threshold', 0.7)
            if hasattr(config, 'signal_confidence_threshold'):
                config.signal_confidence_threshold = params.get('signal_confidence_threshold', 0.6)
            if hasattr(config, 'ensemble_weight_analyst'):
                config.ensemble_weight_analyst = params.get('ensemble_weight_analyst', 0.6)
            if hasattr(config, 'ensemble_weight_tactician'):
                config.ensemble_weight_tactician = params.get('ensemble_weight_tactician', 0.4)
            
            # Store all optimized parameters in custom_params for access
            if not hasattr(config, 'custom_params'):
                config.custom_params = {}
            config.custom_params.update(params)
            
            self.logger.info("✅ Applied optimized parameters to TradingConfig")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply parameters to config: {e}")


# Global instance
_optimized_params_integration: Optional[OptimizedParametersIntegration] = None


def get_optimized_params_integration() -> OptimizedParametersIntegration:
    """Get or create global optimized parameters integration instance."""
    global _optimized_params_integration
    
    if _optimized_params_integration is None:
        _optimized_params_integration = OptimizedParametersIntegration()
    
    return _optimized_params_integration
