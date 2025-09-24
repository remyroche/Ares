"""
Leverage Manager

Simplified leverage management using ML confidence scores.
Based on existing tactician approach with simple confidence-based leverage calculation.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from ..config.trading_config import TradingConfig
from src.config.leverage_constants import MIN_LEVERAGE, MAX_LEVERAGE, validate_leverage

logger = system_logger.getChild('LeverageManager')

@dataclass
class LeverageResult:
    """Leverage calculation result."""
    symbol: str
    recommended_leverage: float
    max_leverage: float
    min_leverage: float
    confidence: float
    leverage_multiplier: float
    metadata: Dict[str, Any]

class LeverageManager:
    """
    Simplified leverage manager using ML confidence scores.
    
    Based on existing tactician approach:
    - Uses ML confidence scores for leverage calculation
    - Simple confidence-based leverage with configurable limits
    - Leverage capped between min and max values
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logger.getChild('LeverageManager')
        
        # Leverage configuration - using centralized constants
        self.min_leverage: float = MIN_LEVERAGE  # Minimum leverage (5x)
        self.max_leverage: float = MAX_LEVERAGE  # Maximum leverage (100x)
        self.leverage_multiplier: float = 1.0  # Leverage multiplier
        self.leverage_combined_threshold: float = 0.75  # Minimum confidence threshold
        
        # State management
        self.is_initialized: bool = False
        self.leverage_history: list[dict[str, Any]] = []
    
    @handles_errors
    async def initialize(self) -> bool:
        """Initialize leverage manager."""
        try:
            self.logger.info("Initializing Leverage Manager...")
            
            # Validate configuration
            if not self._validate_configuration():
                return False
            
            self.is_initialized = True
            self.logger.info("✅ Leverage Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Leverage Manager: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate leverage manager configuration."""
        try:
            if self.min_leverage <= 0 or self.min_leverage >= self.max_leverage:
                self.logger.error("Invalid leverage range configuration")
                return False
            if self.leverage_multiplier <= 0:
                self.logger.error("Invalid leverage_multiplier configuration")
                return False
            if self.leverage_combined_threshold <= 0 or self.leverage_combined_threshold > 1:
                self.logger.error("Invalid leverage_combined_threshold configuration")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    @handles_errors
    @log_execution_time()
    @traced(span_name="calculate_leverage")
    async def calculate_leverage(
        self,
        symbol: str,
        ml_predictions: Dict[str, Any],
        current_price: float = 0.0,
        account_balance: float = 1000.0,
        analyst_confidence: float = 0.5,
        tactician_confidence: float = 0.5
    ) -> LeverageResult:
        """
        Calculate leverage using ML confidence scores with simplified approach.
        
        Args:
            symbol: Trading symbol
            ml_predictions: ML model predictions
            current_price: Current market price
            account_balance: Account balance
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score
            
        Returns:
            LeverageResult: Leverage recommendation
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Leverage Manager not initialized")
            
            # Extract ML predictions
            combined_confidence = ml_predictions.get('combined_confidence', 0.5)
            intensity = ml_predictions.get('intensity', 1.0)
            reliability = ml_predictions.get('reliability', 1.0)
            risk_score = ml_predictions.get('risk_score', 0.0)
            
            # Simplified leverage calculation: ML confidence * multiplier, capped between min/max
            base_leverage = combined_confidence * self.leverage_multiplier
            
            # Apply intensity and reliability adjustments
            intensity_factor = 0.8 + (intensity * 0.4)  # 0.8 to 1.2
            reliability_factor = 0.8 + (reliability * 0.4)  # 0.8 to 1.2
            
            # Apply risk adjustment (higher risk = lower leverage)
            risk_factor = 1.0 - (risk_score * 0.3)  # 0.7 to 1.0
            
            # Calculate final leverage
            adjusted_leverage = base_leverage * intensity_factor * reliability_factor * risk_factor
            # Validate and clamp leverage to centralized limits
            final_leverage = validate_leverage(adjusted_leverage)
            final_leverage = max(self.min_leverage, min(self.max_leverage, final_leverage))
            
            # Create result
            result = LeverageResult(
                symbol=symbol,
                recommended_leverage=final_leverage,
                max_leverage=self.max_leverage,
                min_leverage=self.min_leverage,
                confidence=combined_confidence,
                leverage_multiplier=self.leverage_multiplier,
                metadata={
                    'current_price': current_price,
                    'account_balance': account_balance,
                    'analyst_confidence': analyst_confidence,
                    'tactician_confidence': tactician_confidence,
                    'base_leverage': base_leverage,
                    'intensity_factor': intensity_factor,
                    'reliability_factor': reliability_factor,
                    'risk_factor': risk_factor,
                    'intensity': intensity,
                    'reliability': reliability,
                    'risk_score': risk_score
                }
            )
            
            # Store in history
            self.leverage_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'final_leverage': final_leverage,
                'base_leverage': base_leverage,
                'combined_confidence': combined_confidence,
                'current_price': current_price,
                'account_balance': account_balance
            })
            
            # Maintain history size
            if len(self.leverage_history) > 100:
                self.leverage_history = self.leverage_history[-100:]
            
            self.logger.debug(f"Leverage calculated for {symbol}: {final_leverage:.1f}x (confidence: {combined_confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Leverage calculation failed for {symbol}: {e}")
            raise
    
    def _generate_leverage_reason(self, final_leverage: float, combined_confidence: float, base_leverage: float) -> str:
        """Generate reason for leverage sizing decision."""
        try:
            if combined_confidence < self.leverage_combined_threshold:
                return f'Leverage: {final_leverage:.1f}x (minimum due to low ML confidence {combined_confidence:.3f} below threshold {self.leverage_combined_threshold:.3f})'
            elif final_leverage == self.max_leverage:
                return f'Leverage: {final_leverage:.1f}x (maximum due to high ML confidence {combined_confidence:.3f} * multiplier {self.leverage_multiplier:.2f})'
            elif final_leverage == self.min_leverage:
                return f'Leverage: {final_leverage:.1f}x (minimum due to low ML confidence {combined_confidence:.3f} * multiplier {self.leverage_multiplier:.2f})'
            else:
                return f'Leverage: {final_leverage:.1f}x (ML confidence: {combined_confidence:.3f} * multiplier: {self.leverage_multiplier:.2f} = {base_leverage:.1f}x, capped)'
        except Exception as e:
            self.logger.error(f"❌ Error generating leverage reason: {e}")
            return f'Leverage: {final_leverage:.1f}x (Error generating reason)'
    
    def get_leverage_history(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Get leverage calculation history."""
        if limit:
            return self.leverage_history[-limit:]
        return self.leverage_history.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for leverage management."""
        try:
            if not self.leverage_history:
                return {
                    'total_calculations': 0,
                    'avg_leverage': 0.0,
                    'avg_confidence': 0.0,
                    'min_leverage': self.min_leverage,
                    'max_leverage': self.max_leverage
                }
            
            recent_history = self.leverage_history[-50:]  # Last 50 calculations
            
            avg_leverage = sum(h['final_leverage'] for h in recent_history) / len(recent_history)
            avg_confidence = sum(h['combined_confidence'] for h in recent_history) / len(recent_history)
            
            return {
                'total_calculations': len(self.leverage_history),
                'avg_leverage': avg_leverage,
                'avg_confidence': avg_confidence,
                'min_leverage': self.min_leverage,
                'max_leverage': self.max_leverage,
                'leverage_multiplier': self.leverage_multiplier,
                'leverage_combined_threshold': self.leverage_combined_threshold
            }
            
        except Exception as e:
            self.logger.error(f"❌ Performance metrics calculation failed: {e}")
            return {}
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update leverage configuration."""
        try:
            if 'min_leverage' in new_config:
                self.min_leverage = new_config['min_leverage']
            if 'max_leverage' in new_config:
                self.max_leverage = new_config['max_leverage']
            if 'leverage_multiplier' in new_config:
                self.leverage_multiplier = new_config['leverage_multiplier']
            if 'leverage_combined_threshold' in new_config:
                self.leverage_combined_threshold = new_config['leverage_combined_threshold']
            
            self.logger.info("✅ Leverage configuration updated")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update leverage configuration: {e}")
    
    async def stop(self):
        """Stop leverage manager."""
        try:
            self.logger.info("🛑 Stopping Leverage Manager...")
            self.is_initialized = False
            self.logger.info("✅ Leverage Manager stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping Leverage Manager: {e}")

# Convenience function
async def setup_leverage_manager(config: TradingConfig) -> Optional[LeverageManager]:
    """Setup and initialize leverage manager."""
    try:
        leverage_manager = LeverageManager(config)
        success = await leverage_manager.initialize()
        if success:
            return leverage_manager
        return None
    except Exception as e:
        logger.error(f"❌ Failed to setup leverage manager: {e}")
        return None