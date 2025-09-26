"""
Base Signal Enhancer for TAS/NAS Components

This module provides a unified base class for signal enhancement that can be used
by both TAS and NAS components, eliminating code duplication and ensuring consistent
enhancement patterns.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

# Import shared utilities
from .feature_engineering import UnifiedFeatureEngine, FeatureSet
from .confidence_calculator import UnifiedConfidenceCalculator, ConfidenceMetrics
from .fallback_analyzer import UnifiedFallbackAnalyzer, FallbackAnalysisResult

logger = system_logger.getChild('SignalEnhancerBase')

@dataclass
class EnhancementResult:
    """Container for signal enhancement results."""
    enhanced_signal: Dict[str, Any]
    confidence_metrics: ConfidenceMetrics
    feature_set: FeatureSet
    enhancement_metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class BaseSignalEnhancer(ABC):
    """
    Abstract base class for signal enhancement in both TAS and NAS components.
    
    Provides common functionality for feature extraction, confidence calculation,
    and enhancement logic that can be shared between different signal types.
    """
    
    def __init__(self, enhancement_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base signal enhancer.
        
        Args:
            enhancement_type: Type of enhancement ("nas" or "tas")
            config: Configuration dictionary
        """
        self.enhancement_type = enhancement_type
        self.config = config or {}
        self.logger = logger.getChild(f'{enhancement_type.upper()}SignalEnhancer')
        
        # Initialize shared utilities
        self.feature_engine = UnifiedFeatureEngine(self.config.get('feature_config', {}))
        self.confidence_calculator = UnifiedConfidenceCalculator(self.config.get('confidence_config', {}))
        self.fallback_analyzer = UnifiedFallbackAnalyzer(self.config.get('fallback_config', {}))
        
        # Enhancement-specific configuration
        self.enable_enhancement = self.config.get('enable_enhancement', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.enhancement_models = {}  # Will be loaded by subclasses
        
        # Performance tracking
        self.enhancement_count = 0
        self.successful_enhancements = 0
        self.failed_enhancements = 0
        
    @abstractmethod
    async def _load_enhancement_models(self, models: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load enhancement models (to be implemented by subclasses).
        
        Args:
            models: Pre-trained models for enhancement
            
        Returns:
            bool: True if models loaded successfully
        """
        pass
    
    @abstractmethod
    async def _generate_enhancement_prediction(
        self,
        features: FeatureSet,
        market_data: pd.DataFrame,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate enhancement prediction (to be implemented by subclasses).
        
        Args:
            features: Extracted features
            market_data: Market data
            additional_context: Additional context
            
        Returns:
            Enhancement prediction or None
        """
        pass
    
    @handles_errors
    @traced(span_name="enhance_signal")
    @log_execution_time()
    async def enhance_signal(
        self,
        base_signal: Dict[str, Any],
        market_data: pd.DataFrame,
        regime_data: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> EnhancementResult:
        """
        Enhance a trading signal using the appropriate enhancement method.
        
        Args:
            base_signal: Base signal to enhance
            market_data: Market data for analysis
            regime_data: Current regime information
            additional_context: Additional context for enhancement
            
        Returns:
            EnhancementResult: Enhanced signal with confidence metrics
        """
        try:
            if not self.enable_enhancement:
                tprint_warning(f"⚠️ {self.enhancement_type.upper()} enhancement disabled")
                return self._create_fallback_result(base_signal, "Enhancement disabled")
            
            tprint_info(f"🔄 Enhancing {self.enhancement_type.upper()} signal")
            
            # Extract features
            features = await self.feature_engine.extract_market_features(
                market_data, self.enhancement_type, regime_data, additional_context
            )
            
            # Generate enhancement prediction
            enhancement_prediction = await self._generate_enhancement_prediction(
                features, market_data, additional_context
            )
            
            # Calculate confidence metrics
            base_confidence = base_signal.get('confidence_score', 0.5)
            enhancement_confidence = enhancement_prediction.get('confidence', 0.5) if enhancement_prediction else base_confidence
            
            confidence_metrics = await self.confidence_calculator.calculate_confidence(
                base_confidence=base_confidence,
                enhancement_confidence=enhancement_confidence,
                risk_metrics=base_signal.get('risk_metrics', {}),
                regime_metrics=regime_data,
                signal_type=self.enhancement_type,
                additional_context=additional_context
            )
            
            # Create enhanced signal
            enhanced_signal = self._create_enhanced_signal(
                base_signal, enhancement_prediction, confidence_metrics, features
            )
            
            # Create result
            result = EnhancementResult(
                enhanced_signal=enhanced_signal,
                confidence_metrics=confidence_metrics,
                feature_set=features,
                enhancement_metadata={
                    'enhancement_type': self.enhancement_type,
                    'enhancement_timestamp': datetime.now().isoformat(),
                    'base_confidence': base_confidence,
                    'enhancement_confidence': enhancement_confidence,
                    'final_confidence': confidence_metrics.final_confidence,
                    'feature_count': self._count_total_features(features)
                },
                success=True
            )
            
            self.enhancement_count += 1
            self.successful_enhancements += 1
            
            tprint_success(f"✅ {self.enhancement_type.upper()} signal enhanced (confidence: {confidence_metrics.final_confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {self.enhancement_type.upper()} enhancement failed: {e}")
            self.enhancement_count += 1
            self.failed_enhancements += 1
            
            return self._create_fallback_result(base_signal, str(e))
    
    def _create_enhanced_signal(
        self,
        base_signal: Dict[str, Any],
        enhancement_prediction: Optional[Dict[str, Any]],
        confidence_metrics: ConfidenceMetrics,
        features: FeatureSet
    ) -> Dict[str, Any]:
        """Create enhanced signal from base signal and enhancement."""
        try:
            enhanced_signal = base_signal.copy()
            
            # Update confidence
            enhanced_signal['confidence_score'] = confidence_metrics.final_confidence
            enhanced_signal['enhanced_confidence'] = confidence_metrics.enhanced_confidence
            enhanced_signal['combined_confidence'] = confidence_metrics.combined_confidence
            
            # Add enhancement information
            if enhancement_prediction:
                enhanced_signal[f'{self.enhancement_type}_prediction'] = enhancement_prediction
                enhanced_signal[f'{self.enhancement_type}_confidence'] = enhancement_prediction.get('confidence', 0.0)
                enhanced_signal[f'{self.enhancement_type}_architecture'] = enhancement_prediction.get('architecture', {})
            
            # Add feature information
            enhanced_signal['feature_metrics'] = {
                'price_features': len(features.price_features),
                'volatility_features': len(features.volatility_features),
                'volume_features': len(features.volume_features),
                'technical_features': len(features.technical_features),
                'momentum_features': len(features.momentum_features),
                'regime_features': len(features.regime_features)
            }
            
            # Add confidence components
            enhanced_signal['confidence_components'] = confidence_metrics.confidence_components
            enhanced_signal['risk_factors'] = confidence_metrics.risk_factors
            
            # Add enhancement metadata
            enhanced_signal['enhancement_metadata'] = {
                'enhancement_type': self.enhancement_type,
                'enhancement_timestamp': datetime.now().isoformat(),
                'enhancement_success': True
            }
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced signal creation failed: {e}")
            return base_signal
    
    def _create_fallback_result(
        self,
        base_signal: Dict[str, Any],
        error_message: str
    ) -> EnhancementResult:
        """Create fallback result when enhancement fails."""
        try:
            # Use fallback analysis if available
            fallback_result = self._get_fallback_analysis(base_signal)
            
            # Create fallback confidence
            fallback_confidence = fallback_result.confidence_score if fallback_result else 0.3
            
            # Create minimal confidence metrics
            confidence_metrics = ConfidenceMetrics(
                base_confidence=base_signal.get('confidence_score', 0.5),
                enhanced_confidence=fallback_confidence,
                combined_confidence=fallback_confidence,
                risk_adjusted_confidence=fallback_confidence,
                final_confidence=fallback_confidence,
                confidence_components={
                    'base_confidence': base_signal.get('confidence_score', 0.5),
                    'enhancement_confidence': fallback_confidence,
                    'combined_confidence': fallback_confidence,
                    'risk_adjusted_confidence': fallback_confidence
                },
                risk_factors={},
                metadata={
                    'fallback': True,
                    'error': error_message
                }
            )
            
            # Create fallback signal
            fallback_signal = base_signal.copy()
            fallback_signal['confidence_score'] = fallback_confidence
            fallback_signal['enhancement_metadata'] = {
                'enhancement_type': self.enhancement_type,
                'enhancement_timestamp': datetime.now().isoformat(),
                'enhancement_success': False,
                'fallback_used': True,
                'error_message': error_message
            }
            
            return EnhancementResult(
                enhanced_signal=fallback_signal,
                confidence_metrics=confidence_metrics,
                feature_set=FeatureSet({}, {}, {}, {}, {}, {}, {}),
                enhancement_metadata={
                    'enhancement_type': self.enhancement_type,
                    'fallback': True,
                    'error': error_message
                },
                success=False,
                error_message=error_message
            )
            
        except Exception as e:
            self.logger.error(f"❌ Fallback result creation failed: {e}")
            return EnhancementResult(
                enhanced_signal=base_signal,
                confidence_metrics=ConfidenceMetrics(0.3, 0.3, 0.3, 0.3, 0.3, {}, {}, {}),
                feature_set=FeatureSet({}, {}, {}, {}, {}, {}, {}),
                enhancement_metadata={'error': str(e)},
                success=False,
                error_message=str(e)
            )
    
    def _get_fallback_analysis(self, base_signal: Dict[str, Any]) -> Optional[FallbackAnalysisResult]:
        """Get fallback analysis result."""
        try:
            # This would be implemented to get market data and perform fallback analysis
            # For now, return None to use default fallback
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Fallback analysis failed: {e}")
            return None
    
    def _count_total_features(self, features: FeatureSet) -> int:
        """Count total features in feature set."""
        return (len(features.price_features) + 
                len(features.volatility_features) + 
                len(features.volume_features) + 
                len(features.technical_features) + 
                len(features.momentum_features) + 
                len(features.regime_features))
    
    async def initialize(self, models: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the signal enhancer.
        
        Args:
            models: Pre-trained models for enhancement
            
        Returns:
            bool: True if initialization successful
        """
        try:
            tprint_info(f"🔄 Initializing {self.enhancement_type.upper()} signal enhancer")
            
            # Load enhancement models
            success = await self._load_enhancement_models(models)
            
            if success:
                tprint_success(f"✅ {self.enhancement_type.upper()} signal enhancer initialized")
            else:
                tprint_warning(f"⚠️ {self.enhancement_type.upper()} signal enhancer initialized with fallback")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize {self.enhancement_type.upper()} signal enhancer: {e}")
            return False
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get enhancement performance statistics."""
        return {
            'enhancement_type': self.enhancement_type,
            'total_enhancements': self.enhancement_count,
            'successful_enhancements': self.successful_enhancements,
            'failed_enhancements': self.failed_enhancements,
            'success_rate': self.successful_enhancements / self.enhancement_count if self.enhancement_count > 0 else 0.0,
            'feature_engine_stats': self.feature_engine.get_performance_metrics(),
            'confidence_calculator_stats': self.confidence_calculator.get_performance_metrics(),
            'fallback_analyzer_stats': self.fallback_analyzer.get_performance_metrics()
        }
    
    async def stop(self):
        """Stop the signal enhancer."""
        try:
            self.logger.info(f"🛑 Stopping {self.enhancement_type.upper()} signal enhancer")
            
            # Clear models
            self.enhancement_models.clear()
            
            self.logger.info(f"✅ {self.enhancement_type.upper()} signal enhancer stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping {self.enhancement_type.upper()} signal enhancer: {e}")

# Convenience functions
def create_signal_enhancer(
    enhancement_type: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseSignalEnhancer:
    """
    Factory function to create signal enhancer instances.
    
    Args:
        enhancement_type: Type of enhancement ("nas" or "tas")
        config: Configuration dictionary for the enhancer
        
    Returns:
        BaseSignalEnhancer: Appropriate signal enhancer instance
        
    Raises:
        ValueError: If enhancement_type is not supported
        ImportError: If required enhancer classes are not available
        
    Example:
        >>> nas_enhancer = create_signal_enhancer("nas", {"confidence_threshold": 0.7})
        >>> tas_enhancer = create_signal_enhancer("tas", {"max_model_contributions": 5})
    """
    enhancement_type = enhancement_type.lower().strip()
    
    if enhancement_type == "nas":
        try:
            from src.trading.signal_generation.analyst_signals_refactored import NASSignalEnhancer
            return NASSignalEnhancer(config)
        except ImportError as e:
            logger.error(f"❌ Failed to import NASSignalEnhancer: {e}")
            raise ImportError(
                "NASSignalEnhancer not available. "
                "Please ensure the analyst signals module is properly installed."
            ) from e
    
    elif enhancement_type == "tas":
        try:
            from src.trading.signal_generation.tactician_signals_refactored import TASSignalEnhancer
            return TASSignalEnhancer(config)
        except ImportError as e:
            logger.error(f"❌ Failed to import TASSignalEnhancer: {e}")
            raise ImportError(
                "TASSignalEnhancer not available. "
                "Please ensure the tactician signals module is properly installed."
            ) from e
    
    else:
        raise ValueError(
            f"Unsupported enhancement type: '{enhancement_type}'. "
            f"Supported types are: 'nas', 'tas'"
        )