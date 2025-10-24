"""
Pruner factory for creating optimization pruners.

This module provides a factory for creating different types of pruners
for early stopping and trial pruning.
"""

from typing import Optional, Dict, Any
import logging

from ..validation import PrunerConfig, AresExecutionMode, PrunerStrategy
from ..exceptions import ConfigurationError, PruningError

# Import tprint functions
try:
    from ...tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_performance, tprint_timer, tprint_data_preview,
        tprint_data_format, LogLevel, TPrintConfig
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): pass
    def tprint_warning(*args, **kwargs): pass
    def tprint_success(*args, **kwargs): pass
    def tprint_error(*args, **kwargs): pass
    def tprint_debug(*args, **kwargs): pass
    def tprint_performance(*args, **kwargs): pass
    def tprint_data_preview(*args, **kwargs): pass
    def tprint_data_format(*args, **kwargs): pass


class PrunerFactory:
    """Factory for creating optimization pruners."""
    
    @staticmethod
    def create(config: PrunerConfig) -> Optional[Any]:
        """
        Create a pruner based on configuration.
        
        Args:
            config: Pruner configuration
            
        Returns:
            Configured pruner instance or None
        """
        try:
            if TPRINT_AVAILABLE:
                tprint_info(f"🔧 Creating pruner with strategy: {config.strategy.value}")
                tprint_data_format({
                    'strategy': config.strategy.value,
                    'ares_mode': config.ares_mode.value,
                    'base_patience': config.base_patience,
                    'improvement_threshold': config.improvement_threshold
                }, "pruner_config")
            
            if config.strategy == PrunerStrategy.ADAPTIVE:
                return PrunerFactory._create_adaptive_pruner(config)
            elif config.strategy == PrunerStrategy.CONFIDENCE_BASED:
                return PrunerFactory._create_confidence_pruner(config)
            elif config.strategy == PrunerStrategy.MULTI_FIDELITY:
                return PrunerFactory._create_multi_fidelity_pruner(config)
            elif config.strategy == PrunerStrategy.HYPERBAND:
                return PrunerFactory._create_hyperband_pruner(config)
            elif config.strategy == PrunerStrategy.SUCCESSIVE_HALVING:
                return PrunerFactory._create_successive_halving_pruner(config)
            elif config.strategy == PrunerStrategy.MEDIAN:
                return PrunerFactory._create_median_pruner(config)
            else:
                raise ConfigurationError(f"Unsupported pruner strategy: {config.strategy}")
                
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Failed to create pruner: {e}")
            raise PruningError(f"Failed to create pruner: {e}") from e
    
    @staticmethod
    def _create_adaptive_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create adaptive pruner with Ares mode integration."""
        try:
            from ..enhanced_pruner_system import create_enhanced_pruner
            
            if TPRINT_AVAILABLE:
                tprint_info(f"🔧 Creating adaptive pruner for Ares mode: {config.ares_mode.value}")
            
            # Map Ares execution mode to pruner settings
            mode_settings = {
                AresExecutionMode.LIGHT: {
                    'base_patience': max(1, config.base_patience // 2),
                    'improvement_threshold': config.improvement_threshold * 2,
                    'enable_aggressive_pruning': True
                },
                AresExecutionMode.BLANK: {
                    'base_patience': max(2, config.base_patience // 2),
                    'improvement_threshold': config.improvement_threshold * 1.5,
                    'enable_aggressive_pruning': True
                },
                AresExecutionMode.FULL: {
                    'base_patience': config.base_patience,
                    'improvement_threshold': config.improvement_threshold,
                    'enable_aggressive_pruning': config.enable_aggressive_pruning
                }
            }
            
            settings = mode_settings.get(config.ares_mode, mode_settings[AresExecutionMode.FULL])
            
            if TPRINT_AVAILABLE:
                tprint_data_format(settings, f"adaptive_pruner_settings_{config.ares_mode.value}")
            
            pruner = create_enhanced_pruner(
                ares_mode=config.ares_mode.value,
                strategy='adaptive',
                base_patience=settings['base_patience'],
                improvement_threshold=settings['improvement_threshold'],
                enable_aggressive_pruning=settings['enable_aggressive_pruning']
            )
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Adaptive pruner created successfully")
            
            return pruner
            
        except ImportError:
            # Fallback to median pruner if enhanced pruner not available
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ Enhanced pruner not available, falling back to median pruner")
            return PrunerFactory._create_median_pruner(config)
    
    @staticmethod
    def _create_confidence_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create confidence-based pruner."""
        try:
            from ..enhanced_pruner_system import create_enhanced_pruner
            
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Creating confidence-based pruner")
            
            pruner = create_enhanced_pruner(
                ares_mode=config.ares_mode.value,
                strategy='confidence_based',
                base_patience=config.base_patience,
                improvement_threshold=config.improvement_threshold
            )
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Confidence-based pruner created successfully")
            
            return pruner
            
        except ImportError:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ Enhanced pruner not available, falling back to median pruner")
            return PrunerFactory._create_median_pruner(config)
    
    @staticmethod
    def _create_multi_fidelity_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create multi-fidelity pruner."""
        try:
            from ..enhanced_pruner_system import create_enhanced_pruner
            
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Creating multi-fidelity pruner")
                tprint_data_format({
                    'min_resource': config.min_resource,
                    'max_resource': config.max_resource,
                    'reduction_factor': config.reduction_factor
                }, "multi_fidelity_pruner_config")
            
            pruner = create_enhanced_pruner(
                ares_mode=config.ares_mode.value,
                strategy='multi_fidelity',
                base_patience=config.base_patience,
                improvement_threshold=config.improvement_threshold,
                min_resource=config.min_resource,
                max_resource=config.max_resource,
                reduction_factor=config.reduction_factor
            )
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Multi-fidelity pruner created successfully")
            
            return pruner
            
        except ImportError:
            # Fallback to SuccessiveHalvingPruner
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ Enhanced pruner not available, falling back to successive halving pruner")
            return PrunerFactory._create_successive_halving_pruner(config)
    
    @staticmethod
    def _create_hyperband_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create Hyperband pruner."""
        try:
            import optuna
            from optuna.pruners import HyperbandPruner
            
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Creating Hyperband pruner")
                tprint_data_format({
                    'min_resource': config.min_resource,
                    'max_resource': config.max_resource,
                    'reduction_factor': config.reduction_factor
                }, "hyperband_pruner_config")
            
            pruner = HyperbandPruner(
                min_resource=config.min_resource,
                max_resource=config.max_resource,
                reduction_factor=config.reduction_factor
            )
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Hyperband pruner created successfully")
            
            return pruner
            
        except ImportError:
            if TPRINT_AVAILABLE:
                tprint_error("❌ Optuna is required for Hyperband pruner")
            raise PruningError("Optuna is required for Hyperband pruner")
    
    @staticmethod
    def _create_successive_halving_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create Successive Halving pruner."""
        try:
            import optuna
            from optuna.pruners import SuccessiveHalvingPruner
            
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Creating Successive Halving pruner")
                tprint_data_format({
                    'min_resource': config.min_resource,
                    'reduction_factor': int(config.reduction_factor)
                }, "successive_halving_pruner_config")
            
            pruner = SuccessiveHalvingPruner(
                min_resource=config.min_resource,
                reduction_factor=int(config.reduction_factor)
            )
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Successive Halving pruner created successfully")
            
            return pruner
            
        except ImportError:
            if TPRINT_AVAILABLE:
                tprint_error("❌ Optuna is required for Successive Halving pruner")
            raise PruningError("Optuna is required for Successive Halving pruner")
    
    @staticmethod
    def _create_median_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create Median pruner (fallback)."""
        try:
            import optuna
            from optuna.pruners import MedianPruner
            
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Creating Median pruner (fallback)")
                tprint_data_format({
                    'n_startup_trials': max(1, config.base_patience // 2),
                    'n_warmup_steps': config.base_patience
                }, "median_pruner_config")
            
            pruner = MedianPruner(
                n_startup_trials=max(1, config.base_patience // 2),
                n_warmup_steps=config.base_patience
            )
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Median pruner created successfully")
            
            return pruner
            
        except ImportError:
            # Return None if Optuna not available
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ Optuna not available, returning None for median pruner")
            return None
    
    @staticmethod
    def create_auto_mode_pruner() -> Optional[Any]:
        """Create pruner with automatic Ares mode detection."""
        try:
            from ..enhanced_pruner_system import get_ares_mode_from_context, create_auto_mode_pruner
            
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Creating auto-mode pruner with Ares mode detection")
            
            pruner = create_auto_mode_pruner()
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Auto-mode pruner created successfully")
            
            return pruner
        except ImportError:
            # Fallback to basic median pruner
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ Enhanced pruner not available, falling back to basic median pruner")
            config = PrunerConfig()
            return PrunerFactory._create_median_pruner(config)