"""
Pruner factory for creating optimization pruners.

This module provides a factory for creating different types of pruners
for early stopping and trial pruning.
"""

from typing import Optional, Dict, Any
import logging

from ..validation import PrunerConfig, AresExecutionMode, PrunerStrategy
from ..exceptions import ConfigurationError, PruningError


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
            raise PruningError(f"Failed to create pruner: {e}") from e
    
    @staticmethod
    def _create_adaptive_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create adaptive pruner with Ares mode integration."""
        try:
            from ..enhanced_pruner_system import create_enhanced_pruner
            
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
            
            return create_enhanced_pruner(
                ares_mode=config.ares_mode.value,
                strategy='adaptive',
                base_patience=settings['base_patience'],
                improvement_threshold=settings['improvement_threshold'],
                enable_aggressive_pruning=settings['enable_aggressive_pruning']
            )
            
        except ImportError:
            # Fallback to median pruner if enhanced pruner not available
            return PrunerFactory._create_median_pruner(config)
    
    @staticmethod
    def _create_confidence_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create confidence-based pruner."""
        try:
            from ..enhanced_pruner_system import create_enhanced_pruner
            
            return create_enhanced_pruner(
                ares_mode=config.ares_mode.value,
                strategy='confidence_based',
                base_patience=config.base_patience,
                improvement_threshold=config.improvement_threshold
            )
            
        except ImportError:
            return PrunerFactory._create_median_pruner(config)
    
    @staticmethod
    def _create_multi_fidelity_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create multi-fidelity pruner."""
        try:
            from ..enhanced_pruner_system import create_enhanced_pruner
            
            return create_enhanced_pruner(
                ares_mode=config.ares_mode.value,
                strategy='multi_fidelity',
                base_patience=config.base_patience,
                improvement_threshold=config.improvement_threshold,
                min_resource=config.min_resource,
                max_resource=config.max_resource,
                reduction_factor=config.reduction_factor
            )
            
        except ImportError:
            # Fallback to SuccessiveHalvingPruner
            return PrunerFactory._create_successive_halving_pruner(config)
    
    @staticmethod
    def _create_hyperband_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create Hyperband pruner."""
        try:
            import optuna
            from optuna.pruners import HyperbandPruner
            
            return HyperbandPruner(
                min_resource=config.min_resource,
                max_resource=config.max_resource,
                reduction_factor=config.reduction_factor
            )
            
        except ImportError:
            raise PruningError("Optuna is required for Hyperband pruner")
    
    @staticmethod
    def _create_successive_halving_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create Successive Halving pruner."""
        try:
            import optuna
            from optuna.pruners import SuccessiveHalvingPruner
            
            return SuccessiveHalvingPruner(
                min_resource=config.min_resource,
                reduction_factor=int(config.reduction_factor)
            )
            
        except ImportError:
            raise PruningError("Optuna is required for Successive Halving pruner")
    
    @staticmethod
    def _create_median_pruner(config: PrunerConfig) -> Optional[Any]:
        """Create Median pruner (fallback)."""
        try:
            import optuna
            from optuna.pruners import MedianPruner
            
            return MedianPruner(
                n_startup_trials=max(1, config.base_patience // 2),
                n_warmup_steps=config.base_patience
            )
            
        except ImportError:
            # Return None if Optuna not available
            return None
    
    @staticmethod
    def create_auto_mode_pruner() -> Optional[Any]:
        """Create pruner with automatic Ares mode detection."""
        try:
            from ..enhanced_pruner_system import get_ares_mode_from_context, create_auto_mode_pruner
            return create_auto_mode_pruner()
        except ImportError:
            # Fallback to basic median pruner
            config = PrunerConfig()
            return PrunerFactory._create_median_pruner(config)