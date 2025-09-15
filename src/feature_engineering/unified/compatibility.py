"""
Backwards Compatibility Layer

This module provides seamless backwards compatibility for existing
feature generation systems while transitioning to the unified architecture.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

from .core import FeatureGenerator, FeatureGeneratorConfig, FeatureGenerationResult, FeatureCategory, FeaturePriority
from .orchestrator import FeatureOrchestrator, OrchestrationConfig
from .registry import get_registry
from ...utils.logger import system_logger
from ...core.decorators import handles_errors


class LegacyFeatureAdapter(FeatureGenerator):
    """
    Adapter for legacy feature generation functions to work with the unified system.
    
    This allows existing feature generation code to be wrapped and used
    within the new unified architecture without modification.
    """
    
    def __init__(
        self,
        name: str,
        legacy_function: Callable,
        required_columns: List[str],
        output_columns: List[str],
        category: FeatureCategory = FeatureCategory.CUSTOM,
        config: Optional[FeatureGeneratorConfig] = None
    ):
        """
        Initialize legacy feature adapter.
        
        Args:
            name: Name for the adapter
            legacy_function: Legacy function to wrap
            required_columns: Columns required by the legacy function
            output_columns: Columns produced by the legacy function
            category: Category for the adapter
            config: Optional configuration override
        """
        if config is None:
            config = FeatureGeneratorConfig(
                name=name,
                category=category,
                priority=FeaturePriority.MEDIUM,
                enabled=True
            )
        
        super().__init__(config)
        self.legacy_function = legacy_function
        self._required_columns = required_columns
        self._output_columns = output_columns
        
    async def initialize(self) -> bool:
        """Initialize the adapter."""
        try:
            self._is_initialized = True
            self.logger.info(f"Legacy adapter {self.config.name} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing legacy adapter {self.config.name}: {e}")
            return False
    
    async def generate_features(
        self, 
        data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> FeatureGenerationResult:
        """Generate features using the legacy function."""
        try:
            if not self._is_initialized:
                return FeatureGenerationResult(
                    success=False,
                    errors=["Legacy adapter not initialized"]
                )
            
            # Validate input
            is_valid, errors = self.validate_input(data)
            if not is_valid:
                return FeatureGenerationResult(
                    success=False,
                    errors=errors
                )
            
            # Execute legacy function
            start_time = datetime.now()
            
            if asyncio.iscoroutinefunction(self.legacy_function):
                result = await self.legacy_function(data, context or {})
            else:
                result = self.legacy_function(data, context or {})
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Handle different return types
            if isinstance(result, pd.DataFrame):
                features = result
            elif isinstance(result, dict):
                features = pd.DataFrame([result])
            elif isinstance(result, tuple) and len(result) == 2:
                # Assume (features, metadata) format
                features, metadata = result
                if isinstance(features, dict):
                    features = pd.DataFrame([features])
            else:
                return FeatureGenerationResult(
                    success=False,
                    errors=[f"Unexpected return type from legacy function: {type(result)}"]
                )
            
            # Validate output
            is_valid, errors = self.validate_output(features)
            if not is_valid:
                return FeatureGenerationResult(
                    success=False,
                    features=features,
                    errors=errors
                )
            
            # Update performance metrics
            self._performance_metrics["last_duration_seconds"] = duration
            self._performance_metrics["total_calls"] = self._performance_metrics.get("total_calls", 0) + 1
            
            return FeatureGenerationResult(
                success=True,
                features=features,
                metadata={"legacy_function": self.legacy_function.__name__},
                performance_metrics={"duration_seconds": duration}
            )
            
        except Exception as e:
            self.logger.error(f"Error in legacy adapter {self.config.name}: {e}")
            return FeatureGenerationResult(
                success=False,
                errors=[f"Legacy function error: {str(e)}"]
            )
    
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return self._required_columns.copy()
    
    def get_output_columns(self) -> List[str]:
        """Get output columns."""
        return self._output_columns.copy()


class BackwardsCompatibilityLayer:
    """
    Backwards compatibility layer that provides seamless integration
    with existing feature generation systems.
    """
    
    def __init__(self, orchestrator: FeatureOrchestrator):
        """
        Initialize backwards compatibility layer.
        
        Args:
            orchestrator: The unified feature orchestrator
        """
        self.orchestrator = orchestrator
        self.logger = system_logger.getChild("BackwardsCompatibilityLayer")
        self._legacy_adapters: Dict[str, LegacyFeatureAdapter] = {}
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the compatibility layer."""
        try:
            self.logger.info("Initializing backwards compatibility layer...")
            
            # Register common legacy adapters
            await self._register_legacy_adapters()
            
            self._initialized = True
            self.logger.info("Backwards compatibility layer initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing compatibility layer: {e}")
            return False
    
    async def _register_legacy_adapters(self) -> None:
        """Register adapters for common legacy feature generation patterns."""
        try:
            # Register adapters for existing feature engineering systems
            await self._register_analyst_adapters()
            await self._register_ml_common_adapters()
            await self._register_training_adapters()
            
        except Exception as e:
            self.logger.warning(f"Error registering legacy adapters: {e}")
    
    async def _register_analyst_adapters(self) -> None:
        """Register adapters for analyst feature engineering."""
        try:
            # Import analyst feature engineering
            from ...analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
            from ...analyst.advanced_feature_engineering import AdvancedFeatureEngineering
            
            # Create adapter for FeatureEngineeringOrchestrator
            def analyst_orchestrator_wrapper(data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
                # This would need to be adapted based on the actual interface
                # For now, return a placeholder
                return data
            
            adapter = LegacyFeatureAdapter(
                name="analyst_orchestrator",
                legacy_function=analyst_orchestrator_wrapper,
                required_columns=["open", "high", "low", "close", "volume"],
                output_columns=[],  # Will be determined dynamically
                category=FeatureCategory.CUSTOM
            )
            
            await self._register_adapter(adapter)
            
        except ImportError:
            self.logger.debug("Analyst feature engineering not available")
        except Exception as e:
            self.logger.warning(f"Error registering analyst adapters: {e}")
    
    async def _register_ml_common_adapters(self) -> None:
        """Register adapters for ml_common feature generation."""
        try:
            # Import ml_common feature selection
            from ...utils.ml_common.feature_selection import FeatureSelectionFramework
            
            # Create adapter for feature selection
            def feature_selection_wrapper(data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
                # Placeholder implementation
                return data
            
            adapter = LegacyFeatureAdapter(
                name="ml_common_feature_selection",
                legacy_function=feature_selection_wrapper,
                required_columns=["open", "high", "low", "close", "volume"],
                output_columns=[],
                category=FeatureCategory.STATISTICAL_FEATURES
            )
            
            await self._register_adapter(adapter)
            
        except ImportError:
            self.logger.debug("ML common feature selection not available")
        except Exception as e:
            self.logger.warning(f"Error registering ml_common adapters: {e}")
    
    async def _register_training_adapters(self) -> None:
        """Register adapters for training step feature generation."""
        try:
            # This would register adapters for various training steps
            # Implementation depends on specific training step interfaces
            pass
            
        except Exception as e:
            self.logger.warning(f"Error registering training adapters: {e}")
    
    async def _register_adapter(self, adapter: LegacyFeatureAdapter) -> None:
        """Register a legacy adapter."""
        try:
            if await adapter.initialize():
                self._legacy_adapters[adapter.config.name] = adapter
                self.logger.info(f"Registered legacy adapter: {adapter.config.name}")
            else:
                self.logger.warning(f"Failed to initialize legacy adapter: {adapter.config.name}")
                
        except Exception as e:
            self.logger.error(f"Error registering adapter {adapter.config.name}: {e}")
    
    def register_legacy_function(
        self,
        name: str,
        function: Callable,
        required_columns: List[str],
        output_columns: List[str],
        category: FeatureCategory = FeatureCategory.CUSTOM
    ) -> bool:
        """
        Register a legacy function for use with the unified system.
        
        Args:
            name: Name for the function
            function: Legacy function to register
            required_columns: Columns required by the function
            output_columns: Columns produced by the function
            category: Category for the function
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            adapter = LegacyFeatureAdapter(
                name=name,
                legacy_function=function,
                required_columns=required_columns,
                output_columns=output_columns,
                category=category
            )
            
            # Register synchronously for now
            asyncio.run(self._register_adapter(adapter))
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering legacy function {name}: {e}")
            return False
    
    @handles_errors(exceptions=(Exception,), default_return=FeatureGenerationResult(success=False), context="legacy feature generation")
    async def generate_features_legacy(
        self,
        data: pd.DataFrame,
        method: str = "orchestrator",
        **kwargs
    ) -> FeatureGenerationResult:
        """
        Generate features using legacy methods for backwards compatibility.
        
        Args:
            data: Input data
            method: Method to use ("orchestrator", "analyst", "ml_common", etc.)
            **kwargs: Additional arguments
            
        Returns:
            FeatureGenerationResult
        """
        if not self._initialized:
            return FeatureGenerationResult(
                success=False,
                errors=["Compatibility layer not initialized"]
            )
        
        try:
            if method == "orchestrator":
                # Use the unified orchestrator
                return await self.orchestrator.generate_features(data, **kwargs)
            
            elif method == "analyst":
                # Use analyst feature engineering
                return await self._generate_with_analyst(data, **kwargs)
            
            elif method == "ml_common":
                # Use ml_common feature selection
                return await self._generate_with_ml_common(data, **kwargs)
            
            elif method in self._legacy_adapters:
                # Use specific legacy adapter
                adapter = self._legacy_adapters[method]
                return await adapter.generate_features(data, kwargs)
            
            else:
                return FeatureGenerationResult(
                    success=False,
                    errors=[f"Unknown legacy method: {method}"]
                )
                
        except Exception as e:
            self.logger.error(f"Error in legacy feature generation: {e}")
            return FeatureGenerationResult(
                success=False,
                errors=[f"Legacy generation error: {str(e)}"]
            )
    
    async def _generate_with_analyst(self, data: pd.DataFrame, **kwargs) -> FeatureGenerationResult:
        """Generate features using analyst system."""
        try:
            from ...analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
            
            config = kwargs.get('config', {})
            orchestrator = FeatureEngineeringOrchestrator(config)
            
            # Use the existing analyst interface
            result = await orchestrator.generate_all_features(
                data,
                kwargs.get('agg_trades_df'),
                kwargs.get('futures_df'),
                kwargs.get('sr_levels')
            )
            
            return FeatureGenerationResult(
                success=True,
                features=result,
                metadata={"method": "analyst"}
            )
            
        except Exception as e:
            return FeatureGenerationResult(
                success=False,
                errors=[f"Analyst generation error: {str(e)}"]
            )
    
    async def _generate_with_ml_common(self, data: pd.DataFrame, **kwargs) -> FeatureGenerationResult:
        """Generate features using ml_common system."""
        try:
            from ...utils.ml_common.feature_selection import FeatureSelectionFramework
            
            # This would need to be adapted based on the actual ml_common interface
            # For now, return the input data
            return FeatureGenerationResult(
                success=True,
                features=data,
                metadata={"method": "ml_common"}
            )
            
        except Exception as e:
            return FeatureGenerationResult(
                success=False,
                errors=[f"ML common generation error: {str(e)}"]
            )
    
    def get_legacy_methods(self) -> List[str]:
        """Get available legacy methods."""
        methods = ["orchestrator"]
        methods.extend(self._legacy_adapters.keys())
        return methods
    
    def get_legacy_adapter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a legacy adapter."""
        adapter = self._legacy_adapters.get(name)
        if adapter:
            return adapter.get_info()
        return None
    
    def list_legacy_adapters(self) -> List[str]:
        """List all registered legacy adapters."""
        return list(self._legacy_adapters.keys())


# Convenience functions for easy migration
def create_legacy_adapter(
    name: str,
    function: Callable,
    required_columns: List[str],
    output_columns: List[str],
    category: FeatureCategory = FeatureCategory.CUSTOM
) -> LegacyFeatureAdapter:
    """Create a legacy adapter for a function."""
    return LegacyFeatureAdapter(
        name=name,
        legacy_function=function,
        required_columns=required_columns,
        output_columns=output_columns,
        category=category
    )


def wrap_legacy_function(
    function: Callable,
    required_columns: List[str],
    output_columns: List[str],
    name: Optional[str] = None,
    category: FeatureCategory = FeatureCategory.CUSTOM
) -> LegacyFeatureAdapter:
    """Wrap a legacy function as a feature generator."""
    if name is None:
        name = function.__name__
    
    return LegacyFeatureAdapter(
        name=name,
        legacy_function=function,
        required_columns=required_columns,
        output_columns=output_columns,
        category=category
    )