"""
Comprehensive S/R Detection Integration Module

This module integrates all S/R detection components to provide a unified interface
for the step02_5_sr_optimization pipeline step.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Tuple
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import all S/R detection modules
from .sr_strength_optimizer import SRStrengthOptimizer
from .enhanced_sr_optimization import EnhancedSROptimizer
from .enhanced_sr_confluence import EnhancedSRConfluenceDetector
from .enhanced_sr_validation import EnhancedSRValidator
from .enhanced_sr_detection import EnhancedSRDetector
from .sr_breakout_predictor import SRBreakoutPredictor
from .sr_context_aware_calculator import SRContextAwareCalculator
from .sr_data_integration import SRDataIntegration
from .sr_ensemble_predictor import SREnsemblePredictor
from .sr_parameter_optimizer import SRParameterOptimizer
from .sr_performance_monitor import SRPerformanceMonitor
from .sr_weight_optimizer import SRWeightOptimizer
from .sr_levels_manager import SRLevelsManager

from src.utils.logger import system_logger

logger = system_logger.getChild('SRComprehensiveIntegration')


class SRComprehensiveIntegration:
    """
    Comprehensive S/R Detection Integration
    
    This class provides a unified interface for all S/R detection components,
    ensuring proper data flow and integration with the rest of the project.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize comprehensive S/R integration."""
        self.config = config
        self.logger = logger
        
        # Initialize all S/R components
        self.components = {
            'strength_optimizer': None,
            'enhanced_optimizer': None,
            'confluence_detector': None,
            'validator': None,
            'detector': None,
            'breakout_predictor': None,
            'context_calculator': None,
            'data_integration': None,
            'ensemble_predictor': None,
            'parameter_optimizer': None,
            'performance_monitor': None,
            'weight_optimizer': None,
            'levels_manager': None
        }
        
        # Component initialization status
        self.initialization_status = {}
        
        # Cached results
        self.cached_results = {}
        self.last_update = None
        
    async def initialize(self) -> bool:
        """Initialize all S/R detection components."""
        try:
            self.logger.info("🚀 Initializing comprehensive S/R detection system...")
            
            # Initialize components in order of dependency
            initialization_order = [
                ('detector', self._init_detector),
                ('strength_optimizer', self._init_strength_optimizer),
                ('breakout_predictor', self._init_breakout_predictor),
                ('context_calculator', self._init_context_calculator),
                ('confluence_detector', self._init_confluence_detector),
                ('validator', self._init_validator),
                ('enhanced_optimizer', self._init_enhanced_optimizer),
                ('data_integration', self._init_data_integration),
                ('ensemble_predictor', self._init_ensemble_predictor),
                ('parameter_optimizer', self._init_parameter_optimizer),
                ('performance_monitor', self._init_performance_monitor),
                ('weight_optimizer', self._init_weight_optimizer),
                ('levels_manager', self._init_levels_manager)
            ]
            
            for component_name, init_func in initialization_order:
                try:
                    success = await init_func()
                    self.initialization_status[component_name] = success
                    if success:
                        self.logger.info(f"✅ {component_name} initialized successfully")
                    else:
                        self.logger.warning(f"⚠️ {component_name} initialization failed")
                except Exception as e:
                    self.logger.error(f"❌ Error initializing {component_name}: {e}")
                    self.initialization_status[component_name] = False
            
            # Check if critical components are initialized
            critical_components = ['detector', 'strength_optimizer', 'breakout_predictor']
            critical_success = all(self.initialization_status.get(comp, False) 
                                 for comp in critical_components)
            
            if not critical_success:
                self.logger.error("❌ Critical S/R components failed to initialize")
                return False
            
            self.logger.info("✅ S/R detection system initialization completed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize S/R detection system: {e}")
            return False
    
    async def _init_detector(self) -> bool:
        """Initialize enhanced S/R detector."""
        try:
            self.components['detector'] = EnhancedSRDetector(self.config)
            if hasattr(self.components['detector'], 'initialize'):
                return await self.components['detector'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {e}")
            return False
    
    async def _init_strength_optimizer(self) -> bool:
        """Initialize S/R strength optimizer."""
        try:
            self.components['strength_optimizer'] = SRStrengthOptimizer(self.config)
            if hasattr(self.components['strength_optimizer'], 'initialize'):
                return await self.components['strength_optimizer'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize strength optimizer: {e}")
            return False
    
    async def _init_breakout_predictor(self) -> bool:
        """Initialize S/R breakout predictor."""
        try:
            self.components['breakout_predictor'] = SRBreakoutPredictor(self.config)
            if hasattr(self.components['breakout_predictor'], 'initialize'):
                return await self.components['breakout_predictor'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize breakout predictor: {e}")
            return False
    
    async def _init_context_calculator(self) -> bool:
        """Initialize context-aware S/R calculator."""
        try:
            self.components['context_calculator'] = SRContextAwareCalculator(self.config)
            if hasattr(self.components['context_calculator'], 'initialize'):
                return await self.components['context_calculator'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize context calculator: {e}")
            return False
    
    async def _init_confluence_detector(self) -> bool:
        """Initialize S/R confluence detector."""
        try:
            self.components['confluence_detector'] = EnhancedSRConfluenceDetector(self.config)
            if hasattr(self.components['confluence_detector'], 'initialize'):
                return await self.components['confluence_detector'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize confluence detector: {e}")
            return False
    
    async def _init_validator(self) -> bool:
        """Initialize S/R validator."""
        try:
            self.components['validator'] = EnhancedSRValidator(self.config)
            if hasattr(self.components['validator'], 'initialize'):
                return await self.components['validator'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize validator: {e}")
            return False
    
    async def _init_enhanced_optimizer(self) -> bool:
        """Initialize enhanced S/R optimizer."""
        try:
            self.components['enhanced_optimizer'] = EnhancedSROptimizer(self.config)
            if hasattr(self.components['enhanced_optimizer'], 'initialize'):
                return await self.components['enhanced_optimizer'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced optimizer: {e}")
            return False
    
    async def _init_data_integration(self) -> bool:
        """Initialize S/R data integration."""
        try:
            self.components['data_integration'] = SRDataIntegration(self.config)
            if hasattr(self.components['data_integration'], 'initialize'):
                return await self.components['data_integration'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize data integration: {e}")
            return False
    
    async def _init_ensemble_predictor(self) -> bool:
        """Initialize S/R ensemble predictor."""
        try:
            self.components['ensemble_predictor'] = SREnsemblePredictor(self.config)
            if hasattr(self.components['ensemble_predictor'], 'initialize'):
                return await self.components['ensemble_predictor'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize ensemble predictor: {e}")
            return False
    
    async def _init_parameter_optimizer(self) -> bool:
        """Initialize S/R parameter optimizer."""
        try:
            self.components['parameter_optimizer'] = SRParameterOptimizer(self.config)
            if hasattr(self.components['parameter_optimizer'], 'initialize'):
                return await self.components['parameter_optimizer'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize parameter optimizer: {e}")
            return False
    
    async def _init_performance_monitor(self) -> bool:
        """Initialize S/R performance monitor."""
        try:
            self.components['performance_monitor'] = SRPerformanceMonitor(self.config)
            if hasattr(self.components['performance_monitor'], 'initialize'):
                return await self.components['performance_monitor'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize performance monitor: {e}")
            return False
    
    async def _init_weight_optimizer(self) -> bool:
        """Initialize S/R weight optimizer."""
        try:
            self.components['weight_optimizer'] = SRWeightOptimizer(self.config)
            if hasattr(self.components['weight_optimizer'], 'initialize'):
                return await self.components['weight_optimizer'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize weight optimizer: {e}")
            return False
    
    async def _init_levels_manager(self) -> bool:
        """Initialize S/R levels manager."""
        try:
            self.components['levels_manager'] = SRLevelsManager(self.config)
            if hasattr(self.components['levels_manager'], 'initialize'):
                return await self.components['levels_manager'].initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize levels manager: {e}")
            return False
    
    async def detect_sr_levels(self, market_data: pd.DataFrame, 
                              timeframe: str = '1m',
                              use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Detect S/R levels using all available components.
        
        Args:
            market_data: Historical market data
            timeframe: Data timeframe
            use_ensemble: Whether to use ensemble methods
            
        Returns:
            Dictionary containing comprehensive S/R analysis
        """
        try:
            self.logger.info(f"🔍 Detecting S/R levels for {len(market_data)} data points")
            
            results = {
                'support_levels': [],
                'resistance_levels': [],
                'confluence_zones': [],
                'breakout_predictions': {},
                'context_analysis': {},
                'validation_results': {},
                'performance_metrics': {},
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'data_points': len(market_data),
                    'timeframe': timeframe,
                    'components_used': []
                }
            }
            
            # Step 1: Basic S/R detection
            if self.components['detector']:
                try:
                    detection_result = await self.components['detector'].detect_levels(
                        market_data, timeframe
                    )
                    results['support_levels'].extend(detection_result.get('support_levels', []))
                    results['resistance_levels'].extend(detection_result.get('resistance_levels', []))
                    results['metadata']['components_used'].append('detector')
                except Exception as e:
                    self.logger.error(f"Detector failed: {e}")
            
            # Step 2: Strength optimization
            if self.components['strength_optimizer'] and results['support_levels']:
                try:
                    optimized_levels = await self.components['strength_optimizer'].optimize_strength(
                        results['support_levels'] + results['resistance_levels'],
                        market_data
                    )
                    # Update levels with optimized strength values
                    results['metadata']['components_used'].append('strength_optimizer')
                except Exception as e:
                    self.logger.error(f"Strength optimizer failed: {e}")
            
            # Step 3: Context-aware calculations
            if self.components['context_calculator']:
                try:
                    context_result = await self.components['context_calculator'].calculate_context(
                        market_data,
                        results['support_levels'],
                        results['resistance_levels']
                    )
                    results['context_analysis'] = context_result
                    results['metadata']['components_used'].append('context_calculator')
                except Exception as e:
                    self.logger.error(f"Context calculator failed: {e}")
            
            # Step 4: Confluence detection
            if self.components['confluence_detector']:
                try:
                    confluence_result = await self.components['confluence_detector'].detect_confluence(
                        results['support_levels'],
                        results['resistance_levels'],
                        market_data
                    )
                    results['confluence_zones'] = confluence_result.get('zones', [])
                    results['metadata']['components_used'].append('confluence_detector')
                except Exception as e:
                    self.logger.error(f"Confluence detector failed: {e}")
            
            # Step 5: Breakout predictions
            if self.components['breakout_predictor']:
                try:
                    breakout_result = await self.components['breakout_predictor'].predict_breakouts(
                        market_data,
                        results['support_levels'],
                        results['resistance_levels']
                    )
                    results['breakout_predictions'] = breakout_result
                    results['metadata']['components_used'].append('breakout_predictor')
                except Exception as e:
                    self.logger.error(f"Breakout predictor failed: {e}")
            
            # Step 6: Ensemble prediction (if enabled)
            if use_ensemble and self.components['ensemble_predictor']:
                try:
                    ensemble_result = await self.components['ensemble_predictor'].predict(
                        market_data,
                        results
                    )
                    # Merge ensemble results
                    results['ensemble_predictions'] = ensemble_result
                    results['metadata']['components_used'].append('ensemble_predictor')
                except Exception as e:
                    self.logger.error(f"Ensemble predictor failed: {e}")
            
            # Step 7: Validation
            if self.components['validator']:
                try:
                    validation_result = await self.components['validator'].validate_levels(
                        results['support_levels'],
                        results['resistance_levels'],
                        market_data
                    )
                    results['validation_results'] = validation_result
                    results['metadata']['components_used'].append('validator')
                except Exception as e:
                    self.logger.error(f"Validator failed: {e}")
            
            # Step 8: Performance monitoring
            if self.components['performance_monitor']:
                try:
                    performance_metrics = await self.components['performance_monitor'].analyze_performance(
                        results,
                        market_data
                    )
                    results['performance_metrics'] = performance_metrics
                    results['metadata']['components_used'].append('performance_monitor')
                except Exception as e:
                    self.logger.error(f"Performance monitor failed: {e}")
            
            # Cache results
            self.cached_results = results
            self.last_update = datetime.now()
            
            self.logger.info(
                f"✅ S/R detection completed: "
                f"{len(results['support_levels'])} support, "
                f"{len(results['resistance_levels'])} resistance levels found"
            )
            
            return results
            
        except Exception as e:
            self.logger.exception(f"❌ S/R detection failed: {e}")
            return {
                'error': str(e),
                'support_levels': [],
                'resistance_levels': [],
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'error': True
                }
            }
    
    async def optimize_parameters(self, market_data: pd.DataFrame,
                                validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Optimize S/R detection parameters using all available optimizers.
        
        Args:
            market_data: Training data for optimization
            validation_data: Optional validation data
            
        Returns:
            Dictionary containing optimized parameters
        """
        try:
            self.logger.info("🔧 Optimizing S/R detection parameters...")
            
            optimized_params = {
                'detection_params': {},
                'strength_params': {},
                'confluence_params': {},
                'breakout_params': {},
                'weight_params': {},
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'data_points': len(market_data),
                    'optimizers_used': []
                }
            }
            
            # Parameter optimization
            if self.components['parameter_optimizer']:
                try:
                    param_result = await self.components['parameter_optimizer'].optimize(
                        market_data,
                        validation_data
                    )
                    optimized_params.update(param_result)
                    optimized_params['metadata']['optimizers_used'].append('parameter_optimizer')
                except Exception as e:
                    self.logger.error(f"Parameter optimizer failed: {e}")
            
            # Weight optimization
            if self.components['weight_optimizer']:
                try:
                    weight_result = await self.components['weight_optimizer'].optimize_weights(
                        market_data,
                        validation_data
                    )
                    optimized_params['weight_params'] = weight_result
                    optimized_params['metadata']['optimizers_used'].append('weight_optimizer')
                except Exception as e:
                    self.logger.error(f"Weight optimizer failed: {e}")
            
            # Enhanced optimization
            if self.components['enhanced_optimizer']:
                try:
                    enhanced_result = await self.components['enhanced_optimizer'].optimize(
                        market_data,
                        current_params=optimized_params
                    )
                    optimized_params.update(enhanced_result)
                    optimized_params['metadata']['optimizers_used'].append('enhanced_optimizer')
                except Exception as e:
                    self.logger.error(f"Enhanced optimizer failed: {e}")
            
            self.logger.info("✅ Parameter optimization completed")
            return optimized_params
            
        except Exception as e:
            self.logger.exception(f"❌ Parameter optimization failed: {e}")
            return {
                'error': str(e),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'error': True
                }
            }
    
    async def integrate_data(self, sr_results: Dict[str, Any],
                           market_data: pd.DataFrame,
                           additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Integrate S/R results with additional data sources.
        
        Args:
            sr_results: S/R detection results
            market_data: Market data
            additional_data: Optional additional data sources
            
        Returns:
            Integrated data dictionary
        """
        try:
            if self.components['data_integration']:
                return await self.components['data_integration'].integrate(
                    sr_results,
                    market_data,
                    additional_data
                )
            else:
                self.logger.warning("Data integration component not available")
                return sr_results
                
        except Exception as e:
            self.logger.error(f"Data integration failed: {e}")
            return sr_results
    
    async def update_levels(self, market_data: pd.DataFrame,
                          current_price: float,
                          volume: Optional[float] = None) -> Dict[str, Any]:
        """
        Update S/R levels with new market data.
        
        Args:
            market_data: Latest market data
            current_price: Current market price
            volume: Optional volume data
            
        Returns:
            Updated S/R levels
        """
        try:
            if self.components['levels_manager']:
                return await self.components['levels_manager'].update_levels(
                    market_data,
                    current_price,
                    volume
                )
            else:
                # Fallback to full detection
                return await self.detect_sr_levels(market_data)
                
        except Exception as e:
            self.logger.error(f"Level update failed: {e}")
            return self.cached_results if self.cached_results else {}
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get initialization status of all components."""
        return self.initialization_status.copy()
    
    def get_cached_results(self) -> Optional[Dict[str, Any]]:
        """Get cached S/R detection results."""
        return self.cached_results.copy() if self.cached_results else None


async def create_sr_comprehensive_integration(config: Dict[str, Any]) -> SRComprehensiveIntegration:
    """
    Factory function to create and initialize comprehensive S/R integration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized SRComprehensiveIntegration instance
    """
    integration = SRComprehensiveIntegration(config)
    await integration.initialize()
    return integration