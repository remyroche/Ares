"""
Example Refactored Pre-Training Step

This example demonstrates how to refactor an existing pre-training step to use
the new standardized BaseStep utilities and pre-training abstractions.

BEFORE vs AFTER comparison showing the benefits of generalization.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import the new standardized utilities
from src.training.steps.pre_training.pre_training_utilities import (
    PreTrainingStepBase, PreTrainingConfig, FeatureGenerationResult,
    DataValidationResult, create_pre_training_step
)

# ============================================================================
# BEFORE: Original Pattern (What we're replacing)
# ============================================================================

class OriginalFeatureGenerationStep:
    """
    ❌ ORIGINAL PATTERN - What we're replacing
    
    Issues with this approach:
    1. Direct tprint imports with manual fallbacks
    2. Manual configuration management
    3. Duplicated data loading logic
    4. Manual error handling
    5. Inconsistent artifact management
    6. No hardware optimization
    7. Manual memory management
    """
    
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        # ❌ Manual tprint imports with fallbacks
        try:
            from src.utils.tprint import (
                tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
                tprint_data_preview, tprint_data_format, tprint_performance, tprint_progress,
                tprint_structured, tprint_timer, tprint_exception
            )
            self.tprint = tprint
            self.tprint_info = tprint_info
            self.tprint_success = tprint_success
            self.tprint_warning = tprint_warning
            self.tprint_error = tprint_error
            self.tprint_debug = tprint_debug
            self.tprint_data_preview = tprint_data_preview
            self.tprint_data_format = tprint_data_format
            self.tprint_performance = tprint_performance
            self.tprint_progress = tprint_progress
            self.tprint_structured = tprint_structured
            self.tprint_timer = tprint_timer
            self.tprint_exception = tprint_exception
        except ImportError:
            # ❌ Manual fallback implementations
            def tprint(*args, **kwargs): print(*args)
            def tprint_info(*args, **kwargs): print("INFO:", *args)
            def tprint_success(*args, **kwargs): print("SUCCESS:", *args)
            def tprint_warning(*args, **kwargs): print("WARNING:", *args)
            def tprint_error(*args, **kwargs): print("ERROR:", *args)
            def tprint_debug(*args, **kwargs): print("DEBUG:", *args)
            def tprint_data_preview(*args, **kwargs): pass
            def tprint_data_format(*args, **kwargs): return None
            def tprint_performance(*args, **kwargs): pass
            def tprint_progress(*args, **kwargs): pass
            def tprint_structured(*args, **kwargs): pass
            def tprint_timer(*args, **kwargs): pass
            def tprint_exception(*args, **kwargs): pass
            
            self.tprint = tprint
            self.tprint_info = tprint_info
            self.tprint_success = tprint_success
            self.tprint_warning = tprint_warning
            self.tprint_error = tprint_error
            self.tprint_debug = tprint_debug
            self.tprint_data_preview = tprint_data_preview
            self.tprint_data_format = tprint_data_format
            self.tprint_performance = tprint_performance
            self.tprint_progress = tprint_progress
            self.tprint_structured = tprint_structured
            self.tprint_timer = tprint_timer
            self.tprint_exception = tprint_exception
        
        self.step_name = step_name
        self.config = config or {}
        
        # ❌ Manual configuration management
        self.symbol = config.get('symbol', 'ETHUSDT')
        self.exchange = config.get('exchange', 'binance')
        self.timeframe = config.get('timeframe', '15m')
        self.direction = config.get('direction', 'long')
        self.model = config.get('model', 'Analyst')
        
        self.tprint_info("🔧 Initializing original feature generation step")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """❌ Original execution with manual patterns."""
        self.tprint_info("🚀 Starting original feature generation")
        
        try:
            # ❌ Manual data loading
            data = await self._load_data_manually(config)
            if data is None or data.empty:
                return {'success': False, 'error': 'No data found'}
            
            # ❌ Manual data validation
            if not self._validate_data_manually(data):
                return {'success': False, 'error': 'Data validation failed'}
            
            # ❌ Manual feature generation
            features = await self._generate_features_manually(data, config)
            
            # ❌ Manual artifact saving
            artifacts = await self._save_artifacts_manually(features, config)
            
            self.tprint_success("✅ Original feature generation completed")
            return {'success': True, 'artifacts': artifacts}
            
        except Exception as e:
            self.tprint_error(f"❌ Original feature generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _load_data_manually(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """❌ Manual data loading with duplicated logic."""
        # Manual data loading logic here...
        pass
    
    def _validate_data_manually(self, data: pd.DataFrame) -> bool:
        """❌ Manual data validation."""
        # Manual validation logic here...
        pass
    
    async def _generate_features_manually(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """❌ Manual feature generation without optimization."""
        # Manual feature generation logic here...
        pass
    
    async def _save_artifacts_manually(self, features: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
        """❌ Manual artifact saving."""
        # Manual artifact saving logic here...
        pass

# ============================================================================
# AFTER: Refactored Pattern (What we're implementing)
# ============================================================================

class RefactoredFeatureGenerationStep(PreTrainingStepBase):
    """
    ✅ REFACTORED PATTERN - What we're implementing
    
    Benefits of this approach:
    1. No manual imports - uses BaseStep's built-in tprint integration
    2. Standardized configuration management
    3. Reusable data loading with fallbacks
    4. Comprehensive error handling
    5. Consistent artifact management
    6. Built-in hardware optimization
    7. Automatic memory management
    8. Performance monitoring
    9. Comprehensive logging
    10. Easy to maintain and extend
    """
    
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """✅ Initialize with BaseStep's comprehensive utilities."""
        super().__init__(step_name, config)
        
        # ✅ No manual imports needed - BaseStep provides everything
        # ✅ No manual fallbacks needed - BaseStep handles gracefully
        # ✅ No manual configuration management - PreTrainingConfig handles it
        
        self.tprint_info("🔧 Initializing refactored feature generation step")
        self.tprint_success("✅ Refactored feature generation step initialized")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """✅ Refactored execution using standardized patterns."""
        # ✅ Use standardized execution - handles everything automatically
        return await self.execute_standardized(config)
    
    async def execute_custom(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """✅ Custom execution with additional processing."""
        self.tprint_step_start("🚀 Starting custom refactored feature generation")
        
        try:
            # ✅ Initialize standardized configuration
            pre_config = self._initialize_pre_training_config(config)
            
            # ✅ Load data with standardized patterns and fallbacks
            data = await self._load_data_standardized(config)
            
            # ✅ Validate data with comprehensive quality assessment
            validation_result = await self._validate_data_standardized(data, pre_config)
            
            if not validation_result.success:
                self.tprint_error(f"❌ Data validation failed: {validation_result.error_message}")
                return {
                    'success': False,
                    'artifacts': [],
                    'metrics': validation_result.__dict__,
                    'error': validation_result.error_message
                }
            
            # ✅ Custom processing with hardware optimization
            processed_data = await self._custom_processing(data, pre_config)
            
            # ✅ Generate features with built-in optimization
            feature_result = await self._generate_features_standardized(processed_data, pre_config)
            
            if not feature_result.success:
                self.tprint_error(f"❌ Feature generation failed: {feature_result.error_message}")
                return {
                    'success': False,
                    'artifacts': [],
                    'metrics': feature_result.__dict__,
                    'error': feature_result.error_message
                }
            
            # ✅ Save artifacts with metadata tracking
            artifacts = await self._save_artifacts_standardized(feature_result, pre_config)
            
            # ✅ Generate comprehensive report
            report = await self._generate_comprehensive_report_standardized(feature_result, pre_config)
            
            self.tprint_step_end("✅ Custom refactored feature generation completed")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': {
                    'validation_result': validation_result.__dict__,
                    'feature_result': feature_result.__dict__,
                    'report': report
                }
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Custom refactored feature generation failed: {e}")
            self.tprint_exception(e)
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
    
    async def _custom_processing(self, data: pd.DataFrame, config: PreTrainingConfig) -> pd.DataFrame:
        """✅ Custom processing with hardware optimization."""
        self.tprint_operation_start("⚙️ Custom data processing")
        
        try:
            # ✅ Use BaseStep's hardware optimization
            optimized_data = self._optimize_dataframe_with_hardware(data)
            
            # ✅ Custom processing logic
            processed_data = optimized_data.copy()
            
            # Add custom features
            processed_data['custom_feature_1'] = processed_data['close'].rolling(window=10).mean()
            processed_data['custom_feature_2'] = processed_data['volume'].rolling(window=5).std()
            
            # ✅ Monitor performance
            self._monitor_performance_standardized("custom_processing", config)
            
            self.tprint_operation_end("✅ Custom data processing completed")
            return processed_data
            
        except Exception as e:
            self.tprint_error(f"❌ Custom processing failed: {e}")
            raise

# ============================================================================
# Factory Function Usage
# ============================================================================

def create_feature_generation_step(step_name: str, config: Optional[Dict[str, Any]] = None) -> PreTrainingStepBase:
    """✅ Factory function for easy step creation."""
    return create_pre_training_step(step_name, config)

# ============================================================================
# Usage Examples
# ============================================================================

async def example_usage():
    """Example of how to use the refactored step."""
    
    # ✅ Simple usage with factory function
    step = create_feature_generation_step("example_feature_generation")
    
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'long',
        'model': 'Analyst',
        'lookback_days': 30,
        'enable_hardware_optimization': True,
        'enable_data_preview': True,
        'enable_memory_monitoring': True
    }
    
    # ✅ Execute with standardized patterns
    result = await step.execute(config)
    
    if result['success']:
        print(f"✅ Success! Generated {len(result['artifacts'])} artifacts")
        print(f"📊 Metrics: {result['metrics']}")
    else:
        print(f"❌ Failed: {result['error']}")
    
    # ✅ Custom usage with additional processing
    custom_step = RefactoredFeatureGenerationStep("custom_feature_generation")
    custom_result = await custom_step.execute_custom(config)
    
    if custom_result['success']:
        print(f"✅ Custom success! Generated {len(custom_result['artifacts'])} artifacts")
    else:
        print(f"❌ Custom failed: {custom_result['error']}")

# ============================================================================
# Migration Checklist
# ============================================================================

"""
✅ MIGRATION CHECKLIST

1. Remove direct tprint imports
   ❌ from src.utils.tprint import tprint_info, tprint_data_preview
   ✅ Use self.tprint_info(), self.tprint_data_preview()

2. Remove manual fallback implementations
   ❌ try/except ImportError blocks
   ✅ BaseStep handles gracefully

3. Replace manual configuration management
   ❌ symbol = config.get('symbol', 'ETHUSDT')
   ✅ pre_config = self._initialize_pre_training_config(config)

4. Use standardized data loading
   ❌ data = load_data_manually(symbol, timeframe)
   ✅ data = await self._load_data_standardized(config)

5. Use standardized validation
   ❌ if not validate_data_manually(data):
   ✅ validation_result = await self._validate_data_standardized(data, pre_config)

6. Use standardized feature generation
   ❌ features = await generate_features_manually(data, config)
   ✅ feature_result = await self._generate_features_standardized(data, pre_config)

7. Use standardized artifact management
   ❌ artifacts = await save_artifacts_manually(features, config)
   ✅ artifacts = await self._save_artifacts_standardized(feature_result, pre_config)

8. Add performance monitoring
   ❌ No performance monitoring
   ✅ self._monitor_performance_standardized("operation_name", config)

9. Use comprehensive error handling
   ❌ Basic try/except
   ✅ Comprehensive error handling with logging

10. Use hardware optimization
    ❌ No optimization
    ✅ Built-in hardware optimization with decorators
"""

# ============================================================================
# Benefits Summary
# ============================================================================

"""
🎯 BENEFITS OF REFACTORING

1. **Code Reduction**: ~70% less code in each step
2. **Consistency**: All steps use the same patterns
3. **Maintainability**: Single source of truth for common functionality
4. **Performance**: Built-in hardware optimization and memory management
5. **Error Handling**: Comprehensive error handling and logging
6. **Developer Experience**: Consistent API and rich debugging
7. **Future-Proof**: Easy to add new capabilities
8. **Testing**: Easier to test with standardized patterns
9. **Documentation**: Self-documenting with clear patterns
10. **Onboarding**: New developers can easily understand and contribute
"""

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())