# Auto-Optimization Implementation Roadmap

## 🎯 **Implementation Strategy**

### **Phase 1: Create Auto-Optimization Configuration System**

#### **Step 1.1: Create AutoOptimizationConfig**
**File:** `src/feature_generation/core/auto_optimization_config.py`

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

class OptimizationLevel(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

@dataclass
class AutoOptimizationConfig:
    """Configuration for automatic optimization."""
    # Core settings
    enable_auto_optimization: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # Memory optimization
    enable_memory_optimization: bool = True
    memory_threshold_mb: float = 100.0
    enable_data_compression: bool = True
    enable_chunked_processing: bool = True
    chunk_size: int = 10000
    
    # VectorBT optimization
    enable_vectorbt_optimization: bool = True
    vectorbt_threshold: int = 1000
    enable_gpu_acceleration: bool = False
    
    # Rolling operations optimization
    enable_rolling_optimization: bool = True
    enable_rolling_cache: bool = True
    rolling_cache_size: int = 100
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    enable_optimization_logging: bool = False
    
    # Strategy-specific settings
    conservative_settings: Dict[str, Any] = None
    balanced_settings: Dict[str, Any] = None
    aggressive_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conservative_settings is None:
            self.conservative_settings = {
                'enable_memory_optimization': True,
                'enable_vectorbt_optimization': False,
                'enable_rolling_optimization': False,
                'memory_threshold_mb': 500.0
            }
        
        if self.balanced_settings is None:
            self.balanced_settings = {
                'enable_memory_optimization': True,
                'enable_vectorbt_optimization': True,
                'enable_rolling_optimization': True,
                'memory_threshold_mb': 100.0,
                'vectorbt_threshold': 1000
            }
        
        if self.aggressive_settings is None:
            self.aggressive_settings = {
                'enable_memory_optimization': True,
                'enable_vectorbt_optimization': True,
                'enable_rolling_optimization': True,
                'enable_data_compression': True,
                'enable_chunked_processing': True,
                'memory_threshold_mb': 50.0,
                'vectorbt_threshold': 500
            }
    
    def get_settings_for_level(self) -> Dict[str, Any]:
        """Get settings for current optimization level."""
        if self.optimization_level == OptimizationLevel.CONSERVATIVE:
            return self.conservative_settings
        elif self.optimization_level == OptimizationLevel.BALANCED:
            return self.balanced_settings
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return self.aggressive_settings
        else:
            return self.balanced_settings
```

#### **Step 1.2: Create Optimization Strategy Classes**
**File:** `src/feature_generation/core/optimization_strategies.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class OptimizationStrategy(ABC):
    """Base class for optimization strategies."""
    
    def __init__(self, config: 'AutoOptimizationConfig'):
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self.stats = {
            'optimizations_applied': 0,
            'total_time': 0.0,
            'memory_saved_mb': 0.0
        }
    
    @abstractmethod
    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Optimize data using this strategy."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.stats.copy()

class ConservativeOptimizationStrategy(OptimizationStrategy):
    """Conservative optimization - minimal changes, maximum compatibility."""
    
    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply conservative optimization."""
        start_time = time.time()
        optimized_data = data
        
        # Only basic memory optimization
        if (self.config.enable_memory_optimization and 
            hasattr(generator, 'optimize_dataframe_processing')):
            try:
                optimized_data = generator.optimize_dataframe_processing(data)
                self.stats['optimizations_applied'] += 1
            except Exception as e:
                self.logger.warning(f"Memory optimization failed: {e}")
        
        self.stats['total_time'] += time.time() - start_time
        return optimized_data

class BalancedOptimizationStrategy(OptimizationStrategy):
    """Balanced optimization - good performance/quality tradeoff."""
    
    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply balanced optimization."""
        start_time = time.time()
        optimized_data = data
        
        # Memory optimization
        if (self.config.enable_memory_optimization and 
            hasattr(generator, 'optimize_dataframe_processing')):
            try:
                optimized_data = generator.optimize_dataframe_processing(data)
                self.stats['optimizations_applied'] += 1
            except Exception as e:
                self.logger.warning(f"Memory optimization failed: {e}")
        
        # VectorBT optimization for large datasets
        if (self.config.enable_vectorbt_optimization and 
            len(optimized_data) > self.config.vectorbt_threshold and
            hasattr(generator, '_should_use_vectorbt')):
            try:
                if generator._should_use_vectorbt(optimized_data):
                    # Apply VectorBT-specific optimizations
                    optimized_data = self._apply_vectorbt_optimizations(optimized_data, generator)
                    self.stats['optimizations_applied'] += 1
            except Exception as e:
                self.logger.warning(f"VectorBT optimization failed: {e}")
        
        self.stats['total_time'] += time.time() - start_time
        return optimized_data
    
    def _apply_vectorbt_optimizations(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply VectorBT-specific optimizations."""
        # This would include VectorBT-specific data preparation
        return data

class AggressiveOptimizationStrategy(OptimizationStrategy):
    """Aggressive optimization - maximum performance."""
    
    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply aggressive optimization."""
        start_time = time.time()
        optimized_data = data
        
        # All available optimizations
        if (self.config.enable_memory_optimization and 
            hasattr(generator, 'optimize_dataframe_processing')):
            try:
                optimized_data = generator.optimize_dataframe_processing(data)
                self.stats['optimizations_applied'] += 1
            except Exception as e:
                self.logger.warning(f"Memory optimization failed: {e}")
        
        # Chunked processing for very large datasets
        if (self.config.enable_chunked_processing and 
            len(optimized_data) > 10000 and 
            hasattr(generator, 'chunked_processing')):
            try:
                optimized_data = generator.chunked_processing(
                    optimized_data, 
                    lambda x: x,
                    chunk_size=self.config.chunk_size
                )
                self.stats['optimizations_applied'] += 1
            except Exception as e:
                self.logger.warning(f"Chunked processing failed: {e}")
        
        # VectorBT optimization
        if (self.config.enable_vectorbt_optimization and 
            hasattr(generator, '_should_use_vectorbt')):
            try:
                if generator._should_use_vectorbt(optimized_data):
                    optimized_data = self._apply_vectorbt_optimizations(optimized_data, generator)
                    self.stats['optimizations_applied'] += 1
            except Exception as e:
                self.logger.warning(f"VectorBT optimization failed: {e}")
        
        self.stats['total_time'] += time.time() - start_time
        return optimized_data
    
    def _apply_vectorbt_optimizations(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply aggressive VectorBT optimizations."""
        # This would include aggressive VectorBT optimizations
        return data
```

### **Phase 2: Create Auto-Optimized Base Class**

#### **Step 2.1: Create AutoOptimizedFeatureGenerator**
**File:** `src/feature_generation/core/auto_optimized_feature_generator.py`

```python
from typing import Dict, Any, Optional, Union
import pandas as pd
import logging
import time

from .feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory, FeatureResult
from .optimization_mixin import OptimizationMixin
from .rolling_operations_mixin import RollingOperationsMixin
from .vectorbt_optimization_mixin import VectorBTOptimizationMixin
from .auto_optimization_config import AutoOptimizationConfig, OptimizationLevel
from .optimization_strategies import (
    OptimizationStrategy, 
    ConservativeOptimizationStrategy,
    BalancedOptimizationStrategy,
    AggressiveOptimizationStrategy
)

logger = logging.getLogger(__name__)

class AutoOptimizedFeatureGenerator(FeatureGenerator, 
                                   OptimizationMixin, 
                                   RollingOperationsMixin, 
                                   VectorBTOptimizationMixin):
    """Base class with automatic optimization enabled by default."""
    
    def __init__(self, config: FeatureConfig, 
                 auto_optimization_config: Optional[AutoOptimizationConfig] = None,
                 **kwargs):
        # Initialize base classes
        super().__init__(config)
        
        # Initialize mixins
        OptimizationMixin.__init__(self)
        RollingOperationsMixin.__init__(self)
        VectorBTOptimizationMixin.__init__(self)
        
        # Auto-optimization configuration
        self.auto_optimization_config = auto_optimization_config or AutoOptimizationConfig()
        
        # Apply level-specific settings
        self._apply_level_settings()
        
        # Initialize optimization strategy
        self.optimization_strategy = self._create_optimization_strategy()
        
        # Performance tracking
        self.auto_optimization_stats = {
            'total_optimizations': 0,
            'total_optimization_time': 0.0,
            'memory_savings_mb': 0.0,
            'strategy_used': self.auto_optimization_config.optimization_level.value
        }
        
        self.logger = logger.getChild(f'AutoOptimized{self.__class__.__name__}')
        
        if self.auto_optimization_config.enable_optimization_logging:
            self.logger.info(f"Auto-optimization enabled with {self.auto_optimization_config.optimization_level.value} strategy")
    
    def _apply_level_settings(self):
        """Apply settings based on optimization level."""
        level_settings = self.auto_optimization_config.get_settings_for_level()
        
        for key, value in level_settings.items():
            if hasattr(self.auto_optimization_config, key):
                setattr(self.auto_optimization_config, key, value)
    
    def _create_optimization_strategy(self) -> OptimizationStrategy:
        """Create optimization strategy based on configuration."""
        if self.auto_optimization_config.optimization_level == OptimizationLevel.CONSERVATIVE:
            return ConservativeOptimizationStrategy(self.auto_optimization_config)
        elif self.auto_optimization_config.optimization_level == OptimizationLevel.BALANCED:
            return BalancedOptimizationStrategy(self.auto_optimization_config)
        elif self.auto_optimization_config.optimization_level == OptimizationLevel.AGGRESSIVE:
            return AggressiveOptimizationStrategy(self.auto_optimization_config)
        else:
            return BalancedOptimizationStrategy(self.auto_optimization_config)
    
    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """Generate feature with automatic optimization."""
        start_time = time.time()
        
        # Log optimization start
        if self.auto_optimization_config.enable_optimization_logging:
            self.logger.debug(f"Starting auto-optimization for {self.config.name}")
        
        # Apply automatic optimization
        if self.auto_optimization_config.enable_auto_optimization:
            data = self._auto_optimize_data(data)
        
        # Call parent generate method
        result = super().generate(data, **kwargs)
        
        # Update auto-optimization stats
        optimization_time = time.time() - start_time
        self.auto_optimization_stats['total_optimizations'] += 1
        self.auto_optimization_stats['total_optimization_time'] += optimization_time
        
        # Add optimization info to result metadata
        if result.metadata is None:
            result.metadata = {}
        
        result.metadata.update({
            'auto_optimization_enabled': self.auto_optimization_config.enable_auto_optimization,
            'optimization_strategy': self.auto_optimization_config.optimization_level.value,
            'optimization_time': optimization_time,
            'optimization_stats': self.optimization_strategy.get_stats()
        })
        
        return result
    
    def _auto_optimize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply automatic optimization using the configured strategy."""
        if not self.auto_optimization_config.enable_auto_optimization:
            return data
        
        try:
            # Use the configured optimization strategy
            optimized_data = self.optimization_strategy.optimize_data(data, self)
            
            # Update memory savings stats
            if hasattr(self, 'get_optimization_stats'):
                opt_stats = self.get_optimization_stats()
                self.auto_optimization_stats['memory_savings_mb'] += opt_stats.get('memory_saved_mb', 0.0)
            
            if self.auto_optimization_config.enable_optimization_logging:
                self.logger.debug(f"Auto-optimization completed for {self.config.name}")
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"Auto-optimization failed: {e}, using original data")
            return data
    
    def set_optimization_strategy(self, level: Union[str, OptimizationLevel]):
        """Change optimization strategy at runtime."""
        if isinstance(level, str):
            level = OptimizationLevel(level)
        
        self.auto_optimization_config.optimization_level = level
        self.optimization_strategy = self._create_optimization_strategy()
        
        if self.auto_optimization_config.enable_optimization_logging:
            self.logger.info(f"Optimization strategy changed to {level.value}")
    
    def get_auto_optimization_stats(self) -> Dict[str, Any]:
        """Get automatic optimization statistics."""
        stats = self.auto_optimization_stats.copy()
        
        if stats['total_optimizations'] > 0:
            stats['average_optimization_time'] = (
                stats['total_optimization_time'] / stats['total_optimizations']
            )
        else:
            stats['average_optimization_time'] = 0.0
        
        # Add strategy-specific stats
        stats['strategy_stats'] = self.optimization_strategy.get_stats()
        
        return stats
    
    def reset_auto_optimization_stats(self):
        """Reset automatic optimization statistics."""
        self.auto_optimization_stats = {
            'total_optimizations': 0,
            'total_optimization_time': 0.0,
            'memory_savings_mb': 0.0,
            'strategy_used': self.auto_optimization_config.optimization_level.value
        }
```

### **Phase 3: Update GeneratorFactory**

#### **Step 3.1: Enhance GeneratorFactory**
**File:** `src/feature_generation/core/generator_factory.py` (update existing)

```python
# Add these methods to existing GeneratorFactory class

def create_auto_optimized_generator(self, name: str, category: FeatureCategory,
                                  required_columns: List[str], 
                                  optimization_level: str = "balanced",
                                  auto_optimization_config: Optional[AutoOptimizationConfig] = None,
                                  **kwargs) -> Optional[FeatureGenerator]:
    """Create generator with automatic optimization enabled."""
    
    # Create feature config
    feature_config = FeatureConfig(
        name=name,
        category=category,
        description=f"Auto-optimized generator: {name}",
        required_columns=required_columns,
        parameters=kwargs
    )
    
    # Create auto-optimization config
    if auto_optimization_config is None:
        auto_optimization_config = AutoOptimizationConfig()
        auto_optimization_config.optimization_level = OptimizationLevel(optimization_level)
    
    # Create auto-optimized generator
    generator = AutoOptimizedFeatureGenerator(feature_config, auto_optimization_config)
    
    self.logger.debug(f"Created auto-optimized generator: {name} with {optimization_level} strategy")
    return generator

def create_generator_with_auto_optimization(self, name: str, 
                                          generator_class: Type[FeatureGenerator],
                                          config: FeatureConfig,
                                          optimization_level: str = "balanced",
                                          **kwargs) -> Optional[FeatureGenerator]:
    """Create any generator with auto-optimization enabled."""
    
    # Create auto-optimization config
    auto_optimization_config = AutoOptimizationConfig()
    auto_optimization_config.optimization_level = OptimizationLevel(optimization_level)
    
    # Create generator with auto-optimization
    try:
        generator = generator_class(config, auto_optimization_config=auto_optimization_config, **kwargs)
        self.logger.debug(f"Created {name} with auto-optimization enabled")
        return generator
    except Exception as e:
        self.logger.error(f"Failed to create {name} with auto-optimization: {e}")
        return None

def create_batch_auto_optimized_generators(self, generator_specs: List[Dict[str, Any]]) -> List[FeatureGenerator]:
    """Create multiple auto-optimized generators in batch."""
    generators = []
    
    for spec in generator_specs:
        name = spec.get('name')
        if not name:
            self.logger.warning("Generator spec missing name, skipping")
            continue
        
        # Extract auto-optimization settings
        optimization_level = spec.pop('optimization_level', 'balanced')
        auto_optimization_config = spec.pop('auto_optimization_config', None)
        
        generator = self.create_auto_optimized_generator(
            name=name,
            category=spec.get('category', FeatureCategory.CUSTOM),
            required_columns=spec.get('required_columns', []),
            optimization_level=optimization_level,
            auto_optimization_config=auto_optimization_config,
            **spec.get('parameters', {})
        )
        
        if generator:
            generators.append(generator)
    
    self.logger.info(f"Created {len(generators)} auto-optimized generators in batch")
    return generators
```

### **Phase 4: Update Main __init__.py**

#### **Step 4.1: Export New Classes**
**File:** `src/feature_generation/__init__.py` (update existing)

```python
# Add these imports to existing imports
from .core.auto_optimized_feature_generator import AutoOptimizedFeatureGenerator
from .core.auto_optimization_config import AutoOptimizationConfig, OptimizationLevel
from .core.optimization_strategies import (
    OptimizationStrategy,
    ConservativeOptimizationStrategy,
    BalancedOptimizationStrategy,
    AggressiveOptimizationStrategy
)

# Add to __all__ list
__all__.extend([
    "AutoOptimizedFeatureGenerator",
    "AutoOptimizationConfig", 
    "OptimizationLevel",
    "OptimizationStrategy",
    "ConservativeOptimizationStrategy",
    "BalancedOptimizationStrategy",
    "AggressiveOptimizationStrategy"
])
```

### **Phase 5: Create Usage Examples**

#### **Step 5.1: Update Example Usage**
**File:** `src/feature_generation/example_usage.py` (update existing)

```python
# Add these examples to existing file

def example_6_auto_optimization():
    """Example 6: Automatic optimization with AutoOptimizedFeatureGenerator."""
    print("🚀 Example 6: Automatic Optimization")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    # Create feature config
    config = FeatureConfig(
        name="auto_optimized_sma",
        category=FeatureCategory.CUSTOM,
        description="Auto-optimized SMA with automatic optimization",
        required_columns=["close"],
        default_lookback=20
    )
    
    # Create auto-optimized generator
    class AutoOptimizedSMAGenerator(AutoOptimizedFeatureGenerator):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Data is automatically optimized before reaching here
            return data['close'].rolling(20).mean()
    
    # Create generator with different optimization levels
    for level in ["conservative", "balanced", "aggressive"]:
        auto_config = AutoOptimizationConfig(optimization_level=OptimizationLevel(level))
        generator = AutoOptimizedSMAGenerator(config, auto_optimization_config=auto_config)
        
        # Generate feature
        result = generator.generate(data)
        
        print(f"✅ {level.capitalize()} optimization: {result.success}")
        print(f"⏱️ Total time: {result.computation_time:.3f}s")
        
        # Show optimization stats
        opt_stats = generator.get_auto_optimization_stats()
        print(f"🔧 Optimizations applied: {opt_stats['total_optimizations']}")
        print(f"💾 Memory saved: {opt_stats['memory_savings_mb']:.2f}MB")
        print()

def example_7_factory_auto_optimization():
    """Example 7: Using GeneratorFactory with auto-optimization."""
    print("🚀 Example 7: Factory Auto-Optimization")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    # Get factory
    factory = GeneratorFactory()
    
    # Create auto-optimized generators with different strategies
    strategies = ["conservative", "balanced", "aggressive"]
    
    for strategy in strategies:
        generator = factory.create_auto_optimized_generator(
            name=f"sma_{strategy}",
            category=FeatureCategory.CUSTOM,
            required_columns=["close"],
            optimization_level=strategy
        )
        
        if generator:
            # Generate feature
            result = generator.generate(data)
            
            print(f"✅ {strategy.capitalize()} strategy: {result.success}")
            print(f"⏱️ Computation time: {result.computation_time:.3f}s")
            
            # Show optimization metadata
            if result.metadata and 'optimization_stats' in result.metadata:
                opt_stats = result.metadata['optimization_stats']
                print(f"🔧 Strategy optimizations: {opt_stats['optimizations_applied']}")
        print()

def example_8_runtime_optimization_control():
    """Example 8: Runtime optimization control."""
    print("🚀 Example 8: Runtime Optimization Control")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    # Create feature config
    config = FeatureConfig(
        name="runtime_control_sma",
        category=FeatureCategory.CUSTOM,
        description="SMA with runtime optimization control",
        required_columns=["close"],
        default_lookback=20
    )
    
    # Create auto-optimized generator
    generator = AutoOptimizedFeatureGenerator(config)
    
    # Test different optimization levels at runtime
    for level in ["conservative", "balanced", "aggressive"]:
        # Change optimization strategy
        generator.set_optimization_strategy(level)
        
        # Generate feature
        result = generator.generate(data)
        
        print(f"✅ {level.capitalize()} level: {result.success}")
        print(f"⏱️ Computation time: {result.computation_time:.3f}s")
        
        # Show stats
        opt_stats = generator.get_auto_optimization_stats()
        print(f"🔧 Total optimizations: {opt_stats['total_optimizations']}")
        print(f"💾 Memory saved: {opt_stats['memory_savings_mb']:.2f}MB")
        print()

# Update main() function to include new examples
def main():
    """Run all examples."""
    print("🎯 Feature Generation System - Auto-Optimization Demo")
    print("=" * 80)
    print()
    
    try:
        example_1_basic_usage()
        example_2_with_mixins()
        example_3_factory_pattern()
        example_4_batch_operations()
        example_5_performance_comparison()
        example_6_auto_optimization()  # NEW
        example_7_factory_auto_optimization()  # NEW
        example_8_runtime_optimization_control()  # NEW
        
        print("🎉 All examples completed successfully!")
        print("=" * 80)
        print("📚 New Auto-Optimization Features:")
        print("  ✅ Automatic optimization enabled by default")
        print("  ✅ Configurable optimization strategies")
        print("  ✅ Runtime optimization control")
        print("  ✅ Performance monitoring and statistics")
        print("  ✅ Factory pattern with auto-optimization")
        print("  ✅ Backward compatibility maintained")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
```

## 🎯 **Implementation Checklist**

### **Phase 1: Configuration System**
- [ ] Create `AutoOptimizationConfig` class
- [ ] Create optimization strategy classes
- [ ] Add configuration validation
- [ ] Test configuration system

### **Phase 2: Auto-Optimized Base Class**
- [ ] Create `AutoOptimizedFeatureGenerator` class
- [ ] Implement automatic optimization logic
- [ ] Add performance monitoring
- [ ] Test auto-optimization functionality

### **Phase 3: Factory Updates**
- [ ] Update `GeneratorFactory` with auto-optimization methods
- [ ] Add batch creation methods
- [ ] Test factory functionality
- [ ] Update documentation

### **Phase 4: Integration**
- [ ] Update main `__init__.py` exports
- [ ] Update example usage
- [ ] Test integration
- [ ] Update README

### **Phase 5: Testing and Documentation**
- [ ] Create comprehensive tests
- [ ] Performance benchmarking
- [ ] Update documentation
- [ ] Create migration guide

## 🚀 **Expected Results**

After implementation:

1. **All generators automatically get optimization** by default
2. **Configurable optimization strategies** for different use cases
3. **Runtime optimization control** for dynamic adjustment
4. **Performance monitoring** built-in
5. **Backward compatibility** maintained
6. **Easy migration path** for existing code

The system will be more performant and easier to use while maintaining full backward compatibility!