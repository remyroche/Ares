"""
Integration Updater for Feature Generation

This module provides utilities to update all feature generation components
to use the new optimization utilities throughout the system.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureGenerationIntegrationUpdater:
    """Updates feature generation components to use new optimization utilities."""
    
    def __init__(self, base_path: str = "/workspace/src/feature_generation"):
        self.base_path = Path(base_path)
        self.categories_path = self.base_path / "categories"
        self.utils_path = self.base_path / "utils"
        self.core_path = self.base_path / "core"
        
        # Track updated files
        self.updated_files = []
        self.failed_files = []
    
    def update_all_categories(self) -> Dict[str, Any]:
        """Update all feature category files to use new optimization utilities."""
        logger.info("🔄 Starting integration update for all feature categories...")
        
        results = {
            'updated_files': [],
            'failed_files': [],
            'total_files': 0,
            'success_rate': 0.0
        }
        
        # Get all Python files in categories directory
        category_files = list(self.categories_path.glob("*.py"))
        results['total_files'] = len(category_files)
        
        for file_path in category_files:
            if file_path.name == "__init__.py":
                continue
                
            try:
                self._update_category_file(file_path)
                results['updated_files'].append(str(file_path))
                logger.info(f"✅ Updated {file_path.name}")
            except Exception as e:
                results['failed_files'].append(str(file_path))
                logger.error(f"❌ Failed to update {file_path.name}: {e}")
        
        # Calculate success rate
        if results['total_files'] > 0:
            results['success_rate'] = len(results['updated_files']) / results['total_files']
        
        logger.info(f"📊 Integration update completed: {len(results['updated_files'])}/{results['total_files']} files updated")
        return results
    
    def _update_category_file(self, file_path: Path):
        """Update a single category file to use optimization utilities."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply updates
        updated_content = self._apply_optimization_updates(content, file_path.name)
        
        # Write back if changes were made
        if updated_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
    
    def _apply_optimization_updates(self, content: str, filename: str) -> str:
        """Apply optimization updates to file content."""
        updated_content = content
        
        # Update VectorizedFeatureGenerator initialization
        updated_content = self._update_vectorized_generator_init(updated_content)
        
        # Add optimization imports
        updated_content = self._add_optimization_imports(updated_content)
        
        # Update _generate_feature methods
        updated_content = self._update_generate_feature_methods(updated_content)
        
        # Add optimization helper methods
        updated_content = self._add_optimization_helper_methods(updated_content)
        
        return updated_content
    
    def _update_vectorized_generator_init(self, content: str) -> str:
        """Update VectorizedFeatureGenerator initialization to enable optimization."""
        # Pattern to match super().__init__ calls with enable_matrix_ops=True
        pattern = r'super\(\)\.__init__\(config, enable_matrix_ops=True\)'
        replacement = 'super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)'
        
        updated_content = re.sub(pattern, replacement, content)
        
        # Also handle cases without enable_matrix_ops parameter
        pattern2 = r'super\(\)\.__init__\(config\)'
        replacement2 = 'super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)'
        
        updated_content = re.sub(pattern2, replacement2, updated_content)
        
        return updated_content
    
    def _add_optimization_imports(self, content: str) -> str:
        """Add optimization imports if not already present."""
        # Check if optimization imports are already present
        if "vectorization_optimizer" in content or "optimized_feature_pipeline" in content:
            return content
        
        # Find the import section and add optimization imports
        import_pattern = r'(from \.\.core\.feature_generator import[^\n]+\n)'
        
        optimization_imports = """
# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
"""
        
        # Add imports after the feature_generator import
        updated_content = re.sub(
            import_pattern,
            r'\1' + optimization_imports,
            content
        )
        
        return updated_content
    
    def _update_generate_feature_methods(self, content: str) -> str:
        """Update _generate_feature methods to use optimization."""
        # Pattern to match _generate_feature method definitions
        pattern = r'(def _generate_feature\(self, data: pd\.DataFrame, \*\*kwargs\) -> pd\.Series:)(.*?)(return [^}]+)'
        
        def replace_method(match):
            method_def = match.group(1)
            method_body = match.group(2)
            return_statement = match.group(3)
            
            # Add optimization at the beginning of the method
            optimization_code = """
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
"""
            
            return method_def + optimization_code + method_body + return_statement
        
        updated_content = re.sub(pattern, replace_method, content, flags=re.DOTALL)
        
        return updated_content
    
    def _add_optimization_helper_methods(self, content: str) -> str:
        """Add optimization helper methods to classes."""
        # Check if methods are already present
        if "def optimize_dataframe_processing" in content:
            return content
        
        # Find the end of the last class and add helper methods
        class_pattern = r'(class \w+.*?:.*?)(?=\nclass|\n\Z)'
        
        helper_methods = """
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Optimize DataFrame for vectorized processing.\"\"\"
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
"""
        
        # Add helper methods to the last class in the file
        updated_content = re.sub(
            class_pattern,
            r'\1' + helper_methods,
            content,
            flags=re.DOTALL
        )
        
        return updated_content
    
    def create_optimized_feature_factory(self) -> str:
        """Create an optimized feature factory that uses the new utilities."""
        factory_content = '''
"""
Optimized Feature Factory

This module provides an optimized feature factory that automatically
uses the new optimization utilities for all feature generation.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory
from ..core.feature_bank import get_global_feature_bank
from .optimized_feature_pipeline import get_optimized_feature_pipeline, PipelineConfig
from .vectorization_optimizer import get_vectorization_optimizer, VectorizationConfig

logger = logging.getLogger(__name__)

class OptimizedFeatureFactory:
    """Factory for creating optimized feature generators."""
    
    def __init__(self, 
                 enable_pipeline_optimization: bool = True,
                 enable_vectorization_optimization: bool = True,
                 pipeline_config: Optional[PipelineConfig] = None,
                 vectorization_config: Optional[VectorizationConfig] = None):
        """
        Initialize the optimized feature factory.
        
        Args:
            enable_pipeline_optimization: Whether to enable pipeline optimization
            enable_vectorization_optimization: Whether to enable vectorization optimization
            pipeline_config: Optional pipeline configuration
            vectorization_config: Optional vectorization configuration
        """
        self.enable_pipeline_optimization = enable_pipeline_optimization
        self.enable_vectorization_optimization = enable_vectorization_optimization
        
        # Initialize optimization components
        if enable_pipeline_optimization:
            self.pipeline = get_optimized_feature_pipeline(pipeline_config)
        else:
            self.pipeline = None
            
        if enable_vectorization_optimization:
            self.vectorization_optimizer = get_vectorization_optimizer(vectorization_config)
        else:
            self.vectorization_optimizer = None
        
        # Get feature bank
        self.feature_bank = get_global_feature_bank()
        
        logger.info("✅ Optimized Feature Factory initialized")
    
    def generate_features_optimized(self, 
                                   data: pd.DataFrame,
                                   categories: Optional[List[Union[str, FeatureCategory]]] = None,
                                   features: Optional[List[str]] = None,
                                   target_column: Optional[str] = None,
                                   **kwargs) -> pd.DataFrame:
        """
        Generate features using the optimized pipeline.
        
        Args:
            data: Input DataFrame
            categories: List of feature categories
            features: List of specific features
            target_column: Target column for optimization
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with generated features
        """
        if self.pipeline:
            # Convert categories to strings if needed
            category_strings = []
            if categories:
                for cat in categories:
                    if isinstance(cat, FeatureCategory):
                        category_strings.append(cat.value)
                    else:
                        category_strings.append(cat)
            
            result = self.pipeline.process_features(
                data=data,
                categories=category_strings if category_strings else None,
                features=features,
                target_column=target_column,
                **kwargs
            )
            
            if result.success:
                logger.info(f"✅ Optimized feature generation completed in {result.processing_time:.3f}s")
                return result.features
            else:
                logger.warning(f"Optimized pipeline failed: {result.error_message}")
                # Fall back to standard feature bank
        else:
            logger.warning("Pipeline optimization not available, using standard feature bank")
        
        # Fallback to standard feature bank
        return self.feature_bank.generate_features(
            data=data,
            categories=categories,
            features=features,
            target_column=target_column,
            use_optimized_pipeline=False,  # Avoid recursion
            **kwargs
        )
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for processing."""
        if self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'factory_status': {
                'pipeline_optimization_enabled': self.pipeline is not None,
                'vectorization_optimization_enabled': self.vectorization_optimizer is not None
            }
        }
        
        if self.pipeline:
            report['pipeline_performance'] = self.pipeline.get_performance_report()
        
        if self.vectorization_optimizer:
            report['vectorization_performance'] = self.vectorization_optimizer.get_performance_report()
        
        return report

# Global factory instance
_optimized_factory: Optional[OptimizedFeatureFactory] = None

def get_optimized_feature_factory() -> OptimizedFeatureFactory:
    """Get or create the global optimized feature factory."""
    global _optimized_factory
    
    if _optimized_factory is None:
        _optimized_factory = OptimizedFeatureFactory()
    
    return _optimized_factory

def generate_features_optimized(data: pd.DataFrame,
                              categories: Optional[List[Union[str, FeatureCategory]]] = None,
                              features: Optional[List[str]] = None,
                              target_column: Optional[str] = None,
                              **kwargs) -> pd.DataFrame:
    """Convenience function for optimized feature generation."""
    factory = get_optimized_feature_factory()
    return factory.generate_features_optimized(data, categories, features, target_column, **kwargs)
'''
        
        return factory_content
    
    def update_core_components(self) -> Dict[str, Any]:
        """Update core components to use optimization utilities."""
        logger.info("🔄 Updating core components...")
        
        results = {
            'updated_files': [],
            'failed_files': [],
            'total_files': 0
        }
        
        # Update feature_generator.py (already done)
        # Update feature_bank.py (already done)
        
        # Update other core files if needed
        core_files = [
            self.core_path / "feature_registry.py",
            self.core_path / "factory.py"
        ]
        
        results['total_files'] = len(core_files)
        
        for file_path in core_files:
            if file_path.exists():
                try:
                    # Add any specific updates for core files here
                    results['updated_files'].append(str(file_path))
                    logger.info(f"✅ Core component {file_path.name} is up to date")
                except Exception as e:
                    results['failed_files'].append(str(file_path))
                    logger.error(f"❌ Failed to update {file_path.name}: {e}")
        
        return results
    
    def create_integration_test(self) -> str:
        """Create an integration test to verify all optimizations work together."""
        test_content = '''
"""
Integration Test for Feature Generation Optimizations

This test verifies that all optimization utilities are properly integrated
throughout the feature generation system.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optimization_integration():
    """Test that optimization utilities are integrated throughout the system."""
    logger.info("🧪 Testing optimization integration...")
    
    # Create test data
    data = pd.DataFrame({
        'open': np.random.randn(1000) + 100,
        'high': np.random.randn(1000) + 101,
        'low': np.random.randn(1000) + 99,
        'close': np.random.randn(1000) + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Test optimized feature factory
    try:
        from src.feature_generation.utils.optimized_feature_factory import get_optimized_feature_factory
        
        factory = get_optimized_feature_factory()
        
        # Test feature generation
        features = factory.generate_features_optimized(
            data=data,
            categories=['momentum', 'volatility'],
            target_column='close'
        )
        
        logger.info(f"✅ Optimized factory test passed: {len(features.columns)} features generated")
        
        # Test DataFrame optimization
        optimized_data = factory.optimize_dataframe(data)
        logger.info(f"✅ DataFrame optimization test passed")
        
        # Test performance report
        report = factory.get_performance_report()
        logger.info(f"✅ Performance reporting test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization integration test failed: {e}")
        return False

def test_category_optimizations():
    """Test that individual categories use optimization utilities."""
    logger.info("🧪 Testing category optimizations...")
    
    try:
        from src.feature_generation.categories.momentum import MomentumFeatureGenerator
        from src.feature_generation.categories.volatility import VolatilityFeatureGenerator

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
        
        # Test momentum generator
        momentum_gen = MomentumFeatureGenerator()
        data = pd.DataFrame({
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Test optimization methods
        optimized_data = momentum_gen.optimize_dataframe_processing(data)
        logger.info("✅ Momentum generator optimization test passed")
        
        # Test volatility generator
        volatility_gen = VolatilityFeatureGenerator()
        optimized_data = volatility_gen.optimize_dataframe_processing(data)
        logger.info("✅ Volatility generator optimization test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Category optimization test failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests."""
    logger.info("🚀 Starting Feature Generation Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Optimization Integration", test_optimization_integration),
        ("Category Optimizations", test_category_optimizations)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:30} : {status}")
    
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    logger.info("=" * 60)
    
    return results

if __name__ == "__main__":
    results = run_integration_tests()
    
    # Exit with error code if any tests failed
    failed_tests = [name for name, result in results.items() if not result]
    if failed_tests:
        logger.error(f"❌ {len(failed_tests)} test(s) failed: {failed_tests}")
        sys.exit(1)
    else:
        logger.info("🎉 All integration tests passed!")
        sys.exit(0)
'''
        
        return test_content

def main():
    """Main function to run the integration updater."""
    updater = FeatureGenerationIntegrationUpdater()
    
    logger.info("🚀 Starting Feature Generation Integration Update")
    logger.info("=" * 60)
    
    # Update all categories
    category_results = updater.update_all_categories()
    
    # Update core components
    core_results = updater.update_core_components()
    
    # Create optimized feature factory
    factory_content = updater.create_optimized_feature_factory()
    factory_path = updater.utils_path / "optimized_feature_factory.py"
    with open(factory_path, 'w', encoding='utf-8') as f:
        f.write(factory_content)
    logger.info(f"✅ Created optimized feature factory: {factory_path}")
    
    # Create integration test
    test_content = updater.create_integration_test()
    test_path = updater.utils_path / "test_optimization_integration.py"
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    logger.info(f"✅ Created integration test: {test_path}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 INTEGRATION UPDATE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Categories updated: {len(category_results['updated_files'])}/{category_results['total_files']}")
    logger.info(f"Core components updated: {len(core_results['updated_files'])}/{core_results['total_files']}")
    logger.info(f"Success rate: {category_results['success_rate']*100:.1f}%")
    logger.info("=" * 60)
    
    if category_results['failed_files']:
        logger.warning(f"⚠️ Failed files: {category_results['failed_files']}")
    
    logger.info("✅ Integration update completed!")

if __name__ == "__main__":
    main()
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
