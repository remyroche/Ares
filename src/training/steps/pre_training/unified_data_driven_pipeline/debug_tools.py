"""
Debug Tools for Feature Selection Pipeline

This module provides debugging utilities to make the first 2 stages of feature selection
drastically smaller for debugging purposes.

Key Features:
- Ultra-small configuration overrides for Stage 1 (Battle-tested selection)
- Ultra-small configuration overrides for Stage 2 (Multi-objective optimization)
- Debug mode toggle with parameter validation
- Memory-optimized settings for debugging
"""

from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


@dataclass
class DebugConfig:
    """Debug configuration for ultra-small feature selection stages."""
    
    # Debug mode toggle
    enable_debug_mode: bool = True
    
    # Stage 1 (Battle-tested selection) - ULTRA SMALL
    stage1_max_features: int = 5  # Drastically reduced from 60
    stage1_max_screening_features: int = 10  # Drastically reduced from 100
    stage1_n_bootstrap: int = 3  # Drastically reduced from 25
    stage1_min_ic_threshold: float = 0.001  # Very relaxed threshold
    stage1_min_stability_threshold: float = 0.1  # Very relaxed threshold
    stage1_max_parallel_workers: int = 2  # Reduced parallel processing
    stage1_feature_batch_size: int = 5  # Small batches
    stage1_data_chunk_size: int = 5000  # Small data chunks
    stage1_enable_parallel_processing: bool = False  # Disable parallel for debugging
    
    # Stage 2 (Multi-objective optimization) - ULTRA SMALL
    stage2_max_features: int = 10  # Drastically reduced from 60
    stage2_n_objectives: int = 3  # Minimal objectives
    stage2_max_iterations: int = 5  # Very few iterations
    stage2_population_size: int = 3  # Tiny population
    stage2_enable_parallel_processing: bool = False  # Disable parallel for debugging
    
    # Memory optimization for debugging
    enable_aggressive_gc: bool = True
    gc_frequency_operations: int = 1  # GC after every operation
    enable_memory_mapped_files: bool = True
    enable_data_type_optimization: bool = True
    max_memory_usage_mb: int = 512  # Very small memory limit
    
    # Performance optimizations for debugging
    enable_chunked_processing: bool = True
    enable_feature_streaming: bool = True
    enable_sparse_operations: bool = True
    use_approximate_variance: bool = True
    use_ksg_mi_estimator: bool = True
    mi_sample_ratio: float = 0.1  # Use only 10% of data for MI calculation
    
    # Validation settings
    skip_heavy_validation: bool = True
    skip_economic_validation: bool = True
    skip_stability_analysis: bool = False  # Keep minimal stability analysis
    skip_diversity_analysis: bool = True
    
    def __post_init__(self):
        """Validate debug configuration."""
        if self.enable_debug_mode:
            tprint_info("🔧 [DEBUG] Debug mode enabled with ultra-small parameters")
            tprint_debug(f"🔧 [DEBUG] Stage 1 max features: {self.stage1_max_features}")
            tprint_debug(f"🔧 [DEBUG] Stage 2 max features: {self.stage2_max_features}")
            tprint_debug(f"🔧 [DEBUG] Bootstrap iterations: {self.stage1_n_bootstrap}")
            tprint_debug(f"🔧 [DEBUG] Memory limit: {self.max_memory_usage_mb}MB")


class FeatureSelectionDebugTools:
    """Debug tools for feature selection pipeline."""
    
    def __init__(self, debug_config: Optional[DebugConfig] = None):
        """Initialize debug tools."""
        self.debug_config = debug_config or DebugConfig()
        self.logger = logging.getLogger(__name__)
        
        if self.debug_config.enable_debug_mode:
            tprint_info("🔧 [DEBUG] FeatureSelectionDebugTools initialized")
            tprint_success("✅ [DEBUG] Debug mode active - stages will be drastically smaller")
    
    def get_stage1_debug_overrides(self) -> Dict[str, Any]:
        """Get Stage 1 (Battle-tested selection) debug overrides."""
        if not self.debug_config.enable_debug_mode:
            return {}
        
        tprint_info("🔧 [DEBUG] Generating Stage 1 debug overrides")
        
        overrides = {
            # Core selection parameters - ULTRA SMALL
            'max_features': self.debug_config.stage1_max_features,
            'final_selection_count': self.debug_config.stage1_max_features,
            'max_screening_features': self.debug_config.stage1_max_screening_features,
            
            # Bootstrap and stability - MINIMAL
            'n_bootstrap': self.debug_config.stage1_n_bootstrap,
            'min_ic_threshold': self.debug_config.stage1_min_ic_threshold,
            'min_stability_threshold': self.debug_config.stage1_min_stability_threshold,
            
            # Parallel processing - DISABLED for debugging
            'enable_parallel_processing': self.debug_config.stage1_enable_parallel_processing,
            'max_parallel_workers': self.debug_config.stage1_max_parallel_workers,
            
            # Memory optimization - AGGRESSIVE
            'feature_batch_size': self.debug_config.stage1_feature_batch_size,
            'data_chunk_size': self.debug_config.stage1_data_chunk_size,
            'aggressive_gc': self.debug_config.enable_aggressive_gc,
            'gc_frequency_operations': self.debug_config.gc_frequency_operations,
            'max_memory_usage_mb': self.debug_config.max_memory_usage_mb,
            
            # Performance optimizations - ENABLED
            'enable_chunked_processing': self.debug_config.enable_chunked_processing,
            'enable_feature_streaming': self.debug_config.enable_feature_streaming,
            'enable_memory_mapped_files': self.debug_config.enable_memory_mapped_files,
            'enable_data_type_optimization': self.debug_config.enable_data_type_optimization,
            'enable_sparse_operations': self.debug_config.enable_sparse_operations,
            
            # Approximate methods for speed
            'use_approximate_variance': self.debug_config.use_approximate_variance,
            'use_ksg_mi_estimator': self.debug_config.use_ksg_mi_estimator,
            'mi_sample_ratio': self.debug_config.mi_sample_ratio,
            
            # Validation settings - MINIMAL
            'skip_heavy_validation': self.debug_config.skip_heavy_validation,
            'skip_economic_validation': self.debug_config.skip_economic_validation,
            'skip_stability_analysis': self.debug_config.skip_stability_analysis,
            'skip_diversity_analysis': self.debug_config.skip_diversity_analysis,
            
            # Feature selection methods - SIMPLIFIED
            'final_selection_methods': ['lgbm'],  # Only use LightGBM for speed
            'screening_methods': ['correlation'],  # Only use correlation for speed
            
            # Quantile settings - KEEP MORE FEATURES
            'screening_use_quantile': True,
            'screening_keep_quantile': 0.8,  # Keep top 80% to avoid over-filtering
        }
        
        tprint_success(f"✅ [DEBUG] Generated {len(overrides)} Stage 1 debug overrides")
        tprint_debug(f"🔧 [DEBUG] Key overrides: max_features={overrides['max_features']}, n_bootstrap={overrides['n_bootstrap']}")
        
        return overrides
    
    def get_stage2_debug_overrides(self) -> Dict[str, Any]:
        """Get Stage 2 (Multi-objective optimization) debug overrides."""
        if not self.debug_config.enable_debug_mode:
            return {}
        
        tprint_info("🔧 [DEBUG] Generating Stage 2 debug overrides")
        
        overrides = {
            # Multi-objective parameters - ULTRA SMALL
            'max_features': self.debug_config.stage2_max_features,
            'n_objectives': self.debug_config.stage2_n_objectives,
            'max_iterations': self.debug_config.stage2_max_iterations,
            'population_size': self.debug_config.stage2_population_size,
            
            # Parallel processing - DISABLED for debugging
            'enable_parallel_processing': self.debug_config.stage2_enable_parallel_processing,
            'max_parallel_workers': 1,  # Single worker
            
            # Memory optimization - AGGRESSIVE
            'aggressive_gc': self.debug_config.enable_aggressive_gc,
            'gc_frequency_operations': self.debug_config.gc_frequency_operations,
            'max_memory_usage_mb': self.debug_config.max_memory_usage_mb,
            
            # Performance optimizations - ENABLED
            'enable_chunked_processing': self.debug_config.enable_chunked_processing,
            'enable_feature_streaming': self.debug_config.enable_feature_streaming,
            'enable_memory_mapped_files': self.debug_config.enable_memory_mapped_files,
            'enable_data_type_optimization': self.debug_config.enable_data_type_optimization,
            
            # Validation settings - MINIMAL
            'skip_heavy_validation': self.debug_config.skip_heavy_validation,
            'skip_economic_validation': self.debug_config.skip_economic_validation,
        }
        
        tprint_success(f"✅ [DEBUG] Generated {len(overrides)} Stage 2 debug overrides")
        tprint_debug(f"🔧 [DEBUG] Key overrides: max_features={overrides['max_features']}, max_iterations={overrides['max_iterations']}")
        
        return overrides
    
    def get_combined_debug_overrides(self) -> Dict[str, Any]:
        """Get combined debug overrides for both stages."""
        if not self.debug_config.enable_debug_mode:
            return {}
        
        tprint_info("🔧 [DEBUG] Generating combined debug overrides")
        
        # Combine both stage overrides
        combined_overrides = {}
        combined_overrides.update(self.get_stage1_debug_overrides())
        combined_overrides.update(self.get_stage2_debug_overrides())
        
        # Add global debug settings
        global_debug_settings = {
            'debug_mode': True,
            'debug_stage1_max_features': self.debug_config.stage1_max_features,
            'debug_stage2_max_features': self.debug_config.stage2_max_features,
            'debug_n_bootstrap': self.debug_config.stage1_n_bootstrap,
            'debug_max_iterations': self.debug_config.stage2_max_iterations,
            'debug_memory_limit_mb': self.debug_config.max_memory_usage_mb,
        }
        combined_overrides.update(global_debug_settings)
        
        tprint_success(f"✅ [DEBUG] Generated {len(combined_overrides)} combined debug overrides")
        return combined_overrides
    
    def validate_debug_parameters(self) -> bool:
        """Validate debug parameters for consistency."""
        if not self.debug_config.enable_debug_mode:
            return True
        
        tprint_info("🔧 [DEBUG] Validating debug parameters")
        
        # Check parameter consistency
        issues = []
        
        if self.debug_config.stage1_max_features < 1:
            issues.append("Stage 1 max_features must be >= 1")
        
        if self.debug_config.stage2_max_features < 1:
            issues.append("Stage 2 max_features must be >= 1")
        
        if self.debug_config.stage1_max_screening_features < self.debug_config.stage1_max_features:
            issues.append("Stage 1 max_screening_features must be >= max_features")
        
        if self.debug_config.stage1_n_bootstrap < 1:
            issues.append("Stage 1 n_bootstrap must be >= 1")
        
        if self.debug_config.stage2_max_iterations < 1:
            issues.append("Stage 2 max_iterations must be >= 1")
        
        if self.debug_config.stage2_population_size < 1:
            issues.append("Stage 2 population_size must be >= 1")
        
        if self.debug_config.max_memory_usage_mb < 100:
            issues.append("Memory limit too low (< 100MB)")
        
        if issues:
            tprint_error(f"❌ [DEBUG] Debug parameter validation failed:")
            for issue in issues:
                tprint_error(f"   • {issue}")
            return False
        
        tprint_success("✅ [DEBUG] Debug parameters validated successfully")
        return True
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get a summary of debug configuration."""
        if not self.debug_config.enable_debug_mode:
            return {'debug_mode': False}
        
        return {
            'debug_mode': True,
            'stage1': {
                'max_features': self.debug_config.stage1_max_features,
                'max_screening_features': self.debug_config.stage1_max_screening_features,
                'n_bootstrap': self.debug_config.stage1_n_bootstrap,
                'parallel_processing': self.debug_config.stage1_enable_parallel_processing,
                'feature_batch_size': self.debug_config.stage1_feature_batch_size,
                'data_chunk_size': self.debug_config.stage1_data_chunk_size,
            },
            'stage2': {
                'max_features': self.debug_config.stage2_max_features,
                'n_objectives': self.debug_config.stage2_n_objectives,
                'max_iterations': self.debug_config.stage2_max_iterations,
                'population_size': self.debug_config.stage2_population_size,
                'parallel_processing': self.debug_config.stage2_enable_parallel_processing,
            },
            'memory': {
                'max_memory_mb': self.debug_config.max_memory_usage_mb,
                'aggressive_gc': self.debug_config.enable_aggressive_gc,
                'gc_frequency': self.debug_config.gc_frequency_operations,
            },
            'validation': {
                'skip_heavy_validation': self.debug_config.skip_heavy_validation,
                'skip_economic_validation': self.debug_config.skip_economic_validation,
                'skip_stability_analysis': self.debug_config.skip_stability_analysis,
                'skip_diversity_analysis': self.debug_config.skip_diversity_analysis,
            }
        }


def create_debug_overrides(debug_mode: bool = True, 
                          stage1_max_features: int = 5,
                          stage2_max_features: int = 3,
                          n_bootstrap: int = 3,
                          max_iterations: int = 5) -> Dict[str, Any]:
    """
    Quick function to create debug overrides.
    
    Args:
        debug_mode: Enable debug mode
        stage1_max_features: Maximum features for Stage 1 (default: 5)
        stage2_max_features: Maximum features for Stage 2 (default: 3)
        n_bootstrap: Number of bootstrap iterations (default: 3)
        max_iterations: Maximum iterations for Stage 2 (default: 5)
    
    Returns:
        Dictionary of debug overrides
    """
    if not debug_mode:
        return {}
    
    tprint_info("🔧 [DEBUG] Creating quick debug overrides")
    
    debug_config = DebugConfig(
        enable_debug_mode=True,
        stage1_max_features=stage1_max_features,
        stage2_max_features=stage2_max_features,
        stage1_n_bootstrap=n_bootstrap,
        stage2_max_iterations=max_iterations
    )
    
    debug_tools = FeatureSelectionDebugTools(debug_config)
    
    if not debug_tools.validate_debug_parameters():
        tprint_error("❌ [DEBUG] Debug parameter validation failed")
        return {}
    
    overrides = debug_tools.get_combined_debug_overrides()
    tprint_success(f"✅ [DEBUG] Created debug overrides with {len(overrides)} parameters")
    
    return overrides


def print_debug_info(overrides: Dict[str, Any]) -> None:
    """Print debug information in a readable format."""
    if not overrides.get('debug_mode', False):
        tprint_info("🔧 [DEBUG] Debug mode not enabled")
        return
    
    tprint_info("🔧 [DEBUG] Debug Configuration Summary:")
    tprint_info("=" * 50)
    
    # Stage 1 info
    tprint_info("🎯 Stage 1 (Battle-tested selection):")
    tprint_info(f"   • Max features: {overrides.get('max_features', 'N/A')}")
    tprint_info(f"   • Max screening features: {overrides.get('max_screening_features', 'N/A')}")
    tprint_info(f"   • Bootstrap iterations: {overrides.get('n_bootstrap', 'N/A')}")
    tprint_info(f"   • Parallel processing: {overrides.get('enable_parallel_processing', 'N/A')}")
    tprint_info(f"   • Feature batch size: {overrides.get('feature_batch_size', 'N/A')}")
    
    # Stage 2 info
    tprint_info("🎯 Stage 2 (Multi-objective optimization):")
    tprint_info(f"   • Max features: {overrides.get('debug_stage2_max_features', 'N/A')}")
    tprint_info(f"   • Max iterations: {overrides.get('debug_max_iterations', 'N/A')}")
    tprint_info(f"   • Population size: {overrides.get('population_size', 'N/A')}")
    
    # Memory info
    tprint_info("💾 Memory optimization:")
    tprint_info(f"   • Memory limit: {overrides.get('max_memory_usage_mb', 'N/A')}MB")
    tprint_info(f"   • Aggressive GC: {overrides.get('aggressive_gc', 'N/A')}")
    tprint_info(f"   • GC frequency: {overrides.get('gc_frequency_operations', 'N/A')}")
    
    tprint_info("=" * 50)
    tprint_success("✅ [DEBUG] Debug configuration ready for ultra-fast execution")


# Example usage functions
def get_ultra_small_debug_overrides() -> Dict[str, Any]:
    """Get ultra-small debug overrides for maximum speed."""
    return create_debug_overrides(
        debug_mode=True,
        stage1_max_features=3,  # Ultra-small
        stage2_max_features=2,  # Ultra-small
        n_bootstrap=2,          # Minimal
        max_iterations=3        # Minimal
    )


def get_small_debug_overrides() -> Dict[str, Any]:
    """Get small debug overrides for reasonable speed."""
    return create_debug_overrides(
        debug_mode=True,
        stage1_max_features=5,  # Small
        stage2_max_features=3,  # Small
        n_bootstrap=3,          # Minimal
        max_iterations=5        # Minimal
    )


def get_medium_debug_overrides() -> Dict[str, Any]:
    """Get medium debug overrides for balanced speed/quality."""
    return create_debug_overrides(
        debug_mode=True,
        stage1_max_features=10,  # Medium
        stage2_max_features=5,   # Medium
        n_bootstrap=5,           # Small
        max_iterations=10        # Small
    )


if __name__ == "__main__":
    # Example usage
    tprint_info("🔧 [DEBUG] Testing debug tools")
    
    # Test ultra-small configuration
    ultra_small = get_ultra_small_debug_overrides()
    print_debug_info(ultra_small)
    
    # Test small configuration
    small = get_small_debug_overrides()
    print_debug_info(small)
    
    # Test medium configuration
    medium = get_medium_debug_overrides()
    print_debug_info(medium)
    
    tprint_success("✅ [DEBUG] Debug tools test completed")
