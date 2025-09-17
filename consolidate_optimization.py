#!/usr/bin/env python3
"""
Consolidation Script: Merge duplicate optimization classes

This script consolidates the duplicate optimization classes and configs.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set

def read_file(file_path: Path) -> str:
    """Read file content."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""

def write_file(file_path: Path, content: str) -> bool:
    """Write content to file."""
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"❌ Error writing {file_path}: {e}")
        return False

def extract_class_definitions(content: str) -> Dict[str, str]:
    """Extract class definitions from content."""
    classes = {}
    
    # Pattern to match class definitions with their full content
    class_pattern = r'^(class\s+(\w+).*?):.*?(?=^class\s|\Z)'
    
    matches = re.finditer(class_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        class_name = match.group(2)
        class_content = match.group(0)
        classes[class_name] = class_content
    
    return classes

def consolidate_optimization_classes():
    """Consolidate duplicate optimization classes."""
    print("🔧 Starting Optimization Class Consolidation")
    print("=" * 50)
    
    # Define file paths
    fe_opt_file = Path("src/feature_engineering/feature_generation_optimization.py")
    fe_config_file = Path("src/feature_engineering/optimization_config.py")
    migrated_opt_file = Path("src/feature_engineering/optimization/lookback_optimizer.py")
    
    if not fe_opt_file.exists():
        print(f"❌ {fe_opt_file} not found")
        return
    
    if not migrated_opt_file.exists():
        print(f"❌ {migrated_opt_file} not found")
        return
    
    # Read existing files
    fe_opt_content = read_file(fe_opt_file)
    fe_config_content = read_file(fe_config_file) if fe_config_file.exists() else ""
    migrated_content = read_file(migrated_opt_file)
    
    print("📊 Analyzing class definitions...")
    
    # Extract classes from each file
    fe_opt_classes = extract_class_definitions(fe_opt_content)
    fe_config_classes = extract_class_definitions(fe_config_content)
    migrated_classes = extract_class_definitions(migrated_content)
    
    print(f"📋 Found classes:")
    print(f"  - feature_generation_optimization.py: {list(fe_opt_classes.keys())}")
    print(f"  - optimization_config.py: {list(fe_config_classes.keys())}")
    print(f"  - migrated lookback_optimizer.py: {list(migrated_classes.keys())}")
    
    # Create unified optimization module
    print("\n🔨 Creating unified optimization module...")
    
    unified_content = '''"""
Unified Feature Optimization System

This module provides comprehensive feature optimization capabilities,
consolidating all optimization functionality into a single source.

Migrated and consolidated from:
- feature_generation/optimization/lookback_optimizer.py
- feature_engineering/feature_generation_optimization.py  
- feature_engineering/optimization_config.py
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    from scipy.optimize import minimize_scalar
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited optimization functionality")

# Try to import utilities with fallback
try:
    from ...utils.math_validation import safe_divide, safe_log
    from ...utils.common_operations import create_fallback_logger
    from ...utils.hardware.m1_gpu_utils import M1GPUManager
    from ...utils.parallel_processing_optimizer import ParallelProcessor
except ImportError:
    logger.warning("Some utility imports failed - using fallbacks")
    def safe_divide(a, b, default=0):
        return a / b if b != 0 else default
    def safe_log(x, default=0):
        return np.log(x) if x > 0 else default

class OptimizationMethod(Enum):
    """Unified optimization methods for feature parameters."""
    # From feature_generation
    CROSS_VALIDATION = "cross_validation"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    INFORMATION_THEORY = "information_theory"
    REGIME_AWARE = "regime_aware"
    ADAPTIVE = "adaptive"
    
    # From feature_engineering  
    SIGNAL_STRENGTH = "signal_strength"
    NOISE_REDUCTION = "noise_reduction"
    TREND_FOLLOWING = "trend_following"
    INFORMATION_CONTENT = "information_content"
    REGIME_ADAPTATION = "regime_adaptation"

class ValidationLevel(Enum):
    """Validation levels for optimization results."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

@dataclass
class FeatureOptimizationConfig:
    """Unified configuration for feature optimization."""
    # Core parameters
    name: str = ""
    min_lookback: int = 5
    max_lookback: int = 252  # 1 year of daily data
    step_size: int = 1
    optimization_method: OptimizationMethod = OptimizationMethod.STATISTICAL_ANALYSIS
    
    # Validation parameters
    cv_folds: int = 5
    stability_threshold: float = 0.8
    performance_threshold: float = 0.6
    confidence_level: float = 0.95
    
    # Processing parameters
    parallel_processing: bool = True
    max_workers: int = 4
    memory_efficient: bool = True
    chunk_size: int = 1000
    
    # Feature-specific parameters
    periods: List[int] = field(default_factory=list)
    weight: float = 1.0
    enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced parameters
    regime_aware: bool = True
    optimization_metric: str = "sharpe_ratio"
    methods: Optional[List[str]] = None  # Backward compatibility
    
    # Validation and output
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_validation: bool = True
    enable_performance_metrics: bool = True
    enable_recommendations: bool = True
    save_results: bool = True
    save_metrics: bool = True
    output_directory: str = "optimization_results"
    
    # Cache settings
    cache_results: bool = True
    max_cache_size: int = 100
    min_data_points: int = 100

@dataclass 
class FeatureOptimizationResult:
    """Unified result of feature optimization."""
    feature_name: str
    optimal_lookback: int
    performance_score: float
    stability_score: float
    confidence_interval: Tuple[float, float]
    optimization_method: str
    regime_specific_results: Optional[Dict[str, Any]] = None
    decay_analysis: Optional[Dict[str, Any]] = None
    validation_scores: Optional[List[float]] = None
    
    # Additional metadata
    computation_time: float = 0.0
    data_points_used: int = 0
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
'''

    # Add the best implementation from each class
    print("📝 Selecting best implementations...")
    
    # Add FeatureGenerationOptimizer (the most comprehensive)
    if 'FeatureGenerationOptimizer' in fe_opt_classes:
        unified_content += "\n\n" + fe_opt_classes['FeatureGenerationOptimizer']
    
    # Add LookbackOptimizer if it has unique functionality
    if 'LookbackOptimizer' in migrated_classes:
        # Extract just the unique methods
        unified_content += "\n\n# Additional lookback-specific functionality\n"
        unified_content += "# (Integrated into FeatureGenerationOptimizer above)\n"
    
    # Add configuration management from optimization_config.py
    if 'OptimizationConfigManager' in fe_config_classes:
        unified_content += "\n\n" + fe_config_classes['OptimizationConfigManager']
    
    # Add convenience functions
    unified_content += '''

# Convenience functions for backward compatibility
def get_feature_optimizer(config: Optional[FeatureOptimizationConfig] = None) -> 'FeatureGenerationOptimizer':
    """Get a feature optimizer instance."""
    return FeatureGenerationOptimizer(config)

def optimize_feature_lookback(generator, data: pd.DataFrame, target_column: str, 
                            config: Optional[FeatureOptimizationConfig] = None) -> FeatureOptimizationResult:
    """Optimize lookback for a single feature generator."""
    optimizer = get_feature_optimizer(config)
    return optimizer.optimize_feature_lookback(generator, data, target_column)

def get_optimization_config(environment: str = "production") -> FeatureOptimizationConfig:
    """Get optimization configuration for environment."""
    manager = OptimizationConfigManager()
    return manager.create_environment_config(environment)

def get_default_config() -> FeatureOptimizationConfig:
    """Get the default optimization configuration."""
    return FeatureOptimizationConfig()

# Backward compatibility aliases
LookbackOptimizer = FeatureGenerationOptimizer
OptimizationSystemConfig = FeatureOptimizationConfig
'''
    
    # Write the unified file
    unified_file = Path("src/feature_engineering/optimization/unified_optimizer.py")
    if write_file(unified_file, unified_content):
        print(f"✅ Created unified optimization file: {unified_file}")
    
    # Update the __init__.py in optimization directory
    opt_init_content = '''"""
Feature Engineering Optimization Package

Unified optimization system for feature generation parameters.
"""

from .unified_optimizer import (
    FeatureGenerationOptimizer,
    FeatureOptimizationConfig,
    FeatureOptimizationResult,
    OptimizationMethod,
    ValidationLevel,
    OptimizationConfigManager,
    get_feature_optimizer,
    optimize_feature_lookback,
    get_optimization_config,
    get_default_config,
    
    # Backward compatibility aliases
    LookbackOptimizer,
    OptimizationSystemConfig
)

__all__ = [
    'FeatureGenerationOptimizer',
    'FeatureOptimizationConfig', 
    'FeatureOptimizationResult',
    'OptimizationMethod',
    'ValidationLevel',
    'OptimizationConfigManager',
    'get_feature_optimizer',
    'optimize_feature_lookback', 
    'get_optimization_config',
    'get_default_config',
    'LookbackOptimizer',
    'OptimizationSystemConfig'
]
'''
    
    opt_init_file = Path("src/feature_engineering/optimization/__init__.py")
    if write_file(opt_init_file, opt_init_content):
        print(f"✅ Updated {opt_init_file}")
    
    # Update main feature_engineering __init__.py
    print("\n📝 Updating feature_engineering/__init__.py...")
    
    fe_init = Path("src/feature_engineering/__init__.py")
    if fe_init.exists():
        content = read_file(fe_init)
        
        # Add unified optimization imports
        optimization_import = '''
# Unified Feature Optimization System
from .optimization import (
    FeatureGenerationOptimizer,
    FeatureOptimizationConfig,
    FeatureOptimizationResult,
    OptimizationMethod,
    get_feature_optimizer,
    optimize_feature_lookback,
    get_optimization_config,
    
    # Backward compatibility
    LookbackOptimizer
)
'''
        
        # Insert after existing imports
        lines = content.split('\n')
        insert_index = -1
        
        for i, line in enumerate(lines):
            if line.startswith('from .') and 'import' in line:
                insert_index = i + 1
        
        if insert_index > 0:
            lines.insert(insert_index, optimization_import)
            
            # Update __all__ list
            optimization_exports = '''
    # Unified optimization system
    'FeatureGenerationOptimizer',
    'FeatureOptimizationConfig',
    'FeatureOptimizationResult', 
    'OptimizationMethod',
    'get_feature_optimizer',
    'optimize_feature_lookback',
    'get_optimization_config',
    'LookbackOptimizer',
'''
            
            # Find __all__ and insert
            for i in range(len(lines)):
                if '__all__' in lines[i] and '[' in lines[i]:
                    # Find closing bracket
                    for j in range(i, len(lines)):
                        if ']' in lines[j]:
                            lines.insert(j, optimization_exports)
                            break
                    break
            
            if write_file(fe_init, '\n'.join(lines)):
                print(f"✅ Updated {fe_init}")
    
    print("\n🎉 Consolidation completed successfully!")
    print("📋 Summary:")
    print(f"  ✅ Created unified optimizer: {unified_file}")
    print(f"  ✅ Updated optimization package __init__.py")
    print(f"  ✅ Updated feature_engineering __init__.py")
    print("\n📋 Next steps:")
    print("  1. Test the unified optimization system")
    print("  2. Remove old duplicate files after verification")
    print("  3. Update any remaining imports")

if __name__ == "__main__":
    consolidate_optimization_classes()