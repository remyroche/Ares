"""
Configuration Simplification for CVLSA

This module implements configuration simplification with:
1. Configuration profiles for common use cases
2. Auto-configuration based on dataset characteristics
3. Configuration validation and consistency checks
4. Simplified configuration interfaces
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Type
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
from enum import Enum
import inspect

logger = logging.getLogger(__name__)

class ConfigurationProfile(Enum):
    """Predefined configuration profiles."""
    # Performance profiles
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    
    # Resource profiles
    MEMORY_CONSTRAINED = "memory_constrained"
    CPU_CONSTRAINED = "cpu_constrained"
    GPU_OPTIMIZED = "gpu_optimized"
    
    # Use case profiles
    RESEARCH = "research"
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    
    # Data size profiles
    SMALL_DATASET = "small_dataset"
    MEDIUM_DATASET = "medium_dataset"
    LARGE_DATASET = "large_dataset"

@dataclass
class ConfigurationProfileData:
    """Configuration profile data structure."""
    name: str
    description: str
    category: str
    config: Dict[str, Any]
    requirements: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class AutoConfigurationResult:
    """Result of auto-configuration."""
    success: bool
    config: Dict[str, Any]
    reasoning: List[str]
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class ConfigurationProfiles:
    """Predefined configuration profiles."""
    
    def __init__(self):
        self.profiles: Dict[str, ConfigurationProfileData] = {}
        self._init_profiles()
        
        logger.info("📋 Configuration profiles initialized")
    
    def _init_profiles(self):
        """Initialize predefined configuration profiles."""
        
        # Performance profiles
        self.profiles[ConfigurationProfile.FAST.value] = ConfigurationProfileData(
            name="Fast",
            description="Optimized for speed with reduced accuracy",
            category="performance",
            config={
                'adaptive_cascade': {
                    'max_depth': 3,
                    'genetic_optimization': False,
                    'cascade_pruning': True
                },
                'variable_selection': {
                    'use_parallel': True,
                    'max_workers': 2,
                    'methods': ['variance_threshold', 'random_forest']
                },
                'feature_engineering': {
                    'enable_technical_indicators': False,
                    'enable_interaction_terms': False,
                    'enable_dimensionality_reduction': False
                },
                'performance_memory': {
                    'chunk_size': 500,
                    'enable_model_caching': True,
                    'max_cache_size': 5
                }
            },
            requirements={'min_memory_gb': 4, 'min_cpu_cores': 2},
            recommendations=[
                "Use for quick prototyping and testing",
                "Suitable for small to medium datasets",
                "May sacrifice accuracy for speed"
            ]
        )
        
        self.profiles[ConfigurationProfile.BALANCED.value] = ConfigurationProfileData(
            name="Balanced",
            description="Balanced performance and accuracy",
            category="performance",
            config={
                'adaptive_cascade': {
                    'max_depth': 5,
                    'genetic_optimization': True,
                    'cascade_pruning': True
                },
                'variable_selection': {
                    'use_parallel': True,
                    'max_workers': 4,
                    'methods': ['variance_threshold', 'mutual_info', 'random_forest', 'lasso']
                },
                'feature_engineering': {
                    'enable_technical_indicators': True,
                    'enable_interaction_terms': True,
                    'enable_dimensionality_reduction': True
                },
                'performance_memory': {
                    'chunk_size': 1000,
                    'enable_model_caching': True,
                    'max_cache_size': 10
                }
            },
            requirements={'min_memory_gb': 8, 'min_cpu_cores': 4},
            recommendations=[
                "Recommended for most use cases",
                "Good balance of speed and accuracy",
                "Suitable for medium to large datasets"
            ]
        )
        
        self.profiles[ConfigurationProfile.ACCURATE.value] = ConfigurationProfileData(
            name="Accurate",
            description="Optimized for maximum accuracy",
            category="performance",
            config={
                'adaptive_cascade': {
                    'max_depth': 8,
                    'genetic_optimization': True,
                    'cascade_pruning': False
                },
                'variable_selection': {
                    'use_parallel': True,
                    'max_workers': 8,
                    'methods': ['variance_threshold', 'mutual_info', 'f_regression', 'lasso', 'random_forest', 'rfe']
                },
                'feature_engineering': {
                    'enable_technical_indicators': True,
                    'enable_interaction_terms': True,
                    'enable_dimensionality_reduction': True,
                    'interaction_max_degree': 3
                },
                'performance_memory': {
                    'chunk_size': 2000,
                    'enable_model_caching': True,
                    'max_cache_size': 20
                }
            },
            requirements={'min_memory_gb': 16, 'min_cpu_cores': 8},
            recommendations=[
                "Use when accuracy is critical",
                "Requires significant computational resources",
                "Best for production systems"
            ]
        )
        
        # Resource profiles
        self.profiles[ConfigurationProfile.MEMORY_CONSTRAINED.value] = ConfigurationProfileData(
            name="Memory Constrained",
            description="Optimized for limited memory",
            category="resource",
            config={
                'adaptive_cascade': {
                    'max_depth': 3,
                    'genetic_optimization': False
                },
                'variable_selection': {
                    'use_parallel': False,
                    'max_workers': 1,
                    'methods': ['variance_threshold', 'lasso']
                },
                'feature_engineering': {
                    'enable_technical_indicators': False,
                    'enable_interaction_terms': False,
                    'enable_dimensionality_reduction': True
                },
                'performance_memory': {
                    'chunk_size': 100,
                    'enable_model_caching': False,
                    'memory_efficient': True
                }
            },
            requirements={'max_memory_gb': 4},
            recommendations=[
                "Use on systems with limited RAM",
                "May require longer processing time",
                "Consider cloud computing for large datasets"
            ]
        )
        
        self.profiles[ConfigurationProfile.GPU_OPTIMIZED.value] = ConfigurationProfileData(
            name="GPU Optimized",
            description="Optimized for GPU acceleration",
            category="resource",
            config={
                'adaptive_cascade': {
                    'max_depth': 6,
                    'genetic_optimization': True
                },
                'variable_selection': {
                    'use_parallel': True,
                    'max_workers': 8,
                    'use_multiprocessing': True
                },
                'feature_engineering': {
                    'enable_technical_indicators': True,
                    'enable_interaction_terms': True,
                    'enable_dimensionality_reduction': True
                },
                'performance_memory': {
                    'chunk_size': 5000,
                    'enable_model_caching': True,
                    'use_m1_gpu': True
                }
            },
            requirements={'gpu_available': True, 'min_memory_gb': 8},
            recommendations=[
                "Requires GPU support (CUDA/MPS)",
                "Significantly faster for large datasets",
                "Best for deep learning components"
            ]
        )
        
        # Use case profiles
        self.profiles[ConfigurationProfile.RESEARCH.value] = ConfigurationProfileData(
            name="Research",
            description="Configuration for research and experimentation",
            category="use_case",
            config={
                'adaptive_cascade': {
                    'max_depth': 10,
                    'genetic_optimization': True,
                    'cascade_pruning': False
                },
                'variable_selection': {
                    'use_parallel': True,
                    'max_workers': 8,
                    'methods': ['variance_threshold', 'mutual_info', 'f_regression', 'lasso', 'random_forest', 'rfe', 'extra_trees']
                },
                'feature_engineering': {
                    'enable_technical_indicators': True,
                    'enable_interaction_terms': True,
                    'enable_dimensionality_reduction': True,
                    'interaction_max_degree': 4
                },
                'performance_memory': {
                    'chunk_size': 1000,
                    'enable_model_caching': True,
                    'max_cache_size': 50
                },
                'monitoring': {
                    'enable_experiment_tracking': True,
                    'enable_detailed_analytics': True,
                    'enable_visualization': True
                }
            },
            requirements={'min_memory_gb': 16, 'min_cpu_cores': 8},
            recommendations=[
                "Use for research and experimentation",
                "Provides detailed analytics and logging",
                "May be slower but more comprehensive"
            ]
        )
        
        self.profiles[ConfigurationProfile.PRODUCTION.value] = ConfigurationProfileData(
            name="Production",
            description="Configuration for production deployment",
            category="use_case",
            config={
                'adaptive_cascade': {
                    'max_depth': 5,
                    'genetic_optimization': True,
                    'cascade_pruning': True
                },
                'variable_selection': {
                    'use_parallel': True,
                    'max_workers': 4,
                    'methods': ['variance_threshold', 'mutual_info', 'random_forest']
                },
                'feature_engineering': {
                    'enable_technical_indicators': True,
                    'enable_interaction_terms': False,
                    'enable_dimensionality_reduction': True
                },
                'performance_memory': {
                    'chunk_size': 2000,
                    'enable_model_caching': True,
                    'max_cache_size': 10
                },
                'monitoring': {
                    'enable_experiment_tracking': False,
                    'enable_detailed_analytics': False,
                    'enable_visualization': False
                }
            },
            requirements={'min_memory_gb': 8, 'min_cpu_cores': 4},
            recommendations=[
                "Optimized for production deployment",
                "Balanced performance and resource usage",
                "Minimal logging and monitoring overhead"
            ]
        )
        
        # Data size profiles
        self.profiles[ConfigurationProfile.SMALL_DATASET.value] = ConfigurationProfileData(
            name="Small Dataset",
            description="Configuration for small datasets (< 10K samples)",
            category="data_size",
            config={
                'adaptive_cascade': {
                    'max_depth': 3,
                    'genetic_optimization': False
                },
                'variable_selection': {
                    'use_parallel': False,
                    'max_workers': 2,
                    'methods': ['variance_threshold', 'mutual_info']
                },
                'feature_engineering': {
                    'enable_technical_indicators': True,
                    'enable_interaction_terms': True,
                    'enable_dimensionality_reduction': False
                },
                'performance_memory': {
                    'chunk_size': 1000,
                    'enable_model_caching': True,
                    'max_cache_size': 5
                }
            },
            requirements={'max_samples': 10000},
            recommendations=[
                "Use for small datasets",
                "Fast processing with good accuracy",
                "Suitable for quick experiments"
            ]
        )
        
        self.profiles[ConfigurationProfile.LARGE_DATASET.value] = ConfigurationProfileData(
            name="Large Dataset",
            description="Configuration for large datasets (> 100K samples)",
            category="data_size",
            config={
                'adaptive_cascade': {
                    'max_depth': 6,
                    'genetic_optimization': True,
                    'cascade_pruning': True
                },
                'variable_selection': {
                    'use_parallel': True,
                    'max_workers': 8,
                    'use_multiprocessing': True,
                    'methods': ['variance_threshold', 'lasso', 'random_forest']
                },
                'feature_engineering': {
                    'enable_technical_indicators': False,
                    'enable_interaction_terms': False,
                    'enable_dimensionality_reduction': True
                },
                'performance_memory': {
                    'chunk_size': 5000,
                    'enable_model_caching': True,
                    'max_cache_size': 20,
                    'memory_efficient': True
                }
            },
            requirements={'min_samples': 100000, 'min_memory_gb': 16},
            recommendations=[
                "Use for large datasets",
                "Memory-efficient processing",
                "May require significant computational resources"
            ]
        )
    
    def get_profile(self, profile_name: str) -> Optional[ConfigurationProfileData]:
        """Get configuration profile by name."""
        return self.profiles.get(profile_name)
    
    def list_profiles(self, category: Optional[str] = None) -> List[ConfigurationProfileData]:
        """List available profiles, optionally filtered by category."""
        profiles = list(self.profiles.values())
        
        if category:
            profiles = [p for p in profiles if p.category == category]
        
        return profiles
    
    def get_profile_recommendations(self, dataset_size: int, available_memory_gb: float,
                                  cpu_cores: int, gpu_available: bool = False) -> List[str]:
        """Get profile recommendations based on system characteristics."""
        recommendations = []
        
        # Data size recommendations
        if dataset_size < 10000:
            recommendations.append(ConfigurationProfile.SMALL_DATASET.value)
        elif dataset_size > 100000:
            recommendations.append(ConfigurationProfile.LARGE_DATASET.value)
        else:
            recommendations.append(ConfigurationProfile.MEDIUM_DATASET.value)
        
        # Resource recommendations
        if available_memory_gb < 4:
            recommendations.append(ConfigurationProfile.MEMORY_CONSTRAINED.value)
        elif available_memory_gb > 16:
            recommendations.append(ConfigurationProfile.ACCURATE.value)
        
        if gpu_available:
            recommendations.append(ConfigurationProfile.GPU_OPTIMIZED.value)
        
        # Use case recommendations
        recommendations.extend([
            ConfigurationProfile.BALANCED.value,
            ConfigurationProfile.PRODUCTION.value
        ])
        
        return recommendations

class AutoConfiguration:
    """Automatic configuration system."""
    
    def __init__(self):
        self.profiles = ConfigurationProfiles()
        self.auto_config_history: List[Dict[str, Any]] = []
        
        logger.info("🤖 Auto-configuration system initialized")
    
    def analyze_dataset(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze dataset characteristics for auto-configuration."""
        analysis = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_density': X.shape[1] / X.shape[0] if X.shape[0] > 0 else 0,
            'target_variance': np.var(y),
            'feature_variance_mean': np.mean(np.var(X, axis=0)),
            'feature_variance_std': np.std(np.var(X, axis=0)),
            'data_type': 'numerical',
            'has_missing_values': np.isnan(X).any(),
            'has_infinite_values': np.isinf(X).any(),
            'correlation_strength': self._calculate_correlation_strength(X),
            'linearity_score': self._calculate_linearity_score(X, y),
            'noise_level': self._estimate_noise_level(X, y)
        }
        
        logger.info("📊 Dataset analysis completed")
        return analysis
    
    def _calculate_correlation_strength(self, X: np.ndarray) -> float:
        """Calculate average correlation strength between features."""
        try:
            correlation_matrix = np.corrcoef(X.T)
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            correlations = correlation_matrix[mask]
            return np.mean(np.abs(correlations))
        except Exception:
            return 0.0
    
    def _calculate_linearity_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate linearity score between features and target."""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            r2 = r2_score(y, y_pred)
            return max(0, r2)
        except Exception:
            return 0.0
    
    def _estimate_noise_level(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate noise level in the data."""
        try:
            from sklearn.metrics import mean_squared_error
            
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            mse = mean_squared_error(y, y_pred)
            return mse / (np.var(y) + 1e-8)
        except Exception:
            return 0.5
    
    def analyze_system_resources(self) -> Dict[str, Any]:
        """Analyze system resources for auto-configuration."""
        try:
            import psutil
            
            analysis = {
                'cpu_cores': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'disk_available_gb': psutil.disk_usage('/').free / (1024**3)
            }
            
            # Check for GPU availability
            try:
                import torch
                analysis['gpu_available'] = torch.cuda.is_available() or torch.backends.mps.is_available()
            except ImportError:
                analysis['gpu_available'] = False
            
            logger.info("💻 System resource analysis completed")
            return analysis
            
        except Exception as e:
            logger.warning(f"System resource analysis failed: {e}")
            return {
                'cpu_cores': 4,
                'memory_total_gb': 8,
                'memory_available_gb': 4,
                'disk_total_gb': 100,
                'disk_available_gb': 50,
                'gpu_available': False
            }
    
    def auto_configure(self, X: np.ndarray, y: np.ndarray,
                      use_case: Optional[str] = None,
                      performance_priority: str = 'balanced') -> AutoConfigurationResult:
        """Automatically configure CVLSA based on dataset and system characteristics."""
        logger.info("🤖 Starting auto-configuration...")
        
        reasoning = []
        warnings = []
        recommendations = []
        
        try:
            # Analyze dataset
            dataset_analysis = self.analyze_dataset(X, y)
            reasoning.append(f"Dataset analysis: {dataset_analysis['n_samples']} samples, {dataset_analysis['n_features']} features")
            
            # Analyze system resources
            system_analysis = self.analyze_system_resources()
            reasoning.append(f"System analysis: {system_analysis['cpu_cores']} cores, {system_analysis['memory_available_gb']:.1f} GB RAM")
            
            # Get profile recommendations
            profile_recommendations = self.profiles.get_profile_recommendations(
                dataset_analysis['n_samples'],
                system_analysis['memory_available_gb'],
                system_analysis['cpu_cores'],
                system_analysis.get('gpu_available', False)
            )
            
            # Select best profile based on use case and performance priority
            selected_profile = self._select_best_profile(
                profile_recommendations, use_case, performance_priority
            )
            
            if not selected_profile:
                return AutoConfigurationResult(
                    success=False,
                    config={},
                    reasoning=reasoning,
                    warnings=["No suitable profile found"],
                    recommendations=["Consider manual configuration"]
                )
            
            # Get base configuration
            base_config = self.profiles.get_profile(selected_profile)
            config = base_config.config.copy()
            
            reasoning.append(f"Selected profile: {selected_profile}")
            
            # Customize configuration based on analysis
            config = self._customize_configuration(config, dataset_analysis, system_analysis)
            reasoning.append("Configuration customized based on analysis")
            
            # Add warnings and recommendations
            if dataset_analysis['n_samples'] > 100000 and system_analysis['memory_available_gb'] < 8:
                warnings.append("Large dataset with limited memory - consider memory-constrained profile")
                recommendations.append("Consider using chunked processing")
            
            if dataset_analysis['correlation_strength'] > 0.8:
                warnings.append("High feature correlation detected")
                recommendations.append("Consider dimensionality reduction")
            
            if dataset_analysis['noise_level'] > 0.5:
                warnings.append("High noise level detected")
                recommendations.append("Consider robust models and feature selection")
            
            # Store auto-configuration history
            self.auto_config_history.append({
                'timestamp': time.time(),
                'dataset_analysis': dataset_analysis,
                'system_analysis': system_analysis,
                'selected_profile': selected_profile,
                'config': config
            })
            
            logger.info(f"✅ Auto-configuration completed using profile: {selected_profile}")
            
            return AutoConfigurationResult(
                success=True,
                config=config,
                reasoning=reasoning,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Auto-configuration failed: {e}")
            return AutoConfigurationResult(
                success=False,
                config={},
                reasoning=reasoning,
                warnings=[f"Auto-configuration failed: {e}"],
                recommendations=["Use manual configuration or try a different profile"]
            )
    
    def _select_best_profile(self, recommendations: List[str], use_case: Optional[str],
                           performance_priority: str) -> Optional[str]:
        """Select the best profile from recommendations."""
        if not recommendations:
            return None
        
        # Priority order based on performance priority
        if performance_priority == 'speed':
            priority_order = ['fast', 'memory_constrained', 'small_dataset', 'balanced']
        elif performance_priority == 'accuracy':
            priority_order = ['accurate', 'research', 'balanced', 'production']
        else:  # balanced
            priority_order = ['balanced', 'production', 'accurate', 'fast']
        
        # Filter by use case if specified
        if use_case:
            use_case_profiles = {
                'research': ['research', 'accurate'],
                'production': ['production', 'balanced'],
                'development': ['fast', 'small_dataset']
            }
            
            if use_case in use_case_profiles:
                priority_order = use_case_profiles[use_case] + priority_order
        
        # Find first matching recommendation
        for profile in priority_order:
            if profile in recommendations:
                return profile
        
        # Return first recommendation if no priority match
        return recommendations[0]
    
    def _customize_configuration(self, config: Dict[str, Any], 
                               dataset_analysis: Dict[str, Any],
                               system_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Customize configuration based on analysis."""
        customized_config = config.copy()
        
        # Adjust based on dataset size
        n_samples = dataset_analysis['n_samples']
        if n_samples < 1000:
            # Small dataset optimizations
            customized_config['adaptive_cascade']['max_depth'] = min(3, customized_config['adaptive_cascade']['max_depth'])
            customized_config['performance_memory']['chunk_size'] = min(500, customized_config['performance_memory']['chunk_size'])
        elif n_samples > 100000:
            # Large dataset optimizations
            customized_config['performance_memory']['chunk_size'] = max(2000, customized_config['performance_memory']['chunk_size'])
            customized_config['performance_memory']['memory_efficient'] = True
        
        # Adjust based on feature count
        n_features = dataset_analysis['n_features']
        if n_features > 1000:
            customized_config['feature_engineering']['enable_dimensionality_reduction'] = True
            customized_config['variable_selection']['methods'] = ['variance_threshold', 'lasso', 'random_forest']
        
        # Adjust based on system resources
        cpu_cores = system_analysis['cpu_cores']
        if cpu_cores < 4:
            customized_config['variable_selection']['use_parallel'] = False
            customized_config['variable_selection']['max_workers'] = 1
        elif cpu_cores >= 8:
            customized_config['variable_selection']['max_workers'] = min(8, cpu_cores)
        
        memory_gb = system_analysis['memory_available_gb']
        if memory_gb < 4:
            customized_config['performance_memory']['enable_model_caching'] = False
            customized_config['performance_memory']['chunk_size'] = 100
        elif memory_gb > 16:
            customized_config['performance_memory']['max_cache_size'] = 20
        
        # Adjust based on data characteristics
        if dataset_analysis['correlation_strength'] > 0.8:
            customized_config['variable_selection']['methods'].append('rfe')
        
        if dataset_analysis['noise_level'] > 0.5:
            customized_config['variable_selection']['methods'] = ['variance_threshold', 'lasso', 'random_forest']
        
        return customized_config

class ConfigurationValidator:
    """Configuration validation system."""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
        self._init_validation_rules()
        
        logger.info("✅ Configuration validator initialized")
    
    def _init_validation_rules(self):
        """Initialize validation rules."""
        # Adaptive cascade validation
        self.validation_rules['adaptive_cascade'] = [
            self._validate_max_depth,
            self._validate_genetic_optimization,
            self._validate_cascade_pruning
        ]
        
        # Variable selection validation
        self.validation_rules['variable_selection'] = [
            self._validate_parallel_settings,
            self._validate_selection_methods,
            self._validate_worker_count
        ]
        
        # Feature engineering validation
        self.validation_rules['feature_engineering'] = [
            self._validate_technical_indicators,
            self._validate_interaction_terms,
            self._validate_dimensionality_reduction
        ]
        
        # Performance memory validation
        self.validation_rules['performance_memory'] = [
            self._validate_chunk_size,
            self._validate_cache_settings,
            self._validate_memory_efficiency
        ]
    
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration and return success status and issues."""
        issues = []
        
        for section, rules in self.validation_rules.items():
            if section in config:
                for rule in rules:
                    try:
                        rule_result = rule(config[section])
                        if not rule_result['valid']:
                            issues.extend(rule_result['issues'])
                    except Exception as e:
                        issues.append(f"Validation error in {section}: {e}")
        
        return len(issues) == 0, issues
    
    def _validate_max_depth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate max_depth parameter."""
        issues = []
        max_depth = config.get('max_depth', 5)
        
        if not isinstance(max_depth, int) or max_depth < 1:
            issues.append("max_depth must be a positive integer")
        elif max_depth > 20:
            issues.append("max_depth > 20 may cause performance issues")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_genetic_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate genetic optimization settings."""
        issues = []
        genetic_opt = config.get('genetic_optimization', False)
        
        if not isinstance(genetic_opt, bool):
            issues.append("genetic_optimization must be a boolean")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_cascade_pruning(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cascade pruning settings."""
        issues = []
        cascade_pruning = config.get('cascade_pruning', True)
        
        if not isinstance(cascade_pruning, bool):
            issues.append("cascade_pruning must be a boolean")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_parallel_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parallel processing settings."""
        issues = []
        use_parallel = config.get('use_parallel', True)
        max_workers = config.get('max_workers', 4)
        
        if not isinstance(use_parallel, bool):
            issues.append("use_parallel must be a boolean")
        
        if not isinstance(max_workers, int) or max_workers < 1:
            issues.append("max_workers must be a positive integer")
        elif max_workers > 16:
            issues.append("max_workers > 16 may cause resource issues")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_selection_methods(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate selection methods."""
        issues = []
        methods = config.get('methods', [])
        
        if not isinstance(methods, list):
            issues.append("methods must be a list")
        elif len(methods) == 0:
            issues.append("methods list cannot be empty")
        
        valid_methods = ['variance_threshold', 'mutual_info', 'f_regression', 'lasso', 'random_forest', 'rfe', 'extra_trees']
        for method in methods:
            if method not in valid_methods:
                issues.append(f"Invalid selection method: {method}")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_worker_count(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate worker count settings."""
        issues = []
        max_workers = config.get('max_workers', 4)
        use_parallel = config.get('use_parallel', True)
        
        if use_parallel and max_workers > 1:
            try:
                import multiprocessing
                available_cores = multiprocessing.cpu_count()
                if max_workers > available_cores:
                    issues.append(f"max_workers ({max_workers}) exceeds available CPU cores ({available_cores})")
            except ImportError:
                issues.append("multiprocessing not available")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_technical_indicators(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate technical indicators settings."""
        issues = []
        enable_technical = config.get('enable_technical_indicators', True)
        
        if not isinstance(enable_technical, bool):
            issues.append("enable_technical_indicators must be a boolean")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_interaction_terms(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate interaction terms settings."""
        issues = []
        enable_interactions = config.get('enable_interaction_terms', True)
        max_degree = config.get('interaction_max_degree', 2)
        
        if not isinstance(enable_interactions, bool):
            issues.append("enable_interaction_terms must be a boolean")
        
        if not isinstance(max_degree, int) or max_degree < 1:
            issues.append("interaction_max_degree must be a positive integer")
        elif max_degree > 5:
            issues.append("interaction_max_degree > 5 may cause performance issues")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_dimensionality_reduction(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dimensionality reduction settings."""
        issues = []
        enable_reduction = config.get('enable_dimensionality_reduction', True)
        reduction_method = config.get('reduction_method', 'pca')
        
        if not isinstance(enable_reduction, bool):
            issues.append("enable_dimensionality_reduction must be a boolean")
        
        valid_methods = ['pca', 'tsne', 'svd']
        if reduction_method not in valid_methods:
            issues.append(f"Invalid reduction method: {reduction_method}")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_chunk_size(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chunk size settings."""
        issues = []
        chunk_size = config.get('chunk_size', 1000)
        
        if not isinstance(chunk_size, int) or chunk_size < 1:
            issues.append("chunk_size must be a positive integer")
        elif chunk_size > 10000:
            issues.append("chunk_size > 10000 may cause memory issues")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_cache_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cache settings."""
        issues = []
        enable_caching = config.get('enable_model_caching', True)
        max_cache_size = config.get('max_cache_size', 10)
        
        if not isinstance(enable_caching, bool):
            issues.append("enable_model_caching must be a boolean")
        
        if not isinstance(max_cache_size, int) or max_cache_size < 0:
            issues.append("max_cache_size must be a non-negative integer")
        elif max_cache_size > 100:
            issues.append("max_cache_size > 100 may cause disk space issues")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_memory_efficiency(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate memory efficiency settings."""
        issues = []
        memory_efficient = config.get('memory_efficient', True)
        
        if not isinstance(memory_efficient, bool):
            issues.append("memory_efficient must be a boolean")
        
        return {'valid': len(issues) == 0, 'issues': issues}

class ConfigurationSimplification:
    """Main configuration simplification system."""
    
    def __init__(self):
        self.profiles = ConfigurationProfiles()
        self.auto_config = AutoConfiguration()
        self.validator = ConfigurationValidator()
        
        logger.info("🔧 Configuration simplification initialized")
    
    def get_profile_config(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific profile."""
        profile = self.profiles.get_profile(profile_name)
        if profile:
            return profile.config
        return None
    
    def auto_configure(self, X: np.ndarray, y: np.ndarray,
                      use_case: Optional[str] = None,
                      performance_priority: str = 'balanced') -> AutoConfigurationResult:
        """Auto-configure based on dataset and system characteristics."""
        return self.auto_config.auto_configure(X, y, use_case, performance_priority)
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration."""
        return self.validator.validate_configuration(config)
    
    def get_recommendations(self, dataset_size: int, available_memory_gb: float,
                          cpu_cores: int, gpu_available: bool = False) -> List[str]:
        """Get configuration recommendations."""
        return self.profiles.get_profile_recommendations(
            dataset_size, available_memory_gb, cpu_cores, gpu_available
        )
    
    def list_available_profiles(self, category: Optional[str] = None) -> List[ConfigurationProfileData]:
        """List available configuration profiles."""
        return self.profiles.list_profiles(category)
    
    def create_custom_profile(self, name: str, description: str, category: str,
                            config: Dict[str, Any]) -> bool:
        """Create a custom configuration profile."""
        try:
            profile = ConfigurationProfileData(
                name=name,
                description=description,
                category=category,
                config=config
            )
            
            self.profiles.profiles[name] = profile
            logger.info(f"✅ Custom profile created: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create custom profile: {e}")
            return False


# Factory functions
def create_configuration_simplification() -> ConfigurationSimplification:
    """Create configuration simplification system."""
    return ConfigurationSimplification()