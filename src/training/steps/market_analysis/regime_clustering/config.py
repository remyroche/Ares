#!/usr/bin/env python3
"""
Configuration for Regime Clustering Pipeline.

This module provides configuration templates and validation for the regime clustering
pipeline, including default parameters and validation rules.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import json


@dataclass
class RegimeClusteringConfig:
    """Configuration class for regime clustering pipeline."""
    
    # Core clustering parameters
    target_clusters: int = 20
    min_cluster_size_pct: float = 0.03  # 3%
    max_cluster_size_pct: float = 0.08  # 8%
    max_noise_pct: float = 0.05  # 5%
    
    # Clustering algorithm parameters
    linkage_method: str = 'ward'  # 'ward', 'complete', 'average', 'single'
    distance_threshold: float = None
    min_samples_per_regime: int = 5
    
    # Validation thresholds
    min_silhouette_score: float = 0.3
    max_size_variance: float = 0.01
    min_constraint_satisfaction: float = 0.8
    
    # Analysis parameters
    enable_cluster_naming: bool = True
    enable_visualization: bool = True
    export_formats: List[str] = None  # ['json', 'csv', 'parquet']
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.export_formats is None:
            self.export_formats = ['json', 'csv']
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.target_clusters < 2:
            raise ValueError("target_clusters must be at least 2")
        
        if not (0 < self.min_cluster_size_pct < self.max_cluster_size_pct < 1):
            raise ValueError("Cluster size percentages must be between 0 and 1, with min < max")
        
        if not (0 <= self.max_noise_pct <= 0.1):
            raise ValueError("max_noise_pct must be between 0 and 0.1 (10%)")
        
        if self.linkage_method not in ['ward', 'complete', 'average', 'single']:
            raise ValueError("linkage_method must be one of: ward, complete, average, single")
        
        if not (0 <= self.min_silhouette_score <= 1):
            raise ValueError("min_silhouette_score must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'target_clusters': self.target_clusters,
            'min_cluster_size_pct': self.min_cluster_size_pct,
            'max_cluster_size_pct': self.max_cluster_size_pct,
            'max_noise_pct': self.max_noise_pct,
            'linkage_method': self.linkage_method,
            'distance_threshold': self.distance_threshold,
            'min_samples_per_regime': self.min_samples_per_regime,
            'min_silhouette_score': self.min_silhouette_score,
            'max_size_variance': self.max_size_variance,
            'min_constraint_satisfaction': self.min_constraint_satisfaction,
            'enable_cluster_naming': self.enable_cluster_naming,
            'enable_visualization': self.enable_visualization,
            'export_formats': self.export_formats
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RegimeClusteringConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'RegimeClusteringConfig':
        """Load config from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_to_file(self, config_path: str) -> None:
        """Save config to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Predefined configuration templates
CONFIG_TEMPLATES = {
    'conservative': RegimeClusteringConfig(
        target_clusters=15,
        min_cluster_size_pct=0.05,  # 5%
        max_cluster_size_pct=0.10,  # 10%
        max_noise_pct=0.03,  # 3%
        min_silhouette_score=0.4,
        linkage_method='ward'
    ),
    
    'balanced': RegimeClusteringConfig(
        target_clusters=20,
        min_cluster_size_pct=0.03,  # 3%
        max_cluster_size_pct=0.08,  # 8%
        max_noise_pct=0.05,  # 5%
        min_silhouette_score=0.3,
        linkage_method='ward'
    ),
    
    'aggressive': RegimeClusteringConfig(
        target_clusters=25,
        min_cluster_size_pct=0.02,  # 2%
        max_cluster_size_pct=0.06,  # 6%
        max_noise_pct=0.08,  # 8%
        min_silhouette_score=0.25,
        linkage_method='complete'
    ),
    
    'research': RegimeClusteringConfig(
        target_clusters=30,
        min_cluster_size_pct=0.015,  # 1.5%
        max_cluster_size_pct=0.05,   # 5%
        max_noise_pct=0.10,  # 10%
        min_silhouette_score=0.2,
        linkage_method='average',
        enable_visualization=True,
        export_formats=['json', 'csv', 'parquet']
    )
}


def get_config_template(template_name: str) -> RegimeClusteringConfig:
    """
    Get a predefined configuration template.
    
    Args:
        template_name: Name of the template ('conservative', 'balanced', 'aggressive', 'research')
        
    Returns:
        RegimeClusteringConfig instance
    """
    if template_name not in CONFIG_TEMPLATES:
        available = list(CONFIG_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    
    return CONFIG_TEMPLATES[template_name]


def create_custom_config(**kwargs) -> RegimeClusteringConfig:
    """
    Create a custom configuration with specified parameters.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        RegimeClusteringConfig instance
    """
    default_config = RegimeClusteringConfig()
    config_dict = default_config.to_dict()
    config_dict.update(kwargs)
    
    return RegimeClusteringConfig.from_dict(config_dict)