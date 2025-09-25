"""
Advanced Meta-Learning Module for CLVSA Architectures

This module provides state-of-the-art meta-learning capabilities specifically
designed for tree-based CLVSA models.
"""

from .advanced_meta_learning import (
    AdvancedMetaLearningSystem,
    AdvancedMAML,
    CrossDomainMetaLearning,
    MetaLearningMethod,
    AdvancedMetaLearningConfig,
    MetaTask,
    MetaLearningResult,
    create_advanced_meta_learning_system,
    create_advanced_maml,
    create_cross_domain_meta_learning
)

__all__ = [
    'AdvancedMetaLearningSystem',
    'AdvancedMAML',
    'CrossDomainMetaLearning',
    'MetaLearningMethod',
    'AdvancedMetaLearningConfig',
    'MetaTask',
    'MetaLearningResult',
    'create_advanced_meta_learning_system',
    'create_advanced_maml',
    'create_cross_domain_meta_learning'
]