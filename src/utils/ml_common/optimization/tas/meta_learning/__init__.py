"""
Meta-Learning Components for TAS

Advanced meta-learning capabilities for tree-based architecture search including:
- Model-Agnostic Meta-Learning (MAML) for trees
- Few-shot learning for regime adaptation
- Continual learning for dynamic environments
- Prototypical networks for regime classification
- Uncertainty estimation and confidence scoring
"""

from .tree_meta_learning import TreeMetaLearning, TreeMAML, TreePrototypicalNetwork
from .few_shot_learning import FewShotTreeLearner, TreeFewShotAdapter
from .continual_learning import ContinualTreeLearner, TreeEpisodicMemory

__all__ = [
    'TreeMetaLearning', 'TreeMAML', 'TreePrototypicalNetwork',
    'FewShotTreeLearner', 'TreeFewShotAdapter',
    'ContinualTreeLearner', 'TreeEpisodicMemory'
]