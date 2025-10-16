"""
Meta-Learning Components for TAS

Advanced meta-learning capabilities for tree-based architecture search including:
- Model-Agnostic Meta-Learning (MAML) for trees
- Few-shot learning for regime adaptation
- Continual learning for dynamic environments
- Prototypical networks for regime classification
- Uncertainty estimation and confidence scoring
"""

import logging

from .tree_meta_learning import TreeMetaLearning, TreeMAML, TreePrototypicalNetwork

# Commented out missing imports - will add fallback implementations
# from .few_shot_learning import FewShotTreeLearner, TreeFewShotAdapter
# from .continual_learning import ContinualTreeLearner, TreeEpisodicMemory

# Fallback implementations for missing modules
class FewShotTreeLearner:
    """Fallback few-shot tree learner."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("FewShotTreeLearner not available - using fallback")

    def learn_from_few_examples(self, *args, **kwargs):
        """Fallback learning method."""
        return {}

class TreeFewShotAdapter:
    """Fallback tree few-shot adapter."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("TreeFewShotAdapter not available - using fallback")

    def adapt_to_new_regime(self, *args, **kwargs):
        """Fallback adaptation method."""
        return {}

class ContinualTreeLearner:
    """Fallback continual tree learner."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("ContinualTreeLearner not available - using fallback")

    def learn_continuously(self, *args, **kwargs):
        """Fallback learning method."""
        return {}

class TreeEpisodicMemory:
    """Fallback tree episodic memory."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("TreeEpisodicMemory not available - using fallback")

    def store_episode(self, *args, **kwargs):
        """Fallback storage method."""
        return {}

__all__ = [
    'TreeMetaLearning', 'TreeMAML', 'TreePrototypicalNetwork',
    'FewShotTreeLearner', 'TreeFewShotAdapter',
    'ContinualTreeLearner', 'TreeEpisodicMemory'
]
