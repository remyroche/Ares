"""
Standardized Interfaces for NAS/TAS Systems

This module provides common interfaces that both NAS and TAS implementations
must follow, ensuring consistency and interoperability.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import asyncio

from .config.base_config import UnifiedArchitectureConfig
from .results.result_manager import UnifiedArchitectureResult, ArchitectureResult
from .evaluation.unified_evaluator import EvaluationResult


@dataclass
class SearchResult:
    """Result from architecture search."""
    architectures: List[ArchitectureResult]
    best_architecture: Optional[ArchitectureResult]
    search_metadata: Dict[str, Any]
    execution_time: float


@dataclass
class OptimizationResult:
    """Result from optimization process."""
    optimized_config: Dict[str, Any]
    performance_improvement: float
    optimization_metadata: Dict[str, Any]


class ArchitectureSearchInterface(ABC):
    """
    Common interface for architecture search across NAS and TAS.
    
    This interface ensures that both NAS and TAS implementations
    provide consistent methods for architecture search and evaluation.
    """
    
    @abstractmethod
    async def search(
        self, 
        data: Any, 
        config: UnifiedArchitectureConfig
    ) -> SearchResult:
        """
        Perform architecture search.
        
        Args:
            data: Training data
            config: Search configuration
            
        Returns:
            SearchResult with found architectures
        """
        pass
    
    @abstractmethod
    async def evaluate(
        self, 
        architecture: ArchitectureResult, 
        data: Any
    ) -> EvaluationResult:
        """
        Evaluate a specific architecture.
        
        Args:
            architecture: Architecture to evaluate
            data: Evaluation data
            
        Returns:
            EvaluationResult with performance metrics
        """
        pass
    
    @abstractmethod
    async def optimize(
        self, 
        architectures: List[ArchitectureResult], 
        objectives: List[str]
    ) -> OptimizationResult:
        """
        Optimize architectures for specific objectives.
        
        Args:
            architectures: List of architectures to optimize
            objectives: List of optimization objectives
            
        Returns:
            OptimizationResult with optimized configuration
        """
        pass
    
    @abstractmethod
    def get_supported_architectures(self) -> List[str]:
        """
        Get list of supported architecture types.
        
        Returns:
            List of supported architecture type names
        """
        pass
    
    @abstractmethod
    def get_search_capabilities(self) -> Dict[str, Any]:
        """
        Get search capabilities and limitations.
        
        Returns:
            Dictionary with capability information
        """
        pass


class TrainingPipelineInterface(ABC):
    """
    Common interface for training pipelines across NAS and TAS.
    
    This interface ensures consistent training orchestration
    and model management across both systems.
    """
    
    @abstractmethod
    async def train(
        self, 
        data: Any, 
        config: UnifiedArchitectureConfig
    ) -> UnifiedArchitectureResult:
        """
        Train models with given configuration.
        
        Args:
            data: Training data
            config: Training configuration
            
        Returns:
            UnifiedArchitectureResult with training results
        """
        pass
    
    @abstractmethod
    async def validate(
        self, 
        models: List[Any], 
        data: Any
    ) -> Dict[str, EvaluationResult]:
        """
        Validate trained models.
        
        Args:
            models: List of trained models
            data: Validation data
            
        Returns:
            Dictionary mapping model IDs to evaluation results
        """
        pass
    
    @abstractmethod
    async def save_models(
        self, 
        models: List[Any], 
        path: str
    ) -> bool:
        """
        Save trained models to disk.
        
        Args:
            models: List of models to save
            path: Save path
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def load_models(self, path: str) -> List[Any]:
        """
        Load models from disk.
        
        Args:
            path: Load path
            
        Returns:
            List of loaded models
        """
        pass
    
    @abstractmethod
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status.
        
        Returns:
            Dictionary with training status information
        """
        pass
    
    @abstractmethod
    def cancel_training(self) -> bool:
        """
        Cancel ongoing training.
        
        Returns:
            True if cancellation was successful
        """
        pass


class DataInterface(ABC):
    """
    Common interface for data handling across NAS and TAS.
    
    This interface ensures consistent data processing and
    validation across both systems.
    """
    
    @abstractmethod
    async def load_data(
        self, 
        source: str, 
        config: Dict[str, Any]
    ) -> Any:
        """
        Load data from source.
        
        Args:
            source: Data source identifier
            config: Data loading configuration
            
        Returns:
            Loaded data
        """
        pass
    
    @abstractmethod
    async def preprocess_data(
        self, 
        data: Any, 
        config: Dict[str, Any]
    ) -> Any:
        """
        Preprocess data.
        
        Args:
            data: Raw data
            config: Preprocessing configuration
            
        Returns:
            Preprocessed data
        """
        pass
    
    @abstractmethod
    async def validate_data(
        self, 
        data: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate data quality.
        
        Args:
            data: Data to validate
            config: Validation configuration
            
        Returns:
            Validation results
        """
        pass
    
    @abstractmethod
    def get_data_info(self, data: Any) -> Dict[str, Any]:
        """
        Get information about data.
        
        Args:
            data: Data object
            
        Returns:
            Dictionary with data information
        """
        pass


class EvaluationInterface(ABC):
    """
    Common interface for evaluation across NAS and TAS.
    
    This interface ensures consistent evaluation methods
    and metrics across both systems.
    """
    
    @abstractmethod
    async def evaluate_performance(
        self, 
        model: Any, 
        data: Any
    ) -> EvaluationResult:
        """
        Evaluate model performance.
        
        Args:
            model: Model to evaluate
            data: Evaluation data
            
        Returns:
            EvaluationResult with performance metrics
        """
        pass
    
    @abstractmethod
    async def evaluate_financial(
        self, 
        model: Any, 
        data: Any
    ) -> Dict[str, float]:
        """
        Evaluate financial performance.
        
        Args:
            model: Model to evaluate
            data: Financial data
            
        Returns:
            Dictionary with financial metrics
        """
        pass
    
    @abstractmethod
    async def evaluate_regime(
        self, 
        model: Any, 
        data: Any
    ) -> Dict[str, float]:
        """
        Evaluate regime-specific performance.
        
        Args:
            model: Model to evaluate
            data: Regime data
            
        Returns:
            Dictionary with regime metrics
        """
        pass
    
    @abstractmethod
    def get_evaluation_metrics(self) -> List[str]:
        """
        Get list of available evaluation metrics.
        
        Returns:
            List of metric names
        """
        pass


class UnifiedNASInterface(ArchitectureSearchInterface, TrainingPipelineInterface):
    """
    Unified interface for NAS systems.
    
    Combines architecture search and training capabilities
    specific to Neural Architecture Search.
    """
    
    @abstractmethod
    async def search_neural_architectures(
        self, 
        data: Any, 
        config: UnifiedArchitectureConfig
    ) -> SearchResult:
        """
        Search for neural architectures specifically.
        
        Args:
            data: Training data
            config: Search configuration
            
        Returns:
            SearchResult with neural architectures
        """
        pass
    
    @abstractmethod
    def get_neural_architecture_space(self) -> Dict[str, Any]:
        """
        Get neural architecture search space.
        
        Returns:
            Dictionary describing architecture space
        """
        pass


class UnifiedTASInterface(ArchitectureSearchInterface, TrainingPipelineInterface):
    """
    Unified interface for TAS systems.
    
    Combines architecture search and training capabilities
    specific to Tree Architecture Search.
    """
    
    @abstractmethod
    async def search_tree_architectures(
        self, 
        data: Any, 
        config: UnifiedArchitectureConfig
    ) -> SearchResult:
        """
        Search for tree architectures specifically.
        
        Args:
            data: Training data
            config: Search configuration
            
        Returns:
            SearchResult with tree architectures
        """
        pass
    
    @abstractmethod
    def get_tree_architecture_space(self) -> Dict[str, Any]:
        """
        Get tree architecture search space.
        
        Returns:
            Dictionary describing architecture space
        """
        pass


class HybridInterface(UnifiedNASInterface, UnifiedTASInterface):
    """
    Hybrid interface combining NAS and TAS capabilities.
    
    Provides unified access to both neural and tree architecture
    search capabilities in a single interface.
    """
    
    @abstractmethod
    async def search_hybrid_architectures(
        self, 
        data: Any, 
        config: UnifiedArchitectureConfig
    ) -> SearchResult:
        """
        Search for hybrid neural-tree architectures.
        
        Args:
            data: Training data
            config: Search configuration
            
        Returns:
            SearchResult with hybrid architectures
        """
        pass
    
    @abstractmethod
    def get_hybrid_architecture_space(self) -> Dict[str, Any]:
        """
        Get hybrid architecture search space.
        
        Returns:
            Dictionary describing hybrid architecture space
        """
        pass
    
    @abstractmethod
    async def balance_architectures(
        self, 
        neural_archs: List[ArchitectureResult], 
        tree_archs: List[ArchitectureResult]
    ) -> List[ArchitectureResult]:
        """
        Balance neural and tree architectures.
        
        Args:
            neural_archs: List of neural architectures
            tree_archs: List of tree architectures
            
        Returns:
            Balanced list of architectures
        """
        pass