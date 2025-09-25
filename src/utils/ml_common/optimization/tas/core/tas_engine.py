"""
Advanced Tree Architecture Search Engine

Main engine for tree-based architecture search with advanced capabilities
including meta-learning, hardware optimization, and regime-aware search.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
from enum import Enum

# Import TAS components
from .tas_config import TASConfig, TASSearchConfig, TASOptimizationConfig
from .tas_result import TASResult, TASSearchResult, TASOptimizationResult
from .tree_architecture import TreeArchitecture, TreeArchitectureCandidate
from .search_space import TreeSearchSpace

# Import advanced components
from ..meta_learning.tree_meta_learning import TreeMetaLearning, TreeMAML
from ..search.evolutionary_search import EvolutionaryTreeSearch
from src.utils.nas_tas.bayesian_search import BayesianTreeSearch
from ..search.rl_search import RLTreeSearch
from ..optimization.hardware_optimization import TreeHardwareOptimizer
from ..uncertainty.uncertainty_estimation import TreeUncertaintyEstimator
from ..regime_analysis.tree_regime_analyzer import TreeRegimeAnalyzer
from ..adaptation.real_time_adaptation import TreeRealTimeAdapter
from ..evaluation.tree_evaluator import TreeEvaluator
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategies for tree architecture search."""
    RANDOM = "random"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"
    HYBRID = "hybrid"


class OptimizationMode(Enum):
    """Optimization modes for TAS."""
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    REGIME_AWARE = "regime_aware"
    REAL_TIME = "real_time"
    CONTINUAL = "continual"


@dataclass
class TASEngineConfig:
    """Configuration for the TAS engine."""
    
    # Base configuration
    base_config: TASConfig = field(default_factory=TASConfig)
    search_config: TASSearchConfig = field(default_factory=TASSearchConfig)
    optimization_config: TASOptimizationConfig = field(default_factory=TASOptimizationConfig)
    
    # Advanced features
    enable_meta_learning: bool = True
    enable_hardware_optimization: bool = True
    enable_uncertainty_estimation: bool = True
    enable_regime_analysis: bool = True
    enable_real_time_adaptation: bool = True
    enable_continual_learning: bool = True
    
    # Search strategy
    search_strategy: SearchStrategy = SearchStrategy.HYBRID
    optimization_mode: OptimizationMode = OptimizationMode.REGIME_AWARE
    
    # Performance settings
    max_search_time: int = 3600  # 1 hour
    max_evaluations: int = 1000
    parallel_evaluations: int = 4
    memory_limit_gb: float = 8.0
    
    # Output settings
    save_results: bool = True
    save_models: bool = True
    output_dir: str = "tas_results"
    verbose: bool = True


class TreeArchitectureSearchEngine:
    """
    Advanced Tree Architecture Search Engine.
    
    Provides comprehensive tree-based architecture search with advanced capabilities
    including meta-learning, hardware optimization, uncertainty estimation,
    regime analysis, and real-time adaptation.
    """
    
    def __init__(self, config: TASEngineConfig):
        """Initialize the TAS engine.
        
        Args:
            config: TAS engine configuration
        """
        tprint_info("🚀 Initializing Advanced Tree Architecture Search Engine")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize core components
        tprint_info("🔧 Initializing core components...")
        tprint_debug("🌳 Creating search space...")
        self.search_space = TreeSearchSpace(config.base_config)
        tprint_success("✅ Search space created")
        
        tprint_debug("📊 Creating evaluator...")
        self.evaluator = TreeEvaluator(config.base_config)
        tprint_success("✅ Evaluator created")
        
        # Initialize advanced components
        tprint_info("⚡ Initializing advanced components...")
        self._initialize_advanced_components()
        
        # Search state
        tprint_debug("📊 Initializing search state...")
        self.search_history = []
        self.best_architectures = []
        self.current_search = None
        self.performance_monitor = None
        tprint_success("✅ Search state initialized")
        
        tprint_success("✅ Advanced TAS Engine initialized")
        tprint_info(f"🔍 Search strategy: {config.search_strategy.value}")
        tprint_info(f"⚙️ Optimization mode: {config.optimization_mode.value}")
        tprint_info(f"🧠 Meta-learning: {config.enable_meta_learning}")
        tprint_info(f"🖥️ Hardware optimization: {config.enable_hardware_optimization}")
        tprint_info(f"🎯 Uncertainty estimation: {config.enable_uncertainty_estimation}")
        tprint_info(f"📊 Regime analysis: {config.enable_regime_analysis}")
        tprint_info(f"⚡ Real-time adaptation: {config.enable_real_time_adaptation}")
        self.logger.info("✅ Advanced TAS Engine initialized")
        self.logger.info(f"🔍 Search strategy: {config.search_strategy.value}")
        self.logger.info(f"⚙️ Optimization mode: {config.optimization_mode.value}")
        self.logger.info(f"🧠 Meta-learning: {config.enable_meta_learning}")
        self.logger.info(f"🖥️ Hardware optimization: {config.enable_hardware_optimization}")
        self.logger.info(f"🎯 Uncertainty estimation: {config.enable_uncertainty_estimation}")
        self.logger.info(f"📊 Regime analysis: {config.enable_regime_analysis}")
        self.logger.info(f"⚡ Real-time adaptation: {config.enable_real_time_adaptation}")
    
    def _initialize_advanced_components(self):
        """Initialize advanced TAS components."""
        tprint_debug("🔧 Initializing advanced TAS components...")
        try:
            # Meta-learning components
            if self.config.enable_meta_learning:
                tprint_debug("🧠 Initializing meta-learning components...")
                self.meta_learner = TreeMetaLearning(self.config.base_config)
                self.maml = TreeMAML(self.config.base_config)
                tprint_success("✅ Meta-learning components initialized")
                self.logger.info("✅ Meta-learning components initialized")
            
            # Hardware optimization
            if self.config.enable_hardware_optimization:
                self.hardware_optimizer = TreeHardwareOptimizer(self.config.base_config)
                self.logger.info("✅ Hardware optimization initialized")
            
            # Uncertainty estimation
            if self.config.enable_uncertainty_estimation:
                self.uncertainty_estimator = TreeUncertaintyEstimator(self.config.base_config)
                self.logger.info("✅ Uncertainty estimation initialized")
            
            # Regime analysis
            if self.config.enable_regime_analysis:
                self.regime_analyzer = TreeRegimeAnalyzer(self.config.base_config)
                self.logger.info("✅ Regime analysis initialized")
            
            # Real-time adaptation
            if self.config.enable_real_time_adaptation:
                self.real_time_adapter = TreeRealTimeAdapter(self.config.base_config)
                self.performance_monitor = TreePerformanceMonitor(self.config.base_config)
                self.logger.info("✅ Real-time adaptation initialized")
            
            # Search strategies
            self._initialize_search_strategies()
            
        except Exception as e:
            self.logger.error(f"❌ Advanced components initialization failed: {e}")
            raise
    
    def _initialize_search_strategies(self):
        """Initialize search strategies."""
        try:
            self.search_strategies = {}
            
            # Bayesian search
            self.search_strategies[SearchStrategy.BAYESIAN] = BayesianTreeSearch(
                self.config.search_config
            )
            
            # Evolutionary search
            self.search_strategies[SearchStrategy.EVOLUTIONARY] = EvolutionaryTreeSearch(
                self.config.search_config
            )
            
            # Reinforcement learning search
            self.search_strategies[SearchStrategy.REINFORCEMENT] = RLTreeSearch(
                self.config.search_config
            )
            
            self.logger.info("✅ Search strategies initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Search strategies initialization failed: {e}")
            raise
    
    def search(self,
               train_data: Tuple[np.ndarray, np.ndarray],
               validation_data: Tuple[np.ndarray, np.ndarray],
               test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               regime_data: Optional[Dict[str, Any]] = None,
               search_strategy: Optional[SearchStrategy] = None,
               optimization_mode: Optional[OptimizationMode] = None) -> TASResult:
        """
        Perform advanced tree architecture search.
        
        Args:
            train_data: Training data (X, y)
            validation_data: Validation data (X, y)
            test_data: Optional test data (X, y)
            regime_data: Optional regime information
            search_strategy: Search strategy to use
            optimization_mode: Optimization mode to use
            
        Returns:
            TASResult with search results
        """
        start_time = time.time()
        self.logger.info("🚀 Starting advanced tree architecture search")
        
        # Use provided strategy or default
        strategy = search_strategy or self.config.search_strategy
        mode = optimization_mode or self.config.optimization_mode
        
        self.logger.info(f"🔍 Using search strategy: {strategy.value}")
        self.logger.info(f"⚙️ Using optimization mode: {mode.value}")
        
        try:
            # Prepare search environment
            search_env = self._prepare_search_environment(
                train_data, validation_data, test_data, regime_data
            )
            
            # Select search strategy
            searcher = self._select_search_strategy(strategy)
            
            # Perform search based on optimization mode
            if mode == OptimizationMode.SINGLE_OBJECTIVE:
                result = self._single_objective_search(searcher, search_env)
            elif mode == OptimizationMode.MULTI_OBJECTIVE:
                result = self._multi_objective_search(searcher, search_env)
            elif mode == OptimizationMode.REGIME_AWARE:
                result = self._regime_aware_search(searcher, search_env)
            elif mode == OptimizationMode.REAL_TIME:
                result = self._real_time_search(searcher, search_env)
            elif mode == OptimizationMode.CONTINUAL:
                result = self._continual_search(searcher, search_env)
            else:
                raise ValueError(f"Unknown optimization mode: {mode}")
            
            # Post-process results
            result = self._post_process_results(result, search_env)
            
            # Save results if requested
            if self.config.save_results:
                self._save_search_results(result)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.logger.info(f"✅ Advanced TAS completed in {execution_time:.2f}s")
            self.logger.info(f"🏆 Best architecture: {result.best_architecture}")
            self.logger.info(f"🎯 Best score: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Advanced TAS failed: {e}")
            
            return TASResult(
                best_architecture=None,
                best_score=0.0,
                search_history=[],
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _prepare_search_environment(self,
                                   train_data: Tuple[np.ndarray, np.ndarray],
                                   validation_data: Tuple[np.ndarray, np.ndarray],
                                   test_data: Optional[Tuple[np.ndarray, np.ndarray]],
                                   regime_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare search environment with all necessary components."""
        try:
            search_env = {
                'train_data': train_data,
                'validation_data': validation_data,
                'test_data': test_data,
                'regime_data': regime_data,
                'search_space': self.search_space,
                'evaluator': self.evaluator
            }
            
            # Add advanced components if enabled
            if self.config.enable_meta_learning:
                search_env['meta_learner'] = self.meta_learner
                search_env['maml'] = self.maml
            
            if self.config.enable_hardware_optimization:
                search_env['hardware_optimizer'] = self.hardware_optimizer
            
            if self.config.enable_uncertainty_estimation:
                search_env['uncertainty_estimator'] = self.uncertainty_estimator
            
            if self.config.enable_regime_analysis:
                search_env['regime_analyzer'] = self.regime_analyzer
            
            if self.config.enable_real_time_adaptation:
                search_env['real_time_adapter'] = self.real_time_adapter
                search_env['performance_monitor'] = self.performance_monitor
            
            return search_env
            
        except Exception as e:
            self.logger.error(f"❌ Search environment preparation failed: {e}")
            raise
    
    def _select_search_strategy(self, strategy: SearchStrategy):
        """Select search strategy."""
        if strategy == SearchStrategy.HYBRID:
            # Use multiple strategies in hybrid mode
            return {
                'bayesian': self.search_strategies[SearchStrategy.BAYESIAN],
                'evolutionary': self.search_strategies[SearchStrategy.EVOLUTIONARY],
                'reinforcement': self.search_strategies[SearchStrategy.REINFORCEMENT]
            }
        else:
            return self.search_strategies[strategy]
    
    def _single_objective_search(self, searcher, search_env: Dict[str, Any]) -> TASResult:
        """Perform single-objective search."""
        self.logger.info("🎯 Performing single-objective search")
        
        if isinstance(searcher, dict):  # Hybrid mode
            # Use Bayesian search for single-objective
            searcher = searcher['bayesian']
        
        return searcher.search(
            search_env['train_data'],
            search_env['validation_data'],
            search_env['test_data']
        )
    
    def _multi_objective_search(self, searcher, search_env: Dict[str, Any]) -> TASResult:
        """Perform multi-objective search."""
        self.logger.info("🎯 Performing multi-objective search")
        
        if isinstance(searcher, dict):  # Hybrid mode
            # Use evolutionary search for multi-objective
            searcher = searcher['evolutionary']
        
        return searcher.multi_objective_search(
            search_env['train_data'],
            search_env['validation_data'],
            search_env['test_data']
        )
    
    def _regime_aware_search(self, searcher, search_env: Dict[str, Any]) -> TASResult:
        """Perform regime-aware search."""
        self.logger.info("🎯 Performing regime-aware search")
        
        if not self.config.enable_regime_analysis:
            self.logger.warning("⚠️ Regime analysis not enabled, falling back to single-objective search")
            return self._single_objective_search(searcher, search_env)
        
        # Use regime analyzer for regime-aware search
        regime_analyzer = search_env['regime_analyzer']
        
        # Analyze regimes
        regime_analysis = regime_analyzer.analyze_regimes(
            search_env['train_data'],
            search_env['regime_data']
        )
        
        # Perform regime-specific search
        if isinstance(searcher, dict):  # Hybrid mode
            # Use Bayesian search for regime-aware
            searcher = searcher['bayesian']
        
        return searcher.regime_aware_search(
            search_env['train_data'],
            search_env['validation_data'],
            search_env['test_data'],
            regime_analysis
        )
    
    def _real_time_search(self, searcher, search_env: Dict[str, Any]) -> TASResult:
        """Perform real-time search."""
        self.logger.info("🎯 Performing real-time search")
        
        if not self.config.enable_real_time_adaptation:
            self.logger.warning("⚠️ Real-time adaptation not enabled, falling back to single-objective search")
            return self._single_objective_search(searcher, search_env)
        
        # Use real-time adapter
        real_time_adapter = search_env['real_time_adapter']
        
        return real_time_adapter.real_time_search(
            search_env['train_data'],
            search_env['validation_data'],
            search_env['test_data']
        )
    
    def _continual_search(self, searcher, search_env: Dict[str, Any]) -> TASResult:
        """Perform continual search."""
        self.logger.info("🎯 Performing continual search")
        
        if not self.config.enable_continual_learning:
            self.logger.warning("⚠️ Continual learning not enabled, falling back to single-objective search")
            return self._single_objective_search(searcher, search_env)
        
        # Use meta-learning for continual search
        if self.config.enable_meta_learning:
            meta_learner = search_env['meta_learner']
            return meta_learner.continual_search(
                search_env['train_data'],
                search_env['validation_data'],
                search_env['test_data']
            )
        else:
            return self._single_objective_search(searcher, search_env)
    
    def _post_process_results(self, result: TASResult, search_env: Dict[str, Any]) -> TASResult:
        """Post-process search results."""
        try:
            # Add uncertainty estimates if enabled
            if self.config.enable_uncertainty_estimation and result.best_architecture:
                uncertainty_estimator = search_env['uncertainty_estimator']
                uncertainty = uncertainty_estimator.estimate_uncertainty(
                    result.best_architecture,
                    search_env['validation_data']
                )
                result.uncertainty_estimates = uncertainty
            
            # Add regime analysis if enabled
            if self.config.enable_regime_analysis and result.best_architecture:
                regime_analyzer = search_env['regime_analyzer']
                regime_analysis = regime_analyzer.analyze_architecture_regimes(
                    result.best_architecture,
                    search_env['train_data']
                )
                result.regime_analysis = regime_analysis
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Post-processing failed: {e}")
            self.logger.warning("⚠️ Post-processing failed - returning results without uncertainty estimates and regime analysis")
            return result
    
    def _save_search_results(self, result: TASResult):
        """Save search results."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save result
            result_file = output_dir / "tas_result.json"
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            # Save best architecture if available
            if result.best_architecture and self.config.save_models:
                model_file = output_dir / "best_architecture.json"
                with open(model_file, 'w') as f:
                    json.dump(result.best_architecture.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"💾 Results saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save search results: {e}")
            self.logger.warning("⚠️ Results will only be available in memory - consider checking disk space and permissions")
    
    def adapt_to_new_data(self,
                          new_data: Tuple[np.ndarray, np.ndarray],
                          current_architecture: TreeArchitectureCandidate) -> TreeArchitectureCandidate:
        """
        Adapt current architecture to new data.
        
        Args:
            new_data: New data for adaptation
            current_architecture: Current best architecture
            
        Returns:
            Adapted architecture
        """
        self.logger.info("🔄 Adapting architecture to new data")
        
        try:
            if self.config.enable_meta_learning:
                # Use meta-learning for adaptation
                adapted_architecture = self.meta_learner.adapt_architecture(
                    current_architecture,
                    new_data
                )
                self.logger.info("✅ Architecture adapted using meta-learning")
                return adapted_architecture
            
            elif self.config.enable_real_time_adaptation:
                # Use real-time adaptation
                adapted_architecture = self.real_time_adapter.adapt_architecture(
                    current_architecture,
                    new_data
                )
                self.logger.info("✅ Architecture adapted using real-time adaptation")
                return adapted_architecture
            
            else:
                # Fallback to simple retraining
                self.logger.warning("⚠️ No adaptation method available, returning current architecture")
                return current_architecture
                
        except Exception as e:
            self.logger.error(f"❌ Architecture adaptation failed: {e}")
            return current_architecture
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        try:
            if not self.search_history:
                return {
                    'total_searches': 0,
                    'best_score': 0.0,
                    'average_execution_time': 0.0,
                    'search_strategies_used': [],
                    'optimization_modes_used': []
                }

            # Safely extract scores and times
            valid_scores = [r.best_score for r in self.search_history if hasattr(r, 'best_score') and r.best_score is not None]
            valid_times = [r.execution_time for r in self.search_history if hasattr(r, 'execution_time') and r.execution_time is not None and r.execution_time > 0]
            valid_strategies = [r.search_strategy for r in self.search_history if hasattr(r, 'search_strategy') and r.search_strategy]
            valid_modes = [r.optimization_mode for r in self.search_history if hasattr(r, 'optimization_mode') and r.optimization_mode]

            return {
                'total_searches': len(self.search_history),
                'best_score': max(valid_scores) if valid_scores else 0.0,
                'average_execution_time': np.mean(valid_times) if valid_times else 0.0,
                'search_strategies_used': list(set(valid_strategies)),
                'optimization_modes_used': list(set(valid_modes))
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate search statistics: {e}")
            return {
                'total_searches': len(self.search_history),
                'best_score': 0.0,
                'average_execution_time': 0.0,
                'search_strategies_used': [],
                'optimization_modes_used': [],
                'error': str(e)
            }


# Convenience functions
def create_tas_engine(config: Optional[TASEngineConfig] = None) -> TreeArchitectureSearchEngine:
    """Create a TAS engine with default configuration."""
    if config is None:
        config = TASEngineConfig()
    return TreeArchitectureSearchEngine(config)


def quick_search(train_data: Tuple[np.ndarray, np.ndarray],
                validation_data: Tuple[np.ndarray, np.ndarray],
                test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                search_strategy: SearchStrategy = SearchStrategy.BAYESIAN,
                optimization_mode: OptimizationMode = OptimizationMode.SINGLE_OBJECTIVE) -> TASResult:
    """
    Quick tree architecture search with default settings.
    
    Args:
        train_data: Training data
        validation_data: Validation data
        test_data: Optional test data
        search_strategy: Search strategy
        optimization_mode: Optimization mode
        
    Returns:
        TAS search result
    """
    config = TASEngineConfig(
        search_strategy=search_strategy,
        optimization_mode=optimization_mode,
        enable_meta_learning=False,
        enable_hardware_optimization=False,
        enable_uncertainty_estimation=False,
        enable_regime_analysis=False,
        enable_real_time_adaptation=False
    )
    
    engine = TreeArchitectureSearchEngine(config)
    return engine.search(train_data, validation_data, test_data)