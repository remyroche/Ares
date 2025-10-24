"""
Warm Starting System for Hyperparameter Optimization

This module provides comprehensive warm starting capabilities for HPO,
allowing optimization to continue from previous runs and transfer
knowledge between related optimization tasks.

Enhancement: Warm-starting from previous runs
"""

import numpy as np
import pandas as pd
import json
import pickle
import logging
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
from collections import defaultdict
import hashlib

# Try to import Optuna for warm starting
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class WarmStartConfig:
    """Configuration for warm starting system."""
    
    # Enable warm starting
    enable_warm_start: bool = True
    
    # Warm start sources
    warm_start_file: Optional[str] = None
    warm_start_directory: Optional[str] = None
    warm_start_url: Optional[str] = None
    
    # Knowledge transfer
    enable_knowledge_transfer: bool = True
    transfer_learning_rate: float = 0.1
    similarity_threshold: float = 0.8
    
    # Parameter mapping
    enable_parameter_mapping: bool = True
    parameter_mapping_file: Optional[str] = None
    
    # Performance tracking
    track_warm_start_performance: bool = True
    save_warm_start_log: bool = True
    warm_start_log_file: str = "warm_start_log.json"
    
    # Caching
    enable_warm_start_cache: bool = True
    cache_directory: str = "warm_start_cache"
    cache_expiry_days: int = 30


@dataclass
class WarmStartData:
    """Data structure for warm start information."""
    
    # Basic information
    source_file: str
    timestamp: float
    strategy: str
    model_type: str
    dataset_hash: str
    
    # Optimization results
    best_params: Dict[str, Any]
    best_score: float
    trial_results: List[Dict[str, Any]]
    
    # Search space information
    search_space: Dict[str, Any]
    search_space_hash: str
    
    # Performance metrics
    optimization_time: float
    n_trials: int
    convergence_rate: float
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterMapping:
    """Mapping between different parameter spaces."""
    
    source_params: Dict[str, Any]
    target_params: Dict[str, Any]
    mapping_rules: Dict[str, str]
    confidence: float
    similarity_score: float


class WarmStartManager:
    """Manages warm starting for hyperparameter optimization."""
    
    def __init__(self, config: WarmStartConfig):
        self.config = config
        self.warm_start_data: List[WarmStartData] = []
        self.parameter_mappings: List[ParameterMapping] = []
        self.warm_start_cache: Dict[str, Any] = {}
        self.performance_log: List[Dict[str, Any]] = []
        
        # Load existing warm start data
        self._load_warm_start_data()
        
        # Load parameter mappings
        self._load_parameter_mappings()
        
        logger.info(f"Warm start manager initialized with {len(self.warm_start_data)} data sources")
    
    def _load_warm_start_data(self):
        """Load warm start data from various sources."""
        # Load from file
        if self.config.warm_start_file:
            self._load_from_file(self.config.warm_start_file)
        
        # Load from directory
        if self.config.warm_start_directory:
            self._load_from_directory(self.config.warm_start_directory)
        
        # Load from URL (if implemented)
        if self.config.warm_start_url:
            self._load_from_url(self.config.warm_start_url)
    
    def _load_from_file(self, filepath: str):
        """Load warm start data from a single file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    warm_start_data = self._parse_warm_start_data(item)
                    if warm_start_data:
                        self.warm_start_data.append(warm_start_data)
            else:
                warm_start_data = self._parse_warm_start_data(data)
                if warm_start_data:
                    self.warm_start_data.append(warm_start_data)
            
            logger.info(f"Loaded warm start data from {filepath}")
        except Exception as e:
            logger.warning(f"Failed to load warm start data from {filepath}: {e}")
    
    def _load_from_directory(self, directory: str):
        """Load warm start data from a directory."""
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                logger.warning(f"Warm start directory {directory} does not exist")
                return
            
            # Look for JSON files
            for file_path in directory_path.glob("*.json"):
                self._load_from_file(str(file_path))
            
            # Look for pickle files
            for file_path in directory_path.glob("*.pkl"):
                self._load_from_pickle(str(file_path))
            
            logger.info(f"Loaded warm start data from directory {directory}")
        except Exception as e:
            logger.warning(f"Failed to load warm start data from directory {directory}: {e}")
    
    def _load_from_pickle(self, filepath: str):
        """Load warm start data from pickle file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, WarmStartData):
                self.warm_start_data.append(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, WarmStartData):
                        self.warm_start_data.append(item)
            
            logger.info(f"Loaded warm start data from pickle file {filepath}")
        except Exception as e:
            logger.warning(f"Failed to load warm start data from pickle file {filepath}: {e}")
    
    def _load_from_url(self, url: str):
        """Load warm start data from URL (placeholder for future implementation)."""
        logger.warning("URL loading not implemented yet")
    
    def _parse_warm_start_data(self, data: Dict[str, Any]) -> Optional[WarmStartData]:
        """Parse warm start data from dictionary."""
        try:
            return WarmStartData(
                source_file=data.get('source_file', 'unknown'),
                timestamp=data.get('timestamp', time.time()),
                strategy=data.get('strategy', 'unknown'),
                model_type=data.get('model_type', 'unknown'),
                dataset_hash=data.get('dataset_hash', ''),
                best_params=data.get('best_params', {}),
                best_score=data.get('best_score', 0.0),
                trial_results=data.get('trial_results', []),
                search_space=data.get('search_space', {}),
                search_space_hash=data.get('search_space_hash', ''),
                optimization_time=data.get('optimization_time', 0.0),
                n_trials=data.get('n_trials', 0),
                convergence_rate=data.get('convergence_rate', 0.0),
                metadata=data.get('metadata', {})
            )
        except Exception as e:
            logger.warning(f"Failed to parse warm start data: {e}")
            return None
    
    def _load_parameter_mappings(self):
        """Load parameter mappings from file."""
        if not self.config.parameter_mapping_file:
            return
        
        try:
            with open(self.config.parameter_mapping_file, 'r') as f:
                mappings_data = json.load(f)
            
            for mapping_data in mappings_data:
                mapping = ParameterMapping(
                    source_params=mapping_data.get('source_params', {}),
                    target_params=mapping_data.get('target_params', {}),
                    mapping_rules=mapping_data.get('mapping_rules', {}),
                    confidence=mapping_data.get('confidence', 0.0),
                    similarity_score=mapping_data.get('similarity_score', 0.0)
                )
                self.parameter_mappings.append(mapping)
            
            logger.info(f"Loaded {len(self.parameter_mappings)} parameter mappings")
        except Exception as e:
            logger.warning(f"Failed to load parameter mappings: {e}")
    
    def add_warm_start_data(self, warm_start_data: WarmStartData):
        """Add new warm start data."""
        self.warm_start_data.append(warm_start_data)
        
        # Save to cache if enabled
        if self.config.enable_warm_start_cache:
            self._save_to_cache(warm_start_data)
        
        logger.info(f"Added warm start data for {warm_start_data.model_type} model")
    
    def find_similar_optimizations(self, model_type: str, search_space: Dict[str, Any],
                                 dataset_hash: str, similarity_threshold: float = 0.8) -> List[WarmStartData]:
        """Find similar optimizations for warm starting."""
        similar_data = []
        search_space_hash = self._hash_search_space(search_space)
        
        for data in self.warm_start_data:
            similarity_score = self._calculate_similarity(
                data, model_type, search_space_hash, dataset_hash
            )
            
            if similarity_score >= similarity_threshold:
                data.similarity_score = similarity_score
                similar_data.append(data)
        
        # Sort by similarity score
        similar_data.sort(key=lambda x: x.similarity_score, reverse=True)
        
        logger.info(f"Found {len(similar_data)} similar optimizations")
        return similar_data
    
    def _calculate_similarity(self, data: WarmStartData, model_type: str,
                            search_space_hash: str, dataset_hash: str) -> float:
        """Calculate similarity between warm start data and current optimization."""
        similarity_score = 0.0
        
        # Model type similarity (40% weight)
        if data.model_type == model_type:
            similarity_score += 0.4
        elif self._are_models_similar(data.model_type, model_type):
            similarity_score += 0.2
        
        # Search space similarity (30% weight)
        if data.search_space_hash == search_space_hash:
            similarity_score += 0.3
        else:
            search_space_similarity = self._calculate_search_space_similarity(
                data.search_space, search_space_hash
            )
            similarity_score += 0.3 * search_space_similarity
        
        # Dataset similarity (20% weight)
        if data.dataset_hash == dataset_hash:
            similarity_score += 0.2
        else:
            dataset_similarity = self._calculate_dataset_similarity(
                data.dataset_hash, dataset_hash
            )
            similarity_score += 0.2 * dataset_similarity
        
        # Strategy similarity (10% weight)
        if data.strategy in ['bayesian', 'tpe']:
            similarity_score += 0.1
        
        return similarity_score
    
    def _are_models_similar(self, model1: str, model2: str) -> bool:
        """Check if two model types are similar."""
        similar_models = {
            'lightgbm': ['xgboost', 'catboost'],
            'xgboost': ['lightgbm', 'catboost'],
            'catboost': ['lightgbm', 'xgboost'],
            'random_forest': ['extra_trees', 'gradient_boosting'],
            'ridge': ['lasso', 'elastic_net'],
            'lasso': ['ridge', 'elastic_net'],
            'elastic_net': ['ridge', 'lasso']
        }
        
        return model2 in similar_models.get(model1, [])
    
    def _calculate_search_space_similarity(self, search_space: Dict[str, Any],
                                         target_hash: str) -> float:
        """Calculate similarity between search spaces."""
        # This is a simplified implementation
        # In practice, you'd compare parameter ranges and types
        return 0.5  # Placeholder
    
    def _calculate_dataset_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between datasets."""
        # This is a simplified implementation
        # In practice, you'd compare dataset characteristics
        return 0.5  # Placeholder
    
    def _hash_search_space(self, search_space: Dict[str, Any]) -> str:
        """Create hash of search space for comparison."""
        search_space_str = json.dumps(search_space, sort_keys=True)
        return hashlib.md5(search_space_str.encode()).hexdigest()
    
    def create_warm_start_parameters(self, similar_data: List[WarmStartData],
                                   target_search_space: Dict[str, Any],
                                   n_parameters: int = 10) -> List[Dict[str, Any]]:
        """Create warm start parameters from similar optimizations."""
        warm_start_params = []
        
        for data in similar_data[:n_parameters]:
            # Map parameters to target search space
            mapped_params = self._map_parameters(
                data.best_params, data.search_space, target_search_space
            )
            
            if mapped_params:
                warm_start_params.append(mapped_params)
        
        logger.info(f"Created {len(warm_start_params)} warm start parameters")
        return warm_start_params
    
    def _map_parameters(self, source_params: Dict[str, Any],
                       source_search_space: Dict[str, Any],
                       target_search_space: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map parameters from source to target search space."""
        mapped_params = {}
        
        for param_name, param_value in source_params.items():
            if param_name in target_search_space:
                # Direct mapping
                mapped_params[param_name] = param_value
            else:
                # Try to find similar parameter
                similar_param = self._find_similar_parameter(
                    param_name, target_search_space
                )
                if similar_param:
                    mapped_params[similar_param] = param_value
        
        return mapped_params if mapped_params else None
    
    def _find_similar_parameter(self, param_name: str,
                              target_search_space: Dict[str, Any]) -> Optional[str]:
        """Find similar parameter in target search space."""
        # Common parameter name mappings
        parameter_mappings = {
            'n_estimators': ['n_trees', 'num_trees', 'max_trees'],
            'max_depth': ['max_tree_depth', 'tree_depth'],
            'learning_rate': ['eta', 'step_size', 'lr'],
            'subsample': ['sample_rate', 'bagging_fraction'],
            'colsample_bytree': ['feature_fraction', 'colsample_by_tree'],
            'min_child_samples': ['min_data_in_leaf', 'min_samples_leaf'],
            'reg_alpha': ['l1_regularization', 'l1_penalty'],
            'reg_lambda': ['l2_regularization', 'l2_penalty']
        }
        
        similar_names = parameter_mappings.get(param_name, [])
        for similar_name in similar_names:
            if similar_name in target_search_space:
                return similar_name
        
        return None
    
    def create_optuna_warm_start(self, similar_data: List[WarmStartData],
                               target_search_space: Dict[str, Any]) -> Optional[optuna.Study]:
        """Create Optuna study with warm start data."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available for warm starting")
            return None
        
        try:
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
            # Add warm start trials
            for i, data in enumerate(similar_data[:20]):  # Limit to 20 trials
                mapped_params = self._map_parameters(
                    data.best_params, data.search_space, target_search_space
                )
                
                if mapped_params:
                    # Create trial
                    trial = optuna.trial.create_trial(
                        params=mapped_params,
                        distributions=target_search_space,
                        value=data.best_score
                    )
                    study.add_trial(trial)
            
            logger.info(f"Created Optuna study with {len(study.trials)} warm start trials")
            return study
            
        except Exception as e:
            logger.error(f"Failed to create Optuna warm start: {e}")
            return None
    
    def save_warm_start_data(self, filepath: str, warm_start_data: WarmStartData):
        """Save warm start data to file."""
        try:
            data_dict = {
                'source_file': warm_start_data.source_file,
                'timestamp': warm_start_data.timestamp,
                'strategy': warm_start_data.strategy,
                'model_type': warm_start_data.model_type,
                'dataset_hash': warm_start_data.dataset_hash,
                'best_params': warm_start_data.best_params,
                'best_score': warm_start_data.best_score,
                'trial_results': warm_start_data.trial_results,
                'search_space': warm_start_data.search_space,
                'search_space_hash': warm_start_data.search_space_hash,
                'optimization_time': warm_start_data.optimization_time,
                'n_trials': warm_start_data.n_trials,
                'convergence_rate': warm_start_data.convergence_rate,
                'metadata': warm_start_data.metadata
            }
            
            with open(filepath, 'w') as f:
                json.dump(data_dict, f, indent=2)
            
            logger.info(f"Saved warm start data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save warm start data: {e}")
    
    def _save_to_cache(self, warm_start_data: WarmStartData):
        """Save warm start data to cache."""
        try:
            cache_dir = Path(self.config.cache_directory)
            cache_dir.mkdir(exist_ok=True)
            
            cache_file = cache_dir / f"{warm_start_data.model_type}_{warm_start_data.dataset_hash}.json"
            self.save_warm_start_data(str(cache_file), warm_start_data)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def get_warm_start_summary(self) -> Dict[str, Any]:
        """Get summary of warm start data."""
        if not self.warm_start_data:
            return {'message': 'No warm start data available'}
        
        summary = {
            'total_data_sources': len(self.warm_start_data),
            'model_types': list(set(data.model_type for data in self.warm_start_data)),
            'strategies': list(set(data.strategy for data in self.warm_start_data)),
            'avg_best_score': np.mean([data.best_score for data in self.warm_start_data]),
            'avg_optimization_time': np.mean([data.optimization_time for data in self.warm_start_data]),
            'avg_n_trials': np.mean([data.n_trials for data in self.warm_start_data])
        }
        
        return summary


def create_warm_start_manager(
    enable_warm_start: bool = True,
    warm_start_file: Optional[str] = None,
    **kwargs
) -> WarmStartManager:
    """Create warm start manager with default settings."""
    config = WarmStartConfig(
        enable_warm_start=enable_warm_start,
        warm_start_file=warm_start_file,
        **kwargs
    )
    return WarmStartManager(config)


def create_warm_start_data_from_hpo_result(
    hpo_result: Any,  # HPOResult type
    model_type: str,
    strategy: str,
    search_space: Dict[str, Any],
    dataset_hash: str,
    source_file: str = "unknown"
) -> WarmStartData:
    """Create warm start data from HPO result."""
    return WarmStartData(
        source_file=source_file,
        timestamp=time.time(),
        strategy=strategy,
        model_type=model_type,
        dataset_hash=dataset_hash,
        best_params=hpo_result.best_params,
        best_score=hpo_result.best_score,
        trial_results=hpo_result.trial_results,
        search_space=search_space,
        search_space_hash=hashlib.md5(json.dumps(search_space, sort_keys=True).encode()).hexdigest(),
        optimization_time=hpo_result.optimization_time,
        n_trials=hpo_result.n_trials,
        convergence_rate=getattr(hpo_result, 'convergence_rate', 0.0),
        metadata=getattr(hpo_result, 'metadata', {})
    )