"""
Regime Artifact Schema - Standardized artifact format for regime detection components.

This module provides base classes and schemas for consistent artifact handling
across regime clustering, base models training, and ensemble training.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from src.utils.tprint import tprint

# Import standardized regime extractor for consistent extraction logic
from src.utils.ml_common.data.standardized_regime_extractor import (
    StandardizedRegimeExtractor, extract_regime_labels_standardized, RegimeLabelExtractionError
)


@dataclass
class RegimeLabelsArtifact:
    """
    Standardized format for regime labels/cluster assignments.
    
    This ensures consistent handling of regime labels across all components.
    """
    cluster_assignments: np.ndarray
    n_regimes: int
    regime_distribution: Dict[int, int]
    clustering_method: str
    clustering_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            'cluster_assignments': self.cluster_assignments,
            'n_regimes': self.n_regimes,
            'regime_distribution': self.regime_distribution,
            'clustering_method': self.clustering_method,
            'clustering_params': self.clustering_params,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegimeLabelsArtifact':
        """Create from dictionary format."""
        # Handle numpy array serialization
        assignments = data.get('cluster_assignments')
        if isinstance(assignments, str):
            # Parse string representation: "[2 2 2 ... 4 6 6]"
            clean_str = assignments.strip('[]')
            assignments = np.array([int(x) for x in clean_str.split() if x.strip()])
        elif isinstance(assignments, list):
            assignments = np.array(assignments)
        elif not isinstance(assignments, np.ndarray):
            raise ValueError(f"Invalid cluster_assignments type: {type(assignments)}")
        
        return cls(
            cluster_assignments=assignments,
            n_regimes=data.get('n_regimes', len(np.unique(assignments))),
            regime_distribution=data.get('regime_distribution', {}),
            clustering_method=data.get('clustering_method', 'unknown'),
            clustering_params=data.get('clustering_params', {}),
            metadata=data.get('metadata', {})
        )
    
    def validate(self) -> bool:
        """Validate the artifact structure."""
        try:
            assert isinstance(self.cluster_assignments, np.ndarray), "cluster_assignments must be numpy array"
            assert len(self.cluster_assignments) > 0, "cluster_assignments cannot be empty"
            assert self.n_regimes > 0, "n_regimes must be positive"
            assert self.n_regimes == len(np.unique(self.cluster_assignments)), "n_regimes mismatch with unique assignments"
            
            # Validate regime distribution
            for regime_id, count in self.regime_distribution.items():
                assert count > 0, f"Regime {regime_id} has invalid count: {count}"
            
            tprint(f"✅ [ARTIFACT_SCHEMA] RegimeLabelsArtifact validation passed", color="green")
            return True
        except AssertionError as e:
            tprint(f"❌ [ARTIFACT_SCHEMA] RegimeLabelsArtifact validation failed: {e}", color="red")
            return False


@dataclass
class FeatureContract:
    """
    Feature contract defining expected features for a model.
    
    This ensures consistency between training and inference.
    """
    feature_names: List[str]
    feature_count: int
    feature_types: Dict[str, str]  # feature_name -> type ('base_prediction', 'probability', 'uncertainty', 'meta')
    expected_shape: tuple
    scaler_params: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate contract on creation."""
        if self.feature_count != len(self.feature_names):
            raise ValueError(f"Feature count mismatch: {self.feature_count} != {len(self.feature_names)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            'feature_names': self.feature_names,
            'feature_count': self.feature_count,
            'feature_types': self.feature_types,
            'expected_shape': self.expected_shape,
            'scaler_params': self.scaler_params,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureContract':
        """Create from dictionary format."""
        return cls(
            feature_names=data['feature_names'],
            feature_count=data['feature_count'],
            feature_types=data['feature_types'],
            expected_shape=tuple(data['expected_shape']),
            scaler_params=data.get('scaler_params'),
            metadata=data.get('metadata', {})
        )
    
    def validate_features(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> bool:
        """
        Validate that features match the contract.
        
        Args:
            X: Feature matrix to validate
            feature_names: Optional list of feature names to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        try:
            # Check shape
            if X.shape[1] != self.feature_count:
                raise ValueError(
                    f"❌ Feature count mismatch: expected {self.feature_count}, got {X.shape[1]}"
                )
            
            # Check feature names if provided
            if feature_names is not None:
                if len(feature_names) != self.feature_count:
                    raise ValueError(
                        f"❌ Feature name count mismatch: expected {self.feature_count}, got {len(feature_names)}"
                    )
                
                missing_features = set(self.feature_names) - set(feature_names)
                if missing_features:
                    raise ValueError(
                        f"❌ Missing expected features: {missing_features}"
                    )
                
                extra_features = set(feature_names) - set(self.feature_names)
                if extra_features:
                    tprint(
                        f"⚠️ [FEATURE_CONTRACT] Extra features detected: {extra_features}",
                        color="yellow"
                    )
            
            tprint(
                f"✅ [FEATURE_CONTRACT] Validation passed: {X.shape} matches contract {self.expected_shape}",
                color="green"
            )
            return True
            
        except ValueError as e:
            tprint(f"❌ [FEATURE_CONTRACT] Validation failed: {e}", color="red")
            raise


@dataclass
class BaseModelContract:
    """
    Contract for base models defining their outputs and requirements.
    """
    model_name: str
    model_type: str  # 'classifier', 'regressor', 'ensemble'
    output_type: str  # 'probabilities', 'classes', 'both'
    n_classes: int
    feature_contract: FeatureContract
    training_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'output_type': self.output_type,
            'n_classes': self.n_classes,
            'feature_contract': self.feature_contract.to_dict(),
            'training_timestamp': self.training_timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModelContract':
        """Create from dictionary format."""
        return cls(
            model_name=data['model_name'],
            model_type=data['model_type'],
            output_type=data['output_type'],
            n_classes=data['n_classes'],
            feature_contract=FeatureContract.from_dict(data['feature_contract']),
            training_timestamp=data.get('training_timestamp', datetime.now().isoformat()),
            metadata=data.get('metadata', {})
        )
    
    def is_ensemble_model(self) -> bool:
        """Check if this is an ensemble/meta-learner model."""
        return (
            self.model_type == 'ensemble' or 
            'ensemble' in self.model_name.lower() or
            'stacker' in self.model_name.lower() or
            'meta' in self.model_name.lower()
        )
    
    def is_base_model(self) -> bool:
        """Check if this is a base model (not ensemble/meta-learner)."""
        return not self.is_ensemble_model()


@dataclass
class RegimeModelsArtifact:
    """
    Standardized format for base regime models.
    """
    models: Dict[str, Any]  # model_name -> model object
    model_contracts: Dict[str, BaseModelContract]  # model_name -> contract
    scaler: Any
    feature_names: List[str]
    training_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_base_models(self) -> Dict[str, Any]:
        """Get only base models (exclude ensemble/meta-learners)."""
        base_models = {}
        for name, model in self.models.items():
            if name in self.model_contracts:
                contract = self.model_contracts[name]
                if contract.is_base_model():
                    base_models[name] = model
            elif not self._is_likely_ensemble(name):
                # Fallback for models without contracts
                base_models[name] = model
        
        tprint(
            f"📊 [REGIME_MODELS_ARTIFACT] Extracted {len(base_models)} base models from {len(self.models)} total models",
            color="blue"
        )
        return base_models
    
    def get_ensemble_models(self) -> Dict[str, Any]:
        """Get only ensemble/meta-learner models."""
        ensemble_models = {}
        for name, model in self.models.items():
            if name in self.model_contracts:
                contract = self.model_contracts[name]
                if contract.is_ensemble_model():
                    ensemble_models[name] = model
            elif self._is_likely_ensemble(name):
                # Fallback for models without contracts
                ensemble_models[name] = model
        
        tprint(
            f"📊 [REGIME_MODELS_ARTIFACT] Extracted {len(ensemble_models)} ensemble models from {len(self.models)} total models",
            color="blue"
        )
        return ensemble_models
    
    def _is_likely_ensemble(self, model_name: str) -> bool:
        """Heuristic to identify ensemble models by name."""
        ensemble_keywords = ['stacker', 'ensemble', 'meta', 'voting', 'blender']
        return any(keyword in model_name.lower() for keyword in ensemble_keywords)
    
    def validate_models(self) -> bool:
        """Validate all models in the artifact."""
        try:
            valid_count = 0
            for name, model in self.models.items():
                if model is None:
                    tprint(f"⚠️ [REGIME_MODELS_ARTIFACT] Model '{name}' is None", color="yellow")
                    continue
                
                if not hasattr(model, 'predict'):
                    tprint(f"⚠️ [REGIME_MODELS_ARTIFACT] Model '{name}' missing predict method", color="yellow")
                    continue
                
                valid_count += 1
            
            if valid_count == 0:
                raise ValueError("No valid models found in artifact")
            
            tprint(
                f"✅ [REGIME_MODELS_ARTIFACT] Validation passed: {valid_count}/{len(self.models)} models valid",
                color="green"
            )
            return True
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS_ARTIFACT] Validation failed: {e}", color="red")
            return False


@dataclass
class RegimeEnsembleArtifact:
    """
    Standardized format for regime ensemble model.
    """
    ensemble_model: Any  # The trained meta-learner
    base_model_contracts: Dict[str, BaseModelContract]  # Contracts for base models used
    ensemble_contract: BaseModelContract  # Contract for the ensemble itself
    feature_names: List[str]  # Meta-feature names
    training_metrics: Dict[str, Any]
    calibration_info: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (excluding model objects)."""
        return {
            'base_model_contracts': {k: v.to_dict() for k, v in self.base_model_contracts.items()},
            'ensemble_contract': self.ensemble_contract.to_dict(),
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'calibration_info': self.calibration_info,
            'metadata': self.metadata
        }
    
    def validate(self) -> bool:
        """Validate the ensemble artifact."""
        try:
            assert self.ensemble_model is not None, "Ensemble model is None"
            assert hasattr(self.ensemble_model, 'predict'), "Ensemble model missing predict method"
            assert hasattr(self.ensemble_model, 'predict_proba'), "Ensemble model missing predict_proba method"
            assert len(self.base_model_contracts) > 0, "No base model contracts defined"
            assert len(self.feature_names) > 0, "No feature names defined"
            
            # Validate ensemble contract
            assert self.ensemble_contract.is_ensemble_model(), "Ensemble contract not marked as ensemble"
            
            tprint("✅ [ENSEMBLE_ARTIFACT] Validation passed", color="green")
            return True
            
        except AssertionError as e:
            tprint(f"❌ [ENSEMBLE_ARTIFACT] Validation failed: {e}", color="red")
            return False


class RegimeArtifactExtractor:
    """
    Utility class to extract standardized artifacts from pipeline state.
    
    Provides backward compatibility with old artifact formats while enforcing
    the new standardized format.
    """
    
    @staticmethod
    def extract_regime_labels(
        pipeline_state: Dict[str, Any],
        component_name: str = "ARTIFACT_EXTRACTOR",
        min_samples: int = 10,
        min_regimes: int = 2,
        preferred_method: Optional[str] = None
    ) -> Optional[RegimeLabelsArtifact]:
        """
        Extract regime labels from pipeline state using standardized extractor with metadata enrichment.
        
        This method uses the StandardizedRegimeExtractor for consistent extraction logic
        while wrapping the result in a RegimeLabelsArtifact with metadata.
        
        NOTE: Multiple clustering methods are supported during testing phase. Eventually,
        when a single method is chosen, set `preferred_method` to skip fallbacks.
        
        Args:
            pipeline_state: Pipeline state dictionary
            component_name: Name for logging
            min_samples: Minimum number of samples required
            min_regimes: Minimum number of unique regimes required
            preferred_method: Optional preferred clustering method ('gmm', 'hmm', 'optimal', etc.)
                             If specified, will only look for that method's results (faster, cleaner)
            
        Returns:
            RegimeLabelsArtifact with metadata or None if not found
        """
        tprint(f"🔍 [{component_name}] Extracting regime labels with standardized extractor", color="cyan")
        
        artifacts = pipeline_state.get('artifacts', {})
        
        # Try standardized format first (pre-wrapped artifact)
        if 'regime_labels_artifact' in artifacts:
            tprint(f"✅ [{component_name}] Found pre-wrapped regime_labels_artifact", color="green")
            return RegimeLabelsArtifact.from_dict(artifacts['regime_labels_artifact'])
        
        # Use StandardizedRegimeExtractor for consistent extraction
        try:
            tprint(f"🔧 [{component_name}] Using StandardizedRegimeExtractor for label extraction", color="blue")
            
            # Extract labels using the standardized extractor
            cluster_assignments = extract_regime_labels_standardized(
                pipeline_state,
                min_samples=min_samples,
                min_regimes=min_regimes
            )
            
            # Calculate basic statistics
            unique_regimes, regime_counts = np.unique(cluster_assignments, return_counts=True)
            n_regimes = len(unique_regimes)
            regime_distribution = dict(zip(unique_regimes.astype(int), regime_counts.astype(int)))
            # (n_samples, n_regimes) DataFrame of posterior probabilities (soft labels)
            probabilities: Optional[pd.DataFrame] = None
    
            tprint(f"✅ [{component_name}] Extracted {len(cluster_assignments)} labels with {n_regimes} regimes", color="green")
            tprint(f"📊 [{component_name}] Regime distribution: {regime_distribution}", color="blue")
            
            # Extract metadata from pipeline artifacts for enrichment
            clustering_method = "unknown"
            clustering_params = {}
            metadata = {}
            
            # If preferred method is specified, only look for that method (production mode)
            if preferred_method:
                tprint(f"🎯 [{component_name}] Looking for preferred method: {preferred_method}", color="blue")
                metadata_extracted = RegimeArtifactExtractor._extract_metadata_for_method(
                    artifacts, preferred_method, component_name
                )
                if metadata_extracted:
                    clustering_method, clustering_params, metadata = metadata_extracted
                else:
                    tprint(f"⚠️ [{component_name}] Preferred method '{preferred_method}' not found, using 'unknown'", color="yellow")
            
            # Otherwise, try hierarchical fallback (testing/transition mode)
            else:
                tprint(f"🔍 [{component_name}] Testing mode: trying multiple clustering methods", color="blue")
                
                # Try to get metadata from optimal_regime_clustering_result
                optimal_result = artifacts.get('optimal_regime_clustering_result', {})
                if optimal_result:
                    clustering_method = optimal_result.get('method', 'unknown')
                    clustering_params = optimal_result.get('params', {})
                    metadata = optimal_result.get('metadata', {})
                    tprint(f"📋 [{component_name}] Enriched with metadata from optimal_regime_clustering_result", color="blue")
                
                # Fallback to regime_clustering_result for metadata
                elif 'regime_clustering_result' in artifacts:
                    clustering_result = artifacts['regime_clustering_result']
                    clustering_method = clustering_result.get('method', 'unknown')
                    clustering_params = clustering_result.get('params', {})
                    metadata = clustering_result.get('metadata', {})
                    tprint(f"📋 [{component_name}] Enriched with metadata from regime_clustering_result", color="blue")
                
                # Try GMM/HMM specific results for metadata
                elif 'gmm_regime_discovery_result' in artifacts:
                    gmm_result = artifacts['gmm_regime_discovery_result']
                    clustering_method = 'gmm'
                    clustering_params = gmm_result.get('params', {})
                    metadata = gmm_result.get('metadata', {})
                    tprint(f"📋 [{component_name}] Enriched with metadata from GMM discovery", color="blue")
                
                elif 'hmm_regime_discovery_result' in artifacts:
                    hmm_result = artifacts['hmm_regime_discovery_result']
                    clustering_method = 'hmm'
                    clustering_params = hmm_result.get('params', {})
                    metadata = hmm_result.get('metadata', {})
                    tprint(f"📋 [{component_name}] Enriched with metadata from HMM discovery", color="blue")
            
            # Create enriched artifact
            artifact = RegimeLabelsArtifact(
                cluster_assignments=cluster_assignments,
                n_regimes=n_regimes,
                regime_distribution=regime_distribution,
                clustering_method=clustering_method,
                clustering_params=clustering_params,
                metadata=metadata
            )
            
            tprint(f"✅ [{component_name}] Created RegimeLabelsArtifact with method: {clustering_method}", color="green")
            return artifact
            
        except RegimeLabelExtractionError as e:
            tprint(f"❌ [{component_name}] Standardized extraction failed: {e}", color="red")
            tprint(f"💡 [{component_name}] Ensure regime discovery step has been executed", color="yellow")
            return None
        
        except Exception as e:
            tprint(f"❌ [{component_name}] Unexpected error during extraction: {e}", color="red")
            return None
    
    @staticmethod
    def _extract_metadata_for_method(
        artifacts: Dict[str, Any],
        method: str,
        component_name: str
    ) -> Optional[tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """
        Extract metadata for a specific clustering method (production mode).
        
        Args:
            artifacts: Pipeline artifacts dictionary
            method: Clustering method to extract ('gmm', 'hmm', 'optimal', 'regime_clustering')
            component_name: Name for logging
            
        Returns:
            Tuple of (clustering_method, clustering_params, metadata) or None if not found
        """
        method = method.lower()
        
        # Map method names to artifact keys
        method_mapping = {
            'gmm': 'gmm_regime_discovery_result',
            'hmm': 'hmm_regime_discovery_result',
            'optimal': 'optimal_regime_clustering_result',
            'regime_clustering': 'regime_clustering_result'
        }
        
        artifact_key = method_mapping.get(method)
        if not artifact_key:
            tprint(f"⚠️ [{component_name}] Unknown method: {method}", color="yellow")
            return None
        
        result = artifacts.get(artifact_key, {})
        if not result:
            tprint(f"⚠️ [{component_name}] No results found for method: {method}", color="yellow")
            return None
        
        # Extract metadata based on artifact structure
        clustering_method = result.get('method', method)
        clustering_params = result.get('params', {})
        metadata = result.get('metadata', {})
        
        tprint(f"✅ [{component_name}] Extracted metadata for method: {method}", color="green")
        return (clustering_method, clustering_params, metadata)
    
    @staticmethod
    def extract_base_models(
        pipeline_state: Dict[str, Any],
        component_name: str = "ARTIFACT_EXTRACTOR"
    ) -> Optional[RegimeModelsArtifact]:
        """
        Extract base models from pipeline state.
        
        Args:
            pipeline_state: Pipeline state dictionary
            component_name: Name for logging
            
        Returns:
            RegimeModelsArtifact or None if not found
        """
        tprint(f"🔍 [{component_name}] Extracting base models from pipeline state", color="cyan")
        
        artifacts = pipeline_state.get('artifacts', {})
        regime_models_result = artifacts.get('regime_models_training_result', {})
        
        if not regime_models_result:
            tprint(f"❌ [{component_name}] No regime_models_training_result found", color="red")
            return None
        
        models = regime_models_result.get('models', {})
        if not models:
            tprint(f"❌ [{component_name}] No models found in regime_models_training_result", color="red")
            return None
        
        # Extract or create contracts
        model_contracts = {}
        if 'model_contracts' in regime_models_result:
            for name, contract_dict in regime_models_result['model_contracts'].items():
                model_contracts[name] = BaseModelContract.from_dict(contract_dict)
        
        artifact = RegimeModelsArtifact(
            models=models,
            model_contracts=model_contracts,
            scaler=regime_models_result.get('scaler'),
            feature_names=regime_models_result.get('feature_names', []),
            training_metrics=regime_models_result.get('training_metrics', {}),
            metadata=regime_models_result.get('metadata', {})
        )
        
        tprint(
            f"✅ [{component_name}] Extracted {len(models)} models from pipeline state",
            color="green"
        )
        return artifact

