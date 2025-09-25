"""
Model Manager

Manages model lifecycle including deployment, versioning, monitoring,
and rollback capabilities for the NAS-TAS system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import shutil
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import tprint for comprehensive logging
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
        tprint_success, tprint_progress, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError:
    # Fallback function if tprint is not available
    def tprint(message: str, color: str = "white", **kwargs):
        print(f"[MODEL_MANAGER] {message}")
    def tprint_debug(message: str, **kwargs):
        print(f"[DEBUG] {message}")
    def tprint_info(message: str, **kwargs):
        print(f"[INFO] {message}")
    def tprint_warning(message: str, **kwargs):
        print(f"[WARNING] {message}")
    def tprint_error(message: str, **kwargs):
        print(f"[ERROR] {message}")
    def tprint_success(message: str, **kwargs):
        print(f"[SUCCESS] {message}")
    def tprint_progress(message: str, **kwargs):
        print(f"[PROGRESS] {message}")
    def tprint_performance(message: str, **kwargs):
        print(f"[PERFORMANCE] {message}")
    def tprint_timer(message: str, **kwargs):
        print(f"[TIMER] {message}")
    TPRINT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status."""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    MONITORING = "monitoring"
    DEGRADED = "degraded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentStrategy(Enum):
    """Model deployment strategies."""
    IMMEDIATE = "immediate"      # Deploy immediately
    GRADUAL = "gradual"         # Gradual rollout
    A_B_TESTING = "a_b_testing" # A/B testing
    CANARY = "canary"           # Canary deployment


@dataclass
class ModelManagerConfig:
    """Configuration for model manager."""
    
    # Model storage
    model_storage_path: str = "models/storage"
    model_versions_path: str = "models/versions"
    model_metadata_path: str = "models/metadata"
    
    # Deployment settings
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.IMMEDIATE
    enable_auto_deployment: bool = True
    deployment_threshold: float = 0.7  # Minimum performance for deployment
    
    # Versioning
    enable_model_versioning: bool = True
    max_versions_per_model: int = 10
    version_naming_scheme: str = "semantic"  # "semantic", "timestamp", "incremental"
    
    # Monitoring
    enable_model_monitoring: bool = True
    monitoring_frequency: int = 100  # Check every N predictions
    performance_threshold: float = 0.6  # Minimum performance threshold
    degradation_threshold: float = 0.1  # Performance degradation threshold
    
    # Rollback
    enable_auto_rollback: bool = True
    rollback_threshold: float = 0.5  # Performance threshold for rollback
    max_rollback_attempts: int = 3
    
    # Model lifecycle
    enable_model_retirement: bool = True
    retirement_age_days: int = 30
    retirement_performance_threshold: float = 0.4
    
    # Advanced features
    enable_model_ensembling: bool = True
    enable_dynamic_loading: bool = True
    enable_model_compression: bool = False
    compression_ratio: float = 0.5
    
    # Logging and audit
    enable_audit_logging: bool = True
    audit_log_path: str = "logs/model_audit.log"
    enable_performance_logging: bool = True
    performance_log_path: str = "logs/model_performance.log"


@dataclass
class ModelMetadata:
    """Metadata for a model."""
    
    # Basic information
    model_id: str
    model_type: str
    regime_id: int
    version: str
    
    # Performance metrics
    training_performance: Dict[str, float]
    validation_performance: Dict[str, float]
    test_performance: Dict[str, float]
    
    # Model characteristics
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    model_size: int  # Size in bytes
    
    # Lifecycle information
    created_at: datetime
    deployed_at: Optional[datetime] = None
    status: ModelStatus = ModelStatus.TRAINING
    
    # Performance tracking
    prediction_count: int = 0
    average_confidence: float = 0.0
    last_performance_update: Optional[datetime] = None
    
    # Metadata
    training_data_shape: Tuple[int, int] = (0, 0)
    feature_names: List[str] = field(default_factory=list)
    model_architecture: Optional[Dict[str, Any]] = None


@dataclass
class ModelDeploymentResult:
    """Result from model deployment."""
    
    success: bool
    model_id: str
    version: str
    deployment_time: datetime
    deployment_strategy: DeploymentStrategy
    
    # Deployment metrics
    deployment_duration: float
    model_load_time: float
    initial_performance: Dict[str, float]
    
    # Status information
    status: ModelStatus
    message: str
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class ModelManager:
    """
    Model manager for lifecycle management of trained models.
    
    Handles model deployment, versioning, monitoring, and rollback
    for the NAS-TAS system.
    """
    
    def __init__(self, config: ModelManagerConfig):
        """Initialize model manager.
        
        Args:
            config: Model manager configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize storage paths
        self._initialize_storage()
        
        # Model registry
        self.model_registry = {}  # model_id -> ModelMetadata
        self.deployed_models = {}  # model_id -> deployed_model_info
        self.model_versions = {}  # model_id -> [versions]
        
        # Performance tracking
        self.model_performance = {}  # model_id -> performance_history
        self.model_monitoring = {}  # model_id -> monitoring_info
        
        # Audit logging
        if config.enable_audit_logging:
            self._setup_audit_logging()
        
        tprint_success("✅ Model Manager initialized")
        tprint_info(f"📁 Storage path: {config.model_storage_path}")
        tprint_info(f"🚀 Deployment strategy: {config.deployment_strategy.value}")
        tprint_info(f"⚙️ Auto-deployment: {config.enable_auto_deployment}")
        tprint_info(f"📊 Model monitoring: {config.enable_model_monitoring}")
        
        self.logger.info("✅ Model Manager initialized")
        self.logger.info(f"   Storage path: {config.model_storage_path}")
        self.logger.info(f"   Deployment strategy: {config.deployment_strategy.value}")
        self.logger.info(f"   Auto-deployment: {config.enable_auto_deployment}")
        self.logger.info(f"   Model monitoring: {config.enable_model_monitoring}")
    
    def _initialize_storage(self):
        """Initialize storage directories."""
        try:
            tprint_info("📁 Initializing storage directories")
            
            # Create storage directories with enhanced error handling
            directories_to_create = [
                self.config.model_storage_path,
                self.config.model_versions_path,
                self.config.model_metadata_path
            ]
            
            # Add log directories if needed
            if self.config.enable_audit_logging:
                directories_to_create.append(str(Path(self.config.audit_log_path).parent))
            if self.config.enable_performance_logging:
                directories_to_create.append(str(Path(self.config.performance_log_path).parent))
            
            for directory in directories_to_create:
                try:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                    tprint_success(f"✅ Created directory: {directory}")
                except (OSError, PermissionError) as e:
                    tprint_error(f"❌ Failed to create directory {directory}: {e}")
                    self.logger.error(f"❌ Failed to create directory {directory}: {e}")
                    # Try to create with different permissions
                    try:
                        Path(directory).mkdir(parents=True, exist_ok=True, mode=0o755)
                        tprint_warning(f"⚠️ Created directory with fallback permissions: {directory}")
                    except Exception as e2:
                        tprint_error(f"❌ Fallback directory creation failed: {e2}")
                        raise RuntimeError(f"Failed to create storage directory {directory}: {e}") from e
            
            tprint_success("✅ Storage directories initialized successfully")
            self.logger.info("✅ Storage directories initialized")
            
        except Exception as e:
            tprint_error(f"❌ Storage initialization failed: {e}")
            self.logger.error(f"❌ Storage initialization failed: {e}")
            raise
    
    def _setup_audit_logging(self):
        """Setup audit logging."""
        try:
            audit_handler = logging.FileHandler(self.config.audit_log_path)
            audit_handler.setLevel(logging.INFO)
            audit_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            audit_handler.setFormatter(audit_formatter)
            
            # Create audit logger
            self.audit_logger = logging.getLogger('model_audit')
            self.audit_logger.addHandler(audit_handler)
            self.audit_logger.setLevel(logging.INFO)
            
        except Exception as e:
            self.logger.warning(f"Audit logging setup failed: {e}")
            self.audit_logger = None
    
    def register_models(self, regime_models: Dict[int, Dict[str, Any]]) -> Dict[str, str]:
        """
        Register trained models with the manager.
        
        Args:
            regime_models: Dictionary of regime_id -> {model_type: model_info}
            
        Returns:
            Dictionary of model_id -> version mapping
        """
        tprint_info("📝 Registering models with manager")
        self.logger.info("📝 Registering models with manager")
        registration_results = {}
        
        try:
            for regime_id, models in regime_models.items():
                for model_type, model_info in models.items():
                    # Create model ID
                    model_id = f"regime_{regime_id}_{model_type}"
                    
                    # Generate version
                    version = self._generate_version(model_id)
                    
                    # Create model metadata
                    metadata = ModelMetadata(
                        model_id=model_id,
                        model_type=model_type,
                        regime_id=regime_id,
                        version=version,
                        training_performance=model_info.get('train_metrics', {}),
                        validation_performance=model_info.get('val_metrics', {}),
                        test_performance=model_info.get('test_metrics', {}),
                        feature_importance=model_info.get('feature_importance', {}),
                        hyperparameters=model_info.get('hyperparameters', {}),
                        model_size=self._calculate_model_size(model_info['model']),
                        created_at=datetime.now(),
                        training_data_shape=model_info.get('training_data_shape', (0, 0)),
                        feature_names=model_info.get('feature_names', [])
                    )
                    
                    # Register model
                    self.model_registry[model_id] = metadata
                    self.model_versions[model_id] = self.model_versions.get(model_id, []) + [version]
                    
                    # Save model and metadata
                    self._save_model(model_id, version, model_info['model'], metadata)
                    
                    registration_results[model_id] = version
                    
                    tprint_success(f"   ✅ Registered {model_id} v{version}")
                    self.logger.info(f"   ✅ Registered {model_id} v{version}")
            
            tprint_success(f"📊 Total models registered: {len(registration_results)}")
            self.logger.info(f"📊 Total models registered: {len(registration_results)}")
            return registration_results
            
        except Exception as e:
            tprint_error(f"❌ Model registration failed: {e}")
            self.logger.error(f"❌ Model registration failed: {e}")
            raise
    
    def deploy_models(self, 
                     model_ids: Optional[List[str]] = None,
                     deployment_strategy: Optional[DeploymentStrategy] = None) -> Dict[str, ModelDeploymentResult]:
        """
        Deploy models to production.
        
        Args:
            model_ids: List of model IDs to deploy (None for all)
            deployment_strategy: Deployment strategy to use
            
        Returns:
            Dictionary of model_id -> deployment result
        """
        self.logger.info("🚀 Starting model deployment")
        
        try:
            # Determine models to deploy
            if model_ids is None:
                model_ids = list(self.model_registry.keys())
            
            # Use provided strategy or default
            strategy = deployment_strategy or self.config.deployment_strategy
            
            deployment_results = {}
            
            for model_id in model_ids:
                if model_id not in self.model_registry:
                    self.logger.warning(f"⚠️ Model {model_id} not found in registry")
                    continue
                
                # Deploy model
                result = self._deploy_single_model(model_id, strategy)
                deployment_results[model_id] = result
                
                if result.success:
                    self.logger.info(f"   ✅ Deployed {model_id} v{result.version}")
                else:
                    self.logger.error(f"   ❌ Failed to deploy {model_id}: {result.error_message}")
            
            successful_deployments = sum(1 for r in deployment_results.values() if r.success)
            self.logger.info(f"📊 Deployment completed: {successful_deployments}/{len(deployment_results)} successful")
            
            return deployment_results
            
        except Exception as e:
            self.logger.error(f"❌ Model deployment failed: {e}")
            raise
    
    def _deploy_single_model(self, model_id: str, strategy: DeploymentStrategy) -> ModelDeploymentResult:
        """Deploy a single model."""
        start_time = datetime.now()
        
        try:
            # Get model metadata
            metadata = self.model_registry[model_id]
            
            # Check deployment threshold
            if not self._meets_deployment_threshold(metadata):
                return ModelDeploymentResult(
                    success=False,
                    model_id=model_id,
                    version=metadata.version,
                    deployment_time=start_time,
                    deployment_strategy=strategy,
                    deployment_duration=0.0,
                    model_load_time=0.0,
                    initial_performance={},
                    status=ModelStatus.FAILED,
                    message="Model does not meet deployment threshold",
                    error_message="Performance below threshold"
                )
            
            # Load model
            model_load_start = datetime.now()
            model = self._load_model(model_id, metadata.version)
            model_load_time = (datetime.now() - model_load_start).total_seconds()
            
            # Deploy based on strategy
            if strategy == DeploymentStrategy.IMMEDIATE:
                deployment_success = self._immediate_deployment(model_id, model, metadata)
            elif strategy == DeploymentStrategy.GRADUAL:
                deployment_success = self._gradual_deployment(model_id, model, metadata)
            elif strategy == DeploymentStrategy.A_B_TESTING:
                deployment_success = self._ab_testing_deployment(model_id, model, metadata)
            elif strategy == DeploymentStrategy.CANARY:
                deployment_success = self._canary_deployment(model_id, model, metadata)
            else:
                raise ValueError(f"Unknown deployment strategy: {strategy}")
            
            # Update model status
            if deployment_success:
                metadata.status = ModelStatus.DEPLOYED
                metadata.deployed_at = datetime.now()
                self.deployed_models[model_id] = {
                    'model': model,
                    'metadata': metadata,
                    'deployment_time': datetime.now()
                }
                
                # Setup monitoring if enabled
                if self.config.enable_model_monitoring:
                    self._setup_model_monitoring(model_id)
            
            deployment_duration = (datetime.now() - start_time).total_seconds()
            
            return ModelDeploymentResult(
                success=deployment_success,
                model_id=model_id,
                version=metadata.version,
                deployment_time=start_time,
                deployment_strategy=strategy,
                deployment_duration=deployment_duration,
                model_load_time=model_load_time,
                initial_performance=metadata.validation_performance,
                status=metadata.status,
                message="Deployment successful" if deployment_success else "Deployment failed"
            )
            
        except Exception as e:
            deployment_duration = (datetime.now() - start_time).total_seconds()
            return ModelDeploymentResult(
                success=False,
                model_id=model_id,
                version=metadata.version if 'metadata' in locals() else "unknown",
                deployment_time=start_time,
                deployment_strategy=strategy,
                deployment_duration=deployment_duration,
                model_load_time=0.0,
                initial_performance={},
                status=ModelStatus.FAILED,
                message="Deployment failed",
                error_message=str(e)
            )
    
    def _meets_deployment_threshold(self, metadata: ModelMetadata) -> bool:
        """Check if model meets deployment threshold."""
        validation_performance = metadata.validation_performance
        key_metric = self.config.deployment_threshold
        
        # Check F1 score or accuracy
        f1_score = validation_performance.get('f1_score', 0.0)
        accuracy = validation_performance.get('accuracy', 0.0)
        
        return f1_score >= key_metric or accuracy >= key_metric
    
    def _immediate_deployment(self, model_id: str, model: Any, metadata: ModelMetadata) -> bool:
        """Immediate deployment strategy."""
        try:
            # ⚠️ PLACEHOLDER IMPLEMENTATION - This is a stub function
            tprint_warning(f"⚠️ Using placeholder immediate deployment for {model_id}")
            self.logger.warning(f"⚠️ Using placeholder immediate deployment for {model_id}")
            self.logger.info(f"   📦 Immediate deployment of {model_id}")
            # TODO: Implement actual immediate deployment logic
            return True
        except Exception as e:
            tprint_error(f"❌ Immediate deployment failed: {e}")
            self.logger.error(f"   ❌ Immediate deployment failed: {e}")
            return False
    
    def _gradual_deployment(self, model_id: str, model: Any, metadata: ModelMetadata) -> bool:
        """Gradual deployment strategy."""
        try:
            # ⚠️ PLACEHOLDER IMPLEMENTATION - This is a stub function
            tprint_warning(f"⚠️ Using placeholder gradual deployment for {model_id}")
            self.logger.warning(f"⚠️ Using placeholder gradual deployment for {model_id}")
            self.logger.info(f"   📦 Gradual deployment of {model_id}")
            # TODO: Implement actual gradual deployment logic
            # In a real implementation, this would gradually increase traffic
            return True
        except Exception as e:
            tprint_error(f"❌ Gradual deployment failed: {e}")
            self.logger.error(f"   ❌ Gradual deployment failed: {e}")
            return False
    
    def _ab_testing_deployment(self, model_id: str, model: Any, metadata: ModelMetadata) -> bool:
        """A/B testing deployment strategy."""
        try:
            # ⚠️ PLACEHOLDER IMPLEMENTATION - This is a stub function
            tprint_warning(f"⚠️ Using placeholder A/B testing deployment for {model_id}")
            self.logger.warning(f"⚠️ Using placeholder A/B testing deployment for {model_id}")
            self.logger.info(f"   📦 A/B testing deployment of {model_id}")
            # TODO: Implement actual A/B testing deployment logic
            # In a real implementation, this would split traffic between old and new models
            return True
        except Exception as e:
            tprint_error(f"❌ A/B testing deployment failed: {e}")
            self.logger.error(f"   ❌ A/B testing deployment failed: {e}")
            return False
    
    def _canary_deployment(self, model_id: str, model: Any, metadata: ModelMetadata) -> bool:
        """Canary deployment strategy."""
        try:
            # ⚠️ PLACEHOLDER IMPLEMENTATION - This is a stub function
            tprint_warning(f"⚠️ Using placeholder canary deployment for {model_id}")
            self.logger.warning(f"⚠️ Using placeholder canary deployment for {model_id}")
            self.logger.info(f"   📦 Canary deployment of {model_id}")
            # TODO: Implement actual canary deployment logic
            # In a real implementation, this would deploy to a small subset first
            return True
        except Exception as e:
            tprint_error(f"❌ Canary deployment failed: {e}")
            self.logger.error(f"   ❌ Canary deployment failed: {e}")
            return False
    
    def _setup_model_monitoring(self, model_id: str):
        """Setup monitoring for a deployed model."""
        try:
            self.model_monitoring[model_id] = {
                'start_time': datetime.now(),
                'prediction_count': 0,
                'performance_history': [],
                'alerts': [],
                'status': 'monitoring'
            }
            
            self.logger.info(f"   📊 Monitoring setup for {model_id}")
            
        except Exception as e:
            self.logger.error(f"   ❌ Monitoring setup failed for {model_id}: {e}")
    
    def setup_monitoring(self) -> Dict[str, Any]:
        """Setup monitoring for all deployed models."""
        self.logger.info("📊 Setting up model monitoring")
        
        monitoring_results = {}
        
        try:
            for model_id in self.deployed_models.keys():
                self._setup_model_monitoring(model_id)
                monitoring_results[model_id] = {'status': 'monitoring_enabled'}
            
            self.logger.info(f"✅ Monitoring setup completed for {len(monitoring_results)} models")
            return monitoring_results
            
        except Exception as e:
            self.logger.error(f"❌ Monitoring setup failed: {e}")
            return {'error': str(e)}
    
    def update_model_performance(self, 
                               model_id: str,
                               performance_metrics: Dict[str, float],
                               prediction_confidence: Optional[float] = None):
        """Update model performance metrics."""
        try:
            if model_id not in self.model_registry:
                self.logger.warning(f"⚠️ Model {model_id} not found in registry")
                return
            
            # Update metadata
            metadata = self.model_registry[model_id]
            metadata.prediction_count += 1
            metadata.last_performance_update = datetime.now()
            
            if prediction_confidence is not None:
                # Update average confidence
                total_confidence = metadata.average_confidence * (metadata.prediction_count - 1)
                metadata.average_confidence = (total_confidence + prediction_confidence) / metadata.prediction_count
            
            # Update performance history
            if model_id not in self.model_performance:
                self.model_performance[model_id] = []
            
            performance_entry = {
                'timestamp': datetime.now(),
                'metrics': performance_metrics,
                'confidence': prediction_confidence
            }
            self.model_performance[model_id].append(performance_entry)
            
            # Keep only recent performance (sliding window)
            max_history = 1000
            if len(self.model_performance[model_id]) > max_history:
                self.model_performance[model_id] = self.model_performance[model_id][-max_history:]
            
            # Check for performance degradation
            if self.config.enable_auto_rollback:
                self._check_performance_degradation(model_id, performance_metrics)
            
            self.logger.debug(f"📊 Updated performance for {model_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Performance update failed for {model_id}: {e}")
    
    def _check_performance_degradation(self, model_id: str, current_metrics: Dict[str, float]):
        """Check for performance degradation and trigger rollback if needed."""
        try:
            if model_id not in self.model_performance or len(self.model_performance[model_id]) < 10:
                return  # Not enough history
            
            # Get recent performance
            recent_performance = self.model_performance[model_id][-10:]
            recent_f1_scores = [p['metrics'].get('f1_score', 0) for p in recent_performance]
            
            if not recent_f1_scores:
                return
            
            # Calculate performance trend
            current_f1 = current_metrics.get('f1_score', 0)
            avg_recent_f1 = np.mean(recent_f1_scores)
            
            # Check for degradation
            degradation = avg_recent_f1 - current_f1
            
            if degradation > self.config.degradation_threshold:
                self.logger.warning(f"⚠️ Performance degradation detected for {model_id}: {degradation:.3f}")
                
                # Check if rollback is needed
                if current_f1 < self.config.rollback_threshold:
                    self.logger.warning(f"🚨 Triggering rollback for {model_id}")
                    self.rollback_model(model_id)
            
        except Exception as e:
            self.logger.error(f"❌ Performance degradation check failed for {model_id}: {e}")
    
    def rollback_model(self, model_id: str) -> bool:
        """Rollback model to previous version."""
        try:
            if model_id not in self.model_registry:
                self.logger.error(f"❌ Model {model_id} not found for rollback")
                return False
            
            # Get available versions
            versions = self.model_versions.get(model_id, [])
            if len(versions) < 2:
                self.logger.warning(f"⚠️ No previous version available for {model_id}")
                return False
            
            # Get previous version
            current_version = self.model_registry[model_id].version
            previous_version = versions[-2]  # Second to last version
            
            self.logger.info(f"🔄 Rolling back {model_id} from v{current_version} to v{previous_version}")
            
            # Load previous version
            previous_model = self._load_model(model_id, previous_version)
            
            # Update deployed model
            if model_id in self.deployed_models:
                self.deployed_models[model_id]['model'] = previous_model
                self.deployed_models[model_id]['metadata'].version = previous_version
                self.deployed_models[model_id]['metadata'].status = ModelStatus.ROLLED_BACK
            
            # Update registry
            self.model_registry[model_id].version = previous_version
            self.model_registry[model_id].status = ModelStatus.ROLLED_BACK
            
            self.logger.info(f"✅ Rollback completed for {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Rollback failed for {model_id}: {e}")
            return False
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get status of a specific model."""
        if model_id not in self.model_registry:
            return {'error': f'Model {model_id} not found'}
        
        metadata = self.model_registry[model_id]
        
        return {
            'model_id': model_id,
            'version': metadata.version,
            'status': metadata.status.value,
            'created_at': metadata.created_at.isoformat(),
            'deployed_at': metadata.deployed_at.isoformat() if metadata.deployed_at else None,
            'prediction_count': metadata.prediction_count,
            'average_confidence': metadata.average_confidence,
            'performance': {
                'training': metadata.training_performance,
                'validation': metadata.validation_performance,
                'test': metadata.test_performance
            }
        }
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get summary of all deployments."""
        total_models = len(self.model_registry)
        deployed_models = len(self.deployed_models)
        
        status_counts = {}
        for metadata in self.model_registry.values():
            status = metadata.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_models': total_models,
            'deployed_models': deployed_models,
            'deployment_rate': deployed_models / total_models if total_models > 0 else 0,
            'status_distribution': status_counts,
            'deployed_model_ids': list(self.deployed_models.keys())
        }
    
    def _generate_version(self, model_id: str) -> str:
        """Generate version for model."""
        if self.config.version_naming_scheme == "timestamp":
            return datetime.now().strftime("%Y%m%d_%H%M%S")
        elif self.config.version_naming_scheme == "incremental":
            versions = self.model_versions.get(model_id, [])
            return str(len(versions) + 1)
        else:  # semantic
            return "1.0.0"  # Simplified semantic versioning
    
    def _calculate_model_size(self, model: Any) -> int:
        """Calculate model size in bytes."""
        try:
            # Calculate actual model size
            return len(pickle.dumps(model))
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            self.logger.warning(f"Could not calculate model size: {e}")
            return 1024 * 1024  # 1MB default
        except Exception as e:
            self.logger.error(f"Unexpected error calculating model size: {e}")
            raise
    
    def _save_model(self, model_id: str, version: str, model: Any, metadata: ModelMetadata):
        """Save model and metadata to storage."""
        try:
            # Save model
            model_path = Path(self.config.model_storage_path) / model_id / f"{version}.pkl"
            
            # Create directory with proper error handling
            try:
                model_path.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                tprint_error(f"❌ Failed to create model directory: {e}")
                self.logger.error(f"❌ Failed to create model directory: {e}")
                raise RuntimeError(f"Failed to create model directory for {model_id}: {e}") from e
            
            # Save model with proper error handling
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                tprint_success(f"✅ Model {model_id} v{version} saved successfully")
            except (IOError, OSError, pickle.PicklingError) as e:
                tprint_error(f"❌ Failed to save model {model_id}: {e}")
                self.logger.error(f"❌ Failed to save model {model_id}: {e}")
                raise RuntimeError(f"Failed to save model {model_id}: {e}") from e
            
            # Save metadata
            metadata_path = Path(self.config.model_metadata_path) / f"{model_id}_{version}.json"
            metadata_dict = {
                'model_id': metadata.model_id,
                'model_type': metadata.model_type,
                'regime_id': metadata.regime_id,
                'version': metadata.version,
                'training_performance': metadata.training_performance,
                'validation_performance': metadata.validation_performance,
                'test_performance': metadata.test_performance,
                'feature_importance': metadata.feature_importance,
                'hyperparameters': metadata.hyperparameters,
                'model_size': metadata.model_size,
                'created_at': metadata.created_at.isoformat(),
                'deployed_at': metadata.deployed_at.isoformat() if metadata.deployed_at else None,
                'status': metadata.status.value,
                'prediction_count': metadata.prediction_count,
                'average_confidence': metadata.average_confidence,
                'training_data_shape': metadata.training_data_shape,
                'feature_names': metadata.feature_names
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
        except (IOError, OSError, pickle.PicklingError) as e:
            self.logger.error(f"❌ Could not save model {model_id}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error saving model {model_id}: {e}")
            raise
    
    def _load_model(self, model_id: str, version: str) -> Any:
        """Load model from storage."""
        try:
            model_path = Path(self.config.model_storage_path) / model_id / f"{version}.pkl"

            if not model_path.exists():
                tprint_error(f"❌ Model file not found: {model_path}")
                self.logger.error(f"❌ Model file not found: {model_path}")
                raise FileNotFoundError(f"Model {model_id} v{version} not found at {model_path}")

            # Load model with proper error handling
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except (IOError, OSError, pickle.UnpicklingError) as e:
                tprint_error(f"❌ Failed to load model {model_id} v{version}: {e}")
                self.logger.error(f"❌ Failed to load model {model_id} v{version}: {e}")
                raise RuntimeError(f"Failed to load model {model_id} v{version}: {e}") from e

            # Validate model integrity
            if model is None:
                tprint_error(f"❌ Loaded model {model_id} v{version} is None")
                self.logger.error(f"❌ Loaded model {model_id} v{version} is None")
                raise ValueError(f"Loaded model {model_id} v{version} is None")
            
            # Check if model has required methods
            if not hasattr(model, 'predict'):
                tprint_warning(f"⚠️ Model {model_id} v{version} doesn't have predict method")
                self.logger.warning(f"⚠️ Model {model_id} v{version} doesn't have predict method")

            tprint_success(f"✅ Successfully loaded and validated model {model_id} v{version}")
            self.logger.debug(f"✅ Successfully loaded model {model_id} v{version}")
            return model

        except (FileNotFoundError, IOError, OSError, pickle.UnpicklingError) as e:
            self.logger.error(f"❌ Could not load model {model_id} v{version}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error loading model {model_id} v{version}: {e}")
            raise