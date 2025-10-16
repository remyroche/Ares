from ...tprint import tprint

"""
Model Persistence & Versioning Utilities

This module provides comprehensive model persistence, versioning, and registry
capabilities for tracking model lifecycle and performance.

Key Features:
- Model versioning and metadata tracking
- Performance history and lineage tracking
- Automated model deployment
- Model retirement policies
- Experiment reproducibility

Built on existing utilities:
- Uses serialization_utils.py for model persistence
- Leverages file_utils.py for file operations
- Integrates with common_operations.py for robust error handling
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging
import os
import time

from ...common_operations import safe_json_dump, safe_json_load
from ...common_operations import ensure_directory
from ...common_operations import safe_file_exists
from ...common_operations import create_fallback_logger

# Enhanced dependency management with fast fail
try:
    from ...logger import get_logger
    _LOGGER = get_logger("MLCommon.ModelRegistry")
    tprint("✅ Custom logger available for MLCommon.ModelRegistry")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.ModelRegistry")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER


class ModelRegistry:
    """Comprehensive model registry for persistence and versioning."""

    def __init__(self, registry_path: str = "./model_registry", config: Optional[Dict[str, Any]] = None):
        """Initialize model registry with configuration."""
        self.registry_path = Path(registry_path)
        self.config = config or {}
        self.logger = logger.getChild('ModelRegistry')
        
        _LOGGER.info("🚀 Initializing ModelRegistry...")
        _LOGGER.info(f"📁 Registry path: {self.registry_path}")

        # Configuration defaults
        self.enable_compression = self.config.get('enable_compression', True)
        self.max_versions_per_model = self.config.get('max_versions_per_model', 10)
        self.auto_cleanup = self.config.get('auto_cleanup', True)

        _LOGGER.info(f"⚙️ Configuration - Compression: {self.enable_compression}")
        _LOGGER.info(f"⚙️ Configuration - Max versions per model: {self.max_versions_per_model}")
        _LOGGER.info(f"⚙️ Configuration - Auto cleanup: {self.auto_cleanup}")

        # Ensure registry directory exists
        _LOGGER.debug("🔧 Ensuring registry directory exists...")
        ensure_directory(self.registry_path)

        # Initialize registry metadata
        self.metadata_file = self.registry_path / "registry_metadata.json"
        _LOGGER.debug("🔧 Loading registry metadata...")
        self._load_registry_metadata()
        
        _LOGGER.info("✅ ModelRegistry initialized successfully")

    def save_model_with_metadata(self, model: Any, metadata: Dict[str, Any],
                               version_strategy: str = 'auto',
                               model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Save model with comprehensive metadata.

        Args:
            model: Trained model object
            metadata: Model metadata dictionary
            version_strategy: Versioning strategy ('auto', 'manual', 'timestamp')
            model_name: Model name (auto-generated if None)

        Returns:
            Save result information
        """
        start_time = time.time()
        _LOGGER.info(f"💾 Starting model save with metadata...")
        _LOGGER.info(f"📊 Parameters - Version strategy: {version_strategy}, Model name: {model_name or 'auto-generated'}")
        _LOGGER.debug(f"📊 Metadata keys: {list(metadata.keys()) if metadata else 'None'}")
        
        try:
            # Generate model name if not provided
            if model_name is None:
                _LOGGER.debug("🔧 Generating model name...")
                model_name = self._generate_model_name(model, metadata)
                _LOGGER.info(f"📊 Generated model name: {model_name}")

            # Generate version
            _LOGGER.debug("🔧 Generating version...")
            version = self._generate_version(model_name, version_strategy)
            _LOGGER.info(f"📊 Generated version: {version}")

            # Create model directory
            model_dir = self.registry_path / model_name / version
            _LOGGER.debug(f"🔧 Creating model directory: {model_dir}")
            ensure_directory(model_dir)

            # Save model
            model_path = model_dir / "model.pkl"
            _LOGGER.debug("💾 Saving model pickle...")
            self._save_model_pickle(model, model_path)

            # Save metadata
            metadata_path = model_dir / "metadata.json"
            _LOGGER.debug("💾 Enhancing and saving metadata...")
            enhanced_metadata = self._enhance_metadata(metadata, model_name, version, model)
            self._save_metadata(enhanced_metadata, metadata_path)

            # Update registry
            _LOGGER.debug("🔧 Updating registry entry...")
            self._update_registry_entry(model_name, version, enhanced_metadata)

            result = {
                'model_name': model_name,
                'version': version,
                'model_path': str(model_path),
                'metadata_path': str(metadata_path),
                'success': True
            }

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Model saved successfully in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Results - Model: {model_name} v{version}, Path: {model_path}")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Model save failed after {execution_time:.3f}s: {e}")
            return {'error': str(e), 'success': False}

    def load_model_with_validation(self, model_id: str, version: str = 'latest') -> Dict[str, Any]:
        """
        Load model with validation and metadata.

        Args:
            model_id: Model identifier
            version: Model version ('latest', 'best', or specific version)

        Returns:
            Loaded model and metadata
        """
        start_time = time.time()
        _LOGGER.info(f"📂 Starting model load with validation...")
        _LOGGER.info(f"📊 Parameters - Model ID: {model_id}, Version: {version}")
        
        try:
            # Resolve version
            _LOGGER.debug("🔧 Resolving version...")
            actual_version = self._resolve_version(model_id, version)

            if not actual_version:
                _LOGGER.error(f"❌ Version '{version}' not found for model '{model_id}'")
                raise ValueError(f"Version '{version}' not found for model '{model_id}'")

            _LOGGER.info(f"📊 Resolved version: {actual_version}")

            # Load model
            model_path = self.registry_path / model_id / actual_version / "model.pkl"
            _LOGGER.debug(f"📂 Loading model from: {model_path}")
            model = self._load_model_pickle(model_path)

            # Load metadata
            _LOGGER.debug("📂 Loading metadata...")
            metadata_path = self.registry_path / model_id / actual_version / "metadata.json"
            metadata = self._load_metadata(metadata_path)

            # Validate model
            validation_result = self._validate_loaded_model(model, metadata)

            result = {
                'model': model,
                'metadata': metadata,
                'model_id': model_id,
                'version': actual_version,
                'validation': validation_result,
                'success': True
            }

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Model loaded successfully in {execution_time:.3f}s")
            # Try to load explanation if available
            explanation = self._load_model_explanation(model_id, actual_version)
            if explanation:
                result['explanation'] = explanation
                _LOGGER.info(f"📊 Loaded explanation for model {model_id}")

            _LOGGER.info(f"📊 Results - Model: {model_id} v{actual_version}, Validation: {validation_result.get('status', 'unknown')}")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Model load failed after {execution_time:.3f}s: {e}")
            return {'error': str(e), 'success': False}

    def model_performance_tracking(self, model_id: str, metrics: Dict[str, Any],
                                 dataset_info: Dict[str, Any],
                                 version: Optional[str] = None) -> bool:
        """
        Track model performance over time.

        Args:
            model_id: Model identifier
            metrics: Performance metrics
            dataset_info: Dataset information
            version: Specific version (uses latest if None)

        Returns:
            Success status
        """
        try:
            if version is None:
                version = self._get_latest_version(model_id)

            if not version:
                return False

            # Load existing metadata
            metadata_path = self.registry_path / model_id / version / "metadata.json"
            metadata = self._load_metadata(metadata_path)

            # Add performance tracking
            if 'performance_history' not in metadata:
                metadata['performance_history'] = []

            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'dataset_info': dataset_info
            }

            metadata['performance_history'].append(performance_entry)

            # Update metadata
            self._save_metadata(metadata, metadata_path)

            # Update registry
            self._update_registry_entry(model_id, version, metadata)

            self.logger.info(f"✅ Performance tracked for {model_id} v{version}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Performance tracking failed: {e}")
            return False

    def model_lineage_tracking(self, model_id: str, parent_models: List[str],
                             feature_engineering_steps: List[str],
                             version: Optional[str] = None) -> bool:
        """
        Track model lineage and dependencies.

        Args:
            model_id: Model identifier
            parent_models: List of parent model IDs
            feature_engineering_steps: Feature engineering pipeline steps
            version: Specific version (uses latest if None)

        Returns:
            Success status
        """
        try:
            if version is None:
                version = self._get_latest_version(model_id)

            if not version:
                return False

            # Load existing metadata
            metadata_path = self.registry_path / model_id / version / "metadata.json"
            metadata = self._load_metadata(metadata_path)

            # Add lineage information
            metadata['lineage'] = {
                'parent_models': parent_models,
                'feature_engineering_steps': feature_engineering_steps,
                'created_at': datetime.now().isoformat()
            }

            # Update metadata
            self._save_metadata(metadata, metadata_path)

            # Update registry
            self._update_registry_entry(model_id, version, metadata)

            self.logger.info(f"✅ Lineage tracked for {model_id} v{version}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Lineage tracking failed: {e}")
            return False

    def automated_model_deployment(self, model_id: str, target_environment: str,
                                 version: str = 'latest',
                                 deployment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Automate model deployment process.

        Args:
            model_id: Model identifier
            target_environment: Target deployment environment
            version: Model version
            deployment_config: Deployment configuration

        Returns:
            Deployment result
        """
        try:
            # Load model
            load_result = self.load_model_with_validation(model_id, version)
            if not load_result['success']:
                raise ValueError(f"Failed to load model: {load_result.get('error')}")

            model = load_result['model']
            metadata = load_result['metadata']

            # Validate deployment readiness
            deployment_validation = self._validate_deployment_readiness(metadata, target_environment)

            if not deployment_validation['ready']:
                raise ValueError(f"Model not ready for deployment: {deployment_validation['issues']}")

            # Prepare deployment package
            deployment_package = self._prepare_deployment_package(
                model, metadata, target_environment, deployment_config
            )

            # Log deployment
            deployment_record = {
                'model_id': model_id,
                'version': load_result['version'],
                'target_environment': target_environment,
                'deployment_time': datetime.now().isoformat(),
                'deployment_config': deployment_config,
                'validation_results': deployment_validation
            }

            self._log_deployment(deployment_record)

            result = {
                'deployment_id': self._generate_deployment_id(),
                'model_id': model_id,
                'version': load_result['version'],
                'target_environment': target_environment,
                'deployment_package': deployment_package,
                'success': True
            }

            self.logger.info(f"✅ Model deployment prepared: {model_id} -> {target_environment}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Model deployment failed: {e}")
            return {'error': str(e), 'success': False}

    def model_retirement_policy(self, performance_threshold: float,
                              age_threshold_days: int,
                              auto_retire: bool = False) -> Dict[str, Any]:
        """
        Apply model retirement policy.

        Args:
            performance_threshold: Minimum performance threshold
            age_threshold_days: Maximum age in days
            auto_retire: Whether to automatically retire models

        Returns:
            Retirement analysis and actions taken
        """
        try:
            retirement_analysis = {
                'models_analyzed': 0,
                'models_to_retire': [],
                'retirement_actions': [],
                'policy_applied': {
                    'performance_threshold': performance_threshold,
                    'age_threshold_days': age_threshold_days,
                    'auto_retire': auto_retire
                }
            }

            # Analyze all models
            for model_name in self.registry_metadata.get('models', {}):
                model_info = self.registry_metadata['models'][model_name]
                latest_version = model_info.get('latest_version')

                if latest_version:
                    # Check retirement criteria
                    should_retire, reasons = self._check_retirement_criteria(
                        model_name, latest_version, performance_threshold, age_threshold_days
                    )

                    if should_retire:
                        retirement_analysis['models_to_retire'].append({
                            'model_name': model_name,
                            'version': latest_version,
                            'reasons': reasons
                        })

                        if auto_retire:
                            retirement_result = self._retire_model(model_name, latest_version, reasons)
                            retirement_analysis['retirement_actions'].append(retirement_result)

                retirement_analysis['models_analyzed'] += 1

            self.logger.info(f"✅ Retirement policy applied: "
                           f"{len(retirement_analysis['models_to_retire'])} models flagged for retirement")
            return retirement_analysis

        except Exception as e:
            self.logger.error(f"❌ Retirement policy application failed: {e}")
            return {'error': str(e)}

    def experiment_reproducibility(self, experiment_config: Dict[str, Any],
                                 random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Ensure experiment reproducibility with configuration tracking.

        Args:
            experiment_config: Experiment configuration
            random_seed: Random seed for reproducibility

        Returns:
            Reproducibility setup result
        """
        try:
            # Generate experiment ID
            experiment_id = self._generate_experiment_id(experiment_config)

            # Set up reproducibility
            reproducibility_setup = {
                'experiment_id': experiment_id,
                'random_seed': random_seed or 42,
                'timestamp': datetime.now().isoformat(),
                'config_hash': self._hash_config(experiment_config),
                'environment_info': self._capture_environment_info()
            }

            # Save reproducibility information
            repro_path = self.registry_path / "experiments" / experiment_id
            ensure_directory(repro_path)

            repro_file = repro_path / "reproducibility.json"
            safe_json_dump(reproducibility_setup, repro_file)

            result = {
                'experiment_id': experiment_id,
                'reproducibility_setup': reproducibility_setup,
                'repro_path': str(repro_path),
                'success': True
            }

            self.logger.info(f"✅ Experiment reproducibility setup: {experiment_id}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Experiment reproducibility setup failed: {e}")
            return {'error': str(e), 'success': False}

    def _generate_model_name(self, model: Any, metadata: Dict[str, Any]) -> str:
        """Generate unique model name."""
        try:
            # Use algorithm type from metadata or model
            algorithm = metadata.get('algorithm', type(model).__name__)

            # Add timestamp for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            return f"{algorithm}_{timestamp}"

        except Exception:
            return f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _generate_version(self, model_name: str, strategy: str) -> str:
        """Generate version string."""
        if strategy == 'auto':
            existing_versions = self._get_existing_versions(model_name)
            if existing_versions:
                # Increment version number
                latest_version = max(existing_versions)
                try:
                    version_num = int(latest_version) + 1
                    return str(version_num)
                except ValueError as e:
                    self.logger.warning(f"Could not parse version number '{latest_version}': {e}. Using version 1.")
                    return "1"
            return "1"
        elif strategy == 'timestamp':
            return datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            return "1"

    def _save_model_pickle(self, model: Any, path: Path) -> None:
        """Save model using pickle."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            self.logger.warning(f"Pickle save failed, trying alternative: {e}")
            # Fallback: try joblib if available
            try:
                from joblib import dump
                dump(model, path)
            except ImportError:
                raise e

    def _load_model_pickle(self, path: Path) -> Any:
        """Load model from pickle."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Pickle load failed, trying alternative: {e}")
            # Fallback: try joblib if available
            try:
                from joblib import load
                return load(path)
            except ImportError:
                raise e

    def _enhance_metadata(self, metadata: Dict[str, Any], model_name: str,
                         version: str, model: Any) -> Dict[str, Any]:
        """Enhance metadata with additional information."""
        enhanced = metadata.copy()

        # Add registry information
        enhanced.update({
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_type': getattr(model, '__class__', {}).get('__name__', 'unknown'),
            'registry_info': {
                'registry_path': str(self.registry_path),
                'registry_version': '1.0'
            }
        })

        return enhanced

    def _save_metadata(self, metadata: Dict[str, Any], path: Path) -> None:
        """Save metadata to JSON file."""
        safe_json_dump(metadata, path)

    def _load_metadata(self, path: Path) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        return safe_json_load(path)

    def _load_registry_metadata(self) -> None:
        """Load registry metadata."""
        if self.metadata_file.exists():
            self.registry_metadata = safe_json_load(self.metadata_file)
        else:
            self.registry_metadata = {'models': {}, 'version': '1.0'}

    def _update_registry_entry(self, model_name: str, version: str, metadata: Dict[str, Any]) -> None:
        """Update registry entry for model."""
        if 'models' not in self.registry_metadata:
            self.registry_metadata['models'] = {}

        if model_name not in self.registry_metadata['models']:
            self.registry_metadata['models'][model_name] = {}

        self.registry_metadata['models'][model_name][version] = {
            'created_at': metadata.get('created_at'),
            'model_type': metadata.get('model_type'),
            'latest_performance': metadata.get('performance_history', [])[-1] if metadata.get('performance_history') else None
        }

        # Update latest version
        versions = [v for v in self.registry_metadata['models'][model_name].keys() if v != 'latest_version']
        if versions:
            # Prefer numeric comparison when possible
            def _vkey(v: str):
                try:
                    return (0, int(v))
                except Exception:
                    return (1, v)
            latest = sorted(versions, key=_vkey)[-1]
            self.registry_metadata['models'][model_name]['latest_version'] = latest

        # Save registry metadata
        safe_json_dump(self.registry_metadata, self.metadata_file)

    def _resolve_version(self, model_id: str, version: str) -> Optional[str]:
        """Resolve version string to actual version."""
        if model_id not in self.registry_metadata.get('models', {}):
            return None

        model_versions = self.registry_metadata['models'][model_id]

        if version == 'latest':
            return model_versions.get('latest_version')
        elif version == 'best':
            # Find version with best performance
            best_version = None
            best_score = -float('inf')

            for v, info in model_versions.items():
                if v != 'latest_version':
                    perf = info.get('latest_performance', {})
                    score = perf.get('metrics', {}).get('accuracy', 0)
                    if score > best_score:
                        best_score = score
                        best_version = v

            return best_version
        else:
            return version if version in model_versions else None

    def _validate_loaded_model(self, model: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate loaded model."""
        try:
            validation = {'valid': True, 'issues': []}

            # Check if model has required methods
            required_methods = ['predict']
            for method in required_methods:
                if not hasattr(model, method):
                    validation['issues'].append(f"Missing required method: {method}")
                    validation['valid'] = False

            # Check metadata consistency
            if 'model_type' in metadata:
                actual_type = getattr(model, '__class__', {}).get('__name__', 'unknown')
                if actual_type != metadata['model_type']:
                    validation['issues'].append(f"Model type mismatch: expected {metadata['model_type']}, got {actual_type}")

            return validation

        except Exception as e:
            return {'valid': False, 'issues': [str(e)]}

    def _get_existing_versions(self, model_name: str) -> List[str]:
        """Get existing versions for a model."""
        if model_name in self.registry_metadata.get('models', {}):
            return [v for v in self.registry_metadata['models'][model_name].keys() if v != 'latest_version']
        return []

    def _get_latest_version(self, model_name: str) -> Optional[str]:
        """Get latest version for a model."""
        if model_name in self.registry_metadata.get('models', {}):
            return self.registry_metadata['models'][model_name].get('latest_version')
        return None

    def _validate_deployment_readiness(self, metadata: Dict[str, Any],
                                     target_environment: str) -> Dict[str, Any]:
        """Validate if model is ready for deployment."""
        validation = {'ready': True, 'issues': []}

        # Check required metadata
        required_fields = ['model_type', 'created_at', 'performance_history']
        for field in required_fields:
            if field not in metadata:
                validation['issues'].append(f"Missing required metadata: {field}")
                validation['ready'] = False

        # Check performance requirements
        if 'performance_history' in metadata and metadata['performance_history']:
            latest_perf = metadata['performance_history'][-1]
            min_accuracy = 0.5  # Example threshold
            accuracy = latest_perf.get('metrics', {}).get('accuracy', 0)
            if accuracy < min_accuracy:
                validation['issues'].append(f"Performance below threshold: {accuracy} < {min_accuracy}")
                validation['ready'] = False

        return validation

    def _prepare_deployment_package(self, model: Any, metadata: Dict[str, Any],
                                  target_environment: str,
                                  deployment_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare deployment package."""
        return {
            'model': model,
            'metadata': metadata,
            'target_environment': target_environment,
            'deployment_config': deployment_config,
            'prepared_at': datetime.now().isoformat()
        }

    def _log_deployment(self, deployment_record: Dict[str, Any]) -> None:
        """Log deployment event."""
        # This would typically integrate with a logging system
        self.logger.info(f"Model deployment logged: {deployment_record['model_id']} v{deployment_record['version']}")

    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID."""
        return f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(datetime.now()) % 1000}"

    def _check_retirement_criteria(self, model_name: str, version: str,
                                 performance_threshold: float,
                                 age_threshold_days: int) -> Tuple[bool, List[str]]:
        """Check if model should be retired."""
        try:
            # Load metadata
            metadata_path = self.registry_path / model_name / version / "metadata.json"
            metadata = self._load_metadata(metadata_path)

            reasons = []
            should_retire = False

            # Check performance
            if 'performance_history' in metadata and metadata['performance_history']:
                latest_perf = metadata['performance_history'][-1]
                accuracy = latest_perf.get('metrics', {}).get('accuracy', 1.0)
                if accuracy < performance_threshold:
                    reasons.append(f"Performance below threshold: {accuracy} < {performance_threshold}")
                    should_retire = True

            # Check age
            if 'created_at' in metadata:
                created_at = datetime.fromisoformat(metadata['created_at'])
                age_days = (datetime.now() - created_at).days
                if age_days > age_threshold_days:
                    reasons.append(f"Model too old: {age_days} > {age_threshold_days} days")
                    should_retire = True

            return should_retire, reasons

        except Exception as e:
            return False, [f"Retirement check failed: {e}"]

    def _retire_model(self, model_name: str, version: str, reasons: List[str]) -> Dict[str, Any]:
        """Retire a model."""
        try:
            # Mark as retired in metadata
            metadata_path = self.registry_path / model_name / version / "metadata.json"
            metadata = self._load_metadata(metadata_path)

            metadata['retired'] = {
                'retired_at': datetime.now().isoformat(),
                'reasons': reasons
            }

            self._save_metadata(metadata, metadata_path)

            return {
                'model_name': model_name,
                'version': version,
                'retired': True,
                'reasons': reasons
            }

        except Exception as e:
            return {
                'model_name': model_name,
                'version': version,
                'retired': False,
                'error': str(e)
            }

    def _generate_experiment_id(self, config: Dict[str, Any]) -> str:
        """Generate unique experiment ID."""
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config_hash}"

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash of configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _capture_environment_info(self) -> Dict[str, Any]:
        """Capture environment information for reproducibility."""
        try:
            import platform
            import sys

            return {
                'python_version': sys.version,
                'platform': platform.platform(),
                'architecture': platform.architecture(),
                'processor': platform.processor(),
                'timestamp': datetime.now().isoformat()
            }

        except Exception:
            return {'error': 'Environment capture failed'}

    def _load_model_explanation(self, model_id: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Load model explanation from registry.
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            Explanation data if found, None otherwise
        """
        try:
            explanation_path = self.registry_path / model_id / version / "explanation.json"
            
            if explanation_path.exists():
                _LOGGER.debug(f"📂 Loading explanation from: {explanation_path}")
                explanation_data = safe_json_load(explanation_path)
                return explanation_data
            else:
                _LOGGER.debug(f"📂 No explanation found for model {model_id} version {version}")
                return None
                
        except Exception as e:
            _LOGGER.warning(f"⚠️ Could not load explanation for {model_id}: {e}")
            return None
