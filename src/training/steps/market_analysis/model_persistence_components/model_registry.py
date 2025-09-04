"""Model registry component for model persistence."""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger
from src.core.decorators.errors import handles_errors

class ModelRegistry:
    """Handles model registration and cataloging."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model registry.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('registry', {})
        self.logger = system_logger.getChild('model_registry')
        self.backend = self.config.get('backend', 'local')
        self.base_dir = Path(self.config.get('base_dir', 'models'))
        self.registry_file = self.base_dir / 'model_registry.json'
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from storage."""
        if self.backend == 'local':
            if self.registry_file.exists():
                try:
                    with open(self.registry_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    self.logger.warning(f'Failed to load registry: {str(e)}')
        return {'models': {}, 'tags': {}, 'deployments': {}, 'created_at': datetime.now().isoformat(), 'registry_version': '1.0'}

    @handles_errors(exceptions=(Exception,), default_return=[], context='model registration')
    async def register_models(self, artifacts: Dict[str, Any], metadata: Dict[str, Any], version_info: Dict[str, Any]) -> List[str]:
        """Register models in the registry.
        
        Args:
            artifacts: Saved artifact paths
            metadata: Model metadata
            version_info: Version information
            
        Returns:
            List of registered model IDs
        """
        registered_models = []
        for category, paths in artifacts.items():
            if category in ['metadata', 'training_report']:
                continue
            if isinstance(paths, dict):
                for model_key, model_path in paths.items():
                    if model_path and (not model_key.endswith('_importance')):
                        model_id = await self._register_single_model(model_key, model_path, category, metadata, version_info)
                        if model_id:
                            registered_models.append(model_id)
        self._save_registry()
        return registered_models

    async def _register_single_model(self, model_name: str, model_path: str, category: str, metadata: Dict[str, Any], version_info: Dict[str, Any]) -> Optional[str]:
        """Register a single model.
        
        Args:
            model_name: Model name
            model_path: Path to model file
            category: Model category
            metadata: Model metadata
            version_info: Version information
            
        Returns:
            Model ID or None
        """
        try:
            model_id = f"{version_info['version']}_{model_name}"
            registry_entry = {'model_id': model_id, 'model_name': model_name, 'category': category, 'version': version_info['version'], 'path': model_path, 'registered_at': datetime.now().isoformat(), 'status': 'registered', 'metadata': {'symbol': version_info.get('symbol'), 'exchange': version_info.get('exchange'), 'training_date': metadata.get('created_at')}}
            if 'performance_metrics' in metadata:
                perf = metadata['performance_metrics']
                if 'training_metrics' in perf:
                    registry_entry['performance'] = perf['training_metrics']
            self.registry['models'][model_id] = registry_entry
            self.logger.info(f'Registered model: {model_id}')
            return model_id
        except Exception as e:
            self.logger.error(f'Failed to register model {model_name}: {str(e)}')
            return None

    def _save_registry(self) -> None:
        """Save registry to storage."""
        if self.backend == 'local':
            try:
                self.base_dir.mkdir(parents=True, exist_ok=True)
                with open(self.registry_file, 'w') as f:
                    json.dump(self.registry, f, indent=2)
            except Exception as e:
                self.logger.error(f'Failed to save registry: {str(e)}')

    @handles_errors(exceptions=(Exception,), default_return=[], context='model search')
    async def search_models(self, category: Optional[str]=None, version: Optional[str]=None, symbol: Optional[str]=None, tags: Optional[List[str]]=None) -> List[Dict[str, Any]]:
        """Search for models in the registry.
        
        Args:
            category: Filter by category
            version: Filter by version
            symbol: Filter by symbol
            tags: Filter by tags
            
        Returns:
            List of matching models
        """
        matching_models = []
        for model_id, model_info in self.registry['models'].items():
            if category and model_info.get('category') != category:
                continue
            if version and model_info.get('version') != version:
                continue
            if symbol and model_info.get('metadata', {}).get('symbol') != symbol:
                continue
            if tags:
                model_tags = self.registry['tags'].get(model_id, [])
                if not any((tag in model_tags for tag in tags)):
                    continue
            matching_models.append(model_info)
        matching_models.sort(key=lambda x: x.get('registered_at', ''), reverse=True)
        return matching_models

    @handles_errors(exceptions=(Exception,), default_return=None, context='model retrieval')
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific model from the registry.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information or None
        """
        return self.registry['models'].get(model_id)

    @handles_errors(exceptions=(Exception,), default_return=False, context='model tagging')
    async def tag_model(self, model_id: str, tags: List[str], replace: bool=False) -> bool:
        """Tag a model in the registry.
        
        Args:
            model_id: Model ID
            tags: Tags to add
            replace: Replace existing tags if True
            
        Returns:
            Success status
        """
        if model_id not in self.registry['models']:
            self.logger.error(f'Model not found: {model_id}')
            return False
        if model_id not in self.registry['tags']:
            self.registry['tags'][model_id] = []
        if replace:
            self.registry['tags'][model_id] = tags
        else:
            existing_tags = set(self.registry['tags'][model_id])
            existing_tags.update(tags)
            self.registry['tags'][model_id] = list(existing_tags)
        self.registry['models'][model_id]['tags'] = self.registry['tags'][model_id]
        self.registry['models'][model_id]['updated_at'] = datetime.now().isoformat()
        self._save_registry()
        self.logger.info(f'Tagged model {model_id} with: {tags}')
        return True

    @handles_errors(exceptions=(Exception,), default_return=False, context='model deployment')
    async def mark_for_deployment(self, model_id: str, environment: str, notes: Optional[str]=None) -> bool:
        """Mark a model for deployment.
        
        Args:
            model_id: Model ID
            environment: Deployment environment (e.g., "staging", "production")
            notes: Optional deployment notes
            
        Returns:
            Success status
        """
        if model_id not in self.registry['models']:
            self.logger.error(f'Model not found: {model_id}')
            return False
        deployment_id = f"{model_id}_{environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        deployment_entry = {'deployment_id': deployment_id, 'model_id': model_id, 'environment': environment, 'status': 'pending', 'marked_at': datetime.now().isoformat(), 'notes': notes}
        if deployment_id not in self.registry['deployments']:
            self.registry['deployments'][deployment_id] = deployment_entry
        self.registry['models'][model_id]['deployment_status'] = {environment: 'pending'}
        await self.tag_model(model_id, [f'deploy_{environment}'])
        self._save_registry()
        self.logger.info(f'Marked model {model_id} for deployment to {environment}')
        return True

    @handles_errors(exceptions=(Exception,), default_return={}, context='registry statistics')
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Registry statistics
        """
        stats = {'total_models': len(self.registry['models']), 'models_by_category': {}, 'models_by_version': {}, 'tagged_models': 0, 'deployment_queue': 0, 'recent_registrations': []}
        for model_info in self.registry['models'].values():
            category = model_info.get('category', 'unknown')
            stats['models_by_category'][category] = stats['models_by_category'].get(category, 0) + 1
            version = model_info.get('version', 'unknown')
            stats['models_by_version'][version] = stats['models_by_version'].get(version, 0) + 1
        stats['tagged_models'] = len(self.registry['tags'])
        for deployment in self.registry['deployments'].values():
            if deployment.get('status') == 'pending':
                stats['deployment_queue'] += 1
        recent_models = sorted(self.registry['models'].values(), key=lambda x: x.get('registered_at', ''), reverse=True)[:5]
        stats['recent_registrations'] = [{'model_id': m['model_id'], 'category': m.get('category'), 'registered_at': m.get('registered_at')} for m in recent_models]
        return stats