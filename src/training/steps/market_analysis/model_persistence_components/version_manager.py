"""Version manager component for model persistence."""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger
from src.core.decorators.errors import handles_errors

class VersionManager:
    """Handles model versioning and version tracking."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the version manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('versioning', {})
        self.logger = system_logger.getChild('version_manager')
        self.versioning_scheme = self.config.get('scheme', 'timestamp')
        self.version_format = self.config.get('format', '%Y%m%d_%H%M%S')
        self.max_versions = self.config.get('max_versions', 10)
        self.base_dir = Path(self.config.get('base_dir', 'models'))
        self.version_registry_file = self.base_dir / 'version_registry.json'
        self.version_registry = self._load_version_registry()

    def _load_version_registry(self) -> Dict[str, Any]:
        """Load version registry from file."""
        if self.version_registry_file.exists():
            try:
                with open(self.version_registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f'Failed to load version registry: {str(e)}')
        return {'versions': [], 'latest': None, 'created_at': datetime.now().isoformat()}

    @handles_errors(exceptions=(Exception,), default_return={}, context='version creation')
    async def create_version(self, symbol: str, exchange: str, timestamp: Optional[datetime]=None) -> Dict[str, Any]:
        """Create a new version for model storage.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            Version information dictionary
        """
        if timestamp is None:
            timestamp = datetime.now()
        if self.versioning_scheme == 'timestamp':
            version = timestamp.strftime(self.version_format)
        elif self.versioning_scheme == 'semantic':
            version = await self._get_next_semantic_version()
        elif self.versioning_scheme == 'incremental':
            version = await self._get_next_incremental_version()
        else:
            version = timestamp.strftime(self.version_format)
        version_info = {'version': version, 'symbol': symbol, 'exchange': exchange, 'timestamp': timestamp.isoformat(), 'scheme': self.versioning_scheme, 'status': 'active', 'metadata': {'created_at': datetime.now().isoformat(), 'created_by': 'training_pipeline'}}
        await self._add_to_registry(version_info)
        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        version_info_file = version_dir / 'version_info.json'
        with open(version_info_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        self.logger.info(f'Created version: {version}')
        return version_info

    async def _get_next_semantic_version(self) -> str:
        """Get next semantic version (major.minor.patch)."""
        if not self.version_registry['versions']:
            return '1.0.0'
        latest = self.version_registry['latest']
        if latest and '.' in latest:
            parts = latest.split('.')
            if len(parts) == 3:
                try:
                    major, minor, patch = map(int, parts)
                    return f'{major}.{minor}.{patch + 1}'
                except ValueError:
                    pass
        return '1.0.0'

    async def _get_next_incremental_version(self) -> str:
        """Get next incremental version number."""
        if not self.version_registry['versions']:
            return 'v001'
        max_num = 0
        for version_info in self.version_registry['versions']:
            version = version_info.get('version', '')
            if version.startswith('v') and version[1:].isdigit():
                num = int(version[1:])
                max_num = max(max_num, num)
        return f'v{max_num + 1:03d}'

    async def _add_to_registry(self, version_info: Dict[str, Any]) -> None:
        """Add version to registry and save."""
        self.version_registry['versions'].append(version_info)
        self.version_registry['latest'] = version_info['version']
        self.version_registry['updated_at'] = datetime.now().isoformat()
        if len(self.version_registry['versions']) > self.max_versions:
            old_versions = self.version_registry['versions'][:-self.max_versions]
            self.version_registry['versions'] = self.version_registry['versions'][-self.max_versions:]
            for old_version in old_versions:
                await self._archive_version(old_version)
        self._save_version_registry()

    def _save_version_registry(self) -> None:
        """Save version registry to file."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(self.version_registry_file, 'w') as f:
                json.dump(self.version_registry, f, indent=2)
        except Exception as e:
            self.logger.error(f'Failed to save version registry: {str(e)}')

    async def _archive_version(self, version_info: Dict[str, Any]) -> None:
        """Archive an old version."""
        version = version_info['version']
        version_dir = self.base_dir / version
        if version_dir.exists():
            archive_dir = self.base_dir / 'archive'
            archive_dir.mkdir(exist_ok=True)
            archive_path = archive_dir / version
            if not archive_path.exists():
                version_dir.rename(archive_path)
                self.logger.info(f'Archived version: {version}')

    @handles_errors(exceptions=(Exception,), default_return=[], context='version listing')
    async def list_versions(self, symbol: Optional[str]=None, exchange: Optional[str]=None, status: Optional[str]=None) -> List[Dict[str, Any]]:
        """List available versions with optional filtering.
        
        Args:
            symbol: Filter by symbol
            exchange: Filter by exchange
            status: Filter by status
            
        Returns:
            List of version information
        """
        versions = self.version_registry.get('versions', [])
        filtered_versions = []
        for version_info in versions:
            if symbol and version_info.get('symbol') != symbol:
                continue
            if exchange and version_info.get('exchange') != exchange:
                continue
            if status and version_info.get('status') != status:
                continue
            filtered_versions.append(version_info)
        filtered_versions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return filtered_versions

    @handles_errors(exceptions=(Exception,), default_return=None, context='version retrieval')
    async def get_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Get information for a specific version.
        
        Args:
            version: Version string
            
        Returns:
            Version information or None
        """
        for version_info in self.version_registry.get('versions', []):
            if version_info.get('version') == version:
                return version_info
        return None

    @handles_errors(exceptions=(Exception,), default_return=None, context='latest version retrieval')
    async def get_latest_version(self, symbol: Optional[str]=None, exchange: Optional[str]=None) -> Optional[Dict[str, Any]]:
        """Get the latest version.
        
        Args:
            symbol: Filter by symbol
            exchange: Filter by exchange
            
        Returns:
            Latest version information or None
        """
        versions = await self.list_versions(symbol, exchange, status='active')
        if versions:
            return versions[0]
        return None

    @handles_errors(exceptions=(Exception,), default_return=False, context='version comparison')
    async def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions.
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        v1_info = await self.get_version(version1)
        v2_info = await self.get_version(version2)
        if not v1_info or not v2_info:
            return {'error': 'One or both versions not found'}
        comparison = {'version1': version1, 'version2': version2, 'timestamp_diff': None, 'is_newer': None, 'differences': {}}
        try:
            t1 = datetime.fromisoformat(v1_info['timestamp'])
            t2 = datetime.fromisoformat(v2_info['timestamp'])
            comparison['timestamp_diff'] = str(t2 - t1)
            comparison['is_newer'] = version2 if t2 > t1 else version1
        except:
            pass
        for key in ['symbol', 'exchange', 'scheme']:
            if v1_info.get(key) != v2_info.get(key):
                comparison['differences'][key] = {version1: v1_info.get(key), version2: v2_info.get(key)}
        return comparison

    async def tag_version(self, version: str, tag: str, description: Optional[str]=None) -> bool:
        """Tag a version for easy reference.
        
        Args:
            version: Version to tag
            tag: Tag name
            description: Optional description
            
        Returns:
            Success status
        """
        version_info = await self.get_version(version)
        if not version_info:
            self.logger.error(f'Version not found: {version}')
            return False
        if 'tags' not in version_info['metadata']:
            version_info['metadata']['tags'] = {}
        version_info['metadata']['tags'][tag] = {'created_at': datetime.now().isoformat(), 'description': description}
        self._save_version_registry()
        self.logger.info(f"Tagged version {version} as '{tag}'")
        return True