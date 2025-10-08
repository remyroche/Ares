"""
Reproducibility and Scientific Rigor Tracking.

This module ensures complete reproducibility of pre-training pipeline runs by tracking:
1. Git commit SHA and repository state
2. Environment configuration (conda/pip packages)
3. Random seeds and RNG states
4. Dataset checksums
5. Configuration hashes
6. Data lineage graphs (feature/label dependencies)

This enables:
- Exact reproduction of any pipeline run
- Debugging and validation of historical results
- Compliance with scientific reproducibility standards
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger


@dataclass
class GitInfo:
    """Information about Git repository state."""
    
    commit_sha: Optional[str] = None
    branch: Optional[str] = None
    is_dirty: bool = False
    remote_url: Optional[str] = None
    uncommitted_changes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class EnvironmentInfo:
    """Information about execution environment."""
    
    python_version: str
    platform: str
    hostname: str
    cpu_count: int
    packages: Dict[str, str]  # package_name -> version
    env_variables: Dict[str, str]  # Relevant environment variables
    conda_env: Optional[str] = None


@dataclass
class DatasetInfo:
    """Information about dataset."""
    
    path: str
    checksum: str  # SHA256 hash
    size_bytes: int
    num_rows: int
    num_columns: int
    column_names: List[str]
    timestamp_range: Optional[Tuple[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigInfo:
    """Information about configuration."""
    
    config_hash: str  # Hash of configuration
    config_dict: Dict[str, Any]
    config_file_path: Optional[str] = None


@dataclass
class LineageNode:
    """Node in data lineage graph."""
    
    node_id: str
    node_type: str  # 'data', 'feature', 'label', 'model', 'artifact'
    name: str
    dependencies: List[str]  # List of node_ids this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ReproducibilityManifest:
    """Complete reproducibility manifest for a pipeline run."""
    
    run_id: str
    timestamp: str
    git_info: GitInfo
    environment: EnvironmentInfo
    datasets: Dict[str, DatasetInfo]
    configs: Dict[str, ConfigInfo]
    random_seeds: Dict[str, int]
    lineage_graph: List[LineageNode]
    artifacts: Dict[str, str]  # artifact_name -> checksum
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'git_info': asdict(self.git_info),
            'environment': asdict(self.environment),
            'datasets': {k: asdict(v) for k, v in self.datasets.items()},
            'configs': {k: asdict(v) for k, v in self.configs.items()},
            'random_seeds': self.random_seeds,
            'lineage_graph': [asdict(node) for node in self.lineage_graph],
            'artifacts': self.artifacts,
            'metadata': self.metadata
        }
    
    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> ReproducibilityManifest:
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct objects
        git_info = GitInfo(**data['git_info'])
        environment = EnvironmentInfo(**data['environment'])
        datasets = {k: DatasetInfo(**v) for k, v in data['datasets'].items()}
        configs = {k: ConfigInfo(**v) for k, v in data['configs'].items()}
        lineage_graph = [LineageNode(**node) for node in data['lineage_graph']]
        
        return cls(
            run_id=data['run_id'],
            timestamp=data['timestamp'],
            git_info=git_info,
            environment=environment,
            datasets=datasets,
            configs=configs,
            random_seeds=data['random_seeds'],
            lineage_graph=lineage_graph,
            artifacts=data['artifacts'],
            metadata=data.get('metadata', {})
        )


class ReproducibilityTracker:
    """
    Tracks all information needed for reproducibility.
    """
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the reproducibility tracker.
        
        Args:
            run_id: Unique identifier for this run (generated if not provided)
            logger: Optional logger instance
        """
        self.run_id = run_id or self._generate_run_id()
        self.logger = logger or system_logger.getChild('ReproducibilityTracker')
        
        self.datasets: Dict[str, DatasetInfo] = {}
        self.configs: Dict[str, ConfigInfo] = {}
        self.random_seeds: Dict[str, int] = {}
        self.lineage_nodes: List[LineageNode] = []
        self.artifacts: Dict[str, str] = {}
        
        self.logger.info(f"Initialized reproducibility tracker with run_id: {self.run_id}")
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.sha256(os.urandom(32)).hexdigest()[:8]
        return f"run_{timestamp}_{random_suffix}"
    
    def capture_git_info(self, repo_path: Optional[Path] = None) -> GitInfo:
        """
        Capture Git repository information.
        
        Args:
            repo_path: Path to git repository (defaults to current directory)
        
        Returns:
            GitInfo object
        """
        if repo_path is None:
            repo_path = Path.cwd()
        
        git_info = GitInfo()
        
        try:
            # Check if in git repository
            result = subprocess.run(
                ['git', 'rev-parse', '--is-inside-work-tree'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                self.logger.warning("Not in a git repository")
                return git_info
            
            # Get commit SHA
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                git_info.commit_sha = result.stdout.strip()
            
            # Get branch name
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                git_info.branch = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                changes = result.stdout.strip()
                if changes:
                    git_info.is_dirty = True
                    git_info.uncommitted_changes = changes.split('\n')[:10]  # Limit to 10 files
            
            # Get remote URL
            result = subprocess.run(
                ['git', 'config', '--get', 'remote.origin.url'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                git_info.remote_url = result.stdout.strip()
            
            # Get tags
            result = subprocess.run(
                ['git', 'tag', '--points-at', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                tags = result.stdout.strip()
                if tags:
                    git_info.tags = tags.split('\n')
            
            self.logger.info(
                f"Git info captured: commit={git_info.commit_sha[:8]}, "
                f"branch={git_info.branch}, dirty={git_info.is_dirty}"
            )
        
        except Exception as e:
            self.logger.warning(f"Could not capture git info: {e}")
        
        return git_info
    
    def capture_environment(
        self,
        include_all_packages: bool = False,
        relevant_env_vars: Optional[List[str]] = None
    ) -> EnvironmentInfo:
        """
        Capture environment information.
        
        Args:
            include_all_packages: If True, include all installed packages
            relevant_env_vars: List of environment variable names to capture
        
        Returns:
            EnvironmentInfo object
        """
        # Python version
        python_version = sys.version
        
        # Platform info
        platform_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        
        # Hostname
        hostname = socket.gethostname()
        
        # CPU count
        cpu_count = os.cpu_count() or 1
        
        # Packages
        packages = {}
        if include_all_packages:
            try:
                import pkg_resources
                for dist in pkg_resources.working_set:
                    packages[dist.project_name] = dist.version
            except Exception as e:
                self.logger.warning(f"Could not list all packages: {e}")
        else:
            # Capture key packages
            key_packages = ['numpy', 'pandas', 'scikit-learn', 'torch', 'tensorflow']
            for pkg in key_packages:
                try:
                    module = __import__(pkg.replace('-', '_'))
                    version = getattr(module, '__version__', 'unknown')
                    packages[pkg] = version
                except ImportError:
                    pass
        
        # Environment variables
        env_variables = {}
        if relevant_env_vars is None:
            relevant_env_vars = ['PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS']
        
        for var in relevant_env_vars:
            value = os.getenv(var)
            if value is not None:
                env_variables[var] = value
        
        # Conda environment
        conda_env = os.getenv('CONDA_DEFAULT_ENV')
        
        env_info = EnvironmentInfo(
            python_version=python_version,
            platform=platform_info,
            hostname=hostname,
            cpu_count=cpu_count,
            packages=packages,
            env_variables=env_variables,
            conda_env=conda_env
        )
        
        self.logger.info(
            f"Environment captured: Python {python_version[:10]}, "
            f"{len(packages)} packages tracked"
        )
        
        return env_info
    
    def register_dataset(
        self,
        name: str,
        data: Optional[pd.DataFrame] = None,
        path: Optional[Path] = None
    ) -> DatasetInfo:
        """
        Register a dataset and compute its checksum.
        
        Args:
            name: Dataset name
            data: Optional DataFrame (either data or path must be provided)
            path: Optional path to dataset file
        
        Returns:
            DatasetInfo object
        """
        if data is None and path is None:
            raise ValueError("Either data or path must be provided")
        
        # Compute checksum
        if data is not None:
            # Checksum from DataFrame
            checksum = self._compute_dataframe_checksum(data)
            size_bytes = data.memory_usage(deep=True).sum()
            num_rows = len(data)
            num_columns = len(data.columns)
            column_names = data.columns.tolist()
            
            # Timestamp range
            if isinstance(data.index, pd.DatetimeIndex):
                timestamp_range = (
                    data.index.min().isoformat(),
                    data.index.max().isoformat()
                )
            else:
                timestamp_range = None
            
            path_str = str(path) if path else "in_memory"
        else:
            # Checksum from file
            checksum = self._compute_file_checksum(path)
            size_bytes = path.stat().st_size
            # These would require loading the file
            num_rows = -1
            num_columns = -1
            column_names = []
            timestamp_range = None
            path_str = str(path)
        
        dataset_info = DatasetInfo(
            path=path_str,
            checksum=checksum,
            size_bytes=size_bytes,
            num_rows=num_rows,
            num_columns=num_columns,
            column_names=column_names,
            timestamp_range=timestamp_range
        )
        
        self.datasets[name] = dataset_info
        
        self.logger.info(
            f"Dataset registered: {name}, checksum={checksum[:16]}, "
            f"rows={num_rows}, cols={num_columns}"
        )
        
        return dataset_info
    
    def register_config(
        self,
        name: str,
        config: Dict[str, Any],
        config_file_path: Optional[Path] = None
    ) -> ConfigInfo:
        """
        Register a configuration and compute its hash.
        
        Args:
            name: Config name
            config: Configuration dictionary
            config_file_path: Optional path to config file
        
        Returns:
            ConfigInfo object
        """
        config_hash = self._compute_dict_hash(config)
        
        config_info = ConfigInfo(
            config_hash=config_hash,
            config_dict=config,
            config_file_path=str(config_file_path) if config_file_path else None
        )
        
        self.configs[name] = config_info
        
        self.logger.info(f"Config registered: {name}, hash={config_hash[:16]}")
        
        return config_info
    
    def register_random_seed(self, name: str, seed: int) -> None:
        """Register a random seed."""
        self.random_seeds[name] = seed
        self.logger.debug(f"Random seed registered: {name}={seed}")
    
    def add_lineage_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LineageNode:
        """
        Add a node to the data lineage graph.
        
        Args:
            node_id: Unique node identifier
            node_type: Type of node
            name: Human-readable name
            dependencies: List of node IDs this depends on
            metadata: Optional metadata
        
        Returns:
            LineageNode object
        """
        node = LineageNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.lineage_nodes.append(node)
        
        self.logger.debug(
            f"Lineage node added: {node_id} ({node_type}) -> {len(dependencies or [])} dependencies"
        )
        
        return node
    
    def register_artifact(self, name: str, checksum: str) -> None:
        """Register an artifact and its checksum."""
        self.artifacts[name] = checksum
        self.logger.debug(f"Artifact registered: {name}, checksum={checksum[:16]}")
    
    def create_manifest(
        self,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReproducibilityManifest:
        """
        Create complete reproducibility manifest.
        
        Args:
            metadata: Optional additional metadata
        
        Returns:
            ReproducibilityManifest object
        """
        git_info = self.capture_git_info()
        environment = self.capture_environment()
        
        manifest = ReproducibilityManifest(
            run_id=self.run_id,
            timestamp=datetime.utcnow().isoformat(),
            git_info=git_info,
            environment=environment,
            datasets=self.datasets,
            configs=self.configs,
            random_seeds=self.random_seeds,
            lineage_graph=self.lineage_nodes,
            artifacts=self.artifacts,
            metadata=metadata or {}
        )
        
        self.logger.info(
            f"Manifest created: {len(self.datasets)} datasets, "
            f"{len(self.configs)} configs, {len(self.artifacts)} artifacts, "
            f"{len(self.lineage_nodes)} lineage nodes"
        )
        
        return manifest
    
    def _compute_dataframe_checksum(self, df: pd.DataFrame) -> str:
        """Compute SHA256 checksum of DataFrame."""
        # Use a deterministic representation
        try:
            # Convert to CSV string for hashing
            csv_str = df.to_csv(index=True, header=True)
            return hashlib.sha256(csv_str.encode('utf-8')).hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not compute dataframe checksum: {e}")
            return "unknown"
    
    def _compute_file_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of file."""
        try:
            sha256 = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not compute file checksum: {e}")
            return "unknown"
    
    def _compute_dict_hash(self, d: Dict[str, Any]) -> str:
        """Compute hash of dictionary."""
        try:
            # Convert to JSON string with sorted keys
            json_str = json.dumps(d, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not compute dict hash: {e}")
            return "unknown"


def create_reproducibility_tracker(
    run_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> ReproducibilityTracker:
    """
    Factory function to create a reproducibility tracker.
    
    Args:
        run_id: Optional run identifier
        logger: Optional logger instance
    
    Returns:
        ReproducibilityTracker instance
    """
    return ReproducibilityTracker(run_id=run_id, logger=logger)


__all__ = [
    'ReproducibilityTracker',
    'ReproducibilityManifest',
    'GitInfo',
    'EnvironmentInfo',
    'DatasetInfo',
    'ConfigInfo',
    'LineageNode',
    'create_reproducibility_tracker',
]