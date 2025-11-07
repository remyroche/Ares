"""
Artifact Router - Intelligent routing for different storage formats

This module provides intelligent routing of artifacts to appropriate storage systems
based on data type, content, and use case.

Storage System Responsibilities:
--------------------------------

1. JSON (via serialization_utils.py):
   - Configuration files
   - Metadata dictionaries
   - Small data structures
   - Human-readable data

2. Pickle (via serialization_utils.py):
   - ML models (scikit-learn, xgboost, etc.)
   - Complex Python objects
   - Arbitrary data structures
   - Non-tabular data

3. Parquet (via kline_parquet.py):
   - Historical OHLCV/klines data
   - Raw market data
   - Time-series data with optimization
   - Stored in historical_data/ directory

4. HDF5 Versioned (via versioned_artifacts/):
   - Feature DataFrames for training
   - ML predictions and scores
   - Large tabular datasets requiring versioning
   - Training data with view-based access

Usage:
    router = ArtifactRouter(base_dir="artifacts")

    # Saves to appropriate format automatically
    router.save(model, "trained_model")  # -> Pickle
    router.save(config_dict, "config")  # -> JSON
    router.save(ohlcv_df, "historical_klines", data_category="historical")  # -> Parquet
    router.save(features_df, "training_features", data_category="features")  # -> HDF5
"""

import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.utils.serialization_utils import (
    JSONSerializer,
    PickleSerializer,
    save_json,
    save_pickle,
    load_json,
    load_pickle
)
from src.utils.kline_parquet import KlinesParquetManager
from src.utils.versioned_artifacts import VersionedArtifactStore


class ArtifactRouter:
    """
    Intelligent router for artifact storage based on data type and use case.

    Routes artifacts to the most appropriate storage system:
    - JSON for configs and metadata
    - Pickle for ML models and complex objects
    - Parquet for historical OHLCV data
    - HDF5 for versioned feature DataFrames
    """

    def __init__(
        self,
        base_dir: str = "artifacts",
        versioned_store_dir: str = "versioned_artifacts",
        historical_data_dir: str = "historical_data",
        enable_versioned_artifacts: bool = True
    ):
        """
        Initialize the artifact router.

        Args:
            base_dir: Base directory for traditional artifacts (JSON/Pickle)
            versioned_store_dir: Directory for versioned HDF5 stores
            historical_data_dir: Directory for historical parquet data
            enable_versioned_artifacts: Whether to use versioned artifacts
        """
        self.base_dir = Path(base_dir)
        self.versioned_store_dir = Path(versioned_store_dir)
        self.historical_data_dir = Path(historical_data_dir)
        self.enable_versioned_artifacts = enable_versioned_artifacts

        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.versioned_store_dir.mkdir(parents=True, exist_ok=True)
        self.historical_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize managers
        self.klines_manager = KlinesParquetManager(
            config=type('Config', (), {'base_dir': str(self.historical_data_dir)})()
        )

        # Versioned stores cache (lazy initialization)
        self._versioned_stores: Dict[str, VersionedArtifactStore] = {}

        self.logger = logging.getLogger("ArtifactRouter")

    def _is_json_serializable(self, data: Any, depth: int = 0, max_depth: int = 10) -> bool:
        """
        Check if data is JSON serializable with depth limit to avoid infinite recursion.

        Args:
            data: Data to check
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            True if data is JSON serializable
        """
        if depth > max_depth:
            return False

        # Simple types
        if isinstance(data, (str, int, float, bool, type(None))):
            return True

        # Lists/tuples
        if isinstance(data, (list, tuple)):
            # Check sample if large
            sample_size = min(len(data), 100)
            sample = data[:sample_size] if sample_size > 0 else []
            return all(self._is_json_serializable(item, depth + 1, max_depth) for item in sample)

        # Dicts
        if isinstance(data, dict):
            # Check keys are strings and values are serializable
            if not all(isinstance(k, str) for k in data.keys()):
                return False
            # Check sample if large
            sample_size = min(len(data), 100)
            sample_items = list(data.items())[:sample_size]
            return all(self._is_json_serializable(v, depth + 1, max_depth) for k, v in sample_items)

        # Try actual JSON serialization as final check
        try:
            import json
            json.dumps(data, default=str)
            return True
        except (TypeError, ValueError):
            return False

    def _detect_format(
        self,
        data: Any,
        artifact_name: str,
        artifact_type: Optional[str] = None,
        data_category: Optional[str] = None
    ) -> str:
        """
        Detect the most appropriate storage format for the data.

        Uses hierarchy when in doubt: JSON → Pickle → Parquet → HDF5
        (Prefer simpler formats first, more complex formats when needed)

        Args:
            data: Data to store
            artifact_name: Name of the artifact
            artifact_type: Explicit type hint
            data_category: Category hint (historical, features, predictions, config, model)

        Returns:
            Format string: 'json', 'pickle', 'parquet', or 'hdf5_versioned'
        """
        # Explicit routing based on data_category (highest priority)
        if data_category:
            category_map = {
                'config': 'json',
                'metadata': 'json',
                'parameters': 'json',
                'hpo': 'json',
                'model': 'pickle',
                'historical': 'parquet',
                'klines': 'parquet',
                'ohlcv': 'parquet',
                'features': 'hdf5_versioned',
                'predictions': 'hdf5_versioned',
                'training': 'hdf5_versioned',
            }
            if data_category.lower() in category_map:
                return category_map[data_category.lower()]

        # Enhanced name-based routing with comprehensive keywords
        name_lower = artifact_name.lower()

        # JSON keywords (configs, parameters, hyperparameters)
        json_keywords = ['config', 'metadata', 'params', 'settings', 'parameters', 'hpo',
                        'hyperparameter', 'tuning', 'grid', 'search']
        if any(kw in name_lower for kw in json_keywords):
            return 'json'

        # Pickle keywords (ML models, ensembles, SR levels)
        pickle_keywords = ['model', 'estimator', 'classifier', 'regressor', 'ml',
                          'base', 'ensemble', 'sr', 'stacked', 'voting', 'bagging']
        if any(kw in name_lower for kw in pickle_keywords):
            return 'pickle'

        # Parquet keywords (historical data)
        parquet_keywords = ['historical', 'klines', 'ohlcv', 'candles', 'market_data', 'raw_data']
        if any(kw in name_lower for kw in parquet_keywords):
            return 'parquet'

        # HDF5 keywords (features, clusters, regimes, labels)
        hdf5_keywords = ['feature', 'prediction', 'score', 'training', 'cluster',
                        'label', 'target', 'regime', 'engineered', 'selected']
        if any(kw in name_lower for kw in hdf5_keywords):
            return 'hdf5_versioned' if self.enable_versioned_artifacts else 'pickle'

        # Type-based routing with complexity analysis
        # 1. Check for dictionaries (JSON vs Pickle based on complexity)
        if isinstance(data, dict) and not isinstance(data, pd.DataFrame):
            # Try JSON first (hierarchy preference)
            if self._is_json_serializable(data):
                return 'json'
            else:
                # Complex dict -> Pickle
                return 'pickle'

        # 2. Check for lists/tuples
        if isinstance(data, (list, tuple)):
            if len(data) == 0:
                return 'json'  # Empty collections

            # Small collections -> try JSON first
            if len(data) < 1000:
                if self._is_json_serializable(data):
                    return 'json'

            # Large or complex collections -> Pickle
            return 'pickle'

        # 3. DataFrame routing
        if isinstance(data, pd.DataFrame):
            # Empty DataFrame → JSON
            if data.empty:
                return 'json'

            # Parquet if: less than 10 columns AND has historical keywords
            parquet_df_keywords = ['historical', 'kline', 'klines', 'ohlcv', 'market_data', 'raw_data']
            if len(data.columns) < 10 and any(kw in name_lower for kw in parquet_df_keywords):
                return 'parquet'

            # All other DataFrames → HDF5 (features, predictions, training data, etc.)
            return 'hdf5_versioned' if self.enable_versioned_artifacts else 'pickle'

        # 4. ML models (detected by module name AND semantic keywords)
        # Module-based detection
        model_types = ('sklearn', 'xgboost', 'lightgbm', 'catboost', 'keras',
                      'tensorflow', 'torch', 'pytorch')
        is_ml_model_type = any(model_type in str(type(data).__module__).lower() for model_type in model_types)

        # Semantic keyword detection for ML models
        ml_model_keywords = ['model', 'estimator', 'classifier', 'regressor', 'ml',
                            'base', 'ensemble', 'stacked']
        has_ml_keyword = any(kw in name_lower for kw in ml_model_keywords)

        if is_ml_model_type or has_ml_keyword:
            return 'pickle'

        # 5. NumPy arrays
        if hasattr(data, '__array__'):  # NumPy arrays
            # Small arrays might be JSON serializable
            try:
                if data.size < 1000 and data.ndim <= 2:
                    data_list = data.tolist()
                    if self._is_json_serializable(data_list):
                        return 'json'
            except:
                pass
            # Default to pickle for arrays
            return 'pickle'

        # 6. Simple scalar types -> JSON
        if isinstance(data, (str, int, float, bool, type(None))):
            return 'json'

        # Default: Pickle for any complex/unknown objects
        # (Follows hierarchy: JSON first, but if not suitable, use Pickle)
        return 'pickle'

    def save(
        self,
        data: Any,
        artifact_name: str,
        artifact_type: Optional[str] = None,
        data_category: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Save artifact using the most appropriate storage format.

        Args:
            data: Data to save
            artifact_name: Name for the artifact
            artifact_type: Type hint (optional)
            data_category: Category hint for routing (optional)
            context: Context dict with symbol, exchange, timeframe, etc.
            metadata: Additional metadata
            **kwargs: Additional arguments for specific storage systems

        Returns:
            Path where artifact was saved
        """
        from src.utils.tprint import tprint

        # Get input data type information
        data_type_name = type(data).__name__
        if isinstance(data, pd.DataFrame):
            input_info = f"DataFrame ({len(data)} rows × {len(data.columns)} cols)"
        elif isinstance(data, (list, tuple)):
            input_info = f"{data_type_name} ({len(data)} items)"
        elif hasattr(data, '__array__'):  # NumPy array
            input_info = f"ndarray (shape: {data.shape})"
        else:
            input_info = data_type_name

        # Detect format
        format_type = self._detect_format(data, artifact_name, artifact_type, data_category)

        # Log routing decision
        self.logger.info(f"Routing '{artifact_name}' ({input_info}) to {format_type} storage")
        tprint(f"🔀 Router: {input_info} → {format_type.upper()}")

        # Route to appropriate storage
        if format_type == 'json':
            path = self._save_json(data, artifact_name, metadata)

        elif format_type == 'pickle':
            path = self._save_pickle(data, artifact_name, metadata)

        elif format_type == 'parquet':
            path = self._save_parquet(data, artifact_name, context, metadata)

        elif format_type == 'hdf5_versioned':
            path = self._save_hdf5_versioned(data, artifact_name, context, metadata)

        else:
            raise ValueError(f"Unknown format type: {format_type}")

        # Log final path
        tprint(f"💾 Saved '{artifact_name}' → {path}")
        return path

    def _save_json(self, data: Any, artifact_name: str, metadata: Optional[Dict] = None) -> str:
        """Save data as JSON."""
        filepath = self.base_dir / f"{artifact_name}.json"

        # Combine data and metadata if needed
        if metadata:
            save_data = {
                'data': data,
                'metadata': metadata,
                'saved_at': datetime.now().isoformat()
            }
        else:
            save_data = data

        success = save_json(save_data, str(filepath))
        if success:
            self.logger.info(f"Saved JSON: {filepath}")
            return str(filepath)
        else:
            raise IOError(f"Failed to save JSON to {filepath}")

    def _save_pickle(self, data: Any, artifact_name: str, metadata: Optional[Dict] = None) -> str:
        """Save data as Pickle."""
        filepath = self.base_dir / f"{artifact_name}.pkl"

        # Combine data and metadata if needed
        if metadata:
            save_data = {
                'data': data,
                'metadata': metadata,
                'saved_at': datetime.now().isoformat()
            }
        else:
            save_data = data

        success = save_pickle(save_data, str(filepath))
        if success:
            self.logger.info(f"Saved Pickle: {filepath}")
            return str(filepath)
        else:
            raise IOError(f"Failed to save Pickle to {filepath}")

    def _save_parquet(
        self,
        data: pd.DataFrame,
        artifact_name: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save DataFrame as Parquet using KlinesParquetManager."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Parquet storage requires DataFrame, got {type(data)}")

        context = context or {}
        symbol = context.get('symbol', 'UNKNOWN')
        exchange = context.get('exchange', 'binance')
        interval = context.get('timeframe', '15m')

        success = self.klines_manager.store_klines(
            df=data,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            metadata=metadata
        )

        if success:
            # Get the stored path
            filepath = self.klines_manager._get_storage_path(
                symbol, exchange, interval,
                self.klines_manager._generate_batch_id(symbol, exchange, interval)
            )
            self.logger.info(f"Saved Parquet: {filepath}")
            return str(filepath)
        else:
            raise IOError(f"Failed to save Parquet for {artifact_name}")

    def _save_hdf5_versioned(
        self,
        data: pd.DataFrame,
        artifact_name: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save DataFrame to versioned HDF5 store."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"HDF5 versioned storage requires DataFrame, got {type(data)}")

        context = context or {}

        # Get or create versioned store for this context
        store = self._get_versioned_store(context)

        # Generate version name
        version_name = f"{artifact_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Prepare metadata
        artifact_metadata = {
            'artifact_name': artifact_name,
            'created_at': datetime.now().isoformat(),
            **(context or {}),
            **(metadata or {})
        }

        # Add data to store
        view = store.add_data(
            data=data,
            version_name=version_name,
            metadata=artifact_metadata
        )

        filepath = str(store.store_path / f"{version_name}.h5")
        self.logger.info(f"Saved HDF5 versioned: {filepath}")
        return filepath

    def _get_versioned_store(self, context: Dict[str, Any]) -> VersionedArtifactStore:
        """Get or create a versioned store for the given context."""
        symbol = context.get('symbol', 'UNKNOWN')
        exchange = context.get('exchange', 'binance')
        timeframe = context.get('timeframe', '15m')
        direction = context.get('direction', 'long')
        model = context.get('model', 'analyst')

        # Create unique key for this context
        store_key = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"

        if store_key not in self._versioned_stores:
            store_path = self.versioned_store_dir / store_key
            self._versioned_stores[store_key] = VersionedArtifactStore(
                store_path=store_path,
                auto_version=True,
                enable_row_versioning=True
            )

        return self._versioned_stores[store_key]

    def load(
        self,
        artifact_name: str,
        artifact_type: Optional[str] = None,
        data_category: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Load artifact from appropriate storage.

        Args:
            artifact_name: Name of artifact to load
            artifact_type: Type hint (optional)
            data_category: Category hint (optional)
            context: Context dict with symbol, exchange, etc.

        Returns:
            Loaded data
        """
        # Try to detect format from file extension or name
        json_path = self.base_dir / f"{artifact_name}.json"
        pickle_path = self.base_dir / f"{artifact_name}.pkl"

        if json_path.exists():
            data = load_json(str(json_path))
            # Extract data from wrapper if present
            if isinstance(data, dict) and 'data' in data and 'metadata' in data:
                return data['data']
            return data

        elif pickle_path.exists():
            data = load_pickle(str(pickle_path))
            # Extract data from wrapper if present
            if isinstance(data, dict) and 'data' in data and 'metadata' in data:
                return data['data']
            return data

        # Try parquet
        if data_category in ['historical', 'klines', 'ohlcv'] or \
           any(kw in artifact_name.lower() for kw in ['historical', 'klines', 'ohlcv']):
            try:
                context = context or {}
                symbol = context.get('symbol', 'UNKNOWN')
                exchange = context.get('exchange', 'binance')
                interval = context.get('timeframe', '15m')

                data = self.klines_manager.load_klines(symbol, exchange, interval)
                if not data.empty:
                    return data
            except Exception as e:
                self.logger.warning(f"Failed to load from parquet: {e}")

        # Try versioned HDF5
        if self.enable_versioned_artifacts and context:
            try:
                store = self._get_versioned_store(context)
                versions = store.list_versions()
                matching = [v for v in versions if artifact_name in v]

                if matching:
                    version_name = sorted(matching)[-1]
                    view = store.get_view(version_name)
                    return view.materialize()
            except Exception as e:
                self.logger.warning(f"Failed to load from versioned store: {e}")

        raise FileNotFoundError(f"Artifact '{artifact_name}' not found in any storage")

    def get_storage_info(self, artifact_name: str) -> Dict[str, Any]:
        """
        Get information about where an artifact is stored.

        Args:
            artifact_name: Name of artifact

        Returns:
            Dict with storage information
        """
        info = {
            'artifact_name': artifact_name,
            'found': False,
            'format': None,
            'path': None,
            'size_bytes': None
        }

        # Check JSON
        json_path = self.base_dir / f"{artifact_name}.json"
        if json_path.exists():
            info['found'] = True
            info['format'] = 'json'
            info['path'] = str(json_path)
            info['size_bytes'] = json_path.stat().st_size
            return info

        # Check Pickle
        pickle_path = self.base_dir / f"{artifact_name}.pkl"
        if pickle_path.exists():
            info['found'] = True
            info['format'] = 'pickle'
            info['path'] = str(pickle_path)
            info['size_bytes'] = pickle_path.stat().st_size
            return info

        return info
