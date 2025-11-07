"""
Base Step Class for Autonomous Pipeline Steps

This module provides the abstract base class that all pipeline steps must inherit from.
Each step becomes autonomous with standardized artifact management and outcome file generation.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import traceback

from src.utils.artifact_manager import ArtifactManager
from src.utils.tprint import tprint
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.temporal_validation import require_datetime_index


class BaseStep(ABC):
    """
    Abstract base class for all autonomous pipeline steps.
    
    Each step must:
    - Inherit from this class
    - Implement the execute() method
    - Use artifact_manager for all data I/O
    - Generate Markdown outcome files
    - Be callable only via launcher (no standalone CLI)
    """
    
    def __init__(self, step_name: str, use_versioned_artifacts: bool = True):
        """
        Initialize the base step with lazy loading to reduce startup overhead.

        Args:
            step_name: Unique name for this step (used for artifact paths and outcomes)
            use_versioned_artifacts: Whether to use versioned artifacts system (default: True)
        """
        self.step_name = step_name
        self.logger = logging.getLogger(f"ares.step.{step_name}")
        tprint(f"🛠️ Initialized BaseStep for '{self.step_name}'")

        # Defer heavy initialization until needed (lazy loading)
        self._artifact_manager = None
        self._quality_assessor = None
        self._versioned_store = None
        self.use_versioned_artifacts = use_versioned_artifacts

        # Store context for lazy initialization
        self._current_context = {
            'step_name': step_name,
            'datetime': datetime.now()
        }

        # Mode detection for differentiated execution
        self.execution_mode = None  # Will be set by _detect_execution_mode

    @property
    def artifact_manager(self):
        """Lazy initialization of artifact manager."""
        if self._artifact_manager is None:
            self._artifact_manager = ArtifactManager(config={})
            # Apply deferred context
            self._artifact_manager.set_context(**self._current_context)
        return self._artifact_manager

    @property
    def versioned_store(self):
        """Lazy initialization of versioned artifact store."""
        if self._versioned_store is None and self.use_versioned_artifacts:
            # Extract context for store path
            symbol = self._current_context.get('symbol', 'UNKNOWN')
            exchange = self._current_context.get('exchange', 'binance')
            direction = self._current_context.get('direction', 'long')

            # Create store path
            store_name = f"{symbol}_{exchange}_{direction}"
            store_path = os.path.join("src/utils/versioned_artifacts", store_name)

            self._versioned_store = VersionedArtifactStore(
                store_path=store_path,
                auto_version=True,
                enable_row_versioning=True
            )
            tprint(f"📦 Initialized VersionedArtifactStore at {store_path}")
        return self._versioned_store

    @property
    def quality_assessor(self):
        """Lazy initialization of quality assessor."""
        if self._quality_assessor is None:
            self._quality_assessor = create_cluster_quality_assessor(
                artifact_manager=self.artifact_manager
            )
        return self._quality_assessor

    def set_context(self, **kwargs):
        """Set context for lazy initialization."""
        tprint(
            f"🧭 Updating context for '{self.step_name}' with {list(kwargs.keys())}"
        )

        # Check if use_versioned_artifacts flag is being set
        if 'use_versioned_artifacts' in kwargs:
            self.use_versioned_artifacts = kwargs.pop('use_versioned_artifacts')

        self._current_context.update(kwargs)
        if self._artifact_manager is not None:
            self._artifact_manager.set_context(**self._current_context)
            tprint(
                f"🔄 Applied new context to ArtifactManager for '{self.step_name}'"
            )

        # If versioned store already initialized, reinitialize with new context
        if self._versioned_store is not None and self.use_versioned_artifacts:
            symbol = self._current_context.get('symbol', 'UNKNOWN')
            exchange = self._current_context.get('exchange', 'binance')
            direction = self._current_context.get('direction', 'long')
            store_name = f"{symbol}_{exchange}_{direction}"
            store_path = os.path.join("src/utils/versioned_artifacts", store_name)

            self._versioned_store = VersionedArtifactStore(
                store_path=store_path,
                auto_version=True,
                enable_row_versioning=True
            )
            tprint(f"🔄 Reinitialized VersionedArtifactStore with new context")

    def _detect_execution_mode(self, config: Dict[str, Any]) -> str:
        """
        Detect execution mode based on launcher arguments and step context.
        
        This method can be overridden by subclasses for more specific mode detection.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            'analyst' or 'tactician'
        """
        # Primary detection: Check current step name for Tactician training steps
        is_tactician_training_step = (
            'tactician_base_training' in self.step_name or
            'tactician_ensemble_training' in self.step_name or
            'tactician' in self.step_name.lower()
        )
        
        # Secondary detection: Check execution context
        tactician_execution_context = config.get('execution_context', '').lower()
        is_tactician_context = 'tactician' in tactician_execution_context
        
        # Tertiary detection: Check for explicit mode setting
        explicit_mode = config.get('interaction_generation_mode', '').lower()
        
        # Quaternary detection: Check for Tactician-specific configuration
        tactician_mode_config = config.get('tactician_mode', False)
        
        # Determine mode
        if (is_tactician_training_step or is_tactician_context or 
            explicit_mode == 'tactician' or tactician_mode_config):
            mode = 'tactician'  # Uses MI-based selection
        else:
            mode = 'analyst'  # Uses CMI-based selection
        
        self.logger.info(f"Execution mode detected: {mode}")
        tprint(f"🎯 Execution mode for '{self.step_name}': {mode}")
        return mode
        
    @abstractmethod
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the step logic.
        
        Args:
            config: Configuration dictionary containing all necessary parameters
                   (symbol, exchange, timeframes, execution_mode, etc.)
        
        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': list of artifact paths/metadata created
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
            - 'execution_time': float seconds taken to execute
        """
        pass
    
    def _save_artifact(self, data: Any, artifact_name: str,
                      artifact_type: str = "data",
                      compression: str = "auto",
                      metadata: Optional[Dict] = None,
                      operation_name: Optional[str] = None,
                      tags: Optional[Dict] = None) -> str:
        """
        Save an artifact using versioned artifact store (default) or traditional artifact manager.

        By default, uses versioned artifacts except for:
        - historical_data artifacts (use historical_data/ directory)
        - regime clustering artifacts (excluding regime_ensemble_training)

        Args:
            data: Data to save (DataFrame, dict, model, etc.)
            artifact_name: Name for the artifact
            artifact_type: Type of artifact ("data", "model", "metadata", etc.)
            compression: Compression method ("auto", "gzip", "lz4", "none")
            metadata: Additional metadata to store with artifact
            operation_name: Name of operation for versioned artifacts tagging
            tags: Additional tags for versioned artifacts

        Returns:
            Path where artifact was saved
        """
        try:
            tprint(
                f"💾 Saving artifact '{artifact_name}' (type: {artifact_type}) for '{self.step_name}'"
            )

            # Determine if we should use versioned artifacts
            use_versioned = self.use_versioned_artifacts

            # Exclusion rules
            if 'historical_data' in artifact_name:
                use_versioned = False
                tprint(f"📁 Using traditional storage for historical_data artifact")
            elif 'regime_cluster' in artifact_name and 'regime_ensemble_training' not in self.step_name:
                use_versioned = False
                tprint(f"📁 Using traditional storage for regime clustering artifact")

            # Save using appropriate system
            if use_versioned and self.versioned_store is not None:
                import pandas as pd
                # Convert to DataFrame if needed
                if isinstance(data, pd.DataFrame):
                    # Use add_columns_with_tags for DataFrame with columns
                    version_name = f"{artifact_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    # Add data to store
                    view = self.versioned_store.add_data(
                        data=data,
                        version_name=version_name,
                        metadata={
                            'artifact_name': artifact_name,
                            'artifact_type': artifact_type,
                            'step_name': self.step_name,
                            **(metadata or {})
                        }
                    )

                    # If operation_name provided, tag columns
                    if operation_name:
                        columns_dict = {col: data[col].values for col in data.columns}
                        self.versioned_store.add_columns_with_tags(
                            columns=columns_dict,
                            version_name=version_name,
                            operation_name=operation_name,
                            tags=tags or {}
                        )

                    artifact_path = str(self.versioned_store.store_path / f"{version_name}.h5")
                    tprint(f"✅ Saved artifact '{artifact_name}' to versioned store: {artifact_path}")
                else:
                    # Fall back to traditional artifact manager for non-DataFrame data
                    artifact_path = self.artifact_manager.save(
                        data=data,
                        artifact_name=artifact_name,
                        artifact_type=artifact_type,
                        compression=compression,
                        metadata=metadata
                    )
                    tprint(f"✅ Saved non-DataFrame artifact '{artifact_name}' via traditional manager: {artifact_path}")
            else:
                # Use traditional artifact manager
                artifact_path = self.artifact_manager.save(
                    data=data,
                    artifact_name=artifact_name,
                    artifact_type=artifact_type,
                    compression=compression,
                    metadata=metadata
                )
                tprint(f"✅ Saved artifact '{artifact_name}' via traditional manager: {artifact_path}")

            self.logger.info(f"Saved artifact: {artifact_name} -> {artifact_path}")
            return artifact_path

        except Exception as e:
            self.logger.error(f"Failed to save artifact {artifact_name}: {e}")
            tprint(
                f"❌ Failed to save artifact '{artifact_name}' for '{self.step_name}': {e}"
            )
            raise
    
    def _get_artifact(self, artifact_name: str,
                     artifact_type: str = "data",
                     operation_name: Optional[str] = None) -> Any:
        """
        Retrieve an artifact using versioned artifact store (default) or traditional artifact manager.

        By default, uses versioned artifacts except for:
        - historical_data artifacts (use historical_data/ directory)
        - regime clustering artifacts (excluding regime_ensemble_training)

        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve
            operation_name: If provided, retrieve view by operation name

        Returns:
            Retrieved data
        """
        try:
            tprint(
                f"📂 Retrieving artifact '{artifact_name}' (type: {artifact_type}) for '{self.step_name}'"
            )

            # Determine if we should use versioned artifacts
            use_versioned = self.use_versioned_artifacts

            # Exclusion rules (same as _save_artifact)
            if 'historical_data' in artifact_name:
                use_versioned = False
                tprint(f"📁 Using traditional storage for historical_data artifact")
            elif 'regime_cluster' in artifact_name and 'regime_ensemble_training' not in self.step_name:
                use_versioned = False
                tprint(f"📁 Using traditional storage for regime clustering artifact")

            # Retrieve using appropriate system
            if use_versioned and self.versioned_store is not None:
                try:
                    # Try to get from versioned store
                    if operation_name:
                        # Get view by operation
                        view = self.versioned_store.get_view_by_operation(operation_name)
                        data = view.materialize()
                        tprint(f"✅ Retrieved artifact '{artifact_name}' by operation '{operation_name}' from versioned store")
                    else:
                        # Get most recent version matching artifact name
                        versions = self.versioned_store.list_versions()
                        matching = [v for v in versions if artifact_name in v]

                        if matching:
                            version_name = sorted(matching)[-1]
                            view = self.versioned_store.get_view(version_name)
                            data = view.materialize()
                            tprint(f"✅ Retrieved artifact '{artifact_name}' (version: {version_name}) from versioned store")
                        else:
                            # Fall back to traditional artifact manager
                            tprint(f"⚠️ Artifact '{artifact_name}' not found in versioned store, trying traditional manager")
                            data, resolved_path = self.artifact_manager.get_artifact(
                                artifact_name=artifact_name,
                                artifact_type=artifact_type,
                                return_path=True
                            )
                            if resolved_path:
                                tprint(f"✅ Retrieved artifact '{artifact_name}' from traditional manager: {resolved_path}")
                except Exception as ve:
                    # Fall back to traditional artifact manager on error
                    self.logger.warning(f"Failed to retrieve from versioned store: {ve}, trying traditional manager")
                    data, resolved_path = self.artifact_manager.get_artifact(
                        artifact_name=artifact_name,
                        artifact_type=artifact_type,
                        return_path=True
                    )
                    if resolved_path:
                        tprint(f"✅ Retrieved artifact '{artifact_name}' from traditional manager: {resolved_path}")
            else:
                # Use traditional artifact manager
                data, resolved_path = self.artifact_manager.get_artifact(
                    artifact_name=artifact_name,
                    artifact_type=artifact_type,
                    return_path=True
                )
                if data is not None:
                    if resolved_path:
                        tprint(f"✅ Retrieved artifact '{artifact_name}' from traditional manager: {resolved_path}")
                    else:
                        tprint(f"✅ Retrieved artifact '{artifact_name}' via fallback (no direct file path)")

            if data is not None:
                self.logger.info(f"Retrieved artifact: {artifact_name}")
                return data
            else:
                tprint(f"⚠️ Artifact '{artifact_name}' not found for '{self.step_name}'")
                self.logger.warning(f"Artifact not found: {artifact_name}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to retrieve artifact {artifact_name}: {e}")
            tprint(
                f"❌ Failed to retrieve artifact '{artifact_name}' for '{self.step_name}': {e}"
            )
            raise
    
    def _apply_light_mode_filter(self, data: Any, config: Dict[str, Any], timeframe: str = "15m") -> Any:
        """
        Apply light mode filtering to data if execution mode is 'light'.
        
        In light mode, limits data to the last 20 days to speed up processing.
        
        Args:
            data: Data to filter (should have a tail() method like pandas DataFrame/Series)
            config: Configuration dict containing 'execution_mode'
            timeframe: Timeframe string (e.g., '15m', '1h', '1d')
            
        Returns:
            Filtered data if light mode, original data otherwise
        """
        try:
            execution_mode = config.get('execution_mode', 'light')
            
            if execution_mode.lower() != 'light':
                tprint(
                    f"💡 Light mode disabled for '{self.step_name}' (mode: {execution_mode})"
                )
                return data
            
            # Calculate samples per day for different timeframes
            samples_per_day_map = {
                '1m': 1440,   # 60 * 24
                '3m': 480,    # 20 * 24
                '5m': 288,    # 12 * 24
                '15m': 96,    # 4 * 24
                '30m': 48,    # 2 * 24
                '1h': 24,     # 1 * 24
                '4h': 6,      # 24 / 4
                '1d': 1
            }
            
            samples_per_day = samples_per_day_map.get(timeframe, 96)  # Default to 15m
            days_limit = 20
            light_limit = days_limit * samples_per_day
            tprint(
                f"💡 Applying light mode filter ({timeframe}, {days_limit} days) for '{self.step_name}'"
            )
            
            # Check if data has length attribute and tail method
            if hasattr(data, '__len__') and hasattr(data, 'tail'):
                data_len = len(data)
                if data_len > light_limit:
                    filtered = data.tail(light_limit).copy()
                    self.logger.info(f"BaseStep light mode filtering: reduced data from {data_len:,} to {len(filtered):,} samples ({days_limit} days of {timeframe} data)")
                    tprint(
                        f"✂️ Light mode reduced dataset from {data_len:,} to {len(filtered):,} rows"
                    )
                    return filtered
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Failed to apply light mode filter: {e}")
            return data

    def _save_ml_scored_data(
        self,
        data: Any,
        predictions: Any,
        model_type: str,
        config: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save ML-scored historical data with standardized naming and metadata.
        
        This creates a unified artifact that combines historical data with ML predictions,
        making it easy for backtesting, optimization, and analysis steps to use.
        
        Args:
            data: Historical price/feature data
            predictions: ML model predictions (can be DataFrame or dict)
            model_type: Type of model ('analyst' or 'tactician')
            config: Configuration dictionary with symbol, exchange, timeframe, direction
            metadata: Additional metadata to include
            
        Returns:
            Path where artifact was saved
        """
        try:
            import pandas as pd
            from datetime import datetime
            
            symbol = config.get('symbol', 'UNKNOWN')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            
            # Combine data and predictions
            if isinstance(data, pd.DataFrame) and isinstance(predictions, (pd.DataFrame, pd.Series)):
                # Ensure indices align
                if not data.index.equals(predictions.index):
                    self.logger.warning("Data and predictions indices don't match, aligning...")
                    predictions = predictions.reindex(data.index)
                
                # Combine into scored dataset
                scored_data = data.copy()
                
                # Add predictions with appropriate prefix
                if isinstance(predictions, pd.Series):
                    scored_data[f'{model_type}_prediction'] = predictions
                elif isinstance(predictions, pd.DataFrame):
                    for col in predictions.columns:
                        scored_data[f'{model_type}_{col}'] = predictions[col]
            else:
                # If not DataFrames, package as dict
                scored_data = {
                    'data': data,
                    'predictions': predictions,
                    'model_type': model_type
                }
            
            # Prepare metadata
            artifact_metadata = {
                'model_type': model_type,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'direction': direction,
                'created_at': datetime.now().isoformat(),
                'data_points': len(data) if hasattr(data, '__len__') else 0,
                **(metadata or {})
            }
            
            # Save with standardized name
            artifact_name = f"ml_scored_historical_data_{model_type}_{direction}"
            
            artifact_path = self._save_artifact(
                data=scored_data,
                artifact_name=artifact_name,
                artifact_type='data',
                compression='auto',
                metadata=artifact_metadata
            )
            
            self.logger.info(f"Saved ML scored data: {artifact_name} -> {artifact_path}")
            tprint(
                f"📊 Saved ML scored data '{artifact_name}' to {artifact_path}"
            )
            return artifact_path
            
        except Exception as e:
            self.logger.error(f"Failed to save ML scored data: {e}")
            raise
    
    @require_datetime_index
    def _safe_concat(self, dataframes: List[Any], axis: int = 1,
                     operation_name: str = "concatenate",
                     validate_alignment: bool = True) -> Any:
        """
        Safely concatenate DataFrames with temporal alignment validation.

        Args:
            dataframes: List of DataFrames to concatenate
            axis: Concatenation axis (0=rows, 1=columns)
            operation_name: Description for logging
            validate_alignment: Whether to validate temporal alignment

        Returns:
            Concatenated DataFrame with validated alignment

        Raises:
            ValueError: If DataFrames are not temporally aligned (when validate_alignment=True)
        """
        import pandas as pd

        if not dataframes:
            raise ValueError("No DataFrames provided for concatenation")

        if len(dataframes) == 1:
            return dataframes[0]

        # Filter out None values
        dataframes = [df for df in dataframes if df is not None]

        if not dataframes:
            raise ValueError("All DataFrames are None")

        # Validate temporal alignment if requested
        if validate_alignment and len(dataframes) >= 2:
            try:
                self.artifact_manager._validate_temporal_alignment(
                    *dataframes,
                    operation=operation_name
                )
            except ValueError as e:
                self.logger.error(f"Temporal alignment validation failed: {e}")
                raise

        # Perform safe concatenation
        result = pd.concat(dataframes, axis=axis)

        self.logger.info(
            f"✅ Safe concatenation of {len(dataframes)} DataFrames: "
            f"axis={axis}, result shape={result.shape}"
        )

        return result

    @require_datetime_index
    def _safe_merge(self, left: Any, right: Any,
                    how: str = 'inner',
                    validate_alignment: bool = True,
                    on: Optional[Union[str, List[str]]] = None) -> Any:
        """
        Safely merge DataFrames with temporal alignment validation.

        Args:
            left: Left DataFrame
            right: Right DataFrame
            how: Merge type ('inner', 'outer', 'left', 'right')
            validate_alignment: Whether to validate temporal alignment
            on: Column(s) to merge on (if None, merges on index)

        Returns:
            Merged DataFrame

        Raises:
            ValueError: If temporal alignment validation fails
        """
        import pandas as pd

        if validate_alignment and on is None:
            # For index-based joins, validate indices are compatible
            if how in ['inner', 'left']:
                if not isinstance(left.index, pd.DatetimeIndex):
                    self.logger.warning(
                        f"Left DataFrame in merge does not have DatetimeIndex. "
                        f"Found: {type(left.index).__name__}"
                    )
            if how in ['inner', 'right']:
                if not isinstance(right.index, pd.DatetimeIndex):
                    self.logger.warning(
                        f"Right DataFrame in merge does not have DatetimeIndex. "
                        f"Found: {type(right.index).__name__}"
                    )

        # Merge on index or specified columns
        if on is None:
            result = left.join(right, how=how, rsuffix='_right')
        else:
            result = pd.merge(left, right, on=on, how=how)

        # Log merge statistics
        if on is None:
            overlap = len(left.index.intersection(right.index))
            self.logger.info(
                f"✅ Safe merge: left={len(left)}, right={len(right)}, "
                f"result={len(result)}, overlap={overlap}, how='{how}'"
            )
        else:
            self.logger.info(
                f"✅ Safe merge on columns {on}: left={len(left)}, right={len(right)}, "
                f"result={len(result)}, how='{how}'"
            )

        return result

    @require_datetime_index
    def _align_to_reference(self, reference: Any, *dataframes: Any) -> List[Any]:
        """
        Align multiple DataFrames to a reference DataFrame's index.

        This is useful when combining artifacts that may have different temporal coverage
        due to operations like dropna() or filtering.

        Args:
            reference: Reference DataFrame with target index
            *dataframes: DataFrames to align to reference

        Returns:
            List of aligned DataFrames (same length as input dataframes)

        Raises:
            ValueError: If reference or any DataFrame lacks DatetimeIndex
        """
        import pandas as pd

        if not isinstance(reference.index, pd.DatetimeIndex):
            raise ValueError(
                f"Reference DataFrame must have DatetimeIndex. "
                f"Found: {type(reference.index).__name__}"
            )

        aligned = []
        for i, df in enumerate(dataframes):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    f"DataFrame {i} must have DatetimeIndex for alignment. "
                    f"Found: {type(df.index).__name__}"
                )

            # Align using reindex
            aligned_df = df.reindex(reference.index)

            # Log alignment statistics
            missing = aligned_df.isna().all(axis=1).sum()
            matched = len(df.index.intersection(reference.index))
            self.logger.info(
                f"Aligned DataFrame {i}: {len(df)} -> {len(aligned_df)} rows, "
                f"{matched} matched, {missing} missing timestamps filled with NaN"
            )

            aligned.append(aligned_df)

        return aligned

    def _add_columns_with_tags(self, columns: Dict[str, Any], operation_name: str,
                               tags: Optional[Dict] = None) -> Any:
        """
        Add columns to versioned artifact store with operation tags.

        Args:
            columns: Dictionary of column_name -> values
            operation_name: Name of the operation (e.g., "final_feature_selection")
            tags: Additional metadata tags

        Returns:
            ArtifactView of updated data

        Example:
            self._add_columns_with_tags(
                columns={"feature1": values1, "feature2": values2},
                operation_name="feature_selection",
                tags={"stage": "pre_training"}
            )
        """
        if not self.use_versioned_artifacts or self.versioned_store is None:
            self.logger.warning("Versioned artifacts not enabled, cannot add columns with tags")
            return None

        return self.versioned_store.add_columns_with_tags(
            columns=columns,
            operation_name=operation_name,
            tags=tags or {}
        )

    def _get_columns_by_operation(self, operation_name: str) -> List[str]:
        """
        Get list of columns added by a specific operation.

        Args:
            operation_name: Name of the operation

        Returns:
            List of column names

        Example:
            feature_cols = self._get_columns_by_operation("final_feature_selection")
        """
        if not self.use_versioned_artifacts or self.versioned_store is None:
            self.logger.warning("Versioned artifacts not enabled")
            return []

        return self.versioned_store.get_columns_by_operation(operation_name)

    def _get_columns_by_tag(self, tag_key: str, tag_value: Any) -> List[str]:
        """
        Get list of columns with specific tag value.

        Args:
            tag_key: Tag key to search
            tag_value: Tag value to match

        Returns:
            List of column names

        Example:
            tech_cols = self._get_columns_by_tag("type", "technical_indicator")
        """
        if not self.use_versioned_artifacts or self.versioned_store is None:
            self.logger.warning("Versioned artifacts not enabled")
            return []

        return self.versioned_store.get_columns_by_tag(tag_key, tag_value)

    def _get_view_by_operation(self, operation_name: str) -> Any:
        """
        Get an ArtifactView containing only columns from a specific operation.

        Args:
            operation_name: Name of the operation

        Returns:
            ArtifactView with operation's columns

        Example:
            view = self._get_view_by_operation("final_feature_selection")
            features_df = view.materialize()
        """
        if not self.use_versioned_artifacts or self.versioned_store is None:
            self.logger.warning("Versioned artifacts not enabled")
            return None

        return self.versioned_store.get_view_by_operation(operation_name)

    def _get_sr_levels(self, symbol: str = None, exchange: str = None,
                      timeframe: str = None, direction: str = None) -> Dict[str, Any]:
        """
        Get SR levels dictionary for use in training scripts.
        
        This method provides easy access to the SR levels dictionary that was saved
        by the SR clustering component, making it available to all training scripts
        in pre_training and models_training directories.
        
        Args:
            symbol: Trading symbol to filter by (optional)
            exchange: Exchange to filter by (optional)
            timeframe: Timeframe to filter by (optional)
            direction: Trading direction to filter by (optional)
            
        Returns:
            Dictionary containing SR levels with scores and metadata
        """
        try:
            # Try to get from artifact manager first
            try:
                sr_levels_dict = self._get_artifact(
                    artifact_name='sr_levels_dictionary',
                    artifact_type='data'
                )
                if sr_levels_dict:
                    self.logger.info(f"Retrieved SR levels from artifacts: {len(sr_levels_dict.get('levels', []))} levels")
                    tprint(
                        f"📈 Loaded SR levels dictionary with {len(sr_levels_dict.get('levels', []))} levels from artifacts"
                    )
                    return sr_levels_dict
            except Exception as e:
                self.logger.debug(f"SR levels not found in artifacts: {e}")
                tprint(
                    f"ℹ️ SR levels artifact not available for '{self.step_name}', trying feature bank"
                )
            
            # Fallback to feature bank
            try:
                from src.feature_generation.core.feature_bank import get_global_feature_bank
                feature_bank = get_global_feature_bank()
                sr_levels_dict = feature_bank.get_sr_levels(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    direction=direction
                )
                if sr_levels_dict and not sr_levels_dict.get('error'):
                    self.logger.info(f"Retrieved SR levels from feature bank: {len(sr_levels_dict.get('levels', []))} levels")
                    tprint(
                        f"🏦 Retrieved SR levels from feature bank ({len(sr_levels_dict.get('levels', []))} levels)"
                    )
                    return sr_levels_dict
            except Exception as e:
                self.logger.debug(f"SR levels not available from feature bank: {e}")
                tprint(
                    f"⚠️ Feature bank SR levels unavailable for '{self.step_name}': {e}"
                )
            
            # Return empty result if not found
            self.logger.warning("SR levels dictionary not found in artifacts or feature bank")
            tprint(
                f"⚠️ SR levels dictionary missing for '{self.step_name}'"
            )
            return {
                'levels': [],
                'summary': {'total_levels': 0, 'total_clusters': 0},
                'error': 'SR levels dictionary not found'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get SR levels: {e}")
            tprint(
                f"❌ Failed to load SR levels for '{self.step_name}': {e}"
            )
            return {
                'levels': [],
                'summary': {'total_levels': 0, 'total_clusters': 0},
                'error': str(e)
            }
    
    
    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the step with error handling and outcome generation.
        
        This is the main entry point called by the launcher.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution result with outcome report path
        """
        start_time = datetime.now()
        tprint(f"🚀 Starting execution of step '{self.step_name}'")
        
        try:
            self.logger.info(f"Starting execution of {self.step_name}")
            
            # Detect execution mode
            self.execution_mode = self._detect_execution_mode(config)
            tprint(
                f"🔧 Running '{self.step_name}' in {self.execution_mode} mode"
            )
            
            # Execute the step (async)
            execution_result = await self.execute(config)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_result['execution_time'] = execution_time
            
            # Log completion
            if execution_result.get('success', False):
                self.logger.info(f"Successfully completed {self.step_name} in {execution_time:.2f}s")
                tprint(
                    f"✅ Step '{self.step_name}' completed successfully in {execution_time:.2f}s"
                )
            else:
                self.logger.error(f"Failed to complete {self.step_name} after {execution_time:.2f}s")
                tprint(
                    f"❌ Step '{self.step_name}' reported failure after {execution_time:.2f}s"
                )
            artifacts = execution_result.get('artifacts')
            if artifacts:
                try:
                    artifact_count = len(artifacts)
                except TypeError:
                    artifact_count = 1
                tprint(
                    f"📦 Step '{self.step_name}' produced {artifact_count} artifact(s)"
                )
            outcome_path = execution_result.get('outcome_report_path')
            if outcome_path:
                tprint(
                    f"📝 Outcome report saved to {outcome_path}"
                )
            
            return execution_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Step {self.step_name} failed: {str(e)}\n{traceback.format_exc()}"
            
            self.logger.error(error_msg)
            tprint(
                f"❌ Step '{self.step_name}' crashed after {execution_time:.2f}s: {e}"
            )
            
            # Create failure result
            failure_result = {
                'success': False,
                'error': error_msg,
                'execution_time': execution_time,
                'artifacts': [],
                'metrics': {}
            }
            
            tprint(
                f"🛑 Returning failure result for '{self.step_name}'"
            )
            return failure_result


class StepRegistry:
    """
    Registry for all autonomous steps.
    
    Used by the launcher to discover and execute steps.
    """
    
    def __init__(self):
        self._steps: Dict[str, type] = {}
    
    def register(self, step_name: str, step_class: type):
        """
        Register a step class.
        
        Args:
            step_name: Unique name for the step
            step_class: Step class that inherits from BaseStep
        """
        if not issubclass(step_class, BaseStep):
            raise ValueError(f"Step class {step_class} must inherit from BaseStep")
        
        self._steps[step_name] = step_class
        logging.getLogger("ares.registry").info(f"Registered step: {step_name}")
    
    def get_step(self, step_name: str) -> type:
        """
        Get a registered step class.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Step class
            
        Raises:
            KeyError: If step is not registered
        """
        if step_name not in self._steps:
            raise KeyError(f"Step '{step_name}' not found in registry. Available steps: {list(self._steps.keys())}")
        
        return self._steps[step_name]
    
    def list_steps(self) -> list:
        """
        List all registered step names.
        
        Returns:
            List of step names
        """
        return list(self._steps.keys())
    
    def is_registered(self, step_name: str) -> bool:
        """
        Check if a step is registered.
        
        Args:
            step_name: Name of the step
            
        Returns:
            True if step is registered
        """
        return step_name in self._steps


# Global step registry instance
step_registry = StepRegistry()
