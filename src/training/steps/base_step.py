"""
Base Step Class for Autonomous Pipeline Steps

This module provides the abstract base class that all pipeline steps must inherit from.
Each step becomes autonomous with standardized artifact management and outcome file generation.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta
import traceback

from src.utils.artifact_manager import ArtifactManager
from src.utils.artifact_router import ArtifactRouter
from src.utils.tprint import tprint
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.temporal_validation import require_datetime_index

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas expected in runtime env
    pd = None  # type: ignore[assignment]
    pandas_available = False
else:
    pandas_available = True

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

    _SUPPORTED_TIMEFRAMES = {
        "1m", "3m", "5m", "15m", "30m", "45m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "2w",
        "1mo", "3mo", "6mo", "1y"
    }
    
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
        self._artifact_router = None
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

    def _infer_timeframe_from_artifact_name(self, artifact_name: str) -> Optional[str]:
        """Infer timeframe token from artifact name if present."""
        parts = artifact_name.split('_')
        for part_original in reversed(parts):
            normalized = part_original.lower()
            if normalized in self._SUPPORTED_TIMEFRAMES:
                return part_original
        return None

    @staticmethod
    def _format_context_string(symbol: str, exchange: str, timeframe: Optional[str],
                               direction: str, model: str) -> str:
        timeframe_display = timeframe if timeframe not in (None, "") else "UNKNOWN"
        return f"{symbol}/{exchange} [{timeframe_display}] {direction}/{model}"

    @property
    def artifact_manager(self):
        """Lazy initialization of artifact manager."""
        if self._artifact_manager is None:
            self._artifact_manager = ArtifactManager(config={})
            # Apply deferred context
            if self._current_context:
                # Only forward keys supported by ArtifactManager.set_context
                allowed_keys = {
                    'symbol', 'exchange', 'timeframe', 'datetime',
                    'information', 'direction', 'model'
                }
                context_for_artifact = {
                    k: v for k, v in self._current_context.items()
                    if k in allowed_keys
                }
                self._artifact_manager.set_context(
                    step_name=self.step_name,
                    **context_for_artifact,
                )
        return self._artifact_manager

    @property
    def artifact_router(self):
        """Lazy initialization of artifact router."""
        if self._artifact_router is None:
            self._artifact_router = ArtifactRouter(
                base_dir="artifacts",
                versioned_store_dir="versioned_artifacts",
                historical_data_dir="historical_data",
                enable_versioned_artifacts=self.use_versioned_artifacts
            )
        return self._artifact_router

    @property
    def versioned_store(self):
        """Lazy initialization of versioned artifact store."""
        if self._versioned_store is None and self.use_versioned_artifacts:
            # Extract context for store path with defaults
            symbol = self._current_context.get('symbol', 'UNKNOWN')
            exchange = self._current_context.get('exchange', 'binance')
            timeframe = self._current_context.get('timeframe', '15m')
            direction = self._current_context.get('direction', 'long')
            model = self._current_context.get('model', 'analyst')

            # Create store path with full context separation
            # Format: {symbol}_{exchange}_{timeframe}_{direction}_{model}
            store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"
            store_path = os.path.join("src/utils/versioned_artifacts", store_name)

            self._versioned_store = VersionedArtifactStore(
                store_path=store_path,
                auto_version=True,
                enable_row_versioning=True
            )

            # Store context in store metadata
            if hasattr(self._versioned_store, '_metadata'):
                self._versioned_store._metadata['context'] = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'model': model
                }
                self._versioned_store._save_metadata()

            context_str = f"{symbol}/{exchange} [{timeframe}] {direction}/{model}"
            tprint(f"📦 Initialized VersionedArtifactStore: {context_str} at {store_path}")
        return self._versioned_store

    @property
    def quality_assessor(self):
        """Lazy initialization of quality assessor."""
        if self._quality_assessor is None:
            # Import here to avoid circular import with market_analysis.__init__
            from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
                create_cluster_quality_assessor,
            )

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
            # Only forward keys supported by ArtifactManager.set_context
            allowed_keys = {
                'symbol', 'exchange', 'timeframe', 'datetime',
                'information', 'direction', 'model'
            }
            context_for_artifact = {
                k: v for k, v in self._current_context.items()
                if k in allowed_keys
            }
            self._artifact_manager.set_context(
                step_name=self.step_name,
                **context_for_artifact,
            )
            tprint(
                f"🔄 Applied new context to ArtifactManager for '{self.step_name}'"
            )

        # If versioned store already initialized, reinitialize with new context
        if self._versioned_store is not None and self.use_versioned_artifacts:
            symbol = self._current_context.get('symbol', 'UNKNOWN')
            exchange = self._current_context.get('exchange', 'binance')
            timeframe = self._current_context.get('timeframe', '15m')
            direction = self._current_context.get('direction', 'long')
            model = self._current_context.get('model', 'analyst')

            # Create store path with full context separation
            store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"
            store_path = os.path.join("src/utils/versioned_artifacts", store_name)

            self._versioned_store = VersionedArtifactStore(
                store_path=store_path,
                auto_version=True,
                enable_row_versioning=True
            )

            # Store context in store metadata
            if hasattr(self._versioned_store, '_metadata'):
                self._versioned_store._metadata['context'] = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'model': model
                }
                self._versioned_store._save_metadata()

            context_str = f"{symbol}/{exchange} [{timeframe}] {direction}/{model}"
            tprint(f"🔄 Reinitialized VersionedArtifactStore with new context: {context_str}")

    # ------------------------------------------------------------------
    # Market data loading utilities
    # ------------------------------------------------------------------
    def _load_market_data_from_artifacts(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        artifact_candidates: Optional[List[Tuple[str, str]]] = None,
        artifact_type: str = "data",
    ) -> Optional[Any]:
        """
        Attempt to load market data using artifact manager/router.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe string (e.g., '1h')
            artifact_candidates: Ordered list of (step_name, artifact_name) pairs
            artifact_type: Type hint for artifact retrieval

        Returns:
            Loaded market data object or None if not found
        """
        artifact_candidates = artifact_candidates or [
            ("klines_downloading_processing", "klines_data"),
            ("data_collection", "market_data"),
            ("data_reading", "ohlcv_data"),
        ]

        original_context = self._current_context.copy()

        try:
            for step_name, artifact_name in artifact_candidates:
                try:
                    self.artifact_manager.set_context(
                        step_name=step_name,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                    )

                    data = self._get_artifact(
                        artifact_name=artifact_name,
                        artifact_type=artifact_type,
                    )

                    if data is not None:
                        tprint(
                            f"✅ Loaded market data from {step_name}/{artifact_name}",
                            color="green",
                        )
                        return data

                except Exception as load_error:  # noqa: PERF203 - debug context
                    self.logger.debug(
                        "Artifact load failed from %s/%s: %s",
                        step_name,
                        artifact_name,
                        load_error,
                    )
        finally:
            # Restore original context
            # Only forward keys supported by ArtifactManager.set_context
            allowed_keys = {
                'symbol', 'exchange', 'timeframe', 'datetime',
                'information', 'direction', 'model'
            }
            context_for_artifact = {
                k: v for k, v in original_context.items()
                if k in allowed_keys
            }
            self.artifact_manager.set_context(
                step_name=self.step_name,
                **context_for_artifact,
            )

        return None

    def _load_market_data_from_historical_storage(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = "historical_data",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> Optional[Any]:
        """
        Load market data directly from historical storage using KlinesParquetManager.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe (VectorBT/klines style, e.g., '1h')
            data_dir: Historical data directory root
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Market data DataFrame or None
        """
        try:
            from src.utils.kline_parquet import KlinesParquetManager, StorageConfig

            start_dt: Optional[datetime]
            end_dt: Optional[datetime]

            if start_date is None:
                start_dt = None
            elif pandas_available and pd is not None:
                start_dt = pd.to_datetime(start_date)
            else:
                start_dt = start_date if isinstance(start_date, datetime) else datetime.fromisoformat(str(start_date))

            if end_date is None:
                end_dt = None
            elif pandas_available and pd is not None:
                end_dt = pd.to_datetime(end_date)
            else:
                end_dt = end_date if isinstance(end_date, datetime) else datetime.fromisoformat(str(end_date))

            klines_manager = KlinesParquetManager(
                config=StorageConfig(base_dir=data_dir)
            )

            # Prefer explicit execution_mode from context, then launcher/ENV,
            # and finally fall back to 'light' for legacy behaviour.
            from os import environ as _os_environ
            execution_mode = str(
                self._current_context.get(
                    'execution_mode',
                    self._current_context.get('mode', _os_environ.get('EXECUTION_MODE', 'light')),
                )
            ).lower()

            requested_start = self._current_context.get('start_date')
            requested_end = self._current_context.get('end_date')

            if start_date is None:
                start_date = requested_start
            if end_date is None:
                end_date = requested_end

            timeframe = self._current_context.get('timeframe', timeframe)

            days_limit: Optional[int] = None
            if start_date is None:
                # Add validation for mode-specific days
                tprint(f"🔧 BASESTEP: Using execution_mode={execution_mode}", "INFO")
                tprint(f"🔧 BASESTEP: blank_mode_days in context: {self._current_context.get('blank_mode_days', 'NOT FOUND')}", "INFO")
                tprint(f"🔧 BASESTEP: light_mode_days in context: {self._current_context.get('light_mode_days', 'NOT FOUND')}", "INFO")
                tprint(f"🔧 BASESTEP: Context keys available: {list(self._current_context.keys())}", "INFO")

                # Use centralized execution mode configuration for defaults
                from src.training.steps.market_analysis.shared_utils.execution_mode_lookback_config import get_execution_mode_config
                execution_config = get_execution_mode_config()

                # Support light/blank/full modes for historical loading; other
                # modes fall back to centralized defaults.
                mode_days_defaults = {
                    'light': self._current_context.get('light_mode_days', execution_config.get_data_loading_days('light')),
                    'blank': self._current_context.get('blank_mode_days', execution_config.get_data_loading_days('blank')),
                    'full': self._current_context.get('full_mode_days', execution_config.get_data_loading_days('full')),
                }
                days_limit = mode_days_defaults.get(execution_mode)

                # Add fallback logic with warnings when context values are missing
                if days_limit is None:
                    # Fallback to centralized configuration
                    days_limit = execution_config.get_data_loading_days(execution_mode)
                    tprint(
                        f"⚠️ Using centralized config {execution_mode}_mode_days={days_limit} (not found in context)",
                        "WARNING",
                    )
                    tprint(
                        f"🔧 BASESTEP: Context keys available: {list(self._current_context.keys())}",
                        "INFO",
                    )

                if days_limit is not None:
                    message_prefix = {
                        'light': "💡 Light",
                        'blank': "⚪ Blank",
                        'full': "📅 Full",
                    }.get(execution_mode, "📅 Mode")
                    tprint(
                        f"{message_prefix} mode pre-filter: requesting last {days_limit} days "
                        f"(anchored at latest available data, timeframe {timeframe})",
                    )
                    tprint(
                        f"🔧 BASESTEP: Using execution_mode={execution_mode} with days_limit={days_limit}",
                        "INFO",
                    )

            # If no explicit start_date and we have a mode-specific days_limit,
            # load the last N days anchored at the most recent available data
            # using KlinesParquetManager's last_n_days helper. Otherwise, honour
            # explicit start/end bounds.
            if start_date is None and days_limit is not None:
                market_data = klines_manager.load_klines(
                    symbol=symbol,
                    exchange=exchange,
                    interval=timeframe,
                    last_n_days=days_limit,
                )
            else:
                market_data = klines_manager.load_klines(
                    symbol=symbol,
                    exchange=exchange,
                    interval=timeframe,
                    start_time=start_dt,
                    end_time=end_dt,
                )

            if market_data is not None and not getattr(market_data, "empty", False):
                tprint(
                    f"✅ Loaded {len(market_data)} rows from historical storage",
                    color="green",
                )
                return market_data

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.debug(
                "Historical storage load failed for %s %s %s: %s",
                symbol,
                exchange,
                timeframe,
                exc,
            )

        return None

    def load_market_data_or_fail(
        self,
        config: Dict[str, Any],
        pipeline_state: Optional[Dict[str, Any]] = None,
        *,
        allow_config_override: bool = True,
        light_mode_filter: bool = True,
        artifact_candidates: Optional[List[Tuple[str, str]]] = None,
        skip_artifacts: bool = False,
    ) -> Tuple[Any, str]:
        """
        Load market data using a multi-stage strategy.

        Priority order:
            1. Config override (if allow_config_override)
            2. Pipeline state cache
            3. Artifact manager/router
            4. Historical storage directory

        Args:
            config: Launcher configuration dictionary
            pipeline_state: Mutable pipeline state dictionary
            allow_config_override: Enable config['market_data'] usage
            light_mode_filter: Apply BaseStep light mode filtering
            artifact_candidates: Optional override for artifact search order

        Returns:
            Tuple of (market_data, source_description)

        Raises:
            ValueError: If no market data can be located
        """
        pipeline_state = pipeline_state or {}

        symbol_value = config.get("symbol")
        if not isinstance(symbol_value, str) or not symbol_value:
            raise ValueError("Configuration must include a symbol string")
        symbol = symbol_value

        exchange_value = config.get("exchange")
        if not isinstance(exchange_value, str) or not exchange_value:
            raise ValueError("Configuration must include an exchange string")
        exchange = exchange_value

        timeframe_value = config.get("timeframe")
        if not isinstance(timeframe_value, str) or not timeframe_value:
            timeframe_value = config.get("regime_timeframe")
        if not isinstance(timeframe_value, str) or not timeframe_value:
            raise ValueError("Configuration must include a timeframe string")
        timeframe = timeframe_value

        # Propagate launcher-provided lookback_days into context-specific mode keys
        # so that historical loading honours centralized execution-mode windows.
        try:
            exec_mode_cfg = str(config.get("execution_mode", "")).lower()
            if exec_mode_cfg in {"light", "blank", "full"}:
                lookback_days_val = config.get("lookback_days")
                if isinstance(lookback_days_val, (int, float)) and lookback_days_val > 0:
                    mode_key = f"{exec_mode_cfg}_mode_days"
                    # Respect any explicit context override if already present
                    if mode_key not in self._current_context:
                        self._current_context[mode_key] = int(lookback_days_val)
        except Exception:
            # Non-fatal: fall back to centralized execution_mode configuration
            pass

        # 1. Config override
        if allow_config_override and config.get("market_data") is not None:
            data = config["market_data"]
            tprint("✅ Using market data provided in config", color="green")
            return data, "config.market_data"

        # 2. Pipeline state cache
        pipeline_sources = [
            ("pipeline_state.market_data", pipeline_state.get("market_data")),
            ("pipeline_state.validated_data", pipeline_state.get("validated_data")),
            ("pipeline_state.dataframe", pipeline_state.get("dataframe")),
            ("pipeline_state.raw_market_data", pipeline_state.get("raw_market_data")),
        ]

        for source_name, candidate in pipeline_sources:
            if candidate is not None:
                tprint(f"✅ Using cached market data from {source_name}", color="green")
                return candidate, source_name

        # 3. Artifact manager/router (unless explicitly skipped)
        data = None
        source_name = "historical_data"

        if not skip_artifacts:
            data = self._load_market_data_from_artifacts(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                artifact_candidates=artifact_candidates,
            )

        if data is not None:
            source_name = "artifacts"
        else:
            # 4. Historical storage directory
            data_dir = config.get("data_dir", "historical_data")
            data = self._load_market_data_from_historical_storage(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                start_date=config.get("start_date"),
                end_date=config.get("end_date"),
            )
            source_name = "historical_data"

        if data is None:
            raise ValueError(
                "No market data available. Ensure data collection steps have populated "
                "artifacts or provide market_data in the launcher config."
            )

        # Optional light mode filter
        if light_mode_filter and pandas_available and isinstance(data, pd.DataFrame):
            data = self._apply_light_mode_filter(data, config, timeframe)

        return data, source_name

    def ensure_market_data_in_pipeline_state(
        self,
        config: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        *,
        allow_config_override: bool = True,
        artifact_candidates: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[Any, str]:
        """
        Load market data and persist it into pipeline_state for downstream steps.

        Args:
            config: Step configuration dictionary
            pipeline_state: Pipeline state dictionary (mutated in-place)
            allow_config_override: Whether to honour config["market_data"]
            artifact_candidates: Optional artifact loading order override

        Returns:
            Tuple of (market_data, source_description)
        """
        market_data, source = self.load_market_data_or_fail(
            config,
            pipeline_state,
            allow_config_override=allow_config_override,
            artifact_candidates=artifact_candidates,
        )

        # Persist in pipeline state
        pipeline_state["market_data"] = market_data
        if pandas_available and isinstance(market_data, pd.DataFrame):
            pipeline_state.setdefault("validated_data", market_data)
            pipeline_state.setdefault("dataframe", market_data)

        return market_data, source

    def _detect_execution_mode(self, config: Dict[str, Any]) -> str:
        """
        Detect execution mode based on launcher arguments and step context.

        This method can be overridden by subclasses for more specific mode detection.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            'analyst' or 'tactician'
        """
        requested_cli_mode = str(config.get('execution_mode', '')).lower()
        if requested_cli_mode not in {'full', 'light', 'blank', 'small_dataset'}:
            requested_cli_mode = 'unspecified'
        
        is_tactician_training_step = (
            'tactician_base_training' in self.step_name or
            'tactician_ensemble_training' in self.step_name or
            'tactician' in self.step_name.lower()
        )
        
        tactician_execution_context = config.get('execution_context', '').lower()
        is_tactician_context = 'tactician' in tactician_execution_context
        
        explicit_mode = config.get('interaction_generation_mode', '').lower()
        tactician_mode_config = config.get('tactician_mode', False)
        
        if (is_tactician_training_step or is_tactician_context or 
            explicit_mode == 'tactician' or tactician_mode_config):
            mode = 'tactician'
        else:
            mode = 'analyst'
        
        self.logger.info(
            "Execution mode resolved for %s: requested_cli=%s → persona=%s",
            self.step_name,
            requested_cli_mode,
            mode
        )
        tprint(
            f"🎯 Execution mode for '{self.step_name}': requested_cli={requested_cli_mode} → persona={mode}"
        )
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
                      tags: Optional[Dict] = None,
                      data_category: Optional[str] = None) -> str:
        """
        Save an artifact using intelligent format routing.

        Routes to appropriate storage based on data type and content:
        - JSON: configs, metadata, dictionaries
        - Pickle: ML models, complex objects
        - Parquet (via kline_parquet.py): historical OHLCV data
        - HDF5 (via versioned_artifacts/): feature DataFrames, training data

        Args:
            data: Data to save (DataFrame, dict, model, etc.)
            artifact_name: Name for the artifact
            artifact_type: Type of artifact ("data", "model", "metadata", etc.)
            compression: Compression method ("auto", "gzip", "lz4", "none")
            metadata: Additional metadata to store with artifact
            operation_name: Name of operation for versioned artifacts tagging
            tags: Additional tags for versioned artifacts
            data_category: Explicit category hint (config, model, historical, features, predictions)

        Returns:
            Path where artifact was saved
        """
        try:
            # Build context dict and string for logging
            symbol = self._current_context.get('symbol', 'UNKNOWN')
            exchange = self._current_context.get('exchange', 'binance')
            timeframe = self._current_context.get('timeframe', '15m')
            direction = self._current_context.get('direction', 'long')
            model = self._current_context.get('model', 'analyst')

            # Build context dict for router
            context_dict = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'direction': direction,
                'model': model,
                'step_name': self.step_name
            }

            artifact_timeframe = self._infer_timeframe_from_artifact_name(artifact_name)
            if artifact_timeframe and str(timeframe).lower() != artifact_timeframe.lower():
                tprint(
                    f"🐛 DEBUG: Overriding context timeframe {timeframe} → {artifact_timeframe} "
                    f"based on artifact name '{artifact_name}'",
                    "INFO"
                )
                timeframe = artifact_timeframe
                context_dict['timeframe'] = timeframe

            context_str = self._format_context_string(symbol, exchange, timeframe, direction, model)

            tprint(
                f"💾 Saving artifact '{artifact_name}' (type: {artifact_type}) | {context_str}"
            )
            tprint(f"🐛 DEBUG: BaseStep._save_artifact called with artifact_name={artifact_name}, data_category={data_category}", "INFO")
            tprint(f"🐛 DEBUG: Data type: {type(data)}, shape: {getattr(data, 'shape', 'N/A')}", "INFO")

            # Data description if it's a DataFrame
            if pandas_available and isinstance(data, pd.DataFrame):
                tprint(f"📊 Saving Data Description:")
                tprint(f"   • Rows: {len(data)}")
                tprint(f"   • Columns: {len(data.columns)}")
                tprint(f"   • First 10 columns: {list(data.columns[:10])}")
                if len(data.columns) > 10:
                    tprint(f"   • ... and {len(data.columns) - 10} more columns")
                tprint(f"   • First 5 rows:")
                for idx, row_idx in enumerate(data.index[:5]):
                    row_preview = {col: data.loc[row_idx, col] for col in data.columns[:3]}
                    tprint(f"     [{idx}] {row_idx}: {row_preview}...")
                if len(data) > 5:
                    tprint(f"   • ... and {len(data) - 5} more rows")

            # Auto-detect data category from artifact_name if not provided
            if data_category is None:
                if any(kw in artifact_name.lower() for kw in ['config', 'metadata', 'params', 'settings']):
                    data_category = 'config'
                elif any(kw in artifact_name.lower() for kw in ['model', 'estimator', 'classifier', 'regressor']):
                    data_category = 'model'
                elif any(kw in artifact_name.lower() for kw in ['historical', 'klines', 'ohlcv']):
                    data_category = 'historical'
                elif any(kw in artifact_name.lower() for kw in ['feature', 'training']):
                    data_category = 'features'
                elif any(kw in artifact_name.lower() for kw in ['prediction', 'score']):
                    data_category = 'predictions'

            tprint(f"🐛 DEBUG: Final data_category: {data_category}, use_versioned_artifacts: {self.use_versioned_artifacts}", "INFO")

            metadata_for_save = metadata.copy() if isinstance(metadata, dict) else metadata
            if isinstance(metadata_for_save, dict) and timeframe and 'timeframe' not in metadata_for_save:
                metadata_for_save['timeframe'] = timeframe

            # Use ArtifactRouter for intelligent routing
            tprint("🐛 DEBUG: Calling artifact_router.save()...", "INFO")
            artifact_path = self.artifact_router.save(
                data=data,
                artifact_name=artifact_name,
                artifact_type=artifact_type,
                data_category=data_category,
                context=context_dict,
                metadata=metadata_for_save
            )
            tprint(f"🐛 DEBUG: artifact_router.save() returned: {artifact_path}", "INFO")

            self.logger.info(f"Saved artifact: {artifact_name} -> {artifact_path}")
            tprint(f"✅ Saved artifact '{artifact_name}' | {context_str}")
            return artifact_path

        except Exception as e:
            self.logger.error(f"Failed to save artifact {artifact_name}: {e}")
            tprint(
                f"❌ Failed to save artifact '{artifact_name}' for '{self.step_name}': {e}"
            )
            raise
    
    def _get_artifact(self, artifact_name: str,
                     artifact_type: str = "data",
                     operation_name: Optional[str] = None,
                     data_category: Optional[str] = None) -> Any:
        """
        Retrieve an artifact using intelligent format routing.

        Routes to appropriate storage based on data type and content:
        - JSON: configs, metadata, dictionaries
        - Pickle: ML models, complex objects
        - Parquet (via kline_parquet.py): historical OHLCV data
        - HDF5 (via versioned_artifacts/): feature DataFrames, training data

        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve
            operation_name: If provided, retrieve view by operation name
            data_category: Explicit category hint (config, model, historical, features, predictions)

        Returns:
            Retrieved data
        """
        try:
            # Build context dict and string for logging
            symbol = self._current_context.get('symbol', 'UNKNOWN')
            exchange = self._current_context.get('exchange', 'binance')
            timeframe = self._current_context.get('timeframe', '15m')
            direction = self._current_context.get('direction', 'long')
            model = self._current_context.get('model', 'analyst')
            context_str = f"{symbol}/{exchange} [{timeframe}] {direction}/{model}"

            # Build context dict for router
            context_dict = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'direction': direction,
                'model': model,
                'step_name': self.step_name
            }

            artifact_timeframe = self._infer_timeframe_from_artifact_name(artifact_name)
            if artifact_timeframe and str(timeframe).lower() != artifact_timeframe.lower():
                tprint(
                    f"🐛 DEBUG: Overriding context timeframe {timeframe} → {artifact_timeframe} "
                    f"based on artifact name '{artifact_name}'",
                    "INFO"
                )
                timeframe = artifact_timeframe
                context_dict['timeframe'] = timeframe

            context_str = self._format_context_string(symbol, exchange, timeframe, direction, model)

            tprint(
                f"📂 Retrieving artifact '{artifact_name}' (type: {artifact_type}) | {context_str}"
            )

            # Auto-detect data category from artifact_name if not provided
            if data_category is None:
                if any(kw in artifact_name.lower() for kw in ['config', 'metadata', 'params', 'settings']):
                    data_category = 'config'
                elif any(kw in artifact_name.lower() for kw in ['model', 'estimator', 'classifier', 'regressor']):
                    data_category = 'model'
                elif any(kw in artifact_name.lower() for kw in ['historical', 'klines', 'ohlcv']):
                    data_category = 'historical'
                elif any(kw in artifact_name.lower() for kw in ['feature', 'training']):
                    data_category = 'features'
                elif any(kw in artifact_name.lower() for kw in ['prediction', 'score']):
                    data_category = 'predictions'

            # Use ArtifactRouter for intelligent routing
            try:
                data = self.artifact_router.load(
                    artifact_name=artifact_name,
                    artifact_type=artifact_type,
                    data_category=data_category,
                    context=context_dict
                )

                if data is not None:
                    self.logger.info(f"Retrieved artifact: {artifact_name}")

                    # Data description if it's a DataFrame
                    if pandas_available and isinstance(data, pd.DataFrame):
                        tprint(f"📊 Retrieved Data Description:")
                        tprint(f"   • Rows: {len(data)}")
                        tprint(f"   • Columns: {len(data.columns)}")
                        tprint(f"   • First 10 columns: {list(data.columns[:10])}")
                        if len(data.columns) > 10:
                            tprint(f"   • ... and {len(data.columns) - 10} more columns")
                        tprint(f"   • First 5 rows:")
                        for idx, row_idx in enumerate(data.index[:5]):
                            row_preview = {col: data.loc[row_idx, col] for col in data.columns[:3]}
                            tprint(f"     [{idx}] {row_idx}: {row_preview}...")
                        if len(data) > 5:
                            tprint(f"   • ... and {len(data) - 5} more rows")

                    tprint(f"✅ Retrieved artifact '{artifact_name}' | {context_str}")
                    return data
                else:
                    tprint(f"⚠️ Artifact '{artifact_name}' not found | {context_str}")
                    self.logger.warning(f"Artifact not found: {artifact_name}")
                    return None

            except FileNotFoundError:
                # Fallback to traditional artifact manager if router fails
                self.logger.warning(f"Artifact '{artifact_name}' not found via router, trying traditional manager")
                tprint(f"⚠️ Trying fallback to traditional artifact manager | {context_str}")

                original_artifact_context = self._current_context.copy()
                adjusted_context = original_artifact_context.copy()
                adjusted_context['timeframe'] = timeframe

                # Only forward keys supported by ArtifactManager.set_context to
                # avoid unexpected keyword arguments such as execution_mode.
                allowed_keys = {
                    'symbol', 'exchange', 'timeframe', 'datetime',
                    'information', 'direction', 'model'
                }
                try:
                    adjusted_for_artifact = {
                        k: v for k, v in adjusted_context.items()
                        if k in allowed_keys
                    }
                    self.artifact_manager.set_context(
                        step_name=self.step_name,
                        **adjusted_for_artifact,
                    )
                    data, resolved_path = self.artifact_manager.get_artifact(
                        artifact_name=artifact_name,
                        artifact_type=artifact_type,
                        return_path=True
                    )
                finally:
                    original_for_artifact = {
                        k: v for k, v in original_artifact_context.items()
                        if k in allowed_keys
                    }
                    self.artifact_manager.set_context(
                        step_name=self.step_name,
                        **original_for_artifact,
                    )

                if data is not None and resolved_path:
                    # Data description if it's a DataFrame
                    if pandas_available and isinstance(data, pd.DataFrame):
                        tprint(f"📊 Retrieved Data Description (from fallback):")
                        tprint(f"   • Rows: {len(data)}")
                        tprint(f"   • Columns: {len(data.columns)}")
                        tprint(f"   • First 10 columns: {list(data.columns[:10])}")
                        if len(data.columns) > 10:
                            tprint(f"   • ... and {len(data.columns) - 10} more columns")
                        tprint(f"   • First 5 rows:")
                        for idx, row_idx in enumerate(data.index[:5]):
                            row_preview = {col: data.loc[row_idx, col] for col in data.columns[:3]}
                            tprint(f"     [{idx}] {row_idx}: {row_preview}...")
                        if len(data) > 5:
                            tprint(f"   • ... and {len(data) - 5} more rows")

                    tprint(f"✅ Retrieved artifact '{artifact_name}' from fallback | {context_str}")
                    return data
                else:
                    tprint(f"⚠️ Artifact '{artifact_name}' not found in any storage | {context_str}")
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
            
            if str(execution_mode).lower() != 'light':
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

            # Prefer launcher-provided lookback_days when available; otherwise
            # fall back to the centralized execution_mode lookback configuration.
            days_limit = config.get('lookback_days')
            if not isinstance(days_limit, (int, float)) or days_limit <= 0:
                from src.training.steps.market_analysis.shared_utils.execution_mode_lookback_config import get_execution_mode_config
                execution_config = get_execution_mode_config()
                days_limit = execution_config.get_data_loading_days('light')

            days_limit = int(days_limit)

            samples_per_day = samples_per_day_map.get(timeframe, 96)  # Default to 15m
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
            tprint(f"🔍 Validating temporal alignment for {len(dataframes)} DataFrames in '{operation_name}'")
            try:
                self.artifact_manager._validate_temporal_alignment(
                    *dataframes,
                    operation=operation_name
                )
                tprint(f"✅ Temporal alignment validated successfully")
            except ValueError as e:
                self.logger.error(f"Temporal alignment validation failed: {e}")
                tprint(f"❌ Temporal alignment validation failed for '{operation_name}': {e}")
                raise

        # Perform safe concatenation
        tprint(f"🔗 Concatenating {len(dataframes)} DataFrames (axis={axis}) for '{operation_name}'")
        result = pd.concat(dataframes, axis=axis)

        self.logger.info(
            f"✅ Safe concatenation of {len(dataframes)} DataFrames: "
            f"axis={axis}, result shape={result.shape}"
        )
        tprint(f"✅ Successfully concatenated {len(dataframes)} DataFrames → shape {result.shape}")

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

        tprint(f"🔀 Merging DataFrames (how='{how}', on={on}, left={len(left)} rows, right={len(right)} rows)")

        if validate_alignment and on is None:
            # For index-based joins, validate indices are compatible
            if how in ['inner', 'left']:
                if not isinstance(left.index, pd.DatetimeIndex):
                    self.logger.warning(
                        f"Left DataFrame in merge does not have DatetimeIndex. "
                        f"Found: {type(left.index).__name__}"
                    )
                    tprint(f"⚠️ Left DataFrame lacks DatetimeIndex: {type(left.index).__name__}")
            if how in ['inner', 'right']:
                if not isinstance(right.index, pd.DatetimeIndex):
                    self.logger.warning(
                        f"Right DataFrame in merge does not have DatetimeIndex. "
                        f"Found: {type(right.index).__name__}"
                    )
                    tprint(f"⚠️ Right DataFrame lacks DatetimeIndex: {type(right.index).__name__}")

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
            tprint(f"✅ Merge completed: {len(result)} rows (overlap: {overlap})")
        else:
            self.logger.info(
                f"✅ Safe merge on columns {on}: left={len(left)}, right={len(right)}, "
                f"result={len(result)}, how='{how}'"
            )
            tprint(f"✅ Merge completed on columns {on}: {len(result)} rows")

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

        tprint(f"🎯 Aligning {len(dataframes)} DataFrames to reference (ref: {len(reference)} rows)")

        aligned = []
        for i, df in enumerate(dataframes):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    f"DataFrame {i} must have DatetimeIndex for alignment. "
                    f"Found: {type(df.index).__name__}"
                )

            # Align using reindex
            tprint(f"   Aligning DataFrame {i} ({len(df)} rows)...")
            aligned_df = df.reindex(reference.index)

            # Log alignment statistics
            missing = aligned_df.isna().all(axis=1).sum()
            matched = len(df.index.intersection(reference.index))
            self.logger.info(
                f"Aligned DataFrame {i}: {len(df)} -> {len(aligned_df)} rows, "
                f"{matched} matched, {missing} missing timestamps filled with NaN"
            )
            tprint(f"   ✅ DataFrame {i}: {matched} matched, {missing} missing (filled with NaN)")

            aligned.append(aligned_df)

        tprint(f"✅ Successfully aligned {len(dataframes)} DataFrames to reference")
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
            tprint(f"⚠️ Versioned artifacts not enabled, cannot add columns for operation '{operation_name}'")
            return None

        tprint(f"🏷️  Adding {len(columns)} columns with operation tag '{operation_name}'")
        result = self.versioned_store.add_columns_with_tags(
            columns=columns,
            operation_name=operation_name,
            tags=tags or {}
        )
        tprint(f"✅ Successfully added {len(columns)} columns for operation '{operation_name}'")
        return result

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
            tprint(f"⚠️ Versioned artifacts not enabled, cannot retrieve columns for operation '{operation_name}'")
            return []

        tprint(f"🔍 Retrieving columns for operation '{operation_name}'")
        columns = self.versioned_store.get_columns_by_operation(operation_name)
        tprint(f"✅ Found {len(columns)} columns for operation '{operation_name}'")
        return columns

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
            tprint(f"⚠️ Versioned artifacts not enabled, cannot retrieve columns by tag")
            return []

        tprint(f"🔍 Retrieving columns with tag {tag_key}={tag_value}")
        columns = self.versioned_store.get_columns_by_tag(tag_key, tag_value)
        tprint(f"✅ Found {len(columns)} columns with tag {tag_key}={tag_value}")
        return columns

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
            tprint(f"⚠️ Versioned artifacts not enabled, cannot retrieve view for operation '{operation_name}'")
            return None

        tprint(f"👁️  Retrieving view for operation '{operation_name}'")
        view = self.versioned_store.get_view_by_operation(operation_name)
        tprint(f"✅ Retrieved view for operation '{operation_name}'")
        return view

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
            
            # Detect execution persona (analyst vs tactician) for logging and
            # optimisation purposes. Do not store this under 'execution_mode'
            # in the shared context, as that key is reserved for data-loading
            # modes ('full', 'light', 'blank') controlled by the launcher.
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
