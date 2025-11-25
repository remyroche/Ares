"""
Smart readers with monthly consolidation fallback.

Hybrid Loading Strategy:
- Fast for History: Read monthly_consolidated.parquet (1 file per month)
- Fresh for Recent Data: Read daily files
"""
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
import logging
from collections import defaultdict

from .constants import (
    LAYER_PATHS,
    PREDICTIONS_DIR,
    METADATA_COLUMNS,
)


logger = logging.getLogger(__name__)


class HivePartitionedReader:
    """
    Read from Hive-partitioned predictions.

    Automatically prefers monthly_consolidated.parquet when available
    (fast read of 2880 rows vs. 30 files of 96 rows each).

    Features:
    - Smart monthly consolidation fallback
    - Date range filtering with partition pruning
    - Duplicate removal (keeps last)
    - Model version discovery
    - Efficient monthly aggregation
    """

    def __init__(
        self,
        layer_name: str,
        base_path: Optional[Path] = None
    ):
        """
        Initialize the Hive-partitioned reader.

        Args:
            layer_name: Layer name (specialists, base_models, meta_layer, etc.)
            base_path: Optional base path override (for testing)
        """
        if layer_name not in LAYER_PATHS and base_path is None:
            raise ValueError(
                f"Unknown layer: {layer_name}. "
                f"Must be one of: {list(LAYER_PATHS.keys())}"
            )

        self.layer_name = layer_name

        if base_path is not None:
            self.base_path = base_path / PREDICTIONS_DIR
        else:
            self.base_path = LAYER_PATHS[layer_name] / PREDICTIONS_DIR

        logger.debug(
            f"Initialized HivePartitionedReader for {layer_name} "
            f"at {self.base_path}"
        )

    def load_recent_predictions(
        self,
        days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load predictions with smart monthly consolidation fallback.

        Performance:
        - If monthly_consolidated.parquet exists: 1 file read per month
        - Otherwise: Fall back to daily files

        Args:
            days: Load last N days (alternative to start_date/end_date)
            start_date: Start date (ISO format: "2025-01-01")
            end_date: End date (ISO format: "2025-12-31")
            model_version: Model version to load (default: latest)

        Returns:
            DataFrame with predictions (sorted by index, duplicates removed)

        Raises:
            ValueError: If no data found or invalid parameters
        """
        # Discover model version
        if model_version is None:
            model_version = self._get_latest_model_version()
            logger.info(f"Using latest model version: {model_version}")

        # Calculate date range
        start_dt, end_dt = self._calculate_date_range(
            days=days,
            start_date=start_date,
            end_date=end_date
        )

        logger.info(
            f"Loading predictions for {self.layer_name}/{model_version} "
            f"from {start_dt.date()} to {end_dt.date()}"
        )

        # Find all relevant month partitions
        months = self._discover_months(model_version, start_dt, end_dt)

        if not months:
            raise ValueError(
                f"No data found for {self.layer_name}/{model_version} "
                f"in date range {start_dt.date()} to {end_dt.date()}"
            )

        # Load data with smart fallback
        dfs = []
        for year, month in months:
            month_df = self._load_month_data(
                model_version, year, month, start_dt, end_dt
            )
            if month_df is not None and not month_df.empty:
                dfs.append(month_df)

        if not dfs:
            raise ValueError(
                f"No data found for {self.layer_name}/{model_version}"
            )

        # Combine and filter
        combined = pd.concat(dfs, axis=0)
        combined = combined.sort_index()

        # Apply date filters (additional filtering beyond partition pruning)
        if isinstance(combined.index, pd.DatetimeIndex):
            if start_dt:
                combined = combined[combined.index >= start_dt]
            if end_dt:
                combined = combined[combined.index <= end_dt]

        # Remove duplicates (keep last)
        combined = combined[~combined.index.duplicated(keep='last')]

        # Drop metadata columns
        meta_cols = [c for c in combined.columns if c in METADATA_COLUMNS]
        combined = combined.drop(columns=meta_cols, errors='ignore')

        logger.info(
            f"📊 Loaded {len(combined)} predictions from {self.layer_name}"
        )

        return combined

    def _load_month_data(
        self,
        model_version: str,
        year: int,
        month: int,
        start_dt: datetime,
        end_dt: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Load data for a single month with smart fallback.

        Strategy:
        1. Try monthly_consolidated.parquet (FAST PATH)
        2. Fall back to daily files (SLOW PATH)
        """
        month_path = (
            self.base_path /
            f"model_version={model_version}" /
            f"year={year}" /
            f"month={month:02d}"
        )

        if not month_path.exists():
            return None

        # FAST PATH: Try consolidated file
        consolidated_path = month_path / "monthly_consolidated.parquet"

        if consolidated_path.exists():
            logger.debug(f"📦 Reading consolidated: {year}-{month:02d}")
            df = pd.read_parquet(consolidated_path)
            return df

        # SLOW PATH: Read daily files
        logger.debug(f"📁 Reading daily files for {year}-{month:02d}")
        daily_dfs = self._read_daily_files(month_path, start_dt, end_dt)

        if not daily_dfs:
            return None

        logger.debug(
            f"📁 Read {len(daily_dfs)} daily files for {year}-{month:02d}"
        )
        return pd.concat(daily_dfs, axis=0)

    def _read_daily_files(
        self,
        month_path: Path,
        start_dt: datetime,
        end_dt: datetime
    ) -> List[pd.DataFrame]:
        """
        Read all daily files in a month.

        Applies partition pruning by day.
        """
        dfs = []

        for day_dir in sorted(month_path.glob("day=*")):
            # Extract day from directory name
            day = int(day_dir.name.split('=')[1])

            # Partition pruning: skip days outside range
            year = int(month_path.parent.name.split('=')[1])
            month = int(month_path.name.split('=')[1])
            day_date = datetime(year, month, day)

            if day_date < start_dt.date() or day_date > end_dt.date():
                continue

            data_file = day_dir / "data.parquet"

            if data_file.exists():
                df = pd.read_parquet(data_file)
                dfs.append(df)

        return dfs

    def _get_latest_model_version(self) -> str:
        """Get latest model version from filesystem."""
        if not self.base_path.exists():
            raise ValueError(
                f"Predictions path does not exist: {self.base_path}"
            )

        version_dirs = [
            d.name.split('=')[1]
            for d in self.base_path.glob("model_version=*")
            if d.is_dir()
        ]

        if not version_dirs:
            raise ValueError(
                f"No model versions found in {self.base_path}"
            )

        # Sort versions (assumes semantic versioning: v1.2.3)
        # Remove 'v' prefix for sorting if present
        def version_key(v: str) -> tuple:
            v_clean = v.lstrip('v')
            try:
                return tuple(map(int, v_clean.split('.')))
            except ValueError:
                # Fall back to string sorting if not semantic versioning
                return (v,)

        return sorted(version_dirs, key=version_key)[-1]

    def _discover_months(
        self,
        model_version: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> List[Tuple[int, int]]:
        """
        Discover which year/month partitions exist.

        Returns:
            List of (year, month) tuples
        """
        version_path = self.base_path / f"model_version={model_version}"

        if not version_path.exists():
            return []

        months = []
        for year_dir in version_path.glob("year=*"):
            year = int(year_dir.name.split('=')[1])

            for month_dir in year_dir.glob("month=*"):
                month = int(month_dir.name.split('=')[1])

                # Partition pruning: skip months outside range
                month_start = datetime(year, month, 1)
                # Last day of month
                if month == 12:
                    month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    month_end = datetime(year, month + 1, 1) - timedelta(days=1)

                # Check if month overlaps with date range
                if month_end >= start_dt and month_start <= end_dt:
                    months.append((year, month))

        return sorted(months)

    def _calculate_date_range(
        self,
        days: Optional[int],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Tuple[datetime, datetime]:
        """
        Calculate date range from parameters.

        Args:
            days: Load last N days
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            (start_datetime, end_datetime)
        """
        now = datetime.now()

        # Option 1: Use days parameter
        if days is not None:
            if start_date or end_date:
                raise ValueError(
                    "Cannot specify both 'days' and 'start_date'/'end_date'"
                )
            start_dt = now - timedelta(days=days)
            end_dt = now
            return start_dt, end_dt

        # Option 2: Use start_date/end_date
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            # Default: last 56 days (8 weeks)
            start_dt = now - timedelta(days=56)

        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = now

        if start_dt > end_dt:
            raise ValueError(
                f"start_date ({start_dt}) must be <= end_date ({end_dt})"
            )

        return start_dt, end_dt

    def get_available_model_versions(self) -> List[str]:
        """Get all available model versions."""
        if not self.base_path.exists():
            return []

        versions = [
            d.name.split('=')[1]
            for d in self.base_path.glob("model_version=*")
            if d.is_dir()
        ]

        return sorted(versions)

    def get_date_range(self, model_version: Optional[str] = None) -> Tuple[datetime, datetime]:
        """
        Get the min and max dates available for a model version.

        Args:
            model_version: Model version (default: latest)

        Returns:
            (min_date, max_date)
        """
        if model_version is None:
            model_version = self._get_latest_model_version()

        months = self._discover_months(
            model_version,
            start_dt=datetime(1970, 1, 1),
            end_dt=datetime(2100, 12, 31)
        )

        if not months:
            raise ValueError(f"No data found for {model_version}")

        # Min date: first day of first month
        min_year, min_month = months[0]
        min_date = datetime(min_year, min_month, 1)

        # Max date: last day of last month
        max_year, max_month = months[-1]
        if max_month == 12:
            max_date = datetime(max_year + 1, 1, 1) - timedelta(days=1)
        else:
            max_date = datetime(max_year, max_month + 1, 1) - timedelta(days=1)

        return min_date, max_date


# ============================================================================
# Polars Implementation (Ultra-fast with lazy evaluation)
# ============================================================================

try:
    import polars as pl

    class PolarsHiveReader:
        """
        Even faster reading with Polars lazy evaluation.

        Polars can push filters down to Parquet file level.

        Benefits:
        - Lazy evaluation (only reads needed data)
        - Automatic Hive partition discovery
        - Predicate pushdown (filter at file scan level)
        - Parallel I/O
        - Zero-copy data structures

        WARNING: Ensure daily files are deleted after consolidation
        to avoid reading duplicates!
        """

        def __init__(
            self,
            layer_name: str,
            base_path: Optional[Path] = None
        ):
            """
            Initialize the Polars Hive reader.

            Args:
                layer_name: Layer name (specialists, base_models, etc.)
                base_path: Optional base path override (for testing)
            """
            if layer_name not in LAYER_PATHS and base_path is None:
                raise ValueError(
                    f"Unknown layer: {layer_name}. "
                    f"Must be one of: {list(LAYER_PATHS.keys())}"
                )

            self.layer_name = layer_name

            if base_path is not None:
                self.base_path = base_path / PREDICTIONS_DIR
            else:
                self.base_path = LAYER_PATHS[layer_name] / PREDICTIONS_DIR

            logger.debug(
                f"Initialized PolarsHiveReader for {layer_name} "
                f"at {self.base_path}"
            )

        def load_recent_predictions_lazy(
            self,
            days: int = 56,
            model_version: Optional[str] = None,
            return_pandas: bool = True
        ) -> pd.DataFrame:
            """
            Load with Polars lazy evaluation (BLAZING FAST).

            Polars can scan Hive partitions and push filters down.

            Args:
                days: Number of days to load
                model_version: Model version (default: latest)
                return_pandas: If True, convert to pandas (default: True)

            Returns:
                DataFrame with predictions
            """
            if model_version is None:
                model_version = self._get_latest_model_version()

            logger.info(
                f"Loading last {days} days with Polars lazy evaluation "
                f"for {self.layer_name}/{model_version}"
            )

            # Build glob pattern for Hive partitions
            pattern = str(
                self.base_path /
                f"model_version={model_version}" /
                "**" /
                "*.parquet"
            )

            # Scan Hive partitioned dataset (lazy)
            lf = pl.scan_parquet(
                pattern,
                hive_partitioning=True
            )

            # Filter with lazy evaluation (pushed down to file scan)
            cutoff_date = datetime.now() - timedelta(days=days)

            lf = lf.filter(
                pl.col("_prediction_date") >= cutoff_date
            )

            # Remove metadata columns
            meta_cols = [c for c in METADATA_COLUMNS]
            lf = lf.drop(meta_cols)

            # Remove duplicates (keep last)
            # Note: Polars requires sorting first for efficient unique
            lf = lf.sort("timestamp")  # Assumes index is 'timestamp' column
            lf = lf.unique(subset=["timestamp"], keep="last")

            # Collect (executes lazy query)
            df_polars = lf.collect()

            logger.info(
                f"📊 Loaded {len(df_polars)} predictions with Polars"
            )

            # Convert to pandas if requested
            if return_pandas:
                df_pandas = df_polars.to_pandas()
                # Set index to timestamp if it exists
                if "timestamp" in df_pandas.columns:
                    df_pandas = df_pandas.set_index("timestamp")
                return df_pandas
            else:
                return df_polars

        def _get_latest_model_version(self) -> str:
            """Get latest model version from filesystem."""
            if not self.base_path.exists():
                raise ValueError(
                    f"Predictions path does not exist: {self.base_path}"
                )

            version_dirs = [
                d.name.split('=')[1]
                for d in self.base_path.glob("model_version=*")
                if d.is_dir()
            ]

            if not version_dirs:
                raise ValueError(
                    f"No model versions found in {self.base_path}"
                )

            # Sort versions
            def version_key(v: str) -> tuple:
                v_clean = v.lstrip('v')
                try:
                    return tuple(map(int, v_clean.split('.')))
                except ValueError:
                    return (v,)

            return sorted(version_dirs, key=version_key)[-1]

except ImportError:
    logger.warning(
        "Polars not installed. PolarsHiveReader will not be available. "
        "Install with: pip install polars"
    )

    class PolarsHiveReader:
        """Placeholder class when Polars is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Polars is not installed. Install with: pip install polars"
            )
