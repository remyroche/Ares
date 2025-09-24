"""
Data Utilities Integration Module

This module integrates data utilities from src/utils/data/ for improved data handling
and feature engineering in the hybrid NAS-TAS regime detection system.

Integrated modules:
- src/utils/data/unified_data_utils.py
- src/utils/data/feature_engineer.py
- src/utils/data/quality/ modules
- src/utils/data/processing/ modules
- src/utils/data/validation/ modules
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add src to path for imports
src_path = Path(__file__).parents[4] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

logger = logging.getLogger(__name__)

# =============================================================================
# UNIFIED DATA UTILITIES INTEGRATION
# =============================================================================

class UnifiedDataUtils:
    """Unified interface for all data processing operations."""

    def __init__(self):
        self.logger = logger.getChild('UnifiedDataUtils')

    def process_and_validate(self, data, validate_quality: bool = True,
                           clean_missing_values: bool = True, detect_outliers: bool = True,
                           optimize_dtypes: bool = True) -> Any:
        """Process and validate data in one go."""
        try:
            # Try to use external unified data utils
            from utils.data.unified_data_utils import UnifiedDataUtils as _UnifiedDataUtils
            return _UnifiedDataUtils().process_and_validate(
                data=data,
                validate_quality=validate_quality,
                clean_missing_values=clean_missing_values,
                detect_outliers=detect_outliers,
                optimize_dtypes=optimize_dtypes
            )
        except ImportError:
            # Fallback implementation
            try:
                import pandas as pd

                # Basic processing
                if isinstance(data, pd.DataFrame):
                    processed_data = data.copy()

                    # Optimize dtypes if requested
                    if optimize_dtypes:
                        processed_data = self._optimize_dtypes(processed_data)

                    # Clean missing values if requested
                    if clean_missing_values:
                        processed_data = self._clean_missing_values(processed_data)

                    # Validate quality if requested
                    if validate_quality:
                        quality_report = self._validate_quality(processed_data)
                        self.logger.info(f"Data quality report: {quality_report}")

                    return processed_data
                else:
                    return data
            except Exception as e:
                self.logger.error(f"Error processing data: {e}")
                return data

    def _optimize_dtypes(self, df):
        """Optimize DataFrame data types for memory efficiency."""
        try:
            import pandas as pd
            import numpy as np

            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric
                    try:
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    except:
                        try:
                            df[col] = pd.to_numeric(df[col], downcast='float')
                        except:
                            pass
                elif df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')

            return df
        except Exception as e:
            self.logger.warning(f"Error optimizing dtypes: {e}")
            return df

    def _clean_missing_values(self, df):
        """Clean missing values in DataFrame."""
        try:
            import pandas as pd

            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())

            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')

            return df
        except Exception as e:
            self.logger.warning(f"Error cleaning missing values: {e}")
            return df

    def _validate_quality(self, df) -> Dict[str, Any]:
        """Validate data quality."""
        try:
            import pandas as pd
            import numpy as np

            quality_report = {
                'shape': df.shape,
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
            }

            return quality_report
        except Exception as e:
            self.logger.error(f"Error validating quality: {e}")
            return {'error': str(e)}

# =============================================================================
# FEATURE ENGINEERING INTEGRATION
# =============================================================================

class FeatureEngineer:
    """Feature engineering utilities."""

    def __init__(self):
        self.logger = logger.getChild('FeatureEngineer')

    def engineer_features(self, data, feature_types: List[str] = None) -> Any:
        """Engineer features from data."""
        try:
            # Try to use external feature engineer
            from utils.data.feature_engineer import engineer_features as _engineer_features
            return _engineer_features(data, feature_types)
        except ImportError:
            # Fallback implementation
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(data, pd.DataFrame):
                    return data

                df = data.copy()

                if feature_types is None:
                    feature_types = ['basic', 'technical', 'statistical']

                # Basic features
                if 'basic' in feature_types:
                    self._add_basic_features(df)

                # Technical indicators
                if 'technical' in feature_types:
                    self._add_technical_indicators(df)

                # Statistical features
                if 'statistical' in feature_types:
                    self._add_statistical_features(df)

                return df
            except Exception as e:
                self.logger.error(f"Error engineering features: {e}")
                return data

    def _add_basic_features(self, df):
        """Add basic features like returns, volatility."""
        try:
            # Price returns
            if 'close' in df.columns:
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Volume features
            if 'volume' in df.columns:
                df['volume_change'] = df['volume'].pct_change()
                df['volume_ma_5'] = df['volume'].rolling(5).mean()
                df['volume_ma_20'] = df['volume'].rolling(20).mean()

        except Exception as e:
            self.logger.warning(f"Error adding basic features: {e}")

    def _add_technical_indicators(self, df):
        """Add technical indicators."""
        try:
            # Moving averages
            if 'close' in df.columns:
                df['ma_5'] = df['close'].rolling(5).mean()
                df['ma_20'] = df['close'].rolling(20).mean()
                df['ma_50'] = df['close'].rolling(50).mean()

            # RSI
            if 'close' in df.columns:
                df['rsi'] = self._calculate_rsi(df['close'])

            # Bollinger Bands
            if 'close' in df.columns:
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])

        except Exception as e:
            self.logger.warning(f"Error adding technical indicators: {e}")

    def _add_statistical_features(self, df):
        """Add statistical features."""
        try:
            # Rolling statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                df[f'{col}_rolling_mean_5'] = df[col].rolling(5).mean()
                df[f'{col}_rolling_std_5'] = df[col].rolling(5).std()
                df[f'{col}_rolling_mean_20'] = df[col].rolling(20).mean()
                df[f'{col}_rolling_std_20'] = df[col].rolling(20).std()

        except Exception as e:
            self.logger.warning(f"Error adding statistical features: {e}")

    def _calculate_rsi(self, prices, period: int = 14) -> Any:
        """Calculate RSI (Relative Strength Index)."""
        try:
            import pandas as pd
            import numpy as np

            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi
        except Exception:
            return None

    def _calculate_bollinger_bands(self, prices, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands."""
        try:
            import pandas as pd
            import numpy as np

            middle_band = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()

            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)

            return upper_band, middle_band, lower_band
        except Exception:
            return None, None, None

# =============================================================================
# DATA QUALITY INTEGRATION
# =============================================================================

class DataQualityFramework:
    """Data quality validation and assessment framework."""

    def __init__(self):
        self.logger = logger.getChild('DataQualityFramework')

    def assess_quality(self, data) -> Dict[str, Any]:
        """Assess data quality."""
        try:
            # Try to use external data quality framework
            from utils.data.quality.data_quality import DataQualityFramework as _DataQualityFramework
            return _DataQualityFramework().assess_quality(data)
        except ImportError:
            # Fallback implementation
            try:
                import pandas as pd
                import numpy as np

                if not isinstance(data, pd.DataFrame):
                    return {'error': 'Data must be a DataFrame'}

                quality_metrics = {
                    'completeness': self._assess_completeness(data),
                    'accuracy': self._assess_accuracy(data),
                    'consistency': self._assess_consistency(data),
                    'validity': self._assess_validity(data),
                    'uniqueness': self._assess_uniqueness(data),
                    'timeliness': self._assess_timeliness(data),
                    'overall_score': 0.0
                }

                # Calculate overall score
                weights = {'completeness': 0.25, 'accuracy': 0.20, 'consistency': 0.20,
                          'validity': 0.15, 'uniqueness': 0.15, 'timeliness': 0.05}

                overall_score = sum(quality_metrics[metric] * weight
                                  for metric, weight in weights.items()
                                  if metric in quality_metrics)

                quality_metrics['overall_score'] = overall_score

                return quality_metrics
            except Exception as e:
                self.logger.error(f"Error assessing quality: {e}")
                return {'error': str(e)}

    def _assess_completeness(self, df) -> float:
        """Assess completeness (lack of missing values)."""
        try:
            total_values = df.shape[0] * df.shape[1]
            missing_values = df.isnull().sum().sum()
            return 1.0 - (missing_values / total_values) if total_values > 0 else 0.0
        except Exception:
            return 0.0

    def _assess_accuracy(self, df) -> float:
        """Assess accuracy (reasonable value ranges)."""
        try:
            # Basic heuristic: check if numeric values are in reasonable ranges
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                return 1.0

            # Count values that are likely outliers (beyond 3 std devs)
            outlier_count = 0
            total_count = 0

            for col in numeric_cols:
                if df[col].std() > 0:  # Avoid division by zero
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers = (z_scores > 3).sum()
                    outlier_count += outliers
                    total_count += len(df[col])

            return 1.0 - (outlier_count / total_count) if total_count > 0 else 1.0
        except Exception:
            return 0.5  # Default score

    def _assess_consistency(self, df) -> float:
        """Assess consistency (consistent data types and formats)."""
        try:
            # Check for mixed data types in columns
            consistency_score = 1.0

            for col in df.columns:
                dtype = df[col].dtype
                # Check if column has consistent type
                if df[col].apply(type).nunique() > 1:
                    consistency_score -= 0.1  # Penalty for mixed types

            return max(0.0, consistency_score)
        except Exception:
            return 0.5

    def _assess_validity(self, df) -> float:
        """Assess validity (data follows expected patterns)."""
        try:
            # Basic validation checks
            validity_score = 1.0

            # Check for negative prices/volumes
            price_cols = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower() or 'open' in col.lower()]
            for col in price_cols:
                if (df[col] < 0).any():
                    validity_score -= 0.2

            volume_cols = [col for col in df.columns if 'volume' in col.lower()]
            for col in volume_cols:
                if (df[col] < 0).any():
                    validity_score -= 0.2

            return max(0.0, validity_score)
        except Exception:
            return 0.5

    def _assess_uniqueness(self, df) -> float:
        """Assess uniqueness (lack of duplicate records)."""
        try:
            duplicate_count = df.duplicated().sum()
            total_count = len(df)
            return 1.0 - (duplicate_count / total_count) if total_count > 0 else 1.0
        except Exception:
            return 0.5

    def _assess_timeliness(self, df) -> float:
        """Assess timeliness (data freshness)."""
        try:
            # Basic heuristic: check if data has recent timestamps
            datetime_cols = df.select_dtypes(include=['datetime64']).columns

            if len(datetime_cols) == 0:
                return 0.5  # Neutral score if no timestamps

            # Check if most recent data is within last 24 hours
            latest_time = df[datetime_cols[0]].max()
            from datetime import datetime, timedelta

            if isinstance(latest_time, pd.Timestamp):
                time_diff = datetime.now() - latest_time.to_pydatetime()
                if time_diff < timedelta(hours=24):
                    return 1.0  # Very fresh
                elif time_diff < timedelta(days=7):
                    return 0.7  # Moderately fresh
                else:
                    return 0.3  # Stale data
            else:
                return 0.5  # Unknown
        except Exception:
            return 0.5

# =============================================================================
# MAIN INTEGRATION FUNCTIONS
# =============================================================================

def get_unified_data_utils():
    """Get unified data utilities instance."""
    try:
        from utils.data.unified_data_utils import UnifiedDataUtils as _UnifiedDataUtils
        return _UnifiedDataUtils()
    except ImportError:
        return UnifiedDataUtils()

def get_feature_engineer():
    """Get feature engineer instance."""
    try:
        from utils.data.feature_engineer import FeatureEngineer as _FeatureEngineer
        return _FeatureEngineer()
    except ImportError:
        return FeatureEngineer()

def get_data_quality_framework():
    """Get data quality framework instance."""
    try:
        from utils.data.quality.data_quality import DataQualityFramework as _DataQualityFramework
        return _DataQualityFramework()
    except ImportError:
        return DataQualityFramework()

def process_market_data(data, enhance_features: bool = True, validate_quality: bool = True):
    """Process market data with feature engineering and quality validation."""
    try:
        utils = get_unified_data_utils()
        processed_data = utils.process_and_validate(
            data=data,
            validate_quality=validate_quality,
            clean_missing_values=True,
            detect_outliers=True,
            optimize_dtypes=True
        )

        if enhance_features:
            engineer = get_feature_engineer()
            processed_data = engineer.engineer_features(
                processed_data,
                feature_types=['basic', 'technical', 'statistical']
            )

        return processed_data
    except Exception as e:
        logger.error(f"Error processing market data: {e}")
        return data