"""Resilient access to :mod:`src.utils.common_operations` for NAS/TAS modules."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .dependency_management import dependency_manager
from ..fallback_utilities import FallbackConfig, FallbackHardwareUtils, FallbackMathUtils

__all__ = [
    "COMMON_OPERATIONS_AVAILABLE",
    "CommonUtilities",
    "align_dataframes",
    "calculate_data_quality_metrics",
    "check_disk_space",
    "cleanup_m1_optimizers",
    "create_data_quality_report",
    "create_summary_statistics",
    "ensure_directory",
    "format_datetime",
    "get_current_datetime",
    "get_dataframe_info",
    "get_file_size",
    "get_m1_cpu_optimizer",
    "get_m1_gpu_manager",
    "get_m1_memory_optimizer",
    "get_memory_usage",
    "gpu_context",
    "guard_dataframe_nulls",
    "integrate_with_m1_optimizers",
    "list_parquet_files",
    "memory_checkpoint",
    "optimize_dataframe_dtypes",
    "optimize_memory",
    "safe_apply_function",
    "safe_convert_dtypes",
    "safe_correlation",
    "safe_covariance",
    "safe_dataframe_operation",
    "safe_deepcopy",
    "safe_divide",
    "safe_drop_columns",
    "safe_file_exists",
    "safe_filter_dataframe",
    "safe_float",
    "safe_groupby_operation",
    "safe_int",
    "safe_json_dump",
    "safe_json_load",
    "safe_log",
    "safe_mean",
    "safe_merge_dataframes",
    "safe_percentile",
    "safe_power",
    "safe_read_parquet",
    "safe_rename_columns",
    "safe_sqrt",
    "safe_std",
    "safe_timestamp_conversion",
    "safe_to_parquet",
    "safe_weighted_average",
    "validate_dataframe_columns",
    "validate_dataframe_schema",
    "validate_file_path",
    "validate_finite",
    "validate_positive",
    "validate_range",
    "validate_timestamp_column",
]

logger = logging.getLogger(__name__)

_COMMON_OPS_MODULE = dependency_manager.import_optional(
    "src.utils.common_operations",
    install_hint="pip install nas-tas-commons extras or include src/utils/common_operations.py",
)
COMMON_OPERATIONS_AVAILABLE = _COMMON_OPS_MODULE is not None

_math_fallback = FallbackMathUtils(FallbackConfig(enable_logging=True))
_hardware_fallback = FallbackHardwareUtils(FallbackConfig(enable_logging=True))
_logged_fallbacks: set[str] = set()


def _log_fallback(name: str) -> None:
    if name in _logged_fallbacks:
        return
    logger.warning(
        "Using fallback implementation for src.utils.common_operations.%s because the optional module is unavailable.",
        name,
    )
    _logged_fallbacks.add(name)


def _delegate(name: str, fallback: Optional[Callable[..., Any]]) -> Callable[..., Any]:
    if COMMON_OPERATIONS_AVAILABLE and hasattr(_COMMON_OPS_MODULE, name):
        attr = getattr(_COMMON_OPS_MODULE, name)
        if callable(attr):
            return attr
        # For non-callable attributes (e.g. classes) the caller should use `_class_delegate`.
    if fallback is None:
        def _missing(*_args: Any, **_kwargs: Any) -> Any:
            _log_fallback(name)
            raise RuntimeError(
                f"Optional dependency 'src.utils.common_operations.{name}' is unavailable. "
                "Install NAS/TAS common utilities to access this feature."
            )
        return _missing

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        _log_fallback(name)
        return fallback(*args, **kwargs)

    return _wrapped


def _class_delegate(name: str, fallback: type) -> type:
    if COMMON_OPERATIONS_AVAILABLE and hasattr(_COMMON_OPS_MODULE, name):
        attr = getattr(_COMMON_OPS_MODULE, name)
        if isinstance(attr, type):
            return attr
    _log_fallback(name)
    return fallback


class _FallbackCommonUtilities:
    """Light-weight substitute for :class:`src.utils.common_operations.CommonUtilities`."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.FallbackCommonUtilities")

    def get_m1_status(self) -> Dict[str, bool]:
        return {"m1_available": False, "mps_available": False}

    def optimize_for_m1(self, data: Any) -> Any:
        return data

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "m1_available": False,
            "mps_available": False,
            "platform": os.name,
            "python_version": "{}.{}.{}".format(*os.sys.version_info[:3]),
        }


# ---------------------------------------------------------------------------
# Numeric fallbacks via FallbackMathUtils
# ---------------------------------------------------------------------------
safe_divide = _delegate("safe_divide", _math_fallback.safe_divide)
safe_log = _delegate("safe_log", _math_fallback.safe_log)
safe_sqrt = _delegate("safe_sqrt", _math_fallback.safe_sqrt)
safe_mean = _delegate("safe_mean", _math_fallback.safe_mean)
safe_std = _delegate("safe_std", _math_fallback.safe_std)
safe_correlation = _delegate("safe_correlation", _math_fallback.safe_correlation)
safe_covariance = _delegate("safe_covariance", _math_fallback.safe_covariance)
safe_percentile = _delegate("safe_percentile", _math_fallback.safe_percentile)
safe_weighted_average = _delegate("safe_weighted_average", _math_fallback.safe_weighted_average)
safe_power = _delegate("safe_power", _math_fallback.safe_power)
safe_rename_columns = _delegate("safe_rename_columns", _safe_rename_columns_impl)


# ---------------------------------------------------------------------------
# DataFrame fallbacks
# ---------------------------------------------------------------------------

def _safe_dataframe_operation(
    operation: Callable[[pd.DataFrame, Any], Any],
    dataframe: pd.DataFrame,
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        return operation(dataframe, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"DataFrame operation failed during fallback execution: {exc}") from exc


def _safe_convert_dtypes(dataframe: pd.DataFrame, dtype_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    result = dataframe.copy()
    if dtype_map:
        for column, dtype in dtype_map.items():
            if column in result.columns:
                try:
                    result[column] = result[column].astype(dtype)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Failed to convert column '{column}' to dtype '{dtype}' during fallback conversion: {exc}"
                    ) from exc
    else:
        result = result.convert_dtypes()
    return result


def _safe_merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    try:
        return df1.merge(df2, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to merge DataFrames during fallback execution: {exc}") from exc


def _safe_groupby_operation(
    dataframe: pd.DataFrame,
    by: Union[str, Sequence[str]],
    operation: Callable[[pd.core.groupby.generic.DataFrameGroupBy], Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        grouped = dataframe.groupby(by)
        return operation(grouped, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Fallback groupby operation failed: {exc}") from exc


def _safe_apply_function(
    dataframe: pd.DataFrame,
    func: Callable[[pd.Series], Any],
    axis: int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    try:
        return dataframe.apply(func, axis=axis, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Fallback apply operation failed: {exc}") from exc


def _safe_drop_columns(dataframe: pd.DataFrame, columns: Sequence[str], errors: str = "raise") -> pd.DataFrame:
    try:
        return dataframe.drop(columns=list(columns), errors=errors)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Fallback drop columns failed: {exc}") from exc


def _safe_filter_dataframe(dataframe: pd.DataFrame, condition: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    try:
        mask = condition(dataframe)
        return dataframe.loc[mask]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Fallback DataFrame filter failed: {exc}") from exc


def _safe_timestamp_conversion(dataframe: pd.DataFrame, column: str, utc: bool = True) -> pd.DataFrame:
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not present for timestamp conversion")
    result = dataframe.copy()
    result[column] = pd.to_datetime(result[column], errors="coerce", utc=utc)
    return result


def _safe_rename_columns_impl(dataframe: pd.DataFrame, rename_map: Dict[str, str], **kwargs: Any) -> pd.DataFrame:
    return dataframe.rename(columns=rename_map, **kwargs)


def _safe_to_parquet(dataframe: pd.DataFrame, path: Union[str, Path], **kwargs: Any) -> None:
    try:
        dataframe.to_parquet(path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to write parquet file via fallback implementation: {exc}") from exc


def _safe_read_parquet(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read parquet file via fallback implementation: {exc}") from exc


def _optimize_dataframe_dtypes(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in result.select_dtypes(include=["float64"]).columns:
        result[column] = result[column].astype("float32")
    for column in result.select_dtypes(include=["int64"]).columns:
        result[column] = pd.to_numeric(result[column], downcast="integer")
    return result


def _align_dataframes(primary: pd.DataFrame, secondary: pd.DataFrame, how: str = "inner") -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        joined_index = primary.index.join(secondary.index, how=how)
        return primary.reindex(joined_index), secondary.reindex(joined_index)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to align DataFrames via fallback implementation: {exc}") from exc


def _guard_dataframe_nulls(dataframe: pd.DataFrame, fill_value: Any = 0) -> pd.DataFrame:
    return dataframe.fillna(fill_value)


def _calculate_data_quality_metrics(dataframe: pd.DataFrame) -> Dict[str, Any]:
    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "null_counts": dataframe.isnull().sum().to_dict(),
    }


def _create_data_quality_report(dataframe: pd.DataFrame) -> Dict[str, Any]:
    metrics = _calculate_data_quality_metrics(dataframe)
    metrics["sample"] = dataframe.head(5).to_dict(orient="records")
    return metrics


def _create_summary_statistics(dataframe: pd.DataFrame) -> Dict[str, Any]:
    return {
        "describe": dataframe.describe(include="all", datetime_is_numeric=True).to_dict(),
        "dtypes": dataframe.dtypes.astype(str).to_dict(),
    }


def _get_dataframe_info(dataframe: pd.DataFrame) -> Dict[str, Any]:
    buffer: list[str] = []

    def _collector(line: str) -> None:
        buffer.append(line)

    dataframe.info(buf=_collector)  # type: ignore[arg-type]
    return {"info": buffer}


safe_dataframe_operation = _delegate("safe_dataframe_operation", _safe_dataframe_operation)
safe_convert_dtypes = _delegate("safe_convert_dtypes", _safe_convert_dtypes)
safe_merge_dataframes = _delegate("safe_merge_dataframes", _safe_merge_dataframes)
safe_groupby_operation = _delegate("safe_groupby_operation", _safe_groupby_operation)
safe_apply_function = _delegate("safe_apply_function", _safe_apply_function)
safe_drop_columns = _delegate("safe_drop_columns", _safe_drop_columns)
safe_filter_dataframe = _delegate("safe_filter_dataframe", _safe_filter_dataframe)
safe_timestamp_conversion = _delegate("safe_timestamp_conversion", _safe_timestamp_conversion)
safe_to_parquet = _delegate("safe_to_parquet", _safe_to_parquet)
safe_read_parquet = _delegate("safe_read_parquet", _safe_read_parquet)
optimize_dataframe_dtypes = _delegate("optimize_dataframe_dtypes", _optimize_dataframe_dtypes)
align_dataframes = _delegate("align_dataframes", _align_dataframes)
create_data_quality_report = _delegate("create_data_quality_report", _create_data_quality_report)
calculate_data_quality_metrics = _delegate("calculate_data_quality_metrics", _calculate_data_quality_metrics)
create_summary_statistics = _delegate("create_summary_statistics", _create_summary_statistics)
get_dataframe_info = _delegate("get_dataframe_info", _get_dataframe_info)
guard_dataframe_nulls = _delegate("guard_dataframe_nulls", _guard_dataframe_nulls)


# ---------------------------------------------------------------------------
# Hardware helpers via FallbackHardwareUtils
# ---------------------------------------------------------------------------
get_memory_usage = _delegate("get_memory_usage", _hardware_fallback.get_memory_usage)
optimize_memory = _delegate("optimize_memory", _hardware_fallback.optimize_memory)
memory_checkpoint = _delegate("memory_checkpoint", _hardware_fallback.memory_checkpoint)
gpu_context = _delegate("gpu_context", _hardware_fallback.gpu_context)


# ---------------------------------------------------------------------------
# File and OS helpers
# ---------------------------------------------------------------------------

def _ensure_directory(path: Union[str, Path]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _list_parquet_files(directory: Union[str, Path]) -> Iterable[Path]:
    directory_path = Path(directory)
    if not directory_path.exists():
        return []
    return list(directory_path.glob("*.parquet"))


def _get_file_size(path: Union[str, Path]) -> int:
    return Path(path).stat().st_size


def _check_disk_space(path: Union[str, Path]) -> Dict[str, int]:
    stat = os.statvfs(str(Path(path).resolve()))
    return {
        "total": stat.f_frsize * stat.f_blocks,
        "available": stat.f_frsize * stat.f_bavail,
    }


def _validate_file_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    resolved = Path(path)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path '{resolved}' does not exist")
    return resolved


def _validate_dataframe_columns(dataframe: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        _log_fallback("validate_dataframe_columns")
        logger.error("DataFrame is missing required columns: %s", missing)
        return False
    return True


def _validate_dataframe_schema(dataframe: pd.DataFrame, schema: Dict[str, str]) -> bool:
    mismatched = {
        column: (str(dataframe[column].dtype), expected)
        for column, expected in schema.items()
        if column in dataframe.columns and str(dataframe[column].dtype) != expected
    }
    if mismatched:
        _log_fallback("validate_dataframe_schema")
        logger.error("DataFrame dtype mismatch detected: %s", mismatched)
        return False
    return True


def _validate_timestamp_column(dataframe: pd.DataFrame, column: str) -> bool:
    if column not in dataframe.columns:
        _log_fallback("validate_timestamp_column")
        logger.error("Timestamp column '%s' is missing", column)
        return False
    if not np.issubdtype(dataframe[column].dtype, np.datetime64):
        _log_fallback("validate_timestamp_column")
        logger.error("Column '%s' is not datetime typed", column)
        return False
    return True


def _validate_positive(value: float, *, include_zero: bool = False) -> bool:
    if include_zero:
        valid = value >= 0
    else:
        valid = value > 0
    if not valid:
        _log_fallback("validate_positive")
        logger.error("Value %s failed positive validation (include_zero=%s)", value, include_zero)
    return valid


def _validate_range(value: float, minimum: float, maximum: float) -> bool:
    valid = minimum <= value <= maximum
    if not valid:
        _log_fallback("validate_range")
        logger.error("Value %s outside expected range [%s, %s]", value, minimum, maximum)
    return valid


def _validate_finite(value: float) -> bool:
    valid = np.isfinite(value)
    if not valid:
        _log_fallback("validate_finite")
        logger.error("Value %s is not finite", value)
    return valid


def _safe_file_exists(path: Union[str, Path]) -> bool:
    return Path(path).exists()


def _safe_json_dump(data: Any, path: Union[str, Path], *, indent: int = 2) -> bool:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=indent, default=str)
        return True
    except Exception as exc:  # noqa: BLE001
        _log_fallback("safe_json_dump")
        logger.error("Failed to write JSON to %s: %s", target, exc)
        return False


def _safe_json_load(path: Union[str, Path], default: Any = None) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except Exception as exc:  # noqa: BLE001
        _log_fallback("safe_json_load")
        logger.error("Failed to read JSON from %s: %s", path, exc)
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        _log_fallback("safe_int")
        if default is not None:
            logger.warning("Returning default integer value because conversion failed", exc_info=True)
            return default
        raise


def _safe_float(value: Any, default: Optional[float] = None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        _log_fallback("safe_float")
        if default is not None:
            logger.warning("Returning default float value because conversion failed", exc_info=True)
            return default
        raise


def _safe_deepcopy(value: Any) -> Any:
    import copy

    try:
        return copy.deepcopy(value)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Deep copy failed during fallback execution: {exc}") from exc


def _safe_timestamp_formatter(value: Union[int, float, datetime], fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if isinstance(value, datetime):
        return value.strftime(fmt)
    return datetime.fromtimestamp(float(value)).strftime(fmt)


def _safe_current_datetime() -> datetime:
    return datetime.utcnow()


ensure_directory = _delegate("ensure_directory", _ensure_directory)
list_parquet_files = _delegate("list_parquet_files", _list_parquet_files)
get_file_size = _delegate("get_file_size", _get_file_size)
check_disk_space = _delegate("check_disk_space", _check_disk_space)
validate_file_path = _delegate("validate_file_path", _validate_file_path)
validate_dataframe_columns = _delegate("validate_dataframe_columns", _validate_dataframe_columns)
validate_dataframe_schema = _delegate("validate_dataframe_schema", _validate_dataframe_schema)
validate_timestamp_column = _delegate("validate_timestamp_column", _validate_timestamp_column)
validate_positive = _delegate("validate_positive", _validate_positive)
validate_range = _delegate("validate_range", _validate_range)
validate_finite = _delegate("validate_finite", _validate_finite)
safe_file_exists = _delegate("safe_file_exists", _safe_file_exists)
safe_json_dump = _delegate("safe_json_dump", _safe_json_dump)
safe_json_load = _delegate("safe_json_load", _safe_json_load)
safe_int = _delegate("safe_int", _safe_int)
safe_float = _delegate("safe_float", _safe_float)
safe_deepcopy = _delegate("safe_deepcopy", _safe_deepcopy)
format_datetime = _delegate("format_datetime", _safe_timestamp_formatter)
get_current_datetime = _delegate("get_current_datetime", _safe_current_datetime)


# ---------------------------------------------------------------------------
# Hardware optimisers – fallbacks raise informative errors when unavailable
# ---------------------------------------------------------------------------

def _missing_hardware_utility(name: str) -> Callable[..., Any]:
    def _raiser(*_args: Any, **_kwargs: Any) -> Any:
        _log_fallback(name)
        raise RuntimeError(
            "Optional Apple Silicon optimisation utilities are unavailable. "
            "Install NAS/TAS hardware extras to access this feature."
        )

    return _raiser


def _delegate_hardware(name: str) -> Callable[..., Any]:
    fallback = _missing_hardware_utility(name)
    return _delegate(name, fallback)


get_m1_gpu_manager = _delegate_hardware("get_m1_gpu_manager")
get_m1_memory_optimizer = _delegate_hardware("get_m1_memory_optimizer")
get_m1_cpu_optimizer = _delegate_hardware("get_m1_cpu_optimizer")


def _cleanup_m1_optimizers() -> bool:
    _log_fallback("cleanup_m1_optimizers")
    logger.warning("M1 optimizer cleanup skipped because hardware utilities are unavailable")
    return False


def _integrate_with_m1_optimizers() -> Dict[str, Any]:
    _log_fallback("integrate_with_m1_optimizers")
    logger.warning("Apple Silicon optimizers are unavailable; integration skipped")
    return {
        "gpu_manager": None,
        "memory_optimizer": None,
        "cpu_optimizer": None,
        "memory_monitoring_active": False,
        "cpu_optimizer_active": False,
    }


cleanup_m1_optimizers = _delegate("cleanup_m1_optimizers", _cleanup_m1_optimizers)
integrate_with_m1_optimizers = _delegate("integrate_with_m1_optimizers", _integrate_with_m1_optimizers)


# ---------------------------------------------------------------------------
# Complex types
# ---------------------------------------------------------------------------
CommonUtilities = _class_delegate("CommonUtilities", _FallbackCommonUtilities)
