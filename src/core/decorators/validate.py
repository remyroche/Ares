"""
Request and DTO validation decorators.

Provides decorators for validating function inputs using various
validation strategies (pydantic, dataclasses, custom validators).
"""

import inspect
import logging
from typing import get_type_hints, Callable, Any

from ..errors.base import ValidationError
from .compose import P, R, uniform_wrapper

# Try to import optional validation libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import pydantic
    PYDANTIC_AVAILABLE = True
except ImportError:
    pydantic = None
    PYDANTIC_AVAILABLE = False

def validates(
    *,
    strict: bool = True,
    coerce: bool = False,
    extra: str = "forbid",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Validate function arguments based on type hints.

    Supports pydantic models, dataclasses, and basic type validation.

    Args:
        strict: Whether to enforce strict validation
        coerce: Whether to coerce values to the expected type
        extra: How to handle extra fields ("allow", "forbid", "ignore")

    Example:
        @validates(strict = True)
        def create_user(name: str, age: int, email: str) -> dict:
            return {"name": name, "age": age, "email": email}
    """

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        # Get function signature and type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Bind arguments
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as e:
            msg = f"Invalid arguments: {e}"
            raise ValidationError(message = msg)

        # Validate each argument
        errors = []
        for param_name, param_value in bound.arguments.items():
            if param_name in type_hints:
                expected_type = type_hints[param_name]

                # Skip if no validation needed
                if expected_type is Any:
                    continue

                # Validate the parameter
                try:
                    _validate_param(
                        param_name,
                        param_value,
                        expected_type,
                        strict = strict,
                        coerce = coerce,
                    )

                    # Update with coerced value if needed
                    if coerce and not isinstance(param_value, expected_type):
                        bound.arguments[param_name] = _coerce_value(
                            param_value,
                            expected_type,
                        )
                except ValidationError:
                    raise
                except Exception as e:
                    errors.append(f"{param_name}: {e}")

        if errors:
            msg = f"Validation failed for {func.__name__}"
            raise ValidationError(
                message = msg,
                details={"errors": errors},
            )

        # Call function with validated arguments
        return func(**bound.arguments)

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        # Reuse sync validation logic
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as e:
            msg = f"Invalid arguments: {e}"
            raise ValidationError(message = msg)

        errors = []
        for param_name, param_value in bound.arguments.items():
            if param_name in type_hints:
                expected_type = type_hints[param_name]

                if expected_type is Any:
                    continue

                try:
                    _validate_param(
                        param_name,
                        param_value,
                        expected_type,
                        strict = strict,
                        coerce = coerce,
                    )

                    if coerce and not isinstance(param_value, expected_type):
                        bound.arguments[param_name] = _coerce_value(
                            param_value,
                            expected_type,
                        )
                except ValidationError:
                    raise
                except Exception as e:
                    errors.append(f"{param_name}: {e}")

        if errors:
            msg = f"Validation failed for {func.__name__}"
            raise ValidationError(
                message = msg,
                details={"errors": errors},
            )

        return await func(**bound.arguments)

    return uniform_wrapper("validates", sync_handler, async_handler)

def validate_schema(
    schema: type | dict[str, type],
    *,
    param_name: str | None = None,
    allow_extra: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Validate a specific parameter against a schema.

    Args:
        schema: Schema to validate against (pydantic model, dict, or type)
        param_name: Parameter name to validate (defaults to first param)
        allow_extra: Whether to allow extra fields

    Example:
        @validate_schema(UserCreateSchema)
        def create_user(data: dict) -> User:
            return User(**data)
    """

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        # Determine which parameter to validate
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if param_name:
            if param_name not in sig.parameters:
                msg = f"Parameter {param_name} not found in {func.__name__}"
                raise ValueError(msg)
            param_value = (
                kwargs.get(param_name)
                if param_name in kwargs
                else args[params.index(param_name)]
            )
        else:
            # Default to first parameter
            if not args:
                msg = "No arguments provided to validate"
                raise ValidationError(message = msg)
            param_value = args[0]
            param_name_used = params[0] if params else "arg0"

        # Validate against schema
        validated_value = _validate_against_schema(
            param_value,
            schema,
            param_name or param_name_used,
            allow_extra,
        )

        # Replace with validated value
        if param_name and param_name in kwargs:
            kwargs = dict(kwargs)
            kwargs[param_name] = validated_value
            return func(*args, **kwargs)
        if param_name:
            args = list(args)
            args[params.index(param_name)] = validated_value
            return func(*args, **kwargs)
        args = list(args)
        args[0] = validated_value
        return func(*args, **kwargs)

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        # Reuse sync validation logic
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if param_name:
            if param_name not in sig.parameters:
                msg = f"Parameter {param_name} not found in {func.__name__}"
                raise ValueError(msg)
            param_value = (
                kwargs.get(param_name)
                if param_name in kwargs
                else args[params.index(param_name)]
            )
        else:
            if not args:
                msg = "No arguments provided to validate"
                raise ValidationError(message = msg)
            param_value = args[0]
            param_name_used = params[0] if params else "arg0"

        validated_value = _validate_against_schema(
            param_value,
            schema,
            param_name or param_name_used,
            allow_extra,
        )

        if param_name and param_name in kwargs:
            kwargs = dict(kwargs)
            kwargs[param_name] = validated_value
            return await func(*args, **kwargs)
        if param_name:
            args = list(args)
            args[params.index(param_name)] = validated_value
            return await func(*args, **kwargs)
        args = list(args)
        args[0] = validated_value
        return await func(*args, **kwargs)

    return uniform_wrapper(
        f"validate_schema({schema.__name__ if hasattr(schema, '__name__') else 'dict'})",
        sync_handler,
        async_handler,
    )

def validate_dataframe(
    *,
    columns: list[str] | None = None,
    dtypes: dict[str, type] | None = None,
    min_rows: int = 0,
    max_rows: int | None = None,
    param_name: str = "df",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Validate pandas DataFrame parameters.

    Args:
        columns: Required column names
        dtypes: Expected column data types
        min_rows: Minimum number of rows
        max_rows: Maximum number of rows
        param_name: Parameter name containing the DataFrame

    Example:
        @validate_dataframe(
            columns=["id", "name", "value"],
            dtypes={"id": int, "value": float},
            min_rows = 1
        )
        def process_data(df: pd.DataFrame) -> pd.DataFrame:
            return df.groupby("name").sum()
    """
    if not PANDAS_AVAILABLE:
        msg = "pandas is required for DataFrame validation"
        raise ImportError(msg)

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        # Get the DataFrame parameter
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        if param_name not in bound.arguments:
            msg = f"Parameter {param_name} not found"
            raise ValidationError(message = msg)

        df = bound.arguments[param_name]

        if not isinstance(df, pd.DataFrame):
            msg = f"Parameter {param_name} must be a pandas DataFrame"
            raise ValidationError(
                message = msg,
                field = param_name,
                value = type(df).__name__,
            )

        # Validate columns
        if columns:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                msg = f"Missing required columns: {missing_cols}"
                raise ValidationError(
                    message = msg,
                    field = param_name,
                    details={"missing_columns": list(missing_cols)},
                )

        # Validate dtypes
        if dtypes:
            for col, expected_type in dtypes.items():
                if col in df.columns:
                    actual_type = df[col].dtype
                    # Simple type checking - can be enhanced
                    if expected_type == int and actual_type.kind not in "iu":
                        msg = f"Column {col} has wrong dtype: expected int-like, got {actual_type}"
                        raise ValidationError(
                            message = msg,
                            field = f"{param_name}.{col}",
                        )
                    if expected_type == float and actual_type.kind not in "iuf":
                        msg = f"Column {col} has wrong dtype: expected float-like, got {actual_type}"
                        raise ValidationError(
                            message = msg,
                            field = f"{param_name}.{col}",
                        )

        # Validate row count
        row_count = len(df)
        if row_count < min_rows:
            msg = f"DataFrame has too few rows: {row_count} < {min_rows}"
            raise ValidationError(
                message = msg,
                field = param_name,
                details={"row_count": row_count, "min_rows": min_rows},
            )

        if max_rows is not None and row_count > max_rows:
            msg = f"DataFrame has too many rows: {row_count} > {max_rows}"
            raise ValidationError(
                message = msg,
                field = param_name,
                details={"row_count": row_count, "max_rows": max_rows},
            )

        return func(*args, **kwargs)

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        # Reuse sync validation logic
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        if param_name not in bound.arguments:
            msg = f"Parameter {param_name} not found"
            raise ValidationError(message = msg)

        df = bound.arguments[param_name]

        # Run same validations as sync version
        if not isinstance(df, pd.DataFrame):
            msg = f"Parameter {param_name} must be a pandas DataFrame"
            raise ValidationError(
                message = msg,
                field = param_name,
                value = type(df).__name__,
            )

        # Validate columns
        if columns:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                msg = f"Missing required columns: {missing_cols}"
                raise ValidationError(
                    message = msg,
                    field = param_name,
                    details={"missing_columns": list(missing_cols)},
                )

        # Validate dtypes
        if dtypes:
            for col, expected_type in dtypes.items():
                if col in df.columns:
                    actual_type = df[col].dtype
                    # Simple type checking - can be enhanced
                    if expected_type == int and actual_type.kind not in "iu":
                        msg = f"Column {col} has wrong dtype: expected int-like, got {actual_type}"
                        raise ValidationError(
                            message = msg,
                            field = f"{param_name}.{col}",
                        )
                    if expected_type == float and actual_type.kind not in "iuf":
                        msg = f"Column {col} has wrong dtype: expected float-like, got {actual_type}"
                        raise ValidationError(
                            message = msg,
                            field = f"{param_name}.{col}",
                        )

        # Validate row count
        row_count = len(df)
        if row_count < min_rows:
            msg = f"DataFrame has too few rows: {row_count} < {min_rows}"
            raise ValidationError(
                message = msg,
                field = param_name,
                details={"row_count": row_count, "min_rows": min_rows},
            )

        if max_rows is not None and row_count > max_rows:
            msg = f"DataFrame has too many rows: {row_count} > {max_rows}"
            raise ValidationError(
                message = msg,
                field = param_name,
                details={"row_count": row_count, "max_rows": max_rows},
            )

        return await func(*args, **kwargs)

    return uniform_wrapper("validate_dataframe", sync_handler, async_handler)

# Helper functions

def _validate_param(
    name: str,
    value: Any,
    expected_type: type,
    strict: bool,
    coerce: bool,
) -> None:
    """Validate a single parameter."""
    # Handle pydantic models
    if PYDANTIC_AVAILABLE and hasattr(expected_type, "__pydantic_model__"):
        try:
            if isinstance(value, dict):
                expected_type(**value)
            elif not isinstance(value, expected_type):
                msg = f"Expected {expected_type.__name__}, got {type(value).__name__}"
                raise ValidationError(
                    message = msg,
                    field = name,
                )
        except pydantic.ValidationError as e:
            msg = f"Validation failed for {name}"
            raise ValidationError(
                message = msg,
                field = name,
                details={"pydantic_errors": e.errors()},
            )

    # Basic type checking
    elif strict and not isinstance(value, expected_type):
        msg = f"Expected {expected_type.__name__}, got {type(value).__name__}"
        raise ValidationError(
            message = msg,
            field = name,
            value = value,
        )

def _coerce_value(value: Any, target_type: type) -> Any:
    """Attempt to coerce a value to the target type."""
    try:
        if target_type in (int, float, str, bool):
            return target_type(value)
        return value
    except (ValueError, TypeError):
        return value

def _validate_against_schema(
    value: Any,
    schema: type | dict[str, type],
    param_name: str,
    allow_extra: bool,
) -> Any:
    """Validate a value against a schema."""
    # Handle pydantic models
    if PYDANTIC_AVAILABLE and hasattr(schema, "__pydantic_model__"):
        try:
            if isinstance(value, dict):
                return schema(**value)
            if isinstance(value, schema):
                return value
            msg = f"Cannot validate {type(value).__name__} against {schema.__name__}"
            raise ValidationError(
                message = msg,
                field = param_name,
            )
        except pydantic.ValidationError as e:
            msg = "Schema validation failed"
            raise ValidationError(
                message = msg,
                field = param_name,
                details={"errors": e.errors()},
            )

    # Handle dict schemas
    elif isinstance(schema, dict):
        if not isinstance(value, dict):
            msg = f"Expected dict for {param_name}, got {type(value).__name__}"
            raise ValidationError(
                message = msg,
                field = param_name,
            )

        errors = []
        validated = {}

        # Check required fields
        for field, field_type in schema.items():
            if field not in value:
                errors.append(f"Missing required field: {field}")
            else:
                try:
                    _validate_param(field, value[field], field_type, True, False)
                    validated[field] = value[field]
                except ValidationError as e:
                    errors.append(f"{field}: {e.message}")

        # Check extra fields
        if not allow_extra:
            extra_fields = set(value.keys()) - set(schema.keys())
            if extra_fields:
                errors.append(f"Extra fields not allowed: {extra_fields}")
        else:
            # Include extra fields in validated dict
            for field in value:
                if field not in validated:
                    validated[field] = value[field]

        if errors:
            msg = "Dict validation failed"
            raise ValidationError(
                message = msg,
                field = param_name,
                details={"errors": errors},
            )

        return validated

    # Basic type validation
    else:
        if not isinstance(value, schema):
            msg = f"Expected {schema.__name__}, got {type(value).__name__}"
            raise ValidationError(
                message = msg,
                field = param_name,
            )
        return value


def validate_data_quality(
    *,
    check_duplicates: bool = True,
    check_missing_values: bool = True,
    check_outliers: bool = True,
    max_missing_pct: float = 0.05,
    outlier_std_threshold: float = 3.0,
    param_name: str = "data"
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Validate data quality for DataFrames and datasets.

    Args:
        check_duplicates: Whether to check for duplicate rows
        check_missing_values: Whether to check for missing values
        check_outliers: Whether to check for outliers
        max_missing_pct: Maximum allowed percentage of missing values
        outlier_std_threshold: Standard deviation threshold for outlier detection
        param_name: Parameter name containing the data to validate

    Example:
        @validate_data_quality(max_missing_pct=0.1)
        def process_data(df: pd.DataFrame) -> pd.DataFrame:
            return df.dropna()
    """
    if not PANDAS_AVAILABLE:
        msg = "pandas is required for data quality validation"
        raise ImportError(msg)

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        # Get the data parameter
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        if param_name not in bound.arguments:
            msg = f"Parameter {param_name} not found"
            raise ValidationError(message=msg)

        data = bound.arguments[param_name]

        if isinstance(data, pd.DataFrame):
            _validate_dataframe_quality(data, param_name, check_duplicates,
                                      check_missing_values, check_outliers,
                                      max_missing_pct, outlier_std_threshold)
        elif isinstance(data, (list, tuple)):
            _validate_list_quality(data, param_name, check_duplicates,
                                 check_missing_values, check_outliers,
                                 max_missing_pct, outlier_std_threshold)
        else:
            # For other data types, just check for None
            if data is None:
                msg = f"Parameter {param_name} cannot be None"
                raise ValidationError(message=msg, field=param_name)

        return func(*args, **kwargs)

    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        # Reuse sync validation logic
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        if param_name not in bound.arguments:
            msg = f"Parameter {param_name} not found"
            raise ValidationError(message=msg)

        data = bound.arguments[param_name]

        if isinstance(data, pd.DataFrame):
            _validate_dataframe_quality(data, param_name, check_duplicates,
                                      check_missing_values, check_outliers,
                                      max_missing_pct, outlier_std_threshold)
        elif isinstance(data, (list, tuple)):
            _validate_list_quality(data, param_name, check_duplicates,
                                 check_missing_values, check_outliers,
                                 max_missing_pct, outlier_std_threshold)
        else:
            if data is None:
                msg = f"Parameter {param_name} cannot be None"
                raise ValidationError(message=msg, field=param_name)

        return await func(*args, **kwargs)

    return uniform_wrapper("validate_data_quality", sync_handler, async_handler)


def monitor_step_execution(
    *,
    step_name: str = None,
    log_level: str = "INFO",
    track_metrics: bool = True,
    alert_on_failure: bool = True
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Monitor execution of pipeline steps with logging and metrics.

    Args:
        step_name: Name of the step (defaults to function name)
        log_level: Logging level for execution tracking
        track_metrics: Whether to track performance metrics
        alert_on_failure: Whether to alert on step failures

    Example:
        @monitor_step_execution(step_name="data_processing")
        def process_data(df: pd.DataFrame) -> pd.DataFrame:
            return df.transform(...)
    """
    import time

    logger = logging.getLogger(__name__)

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.time()
        actual_step_name = step_name or func.__name__

        try:
            logger.log(getattr(logging, log_level),
                      f"🚀 Starting step: {actual_step_name}")

            result = func(*args, **kwargs)

            duration = time.time() - start_time
            logger.log(getattr(logging, log_level),
                      f"✅ Completed step: {actual_step_name} in {duration:.2f}s")

            if track_metrics:
                # Could integrate with metrics system here
                logger.debug(f"Step metrics: duration={duration:.2f}s")

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Failed step: {actual_step_name} after {duration:.2f}s: {e}")

            if alert_on_failure:
                # Could integrate with alerting system here
                logger.warning(f"⚠️ Alert: Step {actual_step_name} failed")

            raise

    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.time()
        actual_step_name = step_name or func.__name__

        try:
            logger.log(getattr(logging, log_level),
                      f"🚀 Starting async step: {actual_step_name}")

            result = await func(*args, **kwargs)

            duration = time.time() - start_time
            logger.log(getattr(logging, log_level),
                      f"✅ Completed async step: {actual_step_name} in {duration:.2f}s")

            if track_metrics:
                logger.debug(f"Async step metrics: duration={duration:.2f}s")

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Failed async step: {actual_step_name} after {duration:.2f}s: {e}")

            if alert_on_failure:
                logger.warning(f"⚠️ Alert: Async step {actual_step_name} failed")

            raise

    return uniform_wrapper("monitor_step_execution", sync_handler, async_handler)


def ensure_data_integrity(
    *,
    check_types: bool = True,
    validate_ranges: bool = False,
    param_name: str = "data"
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Ensure data integrity by validating types and ranges.

    Args:
        check_types: Whether to validate data types
        validate_ranges: Whether to validate value ranges
        param_name: Parameter name containing the data

    Example:
        @ensure_data_integrity(validate_ranges=True)
        def process_prices(prices: List[float]) -> float:
            return sum(prices) / len(prices)
    """
    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        if param_name not in bound.arguments:
            msg = f"Parameter {param_name} not found"
            raise ValidationError(message=msg)

        data = bound.arguments[param_name]

        if check_types:
            _validate_data_types(data, param_name)

        if validate_ranges:
            _validate_data_ranges(data, param_name)

        return func(*args, **kwargs)

    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        if param_name not in bound.arguments:
            msg = f"Parameter {param_name} not found"
            raise ValidationError(message=msg)

        data = bound.arguments[param_name]

        if check_types:
            _validate_data_types(data, param_name)

        if validate_ranges:
            _validate_data_ranges(data, param_name)

        return await func(*args, **kwargs)

    return uniform_wrapper("ensure_data_integrity", sync_handler, async_handler)


def validate_pipeline_step(
    *,
    required_inputs: list[str] = None,
    required_outputs: list[str] = None,
    validate_inputs: bool = True,
    validate_outputs: bool = True
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Validate pipeline step inputs and outputs.

    Args:
        required_inputs: List of required input parameters
        required_outputs: List of required output attributes/keys
        validate_inputs: Whether to validate input parameters
        validate_outputs: Whether to validate output structure

    Example:
        @validate_pipeline_step(required_inputs=["data", "config"])
        def process_step(data: pd.DataFrame, config: dict) -> dict:
            return {"processed_data": data, "metrics": {...}}
    """
    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        sig = inspect.signature(func)

        # Validate inputs
        if validate_inputs and required_inputs:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            missing_inputs = []
            for req_input in required_inputs:
                if req_input not in bound.arguments:
                    missing_inputs.append(req_input)
                elif bound.arguments[req_input] is None:
                    missing_inputs.append(req_input)

            if missing_inputs:
                msg = f"Missing required inputs: {missing_inputs}"
                raise ValidationError(message=msg, details={"missing_inputs": missing_inputs})

        # Execute function
        result = func(*args, **kwargs)

        # Validate outputs
        if validate_outputs and required_outputs:
            if isinstance(result, dict):
                missing_outputs = [out for out in required_outputs if out not in result]
                if missing_outputs:
                    msg = f"Missing required outputs: {missing_outputs}"
                    raise ValidationError(message=msg, details={"missing_outputs": missing_outputs})
            else:
                logger.warning(f"Cannot validate outputs for non-dict result in {func.__name__}")

        return result

    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        sig = inspect.signature(func)

        # Validate inputs
        if validate_inputs and required_inputs:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            missing_inputs = []
            for req_input in required_inputs:
                if req_input not in bound.arguments:
                    missing_inputs.append(req_input)
                elif bound.arguments[req_input] is None:
                    missing_inputs.append(req_input)

            if missing_inputs:
                msg = f"Missing required inputs: {missing_inputs}"
                raise ValidationError(message=msg, details={"missing_inputs": missing_inputs})

        # Execute function
        result = await func(*args, **kwargs)

        # Validate outputs
        if validate_outputs and required_outputs:
            if isinstance(result, dict):
                missing_outputs = [out for out in required_outputs if out not in result]
                if missing_outputs:
                    msg = f"Missing required outputs: {missing_outputs}"
                    raise ValidationError(message=msg, details={"missing_outputs": missing_outputs})
            else:
                logger.warning(f"Cannot validate outputs for non-dict result in {func.__name__}")

        return result

    return uniform_wrapper("validate_pipeline_step", sync_handler, async_handler)


# Helper functions for data quality validation

def _validate_dataframe_quality(df: pd.DataFrame, param_name: str,
                               check_duplicates: bool, check_missing: bool,
                               check_outliers: bool, max_missing_pct: float,
                               outlier_std_threshold: float) -> None:
    """Validate DataFrame data quality."""
    if check_duplicates and df.duplicated().any():
        msg = f"DataFrame contains duplicate rows"
        raise ValidationError(message=msg, field=param_name)

    if check_missing:
        missing_pct = df.isnull().mean().mean()
        if missing_pct > max_missing_pct:
            msg = f"Too many missing values: {missing_pct:.1%} > {max_missing_pct:.1%}"
            raise ValidationError(message=msg, field=param_name,
                                details={"missing_percentage": missing_pct})

    if check_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].std() > 0:  # Only check if column has variance
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_pct = (z_scores > outlier_std_threshold).mean()
                if outlier_pct > 0.05:  # More than 5% outliers
                    msg = f"Column {col} has too many outliers: {outlier_pct:.1%}"
                    raise ValidationError(message=msg, field=f"{param_name}.{col}")


def _validate_list_quality(data: list, param_name: str,
                          check_duplicates: bool, check_missing: bool,
                          check_outliers: bool, max_missing_pct: float,
                          outlier_std_threshold: float) -> None:
    """Validate list/array data quality."""
    if not data:
        return  # Empty list is valid

    # Check for duplicates
    if check_duplicates and len(data) != len(set(data)):
        msg = f"List contains duplicate values"
        raise ValidationError(message=msg, field=param_name)

    # Check for missing values (None)
    if check_missing:
        none_count = sum(1 for x in data if x is None)
        none_pct = none_count / len(data)
        if none_pct > max_missing_pct:
            msg = f"Too many None values: {none_pct:.1%} > {max_missing_pct:.1%}"
            raise ValidationError(message=msg, field=param_name,
                                details={"none_percentage": none_pct})

    # Check for outliers in numeric data
    if check_outliers:
        numeric_data = [x for x in data if isinstance(x, (int, float)) and x is not None]
        if len(numeric_data) > 10:  # Only check if we have enough data
            arr = np.array(numeric_data)
            if arr.std() > 0:
                z_scores = np.abs((arr - arr.mean()) / arr.std())
                outlier_pct = (z_scores > outlier_std_threshold).mean()
                if outlier_pct > 0.05:  # More than 5% outliers
                    msg = f"List has too many outliers: {outlier_pct:.1%}"
                    raise ValidationError(message=msg, field=param_name)


def _validate_data_types(data: Any, param_name: str) -> None:
    """Validate basic data types."""
    if data is None:
        msg = f"Parameter {param_name} cannot be None"
        raise ValidationError(message=msg, field=param_name)


def _validate_data_ranges(data: Any, param_name: str) -> None:
    """Validate data value ranges (basic implementation)."""
    if isinstance(data, (list, tuple)):
        # Check for infinite or NaN values in numeric data
        for i, item in enumerate(data):
            if isinstance(item, (int, float)):
                if np.isnan(item) or np.isinf(item):
                    msg = f"Invalid value at index {i}: {item}"
                    raise ValidationError(message=msg, field=f"{param_name}[{i}]")
