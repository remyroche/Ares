from __future__ import annotations

"""
Request and DTO validation decorators.

Provides decorators for validating function inputs using various
validation strategies (pydantic, dataclasses, custom validators).
"""

import inspect
from typing import get_type_hints

from .compose import P, R, uniform_wrapper

# Try to import optional validation libraries
try:
    import pydantic

    PYDANTIC_AVAILABLE = True
except ImportError:
    pydantic = None
    PYDANTIC_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


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
        @validates(strict=True)
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
            raise ValidationError(msg)

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
                        strict=strict,
                        coerce=coerce,
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
                msg,
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
            raise ValidationError(msg)

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
                        strict=strict,
                        coerce=coerce,
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
                msg,
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
                raise ValidationError(msg)
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
                raise ValidationError(msg)
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
            min_rows=1
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
            raise ValidationError(msg)

        df = bound.arguments[param_name]

        if not isinstance(df, pd.DataFrame):
            msg = f"Parameter {param_name} must be a pandas DataFrame"
            raise ValidationError(
                msg,
                field=param_name,
                value=type(df).__name__,
            )

        # Validate columns
        if columns:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                msg = f"Missing required columns: {missing_cols}"
                raise ValidationError(
                    msg,
                    field=param_name,
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
                            msg,
                            field=f"{param_name}.{col}",
                        )
                    if expected_type == float and actual_type.kind not in "iuf":
                        msg = f"Column {col} has wrong dtype: expected float-like, got {actual_type}"
                        raise ValidationError(
                            msg,
                            field=f"{param_name}.{col}",
                        )

        # Validate row count
        row_count = len(df)
        if row_count < min_rows:
            msg = f"DataFrame has too few rows: {row_count} < {min_rows}"
            raise ValidationError(
                msg,
                field=param_name,
                details={"row_count": row_count, "min_rows": min_rows},
            )

        if max_rows is not None and row_count > max_rows:
            msg = f"DataFrame has too many rows: {row_count} > {max_rows}"
            raise ValidationError(
                msg,
                field=param_name,
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
            raise ValidationError(msg)

        df = bound.arguments[param_name]

        # Run same validations as sync version
        if not isinstance(df, pd.DataFrame):
            msg = f"Parameter {param_name} must be a pandas DataFrame"
            raise ValidationError(
                msg,
                field=param_name,
                value=type(df).__name__,
            )

        # Validate columns
        if columns:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                msg = f"Missing required columns: {missing_cols}"
                raise ValidationError(
                    msg,
                    field=param_name,
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
                            msg,
                            field=f"{param_name}.{col}",
                        )
                    if expected_type == float and actual_type.kind not in "iuf":
                        msg = f"Column {col} has wrong dtype: expected float-like, got {actual_type}"
                        raise ValidationError(
                            msg,
                            field=f"{param_name}.{col}",
                        )

        # Validate row count
        row_count = len(df)
        if row_count < min_rows:
            msg = f"DataFrame has too few rows: {row_count} < {min_rows}"
            raise ValidationError(
                msg,
                field=param_name,
                details={"row_count": row_count, "min_rows": min_rows},
            )

        if max_rows is not None and row_count > max_rows:
            msg = f"DataFrame has too many rows: {row_count} > {max_rows}"
            raise ValidationError(
                msg,
                field=param_name,
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
                    msg,
                    field=name,
                )
        except pydantic.ValidationError as e:
            msg = f"Validation failed for {name}"
            raise ValidationError(
                msg,
                field=name,
                details={"pydantic_errors": e.errors()},
            )

    # Basic type checking
    elif strict and not isinstance(value, expected_type):
        msg = f"Expected {expected_type.__name__}, got {type(value).__name__}"
        raise ValidationError(
            msg,
            field=name,
            value=value,
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
                msg,
                field=param_name,
            )
        except pydantic.ValidationError as e:
            msg = "Schema validation failed"
            raise ValidationError(
                msg,
                field=param_name,
                details={"errors": e.errors()},
            )

    # Handle dict schemas
    elif isinstance(schema, dict):
        if not isinstance(value, dict):
            msg = f"Expected dict for {param_name}, got {type(value).__name__}"
            raise ValidationError(
                msg,
                field=param_name,
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
                msg,
                field=param_name,
                details={"errors": errors},
            )

        return validated

    # Basic type validation
    else:
        if not isinstance(value, schema):
            msg = f"Expected {schema.__name__}, got {type(value).__name__}"
            raise ValidationError(
                msg,
                field=param_name,
            )
        return value
