"""
Examples of using the core decorator system.

This file demonstrates various decorator usage patterns and
best practices for the centralized decorator system.
"""

import asyncio
import time
from typing import Dict, List

# Import decorators
from src.core.decorators import (
import numpy as np
    CachePolicy,
    authenticated,
    cached,
    circuit_breaker,
    compose,
    handles_errors,
    import,
    log_call,
)
from src.core.decorators import numpy as np
from src.core.decorators import (
    requires_role,
    retry,
    timeout,
    traced,
    validate_schema,
    validates,
)

# Import errors
from src.core.errors import NotFoundError, ValidationError, register_exception_mapping


# Example 1: Simple validation and error handling
@validates(strict=True)
@handles_errors(ValueError, TypeError, map_to=ValidationError)
def calculate_price(base_price: float, tax_rate: float) -> float:
    """Calculate price with tax."""
    if base_price < 0:
        raise ValueError("Base price cannot be negative")
    return base_price * (1 + tax_rate)


# Example 2: Composed decorators for data processing
@compose(
    log_call(level="INFO", mask_sensitive=True),
    validates(),
    handles_errors(fallback={"error": "processing failed"}),
    cached(policy=CachePolicy.PER_REQUEST),
)
def process_user_data(user_id: str, data: dict) -> dict:
    """Process user data with multiple decorators."""
    # Simulate processing
    return {
        "user_id": user_id,
        "processed": True,
        "timestamp": time.time(),
        **data,
    }


# Example 3: Resilient external API call
@retry(max_attempts=3, delay=1.0, backoff=2.0)
@circuit_breaker(failure_threshold=5, recovery_timeout=60)
@timeout(30.0)
@traced(span_name="external_api_call", record_result=True)
async def fetch_external_data(api_endpoint: str) -> dict:
    """Fetch data from external API with resilience patterns."""
    # Simulate API call
    await asyncio.sleep(0.1)
    
    # Simulate occasional failures
    import random
    if random.random() < 0.1:
        raise ConnectionError("API unavailable")
    
    return {"endpoint": api_endpoint, "data": "external_data"}


# Example 4: Authenticated and authorized endpoint
@authenticated()
@requires_role("admin", "moderator", require_all=False)
@log_call(level="INFO")
@handles_errors(propagate=True)
def delete_content(content_id: str) -> bool:
    """Delete content - requires authentication and proper role."""
    # In a real app, this would delete from database
    print(f"Deleting content {content_id}")
    return True


# Example 5: Complex data validation with custom schema
class UserCreateSchema:
    """Simple schema for user creation."""
    def __init__(self, username: str, email: str, age: int):
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        if "@" not in email:
            raise ValueError("Invalid email format")
        if age < 18:
            raise ValueError("Must be 18 or older")
        
        self.username = username
        self.email = email
        self.age = age


@validate_schema(UserCreateSchema)
@cached(policy=CachePolicy.CROSS_REQUEST, ttl=300)
@traced(span_name="create_user")
def create_user(user_data: dict) -> dict:
    """Create user with schema validation."""
    # Schema validation happens automatically
    return {
        "id": "user_123",
        "created_at": time.time(),
        **user_data,
    }


# Example 6: DataFrame processing with validation
try:
    import pandas as pd
    
    @validate_dataframe(
        columns=["id", "value", "category"],
        dtypes={"id": int, "value": float},
        min_rows=1,
    )
    @handles_errors(pd.errors.EmptyDataError, fallback=pd.DataFrame())
    @cached(policy=CachePolicy.PER_REQUEST)
    def analyze_data(df: pd.DataFrame) -> dict:
        """Analyze DataFrame with validation."""
        return {
            "row_count": len(df),
            "mean_value": df["value"].mean(),
            "categories": df["category"].unique().tolist(),
        }
except ImportError:
    # pandas not available
    pass


# Example 7: Custom error mapping
class CustomBusinessError(Exception):
    """Custom business logic error."""
    pass


# Register custom error mapping
register_exception_mapping(
    CustomBusinessError,
    lambda e: ValidationError(f"Business rule violation: {e}")
)


@handles_errors(CustomBusinessError)
def business_operation(value: int) -> int:
    """Operation with custom error handling."""
    if value > 100:
        raise CustomBusinessError("Value exceeds maximum allowed")
    return value * 2


# Example 8: Async with tracing and caching
@traced(kind="client", attributes={"service": "database"})
@cached(policy=CachePolicy.CROSS_REQUEST, ttl=60)
@retry(max_attempts=2)
async def get_user_from_db(user_id: str) -> dict:
    """Get user from database with caching and tracing."""
    # Simulate database query
    await asyncio.sleep(0.05)
    
    # Add trace events
    from src.core.decorators import span_attribute, span_event
    span_event("query_started", {"user_id": user_id})
    
    result = {"id": user_id, "name": f"User {user_id}"}
    
    span_attribute("result_size", len(str(result)))
    span_event("query_completed")
    
    return result


# Example 9: Method decorator on a class
from src.core.decorators import trace_method


@trace_method(span_prefix="UserService")
class UserService:
    """Service class with automatic method tracing."""
    
    @cached(policy=CachePolicy.PER_REQUEST)
    def get_user(self, user_id: str) -> dict:
        """Get user by ID."""
        return {"id": user_id, "service": "UserService"}
    
    @requires_permission("users.update")
    @validates()
    def update_user(self, user_id: str, data: dict) -> dict:
        """Update user data."""
        return {"id": user_id, "updated": True, **data}


# Example 10: Decorator stacking order matters
@authenticated()  # First: Check authentication
@validates()      # Second: Validate inputs
@cached(policy=CachePolicy.PER_REQUEST)  # Third: Check cache
@traced()         # Fourth: Create trace span
@handles_errors(fallback=None)  # Fifth: Handle errors
def complex_operation(user_id: str, action: str) -> dict:
    """
    Demonstrate decorator stacking order.
    
    Execution order (top to bottom):
    1. Authentication check
    2. Input validation
    3. Cache lookup (return if hit)
    4. Start trace span
    5. Execute function (with error handling)
    6. Cache result
    7. End trace span
    """
    return {
        "user_id": user_id,
        "action": action,
        "result": "success",
    }


# Example usage functions
async def main():
    """Run examples."""
    print("Core Decorator System Examples\n")
    
    # Example 1: Basic usage
    try:
        price = calculate_price(100.0, 0.08)
        print(f"1. Calculated price: ${price:.2f}")
    except ValidationError as e:
        print(f"1. Validation error: {e}")
    
    # Example 2: Composed decorators
    result = process_user_data("user123", {"name": "John", "age": 30})
    print(f"2. Processed user data: {result}")
    
    # Example 3: Async with resilience
    try:
        data = await fetch_external_data("https://api.example.com/data")
        print(f"3. External data: {data}")
    except Exception as e:
        print(f"3. Failed to fetch external data: {e}")
    
    # Example 4: Authentication (would need actual user context)
    # delete_content("content123")
    
    # Example 5: Schema validation
    try:
        user = create_user({
            "username": "johndoe",
            "email": "john@example.com",
            "age": 25,
        })
        print(f"5. Created user: {user}")
    except ValidationError as e:
        print(f"5. Validation failed: {e}")
    
    # Example 7: Custom error
    try:
        result = business_operation(50)
        print(f"7. Business operation result: {result}")
    except Exception as e:
        print(f"7. Business error: {e}")
    
    # Example 8: Async database
    user = await get_user_from_db("user456")
    print(f"8. User from DB: {user}")
    
    # Example 9: Class methods
    service = UserService()
    user = service.get_user("user789")
    print(f"9. User from service: {user}")
    
    # Show cache stats
    from src.core.decorators import cache_stats
    stats = cache_stats()
    print(f"\nCache statistics: {stats}")
    
    # Show trace summary (if any traces were created)
    from src.core.decorators import get_current_trace
    trace = get_current_trace()
    if trace:
        from src.core.decorators import get_trace_summary
        summary = get_trace_summary(trace.trace_id)
        print(f"\nTrace summary: {summary}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())