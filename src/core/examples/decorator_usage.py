from src.utils.tprint import tprint

from ..core.decorators import handles_errors
"""
from src.core.errors.base import ValidationError
Examples of using the core decorator system.

This file demonstrates various decorator usage patterns and
best practices for the centralized decorator system.
"""
import asyncio
import time
from src.utils.decorators import CachePolicy, authenticated, cached, circuit_breaker, compose, handles_errors, log_call, requires_role, retry, timeout, traced, validate_schema, validates
from src.core.errors import ValidationError, register_exception_mapping

import pandas as pd
import random
import logging
import numpy as np

@validates(strict = True)
@handles_errors(ValueError, TypeError, map_to = ValidationError)
def calculate_price(base_price: float, tax_rate: float) -> float:
    """Calculate price with tax."""
    if base_price < 0:
        msg = 'Base price cannot be negative'
        raise ValueError(msg)
    return base_price * (1 + tax_rate)

@compose(log_call(level='INFO', mask_sensitive = True), validates(), handles_errors(fallback={'error': 'processing failed'}), cached(policy = CachePolicy.PER_REQUEST))
def process_user_data(user_id: str, data: dict) -> dict:
    """Process user data with multiple decorators."""
    return {'user_id': user_id, 'processed': True, 'timestamp': time.time(), **data}

@retry(max_attempts = 3, delay = 1.0, backoff = 2.0)
@circuit_breaker(failure_threshold = 5, recovery_timeout = 60)
@timeout(30.0)
@traced(span_name='external_api_call', record_result = True)
async def fetch_external_data(api_endpoint: str) -> dict:
    """Fetch data from external API with resilience patterns."""
    await asyncio.sleep(0.1)
    if random.random() < 0.1:
        msg = 'API unavailable'
        raise ConnectionError(msg)
    return {'endpoint': api_endpoint, 'data': 'external_data'}

@authenticated()
@requires_role('admin', 'moderator', require_all = False)
@log_call(level='INFO')
@handles_errors(propagate = True)
def delete_content(content_id: str) -> bool:
    """Delete content - requires authentication and proper role."""
    tprint(f'Deleting content {content_id}')
    return True

class UserCreateSchema:
    """Simple schema for user creation."""

    def __init__(self, username: str, email: str, age: int) -> None:
        if not username or len(username) < 3:
            msg = 'Username must be at least 3 characters'
            raise ValueError(msg)
        if '@' not in email:
            msg = 'Invalid email format'
            raise ValueError(msg)
        if age < 18:
            msg = 'Must be 18 or older'
            raise ValueError(msg)
        self.username = username
        self.email = email
        self.age = age

@validate_schema(UserCreateSchema)
@cached(policy = CachePolicy.CROSS_REQUEST, ttl = 300)
@traced(span_name='create_user')
def create_user(user_data: dict) -> dict:
    """Create user with schema validation."""
    return {'id': 'user_123', 'created_at': time.time(), **user_data}
try:

    @validate_dataframe(columns=['id', 'value', 'category'], dtypes={'id': int, 'value': float}, min_rows = 1)
    @handles_errors(pd.errors.EmptyDataError, fallback = pd.DataFrame())
    @cached(policy = CachePolicy.PER_REQUEST)
    def analyze_data(df: pd.DataFrame) -> dict:
        """Analyze DataFrame with validation."""
        return {'row_count': len(df), 'mean_value': df['value'].mean(), 'categories': df['category'].unique().tolist()}
except ImportError:
    pass

class CustomBusinessError(Exception):
    """Custom business logic error."""
register_exception_mapping(CustomBusinessError, lambda e: ValidationError(f'Business rule violation: {e}'))

@handles_errors(CustomBusinessError)
def business_operation(value: int) -> int:
    """Operation with custom error handling."""
    if value > 100:
        msg = 'Value exceeds maximum allowed'
        raise CustomBusinessError(msg)
    return value * 2

@traced(kind='client', attributes={'service': 'database'})
@cached(policy = CachePolicy.CROSS_REQUEST, ttl = 60)
@retry(max_attempts = 2)
async def get_user_from_db(user_id: str) -> dict:
    """Get user from database with caching and tracing."""
    await asyncio.sleep(0.05)
    span_event('query_started', {'user_id': user_id})
    result = {'id': user_id, 'name': f'User {user_id}'}
    span_attribute('result_size', len(str(result)))
    span_event('query_completed')
    return result

@trace_method(span_prefix='UserService')
class UserService:
    """Service class with automatic method tracing."""

    @cached(policy = CachePolicy.PER_REQUEST)
    def get_user(self, user_id: str) -> dict:
        """Get user by ID."""
        return {'id': user_id, 'service': 'UserService'}

    @requires_permission('users.update')
    @validates()
    def update_user(self, user_id: str, data: dict) -> dict:
        """Update user data."""
        return {'id': user_id, 'updated': True, **data}

@authenticated()
@validates()
@cached(policy = CachePolicy.PER_REQUEST)
@traced()
@handles_errors(fallback = None)
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
    return {'user_id': user_id, 'action': action, 'result': 'success'}

async def main() -> None:
    """Run examples."""
    tprint('Core Decorator System Examples\n')
    try:
        price = calculate_price(100.0, 0.08)
        tprint(f'1. Calculated price: ${price:.2f}')
    except ValidationError as e:
        tprint(f'1. Validation error: {e}')
    result = process_user_data('user123', {'name': 'John', 'age': 30})
    tprint(f'2. Processed user data: {result}')
    try:
        data = await fetch_external_data('https://api.example.com/data')
        tprint(f'3. External data: {data}')
    except Exception as e:
        tprint(f'3. Failed to fetch external data: {e}')
    try:
        user = create_user({'username': 'johndoe', 'email': 'john@example.com', 'age': 25})
        tprint(f'5. Created user: {user}')
    except ValidationError as e:
        tprint(f'5. Validation failed: {e}')
    try:
        result = business_operation(50)
        tprint(f'7. Business operation result: {result}')
    except Exception as e:
        tprint(f'7. Business error: {e}')
    user = await get_user_from_db('user456')
    tprint(f'8. User from DB: {user}')
    service = UserService()
    user = service.get_user('user789')
    tprint(f'9. User from service: {user}')
    stats = cache_stats()
    tprint(f'\nCache statistics: {stats}')
    trace = get_current_trace()
    if trace:
        pass
        summary = get_trace_summary(trace.trace_id)
        tprint(f'\nTrace summary: {summary}')
if __name__ == '__main__':
    asyncio.run(main())