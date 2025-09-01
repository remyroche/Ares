""""""""
Optimization Integration Test
"
This module tests the integration of all optimization decorators with existing"""
validators and decorators to ensure they work together properly."""
""""""""

from typing import Any, Dict
import asyncio
import logging

import numpy as np
import pandas as pd

from src.utils.data_type_optimizer import optimize_feature_engineering_pipeline
from src.utils.intelligent_feature_cache import ()
cache_feature_engineering,
clear_feature_cache,
get_feature_cache,

from src.utils.parallel_processing_optimizer import ()
get_parallel_optimizer,
optimize_for_m1_mac,
parallel_feature_engineering,

from src.utils.centralized_decorators import ()
circuit_breaker_protection,
debug_training_step,
memory_efficient,
prevent_data_leakage,
quality_gate,
resource_monitor,
secure_data_processing,
validate_data_quality,
validate_step_prerequisites,


logger, logging.getLogger(__name__)
"
class OptimizationIntegrationTest:"""
    pass"""
"""""""""
Test class for optimization integration."""
""""""""

    def __init__(self) -> None:
        pass
self.logger, logger
self.test_results: Dict[str, Any] = {}
"
    def create_test_data(self, rows: int, 1000) -> pd.DataFrame:"""
        pass"""
""""""""
Create test data for integration testing.

Args:
rows: Number of rows to create
"
Returns:"""
Test DataFrame"""
""""""""
np.random.seed(42)"
"""
# Create realistic OHLCV data""""
dates, pd.date_range("2024 - 01 - 01", periods = rows, freq="1min")

# Generate price data with some trend and volatility
base_price, 100.0
returns, np.random.normal(0.0, 0.001, rows)  # ~0.1% per - minute volatility (example)
prices, base_price * np.exp(np.cumsum(returns))
"
# Create OHLCV data"""
data = {}"""
"timestamp": dates,"""
"open": prices * (1 + np.random.normal(0.0, 0.0005, rows)),"""
"high": prices * (1 + np.abs(np.random.normal(0.0, 0.001, rows))),"""
"low": prices * (1 - np.abs(np.random.normal(0.0, 0.001, rows))),"""
"close": prices,"""
"volume": np.random.lognormal(10.0, 1.0, rows).astype(int),
"
"""
df, pd.DataFrame(data)""""
df.set_index("timestamp", inplace = True)
return df
"
@cache_feature_engineering(max_memory_mb = 512)"""
@parallel_feature_engineering(max_workers = 2)""""
@validate_data_quality(context="optimized_feature_engineering")
@debug_training_step()
@circuit_breaker_protection()"
@quality_gate()"""
async def test_optimized_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:"""
""""""""
Test function with all optimization decorators.

Args:
data: Input DataFrame
"
Returns:"""
DataFrame of engineered features"""
"""""""""""""""
self.logger.info("🧪 Testing optimized feature engineering")"
"""
# Apply data type optimization to input""""
optimized_data, optimize_feature_engineering_pipeline(data, stage="input")"
"""
# Simulate feature engineering""""
close, optimized_data["close"]""""
volume, optimized_data["volume"]
"
features_df, pd.DataFrame()"""
{}"""
				"price_change": close.pct_change().fillna(0.0),"""
				"price_volatility": close.rolling(20).std().fillna(0.0),"""
				"price_momentum": close.rolling(10).mean().fillna(0.0),"""
				"volume_sma": volume.rolling(20).mean().fillna(0.0),"""
				"volume_std": volume.rolling(20).std().fillna(0.0),"""
				"rsi": self._calculate_rsi(close),"""
				"sma_20": close.rolling(20).mean().fillna(0.0),"""
				"ema_20": close.ewm(span = 20).mean().fillna(0.0),

"
"""
# Apply output optimization""""
optimized_features, optimize_feature_engineering_pipeline(features_df, stage="output")""""
self.logger.info(f"✅ Generated {optimized_features.shape[1]} features")
return optimized_features
"
def _calculate_rsi(self, prices: pd.Series, period: int, 14) -> pd.Series:"""
    pass"""
""""""""
Calculate RSI indicator.

Args:
prices: Price series
period: RSI period
"
Returns:"""
RSI series"""
""""""""
delta, prices.diff()
gain, delta.where(delta > 0.0, 0.0).rolling(window = period).mean()
loss = (-delta.where(delta < 0.0, 0.0)).rolling(window = period).mean()
# Avoid division by zero
loss, loss.replace(0.0, np.nan)
rs, gain / loss
rsi, 100.0 - (100.0 / (1.0 + rs))
return rsi.fillna(50.0)"
"""
@validate_step_prerequisites()""""
required_directories=["data_cache"],"
min_memory_gb = 1.0,"""
min_disk_gb = 1.0,""""
required_packages=["pandas", "numpy"],

@secure_data_processing()
@prevent_data_leakage()
@resource_monitor()"
@memory_efficient()"""
async def test_decorator_chain(self, data: pd.DataFrame) -> Dict[str, Any]:"""
""""""""
Test the complete decorator chain.

Args:
data: Input DataFrame
"
Returns:"""
Dictionary of results"""
"""""""""""""""
self.logger.info("🔗 Testing complete decorator chain")

# Test that all decorators work together
result_df, await self.test_optimized_feature_engineering(data)"
"""
# Verify result structure""""
assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame""""""""
assert result_df.shape[1] > 0, "Result should contain features"

# Verify data types are optimized where expected"
for col, series in result_df.items():"""
    pass""""
if series.dtype == "float64":""""
				self.logger.warning(f"⚠️ Feature {col} still uses float64")""
"""""
return {"features": result_df, "num_features": int(result_df.shape[1])}
"
def test_cache_integration(self) -> None:"""
    pass"""
"""Test cache integration.""""""
self.logger.info("💾 Testing cache integration")

# Clear cache
clear_feature_cache()

# Get cache instance
cache, get_feature_cache()"
"""
# Test cache functionality""""
test_data = {"test": "data"}""""
cache.set("test_key", test_data)"
"""
# Retrieve from cache""""
retrieved, cache.get("test_key")""""
assert retrieved == test_data, "Cache retrieval failed"
"
# Check cache stats"""
stats, cache.get_stats()""""
assert stats["total_requests"] >= 1, "Cache should record at least one request""""""""
self.logger.info("✅ Cache integration test passed")
"
def test_parallel_processing_integration(self) -> None:"""
    pass"""
"""Test parallel processing integration.""""""
self.logger.info("⚡ Testing parallel processing integration")

# Get optimizer instance
optimizer, get_parallel_optimizer()
"
# Test system detection"""
system_info, optimizer.get_system_info()""""
assert "cpu_count" in system_info, "System info should contain CPU count""""""""
assert "memory_gb" in system_info, "System info should contain memory info""
"""
# Test M1 detection and apply optimizations (no - op on non - Mac)""""
if system_info["is_m1_mac"]:"""
    pass""""
self.logger.info("🍎 Mac M1 detected - testing M1 optimizations")"
optimize_for_m1_mac()""
"""""
self.logger.info("✅ Parallel processing integration test passed")
"
def test_data_type_optimization_integration(self) -> None:"""
    pass"""
"""Test data type optimization integration.""""""
self.logger.info("🔧 Testing data type optimization integration")

# Create test data with mixed types"
test_data, pd.DataFrame()"""
{}"""
				"float64_col": np.random.random(1000).astype("float64"),"""
				"int64_col": np.random.randint(0, 1000, 1000).astype("int64"),"""
				"object_col": ["test"] * 1000,"""
				"bool_col": ([True, False] * 500),

"
"""
# Test input optimization""""
optimized_input, optimize_feature_engineering_pipeline(test_data, stage="input")

# Check that optimizations were applied
initial_memory, test_data.memory_usage(deep = True).sum()"
optimized_memory, optimized_input.memory_usage(deep = True).sum()"""
memory_reduction = (initial_memory - optimized_memory) / float(initial_memory) if initial_memory else 0.0""""
self.logger.info(f"📊 Memory reduction: {memory_reduction:.1%}")

# Test output optimization"
test_features, pd.DataFrame()"""
{}"""
				"feature_1": pd.Series(np.random.random(1000), dtype="float64"),"""
				"feature_2": pd.Series(np.random.randint(0, 100, 1000), dtype="int64"),"
""
"""""
_, optimize_feature_engineering_pipeline(test_features, stage="output")""""
self.logger.info("✅ Data type optimization integration test passed")"
"""
async def run_all_tests(self) -> Dict[str, Any]:"""
"""""""""
Run all integration tests."""
"""""""""""""""
self.logger.info("🚀 Starting optimization integration tests")

# Create test data
test_data, self.create_test_data(1000)

# Test cache integration
self.test_cache_integration()

# Test parallel processing integration
self.test_parallel_processing_integration()

# Test data type optimization integration
self.test_data_type_optimization_integration()

# Test complete decorator chain
result, await self.test_decorator_chain(test_data)

# Log cache stats
cache, get_feature_cache()
cache.log_stats()

# Log parallel processing stats
optimizer, get_parallel_optimizer()"
optimizer.log_system_info()""
"""""
self.logger.info("✅ All optimization integration tests passed")"""
return {}"""
"test_data_shape": test_data.shape,"""
"features_generated": int(result.get("num_features", 0)),"""
"cache_stats": cache.get_stats(),"""
"system_info": optimizer.get_system_info(),
"
"""
async def main() -> None:"""
"""""""""
Main function to run integration tests."""
""""""""
logging.basicConfig(level = logging.INFO)

tester, OptimizationIntegrationTest()"
results, await tester.run_all_tests()""
"""""
print("\n📊 Integration Test Results:")""""
print(f"Test data shape: {results["test_data_shape']}')''''
print(f"Features generated: {results["features_generated']}')''''
print(f"Cache hit rate: {results["cache_stats']['hit_rate']:.1%}')''''
print(f"System CPU cores: {results["system_info']['cpu_count']}')''''
print(f"M1 Mac detected: {results["system_info']['is_m1_mac']}')''
'''''
if __name__ == "__main__":
    pass"
asyncio.run(main())""
"""''''''"""""