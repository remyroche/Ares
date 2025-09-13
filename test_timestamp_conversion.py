#!/usr/bin/env python3
"""
Test script to verify timestamp conversion fix
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_timestamp_conversion():
    """Test the fixed timestamp conversion logic."""
    print("🧪 Testing timestamp conversion logic...")

    # Test the specific problematic case: datetime64 conversion
    print("\n1️⃣ Testing datetime64[ns] conversion (the main issue):")
    dt_series_ns = pd.Series(pd.to_datetime(['2023-01-01 00:00:00', '2023-01-02 00:00:00']))
    print(f"   Series dtype: {dt_series_ns.dtype}")

    # Test the OLD problematic method
    print("   🔴 Testing OLD method (dt_series.view('int64')):")
    try:
        old_result = dt_series_ns.view('int64') // 10**9
        print(f"   ✅ Old method worked: {old_result.tolist()}")
    except Exception as e:
        print(f"   ❌ Old method failed: {e}")

    # Test the NEW fixed method
    print("   🟢 Testing NEW method (dt_series.astype('int64')):")
    try:
        new_result = dt_series_ns.astype('int64') // 10**9
        print(f"   ✅ New method worked: {new_result.tolist()}")
    except Exception as e:
        print(f"   ❌ New method failed: {e}")

    # Test string to datetime conversion
    print("\n2️⃣ Testing string datetime conversion:")
    str_series = pd.Series(['2023-01-01 00:00:00', '2023-01-02 00:00:00'])
    print(f"   String series: {str_series.tolist()}")

    try:
        dt_from_str = pd.to_datetime(str_series)
        print(f"   Converted to datetime: {dt_from_str.dtype}")
        result = dt_from_str.astype('int64') // 10**9
        print(f"   ✅ Converted to timestamps: {result.tolist()}")
    except Exception as e:
        print(f"   ❌ String conversion failed: {e}")

    # Test edge case: already integer timestamps
    print("\n3️⃣ Testing integer timestamps (already converted):")
    int_series = pd.Series([1672531200, 1672617600])
    print(f"   Integer series: {int_series.tolist()}")
    print("   ✅ No conversion needed for integer timestamps")

    print("\n✅ Timestamp conversion test completed!")
    print("🎯 The fix should resolve the 'numpy.datetime64 object cannot be interpreted as an integer' error")

if __name__ == "__main__":
    test_timestamp_conversion()
