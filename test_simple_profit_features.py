#!/usr/bin/env python3
"""Simple test for profit-based feature engineering."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_simple_test_data(n_samples: int = 100) -> pd.DataFrame:
    """Create simple test data."""
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="1min")

    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, n_samples),
        'high': np.random.uniform(105, 115, n_samples),
        'low': np.random.uniform(95, 105, n_samples),
        'close': np.random.uniform(100, 110, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples),
        'potential_profit_pct': np.random.uniform(-0.01, 0.01, n_samples),
    }, index=dates)

    return data

def test_basic_features():
    """Test basic profit features."""
    print("🧪 Testing Basic Profit Features")
    print("=" * 40)

    # Create test data
    test_data = create_simple_test_data(100)
    print(f"Created {len(test_data)} data points")

    # Test basic profit features manually
    profit_pcts = test_data['potential_profit_pct'].values

    # Basic profit features
    test_data['potential_profit_pct_squared'] = profit_pcts ** 2
    test_data['potential_profit_pct_cubed'] = profit_pcts ** 3
    test_data['potential_profit_pct_abs'] = np.abs(profit_pcts)
    test_data['potential_profit_pct_sqrt'] = np.sqrt(np.abs(profit_pcts))
    test_data['potential_profit_pct_log'] = np.log1p(np.abs(profit_pcts))

    print("✅ Basic features created successfully")
    print(f"Features: {[col for col in test_data.columns if 'potential_profit_pct' in col]}")

    return test_data

def test_categorical_features():
    """Test categorical profit features."""
    print("\n🧪 Testing Categorical Profit Features")
    print("=" * 40)

    # Create test data
    test_data = create_simple_test_data(100)
    print(f"Created {len(test_data)} data points")

    profit_pcts = test_data['potential_profit_pct'].values

    # Categorical features
    test_data['potential_profit_pct_sign'] = np.sign(profit_pcts)

    profit_abs = np.abs(profit_pcts)
    test_data['potential_profit_pct_magnitude'] = pd.cut(
        profit_abs,
        bins=[0, 0.001, 0.002, 0.005, np.inf],
        labels=["Tiny", "Small", "Medium", "Large"],
        include_lowest=True
    )

    profit_bins = [-np.inf, -0.005, -0.002, -0.001, 0, 0.001, 0.002, 0.005, np.inf]
    profit_labels = [
        "Large Loss", "Medium Loss", "Small Loss", "Tiny Loss",
        "No Profit", "Tiny Profit", "Small Profit", "Large Profit"
    ]
    test_data['potential_profit_pct_bins'] = pd.cut(
        profit_pcts,
        bins=profit_bins,
        labels=profit_labels,
        include_lowest=True
    )

    test_data['potential_profit_pct_direction_strength'] = profit_abs * np.sign(profit_pcts)

    print("✅ Categorical features created successfully")
    print(f"Features: {[col for col in test_data.columns if 'potential_profit_pct' in col]}")

    return test_data

def test_risk_reward_features():
    """Test risk-reward features."""
    print("\n🧪 Testing Risk-Reward Features")
    print("=" * 40)

    # Create test data
    test_data = create_simple_test_data(100)
    print(f"Created {len(test_data)} data points")

    profit_pcts = test_data['potential_profit_pct'].values

    # Calculate rolling statistics
    window = 20
    rolling_mean = test_data['potential_profit_pct'].rolling(window=window, min_periods=1).mean()
    rolling_std = test_data['potential_profit_pct'].rolling(window=window, min_periods=1).std()

    # Sharpe ratio
    test_data['potential_profit_pct_sharpe'] = np.where(
        rolling_std > 0,
        rolling_mean / rolling_std,
        0.0
    )

    # Sortino ratio
    downside_returns = np.where(profit_pcts < 0, profit_pcts, 0)
    downside_std = pd.Series(downside_returns).rolling(window=window, min_periods=1).std()

    # Ensure both are pandas Series with same index
    downside_std = downside_std.reindex(rolling_mean.index)

    sortino_ratio = np.where(
        downside_std > 0,
        rolling_mean / downside_std,
        0.0
    )
    test_data['potential_profit_pct_sortino'] = pd.Series(sortino_ratio, index=test_data.index).fillna(0.0)

    # Kelly criterion
    profit_series = pd.Series(profit_pcts, index=test_data.index)
    win_rate = (profit_series > 0).rolling(window=window, min_periods=1).mean()
    avg_win = np.where(profit_pcts > 0, profit_pcts, 0)
    avg_win_series = pd.Series(avg_win, index=test_data.index).rolling(window=window, min_periods=1).mean()
    avg_loss = np.where(profit_pcts < 0, np.abs(profit_pcts), 0)
    avg_loss_series = pd.Series(avg_loss, index=test_data.index).rolling(window=window, min_periods=1).mean()

    kelly_ratio = np.where(
        avg_loss_series > 0,
        (win_rate * avg_win_series - (1 - win_rate) * avg_loss_series) / avg_win_series,
        0.0
    )
    test_data['potential_profit_pct_kelly'] = pd.Series(kelly_ratio, index=test_data.index).fillna(0.0)

    # Risk-adjusted return
    test_data['potential_profit_pct_risk_adjusted'] = profit_pcts / (1 + rolling_std)

    print("✅ Risk-reward features created successfully")
    print(f"Features: {[col for col in test_data.columns if 'potential_profit_pct' in col]}")

    return test_data

if __name__ == "__main__":
    # Test each feature category separately
    basic_data = test_basic_features()
    cat_data = test_categorical_features()
    risk_data = test_risk_reward_features()

    print("\n🎉 All tests completed successfully!")
    print(f"Basic features: {len([col for col in basic_data.columns if 'potential_profit_pct' in col])}")
    print(f"Categorical features: {len([col for col in cat_data.columns if 'potential_profit_pct' in col])}")
    print(f"Risk-reward features: {len([col for col in risk_data.columns if 'potential_profit_pct' in col])}")