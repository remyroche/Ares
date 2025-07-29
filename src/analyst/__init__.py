# src/analyst/__init__.py
# This file makes the 'analyst' directory a Python package.

# You can import sub-modules here for easier access, e.g.:
# from .analyst import Analyst
# from .feature_engineering import FeatureEngineeringEngine
# from .regime_classifier import MarketRegimeClassifier
# from .predictive_ensembles import RegimePredictiveEnsembles
# from .liquidation_risk_model import ProbabilisticLiquidationRiskModel
# from .market_health_analyzer import GeneralMarketAnalystModule
# from .specialized_models import SpecializedModels

# src/analyst/data_utils.py
import pandas as pd
import os

def load_klines_data(filename):
    """Loads k-line data from a CSV file."""
    if not os.path.exists(filename):
        print(f"Error: K-lines data file not found at {filename}")
        return pd.DataFrame()
    df = pd.read_csv(filename, index_col='open_time', parse_dates=True)
    df = df[~df.index.duplicated(keep='first')] # Remove duplicates
    # Ensure numeric columns are actually numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_agg_trades_data(filename):
    """Loads aggregated trades data from a CSV file."""
    if not os.path.exists(filename):
        print(f"Error: Agg trades data file not found at {filename}")
        return pd.DataFrame()
    df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
    df = df[~df.index.duplicated(keep='first')] # Remove duplicates
    numeric_cols = ['price', 'quantity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_futures_data(filename):
    """Loads futures data (funding rates, open interest) from a CSV file."""
    if not os.path.exists(filename):
        print(f"Error: Futures data file not found at {filename}")
        return pd.DataFrame()
    df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
    df = df[~df.index.duplicated(keep='first')] # Remove duplicates
    numeric_cols = ['fundingRate', 'openInterest']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def simulate_order_book_data(current_price):
    """Simulates real-time order book data for demonstration."""
    simulated_bids = [
        [current_price - 0.1, 5],
        [current_price - 0.2, 10],
        [current_price - 0.5, 20],
        [current_price - 1.0, 100],
        [current_price - 2.0, 50000 / current_price], # Large buy wall (approx $50k)
        [current_price - 2.5, 15],
        [current_price - 3.0, 120000 / current_price] # Even larger buy wall (approx $120k)
    ]
    simulated_asks = [
        [current_price + 0.1, 7],
        [current_price + 0.2, 12],
        [current_price + 0.5, 25],
        [current_price + 1.0, 80],
        [current_price + 2.0, 60000 / current_price], # Large sell wall (approx $60k)
        [current_price + 2.5, 18],
        [current_price + 3.0, 110000 / current_price] # Even larger sell wall (approx $110k)
    ]
    return {"bids": simulated_bids, "asks": simulated_asks}

