#!/usr/bin/env python3
"""
Market Regime Analysis for ETHUSDT Data

This script analyzes the collected ETHUSDT data to identify different market regimes/clusters
based on various market characteristics including volatility, volume, price movements, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the data for analysis."""
    print("📊 Loading and preparing data...")
    
    # Load aggtrades data
    df = pd.read_parquet('data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet')
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    # Sort by datetime
    df = df.sort_index()
    
    print(f"✅ Loaded {len(df):,} records from {df.index.min()} to {df.index.max()}")
    return df

def calculate_market_features(df, window_size=100):
    """Calculate market features for regime analysis."""
    print("🔍 Calculating market features...")
    
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['price'] = df['price']
    features['price_change'] = df['price'].pct_change()
    features['price_volatility'] = df['price'].rolling(window=window_size).std()
    features['price_range'] = df['price'].rolling(window=window_size).max() - df['price'].rolling(window=window_size).min()
    
    # Volume-based features
    features['volume'] = df['quantity']
    features['volume_ma'] = df['quantity'].rolling(window=window_size).mean()
    features['volume_ratio'] = df['quantity'] / (features['volume_ma'] + 1e-8)
    
    # Trade frequency features
    features['trade_frequency'] = df.groupby(df.index.floor('1min')).size().reindex(df.index, method='ffill')
    
    # Buy/sell pressure - aggregate by time window
    df_resampled = df.resample('1min').agg({
        'quantity': 'sum',
        'is_buyer_maker': lambda x: (x == False).sum() / len(x) if len(x) > 0 else 0.5
    })
    
    features['buy_sell_ratio'] = df_resampled['is_buyer_maker'].reindex(df.index, method='ffill')
    
    # Price momentum
    features['momentum_5'] = df['price'].pct_change(5)
    features['momentum_10'] = df['price'].pct_change(10)
    features['momentum_20'] = df['price'].pct_change(20)
    
    # Technical indicators
    features['rsi'] = calculate_rsi(df['price'], window=14)
    features['bollinger_upper'] = df['price'].rolling(window=20).mean() + 2 * df['price'].rolling(window=20).std()
    features['bollinger_lower'] = df['price'].rolling(window=20).mean() - 2 * df['price'].rolling(window=20).std()
    features['bollinger_position'] = (df['price'] - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower'] + 1e-8)
    
    # Fill NaN values with forward fill and then backward fill
    features = features.ffill().bfill()
    
    # Remove any remaining rows with NaN values
    features = features.dropna()
    
    print(f"✅ Calculated {len(features.columns)} features for {len(features):,} data points")
    return features

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def identify_regimes(features, n_clusters=4):
    """Identify market regimes using clustering."""
    print(f"🎯 Identifying {n_clusters} market regimes...")
    
    # Select features for clustering
    clustering_features = [
        'price_volatility', 'volume_ratio', 'buy_sell_ratio', 
        'momentum_5', 'momentum_10', 'rsi', 'bollinger_position'
    ]
    
    # Prepare data for clustering
    X = features[clustering_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features['regime'] = kmeans.fit_predict(X_scaled)
    
    # Calculate regime characteristics
    regime_stats = features.groupby('regime')[clustering_features].agg(['mean', 'std'])
    
    print("✅ Regime identification completed")
    return features, regime_stats, scaler, kmeans

def analyze_regime_characteristics(features):
    """Analyze characteristics of each regime."""
    print("📈 Analyzing regime characteristics...")
    
    regime_analysis = {}
    
    for regime in sorted(features['regime'].unique()):
        regime_data = features[features['regime'] == regime]
        
        analysis = {
            'count': len(regime_data),
            'percentage': len(regime_data) / len(features) * 100,
            'avg_price': regime_data['price'].mean(),
            'avg_volatility': regime_data['price_volatility'].mean(),
            'avg_volume_ratio': regime_data['volume_ratio'].mean(),
            'avg_buy_sell_ratio': regime_data['buy_sell_ratio'].mean(),
            'avg_momentum_5': regime_data['momentum_5'].mean(),
            'avg_rsi': regime_data['rsi'].mean(),
            'price_range': regime_data['price'].max() - regime_data['price'].min(),
            'duration_hours': (regime_data.index.max() - regime_data.index.min()).total_seconds() / 3600
        }
        
        # Determine regime type based on characteristics
        if analysis['avg_volatility'] > features['price_volatility'].quantile(0.75):
            if analysis['avg_momentum_5'] > 0:
                regime_type = "High Volatility Bull"
            else:
                regime_type = "High Volatility Bear"
        elif analysis['avg_volatility'] < features['price_volatility'].quantile(0.25):
            regime_type = "Low Volatility Consolidation"
        else:
            if analysis['avg_momentum_5'] > 0:
                regime_type = "Moderate Bull"
            else:
                regime_type = "Moderate Bear"
        
        analysis['regime_type'] = regime_type
        regime_analysis[regime] = analysis
    
    print("✅ Regime characteristics analysis completed")
    return regime_analysis

def create_regime_report(regime_analysis, features, timestamp):
    """Create a comprehensive regime report."""
    print("📝 Creating regime report...")
    
    report = f"""
# ETHUSDT Market Regime Analysis Report
Generated: {timestamp}

## Executive Summary
This report analyzes {len(features):,} data points from ETHUSDT trading data to identify {len(regime_analysis)} distinct market regimes.

## Data Overview
- **Total Data Points**: {len(features):,}
- **Time Period**: {features.index.min()} to {features.index.max()}
- **Price Range**: ${features['price'].min():.2f} - ${features['price'].max():.2f}
- **Average Price**: ${features['price'].mean():.2f}
- **Total Volatility**: {features['price_volatility'].mean():.4f}

## Market Regimes Identified

"""
    
    for regime_id, analysis in regime_analysis.items():
        report += f"""
### Regime {regime_id}: {analysis['regime_type']}
- **Data Points**: {analysis['count']:,} ({analysis['percentage']:.1f}% of total)
- **Average Price**: ${analysis['avg_price']:.2f}
- **Volatility**: {analysis['avg_volatility']:.4f}
- **Volume Ratio**: {analysis['avg_volume_ratio']:.2f}
- **Buy/Sell Ratio**: {analysis['avg_buy_sell_ratio']:.2f}
- **5-period Momentum**: {analysis['avg_momentum_5']:.4f}
- **RSI**: {analysis['avg_rsi']:.1f}
- **Price Range**: ${analysis['price_range']:.2f}
- **Duration**: {analysis['duration_hours']:.1f} hours

**Characteristics**: {analysis['regime_type']}

"""
    
    report += f"""
## Key Insights

### Volatility Analysis
- **Highest Volatility Regime**: Regime {max(regime_analysis.keys(), key=lambda x: regime_analysis[x]['avg_volatility'])} ({max(regime_analysis.values(), key=lambda x: x['avg_volatility'])['regime_type']})
- **Lowest Volatility Regime**: Regime {min(regime_analysis.keys(), key=lambda x: regime_analysis[x]['avg_volatility'])} ({min(regime_analysis.values(), key=lambda x: x['avg_volatility'])['regime_type']})

### Volume Analysis
- **Highest Volume Regime**: Regime {max(regime_analysis.keys(), key=lambda x: regime_analysis[x]['avg_volume_ratio'])} ({max(regime_analysis.values(), key=lambda x: x['avg_volume_ratio'])['regime_type']})
- **Lowest Volume Regime**: Regime {min(regime_analysis.keys(), key=lambda x: regime_analysis[x]['avg_volume_ratio'])} ({min(regime_analysis.values(), key=lambda x: x['avg_volume_ratio'])['regime_type']})

### Trading Behavior
- **Most Bullish Regime**: Regime {max(regime_analysis.keys(), key=lambda x: regime_analysis[x]['avg_momentum_5'])} ({max(regime_analysis.values(), key=lambda x: x['avg_momentum_5'])['regime_type']})
- **Most Bearish Regime**: Regime {min(regime_analysis.keys(), key=lambda x: regime_analysis[x]['avg_momentum_5'])} ({min(regime_analysis.values(), key=lambda x: x['avg_momentum_5'])['regime_type']})

## Methodology
- **Clustering Algorithm**: K-Means with {len(regime_analysis)} clusters
- **Features Used**: Price volatility, volume ratio, buy/sell ratio, momentum indicators, RSI, Bollinger position
- **Window Size**: 100 periods for rolling calculations
- **Data Preprocessing**: Standardization and NaN handling

## Recommendations
1. **High Volatility Periods**: Monitor for potential breakout opportunities
2. **Low Volatility Periods**: Consider range-bound trading strategies
3. **Volume Analysis**: Use volume ratios to confirm price movements
4. **Regime Transitions**: Watch for regime changes as potential trading signals

---
*Report generated by Ares Trading System Market Regime Analysis*
"""
    
    return report

def main():
    """Main analysis function."""
    print("🚀 Starting ETHUSDT Market Regime Analysis")
    print("=" * 60)
    
    # Get current timestamp for report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Calculate market features
        features = calculate_market_features(df)
        
        # Identify regimes
        features, regime_stats, scaler, kmeans = identify_regimes(features, n_clusters=4)
        
        # Analyze regime characteristics
        regime_analysis = analyze_regime_characteristics(features)
        
        # Create comprehensive report
        report = create_regime_report(regime_analysis, features, timestamp)
        
        # Save report to file
        report_filename = f"ethusdt_regime_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"✅ Analysis completed successfully!")
        print(f"📄 Report saved to: {report_filename}")
        print(f"📊 Identified {len(regime_analysis)} market regimes")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📈 REGIME SUMMARY")
        print("=" * 60)
        for regime_id, analysis in regime_analysis.items():
            print(f"Regime {regime_id}: {analysis['regime_type']} ({analysis['percentage']:.1f}%)")
        
        return regime_analysis, features
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    regime_analysis, features = main()