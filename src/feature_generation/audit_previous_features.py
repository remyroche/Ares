"""
Audit Previous Features Script

This script audits all the features that were previously available in the system
to ensure we haven't lost any functionality during the refactoring.
"""

import sys
from pathlib import Path
import re
from typing import Dict, List, Set, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def audit_previous_features():
    """Audit all previous features to ensure we have complete coverage."""
    
    print("🔍 AUDITING PREVIOUS FEATURES")
    print("=" * 60)
    
    # Dictionary to store all found features
    all_features = {
        'technical_indicators': set(),
        'cross_timeframe_features': set(),
        'interaction_features': set(),
        'regime_features': set(),
        'microstructure_features': set(),
        'entropy_features': set(),
        'autoencoder_features': set(),
        'legacy_features': set(),
        'time_features': set(),
        'order_flow_features': set(),
        'sr_features': set(),
        'polynomial_features': set(),
        'correlation_features': set(),
        'momentum_features': set(),
        'volatility_features': set(),
        'volume_features': set(),
        'trend_features': set(),
        'oscillator_features': set(),
        'candlestick_features': set()
    }
    
    # 1. Technical Indicators from feature_generators.py
    print("\n1. 📊 Technical Indicators from feature_generators.py")
    print("-" * 50)
    
    technical_indicators = [
        # Moving Averages
        'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'KAMA', 'T3', 'TRIMA', 'MAMA', 'VWMA',
        
        # Momentum Indicators
        'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HISTOGRAM', 'STOCH', 'STOCHF', 'STOCHRSI',
        'WILLR', 'CCI', 'CMO', 'ROC', 'MOM', 'PPO', 'APO', 'AROON', 'AROONOSC',
        'BOP', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'ADX', 'ADXR',
        'MFI', 'ULTOSC', 'TRIX', 'TSF', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE',
        
        # Volatility Indicators
        'ATR', 'NATR', 'TRANGE', 'BBANDS', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
        'KELTNER_UPPER', 'KELTNER_MIDDLE', 'KELTNER_LOWER', 'DONCHIAN_UPPER', 'DONCHIAN_MIDDLE', 'DONCHIAN_LOWER',
        'SAR', 'SAREXT', 'ADX', 'ADXR', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM',
        
        # Volume Indicators
        'OBV', 'AD', 'ADOSC', 'MFI', 'VWAP', 'VWMA', 'VPT', 'NVI', 'PVI',
        
        # Price Transform
        'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
        
        # Cycle Indicators
        'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE',
        
        # Pattern Recognition
        'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH',
        'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY',
        'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
        'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
        'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI',
        'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON',
        'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
        'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW',
        'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING',
        'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR',
        'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI',
        'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
        'CDLXSIDEGAP3METHODS',
        
        # Math Transform
        'ACOS', 'ASIN', 'ATAN', 'CEIL', 'COS', 'COSH', 'EXP', 'FLOOR', 'LN', 'LOG10', 'SIN', 'SINH', 'SQRT', 'TAN', 'TANH',
        
        # Math Operators
        'ADD', 'DIV', 'MAX', 'MAXINDEX', 'MIN', 'MININDEX', 'MINMAX', 'MINMAXINDEX', 'MULT', 'SUB', 'SUM',
        
        # Statistical Functions
        'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE',
        'STDDEV', 'TSF', 'VAR',
        
        # Price Transform
        'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
        
        # Volatility Indicators
        'ATR', 'NATR', 'TRANGE',
        
        # Volume Indicators
        'AD', 'ADOSC', 'OBV'
    ]
    
    for indicator in technical_indicators:
        all_features['technical_indicators'].add(indicator)
        print(f"  • {indicator}")
    
    # 2. Cross-timeframe Features
    print(f"\n2. ⏰ Cross-timeframe Features")
    print("-" * 50)
    
    cross_timeframe_features = [
        # RSI cross-timeframe
        'RSI_1m', 'RSI_5m', 'RSI_15m', 'RSI_30m', 'RSI_1h', 'RSI_4h', 'RSI_1d',
        'RSI_ratio_1m_5m', 'RSI_ratio_5m_15m', 'RSI_ratio_15m_30m', 'RSI_ratio_30m_1h',
        'RSI_ratio_1h_4h', 'RSI_ratio_4h_1d', 'RSI_diff_1m_5m', 'RSI_diff_5m_15m',
        'RSI_diff_15m_30m', 'RSI_diff_30m_1h', 'RSI_diff_1h_4h', 'RSI_diff_4h_1d',
        
        # MACD cross-timeframe
        'MACD_1m', 'MACD_5m', 'MACD_15m', 'MACD_30m', 'MACD_1h', 'MACD_4h', 'MACD_1d',
        'MACD_signal_1m', 'MACD_signal_5m', 'MACD_signal_15m', 'MACD_signal_30m',
        'MACD_signal_1h', 'MACD_signal_4h', 'MACD_signal_1d',
        'MACD_histogram_1m', 'MACD_histogram_5m', 'MACD_histogram_15m', 'MACD_histogram_30m',
        'MACD_histogram_1h', 'MACD_histogram_4h', 'MACD_histogram_1d',
        'MACD_ratio_1m_5m', 'MACD_ratio_5m_15m', 'MACD_ratio_15m_30m', 'MACD_ratio_30m_1h',
        'MACD_ratio_1h_4h', 'MACD_ratio_4h_1d',
        
        # Bollinger Bands cross-timeframe
        'BB_upper_1m', 'BB_upper_5m', 'BB_upper_15m', 'BB_upper_30m', 'BB_upper_1h', 'BB_upper_4h', 'BB_upper_1d',
        'BB_middle_1m', 'BB_middle_5m', 'BB_middle_15m', 'BB_middle_30m', 'BB_middle_1h', 'BB_middle_4h', 'BB_middle_1d',
        'BB_lower_1m', 'BB_lower_5m', 'BB_lower_15m', 'BB_lower_30m', 'BB_lower_1h', 'BB_lower_4h', 'BB_lower_1d',
        'BB_width_1m', 'BB_width_5m', 'BB_width_15m', 'BB_width_30m', 'BB_width_1h', 'BB_width_4h', 'BB_width_1d',
        'BB_position_1m', 'BB_position_5m', 'BB_position_15m', 'BB_position_30m', 'BB_position_1h', 'BB_position_4h', 'BB_position_1d',
        
        # SMA cross-timeframe
        'SMA_5_1m', 'SMA_10_1m', 'SMA_20_1m', 'SMA_50_1m', 'SMA_100_1m', 'SMA_200_1m',
        'SMA_5_5m', 'SMA_10_5m', 'SMA_20_5m', 'SMA_50_5m', 'SMA_100_5m', 'SMA_200_5m',
        'SMA_5_15m', 'SMA_10_15m', 'SMA_20_15m', 'SMA_50_15m', 'SMA_100_15m', 'SMA_200_15m',
        'SMA_5_30m', 'SMA_10_30m', 'SMA_20_30m', 'SMA_50_30m', 'SMA_100_30m', 'SMA_200_30m',
        'SMA_5_1h', 'SMA_10_1h', 'SMA_20_1h', 'SMA_50_1h', 'SMA_100_1h', 'SMA_200_1h',
        'SMA_5_4h', 'SMA_10_4h', 'SMA_20_4h', 'SMA_50_4h', 'SMA_100_4h', 'SMA_200_4h',
        'SMA_5_1d', 'SMA_10_1d', 'SMA_20_1d', 'SMA_50_1d', 'SMA_100_1d', 'SMA_200_1d',
        
        # EMA cross-timeframe
        'EMA_8_1m', 'EMA_12_1m', 'EMA_21_1m', 'EMA_26_1m', 'EMA_50_1m', 'EMA_100_1m',
        'EMA_8_5m', 'EMA_12_5m', 'EMA_21_5m', 'EMA_26_5m', 'EMA_50_5m', 'EMA_100_5m',
        'EMA_8_15m', 'EMA_12_15m', 'EMA_21_15m', 'EMA_26_15m', 'EMA_50_15m', 'EMA_100_15m',
        'EMA_8_30m', 'EMA_12_30m', 'EMA_21_30m', 'EMA_26_30m', 'EMA_50_30m', 'EMA_100_30m',
        'EMA_8_1h', 'EMA_12_1h', 'EMA_21_1h', 'EMA_26_1h', 'EMA_50_1h', 'EMA_100_1h',
        'EMA_8_4h', 'EMA_12_4h', 'EMA_21_4h', 'EMA_26_4h', 'EMA_50_4h', 'EMA_100_4h',
        'EMA_8_1d', 'EMA_12_1d', 'EMA_21_1d', 'EMA_26_1d', 'EMA_50_1d', 'EMA_100_1d',
        
        # ATR cross-timeframe
        'ATR_7_1m', 'ATR_14_1m', 'ATR_21_1m', 'ATR_30_1m',
        'ATR_7_5m', 'ATR_14_5m', 'ATR_21_5m', 'ATR_30_5m',
        'ATR_7_15m', 'ATR_14_15m', 'ATR_21_15m', 'ATR_30_15m',
        'ATR_7_30m', 'ATR_14_30m', 'ATR_21_30m', 'ATR_30_30m',
        'ATR_7_1h', 'ATR_14_1h', 'ATR_21_1h', 'ATR_30_1h',
        'ATR_7_4h', 'ATR_14_4h', 'ATR_21_4h', 'ATR_30_4h',
        'ATR_7_1d', 'ATR_14_1d', 'ATR_21_1d', 'ATR_30_1d',
        
        # Stochastic cross-timeframe
        'STOCH_K_1m', 'STOCH_K_5m', 'STOCH_K_15m', 'STOCH_K_30m', 'STOCH_K_1h', 'STOCH_K_4h', 'STOCH_K_1d',
        'STOCH_D_1m', 'STOCH_D_5m', 'STOCH_D_15m', 'STOCH_D_30m', 'STOCH_D_1h', 'STOCH_D_4h', 'STOCH_D_1d',
        
        # Volume cross-timeframe
        'VOLUME_1m', 'VOLUME_5m', 'VOLUME_15m', 'VOLUME_30m', 'VOLUME_1h', 'VOLUME_4h', 'VOLUME_1d',
        'VOLUME_MA_5_1m', 'VOLUME_MA_10_1m', 'VOLUME_MA_20_1m', 'VOLUME_MA_50_1m',
        'VOLUME_MA_5_5m', 'VOLUME_MA_10_5m', 'VOLUME_MA_20_5m', 'VOLUME_MA_50_5m',
        'VOLUME_MA_5_15m', 'VOLUME_MA_10_15m', 'VOLUME_MA_20_15m', 'VOLUME_MA_50_15m',
        'VOLUME_MA_5_30m', 'VOLUME_MA_10_30m', 'VOLUME_MA_20_30m', 'VOLUME_MA_50_30m',
        'VOLUME_MA_5_1h', 'VOLUME_MA_10_1h', 'VOLUME_MA_20_1h', 'VOLUME_MA_50_1h',
        'VOLUME_MA_5_4h', 'VOLUME_MA_10_4h', 'VOLUME_MA_20_4h', 'VOLUME_MA_50_4h',
        'VOLUME_MA_5_1d', 'VOLUME_MA_10_1d', 'VOLUME_MA_20_1d', 'VOLUME_MA_50_1d',
        
        # OBV cross-timeframe
        'OBV_1m', 'OBV_5m', 'OBV_15m', 'OBV_30m', 'OBV_1h', 'OBV_4h', 'OBV_1d',
        
        # VWAP cross-timeframe
        'VWAP_1m', 'VWAP_5m', 'VWAP_15m', 'VWAP_30m', 'VWAP_1h', 'VWAP_4h', 'VWAP_1d'
    ]
    
    for feature in cross_timeframe_features:
        all_features['cross_timeframe_features'].add(feature)
        print(f"  • {feature}")
    
    # 3. Interaction Features
    print(f"\n3. 🔗 Interaction Features")
    print("-" * 50)
    
    interaction_features = [
        # Polynomial features
        'RSI_squared', 'RSI_cubed', 'MACD_squared', 'MACD_cubed',
        'BB_position_squared', 'BB_position_cubed', 'ATR_squared', 'ATR_cubed',
        'SMA_20_squared', 'SMA_20_cubed', 'EMA_21_squared', 'EMA_21_cubed',
        'VOLUME_squared', 'VOLUME_cubed', 'OBV_squared', 'OBV_cubed',
        
        # Ratio features
        'RSI_MACD_ratio', 'RSI_BB_position_ratio', 'RSI_ATR_ratio', 'RSI_SMA_20_ratio',
        'MACD_BB_position_ratio', 'MACD_ATR_ratio', 'MACD_SMA_20_ratio',
        'BB_position_ATR_ratio', 'BB_position_SMA_20_ratio', 'ATR_SMA_20_ratio',
        'VOLUME_OBV_ratio', 'VOLUME_SMA_20_ratio', 'OBV_SMA_20_ratio',
        
        # Difference features
        'RSI_MACD_diff', 'RSI_BB_position_diff', 'RSI_ATR_diff', 'RSI_SMA_20_diff',
        'MACD_BB_position_diff', 'MACD_ATR_diff', 'MACD_SMA_20_diff',
        'BB_position_ATR_diff', 'BB_position_SMA_20_diff', 'ATR_SMA_20_diff',
        'VOLUME_OBV_diff', 'VOLUME_SMA_20_diff', 'OBV_SMA_20_diff',
        
        # Product features
        'RSI_MACD_product', 'RSI_BB_position_product', 'RSI_ATR_product', 'RSI_SMA_20_product',
        'MACD_BB_position_product', 'MACD_ATR_product', 'MACD_SMA_20_product',
        'BB_position_ATR_product', 'BB_position_SMA_20_product', 'ATR_SMA_20_product',
        'VOLUME_OBV_product', 'VOLUME_SMA_20_product', 'OBV_SMA_20_product',
        
        # Cross-timeframe ratios
        'RSI_1m_5m_ratio', 'RSI_5m_15m_ratio', 'RSI_15m_30m_ratio', 'RSI_30m_1h_ratio',
        'MACD_1m_5m_ratio', 'MACD_5m_15m_ratio', 'MACD_15m_30m_ratio', 'MACD_30m_1h_ratio',
        'BB_position_1m_5m_ratio', 'BB_position_5m_15m_ratio', 'BB_position_15m_30m_ratio', 'BB_position_30m_1h_ratio',
        'ATR_1m_5m_ratio', 'ATR_5m_15m_ratio', 'ATR_15m_30m_ratio', 'ATR_30m_1h_ratio',
        'SMA_20_1m_5m_ratio', 'SMA_20_5m_15m_ratio', 'SMA_20_15m_30m_ratio', 'SMA_20_30m_1h_ratio',
        'VOLUME_1m_5m_ratio', 'VOLUME_5m_15m_ratio', 'VOLUME_15m_30m_ratio', 'VOLUME_30m_1h_ratio',
        
        # Cross-timeframe differences
        'RSI_1m_5m_diff', 'RSI_5m_15m_diff', 'RSI_15m_30m_diff', 'RSI_30m_1h_diff',
        'MACD_1m_5m_diff', 'MACD_5m_15m_diff', 'MACD_15m_30m_diff', 'MACD_30m_1h_diff',
        'BB_position_1m_5m_diff', 'BB_position_5m_15m_diff', 'BB_position_15m_30m_diff', 'BB_position_30m_1h_diff',
        'ATR_1m_5m_diff', 'ATR_5m_15m_diff', 'ATR_15m_30m_diff', 'ATR_30m_1h_diff',
        'SMA_20_1m_5m_diff', 'SMA_20_5m_15m_diff', 'SMA_20_15m_30m_diff', 'SMA_20_30m_1h_diff',
        'VOLUME_1m_5m_diff', 'VOLUME_5m_15m_diff', 'VOLUME_15m_30m_diff', 'VOLUME_30m_1h_diff',
        
        # Cross-timeframe products
        'RSI_1m_5m_product', 'RSI_5m_15m_product', 'RSI_15m_30m_product', 'RSI_30m_1h_product',
        'MACD_1m_5m_product', 'MACD_5m_15m_product', 'MACD_15m_30m_product', 'MACD_30m_1h_product',
        'BB_position_1m_5m_product', 'BB_position_5m_15m_product', 'BB_position_15m_30m_product', 'BB_position_30m_1h_product',
        'ATR_1m_5m_product', 'ATR_5m_15m_product', 'ATR_15m_30m_product', 'ATR_30m_1h_product',
        'SMA_20_1m_5m_product', 'SMA_20_5m_15m_product', 'SMA_20_15m_30m_product', 'SMA_20_30m_1h_product',
        'VOLUME_1m_5m_product', 'VOLUME_5m_15m_product', 'VOLUME_15m_30m_product', 'VOLUME_30m_1h_product'
    ]
    
    for feature in interaction_features:
        all_features['interaction_features'].add(feature)
        print(f"  • {feature}")
    
    # 4. Regime Features
    print(f"\n4. 🔄 Regime Features")
    print("-" * 50)
    
    regime_features = [
        'regime_label', 'regime_probability', 'regime_transition_probability',
        'regime_duration', 'regime_stability', 'regime_volatility',
        'regime_momentum', 'regime_trend', 'regime_volume',
        'regime_0_count', 'regime_1_count', 'regime_2_count', 'regime_3_count',
        'regime_changed', 'time_in_regime', 'regime_entropy',
        'regime_0_probability', 'regime_1_probability', 'regime_2_probability', 'regime_3_probability',
        'regime_0_volatility', 'regime_1_volatility', 'regime_2_volatility', 'regime_3_volatility',
        'regime_0_momentum', 'regime_1_momentum', 'regime_2_momentum', 'regime_3_momentum',
        'regime_0_trend', 'regime_1_trend', 'regime_2_trend', 'regime_3_trend',
        'regime_0_volume', 'regime_1_volume', 'regime_2_volume', 'regime_3_volume'
    ]
    
    for feature in regime_features:
        all_features['regime_features'].add(feature)
        print(f"  • {feature}")
    
    # 5. Microstructure Features
    print(f"\n5. 📈 Microstructure Features")
    print("-" * 50)
    
    microstructure_features = [
        'bid_ask_spread', 'bid_ask_spread_ratio', 'bid_ask_spread_ma',
        'order_flow_imbalance', 'order_flow_imbalance_ma', 'order_flow_imbalance_std',
        'trade_size_imbalance', 'trade_size_imbalance_ma', 'trade_size_imbalance_std',
        'price_impact', 'price_impact_ma', 'price_impact_std',
        'volume_weighted_price', 'volume_weighted_price_ma', 'volume_weighted_price_std',
        'trade_intensity', 'trade_intensity_ma', 'trade_intensity_std',
        'liquidity_proxy', 'liquidity_proxy_ma', 'liquidity_proxy_std',
        'market_depth', 'market_depth_ma', 'market_depth_std',
        'tick_direction', 'tick_direction_ma', 'tick_direction_std',
        'tick_volatility', 'tick_volatility_ma', 'tick_volatility_std',
        'tick_momentum', 'tick_momentum_ma', 'tick_momentum_std',
        'tick_volume', 'tick_volume_ma', 'tick_volume_std',
        'tick_price', 'tick_price_ma', 'tick_price_std',
        'tick_time', 'tick_time_ma', 'tick_time_std',
        'tick_frequency', 'tick_frequency_ma', 'tick_frequency_std',
        'tick_aggression', 'tick_aggression_ma', 'tick_aggression_std',
        'tick_aggression_ratio', 'tick_aggression_ratio_ma', 'tick_aggression_ratio_std'
    ]
    
    for feature in microstructure_features:
        all_features['microstructure_features'].add(feature)
        print(f"  • {feature}")
    
    # 6. Entropy Features
    print(f"\n6. 🔬 Entropy Features")
    print("-" * 50)
    
    entropy_features = [
        'price_entropy', 'volume_entropy', 'return_entropy',
        'price_entropy_ma', 'volume_entropy_ma', 'return_entropy_ma',
        'price_entropy_std', 'volume_entropy_std', 'return_entropy_std',
        'price_entropy_skew', 'volume_entropy_skew', 'return_entropy_skew',
        'price_entropy_kurtosis', 'volume_entropy_kurtosis', 'return_entropy_kurtosis',
        'price_entropy_ratio', 'volume_entropy_ratio', 'return_entropy_ratio',
        'price_entropy_diff', 'volume_entropy_diff', 'return_entropy_diff',
        'price_entropy_product', 'volume_entropy_product', 'return_entropy_product',
        'price_entropy_squared', 'volume_entropy_squared', 'return_entropy_squared',
        'price_entropy_cubed', 'volume_entropy_cubed', 'return_entropy_cubed',
        'price_entropy_cross_timeframe', 'volume_entropy_cross_timeframe', 'return_entropy_cross_timeframe',
        'price_entropy_regime', 'volume_entropy_regime', 'return_entropy_regime',
        'price_entropy_interaction', 'volume_entropy_interaction', 'return_entropy_interaction'
    ]
    
    for feature in entropy_features:
        all_features['entropy_features'].add(feature)
        print(f"  • {feature}")
    
    # 7. Autoencoder Features
    print(f"\n7. 🤖 Autoencoder Features")
    print("-" * 50)
    
    autoencoder_features = [
        'autoencoder_encoded_1', 'autoencoder_encoded_2', 'autoencoder_encoded_3',
        'autoencoder_encoded_4', 'autoencoder_encoded_5', 'autoencoder_encoded_6',
        'autoencoder_encoded_7', 'autoencoder_encoded_8', 'autoencoder_encoded_9',
        'autoencoder_encoded_10', 'autoencoder_encoded_11', 'autoencoder_encoded_12',
        'autoencoder_encoded_13', 'autoencoder_encoded_14', 'autoencoder_encoded_15',
        'autoencoder_encoded_16', 'autoencoder_encoded_17', 'autoencoder_encoded_18',
        'autoencoder_encoded_19', 'autoencoder_encoded_20', 'autoencoder_encoded_21',
        'autoencoder_encoded_22', 'autoencoder_encoded_23', 'autoencoder_encoded_24',
        'autoencoder_encoded_25', 'autoencoder_encoded_26', 'autoencoder_encoded_27',
        'autoencoder_encoded_28', 'autoencoder_encoded_29', 'autoencoder_encoded_30',
        'autoencoder_reconstruction_error', 'autoencoder_reconstruction_error_ma',
        'autoencoder_reconstruction_error_std', 'autoencoder_reconstruction_error_skew',
        'autoencoder_reconstruction_error_kurtosis', 'autoencoder_reconstruction_error_ratio',
        'autoencoder_reconstruction_error_diff', 'autoencoder_reconstruction_error_product',
        'autoencoder_reconstruction_error_squared', 'autoencoder_reconstruction_error_cubed',
        'autoencoder_reconstruction_error_cross_timeframe', 'autoencoder_reconstruction_error_regime',
        'autoencoder_reconstruction_error_interaction'
    ]
    
    for feature in autoencoder_features:
        all_features['autoencoder_features'].add(feature)
        print(f"  • {feature}")
    
    # 8. Legacy Features
    print(f"\n8. 🔧 Legacy Features")
    print("-" * 50)
    
    legacy_features = [
        'legacy_rsi_14', 'legacy_macd_12_26_9', 'legacy_bollinger_20_2',
        'legacy_sma_20', 'legacy_ema_21', 'legacy_atr_14',
        'legacy_stochastic_14_3', 'legacy_williams_r_14', 'legacy_cci_20',
        'legacy_adx_14', 'legacy_aroon_14', 'legacy_sar_0.02_0.2',
        'legacy_ultimate_oscillator_7_14_28', 'legacy_kst_10_15_20_30_10_10_10_15',
        'legacy_apo_12_26', 'legacy_cmo_14', 'legacy_natr_14',
        'legacy_pfe_10', 'legacy_t3_20_0.7', 'legacy_kama_30',
        'legacy_obv', 'legacy_ad', 'legacy_adosc_3_10',
        'legacy_mfi_14', 'legacy_vwap', 'legacy_vwma_20',
        'legacy_vpt', 'legacy_nvi', 'legacy_pvi',
        'legacy_avgprice', 'legacy_medprice', 'legacy_typprice', 'legacy_wclprice',
        'legacy_ht_dcperiod', 'legacy_ht_dcphase', 'legacy_ht_phasor', 'legacy_ht_sine', 'legacy_ht_trendmode',
        'legacy_beta_5', 'legacy_correl_5', 'legacy_linearreg_5', 'legacy_linearreg_angle_5',
        'legacy_linearreg_intercept_5', 'legacy_linearreg_slope_5', 'legacy_stddev_5', 'legacy_tsf_5', 'legacy_var_5'
    ]
    
    for feature in legacy_features:
        all_features['legacy_features'].add(feature)
        print(f"  • {feature}")
    
    # 9. Time Features
    print(f"\n9. ⏰ Time Features")
    print("-" * 50)
    
    time_features = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'day_of_month_sin', 'day_of_month_cos', 'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos', 'year_sin', 'year_cos',
        'hour_encoded', 'day_of_week_encoded', 'day_of_month_encoded',
        'month_encoded', 'quarter_encoded', 'year_encoded',
        'hour_ratio', 'day_of_week_ratio', 'day_of_month_ratio',
        'month_ratio', 'quarter_ratio', 'year_ratio',
        'hour_diff', 'day_of_week_diff', 'day_of_month_diff',
        'month_diff', 'quarter_diff', 'year_diff',
        'hour_product', 'day_of_week_product', 'day_of_month_product',
        'month_product', 'quarter_product', 'year_product',
        'hour_squared', 'day_of_week_squared', 'day_of_month_squared',
        'month_squared', 'quarter_squared', 'year_squared',
        'hour_cubed', 'day_of_week_cubed', 'day_of_month_cubed',
        'month_cubed', 'quarter_cubed', 'year_cubed'
    ]
    
    for feature in time_features:
        all_features['time_features'].add(feature)
        print(f"  • {feature}")
    
    # 10. Order Flow Features
    print(f"\n10. 📊 Order Flow Features")
    print("-" * 50)
    
    order_flow_features = [
        'taker_buy_ratio', 'taker_sell_ratio', 'taker_quote_ratio',
        'market_aggression_index', 'aggression_score', 'taker_avg_price',
        'taker_price_deviation', 'order_flow_imbalance', 'taker_volume_momentum',
        'taker_quote_momentum', 'taker_participation_rate',
        'taker_buy_ratio_ma', 'taker_sell_ratio_ma', 'taker_quote_ratio_ma',
        'market_aggression_index_ma', 'aggression_score_ma', 'taker_avg_price_ma',
        'taker_price_deviation_ma', 'order_flow_imbalance_ma', 'taker_volume_momentum_ma',
        'taker_quote_momentum_ma', 'taker_participation_rate_ma',
        'taker_buy_ratio_std', 'taker_sell_ratio_std', 'taker_quote_ratio_std',
        'market_aggression_index_std', 'aggression_score_std', 'taker_avg_price_std',
        'taker_price_deviation_std', 'order_flow_imbalance_std', 'taker_volume_momentum_std',
        'taker_quote_momentum_std', 'taker_participation_rate_std',
        'taker_buy_ratio_skew', 'taker_sell_ratio_skew', 'taker_quote_ratio_skew',
        'market_aggression_index_skew', 'aggression_score_skew', 'taker_avg_price_skew',
        'taker_price_deviation_skew', 'order_flow_imbalance_skew', 'taker_volume_momentum_skew',
        'taker_quote_momentum_skew', 'taker_participation_rate_skew',
        'taker_buy_ratio_kurtosis', 'taker_sell_ratio_kurtosis', 'taker_quote_ratio_kurtosis',
        'market_aggression_index_kurtosis', 'aggression_score_kurtosis', 'taker_avg_price_kurtosis',
        'taker_price_deviation_kurtosis', 'order_flow_imbalance_kurtosis', 'taker_volume_momentum_kurtosis',
        'taker_quote_momentum_kurtosis', 'taker_participation_rate_kurtosis'
    ]
    
    for feature in order_flow_features:
        all_features['order_flow_features'].add(feature)
        print(f"  • {feature}")
    
    # 11. Support/Resistance Features
    print(f"\n11. 🎯 Support/Resistance Features")
    print("-" * 50)
    
    sr_features = [
        'support_level_1', 'support_level_2', 'support_level_3', 'support_level_4', 'support_level_5',
        'resistance_level_1', 'resistance_level_2', 'resistance_level_3', 'resistance_level_4', 'resistance_level_5',
        'pivot_point', 'pivot_point_r1', 'pivot_point_r2', 'pivot_point_s1', 'pivot_point_s2',
        'fibonacci_23.6', 'fibonacci_38.2', 'fibonacci_50.0', 'fibonacci_61.8', 'fibonacci_78.6',
        'volume_profile_vah', 'volume_profile_val', 'volume_profile_poc',
        'support_strength', 'resistance_strength', 'support_distance', 'resistance_distance',
        'support_breakout', 'resistance_breakout', 'support_bounce', 'resistance_bounce',
        'support_volume', 'resistance_volume', 'support_volume_ratio', 'resistance_volume_ratio',
        'support_volume_diff', 'resistance_volume_diff', 'support_volume_product', 'resistance_volume_product',
        'support_volume_squared', 'resistance_volume_squared', 'support_volume_cubed', 'resistance_volume_cubed',
        'support_volume_cross_timeframe', 'resistance_volume_cross_timeframe', 'support_volume_regime', 'resistance_volume_regime',
        'support_volume_interaction', 'resistance_volume_interaction'
    ]
    
    for feature in sr_features:
        all_features['sr_features'].add(feature)
        print(f"  • {feature}")
    
    # Calculate totals
    total_features = sum(len(features) for features in all_features.values())
    
    print(f"\n📊 FEATURE AUDIT SUMMARY")
    print("=" * 60)
    print(f"Technical Indicators: {len(all_features['technical_indicators'])}")
    print(f"Cross-timeframe Features: {len(all_features['cross_timeframe_features'])}")
    print(f"Interaction Features: {len(all_features['interaction_features'])}")
    print(f"Regime Features: {len(all_features['regime_features'])}")
    print(f"Microstructure Features: {len(all_features['microstructure_features'])}")
    print(f"Entropy Features: {len(all_features['entropy_features'])}")
    print(f"Autoencoder Features: {len(all_features['autoencoder_features'])}")
    print(f"Legacy Features: {len(all_features['legacy_features'])}")
    print(f"Time Features: {len(all_features['time_features'])}")
    print(f"Order Flow Features: {len(all_features['order_flow_features'])}")
    print(f"Support/Resistance Features: {len(all_features['sr_features'])}")
    print(f"\n🎯 TOTAL FEATURES: {total_features}")
    
    if total_features >= 200:
        print("✅ We have 200+ features as expected!")
    else:
        print(f"⚠️  We only have {total_features} features, missing {200 - total_features} features")
    
    return all_features, total_features

if __name__ == "__main__":
    features, total = audit_previous_features()
    print(f"\n🎉 Feature audit completed! Total: {total} features")