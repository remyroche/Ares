# src/analyst/regime_classifier.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans # Placeholder for Wasserstein k-Means
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier # For real-time classification
import pandas_ta as ta
from sr_analyzer import SRLevelAnalyzer # Assuming sr_analyzer.py is accessible

class MarketRegimeClassifier:
    """
    Classifies the current market into one of four primary states:
    BULL_TREND, BEAR_TREND, SIDEWAYS_RANGE, or SR_ZONE_ACTION.
    Also provides a Trend Strength Score.
    """
    def __init__(self, config, sr_analyzer: SRLevelAnalyzer):
        self.config = config.get("analyst", {}).get("market_regime_classifier", {})
        self.sr_analyzer = sr_analyzer
        self.kmeans_model = None # Placeholder for Wasserstein k-Means
        self.lgbm_classifier = None
        self.scaler = None
        self.regime_map = {
            0: "BULL_TREND",
            1: "BEAR_TREND",
            2: "SIDEWAYS_RANGE",
            3: "SR_ZONE_ACTION" # SR_ZONE_ACTION will be determined separately
        }
        self.trained = False

    def calculate_trend_strength(self, klines_df: pd.DataFrame):
        """
        Calculates the Composite Trend Strength Indicator.
        Ranges from -1.0 (Maximum Bearish Trend) to +1.0 (Maximum Bullish Trend).
        """
        if klines_df.empty or len(klines_df) < max(self.config["adx_period"], self.config["macd_fast_period"]):
            return 0.0, 0.0 # Return 0 if insufficient data

        # Directional Component (MACD)
        macd = klines_df.ta.macd(
            fast=self.config["macd_fast_period"],
            slow=self.config["macd_slow_period"],
            signal=self.config["macd_signal_period"],
            append=False
        )
        # Ensure MACDh_12_26_9 exists, adjust column name if periods are different
        macd_hist_col = f'MACDh_{self.config["macd_fast_period"]}_{self.config["macd_slow_period"]}_{self.config["macd_signal_period"]}'
        
        if macd_hist_col not in macd.columns or klines_df['close'].iloc[-1] == 0:
            direction_score = 0.0
        else:
            normalized_hist = macd[macd_hist_col].iloc[-1] / klines_df['close'].iloc[-1]
            scaling_factor = self.config.get("trend_scaling_factor", 100)
            direction_score = np.clip(normalized_hist * scaling_factor, -1, 1)

        # Momentum Component (ADX)
        klines_df.ta.adx(length=self.config["adx_period"], append=True, col_names=('ADX', 'DMP', 'DMN'))
        current_adx = klines_df['ADX'].iloc[-1] if 'ADX' in klines_df.columns else 0

        trend_threshold = self.config.get("trend_threshold", 20)
        max_strength_threshold = self.config.get("max_strength_threshold", 60)

        if max_strength_threshold - trend_threshold > 0:
            momentum_score = np.clip((current_adx - trend_threshold) / 
                                     (max_strength_threshold - trend_threshold), 0, 1)
        else:
            momentum_score = 0.0

        final_trend_strength = direction_score * momentum_score
        return final_trend_strength, current_adx

    def train_classifier(self, historical_features: pd.DataFrame, historical_klines: pd.DataFrame):
        """
        Trains the Market Regime Classifier.
        This is a placeholder for Wasserstein k-Means + LightGBM.
        The `SR_ZONE_ACTION` regime is determined dynamically and not part of the clustering.
        """
        print("Placeholder: Training Market Regime Classifier (Wasserstein k-Means + LightGBM)...")
        # For actual implementation:
        # 1. Preprocess historical_features (e.g., scaling)
        # 2. Apply Wasserstein k-Means (requires a custom implementation or specialized library)
        #    - Cluster based on features like trend, volatility, volume patterns.
        # 3. Map clusters to conceptual regimes (BULL_TREND, BEAR_TREND, SIDEWAYS_RANGE).
        # 4. Train LightGBM classifier on features to predict these mapped regimes.

        # Simulated training for demonstration
        if historical_features.empty:
            print("No historical features to train classifier.")
            return

        # Use a subset of features for clustering/classification
        features_for_clustering = historical_features[['ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error']].dropna()
        if features_for_clustering.empty:
            print("Insufficient features for training classifier after dropping NaNs.")
            return

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features_for_clustering)

        # Simulate K-Means clustering
        n_clusters = self.config.get("kmeans_n_clusters", 3) # Exclude SR_ZONE_ACTION from K-Means
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(scaled_features)

        # Map clusters to simplified regimes for training LightGBM
        # This mapping would be determined by analyzing cluster characteristics (e.g., mean ADX, MACD)
        # For this placeholder, we'll just assign arbitrary labels to clusters
        simulated_labels = np.array([
            "BULL_TREND" if c == 0 else ("BEAR_TREND" if c == 1 else "SIDEWAYS_RANGE")
            for c in clusters
        ])

        # Train LightGBM to predict these labels
        self.lgbm_classifier = LGBMClassifier(random_state=42)
        self.lgbm_classifier.fit(scaled_features, simulated_labels)
        self.trained = True
        print("Market Regime Classifier training placeholder complete.")

    def predict_regime(self, current_features: pd.DataFrame, current_klines: pd.DataFrame, sr_levels: list):
        """
        Predicts the current market regime and provides the Trend Strength Score.
        Dynamically checks for SR_ZONE_ACTION.
        """
        if not self.trained:
            print("Warning: Classifier not trained. Returning default regime.")
            return "UNKNOWN", 0.0, 0.0 # Regime, Trend Strength, ADX

        # Check for SR_ZONE_ACTION first
        # sr_analyzer.py's analyze method expects a DataFrame with 'Close', 'High', 'Low', 'Volume'
        # For real-time, we'd need to pass a small window of recent data or just the current price and SR levels
        
        # To reuse sr_analyzer, we need to adapt the input.
        # Let's assume current_klines contains enough data for sr_analyzer to work.
        # sr_analyzer is initialized with the main Analyst, so it's available.
        
        # Check if current price is interacting with SR levels
        current_price = current_klines['close'].iloc[-1]
        current_atr = current_klines['ATR'].iloc[-1] if 'ATR' in current_klines.columns else 0
        proximity_multiplier = self.config.get("proximity_multiplier", 0.25) # From main config's analyst section

        is_sr_interacting = False
        if sr_levels and current_atr > 0:
            for level_info in sr_levels:
                level_price = level_info['level_price']
                tolerance_abs = current_atr * proximity_multiplier
                if (current_price <= level_price + tolerance_abs) and \
                   (current_price >= level_price - tolerance_abs):
                    is_sr_interacting = True
                    break
        
        if is_sr_interacting:
            print("Detected SR_ZONE_ACTION.")
            return "SR_ZONE_ACTION", 0.0, 0.0 # No trend strength for SR zone

        # If not SR_ZONE_ACTION, proceed with LightGBM classification
        features_for_prediction = current_features[['ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error']].dropna()
        if features_for_prediction.empty:
            print("Insufficient features for regime prediction after dropping NaNs.")
            return "UNKNOWN", 0.0, 0.0

        scaled_features = self.scaler.transform(features_for_prediction.tail(1)) # Predict for the latest data point
        predicted_regime = self.lgbm_classifier.predict(scaled_features)[0]

        # Calculate Trend Strength Score
        trend_strength, current_adx = self.calculate_trend_strength(current_klines)

        print(f"Predicted Regime: {predicted_regime}, Trend Strength: {trend_strength:.2f}, ADX: {current_adx:.2f}")
        return predicted_regime, trend_strength, current_adx
