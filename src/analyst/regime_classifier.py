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
        # The regime_map is now more conceptual, as LGBM will predict string labels
        self.regime_map = {
            "BULL_TREND": "BULL_TREND",
            "BEAR_TREND": "BEAR_TREND",
            "SIDEWAYS_RANGE": "SIDEWAYS_RANGE",
            "SR_ZONE_ACTION": "SR_ZONE_ACTION" # SR_ZONE_ACTION will be determined separately
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
        # Ensure ADX is calculated on the klines_df
        klines_df_copy = klines_df.copy() # Avoid modifying original DataFrame passed in
        klines_df_copy.ta.adx(length=self.config["adx_period"], append=True, col_names=('ADX', 'DMP', 'DMN'))
        current_adx = klines_df_copy['ADX'].iloc[-1] if 'ADX' in klines_df_copy.columns else 0

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
        Trains the Market Regime Classifier using a pseudo-labeling approach.
        The `SR_ZONE_ACTION` regime is determined dynamically and not part of the clustering.
        """
        print("Training Market Regime Classifier (Pseudo-Labeling + LightGBM)...")
        
        if historical_features.empty or historical_klines.empty:
            print("No historical features or klines to train classifier.")
            return

        # Ensure klines and features are aligned by index
        combined_data = historical_klines.join(historical_features, how='inner').dropna()

        if combined_data.empty:
            print("Insufficient aligned data for training classifier after dropping NaNs.")
            return

        # --- Pseudo-Labeling for Market Regimes ---
        # 1. Calculate SMA_50 for overall trend direction
        combined_data['SMA_50'] = combined_data['close'].rolling(window=50).mean()

        # 2. Use ADX from features for trend strength
        # ADX is already in historical_features, so it's in combined_data
        adx_col = 'ADX'

        # 3. Use MACD Histogram from features for short-term momentum
        macd_hist_col = [col for col in combined_data.columns if 'MACDh_' in col][0] # Find MACD Hist column

        # Initialize pseudo-labels
        combined_data['simulated_label'] = "SIDEWAYS_RANGE" # Default to sideways

        # Define thresholds from config
        trend_threshold = self.config.get("trend_threshold", 20) # ADX value for strong trend

        # Apply pseudo-labeling rules
        # Bull Trend: ADX > threshold, MACD_HIST > 0, Close > SMA_50
        bull_condition = (combined_data[adx_col] > trend_threshold) & \
                         (combined_data[macd_hist_col] > 0) & \
                         (combined_data['close'] > combined_data['SMA_50'])
        combined_data.loc[bull_condition, 'simulated_label'] = "BULL_TREND"

        # Bear Trend: ADX > threshold, MACD_HIST < 0, Close < SMA_50
        bear_condition = (combined_data[adx_col] > trend_threshold) & \
                         (combined_data[macd_hist_col] < 0) & \
                         (combined_data['close'] < combined_data['SMA_50'])
        combined_data.loc[bear_condition, 'simulated_label'] = "BEAR_TREND"

        # Features for LightGBM training
        # Use a consistent set of features that are generally available and relevant
        features_for_lgbm = combined_data[['ADX', macd_hist_col, 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting']].dropna()
        labels_for_lgbm = combined_data.loc[features_for_lgbm.index, 'simulated_label']

        if features_for_lgbm.empty or labels_for_lgbm.empty:
            print("Insufficient features or labels for training LightGBM after pseudo-labeling and dropping NaNs.")
            return

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features_for_lgbm)

        # Train LightGBM classifier on pseudo-labels
        self.lgbm_classifier = LGBMClassifier(random_state=42, verbose=-1) # verbose=-1 suppresses training output
        self.lgbm_classifier.fit(scaled_features, labels_for_lgbm)
        self.trained = True
        print("Market Regime Classifier training completed with pseudo-labeling.")

    def predict_regime(self, current_features: pd.DataFrame, current_klines: pd.DataFrame, sr_levels: list):
        """
        Predicts the current market regime and provides the Trend Strength Score.
        Dynamically checks for SR_ZONE_ACTION.
        """
        if not self.trained:
            print("Warning: Classifier not trained. Returning default regime.")
            return "UNKNOWN", 0.0, 0.0 # Regime, Trend Strength, ADX

        # Check for SR_ZONE_ACTION first
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
            # For SR_ZONE_ACTION, trend strength is less relevant, return 0.0 for now
            return "SR_ZONE_ACTION", 0.0, 0.0 

        # If not SR_ZONE_ACTION, proceed with LightGBM classification
        # Ensure features for prediction match those used for training
        # We need to ensure 'ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting' are present
        
        # Dynamically find the MACD_HIST column name
        macd_hist_col = [col for col in current_features.columns if 'MACDh_' in col]
        if not macd_hist_col:
            print("MACD Histogram column not found in current_features. Cannot predict regime.")
            return "UNKNOWN", 0.0, 0.0
        macd_hist_col = macd_hist_col[0]

        required_features = ['ADX', macd_hist_col, 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting']
        
        # Filter current_features to only include the required columns and handle potential missing ones
        features_for_prediction_df = current_features[required_features].copy()
        
        # Fill any NaNs that might appear in the latest features before scaling
        features_for_prediction_df.fillna(0, inplace=True) # Or use a more sophisticated imputation

        if features_for_prediction_df.empty:
            print("Insufficient features for regime prediction after filtering and dropping NaNs.")
            return "UNKNOWN", 0.0, 0.0

        scaled_features = self.scaler.transform(features_for_prediction_df.tail(1)) # Predict for the latest data point
        predicted_regime = self.lgbm_classifier.predict(scaled_features)[0]

        # Calculate Trend Strength Score
        trend_strength, current_adx = self.calculate_trend_strength(current_klines)

        print(f"Predicted Regime: {predicted_regime}, Trend Strength: {trend_strength:.2f}, ADX: {current_adx:.2f}")
        return predicted_regime, trend_strength, current_adx
