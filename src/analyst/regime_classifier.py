# src/analyst/regime_classifier.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans # Placeholder for Wasserstein k-Means
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier # For real-time classification
import pandas_ta as ta
import joblib # For saving/loading models
import os # For path manipulation

from sr_analyzer import SRLevelAnalyzer # Assuming sr_analyzer.py is accessible
from utils.logger import system_logger # Centralized logger


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
        self.logger = system_logger.getChild('MarketRegimeClassifier') # Child logger for classifier

        # The regime_map is now more conceptual, as LGBM will predict string labels
        self.regime_map = {
            "BULL_TREND": "BULL_TREND",
            "BEAR_TREND": "BEAR_TREND",
            "SIDEWAYS_RANGE": "SIDEWAYS_RANGE",
            "SR_ZONE_ACTION": "SR_ZONE_ACTION" # SR_ZONE_ACTION will be determined separately
        }
        self.trained = False
        self.model_path = self.config.get("model_storage_path", "models/analyst/") + "regime_classifier.joblib"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True) # Ensure model storage path exists

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
        current_adx = klines_df_copy['ADX'].iloc[-1] if 'ADX' in klines_df_copy.columns and not klines_df_copy['ADX'].isnull().all() else 0

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
        self.logger.info("Training Market Regime Classifier (Pseudo-Labeling + LightGBM)...")
        
        if historical_features.empty or historical_klines.empty:
            self.logger.warning("No historical features or klines to train classifier.")
            return

        # Ensure klines and features are aligned by index
        combined_data = historical_klines.join(historical_features, how='inner').dropna()

        if combined_data.empty:
            self.logger.warning("Insufficient aligned data for training classifier after dropping NaNs.")
            return

        # --- Pseudo-Labeling for Market Regimes ---
        # 1. Calculate SMA_50 for overall trend direction
        combined_data['SMA_50'] = combined_data['close'].rolling(window=50).mean()

        # 2. Use ADX from features for trend strength
        # ADX is already in historical_features, so it's in combined_data
        adx_col = 'ADX'

        # 3. Use MACD Histogram from features for short-term momentum
        macd_hist_col = [col for col in combined_data.columns if 'MACDh_' in col]
        if not macd_hist_col:
            self.logger.warning("MACD Histogram column not found for pseudo-labeling. Skipping training.")
            return
        macd_hist_col = macd_hist_col[0]

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
            self.logger.warning("Insufficient features or labels for training LightGBM after pseudo-labeling and dropping NaNs.")
            return

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features_for_lgbm)

        # Train LightGBM classifier on pseudo-labels
        self.lgbm_classifier = LGBMClassifier(random_state=42, verbose=-1) # verbose=-1 suppresses training output
        self.lgbm_classifier.fit(scaled_features, labels_for_lgbm)
        self.trained = True
        self.logger.info("Market Regime Classifier training completed with pseudo-labeling.")
        self.save_model() # Save the model after training

    def predict_regime(self, current_features: pd.DataFrame, current_klines: pd.DataFrame, sr_levels: list):
        """
        Predicts the current market regime and provides the Trend Strength Score.
        Dynamically checks for SR_ZONE_ACTION.
        """
        if not self.trained:
            self.logger.warning("Warning: Classifier not trained. Attempting to load model...")
            if not self.load_model():
                self.logger.error("Classifier not trained and failed to load model. Returning default regime.")
                # Calculate Trend Strength Score even if not trained for fallback
                trend_strength, current_adx = self.calculate_trend_strength(current_klines)
                return "UNKNOWN", trend_strength, current_adx # Regime, Trend Strength, ADX

        # Calculate Trend Strength Score and ADX first, as it's needed regardless of regime
        trend_strength, current_adx = self.calculate_trend_strength(current_klines)

        # Check for SR_ZONE_ACTION
        current_price = current_klines['close'].iloc[-1]
        current_atr = current_klines['ATR'].iloc[-1] if 'ATR' in current_klines.columns and not current_klines['ATR'].isnull().all() else 0
        proximity_multiplier = self.config.get("proximity_multiplier", 0.25) # From main config's analyst section

        is_sr_interacting = False
        if sr_levels and current_atr > 0:
            for level_info in sr_levels:
                level_price = level_info['level_price']
                level_type = level_info['type'] # "Support" or "Resistance"
                
                # Only consider relevant levels (e.g., Strong, Very Strong, Moderate)
                if level_info["current_expectation"] not in ["Very Strong", "Strong", "Moderate"]:
                    continue

                tolerance_abs = current_atr * proximity_multiplier
                
                # NEW LOGIC: Only care about Resistance if price is below it, Support if price is above it
                directional_condition_met = False
                if level_type == "Resistance" and current_price <= level_price:
                    directional_condition_met = True
                elif level_type == "Support" and current_price >= level_price:
                    directional_condition_met = True
                
                if directional_condition_met:
                    # Check for proximity within the tolerance band
                    if (current_price <= level_price + tolerance_abs) and \
                       (current_price >= level_price - tolerance_abs):
                        is_sr_interacting = True
                        break # Found an interacting S/R level that meets criteria
        
        if is_sr_interacting:
            self.logger.info("Detected SR_ZONE_ACTION.")
            # For SR_ZONE_ACTION, return the regime but still provide trend strength and ADX
            return "SR_ZONE_ACTION", trend_strength, current_adx 

        # If not SR_ZONE_ACTION, proceed with LightGBM classification
        # Ensure features for prediction match those used for training
        # We need to ensure 'ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting' are present
        
        # Dynamically find the MACD_HIST column name
        macd_hist_col = [col for col in current_features.columns if 'MACDh_' in col]
        if not macd_hist_col:
            self.logger.warning("MACD Histogram column not found in current_features. Cannot predict regime.")
            return "UNKNOWN", trend_strength, current_adx # Return calculated trend strength and ADX

        macd_hist_col = macd_hist_col[0]

        required_features = ['ADX', macd_hist_col, 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting']
        
        # Filter current_features to only include the required columns and handle potential missing ones
        features_for_prediction_df = current_features[required_features].copy()
        
        # Fill any NaNs that might appear in the latest features before scaling
        features_for_prediction_df.fillna(0, inplace=True) # Or use a more sophisticated imputation

        if features_for_prediction_df.empty:
            self.logger.warning("Insufficient features for regime prediction after filtering and dropping NaNs.")
            return "UNKNOWN", trend_strength, current_adx # Return calculated trend strength and ADX

        # Ensure the scaler is fitted before transforming
        if self.scaler is None:
            self.logger.error("Scaler not fitted. Cannot predict regime. Returning UNKNOWN.")
            return "UNKNOWN", trend_strength, current_adx

        scaled_features = self.scaler.transform(features_for_prediction_df.tail(1)) # Predict for the latest data point
        predicted_regime = self.lgbm_classifier.predict(scaled_features)[0]

        self.logger.info(f"Predicted Regime: {predicted_regime}, Trend Strength: {trend_strength:.2f}, ADX: {current_adx:.2f}")
        return predicted_regime, trend_strength, current_adx

    def save_model(self):
        """Saves the trained LGBM classifier and scaler using joblib."""
        try:
            model_data = {
                'lgbm_classifier': self.lgbm_classifier,
                'scaler': self.scaler
            }
            joblib.dump(model_data, self.model_path)
            self.logger.info(f"Market Regime Classifier model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving Market Regime Classifier model: {e}", exc_info=True)

    def load_model(self):
        """Loads the trained LGBM classifier and scaler from file."""
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Market Regime Classifier model file not found at {self.model_path}. Cannot load.")
            self.trained = False
            return False
        try:
            model_data = joblib.load(self.model_path)
            self.lgbm_classifier = model_data['lgbm_classifier']
            self.scaler = model_data['scaler']
            self.trained = True
            self.logger.info(f"Market Regime Classifier model loaded from {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading Market Regime Classifier model from {self.model_path}: {e}", exc_info=True)
            self.trained = False
            return False

# --- Example Usage (Main execution block for demonstration) ---
if __name__ == "__main__":
    # This block is for standalone testing of the MarketRegimeClassifier
    # In the full pipeline, it will be orchestrated by Analyst.

    # Ensure dummy data exists for loading
    from src.analyst.data_utils import create_dummy_data, load_klines_data, load_agg_trades_data, load_futures_data
    from config import CONFIG # Import the main CONFIG dictionary

    klines_filename = CONFIG['KLINES_FILENAME']
    agg_trades_filename = CONFIG['AGG_TRADES_FILENAME']
    futures_filename = CONFIG['FUTURES_FILENAME']

    create_dummy_data(klines_filename, 'klines', num_records=1000)
    create_dummy_data(agg_trades_filename, 'agg_trades', num_records=5000)
    create_dummy_data(futures_filename, 'futures', num_records=500)

    # Load raw data
    klines_df = load_klines_data(klines_filename)
    agg_trades_df = load_agg_trades_data(agg_trades_filename)
    futures_df = load_futures_data(futures_filename)

    if klines_df.empty:
        print("Failed to load klines data for demo. Exiting.")
        sys.exit(1)

    # Simulate SR levels (normally from SRLevelAnalyzer)
    from src.analyst.sr_analyzer import SRLevelAnalyzer
    sr_analyzer_demo = SRLevelAnalyzer(CONFIG["sr_analyzer"])
    daily_df_for_sr = klines_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    daily_df_for_sr.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    sr_levels_demo = sr_analyzer_demo.analyze(daily_df_for_sr)

    # Simulate historical features (normally from FeatureEngineeringEngine)
    from src.analyst.feature_engineering import FeatureEngineeringEngine
    feature_engine_demo = FeatureEngineeringEngine(CONFIG)
    # Ensure ATR is calculated before passing to classifier
    klines_df_for_features = klines_df.copy()
    klines_df_for_features.ta.atr(length=CONFIG['BEST_PARAMS']['atr_period'], append=True, col_names=('ATR'))
    historical_features_demo = feature_engine_demo.generate_all_features(klines_df_for_features, agg_trades_df, futures_df, sr_levels_demo)
    
    if historical_features_demo.empty:
        print("Failed to generate historical features for demo. Exiting.")
        sys.exit(1)

    # Initialize and train the classifier
    classifier = MarketRegimeClassifier(CONFIG, sr_analyzer_demo)
    classifier.train_classifier(historical_features_demo.copy(), klines_df.copy()) # Pass copies for training

    # Test prediction
    print("\n--- Testing Regime Prediction ---")
    # Get the last few data points for prediction
    test_features = historical_features_demo.tail(5)
    test_klines = klines_df.tail(5)

    for i in range(len(test_features)):
        current_feat = test_features.iloc[[i]]
        current_kline = test_klines.iloc[[i]]
        
        # Ensure 'ATR' is in current_kline for predict_regime
        if 'ATR' not in current_kline.columns and 'ATR' in current_feat.columns:
            current_kline['ATR'] = current_feat['ATR']

        regime, trend_strength, adx = classifier.predict_regime(current_feat, current_kline, sr_levels_demo)
        print(f"Timestamp: {current_feat.index[0]} | Price: {current_feat['close'].iloc[0]:.2f} | Predicted Regime: {regime} | Trend Strength: {trend_strength:.2f} | ADX: {adx:.2f}")

    # Test loading the model
    print("\n--- Testing Model Loading ---")
    new_classifier_instance = MarketRegimeClassifier(CONFIG, sr_analyzer_demo)
    if new_classifier_instance.load_model():
        print("Model loaded successfully. Testing prediction with loaded model:")
        current_feat = historical_features_demo.tail(1)
        current_kline = klines_df.tail(1)
        if 'ATR' not in current_kline.columns and 'ATR' in current_feat.columns:
            current_kline['ATR'] = current_feat['ATR']

        regime, trend_strength, adx = new_classifier_instance.predict_regime(current_feat, current_kline, sr_levels_demo)
        print(f"Loaded Model Prediction: Timestamp: {current_feat.index[0]} | Price: {current_feat['close'].iloc[0]:.2f} | Predicted Regime: {regime} | Trend Strength: {trend_strength:.2f} | ADX: {adx:.2f}")
    else:
        print("Failed to load model for testing.")
