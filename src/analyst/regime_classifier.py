# src/analyst/regime_classifier.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans # Using standard KMeans for clustering on features
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier # For real-time classification
import pandas_ta as ta
import joblib # For saving/loading models
import os # For path manipulation
import sys # Import sys for sys.exit() in example usage
from src.analyst.sr_analyzer import SRLevelAnalyzer # Assuming sr_analyzer.py is accessible
from src.utils.logger import system_logger # Centralized logger
from src.config import CONFIG # Import CONFIG to get checkpoint paths


class MarketRegimeClassifier:
    """
    Classifies the current market into one of four primary states:
    BULL_TREND, BEAR_TREND, SIDEWAYS_RANGE, or SR_ZONE_ACTION.
    Also provides a Trend Strength Score.

    This enhanced version adds more advanced statistical and time-series derived features
    to improve the LightGBM classifier's ability to distinguish regimes.
    It also uses KMeans clustering on these features, with cluster labels serving as
    additional input for the LightGBM.
    """
    def __init__(self, config, sr_analyzer: SRLevelAnalyzer):
        self.config = config.get("analyst", {}).get("market_regime_classifier", {})
        self.global_config = config # Store global config to access other sections
        self.sr_analyzer = sr_analyzer
        self.kmeans_model = None # KMeans model for feature-based clustering
        self.lgbm_classifier = None
        self.scaler = None
        self.logger = system_logger.getChild('MarketRegimeClassifier') # Child logger for classifier

        self.regime_map = {
            "BULL_TREND": "BULL_TREND",
            "BEAR_TREND": "BEAR_TREND",
            "SIDEWAYS_RANGE": "SIDEWAYS_RANGE",
            "SR_ZONE_ACTION": "SR_ZONE_ACTION" # SR_ZONE_ACTION will be determined separately
        }
        self.trained = False
        # Default model path, can be overridden by save/load methods
        # Using the new CHECKPOINT_DIR from CONFIG
        self.default_model_path = os.path.join(CONFIG['CHECKPOINT_DIR'], "analyst_models", "regime_classifier.joblib")
        os.makedirs(os.path.dirname(self.default_model_path), exist_ok=True) # Ensure model storage path exists

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

    def _calculate_advanced_features(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates additional statistical and time-series derived features for regime classification.
        """
        if klines_df.empty or len(klines_df) < 50: # Need sufficient history for rolling features
            self.logger.warning("Insufficient data for advanced feature calculation. Returning empty DataFrame.")
            return pd.DataFrame(index=klines_df.index)

        features = pd.DataFrame(index=klines_df.index)
        
        # Ensure 'close' is numeric
        klines_df['close'] = pd.to_numeric(klines_df['close'], errors='coerce')
        klines_df['volume'] = pd.to_numeric(klines_df['volume'], errors='coerce')

        # Rolling Returns and Volatility
        features['log_returns'] = np.log(klines_df['close'] / klines_df['close'].shift(1))
        features['volatility_20'] = features['log_returns'].rolling(window=20).std() * np.sqrt(252) # Annualized daily vol
        features['volatility_50'] = features['log_returns'].rolling(window=50).std() * np.sqrt(252)

        # Statistical Moments of Returns
        features['skewness_20'] = features['log_returns'].rolling(window=20).skew()
        features['kurtosis_20'] = features['log_returns'].rolling(window=20).kurt()

        # Price Action Features (e.g., average candle body/wick size)
        features['candle_body'] = abs(klines_df['open'] - klines_df['close'])
        features['candle_range'] = klines_df['high'] - klines_df['low']
        
        # Avoid division by zero for normalization
        features['avg_body_size_20'] = (features['candle_body'].rolling(window=20).mean() / klines_df['close']).fillna(0)
        features['avg_range_size_20'] = (features['candle_range'].rolling(window=20).mean() / klines_df['close']).fillna(0)

        # Volume-based features
        features['volume_ma_20'] = klines_df['volume'].rolling(window=20).mean()
        features['volume_change'] = (klines_df['volume'] - features['volume_ma_20']) / features['volume_ma_20'].replace(0, np.nan)
        
        # Clean up NaNs from rolling windows
        features.fillna(0, inplace=True)
        
        return features

    def train_classifier(self, historical_features: pd.DataFrame, historical_klines: pd.DataFrame, model_path: str = None):
        """
        Trains the Market Regime Classifier using a pseudo-labeling approach.
        The `SR_ZONE_ACTION` regime is determined dynamically and not part of the clustering.
        """
        self.logger.info("Training Market Regime Classifier (Pseudo-Labeling + LightGBM)...")
        
        if historical_features.empty or historical_klines.empty:
            self.logger.warning("No historical features or klines to train classifier.")
            return

        # Calculate advanced features from historical klines
        advanced_features = self._calculate_advanced_features(historical_klines.copy())
        
        # Merge all features
        combined_data = historical_klines.join(historical_features, how='inner').join(advanced_features, how='inner').dropna()

        if combined_data.empty:
            self.logger.warning("Insufficient aligned data for training classifier after dropping NaNs.")
            return

        # --- Pseudo-Labeling for Market Regimes ---
        # 1. Calculate SMA_50 for overall trend direction
        combined_data['SMA_50'] = combined_data['close'].rolling(window=50).mean()

        # 2. Use ADX from features for trend strength
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

        # Features for LightGBM training (include new advanced features)
        # Ensure all necessary columns are present before slicing
        required_features_for_lgbm = [
            'ADX', macd_hist_col, 'ATR', 'volume_delta', 'autoencoder_reconstruction_error',
            'Is_SR_Interacting', 'log_returns', 'volatility_20', 'skewness_20',
            'kurtosis_20', 'avg_body_size_20', 'avg_range_size_20', 'volume_change'
        ]
        
        # Filter to only existing columns to avoid KeyError if some features are missing
        actual_features_for_lgbm = [col for col in required_features_for_lgbm if col in combined_data.columns]
        
        features_for_lgbm = combined_data[actual_features_for_lgbm].dropna()
        labels_for_lgbm = combined_data.loc[features_for_lgbm.index, 'simulated_label']

        if features_for_lgbm.empty or labels_for_lgbm.empty:
            self.logger.warning("Insufficient features or labels for training LightGBM after pseudo-labeling and dropping NaNs.")
            return

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features_for_lgbm)

        # --- Train KMeans Clustering Model ---
        # Use KMeans on a subset of scaled features to identify underlying market clusters
        # These cluster labels will then be used as an additional feature for the LightGBM
        n_clusters = self.config.get("kmeans_n_clusters", 4) # Configurable number of clusters
        if scaled_features.shape[0] >= n_clusters:
            try:
                self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = self.kmeans_model.fit_predict(scaled_features)
                features_for_lgbm['cluster_label'] = pd.Series(cluster_labels, index=features_for_lgbm.index)
                self.logger.info(f"KMeans clustering trained with {n_clusters} clusters.")
            except Exception as e:
                self.logger.error(f"Error training KMeans clustering: {e}. Skipping cluster feature.", exc_info=True)
                features_for_lgbm['cluster_label'] = -1 # Default to a neutral/unknown cluster
        else:
            self.logger.warning(f"Not enough data ({scaled_features.shape[0]} samples) to train KMeans with {n_clusters} clusters. Skipping cluster feature.")
            features_for_lgbm['cluster_label'] = -1


        # Train LightGBM classifier on pseudo-labels
        self.lgbm_classifier = LGBMClassifier(random_state=42, verbose=-1) # verbose=-1 suppresses training output
        self.lgbm_classifier.fit(features_for_lgbm, labels_for_lgbm) # Pass DataFrame directly
        self.trained = True
        self.logger.info("Market Regime Classifier training completed with pseudo-labeling and advanced features.")
        
        # Save the model after training if a path is provided
        if model_path:
            self.save_model(model_path)

    def predict_regime(self, current_features: pd.DataFrame, current_klines: pd.DataFrame, sr_levels: list):
        """
        Predicts the current market regime and provides the Trend Strength Score.
        Dynamically checks for SR_ZONE_ACTION.
        """
        if not self.trained:
            self.logger.warning("Warning: Classifier not trained. Attempting to load default model...")
            # Use self.default_model_path for loading if no specific path is given
            if not self.load_model(model_path=self.default_model_path):
                self.logger.error("Classifier not trained and failed to load model. Returning default regime.")
                # Calculate Trend Strength Score even if not trained for fallback
                trend_strength, current_adx = self.calculate_trend_strength(current_klines)
                return "UNKNOWN", trend_strength, current_adx # Regime, Trend Strength, ADX

        # Calculate Trend Strength Score and ADX first, as it's needed regardless of regime
        trend_strength, current_adx = self.calculate_trend_strength(current_klines)

        # Check for SR_ZONE_ACTION
        current_price = current_klines['close'].iloc[-1]
        current_atr = current_klines['ATR'].iloc[-1] if 'ATR' in current_klines.columns and not current_klines['ATR'].isnull().all() else 0
        proximity_multiplier = self.global_config['BEST_PARAMS'].get("proximity_multiplier", 0.25) # From main config's analyst section

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

        # Calculate advanced features for current data
        current_advanced_features = self._calculate_advanced_features(current_klines.copy())
        
        # Merge all current features
        # Ensure current_features has all expected columns from feature_engineering
        # before merging with advanced_features
        
        # Create a combined DataFrame for prediction input, ensuring all expected features are present
        # This is crucial because the scaler and LGBM expect the same feature set as during training.
        
        # First, ensure basic features are in current_features_row (from Analyst's feature_engineering)
        # Then, add advanced features to it.
        
        # Ensure current_features is a copy to avoid modifying original
        combined_current_features = current_features.copy()
        
        # Merge advanced features, handling potential index mismatches (e.g., if current_klines is shorter)
        # Use `reindex` and `fillna` to ensure consistency.
        combined_current_features = combined_current_features.merge(
            current_advanced_features, left_index=True, right_index=True, how='left', suffixes=('', '_adv')
        )
        
        # Filter to only existing columns to avoid KeyError if some features are missing
        required_features_for_lgbm = [
            'ADX', macd_hist_col, 'ATR', 'volume_delta', 'autoencoder_reconstruction_error',
            'Is_SR_Interacting', 'log_returns', 'volatility_20', 'skewness_20',
            'kurtosis_20', 'avg_body_size_20', 'avg_range_size_20', 'volume_change'
        ]

        # Add 'cluster_label' if KMeans was trained
        if self.kmeans_model:
            required_features_for_lgbm.append('cluster_label')

        # Filter to only existing columns in combined_current_features
        actual_features_for_lgbm = [col for col in required_features_for_lgbm if col in combined_current_features.columns]
        
        features_for_prediction_df = combined_current_features[actual_features_for_lgbm].dropna()
        
        # Predict cluster label for current features if KMeans was trained
        if self.kmeans_model and 'cluster_label' in features_for_prediction_df.columns:
            if not features_for_prediction_df.empty:
                try:
                    scaler_fitted_cols = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else actual_features_for_lgbm
                    current_scaled_input_for_kmeans = self.scaler.transform(
                        features_for_prediction_df.reindex(columns=scaler_fitted_cols, fill_value=0).tail(1)
                    )
                    predicted_cluster = self.kmeans_model.predict(current_scaled_input_for_kmeans)[0]
                    features_for_prediction_df.loc[features_for_prediction_df.index[-1], 'cluster_label'] = predicted_cluster
                except Exception as e:
                    self.logger.error(f"Error predicting cluster for current features: {e}. Setting cluster_label to -1.", exc_info=True)
                    features_for_prediction_df.loc[features_for_prediction_df.index[-1], 'cluster_label'] = -1
            else:
                features_for_prediction_df.loc[features_for_prediction_df.index[-1], 'cluster_label'] = -1


        if features_for_prediction_df.empty:
            self.logger.warning("Insufficient features for regime prediction after filtering and dropping NaNs.")
            return "UNKNOWN", trend_strength, current_adx # Return calculated trend strength and ADX

        # Ensure the scaler is fitted before transforming
        if self.scaler is None:
            self.logger.error("Scaler not fitted. Cannot predict regime. Returning UNKNOWN.")
            return "UNKNOWN", trend_strength, current_adx

        # Scale the final features for LGBM prediction
        # Use `reindex` to ensure column order matches training, and `fillna` for robustness
        lgbm_input_cols = self.lgbm_classifier.feature_name_ if hasattr(self.lgbm_classifier, 'feature_name_') else actual_features_for_lgbm
        if 'cluster_label' in lgbm_input_cols and 'cluster_label' not in features_for_prediction_df.columns:
            features_for_prediction_df['cluster_label'] = -1 # Add if missing for prediction consistency

        # Reindex the prediction DataFrame to match the training feature order
        features_for_prediction_reindexed = features_for_prediction_df.reindex(columns=lgbm_input_cols, fill_value=0).tail(1)
        
        scaled_features_for_lgbm = self.scaler.transform(features_for_prediction_reindexed)
        predicted_regime = self.lgbm_classifier.predict(scaled_features_for_lgbm)[0]

        self.logger.info(f"Predicted Regime: {predicted_regime}, Trend Strength: {trend_strength:.2f}, ADX: {current_adx:.2f}")
        return predicted_regime, trend_strength, current_adx

    def save_model(self, model_path: str = None):
        """Saves the trained LGBM classifier, scaler, and KMeans model using joblib."""
        path_to_save = model_path if model_path else self.default_model_path
        try:
            model_data = {
                'lgbm_classifier': self.lgbm_classifier,
                'scaler': self.scaler,
                'kmeans_model': self.kmeans_model, # Save KMeans model
                'lgbm_feature_names': self.lgbm_classifier.feature_name_ if hasattr(self.lgbm_classifier, 'feature_name_') else None,
                'scaler_feature_names': self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else None
            }
            joblib.dump(model_data, path_to_save)
            self.logger.info(f"Market Regime Classifier model saved to {path_to_save}")
        except Exception as e:
            self.logger.error(f"Error saving Market Regime Classifier model to {path_to_save}: {e}", exc_info=True)

    def load_model(self, model_path: str = None):
        """Loads the trained LGBM classifier, scaler, and KMeans model from file."""
        path_to_load = model_path if model_path else self.default_model_path
        if not os.path.exists(path_to_load):
            self.logger.warning(f"Market Regime Classifier model file not found at {path_to_load}. Cannot load.")
            self.trained = False
            return False
        try:
            model_data = joblib.load(path_to_load)
            self.lgbm_classifier = model_data['lgbm_classifier']
            self.scaler = model_data['scaler']
            self.kmeans_model = model_data.get('kmeans_model') # Load KMeans model
            
            # Restore feature names if saved (for robust reindexing)
            if 'lgbm_feature_names' in model_data and model_data['lgbm_feature_names'] is not None:
                self.lgbm_classifier.feature_name_ = model_data['lgbm_feature_names']
            if 'scaler_feature_names' in model_data and model_data['scaler_feature_names'] is not None:
                self.scaler.feature_names_in_ = model_data['scaler_feature_names']

            self.trained = True
            self.logger.info(f"Market Regime Classifier model loaded from {path_to_load}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading Market Regime Classifier model from {path_to_load}: {e}", exc_info=True)
            self.trained = False
            return False

# --- Example Usage (Main execution block for demonstration) ---
if __name__ == "__main__":
    # This block is for standalone testing of the MarketRegimeClassifier
    # In the full pipeline, it will be orchestrated by Analyst.

    # Ensure dummy data exists for loading
    from src.analyst.data_utils import create_dummy_data, load_klines_data, load_agg_trades_data, load_futures_data
    from src.config import CONFIG # Import the main CONFIG dictionary

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
    sr_analyzer_demo = SRLevelAnalyzer(CONFIG["analyst"]["sr_analyzer"])
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
    
    # Define a temporary model path for the demo
    demo_model_path = os.path.join(CONFIG['CHECKPOINT_DIR'], "analyst_models", "demo_regime_classifier.joblib")
    
    # Train and save the model using the new model_path argument
    classifier.train_classifier(historical_features_demo.copy(), klines_df.copy(), model_path=demo_model_path) # Pass copies for training

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
    # Load the model using the new model_path argument
    if new_classifier_instance.load_model(model_path=demo_model_path):
        print("Model loaded successfully. Testing prediction with loaded model:")
        current_feat = historical_features_demo.tail(1)
        current_kline = klines_df.tail(1)
        if 'ATR' not in current_kline.columns and 'ATR' in current_feat.columns:
            current_kline['ATR'] = current_feat['ATR']

        regime, trend_strength, adx = new_classifier_instance.predict_regime(current_feat, current_kline, sr_levels_demo)
        print(f"Loaded Model Prediction: Timestamp: {current_feat.index[0]} | Price: {current_feat['close'].iloc[0]:.2f} | Predicted Regime: {regime} | Trend Strength: {trend_strength:.2f} | ADX: {adx:.2f}")
    else:
        print("Failed to load model for testing.")
