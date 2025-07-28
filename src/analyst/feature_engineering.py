# src/analyst/feature_engineering.py
import pandas as pd
import numpy as np
import pywt # For Wavelet Transforms
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier # For GBM Feature Selection
# Suppress TensorFlow warnings for cleaner output in demo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from utils.logger import system_logger # Import the centralized logger


class FeatureEngineeringEngine:
    """
    Handles advanced feature generation including Wavelet Transforms and Autoencoders.
    Also provides functionality for GBM-based feature selection.
    """
    def __init__(self, config):
        self.config = config.get("analyst", {}).get("feature_engineering", {})
        self.logger = system_logger.getChild('FeatureEngineeringEngine')

        self.autoencoder_model = None 
        self.autoencoder_scaler = None
        self.gbm_selector_model = None # For GBM feature selection
        self.selected_features_names = None # Store names of selected features

    def apply_wavelet_transforms(self, data: pd.Series, wavelet='db1', level=None):
        """
        Applies Discrete Wavelet Transform (DWT) to denoise data and extract multi-scale features.
        :param data: pandas Series of price data.
        :param wavelet: Type of wavelet to use (e.g., 'db1', 'haar').
        :param level: Decomposition level. If None, uses config value.
        :return: DataFrame with original and wavelet-transformed features.
        """
        if data.empty:
            self.logger.warning("Wavelet transform: Input data is empty.")
            return pd.DataFrame(index=data.index)

        level = level if level is not None else self.config.get("wavelet_level", 3)
        
        data_clean = data.dropna().values

        # PyWavelets requires input length to be a multiple of 2^level
        # Pad data if necessary
        if len(data_clean) == 0:
            self.logger.warning("Wavelet transform: No clean data points after dropping NaNs.")
            return pd.DataFrame(index=data.index)

        min_len_for_level = 2 ** level
        if len(data_clean) < min_len_for_level:
            self.logger.warning(f"Wavelet transform: Not enough data points ({len(data_clean)}) for level {level}. Minimum required: {min_len_for_level}. Skipping wavelet transform.")
            return pd.DataFrame(index=data.index)

        # Pad to the nearest power of 2 if not already
        original_len = len(data_clean)
        pad_len = int(2**np.ceil(np.log2(original_len)))
        if original_len < pad_len:
            data_padded = np.pad(data_clean, (0, pad_len - original_len), 'edge')
        else:
            data_padded = data_clean

        coeffs = pywt.wavedec(data_padded, wavelet, level=level)
        
        features = pd.DataFrame(index=data.index)
        features[f'original_{data.name}'] = data # Add original data

        # Reconstruct components for features
        # Approximation coefficients (low frequency, trend)
        approx_coeff = coeffs[0]
        approx_reconstructed = pywt.waverec([approx_coeff] + [None] * level, wavelet)[:original_len]
        features[f'{data.name}_wavelet_approx'] = pd.Series(approx_reconstructed, index=data.dropna().index)

        # Detail coefficients (high frequency, noise/details)
        for i, detail_coeff in enumerate(coeffs[1:]):
            detail_reconstructed = pywt.waverec([None] * (i + 1) + [detail_coeff] + [None] * (level - i -1), wavelet)[:original_len]
            features[f'{data.name}_wavelet_detail_{i+1}'] = pd.Series(detail_reconstructed, index=data.dropna().index)
        
        self.logger.info(f"Applied Wavelet Transforms (wavelet={wavelet}, level={level}). Generated {len(features.columns)} features.")
        return features.reindex(data.index)

    def train_autoencoder(self, data: pd.DataFrame):
        """
        Trains an Autoencoder model for unsupervised feature learning.
        The reconstruction error can be used as a feature for anomaly detection.
        :param data: DataFrame of features to train the autoencoder on.
        """
        if data.empty:
            self.logger.warning("Autoencoder training: Input data is empty. Skipping training.")
            self.autoencoder_model = None
            self.autoencoder_scaler = None
            return

        self.logger.info(f"Autoencoder training: Input data shape {data.shape}...")
        
        # Scale data before training
        self.autoencoder_scaler = StandardScaler()
        scaled_data = self.autoencoder_scaler.fit_transform(data.dropna())

        input_dim = scaled_data.shape[1]
        latent_dim = self.config.get("autoencoder_latent_dim", 16)

        # Define the Autoencoder model
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(latent_dim, activation="relu")(input_layer)
        decoder = Dense(input_dim, activation="sigmoid")(encoder) # Sigmoid for normalized data (0-1 range)

        self.autoencoder_model = Model(inputs=input_layer, outputs=decoder)
        self.autoencoder_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # Train the model
        try:
            self.autoencoder_model.fit(scaled_data, scaled_data, epochs=10, batch_size=32, shuffle=True, verbose=0)
            self.logger.info("Autoencoder training completed.")
            # Placeholder for saving the model
            # self.autoencoder_model.save(f"{self.model_storage_path}autoencoder_model.h5")
        except Exception as e:
            self.logger.error(f"Error during Autoencoder training: {e}", exc_info=True)
            self.autoencoder_model = None # Invalidate model if training fails

    def apply_autoencoders(self, data: pd.DataFrame):
        """
        Applies a trained Autoencoder to data to extract reconstruction error as a feature.
        :param data: DataFrame of features to apply the autoencoder on.
        :return: Series of reconstruction errors.
        """
        if self.autoencoder_model is None or self.autoencoder_scaler is None:
            self.logger.warning("Autoencoder model or scaler not trained. Skipping autoencoder features.")
            return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')
        
        if data.empty:
            self.logger.warning("Autoencoder application: Input data is empty.")
            return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')

        # Ensure data columns match training data for scaler
        data_clean = data.dropna()
        if data_clean.empty:
            self.logger.warning("Autoencoder application: No clean data points after dropping NaNs.")
            return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')

        try:
            scaled_data = self.autoencoder_scaler.transform(data_clean)
            reconstructions = self.autoencoder_model.predict(scaled_data, verbose=0)
            mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
            reconstruction_error = pd.Series(mse, index=data_clean.index, name='autoencoder_reconstruction_error')
            self.logger.info(f"Applied Autoencoder. Generated {len(reconstruction_error)} reconstruction error features.")
            return reconstruction_error.reindex(data.index)
        except Exception as e:
            self.logger.error(f"Error applying Autoencoder: {e}", exc_info=True)
            return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')


    def select_features_with_gbm(self, features_df: pd.DataFrame, target_series: pd.Series):
        """
        Performs feature selection using LightGBM feature importances.
        :param features_df: DataFrame of features.
        :param target_series: Series of the target variable (must align with features_df index).
        :return: List of selected feature names.
        """
        if features_df.empty or target_series.empty:
            self.logger.warning("GBM feature selection: Input features or target is empty. Skipping selection.")
            return features_df.columns.tolist() # Return all features if cannot select

        # Align features and target
        aligned_df = features_df.join(target_series.rename('target')).dropna()
        if aligned_df.empty:
            self.logger.warning("GBM feature selection: No aligned data after dropping NaNs. Skipping selection.")
            return features_df.columns.tolist()

        X = aligned_df.drop(columns=['target'])
        y = aligned_df['target']

        if X.empty or y.empty:
            self.logger.warning("GBM feature selection: X or y is empty after alignment. Skipping selection.")
            return features_df.columns.tolist()

        self.logger.info(f"GBM feature selection: Training on {X.shape[1]} features and {len(y)} samples...")

        try:
            # Train a LightGBM classifier
            self.gbm_selector_model = LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.1, verbose=-1) # verbose=-1 suppresses training output
            self.gbm_selector_model.fit(X, y)

            # Get feature importances
            feature_importances = pd.Series(self.gbm_selector_model.feature_importances_, index=X.columns)
            
            # Select features based on threshold
            threshold = self.config.get("gbm_feature_threshold", 0.01)
            selected_features = feature_importances[feature_importances > threshold].index.tolist()

            if not selected_features:
                self.logger.warning(f"No features met the GBM importance threshold ({threshold}). Returning all features.")
                selected_features = features_df.columns.tolist()
            else:
                self.logger.info(f"GBM feature selection completed. Selected {len(selected_features)} features.")
            
            self.selected_features_names = selected_features
            return selected_features
        except Exception as e:
            self.logger.error(f"Error during GBM feature selection: {e}", exc_info=True)
            return features_df.columns.tolist() # Fallback to all features on error

    def generate_all_features(self, klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame, futures_df: pd.DataFrame, sr_levels: list):
        """
        Orchestrates the generation of all raw and engineered features for the Analyst.
        """
        if klines_df.empty:
            self.logger.warning("Feature generation: klines_df is empty, cannot generate features.")
            return pd.DataFrame()

        self.logger.info("Generating all features...")
        # Merge klines with futures data
        merged_df = pd.merge_asof(klines_df.sort_index(), futures_df.sort_index(), 
                                  left_index=True, right_index=True, direction='backward')
        merged_df.fillna(method='ffill', inplace=True)
        merged_df.fillna(0, inplace=True) # Fill any remaining NaNs, e.g., from start of data

        features_df = pd.DataFrame(index=merged_df.index)
        features_df['open'] = merged_df['open']
        features_df['high'] = merged_df['high']
        features_df['low'] = merged_df['low']
        features_df['close'] = merged_df['close']
        features_df['volume'] = merged_df['volume']
        features_df['openInterest'] = merged_df['openInterest']
        features_df['fundingRate'] = merged_df['fundingRate']

        # Add basic TA indicators (using pandas_ta)
        features_df.ta.adx(length=self.config.get("adx_period", 14), append=True, col_names=('ADX', 'DMP', 'DMN'))
        macd = features_df.ta.macd(append=False, fast=self.config.get("macd_fast_period", 12), 
                                  slow=self.config.get("macd_slow_period", 26), 
                                  signal=self.config.get("macd_signal_period", 9))
        macd_hist_col = f'MACDh_{self.config.get("macd_fast_period", 12)}_{self.config.get("macd_slow_period", 26)}_{self.config.get("macd_signal_period", 9)}'
        features_df['MACD_HIST'] = macd[macd_hist_col] if macd_hist_col in macd.columns else 0
        features_df.ta.atr(length=self.config.get("atr_period", 14), append=True, col_names=('ATR'))
        
        # Wavelet Transforms on 'close' price
        close_wavelet_features = self.apply_wavelet_transforms(merged_df['close'], level=self.config.get("wavelet_level", 3))
        if not close_wavelet_features.empty:
            features_df = features_df.merge(close_wavelet_features, left_index=True, right_index=True, how='left')

        # Autoencoder features (reconstruction error)
        # Autoencoder is trained on a subset of features (e.g., price, volume)
        # Ensure the data passed to apply_autoencoders has the same columns as used for training
        ae_input_data = features_df[['close', 'volume']].copy()
        autoencoder_features = self.apply_autoencoders(ae_input_data)
        if not autoencoder_features.empty:
            features_df = features_df.merge(autoencoder_features.to_frame(), left_index=True, right_index=True, how='left')

        # Volume Delta from agg_trades_df
        if not agg_trades_df.empty:
            agg_trades_df['delta'] = agg_trades_df['quantity'] * np.where(agg_trades_df['is_buyer_maker'], -1, 1)
            # Resample delta to klines interval (assuming 1m for now, adjust based on INTERVAL)
            # Ensure agg_trades_df index is datetime for resampling
            if not isinstance(agg_trades_df.index, pd.DatetimeIndex):
                agg_trades_df.index = pd.to_datetime(agg_trades_df.index, unit='ms') # Assuming timestamp in ms if not parsed
            
            volume_delta_resampled = agg_trades_df['delta'].resample('1min').sum().reindex(merged_df.index, fill_value=0)
            features_df['volume_delta'] = volume_delta_resampled
            # Calculate rolling mean and std for z-score
            features_df['volume_delta_mean_60'] = features_df['volume_delta'].rolling(window=60).mean()
            features_df['volume_delta_std_60'] = features_df['volume_delta'].rolling(window=60).std()
            features_df['delta_zscore'] = ((features_df['volume_delta'] - features_df['volume_delta_mean_60']) / 
                                           features_df['volume_delta_std_60'].replace(0, np.nan)).fillna(0)
        else:
            features_df['volume_delta'] = 0
            features_df['volume_delta_mean_60'] = 0
            features_df['volume_delta_std_60'] = 0
            features_df['delta_zscore'] = 0

        # S/R Interaction feature (using sr_analyzer logic)
        if sr_levels and 'ATR' in features_df.columns:
            proximity_multiplier = self.config.get("proximity_multiplier", 0.25) # From main config's BEST_PARAMS or analyst section
            # Ensure ATR is not NaN for calculation
            features_df['ATR_filled'] = features_df['ATR'].fillna(features_df['ATR'].mean() if not features_df['ATR'].empty else 0)
            proximity_threshold_series = features_df['ATR_filled'] * proximity_multiplier
            is_interacting = pd.Series(False, index=features_df.index)
            for level_info in sr_levels:
                level = level_info['level_price']
                # Check interaction for each row
                is_interacting = is_interacting | ((features_df['low'] <= level + proximity_threshold_series) & 
                                                    (features_df['high'] >= level - proximity_threshold_series))
            features_df['Is_SR_Interacting'] = is_interacting.astype(int)
        else:
            features_df['Is_SR_Interacting'] = 0

        features_df.fillna(0, inplace=True) # Fill any remaining NaNs after feature calculation
        self.logger.info("Feature generation complete.")
        return features_df
