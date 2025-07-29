import logging
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import pywt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam # Import Adam optimizer
from src.utils.logger import system_logger
from src.config import CONFIG # Import CONFIG to get checkpoint paths

class FeatureEngineeringEngine:
    """
    Provides a richer feature set for all downstream models.
    Now includes checkpointing for the autoencoder model and scaler, and enhanced error handling.
    """
    def __init__(self, config):
        self.config = config.get("analyst", {}).get("feature_engineering", {})
        self.logger = system_logger.getChild('FeatureEngineeringEngine')
        self.autoencoder_model = None
        self.autoencoder_scaler = None
        
        # Use the new checkpoint directory for model storage
        self.model_storage_path = os.path.join(CONFIG['CHECKPOINT_DIR'], "analyst_models", "feature_engineering")
        os.makedirs(self.model_storage_path, exist_ok=True)
        
        self.autoencoder_model_path = os.path.join(self.model_storage_path, "autoencoder_model.h5")
        self.autoencoder_scaler_path = os.path.join(self.model_storage_path, "autoencoder_scaler.joblib")

    def generate_all_features(self, klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame, futures_df: pd.DataFrame, sr_levels: list):
        """
        Orchestrates the generation of all raw and engineered features for the Analyst.
        Includes robust error handling for each feature calculation step.
        """
        if klines_df.empty:
            self.logger.warning("Feature generation: klines_df is empty, cannot generate features. Returning empty DataFrame.")
            return pd.DataFrame()

        self.logger.info("Generating all features...")
        
        features_df = klines_df.copy() # Start with klines_df and add features to it

        try:
            # Merge klines with futures data first
            features_df = pd.merge_asof(features_df.sort_index(), futures_df.sort_index(), left_index=True, right_index=True, direction='backward').fillna(method='ffill').fillna(0)
        except Exception as e:
            self.logger.error(f"Error merging klines with futures data: {e}. Proceeding without merged futures data.", exc_info=True)
            # If merge fails, ensure futures-related columns are initialized to 0/None
            for col in ['openInterest', 'fundingRate']:
                if col not in features_df.columns:
                    features_df[col] = 0.0


        # 1. Standard Technical Indicators
        try:
            self._calculate_technical_indicators(features_df)
        except Exception as e:
            self.logger.error(f"Error calculating standard technical indicators: {e}. Some indicators may be missing.", exc_info=True)

        # 2. Advanced Volume & Volatility Indicators
        try:
            self._calculate_volume_volatility_indicators(features_df, agg_trades_df)
        except Exception as e:
            self.logger.error(f"Error calculating advanced volume & volatility indicators: {e}. Some indicators may be missing.", exc_info=True)

        # 3. Wavelet Transforms
        try:
            wavelet_features = self.apply_wavelet_transforms(features_df['close'])
            features_df = features_df.join(wavelet_features, how='left')
        except Exception as e:
            self.logger.error(f"Error applying wavelet transforms: {e}. Wavelet features will be missing.", exc_info=True)
            # Add placeholder columns if wavelet features failed
            for col_name in [f'{features_df["close"].name}_wavelet_approx'] + [f'{features_df["close"].name}_wavelet_detail_{i+1}' for i in range(3)]: # Assuming level=3
                if col_name not in features_df.columns:
                    features_df[col_name] = np.nan


        # 4. S/R Interaction Features
        try:
            sr_interaction_features = self._calculate_sr_interaction_types(features_df, sr_levels, self.config.get("proximity_multiplier", 0.25))
            features_df = features_df.join(sr_interaction_features, how='left')
            features_df['Is_SR_Interacting'] = (features_df['Is_SR_Support_Interacting'] | features_df['Is_SR_Resistance_Interacting']).astype(int)
        except Exception as e:
            self.logger.error(f"Error calculating S/R interaction features: {e}. S/R interaction features will be missing.", exc_info=True)
            # Add placeholder columns if S/R features failed
            for col_name in ['Is_SR_Support_Interacting', 'Is_SR_Resistance_Interacting', 'Is_SR_Interacting']:
                if col_name not in features_df.columns:
                    features_df[col_name] = 0 # Default to no interaction


        # Fill any NaNs that might have resulted from feature calculation before autoencoder
        features_df.fillna(method='ffill', inplace=True)
        features_df.fillna(0, inplace=True) # Final fill for leading NaNs

        # 5. Autoencoder features
        autoencoder_input_features_list = [
            col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'original_close', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'] and not col.startswith('wavelet')
        ]
        existing_autoencoder_input_features = [col for col in autoencoder_input_features_list if col in features_df.columns]

        if existing_autoencoder_input_features:
            try:
                autoencoder_features = self.apply_autoencoders(features_df[existing_autoencoder_input_features])
                features_df = features_df.join(autoencoder_features.to_frame(), how='left')
            except Exception as e:
                self.logger.error(f"Error applying Autoencoders: {e}. Autoencoder features will be missing.", exc_info=True)
                features_df['autoencoder_reconstruction_error'] = np.nan # Add column with NaNs
        else:
            self.logger.warning("No suitable features found for autoencoder input. Skipping autoencoder application.")
            features_df['autoencoder_reconstruction_error'] = np.nan # Add column with NaNs

        features_df.fillna(0, inplace=True) # Final fill for any remaining NaNs
        self.logger.info("Feature generation complete.")
        return features_df

    def _calculate_technical_indicators(self, df: pd.DataFrame):
        # Ensure df is not empty before applying TA
        if df.empty:
            self.logger.warning("DataFrame is empty for technical indicator calculation.")
            return

        # Wrap each TA-Lib call in a try-except
        try:
            df.ta.adx(length=self.config.get("adx_period", 14), append=True, col_names=('ADX', 'DMP', 'DMN'))
        except Exception as e:
            self.logger.warning(f"Failed to calculate ADX: {e}. ADX features may be NaN.", exc_info=True)
            df['ADX'] = np.nan # Ensure column exists even if calculation fails
            df['DMP'] = np.nan
            df['DMN'] = np.nan

        try:
            df.ta.macd(append=True, fast=self.config.get("macd_fast_period", 12), slow=self.config.get("macd_slow_period", 26), signal=self.config.get("macd_signal_period", 9), col_names=('MACD', 'MACD_HIST', 'MACD_SIGNAL'))
        except Exception as e:
            self.logger.warning(f"Failed to calculate MACD: {e}. MACD features may be NaN.", exc_info=True)
            df['MACD'] = np.nan
            df['MACD_HIST'] = np.nan
            df['MACD_SIGNAL'] = np.nan

        try:
            df.ta.rsi(length=self.config.get("rsi_period", 14), append=True, col_names=('rsi',))
        except Exception as e:
            self.logger.warning(f"Failed to calculate RSI: {e}. RSI features may be NaN.", exc_info=True)
            df['rsi'] = np.nan

        try:
            df.ta.stoch(length=self.config.get("stoch_period", 14), append=True, col_names=('stoch_k', 'stoch_d'))
        except Exception as e:
            self.logger.warning(f"Failed to calculate Stochastic: {e}. Stochastic features may be NaN.", exc_info=True)
            df['stoch_k'] = np.nan
            df['stoch_d'] = np.nan

        try:
            df.ta.bbands(length=self.config.get("bb_period", 20), append=True, col_names=('bb_lower', 'bb_mid', 'bb_upper', 'bb_bandwidth', 'bb_percent'))
        except Exception as e:
            self.logger.warning(f"Failed to calculate Bollinger Bands: {e}. BB features may be NaN.", exc_info=True)
            df['bb_lower'] = np.nan
            df['bb_mid'] = np.nan
            df['bb_upper'] = np.nan
            df['bb_bandwidth'] = np.nan
            df['bb_percent'] = np.nan

        try:
            df['ATR'] = df.ta.atr(length=self.config.get("atr_period", 14))
        except Exception as e:
            self.logger.warning(f"Failed to calculate ATR: {e}. ATR feature may be NaN.", exc_info=True)
            df['ATR'] = np.nan


    def _calculate_volume_volatility_indicators(self, df: pd.DataFrame, agg_trades_df: pd.DataFrame):
        self.logger.info("Calculating advanced volume and volatility indicators...")
        if df.empty:
            self.logger.warning("DataFrame is empty for volume/volatility indicator calculation.")
            return

        try:
            df.ta.obv(append=True, col_names=('OBV',))
        except Exception as e:
            self.logger.warning(f"Failed to calculate OBV: {e}. OBV feature may be NaN.", exc_info=True)
            df['OBV'] = np.nan
        
        try:
            df.ta.cmf(length=self.config.get("cmf_period", 20), append=True, col_names=('CMF',))
        except Exception as e:
            self.logger.warning(f"Failed to calculate CMF: {e}. CMF feature may be NaN.", exc_info=True)
            df['CMF'] = np.nan

        try:
            df.ta.kc(length=self.config.get("kc_period", 20), append=True, col_names=('KC_lower', 'KC_mid', 'KC_upper'))
        except Exception as e:
            self.logger.warning(f"Failed to calculate Keltner Channels: {e}. KC features may be NaN.", exc_info=True)
            df['KC_lower'] = np.nan
            df['KC_mid'] = np.nan
            df['KC_upper'] = np.nan


        if not agg_trades_df.empty:
            try:
                agg_trades_df['price_x_quantity'] = agg_trades_df['price'] * agg_trades_df['quantity']
                resample_interval = self.config.get('resample_interval', '1T')
                vwap_data = agg_trades_df.resample(resample_interval).agg({
                    'price_x_quantity': 'sum',
                    'quantity': 'sum'
                }).dropna()
                
                # Handle division by zero for VWAP
                vwap_data['VWAP'] = vwap_data['price_x_quantity'] / vwap_data['quantity'].replace(0, np.nan)
                df['VWAP'] = vwap_data['VWAP'].reindex(df.index, method='ffill')
                
                # Handle division by zero for price_vs_vwap
                df['price_vs_vwap'] = (df['close'] - df['VWAP']) / df['VWAP'].replace(0, np.nan)

                agg_trades_df['delta'] = agg_trades_df['quantity'] * np.where(agg_trades_df['is_buyer_maker'], -1, 1)
                df['volume_delta'] = agg_trades_df['delta'].resample(resample_interval).sum().reindex(df.index, fill_value=0)
            except Exception as e:
                self.logger.error(f"Error calculating VWAP or Volume Delta from agg_trades: {e}. VWAP/VolumeDelta features will be missing.", exc_info=True)
                df['VWAP'] = np.nan
                df['price_vs_vwap'] = np.nan
                df['volume_delta'] = np.nan
        else:
            self.logger.warning("Aggregated trades DataFrame is empty. VWAP/VolumeDelta features will be defaulted.")
            df['VWAP'] = np.nan
            df['price_vs_vwap'] = np.nan
            df['volume_delta'] = np.nan


    def apply_wavelet_transforms(self, data: pd.Series, wavelet='db1', level=3):
        if data.empty: 
            self.logger.warning("Input data for wavelet transform is empty.")
            return pd.DataFrame(index=data.index)
        
        data_clean = data.dropna().values
        if len(data_clean) < 2**level:
            self.logger.warning(f"Insufficient data ({len(data_clean)}) for wavelet transform at level {level}. Need at least {2**level}. Returning NaN features.")
            features = pd.DataFrame(index=data.index)
            features[f'{data.name}_wavelet_approx'] = np.nan
            for i in range(level):
                features[f'{data.name}_wavelet_detail_{i+1}'] = np.nan
            return features

        try:
            coeffs = pywt.wavedec(data_clean, wavelet, level=level)
            
            features = pd.DataFrame(index=data.dropna().index)
            
            # Reconstruct and align approximation coefficients
            approx_reconstructed = pywt.waverec([coeffs[0]] + [None] * level, wavelet)
            features[f'{data.name}_wavelet_approx'] = approx_reconstructed[:len(data_clean)]
            
            # Reconstruct and align detail coefficients
            for i, detail_coeff in enumerate(coeffs[1:]):
                detail_reconstructed = pywt.waverec([None] * (i + 1) + [detail_coeff] + [None] * (level - i - 1), wavelet)
                features[f'{data.name}_wavelet_detail_{i+1}'] = detail_reconstructed[:len(data_clean)]
            
            return features.reindex(data.index)
        except Exception as e:
            self.logger.error(f"Error during wavelet transform: {e}. Returning NaN features.", exc_info=True)
            features = pd.DataFrame(index=data.index)
            features[f'{data.name}_wavelet_approx'] = np.nan
            for i in range(level):
                features[f'{data.name}_wavelet_detail_{i+1}'] = np.nan
            return features


    def _calculate_sr_interaction_types(self, df: pd.DataFrame, sr_levels: list, proximity_multiplier: float) -> pd.DataFrame:
        df['Is_SR_Support_Interacting'] = 0
        df['Is_SR_Resistance_Interacting'] = 0
        df['Is_SR_Interacting'] = 0 # Initialize here to ensure column exists
        
        if df.empty:
            self.logger.warning("DataFrame is empty for S/R interaction calculation.")
            return df[['Is_SR_Support_Interacting', 'Is_SR_Resistance_Interacting', 'Is_SR_Interacting']]

        if not sr_levels:
            self.logger.info("No S/R levels provided for interaction calculation.")
            return df[['Is_SR_Support_Interacting', 'Is_SR_Resistance_Interacting', 'Is_SR_Interacting']]
        
        # Ensure 'ATR' and 'close' exist and are numeric
        if 'ATR' not in df.columns or 'close' not in df.columns:
            self.logger.warning("Missing 'ATR' or 'close' column for S/R interaction calculation. Cannot calculate.")
            return df[['Is_SR_Support_Interacting', 'Is_SR_Resistance_Interacting', 'Is_SR_Interacting']]

        df['ATR_filled'] = df['ATR'].fillna(method='ffill').fillna(0)
        is_support_interacting = pd.Series(False, index=df.index)
        is_resistance_interacting = pd.Series(False, index=df.index)
        
        try:
            for level_info in sr_levels:
                level_price = level_info.get('level_price')
                level_type = level_info.get('type')

                if level_price is None or level_type is None:
                    self.logger.warning(f"Invalid S/R level info: {level_info}. Skipping.")
                    continue

                atr_based_tolerance = df['ATR_filled'] * proximity_multiplier
                min_tolerance = df['close'] * 0.001 # Min 0.1% of price
                max_tolerance = df['close'] * 0.005 # Max 0.5% of price
                
                # Handle potential division by zero or NaN in tolerance calculation
                tolerance = np.clip(atr_based_tolerance, min_tolerance, max_tolerance).fillna(0)

                interaction_condition = (df['low'] <= level_price + tolerance) & (df['high'] >= level_price - tolerance)
                
                if level_type == "Support": 
                    is_support_interacting = is_support_interacting | interaction_condition
                elif level_type == "Resistance": 
                    is_resistance_interacting = is_resistance_interacting | interaction_condition
            
            df['Is_SR_Support_Interacting'] = is_support_interacting.astype(int)
            df['Is_SR_Resistance_Interacting'] = is_resistance_interacting.astype(int)
            df['Is_SR_Interacting'] = (df['Is_SR_Support_Interacting'] | df['Is_SR_Resistance_Interacting'])
            df.drop(columns=['ATR_filled'], inplace=True, errors='ignore')
        except Exception as e:
            self.logger.error(f"Error in S/R interaction calculation loop: {e}. S/R interaction features will be defaulted.", exc_info=True)
            df['Is_SR_Support_Interacting'] = 0
            df['Is_SR_Resistance_Interacting'] = 0
            df['Is_SR_Interacting'] = 0

        return df[['Is_SR_Support_Interacting', 'Is_SR_Resistance_Interacting', 'Is_SR_Interacting']]

    def train_autoencoder(self, data: pd.DataFrame):
        self.logger.info(f"Autoencoder training: Input data shape {data.shape}...")
        
        if data.empty:
            self.logger.warning("No data provided for autoencoder training. Skipping.")
            return

        # Drop rows with any NaN values before scaling and training
        data_cleaned = data.dropna()
        if data_cleaned.empty:
            self.logger.warning("Data is empty after dropping NaNs for autoencoder training. Skipping.")
            return

        self.autoencoder_scaler = StandardScaler()
        scaled_data = self.autoencoder_scaler.fit_transform(data_cleaned)
        
        if scaled_data.shape[0] == 0 or scaled_data.shape[1] == 0:
            self.logger.warning("Scaled data is empty or has 0 columns. Cannot train autoencoder.")
            return

        input_dim = scaled_data.shape[1]
        latent_dim = self.config.get("autoencoder_latent_dim", 16)
        
        if input_dim == 0:
            self.logger.warning("Input dimension for autoencoder is 0. Skipping training.")
            return

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(latent_dim, activation="relu")(input_layer)
        decoder = Dense(input_dim, activation="linear")(encoder)
        self.autoencoder_model = Model(inputs=input_layer, outputs=decoder)
        self.autoencoder_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        self.logger.info(f"Training autoencoder with input dim {input_dim}, latent dim {latent_dim}...")
        try:
            # Use data_cleaned's index for aligning scaled_data back to original if needed
            self.autoencoder_model.fit(scaled_data, scaled_data, epochs=10, batch_size=32, shuffle=True, verbose=0)
            
            self.autoencoder_model.save(self.autoencoder_model_path)
            joblib.dump(self.autoencoder_scaler, self.autoencoder_scaler_path)
            self.logger.info("Autoencoder model and scaler saved successfully.")
        except Exception as e:
            self.logger.error(f"Error during Autoencoder training or saving: {e}", exc_info=True)
            self.autoencoder_model = None # Reset model on failure
            self.autoencoder_scaler = None


    def apply_autoencoders(self, data: pd.DataFrame):
        if self.autoencoder_model is None or self.autoencoder_scaler is None:
            if not self.load_autoencoder():
                self.logger.warning("Autoencoder not loaded. Returning NaN for reconstruction error.")
                return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')
        
        if data.empty:
            self.logger.warning("Input data for autoencoder application is empty.")
            return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')

        try:
            # Ensure data columns match scaler's fitted features
            if hasattr(self.autoencoder_scaler, 'feature_names_in_'):
                data_reindexed = data.reindex(columns=self.autoencoder_scaler.feature_names_in_, fill_value=0)
            else:
                self.logger.warning("Scaler does not have 'feature_names_in_'. Assuming column order is consistent.")
                data_reindexed = data

            # Handle cases where data_reindexed might be empty after reindexing
            if data_reindexed.empty:
                self.logger.warning("Data is empty after reindexing for autoencoder application.")
                return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')

            scaled_data = self.autoencoder_scaler.transform(data_reindexed)
            
            if scaled_data.shape[1] == 0:
                self.logger.warning("Scaled data has 0 columns. Cannot apply autoencoder.")
                return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')

            reconstructions = self.autoencoder_model.predict(scaled_data, verbose=0)
            mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
            return pd.Series(mse, index=data.index, name='autoencoder_reconstruction_error')
        except Exception as e:
            self.logger.error(f"Error applying Autoencoder: {e}", exc_info=True)
            return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')

    def load_autoencoder(self):
        if os.path.exists(self.autoencoder_model_path) and os.path.exists(self.autoencoder_scaler_path):
            try:
                self.autoencoder_model = load_model(self.autoencoder_model_path)
                self.autoencoder_scaler = joblib.load(self.autoencoder_scaler_path)
                self.logger.info("Autoencoder model and scaler loaded.")
                return True
            except Exception as e:
                self.logger.error(f"Error loading Autoencoder from {self.autoencoder_model_path} or {self.autoencoder_scaler_path}: {e}", exc_info=True)
        return False
