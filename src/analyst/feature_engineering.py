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
from src.utils.logger import system_logger

class FeatureEngineeringEngine:
    """
    Provides a richer feature set for all downstream models.
    """
    def __init__(self, config):
        self.config = config.get("analyst", {}).get("feature_engineering", {})
        self.logger = system_logger.getChild('FeatureEngineeringEngine')
        self.autoencoder_model = None
        self.autoencoder_scaler = None
        self.model_storage_path = self.config.get("model_storage_path", "models/analyst/feature_engineering/")
        os.makedirs(self.model_storage_path, exist_ok=True)
        self.autoencoder_model_path = os.path.join(self.model_storage_path, "autoencoder_model.h5")
        self.autoencoder_scaler_path = os.path.join(self.model_storage_path, "autoencoder_scaler.joblib")

    def generate_all_features(self, klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame, futures_df: pd.DataFrame, sr_levels: list):
        """
        Orchestrates the generation of all raw and engineered features for the Analyst.
        """
        if klines_df.empty:
            self.logger.warning("Feature generation: klines_df is empty, cannot generate features.")
            return pd.DataFrame()

        self.logger.info("Generating all features...")
        # Merge klines with futures data first
        features_df = pd.merge_asof(klines_df.sort_index(), futures_df.sort_index(), left_index=True, right_index=True, direction='backward').fillna(method='ffill').fillna(0)

        # 1. Standard Technical Indicators
        self._calculate_technical_indicators(features_df)
        
        # 2. Advanced Volume & Volatility Indicators
        self._calculate_volume_volatility_indicators(features_df, agg_trades_df)

        # 3. Wavelet Transforms
        wavelet_features = self.apply_wavelet_transforms(features_df['close'])
        features_df = features_df.join(wavelet_features, how='left')

        # 4. S/R Interaction Features
        sr_interaction_features = self._calculate_sr_interaction_types(features_df, sr_levels, self.config.get("proximity_multiplier", 0.25))
        features_df = features_df.join(sr_interaction_features, how='left')
        features_df['Is_SR_Interacting'] = (features_df['Is_SR_Support_Interacting'] | features_df['Is_SR_Resistance_Interacting']).astype(int)

        features_df = features_df.fillna(method='ffill').fillna(0)

        # 5. Autoencoder features
        autoencoder_input_features = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'original_close'] and 'wavelet' not in col]
        autoencoder_features = self.apply_autoencoders(features_df[autoencoder_input_features])
        features_df = features_df.join(autoencoder_features.to_frame(), how='left')

        features_df.fillna(0, inplace=True)
        self.logger.info("Feature generation complete.")
        return features_df

    def _calculate_technical_indicators(self, df: pd.DataFrame):
        df.ta.adx(length=self.config.get("adx_period", 14), append=True, col_names=('ADX', 'DMP', 'DMN'))
        df.ta.macd(append=True, fast=self.config.get("macd_fast_period", 12), slow=self.config.get("macd_slow_period", 26), signal=self.config.get("macd_signal_period", 9), col_names=('MACD', 'MACD_HIST', 'MACD_SIGNAL'))
        df.ta.rsi(length=self.config.get("rsi_period", 14), append=True, col_names=('rsi',))
        df.ta.stoch(length=self.config.get("stoch_period", 14), append=True, col_names=('stoch_k', 'stoch_d'))
        df.ta.bbands(length=self.config.get("bb_period", 20), append=True, col_names=('bb_lower', 'bb_mid', 'bb_upper', 'bb_bandwidth', 'bb_percent'))
        df['ATR'] = df.ta.atr(length=self.config.get("atr_period", 14))

    def _calculate_volume_volatility_indicators(self, df: pd.DataFrame, agg_trades_df: pd.DataFrame):
        self.logger.info("Calculating advanced volume and volatility indicators...")
        # On-Balance Volume (OBV)
        df.ta.obv(append=True, col_names=('OBV',))
        
        # Chaikin Money Flow (CMF)
        df.ta.cmf(length=self.config.get("cmf_period", 20), append=True, col_names=('CMF',))

        # Keltner Channels
        df.ta.kc(length=self.config.get("kc_period", 20), append=True, col_names=('KC_lower', 'KC_mid', 'KC_upper'))

        # VWAP requires tick-level data, so we calculate it from agg_trades
        if not agg_trades_df.empty:
            agg_trades_df['price_x_quantity'] = agg_trades_df['price'] * agg_trades_df['quantity']
            # Resample to the kline interval
            resample_interval = self.config.get('resample_interval', '1T') # Default to 1 minute
            vwap_data = agg_trades_df.resample(resample_interval).agg({
                'price_x_quantity': 'sum',
                'quantity': 'sum'
            }).dropna()
            vwap_data['VWAP'] = vwap_data['price_x_quantity'] / vwap_data['quantity']
            df['VWAP'] = vwap_data['VWAP'].reindex(df.index, method='ffill')
            df['price_vs_vwap'] = (df['close'] - df['VWAP']) / df['VWAP'] # Feature: Price distance from VWAP

            # Volume Delta
            agg_trades_df['delta'] = agg_trades_df['quantity'] * np.where(agg_trades_df['is_buyer_maker'], -1, 1)
            df['volume_delta'] = agg_trades_df['delta'].resample(resample_interval).sum().reindex(df.index, fill_value=0)
        else:
            df['VWAP'] = df['close'] # Fallback if no agg trades
            df['price_vs_vwap'] = 0
            df['volume_delta'] = 0

    def apply_wavelet_transforms(self, data: pd.Series, wavelet='db1', level=3):
        if data.empty: return pd.DataFrame(index=data.index)
        data_clean = data.dropna().values
        if len(data_clean) < 2**level: return pd.DataFrame(index=data.index)
        coeffs = pywt.wavedec(data_clean, wavelet, level=level)
        approx_reconstructed = pywt.waverec([coeffs[0]] + [None] * level, wavelet)[:len(data_clean)]
        features = pd.DataFrame({f'{data.name}_wavelet_approx': approx_reconstructed}, index=data.dropna().index)
        for i, detail_coeff in enumerate(coeffs[1:]):
            detail_reconstructed = pywt.waverec([None] * (i + 1) + [detail_coeff] + [None] * (level - i - 1), wavelet)[:len(data_clean)]
            features[f'{data.name}_wavelet_detail_{i+1}'] = detail_reconstructed
        return features.reindex(data.index)

    def _calculate_sr_interaction_types(self, df: pd.DataFrame, sr_levels: list, proximity_multiplier: float) -> pd.DataFrame:
        df['Is_SR_Support_Interacting'] = 0
        df['Is_SR_Resistance_Interacting'] = 0
        if not sr_levels:
            df['Is_SR_Interacting'] = 0
            return df[['Is_SR_Support_Interacting', 'Is_SR_Resistance_Interacting', 'Is_SR_Interacting']]
        df['ATR_filled'] = df['ATR'].fillna(method='ffill').fillna(0)
        is_support_interacting = pd.Series(False, index=df.index)
        is_resistance_interacting = pd.Series(False, index=df.index)
        for level_info in sr_levels:
            level_price = level_info['level_price']
            atr_based_tolerance = df['ATR_filled'] * proximity_multiplier
            min_tolerance = df['close'] * 0.01
            max_tolerance = df['close'] * 0.02
            tolerance = np.clip(atr_based_tolerance, min_tolerance, max_tolerance)
            interaction_condition = (df['low'] <= level_price + tolerance) & (df['high'] >= level_price - tolerance)
            if level_info['type'] == "Support": is_support_interacting = is_support_interacting | interaction_condition
            elif level_info['type'] == "Resistance": is_resistance_interacting = is_resistance_interacting | interaction_condition
        df['Is_SR_Support_Interacting'] = is_support_interacting.astype(int)
        df['Is_SR_Resistance_Interacting'] = is_resistance_interacting.astype(int)
        df['Is_SR_Interacting'] = (df['Is_SR_Support_Interacting'] | df['Is_SR_Resistance_Interacting'])
        df.drop(columns=['ATR_filled'], inplace=True, errors='ignore')
        return df[['Is_SR_Support_Interacting', 'Is_SR_Resistance_Interacting', 'Is_SR_Interacting']]

    def train_autoencoder(self, data: pd.DataFrame):
        self.logger.info(f"Autoencoder training: Input data shape {data.shape}...")
        self.autoencoder_scaler = StandardScaler()
        scaled_data = self.autoencoder_scaler.fit_transform(data.dropna())
        input_dim = scaled_data.shape[1]
        latent_dim = self.config.get("autoencoder_latent_dim", 16)
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(latent_dim, activation="relu")(input_layer)
        decoder = Dense(input_dim, activation="linear")(encoder)
        self.autoencoder_model = Model(inputs=input_layer, outputs=decoder)
        self.autoencoder_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.autoencoder_model.fit(scaled_data, scaled_data, epochs=10, batch_size=32, shuffle=True, verbose=0)
        self.autoencoder_model.save(self.autoencoder_model_path)
        joblib.dump(self.autoencoder_scaler, self.autoencoder_scaler_path)

    def apply_autoencoders(self, data: pd.DataFrame):
        if self.autoencoder_model is None:
            if not self.load_autoencoder():
                return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')
        try:
            if hasattr(self.autoencoder_scaler, 'feature_names_in_'):
                data_reindexed = data.reindex(columns=self.autoencoder_scaler.feature_names_in_, fill_value=0)
            else:
                data_reindexed = data
            scaled_data = self.autoencoder_scaler.transform(data_reindexed)
            reconstructions = self.autoencoder_model.predict(scaled_data, verbose=0)
            mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
            return pd.Series(mse, index=data.index, name='autoencoder_reconstruction_error')
        except Exception as e:
            self.logger.error(f"Error applying Autoencoder: {e}")
            return pd.Series(np.nan, index=data.index, name='autoencoder_reconstruction_error')

    def load_autoencoder(self):
        if os.path.exists(self.autoencoder_model_path) and os.path.exists(self.autoencoder_scaler_path):
            try:
                self.autoencoder_model = load_model(self.autoencoder_model_path)
                self.autoencoder_scaler = joblib.load(self.autoencoder_scaler_path)
                return True
            except Exception as e:
                self.logger.error(f"Error loading Autoencoder: {e}")
        return False
