import os
import joblib
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam  # Import Adam optimizer
from src.utils.logger import system_logger
from src.config import CONFIG  # Import CONFIG to get checkpoint paths
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_file_operations,
)
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple


class FeatureEngineeringEngine:
    """
    Provides a richer feature set for all downstream models.
    Now includes checkpointing for the autoencoder model and scaler, and enhanced error handling.
    """

    def __init__(self, config):
        self.config = config.get("analyst", {}).get("feature_engineering", {})
        self.logger = system_logger.getChild("FeatureEngineeringEngine")
        self.autoencoder_model = None
        self.autoencoder_scaler = None

        # Use the new checkpoint directory for model storage
        self.model_storage_path = os.path.join(
            CONFIG["CHECKPOINT_DIR"], "analyst_models", "feature_engineering"
        )
        os.makedirs(self.model_storage_path, exist_ok=True)

        self.autoencoder_model_path = os.path.join(
            self.model_storage_path, "autoencoder_model.h5"
        )
        self.autoencoder_scaler_path = os.path.join(
            self.model_storage_path, "autoencoder_scaler.joblib"
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="generate_all_features",
    )
    def generate_all_features(
        self,
        klines_df: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
        futures_df: pd.DataFrame,
        sr_levels: list,
    ):
        """
        Orchestrates the generation of all raw and engineered features for the Analyst.
        Includes robust error handling for each feature calculation step.
        """
        if klines_df.empty:
            self.logger.warning(
                "Feature generation: klines_df is empty, cannot generate features. Returning empty DataFrame."
            )
            return pd.DataFrame()

        self.logger.info("Generating all features...")

        features_df = klines_df.copy()  # Start with klines_df and add features to it

        try:
            # Merge klines with futures data first
            features_df = (
                pd.merge_asof(
                    features_df.sort_index(),
                    futures_df.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="backward",
                )
                .fillna(method="ffill")
                .fillna(0)
            )
        except Exception as e:
            self.logger.error(
                f"Error merging klines with futures data: {e}. Proceeding without merged futures data.",
                exc_info=True,
            )
            # If merge fails, ensure futures-related columns are initialized to 0/None
            for col in ["fundingRate"]:
                if col not in features_df.columns:
                    features_df[col] = 0.0

        # 1. Standard Technical Indicators
        try:
            self._calculate_technical_indicators(features_df)
        except Exception as e:
            self.logger.error(
                f"Error calculating standard technical indicators: {e}. Some indicators may be missing.",
                exc_info=True,
            )

        # 2. Advanced Volume & Volatility Indicators
        try:
            self._calculate_volume_volatility_indicators(features_df, agg_trades_df)
        except Exception as e:
            self.logger.error(
                f"Error calculating advanced volume & volatility indicators: {e}. Some indicators may be missing.",
                exc_info=True,
            )

        # 2b. Enhanced Volume Indicators (VROC, OBV Divergence)
        try:
            self._calculate_enhanced_volume_indicators(features_df)
        except Exception as e:
            self.logger.error(
                f"Error calculating enhanced volume indicators: {e}. Some indicators may be missing.",
                exc_info=True,
            )

        # 2c. Order Flow and Liquidity Indicators
        try:
            self._calculate_order_flow_indicators(features_df, agg_trades_df)
        except Exception as e:
            self.logger.error(
                f"Error calculating order flow indicators: {e}. Some indicators may be missing.",
                exc_info=True,
            )

        # 2d. Enhanced Funding Rate Analysis
        try:
            self._calculate_enhanced_funding_features(features_df)
        except Exception as e:
            self.logger.error(
                f"Error calculating enhanced funding features: {e}. Some indicators may be missing.",
                exc_info=True,
            )

        # 2e. ML-Enhanced Feature Engineering
        try:
            self._calculate_ml_enhanced_features(features_df)
        except Exception as e:
            self.logger.error(
                f"Error calculating ML-enhanced features: {e}. Some indicators may be missing.",
                exc_info=True,
            )

        # 2f. Time-Based Features
        try:
            self._calculate_time_features(features_df)
        except Exception as e:
            self.logger.error(
                f"Error calculating time-based features: {e}. Some indicators may be missing.",
                exc_info=True,
            )

        # 2g. Volatility Targeting Features
        try:
            target_vol = self.config.get("target_volatility", 0.15)
            self._calculate_volatility_targeting_features(features_df, target_vol)
        except Exception as e:
            self.logger.error(
                f"Error calculating volatility targeting features: {e}. Some indicators may be missing.",
                exc_info=True,
            )

        # 3. Wavelet Transforms
        try:
            wavelet_features = self.apply_wavelet_transforms(features_df["close"])
            features_df = features_df.join(wavelet_features, how="left")
        except Exception as e:
            self.logger.error(
                f"Error applying wavelet transforms: {e}. Wavelet features will be missing.",
                exc_info=True,
            )
            # Add placeholder columns if wavelet features failed
            for col_name in [f"{features_df['close'].name}_wavelet_approx"] + [
                f"{features_df['close'].name}_wavelet_detail_{i + 1}" for i in range(3)
            ]:  # Assuming level=3
                if col_name not in features_df.columns:
                    features_df[col_name] = np.nan

        # 4. S/R Interaction Features
        try:
            sr_interaction_features = self._calculate_sr_interaction_types(
                features_df, sr_levels, self.config.get("proximity_multiplier", 0.25)
            )
            # Use suffix to avoid column overlap
            features_df = features_df.join(sr_interaction_features, how="left", rsuffix="_sr")
            # Remove suffix from column names if they were added
            for col in sr_interaction_features.columns:
                if f"{col}_sr" in features_df.columns:
                    features_df[col] = features_df[f"{col}_sr"]
                    features_df.drop(columns=[f"{col}_sr"], inplace=True)
            
            features_df["Is_SR_Interacting"] = (
                features_df["Is_SR_Support_Interacting"]
                | features_df["Is_SR_Resistance_Interacting"]
            ).astype(int)
        except Exception as e:
            self.logger.error(
                f"Error calculating S/R interaction features: {e}. S/R interaction features will be missing.",
                exc_info=True,
            )
            # Add placeholder columns if S/R features failed
            for col_name in [
                "Is_SR_Support_Interacting",
                "Is_SR_Resistance_Interacting",
                "Is_SR_Interacting",
            ]:
                if col_name not in features_df.columns:
                    features_df[col_name] = 0  # Default to no interaction

        # Fill any NaNs that might have resulted from feature calculation before autoencoder
        features_df.fillna(method="ffill", inplace=True)
        features_df.fillna(0, inplace=True)  # Final fill for leading NaNs

        # 5. Autoencoder features
        autoencoder_input_features_list = [
            col
            for col in features_df.columns
            if col
            not in [
                "open",
                "high",
                "low",
                "close",
                "original_close",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ]
            and not col.startswith("wavelet")
        ]
        existing_autoencoder_input_features = [
            col for col in autoencoder_input_features_list if col in features_df.columns
        ]

        if existing_autoencoder_input_features:
            try:
                # Check if autoencoder exists, if not train it automatically
                autoencoder_enabled = self.config.get("autoencoder_enabled", True)
                if autoencoder_enabled and not self.load_autoencoder():
                    self.logger.info("Autoencoder not found. Training new autoencoder automatically...")
                    training_data = features_df[existing_autoencoder_input_features].dropna()
                    if not training_data.empty:
                        if self.train_autoencoder(training_data):
                            self.logger.info("Autoencoder trained successfully!")
                        else:
                            self.logger.warning("Failed to train autoencoder. Will use NaN values.")
                    else:
                        self.logger.warning("No valid training data for autoencoder. Will use NaN values.")
                elif not autoencoder_enabled:
                    self.logger.info("Autoencoder is disabled in configuration. Skipping autoencoder training.")
                
                # Now apply the autoencoder
                autoencoder_features = self.apply_autoencoders(
                    features_df[existing_autoencoder_input_features]
                )
                features_df = features_df.join(
                    autoencoder_features.to_frame(), how="left"
                )
            except Exception as e:
                self.logger.error(
                    f"Error applying Autoencoders: {e}. Autoencoder features will be missing.",
                    exc_info=True,
                )
                features_df["autoencoder_reconstruction_error"] = (
                    np.nan
                )  # Add column with NaNs
        else:
            self.logger.warning(
                "No suitable features found for autoencoder input. Skipping autoencoder application."
            )
            features_df["autoencoder_reconstruction_error"] = (
                np.nan
            )  # Add column with NaNs

        features_df.fillna(0, inplace=True)  # Final fill for any remaining NaNs
        self.logger.info("Feature generation complete.")
        return features_df

    @handle_data_processing_errors(
        default_return=None, context="calculate_technical_indicators"
    )
    def _calculate_technical_indicators(self, df: pd.DataFrame):
        # Ensure df is not empty before applying TA
        if df.empty:
            self.logger.warning(
                "DataFrame is empty for technical indicator calculation."
            )
            return

        # Wrap each TA-Lib call in a try-except
        try:
            df.ta.adx(
                length=self.config.get("adx_period", 14),
                append=True,
                col_names=("ADX", "DMP", "DMN"),
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to calculate ADX: {e}. ADX features may be NaN.", exc_info=True
            )
            df["ADX"] = np.nan  # Ensure column exists even if calculation fails
            df["DMP"] = np.nan
            df["DMN"] = np.nan

        try:
            df.ta.macd(
                append=True,
                fast=self.config.get("macd_fast_period", 12),
                slow=self.config.get("macd_slow_period", 26),
                signal=self.config.get("macd_signal_period", 9),
                col_names=("MACD", "MACD_HIST", "MACD_SIGNAL"),
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to calculate MACD: {e}. MACD features may be NaN.",
                exc_info=True,
            )
            df["MACD"] = np.nan
            df["MACD_HIST"] = np.nan
            df["MACD_SIGNAL"] = np.nan

        try:
            df.ta.rsi(
                length=self.config.get("rsi_period", 14),
                append=True,
                col_names=("rsi",),
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to calculate RSI: {e}. RSI features may be NaN.", exc_info=True
            )
            df["rsi"] = np.nan

        try:
            df.ta.stoch(
                length=self.config.get("stoch_period", 14),
                append=True,
                col_names=("stoch_k", "stoch_d"),
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to calculate Stochastic: {e}. Stochastic features may be NaN.",
                exc_info=True,
            )
            df["stoch_k"] = np.nan
            df["stoch_d"] = np.nan

        try:
            df.ta.bbands(
                length=self.config.get("bb_period", 20),
                append=True,
                col_names=(
                    "bb_lower",
                    "bb_mid",
                    "bb_upper",
                    "bb_bandwidth",
                    "bb_percent",
                ),
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to calculate Bollinger Bands: {e}. BB features may be NaN.",
                exc_info=True,
            )
            df["bb_lower"] = np.nan
            df["bb_mid"] = np.nan
            df["bb_upper"] = np.nan
            df["bb_bandwidth"] = np.nan
            df["bb_percent"] = np.nan

        try:
            df["ATR"] = df.ta.atr(length=self.config.get("atr_period", 14))
        except Exception as e:
            self.logger.warning(
                f"Failed to calculate ATR: {e}. ATR feature may be NaN.", exc_info=True
            )
            df["ATR"] = np.nan

    @handle_data_processing_errors(
        default_return=None, context="calculate_volume_volatility_indicators"
    )
    def _calculate_volume_volatility_indicators(
        self, df: pd.DataFrame, agg_trades_df: pd.DataFrame
    ):
        self.logger.info("Calculating advanced volume and volatility indicators...")
        if df.empty:
            self.logger.warning(
                "DataFrame is empty for volume/volatility indicator calculation."
            )
            return

        try:
            df.ta.obv(append=True, col_names=("OBV",))
        except Exception as e:
            self.logger.warning(
                f"Failed to calculate OBV: {e}. OBV feature may be NaN.", exc_info=True
            )
            df["OBV"] = np.nan

        try:
            df.ta.cmf(
                length=self.config.get("cmf_period", 20),
                append=True,
                col_names=("CMF",),
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to calculate CMF: {e}. CMF feature may be NaN.", exc_info=True
            )
            df["CMF"] = np.nan

        try:
            df.ta.kc(
                length=self.config.get("kc_period", 20),
                append=True,
                col_names=("KC_lower", "KC_mid", "KC_upper"),
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to calculate Keltner Channels: {e}. KC features may be NaN.",
                exc_info=True,
            )
            df["KC_lower"] = np.nan
            df["KC_mid"] = np.nan
            df["KC_upper"] = np.nan

        if not agg_trades_df.empty:
            try:
                agg_trades_df["price_x_quantity"] = (
                    agg_trades_df["price"] * agg_trades_df["quantity"]
                )
                resample_interval = self.config.get("resample_interval", "1T")
                vwap_data = (
                    agg_trades_df.resample(resample_interval)
                    .agg({"price_x_quantity": "sum", "quantity": "sum"})
                    .dropna()
                )

                # Handle division by zero for VWAP
                vwap_data["VWAP"] = vwap_data["price_x_quantity"] / vwap_data[
                    "quantity"
                ].replace(0, np.nan)
                df["VWAP"] = vwap_data["VWAP"].reindex(df.index, method="ffill")

                # Handle division by zero for price_vs_vwap
                df["price_vs_vwap"] = (df["close"] - df["VWAP"]) / df["VWAP"].replace(
                    0, np.nan
                )

                agg_trades_df["delta"] = agg_trades_df["quantity"] * np.where(
                    agg_trades_df["is_buyer_maker"], -1, 1
                )
                df["volume_delta"] = (
                    agg_trades_df["delta"]
                    .resample(resample_interval)
                    .sum()
                    .reindex(df.index, fill_value=0)
                )
            except Exception as e:
                self.logger.error(
                    f"Error calculating VWAP or Volume Delta from agg_trades: {e}. VWAP/VolumeDelta features will be missing.",
                    exc_info=True,
                )
                df["VWAP"] = np.nan
                df["price_vs_vwap"] = np.nan
                df["volume_delta"] = np.nan
        else:
            self.logger.warning(
                "Aggregated trades DataFrame is empty. VWAP/VolumeDelta features will be defaulted."
            )
            df["VWAP"] = np.nan
            df["price_vs_vwap"] = np.nan
            df["volume_delta"] = np.nan

    @handle_data_processing_errors(
        default_return=pd.DataFrame(), context="apply_wavelet_transforms"
    )
    def apply_wavelet_transforms(self, data: pd.Series, wavelet="db1", level=3):
        if data.empty:
            self.logger.warning("Input data for wavelet transform is empty.")
            return pd.DataFrame(index=data.index)

        data_clean = data.dropna().values
        if len(data_clean) < 2**level:
            self.logger.warning(
                f"Insufficient data ({len(data_clean)}) for wavelet transform at level {level}. Need at least {2**level}. Returning NaN features."
            )
            features = pd.DataFrame(index=data.index)
            features[f"{data.name}_wavelet_approx"] = np.nan
            for i in range(level):
                features[f"{data.name}_wavelet_detail_{i + 1}"] = np.nan
            return features

        try:
            coeffs = pywt.wavedec(data_clean, wavelet, level=level)

            features = pd.DataFrame(index=data.dropna().index)

            # Reconstruct and align approximation coefficients
            approx_reconstructed = pywt.waverec([coeffs[0]] + [None] * level, wavelet)
            features[f"{data.name}_wavelet_approx"] = approx_reconstructed[
                : len(data_clean)
            ]

            # Reconstruct and align detail coefficients
            for i, detail_coeff in enumerate(coeffs[1:]):
                # Create proper coefficient structure for reconstruction
                # We need to provide the full coefficient structure
                coeff_list = [coeffs[0]]  # Start with approximation
                for j in range(1, len(coeffs)):
                    if j == i + 1:  # This is our detail coefficient
                        coeff_list.append(detail_coeff)
                    else:
                        coeff_list.append(None)
                detail_reconstructed = pywt.waverec(coeff_list, wavelet)
                features[f"{data.name}_wavelet_detail_{i + 1}"] = detail_reconstructed[
                    : len(data_clean)
                ]

            return features.reindex(data.index)
        except Exception as e:
            self.logger.error(
                f"Error during wavelet transform: {e}. Returning NaN features.",
                exc_info=True,
            )
            features = pd.DataFrame(index=data.index)
            features[f"{data.name}_wavelet_approx"] = np.nan
            for i in range(level):
                features[f"{data.name}_wavelet_detail_{i + 1}"] = np.nan
            return features

    @handle_data_processing_errors(
        default_return=pd.DataFrame(), context="calculate_sr_interaction_types"
    )
    def _calculate_sr_interaction_types(
        self, df: pd.DataFrame, sr_levels: list, proximity_multiplier: float
    ) -> pd.DataFrame:
        df["Is_SR_Support_Interacting"] = 0
        df["Is_SR_Resistance_Interacting"] = 0
        df["Is_SR_Interacting"] = 0  # Initialize here to ensure column exists

        if df.empty:
            self.logger.warning("DataFrame is empty for S/R interaction calculation.")
            return df[
                [
                    "Is_SR_Support_Interacting",
                    "Is_SR_Resistance_Interacting",
                    "Is_SR_Interacting",
                ]
            ]

        if not sr_levels:
            self.logger.info("No S/R levels provided for interaction calculation.")
            return df[
                [
                    "Is_SR_Support_Interacting",
                    "Is_SR_Resistance_Interacting",
                    "Is_SR_Interacting",
                ]
            ]

        # Ensure 'ATR' and 'close' exist and are numeric
        if "ATR" not in df.columns or "close" not in df.columns:
            self.logger.warning(
                "Missing 'ATR' or 'close' column for S/R interaction calculation. Cannot calculate."
            )
            return df[
                [
                    "Is_SR_Support_Interacting",
                    "Is_SR_Resistance_Interacting",
                    "Is_SR_Interacting",
                ]
            ]

        df["ATR_filled"] = df["ATR"].fillna(method="ffill").fillna(0)
        is_support_interacting = pd.Series(False, index=df.index)
        is_resistance_interacting = pd.Series(False, index=df.index)

        try:
            for level_info in sr_levels:
                level_price = level_info.get("level_price")
                level_type = level_info.get("type")

                if level_price is None or level_type is None:
                    self.logger.warning(
                        f"Invalid S/R level info: {level_info}. Skipping."
                    )
                    continue

                atr_based_tolerance = df["ATR_filled"] * proximity_multiplier
                min_tolerance = df["close"] * 0.001  # Min 0.1% of price
                max_tolerance = df["close"] * 0.005  # Max 0.5% of price

                # Handle potential division by zero or NaN in tolerance calculation
                tolerance = np.clip(
                    atr_based_tolerance, min_tolerance, max_tolerance
                ).fillna(0)

                interaction_condition = (df["low"] <= level_price + tolerance) & (
                    df["high"] >= level_price - tolerance
                )

                if level_type == "Support":
                    is_support_interacting = (
                        is_support_interacting | interaction_condition
                    )
                elif level_type == "Resistance":
                    is_resistance_interacting = (
                        is_resistance_interacting | interaction_condition
                    )

            df["Is_SR_Support_Interacting"] = is_support_interacting.astype(int)
            df["Is_SR_Resistance_Interacting"] = is_resistance_interacting.astype(int)
            df["Is_SR_Interacting"] = (
                df["Is_SR_Support_Interacting"] | df["Is_SR_Resistance_Interacting"]
            )
            df.drop(columns=["ATR_filled"], inplace=True, errors="ignore")

            return df[
                [
                    "Is_SR_Support_Interacting",
                    "Is_SR_Resistance_Interacting",
                    "Is_SR_Interacting",
                ]
            ]
        except Exception as e:
            self.logger.error(
                f"Error during S/R interaction calculation: {e}. Returning default values.",
                exc_info=True,
            )
            return df[
                [
                    "Is_SR_Support_Interacting",
                    "Is_SR_Resistance_Interacting",
                    "Is_SR_Interacting",
                ]
            ]

    @handle_file_operations(default_return=False, context="train_autoencoder")
    def train_autoencoder(self, data: pd.DataFrame):
        self.logger.info(f"Autoencoder training: Input data shape {data.shape}...")

        if data.empty:
            self.logger.warning("No data provided for autoencoder training. Skipping.")
            return False

        # Drop rows with any NaN values before scaling and training
        data_cleaned = data.dropna()
        if data_cleaned.empty:
            self.logger.warning(
                "Data is empty after dropping NaNs for autoencoder training. Skipping."
            )
            return False

        self.autoencoder_scaler = StandardScaler()
        scaled_data = self.autoencoder_scaler.fit_transform(data_cleaned)

        if scaled_data.shape[0] == 0 or scaled_data.shape[1] == 0:
            self.logger.warning(
                "Scaled data is empty or has 0 columns. Cannot train autoencoder."
            )
            return False

        input_dim = scaled_data.shape[1]
        latent_dim = self.config.get("autoencoder_latent_dim", 16)

        if input_dim == 0:
            self.logger.warning(
                "Input dimension for autoencoder is 0. Skipping training."
            )
            return False

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(latent_dim, activation="relu")(input_layer)
        decoder = Dense(input_dim, activation="linear")(encoder)
        self.autoencoder_model = Model(inputs=input_layer, outputs=decoder)
        self.autoencoder_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        self.logger.info(
            f"Training autoencoder with input dim {input_dim}, latent dim {latent_dim}..."
        )
        try:
            # Use data_cleaned's index for aligning scaled_data back to original if needed
            self.autoencoder_model.fit(
                scaled_data,
                scaled_data,
                epochs=10,
                batch_size=32,
                shuffle=True,
                verbose=0,
            )

            self.autoencoder_model.save(self.autoencoder_model_path)
            joblib.dump(self.autoencoder_scaler, self.autoencoder_scaler_path)
            self.logger.info("Autoencoder model and scaler saved successfully.")
            return True
        except Exception as e:
            self.logger.error(
                f"Error during Autoencoder training or saving: {e}", exc_info=True
            )
            self.autoencoder_model = None  # Reset model on failure
            self.autoencoder_scaler = None
            return False

    @handle_data_processing_errors(
        default_return=pd.Series(), context="apply_autoencoders"
    )
    def apply_autoencoders(self, data: pd.DataFrame):
        if self.autoencoder_model is None or self.autoencoder_scaler is None:
            self.logger.warning(
                "Autoencoder not available. Returning NaN for reconstruction error."
            )
            return pd.Series(
                np.nan, index=data.index, name="autoencoder_reconstruction_error"
            )

        if data.empty:
            self.logger.warning("Input data for autoencoder application is empty.")
            return pd.Series(
                np.nan, index=data.index, name="autoencoder_reconstruction_error"
            )

        try:
            # Ensure data columns match scaler's fitted features
            if hasattr(self.autoencoder_scaler, "feature_names_in_"):
                data_reindexed = data.reindex(
                    columns=self.autoencoder_scaler.feature_names_in_, fill_value=0
                )
            else:
                self.logger.warning(
                    "Scaler does not have 'feature_names_in_'. Assuming column order is consistent."
                )
                data_reindexed = data

            # Handle cases where data_reindexed might be empty after reindexing
            if data_reindexed.empty:
                self.logger.warning(
                    "Data is empty after reindexing for autoencoder application."
                )
                return pd.Series(
                    np.nan, index=data.index, name="autoencoder_reconstruction_error"
                )

            scaled_data = self.autoencoder_scaler.transform(data_reindexed)

            if scaled_data.shape[1] == 0:
                self.logger.warning(
                    "Scaled data has 0 columns. Cannot apply autoencoder."
                )
                return pd.Series(
                    np.nan, index=data.index, name="autoencoder_reconstruction_error"
                )

            reconstructions = self.autoencoder_model.predict(scaled_data, verbose=0)
            mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
            return pd.Series(
                mse, index=data.index, name="autoencoder_reconstruction_error"
            )
        except Exception as e:
            self.logger.error(f"Error applying Autoencoder: {e}", exc_info=True)
            return pd.Series(
                np.nan, index=data.index, name="autoencoder_reconstruction_error"
            )

    @handle_file_operations(default_return=False, context="load_autoencoder")
    def load_autoencoder(self):
        if os.path.exists(self.autoencoder_model_path) and os.path.exists(
            self.autoencoder_scaler_path
        ):
            try:
                self.autoencoder_model = load_model(self.autoencoder_model_path)
                self.autoencoder_scaler = joblib.load(self.autoencoder_scaler_path)
                self.logger.info("Autoencoder model and scaler loaded.")
                return True
            except Exception as e:
                self.logger.error(
                    f"Error loading Autoencoder from {self.autoencoder_model_path} or {self.autoencoder_scaler_path}: {e}",
                    exc_info=True,
                )
        return False

    @handle_data_processing_errors(
        default_return=None, context="calculate_enhanced_volume_indicators"
    )
    def _calculate_enhanced_volume_indicators(self, df: pd.DataFrame):
        """Calculate enhanced volume indicators: VROC and OBV Divergence."""
        self.logger.info("Calculating enhanced volume indicators...")

        if df.empty or "volume" not in df.columns:
            self.logger.warning(
                "DataFrame is empty or missing volume data for enhanced volume indicators."
            )
            return

        # Volume Rate of Change (VROC)
        try:
            vroc_period = self.config.get("vroc_period", 25)
            df["VROC"] = (
                (df["volume"] - df["volume"].shift(vroc_period))
                / df["volume"].shift(vroc_period).replace(0, np.nan)
            ) * 100
        except Exception as e:
            self.logger.warning(f"Failed to calculate VROC: {e}")
            df["VROC"] = np.nan

        # On-Balance Volume (OBV) Divergence
        try:
            if "OBV" in df.columns:
                obv_divergence_period = self.config.get("obv_divergence_period", 14)
                price_change = df["close"].pct_change(obv_divergence_period)
                obv_change = df["OBV"].pct_change(obv_divergence_period)
                df["OBV_Divergence"] = price_change - obv_change
            else:
                df["OBV_Divergence"] = np.nan
        except Exception as e:
            self.logger.warning(f"Failed to calculate OBV Divergence: {e}")
            df["OBV_Divergence"] = np.nan

    @handle_data_processing_errors(
        default_return=None, context="calculate_order_flow_indicators"
    )
    def _calculate_order_flow_indicators(
        self, df: pd.DataFrame, agg_trades_df: pd.DataFrame
    ):
        """Calculate order flow and liquidity indicators."""
        self.logger.info("Calculating order flow and liquidity indicators...")

        if df.empty:
            self.logger.warning("DataFrame is empty for order flow indicators.")
            return

        # Initialize default values
        df["Buy_Sell_Pressure_Ratio"] = 0.0
        df["Order_Flow_Imbalance"] = 0.0
        df["Large_Order_Count"] = 0
        df["Liquidity_Score"] = 0.0

        if not agg_trades_df.empty and "is_buyer_maker" in agg_trades_df.columns:
            try:
                resample_interval = self.config.get("resample_interval", "1T")

                # Order Flow Analysis
                agg_trades_df["buy_volume"] = np.where(
                    ~agg_trades_df["is_buyer_maker"],
                    agg_trades_df["quantity"],
                    0,
                )
                agg_trades_df["sell_volume"] = np.where(
                    agg_trades_df["is_buyer_maker"],
                    agg_trades_df["quantity"],
                    0,
                )

                # Resample and calculate ratios
                flow_data = (
                    agg_trades_df.resample(resample_interval)
                    .agg(
                        {
                            "buy_volume": "sum",
                            "sell_volume": "sum",
                            "quantity": ["mean", "count"],
                        }
                    )
                    .fillna(0)
                )

                # Buy/Sell Pressure Ratio
                total_volume = (
                    flow_data[("buy_volume", "sum")] + flow_data[("sell_volume", "sum")]
                )
                df["Buy_Sell_Pressure_Ratio"] = np.where(
                    total_volume > 0,
                    flow_data[("buy_volume", "sum")] / total_volume,
                    0.5,
                )

                # Order Flow Imbalance
                df["Order_Flow_Imbalance"] = np.where(
                    total_volume > 0,
                    (
                        flow_data[("buy_volume", "sum")]
                        - flow_data[("sell_volume", "sum")]
                    )
                    / total_volume,
                    0.0,
                )

                # Large Order Detection
                # avg_trade_size = flow_data[("quantity", "mean")]

                # Count large orders per period
                agg_trades_df["is_large_order"] = (
                    agg_trades_df["quantity"]
                    > agg_trades_df["quantity"].rolling(100).mean() * 2
                )
                large_order_counts = (
                    agg_trades_df["is_large_order"].resample(resample_interval).sum()
                )
                df["Large_Order_Count"] = large_order_counts.reindex(
                    df.index, fill_value=0
                )

                # Simple Liquidity Score (based on trade count and volume)
                trade_counts = flow_data[("quantity", "count")]
                df["Liquidity_Score"] = (trade_counts * total_volume).fillna(0)

                # Reindex to match main DataFrame
                # Ensure the flow_data has the same index as the main DataFrame
                flow_data_reindexed = flow_data.reindex(df.index, method="ffill").fillna(0)
                
                df["Buy_Sell_Pressure_Ratio"] = np.where(
                    flow_data_reindexed[("buy_volume", "sum")] + flow_data_reindexed[("sell_volume", "sum")] > 0,
                    flow_data_reindexed[("buy_volume", "sum")] / (flow_data_reindexed[("buy_volume", "sum")] + flow_data_reindexed[("sell_volume", "sum")]),
                    0.5,
                )
                df["Order_Flow_Imbalance"] = np.where(
                    flow_data_reindexed[("buy_volume", "sum")] + flow_data_reindexed[("sell_volume", "sum")] > 0,
                    (flow_data_reindexed[("buy_volume", "sum")] - flow_data_reindexed[("sell_volume", "sum")]) / (flow_data_reindexed[("buy_volume", "sum")] + flow_data_reindexed[("sell_volume", "sum")]),
                    0.0,
                )
                df["Liquidity_Score"] = flow_data_reindexed[("quantity", "count")] * (flow_data_reindexed[("buy_volume", "sum")] + flow_data_reindexed[("sell_volume", "sum")])

            except Exception as e:
                self.logger.error(f"Error calculating order flow indicators: {e}")

    @handle_data_processing_errors(
        default_return=None, context="calculate_enhanced_funding_features"
    )
    def _calculate_enhanced_funding_features(self, df: pd.DataFrame):
        """Calculate enhanced funding rate features."""
        self.logger.info("Calculating enhanced funding rate features...")

        if df.empty or "fundingRate" not in df.columns:
            self.logger.warning("DataFrame is empty or missing funding rate data.")
            # Initialize default values
            df["Funding_Momentum"] = 0.0
            df["Funding_Divergence"] = 0.0
            df["Funding_Extreme"] = 0.0
            return

        try:
            # Funding Rate Momentum
            funding_momentum_period = self.config.get("funding_momentum_period", 3)
            df["Funding_Momentum"] = df["fundingRate"].diff(funding_momentum_period)

            # Funding Rate Divergence
            price_change = df["close"].pct_change(funding_momentum_period)
            df["Funding_Divergence"] = df["Funding_Momentum"] - price_change

            # Funding Rate Extremes (Z-score)
            funding_window = self.config.get("funding_window", 24)
            funding_mean = df["fundingRate"].rolling(funding_window).mean()
            funding_std = df["fundingRate"].rolling(funding_window).std()
            df["Funding_Extreme"] = (
                df["fundingRate"] - funding_mean
            ) / funding_std.replace(0, np.nan)

        except Exception as e:
            self.logger.warning(f"Failed to calculate enhanced funding features: {e}")
            df["Funding_Momentum"] = 0.0
            df["Funding_Divergence"] = 0.0
            df["Funding_Extreme"] = 0.0

    @handle_data_processing_errors(
        default_return=None, context="calculate_ml_enhanced_features"
    )
    def _calculate_ml_enhanced_features(self, df: pd.DataFrame):
        """Calculate ML-enhanced feature engineering extensions."""
        self.logger.info("Calculating ML-enhanced features...")

        if df.empty:
            self.logger.warning("DataFrame is empty for ML-enhanced features.")
            return

        try:
            # Price Action Patterns
            momentum_period = self.config.get("price_momentum_period", 5)
            df["Price_Momentum"] = df["close"].pct_change(momentum_period)
            df["Price_Acceleration"] = df["Price_Momentum"].diff()

            # Volume Patterns
            df["Volume_Momentum"] = df["volume"].pct_change(momentum_period)
            df["Volume_Acceleration"] = df["Volume_Momentum"].diff()

            # Volatility Patterns
            if "ATR" in df.columns:
                df["Volatility_Momentum"] = df["ATR"].pct_change(momentum_period)
            else:
                df["Volatility_Momentum"] = np.nan

            # Cross-Indicator Features
            if "rsi" in df.columns and "MACD" in df.columns:
                df["RSI_MACD_Divergence"] = df["rsi"] - df["MACD"]
            else:
                df["RSI_MACD_Divergence"] = np.nan

            df["Volume_Price_Divergence"] = df["Volume_Momentum"] - df["Price_Momentum"]

            # Price vs various MAs ratios
            for ma_period in [9, 21, 50, 200]:
                ma_col = f"SMA_{ma_period}"
                if ma_col in df.columns:
                    df[f"Price_SMA_{ma_period}_Ratio"] = df["close"] / df[
                        ma_col
                    ].replace(0, np.nan)

            # Volatility regime indicator
            if "ATR" in df.columns:
                atr_sma = df["ATR"].rolling(20).mean()
                df["Volatility_Regime"] = df["ATR"] / atr_sma.replace(0, np.nan)
            else:
                df["Volatility_Regime"] = np.nan

        except Exception as e:
            self.logger.warning(f"Failed to calculate ML-enhanced features: {e}")

    @handle_data_processing_errors(
        default_return=None, context="calculate_time_features"
    )
    def _calculate_time_features(self, df: pd.DataFrame):
        """Calculate time-based features for session dynamics."""
        self.logger.info("Calculating time-based features...")

        if df.empty:
            self.logger.warning("DataFrame is empty for time-based features.")
            return

        try:
            # Ensure we have a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df.index = pd.to_datetime(df["timestamp"])
                else:
                    self.logger.warning(
                        "No datetime index or timestamp column found for time features"
                    )
                    return

            # Time of Day
            df["Hour"] = df.index.hour
            df["Minute"] = df.index.minute

            # Day of Week
            df["Day_of_Week"] = df.index.dayofweek

            # Session Indicators (UTC-based)
            df["Is_London_Session"] = ((df["Hour"] >= 8) & (df["Hour"] < 16)).astype(
                int
            )
            df["Is_NY_Session"] = ((df["Hour"] >= 13) & (df["Hour"] < 21)).astype(int)
            df["Is_Asia_Session"] = ((df["Hour"] >= 0) & (df["Hour"] < 8)).astype(int)

            # Weekend Effect
            df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)

            # Hour sine/cosine encoding (for cyclical nature)
            df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
            df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

            # Day of week sine/cosine encoding
            df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["Day_of_Week"] / 7)
            df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["Day_of_Week"] / 7)

            # Market overlap indicators
            df["London_NY_Overlap"] = (
                df["Is_London_Session"] & df["Is_NY_Session"]
            ).astype(int)
            df["Asia_London_Overlap"] = (
                df["Is_Asia_Session"] & df["Is_London_Session"]
            ).astype(int)

        except Exception as e:
            self.logger.warning(f"Failed to calculate time-based features: {e}")

    @handle_data_processing_errors(
        default_return=None, context="calculate_volatility_regime_indicators"
    )
    def _calculate_volatility_regime_indicators(self, df: pd.DataFrame):
        """Calculate volatility regime detection indicators."""
        self.logger.info("Calculating volatility regime indicators...")

        if df.empty:
            self.logger.warning("DataFrame is empty for volatility regime indicators.")
            return

        try:
            # Realized Volatility
            returns = df["close"].pct_change()
            vol_window = self.config.get("volatility_window", 24)
            realized_vol = returns.rolling(vol_window).std() * np.sqrt(vol_window)
            df["Realized_Volatility"] = realized_vol

            # Volatility Regime Classification
            vol_quantiles = realized_vol.quantile([0.25, 0.5, 0.75])

            def classify_vol_regime(vol):
                if pd.isna(vol):
                    return "UNKNOWN"
                elif vol <= vol_quantiles[0.25]:
                    return "LOW"
                elif vol <= vol_quantiles[0.5]:
                    return "MEDIUM"
                elif vol <= vol_quantiles[0.75]:
                    return "HIGH"
                else:
                    return "EXTREME"

            df["Volatility_Regime_Label"] = realized_vol.apply(classify_vol_regime)

            # Encode regime as numeric
            regime_mapping = {
                "LOW": 1,
                "MEDIUM": 2,
                "HIGH": 3,
                "EXTREME": 4,
                "UNKNOWN": 0,
            }
            df["Volatility_Regime_Numeric"] = df["Volatility_Regime_Label"].map(
                regime_mapping
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to calculate volatility regime indicators: {e}"
            )
            df["Realized_Volatility"] = np.nan
            df["Volatility_Regime_Label"] = "UNKNOWN"
            df["Volatility_Regime_Numeric"] = 0

    @handle_data_processing_errors(
        default_return=None, context="calculate_volatility_targeting_features"
    )
    def _calculate_volatility_targeting_features(
        self, df: pd.DataFrame, target_volatility: float = 0.15
    ):
        """
        Calculate volatility targeting features for dynamic position sizing.

        Args:
            df: DataFrame with price data
            target_volatility: Target annual volatility (default 15%)
        """
        self.logger.info("Calculating volatility targeting features...")

        if df.empty:
            self.logger.warning("DataFrame is empty for volatility targeting features.")
            return

        try:
            # Calculate various volatility measures
            returns = df["close"].pct_change()

            # 1. Simple Historical Volatility (annualized)
            simple_vol_period = self.config.get("simple_vol_period", 20)
            simple_vol = returns.rolling(simple_vol_period).std() * np.sqrt(252)
            df["Simple_Volatility"] = simple_vol

            # 2. EWMA Volatility (Exponentially Weighted Moving Average)
            ewma_span = self.config.get("ewma_span", 20)
            ewma_vol = returns.ewm(span=ewma_span).std() * np.sqrt(252)
            df["EWMA_Volatility"] = ewma_vol

            # 3. GARCH-like volatility (simplified)
            garch_period = self.config.get("garch_period", 30)
            squared_returns = returns**2
            garch_vol = squared_returns.rolling(garch_period).mean().apply(
                np.sqrt
            ) * np.sqrt(252)
            df["GARCH_Volatility"] = garch_vol

            # 4. Parkinson volatility (high-low estimator)
            if all(col in df.columns for col in ["high", "low"]):
                parkinson_vol = np.sqrt(
                    (np.log(df["high"] / df["low"]) ** 2)
                    .rolling(simple_vol_period)
                    .mean()
                    * 252
                    / (4 * np.log(2))
                )
                df["Parkinson_Volatility"] = parkinson_vol
            else:
                df["Parkinson_Volatility"] = np.nan

            # 5. Calculate volatility targeting multipliers
            max_leverage = self.config.get("max_leverage", 3.0)
            min_leverage = self.config.get("min_leverage", 0.1)

            # Simple volatility targeting
            df["Vol_Target_Multiplier_Simple"] = np.clip(
                target_volatility / simple_vol.replace(0, np.nan),
                min_leverage,
                max_leverage,
            )

            # EWMA volatility targeting
            df["Vol_Target_Multiplier_EWMA"] = np.clip(
                target_volatility / ewma_vol.replace(0, np.nan),
                min_leverage,
                max_leverage,
            )

            # Parkinson volatility targeting
            if not df["Parkinson_Volatility"].isna().all():
                df["Vol_Target_Multiplier_Parkinson"] = np.clip(
                    target_volatility / df["Parkinson_Volatility"].replace(0, np.nan),
                    min_leverage,
                    max_leverage,
                )
            else:
                df["Vol_Target_Multiplier_Parkinson"] = 1.0

            # 6. Adaptive volatility targeting with momentum filter
            momentum_period = self.config.get("momentum_period", 10)
            price_momentum = df["close"].pct_change(momentum_period)

            # Reduce exposure during negative momentum (risk-off)
            momentum_factor = np.where(price_momentum > 0, 1.0, 0.7)
            df["Vol_Target_Multiplier_Adaptive"] = (
                df["Vol_Target_Multiplier_EWMA"] * momentum_factor
            ).clip(min_leverage, max_leverage)

            # 7. Volatility regime detection for targeting
            # vol_percentiles = simple_vol.rolling(252).quantile([0.25, 0.5, 0.75])
            df["Vol_Regime_Quantile"] = simple_vol.rolling(252).rank(pct=True)

            # Adjust targeting based on regime
            regime_adjustment = np.where(
                df["Vol_Regime_Quantile"] > 0.8,
                0.5,  # High vol regime - reduce exposure
                np.where(
                    df["Vol_Regime_Quantile"] < 0.2, 1.5, 1.0
                ),  # Low vol regime - increase exposure
            )

            df["Vol_Target_Multiplier_Regime"] = (
                df["Vol_Target_Multiplier_Simple"] * regime_adjustment
            ).clip(min_leverage, max_leverage)

            # 8. Kelly Criterion enhancement
            # Estimate return/volatility ratio for Kelly-based sizing
            rolling_return = returns.rolling(simple_vol_period).mean() * 252
            kelly_ratio = rolling_return / (simple_vol**2)
            kelly_fraction = np.clip(kelly_ratio, 0, 0.25)  # Cap at 25% Kelly

            df["Kelly_Fraction"] = kelly_fraction
            df["Vol_Target_Multiplier_Kelly"] = (
                df["Vol_Target_Multiplier_Simple"] * (1 + kelly_fraction)
            ).clip(min_leverage, max_leverage)

            # 9. Multi-timeframe volatility
            short_vol = returns.rolling(5).std() * np.sqrt(252)
            medium_vol = returns.rolling(20).std() * np.sqrt(252)
            long_vol = returns.rolling(60).std() * np.sqrt(252)

            df["Short_Term_Vol"] = short_vol
            df["Medium_Term_Vol"] = medium_vol
            df["Long_Term_Vol"] = long_vol

            # Volatility term structure signal
            df["Vol_Term_Structure"] = (short_vol - long_vol) / long_vol.replace(
                0, np.nan
            )

            # 10. Dynamic target adjustment
            # Adjust target based on market conditions
            market_stress_indicator = df.get("Order_Flow_Imbalance", 0).abs()
            stress_adjustment = np.where(market_stress_indicator > 0.3, 0.8, 1.0)

            adjusted_target = target_volatility * stress_adjustment
            df["Dynamic_Target_Vol"] = adjusted_target
            df["Vol_Target_Multiplier_Dynamic"] = np.clip(
                adjusted_target / ewma_vol.replace(0, np.nan),
                min_leverage,
                max_leverage,
            )

            # Fill NaN values
            vol_cols = [
                col
                for col in df.columns
                if "Vol_Target_Multiplier" in col or "Volatility" in col
            ]
            for col in vol_cols:
                df[col] = df[col].fillna(1.0)

        except Exception as e:
            self.logger.warning(
                f"Failed to calculate volatility targeting features: {e}"
            )
            # Initialize default values
            df["Simple_Volatility"] = 0.15
            df["EWMA_Volatility"] = 0.15
            df["Vol_Target_Multiplier_Simple"] = 1.0
            df["Vol_Target_Multiplier_EWMA"] = 1.0
            df["Vol_Target_Multiplier_Adaptive"] = 1.0
