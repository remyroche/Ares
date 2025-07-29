# src/analyst/predictive_ensembles/regime_ensembles/sr_zone_action_ensemble.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# TensorFlow/Keras imports for Deep Learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2 # For L1/L2 regularization

from .base_ensemble import BaseEnsemble
from src.analyst.data_utils import calculate_volume_profile # Import for volume profile features

class SRZoneActionEnsemble(BaseEnsemble):
    """
    Predictive Ensemble specifically designed for SR_ZONE_ACTION market regimes.
    Focuses on reaction at S/R levels, volume, and order flow.
    """
    def __init__(self, config: dict, ensemble_name: str):
        super().__init__(config, ensemble_name)
        self.models = {
            "volume_profile": None, # LGBM model for volume profile features
            "order_flow": None, # Order flow analysis model (conceptual)
            "lgbm": None # LightGBM for general patterns
        }
        self.ensemble_weights = self.config.get("ensemble_weights", {
            "volume_profile": 0.35, "order_flow": 0.35, "lgbm": 0.3
        })
        self.min_confluence_confidence = self.config.get("min_confluence_confidence", 0.7)

        # DL model specific configurations (if any for this ensemble)
        self.dl_config = {
            "dense_units": 64, # Units for any simple dense network
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32,
            "l1_reg_strength": self.config.get("meta_learner_l1_reg", 0.001),
            "l2_reg_strength": self.config.get("meta_learner_l2_reg", 0.001),
        }
        # UPDATED: Added S/R interaction type features
        self.expected_dl_features = ['close', 'volume', 'ADX', 'MACD_HIST', 'ATR', 'volume_delta', 
                                     'autoencoder_reconstruction_error', 'Is_SR_Interacting',
                                     'Is_SR_Support_Interacting', 'Is_SR_Resistance_Interacting']
        self.dl_target_map = {} # To store target label to integer mapping for DL models
        self.volume_profile_features_list = [] # To store feature names for volume_profile model
        self.order_flow_target_map = {} # To store target label to integer mapping for order flow model

    def _prepare_flat_data(self, df: pd.DataFrame, target_series: pd.Series = None):
        """
        Prepares flat data for models.
        Ensures consistent feature columns and handles missing data.
        """
        features = df[self.expected_dl_features].copy()
        features = features.fillna(0) # Fill NaNs

        if target_series is not None:
            aligned_df = features.join(target_series.rename('target')).dropna()
            X = aligned_df.drop(columns=['target'])
            y = aligned_df['target']
            return X, y
        else:
            return features, None

    def _derive_volume_profile_features(self, klines_df: pd.DataFrame, current_price: float) -> pd.Series:
        """
        Derives features from the volume profile relative to the current price.
        """
        if klines_df.empty:
            return pd.Series(dtype='float64')

        vp_data = calculate_volume_profile(klines_df)
        poc = vp_data["poc"]
        hvn_levels = vp_data["hvn_levels"]
        lvn_levels = vp_data["lvn_levels"]
        volume_in_bins = vp_data["volume_in_bins"]

        features = {}
        
        # Price relative to POC
        features['price_vs_poc'] = (current_price - poc) / current_price if current_price > 0 and not np.isnan(poc) else 0.0

        # Proximity to closest HVN
        if hvn_levels:
            closest_hvn = min(hvn_levels, key=lambda x: abs(x - current_price))
            features['dist_to_closest_hvn'] = (current_price - closest_hvn) / current_price if current_price > 0 else 0.0
            features['vol_at_closest_hvn'] = volume_in_bins.get(closest_hvn, 0.0)
        else:
            features['dist_to_closest_hvn'] = 0.0
            features['vol_at_closest_hvn'] = 0.0

        # Proximity to closest LVN
        if lvn_levels:
            closest_lvn = min(lvn_levels, key=lambda x: abs(x - current_price))
            features['dist_to_closest_lvn'] = (current_price - closest_lvn) / current_price if current_price > 0 else 0.0
            features['vol_at_closest_lvn'] = volume_in_bins.get(closest_lvn, 0.0)
        else:
            features['dist_to_closest_lvn'] = 0.0
            features['vol_at_closest_lvn'] = 0.0
        
        # Volume at current price bin (conceptual, requires finding the bin for current_price)
        # For this, we'd need the bins from calculate_volume_profile.
        # For now, a rough estimate:
        current_price_vol_bin = 0.0
        if not volume_in_bins.empty and current_price > 0:
            # Find the bin that current_price falls into
            # This requires access to the bin edges from calculate_volume_profile
            # For now, a rough estimate:
            closest_bin_midpoint = min(volume_in_bins.index, key=lambda x: abs(x - current_price))
            current_price_vol_bin = volume_in_bins.get(closest_bin_midpoint, 0.0)

        features['vol_at_current_price_bin'] = current_price_vol_bin
        features['poc_volume'] = volume_in_bins.get(poc, 0.0) if not np.isnan(poc) else 0.0

        return pd.Series(features)


    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        """
        Trains the individual models and the meta-learner for the SR_ZONE_ACTION ensemble.
        `historical_targets` could be 'REJECTION', 'BREAKTHROUGH_UP', 'BREAKTHROUGH_DOWN'.
        """
        self.logger.info(f"Training {self.ensemble_name} ensemble models...")

        if historical_features.empty or historical_targets.empty:
            self.logger.warning(f"No data to train {self.ensemble_name}. Skipping training.")
            return

        # Ensure historical_features has 'close' column for volume profile derivation
        if 'close' not in historical_features.columns:
            self.logger.error("Historical features missing 'close' column for volume profile. Cannot train.")
            return

        # Prepare flat data for LGBM
        X_flat, y_flat = self._prepare_flat_data(historical_features, historical_targets)

        if X_flat.empty:
            self.logger.warning(f"Insufficient data after preparation for {self.ensemble_name}. Skipping model training.")
            return

        # Derive volume profile features for the historical data
        historical_vp_features_list = []
        for idx, row in historical_features.iterrows():
            # Pass historical klines up to current row for volume profile calculation
            # This is computationally intensive; in practice, you might pre-calculate or use a rolling window.
            historical_vp_features_list.append(self._derive_volume_profile_features(
                historical_features.loc[:idx], 
                row['close'] 
            ))
        historical_vp_features = pd.DataFrame(historical_vp_features_list, index=historical_features.index)
        
        # Align historical_vp_features with X_flat and y_flat
        historical_vp_features = historical_vp_features.loc[X_flat.index].fillna(0)
        
        # Store volume_profile_features_list for consistent prediction
        self.volume_profile_features_list = historical_vp_features.columns.tolist()


        unique_targets = np.unique(y_flat)
        self.dl_target_map = {label: i for i, label in enumerate(unique_targets)}
        # Removed the unused variable assignment for y_flat_encoded
        # y_flat_encoded = np.array([self.dl_target_map[label] for label in y_flat]) # F841 warning is expected here if TabNet is not used

        # --- Train Individual Models ---

        # Volume Profile Model (LGBM Classifier)
        self.logger.info("Training Volume Profile model (LGBM)...")
        try:
            self.models["volume_profile"] = LGBMClassifier(
                random_state=42, 
                verbose=-1,
                reg_alpha=self.dl_config["l1_reg_strength"],
                reg_lambda=self.dl_config["l2_reg_strength"]
            )
            self.models["volume_profile"].fit(historical_vp_features, y_flat)
            self.logger.info("Volume Profile model trained.")
        except Exception as e:
            self.logger.error(f"Error training Volume Profile model: {e}")
            self.models["volume_profile"] = None

        # Order Flow Model (LightGBM Classifier)
        self.logger.info("Training Order Flow model (LGBMClassifier)...")
        # Ensure historical_features contains order flow features
        order_flow_features_for_training = historical_features[['cvd_value', 'cvd_divergence_score', 'is_absorption', 'is_exhaustion', 'aggressive_buy_volume', 'aggressive_sell_volume']].dropna()
        
        # Refined: Simulate a multi-class target for order flow model based on price action after order flow
        # 1. Price moves significantly UP (BREAKTHROUGH_UP)
        # 2. Price moves significantly DOWN (BREAKTHROUGH_DOWN)
        # 3. Price stays relatively flat (HOLD / REJECTION)
        target_lookahead_of = 2 # Look 2 periods ahead for outcome
        price_change_of = historical_features['close'].pct_change(target_lookahead_of).shift(-target_lookahead_of)
        
        # Define thresholds for significant movement (e.g., 0.2%)
        up_threshold = 0.002  
        down_threshold = -0.002 

        simulated_of_target = pd.Series('HOLD_SR', index=price_change_of.index) # Default to HOLD_SR
        
        # Determine labels based on S/R interaction type and price movement
        # Ensure 'Is_SR_Support_Interacting' and 'Is_SR_Resistance_Interacting' are in historical_features
        if 'Is_SR_Support_Interacting' in historical_features.columns and 'Is_SR_Resistance_Interacting' in historical_features.columns:
            # Breakthrough Up through Resistance
            breakthrough_up_cond = (price_change_of > up_threshold) & (historical_features['Is_SR_Resistance_Interacting'] == 1)
            simulated_of_target[breakthrough_up_cond] = 'BREAKOUT_THROUGH_RESISTANCE'

            # Breakdown Through Support
            breakdown_down_cond = (price_change_of < down_threshold) & (historical_features['Is_SR_Support_Interacting'] == 1)
            simulated_of_target[breakdown_down_cond] = 'BREAKDOWN_THROUGH_SUPPORT'

            # Rejection from Resistance (price moves down after interacting with resistance)
            rejection_resistance_cond = (price_change_of < -up_threshold) & (historical_features['Is_SR_Resistance_Interacting'] == 1)
            simulated_of_target[rejection_resistance_cond] = 'REJECTION_FROM_RESISTANCE'

            # Bounce off Support (price moves up after interacting with support)
            bounce_support_cond = (price_change_of > -down_threshold) & (historical_features['Is_SR_Support_Interacting'] == 1)
            simulated_of_target[bounce_support_cond] = 'BOUNCE_OFF_SUPPORT'
        else:
            self.logger.warning("S/R interaction type features not found in historical_features. Using simplified order flow targets.")
            # Fallback to generic breakthrough/rejection if S/R types are not available
            simulated_of_target[price_change_of > up_threshold] = 'BREAKTHROUGH_UP_GENERIC'
            simulated_of_target[price_change_of < down_threshold] = 'BREAKTHROUGH_DOWN_GENERIC'


        # Align order flow features with simulated target
        aligned_of_data = order_flow_features_for_training.join(simulated_of_target.rename('of_target')).dropna()
        
        if not aligned_of_data.empty:
            X_of = aligned_of_data.drop(columns=['of_target'])
            y_of = aligned_of_data['of_target']
            
            # Store target mapping for prediction
            self.order_flow_target_map = {label: i for i, label in enumerate(np.unique(y_of))}

            try:
                self.models["order_flow"] = LGBMClassifier(
                    random_state=42, 
                    verbose=-1,
                    reg_alpha=self.dl_config["l1_reg_strength"],
                    reg_lambda=self.dl_config["l2_reg_strength"]
                )
                self.models["order_flow"].fit(X_of, y_of)
                self.logger.info("Order Flow model trained.")
            except Exception as e:
                self.logger.error(f"Error training Order Flow model: {e}")
                self.models["order_flow"] = None
        else:
            self.logger.warning("Insufficient aligned data for Order Flow model training. Skipping.")
            self.models["order_flow"] = None

        # LightGBM Model
        self.logger.info("Training LightGBM model...")
        try:
            self.models["lgbm"] = LGBMClassifier(
                random_state=42, 
                verbose=-1,
                reg_alpha=self.dl_config["l1_reg_strength"],
                reg_lambda=self.dl_config["l2_reg_strength"]
            )
            self.models["lgbm"].fit(X_flat, y_flat)
            self.logger.info("LightGBM model trained.")
        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {e}")
            self.models["lgbm"] = None

        # --- Train Meta-Learner ---
        meta_features_train = pd.DataFrame(index=X_flat.index)
        
        # Populate meta_features_train with actual (or simulated) predictions/confidences
        if self.models["volume_profile"]:
            vp_probas = self.models["volume_profile"].predict_proba(historical_vp_features)
            # Ensure vp_probas length matches y_flat length
            if len(vp_probas) == len(y_flat):
                vp_conf_values = [vp_probas[i, self.models["volume_profile"].classes_.tolist().index(y_flat.iloc[i])] for i in range(len(y_flat))]
                meta_features_train['volume_profile_conf'] = pd.Series(vp_conf_values, index=X_flat.index)
            else:
                self.logger.warning("Volume Profile probas length mismatch with y_flat. Using random uniform for volume_profile_conf.")
                meta_features_train['volume_profile_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))
        else:
            meta_features_train['volume_profile_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))
        
        if self.models["order_flow"]:
            # Predict probabilities for historical order flow features
            aligned_X_of = X_of.loc[X_flat.index].fillna(0) # Align X_of with X_flat index
            order_flow_probas_hist = self.models["order_flow"].predict_proba(aligned_X_of)
            # Map probabilities to a single directional score for meta-learner
            order_flow_score_values = []
            for i, proba_array in enumerate(order_flow_probas_hist):
                predicted_class = self.models["order_flow"].classes_[np.argmax(proba_array)]
                if predicted_class in ['BREAKOUT_THROUGH_RESISTANCE', 'BOUNCE_OFF_SUPPORT', 'BREAKTHROUGH_UP_GENERIC']:
                    order_flow_score_values.append(proba_array[self.order_flow_target_map[predicted_class]])
                elif predicted_class in ['BREAKDOWN_THROUGH_SUPPORT', 'REJECTION_FROM_RESISTANCE', 'BREAKTHROUGH_DOWN_GENERIC']:
                    order_flow_score_values.append(-proba_array[self.order_flow_target_map[predicted_class]])
                else: # HOLD_SR
                    order_flow_score_values.append(0.0) # Neutral score for HOLD
            
            # Align with X_flat index
            meta_features_train['order_flow_score'] = pd.Series(order_flow_score_values, index=aligned_X_of.index).reindex(X_flat.index).fillna(0.0)
        else:
            meta_features_train['order_flow_score'] = np.random.uniform(-0.5, 0.5, len(X_flat)) # Random directional score

        if self.models["lgbm"]:
            lgbm_probas = self.models["lgbm"].predict_proba(X_flat)
            if len(lgbm_probas) == len(y_flat):
                lgbm_conf_values = [lgbm_probas[i, self.models["lgbm"].classes_.tolist().index(y_flat.iloc[i])] for i in range(len(y_flat))]
                meta_features_train['lgbm_proba'] = pd.Series(lgbm_conf_values, index=X_flat.index)
            else:
                self.logger.warning("LGBM probas length mismatch with y_flat. Using random uniform for lgbm_proba.")
                meta_features_train['lgbm_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))
        else:
            meta_features_train['lgbm_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))

        # Add advanced features to meta_features_train
        meta_features_train = meta_features_train.join(historical_features[['is_wyckoff_sos', 'is_wyckoff_sow', 'is_wyckoff_spring', 'is_wyckoff_upthrust',
                                                                      'is_accumulation_phase', 'is_distribution_phase',
                                                                      'is_liquidity_sweep', 'is_fake_breakout', 'is_large_bid_wall_near', 'is_large_ask_wall_near',
                                                                      'cvd_value', 'cvd_divergence_score', 'is_absorption', 'is_exhaustion',
                                                                      'aggressive_buy_volume', 'aggressive_sell_volume',
                                                                      'htf_trend_bullish', 'mtf_trend_bullish', # Existing features
                                                                      'Is_SR_Support_Interacting', 'Is_SR_Resistance_Interacting']], how='inner').fillna(0) # NEW S/R features

        meta_features_train = meta_features_train.loc[y_flat.index].dropna()
        y_meta_train = y_flat.loc[meta_features_train.index]

        if meta_features_train.empty:
            self.logger.warning("Meta-features training set is empty. Cannot train meta-learner.")
            self.trained = False
            return

        self.meta_learner_features = meta_features_train.columns.tolist()

        self._train_meta_learner(meta_features_train, y_meta_train)
        self.trained = True

    def get_prediction(self, current_features: pd.DataFrame, 
                       klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame, 
                       order_book_data: dict, current_price: float) -> dict:
        """
        Gets a combined prediction and confidence score for the SR_ZONE_ACTION regime.
        """
        if not self.trained:
            self.logger.warning(f"{self.ensemble_name} not trained. Returning HOLD.")
            return {"prediction": "HOLD", "confidence": 0.0}

        self.logger.info(f"Getting prediction from {self.ensemble_name}...")

        current_features_row = current_features.tail(1)
        if current_features_row.empty:
            self.logger.warning("Current features row is empty. Cannot get prediction.")
            return {"prediction": "HOLD", "confidence": 0.0}

        X_current_flat, _ = self._prepare_flat_data(current_features_row)
        if X_current_flat.empty:
            self.logger.warning("Insufficient data for real-time flat prediction. Returning HOLD.")
            return {"prediction": "HOLD", "confidence": 0.0}

        individual_model_outputs = {}

        # Volume Profile Prediction
        if self.models["volume_profile"]:
            try:
                # Derive current volume profile features
                current_vp_features = self._derive_volume_profile_features(klines_df, current_price)
                # Ensure feature order for prediction is same as training
                current_vp_features_reindexed = current_vp_features.reindex(self.volume_profile_features_list, fill_value=0.0).to_frame().T
                
                vp_proba = self.models["volume_profile"].predict_proba(current_vp_features_reindexed)[0]
                # For SR_ZONE_ACTION, we might predict REJECTION or BREAKTHROUGH.
                # Here, we just take the max confidence.
                individual_model_outputs['volume_profile_conf'] = float(np.max(vp_proba))
            except Exception as e:
                self.logger.error(f"Error predicting with Volume Profile model: {e}")
                individual_model_outputs['volume_profile_conf'] = 0.5
        else:
            individual_model_outputs['volume_profile_conf'] = np.random.uniform(0.5, 0.9)


        # Order Flow Prediction
        if self.models["order_flow"]:
            try:
                order_flow_input = current_features_row[['cvd_value', 'cvd_divergence_score', 'is_absorption', 'is_exhaustion', 'aggressive_buy_volume', 'aggressive_sell_volume']].dropna()
                if not order_flow_input.empty:
                    order_flow_proba = self.models["order_flow"].predict_proba(order_flow_input)[0]
                    # Convert multi-class probabilities to a single directional score
                    predicted_class_idx = np.argmax(order_flow_proba)
                    predicted_class = self.models["order_flow"].classes_[predicted_class_idx]
                    
                    order_flow_score = 0.0
                    if predicted_class in ['BREAKOUT_THROUGH_RESISTANCE', 'BOUNCE_OFF_SUPPORT', 'BREAKTHROUGH_UP_GENERIC']:
                        order_flow_score = order_flow_proba[self.order_flow_target_map[predicted_class]]
                    elif predicted_class in ['BREAKDOWN_THROUGH_SUPPORT', 'REJECTION_FROM_RESISTANCE', 'BREAKTHROUGH_DOWN_GENERIC']:
                        order_flow_score = -order_flow_proba[self.order_flow_target_map[predicted_class]]
                    # If 'HOLD_SR', score remains 0.0
                    
                    individual_model_outputs['order_flow_score'] = float(order_flow_score) 
                else:
                    individual_model_outputs['order_flow_score'] = 0.0 # Neutral if no input
            except Exception as e:
                self.logger.error(f"Error predicting with Order Flow model: {e}")
                individual_model_outputs['order_flow_score'] = 0.0 # Neutral on error
        else:
            individual_model_outputs['order_flow_score'] = 0.0 # Default neutral score

        # LightGBM Prediction
        if self.models["lgbm"]:
            try:
                lgbm_features = X_current_flat.copy()
                lgbm_proba = self.models["lgbm"].predict_proba(lgbm_features)[0]
                individual_model_outputs['lgbm_proba'] = float(np.max(lgbm_proba))
            except Exception as e:
                self.logger.error(f"Error predicting with LightGBM: {e}")
                individual_model_outputs['lgbm_proba'] = 0.5
        else:
            individual_model_outputs['lgbm_proba'] = 0.5

        # --- Incorporate Advanced Features ---
        wyckoff_feats = self._get_wyckoff_features(klines_df, current_price)
        manipulation_feats = self._get_manipulation_features(order_book_data, current_price, agg_trades_df)
        order_flow_feats = self._get_order_flow_features(agg_trades_df, order_book_data, klines_df)
        multi_timeframe_feats = self._get_multi_timeframe_features(klines_df, klines_df)

        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in wyckoff_feats.items()})
        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in manipulation_feats.items()})
        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in order_flow_feats.items()})
        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in multi_timeframe_feats.items()})
        
        # NEW: Add S/R interaction type features to individual_model_outputs
        individual_model_outputs['Is_SR_Support_Interacting'] = float(current_features_row['Is_SR_Support_Interacting'].iloc[0])
        individual_model_outputs['Is_SR_Resistance_Interacting'] = float(current_features_row['Is_SR_Resistance_Interacting'].iloc[0])


        # --- Confluence Model (Meta-Learner) ---
        final_prediction = "HOLD"
        final_confidence = 0.0

        if self.meta_learner:
            try:
                meta_features_for_pred = {}
                for feature_name in self.meta_learner_features:
                    meta_features_for_pred[feature_name] = individual_model_outputs.get(feature_name, 0.0)

                meta_input_data = pd.DataFrame([meta_features_for_pred])
                final_prediction, final_confidence = self._get_meta_prediction(meta_input_data.iloc[0].to_dict())
            except Exception as e:
                self.logger.error(f"Error during meta-learner prediction for {self.ensemble_name}: {e}", exc_info=True)
                final_prediction = "HOLD"
                final_confidence = 0.0
        else:
            self.logger.warning(f"Meta-learner not available for {self.ensemble_name}. Averaging confidences.")
            # This fallback averaging is less meaningful now with diverse features like 'order_flow_score'.
            # It's better to rely on the meta-learner. If meta-learner is not trained,
            # the ensemble should ideally return a neutral prediction.
            self.logger.warning("Fallback averaging is less meaningful with implemented models. Returning HOLD.")
            final_prediction = "HOLD"
            final_confidence = 0.0

        self.logger.info(f"Ensemble Result for {self.ensemble_name}: Prediction={final_prediction}, Confidence={final_confidence:.2f}")
        return {"prediction": final_prediction, "confidence": final_confidence}
