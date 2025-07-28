# src/analyst/predictive_ensembles/regime_ensembles/base_ensemble.py
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier # Using LightGBM for the meta-learner
from sklearn.model_selection import cross_val_score, KFold # For conceptual cross-validation
from src.utils.logger import system_logger

class BaseEnsemble(ABC):
    """
    Abstract Base Class for all regime-specific predictive ensembles.
    Defines the common interface and provides shared utility methods.
    """
    def __init__(self, config: dict, ensemble_name: str):
        self.config = config
        self.ensemble_name = ensemble_name
        self.logger = system_logger.getChild(f'Ensemble.{ensemble_name}')
        self.models = {} # Dictionary to hold individual trained models
        self.meta_learner = None # Model to combine individual predictions
        self.scaler = StandardScaler() # Scaler for meta-learner features
        self.trained = False
        self.meta_learner_classes = [] # To store the classes learned by the meta-learner

    @abstractmethod
    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        """
        Abstract method to train all individual models and the meta-learner for this ensemble.
        """
        pass

    @abstractmethod
    def get_prediction(self, current_features: pd.DataFrame, 
                       klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame, 
                       order_book_data: dict, current_price: float) -> dict:
        """
        Abstract method to get a combined prediction and confidence score from the ensemble.
        Returns a dictionary with 'prediction' (e.g., "BUY", "SELL", "HOLD") and 'confidence'.
        """
        pass

    def _train_meta_learner(self, X_meta: pd.DataFrame, y_meta: pd.Series):
        """
        Trains the meta-learner (confluence model) for this ensemble using LightGBM.
        Includes conceptual notes on cross-validation and regularization.
        """
        self.logger.info(f"Training meta-learner for {self.ensemble_name} using LightGBM...")
        if X_meta.empty or y_meta.empty:
            self.logger.warning(f"Meta-learner training: Input data is empty for {self.ensemble_name}. Skipping.")
            self.trained = False
            return

        # Ensure target labels are suitable for classification (e.g., encoded if not already)
        # For simplicity, assuming y_meta contains string labels like "BUY", "SELL", "HOLD"
        # LightGBM can handle string labels directly.

        # Scale features for the meta-learner
        scaled_X_meta = self.scaler.fit_transform(X_meta)

        try:
            # LightGBM Classifier with regularization parameters
            # L1 (reg_alpha) and L2 (reg_lambda) regularization help prevent overfitting.
            # These values would typically be part of the config or tuned via hyperparameter optimization.
            self.meta_learner = LGBMClassifier(
                random_state=42, 
                verbose=-1, # Suppress verbose output during training
                reg_alpha=self.config.get("meta_learner_l1_reg", 0.1), # L1 regularization
                reg_lambda=self.config.get("meta_learner_l2_reg", 0.1) # L2 regularization
            )
            
            # Conceptual Cross-Validation:
            # In a real system, you would perform cross-validation here to get a more robust
            # estimate of the meta-learner's performance and to tune its hyperparameters.
            # For example:
            # kf = KFold(n_splits=5, shuffle=True, random_state=42)
            # cv_scores = cross_val_score(self.meta_learner, scaled_X_meta, y_meta, cv=kf, scoring='accuracy')
            # self.logger.info(f"Meta-learner CV Accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
            
            self.meta_learner.fit(scaled_X_meta, y_meta)
            self.meta_learner_classes = self.meta_learner.classes_ # Store learned classes
            self.logger.info(f"Meta-learner for {self.ensemble_name} trained. Learned classes: {self.meta_learner_classes}")
            self.trained = True
        except Exception as e:
            self.logger.error(f"Error training meta-learner for {self.ensemble_name}: {e}")
            self.meta_learner = None # Invalidate model if training fails
            self.trained = False

    def _get_meta_prediction(self, individual_model_predictions: dict) -> tuple[str, float]:
        """
        Combines individual model predictions using the trained LightGBM meta-learner.
        Returns the final prediction (e.g., "BUY", "SELL", "HOLD") and its confidence score.
        """
        if not self.trained or self.meta_learner is None:
            self.logger.warning(f"Meta-learner for {self.ensemble_name} not trained. Returning default HOLD.")
            return "HOLD", 0.0

        # Prepare features for the meta-learner.
        # This requires ensuring the input features match the training features (column order and names).
        # `individual_model_predictions` is a dict like {'lstm_conf': 0.7, 'transformer_conf': 0.8, ...}
        # We need to convert this into a DataFrame row with the correct columns.
        
        # Create a DataFrame from the input dictionary, ensuring it's a single row
        meta_features_raw = pd.DataFrame([individual_model_predictions])
        
        # Ensure that meta_features_raw has the same columns as were used for training the scaler.
        # This is critical. In a real system, you'd store the feature names from training.
        # For this demo, we assume the keys in individual_model_predictions are consistent.
        
        try:
            # Scale the input features using the fitted scaler
            scaled_meta_features = self.scaler.transform(meta_features_raw)
            
            # Get probability predictions from the meta-learner
            # predict_proba returns probabilities for each class
            probabilities = self.meta_learner.predict_proba(scaled_meta_features)[0]
            
            # Get the predicted class (e.g., "BUY", "SELL", "HOLD")
            predicted_class_idx = np.argmax(probabilities)
            final_prediction = self.meta_learner_classes[predicted_class_idx]
            final_confidence = probabilities[predicted_class_idx]

            self.logger.debug(f"Meta-learner raw predictions: {probabilities}, Final Pred: {final_prediction}, Conf: {final_confidence:.2f}")
            
            return final_prediction, final_confidence

        except Exception as e:
            self.logger.error(f"Error during meta-learner prediction for {self.ensemble_name}: {e}")
            return "HOLD", 0.0 # Fallback on error

    # --- Common Feature Extraction Methods (can be overridden by specific ensembles) ---

    def _get_wyckoff_features(self, klines_df: pd.DataFrame, current_price: float) -> dict:
        """
        Rule-based approximation for Wyckoff pattern features.
        This is a simplified example; real Wyckoff detection is complex.
        """
        if klines_df.empty or len(klines_df) < 50: # Need sufficient history
            return {"is_wyckoff_sos": 0, "is_wyckoff_sow": 0, "is_wyckoff_spring": 0, "is_wyckoff_upthrust": 0}

        recent_klines = klines_df.iloc[-50:] # Look at last 50 candles
        price_range = recent_klines['high'].max() - recent_klines['low'].min()
        avg_volume = recent_klines['volume'].mean()

        # Simplified Spring/Upthrust (false breakout below/above a range)
        is_spring = 0
        is_upthrust = 0

        # Check for a recent low breaking previous support, then quickly recovering
        if len(recent_klines) > 2 and recent_klines['low'].iloc[-1] < recent_klines['low'].iloc[-2] and \
           recent_klines['close'].iloc[-1] > recent_klines['low'].iloc[-2]:
            is_spring = 1 # Price dipped below previous low and closed above it

        # Check for a recent high breaking previous resistance, then quickly falling
        if len(recent_klines) > 2 and recent_klines['high'].iloc[-1] > recent_klines['high'].iloc[-2] and \
           recent_klines['close'].iloc[-1] < recent_klines['high'].iloc[-2]:
            is_upthrust = 1 # Price poked above previous high and closed below it

        # Simplified Sign of Strength/Weakness (strong move with volume)
        is_sos = 0
        is_sow = 0
        if recent_klines['close'].iloc[-1] > recent_klines['open'].iloc[-1] * 1.02 and \
           recent_klines['volume'].iloc[-1] > avg_volume * 1.5: # 2% up move with 1.5x avg volume
            is_sos = 1
        elif recent_klines['close'].iloc[-1] < recent_klines['open'].iloc[-1] * 0.98 and \
             recent_klines['volume'].iloc[-1] > avg_volume * 1.5: # 2% down move with 1.5x avg volume
            is_sow = 1

        self.logger.debug(f"Wyckoff Features: Spring={is_spring}, Upthrust={is_upthrust}, SOS={is_sos}, SOW={is_sow}")
        return {
            "is_wyckoff_sos": is_sos,
            "is_wyckoff_sow": is_sow,
            "is_wyckoff_spring": is_spring,
            "is_wyckoff_upthrust": is_upthrust
        }

    def _get_manipulation_features(self, order_book_data: dict, current_price: float) -> dict:
        """
        Heuristic approximation for detecting market manipulation features.
        Requires real-time order book snapshots.
        """
        bids = order_book_data.get('bids', [])
        asks = order_book_data.get('asks', [])

        # Example: Detecting large order cancellations (spoofing approximation)
        # This would require tracking order book changes over time, which is not feasible with static snapshot.
        # Placeholder for conceptual logic:
        # If a large bid/ask wall appears near price and then quickly disappears without being filled,
        # it could be spoofing. This needs historical order book data.

        # Simplified: Check for unusually large single orders near current price (potential 'iceberg' or 'wall')
        large_order_threshold_usd = self.config.get("order_book", {}).get("large_order_threshold_usd", 100000)
        
        is_large_bid_wall_near = 0
        is_large_ask_wall_near = 0

        if current_price > 0:
            for bid_price, bid_qty in bids:
                if bid_price < current_price and (bid_price * bid_qty) > large_order_threshold_usd:
                    is_large_bid_wall_near = 1
                    break
            for ask_price, ask_qty in asks:
                if ask_price > current_price and (ask_price * ask_qty) > large_order_threshold_usd:
                    is_large_ask_wall_near = 1
                    break
        
        self.logger.debug(f"Manipulation Features: Large Bid Wall={is_large_bid_wall_near}, Large Ask Wall={is_large_ask_wall_near}")
        return {
            "is_liquidity_sweep": 0, # Requires tracking rapid changes
            "is_fake_breakout": 0,   # Requires tracking price action after breaking a level
            "is_large_bid_wall_near": is_large_bid_wall_near,
            "is_large_ask_wall_near": is_large_ask_wall_near
        }

    def _get_order_flow_features(self, agg_trades_df: pd.DataFrame, order_book_data: dict) -> dict:
        """
        Conceptual implementation for Order Flow and Market Microstructure features.
        Approximates CVD and detects absorption/exhaustion from aggregated trades.
        """
        if agg_trades_df.empty:
            return {"cvd_divergence": 0.0, "is_absorption": 0, "is_exhaustion": 0}

        # Calculate Cumulative Volume Delta (CVD)
        # Assuming 'is_buyer_maker' is True for seller-initiated (taker buy) and False for buyer-initiated (taker sell)
        # Or, more commonly, True means buyer is the maker (passive order), so taker is seller.
        # Let's assume 'is_buyer_maker': True for taker-sell (price goes down), False for taker-buy (price goes up)
        # Binance agg trades 'm' field: True if buyer is maker, False if seller is maker.
        # If buyer is maker (m=True), it's a sell order hitting a bid. So delta is negative.
        # If m is False (seller is maker), it's a buy order hitting an ask. So delta is positive.
        
        # Corrected delta calculation based on Binance 'm' field:
        # If m is True (buyer is maker), it's a sell trade, price likely went down, so quantity is negative for delta.
        # If m is False (seller is maker), it's a buy trade, price likely went up, so quantity is positive for delta.
        agg_trades_df['signed_quantity'] = agg_trades_df['quantity'] * np.where(agg_trades_df['is_buyer_maker'], -1, 1)
        
        # Sum signed quantities over a recent period to get a proxy for CVD
        cvd = agg_trades_df['signed_quantity'].sum()
        
        # Simple Absorption/Exhaustion detection
        # Absorption: Price not moving much despite strong directional volume (CVD)
        # Exhaustion: Price moving strongly but CVD is slowing or reversing
        
        is_absorption = 0
        is_exhaustion = 0
        
        if len(agg_trades_df) > 10: # Need enough trades for meaningful analysis
            recent_price_change = agg_trades_df['price'].iloc[-1] - agg_trades_df['price'].iloc[0]
            
            # If large CVD but small price change: absorption
            if abs(cvd) > agg_trades_df['quantity'].sum() * 0.1 and abs(recent_price_change) < (agg_trades_df['price'].mean() * 0.001): # 0.1% price change
                if cvd > 0: # Bullish CVD but no price up means absorption of buys
                    is_absorption = 1
                elif cvd < 0: # Bearish CVD but no price down means absorption of sells
                    is_absorption = 1
            
            # Exhaustion (conceptual): Price moves, but CVD indicates less interest or reversal
            # This is harder to approximate without tracking CVD over time.
            # For now, a placeholder.
            
        self.logger.debug(f"Order Flow Features: CVD={cvd:.2f}, Absorption={is_absorption}, Exhaustion={is_exhaustion}")
        return {
            "cvd_divergence": cvd, # Can be divergence with price
            "is_absorption": is_absorption,
            "is_exhaustion": is_exhaustion
        }

    def _get_multi_timeframe_features(self, klines_df_htf: pd.DataFrame, klines_df_mtf: pd.DataFrame) -> dict:
        """
        Placeholder for multi-timeframe analysis features.
        Assumes klines_df_htf is higher timeframe and klines_df_mtf is medium/lower timeframe.
        """
        # Example: HTF trend direction, alignment of MAs across timeframes.
        # This requires actual data from different timeframes.
        # For a demo, we'll simulate.
        
        htf_trend_bullish = np.random.choice([0, 1])
        mtf_trend_bullish = np.random.choice([0, 1])
        
        self.logger.debug(f"Multi-Timeframe Features: HTF Bullish={htf_trend_bullish}, MTF Bullish={mtf_trend_bullish}")
        return {
            "htf_trend_bullish": htf_trend_bullish,
            "mtf_trend_bullish": mtf_trend_bullish
        }

