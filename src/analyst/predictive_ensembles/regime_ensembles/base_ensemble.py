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
        Detects Springs, Upthrusts, Signs of Strength/Weakness, and conceptual phases.
        """
        if klines_df.empty or len(klines_df) < 100: # Need sufficient history for patterns
            return {
                "is_wyckoff_sos": 0, "is_wyckoff_sow": 0, 
                "is_wyckoff_spring": 0, "is_wyckoff_upthrust": 0,
                "is_accumulation_phase": 0, "is_distribution_phase": 0
            }

        recent_klines = klines_df.iloc[-100:] # Look at last 100 candles for broader patterns
        avg_volume = recent_klines['volume'].mean()

        # --- Springs and Upthrusts (False Breakouts with Volume) ---
        is_spring = 0
        is_upthrust = 0

        # Spring: Price dips below a previous low (support) but quickly recovers with volume
        # Check for a new low that is immediately rejected
        if len(recent_klines) >= 3:
            last_low = recent_klines['low'].iloc[-1]
            prev_lows_window = recent_klines['low'].iloc[-10:-1] # Look at recent lows
            
            # If current low breaks a recent support level AND closes significantly higher
            if last_low < prev_lows_window.min() and \
               recent_klines['close'].iloc[-1] > recent_klines['open'].iloc[-1] and \
               recent_klines['volume'].iloc[-1] > avg_volume * 1.5: # Volume confirmation
                is_spring = 1

        # Upthrust: Price pokes above a previous high (resistance) but quickly falls with volume
        # Check for a new high that is immediately rejected
        if len(recent_klines) >= 3:
            last_high = recent_klines['high'].iloc[-1]
            prev_highs_window = recent_klines['high'].iloc[-10:-1] # Look at recent highs

            # If current high breaks a recent resistance level AND closes significantly lower
            if last_high > prev_highs_window.max() and \
               recent_klines['close'].iloc[-1] < recent_klines['open'].iloc[-1] and \
               recent_klines['volume'].iloc[-1] > avg_volume * 1.5: # Volume confirmation
                is_upthrust = 1

        # --- Signs of Strength (SOS) and Signs of Weakness (SOW) ---
        is_sos = 0
        is_sow = 0
        
        # SOS: Strong upward move on high volume
        if recent_klines['close'].iloc[-1] > recent_klines['open'].iloc[-1] * 1.02 and \
           recent_klines['volume'].iloc[-1] > avg_volume * 2.0: # 2% up move with 2x avg volume
            is_sos = 1
        # SOW: Strong downward move on high volume
        elif recent_klines['close'].iloc[-1] < recent_klines['open'].iloc[-1] * 0.98 and \
             recent_klines['volume'].iloc[-1] > avg_volume * 2.0: # 2% down move with 2x avg volume
            is_sow = 1

        # --- Conceptual Phases (Accumulation/Distribution) ---
        # True detection of Wyckoff phases requires complex pattern recognition over longer periods,
        # often involving visual inspection and subjective judgment.
        # Here, we provide very simplified heuristics.
        is_accumulation_phase = 0
        is_distribution_phase = 0

        # Accumulation: Sideways movement with increasing volume on rallies and decreasing volume on dips
        # Or, price making higher lows and higher highs within a range.
        # Heuristic: Recent price range is narrow, but volume is relatively high on upward moves.
        price_std = recent_klines['close'].std()
        if price_std / current_price < 0.01 and recent_klines['volume'].iloc[-10:].mean() > avg_volume * 1.2:
             # Check for subtle upward bias within the range (e.g., higher lows)
             if recent_klines['low'].iloc[-5:].is_monotonic_increasing:
                 is_accumulation_phase = 1

        # Distribution: Sideways movement with increasing volume on dips and decreasing volume on rallies
        # Or, price making lower highs and lower lows within a range.
        # Heuristic: Recent price range is narrow, but volume is relatively high on downward moves.
        if price_std / current_price < 0.01 and recent_klines['volume'].iloc[-10:].mean() > avg_volume * 1.2:
             # Check for subtle downward bias within the range (e.g., lower highs)
             if recent_klines['high'].iloc[-5:].is_monotonic_decreasing:
                 is_distribution_phase = 1

        self.logger.debug(f"Wyckoff Features: Spring={is_spring}, Upthrust={is_upthrust}, SOS={is_sos}, SOW={is_sow}, Acc={is_accumulation_phase}, Dist={is_distribution_phase}")
        return {
            "is_wyckoff_sos": is_sos,
            "is_wyckoff_sow": is_sow,
            "is_wyckoff_spring": is_spring,
            "is_wyckoff_upthrust": is_upthrust,
            "is_accumulation_phase": is_accumulation_phase,
            "is_distribution_phase": is_distribution_phase
        }

    def _get_manipulation_features(self, order_book_data: dict, current_price: float, agg_trades_df: pd.DataFrame) -> dict:
        """
        Heuristic approximation for detecting market manipulation features.
        This is highly limited by the available data (top-5 order book, aggregated trades).
        True manipulation detection requires Level 2/3 data and real-time order event tracking.
        """
        bids = order_book_data.get('bids', [])
        asks = order_book_data.get('asks', [])

        # --- Spoofing/Layering Heuristic (Conceptual) ---
        # This requires tracking changes in order book depth over time.
        # With only a single snapshot, we can only look for large, static walls.
        # For true spoofing, you need to see large orders appear and disappear rapidly without execution.
        # Placeholder for conceptual logic:
        # if large_wall_appeared_and_disappeared_rapidly: is_spoofing = 1

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
        
        # --- Liquidity Sweeping Heuristic ---
        # Liquidity sweep: Rapid consumption of a large block of orders in the order book.
        # This is hard to detect with aggregated trades and limited depth.
        # Heuristic: A very large single aggregated trade that moves price significantly.
        is_liquidity_sweep = 0
        if not agg_trades_df.empty and len(agg_trades_df) > 1:
            last_trade = agg_trades_df.iloc[-1]
            prev_trades_avg_qty = agg_trades_df['quantity'].iloc[:-1].mean()
            
            # If current trade quantity is much larger than average AND price moved a lot
            if last_trade['quantity'] > prev_trades_avg_qty * 5 and \
               abs(last_trade['price'] - agg_trades_df['price'].iloc[-2]) / agg_trades_df['price'].iloc[-2] > 0.001: # 0.1% price move
                is_liquidity_sweep = 1

        # --- Fake Breakout Heuristic ---
        # This is more about price action, but can be influenced by manipulation.
        # Already conceptually covered in Wyckoff (Upthrust/Spring).
        is_fake_breakout = 0 # Re-using Wyckoff logic for now, or could add specific logic here.

        self.logger.debug(f"Manipulation Features: Large Bid Wall={is_large_bid_wall_near}, Large Ask Wall={is_large_ask_wall_near}, Liquidity Sweep={is_liquidity_sweep}")
        return {
            "is_liquidity_sweep": is_liquidity_sweep,
            "is_fake_breakout": is_fake_breakout,   # Placeholder, could integrate with Wyckoff
            "is_large_bid_wall_near": is_large_bid_wall_near,
            "is_large_ask_wall_near": is_large_ask_wall_near
        }

    def _get_order_flow_features(self, agg_trades_df: pd.DataFrame, order_book_data: dict, klines_df: pd.DataFrame) -> dict:
        """
        Advanced heuristic implementation for Order Flow and Market Microstructure features.
        Highly limited by data granularity (aggregated trades, top-5 order book).
        True order flow requires tick-level data and full order book snapshots.
        """
        if agg_trades_df.empty or klines_df.empty:
            return {"cvd_divergence": 0.0, "is_absorption": 0, "is_exhaustion": 0, "aggressive_buy_volume": 0, "aggressive_sell_volume": 0}

        # --- Cumulative Volume Delta (CVD) ---
        # `signed_quantity` is already calculated in feature_engineering, but we can re-calculate for recent trades
        # Assuming 'is_buyer_maker': True for passive buyer (taker sells), False for passive seller (taker buys)
        # Corrected: Binance 'm' (is_buyer_maker) means if the BUYER was the MAKER.
        # If m=True (buyer is maker), it's a sell order hitting a bid (price likely goes down). So signed_qty = -qty.
        # If m=False (seller is maker), it's a buy order hitting an ask (price likely goes up). So signed_qty = +qty.
        agg_trades_df['signed_quantity'] = agg_trades_df['quantity'] * np.where(agg_trades_df['is_buyer_maker'], -1, 1)
        
        # Consider recent trades for CVD
        recent_trades = agg_trades_df.iloc[-min(len(agg_trades_df), 100):] # Last 100 trades
        cvd = recent_trades['signed_quantity'].sum()
        
        # --- CVD Divergence with Price (Heuristic) ---
        # Divergence: Price moves one way, but CVD moves opposite or stagnates.
        price_change_recent = klines_df['close'].iloc[-1] - klines_df['close'].iloc[-min(len(klines_df), 10):].mean()
        
        cvd_divergence_score = 0.0
        if price_change_recent > 0 and cvd < 0: # Price up, but net selling
            cvd_divergence_score = -1.0 # Bearish divergence
        elif price_change_recent < 0 and cvd > 0: # Price down, but net buying
            cvd_divergence_score = 1.0 # Bullish divergence

        # --- Absorption and Exhaustion Patterns (Heuristic) ---
        # Absorption: Strong buying/selling pressure (high CVD) but price not moving much.
        # Exhaustion: Price moving strongly but CVD or volume is decreasing.
        
        is_absorption = 0
        is_exhaustion = 0

        # Absorption heuristic: High CVD, low price volatility
        if abs(cvd) > recent_trades['quantity'].sum() * 0.2: # Significant CVD
            recent_price_volatility = klines_df['close'].iloc[-10:].std() / klines_df['close'].iloc[-1]
            if recent_price_volatility < 0.001: # Very low volatility (e.g., 0.1%)
                is_absorption = 1
        
        # Exhaustion heuristic: Price continues in a trend, but volume/CVD drops off
        # Requires tracking previous CVD/volume.
        # For simplicity, if current candle is long but volume is low relative to average
        current_candle_range = klines_df['high'].iloc[-1] - klines_df['low'].iloc[-1]
        avg_candle_range = (klines_df['high'] - klines_df['low']).iloc[-50:-1].mean()
        
        if current_candle_range > avg_candle_range * 1.5: # Large candle
            current_volume = klines_df['volume'].iloc[-1]
            avg_volume_recent = klines_df['volume'].iloc[-50:-1].mean()
            if current_volume < avg_volume_recent * 0.8: # But low volume
                is_exhaustion = 1

        # --- Aggressive Buy/Sell Volume (from Agg Trades) ---
        aggressive_buy_volume = recent_trades[recent_trades['signed_quantity'] > 0]['quantity'].sum()
        aggressive_sell_volume = recent_trades[recent_trades['signed_quantity'] < 0]['quantity'].sum()
        
        self.logger.debug(f"Order Flow Features: CVD={cvd:.2f}, CVD Divergence={cvd_divergence_score:.2f}, Absorption={is_absorption}, Exhaustion={is_exhaustion}, AggBuy={aggressive_buy_volume:.2f}, AggSell={aggressive_sell_volume:.2f}")
        return {
            "cvd_value": cvd, # Raw CVD
            "cvd_divergence_score": cvd_divergence_score, 
            "is_absorption": is_absorption,
            "is_exhaustion": is_exhaustion,
            "aggressive_buy_volume": aggressive_buy_volume,
            "aggressive_sell_volume": aggressive_sell_volume
        }

