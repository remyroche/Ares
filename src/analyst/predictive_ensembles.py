import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Placeholder imports for actual models
# from tensorflow.keras.models import load_model
# from lightgbm import LGBMClassifier


class PredictiveEnsembles:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_engineering = FeatureEngineering(self.config)
        self.regime_classifier = RegimeClassifier(self.config)
        self.ml_target_generator = MLTargetGenerator(self.config)
        self.regime_ensembles = RegimePredictiveEnsembles(self.config)

    def get_predictions(self, asset_data):
        """
        ## CHANGE: Updated the prediction workflow to use the full feature set.
        ## This method now calls the new `_prepare_feature_data` helper to ensure
        ## that the features used for live prediction match those used for training,
        ## resolving the previous discrepancy.
        Orchestrates the process of getting predictions for a set of assets.
        """
        all_predictions = {}
        for asset, data in asset_data.items():
            self.logger.info(f"Processing predictions for {asset}...")
            try:
                # 1. Prepare the full feature set for inference
                feature_data = self._prepare_feature_data(
                    data["klines_df"],
                    data.get("agg_trades_df"),
                    data.get("order_book_data"),
                )

                # 2. Get predictions using the full feature set
                # Note: We pass `feature_data`, which has no target columns.
                predictions = self.regime_ensembles.get_all_predictions(
                    asset, feature_data
                )
                all_predictions[asset] = predictions
                self.logger.info(f"Successfully generated predictions for {asset}.")
            except Exception as e:
                self.logger.error(
                    f"Failed to get predictions for {asset}. Error: {e}", exc_info=True
                )
                all_predictions[asset] = {"error": str(e)}
        return all_predictions

    def train_models(self, asset_data):
        """
        Orchestrates the training of all predictive ensemble models.
        """
        self.logger.info("Starting model training orchestration...")
        for asset, data in asset_data.items():
            self.logger.info(f"Starting training for {asset}...")
            try:
                prepared_data = self._prepare_data_for_training(
                    data["klines_df"],
                    data.get("agg_trades_df"),
                    data.get("order_book_data"),
                )
                self.regime_ensembles.train_all_models(asset, prepared_data)
                self.logger.info(f"Successfully completed training for {asset}.")
            except Exception as e:
                self.logger.error(
                    f"Failed to train models for {asset}. Error: {e}", exc_info=True
                )
        self.logger.info("Model training orchestration finished.")

    def _prepare_data_for_training(
        self, klines_df, agg_trades_df=None, order_book_data=None
    ):
        """
        Prepares a complete dataset for training.
        """
        features_df = self.feature_engineering.generate_features(klines_df)
        regime_df = self.regime_classifier.classify_regime(features_df)
        data_with_targets = self.ml_target_generator.generate_targets(
            pd.concat([features_df, regime_df], axis=1)
        )

        order_flow_feats = self._get_order_flow_features(
            agg_trades_df, order_book_data
        )
        multi_timeframe_feats = self._get_multi_timeframe_features(klines_df)
        wyckoff_feats = self._get_wyckoff_features(klines_df)
        manipulation_feats = self._get_manipulation_features(order_book_data)

        # Combine all features
        # Note: Merging requires a common index (timestamp)
        final_data = data_with_targets
        for f_df in [
            order_flow_feats,
            multi_timeframe_feats,
            wyckoff_feats,
            manipulation_feats,
        ]:
            if not f_df.empty:
                final_data = final_data.join(f_df, how="left")

        return final_data.fillna(method="ffill").dropna()

    def _get_order_flow_features(self, agg_trades_df, order_book_data):
        if agg_trades_df is None or agg_trades_df.empty:
            return pd.DataFrame()

        self.logger.info("Generating Order Flow features...")
        # Calculate Cumulative Volume Delta (CVD)
        # 'm' is True if the buyer is the maker, meaning the aggressor was a seller.
        direction = np.where(agg_trades_df["m"], -1, 1)
        signed_volume = agg_trades_df["q"] * direction
        cvd = signed_volume.cumsum()
        features = pd.DataFrame({"cvd": cvd}, index=agg_trades_df.index)

        # Basic order book features
        if order_book_data is not None and not order_book_data.empty:
            best_bid = order_book_data[order_book_data["side"] == "buy"]["price"].max()
            best_ask = order_book_data[order_book_data["side"] == "sell"]["price"].min()
            features["bid_ask_spread"] = best_ask - best_bid

        return features

    def _get_multi_timeframe_features(self, klines_df, htf_period="4H"):
        if klines_df is None or klines_df.empty:
            return pd.DataFrame()

        self.logger.info("Generating Multi-Timeframe features...")
        # Ensure index is datetime
        klines_df.index = pd.to_datetime(klines_df.index)

        # Base timeframe trend
        klines_df["base_ma"] = klines_df["close"].rolling(window=20).mean()
        klines_df["base_trend_up"] = (klines_df["close"] > klines_df["base_ma"]).astype(
            int
        )

        # Higher timeframe trend
        htf_klines = klines_df["close"].resample(htf_period).ohlc()
        htf_klines["htf_ma"] = htf_klines["close"].rolling(window=10).mean()
        htf_klines["htf_trend_up"] = (htf_klines["close"] > htf_klines["htf_ma"]).astype(
            int
        )

        # Align HTF trend back to base timeframe
        features = klines_df[["base_trend_up"]].join(
            htf_klines[["htf_trend_up"]], how="left"
        )
        features["htf_trend_up"] = features["htf_trend_up"].fillna(method="ffill")
        features["trend_alignment"] = (
            features["base_trend_up"] == features["htf_trend_up"]
        ).astype(int)

        return features[["htf_trend_up", "trend_alignment"]]

    def _get_wyckoff_features(self, klines_df):
        if klines_df is None or klines_df.empty:
            return pd.DataFrame()
        self.logger.info("Generating Wyckoff features...")
        volume_ma = klines_df["volume"].rolling(window=20).mean()
        price_range = klines_df["high"] - klines_df["low"]
        is_high_volume = klines_df["volume"] > (volume_ma * 1.5)
        is_wide_range = price_range > price_range.rolling(window=20).mean() * 1.5
        is_up_bar = klines_df["close"] > klines_df["open"]

        sos_bar = is_high_volume & is_wide_range & is_up_bar
        sow_bar = is_high_volume & is_wide_range & ~is_up_bar

        return pd.DataFrame(
            {"wyckoff_sos": sos_bar.astype(int), "wyckoff_sow": sow_bar.astype(int)},
            index=klines_df.index,
        )

    def _get_manipulation_features(self, order_book_data):
        if order_book_data is None or order_book_data.empty:
            return pd.DataFrame()
        self.logger.info("Generating Manipulation features...")
        current_price = (
            order_book_data[order_book_data["side"] == "buy"]["price"].max()
            + order_book_data[order_book_data["side"] == "sell"]["price"].min()
        ) / 2
        avg_order_size = order_book_data["quantity"].mean()

        # Look for large orders far from the current price
        far_buy_orders = order_book_data[
            (order_book_data["side"] == "buy")
            & (order_book_data["price"] < current_price * 0.95)
            & (order_book_data["quantity"] > avg_order_size * 10)
        ]
        far_sell_orders = order_book_data[
            (order_book_data["side"] == "sell")
            & (order_book_data["price"] > current_price * 1.05)
            & (order_book_data["quantity"] > avg_order_size * 10)
        ]

        is_spoofing = not far_buy_orders.empty or not far_sell_orders.empty
        # This feature would apply to the current snapshot, so we create a single-row DF
        # In a real system, you'd align this with the kline timestamp.
        return pd.DataFrame({"is_spoofing": [is_spoofing]}, index=[order_book_data.index[-1]])
