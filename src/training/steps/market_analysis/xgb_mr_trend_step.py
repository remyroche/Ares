"""
XGB Mean Reversion vs Trend Step

This step classifies market regimes into Trend-Following (TF), Mean-Reversion (MR),
or Noise using an XGBoost classifier.

It implements specific logic to detect regimes:
- Trend (TF): Price continues in direction (Sign(FutureRet) == Sign(Slope))
- Mean Reversion (MR): Price reverts/stalls when extended (Abs(ZScore) > 2.0 & Abs(FutureRet) < Threshold)

Features are calculated locally to ensure isolation.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.features_common.transforms.scaling_normalization import ScalingNormalizer
from src.utils.versioned_artifacts.temporal_splits import create_temporal_split_config_for_pipeline
from src.utils.ml_common.standardized_xgb_trainer import StandardizedXGBTrainer, XGBTrainingConfig

logger = logging.getLogger(__name__)

class XGBMrTrendStep(BaseStep):
    """Pipeline step to train XGBoost MR vs Trend classifier."""

    def __init__(self, step_name: str = "xgb_mr_trend_step"):
        """Initialize the XGB MR/Trend step."""
        super().__init__(step_name, use_versioned_artifacts=True)
        # Use a consistent model namespace
        self.model_namespace = "regime_mr_trend"

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the MR/Trend classification training."""
        start_time = time.time()

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            timeframe = str(config.get("timeframe", "15m"))

            # 1. Configuration & Context
            mr_trend_horizon = int(config.get("mr_trend_horizon", 12))  # Default 3h for 15m
            mr_trend_threshold = float(config.get("mr_trend_threshold", 0.015)) # 1.5%

            tprint_info(f"Starting {self.step_name} for {symbol} {timeframe}")
            tprint_info(f"  Horizon: {mr_trend_horizon} bars, Threshold: {mr_trend_threshold:.2%}")

            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction="long", # Direction agnostic really, but required by context
                model=self.model_namespace
            )

            # 2. Load Data
            market_data_config = {**config, "timeframe": timeframe}
            market_data, market_source = self.load_market_data_or_fail(
                market_data_config,
                pipeline_state={},
                allow_config_override=True
            )

            # Ensure DatetimeIndex
            if not isinstance(market_data.index, pd.DatetimeIndex):
                market_data.index = pd.to_datetime(market_data.index)

            # 3. Feature Engineering (Local)
            tprint_info("Generating specific MR/Trend features...")
            features_df = self._generate_features(market_data)

            # 4. Labeling
            tprint_info("Generating Regime Targets...")
            labeled_df = self._generate_labels(
                market_data,
                features_df,
                horizon=mr_trend_horizon,
                threshold=mr_trend_threshold
            )

            if labeled_df.empty:
                raise ValueError("Labeled dataset is empty")

            # Drop NaN rows created by lookbacks/forward looks
            labeled_df = labeled_df.dropna()
            tprint_info(f"  Dataset size after cleaning: {len(labeled_df)}")

            # Analyze Class Balance
            class_counts = labeled_df["target"].value_counts().sort_index()
            tprint_info(f"  Class Balance: {class_counts.to_dict()}")

            # 5. Training with Class Weights
            model, predictions, metrics, feature_pipeline = self._train_model(
                labeled_df,
                config
            )

            # 6. Save Artifacts
            # Save Training Data
            training_data_path = self._save_artifact(
                data=labeled_df.reset_index().rename(columns={"index": "timestamp"}),
                artifact_name="xgb_mr_trend_training_data",
                artifact_type="data",
                metadata={
                    "horizon": mr_trend_horizon,
                    "threshold": mr_trend_threshold,
                    "features": list(features_df.columns)
                }
            )

            # Save Model
            model_path = None
            if model:
                model_path = self._save_artifact(
                    data=model,
                    artifact_name="xgb_mr_trend_model",
                    artifact_type="model",
                    metadata={"model_type": "xgboost", "num_class": 3}
                )

            # 7. Validation Report
            self._generate_report(predictions, labeled_df["target"], metrics, config)

            execution_time = time.time() - start_time
            tprint_success(f"✅ {self.step_name} completed in {execution_time:.2f}s")

            return {
                "success": True,
                "training_data_path": training_data_path,
                "model_path": model_path,
                "metrics": metrics
            }

        except Exception as e:
            tprint_error(f"❌ {self.step_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # --------------------------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------------------------
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate local features specifically for MR vs Trend classification.
        Features: ATR, Volatility, LinReg Slope, SMA Slope, RSI, CMO, Stochastic, Dist from VWAP.
        """
        # Ensure we work with copies
        c = df["close"].copy()
        h = df["high"].copy()
        l = df["low"].copy()
        v = df["volume"].copy()

        features = pd.DataFrame(index=df.index)

        # --- A. Volatility & ATR ---
        # ATR 14
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        features["atr_14_norm"] = atr_14 / c # Normalized ATR

        # Rolling Volatility (Short/Med)
        ret = np.log(c / c.shift(1))
        features["vol_short_12"] = ret.rolling(12).std()
        features["vol_med_48"] = ret.rolling(48).std()

        # Volatility Regime (High/Low) - Ratio
        features["vol_ratio"] = features["vol_short_12"] / (features["vol_med_48"] + 1e-9)

        # --- B. Momentum & Slope ---
        # SMA Slopes (Normalized)
        sma_20 = c.rolling(20).mean()
        sma_50 = c.rolling(50).mean()

        # Slope as pct change of SMA
        features["sma_20_slope"] = (sma_20 - sma_20.shift(5)) / sma_20.shift(5)
        features["sma_50_slope"] = (sma_50 - sma_50.shift(5)) / sma_50.shift(5)

        # Linear Regression Slope (Log Price) over 20 bars
        try:
            import talib
            features["linreg_slope_20"] = talib.LINEARREG_SLOPE(np.log(c), timeperiod=20)
            features["linreg_angle_20"] = talib.LINEARREG_ANGLE(np.log(c), timeperiod=20)
        except ImportError:
            # Simple fallback: 3-point slope proxy (End - Start)
            features["linreg_slope_20"] = (np.log(c) - np.log(c).shift(20)) / 20.0

        # RSI 14
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema_12 = c.ewm(span=12, adjust=False).mean()
        ema_26 = c.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        features["macd_hist"] = macd - signal

        # --- C. Oscillators ---
        # CMO (Chande Momentum Oscillator) 14
        # CMO = 100 * (Su - Sd) / (Su + Sd)
        su = (delta.where(delta > 0, 0)).rolling(14).sum()
        sd = (-delta.where(delta < 0, 0)).rolling(14).sum()
        features["cmo_14"] = 100 * (su - sd) / (su + sd + 1e-9)

        # Stochastic (14, 3, 3)
        low_14 = l.rolling(14).min()
        high_14 = h.rolling(14).max()
        k_raw = 100 * (c - low_14) / (high_14 - low_14 + 1e-9)
        features["stoch_k"] = k_raw.rolling(3).mean()
        features["stoch_d"] = features["stoch_k"].rolling(3).mean()

        # --- D. Time Since Events ---
        # Time since Vol Spike (Z-Score > 2.0)
        # Using 100-bar window for baseline
        vol_baseline = ret.rolling(100).std()
        is_spike = (ret.rolling(20).std() > (2.0 * vol_baseline))
        features["vol_spikes_recent_20"] = is_spike.astype(int).rolling(20).sum()
        features["vol_spikes_recent_50"] = is_spike.astype(int).rolling(50).sum()

        # --- E. Support/Resistance & Mean Distance ---
        # Distance from SMA 50 (Z-Score equivalent)
        sma_50_std = c.rolling(50).std()
        features["z_score_50"] = (c - sma_50) / (sma_50_std + 1e-9)

        # Distance from VWAP (Approximation: using OHLC/3 cumulative)
        # For rolling VWAP (e.g., session or 1-day), on 15m 1 day is 96 bars
        tp = (h + l + c) / 3
        tp_v = tp * v
        vwap_96 = tp_v.rolling(96).sum() / (v.rolling(96).sum() + 1e-9)
        features["dist_vwap_96"] = (c - vwap_96) / vwap_96

        return features

    # --------------------------------------------------------------------------
    # Labeling
    # --------------------------------------------------------------------------
    def _generate_labels(self, market_df: pd.DataFrame, features_df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
        """
        Generate Target Labels:
        0: Noise
        1: Trend (TF)
        2: Mean Reversion (MR)
        """
        df = features_df.copy()
        c = market_df["close"]

        # Future Return
        future_ret = (c.shift(-horizon) - c) / c

        # Current Slope Proxy (using SMA 20 Slope calculated in features)
        slope = df["sma_20_slope"].fillna(0)

        # Z-Score (using z_score_50 from features)
        z_score = df["z_score_50"].fillna(0)

        # Logic
        # TF: Sign match AND Return > Threshold
        is_trend = (np.sign(future_ret) == np.sign(slope)) & (future_ret.abs() > threshold)

        # MR: Return < Threshold AND Price Far (Z > 2.0)
        is_mr = (future_ret.abs() < threshold) & (z_score.abs() > 2.0)

        # Assign Targets
        targets = np.zeros(len(df), dtype=int)
        targets[is_trend] = 1
        targets[is_mr] = 2

        df["target"] = targets

        # Save raw values for reporting
        df["future_ret_raw"] = future_ret
        df["z_score_raw"] = z_score

        # Drop columns with NaN (burn-in)
        return df

    # --------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------
    def _train_model(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Train XGBoost with Class Weights."""

        # Prepare Data
        feature_cols = [c for c in df.columns if c not in ["target", "future_ret_raw", "z_score_raw", "timestamp"]]
        X = df[feature_cols]
        y = df["target"]

        # Compute Sample Weights
        classes = np.unique(y)
        class_weights = {}
        n_samples = len(y)
        n_classes = len(classes)

        for cls in classes:
            n_samples_i = len(y[y == cls])
            if n_samples_i > 0:
                class_weights[cls] = n_samples / (n_classes * n_samples_i)
            else:
                class_weights[cls] = 0

        sample_weights = y.map(class_weights).values

        # Setup Trainer
        model_id = f"{config.get('symbol')}_mr_trend"
        trainer_config = XGBTrainingConfig(
            model_id=model_id,
            task_type="classification",
            objective="multi:softprob",
            num_class=3,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            early_stopping_rounds=30,
        )

        trainer = StandardizedXGBTrainer(model_id, trainer_config)

        # Train & Predict (OOF)
        # Note: StandardizedXGBTrainer handles OOF splitting internally,
        # but here we pass the FULL dataset with weights.
        results = trainer.train_and_predict(
            X=X,
            y=y,
            data_start=X.index.min(),
            data_end=X.index.max(),
            sample_weight=sample_weights,
            eval_metric="mlogloss"
        )

        best_model = results.models[-1] if results.models else None
        oof_preds = results.oof_predictions

        # Calculate Metrics (on OOF)
        metrics = {}
        if not oof_preds.empty:
            # Predictions are probs for 0, 1, 2
            # Use idxmax to get class
            prob_cols = [c for c in oof_preds.columns if "prob_class" in c]
            if prob_cols:
                pred_class = oof_preds[prob_cols].idxmax(axis=1).apply(lambda x: int(x.split('_')[-1]))

                # Align indices
                common_idx = pred_class.index.intersection(y.index)
                y_true = y.loc[common_idx]
                y_pred = pred_class.loc[common_idx]

                from sklearn.metrics import classification_report
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                metrics = report

        feature_pipeline = {"features": feature_cols}

        return best_model, oof_preds, metrics, feature_pipeline

    # --------------------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------------------
    def _generate_report(self, predictions: pd.DataFrame, targets: pd.Series, metrics: Dict, config: Dict):
        """Generate summary report."""
        if predictions.empty:
            return

        # Align
        common_idx = predictions.index.intersection(targets.index)
        y_true = targets.loc[common_idx]

        # Extract Probabilities
        # Assuming columns prob_class_0, prob_class_1, prob_class_2
        prob_cols = [c for c in predictions.columns if "prob_class" in c]
        if not prob_cols:
             return

        y_probs = predictions.loc[common_idx, prob_cols]
        y_pred = y_probs.idxmax(axis=1).apply(lambda x: int(x.split('_')[-1]))

        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        # Report Content
        report = []
        report.append(f"# MR vs Trend Classification Report: {config.get('symbol')}")
        report.append(f"**Date:** {datetime.now().isoformat()}")
        report.append(f"**Horizon:** {config.get('mr_trend_horizon')} bars")
        report.append(f"**Threshold:** {config.get('mr_trend_threshold'):.2%}")
        report.append("\n## Metrics")

        # Class 1 (TF) Metrics
        tf_m = metrics.get('1', {})
        report.append(f"**Trend (Class 1):** Precision: {tf_m.get('precision',0):.2f}, Recall: {tf_m.get('recall',0):.2f}, F1: {tf_m.get('f1-score',0):.2f}")

        # Class 2 (MR) Metrics
        mr_m = metrics.get('2', {})
        report.append(f"**MR (Class 2):** Precision: {mr_m.get('precision',0):.2f}, Recall: {mr_m.get('recall',0):.2f}, F1: {mr_m.get('f1-score',0):.2f}")

        report.append("\n## Confusion Matrix")
        # Handle cases where some classes might not exist in small tests
        labels = [0, 1, 2]
        if cm.shape == (3, 3):
             report.append(f"Noise (0): {cm[0]}")
             report.append(f"Trend (1): {cm[1]}")
             report.append(f"MR (2):    {cm[2]}")

        # Save
        filename = f"outcomes/mr_trend_report_{config.get('symbol')}_{int(time.time())}.md"
        with open(filename, "w") as f:
            f.write("\n".join(report))

        tprint_success(f"Report saved to {filename}")

    # --------------------------------------------------------------------------
    # Helper Methods for Sweep
    # --------------------------------------------------------------------------

    def generate_config_variations(self, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate grid search variations for sweep."""
        import itertools

        variations = {
            "mr_trend_horizon": [8, 12, 16, 24],
            "mr_trend_threshold": [0.01, 0.015, 0.02, 0.025],
        }

        keys = list(variations.keys())
        values = list(variations.values())

        configs = []
        max_configs = int(base_config.get("sweep_max_configs", 20))

        for combo in itertools.product(*values):
            if len(configs) >= max_configs:
                break

            cfg_update = dict(zip(keys, combo))
            new_config = base_config.copy()
            new_config.update(cfg_update)
            new_config["config_signature"] = f"H{cfg_update['mr_trend_horizon']}_T{cfg_update['mr_trend_threshold']}"
            configs.append(new_config)

        return configs

    async def run_config_batch(self, configs: List[Dict[str, Any]], symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """Run a batch of configurations and collect results."""
        results = []
        for i, config in enumerate(configs):
            tprint_info(f"Running sweep config {i+1}/{len(configs)}")
            try:
                res = await self.execute(config)

                metrics = res.get("metrics", {})
                # Extract F1 for Class 1 and Class 2
                f1_trend = metrics.get('1', {}).get('f1-score', 0)
                f1_mr = metrics.get('2', {}).get('f1-score', 0)
                weighted_score = (f1_trend + f1_mr) / 2

                results.append({
                    "config_id": i,
                    "signature": config.get("config_signature"),
                    "mr_trend_horizon": config.get("mr_trend_horizon"),
                    "mr_trend_threshold": config.get("mr_trend_threshold"),
                    "f1_trend": f1_trend,
                    "f1_mr": f1_mr,
                    "weighted_score": weighted_score,
                    "success": True
                })
            except Exception as e:
                results.append({"config_id": i, "success": False, "error": str(e)})

        return results

    def analyze_and_rank_results(self, results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Rank results by weighted score."""
        df = pd.DataFrame(results)
        if df.empty or "weighted_score" not in df.columns:
            return df, {}

        df = df.sort_values("weighted_score", ascending=False)
        best = df.iloc[0].to_dict()
        return df, {"best_config": best}
