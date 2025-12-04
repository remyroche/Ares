"""
Gate Training Step.

This step orchestrates the training of the Gate Model.
It loads data, simulates ungated trades, computes features, trains the model,
and saves artifacts.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

from src.training.steps.base_step import BaseStep
from src.training.steps.model_training.gate_model import GateModel
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error

class GateTrainingStep(BaseStep):
    """
    Step to train the Gate Model.
    """

    def __init__(self, step_name: str = "gate_training_step"):
        super().__init__(step_name)
        self.logger = system_logger.getChild('GateTrainingStep')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the gate training workflow.

        Args:
            config: Configuration dictionary containing:
                - symbol, timeframe, exchange
                - gate_config: specific parameters for the gate

        Returns:
            Dict with success status and artifact paths.
        """
        try:
            tprint(f"🚪 Starting Gate Training for {config.get('symbol')}", "INFO")

            # 1. Load Data (OHLCV + Main Model OOF Predictions)
            # We assume OHLCV is available via normal data loading or artifacts
            # We assume OOF predictions are in 'analyst_ensemble_outputs_oof'

            # Load OHLCV (using feature_generation artifacts for simplicity/consistency)
            # Or use raw data loader if preferred. Let's try 'selected_features' first as it has OHLCV-aligned index
            # Actually, GateModel calculates its own regime features from raw OHLCV.
            # So we need raw OHLCV.
            # We can get it from 'data_validation_step' artifact 'validated_data' or similar.
            # Or simpler: use 'selected_features' and extract/reconstruct if OHLCV columns present.
            # Best robust way: Load from 'validated_data' artifact if possible.

            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')

            # Try to get validated data (OHLCV)
            ohlcv_data = await self._get_artifact(f"validated_data_{symbol}_{timeframe}", config)
            if ohlcv_data is None:
                # Fallback to generic name
                ohlcv_data = await self._get_artifact("validated_data", config)

            if ohlcv_data is None:
                raise ValueError("Could not load OHLCV data (validated_data artifact missing)")

            tprint_success(f"Loaded OHLCV data: {ohlcv_data.shape}")

            # Load Main Model OOF Predictions
            oof_preds = await self._get_artifact("analyst_ensemble_outputs_oof", config)
            if oof_preds is None:
                 # Try fallback
                 oof_preds = await self._get_artifact("analyst_ensemble_outputs", config)

            if oof_preds is None:
                raise ValueError("Could not load Main Model OOF predictions (analyst_ensemble_outputs_oof missing)")

            tprint_success(f"Loaded OOF Predictions: {oof_preds.shape}")

            # Align indices
            common_idx = ohlcv_data.index.intersection(oof_preds.index)
            ohlcv_data = ohlcv_data.loc[common_idx]
            oof_preds = oof_preds.loc[common_idx]

            # Extract single prediction column (assuming 'prediction' or first column)
            if isinstance(oof_preds, pd.DataFrame):
                # Prefer 'prediction' col if exists, else first col
                if 'prediction' in oof_preds.columns:
                    preds = oof_preds['prediction']
                else:
                    preds = oof_preds.iloc[:, 0]
            else:
                preds = oof_preds

            # 2. Simulate Ungated Trades (Baseline)
            # Define Signal Logic
            signal_threshold = config.get('gate_config', {}).get('signal_threshold', 0.05) # Example default
            # Actually, main model output is likely continuous. Let's assume standard scaler-like output or prob.
            # User req: "abs(model_prediction) > signal_strength_threshold"
            # Let's assume prediction is directional (-1 to 1 or similar).

            long_signals = preds > signal_threshold
            short_signals = preds < -signal_threshold

            # We need to simulate the trades to get 'profit' for labeling
            # Simple simulation:
            # Entry: Next Open (or current Close if vectorizing simply) -> Let's use Close to Close for simplicity/speed in this step,
            # or better: Vectorized simulation with fixed horizon or TP/SL.
            # User req: "realized trade (executed as the main model would have executed it) produced profit after costs"
            # Since we don't have the full Tactician logic here, we'll use a simplified assumption:
            # Fixed horizon hold (e.g. 12 bars ~ 3 hours) or until TP/SL.
            # Let's use a standard simplified horizon for the label definition if not provided.
            # Better: if 'labeled_data' exists, use realized returns from there?
            # 'labeled_data' usually has 'target_long', 'target_short'.
            # If those targets represent "Profit after costs", we can use them directly!
            # Let's check if we can use 'labeled_data'.

            labeled_data = await self._get_artifact("labeled_data", config)
            targets = None
            if labeled_data is not None:
                # Use pre-calculated targets if available
                # These targets (e.g. 'target_long') usually capture the "potential profit" logic
                targets = labeled_data.reindex(common_idx)
                tprint("Using labeled_data for trade outcomes.", "INFO")

            # Construct Candidate Universe
            candidates_mask = long_signals | short_signals
            candidate_indices = candidates_mask[candidates_mask].index

            if len(candidate_indices) == 0:
                tprint_warning("No candidates found (no signals > threshold). Skipping gate training.")
                return {'success': True, 'message': 'No candidates'}

            tprint(f"Candidate Universe: {len(candidate_indices)} signals", "INFO")

            # Create Ungated Trade Log (for history features)
            # We need a log of ALL signals as if they were trades.
            trade_log_rows = []

            # Transaction Costs
            tc_bps = config.get('gate_config', {}).get('transaction_costs_bps', 5.0)
            cost_pct = tc_bps / 10000.0

            # Determine outcome for each candidate
            # If we have 'labeled_data', 'target_long' > 0 means profit.
            # If not, we calculate simple forward return.

            # Helper to get forward return
            def get_forward_ret(idx, direction, horizon=12):
                # Simple horizon return
                try:
                    curr_price = ohlcv_data.loc[idx, 'close']
                    future_idx_pos = ohlcv_data.index.get_loc(idx) + horizon
                    if future_idx_pos >= len(ohlcv_data):
                        return 0.0
                    future_price = ohlcv_data.iloc[future_idx_pos]['close']
                    ret = (future_price - curr_price) / curr_price
                    return ret * direction
                except:
                    return 0.0

            labels = []
            valid_indices = []

            # We iterate to build trade log. Vectorization is preferred but for log building loop is ok-ish if N is small.
            # candidates are a subset.
            # Actually, `prepare_features` needs the trade log.

            tprint("Simulating ungated trades...", "INFO")

            # Vectorized approach for efficiency
            # 1. Filter data to candidates
            cand_df = ohlcv_data.loc[candidate_indices].copy()
            cand_preds = preds.loc[candidate_indices]

            # 2. Calculate outcomes
            # Use labeled_data targets if aligned, else compute forward returns
            if targets is not None:
                # If signal is Long, look at target_long
                # If signal is Short, look at target_short (or inverse of target_long if single target)
                # Assuming 'target_long' and 'target_short' columns exist
                outcomes = pd.Series(0.0, index=candidate_indices)
                if 'target_long' in targets.columns:
                    mask_l = cand_preds > 0
                    outcomes[mask_l] = targets.loc[candidate_indices[mask_l], 'target_long']
                if 'target_short' in targets.columns:
                    mask_s = cand_preds < 0
                    outcomes[mask_s] = targets.loc[candidate_indices[mask_s], 'target_short']
                elif 'target_long' in targets.columns:
                     # Fallback if no specific short target, assume symmetry if appropriate or 0
                     pass
            else:
                # Compute returns manually (simplified 12-bar horizon)
                # This is slower
                # Let's assume we rely on labeled_data for consistency with Main Model training
                pass

            # If we don't have outcomes yet (no labeled_data), fast calc:
            if 'outcomes' not in locals():
                 # Fast shift
                 horizon = 12
                 future_close = ohlcv_data['close'].shift(-horizon)
                 ret = (future_close - ohlcv_data['close']) / ohlcv_data['close']

                 cand_ret = ret.loc[candidate_indices]
                 cand_dir = np.sign(cand_preds)
                 outcomes = cand_ret * cand_dir

            # Apply costs
            # Net Profit approx = outcome - (2 * cost) # entry + exit
            # outcome is usually a return.
            net_returns = outcomes - (2 * cost_pct)

            # Create Trade Log DataFrame
            # We assume exit time is entry + horizon (simplified)
            # Note: For feature calc, exit_time is crucial.
            # Assuming fixed 15m bars, 12 bars = 3 hours.
            exit_times = candidate_indices + pd.Timedelta(minutes=15*12)

            trade_log = pd.DataFrame({
                'entry_time': candidate_indices,
                'exit_time': exit_times,
                'profit': net_returns.values, # This serves as PnL proxy
                'realized_return': net_returns.values
            })

            # Generate Labels (Target A)
            y = (net_returns > 0).astype(int)

            tprint_success(f"Generated {len(y)} labels. Positive rate: {y.mean():.2%}")

            # 3. Initialize and Train GateModel
            gate_config = config.get('gate_config', {})
            model = GateModel(config=gate_config)

            # Prepare Features
            tprint("Generating features...", "INFO")
            # We calculate features on ALL ohlcv data first?
            # No, `prepare_features` takes full ohlcv to calculate rolling windows correctly.
            # Then we slice to `candidate_indices` for training.

            X_full = model.prepare_features(ohlcv_data, trade_log)

            # Slice for training
            X_train = X_full.loc[candidate_indices]
            y_train = y # y is already aligned to candidate_indices

            # Drop NaNs
            valid_mask = ~X_train.isnull().any(axis=1) & ~y_train.isnull()
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

            if len(X_train) < 100:
                tprint_warning("Insufficient valid samples after feature generation. Skipping training.")
                return {'success': False, 'error': 'Insufficient samples'}

            # Train
            model.train(X_train, y_train)

            # Calibrate Threshold (default 40th percentile -> keep top 60%)
            model.calibrate_threshold(X_train, percentile=40)

            # 4. Evaluate
            probs = model.predict_proba(X_train)
            preds_bin = model.predict(X_train)

            metrics = {
                'auc': float(roc_auc_score(y_train, probs)),
                'precision': float(precision_score(y_train, preds_bin)),
                'recall': float(recall_score(y_train, preds_bin)),
                'f1': float(f1_score(y_train, preds_bin)),
                'accuracy': float(accuracy_score(y_train, preds_bin)),
                'candidate_count': len(y_train),
                'accepted_count': int(preds_bin.sum()),
                'acceptance_rate': float(preds_bin.mean())
            }

            tprint_success(f"Gate Training Metrics: AUC={metrics['auc']:.4f}, Precision={metrics['precision']:.4f}, AcceptRate={metrics['acceptance_rate']:.2%}")

            # 5. Save Artifacts
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save Model
            model_filename = f"gate_model_{symbol}_{timeframe}_{timestamp_str}.joblib"
            # Use a temporary path or ensure artifact manager handles paths.
            # BaseStep._save_artifact handles paths if we pass data.
            # GateModel.save writes to a file. We can write to a temp file then upload/register.

            # Workaround: serialize model components manually or use save locally then artifact manager?
            # Better: Create a dictionary of model state and save as 'model' type artifact.
            # Actually GateModel has a save method. Let's use it to a temp path.
            temp_path = os.path.join("artifacts", model_filename) # Local temp
            os.makedirs("artifacts", exist_ok=True)
            model.save(temp_path)

            # Register this file as an artifact
            # BaseStep doesn't strictly support uploading arbitrary files easily in all versions,
            # but usually we return paths or use specific methods.
            # Assuming we can just return the path in the artifacts dict for now,
            # or if we want to use the artifact store properly:
            # We can pickling the object and save it using _save_artifact.
            # But GateModel uses joblib which is file-based.
            # Let's trust the temp file approach for now and return the path.

            # Save OOF Predictions (Gate scores on candidates)
            oof_df = pd.DataFrame({
                'gate_score': probs,
                'gate_decision': preds_bin,
                'target': y_train
            }, index=X_train.index)

            oof_path = self._save_artifact(oof_df, f"gate_oof_predictions_{symbol}", "data")

            # Save Metrics
            metrics_path = self._save_artifact(metrics, f"gate_metrics_{symbol}", "metadata")

            return {
                'success': True,
                'artifacts': {
                    'gate_model': temp_path,
                    'gate_oof_predictions': oof_path,
                    'gate_metrics': metrics_path
                },
                'metrics': metrics
            }

        except Exception as e:
            tprint_error(f"Gate Training Failed: {str(e)}")
            self.logger.error(f"Gate Training Failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

def register_gate_training_step():
    """Register the gate training step."""
    from src.training.steps.base_step import step_registry
    step_registry.register("gate_training_step", GateTrainingStep)
    tprint("✅ Gate training step registered", "SUCCESS")

# Auto-register
register_gate_training_step()
