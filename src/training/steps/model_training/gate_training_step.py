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
            direction = config.get('direction', 'long')
            execution_mode = config.get('execution_mode', 'light')

            # Ensure context is aligned with analyst base training so artifacts resolve correctly
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model="analyst",
                execution_mode=execution_mode,
            )

            # Try to load OHLCV using the shared BaseStep helper for consistency
            ohlcv_data, source = self.load_market_data_or_fail(
                config,
                pipeline_state={},
                allow_config_override=False,
            )

            if ohlcv_data is None or not isinstance(ohlcv_data, pd.DataFrame) or ohlcv_data.empty:
                raise ValueError("Could not load OHLCV data for gate training")

            tprint_success(f"Loaded OHLCV data from {source}: {ohlcv_data.shape}")
            if isinstance(ohlcv_data, pd.DataFrame) and not ohlcv_data.empty:
                tprint(
                    f"[GateTrainingStep] OHLCV index range: {ohlcv_data.index.min()} → {ohlcv_data.index.max()} (n={len(ohlcv_data)})",
                    "INFO",
                )

            # Load Main Model OOF Predictions – prefer analyst base OOF outputs
            oof_preds = self._get_artifact(
                "analyst_base_predictions_oof",
                artifact_type="data",
                data_category="predictions",
            )

            if oof_preds is None:
                # Fallback to ensemble outputs if base OOF predictions are not available
                oof_preds = self._get_artifact(
                    "analyst_ensemble_outputs_oof",
                    artifact_type="data",
                    data_category="predictions",
                )
            if oof_preds is None:
                oof_preds = self._get_artifact(
                    "analyst_ensemble_outputs",
                    artifact_type="data",
                    data_category="predictions",
                )

            if oof_preds is None:
                raise ValueError(
                    "Could not load main model predictions for gate training. "
                    "Expected 'analyst_base_predictions_oof' or analyst ensemble outputs."
                )

            tprint_success(f"Loaded prediction artifact for gating: {getattr(oof_preds, 'shape', 'N/A')}")
            if isinstance(oof_preds, (pd.Series, pd.DataFrame)) and len(oof_preds) > 0:
                oof_idx = pd.to_datetime(oof_preds.index)
                tprint(
                    f"[GateTrainingStep] Raw OOF index range: {oof_idx.min()} → {oof_idx.max()} (n={len(oof_idx)})",
                    "INFO",
                )

            # Align indices: snap OOF predictions to the timeframe bar grid
            if isinstance(oof_preds, (pd.Series, pd.DataFrame)):
                oof_index = pd.to_datetime(oof_preds.index)

                tf = str(timeframe).lower().strip()
                if tf.endswith("m") and tf[:-1].isdigit():
                    freq = f"{int(tf[:-1])}min"
                elif tf.endswith("h") and tf[:-1].isdigit():
                    freq = f"{int(tf[:-1])}H"
                elif tf.endswith("d") and tf[:-1].isdigit():
                    freq = f"{int(tf[:-1])}D"
                elif tf.endswith("w") and tf[:-1].isdigit():
                    freq = f"{int(tf[:-1])}W"
                else:
                    freq = "15min"

                aligned_index = oof_index.floor(freq)

                if isinstance(oof_preds, pd.Series):
                    oof_df = oof_preds.to_frame("prediction")
                else:
                    oof_df = oof_preds.copy()

                oof_df.index = aligned_index
                oof_grouped = oof_df.groupby(oof_df.index).mean()
                oof_preds = oof_grouped

                tprint(
                    f"[GateTrainingStep] Bar-aligned OOF index range: {oof_preds.index.min()} → {oof_preds.index.max()} (n={len(oof_preds)})",
                    "INFO",
                )

            common_idx = ohlcv_data.index.intersection(oof_preds.index)
            tprint(
                f"[GateTrainingStep] common_idx length after alignment: {len(common_idx)}",
                "INFO",
            )
            ohlcv_data = ohlcv_data.loc[common_idx]
            oof_preds = oof_preds.loc[common_idx]

            # Extract single prediction column (assuming 'prediction' or first column)
            if isinstance(oof_preds, pd.DataFrame):
                # Prefer 'prediction' col if exists, else first col
                if 'prediction' in oof_preds.columns:
                    pred_col = 'prediction'
                elif 'lightgbm' in oof_preds.columns:
                    pred_col = 'lightgbm'
                else:
                    numeric_cols = oof_preds.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        pred_col = numeric_cols[0]
                    else:
                        pred_col = oof_preds.columns[0]
                preds = oof_preds[pred_col].astype(float)
                if len(preds) > 0:
                    non_nan = preds.notna().sum()
                    try:
                        p_min = float(preds.min())
                        p_max = float(preds.max())
                        p_head = float(preds.iloc[0])
                    except Exception:
                        p_min = p_max = p_head = float('nan')
                    tprint(
                        f"[GateTrainingStep] preds column='{pred_col}', non_nan={non_nan}, min={p_min:.6f}, max={p_max:.6f}, head={p_head:.6f}",
                        "INFO",
                    )
            else:
                preds = oof_preds

            # 2. Simulate Ungated Trades (Baseline)
            # Define Signal Logic with auto-relaxation if no candidates
            signal_threshold = config.get('gate_config', {}).get('signal_threshold', 0.05)  # default
            # Main model output is continuous; we use abs(prediction) > threshold.
            # If the configured threshold yields no candidates, relax to 0.0 so that
            # any directional conviction can be considered for gating.

            thresholds_to_try = [signal_threshold]
            if signal_threshold != 0.0:
                thresholds_to_try.append(0.0)

            long_signals = None
            short_signals = None
            candidates_mask = None
            candidate_indices = None
            effective_threshold = None

            for thr in thresholds_to_try:
                long_signals = preds > thr
                short_signals = preds < -thr

                # Construct Candidate Universe
                candidates_mask = long_signals | short_signals
                candidate_indices = candidates_mask[candidates_mask].index

                if len(candidate_indices) > 0:
                    effective_threshold = thr
                    if thr != signal_threshold:
                        tprint_warning(
                            f"No candidates at signal_threshold={signal_threshold:.4f}; "
                            f"relaxed to {thr:.4f} and found {len(candidate_indices)} candidates."
                        )
                    break

            if candidate_indices is None or len(candidate_indices) == 0:
                tprint_warning(
                    f"No candidates found even after relaxing signal_threshold (initial={signal_threshold}). "
                    "Skipping gate training."
                )
                return {'success': True, 'message': 'No candidates'}

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

            labeled_data = self._get_artifact("labeled_data", artifact_type="data")
            targets = None
            if labeled_data is not None:
                # Use pre-calculated targets if available
                # These targets (e.g. 'target_long') usually capture the "potential profit" logic
                targets = labeled_data.reindex(common_idx)
                tprint("Using labeled_data for trade outcomes.", "INFO")

            tprint(
                f"Candidate Universe: {len(candidate_indices)} signals "
                f"(effective_threshold={effective_threshold if effective_threshold is not None else signal_threshold:.4f})",
                "INFO",
            )

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

            # Basic diagnostics on NaNs at the feature level
            tprint(
                f"[GateTrainingStep] X_full shape before NaN handling: {X_full.shape}",
                "INFO",
            )
            if isinstance(X_full, pd.DataFrame) and len(X_full) > 0:
                nan_per_col = X_full.isnull().sum()
                # Log top features by NaN count for debugging
                try:
                    top_nan = nan_per_col.sort_values(ascending=False).head(10).to_dict()
                    tprint(
                        f"[GateTrainingStep] Top NaN-heavy features (count): {top_nan}",
                        "INFO",
                    )
                except Exception:
                    pass

                # Drop features that are mostly NaN, keep the rest and rely on the model's
                # internal imputer for residual sparsity.
                max_nan_frac = gate_config.get('max_nan_fraction', 0.8)
                col_nan_frac = nan_per_col / float(len(X_full))
                bad_cols = col_nan_frac[col_nan_frac > max_nan_frac].index.tolist()
                if bad_cols:
                    preview = bad_cols[:10]
                    more_flag = "..." if len(bad_cols) > 10 else ""
                    tprint_warning(
                        f"[GateTrainingStep] Dropping {len(bad_cols)} features with NaN fraction > {max_nan_frac}: {preview}{more_flag}",
                    )
                    X_full = X_full.drop(columns=bad_cols)

            if not isinstance(X_full, pd.DataFrame) or X_full.shape[1] == 0:
                tprint_error("[GateTrainingStep] No usable features after NaN-based column pruning. Skipping training.")
                return {'success': False, 'error': 'No features after NaN pruning'}

            # Slice for training
            X_train = X_full.loc[candidate_indices]
            y_train = y  # y is already aligned to candidate_indices

            # Row-level NaN handling: allow partial NaNs and rely on the model's imputer.
            # Require that each row has at least a minimum fraction of non-NaN features.
            if isinstance(X_train, pd.DataFrame) and X_train.shape[1] > 0:
                non_nan_counts = X_train.notnull().sum(axis=1)
                min_non_nan_frac = gate_config.get('min_non_nan_fraction', 0.5)
                min_non_nan = max(1, int(np.ceil(min_non_nan_frac * X_train.shape[1])))
                valid_mask = (non_nan_counts >= min_non_nan) & ~y_train.isnull()
                tprint(
                    f"[GateTrainingStep] NaN-aware row filter: kept {int(valid_mask.sum())} / {len(X_train)} "
                    f"rows with >= {min_non_nan} non-NaN features (frac={min_non_nan_frac:.2f}).",
                    "INFO",
                )
                X_train = X_train[valid_mask]
                y_train = y_train[valid_mask]
            else:
                tprint_error("[GateTrainingStep] X_train is empty or not a DataFrame after slicing. Skipping training.")
                return {'success': False, 'error': 'Empty training matrix'}

            min_samples = 50
            if len(X_train) < min_samples:
                tprint_warning(
                    f"Insufficient valid samples after feature generation (n={len(X_train)} < {min_samples}). Skipping training."
                )
                return {'success': False, 'error': 'Insufficient samples'}

            # Train
            model.train(X_train, y_train)

            # Calibrate Threshold (default 40th percentile -> keep top 60%)
            model.calibrate_threshold(X_train, percentile=40)

            # 4. Evaluate (classification metrics + trade-level pre/post metrics)
            probs = model.predict_proba(X_train)
            preds_bin = model.predict(X_train)

            # Restrict net_returns to the same index used for training to ensure alignment
            train_indices = X_train.index
            baseline_returns = net_returns.loc[train_indices]
            accepted_returns = baseline_returns[preds_bin.astype(bool)]

            def _summary_stats(returns: pd.Series) -> Dict[str, float]:
                if returns is None or len(returns) == 0:
                    return {
                        'count': 0,
                        'win_rate': 0.0,
                        'avg_return': 0.0,
                        'median_return': 0.0,
                        'total_return': 0.0,
                    }
                wins = returns[returns > 0]
                win_rate = float((returns > 0).mean())
                avg_ret = float(returns.mean())
                med_ret = float(returns.median())
                # Approximate compounded return over trades
                total_ret = float((1.0 + returns).prod() - 1.0)
                return {
                    'count': int(len(returns)),
                    'win_rate': win_rate,
                    'avg_return': avg_ret,
                    'median_return': med_ret,
                    'total_return': total_ret,
                }

            pre_stats = _summary_stats(baseline_returns)
            post_stats = _summary_stats(accepted_returns)

            coverage = (
                float(post_stats['count'] / pre_stats['count'])
                if pre_stats['count'] > 0 else 0.0
            )
            avg_lift = float(post_stats['avg_return'] - pre_stats['avg_return'])

            metrics = {
                'auc': float(roc_auc_score(y_train, probs)),
                'precision': float(precision_score(y_train, preds_bin)),
                'recall': float(recall_score(y_train, preds_bin)),
                'f1': float(f1_score(y_train, preds_bin)),
                'accuracy': float(accuracy_score(y_train, preds_bin)),
                'candidate_count': len(y_train),
                'accepted_count': int(preds_bin.sum()),
                'acceptance_rate': float(preds_bin.mean()),
                # Trade-level pre/post metrics based on realized net_returns
                'pre_trade_count': pre_stats['count'],
                'pre_win_rate': pre_stats['win_rate'],
                'pre_avg_return': pre_stats['avg_return'],
                'pre_median_return': pre_stats['median_return'],
                'pre_total_return': pre_stats['total_return'],
                'post_trade_count': post_stats['count'],
                'post_win_rate': post_stats['win_rate'],
                'post_avg_return': post_stats['avg_return'],
                'post_median_return': post_stats['median_return'],
                'post_total_return': post_stats['total_return'],
                'coverage_rate': coverage,
                'avg_return_lift': avg_lift,
            }

            tprint_success(
                f"Gate Training Metrics: AUC={metrics['auc']:.4f}, "
                f"Precision={metrics['precision']:.4f}, AcceptRate={metrics['acceptance_rate']:.2%}, "
                f"PreWin={metrics['pre_win_rate']:.2%}, PostWin={metrics['post_win_rate']:.2%}, "
                f"AvgLift={metrics['avg_return_lift']:.5f}"
            )

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
