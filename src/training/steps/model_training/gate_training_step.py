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
from sklearn.metrics import mean_squared_error, r2_score

from src.training.steps.base_step import BaseStep
from src.training.steps.model_training.gate_model import GateModel
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error
from src.utils.versioned_artifacts.temporal_splits import TemporalSplitConfig
from src.utils.ml_common.confidence_metrics import calculate_calibration_metrics

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

            if isinstance(oof_preds, np.ndarray):
                arr = np.asarray(oof_preds).reshape(-1)
                n_ohlcv = len(ohlcv_data.index)
                if n_ohlcv == 0:
                    raise ValueError("OHLCV data is empty when aligning ndarray predictions for gate training")
                if arr.shape[0] != n_ohlcv:
                    n = min(arr.shape[0], n_ohlcv)
                    tprint_warning(
                        f"[GateTrainingStep] ndarray predictions length {arr.shape[0]} != OHLCV length {n_ohlcv}; "
                        f"aligning using last {n} samples."
                    )
                    arr = arr[-n:]
                    idx = ohlcv_data.index[-n:]
                else:
                    idx = ohlcv_data.index
                oof_preds = pd.DataFrame({'prediction': arr.astype(float)}, index=idx)

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
                gate_cfg = config.get('gate_config', {})
                main_model_col = None
                if isinstance(gate_cfg, dict):
                    main_model_col = gate_cfg.get('main_model_column')

                # Prefer explicitly configured main_model_column when available,
                # then fall back to 'prediction', then 'lightgbm', then first numeric/any column.
                pred_col = None
                if isinstance(main_model_col, str) and main_model_col in oof_preds.columns:
                    pred_col = main_model_col
                elif 'prediction' in oof_preds.columns:
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
            gate_config = config.setdefault('gate_config', {})
            signal_threshold = gate_config.get('signal_threshold', 0.6)  # default

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

            # Transaction Costs
            tc_bps = config.get('gate_config', {}).get('transaction_costs_bps', 5.0)
            cost_pct = tc_bps / 10000.0

            # Determine outcome for each candidate
            tprint("Simulating ungated trades...", "INFO")

            cand_preds = preds.loc[candidate_indices]

            # 2. Calculate outcomes
            # Use labeled_data targets if aligned, else compute forward returns
            if targets is not None:
                # If signal is Long, look at target_long
                # If signal is Short, look at target_short
                outcomes = pd.Series(0.0, index=candidate_indices)
                if 'target_long' in targets.columns:
                    mask_l = cand_preds > 0
                    outcomes[mask_l] = targets.loc[candidate_indices[mask_l], 'target_long']
                if 'target_short' in targets.columns:
                    mask_s = cand_preds < 0
                    outcomes[mask_s] = targets.loc[candidate_indices[mask_s], 'target_short']
                elif 'target_long' in targets.columns:
                     # Fallback if no specific short target, assume symmetry
                     pass
            else:
                # Fallback if no labeled_data
                 horizon = 12
                 future_close = ohlcv_data['close'].shift(-horizon)
                 ret = (future_close - ohlcv_data['close']) / ohlcv_data['close']

                 cand_ret = ret.loc[candidate_indices]
                 cand_dir = np.sign(cand_preds)
                 outcomes = cand_ret * cand_dir

            # Apply costs
            # Net Profit approx = outcome - (2 * cost) # entry + exit
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

            # Generate Labels (Classification Target)
            y = (net_returns > 0).astype(int)

            # Generate Sample Weights based on PnL magnitude (penalize/reward more heavily)
            # Using log-dampened formula: log(1 + |NetReturn|) to handle skewed distributions
            # Normalize to mean 1.0 to preserve effective learning rate
            sample_weights = np.log1p(np.abs(net_returns))
            if sample_weights.mean() > 0:
                sample_weights = sample_weights / sample_weights.mean()
            # Ensure minimum weight for stability
            sample_weights = np.maximum(sample_weights, 0.1)

            tprint_success(f"Generated {len(y)} classification targets. Mean PnL: {net_returns.mean():.4f}")

            # 3. Initialize and Train GateModel
            gate_config = config.get('gate_config', {})
            # If the user has not provided an explicit calibration target, default
            # to blocking trades whose predicted win probability is below 0.5.
            if (
                'min_predicted_pnl' not in gate_config
                and 'calibration_percentile' not in gate_config
                and 'target_coverage' not in gate_config
                and 'min_win_probability' not in gate_config
            ):
                gate_config['min_win_probability'] = 0.55 # Default higher threshold for safety
            model = GateModel(config=gate_config)

            # Prepare Features
            tprint("Generating features...", "INFO")
            X_full = model.prepare_features(ohlcv_data, trade_log, preds=preds)

            # Basic diagnostics on NaNs at the feature level
            tprint(
                f"[GateTrainingStep] X_full shape before NaN handling: {X_full.shape}",
                "INFO",
            )
            if isinstance(X_full, pd.DataFrame) and len(X_full) > 0:
                nan_per_col = X_full.isnull().sum()
                # Drop features that are mostly NaN
                max_nan_frac = gate_config.get('max_nan_fraction', 0.8)
                col_nan_frac = nan_per_col / float(len(X_full))
                bad_cols = col_nan_frac[col_nan_frac > max_nan_frac].index.tolist()
                if bad_cols:
                    tprint_warning(
                        f"[GateTrainingStep] Dropping {len(bad_cols)} features with NaN fraction > {max_nan_frac}.",
                    )
                    X_full = X_full.drop(columns=bad_cols)

            if not isinstance(X_full, pd.DataFrame) or X_full.shape[1] == 0:
                tprint_error("[GateTrainingStep] No usable features after NaN-based column pruning. Skipping training.")
                return {'success': False, 'error': 'No features after NaN pruning'}

            # Slice for training
            X_train = X_full.loc[candidate_indices]
            y_train = y  # y is already aligned to candidate_indices
            w_train = sample_weights.loc[candidate_indices] # Align weights

            # Row-level NaN handling
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
                w_train = w_train[valid_mask]
            else:
                tprint_error("[GateTrainingStep] X_train is empty or not a DataFrame after slicing. Skipping training.")
                return {'success': False, 'error': 'Empty training matrix'}

            min_samples = 50
            if len(X_train) < min_samples:
                tprint_warning(
                    f"Insufficient valid samples after feature generation (n={len(X_train)} < {min_samples}). Skipping training."
                )
                return {'success': False, 'error': 'Insufficient samples'}

            # Temporal train/validation/test split (purged, chronological)
            idx = pd.to_datetime(X_train.index)
            train_mask = np.ones(len(X_train), dtype=bool)
            val_mask = np.zeros(len(X_train), dtype=bool)
            test_mask = np.zeros(len(X_train), dtype=bool)

            temporal_config = None
            try:
                temporal_config = TemporalSplitConfig.create_from_data(
                    data_start=idx.min().to_pydatetime(),
                    data_end=idx.max().to_pydatetime(),
                    train_pct=gate_config.get('train_pct', 0.6),
                    val_pct=gate_config.get('val_pct', 0.2),
                    test_pct=gate_config.get('test_pct', 0.2),
                    embargo_days=gate_config.get('embargo_days', 1),
                    burnin_pct=0.0,
                )
            except Exception as e:
                tprint_warning(
                    f"[GateTrainingStep] Failed to create TemporalSplitConfig: {e}. Falling back to simple chronological split."
                )

            if temporal_config is not None:
                train_mask = (idx >= temporal_config.training.start) & (idx <= temporal_config.training.effective_end)
                val_mask = (idx >= temporal_config.validation.start) & (idx <= temporal_config.validation.effective_end)
                test_mask = (idx >= temporal_config.test.start) & (idx <= temporal_config.test.effective_end)

                # Fallback if validation or training is too small
                if train_mask.sum() < 30 or val_mask.sum() < 20:
                    tprint_warning(
                        f"[GateTrainingStep] Temporal split produced small folds (train={train_mask.sum()}, val={val_mask.sum()}); using simple chronological split instead."
                    )
                    order = np.argsort(idx.values)
                    n_total = len(X_train)
                    train_end = int(n_total * 0.6)
                    val_end = int(n_total * 0.8)
                    train_mask = np.zeros(n_total, dtype=bool)
                    val_mask = np.zeros(n_total, dtype=bool)
                    test_mask = np.zeros(n_total, dtype=bool)
                    train_mask[order[:train_end]] = True
                    val_mask[order[train_end:val_end]] = True
                    test_mask[order[val_end:]] = True

            X_train_fit = X_train[train_mask]
            y_train_fit = y_train[train_mask]
            w_train_fit = w_train[train_mask]

            X_val = X_train[val_mask]
            y_val = y_train[val_mask]
            X_test = X_train[test_mask]
            y_test = y_train[test_mask]

            tprint(
                f"[GateTrainingStep] Temporal split sizes - train={len(X_train_fit)}, val={len(X_val)}, test={len(X_test)}",
                "INFO",
            )

            if len(X_train_fit) < min_samples:
                tprint_warning(
                    f"Insufficient training samples after temporal split (n={len(X_train_fit)} < {min_samples}). Skipping training."
                )
                return {'success': False, 'error': 'Insufficient samples after temporal split'}

            # Train on training window only, using sample weights
            model.train(X_train_fit, y_train_fit, sample_weight=w_train_fit)

            # Probability calibration on validation window (isotonic)
            val_brier_raw = None
            val_brier_cal = None
            val_ece_raw = None
            val_ece_cal = None

            if isinstance(X_val, pd.DataFrame) and len(X_val) >= 20 and y_val.nunique() >= 2:
                try:
                    raw_scores_val = model.predict_raw_score(X_val)
                    prob_raw = np.column_stack([1.0 - raw_scores_val, raw_scores_val])
                    calib_raw = calculate_calibration_metrics(y_val.values, prob_raw)
                    val_brier_raw = calib_raw.get('brier_score')
                    val_ece_raw = calib_raw.get('expected_calibration_error')

                    # Fit calibrator and re-evaluate
                    model.fit_calibrator(X_val, y_val)
                    cal_scores_val = model.predict_score(X_val)
                    prob_cal = np.column_stack([1.0 - cal_scores_val, cal_scores_val])
                    calib_cal = calculate_calibration_metrics(y_val.values, prob_cal)
                    val_brier_cal = calib_cal.get('brier_score')
                    val_ece_cal = calib_cal.get('expected_calibration_error')

                    if val_brier_raw is not None and val_brier_cal is not None:
                        raw_ece_str = f"{val_ece_raw:.4f}" if val_ece_raw is not None else "nan"
                        cal_ece_str = f"{val_ece_cal:.4f}" if val_ece_cal is not None else "nan"
                        tprint(
                            f"[GateTrainingStep] Validation calibration - Brier raw={val_brier_raw:.4f}, cal={val_brier_cal:.4f}; "
                            f"ECE raw={raw_ece_str}, cal={cal_ece_str}",
                            "INFO",
                        )
                except Exception as e:
                    tprint_warning(f"[GateTrainingStep] Calibration step failed: {e}")

            # Calibrate Threshold (default 25th percentile -> block bottom 25%) using training window
            model.calibrate_threshold(X_train_fit, percentile=25)

            # 4. Evaluate (Regression metrics + Strategy lift) on full candidate universe
            scores = model.predict_score(X_train)
            preds_bin = model.predict(X_train) # 1 = Trade, 0 = Block

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
                # Regression Metrics
                'rmse': float(np.sqrt(mean_squared_error(y_train, scores))),
                'r2': float(r2_score(y_train, scores)),
                # Strategy Metrics
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
                # Calibration metrics on validation window
                'brier_val_raw': float(val_brier_raw) if val_brier_raw is not None else None,
                'brier_val_cal': float(val_brier_cal) if val_brier_cal is not None else None,
                'ece_val_raw': float(val_ece_raw) if val_ece_raw is not None else None,
                'ece_val_cal': float(val_ece_cal) if val_ece_cal is not None else None,
            }

            # Volatility-regime stability metrics (p_success vs realized winrate across rv_short buckets)
            volatility_stability = {}
            try:
                if 'rv_short' in X_train.columns:
                    rv_all = X_train['rv_short'].astype(float).replace([np.inf, -np.inf], np.nan)
                    rv_clean = rv_all.dropna()
                    if len(rv_clean) >= 50:
                        low_q = float(rv_clean.quantile(1.0 / 3.0))
                        high_q = float(rv_clean.quantile(2.0 / 3.0))

                        def _collect_stability(mask: np.ndarray, period_name: str) -> None:
                            period_indices = X_train.index[mask]
                            if len(period_indices) == 0:
                                return

                            X_p = X_train.loc[period_indices]
                            rv_p = X_p['rv_short'].astype(float)
                            scores_p = pd.Series(model.predict_score(X_p), index=period_indices)
                            returns_p = baseline_returns.reindex(period_indices)

                            period_stats: Dict[str, Dict[str, float]] = {}
                            for bucket_name, bucket_mask in [
                                ('low', rv_p <= low_q),
                                ('mid', (rv_p > low_q) & (rv_p < high_q)),
                                ('high', rv_p >= high_q),
                            ]:
                                idx_bucket = rv_p.index[bucket_mask]
                                if len(idx_bucket) == 0:
                                    continue
                                bucket_scores = scores_p.reindex(idx_bucket).dropna()
                                bucket_returns = returns_p.reindex(idx_bucket).dropna()
                                if len(bucket_scores) == 0 or len(bucket_returns) == 0:
                                    continue
                                period_stats[bucket_name] = {
                                    'count': int(len(idx_bucket)),
                                    'mean_score': float(bucket_scores.mean()),
                                    'realized_win_rate': float((bucket_returns > 0).mean()),
                                }

                            if period_stats:
                                volatility_stability[period_name] = period_stats

                        _collect_stability(train_mask, 'train')
                        _collect_stability(val_mask, 'validation')
                        _collect_stability(test_mask, 'test')
            except Exception:
                volatility_stability = {}

            if volatility_stability:
                metrics['volatility_stability'] = volatility_stability

            shap_df = None
            shap_importance = getattr(model, 'shap_feature_importance_', None)
            shap_names = getattr(model, 'shap_feature_names_', None)
            if shap_importance is not None and shap_names is not None:
                try:
                    shap_df = pd.DataFrame(
                        {
                            'feature': list(shap_names),
                            'mean_abs_shap': [float(v) for v in shap_importance],
                        }
                    )
                    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
                    metrics['shap_feature_importance'] = dict(
                        zip(shap_df['feature'], shap_df['mean_abs_shap'])
                    )
                except Exception:
                    shap_df = None

            tprint_success(
                f"Gate Training Metrics: RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.4f}, "
                f"AcceptRate={metrics['acceptance_rate']:.2%}, "
                f"PreWin={metrics['pre_win_rate']:.2%}, PostWin={metrics['post_win_rate']:.2%}, "
                f"AvgLift={metrics['avg_return_lift']:.5f}"
            )

            # 5. Save Artifacts
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save Model (globally trained gate with calibration / metrics above)
            model_filename = f"gate_model_{symbol}_{timeframe}_{timestamp_str}.joblib"
            temp_path = os.path.join("artifacts", model_filename)
            os.makedirs("artifacts", exist_ok=True)
            model.save(temp_path)

            # Walk-forward OOF predictions only (expanding window with periodic retraining)
            burnin_days = int(gate_config.get('burnin_days', 60))
            retrain_days = int(gate_config.get('retrain_interval_days', 14))

            idx_dt = pd.to_datetime(X_train.index)
            if len(idx_dt) == 0:
                oof_df = pd.DataFrame(columns=['gate_score', 'gate_decision', 'target'])
            else:
                idx_sorted = idx_dt.sort_values()
                first_ts = idx_sorted.min()
                last_ts = idx_sorted.max()

                burnin_end = first_ts + pd.Timedelta(days=burnin_days)

                # Require at least min_samples for initial training and some horizon beyond burn-in
                if burnin_end >= last_ts or (idx_dt < burnin_end).sum() < min_samples:
                    tprint_warning(
                        f"[GateTrainingStep] Insufficient horizon for burn-in ({burnin_days}d) and walk-forward OOF; "
                        f"skipping OOF generation and saving empty gate_oof_predictions."
                    )
                    oof_df = pd.DataFrame(columns=['gate_score', 'gate_decision', 'target'])
                else:
                    oof_scores = pd.Series(index=X_train.index, dtype=float)
                    oof_decisions = pd.Series(index=X_train.index, dtype=float)

                    current_start = burnin_end
                    while current_start < last_ts:
                        window_end = min(current_start + pd.Timedelta(days=retrain_days), last_ts)

                        train_mask_wf = idx_dt < current_start
                        pred_mask_wf = (idx_dt >= current_start) & (idx_dt <= window_end)

                        n_train_wf = int(train_mask_wf.sum())
                        n_pred_wf = int(pred_mask_wf.sum())
                        if n_train_wf < min_samples or n_pred_wf == 0:
                            current_start = window_end
                            continue

                        X_train_wf = X_train[train_mask_wf]
                        y_train_wf = y_train[train_mask_wf]

                        # Train a fresh GateModel on all past data up to the segment start
                        wf_model = GateModel(config=gate_config)
                        wf_model.train(X_train_wf, y_train_wf)
                        wf_model.calibrate_threshold(X_train_wf, percentile=25)

                        X_pred_wf = X_train[pred_mask_wf]
                        scores_wf = wf_model.predict_score(X_pred_wf)
                        decisions_wf = wf_model.predict(X_pred_wf)

                        oof_scores.loc[X_pred_wf.index] = scores_wf
                        oof_decisions.loc[X_pred_wf.index] = decisions_wf

                        current_start = window_end

                    oof_scores = oof_scores.dropna()
                    oof_decisions = oof_decisions.dropna()

                    common_oof_index = (
                        oof_scores.index
                        .intersection(oof_decisions.index)
                        .intersection(y_train.index)
                    )

                    if len(common_oof_index) == 0:
                        oof_df = pd.DataFrame(columns=['gate_score', 'gate_decision', 'target'])
                    else:
                        oof_df = pd.DataFrame(
                            {
                                'gate_score': oof_scores.loc[common_oof_index],
                                'gate_decision': oof_decisions.loc[common_oof_index].astype(int),
                                'target': y_train.loc[common_oof_index],
                            },
                            index=common_oof_index,
                        )

            oof_path = self._save_artifact(oof_df, f"gate_oof_predictions_{symbol}", "data")

            metrics_path = self._save_artifact(metrics, f"gate_metrics_{symbol}", "metadata")

            shap_path = None
            if shap_df is not None:
                shap_path = self._save_artifact(shap_df, f"gate_shap_importance_{symbol}", "data")

            report_path = None
            try:
                timestamp_report = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = f"gate_training_report_{symbol}_{timeframe}_{direction}_{timestamp_report}"
                outcomes_dir = "outcomes"
                os.makedirs(outcomes_dir, exist_ok=True)
                md_path = os.path.join(outcomes_dir, f"{base_name}.md")
                lines = []
                lines.append("# Gate Training Report\n\n")
                lines.append(f"**Symbol**: {symbol}\n")
                lines.append(f"**Exchange**: {exchange}\n")
                lines.append(f"**Timeframe**: {timeframe}\n")
                lines.append(f"**Direction**: {direction}\n")
                lines.append(f"**Timestamp**: {datetime.now().isoformat()}\n\n")
                lines.append("## Performance and Financial Metrics\n\n")
                for key in [
                    "candidate_count",
                    "accepted_count",
                    "acceptance_rate",
                    "coverage_rate",
                    "pre_trade_count",
                    "pre_win_rate",
                    "pre_avg_return",
                    "pre_median_return",
                    "pre_total_return",
                    "post_trade_count",
                    "post_win_rate",
                    "post_avg_return",
                    "post_median_return",
                    "post_total_return",
                    "avg_return_lift",
                    "rmse",
                    "r2",
                    "brier_val_raw",
                    "brier_val_cal",
                    "ece_val_raw",
                    "ece_val_cal",
                ]:
                    if key in metrics:
                        value = metrics.get(key)
                        if isinstance(value, float):
                            lines.append(f"- **{key}**: {value:.6f}\n")
                        else:
                            lines.append(f"- **{key}**: {value}\n")
                if shap_df is not None and not shap_df.empty:
                    lines.append("\n## SHAP Feature Importance (Top 30)\n\n")
                    lines.append("| Rank | Feature | Mean |SHAP| |\n")
                    lines.append("|------|---------|-------------|\n")
                    for idx_row, row in enumerate(shap_df.itertuples(index=False), start=1):
                        if idx_row > 30:
                            break
                        lines.append(f"| {idx_row} | {row.feature} | {row.mean_abs_shap:.6f} |\n")
                lines.append("\n## Artifacts\n\n")
                lines.append(f"- **Gate model**: `{temp_path}`\n")
                lines.append(f"- **Gate metrics**: `{metrics_path}`\n")
                lines.append(f"- **Gate OOF predictions**: `{oof_path}`\n")
                if shap_path is not None:
                    lines.append(f"- **Gate SHAP importance**: `{shap_path}`\n")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                report_path = md_path
            except Exception:
                report_path = None

            return {
                'success': True,
                'artifacts': {
                    'gate_model': temp_path,
                    'gate_oof_predictions': oof_path,
                    'gate_metrics': metrics_path,
                    'gate_shap_importance': shap_path
                },
                'metrics': metrics,
                'outcome_report_path': report_path
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
