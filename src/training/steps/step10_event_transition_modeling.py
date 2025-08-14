# src/training/steps/step10_event_transition_modeling.py

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import json

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.transition.event_trigger_indexer import EventTriggerIndexer
from src.transition.event_window_dataset import EventWindowDatasetBuilder
from src.transition.path_targets import PathTargetEngineer
from src.transition.baseline_rf import TransitionRandomForest
from src.transition.rolling_window_dataset import RollingWindowDatasetBuilder
from src.transition.multitask_rf import MultiTaskRandomForest


logger = system_logger.getChild("Step10EventTransitionModeling")


@handle_errors(exceptions=(Exception,), default_return=False, context="step10_event_transition_modeling")
async def run_step(
    symbol: str,
    exchange: str = "UNKNOWN",
    timeframe: str = "1m",
    training_config: dict[str, Any] | None = None,
    force_rerun: bool = False,
) -> bool:
    cfg = training_config or {}
    tm_cfg = cfg.get("TRANSITION_MODELING", {})
    if not tm_cfg.get("enabled", False):
        logger.info("Transition modeling disabled; skipping step 10.")
        return True

    # Load previously saved vectorized combined features parquet or rely on in-memory pipeline products
    artifacts_dir = str(tm_cfg.get("artifacts_dir", "checkpoints/transition_datasets"))
    os.makedirs(artifacts_dir, exist_ok=True)

    # Expect upstream pipeline to have produced a combined dataset; attempt to load latest
    # Fallback: user should pass necessary frames into this step in a real orchestration.
    # Here we look for the latest vectorized_features_*.parquet if present
    try:
        data_dir = os.path.join("data", "vectorized")
        latest = None
        if os.path.isdir(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
            if files:
                latest = max(files)
        combined_df = pd.read_parquet(os.path.join(data_dir, latest)) if latest else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to load combined features parquet: {e}", exc_info=True)
        combined_df = pd.DataFrame()

    if combined_df.empty:
        logger.warning("No combined features parquet found; step expects upstream to supply data.")
        return True

    # For states, we need klines (OHLCV) aligned to combined_df index
    # If raw OHLCV were removed in earlier stages, try to reconstruct minimal close from context
    if not all(c in combined_df.columns for c in ["open", "high", "low", "close", "volume"]):
        logger.warning("OHLCV columns missing in combined_df; attempting minimal reconstruction from context.")
        # Attempt to grab 'avg_price' as proxy
        if "avg_price" in combined_df.columns:
            klines_df = pd.DataFrame(index=combined_df.index)
            klines_df["open"] = combined_df["avg_price"]
            klines_df["high"] = combined_df["avg_price"]
            klines_df["low"] = combined_df["avg_price"]
            klines_df["close"] = combined_df["avg_price"]
            klines_df["volume"] = combined_df.get("trade_volume", 0)
        else:
            logger.error("Cannot proceed without OHLCV or avg_price context; aborting.")
            return False
    else:
        klines_df = combined_df[["open","high","low","close","volume"]].copy()

    # Branch: rolling triggerless vs event-triggered
    if bool(tm_cfg.get("rolling_mode", False)):
        try:
            logger.info("Running Step 10 in rolling (triggerless) mode: building samples at every bar")
            rbuilder = RollingWindowDatasetBuilder(cfg, exchange=exchange, symbol=symbol)
            await rbuilder.initialize()
            rds = rbuilder.build(klines_df, combined_df)
            samples = rds.get("samples", [])
            if not samples:
                logger.info("No samples constructed in rolling mode.")
                return True
            horizons = list((tm_cfg.get("rolling", {}) or {}).get("direction_horizons", [5, 15]))
            mtrf = MultiTaskRandomForest(cfg, horizons=horizons)
            result = mtrf.fit(samples)
            # Save a compact report
            rep_path = os.path.join(artifacts_dir, f"{symbol}_{timeframe}_rolling_mtrf_report.json")
            with open(rep_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info({"msg": "Rolling multitask RF trained", "report_path": rep_path})
            return True
        except Exception as e:
            logger.warning(f"Rolling mode failed; falling back to event-triggered pipeline: {e}")

    # 1) Event index
    indexer = EventTriggerIndexer(cfg)
    candidate_labels = None  # auto-detect intensity_* columns
    event_index = indexer.build_event_index(
        combined_df,
        price_data=klines_df,
        volume_data=klines_df[["volume"]] if "volume" in klines_df.columns else None,
        candidate_labels=candidate_labels,
        timeframe=timeframe,
        instrument_id=f"{exchange}:{symbol}",
    )
    if event_index.empty:
        logger.info("No events found for transition modeling after thresholding.")
        return True

    # 2) Build windows and sequences
    ds_builder = EventWindowDatasetBuilder(cfg, exchange=exchange, symbol=symbol)
    await ds_builder.initialize()
    dataset = ds_builder.build(klines_df, combined_df, event_index)
    samples = dataset.get("samples", [])
    if not samples:
        logger.info("No valid samples after windowing/pruning.")
        return True

    # 3) Targets: path-class
    target_eng = PathTargetEngineer(cfg)
    for s in samples:
        s["path_class"] = target_eng.compute_path_class(s, klines_df)

    # 4) Baseline RF
    rf = TransitionRandomForest(cfg)
    rf_result = rf.fit(samples, dataset.get("label_index", []))
    logger.info({"msg": "RF baseline trained", "report": rf_result.get("report", {})})
    if "shap_top_features" in rf_result:
        logger.info({"msg": "Top SHAP features", "features": rf_result["shap_top_features"]})

    # 4a) Reliability calibration (reuse existing infrastructure): bin predicted prob vs empirical outcomes
    try:
        if rf_result.get("val_true") and rf_result.get("val_proba"):
            from src.training.calibration_manager import CalibrationManager
            cm = CalibrationManager(cfg)
            await cm.initialize()
            # Build a minimal structure expected by our combiner: per timeframe/class scalers
            # For now, compute a simple scaling factor per class by comparing mean predicted vs observed frequency
            val_true = rf_result["val_true"]
            val_proba = rf_result["val_proba"]  # {class: [p,...]}
            classes = list(rf_result.get("classes", []))
            import numpy as _np
            scalers: dict[str, dict[str, float]] = {"1m": {}}
            for c in classes:
                preds = _np.array(val_proba.get(str(c), []), dtype=float)
                obs = _np.array([1.0 if t == str(c) else 0.0 for t in val_true], dtype=float)
                if preds.size and obs.size:
                    mean_p = float(_np.clip(_np.mean(preds), 1e-6, 1.0))
                    mean_y = float(_np.mean(obs))
                    scale = float(_np.clip(mean_y / mean_p, 0.5, 1.5))
                    scalers["1m"][str(c)] = scale
            # Persist to seq2seq artifact dir for combiner
            seq_cfg = (cfg.get("TRANSITION_MODELING", {})).get("seq2seq", {})
            artifact_dir = str(seq_cfg.get("artifact_dir_models", "checkpoints/transition_models"))
            os.makedirs(artifact_dir, exist_ok=True)
            with open(os.path.join(artifact_dir, "reliability.json"), "w") as f:
                import json as _json
                f.write(_json.dumps(scalers, indent=2))
            logger.info({"msg": "Saved reliability scalers", "path": os.path.join(artifact_dir, "reliability.json")})
    except Exception as e:
        logger.warning(f"Reliability calibration skipped due to error: {e}")

    # 4b) Optional Seq2Seq training (compact Transformer) for temporal modeling
    try:
        seq_cfg = (cfg.get("TRANSITION_MODELING", {})).get("seq2seq", {})
        if bool(seq_cfg.get("enabled", False)):
            from src.transition.seq2seq_trainer import train_seq2seq
            post_window = int(cfg.get("TRANSITION_MODELING", {}).get("post_window", 20))
            hmm_vocab = int(cfg.get("TRANSITION_MODELING", {}).get("hmm_n_states", 5))
            d_model = int(seq_cfg.get("d_model", 128))
            nhead = int(seq_cfg.get("nhead", 4))
            num_layers = int(seq_cfg.get("num_layers", 2))
            max_epochs = int(seq_cfg.get("max_epochs", 15))
            lr = float(seq_cfg.get("lr", 1e-3))
            precision = str(seq_cfg.get("precision", "32"))
            path_class_weights = seq_cfg.get("path_class_weights", {})
            focal_gamma = float(seq_cfg.get("focal_gamma", 0.0))
            artifact_dir_models = str(seq_cfg.get("artifact_dir_models", "checkpoints/transition_models"))
            cv_folds = int(seq_cfg.get("cv_folds", 1))
            pt_mult = float(cfg.get("TRANSITION_MODELING", {}).get("barriers", {}).get("profit_take_multiplier", 0.002))
            model_type = str(seq_cfg.get("model_type", "transformer"))
            _ = train_seq2seq(
                samples=samples,
                label_index=dataset.get("label_index", []),
                numeric_feature_names=dataset.get("numeric_feature_names", []),
                post_window=post_window,
                hmm_vocab=hmm_vocab,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                max_epochs=max_epochs,
                lr=lr,
                path_class_weights=path_class_weights,
                focal_gamma=focal_gamma,
                precision=precision,
                artifact_dir_models=artifact_dir_models,
                cv_folds=cv_folds,
                pt_mult=pt_mult,
                model_type=model_type,
            )
    except Exception as e:
        logger.warning(f"Seq2Seq training skipped due to error: {e}")

    # 5) Save compact artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save event index
    try:
        event_path = os.path.join(artifacts_dir, f"{symbol}_{timeframe}_{ts}_event_index.parquet")
        event_index.to_parquet(event_path, index=False)
        logger.info(f"Saved event index: {event_path}")
    except Exception as e:
        logger.warning(f"Failed to save event index: {e}", exc_info=True)
    # Save RF report
    try:
        rep_path = os.path.join(artifacts_dir, f"{symbol}_{timeframe}_{ts}_rf_report.json")
        gating = cfg.get("TRANSITION_MODELING", {}).get("inference", {}).get("path_class_thresholds", {})
        tf_ensemble = cfg.get("TRANSITION_MODELING", {}).get("timeframe_ensemble", {})
        with open(rep_path, "w") as f:
            json.dump({"rf_result": rf_result, "gating_thresholds": gating, "timeframe_ensemble": tf_ensemble}, f, indent=2)
        logger.info(f"Saved RF report: {rep_path}")
    except Exception as e:
        logger.warning(f"Failed to save RF report: {e}", exc_info=True)

    return True