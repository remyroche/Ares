# src/training/steps/step2_processing_labeling_feature_engineering.py

import asyncio
import os
import json
from typing import Any

import pandas as pd

from src.utils.logger import system_logger as _logger
from src.training.steps.unified_data_loader import get_unified_data_loader
from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
    OptimizedTripleBarrierLabeling,
)
from src.training.steps.vectorized_labelling_orchestrator import (
    VectorizedLabellingOrchestrator,
)


async def run_step(
    symbol: str,
    exchange_name: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    exchange: str = "BINANCE",
    force_rerun: bool = False,
    pipeline_config: dict[str, Any] | None = None,
) -> bool:
    _logger.info(
        "🚀 Running Step 2: Processing, labeling, meta-labeling & feature engineering...",
    )

    actual_exchange = exchange if exchange != "BINANCE" else exchange_name

    try:
        # 1) Load unified OHLCV data
        config: dict[str, Any] = {
            "symbol": symbol,
            "exchange": actual_exchange,
            "data_dir": data_dir,
            "timeframe": timeframe,
        }
        if pipeline_config:
            config.update({"vectorized_labelling_orchestrator": pipeline_config.get("vectorized_labelling_orchestrator", {})})

        data_loader = get_unified_data_loader(config)
        lookback_days = config.get("lookback_days", 180)
        df = await data_loader.load_unified_data(
            symbol=symbol,
            exchange=actual_exchange,
            timeframe=timeframe,
            lookback_days=lookback_days,
            use_streaming=True,
        )
        if df is None or df.empty:
            raise ValueError(f"No data found for {symbol} on {actual_exchange}")

        # Ensure timestamp column exists and is datetime
        if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])  # best-effort cast
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 2) Compute triple-barrier labels (binary) while preserving OHLCV
        lbl = OptimizedTripleBarrierLabeling(binary_classification=True)
        labeled = lbl.apply_triple_barrier_labeling_vectorized(
            df[[c for c in ["open", "high", "low", "close", "volume", "timestamp"] if c in df.columns]].set_index("timestamp")
        )
        labeled = labeled.reset_index()  # bring timestamp back as column

        # 3) Split into train/validation/test by time (70/15/15)
        n = len(labeled)
        if n < 100:
            _logger.warning("⚠️ Very little data for step 2; proceeding with minimal splits")
        cut1 = int(n * 0.70)
        cut2 = int(n * 0.85)
        labeled_train = labeled.iloc[:cut1].copy()
        labeled_val = labeled.iloc[cut1:cut2].copy()
        labeled_test = labeled.iloc[cut2:].copy()

        # 4) Persist labeled parquet artifacts expected by later steps
        os.makedirs(data_dir, exist_ok=True)
        paths = {
            "train": f"{data_dir}/{actual_exchange}_{symbol}_labeled_train.parquet",
            "validation": f"{data_dir}/{actual_exchange}_{symbol}_labeled_validation.parquet",
            "test": f"{data_dir}/{actual_exchange}_{symbol}_labeled_test.parquet",
        }
        labeled_train.to_parquet(paths["train"], index=False)
        labeled_val.to_parquet(paths["validation"], index=False)
        labeled_test.to_parquet(paths["test"], index=False)
        _logger.info(
            f"✅ Wrote labeled splits: train={len(labeled_train)} val={len(labeled_val)} test={len(labeled_test)}",
        )

        # 5) Run vectorized orchestrator to derive feature space + meta strengths, and persist strengths snapshot
        try:
            orchestrator = VectorizedLabellingOrchestrator(config)
            ok = await orchestrator.initialize()
            if ok:
                # Prepare price/volume inputs for orchestrator
                price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                price_data = df[["timestamp"] + price_cols].set_index("timestamp")
                volume_data = price_data[["volume"]] if "volume" in price_data.columns else pd.DataFrame(index=price_data.index)
                result = await orchestrator.orchestrate_labeling_and_feature_engineering(price_data, volume_data)
                final_df: pd.DataFrame | None = None
                if isinstance(result, dict) and isinstance(result.get("data"), pd.DataFrame):
                    final_df = result["data"]
                # Persist meta strengths if available (columns starting with 'sr_')
                if final_df is not None and not final_df.empty:
                    strength_cols = [c for c in final_df.columns if c.lower().startswith("sr_")]
                    if strength_cols:
                        strengths = final_df[strength_cols].copy()
                        strengths["timestamp"] = strengths.index
                        strengths = strengths.reset_index(drop=True)
                        strengths_path = f"{data_dir}/{actual_exchange}_{symbol}_meta_strengths.parquet"
                        strengths.to_parquet(strengths_path, index=False)
                        _logger.info(
                            f"✅ Saved meta strengths snapshot with {len(strength_cols)} cols to {strengths_path}",
                        )
        except Exception as e:
            _logger.warning(f"Meta strengths persistence skipped: {e}")

        # 6) Persist label distribution per split for diagnostics
        try:
            dist = {
                "train": labeled_train.get("label", pd.Series(dtype=int)).value_counts(dropna=False).to_dict(),
                "validation": labeled_val.get("label", pd.Series(dtype=int)).value_counts(dropna=False).to_dict(),
                "test": labeled_test.get("label", pd.Series(dtype=int)).value_counts(dropna=False).to_dict(),
            }
            with open(
                f"{data_dir}/{actual_exchange}_{symbol}_label_distribution.json",
                "w",
            ) as f:
                json.dump(dist, f, indent=2)
        except Exception as e:
            _logger.warning(f"Label distribution persistence skipped: {e}")

        # 7) Persist label reliability (if available) for downstream gating/stacking
        try:
            from src.training.enhanced_training_manager import EnhancedTrainingManager

            etm = EnhancedTrainingManager(config)
            reliability = etm.get_label_reliability()
            with open(f"{data_dir}/{actual_exchange}_{symbol}_label_reliability.json", "w") as f:
                json.dump(reliability, f, indent=2)
        except Exception:
            pass

        return True

    except Exception as e:
        _logger.exception(f"Step 2 processing/labeling/FE failed: {e}")
        return False


if __name__ == "__main__":
    async def _test():
        ok = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Step 2 test result: {ok}")

    asyncio.run(_test())