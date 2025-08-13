# src/training/steps/step2_processing_labeling_feature_engineering.py

import asyncio
from typing import Any
import os
import pandas as pd

from src.utils.logger import system_logger as _logger
from src.training.steps.unified_data_loader import get_unified_data_loader
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
)
from src.training.enhanced_training_manager_optimized import (
    MemoryEfficientDataManager,
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
    _logger.info("🚀 Running Step 2: Processing, labeling, meta-labeling & feature engineering...")

    actual_exchange = exchange if exchange != "BINANCE" else exchange_name

    try:
        # 1) Load unified OHLCV data
        cfg = pipeline_config or {}
        data_loader = get_unified_data_loader(cfg)
        lookback_days = cfg.get("lookback_days", 180)
        ohlcv = await data_loader.load_unified_data(
            symbol=symbol,
            exchange=actual_exchange,
            timeframe=timeframe,
            lookback_days=lookback_days,
            use_streaming=True,
        )
        if ohlcv is None or ohlcv.empty:
            _logger.error("❌ Step 2: No unified OHLCV data available")
            return False

        # 2) Minimal labeling: create a simple classification target if absent
        df = ohlcv.copy()
        if "close" not in df.columns:
            _logger.error("❌ Step 2: 'close' column missing in OHLCV data")
            return False
        df["return_1"] = df["close"].pct_change().fillna(0.0)
        df["label"] = (df["return_1"] > 0).astype(int)

        # 3) Split into train/validation/test by chronological order (70/15/15)
        n = len(df)
        if n < 100:
            _logger.error("❌ Step 2: Not enough data for splits")
            return False
        tr_end = int(n * 0.7)
        vl_end = int(n * 0.85)
        tr = df.iloc[:tr_end].copy()
        vl = df.iloc[tr_end:vl_end].copy()
        te = df.iloc[vl_end:].copy()

        # 4) Engineer features using vectorized advanced FE
        fe = VectorizedAdvancedFeatureEngineering(cfg.get("vectorized_labelling_orchestrator", {}))
        await fe.initialize()

        def _extract_inputs(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in x.columns]
            price = x[price_cols].copy()
            vol = price[["volume"]].copy() if "volume" in price.columns else pd.DataFrame({"volume": 1.0}, index=price.index)
            return price, vol

        p_tr, v_tr = _extract_inputs(tr)
        p_vl, v_vl = _extract_inputs(vl)
        p_te, v_te = _extract_inputs(te)

        X_tr = pd.DataFrame(await fe.engineer_features(p_tr, v_tr)).reindex(p_tr.index)
        X_vl = pd.DataFrame(await fe.engineer_features(p_vl, v_vl)).reindex(p_vl.index)
        X_te = pd.DataFrame(await fe.engineer_features(p_te, v_te)).reindex(p_te.index)

        # 5) Persist labeled splits for Step 3 compatibility
        os.makedirs(data_dir, exist_ok=True)
        mem = MemoryEfficientDataManager()

        def _save(name: str, base: pd.DataFrame, feats: pd.DataFrame):
            out = base.join(feats, how="left")
            path = f"{data_dir}/{actual_exchange}_{symbol}_labeled_{name}.parquet"
            mem.save_to_parquet(mem.optimize_dataframe(out), path)
            _logger.info(f"✅ Step 2: wrote {name} labeled parquet -> {path}")

        _save("train", tr, X_tr)
        _save("validation", vl, X_vl)
        _save("test", te, X_te)

        _logger.info("✅ Step 2 completed successfully (labels + features saved)")
        return True

    except Exception as e:
        _logger.exception(f"Step 2 processing/labeling/FE failed: {e}")
        return False


if __name__ == "__main__":
    async def _test():
        ok = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Step 2 test result: {ok}")

    asyncio.run(_test())