# src/training/steps/step3_feature_engineering.py

import asyncio
from typing import Any
import pandas as pd
from src.utils.logger import system_logger


async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    **kwargs: Any,
) -> bool:
    """
    Step 3: Engineering the features (post-labeling).
    Loads labeled parquet from Step 2 and produces feature parquet artifacts.
    """
    logger = system_logger.getChild("Step3.FeatureEngineering")
    try:
        from src.training.steps.vectorized_advanced_feature_engineering import (
            VectorizedAdvancedFeatureEngineering,
        )
        # Load labeled data from Step 2 (train split used for schema; features saved per-split)
        labeled_train = pd.read_parquet(f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet")
        labeled_val = pd.read_parquet(f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet")
        labeled_test = pd.read_parquet(f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet")

        # Separate price/volume inputs; index-aligned
        def _extract_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            price = df[[c for c in ["open","high","low","close","volume"] if c in df.columns]].copy()
            if price.empty:
                raise ValueError("Missing OHLCV columns in labeled data")
            vol = price[["volume"]].copy() if "volume" in price.columns else pd.DataFrame({"volume": 1.0}, index=price.index)
            return price, vol

        price_tr, vol_tr = _extract_inputs(labeled_train)
        price_vl, vol_vl = _extract_inputs(labeled_val)
        price_te, vol_te = _extract_inputs(labeled_test)

        fe = VectorizedAdvancedFeatureEngineering({})
        await fe.initialize()

        feats_tr = await fe.engineer_features(price_tr, vol_tr)
        feats_vl = await fe.engineer_features(price_vl, vol_vl)
        feats_te = await fe.engineer_features(price_te, vol_te)

        # Convert to DataFrame and align indices
        def _to_df(d: dict[str, Any], idx) -> pd.DataFrame:
            df = pd.DataFrame(d)
            return df.set_index(idx) if "timestamp" in df.columns else df.reindex(idx)

        X_tr = pd.DataFrame(feats_tr).reindex(price_tr.index)
        X_vl = pd.DataFrame(feats_vl).reindex(price_vl.index)
        X_te = pd.DataFrame(feats_te).reindex(price_te.index)

        # Save features per split
        X_tr.to_parquet(f"{data_dir}/{exchange}_{symbol}_features_train.parquet")
        X_vl.to_parquet(f"{data_dir}/{exchange}_{symbol}_features_validation.parquet")
        X_te.to_parquet(f"{data_dir}/{exchange}_{symbol}_features_test.parquet")
        logger.info("✅ Step 3 feature engineering saved parquet files")
        return True
    except Exception as e:
        logger.exception(f"Step 3 feature engineering failed: {e}")
        return False


if __name__ == "__main__":
    async def _test():
        ok = await run_step("ETHUSDT")
        print(f"Step 3 test result: {ok}")
    asyncio.run(_test())