# src/training/steps/step3_feature_engineering.py

import asyncio
from typing import Any
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
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
    Loads labeled parquet from Step 2 and produces robust feature parquet artifacts for train/val/test.
    """
    logger = system_logger.getChild("Step3.FeatureEngineering")
    try:
        from src.training.steps.vectorized_advanced_feature_engineering import (
            VectorizedAdvancedFeatureEngineering,
        )
        from src.training.enhanced_training_manager_optimized import (
            MemoryEfficientDataManager,
        )

        # 1) Load labeled splits produced by Step 2
        paths = {
            "train": f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
            "validation": f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
            "test": f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet",
        }
        labeled = {name: pd.read_parquet(path) for name, path in paths.items()}
        for split, df in labeled.items():
            logger.info(f"Loaded labeled {split}: {len(df)} rows")

        # 2) Extract OHLCV inputs
        def _extract_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if len(price_cols) < 4:  # expect at least open/high/low/close
                raise ValueError("Missing OHLC columns in labeled data")
            price = df[price_cols].copy()
            vol = price[["volume"]].copy() if "volume" in price.columns else pd.DataFrame({"volume": 1.0}, index=price.index)
            return price, vol

        price_tr, vol_tr = _extract_inputs(labeled["train"]) 
        price_vl, vol_vl = _extract_inputs(labeled["validation"]) 
        price_te, vol_te = _extract_inputs(labeled["test"]) 

        # 3) Initialize FE engine
        fe = VectorizedAdvancedFeatureEngineering({})
        await fe.initialize()

        # 4) Engineer features per split
        feats_tr = await fe.engineer_features(price_tr, vol_tr)
        feats_vl = await fe.engineer_features(price_vl, vol_vl)
        feats_te = await fe.engineer_features(price_te, vol_te)

        X_tr = pd.DataFrame(feats_tr).reindex(price_tr.index)
        X_vl = pd.DataFrame(feats_vl).reindex(price_vl.index)
        X_te = pd.DataFrame(feats_te).reindex(price_te.index)

        # 5) Basic sanitization: drop constant columns, handle inf/nan
        def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
            df = df.replace([np.inf, -np.inf], np.nan)
            nunique = df.nunique(dropna=True)
            low_var_cols = nunique[nunique <= 1].index.tolist()
            if low_var_cols:
                logger.info(f"Dropping {len(low_var_cols)} constant features")
                df = df.drop(columns=low_var_cols, errors="ignore")
            return df.fillna(0)

        X_tr = _sanitize(X_tr)
        X_vl = _sanitize(X_vl)
        X_te = _sanitize(X_te)

        # 6) Correlation pruning (|rho| >= 0.95) on train; apply to val/test
        def _corr_prune(train_df: pd.DataFrame, thr: float = 0.95) -> list[str]:
            if train_df.empty:
                return []
            corr = train_df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            drop_cols = [c for c in upper.columns if any(upper[c] >= thr)]
            return drop_cols

        drop_tr = _corr_prune(X_tr)
        if drop_tr:
            logger.info(f"Correlation prune: dropping {len(drop_tr)} features (|rho|>=0.95)")
            X_tr = X_tr.drop(columns=drop_tr, errors="ignore")
            X_vl = X_vl.drop(columns=drop_tr, errors="ignore")
            X_te = X_te.drop(columns=drop_tr, errors="ignore")

        # 7) Mutual information screen placeholder: keep as-is for speed (could integrate as needed)
        # 8) VIF reduction placeholder: kept for future integration to mirror Step 4 behavior

        # 9) Save features and selected feature lists
        os.makedirs(data_dir, exist_ok=True)
        mem_mgr = MemoryEfficientDataManager()

        def _save(name: str, df: pd.DataFrame):
            path = f"{data_dir}/{exchange}_{symbol}_features_{name}.parquet"
            mem_mgr.save_to_parquet(mem_mgr.optimize_dataframe(df.copy()), path)
            logger.info(f"✅ Saved features {name}: {len(df)} rows, {df.shape[1]} cols -> {path}")

        _save("train", X_tr)
        _save("validation", X_vl)
        _save("test", X_te)

        # Save feature lists per split
        feature_lists = {
            "train": list(X_tr.columns),
            "validation": list(X_vl.columns),
            "test": list(X_te.columns),
            "timestamp": datetime.now().isoformat(),
        }
        with open(f"{data_dir}/{exchange}_{symbol}_selected_features.json", "w") as f:
            json.dump(feature_lists, f, indent=2)

        logger.info("✅ Step 3: Feature engineering completed successfully")
        return True
    except Exception as e:
        logger.exception(f"Step 3 feature engineering failed: {e}")
        return False


if __name__ == "__main__":
    async def _test():
        ok = await run_step("ETHUSDT")
        print(f"Step 3 test result: {ok}")
    asyncio.run(_test())