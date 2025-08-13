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
    Also writes pickle copies with timestamps and a feature hash for Step 5 compatibility.
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

        # Ensure timestamp present and set as index for alignment
        for k in labeled.keys():
            if "timestamp" not in labeled[k].columns and isinstance(labeled[k].index, pd.DatetimeIndex):
                labeled[k] = labeled[k].reset_index().rename(columns={"index": "timestamp"})
            if "timestamp" in labeled[k].columns:
                labeled[k]["timestamp"] = pd.to_datetime(labeled[k]["timestamp"], errors="coerce")
                labeled[k] = labeled[k].dropna(subset=["timestamp"]).sort_values("timestamp")
                labeled[k] = labeled[k].set_index("timestamp")

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

        # 7) Mutual information screen (classification target 'label')
        try:
            from sklearn.feature_selection import mutual_info_classif
            y = None
            if "label" in labeled["train"].columns:
                # Use classification labels from Step 2
                y = labeled["train"]["label"].astype(int).values
            if y is not None and len(np.unique(y)) > 1 and not X_tr.empty:
                numX = X_tr.select_dtypes(include=[np.number])
                if not numX.empty:
                    mi = mutual_info_classif(numX.values, y, discrete_features=False, random_state=42)
                    mi_s = pd.Series(mi, index=numX.columns).sort_values(ascending=False)
                    # Persist MI scores
                    os.makedirs("log/mi", exist_ok=True)
                    with open(f"log/mi/{exchange}_{symbol}_step3_mi.json", "w") as f:
                        json.dump({"mi": mi_s.to_dict()}, f, indent=2)
                    # Selection policy: keep top-k if provided; otherwise drop bottom quantile
                    mi_top_k = int(kwargs.get("mi_top_k", 0) or 0)
                    if mi_top_k > 0:
                        keep_cols = list(mi_s.head(mi_top_k).index)
                    else:
                        mi_quantile = float(kwargs.get("mi_quantile", 0.20))
                        thr = mi_s.quantile(mi_quantile)
                        keep_cols = list(mi_s[mi_s >= thr].index)
                    # Apply keep set
                    X_tr = X_tr[keep_cols]
                    X_vl = X_vl[keep_cols]
                    X_te = X_te[keep_cols]
                    logger.info(f"MI kept {len(keep_cols)} features (top_k={mi_top_k} quantile={kwargs.get('mi_quantile', 0.20)})")
        except Exception as e:
            logger.warning(f"MI screening skipped: {e}")

        # 8) VIF reduction (iterative)
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            vif_thr = float(kwargs.get("vif_threshold", 5.0))
            max_iter = int(kwargs.get("max_vif_iterations", 10))
            num_cols = list(X_tr.select_dtypes(include=[np.number]).columns)
            it = 0
            while it < max_iter and len(num_cols) > 1:
                it += 1
                Xn = X_tr[num_cols].astype(float).fillna(0.0)
                # Standardize to stabilize VIF
                std = Xn.std(ddof=0).replace(0, 1.0)
                Xn = (Xn - Xn.mean()) / std
                vif_vals = pd.Series([variance_inflation_factor(Xn.values, i) for i in range(Xn.shape[1])], index=num_cols)
                max_vif = float(vif_vals.max()) if not vif_vals.empty else 0.0
                if max_vif <= vif_thr:
                    break
                drop_col = str(vif_vals.idxmax())
                logger.info(f"VIF prune: dropping {drop_col} (VIF={max_vif:.2f})")
                num_cols.remove(drop_col)
            # Apply final VIF-selected set
            if num_cols:
                X_tr = X_tr[num_cols]
                X_vl = X_vl[num_cols]
                X_te = X_te[num_cols]
                logger.info(f"VIF kept {len(num_cols)} features (threshold={vif_thr})")
        except Exception as e:
            logger.warning(f"VIF reduction skipped: {e}")

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

        # Save feature lists per split and a feature hash
        feature_lists = {
            "train": list(X_tr.columns),
            "validation": list(X_vl.columns),
            "test": list(X_te.columns),
            "timestamp": datetime.now().isoformat(),
        }
        with open(f"{data_dir}/{exchange}_{symbol}_selected_features.json", "w") as f:
            json.dump(feature_lists, f, indent=2)

        # NEW: also persist pickle copies with timestamps for Step 5 compatibility
        try:
            import pickle
            for split_name, X in ("train", X_tr), ("validation", X_vl), ("test", X_te):
                X_pick = X.copy()
                X_pick["timestamp"] = X_pick.index
                X_pick = X_pick.reset_index(drop=True)
                pkl_path = f"{data_dir}/{exchange}_{symbol}_features_{split_name}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(X_pick, f)
                logger.info(f"✅ Wrote pickle features {split_name}: {pkl_path} rows={len(X_pick)} cols={X_pick.shape[1]}")

            # Write a simple feature hash to ensure downstream consistency
            import hashlib
            def _hash_cols(cols: list[str]) -> str:
                s = ",".join(cols)
                return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
            hash_info = {
                "train_hash": _hash_cols(feature_lists["train"]),
                "validation_hash": _hash_cols(feature_lists["validation"]),
                "test_hash": _hash_cols(feature_lists["test"]),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(f"{data_dir}/{exchange}_{symbol}_feature_hash.json", "w") as f:
                json.dump(hash_info, f, indent=2)
        except Exception as e:
            logger.warning(f"Pickle compatibility write skipped: {e}")

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