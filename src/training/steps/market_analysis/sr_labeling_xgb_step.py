import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.training.steps.market_analysis.components.level_generators import (
    RollingKDELevelGenerator,
    HTFLevelGenerator,
)
try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    IsotonicRegression = None


logger = logging.getLogger(__name__)


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, Plus DI, Minus DI."""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join="outer").max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1 / period).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di


def _generate_sr_levels(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Generate S/R levels from multiple generators and aggregate.

    Adapted from scripts/sr_level_strength_feature_research.generate_sr_levels.
    """
    idx = ohlcv.index
    base = pd.DataFrame(index=idx)

    # KDE levels
    try:
        kde_gen = RollingKDELevelGenerator()
        kde_levels = kde_gen.compute_levels(ohlcv)

        if not kde_levels.empty and "primary_level_volume_depth_ratio" in kde_levels.columns:
            strength = kde_levels["primary_level_volume_depth_ratio"].astype(float)
            strength = strength.replace([np.inf, -np.inf], np.nan)
            if strength.notna().sum() >= 20:
                try:
                    q25 = float(strength.quantile(0.25))
                    weak_mask = strength < q25
                    cols_to_null = [
                        "primary_level_price",
                        "primary_level_type",
                        "primary_level_source",
                        "primary_level_touch_count",
                        "primary_level_first_touch_ts",
                        "primary_level_last_touch_ts",
                        "primary_level_prominence",
                        "primary_level_volume_depth_ratio",
                    ]
                    existing_cols = [c for c in cols_to_null if c in kde_levels.columns]
                    if existing_cols:
                        kde_levels.loc[weak_mask, existing_cols] = np.nan
                    tprint_info(
                        f"Filtered KDE levels: removed bottom 25% by depth ratio (threshold={q25:.3f})"
                    )
                except Exception as exc:
                    tprint_warning(f"Failed to apply KDE quantile filter: {exc}")

        kde_levels = kde_levels.add_prefix("kde_")
        base = base.join(kde_levels, how="left")
        tprint_info("Added KDE-based S/R levels")
    except Exception as exc:
        tprint_warning(f"Failed to compute KDE levels: {exc}")

    # HTF levels (daily pivots)
    try:
        htf_gen = HTFLevelGenerator(use_weekly=False)
        htf_levels = htf_gen.compute_levels(ohlcv)
        htf_levels = htf_levels.add_prefix("htf_")
        base = base.join(htf_levels, how="left")
        tprint_info("Added HTF-based S/R levels")
    except Exception as exc:
        tprint_warning(f"Failed to compute HTF levels: {exc}")

    close = ohlcv["close"].astype(float)

    def _process_row(row: pd.Series) -> Dict[str, Any]:
        price = float(close.at[row.name]) if row.name in close.index else float("nan")
        if not np.isfinite(price):
            return {
                "primary_level_price": float("nan"),
                "primary_level_type": np.nan,
                "primary_level_source": np.nan,
                "primary_level_touch_count": 0,
                "primary_level_prominence": 0.0,
                "primary_level_volume_depth_ratio": 0.0,
                "primary_level_first_touch_ts": pd.NaT,
                "primary_level_last_touch_ts": pd.NaT,
                "confluence_score": 0,
                "weighted_confluence_score": 0.0,
            }

        candidates = []
        for prefix, src_tag in [("kde_", "kde"), ("htf_", "htf")]:
            p_col = f"{prefix}primary_level_price"
            lp = row.get(p_col, np.nan)
            if np.isfinite(lp):
                cand = {
                    "prefix": prefix,
                    "source": src_tag,
                    "price": float(lp),
                    "type": row.get(f"{prefix}primary_level_type", np.nan),
                    "dist": abs(float(lp) - price),
                    "touch_count": row.get(f"{prefix}primary_level_touch_count", 0),
                    "prominence": row.get(f"{prefix}primary_level_prominence", 0.0),
                    "volume_depth_ratio": row.get(
                        f"{prefix}primary_level_volume_depth_ratio", 0.0
                    ),
                    "first_touch_ts": row.get(
                        f"{prefix}primary_level_first_touch_ts", pd.NaT
                    ),
                    "last_touch_ts": row.get(
                        f"{prefix}primary_level_last_touch_ts", pd.NaT
                    ),
                }
                candidates.append(cand)

        if not candidates:
            return {
                "primary_level_price": float("nan"),
                "primary_level_type": np.nan,
                "primary_level_source": np.nan,
                "primary_level_touch_count": 0,
                "primary_level_prominence": 0.0,
                "primary_level_volume_depth_ratio": 0.0,
                "primary_level_first_touch_ts": pd.NaT,
                "primary_level_last_touch_ts": pd.NaT,
                "confluence_score": 0,
                "weighted_confluence_score": 0.0,
            }

        best = min(candidates, key=lambda x: x["dist"])

        confluence_band = 0.002
        confluence_score = 0
        weighted_confluence_score = 0.0

        for cand in candidates:
            d_pct = abs(cand["price"] - best["price"]) / best["price"]
            if d_pct <= confluence_band:
                confluence_score += 1

                base_weight = 1.0
                multiplier = 1.0

                if cand["source"] == "htf":
                    base_weight = 2.0
                elif cand["source"] == "kde":
                    base_weight = 1.0
                    vol_scale = cand.get("volume_depth_ratio", 1.0)
                    if np.isfinite(vol_scale):
                        multiplier = max(0.5, min(vol_scale, 5.0))

                weighted_confluence_score += base_weight * multiplier

        return {
            "primary_level_price": best["price"],
            "primary_level_type": best["type"],
            "primary_level_source": best["source"],
            "primary_level_touch_count": best.get("touch_count", 0),
            "primary_level_prominence": best.get("prominence", 0.0),
            "primary_level_volume_depth_ratio": best.get("volume_depth_ratio", 0.0),
            "primary_level_first_touch_ts": best.get("first_touch_ts", pd.NaT),
            "primary_level_last_touch_ts": best.get("last_touch_ts", pd.NaT),
            "confluence_score": confluence_score,
            "weighted_confluence_score": weighted_confluence_score,
        }

    results = []
    for _, row in base.iterrows():
        results.append(_process_row(row))

    sr = pd.DataFrame(results, index=idx)
    sr["is_support"] = sr["primary_level_type"].astype(str).str.contains(
        "support", case=False, na=False
    )
    sr["is_resistance"] = sr["primary_level_type"].astype(str).str.contains(
        "resistance", case=False, na=False
    )
    return sr


def _build_event_dataset(
    ohlcv: pd.DataFrame,
    sr: pd.DataFrame,
    horizon_bars: int,
    min_ret: float,
    max_samples: int,
    strong_quantile: float,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct event-level dataset and strong/weak labels.

    This is a trimmed adaptation of build_event_dataset from the research script,
    focusing on the strong/weak classification target and feature block used by
    the SRLabelingXGBStep.
    """
    df = ohlcv.join(sr, how="inner")
    df = df.dropna(subset=["primary_level_price"])

    if df.empty:
        raise ValueError("No S/R levels available for event construction")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    level = df["primary_level_price"].astype(float)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=10).mean()

    touch_dist_price = 0.5 * atr_14
    fallback_dist = 0.004 * level
    touch_dist_price = touch_dist_price.fillna(fallback_dist)

    is_within_range = (low <= level) & (high >= level)
    abs_diff = (close - level).abs()
    is_close_proximity = abs_diff <= touch_dist_price
    touch_mask = is_within_range | is_close_proximity

    event_df = df.loc[touch_mask].copy()
    if event_df.empty:
        raise ValueError("No S/R touch events found with current configuration")

    if len(event_df) > max_samples:
        event_df = event_df.iloc[-max_samples:]

    fwd_close_full = close.shift(-horizon_bars)
    fwd_close_evt = fwd_close_full.loc[event_df.index]
    level_evt = level.loc[event_df.index]

    is_support = event_df["is_support"].astype(bool)
    is_resistance = event_df["is_resistance"].astype(bool)

    fwd_ret = pd.Series(index=event_df.index, dtype=float)
    fwd_ret.loc[is_support] = (
        (fwd_close_evt - level_evt) / level_evt
    ).loc[is_support]
    fwd_ret.loc[is_resistance] = (
        (level_evt - fwd_close_evt) / level_evt
    ).loc[is_resistance]

    fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan).dropna()
    event_df = event_df.loc[fwd_ret.index]

    y_reg = fwd_ret

    level_evt_final = event_df["primary_level_price"].astype(float)
    atr_evt_final = atr_14.loc[event_df.index]
    vol_unit = (atr_evt_final / level_evt_final.replace(0.0, np.nan)).abs()
    vol_unit = vol_unit.replace([np.inf, -np.inf], np.nan)

    abs_ret = y_reg.abs()
    if vol_unit.notna().any():
        norm_abs_ret = (abs_ret / vol_unit).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        norm_thr = float(norm_abs_ret.quantile(strong_quantile))
    else:
        norm_abs_ret = abs_ret.copy()
        norm_thr = float(norm_abs_ret.quantile(strong_quantile))

    abs_thr = float(abs_ret.quantile(strong_quantile))
    if norm_thr <= 0:
        norm_thr = max(abs_thr, min_ret)

    tprint_info(
        f"Strong label thresholds: min_ret={min_ret:.4f}, "
        f"strong_quantile={strong_quantile:.2f}, norm_thr={norm_thr:.4f}, abs_thr={abs_thr:.4f}"
    )

    strong_mask = (abs_ret >= min_ret) & (norm_abs_ret >= norm_thr)
    y_cls = strong_mask.astype(int)

    lvl_feats = pd.DataFrame(index=event_df.index)
    lvl_feats["dist_to_level_pct"] = ((close - level) / level).loc[event_df.index]

    src_series = event_df["primary_level_source"].astype(str)
    lvl_feats["src_is_kde"] = src_series.str.contains("kde", case=False, na=False).astype(float)
    lvl_feats["src_is_htf"] = src_series.str.contains("htf|pdh|pdl", case=False, na=False).astype(float)

    lvl_feats["is_support"] = is_support.astype(float)
    lvl_feats["is_resistance"] = is_resistance.astype(float)

    lvl_feats["meta_touch_count"] = event_df["primary_level_touch_count"].astype(float)
    lvl_feats["meta_prominence"] = event_df["primary_level_prominence"].astype(float)
    lvl_feats["meta_vol_depth"] = event_df["primary_level_volume_depth_ratio"].astype(float)

    lvl_feats["confluence_score"] = event_df["confluence_score"].astype(float)
    lvl_feats["weighted_confluence_score"] = event_df["weighted_confluence_score"].astype(float)

    current_ts = event_df.index.to_series()
    first_touch = pd.to_datetime(event_df["primary_level_first_touch_ts"])
    last_touch = pd.to_datetime(event_df["primary_level_last_touch_ts"])

    lvl_feats["level_age_hours"] = (current_ts - first_touch).dt.total_seconds() / 3600.0
    lvl_feats["hours_since_last_test"] = (current_ts - last_touch).dt.total_seconds() / 3600.0
    lvl_feats["level_age_hours"] = lvl_feats["level_age_hours"].fillna(0.0)
    lvl_feats["hours_since_last_test"] = lvl_feats["hours_since_last_test"].fillna(0.0)

    lvl_feats["recent_vol_consumed"] = event_df.get("recent_vol_intensity", 0.0)

    features = lvl_feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return features, y_cls


class SRLabelingXGBStep(BaseStep):
    """XGB-based SR labeling specialist (strong vs weak S/R events)."""

    def __init__(self, step_name: str = "sr_labeling_xgb") -> None:
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("SRLabelingXGBStep") if hasattr(logger, "getChild") else logger
        tprint(f"✅ Initialized {step_name} step (SR Labeling XGB)", "SUCCESS")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        metrics: Dict[str, Any] = {}
        artifacts: list[str] = []

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            timeframe = str(config.get("timeframe", "15m"))
            direction = str(config.get("direction", "long"))

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model="sr_labeling_xgb",
            )

            tprint_info(
                f"🚀 Starting {self.step_name} for {symbol} on {exchange} @ {timeframe}"
            )

            # Load historical OHLCV for the requested timeframe using the
            # centralized BaseStep loader so execution_mode/blank-mode
            # lookback windows are honoured.
            market_data, market_source = self.load_market_data_or_fail(
                {**config, "timeframe": timeframe},
                pipeline_state={},
                allow_config_override=True,
            )

            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                raise ValueError("Loaded market data is empty or not a DataFrame")

            if not isinstance(market_data.index, pd.DatetimeIndex):
                market_data = market_data.copy()
                market_data.index = pd.to_datetime(market_data.index)
                market_data = market_data.sort_index()

            # SR levels and event dataset
            sr = _generate_sr_levels(market_data)

            horizon_bars = int(config.get("sr_labeling_horizon_bars", 48))
            min_ret = float(config.get("sr_labeling_min_ret", 0.005))
            max_samples = int(config.get("sr_labeling_max_samples", 20000))
            strong_quantile = float(config.get("sr_labeling_strong_quantile", 0.7))

            features, y_cls = _build_event_dataset(
                ohlcv=market_data,
                sr=sr,
                horizon_bars=horizon_bars,
                min_ret=min_ret,
                max_samples=max_samples,
                strong_quantile=strong_quantile,
            )

            # Align and sanitize
            y_cls = y_cls.loc[features.index]
            valid_mask = y_cls.notna()
            features = features.loc[valid_mask]
            y_cls = y_cls.loc[valid_mask].astype(int)

            if len(features) < 200:
                raise RuntimeError(f"Insufficient SR events for training: {len(features)} < 200")

            X = features.select_dtypes(include=[np.number]).copy()
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            n = len(X)
            train_end = int(0.6 * n)
            val_end = int(0.8 * n)
            idx = X.index

            X_train = X.iloc[:train_end]
            y_train = y_cls.loc[X_train.index]
            X_val = X.iloc[train_end:val_end]
            y_val = y_cls.loc[X_val.index]
            X_test = X.iloc[val_end:]
            y_test = y_cls.loc[X_test.index]

            enable_calibration = bool(config.get("sr_labeling_enable_calibration", True))
            iso_model = None
            use_calibrated = False

            clf = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=int(config.get("sr_labeling_n_estimators", 400)),
                max_depth=int(config.get("sr_labeling_max_depth", 4)),
                learning_rate=float(config.get("sr_labeling_learning_rate", 0.03)),
                subsample=float(config.get("sr_labeling_subsample", 0.8)),
                colsample_bytree=float(config.get("sr_labeling_colsample_bytree", 0.8)),
                tree_method="hist",
            )

            tprint_info(
                f"Training XGBClassifier for SR strong/weak events on {len(X_train)} train / {len(X_val)} val / {len(X_test)} test samples"
            )
            clf.fit(X_train.values, y_train.values)

            proba_val = None
            if enable_calibration and len(X_val) > 0:
                proba_val = clf.predict_proba(X_val.values)[:, 1]

            proba_test_raw = None
            if len(X_test) > 0:
                proba_test_raw = clf.predict_proba(X_test.values)[:, 1]

            if (
                enable_calibration
                and IsotonicRegression is not None
                and proba_val is not None
                and proba_test_raw is not None
                and len(np.unique(y_val.values)) > 1
            ):
                try:
                    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
                    iso.fit(proba_val, y_val.values)
                    proba_test_cal = iso.predict(proba_test_raw)
                    proba_test_cal = np.clip(proba_test_cal, 0.0, 1.0)

                    from sklearn.metrics import log_loss, roc_auc_score

                    ll_raw = float(log_loss(y_test.values, proba_test_raw))
                    auc_raw = float(roc_auc_score(y_test.values, proba_test_raw))
                    ll_cal = float(log_loss(y_test.values, proba_test_cal))
                    auc_cal = float(roc_auc_score(y_test.values, proba_test_cal))

                    metrics["oof_log_loss_raw"] = ll_raw
                    metrics["oof_auc_raw"] = auc_raw
                    metrics["oof_log_loss"] = ll_cal
                    metrics["oof_auc"] = auc_cal

                    tprint_info(
                        "SR strong/weak hold-out metrics (raw vs isotonic): "
                        f"log_loss_raw={ll_raw:.4f}, AUC_raw={auc_raw:.4f}, "
                        f"log_loss_iso={ll_cal:.4f}, AUC_iso={auc_cal:.4f}"
                    )

                    iso_model = iso
                    use_calibrated = True
                except Exception as exc:
                    tprint_warning(f"Failed to apply isotonic calibration: {exc}")

            if not use_calibrated and proba_test_raw is not None and len(X_test) > 0:
                try:
                    from sklearn.metrics import log_loss, roc_auc_score

                    ll = float(log_loss(y_test.values, proba_test_raw))
                    auc = float(roc_auc_score(y_test.values, proba_test_raw))
                    metrics["oof_log_loss"] = ll
                    metrics["oof_auc"] = auc
                    tprint_info(
                        f"SR strong/weak hold-out metrics: log_loss={ll:.4f}, AUC={auc:.4f}"
                    )
                except Exception as exc:
                    tprint_warning(f"Failed to compute hold-out metrics: {exc}")

            # Dense per-bar probabilities on the 15m grid
            all_proba = clf.predict_proba(X.values)[:, 1]
            if enable_calibration and use_calibrated and iso_model is not None:
                try:
                    all_proba = iso_model.predict(all_proba)
                    all_proba = np.clip(all_proba, 0.0, 1.0)
                except Exception as exc:
                    tprint_warning(f"Failed to apply isotonic calibration to full series: {exc}")

            event_proba = pd.Series(all_proba, index=X.index, name="sr_labeling_xgb_prob")

            dense_series = event_proba.reindex(market_data.index).ffill()

            preds_df = pd.DataFrame(
                {
                    "timestamp": market_data.index,
                    "sr_labeling_xgb_prob": dense_series.values,
                }
            )

            preds_artifact_name = f"sr_labeling_xgb_predictions_{timeframe}"
            preds_metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "direction": direction,
                "source_market_data": market_source,
                "n_events": int(len(event_proba)),
                "n_bars": int(len(market_data)),
                "train_start": str(idx.min()),
                "train_end": str(idx.max()),
            }

            preds_path = self._save_artifact(
                data=preds_df,
                artifact_name=preds_artifact_name,
                artifact_type="data",
                data_category="predictions",
                metadata=preds_metadata,
            )
            artifacts.append(preds_path)

            tprint_success(
                f"Saved SR labeling predictions to {preds_path} (artifact={preds_artifact_name})"
            )

            execution_time = time.time() - start_time
            return {
                "success": True,
                "artifacts": artifacts,
                "metrics": metrics,
                "execution_time": execution_time,
            }

        except Exception as exc:
            execution_time = time.time() - start_time
            err_msg = f"SRLabelingXGBStep failed: {exc}"
            self.logger.error(err_msg)
            tprint_error(err_msg)
            return {
                "success": False,
                "artifacts": artifacts,
                "metrics": metrics,
                "error": str(exc),
                "execution_time": execution_time,
            }
