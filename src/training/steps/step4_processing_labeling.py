# src/training/steps/step2_processing_labeling_feature_engineering.py

import asyncio
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
    OptimizedTripleBarrierLabeling,
)
from src.training.steps.unified_data_loader import get_unified_data_loader
from src.training.steps.vectorized_labelling_orchestrator import (
    VectorizedLabellingOrchestrator,
)

# Import decorators from centralized module
from src.utils.centralized_decorators import (
    artifact_versioning,
    artifact_write_lock,
    auto_fix_data_quality_issues,
    circuit_breaker_protection,
    debug_training_step,
    deterministic_seed,
    guard_dataframe_nulls,
    handle_errors,
    idempotent_step,
    memory_efficient,
    nan_inf_and_constant_guard,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    time_budget_watchdog,
    validate_step_output,
    validate_step_prerequisites,
    with_tracing_span,
)
from src.utils.logger import system_logger as _logger


@with_tracing_span("step4._build_sr_levels", log_args=False)
@guard_dataframe_nulls(mode="warn", arg_index=0)
@handle_errors(exceptions=(Exception,), default_return={"support_levels": [], "resistance_levels": []})
async def _build_sr_levels(price_df: pd.DataFrame) -> dict[str, Any]:
    try:
        lows = price_df["low"].astype(float)
        highs = price_df["high"].astype(float)
        window = min(len(lows), 2000)
        if window <= 0:
            return {"support_levels": [], "resistance_levels": []}
        lt = lows.tail(window).dropna()
        ht = highs.tail(window).dropna()
        if lt.empty or ht.empty:
            return {"support_levels": [], "resistance_levels": []}
        # Use robust percentiles as weak baseline levels, attach low strength
        support_prices = np.percentile(lt.values, [5, 15, 30]).tolist()
        resistance_prices = np.percentile(ht.values, [70, 85, 95]).tolist()

        # Deduplicate and produce dicts with strength
        def _mk_levels(vals, strength=0.2):
            out = []
            seen = set()
            for v in vals:
                r = round(float(v), 8)
                if r in seen:
                    continue
                seen.add(r)
                out.append({"price": r, "strength": float(strength)})
            return out

        return {
            "support_levels": _mk_levels(support_prices, 0.2),
            "resistance_levels": _mk_levels(resistance_prices, 0.2),
        }
    except Exception as e:
        return {"support_levels": [], "resistance_levels": []}


@with_tracing_span("step4._persist_sr_levels", log_args=False)
@handle_errors(exceptions=(Exception,), default_return=None)
def _persist_sr_levels(config: dict[str, Any], sr_levels: dict[str, Any], asof_ts: pd.Timestamp, ) -> None:
    """Append SR levels with timestamps to a persistent parquet for reuse.

    File path: data/training/{exchange}_{symbol}_sr_levels.parquet
    Schema: timestamp | level_type | price | strength | age
    """
    try:
        data_dir = config.get("data_dir", "data/training")
        symbol = config.get("symbol", "SYMB")
        exchange = config.get("exchange", "EXCH")
        path = f"{data_dir}/{exchange}_{symbol}_sr_levels.parquet"
        # Build frame from provided sr_levels
        rows: list[dict[str, Any]] = []
        for kind in ("support_levels", "resistance_levels"):
            for lvl in sr_levels.get(kind, []) or []:
                if isinstance(lvl, dict):
                    price = float(lvl.get("price"))
                    strength = float(lvl.get("strength", 0.2))
                else:
                    price = float(lvl)
                    strength = 0.2
                rows.append(
                    {
                        "timestamp": pd.to_datetime(asof_ts),
                        "level_type": "support"
                        if kind == "support_levels"
                        else "resistance",
                        "price": price,
                        "strength": strength,
                        "age": 0.0,
                    },
                )
        if not rows:
            return
        new_df = pd.DataFrame(rows)
        # Append or create
        import os

        if os.path.exists(path):
            try:
                old = pd.read_parquet(path)
                # Age existing levels: increase age by time delta in minutes
                if not old.empty:
                    max_old_ts = pd.to_datetime(old["timestamp"]).max()
                    delta_min = (
                        float(
                            (pd.to_datetime(asof_ts) - max_old_ts).total_seconds()
                            / 60.0,
                        )
                        if pd.notna(max_old_ts)
                        else 0.0
                    )
                    if "age" in old.columns:
                        old["age"] = old["age"].astype(float) + max(0.0, delta_min)
                combined = pd.concat([old, new_df], axis=0, ignore_index=True)
            except Exception as e:
                combined = new_df
        else:
            combined = new_df
        # Deduplicate near-identical level prices within a small epsilon per type+timestamp bucket
        try:
            eps = 1e-6
            combined["price_round"] = (combined["price"] / eps).round().astype("int64")
            combined = combined.sort_values(
                ["timestamp", "level_type", "price"],
            ).drop_duplicates(["timestamp", "level_type", "price_round"], keep="last")
            combined = combined.drop(columns=["price_round"], errors="ignore")
        except Exception as e:
            pass
        combined.to_parquet(path, index=False)
        _logger.info(f"💾 Persisted SR levels ({len(new_df)} new) -> {path}")
    except Exception as e:
        _logger.warning(f"⚠️ Persist SR levels skipped: {e}")


@deterministic_seed(42)
@idempotent_step(step_key="step4_processing_labeling")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=1800.0)
@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=8.0,
    min_disk_gb=5.0,
    data_quality_checks={"check_data_completeness": True},
)
@secure_data_processing(
    backup_before=True,
    integrity_checks=True,
    memory_cleanup=True,
    data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    cross_validation_isolation=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=90.0,
    disk_threshold_gb=10.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=1000,
    streaming_processing=True,
    memory_pool=True,
    cleanup_frequency=10,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=300.0,
)
@validate_step_output(
    required_files=["labeled_train.parquet", "labeled_validation.parquet", "labeled_test.parquet"],
    data_quality_checks={"check_output_completeness": True},
)
@quality_gate(
    model_performance_thresholds={"min_data_points": 100},
    data_quality_metrics={"completeness_threshold": 0.95},
)
@auto_fix_data_quality_issues
@handle_errors(exceptions=(Exception,), default_return=False, context="step4_processing_labeling")
async def run_step(symbol: str, exchange_name: str = "BINANCE", data_dir: str = "data/training", timeframe: str = "1m", exchange: str = "BINANCE", force_rerun: bool = False, pipeline_config: dict[str, Any] | None = None) -> bool:
    _logger.info(
        "🚀 Running Step 4: Processing & Labeling...",
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
            config.update(
                {
                    "vectorized_labelling_orchestrator": pipeline_config.get(
                        "vectorized_labelling_orchestrator", {},
                    ),
                },
            )

        data_loader = get_unified_data_loader(config)
        from src.config.constants import (
            BLANK_TRAINING_LOOKBACK_DAYS,
        )

        # Use lookback_days from config (should be passed from enhanced training manager)
        lookback_days = config.get("lookback_days", BLANK_TRAINING_LOOKBACK_DAYS)
        df = await data_loader.load_unified_data(
            symbol=symbol,
            exchange=actual_exchange,
            timeframe=timeframe,
            lookback_days=lookback_days,
            use_streaming=True
        )
        if df is None or df.empty:
            msg = f"🚨 No data found for {symbol} on {actual_exchange}"
            raise ValueError(msg)

        # Ensure timestamp column exists and is datetime
        if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])  # best-effort cast
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 2) Compute triple-barrier labels (binary) while preserving OHLCV
        lbl = OptimizedTripleBarrierLabeling(binary_classification=True)
        labeled = lbl.apply_triple_barrier_labeling_vectorized(
            df[
                [
                    c
        for c in ["open", "high", "low", "close", "volume", "timestamp"]
        if c in df.columns
                ]
            ].set_index("timestamp"),
        )
        labeled = labeled.reset_index()  # bring timestamp back as column

        # 3) Split into train/validation/test by time (70/15/15)
        n = len(labeled)
        if n < 100:
            _logger.warning(
                "⚠️ Very little data for step 2; proceeding with minimal splits",
            )
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
            f"✅ Wrote labeled splits: train={len(labeled_train)} val={len(labeled_val)} test={len(labeled_test)}"
        )

        # 5) Run vectorized orchestrator to derive feature space + meta strengths, and persist strengths snapshot
        try:
            orchestrator = VectorizedLabellingOrchestrator(config)
            ok = await orchestrator.initialize()
            if ok:
        # Prepare price/volume inputs for orchestrator
                price_cols = [
                    c
        for c in ["open", "high", "low", "close", "volume"]
        if c in df.columns
                ]
                price_data = df[["timestamp", *price_cols]].set_index("timestamp")
                volume_data = (
                    price_data[["volume"]]
                    if "volume" in price_data.columns
                    else pd.DataFrame(index=price_data.index)
                )

                # Compute SR levels for the price data
                sr_levels = await _build_sr_levels(price_data)
                # Persist detected SR levels with timestamp for reuse across steps
                try:
                    last_ts = pd.to_datetime(price_data.index.max())
                    _persist_sr_levels(config, sr_levels, last_ts)
                except Exception as e:
                    pass

                result = await orchestrator.orchestrate_labeling_and_feature_engineering(
                    price_data=volume_data, volume_data=None, sr_levels=sr_levels,
                )
                final_df: pd.DataFrame | None = None
                if isinstance(result, dict) and isinstance(
                    result.get("data"), pd.DataFrame,
                ):
                    final_df = result["data"]
                # Persist meta strengths if available (columns starting with 'sr_')
                if final_df is not None and not final_df.empty:
                    strength_cols = [
                        c for c in final_df.columns if c.lower().startswith("sr_")
                    ]
                    # Also include key SR context columns that don't start with 'sr_'
                    extra_cols = [
                        c
                        for c in (
                            "support_levels_count",
                            "resistance_levels_count",
                            "nearest_sr_distance",
                        )
                        if c in final_df.columns
                    ]
                    strength_cols = sorted(set(list(strength_cols) + extra_cols))
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
            _logger.warning(f"⚠️ Meta strengths persistence skipped: {e}")

        # 6) Persist label distribution per split for diagnostics
        try:
            dist = {
                "train": labeled_train.get("label", pd.Series(dtype=int))
                .value_counts(dropna=False)
                .to_dict(),
                "validation": labeled_val.get("label", pd.Series(dtype=int))
                .value_counts(dropna=False)
                .to_dict(),
                "test": labeled_test.get("label", pd.Series(dtype=int))
                .value_counts(dropna=False)
                .to_dict(),
            }
            with open(
                f"{data_dir}/{actual_exchange}_{symbol}_label_distribution.json",
                "w",
            ) as f:
                json.dump(dist, f, indent=2)
        except Exception as e:
            _logger.warning(f"⚠️ Label distribution persistence skipped: {e}")

        # 7) Persist label reliability (if available) for downstream gating/stacking
        try:
            from src.training.enhanced_training_manager import EnhancedTrainingManager

            etm = EnhancedTrainingManager(config)
            reliability = etm.get_label_reliability()
            with open(
                f"{data_dir}/{actual_exchange}_{symbol}_label_reliability.json", "w",
            ) as f:
                json.dump(reliability, f, indent=2)
        except Exception as e:
            _logger.warning(f"⚠️ Label reliability persistence skipped: {e}")

        return True

    except Exception as e:
        _logger.exception(f"🚨 Step 2 processing/labeling/FE failed: {e}")
        return False


if __name__ == "__main__":

    async def _test() -> None:
        await run_step("ETHUSDT", "BINANCE", "data/training")

    asyncio.run(_test())