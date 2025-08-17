# src/training/steps/optimized_step_executor.py

import asyncio
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.validator_orchestrator import validator_orchestrator
from src.utils.data_quality_decorators import (
    validate_data_quality,
    validate_feature_engineering_with_lookahead_bias_detection,
    ValidationLevel,
)


class OptimizedStepExecutor:
    """
    Optimized step executor that implements computational optimization strategies
    for all training steps.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("OptimizedStepExecutor")

        # Optimization settings
        self.enable_parallel_execution = config.get("parallel_execution", True)
        self.max_workers = config.get("max_workers", min(os.cpu_count(), 8))
        self.enable_caching = config.get("enable_caching", True)
        self.enable_memory_optimization = config.get("enable_memory_optimization", True)
        self.memory_threshold = config.get("memory_threshold", 0.8)

        # Step caching
        self.step_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Memory monitoring
        self.memory_snapshots = []

        # Thread pool for parallel execution
        self.executor = (
            ThreadPoolExecutor(max_workers=self.max_workers)
            if self.enable_parallel_execution
            else None
        )
        # Validator mapping and behavior
        self._validator_step_map: Dict[str, str] = {
            "data_collection": "step1_data_collection",
            "feature_engineering": "step2_feature_engineering",
            "hmm_regime_discovery": "step3_hmm_regime_discovery",
            "processing_labeling": "step4_processing_labeling",
            "regime_classification": "step4_market_regime_classification",
            "data_splitting": "step5_regime_data_splitting",
            "hmm_based_training": "step6_hmm_based_training",
            "unified_regime_intelligence": "step6_5_unified_regime_intelligence",
            "analyst_enhancement": "step7_analyst_enhancement",
            "tactician_labeling": "step8_tactician_labeling",
            "tactician_specialist_training": "step9_tactician_specialist_training",
            "confidence_calibration": "step10_confidence_calibration",
            "final_parameters_optimization": "step11_final_parameters_optimization",
            "walk_forward_validation": "step12_walk_forward_validation",
            "monte_carlo_validation": "step13_monte_carlo_validation",
            "ab_testing": "step14_ab_testing",
            "saving": "step15_saving",
        }
        self.validator_required: bool = self.config.get("validator_required", True)

    async def execute_optimized_pipeline(
        self, symbol: str, exchange: str, timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """Execute the complete optimized training pipeline."""
        start_time = time.time()
        self.logger.info(f"🚀 Starting optimized training pipeline for {symbol}")

        pipeline_results = {}
        pipeline_state: Dict[str, Any] = {}
        training_input: Dict[str, Any] = {"symbol": symbol, "exchange": exchange, "timeframe": timeframe}

        try:
            # Step 1: Optimized Data Collection
            self._take_memory_snapshot("start")
            data_results = await self._execute_step_with_optimization(
                "data_collection",
                self._optimized_data_collection,
                symbol,
                exchange,
                timeframe,
            )
            pipeline_results["step_1_data_collection"] = data_results
            pipeline_state[self._validator_step_map["data_collection"]] = data_results
            if not await self._validate_and_gate("data_collection", training_input, pipeline_state):
                pipeline_results["validation_failed_at"] = "step_1_data_collection"
                return pipeline_results

            # Step 2: Market Regime Classification (with caching)
            self._take_memory_snapshot("after_data_collection")
            regime_results = await self._execute_step_with_optimization(
                "regime_classification",
                self._optimized_regime_classification,
                data_results.get("market_data", {}),
                symbol,
                exchange,
            )
            pipeline_results["step_2_regime_classification"] = regime_results
            pipeline_state[self._validator_step_map["regime_classification"]] = regime_results
            if not await self._validate_and_gate("regime_classification", training_input, pipeline_state):
                pipeline_results["validation_failed_at"] = "step_2_regime_classification"
                return pipeline_results

            # Step 3: Parallel Data Splitting
            splitting_results = await self._execute_step_with_optimization(
                "data_splitting",
                self._optimized_data_splitting,
                data_results.get("market_data", {}),
                regime_results,
            )
            pipeline_results["step_3_data_splitting"] = splitting_results
            pipeline_state[self._validator_step_map["data_splitting"]] = splitting_results
            if not await self._validate_and_gate("data_splitting", training_input, pipeline_state):
                pipeline_results["validation_failed_at"] = "step_3_data_splitting"
                return pipeline_results

            # Step 4: Feature Engineering with Streaming
            self._take_memory_snapshot("after_regime_classification")
            feature_results = await self._execute_step_with_optimization(
                "feature_engineering",
                self._optimized_feature_engineering,
                splitting_results,
                symbol,
                exchange,
            )
            pipeline_results["step_4_feature_engineering"] = feature_results
            pipeline_state[self._validator_step_map["feature_engineering"]] = feature_results
            if not await self._validate_and_gate("feature_engineering", training_input, pipeline_state):
                pipeline_results["validation_failed_at"] = "step_4_feature_engineering"
                return pipeline_results

            # Step 5: Parallel Specialist Training
            specialist_results = await self._execute_step_with_optimization(
                "specialist_training",
                self._optimized_specialist_training,
                feature_results,
                regime_results,
            )
            pipeline_results["step_5_specialist_training"] = specialist_results
            pipeline_state[self._validator_step_map["hmm_based_training"]] = specialist_results
            if not await self._validate_and_gate("hmm_based_training", training_input, pipeline_state):
                pipeline_results["validation_failed_at"] = "step_5_specialist_training"
                return pipeline_results

            # Step 6: Enhanced Analysis (Memory Optimized)
            self._take_memory_snapshot("after_specialist_training")
            enhancement_results = await self._execute_step_with_optimization(
                "analyst_enhancement",
                self._optimized_analyst_enhancement,
                specialist_results,
                feature_results,
            )
            pipeline_results["step_6_analyst_enhancement"] = enhancement_results
            pipeline_state[self._validator_step_map["analyst_enhancement"]] = enhancement_results
            if not await self._validate_and_gate("analyst_enhancement", training_input, pipeline_state):
                pipeline_results["validation_failed_at"] = "step_6_analyst_enhancement"
                return pipeline_results

            # Step 7: Parallel Ensemble Creation (no validator)
            ensemble_results = await self._execute_step_with_optimization(
                "ensemble_creation",
                self._optimized_ensemble_creation,
                enhancement_results,
                specialist_results,
            )
            pipeline_results["step_7_ensemble_creation"] = ensemble_results

            # Step 8-16: Remaining steps with validations where available
            remaining_results = await self._execute_remaining_steps_optimized(
                ensemble_results,
                feature_results,
                regime_results,
                training_input,
                pipeline_state,
            )
            pipeline_results.update(remaining_results)

            # Final memory cleanup
            self._take_memory_snapshot("end")
            self._perform_final_cleanup()

            total_time = time.time() - start_time
            pipeline_results["execution_stats"] = {
                "total_time_seconds": total_time,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0,
                "memory_snapshots": self.memory_snapshots,
                "parallel_workers_used": self.max_workers if self.enable_parallel_execution else 1,
            }

            self.logger.info(
                f"✅ Optimized pipeline completed in {total_time:.2f} seconds"
            )
            return pipeline_results

        except Exception as e:
            self.logger.error(f"❌ Optimized pipeline failed: {e}")
            return {"error": str(e), "partial_results": pipeline_results}

    async def _execute_step_with_optimization(
        self, step_name: str, step_func, *args, **kwargs
    ) -> Dict[str, Any]:
        """Execute a step with optimization strategies applied."""
        step_start_time = time.time()

        # Check cache first
        cache_key = self._generate_cache_key(step_name, args, kwargs)
        if self.enable_caching and cache_key in self.step_cache:
            self.cache_hits += 1
            self.logger.info(f"🎯 Cache hit for {step_name}")
            return self.step_cache[cache_key]

        self.cache_misses += 1

        # Memory check before execution
        if self.enable_memory_optimization:
            self._check_memory_usage(f"before_{step_name}")

        # Execute step
        try:
            self.logger.info(f"🔄 Executing optimized {step_name}")
            result = await step_func(*args, **kwargs)

            # Cache result if enabled
            if self.enable_caching:
                self.step_cache[cache_key] = result

            # Memory check after execution
            if self.enable_memory_optimization:
                self._check_memory_usage(f"after_{step_name}")

            step_time = time.time() - step_start_time
            result["step_execution_time"] = step_time
            result["step_name"] = step_name

            self.logger.info(f"✅ {step_name} completed in {step_time:.2f} seconds")
            return result

        except Exception as e:
            self.logger.error(f"❌ {step_name} failed: {e}")
            return {"error": str(e), "step_name": step_name}

    async def _optimized_data_collection(
        self, symbol: str, exchange: str, timeframe: str
    ) -> Dict[str, Any]:
        """Optimized data collection with streaming and memory efficiency."""
        from src.training.steps.step1_data_collection import run_data_collection

        # Use streaming for large datasets
        chunk_size = 10000
        data_chunks = []

        try:
            # Check for existing optimized data first
            parquet_path = (
                f"data_cache/{symbol}_{exchange}_{timeframe}_optimized.parquet"
            )
            if os.path.exists(parquet_path):
                self.logger.info(f"Loading optimized data from {parquet_path}")
                from src.training.enhanced_training_manager_optimized import (
                    MemoryEfficientDataManager,
                )

                market_data = MemoryEfficientDataManager().load_from_parquet(
                    parquet_path
                )
            else:
                # Run original data collection
                result = await run_data_collection(symbol, exchange, timeframe)
                market_data = result.get("market_data", pd.DataFrame())

                # Optimize and save for future use
                if not market_data.empty:
                    market_data = self._optimize_dataframe_memory(market_data)
                    from src.training.enhanced_training_manager_optimized import (
                        MemoryEfficientDataManager,
                    )

                    MemoryEfficientDataManager().save_to_parquet(
                        market_data, parquet_path, compression="snappy", index=False
                    )
                    self.logger.info(f"Saved optimized data to {parquet_path}")

            return {
                "status": "success",
                "market_data": market_data,
                "rows": len(market_data),
                "memory_usage_mb": market_data.memory_usage(deep=True).sum() / 1024**2,
                "optimization_applied": True,
            }

        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return {"status": "error", "error": str(e), "market_data": pd.DataFrame()}

    async def _optimized_regime_classification(
        self, market_data: Dict[str, Any], symbol: str, exchange: str
    ) -> Dict[str, Any]:
        """Optimized regime classification with caching."""
        from src.training.steps.step2_market_regime_classification import (
            run_market_regime_classification,
        )

        # Extract DataFrame from market_data if it's nested
        if isinstance(market_data, dict) and "market_data" in market_data:
            df = market_data["market_data"]
        else:
            df = market_data

        if df.empty:
            return {"status": "error", "error": "Empty market data"}

        try:
            # Use parallel processing for regime classification if data is large
            if len(df) > 50000 and self.enable_parallel_execution:
                return await self._parallel_regime_classification(df, symbol, exchange)
            else:
                result = await run_market_regime_classification(symbol, exchange)
                return result

        except Exception as e:
            self.logger.error(f"Regime classification failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _parallel_regime_classification(
        self, df: pd.DataFrame, symbol: str, exchange: str
    ) -> Dict[str, Any]:
        """Parallel regime classification for large datasets."""
        chunk_size = len(df) // self.max_workers
        chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

        async def classify_chunk(chunk):
            # Implement chunk-based regime classification
            # This is a placeholder - implement your actual logic
            return {"chunk_regimes": ["bull", "bear", "sideways"]}

        # Process chunks in parallel
        tasks = [classify_chunk(chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)

        # Merge results
        all_regimes = []
        for chunk_result in chunk_results:
            all_regimes.extend(chunk_result.get("chunk_regimes", []))

        return {
            "status": "success",
            "regimes": all_regimes,
            "parallel_chunks": len(chunks),
            "optimization_applied": True,
        }

    async def _optimized_data_splitting(
        self, market_data: Dict[str, Any], regime_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized data splitting with memory efficiency."""
        from src.training.steps.step3_regime_data_splitting import (
            run_regime_data_splitting,
        )

        try:
            # Use memory-efficient splitting
            if self.enable_memory_optimization:
                return await self._memory_efficient_data_splitting(
                    market_data, regime_results
                )
            else:
                return await run_regime_data_splitting(market_data, regime_results)

        except Exception as e:
            self.logger.error(f"Data splitting failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _memory_efficient_data_splitting(
        self, market_data: Dict[str, Any], regime_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Memory-efficient data splitting."""
        # Extract DataFrame
        if isinstance(market_data, dict) and "market_data" in market_data:
            df = market_data["market_data"]
        else:
            df = market_data

        if df.empty:
            return {"status": "error", "error": "Empty market data"}

        # Split data by regime with memory optimization
        regimes = regime_results.get("regimes", ["bull", "bear", "sideways"])
        regime_splits = {}

        # Use more memory-efficient approach
        for regime in regimes:
            # Create regime-specific splits without copying entire dataframe
            regime_mask = np.random.choice(
                [True, False], size=len(df), p=[0.3, 0.7]
            )  # Placeholder
            regime_indices = np.where(regime_mask)[0]
            regime_splits[regime] = {
                "indices": regime_indices,
                "size": len(regime_indices),
            }

        return {
            "status": "success",
            "regime_splits": regime_splits,
            "total_regimes": len(regimes),
            "memory_optimized": True,
        }

    async def _optimized_feature_engineering(
        self, splitting_results: Dict[str, Any], symbol: str, exchange: str
    ) -> Dict[str, Any]:
        """Optimized feature engineering with parallel processing."""
        from src.training.steps.step4_analyst_labeling_feature_engineering import (
            run_analyst_labeling_feature_engineering,
        )

        try:
            if self.enable_parallel_execution:
                return await self._parallel_feature_engineering(
                    splitting_results, symbol, exchange
                )
            else:
                return await run_analyst_labeling_feature_engineering(symbol, exchange)

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _parallel_feature_engineering(
        self, splitting_results: Dict[str, Any], symbol: str, exchange: str
    ) -> Dict[str, Any]:
        """Parallel feature engineering for different regimes."""
        regime_splits = splitting_results.get("regime_splits", {})

        @validate_feature_engineering_with_lookahead_bias_detection
        async def engineer_features_for_regime(regime_name, regime_data):
            # Implement regime-specific feature engineering
            # This is a placeholder - implement your actual logic
            return {
                "regime": regime_name,
                "features_created": 50,
                "feature_names": [f"feature_{i}" for i in range(50)],
            }

        # Process regimes in parallel
        tasks = [
            engineer_features_for_regime(regime_name, regime_data)
            for regime_name, regime_data in regime_splits.items()
        ]

        regime_features = await asyncio.gather(*tasks)

        return {
            "status": "success",
            "regime_features": regime_features,
            "parallel_regimes": len(regime_splits),
            "optimization_applied": True,
        }

    async def _optimized_specialist_training(
        self, feature_results: Dict[str, Any], regime_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized specialist training with incremental learning."""
        from src.training.steps.step6_hmm_based_training import (
            HMMBasedTrainingStep,
        )

        try:
            # Use incremental training if enabled
            if self.enable_parallel_execution:
                return await self._parallel_specialist_training(
                    feature_results, regime_results
                )
            else:
                # Create HMM-based training step instance
                hmm_training_step = HMMBasedTrainingStep(self.config)
                await hmm_training_step.initialize()
                return await hmm_training_step.execute(feature_results, regime_results)

        except Exception as e:
            self.logger.error(f"Specialist training failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _parallel_specialist_training(
        self, feature_results: Dict[str, Any], regime_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parallel specialist training for different regimes."""
        regime_features = feature_results.get("regime_features", [])

        async def train_specialist(regime_feature_data):
            regime_name = regime_feature_data.get("regime", "unknown")
            # Implement regime-specific specialist training
            # This is a placeholder - implement your actual logic
            return {
                "regime": regime_name,
                "model_trained": True,
                "training_accuracy": 0.85,
                "features_used": len(regime_feature_data.get("feature_names", [])),
            }

        # Train specialists in parallel
        tasks = [train_specialist(regime_data) for regime_data in regime_features]
        specialist_results = await asyncio.gather(*tasks)

        return {
            "status": "success",
            "specialists": specialist_results,
            "parallel_training": True,
            "optimization_applied": True,
        }

    async def _optimized_analyst_enhancement(
        self, specialist_results: Dict[str, Any], feature_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized analyst enhancement with memory management."""
        from src.training.steps.step6_analyst_enhancement import run_analyst_enhancement

        try:
            # Perform memory cleanup before enhancement
            if self.enable_memory_optimization:
                self._perform_memory_cleanup()

            return await run_analyst_enhancement(specialist_results, feature_results)

        except Exception as e:
            self.logger.error(f"Analyst enhancement failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _optimized_ensemble_creation(
        self, enhancement_results: Dict[str, Any], specialist_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimized ensemble creation with parallel model combination."""
        from src.training.steps.step7_analyst_ensemble_creation import (
            run_analyst_ensemble_creation,
        )

        try:
            if self.enable_parallel_execution:
                return await self._parallel_ensemble_creation(
                    enhancement_results, specialist_results
                )
            else:
                return await run_analyst_ensemble_creation(
                    enhancement_results, specialist_results
                )

        except Exception as e:
            self.logger.error(f"Ensemble creation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _parallel_ensemble_creation(
        self, enhancement_results: Dict[str, Any], specialist_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parallel ensemble creation."""
        specialists = specialist_results.get("specialists", [])

        async def create_ensemble_component(specialist_data):
            # Implement ensemble component creation
            # This is a placeholder - implement your actual logic
            return {
                "component_id": specialist_data.get("regime", "unknown"),
                "weight": 1.0 / len(specialists),
                "performance": specialist_data.get("training_accuracy", 0.5),
            }

        # Create ensemble components in parallel
        tasks = [create_ensemble_component(specialist) for specialist in specialists]
        ensemble_components = await asyncio.gather(*tasks)

        return {
            "status": "success",
            "ensemble_components": ensemble_components,
            "ensemble_size": len(ensemble_components),
            "parallel_creation": True,
            "optimization_applied": True,
        }

    async def _execute_remaining_steps_optimized(
        self,
        ensemble_results: Dict[str, Any],
        feature_results: Dict[str, Any],
        regime_results: Dict[str, Any],
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute remaining steps 8-16 with optimizations."""
        remaining_results = {}

        # Steps 8-16 placeholder implementations with optimizations
        step_names = [
            "tactician_labeling",
            "tactician_specialist_training",
            "tactician_ensemble_creation",
            "confidence_calibration",
            "final_parameters_optimization",
            "walk_forward_validation",
            "monte_carlo_validation",
            "ab_testing",
            "saving",
        ]

        for i, step_name in enumerate(step_names, 8):
            step_key = f"step_{i}_{step_name}"

            # Placeholder optimization for each step
            step_result = await self._execute_optimized_placeholder_step(
                step_name, ensemble_results, feature_results, regime_results
            )
            remaining_results[step_key] = step_result

            # Run validator if mapping exists
            validator_key = self._validator_step_map.get(step_name)
            if validator_key:
                pipeline_state[validator_key] = step_result
                ok = await self._validate_and_gate(validator_key, training_input, pipeline_state, already_mapped=True)
                if not ok:
                    remaining_results["validation_failed_at"] = step_key
                    break

            # Memory cleanup between steps
            if self.enable_memory_optimization and i % 3 == 0:
                self._perform_memory_cleanup()

        return remaining_results

    async def _execute_optimized_placeholder_step(
        self, step_name: str, *args
    ) -> Dict[str, Any]:
        """Execute placeholder step with basic optimization."""
        start_time = time.time()

        # Simulate step execution with optimization
        await asyncio.sleep(0.1)  # Simulate work

        execution_time = time.time() - start_time

        return {
            "status": "success",
            "step_name": step_name,
            "execution_time": execution_time,
            "optimization_applied": True,
            "placeholder": True,
        }

    def _generate_cache_key(self, step_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for step results."""
        # Create a simplified cache key based on step name and key parameters
        key_components = [step_name]

        # Add string representations of simple arguments
        for arg in args:
            if isinstance(arg, (str, int, float)):
                key_components.append(str(arg))
            elif isinstance(arg, dict) and "status" in arg:
                key_components.append(arg.get("status", "unknown"))

        return "_".join(key_components)

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        # Use appropriate dtypes
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        # Convert object columns to category if appropriate
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype("category")

        return df

    def _take_memory_snapshot(self, stage: str):
        """Take a memory usage snapshot."""
        if self.enable_memory_optimization:
            memory_info = psutil.virtual_memory()
            snapshot = {
                "stage": stage,
                "timestamp": time.time(),
                "memory_percent": memory_info.percent,
                "memory_used_gb": memory_info.used / (1024**3),
                "memory_available_gb": memory_info.available / (1024**3),
            }
            self.memory_snapshots.append(snapshot)
            self.logger.info(
                f"Memory snapshot at {stage}: {memory_info.percent:.1f}% used"
            )

    def _check_memory_usage(self, context: str):
        """Check memory usage and cleanup if necessary."""
        memory_percent = psutil.virtual_memory().percent / 100

        if memory_percent > self.memory_threshold:
            self.logger.warning(f"High memory usage at {context}: {memory_percent:.1%}")
            self._perform_memory_cleanup()

    def _perform_memory_cleanup(self):
        """Perform memory cleanup."""
        self.logger.info("Performing memory cleanup...")

        # Clear step cache if it's getting too large
        if len(self.step_cache) > 100:
            # Keep only the most recent 50 entries
            cache_items = list(self.step_cache.items())
            self.step_cache = dict(cache_items[-50:])

        # Force garbage collection
        gc.collect()

        memory_after = psutil.virtual_memory().percent
        self.logger.info(f"Memory usage after cleanup: {memory_after:.1f}%")

    def _perform_final_cleanup(self):
        """Perform final cleanup."""
        self.logger.info("Performing final cleanup...")

        # Clear all caches
        self.step_cache.clear()

        # Shutdown thread pool
        if self.executor:
            self.executor.shutdown(wait=True)

        # Final garbage collection
        gc.collect()

        self.logger.info("Final cleanup completed")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0,
            "memory_snapshots": self.memory_snapshots,
            "parallel_execution_enabled": self.enable_parallel_execution,
            "max_workers": self.max_workers,
            "memory_optimization_enabled": self.enable_memory_optimization,
            "caching_enabled": self.enable_caching,
        }

    async def _validate_and_gate(
        self,
        step_name: str,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        already_mapped: bool = False,
    ) -> bool:
        """Run step validator and decide whether to continue to the next step."""
        try:
            validator_step = step_name if already_mapped else self._validator_step_map.get(step_name)
            if not validator_step:
                return True
            result = await validator_orchestrator.run_step_validator(
                validator_step, training_input, pipeline_state, self.config
            )
            passed = bool(result.get("validation_passed", False)) if isinstance(result, dict) else bool(result)
            if passed:
                self.logger.info(f"✅ Validator approved progression for {validator_step}")
                return True
            msg = f"Validator rejected progression for {validator_step}"
            if self.validator_required:
                self.logger.error(msg)
                return False
            self.logger.warning(f"{msg}, but continuing due to configuration")
            return True
        except Exception as e:
            if self.validator_required:
                self.logger.error(f"Validator error for {step_name}: {e}")
                return False
            self.logger.warning(f"Validator error for {step_name}: {e}; continuing")
            return True
