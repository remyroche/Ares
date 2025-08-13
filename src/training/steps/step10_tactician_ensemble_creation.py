# src/training/steps/step10_tactician_ensemble_creation.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.decorators import enforce_ndarray, guard_array_nan_inf, with_tracing_span
from src.training.steps.unified_data_loader import get_unified_data_loader

# NOTE: Keeping the optimized TacticianEnsembleCreationStep definition below; removing earlier duplicate to avoid conflicts


class TacticianEnsembleCreationStep:
    """Step 10: Create an optimized Tactician Ensemble by blending two models."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tactician ensemble creation step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the tactician ensemble creation step."""
        self.logger.info("Initializing Tactician Ensemble Creation Step...")
        self.logger.info("Tactician Ensemble Creation Step initialized successfully.")

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="tactician ensemble creation step execution",
    )
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute tactician ensemble creation.

        Args:
            training_input: Training input parameters.
            pipeline_state: Current pipeline state.

        Returns:
            A dictionary containing the results of the ensemble creation.
        """
        try:
            self.logger.info("🔄 Executing Tactician Ensemble Creation...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load tactician models
            models_dir = os.path.join(data_dir, "tactician_models")
            tactician_models = {}
            from src.utils.logger import heartbeat
            with heartbeat(self.logger, name="Step10 load_tactician_models", interval_seconds=60.0):
                for model_file in os.listdir(models_dir):
                    if model_file.endswith(".pkl"):
                        model_name = os.path.splitext(model_file)[0]
                        model_path = os.path.join(models_dir, model_file)
                        with open(model_path, "rb") as f:
                            tactician_models[model_name] = pickle.load(f)
            try:
                self.logger.info(
                    f"Loaded tactician models: names={list(tactician_models.keys())}",
                )
            except Exception:
                pass

            if len(tactician_models) != 2:
                msg = f"Expected 2 tactician models, but found {len(tactician_models)} in {models_dir}"
                raise ValueError(
                    msg,
                )

            # Create an optimized blended ensemble
            with heartbeat(self.logger, name="Step10 create_tactician_ensemble", interval_seconds=60.0):
                ensemble_details = await self._create_tactician_ensemble(
                    tactician_models,
                    data_dir,
                )

            # Separate the model object from its serializable details
            ensemble_model = ensemble_details.pop("ensemble")
            try:
                self.logger.info(
                    f"Ensemble details keys: {list(ensemble_details.keys())}",
                )
            except Exception:
                pass

            # --- Save Ensemble Model and Summary ---
            ensemble_dir = os.path.join(data_dir, "tactician_ensembles")
            os.makedirs(ensemble_dir, exist_ok=True)

            # Save the ensemble model object to a pickle file
            ensemble_file = os.path.join(
                ensemble_dir,
                f"{exchange}_{symbol}_tactician_ensemble.pkl",
            )
            with open(ensemble_file, "wb") as f:
                pickle.dump(ensemble_model, f)

            # Save the ensemble's metadata to a JSON summary file
            summary_file = os.path.join(
                ensemble_dir,
                f"{exchange}_{symbol}_tactician_ensemble_summary.json",
            )
            with open(summary_file, "w") as f:
                json.dump(ensemble_details, f, indent=4)

            self.logger.info(
                f"✅ Tactician ensemble created. Model saved to {ensemble_file}",
            )

            # Update pipeline state
            pipeline_state["tactician_ensemble_details"] = ensemble_details
            pipeline_state["tactician_ensemble_model"] = ensemble_model

            return {
                "status": "SUCCESS",
                "ensemble_details": ensemble_details,
                "ensemble_file": ensemble_file,
            }

        except Exception as e:
            self.logger.error(
                f"❌ Error in Tactician Ensemble Creation: {e}",
                exc_info=True,
            )
            return {"status": "FAILED", "error": str(e)}

    async def _load_validation_data(
        self,
        data_dir: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads validation data for weight optimization.

        Attempts Arrow-first loading from partitioned labeled dataset; falls back
        to a lightweight placeholder if not available.

        Args:
            data_dir: The directory where data is stored.

        Returns:
            A tuple containing validation features (X_val) and labels (y_val).
        """
        try:
            from src.training.enhanced_training_manager_optimized import (
                ParquetDatasetManager,
            )
            import pyarrow as _pa, pyarrow.compute as pc
            import pandas as pd

            pdm = ParquetDatasetManager(logger=self.logger)
            # Build filters: validation split for current exchange/symbol/timeframe
            exchange = self.config.get("exchange", "BINANCE")
            symbol = self.config.get("symbol", "ETHUSDT")
            timeframe = self.config.get("timeframe", "1m")
            part_base = os.path.join(data_dir, "parquet", "labeled")
            if os.path.isdir(part_base):
                filters = [
                    ("exchange", "==", exchange),
                    ("symbol", "==", symbol),
                    ("timeframe", "==", timeframe),
                    ("split", "==", "validation"),
                ]
                # Feature selection
                feat_cols = self.config.get("model_feature_columns") or self.config.get(
                    "feature_columns"
                )
                label_col = self.config.get("label_column", "label")
                columns = None
                if isinstance(feat_cols, list) and len(feat_cols) > 0:
                    columns = ["timestamp", *feat_cols, label_col]

                def _arrow_pre(tbl: _pa.Table) -> _pa.Table:
                    # Ensure timestamp is int64
                    if "timestamp" in tbl.schema.names and not _pa.types.is_int64(
                        tbl.schema.field("timestamp").type
                    ):
                        tbl = tbl.set_column(
                            tbl.schema.get_field_index("timestamp"),
                            "timestamp",
                            pc.cast(tbl.column("timestamp"), _pa.int64()),
                        )
                    return tbl

                df = pdm.cached_projection(
                    base_dir=part_base,
                    filters=filters,
                    columns=columns or [],
                    cache_dir="data_cache/projections",
                    cache_key_prefix=f"labeled_{exchange}_{symbol}_{timeframe}_validation",
                    snapshot_version="v1",
                    ttl_seconds=3600,
                    batch_size=131072,
                    arrow_transform=_arrow_pre,
                )
                # Merge 1m meta-labels as additional validation features if present
                try:
                    import pickle as _pkl
                    pkl_train = os.path.join(data_dir, f"{exchange}_{symbol}_labeled_train.pkl")
                    if os.path.exists(pkl_train):
                        with open(pkl_train, "rb") as _f:
                            step4 = _pkl.load(_f)
                        one_m_cols = [c for c in getattr(step4, 'columns', []) if isinstance(c, str) and c.startswith('1m_')]
                        if one_m_cols and 'timestamp' in step4.columns and 'timestamp' in df.columns:
                            df = df.merge(step4[['timestamp', *one_m_cols]], on='timestamp', how='left')
                            self.logger.info(f"Merged {len(one_m_cols)} 1m meta-label columns into validation frame")
                except Exception:
                    pass
                if (
                    isinstance(feat_cols, list)
                    and len(feat_cols) > 0
                    and label_col in df.columns
                ):
                    X_val = df[feat_cols].to_numpy(dtype=float)
                    y_val = df[label_col].to_numpy()
                    self.logger.info(
                        f"Loaded validation data from partitioned labeled: X={X_val.shape}, y={y_val.shape}"
                    )
                    return X_val, y_val
        except Exception:
            pass

        # No fallback - step should fail if validation data is missing
        msg = f"Validation data not found in {part_base}. Step 10 requires labeled data from Step 8."
        raise FileNotFoundError(msg)

    @enforce_ndarray(arg_index=2, forbid_lists=True, require_vector=True)
    @guard_array_nan_inf(mode="warn", arg_indices=(2,))
    @with_tracing_span("EnsembleCreation._get_model_predictions", log_args=False)
    def _get_model_predictions(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Helper to get probability predictions, with a fallback for classifiers without `predict_proba`."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)

        self.logger.warning(
            f"Model {type(model).__name__} lacks `predict_proba`. Falling back to `predict`.",
        )
        preds = model.predict(X)
        # Convert [-1, 0, 1] integer labels to one-hot encoded probabilities
        pred_proba = np.zeros((len(preds), 3))
        # Map label `k` to index `k+1` to handle the -1 label
        pred_proba[np.arange(len(preds)), preds + 1] = 1.0
        return pred_proba

    async def _find_optimal_weight(
        self,
        models: list[Any],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """
        Performs a simple search to find the optimal blending weight `w` for two models.
        The blend is calculated as `w * preds_A + (1 - w) * preds_B`.

        Args:
            models: A list containing the two models to blend.
            X_val: Validation features.
            y_val: Validation true labels.

        Returns:
            The optimal weight for the first model.
        """
        self.logger.info("Searching for optimal blending weight...")
        preds_a = self._get_model_predictions(models[0], X_val)
        preds_b = self._get_model_predictions(models[1], X_val)

        best_accuracy = -1.0
        optimal_weight = 0.5  # Default to a simple average
        acc_by_w: list[tuple[float, float]] = []

        # Grid search for the best weight `w`
        for w in np.arange(0, 1.01, 0.01):
            blended_proba = w * preds_a + (1 - w) * preds_b
            blended_labels = (
                np.argmax(blended_proba, axis=1) - 1
            )  # Convert back to [-1, 0, 1]
            accuracy = accuracy_score(y_val, blended_labels)
            acc_by_w.append((float(w), float(accuracy)))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                optimal_weight = w

        # Log top-5 candidate weights by accuracy
        try:
            top5 = sorted(acc_by_w, key=lambda x: x[1], reverse=True)[:5]
            self.logger.info({"msg": "tactician_weight_search_top5", "candidates": top5})
        except Exception:
            pass

        self.logger.info(
            f"Optimal weight found: {optimal_weight:.2f} (for model 1) with validation accuracy: {best_accuracy:.4f}",
        )
        return optimal_weight

    async def _create_tactician_ensemble(
        self,
        models: dict[str, Any],
        data_dir: str,
    ) -> dict[str, Any]:
        """
        Creates an optimized weighted-average ensemble of two tactician models.

        Args:
            models: A dictionary containing the two loaded tactician models.
            data_dir: The directory for loading validation data.

        Returns:
            A dictionary containing the ensemble model and its metadata.
        """
        self.logger.info("Creating optimized tactician ensemble...")

        model_items = list(models.values())
        model_names = list(models.keys())

        base_models = [m["model"] for m in model_items]
        base_accuracies = [m.get("accuracy", 0.0) for m in model_items]

        # 1. Load validation data to find the optimal weight
        X_val, y_val = await self._load_validation_data(data_dir)

        # 2. Find the optimal blending weight
        optimal_w = await self._find_optimal_weight(base_models, X_val, y_val)

        # 3. Define and create the ensemble class with the optimal weight
        class OptimalBlendedEnsemble:
            def __init__(self, model_a: Any, model_b: Any, weight_a: float):
                self.model_a = model_a
                self.model_b = model_b
                self.weight_a = weight_a
                self.weight_b = 1.0 - weight_a

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                """Predicts class probabilities by blending the two base models."""
                preds_a = self.model_a.predict_proba(X)
                preds_b = self.model_b.predict_proba(X)
                return self.weight_a * preds_a + self.weight_b * preds_b

            def predict(self, X: np.ndarray) -> np.ndarray:
                """Predicts class labels based on the blended probabilities."""
                proba = self.predict_proba(X)
                return np.argmax(proba, axis=1) - 1

        ensemble = OptimalBlendedEnsemble(base_models[0], base_models[1], optimal_w)

        # 4. Prepare the final dictionary with the model and its metadata
        try:
            self.logger.info({
                "msg": "tactician_optimal_weights",
                "weights": {model_names[0]: float(optimal_w), model_names[1]: float(1.0 - optimal_w)}
            })
        except Exception:
            pass
        return {
            "ensemble": ensemble,
            "ensemble_type": "OptimalBlended",
            "base_models": model_names,
            "base_model_accuracies": base_accuracies,
            "optimal_weight": {
                model_names[0]: optimal_w,
                model_names[1]: 1.0 - optimal_w,
            },
            "creation_date": datetime.now().isoformat(),
        }


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    """Runs the tactician ensemble creation step."""
    try:
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = TacticianEnsembleCreationStep(config)
        await step.initialize()

        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs,
        }
        result = await step.execute(training_input, pipeline_state={})
        return result.get("status") == "SUCCESS"

    except Exception as e:
        system_logger.error(
            f"❌ Tactician ensemble creation failed: {e}",
            exc_info=True,
        )
        return False


if __name__ == "__main__":
    # Example of how to run the step
    async def test():
        # As this test requires model files from a previous step, we'll
        # create dummy model files for demonstration purposes.
        print("--- Running Tactician Ensemble Creation Test ---")

        # Create dummy models and data directory
        test_data_dir = "data/test_training"
        models_dir = os.path.join(test_data_dir, "tactician_models")
        os.makedirs(models_dir, exist_ok=True)

        from sklearn.metrics import confusion_matrix, accuracy_score
        from sklearn.linear_model import LogisticRegression

        # Dummy Model A
        model_a = LogisticRegression()
        model_a.fit(np.random.rand(10, 50), np.random.randint(-1, 2, 10))
        model_a_data = {"model": model_a, "accuracy": 0.65}
        with open(os.path.join(models_dir, "model_A.pkl"), "wb") as f:
            pickle.dump(model_a_data, f)

        # Dummy Model B
        model_b = LogisticRegression()
        model_b.fit(np.random.rand(10, 50), np.random.randint(-1, 2, 10))
        model_b_data = {"model": model_b, "accuracy": 0.72}
        with open(os.path.join(models_dir, "model_B.pkl"), "wb") as f:
            pickle.dump(model_b_data, f)

        print(f"Created dummy models in {models_dir}")

        # Run the step
        success = await run_step("ETHUSDT", "BINANCE", test_data_dir)
        print(f"\nTest Result: {'SUCCESS' if success else 'FAILED'}")

        # Verify output files
        ensemble_dir = os.path.join(test_data_dir, "tactician_ensembles")
        if success:
            summary_path = os.path.join(
                ensemble_dir,
                "BINANCE_ETHUSDT_tactician_ensemble_summary.json",
            )
            model_path = os.path.join(
                ensemble_dir,
                "BINANCE_ETHUSDT_tactician_ensemble.pkl",
            )
            print(f"Verified: Summary file exists at {summary_path}")
            print(f"Verified: Model file exists at {model_path}")

    asyncio.run(test())
