"""
Step dependency validator for the training pipeline.
Ensures that steps don't proceed if their prerequisites have failed.
"""

import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging

try:
    from .logger import system_logger
    from .pipeline_standards import PipelineStandards, pipeline_standards
    from .warning_symbols import error, warning, critical
except ImportError:
    # Fallback for when running as standalone
    from src.utils.logger import system_logger
    from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
    from src.utils.warning_symbols import error, warning, critical


class StepDependencyValidator:
    """Validates step dependencies to ensure pipeline integrity.
    Prevents steps from running if their prerequisites have failed.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the step dependency validator."""
        self.config = config or {}
        self.logger = system_logger.getChild("StepDependencyValidator")
        self.is_initialized = False

        # Define step dependencies (step -> list of required steps)
        self.step_dependencies = {
            "step01_data_collection": [],
            "step01_5_data_converter": ["step01_data_collection"],
            "step02_data_reading": ["step01_5_data_converter"],
            "step02_5_sr_optimization": ["step02_data_reading"],
            "step03_hmm_regime_discovery": ["step02_5_sr_optimization"],
            "step04_triple_barrier_method": ["step03_hmm_regime_discovery"],
            "step04_regime_data_splitting": ["step04_triple_barrier_method"],
            "step05_labeling": ["step04_triple_barrier_method"],
            "step06_feature_engineering": ["step05_labeling"],
            "step07_enhanced_matrix_operations": ["step06_feature_engineering"],
            "step08_regime_data_splitting": ["step07_enhanced_matrix_operations"],
            "step09_hmm_based_training": ["step08_regime_data_splitting"],
            "step09_5_multi_timeframe_hmm_ensemble": ["step09_hmm_based_training"],
            "step09_5_hmm_lm_generalist_training": ["step09_5_multi_timeframe_hmm_ensemble"],
            "step10_unified_regime_intelligence": ["step09_5_hmm_lm_generalist_training"],
            "step11_analyst_creation": ["step10_unified_regime_intelligence"],
            "step12_analyst_enhancement": ["step11_analyst_creation"],
            "step13_analyst_ensemble_creation": ["step12_analyst_enhancement"],
            "step14_tactician_labeling": ["step13_analyst_ensemble_creation"],
            "step15_tactician_specialist_training": ["step14_tactician_labeling"],
            "step16_confidence_calibration": ["step15_tactician_specialist_training"],
            "step17_final_parameters_optimization": ["step16_confidence_calibration"],
            "step18_walk_forward_validation": ["step17_final_parameters_optimization"],
            "step19_monte_carlo_validation": ["step18_walk_forward_validation"],
            "step20_ab_testing": ["step19_monte_carlo_validation"],
            "step21_saving": ["step20_ab_testing"],
        }

        # Define critical data requirements for each step
        self.critical_data_requirements = {
            "step01_data_collection": {
                "required_files": ["data_cache/klines_*_*_1m_consolidated.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500
            },
            "step01_5_data_converter": {
                "required_files": ["data_cache/unified/*/*/*/*.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500
            },
            "step02_data_reading": {
                "required_files": ["data_cache/unified/*/*/*/*.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500
            },
            "step02_5_sr_optimization": {
                "required_files": ["data_cache/unified/*/*/*/*.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500
            },
            "step03_hmm_regime_discovery": {
                "required_files": ["data/hmm_regimes/*_composite_clusters.parquet"],
                "required_columns": ["composite_cluster_id"],
                "min_rows": 100
            },
            "step04_triple_barrier_method": {
                "required_files": ["data/training/*_triple_barrier_*.parquet"],
                "required_columns": ["triple_barrier_label"],
                "min_rows": 50
            },
            "step05_labeling": {
                "required_files": ["data/training/*_labeled_*.parquet"],
                "required_columns": ["label"],
                "min_rows": 50
            },
            "step06_feature_engineering": {
                "required_files": ["data/training/*_features_train.parquet", "data/training/*_features_val.parquet"],
                "required_columns": ["timestamp", "returns", "volatility"],
                "min_rows": 1000
            },
            "step07_enhanced_matrix_operations": {
                "required_files": ["data/matrix_operations/*_matrix_operations_*.json"],
                "required_columns": [],
                "min_rows": 0
            },
            "step08_hmm_based_training": {
                "required_files": ["data/training/*_hmm_models.pkl"],
                "required_columns": [],
                "min_rows": 0
            },
            "step08_5_unified_regime_intelligence": {
                "required_files": ["data/training/*_unified_intelligence.parquet"],
                "required_columns": ["intelligence_score"],
                "min_rows": 100
            },
            "step09_analyst_enhancement": {
                "required_files": ["data/training/*_analyst_models.pkl"],
                "required_columns": [],
                "min_rows": 0
            },
            "step10_tactician_labeling": {
                "required_files": ["data/training/*_tactician_labels.parquet"],
                "required_columns": ["tactician_label"],
                "min_rows": 100
            },
            "step11_tactician_specialist_training": {
                "required_files": ["data/training/*_specialist_models.pkl"],
                "required_columns": [],
                "min_rows": 0
            },
            "step12_confidence_calibration": {
                "required_files": ["data/training/*_calibration_results.pkl"],
                "required_columns": [],
                "min_rows": 0
            },
            "step13_final_parameters_optimization": {
                "required_files": ["data/training/*_optimization_results.json"],
                "required_columns": [],
                "min_rows": 0
            },
            "step14_walk_forward_validation": {
                "required_files": ["data/training/*_walk_forward_results.json"],
                "required_columns": [],
                "min_rows": 0
            },
            "step15_monte_carlo_validation": {
                "required_files": ["data/training/*_monte_carlo_results.json"],
                "required_columns": [],
                "min_rows": 0
            },
            "step14_ab_testing": {
                "required_files": ["data/training/*_ab_test_results.json"],
                "required_columns": [],
                "min_rows": 0
            },
            "step15_saving": {
                "required_files": ["data/training/*_final_models.pkl"],
                "required_columns": [],
                "min_rows": 0
            },
            "step16_confidence_calibration": {
                "required_files": ["data/training/*_extended_calibration_results.pkl"],
                "required_columns": [],
                "min_rows": 0
            },
            "step17_final_parameters_optimization": {
                "required_files": ["data/training/*_extended_optimization_results.json"],
                "required_columns": [],
                "min_rows": 0
            },
            "step18_walk_forward_validation": {
                "required_files": ["data/training/*_extended_walk_forward_results.json"],
                "required_columns": [],
                "min_rows": 0
            },
            "step19_monte_carlo_validation": {
                "required_files": ["data/training/*_extended_monte_carlo_results.json"],
                "required_columns": [],
                "min_rows": 0
            },
            "step20_ab_testing": {
                "required_files": ["data/training/*_extended_ab_test_results.json"],
                "required_columns": [],
                "min_rows": 0
            },
            "step21_saving": {
                "required_files": ["data/training/*_extended_final_models.pkl"],
                "required_columns": [],
                "min_rows": 0
            }
        }

    async def initialize(self) -> bool:
        """Initialize StepDependencyValidator."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False

    async def validate_step_prerequisites(
        self, 
        step_name: str, 
        checkpoint_dir: str, 
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """Validate prerequisites for a given step."""
        try:
            self.logger.info(f"🔍 Validating prerequisites for {step_name}")

            # Check if step has dependencies
            if step_name not in self.step_dependencies:
                self.logger.info(f"✅ {step_name} has no dependencies")
                return {"valid": True, "reason": "No dependencies"}

            required_steps = self.step_dependencies[step_name]
            self.logger.info(f"📋 {step_name} requires: {required_steps}")

            # If force_rerun is True, we're starting from this step, so skip dependency validation
            if force_rerun:
                self.logger.info(f"✅ Force rerun enabled for {step_name}, skipping dependency validation")
                return {"valid": True, "reason": "Force rerun enabled"}

            # If no dependencies, validation passes
            if not required_steps:
                self.logger.info(f"✅ {step_name} has no dependencies")
                return {"valid": True, "reason": "No dependencies"}

            # Check each required step
            failed_prerequisites = []
            for required_step in required_steps:
                if not await self._check_step_completion(required_step, checkpoint_dir):
                    failed_prerequisites.append(required_step)

            if failed_prerequisites:
                error_msg = f"❌ Prerequisites failed for {step_name}: {failed_prerequisites}"
                self.logger.error(error_msg)
                return {
                    "valid": False,
                    "reason": f"Failed prerequisites: {failed_prerequisites}",
                    "failed_steps": failed_prerequisites
                }

            # Check critical data requirements
            if step_name in self.critical_data_requirements:
                data_validation = await self._validate_critical_data(
                    step_name,
                    self.critical_data_requirements[step_name]
                )
                if not data_validation["valid"]:
                    return data_validation

            self.logger.info(f"✅ All prerequisites met for {step_name}")
            return {"valid": True, "reason": "All prerequisites met"}

        except Exception as e:
            self.logger.error(f"🚨 Error validating prerequisites for {step_name}: {e}")
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}"
            }

    async def _check_step_completion(self, step_name: str, checkpoint_dir: str) -> bool:
        """Check if a step has been completed successfully."""
        try:
            # First, try the individual step checkpoint file
            checkpoint_file = Path(checkpoint_dir) / f"{step_name}.json"

            if checkpoint_file.exists():
                # Read checkpoint
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)

                # Check if step completed successfully
                status = checkpoint_data.get("status", "unknown")
                if status != "completed":
                    self.logger.warning(f"⚠️ {step_name} status: {status} (not completed)")
                    return False

                # Check for any errors in the checkpoint
                errors = checkpoint_data.get("errors", [])
                if errors:
                    self.logger.warning(f"⚠️ {step_name} has errors: {errors}")
                    return False

                self.logger.debug(f"✅ {step_name} completed successfully")
                return True

            # If individual checkpoint not found, try the centralized training progress file
            # Extract exchange, symbol, and timeframe from checkpoint_dir path
            path_parts = Path(checkpoint_dir).parts
            if len(path_parts) >= 4 and path_parts[0] == "checkpoints":
                # Path structure: checkpoints/exchange/symbol/timeframe
                exchange = path_parts[1]  # e.g., "BINANCE"
                symbol = path_parts[2]    # e.g., "ETHUSDT"
                timeframe = path_parts[3] # e.g., "1m"

                # Try the centralized training progress file - match the path structure used by enhanced training manager
                centralized_checkpoint = Path("checkpoints") / exchange / symbol / timeframe / "training_progress.json"

                if centralized_checkpoint.exists():
                    try:
                        with open(centralized_checkpoint, 'r') as f:
                            progress_data = json.load(f)

                        # Check if the step is marked as completed in the centralized file
                        pipeline_state = progress_data.get("pipeline_state", {})

                        # Map step names to their status keys in the centralized file
                        step_status_mapping = {
                            "step01_data_collection": "data_collection",
                            "step02_feature_engineering": "feature_engineering",
                            "step02_5_sr_optimization": "sr_optimization",
                            "step03_hmm_regime_discovery": "hmm_regime_discovery",
                            "step04_processing_labeling": "processing_labeling",
                            "step05_regime_data_splitting": "regime_data_splitting",
                            "step06_hmm_based_training": "hmm_based_training",
                            "step06_5_unified_regime_intelligence": "unified_regime_intelligence",
                            "step07_analyst_enhancement": "analyst_enhancement",
                            "step08_tactician_labeling": "tactician_labeling",
                            "step09_tactician_specialist_training": "tactician_specialist_training",
                            "step10_confidence_calibration": "confidence_calibration",
                            "step11_final_parameters_optimization": "final_parameters_optimization",
                            "step12_walk_forward_validation": "walk_forward_validation",
                            "step13_monte_carlo_validation": "monte_carlo_validation",
                            "step14_ab_testing": "ab_testing",
                            "step15_saving": "saving",
                        }

                        if step_name in step_status_mapping:
                            status_key = step_status_mapping[step_name]
                            step_status = pipeline_state.get(status_key, {})

                            if step_status.get("status") == "SUCCESS" or step_status.get("completed", False):
                                self.logger.debug(f"✅ {step_name} completed successfully (from centralized progress)")
                                return True
                            elif step_status.get("status") == "SKIPPED":
                                self.logger.debug(f"✅ {step_name} was skipped (from centralized progress)")
                                return True
                            else:
                                self.logger.debug(f"⚠️ {step_name} status: {step_status.get('status', 'unknown')} (from centralized progress)")
                        else:
                            self.logger.debug(f"⚠️ No status mapping found for {step_name} in centralized progress")

                    except Exception as e:
                        self.logger.warning(f"⚠️ Error reading centralized checkpoint {centralized_checkpoint}: {e}")

                # Also try alternative paths for different timeframes
                alternative_paths = [
                    Path("checkpoints") / exchange / symbol / timeframe / "training_progress.json",
                    Path("checkpoints") / exchange / symbol / "training_progress.json",
                    Path("checkpoints") / exchange / symbol / timeframe / "progress.json",
                    Path("checkpoints") / exchange / symbol / "progress.json",
                ]

                for alt_checkpoint in alternative_paths:
                    if alt_checkpoint.exists():
                        try:
                            with open(alt_checkpoint, 'r') as f:
                                progress_data = json.load(f)

                            # Check if the step is marked as completed in the centralized file
                            pipeline_state = progress_data.get("pipeline_state", {})

                            # Map step names to their status keys in the centralized file
                            step_status_mapping = {
                                "step01_data_collection": "data_collection",
                                "step02_feature_engineering": "feature_engineering",
                                "step02_5_sr_optimization": "sr_optimization",
                                "step03_hmm_regime_discovery": "hmm_regime_discovery",
                                "step04_processing_labeling": "processing_labeling",
                                "step05_regime_data_splitting": "regime_data_splitting",
                                "step06_hmm_based_training": "hmm_based_training",
                                "step06_5_unified_regime_intelligence": "unified_regime_intelligence",
                                "step07_analyst_enhancement": "analyst_enhancement",
                                "step08_tactician_labeling": "tactician_labeling",
                                "step09_tactician_specialist_training": "tactician_specialist_training",
                                "step10_confidence_calibration": "confidence_calibration",
                                "step11_final_parameters_optimization": "final_parameters_optimization",
                                "step12_walk_forward_validation": "walk_forward_validation",
                                "step13_monte_carlo_validation": "monte_carlo_validation",
                                "step14_ab_testing": "ab_testing",
                                "step15_saving": "saving",
                            }

                            if step_name in step_status_mapping:
                                status_key = step_status_mapping[step_name]
                                step_status = pipeline_state.get(status_key, {})

                                if step_status.get("status") == "SUCCESS" or step_status.get("completed", False):
                                    self.logger.debug(f"✅ {step_name} completed successfully (from alternative path)")
                                    return True
                                elif step_status.get("status") == "SKIPPED":
                                    self.logger.debug(f"✅ {step_name} was skipped (from alternative path)")
                                    return True

                        except Exception as e:
                            self.logger.warning(f"⚠️ Error reading alternative checkpoint {alt_checkpoint}: {e}")

            # If we get here, the step hasn't been completed
            self.logger.debug(f"❌ {step_name} not found in any checkpoint files")
            return False

        except Exception as e:
            self.logger.error(f"Error checking step completion for {step_name}: {e}")
            return False

    async def _validate_critical_data(
        self, 
        step_name: str, 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate critical data requirements for a step."""
        try:
            self.logger.info(f"🔍 Validating critical data for {step_name}")

            # Check required files
            required_files = requirements.get("required_files", [])
            if required_files:
                file_validation = await self._check_required_files(required_files)
                if not file_validation["valid"]:
                    return file_validation

            # Check required columns
            required_columns = requirements.get("required_columns", [])
            if required_columns:
                column_validation = await self._check_required_columns(required_columns)
                if not column_validation["valid"]:
                    return column_validation

            # Check minimum rows
            min_rows = requirements.get("min_rows", 0)
            if min_rows > 0:
                row_validation = await self._check_minimum_rows(min_rows)
                if not row_validation["valid"]:
                    return row_validation

            return {"valid": True, "reason": "All critical data requirements met"}

        except Exception as e:
            self.logger.error(f"Error validating critical data for {step_name}: {e}")
            return {
                "valid": False,
                "reason": f"Data validation error: {str(e)}"
            }

    async def _check_required_files(self, required_files: List[str]) -> Dict[str, Any]:
        """Check if required files exist."""
        try:
            import glob
            from pathlib import Path
            
            missing_files = []
            existing_files = []
            
            for file_pattern in required_files:
                # Handle glob patterns
                if "*" in file_pattern:
                    matching_files = glob.glob(file_pattern)
                    if not matching_files:
                        missing_files.append(file_pattern)
                    else:
                        existing_files.extend(matching_files)
                else:
                    # Single file
                    if Path(file_pattern).exists():
                        existing_files.append(file_pattern)
                    else:
                        missing_files.append(file_pattern)
            
            if missing_files:
                return {
                    "valid": False,
                    "reason": f"Missing required files: {missing_files}",
                    "missing_files": missing_files,
                    "existing_files": existing_files
                }
            
            return {
                "valid": True,
                "reason": f"All required files found: {len(existing_files)} files",
                "existing_files": existing_files
            }
            
        except Exception as e:
            self.logger.error(f"Error checking required files: {e}")
            return {
                "valid": False,
                "reason": f"Error checking files: {str(e)}"
            }

    async def _check_required_columns(self, required_columns: List[str]) -> Dict[str, Any]:
        """Check if required columns exist in the data."""
        try:
            import pandas as pd
            import glob
            
            if not required_columns:
                return {"valid": True, "reason": "No columns required"}
            
            # Find data files to check
            data_files = glob.glob("data/**/*.parquet") + glob.glob("data_cache/**/*.parquet")
            
            if not data_files:
                return {
                    "valid": False,
                    "reason": "No data files found to check columns"
                }
            
            # Check first few files for column existence
            checked_files = 0
            missing_columns = set(required_columns)
            
            for file_path in data_files[:5]:  # Check first 5 files
                try:
                    df = pd.read_parquet(file_path, nrows=1)  # Just read header
                    file_missing = [col for col in required_columns if col not in df.columns]
                    missing_columns = missing_columns.intersection(set(file_missing))
                    checked_files += 1
                    
                    if not missing_columns:
                        break  # All columns found
                        
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")
                    continue
            
            if missing_columns:
                return {
                    "valid": False,
                    "reason": f"Missing required columns: {list(missing_columns)}",
                    "missing_columns": list(missing_columns),
                    "checked_files": checked_files
                }
            
            return {
                "valid": True,
                "reason": f"All required columns found in {checked_files} files",
                "checked_files": checked_files
            }
            
        except Exception as e:
            self.logger.error(f"Error checking required columns: {e}")
            return {
                "valid": False,
                "reason": f"Error checking columns: {str(e)}"
            }

    async def _check_minimum_rows(self, min_rows: int) -> Dict[str, Any]:
        """Check if data has minimum required rows."""
        try:
            import pandas as pd
            import glob
            
            if min_rows <= 0:
                return {"valid": True, "reason": "No minimum row requirement"}
            
            # Find data files to check
            data_files = glob.glob("data/**/*.parquet") + glob.glob("data_cache/**/*.parquet")
            
            if not data_files:
                return {
                    "valid": False,
                    "reason": "No data files found to check row count"
                }
            
            # Check first few files for row count
            checked_files = 0
            total_rows = 0
            
            for file_path in data_files[:3]:  # Check first 3 files
                try:
                    df = pd.read_parquet(file_path)
                    file_rows = len(df)
                    total_rows += file_rows
                    checked_files += 1
                    
                    if total_rows >= min_rows:
                        break  # Enough rows found
                        
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")
                    continue
            
            if total_rows < min_rows:
                return {
                    "valid": False,
                    "reason": f"Insufficient rows: {total_rows} < {min_rows}",
                    "total_rows": total_rows,
                    "required_rows": min_rows,
                    "checked_files": checked_files
                }
            
            return {
                "valid": True,
                "reason": f"Sufficient rows found: {total_rows} >= {min_rows}",
                "total_rows": total_rows,
                "required_rows": min_rows,
                "checked_files": checked_files
            }
            
        except Exception as e:
            self.logger.error(f"Error checking minimum rows: {e}")
            return {
                "valid": False,
                "reason": f"Error checking row count: {str(e)}"
            }

    def get_step_dependencies(self, step_name: str) -> List[str]:
        """Get the list of dependencies for a given step."""
        return self.step_dependencies.get(step_name, [])

    def get_all_steps(self) -> List[str]:
        """Get all available steps."""
        return list(self.step_dependencies.keys())

    def get_step_order(self) -> List[str]:
        """Get steps in dependency order (topological sort)."""
        try:
            from collections import defaultdict, deque
            
            # Build adjacency list and in-degree count
            graph = defaultdict(list)
            in_degree = defaultdict(int)
            
            # Initialize in-degree for all steps
            for step in self.step_dependencies:
                in_degree[step] = 0
            
            # Build graph and calculate in-degrees
            for step, dependencies in self.step_dependencies.items():
                for dep in dependencies:
                    graph[dep].append(step)
                    in_degree[step] += 1
            
            # Topological sort using Kahn's algorithm
            queue = deque([step for step, degree in in_degree.items() if degree == 0])
            result = []
            
            while queue:
                current = queue.popleft()
                result.append(current)
                
                # Reduce in-degree for all neighbors
                for neighbor in graph[current]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            # Check for cycles
            if len(result) != len(self.step_dependencies):
                self.logger.warning("⚠️ Circular dependency detected in step dependencies")
                # Return steps with dependencies first, then independent steps
                dependent_steps = [step for step in self.step_dependencies if self.step_dependencies[step]]
                independent_steps = [step for step in self.step_dependencies if not self.step_dependencies[step]]
                return dependent_steps + independent_steps
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in topological sort: {e}")
            # Fallback to simple ordering
            return list(self.step_dependencies.keys())

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up StepDependencyValidator")
        self.is_initialized = False


# Global instance
step_dependency_validator: Optional[StepDependencyValidator] = None


async def setup_step_dependency_validator(config: Optional[Dict[str, Any]] = None) -> Optional[StepDependencyValidator]:
    """Set up the global step dependency validator."""
    try:
        global step_dependency_validator
        step_dependency_validator = StepDependencyValidator(config)
        success = await step_dependency_validator.initialize()
        if success:
            return step_dependency_validator
        return None
    except Exception as e:
        system_logger.error(f"Error setting up step dependency validator: {e}")
        return None
