"""
Step dependency validator for the training pipeline.
Ensures that steps don't proceed if their prerequisites have failed.
"""

import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.warning_symbols import error, warning, critical


class StepDependencyValidator:
    """
    Validates step dependencies to ensure pipeline integrity.
    Prevents steps from running if their prerequisites have failed.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild("StepDependencyValidator")
        
        # Define step dependencies (step -> list of required steps)
        self.step_dependencies = {
            "step1_data_collection": [],
            "step1_5_data_converter": ["step1_data_collection"],
            "step2_data_reading": ["step1_5_data_converter"],
            "step2_5_sr_optimization": ["step2_data_reading"],
            "step3_hmm_regime_discovery": ["step2_5_sr_optimization"],
            "step4_triple_barrier_method": ["step3_hmm_regime_discovery"],
            "step4_regime_data_splitting": ["step4_triple_barrier_method"],
            "step5_labeling": ["step4_triple_barrier_method"],
            "step6_feature_engineering": ["step5_labeling"],
            "step7_enhanced_matrix_operations": ["step6_feature_engineering"],
            "step8_regime_data_splitting": ["step7_enhanced_matrix_operations"],
            "step9_hmm_based_training": ["step8_regime_data_splitting"],
            "step9_5_multi_timeframe_hmm_ensemble": ["step9_hmm_based_training"],
            "step9_5_hmm_lm_generalist_training": ["step9_5_multi_timeframe_hmm_ensemble"],
            "step10_unified_regime_intelligence": ["step9_5_hmm_lm_generalist_training"],
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
            "step1_data_collection": {
                "required_files": ["data_cache/klines_*_*_1m_consolidated.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500
            },
            "step1_5_data_converter": {
                "required_files": ["data_cache/unified/*/*/*/*.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500
            },
            "step2_data_reading": {
                "required_files": ["data_cache/unified/*/*/*/*.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500
            },
            "step2_5_sr_optimization": {
                "required_files": ["data_cache/unified/*/*/*/*.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500
            },
            "step3_hmm_regime_discovery": {
                "required_files": ["data/hmm_regimes/*_composite_clusters.parquet"],
                "required_columns": ["composite_cluster_id"],
                "min_rows": 100
            },
            "step4_triple_barrier_method": {
                "required_files": ["data/training/*_triple_barrier_*.parquet"],
                "required_columns": ["triple_barrier_label"],
                "min_rows": 50
            },
            "step5_labeling": {
                "required_files": ["data/training/*_labeled_*.parquet"],
                "required_columns": ["label"],
                "min_rows": 50
            },
            "step6_feature_engineering": {
                "required_files": ["data/training/*_features_train.parquet", "data/training/*_features_val.parquet"],
                "required_columns": ["timestamp", "returns", "volatility"],
                "min_rows": 1000
            },
            "step7_enhanced_matrix_operations": {
                "required_files": ["data/matrix_operations/*_matrix_operations_*.json"],
                "required_columns": [],
                "min_rows": 0
            },
            "step8_hmm_based_training": {
                "required_files": ["data/training/*_hmm_models.pkl"],
                "required_columns": [],
                "min_rows": 0
            },
            "step8_5_unified_regime_intelligence": {
                "required_files": ["data/training/*_unified_intelligence.parquet"],
                "required_columns": ["intelligence_score"],
                "min_rows": 100
            },
            "step9_analyst_enhancement": {
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
            }
        }
    
    async def validate_step_prerequisites(
        self, 
        step_name: str, 
        pipeline_state: Dict[str, Any],
        checkpoint_dir: str = "checkpoints",
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """
        Validate that all prerequisites for a step are met.
        
        Args:
            step_name: Name of the step to validate
            pipeline_state: Current pipeline state
            checkpoint_dir: Directory containing checkpoints
            force_rerun: If True, skip dependency validation for the starting step
            
        Returns:
            Dictionary with validation results
        """
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
        """
        Check if a step has completed successfully.
        
        Args:
            step_name: Name of the step to check
            checkpoint_dir: Directory containing checkpoints
            
        Returns:
            True if step completed successfully, False otherwise
        """
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
                                "step1_data_collection": "data_collection",
                                "step2_feature_engineering": "feature_engineering",
                                "step2_5_sr_optimization": "sr_optimization",
                                "step3_hmm_regime_discovery": "hmm_regime_discovery",
                                "step4_processing_labeling": "processing_labeling",
                                "step05_regime_data_splitting": "regime_data_splitting",
                                "step06_hmm_based_training": "hmm_based_training",
                                "step6_5_unified_regime_intelligence": "unified_regime_intelligence",
                                "step7_analyst_enhancement": "analyst_enhancement",
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
                        
                        # If we found the centralized file but step wasn't found, try alternative paths
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
                                "step1_data_collection": "data_collection",
                                "step2_feature_engineering": "feature_engineering",
                                "step2_5_sr_optimization": "sr_optimization",
                                "step3_hmm_regime_discovery": "hmm_regime_discovery",
                                "step4_processing_labeling": "processing_labeling",
                                "step05_regime_data_splitting": "regime_data_splitting",
                                "step06_hmm_based_training": "hmm_based_training",
                                "step6_5_unified_regime_intelligence": "unified_regime_intelligence",
                                "step7_analyst_enhancement": "analyst_enhancement",
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
                            
                            # If we found the centralized file but step wasn't found, try next path
                        except Exception as e:
                            self.logger.warning(f"⚠️ Error reading centralized checkpoint {alt_checkpoint}: {e}")
            
            self.logger.warning(f"⚠️ No checkpoint found for {step_name}")
            return False
            
        except Exception as e:
            self.logger.error(f"🚨 Error checking completion for {step_name}: {e}")
            return False
    
    async def _validate_critical_data(
        self, 
        step_name: str, 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate critical data requirements for a step.
        
        Args:
            step_name: Name of the step
            requirements: Data requirements dictionary
            
        Returns:
            Validation result dictionary
        """
        try:
            self.logger.info(f"🔍 Validating critical data for {step_name}")
            
            # Check required files
            if "required_files" in requirements:
                for file_pattern in requirements["required_files"]:
                    if not await self._check_file_pattern(file_pattern):
                        return {
                            "valid": False,
                            "reason": f"Missing required files: {file_pattern}"
                        }
            
            # Check required columns (if we have data to check)
            if "required_columns" in requirements:
                # This would need to be implemented based on actual data loading
                # For now, we'll assume it's valid if files exist
                pass
            
            # Check minimum rows (if we have data to check)
            if "min_rows" in requirements:
                # This would need to be implemented based on actual data loading
                # For now, we'll assume it's valid if files exist
                pass
            
            return {"valid": True, "reason": "Critical data requirements met"}
            
        except Exception as e:
            self.logger.error(f"🚨 Error validating critical data for {step_name}: {e}")
            return {
                "valid": False,
                "reason": f"Data validation error: {str(e)}"
            }
    
    async def _check_file_pattern(self, file_pattern: str) -> bool:
        """
        Check if files matching a pattern exist.
        
        Args:
            file_pattern: File pattern to check (supports glob patterns)
            
        Returns:
            True if files exist, False otherwise
        """
        try:
            from pathlib import Path
            import glob
            
            # Convert pattern to glob pattern
            if "*" not in file_pattern:
                # Single file
                return Path(file_pattern).exists()
            else:
                # Glob pattern
                files = glob.glob(file_pattern)
                return len(files) > 0
                
        except Exception as e:
            self.logger.error(f"🚨 Error checking file pattern {file_pattern}: {e}")
            return False
    
    def get_step_dependencies(self, step_name: str) -> List[str]:
        """
        Get the list of dependencies for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            List of required step names
        """
        return self.step_dependencies.get(step_name, [])
    
    async def validate_data_requirements(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = "data",
    ) -> Dict[str, Any]:
        """
        Validate that required data files exist for a step.
        
        Args:
            step_name: Name of the step to validate
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Dictionary with validation results
        """
        try:
            self.logger.info(f"🔍 Validating data requirements for {step_name}")
            
            if step_name not in self.critical_data_requirements:
                self.logger.info(f"✅ {step_name} has no specific data requirements")
                return {"valid": True, "reason": "No data requirements defined"}
            
            requirements = self.critical_data_requirements[step_name]
            validation_results = {
                "valid": True,
                "missing_files": [],
                "file_validation_results": {},
                "data_quality_issues": [],
            }
            
            # Check required files
            for file_pattern in requirements.get("required_files", []):
                # Replace placeholders in file pattern
                pattern = file_pattern.replace("{symbol}", symbol).replace("{exchange}", exchange).replace("{timeframe}", timeframe)
                
                # Use glob to find matching files
                import glob
                matching_files = glob.glob(os.path.join(data_dir, pattern))
                
                if not matching_files:
                    validation_results["missing_files"].append(pattern)
                    validation_results["valid"] = False
                else:
                    # Validate each matching file
                    for file_path in matching_files:
                        file_validation = await self._validate_data_file(
                            file_path, requirements.get("required_columns", []), requirements.get("min_rows", 0)
                        )
                        validation_results["file_validation_results"][file_path] = file_validation
                        
                        if not file_validation["valid"]:
                            validation_results["data_quality_issues"].append(f"Data quality issues in {file_path}")
                            validation_results["valid"] = False
            
            if validation_results["valid"]:
                self.logger.info(f"✅ Data requirements met for {step_name}")
            else:
                self.logger.warning(f"⚠️ Data requirements not met for {step_name}: {validation_results['missing_files']}")
            
            return validation_results
            
        except Exception as e:
            self.logger.exception(f"❌ Error validating data requirements for {step_name}: {e}")
            return {"valid": False, "error": str(e)}
    
    async def _validate_data_file(
        self,
        file_path: str,
        required_columns: List[str],
        min_rows: int,
    ) -> Dict[str, Any]:
        """
        Validate a single data file.
        
        Args:
            file_path: Path to the data file
            required_columns: List of required columns
            min_rows: Minimum number of rows required
            
        Returns:
            Validation result dictionary
        """
        try:
            import pandas as pd
            
            validation_result = {
                "valid": True,
                "file_path": file_path,
                "exists": os.path.exists(file_path),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                "missing_columns": [],
                "row_count": 0,
                "has_minimum_rows": False,
                "data_quality_issues": [],
            }
            
            if not validation_result["exists"]:
                validation_result["valid"] = False
                validation_result["data_quality_issues"].append("File does not exist")
                return validation_result
            
            # Try to read the file
            try:
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    validation_result["valid"] = False
                    validation_result["data_quality_issues"].append("Unsupported file format")
                    return validation_result
                
                validation_result["row_count"] = len(df)
                validation_result["has_minimum_rows"] = len(df) >= min_rows
                
                if not validation_result["has_minimum_rows"]:
                    validation_result["valid"] = False
                    validation_result["data_quality_issues"].append(f"Insufficient rows: {len(df)} < {min_rows}")
                
                # Check required columns
                if required_columns:
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    validation_result["missing_columns"] = missing_cols
                    
                    if missing_cols:
                        validation_result["valid"] = False
                        validation_result["data_quality_issues"].append(f"Missing required columns: {missing_cols}")
                
                # Check for null values in critical columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        null_count = df[col].isnull().sum()
                        if null_count > 0:
                            validation_result["data_quality_issues"].append(f"Null values in {col}: {null_count}")
                
            except Exception as e:
                validation_result["valid"] = False
                validation_result["data_quality_issues"].append(f"Error reading file: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "file_path": file_path,
            }
    
    async def validate_step_artifacts(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        artifact_dir: str = "data/training",
    ) -> Dict[str, Any]:
        """
        Validate that step artifacts exist and are valid.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            artifact_dir: Artifact directory
            
        Returns:
            Validation result dictionary
        """
        try:
            self.logger.info(f"🔍 Validating artifacts for {step_name}")
            
            # Define expected artifacts for each step
            expected_artifacts = {
                "step1_data_collection": [
                    f"data_cache/klines_{exchange}_{symbol}_1m_consolidated.parquet",
                ],
                "step1_5_data_converter": [
                    f"data_cache/unified/{exchange}/{symbol}/{timeframe}/**/*.parquet",
                ],
                "step2_feature_engineering": [
                    f"{artifact_dir}/{exchange}_{symbol}_features_train.parquet",
                    f"{artifact_dir}/{exchange}_{symbol}_features_metadata.json",
                ],
                "step2_5_sr_optimization": [
                    f"data/optimization/sr_optimization_results.json",
                    f"optimization_results.json",
                ],
                "step3_hmm_regime_discovery": [
                    f"data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.parquet",
                ],
                "step4_processing_labeling": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_labeled_validation.parquet",
                ],
                            "step05_regime_data_splitting": [
                f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet",
                f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_regime_labels.json",
            ],
                "step06_hmm_based_training": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_hmm_models.pkl",
                ],
                "step6_5_unified_regime_intelligence": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_unified_intelligence.parquet",
                ],
                "step7_analyst_enhancement": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_analyst_models.pkl",
                ],
                "step08_tactician_labeling": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_tactician_labels.parquet",
                ],
                "step09_tactician_specialist_training": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_specialist_models.pkl",
                ],
                "step10_confidence_calibration": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_calibration_results.pkl",
                ],
                "step11_final_parameters_optimization": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_optimization_results.json",
                ],
                "step12_walk_forward_validation": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_walk_forward_results.json",
                ],
                "step13_monte_carlo_validation": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_monte_carlo_results.json",
                ],
                "step14_ab_testing": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_ab_test_results.json",
                ],
                "step15_saving": [
                    f"{artifact_dir}/{exchange}_{symbol}_{timeframe}_final_models.pkl",
                ],
            }
            
            if step_name not in expected_artifacts:
                self.logger.info(f"✅ {step_name} has no specific artifacts")
                return {"valid": True, "reason": "No artifacts defined"}
            
            validation_results = {
                "valid": True,
                "missing_artifacts": [],
                "artifact_validation_results": {},
            }
            
            # Check each expected artifact
            for artifact_pattern in expected_artifacts[step_name]:
                import glob
                matching_artifacts = glob.glob(artifact_pattern)
                
                if not matching_artifacts:
                    validation_results["missing_artifacts"].append(artifact_pattern)
                    validation_results["valid"] = False
                else:
                    # Validate each matching artifact
                    for artifact_path in matching_artifacts:
                        artifact_validation = await self._validate_artifact_file(artifact_path)
                        validation_results["artifact_validation_results"][artifact_path] =artifact_validation
                        
                        if not artifact_validation["valid"]:
                            validation_results["valid"] = False
            
            if validation_results["valid"]:
                self.logger.info(f"✅ Artifacts validated for {step_name}")
            else:
                self.logger.warning(f"⚠️ Artifact validation failed for {step_name}: {validation_results['missing_artifacts']}")
            
            return validation_results
            
        except Exception as e:
            self.logger.exception(f"❌ Error validating artifacts for {step_name}: {e}")
            return {"valid": False, "error": str(e)}
    
    async def _validate_artifact_file(self, artifact_path: str) -> Dict[str, Any]:
        """
        Validate a single artifact file.
        
        Args:
            artifact_path: Path to the artifact file
            
        Returns:
            Validation result dictionary
        """
        try:
            validation_result = {
                "valid": True,
                "artifact_path": artifact_path,
                "exists": os.path.exists(artifact_path),
                "file_size": os.path.getsize(artifact_path) if os.path.exists(artifact_path) else 0,
                "validation_issues": [],
            }
            
            if not validation_result["exists"]:
                validation_result["valid"] = False
                validation_result["validation_issues"].append("Artifact does not exist")
                return validation_result
            
            # Check file size
            if validation_result["file_size"] == 0:
                validation_result["valid"] = False
                validation_result["validation_issues"].append("Artifact file is empty")
            
            # Validate based on file type
            if artifact_path.endswith('.pkl'):
                try:
                    import pickle
                    with open(artifact_path, 'rb') as f:
                        obj = pickle.load(f)
                    validation_result["object_type"] = type(obj).__name__
                except Exception as e:
                    validation_result["valid"] = False
                    validation_result["validation_issues"].append(f"Failed to load pickle: {str(e)}")
            
            elif artifact_path.endswith('.json'):
                try:
                    import json
                    with open(artifact_path, 'r') as f:
                        obj = json.load(f)
                    validation_result["object_type"] = type(obj).__name__
                except Exception as e:
                    validation_result["valid"] = False
                    validation_result["validation_issues"].append(f"Failed to load JSON: {str(e)}")
            
            elif artifact_path.endswith('.parquet'):
                try:
                    import pandas as pd
                    df = pd.read_parquet(artifact_path)
                    validation_result["object_type"] = "DataFrame"
                    validation_result["shape"] = df.shape
                except Exception as e:
                    validation_result["valid"] = False
                    validation_result["validation_issues"].append(f"Failed to load parquet: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "artifact_path": artifact_path,
            }
        return self.step_dependencies.get(step_name, [])
    
    def get_dependent_steps(self, step_name: str) -> List[str]:
        """
        Get the list of steps that depend on the given step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            List of dependent step names
        """
        dependent_steps = []
        for step, dependencies in self.step_dependencies.items():
            if step_name in dependencies:
                dependent_steps.append(step)
        return dependent_steps


# Global instance
step_dependency_validator = StepDependencyValidator()
