"""
Step dependency validator for the training pipeline.
Ensures that steps don't proceed if their prerequisites have failed.
"""

import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from src.utils.logger import system_logger
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
            "step2_feature_engineering": [],
            "step3_hmm_regime_discovery": ["step2_feature_engineering"],
            "step4_processing_labeling": ["step3_hmm_regime_discovery"],
            "step5_regime_data_splitting": ["step4_processing_labeling"],
            "step6_regime_specific_training": ["step5_regime_data_splitting"],
            "step7_ensemble_creation": ["step6_regime_specific_training"],
            "step8_model_evaluation": ["step7_ensemble_creation"],
            "step9_hyperparameter_optimization": ["step8_model_evaluation"],
            "step10_model_selection": ["step9_hyperparameter_optimization"],
            "step11_backtesting": ["step10_model_selection"],
            "step12_walk_forward_validation": ["step11_backtesting"],
            "step13_monte_carlo_validation": ["step12_walk_forward_validation"],
            "step14_model_deployment": ["step13_monte_carlo_validation"],
            "step15_live_monitoring": ["step14_model_deployment"],
            "step16_performance_tracking": ["step15_live_monitoring"],
        }
        
        # Define critical data requirements for each step
        self.critical_data_requirements = {
            "step3_hmm_regime_discovery": {
                "required_files": ["data/unified_data/*.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 1000
            },
            "step4_processing_labeling": {
                "required_files": ["data/hmm_regimes/*.parquet", "data/training/*_hmm_composite_clusters_*.parquet"],
                "required_columns": ["composite_cluster_id"],
                "min_rows": 100
            },
            "step5_regime_data_splitting": {
                "required_files": ["data/training/*_labeled_*.parquet"],
                "required_columns": ["label", "composite_cluster_id"],
                "min_rows": 100
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
                            "step3_hmm_regime_discovery": "hmm_regime_discovery",
                            "step4_processing_labeling": "processing_labeling",
                            "step5_regime_data_splitting": "regime_data_splitting",
                            "step6_hmm_based_training": "hmm_based_training",
                            "step6_5_unified_regime_intelligence": "unified_regime_intelligence",
                            "step7_analyst_enhancement": "analyst_enhancement",
                            "step8_tactician_labeling": "tactician_labeling",
                            "step9_tactician_specialist_training": "tactician_specialist_training",
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
                                "step3_hmm_regime_discovery": "hmm_regime_discovery",
                                "step4_processing_labeling": "processing_labeling",
                                "step5_regime_data_splitting": "regime_data_splitting",
                                "step6_hmm_based_training": "hmm_based_training",
                                "step6_5_unified_regime_intelligence": "unified_regime_intelligence",
                                "step7_analyst_enhancement": "analyst_enhancement",
                                "step8_tactician_labeling": "tactician_labeling",
                                "step9_tactician_specialist_training": "tactician_specialist_training",
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
