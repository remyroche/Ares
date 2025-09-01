#!/usr/bin/env python3
"""Enhanced Validator for Step 3: HMM Regime Discovery with Enhanced Clustering.

This validator ensures that the enhanced HMM regime discovery step has completed successfully
and generated all required artifacts for downstream steps, including enhanced clustering results.
"""

import json
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "src.utils.centralized_decorators",
    "src.utils.logger",
    "src.utils.enhanced_validation_decorators",
    "src.utils.comprehensive_file_validation"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
centralized_decorators = PipelineStandards.safe_import("src.utils.centralized_decorators", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
enhanced_validation = PipelineStandards.safe_import("src.utils.enhanced_validation_decorators", None)
comprehensive_file_validation = PipelineStandards.safe_import("src.utils.comprehensive_file_validation", None)

# Fallback functions if imports fail
def create_fallback_logger():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("Step3.HMMRegimeDiscovery.Validator")

def create_fallback_decorator():
    def decorator(func):
        return func
    return decorator

# Initialize fallbacks
if system_logger is None:
    system_logger = create_fallback_logger()

if centralized_decorators is None:
    handle_errors = create_fallback_decorator()
    validate_data_structure = create_fallback_decorator()
    comprehensive_data_validation = create_fallback_decorator()
    monitor_step_execution = create_fallback_decorator()
    secure_step_execution = create_fallback_decorator()
    validate_pipeline_step = create_fallback_decorator()
    with_tracing_span = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
else:
    handle_errors = centralized_decorators.handle_errors
    validate_data_structure = centralized_decorators.validate_data_structure
    comprehensive_data_validation = centralized_decorators.comprehensive_data_validation
    monitor_step_execution = centralized_decorators.monitor_step_execution
    secure_step_execution = centralized_decorators.secure_step_execution
    validate_pipeline_step = centralized_decorators.validate_pipeline_step
    with_tracing_span = centralized_decorators.with_tracing_span
    quality_gate = centralized_decorators.quality_gate

if enhanced_validation is None:
    comprehensive_step_validation = create_fallback_decorator()
    validate_enhanced_clustering_artifacts = create_fallback_decorator()
    validate_hmm_reliability_metrics = create_fallback_decorator()
else:
    comprehensive_step_validation = enhanced_validation.comprehensive_step_validation
    validate_enhanced_clustering_artifacts = enhanced_validation.validate_enhanced_clustering_artifacts
    validate_hmm_reliability_metrics = enhanced_validation.validate_hmm_reliability_metrics

logger = system_logger.getChild("Step3.HMMRegimeDiscovery.Validator")

class EnhancedStep03Validator:
    """Enhanced validator for Step 3 HMM Regime Discovery with enhanced clustering."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_cache = {}
        self.performance_metrics = {}
        
    @comprehensive_step_validation(
        step_name="step03_hmm_regime_discovery",
        validate_prerequisites=True,
        validate_inputs=True,
        validate_outputs=True,
        validate_data_quality=True,
        cache_validation=True,
        log_level="INFO"
    )
    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError, OSError, FileNotFoundError),
        default_return={"validation_passed": False, "error": "Validation failed"}
    )
    @monitor_step_execution
    @secure_step_execution
    @with_tracing_span("step03_validation")
    async def run_validator(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate Step 3: Enhanced HMM Regime Discovery completion and artifacts.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Validation result dictionary with enhanced clustering validation
        """
        start_time = time.time()
        
        try:
            # Extract parameters with validation
            symbol = training_input.get("symbol", "")
            exchange = training_input.get("exchange", "")
            timeframe = training_input.get("timeframe", "")
            data_dir = training_input.get("data_dir", "data_cache")
            
            # Validate required parameters
            if not all([symbol, exchange, timeframe]):
                return {
                    "validation_passed": False,
                    "error": "Missing required parameters: symbol, exchange, timeframe",
                    "validation_results": {},
                }

            logger.info(f"🔍 Validating Step 3: Enhanced HMM Regime Discovery for {symbol} on {exchange} ({timeframe})")

            # Check pipeline state
            hmm_state = pipeline_state.get("hmm_regime_discovery", {})
            if not hmm_state.get("completed", False):
                return {
                    "validation_passed": False,
                    "error": "Step 3 HMM regime discovery not completed in pipeline state",
                    "validation_results": {},
                }

            # Validate enhanced clustering results
            enhanced_clustering_validation = await self._validate_enhanced_clustering_results(
                training_input, pipeline_state, data_dir
            )
            
            # Validate traditional artifacts
            traditional_artifacts_validation = await self._validate_traditional_artifacts(
                training_input, data_dir
            )
            
            # Validate HMM reliability metrics
            hmm_reliability_validation = await self._validate_hmm_reliability_metrics(
                pipeline_state, data_dir
            )
            
            # Combine validation results
            all_validations = [
                enhanced_clustering_validation,
                traditional_artifacts_validation,
                hmm_reliability_validation
            ]
            
            validation_passed = all(v.get("validation_passed", False) for v in all_validations)
            
            # Calculate comprehensive validation metrics
            total_checks = sum(v.get("total_checks", 0) for v in all_validations)
            passed_checks = sum(v.get("passed_checks", 0) for v in all_validations)
            validation_coverage = passed_checks / total_checks if total_checks > 0 else 0.0
            
            # Collect all errors and warnings
            all_errors = []
            all_warnings = []
            for validation in all_validations:
                all_errors.extend(validation.get("errors", []))
                all_warnings.extend(validation.get("warnings", []))
            
            validation_results = {
                "validation_passed": validation_passed,
                "validation_coverage": validation_coverage,
                "passed_checks": passed_checks,
                "total_checks": total_checks,
                "errors": all_errors,
                "warnings": all_warnings,
                "validation_time": time.time() - start_time,
                "enhanced_clustering": enhanced_clustering_validation,
                "traditional_artifacts": traditional_artifacts_validation,
                "hmm_reliability": hmm_reliability_validation,
            }

            if validation_passed:
                logger.info(f"✅ Step 3 enhanced validation passed: {passed_checks}/{total_checks} checks passed")
                logger.info(f"📊 Validation coverage: {validation_coverage:.2%}")
            else:
                logger.error(f"❌ Step 3 enhanced validation failed: {len(all_errors)} errors found")
                for error in all_errors:
                    logger.error(f"   Error: {error}")

            return validation_results

        except Exception as e:
            logger.exception(f"❌ Step 3 enhanced validator error: {e}")
            return {
                "validation_passed": False,
                "error": str(e),
                "validation_results": {},
                "validation_time": time.time() - start_time,
            }

    @validate_enhanced_clustering_artifacts
    @handle_errors(exceptions=(Exception,), default_return={"validation_passed": False})
    async def _validate_enhanced_clustering_results(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate enhanced clustering results and artifacts."""
        
        symbol = training_input.get("symbol", "")
        exchange = training_input.get("exchange", "")
        timeframe = training_input.get("timeframe", "")
        
        validation_results = {
            "validation_passed": True,
            "total_checks": 0,
            "passed_checks": 0,
            "errors": [],
            "warnings": [],
            "artifact_info": {}
        }
        
        # Check for enhanced clustering report
        reports_dir = Path(data_dir) / "reports"
        enhanced_report_pattern = f"enhanced_clustering_report_*.txt"
        enhanced_reports = list(reports_dir.glob(enhanced_report_pattern))
        
        if enhanced_reports:
            latest_report = max(enhanced_reports, key=lambda x: x.stat().st_mtime)
            validation_results["total_checks"] += 1
            
            try:
                with open(latest_report, 'r') as f:
                    report_content = f.read()
                
                # Validate report content
                required_sections = [
                    "Enhanced Clustering Results",
                    "HMM Reliability Metrics",
                    "Cluster Quality Analysis",
                    "Refinement History"
                ]
                
                missing_sections = [section for section in required_sections if section not in report_content]
                
                if not missing_sections:
                    validation_results["passed_checks"] += 1
                    validation_results["artifact_info"]["enhanced_clustering_report"] = {
                        "exists": True,
                        "path": str(latest_report),
                        "size_bytes": latest_report.stat().st_size,
                        "sections_found": len(required_sections) - len(missing_sections)
                    }
                else:
                    validation_results["errors"].append(f"Enhanced clustering report missing sections: {missing_sections}")
                    
            except Exception as e:
                validation_results["errors"].append(f"Failed to read enhanced clustering report: {e}")
        else:
            validation_results["errors"].append("No enhanced clustering report found")
        
        # Validate enhanced clustering metrics in pipeline state
        enhanced_clustering_metrics = pipeline_state.get("hmm_regime_discovery", {}).get("enhanced_clustering", {})
        
        if enhanced_clustering_metrics:
            validation_results["total_checks"] += 1
            
            required_metrics = [
                "composite_score",
                "hmm_reliability_score", 
                "quality_improvement",
                "iterations"
            ]
            
            missing_metrics = [metric for metric in required_metrics if metric not in enhanced_clustering_metrics]
            
            if not missing_metrics:
                validation_results["passed_checks"] += 1
                
                # Validate metric values
                composite_score = enhanced_clustering_metrics.get("composite_score", 0)
                hmm_reliability = enhanced_clustering_metrics.get("hmm_reliability_score", 0)
                
                if composite_score < 0.3:
                    validation_results["warnings"].append(f"Low composite score: {composite_score:.4f}")
                
                if hmm_reliability < 0.5:
                    validation_results["warnings"].append(f"Low HMM reliability score: {hmm_reliability:.4f}")
                    
            else:
                validation_results["errors"].append(f"Missing enhanced clustering metrics: {missing_metrics}")
        else:
            validation_results["errors"].append("No enhanced clustering metrics found in pipeline state")
        
        # Validate cluster quality metrics
        cluster_quality = pipeline_state.get("hmm_regime_discovery", {}).get("cluster_quality", {})
        
        if cluster_quality:
            validation_results["total_checks"] += 1
            
            required_quality_metrics = [
                "silhouette_score",
                "calinski_harabasz_score", 
                "davies_bouldin_score",
                "composite_score",
                "hmm_reliability_score"
            ]
            
            missing_quality_metrics = [metric for metric in required_quality_metrics if metric not in cluster_quality]
            
            if not missing_quality_metrics:
                validation_results["passed_checks"] += 1
                
                # Validate quality metric ranges
                silhouette = cluster_quality.get("silhouette_score", 0)
                if silhouette < -0.5 or silhouette > 1.0:
                    validation_results["warnings"].append(f"Silhouette score out of expected range: {silhouette:.4f}")
                    
            else:
                validation_results["errors"].append(f"Missing cluster quality metrics: {missing_quality_metrics}")
        else:
            validation_results["errors"].append("No cluster quality metrics found")
        
        validation_results["validation_passed"] = len(validation_results["errors"]) == 0
        return validation_results

    @validate_data_structure
    @handle_errors(exceptions=(Exception,), default_return={"validation_passed": False})
    async def _validate_traditional_artifacts(
        self,
        training_input: Dict[str, Any],
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate traditional HMM regime discovery artifacts."""
        
        symbol = training_input.get("symbol", "")
        exchange = training_input.get("exchange", "")
        timeframe = training_input.get("timeframe", "")
        
        validation_results = {
            "validation_passed": True,
            "total_checks": 0,
            "passed_checks": 0,
            "errors": [],
            "warnings": [],
            "artifact_info": {}
        }
        
        # Define required artifacts
        required_artifacts = [
            f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet",
            f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet",
            f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet",
            f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json",
        ]

        # Check for required artifacts
        missing_artifacts = []
        artifact_info = {}

        for artifact in required_artifacts:
            validation_results["total_checks"] += 1
            artifact_path = Path(data_dir) / artifact
            
            if artifact_path.exists():
                # Get file info
                stat = artifact_path.stat()
                artifact_info[artifact] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "modified_time": stat.st_mtime,
                }

                # Validate file content for parquet files
                if artifact.endswith(".parquet"):
                    try:
                        df = pd.read_parquet(artifact_path)
                        artifact_info[artifact]["rows"] = len(df)
                        artifact_info[artifact]["columns"] = list(df.columns)

                        # Check for required columns based on artifact type
                        if "block_states" in artifact:
                            required_cols = [col for col in df.columns if "state_id" in col or "p_state_" in col]
                            if required_cols:
                                validation_results["passed_checks"] += 1
                            else:
                                validation_results["errors"].append(f"{artifact} - missing state columns")

                        elif "composite_clusters" in artifact:
                            if "composite_cluster_id" in df.columns:
                                validation_results["passed_checks"] += 1
                                
                                # Validate cluster distribution
                                cluster_counts = df["composite_cluster_id"].value_counts()
                                if len(cluster_counts) < 2:
                                    validation_results["warnings"].append(f"{artifact} - only {len(cluster_counts)} clusters found")
                                    
                                # Check for training mode consistency
                                light_mode = os.environ.get("LIGHT_TRAINING_MODE", "0") == "1"
                                blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
                                
                                expected_clusters = 2 if light_mode else (4 if blank_mode else 20)
                                if len(cluster_counts) != expected_clusters:
                                    validation_results["warnings"].append(
                                        f"{artifact} - expected {expected_clusters} clusters, found {len(cluster_counts)}"
                                    )
                            else:
                                validation_results["errors"].append(f"{artifact} - missing composite_cluster_id column")

                        elif "intensity" in artifact:
                            intensity_cols = [col for col in df.columns if "intensity_" in col]
                            if intensity_cols:
                                validation_results["passed_checks"] += 1
                            else:
                                validation_results["errors"].append(f"{artifact} - missing intensity columns")

                    except Exception as e:
                        validation_results["errors"].append(f"{artifact} - failed to read: {e}")

                # Validate JSON metadata
                elif artifact.endswith(".json"):
                    try:
                        with open(artifact_path) as f:
                            meta = json.load(f)
                        artifact_info[artifact]["metadata"] = meta

                        # Check required metadata fields
                        required_meta_fields = ["symbol", "exchange", "timeframe", "blocks_used", "n_composite_clusters"]
                        missing_fields = [field for field in required_meta_fields if field not in meta]
                        if not missing_fields:
                            validation_results["passed_checks"] += 1
                        else:
                            validation_results["errors"].append(f"{artifact} - missing metadata fields: {missing_fields}")

                    except Exception as e:
                        validation_results["errors"].append(f"{artifact} - failed to read JSON: {e}")
            else:
                missing_artifacts.append(artifact)
                artifact_info[artifact] = {"exists": False}
                validation_results["errors"].append(f"Missing artifact: {artifact}")

        # Check for HMM regimes directory artifacts
        hmm_regimes_dir = Path(data_dir) / "hmm_regimes"
        hmm_regimes_artifacts = [
            f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet",
        ]

        for artifact in hmm_regimes_artifacts:
            validation_results["total_checks"] += 1
            artifact_path = hmm_regimes_dir / artifact
            if artifact_path.exists():
                validation_results["passed_checks"] += 1
                artifact_info[f"hmm_regimes/{artifact}"] = {
                    "exists": True,
                    "size_bytes": artifact_path.stat().st_size,
                }
            else:
                missing_artifacts.append(f"hmm_regimes/{artifact}")
                artifact_info[f"hmm_regimes/{artifact}"] = {"exists": False}
                validation_results["errors"].append(f"Missing HMM regimes artifact: {artifact}")

        validation_results["artifact_info"] = artifact_info
        validation_results["missing_artifacts"] = missing_artifacts
        validation_results["validation_passed"] = len(validation_results["errors"]) == 0
        
        return validation_results

    @validate_hmm_reliability_metrics
    @handle_errors(exceptions=(Exception,), default_return={"validation_passed": False})
    async def _validate_hmm_reliability_metrics(
        self,
        pipeline_state: Dict[str, Any],
        data_dir: str
    ) -> Dict[str, Any]:
        """Validate HMM reliability metrics and quality indicators."""
        
        validation_results = {
            "validation_passed": True,
            "total_checks": 0,
            "passed_checks": 0,
            "errors": [],
            "warnings": [],
            "hmm_metrics": {}
        }
        
        # Extract HMM metrics from pipeline state
        hmm_metrics = pipeline_state.get("hmm_regime_discovery", {}).get("cluster_quality", {})
        
        if hmm_metrics:
            validation_results["total_checks"] += 1
            
            # Validate HMM reliability score
            hmm_reliability_score = hmm_metrics.get("hmm_reliability_score", 0)
            if 0 <= hmm_reliability_score <= 1:
                validation_results["passed_checks"] += 1
                validation_results["hmm_metrics"]["reliability_score"] = hmm_reliability_score
                
                if hmm_reliability_score < 0.5:
                    validation_results["warnings"].append(f"Low HMM reliability score: {hmm_reliability_score:.4f}")
            else:
                validation_results["errors"].append(f"Invalid HMM reliability score: {hmm_reliability_score}")
            
            # Validate HMM entropy penalty
            hmm_entropy_penalty = hmm_metrics.get("hmm_entropy_penalty", 0)
            if 0 <= hmm_entropy_penalty <= 1:
                validation_results["passed_checks"] += 1
                validation_results["hmm_metrics"]["entropy_penalty"] = hmm_entropy_penalty
                
                if hmm_entropy_penalty > 0.7:
                    validation_results["warnings"].append(f"High HMM entropy penalty: {hmm_entropy_penalty:.4f}")
            else:
                validation_results["errors"].append(f"Invalid HMM entropy penalty: {hmm_entropy_penalty}")
            
            # Validate HMM transition smoothness
            hmm_transition_smoothness = hmm_metrics.get("hmm_transition_smoothness", 0)
            if 0 <= hmm_transition_smoothness <= 1:
                validation_results["passed_checks"] += 1
                validation_results["hmm_metrics"]["transition_smoothness"] = hmm_transition_smoothness
                
                if hmm_transition_smoothness < 0.3:
                    validation_results["warnings"].append(f"Low HMM transition smoothness: {hmm_transition_smoothness:.4f}")
            else:
                validation_results["errors"].append(f"Invalid HMM transition smoothness: {hmm_transition_smoothness}")
                
        else:
            validation_results["errors"].append("No HMM reliability metrics found in pipeline state")
        
        # Validate HMM model score
        hmm_score = pipeline_state.get("hmm_regime_discovery", {}).get("hmm_score", None)
        if hmm_score is not None:
            validation_results["total_checks"] += 1
            validation_results["passed_checks"] += 1
            validation_results["hmm_metrics"]["hmm_score"] = hmm_score
            
            if hmm_score < -1000:  # Threshold depends on your data scale
                validation_results["warnings"].append(f"Low HMM model score: {hmm_score:.4f}")
        else:
            validation_results["warnings"].append("No HMM model score found")
        
        validation_results["validation_passed"] = len(validation_results["errors"]) == 0
        return validation_results

# Legacy function for backward compatibility
@handle_errors(exceptions=(Exception,), default_return={"validation_passed": False})
async def run_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Legacy validator function for backward compatibility."""
    config = training_input.get("config", {})
    validator = EnhancedStep03Validator(config)
    return await validator.run_validator(training_input, pipeline_state)

async def run_step_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Legacy function name for backward compatibility."""
    return await run_validator(training_input, pipeline_state)