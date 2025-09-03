#!/usr/bin/env python3
"""Validator for Step 3.5: Final Regime Clustering."

This module validates the final regime clustering step outputs with comprehensive
quality checks for regime clustering artifacts and analysis reports.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.base_validator import BaseValidator
from src.utils.logger import system_logger
from src.utils.enhanced_validation_decorators import (
    validate_step3_5_comprehensive,
    smart_validation_cache
)
from src.utils.common_operations import safe_json_load

logger = system_logger.getChild("Step3_5FinalRegimeClusteringValidator")


class Step3_5FinalRegimeClusteringValidator(BaseValidator):
    """Validator for Step 3.5: Final Regime Clustering."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("step3_5_final_regime_clustering", config)
        self.logger = system_logger.getChild("Validator.Step3_5")

    @validate_step3_5_comprehensive
    async def validate_step3_5_final_regime_clustering(
        self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]
    ) -> bool:
        """Validate Step 3.5: Final Regime Clustering."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes
        """
        self.logger.info("🔍 Starting Step 3.5: Final Regime Clustering validation")

        try:
            # Check if final regime clustering directory exists
            final_regime_dir = Path(data_dir) / "training" / "final_regime_clustering"
            if not final_regime_dir.exists():
                self.logger.warning(
                    f"⚠️ Final regime clustering directory not found: {final_regime_dir}"
                )
                return False

            # Validate final regime clustering files
            regime_files = list(final_regime_dir.glob("*.parquet"))
            if not regime_files:
                self.logger.warning("⚠️ No final regime clustering files found")
                return False

            # Validate each regime file
            for regime_file in regime_files:
                if not await self._validate_final_regime_file(regime_file):
                    return False

            # Check for final regime analysis report
            analysis_report = final_regime_dir / f"{exchange}_{symbol}_1m_final_regime_analysis.json"
            if not analysis_report.exists():
                self.logger.warning(f"⚠️ Final regime analysis report not found: {analysis_report}")
                return False

            # Validate analysis report
            if not await self._validate_analysis_report(analysis_report):
                return False

            # Check for regime characteristics file
            characteristics_file = final_regime_dir / f"{exchange}_{symbol}_1m_regime_characteristics.json"
            if not characteristics_file.exists():
                self.logger.warning(f"⚠️ Regime characteristics file not found: {characteristics_file}")
                return False

            # Validate characteristics file
            if not await self._validate_characteristics_file(characteristics_file):
                return False

            self.logger.info("✅ Step 3.5: Final Regime Clustering validation passed")
            return True

        except Exception as e:
            error_context = {
                "step": "step3_5_final_regime_clustering",
                "symbol": symbol,
                "exchange": exchange,
                "data_dir": data_dir,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            self.logger.exception(f"❌ Step 3.5 validation failed: {error_context}")
            return False

    @smart_validation_cache(ttl_seconds=300)  # Cache for 5 minutes
    async def _validate_final_regime_file(self, regime_file: Path) -> bool:
        """Validate a final regime clustering file with caching."""
        try:
            self.logger.info(f"📁 Validating final regime file: {regime_file.name}")

            # Use BaseValidator's file validation'
            file_exists, file_metrics = self.validate_file_exists(str(regime_file), "regime file")
            if not file_exists:
                return False

            # Load and validate the regime file
            df = pd.read_parquet(regime_file)

            # Use BaseValidator's DataFrame validation'
            df_valid, df_metrics = self.validate_dataframe_quality(
                df=df,
                min_rows=100,
                required_columns=["timestamp", "final_regime_id", "regime_confidence"],
                check_data_types=True,
                check_value_ranges=True,
                check_duplicates=True,
                check_temporal_consistency=True
            )

            if not df_valid:
                self.logger.warning(f"⚠️ DataFrame validation failed for {regime_file.name}")
                return False

            # Additional regime-specific validation
            if "final_regime_id" in df.columns:
                unique_regimes = df["final_regime_id"].nunique()
                if unique_regimes < 2 or unique_regimes > 50:
                    self.logger.warning(
                        f"⚠️ Unusual number of regimes ({unique_regimes}) in {regime_file.name}"
                    )

            if "regime_confidence" in df.columns:
                confidence_range = df["regime_confidence"].agg(['min', 'max'])
                if confidence_range['min'] < 0 or confidence_range['max'] > 1:
                    self.logger.warning(f"⚠️ Confidence values out of range [0,1] in {regime_file.name}")

            self.logger.info(f"✅ Final regime file validated: {regime_file.name}")
            return True

        except Exception as e:
            error_context = {
                "file": str(regime_file),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            self.logger.exception(f"❌ Failed to validate final regime file: {error_context}")
            return False

    @smart_validation_cache(ttl_seconds=600)  # Cache for 10 minutes
    async def _validate_analysis_report(self, analysis_report: Path) -> bool:
        """Validate the final regime analysis report with caching."""
        try:
            self.logger.info(f"📊 Validating analysis report: {analysis_report.name}")

            # Use BaseValidator's file validation'
            file_exists, file_metrics = self.validate_file_exists(str(analysis_report), "analysis report")
            if not file_exists:
                return False

            report_data = safe_json_load(analysis_report)

            # Check required fields
            required_fields = ["regime_count", "clustering_metrics", "regime_analysis"]
            missing_fields = [field for field in required_fields if field not in report_data]
            if missing_fields:
                self.logger.warning(
                    f"⚠️ Missing required fields in analysis report: {missing_fields}"
                )
                return False

            # Validate regime count
            regime_count = report_data.get("regime_count", 0)
            if regime_count < 2 or regime_count > 50:
                self.logger.warning(f"⚠️ Unusual regime count in analysis report: {regime_count}")

            # Validate clustering metrics
            clustering_metrics = report_data.get("clustering_metrics", {})
            if not clustering_metrics:
                self.logger.warning("⚠️ Empty clustering metrics in analysis report")
                return False

            # Validate regime analysis
            regime_analysis = report_data.get("regime_analysis", {})
            if not regime_analysis:
                self.logger.warning("⚠️ Empty regime analysis in analysis report")
                return False

            self.logger.info(f"✅ Analysis report validated: {analysis_report.name}")
            return True

        except Exception as e:
            error_context = {
                "file": str(analysis_report),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            self.logger.exception(f"❌ Failed to validate analysis report: {error_context}")
            return False

    @smart_validation_cache(ttl_seconds=600)  # Cache for 10 minutes
    async def _validate_characteristics_file(self, characteristics_file: Path) -> bool:
        """Validate the regime characteristics file with caching."""
        try:
            self.logger.info(f"📊 Validating characteristics file: {characteristics_file.name}")

            # Use BaseValidator's file validation'
            file_exists, file_metrics = self.validate_file_exists(str(characteristics_file), "characteristics file")
            if not file_exists:
                return False

            characteristics_data = safe_json_load(characteristics_file)

            # Check if it's a dictionary'
            if not isinstance(characteristics_data, dict):
                self.logger.warning("⚠️ Characteristics file should contain a dictionary")
                return False

            # Check for regime characteristics
            if not characteristics_data:
                self.logger.warning("⚠️ Empty characteristics data")
                return False

            # Validate each regime's characteristics'
            for regime_id, characteristics in characteristics_data.items():
                if not isinstance(characteristics, dict):
                    self.logger.warning(f"⚠️ Invalid characteristics format for regime {regime_id}")
                    return False

                # Check for basic characteristics
                basic_fields = ["volatility", "momentum", "volume_profile"]
                missing_basic = [field for field in basic_fields if field not in characteristics]
                if missing_basic:
                    self.logger.warning(
                        f"⚠️ Missing basic characteristics for regime {regime_id}: {missing_basic}"
                    )
                    return False

            self.logger.info(f"✅ Characteristics file validated: {characteristics_file.name}")
            return True

        except Exception as e:
            error_context = {
                "file": str(characteristics_file),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            self.logger.exception(f"❌ Failed to validate characteristics file: {error_context}")
            return False

    def validate_step_prerequisites(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate prerequisites for Step 3.5 using BaseValidator methods."""
        validation_result = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "details": {}
        }

        try:
            # Check if step3_hmm_regime_discovery output exists using BaseValidator
            step3_output_dir = Path("data/training")
            step3_files = list(step3_output_dir.glob(f"{exchange}_{symbol}_{timeframe}*hmm*.parquet"))
            
            if not step3_files:
                validation_result["validation_passed"] = False
                validation_result["errors"].append(
                    f"Step 3 HMM regime discovery output not found for {exchange}_{symbol}_{timeframe}"
                )
            else:
                # Validate each file using BaseValidator
                for file_path in step3_files:
                    file_valid, file_metrics = self.validate_file_exists(str(file_path), "step03 output file")
                    if not file_valid:
                        validation_result["warnings"].append(f"File validation failed: {file_path}")
                
                validation_result["details"]["step3_files_found"] = len(step3_files)
                validation_result["details"]["step3_files"] = [str(f) for f in step3_files]

            # Check if parameter optimization results exist
            param_optimization_file = Path("data/optimization/parameter_optimization_results.json")
            if param_optimization_file.exists():
                file_valid, file_metrics = self.validate_file_exists(str(param_optimization_file), "parameter optimization results")
                if not file_valid:
                    validation_result["warnings"].append(f"Parameter optimization file validation failed: {param_optimization_file}")
            else:
                validation_result["warnings"].append(
                    f"Parameter optimization results not found: {param_optimization_file}"
                )

        except Exception as e:
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"Prerequisites validation failed: {str(e)}")

        return validation_result

    def validate_step_output(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate Step 3.5 output files and content using BaseValidator methods."""
        validation_result = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "details": {}
        }

        try:
            # Define expected output files
            output_dir = Path("data/training/final_regime_clustering")
            expected_files = [
                f"{exchange}_{symbol}_{timeframe}_final_regime_clusters.parquet",
                f"{exchange}_{symbol}_{timeframe}_final_regime_analysis.json",
                f"{exchange}_{symbol}_{timeframe}_regime_characteristics.json"
            ]

            # Check if all expected files exist using BaseValidator
            missing_files = []
            existing_files = []
            
            for filename in expected_files:
                file_path = output_dir / filename
                file_valid, file_metrics = self.validate_file_exists(str(file_path), f"expected file: {filename}")
                
                if file_valid:
                    existing_files.append(str(file_path))
                else:
                    missing_files.append(filename)

            if missing_files:
                validation_result["validation_passed"] = False
                validation_result["errors"].extend([
                    f"Missing final regime clustering file: {f}" for f in missing_files
                ])
            else:
                validation_result["details"]["files_found"] = len(existing_files)
                validation_result["details"]["files"] = existing_files

            # Validate file contents using BaseValidator
            if existing_files:
                for file_path in existing_files:
                    if file_path.endswith(".parquet"):
                        try:
                            df = pd.read_parquet(file_path)
                            # Use BaseValidator's DataFrame validation'
                            df_valid, df_metrics = self.validate_dataframe_quality(
                                df, min_rows=100, check_data_types=True
                            )
                            validation_result["details"][f"{Path(file_path).stem}_rows"] = len(df)
                            validation_result["details"][f"{Path(file_path).stem}_columns"] = list(df.columns)
                            validation_result["details"][f"{Path(file_path).stem}_valid"] = df_valid
                        except Exception as e:
                            validation_result["warnings"].append(f"Could not read parquet file {file_path}: {e}")

        except Exception as e:
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"Output validation failed: {str(e)}")

        return validation_result


async def run_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run validation for Step 3.5: Final Regime Clustering."

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info("🔍 Validating Step 3.5: Final Regime Clustering")
    
    try:
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        timeframe = training_input.get("timeframe", "1m")
        data_dir = training_input.get("data_dir", "data_cache")
        
        # Initialize validator with BaseValidator inheritance
        config = training_input.get("config", {})
        validator = Step3_5FinalRegimeClusteringValidator(config)
        
        # Validate prerequisites using BaseValidator methods
        prereq_result = validator.validate_step_prerequisites(symbol, exchange, timeframe)
        
        # Validate step execution
        step_result = await validator.validate_step3_5_final_regime_clustering(
            symbol, exchange, data_dir, training_input
        )
        
        # Validate outputs using BaseValidator methods
        output_result = validator.validate_step_output(symbol, exchange, timeframe)
        
        # Combine results
        validation_passed = (
            prereq_result["validation_passed"] and 
            step_result and 
            output_result["validation_passed"]
        )
        
        return {
            "step_name": "step3_5_final_regime_clustering",
            "validation_passed": validation_passed,
            "prerequisites": prereq_result,
            "step_execution": step_result,
            "outputs": output_result,
            "warnings": prereq_result["warnings"] + output_result["warnings"],
            "errors": prereq_result["errors"] + output_result["errors"]
        }
        
    except Exception as e:
        error_context = {
            "step": "step3_5_final_regime_clustering",
            "symbol": training_input.get("symbol", "UNKNOWN"),
            "exchange": training_input.get("exchange", "UNKNOWN"),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        logger.exception(f"❌ Step 3.5 validation failed: {error_context}")
        return {
            "step_name": "step3_5_final_regime_clustering",
            "validation_passed": False,
            "error": str(e),
            "error_context": error_context
        }


if __name__ == "__main__":
    # Test the validator
    import asyncio
import datetime as datetime
    
    test_input = {
    "symbol": "ETHUSDT",
        "exchange": "BINANCE", 
        "timeframe": "1m",
        "data_dir": "data_cache",
        "config": {}
    }
    
    test_state = {}
    
    result = asyncio.run(run_validator(test_input, test_state))
    print(json.dumps(result, indent=2))