from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Pipeline standards for backtesting."""

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class PipelineStandards:
    """Pipeline standards configuration."""

    def __init__(self):
        self.standards = {
            "data_quality": True,
            "validation": True,
            "logging": True,
            "performance_monitoring": True,
            "error_handling": True,
            "memory_optimization": True,
            "gpu_acceleration": True,
            "parallel_processing": True
        }
        
        logger.info("🚀 PipelineStandards initialized")
        logger.info(f"📊 Active standards: {list(self.standards.keys())}")
        logger.info(f"✅ Enabled standards: {[k for k, v in self.standards.items() if v]}")
        logger.info(f"❌ Disabled standards: {[k for k, v in self.standards.items() if not v]}")
    
    def get_standard(self, standard_name: str) -> bool:
        """Get the status of a specific standard."""
        status = self.standards.get(standard_name, False)
        logger.debug(f"🔍 Standard '{standard_name}': {'Enabled' if status else 'Disabled'}")
        return status
    
    def set_standard(self, standard_name: str, enabled: bool) -> None:
        """Set the status of a specific standard."""
        old_status = self.standards.get(standard_name, False)
        self.standards[standard_name] = enabled
        
        if old_status != enabled:
            logger.info(f"🔄 Standard '{standard_name}' {'enabled' if enabled else 'disabled'}")
        else:
            logger.debug(f"ℹ️ Standard '{standard_name}' already {'enabled' if enabled else 'disabled'}")
    
    def enable_all_standards(self) -> None:
        """Enable all standards."""
        logger.info("🔄 Enabling all pipeline standards...")
        for standard in self.standards:
            self.standards[standard] = True
        logger.info("✅ All pipeline standards enabled")
    
    def disable_all_standards(self) -> None:
        """Disable all standards."""
        logger.warning("⚠️ Disabling all pipeline standards...")
        for standard in self.standards:
            self.standards[standard] = False
        logger.warning("❌ All pipeline standards disabled")
    
    def get_enabled_standards(self) -> list:
        """Get list of enabled standards."""
        enabled = [k for k, v in self.standards.items() if v]
        logger.debug(f"📊 Enabled standards: {enabled}")
        return enabled
    
    def get_disabled_standards(self) -> list:
        """Get list of disabled standards."""
        disabled = [k for k, v in self.standards.items() if not v]
        logger.debug(f"📊 Disabled standards: {disabled}")
        return disabled
    
    def validate_standards(self) -> Dict[str, Any]:
        """Validate current standards configuration."""
        logger.info("🔍 Validating pipeline standards configuration...")
        
        validation_results = {
            "total_standards": len(self.standards),
            "enabled_standards": len(self.get_enabled_standards()),
            "disabled_standards": len(self.get_disabled_standards()),
            "critical_standards_enabled": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check critical standards
        critical_standards = ["data_quality", "validation", "error_handling"]
        for standard in critical_standards:
            if not self.get_standard(standard):
                validation_results["critical_standards_enabled"] = False
                validation_results["warnings"].append(f"Critical standard '{standard}' is disabled")
        
        # Performance recommendations
        if not self.get_standard("performance_monitoring"):
            validation_results["recommendations"].append("Consider enabling performance monitoring for production")
        
        if not self.get_standard("memory_optimization"):
            validation_results["recommendations"].append("Consider enabling memory optimization for large datasets")
        
        logger.info(f"✅ Standards validation completed")
        logger.info(f"📊 Total standards: {validation_results['total_standards']}")
        logger.info(f"✅ Enabled: {validation_results['enabled_standards']}")
        logger.info(f"❌ Disabled: {validation_results['disabled_standards']}")
        
        if validation_results["warnings"]:
            for warning in validation_results["warnings"]:
                logger.warning(f"⚠️ {warning}")
        
        if validation_results["recommendations"]:
            for recommendation in validation_results["recommendations"]:
                logger.info(f"💡 {recommendation}")
        
        return validation_results

# Global instance
pipeline_standards = PipelineStandards()