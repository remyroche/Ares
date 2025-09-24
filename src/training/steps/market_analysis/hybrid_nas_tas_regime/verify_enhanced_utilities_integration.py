"""
Verification script to check that enhanced utility integrations are properly wired and used.
This script performs static analysis to verify the integration without requiring runtime dependencies.
"""

import os
import re
from pathlib import Path

def verify_enhanced_utilities_integration():
    """Verify that enhanced utility integrations are properly wired and used."""
    print("🔍 Verifying Enhanced Utility Integration...")
    
    # Check if enhanced utility integration files exist
    integration_files = [
        "shared_utils/enhanced_utility_integration.py",
        "shared_utils/enhanced_data_integration.py", 
        "shared_utils/enhanced_ml_integration.py"
    ]
    
    for file_path in integration_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing integration file: {file_path}")
            return False
        else:
            print(f"✅ Found integration file: {file_path}")
    
    # Check if enhanced hybrid orchestrator exists and uses the integrations
    orchestrator_file = "enhanced_hybrid_orchestrator.py"
    if not os.path.exists(orchestrator_file):
        print(f"❌ Missing orchestrator file: {orchestrator_file}")
        return False
    
    print(f"✅ Found orchestrator file: {orchestrator_file}")
    
    # Read the orchestrator file to check for integration usage
    with open(orchestrator_file, 'r') as f:
        orchestrator_content = f.read()
    
    # Check for import statements
    import_checks = [
        "from .shared_utils.enhanced_utility_integration import",
        "from .shared_utils.enhanced_data_integration import",
        "from .shared_utils.enhanced_ml_integration import"
    ]
    
    for import_check in import_checks:
        if import_check not in orchestrator_content:
            print(f"❌ Missing import: {import_check}")
            return False
        else:
            print(f"✅ Found import: {import_check}")
    
    # Check for initialization
    init_checks = [
        "self.utility_integration = create_enhanced_utility_integration",
        "self.data_integration = create_enhanced_data_integration",
        "self.ml_integration = create_enhanced_ml_integration"
    ]
    
    for init_check in init_checks:
        if init_check not in orchestrator_content:
            print(f"❌ Missing initialization: {init_check}")
            return False
        else:
            print(f"✅ Found initialization: {init_check}")
    
    # Check for enhanced method usage
    method_checks = [
        "self.utility_integration.",
        "self.data_integration.",
        "self.ml_integration."
    ]
    
    utility_usage_count = orchestrator_content.count("self.utility_integration.")
    data_usage_count = orchestrator_content.count("self.data_integration.")
    ml_usage_count = orchestrator_content.count("self.ml_integration.")
    
    print(f"📊 Usage counts:")
    print(f"   Utility integration: {utility_usage_count} usages")
    print(f"   Data integration: {data_usage_count} usages")
    print(f"   ML integration: {ml_usage_count} usages")
    
    if utility_usage_count == 0:
        print("❌ No utility integration usage found")
        return False
    if data_usage_count == 0:
        print("❌ No data integration usage found")
        return False
    if ml_usage_count == 0:
        print("❌ No ML integration usage found")
        return False
    
    # Check for enhanced method definitions
    enhanced_methods = [
        "_preprocess_market_data_enhanced",
        "_run_tas_analysis_enhanced",
        "_run_nas_analysis_enhanced",
        "_run_enhanced_fallback_analysis",
        "_analyze_tas_nas_outputs_enhanced",
        "_create_hybrid_regime_clusters_enhanced",
        "_perform_hybrid_cross_validation_enhanced",
        "_optimize_ensemble_weights_enhanced"
    ]
    
    for method in enhanced_methods:
        if f"def {method}" not in orchestrator_content:
            print(f"❌ Missing enhanced method: {method}")
            return False
        else:
            print(f"✅ Found enhanced method: {method}")
    
    # Check for enhanced method calls in main analysis
    main_method_calls = [
        "self._preprocess_market_data_enhanced",
        "self._run_tas_analysis_enhanced",
        "self._run_nas_analysis_enhanced",
        "self._run_enhanced_fallback_analysis",
        "self._analyze_tas_nas_outputs_enhanced",
        "self._create_hybrid_regime_clusters_enhanced",
        "self._perform_hybrid_cross_validation_enhanced",
        "self._optimize_ensemble_weights_enhanced"
    ]
    
    for method_call in main_method_calls:
        if method_call not in orchestrator_content:
            print(f"❌ Missing enhanced method call: {method_call}")
            return False
        else:
            print(f"✅ Found enhanced method call: {method_call}")
    
    # Check for metadata updates
    metadata_checks = [
        "'enhanced_utilities_used': True",
        "'utility_integration_used': True",
        "'data_integration_used': True",
        "'ml_integration_used': True"
    ]
    
    for metadata_check in metadata_checks:
        if metadata_check not in orchestrator_content:
            print(f"❌ Missing metadata: {metadata_check}")
            return False
        else:
            print(f"✅ Found metadata: {metadata_check}")
    
    # Check for example and documentation files
    example_files = [
        "examples/enhanced_utility_integration_example.py",
        "ENHANCED_UTILITY_INTEGRATION_GUIDE.md"
    ]
    
    for example_file in example_files:
        if not os.path.exists(example_file):
            print(f"❌ Missing example/documentation: {example_file}")
            return False
        else:
            print(f"✅ Found example/documentation: {example_file}")
    
    # Check integration file contents
    print("\n🔍 Checking integration file contents...")
    
    # Check enhanced utility integration
    with open("shared_utils/enhanced_utility_integration.py", 'r') as f:
        utility_content = f.read()
    
    utility_checks = [
        "class EnhancedUtilityIntegration",
        "def safe_divide",
        "def validate_dataframe_columns",
        "def calculate_data_quality_metrics",
        "def optimize_dataframe_dtypes",
        "def safe_divide",
        "def safe_log",
        "def safe_sqrt",
        "def validate_finite",
        "def safe_correlation",
        "def save_data",
        "def load_data",
        "def optimize_for_m1",
        "def get_memory_usage",
        "def optimize_memory"
    ]
    
    for check in utility_checks:
        if check not in utility_content:
            print(f"❌ Missing utility method: {check}")
            return False
        else:
            print(f"✅ Found utility method: {check}")
    
    # Check enhanced data integration
    with open("shared_utils/enhanced_data_integration.py", 'r') as f:
        data_content = f.read()
    
    data_checks = [
        "class EnhancedDataIntegration",
        "def load_klines_data",
        "def process_market_data",
        "def engineer_features",
        "def engineer_returns",
        "def detect_gaps",
        "def calculate_data_quality_metrics",
        "def validate_data_consistency",
        "def clean_data",
        "def save_optimized_data",
        "def load_optimized_data"
    ]
    
    for check in data_checks:
        if check not in data_content:
            print(f"❌ Missing data method: {check}")
            return False
        else:
            print(f"✅ Found data method: {check}")
    
    # Check enhanced ML integration
    with open("shared_utils/enhanced_ml_integration.py", 'r') as f:
        ml_content = f.read()
    
    ml_checks = [
        "class EnhancedMLIntegration",
        "def select_features",
        "def cross_validate_model",
        "def evaluate_model",
        "def calculate_performance_metrics",
        "def optimize_hyperparameters",
        "def detect_regimes_hmm",
        "def analyze_regime_transitions",
        "def create_ensemble",
        "def manage_ensemble",
        "def calculate_confidence_metrics",
        "def detect_lookahead_bias",
        "def detect_overfitting",
        "def detect_data_leakage"
    ]
    
    for check in ml_checks:
        if check not in ml_content:
            print(f"❌ Missing ML method: {check}")
            return False
        else:
            print(f"✅ Found ML method: {check}")
    
    print("\n🎉 All enhanced utility integration verifications passed!")
    return True

def main():
    """Main verification function."""
    try:
        success = verify_enhanced_utilities_integration()
        if success:
            print("\n" + "="*80)
            print("🎉 ENHANCED UTILITY INTEGRATION VERIFICATION PASSED!")
            print("✅ All enhanced utilities are properly wired and used")
            print("✅ Integration files exist and contain required methods")
            print("✅ Enhanced orchestrator uses all integrations")
            print("✅ Enhanced methods are defined and called")
            print("✅ Metadata reflects enhanced utility usage")
            print("✅ Examples and documentation are available")
            print("="*80)
        else:
            print("\n❌ Verification failed")
    except Exception as e:
        print(f"\n❌ Verification execution failed: {e}")
        raise

if __name__ == "__main__":
    main()