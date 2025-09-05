#!/usr/bin/env python3
"""
Simple pipeline runner that handles errors gracefully and provides a summary.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add the code_quality directory to the path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))

def run_pipeline_with_error_handling():
    """Run the pipeline with comprehensive error handling."""
    print("="*80)
    print("UNIFIED ENHANCED PIPELINE - SIMPLIFIED RUNNER")
    print("="*80)
    
    start_time = time.time()
    results = {}
    
    try:
        from pipelines.pipeline_unified_enhanced import UnifiedEnhancedPipeline
        
        # Initialize pipeline
        pipeline = UnifiedEnhancedPipeline(
            project_root="/workspace/src",
            enable_plugins=False  # Disable plugins to avoid import issues
        )
        
        print(f"Pipeline initialized successfully")
        print(f"Project root: {pipeline.project_root}")
        print(f"Reports directory: {pipeline.reports_dir}")
        
        # Run basic analysis only
        print("\n" + "="*60)
        print("Running Basic Analysis")
        print("="*60)
        
        # Syntax validation
        try:
            print("Running syntax validation...")
            results["syntax_validation"] = pipeline.run_syntax_validation()
            print("✓ Syntax validation completed")
        except Exception as e:
            print(f"✗ Syntax validation failed: {e}")
            results["syntax_validation"] = {"error": str(e)}
        
        # Import validation
        try:
            print("Running import validation...")
            results["import_validation"] = pipeline.run_import_validation()
            print("✓ Import validation completed")
        except Exception as e:
            print(f"✗ Import validation failed: {e}")
            results["import_validation"] = {"error": str(e)}
        
        # Circular imports detection
        try:
            print("Detecting circular imports...")
            results["circular_imports"] = pipeline.detect_circular_imports()
            print("✓ Circular imports detection completed")
        except Exception as e:
            print(f"✗ Circular imports detection failed: {e}")
            results["circular_imports"] = {"error": str(e)}
        
        # Enhanced undefined names analysis (with error handling)
        try:
            print("Running enhanced undefined names analysis...")
            results["undefined_names"] = pipeline.run_enhanced_undefined_names_analysis()
            print("✓ Enhanced undefined names analysis completed")
        except Exception as e:
            print(f"✗ Enhanced undefined names analysis failed: {e}")
            results["undefined_names"] = {"error": str(e)}
        
        # Enhanced dependency analysis
        try:
            print("Running enhanced dependency analysis...")
            results["dependency_analysis"] = pipeline.run_enhanced_dependency_analysis()
            print("✓ Enhanced dependency analysis completed")
        except Exception as e:
            print(f"✗ Enhanced dependency analysis failed: {e}")
            results["dependency_analysis"] = {"error": str(e)}
        
        # Function validation
        try:
            print("Running function validation...")
            results["function_validation"] = pipeline.run_function_validation()
            print("✓ Function validation completed")
        except Exception as e:
            print(f"✗ Function validation failed: {e}")
            results["function_validation"] = {"error": str(e)}
        
        # Enhanced validation
        try:
            print("Running enhanced validation...")
            results["enhanced_validation"] = pipeline.run_enhanced_validation()
            print("✓ Enhanced validation completed")
        except Exception as e:
            print(f"✗ Enhanced validation failed: {e}")
            results["enhanced_validation"] = {"error": str(e)}
        
        # Metrics analysis
        try:
            print("Running metrics analysis...")
            results["metrics"] = pipeline.run_metrics_analysis()
            print("✓ Metrics analysis completed")
        except Exception as e:
            print(f"✗ Metrics analysis failed: {e}")
            results["metrics"] = {"error": str(e)}
        
        # Test coverage analysis
        try:
            print("Running test coverage analysis...")
            results["test_coverage"] = pipeline.run_test_coverage_analysis()
            print("✓ Test coverage analysis completed")
        except Exception as e:
            print(f"✗ Test coverage analysis failed: {e}")
            results["test_coverage"] = {"error": str(e)}
        
        # Code smell detection
        try:
            print("Running code smell detection...")
            results["code_smells"] = pipeline.run_code_smell_detection()
            print("✓ Code smell detection completed")
        except Exception as e:
            print(f"✗ Code smell detection failed: {e}")
            results["code_smells"] = {"error": str(e)}
        
        # Documentation analysis
        try:
            print("Running documentation analysis...")
            results["documentation"] = pipeline.run_documentation_analysis()
            print("✓ Documentation analysis completed")
        except Exception as e:
            print(f"✗ Documentation analysis failed: {e}")
            results["documentation"] = {"error": str(e)}
        
        # Configuration analysis
        try:
            print("Running configuration analysis...")
            results["configuration"] = pipeline.run_configuration_analysis()
            print("✓ Configuration analysis completed")
        except Exception as e:
            print(f"✗ Configuration analysis failed: {e}")
            results["configuration"] = {"error": str(e)}
        
        # Data flow analysis
        try:
            print("Running data flow analysis...")
            results["data_flow"] = pipeline.run_data_flow_analysis()
            print("✓ Data flow analysis completed")
        except Exception as e:
            print(f"✗ Data flow analysis failed: {e}")
            results["data_flow"] = {"error": str(e)}
        
        # Static analysis
        try:
            print("Running static analysis...")
            results["static_analysis"] = pipeline.run_static_analysis()
            print("✓ Static analysis completed")
        except Exception as e:
            print(f"✗ Static analysis failed: {e}")
            results["static_analysis"] = {"error": str(e)}
        
        # AST analysis
        try:
            print("Running AST analysis...")
            results["ast_analysis"] = pipeline.run_ast_analysis()
            print("✓ AST analysis completed")
        except Exception as e:
            print(f"✗ AST analysis failed: {e}")
            results["ast_analysis"] = {"error": str(e)}
        
        # Dead code analysis
        try:
            print("Running dead code analysis...")
            results["dead_code"] = pipeline.run_dead_code_analysis()
            print("✓ Dead code analysis completed")
        except Exception as e:
            print(f"✗ Dead code analysis failed: {e}")
            results["dead_code"] = {"error": str(e)}
        
        # Performance analysis
        try:
            print("Running performance analysis...")
            results["performance"] = pipeline.run_performance_analysis()
            print("✓ Performance analysis completed")
        except Exception as e:
            print(f"✗ Performance analysis failed: {e}")
            results["performance"] = {"error": str(e)}
        
        # Security analysis
        try:
            print("Running security analysis...")
            results["security"] = pipeline.run_security_analysis()
            print("✓ Security analysis completed")
        except Exception as e:
            print(f"✗ Security analysis failed: {e}")
            results["security"] = {"error": str(e)}
        
        # Architecture analysis
        try:
            print("Running architecture analysis...")
            results["architecture"] = pipeline.run_architecture_analysis()
            print("✓ Architecture analysis completed")
        except Exception as e:
            print(f"✗ Architecture analysis failed: {e}")
            results["architecture"] = {"error": str(e)}
        
        # Comprehensive review
        try:
            print("Running comprehensive review...")
            results["comprehensive_review"] = pipeline.run_comprehensive_review()
            print("✓ Comprehensive review completed")
        except Exception as e:
            print(f"✗ Comprehensive review failed: {e}")
            results["comprehensive_review"] = {"error": str(e)}
        
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        results["pipeline_error"] = str(e)
    
    # Generate summary
    total_time = time.time() - start_time
    results["summary"] = {
        "total_execution_time": total_time,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "successful_analyses": len([k for k, v in results.items() if k != "summary" and v is not None and "error" not in v]),
        "failed_analyses": len([k for k, v in results.items() if k != "summary" and v is not None and "error" in v]),
        "total_analyses": len([k for k in results.keys() if k != "summary"])
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("/workspace/code_quality/reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Convert any remaining non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return {k: make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    report_path = reports_dir / f"pipeline_results_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Successful analyses: {results['summary']['successful_analyses']}")
    print(f"Failed analyses: {results['summary']['failed_analyses']}")
    print(f"Total analyses: {results['summary']['total_analyses']}")
    print(f"Report saved to: {report_path}")
    
    # Print detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    for analysis_name, result in results.items():
        if analysis_name == "summary":
            continue
        
        if "error" in result:
            print(f"✗ {analysis_name}: FAILED - {result['error']}")
        else:
            print(f"✓ {analysis_name}: SUCCESS")
            if "execution_time" in result:
                print(f"  Execution time: {result['execution_time']:.2f}s")
            if "total_issues" in result:
                print(f"  Total issues: {result['total_issues']}")
            if "total_files" in result:
                print(f"  Total files: {result['total_files']}")
    
    return results

if __name__ == "__main__":
    run_pipeline_with_error_handling()