"""
Inference Prerequisites Verification Script.

This script checks all prerequisites for running inference in simulation mode:
1. Check if trained models exist in data/artifacts/ (find latest run_id)
2. Check if model_loader.py can load the models successfully
3. Check if data can be fetched from Binance (test connection)
4. Print a summary of what's available and what's missing

Usage:
    python -m extreme_price_movements.inference.verify_inference_prerequisites
"""

import sys
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

# Add project root to path
# For script run: /path/to/Ares/extreme_price_movements/inference/verify_inference_prerequisites.py
# We want: /path/to/Ares
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# Constants
DEFAULT_DATA_ROOT = "data"
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # Test symbols for Binance


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_status(check_name: str, status: bool, details: str = "") -> None:
    """Print check status with color indicators."""
    status_str = "✓ PASS" if status else "✗ FAIL"
    status_color = "\033[92m" if status else "\033[91m"
    reset_color = "\033[0m"
    
    print(f"  {status_color}{status_str}{reset_color}: {check_name}")
    if details:
        print(f"         {details}")


def check_artifacts_directory(data_root: str = DEFAULT_DATA_ROOT) -> Tuple[bool, Optional[str], List[str]]:
    """Check if artifacts directory exists and find latest run_id.
    
    Returns:
        Tuple of (success, latest_run_id, all_run_ids)
    """
    artifacts_dir = os.path.join(data_root, "artifacts")
    
    if not os.path.exists(artifacts_dir):
        return False, None, []
    
    import re
    run_pattern = re.compile(r"^\d{8}_\d{6}$")
    run_ids = []
    
    for name in os.listdir(artifacts_dir):
        if os.path.isdir(os.path.join(artifacts_dir, name)) and run_pattern.match(name):
            run_ids.append(name)
    
    if not run_ids:
        return False, None, []
    
    # Sort chronologically (newest first)
    run_ids.sort(reverse=True)
    latest_run_id = run_ids[0]
    
    return True, latest_run_id, run_ids


def check_model_artifact_structure(run_id: str, data_root: str = DEFAULT_DATA_ROOT) -> Dict[str, bool]:
    """Check what model artifacts are available for a given run_id.
    
    Returns:
        Dictionary with availability of each model type.
    """
    artifacts = {
        "run_id_exists": False,
        "native_models_dir": False,
        "long_mr_model": False,
        "long_tf_model": False,
        "short_mr_model": False,
        "short_tf_model": False,
        "trained_state": False,
        "ridge_weights": False,
        "bucket_params": False,
        "fs_reports": False,
        "labels_cache": False,
    }
    
    run_dir = os.path.join(data_root, "artifacts", run_id)
    
    if not os.path.exists(run_dir):
        return artifacts
    
    artifacts["run_id_exists"] = True
    
    # Check native models
    native_dir = os.path.join(run_dir, "models", "native")
    if os.path.exists(native_dir):
        artifacts["native_models_dir"] = True
        
        for model_name in os.listdir(native_dir):
            model_path = os.path.join(native_dir, model_name)
            if os.path.isdir(model_path):
                # Check if model file exists
                for f in os.listdir(model_path):
                    if f.startswith("model."):
                        if "long_mr" in model_name:
                            artifacts["long_mr_model"] = True
                        elif "long_tf" in model_name:
                            artifacts["long_tf_model"] = True
                        elif "short_mr" in model_name:
                            artifacts["short_mr_model"] = True
                        elif "short_tf" in model_name:
                            artifacts["short_tf_model"] = True
                        break
    
    # Check trained state
    trained_state_path = os.path.join(run_dir, "models", "trained_state.pkl")
    if os.path.exists(trained_state_path):
        artifacts["trained_state"] = True
    
    # Check ridge sizer weights
    ridge_path = os.path.join(run_dir, "ridge_sizer", "sizer_weights.json")
    if os.path.exists(ridge_path):
        artifacts["ridge_weights"] = True
    
    # Check bucket params
    bucket_params_path = os.path.join(run_dir, "models", "bucket_params.json")
    if os.path.exists(bucket_params_path):
        artifacts["bucket_params"] = True
    
    # Check fs_reports
    fs_reports_dir = os.path.join(run_dir, "fs_reports")
    if os.path.exists(fs_reports_dir):
        artifacts["fs_reports"] = True
    
    # Check labels cache
    labels_dir = os.path.join(run_dir, "labels")
    if os.path.exists(labels_dir):
        artifacts["labels_cache"] = True
    
    return artifacts


def check_model_loading(run_id: str, data_root: str = DEFAULT_DATA_ROOT) -> Tuple[bool, Dict[str, Any]]:
    """Check if model_loader can successfully load the models.
    
    Returns:
        Tuple of (success, result_details)
    """
    result = {
        "can_import_model_loader": False,
        "can_find_latest_run_id": False,
        "can_load_model_bundle": False,
        "can_load_bucket_params": False,
        "alpha_models_loaded": False,
        "meta_models_loaded": False,
        "spike_models_loaded": False,
        "ridge_weights_loaded": False,
        "details": {},
        "error": None,
    }

    try:
        # Import model_loader directly from file to avoid inference imports
        model_loader_path = os.path.join(project_root, "extreme_price_movements", "model_loader.py")
        
        # Use importlib to import without executing full module
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_loader", model_loader_path)
        if spec and spec.loader:
            model_loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_loader)
            result["can_import_model_loader"] = True
        
        # Check find_latest_run_id
        found_run_id = model_loader.find_latest_run_id(data_root)
        result["can_find_latest_run_id"] = found_run_id is not None
        result["details"]["found_run_id"] = found_run_id
        
        if found_run_id != run_id:
            result["error"] = f"Found run_id ({found_run_id}) differs from expected ({run_id})"
        
        # Try to load model bundle
        try:
            bundle = model_loader.load_model_bundle(run_id, data_root)
            result["can_load_model_bundle"] = True
            
            # Check what's in the bundle
            alpha = bundle.get("alpha_models", {})
            result["alpha_models_loaded"] = bool(alpha)
            result["details"]["alpha_models"] = {
                "long_mr": bool(alpha.get("long", {}).get("mr")),
                "long_tf": bool(alpha.get("long", {}).get("tf")),
                "short_mr": bool(alpha.get("short", {}).get("mr")),
                "short_tf": bool(alpha.get("short", {}).get("tf")),
            }
            
            result["meta_models_loaded"] = bool(bundle.get("meta_models"))
            result["spike_models_loaded"] = bool(bundle.get("spike_models"))
            result["ridge_weights_loaded"] = bool(bundle.get("ridge_weights"))
            
            result["details"]["meta_models_count"] = len(bundle.get("meta_models", {}))
            result["details"]["spike_models_count"] = len(bundle.get("spike_models", {}))
            
        except Exception as e:
            result["error"] = f"Failed to load model bundle: {str(e)}"
        
        # Try to load bucket params
        try:
            bucket_params = model_loader.load_bucket_params(run_id, data_root)
            result["can_load_bucket_params"] = bool(bucket_params)
            result["details"]["bucket_params_keys"] = list(bucket_params.keys()) if bucket_params else []
        except Exception as e:
            result["details"]["bucket_params_error"] = str(e)
        
    except Exception as e:
        result["error"] = f"Import error: {str(e)}"
    
    success = (
        result["can_import_model_loader"] and
        result["can_find_latest_run_id"] and
        result["can_load_model_bundle"]
    )
    
    return success, result


def check_binance_connection(test_symbols: List[str] = DEFAULT_SYMBOLS) -> Tuple[bool, Dict[str, Any]]:
    """Check if Binance API connection works.
    
    Returns:
        Tuple of (success, result_details)
    """
    result = {
        "can_import_ccxt": False,
        "can_create_exchange": False,
        "can_fetch_time": False,
        "can_fetch_ticker": False,
        "can_fetch_ohlcv": False,
        "test_symbols": test_symbols,
        "details": {},
        "error": None,
    }
    
    try:
        import ccxt
        result["can_import_ccxt"] = True
        
        try:
            # Create Binance exchange instance
            exchange = ccxt.binance({"enableRateLimit": True})
            result["can_create_exchange"] = True
            
            # Try to fetch server time
            try:
                time_data = exchange.fetch_time()
                result["can_fetch_time"] = True
                result["details"]["server_time"] = time_data
            except Exception as e:
                result["details"]["time_error"] = str(e)
            
            # Try to fetch ticker for test symbol
            try:
                ticker = exchange.fetch_ticker(test_symbols[0])
                result["can_fetch_ticker"] = True
                result["details"]["ticker"] = {
                    "symbol": ticker.get("symbol"),
                    "last": ticker.get("last"),
                    "volume": ticker.get("quoteVolume"),
                }
            except Exception as e:
                result["details"]["ticker_error"] = str(e)
            
            # Try to fetch OHLCV
            try:
                ohlcv = exchange.fetch_ohlcv(test_symbols[0], "1h", limit=10)
                result["can_fetch_ohlcv"] = len(ohlcv) > 0
                result["details"]["ohlcv_count"] = len(ohlcv)
                if ohlcv:
                    result["details"]["ohlcv_latest"] = {
                        "timestamp": ohlcv[-1][0],
                        "close": ohlcv[-1][4],
                    }
            except Exception as e:
                result["details"]["ohlcv_error"] = str(e)
                
        except Exception as e:
            result["error"] = f"Failed to create exchange: {str(e)}"
            
    except ImportError as e:
        result["error"] = f"Failed to import ccxt: {str(e)}"
    
    success = result["can_import_ccxt"] and result["can_create_exchange"]
    
    return success, result


def check_local_data(data_root: str = DEFAULT_DATA_ROOT, test_symbols: List[str] = None) -> Dict[str, Any]:
    """Check if local OHLCV data exists.
    
    Returns:
        Dictionary with local data availability.
    """
    if test_symbols is None:
        test_symbols = DEFAULT_SYMBOLS
    
    result = {
        "data_root": data_root,
        "ohlcv_dir_exists": False,
        "available_symbols": [],
        "sample_data_exists": False,
    }
    
    ohlcv_dir = os.path.join(data_root, "ohlcv")
    if os.path.exists(ohlcv_dir):
        result["ohlcv_dir_exists"] = True
        
        # List available symbols
        for f in os.listdir(ohlcv_dir):
            if f.endswith(".meta.json"):
                symbol = f.replace(".meta.json", "").replace("_", "/")
                result["available_symbols"].append(symbol)
        
        # Check if any of our test symbols have data
        for symbol in test_symbols:
            symbol_file = symbol.replace("/", "_") + ".meta.json"
            if symbol_file in os.listdir(ohlcv_dir):
                result["sample_data_exists"] = True
                break
    
    return result


def print_summary(
    artifacts_ok: bool,
    run_id: Optional[str],
    all_run_ids: List[str],
    model_structure: Dict[str, bool],
    model_loading_result: Dict[str, Any],
    binance_result: Dict[str, Any],
    local_data: Dict[str, Any],
) -> None:
    """Print a comprehensive summary of the verification results."""
    
    print_section("PREREQUISITES VERIFICATION SUMMARY")
    
    # Overall status
    all_checks_passed = (
        artifacts_ok and
        model_loading_result.get("can_load_model_bundle", False) and
        binance_result.get("can_create_exchange", False)
    )
    
    overall_status = "READY FOR INFERENCE" if all_checks_passed else "NOT READY"
    status_color = "\033[92m" if all_checks_passed else "\033[91m"
    reset_color = "\033[0m"
    
    print(f"\n  Overall Status: {status_color}{overall_status}{reset_color}")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    # Artifacts check
    print_section("1. MODEL ARTIFACTS")
    print_status("Artifacts directory exists", artifacts_ok)
    if run_id:
        print(f"  Latest run_id: {run_id}")
        if all_run_ids:
            print(f"  All available run_ids: {', '.join(all_run_ids[:5])}{'...' if len(all_run_ids) > 5 else ''}")
    
    # Model structure
    print_section("2. MODEL STRUCTURE")
    print_status("Run ID exists", model_structure.get("run_id_exists", False))
    print_status("Native models directory", model_structure.get("native_models_dir", False))
    print_status("  - long_mr model", model_structure.get("long_mr_model", False))
    print_status("  - long_tf model", model_structure.get("long_tf_model", False))
    print_status("  - short_mr model", model_structure.get("short_mr_model", False))
    print_status("  - short_tf model", model_structure.get("short_tf_model", False))
    print_status("Trained state (pickle)", model_structure.get("trained_state", False))
    print_status("Ridge sizer weights", model_structure.get("ridge_weights", False))
    print_status("Bucket params", model_structure.get("bucket_params", False))
    print_status("Feature selection reports", model_structure.get("fs_reports", False))
    print_status("Labels cache", model_structure.get("labels_cache", False))
    
    # Model loading
    print_section("3. MODEL LOADING")
    print_status("Can import model_loader", model_loading_result.get("can_import_model_loader", False))
    print_status("Can find latest run_id", model_loading_result.get("can_find_latest_run_id", False))
    print_status("Can load model bundle", model_loading_result.get("can_load_model_bundle", False))
    
    if model_loading_result.get("details"):
        details = model_loading_result["details"]
        if "alpha_models" in details:
            print("\n  Alpha models loaded:")
            for model_type, loaded in details["alpha_models"].items():
                print(f"    - {model_type}: {'Yes' if loaded else 'No'}")
        
        if "meta_models_count" in details:
            print(f"\n  Meta models count: {details['meta_models_count']}")
        
        if "spike_models_count" in details:
            print(f"  Spike models count: {details['spike_models_count']}")
        
        print_status(
            "Ridge weights loaded",
            model_loading_result.get("ridge_weights_loaded", False)
        )
        print_status(
            "Bucket params loaded",
            model_loading_result.get("can_load_bucket_params", False)
        )
    
    if model_loading_result.get("error"):
        print(f"\n  Error: {model_loading_result['error']}")
    
    # Binance connection
    print_section("4. BINANCE CONNECTION")
    print_status("Can import ccxt", binance_result.get("can_import_ccxt", False))
    print_status("Can create exchange", binance_result.get("can_create_exchange", False))
    print_status("Can fetch server time", binance_result.get("can_fetch_time", False))
    print_status("Can fetch ticker", binance_result.get("can_fetch_ticker", False))
    print_status("Can fetch OHLCV", binance_result.get("can_fetch_ohlcv", False))
    
    if binance_result.get("details"):
        details = binance_result["details"]
        if "ticker" in details:
            ticker = details["ticker"]
            print(f"\n  Sample ticker ({ticker['symbol']}): ${ticker['last']:.2f}")
        if "ohlcv_count" in details:
            print(f"  OHLCV bars fetched: {details['ohlcv_count']}")
    
    if binance_result.get("error"):
        print(f"\n  Error: {binance_result['error']}")
    
    # Local data
    print_section("5. LOCAL DATA")
    print_status("OHLCV directory exists", local_data.get("ohlcv_dir_exists", False))
    print_status("Sample data available", local_data.get("sample_data_exists", False))
    
    if local_data.get("available_symbols"):
        symbols = local_data["available_symbols"]
        print(f"\n  Available symbols: {len(symbols)}")
        print(f"  Sample: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    
    # Final recommendations
    print_section("RECOMMENDATIONS")
    
    if all_checks_passed:
        print("\n  ✓ All prerequisites met! You can run inference.")
        print("  Run with: python -m extreme_price_movements.inference.run_inference")
    else:
        print("\n  ✗ Some prerequisites are missing:")
        
        if not artifacts_ok:
            print("    - No trained model artifacts found. Run training first.")
        
        if not model_loading_result.get("can_load_model_bundle", False):
            print("    - Cannot load model bundle. Check model files.")
        
        if not binance_result.get("can_create_exchange", False):
            print("    - Cannot connect to Binance. Check internet/API access.")
        
        print("\n  Address the issues above and run this script again.")
    
    print("\n" + "=" * 60)


def main():
    """Main verification function."""
    print("\n" + "=" * 60)
    print("  INFERENCE PREREQUISITES VERIFICATION")
    print("  Checking all requirements for simulation mode")
    print("=" * 60)
    
    data_root = DEFAULT_DATA_ROOT
    
    # 1. Check artifacts directory
    print_section("CHECKING ARTIFACTS")
    artifacts_ok, run_id, all_run_ids = check_artifacts_directory(data_root)
    print_status("Artifacts directory found", artifacts_ok)
    if run_id:
        print(f"  Found run_id: {run_id}")
    else:
        print("  No run_id found!")
    
    # 2. Check model structure
    model_structure = {}
    if run_id:
        model_structure = check_model_artifact_structure(run_id, data_root)
    
    # 3. Check model loading
    model_loading_result = {"error": "No run_id found"}
    if run_id:
        _, model_loading_result = check_model_loading(run_id, data_root)
    
    # 4. Check Binance connection
    binance_ok, binance_result = check_binance_connection()
    
    # 5. Check local data
    local_data = check_local_data(data_root)
    
    # Print summary
    print_summary(
        artifacts_ok=artifacts_ok,
        run_id=run_id,
        all_run_ids=all_run_ids,
        model_structure=model_structure,
        model_loading_result=model_loading_result,
        binance_result=binance_result,
        local_data=local_data,
    )
    
    # Return exit code
    all_ok = (
        artifacts_ok and
        model_loading_result.get("can_load_model_bundle", False) and
        binance_ok
    )
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
