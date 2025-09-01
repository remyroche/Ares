#!/usr/bin/env python3
"""
Demonstration Script for Enhanced Training Manager with Existing Decorators

This script demonstrates how to use the enhanced training manager that ensures:
1. Each step has thorough decorators using existing codebase decorators
2. Each step delivers detailed reports upon completion
3. All reports are stored consistently in a centralized location

Usage:
    python scripts/demo_enhanced_training_manager.py
"""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Dict, Any

# Import the enhanced training manager
import EnhancedTrainingManagerWithReporting,
    EnhancedTrainingManagerWithReporting,
    create_enhanced_training_manager_with_reporting
)
from src.utils.logger import system_logger


import async def load_config
async def load_config() -> Dict[str, Any]:
    """Load configuration for the enhanced training manager."""

    # Try to load from config file first
    config_path = Path("config/enhanced_reporting_config.yaml")
    if config_path.exists():
    pass
    pass
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        system_logger.info(f"📋 Loaded configuration from {config_path}")
    else:
        # Fallback to default configuration
        config = {
            "enhanced_training_manager": {
                "enhanced_training_interval": 3600,
                "max_enhanced_training_history": 100,
                "enable_model_training": True,
                "blank_training_mode": True,  # Use blank mode for demo
                "max_trials": 50,  # Reduced for demo
                "n_trials": 25,    # Reduced for demo
                "lookback_days": 7, # Reduced for demo
                "enable_validators": True,
                "enable_computational_optimization": True,
                "force_rerun": False
            },
            "enhanced_reporting": {
                "enable_detailed_reporting": True,
                "report_level": "detailed",
                "auto_cleanup_reports": True,
                "reports_retention_days": 30,
                "reports_directory": "reports/enhanced_training_pipeline"
            }
        }
        system_logger.info("📋 Using default configuration")

    return config


async def demonstrate_enhanced_training_manager():
    """Demonstrate the enhanced training manager with existing decorators."""

    system_logger.info("🚀 Starting Enhanced Training Manager Demonstration")
    system_logger.info("=" * 80)

    try:
        # Load configuration
    except Exception as e:
        pass
    except Exception as e:
        pass
        config = await load_config()

        # Create enhanced training manager
        system_logger.info("🔧 Creating Enhanced Training Manager with Reporting...")
        manager = await create_enhanced_training_manager_with_reporting(config)

        # Prepare training input
        training_input = {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "timeframe": "1m",
            "lookback_days": config["enhanced_training_manager"]["lookback_days"],
            "training_mode": "blank" if config["enhanced_training_manager"]["blank_training_mode"] else "full",
            "start_step": "step1_data_collection",
            "end_step": "step15_saving"
        }

        system_logger.info("📊 Training Input Configuration:")
        system_logger.info(f"   Symbol: {training_input['symbol']}")
        system_logger.info(f"   Exchange: {training_input['exchange']}")
        system_logger.info(f"   Timeframe: {training_input['timeframe']}")
        system_logger.info(f"   Lookback Days: {training_input['lookback_days']}")
        system_logger.info(f"   Training Mode: {training_input['training_mode']}")
        system_logger.info(f"   Start Step: {training_input['start_step']}")
        system_logger.info(f"   End Step: {training_input['end_step']}")

        # Execute enhanced training pipeline
        system_logger.info("🚀 Executing Enhanced Training Pipeline...")
        system_logger.info("   This will demonstrate:")
        system_logger.info("   1. Thorough decorators for each step")
        system_logger.info("   2. Detailed reports upon completion")
        system_logger.info("   3. Consistent storage in centralized location")
        system_logger.info("=" * 80)

        success = await manager.execute_enhanced_training(training_input)

        if success:
    pass
    pass
            system_logger.info("✅ Enhanced Training Pipeline completed successfully!")
            system_logger.info("📊 Reports have been generated and stored.")

            # Show report locations
            reports_dir = Path("reports/enhanced_training_pipeline")
            if reports_dir.exists():
    pass
    pass
                report_files = list(reports_dir.glob("*.json"))
                summary_files = list(reports_dir.glob("*_summary.txt"))

                system_logger.info("📁 Generated Reports:")
                system_logger.info(f"   📊 JSON Reports: {len(report_files)}")
                system_logger.info(f"   📋 Summary Reports: {len(summary_files)}")
                system_logger.info(f"   📂 Reports Directory: {reports_dir.absolute()}")

                # Show latest report
                if report_files:
    pass
    pass
                    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                    system_logger.info(f"   📄 Latest Report: {latest_report.name}")

                    # Show report content summary
                    try:
                        with open(latest_report, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)

    except Exception as e:
        pass
    except Exception as e:
        pass
                        system_logger.info("📊 Report Summary:")
                        system_logger.info(f"   Pipeline Success: {report_data.get('overall_success', 'N/A')}")
                        system_logger.info(f"   Start Time: {report_data.get('pipeline_start_time', 'N/A')}")
                        system_logger.info(f"   End Time: {report_data.get('pipeline_end_time', 'N/A')}")
                        system_logger.info(f"   Errors: {len(report_data.get('errors', []))}")
                        system_logger.info(f"   Warnings: {len(report_data.get('warnings', []))}")
                        system_logger.info(f"   Recommendations: {len(report_data.get('recommendations', []))}")

                    except Exception as e:
                        system_logger.warning(f"⚠️ Could not read report content: {e}")

        else:
            system_logger.error("❌ Enhanced Training Pipeline failed!")
            system_logger.info("📊 Check the reports directory for error details.")

        system_logger.info("=" * 80)
        system_logger.info("🎉 Enhanced Training Manager Demonstration Complete!")

    except Exception as e:
        system_logger.error(f"💥 Demonstration failed: {e}")
        system_logger.exception("Full error details:")
        raise


async def demonstrate_individual_steps():
    """Demonstrate individual step execution with decorators."""

    system_logger.info("🔧 Demonstrating Individual Step Execution with Decorators")
    system_logger.info("=" * 80)

    try:
        # Load configuration
    except Exception as e:
        pass
    except Exception as e:
        pass
        config = await load_config()

        # Create enhanced training manager
        manager = await create_enhanced_training_manager_with_reporting(config)

        # Demonstrate a few key steps
        steps_to_demo = [
            ("step1_data_collection", "Data Collection"),
            ("step2_feature_engineering", "Feature Engineering"),
            ("step3_hmm_regime_discovery", "HMM Regime Discovery")
        ]

        for step_method_name, step_description in steps_to_demo:
    pass
    pass
            system_logger.info(f"🔄 Demonstrating {step_description}...")

            try:
                # Get the step method
    except Exception as e:
        pass
    except Exception as e:
        pass
                step_method = getattr(manager, f"_execute_{step_method_name}_enhanced")

                # Execute with decorators
                result = await step_method(
                    symbol="BTCUSDT",
                    exchange="binance",
                    timeframe="1m",
                    data_dir="data_cache",
                    force_rerun=False,
                    feature_config={"vectorized_advanced_features": {}} if "feature_engineering" in step_method_name else None,
                    lookback_days=7 if "hmm" in step_method_name else None
                )

                if result:
    pass
    pass
                    system_logger.info(f"✅ {step_description} completed successfully")
                else:
                    system_logger.warning(f"⚠️ {step_description} completed with warnings")

            except Exception as e:
                system_logger.error(f"❌ {step_description} failed: {e}")
                # Continue with next step
                continue

        system_logger.info("=" * 80)
        system_logger.info("🎉 Individual Step Demonstration Complete!")

    except Exception as e:
        system_logger.error(f"💥 Individual step demonstration failed: {e}")
        system_logger.exception("Full error details:")


async def show_decorator_capabilities():
    """Show the capabilities of the existing decorators."""

    system_logger.info("🔍 Decorator Capabilities Overview")
    system_logger.info("=" * 80)

    system_logger.info("📋 Available Decorators:")
    system_logger.info("   1. @handle_errors - Comprehensive error handling and recovery")
    system_logger.info("   2. @monitor_pipeline_step - Step monitoring and validation")
    system_logger.info("   3. @validate_pipeline_input - Input validation and resource checks")
    system_logger.info("   4. @monitor_pipeline_performance - Performance monitoring")

    system_logger.info("")
    system_logger.info("🎯 Decorator Features:")
    system_logger.info("   ✅ Error handling with automatic retry and recovery")
    system_logger.info("   ✅ Memory and CPU usage monitoring")
    system_logger.info("   ✅ Data quality validation")
    system_logger.info("   ✅ Input parameter validation")
    system_logger.info("   ✅ Resource availability checks")
    system_logger.info("   ✅ Performance threshold warnings")
    system_logger.info("   ✅ Comprehensive logging and reporting")

    system_logger.info("")
    system_logger.info("📊 Pipeline Stages Supported:")
    system_logger.info("   📥 DATA_COLLECTION - Data gathering and ingestion")
    system_logger.info("   🔄 DATA_PREPROCESSING - Data cleaning and preparation")
    system_logger.info("   🧮 FEATURE_ENGINEERING - Feature creation and selection")
    system_logger.info("   🤖 MODEL_TRAINING - Model training and optimization")
    system_logger.info("   ✅ VALIDATION - Model validation and testing")
    system_logger.info("   ⚙️ OPTIMIZATION - Hyperparameter optimization")
    system_logger.info("   🚀 DEPLOYMENT - Model deployment and saving")

    system_logger.info("")
    system_logger.info("🔧 Validation Levels:")
    system_logger.info("   🟢 WARNING - Log issues but continue execution")
    system_logger.info("   🟡 STRICT - Stop on critical issues")
    system_logger.info("   🔵 SILENT - Only log summary information")
    system_logger.info("   📊 MONITOR - Monitor performance only")

    system_logger.info("=" * 80)


async def main():
    """Main demonstration function."""

    system_logger.info("🎯 Enhanced Training Manager with Existing Decorators")
    system_logger.info("=" * 80)
    system_logger.info("This demonstration shows how the enhanced training manager")
    system_logger.info("integrates existing decorators for comprehensive monitoring,")
    system_logger.info("detailed reporting, and consistent storage.")
    system_logger.info("=" * 80)

    # Show decorator capabilities
    await show_decorator_capabilities()

    # Ask user what to demonstrate
    print("\\\nWhat would you like to demonstrate?")
    print("1. Full pipeline execution with decorators")
    print("2. Individual step execution with decorators")
    print("3. Both")
    print("4. Exit")

    try:
        choice = input("Enter your choice (1-4): ").strip()

    except Exception as e:
        pass
    except Exception as e:
        pass
        if choice == "1":
    pass
    pass
            await demonstrate_enhanced_training_manager()
        elif choice == "2":
            await demonstrate_individual_steps()
        elif choice == "3":
            await demonstrate_enhanced_training_manager()
            print("\\\n" + "="*80 + "\\\n")
            await demonstrate_individual_steps()
        elif choice == "4":
            system_logger.info("👋 Goodbye!")
            return
        else:
            system_logger.warning("⚠️ Invalid choice, running full demonstration...")
            await demonstrate_enhanced_training_manager()

    except KeyboardInterrupt:
        system_logger.info("\\\n👋 Demonstration interrupted by user")
    except Exception as e:
        system_logger.error(f"💥 Demonstration failed: {e}")
        system_logger.exception("Full error details:")

    system_logger.info("=" * 80)
    system_logger.info("🎉 Demonstration Complete!")
    system_logger.info("📁 Check the reports directory for generated reports")
    system_logger.info("📋 Review the logs for detailed execution information")


if __name__ == "__main__":
    pass
    pass
    # Run the demonstration
    asyncio.run(main())