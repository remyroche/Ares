#!/usr/bin/env python3
"""
Liquidity Regime Optimization Workflow

This script provides a complete workflow that:
1. Runs automated configuration optimization using ml_liquidity_regime_step
2. Analyzes results using the existing liquidity_regime_run_selector.py
3. Provides a comprehensive report with recommendations

Usage:
    python3 scripts/liquidity_optimization_workflow.py \
        --symbol ETHUSDT \
        --exchange binance \
        --max-configs 20
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete liquidity regime optimization workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--max-configs", type=int, default=20, help="Max configurations to test")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Results directory")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip optimization, only analyze existing results")

    return parser.parse_args()


def run_automated_optimization(args: argparse.Namespace) -> bool:
    """Run the automated configuration optimization."""
    
    print("🔬 Step 1: Running Automated Configuration Optimization")
    print("-" * 60)
    
    cmd = [
        "python3", "scripts/automated_liquidity_optimizer.py",
        "--symbol", args.symbol,
        "--exchange", args.exchange,
        "--max-configs", str(args.max_configs),
        "--outcomes-dir", args.outcomes_dir,
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Optimization failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def run_result_analysis(args: argparse.Namespace) -> bool:
    """Run the existing liquidity_regime_run_selector to analyze results."""
    
    print("\n🔍 Step 2: Analyzing Results with Liquidity Regime Run Selector")
    print("-" * 60)
    
    cmd = [
        "python3", "scripts/liquidity_regime_run_selector.py",
        "--symbol", args.symbol,
        "--outcomes-dir", args.outcomes_dir,
        "--top-k", "10",
        "--required-regimes", "0,1,2,3",
        "--min-support-share", "0.01",
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def generate_workflow_report(args: argparse.Namespace) -> None:
    """Generate a comprehensive workflow report."""
    
    print("\n📋 Step 3: Generating Workflow Report")
    print("-" * 60)
    
    outcomes_dir = Path(args.outcomes_dir)
    
    # Find the latest optimization results
    optimization_files = sorted(outcomes_dir.glob(f"liquidity_config_optimization_{args.symbol}_*.csv"))
    analysis_files = sorted(outcomes_dir.glob(f"liquidity_config_optimization_{args.symbol}_*_analysis.json"))
    best_config_files = sorted(outcomes_dir.glob(f"liquidity_best_config_{args.symbol}_*.yaml"))
    
    print(f"\n📁 Generated Files:")
    
    if optimization_files:
        latest_optimization = optimization_files[-1]
        print(f"   📊 Detailed Results: {latest_optimization}")
    
    if analysis_files:
        latest_analysis = analysis_files[-1]
        print(f"   📈 Analysis Summary: {latest_analysis}")
    
    if best_config_files:
        latest_best_config = best_config_files[-1]
        print(f"   🏆 Best Configuration: {latest_best_config}")
    
    # Find existing liquidity cluster quality reports
    cluster_files = sorted(outcomes_dir.glob(f"liquidity_cluster_quality_{args.symbol}_*.csv"))
    
    if cluster_files:
        print(f"\n🔍 Existing Quality Reports (analyzed by run_selector):")
        for file in cluster_files[-5:]:  # Show last 5
            print(f"   📋 {file.name}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Review the best configuration in: {latest_best_config if best_config_files else 'N/A'}")
    print(f"   2. Test the best configuration in your production pipeline")
    print(f"   3. Monitor performance and iterate if needed")
    
    print(f"\n🎯 Recommendations:")
    print(f"   - Use the best configuration for your main trading strategy")
    print(f"   - Consider the top 3 configurations for different market conditions")
    print(f"   - Re-run optimization periodically to adapt to market changes")


def main() -> None:
    args = parse_args()
    
    print("🚀 Liquidity Regime Optimization Workflow")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Max Configurations: {args.max_configs}")
    print(f"Outcomes Directory: {args.outcomes_dir}")
    print("=" * 60)
    
    success = True
    
    # Step 1: Run automated optimization (unless skipped)
    if not args.skip_optimization:
        success &= run_automated_optimization(args)
    else:
        print("\n⏭️ Skipping optimization step (as requested)")
    
    # Step 2: Analyze results with existing selector
    success &= run_result_analysis(args)
    
    # Step 3: Generate workflow report
    generate_workflow_report(args)
    
    # Final status
    print("\n" + "=" * 60)
    if success:
        print("✅ Workflow completed successfully!")
        print(f"📁 Check {args.outcomes_dir}/ for all results and recommendations")
    else:
        print("❌ Workflow completed with errors - check messages above")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
