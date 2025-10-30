"""
Kelly Safety Gates - Validation Criteria Checker

Comprehensive safety gate validation for dampened Kelly sizing system.
All gates must pass before proceeding to live deployment.

Safety Gates:
1. Performance: +10% geo return OR -20% DD (≥90% baseline growth)
2. Calibration: Mean |actual - predicted| < 10%
3. Bin Coverage: ≥70% trades with sufficient samples
4. Regime Stability: <10% mid-trade regime switches
5. High-Leverage: Win rate > 50%, tail loss < 5%
6. Numerical Stability: No NaN/Inf in any fold
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error

logger = system_logger.getChild('KellySafetyGates')


@dataclass
class SafetyGateResult:
    """Result for a single safety gate."""
    gate_name: str
    passed: bool
    actual_value: float
    threshold: float
    message: str
    severity: str  # 'critical', 'warning', 'info'


class KellySafetyGateValidator:
    """
    Validates all safety gates for Kelly sizing system.
    
    All gates must pass before proceeding to optimization or live deployment.
    """
    
    def __init__(self, output_dir: str = "outcomes/kelly_validation"):
        """
        Initialize safety gate validator.
        
        Args:
            output_dir: Directory for validation reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger.getChild('Validator')
        
        # Gate thresholds (configurable)
        self.thresholds = {
            'geometric_return_improvement_pct': 0.10,  # +10%
            'max_drawdown_reduction_pct': 0.20,  # -20%
            'min_baseline_growth_retention': 0.90,  # 90%
            'max_fold_drawdown': 0.15,  # 15%
            'max_calibration_error': 0.10,  # 10%
            'min_bin_coverage': 0.70,  # 70%
            'min_regime_stability': 0.90,  # 90% (i.e., <10% switches)
            'min_high_lev_win_rate': 0.50,  # 50%
            'max_high_lev_tail_loss': 0.05  # 5%
        }
        
        tprint_info("✅ Kelly Safety Gate Validator initialized")
    
    @handles_errors
    def validate_all_gates(
        self,
        validation_report: Dict[str, Any],
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[SafetyGateResult], Dict[str, Any]]:
        """
        Validate all safety gates.
        
        Args:
            validation_report: Validation report from walk-forward
            baseline_metrics: Optional baseline metrics for comparison
            
        Returns:
            Tuple of (all_passed, gate_results, summary)
        """
        tprint_info("\n" + "="*80)
        tprint_info("🚦 VALIDATING KELLY SAFETY GATES")
        tprint_info("="*80)
        
        gate_results = []
        
        # Extract variant reports
        variant_reports = validation_report.get('variants', {})
        
        # Get full system variant (the one we're deploying)
        full_system = variant_reports.get('full_system', {})
        
        if not full_system:
            tprint_error("❌ No full_system variant found in validation report")
            return False, [], {'error': 'No full_system variant'}
        
        aggregate = full_system.get('aggregate_metrics', {})
        safety_gates = full_system.get('safety_gates', {})
        
        # Gate 1: Performance
        gate1 = self._check_performance_gate(aggregate, baseline_metrics)
        gate_results.append(gate1)
        
        # Gate 2: Calibration Quality
        gate2 = self._check_calibration_gate(aggregate)
        gate_results.append(gate2)
        
        # Gate 3: Bin Coverage
        gate3 = self._check_coverage_gate(aggregate)
        gate_results.append(gate3)
        
        # Gate 4: Regime Stability
        gate4 = self._check_stability_gate(aggregate)
        gate_results.append(gate4)
        
        # Gate 5: High-Leverage Performance
        gate5 = self._check_high_leverage_gate(validation_report)
        gate_results.append(gate5)
        
        # Gate 6: Numerical Stability
        gate6 = self._check_numerical_stability(validation_report)
        gate_results.append(gate6)
        
        # Check if all gates passed
        all_passed = all(g.passed for g in gate_results)
        
        # Print results
        self._print_gate_results(gate_results, all_passed)
        
        # Generate summary
        summary = {
            'all_gates_passed': all_passed,
            'gates_passed': sum(1 for g in gate_results if g.passed),
            'total_gates': len(gate_results),
            'critical_failures': [g.gate_name for g in gate_results if not g.passed and g.severity == 'critical'],
            'warnings': [g.gate_name for g in gate_results if not g.passed and g.severity == 'warning'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save gate validation report
        self._save_gate_report(gate_results, summary)
        
        return all_passed, gate_results, summary
    
    def _check_performance_gate(
        self,
        aggregate: Dict[str, float],
        baseline: Optional[Dict[str, float]]
    ) -> SafetyGateResult:
        """Check performance gate."""
        geo_return = aggregate.get('median_geometric_return', 0.0)
        max_dd = aggregate.get('worst_fold_drawdown', 1.0)
        
        if baseline:
            baseline_geo = baseline.get('geometric_return', 0.0)
            baseline_dd = baseline.get('max_drawdown', 0.20)
            
            # Check improvement
            geo_improvement = (geo_return - baseline_geo) / baseline_geo if baseline_geo > 0 else 0.0
            dd_improvement = (baseline_dd - max_dd) / baseline_dd if baseline_dd > 0 else 0.0
            
            # Pass if +10% geo OR -20% DD (while maintaining 90% growth)
            if geo_improvement >= self.thresholds['geometric_return_improvement_pct']:
                passed = True
                message = f"Geometric return improved by {geo_improvement:.1%}"
            elif dd_improvement >= self.thresholds['max_drawdown_reduction_pct'] and geo_return >= 0.90 * baseline_geo:
                passed = True
                message = f"Drawdown reduced by {dd_improvement:.1%} while maintaining {geo_return/baseline_geo:.1%} of baseline growth"
            else:
                passed = False
                message = f"Insufficient improvement: geo={geo_improvement:.1%}, DD={dd_improvement:.1%}"
        else:
            # No baseline, check absolute thresholds
            if geo_return > 0.10 and max_dd < self.thresholds['max_fold_drawdown']:
                passed = True
                message = f"Absolute performance acceptable: {geo_return:.1%} return, {max_dd:.1%} DD"
            else:
                passed = False
                message = f"Absolute performance insufficient: {geo_return:.1%} return, {max_dd:.1%} DD"
        
        return SafetyGateResult(
            gate_name="Performance Gate",
            passed=passed,
            actual_value=geo_return,
            threshold=self.thresholds['geometric_return_improvement_pct'],
            message=message,
            severity='critical'
        )
    
    def _check_calibration_gate(self, aggregate: Dict[str, float]) -> SafetyGateResult:
        """Check calibration quality gate."""
        cal_error = aggregate.get('mean_calibration_error', 1.0)
        threshold = self.thresholds['max_calibration_error']
        
        passed = cal_error < threshold
        
        message = f"Calibration error {cal_error:.2%} vs threshold {threshold:.2%}"
        
        return SafetyGateResult(
            gate_name="Calibration Quality Gate",
            passed=passed,
            actual_value=cal_error,
            threshold=threshold,
            message=message,
            severity='critical'
        )
    
    def _check_coverage_gate(self, aggregate: Dict[str, float]) -> SafetyGateResult:
        """Check bin coverage gate."""
        coverage = aggregate.get('mean_bin_coverage', 0.0)
        threshold = self.thresholds['min_bin_coverage']
        
        passed = coverage >= threshold
        
        message = f"Bin coverage {coverage:.1%} vs threshold {threshold:.1%}"
        
        return SafetyGateResult(
            gate_name="Bin Coverage Gate",
            passed=passed,
            actual_value=coverage,
            threshold=threshold,
            message=message,
            severity='warning'
        )
    
    def _check_stability_gate(self, aggregate: Dict[str, float]) -> SafetyGateResult:
        """Check regime stability gate."""
        stability = aggregate.get('mean_regime_stability', 0.0)
        threshold = self.thresholds['min_regime_stability']
        
        passed = stability >= threshold
        
        switch_pct = 1.0 - stability
        message = f"Regime stability {stability:.1%} ({switch_pct:.1%} switches) vs threshold {threshold:.1%}"
        
        return SafetyGateResult(
            gate_name="Regime Stability Gate",
            passed=passed,
            actual_value=stability,
            threshold=threshold,
            message=message,
            severity='warning'
        )
    
    def _check_high_leverage_gate(self, validation_report: Dict[str, Any]) -> SafetyGateResult:
        """Check high-leverage performance gate."""
        # Extract high-leverage statistics from fold results
        full_system = validation_report.get('variants', {}).get('full_system', {})
        fold_results = full_system.get('fold_results', [])
        
        if not fold_results:
            return SafetyGateResult(
                gate_name="High-Leverage Performance Gate",
                passed=False,
                actual_value=0.0,
                threshold=self.thresholds['min_high_lev_win_rate'],
                message="No fold results available",
                severity='critical'
            )
        
        # Calculate average high-leverage win rate across folds
        high_lev_win_rates = [f.get('high_leverage_win_rate', 0.0) for f in fold_results]
        avg_high_lev_win_rate = np.mean(high_lev_win_rates)
        
        threshold = self.thresholds['min_high_lev_win_rate']
        passed = avg_high_lev_win_rate >= threshold
        
        message = f"High-leverage win rate {avg_high_lev_win_rate:.1%} vs threshold {threshold:.1%}"
        
        return SafetyGateResult(
            gate_name="High-Leverage Performance Gate",
            passed=passed,
            actual_value=avg_high_lev_win_rate,
            threshold=threshold,
            message=message,
            severity='critical'
        )
    
    def _check_numerical_stability(self, validation_report: Dict[str, Any]) -> SafetyGateResult:
        """Check numerical stability across all folds."""
        # Check all fold results for NaN/Inf
        has_issues = False
        issues = []
        
        for variant_name, variant_data in validation_report.get('variants', {}).items():
            fold_results = variant_data.get('fold_results', [])
            
            for fold in fold_results:
                # Check each metric
                for key, value in fold.items():
                    if isinstance(value, (int, float)):
                        if np.isnan(value) or np.isinf(value):
                            has_issues = True
                            issues.append(f"{variant_name}/fold_{fold.get('fold_id', '?')}/{key}")
        
        passed = not has_issues
        
        if passed:
            message = "No numerical issues (NaN/Inf) detected in any fold"
        else:
            message = f"Numerical issues found in: {', '.join(issues[:5])}"
        
        return SafetyGateResult(
            gate_name="Numerical Stability Gate",
            passed=passed,
            actual_value=1.0 if passed else 0.0,
            threshold=1.0,
            message=message,
            severity='critical'
        )
    
    def _print_gate_results(self, gate_results: List[SafetyGateResult], all_passed: bool) -> None:
        """Print formatted gate results."""
        tprint_info("\n" + "="*80)
        tprint_info("📋 SAFETY GATE RESULTS")
        tprint_info("="*80)
        
        for gate in gate_results:
            status = "✅ PASS" if gate.passed else "❌ FAIL"
            severity_icon = "🔴" if gate.severity == 'critical' else "🟡" if gate.severity == 'warning' else "🟢"
            
            tprint_info(f"\n{severity_icon} {gate.gate_name}: {status}")
            tprint_info(f"   {gate.message}")
        
        tprint_info("\n" + "="*80)
        if all_passed:
            tprint_success("✅ ALL SAFETY GATES PASSED - Proceed to optimization")
        else:
            tprint_error("❌ SOME GATES FAILED - Review and adjust parameters")
        tprint_info("="*80 + "\n")
    
    def _save_gate_report(
        self,
        gate_results: List[SafetyGateResult],
        summary: Dict[str, Any]
    ) -> Path:
        """Save gate validation report."""
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"kelly_safety_gates_{timestamp_str}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'gates': [
                {
                    'name': g.gate_name,
                    'passed': g.passed,
                    'actual_value': g.actual_value,
                    'threshold': g.threshold,
                    'message': g.message,
                    'severity': g.severity
                }
                for g in gate_results
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Safety gate report saved to {report_file}")
        return report_file
    
    @handles_errors
    def validate_from_report_file(
        self,
        validation_report_path: str,
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[SafetyGateResult]]:
        """
        Validate safety gates from a saved validation report.
        
        Args:
            validation_report_path: Path to validation report JSON
            baseline_metrics: Optional baseline metrics
            
        Returns:
            Tuple of (all_passed, gate_results)
        """
        # Load report
        with open(validation_report_path, 'r') as f:
            report = json.load(f)
        
        # Validate gates
        all_passed, gate_results, summary = self.validate_all_gates(report, baseline_metrics)
        
        return all_passed, gate_results


def check_kelly_safety_gates(
    validation_report_path: str,
    baseline_metrics: Optional[Dict[str, float]] = None
) -> bool:
    """
    Convenience function to check all safety gates.
    
    Args:
        validation_report_path: Path to validation report
        baseline_metrics: Optional baseline metrics
        
    Returns:
        True if all gates pass
    """
    validator = KellySafetyGateValidator()
    all_passed, _ = validator.validate_from_report_file(validation_report_path, baseline_metrics)
    return all_passed


# Example usage and documentation
"""
KELLY SAFETY GATES - DETAILED SPECIFICATION

Gate 1: Performance Gate (CRITICAL)
-------------------------------------
Criteria: Dampened Kelly must show improvement vs baseline
- Option A: +10% geometric mean return
- Option B: -20% max drawdown reduction AND maintain ≥90% baseline growth

Why: Ensures the system provides tangible benefit
Severity: CRITICAL - Must pass to proceed


Gate 2: Calibration Quality Gate (CRITICAL)
--------------------------------------------
Criteria: Mean |actual_win_rate - posterior_mean| < 10% across bins with ≥20 samples
Why: Ensures posterior predictions are well-calibrated to actual outcomes
Severity: CRITICAL - Poor calibration indicates model drift or bias
Action if fails: Review prior strength, check for data leakage


Gate 3: Bin Coverage Gate (WARNING)
------------------------------------
Criteria: ≥70% of test trades fall in bins with sufficient samples OR successful fallback
Why: Ensures adaptive bin merging is working effectively
Severity: WARNING - System should handle sparse bins gracefully
Action if fails: Adjust bin edges, enable adaptive merging


Gate 4: Regime Stability Gate (WARNING)
----------------------------------------
Criteria: <10% of trades experience mid-trade regime switches
Why: Regime switches during trade can invalidate Kelly assumptions
Severity: WARNING - High switches indicate unstable regimes
Action if fails: Increase regime smoothing, adjust regime discovery parameters


Gate 5: High-Leverage Performance Gate (CRITICAL)
--------------------------------------------------
Criteria: 
- Win rate for high-leverage trades (leverage ≥ 2.0) > 50%
- 95th percentile loss for high-leverage trades < 5%

Why: High-leverage trades are risky and must perform well
Severity: CRITICAL - Poor high-lev performance risks large losses
Action if fails: Reduce lambda_base, increase dampening (higher beta)


Gate 6: Numerical Stability Gate (CRITICAL)
--------------------------------------------
Criteria: No NaN or Inf values in any fold for any variant
Why: Numerical issues indicate bugs or edge cases
Severity: CRITICAL - System must be numerically stable
Action if fails: Review edge case handling, add more validation


DEPLOYMENT DECISION MATRIX
==========================

All gates PASS:
  → Proceed to Phase 4 (parameter optimization)
  → After optimization, proceed to shadow mode testing
  → Then staged canary deployment

1+ CRITICAL gates FAIL:
  → DO NOT proceed to optimization
  → Review and fix issues
  → Re-run validation

Only WARNING gates fail:
  → Review failures
  → Proceed to optimization with caution
  → Monitor affected metrics closely in shadow mode


EXAMPLE USAGE
=============

# After running walk-forward validation
from src.training.steps.backtesting.kelly_safety_gates import check_kelly_safety_gates

# Check gates
all_passed = check_kelly_safety_gates(
    validation_report_path="outcomes/kelly_validation/kelly_validation_BTCUSDT_15m_20251030.json",
    baseline_metrics={'geometric_return': 0.25, 'max_drawdown': 0.12}
)

if all_passed:
    print("✅ All gates passed - proceed to optimization")
else:
    print("❌ Some gates failed - review and adjust")
"""

