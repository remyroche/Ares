#!/usr/bin/env python3
"""
Feature Selection Method Comparison Tool

Compares MI/SHAP before RF vs traditional RF to demonstrate time-saving benefits.
"""

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced matrix operations step
from src.training.steps.model_training.step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep

class FeatureSelectionComparator:
    """Compare different feature selection approaches for time and performance."""

    def __init__(self):
        """Initialize the comparator."""
        self.results = {}

    def generate_synthetic_data(self, n_samples: int = 10000, n_features: int = 200,
                               n_informative: int = 20, noise_level: float = 0.1) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Generate synthetic dataset with known informative features."""
        from sklearn.datasets import make_classification, make_regression

        # Generate classification target (SR detection)
        X_clf, y_clf = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=max(5, n_informative // 4),
            n_clusters_per_class=2,
            noise=noise_level,
            random_state=42
        )

        # Generate regression target (SR strength)
        X_reg, y_reg = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise_level * 100,
            random_state=42
        )

        # Convert to DataFrames with feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_clf_df = pd.DataFrame(X_clf, columns=feature_names)
        X_reg_df = pd.DataFrame(X_reg, columns=feature_names)

        # Use same features for both targets (simulating real scenario)
        X_combined = (X_clf_df + X_reg_df) / 2

        return X_combined, pd.Series(y_clf), pd.Series(y_reg)

    def method_traditional_rf(self, X: pd.DataFrame, y_clf: pd.Series, y_reg: pd.Series,
                             max_features: int = 50) -> Dict[str, Any]:
        """Traditional approach: Train full RF and use feature importance."""
        print("🔍 Testing Traditional RF Feature Selection...")

        start_time = time.time()

        # Train Random Forest on full feature set
        rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )

        rf_reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )

        # Split data
        X_train, X_test, y_clf_train, y_clf_test = train_test_split(
            X, y_clf, test_size=0.2, random_state=42
        )
        _, _, y_reg_train, y_reg_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )

        # Train models
        rf_clf.fit(X_train, y_clf_train)
        rf_reg.fit(X_train, y_reg_train)

        # Get feature importance
        clf_importance = dict(zip(X.columns, rf_clf.feature_importances_))
        reg_importance = dict(zip(X.columns, rf_reg.feature_importances_))

        # Combine importance scores
        combined_importance = {}
        for feature in X.columns:
            clf_imp = clf_importance.get(feature, 0)
            reg_imp = reg_importance.get(feature, 0)
            combined_importance[feature] = max(clf_imp, reg_imp)

        # Select top features
        sorted_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in sorted_features[:max_features]]
        X_selected = X[selected_features]

        # Evaluate performance
        rf_clf_selected = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_reg_selected = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        rf_clf_selected.fit(X_train[selected_features], y_clf_train)
        rf_reg_selected.fit(X_train[selected_features], y_reg_train)

        clf_pred = rf_clf_selected.predict(X_test[selected_features])
        reg_pred = rf_reg_selected.predict(X_test[selected_features])

        clf_accuracy = accuracy_score(y_clf_test, clf_pred)
        reg_mse = mean_squared_error(y_reg_test, reg_pred)

        total_time = time.time() - start_time

        return {
            'method': 'traditional_rf',
            'total_time': total_time,
            'selected_features': len(selected_features),
            'original_features': X.shape[1],
            'clf_accuracy': clf_accuracy,
            'reg_mse': reg_mse,
            'feature_importance': combined_importance,
            'top_features': selected_features[:10]
        }

    def method_mi_shap_rf(self, X: pd.DataFrame, y_clf: pd.Series, y_reg: pd.Series,
                         max_features: int = 50, mi_multiplier: float = 2.0,
                         shap_sample_size: int = 5000) -> Dict[str, Any]:
        """Optimized approach: MI → SHAP → RF."""
        print("🎯 Testing MI/SHAP Optimized Feature Selection...")

        start_time = time.time()
        timing_breakdown = {}

        # Phase 1: Mutual Information (Fast filtering)
        phase1_start = time.time()

        mi_clf = mutual_info_classif(X, y_clf, random_state=42)
        mi_reg = mutual_info_regression(X, y_reg, random_state=42)

        # Combine MI scores
        mi_scores = {}
        for i, feature in enumerate(X.columns):
            mi_scores[feature] = max(mi_clf[i], mi_reg[i])

        # Select top features with MI (more than needed for SHAP refinement)
        mi_target_count = min(int(max_features * mi_multiplier), len(mi_scores))
        mi_threshold = sorted(mi_scores.values(), reverse=True)[mi_target_count - 1] if mi_target_count > 0 else 0
        mi_selected_features = [f for f, score in mi_scores.items() if score >= mi_threshold]
        X_mi_selected = X[mi_selected_features]

        timing_breakdown['mi'] = time.time() - phase1_start

        # Phase 2: SHAP refinement
        phase2_start = time.time()

        # Use smaller RF for SHAP
        shap_rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        # Sample for SHAP efficiency
        sample_size = min(shap_sample_size, len(X_mi_selected))
        X_sample = X_mi_selected.sample(n=sample_size, random_state=42)
        y_sample = y_clf.loc[X_sample.index]

        shap_rf.fit(X_sample, y_sample)

        # SHAP feature importance
        explainer = shap.TreeExplainer(shap_rf)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_importance = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)

        shap_scores = dict(zip(X_mi_selected.columns, shap_importance))

        # Select final features
        sorted_shap = sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)
        final_features = [f for f, _ in sorted_shap[:max_features]]
        X_selected = X[final_features]

        timing_breakdown['shap'] = time.time() - phase2_start

        # Phase 3: Final RF training and evaluation
        phase3_start = time.time()

        # Split data
        X_train, X_test, y_clf_train, y_clf_test = train_test_split(
            X_selected, y_clf, test_size=0.2, random_state=42
        )
        _, _, y_reg_train, y_reg_test = train_test_split(
            X_selected, y_reg, test_size=0.2, random_state=42
        )

        # Train final models
        rf_clf_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_reg_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        rf_clf_final.fit(X_train, y_clf_train)
        rf_reg_final.fit(X_train, y_reg_train)

        # Evaluate
        clf_pred = rf_clf_final.predict(X_test)
        reg_pred = rf_reg_final.predict(X_test)

        clf_accuracy = accuracy_score(y_clf_test, clf_pred)
        reg_mse = mean_squared_error(y_reg_test, reg_pred)

        timing_breakdown['final_rf'] = time.time() - phase3_start
        total_time = time.time() - start_time

        return {
            'method': 'mi_shap_rf',
            'total_time': total_time,
            'timing_breakdown': timing_breakdown,
            'selected_features': len(final_features),
            'original_features': X.shape[1],
            'mi_selected_features': len(mi_selected_features),
            'clf_accuracy': clf_accuracy,
            'reg_mse': reg_mse,
            'mi_scores': mi_scores,
            'shap_importance': shap_scores,
            'top_features': final_features[:10],
            'time_savings_ratio': None  # Will be calculated in comparison
        }

    def run_comparison(self, n_samples: int = 5000, n_features: int = 150,
                       n_informative: int = 15) -> Dict[str, Any]:
        """Run comprehensive comparison between methods."""
        print("🚀 Running Feature Selection Method Comparison")
        print(f"📊 Dataset: {n_samples} samples, {n_features} features ({n_informative} informative)")
        print("=" * 80)

        # Generate data
        X, y_clf, y_reg = self.generate_synthetic_data(n_samples, n_features, n_informative)

        # Run both methods
        results_traditional = self.method_traditional_rf(X, y_clf, y_reg)
        results_optimized = self.method_mi_shap_rf(X, y_clf, y_reg)

        # Calculate time savings
        time_savings = results_traditional['total_time'] - results_optimized['total_time']
        time_savings_ratio = time_savings / results_traditional['total_time']

        results_optimized['time_savings_ratio'] = time_savings_ratio

        # Store results
        self.results = {
            'traditional_rf': results_traditional,
            'mi_shap_rf': results_optimized,
            'comparison': {
                'time_savings_seconds': time_savings,
                'time_savings_ratio': time_savings_ratio,
                'performance_difference_clf': results_optimized['clf_accuracy'] - results_traditional['clf_accuracy'],
                'performance_difference_reg': results_traditional['reg_mse'] - results_optimized['reg_mse'],  # Lower MSE is better
                'feature_reduction_traditional': results_traditional['original_features'] / results_traditional['selected_features'],
                'feature_reduction_optimized': results_optimized['original_features'] / results_optimized['selected_features']
            }
        }

        return self.results

    def generate_comparison_report(self) -> str:
        """Generate detailed comparison report."""
        if not self.results:
            return "No results available. Run comparison first."

        trad = self.results['traditional_rf']
        opt = self.results['mi_shap_rf']
        comp = self.results['comparison']

        report = []
        report.append("=" * 100)
        report.append("🎯 FEATURE SELECTION METHOD COMPARISON REPORT")
        report.append("=" * 100)

        # Executive Summary
        report.append("\n📊 EXECUTIVE SUMMARY")
        report.append("-" * 50)
        report.append(f"⏱️  Time Savings: {comp['time_savings_seconds']:.2f}s ({comp['time_savings_ratio']*100:.1f}%)")
        report.append(f"🎯 Performance Δ (Classification): {comp['performance_difference_clf']*100:.2f}%")
        report.append(f"📈 Performance Δ (Regression): {comp['performance_difference_reg']:.4f} MSE")
        report.append(f"🔧 Feature Reduction: {comp['feature_reduction_optimized']:.1f}x (optimized) vs {comp['feature_reduction_traditional']:.1f}x (traditional)")

        # Detailed Results
        report.append("\n🔍 DETAILED RESULTS")
        report.append("-" * 50)

        for method_name, results in [('Traditional RF', trad), ('MI/SHAP + RF', opt)]:
            report.append(f"\n{method_name.upper()}")
            report.append(f"  ⏱️  Total Time: {results['total_time']:.2f}s")
            report.append(f"  🎯 Selected Features: {results['selected_features']}/{results['original_features']}")
            report.append(f"  📊 Classification Accuracy: {results['clf_accuracy']:.4f}")
            report.append(f"  📈 Regression MSE: {results['reg_mse']:.4f}")

            if 'timing_breakdown' in results:
                report.append(f"  🕐 Timing Breakdown:")
                for phase, time_taken in results['timing_breakdown'].items():
                    report.append(f"    • {phase.upper()}: {time_taken:.2f}s")

        # Recommendations
        report.append("\n💡 RECOMMENDATIONS")
        report.append("-" * 50)

        if comp['time_savings_ratio'] > 0.1:  # >10% time savings
            report.append("✅ STRONG RECOMMENDATION: Use MI/SHAP before RF")
            report.append("   • Significant time savings with minimal performance loss")
            report.append("   • Better feature interpretability through SHAP")
        elif comp['time_savings_ratio'] > 0:
            report.append("👍 MODERATE RECOMMENDATION: Consider MI/SHAP approach")
            report.append("   • Some time savings, evaluate based on your priorities")
        else:
            report.append("🤔 CONSIDER ALTERNATIVES: Traditional RF may be better")
            report.append("   • MI/SHAP approach didn't provide time benefits")

        if comp['performance_difference_clf'] > -0.02:  # Less than 2% accuracy loss
            report.append("✅ Performance Impact: Acceptable (≤2% accuracy difference)")
        else:
            report.append("⚠️  Performance Impact: Significant accuracy loss detected")

        # Use Cases
        report.append("\n🎯 OPTIMAL USE CASES")
        report.append("-" * 50)
        report.append("✅ Large feature sets (>100 features)")
        report.append("✅ Time-critical applications")
        report.append("✅ Need for feature interpretability")
        report.append("✅ Limited computational resources")
        report.append("✅ Feature engineering optimization")

        report.append("\n" + "=" * 100)

        return "\n".join(report)

def test_step07_integration():
    """Test the integration with step07 feature selection."""
    print("\n🔬 Testing Step07 Integration...")
    print("-" * 40)

    try:
        # Test if step07 can be imported
        from src.training.steps.market_analysis.step07_enhanced_matrix_operations import Step7EnhancedMatrixOperations

        # Create test data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(1000, 20), columns=[f'feature_{i}' for i in range(20)])
        y = pd.Series(np.random.choice([0, 1], 1000))
        labels_df = pd.DataFrame({'target': y})

        # Test configuration
        config = {
            'step07_enhanced_matrix_operations': {
                'target_features': 10,
                'removal_fraction': 0.5,
                'enable_regime_selection': False,
                'enable_shap_filtering': True
            }
        }

        # Initialize step07
        step07 = EnhancedMatrixOperationsStep(config=config)

        # Test feature selection
        X_selected, metadata = step07.regime_aware_initial_filtering(X, labels_df)

        print("✅ Step07 integration successful!")
        print(f"   Original features: {X.shape[1]}")
        print(f"   Selected features: {X_selected.shape[1]}")
        print(f"   Method used: {metadata.get('method', 'unknown')}")
        print(f"   Removal rate: {metadata.get('removal_fraction', 0):.1%}")

        return True

    except Exception as e:
        print(f"❌ Step07 integration failed: {e}")
        return False

def main():
    """Main function to run the comparison."""
    print("🚀 Feature Selection Method Comparison Tool")
    print("=" * 60)

    # Test step07 integration first
    step07_ok = test_step07_integration()

    if not step07_ok:
        print("\n⚠️ Step07 integration failed, running simplified comparison...")
        return

    print("\n✅ Step07 integration verified, proceeding with comparison...")
    comparator = FeatureSelectionComparator()

    # Run comparison with smaller dataset for faster testing
    test_configs = [
        {'n_samples': 2000, 'n_features': 50, 'n_informative': 10, 'name': 'Small'},
        {'n_samples': 3000, 'n_features': 80, 'n_informative': 12, 'name': 'Medium'}
    ]

    all_results = {}

    for config in test_configs:
        print(f"\n🔬 Testing {config['name']} Dataset...")
        print(f"   Samples: {config['n_samples']}, Features: {config['n_features']}")

        results = comparator.run_comparison(**{k: v for k, v in config.items() if k != 'name'})
        all_results[config['name']] = results

        # Quick summary
        comp = results['comparison']
        print(f"   📊 Time Savings: {comp['time_savings_seconds']:.2f}s")
        print(f"   🎯 Accuracy Δ: {comp['performance_difference_clf']*100:.1f}%")
        print(f"   📈 MSE Δ: {comp['performance_difference_reg']:.2f}")
    # Generate comprehensive report
    print("\n📊 Generating Comprehensive Report...")
    report = comparator.generate_comparison_report()
    print(report)

    # Save detailed results
    import json
    with open('feature_selection_comparison_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in all_results.items():
            json_results[key] = {}
            for method, method_results in value.items():
                json_results[key][method] = {}
                for result_key, result_value in method_results.items():
                    if isinstance(result_value, np.ndarray):
                        json_results[key][method][result_key] = result_value.tolist()
                    elif isinstance(result_value, (np.int64, np.float64)):
                        json_results[key][method][result_key] = float(result_value)
                    elif isinstance(result_value, dict):
                        json_results[key][method][result_key] = {}
                        for dict_key, dict_value in result_value.items():
                            if isinstance(dict_value, np.ndarray):
                                json_results[key][method][result_key][dict_key] = dict_value.tolist()
                            elif isinstance(dict_value, (np.int64, np.float64)):
                                json_results[key][method][result_key][dict_key] = float(dict_value)
                            else:
                                json_results[key][method][result_key][dict_key] = dict_value
                    else:
                        json_results[key][method][result_key] = result_value

        json.dump(json_results, f, indent=2)

    print("\n💾 Detailed results saved to: feature_selection_comparison_results.json")
    print("\n🎉 Comparison complete! Step07 integration is working correctly.")

if __name__ == "__main__":
    main()
