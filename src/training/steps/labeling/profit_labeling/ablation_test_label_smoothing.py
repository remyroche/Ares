"""
Label Smoothing Ablation Testing Utility

This script helps you run comprehensive ablation tests to evaluate the impact
of different label smoothing components and hyperparameters.

Usage:
    python ablation_test_label_smoothing.py --data_path <path> --output_dir <dir>

    Or import and use programmatically:

    from ablation_test_label_smoothing import run_full_ablation_suite

    results = run_full_ablation_suite(
        data=df,
        labeler_config=config,
        future_returns=returns,
        output_dir='ablation_results'
    )
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .volatility_aware_labeler import VolatilityAwareMultiHorizonLabeler, VolatilityAwareConfig
from .label_smoother import LabelSmoothingConfig, run_ablation_test, compare_ablation_results


def run_full_ablation_suite(
    data: pd.DataFrame,
    labeler_config: Optional[VolatilityAwareConfig] = None,
    future_returns: Optional[pd.Series] = None,
    output_dir: str = 'ablation_results',
    save_plots: bool = True,
    save_csv: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive ablation test suite for label smoothing.

    Tests:
    1. Component ablation (which stages help?)
    2. Hyperparameter sensitivity (how sensitive to param changes?)
    3. Baseline comparison (raw vs smoothed)

    Args:
        data: Market data DataFrame with OHLCV
        labeler_config: Labeler configuration (or None for defaults)
        future_returns: Actual future returns for IC calculation
        output_dir: Directory to save results
        save_plots: Whether to save visualization plots
        save_csv: Whether to save results to CSV

    Returns:
        Dictionary of result DataFrames:
            - 'component_ablation': Results from component tests
            - 'hyperparam_sensitivity': Results from parameter sensitivity
            - 'comparison_summary': Overall comparison metrics
    """
    print("="*80)
    print("LABEL SMOOTHING ABLATION TEST SUITE")
    print("="*80)
    print(f"Data: {len(data)} rows")
    print(f"Output: {output_dir}")
    print()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize labeler
    if labeler_config is None:
        labeler_config = VolatilityAwareConfig()

    labeler = VolatilityAwareMultiHorizonLabeler(labeler_config)

    # Generate raw labels (smoothing disabled)
    print("1️⃣  Generating raw labels (no smoothing)...")
    labeler_config.label_smoothing.enabled = False
    result_raw = labeler.generate_labels(data, price_column='close')
    labels_raw = result_raw.labels

    # TEST 1: Component Ablation
    print("\n2️⃣  Running component ablation tests...")
    labeler_config.label_smoothing.enabled = True
    component_results = _run_component_ablation(
        labels_raw,
        result_raw.quality_scores,
        data,
        future_returns,
        labeler_config
    )

    # TEST 2: Hyperparameter Sensitivity
    print("\n3️⃣  Running hyperparameter sensitivity tests...")
    hyperparam_results = _run_hyperparameter_sensitivity(
        labels_raw,
        result_raw.quality_scores,
        data,
        future_returns,
        labeler_config
    )

    # TEST 3: Generate comparison summary
    print("\n4️⃣  Generating comparison summary...")
    comparison_df = _generate_comparison_summary(component_results, hyperparam_results)

    # Save results
    results = {
        'component_ablation': component_results,
        'hyperparam_sensitivity': hyperparam_results,
        'comparison_summary': comparison_df
    }

    if save_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, df in results.items():
            csv_path = Path(output_dir) / f"{name}_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

    # Save plots
    if save_plots:
        print("\n5️⃣  Generating visualizations...")
        _generate_ablation_plots(results, output_dir)

    # Print summary
    print("\n" + "="*80)
    print("ABLATION TEST SUMMARY")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)

    return results


def _run_component_ablation(
    labels_raw: pd.Series,
    quality_scores: Dict,
    data: pd.DataFrame,
    future_returns: Optional[pd.Series],
    base_config: VolatilityAwareConfig
) -> pd.DataFrame:
    """Run ablation test across components."""
    ablation_modes = [
        'full',
        'classification_only',
        'uncertainty_only',
        'ema_only',
        'classification+uncertainty',
        'uncertainty+ema'
    ]

    # Extract quality for uncertainty
    opportunity_quality = None
    if quality_scores and 'opportunity_quality_scores' in quality_scores:
        opportunity_quality = quality_scores['opportunity_quality_scores']

    # Extract volatility
    volatility = data.get('volatility')

    # Prepare grouping data
    group_by_data = None
    if 'instrument' in data.columns:
        group_by_data = data[['instrument']].copy()
        if isinstance(data.index, pd.DatetimeIndex):
            group_by_data['timestamp'] = data.index

    # Run ablation
    ablation_results = run_ablation_test(
        labels=labels_raw,
        quality_scores=opportunity_quality,
        volatility=volatility,
        group_by_data=group_by_data,
        base_config=base_config.label_smoothing
    )

    # Compare results
    component_df = compare_ablation_results(
        ablation_results,
        future_returns=future_returns,
        print_summary=False
    )

    return component_df


def _run_hyperparameter_sensitivity(
    labels_raw: pd.Series,
    quality_scores: Dict,
    data: pd.DataFrame,
    future_returns: Optional[pd.Series],
    base_config: VolatilityAwareConfig
) -> pd.DataFrame:
    """Test sensitivity to hyperparameter changes."""
    from .label_smoother import LabelSmoother

    # Extract inputs
    opportunity_quality = None
    if quality_scores and 'opportunity_quality_scores' in quality_scores:
        opportunity_quality = quality_scores['opportunity_quality_scores']

    volatility = data.get('volatility')

    group_by_data = None
    if 'instrument' in data.columns:
        group_by_data = data[['instrument']].copy()
        if isinstance(data.index, pd.DatetimeIndex):
            group_by_data['timestamp'] = data.index

    # Test different hyperparameter values
    test_configs = []

    # Epsilon variations (classification smoothing strength)
    for eps in [0.05, 0.08, 0.12, 0.15]:
        config = LabelSmoothingConfig(
            epsilon=eps,
            gamma=1.0,
            ema_decay=0.95,
            uncertainty_source='quality_inverse'
        )
        test_configs.append(('epsilon', eps, config))

    # Gamma variations (uncertainty shrinkage strength)
    for gamma in [0.5, 1.0, 1.5, 2.0]:
        config = LabelSmoothingConfig(
            epsilon=0.08,
            gamma=gamma,
            ema_decay=0.95,
            uncertainty_source='quality_inverse'
        )
        test_configs.append(('gamma', gamma, config))

    # EMA decay variations (temporal smoothing strength)
    for decay in [0.90, 0.95, 0.98]:
        config = LabelSmoothingConfig(
            epsilon=0.08,
            gamma=1.0,
            ema_decay=decay,
            uncertainty_source='quality_inverse'
        )
        test_configs.append(('ema_decay', decay, config))

    # Run tests
    results = []
    for param_name, param_value, config in test_configs:
        smoother = LabelSmoother(config)
        result = smoother.smooth(
            labels=labels_raw,
            quality_scores=opportunity_quality,
            volatility=volatility,
            group_by_data=group_by_data
        )

        labels_final = result['labels_final']
        stats = result['metadata']['statistics']

        row = {
            'parameter': param_name,
            'value': param_value,
            'mean_abs_change': stats['mean_absolute_change'],
            'correlation': stats['correlation_raw_final'],
            'final_std': stats['final_std'],
            'pct_changed': stats['pct_changed']
        }

        # Calculate IC if future returns available
        if future_returns is not None:
            aligned_labels = labels_final.reindex(future_returns.index)
            valid_mask = ~(aligned_labels.isna() | future_returns.isna())
            if valid_mask.sum() > 10:
                ic = aligned_labels[valid_mask].corr(future_returns[valid_mask], method='spearman')
                row['IC'] = ic

        results.append(row)

    return pd.DataFrame(results)


def _generate_comparison_summary(
    component_df: pd.DataFrame,
    hyperparam_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate overall comparison summary."""
    # Get best configurations
    if 'IC' in component_df.columns:
        best_component = component_df.loc[component_df['IC'].idxmax()]
        summary_rows = [
            {
                'test_type': 'component_ablation',
                'best_config': best_component['mode'],
                'IC': best_component.get('IC', np.nan),
                'mean_abs_change': best_component.get('mean_abs_change', np.nan),
                'correlation': np.nan
            }
        ]
    else:
        best_component = component_df.loc[component_df['correlation'].idxmax()]
        summary_rows = [
            {
                'test_type': 'component_ablation',
                'best_config': best_component['mode'],
                'IC': np.nan,
                'mean_abs_change': best_component.get('mean_abs_change', np.nan),
                'correlation': best_component.get('correlation', np.nan)
            }
        ]

    # Get best hyperparameters
    for param in ['epsilon', 'gamma', 'ema_decay']:
        param_df = hyperparam_df[hyperparam_df['parameter'] == param]
        if len(param_df) > 0:
            if 'IC' in param_df.columns:
                best_idx = param_df['IC'].idxmax()
            else:
                best_idx = param_df['correlation'].idxmax()

            best_row = param_df.loc[best_idx]
            summary_rows.append({
                'test_type': f'hyperparam_{param}',
                'best_config': f"{param}={best_row['value']}",
                'IC': best_row.get('IC', np.nan),
                'mean_abs_change': best_row.get('mean_abs_change', np.nan),
                'correlation': best_row.get('correlation', np.nan)
            })

    return pd.DataFrame(summary_rows)


def _generate_ablation_plots(results: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Generate visualization plots for ablation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Component ablation comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Label Smoothing Component Ablation', fontsize=16, fontweight='bold')

    component_df = results['component_ablation']

    # IC comparison
    if 'IC' in component_df.columns:
        ax = axes[0, 0]
        component_df.plot(x='mode', y='IC', kind='bar', ax=ax, color='steelblue')
        ax.set_title('Information Coefficient by Component')
        ax.set_xlabel('Ablation Mode')
        ax.set_ylabel('IC (Spearman)')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Mean absolute change
    ax = axes[0, 1]
    component_df.plot(x='mode', y='mean_abs_change', kind='bar', ax=ax, color='coral')
    ax.set_title('Mean Absolute Change by Component')
    ax.set_xlabel('Ablation Mode')
    ax.set_ylabel('Mean |Δ|')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Correlation
    ax = axes[1, 0]
    component_df.plot(x='mode', y='correlation', kind='bar', ax=ax, color='lightgreen')
    ax.set_title('Raw-Final Correlation by Component')
    ax.set_xlabel('Ablation Mode')
    ax.set_ylabel('Correlation')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Std comparison
    ax = axes[1, 1]
    if 'std' in component_df.columns:
        component_df.plot(x='mode', y='std', kind='bar', ax=ax, color='mediumpurple')
    ax.set_title('Label Std Dev by Component')
    ax.set_xlabel('Ablation Mode')
    ax.set_ylabel('Std Dev')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plot_path = Path(output_dir) / f"component_ablation_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Plot 2: Hyperparameter sensitivity
    hyperparam_df = results['hyperparam_sensitivity']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')

    for idx, param in enumerate(['epsilon', 'gamma', 'ema_decay']):
        param_df = hyperparam_df[hyperparam_df['parameter'] == param]
        ax = axes[idx]

        if 'IC' in param_df.columns:
            ax.plot(param_df['value'], param_df['IC'], marker='o', linewidth=2, markersize=8, label='IC')
            ax.set_ylabel('IC (Spearman)', fontsize=11)

        ax2 = ax.twinx()
        ax2.plot(param_df['value'], param_df['mean_abs_change'], marker='s', linewidth=2,
                markersize=8, color='coral', label='Mean |Δ|')
        ax2.set_ylabel('Mean |Δ|', fontsize=11, color='coral')

        ax.set_xlabel(f'{param} value', fontsize=11)
        ax.set_title(f'{param} Sensitivity')
        ax.grid(True, alpha=0.3)

        # Legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plot_path = Path(output_dir) / f"hyperparam_sensitivity_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()


def main():
    """Command-line interface for ablation testing."""
    parser = argparse.ArgumentParser(description='Run label smoothing ablation tests')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to market data CSV/parquet')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='Directory to save results')
    parser.add_argument('--future_returns_col', type=str, default=None,
                       help='Column name for future returns (for IC calculation)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--no_csv', action='store_true',
                       help='Skip saving CSV files')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    if args.data_path.endswith('.parquet'):
        data = pd.read_parquet(args.data_path)
    else:
        data = pd.read_csv(args.data_path)

    # Extract future returns if specified
    future_returns = None
    if args.future_returns_col and args.future_returns_col in data.columns:
        future_returns = data[args.future_returns_col]

    # Run ablation suite
    results = run_full_ablation_suite(
        data=data,
        future_returns=future_returns,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
        save_csv=not args.no_csv
    )

    print("\n✅ Ablation testing complete!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
