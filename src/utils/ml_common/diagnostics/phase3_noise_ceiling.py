"""
Phase 3: Noise Ceiling & Replicate-Based Checks

This module computes upper bounds on predictability when replicated labels
or multiple annotators exist. If the "noise ceiling" (maximum achievable R²
due to label inconsistency) is lower than model performance, there may be
data leakage or incorrect assumptions.

Metrics:
1. ICC (Intraclass Correlation Coefficient) - Inter-rater reliability
2. Krippendorff's Alpha - Agreement measure for continuous data
3. Pairwise label correlations
4. Expected maximum R² based on label consistency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import json
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class NoiseCeilingResults:
    """Results from noise ceiling analysis."""
    icc_one_way: float
    icc_two_way: float
    krippendorff_alpha: float
    pairwise_correlations: Dict[str, float]
    expected_max_r2: float
    label_variance_ratio: float
    n_raters: int
    n_samples: int


class NoiseCeilingAnalysis:
    """
    Noise Ceiling Analysis for Replicated Labels.

    When multiple annotators/measurements exist per sample, this class
    computes the theoretical maximum predictability due to label noise.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        verbose: bool = True
    ):
        """
        Initialize noise ceiling analysis.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save outputs
        verbose : bool, default=True
            Print progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def compute_icc(
        self,
        ratings: np.ndarray,
        icc_type: str = 'ICC(2,1)'
    ) -> float:
        """
        Compute Intraclass Correlation Coefficient.

        ICC measures consistency/agreement among raters.

        Parameters
        ----------
        ratings : np.ndarray
            Shape (n_samples, n_raters) - each row is ratings for one sample
        icc_type : str, default='ICC(2,1)'
            Type of ICC: 'ICC(1,1)', 'ICC(2,1)', 'ICC(3,1)', etc.

        Returns
        -------
        float
            ICC value (0-1, higher = better agreement)

        References
        ----------
        Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations.
        """
        n_samples, n_raters = ratings.shape

        # Mean squares
        ratings_mean = np.mean(ratings)
        row_means = np.mean(ratings, axis=1)
        col_means = np.mean(ratings, axis=0)

        # Sum of squares
        ss_total = np.sum((ratings - ratings_mean) ** 2)
        ss_rows = n_raters * np.sum((row_means - ratings_mean) ** 2)
        ss_cols = n_samples * np.sum((col_means - ratings_mean) ** 2)
        ss_error = ss_total - ss_rows - ss_cols

        # Mean squares
        ms_rows = ss_rows / (n_samples - 1)
        ms_cols = ss_cols / (n_raters - 1)
        ms_error = ss_error / ((n_samples - 1) * (n_raters - 1))

        # ICC calculations depend on type
        if icc_type == 'ICC(1,1)':
            # One-way random effects, single rater
            ms_within = (ss_total - ss_rows) / (n_samples * (n_raters - 1))
            icc = (ms_rows - ms_within) / (ms_rows + (n_raters - 1) * ms_within)

        elif icc_type == 'ICC(2,1)':
            # Two-way random effects, single rater (absolute agreement)
            icc = (ms_rows - ms_error) / (ms_rows + (n_raters - 1) * ms_error + (n_raters / n_samples) * (ms_cols - ms_error))

        elif icc_type == 'ICC(3,1)':
            # Two-way mixed effects, single rater (consistency)
            icc = (ms_rows - ms_error) / (ms_rows + (n_raters - 1) * ms_error)

        elif icc_type == 'ICC(2,k)':
            # Two-way random effects, average of k raters (absolute agreement)
            icc = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n_samples)

        elif icc_type == 'ICC(3,k)':
            # Two-way mixed effects, average of k raters (consistency)
            icc = (ms_rows - ms_error) / ms_rows

        else:
            raise ValueError(f"Unknown ICC type: {icc_type}")

        # ICC should be between 0 and 1
        icc = np.clip(icc, 0, 1)

        return float(icc)

    def compute_krippendorff_alpha(
        self,
        ratings: np.ndarray,
        level_of_measurement: str = 'interval'
    ) -> float:
        """
        Compute Krippendorff's Alpha for inter-rater reliability.

        Handles missing data and works for continuous (interval) data.

        Parameters
        ----------
        ratings : np.ndarray
            Shape (n_samples, n_raters) - NaN allowed for missing ratings
        level_of_measurement : str, default='interval'
            'nominal', 'ordinal', 'interval', or 'ratio'

        Returns
        -------
        float
            Alpha value (-1 to 1, higher = better agreement, 0 = chance)

        References
        ----------
        Krippendorff, K. (2004). Content Analysis: An Introduction to Its Methodology.
        """
        n_samples, n_raters = ratings.shape

        # Flatten and remove NaNs for pairwise comparisons
        # Build coincidence matrix
        if level_of_measurement == 'interval':
            # For interval data, use squared differences
            # Observed disagreement
            observed_disagreement = 0
            n_comparisons = 0

            for i in range(n_samples):
                sample_ratings = ratings[i, :]
                valid_ratings = sample_ratings[~np.isnan(sample_ratings)]
                n_valid = len(valid_ratings)

                if n_valid >= 2:
                    # All pairwise differences
                    for j in range(n_valid):
                        for k in range(j + 1, n_valid):
                            observed_disagreement += (valid_ratings[j] - valid_ratings[k]) ** 2
                            n_comparisons += 1

            if n_comparisons > 0:
                observed_disagreement /= n_comparisons
            else:
                return 0.0

            # Expected disagreement (if ratings were random)
            all_ratings = ratings.flatten()
            all_ratings = all_ratings[~np.isnan(all_ratings)]

            if len(all_ratings) < 2:
                return 0.0

            expected_disagreement = 0
            n_expected = 0
            for i in range(len(all_ratings)):
                for j in range(i + 1, len(all_ratings)):
                    expected_disagreement += (all_ratings[i] - all_ratings[j]) ** 2
                    n_expected += 1

            if n_expected > 0:
                expected_disagreement /= n_expected
            else:
                return 0.0

            # Krippendorff's alpha
            if expected_disagreement > 0:
                alpha = 1 - (observed_disagreement / expected_disagreement)
            else:
                alpha = 1.0 if observed_disagreement == 0 else 0.0

            return float(alpha)

        else:
            raise NotImplementedError(f"Level {level_of_measurement} not yet implemented")

    def compute_pairwise_correlations(
        self,
        ratings: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute pairwise correlations between all raters.

        Parameters
        ----------
        ratings : np.ndarray
            Shape (n_samples, n_raters)

        Returns
        -------
        dict
            {(rater_i, rater_j): correlation}
        """
        n_samples, n_raters = ratings.shape
        correlations = {}

        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                # Remove NaN pairs
                valid_mask = ~(np.isnan(ratings[:, i]) | np.isnan(ratings[:, j]))

                if np.sum(valid_mask) >= 3:  # Need at least 3 points
                    corr, _ = stats.pearsonr(ratings[valid_mask, i], ratings[valid_mask, j])
                    correlations[f'rater_{i}_vs_{j}'] = float(corr)

        return correlations

    def compute_noise_ceiling(
        self,
        ratings: np.ndarray,
        model_r2: Optional[float] = None
    ) -> NoiseCeilingResults:
        """
        Compute comprehensive noise ceiling metrics.

        Parameters
        ----------
        ratings : np.ndarray
            Shape (n_samples, n_raters) - ratings/labels from multiple sources
        model_r2 : float, optional
            Your model's R² for comparison

        Returns
        -------
        NoiseCeilingResults
            Complete noise ceiling analysis
        """
        if self.verbose:
            print("\n" + "="*80)
            print("NOISE CEILING ANALYSIS")
            print("="*80)

        n_samples, n_raters = ratings.shape

        if self.verbose:
            print(f"  Samples: {n_samples}, Raters: {n_raters}")

        # 1. ICC (One-way and Two-way)
        if self.verbose:
            print("  Computing ICC...")

        icc_one_way = self.compute_icc(ratings, icc_type='ICC(1,1)')
        icc_two_way = self.compute_icc(ratings, icc_type='ICC(2,1)')

        if self.verbose:
            print(f"    ICC(1,1): {icc_one_way:.4f}")
            print(f"    ICC(2,1): {icc_two_way:.4f}")

        # 2. Krippendorff's Alpha
        if self.verbose:
            print("  Computing Krippendorff's Alpha...")

        k_alpha = self.compute_krippendorff_alpha(ratings, level_of_measurement='interval')

        if self.verbose:
            print(f"    Alpha: {k_alpha:.4f}")

        # 3. Pairwise correlations
        if self.verbose:
            print("  Computing pairwise correlations...")

        pairwise_corr = self.compute_pairwise_correlations(ratings)

        if pairwise_corr:
            avg_corr = np.mean(list(pairwise_corr.values()))
            if self.verbose:
                print(f"    Average pairwise correlation: {avg_corr:.4f}")
        else:
            avg_corr = 0.0

        # 4. Expected maximum R²
        # This is the ICC(2,1) - represents consistency of average rating
        # A model predicting the average label can achieve at most this R²
        expected_max_r2 = max(icc_two_way, avg_corr)

        # 5. Label variance ratio
        # Variance within samples vs total variance
        sample_means = np.nanmean(ratings, axis=1)
        within_sample_var = np.nanmean(np.nanvar(ratings, axis=1))
        between_sample_var = np.nanvar(sample_means)
        total_var = within_sample_var + between_sample_var

        if total_var > 0:
            label_variance_ratio = between_sample_var / total_var
        else:
            label_variance_ratio = 0.0

        results = NoiseCeilingResults(
            icc_one_way=icc_one_way,
            icc_two_way=icc_two_way,
            krippendorff_alpha=k_alpha,
            pairwise_correlations=pairwise_corr,
            expected_max_r2=expected_max_r2,
            label_variance_ratio=label_variance_ratio,
            n_raters=n_raters,
            n_samples=n_samples
        )

        # Generate visualizations
        self._plot_noise_ceiling(ratings, results, model_r2)

        # Generate report
        self._generate_noise_ceiling_report(results, model_r2)

        # Save results
        self._save_results(results, model_r2)

        if self.verbose:
            print("\n  " + "-"*76)
            print(f"  Expected Max R²: {expected_max_r2:.4f}")

            if model_r2 is not None:
                if model_r2 > expected_max_r2 + 0.05:
                    print(f"  ⚠️  WARNING: Model R² ({model_r2:.4f}) > Noise Ceiling ({expected_max_r2:.4f})")
                    print("      Check for data leakage or incorrect assumptions!")
                elif model_r2 > expected_max_r2 * 0.8:
                    print(f"  ✓ Model R² ({model_r2:.4f}) near ceiling - good performance!")
                else:
                    print(f"  Model R² ({model_r2:.4f}) below ceiling - room for improvement")

        return results

    def _plot_noise_ceiling(
        self,
        ratings: np.ndarray,
        results: NoiseCeilingResults,
        model_r2: Optional[float] = None
    ):
        """Create visualizations for noise ceiling analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Rater agreement heatmap
        n_samples, n_raters = ratings.shape
        corr_matrix = np.corrcoef(ratings.T)  # Correlation between raters

        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, ax=ax1,
                   xticklabels=[f'R{i}' for i in range(n_raters)],
                   yticklabels=[f'R{i}' for i in range(n_raters)])
        ax1.set_title('Rater Correlation Matrix', fontsize=14, fontweight='bold')

        # 2. Distribution of rater means vs sample means
        sample_means = np.nanmean(ratings, axis=1)
        rater_means = np.nanmean(ratings, axis=0)

        ax2.hist(sample_means, bins=30, alpha=0.7, label='Sample means', edgecolor='black')
        ax2.axvline(np.nanmean(sample_means), color='red', linestyle='--',
                   linewidth=2, label='Overall mean')
        ax2.set_xlabel('Label Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Sample Mean Labels', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Within-sample variance distribution
        within_sample_std = np.nanstd(ratings, axis=1)

        ax3.hist(within_sample_std, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(np.nanmean(within_sample_std), color='red', linestyle='--',
                   linewidth=2, label='Mean std')
        ax3.set_xlabel('Within-Sample Std Dev', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Label Consistency per Sample', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Noise ceiling comparison
        metrics = ['ICC(1,1)', 'ICC(2,1)', "Kripp. α", 'Avg Corr', 'Expected\nMax R²']
        values = [
            results.icc_one_way,
            results.icc_two_way,
            results.krippendorff_alpha,
            np.mean(list(results.pairwise_correlations.values())) if results.pairwise_correlations else 0,
            results.expected_max_r2
        ]

        bars = ax4.bar(metrics, values, alpha=0.7, edgecolor='black', color='skyblue')
        ax4.set_ylabel('Value', fontsize=12)
        ax4.set_title('Noise Ceiling Metrics', fontsize=14, fontweight='bold')
        ax4.set_ylim([0, 1.1])
        ax4.axhline(y=1.0, color='green', linestyle='--', label='Perfect agreement')
        ax4.axhline(y=0.6, color='orange', linestyle='--', label='Acceptable (0.6)')
        ax4.axhline(y=0.4, color='red', linestyle='--', label='Poor (0.4)')

        if model_r2 is not None:
            ax4.axhline(y=model_r2, color='purple', linestyle='-', linewidth=2,
                       label=f'Model R²={model_r2:.3f}')

        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'noise_ceiling_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_noise_ceiling_report(
        self,
        results: NoiseCeilingResults,
        model_r2: Optional[float] = None
    ):
        """Generate markdown report for noise ceiling."""
        report_path = self.output_dir / 'noise_ceiling_report.md'

        with open(report_path, 'w') as f:
            f.write("# Noise Ceiling Analysis Report\n\n")
            f.write(f"**Generated**: {pd.Timestamp.now()}\n\n")
            f.write("---\n\n")

            f.write("## Overview\n\n")
            f.write("The **noise ceiling** represents the maximum predictability achievable "
                   "given the inherent inconsistency in labels. If multiple raters/measurements "
                   "disagree, even a perfect model cannot exceed their agreement level.\n\n")

            f.write(f"- **Samples**: {results.n_samples}\n")
            f.write(f"- **Raters/Replicates**: {results.n_raters}\n\n")

            f.write("## Metrics\n\n")

            f.write("### 1. Intraclass Correlation (ICC)\n\n")
            f.write(f"- **ICC(1,1)** (One-way): {results.icc_one_way:.4f}\n")
            f.write(f"- **ICC(2,1)** (Two-way): {results.icc_two_way:.4f}\n\n")

            f.write("**Interpretation:**\n")
            if results.icc_two_way > 0.75:
                f.write("- ✓ Excellent reliability (>0.75)\n")
            elif results.icc_two_way > 0.60:
                f.write("- ✓ Good reliability (0.60-0.75)\n")
            elif results.icc_two_way > 0.40:
                f.write("- ⚠️ Fair reliability (0.40-0.60)\n")
            else:
                f.write("- ❌ Poor reliability (<0.40)\n")
            f.write("\n")

            f.write("### 2. Krippendorff's Alpha\n\n")
            f.write(f"- **Alpha**: {results.krippendorff_alpha:.4f}\n\n")

            f.write("**Interpretation:**\n")
            if results.krippendorff_alpha > 0.80:
                f.write("- ✓ Excellent agreement (>0.80)\n")
            elif results.krippendorff_alpha > 0.67:
                f.write("- ✓ Good agreement (0.67-0.80)\n")
            elif results.krippendorff_alpha > 0.40:
                f.write("- ⚠️ Tentative agreement (0.40-0.67)\n")
            else:
                f.write("- ❌ Poor agreement (<0.40)\n")
            f.write("\n")

            f.write("### 3. Expected Maximum R²\n\n")
            f.write(f"- **Ceiling**: {results.expected_max_r2:.4f}\n\n")

            f.write("This is the theoretical maximum R² a model can achieve when predicting "
                   "the average label, given rater inconsistency.\n\n")

            if model_r2 is not None:
                f.write(f"- **Your Model R²**: {model_r2:.4f}\n")
                f.write(f"- **Ceiling Utilization**: {(model_r2 / results.expected_max_r2 * 100) if results.expected_max_r2 > 0 else 0:.1f}%\n\n")

                if model_r2 > results.expected_max_r2 + 0.05:
                    f.write("⚠️ **WARNING**: Model performance exceeds noise ceiling!\n\n")
                    f.write("**Possible causes:**\n")
                    f.write("- Data leakage (features contain future information)\n")
                    f.write("- Incorrect cross-validation strategy\n")
                    f.write("- Target encoding issues\n")
                    f.write("- Noise ceiling estimate may be too conservative\n\n")

            f.write("### 4. Label Variance Decomposition\n\n")
            f.write(f"- **Between-sample variance ratio**: {results.label_variance_ratio:.4f}\n\n")
            f.write("Fraction of total label variance that is between samples (vs within-sample disagreement).\n")
            f.write(f"Higher is better - indicates labels differentiate samples.\n\n")

            f.write("## Recommendations\n\n")

            if results.expected_max_r2 < 0.4:
                f.write("**Critical Issue: Low Label Quality**\n\n")
                f.write("- Labels are highly inconsistent (noise ceiling < 0.4)\n")
                f.write("- Consider improving annotation process\n")
                f.write("- May need multiple rounds of labeling with disagreement resolution\n")
                f.write("- Consider if the task is well-defined\n\n")

            elif results.expected_max_r2 < 0.6:
                f.write("**Moderate Label Noise**\n\n")
                f.write("- Labels have moderate inconsistency (ceiling 0.4-0.6)\n")
                f.write("- Model performance will be limited\n")
                f.write("- Focus on robust methods and regularization\n\n")

            else:
                f.write("**Good Label Quality**\n\n")
                f.write("- Labels are reasonably consistent (ceiling > 0.6)\n")
                f.write("- High R² is achievable with good models/features\n\n")

            if model_r2 is not None:
                if model_r2 < results.expected_max_r2 * 0.7:
                    f.write("**Improvement Opportunity**\n\n")
                    f.write(f"- Your model ({model_r2:.3f}) is below the ceiling ({results.expected_max_r2:.3f})\n")
                    f.write("- Room for improvement through:\n")
                    f.write("  - Better features\n")
                    f.write("  - More complex models\n")
                    f.write("  - Hyperparameter tuning\n\n")

        if self.verbose:
            print(f"  Report saved to {report_path}")

    def _save_results(self, results: NoiseCeilingResults, model_r2: Optional[float] = None):
        """Save results to JSON."""
        output = {
            'icc_one_way': float(results.icc_one_way),
            'icc_two_way': float(results.icc_two_way),
            'krippendorff_alpha': float(results.krippendorff_alpha),
            'pairwise_correlations': results.pairwise_correlations,
            'expected_max_r2': float(results.expected_max_r2),
            'label_variance_ratio': float(results.label_variance_ratio),
            'n_raters': results.n_raters,
            'n_samples': results.n_samples,
            'model_r2': model_r2 if model_r2 is not None else None,
            'ceiling_exceeded': model_r2 > results.expected_max_r2 + 0.05 if model_r2 else False
        }

        output_file = self.output_dir / 'noise_ceiling_results.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        if self.verbose:
            print(f"  Results saved to {output_file}")
