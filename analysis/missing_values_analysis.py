#!/usr/bin/env python3
"""
Missing Values Analysis Report
Analyzes the extent, patterns, and causes of missing values in the financial dataset.
"""

import warnings

import matplotlib.pyplot as plt
import pandas as pd

    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
warnings.filterwarnings('ignore')

# Set up plotting with matplotlib only
plt.style.use('default')
# Create a custom color palette similar to seaborn's husl
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

class MissingValuesAnalyzer:
    pass  # TODO: Add proper implementation
    def __init__(self, data_path=None):
        self.data, None
        self.report = {}


    def load_data(self, data_path):
        """Load the dataset for analysis."""
        try:
            self.data = pd.read_csv(data_path)
            print(f"✅ Data loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
            return True
        except Exception as e:
            print(warning(f"Error loading data: {e}"))
        return False


    def analyze_missing_values(self):
        """Comprehensive missing values analysis."""
        if self.data is None:
            print(warning("No data loaded. Please load data first."))
            return

        print("\n" + "="*60)
        print("🔍 MISSING VALUES ANALYSIS REPORT")
        print("="*60)

        # 1. Overall missing values summary
        self._overall_summary()

        # 2. Column-wise analysis
        self._column_analysis()

        # 3. Temporal analysis (if datetime available)
        self._temporal_analysis()

        # 4. Pattern analysis
        self._pattern_analysis()

        # 5. Feature category analysis
        self._feature_category_analysis()

        # 6. Recommendations
        self._generate_recommendations()

        # 7. Generate visualizations
        self._create_visualizations()


    def _overall_summary(self):
        """Overall missing values summary."""
        print("\n📊 OVERALL SUMMARY")
        print("-" * 40)

        total_cells = len(self.data) * len(self.data.columns)
        missing_cells = self.data.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100

        print(f"Total data cells: {total_cells:,}")
        print(f"Missing cells: {missing_cells:,}")
        print(f"Missing percentage: {missing_percentage:.2f}%")

        # Columns with missing values
        columns_with_missing = self.data.columns[self.data.isnull().any()].tolist()
        print(f"Columns with missing values: {len(columns_with_missing)}/{len(self.data.columns)}")

        self.report['overall'] = {
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_percentage': missing_percentage,
            'columns_with_missing': columns_with_missing
        }


    def _column_analysis(self):
        """Detailed column-wise analysis."""
        print("\n📋 COLUMN-WISE ANALYSIS")
        print("-" * 40)

        missing_stats = []
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            missing_pct = (missing_count / len(self.data)) * 100

            if missing_count > 0:
                missing_stats.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct,
                    'data_type': str(self.data[col].dtype)
                })

        # Sort by missing percentage
        missing_stats.sort(key=lambda x: x['missing_percentage'], reverse=True)

        print(f"{'Column':<30} {'Missing':<10} {'%':<8} {'Type':<12}")
        print("-" * 60)

        for stat in missing_stats[:20]:  # Top 20
            print(f"{stat['column']:<30} {stat['missing_count']:<10,} {stat['missing_percentage']:<8.2f} {stat['data_type']:<12}")

        if len(missing_stats) > 20:
            print(f"... and {len(missing_stats) - 20} more columns")

        self.report['column_analysis'] = missing_stats


    def _temporal_analysis(self):
        """Analyze missing values over time."""
        print("\n⏰ TEMPORAL ANALYSIS")
        print("-" * 40)

        # Check if we have datetime columns
        datetime_cols = []
        for col in self.data.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                datetime_cols.append(col)

        if not datetime_cols:
            print("No datetime columns found for temporal analysis.")
            return

        # Use the first datetime column found
        time_col = datetime_cols[0]
        print(f"Using time column: {time_col}")

        try:
            # Convert to datetime if needed
            if self.data[time_col].dtype == 'object':
                self.data[time_col] = pd.to_datetime(self.data[time_col])

            # Group by time periods
            self.data['time_period'] = self.data[time_col].dt.to_period('D')

            daily_missing = self.data.groupby('time_period').apply(
                lambda x: x.isnull().sum().sum() / (len(x) * len(x.columns)) * 100
            )

            print(f"Daily missing percentage range: {daily_missing.min():.2f}% - {daily_missing.max():.2f}%")
            print(f"Average daily missing: {daily_missing.mean():.2f}%")

            # Find periods with high missing values
            high_missing_days = daily_missing[daily_missing > daily_missing.mean() + daily_missing.std()]
            if len(high_missing_days) > 0:
                print(f"Days with high missing values: {len(high_missing_days)}")
                print("Sample high-missing days:")
                for day, pct in high_missing_days.head().items():
                    print(f"  {day}: {pct:.2f}%")

            self.report['temporal'] = {
                'daily_missing': daily_missing.to_dict(),
                'high_missing_days': high_missing_days.to_dict() if len(high_missing_days) > 0 else {}
            }

        except Exception as e:
            print(f"Error in temporal analysis: {e}")


    def _pattern_analysis(self):
        """Analyze patterns in missing values."""
        print("\n🔍 PATTERN ANALYSIS")
        print("-" * 40)

        # Check for systematic patterns
        missing_matrix = self.data.isnull()

        # 1. Consecutive missing values
        consecutive_missing = []
        for col in self.data.columns:
            if missing_matrix[col].any():
                # Find consecutive missing values
                missing_runs = missing_matrix[col].astype(int).groupby(
                    (missing_matrix[col] != missing_matrix[col].shift()).cumsum()
                ).sum()

                max_consecutive = missing_runs.max()
                if max_consecutive > 1:
                    consecutive_missing.append({
                        'column': col,
                        'max_consecutive': max_consecutive,
                        'total_missing': missing_matrix[col].sum()
                    })

        if consecutive_missing:
            print("Columns with consecutive missing values:")
            for item in sorted(consecutive_missing, key=lambda x: x['max_consecutive'], reverse=True)[:10]:
                print(f"  {item['column']}: {item['max_consecutive']} consecutive (total: {item['total_missing']})")

        # 2. Correlation between missing values
        missing_corr = missing_matrix.corr()
        high_corr_pairs = []

        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.5:  # High correlation threshold
                    high_corr_pairs.append({
                        'col1': missing_corr.columns[i],
                        'col2': missing_corr.columns[j],
                        'correlation': corr_val
                    })

        if high_corr_pairs:
            print(f"\nColumns with correlated missing patterns ({len(high_corr_pairs)} pairs):")
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:5]:
                print(f"  {pair['col1']} ↔ {pair['col2']}: {pair['correlation']:.3f}")

        self.report['patterns'] = {
            'consecutive_missing': consecutive_missing,
            'correlated_missing': high_corr_pairs
        }


    def _feature_category_analysis(self):
        """Analyze missing values by feature categories."""
        print("\n📊 FEATURE CATEGORY ANALYSIS")
        print("-" * 40)

        # Categorize features
        categories = {
            'price_data': ['open', 'high', 'low', 'close'],
            'volume_data': ['volume', 'quote_asset_volume', 'taker_buy_base_asset_volume'],
            'technical_indicators': ['ADX', 'MACD', 'RSI', 'BB', 'ATR', 'OBV'],
            'volatility_features': ['Simple_Volatility', 'EWMA_Volatility', 'GARCH_Volatility'],
            'momentum_features': ['Price_Momentum', 'Volume_Momentum', 'Volatility_Momentum'],
            'funding_features': ['fundingRate', 'Funding_Momentum', 'Funding_Divergence'],
            'target_variables': ['target', 'target_sr', 'reward', 'risk']
        }

        category_stats = {}

        for category, keywords in categories.items():
            matching_cols = []
            for col in self.data.columns:
                if any(keyword.lower() in col.lower() for keyword in keywords):
                    matching_cols.append(col)

            if matching_cols:
                category_data = self.data[matching_cols]
                missing_count = category_data.isnull().sum().sum()
                missing_pct = (missing_count / (len(category_data) * len(category_data.columns))) * 100

                category_stats[category] = {
                    'columns': matching_cols,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct,
                    'column_count': len(matching_cols)
                }

        print(f"{'Category':<25} {'Columns':<8} {'Missing %':<12} {'Missing Count':<15}")
        print("-" * 60)

        for category, stats in category_stats.items():
            print(f"{category:<25} {stats['column_count']:<8} {stats['missing_percentage']:<12.2f} {stats['missing_count']:<15,}")

        self.report['categories'] = category_stats


    def _generate_recommendations(self):
        """Generate recommendations based on analysis."""
        print("\n💡 RECOMMENDATIONS")
        print("-" * 40)

        overall_missing, self.report['overall']['missing_percentage']

        if overall_missing < 5:
            print("✅ Overall missing data is low (< 5%). Standard imputation methods should work well.")
        elif overall_missing < 15:
            print(missing(" Moderate missing data (5-15%). Consider advanced imputation techniques."))
        else:
            print(missing("High missing data (> 15%). May need specialized handling."))

        # Specific recommendations based on patterns
        if self.report.get('patterns', {}).get('consecutive_missing'):
            print("\n🔧 Consecutive missing values detected:")
            print("   - Consider forward-fill for short gaps")
            print("   - Use interpolation for longer gaps")
            print("   - Investigate data source reliability")

        if self.report.get('patterns', {}).get('correlated_missing'):
            print("\n🔗 Correlated missing patterns found:")
            print("   - Missing values may be systematic")
            print("   - Consider multivariate imputation")
            print("   - Investigate root cause of missing data")

        # Feature-specific recommendations
        categories, self.report.get('categories', {})
        if 'technical_indicators' in categories:
            ti_missing, categories['technical_indicators']['missing_percentage']
        if ti_missing > 10:
                print(f"\n📈 Technical indicators have {ti_missing:.1f}% missing values:")
                print("   - Consider using shorter lookback periods")
                print("   - Implement proper warm-up periods")
                print("   - Use robust calculation methods")

        if 'target_variables' in categories:
            target_missing, categories['target_variables']['missing_percentage']
        if target_missing > 5:
                print(f"\n🎯 Target variables have {target_missing:.1f}% missing values:")
                print("   - Critical: Investigate target generation logic")
                print("   - Consider alternative target definitions")
                print("   - Ensure proper data alignment")


    def _create_visualizations(self):
        """Create visualizations for the report."""
        print("\n📈 GENERATING VISUALIZATIONS...")

        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Missing Values Analysis Report', fontsize=16, fontweight='bold')

            # 1. Missing values by column (top 20)
            if self.report.get('column_analysis'):
                col_data = self.report['column_analysis'][:20]
                cols = [item['column'] for item in col_data]
                missing_pcts = [item['missing_percentage'] for item in col_data]

                bars = axes[0, 0].barh(range(len(cols)), missing_pcts, color=colors[0])
                axes[0, 0].set_yticks(range(len(cols)))
                axes[0, 0].set_yticklabels(cols, fontsize=8)
                axes[0, 0].set_xlabel('Missing Percentage (%)')
                axes[0, 0].set_title('Missing Values by Column (Top 20)')
                axes[0, 0].grid(True, alpha=0.3)

            # 2. Missing values by category
            if self.report.get('categories'):
                categories = list(self.report['categories'].keys())
                missing_pcts = [self.report['categories'][cat]['missing_percentage'] for cat in categories]

                bars = axes[0, 1].bar(categories, missing_pcts, color=colors[1])
                axes[0, 1].set_ylabel('Missing Percentage (%)')
                axes[0, 1].set_title('Missing Values by Feature Category')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)

            # 3. Temporal analysis (if available)
            if self.report.get('temporal', {}).get('daily_missing'):
                daily_data = self.report['temporal']['daily_missing']
                dates = list(daily_data.keys())
                missing_pcts = list(daily_data.values())

                axes[1, 0].plot(range(len(dates)), missing_pcts, marker='o', markersize=3, color=colors[2])
                axes[1, 0].set_xlabel('Time Period')
                axes[1, 0].set_ylabel('Missing Percentage (%)')
                axes[1, 0].set_title('Missing Values Over Time')
                axes[1, 0].grid(True, alpha=0.3)

            # 4. Overall summary pie chart
            if self.report.get('overall'):
                missing_pct = self.report['overall']['missing_percentage']
                present_pct = 100 - missing_pct

                axes[1, 1].pie([present_pct, missing_pct],
                               labels=['Present', 'Missing'],
                               autopct='%1.1f%%',
                               colors=[colors[3], colors[4]])
                axes[1, 1].set_title('Overall Data Completeness')

            plt.tight_layout()
            plt.savefig('missing_values_report.png', dpi=300, bbox_inches='tight')
            print("✅ Visualizations saved as 'missing_values_report.png'")

        except Exception as e:
            print(warning(f"Error creating visualizations: {e}"))


    def save_report(self, filename='missing_values_report.txt'):
        """Save the analysis report to a file."""
        with open(filename, 'w') as f:
            f.write("MISSING VALUES ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Overall summary
            if self.report.get('overall'):
                f.write("OVERALL SUMMARY:\n")
                f.write(f"Total cells: {self.report['overall']['total_cells']:,}\n")
                f.write(f"Missing cells: {self.report['overall']['missing_cells']:,}\n")
                f.write(f"Missing percentage: {self.report['overall']['missing_percentage']:.2f}%\n\n")

            # Column analysis
            if self.report.get('column_analysis'):
                f.write("COLUMN ANALYSIS (Top 20):\n")
                for stat in self.report['column_analysis'][:20]:
                    f.write(f"{stat['column']}: {stat['missing_count']:,} ({stat['missing_percentage']:.2f}%)\n")
                f.write("\n")

            # Category analysis
            if self.report.get('categories'):
                f.write("CATEGORY ANALYSIS:\n")
                for category, stats in self.report['categories'].items():
                    f.write(f"{category}: {stats['missing_percentage']:.2f}% missing\n")
                f.write("\n")

            # Patterns
            if self.report.get('patterns'):
                f.write("PATTERN ANALYSIS:\n")
                if self.report['patterns'].get('consecutive_missing'):
                    f.write("Consecutive missing values:\n")
                    for item in self.report['patterns']['consecutive_missing'][:10]:
                        f.write(f"  {item['column']}: {item['max_consecutive']} consecutive\n")
                f.write("\n")

        print(f"✅ Report saved as '{filename}'")

def main():
    """Main function to run the analysis."""
    analyzer, MissingValuesAnalyzer()

    # Try to load data from common locations
    data_paths = [
        'data/processed_data.csv',
        'data/features.csv',
        'data/training_data.csv',
        'data_with_targets.csv'
    ]

    data_loaded, False
    for path in data_paths:
        if analyzer.load_data(path):
            data_loaded, True
            break

    if not data_loaded:
        print(warning("Could not find data file. Please specify the path to your dataset."))
        print("Common locations checked:")
        for path in data_paths:
            print(f"  - {path}")
        return

    # Run analysis
    analyzer.analyze_missing_values()

    # Save report
    analyzer.save_report()

if __name__ == "__main__":
    main()
