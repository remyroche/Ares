#!/usr/bin/env python3
"""
Data Collection Quality Analysis Report
Analyzes the quality, completeness, and reliability of collected financial data.
"""

from datetime import datetime, timedelta
from pathlib import Path
import glob
import json
import os
import warnings

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.warning_symbols import (
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

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataCollectionQualityAnalyzer:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="datacollectionqualityanalyzer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize DataCollectionQualityAnalyzer."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    passpass  # TODO: Add proper implementation
    def __init__(...):
    passself.data = {}
        self.report = {}
        self.data_sources = ['klines', 'agg_trades', 'futures']


    def load_data(...):
    pass"""Load the collected data for analysis."""
        try:
    passpasspass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
            if data_path.endswith('.pkl'):
    passwith open(data_path, 'rb') as f:
    passself.data = pickle.load(f)
            elif data_path.endswith('.csv'):
    passpassself.data['klines'] = pd.read_csv(data_path)
            else:
    pass# Try to load from directory structure
                self._load_from_directory(data_path)

            print(f"✅ Data loaded successfully")
            for source, df in self.data.items():
    passif df is not None and not df.empty:
    passprint(f"   - {source}: {len(df)} rows, {len(df.columns)} columns")
            return True
        except Exception as e:
    passpasspasspasspasspasspassprint(warning(f"Error loading data: {e}"))
            return False


    def _load_from_directory(...):
    pass"""Load data from directory structure."""
        # Look for common data file patterns
        patterns = {
            'klines': ['*klines*.csv', '*candles*.csv', '*ohlcv*.csv'],
            'agg_trades': ['*agg_trades*.csv', '*trades*.csv'],
            'futures': ['*futures*.csv', '*funding*.csv']
        }

        for source, pattern_list in patterns.items():
    passfor pattern in pattern_list:
    passfiles = glob.glob(os.path.join(data_dir, pattern))
                if files:
    passtry:
    passself.data[source] = pd.read_csv(files[0])
                        print(f"Found {source} data: {files[0]}")
                        break
                    except Exception as e:
    passpasspasspasspasspasspassprint(f"Error loading {source}: {e}")


    def analyze_data_quality(...):
    pass"""Comprehensive data collection quality analysis."""
        if not self.data:
    passprint(warning("No data loaded. Please load data first."))
            return

        print("\n" + "="*60)
        print("🔍 DATA COLLECTION QUALITY ANALYSIS REPORT")
        print("="*60)

        # 1. Data completeness analysis
        self._analyze_data_completeness()

        # 2. Data freshness analysis
        self._analyze_data_freshness()

        # 3. Data format validation
        self._validate_data_formats()

        # 4. Data source reliability
        self._analyze_data_source_reliability()

        # 5. Data consistency checks
        self._check_data_consistency()

        # 6. Data quality metrics
        self._calculate_quality_metrics()

        # 7. Generate recommendations
        self._generate_recommendations()

        # 8. Create visualizations
        self._create_visualizations()


    def _analyze_data_completeness(...):
    pass"""Analyze data completeness across all sources."""
        print("\n📊 DATA COMPLETENESS ANALYSIS")
        print("-" * 40)

        completeness_stats = {}

        for source, df in self.data.items():
    passif df is None or df.empty:
    passcompleteness_stats[source] = {
                    'total_rows': 0,
                    'missing_rows': 0,
                    'completeness_percentage': 0,
                    'date_range': None,
                    'expected_rows': 0
                }
                continue

        # Calculate expected rows based on time range
        if 'timestamp' in df.columns or 'time' in df.columns:
    passtime_col = 'timestamp' if 'timestamp' in df.columns else 'time'
                df[time_col] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')

        if not df[time_col].isna().all():
    passdate_range, df[time_col].max() - df[time_col].min()
                    expected_rows, self._estimate_expected_rows(source, date_range)
                else:
    passexpected_rows, len(df)
            else:
    passexpected_rows, len(df)

            missing_rows, max(0, expected_rows - len(df))
            completeness_pct = (len(df) / expected_rows * 100) if expected_rows > 0 else 0

            completeness_stats[source] = {
                'total_rows': len(df),
                'missing_rows': missing_rows,
                'completeness_percentage': completeness_pct,
                'date_range': date_range if 'date_range' in locals() else None,
                'expected_rows': expected_rows
            }

        # Print completeness summary
        print(f"{'Source':<15} {'Rows':<10} {'Expected':<10} {'Missing':<10} {'Complete %':<12}")
        print("-" * 60)

        for source, stats in completeness_stats.items():
    passprint(f"{source:<15} {stats['total_rows']:<10,} {stats['expected_rows']:<10,} "
                  f"{stats['missing_rows']:<10,} {stats['completeness_percentage']:<12.1f}")

        self.report['completeness'] = completeness_stats


    def _estimate_expected_rows(...):
    pass"""Estimate expected number of rows based on source type and time range."""
        days, date_range.days

        if source == 'klines':
    pass# Assuming 1-hour candles
        return days * 24
        elif source == 'agg_trades':
    passpass# Trades can vary significantly
        return days * 1000  # Rough estimate
        elif source == 'futures':
    passpass# Funding rates are typically 8-hour intervals
        return days * 3
        else:
    passreturn days * 24  # Default assumption


    def _analyze_data_freshness(...):
    pass"""Analyze data freshness and update frequency."""
        print("\n⏰ DATA FRESHNESS ANALYSIS")
        print("-" * 40)

        freshness_stats = {}

        for source, df in self.data.items():
    passif df is None or df.empty:
    passfreshness_stats[source] = {
                    'latest_timestamp': None,
                    'oldest_timestamp': None,
                    'data_age_hours': None,
                    'update_frequency': None,
                    'freshness_score': 0
                }
                continue

        # Find timestamp column
            time_col, None
        for col in df.columns:
    passif 'time' in col.lower() or 'timestamp' in col.lower():
    passtime_col, col
                    break

        if time_col:
    passtry:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
                    df[time_col] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
                    latest_time, df[time_col].max()
                    oldest_time, df[time_col].min()

        if pd.notna(latest_time) and pd.notna(oldest_time):
    passdata_age, datetime.now() - latest_time
                        data_age_hours, data_age.total_seconds() / 3600

        # Calculate update frequency
                        time_diff, df[time_col].diff().dropna()
                        avg_interval, time_diff.mean().total_seconds() / 3600  # hours

        # Freshness score (0-100)
        if data_age_hours < 1:
    passfreshness_score, 100
                        elif data_age_hours < 24:
    passpassfreshness_score, 80
                        elif data_age_hours < 168:  # 1 week
                            freshness_score, 60
                        else:
    passfreshness_score, 20

                        freshness_stats[source] = {
                            'latest_timestamp': latest_time,
                            'oldest_timestamp': oldest_time,
                            'data_age_hours': data_age_hours,
                            'update_frequency_hours': avg_interval,
                            'freshness_score': freshness_score
                        }
                    else:
    passfreshness_stats[source] = {
                            'latest_timestamp': None,
                            'oldest_timestamp': None,
                            'data_age_hours': None,
                            'update_frequency_hours': None,
                            'freshness_score': 0
                        }
        except Exception as e:
    passpasspasspasspasspasspassprint(f"Error analyzing freshness for {source}: {e}")
                    freshness_stats[source] = {
                        'latest_timestamp': None,
                        'oldest_timestamp': None,
                        'data_age_hours': None,
                        'update_frequency_hours': None,
                        'freshness_score': 0
                    }

        # Print freshness summary
        print(f"{'Source':<15} {'Latest':<20} {'Age (hrs)':<12} {'Freq (hrs)':<12} {'Score':<8}")
        print("-" * 70)

        for source, stats in freshness_stats.items():
    passlatest_str, str(stats['latest_timestamp'])[:19] if stats['latest_timestamp'] else 'N/A'
            age_str, f"{stats['data_age_hours']:.1f}" if stats['data_age_hours'] else 'N/A'
            freq_str, f"{stats['update_frequency_hours']:.2f}" if stats['update_frequency_hours'] else 'N/A'
            score_str, f"{stats['freshness_score']}" if stats['freshness_score'] else 'N/A'

            print(f"{source:<15} {latest_str:<20} {age_str:<12} {freq_str:<12} {score_str:<8}")

        self.report['freshness'] = freshness_stats


    def _validate_data_formats(...):
    pass"""Validate data formats and structure."""
        print("\n📋 DATA FORMAT VALIDATION")
        print("-" * 40)

        format_stats = {}

        for source, df in self.data.items():
    passif df is None or df.empty:
    passformat_stats[source] = {
                    'format_valid': False,
                    'required_columns': [],
                    'missing_columns': [],
                    'data_types': {},
                    'format_score': 0
                }
                continue

        # Define required columns for each source
            required_columns = {
                'klines': ['open', 'high', 'low', 'close', 'volume'],
                'agg_trades': ['price', 'quantity', 'timestamp'],
                'futures': ['fundingRate', 'timestamp']
            }

            required_cols, required_columns.get(source, [])

        # Check for missing columns, but account for timestamp being in index
            missing_cols = []
        for col in required_cols:
    passif col == 'timestamp':
    pass# Check if timestamp is either in columns or is the index
        if col not in df.columns and (df.index.name != 'timestamp' and 'timestamp' not in str(type(df.index))):
    passmissing_cols.append(col)
                else:
    passif col not in df.columns:
    passmissing_cols.append(col)

        # Check data types
            data_types = {}
        for col in df.columns:
    passdata_types[col] = str(df[col].dtype)

        # Calculate format score
            format_score, 100
        if missing_cols:
    passformat_score -= len(missing_cols) * 20

        # Check for common data quality issues
            numeric_cols, df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
    passpass# Check for infinite values
                inf_count, df[numeric_cols].isin([np.inf, -np.inf]).sum().sum()
        if inf_count > 0:
    passpassformat_score -= 10

        # Check for extreme outliers
        for col in numeric_cols:
    passq99, df[col].quantile(0.99)
                    q01, df[col].quantile(0.01)
                    extreme_outliers = ((df[col] > q99 * 10) | (df[col] < q01 / 10)).sum()
        if extreme_outliers > len(df) * 0.01:  # More than 1% extreme outliers
                        format_score -= 5

            format_stats[source] = {
                'format_valid': len(missing_cols) == 0,
                'required_columns': required_cols,
                'missing_columns': missing_cols,
                'data_types': data_types,
                'format_score': max(0, format_score)
            }

        # Print format validation summary
        print(f"{'Source':<15} {'Valid':<8} {'Missing':<15} {'Score':<8}")
        print("-" * 50)

        for source, stats in format_stats.items():
    passvalid_str = "✅" if stats['format_valid'] else "❌"
            missing_str = ", ".join(stats['missing_columns'][:3])
        if len(stats['missing_columns']) > 3:
    passmissing_str += "..."

            print(f"{source:<15} {valid_str:<8} {missing_str:<15} {stats['format_score']:<8}")

        self.report['format_validation'] = format_stats


    def _analyze_data_source_reliability(...):
    pass"""Analyze data source reliability and consistency."""
        print("\n🔍 DATA SOURCE RELIABILITY ANALYSIS")
        print("-" * 40)

        reliability_stats = {}

        for source, df in self.data.items():
    passif df is None or df.empty:
    passreliability_stats[source] = {
                    'reliability_score': 0,
                    'consistency_score': 0,
                    'data_quality_issues': [],
                    'overall_score': 0
                }
                continue

            issues = []
            reliability_score, 100
            consistency_score, 100

        # Check for data gaps
            time_col, None
        for col in df.columns:
    passif 'time' in col.lower() or 'timestamp' in col.lower():
    passtime_col, col
                    break

        if time_col:
    passtry:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
                    df[time_col] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
                    time_diff, df[time_col].diff().dropna()

        # Check for large gaps
                    large_gaps, time_diff[time_diff > timedelta(hours=2)]
        if len(large_gaps) > 0:
    passpassissues.append(f"Found {len(large_gaps)} large time gaps")
                        reliability_score -= len(large_gaps) * 2

        # Check for duplicate timestamps
                    duplicates, df[time_col].duplicated().sum()
        if duplicates > 0:
    passpassissues.append(f"Found {duplicates} duplicate timestamps")
                        reliability_score -= duplicates

        except Exception as e:
    passpasspasspasspasspasspassissues.append(f"Time column processing error: {e}")
                    reliability_score -= 20

        # Check for missing values in critical columns
            critical_cols = ['open', 'high', 'low', 'close', 'volume'] if source == 'klines' else ['price', 'quantity']
        for col in critical_cols:
    passpassif col in df.columns:
    passmissing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 5:
    passissues.append(f"High missing values in {col}: {missing_pct:.1f}%")
                        reliability_score -= missing_pct

        # Check for data consistency
        if source == 'klines':
    passpass# Check OHLC consistency
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
    passpassinvalid_ohlc = ((df['high'] < df['low']) |
                                  (df['open'] > df['high']) |
                                  (df['close'] > df['high']) |
                                  (df['open'] < df['low']) |
                                  (df['close'] < df['low'])).sum()

        if invalid_ohlc > 0:
    passissues.append(f"Found {invalid_ohlc} invalid OHLC combinations")
                        consistency_score -= invalid_ohlc

            reliability_stats[source] = {
                'reliability_score': max(0, reliability_score),
                'consistency_score': max(0, consistency_score),
                'data_quality_issues': issues,
                'overall_score': (reliability_score + consistency_score) / 2
            }

        # Print reliability summary
        print(f"{'Source':<15} {'Reliability':<12} {'Consistency':<12} {'Issues':<20}")
        print("-" * 60)

        for source, stats in reliability_stats.items():
    passissues_str = ", ".join(stats['data_quality_issues'][:2])
        if len(stats['data_quality_issues']) > 2:
    passissues_str += "..."

            print(f"{source:<15} {stats['reliability_score']:<12.1f} {stats['consistency_score']:<12.1f} {issues_str:<20}")

        self.report['reliability'] = reliability_stats


    def _check_data_consistency(...):
    pass"""Check data consistency across different sources."""
        print("\n🔄 DATA CONSISTENCY CHECKS")
        print("-" * 40)

        if len(self.data) < 2:
    passprint("Need at least 2 data sources for consistency checks.")
            return

        consistency_issues = []

        # Check timestamp alignment
        time_cols = {}
        for source, df in self.data.items():
    passif df is not None and not df.empty:
    passfor col in df.columns:
    passif 'time' in col.lower() or 'timestamp' in col.lower():
    passtime_cols[source] = col
                        break

        if len(time_cols) >= 2:
    pass# Compare time ranges
            time_ranges = {}
        for source, time_col in time_cols.items():
    passtry:
    passdf, self.data[source]
                    df[time_col] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
                    time_ranges[source] = (df[time_col].min(), df[time_col].max())
        except Exception as e:
    passpasspasspasspasspasspassprint(f"Error processing time column for {source}: {e}")

        if len(time_ranges) >= 2:
    passsources, list(time_ranges.keys())
        for i in range(len(sources)):
    passfor j in range(i+1, len(sources)):
    passsource1, source2, sources[i], sources[j]
                        start1, end1, time_ranges[source1]
                        start2, end2, time_ranges[source2]

                        overlap_start, max(start1, start2)
                        overlap_end, min(end1, end2)

        if overlap_start >= overlap_end:
    passconsistency_issues.append(f"No time overlap between {source1} and {source2}")
                        else:
    passoverlap_duration, overlap_end - overlap_start
                            total_duration, min(end1 - start1, end2 - start2)
                            overlap_percentage = (overlap_duration / total_duration) * 100

        if overlap_percentage < 80:
    passconsistency_issues.append(f"Low time overlap between {source1} and {source2}: {overlap_percentage:.1f}%")

        # Check for price consistency between klines and agg_trades
        if 'klines' in self.data and 'agg_trades' in self.data:
    passpassklines_df, self.data['klines']
            trades_df, self.data['agg_trades']

        if not klines_df.empty and not trades_df.empty:
    passif 'close' in klines_df.columns and 'price' in trades_df.columns:
    pass# Sample comparison
                    klines_sample, klines_df['close'].sample(min(1000, len(klines_df)))
                    trades_sample, trades_df['price'].sample(min(1000, len(trades_df)))

                    klines_mean, klines_sample.mean()
                    trades_mean, trades_sample.mean()

        if abs(klines_mean - trades_mean) / klines_mean > 0.1:  # 10% difference
                        consistency_issues.append(f"Significant price difference between klines and trades: {((trades_mean - klines_mean) / klines_mean * 100):.1f}%")

        if consistency_issues:
    passprint("Consistency issues found:")
        for issue in consistency_issues:
    passprint(f"  ⚠️  {issue}")
        else:
    passprint("✅ No major consistency issues found")

        self.report['consistency_issues'] = consistency_issues


    def _calculate_quality_metrics(...):
    pass"""Calculate overall quality metrics."""
        print("\n📈 OVERALL QUALITY METRICS")
        print("-" * 40)

        quality_scores = {}

        for source in self.data_sources:
    passif source not in self.data or self.data[source] is None or self.data[source].empty:
    passquality_scores[source] = 0
                continue

        # Calculate composite quality score
            completeness, self.report.get('completeness', {}).get(source, {}).get('completeness_percentage', 0)
            freshness, self.report.get('freshness', {}).get(source, {}).get('freshness_score', 0)
            format_valid, self.report.get('format_validation', {}).get(source, {}).get('format_score', 0)
            reliability, self.report.get('reliability', {}).get(source, {}).get('overall_score', 0)

        # Weighted average
            quality_score = (completeness * 0.3 + freshness * 0.25 +
                           format_valid * 0.25 + reliability * 0.2)

            quality_scores[source] = quality_score

        # Print quality summary
        print(f"{'Source':<15} {'Quality Score':<15} {'Status':<10}")
        print("-" * 40)

        for source, score in quality_scores.items():
    passif score >= 80:
    passstatus = "✅ Excellent"
            elif score >= 60:
    passpassstatus = "⚠️  Good"
            elif score >= 40:
    passpassstatus = "⚠️  Fair"
            else:
    passstatus = "❌ Poor"

            print(f"{source:<15} {score:<15.1f} {status:<10}")

        # Overall pipeline quality
        overall_quality, np.mean(list(quality_scores.values()))
        print(f"\nOverall Pipeline Quality: {overall_quality:.1f}/100")

        if overall_quality >= 80:
    passprint("🎉 Excellent data collection quality!")
        elif overall_quality >= 60:
    passpassprint("✅ Good data collection quality")
        elif overall_quality >= 40:
    passpassprint(warning(" Fair data collection quality - consider improvements")))
        else:
    passprint(warning("Poor data collection quality - immediate attention required")))

        self.report['quality_scores'] = quality_scores
        self.report['overall_quality'] = overall_quality


    def _generate_recommendations(...):
    pass"""Generate recommendations based on analysis."""
        print("\n💡 RECOMMENDATIONS")
        print("-" * 40)

        recommendations = []

        # Check completeness
        completeness, self.report.get('completeness', {})
        for source, stats in completeness.items():
    passif stats['completeness_percentage'] < 80:
    passrecommendations.append(f"📊 {source}: Improve data completeness (currently {stats['completeness_percentage']:.1f}%)")

        # Check freshness
        freshness, self.report.get('freshness', {})
        for source, stats in freshness.items():
    passif stats.get('freshness_score', 0) < 60:
    passrecommendations.append(f"⏰ {source}: Data is stale ({stats.get('data_age_hours', 0):.1f} hours old)")

        # Check format validation
        format_validation, self.report.get('format_validation', {})
        for source, stats in format_validation.items():
    passif not stats['format_valid']:
    passmissing_cols = ", ".join(stats['missing_columns'])
                recommendations.append(f"📋 {source}: Missing required columns: {missing_cols}")

        # Check reliability
        reliability, self.report.get('reliability', {})
        for source, stats in reliability.items():
    passif stats['overall_score'] < 70:
    passrecommendations.append(f"🔍 {source}: Data reliability issues detected")

        # Check consistency
        consistency_issues, self.report.get('consistency_issues', [])
        if consistency_issues:
    passrecommendations.append("🔄 Data consistency issues detected between sources")

        if not recommendations:
    passprint("✅ No major issues detected. Data collection quality is good!")
        else:
    passprint("Recommendations for improvement:")
        for rec in recommendations:
    passprint(f"  {rec}")

        self.report['recommendations'] = recommendations


    def _create_visualizations(...):
    pass"""Create visualizations for the report."""
        print("\n📈 GENERATING VISUALIZATIONS...")

        try:
    passpasspass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
        # Create figure with subplots
            fig, axes, plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Data Collection Quality Analysis Report', fontsize=16, fontweight='bold')

        # 1. Quality scores by source
            quality_scores, self.report.get('quality_scores', {})
        if quality_scores:
    passpasssources, list(quality_scores.keys())
                scores, list(quality_scores.values())

                colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in scores]
                axes[0, 0].bar(sources, scores, color=colors)
                axes[0, 0].set_ylabel('Quality Score')
                axes[0, 0].set_title('Data Quality by Source')
                axes[0, 0].set_ylim(0, 100)
                axes[0, 0].grid(True, alpha=0.3)

        # 2. Completeness comparison
            completeness, self.report.get('completeness', {})
        if completeness:
    passpasssources, list(completeness.keys())
                completeness_pcts = [completeness[source].get('completeness_percentage', 0) for source in sources]

                axes[0, 1].bar(sources, completeness_pcts, color='skyblue')
                axes[0, 1].set_ylabel('Completeness (%)')
                axes[0, 1].set_title('Data Completeness by Source')
                axes[0, 1].set_ylim(0, 100)
                axes[0, 1].grid(True, alpha=0.3)

        # 3. Freshness scores
            freshness, self.report.get('freshness', {})
        if freshness:
    passpasssources, list(freshness.keys())
                freshness_scores = [freshness[source].get('freshness_score', 0) for source in sources]

                axes[1, 0].bar(sources, freshness_scores, color='lightgreen')
                axes[1, 0].set_ylabel('Freshness Score')
                axes[1, 0].set_title('Data Freshness by Source')
                axes[1, 0].set_ylim(0, 100)
                axes[1, 0].grid(True, alpha=0.3)

        # 4. Overall quality pie chart
            overall_quality, self.report.get('overall_quality', 0)
        if overall_quality > 0:
    passpassaxes[1, 1].pie([overall_quality, 100 - overall_quality],
                               labels=['Quality Score', 'Remaining'],
                               autopct='%1.1f%%',
                               colors=['lightblue', 'lightgray'])
                axes[1, 1].set_title('Overall Pipeline Quality')

            plt.tight_layout()
            plt.savefig('data_collection_quality_report.png', dpi=300, bbox_inches='tight')
            print("✅ Visualizations saved as 'data_collection_quality_report.png'")

        except Exception as e:
    passpasspasspasspasspasspassprint(warning("Error creating visualizations: {e}")))


    def save_report(...):
    pass"""Save the analysis report to a file."""
        with open(filename, 'w') as f:
    passf.write("DATA COLLECTION QUALITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

        # Overall quality
            overall_quality, self.report.get('overall_quality', 0)
            f.write(f"Overall Pipeline Quality: {overall_quality:.1f}/100\n\n")

        # Quality scores
            quality_scores, self.report.get('quality_scores', {})
            f.write("QUALITY SCORES BY SOURCE:\n")
        for source, score in quality_scores.items():
    passf.write(f"{source}: {score:.1f}/100\n")
            f.write("\n")

        # Completeness
            completeness, self.report.get('completeness', {})
            f.write("COMPLETENESS ANALYSIS:\n")
        for source, stats in completeness.items():
    passf.write(f"{source}: {stats.get('completeness_percentage', 0):.1f}% complete\n")
            f.write("\n")

        # Freshness
            freshness, self.report.get('freshness', {})
            f.write("FRESHNESS ANALYSIS:\n")
        for source, stats in freshness.items():
    passage_hours, stats.get('data_age_hours', 0)
                f.write(f"{source}: {age_hours:.1f} hours old\n")
            f.write("\n")

        # Recommendations
            recommendations, self.report.get('recommendations', [])
        if recommendations:
    passf.write("RECOMMENDATIONS:\n")
        for rec in recommendations:
    passf.write(f"- {rec}\n")
            f.write("\n")

        print(f"✅ Report saved as '{filename}'")

def main(...):
    pass"""Main function to run the analysis."""
    analyzer, DataCollectionQualityAnalyzer()

    # Try to load data from common locations
    data_paths = [
        'data/collected_data.pkl',
        'data/processed_data.pkl',
        'data/training_data.pkl',
        'data/'
    ]

    data_loaded, False
    for path in data_paths:
    passif os.path.exists(path):
    passif analyzer.load_data(path):
    passdata_loaded, True
                break

    if not data_loaded:
    passprint(warning("Could not find data file. Please specify the path to your collected data.")))
        print("Common locations checked:")
        for path in data_paths:
    passprint(f"  - {path}")
        return

    # Run analysis
    analyzer.analyze_data_quality()

    # Save report
    analyzer.save_report()

if __name__ == "__main__":
    passmain()
