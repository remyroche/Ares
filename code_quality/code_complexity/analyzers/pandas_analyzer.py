"""
Pandas Analyzer for Code Complexity Analysis
Metrics data analysis and visualization using pandas
"""

import logging
from pathlib import Path

try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PandasAnalyzer:
    """Analyzer for pandas-based metrics data analysis and visualization
    
    Pandas provides:
    - Advanced data analysis and statistics
    - Data aggregation and grouping
    - Correlation analysis
    - Trend analysis and time series
    - Data visualization support
    - Export to various formats (CSV, Excel, etc.)
    """
    
    def __init__(self, config):
        """Initialize Pandas analyzer"""
        self.config = config
        self.tool_name = "pandas"
        
    def is_available(self) -> bool:
        """Check if Pandas is available"""
        return PANDAS_AVAILABLE
        
    def analyze_metrics_data(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics data using pandas"""
        if not self.is_available():
            logger.warning("Pandas is not available")
            return {}
            
        try:
            # Convert metrics data to DataFrame
            df = self._create_metrics_dataframe(metrics_data)
            
            if df.empty:
                return {}
                
            # Perform comprehensive analysis
            analysis_results = {
                'descriptive_statistics': self._get_descriptive_statistics(df),
                'correlation_analysis': self._get_correlation_analysis(df),
                'complexity_distribution': self._get_complexity_distribution(df),
                'trend_analysis': self._get_trend_analysis(df),
                'outlier_detection': self._get_outlier_detection(df),
                'aggregated_metrics': self._get_aggregated_metrics(df)
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing metrics data with pandas: {e}")
            return {}
            
    def _create_metrics_dataframe(self, metrics_data: Dict[str, Any]) -> 'pd.DataFrame':
        """Create pandas DataFrame from metrics data"""
        if not PANDAS_AVAILABLE:
            return pd.DataFrame()
            
        rows = []
        
        # Extract file-level metrics
        file_analysis = metrics_data.get('file_analysis', {})
        for file_path, metrics in file_analysis.items():
            row = {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'directory': str(Path(file_path).parent),
                'pyexamine_score': metrics.get('pyexamine_score'),
                'radon_cc': metrics.get('radon_cc'),
                'radon_mi': metrics.get('radon_mi'),
                'xenon_score': metrics.get('xenon_score'),
                'combined_score': metrics.get('combined_score')
            }
            
            # Add Halstead metrics if available
            halstead = metrics.get('halstead_metrics', {})
            if halstead:
                row.update({
                    'halstead_volume': halstead.get('volume'),
                    'halstead_difficulty': halstead.get('difficulty'),
                    'halstead_effort': halstead.get('effort'),
                    'halstead_time': halstead.get('time'),
                    'halstead_bugs': halstead.get('bugs')
                })
                
            # Add raw metrics if available
            raw_metrics = metrics.get('raw_metrics', {})
            if raw_metrics:
                row.update({
                    'lines_of_code': raw_metrics.get('loc'),
                    'comment_lines': raw_metrics.get('comments'),
                    'blank_lines': raw_metrics.get('blanks'),
                    'total_lines': raw_metrics.get('total')
                })
                
            rows.append(row)
            
        return pd.DataFrame(rows)
        
    def _get_descriptive_statistics(self, df: 'pd.DataFrame') -> Dict[str, Any]:
        """Get descriptive statistics for all metrics"""
        if df.empty:
            return {}
            
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_columns].describe()
        
        return {
            'summary_statistics': stats.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
    def _get_correlation_analysis(self, df: 'pd.DataFrame') -> Dict[str, Any]:
        """Get correlation analysis between metrics"""
        if df.empty:
            return {}
            
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return {}
            
        correlation_matrix = df[numeric_columns].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'metric1': correlation_matrix.columns[i],
                        'metric2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
                    
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
        
    def _get_complexity_distribution(self, df: 'pd.DataFrame') -> Dict[str, Any]:
        """Get complexity distribution analysis"""
        if df.empty or 'combined_score' not in df.columns:
            return {}
            
        # Remove null values
        scores = df['combined_score'].dropna()
        
        if scores.empty:
            return {}
            
        # Categorize complexity levels
        low_complexity = len(scores[scores >= 0.7])
        medium_complexity = len(scores[(scores >= 0.4) & (scores < 0.7)])
        high_complexity = len(scores[scores < 0.4])
        
        return {
            'total_files': len(scores),
            'low_complexity': {
                'count': low_complexity,
                'percentage': (low_complexity / len(scores)) * 100
            },
            'medium_complexity': {
                'count': medium_complexity,
                'percentage': (medium_complexity / len(scores)) * 100
            },
            'high_complexity': {
                'count': high_complexity,
                'percentage': (high_complexity / len(scores)) * 100
            },
            'statistics': {
                'mean': scores.mean(),
                'median': scores.median(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
        }
        
    def _get_trend_analysis(self, df: 'pd.DataFrame') -> Dict[str, Any]:
        """Get trend analysis (if historical data is available)"""
        if df.empty:
            return {}
            
        # This would be enhanced with historical data from Wily
        # For now, provide basic trend indicators
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        trends = {}
        for column in numeric_columns:
            values = df[column].dropna()
            if len(values) > 1:
                # Simple trend calculation
                trend_direction = 'increasing' if values.iloc[-1] > values.iloc[0] else 'decreasing'
                trends[column] = {
                    'direction': trend_direction,
                    'change': values.iloc[-1] - values.iloc[0],
                    'change_percentage': ((values.iloc[-1] - values.iloc[0]) / values.iloc[0]) * 100
                }
                
        return trends
        
    def _get_outlier_detection(self, df: 'pd.DataFrame') -> Dict[str, Any]:
        """Detect outliers in complexity metrics"""
        if df.empty:
            return {}
            
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for column in numeric_columns:
            values = df[column].dropna()
            if len(values) > 4:  # Need at least 5 values for outlier detection
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (values < lower_bound) | (values > upper_bound)
                outlier_files = df[outlier_mask]['file_path'].tolist()
                
                if outlier_files:
                    outliers[column] = {
                        'count': len(outlier_files),
                        'files': outlier_files,
                        'bounds': {
                            'lower': lower_bound,
                            'upper': upper_bound
                        }
                    }
                    
        return outliers
        
    def _get_aggregated_metrics(self, df: 'pd.DataFrame') -> Dict[str, Any]:
        """Get aggregated metrics by directory"""
        if df.empty:
            return {}
            
        # Group by directory
        directory_metrics = df.groupby('directory').agg({
            'combined_score': ['mean', 'median', 'std', 'count'],
            'radon_cc': ['mean', 'max'],
            'radon_mi': ['mean', 'min'],
            'lines_of_code': ['sum', 'mean'] if 'lines_of_code' in df.columns else ['count']
        }).round(3)
        
        # Flatten column names
        directory_metrics.columns = ['_'.join(col).strip() for col in directory_metrics.columns]
        
        return {
            'by_directory': directory_metrics.to_dict('index'),
            'overall_summary': {
                'total_files': len(df),
                'total_directories': df['directory'].nunique(),
                'average_complexity': df['combined_score'].mean() if 'combined_score' in df.columns else None
            }
        }
        
    def export_to_csv(self, df: 'pd.DataFrame', output_path: str) -> bool:
        """Export DataFrame to CSV"""
        if not self.is_available() or df.empty:
            return False
            
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Metrics data exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
            
    def export_to_excel(self, df: 'pd.DataFrame', output_path: str) -> bool:
        """Export DataFrame to Excel with multiple sheets"""
        if not self.is_available() or df.empty:
            return False
            
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # Summary statistics sheet
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                if not numeric_columns.empty:
                    df[numeric_columns].describe().to_excel(writer, sheet_name='Statistics')
                    
                # Correlation matrix sheet
                if len(numeric_columns) > 1:
                    df[numeric_columns].corr().to_excel(writer, sheet_name='Correlations')
                    
            logger.info(f"Metrics data exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return False