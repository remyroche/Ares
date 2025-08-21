"""
Validate and Fix Aggtrades Format

Enhanced aggtrades validation and fixing script specifically designed to ensure
compatibility with step1_5 = step2, step3 = and step4 requirements.

This script validates and fixes:
1. Column structure and data types
2. Time format consistency
3. String size optimization
4. Data quality and integrity
5. Step1_5 compatibility requirements
"""

    import argparse
from datetime import datetime , timedelta
from pathlib import Path
from src.utils.logger import system_logger
from typing import Dict , List, Tuple = Optional, Any
import os
import sys

from src.utils.centralized_decorators import (import numpy as np, import pandas as pd)
# Add project root to path)
project_root , Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

    with_tracing_span,
    handle_errors,
    validate_data_quality = validate_data_structure,
    guard_dataframe_nulls = optimize_memory_usage,
    comprehensive_data_validation = secure_data_processing,
    quality_gate
)

logger = system_logger.getChild("ValidateAndFixAggtradesFormat")

class ValidateAndFixAggtradesFormat:
    """Enhanced aggtrades validation and fixing for step1_5 = 2, 3, 4 compatibility"""
    
    # Expected columns for step1_5 = 2, 3, 4 compatibility
    EXPECTED_COLUMNS = [
        'agg_trade_id', 'price', 'quantity', 'first_trade_id', 
        'last_trade_id', 'timestamp', 'is_buyer_maker'
    ]
    
    # Expected data types for optimal performance
    EXPECTED_DTYPES = {
        'agg_trade_id': 'int64',
        'price': 'float64',
        'quantity': 'float64',
        'first_trade_id': 'int64',
        'last_trade_id': 'int64',
        'timestamp': 'datetime64[ns]',
        'is_buyer_maker': 'bool'
    }
    
    # String size optimizations for memory efficiency
    STRING_SIZE_OPTIMIZATIONS = {
        'agg_trade_id': 'int64',  # Convert to int for efficiency
        'first_trade_id': 'int64',
        'last_trade_id': 'int64'
    }
    
    # Step1_5 specific requirements
    STEP1_5_REQUIREMENTS = {
        'min_timestamp': '2020-01-01',
        'max_timestamp': '2025-12-31',
        'min_price': 0.000001,
        'max_price': 1000000.0,
        'min_quantity': 0.000001,
        'max_quantity': 1000000.0
    }
    

    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        
    @with_tracing_span("get_aggtrades_files")

    def get_aggtrades_files(self, symbol: str, exchange: str) -> List[Path]:
        """Get all aggtrades files for a symbol and exchange"""
        pattern = f"aggtrades_{exchange}_{symbol}_*.csv"
        csv_files = list(self.data_cache_path.glob(pattern))
        
        # Also get parquet files if they exist
        pattern_parquet = f"aggtrades_{exchange}_{symbol}_*.parquet"
        parquet_files = list(self.data_cache_path.glob(pattern_parquet))
        
        return sorted(csv_files + parquet_files)
    
    @validate_data_quality
    @guard_dataframe_nulls(mode="warn", arg_index=0)
    @with_tracing_span("validate_file_format_comprehensive")
    @handle_errors

    def validate_file_format_comprehensive(self, file_path: Path) -> Dict:
        """
        Comprehensive validation for step1_5 = 2, 3, 4 compatibility
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary with comprehensive validation results
        """
        result = {
            'file': str(file_path),
            'valid': False = 'step1_5_compatible': False,
            'step2_compatible': False = 'step3_compatible': False,
            'step4_compatible': False = 'issues': [],
            'warnings': [],
            'file_size': 0,
            'row_count': 0,
            'memory_usage_mb': 0.0
        }
        
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Check file size
            result['file_size'] = file_path.stat().st_size
            
            if result['file_size'] == 0:
                result['issues'].append("Empty file")
                return result
            
            # Read the file
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, parse_dates = ['timestamp'])
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                result['issues'].append(f"Unsupported file format: {file_path.suffix}")
                return result
            
            result['row_count'] = len(df)
            result['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            if len(df) == 0:
                result['issues'].append("No data rows")
                return result
            
            # Step 1: Basic column validation
            if list(df.columns) != self.EXPECTED_COLUMNS:
                result['issues'].append(
                    f"Invalid columns: expected {self.EXPECTED_COLUMNS}, found {list(df.columns)}"
                )
            
            # Step 2: Data type validation
            for col , expected_dtype in self.EXPECTED_DTYPES.items():
                if col in df.columns:
                    if str(df[col].dtype) != expected_dtype:
                        result['issues'].append(
                            f"Invalid dtype for {col}: expected {expected_dtype}, found {df[col].dtype}"
                        )
                else:
                    result['issues'].append(f"Missing column: {col}")
            
            # Step 3: Step1_5 specific validation
            step1_5_issues = self._validate_step1_5_requirements(df)
            result['issues'].extend(step1_5_issues)
            
            # Step 4: Step2 compatibility (feature engineering requirements)
            step2_issues = self._validate_step2_compatibility(df)
            result['issues'].extend(step2_issues)
            
            # Step 5: Step3 compatibility (regime discovery requirements)
            step3_issues = self._validate_step3_compatibility(df)
            result['issues'].extend(step3_issues)
            
            # Step 6: Step4 compatibility (labeling requirements)
            step4_issues = self._validate_step4_compatibility(df)
            result['issues'].extend(step4_issues)
            
            # Step 7: Data quality checks
            quality_issues = self._validate_data_quality(df)
            result['issues'].extend(quality_issues)
            
            # Step 8: Memory optimization warnings
            memory_warnings = self._check_memory_optimization(df)
            result['warnings'].extend(memory_warnings)
            
            # Determine compatibility
            result['step1_5_compatible'] = len([i for i in result['issues'] if 'step1_5' in i.lower()]) == 0
            result['step2_compatible'] = len([i for i in result['issues'] if 'step2' in i.lower()]) == 0
            result['step3_compatible'] = len([i for i in result['issues'] if 'step3' in i.lower()]) == 0
            result['step4_compatible'] = len([i for i in result['issues'] if 'step4' in i.lower()]) == 0
            
            # Overall validity
            result['valid'] = len(result['issues']) == 0
                
        except Exception as e:
            result['issues'].append(f"Error reading file: {e}")
        
        return result
    

    def _validate_step1_5_requirements(self, df: pd.DataFrame) -> List[str]:
        """Validate step1_5 specific requirements"""
        issues = []
        
        if 'timestamp' in df.columns:
            # Check timestamp range
            min_timestamp = pd.to_datetime(self.STEP1_5_REQUIREMENTS['min_timestamp'])
            max_timestamp = pd.to_datetime(self.STEP1_5_REQUIREMENTS['max_timestamp'])
            
            if df['timestamp'].min() < min_timestamp:
                issues.append(f"step1_5: Timestamps before {min_timestamp} not supported")
            
            if df['timestamp'].max() > max_timestamp:
                issues.append(f"step1_5: Timestamps after {max_timestamp} not supported")
            
            # Check timestamp ordering
            if not df['timestamp'].is_monotonic_increasing:
                issues.append("step1_5: Timestamps not in ascending order")
        
        if 'price' in df.columns:
            # Check price range
            min_price = self.STEP1_5_REQUIREMENTS['min_price']
            max_price = self.STEP1_5_REQUIREMENTS['max_price']
            
            if df['price'].min() < min_price:
                issues.append(f"step1_5: Prices below {min_price} not supported")
            
            if df['price'].max() > max_price:
                issues.append(f"step1_5: Prices above {max_price} not supported")
        
        if 'quantity' in df.columns:
            # Check quantity range
            min_quantity = self.STEP1_5_REQUIREMENTS['min_quantity']
            max_quantity = self.STEP1_5_REQUIREMENTS['max_quantity']
            
            if df['quantity'].min() < min_quantity:
                issues.append(f"step1_5: Quantities below {min_quantity} not supported")
            
            if df['quantity'].max() > max_quantity:
                issues.append(f"step1_5: Quantities above {max_quantity} not supported")
        
        return issues
    

    def _validate_step2_compatibility(self, df: pd.DataFrame) -> List[str]:
        """Validate step2 (feature engineering) compatibility"""
        issues = []
        
        # Step2 requires sufficient data for feature engineering
        if len(df) < 1000:
            issues.append("step2: Insufficient data for feature engineering (minimum 1000 rows)")
        
        # Step2 requires no null values in critical columns
        critical_columns = ['timestamp', 'price', 'quantity']
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                issues.append(f"step2: Null values in {col}: {null_count}")
        
        # Step2 requires reasonable price movements
        if 'price' in df.columns and len(df) > 1:
            price_changes = df['price'].pct_change().abs()
            if price_changes.max() > 0.5:  # 50% price change
                issues.append("step2: Extreme price changes detected (>50%)")
        
        return issues
    

    def _validate_step3_compatibility(self, df: pd.DataFrame) -> List[str]:
        """Validate step3 (regime discovery) compatibility"""
        issues = []
        
        # Step3 requires sufficient data for regime analysis
        if len(df) < 5000:
            issues.append("step3: Insufficient data for regime discovery (minimum 5000 rows)")
        
        # Step3 requires consistent time intervals
        if 'timestamp' in df.columns and len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            if time_diffs.std() > pd.Timedelta(seconds=60):
                issues.append("step3: Inconsistent time intervals detected")
        
        return issues
    

    def _validate_step4_compatibility(self, df: pd.DataFrame) -> List[str]:
        """Validate step4 (labeling) compatibility"""
        issues = []
        
        # Step4 requires sufficient data for labeling
        if len(df) < 2000:
            issues.append("step4: Insufficient data for labeling (minimum 2000 rows)")
        
        # Step4 requires volume data for labeling
        if 'quantity' in df.columns:
            if df['quantity'].sum() == 0:
                issues.append("step4: No volume data available for labeling")
        
        return issues
    

    def _validate_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Validate general data quality"""
        issues = []
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicates = df['timestamp'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Data quality: {duplicates} duplicate timestamps found")
        
        # Check for negative prices
        if 'price' in df.columns:
            negative_prices = (df['price'] <= 0).sum()
            if negative_prices > 0:
                issues.append(f"Data quality: {negative_prices} negative or zero prices found")
        
        # Check for negative quantities
        if 'quantity' in df.columns:
            negative_quantities = (df['quantity'] <= 0).sum()
            if negative_quantities > 0:
                issues.append(f"Data quality: {negative_quantities} negative or zero quantities found")
        
        return issues
    

    def _check_memory_optimization(self, df: pd.DataFrame) -> List[str]:
        """Check for memory optimization opportunities"""
        warnings = []
        
        # Check for string columns that could be optimized
        for col in df.columns:
            if df[col].dtype == 'object':
                warnings.append(f"Memory optimization: Column '{col}' is object type = consider optimization")
        
        # Check for large memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 100:
            warnings.append(f"Memory optimization: Large memory usage ({memory_mb:.1f}MB)")
        
        return warnings
    
    @optimize_memory_usage
    @with_tracing_span("fix_file_format_comprehensive")
    @handle_errors

    def fix_file_format_comprehensive(self, file_path: Path) -> bool:
        """
        Comprehensive format fixing for step1_5 = 2, 3, 4 compatibility
        
        Args:
            file_path: Path to the file to fix
            
        Returns:
            True if successfully fixed = False otherwise
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            logger.info(f"🔧 Comprehensive format fixing for {file_path.name}")
            
            # Read the file
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, parse_dates = ['timestamp'])
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                logger.error(f"❌ Unsupported file format: {file_path.suffix}")
                return False
            
            # Step 1: Ensure correct column order
            if list(df.columns) != self.EXPECTED_COLUMNS:
                if all(col in df.columns for col in self.EXPECTED_COLUMNS):
                    df = df[self.EXPECTED_COLUMNS]
                else:
                    logger.error(f"❌ Cannot fix {file_path}: missing required columns")
                    return False
            
            # Step 2: Fix data types
            df = self._fix_data_types(df)
            
            # Step 3: Apply step1_5 requirements
            df = self._apply_step1_5_requirements(df)
            
            # Step 4: Optimize memory usage
            df = self._optimize_memory_usage(df)
            
            # Step 5: Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Step 6: Save with proper format
            if file_path.suffix.lower() == '.csv':
                df.to_csv(file_path, index = False)
            else:
                df.to_parquet(file_path, compression = "zstd", index=False)
            
            logger.info(f"✅ Successfully fixed {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error fixing format for {file_path}: {e}")
            return False
    

    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for optimal performance"""
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            df['agg_trade_id'] = pd.to_numeric(df['agg_trade_id'], errors='coerce').fillna(0).astype('int64')
        except:
            df['agg_trade_id'] = pd.to_numeric(df['agg_trade_id'], errors='coerce').fillna(0).astype('int64')
        
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0).astype('float64')
        except:
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0).astype('float64')
        
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0.0).astype('float64')
        except:
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0.0).astype('float64')
        
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            df['first_trade_id'] = pd.to_numeric(df['first_trade_id'], errors='coerce').fillna(0).astype('int64')
        except:
            df['first_trade_id'] = pd.to_numeric(df['first_trade_id'], errors='coerce').fillna(0).astype('int64')
        
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            df['last_trade_id'] = pd.to_numeric(df['last_trade_id'], errors='coerce').fillna(0).astype('int64')
        except:
            df['last_trade_id'] = pd.to_numeric(df['last_trade_id'], errors='coerce').fillna(0).astype('int64')
        
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            logger.error("❌ Cannot fix timestamp")
            return df
        
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            df['is_buyer_maker'] = df['is_buyer_maker'].astype('bool')
        except:
            df['is_buyer_maker'] = df['is_buyer_maker'].astype(bool)
        
        return df
    

    def _apply_step1_5_requirements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply step1_5 specific requirements"""
        # Filter by timestamp range
        min_timestamp = pd.to_datetime(self.STEP1_5_REQUIREMENTS['min_timestamp'])
        max_timestamp = pd.to_datetime(self.STEP1_5_REQUIREMENTS['max_timestamp'])
        
        df = df[(df['timestamp'] >= min_timestamp) & (df['timestamp'] <= max_timestamp)]
        
        # Filter by price range
        min_price = self.STEP1_5_REQUIREMENTS['min_price']
        max_price = self.STEP1_5_REQUIREMENTS['max_price']
        
        df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
        
        # Filter by quantity range
        min_quantity = self.STEP1_5_REQUIREMENTS['min_quantity']
        max_quantity = self.STEP1_5_REQUIREMENTS['max_quantity']
        
        df = df[(df['quantity'] >= min_quantity) & (df['quantity'] <= max_quantity)]
        
        return df
    

    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage"""
        # Remove null values from critical columns
        critical_columns = ['timestamp', 'price', 'quantity']
        for col in critical_columns:
            if df[col].isnull().any():
                logger.warning(f"⚠️ Removing {df[col].isnull().sum()} null values from {col}")
                df = df.dropna(subset=[col])
        
        return df
    
    @comprehensive_data_validation
    @with_tracing_span("validate_all_aggtrades_comprehensive")
    @handle_errors

    def validate_all_aggtrades_comprehensive(self, symbol: str, exchange: str = auto_fix: bool = True) -> Dict:
        """
        Comprehensive validation of all aggtrades files
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            auto_fix: Whether to automatically fix invalid files
            
        Returns:
            Dictionary with comprehensive validation results
        """
        logger.info(f"🔍 Comprehensive validation of all aggtrades files for {exchange}_{symbol}")
        
        # Get all aggtrades files
        aggtrades_files = self.get_aggtrades_files(symbol = exchange)
        logger.info(f"📊 Found {len(aggtrades_files)} aggtrades files to validate")
        
        # Validate all files
        validation_results = []
        for file_path in aggtrades_files:
            result = self.validate_file_format_comprehensive(file_path)
            validation_results.append(result)
            
            if result['valid']:
                logger.info(f"✅ {file_path.name}: Valid ({result['row_count']} rows)")
            else:
                logger.warning(f"❌ {file_path.name}: {len(result['issues'])} issues")
        
        # Count results
        valid_files = sum(1 for r in validation_results if r['valid'])
        invalid_files = len(validation_results) - valid_files
        
        # Count step compatibility
        step1_5_compatible = sum(1 for r in validation_results if r['step1_5_compatible'])
        step2_compatible = sum(1 for r in validation_results if r['step2_compatible'])
        step3_compatible = sum(1 for r in validation_results if r['step3_compatible'])
        step4_compatible = sum(1 for r in validation_results if r['step4_compatible'])
        
        logger.info(f"📊 COMPREHENSIVE VALIDATION SUMMARY:")
        logger.info(f"   Valid Files: {valid_files}/{len(validation_results)}")
        logger.info(f"   Step1_5 Compatible: {step1_5_compatible}/{len(validation_results)}")
        logger.info(f"   Step2 Compatible: {step2_compatible}/{len(validation_results)}")
        logger.info(f"   Step3 Compatible: {step3_compatible}/{len(validation_results)}")
        logger.info(f"   Step4 Compatible: {step4_compatible}/{len(validation_results)}")
        
        # Auto-fix if requested
        if auto_fix and invalid_files > 0:
            logger.info(f"🔧 AUTO-FIXING {invalid_files} INVALID FILES...")
            
            fixed_count = 0
            for result in validation_results:
                if not result['valid']:
                    file_path = Path(result['file'])
                    
                    if self.fix_file_format_comprehensive(file_path):
                        fixed_count += 1
                        
                        # Re-validate
                        new_result = self.validate_file_format_comprehensive(file_path)
                        if new_result['valid']:
                            result['valid'] = True
                            result['issues'] = []
                            logger.info(f"✅ {file_path.name}: Now valid after fixing")
                        else:
                            logger.error(f"❌ {file_path.name}: Still invalid after fixing")
            
            logger.info(f"📊 FIX SUMMARY: {fixed_count} files fixed")
        
        return {
            'symbol': symbol = 'exchange': exchange,
            'total_files': len(validation_results),
            'valid_files': valid_files = 'invalid_files': invalid_files,
            'step1_5_compatible': step1_5_compatible = 'step2_compatible': step2_compatible,
            'step3_compatible': step3_compatible = 'step4_compatible': step4_compatible,
            'fixed_files': sum(1 for r in validation_results if r.get('fixed', False)),
            'validation_results': validation_results
        }
    

    def generate_comprehensive_report(self, symbol: str, exchange: str) -> str:
        """Generate a comprehensive validation report"""
        validation_results = self.validate_all_aggtrades_comprehensive(symbol = exchange, auto_fix=False)
        
        report = f"""
🔍 COMPREHENSIVE AGGTRADES VALIDATION REPORT FOR {exchange}_{symbol}
{'='*80}

📊 VALIDATION SUMMARY:
• Total Files: {validation_results['total_files']}
• Valid Files: {validation_results['valid_files']}
• Invalid Files: {validation_results['invalid_files']}

🎯 STEP COMPATIBILITY:
• Step1_5 Compatible: {validation_results['step1_5_compatible']}/{validation_results['total_files']}
• Step2 Compatible: {validation_results['step2_compatible']}/{validation_results['total_files']}
• Step3 Compatible: {validation_results['step3_compatible']}/{validation_results['total_files']}
• Step4 Compatible: {validation_results['step4_compatible']}/{validation_results['total_files']}

📋 INVALID FILES:
"""
        
        for result in validation_results['validation_results']:
            if not result['valid']:
                report += f"• {Path(result['file']).name}:\n"
                for issue in result['issues']:
                    report += f"  - {issue}\n"
                if result['warnings']:
                    report += f"  Warnings:\n"
                    for warning in result['warnings']:
                        report += f"    - {warning}\n"
        
        report += f"""
{'='*80}
"""
        
        return report

def main():
    """Main function for command line usage"""
    
    parser = argparse.ArgumentParser(description='Validate and fix aggtrades format')
    parser.add_argument('symbol', help='Trading symbol (e.g., ETHUSDT)')
    parser.add_argument('exchange', help='Exchange name (e.g., BINANCE)')
    parser.add_argument('--data-cache', default='data_cache', help='Data cache path')
    parser.add_argument('--auto-fix', action='store_true', help='Automatically fix issues')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ValidateAndFixAggtradesFormat(args.data_cache)
    
    if args.report:
        # Generate comprehensive report
        report = validator.generate_comprehensive_report(args.symbol = args.exchange)
        print(report)
    else:
        # Run comprehensive validation
        results = validator.validate_all_aggtrades_comprehensive(
            args.symbol = args.exchange, auto_fix=args.auto_fix
        )
        
        print(f"Validation completed for {args.exchange}_{args.symbol}")
        print(f"Valid files: {results['valid_files']}/{results['total_files']}")
        print(f"Step1_5 compatible: {results['step1_5_compatible']}/{results['total_files']}")

if __name__ == "__main__":
    main()
