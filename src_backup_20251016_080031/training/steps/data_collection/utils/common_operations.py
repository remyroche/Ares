from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Common utility functions for data collection operations.

This module provides shared utility functions used across data collection steps,
including safe JSON loading, file operations, data validation helpers, and
common data processing utilities.
"""

import json
import os

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import collections

# Setup logging
logger = logging.getLogger(__name__)

def safe_json_load(file_path: str, default: Any = None) -> Any:
    """Safely load JSON data from a file with error handling.

    Args:
        file_path: Path to the JSON file
        default: Default value to return if loading fails

    Returns:
        Parsed JSON data or default value if loading fails
    """
    logger.info(f'📁 Loading JSON file: {file_path}')
    
    try:
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ JSON file not found: {file_path}")
            logger.info(f'📊 Returning default value: {default}')
            return default

        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f'📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)')

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"✅ Successfully loaded JSON from: {file_path}")
        logger.info(f'📊 JSON data type: {type(data)}')
        if isinstance(data, dict):
            logger.info(f'📊 JSON keys: {list(data.keys())}')
        elif isinstance(data, list):
            logger.info(f'📊 JSON list length: {len(data)}')
        return data

    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in file {file_path}: {e}")
        logger.error(f"📊 JSON error details: line {e.lineno}, column {e.colno}")
        return default
    except PermissionError as e:
        logger.error(f"❌ Permission denied reading file {file_path}: {e}")
        return default
    except Exception as e:
        logger.error(f"❌ Unexpected error loading JSON from {file_path}: {e}")
        logger.error(f"📊 Error type: {type(e).__name__}")
        return default

def safe_json_save(data: Any, file_path: str, indent: int = 2) -> bool:
    """Safely save data as JSON to a file with error handling.

    Args:
        data: Data to save as JSON
        file_path: Path to save the JSON file
        indent: JSON indentation level

    Returns:
        True if save was successful, False otherwise
    """
    logger.info(f'💾 Saving JSON data to: {file_path}')
    logger.info(f'📊 Data type: {type(data)}')
    
    if isinstance(data, dict):
        logger.info(f'📊 Dictionary keys: {list(data.keys())}')
    elif isinstance(data, list):
        logger.info(f'📊 List length: {len(data)}')
    
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        if dir_path:
            logger.info(f'📁 Creating directory: {dir_path}')
            os.makedirs(dir_path, exist_ok=True)

        # Estimate file size
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)
        estimated_size = len(json_str.encode('utf-8'))
        logger.info(f'📊 Estimated file size: {estimated_size:,} bytes ({estimated_size/1024:.1f} KB)')

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        # Verify file was created and get actual size
        if os.path.exists(file_path):
            actual_size = os.path.getsize(file_path)
            logger.info(f"✅ Successfully saved JSON to: {file_path}")
            logger.info(f'📊 Actual file size: {actual_size:,} bytes ({actual_size/1024:.1f} KB)')
            return True
        else:
            logger.error(f"❌ File was not created: {file_path}")
            return False

    except PermissionError as e:
        logger.error(f"❌ Permission denied writing to file {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error saving JSON to {file_path}: {e}")
        logger.error(f"📊 Error type: {type(e).__name__}")
        return False

def validate_data_directory(data_dir: str, required_subdirs: List[str] = None) -> Dict[str, Any]:
    """Validate data directory structure and permissions.

    Args:
        data_dir: Path to the data directory
        required_subdirs: List of required subdirectories

    Returns:
        Dictionary with validation results
    """
    logger.info(f'🔍 Validating data directory: {data_dir}')
    logger.info(f'📊 Required subdirectories: {required_subdirs or "None"}')
    
    result = {
        'valid': True,
        'exists': False,
        'readable': False,
        'writable': False,
        'missing_subdirs': [],
        'issues': []
    }

    try:
        # Check if directory exists
        logger.info('🔍 1. Checking if directory exists...')
        if not os.path.exists(data_dir):
            logger.warning(f'⚠️ Data directory does not exist: {data_dir}')
            result['issues'].append(f"Data directory does not exist: {data_dir}")
            result['valid'] = False
            return result

        logger.info('✅ Directory exists')
        result['exists'] = True

        # Check if it's actually a directory
        logger.info('🔍 2. Checking if path is a directory...')
        if not os.path.isdir(data_dir):
            logger.error(f'❌ Path exists but is not a directory: {data_dir}')
            result['issues'].append(f"Path exists but is not a directory: {data_dir}")
            result['valid'] = False
            return result

        logger.info('✅ Path is a directory')

        # Check read permissions
        logger.info('🔍 3. Checking read permissions...')
        try:
            files = os.listdir(data_dir)
            result['readable'] = True
            logger.info(f'✅ Read permissions OK - found {len(files)} items')
            logger.info(f'📊 Directory contents: {files[:10]}{"..." if len(files) > 10 else ""}')
        except PermissionError:
            logger.error(f'❌ No read permission for directory: {data_dir}')
            result['issues'].append(f"No read permission for directory: {data_dir}")
            result['valid'] = False

        # Check write permissions
        logger.info('🔍 4. Checking write permissions...')
        try:
            test_file = os.path.join(data_dir, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            result['writable'] = True
            logger.info('✅ Write permissions OK')
        except PermissionError:
            result['issues'].append(f"No write permission for directory: {data_dir}")
            result['valid'] = False

        # Check required subdirectories
        if required_subdirs:
            logger.info('🔍 5. Checking required subdirectories...')
            for subdir in required_subdirs:
                subdir_path = os.path.join(data_dir, subdir)
                if not os.path.exists(subdir_path):
                    logger.warning(f'⚠️ Missing required subdirectory: {subdir}')
                    result['missing_subdirs'].append(subdir)
                elif not os.path.isdir(subdir_path):
                    logger.error(f'❌ Required subdirectory is not a directory: {subdir_path}')
                    result['issues'].append(f"Required subdirectory is not a directory: {subdir_path}")
                else:
                    logger.info(f'✅ Required subdirectory exists: {subdir}')

        if result['missing_subdirs']:
            logger.warning(f'⚠️ Missing required subdirectories: {result["missing_subdirs"]}')
            result['issues'].append(f"Missing required subdirectories: {result['missing_subdirs']}")

        result['valid'] = result['valid'] and len(result['issues']) == 0

        # Final validation summary
        logger.info('📊 Directory validation summary:')
        logger.info(f'   📊 Valid: {result["valid"]}')
        logger.info(f'   📊 Exists: {result["exists"]}')
        logger.info(f'   📊 Readable: {result["readable"]}')
        logger.info(f'   📊 Writable: {result["writable"]}')
        logger.info(f'   📊 Missing subdirs: {len(result["missing_subdirs"])}')
        logger.info(f'   📊 Issues: {len(result["issues"])}')
        
        if result['valid']:
            logger.info('✅ Directory validation passed')
        else:
            logger.error('❌ Directory validation failed')
            for issue in result['issues']:
                logger.error(f'   ❌ {issue}')

        return result

    except Exception as e:
        logger.error(f'❌ Unexpected error validating directory: {str(e)}')
        logger.error(f'📊 Error type: {type(e).__name__}')
        result['issues'].append(f"Unexpected error validating directory: {str(e)}")
        result['valid'] = False
        return result

def ensure_directory_exists(directory_path: str) -> bool:
    """Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory

    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    logger.info(f'📁 Ensuring directory exists: {directory_path}')
    
    try:
        # Check if directory already exists
        if os.path.exists(directory_path):
            if os.path.isdir(directory_path):
                logger.info(f'✅ Directory already exists: {directory_path}')
                return True
            else:
                logger.error(f'❌ Path exists but is not a directory: {directory_path}')
                return False
        
        # Create directory
        logger.info(f'🔧 Creating directory: {directory_path}')
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        
        # Verify creation
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            logger.info(f'✅ Directory created successfully: {directory_path}')
            return True
        else:
            logger.error(f'❌ Directory creation failed: {directory_path}')
            return False
            
    except PermissionError as e:
        logger.error(f"❌ Permission denied creating directory {directory_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error creating directory {directory_path}: {e}")
        logger.error(f"📊 Error type: {type(e).__name__}")
        return False

def get_file_size_mb(file_path: str) -> Optional[float]:
    """Get file size in megabytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in MB, or None if file doesn't exist or can't be accessed
    """
    logger.info(f'📊 Getting file size: {file_path}')
    
    try:
        if not os.path.exists(file_path):
            logger.warning(f'⚠️ File does not exist: {file_path}')
            return None

        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        
        logger.info(f'📊 File size: {size_bytes:,} bytes ({size_mb:.2f} MB)')
        return size_mb

    except Exception as e:
        logger.warning(f"⚠️ Error getting file size for {file_path}: {e}")
        logger.warning(f"📊 Error type: {type(e).__name__}")
        return None

def validate_parquet_file(file_path: str) -> Dict[str, Any]:
    """Validate a parquet file and return basic information.

    Args:
        file_path: Path to the parquet file

    Returns:
        Dictionary with validation results and file information
    """
    result = {
        'valid': False,
        'exists': False,
        'readable': False,
        'row_count': 0,
        'column_count': 0,
        'columns': [],
        'file_size_mb': 0,
        'error': None
    }

    try:
        if not os.path.exists(file_path):
            result['error'] = f"File does not exist: {file_path}"
            return result

        result['exists'] = True
        result['file_size_mb'] = get_file_size_mb(file_path) or 0

        # Try to read the parquet file
        df = standardized_parquet_handler.read_parquet_standardized(file_path)

        result['readable'] = True
        result['row_count'] = len(df)
        result['column_count'] = len(df.columns)
        result['columns'] = list(df.columns)
        result['valid'] = True

        logger.debug(f"Successfully validated parquet file: {file_path}")
        return result

    except Exception as e:
        result['error'] = str(e)
        logger.warning(f"Error validating parquet file {file_path}: {e}")
        return result

def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic data quality metrics for a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with quality metrics
    """
    try:
        metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'missing_values': {},
            'data_types': {},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }

        # Missing values analysis
        missing = df.isnull().sum()
        metrics['missing_values'] = {
            'total_missing': int(missing.sum()),
            'missing_by_column': missing.to_dict(),
            'missing_percentage': (missing.sum() / (len(df) * len(df.columns))) * 100
        }

        # Data types
        metrics['data_types'] = df.dtypes.astype(str).to_dict()

        # Numeric column statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            metrics['numeric_stats'] = {}
            for col in numeric_columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    metrics['numeric_stats'][col] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'negative_count': int((col_data < 0).sum())
                    }

        return metrics

    except Exception as e:
        logger.error(f"Error calculating data quality metrics: {e}")
        return {'error': str(e)}

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return ".1f"
        size_bytes /= 1024.0
    return ".1f"

def get_directory_stats(directory_path: str) -> Dict[str, Any]:
    """Get statistics about a directory and its contents.

    Args:
        directory_path: Path to the directory

    Returns:
        Dictionary with directory statistics
    """
    stats = {
        'total_files': 0,
        'total_size_bytes': 0,
        'file_types': {},
        'subdirectories': 0,
        'errors': []
    }

    try:
        if not os.path.exists(directory_path):
            stats['errors'].append(f"Directory does not exist: {directory_path}")
            return stats

        for root, dirs, files in os.walk(directory_path):
            stats['subdirectories'] += len(dirs)

            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)

                    stats['total_files'] += 1
                    stats['total_size_bytes'] += file_size

                    # Count file types
                    _, ext = os.path.splitext(file)
                    ext = ext.lower() or 'no_extension'
                    stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1

                except Exception as e:
                    stats['errors'].append(f"Error processing file {file_path}: {str(e)}")

        stats['total_size_mb'] = stats['total_size_bytes'] / (1024 * 1024)

        return stats

    except Exception as e:
        stats['errors'].append(f"Error analyzing directory {directory_path}: {str(e)}")
        return stats

def safe_remove_file(file_path: str) -> bool:
    """Safely remove a file with error handling.

    Args:
        file_path: Path to the file to remove

    Returns:
        True if file was removed successfully, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Successfully removed file: {file_path}")
            return True
        else:
            logger.warning(f"File does not exist, cannot remove: {file_path}")
            return False

    except PermissionError as e:
        logger.error(f"Permission denied removing file {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error removing file {file_path}: {e}")
        return False

def create_backup_file(file_path: str, backup_suffix: str = '.backup') -> Optional[str]:
    """Create a backup of a file.

    Args:
        file_path: Path to the original file
        backup_suffix: Suffix for the backup file

    Returns:
        Path to the backup file, or None if backup failed
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Original file does not exist, cannot create backup: {file_path}")
            return None

        backup_path = file_path + backup_suffix

        # Handle existing backups
        counter = 1
        while os.path.exists(backup_path):
            backup_path = f"{file_path}{backup_suffix}.{counter}"
            counter += 1

        import shutil
        shutil.copy2(file_path, backup_path)

        logger.info(f"Created backup: {backup_path}")
        return backup_path

    except Exception as e:
        logger.error(f"Error creating backup for {file_path}: {e}")
        return None
