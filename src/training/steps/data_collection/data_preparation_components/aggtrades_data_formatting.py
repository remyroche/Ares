from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

import csv
import glob
import os
import shutil
from pathlib import Path
import typing
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

def check_file_format(file_path: Union[str, Path]) -> bool | None:
    """Check if a CSV file follows the correct format.
    Returns True if the file is correctly formatted, False otherwise.
    """
    try:
        with open(file_path, encoding='utf-8') as f:
            first_line = f.readline().strip()
            expected_header = 'timestamp,price,quantity,is_buyer_maker,agg_trade_id'
            if first_line != expected_header:
                return False
            for i, line in enumerate(f):
                if i >= 5:
                    break
                line = line.strip()
                if not line:
                    continue
                fields = line.split(',')
                if len(fields) != 5:
                    return False
                timestamp = fields[0]
                if not timestamp:
                    return False
                try:
                    float(fields[1])
                    float(fields[2])
                except ValueError:
                    return False
            return True
    except Exception:
        return False

def detect_file_format(file_path: Union[str, Path]) -> str | None:
    """Detect the format of a CSV file and return the format type.
    Returns: 'correct', 'format1', 'format2', 'format3', or 'unknown'.
    """
    try:
        with open(file_path, encoding='utf-8') as f:
            first_line = f.readline().strip()
        if first_line == 'timestamp,price,quantity,is_buyer_maker,agg_trade_id':
            return 'correct'
            return 'correct'
        if ';' in first_line and 'agg_trade_id' not in first_line:
            return 'format1'
            return 'format1'
        if 'agg_trade_id' in first_line and ';' in first_line:
            return 'format2'
        if first_line == 'timestamp,price,quantity,is_buyer_maker':
            return 'format3'
            return 'format3'
    except Exception:
        return 'unknown'

class DataFileReformatter:
    """Class to handle reformatting of data files with different formats."""
    @log_important_calls

    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.processors = {'format1': self._process_format1, 'format2': self._process_format2, 'format3': self._process_format3}

    def reformat_file(self, format_type: str) -> bool:
        """Main entry point - delegates to specific processor."""
        processor = self.processors.get(format_type)
        if not processor:
            return False
        try:
            with open(self.input_path, encoding='utf-8') as infile, open(self.output_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                return processor(infile, writer)
        except Exception:
            return False
    @log_all_calls

    def _process_format1(self, infile: Any, writer: Any) -> bool:
        """Process semicolon-delimited format."""
        try:
            writer.writerow(['timestamp', 'price', 'quantity', 'is_buyer_maker', 'agg_trade_id'])
            for line in infile:
                line = line.strip()
                if not line or line.startswith('timestamp'):
                    continue
                fields = line.split(';')
                if len(fields) >= 4:
                    timestamp = fields[0]
                    price = fields[1]
                    quantity = fields[2]
                    is_buyer_maker = fields[3]
                    agg_trade_id = f'agg_{timestamp}_{price}_{quantity}'
                    writer.writerow([timestamp, price, quantity, is_buyer_maker, agg_trade_id])
            return True
        except Exception:
            return False
    @log_all_calls

    def _process_format2(self, infile: Any, writer: Any) -> bool:
        """Process mixed-delimiter format with agg_trade_id."""
        try:
            writer.writerow(['timestamp', 'price', 'quantity', 'is_buyer_maker', 'agg_trade_id'])
            for line in infile:
                line = line.strip()
                if not line or line.startswith('timestamp'):
                    continue
                if ',' in line:
                    ts_part, rest = line.split(',', 1)
                else:
                    ts_part, rest = (line, '')
                timestamp = ts_part.replace(';', ' ')
                other_cols = next(csv.reader([rest])) if rest else []
                price = other_cols[0] if len(other_cols) > 0 else ''
                quantity = other_cols[1] if len(other_cols) > 1 else ''
                is_buyer_maker = other_cols[2] if len(other_cols) > 2 else ''
                agg_trade_id = other_cols[3] if len(other_cols) > 3 else f'agg_{timestamp}_{price}_{quantity}'
                writer.writerow([timestamp, price, quantity, is_buyer_maker, agg_trade_id])
            return True
        except Exception:
            return False
    @log_all_calls

    def _process_format3(self, infile: Any, writer: Any) -> bool:
        """Process format missing agg_trade_id column."""
        try:
            writer.writerow(['timestamp', 'price', 'quantity', 'is_buyer_maker', 'agg_trade_id'])
            for line in infile:
                line = line.strip()
                if not line or line.startswith('timestamp'):
                    continue
                fields = line.split(',')
                if len(fields) >= 4:
                    timestamp = fields[0]
                    price = fields[1]
                    quantity = fields[2]
                    is_buyer_maker = fields[3]
                    agg_trade_id = f'agg_{timestamp}_{price}_{quantity}'
                    writer.writerow([timestamp, price, quantity, is_buyer_maker, agg_trade_id])
            return True
        except Exception:
            return False

def auto_reformat_aggtrades_files() -> None:
    """Automatically detect and reformat all aggtrades CSV files that don't follow the correct format."""
    data_cache_dir = 'data_cache'
    backup_dir = 'data_cache/backup_before_reformat'
    os.makedirs(backup_dir, exist_ok = True)
    pattern = os.path.join(data_cache_dir, 'aggtrades_*_*.csv')
    files = glob.glob(pattern)
    files_to_reformat = []
    files_checked = 0
    for file_path in files:
        files_checked += 1
        if not check_file_format(file_path):
            format_type = detect_file_format(file_path)
            if format_type != 'correct':
                files_to_reformat.append((file_path, format_type))
    if not files_to_reformat:
        return
    # Auto-proceed with reformatting in non-interactive mode
    print('\nAuto-proceeding with reformatting in non-interactive mode...')
    for file_path, format_type in files_to_reformat:
        backup_path = os.path.join(backup_dir, os.path.basename(file_path))
        shutil.copy2(file_path, backup_path)
        temp_output = file_path + '.tmp'
        reformatter = DataFileReformatter(file_path, temp_output)
        if reformatter.reformat_file(format_type):
            shutil.move(temp_output, file_path)
        else:
            shutil.copy2(backup_path, file_path)

def auto_reformat_aggtrades_files_for_exchange(exchange: str, symbol: str) -> None:
    """Automatically detect and reformat aggtrades CSV files for a specific exchange and symbol.
    This is a targeted version that only processes files for the specified exchange/symbol.
    """
    data_cache_dir = 'data_cache'
    backup_dir = 'data_cache/backup_before_reformat'
    os.makedirs(backup_dir, exist_ok = True)
    pattern = os.path.join(data_cache_dir, f'aggtrades_{exchange}_{symbol}_*.csv')
    files = glob.glob(pattern)
    files_to_reformat = []
    files_checked = 0
    for file_path in files:
        files_checked += 1
        if not check_file_format(file_path):
            format_type = detect_file_format(file_path)
            if format_type != 'correct':
                files_to_reformat.append((file_path, format_type))
    if not files_to_reformat:
        return
    for file_path, format_type in files_to_reformat:
        backup_path = os.path.join(backup_dir, os.path.basename(file_path))
        shutil.copy2(file_path, backup_path)
        temp_output = file_path + '.tmp'
        reformatter = DataFileReformatter(file_path, temp_output)
        if reformatter.reformat_file(format_type):
            shutil.move(temp_output, file_path)
        else:
            shutil.copy2(backup_path, file_path)

def create_dummy_files(input_dir: Any) -> None:
    """Creates a comprehensive set of dummy CSV files for demonstration and testing purposes.
    
    This function generates realistic trading data in multiple formats to test
    the data formatting and processing pipeline. It creates various scenarios
    including different data formats, edge cases, and realistic trading patterns.
    
    Args:
        input_dir: Directory path where dummy files will be created
    """
    import random
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np
    
    # Clean and create directory
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)
    
    # Generate realistic trading data parameters
    base_price = 3000.0
    base_volume = 0.1
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    
    # Format 1: Semicolon-separated with proper timestamp format
    file1_path = os.path.join(input_dir, 'aggtrades_format1_2025-01-01.csv')
    with open(file1_path, 'w', newline='', encoding='utf-8') as f:
        f.write('timestamp;price;quantity;is_buyer_maker\n')
        
        # Generate 1000 realistic trading records
        current_time = start_time
        current_price = base_price
        
        for i in range(1000):
            # Simulate price movement
            price_change = random.uniform(-0.01, 0.01) * current_price
            current_price = max(current_price + price_change, 1.0)  # Prevent negative prices
            
            # Generate realistic volume
            volume = random.uniform(0.001, 10.0)
            
            # Generate realistic timestamp (every 1-5 seconds)
            current_time += timedelta(seconds=random.uniform(1, 5))
            
            # Random buyer/seller maker
            is_buyer_maker = random.choice([True, False])
            
            f.write(f'{current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]};{current_price:.2f};{volume:.4f};{is_buyer_maker}\n')
    
    # Format 2: Comma-separated with additional fields
    file2_path = os.path.join(input_dir, 'aggtrades_format2_2025-01-02.csv')
    with open(file2_path, 'w', newline='', encoding='utf-8') as f:
        f.write('timestamp,price,quantity,is_buyer_maker,agg_trade_id,first_trade_id,last_trade_id\n')
        
        # Generate 1000 records with trade IDs
        current_time = start_time + timedelta(days=1)
        current_price = base_price * 1.02  # Slight price increase
        trade_id = 1000000
        
        for i in range(1000):
            price_change = random.uniform(-0.005, 0.005) * current_price
            current_price = max(current_price + price_change, 1.0)
            
            volume = random.uniform(0.001, 5.0)
            current_time += timedelta(seconds=random.uniform(0.5, 3))
            
            is_buyer_maker = random.choice([True, False])
            
            # Generate trade IDs
            first_id = trade_id
            last_id = trade_id + random.randint(1, 10)
            trade_id = last_id + 1
            
            f.write(f'{current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]},{current_price:.2f},{volume:.4f},{is_buyer_maker},{trade_id},{first_id},{last_id}\n')
    
    # Format 3: Tab-separated with different structure
    file3_path = os.path.join(input_dir, 'aggtrades_format3_2025-01-03.tsv')
    with open(file3_path, 'w', newline='', encoding='utf-8') as f:
        f.write('time\tprice\tqty\tis_buyer_maker\ttrade_id\n')
        
        current_time = start_time + timedelta(days=2)
        current_price = base_price * 0.98  # Slight price decrease
        trade_id = 2000000
        
        for i in range(500):  # Fewer records for variety
            price_change = random.uniform(-0.008, 0.008) * current_price
            current_price = max(current_price + price_change, 1.0)
            
            volume = random.uniform(0.01, 2.0)
            current_time += timedelta(seconds=random.uniform(2, 8))
            
            is_buyer_maker = random.choice([True, False])
            trade_id += 1
            
            f.write(f'{current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}\t{current_price:.3f}\t{volume:.5f}\t{is_buyer_maker}\t{trade_id}\n')
    
    # Format 4: JSON Lines format
    file4_path = os.path.join(input_dir, 'aggtrades_format4_2025-01-04.jsonl')
    with open(file4_path, 'w', newline='', encoding='utf-8') as f:
        current_time = start_time + timedelta(days=3)
        current_price = base_price * 1.01
        
        for i in range(300):
            price_change = random.uniform(-0.003, 0.003) * current_price
            current_price = max(current_price + price_change, 1.0)
            
            volume = random.uniform(0.005, 1.0)
            current_time += timedelta(seconds=random.uniform(1, 4))
            
            is_buyer_maker = random.choice([True, False])
            
            record = {
                'timestamp': current_time.isoformat(),
                'price': round(current_price, 2),
                'quantity': round(volume, 4),
                'is_buyer_maker': is_buyer_maker,
                'trade_id': 3000000 + i
            }
            
            f.write(f'{json.dumps(record)}\n')
    
    # Format 5: Corrupted/malformed data for testing error handling
    file5_path = os.path.join(input_dir, 'aggtrades_corrupted_2025-01-05.csv')
    with open(file5_path, 'w', newline='', encoding='utf-8') as f:
        f.write('timestamp,price,quantity,is_buyer_maker\n')
        f.write('2025-01-05 10:00:00.000,3000.50,0.1000,false\n')  # Valid
        f.write('2025-01-05 10:00:01.000,invalid_price,0.2000,true\n')  # Invalid price
        f.write('2025-01-05 10:00:02.000,3001.00,invalid_quantity,false\n')  # Invalid quantity
        f.write('2025-01-05 10:00:03.000,3002.00,0.3000,invalid_boolean\n')  # Invalid boolean
        f.write('invalid_timestamp,3003.00,0.4000,true\n')  # Invalid timestamp
        f.write('2025-01-05 10:00:05.000,3004.00,0.5000,true\n')  # Valid
        f.write('')  # Empty line
        f.write('2025-01-05 10:00:06.000,3005.00,0.6000,false\n')  # Valid
    
    # Format 6: Empty file
    file6_path = os.path.join(input_dir, 'empty_file.csv')
    with open(file6_path, 'w', newline='', encoding='utf-8') as f:
        f.write('timestamp,price,quantity,is_buyer_maker\n')
        # Intentionally empty
    
    # Format 7: Large file for performance testing
    file7_path = os.path.join(input_dir, 'aggtrades_large_2025-01-06.csv')
    with open(file7_path, 'w', newline='', encoding='utf-8') as f:
        f.write('timestamp,price,quantity,is_buyer_maker,trade_id\n')
        
        current_time = start_time + timedelta(days=6)
        current_price = base_price * 1.05
        
        # Generate 10000 records for performance testing
        for i in range(10000):
            if i % 1000 == 0:  # Progress indicator
                print(f"Generating large file: {i}/10000 records")
                
            price_change = random.uniform(-0.002, 0.002) * current_price
            current_price = max(current_price + price_change, 1.0)
            
            volume = random.uniform(0.001, 0.5)
            current_time += timedelta(milliseconds=random.randint(10, 100))
            
            is_buyer_maker = random.choice([True, False])
            trade_id = 4000000 + i
            
            f.write(f'{current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]},{current_price:.2f},{volume:.4f},{is_buyer_maker},{trade_id}\n')
    
    # Create metadata file
    metadata_path = os.path.join(input_dir, 'dummy_files_metadata.json')
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'total_files': 7,
        'files': {
            'format1': {
                'path': 'aggtrades_format1_2025-01-01.csv',
                'format': 'semicolon_separated',
                'records': 1000,
                'description': 'Standard semicolon-separated format'
            },
            'format2': {
                'path': 'aggtrades_format2_2025-01-02.csv',
                'format': 'comma_separated',
                'records': 1000,
                'description': 'Comma-separated with trade IDs'
            },
            'format3': {
                'path': 'aggtrades_format3_2025-01-03.tsv',
                'format': 'tab_separated',
                'records': 500,
                'description': 'Tab-separated format'
            },
            'format4': {
                'path': 'aggtrades_format4_2025-01-04.jsonl',
                'format': 'json_lines',
                'records': 300,
                'description': 'JSON Lines format'
            },
            'corrupted': {
                'path': 'aggtrades_corrupted_2025-01-05.csv',
                'format': 'comma_separated',
                'records': 4,
                'description': 'Corrupted data for error handling tests'
            },
            'empty': {
                'path': 'empty_file.csv',
                'format': 'comma_separated',
                'records': 0,
                'description': 'Empty file for edge case testing'
            },
            'large': {
                'path': 'aggtrades_large_2025-01-06.csv',
                'format': 'comma_separated',
                'records': 10000,
                'description': 'Large file for performance testing'
            }
        },
        'data_characteristics': {
            'base_price': base_price,
            'price_range': f'{base_price * 0.95:.2f} - {base_price * 1.05:.2f}',
            'volume_range': '0.001 - 10.0',
            'time_range': f'{start_time.isoformat()} - {(start_time + timedelta(days=6)).isoformat()}',
            'total_records': 12500
        }
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created {len(metadata['files'])} dummy files in {input_dir}")
    print(f"📊 Total records generated: {metadata['data_characteristics']['total_records']}")
    print(f"📁 Metadata saved to: {metadata_path}")

class CSVNormalizer:
    """Class to handle normalization of CSV files with different formats."""
    @log_important_calls

    def __init__(self, input_directory: str, output_directory: str, write_header: bool = True) -> None:
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.write_header = write_header
        self.target_header = ['timestamp', 'price', 'quantity', 'is_buyer_maker', 'trade_id']
        self.processors = {'format1': self._process_format1_file, 'format2': self._process_format2_file}

    def normalize_trade_csvs(self) -> None:
        """Main entry point - processes all CSV files in the input directory."""
        self._setup_output_directory()
        files_to_process = self._get_csv_files()
        if not files_to_process:
            return
        for filename in files_to_process:
            self._process_single_file(filename)
    @log_all_calls

    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
    @log_all_calls

    def _get_csv_files(self) -> list[str]:
        """Get list of CSV files to process."""
        try:
            return [f for f in os.listdir(self.input_directory) if f.endswith('.csv')]
        except FileNotFoundError:
            return []
    @log_all_calls

    def _process_single_file(self, filename: str) -> None:
        """Process a single CSV file."""
        input_path = os.path.join(self.input_directory, filename)
        output_path = os.path.join(self.output_directory, f'formatted_{filename}')
        try:
            with open(input_path, encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                if self.write_header:
                    writer.writerow(self.target_header)
                format_type = self._detect_file_format(infile)
                if format_type in self.processors:
                    self.processors[format_type](infile, writer)
        except Exception:
            pass
    @log_all_calls

    def _detect_file_format(self, infile: Any) -> str:
        """Detect the format of the CSV file."""
        try:
            header_line = next(infile).strip()
            if ';' in header_line and 'agg_trade_id' not in header_line:
                return 'format1'
            if 'agg_trade_id' in header_line:
                return 'format2'
            return 'unknown'
            return 'unknown'
        except StopIteration:
            return 'empty'
    @log_all_calls

    def _process_format1_file(self, infile: Any, writer: Any) -> None:
        """Process format 1 (semicolon-delimited without trade_id)."""
        for line in infile:
            line = line.strip()
            if not line or line.startswith('timestamp'):
                continue
            row = next(csv.reader([line], delimiter=';'))
            while len(row) < 4:
                row.append('')
            row.append('')
            writer.writerow(row)
    @log_all_calls

    def _process_format2_file(self, infile: Any, writer: Any) -> None:
        """Process format 2 (mixed delimiters with agg_trade_id)."""
        for line in infile:
            line = line.strip()
            if not line or line.startswith('timestamp'):
                continue
            try:
                ts_part, rest_of_line = line.split(',', 1)
                timestamp = ts_part.replace(';', ' ')
                other_cols = next(csv.reader([rest_of_line]))
                price = other_cols[0]
                quantity = other_cols[1]
                is_buyer_maker = other_cols[2]
                trade_id = other_cols[3] if len(other_cols) > 3 else ''
                writer.writerow([timestamp, price, quantity, is_buyer_maker, trade_id])
            except (ValueError, IndexError):
                continue
if __name__ == '__main__':
    auto_reformat_aggtrades_files()
