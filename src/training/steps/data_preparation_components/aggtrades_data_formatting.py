from __future__ import annotations
import csv
import glob
import os
import os.path
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

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
    os.makedirs(backup_dir, exist_ok=True)
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
    response = input('\nDo you want to proceed with reformatting? (y/N): ')
    if response.lower() != 'y':
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

def auto_reformat_aggtrades_files_for_exchange(exchange: str, symbol: str) -> None:
    """Automatically detect and reformat aggtrades CSV files for a specific exchange and symbol.
    This is a targeted version that only processes files for the specified exchange/symbol.
    """
    data_cache_dir = 'data_cache'
    backup_dir = 'data_cache/backup_before_reformat'
    os.makedirs(backup_dir, exist_ok=True)
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
    """Creates a set of dummy CSV files for demonstration purposes.
    This function simulates the two different formats you provided.
    """
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)
    file1_path = os.path.join(input_dir, 'aggtrades_format1_2025-07-13.csv')
    with open(file1_path, 'w', newline='', encoding='utf-8') as f:
        f.write('timestamp;price;quantity;is_buyer_maker\n')
        f.write('2025-07-12 22:00:00.604;2939.2;0.3152;False\n')
        f.write('2025-07-12 22:00:00.614;2939.21;0.1917;False\n')
        f.write('2025-07-12 22:00:00.614;2939.22;0.1702;False\n')
    file2_path = os.path.join(input_dir, 'aggtrades_format2_2025-07-30.csv')
    with open(file2_path, 'w', newline='', encoding='utf-8') as f:
        f.write('timestamp,p;rice,quantity,is_buyer_maker,agg_trade_id\n')
        f.write('2025-07-30;00:00:02.623,3791.56,0.065,False,2338842426\n')
        f.write('2025-07-30;00:00:04.240,3791.55,0.022,True,2338842427\n')
        f.write('2025-07-30;00:00:04.865,3791.55,0.018,True,2338842428\n')
    file3_path = os.path.join(input_dir, 'empty_file.csv')
    open(file3_path, 'w').close()

class CSVNormalizer:
    """Class to handle normalization of CSV files with different formats."""

    def __init__(self, input_directory: str, output_directory: str, write_header: bool=True) -> None:
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

    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def _get_csv_files(self) -> list[str]:
        """Get list of CSV files to process."""
        try:
            return [f for f in os.listdir(self.input_directory) if f.endswith('.csv')]
        except FileNotFoundError:
            return []

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