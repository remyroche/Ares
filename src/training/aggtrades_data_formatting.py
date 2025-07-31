# aggtrades_data_formatting.py

import os
import csv
import shutil
import glob
from pathlib import Path

def check_file_format(file_path):
    """
    Check if a CSV file follows the correct format.
    Returns True if the file is correctly formatted, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read the first line to check the header
            first_line = f.readline().strip()
            
            # Check if header has the correct format
            expected_header = "timestamp,price,quantity,is_buyer_maker,agg_trade_id"
            if first_line == expected_header:
                # Check a few data lines to ensure they're properly formatted
                for i, line in enumerate(f):
                    if i >= 5:  # Check first 5 data lines
                        break
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if line has correct number of fields
                    fields = line.split(',')
                    if len(fields) != 5:
                        return False
                    
                    # Check if timestamp field is properly formatted
                    timestamp = fields[0]
                    if not timestamp or timestamp == '':
                        return False
                    
                    # Check if price and quantity are numeric
                    try:
                        float(fields[1])  # price
                        float(fields[2])  # quantity
                    except ValueError:
                        return False
                
                return True
            else:
                return False
    except Exception as e:
        print(f"Error checking file {file_path}: {e}")
        return False

def detect_file_format(file_path):
    """
    Detect the format of a CSV file and return the format type.
    Returns: 'correct', 'format1', 'format2', 'format3', or 'unknown'
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            
            # Check for correct format
            if first_line == "timestamp,price,quantity,is_buyer_maker,agg_trade_id":
                return 'correct'
            
            # Check for format1 (semicolon-delimited)
            if ';' in first_line and 'agg_trade_id' not in first_line:
                return 'format1'
            
            # Check for format2 (mixed-delimiter with agg_trade_id)
            if 'agg_trade_id' in first_line:
                return 'format2'
            
            # Check for format3 (missing agg_trade_id column)
            if first_line == "timestamp,price,quantity,is_buyer_maker":
                return 'format3'
            
            return 'unknown'
    except Exception as e:
        print(f"Error detecting format for {file_path}: {e}")
        return 'unknown'

def reformat_file(input_path, output_path, format_type):
    """
    Reformat a CSV file to the correct format.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            
            writer = csv.writer(outfile)
            
            # Write the correct header
            writer.writerow(['timestamp', 'price', 'quantity', 'is_buyer_maker', 'agg_trade_id'])
            
            if format_type == 'format1':
                # Process semicolon-delimited format
                for line in infile:
                    line = line.strip()
                    if not line or line.startswith('timestamp;'):
                        continue
                    
                    try:
                        row = next(csv.reader([line], delimiter=';'))
                        if len(row) >= 4:
                            timestamp, price, quantity, is_buyer_maker = row[:4]
                            # Add empty agg_trade_id
                            writer.writerow([timestamp, price, quantity, is_buyer_maker, ''])
                    except Exception as e:
                        print(f"Warning: Skipping malformed line in {input_path}: {line}")
                        continue
            
            elif format_type == 'format2':
                # Process mixed-delimiter format
                for line in infile:
                    line = line.strip()
                    if not line or line.startswith('timestamp'):
                        continue
                    
                    try:
                        # Split on first comma to separate timestamp from rest
                        ts_part, rest_of_line = line.split(',', 1)
                        
                        # Fix timestamp (replace semicolon with space)
                        timestamp = ts_part.replace(';', ' ')
                        
                        # Parse the rest of the line
                        other_cols = next(csv.reader([rest_of_line]))
                        
                        if len(other_cols) >= 4:
                            price, quantity, is_buyer_maker, trade_id = other_cols[:4]
                            writer.writerow([timestamp, price, quantity, is_buyer_maker, trade_id])
                    except Exception as e:
                        print(f"Warning: Skipping malformed line in {input_path}: {line}")
                        continue
            
            elif format_type == 'format3':
                # Process format with missing agg_trade_id column
                for line in infile:
                    line = line.strip()
                    if not line or line.startswith('timestamp,price,quantity,is_buyer_maker'):
                        continue
                    
                    try:
                        # Parse the line (should have 4 fields)
                        fields = line.split(',')
                        if len(fields) == 4:
                            timestamp, price, quantity, is_buyer_maker = fields
                            # Add empty agg_trade_id
                            writer.writerow([timestamp, price, quantity, is_buyer_maker, ''])
                        else:
                            print(f"Warning: Skipping malformed line in {input_path}: {line}")
                            continue
                    except Exception as e:
                        print(f"Warning: Skipping malformed line in {input_path}: {line}")
                        continue
            
            else:
                print(f"Unknown format for {input_path}, skipping...")
                return False
        
        return True
    except Exception as e:
        print(f"Error reformatting {input_path}: {e}")
        return False

def auto_reformat_aggtrades_files():
    """
    Automatically detect and reformat all aggtrades CSV files that don't follow the correct format.
    """
    # Define paths
    data_cache_dir = "data_cache"
    backup_dir = "data_cache/backup_before_reformat"
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Find all aggtrades files
    pattern = os.path.join(data_cache_dir, "aggtrades_BINANCE_ETHUSDT_*.csv")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} aggtrades files to check...")
    
    files_to_reformat = []
    files_checked = 0
    
    for file_path in files:
        files_checked += 1
        print(f"Checking file {files_checked}/{len(files)}: {os.path.basename(file_path)}")
        
        # Check if file is correctly formatted
        if not check_file_format(file_path):
            format_type = detect_file_format(file_path)
            if format_type != 'correct':
                files_to_reformat.append((file_path, format_type))
                print(f"  -> Needs reformatting (detected format: {format_type})")
            else:
                print(f"  -> File appears to be correctly formatted")
        else:
            print(f"  -> File is correctly formatted")
    
    if not files_to_reformat:
        print("\nAll files are already in the correct format!")
        return
    
    print(f"\nFound {len(files_to_reformat)} files that need reformatting:")
    for file_path, format_type in files_to_reformat:
        print(f"  - {os.path.basename(file_path)} ({format_type})")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with reformatting? (y/N): ")
    if response.lower() != 'y':
        print("Reformatting cancelled.")
        return
    
    # Reformat files
    print("\nStarting reformatting process...")
    
    for file_path, format_type in files_to_reformat:
        print(f"\nReformatting: {os.path.basename(file_path)}")
        
        # Create backup
        backup_path = os.path.join(backup_dir, os.path.basename(file_path))
        shutil.copy2(file_path, backup_path)
        print(f"  -> Created backup: {backup_path}")
        
        # Create temporary output file
        temp_output = file_path + ".tmp"
        
        # Reformat the file
        if reformat_file(file_path, temp_output, format_type):
            # Replace original with reformatted version
            shutil.move(temp_output, file_path)
            print(f"  -> Successfully reformatted: {os.path.basename(file_path)}")
        else:
            # Restore from backup if reformatting failed
            shutil.copy2(backup_path, file_path)
            print(f"  -> Failed to reformat, restored from backup: {os.path.basename(file_path)}")
    
    print(f"\nReformatting complete! Backup files are in: {backup_dir}")

def create_dummy_files(input_dir):
    """
    Creates a set of dummy CSV files for demonstration purposes.
    This function simulates the two different formats you provided.
    """
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)

    # --- Create File 1: Semicolon-delimited format ---
    file1_path = os.path.join(input_dir, 'aggtrades_format1_2025-07-13.csv')
    with open(file1_path, 'w', newline='', encoding='utf-8') as f:
        f.write("timestamp;price;quantity;is_buyer_maker\n")
        f.write("2025-07-12 22:00:00.604;2939.2;0.3152;False\n")
        f.write("2025-07-12 22:00:00.614;2939.21;0.1917;False\n")
        f.write("2025-07-12 22:00:00.614;2939.22;0.1702;False\n")
    print(f"Created dummy file: {file1_path}")

    # --- Create File 2: Mixed-delimiter format ---
    file2_path = os.path.join(input_dir, 'aggtrades_format2_2025-07-30.csv')
    with open(file2_path, 'w', newline='', encoding='utf-8') as f:
        # Note the malformed "p;rice" in the header, as in your example
        f.write("timestamp,p;rice,quantity,is_buyer_maker,agg_trade_id\n")
        f.write("2025-07-30;00:00:02.623,3791.56,0.065,False,2338842426\n")
        f.write("2025-07-30;00:00:04.240,3791.55,0.022,True,2338842427\n")
        f.write("2025-07-30;00:00:04.865,3791.55,0.018,True,2338842428\n")
    print(f"Created dummy file: {file2_path}")
    
    # --- Create an empty file to test edge cases ---
    file3_path = os.path.join(input_dir, 'empty_file.csv')
    open(file3_path, 'w').close()
    print(f"Created dummy file: {file3_path}")


def normalize_trade_csvs(input_directory, output_directory, write_header=True):
    """
    Processes all CSV files in an input directory, normalizes them to a
    consistent format, and saves them to an output directory.

    Args:
        input_directory (str): The path to the directory containing the source CSV files.
        output_directory (str): The path where the formatted CSVs will be saved.
        write_header (bool): If True, a header row will be written to the output files.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Find all CSV files in the input directory
    try:
        files_to_process = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{input_directory}'")
        return

    print(f"\nFound {len(files_to_process)} CSV files to process.")

    # Define the standard header for the output files
    target_header = ['timestamp', 'price', 'quantity', 'is_buyer_maker', 'trade_id']

    for filename in files_to_process:
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f"formatted_{filename}")
        
        print(f"Processing '{filename}'...")

        try:
            with open(input_path, 'r', encoding='utf-8') as infile, \
                 open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                
                writer = csv.writer(outfile)

                # Write the standardized header to the new file if requested
                if write_header:
                    writer.writerow(target_header)

                # Read the first line of the input file to determine its format
                try:
                    header_line = next(infile).strip()
                except StopIteration:
                    print(f"  - Warning: File '{filename}' is empty. Skipping.")
                    continue

                # --- FORMAT 1 DETECTION & PROCESSING ---
                # Detects format with semicolon delimiter and no trade_id.
                # e.g., "timestamp;price;quantity;is_buyer_maker"
                if ';' in header_line and 'agg_trade_id' not in header_line:
                    print(f"  - Detected Format 1 (semicolon-delimited).")
                    # Process the rest of the file
                    for line in infile:
                        line = line.strip()
                        if not line: continue
                        # Parse the row using the correct delimiter
                        row = next(csv.reader([line], delimiter=';'))
                        # row = [timestamp, price, quantity, is_buyer_maker]
                        # Add a blank value for the missing 'trade_id' column
                        row.append('') 
                        writer.writerow(row)

                # --- FORMAT 2 DETECTION & PROCESSING ---
                # Detects format with mixed delimiters and an agg_trade_id.
                # e.g., "timestamp,p;rice,quantity,is_buyer_maker,agg_trade_id"
                elif 'agg_trade_id' in header_line:
                    print(f"  - Detected Format 2 (mixed-delimiter).")
                    # Process the rest of the file
                    for line in infile:
                        line = line.strip()
                        if not line: continue
                        
                        try:
                            # The timestamp part is everything before the first comma
                            ts_part, rest_of_line = line.split(',', 1)
                            
                            # The timestamp itself contains a semicolon that needs to be replaced
                            timestamp = ts_part.replace(';', ' ')
                            
                            # The rest of the line is a standard comma-separated string
                            other_cols = next(csv.reader([rest_of_line]))
                            
                            price, quantity, is_buyer_maker, trade_id = other_cols
                            writer.writerow([timestamp, price, quantity, is_buyer_maker, trade_id])
                        except (ValueError, IndexError):
                            print(f"  - Warning: Skipping malformed line in '{filename}': {line}")
                            continue
                
                else:
                    print(f"  - Warning: Could not determine format for '{filename}'. Skipping file.")

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

    print("\nNormalization complete.")


if __name__ == '__main__':
    # Run the automatic reformatting
    print("=== Aggtrades CSV File Auto-Reformatter ===")
    print("This script will automatically detect and reformat aggtrades CSV files")
    print("that don't follow the correct format.\n")
    
    auto_reformat_aggtrades_files()
