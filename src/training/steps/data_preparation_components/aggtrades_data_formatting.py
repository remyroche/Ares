# aggtrades_data_formatting.py

import csv
import glob
import os
import shutil


def check_file_format(file_path):
    """
    Check if a CSV file follows the correct format.
    Returns True if the file is correctly formatted, False otherwise.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
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
                    fields = line.split(",")
                    if len(fields) != 5:
                        return False

                    # Check if timestamp field is properly formatted
                    timestamp = fields[0]
                    if not timestamp or timestamp == "":
                        return False

                    # Check if price and quantity are numeric
                    try:
                        float(fields[1])  # price
                        float(fields[2])  # quantity
                    except ValueError:
                        return False

                return True
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
        with open(file_path, encoding="utf-8") as f:
            first_line = f.readline().strip()

            # Check for correct format
            if first_line == "timestamp,price,quantity,is_buyer_maker,agg_trade_id":
                return "correct"

            # Check for format1 (semicolon-delimited)
            if ";" in first_line and "agg_trade_id" not in first_line:
                return "format1"

            # Check for format2 (mixed-delimiter with agg_trade_id)
            if "agg_trade_id" in first_line:
                return "format2"

            # Check for format3 (missing agg_trade_id column)
            if first_line == "timestamp,price,quantity,is_buyer_maker":
                return "format3"

            return "unknown"
    except Exception as e:
        print(f"Error detecting format for {file_path}: {e}")
        return "unknown"


class DataFileReformatter:
    """Class to handle reformatting of data files with different formats."""

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.processors = {
            "format1": self._process_format1,
            "format2": self._process_format2,
            "format3": self._process_format3,
        }

    def reformat_file(self, format_type: str) -> bool:
        """Main entry point - delegates to specific processor."""
        processor = self.processors.get(format_type)
        if not processor:
            print(f"Unknown format type: {format_type}")
            return False

        try:
            with open(self.input_path, encoding="utf-8") as infile:
                with open(
                    self.output_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as outfile:
                    writer = csv.writer(outfile)
                    return processor(infile, writer)
        except Exception as e:
            print(f"Error reformatting file {self.input_path}: {e}")
            return False

    def _process_format1(self, infile, writer) -> bool:
        """Process semicolon-delimited format."""
        try:
            # Write header
            writer.writerow(
                ["timestamp", "price", "quantity", "is_buyer_maker", "agg_trade_id"],
            )

            # Process data lines
            for line in infile:
                line = line.strip()
                if not line or line.startswith("timestamp"):
                    continue

                # Split by semicolon
                fields = line.split(";")
                if len(fields) >= 4:
                    timestamp = fields[0]
                    price = fields[1]
                    quantity = fields[2]
                    is_buyer_maker = fields[3]
                    agg_trade_id = (
                        f"agg_{timestamp}_{price}_{quantity}"  # Generate dummy ID
                    )

                    writer.writerow(
                        [timestamp, price, quantity, is_buyer_maker, agg_trade_id],
                    )

            return True
        except Exception as e:
            print(f"Error processing format1: {e}")
            return False

    def _process_format2(self, infile, writer) -> bool:
        """Process mixed-delimiter format with agg_trade_id."""
        try:
            # Write header
            writer.writerow(
                ["timestamp", "price", "quantity", "is_buyer_maker", "agg_trade_id"],
            )

            # Process data lines
            for line in infile:
                line = line.strip()
                if not line or line.startswith("timestamp"):
                    continue

                # Handle mixed delimiter format
                if "," in line and ";" in line:
                    # Split by comma first, then handle semicolon-separated parts
                    parts = line.split(",")
                    if len(parts) >= 4:
                        timestamp = parts[0]
                        price = parts[1]
                        quantity = parts[2]
                        is_buyer_maker = parts[3]
                        agg_trade_id = (
                            parts[4]
                            if len(parts) > 4
                            else f"agg_{timestamp}_{price}_{quantity}"
                        )

                        writer.writerow(
                            [timestamp, price, quantity, is_buyer_maker, agg_trade_id],
                        )

            return True
        except Exception as e:
            print(f"Error processing format2: {e}")
            return False

    def _process_format3(self, infile, writer) -> bool:
        """Process format missing agg_trade_id column."""
        try:
            # Write header
            writer.writerow(
                ["timestamp", "price", "quantity", "is_buyer_maker", "agg_trade_id"],
            )

            # Process data lines
            for line in infile:
                line = line.strip()
                if not line or line.startswith("timestamp"):
                    continue

                # Split by comma
                fields = line.split(",")
                if len(fields) >= 4:
                    timestamp = fields[0]
                    price = fields[1]
                    quantity = fields[2]
                    is_buyer_maker = fields[3]
                    agg_trade_id = (
                        f"agg_{timestamp}_{price}_{quantity}"  # Generate dummy ID
                    )

                    writer.writerow(
                        [timestamp, price, quantity, is_buyer_maker, agg_trade_id],
                    )

            return True
        except Exception as e:
            print(f"Error processing format3: {e}")
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

    # Find all aggtrades files for any exchange and symbol
    pattern = os.path.join(data_cache_dir, "aggtrades_*_*.csv")
    files = glob.glob(pattern)

    print(f"Found {len(files)} aggtrades files to check...")

    files_to_reformat = []
    files_checked = 0

    for file_path in files:
        files_checked += 1
        print(
            f"Checking file {files_checked}/{len(files)}: {os.path.basename(file_path)}",
        )

        # Check if file is correctly formatted
        if not check_file_format(file_path):
            format_type = detect_file_format(file_path)
            if format_type != "correct":
                files_to_reformat.append((file_path, format_type))
                print(f"  -> Needs reformatting (detected format: {format_type})")
            else:
                print("  -> File appears to be correctly formatted")
        else:
            print("  -> File is correctly formatted")

    if not files_to_reformat:
        print("\nAll files are already in the correct format!")
        return

    print(f"\nFound {len(files_to_reformat)} files that need reformatting:")
    for file_path, format_type in files_to_reformat:
        print(f"  - {os.path.basename(file_path)} ({format_type})")

    # Ask for confirmation
    response = input("\nDo you want to proceed with reformatting? (y/N): ")
    if response.lower() != "y":
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
        reformatter = DataFileReformatter(file_path, temp_output)
        if reformatter.reformat_file(format_type):
            # Replace original with reformatted version
            shutil.move(temp_output, file_path)
            print(f"  -> Successfully reformatted: {os.path.basename(file_path)}")
        else:
            # Restore from backup if reformatting failed
            shutil.copy2(backup_path, file_path)
            print(
                f"  -> Failed to reformat, restored from backup: {os.path.basename(file_path)}",
            )

    print(f"\nReformatting complete! Backup files are in: {backup_dir}")


def auto_reformat_aggtrades_files_for_exchange(exchange: str, symbol: str):
    """
    Automatically detect and reformat aggtrades CSV files for a specific exchange and symbol.
    This is a targeted version that only processes files for the specified exchange/symbol.
    """
    # Define paths
    data_cache_dir = "data_cache"
    backup_dir = "data_cache/backup_before_reformat"

    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)

    # Find aggtrades files for the specific exchange and symbol
    pattern = os.path.join(data_cache_dir, f"aggtrades_{exchange}_{symbol}_*.csv")
    files = glob.glob(pattern)

    print(f"Found {len(files)} aggtrades files for {exchange}_{symbol} to check...")

    files_to_reformat = []
    files_checked = 0

    for file_path in files:
        files_checked += 1
        print(
            f"Checking file {files_checked}/{len(files)}: {os.path.basename(file_path)}",
        )

        # Check if file is correctly formatted
        if not check_file_format(file_path):
            format_type = detect_file_format(file_path)
            if format_type != "correct":
                files_to_reformat.append((file_path, format_type))
                print(f"  -> Needs reformatting (detected format: {format_type})")
            else:
                print("  -> File appears to be correctly formatted")
        else:
            print("  -> File is correctly formatted")

    if not files_to_reformat:
        print(f"\nAll {exchange}_{symbol} files are already in the correct format!")
        return

    print(f"\nFound {len(files_to_reformat)} files that need reformatting:")
    for file_path, format_type in files_to_reformat:
        print(f"  - {os.path.basename(file_path)} ({format_type})")

    # Reformat files without asking for confirmation (for automated use)
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
        reformatter = DataFileReformatter(file_path, temp_output)
        if reformatter.reformat_file(format_type):
            # Replace original with reformatted version
            shutil.move(temp_output, file_path)
            print(f"  -> Successfully reformatted: {os.path.basename(file_path)}")
        else:
            # Restore from backup if reformatting failed
            shutil.copy2(backup_path, file_path)
            print(
                f"  -> Failed to reformat, restored from backup: {os.path.basename(file_path)}",
            )

    print(
        f"\nReformatting complete for {exchange}_{symbol}! Backup files are in: {backup_dir}",
    )


def create_dummy_files(input_dir):
    """
    Creates a set of dummy CSV files for demonstration purposes.
    This function simulates the two different formats you provided.
    """
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)

    # --- Create File 1: Semicolon-delimited format ---
    file1_path = os.path.join(input_dir, "aggtrades_format1_2025-07-13.csv")
    with open(file1_path, "w", newline="", encoding="utf-8") as f:
        f.write("timestamp;price;quantity;is_buyer_maker\n")
        f.write("2025-07-12 22:00:00.604;2939.2;0.3152;False\n")
        f.write("2025-07-12 22:00:00.614;2939.21;0.1917;False\n")
        f.write("2025-07-12 22:00:00.614;2939.22;0.1702;False\n")
    print(f"Created dummy file: {file1_path}")

    # --- Create File 2: Mixed-delimiter format ---
    file2_path = os.path.join(input_dir, "aggtrades_format2_2025-07-30.csv")
    with open(file2_path, "w", newline="", encoding="utf-8") as f:
        # Note the malformed "p;rice" in the header, as in your example
        f.write("timestamp,p;rice,quantity,is_buyer_maker,agg_trade_id\n")
        f.write("2025-07-30;00:00:02.623,3791.56,0.065,False,2338842426\n")
        f.write("2025-07-30;00:00:04.240,3791.55,0.022,True,2338842427\n")
        f.write("2025-07-30;00:00:04.865,3791.55,0.018,True,2338842428\n")
    print(f"Created dummy file: {file2_path}")

    # --- Create an empty file to test edge cases ---
    file3_path = os.path.join(input_dir, "empty_file.csv")
    open(file3_path, "w").close()
    print(f"Created dummy file: {file3_path}")


class CSVNormalizer:
    """Class to handle normalization of CSV files with different formats."""

    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        write_header: bool = True,
    ):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.write_header = write_header
        self.target_header = [
            "timestamp",
            "price",
            "quantity",
            "is_buyer_maker",
            "trade_id",
        ]
        self.processors = {
            "format1": self._process_format1_file,
            "format2": self._process_format2_file,
        }

    def normalize_trade_csvs(self) -> None:
        """Main entry point - processes all CSV files in the input directory."""
        self._setup_output_directory()
        files_to_process = self._get_csv_files()

        if not files_to_process:
            print("No CSV files found to process.")
            return

        print(f"\nFound {len(files_to_process)} CSV files to process.")

        for filename in files_to_process:
            self._process_single_file(filename)

        print("\nNormalization complete.")

    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            print(f"Created output directory: {self.output_directory}")

    def _get_csv_files(self) -> list[str]:
        """Get list of CSV files to process."""
        try:
            return [f for f in os.listdir(self.input_directory) if f.endswith(".csv")]
        except FileNotFoundError:
            print(f"Error: Input directory not found at '{self.input_directory}'")
            return []

    def _process_single_file(self, filename: str) -> None:
        """Process a single CSV file."""
        input_path = os.path.join(self.input_directory, filename)
        output_path = os.path.join(self.output_directory, f"formatted_{filename}")

        print(f"Processing '{filename}'...")

        try:
            with (
                open(input_path, encoding="utf-8") as infile,
                open(output_path, "w", newline="", encoding="utf-8") as outfile,
            ):
                writer = csv.writer(outfile)

                # Write header if requested
                if self.write_header:
                    writer.writerow(self.target_header)

                # Detect and process format
                format_type = self._detect_file_format(infile)
                if format_type in self.processors:
                    self.processors[format_type](infile, writer)
                else:
                    print(
                        f"  - Warning: Could not determine format for '{filename}'. Skipping file.",
                    )

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

    def _detect_file_format(self, infile) -> str:
        """Detect the format of the CSV file."""
        try:
            header_line = next(infile).strip()

            # Format 1: semicolon-delimited without trade_id
            if ";" in header_line and "agg_trade_id" not in header_line:
                return "format1"

            # Format 2: mixed delimiters with agg_trade_id
            if "agg_trade_id" in header_line:
                return "format2"

            return "unknown"

        except StopIteration:
            print("  - Warning: File is empty. Skipping.")
            return "empty"

    def _process_format1_file(self, infile, writer) -> None:
        """Process format 1 (semicolon-delimited without trade_id)."""
        print("  - Detected Format 1 (semicolon-delimited).")
        for line in infile:
            line = line.strip()
            if not line:
                continue
            # Parse the row using the correct delimiter
            row = next(csv.reader([line], delimiter=";"))
            # Add a blank value for the missing 'trade_id' column
            row.append("")
            writer.writerow(row)

    def _process_format2_file(self, infile, writer) -> None:
        """Process format 2 (mixed delimiters with agg_trade_id)."""
        print("  - Detected Format 2 (mixed-delimiter).")
        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                # The timestamp part is everything before the first comma
                ts_part, rest_of_line = line.split(",", 1)

                # The timestamp itself contains a semicolon that needs to be replaced
                timestamp = ts_part.replace(";", " ")

                # The rest of the line is a standard comma-separated string
                other_cols = next(csv.reader([rest_of_line]))

                price, quantity, is_buyer_maker, trade_id = other_cols
                writer.writerow([timestamp, price, quantity, is_buyer_maker, trade_id])
            except (ValueError, IndexError):
                print(f"  - Warning: Skipping malformed line: {line}")
                continue


if __name__ == "__main__":
    # Run the automatic reformatting
    print("=== Aggtrades CSV File Auto-Reformatter ===")
    print("This script will automatically detect and reformat aggtrades CSV files")
    print("that don't follow the correct format.\n")

    auto_reformat_aggtrades_files()
