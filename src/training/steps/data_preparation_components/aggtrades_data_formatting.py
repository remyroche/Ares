# aggtrades_data_formatting.py

import csv
import glob
import os
import shutil


def check_file_format(file_path) -> bool | None:
    """Check if a CSV file follows the correct format.
    Returns True if the file is correctly formatted, False otherwise.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            # Read the first line to check the header
            first_line = f.readline().strip()

            # Check if header has the correct format
            expected_header = "timestamp,price,quantity,is_buyer_maker,agg_trade_id"
            if first_line != expected_header:
                return False

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
                if not timestamp:
                    return False

                # Check if price and quantity are numeric
                try:
                    float(fields[1])  # price
                    float(fields[2])  # quantity
                except ValueError:
                    return False

            return True
    except Exception:
        return False


def detect_file_format(file_path) -> str | None:
    """Detect the format of a CSV file and return the format type.
    Returns: 'correct', 'format1', 'format2', 'format3', or 'unknown'.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            first_line = f.readline().strip()

        # Check for correct format
        if first_line == "timestamp,price,quantity,is_buyer_maker,agg_trade_id":
            return "correct"
            return "correct"

        # Check for format1 (semicolon-delimited)
        if ";" in first_line and "agg_trade_id" not in first_line:
            return "format1"
            return "format1"

        # Check for format2 (mixed-delimiter with agg_trade_id)
        if "agg_trade_id" in first_line and ";" in first_line:
            return "format2"

        # Check for format3 (missing agg_trade_id column)
        if first_line == "timestamp,price,quantity,is_buyer_maker":
            return "format3"
            return "format3"

    except Exception:
        return "unknown"


class DataFileReformatter:
    """Class to handle reformatting of data files with different formats."""

    def __init__(self, input_path: str, output_path: str) -> None:
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
            return False

        try:
            with (
                open(self.input_path, encoding="utf-8") as infile,
                open(
                    self.output_path,
                    "w",
                    newline="",
                    encoding="utf-8"
                ) as outfile,
            ):
                writer = csv.writer(outfile)
                return processor(infile, writer)
        except Exception:
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
                    agg_trade_id = f"agg_{timestamp}_{price}_{quantity}"

                    writer.writerow(
                        [timestamp, price, quantity, is_buyer_maker, agg_trade_id],
                    )

            return True
        except Exception:
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

                # Handle mixed delimiter format: timestamp contains a semicolon
                # Replace semicolon in the timestamp with a space, parse the rest as CSV
                if "," in line:
                    ts_part, rest = line.split(",", 1)
                else:
                    # Fallback: treat entire line as ts_part
                    ts_part, rest = line, ""

                timestamp = ts_part.replace(";", " ")
                other_cols = next(csv.reader([rest])) if rest else []

                price = other_cols[0] if len(other_cols) > 0 else ""
                quantity = other_cols[1] if len(other_cols) > 1 else ""
                is_buyer_maker = other_cols[2] if len(other_cols) > 2 else ""
                agg_trade_id = other_cols[3] if len(other_cols) > 3 else f"agg_{timestamp}_{price}_{quantity}"

                writer.writerow([timestamp, price, quantity, is_buyer_maker, agg_trade_id])


            return True
        except Exception:
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
                    agg_trade_id = f"agg_{timestamp}_{price}_{quantity}"

                    writer.writerow(
                        [timestamp, price, quantity, is_buyer_maker, agg_trade_id],
                    )

            return True
        except Exception:
            return False


def auto_reformat_aggtrades_files() -> None:
    """Automatically detect and reformat all aggtrades CSV files that don't follow the correct format."""
    # Define paths
    data_cache_dir = "data_cache"
    backup_dir = "data_cache/backup_before_reformat"

    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)

    # Find all aggtrades files for any exchange and symbol
    pattern = os.path.join(data_cache_dir, "aggtrades_*_*.csv")
    files = glob.glob(pattern)

    files_to_reformat = []
    files_checked = 0

    for file_path in files:
        files_checked += 1

        # Check if file is correctly formatted
        if not check_file_format(file_path):
            format_type = detect_file_format(file_path)
            if format_type != "correct":
                files_to_reformat.append((file_path, format_type))

    if not files_to_reformat:
        return

    # Ask for confirmation
    response = input("\nDo you want to proceed with reformatting? (y/N): ")
    if response.lower() != "y":
        return

    # Reformat files
    for file_path, format_type in files_to_reformat:
        # Create backup
        backup_path = os.path.join(backup_dir, os.path.basename(file_path))
        shutil.copy2(file_path, backup_path)

        # Create temporary output file
        temp_output = file_path + ".tmp"

        # Reformat the file
        reformatter = DataFileReformatter(file_path, temp_output)
        if reformatter.reformat_file(format_type):
            # Replace original with reformatted version
            shutil.move(temp_output, file_path)
        else:
            # Restore from backup if reformatting failed
            shutil.copy2(backup_path, file_path)






class CSVNormalizer:
    """Class to handle normalization of CSV files with different formats."""

    def __init__(
        self, input_directory: str, output_directory: str, write_header: bool = True
    ) -> None:
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

    def _process_single_file(self, filename: str) -> None:
        """Process a single CSV file."""
        input_path = os.path.join(self.input_directory, filename)
        output_path = os.path.join(self.output_directory, f"formatted_{filename}")

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

        except Exception:
            # Swallow errors for robustness in batch runs
            pass

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

            return "unknown"
        except StopIteration:
            return "empty"

    def _process_format1_file(self, infile, writer) -> None:
        """Process format 1 (semicolon-delimited without trade_id)."""
        for line in infile:
            line = line.strip()
            if not line or line.startswith("timestamp"):
                continue
            # Parse the row using the correct delimiter
            row = next(csv.reader([line], delimiter=";"))
            # Ensure 4 columns exist
            while len(row) < 4:
                row.append("")

            # Add a blank value for the missing 'trade_id' column
            row.append("")
            writer.writerow(row)

    def _process_format2_file(self, infile, writer) -> None:
        """Process format 2 (mixed delimiters with agg_trade_id)."""
        for line in infile:
            line = line.strip()
            if not line or line.startswith("timestamp"):
                continue

            try:
                # The timestamp part is everything before the first comma
                ts_part, rest_of_line = line.split(",", 1)

                # The timestamp itself contains a semicolon that needs to be replaced
                timestamp = ts_part.replace(";", " ")

                # The rest of the line is a standard comma-separated string
                other_cols = next(csv.reader([rest_of_line]))

                price = other_cols[0]
                quantity = other_cols[1]
                is_buyer_maker = other_cols[2]
                trade_id = other_cols[3] if len(other_cols) > 3 else ""
                writer.writerow([timestamp, price, quantity, is_buyer_maker, trade_id])
            except (ValueError, IndexError):
                continue


if __name__ == "__main__":
    # Run the automatic reformatting
    auto_reformat_aggtrades_files()