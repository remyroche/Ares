#!/usr/bin/env python3
"""
Script to rename all files in sr_levels/ directories to include timestamps down to the minute.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_sr_levels_directories():
    """Find all sr_levels directories."""
    sr_levels_dirs = []

    # Check common locations
    base_paths = [
        "/Users/remyroche/Documents/Ares/historical_data",
        "/Users/remyroche/Documents/Ares/src"
    ]

    for base_path in base_paths:
        for root, dirs, files in os.walk(base_path):
            if "sr_levels" in dirs:
                sr_levels_path = Path(root) / "sr_levels"
                if sr_levels_path.exists():
                    sr_levels_dirs.append(sr_levels_path)

    return sr_levels_dirs

def rename_files_with_timestamp(sr_levels_dir: Path):
    """Rename all files in sr_levels directory to include timestamp."""
    logger.info(f"Processing directory: {sr_levels_dir}")

    files_renamed = 0
    errors = 0

    # Get all files in the directory
    files = list(sr_levels_dir.glob("*"))
    files = [f for f in files if f.is_file()]

    if not files:
        logger.info(f"No files found in {sr_levels_dir}")
        return 0, 0

    # Get current timestamp down to the minute
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Create backup directory
    backup_dir = sr_levels_dir / f"backup_{timestamp}"
    backup_dir.mkdir(exist_ok=True)
    logger.info(f"Created backup directory: {backup_dir}")

    for file_path in files:
        try:
            # Get file extension
            file_extension = file_path.suffix

            # Get base filename without extension
            base_name = file_path.stem

            # Create new filename with timestamp
            if base_name.endswith(f"_{timestamp}"):
                # File already has current timestamp
                logger.info(f"File already has current timestamp: {file_path.name}")
                continue

            new_base_name = f"{base_name}_{timestamp}"
            new_filename = f"{new_base_name}{file_extension}"
            new_file_path = sr_levels_dir / new_filename

            # Create backup
            backup_path = backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)

            # Rename file
            file_path.rename(new_file_path)

            logger.info(f"Renamed: {file_path.name} -> {new_filename}")
            files_renamed += 1

        except Exception as e:
            logger.error(f"Error renaming {file_path}: {e}")
            errors += 1

    # Create a summary file
    summary_file = sr_levels_dir / f"rename_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"File rename operation completed at {datetime.now().isoformat()}\n")
        f.write(f"Directory: {sr_levels_dir}\n")
        f.write(f"Files renamed: {files_renamed}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"Backup directory: {backup_dir}\n")

    logger.info(f"Summary written to: {summary_file}")
    return files_renamed, errors

def main():
    """Main function to rename all sr_levels files."""
    logger.info("Starting SR levels file renaming operation")

    # Find all sr_levels directories
    sr_levels_dirs = get_sr_levels_directories()

    if not sr_levels_dirs:
        logger.warning("No sr_levels directories found")
        return

    logger.info(f"Found {len(sr_levels_dirs)} sr_levels directories:")
    for sr_dir in sr_levels_dirs:
        logger.info(f"  - {sr_dir}")

    total_files_renamed = 0
    total_errors = 0

    # Process each sr_levels directory
    for sr_levels_dir in sr_levels_dirs:
        try:
            files_renamed, errors = rename_files_with_timestamp(sr_levels_dir)
            total_files_renamed += files_renamed
            total_errors += errors
        except Exception as e:
            logger.error(f"Error processing {sr_levels_dir}: {e}")
            total_errors += 1

    logger.info("=" * 60)
    logger.info("OPERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total directories processed: {len(sr_levels_dirs)}")
    logger.info(f"Total files renamed: {total_files_renamed}")
    logger.info(f"Total errors: {total_errors}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
