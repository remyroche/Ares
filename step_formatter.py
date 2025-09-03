#!/usr/bin/env python3
"""
Step Formatter Script

This script automatically detects mentions of step01, step02... up to step09 in both
file contents and file names, and adds a leading zero to make them step01, step02... etc.

Usage:
    python step_formatter.py [--dry-run] [--backup] [--recursive] [path]

Options:
    --dry-run     Show what would be changed without making changes
    --backup      Create backup files before making changes
    --recursive   Process subdirectories recursively
    path          Directory or file to process (default: current directory)
"""

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class StepFormatter:
    def __init__(self, dry_run: bool = False, backup: bool = False):
        self.dry_run = dry_run
        self.backup = backup
        self.changes_made = 0
        self.files_processed = 0

        # Regex pattern to match step01, step02, ..., step09
        # Matches: step01, step02, step03, step04, step05, step06, step07, step08, step09
        # Does NOT match: step0, step10, step11, step12, etc.
        self.step_pattern = re.compile(r"\bstep([1-9])\b")

        # File extensions to process for content changes
        self.text_extensions = {
            ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
            ".ini", ".cfg", ".log", ".rst", ".csv", ".xml", ".html",
            ".js", ".ts", ".jsx", ".tsx", ".css", ".scss", ".sql",
            ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
        }

    def format_step_content(self, content: str) -> tuple[str, int]:
        """
        Format step mentions in content by adding leading zeros.

        Args:
            content: The text content to process

        Returns:
            Tuple of (formatted_content, number_of_changes)
        """
        changes = 0

        def replace_step(match):
            nonlocal changes
            step_num = match.group(1)
            changes += 1
            return f"step0{step_num}"

        formatted_content = self.step_pattern.sub(replace_step, content)
        return formatted_content, changes

    def format_filename(self, filename: str) -> tuple[str, int]:
        """
        Format step mentions in filename by adding leading zeros.

        Args:
            filename: The filename to process

        Returns:
            Tuple of (formatted_filename, number_of_changes)
        """
        changes = 0

        def replace_step(match):
            nonlocal changes
            step_num = match.group(1)
            changes += 1
            return f"step0{step_num}"

        formatted_filename = self.step_pattern.sub(replace_step, filename)
        return formatted_filename, changes

    def should_process_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be processed for content changes.

        Args:
            file_path: Path to the file

        Returns:
            True if file should be processed
        """
        # Skip hidden files and directories
        if file_path.name.startswith("."):
            return False

        # Skip binary files and non-text files
        if file_path.suffix.lower() in self.text_extensions:
            return True

        # Skip directories
        if file_path.is_dir():
            return False

        # Skip very large files (>10MB) to avoid memory issues
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                logger.warning(f"Skipping large file: {file_path}")
                return False
        except OSError:
            return False

        return False

    def process_file_content(self, file_path: Path) -> int:
        """
        Process a single file's content for step mentions.

        Args:
            file_path: Path to the file to process

        Returns:
            Number of changes made
        """
        if not self.should_process_file(file_path):
            return 0

        try:
            # Read file content
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Check if content contains step mentions
            if not self.step_pattern.search(content):
                return 0

            # Format the content
            formatted_content, changes = self.format_step_content(content)

            if changes > 0:
                if self.backup:
                    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"Created backup: {backup_path}")

                if not self.dry_run:
                    # Write formatted content back to file
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(formatted_content)
                    logger.info(f"Updated {file_path}: {changes} step mentions formatted")
                else:
                    logger.info(f"Would update {file_path}: {changes} step mentions would be formatted")

                return changes

        except Exception as e:
            logger.exception(f"Error processing file {file_path}: {e}")

        return 0

    def process_filename(self, file_path: Path) -> int:
        """
        Process a filename for step mentions.

        Args:
            file_path: Path to the file

        Returns:
            Number of changes made
        """
        filename = file_path.name

        # Check if filename contains step mentions
        if not self.step_pattern.search(filename):
            return 0

        # Format the filename
        formatted_filename, changes = self.format_filename(filename)

        if changes > 0:
            new_path = file_path.parent / formatted_filename

            if self.backup:
                backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                shutil.copy2(file_path, backup_path)
                logger.info(f"Created backup: {backup_path}")

            if not self.dry_run:
                try:
                    file_path.rename(new_path)
                    logger.info(f"Renamed: {file_path.name} -> {formatted_filename}")
                    return changes
                except Exception as e:
                    logger.exception(f"Error renaming file {file_path}: {e}")
                    return 0
            else:
                logger.info(f"Would rename: {file_path.name} -> {formatted_filename}")
                return changes

        return 0

    def process_directory(self, directory: Path, recursive: bool = False) -> dict[str, int]:
        """
        Process a directory for step mentions in both file contents and names.

        Args:
            directory: Directory to process
            recursive: Whether to process subdirectories

        Returns:
            Dictionary with statistics about the operation
        """
        stats = {
            "files_processed": 0,
            "content_changes": 0,
            "filename_changes": 0,
            "total_changes": 0,
        }

        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return stats

        if not directory.is_dir():
            logger.error(f"Path is not a directory: {directory}")
            return stats

        # Get all files to process
        files = list(directory.rglob("*")) if recursive else list(directory.iterdir())

        # Process each file
        for file_path in files:
            if file_path.is_file():
                stats["files_processed"] += 1

                # Process file content
                content_changes = self.process_file_content(file_path)
                stats["content_changes"] += content_changes

                # Process filename
                filename_changes = self.process_filename(file_path)
                stats["filename_changes"] += filename_changes

                stats["total_changes"] += content_changes + filename_changes

        return stats

    def process_path(self, path: str, recursive: bool = False) -> dict[str, int]:
        """
        Process a file or directory for step mentions.

        Args:
            path: Path to file or directory
            recursive: Whether to process subdirectories (only applies to directories)

        Returns:
            Dictionary with statistics about the operation
        """
        path_obj = Path(path)

        if path_obj.is_file():
            # Process single file
            stats = {
                "files_processed": 1,
                "content_changes": 0,
                "filename_changes": 0,
                "total_changes": 0,
            }

            # Process file content
            content_changes = self.process_file_content(path_obj)
            stats["content_changes"] += content_changes

            # Process filename
            filename_changes = self.process_filename(path_obj)
            stats["filename_changes"] += filename_changes

            stats["total_changes"] += content_changes + filename_changes

            return stats

        if path_obj.is_dir():
            # Process directory
            return self.process_directory(path_obj, recursive)

        logger.error(f"Path does not exist: {path}")
        return {
            "files_processed": 0,
            "content_changes": 0,
            "filename_changes": 0,
            "total_changes": 0,
        }

def main():
    parser = argparse.ArgumentParser(
        description="Format step mentions by adding leading zeros (step01 -> step01)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup files before making changes",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process subdirectories recursively",
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory or file to process (default: current directory)",
    )

    args = parser.parse_args()

    # Create formatter
    formatter = StepFormatter(dry_run=args.dry_run, backup=args.backup)

    # Display mode
    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info(f"Starting step formatter in {mode} mode")
    if args.backup:
        logger.info("Backup mode enabled - backup files will be created")

    # Process the specified path
    try:
        stats = formatter.process_path(args.path, args.recursive)

        # Display results
        logger.info("=" * 50)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Files processed: {stats['files_processed']}")
        logger.info(f"Content changes: {stats['content_changes']}")
        logger.info(f"Filename changes: {stats['filename_changes']}")
        logger.info(f"Total changes: {stats['total_changes']}")

        if args.dry_run and stats["total_changes"] > 0:
            logger.info("\nThis was a dry run. Remove --dry-run flag to apply changes.")
        elif stats["total_changes"] == 0:
            logger.info("\nNo step mentions found to format.")

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
