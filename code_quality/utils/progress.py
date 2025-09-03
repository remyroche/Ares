"""
Rich progress tracking for code quality tools.
"""

import time
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


class ProgressTracker:
    """Rich progress tracking for code quality operations."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.progress = None
        self.current_task = None
        self.start_time = None

    def __enter__(self):
        """Context manager entry."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.progress:
            self.progress.stop()

    def add_task(self, description: str, total: int = 0) -> int:
        """Add a new task to track."""
        if self.progress:
            return self.progress.add_task(description, total=total)
        return 0

    def update(self, task_id: int, advance: int = 1, description: str = None):
        """Update task progress."""
        if self.progress:
            if description:
                self.progress.update(task_id, description=description)
            self.progress.advance(task_id, advance)

    def complete_task(self, task_id: int):
        """Mark a task as complete."""
        if self.progress:
            self.progress.update(task_id, completed=self.progress.tasks[task_id].total)


class CodeQualityProgress:
    """Specialized progress tracking for code quality operations."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.start_time = None

    def start_operation(self, operation_name: str):
        """Start a new operation."""
        self.start_time = time.time()
        self.console.print(f"\n[bold blue]🚀 Starting {operation_name}...[/bold blue]")

    def end_operation(self, operation_name: str, success: bool = True):
        """End an operation."""
        if self.start_time:
            duration = time.time() - self.start_time
            status = "✅ Completed" if success else "❌ Failed"
            self.console.print(f"\n[bold green]{status} {operation_name} in {duration:.2f}s[/bold green]")

    def track_file_processing(self, files: list[str], operation: str = "Processing"):
        """Track progress of file processing operations."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:

            task = progress.add_task(f"{operation} files", total=len(files))

            for i, file_path in enumerate(files):
                file_name = Path(file_path).name
                progress.update(task, description=f"{operation} {file_name}")
                yield i, file_path
                progress.advance(task)

    def track_tool_execution(self, tools: list[str], operation: str = "Running"):
        """Track progress of tool execution."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:

            task = progress.add_task(f"{operation} tools", total=len(tools))

            for i, tool in enumerate(tools):
                progress.update(task, description=f"{operation} {tool}")
                yield i, tool
                progress.advance(task)

    def show_summary(self, results: dict[str, Any], operation: str = "Operation"):
        """Show a summary of operation results."""
        table = Table(title=f"{operation} Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Add summary rows
        for key, value in results.items():
            if isinstance(value, int | float | str | bool):
                table.add_row(key.replace("_", " ").title(), str(value))
            elif isinstance(value, list):
                table.add_row(key.replace("_", " ").title(), f"{len(value)} items")
            elif isinstance(value, dict):
                table.add_row(key.replace("_", " ").title(), f"{len(value)} entries")

        self.console.print(table)

    def show_file_results(self, file_results: list[dict[str, Any]], operation: str = "Results"):
        """Show detailed results for individual files."""
        if not file_results:
            self.console.print(f"[yellow]No {operation.lower()} to display[/yellow]")
            return

        table = Table(title=f"{operation} by File", box=box.ROUNDED)
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")

        for result in file_results:
            file_path = Path(result.get("file", "Unknown")).name
            success = result.get("success", False)
            status = "✅ Success" if success else "❌ Failed"
            details = result.get("message", "No details")

            table.add_row(file_path, status, details)

        self.console.print(table)

    def show_tool_results(self, tool_results: dict[str, Any], operation: str = "Tool Results"):
        """Show results organized by tool."""
        table = Table(title=f"{operation}", box=box.ROUNDED)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Files Processed", style="magenta")
        table.add_column("Success Rate", style="blue")

        for tool_name, results in tool_results.items():
            if isinstance(results, dict):
                files_processed = results.get("files_processed", 0)
                successful = results.get("successful", 0)
                success_rate = f"{(successful/files_processed)*100:.1f}%" if files_processed > 0 else "N/A"
                status = "✅ Active" if results.get("enabled", True) else "❌ Disabled"

                table.add_row(tool_name, status, str(files_processed), success_rate)

        self.console.print(table)


class LiveProgressDisplay:
    """Live updating progress display for long-running operations."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.layout = Layout()
        self.current_status = {}

    def setup_layout(self):
        """Setup the layout for live display."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        self.layout["header"].update(Panel("Code Quality Tools", style="bold blue"))
        self.layout["footer"].update(Panel("Press Ctrl+C to stop", style="dim"))

    def update_status(self, operation: str, current_file: str, progress: float, details: str = ""):
        """Update the live status display."""
        self.current_status = {
            "operation": operation,
            "current_file": current_file,
            "progress": progress,
            "details": details,
        }

        # Update main content
        main_content = Group(
            f"[bold]Operation:[/bold] {operation}",
            f"[bold]Current File:[/bold] {Path(current_file).name}",
            f"[bold]Progress:[/bold] {progress:.1%}",
            f"[bold]Details:[/bold] {details}" if details else "",
        )

        self.layout["main"].update(Panel(main_content, title="Status"))

    def show_live(self, operation_func, *args, **kwargs):
        """Show live progress for an operation."""
        self.setup_layout()

        with Live(self.layout, refresh_per_second=4, console=self.console):
            try:
                result = operation_func(*args, **kwargs)
                self.layout["main"].update(Panel("✅ Operation completed successfully!", style="bold green"))
                return result
            except KeyboardInterrupt:
                self.layout["main"].update(Panel("⚠️ Operation interrupted by user", style="bold yellow"))
                raise
            except Exception as e:
                self.layout["main"].update(Panel(f"❌ Operation failed: {str(e)}", style="bold red"))
                raise


class ProgressManager:
    """High-level progress management for code quality operations."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.progress = CodeQualityProgress(console)
        self.live_display = LiveProgressDisplay(console)

    def run_with_progress(self, operation_name: str, operation_func, *args, **kwargs):
        """Run an operation with full progress tracking."""
        try:
            self.progress.start_operation(operation_name)

            # Run the operation
            result = operation_func(*args, **kwargs)

            self.progress.end_operation(operation_name, success=True)
            return result

        except Exception:
            self.progress.end_operation(operation_name, success=False)
            raise

    def track_file_operation(self, files: list[str], operation_name: str, operation_func):
        """Track a file-based operation with progress."""
        def tracked_operation():
            results = []
            for _i, file_path in self.progress.track_file_processing(files, operation_name):
                try:
                    result = operation_func(file_path)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "file": file_path,
                        "success": False,
                        "error": str(e),
                    })
            return results

        return self.run_with_progress(operation_name, tracked_operation)

    def track_tool_operation(self, tools: list[str], operation_name: str, operation_func):
        """Track a tool-based operation with progress."""
        def tracked_operation():
            results = {}
            for _i, tool in self.progress.track_tool_execution(tools, operation_name):
                try:
                    result = operation_func(tool)
                    results[tool] = result
                except Exception as e:
                    results[tool] = {
                        "success": False,
                        "error": str(e),
                    }
            return results

        return self.run_with_progress(operation_name, tracked_operation)
