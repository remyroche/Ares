from pathlib import Path
from collections import defaultdict

def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

def main():
    root = Path(__file__).resolve().parent

    files = []
    folder_sizes = defaultdict(int)

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".py":
            continue

        try:
            size = path.stat().st_size
        except OSError:
            continue

        files.append((path, size))

        # Add this file's size to every parent folder up to root
        current = path.parent
        while True:
            folder_sizes[current] += size
            if current == root:
                break
            current = current.parent

    # Sort largest first
    files.sort(key=lambda x: x[1], reverse=True)
    folders = sorted(folder_sizes.items(), key=lambda x: x[1], reverse=True)

    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("FILES BY SIZE")
    report_lines.append("=" * 80)
    for path, size in files:
        if size >= 1024 * 1024:
            rel_path = path.relative_to(root)
            report_lines.append(f"{format_size(size):>10}  {rel_path}")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("FOLDERS BY TOTAL CONTENT SIZE")
    report_lines.append("=" * 80)
    for folder, size in folders:
        rel_folder = "." if folder == root else folder.relative_to(root)
        report_lines.append(f"{format_size(size):>10}  {rel_folder}")

    report_text = "\n".join(report_lines)

    print(report_text)

    # Save to memory_cleaner.txt
    report_path = root / "memory_cleaner.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nReport generated and saved to {report_path.relative_to(root)}")

if __name__ == "__main__":
    main()
