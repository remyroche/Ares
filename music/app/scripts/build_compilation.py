import argparse
import sys
from app.db import SessionLocal
from app.services.compilation_service import CompilationService
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Trigger weekly compilation build manually"
    )
    parser.add_argument(
        "--date", required=False, help="Reference date (YYYY-MM-DD), defaults to today"
    )

    args = parser.parse_args()

    ref_date = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else datetime.now().date()
    )

    db = SessionLocal()
    try:
        service = CompilationService(db)
        comp = service.create_weekly_compilation(ref_date)
        if comp:
            print(f"Compilation {comp.id} built successfully.")
        else:
            print("Not enough tracks to build a compilation.")
    except Exception as e:
        print(f"Failed to build compilation: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
