#!/usr/bin/env python3
"""
Simple runner for DHIS2 automated pipeline
Place in same folder as dhis2_fetcher.py and config.py
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add the parent directory to path to import from config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run full DHIS2 automated pipeline.")
    parser.add_argument(
        "--facilities",
        nargs="?",
        default=None,
        help="Optional facility selection: 'all', comma-separated names/UIDs, or omit value to prompt.",
    )
    parser.add_argument(
        "--merge-mode",
        choices=["replace", "new_only"],
        default="replace",
        help="For facility mode: replace updates existing TEIs; new_only appends only new TEIs.",
    )
    args = parser.parse_args()

    # Load environment variables from the project .env for local/CLI runs.
    # (utils/config.py intentionally does not call load_dotenv.)
    load_dotenv(REPO_ROOT / ".env", override=True)

    print("=" * 70)
    print("🤖 DHIS2 AUTOMATED PIPELINE")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print()

    try:
        # Import from the same directory
        from dhis2_fetcher import run_automated_pipeline

        print("✅ Import successful")
        print("🚀 Starting pipeline...")
        print()

        facility_selection = args.facilities
        if "--facilities" in sys.argv and args.facilities is None:
            facility_selection = ""

        # Run the pipeline
        success = run_automated_pipeline(
            facility_selection=facility_selection,
            merge_mode=args.merge_mode,
        )

        print()
        print("=" * 70)

        if success:
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            print("❌ PIPELINE FAILED")

        print(f"End time: {datetime.now()}")
        print("=" * 70)
        return success

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
