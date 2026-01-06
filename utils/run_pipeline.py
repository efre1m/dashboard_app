#!/usr/bin/env python3
"""
Simple runner for DHIS2 automated pipeline
Place in same folder as dhis2_fetcher.py and config.py
"""

import os
import sys
from datetime import datetime

# Add the parent directory to path to import from config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
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

        # Run the pipeline
        success = run_automated_pipeline()

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
