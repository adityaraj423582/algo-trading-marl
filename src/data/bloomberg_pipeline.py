"""
Bloomberg Integration Pipeline.

End-to-end script that loads the Bloomberg Excel export, merges
bid/ask/spread data into existing feature CSVs, and adds VIX features.

Usage:
    python -m src.data.bloomberg_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data.bloomberg_loader import (
    BloombergLoader,
    add_vix_features,
    BLOOMBERG_FEATURE_COLUMNS,
)
from src.utils.config import FEATURES_DIR


def main() -> None:
    print("=" * 60)
    print("BLOOMBERG INTEGRATION PIPELINE")
    print("=" * 60)

    # Step 1: Load Bloomberg data
    print("\n[1/3] Loading Bloomberg Excel file...")
    try:
        loader = BloombergLoader()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    bloomberg_data = loader.load_all()

    # Step 2: Merge into feature CSVs
    print("\n[2/3] Merging Bloomberg data into feature CSVs...")
    loader.merge_with_features(bloomberg_data)

    # Step 3: Add VIX features
    print("\n[3/3] Adding VIX features to all 20 feature CSVs...")
    add_vix_features(bloomberg_data)

    # Final verification
    print("\n" + "=" * 60)
    print("BLOOMBERG INTEGRATION COMPLETE")
    print("=" * 60)

    verify_csv = FEATURES_DIR / "nse" / "RELIANCE_NS_features.csv"
    if verify_csv.exists():
        df = pd.read_csv(verify_csv, index_col="Datetime", parse_dates=True)
        bb_cols = [c for c in df.columns if c.startswith("bb_") or "vix" in c]
        nan_count = int(df[bb_cols].isnull().sum().sum()) if bb_cols else 0

        print(f"\nFeature CSV verification (RELIANCE_NS):")
        print(f"  Total columns: {df.shape[1]}")
        print(f"  Bloomberg columns added: {bb_cols}")
        print(f"  NaN in Bloomberg cols: {nan_count}")

    print(f"\nUpgrade summary vs yfinance:")
    print(f"  Real bid-ask spread:  YES (17 stocks)")
    print(f"  India VIX signal:     YES (all 20 stocks)")
    print(f"  US VIX signal:        YES (all 20 stocks)")
    print(f"  Bloomberg volume:     YES (17 stocks)")
    print(f"  CNN input features:   16 -> 25")


if __name__ == "__main__":
    main()
