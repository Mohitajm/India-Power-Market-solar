"""
scripts/build_features.py — Architecture v10_revised (15-Minute Block Feature Builder)
======================================================================================
Orchestrates the feature engineering pipeline for DAM and RTM markets.
All 15-minute resolution logic (BPH=4, 96 blocks/day) is handled by the
modules in src/features/:
  - price_features.py   (lag shifts ×4, rolling windows ×4)
  - bid_stack_features.py (groupby delivery_start_ist, lag shift(4))
  - grid_features.py    (hourly→96-block expansion in loader, shift(4))
  - weather_features.py (auto-detect BPH, shift accordingly)
  - calendar_features.py (timestamp-based, resolution-agnostic)
  - pipeline.py         (orchestrator: complete-day=96, target_block 1-96,
                          SNAPSHOT_BLOCK=33, mcp_same_block_yesterday)

This script is a thin entry point. All feature engineering logic lives
in src/features/pipeline.py::build_all_features().
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.pipeline import build_all_features


def main():
    parser = argparse.ArgumentParser(
        description="Build 96-block feature matrices for DAM and RTM markets.")
    parser.add_argument(
        "--config", default="config/backtest_config.yaml",
        help="Path to backtest configuration YAML.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 60)
    print("FEATURE ENGINEERING — Architecture v10_revised (96 blocks)")
    print("=" * 60)
    print(f"Config: {config_path}")

    build_all_features(str(config_path))

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()
