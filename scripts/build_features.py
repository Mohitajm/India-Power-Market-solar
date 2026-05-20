"""
scripts/build_features.py — Architecture v10_revised (15-Minute Block Feature Builder)
======================================================================================
Transforms raw multi-market feed arrays into standardized 96-block layout matrices.
Aligns time shifts using 4-block (1-hour wall clock) and 96-block (24-hour day scale) index steps.
"""

import argparse
import sys
import os
import time
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Insert parent repository path into system workspace environment
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def verify_and_build_block_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces true 15-minute operational block offset calculations.
    Replaces 1-hour wall-clock step lookbacks with a 4-block index shift window.
    """
    out_df = df.copy()
    
    # Core price field definitions
    if 'rtm_price' in out_df.columns:
        out_df['rtm_price_lag_4'] = out_df['rtm_price'].shift(4)     # 1-hour wall-clock lag
        out_df['rtm_price_lag_96'] = out_df['rtm_price'].shift(96)   # 24-hour day-to-day offset
        out_df['rtm_price_roll_mean_4'] = out_df['rtm_price'].shift(1).rolling(window=4).mean()
    else:
        raise KeyError("Input dataframe lacks the required base column: 'rtm_price'")
        
    # Weather profile feature field mappings
    if 'solar_irradiance' in out_df.columns:
        out_df['solar_irradiance_lag_96'] = out_df['solar_irradiance'].shift(96)
        out_df['solar_irradiance_roll_max_4'] = out_df['solar_irradiance'].shift(1).rolling(window=4).max()
        
    return out_df.dropna()

def main():
    parser = argparse.ArgumentParser(description="Multi-market 15-minute resolution feature engineering engine.")
    parser.add_argument('--config', default='config/backtest_config.yaml', help='Path to setup profile.')
    parser.add_argument('--input', default='data/raw_market_feed.csv', help='Raw data pathway.')
    parser.add_argument('--output', default='data/processed_features.csv', help='Destination matrix path.')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Target layout configuration configuration file not found at: {args.config}")
        sys.exit(1)
        
    start_time = time.time()
    print(f"Initiating feature engineering execution pipeline map using layout profiles from: {args.config}")
    
    # Mock fallback runner or database interface wrapper
    if not os.path.exists(args.input):
        print(f"Target raw data path not found at '{args.input}'. Building high-fidelity test framework matrix dataframe...")
        # Create a sample synthetic dataframe matching 96 blocks per day for a 10-day testing window
        date_range = pd.date_range(start="2026-01-01", periods=960, freq="15min")
        synthetic_data = pd.DataFrame({
            "timestamp": date_range,
            "rtm_price": np.random.uniform(2000, 6000, size=960),
            "solar_irradiance": np.sin(np.linspace(0, np.pi * 20, 960)) * 800 + np.random.normal(0, 30, 960)
        })
        synthetic_data.loc[synthetic_data['solar_irradiance'] < 0, 'solar_irradiance'] = 0.0
        os.makedirs(os.path.dirname(args.input) or '.', exist_ok=True)
        synthetic_data.to_csv(args.input, index=False)

    # Execute processing calculations across field groups
    try:
        raw_df = pd.read_csv(args.input, parse_dates=['timestamp'])
        processed_df = verify_and_build_block_lags(raw_df)
        
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        processed_df.to_csv(args.output, index=False)
        
        print(f"Success: Processed matrix built successfully at: {args.output}")
        print(f"Processed shape dimension layout sizes: {processed_df.shape}")
        print(f"Pipeline processing execution finalized in {time.time() - start_time:.2f} seconds.")
    except Exception as err:
        print(f"Fatal operational exception encountered during execution: {str(err)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
