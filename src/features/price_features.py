"""
src/features/price_features.py — Phase 1 improvements
=======================================================
Phase 1 additions vs original:
  1. Spike features (SPIKE_THRESHOLD = 10,000 Rs/MWh — top-2 IEX bands)
  2. All spike features are lagged 1h (4 blocks) — no leakage
  3. All original features preserved identically

New features added:
  spike_flag_lag_1h          Binary: block 1h ago >= 10,000?
  spike_intensity_lag_1h     Z-score vs 7-day rolling mean, clipped [-5,5]
  consecutive_spikes_lag_1h  Consecutive spike block count, capped at 96
  spike_freq_24h_lag_1h      Fraction of last 24h in spike territory
  mcp_zscore_24h_lag_1h      Z-score vs 24h rolling mean/std, clipped [-5,5]
"""

import pandas as pd
import numpy as np

BPH             = 4        # blocks per hour (15-min resolution)
SPIKE_THRESHOLD = 10_000   # Rs/MWh — top-2 IEX price bands (10001-11000, 11001-12000)


def build_price_features(prices_df, market):
    df = prices_df.sort_values('delivery_start_ist').copy()
    df = df.set_index('delivery_start_ist')

    mcp_col  = 'mcp_rs_mwh'
    mcv_col  = 'mcv_mwh'
    buy_col  = 'purchase_bid_mwh'
    sell_col = 'sell_bid_mwh'

    features = pd.DataFrame(index=df.index)

    # ── Original lag features ──────────────────────────────────────────────
    features['mcp_lag_1h']   = df[mcp_col].shift(1 * BPH)
    features['mcp_lag_2h']   = df[mcp_col].shift(2 * BPH)
    features['mcp_lag_4h']   = df[mcp_col].shift(4 * BPH)
    features['mcp_lag_24h']  = df[mcp_col].shift(24 * BPH)
    features['mcp_lag_168h'] = df[mcp_col].shift(168 * BPH)

    # ── Original rolling statistics ────────────────────────────────────────
    w24  = 24  * BPH
    w168 = 168 * BPH

    roll_mean_24h  = df[mcp_col].rolling(window=w24,  min_periods=w24 // 2).mean()
    roll_std_24h   = df[mcp_col].rolling(window=w24,  min_periods=w24 // 2).std()
    roll_mean_168h = df[mcp_col].rolling(window=w168, min_periods=w168 // 4).mean()
    roll_std_168h  = df[mcp_col].rolling(window=w168, min_periods=w168 // 4).std()

    features['mcp_rolling_mean_24h']  = roll_mean_24h
    features['mcp_rolling_std_24h']   = roll_std_24h
    features['mcp_rolling_mean_168h'] = roll_mean_168h

    # ── Original volume features ───────────────────────────────────────────
    features['mcv_lag_1h']           = df[mcv_col].shift(1 * BPH)
    features['mcv_rolling_mean_24h'] = df[mcv_col].rolling(
        window=w24, min_periods=w24 // 2
    ).mean()

    # ── Original bid-pressure ratio ────────────────────────────────────────
    bid_ratio = df[buy_col] / df[sell_col]
    bid_ratio = bid_ratio.replace([np.inf, -np.inf], np.nan)
    features['bid_ratio_lag_1h'] = bid_ratio.shift(1 * BPH)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1 NEW — Spike features
    # All computed on raw series first, then lagged 1h to avoid leakage.
    # ══════════════════════════════════════════════════════════════════════
    mcp = df[mcp_col]

    # 1. Binary spike flag
    is_spike = (mcp >= SPIKE_THRESHOLD).astype(int)

    # 2. Spike intensity: z-score vs 7-day rolling mean/std, clipped [-5, 5]
    spike_intensity = (
        (mcp - roll_mean_168h) / roll_std_168h.replace(0, np.nan)
    ).clip(-5, 5).fillna(0)

    # 3. Consecutive spike count — capped at 96 (1 full day) to prevent extremes
    spike_groups       = (is_spike != is_spike.shift()).cumsum()
    consecutive_spikes = is_spike.groupby(spike_groups).cumcount() + 1
    consecutive_spikes = consecutive_spikes.where(is_spike == 1, 0).clip(0, 96)

    # 4. Rolling spike frequency over last 24h
    spike_freq_24h = is_spike.rolling(window=w24, min_periods=w24 // 2).mean().fillna(0)

    # 5. Z-score vs 24h rolling mean/std, clipped [-5, 5]
    mcp_zscore_24h = (
        (mcp - roll_mean_24h) / roll_std_24h.replace(0, np.nan)
    ).clip(-5, 5).fillna(0)

    # ── Lag all spike features by 1h (4 blocks) ───────────────────────────
    features['spike_flag_lag_1h']         = is_spike.shift(1 * BPH)
    features['spike_intensity_lag_1h']    = spike_intensity.shift(1 * BPH)
    features['consecutive_spikes_lag_1h'] = consecutive_spikes.shift(1 * BPH)
    features['spike_freq_24h_lag_1h']     = spike_freq_24h.shift(1 * BPH)
    features['mcp_zscore_24h_lag_1h']     = mcp_zscore_24h.shift(1 * BPH)

    return features
