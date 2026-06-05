"""
src/features/bid_stack_features.py — Phase 1 improvements
===========================================================
Phase 1 additions vs original:
  1. bs_very_high_supply_mw   — sell supply in top-2 IEX bands (>=10,001)
  2. bs_supply_exhaustion      — log1p(buy/cheap+1) clipped [0,10], safe from overflow
  3. bs_mid_supply_mw          — supply in Rs 5,001-8,000 bands (buffer zone)
  4. bs_stress_regime          — binary: buy/sell>1.2 AND cheap_share<0.20

All original 8 features preserved identically.
All features lagged 1h (4 blocks) as before.
"""

import pandas as pd
import numpy as np

BPH             = 4
SPIKE_THRESHOLD = 10_000   # consistent with price_features.py


def build_bid_stack_features(bid_stack_df, market):
    df = bid_stack_df.sort_values(['delivery_start_ist', 'price_band_rs_mwh'])

    grouped    = df.groupby('delivery_start_ist')
    total_buy  = grouped['buy_demand_mw'].sum()
    total_sell = grouped['sell_supply_mw'].sum()

    feats = pd.DataFrame(index=total_buy.index)
    feats['bs_total_buy_mw']  = total_buy
    feats['bs_total_sell_mw'] = total_sell

    feats['bs_buy_sell_ratio'] = (
        feats['bs_total_buy_mw'] / feats['bs_total_sell_mw']
    ).replace([np.inf, -np.inf], np.nan)

    # ── Original band-based features ───────────────────────────────────────
    high_bands  = ['8001-9000', '9001-10000', '10001-11000', '11001-12000']
    high_supply = (
        df[df['price_band_rs_mwh'].isin(high_bands)]
        .groupby('delivery_start_ist')['sell_supply_mw'].sum()
    )
    feats['bs_supply_margin_mw'] = high_supply.reindex(feats.index, fill_value=0)

    cheap_bands  = ['0-1000', '1001-2000', '2001-3000']
    cheap_supply = (
        df[df['price_band_rs_mwh'].isin(cheap_bands)]
        .groupby('delivery_start_ist')['sell_supply_mw'].sum()
    )
    feats['bs_cheap_supply_mw']    = cheap_supply.reindex(feats.index, fill_value=0)
    feats['bs_cheap_supply_share'] = (
        feats['bs_cheap_supply_mw'] / feats['bs_total_sell_mw']
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    totals = (
        df
        .merge(total_buy.rename('total_buy'),  on='delivery_start_ist')
        .merge(total_sell.rename('total_sell'), on='delivery_start_ist')
    )
    totals['buy_share_sq']  = (
        totals['buy_demand_mw']  / totals['total_buy'].replace(0, np.nan)
    ) ** 2
    totals['sell_share_sq'] = (
        totals['sell_supply_mw'] / totals['total_sell'].replace(0, np.nan)
    ) ** 2

    feats['bs_buy_hhi']  = (
        totals.groupby('delivery_start_ist')['buy_share_sq'].sum()
        .reindex(feats.index, fill_value=0)
    )
    feats['bs_sell_hhi'] = (
        totals.groupby('delivery_start_ist')['sell_share_sq'].sum()
        .reindex(feats.index, fill_value=0)
    )

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1 NEW — Spike-oriented supply-stack features
    # ══════════════════════════════════════════════════════════════════════

    # 1. Very-high-price supply: top-2 IEX bands only (>=10,001 Rs/MWh)
    very_high_bands  = ['10001-11000', '11001-12000']
    very_high_supply = (
        df[df['price_band_rs_mwh'].isin(very_high_bands)]
        .groupby('delivery_start_ist')['sell_supply_mw'].sum()
    )
    feats['bs_very_high_supply_mw'] = very_high_supply.reindex(feats.index, fill_value=0)

    # 2. Supply exhaustion — log1p transform prevents overflow; clipped [0,10]
    #    log1p(12000/1) ~ 9.4 which is the physical ceiling
    raw_exhaustion = feats['bs_total_buy_mw'] / (feats['bs_cheap_supply_mw'] + 1)
    feats['bs_supply_exhaustion'] = np.log1p(raw_exhaustion).clip(0, 10)

    # 3. Mid-stack supply (Rs 5,001 - 8,000) — buffer zone between cheap and premium
    mid_bands  = ['5001-6000', '6001-7000', '7001-8000']
    mid_supply = (
        df[df['price_band_rs_mwh'].isin(mid_bands)]
        .groupby('delivery_start_ist')['sell_supply_mw'].sum()
    )
    feats['bs_mid_supply_mw'] = mid_supply.reindex(feats.index, fill_value=0)

    # 4. Stress regime binary: demand pressure AND supply thinness simultaneously
    stress_demand = feats['bs_buy_sell_ratio'] > 1.2
    stress_supply = feats['bs_cheap_supply_share'] < 0.20
    feats['bs_stress_regime'] = (stress_demand & stress_supply).astype(int)

    # ── Lag ALL features by 1h (4 blocks) ─────────────────────────────────
    lagged = feats.shift(1 * BPH).add_suffix('_lag_1h')

    return lagged
