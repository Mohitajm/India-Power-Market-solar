"""
src/features/grid_features.py — Phase 2 (Priority 1-5 additions)
=================================================================
Phase 2 additions vs Phase 1 (original):

PRIORITY 1 — Grid Frequency (Hz)                    ← strongest new signal
  grid_frequency_hz              raw instantaneous frequency
  grid_freq_deviation            signed deviation from 50.0 Hz
  grid_freq_deficit_flag         binary: freq < 49.90 (supply deficit / spike risk)
  grid_freq_surplus_flag         binary: freq > 50.05 (excess supply / price depression)
  grid_freq_rolling_1h           1-hour rolling mean frequency (regime signal)
  grid_freq_lag_1h               frequency 1 hour ago

PRIORITY 2 — Hydro MW separated
  grid_hydro_mw                  absolute hydro generation
  grid_hydro_share               hydro as fraction of total generation
  grid_hydro_lag_24h             hydro same hour yesterday
  grid_hydro_ramp_1h             hydro ramp over last hour (peaking signal)

PRIORITY 3 — Gas MW (marginal plant proxy)
  grid_gas_mw                    gas generation (sets marginal cost)
  grid_gas_lag_24h               gas same hour yesterday
  grid_gas_share                 gas as fraction of total generation

PRIORITY 4 — Net Transnational Exchange
  grid_net_transnational_mw      import(+) / export(-) in MW
  grid_is_net_importer           binary: India net importing (demand > domestic supply)
  grid_transnational_lag_24h     transnational exchange same hour yesterday

PRIORITY 5 — CWC Reservoir Storage % (optional weekly parquet)
  grid_reservoir_storage_pct     national weighted reservoir storage % full
  grid_reservoir_deficit         binary: storage < 40% (hydro constraint risk)
  grid_reservoir_rolling_4w      4-week rolling mean storage (seasonal trend)

PHASE 2 extended features (from grid_features.py Phase 2 plan):
  grid_net_demand_lag_168h       same hour last week
  grid_renewable_share_lag_1h    lagged RE share
  grid_demand_lag_24h            raw demand 24h ago
  grid_wind_lag_24h              wind 24h ago
  grid_solar_lag_24h             solar 24h ago
  grid_thermal_util_lag_1h       lagged thermal utilisation
  grid_net_demand_rolling_7d     7-day rolling net demand mean

All original features are preserved unchanged:
  grid_demand_mw, grid_net_demand_mw, grid_solar_mw, grid_wind_mw,
  grid_total_gen_mw, grid_fuel_mix_imputed, grid_net_demand_delta_1h,
  grid_net_demand_lag_24h, grid_solar_ramp_1h, grid_demand_gen_gap,
  grid_thermal_util, grid_renewable_share

Column-name safety: every new column is guarded with a multi-name
candidate lookup (_col) so the function works whether your parquet
uses 'frequency_hz', 'freq_hz', 'grid_frequency', etc.  Missing
columns produce NaN columns that are dropped by pipeline.py's
NaN-drop stage — they never cause a KeyError.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BPH = 4   # blocks per hour (15-min resolution)


# ── Column-name candidate lookup ───────────────────────────────────────────
# Each entry covers the most likely column names across different NLDC
# parquet builds.  Returns the first match, or a zero-filled Series if
# none is found.

def _col(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """
    Return df[first_found_candidate] or a zero Series if none found.
    Prints a one-time warning so the user knows a feature is missing.
    """
    for c in candidates:
        if c in df.columns:
            return df[c]
    print(f"  [grid_features] WARNING: none of {candidates} found in grid parquet. "
          f"Feature will be zero-filled. Add the column or check your parquet schema.")
    return pd.Series(0.0, index=df.index)


def build_grid_features(grid_df: pd.DataFrame,
                         reservoir_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build grid features from the expanded 15-min grid DataFrame.

    Parameters
    ----------
    grid_df      : Output of DataLoader._load_grid() — one row per 15-min
                   block (96/day), indexed by delivery_start_ist.
                   Must contain the standard NLDC columns; new NLDC PSP
                   columns (frequency, hydro, gas, nuclear, transnational)
                   are used if present, silently zero-filled if absent.

    reservoir_df : Optional CWC reservoir parquet (weekly frequency).
                   Expected columns: date (YYYY-MM-DD str), reservoir_pct
                   If None, Priority 5 reservoir features are skipped.

    Returns
    -------
    pd.DataFrame indexed on delivery_start_ist.
    """
    df    = grid_df.set_index('delivery_start_ist').sort_index()
    feats = pd.DataFrame(index=df.index)

    # ══════════════════════════════════════════════════════════════════════
    # ORIGINAL features — unchanged
    # ══════════════════════════════════════════════════════════════════════

    feats['grid_demand_mw']     = df['all_india_demand_mw']
    feats['grid_net_demand_mw'] = df['net_demand_mw']
    feats['grid_solar_mw']      = df['all_india_solar_mw']
    feats['grid_wind_mw']       = df['all_india_wind_mw']
    feats['grid_total_gen_mw']  = df['total_generation_mw']

    feats['grid_fuel_mix_imputed'] = (
        df['fuel_mix_imputed'].astype(int)
        if 'fuel_mix_imputed' in df.columns else 0
    )

    feats['grid_net_demand_delta_1h'] = (
        feats['grid_net_demand_mw'] - feats['grid_net_demand_mw'].shift(1 * BPH)
    )
    feats['grid_net_demand_lag_24h'] = feats['grid_net_demand_mw'].shift(24 * BPH)
    feats['grid_solar_ramp_1h']      = (
        feats['grid_solar_mw'] - feats['grid_solar_mw'].shift(1 * BPH)
    )
    feats['grid_demand_gen_gap'] = (
        feats['grid_demand_mw'] - feats['grid_total_gen_mw']
    )

    thermal_col = 'total_thermal_mw'
    thermal = (
        df[thermal_col] if thermal_col in df.columns
        else df['total_generation_mw'] - (
            df['all_india_solar_mw'] + df['all_india_wind_mw']
        )
    )
    feats['grid_thermal_util'] = thermal / 180_000.0

    feats['grid_renewable_share'] = (
        (feats['grid_solar_mw'] + feats['grid_wind_mw'])
        / feats['grid_demand_mw']
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2 extended features — supply/demand fundamentals
    # ══════════════════════════════════════════════════════════════════════

    feats['grid_net_demand_lag_168h'] = feats['grid_net_demand_mw'].shift(168 * BPH)

    feats['grid_renewable_share_lag_1h'] = feats['grid_renewable_share'].shift(1 * BPH)

    feats['grid_demand_lag_24h'] = feats['grid_demand_mw'].shift(24 * BPH)
    feats['grid_wind_lag_24h']   = feats['grid_wind_mw'].shift(24 * BPH)
    feats['grid_solar_lag_24h']  = feats['grid_solar_mw'].shift(24 * BPH)

    feats['grid_thermal_util_lag_1h'] = feats['grid_thermal_util'].shift(1 * BPH)

    feats['grid_net_demand_rolling_7d'] = (
        feats['grid_net_demand_mw']
        .rolling(window=168 * BPH, min_periods=48 * BPH)
        .mean()
    )

    # ══════════════════════════════════════════════════════════════════════
    # PRIORITY 1 — Grid Frequency (Hz)
    # Physical meaning: 50.0 Hz = perfect balance.
    #   < 49.9 Hz → deficit → scarcity → price spike risk
    #   > 50.1 Hz → surplus → excess supply → price depression
    # This is the single strongest new feature for spike prediction.
    # ══════════════════════════════════════════════════════════════════════

    freq = _col(df,
                'frequency_hz',           # most likely name
                'freq_hz',
                'grid_frequency',
                'all_india_frequency',
                'system_frequency_hz')

    # Only compute frequency features if the column was actually found
    # (zero-filled series has no predictive value for frequency features)
    has_freq = any(c in df.columns for c in
                   ['frequency_hz', 'freq_hz', 'grid_frequency',
                    'all_india_frequency', 'system_frequency_hz'])

    if has_freq:
        feats['grid_frequency_hz']    = freq
        feats['grid_freq_deviation']  = freq - 50.0          # signed: neg = deficit

        # Binary regime flags — thresholds from CERC DSM regulations
        feats['grid_freq_deficit_flag']  = (freq < 49.90).astype(int)
        feats['grid_freq_surplus_flag']  = (freq > 50.05).astype(int)

        # 1-hour rolling mean — smooths telemetry noise, captures sustained regime
        feats['grid_freq_rolling_1h'] = (
            freq.rolling(window=1 * BPH, min_periods=1).mean()
        )
        # 1-hour lag — information available before the target block
        feats['grid_freq_lag_1h'] = freq.shift(1 * BPH)

        print(f"  [grid_features] Frequency features built. "
              f"Range: {freq.min():.3f} – {freq.max():.3f} Hz")
    else:
        print("  [grid_features] Frequency column not found — Priority 1 features skipped.")

    # ══════════════════════════════════════════════════════════════════════
    # PRIORITY 2 — Hydro MW separated
    # Hydro is the most flexible large-scale generation in India.
    # It is dispatched at marginal cost ~zero when reservoirs are full,
    # making it a strong seasonal price depressant.
    # Separating it from total_generation exposes this directly.
    # ══════════════════════════════════════════════════════════════════════

    hydro = _col(df,
                 'hydro_mw',
                 'all_india_hydro_mw',
                 'total_hydro_mw',
                 'hydro_generation_mw')

    has_hydro = any(c in df.columns for c in
                    ['hydro_mw', 'all_india_hydro_mw',
                     'total_hydro_mw', 'hydro_generation_mw'])

    if has_hydro:
        feats['grid_hydro_mw'] = hydro

        feats['grid_hydro_share'] = (
            hydro / feats['grid_total_gen_mw']
        ).replace([np.inf, -np.inf], 0).fillna(0)

        feats['grid_hydro_lag_24h'] = hydro.shift(24 * BPH)

        feats['grid_hydro_ramp_1h'] = hydro - hydro.shift(1 * BPH)

        print(f"  [grid_features] Hydro features built. "
              f"Range: {hydro.min():.0f} – {hydro.max():.0f} MW")
    else:
        print("  [grid_features] Hydro column not found — Priority 2 features skipped.")

    # ══════════════════════════════════════════════════════════════════════
    # PRIORITY 3 — Gas MW (marginal plant proxy)
    # Gas plants are the most expensive dispatchable thermal in India
    # (₹6–10/kWh variable cost vs ₹2–3/kWh for coal).
    # When gas plants are running heavily → system at the expensive end
    # of the merit order → prices elevated → high DAM MCP likely.
    # ══════════════════════════════════════════════════════════════════════

    gas = _col(df,
               'gas_mw',
               'all_india_gas_mw',
               'gas_generation_mw',
               'gas_naptha_mw')

    has_gas = any(c in df.columns for c in
                  ['gas_mw', 'all_india_gas_mw',
                   'gas_generation_mw', 'gas_naptha_mw'])

    if has_gas:
        feats['grid_gas_mw']      = gas
        feats['grid_gas_lag_24h'] = gas.shift(24 * BPH)
        feats['grid_gas_share']   = (
            gas / feats['grid_total_gen_mw']
        ).replace([np.inf, -np.inf], 0).fillna(0)

        print(f"  [grid_features] Gas features built. "
              f"Range: {gas.min():.0f} – {gas.max():.0f} MW")
    else:
        print("  [grid_features] Gas column not found — Priority 3 features skipped.")

    # ══════════════════════════════════════════════════════════════════════
    # PRIORITY 4 — Net Transnational Exchange (MW)
    # India imports from Bhutan (cheap hydro) and occasionally Nepal.
    # India exports to Bangladesh and sometimes Nepal.
    # Convention: +ve = import (helps supply → lower prices)
    #             -ve = export (reduces supply → higher prices)
    # The NLDC PSP TimeSeries column K gives this at 15-min resolution.
    # ══════════════════════════════════════════════════════════════════════

    trans = _col(df,
                 'net_transnational_mw',
                 'net_transnational_exchange_mw',
                 'net_international_exchange_mw',
                 'transnational_exchange_mw',
                 'cross_border_mw')

    has_trans = any(c in df.columns for c in
                    ['net_transnational_mw', 'net_transnational_exchange_mw',
                     'net_international_exchange_mw', 'transnational_exchange_mw',
                     'cross_border_mw'])

    if has_trans:
        feats['grid_net_transnational_mw'] = trans

        # Binary flag: is India a net importer right now?
        feats['grid_is_net_importer'] = (trans > 0).astype(int)

        feats['grid_transnational_lag_24h'] = trans.shift(24 * BPH)

        print(f"  [grid_features] Transnational exchange features built. "
              f"Range: {trans.min():.0f} – {trans.max():.0f} MW")
    else:
        print("  [grid_features] Transnational column not found — Priority 4 features skipped.")

    # ══════════════════════════════════════════════════════════════════════
    # PRIORITY 5 — CWC Reservoir Storage % (optional weekly parquet)
    # Reservoir storage drives medium-term hydro availability.
    # Low storage in Oct-Dec → hydro constrained → thermal must run →
    # higher marginal cost → higher DAM prices.
    # High storage in Aug-Sep (post-monsoon) → abundant cheap hydro →
    # lower prices.
    #
    # reservoir_df expected schema:
    #   date             : str 'YYYY-MM-DD' (weekly observation date)
    #   reservoir_pct    : float, 0-100 (% of live storage capacity filled)
    #
    # The weekly value is forward-filled to 15-min blocks.
    # ══════════════════════════════════════════════════════════════════════

    if reservoir_df is not None:
        try:
            res = reservoir_df.copy()

            # Normalise date column
            if 'date' not in res.columns:
                possible_date_cols = ['Date', 'DATE', 'report_date', 'week_date']
                for c in possible_date_cols:
                    if c in res.columns:
                        res = res.rename(columns={c: 'date'})
                        break

            # Normalise reservoir_pct column
            if 'reservoir_pct' not in res.columns:
                possible_pct_cols = [
                    'storage_pct', 'storage_percent', 'pct_full',
                    'reservoir_storage_pct', 'live_storage_pct'
                ]
                for c in possible_pct_cols:
                    if c in res.columns:
                        res = res.rename(columns={c: 'reservoir_pct'})
                        break

            res['date'] = pd.to_datetime(res['date'])
            res = res.sort_values('date').drop_duplicates('date')

            # Build a daily reindex then forward-fill to 15-min blocks
            # by merging on date portion of the 15-min index
            idx_dates = feats.index.normalize()   # 15-min → date (midnight)

            # Reindex reservoir to all dates in the feature window
            all_dates = pd.date_range(
                start=idx_dates.min(),
                end=idx_dates.max(),
                freq='D',
                tz=feats.index.tz
            )
            res_daily = (
                res.set_index('date')['reservoir_pct']
                .reindex(all_dates)
                .ffill()    # forward-fill: weekly observation holds until next
                .bfill()    # back-fill the very start if needed
            )

            # Map each 15-min block to its date's reservoir value
            feats['grid_reservoir_storage_pct'] = (
                idx_dates.map(res_daily).values
            )

            # Deficit flag: below 40% historically correlates with hydro constraints
            feats['grid_reservoir_deficit'] = (
                feats['grid_reservoir_storage_pct'] < 40.0
            ).astype(int)

            # 4-week (28-day) rolling mean — seasonal storage trend
            feats['grid_reservoir_rolling_4w'] = (
                feats['grid_reservoir_storage_pct']
                .rolling(window=28 * 24 * BPH, min_periods=7 * 24 * BPH)
                .mean()
            )

            valid_pct = feats['grid_reservoir_storage_pct'].dropna()
            print(f"  [grid_features] Reservoir features built. "
                  f"Range: {valid_pct.min():.1f}% – {valid_pct.max():.1f}%")

        except Exception as e:
            print(f"  [grid_features] WARNING: Reservoir merge failed: {e}. "
                  f"Priority 5 features skipped.")
    else:
        print("  [grid_features] reservoir_df=None — Priority 5 features skipped.")
        print("                  Pass reservoir_df to build_grid_features() to enable.")

    return feats
