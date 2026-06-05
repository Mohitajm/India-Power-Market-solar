"""
src/features/weather_features.py — Phase 2 improvements
=========================================================
Phase 2 additions vs Phase 1:
  1. wx_temp_lag_1h          — temperature 1h ago (short-term AC ramp signal)
  2. wx_temp_rolling_7d      — 7-day rolling mean temperature (heatwave regime)
  3. wx_shortwave_lag_24h    — solar irradiance same hour yesterday
  4. wx_cdh_rolling_7d       — 7-day rolling mean CDH (sustained heat load)
  5. wx_wind_ramp_1h         — wind speed change vs 1h ago (intermittency signal)

All original features preserved identically.
wx_temp_spread MUST remain — pipeline.py references it in rtm_passthrough_cols.
"""

import pandas as pd
import numpy as np

_DEFAULT_BPH = 4


def _detect_bph(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return _DEFAULT_BPH
    try:
        idx   = pd.DatetimeIndex(df.index)
        delta = (idx[1] - idx[0]).total_seconds()
        if delta <= 900:
            return 4
        elif delta <= 1800:
            return 2
        else:
            return 1
    except Exception:
        return _DEFAULT_BPH


def build_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    if 'delivery_start_ist' in weather_df.columns:
        df = weather_df.set_index('delivery_start_ist').sort_index()
    else:
        df = weather_df.sort_index()

    bph  = _detect_bph(df)
    feats = pd.DataFrame(index=df.index)

    # ── Original features ──────────────────────────────────────────────────
    feats['wx_national_temp']      = df.get('national_temp',      np.nan)
    feats['wx_delhi_temp']         = df.get('delhi_temp',         np.nan)
    feats['wx_national_shortwave'] = df.get('national_shortwave', np.nan)
    feats['wx_chennai_wind']       = df.get('chennai_wind',       np.nan)
    feats['wx_national_cloud']     = df.get('national_cloud',     np.nan)

    feats['wx_cooling_degree_hours'] = np.maximum(
        0, feats['wx_national_temp'] - 24
    )
    feats['wx_heat_index'] = (
        feats['wx_national_temp']
        * (df.get('national_humidity', pd.Series(70, index=df.index)) / 100)
    )
    feats['wx_temp_lag_24h'] = feats['wx_national_temp'].shift(24 * bph)
    feats['wx_shortwave_delta_1h'] = (
        feats['wx_national_shortwave']
        - feats['wx_national_shortwave'].shift(1 * bph)
    )
    # MUST keep this name — pipeline.py references it
    feats['wx_temp_spread'] = feats['wx_delhi_temp'] - feats['wx_national_temp']

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2 NEW — Extended weather features
    # ══════════════════════════════════════════════════════════════════════

    # 1. Temperature 1h ago — short-term AC load ramp signal
    feats['wx_temp_lag_1h'] = feats['wx_national_temp'].shift(1 * bph)

    # 2. 7-day rolling mean temperature — heatwave / cold-spell regime
    feats['wx_temp_rolling_7d'] = (
        feats['wx_national_temp']
        .rolling(window=168 * bph, min_periods=24 * bph)
        .mean()
    )

    # 3. Solar irradiance same hour yesterday — most stable solar predictor
    feats['wx_shortwave_lag_24h'] = feats['wx_national_shortwave'].shift(24 * bph)

    # 4. 7-day rolling mean CDH — sustained heat load (structural demand signal)
    feats['wx_cdh_rolling_7d'] = (
        feats['wx_cooling_degree_hours']
        .rolling(window=168 * bph, min_periods=24 * bph)
        .mean()
    )

    # 5. Wind speed ramp vs 1h ago — intermittency signal for RE generation
    feats['wx_wind_ramp_1h'] = (
        feats['wx_chennai_wind']
        - feats['wx_chennai_wind'].shift(1 * bph)
    )

    return feats
