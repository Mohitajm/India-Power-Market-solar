"""
scripts/export_backtest_csvs_rtc.py — Architecture v10 RTC FINAL (Reverse DC)
===============================================================================
Reads daily JSON files produced by run_phase3b_backtest_rtc.py and exports:
  1. results/phase3b_rtc/phase3b_rtc_all_blocks.csv  — 96 rows × N days (~88 cols)
  2. results/phase3b_rtc/phase3b_rtc_summary.csv     — 1 row per day (P&L summary)

The per-day CSV files (results/phase3b_rtc/csv/phase3b_rtc_YYYY-MM-DD.csv) are
already written by the runner during backtest. This script re-reads the daily
JSONs to regenerate a clean all-blocks CSV with the full column set, which is
useful after architecture changes or config updates without re-running the backtest.

Reverse DC System — changes vs old export_backtest_csvs.py:
  - s_c_da / s_c_rt / s_c_actual columns REMOVED (no s_c in Reverse DC)
  - C_RTC_da column ADDED  (Stage 1 captive delivery, = RTC_committed ∀t)
  - C_RTC_rt column ADDED  (Stage 2A/2B captive delivery plan)
  - rtc_notice_block ADDED (which block issued each consumer notice)
  - THRESHOLD_mw ADDED     (= rtc_floor_pct × rtc_mw, explicit contract value)
  - dc_con_mw ADDED        (DC-DC converter capacity, replaces solar_inverter_mw)
  - dsm_net_cash column ADDED (raw DSM settlement cash before RTC penalty)
  - block_iex_net column ADDED (separate IEX P&L per block)

Usage:
    python scripts/export_backtest_csvs_rtc.py
    python scripts/export_backtest_csvs_rtc.py --results results/phase3b_rtc
    python scripts/export_backtest_csvs_rtc.py --results results/phase3b_rtc --verbose
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T_BLOCKS          = 96
DT                = 0.25
RESCHEDULE_BLOCKS = {34, 42, 50, 58}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _g(arr, i, default=0.0):
    """Safe getter from list/array by index."""
    if arr is None or i >= len(arr):
        return default
    v = arr[i]
    return float(v) if v is not None else default


def _gs(arr, i, default=""):
    """Safe string getter."""
    if arr is None or i >= len(arr):
        return default
    v = arr[i]
    return str(v) if v is not None else default


def _block_time(b: int) -> str:
    minutes = b * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _load_jsons(results_dir: Path) -> list:
    """Load all daily JSON files from results/daily/."""
    daily_dir = results_dir / "daily"
    files = sorted(daily_dir.glob("phase3b_rtc_*.json"))
    if not files:
        # Fallback: try old naming convention
        files = sorted(daily_dir.glob("result_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {daily_dir}\n"
            f"Run: python scripts/run_phase3b_backtest_rtc.py first."
        )
    print(f"  Found {len(files)} daily JSON files in {daily_dir}")
    records = []
    for f in files:
        try:
            records.append(json.load(open(f, encoding="utf-8")))
        except Exception as e:
            print(f"  WARNING: Could not load {f.name}: {e}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# PER-BLOCK ROW BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_rows(rec: dict, verbose: bool = False) -> list:
    """
    Convert one daily JSON record into a list of 96 per-block row dicts.

    JSON keys read (all produced by run_phase3b_backtest_rtc.py):
      Scalars:   date, rtc_committed_mw, rtc_ceiling_mw, rtc_min_mw,
                 THRESHOLD_mw, dc_con_mw, p_max_mw, r_ppa_rs_mwh,
                 avail_cap_mwh, net_revenue, captive_net, iex_net,
                 rtc_penalty, degradation, bess_total_value
      Arrays(96): C_RTC_da, C_RTC_rt, captive_actual, soc_path(97),
                  rtc_penalty_by_block, rtc_notice_block,
                  x_c, x_d, s_cd_da, c_d_da, schedule_da, setpoint_da,
                  s_cd_rt, c_d_rt, y_c, y_d, schedule_rt, setpoint_rt,
                  captive_committed, s_cd_actual, c_d_actual,
                  actual_dam_prices, actual_rtm_prices, solar_da, solar_at,
                  solar_band_mask, rtc_notice_issued, rtc_notice_target,
                  block_captive_net, block_iex_net, block_degradation,
                  block_net, no_bess_dsm, no_bess_rtc_penalty,
                  no_bess_revenue, bess_total_value_by_block,
                  dsm_results (list of dicts, one per block)
    """
    date          = rec.get("date", "")
    rtc_c         = float(rec.get("rtc_committed_mw", 0.0))
    rtc_ceil      = float(rec.get("rtc_ceiling_mw",   5.0))
    rtc_min       = float(rec.get("rtc_min_mw",       1.0))
    THRESHOLD     = float(rec.get("THRESHOLD_mw",     rtc_c * 0.80))
    dc_con        = float(rec.get("dc_con_mw",        16.4))
    p_max         = float(rec.get("p_max_mw",         16.4))
    r_ppa         = float(rec.get("r_ppa_rs_mwh",     5000.0))
    avail_cap     = float(rec.get("avail_cap_mwh",    4.1))
    rtc_lo        = rtc_c * 0.95
    rtc_hi        = rtc_c * 1.05

    # 96-element arrays
    C_RTC_da      = rec.get("C_RTC_da",            [rtc_c] * T_BLOCKS)
    C_RTC_rt      = rec.get("C_RTC_rt",            [rtc_c] * T_BLOCKS)
    cap_act       = rec.get("captive_actual",       [0.0]  * T_BLOCKS)
    soc_path      = rec.get("soc_path",             [0.0]  * (T_BLOCKS + 1))
    pen_by_blk    = rec.get("rtc_penalty_by_block", [0.0]  * T_BLOCKS)
    notice_blk    = rec.get("rtc_notice_block",     [-1]   * T_BLOCKS)
    notice_issued = rec.get("rtc_notice_issued",    [False] * T_BLOCKS)
    notice_target = rec.get("rtc_notice_target",    [-1]   * T_BLOCKS)

    x_c           = rec.get("x_c",            [0.0] * T_BLOCKS)
    x_d           = rec.get("x_d",            [0.0] * T_BLOCKS)
    s_cd_da       = rec.get("s_cd_da",         [0.0] * T_BLOCKS)
    c_d_da        = rec.get("c_d_da",          [0.0] * T_BLOCKS)
    sch_da        = rec.get("schedule_da",     [0.0] * T_BLOCKS)
    spt_da        = rec.get("setpoint_da",     [0.0] * T_BLOCKS)
    solar_band    = rec.get("solar_band_mask", [False] * T_BLOCKS)

    s_cd_rt       = rec.get("s_cd_rt",         [0.0] * T_BLOCKS)
    c_d_rt        = rec.get("c_d_rt",          [0.0] * T_BLOCKS)
    y_c           = rec.get("y_c",             [0.0] * T_BLOCKS)
    y_d           = rec.get("y_d",             [0.0] * T_BLOCKS)
    sch_rt        = rec.get("schedule_rt",     [0.0] * T_BLOCKS)
    spt_rt        = rec.get("setpoint_rt",     [0.0] * T_BLOCKS)
    cap_comm      = rec.get("captive_committed",[rtc_c] * T_BLOCKS)

    s_cd_act      = rec.get("s_cd_actual",     [0.0] * T_BLOCKS)
    c_d_act       = rec.get("c_d_actual",      [0.0] * T_BLOCKS)

    dam_p         = rec.get("actual_dam_prices",[0.0] * T_BLOCKS)
    rtm_p         = rec.get("actual_rtm_prices",[0.0] * T_BLOCKS)
    sol_da        = rec.get("solar_da",         [0.0] * T_BLOCKS)
    sol_at        = rec.get("solar_at",         [0.0] * T_BLOCKS)

    bl_capnet     = rec.get("block_captive_net",  [0.0] * T_BLOCKS)
    bl_iex        = rec.get("block_iex_net",      [0.0] * T_BLOCKS)
    bl_deg        = rec.get("block_degradation",  [0.0] * T_BLOCKS)
    bl_net        = rec.get("block_net",          [0.0] * T_BLOCKS)
    nb_dsm        = rec.get("no_bess_dsm",        [0.0] * T_BLOCKS)
    nb_pen        = rec.get("no_bess_rtc_penalty",[0.0] * T_BLOCKS)
    nb_rev        = rec.get("no_bess_revenue",    [0.0] * T_BLOCKS)
    bess_val_blk  = rec.get("bess_total_value_by_block", [0.0] * T_BLOCKS)

    dsm_res       = rec.get("dsm_results", [{}] * T_BLOCKS)

    # Cumulative accumulators
    c_iex = c_cap = c_dsm_pen = c_dsm_hc = c_rtc_pen = 0.0
    c_deg = c_net = c_bess = c_no_bess = c_short_mwh = c_disch = 0.0

    rows = []
    for B in range(T_BLOCKS):
        dsm = dsm_res[B] if B < len(dsm_res) else {}

        cap_a  = _g(cap_act, B)
        xc_B   = _g(x_c, B);  xd_B = _g(x_d, B)
        yc_B   = _g(y_c, B);  yd_B = _g(y_d, B)
        cd_a   = _g(c_d_act, B)
        tot_d  = xd_B + cd_a + yd_B

        short_mw  = max(0.0, THRESHOLD - cap_a)
        short_mwh = short_mw * DT
        rtc_pen_B = _g(pen_by_blk, B)
        iex_B     = _g(bl_iex, B)
        cap_net_B = _g(bl_capnet, B)
        deg_B     = _g(bl_deg, B)
        net_B     = _g(bl_net, B)

        # dsm_net_cash = captive net before RTC penalty deduction
        dsm_net_cash = cap_net_B + rtc_pen_B   # reverse: capnet = dsm - rtcpen

        c_iex       += iex_B
        c_cap       += cap_net_B
        c_dsm_pen   += float(dsm.get("dsm_penalty", 0.0))
        c_dsm_hc    += float(dsm.get("dsm_haircut", 0.0))
        c_rtc_pen   += rtc_pen_B
        c_deg       += deg_B
        c_net       += net_B
        bv_B         = float(bess_val_blk[B]) if B < len(bess_val_blk) else (
                        _g(nb_dsm,B)+_g(nb_pen,B)
                        - float(dsm.get("dsm_penalty",0))-float(dsm.get("dsm_haircut",0))
                        - rtc_pen_B + iex_B - deg_B)
        c_bess      += bv_B
        c_no_bess   += _g(nb_rev, B)
        c_short_mwh += short_mwh
        c_disch     += tot_d * DT / 0.9487
        cycles       = c_disch / 72.0   # USABLE = 72 MWh

        # Dispatch case
        if _g(s_cd_act, B) > 0 and cd_a < 1e-4:
            dcase = "A"   # solar surplus — BESS charging
        elif cd_a > 1e-4:
            dcase = "B"   # deficit — BESS discharging to captive
        else:
            dcase = "C"   # exact match

        row = {
            # ── Identifiers (7) ────────────────────────────────────────────
            "date":                   date,
            "block":                  B,
            "block_time_ist":         _block_time(B),
            "is_reschedule_block":    B in RESCHEDULE_BLOCKS,
            "rtc_notice_issued":      bool(notice_issued[B]) if B < len(notice_issued) else False,
            "rtc_notice_target_block": int(_g(notice_target, B, -1)),
            "rtc_notice_block":       int(notice_blk[B]) if B < len(notice_blk) else -1,

            # ── Parameters (6) ─────────────────────────────────────────────
            "p_max_mw":               p_max,
            "dc_con_mw":              dc_con,
            "r_ppa_rs_mwh":           r_ppa,
            "avail_cap_mwh":          avail_cap,
            "THRESHOLD_mw":           THRESHOLD,
            "architecture":           rec.get("architecture", "v10_rtc_reverse_dc"),

            # ── RTC Contract (6) ───────────────────────────────────────────
            "rtc_committed_mw":       rtc_c,
            "rtc_ceiling_mw":         rtc_ceil,
            "rtc_floor_pct":          THRESHOLD / rtc_ceil if rtc_ceil > 0 else 0.80,
            "rtc_band_lo":            rtc_lo,
            "rtc_band_hi":            rtc_hi,
            "rtc_delivery_ok":        cap_a >= THRESHOLD,

            # ── Solar (4) ──────────────────────────────────────────────────
            "z_sol_da_mw":            _g(sol_da, B),
            "z_sol_at_mw":            _g(sol_at, B),
            "sol_forecast_error_mw":  _g(sol_at, B) - _g(sol_da, B),
            "is_solar_band":          bool(solar_band[B]) if B < len(solar_band) else False,

            # ── Prices (4) ─────────────────────────────────────────────────
            "actual_dam_price_rs_mwh":  _g(dam_p, B),
            "actual_rtm_price_rs_mwh":  _g(rtm_p, B),

            # ── Stage 1 DA (8) — s_c_da REMOVED ───────────────────────────
            "x_c_da_mw":              _g(x_c, B),
            "x_d_da_mw":              _g(x_d, B),
            "dam_net_mw":             _g(x_d, B) - _g(x_c, B),
            "s_cd_da_mw":             _g(s_cd_da, B),
            "c_d_da_mw":              _g(c_d_da, B),
            "C_RTC_da_mw":            _g(C_RTC_da, B),
            "schedule_da_mw":         _g(sch_da, B),
            "setpoint_da_mw":         _g(spt_da, B),

            # ── Stage 2B / 2A RT (8) — s_c_rt REMOVED ─────────────────────
            "s_cd_rt_mw":             _g(s_cd_rt, B),
            "c_d_rt_mw":              _g(c_d_rt, B),
            "C_RTC_rt_mw":            _g(C_RTC_rt, B),
            "y_c_mw":                 yc_B,
            "y_d_mw":                 yd_B,
            "y_net_mw":               yd_B - yc_B,
            "schedule_rt_mw":         _g(sch_rt, B),
            "setpoint_rt_mw":         _g(spt_rt, B),
            "captive_committed_mw":   _g(cap_comm, B),

            # ── Actuals (5) — s_c_actual REMOVED ──────────────────────────
            "active_setpoint_mw":     _g(spt_rt, B),
            "s_cd_actual_mw":         _g(s_cd_act, B),
            "c_d_actual_mw":          cd_a,
            "captive_actual_mw":      cap_a,
            "dispatch_case":          dcase,

            # ── SoC (2) ────────────────────────────────────────────────────
            "soc_actual_start_mwh":   _g(soc_path, B),
            "soc_actual_end_mwh":     _g(soc_path, B + 1),

            # ── DSM (10) ───────────────────────────────────────────────────
            "dsm_net_cash_rs":        round(dsm_net_cash, 2),
            "contract_rate_rs_mwh":   float(dsm.get("charge_rate", r_ppa)),
            "actual_total_mw":        cap_a,
            "scheduled_total_mw":     _g(sch_rt, B),
            "deviation_mwh":          float(dsm.get("dws_mwh",   0.0)),
            "deviation_pct":          float(dsm.get("dws_pct",   0.0)),
            "deviation_band":         str(dsm.get("band",        "0-10%")),
            "deviation_direction":    str(dsm.get("direction",   "within")),
            "charge_rate_rs_mwh":     float(dsm.get("charge_rate", r_ppa)),
            "charge_rate_multiplier": float(dsm.get("charge_rate_mult", 1.0)),

            # ── Under-injection P&L (5) ────────────────────────────────────
            "under_revenue_received_rs":   float(dsm.get("under_revenue_received", 0.0)),
            "under_dsm_penalty_rs":        float(dsm.get("under_dsm_penalty",      0.0)),
            "under_net_cash_flow_rs":      float(dsm.get("under_net_cash",         0.0)),
            "under_if_fully_scheduled_rs": float(dsm.get("under_if_fully_sched",   0.0)),
            "under_financial_damage_rs":   float(dsm.get("under_damage",           0.0)),

            # ── Over-injection P&L (5) ─────────────────────────────────────
            "over_revenue_sched_qty_rs":   float(dsm.get("over_revenue_sched",   0.0)),
            "over_revenue_dev_qty_rs":     float(dsm.get("over_revenue_dev",     0.0)),
            "over_total_received_rs":      float(dsm.get("over_total_received",  0.0)),
            "over_if_all_at_cr_rs":        float(dsm.get("over_if_all_cr",       0.0)),
            "over_revenue_haircut_rs":     float(dsm.get("over_haircut",         0.0)),

            # ── RTC Penalty (4) ────────────────────────────────────────────
            "rtc_shortfall_mw":       short_mw,
            "rtc_shortfall_mwh":      short_mwh,
            "rtc_penalty_rs":         rtc_pen_B,
            "rtc_delivery_ok":        cap_a >= THRESHOLD,

            # ── IEX (4) ────────────────────────────────────────────────────
            "iex_dam_revenue_rs":     _g(dam_p,B)*(xd_B-xc_B)*DT,
            "iex_rtm_revenue_rs":     _g(rtm_p,B)*(yd_B-yc_B)*DT,
            "iex_fees_rs":            200.0*(xc_B+xd_B+yc_B+yd_B)*DT,
            "iex_net_rs":             iex_B,

            # ── Block P&L (4) ──────────────────────────────────────────────
            "block_captive_net_rs":   cap_net_B,
            "block_iex_net_rs":       iex_B,
            "block_degradation_rs":   deg_B,
            "block_net_rs":           net_B,

            # ── BESS ROI (5) ───────────────────────────────────────────────
            "no_bess_dsm_rs":         _g(nb_dsm, B),
            "no_bess_rtc_penalty_rs": _g(nb_pen, B),
            "no_bess_revenue_rs":     _g(nb_rev, B),
            "bess_dsm_savings_rs":    _g(nb_dsm,B) - float(dsm.get("dsm_penalty",0))
                                      - float(dsm.get("dsm_haircut",0)),
            "bess_total_value_rs":    bv_B,

            # ── Cumulative (11) ────────────────────────────────────────────
            "cum_bess_cycles":        round(cycles,     4),
            "cum_iex_net_rs":         round(c_iex,      2),
            "cum_captive_net_rs":     round(c_cap,      2),
            "cum_dsm_penalty_rs":     round(c_dsm_pen,  2),
            "cum_dsm_haircut_rs":     round(c_dsm_hc,   2),
            "cum_rtc_penalty_rs":     round(c_rtc_pen,  2),
            "cum_degradation_rs":     round(c_deg,      2),
            "cum_net_revenue_rs":     round(c_net,      2),
            "cum_bess_value_rs":      round(c_bess,     2),
            "cum_no_bess_revenue_rs": round(c_no_bess,  2),
            "cum_rtc_shortfall_mwh":  round(c_short_mwh, 4),
        }
        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_summary(records: list) -> pd.DataFrame:
    """One row per day — daily P&L summary."""
    rows = []
    for rec in records:
        rows.append({
            "date":                 rec.get("date",             ""),
            "architecture":         rec.get("architecture",     "v10_rtc_reverse_dc"),
            "rtc_committed_mw":     rec.get("rtc_committed_mw", 0.0),
            "rtc_ceiling_mw":       rec.get("rtc_ceiling_mw",   5.0),
            "THRESHOLD_mw":         rec.get("THRESHOLD_mw",     4.0),
            "dc_con_mw":            rec.get("dc_con_mw",        16.4),
            "soc_initial_mwh":      rec.get("soc_initial_mwh",  40.0),
            "soc_final_mwh":        rec.get("soc_final_mwh",    0.0),
            "expected_revenue_rs":  rec.get("expected_revenue", 0.0),
            "net_revenue_rs":       rec.get("net_revenue",      0.0),
            "captive_net_rs":       rec.get("captive_net",      0.0),
            "iex_net_rs":           rec.get("iex_net",          0.0),
            "rtc_penalty_rs":       rec.get("rtc_penalty",      0.0),
            "degradation_rs":       rec.get("degradation",      0.0),
            "no_bess_revenue_rs":   rec.get("no_bess_revenue",  0.0),
            "bess_dsm_savings_rs":  rec.get("bess_dsm_savings", 0.0),
            "bess_rtc_savings_rs":  rec.get("bess_rtc_pen_savings",
                                    rec.get("bess_rtc_savings", 0.0)),
            "bess_total_value_rs":  rec.get("bess_total_value", 0.0),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        description="Export RTC backtest results to CSV — Reverse DC System")
    ap.add_argument("--results", type=str,
                    default="results/phase3b_rtc",
                    help="Results directory (default: results/phase3b_rtc)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-day progress")
    return ap.parse_args()


def main():
    args        = parse_args()
    results_dir = Path(args.results)

    print("=" * 65)
    print("EXPORT BACKTEST CSVs — RTC Reverse DC System v10 FINAL")
    print("=" * 65)
    print(f"Results dir: {results_dir}")

    # ── Load JSONs ────────────────────────────────────────────────────────────
    records = _load_jsons(results_dir)
    print(f"  Dates: {sorted(r.get('date','?') for r in records)}")

    # ── Build all-blocks CSV ──────────────────────────────────────────────────
    all_rows = []
    for rec in records:
        date = rec.get("date", "?")
        if args.verbose:
            print(f"  Processing {date} …")
        rows = _build_rows(rec, verbose=args.verbose)
        all_rows.extend(rows)

    all_df = pd.DataFrame(all_rows)

    # Column order — matches Architecture v10 RTC CSV schema (§19)
    col_order = [
        # Identifiers
        "date", "block", "block_time_ist", "is_reschedule_block",
        "rtc_notice_issued", "rtc_notice_target_block", "rtc_notice_block",
        # Parameters
        "p_max_mw", "dc_con_mw", "r_ppa_rs_mwh", "avail_cap_mwh",
        "THRESHOLD_mw", "architecture",
        # RTC Contract
        "rtc_committed_mw", "rtc_ceiling_mw", "rtc_floor_pct",
        "rtc_band_lo", "rtc_band_hi", "rtc_delivery_ok",
        # Solar
        "z_sol_da_mw", "z_sol_at_mw", "sol_forecast_error_mw", "is_solar_band",
        # Prices
        "actual_dam_price_rs_mwh", "actual_rtm_price_rs_mwh",
        # Stage 1 DA  (s_c_da_mw REMOVED)
        "x_c_da_mw", "x_d_da_mw", "dam_net_mw",
        "s_cd_da_mw", "c_d_da_mw", "C_RTC_da_mw",
        "schedule_da_mw", "setpoint_da_mw",
        # Stage 2B/2A RT  (s_c_rt_mw REMOVED)
        "s_cd_rt_mw", "c_d_rt_mw", "C_RTC_rt_mw",
        "y_c_mw", "y_d_mw", "y_net_mw",
        "schedule_rt_mw", "setpoint_rt_mw", "captive_committed_mw",
        # Actuals  (s_c_actual_mw REMOVED)
        "active_setpoint_mw", "s_cd_actual_mw", "c_d_actual_mw",
        "captive_actual_mw", "dispatch_case",
        # SoC
        "soc_actual_start_mwh", "soc_actual_end_mwh",
        # DSM
        "dsm_net_cash_rs",
        "contract_rate_rs_mwh", "actual_total_mw", "scheduled_total_mw",
        "deviation_mwh", "deviation_pct", "deviation_band",
        "deviation_direction", "charge_rate_rs_mwh", "charge_rate_multiplier",
        # Under-injection
        "under_revenue_received_rs", "under_dsm_penalty_rs",
        "under_net_cash_flow_rs", "under_if_fully_scheduled_rs",
        "under_financial_damage_rs",
        # Over-injection
        "over_revenue_sched_qty_rs", "over_revenue_dev_qty_rs",
        "over_total_received_rs", "over_if_all_at_cr_rs", "over_revenue_haircut_rs",
        # RTC Penalty
        "rtc_shortfall_mw", "rtc_shortfall_mwh", "rtc_penalty_rs",
        # IEX
        "iex_dam_revenue_rs", "iex_rtm_revenue_rs", "iex_fees_rs", "iex_net_rs",
        # Block P&L
        "block_captive_net_rs", "block_iex_net_rs",
        "block_degradation_rs", "block_net_rs",
        # BESS ROI
        "no_bess_dsm_rs", "no_bess_rtc_penalty_rs", "no_bess_revenue_rs",
        "bess_dsm_savings_rs", "bess_total_value_rs",
        # Cumulative
        "cum_bess_cycles", "cum_iex_net_rs", "cum_captive_net_rs",
        "cum_dsm_penalty_rs", "cum_dsm_haircut_rs", "cum_rtc_penalty_rs",
        "cum_degradation_rs", "cum_net_revenue_rs", "cum_bess_value_rs",
        "cum_no_bess_revenue_rs", "cum_rtc_shortfall_mwh",
    ]
    # Only keep columns that exist in the DataFrame
    col_order = [c for c in col_order if c in all_df.columns]
    # Append any extra columns not in the schema order
    extra = [c for c in all_df.columns if c not in col_order]
    all_df = all_df[col_order + extra]

    all_csv = results_dir / "phase3b_rtc_all_blocks.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"\n✅ All-blocks CSV:  {all_csv}")
    print(f"   Rows: {len(all_df)}  |  Cols: {len(all_df.columns)}  "
          f"|  Days: {all_df['date'].nunique()}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    sdf     = _build_summary(records)
    sum_csv = results_dir / "phase3b_rtc_summary.csv"
    sdf.to_csv(sum_csv, index=False)
    print(f"✅ Summary CSV:     {sum_csv}")
    print(f"   Days: {len(sdf)}")

    # ── Print P&L Summary ─────────────────────────────────────────────────────
    if len(sdf) > 0:
        print(f"\n{'─'*65}")
        print(f"{'BACKTEST P&L SUMMARY':^65}")
        print(f"{'─'*65}")
        print(f"  Days run:              {len(sdf)}")
        print(f"  Avg RTC committed:     {sdf['rtc_committed_mw'].mean():.2f} MW")
        print(f"  Total net revenue:     ₹{sdf['net_revenue_rs'].sum():>15,.0f}")
        print(f"  Total captive net:     ₹{sdf['captive_net_rs'].sum():>15,.0f}")
        print(f"  Total IEX net:         ₹{sdf['iex_net_rs'].sum():>15,.0f}")
        print(f"  Total RTC penalties:   ₹{sdf['rtc_penalty_rs'].sum():>15,.0f}")
        print(f"  Total degradation:     ₹{sdf['degradation_rs'].sum():>15,.0f}")
        print(f"  Total BESS value:      ₹{sdf['bess_total_value_rs'].sum():>15,.0f}")
        print(f"  Avg EOD SoC:           {sdf['soc_final_mwh'].mean():.1f} MWh")
        print(f"{'─'*65}")

        # Per-day table
        print(f"\n  {'Date':<12} {'RTC':>6} {'Net Rev':>12} "
              f"{'Cap Net':>12} {'IEX':>10} {'Penalty':>10} {'SoC End':>8}")
        print(f"  {'-'*12} {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
        for _, r in sdf.iterrows():
            print(f"  {r['date']:<12} {r['rtc_committed_mw']:>5.2f} "
                  f"₹{r['net_revenue_rs']:>10,.0f} "
                  f"₹{r['captive_net_rs']:>10,.0f} "
                  f"₹{r['iex_net_rs']:>8,.0f} "
                  f"₹{r['rtc_penalty_rs']:>8,.0f} "
                  f"{r['soc_final_mwh']:>7.1f}")


if __name__ == "__main__":
    main()
