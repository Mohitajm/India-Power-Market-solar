"""
src/optimizer/two_stage_bess_rtc.py — Architecture v10 RTC FINAL (Reverse DC System)
======================================================================================
Three-stage Solar+BESS optimizer — Reverse DC topology with RTC captive contract.

TOPOLOGY (Reverse DC):
  Solar SCBs → DC-DC Converter (dc_con MW) → DC Bus ← BESS (parallel)
  DC Bus → single PCS (p_max MW) → AC Bus → Grid Metering Point → Captive + IEX

ALL power flows through the single PCS:
  Import (δ=1): x_c (grid→BESS via PCS)
  Export (δ=0): s_cd (solar→captive), c_d (BESS→captive), x_d (BESS→IEX)
  ALL export flows share PCS: x_d + c_d + s_cd ≤ p_max × (1−δ)

CHANGES FROM PREVIOUS VERSION:
  FIX-1  Reverse DC topology — dc_con replaces S_inv in C3c, s_cd bounded by dc_con
  FIX-2  C2: x_d+c_d+s_cd ≤ p_max  (s_cd added — all three share single PCS)
  FIX-3  C3b: xd+yd+cd+s_cd ≤ p_max×(1−δ)  (s_cd added to export constraint)
  FIX-4  C3c: s_cd ≤ dc_con×(1−δ)  (dc_con replaces S_inv)
  FIX-5  C4: SoC dynamics = eta_c×x_c×DT − c_d/eta_d×DT  (s_cd removed — solar
         goes directly to PCS AC output, does not flow through BESS SoC)
  FIX-6  Discharge capacity = min(p_max, (soc−e_min)×eta_d/DT)  (s_c removed)
  FIX-7  THRESHOLD = p.THRESHOLD = rtc_floor_pct × rtc_mw  (replaces hardcoded 4.0)
  FIX-8  Setpoint: inner ±5% (RTC notice boundary) + outer ±10% (DSM free band)
  FIX-9  Outputs renamed: C_RTC_da (Stage 1), C_RTC_rt (Stage 2A/2B)
  FIX-10 rtc_notice_block[B]: records which block B issued each consumer notice
"""

import pulp
import numpy as np
from typing import Dict, List, Optional, Tuple

T_BLOCKS          = 96
DT                = 0.25
RESCHEDULE_BLOCKS = [34, 42, 50, 58]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_solar_band_mask_rtc(solar: np.ndarray,
                                threshold: float = 0.5,
                                buffer: int = 2) -> np.ndarray:
    """Boolean mask True during solar-generation hours (±buffer blocks)."""
    mask         = np.zeros(len(solar), dtype=bool)
    solar_blocks = [t for t in range(len(solar)) if solar[t] > threshold]
    if solar_blocks:
        s = max(0,            min(solar_blocks) - buffer)
        e = min(len(solar)-1, max(solar_blocks) + buffer)
        mask[s:e+1] = True
    return mask


def compute_setpoint_rtc(soc: float, schedule: float,
                          e_min: float, e_max: float,
                          eta_c: float, eta_d: float,
                          rtc_tol_pct: float = 0.05,
                          dsm_tol_pct: float = 0.10) -> float:
    """
    Setpoint — DSM-regulation aligned, dual-band clamp.

    Formula:
      raw = schedule × (0.90 + 0.20 × bias_ratio)

    Clamps:
      Inner band (RTC ±5%): [schedule×0.95, schedule×1.05]
        — EMS stays here silently, no consumer notice needed
      Outer band (DSM ±10%): [schedule×0.90, schedule×1.10]
        — penalty-free per CERC DSM 2024

    The setpoint is clipped to the OUTER DSM band.
    If the raw setpoint falls outside the INNER RTC band, the advance-notice
    protocol in Stage 2A/2B handles consumer notification separately.

    Parameters
    ----------
    schedule     : MW filed with SLDC (= C_RTC_da[t] + dam_net[t])
    rtc_tol_pct  : inner RTC free band (default 0.05 = ±5%)
    dsm_tol_pct  : outer DSM free band (default 0.10 = ±10%)
    """
    dr  = max(0.0, (soc - e_min) * eta_d)
    cr  = max(0.0, (e_max - soc) / eta_c)
    br  = dr / (dr + cr + 1e-9)
    raw = schedule * (0.90 + 0.20 * br)
    # Outer DSM ±10% clamp
    lo  = schedule * (1.0 - dsm_tol_pct)
    hi  = schedule * (1.0 + dsm_tol_pct)
    return float(np.clip(raw, lo, hi))


def compute_contract_rate(rtc_committed: float, x_d: float, y_d: float,
                          p_dam: float, p_rtm: float, r_ppa: float) -> float:
    """Blended CR across PPA + DAM sell + RTM sell."""
    ppa = max(0.0, rtc_committed)
    dam = max(0.0, x_d)
    rtm = max(0.0, y_d)
    tot = ppa + dam + rtm
    return (ppa*r_ppa + dam*p_dam + rtm*p_rtm) / tot if tot > 1e-9 else r_ppa


def compute_dsm_settlement(captive_actual: float, scheduled: float,
                           cr: float, avail_cap: float) -> dict:
    """CERC DSM 2024 three-band settlement for one 15-min block."""
    act_mwh = captive_actual * DT
    sch_mwh = scheduled * DT
    dws     = (captive_actual - scheduled) * DT
    pct     = abs(dws) / avail_cap * 100.0 if avail_cap > 0 else 0.0
    over    = dws > 0

    if pct <= 10.0:
        rate, mult, band = cr,        1.0,  "0-10%"
    elif pct <= 15.0:
        rate, mult, band = (0.90*cr, 0.90, "10-15%") if over \
                      else (1.10*cr, 1.10, "10-15%")
    else:
        rate, mult, band = (0.0,    0.0,  ">15%") if over \
                      else (1.50*cr, 1.50, ">15%")

    direction = "within" if pct <= 10 else ("over" if over else "under")
    r = {"dws_mwh": dws, "dws_pct": pct, "band": band,
         "direction": direction, "charge_rate": rate, "charge_rate_mult": mult,
         "net_captive_cash": 0.0, "dsm_penalty": 0.0, "dsm_haircut": 0.0,
         "financial_damage": 0.0,
         "under_revenue_received": 0.0, "under_dsm_penalty": 0.0,
         "under_net_cash": 0.0, "under_if_fully_sched": 0.0, "under_damage": 0.0,
         "over_revenue_sched": 0.0, "over_revenue_dev": 0.0,
         "over_total_received": 0.0, "over_if_all_cr": 0.0, "over_haircut": 0.0}
    if pct <= 10.0:
        r["net_captive_cash"] = act_mwh * cr
    elif dws < 0:
        rev = act_mwh*cr; pen = abs(dws)*rate; net = rev-pen; ifs = sch_mwh*cr
        r.update({"under_revenue_received": rev, "under_dsm_penalty": pen,
                  "under_net_cash": net, "under_if_fully_sched": ifs,
                  "under_damage": ifs-net, "net_captive_cash": net,
                  "dsm_penalty": pen, "financial_damage": ifs-net})
    else:
        rs = sch_mwh*cr; rd = dws*rate; tr = rs+rd; ia = act_mwh*cr; hc = max(0.0,ia-tr)
        r.update({"over_revenue_sched": rs, "over_revenue_dev": rd,
                  "over_total_received": tr, "over_if_all_cr": ia,
                  "over_haircut": hc, "net_captive_cash": tr, "dsm_haircut": hc})
    return r


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1
# ══════════════════════════════════════════════════════════════════════════════

class TwoStageBESSRTC:
    """
    Stage 1 MILP — D-1 10:00 IST.

    Reverse DC System constraints:
      C2:  x_d[t] + c_d[t] + s_cd[t] ≤ p_max   (all three share single PCS)
      C3a: x_c[t]  ≤ p_max × delta[t]            (grid import — δ=1)
      C3b: x_d[t] + c_d[t] + s_cd[t] ≤ p_max × (1−delta[t])  (all export)
      C3c: s_cd[t] ≤ dc_con × (1−delta[t])        (DC converter limit)
      C4:  soc dynamics driven by x_c and c_d only
           (s_cd goes to PCS AC output directly, bypasses BESS SoC)

    Outputs: C_RTC_da[t] = s_cd_da[t] + c_d_da[t] = RTC_committed ∀t
    """

    def __init__(self, params, config: Dict):
        self.p           = params
        self.lambda_risk = config.get("lambda_risk", 0.0)
        self.risk_alpha  = config.get("risk_alpha",  0.10)

    def solve(self, dam_scenarios: np.ndarray,
              rtm_scenarios: np.ndarray,
              solar_da:      np.ndarray) -> Dict:
        p        = self.p
        S        = dam_scenarios.shape[0]
        p_max    = p.p_max_mw
        dc_con   = p.dc_con_mw               # DC converter capacity (FIX-1)
        r_ppa    = p.ppa_rate_rs_mwh
        USABLE   = p.usable_energy_mwh       # = 72 MWh
        THRESHOLD = p.THRESHOLD              # = rtc_floor_pct × rtc_mw = 4.0 MW (FIX-7)

        solar_da   = np.clip(solar_da, 0.0, dc_con)
        solar_mask = compute_solar_band_mask_rtc(
            solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

        prob  = pulp.LpProblem("Stage1_RTC", pulp.LpMaximize)

        # Decision variables
        RTC_c   = pulp.LpVariable("RTC_c", lowBound=p.rtc_min_mw, upBound=p.rtc_mw)
        x_c     = pulp.LpVariable.dicts("xc",  range(T_BLOCKS), 0, p_max)
        x_d     = pulp.LpVariable.dicts("xd",  range(T_BLOCKS), 0, p_max)
        s_c     = pulp.LpVariable.dicts("sc",  range(T_BLOCKS), 0, p_max)   # solar→BESS via DC Bus (bypasses PCS)
        s_cd    = pulp.LpVariable.dicts("scd", range(T_BLOCKS), 0, dc_con)  # solar→captive via DC Bus→PCS
        c_d     = pulp.LpVariable.dicts("cd",  range(T_BLOCKS), 0, p_max)
        p_short = pulp.LpVariable.dicts("psh", range(T_BLOCKS), 0)
        delta   = pulp.LpVariable.dicts("dlt", range(T_BLOCKS), cat="Binary")

        soc  = {si: pulp.LpVariable.dicts(f"soc{si}", range(T_BLOCKS+1),
                                           p.e_min_mwh, p.e_max_mwh)
                for si in range(S)}
        zeta = pulp.LpVariable("zeta")
        u    = pulp.LpVariable.dicts("u", range(S), lowBound=0)

        # First-stage constraints (shared across scenarios)
        for t in range(T_BLOCKS):
            sol_t = float(solar_da[t])

            # C1: solar balance (no curtailment)
            # s_c = solar→BESS via DC Bus (charges BESS directly, no PCS needed)
            # s_cd = solar→captive via DC Bus→PCS→AC
            # All solar is consumed: s_c + s_cd = solar_da
            prob += s_c[t] + s_cd[t] == sol_t,                          f"C1_{t}"

            # C_RTC: flat constant delivery (C_RTC_da)
            prob += s_cd[t] + c_d[t] == RTC_c,                          f"CRTC_{t}"

            # C_PSHORT: THRESHOLD (not hardcoded 4.0) — FIX-7
            prob += p_short[t] >= THRESHOLD - (s_cd[t] + c_d[t]),       f"CPSH_{t}"

            # C2: PCS total output limit — all three export flows share PCS (FIX-2)
            prob += x_d[t] + c_d[t] + s_cd[t] <= p_max,                f"C2_{t}"

            # C3: AC bus mutual exclusion (Reverse DC)
            # C3a: grid import (δ=1)
            prob += x_c[t] <= p_max * delta[t],                         f"C3a_{t}"
            # C3b: ALL export flows share PCS (δ=0) — FIX-3: s_cd added
            prob += x_d[t] + c_d[t] + s_cd[t] <= p_max*(1-delta[t]),   f"C3b_{t}"
            # C3c: DC converter limit on solar — FIX-4: dc_con replaces S_inv
            prob += s_cd[t] <= dc_con * (1 - delta[t]),                 f"C3c_{t}"

        # Scenario loop
        USABLE = p.usable_energy_mwh
        scen_revs = []
        for si in range(S):
            prob += soc[si][0] == p.soc_initial_mwh,                    f"SOD_{si}"
            if p.soc_terminal_mode == "hard":
                prob += soc[si][T_BLOCKS] == p.soc_terminal_min_mwh,    f"EOD_{si}"
            else:
                prob += soc[si][T_BLOCKS] >= p.soc_terminal_min_mwh,    f"EOD_{si}"

            # C7: max 1 cycle/day
            prob += pulp.lpSum(
                [(x_d[t] + c_d[t]) * DT / p.eta_discharge
                 for t in range(T_BLOCKS)]
            ) <= USABLE,                                                  f"C7_{si}"

            rev = 0
            for t in range(T_BLOCKS):
                pd_t = float(dam_scenarios[si, t])

                # C4: SoC dynamics
                # s_c (solar→BESS via DC Bus) charges BESS — no PCS needed, no delta restriction
                # x_c (grid→BESS via PCS) charges BESS — requires delta=1
                # c_d (BESS→captive via PCS) discharges BESS
                # s_cd (solar→captive via PCS) does NOT affect SoC — bypasses BESS
                prob += soc[si][t+1] == (
                    soc[si][t]
                    + p.eta_charge    * (s_c[t] + x_c[t]) * DT
                    - (1.0/p.eta_discharge) * c_d[t] * DT
                ),                                                        f"C4_{si}_{t}"

                # C6: SoC solar band
                if solar_mask[t]:
                    prob += soc[si][t] >= p.soc_solar_low,              f"C6lo_{si}_{t}"
                    prob += soc[si][t] <= p.soc_solar_high,             f"C6hi_{si}_{t}"

                # Objective terms
                rev += pd_t  * x_d[t]              * DT   # DAM sell
                rev -= pd_t  * x_c[t]              * DT   # DAM buy
                rev += r_ppa * (s_cd[t] + c_d[t])  * DT   # PPA (C_RTC_da)
                rev -= p.iex_fee_rs_mwh*(x_c[t]+x_d[t])* DT  # IEX fees
                rev -= r_ppa * p_short[t]           * DT   # THRESHOLD penalty

            prob += u[si] >= zeta - rev
            scen_revs.append(rev)

        avg_rev = pulp.lpSum(scen_revs) / S
        cvar    = zeta - (1.0/(S*self.risk_alpha)) * pulp.lpSum(u.values())
        prob.setObjective(avg_rev + self.lambda_risk * cvar)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        z96 = [0.0] * T_BLOCKS
        if pulp.LpStatus[prob.status] != "Optimal":
            return {"status": "Infeasible", "RTC_committed": p.rtc_min_mw,
                    "x_c": z96, "x_d": z96, "s_cd_da": z96, "c_d_da": z96,
                    "C_RTC_da": z96, "dam_net": z96,
                    "schedule_da": z96, "setpoint_da": z96,
                    "solar_band_mask": [False]*T_BLOCKS, "expected_revenue": 0.0,
                    "scenarios": []}

        rtc_val  = float(pulp.value(RTC_c) or p.rtc_min_mw)
        xc_v     = [max(0.0, pulp.value(x_c[t])  or 0.0) for t in range(T_BLOCKS)]
        xd_v     = [max(0.0, pulp.value(x_d[t])  or 0.0) for t in range(T_BLOCKS)]
        sc_v     = [max(0.0, pulp.value(s_c[t])  or 0.0) for t in range(T_BLOCKS)]
        scd_v    = [max(0.0, pulp.value(s_cd[t]) or 0.0) for t in range(T_BLOCKS)]
        cd_v     = [max(0.0, pulp.value(c_d[t])  or 0.0) for t in range(T_BLOCKS)]

        # C_RTC_da[t] = s_cd_da[t] + c_d_da[t] = RTC_committed ∀t (FIX-9)
        c_rtc_da = [scd_v[t] + cd_v[t] for t in range(T_BLOCKS)]
        dam_net  = [xd_v[t]  - xc_v[t]  for t in range(T_BLOCKS)]
        sched_da = [rtc_val  + dam_net[t] for t in range(T_BLOCKS)]

        soc_mean = [float(np.mean([pulp.value(soc[si][t]) or 0.0 for si in range(S)]))
                    for t in range(T_BLOCKS+1)]

        # Setpoint: DSM-aligned dual-band clamp (FIX-8)
        sp_da = [compute_setpoint_rtc(soc_mean[t], sched_da[t],
                                      p.e_min_mwh, p.e_max_mwh,
                                      p.eta_charge, p.eta_discharge,
                                      p.rtc_tol_pct, p.dsm_tol_pct)
                 for t in range(T_BLOCKS)]

        return {
            "status":           "Optimal",
            "expected_revenue": float(pulp.value(avg_rev) or 0.0),
            "RTC_committed":    rtc_val,
            "x_c": xc_v, "x_d": xd_v,
            "s_c_da": sc_v, "s_cd_da": scd_v, "c_d_da": cd_v,
            "C_RTC_da":   c_rtc_da,        # FIX-9: explicit label
            "captive_da": c_rtc_da,         # backward compat alias
            "dam_net": dam_net,
            "schedule_da": sched_da, "setpoint_da": sp_da,
            "solar_band_mask": solar_mask.tolist(),
            "scenarios": [{"soc": [pulp.value(soc[si][t])
                                   for t in range(T_BLOCKS+1)]}
                          for si in range(S)],
        }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2B
# ══════════════════════════════════════════════════════════════════════════════

def reschedule_captive_rtc(params,
                            trigger_block:          int,
                            soc_actual:             float,
                            solar_nc_row:           np.ndarray,
                            solar_da:               np.ndarray,
                            rtm_q50:                np.ndarray,
                            x_c_s1:                 np.ndarray,
                            x_d_s1:                 np.ndarray,
                            y_c_committed:          np.ndarray,
                            y_d_committed:          np.ndarray,
                            rtc_committed:          float,
                            captive_committed_prev: np.ndarray,
                            rtc_notice:             np.ndarray,
                            rtc_notice_block:       np.ndarray,   # FIX-10
                            cycle_used_so_far:      float = 0.0) -> Dict:
    """
    Stage 2B: NC nowcast reschedule.
    Same Reverse DC constraints as Stage 1.
    Updates C_RTC_rt[t] and rtc_notice_block[B].
    """
    p         = params
    B         = trigger_block
    remaining = T_BLOCKS - B
    p_max     = p.p_max_mw
    dc_con    = p.dc_con_mw
    r_ppa     = p.ppa_rate_rs_mwh
    THRESHOLD = p.THRESHOLD
    RTM_LEAD  = p.rtm_lead_blocks

    solar_blend = np.zeros(remaining, dtype=float)
    for k in range(remaining):
        t_abs = B + k
        solar_blend[k] = (float(solar_nc_row[k]) if k < len(solar_nc_row)
                          else float(solar_da[t_abs]) if t_abs < T_BLOCKS else 0.0)
    solar_blend = np.clip(solar_blend, 0.0, dc_con)
    solar_mask  = compute_solar_band_mask_rtc(solar_da, p.solar_threshold_mw,
                                               p.solar_buffer_blocks)

    xc_r  = np.array(x_c_s1[B:],       dtype=float)
    xd_r  = np.array(x_d_s1[B:],       dtype=float)
    yc_r  = np.array(y_c_committed[B:], dtype=float)
    yd_r  = np.array(y_d_committed[B:], dtype=float)
    rtm_r = np.array(rtm_q50[B:],       dtype=float)

    prob  = pulp.LpProblem(f"S2B_b{B}", pulp.LpMaximize)
    scd   = pulp.LpVariable.dicts("scd", range(remaining), 0, dc_con)
    cd    = pulp.LpVariable.dicts("cd",  range(remaining), 0, p_max)
    psh   = pulp.LpVariable.dicts("psh", range(remaining), 0)
    soc_v = pulp.LpVariable.dicts("soc", range(remaining+1), p.e_min_mwh, p.e_max_mwh)
    dl    = pulp.LpVariable.dicts("dl",  range(remaining), cat="Binary")

    cap_rt = {}
    for k in range(remaining):
        t_abs = B + k
        lo, hi = (p.rtc_min_mw, p.rtc_mw) if rtc_notice[t_abs] \
                 else p.rtc_band(rtc_committed)
        cap_rt[k] = pulp.LpVariable(f"crt_{k}", lowBound=lo, upBound=hi)

    prob += soc_v[0] == float(np.clip(soc_actual, p.e_min_mwh, p.e_max_mwh))
    if p.soc_terminal_mode == "hard":
        prob += soc_v[remaining] == p.soc_terminal_min_mwh
    else:
        prob += soc_v[remaining] >= p.soc_terminal_min_mwh

    USABLE = p.usable_energy_mwh
    cycle_budget = max(0.0, USABLE - cycle_used_so_far)
    prob += pulp.lpSum(
        [(cd[k] + float(xd_r[k]) + (float(yd_r[k]) if k < RTM_LEAD else 0.0))
         * DT / p.eta_discharge for k in range(remaining)]
    ) <= cycle_budget, "C7_2b"

    rtc_notice_out       = rtc_notice.copy()
    rtc_notice_block_out = rtc_notice_block.copy()
    rev = 0

    for k in range(remaining):
        t_abs = B + k
        xc_k  = float(xc_r[k])
        xd_k  = float(xd_r[k])
        yc_k  = float(yc_r[k]) if k < RTM_LEAD else 0.0
        yd_k  = float(yd_r[k]) if k < RTM_LEAD else 0.0
        pr_k  = float(rtm_r[k])
        sol_k = float(solar_blend[k])

        # C1
        prob += scd[k] <= sol_k,                                      f"C1_{k}"
        # C_RTC → C_RTC_rt
        prob += scd[k] + cd[k] == cap_rt[k],                         f"CRTC_{k}"
        # C_PSHORT with THRESHOLD
        prob += psh[k] >= THRESHOLD - (scd[k] + cd[k]),              f"PSH_{k}"
        # C2: ALL export flows (FIX-2)
        prob += xd_k + yd_k + cd[k] + scd[k] <= p_max,              f"C2_{k}"
        # C3a
        prob += xc_k + yc_k <= p_max * dl[k],                       f"C3a_{k}"
        # C3b: s_cd included (FIX-3)
        prob += xd_k + yd_k + cd[k] + scd[k] <= p_max*(1-dl[k]),   f"C3b_{k}"
        # C3c: dc_con (FIX-4)
        prob += scd[k] <= dc_con * (1-dl[k]),                        f"C3c_{k}"
        # Buffer
        if k < p.captive_buffer_blocks:
            ct = float(captive_committed_prev[t_abs])
            prob += cap_rt[k] >= ct - p.captive_buffer_tolerance_mw, f"CBlo_{k}"
            prob += cap_rt[k] <= ct + p.captive_buffer_tolerance_mw, f"CBhi_{k}"
        # C4: SoC dynamics
        # s_c (solar→BESS via DC Bus) = sol_k - scd[k] (surplus after captive delivery)
        # scd[k] is LP variable so express inline: eta_c*(sol_k - scd[k]) = eta_c*sol_k - eta_c*scd[k]
        prob += soc_v[k+1] == (
            soc_v[k]
            + p.eta_charge * sol_k * DT
            - p.eta_charge * scd[k] * DT
            + p.eta_charge * (xc_k + yc_k) * DT
            - (1.0/p.eta_discharge) * (cd[k] + xd_k + yd_k) * DT
        ),                                                            f"C4_{k}"
        # C6
        if solar_mask[t_abs]:
            prob += soc_v[k] >= p.soc_solar_low,                    f"C6lo_{k}"
            prob += soc_v[k] <= p.soc_solar_high,                   f"C6hi_{k}"

        rev += r_ppa * (scd[k] + cd[k]) * DT
        rev += pr_k  * xd_k * DT - pr_k * xc_k * DT
        rev -= p.iex_fee_rs_mwh * (xc_k + xd_k) * DT
        rev -= r_ppa * psh[k] * DT

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    scd_out = np.zeros(T_BLOCKS)
    cd_out  = np.zeros(T_BLOCKS)
    cap_out = np.full(T_BLOCKS, rtc_committed, dtype=float)

    if status == "Optimal":
        for k in range(remaining):
            t_abs = B + k
            scd_out[t_abs] = max(0.0, pulp.value(scd[k]) or 0.0)
            cd_out[t_abs]  = max(0.0, pulp.value(cd[k])  or 0.0)
            cap_val = float(pulp.value(cap_rt[k]) or rtc_committed)
            cap_out[t_abs] = cap_val
            dev = abs(cap_val - rtc_committed) / (rtc_committed + 1e-9)
            if dev > p.rtc_tol_pct:
                t_notice = t_abs + p.rtc_advance_blocks
                if t_notice < T_BLOCKS:
                    rtc_notice_out[t_notice] = True
                    rtc_notice_block_out[B] = t_notice   # FIX-10

    return {"status": status,
            "s_cd_rt": scd_out, "c_d_rt": cd_out,
            "C_RTC_rt": cap_out,              # FIX-9
            "captive_rt": cap_out,            # backward compat alias
            "rtc_notice": rtc_notice_out,
            "rtc_notice_block": rtc_notice_block_out}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2A
# ══════════════════════════════════════════════════════════════════════════════

def solve_stage2a_rtc(params,
                       block_B:           int,
                       soc_actual_B:      float,
                       dam_actual:        np.ndarray,
                       rtm_q50:           np.ndarray,
                       p_rtm_lag4:        float,
                       s_cd_rt:           np.ndarray,
                       c_d_rt:            np.ndarray,
                       y_c_committed:     np.ndarray,
                       y_d_committed:     np.ndarray,
                       x_c_s1:            np.ndarray,
                       x_d_s1:            np.ndarray,
                       solar_da:          np.ndarray,
                       rtc_committed:     float,
                       captive_committed: np.ndarray,
                       rtc_notice:        np.ndarray,
                       rtc_notice_block:  np.ndarray,   # FIX-10
                       cycle_used_so_far: float = 0.0) -> Tuple[float, float,
                                                                 np.ndarray,
                                                                 np.ndarray]:
    """
    Stage 2A: receding-horizon MPC.
    Bids y_c/y_d for B+RTM_LEAD. Updates C_RTC_rt and rtc_notice_block.
    Same Reverse DC constraints as Stage 1.
    """
    p         = params
    p_max     = p.p_max_mw
    dc_con    = p.dc_con_mw
    r_ppa     = p.ppa_rate_rs_mwh
    THRESHOLD = p.THRESHOLD
    RTM_LEAD  = p.rtm_lead_blocks
    bid_block = block_B + RTM_LEAD
    rtc_notice_out       = rtc_notice.copy()
    rtc_notice_block_out = rtc_notice_block.copy()

    if bid_block >= T_BLOCKS:
        return 0.0, 0.0, rtc_notice_out, rtc_notice_block_out

    # Lag-4 RTM price bias
    rtm_adj = rtm_q50.copy().astype(float)
    if not np.isnan(p_rtm_lag4) and block_B >= 4:
        bias = p_rtm_lag4 - float(rtm_q50[block_B-4])
        for t in range(bid_block, T_BLOCKS):
            rtm_adj[t] = max(0.0, rtm_adj[t] + bias * (0.85 ** (t - block_B)))

    solar_mask = compute_solar_band_mask_rtc(solar_da, p.solar_threshold_mw,
                                              p.solar_buffer_blocks)

    # Roll SoC forward B → bid_block (x_c and c_d only — FIX-5)
    soc_rf = float(np.clip(soc_actual_B, p.e_min_mwh, p.e_max_mwh))
    for t in range(block_B, bid_block):
        xc_t = float(x_c_s1[t]); yc_t = float(y_c_committed[t])
        cd_t = float(c_d_rt[t]); xd_t = float(x_d_s1[t])
        yd_t = float(y_d_committed[t])
        ch   = p.eta_charge * (xc_t + yc_t) * DT
        di   = (cd_t + xd_t + yd_t) / p.eta_discharge * DT
        soc_rf = float(np.clip(soc_rf + ch - di, p.e_min_mwh, p.e_max_mwh))

    remaining = T_BLOCKS - bid_block
    if remaining <= 0:
        return 0.0, 0.0, rtc_notice_out, rtc_notice_block_out

    prob   = pulp.LpProblem(f"S2A_b{block_B}", pulp.LpMaximize)
    y_c    = pulp.LpVariable.dicts("yc",  range(remaining), 0, p_max)
    y_d    = pulp.LpVariable.dicts("yd",  range(remaining), 0, p_max)
    psh    = pulp.LpVariable.dicts("psh", range(remaining), 0)
    soc_lp = pulp.LpVariable.dicts("soc", range(remaining+1), p.e_min_mwh, p.e_max_mwh)
    dl     = pulp.LpVariable.dicts("dl",  range(remaining), cat="Binary")

    cap_rt_lp = {}
    for k in range(remaining):
        t_abs = bid_block + k
        lo, hi = (p.rtc_min_mw, p.rtc_mw) if (t_abs < T_BLOCKS and rtc_notice[t_abs]) \
                 else p.rtc_band(rtc_committed)
        cap_rt_lp[k] = pulp.LpVariable(f"crt_{k}", lowBound=lo, upBound=hi)

    prob += soc_lp[0] == soc_rf
    if p.soc_terminal_mode == "hard":
        prob += soc_lp[remaining] == p.soc_terminal_min_mwh
    else:
        prob += soc_lp[remaining] >= p.soc_terminal_min_mwh

    USABLE = p.usable_energy_mwh
    cycle_budget = max(0.0, USABLE - cycle_used_so_far)
    prob += pulp.lpSum(
        [(float(c_d_rt[bid_block+k]) + float(x_d_s1[bid_block+k]) + y_d[k])
         * DT / p.eta_discharge for k in range(remaining)]
    ) <= cycle_budget, "C7_2a"

    rev = 0
    for k in range(remaining):
        ta   = bid_block + k
        xc_t = float(x_c_s1[ta]); xd_t = float(x_d_s1[ta])
        scd_t = float(s_cd_rt[ta]); cd_t = float(c_d_rt[ta])
        pr   = float(rtm_adj[ta])

        # C_RTC_rt
        prob += scd_t + cd_t == cap_rt_lp[k],                        f"CRTC_{k}"
        # C_PSHORT with THRESHOLD (FIX-7)
        prob += psh[k] >= THRESHOLD - (scd_t + cd_t),                f"PSH_{k}"
        # C2: ALL export flows (FIX-2)
        prob += xd_t + y_d[k] + cd_t + scd_t <= p_max,              f"C2_{k}"
        # C3a
        prob += xc_t + y_c[k] <= p_max * dl[k],                     f"C3a_{k}"
        # C3b: s_cd_t included (FIX-3)
        prob += xd_t + y_d[k] + cd_t + scd_t <= p_max*(1-dl[k]),   f"C3b_{k}"
        # C3c: dc_con (FIX-4)
        prob += scd_t <= dc_con * (1-dl[k]),                         f"C3c_{k}"
        # C4: SoC dynamics
        # s_c (solar→BESS via DC Bus) = solar_da[ta] - scd_t  (surplus after captive delivery)
        # scd_t is the DA-planned captive delivery (fixed in Stage 2A planning horizon)
        s_c_ta = max(0.0, float(solar_da[ta]) - scd_t)
        prob += soc_lp[k+1] == (
            soc_lp[k]
            + p.eta_charge    * (s_c_ta + xc_t + y_c[k]) * DT
            - (1.0/p.eta_discharge) * (cd_t + xd_t + y_d[k]) * DT
        ),                                                            f"C4_{k}"
        if solar_mask[ta]:
            prob += soc_lp[k] >= p.soc_solar_low,                   f"C6lo_{k}"
            prob += soc_lp[k] <= p.soc_solar_high,                  f"C6hi_{k}"

        rev += pr * y_d[k] * DT - pr * y_c[k] * DT
        rev += r_ppa * cap_rt_lp[k] * DT
        rev -= p.iex_fee_rs_mwh * (y_c[k] + y_d[k]) * DT
        rev -= r_ppa * psh[k] * DT

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        y_c_bid = max(0.0, pulp.value(y_c[0]) or 0.0)
        y_d_bid = max(0.0, pulp.value(y_d[0]) or 0.0)
        for k in range(remaining):
            cap_val = float(pulp.value(cap_rt_lp[k]) or rtc_committed)
            dev = abs(cap_val - rtc_committed) / (rtc_committed + 1e-9)
            if dev > p.rtc_tol_pct:
                t_notice = bid_block + k + p.rtc_advance_blocks
                if t_notice < T_BLOCKS:
                    rtc_notice_out[t_notice] = True
                    rtc_notice_block_out[block_B] = t_notice  # FIX-10
        return y_c_bid, y_d_bid, rtc_notice_out, rtc_notice_block_out

    return 0.0, 0.0, rtc_notice_out, rtc_notice_block_out


# ══════════════════════════════════════════════════════════════════════════════
# ACTUALS SETTLEMENT
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_actuals_rtc(params,
                          stage1_result:    Dict,
                          dam_actual:       np.ndarray,
                          rtm_actual:       np.ndarray,
                          rtm_q50:          np.ndarray,
                          solar_da:         np.ndarray,
                          solar_nc:         np.ndarray,
                          solar_at:         np.ndarray,
                          reschedule_blocks: List[int] = RESCHEDULE_BLOCKS,
                          verbose:          bool = False) -> Dict:
    """
    Block-by-block settlement — Reverse DC System.

    Dispatch capacity (FIX-6):
      disch_cap = min(p_max, (soc - e_min) * eta_d / DT)  [s_c removed]

    SoC update (FIX-5):
      charge_e    = eta_c * (x_c + y_c) * DT
      discharge_e = c_d_actual / eta_d * DT
      (s_cd does not affect SoC — it goes directly to PCS AC output)

    Outputs:
      C_RTC_da[t]:      Stage 1 captive delivery plan
      C_RTC_rt[t]:      Stage 2A/2B captive delivery plan
      rtc_notice_block: which block issued each consumer notice (FIX-10)
    """
    p             = params
    r_ppa         = p.ppa_rate_rs_mwh
    RTM_LEAD      = p.rtm_lead_blocks
    avail_cap     = p.avail_cap_mwh
    THRESHOLD     = p.THRESHOLD              # = rtc_floor_pct × rtc_mw
    rtc_committed = float(stage1_result["RTC_committed"])

    x_c_s1   = np.array(stage1_result["x_c"],    dtype=float)
    x_d_s1   = np.array(stage1_result["x_d"],    dtype=float)
    s_c_rt   = np.array(stage1_result.get("s_c_da", np.zeros(T_BLOCKS)), dtype=float)
    s_cd_rt  = np.array(stage1_result["s_cd_da"], dtype=float)
    c_d_rt   = np.array(stage1_result["c_d_da"],  dtype=float)

    y_c_committed     = np.zeros(T_BLOCKS)
    y_d_committed     = np.zeros(T_BLOCKS)
    captive_committed = np.full(T_BLOCKS, rtc_committed, dtype=float)
    rtc_notice        = np.zeros(T_BLOCKS, dtype=bool)
    rtc_notice_block  = np.full(T_BLOCKS, -1, dtype=int)   # FIX-10
    soc_path          = np.zeros(T_BLOCKS + 1)
    soc_path[0]       = p.soc_initial_mwh

    scd_act   = np.zeros(T_BLOCKS); cd_act    = np.zeros(T_BLOCKS)
    sc_act    = np.zeros(T_BLOCKS); cap_act   = np.zeros(T_BLOCKS)
    setpt     = np.zeros(T_BLOCKS); sch_rt    = np.zeros(T_BLOCKS)
    bl_capnet = np.zeros(T_BLOCKS); bl_cappen = np.zeros(T_BLOCKS)
    bl_iex    = np.zeros(T_BLOCKS); bl_deg    = np.zeros(T_BLOCKS)
    bl_net    = np.zeros(T_BLOCKS)
    nb_dsm    = np.zeros(T_BLOCKS); nb_rev    = np.zeros(T_BLOCKS)
    nb_pen    = np.zeros(T_BLOCKS)
    dsm_res   = []
    rtc_notice_issued = np.zeros(T_BLOCKS, dtype=bool)
    rtc_notice_target = np.full(T_BLOCKS, -1, dtype=int)
    cum_disch = 0.0

    for B in range(T_BLOCKS):
        lag4 = float(rtm_actual[B-4]) if B >= 4 else np.nan

        # ── Stage 2B ───────────────────────────────────────────────────────
        if B in reschedule_blocks:
            nc_row = solar_nc[B] if B < len(solar_nc) else np.zeros(12)
            r2b = reschedule_captive_rtc(
                params=p, trigger_block=B, soc_actual=soc_path[B],
                solar_nc_row=nc_row, solar_da=solar_da, rtm_q50=rtm_q50,
                x_c_s1=x_c_s1, x_d_s1=x_d_s1,
                y_c_committed=y_c_committed, y_d_committed=y_d_committed,
                rtc_committed=rtc_committed,
                captive_committed_prev=captive_committed.copy(),
                rtc_notice=rtc_notice,
                rtc_notice_block=rtc_notice_block,   # FIX-10
                cycle_used_so_far=cum_disch,
            )
            if r2b["status"] == "Optimal":
                s_cd_rt[B:]           = r2b["s_cd_rt"][B:]
                c_d_rt[B:]            = r2b["c_d_rt"][B:]
                old_notice            = rtc_notice.copy()
                rtc_notice            = r2b["rtc_notice"]
                rtc_notice_block      = r2b["rtc_notice_block"]  # FIX-10
                captive_committed[B:] = r2b["C_RTC_rt"][B:]
                for tb in range(B, T_BLOCKS):
                    if rtc_notice[tb] and not old_notice[tb]:
                        rtc_notice_issued[B] = True
                        rtc_notice_target[B] = tb

        # ── Stage 2A ───────────────────────────────────────────────────────
        bid_b = B + RTM_LEAD
        if bid_b < T_BLOCKS:
            yc_b, yd_b, rtc_notice, rtc_notice_block = solve_stage2a_rtc(
                params=p, block_B=B, soc_actual_B=soc_path[B],
                dam_actual=dam_actual, rtm_q50=rtm_q50, p_rtm_lag4=lag4,
                s_cd_rt=s_cd_rt, c_d_rt=c_d_rt,
                y_c_committed=y_c_committed, y_d_committed=y_d_committed,
                x_c_s1=x_c_s1, x_d_s1=x_d_s1, solar_da=solar_da,
                rtc_committed=rtc_committed, captive_committed=captive_committed,
                rtc_notice=rtc_notice,
                rtc_notice_block=rtc_notice_block,   # FIX-10
                cycle_used_so_far=cum_disch,
            )
            y_c_committed[bid_b] = yc_b
            y_d_committed[bid_b] = yd_b

        # ── Step 1: Inputs ─────────────────────────────────────────────────
        xc_B = float(x_c_s1[B]); xd_B = float(x_d_s1[B])
        yc_B = float(y_c_committed[B]); yd_B = float(y_d_committed[B])
        C_RTC_rt_B = float(s_cd_rt[B] + c_d_rt[B])
        sch_rt_B   = C_RTC_rt_B + (xd_B - xc_B) + (yd_B - yc_B)
        sch_rt[B]  = sch_rt_B
        z_at       = float(solar_at[B])

        # Setpoint: DSM dual-band (FIX-8)
        sp_B = compute_setpoint_rtc(soc_path[B], sch_rt_B,
                                    p.e_min_mwh, p.e_max_mwh,
                                    p.eta_charge, p.eta_discharge,
                                    p.rtc_tol_pct, p.dsm_tol_pct)
        setpt[B] = sp_B

        # Discharge capacity (FIX-6): BESS only, s_c removed
        disch_cap = max(0.0, min(
            p.p_max_mw - xd_B - yd_B,
            (soc_path[B] - p.e_min_mwh) * p.eta_discharge / DT))
        charg_cap = max(0.0, min(
            p.p_max_mw - xc_B - yc_B,
            (p.e_max_mwh - soc_path[B]) / (p.eta_charge * DT)))

        # ── Step 2: Dispatch (Case A/B/C) ─────────────────────────────────
        if z_at > sp_B + 1e-6:          # Case A: surplus solar
            sc_a  = min(charg_cap, z_at - sp_B)
            scd_a = z_at - sc_a
            cd_a  = 0.0
        elif z_at < sp_B - 1e-6:        # Case B: deficit → BESS
            scd_a = z_at
            cd_a  = min(disch_cap, sp_B - z_at)
            sc_a  = 0.0
        else:                            # Case C: match
            scd_a = z_at; sc_a = 0.0; cd_a = 0.0

        cap_a = scd_a + cd_a
        sc_act[B]  = sc_a; scd_act[B] = scd_a
        cd_act[B]  = cd_a; cap_act[B] = cap_a

        # ── Step 3: DSM ────────────────────────────────────────────────────
        cr  = compute_contract_rate(rtc_committed, xd_B, yd_B,
                                    float(dam_actual[B]), float(rtm_actual[B]), r_ppa)
        dsm = compute_dsm_settlement(cap_a, sch_rt_B, cr, avail_cap)
        dsm_res.append(dsm)

        # ── Step 4: RTC shortfall — THRESHOLD (FIX-7) ─────────────────────
        short_mw  = max(0.0, THRESHOLD - cap_a)
        short_mwh = short_mw * DT
        rtc_pen   = short_mwh * r_ppa
        bl_cappen[B] = rtc_pen
        bl_capnet[B] = dsm["net_captive_cash"] - rtc_pen

        # ── Step 5: IEX ────────────────────────────────────────────────────
        iex_dam  = float(dam_actual[B]) * (xd_B - xc_B) * DT
        iex_rtm  = float(rtm_actual[B]) * (yd_B - yc_B) * DT
        iex_fees = p.iex_fee_rs_mwh * (xc_B + xd_B + yc_B + yd_B) * DT
        bl_iex[B] = iex_dam + iex_rtm - iex_fees

        # ── Step 6: SoC update ────────────────────────────────────────────
        # sc_a  = solar→BESS via DC Bus (Case A surplus; charges BESS directly)
        # xc_B, yc_B = grid→BESS via PCS (require delta=1)
        # scd_a = solar→captive via PCS — does NOT affect SoC
        tot_d = xd_B + cd_a + yd_B
        ch_e  = p.eta_charge * (sc_a + xc_B + yc_B) * DT
        di_e  = tot_d / p.eta_discharge * DT
        soc_path[B+1] = float(np.clip(
            soc_path[B] + ch_e - di_e, p.e_min_mwh, p.e_max_mwh))
        cum_disch += tot_d * DT / p.eta_discharge

        # ── Step 7: Block P&L ──────────────────────────────────────────────
        deg       = p.degradation_cost_rs_mwh * tot_d * DT
        bl_deg[B] = deg
        bl_net[B] = bl_capnet[B] + bl_iex[B] - deg

        # ── Step 8: No-BESS counterfactual ────────────────────────────────
        nb_d  = compute_dsm_settlement(z_at, float(captive_committed[B]), r_ppa, avail_cap)
        nb_pen[B] = max(0.0, THRESHOLD - z_at) * DT * r_ppa
        nb_dsm[B] = nb_d["dsm_penalty"] + nb_d["dsm_haircut"]
        nb_rev[B] = z_at * DT * r_ppa - nb_dsm[B] - nb_pen[B]

        if verbose and B % 16 == 0:
            print(f"  B={B:02d} soc={soc_path[B]:.1f}→{soc_path[B+1]:.1f} "
                  f"sol={z_at:.2f} cap={cap_a:.2f} "
                  f"THRESHOLD={THRESHOLD:.1f} pen=₹{rtc_pen:,.0f} "
                  f"iex=₹{bl_iex[B]:,.0f} net=₹{bl_net[B]:,.0f}")

    with_dsm     = sum(d["dsm_penalty"] + d["dsm_haircut"] for d in dsm_res)
    bess_dsm_sav = float(np.sum(nb_dsm)) - with_dsm
    bess_pen_sav = float(np.sum(nb_pen)) - float(np.sum(bl_cappen))

    return {
        "net_revenue":           float(np.sum(bl_net)),
        "captive_net_total":     float(np.sum(bl_capnet)),
        "iex_net_total":         float(np.sum(bl_iex)),
        "rtc_penalty_total":     float(np.sum(bl_cappen)),
        "degradation_total":     float(np.sum(bl_deg)),
        "no_bess_revenue_total": float(np.sum(nb_rev)),
        "bess_dsm_savings":      bess_dsm_sav,
        "bess_rtc_pen_savings":  bess_pen_sav,
        "bess_total_value":      bess_dsm_sav + bess_pen_sav
                                 + float(np.sum(bl_iex))
                                 - float(np.sum(bl_deg)),
        "soc_path":              soc_path,
        "s_c_actual":            sc_act,     "s_cd_actual": scd_act,
        "c_d_actual":            cd_act,     "captive_actual": cap_act,
        "C_RTC_da":              stage1_result.get("C_RTC_da", stage1_result.get("captive_da")),
        "C_RTC_rt":              captive_committed,     # FIX-9
        "captive_committed":     captive_committed,     # backward compat
        "setpoint":              setpt,     "schedule_rt": sch_rt,
        "block_captive_net":     bl_capnet, "block_captive_penalty": bl_cappen,
        "block_iex_net":         bl_iex,    "block_degradation":     bl_deg,
        "block_net":             bl_net,
        "no_bess_dsm":           nb_dsm,    "no_bess_rtc_penalty":   nb_pen,
        "no_bess_revenue":       nb_rev,
        "dsm_results":           dsm_res,
        "rtc_notice_issued":     rtc_notice_issued,
        "rtc_notice_target":     rtc_notice_target,
        "rtc_notice_block":      rtc_notice_block,   # FIX-10
        "x_c": x_c_s1,  "x_d": x_d_s1,
        "y_c": y_c_committed, "y_d": y_d_committed,
        "s_c_rt": s_c_rt, "s_cd_rt": s_cd_rt, "c_d_rt": c_d_rt,
        "THRESHOLD": THRESHOLD,
    }
